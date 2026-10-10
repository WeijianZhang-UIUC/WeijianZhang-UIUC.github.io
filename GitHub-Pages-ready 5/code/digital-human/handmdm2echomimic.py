#!/usr/bin/env python3
"""
HandMDM + EchoMimicV2 端到端集成管线
=====================================

文本 → HandMDM (3D手势生成) → 空间投影 → DWPose格式 → EchoMimicV2 (视频渲染)

用法:
  # 方式 1: 从文本生成
  python handmdm2echomimic.py \
      --text "right hand waves hello" \
      --ref_img ./reference.png \
      --audio ./audio.wav \
      --duration 3.0 \
      --output ./output.mp4

  # 方式 2: 从预生成的 .npy 运动文件 (跳过 HandMDM)
  python handmdm2echomimic.py \
      --motion_file ./wave_6s_motion.npy \
      --ref_img ./reference.png \
      --audio ./audio.wav \
      --output ./output.mp4

两种模式:
  --mode simple : 启发式手部姿态提取 (默认, 无需额外依赖)
  --mode smplx  : SMPL-X正向运动学 (需 pip install smplx + SMPL-H模型)
"""

import os
import sys
import argparse
import logging
from tkinter import Image
from tkinter import Image
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
FPS_HANDMDM = 25       # HandMDM 原生帧率
FPS_ECHOMIMIC = 24     # EchoMimicV2 期望帧率
IMG_SIZE = 768         # EchoMimicV2 默认画布尺寸
HAND_KPTS_PER_HAND = 21  # DWPose 手部关键点数

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HandMDM2EchoMimic")


# ============================================================
# Phase 0: HandMDM 模型加载与推理
# ============================================================
def setup_handmdm_path():
    """将 HandMDM 加入 sys.path"""
    handmdm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HandMDM")
    if handmdm_dir not in sys.path:
        sys.path.insert(0, handmdm_dir)
    return handmdm_dir


def load_handmdm_model(checkpoint_path, device="cpu"):
    """加载 HandMDM 扩散模型 + 文本编码器"""
    setup_handmdm_path()
    from infer import load_model
    diffusion, text_encoder = load_model(checkpoint_path, device=device)
    logger.info(f"HandMDM 模型已加载: {checkpoint_path}")
    return diffusion, text_encoder


def generate_handmdm_motion(diffusion, text_encoder, text, length,
                             guidance=15.0, seed=1234, device="cpu"):
    """用 HandMDM 从文本生成运动特征 (T, 274) at 25fps"""
    setup_handmdm_path()
    from infer import generate
    motion = generate(diffusion, text_encoder, text,
                      length=length, guidance=guidance, seed=seed, device=device)
    logger.info(f"HandMDM 生成运动: shape={motion.shape} (T, 274) @{FPS_HANDMDM}fps")
    return motion


# ============================================================
# Phase 1: 帧率转换 25fps → 24fps
# ============================================================
def convert_fps(motion_274, src_fps=FPS_HANDMDM, tgt_fps=FPS_ECHOMIMIC):
    """沿时间轴线性插值重采样"""
    T_src = motion_274.shape[0]
    duration = T_src / src_fps
    T_tgt = int(np.ceil(duration * tgt_fps))
    if T_src == T_tgt:
        logger.info(f"帧率已匹配: {T_src} 帧 @{src_fps}fps")
        return motion_274
    src_t = np.linspace(0, 1, T_src)
    dst_t = np.linspace(0, 1, T_tgt)
    result = np.zeros((T_tgt, motion_274.shape[1]), dtype=motion_274.dtype)
    for d in range(motion_274.shape[1]):
        result[:, d] = np.interp(dst_t, src_t, motion_274[:, d])
    logger.info(f"帧率转换: {T_src}帧@{src_fps}fps → {T_tgt}帧@{tgt_fps}fps")
    return result


# ============================================================
# Phase 2: 3D手部关节提取
# ============================================================

# ─── 6D旋转 → 旋转矩阵 (numpy版Gram-Schmidt) ───
def rotation_6d_to_matrix_np(d6):
    """numpy 版 6D旋转 → 3x3 旋转矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


# ─── 手部骨骼定义 ───
# DWPose 21关键点: 0=手腕, 1-4=拇指, 5-8=食指, 9-12=中指, 13-16=无名指, 17-20=小指
# MANO 15关节 (SMPL-H标准顺序):
#   [0,1,2]=食指(MCP,PIP,DIP), [3,4,5]=中指(MCP,PIP,DIP),
#   [6,7,8]=小指(MCP,PIP,DIP), [9,10,11]=无名指(MCP,PIP,DIP),
#   [12,13,14]=拇指(CMC,MCP,IP)

# DWPose 手指骨骼链 (从近端到远端, DWPose索引)
_FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],     # CMC, MCP, IP, TIP
    "index":  [5, 6, 7, 8],     # MCP, PIP, DIP, TIP
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# MANO 关节(0-based) → DWPose 关键点
# 食指: MANO 0,1,2 → DWPose 5,6,7
# 中指: MANO 3,4,5 → DWPose 9,10,11
# 小指: MANO 6,7,8 → DWPose 17,18,19
# 无名指: MANO 9,10,11 → DWPose 13,14,15
# 拇指: MANO 12,13,14 → DWPose 1,2,3
_MANO_TO_DWPOSE = {
    0: 5, 1: 6, 2: 7,      # 食指
    3: 9, 4: 10, 5: 11,    # 中指
    6: 17, 7: 18, 8: 19,   # 小指
    9: 13, 10: 14, 11: 15, # 无名指
    12: 1, 13: 2, 14: 3,   # 拇指
}

# 每根手指的 MANO 关节索引 (0-based, 对应上面_MANO_TO_DWPOSE的key)
_MANO_FINGER_IDX = {
    "thumb":  [12, 13, 14],
    "index":  [0, 1, 2],
    "middle": [3, 4, 5],
    "ring":   [9, 10, 11],
    "pinky":  [6, 7, 8],
}

# DWPose 指尖外推 (指尖 → (直接父关节, 祖父关节))
_DWPOSE_TIP_EXTRAP = {
    4: (3, 2),    # 拇指尖 ← IP(3), MCP(2)
    8: (7, 6),    # 食指尖 ← DIP(7), PIP(6)
    12: (11, 10), # 中指尖
    16: (15, 14), # 无名指尖
    20: (19, 18), # 小指尖
}

# MANO 关节亲子关系: child_idx → parent_idx (-1 = 手腕)
_MANO_PARENT = {
    0: -1,   # index MCP → wrist
    1: 0,    # index PIP → MCP
    2: 1,    # index DIP → PIP
    3: -1,   # middle MCP → wrist
    4: 3,    # middle PIP → MCP
    5: 4,    # middle DIP → PIP
    6: -1,   # pinky MCP → wrist
    7: 6,    # pinky PIP → MCP
    8: 7,    # pinky DIP → PIP
    9: -1,   # ring MCP → wrist
    10: 9,   # ring PIP → MCP
    11: 10,  # ring DIP → PIP
    12: -1,  # thumb CMC → wrist
    13: 12,  # thumb MCP → CMC
    14: 13,  # thumb IP → MCP
}

# SMPL-H 52关节中: neck=12, l_shoulder=16, r_shoulder=17
_BODY_NECK_IDX = 12
_BODY_LSHOULDER_IDX = 16
_BODY_RSHOULDER_IDX = 17


def _mano_rest_pose_hand():
    """MANO rest pose 15个手部关节位置 (右手, 相对手腕原点).

    MANO 坐标系: X=左右, Y=上下, Z=前后.
    手指指向 -Z (前), 手掌朝向 -Y (下).
    返回值: (15, 3) numpy array
    """
    k = np.zeros((15, 3), dtype=np.float32)
    # 食指: MANO[0-2] = MCP, PIP, DIP
    k[0] = [0.025, -0.005, -0.065]
    k[1] = [0.025, -0.005, -0.095]
    k[2] = [0.025, -0.005, -0.115]
    # 中指: MANO[3-5]
    k[3] = [0.0, -0.005, -0.07]
    k[4] = [0.0, -0.005, -0.105]
    k[5] = [0.0, -0.005, -0.125]
    # 小指: MANO[6-8]
    k[6] = [-0.02, -0.005, -0.055]
    k[7] = [-0.03, -0.005, -0.075]
    k[8] = [-0.035, -0.005, -0.085]
    # 无名指: MANO[9-11]
    k[9] = [-0.01, -0.005, -0.065]
    k[10] = [-0.01, -0.005, -0.095]
    k[11] = [-0.01, -0.005, -0.11]
    # 拇指: MANO[12-14] = CMC, MCP, IP
    k[12] = [0.035, 0.01, -0.005]
    k[13] = [0.05, 0.005, -0.025]
    k[14] = [0.055, 0.0, -0.05]
    return k


def _compute_alignment_rotation(src_dir, dst_dir):
    """计算将 src_dir 对齐到 dst_dir 的最小旋转矩阵 (Rodrigues)."""
    src = src_dir / (np.linalg.norm(src_dir) + 1e-8)
    dst = dst_dir / (np.linalg.norm(dst_dir) + 1e-8)

    v = np.cross(src, dst)
    c = np.dot(src, dst)

    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(src, perp)
        v = v / (np.linalg.norm(v) + 1e-8)
        return -np.eye(3, dtype=np.float32) + 2.0 * np.outer(v, v)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + vx @ vx * (1.0 - c) / (np.dot(v, v) + 1e-8)
    return R


def _create_flat_hand_template():
    """创建"平手"模板(21个关键点, 归一化3D坐标, 以手腕为原点)"""
    k = np.zeros((21, 3), dtype=np.float32)
    # 拇指 (向左下)
    k[1] = [-1.0, -0.3, 0.0]
    k[2] = [-1.8, -0.5, 0.0]
    k[3] = [-2.3, -0.4, 0.0]
    k[4] = [-2.8, -0.2, 0.0]
    # 食指 (向上)
    k[5] = [-0.3, 0.8, 0.0]
    k[6] = [-0.2, 1.6, 0.0]
    k[7] = [-0.1, 2.2, 0.0]
    k[8] = [0.0, 2.8, 0.0]
    # 中指 (最长)
    k[9] = [0.0, 0.9, 0.0]
    k[10] = [0.0, 1.8, 0.0]
    k[11] = [0.0, 2.5, 0.0]
    k[12] = [0.0, 3.2, 0.0]
    # 无名指
    k[13] = [0.3, 0.8, 0.0]
    k[14] = [0.5, 1.5, 0.0]
    k[15] = [0.6, 2.0, 0.0]
    k[16] = [0.7, 2.5, 0.0]
    # 小指
    k[17] = [0.6, 0.5, 0.0]
    k[18] = [0.8, 1.1, 0.0]
    k[19] = [1.0, 1.5, 0.0]
    k[20] = [1.2, 1.9, 0.0]
    return k


def extract_hand_rotations(motion_274):
    """
    从 HandMDM 274维特征中提取手部旋转.

    HandMDM 274维结构:
      body_pose:      0:78   (13关节×6D)
      left_hand_pose: 78:168 (15关节×6D)
      right_hand_pose:168:258 (15关节×6D)
      jaw_pose:       258:264 (1关节×6D)
      expression:     264:274 (10维)
    """
    T = motion_274.shape[0]
    lh = motion_274[:, 78:168].reshape(T, 15, 6)
    rh = motion_274[:, 168:258].reshape(T, 15, 6)
    return lh, rh


def heuristic_hand_joints_3d(motion_274):
    """
    简化模式: 从 HandMDM 6D旋转 → 3D手部关键点.

    使用平手模板 + 逐指FK旋转 → (T, 21, 3) × 2.
    HandMDM的6D旋转 → 旋转矩阵 → 应用到手指骨骼链.
    加入 MANO→模板坐标系对齐, 避免手指扭曲.
    """
    T = motion_274.shape[0]
    lh_rot6d, rh_rot6d = extract_hand_rotations(motion_274)
    template = _create_flat_hand_template()
    mano_rest = _mano_rest_pose_hand()

    lh_kpts = np.zeros((T, 21, 3), dtype=np.float32)
    rh_kpts = np.zeros((T, 21, 3), dtype=np.float32)

    for t in range(T):
        lh_t = template.copy()
        rh_t = template.copy()

        for finger_name, chain in _FINGER_CHAINS.items():
            mano_idx_list = _MANO_FINGER_IDX[finger_name]

            lh_t = _fk_finger(lh_t, chain, mano_idx_list, lh_rot6d[t],
                             mano_rest, _MANO_PARENT, template)
            rh_t = _fk_finger(rh_t, chain, mano_idx_list, rh_rot6d[t],
                             mano_rest, _MANO_PARENT, template)

        lh_kpts[t] = lh_t
        rh_kpts[t] = rh_t

    logger.info(f"[simple] 手部3D关节: left={lh_kpts.shape}, right={rh_kpts.shape}")
    return lh_kpts, rh_kpts


def _fk_finger(kpts, chain, mano_idx_list, rot6d_15,
               mano_rest, mano_parent, template):
    """
    对一根手指做正向运动学 (含 MANO→模板坐标系对齐).

    核心修复: MANO 旋转定义在 MANO rest pose 坐标系下,
    而平手模板在 XY 平面. 直接用 MANO 旋转矩阵会因骨骼方向
    不一致导致手指扭曲. 解决方法是:
      R_template = R_align @ R_mano @ R_align^T
    其中 R_align 将 MANO 骨骼方向对齐到模板骨骼方向.

    Args:
        kpts: (21, 3) 当前帧模板关键点 (原地修改)
        chain: [i0, i1, i2, i3] DWPose关节索引
        mano_idx_list: [m0, m1, m2] MANO关节在15维中的索引
        rot6d_15: (15, 6) 当前帧手部旋转
        mano_rest: (15, 3) MANO rest pose 关节位置
        mano_parent: dict MANO 关节亲子关系
        template: (21, 3) 平手模板
    """
    for i in range(len(mano_idx_list)):
        mano_idx = mano_idx_list[i]
        rot_mat_mano = rotation_6d_to_matrix_np(rot6d_15[mano_idx])  # (3,3)

        # ── 计算 MANO→模板 对齐旋转 ──
        parent_mano = mano_parent[mano_idx]
        if parent_mano >= 0:
            mano_dir = mano_rest[mano_idx] - mano_rest[parent_mano]
        else:
            mano_dir = mano_rest[mano_idx]  # parent=wrist(原点)

        child_kpt = chain[i]
        if i > 0:
            parent_kpt_idx = chain[i - 1]
            template_dir = template[child_kpt] - template[parent_kpt_idx]
        else:
            template_dir = template[child_kpt]  # parent=wrist(原点)

        R_align = _compute_alignment_rotation(mano_dir, template_dir)
        rot_mat_aligned = R_align @ rot_mat_mano @ R_align.T

        parent_kpt = kpts[chain[i]]

        # 旋转下游所有关节
        for j in range(i + 1, len(chain)):
            child_idx = chain[j]
            rel = kpts[child_idx] - parent_kpt
            kpts[child_idx] = parent_kpt + rot_mat_aligned @ rel

    return kpts


# ─── SMPL-X 完整模式 ───
def smplx_hand_joints_3d(motion_274, device="cpu"):
    """
    完整模式: 通过 SMPL-X 正向运动学 → 3D手部关节.

    需要: pip install smplx
    需要 SMPL-H 模型文件 (HandMDM/models_smplx_v1_1/models/smplx/SMPLH_NEUTRAL.npz)
    """
    setup_handmdm_path()

    import torch
    from src.tools.geometry import to_matrix, matrix_to

    # SMPL-H 52关节: 1(global) + 21(body) + 15(l_hand) + 15(r_hand)
    LH_START, LH_END = 22, 37
    RH_START, RH_END = 37, 52
    body_pose_lower_joints = [0, 1, 3, 4, 6, 7, 9, 10]  # 缺省的下体关节

    T = motion_274.shape[0]
    all_joints = np.zeros((T, 52, 3), dtype=np.float32)

    def _to_aa(rot6d_np):
        t6 = torch.tensor(rot6d_np, dtype=torch.float32)
        return matrix_to("axisangle", to_matrix("rot6d", t6)).numpy()

    for t in range(T):
        frame = motion_274[t]

        body_aa = _to_aa(frame[0:78].reshape(13, 6))
        lh_aa   = _to_aa(frame[78:168].reshape(15, 6))
        rh_aa   = _to_aa(frame[168:258].reshape(15, 6))

        # 补全下体关节 → 21体关节
        body_full = body_aa.copy()
        for idx in sorted(body_pose_lower_joints):
            body_full = np.insert(body_full, idx, np.zeros(3), axis=0)

        go = np.zeros((1, 3), dtype=np.float32)
        full_aa = np.concatenate([go, body_full, lh_aa, rh_aa], axis=0)

        try:
            joints_3d = _run_smplx_fk(full_aa)
            all_joints[t] = joints_3d
        except Exception as e:
            logger.warning(f"SMPL-X FK t={t} 失败: {e}, 沿用前一帧")
            if t > 0:
                all_joints[t] = all_joints[t - 1]

    lh_3d = all_joints[:, LH_START:LH_END, :]
    rh_3d = all_joints[:, RH_START:RH_END, :]
    lh_dw = mano15_to_dwpose21_batch(lh_3d)
    rh_dw = mano15_to_dwpose21_batch(rh_3d)

    logger.info(f"[smplx] 手部3D关节: left={lh_dw.shape}, right={rh_dw.shape}")
    return lh_dw, rh_dw


_smplx_model_cache = None


def _get_smplx_model():
    """加载/缓存 SMPL-X(H) 模型"""
    global _smplx_model_cache
    if _smplx_model_cache is not None:
        return _smplx_model_cache

    handmdm_dir = setup_handmdm_path()
    model_dir = os.path.join(handmdm_dir, "models_smplx_v1_1", "models", "smplx")

    smplh_path = os.path.join(model_dir, "SMPLH_NEUTRAL.npz")
    smplx_path = os.path.join(model_dir, "SMPLX_NEUTRAL.npz")

    if not os.path.exists(smplh_path):
        if os.path.exists(smplx_path):
            logger.info(f"复制 SMPLX → SMPLH: {smplx_path} → {smplh_path}")
            import shutil
            shutil.copy2(smplx_path, smplh_path)
        else:
            raise FileNotFoundError(
                f"缺少 SMPL-H 模型文件。请将 SMPLH_NEUTRAL.npz 放到 {model_dir}"
            )

    from src.tools.smplx_hack import SMPLHLayer
    import torch

    smpl = SMPLHLayer(model_dir, ext="npz", gender="neutral", num_betas=10)
    smpl.eval()
    _smplx_model_cache = smpl
    logger.info("SMPL-X 模型已缓存")
    return _smplx_model_cache


def _run_smplx_fk(axis_angle_52):
    """单帧 SMPL-X 正向运动学: (52,3) aa → (52,3) joints"""
    import torch
    from src.tools.geometry import to_matrix

    smpl = _get_smplx_model()
    aa = torch.tensor(axis_angle_52, dtype=torch.float32).unsqueeze(0)

    go_m = to_matrix("axisangle", aa[:, 0:1, :])
    bp_m = to_matrix("axisangle", aa[:, 1:22, :])
    lh_m = to_matrix("axisangle", aa[:, 22:37, :])
    rh_m = to_matrix("axisangle", aa[:, 37:52, :])

    with torch.no_grad():
        out = smpl(global_orient=go_m, body_pose=bp_m,
                   left_hand_pose=lh_m, right_hand_pose=rh_m,
                   betas=torch.zeros(1, 10))
    return out.joints.squeeze(0).numpy()


def mano15_to_dwpose21_batch(mano_3d):
    """
    MANO 15关节(3D) → DWPose 21关键点(3D).

    Args:
        mano_3d: (T, 15, 3)
    Returns:
        (T, 21, 3)
    """
    T = mano_3d.shape[0]
    out = np.zeros((T, 21, 3), dtype=np.float32)

    # 手腕 = MCP中点 - 偏移
    mcp_idx = [0, 3, 6, 9, 12]
    for t in range(T):
        out[t, 0] = np.mean(mano_3d[t, mcp_idx], axis=0) - np.array([0, 0.8, 0])

    # 直接映射 (MANO 0-based → DWPose)
    for mano_i, dw_i in _MANO_TO_DWPOSE.items():
        out[:, dw_i, :] = mano_3d[:, mano_i, :]

    # 外推指尖
    for tip, (parent, grandparent) in _DWPOSE_TIP_EXTRAP.items():
        direction = out[:, parent, :] - out[:, grandparent, :]
        nrm = np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8
        bone_len = np.linalg.norm(
            out[:, parent, :] - out[:, grandparent, :], axis=-1, keepdims=True)
        out[:, tip, :] = out[:, parent, :] + (direction / nrm) * bone_len * 0.7

    return out


# ============================================================
# Phase 3: 3D → 2D 空间投影
# ============================================================
def extract_2d_anchors_from_ref(ref_img_path, dwpose_det=None, dwpose_pose=None, device="cpu"):
    """对参考图跑 DWPose, 提取脖颈/双肩 2D 像素坐标.

    DWPose 18点格式:
      0=nose, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist,
      5=l_shoulder, 6=l_elbow, 7=l_wrist, ...

    Returns:
        neck_2d: (2,) 脖颈像素坐标
        l_sh_2d: (2,) 左肩像素坐标
        r_sh_2d: (2,) 右肩像素坐标
        sw_2d: float 2D肩宽 (像素)
    """
    import cv2
    from PIL import Image

    # 初始化 DWPose detector (如未提供实例)
    detector = dwpose_det
    if detector is None or isinstance(detector, str):
        det_path = dwpose_det
        pose_path = dwpose_pose
        if det_path is None or pose_path is None:
            raise ValueError(
                "必须提供 dwpose_det 和 dwpose_pose 路径 (DWPose ONNX 模型), "
                "或传入已初始化的 DWposeDetector 实例")

        echomimic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2")
        if echomimic_dir not in sys.path:
            sys.path.insert(0, echomimic_dir)

        from src.models.dwpose.dwpose_detector import DWposeDetector
        detector = DWposeDetector(
            model_det=det_path, model_pose=pose_path, device=device)

    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        pil_img = np.array(Image.open(ref_img_path).convert('RGB'))
        ref_img = pil_img[:, :, ::-1]  # RGB → BGR
    H, W = ref_img.shape[:2]

    pose = detector(ref_img)
    body = pose['bodies']['candidate']  # (nums*18, 2) 归一化 [0,1]

    if body.shape[0] == 0:
        logger.warning("DWPose 未检测到人体, 使用默认画布中心锚点")
        hw = IMG_SIZE // 2
        neck_2d = np.array([hw, hw - 80], dtype=np.float32)
        l_sh_2d = np.array([hw - 80, hw - 120], dtype=np.float32)
        r_sh_2d = np.array([hw + 80, hw - 120], dtype=np.float32)
        sw_2d = np.linalg.norm(l_sh_2d - r_sh_2d)
        return neck_2d, l_sh_2d, r_sh_2d, sw_2d

    # DWposeDetector 返回的 candidate 是 (nums*18, 2), 需 reshape 为 (nums, 18, 2)
    nums = body.shape[0] // 18
    bodies_reshaped = body.reshape(nums, 18, 2)  # 归一化 [0,1]

    # 取第一个人的关键点, 转回像素坐标
    neck_2d = bodies_reshaped[0, 1] * np.array([W, H])   # keypoint 1 = neck
    l_sh_2d = bodies_reshaped[0, 5] * np.array([W, H])   # keypoint 5 = l_shoulder
    r_sh_2d = bodies_reshaped[0, 2] * np.array([W, H])   # keypoint 2 = r_shoulder
    sw_2d = np.linalg.norm(l_sh_2d - r_sh_2d)

    logger.info(f"DWPose 2D锚点: neck=({neck_2d[0]:.0f},{neck_2d[1]:.0f}), "
                f"shoulder_width_2d={sw_2d:.1f}px (ref_img={W}x{H})")
    return neck_2d, l_sh_2d, r_sh_2d, sw_2d


def extract_body_anchors_3d(motion_274, mode="simple", device="cpu"):
    """提取颈部、双肩3D位置.

    simple 模式: 硬编码 SMPL-H rest pose 上身关节静态位置 (模板单位, 与平手模板一致).
                 手长 ~3.2 单位, 颈在手腕上方 ~10 单位, 肩宽 ~8 单位.
    smplx  模式: 逐帧 SMPL-H FK → 真实3D关节 (米), 从 52 关节中提取.
    """
    T = motion_274.shape[0]

    if mode == "simple":
        # 脖子在手腕"下方"(模板Y轴), 手在脖子下方 ≈ 12单位, 肩宽 ≈ 8单位
        # 这样 hand_3d - neck_3d 的 Y 为正值, 投影后手在画面脖子下方
        neck = np.full((T, 3), [0.0, -12.0, 0.0], dtype=np.float32)
        l_sh = np.full((T, 3), [-4.0, -12.0, 0.0], dtype=np.float32)
        r_sh = np.full((T, 3), [4.0, -12.0, 0.0], dtype=np.float32)
        logger.info(f"[simple] body anchors: neck~[0,-12,0], shoulder_width_3d~8.0")
        return neck, l_sh, r_sh

    # smplx 模式: 逐帧 SMPL-H FK
    setup_handmdm_path()
    import torch
    from src.tools.geometry import to_matrix, matrix_to

    body_pose_lower_joints = [0, 1, 3, 4, 6, 7, 9, 10]

    def _to_aa(rot6d_np):
        t6 = torch.tensor(rot6d_np, dtype=torch.float32)
        return matrix_to("axisangle", to_matrix("rot6d", t6)).numpy()

    neck = np.zeros((T, 3), dtype=np.float32)
    l_sh = np.zeros((T, 3), dtype=np.float32)
    r_sh = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        frame = motion_274[t]
        body_aa = _to_aa(frame[0:78].reshape(13, 6))

        body_full = body_aa.copy()
        for idx in sorted(body_pose_lower_joints):
            body_full = np.insert(body_full, idx, np.zeros(3), axis=0)

        go = np.zeros((1, 3), dtype=np.float32)
        lh_aa = np.zeros((15, 3), dtype=np.float32)
        rh_aa = np.zeros((15, 3), dtype=np.float32)
        full_aa = np.concatenate([go, body_full, lh_aa, rh_aa], axis=0)

        try:
            joints_3d = _run_smplx_fk(full_aa)
            neck[t] = joints_3d[_BODY_NECK_IDX]
            l_sh[t] = joints_3d[_BODY_LSHOULDER_IDX]
            r_sh[t] = joints_3d[_BODY_RSHOULDER_IDX]
        except Exception as e:
            logger.warning(f"SMPL-X body FK t={t} 失败: {e}, 沿用前一帧")
            if t > 0:
                neck[t], l_sh[t], r_sh[t] = neck[t - 1], l_sh[t - 1], r_sh[t - 1]

    sw = np.linalg.norm(l_sh - r_sh, axis=-1).mean()
    logger.info(f"[smplx] body anchors: mean_shoulder_width_3d={sw:.3f}m")
    return neck, l_sh, r_sh


def project_3d_to_2d(lh_3d, rh_3d, neck_3d, l_shoulder_3d, r_shoulder_3d,
                     neck_2d, l_sh_2d, r_sh_2d, sw_2d, img_size=IMG_SIZE):
    """3D→2D投影: 左右手各自相对于同侧肩膀定位, 肩宽比缩放.

    lh_2d = l_sh_2d + (lh_3d - l_shoulder_3d)[:,:2] * scale_factor
    rh_2d = r_sh_2d + (rh_3d - r_shoulder_3d)[:,:2] * scale_factor
    scale_factor = 2D肩宽(px) / 3D肩宽(3D单位)

    Args:
        neck_2d: (2,) 脖颈像素坐标 (调试用)
        l_sh_2d: (2,) 左肩像素坐标
        r_sh_2d: (2,) 右肩像素坐标
        sw_2d: float 2D肩宽 (像素)
    Returns:
        hand_2d: (T, 42, 2) 合并双手2D关键点, 像素坐标
    """
    T = lh_3d.shape[0]
    eps = 1e-6

    sw_3d = np.mean(np.linalg.norm(l_shoulder_3d - r_shoulder_3d, axis=-1))
    scale_factor = sw_2d / (sw_3d + eps) if sw_3d > eps else 1.0

    logger.info(f"投影参数: sw_2d={sw_2d:.1f}px, sw_3d={sw_3d:.3f}, "
                f"scale={scale_factor:.1f}px/unit")
    logger.info(f"  左肩2D=({l_sh_2d[0]:.0f},{l_sh_2d[1]:.0f}) 右肩2D=({r_sh_2d[0]:.0f},{r_sh_2d[1]:.0f})")

    hand_2d = np.zeros((T, HAND_KPTS_PER_HAND * 2, 2), dtype=np.float32)

    for t in range(T):
        # 左手相对于左肩, 右手相对于右肩
        lh_2d = l_sh_2d + (lh_3d[t] - l_shoulder_3d[t])[:, :2] * scale_factor
        rh_2d = r_sh_2d + (rh_3d[t] - r_shoulder_3d[t])[:, :2] * scale_factor

        combined = np.concatenate([lh_2d, rh_2d], axis=0)
        hand_2d[t] = np.clip(combined, 0, img_size - 1)

    logger.info(f"3D→2D投影: output={hand_2d.shape} [T,42,2]")
    return hand_2d


# ============================================================
# Phase 4: DWPose 格式导出
# ============================================================
def export_dwpose_frames(hand_2d, output_dir, img_size=IMG_SIZE):
    """
    将 (T,42,2) 手部关键点导出为 EchoMimicV2 DWPose 逐帧 .npy.

    每帧格式:
      {
        'bodies': {'candidate': (1,18,2), 'score': (1,18)},
        'hands': (2,21,2),            ← 归一化 [0,1]
        'hands_score': (2,21),
        'faces': (1,70,2),
        'faces_score': (1,70),
        'draw_pose_params': [H, W, rb, re, cb, ce]
      }

    Returns:
        frames_dir: single_frames 子目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "single_frames")
    os.makedirs(frames_dir, exist_ok=True)

    T = hand_2d.shape[0]

    for t in range(T):
        kpts_norm = hand_2d[t] / img_size
        lh_norm = kpts_norm[:21]
        rh_norm = kpts_norm[21:42]
        hands = np.stack([lh_norm, rh_norm], axis=0)

        hands_score = np.ones((2, 21), dtype=np.float32)
        bodies_candidate = np.zeros((1, 18, 2), dtype=np.float32)
        bodies_score = np.zeros((1, 18), dtype=np.float32)
        faces = np.zeros((1, 70, 2), dtype=np.float32)
        faces_score = np.zeros((1, 70), dtype=np.float32)

        pose_dict = {
            'bodies': {
                'candidate': bodies_candidate,
                'score': bodies_score,
            },
            'hands': hands,
            'hands_score': hands_score,
            'faces': faces,
            'faces_score': faces_score,
            'draw_pose_params': [img_size, img_size,
                                  0, img_size, 0, img_size],
        }

        np.save(os.path.join(frames_dir, f"{t}.npy"), pose_dict)

    # 额外保存合并文件 (调试用)
    np.save(os.path.join(output_dir, "all_frames_hand_kpts.npy"), hand_2d)

    logger.info(f"DWPose导出: {T} 帧 → {frames_dir}/")
    return frames_dir


# ============================================================
# Phase 5: EchoMimicV2 渲染
# ============================================================
def setup_echomimic_path():
    """将 echomimic_v2 加入 sys.path"""
    echomimic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2")
    if echomimic_dir not in sys.path:
        sys.path.insert(0, echomimic_dir)
    return echomimic_dir


def load_echomimic_pipeline(config_path, device="cuda"):
    """加载 EchoMimicV2 完整管线"""
    setup_echomimic_path()

    from omegaconf import OmegaConf
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    echomimic_root = os.path.dirname(os.path.abspath(setup_echomimic_path()))

    config = OmegaConf.load(config_path)
    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA 不可用, 回退到 CPU")
        device = "cpu"
        weight_dtype = torch.float32

    # 将 config 中的相对路径解析为绝对路径 (相对于 echomimic_v2 根目录)
    def _resolve(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        # 先尝试相对于 config 文件, 再尝试相对于 echomimic_v2 根
        for base in [config_dir, echomimic_root]:
            resolved = os.path.join(base, p)
            if os.path.exists(resolved):
                return resolved
        return os.path.join(echomimic_root, p)

    config.pretrained_base_model_path = _resolve(config.pretrained_base_model_path)
    config.pretrained_vae_path = _resolve(config.pretrained_vae_path)
    config.denoising_unet_path = _resolve(config.denoising_unet_path)
    config.reference_unet_path = _resolve(config.reference_unet_path)
    config.pose_encoder_path = _resolve(config.pose_encoder_path)
    config.motion_module_path = _resolve(config.motion_module_path)
    config.audio_model_path = _resolve(config.audio_model_path)

    inference_config_path = _resolve(config.inference_config)
    infer_config = OmegaConf.load(inference_config_path)

    logger.info("加载 EchoMimicV2 组件...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device, dtype=weight_dtype)

    from src.models.unet_2d_condition import UNet2DConditionModel
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"))

    from src.models.unet_3d_emo import EMOUNet3DConditionModel
    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path, config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

    from src.models.pose_encoder import PoseEncoder
    pose_net = PoseEncoder(320, conditioning_channels=3,
                           block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device)
    pose_net.load_state_dict(torch.load(config.pose_encoder_path, map_location="cpu"))

    from src.models.whisper.audio2feature import load_audio_model
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
    pipe = EchoMimicV2Pipeline(
        vae=vae, reference_unet=reference_unet, denoising_unet=denoising_unet,
        audio_guider=audio_processor, pose_encoder=pose_net, scheduler=scheduler,
    ).to(device, dtype=weight_dtype)

    logger.info("EchoMimicV2 管线就绪")
    return pipe, infer_config, weight_dtype, device


def render_echomimic_video(
    pipe, ref_img_path, audio_path, frames_dir, output_video,
    infer_config, weight_dtype, device,
    width=768, height=768, steps=30, cfg=2.5, seed=3407,
    context_frames=12, context_overlap=3, sample_rate=16000,
):
    """
    使用 EchoMimicV2 渲染最终视频.

    Args:
        frames_dir: 包含 {idx}.npy 文件的目录
    """
    import torch
    from PIL import Image
    from src.utils.dwpose_util import draw_pose_select_v2
    from src.utils.util import save_videos_grid
    from moviepy.editor import AudioFileClip, VideoFileClip
    import tempfile

    fps = FPS_ECHOMIMIC
    ref_image_pil = Image.open(ref_img_path).resize((width, height))
    audio_clip = AudioFileClip(audio_path)

    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.npy')])
    total_frames = min(int(audio_clip.duration * fps), n_frames)
    logger.info(f"渲染帧数: {total_frames} (音频{audio_clip.duration:.1f}s, 姿态{n_frames}帧)")

    pose_list = []
    for idx in range(total_frames):
        pose_path = os.path.join(frames_dir, f"{idx}.npy")
        detected_pose = np.load(pose_path, allow_pickle=True).tolist()

        imh, imw, rb, re, cb, ce = detected_pose['draw_pose_params']


        # ========== 调试代码 start ==========
        import numpy as np
        from PIL import Image
        print("\n========== 单帧手部关键点调试 ==========")
        # 1. 打印手部关键点数组信息
        hands_arr = detected_pose["hands"]
        print(f"hands 数组形状: {hands_arr.shape}")
        print(f"hands X坐标最小/最大: {hands_arr[...,0].min():.4f} / {hands_arr[...,0].max():.4f}")
        print(f"hands Y坐标最小/最大: {hands_arr[...,1].min():.4f} / {hands_arr[...,1].max():.4f}")

        # 2. 检测身体关键点是否全0（EchoMimic不画手的头号原因）
        body_kpt = detected_pose["bodies"]["candidate"]
        print(f"身体关键点是否全部为0: {np.all(body_kpt == 0)}")

        # 3. 打印画布裁剪范围
        print(f"draw_pose_params: {detected_pose['draw_pose_params']}")

        # 4. 绘制姿态并保存图片，肉眼看有没有手
        im_debug = draw_pose_select_v2(detected_pose, imh, imw, ref_w=800)
        img_save = Image.fromarray(np.transpose(np.array(im_debug), (1, 2, 0)))
        img_save.save("debug_pose_frame.png")
        print("已生成图片 debug_pose_frame.png，打开查看是否有绿色手部骨架！")
        # ========== 调试代码 end ==========



        imh, imw = int(imh), int(imw)

        tgt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        im = draw_pose_select_v2(detected_pose, imh, imw, ref_w=800)
        im = np.transpose(np.array(im), (1, 2, 0))
        tgt_mask[int(rb):int(re), int(cb):int(ce), :] = im

        tgt_pil = Image.fromarray(tgt_mask).convert('RGB')
        pose_list.append(
            torch.Tensor(np.array(tgt_pil)).to(dtype=weight_dtype, device=device)
            .permute(2, 0, 1) / 255.0
        )

    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    generator = torch.manual_seed(seed)

    logger.info("EchoMimicV2 推理中...")
    video = pipe(
        ref_image_pil, audio_path,
        poses_tensor[:, :, :total_frames, ...],
        width, height, total_frames, steps, cfg,
        generator=generator, audio_sample_rate=sample_rate,
        context_frames=context_frames, fps=fps,
        context_overlap=context_overlap, start_idx=0,
    ).videos

    final_len = min(video.shape[2], poses_tensor.shape[2], total_frames)
    video_sig = video[:, :, :final_len, :, :]

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name

    save_videos_grid(video_sig, tmp_path, n_rows=1, fps=fps)

    video_clip = VideoFileClip(tmp_path)
    video_clip = video_clip.set_audio(audio_clip.set_duration(final_len / fps))
    video_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", threads=2)

    os.unlink(tmp_path)
    logger.info(f"视频已保存: {output_video}")
    return output_video


# ============================================================
# Phase 6: 主管线
# ============================================================
def run_pipeline(
    text_prompt=None,
    motion_file=None,
    ref_img_path=None,
    audio_path=None,
    duration_sec=None,
    output_video=None,
    mode="simple",
    handmdm_checkpoint=None,
    echomimic_config=None,
    device="cpu",
    guidance=15.0,
    seed=1234,
    output_dir="./pipeline_output",
    dwpose_det=None,
    dwpose_pose=None,
):
    """端到端管线主函数.

    Args:
        text_prompt: 文本描述 (与 motion_file 二选一)
        motion_file: 预生成的 (T,274) .npy 文件 (与 text_prompt 二选一)
        dwpose_det: DWPose 检测模型路径 (yolox_l.onnx)
        dwpose_pose: DWPose 姿态模型路径 (dw-ll_ucoco_384.onnx)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if handmdm_checkpoint is None:
        handmdm_checkpoint = os.path.join(
            base_dir, "HandMDM", "models",
            "mdm_bobsl3dt_phonology_hms", "checkpoints", "last.ckpt")
    if echomimic_config is None:
        echomimic_config = os.path.join(
            base_dir, "echomimic_v2", "configs", "prompts", "infer.yaml")

    # ── Step 0: 从参考图提取2D锚点 ──
    logger.info("=" * 60)
    logger.info("Step 0/6: DWPose 2D锚点提取 (参考图)")
    try:
        neck_2d, l_sh_2d, r_sh_2d, sw_2d = extract_2d_anchors_from_ref(
            ref_img_path, dwpose_det=dwpose_det, dwpose_pose=dwpose_pose, device=device)
    except Exception as e:
        logger.warning(f"DWPose 2D锚点提取失败: {e}, 使用默认锚点")
        hw = IMG_SIZE // 2
        neck_2d = np.array([hw, hw - 80], dtype=np.float32)
        l_sh_2d = np.array([hw - 80, hw - 120], dtype=np.float32)
        r_sh_2d = np.array([hw + 80, hw - 120], dtype=np.float32)
        sw_2d = 160.0

    # ── Step 1: 获取运动数据 (文本生成 或 直接加载) ──
    logger.info("=" * 60)
    if motion_file is not None:
        logger.info(f"Step 1/6: 加载预生成运动文件 [{motion_file}]")
        motion_274 = np.load(motion_file)
        if motion_274.ndim != 2 or motion_274.shape[1] not in (274, 284):
            raise ValueError(
                f"运动文件 shape={motion_274.shape}, 期望 (T, 274) 或 (T, 284)")
        if motion_274.shape[1] == 284:
            motion_274 = motion_274[:, :274]
        T = motion_274.shape[0]
        dur = T / FPS_HANDMDM
        logger.info(f"已加载: {T}帧 @{FPS_HANDMDM}fps ({dur:.2f}s)")
    elif text_prompt is not None:
        if duration_sec is None:
            raise ValueError("--text 模式下 --duration 为必选项")
        logger.info(f"Step 1/6: HandMDM 生成 [{text_prompt}] ({duration_sec}s)")
        diffusion, text_encoder = load_handmdm_model(handmdm_checkpoint, device=device)
        handmdm_len = max(int(duration_sec * FPS_HANDMDM), 14)
        motion_274 = generate_handmdm_motion(
            diffusion, text_encoder, text_prompt,
            length=handmdm_len, guidance=guidance, seed=seed, device=device)
    else:
        raise ValueError("必须提供 --text 或 --motion_file 之一")

    # ── Step 2: 帧率转换 ──
    logger.info("=" * 60)
    logger.info("Step 2/6: 帧率转换 25fps → 24fps")
    motion_274_24 = convert_fps(motion_274)

    # ── Step 3: 3D手部关节 ──
    logger.info("=" * 60)
    logger.info(f"Step 3/6: 3D手部关节提取 (mode={mode})")
    if mode == "smplx":
        lh_3d, rh_3d = smplx_hand_joints_3d(motion_274_24, device=device)
    else:
        lh_3d, rh_3d = heuristic_hand_joints_3d(motion_274_24)

    # ── Step 4: 3D→2D ──
    logger.info("=" * 60)
    logger.info("Step 4/6: 空间投影 3D→2D")
    neck_3d, l_sh, r_sh = extract_body_anchors_3d(motion_274_24, mode=mode, device=device)
    hand_2d = project_3d_to_2d(lh_3d, rh_3d, neck_3d, l_sh, r_sh,
                                neck_2d=neck_2d, l_sh_2d=l_sh_2d,
                                r_sh_2d=r_sh_2d, sw_2d=sw_2d)

    # ── Step 5: DWPose导出 ──
    logger.info("=" * 60)
    logger.info("Step 5/6: DWPose逐帧导出")
    pose_dir = os.path.join(output_dir, "pose")
    frames_dir = export_dwpose_frames(hand_2d, pose_dir)

    # ── Step 6: EchoMimicV2 ──
    logger.info("=" * 60)
    logger.info("Step 6/6: EchoMimicV2 渲染")
    pipe, infer_cfg, wdt, dev = load_echomimic_pipeline(echomimic_config, device=device)
    result = render_echomimic_video(
        pipe, ref_img_path, audio_path, frames_dir, output_video,
        infer_cfg, wdt, dev)

    logger.info("=" * 60)
    logger.info(f"完成! → {result}")
    return result


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="HandMDM + EchoMimicV2 端到端集成管线")

    parser.add_argument("--text", type=str, default=None,
                        help="手势文本描述 (英文), 与 --motion_file 二选一")
    parser.add_argument("--motion_file", type=str, default=None,
                        help="HandMDM 预生成的 (T,274) .npy 运动文件, 跳过文本生成步骤")
    parser.add_argument("--ref_img", type=str, required=True,
                        help="参考人像图片路径")
    parser.add_argument("--audio", type=str, required=True,
                        help="驱动音频路径 (.wav)")
    parser.add_argument("--duration", type=float, default=None,
                        help="视频时长 (秒), --text 模式下必需")
    parser.add_argument("--output", type=str, default="./output_video.mp4",
                        help="输出视频路径")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "smplx"],
                        help="手部提取: simple(启发式,默认) | smplx(SMPL-X)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备: cpu | cuda")
    parser.add_argument("--guidance", type=float, default=15.0,
                        help="HandMDM CFG 引导强度")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--handmdm_ckpt", type=str, default=None,
                        help="HandMDM checkpoint")
    parser.add_argument("--echomimic_config", type=str, default=None,
                        help="EchoMimicV2 config.yaml")
    parser.add_argument("--output_dir", type=str, default="./pipeline_output",
                        help="中间文件目录")
    parser.add_argument("--dwpose_det", type=str, default=None,
                        help="DWPose 检测模型路径 (yolox_l.onnx)")
    parser.add_argument("--dwpose_pose", type=str, default=None,
                        help="DWPose 姿态模型路径 (dw-ll_ucoco_384.onnx)")

    args = parser.parse_args()

    if not args.text and not args.motion_file:
        parser.error("必须提供 --text 或 --motion_file 之一")
    if args.text and not args.duration:
        parser.error("--text 模式下 --duration 为必选项")

    run_pipeline(
        text_prompt=args.text,
        motion_file=args.motion_file,
        ref_img_path=args.ref_img,
        audio_path=args.audio,
        duration_sec=args.duration,
        output_video=args.output,
        mode=args.mode,
        handmdm_checkpoint=args.handmdm_ckpt,
        echomimic_config=args.echomimic_config,
        device=args.device,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        dwpose_det=args.dwpose_det,
        dwpose_pose=args.dwpose_pose,
    )


if __name__ == "__main__":
    main()
