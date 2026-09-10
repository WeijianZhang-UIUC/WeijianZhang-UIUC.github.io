#!/usr/bin/env python3
"""
HandMDM .npy → EchoMimicV2 姿态预处理脚本
===========================================

输入: HandMDM 推理输出的 (T, 274) .npy + 参考图
输出: 整理好的文件夹，包含逐帧 DWPose 格式 .npy，可直接给 EchoMimicV2 使用

管线:
  输入.npy (T,274) @25fps
    → 帧率转换 25→24fps
    → 6D旋转 → 3D手部关节
    → 空间投影 3D→2D
    → DWPose格式逐帧导出
    → 输出文件夹

用法:
  python prepare_pose_for_echomimic.py \
      --input motion.npy \
      --ref_img reference.png \
      --output ./pose_output
"""

import os
import sys
import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PreparePose")

# ============================================================
# 全局配置
# ============================================================
FPS_HANDMDM = 25
FPS_ECHOMIMIC = 24
IMG_SIZE = 768
HAND_KPTS = 21

# ─── MANO 15关节 → DWPose 21关键点 骨骼定义 ───
# MANO (SMPL-H标准顺序):
#   [0,1,2]=食指(MCP,PIP,DIP), [3,4,5]=中指, [6,7,8]=小指,
#   [9,10,11]=无名指, [12,13,14]=拇指(CMC,MCP,IP)
# DWPose: 0=手腕, 1-4=拇指, 5-8=食指, 9-12=中指, 13-16=无名指, 17-20=小指

_FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

_MANO_FINGER_IDX = {
    "thumb":  [12, 13, 14],
    "index":  [0, 1, 2],
    "middle": [3, 4, 5],
    "ring":   [9, 10, 11],
    "pinky":  [6, 7, 8],
}

# MANO 关节亲子关系
_MANO_PARENT = {
    0: -1, 1: 0, 2: 1, 3: -1, 4: 3, 5: 4,
    6: -1, 7: 6, 8: 7, 9: -1, 10: 9, 11: 10,
    12: -1, 13: 12, 14: 13,
}


# ============================================================
# 数学工具
# ============================================================
def rotation_6d_to_matrix_np(d6):
    """6D旋转 → 3×3旋转矩阵 (Gram-Schmidt)"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _mano_rest_pose_hand():
    """MANO rest pose 15个手部关节位置 (右手, 相对手腕原点)."""
    k = np.zeros((15, 3), dtype=np.float32)
    k[0] = [0.025, -0.005, -0.065]   # index MCP
    k[1] = [0.025, -0.005, -0.095]   # index PIP
    k[2] = [0.025, -0.005, -0.115]   # index DIP
    k[3] = [0.0, -0.005, -0.07]      # middle MCP
    k[4] = [0.0, -0.005, -0.105]     # middle PIP
    k[5] = [0.0, -0.005, -0.125]     # middle DIP
    k[6] = [-0.02, -0.005, -0.055]   # pinky MCP
    k[7] = [-0.03, -0.005, -0.075]   # pinky PIP
    k[8] = [-0.035, -0.005, -0.085]  # pinky DIP
    k[9] = [-0.01, -0.005, -0.065]   # ring MCP
    k[10] = [-0.01, -0.005, -0.095]  # ring PIP
    k[11] = [-0.01, -0.005, -0.11]   # ring DIP
    k[12] = [0.035, 0.01, -0.005]    # thumb CMC
    k[13] = [0.05, 0.005, -0.025]    # thumb MCP
    k[14] = [0.055, 0.0, -0.05]      # thumb IP
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


def _flat_hand_template():
    """平手模板 (21关键点, 3D, 手腕为原点)"""
    k = np.zeros((21, 3), dtype=np.float32)
    k[1] = [-1.0, -0.3, 0.0]     # 拇指CMC
    k[2] = [-1.8, -0.5, 0.0]     # 拇指MCP
    k[3] = [-2.3, -0.4, 0.0]     # 拇指IP
    k[4] = [-2.8, -0.2, 0.0]     # 拇指TIP
    k[5] = [-0.3, 0.8, 0.0]      # 食指MCP
    k[6] = [-0.2, 1.6, 0.0]      # 食指PIP
    k[7] = [-0.1, 2.2, 0.0]      # 食指DIP
    k[8] = [0.0, 2.8, 0.0]       # 食指TIP
    k[9] = [0.0, 0.9, 0.0]       # 中指MCP
    k[10] = [0.0, 1.8, 0.0]      # 中指PIP
    k[11] = [0.0, 2.5, 0.0]      # 中指DIP
    k[12] = [0.0, 3.2, 0.0]      # 中指TIP
    k[13] = [0.3, 0.8, 0.0]      # 无名指MCP
    k[14] = [0.5, 1.5, 0.0]      # 无名指PIP
    k[15] = [0.6, 2.0, 0.0]      # 无名指DIP
    k[16] = [0.7, 2.5, 0.0]      # 无名指TIP
    k[17] = [0.6, 0.5, 0.0]      # 小指MCP
    k[18] = [0.8, 1.1, 0.0]      # 小指PIP
    k[19] = [1.0, 1.5, 0.0]      # 小指DIP
    k[20] = [1.2, 1.9, 0.0]      # 小指TIP
    return k


# ============================================================
# Step 1: 帧率转换
# ============================================================
def convert_fps(motion_274, src_fps=FPS_HANDMDM, tgt_fps=FPS_ECHOMIMIC):
    """25fps → 24fps 线性插值"""
    T_src = motion_274.shape[0]
    if T_src == 0:
        raise ValueError("输入 .npy 帧数为0")
    duration = T_src / src_fps
    T_tgt = max(int(np.ceil(duration * tgt_fps)), 1)
    if T_src == T_tgt:
        logger.info(f"帧率无需转换: {T_src}帧")
        return motion_274
    src_t = np.linspace(0, 1, T_src)
    dst_t = np.linspace(0, 1, T_tgt)
    result = np.zeros((T_tgt, motion_274.shape[1]), dtype=motion_274.dtype)
    for d in range(motion_274.shape[1]):
        result[:, d] = np.interp(dst_t, src_t, motion_274[:, d])
    logger.info(f"帧率转换: {T_src}帧@{src_fps}fps → {T_tgt}帧@{tgt_fps}fps ({duration:.2f}s)")
    return result


# ============================================================
# Step 2: 3D手部关节提取
# ============================================================
def extract_3d_hands(motion_274):
    """
    274维特征 → 双手 3D 关键点.

    HandMDM 274维结构:
      [0:78]   body_pose (13关节×6D)
      [78:168] left_hand_pose (15关节×6D)
      [168:258] right_hand_pose (15关节×6D)
      [258:264] jaw_pose (1关节×6D)
      [264:274] expression (10维)
    """
    T = motion_274.shape[0]
    lh_rot = motion_274[:, 78:168].reshape(T, 15, 6)
    rh_rot = motion_274[:, 168:258].reshape(T, 15, 6)
    template = _flat_hand_template()
    mano_rest = _mano_rest_pose_hand()

    lh = np.zeros((T, 21, 3), dtype=np.float32)
    rh = np.zeros((T, 21, 3), dtype=np.float32)

    for t in range(T):
        lt = template.copy()
        rt = template.copy()
        for fname, chain in _FINGER_CHAINS.items():
            midx = _MANO_FINGER_IDX[fname]
            lt = _fk_chain(lt, chain, midx, lh_rot[t], mano_rest, _MANO_PARENT, template)
            rt = _fk_chain(rt, chain, midx, rh_rot[t], mano_rest, _MANO_PARENT, template)
        lh[t] = lt
        rh[t] = rt

    logger.info(f"3D手部: left={lh.shape}, right={rh.shape}")
    return lh, rh


def _fk_chain(kpts, chain, mano_idx_list, rot6d_15,
              mano_rest, mano_parent, template):
    """单指正向运动学: 含 MANO→模板坐标系对齐.

    R_template = R_align @ R_mano @ R_align^T
    其中 R_align 将 MANO 骨骼方向对齐到模板骨骼方向.
    """
    for i in range(len(mano_idx_list)):
        mano_idx = mano_idx_list[i]
        R_mano = rotation_6d_to_matrix_np(rot6d_15[mano_idx])

        parent_mano = mano_parent[mano_idx]
        if parent_mano >= 0:
            mano_dir = mano_rest[mano_idx] - mano_rest[parent_mano]
        else:
            mano_dir = mano_rest[mano_idx]

        child_kpt = chain[i]
        if i > 0:
            parent_kpt_idx = chain[i - 1]
            template_dir = template[child_kpt] - template[parent_kpt_idx]
        else:
            template_dir = template[child_kpt]

        R_align = _compute_alignment_rotation(mano_dir, template_dir)
        R_aligned = R_align @ R_mano @ R_align.T

        parent = kpts[chain[i]]
        for j in range(i + 1, len(chain)):
            child = chain[j]
            kpts[child] = parent + R_aligned @ (kpts[child] - parent)
    return kpts


# ============================================================
# Step 3: 3D → 2D 空间投影
# ============================================================
def project_to_2d(lh_3d, rh_3d, neck_2d, l_sh_2d, r_sh_2d, sw_2d, img_size=IMG_SIZE):
    """左右手各自相对于同侧肩膀定位, 肩宽比缩放.

    lh_2d = l_sh_2d + (lh_3d - l_sh_3d)[:,:2] * scale
    rh_2d = r_sh_2d + (rh_3d - r_sh_3d)[:,:2] * scale
    """
    T = lh_3d.shape[0]

    # 3D锚点: 脖子在手腕下方(Y=-12), 与 handmdm2echomimic.py 一致
    neck_3d = np.array([0.0, -12.0, 0.0], dtype=np.float32)
    l_sh_3d = np.array([-4.0, -12.0, 0.0], dtype=np.float32)
    r_sh_3d = np.array([4.0, -12.0, 0.0], dtype=np.float32)

    sw_3d = np.linalg.norm(l_sh_3d - r_sh_3d)
    scale = sw_2d / (sw_3d + 1e-6) if sw_3d > 1e-6 else 1.0

    logger.info(f"投影: sw_2d={sw_2d:.1f}px, sw_3d={sw_3d:.1f}, scale={scale:.1f}px/unit")

    hand_2d = np.zeros((T, HAND_KPTS * 2, 2), dtype=np.float32)

    for t in range(T):
        lh_2d = l_sh_2d + (lh_3d[t] - l_sh_3d)[:, :2] * scale
        rh_2d = r_sh_2d + (rh_3d[t] - r_sh_3d)[:, :2] * scale
        combined = np.concatenate([lh_2d, rh_2d], axis=0)
        hand_2d[t] = np.clip(combined, 0, img_size - 1)

    logger.info(f"2D投影: {hand_2d.shape} [T,42,2]")
    return hand_2d


# ============================================================
# Step 4: DWPose 格式导出
# ============================================================
def extract_2d_anchors_from_ref(ref_img_path, dwpose_det=None, dwpose_pose=None, device="cpu"):
    """对参考图跑 DWPose, 提取脖颈/双肩 2D 像素坐标.

    DWPose 18点: 1=neck, 2=r_shoulder, 5=l_shoulder.
    """
    import cv2
    from PIL import Image

    detector = dwpose_det
    if detector is None or isinstance(detector, str):
        det_path = dwpose_det
        pose_path = dwpose_pose
        if det_path is None or pose_path is None:
            raise ValueError("必须提供 DWPose ONNX 模型路径")
        import sys
        echomimic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2")
        if echomimic_dir not in sys.path:
            sys.path.insert(0, echomimic_dir)
        from src.models.dwpose.dwpose_detector import DWposeDetector
        detector = DWposeDetector(
            model_det=det_path, model_pose=pose_path, device=device)

    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        pil_img = np.array(Image.open(ref_img_path).convert('RGB'))
        ref_img = pil_img[:, :, ::-1]
    H, W = ref_img.shape[:2]

    pose = detector(ref_img)
    body = pose['bodies']['candidate']  # (nums*18, 2)

    if body.shape[0] == 0:
        logger.warning("DWPose 未检测到人体, 使用默认锚点")
        hw = IMG_SIZE // 2
        return (np.array([hw, hw - 80], dtype=np.float32),
                np.array([hw - 80, hw - 120], dtype=np.float32),
                np.array([hw + 80, hw - 120], dtype=np.float32),
                160.0)

    nums = body.shape[0] // 18
    bodies_reshaped = body.reshape(nums, 18, 2)
    neck_2d = bodies_reshaped[0, 1] * np.array([W, H])
    l_sh_2d = bodies_reshaped[0, 5] * np.array([W, H])
    r_sh_2d = bodies_reshaped[0, 2] * np.array([W, H])
    sw_2d = np.linalg.norm(l_sh_2d - r_sh_2d)

    logger.info(f"2D锚点: neck=({neck_2d[0]:.0f},{neck_2d[1]:.0f}), sw_2d={sw_2d:.1f}px")
    return neck_2d, l_sh_2d, r_sh_2d, sw_2d


def export_frames(hand_2d, out_dir, img_size=IMG_SIZE):
    """
    导出 EchoMimicV2 兼容的逐帧 .npy.

    输出结构:
      out_dir/
        all_frames_hand_kpts.npy    ← 合并文件 (T,42,2)
        single_frames/              ← 逐帧文件
          0.npy, 1.npy, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "single_frames")
    os.makedirs(frames_dir, exist_ok=True)

    T = hand_2d.shape[0]

    for t in range(T):
        kpts_norm = hand_2d[t] / img_size
        hands_arr = np.stack([kpts_norm[:21], kpts_norm[21:42]], axis=0)

        frame = {
            'bodies': {
                'candidate': np.zeros((1, 18, 2), dtype=np.float32),
                'score': np.zeros((1, 18), dtype=np.float32),
            },
            'hands': hands_arr,
            'hands_score': np.ones((2, 21), dtype=np.float32),
            'faces': np.zeros((1, 70, 2), dtype=np.float32),
            'faces_score': np.zeros((1, 70), dtype=np.float32),
            'draw_pose_params': [
                img_size, img_size,
                0, img_size, 0, img_size,
            ],
        }
        np.save(os.path.join(frames_dir, f"{t}.npy"), frame)

    # 合并文件 (调试用)
    np.save(os.path.join(out_dir, "all_frames_hand_kpts.npy"), hand_2d)

    logger.info(f"导出: {T} 帧 → {frames_dir}/")
    return frames_dir


# ============================================================
# Step 5: 可视化 (调试)
# ============================================================
def visualize(hand_2d, ref_img_path, out_dir, img_size=IMG_SIZE):
    """将手部关键点叠加到参考图上保存中间帧"""
    os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            raise ValueError("cv2.imread 返回 None")
        ref_img = cv2.resize(ref_img, (img_size, img_size))
        use_cv2 = True
    except Exception:
        from PIL import Image
        ref_img = np.array(Image.open(ref_img_path).resize((img_size, img_size)).convert('RGB'))
        ref_img = ref_img[:, :, ::-1]  # RGB → BGR (模拟cv2格式)
        use_cv2 = False

    T = hand_2d.shape[0]
    mid_t = T // 2
    canvas = ref_img.copy()

    for (x, y) in hand_2d[mid_t]:
        pt = (int(round(x)), int(round(y)))
        if use_cv2:
            import cv2
            cv2.circle(canvas, pt, radius=2, color=(0, 255, 0), thickness=-1)
        else:
            x0, y0 = max(0, pt[0]-2), max(0, pt[1]-2)
            x1, y1 = min(img_size, pt[0]+3), min(img_size, pt[1]+3)
            canvas[y0:y1, x0:x1] = [0, 255, 0]  # BGR green

    vis_path = os.path.join(out_dir, "hand_vis_mid.png")
    if use_cv2:
        import cv2
        cv2.imwrite(vis_path, canvas)
    else:
        from PIL import Image
        Image.fromarray(canvas[:, :, ::-1]).save(vis_path)  # BGR→RGB
    logger.info(f"可视化: {vis_path}")


# ============================================================
# 主管线
# ============================================================
def process(input_npy, ref_img, output_dir, no_vis=False,
            dwpose_det=None, dwpose_pose=None, device="cpu"):
    """
    主处理函数.

    Args:
        input_npy: HandMDM 输出的 (T,274) .npy 路径
        ref_img: 参考人像路径
        output_dir: 输出目录
        no_vis: 跳过可视化
        dwpose_det: DWPose 检测模型路径
        dwpose_pose: DWPose 姿态模型路径
    """
    # ── 加载 ──
    logger.info(f"加载: {input_npy}")
    motion = np.load(input_npy)
    if motion.ndim != 2 or motion.shape[1] not in (274, 284):
        raise ValueError(
            f"输入 .npy shape={motion.shape}, 期望 (T, 274) 或 (T, 284)")
    if motion.shape[1] == 284:
        motion = motion[:, :274]  # 截断为274
    logger.info(f"输入运动: {motion.shape[0]}帧 × {motion.shape[1]}维")

    # ── 帧率转换 ──
    motion_24 = convert_fps(motion)

    # ── 2D锚点提取 ──
    logger.info("DWPose 2D锚点提取...")
    try:
        neck_2d, l_sh_2d, r_sh_2d, sw_2d = extract_2d_anchors_from_ref(
            ref_img, dwpose_det=dwpose_det, dwpose_pose=dwpose_pose, device=device)
    except Exception as e:
        logger.warning(f"2D锚点提取失败: {e}, 使用默认值")
        hw = IMG_SIZE // 2
        neck_2d = np.array([hw, hw - 80], dtype=np.float32)
        l_sh_2d = np.array([hw - 80, hw - 120], dtype=np.float32)
        r_sh_2d = np.array([hw + 80, hw - 120], dtype=np.float32)
        sw_2d = 160.0

    # ── 3D提取 ──
    lh, rh = extract_3d_hands(motion_24)

    # ── 3D→2D ──
    hand_2d = project_to_2d(lh, rh, neck_2d, l_sh_2d, r_sh_2d, sw_2d)

    # ── 导出 ──
    frames_dir = export_frames(hand_2d, output_dir)

    # ── 可视化 ──
    if not no_vis:
        visualize(hand_2d, ref_img, output_dir)

    # ── 汇总 ──
    logger.info("=" * 50)
    logger.info(f"完成!")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  逐帧文件: {frames_dir}/ (0.npy ~ {hand_2d.shape[0]-1}.npy)")
    logger.info(f"  帧数: {hand_2d.shape[0]} @{FPS_ECHOMIMIC}fps")
    logger.info(f"  时长: {hand_2d.shape[0]/FPS_ECHOMIMIC:.2f}s")
    logger.info(f"  EchoMimicV2 用法: --pose_dir {frames_dir}")
    return frames_dir


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="HandMDM .npy → EchoMimicV2 姿态预处理")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="HandMDM 输出的 (T,274) .npy 文件")
    parser.add_argument("--ref_img", "-r", type=str, required=True,
                        help="参考人像图片")
    parser.add_argument("--output", "-o", type=str, default="./pose_output",
                        help="输出目录 (默认 ./pose_output)")
    parser.add_argument("--no_vis", action="store_true",
                        help="跳过可视化叠加图")
    parser.add_argument("--dwpose_det", type=str, default=None,
                        help="DWPose 检测模型路径 (yolox_l.onnx)")
    parser.add_argument("--dwpose_pose", type=str, default=None,
                        help="DWPose 姿态模型路径 (dw-ll_ucoco_384.onnx)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备: cpu | cuda")
    args = parser.parse_args()

    process(args.input, args.ref_img, args.output, no_vis=args.no_vis,
            dwpose_det=args.dwpose_det, dwpose_pose=args.dwpose_pose,
            device=args.device)


if __name__ == "__main__":
    main()
