#!/usr/bin/env python3
"""Orchestrate the HandMDM -> pose bridge -> EchoMimic research pipeline.

This repository contains only the integration layer. HandMDM, EchoMimic, model
weights, and SMPL/DWPose assets remain external dependencies.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(command: list[str], dry_run: bool) -> None:
    print("$", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def require(path: Path, label: str, dry_run: bool) -> None:
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HandMDM motion and convert it for EchoMimic pose conditioning."
    )
    parser.add_argument("--text", required=True, help="Natural-language hand-motion prompt")
    parser.add_argument("--reference", required=True, type=Path, help="Reference portrait")
    parser.add_argument("--audio", required=True, type=Path, help="Driving WAV audio")
    parser.add_argument("--handmdm-root", required=True, type=Path)
    parser.add_argument("--echomimic-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--length", type=int, default=150, help="HandMDM frames at 25 fps")
    parser.add_argument("--guidance", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dwpose-det", type=Path)
    parser.add_argument("--dwpose-pose", type=Path)
    parser.add_argument(
        "--render-command",
        help=(
            "Optional EchoMimic render command template. Available placeholders: "
            "{echomimic_root}, {reference}, {audio}, {pose_dir}, {output_dir}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge = Path(__file__).with_name("prepare_pose_for_echomimic.py")
    handmdm_infer = args.handmdm_root / "infer.py"
    echomimic_infer = args.echomimic_root / "infer_audio2vid_pose.py"

    for path, label in (
        (bridge, "pose bridge"),
        (handmdm_infer, "HandMDM inference script"),
        (echomimic_infer, "EchoMimic inference script"),
        (args.reference, "reference image"),
        (args.audio, "audio file"),
        (args.checkpoint, "HandMDM checkpoint"),
    ):
        require(path, label, args.dry_run)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = args.output_dir / "handmdm_motion.npy"
    pose_output = args.output_dir / "pose_bridge"

    handmdm_command = [
        sys.executable,
        str(handmdm_infer),
        args.text,
        "--checkpoint",
        str(args.checkpoint),
        "--length",
        str(args.length),
        "--guidance",
        str(args.guidance),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output",
        str(motion_path),
    ]
    run(handmdm_command, args.dry_run)

    bridge_command = [
        sys.executable,
        str(bridge),
        "--input",
        str(motion_path),
        "--ref_img",
        str(args.reference),
        "--output",
        str(pose_output),
        "--device",
        args.device,
    ]
    if args.dwpose_det:
        bridge_command.extend(["--dwpose_det", str(args.dwpose_det)])
    if args.dwpose_pose:
        bridge_command.extend(["--dwpose_pose", str(args.dwpose_pose)])
    run(bridge_command, args.dry_run)

    pose_dir = pose_output / "single_frames"
    if args.render_command:
        values = {
            "echomimic_root": str(args.echomimic_root),
            "reference": str(args.reference),
            "audio": str(args.audio),
            "pose_dir": str(pose_dir),
            "output_dir": str(args.output_dir),
        }
        render_command = shlex.split(args.render_command.format(**values))
        run(render_command, args.dry_run)
    else:
        print("\nPose conversion complete.")
        print(f"Pose frames: {pose_dir}")
        print(
            "EchoMimic render was not started. Its local pose interface expects a "
            "different landmark serialization in some revisions; provide a verified "
            "--render-command for the exact checkout you are using."
        )


if __name__ == "__main__":
    main()
