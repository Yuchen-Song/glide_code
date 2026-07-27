#!/usr/bin/env python3
"""Collect gripper teleoperation data for the marker or tomato-plate task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glide_runtime.launcher import (  # noqa: E402
    add_recording_args,
    camera_args,
    execute,
    infer_gripper_task,
)


RUNTIME_DIR = REPO_ROOT / "third_party" / "i2rt" / "glide_runtime"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_recording_args(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--glide", action="store_true", help="Enable GLIDE task guardrails.")
    mode.add_argument("--manual", action="store_true", help="Use the manual intervention workflow.")
    parser.add_argument(
        "--task-profile",
        choices=("marker", "plate"),
        help="Override automatic task inference for unusual task wording.",
    )
    args, advanced = parser.parse_known_args()
    task_profile = infer_gripper_task(args.task, args.task_profile, parser)

    if args.glide:
        runtime = RUNTIME_DIR / f"teleop_gripper_{task_profile}_glide.py"
    elif args.manual and task_profile == "plate":
        runtime = RUNTIME_DIR / "teleop_gripper_plate_manual.py"
    else:
        runtime = RUNTIME_DIR / "teleop_gripper.py"

    runtime_args = [
        "--repo-id",
        args.repo_id,
        "--num-episodes",
        str(args.num_episodes),
        "--task",
        args.task,
        "--fix-realsense-controls",
        "--gripper-force-threshold",
        "0",
    ]
    if args.glide:
        runtime_args.append(f"--{task_profile}-task-filter")
    elif args.manual and task_profile == "marker":
        runtime_args.append("--lock")
    runtime_args.extend(
        camera_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(advanced)
    execute(runtime, runtime_args, print_command=args.print_command)


if __name__ == "__main__":
    main()
