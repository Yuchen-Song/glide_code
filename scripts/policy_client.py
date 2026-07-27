#!/usr/bin/env python3
"""Run and record a gripper policy for the marker or tomato-plate task."""

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
    policy_connection_args,
)


RUNTIME_DIR = REPO_ROOT / "glide_runtime"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_recording_args(parser)
    parser.add_argument("--guarded", action="store_true", help="Enable GLIDE policy guardrails.")
    parser.add_argument(
        "--task-profile",
        choices=("marker", "plate"),
        help="Override automatic task inference for unusual task wording.",
    )
    args, advanced = parser.parse_known_args()
    task_profile = infer_gripper_task(args.task, args.task_profile, parser)

    runtime = (
        RUNTIME_DIR / f"policy_gripper_{task_profile}_guarded.py"
        if args.guarded
        else RUNTIME_DIR / "policy_gripper.py"
    )
    runtime_args = [
        "--repo-id",
        args.repo_id,
        "--num-episodes",
        str(args.num_episodes),
        "--task",
        args.task,
        "--action-horizon",
        "10",
        "--image-writer-threads",
        "12",
        "--video-encode-workers",
        "3",
        "--fix-realsense-controls",
        "--gripper-force-threshold",
        "0",
    ]
    runtime_args.extend(
        policy_connection_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(
        camera_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(advanced)
    execute(runtime, runtime_args, print_command=args.print_command)


if __name__ == "__main__":
    main()
