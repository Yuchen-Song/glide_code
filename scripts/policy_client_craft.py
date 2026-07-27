#!/usr/bin/env python3
"""Run and record a CRAFT-hand policy for the wine-serving task."""

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
    policy_connection_args,
)


RUNTIME = REPO_ROOT / "glide_runtime" / "policy_craft.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_recording_args(parser)
    parser.add_argument("--guarded", action="store_true", help="Enable GLIDE wine-pour guardrails.")
    args, advanced = parser.parse_known_args()

    runtime_args = [
        "--repo-id",
        args.repo_id,
        "--num-episodes",
        str(args.num_episodes),
        "--task",
        args.task,
        "--action-horizon",
        "50",
        "--frequency",
        "45",
        "--arm-sides",
        "both",
        "--cameras",
        "head",
        "left_wrist",
        "right_wrist",
        "--left-gripper",
        "linear_4310",
        "--right-gripper",
        "no_gripper",
        "--image-writer-threads",
        "12",
        "--video-encode-workers",
        "3",
        "--gripper-force-threshold",
        "0",
    ]
    if args.guarded:
        runtime_args.append("--pour-task-filter")
    runtime_args.extend(
        policy_connection_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(
        camera_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(advanced)
    execute(RUNTIME, runtime_args, print_command=args.print_command)


if __name__ == "__main__":
    main()
