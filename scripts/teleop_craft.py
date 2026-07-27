#!/usr/bin/env python3
"""Collect CRAFT-hand teleoperation data for the wine-serving task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glide_runtime.launcher import add_recording_args, camera_args, execute  # noqa: E402


RUNTIME_DIR = REPO_ROOT / "third_party" / "i2rt" / "i2rt-craft-hand"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_recording_args(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--glide", action="store_true", help="Enable GLIDE wine-pour guardrails.")
    mode.add_argument("--manual", action="store_true", help="Use the manual intervention workflow.")
    args, advanced = parser.parse_known_args()

    if args.glide:
        runtime = RUNTIME_DIR / "teleop_craft_glide.py"
        force_threshold = "0.8"
    elif args.manual:
        runtime = RUNTIME_DIR / "teleop_craft_manual.py"
        force_threshold = "0.5"
    else:
        runtime = RUNTIME_DIR / "teleop_craft.py"
        force_threshold = "0.5"

    runtime_args = [
        "--repo-id",
        args.repo_id,
        "--num-episodes",
        str(args.num_episodes),
        "--task",
        args.task,
        "--arm-sides",
        "both",
        "--left-gripper",
        "linear_4310",
        "--right-gripper",
        "no_gripper",
        "--craft-hand-mode",
        "drive",
        "--gripper-force-threshold",
        force_threshold,
    ]
    if not any(
        token == "--quest-adb-serial" or token.startswith("--quest-adb-serial=")
        for token in advanced
    ):
        quest_serial = os.environ.get("GLIDE_QUEST_ADB_SERIAL")
        if quest_serial and not args.print_command:
            runtime_args.extend(("--quest-adb-serial", quest_serial))
    runtime_args.extend(
        camera_args(advanced, print_command=args.print_command, parser=parser)
    )
    runtime_args.extend(advanced)
    execute(runtime, runtime_args, print_command=args.print_command)


if __name__ == "__main__":
    main()
