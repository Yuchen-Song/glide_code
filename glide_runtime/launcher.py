"""Shared command construction for the four public experiment launchers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

CAMERA_SETTINGS = (
    ("--head-serial", "GLIDE_HEAD_CAMERA_SERIAL", "<head-camera-serial>"),
    ("--left-wrist-serial", "GLIDE_LEFT_WRIST_CAMERA_SERIAL", "<left-wrist-camera-serial>"),
    ("--right-wrist-serial", "GLIDE_RIGHT_WRIST_CAMERA_SERIAL", "<right-wrist-camera-serial>"),
)


def add_recording_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repository ID.")
    parser.add_argument("--num-episodes", required=True, type=int, help="Number of episodes to collect.")
    parser.add_argument("--task", required=True, help="Natural-language task instruction.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved internal command without importing hardware dependencies.",
    )


def option_is_present(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def infer_gripper_task(task: str, explicit_profile: str | None, parser: argparse.ArgumentParser) -> str:
    if explicit_profile is not None:
        return explicit_profile
    normalized = task.casefold()
    if any(token in normalized for token in ("marker", "handover", "hand over")):
        return "marker"
    if any(token in normalized for token in ("plate", "tomato")):
        return "plate"
    parser.error(
        "could not infer marker or plate behavior from --task; "
        "pass --task-profile marker or --task-profile plate"
    )
    raise AssertionError("argparse.error exits")


def camera_args(
    passthrough: Sequence[str],
    *,
    print_command: bool,
    parser: argparse.ArgumentParser,
) -> list[str]:
    result: list[str] = []
    missing: list[str] = []
    for option, environment_name, placeholder in CAMERA_SETTINGS:
        if option_is_present(passthrough, option):
            continue
        if print_command:
            result.extend((option, placeholder))
        else:
            value = os.environ.get(environment_name)
            if value:
                result.extend((option, value))
            else:
                missing.append(environment_name)
    if missing:
        parser.error(
            "camera serials are not configured; set "
            + ", ".join(missing)
            + " (see .env.example), or pass the corresponding advanced serial flags"
        )
    return result


def policy_connection_args(
    passthrough: Sequence[str],
    *,
    print_command: bool,
    parser: argparse.ArgumentParser,
) -> list[str]:
    result: list[str] = []
    if not option_is_present(passthrough, "--host"):
        host = os.environ.get("GLIDE_POLICY_HOST")
        if not host and not print_command:
            parser.error("set GLIDE_POLICY_HOST (see .env.example), or pass the advanced --host flag")
        result.extend(("--host", "<policy-server-host>" if print_command else host))
    if not option_is_present(passthrough, "--port"):
        result.extend(("--port", os.environ.get("GLIDE_POLICY_PORT", "8000")))
    if not option_is_present(passthrough, "--api-key"):
        api_key = os.environ.get("GLIDE_POLICY_API_KEY")
        if api_key and not print_command:
            result.extend(("--api-key", api_key))
    return result


def execute(
    script: Path,
    arguments: Sequence[str],
    *,
    print_command: bool,
) -> None:
    relative_script = script.relative_to(REPO_ROOT)
    display_command = ["python", str(relative_script), *arguments]
    if print_command:
        print(shlex.join(display_command))
        return

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(REPO_ROOT), existing_pythonpath))
    )
    os.execve(sys.executable, [sys.executable, str(script), *arguments], environment)
