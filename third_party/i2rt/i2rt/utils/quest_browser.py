"""Shared Quest Browser launch helpers for local Vuer teleop."""

from __future__ import annotations

import argparse
import os
import subprocess
import time


DEFAULT_VUER_URL = "https://vuer.ai?ws=wss://localhost:8012"


def _adb_base_args(serial: str | None) -> list[str]:
    base = ["adb"]
    if serial:
        base.extend(["-s", serial])
    return base


def _run_adb(args: list[str], label: str, timeout: float = 5.0) -> bool:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"{label}=failed reason={type(exc).__name__}: {exc}")
        return False
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        print(f"{label}=failed returncode={result.returncode} message={message}")
        return False
    print(f"{label}=ok")
    return True


def launch_quest_vuer_browser(
    serial: str | None = None,
    url: str = DEFAULT_VUER_URL,
    force_new: bool = True,
) -> None:
    """Best-effort Quest Browser refresh for local Vuer teleop."""
    serial = serial or os.environ.get("QUEST_ADB_SERIAL")
    adb = _adb_base_args(serial)
    serial_label = serial if serial else "default"
    print(f"quest_browser_launch=starting serial={serial_label} url={url}")
    _run_adb(adb + ["reverse", "tcp:8012", "tcp:8012"], "quest_adb_reverse")
    if force_new:
        _run_adb(adb + ["shell", "am", "force-stop", "com.oculus.browser"], "quest_browser_force_stop")
    _run_adb(
        adb
        + [
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            url,
        ],
        "quest_browser_start",
    )


def add_quest_browser_args(parser: argparse.ArgumentParser) -> None:
    """Add the common CRAFT-style Quest Browser launch flags."""
    parser.add_argument(
        "--quest-adb-serial",
        default=None,
        help="ADB serial/IP to target for the standard Quest Browser launch.",
    )
    parser.add_argument("--quest-browser-delay", type=float, default=1.0)


def launch_standard_quest_browser(
    args: argparse.Namespace,
    *,
    ngrok: bool | None = None,
    url: str = DEFAULT_VUER_URL,
) -> None:
    """Launch the standard local Quest Browser flow unless Vuer uses ngrok."""
    use_ngrok = bool(getattr(args, "ngrok", False) if ngrok is None else ngrok)
    if use_ngrok:
        print("quest_browser_launch=skipped reason=ngrok")
        return
    time.sleep(max(0.0, float(getattr(args, "quest_browser_delay", 1.0))))
    launch_quest_vuer_browser(
        serial=getattr(args, "quest_adb_serial", None),
        url=url,
        force_new=True,
    )
