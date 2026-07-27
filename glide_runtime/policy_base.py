#!/usr/bin/env python3
"""I2RT dual-arm policy client.

Connects to an OpenPI policy server, streams RealSense observations, and
executes predicted joint position actions on the dual-arm robot.

Emergency stop: CTRL-C
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np


def _ensure_openpi_client_import() -> None:
    try:
        import openpi_client  # noqa: F401
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        local_client = repo_root / "third_party" / "openpi" / "packages" / "openpi-client" / "src"
        if local_client.is_dir():
            sys.path.insert(0, str(local_client))
        else:
            raise


_ensure_openpi_client_import()
from openpi_client import action_chunk_broker  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


try:
    import pyrealsense2 as rs  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError(
        "pyrealsense2 is required for RealSense cameras. Install it in your robot environment."
    ) from exc


I2RT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "i2rt"
if I2RT_DIR.is_dir() and str(I2RT_DIR) not in sys.path:
    sys.path.insert(0, str(I2RT_DIR))

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.utils import GripperType  # noqa: E402


GRIPPER_FORCE_CHECK_MIN = 0.1
GRIPPER_FORCE_CHECK_MAX = 0.9


@dataclass
class RealSenseConfig:
    serial: str
    width: int
    height: int
    fps: int
    warmup_s: float = 1.0
    allow_fallback: bool = False


class RealSenseStream:
    def __init__(self, config: RealSenseConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.profile = None
        self.stop_event = Event()
        self.frame_lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: Optional[float] = None
        self.thread: Optional[Thread] = None
        self.started = False

    def _build_rs_config(self, use_defaults: bool = False) -> rs.config:
        rs_config = rs.config()
        rs.config.enable_device(rs_config, self.config.serial)
        if use_defaults:
            rs_config.enable_stream(rs.stream.color)
        else:
            rs_config.enable_stream(
                rs.stream.color,
                self.config.width,
                self.config.height,
                rs.format.rgb8,
                self.config.fps,
            )
        return rs_config

    def start(self) -> None:
        try:
            self.profile = self.pipeline.start(self._build_rs_config())
        except RuntimeError as exc:
            if not self.config.allow_fallback:
                raise RuntimeError(
                    "Failed to start RealSense stream for serial "
                    f"{self.config.serial} with {self.config.width}x{self.config.height}@{self.config.fps}."
                ) from exc
            self.profile = self.pipeline.start(self._build_rs_config(use_defaults=True))
        self.started = True
        self._warmup()
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _warmup(self) -> None:
        start = time.monotonic()
        while time.monotonic() - start < self.config.warmup_s:
            self._try_read()
            time.sleep(0.05)

    def _try_read(self, timeout_ms: int = 200) -> None:
        if not self.started:
            return
        ok, frames = self.pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
        if not ok or frames is None:
            return
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return
        frame = np.asanyarray(color_frame.get_data()).copy()
        with self.frame_lock:
            self.latest_frame = frame
            self.latest_timestamp = time.monotonic()

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            self._try_read(timeout_ms=500)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame

    def stop(self) -> None:
        if not self.started:
            return
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.pipeline.stop()
        self.started = False


def list_realsense_devices() -> list[tuple[str, str]]:
    devices = []
    ctx = rs.context()
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append((serial, name))
    return devices


def ensure_realsense_serial(serial: str, label: str) -> None:
    devices = list_realsense_devices()
    serials = [s for s, _name in devices]
    if serial not in serials:
        readable = ", ".join([f"{s} ({name})" for s, name in devices]) or "none"
        raise RuntimeError(f"{label} serial '{serial}' not found. Available devices: {readable}.")


def ensure_can_interface_ready(channel: str) -> None:
    sysfs_path = f"/sys/class/net/{channel}"
    if not os.path.exists(sysfs_path):
        raise RuntimeError(
            f"CAN interface '{channel}' not found. Check `ip link show` and pass --left-channel/--right-channel."
        )
    operstate_path = os.path.join(sysfs_path, "operstate")
    try:
        with open(operstate_path, "r", encoding="utf-8") as handle:
            state = handle.read().strip()
    except OSError:
        return
    if state == "down":
        raise RuntimeError(
            f"CAN interface '{channel}' is down. Bring it up before running (e.g., "
            f"`sudo ip link set {channel} up type can bitrate 1000000`)."
        )


def _normalize_qpos(qpos: np.ndarray, num_dofs: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=float)
    if qpos.size > num_dofs:
        qpos = qpos[:num_dofs]
    elif qpos.size < num_dofs:
        pad = num_dofs - qpos.size
        qpos = np.concatenate([qpos, np.zeros(pad, dtype=float)])
    return qpos


def _build_home_qpos(robot, gripper_index: Optional[int], gripper_limits: Optional[tuple[float, float]],
                     gripper_invert: bool) -> np.ndarray:
    home = np.zeros(robot.num_dofs(), dtype=float)
    if gripper_index is None:
        return home
    if gripper_limits is None:
        gripper_pos = float(robot.get_joint_pos()[gripper_index])
        home[gripper_index] = gripper_pos
        return home
    gripper_open = 1.0
    gripper_close = 0.0
    if gripper_invert:
        gripper_open, gripper_close = gripper_close, gripper_open
    home[gripper_index] = gripper_open
    return home


def _split_action(action: np.ndarray, left_dofs: int, right_dofs: int) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=float).reshape(-1)
    expected = left_dofs + right_dofs
    if action.size > expected:
        action = action[:expected]
    return action[:left_dofs], action[left_dofs:left_dofs + right_dofs]


def _clip_gripper_if_needed(cmd: np.ndarray, gripper_index: Optional[int], gripper_limits: Optional[tuple[float, float]]) -> np.ndarray:
    if gripper_index is None or gripper_limits is None:
        return cmd
    cmd = cmd.copy()
    cmd[gripper_index] = float(np.clip(cmd[gripper_index], 0.0, 1.0))
    return cmd


class GripperForceLimiter:
    def __init__(
        self,
        robot,
        gripper_index: Optional[int],
        gripper_invert: bool,
        force_threshold: float,
        force_ema_alpha: float,
        backoff: float,
    ) -> None:
        self.robot = robot
        self.gripper_index = gripper_index
        self.force_threshold = force_threshold
        self.force_ema_alpha = force_ema_alpha
        self.backoff = backoff
        self.gripper_blocked = False
        self.eff_ema: Optional[float] = None
        self.gripper_open = 1.0
        self.gripper_close = 0.0
        if gripper_invert:
            self.gripper_open, self.gripper_close = self.gripper_close, self.gripper_open

    def _get_current_pos(self, obs: dict) -> Optional[float]:
        if self.gripper_index is None:
            return None
        gripper_pos_obs = obs.get("gripper_pos")
        if gripper_pos_obs is not None and len(gripper_pos_obs) > 0:
            return float(gripper_pos_obs[0])
        joint_pos = obs.get("joint_pos")
        if joint_pos is not None and len(joint_pos) > self.gripper_index:
            return float(joint_pos[self.gripper_index])
        current_q = self.robot.get_joint_pos()
        if len(current_q) > self.gripper_index:
            return float(current_q[self.gripper_index])
        return None

    def apply(
        self, cmd: np.ndarray
    ) -> tuple[np.ndarray, Optional[float], Optional[float], Optional[float], bool]:
        if self.gripper_index is None or self.gripper_index >= len(cmd):
            return cmd, None, None, None, False

        obs = self.robot.get_observations()
        if obs is None:
            return cmd, None, None, None, False

        joint_eff = obs.get("joint_eff")
        if joint_eff is None or len(joint_eff) <= self.gripper_index:
            return cmd, None, None, None, False

        eff = float(joint_eff[self.gripper_index])
        if self.force_ema_alpha is not None and self.force_ema_alpha > 0.0:
            if self.eff_ema is None:
                self.eff_ema = eff
            else:
                self.eff_ema = self.force_ema_alpha * eff + (1.0 - self.force_ema_alpha) * self.eff_ema
            eff = self.eff_ema

        current_pos = self._get_current_pos(obs)
        target_pos = float(cmd[self.gripper_index])
        if current_pos is None:
            return cmd, eff, None, target_pos, False

        closing = (target_pos - current_pos) * (self.gripper_close - self.gripper_open) > 0
        if self.gripper_blocked and closing:
            cmd = cmd.copy()
            cmd[self.gripper_index] = current_pos
            return cmd, eff, current_pos, target_pos, True

        if self.gripper_blocked and not closing:
            self.gripper_blocked = False

        if not (GRIPPER_FORCE_CHECK_MIN <= current_pos <= GRIPPER_FORCE_CHECK_MAX):
            return cmd, eff, current_pos, target_pos, False

        if (
            self.force_threshold is not None
            and self.force_threshold > 0.0
            and closing
            and eff >= self.force_threshold
        ):
            already_blocked = self.gripper_blocked
            self.gripper_blocked = True
            cmd = cmd.copy()
            if not already_blocked:
                backoff_amount = self.backoff if self.backoff is not None else 0.0
                if backoff_amount > 0.0:
                    backoff_dir = np.sign(self.gripper_open - self.gripper_close)
                    backoff_pos = current_pos + backoff_dir * backoff_amount
                    lower = min(self.gripper_open, self.gripper_close)
                    upper = max(self.gripper_open, self.gripper_close)
                    cmd[self.gripper_index] = float(np.clip(backoff_pos, lower, upper))
                else:
                    cmd[self.gripper_index] = current_pos
            else:
                cmd[self.gripper_index] = current_pos
            return cmd, eff, current_pos, target_pos, True

        return cmd, eff, current_pos, target_pos, False


def main() -> None:
    parser = argparse.ArgumentParser(description="I2RT dual-arm policy client (CTRL-C to stop).")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)

    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-force-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-force-verbose", action="store_true")
    parser.add_argument("--gripper-force-print-interval", type=float, default=0.05)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=1.0)
    parser.add_argument("--gripper-backoff", type=float, default=0.15)

    parser.add_argument("--ready-qpos", type=float, nargs="+", default=[0, 1.047, 1.047, 0, 0, 0, 0, 0])
    parser.add_argument("--home-time", type=float, default=2.0)
    parser.add_argument("--no-home", action="store_true")

    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--head-serial", type=str, default="<head-camera-serial>")
    parser.add_argument("--left-wrist-serial", type=str, default="<left-wrist-camera-serial>")
    parser.add_argument("--right-wrist-serial", type=str, default="<right-wrist-camera-serial>")
    parser.add_argument("--head-width", type=int, default=640)
    parser.add_argument("--head-height", type=int, default=480)
    parser.add_argument("--head-fps", type=int, default=30)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)
    parser.add_argument("--allow-camera-fallback", action="store_true")

    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--no-resize", action="store_true")

    args = parser.parse_args()

    if args.list_cameras:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
        else:
            print("Available RealSense devices:")
            for serial, name in devices:
                print(f"  {serial}  {name}")
        return

    missing = [
        ("head", args.head_serial),
        ("left_wrist", args.left_wrist_serial),
        ("right_wrist", args.right_wrist_serial),
    ]
    missing = [label for label, serial in missing if serial is None]
    if missing:
        raise RuntimeError(f"Missing camera serials for: {', '.join(missing)}. Use --list-cameras to inspect.")

    ensure_can_interface_ready(args.left_channel)
    ensure_can_interface_ready(args.right_channel)

    ensure_realsense_serial(args.head_serial, "Head")
    ensure_realsense_serial(args.left_wrist_serial, "Left wrist")
    ensure_realsense_serial(args.right_wrist_serial, "Right wrist")

    head_cam = RealSenseStream(
        RealSenseConfig(
            serial=args.head_serial,
            width=args.head_width,
            height=args.head_height,
            fps=args.head_fps,
            allow_fallback=args.allow_camera_fallback,
        )
    )
    left_cam = RealSenseStream(
        RealSenseConfig(
            serial=args.left_wrist_serial,
            width=args.wrist_width,
            height=args.wrist_height,
            fps=args.wrist_fps,
            allow_fallback=args.allow_camera_fallback,
        )
    )
    right_cam = RealSenseStream(
        RealSenseConfig(
            serial=args.right_wrist_serial,
            width=args.wrist_width,
            height=args.wrist_height,
            fps=args.wrist_fps,
            allow_fallback=args.allow_camera_fallback,
        )
    )

    left_robot = None
    right_robot = None
    stop_event = Event()

    def _handle_sigint(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        print("Starting camera streams...")
        head_cam.start()
        left_cam.start()
        right_cam.start()

        left_gripper_type = GripperType.from_string_name(args.left_gripper)
        right_gripper_type = GripperType.from_string_name(args.right_gripper)
        left_robot = get_yam_robot(channel=args.left_channel, gripper_type=left_gripper_type)
        right_robot = get_yam_robot(channel=args.right_channel, gripper_type=right_gripper_type)

        left_info = left_robot.get_robot_info()
        right_info = right_robot.get_robot_info()
        left_gripper_index = left_info.get("gripper_index")
        right_gripper_index = right_info.get("gripper_index")
        left_gripper_limits = left_info.get("gripper_limits")
        right_gripper_limits = right_info.get("gripper_limits")

        left_home = _build_home_qpos(left_robot, left_gripper_index, left_gripper_limits, args.left_gripper_invert)
        right_home = _build_home_qpos(right_robot, right_gripper_index, right_gripper_limits, args.right_gripper_invert)

        left_limiter = GripperForceLimiter(
            left_robot,
            left_gripper_index,
            args.left_gripper_invert,
            args.gripper_force_threshold,
            args.gripper_force_ema_alpha,
            args.gripper_backoff,
        )
        right_limiter = GripperForceLimiter(
            right_robot,
            right_gripper_index,
            args.right_gripper_invert,
            args.gripper_force_threshold,
            args.gripper_force_ema_alpha,
            args.gripper_backoff,
        )

        ready_left = _normalize_qpos(args.ready_qpos, left_robot.num_dofs())
        ready_right = _normalize_qpos(args.ready_qpos, right_robot.num_dofs())

        print("Moving to ready pose...")
        left_robot.move_joints(ready_left, time_interval_s=args.home_time)
        right_robot.move_joints(ready_right, time_interval_s=args.home_time)
        last_left_cmd = ready_left.copy()
        last_right_cmd = ready_right.copy()

        policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        metadata = policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

        if args.action_horizon is None:
            action_horizon = int(metadata.get("action_horizon", 10))
        else:
            action_horizon = args.action_horizon
        if action_horizon < 1:
            action_horizon = 1

        if action_horizon > 1:
            policy = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=action_horizon)
            print(f"Using action horizon {action_horizon} (open-loop chunking).")
        else:
            print("Using action horizon 1 (closed-loop).")

        # Warm up model
        for _ in range(max(0, args.warmup_steps)):
            obs_pack = _build_observation(
                head_cam,
                left_cam,
                right_cam,
                left_robot,
                right_robot,
                args,
                resize=not args.no_resize,
            )
            if obs_pack is None:
                time.sleep(0.01)
                continue
            policy.infer(obs_pack[0])
        policy.reset()

        print("Policy loop started. Emergency stop: CTRL-C")

        step_time = 1.0 / args.frequency if args.frequency > 0 else 0.0
        last_cam_warn_time = 0.0
        last_action_warn_time = 0.0
        last_left_gripper_warn_time = 0.0
        last_right_gripper_warn_time = 0.0
        last_left_gripper_print_time = 0.0
        last_right_gripper_print_time = 0.0
        step = 0
        while not stop_event.is_set():
            step_start = time.time()
            obs_pack = _build_observation(
                head_cam,
                left_cam,
                right_cam,
                left_robot,
                right_robot,
                args,
                resize=not args.no_resize,
            )
            if obs_pack is None:
                now = time.monotonic()
                if now - last_cam_warn_time > 2.0:
                    print("Waiting for camera frames...")
                    last_cam_warn_time = now
                time.sleep(0.01)
                continue
            obs, left_state, right_state = obs_pack
            action = policy.infer(obs)
            action_vec = np.asarray(action.get("actions", []), dtype=float).reshape(-1)

            left_dofs = left_robot.num_dofs()
            right_dofs = right_robot.num_dofs()
            expected = left_dofs + right_dofs
            if action_vec.size != expected:
                # Mismatched action_dim: hold current positions for safety.
                now = time.monotonic()
                if now - last_action_warn_time > 2.0:
                    print(
                        f"Action dim mismatch (got {action_vec.size}, expected {expected}); holding position."
                    )
                    last_action_warn_time = now
                left_cmd = last_left_cmd
                right_cmd = last_right_cmd
            else:
                left_cmd, right_cmd = _split_action(action_vec, left_dofs, right_dofs)

            left_cmd = _clip_gripper_if_needed(left_cmd, left_gripper_index, left_gripper_limits)
            right_cmd = _clip_gripper_if_needed(right_cmd, right_gripper_index, right_gripper_limits)

            left_cmd, left_eff, left_gripper_pos, left_gripper_goal, left_blocked = left_limiter.apply(left_cmd)
            right_cmd, right_eff, right_gripper_pos, right_gripper_goal, right_blocked = right_limiter.apply(right_cmd)

            if args.gripper_force_verbose:
                now = time.monotonic()
                if left_eff is not None and now - last_left_gripper_print_time > args.gripper_force_print_interval:
                    left_pos_label = f"{left_gripper_pos:.4f}" if left_gripper_pos is not None else "n/a"
                    left_goal_label = f"{left_gripper_goal:.4f}" if left_gripper_goal is not None else "n/a"
                    print(
                        "Left gripper eff={:.3f} pos={} target={}".format(
                            left_eff,
                            left_pos_label,
                            left_goal_label,
                        )
                    )
                    last_left_gripper_print_time = now
                if right_eff is not None and now - last_right_gripper_print_time > args.gripper_force_print_interval:
                    right_pos_label = f"{right_gripper_pos:.4f}" if right_gripper_pos is not None else "n/a"
                    right_goal_label = f"{right_gripper_goal:.4f}" if right_gripper_goal is not None else "n/a"
                    print(
                        "Right gripper eff={:.3f} pos={} target={}".format(
                            right_eff,
                            right_pos_label,
                            right_goal_label,
                        )
                    )
                    last_right_gripper_print_time = now

            if left_blocked:
                now = time.monotonic()
                if now - last_left_gripper_warn_time > 1.0:
                    print(f"Left gripper force threshold hit ({left_eff:.2f}); holding position.")
                    last_left_gripper_warn_time = now
            if right_blocked:
                now = time.monotonic()
                if now - last_right_gripper_warn_time > 1.0:
                    print(f"Right gripper force threshold hit ({right_eff:.2f}); holding position.")
                    last_right_gripper_warn_time = now

            left_robot.command_joint_pos(left_cmd)
            right_robot.command_joint_pos(right_cmd)
            last_left_cmd = left_cmd
            last_right_cmd = right_cmd

            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break

            if step_time > 0:
                elapsed = time.time() - step_start
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if not args.no_home and left_robot is not None and right_robot is not None:
                print("Returning to home pose...")
                left_robot.move_joints(left_home, time_interval_s=args.home_time)
                right_robot.move_joints(right_home, time_interval_s=args.home_time)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if left_robot is not None:
                left_robot.close()
            if right_robot is not None:
                right_robot.close()
            head_cam.stop()
            left_cam.stop()
            right_cam.stop()
            print("Shutdown complete.")


def _build_observation(
    head_cam: RealSenseStream,
    left_cam: RealSenseStream,
    right_cam: RealSenseStream,
    left_robot,
    right_robot,
    args,
    resize: bool,
) -> Optional[tuple[dict, np.ndarray, np.ndarray]]:
    head_img = head_cam.get_latest_frame()
    left_img = left_cam.get_latest_frame()
    right_img = right_cam.get_latest_frame()

    if head_img is None or left_img is None or right_img is None:
        return None

    if resize:
        head_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(head_img, args.resize_height, args.resize_width)
        )
        left_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_img, args.resize_height, args.resize_width)
        )
        right_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(right_img, args.resize_height, args.resize_width)
        )
    else:
        head_img = image_tools.convert_to_uint8(head_img)
        left_img = image_tools.convert_to_uint8(left_img)
        right_img = image_tools.convert_to_uint8(right_img)

    left_state = left_robot.get_joint_pos()
    right_state = right_robot.get_joint_pos()
    state_vec = np.concatenate([left_state, right_state]).astype(np.float32)

    obs = {
        # Keys expected by LeRobotI2RTDualArmDataConfig repack.
        "observation.images.head": head_img,
        "observation.images.left_wrist": left_img,
        "observation.images.right_wrist": right_img,
        "observation.state": state_vec,
        # Keys expected by I2RTDualArmInputs if repack is not active.
        "head_image": head_img,
        "left_wrist_image": left_img,
        "right_wrist_image": right_img,
        "state": state_vec,
    }
    if args.prompt:
        obs["prompt"] = args.prompt
    return obs, left_state, right_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()