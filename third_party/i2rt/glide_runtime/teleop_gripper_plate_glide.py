import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import pyrealsense2 as rs

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    serialize_dict,
    write_info,
    write_jsonlines,
)
from lerobot.common.datasets.video_utils import encode_video_frames

TELEVISION_DIR = Path(__file__).resolve().parents[1] / "TeleVision"
if TELEVISION_DIR.is_dir():
    import sys

    # Allow importing TeleVision scripts without turning the folder into a package.
    if str(TELEVISION_DIR) not in sys.path:
        sys.path.insert(0, str(TELEVISION_DIR))

from teleop_dual_arm import (  # noqa: E402
    CONTROL_FREQUENCY,
    VuerControllerTeleop,
    build_command,
    compute_target_pose,
    ensure_can_interface_ready,
    maybe_limit_gripper_close,
    move_to_ready_pose,
    reset_to_home,
    setup_arm,
    sync_arm_state_from_robot,
    update_gripper_from_controller,
    vuer_to_robot_matrix,
)


@dataclass
class RealSenseConfig:
    serial: str
    width: int
    height: int
    fps: int
    warmup_s: float = 1.0
    allow_fallback: bool = False
    fixed_color_controls: bool = False
    exposure: float | None = None
    gain: float | None = None
    brightness: float | None = None
    disable_hdr: bool = False
    disable_auto_white_balance: bool = False
    white_balance: float | None = None


class RealSenseStream:
    def __init__(self, config: RealSenseConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.profile = None
        self.stop_event = Event()
        self.frame_lock = Lock()
        self.latest_frame = None
        self.latest_timestamp = None
        self.thread = None
        self.started = False
        self._saved_color_options = []

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
        self._apply_fixed_color_controls()
        self._warmup()
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _get_color_sensor(self):
        device = self.profile.get_device()
        for sensor in device.query_sensors():
            try:
                for stream_profile in sensor.get_stream_profiles():
                    if stream_profile.stream_type() == rs.stream.color:
                        return sensor
            except RuntimeError as exc:
                print(f"RealSense {self.config.serial}: failed to inspect sensor profiles: {exc}")
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.exposure) or sensor.supports(rs.option.enable_auto_exposure):
                return sensor
        return None

    def _get_sensor_name(self, sensor) -> str:
        try:
            return sensor.get_info(rs.camera_info.name)
        except RuntimeError:
            return "unknown sensor"

    def _remember_supported_option(self, sensor, option, label: str) -> None:
        if any(saved_option == option for _sensor, saved_option, _value, _label in self._saved_color_options):
            return
        try:
            self._saved_color_options.append((sensor, option, sensor.get_option(option), label))
        except RuntimeError as exc:
            print(f"RealSense {self.config.serial}: failed to remember previous {label}: {exc}")

    def _set_supported_option(self, sensor, option_name: str, value: float, label: str) -> None:
        option = getattr(rs.option, option_name, None)
        if option is None or not sensor.supports(option):
            print(f"RealSense {self.config.serial}: {label} is not supported; leaving it unchanged.")
            return

        try:
            self._remember_supported_option(sensor, option, label)
            option_range = sensor.get_option_range(option)
            clamped = min(max(float(value), option_range.min), option_range.max)
            if clamped != value:
                print(
                    f"RealSense {self.config.serial}: requested {label}={value} is outside "
                    f"[{option_range.min}, {option_range.max}], using {clamped}."
                )
            sensor.set_option(option, clamped)
            actual = sensor.get_option(option)
        except RuntimeError as exc:
            print(f"RealSense {self.config.serial}: failed to set {label}: {exc}")
            return
        print(f"RealSense {self.config.serial}: fixed {label}={actual}.")

    def _disable_supported_option(self, sensor, option_name: str, label: str) -> None:
        option = getattr(rs.option, option_name, None)
        if option is None or not sensor.supports(option):
            return
        try:
            self._remember_supported_option(sensor, option, label)
            sensor.set_option(option, 0.0)
        except RuntimeError as exc:
            print(f"RealSense {self.config.serial}: failed to disable {label}: {exc}")
            return
        print(f"RealSense {self.config.serial}: disabled {label}.")

    def _apply_fixed_color_controls(self) -> None:
        if not self.config.fixed_color_controls:
            return

        sensor = self._get_color_sensor()
        if sensor is None:
            print(f"RealSense {self.config.serial}: no color sensor found for fixed controls.")
            return

        print(f"RealSense {self.config.serial}: applying fixed controls to {self._get_sensor_name(sensor)}.")
        self._set_supported_option(sensor, "enable_auto_exposure", 0.0, "auto exposure")
        if self.config.disable_auto_white_balance:
            self._disable_supported_option(sensor, "enable_auto_white_balance", "auto white balance")
        if self.config.disable_hdr:
            self._disable_supported_option(sensor, "hdr_enabled", "HDR")
        if self.config.exposure is not None:
            self._set_supported_option(sensor, "exposure", self.config.exposure, "exposure")
        if self.config.gain is not None:
            self._set_supported_option(sensor, "gain", self.config.gain, "gain/ISO")
        if self.config.brightness is not None:
            self._set_supported_option(sensor, "brightness", self.config.brightness, "brightness")
        if self.config.white_balance is not None:
            self._set_supported_option(sensor, "white_balance", self.config.white_balance, "white balance")

    def _restore_color_controls(self) -> None:
        for sensor, option, value, label in reversed(self._saved_color_options):
            try:
                sensor.set_option(option, value)
                actual = sensor.get_option(option)
            except RuntimeError as exc:
                print(f"RealSense {self.config.serial}: failed to restore {label}: {exc}")
                continue
            print(f"RealSense {self.config.serial}: restored {label}={actual}.")
        self._saved_color_options.clear()

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

    def get_latest_frame(self) -> np.ndarray | None:
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
        self._restore_color_controls()
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


def fit_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame is None:
        raise ValueError("Empty frame received from RealSense.")
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame

    out = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    src_h = min(target_h, frame.shape[0])
    src_w = min(target_w, frame.shape[1])
    src_y = max((frame.shape[0] - src_h) // 2, 0)
    src_x = max((frame.shape[1] - src_w) // 2, 0)
    dst_y = max((target_h - src_h) // 2, 0)
    dst_x = max((target_w - src_w) // 2, 0)
    out[dst_y : dst_y + src_h, dst_x : dst_x + src_w] = frame[src_y : src_y + src_h, src_x : src_x + src_w]
    return out


class VuerControllerTeleopCam(VuerControllerTeleop):
    def update_image(self, frame: np.ndarray | None) -> bool:
        if frame is None:
            return False
        frame = fit_frame(frame, self.resolution[0], self.resolution[1])
        stereo = np.concatenate((frame, frame), axis=1)
        np.copyto(self.img_array, stereo)
        return True


def build_image_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


TELEOP_BUTTON_NAMES = [
    "left_trigger",
    "left_squeeze",
    "left_aButton",
    "left_bButton",
    "right_trigger",
    "right_squeeze",
    "right_aButton",
    "right_bButton",
]


def build_matrix_feature() -> dict:
    return {"dtype": "float32", "shape": (4, 4), "names": ["row", "column"]}


def get_teleop_matrix(teleop: VuerControllerTeleop | None, side: str) -> np.ndarray:
    if teleop is None:
        return np.zeros((4, 4), dtype=np.float32)
    mat = teleop.get_controller_matrix(side)
    if mat is None:
        return np.zeros((4, 4), dtype=np.float32)
    return np.asarray(mat, dtype=np.float32)


def get_teleop_head_matrix(teleop: VuerControllerTeleop | None) -> np.ndarray:
    tv = getattr(teleop, "tv", None) if teleop is not None else None
    if tv is None or not hasattr(tv, "head_matrix"):
        return np.zeros((4, 4), dtype=np.float32)
    mat = tv.head_matrix
    if mat is None:
        return np.zeros((4, 4), dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        return np.zeros((4, 4), dtype=np.float32)
    return mat


def get_teleop_buttons(
    left_state: dict | None,
    right_state: dict | None,
) -> np.ndarray:
    def button_value(state: dict | None, name: str) -> float:
        if state is None:
            return 0.0
        return float(bool(state.get(name, False)))

    return np.asarray(
        [
            button_value(left_state, "trigger"),
            button_value(left_state, "squeeze"),
            button_value(left_state, "aButton"),
            button_value(left_state, "bButton"),
            button_value(right_state, "trigger"),
            button_value(right_state, "squeeze"),
            button_value(right_state, "aButton"),
            button_value(right_state, "bButton"),
        ],
        dtype=np.float32,
    )


def build_joint_names(prefix: str, total_dofs: int, gripper_index: int | None) -> list[str]:
    names = []
    for idx in range(total_dofs):
        if gripper_index is not None and idx == gripper_index:
            names.append(f"{prefix}_gripper")
        else:
            names.append(f"{prefix}_joint_{idx}")
    return names


def move_to_ready_pose_open_gripper(arm_state: dict, ready_qpos: np.ndarray, time_interval_s: float) -> None:
    robot = arm_state["robot"]
    ready_cmd = np.asarray(ready_qpos, dtype=float).copy()
    if len(ready_cmd) != robot.num_dofs():
        if len(ready_cmd) > robot.num_dofs():
            ready_cmd = ready_cmd[: robot.num_dofs()]
        else:
            pad = robot.num_dofs() - len(ready_cmd)
            ready_cmd = np.concatenate([ready_cmd, np.zeros(pad)])
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is not None and gripper_index < len(ready_cmd):
        ready_cmd[gripper_index] = float(arm_state.get("gripper_open", ready_cmd[gripper_index]))
    arm_state.pop("plate_gripper_filtered_pos", None)
    arm_state.pop("plate_gripper_desired_pos", None)
    move_to_ready_pose(arm_state, ready_cmd, time_interval_s)


def maybe_save_episode(dataset: LeRobotDataset) -> bool:
    if dataset.episode_buffer is None:
        return False
    if dataset.episode_buffer.get("size", 0) <= 0:
        return False
    dataset.save_episode()
    return True


def has_episode_frames(dataset: LeRobotDataset | None) -> bool:
    if dataset is None or dataset.episode_buffer is None:
        return False
    return dataset.episode_buffer.get("size", 0) > 0


def is_squeeze_pressed(controller_state: dict | None, squeeze_threshold: float) -> bool:
    if controller_state is None:
        return False
    squeeze_value = float(controller_state.get("squeezeValue", 0.0))
    return bool(controller_state.get("squeeze")) or squeeze_value >= squeeze_threshold


def get_handle_squeeze_states(
    left_state: dict | None, right_state: dict | None, squeeze_threshold: float
) -> tuple[bool, bool]:
    return (
        is_squeeze_pressed(left_state, squeeze_threshold),
        is_squeeze_pressed(right_state, squeeze_threshold),
    )


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def clamp_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= max_norm:
        return vec
    return vec * (max_norm / max(norm, 1e-9))


def project_rotation(rot: np.ndarray) -> np.ndarray:
    u, _s, vh = np.linalg.svd(rot)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


@dataclass
class PlateLiftFilterConfig:
    enabled: bool
    pos_alpha: float
    max_speed: float
    carry_max_speed: float
    carry_max_accel: float
    carry_max_xy_speed: float
    carry_max_z_speed: float
    carry_max_xy_accel: float
    carry_max_z_accel: float
    differential_gain: float
    carry_differential_gain: float
    max_height_diff: float
    carry_max_height_diff: float
    max_separation_delta: float
    carry_max_separation_delta: float
    carry_max_compression: float
    down_margin: float
    carry_down_margin: float
    orientation_weight: float
    carry_orientation_weight: float


class PlateLiftTeleopFilter:
    """Task-space filter for carrying a shallow, spill-prone object with two arms."""

    def __init__(self, config: PlateLiftFilterConfig, frequency: float):
        self.config = config
        self.dt = 1.0 / max(float(frequency), 1e-6)
        self.prev_left_pos = None
        self.prev_right_pos = None
        self.prev_left_step = None
        self.prev_right_step = None
        self.prev_mid_step = None
        self.carry_active = False
        self.carry_left_pos = None
        self.carry_right_pos = None
        self.carry_left_rot = None
        self.carry_right_rot = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PlateLiftTeleopFilter":
        config = PlateLiftFilterConfig(
            enabled=bool(args.plate_task_filter),
            pos_alpha=float(np.clip(args.plate_filter_alpha, 0.0, 1.0)),
            max_speed=max(float(args.plate_max_ee_speed), 0.0),
            carry_max_speed=max(float(args.plate_carry_max_ee_speed), 0.0),
            carry_max_accel=max(float(args.plate_carry_max_ee_accel), 0.0),
            carry_max_xy_speed=max(float(args.plate_carry_max_xy_speed), 0.0),
            carry_max_z_speed=max(float(args.plate_carry_max_z_speed), 0.0),
            carry_max_xy_accel=max(float(args.plate_carry_max_xy_accel), 0.0),
            carry_max_z_accel=max(float(args.plate_carry_max_z_accel), 0.0),
            differential_gain=float(np.clip(args.plate_differential_gain, 0.0, 1.0)),
            carry_differential_gain=float(np.clip(args.plate_carry_differential_gain, 0.0, 1.0)),
            max_height_diff=max(float(args.plate_max_height_diff), 0.0),
            carry_max_height_diff=max(float(args.plate_carry_max_height_diff), 0.0),
            max_separation_delta=max(float(args.plate_max_separation_delta), 0.0),
            carry_max_separation_delta=max(float(args.plate_carry_max_separation_delta), 0.0),
            carry_max_compression=max(float(args.plate_carry_max_compression), 0.0),
            down_margin=max(float(args.plate_down_margin), 0.0),
            carry_down_margin=max(float(args.plate_carry_down_margin), 0.0),
            orientation_weight=float(np.clip(args.plate_orientation_weight, 0.0, 1.0)),
            carry_orientation_weight=float(np.clip(args.plate_carry_orientation_weight, 0.0, 1.0)),
        )
        return cls(config, args.frequency)

    def reset(self) -> None:
        self.prev_left_pos = None
        self.prev_right_pos = None
        self.prev_left_step = None
        self.prev_right_step = None
        self.prev_mid_step = None
        self.carry_active = False
        self.carry_left_pos = None
        self.carry_right_pos = None
        self.carry_left_rot = None
        self.carry_right_rot = None

    def _filter_position(
        self,
        raw_pos: np.ndarray,
        prev_pos: np.ndarray | None,
        max_speed: float | None = None,
        prev_step: np.ndarray | None = None,
        max_accel: float | None = None,
    ) -> np.ndarray:
        if prev_pos is None:
            return raw_pos
        alpha = self.config.pos_alpha
        filtered = prev_pos + alpha * (raw_pos - prev_pos)
        speed = self.config.max_speed if max_speed is None else max_speed
        max_step = speed * self.dt
        if max_step > 0.0:
            step = filtered - prev_pos
            filtered = prev_pos + clamp_vector_norm(step, max_step)
        if prev_step is not None and max_accel is not None and max_accel > 0.0:
            step = filtered - prev_pos
            max_step_delta = max_accel * self.dt * self.dt
            step = prev_step + clamp_vector_norm(step - prev_step, max_step_delta)
            if max_step > 0.0:
                step = clamp_vector_norm(step, max_step)
            filtered = prev_pos + step
        return filtered

    def _filter_carry_midpoint(self, raw_mid: np.ndarray, prev_mid: np.ndarray) -> np.ndarray:
        filtered = prev_mid + self.config.pos_alpha * (raw_mid - prev_mid)
        step = filtered - prev_mid

        max_xy_step = self.config.carry_max_xy_speed * self.dt
        if max_xy_step > 0.0:
            step[:2] = clamp_vector_norm(step[:2], max_xy_step)

        max_z_step = self.config.carry_max_z_speed * self.dt
        if max_z_step > 0.0:
            step[2] = float(np.clip(step[2], -max_z_step, max_z_step))

        if self.prev_mid_step is not None:
            step_delta = step - self.prev_mid_step
            max_xy_delta = self.config.carry_max_xy_accel * self.dt * self.dt
            if max_xy_delta > 0.0:
                step_delta[:2] = clamp_vector_norm(step_delta[:2], max_xy_delta)
            max_z_delta = self.config.carry_max_z_accel * self.dt * self.dt
            if max_z_delta > 0.0:
                step_delta[2] = float(np.clip(step_delta[2], -max_z_delta, max_z_delta))
            step = self.prev_mid_step + step_delta
            if max_xy_step > 0.0:
                step[:2] = clamp_vector_norm(step[:2], max_xy_step)
            if max_z_step > 0.0:
                step[2] = float(np.clip(step[2], -max_z_step, max_z_step))

        self.prev_mid_step = step
        return prev_mid + step

    def _filter_rotation(self, init_rot: np.ndarray, target_rot: np.ndarray, weight: float | None = None) -> np.ndarray:
        weight = self.config.orientation_weight if weight is None else weight
        if weight <= 0.0:
            return target_rot
        if weight >= 1.0:
            return init_rot
        return project_rotation((1.0 - weight) * target_rot + weight * init_rot)

    def filter_single(
        self,
        side: str,
        target_pose: np.ndarray,
        init_pose: np.ndarray,
    ) -> np.ndarray:
        if not self.config.enabled:
            return target_pose
        filtered = target_pose.copy()
        raw_pos = target_pose[:3, 3].copy()
        raw_pos[2] = max(raw_pos[2], float(init_pose[2, 3]) - self.config.down_margin)
        if side == "left":
            prev_pos = self.prev_left_pos if self.prev_left_pos is not None else init_pose[:3, 3].copy()
            filtered_pos = self._filter_position(raw_pos, prev_pos)
            self.prev_left_pos = filtered_pos
        else:
            prev_pos = self.prev_right_pos if self.prev_right_pos is not None else init_pose[:3, 3].copy()
            filtered_pos = self._filter_position(raw_pos, prev_pos)
            self.prev_right_pos = filtered_pos
        filtered[:3, 3] = filtered_pos
        filtered[:3, :3] = self._filter_rotation(init_pose[:3, :3], target_pose[:3, :3])
        return filtered

    def _deactivate_carry(self) -> None:
        self.carry_active = False
        self.carry_left_pos = None
        self.carry_right_pos = None
        self.carry_left_rot = None
        self.carry_right_rot = None
        self.prev_left_step = None
        self.prev_right_step = None
        self.prev_mid_step = None

    def _ensure_carry_reference(
        self,
        left_target_pose: np.ndarray,
        right_target_pose: np.ndarray,
    ) -> None:
        if self.carry_active:
            return
        self.carry_left_pos = (
            self.prev_left_pos.copy() if self.prev_left_pos is not None else left_target_pose[:3, 3].copy()
        )
        self.carry_right_pos = (
            self.prev_right_pos.copy() if self.prev_right_pos is not None else right_target_pose[:3, 3].copy()
        )
        self.carry_left_rot = left_target_pose[:3, :3].copy()
        self.carry_right_rot = right_target_pose[:3, :3].copy()
        self.carry_active = True

    def filter_pair(
        self,
        left_target_pose: np.ndarray,
        right_target_pose: np.ndarray,
        left_init_pose: np.ndarray,
        right_init_pose: np.ndarray,
        stabilize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.enabled:
            return left_target_pose, right_target_pose
        if not stabilize:
            self._deactivate_carry()
            return (
                self.filter_single("left", left_target_pose, left_init_pose),
                self.filter_single("right", right_target_pose, right_init_pose),
            )

        self._ensure_carry_reference(left_target_pose, right_target_pose)

        left_filtered = left_target_pose.copy()
        right_filtered = right_target_pose.copy()

        left_init_pos = self.carry_left_pos
        right_init_pos = self.carry_right_pos
        init_mid = 0.5 * (left_init_pos + right_init_pos)
        init_half_sep = 0.5 * (left_init_pos - right_init_pos)

        raw_left_pos = left_target_pose[:3, 3]
        raw_right_pos = right_target_pose[:3, 3]
        raw_mid = 0.5 * (raw_left_pos + raw_right_pos)
        raw_half_sep = 0.5 * (raw_left_pos - raw_right_pos)

        raw_mid[2] = max(raw_mid[2], float(init_mid[2]) - self.config.carry_down_margin)

        half_delta = self.config.carry_differential_gain * (raw_half_sep - init_half_sep)
        half_delta[:2] = clamp_vector_norm(half_delta[:2], 0.5 * self.config.carry_max_separation_delta)
        half_delta[2] = float(
            np.clip(
                half_delta[2],
                -0.5 * self.config.carry_max_height_diff,
                0.5 * self.config.carry_max_height_diff,
            )
        )
        filtered_half_sep = init_half_sep + half_delta
        init_xy_norm = float(np.linalg.norm(init_half_sep[:2]))
        filtered_xy_norm = float(np.linalg.norm(filtered_half_sep[:2]))
        min_xy_norm = max(init_xy_norm - 0.5 * self.config.carry_max_compression, 0.0)
        if filtered_xy_norm < min_xy_norm and min_xy_norm > 0.0:
            if filtered_xy_norm > 1e-9:
                filtered_half_sep[:2] *= min_xy_norm / filtered_xy_norm
            else:
                filtered_half_sep[:2] = init_half_sep[:2] * (min_xy_norm / max(init_xy_norm, 1e-9))
        filtered_half_sep[2] = float(
            np.clip(
                filtered_half_sep[2],
                -0.5 * self.config.carry_max_height_diff,
                0.5 * self.config.carry_max_height_diff,
            )
        )

        left_pos = raw_mid + filtered_half_sep
        right_pos = raw_mid - filtered_half_sep
        left_prev = self.prev_left_pos if self.prev_left_pos is not None else left_init_pos.copy()
        right_prev = self.prev_right_pos if self.prev_right_pos is not None else right_init_pos.copy()
        prev_mid = 0.5 * (left_prev + right_prev)
        filtered_mid = self._filter_carry_midpoint(raw_mid, prev_mid)
        left_pos = filtered_mid + filtered_half_sep
        right_pos = filtered_mid - filtered_half_sep
        max_ee_step = self.config.carry_max_speed * self.dt
        if max_ee_step > 0.0:
            left_pos = left_prev + clamp_vector_norm(left_pos - left_prev, max_ee_step)
            right_pos = right_prev + clamp_vector_norm(right_pos - right_prev, max_ee_step)
            self.prev_mid_step = 0.5 * (left_pos + right_pos) - prev_mid

        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos
        self.prev_left_step = left_pos - left_prev
        self.prev_right_step = right_pos - right_prev

        left_filtered[:3, 3] = left_pos
        right_filtered[:3, 3] = right_pos
        left_filtered[:3, :3] = self._filter_rotation(
            self.carry_left_rot,
            left_target_pose[:3, :3],
            self.config.carry_orientation_weight,
        )
        right_filtered[:3, :3] = self._filter_rotation(
            self.carry_right_rot,
            right_target_pose[:3, :3],
            self.config.carry_orientation_weight,
        )
        return left_filtered, right_filtered


def filter_controller_rotation(
    init_controller: np.ndarray,
    current_controller: np.ndarray,
    mode: str,
) -> np.ndarray:
    filtered = current_controller.copy()
    if mode == "none":
        return filtered
    if mode == "locked":
        filtered[:3, :3] = init_controller[:3, :3]
        return filtered
    if mode == "yaw":
        delta_rot = current_controller[:3, :3] @ init_controller[:3, :3].T
        yaw = float(np.arctan2(delta_rot[1, 0], delta_rot[0, 0]))
        filtered[:3, :3] = yaw_rotation_matrix(yaw) @ init_controller[:3, :3]
        return filtered
    raise ValueError(f"Unknown rotation lock mode: {mode}")


def get_rotation_lock_mode(
    left_state: dict | None,
    right_state: dict | None,
    squeeze_threshold: float,
) -> str:
    left_squeeze_active, right_squeeze_active = get_handle_squeeze_states(
        left_state, right_state, squeeze_threshold
    )
    if left_squeeze_active:
        return "locked"
    if right_squeeze_active:
        return "yaw"
    return "none"


def is_trigger_pressed(controller_state: dict | None, trigger_threshold: float) -> bool:
    if controller_state is None:
        return False
    return float(controller_state.get("triggerValue", 0.0)) >= trigger_threshold


def get_measured_gripper_pos(arm_state: dict) -> float | None:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return None
    robot = arm_state["robot"]
    obs = robot.get_observations()
    if obs is not None:
        gripper_pos_obs = obs.get("gripper_pos")
        if gripper_pos_obs is not None and len(gripper_pos_obs) > 0:
            return float(gripper_pos_obs[0])
    current_q = robot.get_joint_pos()
    if len(current_q) > gripper_index:
        return float(current_q[gripper_index])
    return None


def toggle_gripper_from_trigger(arm_state: dict, reopen_close_fraction: float = 0.5) -> str | None:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return None

    current_pos = get_measured_gripper_pos(arm_state)
    if current_pos is None:
        current_pos = arm_state.get("plate_gripper_desired_pos")
    if current_pos is None:
        current_pos = float(arm_state.get("gripper_pos", arm_state["gripper_open"]))

    open_pos = float(arm_state["gripper_open"])
    close_pos = float(arm_state["gripper_close"])
    span = close_pos - open_pos
    if abs(span) < 1e-9:
        is_open = abs(current_pos - open_pos) <= abs(current_pos - close_pos)
    else:
        close_fraction = float(np.clip((float(current_pos) - open_pos) / span, 0.0, 1.0))
        is_open = close_fraction < float(np.clip(reopen_close_fraction, 0.0, 1.0))
    target_label = "close" if is_open else "open"
    target_pos = close_pos if is_open else open_pos

    arm_state["gripper_goal"] = target_pos
    arm_state["gripper_pos"] = target_pos
    if target_label == "open":
        arm_state["gripper_blocked"] = False
    return target_label


def get_gripper_close_fraction(arm_state: dict) -> float:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return 0.0
    open_pos = float(arm_state["gripper_open"])
    close_pos = float(arm_state["gripper_close"])
    current_pos = float(arm_state.get("gripper_pos", arm_state.get("gripper_goal", open_pos)))
    span = close_pos - open_pos
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((current_pos - open_pos) / span, 0.0, 1.0))


def set_gripper_target_label(arm_state: dict, target_label: str) -> bool:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return False
    target_pos = float(arm_state["gripper_close"] if target_label == "close" else arm_state["gripper_open"])
    arm_state["gripper_goal"] = target_pos
    arm_state["gripper_pos"] = target_pos
    arm_state["plate_gripper_desired_pos"] = target_pos
    if target_label == "close":
        arm_state["plate_gripper_last_close_time"] = time.monotonic()
    else:
        arm_state["gripper_blocked"] = False
    return True


def toggle_plate_grippers_from_triggers(
    left_arm: dict,
    right_arm: dict,
    args: argparse.Namespace,
) -> str | None:
    left_closed = get_gripper_close_fraction(left_arm)
    right_closed = get_gripper_close_fraction(right_arm)
    target_label = "open" if min(left_closed, right_closed) >= args.plate_gripper_reopen_threshold else "close"
    if target_label == "open":
        last_close_time = max(
            float(left_arm.get("plate_gripper_last_close_time", 0.0)),
            float(right_arm.get("plate_gripper_last_close_time", 0.0)),
        )
        if time.monotonic() - last_close_time < args.plate_gripper_reopen_min_hold_time:
            return None
    left_ok = set_gripper_target_label(left_arm, target_label)
    right_ok = set_gripper_target_label(right_arm, target_label)
    if not left_ok and not right_ok:
        return None
    return target_label


def should_stabilize_plate(
    left_arm: dict,
    right_arm: dict,
    left_state: dict | None,
    right_state: dict | None,
    args: argparse.Namespace,
) -> bool:
    if not args.plate_task_filter:
        return False
    both_triggers = (
        is_trigger_pressed(left_state, args.trigger_threshold)
        and is_trigger_pressed(right_state, args.trigger_threshold)
    )
    if both_triggers:
        return True
    left_closed = get_gripper_close_fraction(left_arm) >= args.plate_grasp_close_threshold
    right_closed = get_gripper_close_fraction(right_arm) >= args.plate_grasp_close_threshold
    return left_closed and right_closed


def limit_plate_gripper_command_rate(
    arm_state: dict,
    args: argparse.Namespace,
    dt: float,
    command_updated: bool = True,
) -> None:
    if not args.plate_task_filter:
        return
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return
    max_speed = max(float(args.plate_gripper_max_speed), 0.0)
    if max_speed <= 0.0:
        return

    commanded_pos = float(
        arm_state.get(
            "gripper_pos",
            arm_state.get("gripper_goal", arm_state.get("gripper_open", 0.0)),
        )
    )
    if command_updated or "plate_gripper_desired_pos" not in arm_state:
        arm_state["plate_gripper_desired_pos"] = commanded_pos
    target_pos = float(arm_state["plate_gripper_desired_pos"])
    close_fraction = float(np.clip(args.plate_gripper_max_close_fraction, 0.0, 1.0))
    open_pos = float(arm_state["gripper_open"])
    close_pos = float(arm_state["gripper_close"])
    closest_allowed_pos = open_pos + close_fraction * (close_pos - open_pos)
    if close_pos < open_pos:
        target_pos = max(target_pos, closest_allowed_pos)
    else:
        target_pos = min(target_pos, closest_allowed_pos)

    prev_pos = arm_state.get("plate_gripper_filtered_pos")
    if prev_pos is None:
        measured_pos = get_measured_gripper_pos(arm_state)
        prev_pos = target_pos if measured_pos is None else measured_pos

    max_step = max_speed * max(float(dt), 0.0)
    filtered_pos = float(np.clip(target_pos, float(prev_pos) - max_step, float(prev_pos) + max_step))
    arm_state["plate_gripper_filtered_pos"] = filtered_pos
    arm_state["gripper_goal"] = filtered_pos
    arm_state["gripper_pos"] = filtered_pos


def add_dataset_frame(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    teleop: VuerControllerTeleop | None,
    head_img: np.ndarray | None,
    left_img: np.ndarray | None,
    right_img: np.ndarray | None,
    left_arm: dict,
    right_arm: dict,
    left_cmd: np.ndarray,
    right_cmd: np.ndarray,
) -> bool:
    if head_img is None:
        return False
    if not args.no_wrist and (left_img is None or right_img is None):
        return False

    left_joint_state = left_arm["robot"].get_joint_pos()
    right_joint_state = right_arm["robot"].get_joint_pos()
    state_vec = np.concatenate([left_joint_state, right_joint_state]).astype(np.float32)
    action_vec = np.concatenate([left_cmd, right_cmd]).astype(np.float32)
    tv = getattr(teleop, "tv", None) if teleop is not None else None
    left_controller_state = tv.left_controller_state if tv is not None else None
    right_controller_state = tv.right_controller_state if tv is not None else None
    frame = {
        "observation.images.head": head_img,
        "observation.state": state_vec,
        "action": action_vec,
        "teleoperation.matrices.left_controller": get_teleop_matrix(teleop, "left"),
        "teleoperation.matrices.right_controller": get_teleop_matrix(teleop, "right"),
        "teleoperation.matrices.head": get_teleop_head_matrix(teleop),
        "teleoperation.buttons": get_teleop_buttons(left_controller_state, right_controller_state),
        "task": args.task,
    }
    if not args.no_wrist:
        frame["observation.images.left_wrist"] = left_img
        frame["observation.images.right_wrist"] = right_img
    dataset.add_frame(frame)
    return True


def get_dataset_root(repo_id: str, dataset_root: str | None) -> Path:
    return Path(dataset_root) if dataset_root else HF_LEROBOT_HOME / repo_id


def remove_dataset_root(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    print(f"[Overwrite] Removing existing dataset root: {root}")
    if root.is_symlink() or root.is_file():
        root.unlink()
    else:
        shutil.rmtree(root)


def is_empty_dataset_root(root: Path) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        return False

    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        return True

    info_path = root / "meta" / "info.json"
    if set(files) != {info_path}:
        return False

    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return False
    return info.get("total_episodes", 0) == 0 and info.get("total_frames", 0) == 0


def get_missing_resume_metadata(root: Path) -> list[str]:
    required_meta_files = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
    return [f"meta/{name}" for name in required_meta_files if not (root / "meta" / name).exists()]


def normalize_feature_value(value):
    if isinstance(value, tuple):
        return [normalize_feature_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_feature_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_feature_value(val) for key, val in value.items() if key != "info"}
    return value


def validate_resume_dataset(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    dataset_fps: int,
    features: dict,
) -> bool:
    ok = True
    if dataset.meta.fps != dataset_fps:
        print(f"[Resume] Error: existing dataset fps is {dataset.meta.fps}, requested {dataset_fps}.")
        ok = False
    if dataset.meta.robot_type != args.robot_type:
        print(
            f"[Resume] Error: existing robot_type is {dataset.meta.robot_type!r}, "
            f"requested {args.robot_type!r}."
        )
        ok = False

    expected_features = normalize_feature_value({**features, **DEFAULT_FEATURES})
    existing_features = normalize_feature_value(dataset.meta.features)
    if existing_features != expected_features:
        print("[Resume] Error: existing dataset features do not match this recorder configuration.")
        existing_keys = set(existing_features)
        expected_keys = set(expected_features)
        missing = sorted(expected_keys - existing_keys)
        unexpected = sorted(existing_keys - expected_keys)
        if missing:
            print(f"[Resume] Missing existing feature keys: {missing}")
        if unexpected:
            print(f"[Resume] Unexpected existing feature keys: {unexpected}")
        for key in sorted(existing_keys & expected_keys):
            if existing_features[key] != expected_features[key]:
                print(
                    f"[Resume] Feature mismatch for {key}: "
                    f"existing={existing_features[key]}, expected={expected_features[key]}"
                )
        ok = False
    return ok


def create_or_resume_dataset(args: argparse.Namespace, dataset_fps: int, features: dict) -> LeRobotDataset:
    root = get_dataset_root(args.repo_id, args.dataset_root)
    if args.resume:
        if not root.exists():
            raise FileNotFoundError(f"--resume requested, but dataset root does not exist: {root}")
        if is_empty_dataset_root(root):
            print(
                "[Resume] Dataset root is empty or contains only zero-episode metadata; "
                "creating a new local dataset instead."
            )
            remove_dataset_root(root)
            return LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=dataset_fps,
                features=features,
                robot_type=args.robot_type,
                root=args.dataset_root,
                use_videos=True,
                image_writer_threads=args.image_writer_threads,
                image_writer_processes=args.image_writer_processes,
            )
        missing_metadata = get_missing_resume_metadata(root)
        if missing_metadata:
            raise RuntimeError(
                "Cannot resume incomplete local dataset. Missing "
                f"{missing_metadata} under {root}. Use --overwrite to recreate it, "
                "or repair/delete the partial dataset folder."
            )
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, download_videos=False)
        if not validate_resume_dataset(dataset, args, dataset_fps, features):
            raise RuntimeError("Cannot resume because the existing dataset is incompatible.")
        if args.image_writer_processes or args.image_writer_threads:
            dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
        dataset.episode_buffer = dataset.create_episode_buffer()
        print(
            f"[Resume] Loaded {dataset.meta.total_episodes} existing episodes; "
            f"next episode index is {dataset.episode_buffer['episode_index']}."
        )
        return dataset

    if args.overwrite:
        remove_dataset_root(root)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=dataset_fps,
        features=features,
        robot_type=args.robot_type,
        root=args.dataset_root,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def run_preflight_checks(args: argparse.Namespace, dataset_fps: int) -> bool:
    ok = True
    root = get_dataset_root(args.repo_id, args.dataset_root)
    print(f"[Preflight] Dataset root: {root}")
    print(f"[Preflight] Dataset fps: {dataset_fps}")
    print(f"[Preflight] LeRobot codebase version: {CODEBASE_VERSION}")
    if root.exists():
        print("[Preflight] Dataset root already exists; LeRobotDataset.create will not overwrite it.")
        if args.resume and is_empty_dataset_root(root):
            print("[Preflight] --resume set, but dataset root is empty; a new local dataset will be created.")
            return ok
        missing_metadata = get_missing_resume_metadata(root)
        if args.resume and missing_metadata:
            print(f"[Preflight] Error: cannot resume incomplete local dataset; missing {missing_metadata}.")
            print("[Preflight] Use --overwrite to recreate it, or repair/delete the partial dataset folder.")
            ok = False
        meta_dir = root / "meta"
        info_path = meta_dir / "info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                existing_version = info.get("codebase_version")
                if existing_version and existing_version != CODEBASE_VERSION:
                    print(
                        "[Preflight] Warning: dataset codebase_version is "
                        f"{existing_version} (expected {CODEBASE_VERSION})."
                    )
            except json.JSONDecodeError:
                print("[Preflight] Warning: failed to parse info.json.")
        else:
            print("[Preflight] Warning: info.json missing; dataset may be incomplete.")

        for fname in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
            if not (meta_dir / fname).exists():
                print(f"[Preflight] Warning: meta/{fname} missing.")

        if args.overwrite:
            print("[Preflight] --overwrite set; existing dataset root will be removed before create.")
        elif args.resume:
            print("[Preflight] --resume set; existing dataset will be loaded and appended to.")
        else:
            print("[Preflight] Choose a new --repo-id, pass --resume, pass --overwrite, or delete the existing dataset folder.")
            ok = False
    elif args.resume:
        print("[Preflight] --resume requested, but dataset root does not exist.")
        ok = False
    return ok


def _cleanup_episode_images(dataset: LeRobotDataset, episode_index: int) -> None:
    for cam_key in dataset.meta.camera_keys:
        img_dir = dataset._get_image_file_path(
            episode_index=episode_index, image_key=cam_key, frame_index=0
        ).parent
        if img_dir.is_dir():
            shutil.rmtree(img_dir, ignore_errors=True)
    images_root = dataset.root / "images"
    if images_root.is_dir() and not any(images_root.iterdir()):
        images_root.rmdir()


def discard_current_episode(dataset: LeRobotDataset) -> bool:
    if dataset.episode_buffer is None:
        return False
    episode_index = dataset.episode_buffer.get("episode_index")
    dataset._wait_image_writer()
    dataset.clear_episode_buffer()
    if episode_index is not None:
        _cleanup_episode_images(dataset, episode_index)
    return True


def delete_last_episode(dataset: LeRobotDataset) -> bool:
    if not dataset.meta.episodes:
        print("No saved episodes to delete.")
        return False
    dataset._wait_image_writer()
    episode_index = max(dataset.meta.episodes.keys())

    data_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
    if data_path.is_file():
        data_path.unlink()

    for cam_key in dataset.meta.camera_keys:
        img_dir = dataset._get_image_file_path(
            episode_index=episode_index, image_key=cam_key, frame_index=0
        ).parent
        if img_dir.is_dir():
            shutil.rmtree(img_dir, ignore_errors=True)

    for video_key in dataset.meta.video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)
        if video_path.is_file():
            video_path.unlink()

    dataset.meta.episodes.pop(episode_index, None)
    dataset.meta.episodes_stats.pop(episode_index, None)

    total_episodes = len(dataset.meta.episodes)
    total_frames = sum(ep["length"] for ep in dataset.meta.episodes.values())
    dataset.meta.info["total_episodes"] = total_episodes
    dataset.meta.info["total_frames"] = total_frames
    if total_episodes:
        max_chunk = max(idx // dataset.meta.info["chunks_size"] for idx in dataset.meta.episodes)
        dataset.meta.info["total_chunks"] = max_chunk + 1
        dataset.meta.info["splits"] = {"train": f"0:{total_episodes}"}
    else:
        dataset.meta.info["total_chunks"] = 0
        dataset.meta.info["splits"] = {}
    dataset.meta.info["total_videos"] = total_episodes * len(dataset.meta.video_keys)

    if dataset.meta.episodes_stats:
        dataset.meta.stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    else:
        dataset.meta.stats = {}

    write_info(dataset.meta.info, dataset.root)

    episodes_payload = [dataset.meta.episodes[idx] for idx in sorted(dataset.meta.episodes)]
    write_jsonlines(episodes_payload, dataset.root / EPISODES_PATH)

    episodes_stats_payload = [
        {"episode_index": idx, "stats": serialize_dict(dataset.meta.episodes_stats[idx])}
        for idx in sorted(dataset.meta.episodes_stats)
    ]
    write_jsonlines(episodes_stats_payload, dataset.root / EPISODES_STATS_PATH)

    print(f"Deleted episode {episode_index}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--squeeze", action="store_true", help="Record/teleop only while handle squeeze is held.")
    parser.add_argument("--no-wrist", action="store_true", help="Record only the head camera and skip wrist cameras.")
    parser.add_argument("--gripper-mode", type=str, choices=["none", "toggle", "trigger", "squeeze"], default=None)
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    parser.add_argument("--squeeze-threshold", type=float, default=0.1)
    parser.add_argument("--gripper-force-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-force-verbose", action="store_true")
    parser.add_argument("--gripper-force-print-interval", type=float, default=0.05)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=1.0)
    parser.add_argument("--gripper-backoff", type=float, default=0.1)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--lock-orientation", action="store_true")
    parser.add_argument(
        "--lock",
        action="store_true",
        help=(
            "In normal teleop mode, use controller squeeze as a rotation lock: "
            "right squeeze allows yaw only, left squeeze locks all rotations for both EEs."
        ),
    )
    parser.add_argument(
        "--plate-task-filter",
        action="store_true",
        help=(
            "Enable a bimanual task-space filter for lifting a plate/tray: preserve grasp spacing, "
            "limit left/right height mismatch, smooth EE motion, and optionally suppress wrist orientation changes."
        ),
    )
    parser.add_argument("--plate-filter-alpha", type=float, default=0.40)
    parser.add_argument("--plate-max-ee-speed", type=float, default=0.26)
    parser.add_argument("--plate-carry-max-ee-speed", type=float, default=0.22)
    parser.add_argument("--plate-carry-max-ee-accel", type=float, default=0.45)
    parser.add_argument("--plate-carry-max-xy-speed", type=float, default=0.24)
    parser.add_argument("--plate-carry-max-z-speed", type=float, default=0.055)
    parser.add_argument("--plate-carry-max-xy-accel", type=float, default=0.85)
    parser.add_argument("--plate-carry-max-z-accel", type=float, default=0.18)
    parser.add_argument("--plate-differential-gain", type=float, default=0.65)
    parser.add_argument("--plate-carry-differential-gain", type=float, default=0.03)
    parser.add_argument("--plate-max-height-diff", type=float, default=0.06)
    parser.add_argument("--plate-carry-max-height-diff", type=float, default=0.008)
    parser.add_argument("--plate-max-separation-delta", type=float, default=0.18)
    parser.add_argument("--plate-carry-max-separation-delta", type=float, default=0.02)
    parser.add_argument("--plate-carry-max-compression", type=float, default=0.0)
    parser.add_argument("--plate-down-margin", type=float, default=0.35)
    parser.add_argument("--plate-carry-down-margin", type=float, default=0.03)
    parser.add_argument("--plate-orientation-weight", type=float, default=0.0)
    parser.add_argument("--plate-carry-orientation-weight", type=float, default=0.9)
    parser.add_argument("--plate-grasp-close-threshold", type=float, default=0.35)
    parser.add_argument("--plate-gripper-max-speed", type=float, default=0.8)
    parser.add_argument("--plate-gripper-max-close-fraction", type=float, default=1.0)
    parser.add_argument("--plate-gripper-reopen-threshold", type=float, default=0.9)
    parser.add_argument("--plate-gripper-reopen-min-hold-time", type=float, default=1.5)
    parser.add_argument("--frequency", type=float, default=CONTROL_FREQUENCY)
    parser.add_argument("--home-time", type=float, default=2.0)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--ik-frame", type=str, default="link_6")
    parser.add_argument("--ik-dt", type=float, default=None)
    parser.add_argument("--ik-alpha", type=float, default=0.2)
    parser.add_argument("--ik-pos-cost", type=float, default=10.0)
    parser.add_argument("--ik-ori-cost", type=float, default=1.0)
    parser.add_argument("--ik-posture-cost", type=float, default=1e-3)
    parser.add_argument("--ik-damping-cost", type=float, default=1e-1)
    parser.add_argument("--ik-lm-damping", type=float, default=1e-4)
    parser.add_argument("--ik-gain", type=float, default=0.5)
    parser.add_argument("--ik-solver", type=str, default=None)
    parser.add_argument("--ik-solve-damping", type=float, default=1e-12)
    parser.add_argument("--ready-qpos", type=float, nargs="+", default=[0, 1.047, 1.047, 0, 0, 0, 0, 0])
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    dataset_mode = parser.add_mutually_exclusive_group()
    dataset_mode.add_argument("--overwrite", action="store_true", help="Delete an existing local dataset root before create.")
    dataset_mode.add_argument("--resume", action="store_true", help="Append new episodes to an existing local dataset root.")
    parser.add_argument("--task", type=str, default="teleop dual-arm demo")
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--reset-time", type=float, default=10.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--video-encode-workers", type=int, default=1)
    parser.add_argument("--head-serial", type=str, default="<head-camera-serial>")
    parser.add_argument("--left-wrist-serial", type=str, default="<left-wrist-camera-serial>")
    parser.add_argument("--right-wrist-serial", type=str, default="<right-wrist-camera-serial>")
    parser.add_argument("--head-width", type=int, default=640)
    parser.add_argument("--head-height", type=int, default=480)
    parser.add_argument("--head-fps", type=int, default=30)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)
    parser.add_argument(
        "--fix-realsense-controls",
        action="store_true",
        help="Disable RealSense color auto exposure. Other color controls are changed only when their flags are set.",
    )
    parser.add_argument("--rs-exposure", type=float, default=None, help="Fixed RealSense color exposure value.")
    parser.add_argument(
        "--rs-gain",
        "--rs-iso",
        dest="rs_gain",
        type=float,
        default=None,
        help="Fixed RealSense color gain. This is the RealSense ISO-like control.",
    )
    parser.add_argument("--rs-brightness", type=float, default=None, help="Fixed RealSense color brightness value.")
    parser.add_argument("--rs-white-balance", type=float, default=None, help="Fixed RealSense color white balance value.")
    parser.add_argument(
        "--rs-disable-auto-white-balance",
        action="store_true",
        help="Disable RealSense color auto white balance when fixing controls.",
    )
    parser.add_argument("--rs-allow-auto-white-balance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rs-disable-hdr", action="store_true", help="Disable RealSense color HDR if supported.")
    parser.add_argument("--rs-keep-hdr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List RealSense cameras with a global librealsense query, then exit.",
    )
    parser.add_argument("--allow-camera-fallback", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    if args.gripper_mode is None:
        args.gripper_mode = "toggle" if args.squeeze or args.plate_task_filter else "trigger"
    elif args.gripper_mode == "toggle" and not args.squeeze and not args.plate_task_filter:
        parser.error("--gripper-mode toggle requires --squeeze or --plate-task-filter")
    if args.lock and args.squeeze:
        print("Warning: --lock is only used outside --squeeze mode; ignoring rotation squeeze locks.")
    plate_task_filter = PlateLiftTeleopFilter.from_args(args)
    if args.plate_task_filter:
        print(
            "Plate task filter enabled: smooth/free approach, then tight bimanual stabilization "
            "once both grippers are closing or closed; loaded carry uses faster horizontal motion with "
            "slow vertical/tilt motion, and gripper commands are rate-limited with optional close limiting "
            "through --plate-gripper-max-close-fraction."
        )

    if args.list_cameras:
        for serial, name in list_realsense_devices():
            print(f"{serial}  {name}")
        return

    dataset_fps = int(round(args.frequency))
    if abs(args.frequency - dataset_fps) > 1e-3:
        print(f"Warning: --frequency {args.frequency} is not integer; dataset fps set to {dataset_fps}.")

    head_fps = args.head_fps if args.head_fps is not None else dataset_fps
    wrist_fps = args.wrist_fps if args.wrist_fps is not None else dataset_fps

    if not args.skip_preflight:
        ok = run_preflight_checks(args, dataset_fps)
        if args.preflight_only:
            return
        if not ok:
            return

    teleop = None
    left_arm = None
    right_arm = None
    head_cam = None
    left_wrist_cam = None
    right_wrist_cam = None
    dataset = None

    try:
        ensure_can_interface_ready(args.left_channel)
        ensure_can_interface_ready(args.right_channel)
        head_cam = RealSenseStream(
            RealSenseConfig(
                args.head_serial,
                args.head_width,
                args.head_height,
                head_fps,
                allow_fallback=args.allow_camera_fallback,
                fixed_color_controls=args.fix_realsense_controls,
                exposure=args.rs_exposure,
                gain=args.rs_gain,
                brightness=args.rs_brightness,
                disable_hdr=args.rs_disable_hdr,
                disable_auto_white_balance=args.rs_disable_auto_white_balance,
                white_balance=args.rs_white_balance,
            )
        )
        if not args.no_wrist:
            left_wrist_cam = RealSenseStream(
                RealSenseConfig(
                    args.left_wrist_serial,
                    args.wrist_width,
                    args.wrist_height,
                    wrist_fps,
                    allow_fallback=args.allow_camera_fallback,
                    fixed_color_controls=args.fix_realsense_controls,
                    exposure=args.rs_exposure,
                    gain=args.rs_gain,
                    brightness=args.rs_brightness,
                    disable_hdr=args.rs_disable_hdr,
                    disable_auto_white_balance=args.rs_disable_auto_white_balance,
                    white_balance=args.rs_white_balance,
                )
            )
            right_wrist_cam = RealSenseStream(
                RealSenseConfig(
                    args.right_wrist_serial,
                    args.wrist_width,
                    args.wrist_height,
                    wrist_fps,
                    allow_fallback=args.allow_camera_fallback,
                    fixed_color_controls=args.fix_realsense_controls,
                    exposure=args.rs_exposure,
                    gain=args.rs_gain,
                    brightness=args.rs_brightness,
                    disable_hdr=args.rs_disable_hdr,
                    disable_auto_white_balance=args.rs_disable_auto_white_balance,
                    white_balance=args.rs_white_balance,
                )
            )
        head_cam.start()
        if not args.no_wrist:
            left_wrist_cam.start()
            right_wrist_cam.start()

        head_stream = head_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
        head_width = int(round(head_stream.width()))
        head_height = int(round(head_stream.height()))
        if (head_width, head_height) != (args.head_width, args.head_height):
            print(
                f"Head camera stream is {head_width}x{head_height}; overriding requested "
                f"{args.head_width}x{args.head_height}."
            )
        if not args.no_wrist:
            left_stream = left_wrist_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
            left_width = int(round(left_stream.width()))
            left_height = int(round(left_stream.height()))
            right_stream = right_wrist_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
            right_width = int(round(right_stream.width()))
            right_height = int(round(right_stream.height()))
            if (left_width, left_height) != (args.wrist_width, args.wrist_height):
                print(
                    f"Left wrist camera stream is {left_width}x{left_height}; overriding requested "
                    f"{args.wrist_width}x{args.wrist_height}."
                )
            if (right_width, right_height) != (args.wrist_width, args.wrist_height):
                print(
                    f"Right wrist camera stream is {right_width}x{right_height}; overriding requested "
                    f"{args.wrist_width}x{args.wrist_height}."
                )

        ik_frame = args.site or args.ik_frame
        ik_dt = args.ik_dt if args.ik_dt is not None else 1.0 / args.frequency
        left_arm = setup_arm(
            args.left_channel,
            args.left_gripper,
            ik_frame,
            args.left_gripper_invert,
            ik_dt,
            args.ik_alpha,
            args.ik_pos_cost,
            args.ik_ori_cost,
            args.ik_posture_cost,
            args.ik_damping_cost,
            args.ik_lm_damping,
            args.ik_gain,
            args.ik_solver,
            args.ik_solve_damping,
        )
        right_arm = setup_arm(
            args.right_channel,
            args.right_gripper,
            ik_frame,
            args.right_gripper_invert,
            ik_dt,
            args.ik_alpha,
            args.ik_pos_cost,
            args.ik_ori_cost,
            args.ik_posture_cost,
            args.ik_damping_cost,
            args.ik_lm_damping,
            args.ik_gain,
            args.ik_solver,
            args.ik_solve_damping,
        )
        teleop = VuerControllerTeleopCam(resolution=(head_height, head_width))

        left_joint_names = build_joint_names(
            "left", left_arm["robot"].num_dofs(), left_arm["gripper_index"]
        )
        right_joint_names = build_joint_names(
            "right", right_arm["robot"].num_dofs(), right_arm["gripper_index"]
        )
        state_names = left_joint_names + right_joint_names
        action_names = list(state_names)

        features = {
            "observation.images.head": build_image_feature(head_height, head_width),
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
            "teleoperation.matrices.left_controller": build_matrix_feature(),
            "teleoperation.matrices.right_controller": build_matrix_feature(),
            "teleoperation.matrices.head": build_matrix_feature(),
            "teleoperation.buttons": {
                "dtype": "float32",
                "shape": (len(TELEOP_BUTTON_NAMES),),
                "names": TELEOP_BUTTON_NAMES,
            },
        }
        if not args.no_wrist:
            features["observation.images.left_wrist"] = build_image_feature(left_height, left_width)
            features["observation.images.right_wrist"] = build_image_feature(right_height, right_width)

        dataset = create_or_resume_dataset(args, dataset_fps, features)
        if args.vcodec != "libsvtav1":
            print(
                "Warning: lerobot 0.1.0 does not expose vcodec in LeRobotDataset; "
                f"using custom encoder override '{args.vcodec}'."
            )

        def _encode_episode_videos_with_codec(episode_index: int) -> dict:
            def _encode_one_camera(key: str) -> tuple[str, str]:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                if not video_path.is_file():
                    img_dir = dataset._get_image_file_path(
                        episode_index=episode_index, image_key=key, frame_index=0
                    ).parent
                    encode_video_frames(img_dir, video_path, dataset.fps, vcodec=args.vcodec, overwrite=True)
                return key, str(video_path)

            keys = list(dataset.meta.video_keys)
            if len(keys) <= 1 or args.video_encode_workers <= 1:
                return dict(_encode_one_camera(key) for key in keys)

            max_workers = min(max(1, args.video_encode_workers), len(keys))
            video_paths = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_encode_one_camera, key) for key in keys]
                for future in as_completed(futures):
                    key, path = future.result()
                    video_paths[key] = path
            return video_paths

        dataset.encode_episode_videos = _encode_episode_videos_with_codec

        print("Moving both arms to ready pose...")
        move_to_ready_pose_open_gripper(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
        move_to_ready_pose_open_gripper(right_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
        sync_arm_state_from_robot(left_arm)
        sync_arm_state_from_robot(right_arm)
        if args.squeeze:
            print(
                "Ready pose reached. Hold a handle squeeze to teleop that arm and record both arms; "
                "release all squeezes to pause."
            )
            print("Left controller: X saves accumulated squeeze segments, Y discards; Y when idle deletes last episode.")
        else:
            print("Ready pose reached. Press A on right controller to start teleop/recording.")
            print("Left controller: X saves episode, Y discards; Y when idle deletes last episode.")
    except Exception:
        if dataset is not None:
            dataset.stop_image_writer()
        if head_cam is not None:
            head_cam.stop()
        if left_wrist_cam is not None:
            left_wrist_cam.stop()
        if right_wrist_cam is not None:
            right_wrist_cam.stop()
        if teleop is not None:
            teleop.cleanup()
        if left_arm is not None:
            left_arm["robot"].close()
        if right_arm is not None:
            right_arm["robot"].close()
        raise

    left_init_controller = None
    right_init_controller = None
    left_init_pose = None
    right_init_pose = None

    print("Waiting for both controllers to be valid...")
    last_warn_time = 0.0
    last_a_pressed = False
    last_b_pressed = False
    last_x_pressed = False
    last_y_pressed = False
    last_left_trigger_pressed = False
    last_right_trigger_pressed = False
    last_left_gripper_warn_time = 0.0
    last_right_gripper_warn_time = 0.0
    last_left_gripper_print_time = 0.0
    last_right_gripper_print_time = 0.0
    last_num_episodes_warn_time = 0.0
    last_rotation_lock_mode = "none"

    teleop_enabled = False
    awaiting_reference = False
    recording = False
    squeeze_recording = False
    episode_idx = dataset.meta.total_episodes if dataset is not None else 0
    episode_start_time = None

    left_cmd = left_arm["robot"].get_joint_pos()
    right_cmd = right_arm["robot"].get_joint_pos()

    emergency_stop = False
    try:
        while True:
            head_preview = head_cam.get_latest_frame() if head_cam is not None else None
            if teleop is not None:
                teleop.update_image(head_preview)
            right_state = teleop.tv.right_controller_state if teleop and teleop.tv else None
            left_state = teleop.tv.left_controller_state if teleop and teleop.tv else None

            b_pressed = bool(right_state.get("bButton")) if right_state else False
            if b_pressed and not last_b_pressed:
                raise KeyboardInterrupt
            last_b_pressed = b_pressed

            a_pressed = bool(right_state.get("aButton")) if right_state else False
            # On the left controller, TeleVision maps X/Y to aButton/bButton.
            x_pressed = bool(left_state.get("aButton")) if left_state else False
            y_pressed = bool(left_state.get("bButton")) if left_state else False

            a_rising = a_pressed and not last_a_pressed
            x_rising = x_pressed and not last_x_pressed
            y_rising = y_pressed and not last_y_pressed
            last_a_pressed = a_pressed
            last_x_pressed = x_pressed
            last_y_pressed = y_pressed
            left_trigger_pressed = is_trigger_pressed(left_state, args.trigger_threshold)
            right_trigger_pressed = is_trigger_pressed(right_state, args.trigger_threshold)
            left_trigger_rising = left_trigger_pressed and not last_left_trigger_pressed
            right_trigger_rising = right_trigger_pressed and not last_right_trigger_pressed
            last_left_trigger_pressed = left_trigger_pressed
            last_right_trigger_pressed = right_trigger_pressed

            if args.squeeze:
                if y_rising:
                    if squeeze_recording or has_episode_frames(dataset):
                        squeeze_recording = False
                        discard_current_episode(dataset)
                        print("Current episode discarded. Resetting to ready pose.")
                        move_to_ready_pose_open_gripper(
                            left_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                        )
                        move_to_ready_pose_open_gripper(
                            right_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                        )
                        sync_arm_state_from_robot(left_arm)
                        sync_arm_state_from_robot(right_arm)
                        left_cmd = left_arm["robot"].get_joint_pos()
                        right_cmd = right_arm["robot"].get_joint_pos()
                        left_init_controller = None
                        right_init_controller = None
                        left_init_pose = None
                        right_init_pose = None
                        plate_task_filter.reset()
                        print("Ready pose reached. Hold a handle squeeze to teleop that arm and record both arms.")
                    else:
                        if delete_last_episode(dataset):
                            episode_idx = dataset.meta.total_episodes
                    time.sleep(0.1)
                    continue

                if x_rising:
                    squeeze_recording = False
                    saved = maybe_save_episode(dataset)
                    if saved:
                        episode_idx = dataset.meta.total_episodes
                        print(f"Episode {episode_idx} saved.")
                    else:
                        print("No frames recorded; nothing to save.")
                        discard_current_episode(dataset)
                    move_to_ready_pose_open_gripper(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                    move_to_ready_pose_open_gripper(right_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = left_arm["robot"].get_joint_pos()
                    right_cmd = right_arm["robot"].get_joint_pos()
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_task_filter.reset()
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        break
                    print("Ready pose reached. Hold a handle squeeze to teleop that arm and record both arms.")
                    time.sleep(0.1)
                    continue

                left_squeeze_active, right_squeeze_active = get_handle_squeeze_states(
                    left_state, right_state, args.squeeze_threshold
                )
                squeeze_active = left_squeeze_active or right_squeeze_active
                if squeeze_active and args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    now = time.monotonic()
                    if not squeeze_recording and now - last_num_episodes_warn_time > 1.0:
                        print("Reached requested number of episodes; ignoring squeeze.")
                        last_num_episodes_warn_time = now
                    squeeze_active = False

                if not squeeze_active:
                    if squeeze_recording:
                        squeeze_recording = False
                        left_init_controller = None
                        right_init_controller = None
                        left_init_pose = None
                        right_init_pose = None
                        plate_task_filter.reset()
                        sync_arm_state_from_robot(left_arm)
                        sync_arm_state_from_robot(right_arm)
                        left_cmd = left_arm["robot"].get_joint_pos()
                        right_cmd = right_arm["robot"].get_joint_pos()
                        print("All squeezes released. Recording paused; holding current pose.")

                    left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                    right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                    left_arm["robot"].command_joint_pos(left_cmd)
                    right_arm["robot"].command_joint_pos(right_cmd)
                    time.sleep(1.0 / args.frequency)
                    continue

                left_mat = teleop.get_controller_matrix("left") if left_squeeze_active else None
                right_mat = teleop.get_controller_matrix("right") if right_squeeze_active else None
                if (left_squeeze_active and left_mat is None) or (right_squeeze_active and right_mat is None):
                    now = time.monotonic()
                    if now - last_warn_time > 1.0:
                        print("Waiting for valid squeezed controller pose; holding current pose.")
                        last_warn_time = now
                    left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                    right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                    left_arm["robot"].command_joint_pos(left_cmd)
                    right_arm["robot"].command_joint_pos(right_cmd)
                    time.sleep(0.01)
                    continue

                if left_mat is not None:
                    left_mat = vuer_to_robot_matrix(left_mat)
                if right_mat is not None:
                    right_mat = vuer_to_robot_matrix(right_mat)

                if not squeeze_recording:
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                    right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_task_filter.reset()
                    squeeze_recording = True
                    next_ep = dataset.meta.total_episodes + 1
                    if has_episode_frames(dataset):
                        print("Squeeze held. Reinitialized active VR pose; appending to current episode.")
                    elif args.num_episodes > 0:
                        print(f"Squeeze held. Recording episode {next_ep}/{args.num_episodes}")
                    else:
                        print(f"Squeeze held. Recording episode {next_ep}")
                    time.sleep(0.05)
                    continue

                if left_squeeze_active and left_init_controller is None:
                    sync_arm_state_from_robot(left_arm)
                    left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                    left_init_controller = left_mat.copy()
                    left_init_pose = left_arm["target_pose"].copy()
                    print("Left squeeze held. Left arm teleop active.")
                elif not left_squeeze_active and left_init_controller is not None:
                    sync_arm_state_from_robot(left_arm)
                    left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                    left_init_controller = None
                    left_init_pose = None
                    print("Left squeeze released. Holding left arm.")

                if right_squeeze_active and right_init_controller is None:
                    sync_arm_state_from_robot(right_arm)
                    right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                    right_init_controller = right_mat.copy()
                    right_init_pose = right_arm["target_pose"].copy()
                    print("Right squeeze held. Right arm teleop active.")
                elif not right_squeeze_active and right_init_controller is not None:
                    sync_arm_state_from_robot(right_arm)
                    right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                    right_init_controller = None
                    right_init_pose = None
                    print("Right squeeze released. Holding right arm.")

                left_gripper_command_updated = False
                right_gripper_command_updated = False
                if args.gripper_mode == "toggle":
                    plate_trigger_toggle = args.plate_task_filter and left_squeeze_active and right_squeeze_active and (
                        (left_trigger_rising and right_trigger_pressed)
                        or (right_trigger_rising and left_trigger_pressed)
                    )
                    if plate_trigger_toggle:
                        target_label = toggle_plate_grippers_from_triggers(left_arm, right_arm, args)
                        if target_label is not None:
                            print(f"Both triggers toggled grippers {target_label}.")
                            left_gripper_command_updated = True
                            right_gripper_command_updated = True
                    elif left_squeeze_active and left_trigger_rising and not args.plate_task_filter:
                        reopen_threshold = args.plate_gripper_reopen_threshold if args.plate_task_filter else 0.5
                        target_label = toggle_gripper_from_trigger(left_arm, reopen_threshold)
                        if target_label is not None:
                            print(f"Left trigger toggled gripper {target_label}.")
                        left_gripper_command_updated = True
                    if right_squeeze_active and right_trigger_rising and not args.plate_task_filter:
                        reopen_threshold = args.plate_gripper_reopen_threshold if args.plate_task_filter else 0.5
                        target_label = toggle_gripper_from_trigger(right_arm, reopen_threshold)
                        if target_label is not None:
                            print(f"Right trigger toggled gripper {target_label}.")
                        right_gripper_command_updated = True
                elif args.gripper_mode != "none":
                    if left_squeeze_active:
                        update_gripper_from_controller(
                            left_arm,
                            left_state,
                            args.gripper_mode,
                            args.left_gripper_invert,
                        )
                        left_gripper_command_updated = True
                    if right_squeeze_active:
                        update_gripper_from_controller(
                            right_arm,
                            right_state,
                            args.gripper_mode,
                            args.right_gripper_invert,
                        )
                        right_gripper_command_updated = True

                plate_filter_dt = 1.0 / max(float(args.frequency), 1e-6)
                if left_squeeze_active:
                    limit_plate_gripper_command_rate(
                        left_arm,
                        args,
                        plate_filter_dt,
                        command_updated=left_gripper_command_updated,
                    )
                if right_squeeze_active:
                    limit_plate_gripper_command_rate(
                        right_arm,
                        args,
                        plate_filter_dt,
                        command_updated=right_gripper_command_updated,
                    )

                left_gripper_blocked, left_eff, left_gripper_pos, left_gripper_goal = (
                    maybe_limit_gripper_close(
                        left_arm,
                        args.gripper_force_threshold,
                        args.gripper_force_ema_alpha,
                        args.gripper_backoff,
                    )
                    if left_squeeze_active
                    else (False, None, None, None)
                )
                right_gripper_blocked, right_eff, right_gripper_pos, right_gripper_goal = (
                    maybe_limit_gripper_close(
                        right_arm,
                        args.gripper_force_threshold,
                        args.gripper_force_ema_alpha,
                        args.gripper_backoff,
                    )
                    if right_squeeze_active
                    else (False, None, None, None)
                )
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
                if left_gripper_blocked:
                    now = time.monotonic()
                    if now - last_left_gripper_warn_time > 1.0:
                        print(f"Left gripper force threshold hit ({left_eff:.2f}); holding position.")
                        last_left_gripper_warn_time = now
                if right_gripper_blocked:
                    now = time.monotonic()
                    if now - last_right_gripper_warn_time > 1.0:
                        print(f"Right gripper force threshold hit ({right_eff:.2f}); holding position.")
                        last_right_gripper_warn_time = now

                left_target_pose = None
                right_target_pose = None

                if left_squeeze_active and left_init_controller is not None and left_mat is not None:
                    left_target_pose = compute_target_pose(
                        left_init_pose,
                        left_init_controller,
                        left_mat,
                        args.pos_scale,
                        args.lock_orientation,
                        args.rot_scale,
                    )
                if right_squeeze_active and right_init_controller is not None and right_mat is not None:
                    right_target_pose = compute_target_pose(
                        right_init_pose,
                        right_init_controller,
                        right_mat,
                        args.pos_scale,
                        args.lock_orientation,
                        args.rot_scale,
                    )

                if left_target_pose is not None and right_target_pose is not None:
                    left_target_pose, right_target_pose = plate_task_filter.filter_pair(
                        left_target_pose,
                        right_target_pose,
                        left_init_pose,
                        right_init_pose,
                        stabilize=should_stabilize_plate(left_arm, right_arm, left_state, right_state, args),
                    )
                elif left_target_pose is not None:
                    left_target_pose = plate_task_filter.filter_single("left", left_target_pose, left_init_pose)
                elif right_target_pose is not None:
                    right_target_pose = plate_task_filter.filter_single("right", right_target_pose, right_init_pose)

                if left_target_pose is not None:
                    left_success, left_q = left_arm["kin"].ik(left_target_pose, init_q=left_arm["target_q"])
                    if left_success:
                        left_arm["target_q"] = left_q
                        left_arm["target_pose"] = left_target_pose
                        left_cmd = build_command(
                            left_q,
                            left_arm["gripper_pos"],
                            left_arm["arm_dofs"],
                            left_arm["gripper_index"],
                            left_arm["robot"].num_dofs(),
                        )
                    else:
                        now = time.monotonic()
                        if now - last_warn_time > 1.0:
                            print("Left arm IK failed; holding last command.")
                            last_warn_time = now
                left_arm["robot"].command_joint_pos(left_cmd)

                if right_target_pose is not None:
                    right_success, right_q = right_arm["kin"].ik(right_target_pose, init_q=right_arm["target_q"])
                    if right_success:
                        right_arm["target_q"] = right_q
                        right_arm["target_pose"] = right_target_pose
                        right_cmd = build_command(
                            right_q,
                            right_arm["gripper_pos"],
                            right_arm["arm_dofs"],
                            right_arm["gripper_index"],
                            right_arm["robot"].num_dofs(),
                        )
                    else:
                        now = time.monotonic()
                        if now - last_warn_time > 1.0:
                            print("Right arm IK failed; holding last command.")
                            last_warn_time = now
                right_arm["robot"].command_joint_pos(right_cmd)

                if squeeze_recording:
                    frame_count = dataset.episode_buffer.get("size", 0) if dataset.episode_buffer is not None else 0
                    elapsed = frame_count / float(dataset_fps) if dataset_fps > 0 else 0.0
                    if args.episode_time > 0 and elapsed >= args.episode_time:
                        squeeze_recording = False
                        left_init_controller = None
                        right_init_controller = None
                        left_init_pose = None
                        right_init_pose = None
                        plate_task_filter.reset()
                        saved = maybe_save_episode(dataset)
                        if saved:
                            episode_idx = dataset.meta.total_episodes
                            print(f"Episode {episode_idx} saved.")
                        else:
                            print("No frames recorded; nothing to save.")
                            discard_current_episode(dataset)
                        move_to_ready_pose_open_gripper(
                            left_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                        )
                        move_to_ready_pose_open_gripper(
                            right_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                        )
                        sync_arm_state_from_robot(left_arm)
                        sync_arm_state_from_robot(right_arm)
                        left_cmd = left_arm["robot"].get_joint_pos()
                        right_cmd = right_arm["robot"].get_joint_pos()
                        if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                            break
                        print("Ready pose reached. Hold a handle squeeze to teleop that arm and record both arms.")
                        time.sleep(0.1)
                        continue

                    head_img = head_preview if head_preview is not None else head_cam.get_latest_frame()
                    left_img = left_wrist_cam.get_latest_frame() if left_wrist_cam is not None else None
                    right_img = right_wrist_cam.get_latest_frame() if right_wrist_cam is not None else None
                    add_dataset_frame(
                        dataset,
                        args,
                        teleop,
                        head_img,
                        left_img,
                        right_img,
                        left_arm,
                        right_arm,
                        left_cmd,
                        right_cmd,
                    )

                time.sleep(1.0 / args.frequency)
                continue

            if y_rising:
                if recording or awaiting_reference:
                    teleop_enabled = False
                    awaiting_reference = False
                    recording = False
                    episode_start_time = None
                    plate_task_filter.reset()
                    discard_current_episode(dataset)
                    print("Current episode discarded. Resetting to ready pose.")
                    move_to_ready_pose_open_gripper(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                    move_to_ready_pose_open_gripper(
                        right_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                    )
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = left_arm["robot"].get_joint_pos()
                    right_cmd = right_arm["robot"].get_joint_pos()
                    print("Ready pose reached. Press A on right controller to start teleop/recording.")
                else:
                    if delete_last_episode(dataset):
                        episode_idx = dataset.meta.total_episodes
                time.sleep(0.1)
                continue

            if x_rising and recording:
                teleop_enabled = False
                awaiting_reference = False
                recording = False
                episode_start_time = None
                plate_task_filter.reset()
                saved = maybe_save_episode(dataset)
                if saved:
                    episode_idx = dataset.meta.total_episodes
                    print(f"Episode {episode_idx} saved.")
                else:
                    print("No frames recorded; nothing to save.")
                    discard_current_episode(dataset)
                move_to_ready_pose_open_gripper(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                move_to_ready_pose_open_gripper(right_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                sync_arm_state_from_robot(left_arm)
                sync_arm_state_from_robot(right_arm)
                left_cmd = left_arm["robot"].get_joint_pos()
                right_cmd = right_arm["robot"].get_joint_pos()
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    break
                print("Ready pose reached. Press A on right controller to start teleop/recording.")
                time.sleep(0.1)
                continue

            if a_rising and not recording and not awaiting_reference:
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    print("Reached requested number of episodes; ignoring A.")
                else:
                    awaiting_reference = True
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_task_filter.reset()
                    print("A pressed. Waiting to capture controller reference.")

            left_mat = None
            right_mat = None
            if awaiting_reference or teleop_enabled:
                left_mat = teleop.get_controller_matrix("left")
                right_mat = teleop.get_controller_matrix("right")
                if left_mat is None or right_mat is None:
                    time.sleep(0.01)
                    continue

                left_mat = vuer_to_robot_matrix(left_mat)
                right_mat = vuer_to_robot_matrix(right_mat)

                if awaiting_reference:
                    left_init_controller = left_mat.copy()
                    right_init_controller = right_mat.copy()
                    left_init_pose = left_arm["target_pose"].copy()
                    right_init_pose = right_arm["target_pose"].copy()
                    last_rotation_lock_mode = "none"
                    plate_task_filter.reset()
                    awaiting_reference = False
                    teleop_enabled = True
                    recording = True
                    episode_start_time = time.monotonic()
                    next_ep = dataset.meta.total_episodes + 1
                    if args.num_episodes > 0:
                        print(f"Recording episode {next_ep}/{args.num_episodes}")
                    else:
                        print(f"Recording episode {next_ep}")
                    time.sleep(0.2)
                    continue

                if args.lock:
                    rotation_lock_mode = get_rotation_lock_mode(
                        left_state, right_state, args.squeeze_threshold
                    )
                    if rotation_lock_mode != last_rotation_lock_mode:
                        if rotation_lock_mode == "locked":
                            print("Left squeeze held: locking all EE rotations.")
                        elif rotation_lock_mode == "yaw":
                            print("Right squeeze held: locking EE roll/pitch; yaw remains active.")
                        else:
                            print("Rotation squeeze lock released.")
                        last_rotation_lock_mode = rotation_lock_mode
                    left_mat = filter_controller_rotation(
                        left_init_controller,
                        left_mat,
                        rotation_lock_mode,
                    )
                    right_mat = filter_controller_rotation(
                        right_init_controller,
                        right_mat,
                        rotation_lock_mode,
                    )

            if not teleop_enabled:
                time.sleep(0.01)
                continue

            left_gripper_command_updated = False
            right_gripper_command_updated = False
            if args.gripper_mode == "toggle":
                plate_trigger_toggle = args.plate_task_filter and (
                    (left_trigger_rising and right_trigger_pressed)
                    or (right_trigger_rising and left_trigger_pressed)
                )
                if plate_trigger_toggle:
                    target_label = toggle_plate_grippers_from_triggers(left_arm, right_arm, args)
                    if target_label is not None:
                        print(f"Both triggers toggled grippers {target_label}.")
                        left_gripper_command_updated = True
                        right_gripper_command_updated = True
                elif left_trigger_rising and not args.plate_task_filter:
                    reopen_threshold = args.plate_gripper_reopen_threshold if args.plate_task_filter else 0.5
                    target_label = toggle_gripper_from_trigger(left_arm, reopen_threshold)
                    if target_label is not None:
                        print(f"Left trigger toggled gripper {target_label}.")
                    left_gripper_command_updated = True
                if right_trigger_rising and not args.plate_task_filter:
                    reopen_threshold = args.plate_gripper_reopen_threshold if args.plate_task_filter else 0.5
                    target_label = toggle_gripper_from_trigger(right_arm, reopen_threshold)
                    if target_label is not None:
                        print(f"Right trigger toggled gripper {target_label}.")
                    right_gripper_command_updated = True
            elif args.gripper_mode != "none":
                update_gripper_from_controller(
                    left_arm,
                    left_state,
                    args.gripper_mode,
                    args.left_gripper_invert,
                )
                update_gripper_from_controller(
                    right_arm,
                    right_state,
                    args.gripper_mode,
                    args.right_gripper_invert,
                )
                left_gripper_command_updated = True
                right_gripper_command_updated = True

            if args.gripper_mode != "none":
                plate_filter_dt = 1.0 / max(float(args.frequency), 1e-6)
                limit_plate_gripper_command_rate(
                    left_arm,
                    args,
                    plate_filter_dt,
                    command_updated=left_gripper_command_updated,
                )
                limit_plate_gripper_command_rate(
                    right_arm,
                    args,
                    plate_filter_dt,
                    command_updated=right_gripper_command_updated,
                )

            left_gripper_blocked, left_eff, left_gripper_pos, left_gripper_goal = maybe_limit_gripper_close(
                left_arm,
                args.gripper_force_threshold,
                args.gripper_force_ema_alpha,
                args.gripper_backoff,
            )
            right_gripper_blocked, right_eff, right_gripper_pos, right_gripper_goal = maybe_limit_gripper_close(
                right_arm,
                args.gripper_force_threshold,
                args.gripper_force_ema_alpha,
                args.gripper_backoff,
            )
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
            if left_gripper_blocked:
                now = time.monotonic()
                if now - last_left_gripper_warn_time > 1.0:
                    print(f"Left gripper force threshold hit ({left_eff:.2f}); holding position.")
                    last_left_gripper_warn_time = now
            if right_gripper_blocked:
                now = time.monotonic()
                if now - last_right_gripper_warn_time > 1.0:
                    print(f"Right gripper force threshold hit ({right_eff:.2f}); holding position.")
                    last_right_gripper_warn_time = now

            left_target_pose = compute_target_pose(
                left_init_pose,
                left_init_controller,
                left_mat,
                args.pos_scale,
                args.lock_orientation,
                args.rot_scale,
            )
            right_target_pose = compute_target_pose(
                right_init_pose,
                right_init_controller,
                right_mat,
                args.pos_scale,
                args.lock_orientation,
                args.rot_scale,
            )
            left_target_pose, right_target_pose = plate_task_filter.filter_pair(
                left_target_pose,
                right_target_pose,
                left_init_pose,
                right_init_pose,
                stabilize=should_stabilize_plate(left_arm, right_arm, left_state, right_state, args),
            )

            left_success, left_q = left_arm["kin"].ik(left_target_pose, init_q=left_arm["target_q"])
            if left_success:
                left_arm["target_q"] = left_q
                left_arm["target_pose"] = left_target_pose
                left_cmd = build_command(
                    left_q,
                    left_arm["gripper_pos"],
                    left_arm["arm_dofs"],
                    left_arm["gripper_index"],
                    left_arm["robot"].num_dofs(),
                )
                left_arm["robot"].command_joint_pos(left_cmd)
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Left arm IK failed; holding last command.")
                    last_warn_time = now

            right_success, right_q = right_arm["kin"].ik(right_target_pose, init_q=right_arm["target_q"])
            if right_success:
                right_arm["target_q"] = right_q
                right_arm["target_pose"] = right_target_pose
                right_cmd = build_command(
                    right_q,
                    right_arm["gripper_pos"],
                    right_arm["arm_dofs"],
                    right_arm["gripper_index"],
                    right_arm["robot"].num_dofs(),
                )
                right_arm["robot"].command_joint_pos(right_cmd)
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Right arm IK failed; holding last command.")
                    last_warn_time = now

            now = time.monotonic()
            if recording:
                elapsed = now - episode_start_time if episode_start_time is not None else 0.0
                if args.episode_time > 0 and elapsed >= args.episode_time:
                    teleop_enabled = False
                    awaiting_reference = False
                    recording = False
                    episode_start_time = None
                    plate_task_filter.reset()
                    saved = maybe_save_episode(dataset)
                    if saved:
                        episode_idx = dataset.meta.total_episodes
                        print(f"Episode {episode_idx} saved.")
                    else:
                        print("No frames recorded; nothing to save.")
                        discard_current_episode(dataset)
                    move_to_ready_pose_open_gripper(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
                    move_to_ready_pose_open_gripper(
                        right_arm, np.array(args.ready_qpos, dtype=float), args.home_time
                    )
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = left_arm["robot"].get_joint_pos()
                    right_cmd = right_arm["robot"].get_joint_pos()
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        break
                    print("Ready pose reached. Press A on right controller to start teleop/recording.")
                    time.sleep(0.1)
                    continue
                else:
                    head_img = head_preview if head_preview is not None else head_cam.get_latest_frame()
                    left_img = left_wrist_cam.get_latest_frame() if left_wrist_cam is not None else None
                    right_img = right_wrist_cam.get_latest_frame() if right_wrist_cam is not None else None
                    add_dataset_frame(
                        dataset,
                        args,
                        teleop,
                        head_img,
                        left_img,
                        right_img,
                        left_arm,
                        right_arm,
                        left_cmd,
                        right_cmd,
                    )

            time.sleep(1.0 / args.frequency)

    except KeyboardInterrupt:
        emergency_stop = True
        print("\nCtrl+C or B received. Discarding unsaved episode, then returning home.")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if emergency_stop:
                if dataset is not None and has_episode_frames(dataset):
                    discard_current_episode(dataset)
            elif dataset is not None and (recording or (args.squeeze and has_episode_frames(dataset))):
                print("Saving partial episode before shutdown...")
                maybe_save_episode(dataset)
        finally:
            if left_arm is not None:
                reset_to_home(left_arm, args.home_time)
            if right_arm is not None:
                reset_to_home(right_arm, args.home_time)
            signal.signal(signal.SIGINT, original_handler)
            if left_arm is not None:
                left_arm["robot"].close()
            if right_arm is not None:
                right_arm["robot"].close()
            if teleop is not None:
                teleop.cleanup()
            if head_cam is not None:
                head_cam.stop()
            if left_wrist_cam is not None:
                left_wrist_cam.stop()
            if right_wrist_cam is not None:
                right_wrist_cam.stop()
            if dataset is not None:
                dataset.stop_image_writer()
            print("Teleop record shutdown complete.")


if __name__ == "__main__":
    main()
