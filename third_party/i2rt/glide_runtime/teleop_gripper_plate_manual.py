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

    def _apply_fixed_color_controls(self) -> None:
        if not self.config.fixed_color_controls:
            return

        sensor = self._get_color_sensor()
        if sensor is None:
            print(f"RealSense {self.config.serial}: no color sensor found for fixed controls.")
            return

        print(f"RealSense {self.config.serial}: applying fixed controls to {self._get_sensor_name(sensor)}.")
        self._set_supported_option(sensor, "enable_auto_exposure", 0.0, "auto exposure")

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


def is_record_squeeze_active(left_state: dict | None, right_state: dict | None, args: argparse.Namespace) -> bool:
    left_squeeze = is_squeeze_pressed(left_state, args.squeeze_threshold)
    right_squeeze = is_squeeze_pressed(right_state, args.squeeze_threshold)
    return left_squeeze or right_squeeze


def get_handle_squeeze_states(
    left_state: dict | None, right_state: dict | None, squeeze_threshold: float
) -> tuple[bool, bool]:
    return (
        is_squeeze_pressed(left_state, squeeze_threshold),
        is_squeeze_pressed(right_state, squeeze_threshold),
    )


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


def toggle_gripper_from_trigger(arm_state: dict) -> str | None:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return None

    current_pos = get_measured_gripper_pos(arm_state)
    if current_pos is None:
        current_pos = float(arm_state.get("gripper_pos", arm_state["gripper_open"]))

    open_pos = float(arm_state["gripper_open"])
    close_pos = float(arm_state["gripper_close"])
    is_open = abs(current_pos - open_pos) <= abs(current_pos - close_pos)
    target_label = "close" if is_open else "open"
    target_pos = close_pos if is_open else open_pos

    arm_state["gripper_goal"] = target_pos
    arm_state["gripper_pos"] = target_pos
    if target_label == "open":
        arm_state["gripper_blocked"] = False
    return target_label


def is_plate_gripper_closed(arm_state: dict, tolerance: float) -> bool:
    if arm_state.get("gripper_index") is None:
        return False

    close_pos = float(arm_state["gripper_close"])
    open_pos = float(arm_state["gripper_open"])
    goal = float(arm_state.get("gripper_goal", arm_state.get("gripper_pos", open_pos)))
    travel = abs(open_pos - close_pos)
    allowed_error = max(0.0, tolerance) * (travel if travel > 1e-9 else 1.0)
    return abs(goal - close_pos) <= allowed_error


def rotation_angle(rot: np.ndarray) -> float:
    cos_theta = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def rotation_delta_power(delta_rot: np.ndarray, scale: float) -> np.ndarray:
    scale = float(np.clip(scale, 0.0, 1.0))
    if scale <= 0.0:
        return np.eye(3)
    if scale >= 1.0:
        return delta_rot

    theta = rotation_angle(delta_rot)
    if theta < 1e-9:
        return np.eye(3)

    if np.pi - theta < 1e-4:
        diag = np.diag(delta_rot)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[0] = np.copysign(axis[0], delta_rot[2, 1] - delta_rot[1, 2])
        elif axis[1] >= axis[0] and axis[1] >= axis[2]:
            axis[1] = np.copysign(axis[1], delta_rot[0, 2] - delta_rot[2, 0])
        else:
            axis[2] = np.copysign(axis[2], delta_rot[1, 0] - delta_rot[0, 1])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return np.eye(3)
        axis = axis / axis_norm
    else:
        axis = np.array(
            [
                delta_rot[2, 1] - delta_rot[1, 2],
                delta_rot[0, 2] - delta_rot[2, 0],
                delta_rot[1, 0] - delta_rot[0, 1],
            ]
        ) / (2.0 * np.sin(theta))

    scaled_theta = theta * scale
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(scaled_theta) * k + (1.0 - np.cos(scaled_theta)) * (k @ k)


@dataclass
class PlateFilter:
    height_threshold: float
    distance_threshold: float
    rotation_threshold: float
    active: bool = False
    anchor_distance: float | None = None
    anchor_left_rotation: np.ndarray | None = None
    anchor_right_rotation: np.ndarray | None = None
    anchor_xy_direction: np.ndarray | None = None

    def reset(self) -> None:
        self.active = False
        self.anchor_distance = None
        self.anchor_left_rotation = None
        self.anchor_right_rotation = None
        self.anchor_xy_direction = None

    def apply(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_active: bool = True,
        right_active: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_filtered = left_pose.copy()
        right_filtered = right_pose.copy()

        self._constrain_height(left_filtered, right_filtered, left_active, right_active)
        if not self.active:
            self._capture_anchor(left_filtered, right_filtered)

        self._constrain_distance(left_filtered, right_filtered, left_active, right_active)
        self._constrain_height(left_filtered, right_filtered, left_active, right_active)
        if left_active:
            self._constrain_rotation(left_filtered, self.anchor_left_rotation)
        if right_active:
            self._constrain_rotation(right_filtered, self.anchor_right_rotation)
        return left_filtered, right_filtered

    def _capture_anchor(self, left_pose: np.ndarray, right_pose: np.ndarray) -> None:
        delta = right_pose[:3, 3] - left_pose[:3, 3]
        self.anchor_distance = float(np.linalg.norm(delta))
        xy_delta = delta[:2]
        xy_norm = float(np.linalg.norm(xy_delta))
        if xy_norm > 1e-9:
            self.anchor_xy_direction = xy_delta / xy_norm
        else:
            self.anchor_xy_direction = np.array([1.0, 0.0])
        self.anchor_left_rotation = left_pose[:3, :3].copy()
        self.anchor_right_rotation = right_pose[:3, :3].copy()
        self.active = True

    def _constrain_height(
        self, left_pose: np.ndarray, right_pose: np.ndarray, left_active: bool, right_active: bool
    ) -> None:
        if not left_active and not right_active:
            return

        threshold = max(0.0, self.height_threshold)
        left_z = float(left_pose[2, 3])
        right_z = float(right_pose[2, 3])
        diff = left_z - right_z
        if abs(diff) <= threshold:
            return
        desired_diff = float(np.clip(diff, -threshold, threshold))
        if left_active and not right_active:
            left_pose[2, 3] = right_z + desired_diff
            return
        if right_active and not left_active:
            right_pose[2, 3] = left_z - desired_diff
            return
        midpoint = 0.5 * (left_z + right_z)
        half_diff = 0.5 * desired_diff
        left_pose[2, 3] = midpoint + half_diff
        right_pose[2, 3] = midpoint - half_diff

    def _constrain_distance(
        self, left_pose: np.ndarray, right_pose: np.ndarray, left_active: bool, right_active: bool
    ) -> None:
        if not left_active and not right_active:
            return
        if self.anchor_distance is None:
            return

        threshold = max(0.0, self.distance_threshold)
        left_pos = left_pose[:3, 3]
        right_pos = right_pose[:3, 3]
        delta = right_pos - left_pos
        distance = float(np.linalg.norm(delta))
        error = distance - self.anchor_distance
        if abs(error) <= threshold:
            return

        target_distance = self.anchor_distance + np.sign(error) * threshold
        z_delta = float(delta[2])
        target_xy_distance = float(np.sqrt(max(target_distance * target_distance - z_delta * z_delta, 0.0)))

        xy_delta = delta[:2]
        xy_norm = float(np.linalg.norm(xy_delta))
        if xy_norm > 1e-9:
            xy_direction = xy_delta / xy_norm
        elif self.anchor_xy_direction is not None:
            xy_direction = self.anchor_xy_direction
        else:
            xy_direction = np.array([1.0, 0.0])

        if left_active and not right_active:
            left_pose[:2, 3] = right_pos[:2] - target_xy_distance * xy_direction
            return
        if right_active and not left_active:
            right_pose[:2, 3] = left_pos[:2] + target_xy_distance * xy_direction
            return

        xy_midpoint = 0.5 * (left_pos[:2] + right_pos[:2])
        left_pose[:2, 3] = xy_midpoint - 0.5 * target_xy_distance * xy_direction
        right_pose[:2, 3] = xy_midpoint + 0.5 * target_xy_distance * xy_direction

    def _constrain_rotation(self, pose: np.ndarray, anchor_rotation: np.ndarray | None) -> None:
        if anchor_rotation is None:
            return

        threshold = max(0.0, self.rotation_threshold)
        delta_rot = pose[:3, :3] @ anchor_rotation.T
        angle = rotation_angle(delta_rot)
        if angle <= threshold:
            return
        if angle < 1e-9:
            pose[:3, :3] = anchor_rotation
            return
        pose[:3, :3] = rotation_delta_power(delta_rot, threshold / angle) @ anchor_rotation


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
    parser = argparse.ArgumentParser(
        description=(
            "Dual-arm VR teleop LeRobot recorder. Press right A to latch recording for both arms, "
            "or hold a VR squeeze button to record gated teleop segments. When both grippers are "
            "commanded closed, a plate filter constrains the two end-effectors before motor "
            "commands are sent."
        )
    )
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-mode", type=str, choices=["none", "toggle", "trigger", "squeeze"], default="toggle")
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    parser.add_argument(
        "--squeeze-controller",
        type=str,
        choices=["left", "right", "either", "both"],
        default="either",
        help="Deprecated; each handle squeeze now gates its own arm, and either squeeze records both arms.",
    )
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
        "--plate-gripper-closed-threshold",
        type=float,
        default=0.1,
        help="Fraction of gripper travel from the closed command that engages the plate filter.",
    )
    parser.add_argument(
        "--plate-height-threshold",
        type=float,
        default=0.005,
        help="Allowed left/right EE height difference in meters while the plate filter is active.",
    )
    parser.add_argument(
        "--plate-distance-threshold",
        type=float,
        default=0.01,
        help="Allowed deviation in meters from the left/right EE distance captured when the plate filter engages.",
    )
    parser.add_argument(
        "--plate-rotation-threshold",
        type=float,
        default=0.1,
        help="Allowed per-EE rotation change in radians from the orientation captured when the plate filter engages.",
    )
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
        help="Disable RealSense color auto exposure.",
    )
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--allow-camera-fallback", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

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

    ensure_realsense_serial(args.head_serial, "head")
    ensure_realsense_serial(args.left_wrist_serial, "left wrist")
    ensure_realsense_serial(args.right_wrist_serial, "right wrist")

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
            )
        )
        left_wrist_cam = RealSenseStream(
            RealSenseConfig(
                args.left_wrist_serial,
                args.wrist_width,
                args.wrist_height,
                wrist_fps,
                allow_fallback=args.allow_camera_fallback,
                fixed_color_controls=args.fix_realsense_controls,
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
            )
        )
        head_cam.start()
        left_wrist_cam.start()
        right_wrist_cam.start()

        head_stream = head_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
        head_width = int(round(head_stream.width()))
        head_height = int(round(head_stream.height()))
        left_stream = left_wrist_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
        left_width = int(round(left_stream.width()))
        left_height = int(round(left_stream.height()))
        right_stream = right_wrist_cam.profile.get_stream(rs.stream.color).as_video_stream_profile()
        right_width = int(round(right_stream.width()))
        right_height = int(round(right_stream.height()))
        if (head_width, head_height) != (args.head_width, args.head_height):
            print(
                f"Head camera stream is {head_width}x{head_height}; overriding requested "
                f"{args.head_width}x{args.head_height}."
            )
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
            "observation.images.left_wrist": build_image_feature(left_height, left_width),
            "observation.images.right_wrist": build_image_feature(right_height, right_width),
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
        print(
            "Ready pose reached. Press A on right controller to record both arms, or hold a handle "
            "squeeze to teleop that arm and record both arms."
        )
        print("Left controller: X saves current episode, Y discards; Y when idle deletes last episode.")
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
    plate_filter = PlateFilter(
        args.plate_height_threshold,
        args.plate_distance_threshold,
        args.plate_rotation_threshold,
    )

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
    last_plate_active = False

    squeeze_recording = False
    button_recording = False
    awaiting_button_reference = False
    episode_idx = dataset.meta.total_episodes if dataset is not None else 0

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

            if y_rising:
                if button_recording or awaiting_button_reference or squeeze_recording or has_episode_frames(dataset):
                    button_recording = False
                    awaiting_button_reference = False
                    squeeze_recording = False
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
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_filter.reset()
                    last_plate_active = False
                    print("Ready pose reached. Press A to record both arms, or hold a handle squeeze.")
                else:
                    if delete_last_episode(dataset):
                        episode_idx = dataset.meta.total_episodes
                time.sleep(0.1)
                continue

            if x_rising:
                button_recording = False
                awaiting_button_reference = False
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
                plate_filter.reset()
                last_plate_active = False
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    break
                print("Ready pose reached. Press A to record both arms, or hold a handle squeeze.")
                time.sleep(0.1)
                continue

            if a_rising and not button_recording and not awaiting_button_reference:
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    print("Reached requested number of episodes; ignoring A.")
                else:
                    awaiting_button_reference = True
                    squeeze_recording = False
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_filter.reset()
                    last_plate_active = False
                    print("A pressed. Waiting to capture both controller references.")

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

            left_active = button_recording or awaiting_button_reference or left_squeeze_active
            right_active = button_recording or awaiting_button_reference or right_squeeze_active
            teleop_active = left_active or right_active

            if not teleop_active:
                if squeeze_recording:
                    squeeze_recording = False
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
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

            left_mat = teleop.get_controller_matrix("left") if left_active else None
            right_mat = teleop.get_controller_matrix("right") if right_active else None
            if (left_active and left_mat is None) or (right_active and right_mat is None):
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Waiting for valid controller pose; holding current pose.")
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

            if awaiting_button_reference:
                sync_arm_state_from_robot(left_arm)
                sync_arm_state_from_robot(right_arm)
                left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                left_init_controller = left_mat.copy()
                right_init_controller = right_mat.copy()
                left_init_pose = left_arm["target_pose"].copy()
                right_init_pose = right_arm["target_pose"].copy()
                awaiting_button_reference = False
                button_recording = True
                squeeze_recording = False
                next_ep = dataset.meta.total_episodes + 1
                if args.num_episodes > 0:
                    print(f"Recording episode {next_ep}/{args.num_episodes}")
                else:
                    print(f"Recording episode {next_ep}")
                time.sleep(0.2)
                continue

            if squeeze_active and not button_recording and not squeeze_recording:
                sync_arm_state_from_robot(left_arm)
                sync_arm_state_from_robot(right_arm)
                left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                left_init_controller = None
                right_init_controller = None
                left_init_pose = None
                right_init_pose = None
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

            if left_active and left_init_controller is None:
                sync_arm_state_from_robot(left_arm)
                left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                left_init_controller = left_mat.copy()
                left_init_pose = left_arm["target_pose"].copy()
                print("Left squeeze held. Left arm teleop active.")
            elif not left_active and left_init_controller is not None:
                sync_arm_state_from_robot(left_arm)
                left_cmd = np.asarray(left_arm["robot"].get_joint_pos(), dtype=float).copy()
                left_init_controller = None
                left_init_pose = None
                print("Left squeeze released. Holding left arm.")

            if right_active and right_init_controller is None:
                sync_arm_state_from_robot(right_arm)
                right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                right_init_controller = right_mat.copy()
                right_init_pose = right_arm["target_pose"].copy()
                print("Right squeeze held. Right arm teleop active.")
            elif not right_active and right_init_controller is not None:
                sync_arm_state_from_robot(right_arm)
                right_cmd = np.asarray(right_arm["robot"].get_joint_pos(), dtype=float).copy()
                right_init_controller = None
                right_init_pose = None
                print("Right squeeze released. Holding right arm.")

            left_trigger_pressed = is_trigger_pressed(left_state, args.trigger_threshold)
            right_trigger_pressed = is_trigger_pressed(right_state, args.trigger_threshold)
            left_trigger_rising = left_trigger_pressed and not last_left_trigger_pressed
            right_trigger_rising = right_trigger_pressed and not last_right_trigger_pressed
            last_left_trigger_pressed = left_trigger_pressed
            last_right_trigger_pressed = right_trigger_pressed

            if args.gripper_mode == "toggle":
                if left_active and left_trigger_rising:
                    target_label = toggle_gripper_from_trigger(left_arm)
                    if target_label is not None:
                        print(f"Left trigger toggled gripper {target_label}.")
                if right_active and right_trigger_rising:
                    target_label = toggle_gripper_from_trigger(right_arm)
                    if target_label is not None:
                        print(f"Right trigger toggled gripper {target_label}.")
            elif args.gripper_mode != "none":
                if left_active:
                    update_gripper_from_controller(
                        left_arm,
                        left_state,
                        args.gripper_mode,
                        args.left_gripper_invert,
                    )
                if right_active:
                    update_gripper_from_controller(
                        right_arm,
                        right_state,
                        args.gripper_mode,
                        args.right_gripper_invert,
                    )

            left_gripper_blocked, left_eff, left_gripper_pos, left_gripper_goal = (
                maybe_limit_gripper_close(
                    left_arm,
                    args.gripper_force_threshold,
                    args.gripper_force_ema_alpha,
                    args.gripper_backoff,
                )
                if left_active
                else (False, None, None, None)
            )
            right_gripper_blocked, right_eff, right_gripper_pos, right_gripper_goal = (
                maybe_limit_gripper_close(
                    right_arm,
                    args.gripper_force_threshold,
                    args.gripper_force_ema_alpha,
                    args.gripper_backoff,
                )
                if right_active
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

            left_target_pose = left_arm["target_pose"].copy()
            right_target_pose = right_arm["target_pose"].copy()
            if left_active and left_init_controller is not None and left_mat is not None:
                left_target_pose = compute_target_pose(
                    left_init_pose,
                    left_init_controller,
                    left_mat,
                    args.pos_scale,
                    args.lock_orientation,
                    args.rot_scale,
                )
            if right_active and right_init_controller is not None and right_mat is not None:
                right_target_pose = compute_target_pose(
                    right_init_pose,
                    right_init_controller,
                    right_mat,
                    args.pos_scale,
                    args.lock_orientation,
                    args.rot_scale,
                )

            plate_closed = is_plate_gripper_closed(
                left_arm, args.plate_gripper_closed_threshold
            ) and is_plate_gripper_closed(right_arm, args.plate_gripper_closed_threshold)
            if plate_closed and (left_active or right_active):
                left_target_pose, right_target_pose = plate_filter.apply(
                    left_target_pose,
                    right_target_pose,
                    left_active,
                    right_active,
                )
                if not last_plate_active:
                    print(
                        "Plate filter engaged: constraining EE height, distance, and rotation "
                        "until a gripper opens."
                    )
                last_plate_active = True
            else:
                if last_plate_active:
                    print("Plate filter released: a gripper opened.")
                plate_filter.reset()
                last_plate_active = False

            if left_active and left_init_controller is not None and left_mat is not None:
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

            if right_active and right_init_controller is not None and right_mat is not None:
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

            if squeeze_recording or button_recording:
                frame_count = dataset.episode_buffer.get("size", 0) if dataset.episode_buffer is not None else 0
                elapsed = frame_count / float(dataset_fps) if dataset_fps > 0 else 0.0
                if args.episode_time > 0 and elapsed >= args.episode_time:
                    button_recording = False
                    squeeze_recording = False
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    plate_filter.reset()
                    last_plate_active = False
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
                    print("Ready pose reached. Press A to record both arms, or hold a handle squeeze.")
                    time.sleep(0.1)
                    continue
                else:
                    head_img = head_preview if head_preview is not None else head_cam.get_latest_frame()
                    left_img = left_wrist_cam.get_latest_frame()
                    right_img = right_wrist_cam.get_latest_frame()
                    if head_img is not None and left_img is not None and right_img is not None:
                        left_joint_state = left_arm["robot"].get_joint_pos()
                        right_joint_state = right_arm["robot"].get_joint_pos()
                        state_vec = np.concatenate([left_joint_state, right_joint_state]).astype(np.float32)
                        action_vec = np.concatenate([left_cmd, right_cmd]).astype(np.float32)
                        dataset.add_frame(
                            {
                                "observation.images.head": head_img,
                                "observation.images.left_wrist": left_img,
                                "observation.images.right_wrist": right_img,
                                "observation.state": state_vec,
                                "action": action_vec,
                                "teleoperation.matrices.left_controller": get_teleop_matrix(teleop, "left"),
                                "teleoperation.matrices.right_controller": get_teleop_matrix(teleop, "right"),
                                "teleoperation.matrices.head": get_teleop_head_matrix(teleop),
                                "teleoperation.buttons": get_teleop_buttons(left_state, right_state),
                                "task": args.task,
                            }
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
            elif dataset is not None and has_episode_frames(dataset):
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
