#!/usr/bin/env python3
"""I2RT dual-arm policy client with SpaceMouse control + LeRobot recording.

Controls:
- Right SpaceMouse button 1: start policy motion and begin recording an episode.
- Right SpaceMouse button 2: save current episode if recording, then reset to ready pose.
- Ctrl+C: emergency stop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import signal
import shutil
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional

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

try:
    import pyspacemouse
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError(
        "pyspacemouse is required for SpaceMouse controls. Install it in your robot environment."
    ) from exc

from glide_runtime.policy_base import (  # noqa: E402
    GripperForceLimiter,
    RealSenseConfig,
    RealSenseStream,
    _build_home_qpos,
    _build_observation,
    _clip_gripper_if_needed,
    _normalize_qpos,
    _split_action,
    action_chunk_broker,
    ensure_can_interface_ready,
    ensure_realsense_serial,
    get_yam_robot,
    list_realsense_devices,
    websocket_client_policy,
)
from glide_runtime.policy_base import GripperType  # noqa: E402


class FixedControlRealSenseStream(RealSenseStream):
    def __init__(self, config: RealSenseConfig):
        super().__init__(config)
        self._saved_color_options = []

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
        if not getattr(self.config, "fixed_color_controls", False):
            return

        sensor = self._get_color_sensor()
        if sensor is None:
            print(f"RealSense {self.config.serial}: no color sensor found for fixed controls.")
            return

        print(f"RealSense {self.config.serial}: applying fixed controls to {self._get_sensor_name(sensor)}.")
        self._set_supported_option(sensor, "enable_auto_exposure", 0.0, "auto exposure")
        if getattr(self.config, "disable_auto_white_balance", False):
            self._disable_supported_option(sensor, "enable_auto_white_balance", "auto white balance")
        if getattr(self.config, "disable_hdr", False):
            self._disable_supported_option(sensor, "hdr_enabled", "HDR")
        if getattr(self.config, "exposure", None) is not None:
            self._set_supported_option(sensor, "exposure", self.config.exposure, "exposure")
        if getattr(self.config, "gain", None) is not None:
            self._set_supported_option(sensor, "gain", self.config.gain, "gain/ISO")
        if getattr(self.config, "brightness", None) is not None:
            self._set_supported_option(sensor, "brightness", self.config.brightness, "brightness")
        if getattr(self.config, "white_balance", None) is not None:
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

    def stop(self) -> None:
        if not self.started:
            return
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._restore_color_controls()
        self.pipeline.stop()
        self.started = False


def _build_realsense_config(
    serial: str,
    width: int,
    height: int,
    fps: int,
    args: argparse.Namespace,
) -> RealSenseConfig:
    config = RealSenseConfig(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        allow_fallback=args.allow_camera_fallback,
    )
    config.fixed_color_controls = args.fix_realsense_controls
    config.exposure = args.rs_exposure
    config.gain = args.rs_gain
    config.brightness = args.rs_brightness
    config.disable_hdr = args.rs_disable_hdr
    config.disable_auto_white_balance = args.rs_disable_auto_white_balance
    config.white_balance = args.rs_white_balance
    return config


def _open_device(device_name: str, device_index: int):
    try:
        return pyspacemouse.open(device=device_name, device_index=device_index)
    except TypeError:
        try:
            return pyspacemouse.open(device=device_name, DeviceNumber=device_index)
        except Exception:
            return None
    except Exception:
        return None


def _open_by_path(device_path: str):
    if not hasattr(pyspacemouse, "open_by_path"):
        return None
    try:
        return pyspacemouse.open_by_path(device_path)
    except Exception:
        return None


def _read_uevent(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key] = value
    except OSError:
        return {}
    return data


def _find_receiver_groups(vendor_id: str = "256f", product_id: str = "c652") -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    try:
        entries = os.listdir("/sys/class/hidraw")
    except OSError:
        return []
    for entry in entries:
        uevent_path = os.path.join("/sys/class/hidraw", entry, "device", "uevent")
        info = _read_uevent(uevent_path)
        hid_id = info.get("HID_ID", "").lower()
        if vendor_id not in hid_id or product_id not in hid_id:
            continue
        phys = info.get("HID_PHYS", "")
        receiver_key = phys.split("/input")[0] if phys else entry
        input_index = None
        if "input" in phys:
            try:
                input_index = int(phys.rsplit("input", 1)[1])
            except ValueError:
                input_index = None
        device_path = os.path.join("/dev", entry)
        groups.setdefault(receiver_key, []).append(
            {
                "hidraw": entry,
                "path": device_path,
                "phys": phys,
                "input_index": input_index,
            }
        )
    for entries in groups.values():
        entries.sort(
            key=lambda item: (
                item["input_index"] is None,
                item["input_index"] if item["input_index"] is not None else 0,
                item["hidraw"],
            )
        )
    return list(groups.values())


def list_spacemouse_receivers() -> None:
    groups = _find_receiver_groups()
    if not groups:
        print("No 3Dconnexion receiver hidraw devices found.")
        return
    for idx, group in enumerate(groups):
        print(f"receiver {idx}")
        for item in group:
            input_index = item["input_index"]
            suffix = f" input{input_index}" if input_index is not None else ""
            print(f"  {item['path']}  {item['phys']}{suffix}")


def _open_first_working(group: list[dict]):
    for item in group:
        device = _open_by_path(item["path"])
        if device is None or not hasattr(device, "read"):
            continue
        ok = False
        for _ in range(5):
            state = device.read()
            if state is not None and hasattr(state, "x"):
                ok = True
                break
            time.sleep(0.02)
        if ok:
            return device, item["path"]
        try:
            device.close()
        except Exception:
            pass
    return None, None


def _get_connected_descriptors(max_devices: int = 2) -> list[tuple[str, int]]:
    if not hasattr(pyspacemouse, "get_connected_devices"):
        return []
    try:
        connected = pyspacemouse.get_connected_devices() or []
    except Exception:
        return []
    descriptors = []
    used_indices: dict[str, int] = {}
    for name in connected:
        index = used_indices.get(name, 0)
        descriptors.append((name, index))
        used_indices[name] = index + 1
        if len(descriptors) >= max_devices:
            break
    return descriptors


def open_two_devices(left_path: Optional[str] = None, right_path: Optional[str] = None):
    devices = []
    if right_path and not left_path:
        right = _open_by_path(right_path)
        if right is not None:
            devices.append((right, right_path))
        return devices
    if left_path and not right_path:
        left = _open_by_path(left_path)
        if left is not None:
            devices.append((left, left_path))
        return devices
    if left_path and right_path:
        left = _open_by_path(left_path)
        right = _open_by_path(right_path)
        if left is not None:
            devices.append((left, left_path))
        if right is not None:
            devices.append((right, right_path))
        return devices

    groups = _find_receiver_groups()
    if len(groups) >= 2:
        left, left_dev_path = _open_first_working(groups[0])
        right, right_dev_path = _open_first_working(groups[1])
        if left is not None:
            devices.append((left, left_dev_path))
        if right is not None:
            devices.append((right, right_dev_path))
        return devices

    descriptors = _get_connected_descriptors(max_devices=2)
    for device_name, device_index in descriptors:
        device = _open_device(device_name, device_index)
        if device is not None and hasattr(device, "read"):
            devices.append((device, f"{device_name}#{device_index}"))
    return devices


class SpaceMouseButtonReader:
    """Reads button states from the right SpaceMouse device."""

    def __init__(
        self,
        left_path: Optional[str] = None,
        right_path: Optional[str] = None,
        max_devices: int = 2,
        swap_devices: bool = False,
    ):
        self._device = None
        self._devices = []
        self._has_read_all = hasattr(pyspacemouse, "read_all")
        self._has_read = hasattr(pyspacemouse, "read")
        if max_devices >= 2:
            opened = open_two_devices(left_path=left_path, right_path=right_path)
            if opened:
                selected_index = 0 if swap_devices and len(opened) > 1 else -1
                right_device, _source = opened[selected_index]
                if right_device is not None:
                    self._devices = [right_device]
        if not self._devices:
            try:
                device = pyspacemouse.open()
            except Exception:
                device = None
            if device is not None and hasattr(device, "read"):
                self._device = device

    @property
    def device_count(self) -> int:
        if self._devices:
            return len(self._devices)
        if self._device is not None:
            return 1
        return 0

    def _read_state(self):
        if self._devices:
            return [device.read() for device in self._devices]
        if self._device is not None:
            return self._device.read()
        if self._has_read_all:
            return pyspacemouse.read_all()
        if self._has_read:
            return pyspacemouse.read()
        raise AttributeError("pyspacemouse has no read/read_all API")

    @staticmethod
    def _parse_buttons(device_state) -> list[int]:
        if device_state is None:
            return [0, 0]

        if hasattr(device_state, "buttons"):
            buttons = list(getattr(device_state, "buttons", [0, 0]))
            if len(buttons) >= 2:
                return [1 if buttons[0] else 0, 1 if buttons[-1] else 0]
            if len(buttons) == 1:
                return [1 if buttons[0] else 0, 0]
            return [0, 0]

        if isinstance(device_state, (list, tuple, np.ndarray)) and len(device_state) > 6:
            extra = device_state[6]
            if isinstance(extra, (list, tuple, np.ndarray)):
                extra_buttons = list(extra)
                if len(extra_buttons) >= 2:
                    return [1 if extra_buttons[0] else 0, 1 if extra_buttons[-1] else 0]
                if len(extra_buttons) == 1:
                    return [1 if extra_buttons[0] else 0, 0]
            elif isinstance(extra, (int, float, np.number)):
                bits = int(extra)
                return [1 if bits & 0x1 else 0, 1 if bits & 0x2 else 0]

        return [0, 0]

    def get_buttons(self) -> list[int]:
        state = self._read_state()
        if state is None:
            return [0, 0, 0, 0]

        if isinstance(state, (list, tuple)) and state and isinstance(state[0], (int, float, np.number)):
            state = [state]
        elif not isinstance(state, (list, tuple)):
            state = [state]

        buttons = []
        for device_state in state:
            buttons.extend(self._parse_buttons(device_state))

        while len(buttons) < 4:
            buttons.extend([0, 0])
        return buttons[:4]

    def close(self) -> None:
        for device in self._devices:
            try:
                device.close()
            except Exception:
                pass
        if self._device is not None:
            self._device.close()
        elif hasattr(pyspacemouse, "close"):
            pyspacemouse.close()


def _build_ready_open_qpos(
    robot,
    ready_qpos: np.ndarray,
    gripper_index: Optional[int],
    gripper_limits: Optional[tuple[float, float]],
    gripper_invert: bool,
) -> np.ndarray:
    ready = _normalize_qpos(np.asarray(ready_qpos, dtype=float), robot.num_dofs())
    if gripper_index is None or gripper_index >= len(ready):
        return ready
    if gripper_limits is None:
        current_q = robot.get_joint_pos()
        if len(current_q) > gripper_index:
            ready[gripper_index] = float(current_q[gripper_index])
        return ready
    gripper_open = 1.0
    gripper_close = 0.0
    if gripper_invert:
        gripper_open, gripper_close = gripper_close, gripper_open
    ready[gripper_index] = gripper_open
    return ready


def build_image_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


def build_joint_names(prefix: str, total_dofs: int, gripper_index: int | None) -> list[str]:
    names = []
    for idx in range(total_dofs):
        if gripper_index is not None and idx == gripper_index:
            names.append(f"{prefix}_gripper")
        else:
            names.append(f"{prefix}_joint_{idx}")
    return names


def maybe_save_episode(dataset: LeRobotDataset | None) -> bool:
    if dataset is None or dataset.episode_buffer is None:
        return False
    if dataset.episode_buffer.get("size", 0) <= 0:
        return False
    dataset.save_episode()
    return True


def get_dataset_root(repo_id: str, dataset_root: str | None) -> Path:
    return Path(dataset_root) if dataset_root else HF_LEROBOT_HOME / repo_id


def remove_dataset_root(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    print(f"[Resume] Removing empty dataset root: {root}")
    if root.is_symlink() or root.is_file():
        root.unlink()
    else:
        shutil.rmtree(root)


def is_empty_dataset_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
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
                f"{missing_metadata} under {root}. Repair/delete the partial dataset folder."
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
            print("[Preflight] Repair/delete the partial dataset folder.")
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

        if args.resume:
            print("[Preflight] --resume set; existing dataset will be loaded and appended to.")
        else:
            print("[Preflight] Choose a new --repo-id, pass --resume, or delete the existing dataset folder.")
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


def discard_current_episode(dataset: LeRobotDataset | None) -> bool:
    if dataset is None or dataset.episode_buffer is None:
        return False
    episode_index = dataset.episode_buffer.get("episode_index")
    dataset._wait_image_writer()
    dataset.clear_episode_buffer()
    if episode_index is not None:
        _cleanup_episode_images(dataset, episode_index)
    return True


def delete_last_episode(dataset: LeRobotDataset | None) -> bool:
    if dataset is None or not dataset.meta.episodes:
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


def _wait_first_frames(
    head_cam: RealSenseStream,
    left_cam: RealSenseStream,
    right_cam: RealSenseStream,
    timeout_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.monotonic()
    while True:
        head_frame = head_cam.get_latest_frame()
        left_frame = left_cam.get_latest_frame()
        right_frame = right_cam.get_latest_frame()
        if head_frame is not None and left_frame is not None and right_frame is not None:
            return head_frame, left_frame, right_frame
        if time.monotonic() - start > timeout_s:
            raise RuntimeError("Timed out waiting for initial camera frames.")
        time.sleep(0.05)


def add_record_frame(
    dataset: LeRobotDataset | None,
    args: argparse.Namespace,
    head_cam: RealSenseStream,
    left_cam: RealSenseStream,
    right_cam: RealSenseStream,
    left_robot,
    right_robot,
    left_cmd: np.ndarray,
    right_cmd: np.ndarray,
) -> bool:
    if dataset is None:
        return False
    head_img = head_cam.get_latest_frame()
    left_img = left_cam.get_latest_frame()
    right_img = right_cam.get_latest_frame()
    if head_img is None or left_img is None or right_img is None:
        return False

    left_state = np.asarray(left_robot.get_joint_pos(), dtype=np.float32)
    right_state = np.asarray(right_robot.get_joint_pos(), dtype=np.float32)
    state_vec = np.concatenate([left_state, right_state]).astype(np.float32)
    action_vec = np.concatenate([left_cmd, right_cmd]).astype(np.float32)

    dataset.add_frame(
        {
            "observation.images.head": head_img,
            "observation.images.left_wrist": left_img,
            "observation.images.right_wrist": right_img,
            "observation.state": state_vec,
            "action": action_vec,
            "task": args.task,
        }
    )
    return True


def _parse_control_buttons(buttons: list[int], swap_devices: bool) -> tuple[bool, bool]:
    buttons = list(buttons)
    while len(buttons) < 2:
        buttons.append(0)
    return bool(buttons[0]), bool(buttons[1])


def _print_controls() -> None:
    print("\n" + "=" * 72)
    print("I2RT POLICY + SPACEMOUSE + LEROBOT RECORDING")
    print("=" * 72)
    print("  Right SpaceMouse Button 1: Start policy motion + begin episode recording")
    print("  Right SpaceMouse Button 2: Save episode if recording + reset to ready pose")
    print("  Ctrl+C                   : Emergency stop")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="I2RT dual-arm policy client with SpaceMouse control and LeRobot recording."
    )

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
    parser.add_argument("--gripper-backoff", type=float, default=0.1)

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
    parser.add_argument("--allow-camera-fallback", action="store_true")

    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--no-resize", action="store_true")

    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Append new episodes to an existing local dataset root.")
    parser.add_argument("--task", type=str, default="policy evaluation")
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--video-encode-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")

    parser.add_argument("--list-spacemouse", action="store_true")
    parser.add_argument("--left-path", type=str, default=None)
    parser.add_argument("--right-path", type=str, default=None)
    parser.add_argument("--swap-devices", action="store_true")

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

    if args.list_spacemouse:
        list_spacemouse_receivers()
        return

    if not args.repo_id:
        raise RuntimeError("--repo-id is required for recording. Example: --repo-id data/my_eval_run.")
    if args.prompt is None:
        args.prompt = args.task

    dataset_fps = int(round(args.frequency))
    if abs(args.frequency - dataset_fps) > 1e-3:
        print(f"Warning: --frequency {args.frequency} is not integer; dataset fps set to {dataset_fps}.")

    if not args.skip_preflight:
        ok = run_preflight_checks(args, dataset_fps)
        if args.preflight_only:
            return
        if not ok:
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

    head_cam = FixedControlRealSenseStream(
        _build_realsense_config(
            args.head_serial,
            args.head_width,
            args.head_height,
            args.head_fps,
            args,
        )
    )
    left_cam = FixedControlRealSenseStream(
        _build_realsense_config(
            args.left_wrist_serial,
            args.wrist_width,
            args.wrist_height,
            args.wrist_fps,
            args,
        )
    )
    right_cam = FixedControlRealSenseStream(
        _build_realsense_config(
            args.right_wrist_serial,
            args.wrist_width,
            args.wrist_height,
            args.wrist_fps,
            args,
        )
    )

    left_robot = None
    right_robot = None
    spacemouse = None
    dataset = None
    policy = None
    left_home = None
    right_home = None
    recording = False
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

        ready_left = _build_ready_open_qpos(
            left_robot, np.array(args.ready_qpos, dtype=float), left_gripper_index, left_gripper_limits, args.left_gripper_invert
        )
        ready_right = _build_ready_open_qpos(
            right_robot,
            np.array(args.ready_qpos, dtype=float),
            right_gripper_index,
            right_gripper_limits,
            args.right_gripper_invert,
        )

        print("Moving to ready pose (gripper open)...")
        left_robot.move_joints(ready_left, time_interval_s=args.home_time)
        right_robot.move_joints(ready_right, time_interval_s=args.home_time)
        last_left_cmd = ready_left.copy()
        last_right_cmd = ready_right.copy()

        head_frame, left_frame, right_frame = _wait_first_frames(head_cam, left_cam, right_cam)
        head_height, head_width = head_frame.shape[:2]
        left_height, left_width = left_frame.shape[:2]
        right_height, right_width = right_frame.shape[:2]

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

        left_joint_names = build_joint_names("left", left_robot.num_dofs(), left_gripper_index)
        right_joint_names = build_joint_names("right", right_robot.num_dofs(), right_gripper_index)
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

        spacemouse = SpaceMouseButtonReader(
            left_path=args.left_path,
            right_path=args.right_path,
            max_devices=2,
            swap_devices=args.swap_devices,
        )
        if spacemouse.device_count == 0:
            raise RuntimeError("Failed to open any SpaceMouse device.")

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

        _print_controls()
        print("Ready pose reached. Press RIGHT SpaceMouse button 1 to start recording.")

        step_time = 1.0 / args.frequency if args.frequency > 0 else 0.0
        last_cam_warn_time = 0.0
        last_action_warn_time = 0.0
        last_left_gripper_warn_time = 0.0
        last_right_gripper_warn_time = 0.0
        last_left_gripper_print_time = 0.0
        last_right_gripper_print_time = 0.0

        teleop_enabled = False
        recording = False
        episode_start_time = None
        last_right_button1 = False
        last_right_button2 = False
        step = 0

        def _reset_to_ready_pose(status: str) -> None:
            nonlocal last_left_cmd, last_right_cmd
            teleop_note = (
                "Press RIGHT SpaceMouse button 1 to start recording."
                if args.num_episodes <= 0 or dataset.meta.total_episodes < args.num_episodes
                else "Requested number of episodes reached."
            )
            print(status)
            left_robot.move_joints(ready_left, time_interval_s=args.home_time)
            right_robot.move_joints(ready_right, time_interval_s=args.home_time)
            last_left_cmd = ready_left.copy()
            last_right_cmd = ready_right.copy()
            left_limiter.gripper_blocked = False
            right_limiter.gripper_blocked = False
            policy.reset()
            print(f"Ready pose reached. {teleop_note}")

        while not stop_event.is_set():
            step_start = time.time()
            buttons = spacemouse.get_buttons()
            right_button1, right_button2 = _parse_control_buttons(buttons, args.swap_devices)
            start_rising = right_button1 and not last_right_button1
            reset_rising = right_button2 and not last_right_button2

            last_right_button1 = right_button1
            last_right_button2 = right_button2

            if reset_rising:
                if recording:
                    teleop_enabled = False
                    recording = False
                    episode_start_time = None
                    saved = maybe_save_episode(dataset)
                    if saved:
                        print(f"Episode {dataset.meta.total_episodes} saved.")
                    else:
                        print("No frames recorded; nothing to save.")
                        discard_current_episode(dataset)
                    _reset_to_ready_pose("Right SpaceMouse button 2 pressed. Saving and resetting.")
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        break
                else:
                    teleop_enabled = False
                    _reset_to_ready_pose("Right SpaceMouse button 2 pressed. Resetting.")
                continue

            if not teleop_enabled:
                if start_rising:
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        print("Reached requested number of episodes; start ignored.")
                        continue
                    teleop_enabled = True
                    recording = True
                    episode_start_time = time.monotonic()
                    policy.reset()
                    next_ep = dataset.meta.total_episodes + 1
                    if args.num_episodes > 0:
                        print(f"Recording episode {next_ep}/{args.num_episodes}.")
                    else:
                        print(f"Recording episode {next_ep}.")
                else:
                    if step_time > 0:
                        elapsed = time.time() - step_start
                        if elapsed < step_time:
                            time.sleep(step_time - elapsed)
                    else:
                        time.sleep(0.01)
                    continue

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

            obs, _left_state, _right_state = obs_pack
            action = policy.infer(obs)
            action_vec = np.asarray(action.get("actions", []), dtype=float).reshape(-1)

            left_dofs = left_robot.num_dofs()
            right_dofs = right_robot.num_dofs()
            expected = left_dofs + right_dofs
            if action_vec.size != expected:
                now = time.monotonic()
                if now - last_action_warn_time > 2.0:
                    print(f"Action dim mismatch (got {action_vec.size}, expected {expected}); holding position.")
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
                    print(f"Left gripper eff={left_eff:.3f} pos={left_pos_label} target={left_goal_label}")
                    last_left_gripper_print_time = now
                if right_eff is not None and now - last_right_gripper_print_time > args.gripper_force_print_interval:
                    right_pos_label = f"{right_gripper_pos:.4f}" if right_gripper_pos is not None else "n/a"
                    right_goal_label = f"{right_gripper_goal:.4f}" if right_gripper_goal is not None else "n/a"
                    print(f"Right gripper eff={right_eff:.3f} pos={right_pos_label} target={right_goal_label}")
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

            if recording:
                frame_added = add_record_frame(
                    dataset,
                    args,
                    head_cam,
                    left_cam,
                    right_cam,
                    left_robot,
                    right_robot,
                    left_cmd,
                    right_cmd,
                )
                if not frame_added:
                    now = time.monotonic()
                    if now - last_cam_warn_time > 2.0:
                        print("Recording paused: waiting for camera frames...")
                        last_cam_warn_time = now

                if args.episode_time > 0 and episode_start_time is not None:
                    if time.monotonic() - episode_start_time >= args.episode_time:
                        teleop_enabled = False
                        recording = False
                        episode_start_time = None
                        saved = maybe_save_episode(dataset)
                        if saved:
                            print(f"Episode {dataset.meta.total_episodes} saved (episode time reached).")
                        else:
                            print("No frames recorded; nothing to save.")
                            discard_current_episode(dataset)
                        _reset_to_ready_pose("Episode time reached. Resetting to ready pose.")
                        if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                            break
                        continue

            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break

            if step_time > 0:
                elapsed = time.time() - step_start
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

    except KeyboardInterrupt:
        print("Emergency stop requested. Shutting down...")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if dataset is not None and recording:
                print("Saving partial episode before shutdown...")
                saved = maybe_save_episode(dataset)
                if saved:
                    print(f"Episode {dataset.meta.total_episodes} saved.")
                else:
                    discard_current_episode(dataset)
            if (
                not args.no_home
                and left_robot is not None
                and right_robot is not None
                and left_home is not None
                and right_home is not None
            ):
                print("Returning to home pose...")
                left_robot.move_joints(left_home, time_interval_s=args.home_time)
                right_robot.move_joints(right_home, time_interval_s=args.home_time)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if left_robot is not None:
                left_robot.close()
            if right_robot is not None:
                right_robot.close()
            if spacemouse is not None:
                spacemouse.close()
            head_cam.stop()
            left_cam.stop()
            right_cam.stop()
            if dataset is not None:
                dataset.stop_image_writer()
            print("Shutdown complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
