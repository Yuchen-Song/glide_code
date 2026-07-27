#!/usr/bin/env python3
"""I2RT dual-arm policy client with SpaceMouse gating, plate guardrail, and LeRobot recording.

Controls:
- Right SpaceMouse button 1: start policy motion and begin recording an episode.
- Right SpaceMouse button 2: save current episode if recording, then reset both arms to ready pose.
- Ctrl+C: emergency stop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import signal
import time
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Optional

import numpy as np

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
from i2rt.robots.utils import I2RT_ROOT  # noqa: E402
from glide_runtime.policy_gripper import (  # noqa: E402
    FixedControlRealSenseStream,
    _build_realsense_config,
    _wait_first_frames,
    add_record_frame,
    build_image_feature,
    build_joint_names,
    create_or_resume_dataset,
    discard_current_episode,
    maybe_save_episode,
)
from lerobot.common.datasets.video_utils import encode_video_frames  # noqa: E402

if TYPE_CHECKING:
    from i2rt.robots.pink_kinematics import PinkKinematics


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


def _parse_control_buttons(buttons: list[int], swap_devices: bool) -> tuple[bool, bool]:
    buttons = list(buttons)
    while len(buttons) < 2:
        buttons.append(0)
    return bool(buttons[0]), bool(buttons[1])


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
    """Task-space filter copied from the recorder and applied to policy actions."""

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


def _arm_dofs(robot, gripper_index: Optional[int]) -> int:
    return int(gripper_index) if gripper_index is not None else int(robot.num_dofs())


def _build_kinematics(args: argparse.Namespace) -> PinkKinematics:
    from i2rt.robots.pink_kinematics import PinkKinematics

    ik_frame = args.site or args.ik_frame
    ik_dt = args.ik_dt if args.ik_dt is not None else 1.0 / max(float(args.frequency), 1e-6)
    urdf_path = os.path.join(I2RT_ROOT, "robot_models", "yam", "yam.urdf")
    return PinkKinematics(
        urdf_path,
        ik_frame,
        dt=ik_dt,
        alpha=args.ik_alpha,
        position_cost=args.ik_pos_cost,
        orientation_cost=args.ik_ori_cost,
        posture_cost=args.ik_posture_cost,
        damping_cost=args.ik_damping_cost,
        lm_damping=args.ik_lm_damping,
        gain=args.ik_gain,
        solver=args.ik_solver,
        solve_damping=args.ik_solve_damping,
    )


def _slice_arm_q(qpos: np.ndarray, arm_dofs: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=float).reshape(-1)
    if qpos.size < arm_dofs:
        return np.concatenate([qpos, np.zeros(arm_dofs - qpos.size, dtype=float)])
    return qpos[:arm_dofs].copy()


def _gripper_open_close(gripper_invert: bool) -> tuple[float, float]:
    gripper_open = 1.0
    gripper_close = 0.0
    if gripper_invert:
        gripper_open, gripper_close = gripper_close, gripper_open
    return gripper_open, gripper_close


def _seed_plate_gripper_state(cmd: np.ndarray, gripper_index: Optional[int], state: dict) -> None:
    state.clear()
    if gripper_index is None or gripper_index >= len(cmd):
        return
    state["plate_gripper_filtered_pos"] = float(cmd[gripper_index])
    state["plate_gripper_desired_pos"] = float(cmd[gripper_index])


def _limit_plate_gripper_command_rate(
    cmd: np.ndarray,
    state: dict,
    args: argparse.Namespace,
    dt: float,
    gripper_index: Optional[int],
    gripper_open: float,
    gripper_close: float,
) -> np.ndarray:
    if not args.plate_task_filter or gripper_index is None or gripper_index >= len(cmd):
        return cmd
    max_speed = max(float(args.plate_gripper_max_speed), 0.0)
    if max_speed <= 0.0:
        return cmd

    cmd = cmd.copy()
    target_pos = float(cmd[gripper_index])
    state["plate_gripper_desired_pos"] = target_pos

    close_fraction = float(np.clip(args.plate_gripper_max_close_fraction, 0.0, 1.0))
    closest_allowed_pos = gripper_open + close_fraction * (gripper_close - gripper_open)
    if gripper_close < gripper_open:
        target_pos = max(target_pos, closest_allowed_pos)
    else:
        target_pos = min(target_pos, closest_allowed_pos)

    prev_pos = state.get("plate_gripper_filtered_pos")
    if prev_pos is None:
        prev_pos = target_pos

    max_step = max_speed * max(float(dt), 0.0)
    filtered_pos = float(np.clip(target_pos, float(prev_pos) - max_step, float(prev_pos) + max_step))
    state["plate_gripper_filtered_pos"] = filtered_pos
    cmd[gripper_index] = filtered_pos
    return cmd


def _gripper_close_fraction_from_cmd(
    cmd: np.ndarray,
    gripper_index: Optional[int],
    gripper_open: float,
    gripper_close: float,
) -> float:
    if gripper_index is None or gripper_index >= len(cmd):
        return 0.0
    span = gripper_close - gripper_open
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((float(cmd[gripper_index]) - gripper_open) / span, 0.0, 1.0))


def _apply_plate_filter_to_commands(
    left_cmd: np.ndarray,
    right_cmd: np.ndarray,
    last_left_cmd: np.ndarray,
    last_right_cmd: np.ndarray,
    left_kin: PinkKinematics,
    right_kin: PinkKinematics,
    left_arm_dofs: int,
    right_arm_dofs: int,
    left_init_pose: np.ndarray,
    right_init_pose: np.ndarray,
    plate_task_filter: PlateLiftTeleopFilter,
    stabilize: bool,
    last_warn_time: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    left_target_pose = left_kin.fk(_slice_arm_q(left_cmd, left_arm_dofs))
    right_target_pose = right_kin.fk(_slice_arm_q(right_cmd, right_arm_dofs))
    left_target_pose, right_target_pose = plate_task_filter.filter_pair(
        left_target_pose,
        right_target_pose,
        left_init_pose,
        right_init_pose,
        stabilize=stabilize,
    )

    left_out = left_cmd.copy()
    right_out = right_cmd.copy()

    left_seed_q = _slice_arm_q(last_left_cmd, left_arm_dofs)
    left_success, left_q = left_kin.ik(left_target_pose, init_q=left_seed_q)
    if left_success:
        left_out[:left_arm_dofs] = left_q
    else:
        now = time.monotonic()
        if now - last_warn_time > 1.0:
            print("Left arm guardrail IK failed; holding last arm command.")
            last_warn_time = now
        left_out[:left_arm_dofs] = left_seed_q

    right_seed_q = _slice_arm_q(last_right_cmd, right_arm_dofs)
    right_success, right_q = right_kin.ik(right_target_pose, init_q=right_seed_q)
    if right_success:
        right_out[:right_arm_dofs] = right_q
    else:
        now = time.monotonic()
        if now - last_warn_time > 1.0:
            print("Right arm guardrail IK failed; holding last arm command.")
            last_warn_time = now
        right_out[:right_arm_dofs] = right_seed_q

    return left_out, right_out, last_warn_time


def _print_controls() -> None:
    print("\n" + "=" * 72)
    print("I2RT POLICY + SPACEMOUSE GATE")
    print("=" * 72)
    print("  Right SpaceMouse Button 1: Start policy motion")
    print("  Right SpaceMouse Button 2: Reset to ready pose (gripper open)")
    print("  Ctrl+C                   : Emergency stop")
    print("=" * 72 + "\n")



def _print_recording_controls() -> None:
    print("Controls:")
    print("  Right SpaceMouse button 1: start policy motion and begin recording an episode")
    print("  Right SpaceMouse button 2: save current episode if recording, then reset to ready pose")
    print("  Ctrl+C: emergency stop")

def main() -> None:
    parser = argparse.ArgumentParser(description="I2RT dual-arm policy client with SpaceMouse start/reset/kill.")

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

    parser.add_argument(
        "--plate-task-filter",
        action="store_true",
        help=(
            "Enable the recorder's bimanual plate guardrail on policy actions: FK joint targets to "
            "end-effector poses, constrain/smooth task-space motion, then IK back to joint commands."
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



    # Recording / dataset options imported from the spacemouse recorder.
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
    parser.add_argument("--dataset-fps", type=int, default=0, help="LeRobot dataset fps; defaults to --frequency.")

    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--no-resize", action="store_true")

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
        left_arm_dofs = _arm_dofs(left_robot, left_gripper_index)
        right_arm_dofs = _arm_dofs(right_robot, right_gripper_index)
        left_gripper_open, left_gripper_close = _gripper_open_close(args.left_gripper_invert)
        right_gripper_open, right_gripper_close = _gripper_open_close(args.right_gripper_invert)

        left_home = _build_home_qpos(left_robot, left_gripper_index, left_gripper_limits, args.left_gripper_invert)
        right_home = _build_home_qpos(right_robot, right_gripper_index, right_gripper_limits, args.right_gripper_invert)

        plate_task_filter = PlateLiftTeleopFilter.from_args(args)
        left_plate_kin = None
        right_plate_kin = None
        left_plate_init_pose = None
        right_plate_init_pose = None
        left_plate_gripper_state: dict[str, float] = {}
        right_plate_gripper_state: dict[str, float] = {}
        if args.plate_task_filter:
            left_plate_kin = _build_kinematics(args)
            right_plate_kin = _build_kinematics(args)
            print(
                "Plate task filter enabled: policy joint actions will be constrained in EE task space "
                "and gripper commands will be rate-limited before execution."
            )

        def _reset_plate_guardrail(left_seed_cmd: np.ndarray, right_seed_cmd: np.ndarray) -> None:
            nonlocal left_plate_init_pose, right_plate_init_pose
            if not args.plate_task_filter:
                return
            plate_task_filter.reset()
            left_plate_init_pose = left_plate_kin.fk(_slice_arm_q(left_robot.get_joint_pos(), left_arm_dofs))
            right_plate_init_pose = right_plate_kin.fk(_slice_arm_q(right_robot.get_joint_pos(), right_arm_dofs))
            _seed_plate_gripper_state(left_seed_cmd, left_gripper_index, left_plate_gripper_state)
            _seed_plate_gripper_state(right_seed_cmd, right_gripper_index, right_plate_gripper_state)

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
        _reset_plate_guardrail(last_left_cmd, last_right_cmd)

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

        dataset_fps = int(getattr(args, "dataset_fps", 0) or round(args.frequency))
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

        _print_recording_controls()
        print("Ready pose reached. Press RIGHT SpaceMouse button 1 to start recording.")

        step_time = 1.0 / args.frequency if args.frequency > 0 else 0.0
        last_cam_warn_time = 0.0
        last_action_warn_time = 0.0
        last_left_gripper_warn_time = 0.0
        last_right_gripper_warn_time = 0.0
        last_left_gripper_print_time = 0.0
        last_right_gripper_print_time = 0.0
        last_plate_ik_warn_time = 0.0

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
            _reset_plate_guardrail(last_left_cmd, last_right_cmd)
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
                    _reset_plate_guardrail(last_left_cmd, last_right_cmd)
                    next_ep = dataset.meta.total_episodes + 1
                    if args.num_episodes > 0:
                        print(f"Recording episode {next_ep}/{args.num_episodes} with plate guardrail enabled.")
                    else:
                        print(f"Recording episode {next_ep} with plate guardrail enabled.")
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

            obs, left_state, right_state = obs_pack
            action = policy.infer(obs)
            action_vec = np.asarray(action.get("actions", []), dtype=float).reshape(-1)

            left_dofs = left_robot.num_dofs()
            right_dofs = right_robot.num_dofs()
            expected = left_dofs + right_dofs
            if action_vec.size != expected:
                action_valid = False
                now = time.monotonic()
                if now - last_action_warn_time > 2.0:
                    print(f"Action dim mismatch (got {action_vec.size}, expected {expected}); holding position.")
                    last_action_warn_time = now
                left_cmd = last_left_cmd
                right_cmd = last_right_cmd
            else:
                action_valid = True
                left_cmd, right_cmd = _split_action(action_vec, left_dofs, right_dofs)

            left_cmd = _clip_gripper_if_needed(left_cmd, left_gripper_index, left_gripper_limits)
            right_cmd = _clip_gripper_if_needed(right_cmd, right_gripper_index, right_gripper_limits)

            if args.plate_task_filter and action_valid:
                plate_dt = 1.0 / max(float(args.frequency), 1e-6)
                left_cmd = _limit_plate_gripper_command_rate(
                    left_cmd,
                    left_plate_gripper_state,
                    args,
                    plate_dt,
                    left_gripper_index,
                    left_gripper_open,
                    left_gripper_close,
                )
                right_cmd = _limit_plate_gripper_command_rate(
                    right_cmd,
                    right_plate_gripper_state,
                    args,
                    plate_dt,
                    right_gripper_index,
                    right_gripper_open,
                    right_gripper_close,
                )
                left_close = _gripper_close_fraction_from_cmd(
                    left_cmd,
                    left_gripper_index,
                    left_gripper_open,
                    left_gripper_close,
                )
                right_close = _gripper_close_fraction_from_cmd(
                    right_cmd,
                    right_gripper_index,
                    right_gripper_open,
                    right_gripper_close,
                )
                stabilize_plate = (
                    left_close >= args.plate_grasp_close_threshold
                    and right_close >= args.plate_grasp_close_threshold
                )
                if left_plate_init_pose is None or right_plate_init_pose is None:
                    _reset_plate_guardrail(last_left_cmd, last_right_cmd)
                left_cmd, right_cmd, last_plate_ik_warn_time = _apply_plate_filter_to_commands(
                    left_cmd,
                    right_cmd,
                    last_left_cmd,
                    last_right_cmd,
                    left_plate_kin,
                    right_plate_kin,
                    left_arm_dofs,
                    right_arm_dofs,
                    left_plate_init_pose,
                    right_plate_init_pose,
                    plate_task_filter,
                    stabilize_plate,
                    last_plate_ik_warn_time,
                )

            left_cmd, left_eff, left_gripper_pos, left_gripper_goal, left_blocked = left_limiter.apply(left_cmd)
            right_cmd, right_eff, right_gripper_pos, right_gripper_goal, right_blocked = right_limiter.apply(right_cmd)
            if args.plate_task_filter:
                _seed_plate_gripper_state(left_cmd, left_gripper_index, left_plate_gripper_state)
                _seed_plate_gripper_state(right_cmd, right_gripper_index, right_plate_gripper_state)

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
