#!/usr/bin/env python3
"""Wine-pour guardrail wrapper for ``teleop_craft.py``.

Failure cases this script targets:
- VR tracking dropouts or hand jumps producing sudden IK targets.
- Left gripper tilting the bottle before it is above the cup.
- Right CRAFT hand rolling the cup enough for liquid to spill.
- Bottle/cup collision or table contact during pickup and put-down.
- Over-closing or side-splaying the dexterous hand while grasping a cup.

The base teleop script still owns hardware setup, recording, IK, and raw data
logging. This wrapper imports it, patches the retargeter output, and then runs
the original main loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "teleop_craft.py"
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unit(vec: np.ndarray, min_norm: float = 1e-9) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < min_norm:
        return None
    return np.asarray(vec, dtype=np.float64) / norm


def _optional_float(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in {"none", "nan", "off", "disable", "disabled"}:
        return None
    return float(text)


def _as_vec3(values: tuple[float, float, float] | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected three values, got shape {arr.shape}")
    return arr


def _axis_vector(axis: str) -> np.ndarray | None:
    if axis == "auto":
        return None
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[1:] if axis.startswith("-") else axis
    index = {"x": 0, "y": 1, "z": 2}[name]
    vec = np.zeros(3, dtype=np.float64)
    vec[index] = sign
    return vec


def _format_axis(axis: np.ndarray | None) -> str:
    if axis is None:
        return "auto"
    labels = ("x", "y", "z")
    index = int(np.argmax(np.abs(axis)))
    sign = "-" if axis[index] < 0.0 else ""
    return f"{sign}{labels[index]}"


def _pose_axis_closest_to_up(pose: np.ndarray) -> np.ndarray:
    rot = np.asarray(pose, dtype=np.float64)[:3, :3]
    candidates = (
        _axis_vector("x"),
        _axis_vector("-x"),
        _axis_vector("y"),
        _axis_vector("-y"),
        _axis_vector("z"),
        _axis_vector("-z"),
    )
    valid = [axis for axis in candidates if axis is not None]
    return max(valid, key=lambda axis: float(np.dot(rot @ axis, WORLD_UP))).copy()


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = _unit(source)
    dst = _unit(target)
    if src is None or dst is None:
        return np.eye(3, dtype=np.float64)

    dot = _clamp(float(np.dot(src, dst)), -1.0, 1.0)
    cross = np.cross(src, dst)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(src, fallback))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _unit(np.cross(src, fallback))
        assert axis is not None
        return _axis_angle(axis, math.pi)

    axis = cross / cross_norm
    angle = math.atan2(cross_norm, dot)
    return _axis_angle(axis, angle)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    rot = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T
    trace = float(np.trace(rot))
    return math.acos(_clamp((trace - 1.0) * 0.5, -1.0, 1.0))


def _limit_rotation_step(previous: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return target
    delta = np.asarray(target, dtype=np.float64) @ np.asarray(previous, dtype=np.float64).T
    angle = _rotation_angle(target, previous)
    if angle <= max_step or angle < 1e-9:
        return target
    axis = _unit(np.asarray([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]))
    if axis is None:
        return previous
    return _axis_angle(axis, max_step) @ previous


def _limit_translation_step(previous: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return target
    delta = np.asarray(target, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm < 1e-12:
        return target
    return np.asarray(previous, dtype=np.float64) + delta * (max_step / norm)


def _tool_point(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return pose[:3, 3] + pose[:3, :3] @ offset


def _tilt_angle(pose: np.ndarray, local_axis: np.ndarray) -> float:
    axis_world = pose[:3, :3] @ local_axis
    axis_world = _unit(axis_world)
    if axis_world is None:
        return 0.0
    return math.acos(_clamp(float(np.dot(axis_world, WORLD_UP)), -1.0, 1.0))


def _cap_tilt(pose: np.ndarray, local_axis: np.ndarray, max_tilt_rad: float) -> tuple[np.ndarray, bool]:
    max_tilt_rad = max(0.0, float(max_tilt_rad))
    out = np.asarray(pose, dtype=np.float64).copy()
    axis_world = _unit(out[:3, :3] @ local_axis)
    if axis_world is None:
        return out, False

    dot = _clamp(float(np.dot(axis_world, WORLD_UP)), -1.0, 1.0)
    angle = math.acos(dot)
    if angle <= max_tilt_rad:
        return out, False

    planar = axis_world - dot * WORLD_UP
    planar_unit = _unit(planar)
    if planar_unit is None:
        limited_axis = WORLD_UP.copy()
    else:
        limited_axis = math.cos(max_tilt_rad) * WORLD_UP + math.sin(max_tilt_rad) * planar_unit
    correction = _rotation_between_vectors(axis_world, limited_axis)
    out[:3, :3] = correction @ out[:3, :3]
    return out, True


@dataclass(frozen=True)
class GuardrailBounds:
    x: tuple[float | None, float | None] = (None, None)
    y: tuple[float | None, float | None] = (None, None)
    z: tuple[float | None, float | None] = (0.085, 0.72)

    @classmethod
    def from_flat(cls, values: tuple[float | None, ...] | list[float | None]) -> "GuardrailBounds":
        if len(values) != 6:
            raise ValueError("bounds need six values: x_min x_max y_min y_max z_min z_max")
        return cls(x=(values[0], values[1]), y=(values[2], values[3]), z=(values[4], values[5]))

    def clamp_pose(self, pose: np.ndarray) -> tuple[np.ndarray, bool]:
        out = np.asarray(pose, dtype=np.float64).copy()
        changed = False
        for idx, (low, high) in enumerate((self.x, self.y, self.z)):
            value = float(out[idx, 3])
            if low is not None and value < low:
                out[idx, 3] = low
                changed = True
            if high is not None and value > high:
                out[idx, 3] = high
                changed = True
        return out, changed


@dataclass(frozen=True)
class WinePourGuardrailConfig:
    enabled: bool = True
    task_defaults: bool = True
    left_bounds: GuardrailBounds = field(default_factory=GuardrailBounds)
    right_bounds: GuardrailBounds = field(default_factory=GuardrailBounds)
    cup_upright_axis: np.ndarray | None = None
    bottle_upright_axis: np.ndarray | None = None
    cup_upright_max_rad: float = math.radians(18.0)
    cup_carry_max_tilt_rad: float = math.radians(85.0)
    bottle_carry_max_tilt_rad: float = math.radians(90.0)
    bottle_pour_max_tilt_rad: float = math.radians(135.0)
    pour_start_tilt_rad: float = math.radians(36.0)
    bottle_pour_tilt_boost_gain: float = 1.85
    bottle_pour_tilt_boost_start_rad: float = math.radians(28.0)
    require_pour_alignment: bool = False
    lock_cup_during_pour: bool = True
    align_assist: bool = True
    align_assist_start_rad: float = math.radians(32.0)
    align_assist_full_rad: float = math.radians(78.0)
    align_assist_strength: float = 0.65
    align_assist_target_height_m: float = 0.16
    align_assist_max_correction_m: float = 0.18
    align_radius_m: float = 0.13
    pour_min_height_m: float = 0.06
    pour_max_height_m: float = 0.44
    min_ee_distance_m: float = 0.105
    max_translation_speed_mps: float = 0.32
    max_angular_speed_radps: float = 2.60
    bottle_mouth_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    cup_rim_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    craft_grip_max_fraction: float = 0.68
    craft_thumb_max_fraction: float = 0.58
    craft_side_max_fraction: float = 0.22


class WinePourGuardrail:
    def __init__(self, config: WinePourGuardrailConfig) -> None:
        self.config = config
        self.previous: dict[str, np.ndarray] = {}
        self._learned_axes: dict[str, np.ndarray] = {}

    def reset(self, left_pose: np.ndarray | None, right_pose: np.ndarray | None) -> None:
        self.previous.clear()
        self._learned_axes.clear()
        if left_pose is not None:
            self.previous["left"] = np.asarray(left_pose, dtype=np.float64).copy()
            if self.config.bottle_upright_axis is None:
                self._learned_axes["left"] = _pose_axis_closest_to_up(left_pose)
        if right_pose is not None:
            self.previous["right"] = np.asarray(right_pose, dtype=np.float64).copy()
            if self.config.cup_upright_axis is None:
                self._learned_axes["right"] = _pose_axis_closest_to_up(right_pose)

    def _upright_axis(self, side: str) -> np.ndarray:
        configured = self.config.bottle_upright_axis if side == "left" else self.config.cup_upright_axis
        if configured is not None:
            return configured
        learned = self._learned_axes.get(side)
        if learned is not None:
            return learned
        previous = self.previous.get(side)
        if previous is not None:
            learned = _pose_axis_closest_to_up(previous)
            self._learned_axes[side] = learned
            return learned
        fallback = _axis_vector("y")
        assert fallback is not None
        return fallback

    def filter_output(self, output: Any, dt: float) -> Any:
        if not self.config.enabled:
            return output

        left = None if output.left_pose is None else np.asarray(output.left_pose, dtype=np.float64).copy()
        right = None if output.right_pose is None else np.asarray(output.right_pose, dtype=np.float64).copy()
        reasons: dict[str, list[str]] = {"left": [], "right": []}
        if left is not None:
            left = self._boost_bottle_pour_tilt(left, reasons)

        pour_requested = self._pour_requested(left)
        pour_aligned = left is not None and right is not None and self._pour_geometry_ok(left, right)

        if right is not None:
            cup_max_tilt = self.config.cup_carry_max_tilt_rad
            if self.config.lock_cup_during_pour and pour_requested and pour_aligned:
                cup_max_tilt = self.config.cup_upright_max_rad
            right, changed = _cap_tilt(right, self._upright_axis("right"), cup_max_tilt)
            if changed:
                reasons["right"].append("cup_upright")

        if left is not None:
            max_bottle_tilt = self._allowed_bottle_tilt(left, right, reasons)
            left, changed = _cap_tilt(left, self._upright_axis("left"), max_bottle_tilt)
            if changed:
                reasons["left"].append("bottle_tilt_gate")

        if left is not None and right is not None:
            left = self._assist_alignment(left, right, reasons)

        if left is not None:
            left, changed = self.config.left_bounds.clamp_pose(left)
            if changed:
                reasons["left"].append("workspace")
        if right is not None:
            right, changed = self.config.right_bounds.clamp_pose(right)
            if changed:
                reasons["right"].append("workspace")

        left, right = self._separate_arms(left, right, reasons)

        left = self._rate_limit("left", left, dt, reasons)
        right = self._rate_limit("right", right, dt, reasons)

        if left is not None:
            output.left_pose = left
            self.previous["left"] = left.copy()
        if right is not None:
            output.right_pose = right
            self.previous["right"] = right.copy()
        self._update_diagnostics(output, left, right, reasons)
        return output

    def _pour_requested(self, left_pose: np.ndarray | None) -> bool:
        return (
            left_pose is not None
            and _tilt_angle(left_pose, self._upright_axis("left")) >= self.config.pour_start_tilt_rad
        )

    def _boost_bottle_pour_tilt(
        self,
        left_pose: np.ndarray,
        reasons: dict[str, list[str]],
    ) -> np.ndarray:
        gain = max(1.0, float(self.config.bottle_pour_tilt_boost_gain))
        if gain <= 1.0:
            return left_pose
        axis = self._upright_axis("left")
        tilt = _tilt_angle(left_pose, axis)
        start = self.config.bottle_pour_tilt_boost_start_rad
        if tilt <= start:
            return left_pose

        boosted_tilt = start + (tilt - start) * gain
        boosted_tilt = min(boosted_tilt, self.config.bottle_pour_max_tilt_rad)
        boosted, changed = _cap_tilt(left_pose, axis, boosted_tilt)
        if changed:
            # _cap_tilt only reduces tilt, so build the intended larger tilt explicitly.
            boosted = self._set_tilt(left_pose, axis, boosted_tilt)
            reasons["left"].append("pour_tilt_boost")
        else:
            boosted = self._set_tilt(left_pose, axis, boosted_tilt)
            if not np.allclose(boosted[:3, :3], left_pose[:3, :3], atol=1e-9):
                reasons["left"].append("pour_tilt_boost")
        return boosted

    @staticmethod
    def _set_tilt(pose: np.ndarray, local_axis: np.ndarray, target_tilt: float) -> np.ndarray:
        out = np.asarray(pose, dtype=np.float64).copy()
        axis_world = _unit(out[:3, :3] @ local_axis)
        if axis_world is None:
            return out
        dot = _clamp(float(np.dot(axis_world, WORLD_UP)), -1.0, 1.0)
        planar = axis_world - dot * WORLD_UP
        planar_unit = _unit(planar)
        if planar_unit is None:
            return out
        target_axis = math.cos(target_tilt) * WORLD_UP + math.sin(target_tilt) * planar_unit
        correction = _rotation_between_vectors(axis_world, target_axis)
        out[:3, :3] = correction @ out[:3, :3]
        return out

    def _assist_alignment(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        reasons: dict[str, list[str]],
    ) -> np.ndarray:
        if not self.config.align_assist:
            return left_pose
        tilt = _tilt_angle(left_pose, self._upright_axis("left"))
        if tilt <= self.config.align_assist_start_rad:
            return left_pose

        transition = max(1e-6, self.config.align_assist_full_rad - self.config.align_assist_start_rad)
        amount = _clamp((tilt - self.config.align_assist_start_rad) / transition, 0.0, 1.0)
        amount *= _clamp(self.config.align_assist_strength, 0.0, 1.0)
        if amount <= 0.0:
            return left_pose

        bottle_point = _tool_point(left_pose, self.config.bottle_mouth_offset_m)
        cup_point = _tool_point(right_pose, self.config.cup_rim_offset_m)
        desired_bottle_point = cup_point + np.asarray(
            [0.0, 0.0, self.config.align_assist_target_height_m],
            dtype=np.float64,
        )
        correction = desired_bottle_point - bottle_point
        max_correction = max(0.0, self.config.align_assist_max_correction_m)
        norm = float(np.linalg.norm(correction))
        if norm > max_correction > 0.0:
            correction = correction * (max_correction / norm)
        out = left_pose.copy()
        out[:3, 3] = out[:3, 3] + amount * correction
        reasons["left"].append("align_assist")
        return out

    def _allowed_bottle_tilt(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray | None,
        reasons: dict[str, list[str]],
    ) -> float:
        requested_tilt = _tilt_angle(left_pose, self._upright_axis("left"))
        if requested_tilt < self.config.pour_start_tilt_rad:
            return self.config.bottle_carry_max_tilt_rad
        if not self.config.require_pour_alignment:
            return self.config.bottle_pour_max_tilt_rad
        if right_pose is None:
            reasons["left"].append("pour_blocked_no_cup")
            return self.config.bottle_carry_max_tilt_rad

        bottle_point = _tool_point(left_pose, self.config.bottle_mouth_offset_m)
        cup_point = _tool_point(right_pose, self.config.cup_rim_offset_m)
        aligned, height_ok = self._pour_geometry_flags(bottle_point, cup_point)
        if not aligned:
            reasons["left"].append("pour_blocked_align")
        if not height_ok:
            reasons["left"].append("pour_blocked_height")
        return self.config.bottle_pour_max_tilt_rad if aligned and height_ok else self.config.bottle_carry_max_tilt_rad

    def _pour_geometry_ok(self, left_pose: np.ndarray, right_pose: np.ndarray) -> bool:
        bottle_point = _tool_point(left_pose, self.config.bottle_mouth_offset_m)
        cup_point = _tool_point(right_pose, self.config.cup_rim_offset_m)
        return all(self._pour_geometry_flags(bottle_point, cup_point))

    def _pour_geometry_flags(self, bottle_point: np.ndarray, cup_point: np.ndarray) -> tuple[bool, bool]:
        lateral = float(np.linalg.norm((bottle_point - cup_point)[:2]))
        height = float(bottle_point[2] - cup_point[2])
        aligned = lateral <= self.config.align_radius_m
        height_ok = self.config.pour_min_height_m <= height <= self.config.pour_max_height_m
        return aligned, height_ok

    def _separate_arms(
        self,
        left: np.ndarray | None,
        right: np.ndarray | None,
        reasons: dict[str, list[str]],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if left is None or right is None or self.config.min_ee_distance_m <= 0.0:
            return left, right

        delta = left[:3, 3] - right[:3, 3]
        distance = float(np.linalg.norm(delta))
        if distance >= self.config.min_ee_distance_m:
            return left, right

        direction = _unit(delta)
        if direction is None:
            prev_left = self.previous.get("left")
            prev_right = self.previous.get("right")
            if prev_left is not None and prev_right is not None:
                direction = _unit(prev_left[:3, 3] - prev_right[:3, 3])
        if direction is None:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

        center = 0.5 * (left[:3, 3] + right[:3, 3])
        left = left.copy()
        right = right.copy()
        left[:3, 3] = center + 0.5 * self.config.min_ee_distance_m * direction
        right[:3, 3] = center - 0.5 * self.config.min_ee_distance_m * direction
        left, _ = self.config.left_bounds.clamp_pose(left)
        right, _ = self.config.right_bounds.clamp_pose(right)
        reasons["left"].append("ee_separation")
        reasons["right"].append("ee_separation")
        return left, right

    def _rate_limit(
        self,
        side: str,
        pose: np.ndarray | None,
        dt: float,
        reasons: dict[str, list[str]],
    ) -> np.ndarray | None:
        if pose is None:
            return None
        previous = self.previous.get(side)
        if previous is None:
            return pose

        out = pose.copy()
        max_translation_step = self.config.max_translation_speed_mps * max(float(dt), 1e-4)
        limited_pos = _limit_translation_step(previous[:3, 3], out[:3, 3], max_translation_step)
        if not np.allclose(limited_pos, out[:3, 3], atol=1e-9):
            out[:3, 3] = limited_pos
            reasons[side].append("guardrail_xyz_rate")

        max_rotation_step = self.config.max_angular_speed_radps * max(float(dt), 1e-4)
        limited_rot = _limit_rotation_step(previous[:3, :3], out[:3, :3], max_rotation_step)
        if not np.allclose(limited_rot, out[:3, :3], atol=1e-9):
            out[:3, :3] = limited_rot
            reasons[side].append("guardrail_rot_rate")
        return out

    def _update_diagnostics(
        self,
        output: Any,
        left: np.ndarray | None,
        right: np.ndarray | None,
        reasons: dict[str, list[str]],
    ) -> None:
        for side, pose in (("left", left), ("right", right)):
            diag = output.diagnostics.get(side)
            if diag is None:
                continue
            if pose is not None:
                diag.target_xyz = pose[:3, 3].copy()
                diag.filtered_translation = pose[:3, 3].copy()
            unique = tuple(dict.fromkeys(reasons[side]))
            if not unique:
                continue
            suffix = "guardrail:" + ",".join(unique)
            diag.reason = suffix if diag.reason in {"", "ok"} else f"{diag.reason}+{suffix}"


def build_guardrail_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-guardrail", dest="guardrail_enabled", action="store_false", default=True)
    parser.add_argument(
        "--no-guardrail-task-defaults",
        dest="guardrail_task_defaults",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--guardrail-left-bounds",
        type=_optional_float,
        nargs=6,
        default=[None, None, None, None, 0.085, 0.72],
    )
    parser.add_argument(
        "--guardrail-right-bounds",
        type=_optional_float,
        nargs=6,
        default=[None, None, None, None, 0.085, 0.72],
    )
    parser.add_argument(
        "--guardrail-cup-upright-axis",
        choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
        default="auto",
    )
    parser.add_argument(
        "--guardrail-bottle-upright-axis",
        choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
        default="auto",
    )
    parser.add_argument("--guardrail-cup-upright-max-deg", type=float, default=18.0)
    parser.add_argument("--guardrail-cup-carry-max-tilt-deg", type=float, default=85.0)
    parser.add_argument("--guardrail-bottle-carry-max-tilt-deg", type=float, default=90.0)
    parser.add_argument("--guardrail-bottle-pour-max-tilt-deg", type=float, default=135.0)
    parser.add_argument("--guardrail-pour-start-tilt-deg", type=float, default=36.0)
    parser.add_argument("--guardrail-bottle-pour-tilt-boost-gain", type=float, default=1.85)
    parser.add_argument("--guardrail-bottle-pour-tilt-boost-start-deg", type=float, default=28.0)
    parser.add_argument(
        "--guardrail-require-pour-alignment",
        action="store_true",
        help="Block large bottle tilt unless the configured bottle/cup tool points are aligned.",
    )
    parser.add_argument(
        "--no-guardrail-lock-cup-during-pour",
        dest="guardrail_lock_cup_during_pour",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-guardrail-align-assist",
        dest="guardrail_align_assist",
        action="store_false",
        default=True,
    )
    parser.add_argument("--guardrail-align-assist-start-deg", type=float, default=32.0)
    parser.add_argument("--guardrail-align-assist-full-deg", type=float, default=78.0)
    parser.add_argument("--guardrail-align-assist-strength", type=float, default=0.65)
    parser.add_argument("--guardrail-align-assist-target-height", type=float, default=0.16)
    parser.add_argument("--guardrail-align-assist-max-correction", type=float, default=0.18)
    parser.add_argument("--guardrail-align-radius", type=float, default=0.13)
    parser.add_argument("--guardrail-pour-min-height", type=float, default=0.06)
    parser.add_argument("--guardrail-pour-max-height", type=float, default=0.44)
    parser.add_argument("--guardrail-min-ee-distance", type=float, default=0.105)
    parser.add_argument("--guardrail-max-translation-speed", type=float, default=0.32)
    parser.add_argument("--guardrail-max-angular-speed", type=float, default=2.60)
    parser.add_argument("--guardrail-bottle-mouth-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--guardrail-cup-rim-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--guardrail-craft-grip-max", type=float, default=0.68)
    parser.add_argument("--guardrail-craft-thumb-max", type=float, default=0.58)
    parser.add_argument("--guardrail-craft-side-max", type=float, default=0.22)
    parser.add_argument("--guardrail-help", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> WinePourGuardrailConfig:
    return WinePourGuardrailConfig(
        enabled=bool(args.guardrail_enabled),
        task_defaults=bool(args.guardrail_task_defaults),
        left_bounds=GuardrailBounds.from_flat(args.guardrail_left_bounds),
        right_bounds=GuardrailBounds.from_flat(args.guardrail_right_bounds),
        cup_upright_axis=_axis_vector(args.guardrail_cup_upright_axis),
        bottle_upright_axis=_axis_vector(args.guardrail_bottle_upright_axis),
        cup_upright_max_rad=math.radians(max(0.0, args.guardrail_cup_upright_max_deg)),
        cup_carry_max_tilt_rad=math.radians(max(0.0, args.guardrail_cup_carry_max_tilt_deg)),
        bottle_carry_max_tilt_rad=math.radians(max(0.0, args.guardrail_bottle_carry_max_tilt_deg)),
        bottle_pour_max_tilt_rad=math.radians(max(0.0, args.guardrail_bottle_pour_max_tilt_deg)),
        pour_start_tilt_rad=math.radians(max(0.0, args.guardrail_pour_start_tilt_deg)),
        bottle_pour_tilt_boost_gain=max(1.0, args.guardrail_bottle_pour_tilt_boost_gain),
        bottle_pour_tilt_boost_start_rad=math.radians(
            max(0.0, args.guardrail_bottle_pour_tilt_boost_start_deg)
        ),
        require_pour_alignment=bool(args.guardrail_require_pour_alignment),
        lock_cup_during_pour=bool(args.guardrail_lock_cup_during_pour),
        align_assist=bool(args.guardrail_align_assist),
        align_assist_start_rad=math.radians(max(0.0, args.guardrail_align_assist_start_deg)),
        align_assist_full_rad=math.radians(max(0.0, args.guardrail_align_assist_full_deg)),
        align_assist_strength=_clamp(args.guardrail_align_assist_strength, 0.0, 1.0),
        align_assist_target_height_m=float(args.guardrail_align_assist_target_height),
        align_assist_max_correction_m=max(0.0, args.guardrail_align_assist_max_correction),
        align_radius_m=max(0.0, args.guardrail_align_radius),
        pour_min_height_m=float(args.guardrail_pour_min_height),
        pour_max_height_m=float(args.guardrail_pour_max_height),
        min_ee_distance_m=max(0.0, args.guardrail_min_ee_distance),
        max_translation_speed_mps=max(0.0, args.guardrail_max_translation_speed),
        max_angular_speed_radps=max(0.0, args.guardrail_max_angular_speed),
        bottle_mouth_offset_m=_as_vec3(args.guardrail_bottle_mouth_offset),
        cup_rim_offset_m=_as_vec3(args.guardrail_cup_rim_offset),
        craft_grip_max_fraction=_clamp(args.guardrail_craft_grip_max, 0.0, 1.0),
        craft_thumb_max_fraction=_clamp(args.guardrail_craft_thumb_max, 0.0, 1.0),
        craft_side_max_fraction=_clamp(args.guardrail_craft_side_max, 0.0, 1.0),
    )


def _cli_flags(argv: list[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in argv if token.startswith("--")}


def _apply_task_defaults(args: argparse.Namespace, base_argv: list[str], config: WinePourGuardrailConfig) -> None:
    if not config.enabled or not config.task_defaults:
        return
    flags = _cli_flags(base_argv)
    defaults = {
        "pos_scale": (0.90, "--pos-scale"),
        "translation_alpha": (0.65, "--translation-alpha"),
        "rotation_alpha": (0.75, "--rotation-alpha"),
        "max_target_translation_speed": (0.36, "--max-target-translation-speed"),
        "max_target_angular_speed": (2.80, "--max-target-angular-speed"),
        "max_input_jump": (0.25, "--max-input-jump"),
        "max_input_rotation_jump": (1.25, "--max-input-rotation-jump"),
        "max_arm_joint_step": (0.045, "--max-arm-joint-step"),
        "max_gripper_speed": (0.55, "--max-gripper-speed"),
        "motion_scale": (0.92, "--motion-scale"),
        "thumb_scale": (1.55, "--thumb-scale"),
        "side_scale": (0.24, "--side-scale"),
        "thumb_side_scale": (0.55, "--thumb-side-scale"),
        "side_deadzone": (0.18, "--side-deadzone"),
        "max_step_raw": (56, "--max-step-raw"),
        "thumb_max_step_raw": (130, "--thumb-max-step-raw"),
        "max_velocity_raw": (950.0, "--max-velocity-raw"),
        "side_max_velocity_raw": (360.0, "--side-max-velocity-raw"),
    }
    for attr, (value, flag) in defaults.items():
        if flag not in flags and hasattr(args, attr):
            setattr(args, attr, value)


def _print_guardrail_summary(config: WinePourGuardrailConfig) -> None:
    status = "enabled" if config.enabled else "disabled"
    print(
        "wine_pour_guardrail="
        f"{status} cup_upright_max_deg={math.degrees(config.cup_upright_max_rad):.1f} "
        f"cup_axis={_format_axis(config.cup_upright_axis)} "
        f"cup_carry_max_tilt_deg={math.degrees(config.cup_carry_max_tilt_rad):.1f} "
        f"bottle_carry_max_tilt_deg={math.degrees(config.bottle_carry_max_tilt_rad):.1f} "
        f"bottle_pour_max_tilt_deg={math.degrees(config.bottle_pour_max_tilt_rad):.1f} "
        f"pour_tilt_boost_gain={config.bottle_pour_tilt_boost_gain:.2f} "
        f"bottle_axis={_format_axis(config.bottle_upright_axis)} "
        f"align_assist={config.align_assist} "
        f"require_pour_alignment={config.require_pour_alignment} "
        f"align_radius={config.align_radius_m:.3f} "
        f"pour_height=({config.pour_min_height_m:.3f},{config.pour_max_height_m:.3f}) "
        f"craft_grip_max={config.craft_grip_max_fraction:.2f}"
    )


def _patch_parse_args(base: ModuleType, config_holder: dict[str, WinePourGuardrailConfig]) -> None:
    original_parse_args = base.parse_args
    guardrail_parser = build_guardrail_parser()

    def parse_args_with_guardrails() -> argparse.Namespace:
        original_argv = sys.argv[:]
        guardrail_args, base_argv = guardrail_parser.parse_known_args(original_argv[1:])
        if guardrail_args.guardrail_help:
            guardrail_parser.print_help()
            raise SystemExit(0)

        config = config_from_args(guardrail_args)
        config_holder["config"] = config
        try:
            sys.argv = [original_argv[0], *base_argv]
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        _apply_task_defaults(args, base_argv, config)
        args.guardrail_config = config
        _print_guardrail_summary(config)
        return args

    base.parse_args = parse_args_with_guardrails


def _patch_retargeter(base: ModuleType, config_holder: dict[str, WinePourGuardrailConfig]) -> None:
    original_cls = base.QuestToYamRetargeter

    class GuardedQuestToYamRetargeter(original_cls):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any | None = None) -> None:
            super().__init__(config)
            self._wine_guardrail = WinePourGuardrail(config_holder.get("config", WinePourGuardrailConfig()))

        def calibrate(
            self,
            frame: Any,
            left_eef_pose: np.ndarray | None,
            right_eef_pose: np.ndarray | None,
            now: float | None = None,
        ) -> bool:
            ok = super().calibrate(frame, left_eef_pose, right_eef_pose, now=now)
            if ok:
                self._wine_guardrail.reset(left_eef_pose, right_eef_pose)
            return ok

        def update(self, frame: Any, dt: float, now: float | None = None) -> Any:
            output = super().update(frame, dt, now=now)
            return self._wine_guardrail.filter_output(output, dt)

    base.QuestToYamRetargeter = GuardedQuestToYamRetargeter


def _limit_bend_motor(motor_id: int, raw: int, max_fraction: float, specs: dict[int, Any], clamp_raw: Any) -> int:
    spec = specs[motor_id]
    if spec.bend_raw is None:
        return clamp_raw(motor_id, raw)
    default = float(spec.default_raw)
    bend = float(spec.bend_raw)
    limited = default + _clamp(max_fraction, 0.0, 1.0) * (bend - default)
    low = min(default, limited)
    high = max(default, limited)
    return clamp_raw(motor_id, _clamp(float(raw), low, high))


def _limit_side_motor(motor_id: int, raw: int, max_fraction: float, specs: dict[int, Any], clamp_raw: Any) -> int:
    spec = specs[motor_id]
    default = float(spec.default_raw)
    if raw >= default:
        limit = default + _clamp(max_fraction, 0.0, 1.0) * (float(spec.safe_max_raw) - default)
    else:
        limit = default - _clamp(max_fraction, 0.0, 1.0) * (default - float(spec.safe_min_raw))
    low = min(default, limit)
    high = max(default, limit)
    return clamp_raw(motor_id, _clamp(float(raw), low, high))


def guard_craft_targets(targets: dict[int, int], config: WinePourGuardrailConfig) -> dict[int, int]:
    if not config.enabled:
        return targets

    from craft_hand_helpers.motor_config import (
        FINGER_MOTORS,
        MOTOR_SPECS,
        SIDE_MOTOR_IDS,
        clamp_raw_for_motor,
    )

    thumb_motor_ids = set(FINGER_MOTORS["thumb"].values())
    guarded: dict[int, int] = {}
    for motor_id, raw in targets.items():
        if motor_id in SIDE_MOTOR_IDS:
            guarded[motor_id] = _limit_side_motor(
                motor_id,
                int(raw),
                config.craft_side_max_fraction,
                MOTOR_SPECS,
                clamp_raw_for_motor,
            )
        elif motor_id in thumb_motor_ids:
            guarded[motor_id] = _limit_bend_motor(
                motor_id,
                int(raw),
                config.craft_thumb_max_fraction,
                MOTOR_SPECS,
                clamp_raw_for_motor,
            )
        else:
            guarded[motor_id] = _limit_bend_motor(
                motor_id,
                int(raw),
                config.craft_grip_max_fraction,
                MOTOR_SPECS,
                clamp_raw_for_motor,
            )
    return guarded


def _patch_craft_retarget(base: ModuleType, config_holder: dict[str, WinePourGuardrailConfig]) -> None:
    original_retarget_craft = base.retarget_craft

    def retarget_craft_with_guardrails(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[int, int], dict[str, float], float | None]:
        targets, signals, last_seen = original_retarget_craft(*args, **kwargs)
        config = config_holder.get("config", WinePourGuardrailConfig())
        guarded_targets = guard_craft_targets(targets, config)
        if guarded_targets != targets:
            target_filter = kwargs.get("target_filter")
            if target_filter is not None:
                target_filter.previous_targets = dict(guarded_targets)
                for motor_id, raw in guarded_targets.items():
                    state = target_filter.one_euro_states.get(motor_id)
                    if state is not None:
                        state.value = float(raw)
            signals = dict(signals)
            signals["guardrail_craft_grip_max"] = config.craft_grip_max_fraction
            signals["guardrail_craft_thumb_max"] = config.craft_thumb_max_fraction
            signals["guardrail_craft_side_max"] = config.craft_side_max_fraction
        return guarded_targets, signals, last_seen

    base.retarget_craft = retarget_craft_with_guardrails


def load_base_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("i2rt_craft_hand_teleop_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base teleop script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base_module()
    config_holder: dict[str, WinePourGuardrailConfig] = {}
    _patch_parse_args(base, config_holder)
    _patch_retargeter(base, config_holder)
    _patch_craft_retarget(base, config_holder)
    base.main()


if __name__ == "__main__":
    main()
