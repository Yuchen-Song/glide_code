"""Retarget Quest hand tracking signals into calibrated CRAFT raw ticks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .motor_config import (
    FINGER_KEYPOINTS,
    FINGER_MOTORS,
    MOTOR_SPECS,
    SIDE_MOTOR_IDS,
    SIDE_MOTORS,
    SIDE_SIGNS,
    clamp,
    clamp_raw_for_motor,
    clamp_targets_to_safe_limits,
    parse_active_fingers,
    parse_side_fingers,
    raw_defaults,
)


WEBXR25_TO_OPENPOSE21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]


@dataclass(frozen=True)
class ThumbCalibration:
    """Quest feature ranges for the current operator's right thumb.

    Values come from the May 16, 2026 Quest captures:
    side sweep, forward/back sweep, curl sweep, and the stable default hand
    baseline. Ranges use robust p5/p95 endpoints instead of raw min/max.
    """

    side_default: float = 0.830913
    side_low: float = 0.45647
    side_high: float = 0.95340
    lift_default: float = 0.199498
    lift_low: float = 0.06901
    lift_high: float = 0.81352
    base_z_default: float = 0.134977
    base_z_low: float = 0.05050
    base_z_high: float = 0.67988
    curl_default: float = 0.160799
    curl_high: float = 0.45800
    tip_palm_default: float = 1.116165
    tip_palm_curl: float = 0.52822


DEFAULT_THUMB_CALIBRATION = ThumbCalibration()


def quest_landmarks_to_openpose21(landmarks_3d: np.ndarray, source_format: str = "auto") -> np.ndarray:
    landmarks = np.asarray(landmarks_3d, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] < 3:
        raise ValueError(f"Expected hand landmarks with shape Nx3, got {landmarks.shape}")
    if source_format == "auto":
        source_format = "webxr25" if landmarks.shape[0] >= 25 else "openpose21"
    if source_format == "webxr25":
        if landmarks.shape[0] < 25:
            raise ValueError(f"webxr25 hand landmarks need at least 25 points, got {landmarks.shape[0]}")
        return landmarks[WEBXR25_TO_OPENPOSE21, :3]
    if source_format == "openpose21":
        if landmarks.shape[0] < 21:
            raise ValueError(f"openpose21 hand landmarks need at least 21 points, got {landmarks.shape[0]}")
        return landmarks[:21, :3]
    raise ValueError(f"Unsupported hand landmark format: {source_format}")


def unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return vector / norm


def palm_axes(keypoints_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if keypoints_3d.shape[0] < 21:
        return None
    x_axis = unit_vector(keypoints_3d[5] - keypoints_3d[13])
    y_axis = unit_vector(keypoints_3d[9] - keypoints_3d[0])
    if x_axis is None or y_axis is None:
        return None
    z_axis = unit_vector(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    x_axis = unit_vector(np.cross(y_axis, z_axis))
    if x_axis is None:
        return None
    return x_axis, y_axis


def side_deadzone(value: float, deadzone: float) -> float:
    deadzone = clamp(deadzone, 0.0, 0.95)
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return float(np.sign(value) * scaled)


def curl_deadzone(value: float, deadzone: float) -> float:
    deadzone = clamp(deadzone, 0.0, 0.95)
    value = clamp(value, 0.0, 1.0)
    if value <= deadzone:
        return 0.0
    return float((value - deadzone) / (1.0 - deadzone))


def angle_radians(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    ab_norm = np.linalg.norm(ab)
    cb_norm = np.linalg.norm(cb)
    if ab_norm < 1e-8 or cb_norm < 1e-8:
        return float(np.pi)
    dot = float(np.dot(ab / ab_norm, cb / cb_norm))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def finger_curl(keypoints_3d: np.ndarray, finger: str) -> float:
    ids = FINGER_KEYPOINTS[finger]
    chain = [0] + ids
    angles = [
        angle_radians(keypoints_3d[chain[i]], keypoints_3d[chain[i + 1]], keypoints_3d[chain[i + 2]])
        for i in range(len(chain) - 2)
    ]
    curl_values = [(np.pi - angle) / (np.pi / 2.0) for angle in angles]
    return float(clamp(np.mean(curl_values), 0.0, 1.0))


def _hand_scale(keypoints_3d: np.ndarray) -> float:
    distances = [
        np.linalg.norm(keypoints_3d[5] - keypoints_3d[17]),  # palm width
        np.linalg.norm(keypoints_3d[0] - keypoints_3d[9]),  # wrist to middle MCP
        np.linalg.norm(keypoints_3d[5] - keypoints_3d[9]) * 2.0,
    ]
    valid = [float(distance) for distance in distances if distance > 1e-6]
    return max(valid) if valid else 1.0


def thumb_geometric_close(keypoints_3d: np.ndarray, scale_multiplier: float = 1.0) -> tuple[float, dict[str, float]]:
    """Estimate thumb closure from thumb-tip position, not thumb joint curl."""
    hand_scale = _hand_scale(keypoints_3d)
    scale_multiplier = max(0.05, float(scale_multiplier))
    thumb_tip = keypoints_3d[4]
    palm_center = np.mean(keypoints_3d[[0, 5, 9, 13, 17]], axis=0)

    palm_distance = float(np.linalg.norm(thumb_tip - palm_center) / hand_scale)
    mcp_distance = float(
        min(np.linalg.norm(thumb_tip - keypoints_3d[index]) for index in (5, 9, 13)) / hand_scale
    )
    index_tip_distance = float(np.linalg.norm(thumb_tip - keypoints_3d[8]) / hand_scale)

    palm_close = clamp((1.25 - palm_distance) / 0.80, 0.0, 1.0)
    mcp_close = clamp((1.10 - mcp_distance) / 0.70, 0.0, 1.0)
    index_close = clamp((1.15 - index_tip_distance) / 0.90, 0.0, 1.0)
    close = clamp(max(palm_close, mcp_close, 0.75 * index_close) * scale_multiplier, 0.0, 1.0)
    return close, {
        "thumb_geom": close,
        "thumb_palm_close": palm_close,
        "thumb_mcp_close": mcp_close,
        "thumb_index_close": index_close,
    }


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def piecewise_signed(value: float, low: float, center: float, high: float) -> float:
    """Map low/center/high feature values to -1/0/+1."""
    if value >= center:
        denom = max(1e-6, high - center)
        return clamp((value - center) / denom, 0.0, 1.0)
    denom = max(1e-6, center - low)
    return clamp((value - center) / denom, -1.0, 0.0)


def finger_side_values(
    keypoints_3d: np.ndarray,
    angle_range: float = 0.55,
    deadzone: float = 0.08,
) -> dict[str, float]:
    axes = palm_axes(keypoints_3d)
    if axes is None:
        return {finger: 0.0 for finger in SIDE_MOTORS}
    angle_range = max(1e-6, angle_range)
    x_axis, y_axis = axes
    values: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "pinky"):
        mcp, pip = FINGER_KEYPOINTS[finger][0], FINGER_KEYPOINTS[finger][1]
        direction = unit_vector(keypoints_3d[pip] - keypoints_3d[mcp])
        if direction is None:
            values[finger] = 0.0
            continue
        lateral = float(np.dot(direction, x_axis))
        forward = abs(float(np.dot(direction, y_axis)))
        angle = float(np.arctan2(lateral, max(forward, 1e-6)))
        values[finger] = side_deadzone(clamp(angle / angle_range, -1.0, 1.0), deadzone)

    # Thumb side-to-side should follow the base/MCP segment. Using the tip vector
    # couples distal curl/opposition into the side motor and makes lateral motion
    # look like forward bend.
    thumb_direction = unit_vector(keypoints_3d[2] - keypoints_3d[1])
    if thumb_direction is None:
        thumb_direction = unit_vector(keypoints_3d[4] - keypoints_3d[1])
    if thumb_direction is None:
        values["thumb"] = 0.0
    else:
        lateral = float(np.dot(thumb_direction, x_axis))
        forward = abs(float(np.dot(thumb_direction, y_axis)))
        angle = float(np.arctan2(lateral, max(forward, 1e-6)))
        values["thumb"] = side_deadzone(clamp(angle / angle_range, -1.0, 1.0), deadzone)
    return values


def thumb_base_features(keypoints_3d: np.ndarray) -> dict[str, float]:
    axes = palm_axes(keypoints_3d)
    features: dict[str, float] = {}
    thumb_direction = unit_vector(keypoints_3d[2] - keypoints_3d[1])
    if axes is not None and thumb_direction is not None:
        x_axis, y_axis = axes
        z_axis = unit_vector(np.cross(x_axis, y_axis))
        lateral = float(np.dot(thumb_direction, x_axis))
        forward = float(np.dot(thumb_direction, y_axis))
        lift = float(np.dot(thumb_direction, z_axis)) if z_axis is not None else 0.0
        features.update(
            {
                "thumb_base_x": lateral,
                "thumb_base_y": forward,
                "thumb_base_z": lift,
                "thumb_base_side_angle_rad": float(np.arctan2(lateral, max(abs(forward), 1e-6))),
                "thumb_base_lift_angle_rad": float(np.arctan2(lift, max(abs(forward), 1e-6))),
            }
        )
    features["thumb_curl"] = float(finger_curl(keypoints_3d, "thumb"))
    features["thumb_tip_palm_dist"] = float(
        np.linalg.norm(keypoints_3d[4] - np.mean(keypoints_3d[[0, 5, 9, 13, 17]], axis=0)) / _hand_scale(keypoints_3d)
    )
    return features


def retarget_calibrated_thumb_to_raw(
    keypoints_3d: np.ndarray,
    calibration: ThumbCalibration = DEFAULT_THUMB_CALIBRATION,
    side_gain: float = 1.0,
    side_limit: float = 1.0,
    forward_gain: float = 0.45,
    curl_gain: float = 1.0,
    side_sign: float = 1.0,
    forward_sign: float = -1.0,
    curl_source: str = "joint",
    forward_source: str = "lift",
    forward_deadzone: float = 0.25,
    forward_curl_suppression: float = 0.70,
    forward_side_suppression: float = 0.35,
) -> tuple[dict[int, int], dict[str, float]]:
    targets = raw_defaults()
    features = thumb_base_features(keypoints_3d)

    side_feature = float(features.get("thumb_base_side_angle_rad", calibration.side_default))
    side_axis = piecewise_signed(side_feature, calibration.side_low, calibration.side_default, calibration.side_high)

    if forward_source == "z":
        forward_feature = float(features.get("thumb_base_z", calibration.base_z_default))
        forward_axis = piecewise_signed(
            forward_feature,
            calibration.base_z_low,
            calibration.base_z_default,
            calibration.base_z_high,
        )
    else:
        forward_feature = float(features.get("thumb_base_lift_angle_rad", calibration.lift_default))
        forward_axis = piecewise_signed(
            forward_feature,
            calibration.lift_low,
            calibration.lift_default,
            calibration.lift_high,
        )

    curl_feature = float(features.get("thumb_curl", calibration.curl_default))
    curl_from_joint = clamp(
        (curl_feature - calibration.curl_default) / max(1e-6, calibration.curl_high - calibration.curl_default),
        0.0,
        1.0,
    )
    distance_feature = float(features.get("thumb_tip_palm_dist", calibration.tip_palm_default))
    curl_from_distance = clamp(
        (calibration.tip_palm_default - distance_feature)
        / max(1e-6, calibration.tip_palm_default - calibration.tip_palm_curl),
        0.0,
        1.0,
    )
    if curl_source == "joint":
        curl_axis = curl_from_joint
    elif curl_source == "distance":
        curl_axis = curl_from_distance
    else:
        curl_axis = clamp(0.60 * curl_from_joint + 0.40 * curl_from_distance, 0.0, 1.0)

    forward_axis_raw = forward_axis
    forward_axis = side_deadzone(forward_axis, forward_deadzone)
    forward_suppression = (
        1.0 - clamp(float(forward_curl_suppression), 0.0, 1.0) * curl_axis
    ) * (
        1.0 - clamp(float(forward_side_suppression), 0.0, 1.0) * abs(side_axis)
    )
    forward_axis *= clamp(forward_suppression, 0.0, 1.0)

    side_limit = clamp(float(side_limit), 0.0, 1.0)
    side_command = clamp(side_axis * float(side_gain) * float(side_sign), -side_limit, side_limit)
    forward_command = clamp(forward_axis * float(forward_gain) * float(forward_sign), -1.0, 1.0)
    curl_command = clamp(curl_axis * float(curl_gain), 0.0, 1.0)

    apply_signed_fraction(targets, SIDE_MOTORS["thumb"], side_command)
    apply_signed_fraction(targets, FINGER_MOTORS["thumb"]["mcp_forward"], forward_command)
    apply_fraction(targets, FINGER_MOTORS["thumb"]["bend"], curl_command)

    signals = {
        **features,
        "thumb_cal_side_axis": side_axis,
        "thumb_cal_forward_axis_raw": forward_axis_raw,
        "thumb_cal_forward_axis": forward_axis,
        "thumb_cal_forward_suppression": forward_suppression,
        "thumb_cal_curl_joint": curl_from_joint,
        "thumb_cal_curl_distance": curl_from_distance,
        "thumb_cal_curl_axis": curl_axis,
        "thumb_side_limit": side_limit,
        "thumb_side": side_command,
        "thumb_forward": forward_command,
        "thumb_drive": curl_command,
    }
    return targets, signals


def retarget_calibrated_thumb_side_to_raw(
    keypoints_3d: np.ndarray,
    calibration: ThumbCalibration = DEFAULT_THUMB_CALIBRATION,
    side_gain: float = 1.0,
    side_limit: float = 1.0,
    side_sign: float = 1.0,
) -> tuple[int, dict[str, float]]:
    features = thumb_base_features(keypoints_3d)
    side_feature = float(features.get("thumb_base_side_angle_rad", calibration.side_default))
    side_axis = piecewise_signed(side_feature, calibration.side_low, calibration.side_default, calibration.side_high)
    side_limit = clamp(float(side_limit), 0.0, 1.0)
    side_command = clamp(side_axis * float(side_gain) * float(side_sign), -side_limit, side_limit)
    targets = raw_defaults()
    apply_signed_fraction(targets, SIDE_MOTORS["thumb"], side_command)
    return targets[SIDE_MOTORS["thumb"]], {
        "thumb_base_side_angle_rad": side_feature,
        "thumb_cal_side_axis": side_axis,
        "thumb_side_limit": side_limit,
        "thumb_side": side_command,
    }


def apply_fraction(targets: dict[int, int], motor_id: int, fraction: float) -> None:
    spec = MOTOR_SPECS[motor_id]
    if spec.bend_raw is None:
        return
    fraction = clamp(fraction, 0.0, 1.0)
    raw = spec.default_raw + fraction * (spec.bend_raw - spec.default_raw)
    targets[motor_id] = clamp_raw_for_motor(motor_id, raw)


def apply_signed_fraction(targets: dict[int, int], motor_id: int, fraction: float) -> None:
    spec = MOTOR_SPECS[motor_id]
    fraction = clamp(fraction, -1.0, 1.0)
    if fraction >= 0.0:
        raw = spec.default_raw + fraction * (spec.safe_max_raw - spec.default_raw)
    else:
        raw = spec.default_raw + fraction * (spec.default_raw - spec.safe_min_raw)
    targets[motor_id] = int(round(clamp(raw, spec.safe_min_raw, spec.safe_max_raw)))


def retarget_grip_fraction_to_raw(
    grip_fraction: float,
    motion_scale: float = 0.65,
    include_mcp: bool = True,
    thumb_scale: float = 1.0,
    active_fingers: Iterable[str] | None = None,
    curl_deadzone_value: float = 0.0,
) -> tuple[dict[int, int], dict[str, float]]:
    targets = raw_defaults()
    active = set(FINGER_KEYPOINTS if active_fingers is None else active_fingers)
    drive = clamp(curl_deadzone(grip_fraction, curl_deadzone_value) * motion_scale, 0.0, 1.0)
    signals = {finger: 0.0 for finger in FINGER_KEYPOINTS}
    for finger in ("index", "middle", "ring", "pinky"):
        if finger not in active:
            continue
        apply_fraction(targets, FINGER_MOTORS[finger]["pip_dip"], drive)
        if include_mcp:
            apply_fraction(targets, FINGER_MOTORS[finger]["mcp_forward"], 0.65 * drive)
        signals[finger] = drive

    if "thumb" in active:
        thumb_drive = clamp(drive * thumb_scale, 0.0, 1.0)
        apply_fraction(targets, FINGER_MOTORS["thumb"]["bend"], thumb_drive)
        if include_mcp:
            apply_fraction(targets, FINGER_MOTORS["thumb"]["mcp_forward"], 0.55 * thumb_drive)
        signals["thumb"] = thumb_drive
        signals["thumb_drive"] = thumb_drive
    else:
        signals["thumb_drive"] = 0.0
    signals["grip"] = drive
    return targets, signals


def retarget_openpose_keypoints_to_raw(
    keypoints_3d: np.ndarray,
    motion_scale: float,
    include_mcp: bool,
    include_side: bool = False,
    side_scale: float = 0.25,
    side_deadzone_value: float = 0.08,
    side_angle_range: float = 0.55,
    side_fingers: Iterable[str] | None = None,
    mcp_forward_ratio: float = 0.65,
    thumb_scale: float = 1.0,
    thumb_mode: str = "curl",
    thumb_geometry_scale: float = 1.0,
    thumb_geometry_deadzone: float = 0.0,
    thumb_forward_ratio: float = 0.55,
    thumb_side_scale: float | None = None,
    thumb_side_suppression_start: float = 0.45,
    thumb_side_suppression_end: float = 0.85,
    thumb_side_suppression_strength: float = 0.0,
    thumb_retarget: str = "generic",
    thumb_side_gain: float = 1.0,
    thumb_side_limit: float = 1.0,
    thumb_forward_gain: float = 0.45,
    thumb_curl_gain: float = 1.0,
    thumb_side_sign: float = 1.0,
    thumb_forward_sign: float = -1.0,
    thumb_curl_source: str = "joint",
    thumb_forward_source: str = "lift",
    thumb_forward_deadzone: float = 0.25,
    thumb_forward_curl_suppression: float = 0.70,
    thumb_forward_side_suppression: float = 0.35,
    open_palm_release: bool = False,
    open_palm_threshold: float = 0.16,
    open_palm_transition: float = 0.08,
    open_palm_blend: float = 1.0,
    open_palm_reset_side: bool = True,
    per_finger_open_release: bool = False,
    active_fingers: Iterable[str] | None = None,
    curl_deadzone_value: float = 0.0,
    thumb_curl_deadzone_value: float | None = None,
) -> tuple[dict[int, int], dict[str, float]]:
    targets = raw_defaults()
    curls = {finger: finger_curl(keypoints_3d, finger) for finger in FINGER_KEYPOINTS}
    active = set(FINGER_KEYPOINTS if active_fingers is None else active_fingers)
    side_values = (
        finger_side_values(keypoints_3d, angle_range=side_angle_range, deadzone=side_deadzone_value)
        if include_side
        else {}
    )
    use_calibrated_thumb = thumb_retarget == "calibrated" and "thumb" in active
    use_side_calibrated_thumb = thumb_retarget in {"generic-centered", "side-calibrated"} and "thumb" in active

    for finger in ("index", "middle", "ring", "pinky"):
        if finger not in active:
            continue
        curl = clamp(curl_deadzone(curls[finger], curl_deadzone_value) * motion_scale, 0.0, 1.0)
        apply_fraction(targets, FINGER_MOTORS[finger]["pip_dip"], curl)
        if include_mcp:
            apply_fraction(targets, FINGER_MOTORS[finger]["mcp_forward"], mcp_forward_ratio * curl)
        curls[f"{finger}_drive"] = curl

    if use_calibrated_thumb:
        thumb_targets, thumb_signals = retarget_calibrated_thumb_to_raw(
            keypoints_3d,
            side_gain=thumb_side_gain,
            side_limit=thumb_side_limit,
            forward_gain=thumb_forward_gain,
            curl_gain=thumb_curl_gain,
            side_sign=thumb_side_sign,
            forward_sign=thumb_forward_sign,
            curl_source=thumb_curl_source,
            forward_source=thumb_forward_source,
            forward_deadzone=thumb_forward_deadzone,
            forward_curl_suppression=thumb_forward_curl_suppression,
            forward_side_suppression=thumb_forward_side_suppression,
        )
        targets[FINGER_MOTORS["thumb"]["bend"]] = thumb_targets[FINGER_MOTORS["thumb"]["bend"]]
        if include_mcp:
            targets[FINGER_MOTORS["thumb"]["mcp_forward"]] = thumb_targets[FINGER_MOTORS["thumb"]["mcp_forward"]]
        enabled_side = tuple(SIDE_MOTORS) if side_fingers is None else tuple(side_fingers)
        if include_side and "thumb" in enabled_side:
            targets[SIDE_MOTORS["thumb"]] = thumb_targets[SIDE_MOTORS["thumb"]]
        curls.update(thumb_signals)
        curls["thumb_metric_signal"] = float(thumb_signals["thumb_cal_curl_axis"])
    elif "thumb" in active:
        thumb_deadzone = curl_deadzone_value if thumb_curl_deadzone_value is None else float(thumb_curl_deadzone_value)
        thumb_curl_signal = curl_deadzone(curls["thumb"], thumb_deadzone)
        thumb_geo_raw, thumb_geo_signals = thumb_geometric_close(keypoints_3d, thumb_geometry_scale)
        thumb_geo_signal = curl_deadzone(thumb_geo_raw, thumb_geometry_deadzone)
        thumb_geo_signals["thumb_geom_raw"] = thumb_geo_raw
        thumb_geo_signals["thumb_geom"] = thumb_geo_signal
        thumb_geo_signals["thumb_geom_deadzone"] = float(thumb_geometry_deadzone)
        if thumb_mode == "geometric":
            thumb_signal = thumb_geo_signal
        elif thumb_mode == "hybrid":
            thumb_signal = max(thumb_curl_signal, thumb_geo_signal)
        else:
            thumb_signal = thumb_curl_signal
        thumb_side_source = abs(float(side_values.get("thumb", 0.0)))
        suppression = smoothstep(
            float(thumb_side_suppression_start),
            float(thumb_side_suppression_end),
            thumb_side_source,
        )
        suppression *= clamp(float(thumb_side_suppression_strength), 0.0, 1.0)
        thumb_signal *= 1.0 - suppression
        thumb_curl = clamp(thumb_signal * motion_scale * thumb_scale, 0.0, 1.0)
        apply_fraction(targets, FINGER_MOTORS["thumb"]["bend"], thumb_curl)
        if include_mcp:
            apply_fraction(targets, FINGER_MOTORS["thumb"]["mcp_forward"], thumb_forward_ratio * thumb_curl)
        curls["thumb_drive"] = thumb_curl
        curls["thumb_curl_signal"] = thumb_curl_signal
        curls["thumb_metric_signal"] = thumb_signal
        curls["thumb_side_source"] = thumb_side_source
        curls["thumb_side_suppression"] = suppression
        curls.update(thumb_geo_signals)
    else:
        curls["thumb_drive"] = 0.0

    if use_side_calibrated_thumb:
        enabled_side = tuple(SIDE_MOTORS) if side_fingers is None else tuple(side_fingers)
        if include_side and "thumb" in enabled_side:
            thumb_side_target, thumb_side_signals = retarget_calibrated_thumb_side_to_raw(
                keypoints_3d,
                side_gain=thumb_side_gain,
                side_limit=thumb_side_limit,
                side_sign=thumb_side_sign,
            )
            targets[SIDE_MOTORS["thumb"]] = thumb_side_target
            curls.update(thumb_side_signals)

    if include_side:
        enabled_side = tuple(SIDE_MOTORS) if side_fingers is None else tuple(side_fingers)
        for finger in enabled_side:
            if finger == "thumb" and (use_calibrated_thumb or use_side_calibrated_thumb):
                continue
            if finger not in active:
                continue
            finger_side_scale = side_scale
            if finger == "thumb" and thumb_side_scale is not None:
                finger_side_scale = float(thumb_side_scale)
            signed_side = clamp(side_values[finger] * finger_side_scale * SIDE_SIGNS[finger], -1.0, 1.0)
            apply_signed_fraction(targets, SIDE_MOTORS[finger], signed_side)
            curls[f"{finger}_side"] = signed_side

    if open_palm_release:
        open_values_by_finger = {
            finger: curl_deadzone(curls[finger], curl_deadzone_value)
            for finger in ("index", "middle", "ring", "pinky")
            if finger in active
        }
        if "thumb" in active:
            open_values_by_finger["thumb"] = float(
                curls.get("thumb_metric_signal", curl_deadzone(curls["thumb"], curl_deadzone_value))
            )
        open_values = list(open_values_by_finger.values())
        open_signal = max(open_values) if open_values else 1.0
        blend = clamp(float(open_palm_blend), 0.0, 1.0)
        transition = max(1e-6, float(open_palm_transition))
        release = clamp(
            (float(open_palm_threshold) - open_signal) / transition,
            0.0,
            1.0,
        )
        release *= blend
        defaults = raw_defaults()
        if per_finger_open_release:
            for finger, signal in open_values_by_finger.items():
                finger_release = clamp(
                    (float(open_palm_threshold) - signal) / transition,
                    0.0,
                    1.0,
                )
                finger_release *= blend
                curls[f"{finger}_open_release"] = float(finger_release)
                if finger_release <= 0.0:
                    continue
                motors = FINGER_MOTORS.get(finger)
                if motors is None:
                    continue
                for motor_id in motors.values():
                    targets[motor_id] = int(
                        round(targets[motor_id] + finger_release * (defaults[motor_id] - targets[motor_id]))
                    )
        if release > 0.0:
            reset_ids: set[int] = set()
            if not per_finger_open_release:
                for finger in active:
                    motors = FINGER_MOTORS.get(finger)
                    if motors is not None:
                        reset_ids.update(motors.values())
            if include_side and open_palm_reset_side:
                enabled_side = tuple(SIDE_MOTORS) if side_fingers is None else tuple(side_fingers)
                for finger in enabled_side:
                    if finger in active:
                        reset_ids.add(SIDE_MOTORS[finger])
            for motor_id in reset_ids:
                targets[motor_id] = int(round(targets[motor_id] + release * (defaults[motor_id] - targets[motor_id])))
        curls["open_palm_signal"] = float(open_signal)
        curls["open_palm_release"] = float(release)
    return targets, curls


def retarget_quest_landmarks_to_raw(
    landmarks_3d: np.ndarray,
    motion_scale: float = 0.65,
    include_mcp: bool = True,
    include_side: bool = False,
    side_scale: float = 0.22,
    side_deadzone_value: float = 0.12,
    side_angle_range: float = 0.55,
    side_fingers: Iterable[str] | None = None,
    mcp_forward_ratio: float = 0.65,
    thumb_scale: float = 1.4,
    thumb_mode: str = "curl",
    thumb_geometry_scale: float = 1.0,
    thumb_geometry_deadzone: float = 0.0,
    thumb_forward_ratio: float = 0.55,
    thumb_side_scale: float | None = None,
    thumb_side_suppression_start: float = 0.45,
    thumb_side_suppression_end: float = 0.85,
    thumb_side_suppression_strength: float = 0.0,
    thumb_retarget: str = "generic",
    thumb_side_gain: float = 1.0,
    thumb_side_limit: float = 1.0,
    thumb_forward_gain: float = 0.45,
    thumb_curl_gain: float = 1.0,
    thumb_side_sign: float = 1.0,
    thumb_forward_sign: float = -1.0,
    thumb_curl_source: str = "joint",
    thumb_forward_source: str = "lift",
    thumb_forward_deadzone: float = 0.25,
    thumb_forward_curl_suppression: float = 0.70,
    thumb_forward_side_suppression: float = 0.35,
    open_palm_release: bool = False,
    open_palm_threshold: float = 0.16,
    open_palm_transition: float = 0.08,
    open_palm_blend: float = 1.0,
    open_palm_reset_side: bool = True,
    per_finger_open_release: bool = False,
    active_fingers: Iterable[str] | None = None,
    curl_deadzone_value: float = 0.0,
    thumb_curl_deadzone_value: float | None = None,
    landmark_format: str = "auto",
) -> tuple[dict[int, int], dict[str, float]]:
    openpose = quest_landmarks_to_openpose21(landmarks_3d, source_format=landmark_format)
    return retarget_openpose_keypoints_to_raw(
        openpose,
        motion_scale=motion_scale,
        include_mcp=include_mcp,
        include_side=include_side,
        side_scale=side_scale,
        side_deadzone_value=side_deadzone_value,
        side_angle_range=side_angle_range,
        side_fingers=side_fingers,
        mcp_forward_ratio=mcp_forward_ratio,
        thumb_scale=thumb_scale,
        thumb_mode=thumb_mode,
        thumb_geometry_scale=thumb_geometry_scale,
        thumb_geometry_deadzone=thumb_geometry_deadzone,
        thumb_forward_ratio=thumb_forward_ratio,
        thumb_side_scale=thumb_side_scale,
        thumb_side_suppression_start=thumb_side_suppression_start,
        thumb_side_suppression_end=thumb_side_suppression_end,
        thumb_side_suppression_strength=thumb_side_suppression_strength,
        thumb_retarget=thumb_retarget,
        thumb_side_gain=thumb_side_gain,
        thumb_side_limit=thumb_side_limit,
        thumb_forward_gain=thumb_forward_gain,
        thumb_curl_gain=thumb_curl_gain,
        thumb_side_sign=thumb_side_sign,
        thumb_forward_sign=thumb_forward_sign,
        thumb_curl_source=thumb_curl_source,
        thumb_forward_source=thumb_forward_source,
        thumb_forward_deadzone=thumb_forward_deadzone,
        thumb_forward_curl_suppression=thumb_forward_curl_suppression,
        thumb_forward_side_suppression=thumb_forward_side_suppression,
        open_palm_release=open_palm_release,
        open_palm_threshold=open_palm_threshold,
        open_palm_transition=open_palm_transition,
        open_palm_blend=open_palm_blend,
        open_palm_reset_side=open_palm_reset_side,
        per_finger_open_release=per_finger_open_release,
        active_fingers=active_fingers,
        curl_deadzone_value=curl_deadzone_value,
        thumb_curl_deadzone_value=thumb_curl_deadzone_value,
    )


def landmark_pinch_fraction(
    landmarks_3d: np.ndarray,
    open_distance: float = 0.095,
    closed_distance: float = 0.025,
    landmark_format: str = "auto",
) -> float:
    openpose = quest_landmarks_to_openpose21(landmarks_3d, source_format=landmark_format)
    thumb_tip = openpose[4]
    index_tip = openpose[8]
    distance = float(np.linalg.norm(thumb_tip - index_tip))
    denom = max(1e-6, open_distance - closed_distance)
    return clamp((open_distance - distance) / denom, 0.0, 1.0)


def parse_retarget_fingers(active_fingers: str, side_fingers: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return parse_active_fingers(active_fingers), parse_side_fingers(side_fingers)


def smooth_targets(
    previous: dict[int, int] | None,
    current: dict[int, int],
    alpha: float,
    max_step_raw: int,
) -> dict[int, int]:
    if previous is None:
        return clamp_targets_to_safe_limits(current)
    alpha = clamp(alpha, 0.0, 1.0)
    max_step_raw = max(0, int(max_step_raw))
    smoothed: dict[int, int] = {}
    for motor_id, raw in current.items():
        previous_raw = previous.get(motor_id, raw)
        blended = previous_raw + alpha * (raw - previous_raw)
        if max_step_raw:
            delta = clamp(blended - previous_raw, -max_step_raw, max_step_raw)
        else:
            delta = 0.0
        smoothed[motor_id] = clamp_raw_for_motor(motor_id, previous_raw + delta)
    return smoothed


@dataclass
class OneEuroMotorState:
    value: float
    derivative: float = 0.0


class TargetFilter:
    VALID_MODES = {"ema", "one-euro"}

    def __init__(
        self,
        initial_targets: dict[int, int],
        filter_mode: str = "one-euro",
        smoothing: float = 0.35,
        max_step_raw: int = 80,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.04,
        one_euro_d_cutoff: float = 1.0,
        max_velocity_raw: float = 0.0,
        side_max_velocity_raw: float = 0.0,
        nominal_hz: float = 30.0,
    ) -> None:
        if filter_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported filter mode: {filter_mode}")
        self.filter_mode = filter_mode
        self.smoothing = clamp(smoothing, 0.0, 1.0)
        self.max_step_raw = max(0, int(max_step_raw))
        self.one_euro_min_cutoff = max(1e-6, float(one_euro_min_cutoff))
        self.one_euro_beta = max(0.0, float(one_euro_beta))
        self.one_euro_d_cutoff = max(1e-6, float(one_euro_d_cutoff))
        self.max_velocity_raw = max(0.0, float(max_velocity_raw))
        self.side_max_velocity_raw = max(0.0, float(side_max_velocity_raw))
        self.nominal_dt = 1.0 / max(1.0, float(nominal_hz))
        self.previous_time: float | None = None
        self.previous_targets = clamp_targets_to_safe_limits(initial_targets)
        self.one_euro_states = {
            motor_id: OneEuroMotorState(float(raw)) for motor_id, raw in self.previous_targets.items()
        }

    @classmethod
    def from_args(cls, args, initial_targets: dict[int, int]) -> "TargetFilter":
        return cls(
            initial_targets=initial_targets,
            filter_mode=args.filter_mode,
            smoothing=args.smoothing,
            max_step_raw=args.max_step_raw,
            one_euro_min_cutoff=args.one_euro_min_cutoff,
            one_euro_beta=args.one_euro_beta,
            one_euro_d_cutoff=args.one_euro_d_cutoff,
            max_velocity_raw=args.max_velocity_raw,
            side_max_velocity_raw=args.side_max_velocity_raw,
            nominal_hz=args.frequency,
        )

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return float(1.0 / (1.0 + tau / dt))

    def _velocity_limit_for_motor(self, motor_id: int) -> float:
        if motor_id in SIDE_MOTOR_IDS and self.side_max_velocity_raw > 0.0:
            return self.side_max_velocity_raw
        return self.max_velocity_raw

    def _apply_velocity_limit(self, motor_id: int, desired: float, previous: float, dt: float) -> float:
        limit = self._velocity_limit_for_motor(motor_id)
        if limit <= 0.0:
            return desired
        max_delta = limit * max(dt, 1e-6)
        return previous + clamp(desired - previous, -max_delta, max_delta)

    def _one_euro_value(self, motor_id: int, raw: float, dt: float) -> float:
        initial = float(self.previous_targets.get(motor_id, raw))
        state = self.one_euro_states.setdefault(motor_id, OneEuroMotorState(initial))
        derivative = (raw - state.value) / max(dt, 1e-6)
        derivative_alpha = self._alpha(self.one_euro_d_cutoff, dt)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * state.derivative
        cutoff = self.one_euro_min_cutoff + self.one_euro_beta * abs(derivative_hat)
        value_alpha = self._alpha(cutoff, dt)
        value_hat = value_alpha * raw + (1.0 - value_alpha) * state.value
        state.value = value_hat
        state.derivative = derivative_hat
        return value_hat

    def filter(self, current: dict[int, int], now: float | None = None) -> dict[int, int]:
        now = time.monotonic() if now is None else now
        if self.previous_time is None:
            dt = self.nominal_dt
        else:
            dt = clamp(now - self.previous_time, 1e-4, 0.25)
        self.previous_time = now

        safe_current = clamp_targets_to_safe_limits(current)
        if self.filter_mode == "ema":
            filtered = smooth_targets(self.previous_targets, safe_current, self.smoothing, self.max_step_raw)
        else:
            filtered = {}
            for motor_id, raw in safe_current.items():
                previous = float(self.previous_targets.get(motor_id, raw))
                desired = self._one_euro_value(motor_id, float(raw), dt)
                limited = self._apply_velocity_limit(motor_id, desired, previous, dt)
                filtered[motor_id] = clamp_raw_for_motor(motor_id, limited)

        self.previous_targets = filtered
        return filtered
