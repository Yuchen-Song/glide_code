"""Calibrated CRAFT hand motor map for the current 15-motor build."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


CRAFT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEQKO2-if00-port0"
CRAFT_BAUDRATE = 57600
RAW_TICKS_PER_REV = 4096


@dataclass(frozen=True)
class MotorSpec:
    finger: str
    joint: str
    motor_id: int
    default_raw: int
    safe_min_raw: int
    safe_max_raw: int
    bend_raw: int | None = None


# Local craft-hand defaults are stored on the active extended-position branch.
MOTOR_SPECS: dict[int, MotorSpec] = {
    0: MotorSpec("middle", "middle_mcp_side", 0, 2163, 1588, 2659),
    1: MotorSpec("index", "index_mcp_side", 1, 1100, 835, 1517),
    2: MotorSpec("pinky", "pinky_mcp_forward", 2, 551, 516, 2299, 2299),
    3: MotorSpec("index", "index_mcp_forward", 3, 2232, 2165, 4149, 4149),
    4: MotorSpec("middle", "middle_mcp_forward", 4, 2781, 991, 2783, 991),
    5: MotorSpec("thumb", "thumb_forward", 5, 3475, 1167, 3732, 1167),
    6: MotorSpec("ring", "ring_pip_dip_bend", 6, 3538, 3538, 6387, 6387),
    7: MotorSpec("index", "index_pip_dip_bend", 7, 43, 43, 2787, 2787),
    8: MotorSpec("ring", "ring_mcp_side", 8, 4111, 3890, 4628),
    9: MotorSpec("ring", "ring_mcp_forward", 9, 556, 410, 2439, 2439),
    10: MotorSpec("middle", "middle_pip_dip_bend", 10, 1702, -982, 1702, -982),
    11: MotorSpec("thumb", "thumb_side", 11, 3786, 3206, 4096),
    12: MotorSpec("pinky", "pinky_pip_dip_bend", 12, 646, 646, 3350, 3350),
    13: MotorSpec("pinky", "pinky_mcp_side", 13, 3748, 3383, 4242),
    14: MotorSpec("thumb", "thumb_bend", 14, 2699, 89, 2699, 89),
}

CRAFT_MOTOR_IDS = sorted(MOTOR_SPECS)

FINGER_KEYPOINTS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGER_MOTORS = {
    "thumb": {"bend": 14, "mcp_forward": 5},
    "index": {"mcp_forward": 3, "pip_dip": 7},
    "middle": {"mcp_forward": 4, "pip_dip": 10},
    "ring": {"mcp_forward": 9, "pip_dip": 6},
    "pinky": {"mcp_forward": 2, "pip_dip": 12},
}

SIDE_MOTORS = {
    "thumb": 11,
    "index": 1,
    "middle": 0,
    "ring": 8,
    "pinky": 13,
}
SIDE_MOTOR_IDS = frozenset(SIDE_MOTORS.values())

SIDE_SIGNS = {
    "thumb": 1.0,
    "index": -1.0,
    "middle": 1.0,
    "ring": -1.0,
    "pinky": -1.0,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def raw_defaults() -> dict[int, int]:
    return {motor_id: spec.default_raw for motor_id, spec in MOTOR_SPECS.items()}


def clamp_raw_for_motor(motor_id: int, raw: int | float) -> int:
    spec = MOTOR_SPECS[motor_id]
    low = float(spec.safe_min_raw)
    high = float(spec.safe_max_raw)
    if spec.bend_raw is not None:
        bend_low = float(min(spec.default_raw, spec.bend_raw))
        bend_high = float(max(spec.default_raw, spec.bend_raw))
        low = max(low, bend_low)
        high = min(high, bend_high)
    return int(round(clamp(float(raw), low, high)))


def clamp_targets_to_safe_limits(
    targets: dict[int, int],
    motor_ids: Iterable[int] | None = None,
) -> dict[int, int]:
    selected = targets.keys() if motor_ids is None else motor_ids
    return {motor_id: clamp_raw_for_motor(motor_id, targets[motor_id]) for motor_id in selected}


def nearest_equivalent_raw(target: int, reference: int) -> int:
    candidates = [target + RAW_TICKS_PER_REV * offset for offset in range(-3, 4)]
    return min(candidates, key=lambda candidate: abs(candidate - reference))


def calibrated_equivalent_raw(motor_id: int, target: int, reference: int, default_reference: int) -> int:
    """Return an equivalent raw target without changing bend-motor branches.

    For bend motors, `default_reference` is the run's stable open/default branch.
    `reference` is still used for non-bend motors, which keep nearest wrapping.
    """
    spec = MOTOR_SPECS[motor_id]
    if spec.bend_raw is None:
        return nearest_equivalent_raw(target, reference)

    raw_offset = target - spec.default_raw
    bend_offset = spec.bend_raw - spec.default_raw
    low_offset = min(0, bend_offset)
    high_offset = max(0, bend_offset)
    if low_offset <= raw_offset <= high_offset:
        default_branch = nearest_equivalent_raw(spec.default_raw, default_reference)
        return default_branch + raw_offset

    return nearest_equivalent_raw(target, reference)


def parse_motor_ids(value: str) -> list[int]:
    if value.strip().lower() in {"all", ""}:
        return CRAFT_MOTOR_IDS.copy()
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if start <= end else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(part))
    ids = list(dict.fromkeys(ids))
    bad = [motor_id for motor_id in ids if motor_id not in MOTOR_SPECS]
    if bad:
        raise ValueError(f"Unsupported CRAFT motor IDs: {bad}")
    return ids


def parse_fingers(value: str, valid: Iterable[str]) -> tuple[str, ...]:
    valid_tuple = tuple(valid)
    normalized = value.strip().lower()
    if normalized in {"all", ""}:
        return valid_tuple
    if normalized in {"none", "off"}:
        return ()
    fingers = tuple(dict.fromkeys(part.strip().lower() for part in normalized.split(",") if part.strip()))
    bad = [finger for finger in fingers if finger not in valid_tuple]
    if bad:
        raise ValueError(f"Unsupported fingers: {bad}. Use comma-separated values from: {', '.join(valid_tuple)}")
    return fingers


def parse_active_fingers(value: str) -> tuple[str, ...]:
    return parse_fingers(value, FINGER_KEYPOINTS)


def parse_side_fingers(value: str) -> tuple[str, ...]:
    return parse_fingers(value, SIDE_MOTORS)
