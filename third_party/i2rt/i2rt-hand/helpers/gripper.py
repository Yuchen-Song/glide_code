from __future__ import annotations

from typing import Literal

import numpy as np


GripperMode = Literal["none", "pinch"]


def update_gripper_from_pinch(
    arm_state: dict | None,
    pinch: float | None,
    *,
    invert: bool,
    mode: GripperMode,
    alpha: float,
    deadband: float,
    max_speed: float,
    dt: float,
) -> bool:
    """Map Quest pinch to a normalized gripper command with basic conditioning."""
    if arm_state is None or mode == "none" or pinch is None or arm_state.get("gripper_index") is None:
        return False

    raw_goal = 1.0 - float(np.clip(pinch, 0.0, 1.0))
    if invert:
        raw_goal = 1.0 - raw_goal
    arm_state["gripper_raw_goal"] = raw_goal

    prev_goal = float(arm_state.get("gripper_goal", arm_state.get("gripper_pos", raw_goal)))
    if deadband > 0.0 and abs(raw_goal - prev_goal) < deadband:
        goal = prev_goal
    else:
        blend = float(np.clip(alpha, 0.0, 1.0))
        goal = prev_goal + blend * (raw_goal - prev_goal)

    current_pos = float(arm_state.get("gripper_pos", goal))
    if max_speed > 0.0:
        max_step = float(max_speed) * max(float(dt), 1e-4)
        goal = current_pos + float(np.clip(goal - current_pos, -max_step, max_step))

    if arm_state.get("gripper_blocked"):
        close_pos = float(arm_state["gripper_close"])
        open_pos = float(arm_state["gripper_open"])
        closing = (goal - current_pos) * (close_pos - open_pos) > 0.0
        if closing:
            return False
        arm_state["gripper_blocked"] = False

    arm_state["gripper_goal"] = float(np.clip(goal, 0.0, 1.0))
    arm_state["gripper_pos"] = arm_state["gripper_goal"]
    return True
