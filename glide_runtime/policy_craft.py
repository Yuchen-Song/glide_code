#!/usr/bin/env python3
"""I2RT arm + CRAFT-hand policy client with optional pour guardrails.

Controls:
- Right SpaceMouse button 1: start from ready; discard/reset if pressed during an episode.
- Right SpaceMouse button 2: save current episode if recording, then reset to ready/default pose.
- Ctrl+C: emergency stop.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import grp
import importlib.util
import logging
import os
import pwd
import signal
import sys
import time
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import numpy as np

from glide_runtime.policy_base import image_tools  # noqa: E402
from glide_runtime.policy_gripper import (  # noqa: E402
    FixedControlRealSenseStream,
    RealSenseStream,
    SpaceMouseButtonReader,
    _build_home_qpos,
    _build_realsense_config,
    _normalize_qpos,
    _parse_control_buttons,
    action_chunk_broker,
    build_image_feature,
    build_joint_names,
    create_or_resume_dataset,
    discard_current_episode,
    ensure_can_interface_ready,
    ensure_realsense_serial,
    get_yam_robot,
    list_realsense_devices,
    list_spacemouse_receivers,
    maybe_save_episode,
    run_preflight_checks,
    websocket_client_policy,
)
from glide_runtime.policy_base import GripperType  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
I2RT_ROOT = REPO_ROOT / "third_party" / "i2rt"
I2RT_CRAFT_HAND_DIR = I2RT_ROOT / "i2rt-craft-hand"
for import_path in (I2RT_ROOT, I2RT_CRAFT_HAND_DIR):
    if import_path.is_dir() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from helpers.async_craft_io import AsyncCraftIO  # noqa: E402
from helpers.module_loader import load_helper_package  # noqa: E402


load_helper_package("craft_hand_helpers", I2RT_ROOT / "craft-hand" / "helpers")

from craft_hand_helpers.dynamixel_io import CraftHandOutput, add_craft_output_args  # noqa: E402
from craft_hand_helpers.motor_config import (  # noqa: E402
    clamp_targets_to_safe_limits,
    parse_motor_ids,
    raw_defaults,
)


DEFAULT_TASK = "i2rt right arm plus CRAFT dexterous hand"
DEFAULT_LEROBOT_HOME = Path(
    os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot")
)
POUR_GUARDRAIL_SCRIPT = I2RT_CRAFT_HAND_DIR / "teleop_craft_glide.py"
_POUR_GUARDRAIL_MODULE: ModuleType | None = None


@dataclass
class PourGuardrailRuntime:
    module: ModuleType
    config: object
    guardrail: object
    left_gripper_max_open_speed: float
    left_gripper_max_close_speed: float
    left_gripper_max_close_fraction: float


def _load_pour_guardrail_module() -> ModuleType:
    global _POUR_GUARDRAIL_MODULE
    if _POUR_GUARDRAIL_MODULE is not None:
        return _POUR_GUARDRAIL_MODULE
    spec = importlib.util.spec_from_file_location("i2rt_craft_hand_pour_guardrail", POUR_GUARDRAIL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pour guardrail implementation from {POUR_GUARDRAIL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _POUR_GUARDRAIL_MODULE = module
    return module


def _clamp_fraction(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _optional_float(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in {"none", "nan", "off", "disable", "disabled"}:
        return None
    return float(text)


def _cli_flags(argv: list[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in argv if token.startswith("--")}


def _apply_legacy_pour_guardrail_aliases(args: argparse.Namespace) -> None:
    flags = _cli_flags(sys.argv[1:])

    alias_map = (
        (
            "--pour-max-ee-speed",
            "pour_max_ee_speed",
            "--guardrail-max-translation-speed",
            "guardrail_max_translation_speed",
        ),
        (
            "--pour-min-ee-xy-distance",
            "pour_min_ee_xy_distance",
            "--guardrail-min-ee-distance",
            "guardrail_min_ee_distance",
        ),
        (
            "--pour-active-min-ee-xy-distance",
            "pour_active_min_ee_xy_distance",
            "--guardrail-min-ee-distance",
            "guardrail_min_ee_distance",
        ),
        (
            "--pour-alignment-xy-max",
            "pour_alignment_xy_max",
            "--guardrail-align-radius",
            "guardrail_align_radius",
        ),
        (
            "--pour-alignment-z-max",
            "pour_alignment_z_max",
            "--guardrail-pour-max-height",
            "guardrail_pour_max_height",
        ),
        (
            "--pour-tilt-start-deg",
            "pour_tilt_start_deg",
            "--guardrail-pour-start-tilt-deg",
            "guardrail_pour_start_tilt_deg",
        ),
        (
            "--pour-prealign-tilt-limit-deg",
            "pour_prealign_tilt_limit_deg",
            "--guardrail-bottle-carry-max-tilt-deg",
            "guardrail_bottle_carry_max_tilt_deg",
        ),
        (
            "--pour-tilt-limit-deg",
            "pour_tilt_limit_deg",
            "--guardrail-bottle-pour-max-tilt-deg",
            "guardrail_bottle_pour_max_tilt_deg",
        ),
        (
            "--pour-cup-max-close-fraction",
            "pour_cup_max_close_fraction",
            "--guardrail-craft-grip-max",
            "guardrail_craft_grip_max",
        ),
        (
            "--pour-cup-max-side-fraction",
            "pour_cup_max_side_fraction",
            "--guardrail-craft-side-max",
            "guardrail_craft_side_max",
        ),
    )
    for legacy_flag, legacy_attr, guardrail_flag, guardrail_attr in alias_map:
        if legacy_flag in flags and guardrail_flag not in flags:
            setattr(args, guardrail_attr, getattr(args, legacy_attr))

    if args.pour_table_z is None:
        return

    z_min = float(args.pour_table_z) + max(float(args.pour_table_clearance), 0.0)
    if "--guardrail-left-bounds" not in flags:
        left_bounds = list(args.guardrail_left_bounds)
        left_bounds[4] = z_min
        args.guardrail_left_bounds = left_bounds
    if "--guardrail-right-bounds" not in flags:
        right_bounds = list(args.guardrail_right_bounds)
        right_bounds[4] = z_min
        args.guardrail_right_bounds = right_bounds


def _build_pour_guardrail_runtime(args: argparse.Namespace) -> PourGuardrailRuntime | None:
    if not args.pour_task_filter:
        return None
    module = _load_pour_guardrail_module()
    _apply_legacy_pour_guardrail_aliases(args)
    config = module.config_from_args(args)
    if not config.enabled:
        return None
    if hasattr(module, "_print_guardrail_summary"):
        module._print_guardrail_summary(config)
    return PourGuardrailRuntime(
        module=module,
        config=config,
        guardrail=module.WinePourGuardrail(config),
        left_gripper_max_open_speed=max(float(args.pour_left_gripper_max_open_speed), 0.0),
        left_gripper_max_close_speed=max(float(args.pour_left_gripper_max_close_speed), 0.0),
        left_gripper_max_close_fraction=_clamp_fraction(args.pour_left_gripper_max_close_fraction),
    )


def _apply_default_dataset_root(args: argparse.Namespace) -> None:
    if args.dataset_root:
        return
    args.dataset_root = str(DEFAULT_LEROBOT_HOME / args.repo_id)


def _episodes_complete(args: argparse.Namespace, dataset) -> bool:
    if args.no_record or dataset is None or args.num_episodes <= 0:
        return False
    return dataset.meta.total_episodes >= args.num_episodes


def _ready_start_message(args: argparse.Namespace, dataset) -> str:
    if args.no_record:
        return "Ready/default reached. Press RIGHT SpaceMouse button 1 to start policy motion."
    if _episodes_complete(args, dataset):
        return "Requested number of episodes reached."
    return "Ready/default reached. Press RIGHT SpaceMouse button 1 to start recording."


def _print_start_message(args: argparse.Namespace, dataset) -> None:
    if args.no_record:
        print("Policy motion started (recording disabled).")
        return
    next_ep = dataset.meta.total_episodes + 1
    if args.num_episodes > 0:
        print(f"Recording episode {next_ep}/{args.num_episodes}.")
    else:
        print(f"Recording episode {next_ep}.")


def _group_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _current_group_names() -> list[str]:
    return [_group_name(gid) for gid in os.getgroups()]


def _check_craft_port_access(args: argparse.Namespace) -> None:
    if args.craft_hand_mode != "drive":
        return

    port = Path(args.craft_port)
    if not port.exists():
        raise RuntimeError(
            f"CRAFT serial port not found: {args.craft_port}. "
            "Check that the FTDI adapter is plugged in, or pass --craft-port."
        )

    resolved_port = port.resolve()
    try:
        stat_result = resolved_port.stat()
    except OSError:
        stat_result = port.stat()

    if os.access(resolved_port, os.R_OK | os.W_OK):
        return

    device_group = _group_name(stat_result.st_gid)
    user = _user_name(os.getuid())
    groups = ",".join(_current_group_names())
    raise RuntimeError(
        "No read/write permission for CRAFT serial port "
        f"{args.craft_port} -> {resolved_port} "
        f"(owner={_user_name(stat_result.st_uid)} group={device_group}). "
        f"Current user {user} is in groups: {groups}. "
        f"Persistent fix: sudo usermod -aG {device_group} {user}, then log out/in or reboot. "
        f"Temporary fix: sudo chmod a+rw {resolved_port}. "
        "For a no-motor dry run, pass --craft-hand-mode shadow."
    )


class CraftStateCache:
    def __init__(self, motor_ids: list[int], initial_targets: dict[int, int]) -> None:
        self.motor_ids = list(motor_ids)
        self.present = {motor_id: int(initial_targets[motor_id]) for motor_id in self.motor_ids}
        self.next_read_time = 0.0
        self.last_warn_time = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([self.present[motor_id] for motor_id in self.motor_ids], dtype=np.float64)

    def maybe_update(self, craft: CraftHandOutput | None, now: float, read_hz: float) -> None:
        if craft is None or read_hz <= 0.0 or now < self.next_read_time:
            return
        self.next_read_time = now + 1.0 / max(read_hz, 1e-6)
        try:
            self.present.update(craft.client.read_raw_positions(self.motor_ids, attempts=1))
        except Exception as exc:
            if now - self.last_warn_time >= 1.0:
                print(f"craft_state_read_failed={type(exc).__name__}: {exc}")
                self.last_warn_time = now


def craft_joint_names(motor_ids: list[int]) -> list[str]:
    return [f"craft_motor_{motor_id}_raw" for motor_id in motor_ids]


def _active_sides(args: argparse.Namespace) -> tuple[str, ...]:
    return ("left", "right") if args.arm_sides == "both" else ("right",)


def _wait_first_frames(
    cam_streams: dict[str, RealSenseStream],
    timeout_s: float = 5.0,
) -> dict[str, np.ndarray]:
    start = time.monotonic()
    while True:
        frames = {name: stream.get_latest_frame() for name, stream in cam_streams.items()}
        if all(frame is not None for frame in frames.values()):
            return {name: frame for name, frame in frames.items() if frame is not None}
        if time.monotonic() - start > timeout_s:
            missing = [name for name, frame in frames.items() if frame is None]
            raise RuntimeError(f"Timed out waiting for initial camera frames: {missing}.")
        time.sleep(0.05)


def _arm_state(robot) -> np.ndarray:
    return np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1)


def _active_arm_items(left_robot, right_robot) -> tuple[tuple[str, object], ...]:
    items = []
    if left_robot is not None:
        items.append(("left", left_robot))
    if right_robot is not None:
        items.append(("right", right_robot))
    return tuple(items)


def _build_pour_guardrail_arm_state(
    robot,
    gripper_index: int | None,
    gripper_limits: tuple[float, float] | None,
    gripper_invert: bool,
    args: argparse.Namespace,
) -> dict:
    from i2rt.robots.pink_kinematics import PinkKinematics  # noqa: WPS433
    from i2rt.robots.utils import I2RT_ROOT as I2RT_PACKAGE_ROOT  # noqa: WPS433

    urdf_path = Path(I2RT_PACKAGE_ROOT) / "robot_models" / "yam" / "yam.urdf"
    ik_frame = args.site or args.ik_frame
    ik_dt = args.ik_dt if args.ik_dt is not None else (1.0 / args.frequency if args.frequency > 0 else 1.0 / 45.0)
    kin = PinkKinematics(
        str(urdf_path),
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
    arm_dofs = int(kin._model.nq)
    total_dofs = int(robot.num_dofs())
    if arm_dofs > total_dofs:
        raise RuntimeError(
            f"FK/IK model has {arm_dofs} joints, but robot command vector has only {total_dofs} DOFs."
        )
    if gripper_index is not None and int(gripper_index) != arm_dofs:
        raise RuntimeError(
            "Unexpected gripper layout for FK/IK guardrail: "
            f"kinematic arm DOFs={arm_dofs}, gripper_index={gripper_index}, total_dofs={total_dofs}. "
            "Expected the gripper command immediately after the arm joints."
        )
    if gripper_index is None and total_dofs != arm_dofs:
        raise RuntimeError(
            "Unexpected no-gripper command layout for FK/IK guardrail: "
            f"kinematic arm DOFs={arm_dofs}, total_dofs={total_dofs}."
        )
    current_q = np.asarray(robot.get_joint_pos(), dtype=float)
    target_q = current_q[:arm_dofs].copy()
    target_pose = kin.fk(target_q)
    gripper_open = 1.0
    gripper_close = 0.0
    if gripper_invert:
        gripper_open, gripper_close = gripper_close, gripper_open
    gripper_pos = float(current_q[gripper_index]) if gripper_index is not None and len(current_q) > gripper_index else 0.0
    return {
        "robot": robot,
        "kin": kin,
        "arm_dofs": arm_dofs,
        "gripper_index": gripper_index,
        "gripper_limits": gripper_limits,
        "target_q": target_q,
        "target_pose": target_pose,
        "init_pose": target_pose.copy(),
        "gripper_pos": gripper_pos,
        "gripper_goal": gripper_pos,
        "gripper_open": gripper_open,
        "gripper_close": gripper_close,
    }


def _reset_pour_guardrail_arm_state(arm_state: dict, qpos: np.ndarray) -> None:
    qpos = _normalize_qpos(np.asarray(qpos, dtype=float), arm_state["robot"].num_dofs())
    arm_dofs = arm_state["arm_dofs"]
    arm_state["target_q"] = qpos[:arm_dofs].copy()
    arm_state["target_pose"] = arm_state["kin"].fk(arm_state["target_q"])
    arm_state["init_pose"] = arm_state["target_pose"].copy()
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is not None and len(qpos) > gripper_index:
        arm_state["gripper_pos"] = float(qpos[gripper_index])
        arm_state["gripper_goal"] = float(qpos[gripper_index])
    for key in ("pour_gripper_filtered_pos", "pour_gripper_desired_pos"):
        arm_state.pop(key, None)


def _reset_pour_guardrail_filter(runtime: PourGuardrailRuntime, pour_arm_states: dict[str, dict]) -> None:
    runtime.guardrail.reset(
        pour_arm_states["left"]["target_pose"] if "left" in pour_arm_states else None,
        pour_arm_states["right"]["target_pose"] if "right" in pour_arm_states else None,
    )


def _sync_pour_gripper_target_from_cmd(arm_state: dict, cmd: np.ndarray) -> None:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None or gripper_index >= len(cmd):
        return
    arm_state["gripper_pos"] = float(cmd[gripper_index])
    arm_state["gripper_goal"] = float(cmd[gripper_index])


def _write_pour_gripper_target_to_cmd(arm_state: dict, cmd: np.ndarray) -> np.ndarray:
    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None or gripper_index >= len(cmd):
        return cmd
    out = cmd.copy()
    out[gripper_index] = float(arm_state.get("gripper_pos", out[gripper_index]))
    return out


def _gripper_close_fraction(pos: float, open_pos: float, close_pos: float) -> float:
    span = close_pos - open_pos
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((float(pos) - open_pos) / span, 0.0, 1.0))


def _limit_pour_left_gripper_command(arm_state: dict | None, runtime: PourGuardrailRuntime, dt: float) -> None:
    if arm_state is None or arm_state.get("gripper_index") is None:
        return
    if not getattr(runtime.config, "enabled", True):
        return
    target_pos = float(arm_state.get("gripper_pos", arm_state.get("gripper_goal", arm_state["gripper_open"])))
    desired_key = "pour_gripper_desired_pos"
    arm_state[desired_key] = target_pos
    desired_pos = float(arm_state[desired_key])

    max_close = float(runtime.left_gripper_max_close_fraction)
    open_pos = float(arm_state["gripper_open"])
    close_pos = float(arm_state["gripper_close"])
    closest_allowed = open_pos + max_close * (close_pos - open_pos)
    if close_pos < open_pos:
        desired_pos = max(desired_pos, closest_allowed)
    else:
        desired_pos = min(desired_pos, closest_allowed)

    prev = float(arm_state.get("pour_gripper_filtered_pos", arm_state.get("gripper_pos", desired_pos)))
    prev_close = _gripper_close_fraction(prev, open_pos, close_pos)
    target_close = _gripper_close_fraction(desired_pos, open_pos, close_pos)
    opening = target_close < prev_close - 1e-4
    speed = runtime.left_gripper_max_open_speed if opening else runtime.left_gripper_max_close_speed
    max_step = max(float(speed), 0.0) * max(float(dt), 0.0)
    filtered = float(np.clip(desired_pos, prev - max_step, prev + max_step))
    arm_state["pour_gripper_filtered_pos"] = filtered
    arm_state["gripper_goal"] = filtered
    arm_state["gripper_pos"] = filtered


def _guardrail_diag() -> SimpleNamespace:
    return SimpleNamespace(
        reason="ok",
        target_xyz=np.zeros(3, dtype=np.float64),
        filtered_translation=np.zeros(3, dtype=np.float64),
    )


def _guardrail_output(left_pose: np.ndarray | None, right_pose: np.ndarray | None) -> SimpleNamespace:
    return SimpleNamespace(
        left_pose=None if left_pose is None else np.asarray(left_pose, dtype=np.float64).copy(),
        right_pose=None if right_pose is None else np.asarray(right_pose, dtype=np.float64).copy(),
        diagnostics={"left": _guardrail_diag(), "right": _guardrail_diag()},
    )


def _guard_pour_craft_targets(
    runtime: PourGuardrailRuntime,
    targets: dict[int, int],
    motor_ids: list[int],
) -> tuple[dict[int, int], bool]:
    if not motor_ids:
        return targets, False
    full_targets = raw_defaults()
    full_targets.update({motor_id: int(targets[motor_id]) for motor_id in motor_ids})
    guarded_full = runtime.module.guard_craft_targets(full_targets, runtime.config)
    guarded = {motor_id: int(guarded_full[motor_id]) for motor_id in motor_ids}
    guarded = clamp_targets_to_safe_limits(guarded, motor_ids)
    return guarded, guarded != targets


def _apply_pour_task_guardrail(
    arm_action_cmds: dict[str, np.ndarray],
    craft_action_targets: dict[int, int],
    pour_arm_states: dict[str, dict],
    runtime: PourGuardrailRuntime | None,
    args: argparse.Namespace,
    craft_motor_ids: list[int],
) -> tuple[dict[str, np.ndarray], dict[int, int], dict[str, bool], bool]:
    if runtime is None:
        return arm_action_cmds, craft_action_targets, {side: True for side in arm_action_cmds}, False

    guarded_craft_targets, craft_limited = _guard_pour_craft_targets(
        runtime,
        craft_action_targets,
        craft_motor_ids,
    )

    dt = 1.0 / max(float(args.frequency), 1e-6)
    left_arm = pour_arm_states.get("left")
    right_arm = pour_arm_states.get("right")
    left_cmd = arm_action_cmds.get("left")
    right_cmd = arm_action_cmds.get("right")

    if left_arm is not None and left_cmd is not None:
        _sync_pour_gripper_target_from_cmd(left_arm, left_cmd)
        _limit_pour_left_gripper_command(left_arm, runtime, dt)
        left_cmd = _write_pour_gripper_target_to_cmd(left_arm, left_cmd)
    if right_arm is not None and right_cmd is not None:
        _sync_pour_gripper_target_from_cmd(right_arm, right_cmd)

    left_pose = (
        left_arm["kin"].fk(np.asarray(left_cmd[: left_arm["arm_dofs"]], dtype=float))
        if left_arm is not None and left_cmd is not None
        else None
    )
    right_pose = (
        right_arm["kin"].fk(np.asarray(right_cmd[: right_arm["arm_dofs"]], dtype=float))
        if right_arm is not None and right_cmd is not None
        else None
    )

    guarded_output = runtime.guardrail.filter_output(_guardrail_output(left_pose, right_pose), dt)
    left_pose = guarded_output.left_pose
    right_pose = guarded_output.right_pose

    guarded_cmds = dict(arm_action_cmds)
    ik_success = {side: True for side in arm_action_cmds}
    for side, arm_state, cmd, pose in (
        ("left", left_arm, left_cmd, left_pose),
        ("right", right_arm, right_cmd, right_pose),
    ):
        if arm_state is None or cmd is None or pose is None:
            continue
        success, qpos = arm_state["kin"].ik(pose, init_q=arm_state["target_q"])
        guarded = cmd.copy()
        if success:
            arm_state["target_q"] = qpos
            arm_state["target_pose"] = pose
            guarded[: arm_state["arm_dofs"]] = qpos
        ik_success[side] = bool(success)
        guarded_cmds[side] = _write_pour_gripper_target_to_cmd(arm_state, guarded)

    return guarded_cmds, guarded_craft_targets, ik_success, craft_limited


def _state_vector(
    arm_items: tuple[tuple[str, object], ...],
    craft_state_cache: CraftStateCache,
) -> np.ndarray:
    parts = [_arm_state(robot) for _side, robot in arm_items]
    parts.append(craft_state_cache.vector())
    return np.concatenate(parts).astype(np.float32)


def _arm_command_parts(
    action_vec: np.ndarray,
    arm_items: tuple[tuple[str, object], ...],
) -> tuple[dict[str, np.ndarray], int]:
    commands: dict[str, np.ndarray] = {}
    offset = 0
    for side, robot in arm_items:
        dofs = robot.num_dofs()
        commands[side] = action_vec[offset : offset + dofs].astype(float)
        offset += dofs
    return commands, offset


def _concat_arm_commands(
    arm_items: tuple[tuple[str, object], ...],
    commands: dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate([commands[side] for side, _robot in arm_items]).astype(np.float32)


def _build_observation(
    cam_streams: dict[str, RealSenseStream],
    arm_items: tuple[tuple[str, object], ...],
    craft_state_cache: CraftStateCache,
    args: argparse.Namespace,
    resize: bool,
) -> tuple[dict, np.ndarray] | None:
    raw_images = {name: stream.get_latest_frame() for name, stream in cam_streams.items()}
    if any(image is None for image in raw_images.values()):
        return None

    policy_images = {}
    for name, image in raw_images.items():
        if resize:
            policy_images[name] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, args.resize_height, args.resize_width)
            )
        else:
            policy_images[name] = image_tools.convert_to_uint8(image)

    state_vec = _state_vector(arm_items, craft_state_cache)
    obs = {
        "observation.state": state_vec,
        "state": state_vec,
    }
    for name, image in policy_images.items():
        obs[f"observation.images.{name}"] = image
    if "head" in policy_images:
        obs["head_image"] = policy_images["head"]
    if "left_wrist" in policy_images:
        obs["left_wrist_image"] = policy_images["left_wrist"]
    if "right_wrist" in policy_images:
        obs["right_wrist_image"] = policy_images["right_wrist"]
    if args.prompt:
        obs["prompt"] = args.prompt
    return obs, state_vec


def _target_dict_from_action(values: np.ndarray, motor_ids: list[int]) -> dict[int, int]:
    raw_targets = {motor_id: int(round(float(value))) for motor_id, value in zip(motor_ids, values, strict=True)}
    return clamp_targets_to_safe_limits(raw_targets, motor_ids)


def _limit_arm_step(command: np.ndarray, previous: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return command
    delta = np.clip(command - previous, -max_step, max_step)
    return previous + delta


def _limit_craft_step(
    targets: dict[int, int],
    previous: dict[int, int],
    motor_ids: list[int],
    max_step_raw: int,
) -> dict[int, int]:
    if max_step_raw <= 0:
        return targets
    limited = {}
    for motor_id in motor_ids:
        delta = int(targets[motor_id]) - int(previous[motor_id])
        delta = int(np.clip(delta, -max_step_raw, max_step_raw))
        limited[motor_id] = int(previous[motor_id]) + delta
    return clamp_targets_to_safe_limits(limited, motor_ids)


def _format_craft_targets(targets: dict[int, int], motor_ids: list[int]) -> str:
    return " ".join(f"{motor_id}:{targets[motor_id]}" for motor_id in motor_ids)


def _policy_infer(policy, obs: dict, expected_action_dim: int | None = None) -> dict:
    try:
        return policy.infer(obs)
    except RuntimeError as exc:
        message = str(exc)
        state_dim = int(np.asarray(obs.get("state", [])).reshape(-1).size)
        if "operands could not be broadcast together" in message and "(21,)" in message and "(28,)" in message:
            expected_text = "" if expected_action_dim is None else f" and expects {expected_action_dim}-D actions"
            raise RuntimeError(
                "Policy/client shape mismatch. "
                f"This client is sending a {state_dim}-D observation.state{expected_text}, but the policy server "
                "loaded 21-D normalization stats from the checkpoint. For dual-arm CRAFT, rerun norm-stat "
                "generation and retrain/serve a 28-D checkpoint. For the old right-arm checkpoint, serve with "
                "`--policy.config=pour_wine_glass_v1_legacy_21d` and run this client with "
                "`--arm-sides right --cameras head`."
            ) from exc
        raise


def _add_record_frame(
    dataset,
    args: argparse.Namespace,
    cam_streams: dict[str, RealSenseStream],
    arm_items: tuple[tuple[str, object], ...],
    craft_state_cache: CraftStateCache,
    arm_action_cmds: dict[str, np.ndarray],
    craft_action_targets: dict[int, int],
) -> bool:
    if dataset is None:
        return False
    raw_images = {name: stream.get_latest_frame() for name, stream in cam_streams.items()}
    if any(image is None for image in raw_images.values()):
        return False

    state_vec = _state_vector(arm_items, craft_state_cache)
    craft_action_vec = np.asarray(
        [craft_action_targets[motor_id] for motor_id in craft_state_cache.motor_ids],
        dtype=np.float32,
    )
    action_vec = np.concatenate([_concat_arm_commands(arm_items, arm_action_cmds), craft_action_vec]).astype(np.float32)

    frame = {
        "observation.state": state_vec,
        "action": action_vec,
        "task": args.task,
    }
    for name, image in raw_images.items():
        frame[f"observation.images.{name}"] = image
    dataset.add_frame(frame)
    return True


def _print_controls(recording_enabled: bool = True) -> None:
    print("\n" + "=" * 72)
    print("I2RT ARM + CRAFT POLICY + SPACEMOUSE RECORDING")
    print("=" * 72)
    if recording_enabled:
        print("  Right SpaceMouse Button 1: Start from ready; discard/reset during episode")
        print("  Right SpaceMouse Button 2: Save episode if recording + reset to ready/default")
    else:
        print("  Right SpaceMouse Button 1: Start from ready; reset during motion")
        print("  Right SpaceMouse Button 2: Reset to ready/default")
    print("  Ctrl+C                   : Emergency stop")
    print("=" * 72 + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="I2RT right-arm + CRAFT-hand policy client with SpaceMouse control and LeRobot recording."
    )

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--frequency", type=float, default=45.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)

    parser.add_argument("--arm-sides", choices=["right", "both"], default="both")
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper", type=str, default="no_gripper")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-force-threshold", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--gripper-force-verbose", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gripper-force-print-interval", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--gripper-backoff", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--arm-action-dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-arm-joint-step", type=float, default=0.06)
    parser.add_argument("--ready-qpos", type=float, nargs="+", default=[0, 1.047, 1.047, 0, 0, 0, 1.0])
    parser.add_argument("--home-time", type=float, default=2.0)
    parser.add_argument("--no-home", action="store_true")

    parser.add_argument("--craft-hand-mode", choices=["disabled", "shadow", "drive"], default="drive")
    parser.add_argument("--craft-io-mode", choices=["async", "sync"], default="async")
    parser.add_argument("--craft-command-hz", type=float, default=15.0)
    parser.add_argument("--craft-state-read-hz", type=float, default=0.0)
    parser.add_argument("--max-craft-step-raw", type=int, default=240)
    add_craft_output_args(parser)
    parser.set_defaults(craft_baudrate=1000000, current_limit=230, default_ramp_seconds=2.0)

    parser.add_argument(
        "--pour-task-filter",
        action="store_true",
        help=(
            "Enable wine-pour task guardrails from teleop_craft_glide.py. "
            "Policy joint actions are mapped through FK/IK so the updated workspace bounds, "
            "pose-rate limits, bottle tilt gating/assist, cup upright guard, and CRAFT "
            "closure caps can be applied."
        ),
    )
    parser.add_argument("--no-guardrail", dest="guardrail_enabled", action="store_false", default=True)
    parser.add_argument(
        "--no-guardrail-task-defaults",
        dest="guardrail_task_defaults",
        action="store_false",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--guardrail-left-bounds",
        type=_optional_float,
        nargs=6,
        default=[None, None, None, None, 0.085, 0.72],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument(
        "--guardrail-right-bounds",
        type=_optional_float,
        nargs=6,
        default=[None, None, None, None, 0.085, 0.72],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
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
    parser.add_argument("--guardrail-require-pour-alignment", action="store_true")
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
    parser.add_argument("--guardrail-help", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pour-filter-alpha", type=float, default=0.65, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-ee-speed", type=float, default=0.36, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-xy-speed", type=float, default=0.34, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-z-speed", type=float, default=0.18, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-ee-accel", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-xy-accel", type=float, default=2.2, help=argparse.SUPPRESS)
    parser.add_argument("--pour-max-z-accel", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-placement-z-max", type=float, default=0.16, help=argparse.SUPPRESS)
    parser.add_argument("--pour-placement-max-down-speed", type=float, default=0.045, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pour-table-z",
        type=float,
        default=None,
        help="Legacy shortcut for setting both guardrail lower z bounds to table_z + pour_table_clearance.",
    )
    parser.add_argument("--pour-table-clearance", type=float, default=0.035)
    parser.add_argument("--pour-min-ee-xy-distance", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-active-min-ee-xy-distance", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-alignment-xy-max", type=float, default=0.30, help=argparse.SUPPRESS)
    parser.add_argument("--pour-alignment-z-max", type=float, default=0.20, help=argparse.SUPPRESS)
    parser.add_argument("--pour-operator-alignment-distance-max", type=float, default=0.78, help=argparse.SUPPRESS)
    parser.add_argument("--pour-operator-alignment-z-max", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--pour-speed-scale", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--pour-bottle-grasp-close-threshold", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--pour-cup-grip-close-threshold", type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument("--pour-tilt-start-deg", type=float, default=18.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-prealign-tilt-limit-deg", type=float, default=70.0, help=argparse.SUPPRESS)
    parser.add_argument("--pour-tilt-limit-deg", type=float, default=135.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pour-bottle-tilt-rate-deg-s",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    pour_roll_group = parser.add_mutually_exclusive_group()
    pour_roll_group.add_argument(
        "--pour-left-roll-passthrough",
        dest="pour_left_roll_passthrough",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    pour_roll_group.add_argument(
        "--no-pour-left-roll-passthrough",
        dest="pour_left_roll_passthrough",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pour-cup-upright-weight", type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument("--pour-left-gripper-max-open-speed", type=float, default=0.55)
    parser.add_argument("--pour-left-gripper-max-close-speed", type=float, default=0.85)
    parser.add_argument("--pour-left-gripper-max-close-fraction", type=float, default=0.96)
    parser.add_argument("--pour-cup-max-close-fraction", type=float, default=0.84, help=argparse.SUPPRESS)
    parser.add_argument("--pour-cup-max-side-fraction", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--site", type=str, default=None, help="Alias for --ik-frame.")
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
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
    )
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
    parser.add_argument(
        "--rs-white-balance",
        type=float,
        default=None,
        help="Fixed RealSense color white balance value.",
    )
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
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--no-record", action="store_true", help="Run policy control without creating/saving a dataset.")
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm_craft_hand")
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

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.guardrail_help:
        _load_pour_guardrail_module().build_guardrail_parser().print_help()
        return

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

    if not args.no_record and not args.repo_id:
        raise RuntimeError("--repo-id is required for recording. Example: --repo-id data/my_craft_eval.")
    if not args.no_record:
        _apply_default_dataset_root(args)
    if args.prompt is None:
        args.prompt = args.task
    pour_guardrail = _build_pour_guardrail_runtime(args)
    if pour_guardrail is not None:
        print(
            "Pour guardrail enabled: policy joint actions will be mapped through FK/IK with "
            "the updated wine-pour pose guardrail and CRAFT closure caps."
        )

    dataset_fps = int(round(args.frequency))
    if abs(args.frequency - dataset_fps) > 1e-3:
        print(f"Warning: --frequency {args.frequency} is not integer; dataset fps set to {dataset_fps}.")

    if args.no_record:
        print("Recording disabled (--no-record); dataset creation, frame writes, and video encoding will be skipped.")

    if not args.no_record and not args.skip_preflight:
        ok = run_preflight_checks(args, dataset_fps)
        if args.preflight_only:
            return
        if not ok:
            return

    _check_craft_port_access(args)

    active_sides = _active_sides(args)
    for side in active_sides:
        ensure_can_interface_ready(args.left_channel if side == "left" else args.right_channel)

    selected_cams = set(args.cameras)
    if "head" in selected_cams:
        ensure_realsense_serial(args.head_serial, "Head")
    if "left_wrist" in selected_cams:
        ensure_realsense_serial(args.left_wrist_serial, "Left wrist")
    if "right_wrist" in selected_cams:
        ensure_realsense_serial(args.right_wrist_serial, "Right wrist")

    cam_specs = {
        "head": (args.head_serial, args.head_width, args.head_height, args.head_fps),
        "left_wrist": (args.left_wrist_serial, args.wrist_width, args.wrist_height, args.wrist_fps),
        "right_wrist": (args.right_wrist_serial, args.wrist_width, args.wrist_height, args.wrist_fps),
    }
    cam_streams: dict[str, FixedControlRealSenseStream] = {}
    for name in ("head", "left_wrist", "right_wrist"):
        if name not in selected_cams:
            continue
        serial, width, height, fps = cam_specs[name]
        cam_streams[name] = FixedControlRealSenseStream(
            _build_realsense_config(serial, width, height, fps, args)
        )

    left_robot = None
    right_robot = None
    spacemouse = None
    dataset = None
    policy = None
    left_home = None
    right_home = None
    pour_arm_states: dict[str, dict] = {}
    recording = False
    stop_event = Event()

    def _handle_sigint(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        print("Starting camera streams...")
        for name, stream in cam_streams.items():
            serial, width, height, fps = cam_specs[name]
            print(f"  {name}: {width}x{height}@{fps}")
            try:
                stream.start()
            except RuntimeError:
                if fps == 30 or args.allow_camera_fallback:
                    raise
                print(f"{name} camera failed at {width}x{height}@{fps}; retrying 30 FPS.")
                stream.stop()
                retry_stream = FixedControlRealSenseStream(
                    _build_realsense_config(serial, width, height, 30, args)
                )
                retry_stream.start()
                cam_streams[name] = retry_stream

        if "left" in active_sides:
            left_gripper_type = GripperType.from_string_name(args.left_gripper)
            left_robot = get_yam_robot(channel=args.left_channel, gripper_type=left_gripper_type)

        right_gripper_type = GripperType.from_string_name(args.right_gripper)
        right_robot = get_yam_robot(channel=args.right_channel, gripper_type=right_gripper_type)

        left_info = left_robot.get_robot_info() if left_robot is not None else None
        right_info = right_robot.get_robot_info()
        if pour_guardrail is not None:
            if left_robot is not None and left_info is not None:
                pour_arm_states["left"] = _build_pour_guardrail_arm_state(
                    left_robot,
                    left_info.get("gripper_index"),
                    left_info.get("gripper_limits"),
                    args.left_gripper_invert,
                    args,
                )
            pour_arm_states["right"] = _build_pour_guardrail_arm_state(
                right_robot,
                right_info.get("gripper_index"),
                right_info.get("gripper_limits"),
                args.right_gripper_invert,
                args,
            )
            print(f"Pour task FK/IK frame: {args.site or args.ik_frame}.")
            if "left" not in pour_arm_states:
                print("Pour task filter note: left arm is inactive, so bottle-side guardrails are skipped.")
        if left_robot is not None and left_info is not None:
            left_home = _build_home_qpos(
                left_robot,
                left_info.get("gripper_index"),
                left_info.get("gripper_limits"),
                args.left_gripper_invert,
            )
        right_home = _build_home_qpos(
            right_robot,
            right_info.get("gripper_index"),
            right_info.get("gripper_limits"),
            args.right_gripper_invert,
        )

        arm_items = _active_arm_items(left_robot, right_robot)
        ready_seed = np.asarray(args.ready_qpos, dtype=float)
        last_arm_cmds: dict[str, np.ndarray] = {}
        print(f"Moving {','.join(side for side, _robot in arm_items)} arm(s) to ready pose...")
        for side, robot in arm_items:
            ready_qpos = _normalize_qpos(ready_seed, robot.num_dofs())
            robot.move_joints(ready_qpos, time_interval_s=args.home_time)
            last_arm_cmds[side] = ready_qpos.copy()
            if pour_guardrail is not None and side in pour_arm_states:
                _reset_pour_guardrail_arm_state(pour_arm_states[side], ready_qpos)
        if pour_guardrail is not None:
            _reset_pour_guardrail_filter(pour_guardrail, pour_arm_states)

        first_frames = _wait_first_frames(cam_streams)
        for cam_name, frame in first_frames.items():
            height, width = frame.shape[:2]
            spec_width = args.head_width if cam_name == "head" else args.wrist_width
            spec_height = args.head_height if cam_name == "head" else args.wrist_height
            if (width, height) != (spec_width, spec_height):
                print(
                    f"{cam_name} camera stream is {width}x{height}; overriding requested "
                    f"{spec_width}x{spec_height}."
                )

        craft_motor_ids = parse_motor_ids(args.craft_motors)
        state_names: list[str] = []
        if left_robot is not None and left_info is not None:
            state_names.extend(build_joint_names("left", left_robot.num_dofs(), left_info.get("gripper_index")))
        state_names.extend(build_joint_names("right", right_robot.num_dofs(), right_info.get("gripper_index")))
        state_names.extend(craft_joint_names(craft_motor_ids))
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "action": {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)},
        }
        for cam_name, frame in first_frames.items():
            height, width = frame.shape[:2]
            features[f"observation.images.{cam_name}"] = build_image_feature(height, width)
        expected_action_dim = len(state_names)
        if not args.no_record:
            dataset = create_or_resume_dataset(args, dataset_fps, features)
            if args.vcodec != "libsvtav1":
                print(
                    "Warning: lerobot 0.1.0 does not expose vcodec in LeRobotDataset; "
                    f"using custom encoder override '{args.vcodec}'."
                )

            def _encode_episode_videos_with_codec(episode_index: int) -> dict:
                from lerobot.common.datasets.video_utils import encode_video_frames

                video_paths = {}
                for key in dataset.meta.video_keys:
                    video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                    if not video_path.is_file():
                        img_dir = dataset._get_image_file_path(
                            episode_index=episode_index,
                            image_key=key,
                            frame_index=0,
                        ).parent
                        encode_video_frames(img_dir, video_path, dataset.fps, vcodec=args.vcodec, overwrite=True)
                    video_paths[key] = str(video_path)
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

        craft_context = (
            CraftHandOutput.from_args(args)
            if args.craft_hand_mode == "drive"
            else nullcontext(None)
        )

        with craft_context as craft:
            initial_craft_targets = craft.last_targets.copy() if craft is not None else raw_defaults()
            craft_defaults = raw_defaults()
            craft_action_targets = {motor_id: int(initial_craft_targets[motor_id]) for motor_id in craft_motor_ids}
            craft_state_cache = CraftStateCache(craft_motor_ids, craft_action_targets)
            craft_io: AsyncCraftIO | None = None

            def sync_craft_targets_from_output() -> None:
                nonlocal craft_action_targets
                if args.craft_hand_mode == "disabled":
                    source = craft_defaults
                elif craft is not None:
                    source = craft.last_targets
                else:
                    source = craft_action_targets
                craft_action_targets = {motor_id: int(source[motor_id]) for motor_id in craft_motor_ids}
                craft_state_cache.present.update(craft_action_targets)

            def start_async_craft_io() -> None:
                nonlocal craft_io
                if craft is None or args.craft_io_mode != "async" or craft_io is not None:
                    return
                craft_io = AsyncCraftIO(
                    craft=craft,
                    motor_ids=craft_motor_ids,
                    command_hz=args.craft_command_hz,
                    state_read_hz=args.craft_state_read_hz,
                    initial_targets=craft_action_targets,
                ).start()
                print(
                    f"craft_io=async command_hz={args.craft_command_hz:.1f} "
                    f"state_read_hz={args.craft_state_read_hz:.1f}"
                )

            def stop_async_craft_io(label: str) -> None:
                nonlocal craft_io
                if craft_io is None:
                    return
                stopped = craft_io.stop()
                print(f"craft_io_stopped label={label} ok={stopped}")
                craft_io = None

            def reset_to_ready(status: str) -> None:
                print(status)
                stop_async_craft_io("reset")
                for side, robot in arm_items:
                    ready_qpos = _normalize_qpos(ready_seed, robot.num_dofs())
                    robot.move_joints(ready_qpos, time_interval_s=args.home_time)
                    last_arm_cmds[side] = ready_qpos.copy()
                    if pour_guardrail is not None and side in pour_arm_states:
                        _reset_pour_guardrail_arm_state(pour_arm_states[side], ready_qpos)
                if craft is not None:
                    craft.move_to_defaults("reset_default")
                else:
                    craft_action_targets.update(
                        {motor_id: int(craft_defaults[motor_id]) for motor_id in craft_motor_ids}
                    )
                sync_craft_targets_from_output()
                if pour_guardrail is not None:
                    _reset_pour_guardrail_filter(pour_guardrail, pour_arm_states)
                policy.reset()
                start_async_craft_io()
                print(_ready_start_message(args, dataset))

            if craft is not None and args.craft_io_mode == "async":
                start_async_craft_io()
            elif craft is not None:
                print("craft_io=sync")
            else:
                print(f"craft_io=none mode={args.craft_hand_mode}")

            for _ in range(max(0, args.warmup_steps)):
                obs_pack = _build_observation(
                    cam_streams,
                    arm_items,
                    craft_state_cache,
                    args,
                    resize=not args.no_resize,
                )
                if obs_pack is None:
                    time.sleep(0.01)
                    continue
                _policy_infer(policy, obs_pack[0], expected_action_dim)
            policy.reset()

            _print_controls(recording_enabled=not args.no_record)
            print(_ready_start_message(args, dataset))

            step_time = 1.0 / args.frequency if args.frequency > 0 else 0.0
            last_cam_warn_time = 0.0
            last_action_warn_time = 0.0
            last_print_time = 0.0
            last_pour_ik_warn_time = 0.0

            teleop_enabled = False
            recording = False
            episode_start_time = None
            last_right_button1 = False
            last_right_button2 = False
            step = 0

            try:
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
                            reset_to_ready("Right SpaceMouse button 2 pressed. Saving and resetting.")
                            if _episodes_complete(args, dataset):
                                break
                        else:
                            teleop_enabled = False
                            reset_to_ready("Right SpaceMouse button 2 pressed. Resetting.")
                        continue

                    if start_rising and teleop_enabled:
                        teleop_enabled = False
                        if recording:
                            recording = False
                            episode_start_time = None
                            discarded = discard_current_episode(dataset)
                            if discarded:
                                print("Current episode discarded; video writing skipped.")
                            else:
                                print("No active recording to discard.")
                            reset_to_ready("Right SpaceMouse button 1 pressed during episode. Discarding and resetting.")
                        else:
                            reset_to_ready("Right SpaceMouse button 1 pressed during motion. Resetting.")
                        continue

                    if not teleop_enabled:
                        if start_rising:
                            if _episodes_complete(args, dataset):
                                print("Reached requested number of episodes; start ignored.")
                                continue
                            teleop_enabled = True
                            recording = not args.no_record
                            episode_start_time = time.monotonic() if recording else None
                            policy.reset()
                            if pour_guardrail is not None:
                                for side, _robot in arm_items:
                                    _reset_pour_guardrail_arm_state(pour_arm_states[side], last_arm_cmds[side])
                                _reset_pour_guardrail_filter(pour_guardrail, pour_arm_states)
                            _print_start_message(args, dataset)
                        else:
                            if step_time > 0:
                                elapsed = time.time() - step_start
                                if elapsed < step_time:
                                    time.sleep(step_time - elapsed)
                            else:
                                time.sleep(0.01)
                            continue

                    obs_pack = _build_observation(
                        cam_streams,
                        arm_items,
                        craft_state_cache,
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

                    obs, _state_vec = obs_pack
                    action = _policy_infer(policy, obs, expected_action_dim)
                    action_vec = np.asarray(action.get("actions", []), dtype=float).reshape(-1)

                    if action_vec.size != expected_action_dim:
                        now = time.monotonic()
                        if now - last_action_warn_time > 2.0:
                            print(
                                f"Action dim mismatch (got {action_vec.size}, expected {expected_action_dim}); "
                                "holding position."
                            )
                            last_action_warn_time = now
                        arm_action_cmds = {side: last_arm_cmds[side] for side, _robot in arm_items}
                        next_craft_targets = craft_action_targets
                    else:
                        raw_arm_cmds, craft_offset = _arm_command_parts(action_vec, arm_items)
                        arm_action_cmds = {
                            side: _limit_arm_step(
                                command,
                                last_arm_cmds[side],
                                args.max_arm_joint_step,
                            )
                            for side, command in raw_arm_cmds.items()
                        }
                        if args.craft_hand_mode == "disabled":
                            next_craft_targets = {
                                motor_id: int(craft_defaults[motor_id]) for motor_id in craft_motor_ids
                            }
                        else:
                            raw_targets = _target_dict_from_action(action_vec[craft_offset:], craft_motor_ids)
                            next_craft_targets = _limit_craft_step(
                                raw_targets,
                                craft_action_targets,
                                craft_motor_ids,
                                args.max_craft_step_raw,
                            )

                    if pour_guardrail is not None:
                        arm_action_cmds, next_craft_targets, pour_ik_success, _craft_limited = (
                            _apply_pour_task_guardrail(
                                arm_action_cmds,
                                next_craft_targets,
                                pour_arm_states,
                                pour_guardrail,
                                args,
                                craft_motor_ids,
                            )
                        )
                        failed_ik_sides = [side for side, success in pour_ik_success.items() if not success]
                        if failed_ik_sides:
                            for side in failed_ik_sides:
                                arm_action_cmds[side] = last_arm_cmds[side]
                            for side, arm_state in pour_arm_states.items():
                                if side in arm_action_cmds:
                                    _reset_pour_guardrail_arm_state(arm_state, arm_action_cmds[side])
                            _reset_pour_guardrail_filter(pour_guardrail, pour_arm_states)
                            now = time.monotonic()
                            if now - last_pour_ik_warn_time > 1.0:
                                print(
                                    "Pour task IK failed for "
                                    f"{', '.join(failed_ik_sides)} arm; holding previous arm joints."
                                )
                                last_pour_ik_warn_time = now

                    for side, robot in arm_items:
                        robot.command_joint_pos(arm_action_cmds[side])
                        last_arm_cmds[side] = arm_action_cmds[side]

                    craft_targets_updated = next_craft_targets != craft_action_targets
                    craft_action_targets = next_craft_targets
                    now_mono = time.monotonic()
                    if craft_io is not None:
                        if craft_targets_updated:
                            craft_io.submit(craft_action_targets, now=now_mono)
                        craft_state_cache.present.update(craft_io.present_snapshot())
                    elif craft is not None:
                        if craft_targets_updated:
                            craft.write_raw(craft_action_targets)
                        craft_state_cache.maybe_update(craft, now_mono, args.craft_state_read_hz)
                    else:
                        craft_state_cache.present.update(craft_action_targets)

                    if recording:
                        frame_added = _add_record_frame(
                            dataset,
                            args,
                            cam_streams,
                            arm_items,
                            craft_state_cache,
                            arm_action_cmds,
                            craft_action_targets,
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
                                reset_to_ready("Episode time reached. Resetting to ready/default.")
                                if _episodes_complete(args, dataset):
                                    break
                                continue

                    if now_mono - last_print_time > 2.0:
                        print(
                            f"policy_step={step} "
                            f"craft_targets={_format_craft_targets(craft_action_targets, craft_motor_ids)}"
                        )
                        last_print_time = now_mono

                    step += 1
                    if args.max_steps > 0 and step >= args.max_steps:
                        break

                    if step_time > 0:
                        elapsed = time.time() - step_start
                        if elapsed < step_time:
                            time.sleep(step_time - elapsed)
            finally:
                stop_async_craft_io("shutdown")
                if dataset is not None and recording:
                    print("Saving partial episode before shutdown...")
                    saved = maybe_save_episode(dataset)
                    if saved:
                        print(f"Episode {dataset.meta.total_episodes} saved.")
                    else:
                        discard_current_episode(dataset)
                if not args.no_home:
                    if left_robot is not None and left_home is not None:
                        print("Returning left arm to home pose...")
                        left_robot.move_joints(left_home, time_interval_s=args.home_time)
                    if right_robot is not None and right_home is not None:
                        print("Returning right arm to home pose...")
                        right_robot.move_joints(right_home, time_interval_s=args.home_time)

    except KeyboardInterrupt:
        print("Emergency stop requested. Shutting down...")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if left_robot is not None:
                left_robot.close()
            if right_robot is not None:
                right_robot.close()
            if spacemouse is not None:
                spacemouse.close()
            for stream in cam_streams.values():
                stream.stop()
            if dataset is not None:
                dataset.stop_image_writer()
        finally:
            signal.signal(signal.SIGINT, original_handler)
        print("Shutdown complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
