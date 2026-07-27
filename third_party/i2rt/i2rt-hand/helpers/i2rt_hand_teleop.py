#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEVISION_DIR = REPO_ROOT / "TeleVision"
I2RT_CONTROLLER_DIR = REPO_ROOT / "i2rt-controller"
for path in (REPO_ROOT, TELEVISION_DIR, I2RT_CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from i2rt_controller_teleop import (  # noqa: E402
    RealSenseConfig,
    RealSenseStream,
    build_image_feature,
    build_joint_names,
    delete_last_episode,
    discard_current_episode,
    ensure_realsense_serial,
    ensure_unique_dataset_target,
    get_frame_with_age,
    list_realsense_devices,
    maybe_save_episode,
    run_preflight_checks,
)
from i2rt.utils.quest_browser import add_quest_browser_args, launch_standard_quest_browser  # noqa: E402

from .dual_arm_helpers import (
    build_command,
    classify_left_hand_controls,
    ensure_can_interface_ready,
    maybe_limit_gripper_close,
    move_to_ready_pose,
    reset_to_home,
    setup_arm,
    sync_arm_state_from_robot,
)

from .gripper import update_gripper_from_pinch
from .quest_source import QuestSource
from .retargeter import QuestToYamRetargeter, RetargetConfig


YAM_GOLD_STANDARD_PROFILE = {
    "one_arm": "none",
    "pose_source": "wrist",
    "vuer_preprocessor": "open_television",
    "left_gripper": "linear_4310",
    "right_gripper": "linear_4310",
    "gripper_mode": "pinch",
    "gripper_pinch_source": "landmarks",
    "gripper_alpha": 1.0,
    "gripper_deadband": 0.0,
    "max_gripper_speed": 0.0,
    "gripper_force_threshold": 0.0,
    "pos_scale": 1.0,
    "translation_alpha": 0.60,
    "max_arm_joint_step": 0.030,
    "max_target_translation_speed": 0.60,
    "unlock_orientation": True,
    "orientation_scale": 0.90,
    "rotation_alpha": 0.70,
    "max_target_angular_speed": 2.40,
    "max_input_jump": 0.40,
    "max_input_rotation_jump": 3.14,
    "ik_ori_cost": 3.0,
    "stale_timeout": 0.15,
    "print_hz": 2.0,
    "ready_qpos": [0, 1.047, 1.047, 0, 0, 0, 1.0],
    "duration": 0.0,
}


def active_sides(args: argparse.Namespace) -> tuple[str, ...]:
    if args.one_arm == "left":
        return ("left",)
    if args.one_arm == "right":
        return ("right",)
    return ("left", "right")


def limit_arm_joint_step(target_q: np.ndarray, previous_q: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return target_q
    delta = np.asarray(target_q, dtype=np.float64) - np.asarray(previous_q, dtype=np.float64)
    return np.asarray(previous_q, dtype=np.float64) + np.clip(delta, -max_step, max_step)


def command_pose(
    arm_state: dict | None,
    pose: np.ndarray | None,
    max_step: float,
    dry_run: bool,
    active_joint_indices: tuple[int, ...] | None = None,
) -> tuple[bool, float | None]:
    if arm_state is None:
        return pose is None, None

    previous_q = arm_state["target_q"].copy()
    if pose is not None:
        ok, q = arm_state["kin"].ik(pose, init_q=arm_state["target_q"])
        if not ok:
            return False, None
        q = limit_arm_joint_step(q, previous_q, max_step)
        if active_joint_indices is not None:
            masked_q = previous_q.copy()
            masked_q[list(active_joint_indices)] = q[list(active_joint_indices)]
            q = masked_q
        arm_state["target_q"] = q
        arm_state["target_pose"] = arm_state["kin"].fk(q) if active_joint_indices is not None else pose.copy()
    else:
        q = previous_q

    step = float(np.max(np.abs(q - previous_q))) if q.size else 0.0
    cmd = build_command(
        q,
        arm_state["gripper_pos"],
        arm_state["arm_dofs"],
        arm_state["gripper_index"],
        arm_state["robot"].num_dofs(),
    )
    if not dry_run:
        arm_state["robot"].command_joint_pos(cmd)
    return True, step


def setup_side_arm(args: argparse.Namespace, side: str, ik_dt: float) -> dict:
    if side == "left":
        channel = args.left_channel
        gripper = args.left_gripper
        invert = args.left_gripper_invert
    else:
        channel = args.right_channel
        gripper = args.right_gripper
        invert = args.right_gripper_invert
    return setup_arm(
        channel,
        gripper,
        args.ik_frame,
        invert,
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


def target_pose_or_none(arm_state: dict | None) -> np.ndarray | None:
    return None if arm_state is None else arm_state["target_pose"]


def calibrate_from_current_pose(
    retargeter: QuestToYamRetargeter,
    frame,
    left_arm: dict | None,
    right_arm: dict | None,
) -> bool:
    return retargeter.calibrate(
        frame,
        target_pose_or_none(left_arm),
        target_pose_or_none(right_arm),
        now=frame.timestamp,
    )


def present_arms(left_arm: dict | None, right_arm: dict | None) -> tuple[dict, ...]:
    return tuple(arm for arm in (left_arm, right_arm) if arm is not None)


def active_arm_items(left_arm: dict | None, right_arm: dict | None) -> tuple[tuple[str, dict], ...]:
    items: list[tuple[str, dict]] = []
    if left_arm is not None:
        items.append(("left", left_arm))
    if right_arm is not None:
        items.append(("right", right_arm))
    return tuple(items)


def validate_active_joint_indices(values: list[int] | None, arm_dofs: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    indices = tuple(dict.fromkeys(int(value) for value in values))
    bad = [idx for idx in indices if idx < 0 or idx >= arm_dofs]
    if bad:
        raise ValueError(f"active arm joint indices {bad} are outside 0..{arm_dofs - 1}")
    return indices


def reset_active_arms_to_ready(
    arms: tuple[dict, ...],
    ready_qpos: np.ndarray,
    home_time: float,
    dry_run: bool,
) -> None:
    if not dry_run and arms:
        with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="ready") as executor:
            futures = [executor.submit(move_to_ready_pose, arm, ready_qpos, home_time) for arm in arms]
            for future in futures:
                future.result()
    for arm in arms:
        sync_arm_state_from_robot(arm)


def save_episode_while_moving_active_to_ready(
    dataset: LeRobotDataset,
    arms: tuple[dict, ...],
    ready_qpos: np.ndarray,
    home_time: float,
    dry_run: bool,
) -> bool:
    ready_qpos = np.asarray(ready_qpos, dtype=float)
    workers = 1 if dry_run else max(1, len(arms) + 1)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="save_ready") as executor:
        save_future = executor.submit(maybe_save_episode, dataset)
        ready_futures = []
        if not dry_run:
            ready_futures = [executor.submit(move_to_ready_pose, arm, ready_qpos, home_time) for arm in arms]
        for future in ready_futures:
            future.result()
        saved = save_future.result()
    for arm in arms:
        sync_arm_state_from_robot(arm)
    return saved


def _selected_camera_configs(
    args: argparse.Namespace,
    head_fps: int,
    wrist_fps: int,
) -> dict[str, tuple[str, int, int, int]]:
    return {
        "head": (args.head_serial, args.head_width, args.head_height, head_fps),
        "left_wrist": (args.left_wrist_serial, args.wrist_width, args.wrist_height, wrist_fps),
        "right_wrist": (args.right_wrist_serial, args.wrist_width, args.wrist_height, wrist_fps),
    }


def start_realsense_streams(
    args: argparse.Namespace,
    selected_cams: set[str],
    head_fps: int,
    wrist_fps: int,
) -> tuple[dict[str, RealSenseStream], dict[str, tuple[int, int]]]:
    cam_streams: dict[str, RealSenseStream] = {}
    cam_sizes: dict[str, tuple[int, int]] = {}
    cam_configs = _selected_camera_configs(args, head_fps, wrist_fps)

    for name in ("head", "left_wrist", "right_wrist"):
        if name not in selected_cams:
            continue
        serial, width, height, fps = cam_configs[name]
        stream = RealSenseStream(
            RealSenseConfig(serial, width, height, fps, allow_fallback=args.allow_camera_fallback)
        )
        stream.start()
        cam_streams[name] = stream
        video_profile = stream.profile.get_stream(rs.stream.color).as_video_stream_profile()
        cam_sizes[name] = (int(video_profile.width()), int(video_profile.height()))

    return cam_streams, cam_sizes


def create_lerobot_dataset(
    args: argparse.Namespace,
    arm_items: tuple[tuple[str, dict], ...],
    cam_sizes: dict[str, tuple[int, int]],
    dataset_fps: int,
    selected_cams: set[str],
) -> LeRobotDataset:
    state_names: list[str] = []
    for side, arm in arm_items:
        state_names.extend(build_joint_names(side, arm["robot"].num_dofs(), arm["gripper_index"]))
    action_names = list(state_names)

    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
    }
    for cam_name in ("head", "left_wrist", "right_wrist"):
        if cam_name not in selected_cams:
            continue
        width, height = cam_sizes[cam_name]
        features[f"observation.images.{cam_name}"] = build_image_feature(height, width)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=dataset_fps,
        features=features,
        robot_type=args.robot_type,
        root=args.dataset_root,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    def _encode_episode_videos_with_codec(episode_index: int) -> dict:
        def encode_one(key: str) -> tuple[str, str]:
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
            if not video_path.is_file():
                img_dir = dataset._get_image_file_path(
                    episode_index=episode_index,
                    image_key=key,
                    frame_index=0,
                ).parent
                encode_video_frames(img_dir, video_path, dataset.fps, vcodec=args.vcodec, overwrite=True)
            return key, str(video_path)

        video_keys = list(dataset.meta.video_keys)
        if not video_keys:
            return {}
        workers = max(1, min(args.video_encode_workers, len(video_keys)))
        if workers == 1:
            return dict(encode_one(key) for key in video_keys)
        print(f"[save] encoding {len(video_keys)} videos with {workers} workers.")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video_encode") as executor:
            return dict(executor.map(encode_one, video_keys))

    dataset.encode_episode_videos = _encode_episode_videos_with_codec
    return dataset


def add_lerobot_frame_if_fresh(
    *,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    arm_items: tuple[tuple[str, dict], ...],
    cam_streams: dict[str, RealSenseStream],
    now: float,
    stale_drop_count: int,
    last_stale_warn_time: float,
) -> tuple[int, float]:
    imgs: dict[str, np.ndarray] = {}
    ages: dict[str, float] = {}
    for cam_name, stream in cam_streams.items():
        img, age = get_frame_with_age(stream, now)
        if img is None:
            continue
        imgs[cam_name] = img
        ages[cam_name] = age

    have_all = len(imgs) == len(cam_streams)
    fresh_enough = have_all and all(age <= args.max_frame_age for age in ages.values())
    if fresh_enough:
        state_vec = np.concatenate(
            [arm["robot"].get_joint_pos() for _side, arm in arm_items]
        ).astype(np.float32)
        action_vec = np.concatenate(
            [np.asarray(_command_vector(arm), dtype=np.float64) for _side, arm in arm_items]
        ).astype(np.float32)
        frame_dict = {
            "observation.state": state_vec,
            "action": action_vec,
            "task": args.task,
        }
        for cam_name, img in imgs.items():
            frame_dict[f"observation.images.{cam_name}"] = img
        dataset.add_frame(frame_dict)
    elif have_all:
        stale_drop_count += 1
        if now - last_stale_warn_time > args.stale_warn_interval:
            worst_ms = max(ages.values()) * 1000.0
            per_cam = " ".join(f"{name}={age * 1000.0:.1f}" for name, age in ages.items())
            print(
                f"[stale] dropped {stale_drop_count} frames; "
                f"worst age {worst_ms:.1f} ms > {args.max_frame_age * 1000.0:.1f} ms ({per_cam})"
            )
            last_stale_warn_time = now

    return stale_drop_count, last_stale_warn_time


def apply_profile(args: argparse.Namespace, cli_options: set[str] | None = None) -> None:
    if args.profile != "yam-gold":
        return
    cli_options = cli_options or set()
    for name, value in YAM_GOLD_STANDARD_PROFILE.items():
        flag = f"--{name.replace('_', '-')}"
        if flag in cli_options:
            continue
        setattr(args, name, list(value) if isinstance(value, list) else value)


def format_delta_mm(diag, scale: float) -> str:
    delta = np.asarray(diag.raw_hand_delta, dtype=np.float64) * scale * 1000.0
    return f"({delta[0]:.1f},{delta[1]:.1f},{delta[2]:.1f})"


def format_gripper_debug(arm_state: dict | None, pinch: float | None) -> str:
    if arm_state is None or arm_state.get("gripper_index") is None:
        return "pinch=None cmd=None obs=None eff=None"
    obs_pos = None
    obs_eff = None
    try:
        obs = arm_state["robot"].get_observations()
        gripper_index = arm_state["gripper_index"]
        if obs is not None:
            gripper_pos = obs.get("gripper_pos")
            if gripper_pos is not None and len(gripper_pos) > 0:
                obs_pos = float(gripper_pos[0])
            joint_eff = obs.get("joint_eff")
            if joint_eff is not None and len(joint_eff) > gripper_index:
                obs_eff = float(joint_eff[gripper_index])
    except Exception:
        pass
    raw_goal = arm_state.get("gripper_raw_goal")
    goal = arm_state.get("gripper_goal")
    pos = arm_state.get("gripper_pos")
    return (
        f"pinch={pinch if pinch is not None else None} "
        f"raw={raw_goal if raw_goal is not None else None} "
        f"goal={goal if goal is not None else None} "
        f"cmd={pos if pos is not None else None} "
        f"obs={obs_pos if obs_pos is not None else None} "
        f"eff={obs_eff if obs_eff is not None else None}"
    )


def held_button_fired(
    *,
    button_name: str,
    action_name: str,
    pressed: bool,
    now: float,
    hold_seconds: float,
    hold_started_at: dict[str, float | None],
    hold_fired: dict[str, bool],
) -> bool:
    if hold_seconds <= 0.0:
        return pressed
    if not pressed:
        hold_started_at[button_name] = None
        hold_fired[button_name] = False
        return False

    started_at = hold_started_at.get(button_name)
    if started_at is None:
        hold_started_at[button_name] = now
        hold_fired[button_name] = False
        print(f"control_hold_start action={action_name} hold_seconds={hold_seconds:.1f}")
        return False

    if hold_fired.get(button_name, False):
        return False

    held = now - started_at
    if held < hold_seconds:
        return False

    hold_fired[button_name] = True
    print(f"control_hold_fire action={action_name} held_seconds={held:.1f}")
    return True


def _command_vector(arm_state: dict) -> list[float]:
    cmd = build_command(
        arm_state["target_q"],
        arm_state["gripper_pos"],
        arm_state["arm_dofs"],
        arm_state["gripper_index"],
        arm_state["robot"].num_dofs(),
    )
    return np.asarray(cmd, dtype=np.float64).reshape(-1).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest hand to i2rt/YAM teleop.")
    parser.add_argument(
        "--profile",
        choices=["cautious", "yam-gold"],
        default="cautious",
        help="Named parameter bundle. yam-gold is the current full bimanual all-7DOF profile.",
    )
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-mode", choices=["none", "pinch"], default="none")
    parser.add_argument("--gripper-alpha", type=float, default=0.25)
    parser.add_argument("--gripper-deadband", type=float, default=0.04)
    parser.add_argument("--max-gripper-speed", type=float, default=0.75)
    parser.add_argument("--gripper-pinch-source", choices=["auto", "landmarks", "state"], default="auto")
    parser.add_argument("--one-arm", choices=["left", "right", "none"], default="right")
    parser.add_argument("--pose-source", choices=["wrist", "palm"], default="wrist")
    parser.add_argument("--vuer-preprocessor", choices=["open_television", "legacy"], default="open_television")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--show-hands", action="store_true")
    parser.add_argument("--ngrok", action="store_true")
    add_quest_browser_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Run IK and print targets without commanding motors.")
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Active teleop seconds after calibration; <=0 runs until stopped.",
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--task", type=str, default="quest hand teleop dual-arm demo")
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--reset-time", type=float, default=10.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--video-encode-workers", type=int, default=3)
    parser.add_argument("--head-serial", type=str, default="<head-camera-serial>")
    parser.add_argument("--left-wrist-serial", type=str, default="<left-wrist-camera-serial>")
    parser.add_argument("--right-wrist-serial", type=str, default="<right-wrist-camera-serial>")
    parser.add_argument("--head-width", type=int, default=640)
    parser.add_argument("--head-height", type=int, default=480)
    parser.add_argument("--head-fps", type=int, default=30)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--allow-camera-fallback", action="store_true")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
        help="Which RealSense cameras to record into the LeRobot dataset.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--max-frame-age", type=float, default=0.050)
    parser.add_argument("--stale-warn-interval", type=float, default=1.0)
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Calibrate as soon as tracking is valid; otherwise wait for left fist.",
    )
    parser.add_argument("--hold-current-on-exit", action="store_true")
    parser.add_argument("--home-time", type=float, default=1.0)
    parser.add_argument("--ready-qpos", type=float, nargs="+", default=[0, 1.047, 1.047, 0, 0, 0, 0, 0])
    parser.add_argument("--ik-frame", type=str, default="link_6")
    parser.add_argument("--ik-alpha", type=float, default=0.2)
    parser.add_argument("--ik-pos-cost", type=float, default=10.0)
    parser.add_argument("--ik-ori-cost", type=float, default=0.2)
    parser.add_argument("--ik-posture-cost", type=float, default=1e-3)
    parser.add_argument("--ik-damping-cost", type=float, default=1e-1)
    parser.add_argument("--ik-lm-damping", type=float, default=1e-4)
    parser.add_argument("--ik-gain", type=float, default=0.5)
    parser.add_argument("--ik-solver", type=str, default=None)
    parser.add_argument("--ik-solve-damping", type=float, default=1e-12)
    parser.add_argument("--pos-scale", type=float, default=0.05)
    parser.add_argument("--orientation-scale", type=float, default=0.25)
    parser.add_argument("--unlock-orientation", action="store_true")
    parser.add_argument("--translation-alpha", type=float, default=0.18)
    parser.add_argument("--rotation-alpha", type=float, default=0.12)
    parser.add_argument("--max-target-translation-speed", type=float, default=0.05)
    parser.add_argument("--max-target-angular-speed", type=float, default=0.25)
    parser.add_argument("--max-input-jump", type=float, default=0.12)
    parser.add_argument("--max-input-rotation-jump", type=float, default=0.70)
    parser.add_argument("--stale-timeout", type=float, default=0.15)
    parser.add_argument("--max-arm-joint-step", type=float, default=0.002)
    parser.add_argument(
        "--active-arm-joints",
        type=int,
        nargs="*",
        default=None,
        help="Optional 0-based arm joint indices allowed to move after IK; omitted means all arm joints.",
    )
    parser.add_argument("--hand-extend-threshold", type=float, default=0.35)
    parser.add_argument("--hand-curl-threshold", type=float, default=0.55)
    parser.add_argument(
        "--control-hold-seconds",
        type=float,
        default=3.0,
        help="Continuous hold time required before X/save fires. A/fist, Y/discard, and B/stop remain immediate.",
    )
    parser.add_argument("--gripper-force-threshold", type=float, default=0.8)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=0.8)
    parser.add_argument("--gripper-backoff", type=float, default=0.02)
    parser.add_argument("--print-hz", type=float, default=2.0)
    args = parser.parse_args()
    if not args.list_cameras and not args.repo_id:
        parser.error("--repo-id is required for i2rt-hand recording")
    cli_options = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    apply_profile(args, cli_options)
    return args


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        for serial, name in list_realsense_devices():
            print(f"{serial}  {name}")
        return

    period = 1.0 / max(args.frequency, 1.0)
    dataset_fps = int(round(args.frequency))
    if abs(args.frequency - dataset_fps) > 1e-3:
        print(f"Warning: --frequency {args.frequency} is not integer; dataset fps set to {dataset_fps}.")
    head_fps = args.head_fps if args.head_fps is not None else dataset_fps
    wrist_fps = args.wrist_fps if args.wrist_fps is not None else dataset_fps

    ensure_unique_dataset_target(args)
    if not args.skip_preflight:
        ok = run_preflight_checks(args, dataset_fps)
        if args.preflight_only:
            return
        if not ok:
            return

    selected_cams = set(args.cameras)
    if "head" in selected_cams:
        ensure_realsense_serial(args.head_serial, "head")
    if "left_wrist" in selected_cams:
        ensure_realsense_serial(args.left_wrist_serial, "left wrist")
    if "right_wrist" in selected_cams:
        ensure_realsense_serial(args.right_wrist_serial, "right wrist")

    teleop_active_sides = active_sides(args)
    config = RetargetConfig(
        active_sides=teleop_active_sides,
        pose_source=args.pose_source,
        vuer_preprocessor=args.vuer_preprocessor,
        pos_scale=args.pos_scale,
        orientation_scale=args.orientation_scale,
        lock_orientation=not args.unlock_orientation,
        translation_alpha=args.translation_alpha,
        rotation_alpha=args.rotation_alpha,
        max_target_translation_speed=args.max_target_translation_speed,
        max_target_angular_speed=args.max_target_angular_speed,
        max_input_jump=args.max_input_jump,
        max_input_rotation_jump=args.max_input_rotation_jump,
        stale_timeout=args.stale_timeout,
        gripper_pinch_source=args.gripper_pinch_source,
    )
    retargeter = QuestToYamRetargeter(config)
    left_arm = None
    right_arm = None
    dataset: LeRobotDataset | None = None
    cam_streams: dict[str, RealSenseStream] = {}
    stop_reason = "unknown"
    recording = False
    episode_start_time: float | None = None
    stale_drop_count = 0
    last_stale_warn_time = 0.0

    try:
        for side in teleop_active_sides:
            ensure_can_interface_ready(args.left_channel if side == "left" else args.right_channel)
        ik_dt = period
        if "left" in teleop_active_sides:
            left_arm = setup_side_arm(args, "left", ik_dt)
        if "right" in teleop_active_sides:
            right_arm = setup_side_arm(args, "right", ik_dt)
        active_joint_indices = validate_active_joint_indices(
            args.active_arm_joints,
            max(arm["arm_dofs"] for arm in present_arms(left_arm, right_arm)),
        )

        cam_streams, cam_sizes = start_realsense_streams(args, selected_cams, head_fps, wrist_fps)
        arm_items = active_arm_items(left_arm, right_arm)
        dataset = create_lerobot_dataset(args, arm_items, cam_sizes, dataset_fps, selected_cams)

        ready = np.asarray(args.ready_qpos, dtype=np.float64)
        print("Moving active arms to ready pose...")
        reset_active_arms_to_ready(present_arms(left_arm, right_arm), ready, args.home_time, args.dry_run)
        print(
            "i2rt_hand_yam=ready "
            f"profile={args.profile} "
            f"active_sides={','.join(teleop_active_sides)} pose_source={args.pose_source} "
            f"pos_scale={args.pos_scale} max_target_speed={args.max_target_translation_speed} "
            f"max_joint_step={args.max_arm_joint_step} gripper_mode={args.gripper_mode} "
            f"gripper_pinch_source={args.gripper_pinch_source} "
            f"duration={args.duration} dry_run={args.dry_run} "
            f"active_arm_joints={active_joint_indices if active_joint_indices is not None else 'all'}"
        )
        print(f"dataset_root={dataset.root}")
        print("Quest Vuer URL: https://vuer.ai?ws=wss://localhost:8012")
        print("Quest Browser is launched with the standard local flow.")
        print("Quest display mode: clean passthrough; Vuer image background disabled.")
        print("Left fist recalibrates hand/head reference and starts recording.")
        print(
            f"Left thumbs-up held {args.control_hold_seconds:.1f}s saves the episode "
            "and returns active arms to ready."
        )
        print("Left middle finger immediately discards the current episode and returns active arms to ready.")
        print("Left middle finger while idle immediately deletes the last saved episode.")
        print("Left pinky stops the run immediately and returns active arms home.")

        hold_started_at = {"xButton": None}
        hold_fired = {"xButton": False}
        y_was_pressed = False
        last_print = 0.0
        next_tick = time.monotonic()

        with QuestSource(
            fps=args.fps,
            show_left=args.show_hands,
            show_right=args.show_hands,
            ngrok=args.ngrok,
        ) as source:
            launch_standard_quest_browser(args, ngrok=args.ngrok)
            while True:
                now_mono = time.monotonic()
                if now_mono < next_tick:
                    time.sleep(min(next_tick - now_mono, 0.01))
                    continue
                next_tick += period
                frame = source.latest_frame()
                buttons = classify_left_hand_controls(
                    frame.left_landmarks,
                    args.hand_extend_threshold,
                    args.hand_curl_threshold,
                )
                x_fired = held_button_fired(
                    button_name="xButton",
                    action_name="save",
                    pressed=buttons["xButton"],
                    now=now_mono,
                    hold_seconds=args.control_hold_seconds,
                    hold_started_at=hold_started_at,
                    hold_fired=hold_fired,
                )
                y_pressed = buttons["yButton"]
                y_fired = y_pressed and not y_was_pressed
                if y_fired:
                    print("control_immediate_fire action=discard_delete")
                y_was_pressed = y_pressed
                if buttons["bButton"]:
                    stop_reason = "quest_stop_gesture"
                    raise KeyboardInterrupt

                if y_fired:
                    if recording:
                        print("[discard] middle-finger gesture received; dropping current episode.")
                        discard_current_episode(dataset)
                        recording = False
                        episode_start_time = None
                    else:
                        delete_last_episode(dataset)
                    print("reset=ready moving_active_arms_to_ready")
                    reset_active_arms_to_ready(
                        present_arms(left_arm, right_arm),
                        ready,
                        args.home_time,
                        args.dry_run,
                    )
                    print("reset=ready done waiting_for_left_fist")
                    time.sleep(0.1)
                    next_tick = time.monotonic() + period
                    continue

                if x_fired:
                    if recording:
                        recording = False
                        episode_start_time = None
                        saved = save_episode_while_moving_active_to_ready(
                            dataset,
                            present_arms(left_arm, right_arm),
                            ready,
                            args.home_time,
                            args.dry_run,
                        )
                        if saved:
                            print(f"Episode {dataset.meta.total_episodes} saved.")
                        else:
                            print("No frames recorded; nothing to save.")
                            discard_current_episode(dataset)
                        if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                            stop_reason = "num_episodes_reached"
                            break
                        print("Ready pose reached. Make a left fist to start the next episode.")
                        time.sleep(0.1)
                        next_tick = time.monotonic() + period
                    else:
                        print("thumbs_up_ignored=no_active_episode")
                    continue

                if not recording:
                    if args.auto_start or buttons["aButton"]:
                        # Fist is the hand-gesture equivalent of controller A:
                        # each episode starts from a fresh hand/head reference.
                        ok = calibrate_from_current_pose(retargeter, frame, left_arm, right_arm)
                        if ok:
                            recording = True
                            episode_start_time = now_mono
                            stale_drop_count = 0
                            print("calibration=fresh_on_episode_start recording=started")
                        elif now_mono - last_print > 1.0 / max(args.print_hz, 0.1):
                            print(
                                "waiting_for_valid_calibration "
                                f"head={frame.head_valid} left={frame.left_valid} right={frame.right_valid}"
                            )
                            last_print = now_mono
                    elif now_mono - last_print > 1.0 / max(args.print_hz, 0.1):
                        print("waiting_for_left_fist")
                        last_print = now_mono
                    continue

                episode_elapsed = now_mono - episode_start_time if episode_start_time is not None else 0.0
                episode_limit = args.episode_time if args.episode_time > 0.0 else args.duration
                if episode_limit > 0.0 and episode_elapsed >= episode_limit:
                    recording = False
                    episode_start_time = None
                    saved = save_episode_while_moving_active_to_ready(
                        dataset,
                        present_arms(left_arm, right_arm),
                        ready,
                        args.home_time,
                        args.dry_run,
                    )
                    if saved:
                        print(f"Episode {dataset.meta.total_episodes} saved.")
                    else:
                        print("No frames recorded; nothing to save.")
                        discard_current_episode(dataset)
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        stop_reason = "num_episodes_reached"
                        break
                    if args.episode_time <= 0.0:
                        stop_reason = "duration_elapsed"
                        break
                    print("Ready pose reached. Make a left fist to start the next episode.")
                    time.sleep(0.1)
                    next_tick = time.monotonic() + period
                    continue

                output = retargeter.update(frame, dt=period, now=frame.timestamp)
                update_gripper_from_pinch(
                    left_arm,
                    output.left_gripper,
                    invert=args.left_gripper_invert,
                    mode=args.gripper_mode,
                    alpha=args.gripper_alpha,
                    deadband=args.gripper_deadband,
                    max_speed=args.max_gripper_speed,
                    dt=period,
                )
                update_gripper_from_pinch(
                    right_arm,
                    output.right_gripper,
                    invert=args.right_gripper_invert,
                    mode=args.gripper_mode,
                    alpha=args.gripper_alpha,
                    deadband=args.gripper_deadband,
                    max_speed=args.max_gripper_speed,
                    dt=period,
                )
                if args.gripper_mode != "none":
                    for arm in present_arms(left_arm, right_arm):
                        maybe_limit_gripper_close(
                            arm,
                            args.gripper_force_threshold,
                            args.gripper_force_ema_alpha,
                            args.gripper_backoff,
                        )

                left_pose = output.left_pose if "left" in teleop_active_sides else None
                right_pose = output.right_pose if "right" in teleop_active_sides else None
                left_ok, left_step = command_pose(
                    left_arm,
                    left_pose,
                    args.max_arm_joint_step,
                    args.dry_run,
                    active_joint_indices,
                )
                right_ok, right_step = command_pose(
                    right_arm,
                    right_pose,
                    args.max_arm_joint_step,
                    args.dry_run,
                    active_joint_indices,
                )
                stale_drop_count, last_stale_warn_time = add_lerobot_frame_if_fresh(
                    dataset=dataset,
                    args=args,
                    arm_items=arm_items,
                    cam_streams=cam_streams,
                    now=now_mono,
                    stale_drop_count=stale_drop_count,
                    last_stale_warn_time=last_stale_warn_time,
                )

                if now_mono - last_print > 1.0 / max(args.print_hz, 0.1):
                    ld = output.diagnostics["left"]
                    rd = output.diagnostics["right"]
                    print(
                        "teleop "
                        f"left_ok={left_ok} left_reason={ld.reason} left_step={left_step} "
                        f"left_xyz=({ld.target_xyz[0]:.3f},{ld.target_xyz[1]:.3f},{ld.target_xyz[2]:.3f}) "
                        f"left_cmd_delta_mm={format_delta_mm(ld, args.pos_scale)} "
                        f"right_ok={right_ok} right_reason={rd.reason} right_step={right_step} "
                        f"right_xyz=({rd.target_xyz[0]:.3f},{rd.target_xyz[1]:.3f},{rd.target_xyz[2]:.3f}) "
                        f"right_cmd_delta_mm={format_delta_mm(rd, args.pos_scale)} "
                        f"right_rot_deg=({rd.raw_rotation_deg:.1f}->{rd.filtered_rotation_deg:.1f}) "
                        f"right_grip=({format_gripper_debug(right_arm, output.right_gripper)})"
                    )
                    last_print = now_mono

    except KeyboardInterrupt:
        if stop_reason == "unknown":
            stop_reason = "keyboard_interrupt"
        print("\ni2rt_hand_yam=stopping")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if dataset is not None and recording:
                print("Discarding unsaved episode before shutdown. Use thumbs-up before pinky to save.")
                discard_current_episode(dataset)
            arms = present_arms(left_arm, right_arm)
            if arms:
                if args.hold_current_on_exit or args.dry_run:
                    for arm in arms:
                        sync_arm_state_from_robot(arm)
                        cmd = build_command(
                            arm["target_q"],
                            arm["gripper_pos"],
                            arm["arm_dofs"],
                            arm["gripper_index"],
                            arm["robot"].num_dofs(),
                        )
                        if not args.dry_run:
                            arm["robot"].command_joint_pos(cmd)
                else:
                    with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="home") as executor:
                        futures = [executor.submit(reset_to_home, arm, args.home_time) for arm in arms]
                        for future in futures:
                            future.result()
        finally:
            signal.signal(signal.SIGINT, original_handler)
            for cam in cam_streams.values():
                cam.stop()
            if dataset is not None:
                dataset.stop_image_writer()
            if left_arm is not None:
                left_arm["robot"].close()
            if right_arm is not None:
                right_arm["robot"].close()
            print("i2rt_hand_yam=shutdown_complete")


if __name__ == "__main__":
    main()
