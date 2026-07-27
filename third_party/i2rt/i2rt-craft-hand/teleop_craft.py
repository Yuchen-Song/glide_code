#!/usr/bin/env python3
"""Quest hand teleop for selected i2rt/YAM arms plus the CRAFT dexterous hand.

This active integrated path keeps the i2rt/YAM arm lifecycle in this folder and
loads CRAFT motion-tracking helpers from `craft-hand`.
"""

from __future__ import annotations

import argparse
import grp
import os
import pwd
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
I2RT_CONTROLLER_DIR = REPO_ROOT / "i2rt-controller"
for path in (REPO_ROOT, THIS_DIR, I2RT_CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers.module_loader import load_helper_package  # noqa: E402
from helpers.async_craft_io import AsyncCraftIO  # noqa: E402


load_helper_package("i2rt_hand_helpers", REPO_ROOT / "i2rt-hand" / "helpers")
load_helper_package("craft_hand_helpers", REPO_ROOT / "craft-hand" / "helpers")

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
from i2rt_hand_helpers.dual_arm_helpers import (  # noqa: E402
    build_command,
    classify_left_hand_controls,
    ensure_can_interface_ready,
    maybe_limit_gripper_close,
    move_to_ready_pose,
    reset_to_home,
    setup_arm,
    sync_arm_state_from_robot,
)
from i2rt_hand_helpers.gripper import update_gripper_from_pinch  # noqa: E402
from i2rt_hand_helpers.i2rt_hand_teleop import (  # noqa: E402
    QuestSource,
    QuestToYamRetargeter,
    RetargetConfig,
    command_pose,
    format_delta_mm,
    held_button_fired,
)
from i2rt_hand_helpers.types import QuestFrame  # noqa: E402
from craft_hand_helpers.dynamixel_io import CraftHandOutput, add_craft_output_args  # noqa: E402
from craft_hand_helpers.motor_config import (  # noqa: E402
    FINGER_MOTORS,
    SIDE_MOTORS,
    clamp_raw_for_motor,
    parse_motor_ids,
    raw_defaults,
)
from craft_hand_helpers.preprocess import QuestHandPreprocessor  # noqa: E402
from craft_hand_helpers.retarget import (  # noqa: E402
    TargetFilter,
    parse_active_fingers,
    parse_side_fingers,
    retarget_quest_landmarks_to_raw,
)
from i2rt.utils.quest_browser import add_quest_browser_args, launch_standard_quest_browser  # noqa: E402


MAX_CURRENT_LIMIT = 280
MAX_MOTION_SCALE = 2.0
MAX_THUMB_SCALE = 2.5
MAX_SIDE_SCALE = 0.80
MAX_THUMB_SIDE_SCALE = 1.20
MAX_STEP_RAW = 240
MAX_VELOCITY_RAW = 2500.0
MAX_SIDE_VELOCITY_RAW = 1400.0


CRAFT_HELPER_SOURCE = "craft-hand"
THUMB_FAST_STEP_MOTOR_IDS = (
    FINGER_MOTORS["thumb"]["mcp_forward"],
    SIDE_MOTORS["thumb"],
    FINGER_MOTORS["thumb"]["bend"],
)
DEFAULT_DATASET_BASE_ROOT = Path(
    os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot")
)
DEFAULT_DATASET_OWNER = os.environ.get("GLIDE_DATASET_OWNER", "")


TELEOP_MATRIX_KEYS = {
    "head": "teleoperation.matrices.head",
    "left_hand": "teleoperation.matrices.left_hand",
    "right_hand": "teleoperation.matrices.right_hand",
}
TELEOP_LANDMARK_KEYS = {
    "left": "teleoperation.landmark_matrices.left_hand",
    "right": "teleoperation.landmark_matrices.right_hand",
}
TELEOP_HAND_STATE_KEY = "teleoperation.hand_states"
TELEOP_TRACKING_KEY = "teleoperation.tracking"
HAND_STATE_FIELDS = ("pinch", "squeeze", "tap", "pinchValue", "squeezeValue", "tapValue")
HAND_STATE_NAMES = tuple(
    f"{side}_{field.replace('Value', '_value').lower()}"
    for side in ("left", "right")
    for field in HAND_STATE_FIELDS
)
TELEOP_TRACKING_NAMES = (
    "head_valid",
    "left_hand_valid",
    "right_hand_valid",
    "head_event_count",
    "left_hand_event_count",
    "right_hand_event_count",
    "head_packet_age_s",
    "left_hand_packet_age_s",
    "right_hand_packet_age_s",
)


COMBINED_YAM_GOLD_PROFILE = {
    "pose_source": "wrist",
    "vuer_preprocessor": "open_television",
    "left_gripper": "linear_4310",
    "right_gripper": "no_gripper",
    "gripper_mode": "pinch",
    "gripper_pinch_source": "landmarks",
    "gripper_alpha": 1.0,
    "gripper_deadband": 0.0,
    "max_gripper_speed": 0.0,
    "gripper_force_threshold": 0.0,
    "pos_scale": 1.3,
    "translation_alpha": 0.90,
    "max_arm_joint_step": 0.060,
    "max_target_translation_speed": 1.20,
    "unlock_orientation": True,
    "orientation_scale": 0.90,
    "rotation_alpha": 0.90,
    "max_target_angular_speed": 4.80,
    "max_input_jump": 0.40,
    "max_input_rotation_jump": 3.14,
    "ik_ori_cost": 3.0,
    "stale_timeout": 0.15,
    "print_hz": 2.0,
    "home_time": 2.0,
    "ready_qpos": [0, 1.047, 1.047, 0, 0, 0, 1.0],
    "duration": 0.0,
}


def active_sides(args: argparse.Namespace) -> tuple[str, ...]:
    if args.arm_sides is not None:
        if args.arm_sides == "both":
            return ("left", "right")
        return (args.arm_sides,)
    if args.one_arm == "left":
        return ("left",)
    if args.one_arm == "none":
        return ("left", "right")
    return ("right",)


def fix_dataset_owner(dataset_root: Path, owner: str | None) -> None:
    """Best-effort ownership repair for datasets created from sudo hardware runs."""
    if not owner:
        return
    if not dataset_root.exists():
        return
    try:
        user = pwd.getpwnam(owner)
    except KeyError:
        print(f"dataset_owner=skipped reason=unknown_user owner={owner}")
        return
    uid = user.pw_uid
    gid = user.pw_gid
    group_name = grp.getgrgid(gid).gr_name
    changed = 0
    failed = 0

    def chown_path(path: Path) -> None:
        nonlocal changed, failed
        try:
            stat = path.lstat()
            if stat.st_uid == uid and stat.st_gid == gid:
                return
            os.chown(path, uid, gid, follow_symlinks=False)
            changed += 1
        except OSError as exc:
            failed += 1
            if failed <= 3:
                print(f"dataset_owner_chown_failed path={path} reason={type(exc).__name__}: {exc}")

    chown_path(dataset_root)
    for dirpath, dirnames, filenames in os.walk(dataset_root, followlinks=False):
        for name in dirnames:
            chown_path(Path(dirpath) / name)
        for name in filenames:
            chown_path(Path(dirpath) / name)
    status = "ok" if failed == 0 else "partial"
    print(f"dataset_owner={status} owner={owner}:{group_name} root={dataset_root} changed={changed} failed={failed}")


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


def target_pose_or_none(arm_state: dict | None) -> np.ndarray | None:
    return None if arm_state is None else arm_state["target_pose"]


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


def save_episode_while_resetting(
    *,
    dataset: LeRobotDataset,
    arms: tuple[dict, ...],
    ready_qpos: np.ndarray,
    home_time: float,
    dry_run: bool,
    craft: CraftHandOutput | None,
) -> bool:
    workers = 1 if dry_run else max(1, len(arms) + 2)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="save_ready") as executor:
        save_future = executor.submit(maybe_save_episode, dataset)
        futures = []
        if not dry_run:
            futures.extend(executor.submit(move_to_ready_pose, arm, ready_qpos, home_time) for arm in arms)
        if craft is not None:
            futures.append(executor.submit(craft.move_to_defaults, "save_default"))
        for future in futures:
            future.result()
        saved = save_future.result()
    for arm in arms:
        sync_arm_state_from_robot(arm)
    return saved


def reset_outputs_to_ready(
    *,
    arms: tuple[dict, ...],
    ready_qpos: np.ndarray,
    home_time: float,
    dry_run: bool,
    craft: CraftHandOutput | None,
) -> None:
    futures = []
    workers = max(1, len(arms) + (1 if craft is not None else 0))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reset_ready") as executor:
        if not dry_run:
            futures.extend(executor.submit(move_to_ready_pose, arm, ready_qpos, home_time) for arm in arms)
        if craft is not None:
            futures.append(executor.submit(craft.move_to_defaults, "reset_default"))
        for future in futures:
            future.result()
    for arm in arms:
        sync_arm_state_from_robot(arm)


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


def craft_joint_names(motor_ids: list[int]) -> list[str]:
    return [f"craft_motor_{motor_id}_raw" for motor_id in motor_ids]


def build_matrix_feature() -> dict:
    return {"dtype": "float32", "shape": (4, 4), "names": ["row", "column"]}


def build_landmark_matrix_feature() -> dict:
    return {"dtype": "float32", "shape": (25, 4, 4), "names": ["joint", "row", "column"]}


def build_teleoperation_raw_features() -> dict:
    features = {key: build_matrix_feature() for key in TELEOP_MATRIX_KEYS.values()}
    features.update({key: build_landmark_matrix_feature() for key in TELEOP_LANDMARK_KEYS.values()})
    features[TELEOP_HAND_STATE_KEY] = {
        "dtype": "float32",
        "shape": (len(HAND_STATE_NAMES),),
        "names": list(HAND_STATE_NAMES),
    }
    features[TELEOP_TRACKING_KEY] = {
        "dtype": "float32",
        "shape": (len(TELEOP_TRACKING_NAMES),),
        "names": list(TELEOP_TRACKING_NAMES),
    }
    return features


def matrix_or_zero(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((4, 4), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        return np.zeros((4, 4), dtype=np.float32)
    return arr


def raw_landmark_matrices_or_zero(source: QuestSource, side: str, valid: bool) -> np.ndarray:
    if not valid:
        return np.zeros((25, 4, 4), dtype=np.float32)
    shared = source.left_landmarks_shared if side == "left" else source.right_landmarks_shared
    with shared.get_lock():
        raw = np.asarray(shared[:], dtype=np.float64)
    if raw.size < 25 * 16 or not np.isfinite(raw[: 25 * 16]).all():
        return np.zeros((25, 4, 4), dtype=np.float32)
    flat = raw[: 25 * 16].reshape(25, 16)
    return np.asarray([flat[idx].reshape(4, 4, order="F") for idx in range(25)], dtype=np.float32)


def hand_state_vector(frame: QuestFrame) -> np.ndarray:
    values: list[float] = []
    for side in ("left", "right"):
        state = frame.hand_state(side)
        for field in HAND_STATE_FIELDS:
            value = 0.0 if state is None else state.get(field, 0.0)
            values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def packet_age_s(frame_timestamp: float, packet_timestamp: float) -> float:
    if packet_timestamp <= 0.0:
        return 0.0
    return max(0.0, float(frame_timestamp - packet_timestamp))


def tracking_vector(frame: QuestFrame) -> np.ndarray:
    return np.asarray(
        [
            float(frame.head_valid),
            float(frame.left_valid),
            float(frame.right_valid),
            float(frame.head_event_count),
            float(frame.left_event_count),
            float(frame.right_event_count),
            packet_age_s(frame.timestamp, frame.head_last_timestamp),
            packet_age_s(frame.timestamp, frame.left_last_timestamp),
            packet_age_s(frame.timestamp, frame.right_last_timestamp),
        ],
        dtype=np.float32,
    )


def collect_teleoperation_raw(source: QuestSource, frame: QuestFrame) -> dict[str, np.ndarray]:
    return {
        TELEOP_MATRIX_KEYS["head"]: matrix_or_zero(frame.head_mat),
        TELEOP_MATRIX_KEYS["left_hand"]: matrix_or_zero(frame.left_hand_mat),
        TELEOP_MATRIX_KEYS["right_hand"]: matrix_or_zero(frame.right_hand_mat),
        TELEOP_LANDMARK_KEYS["left"]: raw_landmark_matrices_or_zero(source, "left", frame.left_valid),
        TELEOP_LANDMARK_KEYS["right"]: raw_landmark_matrices_or_zero(source, "right", frame.right_valid),
        TELEOP_HAND_STATE_KEY: hand_state_vector(frame),
        TELEOP_TRACKING_KEY: tracking_vector(frame),
    }


def create_combined_dataset(
    args: argparse.Namespace,
    arm_items: tuple[tuple[str, dict], ...],
    craft_motor_ids: list[int],
    cam_sizes: dict[str, tuple[int, int]],
    dataset_fps: int,
    selected_cams: set[str],
) -> LeRobotDataset:
    state_names: list[str] = []
    for side, arm in arm_items:
        state_names.extend(build_joint_names(side, arm["robot"].num_dofs(), arm["gripper_index"]))
    state_names.extend(craft_joint_names(craft_motor_ids))
    action_names = list(state_names)
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
    }
    features.update(build_teleoperation_raw_features())
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


def command_vector(arm_state: dict) -> np.ndarray:
    return np.asarray(
        build_command(
            arm_state["target_q"],
            arm_state["gripper_pos"],
            arm_state["arm_dofs"],
            arm_state["gripper_index"],
            arm_state["robot"].num_dofs(),
        ),
        dtype=np.float64,
    ).reshape(-1)


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


def add_lerobot_frame_if_fresh(
    *,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    arm_items: tuple[tuple[str, dict], ...],
    craft_state_cache: CraftStateCache,
    craft_action_targets: dict[int, int],
    teleop_raw: dict[str, np.ndarray],
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
        craft_action_vec = np.asarray(
            [craft_action_targets[motor_id] for motor_id in craft_state_cache.motor_ids],
            dtype=np.float64,
        )
        state_parts = [arm["robot"].get_joint_pos() for _side, arm in arm_items]
        action_parts = [command_vector(arm) for _side, arm in arm_items]
        if craft_state_cache.motor_ids:
            state_parts.append(craft_state_cache.vector())
            action_parts.append(craft_action_vec)
        frame_dict = {
            "observation.state": np.concatenate(state_parts).astype(np.float32),
            "action": np.concatenate(action_parts).astype(np.float32),
            "task": args.task,
        }
        frame_dict.update(teleop_raw)
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


def preprocess_craft_landmarks(
    mode: str,
    preprocessor: QuestHandPreprocessor,
    landmarks_3d,
    wrist_matrix,
):
    if landmarks_3d is None or mode == "none":
        return landmarks_3d
    if mode == "wrist-relative":
        return preprocessor.process(landmarks_3d, wrist_matrix, robot_frame=False, hand_frame="none")
    return preprocessor.process(landmarks_3d, wrist_matrix, robot_frame=True, hand_frame="inspire")


def compact_targets(targets: dict[int, int], motor_ids: list[int]) -> str:
    return " ".join(f"{motor_id}:{targets[motor_id]}" for motor_id in motor_ids if motor_id in targets)


def signal_summary(signals: dict[str, float]) -> str:
    keys = (
        "craft_inactive_no_right_arm",
        "craft_disabled",
        "craft_hold_previous",
        "craft_lost_defaults",
        "craft_start_delay",
        "thumb_drive",
        "thumb_curl_signal",
        "thumb_metric_signal",
        "thumb_geom",
        "thumb_geom_raw",
        "thumb_geom_deadzone",
        "thumb_palm_close",
        "thumb_mcp_close",
        "thumb_index_close",
        "thumb_cal_side_axis",
        "thumb_side_limit",
        "index_drive",
        "middle_drive",
        "ring_drive",
        "pinky_drive",
        "thumb_side",
        "thumb_side_source",
        "thumb_side_suppression",
        "index_side",
        "middle_side",
        "ring_side",
        "pinky_side",
        "open_palm_signal",
        "open_palm_release",
        "thumb_open_release",
        "index_open_release",
        "middle_open_release",
        "ring_open_release",
        "pinky_open_release",
    )
    parts = []
    for key in keys:
        value = signals.get(key)
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts)


def format_optional_ms(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}"


def apply_thumb_step_override(
    *,
    target_filter: TargetFilter,
    filtered_targets: dict[int, int],
    raw_targets: dict[int, int],
    previous_targets: dict[int, int],
    max_step_raw: int,
) -> dict[int, int]:
    if max_step_raw <= 0:
        return filtered_targets

    adjusted = dict(filtered_targets)
    changed = False
    for motor_id in THUMB_FAST_STEP_MOTOR_IDS:
        if motor_id not in raw_targets:
            continue
        previous_raw = previous_targets.get(motor_id)
        if previous_raw is None:
            previous_raw = target_filter.previous_targets.get(motor_id, raw_targets[motor_id])
        desired_raw = raw_targets[motor_id]
        delta = max(-max_step_raw, min(max_step_raw, int(desired_raw) - int(previous_raw)))
        stepped_raw = clamp_raw_for_motor(motor_id, int(previous_raw) + delta)
        if adjusted.get(motor_id) != stepped_raw:
            adjusted[motor_id] = stepped_raw
            changed = True

    if changed:
        target_filter.previous_targets = dict(target_filter.previous_targets)
        for motor_id in THUMB_FAST_STEP_MOTOR_IDS:
            if motor_id not in adjusted:
                continue
            target_filter.previous_targets[motor_id] = adjusted[motor_id]
            state = target_filter.one_euro_states.get(motor_id)
            if state is not None:
                state.value = float(adjusted[motor_id])
    return adjusted


def retarget_craft(
    *,
    args: argparse.Namespace,
    frame,
    preprocessor: QuestHandPreprocessor,
    target_filter: TargetFilter,
    active_fingers: tuple[str, ...],
    side_fingers: tuple[str, ...],
    default_targets: dict[int, int],
    previous_targets: dict[int, int],
    episode_started_at: float | None,
    last_seen_right_hand_at: float | None,
    now: float,
) -> tuple[dict[int, int], dict[str, float], float | None]:
    raw_landmarks = frame.right_landmarks if frame.right_valid else None
    if raw_landmarks is None:
        if last_seen_right_hand_at is not None and now - last_seen_right_hand_at >= args.craft_lost_hand_timeout:
            return (
                target_filter.filter(default_targets, now=now),
                {"craft_lost_defaults": 1.0},
                last_seen_right_hand_at,
            )
        return previous_targets, {"craft_hold_previous": 1.0}, last_seen_right_hand_at

    last_seen_right_hand_at = now
    if episode_started_at is not None and now - episode_started_at < args.craft_start_delay:
        return target_filter.filter(default_targets, now=now), {"craft_start_delay": 1.0}, last_seen_right_hand_at

    landmarks = preprocess_craft_landmarks(args.craft_preprocess, preprocessor, raw_landmarks, frame.right_hand_mat)
    raw_targets, signals = retarget_quest_landmarks_to_raw(
        landmarks,
        motion_scale=args.motion_scale,
        include_mcp=args.include_mcp,
        include_side=args.include_side,
        side_scale=args.side_scale,
        side_deadzone_value=args.side_deadzone,
        side_angle_range=args.side_angle_range,
        side_fingers=side_fingers,
        mcp_forward_ratio=args.mcp_forward_ratio,
        thumb_scale=args.thumb_scale,
        thumb_mode=args.thumb_mode,
        thumb_geometry_scale=args.thumb_geometry_scale,
        thumb_geometry_deadzone=args.thumb_geometry_deadzone,
        thumb_forward_ratio=args.thumb_forward_ratio,
        thumb_side_scale=args.thumb_side_scale,
        thumb_retarget=args.thumb_retarget,
        thumb_side_gain=args.thumb_side_gain,
        thumb_side_limit=args.thumb_side_limit,
        thumb_side_sign=args.thumb_side_sign,
        open_palm_release=args.open_palm_release,
        open_palm_threshold=args.open_palm_threshold,
        open_palm_transition=args.open_palm_transition,
        open_palm_blend=args.open_palm_blend,
        open_palm_reset_side=not args.open_palm_keep_side,
        per_finger_open_release=args.per_finger_open_release,
        active_fingers=active_fingers,
        curl_deadzone_value=args.curl_deadzone,
        thumb_curl_deadzone_value=args.thumb_curl_deadzone,
        landmark_format=args.landmark_format,
    )
    filtered_targets = target_filter.filter(raw_targets, now=now)
    filtered_targets = apply_thumb_step_override(
        target_filter=target_filter,
        filtered_targets=filtered_targets,
        raw_targets=raw_targets,
        previous_targets=previous_targets,
        max_step_raw=args.thumb_max_step_raw,
    )
    return filtered_targets, signals, last_seen_right_hand_at


def apply_profile(args: argparse.Namespace, cli_options: set[str] | None = None) -> None:
    if args.profile != "yam-gold":
        return
    cli_options = cli_options or set()
    for name, value in COMBINED_YAM_GOLD_PROFILE.items():
        flag = f"--{name.replace('_', '-')}"
        if name == "unlock_orientation" and "--lock-orientation" in cli_options:
            continue
        if flag in cli_options:
            continue
        setattr(args, name, list(value) if isinstance(value, list) else value)


def resolve_frequency_args(
    args: argparse.Namespace,
    cli_options: set[str],
    parser: argparse.ArgumentParser,
) -> None:
    if args.arm_frequency is None:
        args.arm_frequency = args.frequency
    elif "--frequency" in cli_options and abs(args.frequency - args.arm_frequency) > 1e-6:
        parser.error("--frequency and --arm-frequency were both set differently; use only one arm loop flag")
    else:
        args.frequency = args.arm_frequency

    if args.craft_command_hz is None:
        args.craft_command_hz = args.craft_frequency


def create_craft_target_filter(args: argparse.Namespace, initial_targets: dict[int, int]) -> TargetFilter:
    return TargetFilter(
        initial_targets=initial_targets,
        filter_mode=args.filter_mode,
        smoothing=args.smoothing,
        max_step_raw=args.max_step_raw,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        max_velocity_raw=args.max_velocity_raw,
        side_max_velocity_raw=args.side_max_velocity_raw,
        nominal_hz=args.craft_frequency,
    )


def validate_args(args: argparse.Namespace) -> None:
    checks = (
        ("--frequency", args.frequency, 1.0, 60.0),
        ("--arm-frequency", args.arm_frequency, 1.0, 60.0),
        ("--craft-frequency", args.craft_frequency, 1.0, 60.0),
        ("--motion-scale", args.motion_scale, 0.0, MAX_MOTION_SCALE),
        ("--thumb-scale", args.thumb_scale, 0.0, MAX_THUMB_SCALE),
        ("--thumb-geometry-scale", args.thumb_geometry_scale, 0.05, 3.0),
        ("--thumb-geometry-deadzone", args.thumb_geometry_deadzone, 0.0, 0.95),
        ("--thumb-forward-ratio", args.thumb_forward_ratio, 0.0, 1.0),
        ("--curl-deadzone", args.curl_deadzone, 0.0, 0.50),
        ("--mcp-forward-ratio", args.mcp_forward_ratio, 0.0, 1.0),
        ("--thumb-curl-deadzone", args.thumb_curl_deadzone, 0.0, 0.50),
        ("--side-scale", args.side_scale, 0.0, MAX_SIDE_SCALE),
        ("--thumb-side-gain", args.thumb_side_gain, 0.0, MAX_THUMB_SIDE_SCALE),
        ("--thumb-side-limit", args.thumb_side_limit, 0.0, 1.0),
        ("--side-deadzone", args.side_deadzone, 0.0, 0.80),
        ("--side-angle-range", args.side_angle_range, 0.05, 2.0),
        ("--open-palm-threshold", args.open_palm_threshold, 0.0, 1.0),
        ("--open-palm-transition", args.open_palm_transition, 0.001, 1.0),
        ("--open-palm-blend", args.open_palm_blend, 0.0, 1.0),
        ("--current-limit", args.current_limit, 0, MAX_CURRENT_LIMIT),
        ("--max-step-raw", args.max_step_raw, 0, MAX_STEP_RAW),
        ("--thumb-max-step-raw", args.thumb_max_step_raw, 0, MAX_STEP_RAW),
        ("--max-velocity-raw", args.max_velocity_raw, 0.0, MAX_VELOCITY_RAW),
        ("--side-max-velocity-raw", args.side_max_velocity_raw, 0.0, MAX_SIDE_VELOCITY_RAW),
        ("--craft-lost-hand-timeout", args.craft_lost_hand_timeout, 0.0, 5.0),
        ("--craft-state-read-hz", args.craft_state_read_hz, 0.0, 30.0),
        ("--craft-command-hz", args.craft_command_hz, 1.0, 60.0),
    )
    for flag, value, low, high in checks:
        if not low <= value <= high:
            raise ValueError(f"{flag} must be between {low} and {high}; got {value}")
    if args.thumb_side_scale is not None and not 0.0 <= args.thumb_side_scale <= MAX_THUMB_SIDE_SCALE:
        raise ValueError(
            f"--thumb-side-scale must be between 0.0 and {MAX_THUMB_SIDE_SCALE}; got {args.thumb_side_scale}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["yam-gold"], default="yam-gold")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument(
        "--arm-sides",
        choices=["right", "both"],
        default=None,
        help=(
            "Required for recording runs. Use right for right YAM plus CRAFT, "
            "or both for left YAM plus right YAM and CRAFT."
        ),
    )
    parser.add_argument(
        "--one-arm",
        choices=["left", "right", "none"],
        default="none",
        help="Compatibility with i2rt-hand: left/right select one arm, none means both arms.",
    )
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="no_gripper")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-mode", choices=["none", "pinch"], default="none")
    parser.add_argument("--gripper-alpha", type=float, default=1.0)
    parser.add_argument("--gripper-deadband", type=float, default=0.0)
    parser.add_argument("--max-gripper-speed", type=float, default=0.0)
    parser.add_argument("--gripper-pinch-source", choices=["auto", "landmarks", "state"], default="landmarks")
    parser.add_argument("--pose-source", choices=["wrist", "palm"], default="wrist")
    parser.add_argument("--vuer-preprocessor", choices=["open_television", "legacy"], default="open_television")
    parser.add_argument(
        "--frequency",
        type=float,
        default=45.0,
        help="Arm loop and dataset FPS. Kept for compatibility; equivalent to --arm-frequency.",
    )
    parser.add_argument(
        "--arm-frequency",
        type=float,
        default=None,
        help="Explicit arm loop and dataset FPS. Overrides --frequency when provided.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--show-hands", dest="show_hands", action="store_true", default=True)
    parser.add_argument("--hide-hands", dest="show_hands", action="store_false")
    parser.add_argument("--ngrok", action="store_true")
    add_quest_browser_args(parser)
    parser.set_defaults(quest_adb_serial=os.environ.get("GLIDE_QUEST_ADB_SERIAL"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run IK and CRAFT retargeting without commanding motors.",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Exact LeRobot dataset root. If omitted, defaults to "
            f"{DEFAULT_DATASET_BASE_ROOT}/<repo-id>."
        ),
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--dataset-owner",
        type=str,
        default=DEFAULT_DATASET_OWNER,
        help="Best-effort chown target for generated dataset files. Use empty string to disable.",
    )
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm_craft_hand")
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
        "--skip-camera-serial-check",
        action="store_true",
        help="Skip RealSense device enumeration and open explicitly configured camera serials directly.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--max-frame-age", type=float, default=0.050)
    parser.add_argument("--stale-warn-interval", type=float, default=1.0)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--hold-current-on-exit", action="store_true")
    parser.add_argument("--home-time", type=float, default=2.0)
    parser.add_argument("--ready-qpos", type=float, nargs="+", default=[0, 1.047, 1.047, 0, 0, 0])
    parser.add_argument("--ik-frame", type=str, default="link_6")
    parser.add_argument("--ik-alpha", type=float, default=0.2)
    parser.add_argument("--ik-pos-cost", type=float, default=10.0)
    parser.add_argument("--ik-ori-cost", type=float, default=3.0)
    parser.add_argument("--ik-posture-cost", type=float, default=1e-3)
    parser.add_argument("--ik-damping-cost", type=float, default=1e-1)
    parser.add_argument("--ik-lm-damping", type=float, default=1e-4)
    parser.add_argument("--ik-gain", type=float, default=0.5)
    parser.add_argument("--ik-solver", type=str, default=None)
    parser.add_argument("--ik-solve-damping", type=float, default=1e-12)
    parser.add_argument("--pos-scale", type=float, default=0.5)
    parser.add_argument("--orientation-scale", type=float, default=0.90)
    parser.add_argument("--unlock-orientation", action="store_true", default=True)
    parser.add_argument("--lock-orientation", dest="unlock_orientation", action="store_false")
    parser.add_argument("--translation-alpha", type=float, default=0.60)
    parser.add_argument("--rotation-alpha", type=float, default=0.70)
    parser.add_argument("--max-target-translation-speed", type=float, default=0.60)
    parser.add_argument("--max-target-angular-speed", type=float, default=2.40)
    parser.add_argument("--max-input-jump", type=float, default=0.40)
    parser.add_argument("--max-input-rotation-jump", type=float, default=3.14)
    parser.add_argument("--stale-timeout", type=float, default=0.15)
    parser.add_argument("--max-arm-joint-step", type=float, default=0.030)
    parser.add_argument("--active-arm-joints", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--hand-extend-threshold", type=float, default=0.35)
    parser.add_argument("--hand-curl-threshold", type=float, default=0.55)
    parser.add_argument("--control-hold-seconds", type=float, default=3.0)
    parser.add_argument("--gripper-force-threshold", type=float, default=0.0)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=0.8)
    parser.add_argument("--gripper-backoff", type=float, default=0.02)
    parser.add_argument("--print-hz", type=float, default=2.0)

    parser.add_argument(
        "--craft-hand-mode",
        choices=["disabled", "shadow", "drive"],
        default="drive",
        help=(
            "CRAFT hand behavior: disabled leaves hardware untouched and records held defaults; "
            "shadow retargets/records without connecting; drive connects and commands motors."
        ),
    )
    parser.add_argument("--drive-craft", action="store_true", help="Legacy alias for --craft-hand-mode drive.")
    parser.add_argument(
        "--craft-preprocess",
        choices=["none", "wrist-relative", "open-television"],
        default="open-television",
    )
    parser.add_argument("--craft-start-delay", type=float, default=0.5)
    parser.add_argument("--craft-lost-hand-timeout", type=float, default=0.50)
    parser.add_argument(
        "--craft-frequency",
        type=float,
        default=15.0,
        help="CRAFT retarget/filter update rate, independent of the arm loop.",
    )
    parser.add_argument("--craft-state-read-hz", type=float, default=0.0)
    parser.add_argument(
        "--craft-io-mode",
        choices=["async", "sync"],
        default="async",
        help="Use async CRAFT serial I/O so hand writes cannot block the arm loop, or legacy sync I/O.",
    )
    parser.add_argument(
        "--craft-command-hz",
        type=float,
        default=None,
        help="Async CRAFT write rate. Defaults to --craft-frequency.",
    )
    parser.add_argument("--motion-scale", type=float, default=1.1025)
    parser.add_argument("--thumb-scale", type=float, default=2.15)
    parser.add_argument(
        "--thumb-mode",
        choices=["curl", "geometric", "hybrid"],
        default="hybrid",
        help=(
            "Thumb bend source for CRAFT: curl uses thumb joint angles, geometric uses thumb-tip closeness, "
            "and hybrid uses the stronger signal from both."
        ),
    )
    parser.add_argument(
        "--thumb-geometry-scale",
        type=float,
        default=0.65,
        help="Gain for the geometric thumb-tip closeness fallback used by geometric and hybrid thumb modes.",
    )
    parser.add_argument(
        "--thumb-geometry-deadzone",
        type=float,
        default=0.25,
        help="Ignore geometric thumb fallback below this value so a straight/open thumb stays open.",
    )
    parser.add_argument("--thumb-forward-ratio", type=float, default=0.95)
    parser.add_argument("--curl-deadzone", type=float, default=0.035)
    parser.add_argument("--mcp-forward-ratio", type=float, default=0.80)
    parser.add_argument("--thumb-curl-deadzone", type=float, default=0.18)
    parser.add_argument("--include-mcp", dest="include_mcp", action="store_true", default=True)
    parser.add_argument("--no-include-mcp", dest="include_mcp", action="store_false")
    parser.add_argument("--include-side", dest="include_side", action="store_true", default=True)
    parser.add_argument("--no-include-side", dest="include_side", action="store_false")
    parser.add_argument("--side-scale", type=float, default=0.45)
    parser.add_argument("--thumb-side-scale", type=float, default=1.10)
    parser.add_argument("--side-deadzone", type=float, default=0.10)
    parser.add_argument("--side-angle-range", type=float, default=0.35)
    parser.add_argument(
        "--thumb-retarget",
        choices=["generic", "generic-centered", "side-calibrated", "calibrated"],
        default="side-calibrated",
        help="Thumb mapping mode. generic-centered/side-calibrated keep generic curl and use calibrated thumb side.",
    )
    parser.add_argument("--thumb-side-gain", type=float, default=1.0)
    parser.add_argument(
        "--thumb-side-limit",
        type=float,
        default=1.0,
        help="Cap calibrated thumb side command after gain. Use below 1.0 to reduce mechanical coupling.",
    )
    parser.add_argument("--thumb-side-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--active-fingers", default="all")
    parser.add_argument("--side-fingers", default="all")
    parser.add_argument("--landmark-format", choices=["auto", "webxr25", "openpose21"], default="auto")
    parser.add_argument("--open-palm-release", dest="open_palm_release", action="store_true", default=False)
    parser.add_argument("--no-open-palm-release", dest="open_palm_release", action="store_false")
    parser.add_argument("--open-palm-threshold", type=float, default=0.24)
    parser.add_argument("--open-palm-transition", type=float, default=0.08)
    parser.add_argument("--open-palm-blend", type=float, default=1.0)
    parser.add_argument("--per-finger-open-release", dest="per_finger_open_release", action="store_true", default=True)
    parser.add_argument("--no-per-finger-open-release", dest="per_finger_open_release", action="store_false")
    parser.add_argument("--open-palm-keep-side", action="store_true", default=True)
    parser.add_argument("--open-palm-reset-side", dest="open_palm_keep_side", action="store_false")
    parser.add_argument("--filter-mode", choices=tuple(sorted(TargetFilter.VALID_MODES)), default="ema")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--max-step-raw", type=int, default=80)
    parser.add_argument(
        "--thumb-max-step-raw",
        type=int,
        default=240,
        help="Per-update step cap for thumb forward/side/bend motors after filtering. Set 0 to disable.",
    )
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.6)
    parser.add_argument("--one-euro-beta", type=float, default=0.16)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0)
    parser.add_argument("--max-velocity-raw", type=float, default=1800.0)
    parser.add_argument("--side-max-velocity-raw", type=float, default=700.0)
    add_craft_output_args(parser)
    parser.set_defaults(craft_baudrate=1000000, current_limit=230, default_ramp_seconds=2.0)

    args = parser.parse_args()
    cli_options = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    if not args.list_cameras:
        required_flags = ("--repo-id", "--num-episodes", "--task", "--arm-sides")
        missing_flags = [flag for flag in required_flags if flag not in cli_options]
        if missing_flags:
            parser.error(f"recording runs require explicit {' '.join(missing_flags)}")
        if "--dataset-root" not in cli_options:
            args.dataset_root = str(DEFAULT_DATASET_BASE_ROOT / args.repo_id)
    if args.craft_hand_mode is None:
        args.craft_hand_mode = "drive" if args.drive_craft else "shadow"
    elif args.drive_craft and args.craft_hand_mode != "drive":
        parser.error("--drive-craft conflicts with --craft-hand-mode disabled/shadow")
    args.drive_craft = args.craft_hand_mode == "drive"
    if "--arm-sides" in cli_options and "--one-arm" in cli_options:
        parser.error("Use either --arm-sides or --one-arm, not both")
    apply_profile(args, cli_options)
    resolve_frequency_args(args, cli_options, parser)
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        for serial, name in list_realsense_devices():
            print(f"{serial}  {name}")
        return

    period = 1.0 / max(args.arm_frequency, 1.0)
    craft_period = 1.0 / max(args.craft_frequency, 1.0)
    dataset_fps = int(round(args.arm_frequency))
    if abs(args.arm_frequency - dataset_fps) > 1e-3:
        print(f"Warning: --arm-frequency {args.arm_frequency} is not integer; dataset fps set to {dataset_fps}.")
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
    if not args.skip_camera_serial_check:
        if "head" in selected_cams:
            ensure_realsense_serial(args.head_serial, "head")
        if "left_wrist" in selected_cams:
            ensure_realsense_serial(args.left_wrist_serial, "left wrist")
        if "right_wrist" in selected_cams:
            ensure_realsense_serial(args.right_wrist_serial, "right wrist")
    else:
        print("camera_serial_check=skipped using_explicit_serials")

    teleop_active_sides = active_sides(args)
    craft_active = "right" in teleop_active_sides
    for side in teleop_active_sides:
        ensure_can_interface_ready(args.left_channel if side == "left" else args.right_channel)
    active_fingers = parse_active_fingers(args.active_fingers)
    side_fingers = parse_side_fingers(args.side_fingers)
    craft_motor_ids = parse_motor_ids(args.craft_motors) if craft_active else []
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
    craft_preprocessor = QuestHandPreprocessor(side="right")
    left_arm = None
    right_arm = None
    dataset: LeRobotDataset | None = None
    cam_streams: dict[str, RealSenseStream] = {}
    recording = False
    episode_start_time: float | None = None
    episode_started_at: float | None = None
    stop_reason = "unknown"

    try:
        if "left" in teleop_active_sides:
            left_arm = setup_side_arm(args, "left", period)
        if "right" in teleop_active_sides:
            right_arm = setup_side_arm(args, "right", period)
        arms = present_arms(left_arm, right_arm)
        arm_items = active_arm_items(left_arm, right_arm)
        active_joint_indices = validate_active_joint_indices(
            args.active_arm_joints,
            max(arm["arm_dofs"] for arm in arms),
        )
        cam_streams, cam_sizes = start_realsense_streams(args, selected_cams, head_fps, wrist_fps)
        dataset = create_combined_dataset(args, arm_items, craft_motor_ids, cam_sizes, dataset_fps, selected_cams)

        ready = np.asarray(args.ready_qpos, dtype=np.float64)
        print(f"Moving {','.join(teleop_active_sides)} arm(s) to ready pose...")
        reset_active_arms_to_ready(arms, ready, args.home_time, args.dry_run)
        print(
            "i2rt_craft_hand=ready "
            f"profile={args.profile} active_sides={','.join(teleop_active_sides)} pose_source={args.pose_source} "
            f"pos_scale={args.pos_scale} max_target_speed={args.max_target_translation_speed} "
            f"max_joint_step={args.max_arm_joint_step} left_gripper={args.left_gripper} "
            f"right_gripper={args.right_gripper} "
            f"craft_helper_source={CRAFT_HELPER_SOURCE} "
            f"arm_frequency={args.arm_frequency} craft_frequency={args.craft_frequency} "
            f"craft_active={craft_active} craft_hand_mode={args.craft_hand_mode} drive_craft={args.drive_craft} "
            f"craft_io_mode={args.craft_io_mode} craft_command_hz={args.craft_command_hz} "
            f"thumb_retarget={args.thumb_retarget} thumb_mode={args.thumb_mode} "
            f"thumb_side_gain={args.thumb_side_gain} thumb_side_limit={args.thumb_side_limit} "
            f"side_scale={args.side_scale} side_deadzone={args.side_deadzone} "
            f"filter_mode={args.filter_mode} max_step_raw={args.max_step_raw} "
            f"thumb_geometry_scale={args.thumb_geometry_scale} "
            f"thumb_geometry_deadzone={args.thumb_geometry_deadzone} "
            f"craft_motors={','.join(str(motor_id) for motor_id in craft_motor_ids)} "
            f"duration={args.duration} dry_run={args.dry_run} "
            f"active_arm_joints={active_joint_indices if active_joint_indices is not None else 'all'}"
        )
        print(f"dataset_root={dataset.root}")
        print("Quest Vuer URL: https://vuer.ai?ws=wss://localhost:8012")
        print("Quest Browser is launched with the standard local flow.")
        print(
            "Left fist starts/recalibrates. Left thumbs-up held saves. "
            "Left middle finger discards/deletes. Left pinky stops."
        )

        craft_context = (
            CraftHandOutput.from_args(args)
            if craft_active and args.craft_hand_mode == "drive" and not args.dry_run
            else nullcontext(None)
        )
        with craft_context as craft:
            initial_craft_targets = craft.last_targets.copy() if craft is not None else raw_defaults()
            craft_filter = create_craft_target_filter(args, initial_craft_targets)
            craft_defaults = raw_defaults()
            craft_action_targets = {motor_id: int(initial_craft_targets[motor_id]) for motor_id in craft_motor_ids}
            craft_state_cache = CraftStateCache(craft_motor_ids, craft_action_targets)
            craft_io: AsyncCraftIO | None = None
            last_seen_right_hand_at: float | None = None
            stale_drop_count = 0
            last_stale_warn_time = 0.0

            def sync_craft_targets_from_output() -> None:
                nonlocal craft_action_targets
                source = craft.last_targets if craft is not None else craft_defaults
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

            if craft is not None and args.craft_io_mode == "async":
                start_async_craft_io()
            elif craft is not None:
                print("craft_io=sync")

            try:
                hold_started_at = {"xButton": None}
                hold_fired = {"xButton": False}
                y_was_pressed = False
                last_print = 0.0
                loop_stats_started_at = time.monotonic()
                loop_count_since_print = 0
                craft_update_count_since_print = 0
                last_late_ms = 0.0
                next_tick = time.monotonic()
                next_craft_tick = time.monotonic()
                last_craft_signals: dict[str, float] = {"craft_hold_previous": 1.0}
                with QuestSource(
                    fps=args.fps,
                    show_left=args.show_hands,
                    show_right=args.show_hands,
                    ngrok=args.ngrok,
                ) as source:
                    launch_standard_quest_browser(args)
                    while True:
                        now_mono = time.monotonic()
                        if now_mono < next_tick:
                            time.sleep(min(next_tick - now_mono, 0.01))
                            continue
                        scheduled_tick = next_tick
                        next_tick += period
                        last_late_ms = max(0.0, (now_mono - scheduled_tick) * 1000.0)
                        loop_count_since_print += 1
                        frame = source.latest_frame()
                        teleop_raw = collect_teleoperation_raw(source, frame)
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
                            stop_async_craft_io("discard")
                            if recording:
                                print("[discard] middle-finger gesture received; dropping current episode.")
                                discard_current_episode(dataset)
                                recording = False
                                episode_start_time = None
                                episode_started_at = None
                            else:
                                delete_last_episode(dataset)
                            print("reset=ready moving_active_arms_and_craft_to_ready")
                            reset_outputs_to_ready(
                                arms=arms,
                                ready_qpos=ready,
                                home_time=args.home_time,
                                dry_run=args.dry_run,
                                craft=craft,
                            )
                            sync_craft_targets_from_output()
                            start_async_craft_io()
                            print("reset=ready done waiting_for_left_fist")
                            time.sleep(0.1)
                            next_tick = time.monotonic() + period
                            next_craft_tick = time.monotonic()
                            continue

                        if x_fired:
                            if recording:
                                stop_async_craft_io("save")
                                recording = False
                                episode_start_time = None
                                episode_started_at = None
                                saved = save_episode_while_resetting(
                                    dataset=dataset,
                                    arms=arms,
                                    ready_qpos=ready,
                                    home_time=args.home_time,
                                    dry_run=args.dry_run,
                                    craft=craft,
                                )
                                if saved:
                                    print(f"Episode {dataset.meta.total_episodes} saved.")
                                else:
                                    print("No frames recorded; nothing to save.")
                                    discard_current_episode(dataset)
                                sync_craft_targets_from_output()
                                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                                    stop_reason = "num_episodes_reached"
                                    break
                                print("Ready/default reached. Make a left fist to start the next episode.")
                                start_async_craft_io()
                                time.sleep(0.1)
                                next_tick = time.monotonic() + period
                                next_craft_tick = time.monotonic()
                            else:
                                print("thumbs_up_ignored=no_active_episode")
                            continue

                        if not recording:
                            if args.auto_start or buttons["aButton"]:
                                ok = retargeter.calibrate(
                                    frame,
                                    target_pose_or_none(left_arm),
                                    target_pose_or_none(right_arm),
                                    now=frame.timestamp,
                                )
                                if ok:
                                    recording = True
                                    episode_start_time = now_mono
                                    episode_started_at = now_mono
                                    stale_drop_count = 0
                                    last_seen_right_hand_at = None
                                    loop_stats_started_at = now_mono
                                    loop_count_since_print = 0
                                    craft_update_count_since_print = 0
                                    next_craft_tick = now_mono
                                    last_craft_signals = {"craft_start_delay": 1.0}
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
                            stop_async_craft_io("duration")
                            recording = False
                            episode_start_time = None
                            episode_started_at = None
                            saved = save_episode_while_resetting(
                                dataset=dataset,
                                arms=arms,
                                ready_qpos=ready,
                                home_time=args.home_time,
                                dry_run=args.dry_run,
                                craft=craft,
                            )
                            if saved:
                                print(f"Episode {dataset.meta.total_episodes} saved.")
                            else:
                                print("No frames recorded; nothing to save.")
                                discard_current_episode(dataset)
                            sync_craft_targets_from_output()
                            if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                                stop_reason = "num_episodes_reached"
                                break
                            if args.episode_time <= 0.0:
                                stop_reason = "duration_elapsed"
                                break
                            print("Ready/default reached. Make a left fist to start the next episode.")
                            start_async_craft_io()
                            time.sleep(0.1)
                            next_tick = time.monotonic() + period
                            next_craft_tick = time.monotonic()
                            continue

                        yam_output = retargeter.update(frame, dt=period, now=frame.timestamp)
                        update_gripper_from_pinch(
                            left_arm,
                            yam_output.left_gripper,
                            invert=args.left_gripper_invert,
                            mode=args.gripper_mode,
                            alpha=args.gripper_alpha,
                            deadband=args.gripper_deadband,
                            max_speed=args.max_gripper_speed,
                            dt=period,
                        )
                        update_gripper_from_pinch(
                            right_arm,
                            yam_output.right_gripper,
                            invert=args.right_gripper_invert,
                            mode=args.gripper_mode,
                            alpha=args.gripper_alpha,
                            deadband=args.gripper_deadband,
                            max_speed=args.max_gripper_speed,
                            dt=period,
                        )
                        if args.gripper_mode != "none":
                            for arm in arms:
                                maybe_limit_gripper_close(
                                    arm,
                                    args.gripper_force_threshold,
                                    args.gripper_force_ema_alpha,
                                    args.gripper_backoff,
                                )
                        left_pose = yam_output.left_pose if "left" in teleop_active_sides else None
                        right_pose = yam_output.right_pose if "right" in teleop_active_sides else None
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
                        craft_targets_updated = False
                        if not craft_active:
                            craft_signals = {"craft_inactive_no_right_arm": 1.0}
                        elif args.craft_hand_mode == "disabled":
                            craft_signals = {"craft_disabled": 1.0}
                        else:
                            if now_mono >= next_craft_tick:
                                while next_craft_tick <= now_mono:
                                    next_craft_tick += craft_period
                                craft_action_targets, last_craft_signals, last_seen_right_hand_at = retarget_craft(
                                    args=args,
                                    frame=frame,
                                    preprocessor=craft_preprocessor,
                                    target_filter=craft_filter,
                                    active_fingers=active_fingers,
                                    side_fingers=side_fingers,
                                    default_targets=craft_defaults,
                                    previous_targets=craft_action_targets,
                                    episode_started_at=episode_started_at,
                                    last_seen_right_hand_at=last_seen_right_hand_at,
                                    now=now_mono,
                                )
                                craft_targets_updated = True
                                craft_update_count_since_print += 1
                            craft_signals = last_craft_signals
                        if craft_io is not None:
                            if craft_targets_updated:
                                craft_io.submit(craft_action_targets, now=now_mono)
                            craft_state_cache.present.update(craft_io.present_snapshot())
                        elif craft is not None:
                            if craft_targets_updated:
                                craft.write_raw(craft_action_targets)
                            craft_state_cache.maybe_update(craft, now_mono, args.craft_state_read_hz)
                        if craft is None or args.craft_state_read_hz <= 0.0:
                            craft_state_cache.present.update(craft_action_targets)

                        stale_drop_count, last_stale_warn_time = add_lerobot_frame_if_fresh(
                            dataset=dataset,
                            args=args,
                            arm_items=arm_items,
                            craft_state_cache=craft_state_cache,
                            craft_action_targets=craft_action_targets,
                            teleop_raw=teleop_raw,
                            cam_streams=cam_streams,
                            now=now_mono,
                            stale_drop_count=stale_drop_count,
                            last_stale_warn_time=last_stale_warn_time,
                        )

                        if now_mono - last_print > 1.0 / max(args.print_hz, 0.1):
                            arm_status_parts: list[str] = []
                            if left_arm is not None:
                                ld = yam_output.diagnostics["left"]
                                arm_status_parts.append(
                                    f"left_ok={left_ok} left_reason={ld.reason} left_step={left_step} "
                                    f"left_xyz=({ld.target_xyz[0]:.3f},{ld.target_xyz[1]:.3f},{ld.target_xyz[2]:.3f}) "
                                    f"left_cmd_delta_mm={format_delta_mm(ld, args.pos_scale)} "
                                    f"left_rot_deg=({ld.raw_rotation_deg:.1f}->{ld.filtered_rotation_deg:.1f})"
                                )
                            if right_arm is not None:
                                rd = yam_output.diagnostics["right"]
                                arm_status_parts.append(
                                    f"right_ok={right_ok} right_reason={rd.reason} right_step={right_step} "
                                    f"right_xyz=({rd.target_xyz[0]:.3f},{rd.target_xyz[1]:.3f},{rd.target_xyz[2]:.3f}) "
                                    f"right_cmd_delta_mm={format_delta_mm(rd, args.pos_scale)} "
                                    f"right_rot_deg=({rd.raw_rotation_deg:.1f}->{rd.filtered_rotation_deg:.1f})"
                                )
                            print_now = time.monotonic()
                            loop_elapsed = max(print_now - loop_stats_started_at, 1e-6)
                            loop_hz = loop_count_since_print / loop_elapsed
                            craft_retarget_hz = craft_update_count_since_print / loop_elapsed
                            loop_stats_started_at = print_now
                            loop_count_since_print = 0
                            craft_update_count_since_print = 0
                            if craft_io is not None:
                                craft_io_stats = craft_io.stats(print_now)
                                craft_io_text = (
                                    f"craft_io=async loop_hz={loop_hz:.1f} late_ms={last_late_ms:.1f} "
                                    f"craft_retarget_hz={craft_retarget_hz:.1f} "
                                    f"craft_write_hz={craft_io_stats.write_hz:.1f} "
                                    f"craft_target_age_ms={format_optional_ms(craft_io_stats.last_target_age_ms)} "
                                    f"craft_skipped={craft_io_stats.skipped_targets} "
                                    f"craft_errors={craft_io_stats.write_failures}/{craft_io_stats.read_failures}"
                                )
                            elif craft is not None:
                                craft_io_text = (
                                    f"craft_io=sync loop_hz={loop_hz:.1f} late_ms={last_late_ms:.1f} "
                                    f"craft_retarget_hz={craft_retarget_hz:.1f}"
                                )
                            else:
                                craft_io_text = (
                                    f"craft_io=none loop_hz={loop_hz:.1f} late_ms={last_late_ms:.1f} "
                                    f"craft_retarget_hz={craft_retarget_hz:.1f}"
                                )
                            print(
                                "teleop "
                                f"{' '.join(arm_status_parts)} "
                                f"{craft_io_text} "
                                f"craft=({signal_summary(craft_signals)} "
                                f"targets={compact_targets(craft_action_targets, craft_motor_ids)})"
                            )
                            last_print = now_mono
            except KeyboardInterrupt:
                if stop_reason == "unknown":
                    stop_reason = "keyboard_interrupt"
                print("\ni2rt_craft_hand=stopping")
            finally:
                original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    stop_async_craft_io("shutdown")
                    if dataset is not None and recording:
                        print("Discarding unsaved episode before shutdown. Use thumbs-up before pinky to save.")
                        discard_current_episode(dataset)
                    if arms:
                        if args.hold_current_on_exit or args.dry_run:
                            for arm in arms:
                                sync_arm_state_from_robot(arm)
                                if not args.dry_run:
                                    arm["robot"].command_joint_pos(command_vector(arm))
                        else:
                            with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="home") as executor:
                                futures = [executor.submit(reset_to_home, arm, args.home_time) for arm in arms]
                                for future in futures:
                                    future.result()
                finally:
                    signal.signal(signal.SIGINT, original_handler)
    finally:
        for cam in cam_streams.values():
            cam.stop()
        if dataset is not None:
            dataset.stop_image_writer()
            fix_dataset_owner(Path(dataset.root), args.dataset_owner)
        if left_arm is not None:
            left_arm["robot"].close()
        if right_arm is not None:
            right_arm["robot"].close()
        print("i2rt_craft_hand=shutdown_complete")


if __name__ == "__main__":
    main()
