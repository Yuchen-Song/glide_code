"""Quest-controller LeRobot recorder for the i2rt/YAM dual-arm setup.

This is the user-facing controller endpoint for hardware data collection. It
combines Quest controller poses/buttons from Vuer, dual YAM arm control over
CAN, optional RealSense video recording, and LeRobot episode lifecycle
management.

By default the Quest view is input-only passthrough: RealSense cameras are
recorded into the dataset but are not rendered as headset image planes unless
`--vr-image-background` is explicitly enabled.
"""

import argparse
import json
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
from PIL import Image
import pyrealsense2 as rs

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_PATH, EPISODES_STATS_PATH, serialize_dict, write_info, write_jsonlines
from lerobot.common.datasets.video_utils import encode_video_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEVISION_DIR = REPO_ROOT / "TeleVision"
for path in (REPO_ROOT, TELEVISION_DIR):
    if path.is_dir() and str(path) not in sys.path:
        # Allow importing TeleVision scripts without turning the folder into a package.
        sys.path.insert(0, str(path))

from i2rt.utils.quest_browser import add_quest_browser_args, launch_standard_quest_browser  # noqa: E402
from teleop_dual_arm import (  # noqa: E402
    CONTROL_FREQUENCY,
    VuerControllerTeleop,
    build_command,
    clamp_eef_pose,
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

EPISODE_LABEL_UNLABELED = "unlabeled"
EPISODE_LABEL_SUCCESS = "success"
EPISODE_LABEL_FAIL = "fail"


# RealSense streams are data-recording inputs first. The headset display path
# can optionally reuse one stream, but camera capture is deliberately decoupled
# from the Quest passthrough interface.
@dataclass
class RealSenseConfig:
    serial: str
    width: int
    height: int
    fps: int
    warmup_s: float = 1.0
    allow_fallback: bool = False


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
            # Fallback asks librealsense for the device default stream profile.
            # Use only for diagnosis; explicit profiles keep recorded datasets consistent.
            self.profile = self.pipeline.start(self._build_rs_config(use_defaults=True))
        self.started = True
        self._warmup()
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

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


def rewrite_episodes_metadata(dataset: LeRobotDataset) -> None:
    episodes_payload = [dataset.meta.episodes[idx] for idx in sorted(dataset.meta.episodes)]
    write_jsonlines(episodes_payload, dataset.root / EPISODES_PATH)


def annotate_episode_label(dataset: LeRobotDataset, episode_index: int, episode_label: str, save_source: str) -> None:
    episode = dataset.meta.episodes.get(episode_index)
    if episode is None:
        print(f"[save] warning=missing_episode_metadata episode={episode_index}")
        return
    episode["episode_label"] = episode_label
    episode["save_source"] = save_source
    rewrite_episodes_metadata(dataset)


def controller_squeeze_pressed(controller_state: dict | None, threshold: float) -> bool:
    if controller_state is None:
        return False
    if controller_state.get("squeeze"):
        return True
    try:
        return float(controller_state.get("squeezeValue", 0.0)) >= threshold
    except (TypeError, ValueError):
        return False


def maybe_save_episode(
    dataset: LeRobotDataset,
    episode_label: str = EPISODE_LABEL_UNLABELED,
    save_source: str = "unknown",
) -> bool:
    if dataset.episode_buffer is None:
        return False
    if dataset.episode_buffer.get("size", 0) <= 0:
        return False
    episode_index = dataset.episode_buffer.get("episode_index")
    frame_count = dataset.episode_buffer.get("size", 0)
    start = time.monotonic()
    dataset.save_episode()
    annotate_episode_label(dataset, episode_index, episode_label, save_source)
    elapsed = time.monotonic() - start
    print(
        f"[save] episode {episode_index} finalized ({frame_count} frames) "
        f"label={episode_label} source={save_source} in {elapsed:.2f}s."
    )
    return True


def get_dataset_root(repo_id: str, dataset_root: str | None) -> Path:
    return Path(dataset_root) if dataset_root else HF_LEROBOT_HOME / repo_id


def ensure_unique_dataset_target(args: argparse.Namespace) -> Path:
    """Avoid LeRobot create failures by selecting a fresh dataset target.

    This never deletes or overwrites existing data. If the requested repo/root
    already exists, it appends a numeric suffix and mutates args so the later
    LeRobotDataset.create call uses the new target.
    """
    root = get_dataset_root(args.repo_id, args.dataset_root)
    if not root.exists():
        return root

    original_root = root
    if args.dataset_root:
        for idx in range(1, 1000):
            candidate = original_root.with_name(f"{original_root.name}-{idx:03d}")
            if not candidate.exists():
                args.dataset_root = str(candidate)
                print(f"[Preflight] Requested dataset root already exists: {original_root}")
                print(f"[Preflight] Auto-selected new dataset root: {candidate}")
                return candidate
    else:
        original_repo_id = args.repo_id
        for idx in range(1, 1000):
            candidate_repo_id = f"{original_repo_id}-{idx:03d}"
            candidate = get_dataset_root(candidate_repo_id, None)
            if not candidate.exists():
                args.repo_id = candidate_repo_id
                print(f"[Preflight] Requested dataset root already exists: {original_root}")
                print(f"[Preflight] Auto-selected new repo-id: {candidate_repo_id}")
                return candidate

    raise RuntimeError(f"Could not find a free dataset target derived from {original_root}")


def run_preflight_checks(args: argparse.Namespace, dataset_fps: int) -> bool:
    """Validate the target dataset before hardware motion starts."""
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

        print("[Preflight] Choose a new --repo-id or delete the existing dataset folder.")
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
    """Drop the in-memory episode and queued image writes without encoding it."""
    if dataset.episode_buffer is None:
        return False
    episode_index = dataset.episode_buffer.get("episode_index")

    writer = getattr(dataset, "image_writer", None)
    dropped = 0
    if writer is not None and writer.queue is not None:
        queue = writer.queue
        while True:
            try:
                queue.get_nowait()
            except Exception:
                break
            queue.task_done()
            dropped += 1
    if dropped:
        print(f"[discard] dropped {dropped} queued image writes without flushing.")

    dataset.clear_episode_buffer()
    if episode_index is not None:
        _cleanup_episode_images(dataset, episode_index)
    return True


def delete_last_episode(dataset: LeRobotDataset) -> bool:
    """Remove the most recently saved episode and rewrite LeRobot metadata."""
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

    rewrite_episodes_metadata(dataset)

    episodes_stats_payload = [
        {"episode_index": idx, "stats": serialize_dict(dataset.meta.episodes_stats[idx])}
        for idx in sorted(dataset.meta.episodes_stats)
    ]
    write_jsonlines(episodes_stats_payload, dataset.root / EPISODES_STATS_PATH)

    print(f"Deleted episode {episode_index}.")
    return True


def parallel_reset_to_home(left_arm: dict | None, right_arm: dict | None, home_time: float) -> None:
    arms = [arm for arm in (left_arm, right_arm) if arm is not None]
    if not arms:
        return
    with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="home") as executor:
        futures = [executor.submit(reset_to_home, arm, home_time) for arm in arms]
        for future in futures:
            future.result()


def save_episode_while_moving_to_ready(
    dataset: LeRobotDataset,
    left_arm: dict,
    right_arm: dict,
    ready_qpos: np.ndarray,
    home_time: float,
    episode_label: str = EPISODE_LABEL_UNLABELED,
    save_source: str = "unknown",
) -> bool:
    """Finalize the episode while the robot returns to ready pose."""
    ready_qpos = np.asarray(ready_qpos, dtype=float)
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="save_ready") as executor:
        save_future = executor.submit(maybe_save_episode, dataset, episode_label, save_source)
        left_future = executor.submit(move_to_ready_pose, left_arm, ready_qpos, home_time)
        right_future = executor.submit(move_to_ready_pose, right_arm, ready_qpos, home_time)
        left_future.result()
        right_future.result()
        return save_future.result()


def draw_vr_crosshair(img: np.ndarray, color=(0, 255, 0), arm_len: int = 60, thickness: int = 3) -> None:
    """Draw a centered crosshair + dot in-place for VR display alignment."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half_t = max(1, thickness // 2)
    x0 = max(0, cx - arm_len)
    x1 = min(w, cx + arm_len + 1)
    img[max(0, cy - half_t):min(h, cy + half_t + 1), x0:x1] = color
    y0 = max(0, cy - arm_len)
    y1 = min(h, cy + arm_len + 1)
    img[y0:y1, max(0, cx - half_t):min(w, cx + half_t + 1)] = color
    dot = max(3, thickness + 2)
    img[max(0, cy - dot):min(h, cy + dot + 1),
        max(0, cx - dot):min(w, cx + dot + 1)] = color


def get_frame_with_age(cam: RealSenseStream, now: float):
    """Return (frame, age_seconds) or (None, None) if no frame yet.

    Reads cam.latest_frame / cam.latest_timestamp under the stream's own lock
    so this is safe against the background grabber thread.
    """
    with cam.frame_lock:
        frame = cam.latest_frame
        ts = cam.latest_timestamp
    if frame is None or ts is None:
        return None, None
    return frame, now - ts


def parallel_move_to_ready(executor: ThreadPoolExecutor, left_arm: dict, right_arm: dict,
                            ready_qpos: np.ndarray, home_time: float) -> None:
    """Run both arms' ready-pose moves concurrently on the existing 2-worker pool.

    Independent CAN channels, no shared state, so wall-clock time is shorter
    than a sequential left-then-right move.
    """
    fl = executor.submit(move_to_ready_pose, left_arm, ready_qpos, home_time)
    fr = executor.submit(move_to_ready_pose, right_arm, ready_qpos, home_time)
    fl.result()
    fr.result()


def process_arm_tick(
    arm_state: dict,
    controller_state: dict | None,
    controller_mat: np.ndarray,
    init_controller: np.ndarray,
    init_pose: np.ndarray,
    gripper_mode: str,
    gripper_invert: bool,
    gripper_force_threshold: float,
    gripper_force_ema_alpha: float,
    gripper_backoff: float,
    pos_scale: float,
    lock_orientation: bool,
    eef_max_radius: float | None = None,
    eef_min_z: float | None = None,
    eef_min_x: float | None = None,
) -> dict:
    """One arm's tick: gripper update -> force check -> IK -> command.

    Runs in a worker thread. Mutates only its own arm_state dict and issues
    CAN traffic on its own channel, so two of these execute in parallel
    without synchronization.
    """
    if gripper_mode != "none" and controller_state is not None:
        update_gripper_from_controller(arm_state, controller_state, gripper_mode, gripper_invert)

    blocked, eff, pos, goal = maybe_limit_gripper_close(
        arm_state, gripper_force_threshold, gripper_force_ema_alpha, gripper_backoff
    )

    virtual_pose = compute_target_pose(
        init_pose, init_controller, controller_mat, pos_scale, lock_orientation
    )
    target_pose = clamp_eef_pose(virtual_pose, eef_max_radius, eef_min_z, eef_min_x)

    # Advance the virtual (unclamped) IK tracker; seeding the real IK with
    # this state gives immediate tracking when returning to reachable range.
    virt_ok, virt_q = arm_state["kin"].ik(
        virtual_pose, init_q=arm_state.get("virtual_target_q", arm_state["target_q"])
    )
    if virt_ok:
        arm_state["virtual_target_q"] = virt_q
        arm_state["virtual_target_pose"] = virtual_pose

    success, q = arm_state["kin"].ik(
        target_pose, init_q=arm_state.get("virtual_target_q", arm_state["target_q"])
    )
    cmd = None
    if success:
        arm_state["target_q"] = q
        arm_state["target_pose"] = target_pose
        cmd = build_command(
            q,
            arm_state["gripper_pos"],
            arm_state["arm_dofs"],
            arm_state["gripper_index"],
            arm_state["robot"].num_dofs(),
        )
        arm_state["robot"].command_joint_pos(cmd)

    return {
        "success": success,
        "cmd": cmd,
        "blocked": blocked,
        "eff": eff,
        "pos": pos,
        "goal": goal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-mode", type=str, choices=["none", "trigger", "squeeze"], default="trigger")
    parser.add_argument(
        "--label-save-squeeze-threshold",
        type=float,
        default=0.75,
        help="Side-grip squeezeValue threshold for labeled saves: left=success, right=fail.",
    )
    parser.add_argument("--gripper-force-threshold", type=float, default=0.8)
    parser.add_argument("--gripper-force-verbose", action="store_true")
    parser.add_argument("--gripper-force-print-interval", type=float, default=0.5)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=0.8)
    parser.add_argument("--gripper-backoff", type=float, default=0.02)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--lock-orientation", action="store_true")
    parser.add_argument("--frequency", type=float, default=CONTROL_FREQUENCY)
    parser.add_argument("--ngrok", action="store_true")
    add_quest_browser_args(parser)
    parser.add_argument("--home-time", type=float, default=1.0)
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
    parser.add_argument("--task", type=str, default="teleop dual-arm demo")
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--episode-time", type=float, default=0.0)
    parser.add_argument("--reset-time", type=float, default=10.0)
    parser.add_argument("--robot-type", type=str, default="yam_dual_arm")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument(
        "--video-encode-workers",
        type=int,
        default=3,
        help="Number of camera videos to encode in parallel while saving an episode.",
    )
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
        help="Which cameras to record. Omit wrists to cut image-writer load 3x.",
    )
    parser.add_argument(
        "--vr-camera",
        type=str,
        choices=["head", "left_wrist", "right_wrist", "none"],
        default="none",
        help="Which camera to render into the VR headset when --vr-image-background is enabled.",
    )
    parser.add_argument(
        "--vr-image-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render a stereo RealSense image background in Quest. Default off so "
            "Quest passthrough stays visible; RealSense recording is unaffected."
        ),
    )
    parser.add_argument(
        "--vr-crosshair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a centered crosshair on the VR camera feed (default: on). "
             "Use --no-vr-crosshair to disable.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--max-frame-age",
        type=float,
        default=0.050,
        help="Reject a (state, action, images) record if any camera frame is older than this (seconds). Default 50 ms.",
    )
    parser.add_argument(
        "--stale-warn-interval",
        type=float,
        default=1.0,
        help="Seconds between warnings when frames are being dropped due to staleness.",
    )
    parser.add_argument(
        "--eef-max-radius",
        type=float,
        default=0.695,
        help="Max EEF distance (m) from arm base origin. Default = 95%% of observed "
             "max from eef_traj_20260416_113928.npz. Set 0/negative to disable.",
    )
    parser.add_argument(
        "--eef-min-z",
        type=float,
        default=0.05,
        help="Min EEF z (m) in arm base frame. Use a large negative value to disable.",
    )
    parser.add_argument(
        "--eef-min-x",
        type=float,
        default=None,
        help="Min EEF x (m). Omit to disable.",
    )
    args = parser.parse_args()

    # Convert CLI clamp values once so the hot loop can pass simple scalars
    # into the per-arm worker without repeatedly interpreting disabled values.
    eef_max_r = args.eef_max_radius if args.eef_max_radius and args.eef_max_radius > 0 else None
    eef_min_z = args.eef_min_z if args.eef_min_z is not None else None
    eef_min_x = args.eef_min_x

    if args.list_cameras:
        for serial, name in list_realsense_devices():
            print(f"{serial}  {name}")
        return

    dataset_fps = int(round(args.frequency))
    if abs(args.frequency - dataset_fps) > 1e-3:
        print(f"Warning: --frequency {args.frequency} is not integer; dataset fps set to {dataset_fps}.")

    head_fps = args.head_fps if args.head_fps is not None else dataset_fps
    wrist_fps = args.wrist_fps if args.wrist_fps is not None else dataset_fps

    # Repeated lab runs often leave partial dataset folders. Select a fresh
    # target before preflight so the operator does not have to delete anything.
    ensure_unique_dataset_target(args)

    if not args.skip_preflight:
        ok = run_preflight_checks(args, dataset_fps)
        if args.preflight_only:
            return
        if not ok:
            return

    # Only validate serials we actually intend to open. Consider both the
    # recorded cameras and the VR-display camera (which may be an extra
    # stream not in --cameras).
    selected_serial_check = set(args.cameras)
    if args.vr_image_background and args.vr_camera != "none":
        selected_serial_check.add(args.vr_camera)
    if "head" in selected_serial_check:
        ensure_realsense_serial(args.head_serial, "head")
    if "left_wrist" in selected_serial_check:
        ensure_realsense_serial(args.left_wrist_serial, "left wrist")
    if "right_wrist" in selected_serial_check:
        ensure_realsense_serial(args.right_wrist_serial, "right wrist")

    teleop = None
    left_arm = None
    right_arm = None
    head_cam = None
    left_wrist_cam = None
    right_wrist_cam = None
    vr_cam_stream: RealSenseStream | None = None
    vr_cam_is_extra = False
    dataset = None
    executor: ThreadPoolExecutor | None = None

    selected_cams = set(args.cameras)
    cam_streams: dict[str, RealSenseStream] = {}
    cam_sizes: dict[str, tuple[int, int]] = {}  # name -> (width, height)

    try:
        # Bring up CAN first. If a motor bus is down, fail before opening
        # cameras or starting the Quest/Vuer process.
        ensure_can_interface_ready(args.left_channel)
        ensure_can_interface_ready(args.right_channel)

        cam_configs = {
            "head": (args.head_serial, args.head_width, args.head_height, head_fps),
            "left_wrist": (args.left_wrist_serial, args.wrist_width, args.wrist_height, wrist_fps),
            "right_wrist": (args.right_wrist_serial, args.wrist_width, args.wrist_height, wrist_fps),
        }
        # Start only the RealSense streams requested for dataset recording.
        # This keeps the fast/debug path from paying wrist-camera capture cost.
        for name in ("head", "left_wrist", "right_wrist"):
            if name not in selected_cams:
                continue
            serial, w, h, fps = cam_configs[name]
            stream = RealSenseStream(
                RealSenseConfig(serial, w, h, fps, allow_fallback=args.allow_camera_fallback)
            )
            stream.start()
            cam_streams[name] = stream
            vs = stream.profile.get_stream(rs.stream.color).as_video_stream_profile()
            cam_sizes[name] = (int(vs.width()), int(vs.height()))

        # Back-compat aliases for the rest of the code that still expects these
        # named handles. Missing cams are left as None and the record path skips
        # them when pairing frames.
        head_cam = cam_streams.get("head")
        left_wrist_cam = cam_streams.get("left_wrist")
        right_wrist_cam = cam_streams.get("right_wrist")

        # Optional VR image background gets a camera feed written into the Vuer
        # shared-memory image buffer. The default Quest view is input-only
        # passthrough, so RealSense recording stays separate from what is
        # rendered in the headset.
        vr_cam_stream: RealSenseStream | None = None
        vr_cam_is_extra = False
        if args.vr_image_background and args.vr_camera != "none":
            if args.vr_camera in cam_streams:
                vr_cam_stream = cam_streams[args.vr_camera]
            else:
                serial, w, h, fps = cam_configs[args.vr_camera]
                vr_cam_stream = RealSenseStream(
                    RealSenseConfig(serial, w, h, fps, allow_fallback=args.allow_camera_fallback)
                )
                vr_cam_stream.start()
                vr_cam_is_extra = True

        # IK timing follows the control loop by default, keeping command
        # updates and recorded actions on the same nominal clock.
        ik_frame = args.site or args.ik_frame
        ik_dt = args.ik_dt if args.ik_dt is not None else 1.0 / args.frequency
        left_arm = setup_arm(
            args.left_channel, args.left_gripper, ik_frame, args.left_gripper_invert,
            ik_dt, args.ik_alpha, args.ik_pos_cost, args.ik_ori_cost,
            args.ik_posture_cost, args.ik_damping_cost, args.ik_lm_damping,
            args.ik_gain, args.ik_solver, args.ik_solve_damping,
        )
        right_arm = setup_arm(
            args.right_channel, args.right_gripper, ik_frame, args.right_gripper_invert,
            ik_dt, args.ik_alpha, args.ik_pos_cost, args.ik_ori_cost,
            args.ik_posture_cost, args.ik_damping_cost, args.ik_lm_damping,
            args.ik_gain, args.ik_solver, args.ik_solve_damping,
        )
        teleop = VuerControllerTeleop(image_background=args.vr_image_background, ngrok=args.ngrok)
        launch_standard_quest_browser(args, ngrok=args.ngrok)
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arm")

        # Build the LeRobot schema from the actual arm/gripper DOF layout and
        # selected camera set. This keeps head-only debug datasets valid.
        left_joint_names = build_joint_names("left", left_arm["robot"].num_dofs(), left_arm["gripper_index"])
        right_joint_names = build_joint_names("right", right_arm["robot"].num_dofs(), right_arm["gripper_index"])
        state_names = left_joint_names + right_joint_names
        action_names = list(state_names)

        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
            "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        }
        for cam_name in ("head", "left_wrist", "right_wrist"):
            if cam_name not in selected_cams:
                continue
            w, h = cam_sizes[cam_name]
            features[f"observation.images.{cam_name}"] = build_image_feature(h, w)

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=dataset_fps, features=features,
            robot_type=args.robot_type, root=args.dataset_root, use_videos=True,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )

        # Episode saves can spend most of their time encoding three videos.
        # Parallelize the video-key work while the arms are already returning
        # to ready so X/save feels closer to real time.
        def _encode_episode_videos_with_codec(episode_index: int) -> dict:
            def encode_one(key: str) -> tuple[str, str]:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                if not video_path.is_file():
                    img_dir = dataset._get_image_file_path(
                        episode_index=episode_index, image_key=key, frame_index=0
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
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video_encode") as encode_executor:
                return dict(encode_executor.map(encode_one, video_keys))

        dataset.encode_episode_videos = _encode_episode_videos_with_codec

        print("Moving both arms to ready pose...")
        parallel_move_to_ready(executor, left_arm, right_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
        sync_arm_state_from_robot(left_arm)
        sync_arm_state_from_robot(right_arm)
        print("Ready pose reached. Press A on right controller to start teleop/recording.")
        print("Left controller: side squeeze saves success, X saves unlabeled, Y discards/deletes.")
        print("Right controller: side squeeze saves fail, B stops the run without saving current episode.")
        if args.gripper_mode == "squeeze":
            print("Warning: --gripper-mode squeeze also uses side squeeze for grippers and labeled saves.")
        print("Quest Vuer URL: https://vuer.ai?ws=wss://localhost:8012")
        print("Quest Browser is launched with the standard local flow.")
        if args.vr_image_background:
            if args.vr_camera == "none":
                print("Quest display mode: image background enabled, but --vr-camera none selected.")
            else:
                print(f"Quest display mode: rendering {args.vr_camera} RealSense image background.")
        else:
            print("Quest display mode: clean passthrough; Vuer image background disabled.")
        print(f"Max frame age: {args.max_frame_age * 1000:.1f} ms (stale pairs dropped from dataset).")
    except Exception:
        if executor is not None:
            executor.shutdown(wait=False)
        if dataset is not None:
            dataset.stop_image_writer()
        for cam in (head_cam, left_wrist_cam, right_wrist_cam):
            if cam is not None:
                cam.stop()
        if vr_cam_is_extra and vr_cam_stream is not None:
            vr_cam_stream.stop()
        if teleop is not None:
            teleop.cleanup()
        for arm in (left_arm, right_arm):
            if arm is not None:
                arm["robot"].close()
        raise

    left_init_controller = None
    right_init_controller = None
    left_init_pose = None
    right_init_pose = None

    print("Waiting for both controllers to be valid...")
    last_warn_time = 0.0
    last_a_pressed = False
    last_b_pressed = False
    last_x_pressed = False
    last_y_pressed = False
    last_left_squeeze_pressed = False
    last_right_squeeze_pressed = False
    last_left_gripper_warn_time = 0.0
    last_right_gripper_warn_time = 0.0
    last_left_gripper_print_time = 0.0
    last_right_gripper_print_time = 0.0
    last_stale_warn_time = 0.0
    stale_drop_count = 0

    teleop_enabled = False
    awaiting_reference = False
    recording = False
    episode_idx = dataset.meta.total_episodes if dataset is not None else 0
    episode_start_time = None

    left_cmd = left_arm["robot"].get_joint_pos()
    right_cmd = right_arm["robot"].get_joint_pos()

    # The main loop is phase-locked instead of "sleep after work" so occasional
    # camera/dataset stalls do not permanently lower the control rate.
    dt = 1.0 / args.frequency
    next_tick = time.monotonic() + dt

    def _sleep_until_next_tick() -> None:
        """Period-locked sleep: preserves the phase so loop rate == args.frequency on average."""
        nonlocal next_tick
        now_ = time.monotonic()
        sleep_s = next_tick - now_
        if sleep_s > 0:
            time.sleep(sleep_s)
        elif sleep_s < -dt:
            # Fell badly behind (e.g. blocked on save). Resync phase.
            next_tick = time.monotonic()
        next_tick += dt

    last_vr_frame_ts: float | None = None

    def push_vr_frame() -> None:
        """Copy the latest camera frame into Vuer when image background is enabled.

        Default controller runs keep the Quest view input-only, so passthrough
        stays visible and this path returns immediately.
        """
        nonlocal last_vr_frame_ts
        if not args.vr_image_background:
            return
        if vr_cam_stream is None or teleop is None:
            return
        with vr_cam_stream.frame_lock:
            frame = vr_cam_stream.latest_frame
            ts = vr_cam_stream.latest_timestamp
        if frame is None or ts is None or ts == last_vr_frame_ts:
            return
        last_vr_frame_ts = ts
        eye_h, eye_w = teleop.resolution
        resized = np.array(
            Image.fromarray(frame).resize((eye_w, eye_h), Image.BILINEAR),
            copy=True,
        )
        if args.vr_crosshair:
            draw_vr_crosshair(resized)
        teleop.img_array[:, :eye_w] = resized
        teleop.img_array[:, eye_w:] = resized

    try:
        while True:
            push_vr_frame()

            right_state = teleop.tv.right_controller_state if teleop and teleop.tv else None
            left_state = teleop.tv.left_controller_state if teleop and teleop.tv else None

            # Vuer exposes each controller's face buttons as aButton/bButton.
            # User-facing docs keep the physical labels: right A/B, left X/Y.
            b_pressed = bool(right_state.get("bButton")) if right_state else False
            if b_pressed and not last_b_pressed:
                raise KeyboardInterrupt
            last_b_pressed = b_pressed

            a_pressed = bool(right_state.get("aButton")) if right_state else False
            x_pressed = bool(left_state.get("aButton")) if left_state else False
            y_pressed = bool(left_state.get("bButton")) if left_state else False
            left_squeeze_pressed = controller_squeeze_pressed(left_state, args.label_save_squeeze_threshold)
            right_squeeze_pressed = controller_squeeze_pressed(right_state, args.label_save_squeeze_threshold)

            a_rising = a_pressed and not last_a_pressed
            x_rising = x_pressed and not last_x_pressed
            y_rising = y_pressed and not last_y_pressed
            left_squeeze_rising = left_squeeze_pressed and not last_left_squeeze_pressed
            right_squeeze_rising = right_squeeze_pressed and not last_right_squeeze_pressed
            last_a_pressed = a_pressed
            last_x_pressed = x_pressed
            last_y_pressed = y_pressed
            last_left_squeeze_pressed = left_squeeze_pressed
            last_right_squeeze_pressed = right_squeeze_pressed

            if y_rising:
                # Y is the destructive/reset path: discard the current buffer
                # while recording, or delete the last saved episode when idle.
                if recording or awaiting_reference:
                    teleop_enabled = False
                    awaiting_reference = False
                    recording = False
                    episode_start_time = None
                    discard_current_episode(dataset)
                    print("Current episode discarded. Resetting to ready pose.")
                else:
                    if delete_last_episode(dataset):
                        episode_idx = dataset.meta.total_episodes
                    print("Resetting to ready pose.")
                parallel_move_to_ready(
                    executor, left_arm, right_arm,
                    np.array(args.ready_qpos, dtype=float), args.home_time,
                )
                sync_arm_state_from_robot(left_arm)
                sync_arm_state_from_robot(right_arm)
                left_cmd = left_arm["robot"].get_joint_pos()
                right_cmd = right_arm["robot"].get_joint_pos()
                left_init_controller = None
                right_init_controller = None
                left_init_pose = None
                right_init_pose = None
                print("Ready pose reached. Press A on right controller to capture a fresh reference.")
                time.sleep(0.1)
                next_tick = time.monotonic() + dt
                continue

            save_request = None
            if recording:
                if left_squeeze_rising:
                    save_request = (EPISODE_LABEL_SUCCESS, "left_squeeze")
                elif right_squeeze_rising:
                    save_request = (EPISODE_LABEL_FAIL, "right_squeeze")
                elif x_rising:
                    save_request = (EPISODE_LABEL_UNLABELED, "left_x")
            elif left_squeeze_rising or right_squeeze_rising:
                print("Labeled save ignored because no episode is recording.")

            if save_request is not None:
                episode_label, save_source = save_request
                # Save paths return to ready and force a fresh A reference so
                # the next episode starts from a known ready pose.
                teleop_enabled = False
                awaiting_reference = False
                recording = False
                episode_start_time = None
                print(f"Save requested source={save_source} label={episode_label}.")
                saved = save_episode_while_moving_to_ready(
                    dataset,
                    left_arm,
                    right_arm,
                    np.array(args.ready_qpos, dtype=float),
                    args.home_time,
                    episode_label=episode_label,
                    save_source=save_source,
                )
                if saved:
                    episode_idx = dataset.meta.total_episodes
                    print(f"Episode {episode_idx} saved label={episode_label} source={save_source}.")
                else:
                    print("No frames recorded; nothing to save.")
                    discard_current_episode(dataset)
                sync_arm_state_from_robot(left_arm)
                sync_arm_state_from_robot(right_arm)
                left_cmd = left_arm["robot"].get_joint_pos()
                right_cmd = right_arm["robot"].get_joint_pos()
                left_init_controller = None
                right_init_controller = None
                left_init_pose = None
                right_init_pose = None
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    break
                print("Ready pose reached. Press A on right controller to start teleop/recording.")
                time.sleep(0.1)
                next_tick = time.monotonic() + dt
                continue

            if a_rising and not recording and not awaiting_reference:
                if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                    print("Reached requested number of episodes; ignoring A.")
                else:
                    # A starts a new relative-control episode from the robot's
                    # actual current pose, not from stale target state.
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = left_arm["robot"].get_joint_pos()
                    right_cmd = right_arm["robot"].get_joint_pos()
                    awaiting_reference = True
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    print("A pressed. Robot state synced; waiting to capture controller reference.")

            left_mat = None
            right_mat = None
            if awaiting_reference or teleop_enabled:
                left_mat = teleop.get_controller_matrix("left")
                right_mat = teleop.get_controller_matrix("right")
                if left_mat is None or right_mat is None:
                    time.sleep(0.01)
                    next_tick = time.monotonic() + dt
                    continue

                left_mat = vuer_to_robot_matrix(left_mat)
                right_mat = vuer_to_robot_matrix(right_mat)

                if awaiting_reference:
                    # Store both controller and robot poses at the same moment.
                    # Teleop then tracks controller deltas relative to this pair.
                    left_init_controller = left_mat.copy()
                    right_init_controller = right_mat.copy()
                    left_init_pose = left_arm["target_pose"].copy()
                    right_init_pose = right_arm["target_pose"].copy()
                    awaiting_reference = False
                    teleop_enabled = True
                    recording = True
                    episode_start_time = time.monotonic()
                    stale_drop_count = 0
                    next_ep = dataset.meta.total_episodes + 1
                    if args.num_episodes > 0:
                        print(f"Recording episode {next_ep}/{args.num_episodes}")
                    else:
                        print(f"Recording episode {next_ep}")
                    time.sleep(0.2)
                    next_tick = time.monotonic() + dt
                    continue

            if not teleop_enabled:
                time.sleep(0.01)
                next_tick = time.monotonic() + dt
                continue

            # Snapshot controller button/trigger states so worker threads don't
            # race the Vuer subprocess mutating them mid-tick.
            left_state_snap = dict(left_state) if left_state else None
            right_state_snap = dict(right_state) if right_state else None

            # Submit left/right arm work together. Each worker owns one arm
            # state dict and CAN channel, so this is the live-control latency
            # improvement that replaced the old separate low-latency file.
            fut_left = executor.submit(
                process_arm_tick,
                left_arm, left_state_snap, left_mat,
                left_init_controller, left_init_pose,
                args.gripper_mode, args.left_gripper_invert,
                args.gripper_force_threshold, args.gripper_force_ema_alpha, args.gripper_backoff,
                args.pos_scale, args.lock_orientation,
                eef_max_r, eef_min_z, eef_min_x,
            )
            fut_right = executor.submit(
                process_arm_tick,
                right_arm, right_state_snap, right_mat,
                right_init_controller, right_init_pose,
                args.gripper_mode, args.right_gripper_invert,
                args.gripper_force_threshold, args.gripper_force_ema_alpha, args.gripper_backoff,
                args.pos_scale, args.lock_orientation,
                eef_max_r, eef_min_z, eef_min_x,
            )
            left_result = fut_left.result()
            right_result = fut_right.result()

            if left_result["success"]:
                left_cmd = left_result["cmd"]
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Left arm IK failed; holding last command.")
                    last_warn_time = now

            if right_result["success"]:
                right_cmd = right_result["cmd"]
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Right arm IK failed; holding last command.")
                    last_warn_time = now

            now = time.monotonic()

            if args.gripper_force_verbose:
                if left_result["eff"] is not None and now - last_left_gripper_print_time > args.gripper_force_print_interval:
                    pos_s = f"{left_result['pos']:.4f}" if left_result["pos"] is not None else "n/a"
                    goal_s = f"{left_result['goal']:.4f}" if left_result["goal"] is not None else "n/a"
                    print(f"Left gripper eff={left_result['eff']:.3f} pos={pos_s} target={goal_s}")
                    last_left_gripper_print_time = now
                if right_result["eff"] is not None and now - last_right_gripper_print_time > args.gripper_force_print_interval:
                    pos_s = f"{right_result['pos']:.4f}" if right_result["pos"] is not None else "n/a"
                    goal_s = f"{right_result['goal']:.4f}" if right_result["goal"] is not None else "n/a"
                    print(f"Right gripper eff={right_result['eff']:.3f} pos={pos_s} target={goal_s}")
                    last_right_gripper_print_time = now
            if left_result["blocked"] and now - last_left_gripper_warn_time > 1.0:
                print(f"Left gripper force threshold hit ({left_result['eff']:.2f}); holding position.")
                last_left_gripper_warn_time = now
            if right_result["blocked"] and now - last_right_gripper_warn_time > 1.0:
                print(f"Right gripper force threshold hit ({right_result['eff']:.2f}); holding position.")
                last_right_gripper_warn_time = now

            if recording:
                elapsed = now - episode_start_time if episode_start_time is not None else 0.0
                if args.episode_time > 0 and elapsed >= args.episode_time:
                    # Timed runs use the same save/reset path as X.
                    teleop_enabled = False
                    awaiting_reference = False
                    recording = False
                    episode_start_time = None
                    saved = save_episode_while_moving_to_ready(
                        dataset,
                        left_arm,
                        right_arm,
                        np.array(args.ready_qpos, dtype=float),
                        args.home_time,
                        episode_label=EPISODE_LABEL_UNLABELED,
                        save_source="timer",
                    )
                    if saved:
                        episode_idx = dataset.meta.total_episodes
                        print(f"Episode {episode_idx} saved label={EPISODE_LABEL_UNLABELED} source=timer.")
                    else:
                        print("No frames recorded; nothing to save.")
                        discard_current_episode(dataset)
                    sync_arm_state_from_robot(left_arm)
                    sync_arm_state_from_robot(right_arm)
                    left_cmd = left_arm["robot"].get_joint_pos()
                    right_cmd = right_arm["robot"].get_joint_pos()
                    left_init_controller = None
                    right_init_controller = None
                    left_init_pose = None
                    right_init_pose = None
                    if args.num_episodes > 0 and dataset.meta.total_episodes >= args.num_episodes:
                        break
                    print("Ready pose reached. Press A on right controller to start teleop/recording.")
                    time.sleep(0.1)
                    next_tick = time.monotonic() + dt
                    continue

                # Pair state/action with camera frames under a max-age budget.
                # If any frame is stale, we drop this tick from the dataset but
                # the arms keep moving; teleop never waits on cameras.
                imgs: dict[str, np.ndarray] = {}
                ages: dict[str, float] = {}
                for cam_name, stream in cam_streams.items():
                    img, age = get_frame_with_age(stream, now)
                    if img is None:
                        continue
                    imgs[cam_name] = img
                    ages[cam_name] = age
                have_all = len(imgs) == len(cam_streams)
                max_age = args.max_frame_age
                fresh_enough = have_all and all(a <= max_age for a in ages.values())

                if fresh_enough:
                    left_joint_state = left_arm["robot"].get_joint_pos()
                    right_joint_state = right_arm["robot"].get_joint_pos()
                    state_vec = np.concatenate([left_joint_state, right_joint_state]).astype(np.float32)
                    action_vec = np.concatenate([left_cmd, right_cmd]).astype(np.float32)
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
                        per_cam = " ".join(f"{k}={v*1000:.1f}" for k, v in ages.items())
                        print(
                            f"[stale] dropped {stale_drop_count} frames; "
                            f"worst age {worst_ms:.1f} ms > {max_age * 1000.0:.1f} ms ({per_cam})"
                        )
                        last_stale_warn_time = now

            _sleep_until_next_tick()

    except KeyboardInterrupt:
        print("\nCtrl+C or B received. Stopping recording...")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if dataset is not None and recording:
                # B/Ctrl+C means stop, not save. The operator must press X first
                # if the current episode should be kept.
                print("Discarding unsaved episode before shutdown. Press left X before B to save.")
                discard_current_episode(dataset)
        finally:
            if left_arm is not None:
                reset_to_home(left_arm, args.home_time)
            if right_arm is not None:
                reset_to_home(right_arm, args.home_time)
            signal.signal(signal.SIGINT, original_handler)
            if executor is not None:
                executor.shutdown(wait=False)
            for arm in (left_arm, right_arm):
                if arm is not None:
                    arm["robot"].close()
            if teleop is not None:
                teleop.cleanup()
            if vr_cam_is_extra and vr_cam_stream is not None:
                vr_cam_stream.stop()
            for cam in (head_cam, left_wrist_cam, right_wrist_cam):
                if cam is not None:
                    cam.stop()
            if dataset is not None:
                dataset.stop_image_writer()
            print("Teleop record shutdown complete.")


if __name__ == "__main__":
    main()
