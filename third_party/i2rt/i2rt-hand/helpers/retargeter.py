from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .open_television_adapter import OpenTeleVisionFrameAdapter, OpenTeleVisionProcessedFrame
from .types import QuestFrame, SIDE_NAMES


WEBXR25_TO_OPENPOSE21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]

GRD_YUP_TO_GRD_ZUP = np.array(
    [
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def fast_mat_inv(mat: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = mat[:3, :3].T
    out[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return out


def vuer_to_robot_matrix(mat: np.ndarray) -> np.ndarray:
    return GRD_YUP_TO_GRD_ZUP @ mat @ fast_mat_inv(GRD_YUP_TO_GRD_ZUP)


def matrix_is_valid(mat: np.ndarray | None) -> bool:
    if mat is None:
        return False
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        return False
    return abs(float(np.linalg.det(arr[:3, :3]))) > 1e-8


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    delta = Rotation.from_matrix(a) * Rotation.from_matrix(b).inv()
    return float(np.linalg.norm(delta.as_rotvec()))


def _normalize(vec: np.ndarray, min_norm: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < min_norm:
        return None
    return np.asarray(vec, dtype=np.float64) / norm


def quest_landmarks_to_openpose21(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    if pts.shape[0] >= 25:
        return pts[WEBXR25_TO_OPENPOSE21, :3]
    if pts.shape[0] >= 21:
        return pts[:21, :3]
    return None


def palm_pose_from_landmarks(landmarks: np.ndarray | None) -> np.ndarray | None:
    """Build a wrist-like palm pose from wrist and MCP joints only."""
    keypoints = quest_landmarks_to_openpose21(landmarks)
    if keypoints is None:
        return None

    wrist = keypoints[0]
    index_mcp = keypoints[5]
    middle_mcp = keypoints[9]
    ring_mcp = keypoints[13]
    pinky_mcp = keypoints[17]
    mcp_center = np.mean([index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)

    x_axis = _normalize(index_mcp - pinky_mcp)
    y_seed = _normalize(middle_mcp - wrist)
    if x_axis is None or y_seed is None:
        return None
    z_axis = _normalize(np.cross(x_axis, y_seed))
    if z_axis is None:
        return None
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = 0.35 * wrist + 0.65 * mcp_center
    return pose


def landmark_pinch_value(
    landmarks: np.ndarray | None,
    open_distance: float = 0.095,
    closed_distance: float = 0.050,
) -> float | None:
    keypoints = quest_landmarks_to_openpose21(landmarks)
    if keypoints is None:
        return None
    distance = float(np.linalg.norm(keypoints[4] - keypoints[8]))
    denom = max(1e-6, open_distance - closed_distance)
    return float(np.clip((open_distance - distance) / denom, 0.0, 1.0))


def state_pinch_value(state: dict[str, float | bool] | None) -> float | None:
    if state is None:
        return None
    if "pinchValue" in state:
        return float(state["pinchValue"])
    if "pinch" in state:
        return float(bool(state["pinch"]))
    return None


def _limit_vector_step(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if max_step <= 0.0 or norm <= max_step or norm < 1e-12:
        return target
    return current + delta * (max_step / norm)


def _rotation_step(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    prev_r = Rotation.from_matrix(prev)
    target_r = Rotation.from_matrix(target)
    delta = target_r * prev_r.inv()
    rotvec = delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if max_step > 0.0 and angle > max_step and angle > 1e-12:
        rotvec = rotvec * (max_step / angle)
    return (Rotation.from_rotvec(rotvec) * prev_r).as_matrix()


def _slerp_alpha(prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    prev_r = Rotation.from_matrix(prev)
    target_r = Rotation.from_matrix(target)
    delta = target_r * prev_r.inv()
    rotvec = delta.as_rotvec() * float(np.clip(alpha, 0.0, 1.0))
    return (Rotation.from_rotvec(rotvec) * prev_r).as_matrix()


def _rotation_with_gain(rot: np.ndarray, gain: float) -> np.ndarray:
    rotvec = Rotation.from_matrix(rot).as_rotvec() * float(gain)
    return Rotation.from_rotvec(rotvec).as_matrix()


@dataclass
class RetargetConfig:
    active_sides: tuple[str, ...] = SIDE_NAMES
    pose_source: Literal["wrist", "palm"] = "wrist"
    input_frame: Literal["vuer", "robot"] = "vuer"
    vuer_preprocessor: Literal["open_television", "legacy"] = "open_television"
    pos_scale: float = 0.05
    orientation_scale: float = 0.25
    lock_orientation: bool = True
    translation_alpha: float = 0.18
    rotation_alpha: float = 0.12
    max_target_translation_speed: float = 0.05
    max_target_angular_speed: float = 0.25
    max_input_jump: float = 0.12
    max_input_rotation_jump: float = 0.70
    stale_timeout: float = 0.15
    require_fresh_head: bool = False
    recovery_frames: int = 3
    eef_max_radius: float | None = 0.695
    eef_min_z: float | None = 0.05
    eef_min_x: float | None = None
    pinch_open_distance: float = 0.095
    pinch_closed_distance: float = 0.050
    gripper_pinch_source: Literal["auto", "landmarks", "state"] = "auto"
    left_axis_map: np.ndarray | None = None
    right_axis_map: np.ndarray | None = None

    def axis_map(self, side: str) -> np.ndarray:
        mat = self.left_axis_map if side == "left" else self.right_axis_map
        return np.eye(3, dtype=np.float64) if mat is None else np.asarray(mat, dtype=np.float64)


@dataclass
class ArmRetargetState:
    side: str
    calibrated: bool = False
    init_head_mat: np.ndarray | None = None
    init_hand_mat: np.ndarray | None = None
    init_head_relative_pos: np.ndarray | None = None
    init_robot_eef_pose: np.ndarray | None = None
    filtered_target_pose: np.ndarray | None = None
    last_good_target_pose: np.ndarray | None = None
    last_input_pose: np.ndarray | None = None
    last_valid_time: float = 0.0
    recovery_count: int = 0


@dataclass
class RetargetDiagnostics:
    side: str
    input_valid: bool
    held_last_target: bool
    reason: str = ""
    raw_hand_delta: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    filtered_translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    raw_rotation_deg: float = 0.0
    filtered_rotation_deg: float = 0.0
    target_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    pinch: float | None = None
    ik_success: bool | None = None
    ik_error: float | None = None
    max_joint_delta: float | None = None


@dataclass
class RetargetOutput:
    left_pose: np.ndarray | None
    right_pose: np.ndarray | None
    left_gripper: float | None
    right_gripper: float | None
    diagnostics: dict[str, RetargetDiagnostics]


class QuestToYamRetargeter:
    """Open-TeleVision-backed retargeter for Quest hand input and YAM EEF targets."""

    def __init__(self, config: RetargetConfig | None = None) -> None:
        self.config = config or RetargetConfig()
        self.states = {side: ArmRetargetState(side=side) for side in SIDE_NAMES}
        self._open_tv = OpenTeleVisionFrameAdapter() if self._use_open_tv_wrist_preprocessor else None
        self._cached_frame_id: int | None = None
        self._cached_open_tv_frame: OpenTeleVisionProcessedFrame | None = None

    @property
    def _use_open_tv_wrist_preprocessor(self) -> bool:
        return (
            self.config.input_frame == "vuer"
            and self.config.pose_source == "wrist"
            and self.config.vuer_preprocessor == "open_television"
        )

    def _open_tv_frame(self, frame: QuestFrame) -> OpenTeleVisionProcessedFrame:
        if self._open_tv is None:
            raise RuntimeError("Open-TeleVision preprocessor is not enabled")
        frame_id = id(frame)
        if self._cached_frame_id != frame_id or self._cached_open_tv_frame is None:
            self._cached_open_tv_frame = self._open_tv.process(frame)
            self._cached_frame_id = frame_id
        return self._cached_open_tv_frame

    def _maybe_convert(self, mat: np.ndarray) -> np.ndarray:
        return vuer_to_robot_matrix(mat) if self.config.input_frame == "vuer" else np.asarray(mat, dtype=np.float64)

    def _source_pose(self, frame: QuestFrame, side: str) -> np.ndarray | None:
        if self._use_open_tv_wrist_preprocessor:
            return self._open_tv_frame(frame).wrist_mat(side)
        if self.config.pose_source == "palm":
            pose = palm_pose_from_landmarks(frame.landmarks(side))
            if pose is not None:
                return self._maybe_convert(pose)
        hand = frame.hand_mat(side)
        return self._maybe_convert(hand) if matrix_is_valid(hand) else None

    def _head_pose(self, frame: QuestFrame) -> np.ndarray | None:
        if self._use_open_tv_wrist_preprocessor:
            return self._open_tv_frame(frame).head_mat
        if not frame.head_valid or not matrix_is_valid(frame.head_mat):
            return None
        return self._maybe_convert(frame.head_mat)

    def _is_side_fresh(self, frame: QuestFrame, side: str, now: float) -> bool:
        ts = frame.hand_last_timestamp(side)
        return ts > 0.0 and now - ts <= self.config.stale_timeout

    def _is_head_fresh(self, frame: QuestFrame, now: float) -> bool:
        if not self.config.require_fresh_head:
            return True
        return frame.head_last_timestamp > 0.0 and now - frame.head_last_timestamp <= self.config.stale_timeout

    @staticmethod
    def _head_relative_position(head: np.ndarray, hand: np.ndarray) -> np.ndarray:
        return head[:3, :3].T @ (hand[:3, 3] - head[:3, 3])

    def _operator_relative_position(self, head: np.ndarray, hand: np.ndarray) -> np.ndarray:
        if self._use_open_tv_wrist_preprocessor:
            return hand[:3, 3].copy()
        return self._head_relative_position(head, hand)

    @staticmethod
    def clamp_eef_pose(
        pose: np.ndarray,
        max_radius: float | None,
        min_z: float | None,
        min_x: float | None,
    ) -> np.ndarray:
        out = pose.copy()
        x, y, z = out[:3, 3]
        if min_z is not None and z < min_z:
            z = min_z
        if min_x is not None and x < min_x:
            x = min_x
        if max_radius is not None:
            radius = float(np.linalg.norm([x, y, z]))
            if radius > max_radius and radius > 1e-12:
                x, y, z = np.array([x, y, z], dtype=np.float64) * (max_radius / radius)
        out[:3, 3] = [x, y, z]
        return out

    def can_calibrate(self, frame: QuestFrame, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self._head_pose(frame) is None or not self._is_head_fresh(frame, now):
            return False
        for side in self.config.active_sides:
            if not frame.hand_valid(side) or not self._is_side_fresh(frame, side, now):
                return False
            if self._source_pose(frame, side) is None:
                return False
        return True

    def calibrate(
        self,
        frame: QuestFrame,
        left_eef_pose: np.ndarray | None,
        right_eef_pose: np.ndarray | None,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        head = self._head_pose(frame)
        if head is None or not self._is_head_fresh(frame, now):
            return False

        eef_by_side = {"left": left_eef_pose, "right": right_eef_pose}
        for side in self.config.active_sides:
            pose = self._source_pose(frame, side)
            eef = eef_by_side[side]
            if pose is None or eef is None or not matrix_is_valid(eef) or not self._is_side_fresh(frame, side, now):
                return False

        for side in self.config.active_sides:
            pose = self._source_pose(frame, side)
            assert pose is not None
            eef = np.asarray(eef_by_side[side], dtype=np.float64)
            state = self.states[side]
            state.calibrated = True
            state.init_head_mat = head.copy()
            state.init_hand_mat = pose.copy()
            state.init_head_relative_pos = self._operator_relative_position(head, pose)
            state.init_robot_eef_pose = eef.copy()
            state.filtered_target_pose = eef.copy()
            state.last_good_target_pose = eef.copy()
            state.last_input_pose = pose.copy()
            state.last_valid_time = now
            state.recovery_count = self.config.recovery_frames
        return True

    def _pinch(self, frame: QuestFrame, side: str) -> float | None:
        state_value = state_pinch_value(frame.hand_state(side))
        landmark_value = landmark_pinch_value(
            frame.landmarks(side),
            open_distance=self.config.pinch_open_distance,
            closed_distance=self.config.pinch_closed_distance,
        )
        if self.config.gripper_pinch_source == "state":
            return state_value
        if self.config.gripper_pinch_source == "landmarks":
            return landmark_value
        return landmark_value if landmark_value is not None else state_value

    def _held_diag(self, side: str, state: ArmRetargetState, reason: str, pinch: float | None) -> tuple[np.ndarray | None, RetargetDiagnostics]:
        pose = state.last_good_target_pose.copy() if state.last_good_target_pose is not None else None
        xyz = pose[:3, 3].copy() if pose is not None else np.zeros(3, dtype=np.float64)
        return pose, RetargetDiagnostics(
            side=side,
            input_valid=False,
            held_last_target=True,
            reason=reason,
            filtered_translation=xyz,
            target_xyz=xyz,
            pinch=pinch,
        )

    def _update_side(self, frame: QuestFrame, side: str, dt: float, now: float) -> tuple[np.ndarray | None, RetargetDiagnostics]:
        state = self.states[side]
        pinch = self._pinch(frame, side)
        if side not in self.config.active_sides:
            return self._held_diag(side, state, "inactive", pinch)
        if not state.calibrated:
            return self._held_diag(side, state, "not_calibrated", pinch)

        head = self._head_pose(frame)
        hand = self._source_pose(frame, side)
        if head is None or hand is None:
            return self._held_diag(side, state, "missing_pose", pinch)
        if not frame.hand_valid(side) or not self._is_side_fresh(frame, side, now):
            state.recovery_count = 0
            return self._held_diag(side, state, "stale_or_invalid_hand", pinch)
        if not self._is_head_fresh(frame, now):
            state.recovery_count = 0
            return self._held_diag(side, state, "stale_head", pinch)

        if state.last_input_pose is not None:
            input_jump = float(np.linalg.norm(hand[:3, 3] - state.last_input_pose[:3, 3]))
            input_rot_jump = rotation_angle(hand[:3, :3], state.last_input_pose[:3, :3])
            if input_jump > self.config.max_input_jump:
                state.recovery_count = 0
                return self._held_diag(side, state, "input_translation_jump", pinch)
            if input_rot_jump > self.config.max_input_rotation_jump:
                state.recovery_count = 0
                return self._held_diag(side, state, "input_rotation_jump", pinch)

        if state.recovery_count < self.config.recovery_frames:
            state.recovery_count += 1
            state.last_input_pose = hand.copy()
            return self._held_diag(side, state, "recovering", pinch)

        assert state.init_head_relative_pos is not None
        assert state.init_robot_eef_pose is not None
        assert state.init_hand_mat is not None
        current_rel = self._operator_relative_position(head, hand)
        human_delta = current_rel - state.init_head_relative_pos

        target = state.init_robot_eef_pose.copy()
        axis_delta = self.config.axis_map(side) @ (self.config.pos_scale * human_delta)
        target[:3, 3] = state.init_robot_eef_pose[:3, 3] + axis_delta

        raw_rotation_deg = 0.0
        if not self.config.lock_orientation:
            raw_delta_rot = hand[:3, :3] @ state.init_hand_mat[:3, :3].T
            raw_rotation_deg = np.degrees(float(np.linalg.norm(Rotation.from_matrix(raw_delta_rot).as_rotvec())))
            scaled_delta = _rotation_with_gain(raw_delta_rot, self.config.orientation_scale)
            target[:3, :3] = scaled_delta @ state.init_robot_eef_pose[:3, :3]

        target = self.clamp_eef_pose(target, self.config.eef_max_radius, self.config.eef_min_z, self.config.eef_min_x)

        previous = state.filtered_target_pose if state.filtered_target_pose is not None else target
        max_translation_step = self.config.max_target_translation_speed * max(dt, 1e-4)
        limited_pos = _limit_vector_step(previous[:3, 3], target[:3, 3], max_translation_step)
        filtered_pos = previous[:3, 3] + self.config.translation_alpha * (limited_pos - previous[:3, 3])

        max_rotation_step = self.config.max_target_angular_speed * max(dt, 1e-4)
        limited_rot = _rotation_step(previous[:3, :3], target[:3, :3], max_rotation_step)
        filtered_rot = _slerp_alpha(previous[:3, :3], limited_rot, self.config.rotation_alpha)

        filtered = target.copy()
        filtered[:3, 3] = filtered_pos
        filtered[:3, :3] = filtered_rot

        state.filtered_target_pose = filtered.copy()
        state.last_good_target_pose = filtered.copy()
        state.last_input_pose = hand.copy()
        state.last_valid_time = now

        filtered_rotation_deg = rotation_angle(filtered[:3, :3], state.init_robot_eef_pose[:3, :3])
        diag = RetargetDiagnostics(
            side=side,
            input_valid=True,
            held_last_target=False,
            reason="ok",
            raw_hand_delta=human_delta,
            filtered_translation=filtered[:3, 3].copy(),
            raw_rotation_deg=raw_rotation_deg,
            filtered_rotation_deg=float(np.degrees(filtered_rotation_deg)),
            target_xyz=filtered[:3, 3].copy(),
            pinch=pinch,
        )
        return filtered, diag

    def update(self, frame: QuestFrame, dt: float, now: float | None = None) -> RetargetOutput:
        now = time.time() if now is None else now
        dt = max(float(dt), 1e-4)
        left_pose, left_diag = self._update_side(frame, "left", dt, now)
        right_pose, right_diag = self._update_side(frame, "right", dt, now)
        return RetargetOutput(
            left_pose=left_pose,
            right_pose=right_pose,
            left_gripper=self._pinch(frame, "left"),
            right_gripper=self._pinch(frame, "right"),
            diagnostics={"left": left_diag, "right": right_diag},
        )
