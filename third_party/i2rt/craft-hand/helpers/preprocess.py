"""OpenTeleVision-style hand preprocessing for Quest landmarks."""

from __future__ import annotations

import numpy as np


TIP_INDICES = [4, 9, 14, 19, 24]

HAND_TO_INSPIRE = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

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
    ret = np.eye(4, dtype=np.float64)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def matrix_is_valid(mat: np.ndarray | None) -> bool:
    if mat is None:
        return False
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        return False
    return bool(np.isfinite(mat).all() and abs(np.linalg.det(mat[:3, :3])) > 1e-8)


def mat_update(prev_mat: np.ndarray, mat: np.ndarray | None) -> np.ndarray:
    if not matrix_is_valid(mat):
        return prev_mat
    return np.asarray(mat, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


def vuer_matrix_to_robot(mat: np.ndarray) -> np.ndarray:
    return GRD_YUP_TO_GRD_ZUP @ mat @ fast_mat_inv(GRD_YUP_TO_GRD_ZUP)


def vuer_points_to_robot(points: np.ndarray) -> np.ndarray:
    return transform_points(points, GRD_YUP_TO_GRD_ZUP)


def wrist_relative_landmarks(
    landmarks_3d: np.ndarray,
    wrist_matrix: np.ndarray | None,
    robot_frame: bool = True,
    hand_frame: str = "none",
) -> np.ndarray:
    """Return landmarks in the wrist frame, matching OpenTeleVision's idea.

    `hand_frame="inspire"` applies the same fixed transform that upstream uses
    before dex-retargeting. For CRAFT's geometric retargeter, rigid transforms
    preserve curls and palm-relative side angles, so this mainly makes the
    signal independent of arm/head motion.
    """
    landmarks = np.asarray(landmarks_3d, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] < 3:
        raise ValueError(f"Expected landmarks Nx3, got {landmarks.shape}")
    points = landmarks[:, :3]

    if robot_frame:
        points = vuer_points_to_robot(points)
        wrist = vuer_matrix_to_robot(wrist_matrix) if matrix_is_valid(wrist_matrix) else None
    else:
        wrist = wrist_matrix if matrix_is_valid(wrist_matrix) else None

    if wrist is None:
        points = points - points[0]
    else:
        points = transform_points(points, fast_mat_inv(wrist))

    if hand_frame == "inspire":
        points = transform_points(points, HAND_TO_INSPIRE.T)
    elif hand_frame != "none":
        raise ValueError(f"Unsupported hand frame: {hand_frame}")
    return points


class QuestHandPreprocessor:
    """Stateful preprocessor with invalid-matrix fallback like OpenTeleVision."""

    def __init__(self, side: str = "right") -> None:
        self.side = side
        self._last_wrist = np.eye(4, dtype=np.float64)

    def process(
        self,
        landmarks_3d: np.ndarray,
        wrist_matrix: np.ndarray | None,
        robot_frame: bool = True,
        hand_frame: str = "none",
    ) -> np.ndarray:
        self._last_wrist = mat_update(self._last_wrist, wrist_matrix)
        return wrist_relative_landmarks(
            landmarks_3d,
            self._last_wrist,
            robot_frame=robot_frame,
            hand_frame=hand_frame,
        )
