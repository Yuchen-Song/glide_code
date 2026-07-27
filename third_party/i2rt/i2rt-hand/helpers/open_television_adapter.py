from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .types import QuestFrame


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENTELEVISION_ROOT = REPO_ROOT / "third_party" / "open_television" / "TeleVision"
OPENTELEVISION_TELEOP_DIR = OPENTELEVISION_ROOT / "teleop"


@contextmanager
def _upstream_import_context() -> Iterator[None]:
    """Import Open-TeleVision modules from the cloned repo, not local copies."""
    module_names = ("Preprocessor", "constants_vuer", "motion_utils")
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(OPENTELEVISION_TELEOP_DIR))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in module_names:
            sys.modules.pop(name, None)
            module = saved_modules[name]
            if module is not None:
                sys.modules[name] = module


def _load_vuer_preprocessor_class():
    if not OPENTELEVISION_TELEOP_DIR.exists():
        raise FileNotFoundError(
            f"Open-TeleVision clone not found at {OPENTELEVISION_ROOT}. "
            "Clone https://github.com/OpenTeleVision/TeleVision.git there first."
        )
    with _upstream_import_context():
        module = importlib.import_module("Preprocessor")
    return module.VuerPreprocessor


def _matrix_or_zero(mat: np.ndarray | None) -> np.ndarray:
    if mat is None:
        return np.zeros((4, 4), dtype=np.float64)
    arr = np.asarray(mat, dtype=np.float64)
    return arr if arr.shape == (4, 4) else np.zeros((4, 4), dtype=np.float64)


def _landmarks_or_zeros(landmarks: np.ndarray | None) -> np.ndarray:
    if landmarks is None:
        return np.zeros((25, 3), dtype=np.float64)
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] >= 3:
        if pts.shape[0] >= 25:
            return pts[:25, :3]
        padded = np.zeros((25, 3), dtype=np.float64)
        padded[: pts.shape[0], :] = pts[:, :3]
        return padded
    return np.zeros((25, 3), dtype=np.float64)


class _QuestFrameTeleVisionShim:
    def __init__(self, frame: QuestFrame) -> None:
        # Open-TeleVision's mat_update keeps its previous/default pose when the
        # incoming matrix has determinant zero. Use that path for missing packets.
        self.head_matrix = _matrix_or_zero(frame.head_mat)
        self.left_hand = _matrix_or_zero(frame.left_hand_mat)
        self.right_hand = _matrix_or_zero(frame.right_hand_mat)
        self.left_landmarks = _landmarks_or_zeros(frame.left_landmarks)
        self.right_landmarks = _landmarks_or_zeros(frame.right_landmarks)


@dataclass
class OpenTeleVisionProcessedFrame:
    head_mat: np.ndarray
    left_wrist_mat: np.ndarray
    right_wrist_mat: np.ndarray
    left_fingers: np.ndarray
    right_fingers: np.ndarray

    def wrist_mat(self, side: str) -> np.ndarray:
        return self.left_wrist_mat if side == "left" else self.right_wrist_mat

    def fingers(self, side: str) -> np.ndarray:
        return self.left_fingers if side == "left" else self.right_fingers


class OpenTeleVisionFrameAdapter:
    """Thin adapter around upstream Open-TeleVision's VuerPreprocessor."""

    def __init__(self) -> None:
        self._preprocessor = _load_vuer_preprocessor_class()()

    def process(self, frame: QuestFrame) -> OpenTeleVisionProcessedFrame:
        head, left_wrist, right_wrist, left_fingers, right_fingers = self._preprocessor.process(
            _QuestFrameTeleVisionShim(frame)
        )
        return OpenTeleVisionProcessedFrame(
            head_mat=np.asarray(head, dtype=np.float64),
            left_wrist_mat=np.asarray(left_wrist, dtype=np.float64),
            right_wrist_mat=np.asarray(right_wrist, dtype=np.float64),
            left_fingers=np.asarray(left_fingers, dtype=np.float64),
            right_fingers=np.asarray(right_fingers, dtype=np.float64),
        )
