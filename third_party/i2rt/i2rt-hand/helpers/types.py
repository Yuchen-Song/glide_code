from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SIDE_NAMES = ("left", "right")


@dataclass
class QuestFrame:
    """One host-side snapshot of Quest/Vuer tracking state."""

    timestamp: float
    head_mat: np.ndarray | None = None
    left_hand_mat: np.ndarray | None = None
    right_hand_mat: np.ndarray | None = None
    left_landmarks: np.ndarray | None = None
    right_landmarks: np.ndarray | None = None
    left_state: dict[str, float | bool] | None = None
    right_state: dict[str, float | bool] | None = None
    head_valid: bool = False
    left_valid: bool = False
    right_valid: bool = False
    head_event_count: int = 0
    left_event_count: int = 0
    right_event_count: int = 0
    head_last_timestamp: float = 0.0
    left_last_timestamp: float = 0.0
    right_last_timestamp: float = 0.0

    def hand_mat(self, side: str) -> np.ndarray | None:
        return self.left_hand_mat if side == "left" else self.right_hand_mat

    def landmarks(self, side: str) -> np.ndarray | None:
        return self.left_landmarks if side == "left" else self.right_landmarks

    def hand_state(self, side: str) -> dict[str, float | bool] | None:
        return self.left_state if side == "left" else self.right_state

    def hand_valid(self, side: str) -> bool:
        return self.left_valid if side == "left" else self.right_valid

    def hand_event_count(self, side: str) -> int:
        return self.left_event_count if side == "left" else self.right_event_count

    def hand_last_timestamp(self, side: str) -> float:
        return self.left_last_timestamp if side == "left" else self.right_last_timestamp
