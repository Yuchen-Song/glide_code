from __future__ import annotations

import asyncio
import os
import signal
import time
from multiprocessing import Array, Process, Value
from pathlib import Path
from typing import Any

import numpy as np
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import Hands, MotionControllers, OrbitControls, Scene

from .types import QuestFrame


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CERT_FILE = REPO_ROOT / "TeleVision" / "cert.pem"
DEFAULT_KEY_FILE = REPO_ROOT / "TeleVision" / "key.pem"


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("matrix", "wrist", "hand", "landmarks", "joints", "data", "value"):
            arr = _numeric_array(value.get(key))
            if arr is not None:
                return arr
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


def _payload_by_keys(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _hand_payload(data: dict[str, Any], side: str) -> Any:
    return _payload_by_keys(
        data,
        (
            f"{side}Hand",
            f"{side}_hand_matrix",
            f"{side}Matrix",
            f"{side}_matrix",
            f"{side}Wrist",
            f"{side}_wrist",
        ),
    )


def _landmark_payload(data: dict[str, Any], side: str) -> Any:
    return _payload_by_keys(
        data,
        (
            f"{side}Landmarks",
            f"{side}_landmarks",
            f"{side}Joints",
            f"{side}_joints",
            side,
            f"{side}_hand",
        ),
    )


def _state_payload(data: dict[str, Any], side: str) -> dict[str, Any] | None:
    for key in (f"{side}State", f"{side}_state", f"{side}HandState"):
        state = data.get(key)
        if isinstance(state, dict):
            return state
    return None


def _write_landmarks(flat: np.ndarray, landmarks_shared) -> bool:
    if flat.size >= 400:
        landmarks_shared[:] = flat[:400]
        return True
    if flat.size >= 25 * 3:
        transforms = np.tile(np.eye(4, dtype=np.float64).reshape(16, order="F"), 25)
        points = flat[: 25 * 3].reshape(25, 3)
        for idx, xyz in enumerate(points):
            base = idx * 16
            mat = transforms[base : base + 16].reshape(4, 4, order="F")
            mat[:3, 3] = xyz
            transforms[base : base + 16] = mat.reshape(16, order="F")
        landmarks_shared[:] = transforms
        return True
    return False


def _write_state(state: dict[str, Any] | None, state_shared) -> bool:
    if state is None:
        return False
    state_shared[0] = float(state.get("pinch", False))
    state_shared[1] = float(state.get("squeeze", False))
    state_shared[2] = float(state.get("tap", False))
    state_shared[3] = float(state.get("pinchValue", 0.0))
    state_shared[4] = float(state.get("squeezeValue", 0.0))
    state_shared[5] = float(state.get("tapValue", 0.0))
    return True


def _mark_valid(valid_flag, timestamp, count) -> None:
    with valid_flag.get_lock():
        valid_flag.value = True
    now = time.time()
    with timestamp.get_lock():
        timestamp.value = now
    with count.get_lock():
        count.value += 1


def _run_quest_app(
    cert_file: str,
    key_file: str,
    ngrok: bool,
    fps: int,
    show_left: bool,
    show_right: bool,
    head_shared,
    left_hand_shared,
    right_hand_shared,
    left_landmarks_shared,
    right_landmarks_shared,
    left_state_shared,
    right_state_shared,
    head_valid,
    left_valid,
    right_valid,
    head_timestamp,
    left_timestamp,
    right_timestamp,
    head_count,
    left_count,
    right_count,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    original_deserialize = ClientEvent._deserialize

    @classmethod
    def quiet_deserialize(cls, *, etype, ts=None, **kwargs):
        # Vuer's current client omits ts on CAMERA_MOVE and prints every frame.
        # Use receipt time in this probe process so debug logs stay readable.
        return ClientEvent(etype=etype, ts=time.time() if ts is None else ts, **kwargs)

    ClientEvent._deserialize = quiet_deserialize
    if ngrok:
        app = Vuer(host="0.0.0.0", queries=dict(grid=False), queue_len=3)
    else:
        app = Vuer(host="0.0.0.0", cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

    async def on_camera_move(event, session, fps=fps):
        try:
            matrix = event.value.get("camera", {}).get("matrix") if isinstance(event.value, dict) else None
            arr = _numeric_array(matrix)
            if arr is not None and arr.size >= 16:
                head_shared[:] = arr[:16]
                _mark_valid(head_valid, head_timestamp, head_count)
        except Exception as exc:
            print(f"quest_source_camera_error={type(exc).__name__}: {exc}")

    def write_side(data: dict[str, Any], side: str) -> None:
        hand_shared = left_hand_shared if side == "left" else right_hand_shared
        landmarks_shared = left_landmarks_shared if side == "left" else right_landmarks_shared
        state_shared = left_state_shared if side == "left" else right_state_shared
        valid = left_valid if side == "left" else right_valid
        timestamp = left_timestamp if side == "left" else right_timestamp
        count = left_count if side == "left" else right_count

        wrote = False
        hand = _numeric_array(_hand_payload(data, side))
        landmarks = _numeric_array(_landmark_payload(data, side))
        if hand is not None and hand.size >= 16:
            hand_shared[:] = hand[:16]
            wrote = True
        if landmarks is not None:
            if landmarks.size >= 400:
                hand_shared[:] = landmarks[:16]
            wrote = _write_landmarks(landmarks, landmarks_shared) or wrote
        wrote = _write_state(_state_payload(data, side), state_shared) or wrote
        if wrote:
            _mark_valid(valid, timestamp, count)

    async def on_hand_move(event, session, fps=fps):
        try:
            data = event.value if isinstance(event.value, dict) else {}
            write_side(data, "left")
            write_side(data, "right")
        except Exception as exc:
            print(f"quest_source_hand_error={type(exc).__name__}: {exc}")

    async def main(session, fps=fps):
        session.set @ Scene(
            bgChildren=[
                Hands(fps=fps, stream=True, key="hands", showLeft=show_left, showRight=show_right),
                MotionControllers(stream=True, key="motion-controller", left=True, right=True),
                OrbitControls(stream=True, makeDefault=True, key="camera-controls"),
            ]
        )
        while True:
            await asyncio.sleep(0.03)

    app.add_handler("CAMERA_MOVE")(on_camera_move)
    app.add_handler("HAND_MOVE")(on_hand_move)
    app.spawn(start=False)(main)
    try:
        app.run()
    finally:
        ClientEvent._deserialize = original_deserialize


class QuestSource:
    """Live Quest/Vuer source with packet freshness metadata."""

    def __init__(
        self,
        fps: int = 60,
        cert_file: str | os.PathLike[str] = DEFAULT_CERT_FILE,
        key_file: str | os.PathLike[str] = DEFAULT_KEY_FILE,
        ngrok: bool = False,
        show_left: bool = False,
        show_right: bool = False,
    ) -> None:
        self.head_shared = Array("d", 16, lock=True)
        self.left_hand_shared = Array("d", 16, lock=True)
        self.right_hand_shared = Array("d", 16, lock=True)
        self.left_landmarks_shared = Array("d", 400, lock=True)
        self.right_landmarks_shared = Array("d", 400, lock=True)
        self.left_state_shared = Array("d", 6, lock=True)
        self.right_state_shared = Array("d", 6, lock=True)
        self._head_valid = Value("b", False, lock=True)
        self._left_valid = Value("b", False, lock=True)
        self._right_valid = Value("b", False, lock=True)
        self._head_timestamp = Value("d", 0.0, lock=True)
        self._left_timestamp = Value("d", 0.0, lock=True)
        self._right_timestamp = Value("d", 0.0, lock=True)
        self._head_count = Value("i", 0, lock=True)
        self._left_count = Value("i", 0, lock=True)
        self._right_count = Value("i", 0, lock=True)
        self.process = Process(
            target=_run_quest_app,
            args=(
                str(cert_file),
                str(key_file),
                ngrok,
                int(fps),
                show_left,
                show_right,
                self.head_shared,
                self.left_hand_shared,
                self.right_hand_shared,
                self.left_landmarks_shared,
                self.right_landmarks_shared,
                self.left_state_shared,
                self.right_state_shared,
                self._head_valid,
                self._left_valid,
                self._right_valid,
                self._head_timestamp,
                self._left_timestamp,
                self._right_timestamp,
                self._head_count,
                self._left_count,
                self._right_count,
            ),
        )
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def _flag(value) -> bool:
        with value.get_lock():
            return bool(value.value)

    @staticmethod
    def _number(value) -> float:
        with value.get_lock():
            return float(value.value)

    @staticmethod
    def _count(value) -> int:
        with value.get_lock():
            return int(value.value)

    @staticmethod
    def _state(shared) -> dict[str, float | bool]:
        data = np.array(shared[:], dtype=np.float64)
        return {
            "pinch": bool(data[0]),
            "squeeze": bool(data[1]),
            "tap": bool(data[2]),
            "pinchValue": float(data[3]),
            "squeezeValue": float(data[4]),
            "tapValue": float(data[5]),
        }

    @staticmethod
    def _landmarks(shared) -> np.ndarray | None:
        arr = np.array(shared[:], dtype=np.float64)
        if not np.any(arr):
            return None
        mats = arr.reshape(25, 16)
        points = []
        for idx in range(25):
            mat = mats[idx].reshape(4, 4, order="F")
            points.append(mat[:3, 3])
        return np.asarray(points, dtype=np.float64)

    def latest_frame(self) -> QuestFrame:
        head_valid = self._flag(self._head_valid)
        left_valid = self._flag(self._left_valid)
        right_valid = self._flag(self._right_valid)
        return QuestFrame(
            timestamp=time.time(),
            head_mat=np.array(self.head_shared[:], dtype=np.float64).reshape(4, 4, order="F") if head_valid else None,
            left_hand_mat=np.array(self.left_hand_shared[:], dtype=np.float64).reshape(4, 4, order="F") if left_valid else None,
            right_hand_mat=np.array(self.right_hand_shared[:], dtype=np.float64).reshape(4, 4, order="F") if right_valid else None,
            left_landmarks=self._landmarks(self.left_landmarks_shared) if left_valid else None,
            right_landmarks=self._landmarks(self.right_landmarks_shared) if right_valid else None,
            left_state=self._state(self.left_state_shared) if left_valid else None,
            right_state=self._state(self.right_state_shared) if right_valid else None,
            head_valid=head_valid,
            left_valid=left_valid,
            right_valid=right_valid,
            head_event_count=self._count(self._head_count),
            left_event_count=self._count(self._left_count),
            right_event_count=self._count(self._right_count),
            head_last_timestamp=self._number(self._head_timestamp),
            left_last_timestamp=self._number(self._left_timestamp),
            right_last_timestamp=self._number(self._right_timestamp),
        )

    def cleanup(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)

    def __enter__(self) -> "QuestSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
