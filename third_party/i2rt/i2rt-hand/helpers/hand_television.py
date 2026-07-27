"""i2rt hand TeleVision bridge with Quest hand packet support.

This file is copied from `TeleVision/TeleVision.py` for the hand-only arm
experiment. Keeping it under `i2rt-hand/` prevents the normal i2rt controller
teleop path from changing while we debug Quest hand input.

The important addition here is robust `HAND_MOVE` parsing:
- Accepts several Vuer hand packet key layouts.
- Stores wrist transforms for left and right hands.
- Stores 25 landmark transforms or position-only landmarks when available.
- Stores Quest hand gesture state values such as pinch, squeeze, and tap.

`helpers.dual_arm_helpers` consumes these shared-memory hand values and maps
them onto arm motion, grippers, and recording controls.
"""

import signal
import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene, MotionControllers, Scene
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
from typing import Any
import numpy as np
import asyncio


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("matrix", "wrist", "hand", "landmarks", "joints", "data", "value"):
            if key in value:
                arr = _numeric_array(value[key])
                if arr is not None:
                    return arr
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _payload_by_keys(hand_data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in hand_data and hand_data[key] is not None:
            return hand_data[key]
    return None


def _combined_side_payload(hand_data: dict[str, Any], side: str) -> Any:
    for key in (side, f"{side}_hand", f"{side}_landmarks"):
        if key in hand_data and hand_data[key] is not None:
            return hand_data[key]
    return None


def _hand_payload(hand_data: dict[str, Any], side: str) -> Any:
    return _payload_by_keys(
        hand_data,
        (
            f"{side}Hand",
            f"{side}_hand_matrix",
            f"{side}Matrix",
            f"{side}_matrix",
            f"{side}Wrist",
            f"{side}_wrist",
        ),
    )


def _landmark_payload(hand_data: dict[str, Any], side: str) -> Any:
    return _payload_by_keys(
        hand_data,
        (
            f"{side}Landmarks",
            f"{side}_landmarks",
            f"{side}Joints",
            f"{side}_joints",
        ),
    )


def _state_payload(hand_data: dict[str, Any], side: str) -> dict[str, Any] | None:
    for key in (f"{side}State", f"{side}_state", f"{side}HandState"):
        state = hand_data.get(key)
        if isinstance(state, dict):
            return state
    return None


def _write_hand_arrays(flat: np.ndarray, hand_shared, landmarks_shared, valid_flag) -> None:
    wrote = False
    if flat.size == 16:
        hand_shared[:] = flat[:16]
        wrote = True
    elif flat.size >= 400:
        hand_shared[:] = flat[:16]
        landmarks_shared[:] = flat[:400]
        wrote = True
    elif flat.size >= 25 * 3:
        transforms = np.tile(np.eye(4, dtype=np.float64).reshape(16, order="F"), 25)
        points = flat[: 25 * 3].reshape(25, 3)
        for idx, xyz in enumerate(points):
            base = idx * 16
            mat = transforms[base : base + 16].reshape(4, 4, order="F")
            mat[:3, 3] = xyz
            transforms[base : base + 16] = mat.reshape(16, order="F")
        landmarks_shared[:] = transforms
        wrote = True
    if wrote:
        with valid_flag.get_lock():
            valid_flag.value = True


def _write_hand_state(state: dict[str, Any] | None, state_shared, valid_flag) -> None:
    if state is None:
        return
    state_shared[0] = float(state.get('pinch', False))
    state_shared[1] = float(state.get('squeeze', False))
    state_shared[2] = float(state.get('tap', False))
    state_shared[3] = float(state.get('pinchValue', 0.0))
    state_shared[4] = float(state.get('squeezeValue', 0.0))
    state_shared[5] = float(state.get('tapValue', 0.0))
    with valid_flag.get_lock():
        valid_flag.value = True


def _write_hand_payload(
    hand_data: dict[str, Any],
    side: str,
    hand_shared,
    landmarks_shared,
    state_shared,
    valid_flag,
) -> None:
    hand = _numeric_array(_hand_payload(hand_data, side))
    landmarks = _numeric_array(_landmark_payload(hand_data, side))
    if hand is None and landmarks is None:
        landmarks = _numeric_array(_combined_side_payload(hand_data, side))
    if hand is not None:
        _write_hand_arrays(hand, hand_shared, landmarks_shared, valid_flag)
    if landmarks is not None:
        _write_hand_arrays(landmarks, hand_shared, landmarks_shared, valid_flag)
    _write_hand_state(_state_payload(hand_data, side), state_shared, valid_flag)


def _input_only_scene(fps: int = 60) -> Scene:
    return Scene(
        bgChildren=[
            Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False),
            MotionControllers(stream=True, key="motion-controller", left=True, right=True),
        ]
    )


class OpenTeleVision:
    def __init__(
        self,
        img_shape,
        shm_name,
        queue,
        toggle_streaming,
        stream_mode="image",
        cert_file="./cert.pem",
        key_file="./key.pem",
        ngrok=False,
        image_background=True,
    ):
        # self.app=Vuer()
        self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.image_background = image_background

        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        self.app.add_handler("CONTROLLER_MOVE")(self.on_controller_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=False)(self.main_image)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 400, lock=True)  # 25 joints * 16 values each
        self.right_landmarks_shared = Array('d', 400, lock=True)  # 25 joints * 16 values each

        # Hand state arrays for gesture tracking (pinch, squeeze, tap)
        self.left_hand_state_shared = Array('d', 6, lock=True)  # pinch, squeeze, tap, pinchValue, squeezeValue, tapValue
        self.right_hand_state_shared = Array('d', 6, lock=True)

        # Controller shared memory arrays
        self.left_controller_matrix_shared = Array('d', 16, lock=True)
        self.right_controller_matrix_shared = Array('d', 16, lock=True)

        # Controller button states (using 'd' for double to store boolean and float values)
        # Order: trigger, squeeze, touchpad, thumbstick, aButton, bButton, triggerValue, squeezeValue, touchpadX, touchpadY, thumbstickX, thumbstickY
        self.left_controller_state_shared = Array('d', 12, lock=True)
        self.right_controller_state_shared = Array('d', 12, lock=True)

        self.head_matrix_shared = Array('d', 16, lock=True)
        # self.aspect_shared = Value('d', 1.0, lock=True)

        self._left_hand_valid_flag_shared = Value('b', False, lock=True)
        self._right_hand_valid_flag_shared = Value('b', False, lock=True)
        self._left_controller_valid_flag_shared = Value('b', False, lock=True)
        self._right_controller_valid_flag_shared = Value('b', False, lock=True)


        # Use a module-level function wrapper to avoid pickling issues
        # Pass only picklable parameters, create app inside child process
        self.process = Process(target=_run_teleop_app, args=(
            cert_file, key_file, ngrok, stream_mode, shm_name,
            self.img_shape, self.img_height, self.img_width,
            self.left_hand_shared, self.right_hand_shared,
            self.left_landmarks_shared, self.right_landmarks_shared,
            self.left_hand_state_shared, self.right_hand_state_shared,
            self.left_controller_matrix_shared, self.right_controller_matrix_shared,
            self.left_controller_state_shared, self.right_controller_state_shared,
            self.head_matrix_shared,
            self._left_hand_valid_flag_shared, self._right_hand_valid_flag_shared,
            self._left_controller_valid_flag_shared, self._right_controller_valid_flag_shared,
            self.image_background,
        ))
        self.process.daemon = True
        self.process.start()


    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.app.run()

    async def on_cam_move(self, event, session, fps=60):
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # with self.aspect_shared.get_lock():
            #     self.aspect_shared.value = event.value['camera']['aspect']
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    async def on_hand_move(self, event, session, fps=60):
        """Handle hand movement events with new vuer API structure"""
        try:
            hand_data = event.value if isinstance(event.value, dict) else {}
            _write_hand_payload(
                hand_data,
                'left',
                self.left_hand_shared,
                self.left_landmarks_shared,
                self.left_hand_state_shared,
                self._left_hand_valid_flag_shared,
            )
            _write_hand_payload(
                hand_data,
                'right',
                self.right_hand_shared,
                self.right_landmarks_shared,
                self.right_hand_state_shared,
                self._right_hand_valid_flag_shared,
            )
        except Exception as e:
            print(f"Error in hand handler: {e}")
            pass

    async def on_controller_move(self, event, session, fps=60):
        """Handle controller movement events from Quest 3 motion controllers"""
        try:
            controller_data = event.value
            # print(f"Controller data received: {controller_data.keys()}")

            # Update left controller if present
            if 'left' in controller_data and controller_data['left'] is not None:
                # print("left controller data size", len(controller_data['left']))
                if len(controller_data['left']) >= 16:
                    self.left_controller_matrix_shared[:] = controller_data['left']
                    with self._left_controller_valid_flag_shared.get_lock():
                        self._left_controller_valid_flag_shared.value = True
                else:
                    print(f"Warning: Left controller data size is {len(controller_data['left'])}, expected 16")

            # Update right controller if present
            if 'right' in controller_data and controller_data['right'] is not None:
                # print("right controller data size", len(controller_data['right']))
                if len(controller_data['right']) >= 16:
                    self.right_controller_matrix_shared[:] = controller_data['right']
                    with self._right_controller_valid_flag_shared.get_lock():
                        self._right_controller_valid_flag_shared.value = True
                else:
                    print(f"Warning: Right controller data size is {len(controller_data['right'])}, expected 16")

            # Update left controller state
            if 'leftState' in controller_data and controller_data['leftState'] is not None:
                state = controller_data['leftState']
                # print(f"Left controller state: {state}")
                self.left_controller_state_shared[0] = float(state.get('trigger', False))
                self.left_controller_state_shared[1] = float(state.get('squeeze', False))
                self.left_controller_state_shared[2] = float(state.get('touchpad', False))
                self.left_controller_state_shared[3] = float(state.get('thumbstick', False))
                self.left_controller_state_shared[4] = float(state.get('aButton', False))
                self.left_controller_state_shared[5] = float(state.get('bButton', False))
                self.left_controller_state_shared[6] = float(state.get('triggerValue', 0.0))
                self.left_controller_state_shared[7] = float(state.get('squeezeValue', 0.0))
                touchpad_val = state.get('touchpadValue', [0.0, 0.0])
                self.left_controller_state_shared[8] = float(touchpad_val[0])
                self.left_controller_state_shared[9] = float(touchpad_val[1])
                thumbstick_val = state.get('thumbstickValue', [0.0, 0.0])
                self.left_controller_state_shared[10] = float(thumbstick_val[0])
                self.left_controller_state_shared[11] = float(thumbstick_val[1])

            # Update right controller state
            if 'rightState' in controller_data and controller_data['rightState'] is not None:
                state = controller_data['rightState']
                # print(f"Right controller state: {state}")
                self.right_controller_state_shared[0] = float(state.get('trigger', False))
                self.right_controller_state_shared[1] = float(state.get('squeeze', False))
                self.right_controller_state_shared[2] = float(state.get('touchpad', False))
                self.right_controller_state_shared[3] = float(state.get('thumbstick', False))
                self.right_controller_state_shared[4] = float(state.get('aButton', False))
                self.right_controller_state_shared[5] = float(state.get('bButton', False))
                self.right_controller_state_shared[6] = float(state.get('triggerValue', 0.0))
                self.right_controller_state_shared[7] = float(state.get('squeezeValue', 0.0))
                touchpad_val = state.get('touchpadValue', [0.0, 0.0])
                self.right_controller_state_shared[8] = float(touchpad_val[0])
                self.right_controller_state_shared[9] = float(touchpad_val[1])
                thumbstick_val = state.get('thumbstickValue', [0.0, 0.0])
                self.right_controller_state_shared[10] = float(thumbstick_val[0])
                self.right_controller_state_shared[11] = float(thumbstick_val[1])

        except Exception as e:
            print(f"Error in controller handler: {e}")
            import traceback
            traceback.print_exc()
            pass

    async def main_image(self, session, fps=60):
        session.set @ _input_only_scene(fps)
        if not self.image_background:
            while True:
                await asyncio.sleep(0.03)
        end_time = time.time()
        while True:
            try:
                start = time.time()
                # Check if session is still connected
                if not hasattr(session, 'CURRENT_WS_ID') or session.CURRENT_WS_ID not in session.vuer.ws:
                    print("Session disconnected, breaking loop")
                    break

                display_image = self.img_array

                # Use a try-catch around the upsert operations to handle session issues
                try:
                    session.upsert(
                    [ImageBackground(
                        # Can scale the images down.
                        display_image[::2, :self.img_width],
                        # display_image[:self.img_height:2, ::2],
                        # 'jpg' encoding is significantly faster than 'png'.
                        format="jpeg",
                        quality=80,
                        key="left-image",
                        interpolate=True,
                        # fixed=True,
                        aspect=1.66667,
                        # distanceToCamera=0.5,
                        height = 8,
                        position=[0, -1, 3],
                        # rotation=[0, 0, 0],
                        layers=1,
                        alphaSrc="./vinette.jpg"
                    ),
                    ImageBackground(
                        # Can scale the images down.
                        display_image[::2, self.img_width:],
                        # display_image[self.img_height::2, ::2],
                        # 'jpg' encoding is significantly faster than 'png'.
                        format="jpeg",
                        quality=80,
                        key="right-image",
                        interpolate=True,
                        # fixed=True,
                        aspect=1.66667,
                        # distanceToCamera=0.5,
                        height = 8,
                        position=[0, -1, 3],
                        # rotation=[0, 0, 0],
                        layers=2,
                        alphaSrc="./vinette.jpg"
                    )],
                    to="bgChildren",
                    )
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    # If upsert fails, the session might be disconnected
                    if "Websocket session is missing" in str(e):
                        print("Websocket session lost, breaking loop")
                        break
                    # For other errors, continue trying

                end_time = time.time()
                await asyncio.sleep(0.03)

            except Exception as e:
                print(f"Error in main_image loop: {e}")
                await asyncio.sleep(0.1)  # Wait a bit before retrying

    @property
    def left_hand(self):
        # with self.left_hand_shared.get_lock():
        #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")


    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")


    @property
    def left_landmarks(self):
        """Get left hand landmarks as 25x16 array (25 joints, each with 4x4 transform matrix)"""
        with self._left_hand_valid_flag_shared.get_lock():
            if self._left_hand_valid_flag_shared.value:
                return np.array(self.left_landmarks_shared[:]).reshape(25, 16)
            return None

    @property
    def right_landmarks(self):
        """Get right hand landmarks as 25x16 array (25 joints, each with 4x4 transform matrix)"""
        with self._right_hand_valid_flag_shared.get_lock():
            if self._right_hand_valid_flag_shared.value:
                return np.array(self.right_landmarks_shared[:]).reshape(25, 16)
            return None

    @property
    def left_hand_landmarks_3d(self):
        """Get left hand landmark positions as 25x3 array (just positions, not full transforms)"""
        landmarks = self.left_landmarks
        if landmarks is not None:
            # Extract position from each 4x4 transform matrix (last column, first 3 values)
            positions = []
            for i in range(25):
                transform = landmarks[i].reshape(4, 4, order="F")
                positions.append(transform[:3, 3])  # Translation component
            return np.array(positions)
        return None

    @property
    def right_hand_landmarks_3d(self):
        """Get right hand landmark positions as 25x3 array (just positions, not full transforms)"""
        landmarks = self.right_landmarks
        if landmarks is not None:
            # Extract position from each 4x4 transform matrix (last column, first 3 values)
            positions = []
            for i in range(25):
                transform = landmarks[i].reshape(4, 4, order="F")
                positions.append(transform[:3, 3])  # Translation component
            return np.array(positions)
        return None

    @property
    def left_hand_state(self):
        """Get left hand gesture state (pinch, squeeze, tap)"""
        with self._left_hand_valid_flag_shared.get_lock():
            if self._left_hand_valid_flag_shared.value:
                state_data = np.array(self.left_hand_state_shared[:])
                return {
                    'pinch': bool(state_data[0]),
                    'squeeze': bool(state_data[1]),
                    'tap': bool(state_data[2]),
                    'pinchValue': float(state_data[3]),
                    'squeezeValue': float(state_data[4]),
                    'tapValue': float(state_data[5])
                }
            return None

    @property
    def right_hand_state(self):
        """Get right hand gesture state (pinch, squeeze, tap)"""
        with self._right_hand_valid_flag_shared.get_lock():
            if self._right_hand_valid_flag_shared.value:
                state_data = np.array(self.right_hand_state_shared[:])
                return {
                    'pinch': bool(state_data[0]),
                    'squeeze': bool(state_data[1]),
                    'tap': bool(state_data[2]),
                    'pinchValue': float(state_data[3]),
                    'squeezeValue': float(state_data[4]),
                    'tapValue': float(state_data[5])
                }
            return None

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    # @property
    # def aspect(self):
    #     # with self.aspect_shared.get_lock():
    #         # return float(self.aspect_shared.value)
    #     return float(self.aspect_shared.value)

    @property
    def is_left_hand_valid(self):
        """Check if left hand data is valid/available"""
        with self._left_hand_valid_flag_shared.get_lock():
            return bool(self._left_hand_valid_flag_shared.value)

    @property
    def is_right_hand_valid(self):
        with self._right_hand_valid_flag_shared.get_lock():
            return bool(self._right_hand_valid_flag_shared.value)

    @property
    def left_controller_matrix(self):
        """Get left controller 4x4 transform matrix"""
        with self._left_controller_valid_flag_shared.get_lock():
            if self._left_controller_valid_flag_shared.value:
                return np.array(self.left_controller_matrix_shared[:]).reshape(4, 4, order="F")
            return None

    @property
    def right_controller_matrix(self):
        """Get right controller 4x4 transform matrix"""
        with self._right_controller_valid_flag_shared.get_lock():
            if self._right_controller_valid_flag_shared.value:
                return np.array(self.right_controller_matrix_shared[:]).reshape(4, 4, order="F")
            return None

    @property
    def left_controller_state(self):
        """Get left controller button/input states as dict"""
        with self._left_controller_valid_flag_shared.get_lock():
            if self._left_controller_valid_flag_shared.value:
                state_data = np.array(self.left_controller_state_shared[:])
                return {
                    'trigger': bool(state_data[0]),
                    'squeeze': bool(state_data[1]),
                    'touchpad': bool(state_data[2]),
                    'thumbstick': bool(state_data[3]),
                    'aButton': bool(state_data[4]),
                    'bButton': bool(state_data[5]),
                    'triggerValue': float(state_data[6]),
                    'squeezeValue': float(state_data[7]),
                    'touchpadValue': [float(state_data[8]), float(state_data[9])],
                    'thumbstickValue': [float(state_data[10]), float(state_data[11])]
                }
            return None

    @property
    def right_controller_state(self):
        """Get right controller button/input states as dict"""
        with self._right_controller_valid_flag_shared.get_lock():
            if self._right_controller_valid_flag_shared.value:
                state_data = np.array(self.right_controller_state_shared[:])
                return {
                    'trigger': bool(state_data[0]),
                    'squeeze': bool(state_data[1]),
                    'touchpad': bool(state_data[2]),
                    'thumbstick': bool(state_data[3]),
                    'aButton': bool(state_data[4]),
                    'bButton': bool(state_data[5]),
                    'triggerValue': float(state_data[6]),
                    'squeezeValue': float(state_data[7]),
                    'touchpadValue': [float(state_data[8]), float(state_data[9])],
                    'thumbstickValue': [float(state_data[10]), float(state_data[11])]
                }
            return None

    @property
    def is_left_controller_valid(self):
        """Check if left controller data is valid/available"""
        with self._left_controller_valid_flag_shared.get_lock():
            return bool(self._left_controller_valid_flag_shared.value)

    @property
    def is_right_controller_valid(self):
        """Check if right controller data is valid/available"""
        with self._right_controller_valid_flag_shared.get_lock():
            return bool(self._right_controller_valid_flag_shared.value)


def _run_teleop_app(cert_file, key_file, ngrok, stream_mode, shm_name,
                    img_shape, img_height, img_width,
                    left_hand_shared, right_hand_shared,
                    left_landmarks_shared, right_landmarks_shared,
                    left_hand_state_shared, right_hand_state_shared,
                    left_controller_matrix_shared, right_controller_matrix_shared,
                    left_controller_state_shared, right_controller_state_shared,
                    head_matrix_shared,
                    _left_hand_valid_flag_shared, _right_hand_valid_flag_shared,
                    _left_controller_valid_flag_shared, _right_controller_valid_flag_shared,
                    image_background=True):
    """Module-level function to run the teleop app in a separate process.
    Creates the app inside the child process to avoid pickling issues."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create app inside child process (avoids pickling Vuer object)
    if ngrok:
        app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
    else:
        app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

    # Create wrapper functions for handlers that access shared memory
    async def on_cam_move_wrapper(event, session, fps=60):
        try:
            head_matrix_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_hand_move_wrapper(event, session, fps=60):
        """Handle Quest hand tracking events from the Vuer subprocess."""
        try:
            hand_data = event.value if isinstance(event.value, dict) else {}
            _write_hand_payload(
                hand_data,
                'left',
                left_hand_shared,
                left_landmarks_shared,
                left_hand_state_shared,
                _left_hand_valid_flag_shared,
            )
            _write_hand_payload(
                hand_data,
                'right',
                right_hand_shared,
                right_landmarks_shared,
                right_hand_state_shared,
                _right_hand_valid_flag_shared,
            )
        except Exception as e:
            print(f"Error in hand handler: {e}")
            pass

    async def on_controller_move_wrapper(event, session, fps=60):
        """Handle controller movement events from Quest 3 motion controllers"""
        try:
            controller_data = event.value

            # Update left controller if present
            if 'left' in controller_data and controller_data['left'] is not None:
                if len(controller_data['left']) >= 16:
                    left_controller_matrix_shared[:] = controller_data['left']
                    with _left_controller_valid_flag_shared.get_lock():
                        _left_controller_valid_flag_shared.value = True

            # Update right controller if present
            if 'right' in controller_data and controller_data['right'] is not None:
                if len(controller_data['right']) >= 16:
                    right_controller_matrix_shared[:] = controller_data['right']
                    with _right_controller_valid_flag_shared.get_lock():
                        _right_controller_valid_flag_shared.value = True

            # Update left controller state
            if 'leftState' in controller_data and controller_data['leftState'] is not None:
                state = controller_data['leftState']
                left_controller_state_shared[0] = float(state.get('trigger', False))
                left_controller_state_shared[1] = float(state.get('squeeze', False))
                left_controller_state_shared[2] = float(state.get('touchpad', False))
                left_controller_state_shared[3] = float(state.get('thumbstick', False))
                left_controller_state_shared[4] = float(state.get('aButton', False))
                left_controller_state_shared[5] = float(state.get('bButton', False))
                left_controller_state_shared[6] = float(state.get('triggerValue', 0.0))
                left_controller_state_shared[7] = float(state.get('squeezeValue', 0.0))
                touchpad_val = state.get('touchpadValue', [0.0, 0.0])
                left_controller_state_shared[8] = float(touchpad_val[0])
                left_controller_state_shared[9] = float(touchpad_val[1])
                thumbstick_val = state.get('thumbstickValue', [0.0, 0.0])
                left_controller_state_shared[10] = float(thumbstick_val[0])
                left_controller_state_shared[11] = float(thumbstick_val[1])

            # Update right controller state
            if 'rightState' in controller_data and controller_data['rightState'] is not None:
                state = controller_data['rightState']
                right_controller_state_shared[0] = float(state.get('trigger', False))
                right_controller_state_shared[1] = float(state.get('squeeze', False))
                right_controller_state_shared[2] = float(state.get('touchpad', False))
                right_controller_state_shared[3] = float(state.get('thumbstick', False))
                right_controller_state_shared[4] = float(state.get('aButton', False))
                right_controller_state_shared[5] = float(state.get('bButton', False))
                right_controller_state_shared[6] = float(state.get('triggerValue', 0.0))
                right_controller_state_shared[7] = float(state.get('squeezeValue', 0.0))
                touchpad_val = state.get('touchpadValue', [0.0, 0.0])
                right_controller_state_shared[8] = float(touchpad_val[0])
                right_controller_state_shared[9] = float(touchpad_val[1])
                thumbstick_val = state.get('thumbstickValue', [0.0, 0.0])
                right_controller_state_shared[10] = float(thumbstick_val[0])
                right_controller_state_shared[11] = float(thumbstick_val[1])

        except Exception as e:
            print(f"Error in controller handler: {e}")
            import traceback
            traceback.print_exc()
            pass

    async def main_image_wrapper(session, fps=60):
        session.set @ _input_only_scene(fps)
        if not image_background:
            while True:
                await asyncio.sleep(0.03)

        # Access shared memory for images
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)

        end_time = time.time()
        while True:
            try:
                start = time.time()
                # Check if session is still connected
                if not hasattr(session, 'CURRENT_WS_ID') or session.CURRENT_WS_ID not in session.vuer.ws:
                    print("Session disconnected, breaking loop")
                    break

                display_image = img_array

                try:
                    session.upsert(
                    [ImageBackground(
                        display_image[::2, :img_width],
                        format="jpeg",
                        quality=80,
                        key="left-image",
                        interpolate=True,
                        aspect=1.66667,
                        height = 8,
                        position=[0, -1, 3],
                        layers=1,
                        alphaSrc="./vinette.jpg"
                    ),
                    ImageBackground(
                        display_image[::2, img_width:],
                        format="jpeg",
                        quality=80,
                        key="right-image",
                        interpolate=True,
                        aspect=1.66667,
                        height = 8,
                        position=[0, -1, 3],
                        layers=2,
                        alphaSrc="./vinette.jpg"
                    )],
                    to="bgChildren",
                    )
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    if "Websocket session is missing" in str(e):
                        print("Websocket session lost, breaking loop")
                        break

                end_time = time.time()
                await asyncio.sleep(0.03)

            except Exception as e:
                print(f"Error in main_image loop: {e}")
                await asyncio.sleep(0.1)

    # Set up handlers
    app.add_handler("CAMERA_MOVE")(on_cam_move_wrapper)
    app.add_handler("HAND_MOVE")(on_hand_move_wrapper)
    app.add_handler("CONTROLLER_MOVE")(on_controller_move_wrapper)

    if stream_mode == "image":
        app.spawn(start=False)(main_image_wrapper)

    # Run the app
    app.run()


if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
    img_height, img_width = resolution_cropped[:2]  # 450 * 600
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    shm_name = shm.name
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(resolution_cropped, shm_name, None, None, cert_file="cert.pem", key_file="key.pem")
    while True:
        # print("is_left_hand_valid", tv.is_left_hand_valid)
        # print("is_right_hand_valid", tv.is_right_hand_valid)
        # print("is_left_controller_valid", tv.is_left_controller_valid)
        # print("is_right_controller_valid", tv.is_right_controller_valid)


        # Print hand tracking data
        # if tv.is_left_hand_valid:
        #     print("Left hand matrix:", tv.left_hand)
        #     left_state = tv.left_hand_state
        #     if left_state:
        #         print(f"Left hand - Pinch: {left_state['pinchValue']:.2f}, Squeeze: {left_state['squeezeValue']:.2f}, Tap: {left_state['tapValue']:.2f}")

        # if tv.is_right_hand_valid:
        #     print("Right hand matrix:", tv.right_hand)
        #     right_state = tv.right_hand_state
        #     if right_state:
        #         print(f"Right hand - Pinch: {right_state['pinchValue']:.2f}, Squeeze: {right_state['squeezeValue']:.2f}, Tap: {right_state['tapValue']:.2f}")

        # Print controller data if available
        # if tv.is_left_controller_valid:
        #     print("Left controller matrix:", tv.left_controller_matrix)
        #     print("Left controller state:", tv.left_controller_state)

        if tv.is_right_controller_valid:
            print("Right controller matrix:", tv.right_controller_matrix)
            # right_state = tv.right_controller_state
            # if right_state:
            #     print(f"Right controller - Trigger: {right_state['triggerValue']:.2f}, Squeeze: {right_state['squeezeValue']:.2f}")
            #     # print(f"Right controller - Thumbstick: {right_state['thumbstickValue']}")

        time.sleep(1)
