import argparse
import os
import signal
import time
from multiprocessing import Event, Queue, shared_memory

import numpy as np

from TeleVision import OpenTeleVision
from constants_vuer import grd_yup2grd_zup
from motion_utils import fast_mat_inv
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.pink_kinematics import PinkKinematics
from i2rt.robots.utils import GripperType, I2RT_ROOT


CONTROL_FREQUENCY = 30.0


class VuerControllerTeleop:
    def __init__(self, resolution=(720, 1280), image_background=True, ngrok=False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cert_file = os.path.join(script_dir, "cert.pem")
        key_file = os.path.join(script_dir, "key.pem")
        self.resolution = resolution
        self.img_shape = (self.resolution[0], 2 * self.resolution[1], 3)
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.img_array[:] = 0

        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution,
            self.shm.name,
            image_queue,
            toggle_streaming,
            cert_file=cert_file,
            key_file=key_file,
            ngrok=ngrok,
            image_background=image_background,
        )

    def get_controller_matrix(self, side: str) -> np.ndarray | None:
        if side == "left":
            if not self.tv.is_left_controller_valid:
                return None
            mat = self.tv.left_controller_matrix
        else:
            if not self.tv.is_right_controller_valid:
                return None
            mat = self.tv.right_controller_matrix
        if mat is None:
            return None
        if np.linalg.det(mat[:3, :3]) == 0:
            return None
        return mat

    def cleanup(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
        if self.tv is not None and hasattr(self.tv, "process"):
            if self.tv.process.is_alive():
                self.tv.process.terminate()
                self.tv.process.join()


def ensure_can_interface_ready(channel: str) -> None:
    sysfs_path = f"/sys/class/net/{channel}"
    if not os.path.exists(sysfs_path):
        raise RuntimeError(
            f"CAN interface '{channel}' not found. Check `ip link show` and pass --left-channel/--right-channel."
        )
    operstate_path = os.path.join(sysfs_path, "operstate")
    try:
        with open(operstate_path, "r", encoding="utf-8") as handle:
            state = handle.read().strip()
    except OSError:
        return
    if state == "down":
        raise RuntimeError(
            f"CAN interface '{channel}' is down. Bring it up before running (e.g., "
            f"`sudo ip link set {channel} up type can bitrate 1000000`)."
        )


def vuer_to_robot_matrix(mat: np.ndarray) -> np.ndarray:
    return grd_yup2grd_zup @ mat @ fast_mat_inv(grd_yup2grd_zup)


def build_command(arm_q: np.ndarray, gripper_value: float, arm_dofs: int, gripper_index: int | None, total_dofs: int) -> np.ndarray:
    if gripper_index is None:
        return arm_q
    cmd = np.zeros(total_dofs)
    cmd[:arm_dofs] = arm_q
    cmd[gripper_index] = gripper_value
    return cmd


def setup_arm(
    channel: str,
    gripper_name: str,
    ik_frame: str,
    gripper_invert: bool,
    ik_dt: float,
    ik_alpha: float,
    ik_pos_cost: float,
    ik_ori_cost: float,
    ik_posture_cost: float,
    ik_damping_cost: float,
    ik_lm_damping: float,
    ik_gain: float,
    ik_solver: str | None,
    ik_solve_damping: float,
) -> dict:
    gripper_type = GripperType.from_string_name(gripper_name)
    robot = get_yam_robot(channel=channel, gripper_type=gripper_type)
    robot_info = robot.get_robot_info()
    gripper_index = robot_info.get("gripper_index")
    gripper_limits = robot_info.get("gripper_limits")

    arm_dofs = gripper_index if gripper_index is not None else robot.num_dofs()
    urdf_path = os.path.join(I2RT_ROOT, "robot_models", "yam", "yam.urdf")
    kin = PinkKinematics(
        urdf_path,
        ik_frame,
        dt=ik_dt,
        alpha=ik_alpha,
        position_cost=ik_pos_cost,
        orientation_cost=ik_ori_cost,
        posture_cost=ik_posture_cost,
        damping_cost=ik_damping_cost,
        lm_damping=ik_lm_damping,
        gain=ik_gain,
        solver=ik_solver,
        solve_damping=ik_solve_damping,
    )

    current_q = robot.get_joint_pos()
    target_q = np.copy(current_q[:arm_dofs])
    target_pose = kin.fk(target_q)

    if gripper_index is not None:
        gripper_pos = float(current_q[gripper_index])
        gripper_open = 1.0
        gripper_close = 0.0
        if gripper_invert:
            gripper_open, gripper_close = gripper_close, gripper_open
    else:
        gripper_pos = 0.0
        gripper_open = 0.0
        gripper_close = 0.0

    if gripper_limits is None:
        home_gripper = gripper_pos
    else:
        home_gripper = gripper_open

    return {
        "robot": robot,
        "kin": kin,
        "arm_dofs": arm_dofs,
        "gripper_index": gripper_index,
        "gripper_limits": gripper_limits,
        "target_q": target_q,
        "target_pose": target_pose,
        # "Virtual" state: what the arm would be doing without the EEF clamp.
        # Advanced every tick by running IK on the unclamped pose, so seeding
        # the real IK with this gives instant re-entry when the controller
        # returns to the reachable workspace — no alpha-filter catch-up lag.
        "virtual_target_q": np.copy(target_q),
        "virtual_target_pose": target_pose.copy(),
        "gripper_pos": gripper_pos,
        "gripper_goal": gripper_pos,
        "gripper_blocked": False,
        "gripper_open": gripper_open,
        "gripper_close": gripper_close,
        "home_arm": np.zeros(arm_dofs),
        "home_gripper": home_gripper,
    }


def update_gripper_from_controller(arm_state: dict, controller_state: dict | None, mode: str, invert: bool) -> None:
    if controller_state is None or mode == "none":
        return
    if mode == "trigger":
        value = 1.0 - float(controller_state.get("triggerValue", 0.0))
    else:
        value = 1.0 - float(controller_state.get("squeezeValue", 0.0))
    if invert:
        value = 1.0 - value

    gripper_limits = arm_state["gripper_limits"]
    if gripper_limits is None:
        goal = value
    else:
        goal = float(np.clip(value, 0.0, 1.0))

    arm_state["gripper_goal"] = goal
    if arm_state.get("gripper_blocked"):
        open_pos = arm_state["gripper_open"]
        close_pos = arm_state["gripper_close"]
        current_pos = arm_state.get("gripper_pos", goal)
        closing = (goal - current_pos) * (close_pos - open_pos) > 0
        if closing:
            return
        arm_state["gripper_blocked"] = False
    arm_state["gripper_pos"] = goal


def maybe_limit_gripper_close(
    arm_state: dict,
    force_threshold: float,
    force_ema_alpha: float,
    backoff: float | None,
) -> tuple[bool, float | None, float | None, float | None]:
    if force_threshold is None or force_threshold <= 0.0:
        return False, None, None, arm_state.get("gripper_goal", arm_state.get("gripper_pos"))

    gripper_index = arm_state.get("gripper_index")
    if gripper_index is None:
        return False, None, None, None

    robot = arm_state["robot"]
    obs = robot.get_observations()
    if obs is None:
        return False, None, None, None

    joint_eff = obs.get("joint_eff")
    if joint_eff is None or len(joint_eff) <= gripper_index:
        return False, None, None, None
    eff = float(joint_eff[gripper_index])
    if force_ema_alpha is not None and force_ema_alpha > 0.0:
        ema_key = "gripper_eff_ema"
        prev_ema = arm_state.get(ema_key)
        if prev_ema is None:
            filtered_eff = eff
        else:
            filtered_eff = force_ema_alpha * eff + (1.0 - force_ema_alpha) * prev_ema
        arm_state[ema_key] = filtered_eff
        eff = filtered_eff

    current_pos = None
    gripper_pos_obs = obs.get("gripper_pos")
    if gripper_pos_obs is not None and len(gripper_pos_obs) > 0:
        current_pos = float(gripper_pos_obs[0])
    else:
        current_q = robot.get_joint_pos()
        if len(current_q) > gripper_index:
            current_pos = float(current_q[gripper_index])

    if current_pos is None:
        return False, eff, None, arm_state["gripper_pos"]

    target_pos = arm_state.get("gripper_goal", arm_state["gripper_pos"])
    close_pos = arm_state["gripper_close"]
    open_pos = arm_state["gripper_open"]
    closing = (target_pos - current_pos) * (close_pos - open_pos) > 0

    if force_threshold is not None and force_threshold > 0.0 and closing and eff >= force_threshold:
        already_blocked = arm_state.get("gripper_blocked", False)
        arm_state["gripper_blocked"] = True
        if not already_blocked:
            backoff_amount = backoff if backoff is not None else 0.0
            if backoff_amount > 0.0:
                backoff_dir = np.sign(open_pos - close_pos)
                backoff_pos = current_pos + backoff_dir * backoff_amount
                lower = min(open_pos, close_pos)
                upper = max(open_pos, close_pos)
                arm_state["gripper_pos"] = float(np.clip(backoff_pos, lower, upper))
            else:
                arm_state["gripper_pos"] = current_pos
        return True, eff, current_pos, target_pos
    return False, eff, current_pos, target_pos


def _scale_rotation_delta(delta_rot: np.ndarray, scale: float) -> np.ndarray:
    """Scale a controller rotation while preserving the CRAFT-hand API."""
    if scale == 1.0:
        return delta_rot
    if scale == 0.0:
        return np.eye(3)

    cos_theta = float(np.clip((np.trace(delta_rot) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.eye(3)

    if np.pi - theta < 1e-4:
        diag = np.diag(delta_rot)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[0] = np.copysign(axis[0], delta_rot[2, 1] - delta_rot[1, 2])
        elif axis[1] >= axis[0] and axis[1] >= axis[2]:
            axis[1] = np.copysign(axis[1], delta_rot[0, 2] - delta_rot[2, 0])
        else:
            axis[2] = np.copysign(axis[2], delta_rot[1, 0] - delta_rot[0, 1])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return np.eye(3)
        axis = axis / axis_norm
    else:
        axis = np.array(
            [
                delta_rot[2, 1] - delta_rot[1, 2],
                delta_rot[0, 2] - delta_rot[2, 0],
                delta_rot[1, 0] - delta_rot[0, 1],
            ]
        ) / (2.0 * np.sin(theta))

    scaled_theta = theta * scale
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(scaled_theta) * k + (1.0 - np.cos(scaled_theta)) * (k @ k)


def compute_target_pose(
    init_pose: np.ndarray,
    init_controller: np.ndarray,
    current_controller: np.ndarray,
    pos_scale: float,
    lock_orientation: bool,
    rot_scale: float = 1.0,
) -> np.ndarray:
    target_pose = init_pose.copy()
    delta_pos = (current_controller[:3, 3] - init_controller[:3, 3]) * pos_scale
    target_pose[:3, 3] = init_pose[:3, 3] + delta_pos

    if not lock_orientation:
        delta_rot = current_controller[:3, :3] @ init_controller[:3, :3].T
        target_pose[:3, :3] = _scale_rotation_delta(delta_rot, rot_scale) @ init_pose[:3, :3]
    return target_pose


def clamp_eef_pose(
    pose: np.ndarray,
    max_radius: float | None = None,
    min_z: float | None = None,
    min_x: float | None = None,
) -> np.ndarray:
    """Clamp an EEF target pose into a safe workspace envelope.

    The envelope is a ball around the arm's own base origin (0,0,0 in the
    URDF base frame) intersected with optional axis-aligned half-spaces.
    Measured on this robot (see eef_traj_20260416_113928.npz):
        observed r_max ~= 0.75 m, observed min z ~= 0.05 m.
    Defaults here are chosen below those bounds so IK never chases poses
    the arm physically can't hold without stalling motors.

    The position is clamped in place on a copy; orientation is untouched.
    Pass None for any constraint to disable it.
    """
    out = pose.copy()
    x, y, z = out[0, 3], out[1, 3], out[2, 3]

    if min_z is not None and z < min_z:
        z = min_z
    if min_x is not None and x < min_x:
        x = min_x

    if max_radius is not None:
        r = float(np.sqrt(x * x + y * y + z * z))
        if r > max_radius and r > 0.0:
            scale = max_radius / r
            x, y, z = x * scale, y * scale, z * scale

    out[0, 3], out[1, 3], out[2, 3] = x, y, z
    return out


def reset_to_home(arm_state: dict, time_interval_s: float) -> None:
    robot = arm_state["robot"]
    cmd = build_command(
        arm_state["home_arm"],
        arm_state["home_gripper"],
        arm_state["arm_dofs"],
        arm_state["gripper_index"],
        robot.num_dofs(),
    )
    robot.move_joints(cmd, time_interval_s=time_interval_s)


def move_to_ready_pose(arm_state: dict, ready_qpos: np.ndarray, time_interval_s: float) -> None:
    robot = arm_state["robot"]
    ready_qpos = np.asarray(ready_qpos, dtype=float)
    if len(ready_qpos) != robot.num_dofs():
        if len(ready_qpos) > robot.num_dofs():
            ready_qpos = ready_qpos[: robot.num_dofs()]
        else:
            pad = robot.num_dofs() - len(ready_qpos)
            ready_qpos = np.concatenate([ready_qpos, np.zeros(pad)])
    robot.move_joints(ready_qpos, time_interval_s=time_interval_s)


def sync_arm_state_from_robot(arm_state: dict) -> None:
    robot = arm_state["robot"]
    current_q = robot.get_joint_pos()
    arm_state["target_q"] = np.copy(current_q[: arm_state["arm_dofs"]])
    arm_state["target_pose"] = arm_state["kin"].fk(arm_state["target_q"])
    # Re-seat the virtual tracker on the current physical state so the next
    # unclamped IK solve doesn't start from a stale pre-reset guess.
    arm_state["virtual_target_q"] = np.copy(arm_state["target_q"])
    arm_state["virtual_target_pose"] = arm_state["target_pose"].copy()
    gripper_index = arm_state["gripper_index"]
    if gripper_index is not None:
        arm_state["gripper_pos"] = float(current_q[gripper_index])
        arm_state["gripper_goal"] = arm_state["gripper_pos"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-channel", type=str, default="can1")
    parser.add_argument("--right-channel", type=str, default="can0")
    parser.add_argument("--left-gripper", type=str, default="linear_4310")
    parser.add_argument("--right-gripper", type=str, default="linear_4310")
    parser.add_argument("--left-gripper-invert", action="store_true")
    parser.add_argument("--right-gripper-invert", action="store_true")
    parser.add_argument("--gripper-mode", type=str, choices=["none", "trigger", "squeeze"], default="trigger")
    parser.add_argument("--gripper-force-threshold", type=float, default=0.36)
    parser.add_argument("--gripper-force-verbose", action="store_true")
    parser.add_argument("--gripper-force-print-interval", type=float, default=0.5)
    parser.add_argument("--gripper-force-ema-alpha", type=float, default=1)
    parser.add_argument("--gripper-backoff", type=float, default=0.05)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--lock-orientation", action="store_true")
    parser.add_argument("--frequency", type=float, default=CONTROL_FREQUENCY)
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
    parser.add_argument(
        "--eef-max-radius",
        type=float,
        default=0.695,
        help="Max distance (m) from arm base origin to allowed EEF target. "
             "Default = 95%% of observed max (min(left r_max, right r_max) * 0.95 "
             "from eef_traj_20260416_113928.npz). Set 0 or negative to disable.",
    )
    parser.add_argument(
        "--eef-min-z",
        type=float,
        default=0.05,
        help="Minimum EEF z (m) in arm base frame to avoid hitting the base plate. "
             "Set to a large negative number to disable.",
    )
    parser.add_argument(
        "--eef-min-x",
        type=float,
        default=None,
        help="Minimum EEF x (m). Omit to disable (default).",
    )
    args = parser.parse_args()
    eef_max_r = args.eef_max_radius if args.eef_max_radius and args.eef_max_radius > 0 else None
    eef_min_z = args.eef_min_z if args.eef_min_z is not None else None
    eef_min_x = args.eef_min_x

    teleop = None
    left_arm = None
    right_arm = None

    try:
        ensure_can_interface_ready(args.left_channel)
        ensure_can_interface_ready(args.right_channel)
        ik_frame = args.site or args.ik_frame
        ik_dt = args.ik_dt if args.ik_dt is not None else 1.0 / args.frequency
        left_arm = setup_arm(
            args.left_channel,
            args.left_gripper,
            ik_frame,
            args.left_gripper_invert,
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
        right_arm = setup_arm(
            args.right_channel,
            args.right_gripper,
            ik_frame,
            args.right_gripper_invert,
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
        teleop = VuerControllerTeleop()

        print("Moving both arms to ready pose...")
        move_to_ready_pose(left_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
        move_to_ready_pose(right_arm, np.array(args.ready_qpos, dtype=float), args.home_time)
        sync_arm_state_from_robot(left_arm)
        sync_arm_state_from_robot(right_arm)
        print("Ready pose reached. Press A on right controller to start teleop.")
    except Exception:
        if teleop is not None:
            teleop.cleanup()
        if left_arm is not None:
            left_arm["robot"].close()
        if right_arm is not None:
            right_arm["robot"].close()
        raise

    left_init_controller = None
    right_init_controller = None
    left_init_pose = None
    right_init_pose = None

    print("Waiting for both controllers to be valid...")
    last_warn_time = 0.0
    last_b_pressed = False
    a_pressed_once = False
    last_left_gripper_warn_time = 0.0
    last_right_gripper_warn_time = 0.0
    last_left_gripper_print_time = 0.0
    last_right_gripper_print_time = 0.0

    try:
        while True:
            right_state = None
            if teleop is not None and teleop.tv is not None:
                right_state = teleop.tv.right_controller_state
            b_pressed = bool(right_state.get("bButton")) if right_state else False
            if b_pressed and not last_b_pressed:
                raise KeyboardInterrupt
            last_b_pressed = b_pressed
            if not a_pressed_once:
                a_pressed = bool(right_state.get("aButton")) if right_state else False
                if not a_pressed:
                    time.sleep(0.01)
                    continue
                a_pressed_once = True
                print("A pressed. Teleop enabled.")

            left_mat = teleop.get_controller_matrix("left")
            right_mat = teleop.get_controller_matrix("right")
            if left_mat is None or right_mat is None:
                time.sleep(0.01)
                continue

            left_mat = vuer_to_robot_matrix(left_mat)
            right_mat = vuer_to_robot_matrix(right_mat)

            if left_init_controller is None or right_init_controller is None:
                left_init_controller = left_mat.copy()
                right_init_controller = right_mat.copy()
                left_init_pose = left_arm["target_pose"].copy()
                right_init_pose = right_arm["target_pose"].copy()
                print("Controller reference set. Starting teleop.")
                time.sleep(0.2)
                continue

            if args.gripper_mode != "none":
                update_gripper_from_controller(
                    left_arm,
                    teleop.tv.left_controller_state,
                    args.gripper_mode,
                    args.left_gripper_invert,
                )
                update_gripper_from_controller(
                    right_arm,
                    teleop.tv.right_controller_state,
                    args.gripper_mode,
                    args.right_gripper_invert,
                )

            left_gripper_blocked, left_eff, left_gripper_pos, left_gripper_goal = maybe_limit_gripper_close(
                left_arm,
                args.gripper_force_threshold,
                args.gripper_force_ema_alpha,
                args.gripper_backoff,
            )
            right_gripper_blocked, right_eff, right_gripper_pos, right_gripper_goal = maybe_limit_gripper_close(
                right_arm,
                args.gripper_force_threshold,
                args.gripper_force_ema_alpha,
                args.gripper_backoff,
            )
            if args.gripper_force_verbose:
                now = time.monotonic()
                if left_eff is not None and now - last_left_gripper_print_time > args.gripper_force_print_interval:
                    left_pos_label = f"{left_gripper_pos:.4f}" if left_gripper_pos is not None else "n/a"
                    left_goal_label = f"{left_gripper_goal:.4f}" if left_gripper_goal is not None else "n/a"
                    print(
                        "Left gripper eff={:.3f} pos={} target={}".format(
                            left_eff,
                            left_pos_label,
                            left_goal_label,
                        )
                    )
                    last_left_gripper_print_time = now
                if right_eff is not None and now - last_right_gripper_print_time > args.gripper_force_print_interval:
                    right_pos_label = f"{right_gripper_pos:.4f}" if right_gripper_pos is not None else "n/a"
                    right_goal_label = f"{right_gripper_goal:.4f}" if right_gripper_goal is not None else "n/a"
                    print(
                        "Right gripper eff={:.3f} pos={} target={}".format(
                            right_eff,
                            right_pos_label,
                            right_goal_label,
                        )
                    )
                    last_right_gripper_print_time = now
            if left_gripper_blocked:
                now = time.monotonic()
                if now - last_left_gripper_warn_time > 1.0:
                    print(f"Left gripper force threshold hit ({left_eff:.2f}); holding position.")
                    last_left_gripper_warn_time = now
            if right_gripper_blocked:
                now = time.monotonic()
                if now - last_right_gripper_warn_time > 1.0:
                    print(f"Right gripper force threshold hit ({right_eff:.2f}); holding position.")
                    last_right_gripper_warn_time = now

            left_virtual_pose = compute_target_pose(
                left_init_pose,
                left_init_controller,
                left_mat,
                args.pos_scale,
                args.lock_orientation,
            )
            right_virtual_pose = compute_target_pose(
                right_init_pose,
                right_init_controller,
                right_mat,
                args.pos_scale,
                args.lock_orientation,
            )
            left_target_pose = clamp_eef_pose(left_virtual_pose, eef_max_r, eef_min_z, eef_min_x)
            right_target_pose = clamp_eef_pose(right_virtual_pose, eef_max_r, eef_min_z, eef_min_x)

            # Advance the virtual (unclamped) IK trackers so they follow the
            # controller even while the real robot is pinned at the clamp
            # boundary. Seeding the real IK below with the virtual q means that
            # when the user returns to the reachable workspace the robot picks
            # up tracking immediately instead of lagging through the alpha
            # filter.
            left_virt_ok, left_virt_q = left_arm["kin"].ik(
                left_virtual_pose, init_q=left_arm["virtual_target_q"]
            )
            if left_virt_ok:
                left_arm["virtual_target_q"] = left_virt_q
                left_arm["virtual_target_pose"] = left_virtual_pose
            right_virt_ok, right_virt_q = right_arm["kin"].ik(
                right_virtual_pose, init_q=right_arm["virtual_target_q"]
            )
            if right_virt_ok:
                right_arm["virtual_target_q"] = right_virt_q
                right_arm["virtual_target_pose"] = right_virtual_pose

            left_success, left_q = left_arm["kin"].ik(
                left_target_pose, init_q=left_arm["virtual_target_q"]
            )
            if left_success:
                left_arm["target_q"] = left_q
                left_arm["target_pose"] = left_target_pose
                cmd = build_command(
                    left_q,
                    left_arm["gripper_pos"],
                    left_arm["arm_dofs"],
                    left_arm["gripper_index"],
                    left_arm["robot"].num_dofs(),
                )
                left_arm["robot"].command_joint_pos(cmd)
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Left arm IK failed; holding last command.")
                    last_warn_time = now

            right_success, right_q = right_arm["kin"].ik(
                right_target_pose, init_q=right_arm["virtual_target_q"]
            )
            if right_success:
                right_arm["target_q"] = right_q
                right_arm["target_pose"] = right_target_pose
                cmd = build_command(
                    right_q,
                    right_arm["gripper_pos"],
                    right_arm["arm_dofs"],
                    right_arm["gripper_index"],
                    right_arm["robot"].num_dofs(),
                )
                right_arm["robot"].command_joint_pos(cmd)
            else:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print("Right arm IK failed; holding last command.")
                    last_warn_time = now

            time.sleep(1.0 / args.frequency)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Returning both arms to home pose...")
    finally:
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if left_arm is not None:
                reset_to_home(left_arm, args.home_time)
            if right_arm is not None:
                reset_to_home(right_arm, args.home_time)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if left_arm is not None:
                left_arm["robot"].close()
            if right_arm is not None:
                right_arm["robot"].close()
            if teleop is not None:
                teleop.cleanup()
            print("Teleop shutdown complete.")


if __name__ == "__main__":
    main()
