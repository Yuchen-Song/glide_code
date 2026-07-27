from __future__ import annotations

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Pinocchio is required for PinkKinematics. Install with `pip install pin` or "
        "`conda install -c conda-forge pinocchio`."
    ) from exc

try:
    import pink
    from pink.tasks import DampingTask, FrameTask, PostureTask
    import qpsolvers
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Pink IK dependencies are missing. Install with `pip install pink qpsolvers` "
        "or `conda install -c conda-forge pink qpsolvers`."
    ) from exc


class PinkKinematics:
    def __init__(
        self,
        urdf_path: str,
        frame_name: str,
        dt: float = 0.02,
        alpha: float = 0.2,
        position_cost: float = 10.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-3,
        damping_cost: float = 1e-1,
        lm_damping: float = 1e-4,
        gain: float = 0.5,
        solver: str | None = None,
        solve_damping: float = 1e-12,
    ):
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._frame_name = frame_name
        self._frame_id = self._model.getFrameId(frame_name)
        if self._frame_id == len(self._model.frames):
            raise ValueError(f"Frame '{frame_name}' not found in URDF: {urdf_path}")

        self._dt = float(dt)
        self._alpha = float(alpha)
        self._solve_damping = float(solve_damping)

        self._q = pin.neutral(self._model)
        self._configuration = pink.Configuration(self._model, self._data, self._q)

        self._position_cost = float(position_cost)
        self._orientation_cost = float(orientation_cost)
        self._frame_task = FrameTask(
            frame_name,
            position_cost=self._position_cost,
            orientation_cost=self._orientation_cost,
            lm_damping=lm_damping,
            gain=gain,
        )
        self._posture_task = PostureTask(cost=posture_cost)
        self._damping_task = DampingTask(cost=damping_cost)
        self._tasks = [self._frame_task, self._posture_task, self._damping_task]
        for task in self._tasks:
            if hasattr(task, "set_target_from_configuration"):
                task.set_target_from_configuration(self._configuration)

        if solver is not None:
            self._solver = solver
        elif "proxqp" in qpsolvers.available_solvers:
            self._solver = "proxqp"
        elif qpsolvers.available_solvers:
            self._solver = qpsolvers.available_solvers[0]
        else:
            raise RuntimeError("No QP solver available. Install a qpsolver backend (e.g., proxqp).")

    def fk(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._frame_id].homogeneous

    def ik(
        self,
        target_pose: np.ndarray,
        init_q: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[bool, np.ndarray]:
        if init_q is not None:
            if len(init_q) != self._model.nq:
                raise ValueError(f"init_q has length {len(init_q)} but model expects {self._model.nq}")
            self._q = np.asarray(init_q, dtype=float).copy()

        dt = self._dt if dt is None else float(dt)
        self._configuration.update(self._q)
        self._frame_task.set_orientation_cost(self._orientation_cost)
        self._frame_task.set_target(pin.SE3(target_pose[:3, :3], target_pose[:3, 3]))

        try:
            v = pink.solve_ik(
                self._configuration,
                self._tasks,
                dt,
                solver=self._solver,
                damping=self._solve_damping,
            )
        except Exception:
            return False, self._q.copy()

        v_min = 0.5 * (self._model.lowerPositionLimit - self._q) / dt
        v_max = 0.5 * (self._model.upperPositionLimit - self._q) / dt
        v = np.clip(v, v_min, v_max)
        q_next = pin.integrate(self._model, self._q, v * dt)
        self._q = pin.interpolate(self._model, self._q, q_next, self._alpha)
        self._configuration.update(self._q)

        return True, self._q.copy()
