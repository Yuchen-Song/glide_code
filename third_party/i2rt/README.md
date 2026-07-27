# Curated i2rt runtime

This directory is a runtime-only i2rt subset for the GLIDE experiments. Use
the four public launchers in the repository-level `scripts/` directory rather
than invoking files here directly.

Included components:

- `i2rt/`: YAM robot, CAN motor, gripper, and kinematics support.
- `glide_runtime/`: internal two-gripper teleoperation implementations.
- `i2rt-craft-hand/`: internal CRAFT-hand teleoperation implementations.
- `TeleVision/`: Quest/Vuer bridge used by two-gripper teleoperation.
- `i2rt-controller/`, `i2rt-hand/`, and `craft-hand/`: support imported by the
  CRAFT runtime.
- `third_party/open_television/`: hand-tracking preprocessing support.

Training, policy deployment servers, examples, tests, calibration output, and
unrelated command-line utilities from the source checkout are excluded.

Install this package in the robot environment with:

```bash
python -m pip install -e "third_party/i2rt[spacemouse]"
```

Quest/Vuer requires local TLS files as described in `TeleVision/README.md`.
