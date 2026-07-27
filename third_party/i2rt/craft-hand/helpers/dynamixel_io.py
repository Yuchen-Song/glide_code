"""Minimal Dynamixel output layer for the calibrated CRAFT hand."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

from .motor_config import (
    CRAFT_BAUDRATE,
    CRAFT_PORT,
    calibrated_equivalent_raw,
    clamp_targets_to_safe_limits,
    nearest_equivalent_raw,
    parse_motor_ids,
    raw_defaults,
)


PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4


def signed_to_unsigned(value: int, size: int) -> int:
    if value < 0:
        value = (1 << (8 * size)) + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    bit_size = 8 * size
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)
    return value


class DynamixelClient:
    def __init__(self, motor_ids: Sequence[int], port: str = CRAFT_PORT, baudrate: int = CRAFT_BAUDRATE) -> None:
        import dynamixel_sdk

        self.dxl = dynamixel_sdk
        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)
        self._sync_writers = {}
        self._sync_readers = {}

    @property
    def is_connected(self) -> bool:
        return bool(self.port_handler.is_open)

    def connect(self) -> None:
        if self.is_connected:
            return
        if not self.port_handler.openPort():
            raise OSError(f"Failed to open Dynamixel port {self.port_name}")
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise OSError(f"Failed to set Dynamixel baudrate {self.baudrate}")

    def close_port(self) -> None:
        if self.is_connected:
            self.port_handler.closePort()

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            return
        if disable_torque:
            try:
                self.set_torque(False, attempts=2)
            except Exception as exc:
                logging.warning("Failed to disable CRAFT torque during disconnect: %s", exc)
        self.close_port()

    def handle_packet_result(
        self, comm_result: int, dxl_error: int | None = None, dxl_id: int | None = None, context: str = ""
    ) -> bool:
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            prefix = f"[Motor ID: {dxl_id}] " if dxl_id is not None else ""
            logging.error("%s%s%s", f"{context}: " if context else "", prefix, error_message)
            return False
        return True

    def write_byte(self, motor_ids: Sequence[int], value: int, address: int) -> list[int]:
        if not self.is_connected:
            raise OSError("DynamixelClient.connect() must be called first")
        errored = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                address,
                int(value),
            )
            if not self.handle_packet_result(comm_result, dxl_error, motor_id, context="write_byte"):
                errored.append(motor_id)
        return errored

    def sync_write(self, motor_ids: Sequence[int], values: Sequence[int | float], address: int, size: int) -> None:
        if not self.is_connected:
            raise OSError("DynamixelClient.connect() must be called first")
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler,
                self.packet_handler,
                address,
                size,
            )
        writer = self._sync_writers[key]
        errored = []
        for motor_id, value in zip(motor_ids, values, strict=True):
            unsigned = signed_to_unsigned(int(value), size)
            if not writer.addParam(motor_id, unsigned.to_bytes(size, byteorder="little")):
                errored.append(motor_id)
        if errored:
            logging.error("Sync write addParam failed for IDs: %s", errored)
        comm_result = writer.txPacket()
        self.handle_packet_result(comm_result, context="sync_write")
        writer.clearParam()

    def set_torque(self, enabled: bool, attempts: int = 8, retry_interval: float = 0.1) -> None:
        remaining = list(self.motor_ids)
        state = "enabled" if enabled else "disabled"
        for attempt in range(1, attempts + 1):
            remaining = self.write_byte(remaining, int(enabled), ADDR_TORQUE_ENABLE)
            if not remaining:
                print(f"craft_torque_{state}=ok attempts={attempt}")
                return
            print(f"craft_torque_{state}_retry={attempt} remaining={remaining}")
            time.sleep(retry_interval)
        raise RuntimeError(f"Could not set CRAFT torque {state} for IDs after {attempts} attempts: {remaining}")

    def _read_raw_positions_individual(self, motor_ids: Sequence[int], attempts: int) -> dict[int, int]:
        raw: dict[int, int] = {}
        for motor_id in motor_ids:
            for attempt in range(1, attempts + 1):
                value, comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
                    self.port_handler,
                    motor_id,
                    ADDR_PRESENT_POSITION,
                )
                if self.handle_packet_result(comm_result, dxl_error, motor_id, context="read_present_position"):
                    raw[motor_id] = unsigned_to_signed(value, 4)
                    break
                print(f"craft_read_position_retry={attempt} motor_id={motor_id}")
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Could not read CRAFT present position for motor ID {motor_id}")
        return raw

    def _read_raw_positions_sync(self, motor_ids: Sequence[int], attempts: int) -> dict[int, int]:
        key = (ADDR_PRESENT_POSITION, LEN_GOAL_POSITION)
        if key not in self._sync_readers:
            self._sync_readers[key] = self.dxl.GroupSyncRead(
                self.port_handler,
                self.packet_handler,
                ADDR_PRESENT_POSITION,
                LEN_GOAL_POSITION,
            )
        reader = self._sync_readers[key]
        last_missing: list[int] = []
        for attempt in range(1, attempts + 1):
            reader.clearParam()
            add_failed = []
            for motor_id in motor_ids:
                if not reader.addParam(motor_id):
                    add_failed.append(motor_id)
            if add_failed:
                raise RuntimeError(f"Sync read addParam failed for IDs: {add_failed}")

            comm_result = reader.txRxPacket()
            if not self.handle_packet_result(comm_result, context="sync_read_present_position"):
                print(f"craft_sync_read_position_retry={attempt}")
                time.sleep(0.05)
                continue

            raw: dict[int, int] = {}
            missing = []
            for motor_id in motor_ids:
                if not reader.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_GOAL_POSITION):
                    missing.append(motor_id)
                    continue
                value = reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_GOAL_POSITION)
                raw[motor_id] = unsigned_to_signed(value, LEN_GOAL_POSITION)
            if not missing:
                return raw
            last_missing = missing
            print(f"craft_sync_read_position_retry={attempt} missing={missing}")
            time.sleep(0.05)
        raise RuntimeError(f"Could not sync read CRAFT present position for motor IDs: {last_missing}")

    def read_raw_positions(
        self,
        motor_ids: Sequence[int] | None = None,
        attempts: int = 5,
        *,
        sync: bool = True,
    ) -> dict[int, int]:
        if not self.is_connected:
            raise OSError("DynamixelClient.connect() must be called first")
        ids = list(self.motor_ids if motor_ids is None else motor_ids)
        if not sync:
            return self._read_raw_positions_individual(ids, attempts)
        try:
            return self._read_raw_positions_sync(ids, attempts)
        except Exception as exc:
            print(f"craft_sync_read_position_fallback={type(exc).__name__}: {exc}")
            return self._read_raw_positions_individual(ids, attempts)


class CraftHandOutput:
    def __init__(
        self,
        motor_ids: list[int],
        port: str = CRAFT_PORT,
        baudrate: int = CRAFT_BAUDRATE,
        current_limit: int = 130,
        p_gain: int = 650,
        i_gain: int = 60,
        d_gain: int = 220,
        start_default: bool = True,
        end_default: bool = True,
        default_ramp_seconds: float = 4.0,
        default_ramp_hz: float = 25.0,
        disable_torque_on_exit: bool = True,
    ) -> None:
        self.motor_ids = motor_ids
        self.client = DynamixelClient(motor_ids, port=port, baudrate=baudrate)
        self.current_limit = current_limit
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.start_default = start_default
        self.end_default = end_default
        self.default_ramp_seconds = max(0.0, default_ramp_seconds)
        self.default_ramp_hz = max(1.0, default_ramp_hz)
        self.disable_torque_on_exit = disable_torque_on_exit
        self.default_targets = {motor_id: raw_defaults()[motor_id] for motor_id in motor_ids}
        self.default_branch_targets = self.default_targets.copy()
        self.last_targets = self.default_targets.copy()

    @classmethod
    def from_args(cls, args) -> "CraftHandOutput":
        return cls(
            motor_ids=parse_motor_ids(args.craft_motors),
            port=args.craft_port,
            baudrate=args.craft_baudrate,
            current_limit=args.current_limit,
            p_gain=args.p_gain,
            i_gain=args.i_gain,
            d_gain=args.d_gain,
            start_default=args.start_default,
            end_default=args.end_default,
            default_ramp_seconds=args.default_ramp_seconds,
            default_ramp_hz=args.default_ramp_hz,
            disable_torque_on_exit=not args.hold_torque_on_exit,
        )

    def __enter__(self) -> "CraftHandOutput":
        self.client.connect()
        print(f"craft_connected={self.client.port_name} baud={self.client.baudrate}")
        print("craft_motor_ids=" + ",".join(str(motor_id) for motor_id in self.motor_ids))
        self.client.set_torque(False)
        time.sleep(0.15)
        self.client.sync_write(self.motor_ids, np.ones(len(self.motor_ids)) * 5, ADDR_OPERATING_MODE, 1)
        self.client.sync_write(self.motor_ids, np.ones(len(self.motor_ids)) * self.p_gain, ADDR_POSITION_P_GAIN, 2)
        self.client.sync_write(self.motor_ids, np.ones(len(self.motor_ids)) * self.i_gain, ADDR_POSITION_I_GAIN, 2)
        self.client.sync_write(self.motor_ids, np.ones(len(self.motor_ids)) * self.d_gain, ADDR_POSITION_D_GAIN, 2)
        self.client.sync_write(self.motor_ids, np.ones(len(self.motor_ids)) * self.current_limit, ADDR_GOAL_CURRENT, 2)
        try:
            present = self.client.read_raw_positions(self.motor_ids)
            self.last_targets = present
            print("craft_present_start=" + " ".join(f"{motor_id}:{present[motor_id]}" for motor_id in self.motor_ids))
        except Exception as exc:
            print(f"craft_present_start=defaults reason={type(exc).__name__}: {exc}")
            self.last_targets = self.default_targets.copy()
        self._update_default_branches(self.last_targets)
        self._sync_write_raw_targets(self.last_targets)
        self.client.set_torque(True)
        time.sleep(0.15)
        if self.start_default:
            self.move_to_defaults(label="start_default")
        return self

    def _sync_write_raw_targets(self, targets: dict[int, int]) -> None:
        self.client.sync_write(
            self.motor_ids,
            [targets[motor_id] for motor_id in self.motor_ids],
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self.last_targets = targets.copy()

    def _update_default_branches(self, reference: dict[int, int]) -> None:
        self.default_branch_targets = {
            motor_id: nearest_equivalent_raw(self.default_targets[motor_id], reference[motor_id])
            for motor_id in self.motor_ids
        }

    def write_raw(self, targets: dict[int, int]) -> None:
        safe_targets = clamp_targets_to_safe_limits(targets, self.motor_ids)
        command_targets = {
            motor_id: calibrated_equivalent_raw(
                motor_id,
                safe_targets[motor_id],
                self.last_targets[motor_id],
                self.default_branch_targets[motor_id],
            )
            for motor_id in self.motor_ids
        }
        self._sync_write_raw_targets(command_targets)

    def ramp_to_raw(self, targets: dict[int, int], label: str, seconds: float | None = None) -> None:
        start = self.last_targets.copy()
        clamped = clamp_targets_to_safe_limits(targets, self.motor_ids)
        target_values = {
            motor_id: calibrated_equivalent_raw(
                motor_id,
                clamped[motor_id],
                start[motor_id],
                self.default_branch_targets[motor_id],
            )
            for motor_id in self.motor_ids
        }
        duration = self.default_ramp_seconds if seconds is None else max(0.0, seconds)
        steps = max(1, int(round(duration * self.default_ramp_hz)))
        sleep_s = duration / steps if duration > 0 else 0.0
        print(
            f"craft_{label}_ramp=start seconds={duration:.2f} steps={steps} "
            + "targets="
            + " ".join(f"{motor_id}:{target_values[motor_id]}" for motor_id in self.motor_ids)
        )
        for step in range(1, steps + 1):
            fraction = step / steps
            command = {
                motor_id: int(round(start[motor_id] + fraction * (target_values[motor_id] - start[motor_id])))
                for motor_id in self.motor_ids
            }
            self._sync_write_raw_targets(command)
            if sleep_s:
                time.sleep(sleep_s)
        self._sync_write_raw_targets(target_values)
        print(f"craft_{label}_ramp=done")

    def move_to_defaults(self, label: str = "default") -> None:
        self.ramp_to_raw(self.default_targets, label=label)

    def __exit__(self, *exc: object) -> None:
        try:
            if self.end_default:
                self.move_to_defaults(label="end_default")
        finally:
            self.client.disconnect(disable_torque=self.disable_torque_on_exit)
            if self.disable_torque_on_exit:
                print("craft_torque=disabled_on_exit")
            else:
                print("craft_torque=holding_on_exit")


def add_craft_output_args(parser) -> None:
    parser.add_argument("--craft-port", default=CRAFT_PORT)
    parser.add_argument("--craft-baudrate", type=int, default=CRAFT_BAUDRATE)
    parser.add_argument(
        "--craft-motors", default="0-14", help="CRAFT motor IDs to command, e.g. all, 0-14, or 3,7,14."
    )
    parser.add_argument("--current-limit", type=int, default=130)
    parser.add_argument("--p-gain", type=int, default=650)
    parser.add_argument("--i-gain", type=int, default=60)
    parser.add_argument("--d-gain", type=int, default=220)
    parser.add_argument("--default-ramp-seconds", type=float, default=4.0)
    parser.add_argument("--default-ramp-hz", type=float, default=25.0)
    parser.add_argument("--start-default", dest="start_default", action="store_true", default=True)
    parser.add_argument("--no-start-default", dest="start_default", action="store_false")
    parser.add_argument("--end-default", dest="end_default", action="store_true", default=True)
    parser.add_argument("--no-end-default", dest="end_default", action="store_false")
    parser.add_argument("--hold-torque-on-exit", action="store_true")
