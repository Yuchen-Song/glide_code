"""Asynchronous CRAFT hand serial I/O helpers."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CraftIOStats:
    submitted: int
    writes: int
    skipped_targets: int
    write_failures: int
    read_failures: int
    write_hz: float
    last_write_age_ms: float | None
    last_target_age_ms: float | None
    running: bool


class AsyncCraftIO:
    """Write latest CRAFT targets on a background serial thread.

    The teleop loop can submit targets at arm-control rate without waiting for
    Dynamixel sync writes. The worker intentionally keeps only the newest target
    so slow serial I/O cannot build a backlog.
    """

    def __init__(
        self,
        *,
        craft,
        motor_ids: Sequence[int],
        command_hz: float,
        state_read_hz: float,
        initial_targets: Mapping[int, int],
    ) -> None:
        self._craft = craft
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._command_period = 1.0 / max(float(command_hz), 1e-6)
        self._read_period = 0.0 if state_read_hz <= 0.0 else 1.0 / float(state_read_hz)
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False

        self._latest_targets = self._copy_targets(initial_targets)
        self._present = self._latest_targets.copy()
        self._pending_generation = 0
        self._written_generation = 0
        self._last_submit_time: float | None = None
        self._last_write_time: float | None = None
        self._last_target_age_ms: float | None = None
        self._started_at: float | None = None
        self._submitted = 0
        self._writes = 0
        self._skipped_targets = 0
        self._write_failures = 0
        self._read_failures = 0

    def _copy_targets(self, targets: Mapping[int, int]) -> dict[int, int]:
        return {motor_id: int(targets[motor_id]) for motor_id in self._motor_ids}

    def start(self) -> "AsyncCraftIO":
        with self._condition:
            if self._running:
                return self
            self._running = True
            self._started_at = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="craft_async_io", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> bool:
        with self._condition:
            self._running = False
            self._condition.notify_all()
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        stopped = not thread.is_alive()
        if stopped:
            with self._condition:
                self._thread = None
        return stopped

    def submit(self, targets: Mapping[int, int], now: float | None = None) -> None:
        submitted_at = time.monotonic() if now is None else float(now)
        copied = self._copy_targets(targets)
        with self._condition:
            if self._pending_generation != self._written_generation:
                self._skipped_targets += 1
            self._latest_targets = copied
            self._last_submit_time = submitted_at
            self._submitted += 1
            self._pending_generation += 1
            if self._read_period <= 0.0:
                self._present.update(copied)
            self._condition.notify()

    def present_snapshot(self) -> dict[int, int]:
        with self._condition:
            return self._present.copy()

    def stats(self, now: float | None = None) -> CraftIOStats:
        current = time.monotonic() if now is None else float(now)
        with self._condition:
            elapsed = max(current - self._started_at, 1e-6) if self._started_at is not None else 1e-6
            last_write_age_ms = (
                None if self._last_write_time is None else max(0.0, (current - self._last_write_time) * 1000.0)
            )
            return CraftIOStats(
                submitted=self._submitted,
                writes=self._writes,
                skipped_targets=self._skipped_targets,
                write_failures=self._write_failures,
                read_failures=self._read_failures,
                write_hz=self._writes / elapsed,
                last_write_age_ms=last_write_age_ms,
                last_target_age_ms=self._last_target_age_ms,
                running=self._running,
            )

    def _run(self) -> None:
        next_write_at = 0.0
        next_read_at = 0.0
        while True:
            with self._condition:
                while self._running and self._pending_generation == self._written_generation:
                    self._condition.wait(timeout=0.05)
                if not self._running:
                    return

                now = time.monotonic()
                wait_s = next_write_at - now
                if wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue

                targets = self._latest_targets.copy()
                generation = self._pending_generation
                submitted_at = self._last_submit_time

            try:
                self._craft.write_raw(targets)
                wrote = True
            except Exception:
                logging.exception("craft_async_write_failed")
                wrote = False

            write_done_at = time.monotonic()
            with self._condition:
                if wrote:
                    self._writes += 1
                    self._written_generation = generation
                    self._last_write_time = write_done_at
                    if submitted_at is not None:
                        self._last_target_age_ms = max(0.0, (write_done_at - submitted_at) * 1000.0)
                    if self._read_period <= 0.0:
                        self._present.update(targets)
                else:
                    self._write_failures += 1
                next_write_at = write_done_at + self._command_period

            if self._read_period > 0.0 and write_done_at >= next_read_at:
                try:
                    present = self._craft.client.read_raw_positions(self._motor_ids, attempts=1)
                except Exception:
                    logging.exception("craft_async_state_read_failed")
                    with self._condition:
                        self._read_failures += 1
                else:
                    with self._condition:
                        self._present.update({motor_id: int(present[motor_id]) for motor_id in self._motor_ids})
                next_read_at = time.monotonic() + self._read_period
