"""Serialized SDK execution and trace support for coordinated Expert runs."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

DEVICE_SWITCH_QUIET_S = 0.1


class ExecutionCoordinator:
    def __init__(self, trace_path: Path | None = None, *, quiet_s: float = DEVICE_SWITCH_QUIET_S) -> None:
        self.lock = threading.Lock()
        self.trace_path = trace_path
        self.quiet_s = float(quiet_s)
        self.active_device: str | None = None
        self.last_operation_end_ns: int | None = None
        self.records: list[dict[str, Any]] = []

    def _trace(self, event: str, **fields: Any) -> None:
        row = {"event": event, "monotonic_ns": time.monotonic_ns(), **fields}
        self.records.append(row)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def call(self, *, phase: str, device: str, operation: str, fn: Callable[[], Any]) -> Any:
        with self.lock:
            if self.active_device is not None and self.active_device != device:
                self._trace("device_switch", phase=phase, device=device, operation=operation,
                            previous_device=self.active_device)
                print(f"[EXEC_COORD] DEVICE_SWITCH {self.active_device} -> {device}", flush=True)
                time.sleep(self.quiet_s)
            self.active_device = device
            start = time.monotonic_ns()
            self._trace("sdk_call_start", phase=phase, device=device, operation=operation)
            result = None
            exception = None
            try:
                result = fn()
                return result
            except BaseException as exc:
                exception = repr(exc)
                raise
            finally:
                end = time.monotonic_ns()
                duration_ms = (end - start) / 1e6
                self._trace("sdk_call_end", phase=phase, device=device, operation=operation,
                            duration_ms=duration_ms, result=result, exception=exception)
                if duration_ms > 500:
                    print(f"[EXEC_COORD] SLOW_CALL device={device} op={operation} duration={duration_ms:.1f}ms", flush=True)
                self.last_operation_end_ns = end

    def phase_start(self, phase: str, device: str | None = None) -> None:
        self._trace("phase_start", phase=phase, device=device)
        print(f"[EXEC_COORD] PHASE_START {phase}", flush=True)

    def phase_end(self, phase: str, elapsed_s: float, device: str | None = None) -> None:
        self._trace("phase_end", phase=phase, device=device, duration_ms=elapsed_s * 1000)
        print(f"[EXEC_COORD] PHASE_END {phase} elapsed={elapsed_s:.3f}", flush=True)

    def summary(self) -> dict[str, Any]:
        timestamps = [int(r["monotonic_ns"]) for r in self.records]
        calls = [r for r in self.records if r["event"] == "sdk_call_end"]
        by_device: dict[str, dict[str, Any]] = defaultdict(lambda: {"call_count": 0, "total_sdk_time_ms": 0.0, "max_sdk_call_ms": 0.0})
        by_op: dict[str, list[float]] = defaultdict(list)
        for row in calls:
            d = by_device[row["device"]]; d["call_count"] += 1; d["total_sdk_time_ms"] += row["duration_ms"]; d["max_sdk_call_ms"] = max(d["max_sdk_call_ms"], row["duration_ms"])
            by_op[row["operation"]].append(row["duration_ms"])
        return {"total_duration_s": ((max(timestamps) - min(timestamps)) / 1e9 if timestamps else 0.0), "phase_count": len([r for r in self.records if r["event"] == "phase_start"]), "sdk_call_count": len(calls), "devices": dict(by_device), "operations": {k: {"call_count": len(v), "mean_ms": sum(v)/len(v), "max_ms": max(v)} for k, v in by_op.items()}, "slow_calls": [r for r in calls if r["duration_ms"] > 500], "device_switch_count": len([r for r in self.records if r["event"] == "device_switch"]), "motion_timeout_count": len([r for r in self.records if r["event"] == "timeout"]), "sdk_error_count": len([r for r in calls if r.get("exception")])}

    def write_summary(self, path: Path) -> None:
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
