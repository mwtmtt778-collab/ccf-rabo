#!/usr/bin/env python3
"""Ready-gated, fail-stop three-Nut Expert V2.

V2 reuses the verified Nut B grasp template for every Nut and preserves the
existing release, retreat, left-flow, episode/reset, gate, and diagnostic
behavior. No world reset is used for RIGHT_READY or LEFT_READY.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "three_nut_closed_loop"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD,
    DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD,
    HAND_OPEN,
    KNOWN_FIXED_NUT_WORLD_POSE,
    KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
    LEFT_FIXED_PLACE_JOINT_PATHS,
    LEFT_FIXED_PLACE_REFERENCE_EPISODE,
    LEFT_PLACE_POSES,
    LEFT_PRE_JOINTS,
    NUT_IDS,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import (  # noqa: E402
    build_right_safe_lift_pose,
    compute_right_approach_pose,
    compute_right_grasp_pose,
    pose_to_list,
)
from expert.left_nut_grasp_planner import (  # noqa: E402
    SAFE_HIGH_POSE,
    LeftNutGraspPlanner,
    jsonable,
    normalize_pose_check,
    result_failed,
)
from tools.motion_monitor import (  # noqa: E402
    MotionMonitor,
    evaluate_motion_result,
    extract_joint_target,
    max_abs_error,
    read_joint_method,
)
from agents.three_nut_expert.execution import ExecutionCoordinator  # noqa: E402
from tools.probe_act_recording_sources import IntegratedRecordingProbe  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import (  # noqa: E402
    LEFT_SAFE_LIFT_DELTA_Z_M,
    RIGHT_OBSERVATION_JOINTS,
    RIGHT_OBSERVATION_STABLE_WAIT_S,
    RIGHT_RELEASE_OPEN_WAIT_S,
    safe_lift_pose_from_grasp,
)
from tools.test_dual_closed_loop_v2_2 import (  # noqa: E402
    RIGHT_RELEASE_POSE,
    RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
    build_right_release_safe_height_pose,
)
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402
from tools.test_nut_camera_calibration import set_pose_with_retry  # noqa: E402
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    TARGET_NUT_B_WORLD_POSE,
    detect_released_nut,
    make_right_bundle,
    shutdown_bundle,
)


NUT_SEQUENCE = ("C", "B", "A")
LEFT_INITIAL_READY_PATH = tuple(tuple(float(v) for v in joints) for joints in LEFT_PRE_JOINTS)
LEFT_READY_GOAL = LEFT_INITIAL_READY_PATH[-1]
LEFT_RETURN_READY_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)
RELEASE_TARGET_WORLD_XYZ = tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3])
class ThreeNutClosedLoopError(RuntimeError):
    pass


class StateFault(ThreeNutClosedLoopError):
    def __init__(self, message: str, context: dict[str, Any]) -> None:
        super().__init__(message)
        self.context = context


def now_text(milliseconds: bool = False) -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds" if milliseconds else "seconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_sequence_arg(value: str) -> tuple[str, ...]:
    cleaned = value.replace(" ", "").upper()
    sequence = tuple(item for item in cleaned.split(",") if item) if "," in cleaned else tuple(cleaned)
    if not sequence:
        raise ThreeNutClosedLoopError("--sequence must not be empty")
    invalid = [key for key in sequence if key not in NUT_SEQUENCE]
    if invalid:
        raise ThreeNutClosedLoopError(f"--sequence contains invalid nut IDs: {invalid}")
    return sequence


def validate_static_contract(sequence: tuple[str, ...]) -> None:
    if NUT_SEQUENCE != ("C", "B", "A"):
        raise ThreeNutClosedLoopError("V2 default sequence must remain C,B,A")
    b = KNOWN_FIXED_NUT_WORLD_POSE["B"]
    computed_b = compute_right_grasp_pose((b.x, b.y, b.z))
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        for actual, expected in zip(pose_to_list(computed_b), pose_to_list(RIGHT_NUT_B_GRASP_POSE))
    ):
        raise ThreeNutClosedLoopError("Nut B verified right grasp regression failed")
    for key in sequence:
        if key not in NUT_IDS or key not in NUT_SPECS or key not in LEFT_PLACE_POSES:
            raise ThreeNutClosedLoopError(f"incomplete static contract for Nut {key}")


def pose_check_cartesian(arm: Any, pose: list[float], *, coordinator: ExecutionCoordinator | None = None, phase: str = "POSE_CHECK") -> tuple[list[float] | None, dict[str, Any]]:
    if not hasattr(arm, "pose_check"):
        return None, {
            "pose_check_reachable": False,
            "pose_check_target_joint_available": False,
            "reason": "pose_check unavailable",
        }
    try:
        call = lambda: arm.pose_check(*pose[:3], roll=pose[3], pitch=pose[4], yaw=pose[5])
        raw = coordinator.call(phase=phase, device="LEFT_ARM" if phase.startswith("LEFT") else "RIGHT_ARM", operation="pose_check", fn=call) if coordinator else call()
    except Exception as exc:
        return None, {
            "pose_check_reachable": False,
            "pose_check_target_joint_available": False,
            "reason": repr(exc),
        }
    status, reason, normalized = normalize_pose_check(raw)
    target = extract_joint_target(raw)
    return target, {
        "pose_check_reachable": status == "PASS",
        "pose_check_status": status,
        "pose_check_reason": reason,
        "pose_check_target_joint_available": target is not None,
        "raw": normalized,
    }


def sdk_failed(value: Any) -> tuple[bool, str]:
    return result_failed(value)


@dataclass
class ExpertStateRunner:
    trial_id: int
    monitor: MotionMonitor
    report_path: Path
    nut: str | None = None
    state: str = "EPISODE_INIT"
    previous_state: str = ""
    last_success_state: str = ""
    states: list[dict[str, Any]] = field(default_factory=list)
    fault_path: str | None = None
    nonblocking_motion: bool = False
    coordinator: ExecutionCoordinator | None = None
    phase_started_monotonic: float = 0.0

    def _wait_nonblocking_settle(self, arm: Any, label: str, before: list[float] | None) -> tuple[list[float], dict[str, Any]]:
        timeout_s = 20.0 if "TRANSPORT" in label else 15.0
        prev = list(before[:7]) if before is not None and len(before) >= 7 else None
        started = time.monotonic()
        movement_started = False
        stable_count = 0
        max_delta = 0.0
        while time.monotonic() - started < timeout_s:
            time.sleep(0.2)
            try:
                if self.coordinator is not None:
                    device = "LEFT_ARM" if label.startswith("LEFT") else "RIGHT_ARM"
                    raw = self.coordinator.call(phase=label, device=device, operation="get_joint_angles", fn=lambda: arm.get_joint_angles())
                else:
                    raw = arm.get_joint_angles()
                current = [float(v) for v in list(raw)[:7]]
            except BaseException as exc:
                print(f"[NONBLOCKING_MOTION] phase={label} error={exc!r}", flush=True)
                raise ThreeNutClosedLoopError(f"{label}: nonblocking settle read failed: {exc}") from exc
            if len(current) != 7:
                raise ThreeNutClosedLoopError(f"{label}: nonblocking settle returned {len(current)} joints")
            if prev is None:
                prev = current
                continue
            delta = max(abs(a - b) for a, b in zip(current, prev))
            max_delta = max(max_delta, delta)
            if delta > 0.003:
                if not movement_started:
                    movement_started = True
                    print(f"[NONBLOCKING_MOTION] phase={label} movement_started=True", flush=True)
                stable_count = 0
            elif delta < 0.002 and (movement_started or time.monotonic() - started >= 1.0):
                stable_count += 1
            else:
                stable_count = 0
            prev = current
            if stable_count >= 3:
                elapsed = time.monotonic() - started
                print(f"[NONBLOCKING_MOTION] phase={label} settled=True elapsed={elapsed:.3f} max_delta={max_delta:.6f}", flush=True)
                return current, {"settled": True, "movement_started": movement_started, "max_delta": max_delta, "settle_elapsed_s": elapsed}
        print(f"[NONBLOCKING_MOTION] phase={label} timeout={timeout_s:g}s", flush=True)
        if self.coordinator is not None:
            self.coordinator._trace("timeout", phase=label, device=("LEFT_ARM" if label.startswith("LEFT") else "RIGHT_ARM"), operation="motion_settle")
        raise ThreeNutClosedLoopError(f"{label}: NONBLOCKING_MOTION_TIMEOUT")

    def enter(self, state: str, *, nut: str | None = None) -> None:
        self.previous_state = self.state
        self.state = state
        if nut is not None:
            self.nut = nut
        print(f"\n[STATE] {self.previous_state} -> {self.state} nut={self.nut or '-'}", flush=True)
        if self.coordinator is not None:
            self.coordinator.phase_start(self.state)
            self.phase_started_monotonic = time.monotonic()

    def pass_state(self, details: dict[str, Any] | None = None) -> None:
        row = {
            "state": self.state,
            "previous_state": self.previous_state,
            "nut": self.nut,
            "status": "STATE_PASS",
            "timestamp": now_text(True),
            "details": details or {},
        }
        self.states.append(jsonable(row))
        self.last_success_state = self.state
        if self.coordinator is not None and self.phase_started_monotonic:
            self.coordinator.phase_end(self.state, time.monotonic() - self.phase_started_monotonic)
        print(f"[STATE_PASS] {self.state}", flush=True)

    def fault(self, exc: BaseException, context: dict[str, Any] | None = None) -> StateFault:
        context = dict(context or {})
        snapshot = {
            "trial_id": self.trial_id,
            "nut": self.nut,
            "state": self.state,
            "previous_state": self.previous_state,
            "command_method": context.get("command_method", ""),
            "command_label": context.get("command_label", ""),
            "target_pose": context.get("target_pose"),
            "target_joint": context.get("target_joint"),
            "joint_before": context.get("joint_before"),
            "joint_after": context.get("joint_after"),
            "recent_joint_history": context.get("recent_joint_history", []),
            "max_joint_error": context.get("max_joint_error"),
            "final_joint_error": context.get("final_joint_error"),
            "max_joint_jump": context.get("max_joint_jump"),
            "current_joint": context.get("current_joint"),
            "anchor_joint": context.get("anchor_joint"),
            "max_abs_error": context.get("max_abs_error"),
            "threshold": context.get("threshold"),
            "current_joint_before": context.get("current_joint_before"),
            "initial_max_abs_error": context.get("initial_max_abs_error"),
            "sdk_return": context.get("sdk_return"),
            "current_joint_after": context.get("current_joint_after"),
            "final_max_abs_error": context.get("final_max_abs_error"),
            "command_start_time": context.get("command_start_time"),
            "command_end_time": context.get("command_end_time"),
            "elapsed_s": context.get("elapsed_s"),
            "timeout": bool(context.get("timeout", False)),
            "sdk_returned_after_timeout": context.get("sdk_returned_after_timeout", False),
            "motion_monitor": context.get("motion_monitor", {}),
            "pose_check_reachable": context.get("pose_check_reachable"),
            "pose_check_target_joint_available": context.get("pose_check_target_joint_available"),
            "joint_target_based_divergence_available": context.get("joint_target_based_divergence_available"),
            "cartesian_endpoint_feedback_available": context.get("cartesian_endpoint_feedback_available"),
            "endpoint_verification_status": context.get("endpoint_verification_status"),
            "motion_gate_status": context.get("motion_gate_status"),
            "actual_ee_pose": context.get("actual_ee_pose"),
            "position_error_m": context.get("position_error_m"),
            "orientation_error_deg": context.get("orientation_error_deg"),
            "exception": repr(exc),
            "last_success_state": self.last_success_state,
            "status": "FAULT",
            "created_at": now_text(True),
        }
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        path = REPORT_DIR / f"fault_{stamp}.json"
        write_json(path, snapshot)
        self.fault_path = str(path.relative_to(PROJECT_ROOT))
        self.states.append({
            "state": self.state,
            "previous_state": self.previous_state,
            "nut": self.nut,
            "status": "STATE_FAULT",
            "fault_snapshot": self.fault_path,
            "exception": repr(exc),
        })
        print(f"[STATE_FAULT] {self.state}: {exc!r}", flush=True)
        print(f"[FAULT_SNAPSHOT] {self.fault_path}", flush=True)
        return StateFault(str(exc), snapshot)

    def arm_action(
        self,
        *,
        label: str,
        arm: Any,
        command_method: str,
        fn: Callable[[], Any],
        target_joint: list[float] | None = None,
        target_pose: list[float] | None = None,
        pose_check: dict[str, Any] | None = None,
        context_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonblocking = bool(self.nonblocking_motion or self.coordinator is not None)
        if self.coordinator is not None:
            try:
                device = "LEFT_ARM" if label.startswith("LEFT") else "RIGHT_ARM"
                before = list(self.coordinator.call(phase=label, device=device, operation="get_joint_angles", fn=lambda: arm.get_joint_angles()))[:7]
                before_error = None
            except Exception as exc:
                before, before_error = None, repr(exc)
        else:
            before, before_error = read_joint_method(arm, ("get_joint_angles", "get_joints", "get_qpos"))
        start_text = now_text(True)
        started = time.monotonic()
        timeout_s = float(self.monitor.config["action_timeout_s"])
        monitor_enabled = bool(self.monitor.enabled) and not nonblocking
        timeout_event = threading.Event()

        def mark_timeout() -> None:
            timeout_event.set()
            print(f"[TIMEOUT_PENDING] {label} exceeded {timeout_s:g}s; no later state will run", flush=True)

        timer = threading.Timer(timeout_s, mark_timeout) if monitor_enabled else None
        if timer is not None:
            timer.daemon = True
        monitor_started = False
        monitor_result = None
        value = None
        command_exc: BaseException | None = None
        after: list[float] | None = None
        after_error: str | None = None
        settle: dict[str, Any] = {}
        try:
            if monitor_enabled:
                self.monitor.start_motion_monitor(
                    label,
                    arm,
                    target_joint=target_joint,
                    target_ee_pose=target_pose,
                    command_method=command_method,
                )
                monitor_started = True
                assert timer is not None
                timer.start()
            if nonblocking:
                print(f"[NONBLOCKING_MOTION] phase={label} command={command_method} sent=True", flush=True)
            if self.coordinator is not None:
                device = "LEFT_ARM" if label.startswith("LEFT") else "RIGHT_ARM"
                value = self.coordinator.call(phase=label, device=device, operation=command_method, fn=fn)
            else:
                value = fn()
            failed, reason = sdk_failed(value)
            if failed:
                command_exc = ThreeNutClosedLoopError(f"{label}: SDK return failed: {reason}")
            elif nonblocking:
                after, settle = self._wait_nonblocking_settle(arm, label, before)
        except BaseException as exc:
            command_exc = exc
        finally:
            if timer is not None:
                timer.cancel()
            elapsed = time.monotonic() - started
            timed_out = monitor_enabled and (timeout_event.is_set() or elapsed >= timeout_s)
            if monitor_started:
                try:
                    monitor_result = self.monitor.stop_motion_monitor(
                        status="FAILED" if command_exc or timed_out else "COMPLETED",
                        error_info=repr(command_exc) if command_exc else ("TIMEOUT" if timed_out else ""),
                    )
                except Exception as exc:
                    command_exc = command_exc or exc
            if not nonblocking:
                after, after_error = read_joint_method(arm, ("get_joint_angles", "get_joints", "get_qpos"))

        gate = (
            evaluate_motion_result(
                monitor_result,
                timeout=timed_out,
                sdk_return_ok=command_exc is None,
            )
            if monitor_enabled
            else {
                "pass": command_exc is None,
                "status": "PASS_MONITOR_DISABLED" if command_exc is None else "FAULT",
                "reason": "motion_monitor_disabled",
                "sdk_return_ok": command_exc is None,
                "timeout": False,
                "diverged": False,
                "max_joint_error": None,
                "final_joint_error": None,
                "max_joint_jump": None,
                "diagnosis": "MOTION_MONITOR_DISABLED",
                "cartesian_endpoint_feedback_available": None,
                "endpoint_verification_status": "DISABLED",
                "metrics": {},
            }
        )
        metrics = gate.get("metrics") or {}
        context = {
            "command_method": command_method,
            "command_label": label,
            "target_pose": target_pose,
            "target_joint": target_joint,
            "joint_before": before,
            "joint_after": after,
            "joint_read_before_error": before_error,
            "joint_read_after_error": after_error,
            "recent_joint_history": (monitor_result or {}).get("recent_joint_history", []),
            "max_joint_error": gate.get("max_joint_error"),
            "final_joint_error": gate.get("final_joint_error"),
            "max_joint_jump": gate.get("max_joint_jump"),
            "command_start_time": start_text,
            "command_end_time": now_text(True),
            "elapsed_s": round(elapsed, 4),
            "timeout": timed_out,
            "sdk_returned_after_timeout": bool(timed_out),
            "motion_monitor": monitor_result or {},
            "nonblocking_motion": nonblocking,
            "motion_settle": locals().get("settle", {}),
            "gate": gate,
            "sdk_return": jsonable(value),
            "pose_check": pose_check or {},
            "pose_check_reachable": (pose_check or {}).get("pose_check_reachable"),
            "pose_check_target_joint_available": (pose_check or {}).get("pose_check_target_joint_available"),
            "joint_target_based_divergence_available": metrics.get(
                "joint_target_based_divergence_available",
                target_joint is not None,
            ),
            "cartesian_endpoint_feedback_available": gate.get(
                "cartesian_endpoint_feedback_available"
            ),
            "endpoint_verification_status": gate.get("endpoint_verification_status"),
            "motion_gate_status": gate.get("status"),
            "actual_ee_pose": metrics.get("final_actual_ee_pose"),
            "position_error_m": metrics.get("final_position_error"),
            "orientation_error_deg": metrics.get("final_orientation_error"),
        }
        context.update(jsonable(context_extra or {}))
        alignment_anchor = context.get("anchor_joint")
        if alignment_anchor is not None:
            context.setdefault("current_joint_after", after)
            context.setdefault("final_max_abs_error", max_abs_error(after, alignment_anchor))
        if command_exc is not None:
            raise self.fault(command_exc, context)
        if not gate["pass"]:
            raise self.fault(ThreeNutClosedLoopError(f"{label}: motion gate failed: {gate['reason']}"), context)
        if gate.get("status") == "PASS_WITHOUT_CARTESIAN_ENDPOINT_FEEDBACK":
            print(
                f"[MOTION_GATE] {label}: PASS_WITHOUT_CARTESIAN_ENDPOINT_FEEDBACK "
                "(get_pose unavailable/unparseable; diagnostic only)",
                flush=True,
            )
        return context

    def hand_action(self, *, label: str, fn: Callable[[], Any], command_method: str) -> Any:
        start = time.monotonic()
        start_text = now_text(True)
        timeout_s = float(self.monitor.config["action_timeout_s"])
        timeout_event = threading.Event()
        timer = threading.Timer(timeout_s, timeout_event.set) if self.monitor.enabled else None
        if timer is not None:
            timer.daemon = True
            timer.start()
        try:
            if self.coordinator is not None:
                hand_device = "LEFT_HAND" if command_method.startswith("left_hand") else "RIGHT_HAND"
                value = self.coordinator.call(phase=label, device=hand_device, operation=command_method, fn=fn)
            else:
                value = fn()
        except BaseException as exc:
            if timer is not None:
                timer.cancel()
            raise self.fault(exc, {
                "command_method": command_method,
                "command_label": label,
                "command_start_time": start_text,
                "command_end_time": now_text(True),
                "elapsed_s": time.monotonic() - start,
                "timeout": timeout_event.is_set(),
            })
        if timer is not None:
            timer.cancel()
        elapsed = time.monotonic() - start
        timed_out = bool(self.monitor.enabled) and (
            timeout_event.is_set() or elapsed >= timeout_s
        )
        failed, reason = sdk_failed(value)
        if timed_out or failed:
            message = f"{label}: " + ("TIMEOUT" if timed_out else f"SDK return failed: {reason}")
            raise self.fault(ThreeNutClosedLoopError(message), {
                "command_method": command_method,
                "command_label": label,
                "command_start_time": start_text,
                "command_end_time": now_text(True),
                "elapsed_s": elapsed,
                "timeout": timed_out,
                "sdk_returned_after_timeout": timed_out,
            })
        return value

    def ready_check(self, arm: Any, target: list[float], label: str) -> dict[str, Any]:
        count = int(self.monitor.config["ready_sample_count"])
        interval = float(self.monitor.config["ready_sample_interval_s"])
        error_limit = float(self.monitor.config["ready_joint_error_threshold_rad"])
        stability_limit = float(self.monitor.config["ready_stability_delta_threshold_rad"])
        samples: list[list[float]] = []
        for index in range(count):
            joints, error = read_joint_method(arm, ("get_joint_angles", "get_joints", "get_qpos"))
            if joints is None:
                raise self.fault(ThreeNutClosedLoopError(f"{label}: joint state unavailable: {error}"), {
                    "command_method": "get_joint_angles",
                    "target_joint": target,
                    "joint_after": joints,
                })
            samples.append(joints)
            if index + 1 < count:
                time.sleep(interval)
        errors = [max_abs_error(target, sample) for sample in samples]
        deltas = [max_abs_error(a, b) for a, b in zip(samples, samples[1:])]
        max_error = max(value for value in errors if value is not None)
        max_delta = max((value for value in deltas if value is not None), default=0.0)
        result = {
            "confirmed": max_error <= error_limit and max_delta <= stability_limit,
            "samples": samples,
            "max_abs_joint_error": max_error,
            "max_adjacent_joint_delta": max_delta,
            "thresholds": {
                "joint_error_rad": error_limit,
                "stability_delta_rad": stability_limit,
                "status": "PROVISIONAL_THRESHOLD",
            },
        }
        if not result["confirmed"]:
            raise self.fault(ThreeNutClosedLoopError(f"{label}: READY_CHECK failed"), {
                "command_method": "READY_CHECK",
                "target_joint": target,
                "joint_after": samples[-1],
                "recent_joint_history": samples,
                "max_joint_error": max_error,
                "max_joint_jump": max_delta,
            })
        return result


def move_joints_path(
    runner: ExpertStateRunner,
    arm: Any,
    path: tuple[tuple[float, ...], ...],
    label: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, joints in enumerate(path, start=1):
        target = list(joints)
        rows.append(runner.arm_action(
            label=f"{label}_{index}",
            arm=arm,
            command_method="move_joints",
            target_joint=target,
            fn=lambda target=target: arm.move_joints(target, blocking=False if runner.coordinator else not runner.nonblocking_motion),
        ))
    return rows


def move_pose(
    runner: ExpertStateRunner,
    arm: Any,
    pose: Pose6 | list[float],
    label: str,
) -> dict[str, Any]:
    values = pose_to_list(pose) if isinstance(pose, Pose6) else [float(v) for v in pose]
    target_joint, pose_check = pose_check_cartesian(arm, values, coordinator=runner.coordinator, phase=label)
    if not pose_check["pose_check_reachable"]:
        raise runner.fault(ThreeNutClosedLoopError(f"{label}: Cartesian pose_check failed: {pose_check['pose_check_reason']}"), {
            "command_method": "pose_check",
            "target_pose": values,
            "target_joint": target_joint,
            "pose_check_reachable": False,
            "pose_check_target_joint_available": target_joint is not None,
            "motion_monitor": {"pose_check": pose_check},
        })

    def send_move_command() -> Any:
        if label.startswith(("RIGHT_APPROACH_", "RIGHT_PICK_", "RIGHT_SAFE_LIFT_")):
            print("[RIGHT_MOVE_COMMAND]")
            print(f"{label} = {json.dumps(values)}")
        return arm.move_to(*values[:3], roll=values[3], pitch=values[4], yaw=values[5], blocking=False if runner.coordinator else not runner.nonblocking_motion)

    return runner.arm_action(
        label=label,
        arm=arm,
        command_method="move_to",
        target_joint=target_joint,
        target_pose=values,
        pose_check=pose_check,
        fn=send_move_command,
    )


def go_right_ready(runner: ExpertStateRunner, right_arm: Any) -> None:
    runner.enter("RIGHT_READY")
    rows = move_joints_path(runner, right_arm, RIGHT_OBSERVATION_JOINTS, "RIGHT_READY")
    time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
    runner.pass_state({"path": rows})
    runner.enter("RIGHT_READY_CHECK")
    runner.pass_state(runner.ready_check(right_arm, list(RIGHT_OBSERVATION_JOINTS[-1]), "RIGHT_READY_CHECK"))


def go_left_initial_ready(runner: ExpertStateRunner, left_arm: Any) -> None:
    """Run the verified startup-only LEFT_READY_1 -> LEFT_READY_2 path."""
    runner.enter("LEFT_READY")
    rows = move_joints_path(runner, left_arm, LEFT_INITIAL_READY_PATH, "LEFT_READY")
    runner.pass_state({
        "path": rows,
        "path_role": "LEFT_INITIAL_READY_PATH",
        "source": "agents.three_nut_expert.config.LEFT_PRE_JOINTS",
    })
    runner.enter("LEFT_READY_CHECK")
    runner.pass_state(runner.ready_check(left_arm, list(LEFT_READY_GOAL), "LEFT_READY_CHECK"))


def build_left_return_ready_path(
    q_start: Sequence[float],
    q_goal: Sequence[float] = LEFT_READY_GOAL,
) -> tuple[tuple[float, ...], ...]:
    """Interpolate the post-task return without reusing LEFT_READY_1."""
    start = tuple(float(value) for value in q_start)
    goal = tuple(float(value) for value in q_goal)
    if len(start) != len(goal):
        raise ThreeNutClosedLoopError(
            f"LEFT_RETURN_READY joint dimension mismatch: start={len(start)} goal={len(goal)}"
        )
    waypoints = tuple(
        tuple(start_value + alpha * (goal_value - start_value) for start_value, goal_value in zip(start, goal))
        for alpha in LEFT_RETURN_READY_ALPHAS[:-1]
    )
    return (*waypoints, goal)


def go_left_return_ready(runner: ExpertStateRunner, left_arm: Any) -> None:
    """Return from the actual post-retreat joints through five monitored segments."""
    runner.enter("LEFT_RETURN_READY")
    q_start, read_error = read_joint_method(left_arm, ("get_joint_angles", "get_joints", "get_qpos"))
    if q_start is None:
        raise runner.fault(ThreeNutClosedLoopError(
            f"LEFT_RETURN_READY: joint state unavailable after safe retreat: {read_error}"
        ), {
            "command_method": "get_joint_angles",
            "command_label": "LEFT_RETURN_READY_READ_ACTUAL",
            "target_joint": list(LEFT_READY_GOAL),
        })
    if len(q_start) != len(LEFT_READY_GOAL):
        raise runner.fault(ThreeNutClosedLoopError(
            f"LEFT_RETURN_READY: expected {len(LEFT_READY_GOAL)} actual joints, got {len(q_start)}"
        ), {
            "command_method": "get_joint_angles",
            "command_label": "LEFT_RETURN_READY_READ_ACTUAL",
            "target_joint": list(LEFT_READY_GOAL),
            "joint_before": q_start,
        })
    path = build_left_return_ready_path(q_start)
    rows = move_joints_path(runner, left_arm, path, "LEFT_RETURN_READY")
    runner.pass_state({
        "q_start": q_start,
        "q_goal": list(LEFT_READY_GOAL),
        "alphas": list(LEFT_RETURN_READY_ALPHAS),
        "path": rows,
        "path_role": "LEFT_RETURN_READY_PATH",
        "excluded_waypoint": list(LEFT_INITIAL_READY_PATH[0]),
    })
    runner.enter("LEFT_READY_CHECK")
    runner.pass_state(runner.ready_check(left_arm, list(LEFT_READY_GOAL), "LEFT_READY_CHECK"))


def vertical_retreat_from_place(place: Pose6) -> list[float]:
    return [place.x, place.y, place.z + LEFT_SAFE_LIFT_DELTA_Z_M, place.roll, place.pitch, place.yaw]


def place_above_pose(place: Pose6, safe_z: float) -> list[float]:
    """Keep the verified Place x/y/RPY while staying at the current safe height."""
    return [place.x, place.y, float(safe_z), place.roll, place.pitch, place.yaw]


def execute_recorded_joint_path(
    runner: ExpertStateRunner,
    left_arm: Any,
    key: str,
    segment: str,
) -> dict[str, Any]:
    """Execute sparse actual_joint samples from the successful reference Episode."""
    try:
        waypoints = LEFT_FIXED_PLACE_JOINT_PATHS[key][segment]
    except KeyError as exc:
        raise ThreeNutClosedLoopError(
            f"no deterministic {segment} joint path for Nut {key}"
        ) from exc
    entry_alignment = None
    if segment == "transport":
        if not (
            runner.last_success_state == "LEFT_SAFE_LIFT"
            and runner.previous_state == "LEFT_SAFE_LIFT"
            and runner.state == "LEFT_PLACE_ABOVE"
        ):
            raise runner.fault(
                ThreeNutClosedLoopError(
                    "DETERMINISTIC_PLACE_ENTRY_STATE_INVALID: "
                    f"last_success_state={runner.last_success_state!r}"
                ),
                {
                    "command_method": "DETERMINISTIC_PLACE_ENTRY_GATE",
                    "command_label": f"LEFT_FIXED_ENTRY_ALIGN_{key}",
                    "required_last_success_state": "LEFT_SAFE_LIFT",
                    "required_state": "LEFT_PLACE_ABOVE",
                },
            )
        current_before, read_error = read_joint_method(
            left_arm, ("get_joint_angles", "get_joints", "get_qpos")
        )
        anchor_joint = list(waypoints[0].joints)
        initial_error = (
            max_abs_error(current_before, anchor_joint)
            if current_before is not None and len(current_before) == len(anchor_joint)
            else None
        )
        entry_context = {
            "current_joint_before": current_before,
            "anchor_joint": anchor_joint,
            "initial_max_abs_error": initial_error,
            "eligibility_threshold": DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD,
            "final_threshold": DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD,
            "joint_read_before_error": read_error,
        }
        if initial_error is None or initial_error > DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD:
            raise runner.fault(
                ThreeNutClosedLoopError(
                    "DETERMINISTIC_PLACE_ENTRY_TOO_FAR: "
                    f"Nut {key} initial_max_abs_error={initial_error} "
                    f"threshold={DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD}"
                ),
                {
                    "command_method": "DETERMINISTIC_PLACE_ENTRY_GATE",
                    "command_label": f"LEFT_FIXED_ENTRY_ALIGN_{key}",
                    **entry_context,
                    "threshold": DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD,
                },
            )
        align_motion = runner.arm_action(
            label=f"LEFT_FIXED_ENTRY_ALIGN_{key}",
            arm=left_arm,
            command_method="move_joints",
            target_joint=anchor_joint,
            fn=lambda: left_arm.move_joints(anchor_joint, blocking=False if runner.coordinator else not runner.nonblocking_motion),
            context_extra=entry_context,
        )
        if runner.coordinator is not None:
            try:
                current_after = list(runner.coordinator.call(phase=f"LEFT_FIXED_ENTRY_ALIGN_{key}", device="LEFT_ARM", operation="get_joint_angles", fn=lambda: left_arm.get_joint_angles()))[:7]
                read_after_error = None
            except Exception as exc:
                current_after, read_after_error = None, repr(exc)
        else:
            current_after, read_after_error = read_joint_method(
                left_arm, ("get_joint_angles", "get_joints", "get_qpos")
            )
        final_error = (
            max_abs_error(current_after, anchor_joint)
            if current_after is not None and len(current_after) == len(anchor_joint)
            else None
        )
        alignment_report = {
            **entry_context,
            "sdk_return": align_motion.get("sdk_return"),
            "current_joint_after": current_after,
            "final_max_abs_error": final_error,
            "joint_read_after_error": read_after_error,
        }
        if final_error is None or final_error > DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD:
            raise runner.fault(
                ThreeNutClosedLoopError(
                    "DETERMINISTIC_PLACE_ENTRY_ALIGN_FAILED: "
                    f"Nut {key} final_max_abs_error={final_error} "
                    f"threshold={DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD}"
                ),
                {
                    "command_method": "move_joints",
                    "command_label": f"LEFT_FIXED_ENTRY_ALIGN_{key}",
                    **alignment_report,
                    "threshold": DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD,
                },
            )
        entry_alignment = {**alignment_report, "motion": align_motion}
    path = tuple(tuple(float(value) for value in item.joints) for item in waypoints)
    rows = move_joints_path(runner, left_arm, path, f"LEFT_FIXED_{segment.upper()}_{key}")
    return {
        "reference_episode": LEFT_FIXED_PLACE_REFERENCE_EPISODE,
        "segment": segment,
        "entry_alignment": entry_alignment,
        "waypoints": [
            {
                "joints": list(item.joints),
                "source_trajectory": item.source_trajectory,
                "sample_index": item.sample_index,
                "time_s": item.time_s,
            }
            for item in waypoints
        ],
        "path": rows,
    }


def execute_right_transfer(
    runner: ExpertStateRunner,
    key: str,
    right_bundle: Any,
    *,
    runtime_nut_world_xyz: Sequence[float],
    nut_xyz_source: str,
    vision_radius_m: float,
    settle_after_release_s: float,
) -> list[float]:
    runtime_xyz = [float(value) for value in runtime_nut_world_xyz]
    grasp_pose = compute_right_grasp_pose(runtime_xyz)
    approach_pose = compute_right_approach_pose(grasp_pose)
    print("[NUT_RUNTIME_TARGET]")
    print(f"nut={key}")
    print(f"source={nut_xyz_source}")
    print(f"world_xyz={json.dumps(runtime_xyz)}")
    print("[RIGHT_GRASP_COMPUTE]")
    print(f"grasp_pose={json.dumps(pose_to_list(grasp_pose))}")
    print("[RIGHT_APPROACH_COMPUTE]")
    print(f"approach_pose={json.dumps(pose_to_list(approach_pose))}")
    runner.enter("RIGHT_APPROACH", nut=key)
    approach_move = move_pose(runner, right_bundle.right_arm, approach_pose, f"RIGHT_APPROACH_{key}")
    runner.pass_state({
        "approach_pose": pose_to_list(approach_pose),
        "grasp_pose": pose_to_list(grasp_pose),
        "nut_world_xyz": runtime_xyz,
        "nut_xyz_source": nut_xyz_source,
        "move": approach_move,
    })

    runner.enter("RIGHT_THUMB_TUCK", nut=key)
    thumb_tuck_result = runner.hand_action(
        label="RIGHT_THUMB_TUCK",
        command_method="right_hand.clench",
        fn=lambda: right_bundle.right_hand.clench(thumb_rotation=1.0),
    )
    runner.pass_state({
        "command": "right_hand.clench",
        "thumb_rotation": 1.0,
        "sdk_return": jsonable(thumb_tuck_result),
    })

    runner.enter("RIGHT_GRASP", nut=key)
    pick_move = move_pose(runner, right_bundle.right_arm, grasp_pose, f"RIGHT_PICK_{key}")
    runner.pass_state({
        "grasp_pose": pose_to_list(grasp_pose),
        "source": "KNOWN_FIXED_NUT_WORLD_POSE -> compute_right_grasp_pose using VERIFIED Nut B template",
        "nut_world_xyz": runtime_xyz,
        "nut_xyz_source": nut_xyz_source,
        "move": pick_move,
    })

    runner.enter("RIGHT_GRASP_FORCE", nut=key)
    grasp_force_result = runner.hand_action(
        label="RIGHT_GRASP_FORCE",
        command_method="right_hand.grasp_force",
        fn=lambda: right_bundle.right_hand.grasp_force(**RIGHT_GRASP_FORCE),
    )
    if DEFAULT_HOLD_AFTER_GRASP_S > 0:
        time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
    runner.pass_state({
        "command": "right_hand.grasp_force",
        "target": dict(RIGHT_GRASP_FORCE),
        "sdk_return": jsonable(grasp_force_result),
    })

    runner.enter("RIGHT_LIFT")
    safe_lift_pose = build_right_safe_lift_pose(grasp_pose)
    print("[RIGHT_SAFE_LIFT_COMPUTE]")
    print(f"safe_lift_pose={json.dumps(pose_to_list(safe_lift_pose))}")
    safe_lift_move = move_pose(
        runner,
        right_bundle.right_arm,
        safe_lift_pose,
        f"RIGHT_SAFE_LIFT_{key}",
    )
    runner.pass_state({
        "safe_lift_pose": pose_to_list(safe_lift_pose),
        "source_grasp_pose": pose_to_list(grasp_pose),
        "move": safe_lift_move,
    })

    runner.enter("RIGHT_RELEASE")
    move_pose(runner, right_bundle.right_arm, RIGHT_RELEASE_POSE, "RIGHT_RELEASE_POSE")
    runner.hand_action(label="RIGHT_RELEASE_OPEN", command_method="right_hand.clench", fn=lambda: right_bundle.right_hand.clench(*list(HAND_OPEN)))
    release_started = time.monotonic()
    runner.pass_state({"release_pose": pose_to_list(RIGHT_RELEASE_POSE)})

    runner.enter("RIGHT_SAFE_RETREAT")
    time.sleep(RIGHT_RELEASE_OPEN_WAIT_S)
    safe_pose = build_right_release_safe_height_pose()
    row = move_pose(runner, right_bundle.right_arm, safe_pose, "RIGHT_RELEASE_SAFE_HEIGHT")
    runner.pass_state({"move": row, "offset_z_m": RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z})
    go_right_ready(runner, right_bundle.right_arm)

    remaining = max(0.0, settle_after_release_s - (time.monotonic() - release_started))
    time.sleep(remaining)
    runner.enter("LEFT_VISION")
    released = detect_released_nut(RELEASE_TARGET_WORLD_XYZ, vision_radius_m)
    if not released.get("detected") or not released.get("nut_world_xyz"):
        raise runner.fault(ThreeNutClosedLoopError(f"release vision failed: {released.get('error')}"), {
            "command_method": "detect_released_nut",
        })
    position = [float(v) for v in released["nut_world_xyz"]]
    runner.pass_state({"released_nut_position": position})
    return position


def execute_left_pick_place(
    runner: ExpertStateRunner,
    key: str,
    left_bundle: Any,
    nut_world_xyz: list[float],
    *,
    deterministic_place_path: bool = False,
    terminal_after_release: bool = False,
) -> None:
    if deterministic_place_path:
        if key not in LEFT_FIXED_PLACE_JOINT_PATHS:
            raise ThreeNutClosedLoopError(f"no deterministic place path for Nut {key}")
        if not terminal_after_release and "retreat" not in LEFT_FIXED_PLACE_JOINT_PATHS[key]:
            raise ThreeNutClosedLoopError(
                f"Nut {key} has no successful deterministic retreat; it must be terminal"
            )
    planner = LeftNutGraspPlanner(left_arm=left_bundle.left_arm, left_hand=left_bundle.left_hand)
    plan = planner.build_plan(nut_world_xyz)
    preflight = planner.preflight_grasp_chain(plan)
    if not preflight["success"]:
        raise runner.fault(ThreeNutClosedLoopError(f"left preflight failed: {preflight['reason']}"), {
            "command_method": "pose_check",
            "target_pose": preflight.get("failed_pose"),
        })

    runner.enter("LEFT_APPROACH")
    runner.hand_action(label="LEFT_OPEN", command_method="left_hand.open/clench", fn=lambda: left_bundle.left_hand.open() if hasattr(left_bundle.left_hand, "open") else left_bundle.left_hand.clench(list(HAND_OPEN)))
    approach_rows = []
    for item in plan["path"]:
        if item["stage"] not in {"SAFE_HIGH", "PREGRASP"}:
            continue
        approach_rows.append(move_pose(runner, left_bundle.left_arm, item["pose"], f"LEFT_{item['stage']}"))
    runner.pass_state({"pregrasp_pose": plan["pregrasp_pose"], "moves": approach_rows})

    runner.enter("LEFT_THUMB_TUCK", nut=key)
    thumb_tuck_result = runner.hand_action(
        label="LEFT_THUMB_TUCK",
        command_method="left_hand.clench",
        fn=lambda: left_bundle.left_hand.clench(thumb_rotation=1.0),
    )
    time.sleep(0.7)
    runner.pass_state({
        "command": "left_hand.clench",
        "thumb_rotation": 1.0,
        "sdk_return": jsonable(thumb_tuck_result),
    })

    runner.enter("LEFT_DESCENT", nut=key)
    descent_rows = [
        move_pose(runner, left_bundle.left_arm, item["pose"], f"LEFT_{item['stage']}")
        for item in plan["path"]
        if item["stage"].startswith("DESCENT_")
    ]
    runner.pass_state({"moves": descent_rows})

    runner.enter("LEFT_GRASP", nut=key)
    grasp_item = next(item for item in plan["path"] if item["stage"] == "GRASP")
    grasp_move = move_pose(runner, left_bundle.left_arm, grasp_item["pose"], "LEFT_GRASP")
    runner.pass_state({"grasp_pose": plan["grasp_pose"], "move": grasp_move})

    runner.enter("LEFT_GRASP_FORCE", nut=key)
    grasp_force_result = runner.hand_action(
        label="LEFT_GRASP_FORCE",
        command_method="left_hand.grasp_force",
        fn=lambda: left_bundle.left_hand.grasp_force(**planner.config.grasp_force),
    )
    runner.pass_state({
        "command": "left_hand.grasp_force",
        "target": dict(planner.config.grasp_force),
        "sdk_return": jsonable(grasp_force_result),
    })

    runner.enter("LEFT_SAFE_LIFT")
    time.sleep(0.5)
    lift_rows = [move_pose(runner, left_bundle.left_arm, item["pose"], f"LEFT_{item['stage']}") for item in plan["lift_waypoints"]]
    extra_safe_lift = safe_lift_pose_from_grasp(plan["grasp_pose"])
    lift_rows.append(move_pose(runner, left_bundle.left_arm, extra_safe_lift, "LEFT_SAFE_LIFT_FINAL"))
    runner.pass_state({"moves": lift_rows, "safe_lift_pose": extra_safe_lift})

    place = LEFT_PLACE_POSES[key]
    place_above = place_above_pose(place, extra_safe_lift[2])
    if deterministic_place_path:
        runner.enter("LEFT_PLACE_ABOVE")
        transport = execute_recorded_joint_path(
            runner, left_bundle.left_arm, key, "transport"
        )
        runner.pass_state({
            "motion": transport,
            "place_above_pose": place_above,
            "execution": "RECORDED_ACTUAL_JOINT_WAYPOINTS",
        })

        runner.enter("LEFT_PLACE")
        fixed_place = execute_recorded_joint_path(
            runner, left_bundle.left_arm, key, "place"
        )
        runner.hand_action(
            label="LEFT_RELEASE_OPEN",
            command_method="left_hand.clench",
            fn=lambda: left_bundle.left_hand.clench(*list(HAND_OPEN)),
        )
        runner.pass_state({
            "motion": fixed_place,
            "place_pose": pose_to_list(place),
            "execution": "RECORDED_ACTUAL_JOINT_WAYPOINTS",
            "terminal_after_release": terminal_after_release,
        })
        if terminal_after_release:
            return

        runner.enter("LEFT_SAFE_RETREAT")
        fixed_retreat = execute_recorded_joint_path(
            runner, left_bundle.left_arm, key, "retreat"
        )
        runner.pass_state({
            "motion": fixed_retreat,
            "safe_retreat_pose": vertical_retreat_from_place(place),
            "execution": "RECORDED_ACTUAL_JOINT_WAYPOINTS",
        })
        go_left_return_ready(runner, left_bundle.left_arm)
        return

    runner.enter("LEFT_PLACE_ABOVE")
    place_above_row = move_pose(
        runner,
        left_bundle.left_arm,
        place_above,
        f"LEFT_PLACE_ABOVE_{key}",
    )
    runner.pass_state({
        "move": place_above_row,
        "place_above_pose": place_above,
        "safe_z_source": "extra_safe_lift.z",
        "place_xy_rpy_source": f"LEFT_PLACE_POSES[{key!r}]",
    })

    runner.enter("LEFT_PLACE")
    place_row = move_pose(runner, left_bundle.left_arm, place, f"LEFT_PLACE_{key}")
    runner.hand_action(label="LEFT_RELEASE_OPEN", command_method="left_hand.clench", fn=lambda: left_bundle.left_hand.clench(*list(HAND_OPEN)))
    runner.pass_state({"move": place_row, "place_pose": pose_to_list(place)})

    runner.enter("LEFT_SAFE_RETREAT")
    retreat = vertical_retreat_from_place(place)
    retreat_row = move_pose(runner, left_bundle.left_arm, retreat, f"LEFT_SAFE_RETREAT_{key}")
    runner.pass_state({
        "move": retreat_row,
        "safe_retreat_pose": retreat,
        "geometry": f"vertical + existing {LEFT_SAFE_LIFT_DELTA_Z_M:g}m safe-lift delta",
    })
    go_left_return_ready(runner, left_bundle.left_arm)


def reset_nuts_to_nominal(pose_setter: Any) -> list[dict[str, Any]]:
    """Deterministically place A/B/C at verified fixed scene poses; never jitter."""
    rows = []
    for key in ("A", "B", "C"):
        pose = pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE[key])
        result = set_pose_with_retry(pose_setter, NUT_IDS[key], pose)
        if not result.get("ok"):
            raise ThreeNutClosedLoopError(f"EPISODE_INIT Nut {key} SetEntityPose failed: {result.get('error')}")
        rows.append({"nut": key, "entity_id": NUT_IDS[key], "pose": pose, "result": result})
    return rows


def make_pose_setter() -> Any:
    from rabo_dev_kit import SetEntityPose
    return SetEntityPose(world=WORLD_ID)


def shutdown_pose_setter(value: Any | None) -> None:
    if value is not None and hasattr(value, "shutdown"):
        try:
            value.shutdown()
        except Exception as exc:
            print(f"pose setter shutdown warning: {exc!r}")


def run_left_ready_test(args: argparse.Namespace) -> int:
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"left_ready_test_{started.strftime('%Y%m%d_%H%M%S')}.json"
    monitor = MotionMonitor(enabled=motion_gate_enabled(args))
    runner = ExpertStateRunner(1, monitor, report_path, nonblocking_motion=bool(args.nonblocking_motion))
    left_bundle = None
    report: dict[str, Any] = {
        "experiment": "left_ready_candidate_test",
        "status": "RUNNING",
        "threshold_status": "PROVISIONAL_THRESHOLD_REQUIRES_RABO_TUNING",
        "motion_gate_config": dict(monitor.config),
        "states": runner.states,
        "tests": [],
    }
    try:
        left_bundle = make_left_bundle()
        go_left_initial_ready(runner, left_bundle.left_arm)
        report["tests"].append({"test": 1, "chain": "current -> LEFT_READY", "status": "PASS"})

        runner.enter("LEFT_READY_TEST_2")
        move_pose(runner, left_bundle.left_arm, list(SAFE_HIGH_POSE), "LEFT_READY_TEST_SAFE_HIGH")
        go_left_return_ready(runner, left_bundle.left_arm)
        report["tests"].append({"test": 2, "chain": "LEFT_READY -> existing SAFE_HIGH -> LEFT_READY", "status": "PASS"})

        for key in NUT_SEQUENCE:
            runner.enter(f"LEFT_READY_TEST_3_{key}", nut=key)
            move_pose(runner, left_bundle.left_arm, list(SAFE_HIGH_POSE), f"LEFT_TEST_{key}_SAFE_HIGH")
            move_pose(runner, left_bundle.left_arm, LEFT_PLACE_POSES[key], f"LEFT_TEST_PLACE_{key}")
            move_pose(runner, left_bundle.left_arm, vertical_retreat_from_place(LEFT_PLACE_POSES[key]), f"LEFT_TEST_SAFE_RETREAT_{key}")
            go_left_return_ready(runner, left_bundle.left_arm)
            report["tests"].append({
                "test": 3,
                "nut": key,
                "chain": f"LEFT_PLACE_{key} -> vertical safe retreat -> LEFT_READY",
                "status": "PASS",
            })
        runner.enter("DONE")
        runner.pass_state({"left_ready_status": "RABO_RUNTIME_CONFIRMED_FOR_THIS_RUN"})
        report["status"] = "PASS"
        return_code = 0
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if not isinstance(exc, StateFault):
            exc = runner.fault(exc)
        report.update({"status": "FAILED", "failed_phase": runner.state, "failure_reason": repr(exc), "fault_snapshot": runner.fault_path})
        return_code = 130 if interrupted else 1
    finally:
        shutdown_left_bundle(left_bundle)
        report["states"] = runner.states
        report["finished_at"] = now_text()
        write_json(report_path, report)
        print(f"Result: {report['status']}\nReport: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def run_left_return_ready_c_test(args: argparse.Namespace) -> int:
    """Exercise only the C safe-retreat-to-Ready return chain."""
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"left_return_ready_c_test_{started.strftime('%Y%m%d_%H%M%S')}.json"
    left_bundle = None
    report: dict[str, Any] = {
        "experiment": "left_safe_retreat_c_return_ready_test",
        "status": "RUNNING",
        "requested_cycles": int(args.return_ready_cycles),
        "chain_under_test": "LEFT_SAFE_RETREAT_C -> LEFT_RETURN_READY -> LEFT_READY_CHECK",
        "initialization_path": "LEFT_INITIAL_READY_PATH (setup only)",
        "return_alphas": list(LEFT_RETURN_READY_ALPHAS),
        "return_goal": list(LEFT_READY_GOAL),
        "forbidden_return_waypoint": list(LEFT_INITIAL_READY_PATH[0]),
        "tests": [],
    }
    runner: ExpertStateRunner | None = None
    try:
        left_bundle = make_left_bundle()
        setup_monitor = MotionMonitor(enabled=motion_gate_enabled(args))
        report["motion_gate_config"] = dict(setup_monitor.config)
        runner = ExpertStateRunner(0, setup_monitor, report_path, nonblocking_motion=bool(args.nonblocking_motion))
        go_left_initial_ready(runner, left_bundle.left_arm)
        report["setup_states"] = list(runner.states)

        retreat = vertical_retreat_from_place(LEFT_PLACE_POSES["C"])
        for cycle in range(1, int(args.return_ready_cycles) + 1):
            monitor = MotionMonitor(enabled=motion_gate_enabled(args))
            runner = ExpertStateRunner(cycle, monitor, report_path, nonblocking_motion=bool(args.nonblocking_motion))
            test = {
                "cycle": cycle,
                "status": "RUNNING",
                "safe_retreat_pose": retreat,
                "states": runner.states,
            }
            report["tests"].append(test)
            runner.enter("LEFT_SAFE_RETREAT", nut="C")
            retreat_row = move_pose(
                runner,
                left_bundle.left_arm,
                retreat,
                "LEFT_SAFE_RETREAT_C",
            )
            runner.pass_state({"move": retreat_row, "safe_retreat_pose": retreat})
            go_left_return_ready(runner, left_bundle.left_arm)
            test["status"] = "PASS"
            test["states"] = runner.states

        report["status"] = "PASS"
        report["passed_cycles"] = len(report["tests"])
        return_code = 0
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if runner is not None and not isinstance(exc, StateFault):
            exc = runner.fault(exc)
        report.update({
            "status": "FAILED",
            "passed_cycles": sum(test["status"] == "PASS" for test in report["tests"]),
            "failed_phase": runner.state if runner is not None else "SETUP",
            "failure_reason": repr(exc),
            "fault_snapshot": runner.fault_path if runner is not None else None,
        })
        return_code = 130 if interrupted else 1
    finally:
        shutdown_left_bundle(left_bundle)
        report["finished_at"] = now_text()
        write_json(report_path, report)
        print(f"Result: {report['status']}\nReport: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def plan_summary(
    sequence: tuple[str, ...],
    reset_to_nominal: bool,
    deterministic_place_path: bool = False,
) -> dict[str, Any]:
    dynamic_per_nut = [
        "RIGHT_APPROACH", "RIGHT_THUMB_TUCK", "RIGHT_GRASP", "RIGHT_GRASP_FORCE", "RIGHT_LIFT",
        "RIGHT_RELEASE", "RIGHT_SAFE_RETREAT", "RIGHT_READY", "RIGHT_READY_CHECK",
        "LEFT_VISION", "LEFT_APPROACH", "LEFT_THUMB_TUCK", "LEFT_DESCENT",
        "LEFT_GRASP", "LEFT_GRASP_FORCE", "LEFT_SAFE_LIFT",
    ]
    per_nut = dynamic_per_nut + (
        [
            "LEFT_FIXED_ENTRY_ALIGN", "LEFT_FIXED_TRANSPORT", "LEFT_FIXED_PLACE", "LEFT_RELEASE",
            "LEFT_FIXED_RETREAT_IF_NONTERMINAL", "LEFT_RETURN_READY_IF_NONTERMINAL",
            "LEFT_READY_CHECK_IF_NONTERMINAL",
        ]
        if deterministic_place_path
        else [
            "LEFT_PLACE_ABOVE", "LEFT_PLACE", "LEFT_SAFE_RETREAT",
            "LEFT_RETURN_READY", "LEFT_READY_CHECK",
        ]
    )
    return {
        "status": "PLAN_ONLY_NO_RABO_SDK",
        "sequence": list(sequence),
        "use_scene_initial_pose": not reset_to_nominal,
        "randomize_nuts": False,
        "set_entity_pose_on_episode_init": reset_to_nominal,
        "right_pick_uses_vision": False,
        "right_pick_target_source": "KNOWN_FIXED_NUT_WORLD_POSE with VERIFIED Nut B template",
        "left_pick_uses_post_release_vision": True,
        "deterministic_place_path": deterministic_place_path,
        "deterministic_place_reference_episode": (
            LEFT_FIXED_PLACE_REFERENCE_EPISODE if deterministic_place_path else None
        ),
        "episode_init": (
            "deterministic SetEntityPose A/B/C to VERIFIED_SCENE_FIXED_POSE"
            if reset_to_nominal
            else "use Rabo scene initial state; no SetEntityPose"
        ),
        "initial_ready": ["RIGHT_READY", "RIGHT_READY_CHECK", "LEFT_READY", "LEFT_READY_CHECK"],
        "per_nut_template": per_nut,
        "final": (
            ["FINAL_NUT_PLACE", "FINAL_NUT_RELEASE", "DONE"]
            if deterministic_place_path
            else ["RIGHT_READY", "RIGHT_READY_CHECK", "LEFT_RETURN_READY", "LEFT_READY_CHECK", "DONE"]
        ),
        "left_initial_ready_path": [list(joints) for joints in LEFT_INITIAL_READY_PATH],
        "left_return_ready": {
            "source": "actual joints read after LEFT_SAFE_RETREAT",
            "goal": list(LEFT_READY_GOAL),
            "alphas": list(LEFT_RETURN_READY_ALPHAS),
        },
    }


def run(args: argparse.Namespace) -> int:
    sequence = parse_sequence_arg(args.sequence)
    validate_static_contract(sequence)
    gate_enabled = motion_gate_enabled(args)
    print(f"[MOTION_GATE] enabled={gate_enabled}", flush=True)
    if args.plan_only:
        print(json.dumps(
            plan_summary(
                sequence,
                bool(args.reset_to_nominal),
                bool(args.deterministic_place_path),
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.test_left_ready:
        return run_left_ready_test(args)
    if args.test_left_return_ready:
        return run_left_return_ready_c_test(args)

    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"three_nut_v2_{started.strftime('%Y%m%d_%H%M%S')}.json"
    coordinator = None
    if args.coordinated_execution:
        trace_stamp = started.strftime("%Y%m%d_%H%M%S_%f")
        coordinator = ExecutionCoordinator(PROJECT_ROOT / "reports" / "execution_trace" / f"{trace_stamp}.jsonl")
    report: dict[str, Any] = {
        "experiment": "three_nut_closed_loop_v2",
        "timestamp": started.isoformat(timespec="seconds"),
        "status": "RUNNING",
        "sequence": list(sequence),
        "use_scene_initial_pose": not bool(args.reset_to_nominal),
        "randomize_nuts": False,
        "set_entity_pose_on_episode_init": bool(args.reset_to_nominal),
        "reset_to_nominal": bool(args.reset_to_nominal),
        "nominal_pose_status": KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
        "known_fixed_nut_world_poses": {
            key: pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE[key]) for key in ("A", "B", "C")
        },
        "right_pick_uses_vision": False,
        "right_pick_target_source": "KNOWN_FIXED_NUT_WORLD_POSE with VERIFIED Nut B template",
        "left_pick_uses_post_release_vision": True,
        "deterministic_place_path": bool(args.deterministic_place_path),
        "motion_gate_enabled": gate_enabled,
        "nonblocking_motion": bool(args.nonblocking_motion),
        "coordinated_execution": bool(args.coordinated_execution),
        "geometry_policy": "VERIFIED Nut B grasp template for A/B/C; vertical RIGHT_APPROACH added",
        "threshold_status": "PROVISIONAL_THRESHOLD_REQUIRES_RABO_TUNING",
        "trials": [],
    }
    pose_setter = right_bundle = left_bundle = None
    probe: IntegratedRecordingProbe | None = None
    try:
        if args.reset_to_nominal:
            pose_setter = make_pose_setter()
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        for trial_id in range(1, args.trials + 1):
            monitor = MotionMonitor(enabled=gate_enabled)
            report.setdefault("motion_gate_config", dict(monitor.config))
            runner = ExpertStateRunner(trial_id, monitor, report_path, nonblocking_motion=bool(args.nonblocking_motion), coordinator=coordinator)
            trial = {"trial_id": trial_id, "status": "RUNNING", "states": runner.states}
            report["trials"].append(trial)
            try:
                runner.enter("EPISODE_INIT")
                resets = reset_nuts_to_nominal(pose_setter) if args.reset_to_nominal else []
                runner.pass_state({
                    "use_scene_initial_pose": not bool(args.reset_to_nominal),
                    "randomize_nuts": False,
                    "set_entity_pose_on_episode_init": bool(args.reset_to_nominal),
                    "nut_initialization": resets,
                    "world_reset_used": False,
                })
                go_right_ready(runner, right_bundle.right_arm)
                go_left_initial_ready(runner, left_bundle.left_arm)
                runner.enter("READY_CHECK")
                runner.pass_state({"right_ready": True, "left_ready": True})

                if args.recording_probe and probe is None:
                    output = PROJECT_ROOT / "reports" / "act_recording_probe" / started.strftime("%Y%m%d_%H%M%S")
                    probe = IntegratedRecordingProbe(output_dir=output, target_fps=float(args.recording_probe_fps))
                    probe.start(
                        left_arm=left_bundle.left_arm,
                        right_arm=right_bundle.right_arm,
                        left_hand=left_bundle.left_hand,
                        right_hand=right_bundle.right_hand,
                    )
                    probe.mark_phase("START_RECORDING", event="ENTER", trial_id=trial_id)

                for index, key in enumerate(sequence):
                    if probe is not None:
                        probe.mark_phase("NUT", event="ENTER", nut=key)
                    runtime_nut_world_xyz = pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE[key])[:3]
                    released_xyz = execute_right_transfer(
                        runner,
                        key,
                        right_bundle,
                        runtime_nut_world_xyz=runtime_nut_world_xyz,
                        nut_xyz_source=KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
                        vision_radius_m=float(args.vision_target_radius_m),
                        settle_after_release_s=float(args.settle_after_release_s),
                    )
                    execute_left_pick_place(
                        runner,
                        key,
                        left_bundle,
                        released_xyz,
                        deterministic_place_path=bool(args.deterministic_place_path),
                        terminal_after_release=(
                            bool(args.deterministic_place_path) and index == len(sequence) - 1
                        ),
                    )
                    if probe is not None:
                        probe.mark_phase("NUT", event="EXIT", nut=key, success=True)

                if not args.deterministic_place_path:
                    go_right_ready(runner, right_bundle.right_arm)
                    go_left_return_ready(runner, left_bundle.left_arm)
                runner.enter("DONE")
                runner.pass_state({"completed_nuts": list(sequence)})
                trial["status"] = "PASS"
            except BaseException as exc:
                if not isinstance(exc, StateFault):
                    exc = runner.fault(exc)
                trial.update({
                    "status": "FAILED",
                    "failed_nut": runner.nut,
                    "failed_phase": runner.state,
                    "failure_reason": repr(exc),
                    "failure_detail": getattr(exc, "context", {}),
                    "fault_snapshot": runner.fault_path,
                })
                raise
            finally:
                trial["states"] = runner.states
                write_json(report_path, report)
            report["status"] = "PASS"
        report["overall_success"] = True
        return_code = 0
    except KeyboardInterrupt as exc:
        report.update({"status": "INTERRUPTED", "overall_success": False, "failure_reason": repr(exc)})
        return_code = 130
    except BaseException as exc:
        report.update({"status": "FAILED", "overall_success": False, "failure_reason": repr(exc)})
        return_code = 1
    finally:
        if coordinator is not None:
            coordinator.write_summary(coordinator.trace_path.with_name(coordinator.trace_path.stem + "_summary.json"))
        if probe is not None:
            try:
                report["recording_probe"] = probe.stop(task_status=report["status"])
            except Exception as exc:
                report["recording_probe_stop_error"] = repr(exc)
        shutdown_bundle(right_bundle)
        shutdown_left_bundle(left_bundle)
        shutdown_pose_setter(pose_setter)
        report["finished_at"] = now_text()
        write_json(report_path, report)
        print(f"Result: {report['status']}\nReport: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ready-gated fail-stop C -> B -> A three-Nut Expert V2.")
    parser.add_argument("--headless", action="store_true", default=True, help="Headless mode (default; only supported mode).")
    parser.add_argument("--trials", type=int, default=1, help="Number of complete episodes (default: 1).")
    parser.add_argument("--sequence", default="C,B,A", help="Nut sequence, e.g. C, C,B, or C,B,A (default: C,B,A).")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Explicitly request the default-enabled standalone MotionMonitor gate.",
    )
    parser.add_argument(
        "--no-motion-gate",
        action="store_true",
        help="Disable MotionMonitor sampling and all monitor-derived fail-stop gates.",
    )
    parser.add_argument(
        "--deterministic-place-path",
        action="store_true",
        help=(
            "Use recorded actual_joint waypoints for fixed C/B transport/place; "
            "the final Nut ends after release. Default keeps Cartesian fallback."
        ),
    )
    parser.add_argument(
        "--nonblocking-motion",
        action="store_true",
        help="Send each arm motion once with blocking=False and wait for joint settling.",
    )
    parser.add_argument(
        "--coordinated-execution",
        action="store_true",
        help="Serialize multi-device SDK invocations and emit an execution trace.",
    )
    parser.add_argument("--test-left-ready", action="store_true", help="Run only the no-Nut LEFT_READY candidate motion-chain test.")
    parser.add_argument(
        "--test-left-return-ready",
        action="store_true",
        help="Run only LEFT_SAFE_RETREAT_C -> segmented LEFT_RETURN_READY -> LEFT_READY_CHECK.",
    )
    parser.add_argument(
        "--return-ready-cycles",
        type=int,
        default=5,
        help="Cycles for --test-left-return-ready (default: 5).",
    )
    parser.add_argument(
        "--reset-to-nominal",
        action="store_true",
        help="Debug only: deterministically SetEntityPose A/B/C to VERIFIED_SCENE_FIXED_POSE at EPISODE_INIT.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Print the state plan without importing or driving the Rabo SDK.")
    parser.add_argument("--settle-after-release-s", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S)
    parser.add_argument("--vision-target-radius-m", type=float, default=DEFAULT_VISION_TARGET_RADIUS_M)
    parser.add_argument("--recording-probe", "--probe", dest="recording_probe", action="store_true")
    parser.add_argument("--recording-probe-fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.monitor and args.no_motion_gate:
        parser.error("--monitor and --no-motion-gate are mutually exclusive")
    try:
        parse_sequence_arg(args.sequence)
    except ThreeNutClosedLoopError as exc:
        parser.error(str(exc))
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.return_ready_cycles < 1:
        parser.error("--return-ready-cycles must be >= 1")
    if args.trials > 1 and not args.reset_to_nominal:
        parser.error(
            "--trials > 1 requires --reset-to-nominal; without an explicit reset only the first "
            "episode can use the Rabo scene initial state"
        )
    minimum_wait = RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S
    if args.settle_after_release_s < minimum_wait:
        parser.error(f"--settle-after-release-s must be at least {minimum_wait:g}s")
    if args.vision_target_radius_m <= 0 or args.recording_probe_fps <= 0:
        parser.error("vision radius and recording FPS must be > 0")
    if args.deterministic_place_path:
        deterministic_sequence = parse_sequence_arg(args.sequence)
        unsupported = [key for key in deterministic_sequence if key not in LEFT_FIXED_PLACE_JOINT_PATHS]
        if unsupported:
            parser.error(f"deterministic place path is unavailable for: {unsupported}")
        missing_retreat = [
            key
            for key in deterministic_sequence[:-1]
            if "retreat" not in LEFT_FIXED_PLACE_JOINT_PATHS[key]
        ]
        if missing_retreat:
            parser.error(f"non-terminal deterministic retreat is unavailable for: {missing_retreat}")
    return args


def motion_gate_enabled(args: argparse.Namespace) -> bool:
    """Standalone Expert keeps its historical gate unless explicitly disabled."""
    return not bool(args.no_motion_gate)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
