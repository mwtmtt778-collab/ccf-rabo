#!/usr/bin/env python3
"""Test whether observed ``state[t+1]`` can serve as an ACT action candidate.

Record mode runs only the existing V2 Nut-B right-arm grasp/lift primitives and
samples the established 26D state schema at 5 Hz.  Replay mode consumes only
the saved candidate actions; Expert Cartesian targets and grasp-force events
are deliberately unavailable to the replay path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "state_trajectory_replay"
STATE_SCHEMA = [
    {"slice": "0:7", "name": "left_arm", "source": "get_joint_angles()"},
    {"slice": "7:14", "name": "right_arm", "source": "get_joint_angles()"},
    {"slice": "14:20", "name": "left_hand", "source": "get_clench()"},
    {"slice": "20:26", "name": "right_hand", "source": "get_clench()"},
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    KNOWN_FIXED_NUT_WORLD_POSE,
    RIGHT_GRASP_FORCE,
)
from agents.three_nut_expert.expert import (  # noqa: E402
    build_right_safe_lift_pose,
    compute_right_approach_pose,
    compute_right_grasp_pose,
    pose_to_list,
)
from expert.left_nut_grasp_planner import result_failed  # noqa: E402
from tools.motion_monitor import MotionMonitor  # noqa: E402
from tools.probe_act_recording_sources import read_state26_timed  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    make_right_bundle,
    shutdown_bundle,
)
from tools.test_three_nut_closed_loop_v2 import (  # noqa: E402
    ExpertStateRunner,
    go_right_ready,
    move_pose,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def sdk_signature(value: Any) -> dict[str, Any]:
    result = {"available": value is not None, "signature": "UNKNOWN", "inspection_error": None}
    if value is None:
        return result
    try:
        result["signature"] = str(inspect.signature(value))
    except Exception as exc:
        result["inspection_error"] = repr(exc)
    return result


class StateSampler:
    def __init__(self, devices: dict[str, Any], fps: float) -> None:
        self.devices = devices
        self.fps = fps
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.overrun_periods = 0
        self.start_monotonic = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.start_monotonic = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="state26-5hz-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        interval = 1.0 / self.fps
        deadline = self.start_monotonic
        while not self._stop.is_set():
            now = time.monotonic()
            if now < deadline:
                self._stop.wait(min(deadline - now, 0.01))
                continue
            read_started = time.monotonic()
            try:
                state, components, durations = read_state26_timed(self.devices)
                read_finished = time.monotonic()
                self.samples.append({
                    "timestamp": read_finished - self.start_monotonic,
                    "read_duration_s": read_finished - read_started,
                    "state": state,
                    "components": components,
                    "component_read_duration_s": durations,
                })
            except Exception as exc:
                read_finished = time.monotonic()
                self.errors.append({
                    "timestamp": read_finished - self.start_monotonic,
                    "read_duration_s": read_finished - read_started,
                    "error": repr(exc),
                })
            deadline += interval
            if read_finished > deadline:
                missed = max(1, int((read_finished - deadline) / interval) + 1)
                self.overrun_periods += missed
                deadline += missed * interval


class EventLog:
    def __init__(self, sampler: StateSampler) -> None:
        self.sampler = sampler
        self.phases: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []

    def phase(self, name: str) -> None:
        self.phases.append({"phase": name, "timestamp": time.monotonic() - self.sampler.start_monotonic})

    def command(self, phase: str, method: str, payload: Any) -> None:
        self.commands.append({
            "phase": phase,
            "method": method,
            "payload": payload,
            "timestamp": time.monotonic() - self.sampler.start_monotonic,
            "diagnostic_only": True,
        })


def effective_hz(timestamps: np.ndarray) -> float | None:
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return None
    return float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))


def make_devices(right_bundle: Any, left_bundle: Any) -> dict[str, Any]:
    return {
        "left_arm": left_bundle.left_arm,
        "right_arm": right_bundle.right_arm,
        "left_hand": left_bundle.left_hand,
        "right_hand": right_bundle.right_hand,
    }


def record(args: argparse.Namespace) -> int:
    if not args.execute:
        print(json.dumps({
            "mode": "record",
            "execute": False,
            "status": "PLAN_ONLY_NO_SDK_CLIENTS_CREATED",
            "chain": ["RIGHT_READY", "RIGHT_APPROACH_B", "RIGHT_THUMB_TUCK", "RIGHT_GRASP_B", "RIGHT_GRASP_FORCE", "RIGHT_LIFT_B", "HOLD"],
            "candidate_action": "states[t+1]",
        }, indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    right_bundle = left_bundle = None
    sampler: StateSampler | None = None
    report: dict[str, Any] = {"mode": "record", "status": "RUNNING", "started_at": now_iso()}
    try:
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        devices = make_devices(right_bundle, left_bundle)
        sampler = StateSampler(devices, args.fps)
        sampler.start()
        events = EventLog(sampler)
        monitor = MotionMonitor(enabled=True, output_root=output_dir / "motion_monitor")
        runner = ExpertStateRunner(1, monitor, output_dir / "expert_report.json")

        events.phase("RIGHT_READY")
        events.command("RIGHT_READY", "right_arm.move_joints", "V2 RIGHT_OBSERVATION_JOINTS")
        go_right_ready(runner, right_bundle.right_arm)

        nut_pose = KNOWN_FIXED_NUT_WORLD_POSE["B"]
        nut_xyz = [float(nut_pose.x), float(nut_pose.y), float(nut_pose.z)]
        grasp_pose = compute_right_grasp_pose(nut_xyz)
        approach_pose = compute_right_approach_pose(grasp_pose)

        events.phase("RIGHT_APPROACH_B")
        events.command("RIGHT_APPROACH_B", "right_arm.move_to", pose_to_list(approach_pose))
        runner.enter("RIGHT_APPROACH", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, approach_pose, "RIGHT_APPROACH_B")})

        events.phase("RIGHT_THUMB_TUCK")
        events.command("RIGHT_THUMB_TUCK", "right_hand.clench", {"thumb_rotation": 1.0})
        runner.enter("RIGHT_THUMB_TUCK", nut="B")
        result = runner.hand_action(
            label="RIGHT_THUMB_TUCK", command_method="right_hand.clench",
            fn=lambda: right_bundle.right_hand.clench(thumb_rotation=1.0),
        )
        runner.pass_state({"sdk_return": repr(result)})

        events.phase("RIGHT_GRASP_B")
        events.command("RIGHT_GRASP_B", "right_arm.move_to", pose_to_list(grasp_pose))
        runner.enter("RIGHT_GRASP", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, grasp_pose, "RIGHT_PICK_B")})

        events.phase("RIGHT_GRASP_FORCE")
        events.command("RIGHT_GRASP_FORCE", "right_hand.grasp_force", dict(RIGHT_GRASP_FORCE))
        runner.enter("RIGHT_GRASP_FORCE", nut="B")
        result = runner.hand_action(
            label="RIGHT_GRASP_FORCE", command_method="right_hand.grasp_force",
            fn=lambda: right_bundle.right_hand.grasp_force(**RIGHT_GRASP_FORCE),
        )
        if DEFAULT_HOLD_AFTER_GRASP_S > 0:
            time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
        runner.pass_state({"sdk_return": repr(result)})

        lift_pose = build_right_safe_lift_pose(grasp_pose)
        events.phase("RIGHT_LIFT_B")
        events.command("RIGHT_LIFT_B", "right_arm.move_to", pose_to_list(lift_pose))
        runner.enter("RIGHT_LIFT", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, lift_pose, "RIGHT_SAFE_LIFT_B")})
        events.phase("HOLD")
        time.sleep(args.hold_seconds)
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
    finally:
        if sampler is not None:
            sampler.stop()
        shutdown_left_bundle(left_bundle)
        shutdown_bundle(right_bundle)

    samples = sampler.samples if sampler is not None else []
    states = np.asarray([row["state"] for row in samples], dtype=np.float32).reshape((-1, 26))
    timestamps = np.asarray([row["timestamp"] for row in samples], dtype=np.float64)
    candidate_actions = states[1:].copy()
    np.savez_compressed(
        output_dir / "trajectory.npz",
        states=states,
        timestamps=timestamps,
        candidate_actions=candidate_actions,
    )
    phase_rows = events.phases if "events" in locals() else []
    command_rows = events.commands if "events" in locals() else []
    metadata = {
        "created_at": now_iso(), "git_commit": git_commit(), "nut": "B",
        "fps_requested": args.fps, "effective_state_fps": effective_hz(timestamps),
        "duration_s": float(timestamps[-1]) if len(timestamps) else 0.0,
        "state_schema": STATE_SCHEMA,
        "action_candidate_definition": "NEXT_OBSERVED_STATE",
        "candidate_action_formula": "candidate_actions[t] = states[t+1]",
        "expert_source": "tools/test_three_nut_closed_loop_v2.py imported primitives",
        "phase_timestamps": phase_rows,
        "expert_command_events": command_rows,
        "expert_command_events_role": "DIAGNOSTIC_ONLY_NOT_REPLAY_INPUT",
        "sdk_signatures": {
            "right_arm.move_joints": sdk_signature(getattr(getattr(right_bundle, "right_arm", None), "move_joints", None)),
            "right_hand.clench": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "clench", None)),
            "right_hand.grasp_force": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "grasp_force", None)),
        },
    }
    report.update({
        "finished_at": now_iso(), "output_dir": str(output_dir),
        "record_state_count": len(states), "record_effective_hz": effective_hz(timestamps),
        "state_read_errors": sampler.errors if sampler is not None else [],
        "state_sampler_overrun_periods": sampler.overrun_periods if sampler is not None else 0,
        "candidate_action_count": len(candidate_actions),
    })
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "COMPLETED" and len(states) >= 2 else 1


def validate_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"states", "timestamps", "candidate_actions"} - set(data.files)
        if missing:
            raise ValueError(f"trajectory missing arrays: {sorted(missing)}")
        states = np.asarray(data["states"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        actions = np.asarray(data["candidate_actions"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 26:
        raise ValueError(f"states must have shape [N,26], got {states.shape}")
    if timestamps.shape != (len(states),):
        raise ValueError(f"timestamps must have shape [{len(states)}], got {timestamps.shape}")
    if actions.shape != (max(0, len(states) - 1), 26):
        raise ValueError(f"candidate_actions must have shape [{max(0, len(states)-1)},26], got {actions.shape}")
    if not np.array_equal(actions, states[1:].astype(np.float64)):
        raise ValueError("candidate_actions are not exactly states[1:]")
    if not np.isfinite(states).all() or not np.isfinite(timestamps).all():
        raise ValueError("trajectory contains non-finite values")
    return states, timestamps, actions


def select_waypoints(actions: np.ndarray, arm_threshold: float, hand_threshold: float) -> list[int]:
    if not len(actions):
        return []
    selected = [0]
    anchor = actions[0]
    for index in range(1, len(actions)):
        arm_delta = float(np.max(np.abs(actions[index, 7:14] - anchor[7:14])))
        hand_delta = float(np.max(np.abs(actions[index, 20:26] - anchor[20:26])))
        if arm_delta >= arm_threshold or hand_delta >= hand_threshold:
            selected.append(index)
            anchor = actions[index]
    if selected[-1] != len(actions) - 1:
        selected.append(len(actions) - 1)
    return selected


def require_sdk_success(value: Any) -> None:
    failed, reason = result_failed(value)
    if failed:
        raise RuntimeError(f"SDK command failed: {reason}")


def replay(args: argparse.Namespace) -> int:
    states, timestamps, actions = validate_trajectory(args.trajectory)
    indices = select_waypoints(actions, args.arm_waypoint_threshold, args.hand_waypoint_threshold)
    plan = {
        "mode": "replay", "trajectory": str(args.trajectory), "execute": args.execute,
        "input_contract": "ONLY trajectory.npz:candidate_actions (= states[t+1])",
        "expert_commands_used": False, "replay_level": args.replay_level.upper() + "_REPLAY",
        "selected_waypoint_count": len(indices), "source_action_count": len(actions),
        "left_channels_commanded": False,
    }
    if not args.execute:
        print(json.dumps({**plan, "status": "PLAN_ONLY_NO_SDK_CLIENTS_CREATED"}, indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"replay_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    right_bundle = left_bundle = None
    report: dict[str, Any] = {**plan, "status": "RUNNING", "started_at": now_iso()}
    command_count = 0
    arm_ok = True
    hand_values_legal = bool(len(actions)) and bool(np.all((actions[:, 20:26] >= 0.0) & (actions[:, 20:26] <= 1.0)))
    hand_ok = hand_values_legal
    command_errors: list[dict[str, Any]] = []
    action_tick_count = 0
    schedule_lateness_s: list[float] = []
    started = time.monotonic()
    final_state: Sequence[float] | None = None
    try:
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        devices = make_devices(right_bundle, left_bundle)
        if args.replay_level == "geometry":
            monitor = MotionMonitor(enabled=True, output_root=output_dir / "motion_monitor")
            runner = ExpertStateRunner(1, monitor, output_dir / "replay_motion_report.json")
            last_arm: np.ndarray | None = None
            last_hand: np.ndarray | None = None
            for index in indices:
                action = actions[index]
                hand = action[20:26]
                arm = action[7:14]
                if hand_ok and (last_hand is None or np.max(np.abs(hand - last_hand)) >= args.hand_waypoint_threshold):
                    try:
                        runner.hand_action(
                            label=f"STATE_REPLAY_HAND_{index:06d}",
                            command_method="right_hand.clench",
                            fn=lambda hand=hand.copy(): right_bundle.right_hand.clench(*hand.tolist()),
                        )
                        command_count += 1
                        last_hand = hand.copy()
                    except Exception as exc:
                        hand_ok = False
                        command_errors.append({"action_index": index, "component": "right_hand", "error": repr(exc)})
                if last_arm is None or np.max(np.abs(arm - last_arm)) >= args.arm_waypoint_threshold:
                    try:
                        target = arm.tolist()
                        runner.arm_action(
                            label=f"STATE_REPLAY_ARM_{index:06d}",
                            arm=right_bundle.right_arm,
                            command_method="move_joints",
                            target_joint=target,
                            fn=lambda target=target: right_bundle.right_arm.move_joints(target),
                        )
                        command_count += 1
                        last_arm = arm.copy()
                    except Exception as exc:
                        arm_ok = False
                        command_errors.append({"action_index": index, "component": "right_arm", "error": repr(exc)})
                        break
                action_tick_count += 1
        else:
            timing_started = time.monotonic()
            first_source_time = float(timestamps[1]) if len(timestamps) > 1 else 0.0
            for index, action in enumerate(actions):
                due = timing_started + float(timestamps[index + 1] - first_source_time)
                remaining = due - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                schedule_lateness_s.append(max(0.0, time.monotonic() - due))
                hand = action[20:26]
                arm = action[7:14]
                if hand_ok:
                    try:
                        value = right_bundle.right_hand.clench(*hand.tolist(), blocking=False)
                        require_sdk_success(value)
                        command_count += 1
                    except Exception as exc:
                        hand_ok = False
                        command_errors.append({"action_index": index, "component": "right_hand", "error": repr(exc)})
                try:
                    value = right_bundle.right_arm.move_joints(arm.tolist(), blocking=False)
                    require_sdk_success(value)
                    command_count += 1
                except Exception as exc:
                    arm_ok = False
                    command_errors.append({"action_index": index, "component": "right_arm", "error": repr(exc)})
                    break
                action_tick_count += 1
        time.sleep(args.final_settle_seconds)
        final_state, _, _ = read_state26_timed(devices)
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        if not hand_values_legal:
            hand_ok = False
    finally:
        elapsed = time.monotonic() - started
        shutdown_left_bundle(left_bundle)
        shutdown_bundle(right_bundle)

    expected = actions[-1] if len(actions) else None
    error = None if final_state is None or expected is None else np.abs(np.asarray(final_state) - expected)
    right_arm_error = None if error is None else float(np.max(error[7:14]))
    right_hand_error = None if error is None else float(np.max(error[20:26]))
    final_error = None if error is None else error.tolist()
    arm_pass = arm_ok and right_arm_error is not None and right_arm_error <= args.final_arm_error_threshold
    hand_pass = hand_ok and right_hand_error is not None and right_hand_error <= args.final_hand_error_threshold
    timing_span = elapsed - args.final_settle_seconds
    timing_effective_hz = action_tick_count / timing_span if timing_span > 0 else None
    five_hz = bool(
        args.replay_level == "timing"
        and arm_pass and hand_pass and action_tick_count == len(actions)
        and timing_effective_hz is not None and timing_effective_hz >= 0.9 * args.fps
        and not command_errors
    )
    if arm_pass and hand_pass and five_hz:
        conclusion = "STATE_NEXT_ACTION_FEASIBLE"
    elif arm_pass and hand_pass:
        conclusion = "STATE_NEXT_ACTION_GEOMETRY_ONLY"
    elif arm_pass and not hand_pass:
        conclusion = "ARM_FEASIBLE_HAND_NOT_FEASIBLE"
    else:
        conclusion = "STATE_NEXT_ACTION_NOT_FEASIBLE"
    report.update({
        "finished_at": now_iso(), "output_dir": str(output_dir),
        "record_state_count": len(states), "record_effective_hz": effective_hz(timestamps),
        "replay_command_count": command_count, "replay_duration": elapsed,
        "replay_effective_command_hz": command_count / elapsed if elapsed > 0 else None,
        "replay_action_tick_count": action_tick_count,
        "replay_effective_action_hz": timing_effective_hz if args.replay_level == "timing" else None,
        "timing_schedule_lateness_ms": {
            "mean": float(np.mean(schedule_lateness_s) * 1000.0) if schedule_lateness_s else None,
            "p95": float(np.percentile(schedule_lateness_s, 95) * 1000.0) if schedule_lateness_s else None,
            "max": float(np.max(schedule_lateness_s) * 1000.0) if schedule_lateness_s else None,
        },
        "arm_geometry_replay_possible": arm_pass,
        "hand_state_replay_possible": hand_pass,
        "geometry_replay_possible": arm_pass and hand_pass,
        "hand_recorded_values_legal_for_clench_0_to_1": hand_values_legal,
        "five_hz_realtime_replay_possible": five_hz,
        "five_hz_reason": (
            "verified by non-blocking blocking=False replay and measured schedule/final-state gates"
            if five_hz else
            "not tested in geometry mode" if args.replay_level == "geometry" else
            "timing replay did not satisfy command, schedule, and final-state gates"
        ),
        "final_state_error_26d": final_error,
        "right_arm_final_error": right_arm_error,
        "right_hand_final_error": right_hand_error,
        "replay_command_errors": command_errors,
        "physical_pick_success": "UNKNOWN_REQUIRES_VISUAL_CONFIRMATION",
        "conclusion": conclusion,
        "pure_state_replay": True,
        "grasp_force_used_during_replay": False,
        "sdk_signatures": {
            "right_arm.move_joints": sdk_signature(getattr(getattr(right_bundle, "right_arm", None), "move_joints", None)),
            "right_hand.clench": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "clench", None)),
        },
    })
    write_json(output_dir / "metadata.json", {
        "created_at": now_iso(), "git_commit": git_commit(), "state_schema": STATE_SCHEMA,
        "source_trajectory": str(args.trajectory),
        "selected_action_indices": indices if args.replay_level == "geometry" else list(range(len(actions))),
        "action_candidate_definition": "NEXT_OBSERVED_STATE",
        "replay_order_within_selected_sample": (
            "right_hand.clench then blocking right_arm.move_joints when changed"
            if args.replay_level == "geometry" else
            "right_hand.clench(blocking=False) then right_arm.move_joints(blocking=False) at each saved tick"
        ),
        "expert_command_events_loaded": False,
        "public_sdk_evidence": {
            "arm": ".ai/reference/rabo_docs/LinkerArmA7 · rabo (2026_8_13 18：31：39).html: move_joints(joint_angles, blocking=True)",
            "hand": ".ai/reference/rabo_docs/LinkerHandO6Left · rabo (2026_8_13 18：33：36).html: clench(..., blocking=True); get_clench is its normalized inverse",
            "grasp_force": "separate server-side force-control action; intentionally unavailable to pure state replay",
        },
    })
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "COMPLETED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--nut", choices=("B",), default="B", help="record mode is intentionally limited to Nut B")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--trajectory", type=Path, help="trajectory.npz created by record mode")
    parser.add_argument("--execute", action="store_true", help="create SDK clients and command the robot; omitted means safe plan-only")
    parser.add_argument(
        "--replay-level", choices=("geometry", "timing"), default="geometry",
        help="geometry uses blocking monitored waypoints; timing sends saved 5Hz targets with documented blocking=False",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--final-settle-seconds", type=float, default=1.0)
    parser.add_argument("--arm-waypoint-threshold", type=float, default=0.03)
    parser.add_argument("--hand-waypoint-threshold", type=float, default=0.03)
    parser.add_argument("--final-arm-error-threshold", type=float, default=0.15)
    parser.add_argument("--final-hand-error-threshold", type=float, default=0.10)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive")
    if args.mode == "replay" and args.trajectory is None:
        parser.error("--trajectory is required in replay mode")
    if args.mode == "record" and args.trajectory is not None:
        parser.error("--trajectory is only valid in replay mode")
    return record(args) if args.mode == "record" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
