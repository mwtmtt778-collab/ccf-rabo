#!/usr/bin/env python3
"""Record and replay a right-side Hybrid 28D ACT action feasibility test.

The action is ``[next_observed_state_26d, left_grasp_mode,
right_grasp_mode]``.  Replay loads only ``trajectory.npz`` as trajectory input;
Expert Cartesian geometry, phase timestamps, and command audit events are not
loaded by the replay path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "hybrid_action_replay"
ACTION_CONTRACT = "HYBRID_NEXT_STATE_PLUS_GRASP_MODE_V1"
STATE_DIM = 26
ACTION_DIM = 28
ACTION_SCHEMA = [
    {"slice": "0:7", "name": "left_arm_next_joint_state"},
    {"slice": "7:14", "name": "right_arm_next_joint_state"},
    {"slice": "14:20", "name": "left_hand_next_clench_state"},
    {"slice": "20:26", "name": "right_hand_next_clench_state"},
    {"index": 26, "name": "left_grasp_mode", "values": {"0": "position_clench", "1": "force_grasp"}},
    {"index": 27, "name": "right_grasp_mode", "values": {"0": "position_clench", "1": "force_grasp"}},
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import RIGHT_GRASP_FORCE  # noqa: E402
from expert.left_nut_grasp_planner import result_failed  # noqa: E402
from tools.motion_monitor import MotionMonitor  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_right_release_stability import make_right_bundle, shutdown_bundle  # noqa: E402
from tools.test_state_trajectory_replay import (  # noqa: E402
    STATE_SCHEMA,
    StateSampler,
    effective_hz,
    git_commit,
    make_devices,
    sdk_signature,
    write_json,
)
from tools.test_three_nut_closed_loop_v2 import ExpertStateRunner  # noqa: E402


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def distribution_ms(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array) * 1000.0) if len(array) else None,
        "median": float(np.median(array) * 1000.0) if len(array) else None,
        "p90": float(np.percentile(array, 90) * 1000.0) if len(array) else None,
        "p95": float(np.percentile(array, 95) * 1000.0) if len(array) else None,
        "max": float(np.max(array) * 1000.0) if len(array) else None,
    }


def require_sdk_success(value: Any) -> None:
    failed, reason = result_failed(value)
    if failed:
        raise RuntimeError(f"SDK command failed: {reason}")


class PersistentModeAudit:
    """Timestamp persistent mode changes exactly at hand command invocation."""

    def __init__(self, sampler: StateSampler) -> None:
        self.sampler = sampler
        self.events: list[dict[str, Any]] = []
        self.phases: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._modes = [0, 0]

    def _timestamp(self) -> float:
        return time.monotonic() - self.sampler.start_monotonic

    def phase(self, name: str) -> None:
        with self._lock:
            self.phases.append({"phase": name, "timestamp": self._timestamp()})

    def command(
        self,
        phase: str,
        method: str,
        payload: Any,
        *,
        hand: str | None = None,
        grasp_mode_after: int | None = None,
    ) -> None:
        with self._lock:
            before = list(self._modes)
            if grasp_mode_after is not None:
                if hand == "left":
                    self._modes[0] = int(grasp_mode_after)
                elif hand == "right":
                    self._modes[1] = int(grasp_mode_after)
                else:
                    raise ValueError("mode-changing command requires left/right hand")
            self.events.append({
                "phase": phase,
                "method": method,
                "payload": payload,
                "timestamp": self._timestamp(),
                "persistent_grasp_mode_before": before,
                "persistent_grasp_mode_after": list(self._modes),
                "mode_change_source": "ACTUAL_HAND_COMMAND_INVOCATION" if grasp_mode_after is not None else None,
                "diagnostic_only": True,
            })


def align_modes_to_states(timestamps: np.ndarray, command_events: list[dict[str, Any]]) -> np.ndarray:
    """Apply the latest persistent command mode whose event time is <= state time."""
    modes = np.zeros((len(timestamps), 2), dtype=np.float32)
    mode_events = sorted(
        (
            event for event in command_events
            if event.get("mode_change_source") == "ACTUAL_HAND_COMMAND_INVOCATION"
        ),
        key=lambda event: float(event["timestamp"]),
    )
    current = np.zeros(2, dtype=np.float32)
    event_index = 0
    for state_index, state_time in enumerate(timestamps):
        while event_index < len(mode_events) and float(mode_events[event_index]["timestamp"]) <= float(state_time):
            current = np.asarray(mode_events[event_index]["persistent_grasp_mode_after"], dtype=np.float32)
            event_index += 1
        modes[state_index] = current
    return modes


def build_hybrid_actions(states: np.ndarray, modes_at_state: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    modes_at_state = np.asarray(modes_at_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"states must have shape [N,{STATE_DIM}], got {states.shape}")
    if modes_at_state.shape != (len(states), 2):
        raise ValueError(f"grasp_modes_at_state must have shape [{len(states)},2], got {modes_at_state.shape}")
    return np.concatenate((states[1:], modes_at_state[1:]), axis=1)


def transition_counts(modes_at_state: np.ndarray) -> dict[str, int]:
    if len(modes_at_state) < 2:
        return {"total": 0, "right_rising": 0, "right_falling": 0}
    deltas = np.diff(np.asarray(modes_at_state)[:, 1])
    rising = int(np.count_nonzero(deltas == 1))
    falling = int(np.count_nonzero(deltas == -1))
    return {"total": rising + falling, "right_rising": rising, "right_falling": falling}


def validate_contract_arrays(
    states: np.ndarray,
    timestamps: np.ndarray,
    modes_at_state: np.ndarray,
    hybrid_actions: np.ndarray,
) -> dict[str, Any]:
    states = np.asarray(states)
    timestamps = np.asarray(timestamps)
    modes_at_state = np.asarray(modes_at_state)
    hybrid_actions = np.asarray(hybrid_actions)
    errors: list[str] = []
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        errors.append(f"states shape is {states.shape}, expected [N,26]")
    if timestamps.shape != (len(states),):
        errors.append(f"timestamps shape is {timestamps.shape}, expected [{len(states)}]")
    if modes_at_state.shape != (len(states), 2):
        errors.append(f"grasp_modes_at_state shape is {modes_at_state.shape}, expected [{len(states)},2]")
    expected_action_shape = (max(0, len(states) - 1), ACTION_DIM)
    if hybrid_actions.shape != expected_action_shape:
        errors.append(f"hybrid_actions shape is {hybrid_actions.shape}, expected {expected_action_shape}")
    shapes_ok = not errors
    if shapes_ok:
        if not np.isfinite(states).all() or not np.isfinite(timestamps).all() or not np.isfinite(hybrid_actions).all():
            errors.append("arrays contain NaN or Inf")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            errors.append("timestamps are not strictly increasing")
        if not np.allclose(hybrid_actions[:, :26], states[1:], rtol=0.0, atol=0.0):
            errors.append("hybrid_actions[:,0:26] != states[1:]")
        flags = hybrid_actions[:, 26:28]
        if not np.all(np.isin(flags, (0.0, 1.0))):
            errors.append("hybrid grasp flags contain values outside {0,1}")
        if not np.all(np.isin(modes_at_state, (0.0, 1.0))):
            errors.append("grasp_modes_at_state contains values outside {0,1}")
        if not np.all(flags[:, 0] == 0.0):
            errors.append("left_grasp_mode is not always 0")
        if not np.allclose(flags, modes_at_state[1:], rtol=0.0, atol=0.0):
            errors.append("hybrid action modes != grasp_modes_at_state[1:]")
        if not np.all((hybrid_actions[:, 14:26] >= 0.0) & (hybrid_actions[:, 14:26] <= 1.0)):
            errors.append("recorded normalized hand clench state is outside [0,1]")
        if transition_counts(modes_at_state)["right_rising"] < 1:
            errors.append("right_grasp_mode has no 0->1 transition")
    result = {"pass": not errors, "errors": errors, "action_contract": ACTION_CONTRACT}
    if errors:
        raise ValueError("hybrid action contract check failed: " + "; ".join(errors))
    return result


def load_trajectory_only(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only NPZ arrays; never read sibling metadata or command event files."""
    with np.load(path, allow_pickle=False) as data:
        required = {"states", "timestamps", "grasp_modes_at_state", "hybrid_actions"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"trajectory missing arrays: {sorted(missing)}")
        states = np.asarray(data["states"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        modes = np.asarray(data["grasp_modes_at_state"], dtype=np.float64)
        actions = np.asarray(data["hybrid_actions"], dtype=np.float64)
    check = validate_contract_arrays(states, timestamps, modes, actions)
    return states, timestamps, modes, actions, check


class HybridHandExecutor:
    """Fixed actuator semantics for one persistent right-hand grasp mode."""

    def __init__(
        self,
        clench: Callable[[list[float], bool], Any],
        grasp_force: Callable[[bool], Any],
    ) -> None:
        self._clench = clench
        self._grasp_force = grasp_force
        self.mode = 0
        self.grasp_force_call_count = 0
        self.clench_command_count = 0
        self.clench_commands_while_force_mode_active = 0
        self.rising_edge_count = 0
        self.falling_edge_count = 0

    def apply(self, target_clench: Sequence[float], target_mode: int, *, blocking: bool) -> str:
        target_mode = int(target_mode)
        if target_mode not in (0, 1):
            raise ValueError(f"invalid right_grasp_mode: {target_mode}")
        if self.mode == 0 and target_mode == 1:
            self.rising_edge_count += 1
            self.grasp_force_call_count += 1
            value = self._grasp_force(blocking)
            require_sdk_success(value)
            self.mode = 1
            return "GRASP_FORCE_RISING_EDGE"
        if self.mode == 1 and target_mode == 1:
            return "FORCE_MODE_HOLD_NO_HAND_COMMAND"
        if self.mode == 1 and target_mode == 0:
            self.falling_edge_count += 1
            self.mode = 0
            self.clench_command_count += 1
            value = self._clench([float(v) for v in target_clench], blocking)
            require_sdk_success(value)
            return "POSITION_MODE_FALLING_EDGE_CLENCH"
        if self.mode == 1:
            self.clench_commands_while_force_mode_active += 1
            raise RuntimeError("internal contract violation: clench attempted while force mode active")
        self.clench_command_count += 1
        value = self._clench([float(v) for v in target_clench], blocking)
        require_sdk_success(value)
        return "POSITION_MODE_CLENCH"


def geometry_indices(actions: np.ndarray, arm_threshold: float, hand_threshold: float) -> list[int]:
    if not len(actions):
        return []
    selected = {0, len(actions) - 1}
    anchor = actions[0]
    previous_mode = int(actions[0, 27])
    for index in range(1, len(actions)):
        action = actions[index]
        arm_delta = float(np.max(np.abs(action[7:14] - anchor[7:14])))
        hand_delta = float(np.max(np.abs(action[20:26] - anchor[20:26])))
        mode = int(action[27])
        if arm_delta >= arm_threshold or (mode == 0 and hand_delta >= hand_threshold):
            selected.add(index)
            anchor = action
        if mode != previous_mode:
            selected.add(index)
        previous_mode = mode
    return sorted(selected)


def record(args: argparse.Namespace) -> int:
    if not args.execute:
        print(json.dumps({
            "mode": "record", "status": "PLAN_ONLY_NO_SDK_CLIENTS_CREATED",
            "action_contract": ACTION_CONTRACT, "action_dim": ACTION_DIM,
            "chain": ["RIGHT_READY", "RIGHT_APPROACH_B", "RIGHT_THUMB_TUCK", "RIGHT_GRASP_B", "RIGHT_GRASP_FORCE", "RIGHT_LIFT_B", "HOLD"],
        }, indent=2))
        return 0

    # Geometry and phase implementation are imported only for record mode.
    from agents.three_nut_expert.config import KNOWN_FIXED_NUT_WORLD_POSE
    from agents.three_nut_expert.expert import (
        build_right_safe_lift_pose,
        compute_right_approach_pose,
        compute_right_grasp_pose,
        pose_to_list,
    )
    from tools.test_right_release_stability import DEFAULT_HOLD_AFTER_GRASP_S
    from tools.test_three_nut_closed_loop_v2 import go_right_ready, move_pose

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    right_bundle = left_bundle = None
    sampler: StateSampler | None = None
    audit: PersistentModeAudit | None = None
    report: dict[str, Any] = {"mode": "record", "replay_level": None, "status": "RUNNING"}
    try:
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        sampler = StateSampler(make_devices(right_bundle, left_bundle), args.fps)
        sampler.start()
        audit = PersistentModeAudit(sampler)
        runner = ExpertStateRunner(
            1,
            MotionMonitor(enabled=True, output_root=output_dir / "motion_monitor"),
            output_dir / "expert_report.json",
        )

        audit.phase("RIGHT_READY")
        audit.command("RIGHT_READY", "right_arm.move_joints", "V2 RIGHT_OBSERVATION_JOINTS")
        go_right_ready(runner, right_bundle.right_arm)

        nut_pose = KNOWN_FIXED_NUT_WORLD_POSE["B"]
        nut_xyz = [float(nut_pose.x), float(nut_pose.y), float(nut_pose.z)]
        grasp_pose = compute_right_grasp_pose(nut_xyz)
        approach_pose = compute_right_approach_pose(grasp_pose)

        audit.phase("RIGHT_APPROACH_B")
        audit.command("RIGHT_APPROACH_B", "right_arm.move_to", pose_to_list(approach_pose))
        runner.enter("RIGHT_APPROACH", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, approach_pose, "RIGHT_APPROACH_B")})

        audit.phase("RIGHT_THUMB_TUCK")
        runner.enter("RIGHT_THUMB_TUCK", nut="B")

        def send_thumb_tuck() -> Any:
            audit.command(
                "RIGHT_THUMB_TUCK", "right_hand.clench", {"thumb_rotation": 1.0},
                hand="right", grasp_mode_after=0,
            )
            return right_bundle.right_hand.clench(thumb_rotation=1.0)

        value = runner.hand_action(
            label="RIGHT_THUMB_TUCK", command_method="right_hand.clench", fn=send_thumb_tuck
        )
        runner.pass_state({"sdk_return": repr(value)})

        audit.phase("RIGHT_GRASP_B")
        audit.command("RIGHT_GRASP_B", "right_arm.move_to", pose_to_list(grasp_pose))
        runner.enter("RIGHT_GRASP", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, grasp_pose, "RIGHT_PICK_B")})

        audit.phase("RIGHT_GRASP_FORCE")
        runner.enter("RIGHT_GRASP_FORCE", nut="B")

        def send_grasp_force() -> Any:
            audit.command(
                "RIGHT_GRASP_FORCE", "right_hand.grasp_force", dict(RIGHT_GRASP_FORCE),
                hand="right", grasp_mode_after=1,
            )
            return right_bundle.right_hand.grasp_force(**RIGHT_GRASP_FORCE)

        value = runner.hand_action(
            label="RIGHT_GRASP_FORCE", command_method="right_hand.grasp_force", fn=send_grasp_force
        )
        if DEFAULT_HOLD_AFTER_GRASP_S > 0:
            time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
        runner.pass_state({"sdk_return": repr(value)})

        lift_pose = build_right_safe_lift_pose(grasp_pose)
        audit.phase("RIGHT_LIFT_B")
        audit.command("RIGHT_LIFT_B", "right_arm.move_to", pose_to_list(lift_pose))
        runner.enter("RIGHT_LIFT", nut="B")
        runner.pass_state({"move": move_pose(runner, right_bundle.right_arm, lift_pose, "RIGHT_SAFE_LIFT_B")})
        audit.phase("HOLD")
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
    states = np.asarray([sample["state"] for sample in samples], dtype=np.float32).reshape((-1, STATE_DIM))
    timestamps = np.asarray([sample["timestamp"] for sample in samples], dtype=np.float64)
    events = audit.events if audit is not None else []
    modes = align_modes_to_states(timestamps, events)
    actions = build_hybrid_actions(states, modes)
    contract_check: dict[str, Any] | None = None
    try:
        contract_check = validate_contract_arrays(states, timestamps, modes, actions)
    except Exception as exc:
        report.update(status="FAILED_CONTRACT", contract_error=repr(exc))
    np.savez_compressed(
        output_dir / "trajectory.npz",
        states=states,
        timestamps=timestamps,
        grasp_modes_at_state=modes,
        hybrid_actions=actions,
    )
    write_json(output_dir / "command_events.json", {
        "role": "AUDIT_ONLY_NOT_REPLAY_INPUT",
        "phase_events": audit.phases if audit is not None else [],
        "command_events": events,
    })
    counts = transition_counts(modes)
    recorded_force_calls = sum(event.get("method") == "right_hand.grasp_force" for event in events)
    recorded_clench_while_force = sum(
        event.get("method") == "right_hand.clench"
        and event.get("persistent_grasp_mode_before", [0, 0])[1] == 1
        for event in events
    )
    metadata = {
        "created_at": now_iso(), "git_commit": git_commit(), "nut": "B",
        "action_contract": ACTION_CONTRACT, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "fps": args.fps, "effective_fps": effective_hz(timestamps),
        "state_schema": STATE_SCHEMA, "action_schema": ACTION_SCHEMA,
        "mode_alignment": "latest persistent hand-command mode with command timestamp <= state timestamp",
        "mode_change_source": "actual hand command invocation wrapper; never inferred from phase names",
        "expert_source": "imported V2 Nut-B geometry/Ready/force configuration",
        "contract_check": contract_check,
        "sdk_signatures": {
            "right_arm.move_joints": sdk_signature(getattr(getattr(right_bundle, "right_arm", None), "move_joints", None)),
            "right_hand.clench": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "clench", None)),
            "right_hand.grasp_force": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "grasp_force", None)),
        },
    }
    report.update({
        "finished_at": now_iso(), "output_dir": str(output_dir),
        "record_state_count": len(states), "record_effective_hz": effective_hz(timestamps),
        "hybrid_action_count": len(actions), "action_dim": ACTION_DIM,
        "grasp_mode_transition_count": counts["total"],
        "right_grasp_rising_edge_count": counts["right_rising"],
        "right_grasp_falling_edge_count": counts["right_falling"],
        "grasp_force_call_count": recorded_force_calls,
        "clench_commands_while_force_mode_active": recorded_clench_while_force,
        "right_grasp_mode_final": int(modes[-1, 1]) if len(modes) else None,
        "state_read_errors": sampler.errors if sampler is not None else [],
        "state_sampler_overrun_periods": sampler.overrun_periods if sampler is not None else 0,
        "contract_check": contract_check,
    })
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "COMPLETED" else 1


def ask_physical_result() -> tuple[bool | None, str]:
    if not sys.stdin.isatty():
        return None, "UNKNOWN"
    try:
        answer = input("Did the right hand lift and hold Nut B? [y/n/u] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "u"
    if answer == "y":
        return True, "USER_VISUAL_CONFIRMATION"
    if answer == "n":
        return False, "USER_VISUAL_CONFIRMATION"
    return None, "UNKNOWN"


def replay(args: argparse.Namespace) -> int:
    try:
        states, timestamps, modes, actions, contract_check = load_trajectory_only(args.trajectory)
    except Exception as exc:
        print(json.dumps({
            "mode": "replay", "status": "FAILED_CLOSED_CONTRACT",
            "trajectory": str(args.trajectory), "error": repr(exc),
            "sdk_clients_created": False,
        }, indent=2))
        return 2

    selected = geometry_indices(actions, args.arm_waypoint_threshold, args.hand_waypoint_threshold)
    mode_transition_indices = ([0] if len(actions) and int(actions[0, 27]) != int(modes[0, 1]) else []) + [
        index for index in range(1, len(actions)) if int(actions[index, 27]) != int(actions[index - 1, 27])
    ]
    if args.replay_level == "geometry" and not set(mode_transition_indices).issubset(selected):
        raise RuntimeError("internal fail-closed: geometry selection dropped a grasp-mode transition")
    plan = {
        "mode": "replay", "replay_level": args.replay_level,
        "trajectory": str(args.trajectory), "execute": args.execute,
        "trajectory_input_source": "trajectory.npz only",
        "expert_geometry_used_as_replay_input": False,
        "command_events_loaded": False,
        "action_contract": ACTION_CONTRACT, "action_dim": ACTION_DIM,
        "hybrid_action_count": len(actions), "contract_check": contract_check,
        "selected_action_count": len(selected) if args.replay_level == "geometry" else len(actions),
        "forced_mode_transition_indices": mode_transition_indices,
    }
    if not args.execute:
        print(json.dumps({**plan, "status": "PLAN_ONLY_NO_SDK_CLIENTS_CREATED"}, indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"replay_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    right_bundle = left_bundle = None
    command_errors: list[dict[str, Any]] = []
    command_log: list[dict[str, Any]] = []
    command_count = 0
    arm_command_count = 0
    action_tick_count = 0
    lateness_s: list[float] = []
    arm_ok = True
    final_state: Sequence[float] | None = None
    physical_pick_success: bool | None = None
    physical_source = "UNKNOWN"
    execution_elapsed: float | None = None
    report: dict[str, Any] = {**plan, "status": "RUNNING", "started_at": now_iso()}
    started = time.monotonic()
    try:
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        devices = make_devices(right_bundle, left_bundle)

        def send_clench(target: list[float], blocking: bool) -> Any:
            return right_bundle.right_hand.clench(*target, blocking=blocking)

        def send_force(blocking: bool) -> Any:
            return right_bundle.right_hand.grasp_force(**RIGHT_GRASP_FORCE, blocking=blocking)

        hand_executor = HybridHandExecutor(send_clench, send_force)
        if args.replay_level == "geometry":
            runner = ExpertStateRunner(
                1,
                MotionMonitor(enabled=True, output_root=output_dir / "motion_monitor"),
                output_dir / "replay_motion_report.json",
            )
            last_arm: np.ndarray | None = None
            last_position_hand: np.ndarray | None = None
            for index in selected:
                action = actions[index]
                mode = int(action[27])
                hand = action[20:26]
                previous_mode = hand_executor.mode
                should_apply_hand = (
                    mode != previous_mode
                    or (mode == 0 and (
                        last_position_hand is None
                        or float(np.max(np.abs(hand - last_position_hand))) >= args.hand_waypoint_threshold
                    ))
                )
                if should_apply_hand:
                    try:
                        semantic = hand_executor.apply(hand, mode, blocking=True)
                        command_count += 1
                        command_log.append({"action_index": index, "component": "right_hand", "semantic": semantic})
                        if mode == 0:
                            last_position_hand = hand.copy()
                    except Exception as exc:
                        command_errors.append({"action_index": index, "component": "right_hand", "error": repr(exc)})
                        break
                arm = action[7:14]
                if last_arm is None or float(np.max(np.abs(arm - last_arm))) >= args.arm_waypoint_threshold:
                    target = arm.tolist()
                    try:
                        runner.arm_action(
                            label=f"HYBRID28_REPLAY_ARM_{index:06d}",
                            arm=right_bundle.right_arm,
                            command_method="move_joints",
                            target_joint=target,
                            fn=lambda target=target: right_bundle.right_arm.move_joints(target),
                        )
                        arm_command_count += 1
                        command_count += 1
                        last_arm = arm.copy()
                    except Exception as exc:
                        arm_ok = False
                        command_errors.append({"action_index": index, "component": "right_arm", "error": repr(exc)})
                        break
                action_tick_count += 1
        else:
            schedule_started = time.monotonic()
            first_source_time = float(timestamps[1])
            for index, action in enumerate(actions):
                due = schedule_started + float(timestamps[index + 1] - first_source_time)
                remaining = due - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                lateness_s.append(max(0.0, time.monotonic() - due))
                mode = int(action[27])
                try:
                    semantic = hand_executor.apply(action[20:26], mode, blocking=False)
                    if semantic != "FORCE_MODE_HOLD_NO_HAND_COMMAND":
                        command_count += 1
                    command_log.append({"action_index": index, "component": "right_hand", "semantic": semantic})
                except Exception as exc:
                    command_errors.append({"action_index": index, "component": "right_hand", "error": repr(exc)})
                    break
                try:
                    value = right_bundle.right_arm.move_joints(action[7:14].tolist(), blocking=False)
                    require_sdk_success(value)
                    arm_command_count += 1
                    command_count += 1
                except Exception as exc:
                    arm_ok = False
                    command_errors.append({"action_index": index, "component": "right_arm", "error": repr(exc)})
                    break
                action_tick_count += 1
        time.sleep(args.final_settle_seconds)
        final_state, _, _ = read_state26_timed(devices)
        execution_elapsed = time.monotonic() - started
        physical_pick_success, physical_source = ask_physical_result()
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        hand_executor = locals().get("hand_executor")
    finally:
        elapsed = execution_elapsed if execution_elapsed is not None else time.monotonic() - started
        shutdown_left_bundle(left_bundle)
        shutdown_bundle(right_bundle)

    hand_executor = locals().get("hand_executor")
    expected = actions[-1, :26]
    errors = None if final_state is None else np.abs(np.asarray(final_state, dtype=np.float64) - expected)
    right_arm_error = float(np.max(errors[7:14])) if errors is not None else None
    right_hand_error = float(np.max(errors[20:26])) if errors is not None else None
    arm_pass = bool(arm_ok and right_arm_error is not None and right_arm_error <= args.final_arm_error_threshold)
    timing_span = max(0.0, elapsed - args.final_settle_seconds)
    effective_action_hz = action_tick_count / timing_span if timing_span > 0 else None
    timing_pass = bool(
        args.replay_level == "timing"
        and action_tick_count == len(actions)
        and effective_action_hz is not None
        and effective_action_hz >= 0.9 * args.fps
        and not command_errors
    )
    force_calls = hand_executor.grasp_force_call_count if hand_executor is not None else 0
    clench_in_force = hand_executor.clench_commands_while_force_mode_active if hand_executor is not None else 0
    executor_pass = bool(arm_pass and not command_errors and force_calls == 1 and clench_in_force == 0)
    if not executor_pass:
        conclusion = "RIGHT_HYBRID28_NOT_FEASIBLE"
        conclusion_reason = "arm or fixed actuator executor contract failed"
    elif physical_pick_success is False:
        conclusion = "RIGHT_HYBRID28_FORCE_FIX_INSUFFICIENT"
        conclusion_reason = "user confirmed Nut B was not lifted and held"
    elif physical_pick_success is True and timing_pass:
        conclusion = "RIGHT_HYBRID28_FEASIBLE"
        conclusion_reason = "right-side timing replay and user-confirmed physical pick passed"
    elif physical_pick_success is True:
        conclusion = "RIGHT_HYBRID28_GEOMETRY_ONLY"
        conclusion_reason = "geometry replay physically passed; 5Hz timing is not verified by this run"
    else:
        conclusion = "RIGHT_HYBRID28_NOT_FEASIBLE"
        conclusion_reason = "physical result is UNKNOWN; fail closed rather than claim feasibility"
    counts = transition_counts(modes)
    report.update({
        "finished_at": now_iso(), "output_dir": str(output_dir),
        "record_state_count": len(states), "record_effective_hz": effective_hz(timestamps),
        "hybrid_action_count": len(actions), "action_dim": ACTION_DIM,
        "grasp_mode_transition_count": counts["total"],
        "right_grasp_rising_edge_count": counts["right_rising"],
        "right_grasp_falling_edge_count": counts["right_falling"],
        "grasp_force_call_count": force_calls,
        "clench_commands_while_force_mode_active": clench_in_force,
        "right_grasp_mode_final": hand_executor.mode if hand_executor is not None else None,
        "right_arm_final_error": right_arm_error,
        "right_hand_final_error": right_hand_error,
        "replay_command_count": command_count,
        "arm_replay_command_count": arm_command_count,
        "replay_duration_s": elapsed,
        "replay_effective_command_hz": command_count / elapsed if elapsed > 0 else None,
        "replay_effective_action_hz": effective_action_hz if args.replay_level == "timing" else None,
        "schedule_lateness_ms": distribution_ms(lateness_s),
        "command_errors": command_errors,
        "command_log": command_log,
        "physical_pick_success": physical_pick_success,
        "physical_pick_success_source": physical_source,
        "timing_5hz_verified": timing_pass,
        "hidden_expert_input_used": False,
        "conclusion": conclusion,
        "conclusion_reason": conclusion_reason,
        "scope": "RIGHT_SIDE_NUT_B_ONLY",
        "sdk_signatures": {
            "right_arm.move_joints": sdk_signature(getattr(getattr(right_bundle, "right_arm", None), "move_joints", None)),
            "right_hand.clench": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "clench", None)),
            "right_hand.grasp_force": sdk_signature(getattr(getattr(right_bundle, "right_hand", None), "grasp_force", None)),
        },
    })
    write_json(output_dir / "metadata.json", {
        "created_at": now_iso(), "git_commit": git_commit(),
        "action_contract": ACTION_CONTRACT, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "state_schema": STATE_SCHEMA, "action_schema": ACTION_SCHEMA,
        "source_trajectory": str(args.trajectory),
        "trajectory_file_was_only_replay_input": True,
        "command_events_loaded": False, "expert_geometry_used_as_replay_input": False,
        "fixed_actuator_configuration": {"right_force_mode": dict(RIGHT_GRASP_FORCE)},
        "selected_action_indices": selected if args.replay_level == "geometry" else list(range(len(actions))),
        "forced_mode_transition_indices": mode_transition_indices,
    })
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "COMPLETED" else 1


def synthetic_contract_test() -> int:
    states = np.zeros((7, STATE_DIM), dtype=np.float32)
    for index in range(len(states)):
        states[index, :14] = index * 0.05
        states[index, 14:26] = min(index * 0.1, 1.0)
    timestamps = np.arange(7, dtype=np.float64) * 0.2
    modes = np.asarray(
        [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 0]],
        dtype=np.float32,
    )
    actions = build_hybrid_actions(states, modes)
    check = validate_contract_arrays(states, timestamps, modes, actions)
    calls: list[dict[str, Any]] = []

    def fake_clench(target: list[float], blocking: bool) -> bool:
        calls.append({"method": "clench", "target": target, "blocking": blocking})
        return True

    def fake_force(blocking: bool) -> bool:
        calls.append({"method": "grasp_force", "config": dict(RIGHT_GRASP_FORCE), "blocking": blocking})
        return True

    executor = HybridHandExecutor(fake_clench, fake_force)
    for action in actions:
        executor.apply(action[20:26], int(action[27]), blocking=False)
    result = {
        "status": "PASS" if (
            actions.shape == (6, 28)
            and np.array_equal(actions[:, :26], states[1:])
            and executor.grasp_force_call_count == 1
            and executor.clench_commands_while_force_mode_active == 0
            and executor.falling_edge_count == 1
        ) else "FAIL",
        "sdk_clients_created": False,
        "hybrid_actions_shape": list(actions.shape),
        "next_state_exact_match": bool(np.array_equal(actions[:, :26], states[1:])),
        "right_rising_edges": transition_counts(modes)["right_rising"],
        "right_falling_edges": transition_counts(modes)["right_falling"],
        "grasp_force_call_count": executor.grasp_force_call_count,
        "clench_commands_while_force_mode_active": executor.clench_commands_while_force_mode_active,
        "contract_check": check,
        "calls": calls,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "replay"))
    parser.add_argument("--self-test", action="store_true", help="run a synthetic 28D/edge test without SDK clients")
    parser.add_argument("--nut", choices=("B",), default="B")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--execute", action="store_true", help="create SDK clients and command the robot")
    parser.add_argument("--replay-level", choices=("geometry", "timing"), default="geometry")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--final-settle-seconds", type=float, default=1.0)
    parser.add_argument("--arm-waypoint-threshold", type=float, default=0.03)
    parser.add_argument("--hand-waypoint-threshold", type=float, default=0.03)
    parser.add_argument("--final-arm-error-threshold", type=float, default=0.15)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        if args.mode is not None or args.execute or args.trajectory is not None:
            parser.error("--self-test cannot be combined with mode/execute/trajectory")
        return synthetic_contract_test()
    if args.mode is None:
        parser.error("--mode is required unless --self-test is used")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive")
    if args.mode == "replay" and args.trajectory is None:
        parser.error("--trajectory is required in replay mode")
    if args.mode == "record" and args.trajectory is not None:
        parser.error("--trajectory is only valid in replay mode")
    return record(args) if args.mode == "record" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
