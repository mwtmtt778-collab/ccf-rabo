#!/usr/bin/env python3
"""Collect one formal ACT V1 C -> B -> A Expert episode.

Without ``--execute`` this script performs only static and synthetic contract
checks.  Execute mode never resets the scene; the operator must use Web Reset
before starting it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import select
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "act_raw"
ACTION_CONTRACT = "HYBRID_NEXT_STATE_PLUS_GRASP_MODE_V1"
STATE_DIM = 26
ACTION_DIM = 28
TOP_SOURCE = "fixed_rgb"
TOP_DATASET_NAME = "cam_top"
EXPECTED_TOP_WIDTH = 960
EXPECTED_TOP_HEIGHT = 540
EXPECTED_TOP_ENCODING = "rgb8"
CAMERA_P95_LIMIT_S = 0.300
CAMERA_MAX_LIMIT_S = 0.500

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    KNOWN_FIXED_NUT_WORLD_POSE,
    KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from tools.motion_monitor import MotionMonitor  # noqa: E402
from tools.probe_act_recording_sources import (  # noqa: E402
    discover_camera_topics,
    read_state26_timed,
)
from tools.record_act_episode import RawCameraRecorder, jsonable, now_iso  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import LEFT_SAFE_LIFT_DELTA_Z_M  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import (  # noqa: E402
    RIGHT_OBSERVATION_STABLE_WAIT_S,
    RIGHT_RELEASE_OPEN_WAIT_S,
)
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    make_right_bundle,
    shutdown_bundle,
)
from tools.test_three_nut_closed_loop_v2 import (  # noqa: E402
    ExpertStateRunner,
    NUT_SEQUENCE,
    execute_left_pick_place,
    execute_right_transfer,
    go_left_initial_ready,
    go_left_return_ready,
    go_right_ready,
    parse_sequence_arg,
    validate_static_contract,
)


class EpisodeAbort(RuntimeError):
    """Raised between Expert commands after the operator presses q."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class KeyboardAbortWatcher:
    """Watch stdin without blocking Expert or recorder threads."""

    def __init__(self, abort_event: threading.Event) -> None:
        self.abort_event = abort_event
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._fd: int | None = None
        self._termios: Any = None
        self._old_attrs: Any = None

    def start(self) -> None:
        try:
            self._fd = sys.stdin.fileno()
        except (AttributeError, OSError):
            return
        if os.isatty(self._fd):
            try:
                import termios
                import tty

                self._termios = termios
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except Exception:
                self._termios = None
                self._old_attrs = None
        self.thread = threading.Thread(target=self._run, name="act-q-abort-watcher", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self._fd is not None
        while not self.stop_event.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                value = os.read(self._fd, 1)
                if not value:  # Non-interactive stdin reached EOF.
                    return
                if value.lower() == b"q":
                    self.abort_event.set()
                    print(
                        "\n[ABORT_REQUESTED] q received; the active blocking SDK command may "
                        "finish, then no later Expert command will be sent.",
                        flush=True,
                    )
                    return
            except (OSError, ValueError):
                return

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self._termios is not None and self._old_attrs is not None and self._fd is not None:
            with contextlib.suppress(Exception):
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_attrs)


class CollectingExpertRunner(ExpertStateRunner):
    """Add abort boundaries and real hand-command events to the current runner."""

    def __init__(
        self,
        *args: Any,
        abort_event: threading.Event,
        command_events: list[dict[str, Any]],
        timeline_origin: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.abort_event = abort_event
        self.command_events = command_events
        self.timeline_origin = timeline_origin
        self._command_event_lock = threading.Lock()

    def check_abort(self) -> None:
        if self.abort_event.is_set():
            raise EpisodeAbort("operator requested q abort")

    def enter(self, state: str, *, nut: str | None = None) -> None:
        self.check_abort()
        super().enter(state, nut=nut)

    def arm_action(self, **kwargs: Any) -> dict[str, Any]:
        self.check_abort()
        return super().arm_action(**kwargs)

    @staticmethod
    def _hand_and_mode(command_method: str) -> tuple[str, int]:
        hand = "left" if command_method.startswith("left_hand.") else "right"
        if not command_method.startswith(("left_hand.", "right_hand.")):
            raise ValueError(f"unknown hand command method: {command_method}")
        return hand, int(command_method.endswith(".grasp_force"))

    def hand_action(self, *, label: str, fn: Callable[[], Any], command_method: str) -> Any:
        self.check_abort()
        hand, mode_after = self._hand_and_mode(command_method)
        event: dict[str, Any] = {}

        def tracked_invocation() -> Any:
            invoked = time.monotonic()
            event.update({
                "timestamp": now_iso(),
                "monotonic_timestamp": invoked,
                "time_s": invoked - self.timeline_origin,
                "hand": hand,
                "method": command_method,
                "label": label,
                "mode_after": mode_after,
                "success": None,
            })
            with self._command_event_lock:
                self.command_events.append(event)
            return fn()

        try:
            result = super().hand_action(
                label=label,
                fn=tracked_invocation,
                command_method=command_method,
            )
        except BaseException as exc:
            if event:
                event["success"] = False
                event["error"] = repr(exc)
                event["completed_timestamp"] = now_iso()
            raise
        event["success"] = True
        event["completed_timestamp"] = now_iso()
        return result


class StateSampler:
    """Poll 26D state on its own fixed-rate thread."""

    def __init__(self, devices: dict[str, Any], origin: float, fps: float) -> None:
        self.devices = devices
        self.origin = origin
        self.fps = fps
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.overrun_periods = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="act-v1-state-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        interval = 1.0 / self.fps
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < deadline:
                self.stop_event.wait(min(deadline - now, 0.005))
                continue
            read_start = time.monotonic()
            try:
                state, components, component_durations = read_state26_timed(self.devices)
                read_end = time.monotonic()
                self.samples.append({
                    "monotonic_timestamp": read_end,
                    "time_s": read_end - self.origin,
                    "wall_timestamp": now_iso(),
                    "read_duration_s": read_end - read_start,
                    "state": state,
                    "components": components,
                    "component_read_duration_s": component_durations,
                })
            except Exception as exc:
                read_end = time.monotonic()
                self.errors.append({
                    "monotonic_timestamp": read_end,
                    "time_s": read_end - self.origin,
                    "wall_timestamp": now_iso(),
                    "read_duration_s": read_end - read_start,
                    "error": repr(exc),
                })
            deadline += interval
            if read_end > deadline:
                missed = max(1, int((read_end - deadline) / interval) + 1)
                self.overrun_periods += missed
                deadline += missed * interval

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                self.errors.append({"error": "state sampler did not stop within 5 seconds"})

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.samples:
                return
            if self.errors:
                raise RuntimeError(f"initial 26D state read failed: {self.errors[0]}")
            time.sleep(0.01)
        raise TimeoutError(f"no valid 26D state sample arrived within {timeout_s:g}s")


def wait_for_first_top_frame(recorder: RawCameraRecorder, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = recorder.records.get(TOP_SOURCE)
        if record is not None:
            with recorder._lock:  # Same lock used by RawCameraRecorder's callback/writer.
                if record.saved:
                    first = record.saved[0]
                    if (
                        first.get("width") == EXPECTED_TOP_WIDTH
                        and first.get("height") == EXPECTED_TOP_HEIGHT
                        and str(first.get("encoding", "")).lower() == EXPECTED_TOP_ENCODING
                    ):
                        return
                    raise RuntimeError(
                        "top frame contract mismatch: expected 960x540 rgb8, got "
                        f"{first.get('width')}x{first.get('height')} {first.get('encoding')}"
                    )
        time.sleep(0.02)
    raise TimeoutError(f"no valid top frame arrived within {timeout_s:g}s")


def modes_for_states(
    state_timestamps: np.ndarray,
    command_events: Sequence[dict[str, Any]],
) -> np.ndarray:
    modes = np.zeros((len(state_timestamps), 2), dtype=np.int8)
    current = [0, 0]
    successful = sorted(
        (event for event in command_events if event.get("success") is True),
        key=lambda event: float(event["time_s"]),
    )
    event_index = 0
    for state_index, timestamp in enumerate(state_timestamps):
        while event_index < len(successful) and float(successful[event_index]["time_s"]) <= timestamp:
            event = successful[event_index]
            current[0 if event["hand"] == "left" else 1] = int(event["mode_after"])
            event_index += 1
        modes[state_index] = current
    return modes


def causal_camera_alignment(
    state_timestamps: np.ndarray,
    camera_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(camera_timestamps, state_timestamps, side="right") - 1
    ages = np.full(len(state_timestamps), np.nan, dtype=np.float64)
    valid = indices >= 0
    ages[valid] = state_timestamps[valid] - camera_timestamps[indices[valid]]
    return indices.astype(np.int64), ages


def camera_age_summary(ages: np.ndarray) -> dict[str, float | None]:
    valid = ages[np.isfinite(ages)]
    if not len(valid):
        return {name: None for name in ("mean_ms", "median_ms", "p90_ms", "p95_ms", "max_ms")}
    return {
        "mean_ms": float(np.mean(valid) * 1000.0),
        "median_ms": float(np.median(valid) * 1000.0),
        "p90_ms": float(np.percentile(valid, 90) * 1000.0),
        "p95_ms": float(np.percentile(valid, 95) * 1000.0),
        "max_ms": float(np.max(valid) * 1000.0),
    }


def build_episode_arrays(
    samples: Sequence[dict[str, Any]],
    command_events: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    states = np.asarray([sample["state"] for sample in samples], dtype=np.float32)
    if not len(samples):
        states = np.empty((0, STATE_DIM), dtype=np.float32)
    state_timestamps = np.asarray([sample["time_s"] for sample in samples], dtype=np.float64)
    state_monotonic_timestamps = np.asarray(
        [sample["monotonic_timestamp"] for sample in samples], dtype=np.float64
    )
    state_wall_timestamps = np.asarray(
        [sample["wall_timestamp"] for sample in samples], dtype=np.str_
    )
    state_read_durations = np.asarray(
        [sample["read_duration_s"] for sample in samples], dtype=np.float64
    )
    grasp_modes = modes_for_states(state_timestamps, command_events)
    if len(states) >= 2:
        hybrid_actions = np.concatenate(
            (states[1:], grasp_modes[1:].astype(np.float32)), axis=1
        ).astype(np.float32, copy=False)
    else:
        hybrid_actions = np.empty((0, ACTION_DIM), dtype=np.float32)

    sorted_frames = sorted(frames, key=lambda item: float(item["arrival_time_s"]))
    camera_timestamps = np.asarray(
        [frame["arrival_time_s"] for frame in sorted_frames], dtype=np.float64
    )
    camera_ros_timestamps = np.asarray(
        [frame["ros_timestamp"] if frame.get("ros_timestamp") is not None else np.nan for frame in sorted_frames],
        dtype=np.float64,
    )
    camera_frame_paths = np.asarray([frame["path"] for frame in sorted_frames], dtype=np.str_)
    camera_indices, camera_ages = causal_camera_alignment(state_timestamps, camera_timestamps)
    return {
        "states": states,
        "state_timestamps": state_timestamps,
        "state_monotonic_timestamps": state_monotonic_timestamps,
        "state_wall_timestamps": state_wall_timestamps,
        "state_read_durations": state_read_durations,
        "grasp_modes_at_state": grasp_modes,
        "hybrid_actions": hybrid_actions,
        "camera_timestamps": camera_timestamps,
        "camera_ros_timestamps": camera_ros_timestamps,
        "camera_frame_paths": camera_frame_paths,
        "camera_frame_index_for_state": camera_indices,
        "camera_age_s": camera_ages,
    }


def evaluate_quality(
    arrays: dict[str, np.ndarray],
    *,
    state_errors: Sequence[dict[str, Any]],
    expert_done: bool,
    completed_nuts: Sequence[str],
    camera_record: Any,
    camera_writer_error: str | None,
) -> dict[str, Any]:
    states = arrays["states"]
    actions = arrays["hybrid_actions"]
    modes = arrays["grasp_modes_at_state"]
    state_times = arrays["state_timestamps"]
    camera_times = arrays["camera_timestamps"]
    indices = arrays["camera_frame_index_for_state"]
    ages = arrays["camera_age_s"]
    valid_indices = indices >= 0
    sorted_frames = sorted(camera_record.saved, key=lambda item: float(item["arrival_time_s"]))
    age_stats = camera_age_summary(ages)
    checks = {
        "states_rank_2": states.ndim == 2,
        "states_dim_26": states.ndim == 2 and states.shape[1] == STATE_DIM,
        "at_least_one_transition": len(states) >= 2,
        "hybrid_actions_rank_2": actions.ndim == 2,
        "hybrid_actions_shape_n_minus_1_by_28": actions.shape == (max(0, len(states) - 1), ACTION_DIM),
        "action_state_exact_match": bool(
            actions.ndim == 2
            and actions.shape[0] == max(0, len(states) - 1)
            and np.array_equal(actions[:, :STATE_DIM], states[1:])
        ),
        "grasp_flags_binary": bool(np.isin(modes, (0, 1)).all()),
        "action_flags_match_next_state_modes": bool(
            actions.shape == (max(0, len(states) - 1), ACTION_DIM)
            and np.array_equal(actions[:, STATE_DIM:], modes[1:].astype(actions.dtype))
        ),
        "state_timestamps_strictly_increasing": bool(
            len(state_times) < 2 or np.all(np.diff(state_times) > 0)
        ),
        "camera_timestamps_strictly_increasing": bool(
            len(camera_times) < 2 or np.all(np.diff(camera_times) > 0)
        ),
        "finite_states_and_actions": bool(np.isfinite(states).all() and np.isfinite(actions).all()),
        "normalized_hand_state_in_range": bool(
            states.ndim == 2
            and states.shape[1] == STATE_DIM
            and np.isfinite(states[:, 14:26]).all()
            and np.all((states[:, 14:26] >= 0.0) & (states[:, 14:26] <= 1.0))
        ),
        "no_state_read_error": not state_errors,
        "expert_complete_done": expert_done,
        "completed_nuts_c_b_a": list(completed_nuts) == ["C", "B", "A"],
        "all_observations_have_causal_top_frame": bool(len(indices) == len(states) and valid_indices.all()),
        "camera_never_uses_future_frame": bool(
            valid_indices.all()
            and np.all(camera_times[indices] <= state_times)
            and np.all(ages >= 0.0)
        ) if len(indices) else False,
        "top_frames_960x540_rgb8": bool(
            sorted_frames
            and all(
                frame.get("width") == EXPECTED_TOP_WIDTH
                and frame.get("height") == EXPECTED_TOP_HEIGHT
                and str(frame.get("encoding", "")).lower() == EXPECTED_TOP_ENCODING
                for frame in sorted_frames
            )
        ),
        "top_camera_writer_clean": bool(
            camera_writer_error is None
            and camera_record.write_error_count == 0
            and camera_record.queue_drop_count == 0
            and camera_record.subscribe_error is None
        ),
        "camera_age_p95_le_300ms": bool(
            len(ages) and np.isfinite(ages).all() and float(np.percentile(ages, 95)) <= CAMERA_P95_LIMIT_S
        ),
        "camera_age_max_le_500ms": bool(
            len(ages) and np.isfinite(ages).all() and float(np.max(ages)) <= CAMERA_MAX_LIMIT_S
        ),
    }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "quality_gate_pass": bool(all(checks.values())),
        "accepted_for_training": False,
        "checks": checks,
        "camera_age": age_stats,
        "thresholds": {
            "camera_age_p95_ms": CAMERA_P95_LIMIT_S * 1000.0,
            "camera_age_max_ms": CAMERA_MAX_LIMIT_S * 1000.0,
        },
        "state_count": int(len(states)),
        "transition_count": int(max(0, len(states) - 1)),
        "state_read_error_count": len(state_errors),
    }


def synthetic_contract_self_test() -> dict[str, Any]:
    samples = []
    for index in range(4):
        state = np.linspace(0.0, 0.9, STATE_DIM, dtype=np.float32)
        state[:14] += index
        samples.append({
            "state": state.tolist(),
            "time_s": 0.2 + 0.2 * index,
            "monotonic_timestamp": 100.2 + 0.2 * index,
            "wall_timestamp": f"synthetic-{index}",
            "read_duration_s": 0.001,
        })
    events = [
        {
            "time_s": 0.35,
            "hand": "left",
            "method": "left_hand.grasp_force",
            "mode_after": 1,
            "success": True,
        },
        {
            "time_s": 0.55,
            "hand": "right",
            "method": "right_hand.grasp_force",
            "mode_after": 1,
            "success": True,
        },
        {
            "time_s": 0.65,
            "hand": "right",
            "method": "right_hand.clench",
            "mode_after": 0,
            "success": False,
        },
        {
            "time_s": 0.75,
            "hand": "left",
            "method": "left_hand.clench",
            "mode_after": 0,
            "success": True,
        },
    ]
    frames = [
        {
            "path": f"cameras/cam_top/frame_{index:06d}.ppm",
            "arrival_time_s": 0.1 + 0.2 * index,
            "ros_timestamp": 1000.0 + index,
            "width": EXPECTED_TOP_WIDTH,
            "height": EXPECTED_TOP_HEIGHT,
            "encoding": EXPECTED_TOP_ENCODING,
        }
        for index in range(4)
    ]

    class SyntheticCameraRecord:
        saved = frames
        write_error_count = 0
        queue_drop_count = 0
        subscribe_error = None

    arrays = build_episode_arrays(samples, events, frames)
    quality = evaluate_quality(
        arrays,
        state_errors=[],
        expert_done=True,
        completed_nuts=("C", "B", "A"),
        camera_record=SyntheticCameraRecord(),
        camera_writer_error=None,
    )
    if quality["result"] != "PASS":
        raise AssertionError(f"synthetic quality contract failed: {quality['checks']}")
    expected_modes = [[0, 0], [1, 0], [1, 1], [0, 1]]
    if arrays["grasp_modes_at_state"].tolist() != expected_modes:
        raise AssertionError("persistent grasp mode alignment failed")
    future_indices, _ = causal_camera_alignment(
        np.asarray([0.05], dtype=np.float64), np.asarray([0.10], dtype=np.float64)
    )
    if future_indices.tolist() != [-1]:
        raise AssertionError("causal alignment admitted a future frame")
    return {
        "result": "PASS",
        "states_shape": list(arrays["states"].shape),
        "hybrid_actions_shape": list(arrays["hybrid_actions"].shape),
        "action_state_exact_match": quality["checks"]["action_state_exact_match"],
        "future_frame_rejected": True,
    }


def run_expert(
    runner: CollectingExpertRunner,
    sequence: Sequence[str],
    right_bundle: Any,
    left_bundle: Any,
    *,
    settle_after_release_s: float,
    vision_target_radius_m: float,
) -> list[str]:
    """Orchestrate the existing Expert functions without reset or new geometry."""
    completed: list[str] = []
    runner.enter("EPISODE_INIT")
    runner.pass_state({
        "use_scene_initial_pose": True,
        "set_entity_pose_on_episode_init": False,
        "world_reset_used": False,
        "operator_reset_required": "Web Reset before collector start",
    })
    go_right_ready(runner, right_bundle.right_arm)
    go_left_initial_ready(runner, left_bundle.left_arm)
    runner.enter("READY_CHECK")
    runner.pass_state({"right_ready": True, "left_ready": True})
    for key in sequence:
        runtime_xyz = pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE[key])[:3]
        released_xyz = execute_right_transfer(
            runner,
            key,
            right_bundle,
            runtime_nut_world_xyz=runtime_xyz,
            nut_xyz_source=KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
            vision_radius_m=vision_target_radius_m,
            settle_after_release_s=settle_after_release_s,
        )
        execute_left_pick_place(runner, key, left_bundle, released_xyz)
        completed.append(key)
    go_right_ready(runner, right_bundle.right_arm)
    go_left_return_ready(runner, left_bundle.left_arm)
    runner.enter("DONE")
    runner.pass_state({"completed_nuts": completed})
    return completed


def unique_episode_id(requested: str | None) -> str:
    if requested:
        return requested
    return datetime.now().astimezone().strftime("episode_%Y%m%d_%H%M%S_%f")


def move_episode(source: Path, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite episode: {destination}")
    os.replace(source, destination)
    return destination


def execute_episode(args: argparse.Namespace) -> int:
    sequence = parse_sequence_arg(args.sequence)
    if tuple(sequence) != NUT_SEQUENCE:
        raise SystemExit("formal ACT V1 collector requires exact sequence C,B,A")
    validate_static_contract(tuple(sequence))
    if not math.isclose(LEFT_SAFE_LIFT_DELTA_Z_M, 0.12, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"LEFT_SAFE_LIFT_DELTA_Z_M must be 0.12, got {LEFT_SAFE_LIFT_DELTA_Z_M}")

    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    temporary_root = output_root / "temporary"
    accepted_root = output_root / "accepted"
    rejected_root = output_root / "rejected"
    temporary_root.mkdir(parents=True, exist_ok=True)
    episode_id = unique_episode_id(args.episode_id)
    episode_dir = temporary_root / episode_id
    if episode_dir.exists():
        raise SystemExit(f"refusing to overwrite existing episode: {episode_dir}")
    episode_dir.mkdir(parents=True)
    (PROJECT_ROOT / "logs" / "ros").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))

    started_at = now_iso()
    timeline_origin = time.monotonic()
    abort_event = threading.Event()
    watcher = KeyboardAbortWatcher(abort_event)
    camera_topics, discovery = discover_camera_topics()
    top_topics = {TOP_SOURCE: camera_topics[TOP_SOURCE]} if TOP_SOURCE in camera_topics else {}
    recorder = RawCameraRecorder(top_topics, episode_dir, queue_size=args.queue_size)
    right_bundle = left_bundle = None
    sampler: StateSampler | None = None
    runner: CollectingExpertRunner | None = None
    command_events: list[dict[str, Any]] = []
    completed_nuts: list[str] = []
    expert_done = False
    expert_status = "NOT_STARTED"
    failure_reason: str | None = None
    shutdown_errors: list[str] = []

    print(f"ACT V1 single episode: {episode_id}", flush=True)
    print("Reset policy: Web Reset only; this collector calls no reset API.", flush=True)
    print("Press q to abort after the current blocking SDK command returns.", flush=True)
    try:
        right_bundle = make_right_bundle()
        left_bundle = make_left_bundle()
        devices = {
            "left_arm": left_bundle.left_arm,
            "right_arm": right_bundle.right_arm,
            "left_hand": left_bundle.left_hand,
            "right_hand": right_bundle.right_hand,
        }
        recorder.start()
        if recorder.init_error:
            raise RuntimeError(f"top camera initialization failed: {recorder.init_error}")
        if TOP_SOURCE not in recorder.records:
            raise RuntimeError("top fixed_rgb camera was not discovered")
        recorder.begin(timeline_origin)
        print("Waiting for first valid 960x540 rgb8 top frame...", flush=True)
        wait_for_first_top_frame(recorder, args.camera_ready_timeout)
        print("Top camera ready; starting formal state timeline and Expert.", flush=True)
        sampler = StateSampler(devices, timeline_origin, args.fps)
        sampler.start()
        sampler.wait_until_ready()
        watcher.start()
        runner = CollectingExpertRunner(
            1,
            MotionMonitor(enabled=True),
            episode_dir / "expert_report.json",
            abort_event=abort_event,
            command_events=command_events,
            timeline_origin=timeline_origin,
        )
        expert_status = "RUNNING"
        completed_nuts = run_expert(
            runner,
            sequence,
            right_bundle,
            left_bundle,
            settle_after_release_s=args.settle_after_release_s,
            vision_target_radius_m=args.vision_target_radius_m,
        )
        expert_done = runner.state == "DONE" and runner.last_success_state == "DONE"
        expert_status = "PASS" if expert_done else "FAILED"
    except EpisodeAbort as exc:
        expert_status = "ABORTED"
        failure_reason = repr(exc)
    except KeyboardInterrupt as exc:
        abort_event.set()
        expert_status = "ABORTED"
        failure_reason = repr(exc)
    except BaseException as exc:
        expert_status = "FAULT"
        failure_reason = repr(exc)
        print(f"Episode fault: {failure_reason}", flush=True)
    finally:
        watcher.close()
        if sampler is not None:
            sampler.stop()
        recorder.close()
        try:
            shutdown_bundle(right_bundle)
        except Exception as exc:
            shutdown_errors.append(f"right_bundle: {repr(exc)}")
        try:
            shutdown_left_bundle(left_bundle)
        except Exception as exc:
            shutdown_errors.append(f"left_bundle: {repr(exc)}")

    camera_metadata = recorder.write_metadata()
    record = recorder.records.get(TOP_SOURCE)
    if record is None:
        # A stand-in lets the quality report remain fail-closed and complete.
        class MissingRecord:
            saved: list[dict[str, Any]] = []
            write_error_count = 0
            queue_drop_count = 0
            subscribe_error = "top camera unavailable"

        record = MissingRecord()
    samples = sampler.samples if sampler is not None else []
    state_errors = sampler.errors if sampler is not None else [{"error": "state sampler not started"}]
    arrays = build_episode_arrays(samples, command_events, record.saved)
    quality = evaluate_quality(
        arrays,
        state_errors=state_errors,
        expert_done=expert_done,
        completed_nuts=completed_nuts,
        camera_record=record,
        camera_writer_error=recorder.writer_shutdown_error,
    )
    np.savez_compressed(episode_dir / "telemetry.npz", **arrays)
    write_json(episode_dir / "command_events.json", command_events)
    write_json(episode_dir / "expert_states.json", runner.states if runner is not None else [])

    rejection_reason: str | None = None
    accepted = False
    if expert_status != "PASS":
        rejection_reason = "operator_q_abort" if expert_status == "ABORTED" else "expert_fault"
    elif quality["result"] != "PASS":
        rejection_reason = "quality_gate_failed"

    metadata = {
        "format": "rabo_act_v1_single_episode",
        "episode_id": episode_id,
        "started_at": started_at,
        "finished_at": now_iso(),
        "sequence": list(sequence),
        "fps": args.fps,
        "action_contract": ACTION_CONTRACT,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "telemetry_schema": {
            "states": ["N", 26],
            "state_timestamps": ["N"],
            "state_monotonic_timestamps": ["N"],
            "state_wall_timestamps": ["N"],
            "state_read_durations": ["N"],
            "grasp_modes_at_state": ["N", 2],
            "hybrid_actions": ["N-1", 28],
            "camera_timestamps": ["F"],
            "camera_ros_timestamps": ["F"],
            "camera_frame_paths": ["F"],
            "camera_frame_index_for_state": ["N"],
            "camera_age_s": ["N"],
        },
        "observation": "26D state + causal top RGB only",
        "camera_alignment": "LATEST_PREVIOUS_FRAME",
        "camera_timestamp_basis": "monotonic callback arrival relative to episode origin",
        "camera_discovery": discovery,
        "camera": camera_metadata.get(TOP_DATASET_NAME, {}),
        "expert_status": expert_status,
        "expert_done": expert_done,
        "completed_nuts": completed_nuts,
        "failure_reason": failure_reason,
        "abort_requested": abort_event.is_set(),
        "state_read_errors": state_errors,
        "state_overrun_periods": sampler.overrun_periods if sampler is not None else 0,
        "shutdown_errors": shutdown_errors,
        "left_safe_lift_delta_z_m": LEFT_SAFE_LIFT_DELTA_Z_M,
        "accepted_for_training": False,
        "rejection_reason": rejection_reason,
        "reset_policy": "operator Web Reset only; collector performs no reset",
    }
    if rejection_reason is None:
        try:
            answer = input("Save this episode? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer == "y":
            accepted = True
        else:
            rejection_reason = "user_declined"

    metadata["accepted_for_training"] = accepted
    metadata["rejection_reason"] = rejection_reason
    quality["accepted_for_training"] = accepted
    quality["rejection_reason"] = rejection_reason
    write_json(episode_dir / "quality_report.json", quality)
    write_json(episode_dir / "metadata.json", metadata)
    destination = move_episode(episode_dir, accepted_root if accepted else rejected_root)
    print(f"Expert status: {expert_status}", flush=True)
    print(f"Quality: {quality['result']}", flush=True)
    print(f"Accepted for training: {accepted}", flush=True)
    print(f"Episode: {destination}", flush=True)
    return 0 if accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one formal ACT V1 C -> B -> A Expert episode."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Drive the current blocking Expert once. Without this flag: dry-run only.",
    )
    parser.add_argument("--sequence", default="C,B,A", help="Frozen formal sequence (default: C,B,A).")
    parser.add_argument("--fps", type=float, default=5.0, help="State sampling frequency (default: 5Hz).")
    parser.add_argument("--monitor", action="store_true", help="Compatibility flag; MotionMonitor is always enabled.")
    parser.add_argument("--episode-id", help="Optional unique episode directory name.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--queue-size", type=int, default=96)
    parser.add_argument("--camera-ready-timeout", type=float, default=15.0)
    parser.add_argument("--settle-after-release-s", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S)
    parser.add_argument("--vision-target-radius-m", type=float, default=DEFAULT_VISION_TARGET_RADIUS_M)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        sequence = parse_sequence_arg(args.sequence)
    except Exception as exc:
        parser.error(str(exc))
    if tuple(sequence) != NUT_SEQUENCE:
        parser.error("formal ACT V1 collector requires exact sequence C,B,A")
    if args.fps <= 0 or args.queue_size <= 0 or args.camera_ready_timeout <= 0:
        parser.error("fps, queue-size, and camera-ready-timeout must be > 0")
    if args.vision_target_radius_m <= 0:
        parser.error("vision-target-radius-m must be > 0")
    minimum_settle = RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S
    if args.settle_after_release_s < minimum_settle:
        parser.error(f"settle-after-release-s must be at least {minimum_settle:g}s")

    validate_static_contract(tuple(sequence))
    if not math.isclose(LEFT_SAFE_LIFT_DELTA_Z_M, 0.12, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"height contract failed: LEFT_SAFE_LIFT_DELTA_Z_M={LEFT_SAFE_LIFT_DELTA_Z_M}")
    if args.execute:
        return execute_episode(args)
    result = synthetic_contract_self_test()
    print("DRY RUN: no robot, hand, camera, or reset API was invoked.")
    print(json.dumps({
        "static_contract": "PASS",
        "sequence": list(sequence),
        "fps": args.fps,
        "top_camera_only": True,
        "left_safe_lift_delta_z_m": LEFT_SAFE_LIFT_DELTA_Z_M,
        "synthetic_contract_self_test": result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
