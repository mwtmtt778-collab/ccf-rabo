#!/usr/bin/env python3
"""Offline conversion of one raw Nut C rosbag episode to ACT 5 Hz arrays."""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_c_only_serial_expert import TOP_RGB_TOPIC  # noqa: E402
from tools.test_hybrid_action_replay import (  # noqa: E402
    ACTION_CONTRACT,
    ACTION_DIM,
    STATE_DIM,
    build_hybrid_actions,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_arm_states(path: Path) -> tuple[dict[str, list[tuple[int, list[float]]]], np.ndarray]:
    samples: dict[str, list[tuple[int, list[float]]]] = {"left_arm": [], "right_arm": []}
    timestamps = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        timestamp_ns = int(row["timestamp_monotonic_ns"])
        left = [float(value) for value in row["left_arm"]]
        right = [float(value) for value in row["right_arm"]]
        if len(left) != 7 or len(right) != 7:
            raise ValueError(f"invalid arm_state.jsonl row {line_number}: expected left7 + right7")
        samples["left_arm"].append((timestamp_ns, left))
        samples["right_arm"].append((timestamp_ns, right))
        timestamps.append(timestamp_ns)
    return samples, np.asarray(timestamps, dtype=np.int64)


def wall_to_monotonic_ns(wall_ns: Sequence[int] | np.ndarray, start_wall_ns: int, start_monotonic_ns: int) -> np.ndarray:
    return np.asarray(wall_ns, dtype=np.int64) - (int(start_wall_ns) - int(start_monotonic_ns))


def timeline_ns(start_ns: int, end_ns: int, hz: float = 5.0) -> np.ndarray:
    if end_ns < start_ns:
        raise ValueError("timeline end precedes start")
    period_ns = int(round(1e9 / hz))
    return np.arange(start_ns, end_ns + 1, period_ns, dtype=np.int64)


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    if not len(source_ns):
        return np.full(len(target_ns), -1, dtype=np.int64)
    result = np.empty(len(target_ns), dtype=np.int64)
    for index, target in enumerate(target_ns):
        right = int(np.searchsorted(source_ns, target, side="left"))
        if right == 0:
            result[index] = 0
        elif right == len(source_ns):
            result[index] = len(source_ns) - 1
        else:
            left = right - 1
            result[index] = left if target - source_ns[left] <= source_ns[right] - target else right
    return result


def latest_causal_values(samples: list[tuple[int, list[float]]], targets_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.full((len(targets_ns), 7), np.nan, dtype=np.float32), np.full(len(targets_ns), -1, dtype=np.int64)
    ordered = sorted(samples, key=lambda row: row[0])
    stamps = [row[0] for row in ordered]
    values = np.empty((len(targets_ns), 7), dtype=np.float32)
    ages = np.empty(len(targets_ns), dtype=np.int64)
    for index, target in enumerate(targets_ns):
        source_index = bisect.bisect_right(stamps, int(target)) - 1
        if source_index < 0:
            values[index] = np.nan
            ages[index] = -1
        else:
            values[index] = ordered[source_index][1]
            ages[index] = int(target) - stamps[source_index]
    return values, ages


def reconstruct_hands(events: list[dict[str, Any]], targets_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hands = np.zeros((len(targets_ns), 12), dtype=np.float32)
    modes = np.zeros((len(targets_ns), 2), dtype=np.float32)
    command_events = sorted(
        (row for row in events if row.get("event") == "command_start" and row.get("device") in {"LEFT_HAND", "RIGHT_HAND"}),
        key=lambda row: int(row["timestamp_monotonic_ns"]),
    )
    current_hands = {"LEFT_HAND": np.zeros(6, dtype=np.float32), "RIGHT_HAND": np.zeros(6, dtype=np.float32)}
    current_modes = {"LEFT_HAND": 0.0, "RIGHT_HAND": 0.0}
    event_index = 0
    for target_index, target_ns in enumerate(targets_ns):
        while event_index < len(command_events) and int(command_events[event_index]["timestamp_monotonic_ns"]) <= int(target_ns):
            row = command_events[event_index]
            device = str(row["device"])
            target = row.get("target")
            clench = None
            if isinstance(target, list) and len(target) == 6 and row.get("command") in {"clench", "open", "release"}:
                clench = target
            elif isinstance(target, dict) and target.get("clench6_after") is not None:
                clench = target["clench6_after"]
            if clench is not None:
                current_hands[device] = np.asarray(clench, dtype=np.float32)
            if isinstance(target, dict) and target.get("grasp_mode_after") is not None:
                current_modes[device] = float(target["grasp_mode_after"])
            event_index += 1
        hands[target_index, :6] = current_hands["LEFT_HAND"]
        hands[target_index, 6:] = current_hands["RIGHT_HAND"]
        modes[target_index] = [current_modes["LEFT_HAND"], current_modes["RIGHT_HAND"]]
    return hands, modes


def build_aligned_episode(
    camera_ns: Sequence[int],
    arm_samples: dict[str, list[tuple[int, list[float]]]],
    events: list[dict[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
    hz: float = 5.0,
) -> dict[str, np.ndarray]:
    targets = timeline_ns(start_ns, end_ns, hz)
    left, left_age = latest_causal_values(arm_samples.get("left_arm", []), targets)
    right, right_age = latest_causal_values(arm_samples.get("right_arm", []), targets)
    hands, modes = reconstruct_hands(events, targets)
    states = np.concatenate((left, right, hands), axis=1).astype(np.float32, copy=False)
    actions = build_hybrid_actions(states, modes).astype(np.float32, copy=False)
    camera_stamps = np.asarray(camera_ns, dtype=np.int64)
    camera_indices = nearest_indices(camera_stamps, targets)
    camera_delta_ns = np.full(len(targets), np.iinfo(np.int64).min, dtype=np.int64)
    valid = camera_indices >= 0
    if valid.any():
        camera_delta_ns[valid] = camera_stamps[camera_indices[valid]] - targets[valid]
    return {
        "states": states,
        "state_timestamps": (targets - start_ns).astype(np.float64) / 1e9,
        "state_monotonic_ns": targets,
        "grasp_modes_at_state": modes,
        "hybrid_actions": actions,
        "camera_timestamps_monotonic_ns": camera_stamps,
        "camera_frame_index_for_state": camera_indices,
        "camera_delta_s": camera_delta_ns.astype(np.float64) / 1e9,
        "camera_age_s": np.abs(camera_delta_ns.astype(np.float64)) / 1e9,
        "left_arm_state_age_s": left_age.astype(np.float64) / 1e9,
        "right_arm_state_age_s": right_age.astype(np.float64) / 1e9,
    }


def image_to_ppm(msg: Any) -> bytes:
    width, height = int(msg.width), int(msg.height)
    encoding = str(msg.encoding).lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported TOP image encoding: {encoding}")
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(msg.step)
    rows = data.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        rows = rows[:, :, ::-1]
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rows.tobytes()


def read_rosbag(bag_path: Path) -> list[tuple[int, Any]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(f"ROS 2 bag Python APIs unavailable: {exc!r}") from exc
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    topic_types = {row.name: row.type for row in reader.get_all_topics_and_types()}
    message_classes = {name: get_message(type_name) for name, type_name in topic_types.items()}
    camera: list[tuple[int, Any]] = []
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        if topic == TOP_RGB_TOPIC:
            camera.append((int(timestamp_ns), deserialize_message(raw, message_classes[topic])))
    return camera


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if len(finite) else None


def quality_report(
    arrays: dict[str, np.ndarray],
    camera_ns: np.ndarray,
    arm_state_ns: np.ndarray,
    events: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    task_profile = str(meta.get("task_profile", "fixed_point_c_baseline"))
    visual_mode = str(meta.get("visual_mode", "async_latest_hold"))
    requires_visual_generalization = bool(meta.get("requires_visual_generalization", False))
    duration_s = float(arrays["state_timestamps"][-1]) if len(arrays["state_timestamps"]) else 0.0
    raw_duration_s = (int(camera_ns[-1]) - int(camera_ns[0])) / 1e9 if len(camera_ns) >= 2 else 0.0
    camera_fps = (len(camera_ns) - 1) / raw_duration_s if raw_duration_s > 0 else None
    arm_duration_s = (int(arm_state_ns[-1]) - int(arm_state_ns[0])) / 1e9 if len(arm_state_ns) >= 2 else 0.0
    arm_hz = (len(arm_state_ns) - 1) / arm_duration_s if arm_duration_s > 0 else None
    left_coverage = float(np.mean(arrays["left_arm_state_age_s"] >= 0.0)) if len(arrays["states"]) else 0.0
    right_coverage = float(np.mean(arrays["right_arm_state_age_s"] >= 0.0)) if len(arrays["states"]) else 0.0
    finite = bool(np.isfinite(arrays["states"]).all() and np.isfinite(arrays["hybrid_actions"]).all())
    age = arrays["camera_age_s"]
    arm_age = np.maximum(arrays["left_arm_state_age_s"], arrays["right_arm_state_age_s"])
    camera_p95 = percentile_or_none(age, 95)
    arm_p95 = percentile_or_none(arm_age[arm_age >= 0.0], 95)
    camera_timestamps_valid = bool(
        len(camera_ns) > 0
        and np.all(camera_ns > 0)
        and (len(camera_ns) == 1 or np.all(np.diff(camera_ns) >= 0))
    )
    hand_source_valid = meta.get("hand_state_source", "command_hold_last") == "command_hold_last"
    checks = {
        "expert_pass": meta.get("expert_status") == "PASS",
        "state26": arrays["states"].ndim == 2 and arrays["states"].shape[1] == STATE_DIM,
        "hybrid28": arrays["hybrid_actions"].shape == (max(0, len(arrays["states"]) - 1), ACTION_DIM),
        "finite_state_and_action": finite,
        "camera_present": len(camera_ns) > 0,
        "camera_raw_count_positive": len(camera_ns) > 0,
        "camera_timestamps_monotonic_and_valid": camera_timestamps_valid,
        "camera_raw_fps_gte_5": camera_fps is not None and camera_fps >= 5.0,
        "camera_p95_age_lte_0_2s": camera_p95 is not None and camera_p95 <= 0.2,
        "left_arm_full_coverage": left_coverage == 1.0,
        "right_arm_full_coverage": right_coverage == 1.0,
        "arm_p95_age_lte_0_1s": arm_p95 is not None and arm_p95 <= 0.1,
        "hand_state_source_command_hold_last": hand_source_valid,
    }
    if task_profile == "fixed_point_c_baseline":
        hard_check_names = (
            "expert_pass",
            "state26",
            "hybrid28",
            "finite_state_and_action",
            "camera_present",
            "camera_raw_count_positive",
            "camera_timestamps_monotonic_and_valid",
            "left_arm_full_coverage",
            "right_arm_full_coverage",
            "arm_p95_age_lte_0_1s",
            "hand_state_source_command_hold_last",
        )
    else:
        hard_check_names = (
            "expert_pass",
            "state26",
            "hybrid28",
            "finite_state_and_action",
            "camera_present",
            "camera_raw_fps_gte_5",
            "camera_p95_age_lte_0_2s",
            "left_arm_full_coverage",
            "right_arm_full_coverage",
            "arm_p95_age_lte_0_1s",
        )
    hard_checks = {name: checks[name] for name in hard_check_names}
    accepted = all(hard_checks.values())
    return {
        "task_profile": task_profile,
        "visual_mode": visual_mode,
        "requires_visual_generalization": requires_visual_generalization,
        "raw_camera_frame_count": int(len(camera_ns)),
        "raw_camera_effective_fps": camera_fps,
        "camera_raw_count": int(len(camera_ns)),
        "camera_raw_fps": camera_fps,
        "arm_state_count": int(len(arm_state_ns)),
        "arm_state_effective_hz": arm_hz,
        "arm_state_source": "passive_arm_state_jsonl",
        "act_timeline_length": int(len(arrays["states"])),
        "duration_s": duration_s,
        "camera_age_median_ms": None if percentile_or_none(age, 50) is None else 1000.0 * float(percentile_or_none(age, 50)),
        "camera_age_p95_ms": None if camera_p95 is None else 1000.0 * camera_p95,
        "camera_age_max_ms": 1000.0 * float(np.max(age)) if len(age) else None,
        "arm_age_median_ms": None if percentile_or_none(arm_age[arm_age >= 0.0], 50) is None else 1000.0 * float(percentile_or_none(arm_age[arm_age >= 0.0], 50)),
        "arm_age_p95_ms": None if arm_p95 is None else 1000.0 * arm_p95,
        "arm_age_max_ms": 1000.0 * float(np.max(arm_age[arm_age >= 0.0])) if np.any(arm_age >= 0.0) else None,
        "arm_left_coverage": left_coverage,
        "arm_right_coverage": right_coverage,
        "left_arm_state_coverage": left_coverage,
        "right_arm_state_coverage": right_coverage,
        "camera_age_s": {
            "median": percentile_or_none(age, 50),
            "p95": camera_p95,
            "max": float(np.max(age)) if len(age) else None,
        },
        "hand_state_source": "command_hold_last",
        "raw_arm_topics_recorded": bool(meta.get("raw_arm_topics_recorded", False)),
        "raw_hand_topics_recorded": bool(meta.get("raw_hand_topics_recorded", False)),
        "action_event_count": len(events),
        "expert_status": meta.get("expert_status"),
        "checks": checks,
        "hard_acceptance_checks": hard_checks,
        "accepted": accepted,
        "accepted_for_training": accepted,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a raw Nut C rosbag episode to the existing 5 Hz ACT schema.")
    parser.add_argument("--episode", type=Path, required=True, help="raw episode directory")
    parser.add_argument("--output", type=Path, help="output episode directory (default: <episode>/act_5hz)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode = args.episode.resolve()
    output = args.output.resolve() if args.output else episode / "act_5hz"
    output.mkdir(parents=True, exist_ok=True)
    meta = read_json(episode / "episode_meta.json")
    events = read_events(episode / meta.get("action_events_path", "action_events.jsonl"))
    camera = read_rosbag(episode / meta.get("bag_path", "bag"))
    arm_samples, arm_state_ns = read_arm_states(episode / meta.get("arm_state_path", "arm_state.jsonl"))
    command_starts = [row for row in events if row.get("event") == "command_start"]
    terminal = [row for row in events if row.get("event") == "expert_status"]
    start_ns = int(command_starts[0]["timestamp_monotonic_ns"]) if command_starts else int(meta["start_monotonic_ns"])
    end_ns = int(terminal[-1]["timestamp_monotonic_ns"]) if terminal else int(meta["end_monotonic_ns"])
    camera_wall_ns = np.asarray([row[0] for row in camera], dtype=np.int64)
    camera_ns = wall_to_monotonic_ns(camera_wall_ns, int(meta["start_wall_ns"]), int(meta["start_monotonic_ns"]))
    arrays = build_aligned_episode(camera_ns, arm_samples, events, start_ns=start_ns, end_ns=end_ns)
    wall_minus_monotonic_ns = int(meta["start_wall_ns"]) - int(meta["start_monotonic_ns"])
    arrays["state_wall_ns"] = arrays["state_monotonic_ns"] + wall_minus_monotonic_ns
    arrays["state_monotonic_timestamps"] = arrays["state_monotonic_ns"].astype(np.float64) / 1e9
    arrays["camera_timestamps_wall_ns"] = camera_wall_ns
    arrays["camera_monotonic_ns"] = camera_ns
    arrays["camera_timestamps"] = (camera_ns - start_ns).astype(np.float64) / 1e9
    arrays["camera_ros_timestamps"] = np.full(len(camera_ns), np.nan, dtype=np.float64)
    arrays["state_read_durations"] = np.zeros(len(arrays["states"]), dtype=np.float64)

    image_dir = output / "images" / "top"
    image_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index, (_timestamp, msg) in enumerate(camera):
        relative = Path("images") / "top" / f"frame_{index:06d}.ppm"
        (output / relative).write_bytes(image_to_ppm(msg))
        frame_paths.append(str(relative))
    arrays["camera_frame_paths"] = np.asarray(frame_paths, dtype=np.str_)
    np.savez_compressed(output / "telemetry.npz", **arrays)
    alignment = [
        {
            "target_timestep_wall_ns": int(target),
            "image_index": int(index),
            "target_timestep_monotonic_ns": int(target_monotonic),
            "image_timestamp_monotonic_ns": int(camera_ns[index]) if index >= 0 else None,
            "image_timestamp_wall_ns": int(camera_wall_ns[index]) if index >= 0 else None,
            "delta_s": float(delta),
            "abs_age_s": float(age),
        }
        for target, target_monotonic, index, delta, age in zip(
            arrays["state_wall_ns"], arrays["state_monotonic_ns"], arrays["camera_frame_index_for_state"], arrays["camera_delta_s"], arrays["camera_age_s"]
        )
    ]
    write_json(output / "camera_alignment.json", alignment)
    quality = quality_report(arrays, camera_ns, arm_state_ns, events, meta)
    write_json(output / "quality_report.json", quality)
    output_meta = {
        "format": "rabo_act_v1_single_episode",
        "episode_id": meta["episode_id"],
        "sequence": ["C"],
        "task_profile": str(meta.get("task_profile", "fixed_point_c_baseline")),
        "visual_mode": str(meta.get("visual_mode", "async_latest_hold")),
        "requires_visual_generalization": bool(meta.get("requires_visual_generalization", False)),
        "fps": 5.0,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_contract": ACTION_CONTRACT,
        "state_schema": ["left_arm7", "right_arm7", "left_hand_clench6", "right_hand_clench6"],
        "telemetry_schema": {
            "states": ["N", 26],
            "state_timestamps": ["N"],
            "state_monotonic_ns": ["N"],
            "grasp_modes_at_state": ["N", 2],
            "hybrid_actions": ["N-1", 28],
            "camera_timestamps_wall_ns": ["F"],
            "camera_timestamps_monotonic_ns": ["F"],
            "camera_monotonic_ns": ["F"],
            "camera_frame_paths": ["F"],
            "camera_frame_index_for_state": ["N"],
            "camera_delta_s": ["N"],
            "camera_age_s": ["N"],
        },
        "camera": {"name": "top", "topic": TOP_RGB_TOPIC, "recording": "rosbag_top_only", "alignment": "NEAREST_FRAME_OFFLINE"},
        "arm_alignment": "LATEST_CAUSAL_PASSIVE_ARM_STATE_JSONL",
        "arm_state_source": "passive_arm_state_jsonl",
        "arm_state_joint_order": "SDK_J1_TO_J7",
        "raw_arm_topics_recorded": False,
        "hand_state_source": "command_hold_last",
        "raw_hand_topics_recorded": False,
        "hand_state_note": "left/right hand6 are reconstructed exclusively from action_events command hold-last",
        "source_raw_episode": str(episode),
        "expert_status": meta.get("expert_status"),
        "accepted_for_training": quality["accepted_for_training"],
    }
    write_json(output / "metadata.json", output_meta)
    print(str(output), flush=True)
    print(f"Accepted for training: {quality['accepted_for_training']}", flush=True)
    return 0 if quality["accepted_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
