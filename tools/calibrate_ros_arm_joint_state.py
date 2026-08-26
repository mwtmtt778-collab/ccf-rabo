#!/usr/bin/env python3
"""Read-only calibration of ROS JointState channels against A7 SDK joint order.

Each invocation records one stationary snapshot.  A second (or later) snapshot
in the same session triggers delta-based one-to-one mapping inference.  The
tool never sends a robot motion command.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "act_joint_calibration"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
EXPECTED_DOF = 7
DEFAULT_MAX_ERROR_RAD = 0.02
DEFAULT_MIN_MOTION_RAD = 0.02
DEFAULT_UNIQUENESS_MARGIN = 1e-4

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS  # noqa: E402
from tools.map_rabo_runtime_interfaces import (  # noqa: E402
    Topic,
    annotate_topics,
    detect_namespace,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ros_stamp(msg: Any) -> dict[str, Any]:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return {"sec": sec, "nanosec": nanosec, "float": sec + nanosec * 1e-9}


def channel_identity(channel: dict[str, Any]) -> str:
    return f"{channel['topic']}#{int(channel['index'])}"


def snapshot_channels(snapshot: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    channels = snapshot.get("ros", {}).get(arm, {}).get("channels", [])
    return {channel_identity(row): row for row in channels}


def _pair_fit(
    snapshots: list[dict[str, Any]],
    arm: str,
    sdk_index: int,
    channel_id: str,
) -> dict[str, Any]:
    sdk = [float(row["sdk"][arm]["joint7"][sdk_index]) for row in snapshots]
    channels = [snapshot_channels(row, arm)[channel_id] for row in snapshots]
    ros = [float(row["position"]) for row in channels]
    delta_sdk = [value - sdk[0] for value in sdk[1:]]
    delta_ros = [value - ros[0] for value in ros[1:]]
    candidates = []
    for sign in (1, -1):
        delta_errors = [a - sign * b for a, b in zip(delta_sdk, delta_ros)]
        delta_rmse = math.sqrt(sum(value * value for value in delta_errors) / len(delta_errors))
        offset = sum(a - sign * b for a, b in zip(sdk, ros)) / len(sdk)
        errors = [abs(a - (sign * b + offset)) for a, b in zip(sdk, ros)]
        candidates.append({
            "sign": sign,
            "offset": offset,
            "delta_rmse": delta_rmse,
            "max_abs_error": max(errors),
            "mean_abs_error": sum(errors) / len(errors),
            "errors_by_pose": errors,
        })
    best = min(candidates, key=lambda item: (item["delta_rmse"], item["max_abs_error"]))
    first_channel = channels[0]
    return {
        **best,
        "channel_id": channel_id,
        "topic": first_channel["topic"],
        "index": int(first_channel["index"]),
        "name": first_channel.get("name", ""),
        "delta_sdk": delta_sdk,
        "delta_ros": delta_ros,
    }


def infer_arm_mapping(
    snapshots: list[dict[str, Any]],
    arm: str,
    *,
    max_error_rad: float = DEFAULT_MAX_ERROR_RAD,
    min_motion_rad: float = DEFAULT_MIN_MOTION_RAD,
    uniqueness_margin: float = DEFAULT_UNIQUENESS_MARGIN,
) -> dict[str, Any]:
    if len(snapshots) < 2:
        return {"status": "NEED_MORE_SNAPSHOTS", "arm": arm, "snapshot_count": len(snapshots)}
    common = set(snapshot_channels(snapshots[0], arm))
    for row in snapshots[1:]:
        common &= set(snapshot_channels(row, arm))
    channel_ids = sorted(common)
    if len(channel_ids) < EXPECTED_DOF:
        return {
            "status": "FAIL",
            "reason": "fewer than seven common ROS channels",
            "arm": arm,
            "common_channel_count": len(channel_ids),
        }
    if len(channel_ids) > 10:
        return {
            "status": "FAIL",
            "reason": "too many ambiguous ROS channels; narrow discovery before inference",
            "arm": arm,
            "common_channel_count": len(channel_ids),
        }

    excitation = []
    for sdk_index in range(EXPECTED_DOF):
        values = [float(row["sdk"][arm]["joint7"][sdk_index]) for row in snapshots]
        excitation.append(max(values) - min(values))
    under_excited = [index for index, value in enumerate(excitation) if value < min_motion_rad]
    if under_excited:
        return {
            "status": "NEED_MORE_POSES",
            "reason": "every SDK joint must change enough for delta matching",
            "arm": arm,
            "sdk_joint_excitation_rad": excitation,
            "under_excited_sdk_joints": [f"J{index + 1}" for index in under_excited],
            "min_motion_rad": min_motion_rad,
        }

    fits = [
        [_pair_fit(snapshots, arm, sdk_index, channel_id) for channel_id in channel_ids]
        for sdk_index in range(EXPECTED_DOF)
    ]
    best: tuple[float, tuple[int, ...]] | None = None
    second: tuple[float, tuple[int, ...]] | None = None
    for assignment in itertools.permutations(range(len(channel_ids)), EXPECTED_DOF):
        total = sum(fits[sdk_index][channel_index]["delta_rmse"] for sdk_index, channel_index in enumerate(assignment))
        item = (total, assignment)
        if best is None or total < best[0]:
            second = best
            best = item
        elif second is None or total < second[0]:
            second = item
    assert best is not None
    margin = math.inf if second is None else second[0] - best[0]
    mapping = []
    for sdk_index, channel_index in enumerate(best[1]):
        mapping.append({"sdk_joint": f"J{sdk_index + 1}", "sdk_index": sdk_index, **fits[sdk_index][channel_index]})

    pose_metrics = []
    for pose_index, snapshot in enumerate(snapshots):
        errors = [item["errors_by_pose"][pose_index] for item in mapping]
        sdk_joint7 = [float(value) for value in snapshot["sdk"][arm]["joint7"]]
        reordered_ros_joint7 = []
        for item in mapping:
            channel = snapshot_channels(snapshot, arm)[item["channel_id"]]
            reordered_ros_joint7.append(
                item["sign"] * float(channel["position"]) + item["offset"]
            )
        pose_metrics.append({
            "snapshot": snapshot["snapshot"],
            "sdk_joint7": sdk_joint7,
            "reordered_ros_joint7": reordered_ros_joint7,
            "max_abs_error": max(errors),
            "mean_abs_error": sum(errors) / len(errors),
        })
    maximum_error = max(item["max_abs_error"] for item in pose_metrics)
    status = "PASS"
    reasons = []
    if margin <= uniqueness_margin:
        status = "FAIL"
        reasons.append("mapping assignment is not unique")
    if maximum_error > max_error_rad:
        status = "FAIL"
        reasons.append("reordered ROS values exceed max error threshold")
    return {
        "status": status,
        "arm": arm,
        "snapshot_count": len(snapshots),
        "mapping": mapping,
        "pose_metrics": pose_metrics,
        "assignment_cost": best[0],
        "second_assignment_cost": second[0] if second else None,
        "assignment_margin": margin,
        "thresholds": {
            "max_error_rad": max_error_rad,
            "min_motion_rad": min_motion_rad,
            "uniqueness_margin": uniqueness_margin,
        },
        "reasons": reasons,
        "joint_state_names_nonempty": all(bool(item["name"]) for item in mapping),
    }


def infer_session(session: Path, **kwargs: float) -> dict[str, Any]:
    snapshots = []
    for path in sorted(session.glob("*.json")):
        if path.name == "mapping.json":
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("schema") == "ROS_ARM_JOINT_CALIBRATION_SNAPSHOT_V1":
            snapshots.append(row)
    result = {
        "schema": "ROS_ARM_JOINT_MAPPING_V1",
        "created_at": now_iso(),
        "session": str(session),
        "snapshot_files": [f"{row['snapshot']}.json" for row in snapshots],
        "left": infer_arm_mapping(snapshots, "left", **kwargs),
        "right": infer_arm_mapping(snapshots, "right", **kwargs),
    }
    result["status"] = "PASS" if result["left"]["status"] == result["right"]["status"] == "PASS" else "FAIL"
    write_json(session / "mapping.json", result)
    return result


def _latest_open_session(snapshot: str) -> Path | None:
    if not REPORT_ROOT.exists():
        return None
    candidates = [
        path for path in REPORT_ROOT.iterdir()
        if path.is_dir() and (path / "pose1.json").exists() and not (path / f"{snapshot}.json").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_session(snapshot: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else REPORT_ROOT / requested
    if snapshot != "pose1":
        existing = _latest_open_session(snapshot)
        if existing is not None:
            return existing
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return REPORT_ROOT / stamp


def discover_and_sample(duration_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from rabo_robocap import LinkerArmA7
    except Exception as exc:
        raise RuntimeError(f"Rabo ROS/SDK imports unavailable: {exc!r}") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("act_arm_joint_calibration_readonly")
    latest: dict[str, dict[str, Any]] = {}
    latest_lock = threading.Lock()
    subscriptions = []
    left_arm = right_arm = None
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    try:
        graph_topics = [Topic(name, list(types)) for name, types in node.get_topic_names_and_types()]
        namespace = detect_namespace(graph_topics)
        annotate_topics(graph_topics, namespace)
        candidates = [
            topic for topic in graph_topics
            if topic.primary_type == JOINT_STATE_TYPE and topic.device in {"LEFT_ARM", "RIGHT_ARM"}
        ]
        if not candidates:
            raise RuntimeError("no left/right arm sensor_msgs/msg/JointState topics discovered")

        def make_callback(topic_name: str) -> Any:
            def callback(msg: Any) -> None:
                received = time.monotonic()
                row = {
                    "topic": topic_name,
                    "name": [str(value) for value in msg.name],
                    "position": [float(value) for value in msg.position],
                    "ros_timestamp": ros_stamp(msg),
                    "receive_monotonic": received,
                    "receive_wall_timestamp": now_iso(),
                }
                with latest_lock:
                    latest[topic_name] = row
            return callback

        for topic in candidates:
            subscriptions.append(node.create_subscription(JointState, topic.name, make_callback(topic.name), qos))

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            with latest_lock:
                if all(topic.name in latest for topic in candidates):
                    break
        with latest_lock:
            missing = [topic.name for topic in candidates if topic.name not in latest]
        if missing:
            raise RuntimeError(f"JointState snapshot timeout; missing topics: {missing}")

        left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
        right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
        sdk_started = time.monotonic()
        left_sdk_raw = [float(value) for value in left_arm.get_joint_angles()]
        right_sdk_raw = [float(value) for value in right_arm.get_joint_angles()]
        sdk_finished = time.monotonic()
        if len(left_sdk_raw) < EXPECTED_DOF or len(right_sdk_raw) < EXPECTED_DOF:
            raise RuntimeError(
                "SDK arm state must expose at least 7 values, "
                f"got left={len(left_sdk_raw)} right={len(right_sdk_raw)}"
            )
        left_sdk = left_sdk_raw[:EXPECTED_DOF]
        right_sdk = right_sdk_raw[:EXPECTED_DOF]

        refresh_deadline = time.monotonic() + 0.25
        while time.monotonic() < refresh_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        with latest_lock:
            captured = {key: dict(value) for key, value in latest.items()}

        ros_by_arm: dict[str, Any] = {}
        for arm, device_label in (("left", "LEFT_ARM"), ("right", "RIGHT_ARM")):
            topics = [topic for topic in candidates if topic.device == device_label]
            channels = []
            topic_rows = []
            for topic in topics:
                row = captured[topic.name]
                topic_rows.append(row)
                names = row["name"]
                for index, position in enumerate(row["position"]):
                    channels.append({
                        "topic": topic.name,
                        "index": index,
                        "name": names[index] if index < len(names) else "",
                        "position": position,
                        "ros_timestamp": row["ros_timestamp"],
                        "receive_monotonic": row["receive_monotonic"],
                        "receive_wall_timestamp": row["receive_wall_timestamp"],
                    })
            ros_by_arm[arm] = {"topics": topic_rows, "channels": channels}
        sdk = {
            "left": {"joint7": left_sdk, "raw_value_count": len(left_sdk_raw)},
            "right": {"joint7": right_sdk, "raw_value_count": len(right_sdk_raw)},
            "read_started_monotonic": sdk_started,
            "read_finished_monotonic": sdk_finished,
        }
        discovery = {
            "namespace": namespace,
            "candidate_topics": {
                "left": [topic.name for topic in candidates if topic.device == "LEFT_ARM"],
                "right": [topic.name for topic in candidates if topic.device == "RIGHT_ARM"],
            },
        }
        return {"sdk": sdk, "ros": ros_by_arm}, discovery
    finally:
        for device in (left_arm, right_arm):
            if device is not None and hasattr(device, "shutdown"):
                with contextlib.suppress(Exception):
                    device.shutdown()
        for subscription in subscriptions:
            with contextlib.suppress(Exception):
                node.destroy_subscription(subscription)
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


def synthetic_snapshot(label: str, sdk_left: list[float], sdk_right: list[float]) -> dict[str, Any]:
    permutations = {"left": (3, 0, 6, 2, 5, 1, 4), "right": (5, 2, 0, 6, 1, 4, 3)}
    signs = {"left": (1, -1, 1, 1, -1, 1, 1), "right": (-1, 1, 1, -1, 1, 1, 1)}
    offsets = {"left": (0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.0), "right": (-0.1, 0.0, 0.2, -0.2, 0.1, 0.0, 0.3)}
    out = {"schema": "ROS_ARM_JOINT_CALIBRATION_SNAPSHOT_V1", "snapshot": label, "sdk": {}, "ros": {}}
    for arm, sdk in (("left", sdk_left), ("right", sdk_right)):
        out["sdk"][arm] = {"joint7": sdk}
        channels = []
        for ros_slot, sdk_index in enumerate(permutations[arm]):
            sign = signs[arm][sdk_index]
            value = (sdk[sdk_index] - offsets[arm][sdk_index]) / sign
            channels.append({
                "topic": f"/{arm}/joint_channel_{ros_slot}", "index": 0,
                "name": f"{arm}_physical_joint_{sdk_index + 1}", "position": value,
            })
        out["ros"][arm] = {"channels": channels}
    return out


def synthetic_self_test() -> dict[str, Any]:
    snapshots = [
        synthetic_snapshot("pose1", [0.0, -0.4, 0.2, -0.8, 0.3, 0.6, -0.2], [0.1, -0.3, 0.4, -0.7, 0.2, 0.5, -0.1]),
        synthetic_snapshot("pose2", [0.3, -0.1, -0.2, -0.2, 0.8, 0.1, 0.5], [-0.4, 0.2, 0.1, -0.1, 0.7, -0.2, 0.4]),
        synthetic_snapshot("pose3", [-0.2, 0.3, 0.7, -0.5, -0.1, -0.4, 0.1], [0.6, -0.6, -0.2, 0.3, -0.4, 0.8, 0.2]),
    ]
    left = infer_arm_mapping(snapshots, "left")
    right = infer_arm_mapping(snapshots, "right")
    expected_names = {
        arm: [f"{arm}_physical_joint_{index}" for index in range(1, 8)]
        for arm in ("left", "right")
    }
    checks = {
        "left_pass": left["status"] == "PASS",
        "right_pass": right["status"] == "PASS",
        "left_restored_sdk_order": [row["name"] for row in left.get("mapping", [])] == expected_names["left"],
        "right_restored_sdk_order": [row["name"] for row in right.get("mapping", [])] == expected_names["right"],
        "left_max_error_le_002": max(row["max_abs_error"] for row in left.get("pose_metrics", [])) <= 0.02,
        "right_max_error_le_002": max(row["max_abs_error"] for row in right.get("pose_metrics", [])) <= 0.02,
    }
    return {"result": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "left": left, "right": right}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", help="Snapshot label, normally pose1 then pose2 (or more poses).")
    parser.add_argument("--session", type=Path, help="Existing session directory/name; pose2 auto-selects latest open pose1 session.")
    parser.add_argument("--duration", type=float, default=3.0, help="ROS discovery/sample timeout (default: 3s).")
    parser.add_argument("--max-error-rad", type=float, default=DEFAULT_MAX_ERROR_RAD)
    parser.add_argument("--min-motion-rad", type=float, default=DEFAULT_MIN_MOTION_RAD)
    parser.add_argument("--self-test", action="store_true", help="Run shuffled-channel mapping unit test without ROS/SDK.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        result = synthetic_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["result"] == "PASS" else 2
    if not args.snapshot or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.snapshot):
        raise SystemExit("--snapshot is required and must use only letters, digits, dot, underscore, or dash")
    if args.duration <= 0 or args.max_error_rad <= 0 or args.min_motion_rad <= 0:
        raise SystemExit("duration and thresholds must be > 0")
    session = resolve_session(args.snapshot, args.session)
    session.mkdir(parents=True, exist_ok=True)
    path = session / f"{args.snapshot}.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite snapshot: {path}")
    sampled, discovery = discover_and_sample(args.duration)
    snapshot = {
        "schema": "ROS_ARM_JOINT_CALIBRATION_SNAPSHOT_V1",
        "snapshot": args.snapshot,
        "created_at": now_iso(),
        "wall_timestamp": now_iso(),
        "session": str(session),
        "read_only": True,
        "motion_commands_sent": False,
        "discovery": discovery,
        **sampled,
    }
    write_json(path, snapshot)
    mapping = infer_session(
        session,
        max_error_rad=args.max_error_rad,
        min_motion_rad=args.min_motion_rad,
        uniqueness_margin=DEFAULT_UNIQUENESS_MARGIN,
    )
    print(f"Snapshot: {path}")
    print(f"Session: {session}")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    return 0 if mapping["status"] == "PASS" or len(mapping["snapshot_files"]) < 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
