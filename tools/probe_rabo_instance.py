#!/usr/bin/env python3
"""Read-only Rabo instance inventory and old-to-new migration probe.

This process only inspects the ROS graph, subscribes to JointState messages,
checks safe environment identifiers, and checks whether Rabo packages exist.
It never constructs a robot SDK control client and never calls a command API.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "rabo_instance_probe.json"

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
SAFE_ENV_KEYS = (
    "RABO_SCENE_ID",
    "SCENE_ID",
    "RABO_WORLD_ID",
    "WORLD_ID",
    "ROS_NAMESPACE",
    "ROS_DOMAIN_ID",
    "ROS_DISTRO",
    "RMW_IMPLEMENTATION",
)
UNRESOLVED = "UNRESOLVED"

OLD_INSTANCE = {
    "scene_id": UNRESOLVED,
    "world_id": "ww4b1e6f8392584c89902a6783a9f53075",
    "runtime_namespace": "/gs_1eebee6f37512bbc1d125b25511e912c",
    "left_arm": "rbd03ebf4ebf83c6a6a64754454bc520a",
    "right_arm": "r412d237980e3167577d7aece10f7aedb",
    "left_hand": "r136d7b4b6e527ea3875679b4bf7eeb7d",
    "right_hand": "rcd72e2daf71f064c29aa45d4eeceeca9",
    "top_camera": "/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0",
    "left_wrist_camera": "/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3",
    "right_wrist_camera": "/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc",
    "nut_a_object": "thing_3eb64241-4166-4c4e-9144-edf733c885f1",
    "nut_b_object": "thing_e937f633-faed-4b2e-b609-a7b80825a64a",
    "nut_c_object": "thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7",
    "storage_box_object": "thing_8e768252-b4a8-47d7-82fc-320981b50c01",
}

ENTITY_RE = re.compile(r"^(?P<entity>[^/]+)_tp_(?P<kind>[a-z0-9]+)_", re.IGNORECASE)
FULL_DEVICE_ID_RE = re.compile(r"^r[0-9a-f]{32}$")
FULL_DEVICE_ID_SEARCH_RE = re.compile(r"(?<![0-9a-f])r[0-9a-f]{32}(?![0-9a-f])")
WORLD_PATH_RE = re.compile(r"(?:^|/)world/(?P<world>[^/]+)(?:/|$)", re.IGNORECASE)
SIDE_PATTERNS = {
    "left": re.compile(r"(?:^|[/_.-])left(?:$|[/_.-])", re.IGNORECASE),
    "right": re.compile(r"(?:^|[/_.-])right(?:$|[/_.-])", re.IGNORECASE),
}


def namespace_of(topic: str) -> str | None:
    parts = topic.split("/")
    if len(parts) > 2 and parts[1].startswith("gs_"):
        return "/" + parts[1]
    return None


def topic_entity(topic: str) -> tuple[str | None, str | None]:
    leaf = topic.rstrip("/").rsplit("/", 1)[-1]
    match = ENTITY_RE.match(leaf)
    if not match:
        return None, None
    return match.group("entity"), match.group("kind").lower()


def _normalized_pairs(values: Iterable[Sequence[Any]]) -> list[tuple[str, list[str]]]:
    return sorted((str(name), sorted(str(value) for value in types)) for name, types in values)


def _safe_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {key: str(source[key]) for key in SAFE_ENV_KEYS if source.get(key)}


def _sdk_presence() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("rabo_robocap", "rabo_dev_kit"):
        try:
            spec = importlib.util.find_spec(name)
        except BaseException as exc:
            result[name] = {"available": False, "error": repr(exc)}
            continue
        result[name] = {
            "available": spec is not None,
            "origin": str(spec.origin) if spec is not None and spec.origin else None,
            "note": "package presence only; no SDK object was constructed",
        }
    return result


def _explicit_side(
    entity: str,
    metadata: Sequence[dict[str, Any]],
    graph_strings: Sequence[str],
) -> tuple[str, str | None]:
    for item in metadata:
        identifiers = " ".join(
            str(item.get(key, ""))
            for key in ("device_id", "generated_id", "robot_name", "namespace", "name")
        )
        if entity not in identifiers:
            continue
        side = str(item.get("side", "")).lower()
        if side in SIDE_PATTERNS:
            return side, "OFFICIAL_METADATA"
    matched: set[str] = set()
    for value in graph_strings:
        if entity not in value:
            continue
        for side, pattern in SIDE_PATTERNS.items():
            if pattern.search(value):
                matched.add(side)
    if len(matched) == 1:
        return next(iter(matched)), "ROS_GRAPH_NAME"
    return UNRESOLVED, None


def _metadata_for_entity(entity: str, metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = []
    for item in metadata:
        identifiers = " ".join(
            str(item.get(key, ""))
            for key in ("device_id", "generated_id", "robot_name", "namespace", "name")
        )
        if entity in identifiers:
            matches.append(item)
    return matches[0] if len(matches) == 1 else {}


def _device_id_for_entity(
    entity: str,
    metadata: dict[str, Any],
    graph_strings: Sequence[str],
) -> tuple[str, str | None]:
    metadata_id = metadata.get("device_id")
    if metadata_id:
        return str(metadata_id), "OFFICIAL_METADATA"
    if FULL_DEVICE_ID_RE.fullmatch(entity):
        return entity, "ROS_GENERATED_ENTITY"
    matches = {
        match.group(0)
        for value in graph_strings
        if entity in value
        for match in FULL_DEVICE_ID_SEARCH_RE.finditer(value)
        if match.group(0).startswith(entity)
    }
    if len(matches) == 1:
        return next(iter(matches)), "ROS_GRAPH_NAME"
    return UNRESOLVED, None


def _joint_label(sample: dict[str, Any] | None) -> str:
    if not sample:
        return UNRESOLVED
    names = sample.get("name")
    if not isinstance(names, list) or len(names) != 1:
        return UNRESOLVED
    name = str(names[0])
    patterns = (r"(?:^|[_-])j([1-7])(?:$|[_-])", r"(?:^|[_-])joint[_-]?([1-7])(?:$|[_-])")
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return "J" + match.group(1)
    return UNRESOLVED


def _role_record(candidates: Sequence[dict[str, Any]], side: str) -> dict[str, Any]:
    confirmed = [item for item in candidates if item["side"] == side]
    if len(confirmed) != 1:
        return {
            "status": UNRESOLVED,
            "candidates": [item["generated_id"] for item in candidates],
            "reason": "side/role is not uniquely confirmed by metadata",
        }
    item = dict(confirmed[0])
    item["status"] = "RESOLVED"
    return item


def analyze_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    topics = _normalized_pairs(snapshot.get("topics", []))
    services = _normalized_pairs(snapshot.get("services", []))
    nodes = [(str(name), str(namespace)) for name, namespace in snapshot.get("nodes", [])]
    joint_samples = dict(snapshot.get("joint_samples", {}))
    metadata = list(snapshot.get("device_metadata", []))
    environment = dict(snapshot.get("environment", {}))
    graph_strings = [name for name, _ in topics] + [name for name, _ in services]
    graph_strings += [f"{namespace}/{name}" for name, namespace in nodes]

    namespace_counts = Counter(filter(None, (namespace_of(name) for name, _ in topics)))
    runtime_namespace = namespace_counts.most_common(1)[0][0] if namespace_counts else UNRESOLVED

    topic_records: list[dict[str, Any]] = []
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for name, types in topics:
        entity, kind = topic_entity(name)
        record = {
            "topic": name,
            "message_types": types,
            "namespace": namespace_of(name),
            "generated_entity": entity,
            "interface_kind": kind,
        }
        topic_records.append(record)
        if entity and kind:
            groups[entity][kind].append(record)

    arm_candidates: list[dict[str, Any]] = []
    hand_candidates: list[dict[str, Any]] = []
    for entity, interfaces in sorted(groups.items()):
        joint_topics = [
            item for item in interfaces.get("ps", []) if JOINT_STATE_TYPE in item["message_types"]
        ]
        side, side_source = _explicit_side(entity, metadata, graph_strings)
        item_metadata = _metadata_for_entity(entity, metadata)
        device_id, device_id_source = _device_id_for_entity(
            entity, item_metadata, graph_strings
        )
        common = {
            "generated_id": entity,
            "device_id": device_id,
            "device_id_source": device_id_source,
            "display_name": item_metadata.get("display_name", item_metadata.get("name", UNRESOLVED)),
            "robot_name": item_metadata.get("robot_name", UNRESOLVED),
            "serial": item_metadata.get("serial", item_metadata.get("serial_number", UNRESOLVED)),
            "namespace": namespace_of(joint_topics[0]["topic"]) if joint_topics else runtime_namespace,
            "side": side,
            "side_source": side_source,
            "joint_state_topics": [item["topic"] for item in joint_topics],
            "interface_counts": {key: len(value) for key, value in sorted(interfaces.items())},
        }
        if len(joint_topics) == 7:
            arm_candidates.append(
                {**common, "device_type": item_metadata.get("device_type", "A7_CANDIDATE"),
                 "type_source": "OFFICIAL_METADATA" if item_metadata.get("device_type") else "ROS_7_JOINT_INTERFACES"}
            )
        elif len(joint_topics) >= 10 or (interfaces.get("fts") and interfaces.get("pm")):
            hand_candidates.append(
                {**common, "device_type": item_metadata.get("device_type", "O6_CANDIDATE"),
                 "type_source": "OFFICIAL_METADATA" if item_metadata.get("device_type") else "ROS_HAND_INTERFACE_SHAPE"}
            )

    classified_entities = {
        item["generated_id"] for item in arm_candidates + hand_candidates
    }
    unclassified_entities = [
        {
            "generated_id": entity,
            "namespace": namespace_of(next(iter(next(iter(interfaces.values()))))["topic"]),
            "interface_counts": {
                key: len(value) for key, value in sorted(interfaces.items())
            },
        }
        for entity, interfaces in sorted(groups.items())
        if entity not in classified_entities and interfaces
    ]

    left_arm = _role_record(arm_candidates, "left")
    right_arm = _role_record(arm_candidates, "right")
    left_hand = _role_record(hand_candidates, "left")
    right_hand = _role_record(hand_candidates, "right")

    image_records = [item for item in topic_records if IMAGE_TYPES.intersection(item["message_types"])]
    arm_entities = {item["generated_id"] for item in arm_candidates}
    confirmed_arm_sides = {
        item["generated_id"]: item["side"]
        for item in arm_candidates
        if item["side"] in SIDE_PATTERNS
    }
    for item in image_records:
        roles: list[str] = []
        if item["generated_entity"] in arm_entities:
            roles.append("WRIST_CANDIDATE")
            side = confirmed_arm_sides.get(item["generated_entity"])
            if side:
                roles.append(f"{side.upper()}_WRIST_CANDIDATE")
        else:
            roles.append("TOP_CANDIDATE")
        item["candidate_roles"] = roles
    top_candidates = [item for item in image_records if item["generated_entity"] not in arm_entities]
    top = dict(top_candidates[0]) if len(top_candidates) == 1 else {
        "status": UNRESOLVED,
        "candidates": [item["topic"] for item in top_candidates],
    }
    if len(top_candidates) == 1:
        top["status"] = "RESOLVED"

    wrist_by_entity = {
        item["generated_entity"]: item for item in image_records if item["generated_entity"] in arm_entities
    }
    wrist_roles: dict[str, dict[str, Any]] = {}
    for side, arm in (("left", left_arm), ("right", right_arm)):
        entity = arm.get("generated_id") if arm.get("status") == "RESOLVED" else None
        wrist = wrist_by_entity.get(entity) if entity else None
        wrist_roles[side] = dict(wrist) if wrist else {
            "status": UNRESOLVED,
            "candidates": [item["topic"] for item in wrist_by_entity.values()],
        }
        if wrist:
            wrist_roles[side]["status"] = "RESOLVED"

    world_candidates = {
        match.group("world")
        for name, _ in services
        for match in [WORLD_PATH_RE.search(name)]
        if match
    }
    world_env = environment.get("RABO_WORLD_ID") or environment.get("WORLD_ID")
    world_id = world_env or (next(iter(world_candidates)) if len(world_candidates) == 1 else UNRESOLVED)
    scene_id = environment.get("RABO_SCENE_ID") or environment.get("SCENE_ID") or UNRESOLVED

    arm_state_groups: dict[str, Any] = {}
    hand_state_groups: dict[str, Any] = {}
    for candidate, output in ((arm_candidates, arm_state_groups), (hand_candidates, hand_state_groups)):
        for item in candidate:
            output[item["generated_id"]] = [
                {
                    "topic": topic,
                    "reported_joint_names": joint_samples.get(topic, {}).get("name", []),
                    "sdk_joint": _joint_label(joint_samples.get(topic)),
                }
                for topic in item["joint_state_topics"]
            ]

    hand_entities = {item["generated_id"] for item in hand_candidates}
    hand_related_topics = [
        item
        for item in topic_records
        if item["generated_entity"] in hand_entities
        or re.search(r"(?:^|[/_.-])(hand|o6)(?:$|[/_.-])", item["topic"], re.IGNORECASE)
    ]
    hand_related_services = [
        {"name": name, "types": types}
        for name, types in services
        if any(entity in name for entity in hand_entities)
        or re.search(r"(?:^|[/_.-])(hand|o6)(?:$|[/_.-])", name, re.IGNORECASE)
    ]

    migration: dict[str, dict[str, Any]] = {}
    for key, role in (
        ("left_arm", left_arm), ("right_arm", right_arm),
        ("left_hand", left_hand), ("right_hand", right_hand),
    ):
        migration[key] = {
            "old": OLD_INSTANCE[key],
            "new": role.get("device_id", UNRESOLVED) if role.get("status") == "RESOLVED" else UNRESOLVED,
            "new_generated_id": role.get("generated_id", UNRESOLVED),
        }
    migration["top_camera"] = {
        "old": OLD_INSTANCE["top_camera"],
        "new": top.get("topic", UNRESOLVED) if top.get("status") == "RESOLVED" else UNRESOLVED,
    }
    for key, role in (
        ("left_wrist_camera", wrist_roles["left"]),
        ("right_wrist_camera", wrist_roles["right"]),
    ):
        migration[key] = {
            "old": OLD_INSTANCE[key],
            "new": role.get("topic", UNRESOLVED) if role.get("status") == "RESOLVED" else UNRESOLVED,
        }
    migration["scene_id"] = {"old": OLD_INSTANCE["scene_id"], "new": scene_id}
    migration["world_id"] = {"old": OLD_INSTANCE["world_id"], "new": world_id}
    migration["runtime_namespace"] = {
        "old": OLD_INSTANCE["runtime_namespace"], "new": runtime_namespace
    }
    for key in ("nut_a_object", "nut_b_object", "nut_c_object", "storage_box_object"):
        migration[key] = {"old": OLD_INSTANCE[key], "new": UNRESOLVED}

    unresolved: list[str] = []
    for key in (
        "left_arm", "right_arm", "left_hand", "right_hand", "top_camera",
        "left_wrist_camera", "right_wrist_camera", "scene_id", "world_id",
        "nut_a_object", "nut_b_object", "nut_c_object", "storage_box_object",
    ):
        if migration[key]["new"] == UNRESOLVED:
            unresolved.append(key)
    if any(
        row["sdk_joint"] == UNRESOLVED
        for rows in arm_state_groups.values()
        for row in rows
    ):
        unresolved.append("arm J1..J7 semantic mapping")

    return {
        "probe_mode": "READ_ONLY",
        "scene": {
            "scene_id": scene_id,
            "world_id": world_id,
            "world_candidates_from_services": sorted(world_candidates),
            "runtime_namespace": runtime_namespace,
            "namespace_counts": dict(sorted(namespace_counts.items())),
            "safe_environment": environment,
        },
        "devices": {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "arm_candidates": arm_candidates,
            "hand_candidates": hand_candidates,
            "unclassified_generated_entities": unclassified_entities,
            "hand_related_topics": hand_related_topics,
            "hand_related_services": hand_related_services,
            "official_metadata_records": metadata,
            "metadata_status": (
                "AVAILABLE" if metadata else
                "UNAVAILABLE_FROM_CONFIRMED_READ_ONLY_API; roles remain unresolved unless ROS graph names are explicit"
            ),
            "sdk_packages": snapshot.get("sdk_packages", {}),
        },
        "cameras": {
            "all": image_records,
            "top": top,
            "left_wrist": wrist_roles["left"],
            "right_wrist": wrist_roles["right"],
            "top_candidates": top_candidates,
            "wrist_candidates": list(wrist_by_entity.values()),
        },
        "state_topics": {
            "all_joint_state_topics": [
                item for item in topic_records if JOINT_STATE_TYPE in item["message_types"]
            ],
            "arm_groups": arm_state_groups,
            "hand_groups": hand_state_groups,
        },
        "services": [{"name": name, "types": types} for name, types in services],
        "nodes": [{"name": name, "namespace": namespace} for name, namespace in nodes],
        "migration": migration,
        "unresolved": unresolved,
    }


def collect_ros_snapshot(wait_seconds: float) -> dict[str, Any]:
    try:
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
    except BaseException as exc:
        raise RuntimeError(
            "Rabo ROS read-only probe requires rclpy and sensor_msgs in the platform environment: "
            f"{exc!r}"
        ) from exc

    os.environ.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))
    owns_rclpy = not bool(rclpy.ok())
    if owns_rclpy:
        rclpy.init(args=None)
    node = rclpy.create_node("rabo_instance_probe_read_only")
    subscriptions: list[Any] = []
    samples: dict[str, dict[str, Any]] = {}
    try:
        deadline = time.monotonic() + max(0.0, wait_seconds)
        topics: list[tuple[str, list[str]]] = []
        while True:
            topics = _normalized_pairs(node.get_topic_names_and_types())
            if topics or time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        services = _normalized_pairs(node.get_service_names_and_types())
        nodes = sorted((str(name), str(namespace)) for name, namespace in node.get_node_names_and_namespaces())
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        for topic, types in topics:
            if JOINT_STATE_TYPE not in types:
                continue

            def callback(message: Any, *, selected_topic: str = topic) -> None:
                if selected_topic not in samples:
                    samples[selected_topic] = {
                        "name": [str(value) for value in getattr(message, "name", [])],
                        "position_count": len(getattr(message, "position", [])),
                    }

            subscriptions.append(node.create_subscription(JointState, topic, callback, qos))
        sample_deadline = time.monotonic() + max(0.0, wait_seconds)
        while time.monotonic() < sample_deadline and len(samples) < len(subscriptions):
            rclpy.spin_once(node, timeout_sec=0.1)
        return {
            "topics": topics,
            "services": services,
            "nodes": nodes,
            "joint_samples": samples,
            "device_metadata": [],
            "environment": _safe_environment(),
            "sdk_packages": _sdk_presence(),
        }
    finally:
        for subscription in subscriptions:
            with contextlib.suppress(Exception):
                node.destroy_subscription(subscription)
        with contextlib.suppress(Exception):
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            with contextlib.suppress(Exception):
                rclpy.shutdown()


def _print_role(label: str, value: dict[str, Any]) -> None:
    print(f"{label} candidate:")
    print(f"  status: {value.get('status', 'CANDIDATE')}")
    print(f"  type: {value.get('device_type', UNRESOLVED)}")
    print(f"  name: {value.get('display_name', UNRESOLVED)}")
    print(f"  device_id: {value.get('device_id', UNRESOLVED)}")
    print(f"  generated_id: {value.get('generated_id', UNRESOLVED)}")
    print(f"  serial: {value.get('serial', UNRESOLVED)}")
    print(f"  namespace: {value.get('namespace', UNRESOLVED)}")
    if value.get("candidates"):
        print(f"  candidates: {value['candidates']}")


def print_report(report: dict[str, Any]) -> None:
    print("=" * 40)
    print("RABO INSTANCE PROBE")
    print("=" * 40)
    print("Scene:")
    for key in ("scene_id", "world_id", "runtime_namespace"):
        print(f"  {key}: {report['scene'][key]}")
    print("\nDevices:")
    for key in ("left_arm", "right_arm", "left_hand", "right_hand"):
        _print_role(key.upper(), report["devices"][key])
    print("\nAll arm candidates:")
    for item in report["devices"]["arm_candidates"]:
        print(f"  {item['generated_id']}: {item['device_type']} side={item['side']} topics={len(item['joint_state_topics'])}")
    print("All hand candidates:")
    for item in report["devices"]["hand_candidates"]:
        print(f"  {item['generated_id']}: {item['device_type']} side={item['side']} topics={len(item['joint_state_topics'])}")
    print("SDK packages (presence check only):")
    for name, value in report["devices"]["sdk_packages"].items():
        print(f"  {name}: available={value.get('available', False)} origin={value.get('origin')}")
    print("Hand-related topics/services:")
    for item in report["devices"]["hand_related_topics"]:
        print(f"  topic: {item['topic']} types={item['message_types']}")
    for item in report["devices"]["hand_related_services"]:
        print(f"  service: {item['name']} types={item['types']}")
    print("\nCameras:")
    for item in report["cameras"]["all"]:
        print(
            f"  {item['topic']} types={item['message_types']} "
            f"namespace={item['namespace']} entity={item['generated_entity']} "
            f"roles={item['candidate_roles']}"
        )
    for label in ("top", "left_wrist", "right_wrist"):
        value = report["cameras"][label]
        print(f"  {label.upper()}: {value.get('topic', UNRESOLVED)} status={value.get('status', UNRESOLVED)}")
    print("\nArm state topic groups:")
    for entity, rows in report["state_topics"]["arm_groups"].items():
        print(f"  {entity}:")
        for row in rows:
            print(f"    {row['sdk_joint']}: {row['topic']} names={row['reported_joint_names']}")
    print("=" * 40)
    print("OLD -> NEW MIGRATION")
    print("=" * 40)
    for key, value in report["migration"].items():
        print(f"{key.upper()}:")
        print(f"  old: {value['old']}")
        print(f"  new: {value['new']}")
        if "new_generated_id" in value:
            print(f"  new_generated_id: {value['new_generated_id']}")
    print(f"\nUnresolved: {report['unresolved']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = collect_ros_snapshot(args.wait_seconds)
        report = analyze_snapshot(snapshot)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_report(report)
        print(f"\nJSON: {output}")
        print("Robot commands sent: NO")
        return 0
    except BaseException as exc:
        print(f"PROBE FAULT: {exc}")
        print("Robot commands sent: NO")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
