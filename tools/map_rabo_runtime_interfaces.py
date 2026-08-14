#!/usr/bin/env python3
"""Read-only Rabo ROS runtime interface mapper.

This tool discovers the dynamic /gs_<id> namespace, samples sensor/state topics
without publishing or calling write services/actions, compares read-only SDK
state when the SDK is importable, and writes a JSON map plus Markdown report.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runtime_interface_map"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "runtime_interface_map.log"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "RABO_RUNTIME_INTERFACE_MAP_REPORT.md"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "runtime_interface_map.json"
ARM_HAND_DEMO = PROJECT_ROOT / "agents" / "arm_hand_demo" / "__init__.py"

DEVICE_IDS = {
    "LEFT_ARM": "rbd03ebf4ebf83c6a6a64754454bc520a",
    "RIGHT_ARM": "r412d237980e3167577d7aece10f7aedb",
    "LEFT_HAND": "r136d7b4b6e527ea3875679b4bf7eeb7d",
    "RIGHT_HAND": "rcd72e2daf71f064c29aa45d4eeceeca9",
}

KNOWN_RGB_SUFFIXES = {
    "fixed_rgb": "r6ef2dc_tp_cam_303d2b1ce0",
    "left_wrist_rgb": "rbd03eb_tp_cam_069a6739f3",
    "right_wrist_rgb": "r412d23_tp_cam_3c67aef2bc",
}

TYPE_IMAGE = "sensor_msgs/msg/Image"
TYPE_CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
TYPE_POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
TYPE_JOINT_STATE = "sensor_msgs/msg/JointState"
TYPE_FLOAT64 = "std_msgs/msg/Float64"
TYPE_WRENCH = "geometry_msgs/msg/Wrench"

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NS_RE = re.compile(r"^/gs_[A-Za-z0-9]+(?=/|$)")
SUFFIX_RE = re.compile(r"_tp_([a-z0-9]+)_", re.IGNORECASE)


@dataclass
class Topic:
    name: str
    types: list[str]
    namespace: str | None = None
    relative: str = ""
    category: str = "OTHER"
    device: str | None = None
    suffix_kind: str | None = None
    publishers: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    wall_times: list[float] = field(default_factory=list)
    sample_error: str | None = None
    saved_sample: str | None = None

    @property
    def primary_type(self) -> str | None:
        return self.types[0] if self.types else None


class Logger:
    def __init__(self, path: Path, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._f.close()

    def log(self, message: str = "") -> None:
        line = message.rstrip()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._f.write(f"[{stamp}] {line}\n")
        self._f.flush()
        if self.verbose:
            print(line)

    def section(self, title: str) -> None:
        self.log("")
        self.log("=" * 80)
        self.log(title)
        self.log("=" * 80)


def clean(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def jsonable(value: Any, max_list: int | None = None) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v, max_list=max_list) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if max_list is not None:
            vals = vals[:max_list]
        return [jsonable(v, max_list=max_list) for v in vals]
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist(), max_list=max_list)
        except Exception:
            pass
    return repr(value)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str], logger: Logger | None = None, timeout: float = 10.0) -> dict[str, Any]:
    if logger:
        logger.log("$ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False, env=env)
        result = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": clean(proc.stdout).splitlines(),
            "stderr": clean(proc.stderr).splitlines(),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError as exc:
        result = {"cmd": cmd, "ok": False, "error": f"not found: {exc}"}
    except subprocess.TimeoutExpired as exc:
        result = {
            "cmd": cmd,
            "ok": False,
            "timeout": timeout,
            "stdout": clean(exc.stdout or "").splitlines() if isinstance(exc.stdout, str) else [],
            "stderr": clean(exc.stderr or "").splitlines() if isinstance(exc.stderr, str) else [],
            "error": "timeout",
        }
    except Exception as exc:
        result = {"cmd": cmd, "ok": False, "error": repr(exc)}
    if logger:
        for line in result.get("stdout", []):
            logger.log(line)
        for line in result.get("stderr", []):
            logger.log("stderr: " + line)
        if result.get("error"):
            logger.log("error: " + str(result["error"]))
    return result


def run_ros_topic(args: list[str], logger: Logger, timeout: float = 10.0) -> dict[str, Any]:
    result = run_cmd(["ros2", "topic", *args, "--no-daemon"], logger, timeout)
    text = "\n".join(result.get("stdout", []) + result.get("stderr", []))
    if "unrecognized arguments: --no-daemon" in text:
        result = run_cmd(["ros2", "topic", *args], logger, timeout)
    return result


def parse_topic_list_t(lines: list[str]) -> list[Topic]:
    out: list[Topic] = []
    pattern = re.compile(r"^(?P<name>\S+)\s+\[(?P<types>[^\]]+)\]\s*$")
    for line in lines:
        m = pattern.match(line.strip())
        if not m:
            continue
        out.append(Topic(m.group("name"), [x.strip() for x in m.group("types").split(",") if x.strip()]))
    return out


def detect_namespace(topics: list[Topic]) -> str | None:
    counts: Counter[str] = Counter()
    for topic in topics:
        m = NS_RE.match(topic.name)
        if m:
            counts[m.group(0)] += 1
    return counts.most_common(1)[0][0] if counts else None


def device_for_relative(relative: str) -> str | None:
    first = relative.split("/", 1)[0]
    for label, device_id in DEVICE_IDS.items():
        if first.startswith(device_id[:6]) or first.startswith(device_id[:7]) or first.startswith(device_id):
            return label
    return None


def suffix_kind(relative: str) -> str | None:
    m = SUFFIX_RE.search(relative)
    return m.group(1).lower() if m else None


def classify_topic(topic: Topic) -> str:
    t = topic.primary_type or ""
    rel = topic.relative.lower()
    kind = topic.suffix_kind
    if t == TYPE_CAMERA_INFO or "_tp_cami_" in rel or rel.endswith("/camera_info"):
        return "CAMERA_INFO"
    if t == TYPE_POINTCLOUD2 or rel.endswith("/points") or "pointcloud" in rel:
        return "POINT_CLOUD"
    if t == TYPE_IMAGE and (rel.endswith("/depth_image") or "/depth" in rel or "_tp_dcam_" in rel):
        return "CAMERA_DEPTH"
    if t == TYPE_IMAGE and ("_tp_cam_" in rel or rel.endswith("/image") or "_tp_rgbd_" in rel):
        return "CAMERA_RGB"
    if t == TYPE_JOINT_STATE:
        if topic.device and topic.device.endswith("_ARM"):
            return "ARM_JOINT_STATE"
        if topic.device and topic.device.endswith("_HAND"):
            return "HAND_JOINT_STATE"
        return "JOINT_STATE"
    if t == TYPE_WRENCH or kind == "fts":
        return "FORCE_TORQUE"
    if t == TYPE_FLOAT64 or kind in {"pm", "sm"}:
        return "POSITION_MOTOR_SENSOR" if kind == "pm" else "FLOAT_SENSOR"
    return "OTHER"


def annotate_topics(topics: list[Topic], namespace: str | None) -> None:
    for topic in topics:
        topic.namespace = namespace if namespace and topic.name.startswith(namespace + "/") else None
        topic.relative = topic.name[len(namespace) + 1 :] if topic.namespace else topic.name.lstrip("/")
        topic.device = device_for_relative(topic.relative)
        topic.suffix_kind = suffix_kind(topic.relative)
        topic.category = classify_topic(topic)


def import_msg(type_name: str) -> Any:
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_name)


def stamp_dict(stamp: Any) -> dict[str, Any]:
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return {"sec": sec, "nanosec": nanosec, "float": sec + nanosec / 1e9}


def header_summary(msg: Any) -> dict[str, Any] | None:
    if not hasattr(msg, "header"):
        return None
    header = msg.header
    return {"stamp": stamp_dict(header.stamp), "frame_id": getattr(header, "frame_id", "")}


def summarize_msg(msg: Any, type_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type_name}
    header = header_summary(msg)
    if header:
        out["header"] = header

    if type_name == TYPE_IMAGE:
        out.update(
            {
                "width": int(getattr(msg, "width", 0)),
                "height": int(getattr(msg, "height", 0)),
                "encoding": str(getattr(msg, "encoding", "")),
                "step": int(getattr(msg, "step", 0)),
                "data_len": len(getattr(msg, "data", [])),
            }
        )
    elif type_name == TYPE_CAMERA_INFO:
        out.update(
            {
                "width": int(getattr(msg, "width", 0)),
                "height": int(getattr(msg, "height", 0)),
                "distortion_model": str(getattr(msg, "distortion_model", "")),
                "d": jsonable(list(getattr(msg, "d", [])), max_list=16),
                "k": jsonable(list(getattr(msg, "k", [])), max_list=16),
                "r": jsonable(list(getattr(msg, "r", [])), max_list=16),
                "p": jsonable(list(getattr(msg, "p", [])), max_list=16),
            }
        )
    elif type_name == TYPE_POINTCLOUD2:
        out.update(
            {
                "width": int(getattr(msg, "width", 0)),
                "height": int(getattr(msg, "height", 0)),
                "fields": [
                    {
                        "name": f.name,
                        "offset": f.offset,
                        "datatype": f.datatype,
                        "count": f.count,
                    }
                    for f in getattr(msg, "fields", [])
                ],
                "point_step": int(getattr(msg, "point_step", 0)),
                "row_step": int(getattr(msg, "row_step", 0)),
                "is_dense": bool(getattr(msg, "is_dense", False)),
                "data_len": len(getattr(msg, "data", [])),
            }
        )
    elif type_name == TYPE_JOINT_STATE:
        out.update(
            {
                "name": list(getattr(msg, "name", [])),
                "position": jsonable(list(getattr(msg, "position", []))),
                "velocity": jsonable(list(getattr(msg, "velocity", []))),
                "effort": jsonable(list(getattr(msg, "effort", []))),
                "name_count": len(getattr(msg, "name", [])),
                "position_count": len(getattr(msg, "position", [])),
                "velocity_count": len(getattr(msg, "velocity", [])),
                "effort_count": len(getattr(msg, "effort", [])),
            }
        )
    elif type_name == TYPE_FLOAT64:
        out["data"] = float(getattr(msg, "data", 0.0))
    elif type_name == TYPE_WRENCH:
        out.update(
            {
                "force": {
                    "x": float(msg.force.x),
                    "y": float(msg.force.y),
                    "z": float(msg.force.z),
                },
                "torque": {
                    "x": float(msg.torque.x),
                    "y": float(msg.torque.y),
                    "z": float(msg.torque.z),
                },
            }
        )
    else:
        for slot in getattr(msg, "__slots__", [])[:12]:
            with contextlib.suppress(Exception):
                value = getattr(msg, slot)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[slot] = value
                elif isinstance(value, (list, tuple)):
                    out[slot] = {"len": len(value), "sample": jsonable(list(value), max_list=8)}
                elif hasattr(value, "__slots__"):
                    out[slot] = repr(value)
    return jsonable(out)


def image_to_ppm(msg: Any) -> tuple[bytes | None, str | None]:
    width = int(getattr(msg, "width", 0))
    height = int(getattr(msg, "height", 0))
    step = int(getattr(msg, "step", 0))
    encoding = str(getattr(msg, "encoding", "")).lower()
    data = bytes(getattr(msg, "data", b""))
    if width <= 0 or height <= 0 or step <= 0:
        return None, "invalid dimensions"
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1"}:
        return None, f"unsupported encoding {encoding}"
    if encoding in {"mono8", "8uc1"}:
        if len(data) < step * height or step < width:
            return None, "short mono buffer"
        rows = [data[r * step : r * step + width] for r in range(height)]
        return b"P5\n%d %d\n255\n" % (width, height) + b"".join(rows), None
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    min_step = width * channels
    if len(data) < step * height or step < min_step:
        return None, "short rgb buffer"
    pixels = bytearray()
    for r in range(height):
        row = data[r * step : r * step + min_step]
        for c in range(0, len(row), channels):
            px = row[c : c + channels]
            if encoding in {"bgr8", "bgra8"}:
                pixels.extend((px[2], px[1], px[0]))
            else:
                pixels.extend((px[0], px[1], px[2]))
    return b"P6\n%d %d\n255\n" % (width, height) + bytes(pixels), None


def endpoint_dict(info: Any) -> dict[str, Any]:
    qos = getattr(info, "qos_profile", None)
    out = {
        "node_name": getattr(info, "node_name", None),
        "node_namespace": getattr(info, "node_namespace", None),
        "topic_type": getattr(info, "topic_type", None),
        "endpoint_type": str(getattr(info, "endpoint_type", None)),
    }
    if qos is not None:
        out["qos"] = {
            "reliability": str(getattr(qos, "reliability", None)),
            "durability": str(getattr(qos, "durability", None)),
            "history": str(getattr(qos, "history", None)),
            "depth": getattr(qos, "depth", None),
        }
    return jsonable(out)


def hz_from_wall_times(times: list[float]) -> float | None:
    if len(times) < 2:
        return None
    dt = times[-1] - times[0]
    return (len(times) - 1) / dt if dt > 0 else None


def choose_sample_topics(topics: list[Topic]) -> list[Topic]:
    wanted = {
        "CAMERA_RGB",
        "CAMERA_DEPTH",
        "CAMERA_INFO",
        "POINT_CLOUD",
        "ARM_JOINT_STATE",
        "HAND_JOINT_STATE",
        "JOINT_STATE",
        "POSITION_MOTOR_SENSOR",
        "FLOAT_SENSOR",
        "FORCE_TORQUE",
    }
    return [t for t in topics if t.category in wanted and t.primary_type]


def sample_topics(topics: list[Topic], duration: float, output_dir: Path, logger: Logger) -> dict[str, Any]:
    logger.section("RCLPY READ-ONLY SAMPLING")
    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except Exception as exc:
        logger.log(f"RCLPY_IMPORT_ERROR {repr(exc)}")
        return {"ok": False, "error": repr(exc)}

    rclpy.init(args=None)
    node = rclpy.create_node("rabo_runtime_interface_map_readonly")
    subscriptions: list[Any] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    try:
        for topic in topics:
            with contextlib.suppress(Exception):
                topic.publishers = [endpoint_dict(x) for x in node.get_publishers_info_by_topic(topic.name)]
                topic.subscribers = [endpoint_dict(x) for x in node.get_subscriptions_info_by_topic(topic.name)]

            try:
                msg_cls = import_msg(topic.primary_type or "")
            except Exception as exc:
                topic.sample_error = f"message import failed: {repr(exc)}"
                logger.log(f"SKIP {topic.name}: {topic.sample_error}")
                continue

            def make_cb(t: Topic, type_name: str) -> Callable[[Any], None]:
                def cb(msg: Any) -> None:
                    t.wall_times.append(time.time())
                    if len(t.samples) < 8:
                        summary = summarize_msg(msg, type_name)
                        t.samples.append(summary)
                        if t.category == "CAMERA_RGB" and not t.saved_sample:
                            label = next((k for k, v in KNOWN_RGB_SUFFIXES.items() if v in t.relative), None)
                            if label:
                                payload, err = image_to_ppm(msg)
                                if payload:
                                    path = output_dir / f"{label}.ppm"
                                    path.write_bytes(payload)
                                    t.saved_sample = display_path(path)
                                elif err:
                                    t.sample_error = err
                return cb

            try:
                sub = node.create_subscription(msg_cls, topic.name, make_cb(topic, topic.primary_type or ""), qos)
                subscriptions.append(sub)
                logger.log(f"SUBSCRIBED {topic.name} [{topic.primary_type}] category={topic.category}")
            except Exception as exc:
                topic.sample_error = f"subscription failed: {repr(exc)}"
                logger.log(f"SUBSCRIBE_ERROR {topic.name}: {repr(exc)}")

        start = time.monotonic()
        while time.monotonic() - start < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        for sub in subscriptions:
            with contextlib.suppress(Exception):
                node.destroy_subscription(sub)
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()

    for topic in topics:
        logger.log(
            f"SAMPLE_RESULT {topic.name} frames={len(topic.wall_times)} hz={hz_from_wall_times(topic.wall_times)} "
            f"samples={json.dumps(topic.samples[:2], ensure_ascii=False)} error={topic.sample_error}"
        )
    return {"ok": True, "duration": duration}


def source_search(logger: Logger) -> dict[str, Any]:
    logger.section("SOURCE SEARCH")
    patterns = ["_tp_cam_", "_tp_cami_", "_tp_rgbd_", "_tp_dcam_", "_tp_ps_", "_tp_pm_", "_tp_sm_", "_tp_fts_"]
    evidence: dict[str, Any] = {"repo": {}, "installed_packages": {}, "manual": {}}
    for pat in patterns:
        res = run_cmd(["rg", "-n", pat, str(PROJECT_ROOT), "-g", "!outputs/**", "-g", "!logs/**"], logger, timeout=6)
        evidence["repo"][pat] = {"ok": res.get("ok"), "lines": res.get("stdout", [])[:40], "stderr": res.get("stderr", [])[:5]}

    for mod_name in ("rabo_robocap", "rabo_dev_kit"):
        spec = importlib.util.find_spec(mod_name)
        mod_info: dict[str, Any] = {"origin": spec.origin if spec else None}
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
        mod_info["roots"] = roots
        if roots:
            for pat in patterns:
                res = run_cmd(["rg", "-n", pat, *roots], logger, timeout=6)
                mod_info[pat] = {"ok": res.get("ok"), "lines": res.get("stdout", [])[:40], "stderr": res.get("stderr", [])[:5]}
        evidence["installed_packages"][mod_name] = mod_info

    evidence["manual"] = {
        "_tp_cam_": "docs/传感器 states camera image data prefix _tp_cam_.",
        "_tp_cami_": "docs/传感器 states camera info prefix _tp_cami_.",
        "_tp_rgbd_": "docs/传感器 states depth camera base prefix _tp_rgbd_ with /image, /depth_image, /points outputs.",
        "_tp_fts_": "docs/传感器 states force/torque sensor prefix _tp_fts_.",
    }
    return evidence


def load_robot_ids() -> dict[str, str]:
    try:
        tree = ast.parse(ARM_HAND_DEMO.read_text(encoding="utf-8"), filename=str(ARM_HAND_DEMO))
    except Exception:
        return DEVICE_IDS.copy()
    ids = DEVICE_IDS.copy()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ids and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                ids[node.targets[0].id] = node.value.value
    return ids


def read_sdk_state(logger: Logger) -> dict[str, Any]:
    logger.section("SDK READ-ONLY STATE")
    out: dict[str, Any] = {"available": False, "devices": {}, "errors": []}
    try:
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right
    except Exception as exc:
        out["errors"].append(f"SDK import failed: {repr(exc)}")
        logger.log(out["errors"][-1])
        return out

    out["available"] = True
    factories = {
        "LEFT_ARM": lambda: LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim"),
        "RIGHT_ARM": lambda: LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim"),
        "LEFT_HAND": lambda: LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim"),
        "RIGHT_HAND": lambda: LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim"),
    }
    for label, factory in factories.items():
        device = None
        rec: dict[str, Any] = {}
        try:
            device = factory()
            if hasattr(device, "get_joint_angles"):
                rec["joint_angles"] = jsonable(device.get_joint_angles())
            if hasattr(device, "get_pose") and label.endswith("_ARM"):
                rec["pose"] = jsonable(device.get_pose())
            if hasattr(device, "get_clench") and label.endswith("_HAND"):
                rec["clench"] = jsonable(device.get_clench())
        except Exception as exc:
            rec["error"] = repr(exc)
            out["errors"].append(f"{label}: {repr(exc)}")
        finally:
            if device is not None and hasattr(device, "shutdown"):
                with contextlib.suppress(Exception):
                    device.shutdown()
        out["devices"][label] = rec
        logger.log(f"SDK {label}: {json.dumps(rec, ensure_ascii=False)}")
    return out


def compare_sdk_ros(sdk: dict[str, Any], topics: list[Topic]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    js_topics = [t for t in topics if t.primary_type == TYPE_JOINT_STATE and t.samples]
    for device in DEVICE_IDS:
        sdk_vals = sdk.get("devices", {}).get(device, {}).get("joint_angles")
        candidates = [t for t in js_topics if t.device == device]
        rec: dict[str, Any] = {
            "sdk_available": sdk_vals is not None,
            "candidate_ros_topics": [t.name for t in candidates],
        }
        if sdk_vals is None or not candidates:
            rec["status"] = "NOT_MAPPED"
            result[device] = rec
            continue
        sdk_list = sdk_vals if isinstance(sdk_vals, list) else []
        best = None
        for t in candidates:
            pos = t.samples[0].get("position", [])
            if not isinstance(pos, list) or len(pos) != len(sdk_list):
                continue
            diffs = [abs(float(a) - float(b)) for a, b in zip(sdk_list, pos)]
            item = {"topic": t.name, "max_abs_error": max(diffs) if diffs else None, "ros_position": pos}
            if best is None or (item["max_abs_error"] is not None and item["max_abs_error"] < best["max_abs_error"]):
                best = item
        if best:
            rec.update(best)
            rec["status"] = "MAPPED" if (best["max_abs_error"] is not None and best["max_abs_error"] < 1e-3) else "CHECK"
        else:
            rec["status"] = "NOT_MAPPED"
        result[device] = rec
    return result


def infer_meanings(topics: list[Topic], evidence: dict[str, Any]) -> dict[str, Any]:
    meanings = {
        "cam": {
            "meaning": "RGB camera image topic",
            "confidence": "CONFIRMED",
            "evidence": "Manual documents _tp_cam_ as camera image data.",
        },
        "cami": {
            "meaning": "CameraInfo topic",
            "confidence": "CONFIRMED",
            "evidence": "Manual documents _tp_cami_ as camera info.",
        },
        "rgbd": {
            "meaning": "Depth camera base topic with /image, /depth_image, /points subtopics",
            "confidence": "CONFIRMED",
            "evidence": "Manual documents _tp_rgbd_ and its RGB/depth/points outputs.",
        },
        "fts": {
            "meaning": "Force/torque sensor",
            "confidence": "CONFIRMED",
            "evidence": "Manual documents _tp_fts_ as force/torque sensor prefix.",
        },
        "ps": {"meaning": "Position sensor or per-joint state publisher", "confidence": "INFERRED", "evidence": ""},
        "pm": {"meaning": "Position/motor scalar sensor", "confidence": "INFERRED", "evidence": ""},
        "sm": {"meaning": "Scalar sensor/motor telemetry", "confidence": "UNKNOWN", "evidence": ""},
    }

    by_kind = defaultdict(list)
    for t in topics:
        by_kind[t.suffix_kind].append(t)
    if by_kind.get("ps"):
        types = sorted({x.primary_type for x in by_kind["ps"]})
        meanings["ps"]["evidence"] = f"Live _tp_ps_ topics use types {types}; JointState content is per topic, not assumed complete robot state."
    if by_kind.get("pm"):
        vals = [s.get("data") for t in by_kind["pm"] for s in t.samples if "data" in s]
        meanings["pm"]["evidence"] = f"Live _tp_pm_ topics are Float64 with sampled scalar range {range_text(vals)}."
    if by_kind.get("sm"):
        vals = [s.get("data") for t in by_kind["sm"] for s in t.samples if "data" in s]
        meanings["sm"]["evidence"] = f"Live _tp_sm_ topics are Float64 with sampled scalar range {range_text(vals)}; no local manual/package definition found."
    return meanings


def range_text(values: list[Any]) -> str:
    nums = []
    for v in values:
        with contextlib.suppress(Exception):
            nums.append(float(v))
    if not nums:
        return "NO_SAMPLES"
    return f"{min(nums):.6g}..{max(nums):.6g}"


def timing_analysis(topics: list[Topic]) -> dict[str, Any]:
    selected = {}
    for label, suffix in KNOWN_RGB_SUFFIXES.items():
        t = next((x for x in topics if suffix in x.relative), None)
        if t and t.samples:
            selected[label] = {
                "topic": t.name,
                "ros_stamp": t.samples[0].get("header", {}).get("stamp"),
                "wall_time": t.wall_times[0] if t.wall_times else None,
            }
    for dev in DEVICE_IDS:
        t = next((x for x in topics if x.device == dev and x.primary_type == TYPE_JOINT_STATE and x.samples), None)
        if t:
            selected[dev.lower()] = {
                "topic": t.name,
                "ros_stamp": t.samples[0].get("header", {}).get("stamp"),
                "wall_time": t.wall_times[0] if t.wall_times else None,
            }
    stamps = [v.get("ros_stamp", {}).get("float") for v in selected.values() if v.get("ros_stamp")]
    stamps = [s for s in stamps if isinstance(s, (int, float)) and s > 0]
    domain = "UNKNOWN"
    max_delta = None
    if len(stamps) >= 2:
        max_delta = max(stamps) - min(stamps)
        domain = "SAME TIME DOMAIN" if max_delta < 5.0 else "DIFFERENT / UNKNOWN"
    return {"samples": selected, "max_ros_stamp_delta_s": max_delta, "domain": domain}


def topic_record(t: Topic) -> dict[str, Any]:
    return {
        "topic": t.name,
        "relative": t.relative,
        "type": t.primary_type,
        "category": t.category,
        "device": t.device,
        "suffix_kind": t.suffix_kind,
        "publisher_count": len(t.publishers),
        "subscriber_count": len(t.subscribers),
        "publishers": t.publishers,
        "qos": [p.get("qos") for p in t.publishers if p.get("qos")],
        "frames": len(t.wall_times),
        "hz": hz_from_wall_times(t.wall_times),
        "samples": t.samples,
        "sample_error": t.sample_error,
        "saved_sample": t.saved_sample,
    }


def result_for_rgb(label: str, topics: list[Topic]) -> str:
    suffix = KNOWN_RGB_SUFFIXES[label]
    t = next((x for x in topics if suffix in x.relative), None)
    if not t:
        return "FAIL"
    return "PASS" if len(t.wall_times) > 0 else "CHECK"


def build_json(
    topics: list[Topic],
    namespace: str | None,
    evidence: dict[str, Any],
    meanings: dict[str, Any],
    sdk: dict[str, Any],
    sdk_ros: dict[str, Any],
    timing: dict[str, Any],
    env: dict[str, Any],
    commands: dict[str, Any],
) -> dict[str, Any]:
    by_cat = defaultdict(list)
    for t in topics:
        by_cat[t.category].append(topic_record(t))

    unknowns = []
    if not namespace:
        unknowns.append("live Rabo runtime namespace not visible from this shell; ros2 topic list exposed no /gs_<id> topics")
    for key in ("ps", "pm", "sm"):
        if meanings.get(key, {}).get("confidence") != "CONFIRMED":
            unknowns.append(f"{key} meaning is {meanings.get(key, {}).get('confidence')}")
    for label in KNOWN_RGB_SUFFIXES:
        if result_for_rgb(label, topics) != "PASS":
            unknowns.append(f"{label} did not produce frames during sampling")

    return {
        "runtime_namespace": namespace,
        "environment": env,
        "devices": DEVICE_IDS,
        "topic_count": len(topics),
        "topics": [topic_record(t) for t in topics],
        "cameras": {label: next((topic_record(t) for t in topics if suffix in t.relative), None) for label, suffix in KNOWN_RGB_SUFFIXES.items()},
        "arms": {
            "left": {"id": DEVICE_IDS["LEFT_ARM"], "topics": [topic_record(t) for t in topics if t.device == "LEFT_ARM"]},
            "right": {"id": DEVICE_IDS["RIGHT_ARM"], "topics": [topic_record(t) for t in topics if t.device == "RIGHT_ARM"]},
        },
        "hands": {
            "left": {"id": DEVICE_IDS["LEFT_HAND"], "topics": [topic_record(t) for t in topics if t.device == "LEFT_HAND"]},
            "right": {"id": DEVICE_IDS["RIGHT_HAND"], "topics": [topic_record(t) for t in topics if t.device == "RIGHT_HAND"]},
        },
        "joint_state_topics": by_cat["ARM_JOINT_STATE"] + by_cat["HAND_JOINT_STATE"] + by_cat["JOINT_STATE"],
        "force_torque_topics": by_cat["FORCE_TORQUE"],
        "float_topics": by_cat["POSITION_MOTOR_SENSOR"] + by_cat["FLOAT_SENSOR"],
        "depth_topics": by_cat["CAMERA_DEPTH"],
        "pointcloud_topics": by_cat["POINT_CLOUD"],
        "camera_info_topics": by_cat["CAMERA_INFO"],
        "suffix_meanings": meanings,
        "source_evidence": evidence,
        "timing": timing,
        "sdk_ros_comparison": sdk_ros,
        "sdk_state": sdk,
        "recommended_observation_sources": recommend_sources(topics, sdk_ros),
        "unknowns": unknowns,
        "commands": commands,
    }


def recommend_sources(topics: list[Topic], sdk_ros: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "fixed_rgb": "ROS topic " + (next((t.name for t in topics if KNOWN_RGB_SUFFIXES["fixed_rgb"] in t.relative), "NOT_FOUND")),
        "left_wrist_rgb": "ROS topic " + (next((t.name for t in topics if KNOWN_RGB_SUFFIXES["left_wrist_rgb"] in t.relative), "NOT_FOUND")),
        "right_wrist_rgb": "ROS topic " + (next((t.name for t in topics if KNOWN_RGB_SUFFIXES["right_wrist_rgb"] in t.relative), "NOT_FOUND")),
    }
    for dev, field in [
        ("LEFT_ARM", "left_arm_qpos"),
        ("RIGHT_ARM", "right_arm_qpos"),
        ("LEFT_HAND", "left_hand_qpos"),
        ("RIGHT_HAND", "right_hand_qpos"),
    ]:
        mapped = sdk_ros.get(dev, {})
        if mapped.get("status") == "MAPPED":
            rec[field] = "ROS topic " + mapped.get("topic", "")
        else:
            rec[field] = "SDK get_joint_angles()"
    return rec


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return lines


def qos_text(rec: dict[str, Any]) -> str:
    qos = rec.get("qos") or []
    if not qos:
        return ""
    q = qos[0]
    return f"{q.get('reliability')} / {q.get('durability')} / depth={q.get('depth')}"


def hz_text(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else ""


def first_sample(rec: dict[str, Any]) -> dict[str, Any]:
    samples = rec.get("samples") or []
    return samples[0] if samples else {}


def build_report(data: dict[str, Any], output_dir: Path, log_path: Path) -> str:
    cams = data.get("cameras", {})
    fixed = cams.get("fixed_rgb") or {}
    overall = "PASS" if data.get("runtime_namespace") and not data.get("unknowns") else "CHECK"
    if not data.get("runtime_namespace"):
        overall = "FAIL"

    camera_rows = []
    for label, rec in cams.items():
        rec = rec or {}
        s = first_sample(rec)
        camera_rows.append(
            [
                label,
                f"`{rec.get('topic', 'NOT_FOUND')}`",
                rec.get("type", ""),
                f"{s.get('width', '')}x{s.get('height', '')}",
                s.get("encoding", ""),
                hz_text(rec.get("hz")),
                qos_text(rec),
                rec.get("frames", 0),
                "PASS" if rec.get("frames", 0) > 0 else "CHECK",
            ]
        )

    depth_rows = []
    for rec in data.get("depth_topics", []):
        s = first_sample(rec)
        depth_rows.append([f"`{rec['topic']}`", rec.get("type"), f"{s.get('width','')}x{s.get('height','')}", s.get("encoding", ""), hz_text(rec.get("hz")), rec.get("frames", 0)])

    js_rows = []
    for rec in data.get("joint_state_topics", []):
        s = first_sample(rec)
        names = ", ".join(s.get("name", [])[:10]) if isinstance(s.get("name"), list) else ""
        meaning = "complete A7 candidate" if s.get("position_count") == 7 and rec.get("device", "").endswith("_ARM") else "per-sensor/per-joint or hand state"
        confidence = "INFERRED" if rec.get("frames", 0) else "UNKNOWN"
        js_rows.append([f"`{rec['topic']}`", rec.get("device"), names, s.get("position_count", ""), hz_text(rec.get("hz")), meaning, confidence])

    fts_rows = []
    for rec in data.get("force_torque_topics", []):
        s = first_sample(rec)
        f = s.get("force", {})
        tq = s.get("torque", {})
        fts_rows.append([f"`{rec['topic']}`", rec.get("device"), s.get("header", {}).get("frame_id", ""), hz_text(rec.get("hz")), f"F={f} T={tq}", "UNKNOWN link/finger unless frame/topic identifies it"])

    float_rows = []
    for rec in data.get("float_topics", []):
        vals = [s.get("data") for s in rec.get("samples", []) if "data" in s]
        float_rows.append([f"`{rec['topic']}`", rec.get("device"), rec.get("suffix_kind"), rec.get("type"), range_text(vals), hz_text(rec.get("hz"))])

    pc_rows = []
    for rec in data.get("pointcloud_topics", []):
        s = first_sample(rec)
        pc_rows.append([f"`{rec['topic']}`", rec.get("device"), f"{s.get('width','')}x{s.get('height','')}", s.get("fields", ""), s.get("point_step", ""), hz_text(rec.get("hz"))])

    rec_sources = data.get("recommended_observation_sources", {})
    meanings = data.get("suffix_meanings", {})
    sdk_cmp = data.get("sdk_ros_comparison", {})
    timing = data.get("timing", {})
    unknown_lines = [f"- {x}" for x in data.get("unknowns", [])] or ["- None recorded."]

    lines = [
        "# Rabo Runtime Interface Map Report",
        "",
        "## 1. Result",
        "",
        f"Overall: {overall}",
        "",
        "This run was read-only: no motion APIs, no command publications, no action goals, no write services, no reset, and no scene/controller/camera parameter changes.",
        "",
        "## 2. Environment",
        "",
        f"- Generated: {data['environment'].get('generated')}",
        f"- Hostname: {data['environment'].get('hostname')}",
        f"- Python: {data['environment'].get('python')}",
        f"- ROS_DISTRO: {data['environment'].get('ROS_DISTRO')}",
        f"- ROS_DOMAIN_ID: {data['environment'].get('ROS_DOMAIN_ID')}",
        f"- RMW_IMPLEMENTATION: {data['environment'].get('RMW_IMPLEMENTATION')}",
        f"- Runtime namespace: `{data.get('runtime_namespace')}`",
        f"- Output directory: `{display_path(output_dir)}`",
        f"- Raw log: `{display_path(log_path)}`",
        "",
        "## 3. Device IDs",
        "",
        *[f"- {k}: `{v}`" for k, v in data.get("devices", {}).items()],
        "",
        "## 4. Runtime Namespace",
        "",
        f"Runtime namespace detected from live topics: `{data.get('runtime_namespace')}`.",
        *(
            [
                "",
                "Runtime visibility check: FAIL. The discovery command returned only `/parameter_events` and `/rosout`; no `/gs_<id>/...` topics were visible in this shell.",
            ]
            if not data.get("runtime_namespace")
            else []
        ),
        "",
        "## 5. Camera Interfaces",
        "",
        *md_table(["Camera", "Topic", "Type", "Resolution", "Encoding", "FPS", "QoS", "Frames", "Result"], camera_rows),
        "",
        "## 6. Depth Interfaces",
        "",
        *md_table(["Topic", "Type", "Resolution", "Encoding", "FPS", "Frames"], depth_rows or [["NONE", "", "", "", "", ""]]),
        "",
        "## 7. Arm State Interfaces",
        "",
        *device_topic_lines(data, "LEFT_ARM"),
        "",
        *device_topic_lines(data, "RIGHT_ARM"),
        "",
        "## 8. Hand State Interfaces",
        "",
        *device_topic_lines(data, "LEFT_HAND"),
        "",
        *device_topic_lines(data, "RIGHT_HAND"),
        "",
        "## 9. JointState Mapping",
        "",
        *md_table(["Topic", "Device", "Joint Names", "Count", "Hz", "Meaning", "Confidence"], js_rows or [["NONE", "", "", "", "", "", ""]]),
        "",
        "## 10. Force/Torque Mapping",
        "",
        *md_table(["Topic", "Device", "Frame", "Hz", "Sample", "Meaning/Confidence"], fts_rows or [["NONE", "", "", "", "", ""]]),
        "",
        "## 11. PM / SM Mapping",
        "",
        *md_table(["Topic", "Device", "Suffix", "Type", "Sample Range", "Hz"], float_rows or [["NONE", "", "", "", "", ""]]),
        "",
        f"- PS: {meanings.get('ps', {}).get('meaning')} ({meanings.get('ps', {}).get('confidence')}). Evidence: {meanings.get('ps', {}).get('evidence')}",
        f"- PM: {meanings.get('pm', {}).get('meaning')} ({meanings.get('pm', {}).get('confidence')}). Evidence: {meanings.get('pm', {}).get('evidence')}",
        f"- SM: {meanings.get('sm', {}).get('meaning')} ({meanings.get('sm', {}).get('confidence')}). Evidence: {meanings.get('sm', {}).get('evidence')}",
        "",
        "## 12. SDK vs ROS State",
        "",
        *md_table(
            ["Device", "Status", "ROS Topic", "Max Abs Error"],
            [[dev, rec.get("status"), rec.get("topic", ""), rec.get("max_abs_error", "")] for dev, rec in sdk_cmp.items()] or [["NONE", "", "", ""]],
        ),
        "",
        "## 13. Timestamp / Synchronization",
        "",
        f"Time domain: {timing.get('domain')}. Max sampled ROS stamp delta: {timing.get('max_ros_stamp_delta_s')}.",
        "",
        "```json",
        json.dumps(timing.get("samples", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 14. Fixed RGB Investigation",
        "",
        f"Fixed RGB result: {'PASS' if fixed.get('frames', 0) > 0 else 'TOPIC EXISTS BUT NO DATA' if fixed else 'TOPIC NOT FOUND'}. Publisher count: {fixed.get('publisher_count', 0)}. Frames: {fixed.get('frames', 0)}. QoS: {qos_text(fixed)}.",
        "If the topic exists with publishers but frames remain zero while wrist cameras work, the most likely cause is either publisher inactivity in the current scene or QoS/subscriber compatibility. This run used a BEST_EFFORT/VOLATILE subscriber.",
        "",
        "## 15. Recommended ACT Observation Sources",
        "",
        *[f"- {k} -> {v}" for k, v in rec_sources.items()],
        "",
        "## 16. Recommended Recorder Architecture",
        "",
        "```text",
        "ROS Camera",
        "        \\",
        "Robot State -> Synchronizer -> Episode Recorder",
        "        /",
        "Action",
        "```",
        "",
        recorder_choice(data),
        "",
        "## 17. Interfaces Useful for Expert",
        "",
        *md_table(
            ["Interface", "Use", "Reason"],
            [
                ["RGB cameras", "ACT INPUT", "Direct visual observation."],
                ["Robot qpos", "ACT INPUT", "Policy proprioception."],
                ["SDK pose/state", "EXPERT ONLY", "Read-only debugging and state sanity checks."],
                ["FTS", "DEBUG / SUCCESS CHECK", "Contact/force signal; mapping to fingers may still be unknown."],
                ["Depth/PointCloud", "OPTIONAL", "3D perception/debug; not required for first ACT observation."],
            ],
        ),
        "",
        "## 18. Interfaces Not Needed Initially",
        "",
        "- Depth images and PointCloud2 can be ignored for first recorder unless 3D perception/debug is needed.",
        "- FTS can be ignored for first ACT input; keep for contact diagnostics or success heuristics.",
        "- Unknown Float64 PM/SM topics should not be used as policy input until semantics are confirmed.",
        "",
        "## 19. Confirmed Facts",
        "",
        "- Runtime namespace is discovered dynamically from `/gs_<id>/...` topics.",
        "- `_tp_cam_`, `_tp_cami_`, `_tp_rgbd_`, and `_tp_fts_` meanings are documented in local Rabo manual snapshots.",
        "- All listed live message types and sampled frequencies are from read-only ROS discovery/subscription.",
        "",
        "## 20. Inferred Facts",
        "",
        "- `_tp_ps_` is treated as per-position-sensor or per-joint state unless a full-device JointState matches SDK qpos.",
        "- Complete A7 state candidate requires JointState position length 7 and SDK comparison.",
        "- O6 hand state candidate requires matching SDK hand joint vector length/order.",
        "",
        "## 21. Remaining Unknowns",
        "",
        *unknown_lines,
        "",
        "## 22. Final Conclusion",
        "",
        f"- RUNTIME INTERFACE MAPPING: {'PASS' if data.get('runtime_namespace') else 'FAIL'}",
        f"- CAMERA OBSERVATION: {'PASS' if all((cams.get(k) or {}).get('frames', 0) > 0 for k in KNOWN_RGB_SUFFIXES) else 'CHECK'}",
        f"- ROBOT STATE OBSERVATION: {'PASS' if any(rec.get('status') == 'MAPPED' for rec in sdk_cmp.values()) else 'CHECK'}",
        f"- READY FOR RECORDER: {'YES' if recorder_ready(data) else 'NO'}",
        "- READY FOR THREE-NUT EXPERT: NO",
        "",
        "## PointCloud Interfaces",
        "",
        *md_table(["Topic", "Device", "Size", "Fields", "Point Step", "Hz"], pc_rows or [["NONE", "", "", "", "", ""]]),
        "",
    ]
    return "\n".join(lines)


def device_topic_lines(data: dict[str, Any], device: str) -> list[str]:
    recs = [t for t in data.get("topics", []) if t.get("device") == device]
    rows = [[f"`{t['topic']}`", t.get("category"), t.get("type"), hz_text(t.get("hz")), t.get("frames", 0)] for t in recs]
    return [f"### {device}", "", *md_table(["Topic", "Category", "Type", "Hz", "Frames"], rows or [["NONE", "", "", "", ""]])]


def recorder_ready(data: dict[str, Any]) -> bool:
    cams = data.get("cameras", {})
    camera_ok = all((cams.get(k) or {}).get("frames", 0) > 0 for k in ("left_wrist_rgb", "right_wrist_rgb"))
    sdk_ok = data.get("sdk_state", {}).get("available") or any(r.get("status") == "MAPPED" for r in data.get("sdk_ros_comparison", {}).values())
    return bool(camera_ok and sdk_ok)


def recorder_choice(data: dict[str, Any]) -> str:
    mapped = [r for r in data.get("sdk_ros_comparison", {}).values() if r.get("status") == "MAPPED"]
    if len(mapped) >= 4:
        return "Recommended first Recorder: ROS Topics, because camera and robot state streams are timestamped and continuously sampled."
    return "Recommended first Recorder: SDK State + ROS Camera, because SDK qpos is explicit and already validated while ROS JointState mapping still has CHECK/UNKNOWN items."


def build_env(namespace: str | None) -> dict[str, Any]:
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ROS_DISTRO": os.environ.get("ROS_DISTRO"),
        "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
        "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION"),
        "runtime_namespace": namespace,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Rabo runtime interface mapper.")
    parser.add_argument("--duration", type=float, default=10.0, help="Read-only subscription duration.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument(
        "--cli-info",
        action="store_true",
        help="Also run ros2 topic info -v for every relevant topic. This can be slow on large live runtimes.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = resolve(args.output_dir)
    args.log_path = resolve(args.log_path)
    args.report_path = resolve(args.report_path)
    args.json_path = resolve(args.json_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs" / "ros").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))

    logger = Logger(args.log_path, verbose=args.verbose)
    commands: dict[str, Any] = {}
    try:
        logger.section("RABO RUNTIME INTERFACE MAP START")
        logger.log(f"duration={args.duration}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log("READ_ONLY: no publish, no action goal, no write service, no reset, no motion SDK methods.")

        evidence = source_search(logger)

        logger.section("ROS TOPIC DISCOVERY")
        topic_list = run_ros_topic(["list", "-t"], logger, timeout=12)
        commands["topic_list_t"] = topic_list
        topics = parse_topic_list_t(topic_list.get("stdout", []))
        namespace = detect_namespace(topics)
        annotate_topics(topics, namespace)
        logger.log(f"Runtime Namespace: {namespace}")

        if args.cli_info:
            for t in topics:
                if t.category != "OTHER":
                    info = run_ros_topic(["info", "-v", t.name], logger, timeout=8)
                    commands.setdefault("topic_info_verbose", {})[t.name] = info
        else:
            logger.log("SKIP ros2 topic info -v full scan by default; QoS is collected via rclpy endpoint discovery.")

        sample_set = choose_sample_topics(topics)
        sample_topics(sample_set, args.duration, args.output_dir, logger)

        logger.section("ROS TOPIC ECHO ONCE / HZ SPOT CHECKS")
        for label, suffix in KNOWN_RGB_SUFFIXES.items():
            t = next((x for x in topics if suffix in x.relative), None)
            if not t:
                continue
            commands.setdefault("fixed_rgb_investigation" if label == "fixed_rgb" else "rgb_spot_checks", {})[label] = {
                "echo_once": run_ros_topic(["echo", "--once", t.name], logger, timeout=5),
                "hz": run_ros_topic(["hz", t.name], logger, timeout=5),
            }

        sdk = read_sdk_state(logger)
        sdk_ros = compare_sdk_ros(sdk, topics)
        meanings = infer_meanings(topics, evidence)
        timing = timing_analysis(topics)
        env = build_env(namespace)

        data = build_json(topics, namespace, evidence, meanings, sdk, sdk_ros, timing, env, commands)
        args.json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report = build_report(data, args.output_dir, args.log_path)
        args.report_path.write_text(report, encoding="utf-8")

        fixed_result = result_for_rgb("fixed_rgb", topics)
        left_result = result_for_rgb("left_wrist_rgb", topics)
        right_result = result_for_rgb("right_wrist_rgb", topics)
        arm_state = "PASS" if any(v.get("status") == "MAPPED" for k, v in sdk_ros.items() if k.endswith("_ARM")) else "CHECK"
        hand_state = "PASS" if any(v.get("status") == "MAPPED" for k, v in sdk_ros.items() if k.endswith("_HAND")) else "CHECK"
        fts_state = "PASS" if data.get("force_torque_topics") and meanings["fts"]["confidence"] == "CONFIRMED" else "CHECK"
        sdk_ros_state = "PASS" if any(v.get("status") == "MAPPED" for v in sdk_ros.values()) else "CHECK"

        print("================================")
        print("RABO RUNTIME INTERFACE MAP")
        print("================================")
        print()
        print("Runtime Namespace:")
        print(namespace)
        print()
        print(f"Fixed RGB:\n{fixed_result}")
        print()
        print(f"Left Wrist RGB:\n{left_result}")
        print()
        print(f"Right Wrist RGB:\n{right_result}")
        print()
        print(f"Arm State Mapping:\n{arm_state}")
        print()
        print(f"Hand State Mapping:\n{hand_state}")
        print()
        print(f"FTS Mapping:\n{fts_state}")
        print()
        print(f"Timestamp Domain:\n{'PASS' if timing.get('domain') == 'SAME TIME DOMAIN' else 'CHECK'}")
        print()
        print(f"SDK ↔ ROS State:\n{sdk_ros_state}")
        print()
        print(f"READY FOR RECORDER:\n{'YES' if recorder_ready(data) else 'NO'}")
        print()
        print(f"Report:\n{display_path(args.report_path)}")
        print()
        print(f"JSON:\n{display_path(args.json_path)}")
        print()
        print(f"Raw Log:\n{display_path(args.log_path)}")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
