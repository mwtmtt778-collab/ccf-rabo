#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


KEYWORDS = {
    "camera": ["camera", "image", "rgb", "depth"],
    "robot_state": ["joint", "state", "qpos", "position", "arm", "hand", "o6", "a7", "p7"],
    "robot_control": ["command", "cmd", "control", "action", "trajectory", "servo", "move"],
    "objects": ["nut", "object", "pose", "thing"],
    "targets": ["target", "goal", "placement", "area", "bin", "tray"],
    "imu": ["imu"],
    "reset": ["reset", "reset_scene", "reset_env", "reset_episode", "restart"],
}

SOURCE_SEARCH = (
    "camera|image|rgb|joint|qpos|state|command|control|arm|hand|o6|a7|p7|nut|object|pose|target|reset|expert|demo|pick|place"
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], timeout: float = 8.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.splitlines(),
            "stderr": proc.stderr.splitlines(),
        }
    except FileNotFoundError as exc:
        return {"available": False, "error": f"not found: {exc}"}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def safe_import(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, repr(exc)


def keyword_categories(name: str) -> list[str]:
    low = name.lower()
    return [cat for cat, keys in KEYWORDS.items() if any(k in low for k in keys)]


def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [ensure_jsonable(v) for v in value]
    return str(value)


@dataclass
class TopicSample:
    count: int = 0
    times: list[float] = field(default_factory=list)
    first_summary: dict[str, Any] | None = None
    first_msg: Any | None = None


def summarize_msg(msg: Any, max_items: int = 12) -> dict[str, Any]:
    out: dict[str, Any] = {"class": msg.__class__.__module__ + "." + msg.__class__.__name__}
    for name in getattr(msg, "__slots__", [])[:max_items]:
        try:
            value = getattr(msg, name)
        except Exception:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = {"len": len(value), "sample": [ensure_jsonable(x) for x in list(value)[:5]]}
        elif hasattr(value, "__slots__"):
            out[name] = summarize_msg(value, max_items=5)
        else:
            out[name] = str(type(value))
    return out


def image_summary(msg: Any) -> dict[str, Any]:
    summary = summarize_msg(msg)
    for attr in ("width", "height", "encoding", "step"):
        if hasattr(msg, attr):
            summary[attr] = getattr(msg, attr)
    if hasattr(msg, "format"):
        summary["format"] = getattr(msg, "format")
    if hasattr(msg, "data"):
        try:
            summary["data_len"] = len(msg.data)
        except Exception:
            pass
    return summary


def msg_to_image_bytes(msg: Any) -> tuple[bytes | None, str | None]:
    # Avoid cv_bridge dependency. Save RGB/BGR mono sensor_msgs/Image as PPM/PGM bytes.
    if not all(hasattr(msg, x) for x in ("width", "height", "encoding", "data")):
        return None, "not a raw Image message"
    width, height, encoding = int(msg.width), int(msg.height), str(msg.encoding).lower()
    data = bytes(msg.data)
    if encoding in ("rgb8", "bgr8"):
        expected = width * height * 3
        if len(data) < expected:
            return None, f"short image buffer {len(data)} < {expected}"
        pixels = bytearray(data[:expected])
        if encoding == "bgr8":
            for i in range(0, expected, 3):
                pixels[i], pixels[i + 2] = pixels[i + 2], pixels[i]
        return b"P6\n%d %d\n255\n" % (width, height) + bytes(pixels), "ppm"
    if encoding in ("mono8", "8uc1"):
        expected = width * height
        if len(data) < expected:
            return None, f"short image buffer {len(data)} < {expected}"
        return b"P5\n%d %d\n255\n" % (width, height) + data[:expected], "pgm"
    return None, f"unsupported safe image encoding: {encoding}"


def fps_stats(times: list[float]) -> dict[str, Any]:
    if len(times) < 2:
        return {"frames": len(times), "avg_fps": None, "min_fps": None, "max_fps": None}
    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not intervals:
        return {"frames": len(times), "avg_fps": None, "min_fps": None, "max_fps": None}
    fps_values = [1.0 / dt for dt in intervals]
    duration = times[-1] - times[0]
    return {
        "frames": len(times),
        "duration_s": duration,
        "avg_fps": (len(times) - 1) / duration if duration > 0 else None,
        "min_fps": min(fps_values),
        "max_fps": max(fps_values),
    }


def endpoint_info_to_dict(info: Any) -> dict[str, Any]:
    qos = getattr(info, "qos_profile", None)
    out = {
        "node_name": getattr(info, "node_name", None),
        "node_namespace": getattr(info, "node_namespace", None),
        "topic_type": getattr(info, "topic_type", None),
        "endpoint_type": str(getattr(info, "endpoint_type", None)),
        "gid": str(getattr(info, "endpoint_gid", None)),
    }
    if qos is not None:
        out["qos"] = {
            "reliability": str(getattr(qos, "reliability", None)),
            "durability": str(getattr(qos, "durability", None)),
            "history": str(getattr(qos, "history", None)),
            "depth": getattr(qos, "depth", None),
        }
    return out


class RosRuntimeProbe:
    def __init__(self, duration: float, output_dir: Path, save_images: bool, verbose: bool):
        self.duration = duration
        self.output_dir = output_dir
        self.save_images = save_images
        self.verbose = verbose
        self.report: dict[str, Any] = {
            "ros": {"available": False, "errors": []},
            "topics": [],
            "services": [],
            "actions": [],
            "cameras": [],
            "robot_state": {"topics": [], "samples": []},
            "robot_control": {"topics": [], "services": [], "actions": []},
            "objects": [],
            "targets": [],
            "reset": {"interfaces": []},
            "frequencies": {},
        }

    def log_error(self, where: str, exc: Exception | str) -> None:
        self.report["ros"]["errors"].append({"where": where, "error": repr(exc) if not isinstance(exc, str) else exc})

    def run(self) -> dict[str, Any]:
        rclpy, err = safe_import("rclpy")
        if rclpy is None:
            self.report["ros"]["available"] = False
            self.report["ros"]["errors"].append({"where": "import rclpy", "error": err})
            return self.report

        self.report["ros"]["available"] = True
        try:
            rclpy.init(args=None)
            node = rclpy.create_node("rabo_runtime_readonly_probe")
        except Exception as exc:
            self.log_error("rclpy init/create_node", exc)
            return self.report

        try:
            self._discover(node)
            self._sample_topics(node)
        finally:
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception as exc:
                self.log_error("rclpy shutdown", exc)
        return self.report

    def _discover(self, node: Any) -> None:
        try:
            self.report["ros"]["nodes"] = sorted(node.get_node_names())
        except Exception as exc:
            self.log_error("get_node_names", exc)
            self.report["ros"]["nodes"] = []

        try:
            topic_pairs = sorted(node.get_topic_names_and_types(), key=lambda x: x[0])
        except Exception as exc:
            self.log_error("get_topic_names_and_types", exc)
            topic_pairs = []

        for name, types in topic_pairs:
            topic_type = types[0] if types else None
            item = {
                "name": name,
                "types": list(types),
                "categories": keyword_categories(name),
                "publishers": [],
                "subscribers": [],
                "publisher_count": None,
                "subscriber_count": None,
            }
            try:
                pubs = node.get_publishers_info_by_topic(name)
                subs = node.get_subscriptions_info_by_topic(name)
                item["publishers"] = [endpoint_info_to_dict(x) for x in pubs]
                item["subscribers"] = [endpoint_info_to_dict(x) for x in subs]
                item["publisher_count"] = len(pubs)
                item["subscriber_count"] = len(subs)
            except Exception as exc:
                item["endpoint_error"] = repr(exc)
            self.report["topics"].append(item)

            if "robot_control" in item["categories"]:
                self.report["robot_control"]["topics"].append(item)
            if "reset" in item["categories"]:
                self.report["reset"]["interfaces"].append({"kind": "topic", **item})
            if "objects" in item["categories"]:
                self.report["objects"].append({"kind": "topic", **item})
            if "targets" in item["categories"]:
                self.report["targets"].append({"kind": "topic", **item})

        try:
            service_pairs = sorted(node.get_service_names_and_types(), key=lambda x: x[0])
        except Exception as exc:
            self.log_error("get_service_names_and_types", exc)
            service_pairs = []
        for name, types in service_pairs:
            item = {"name": name, "types": list(types), "categories": keyword_categories(name)}
            self.report["services"].append(item)
            if "robot_control" in item["categories"]:
                self.report["robot_control"]["services"].append(item)
            if "reset" in item["categories"]:
                self.report["reset"]["interfaces"].append({"kind": "service", **item})

        # rclpy action discovery is not consistently exposed; CLI fallback is recorded separately.
        self.report["actions"] = []

    def _topic_class(self, type_name: str) -> Any | None:
        try:
            pkg, _, msg = type_name.partition("/msg/")
            if not pkg or not msg:
                return None
            mod = importlib.import_module(f"{pkg}.msg")
            return getattr(mod, msg)
        except Exception:
            return None

    def _sample_topics(self, node: Any) -> None:
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy
        except Exception as exc:
            self.log_error("import qos", exc)
            return

        samples: dict[str, TopicSample] = defaultdict(TopicSample)
        subscriptions = []
        camera_topics: list[dict[str, Any]] = []
        state_topics: list[dict[str, Any]] = []
        pose_topics: list[dict[str, Any]] = []

        for item in self.report["topics"]:
            name = item["name"]
            types = item.get("types") or []
            msg_type = types[0] if types else ""
            cats = set(item.get("categories") or [])
            is_camera = "camera" in cats and msg_type in (
                "sensor_msgs/msg/Image",
                "sensor_msgs/msg/CompressedImage",
            )
            is_state = msg_type == "sensor_msgs/msg/JointState" or ("robot_state" in cats and "state" in cats)
            is_pose = msg_type in (
                "geometry_msgs/msg/PoseStamped",
                "geometry_msgs/msg/PoseArray",
                "nav_msgs/msg/Odometry",
                "tf2_msgs/msg/TFMessage",
            ) and (cats & {"objects", "targets", "robot_state"})
            if is_camera:
                camera_topics.append(item)
            if is_state:
                state_topics.append(item)
            if is_pose:
                pose_topics.append(item)

        def make_cb(topic: str):
            def cb(msg: Any) -> None:
                s = samples[topic]
                s.count += 1
                s.times.append(time.monotonic())
                if s.first_summary is None:
                    s.first_msg = msg
                    s.first_summary = image_summary(msg) if "Image" in msg.__class__.__name__ else summarize_msg(msg)
            return cb

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        for item in camera_topics + state_topics + pose_topics:
            topic = item["name"]
            msg_cls = self._topic_class((item.get("types") or [""])[0])
            if msg_cls is None:
                continue
            try:
                subscriptions.append(node.create_subscription(msg_cls, topic, make_cb(topic), qos))
            except Exception as exc:
                item["subscription_error"] = repr(exc)

        end = time.monotonic() + self.duration
        while time.monotonic() < end:
            try:
                import rclpy

                rclpy.spin_once(node, timeout_sec=0.1)
            except Exception as exc:
                self.log_error("spin_once", exc)
                break

        for i, item in enumerate(camera_topics):
            topic = item["name"]
            s = samples[topic]
            entry = {
                "topic": topic,
                "message_type": (item.get("types") or [None])[0],
                "publisher_count": item.get("publisher_count"),
                "subscriber_count": item.get("subscriber_count"),
                "first_frame": s.first_summary,
                "fps": fps_stats(s.times),
                "status": "AVAILABLE" if s.count else "NO_FRAMES",
            }
            if self.save_images and s.first_msg is not None:
                data, ext_or_error = msg_to_image_bytes(s.first_msg)
                if data is not None:
                    path = self.output_dir / f"camera_{len(self.report['cameras'])}.{ext_or_error}"
                    path.write_bytes(data)
                    entry["saved_first_frame"] = str(path)
                else:
                    entry["save_image_error"] = ext_or_error
            self.report["cameras"].append(entry)

        for item in state_topics:
            topic = item["name"]
            s = samples[topic]
            sample = {
                "topic": topic,
                "message_type": (item.get("types") or [None])[0],
                "publisher_count": item.get("publisher_count"),
                "subscriber_count": item.get("subscriber_count"),
                "first_sample": s.first_summary,
                "fps": fps_stats(s.times),
                "status": "AVAILABLE" if s.count else "NO_FRAMES",
            }
            first = s.first_msg
            if first is not None and hasattr(first, "name"):
                names = list(getattr(first, "name", []))
                sample["joint_names"] = names
                sample["joint_count"] = len(names)
                for field in ("position", "velocity", "effort"):
                    value = list(getattr(first, field, []))
                    sample[f"{field}_shape"] = [len(value)]
                    sample[f"{field}_sample"] = value[:12]
            self.report["robot_state"]["samples"].append(sample)
        self.report["robot_state"]["topics"] = state_topics
        self.report["frequencies"]["robot_state_topics"] = [
            {"topic": x["topic"], **x["fps"]} for x in self.report["robot_state"]["samples"]
        ]
        self.report["frequencies"]["camera_topics"] = [
            {"topic": x["topic"], **x["fps"]} for x in self.report["cameras"]
        ]


def inspect_static_source(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "matches": [],
        "known_ids": {},
        "control_examples": [],
        "reset_examples": [],
        "sdk_imports": {},
    }
    for path in sorted(root.rglob("*")):
        if path.is_dir() or ".git" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(SOURCE_SEARCH, line, flags=re.IGNORECASE):
                rel = str(path.relative_to(root))
                record = {"file": rel, "line": lineno, "text": line.strip()}
                out["matches"].append(record)
                low = line.lower()
                if any(k in low for k in ("move_joints", "move_to", "clench", "grasp_force", "setentitypose", ".set(")):
                    out["control_examples"].append(record)
                if "reset" in low or "restart" in low:
                    out["reset_examples"].append(record)

    arm_demo = root / "agents" / "arm_hand_demo" / "__init__.py"
    if arm_demo.exists():
        text = arm_demo.read_text(encoding="utf-8", errors="ignore")
        for name in (
            "WORLD_ID",
            "NUT_A_ID",
            "NUT_B_ID",
            "NUT_C_ID",
            "RIGHT_ARM_ID",
            "RIGHT_HAND_ID",
            "LEFT_ARM_ID",
            "LEFT_HAND_ID",
        ):
            m = re.search(rf"^{name}\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.MULTILINE)
            if m:
                out["known_ids"][name] = m.group(1)

    for mod_name in ("rabo_robocap", "rabo_dev_kit"):
        mod, err = safe_import(mod_name)
        if mod is None:
            out["sdk_imports"][mod_name] = {"available": False, "error": err}
            continue
        classes = {}
        for name in ("LinkerArmA7", "LinkerHandO6Right", "LinkerHandO6Left", "SetEntityPose", "RemoteControl"):
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            methods = [m for m, fn in inspect.getmembers(obj) if not m.startswith("_") and callable(fn)]
            classes[name] = {"methods": methods[:80]}
        out["sdk_imports"][mod_name] = {"available": True, "file": getattr(mod, "__file__", None), "classes": classes}
    return out


def collect_environment(root: Path) -> dict[str, Any]:
    env = {
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "pwd": str(root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    env["ros2_cli"] = {
        "which": run_cmd(["which", "ros2"], timeout=2),
        "node_list": run_cmd(["ros2", "node", "list", "--no-daemon"], timeout=8),
        "topic_list": run_cmd(["ros2", "topic", "list", "-t", "--no-daemon"], timeout=8),
        "service_list": run_cmd(["ros2", "service", "list", "-t", "--no-daemon"], timeout=8),
        "action_list": run_cmd(["ros2", "action", "list", "-t", "--no-daemon"], timeout=8),
    }
    return env


def interface_summary(data: dict[str, Any]) -> dict[str, Any]:
    control_topics = data["ros_probe"]["robot_control"].get("topics", [])
    control_services = data["ros_probe"]["robot_control"].get("services", [])
    control_actions = data["ros_probe"].get("actions", [])
    can_control_arm = any("arm" in x.get("name", "").lower() for x in control_topics + control_services)
    can_control_hand = any(("hand" in x.get("name", "").lower() or "o6" in x.get("name", "").lower()) for x in control_topics + control_services)
    nut_ids = data["static_source"]["known_ids"]
    can_read_nut_pose = any(
        item.get("status") == "AVAILABLE" for item in data["ros_probe"].get("objects", []) if "pose" in item.get("name", "").lower()
    )
    # Static IDs are object names, not proof that pose can be read.
    return {
        "camera_count": len(data["ros_probe"].get("cameras", [])),
        "robot_state_topic_count": len(data["ros_probe"]["robot_state"].get("topics", [])),
        "robot_control_interface_count": len(control_topics) + len(control_services) + len(control_actions),
        "known_nut_ids": {k: v for k, v in nut_ids.items() if k.startswith("NUT_")},
        "can_read_nut_pose": "YES" if can_read_nut_pose else "NO",
        "can_control_arm": "INTERFACE FOUND" if can_control_arm else "NOT FOUND",
        "can_control_hand": "INTERFACE FOUND" if can_control_hand else "NOT FOUND",
    }


def write_markdown(data: dict[str, Any], path: Path) -> None:
    summary = data["summary"]
    ros = data["ros_probe"]
    static = data["static_source"]
    lines = [
        "# Rabo Runtime Report",
        "",
        f"Generated: `{data['environment']['generated_at']}`",
        "",
        "## Environment",
        "",
        f"- Hostname: `{data['environment']['hostname']}`",
        f"- PWD: `{data['environment']['pwd']}`",
        f"- Python: `{data['environment']['python_version'].splitlines()[0]}`",
        f"- ROS_DISTRO: `{data['environment'].get('ros_distro')}`",
        f"- ROS_DOMAIN_ID: `{data['environment'].get('ros_domain_id')}`",
        f"- RMW_IMPLEMENTATION: `{data['environment'].get('rmw_implementation')}`",
        "",
        "## Summary",
        "",
        f"- Cameras discovered: `{summary['camera_count']}`",
        f"- Robot state topics: `{summary['robot_state_topic_count']}`",
        f"- Robot control interfaces: `{summary['robot_control_interface_count']}`",
        f"- Can read nut ground-truth pose: `{summary['can_read_nut_pose']}`",
        f"- Can control arm: `{summary['can_control_arm']}`",
        f"- Can control hand: `{summary['can_control_hand']}`",
        "",
        "## Cameras",
        "",
    ]
    if ros.get("cameras"):
        for cam in ros["cameras"]:
            first = cam.get("first_frame") or {}
            fps = cam.get("fps") or {}
            lines.append(
                f"- `{cam['topic']}` type=`{cam['message_type']}` status=`{cam['status']}` "
                f"size=`{first.get('width')}x{first.get('height')}` encoding=`{first.get('encoding') or first.get('format')}` "
                f"avg_fps=`{fps.get('avg_fps')}` min_fps=`{fps.get('min_fps')}` max_fps=`{fps.get('max_fps')}`"
            )
    else:
        lines.append("- No RGB camera frames discovered.")

    lines += ["", "## Robot State", ""]
    if ros["robot_state"].get("samples"):
        for st in ros["robot_state"]["samples"]:
            lines.append(
                f"- `{st['topic']}` type=`{st['message_type']}` status=`{st['status']}` "
                f"joint_count=`{st.get('joint_count')}` position_shape=`{st.get('position_shape')}` "
                f"velocity_shape=`{st.get('velocity_shape')}` effort_shape=`{st.get('effort_shape')}`"
            )
            if st.get("joint_names"):
                lines.append(f"  - joint_names: `{st['joint_names']}`")
    else:
        lines.append("- No sampled robot state topic.")

    lines += ["", "## Robot Control Interfaces (Read-Only Discovery)", ""]
    for kind in ("topics", "services", "actions"):
        items = ros["robot_control"].get(kind, []) if kind != "actions" else ros.get("actions", [])
        if items:
            lines.append(f"### {kind.title()}")
            for item in items:
                lines.append(f"- `{item.get('name')}` type=`{item.get('types')}`")
    if not any(ros["robot_control"].get(k) for k in ("topics", "services")) and not ros.get("actions"):
        lines.append("- No command/control interfaces found by ROS discovery.")
    lines.append("")
    lines.append("No command/control/reset interface was called by this probe.")

    lines += ["", "## ROS Discovery Raw Result", ""]
    lines.append(f"- ROS available through rclpy: `{ros.get('ros', {}).get('available')}`")
    lines.append(f"- ROS nodes: `{ros.get('ros', {}).get('nodes', [])}`")
    lines.append(f"- Topic count: `{len(ros.get('topics', []))}`")
    for topic in ros.get("topics", []):
        lines.append(
            f"  - `{topic.get('name')}` type=`{topic.get('types')}` "
            f"publishers=`{topic.get('publisher_count')}` subscribers=`{topic.get('subscriber_count')}`"
        )
    lines.append(f"- Service count: `{len(ros.get('services', []))}`")
    for service in ros.get("services", []):
        lines.append(f"  - `{service.get('name')}` type=`{service.get('types')}`")
    if ros.get("ros", {}).get("errors"):
        lines.append(f"- ROS errors: `{ros.get('ros', {}).get('errors')}`")

    lines += ["", "## Known Object IDs From Source", ""]
    if summary["known_nut_ids"]:
        for k, v in summary["known_nut_ids"].items():
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- No nut IDs found in source.")

    lines += ["", "## Object / Target Runtime Discovery", ""]
    lines.append(f"- Object candidates: `{len(ros.get('objects', []))}`")
    lines.append(f"- Target candidates: `{len(ros.get('targets', []))}`")
    lines.append("- Runtime pose availability is recorded in `outputs/runtime_probe/runtime_probe.json`.")

    lines += ["", "## Reset / Episode Interfaces", ""]
    if ros["reset"].get("interfaces"):
        for item in ros["reset"]["interfaces"]:
            lines.append(f"- `{item.get('kind')}` `{item.get('name')}` type=`{item.get('types')}`")
    else:
        lines.append("- No reset interface found by ROS discovery.")
    if static.get("reset_examples"):
        lines.append("- Static reset-related source matches exist; inspect JSON for exact file/line.")

    lines += ["", "## Static Source Findings", ""]
    lines.append(f"- Total keyword matches: `{len(static.get('matches', []))}`")
    lines.append(f"- Control/example matches: `{len(static.get('control_examples', []))}`")
    lines.append(f"- SDK imports: `{json.dumps(static.get('sdk_imports', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append("Machine-readable report: `outputs/runtime_probe/runtime_probe.json`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(data: dict[str, Any]) -> None:
    ros = data["ros_probe"]
    summary = data["summary"]
    print("==============================")
    print("RABO RUNTIME PROBE SUMMARY")
    print("==============================")
    print("Cameras:")
    for cam in ros.get("cameras", []):
        fps = cam.get("fps", {})
        first = cam.get("first_frame") or {}
        print(f"  {cam['topic']} {first.get('width')}x{first.get('height')} {first.get('encoding') or first.get('format')} fps={fps.get('avg_fps')} status={cam.get('status')}")
    if not ros.get("cameras"):
        print("  NONE")
    print("Robot state:")
    for st in ros["robot_state"].get("samples", []):
        print(f"  {st['topic']} joints={st.get('joint_count')} pos_shape={st.get('position_shape')} fps={st.get('fps', {}).get('avg_fps')} status={st.get('status')}")
    if not ros["robot_state"].get("samples"):
        print("  NONE")
    print("Robot control:")
    for item in ros["robot_control"].get("topics", []):
        print(f"  topic {item.get('name')} {item.get('types')}")
    for item in ros["robot_control"].get("services", []):
        print(f"  service {item.get('name')} {item.get('types')}")
    if not ros["robot_control"].get("topics") and not ros["robot_control"].get("services"):
        print("  NONE")
    print("Nut objects:")
    for k, v in summary["known_nut_ids"].items():
        print(f"  {k}: {v}")
    if not summary["known_nut_ids"]:
        print("  NONE")
    print("Target regions:")
    for item in ros.get("targets", [])[:20]:
        print(f"  {item.get('kind', 'topic')} {item.get('name')} {item.get('types')}")
    if not ros.get("targets"):
        print("  NONE")
    print("Reset:")
    for item in ros["reset"].get("interfaces", []):
        print(f"  {item.get('kind')} {item.get('name')} {item.get('types')}")
    if not ros["reset"].get("interfaces"):
        print("  NONE")
    print("Camera FPS:")
    print(f"  {data['ros_probe']['frequencies'].get('camera_topics', [])}")
    print("State FPS:")
    print(f"  {data['ros_probe']['frequencies'].get('robot_state_topics', [])}")
    print(f"Can read nut pose: {summary['can_read_nut_pose']}")
    print(f"Can control arm: {summary['can_control_arm']}")
    print(f"Can control hand: {summary['can_control_hand']}")
    print("==============================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Rabo runtime environment probe.")
    parser.add_argument("--duration", type=float, default=8.0, help="Read-only sampling duration in seconds.")
    parser.add_argument("--save-images", action="store_true", help="Save first safe RGB/mono frame per camera.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runtime_probe"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/runtime_probe/latest_report.md"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report_path = args.report_path if args.report_path.is_absolute() else root / args.report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "environment": collect_environment(root),
        "static_source": inspect_static_source(root),
        "ros_probe": {},
    }
    try:
        data["ros_probe"] = RosRuntimeProbe(args.duration, output_dir, args.save_images, args.verbose).run()
    except Exception as exc:
        data["ros_probe"] = {"ros": {"available": False, "errors": [{"where": "RosRuntimeProbe.run", "error": repr(exc)}]}}
    data["summary"] = interface_summary(data)

    json_path = output_dir / "runtime_probe.json"
    json_path.write_text(json.dumps(ensure_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(data, report_path)
    print_summary(data)
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
