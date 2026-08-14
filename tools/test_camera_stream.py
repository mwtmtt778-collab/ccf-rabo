#!/usr/bin/env python3
"""Read-only RGB camera stream test for the Rabo sim runtime."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camera_test"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "camera_test.log"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "RABO_CAMERA_TEST_REPORT.md"

KNOWN_CAMERAS = {
    "fixed_rgb": "r6ef2dc_tp_cam_303d2b1ce0",
    "left_wrist_rgb": "rbd03eb_tp_cam_069a6739f3",
    "right_wrist_rgb": "r412d23_tp_cam_3c67aef2bc",
}

SCAN_KEYWORDS = ("_tp_cam_", "camera", "image")
SUPPORTED_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}
RAW_IMAGE_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8", "mono8"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class TopicRecord:
    name: str
    type_names: list[str]
    matched_by: list[str] = field(default_factory=list)
    info_verbose: dict[str, Any] = field(default_factory=dict)
    frames: int = 0
    first_time_monotonic: float | None = None
    last_time_monotonic: float | None = None
    first_summary: dict[str, Any] | None = None
    sample_path: str | None = None
    sample_error: str | None = None
    subscribe_error: str | None = None

    @property
    def primary_type(self) -> str | None:
        for type_name in self.type_names:
            if type_name in SUPPORTED_TYPES:
                return type_name
        return self.type_names[0] if self.type_names else None


class TeeLogger:
    def __init__(self, path: Path, verbose: bool):
        self.path = path
        self.verbose = verbose
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def log(self, message: str = "") -> None:
        line = message.rstrip()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"[{timestamp}] {line}\n")
        self._file.flush()
        if self.verbose:
            print(line)


def normalize_topic(name: str) -> str:
    return name[1:] if name.startswith("/") else name


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_output(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def run_cmd(cmd: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": clean_output(proc.stdout).splitlines(),
            "stderr": clean_output(proc.stderr).splitlines(),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "ok": False, "error": f"not found: {exc}"}
    except Exception as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def run_ros2_topic_cmd(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
    cmd = ["ros2", "topic", *args, "--no-daemon"]
    result = run_cmd(cmd, timeout=timeout)
    text = "\n".join(result.get("stderr", []) + result.get("stdout", []))
    if "unrecognized arguments: --no-daemon" not in text:
        return result
    return run_cmd(["ros2", "topic", *args], timeout=timeout)


def parse_topic_list_t(lines: list[str]) -> list[TopicRecord]:
    topics: list[TopicRecord] = []
    pattern = re.compile(r"^(?P<name>\S+)\s+\[(?P<types>[^\]]+)\]\s*$")
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        type_names = [item.strip() for item in match.group("types").split(",") if item.strip()]
        topics.append(TopicRecord(name=match.group("name"), type_names=type_names))
    return topics


def topic_has_supported_type(topic: TopicRecord) -> bool:
    return any(type_name in SUPPORTED_TYPES for type_name in topic.type_names)


def select_camera_topics(topics: list[TopicRecord]) -> list[TopicRecord]:
    by_norm = {normalize_topic(topic.name): topic for topic in topics}
    selected: dict[str, TopicRecord] = {}

    for label, known_name in KNOWN_CAMERAS.items():
        topic = by_norm.get(normalize_topic(known_name))
        if topic is None:
            continue
        topic.matched_by.append(f"known:{label}")
        selected[normalize_topic(topic.name)] = topic

    for topic in topics:
        normalized = normalize_topic(topic.name)
        lowered = normalized.lower()
        if not any(keyword in lowered for keyword in SCAN_KEYWORDS):
            continue
        if not topic_has_supported_type(topic):
            topic.matched_by.append("scan:unsupported_type")
        else:
            topic.matched_by.append("scan")
        selected.setdefault(normalized, topic)

    return list(selected.values())


def load_ros_message_class(type_name: str) -> Any:
    try:
        from rosidl_runtime_py.utilities import get_message

        return get_message(type_name)
    except Exception:
        package, _, message = type_name.partition("/msg/")
        module = __import__(f"{package}.msg", fromlist=[message])
        return getattr(module, message)


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


def summarize_msg(msg: Any) -> dict[str, Any]:
    summary = {"class": msg.__class__.__module__ + "." + msg.__class__.__name__}
    for attr in ("height", "width", "encoding", "is_bigendian", "step", "format"):
        if hasattr(msg, attr):
            summary[attr] = getattr(msg, attr)
    if hasattr(msg, "data"):
        try:
            summary["data_len"] = len(msg.data)
        except Exception as exc:
            summary["data_len_error"] = repr(exc)
    if hasattr(msg, "header"):
        header = getattr(msg, "header")
        summary["header"] = {
            "frame_id": getattr(header, "frame_id", None),
            "stamp": repr(getattr(header, "stamp", None)),
        }
    return jsonable(summary)


def extension_from_compressed_format(fmt: str) -> str:
    lowered = fmt.lower()
    if "png" in lowered:
        return "png"
    if "jpeg" in lowered or "jpg" in lowered:
        return "jpg"
    return "bin"


def raw_image_to_file_bytes(msg: Any) -> tuple[bytes | None, str | None, str | None]:
    required = ("height", "width", "encoding", "step", "data")
    if not all(hasattr(msg, attr) for attr in required):
        return None, None, "message is not sensor_msgs/msg/Image"

    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()
    data = bytes(msg.data)

    if encoding not in RAW_IMAGE_ENCODINGS:
        return None, None, f"unsupported raw encoding: {encoding}"
    if width <= 0 or height <= 0 or step <= 0:
        return None, None, f"invalid image dimensions width={width} height={height} step={step}"

    if encoding == "mono8":
        min_step = width
        if step < min_step or len(data) < step * height:
            return None, None, f"short mono8 buffer len={len(data)} step={step} height={height}"
        rows = [data[row * step : row * step + width] for row in range(height)]
        return b"P5\n%d %d\n255\n" % (width, height) + b"".join(rows), "pgm", None

    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    min_step = width * channels
    if step < min_step or len(data) < step * height:
        return None, None, f"short {encoding} buffer len={len(data)} step={step} height={height}"

    pixels = bytearray()
    for row in range(height):
        row_data = data[row * step : row * step + min_step]
        for col in range(0, len(row_data), channels):
            pixel = row_data[col : col + channels]
            if encoding in {"bgr8", "bgra8"}:
                pixels.extend((pixel[2], pixel[1], pixel[0]))
            else:
                pixels.extend((pixel[0], pixel[1], pixel[2]))
    return b"P6\n%d %d\n255\n" % (width, height) + bytes(pixels), "ppm", None


def save_sample_image(topic: TopicRecord, msg: Any, output_dir: Path) -> None:
    safe_name = normalize_topic(topic.name).replace("/", "__")

    if topic.primary_type == "sensor_msgs/msg/CompressedImage":
        data = bytes(getattr(msg, "data", b""))
        fmt = str(getattr(msg, "format", ""))
        ext = extension_from_compressed_format(fmt)
        if not data:
            topic.sample_error = "compressed image has empty data"
            return
        path = output_dir / f"{safe_name}.{ext}"
        path.write_bytes(data)
        topic.sample_path = display_path(path)
        return

    payload, ext, error = raw_image_to_file_bytes(msg)
    if error:
        topic.sample_error = error
        return
    if payload is None or ext is None:
        topic.sample_error = "image conversion produced no bytes"
        return
    path = output_dir / f"{safe_name}.{ext}"
    path.write_bytes(payload)
    topic.sample_path = display_path(path)


def inspect_topic_info(topic: TopicRecord, logger: TeeLogger) -> None:
    result = run_ros2_topic_cmd(["info", "-v", topic.name], timeout=10.0)
    topic.info_verbose = result
    logger.log("$ " + " ".join(result.get("cmd", ["ros2", "topic", "info", "-v", topic.name])))
    for line in result.get("stdout", []):
        logger.log(line)
    for line in result.get("stderr", []):
        logger.log("stderr: " + line)
    if "error" in result:
        logger.log("error: " + result["error"])


def subscribe_and_sample(
    topics: list[TopicRecord],
    duration: float,
    output_dir: Path,
    logger: TeeLogger,
) -> None:
    if not topics:
        logger.log("NO_CAMERA_CANDIDATES: skip rclpy subscription")
        return

    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except Exception as exc:
        for topic in topics:
            topic.subscribe_error = f"import rclpy failed: {repr(exc)}"
        logger.log(f"RCLPY_IMPORT_ERROR: {repr(exc)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = PROJECT_ROOT / "logs" / "ros"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))
    rclpy.init(args=None)
    node = rclpy.create_node("rabo_camera_readonly_test")
    subscriptions: list[Any] = []

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def make_callback(topic: TopicRecord) -> Callable[[Any], None]:
        def callback(msg: Any) -> None:
            now = time.monotonic()
            topic.frames += 1
            topic.last_time_monotonic = now
            if topic.first_time_monotonic is None:
                topic.first_time_monotonic = now
                topic.first_summary = summarize_msg(msg)
                save_sample_image(topic, msg, output_dir)
                logger.log(f"FIRST_FRAME {topic.name}: {json.dumps(topic.first_summary, ensure_ascii=False)}")
                if topic.sample_path:
                    logger.log(f"SAVED_SAMPLE {topic.name}: {topic.sample_path}")
                if topic.sample_error:
                    logger.log(f"SAMPLE_SAVE_ERROR {topic.name}: {topic.sample_error}")
        return callback

    try:
        for topic in topics:
            if not topic_has_supported_type(topic):
                topic.subscribe_error = f"unsupported topic types: {topic.type_names}"
                logger.log(f"SKIP_UNSUPPORTED {topic.name}: {topic.type_names}")
                continue
            try:
                msg_class = load_ros_message_class(topic.primary_type or "")
                subscription = node.create_subscription(msg_class, topic.name, make_callback(topic), qos)
                subscriptions.append(subscription)
                logger.log(f"SUBSCRIBED {topic.name} [{topic.primary_type}]")
            except Exception as exc:
                topic.subscribe_error = repr(exc)
                logger.log(f"SUBSCRIBE_ERROR {topic.name}: {repr(exc)}")

        start = time.monotonic()
        while time.monotonic() - start < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        for subscription in subscriptions:
            with contextlib.suppress(Exception):
                node.destroy_subscription(subscription)
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


def topic_rate(topic: TopicRecord) -> float | None:
    if topic.frames < 2 or topic.first_time_monotonic is None or topic.last_time_monotonic is None:
        return None
    elapsed = topic.last_time_monotonic - topic.first_time_monotonic
    if elapsed <= 0:
        return None
    return (topic.frames - 1) / elapsed


def build_report(
    topics: list[TopicRecord],
    all_topics: list[TopicRecord],
    topic_list_result: dict[str, Any],
    duration: float,
    output_dir: Path,
    log_path: Path,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    known_rows = []
    by_norm = {normalize_topic(topic.name): topic for topic in all_topics}
    for label, known_name in KNOWN_CAMERAS.items():
        topic = by_norm.get(normalize_topic(known_name))
        status = "FOUND" if topic else "MISSING"
        types = ", ".join(topic.type_names) if topic else ""
        known_rows.append(f"| {label} | `{known_name}` | {status} | {types} |")

    sampled_rows = []
    for topic in topics:
        rate = topic_rate(topic)
        rate_text = f"{rate:.2f}" if rate is not None else ""
        sample = topic.sample_path or ""
        error = topic.subscribe_error or topic.sample_error or ""
        sampled_rows.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                topic.name,
                ", ".join(topic.type_names),
                topic.frames,
                rate_text,
                sample,
                error.replace("\n", " "),
            )
        )

    if not sampled_rows:
        sampled_rows.append("|  |  | 0 |  |  | No camera candidate topics found. |")

    sampled_by_norm = {normalize_topic(topic.name): topic for topic in topics}
    known_topics_ok = [
        (sampled_by_norm.get(normalize_topic(known_name)) is not None)
        and (sampled_by_norm[normalize_topic(known_name)].frames > 0)
        for known_name in KNOWN_CAMERAS.values()
    ]
    result = "PASS" if all(known_topics_ok) else "CHECK"
    if any(topic.subscribe_error for topic in topics):
        result = "CHECK"

    lines = [
        "# RABO Camera Test Report",
        "",
        f"- Generated: {now}",
        f"- Duration: {duration:.1f}s",
        f"- Result: {result}",
        f"- Output directory: `{display_path(output_dir)}`",
        f"- Raw log: `{display_path(log_path)}`",
        "",
        "## Read-only constraints",
        "",
        "- No arm motion APIs were called.",
        "- No hand actuation APIs were called.",
        "- No object/entity pose or simulation reset APIs were called.",
        "- No camera parameters were modified.",
        "",
        "## Known RGB topics",
        "",
        "| Camera | Expected topic | Discovery | ROS type |",
        "| --- | --- | --- | --- |",
        *known_rows,
        "",
        "## Sampled topics",
        "",
        "| Topic | Type | Frames | Approx FPS | Sample | Error |",
        "| --- | --- | ---: | ---: | --- | --- |",
        *sampled_rows,
        "",
        "## Discovery command",
        "",
        "```text",
        "$ " + " ".join(topic_list_result.get("cmd", ["ros2", "topic", "list", "-t"])),
        *topic_list_result.get("stdout", []),
        *("stderr: " + line for line in topic_list_result.get("stderr", [])),
        *(["error: " + topic_list_result["error"]] if "error" in topic_list_result else []),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_json_summary(path: Path, topics: list[TopicRecord], topic_list_result: dict[str, Any]) -> None:
    payload = {
        "known_cameras": KNOWN_CAMERAS,
        "topic_list": topic_list_result,
        "camera_topics": [
            {
                "name": topic.name,
                "type_names": topic.type_names,
                "matched_by": topic.matched_by,
                "frames": topic.frames,
                "approx_fps": topic_rate(topic),
                "first_summary": topic.first_summary,
                "sample_path": topic.sample_path,
                "sample_error": topic.sample_error,
                "subscribe_error": topic.subscribe_error,
                "info_verbose": topic.info_verbose,
            }
            for topic in topics
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Rabo RGB camera stream test.")
    parser.add_argument("--duration", type=float, default=10.0, help="Sampling duration in seconds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved samples.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="Raw log path.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Markdown report path.")
    parser.add_argument("--verbose", action="store_true", help="Also print log lines to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = resolve_project_path(args.output_dir)
    args.log_path = resolve_project_path(args.log_path)
    args.report_path = resolve_project_path(args.report_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    ros_log_dir = PROJECT_ROOT / "logs" / "ros"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))

    logger = TeeLogger(args.log_path, args.verbose)
    try:
        logger.log("RABO CAMERA READ-ONLY TEST START")
        logger.log(f"duration={args.duration}")
        logger.log(f"output_dir={args.output_dir}")

        topic_list_result = run_ros2_topic_cmd(["list", "-t"], timeout=10.0)
        logger.log("$ " + " ".join(topic_list_result.get("cmd", ["ros2", "topic", "list", "-t"])))
        for line in topic_list_result.get("stdout", []):
            logger.log(line)
        for line in topic_list_result.get("stderr", []):
            logger.log("stderr: " + line)
        if "error" in topic_list_result:
            logger.log("error: " + topic_list_result["error"])

        all_topics = parse_topic_list_t(topic_list_result.get("stdout", []))
        camera_topics = select_camera_topics(all_topics)
        logger.log(f"camera_candidate_count={len(camera_topics)}")

        for topic in camera_topics:
            inspect_topic_info(topic, logger)

        subscribe_and_sample(camera_topics, args.duration, args.output_dir, logger)

        summary_path = args.output_dir / "camera_test_summary.json"
        write_json_summary(summary_path, camera_topics, topic_list_result)
        logger.log(f"WROTE_JSON_SUMMARY {display_path(summary_path)}")

        report = build_report(
            topics=camera_topics,
            all_topics=all_topics,
            topic_list_result=topic_list_result,
            duration=args.duration,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
        args.report_path.write_text(report, encoding="utf-8")
        logger.log(f"WROTE_REPORT {display_path(args.report_path)}")

        pass_count = sum(1 for topic in camera_topics if topic.frames > 0)
        logger.log(f"CAMERA_STREAM_RESULT frames_positive_topics={pass_count}/{len(camera_topics)}")
        print(f"Camera test complete. Report: {args.report_path}")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
