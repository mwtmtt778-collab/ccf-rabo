#!/usr/bin/env python3
"""Read-only short probe for the Rabo-to-LeRobot recording interface.

The probe never publishes commands and never calls motion/reset APIs.  While a
human runs a short representative motion in another terminal, it checks:

* the 26D ACT baseline state (7+7 arm joints, 6+6 hand clench values),
* whether state polling can sustain the requested frequency,
* whether the hand clench values change during grasp/release,
* the fixed, left-wrist, and right-wrist RGB streams,
* approximate camera/state receive-time alignment.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_HAND_DEMO = PROJECT_ROOT / "agents" / "arm_hand_demo" / "__init__.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "act_recording_probe"

REQUIRED_IDS = ("LEFT_ARM_ID", "RIGHT_ARM_ID", "LEFT_HAND_ID", "RIGHT_HAND_ID")
EXPECTED_COMPONENT_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand_clench": 6,
    "right_hand_clench": 6,
}
KNOWN_CAMERAS = {
    "fixed_rgb": "r6ef2dc_tp_cam_303d2b1ce0",
    "left_wrist_rgb": "rbd03eb_tp_cam_069a6739f3",
    "right_wrist_rgb": "r412d23_tp_cam_3c67aef2bc",
}
SUPPORTED_IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        with contextlib.suppress(Exception):
            return jsonable(value.tolist())
    return repr(value)


def float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected list/tuple, got {type(value).__name__}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("state contains NaN or Inf")
    return result


def load_robot_ids(path: Path = ARM_HAND_DEMO) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ids: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in REQUIRED_IDS:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            ids[target.id] = node.value.value
    missing = [name for name in REQUIRED_IDS if name not in ids]
    if missing:
        raise RuntimeError(f"missing robot IDs in {path}: {missing}")
    return ids


def clean_output(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def discover_camera_topics() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    commands = [
        ["ros2", "topic", "list", "-t", "--no-daemon"],
        ["ros2", "topic", "list", "-t"],
    ]
    result: dict[str, Any] = {}
    stdout = ""
    for command in commands:
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
            stdout = clean_output(proc.stdout)
            stderr = clean_output(proc.stderr)
            result = {
                "command": command,
                "returncode": proc.returncode,
                "stderr": stderr.splitlines(),
            }
            if proc.returncode == 0:
                break
            if "unrecognized arguments: --no-daemon" not in stderr:
                break
        except Exception as exc:
            result = {"command": command, "error": repr(exc)}
            break

    topic_types: dict[str, list[str]] = {}
    pattern = re.compile(r"^(?P<name>\S+)\s+\[(?P<types>[^\]]+)\]\s*$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        topic_types[match.group("name")] = [
            item.strip() for item in match.group("types").split(",") if item.strip()
        ]

    selected: dict[str, dict[str, str]] = {}
    for label, suffix in KNOWN_CAMERAS.items():
        candidates = [
            (name, types)
            for name, types in topic_types.items()
            if (name == suffix or name.endswith("/" + suffix))
            and any(type_name in SUPPORTED_IMAGE_TYPES for type_name in types)
        ]
        if not candidates:
            continue
        name, types = sorted(candidates, key=lambda item: len(item[0]))[0]
        selected[label] = {
            "topic": name,
            "type": next(type_name for type_name in types if type_name in SUPPORTED_IMAGE_TYPES),
        }

    result["topic_count"] = len(topic_types)
    result["selected"] = selected
    return selected, result


def load_message_class(type_name: str) -> Any:
    try:
        from rosidl_runtime_py.utilities import get_message

        return get_message(type_name)
    except Exception:
        package, _, message = type_name.partition("/msg/")
        module = __import__(f"{package}.msg", fromlist=[message])
        return getattr(module, message)


def ros_stamp_seconds(msg: Any) -> float | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except Exception:
        return None


def save_first_image(msg: Any, type_name: str, path_prefix: Path) -> tuple[str | None, str | None]:
    try:
        if type_name == "sensor_msgs/msg/CompressedImage":
            payload = bytes(msg.data)
            if not payload:
                return None, "compressed image is empty"
            fmt = str(getattr(msg, "format", "")).lower()
            suffix = ".png" if "png" in fmt else ".jpg"
            path = path_prefix.with_suffix(suffix)
            path.write_bytes(payload)
            return str(path), None

        width = int(msg.width)
        height = int(msg.height)
        step = int(msg.step)
        encoding = str(msg.encoding).lower()
        data = bytes(msg.data)
        if encoding == "mono8":
            rows = [data[row * step : row * step + width] for row in range(height)]
            payload = b"P5\n%d %d\n255\n" % (width, height) + b"".join(rows)
            path = path_prefix.with_suffix(".pgm")
        elif encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
            channels = 4 if "a8" in encoding else 3
            row_width = width * channels
            pixels = bytearray()
            for row in range(height):
                row_data = data[row * step : row * step + row_width]
                for col in range(0, len(row_data), channels):
                    pixel = row_data[col : col + channels]
                    if encoding.startswith("bgr"):
                        pixels.extend((pixel[2], pixel[1], pixel[0]))
                    else:
                        pixels.extend((pixel[0], pixel[1], pixel[2]))
            payload = b"P6\n%d %d\n255\n" % (width, height) + bytes(pixels)
            path = path_prefix.with_suffix(".ppm")
        else:
            return None, f"unsupported image encoding: {encoding}"
        path.write_bytes(payload)
        return str(path), None
    except Exception as exc:
        return None, repr(exc)


@dataclass
class CameraRecord:
    label: str
    topic: str
    type_name: str
    arrivals: list[float] = field(default_factory=list)
    ros_stamps: list[float] = field(default_factory=list)
    first_summary: dict[str, Any] | None = None
    sample_path: str | None = None
    sample_error: str | None = None
    subscribe_error: str | None = None


class CameraSampler:
    def __init__(self, topics: dict[str, dict[str, str]], output_dir: Path):
        self.records = {
            label: CameraRecord(label, info["topic"], info["type"]) for label, info in topics.items()
        }
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._active = False
        self._window_start = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._node: Any = None
        self._rclpy: Any = None
        self._executor: Any = None
        self._owns_rclpy_context = False
        self._subscriptions: list[Any] = []
        self.init_error: str | None = None

    def start(self) -> None:
        if not self.records:
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

            self._rclpy = rclpy
            self._owns_rclpy_context = not rclpy.ok()
            if self._owns_rclpy_context:
                rclpy.init(args=None)
            self._node = rclpy.create_node("rabo_act_recording_readonly_probe")
            # Do not use rclpy's global executor: V1's point-cloud detector
            # spins another node in the main thread during this probe.
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                # The three verified Rabo RGB publishers are
                # RELIABLE/VOLATILE.  Match them exactly; this also avoids the
                # platform bridge behavior where discovery succeeds but no
                # frames arrive at a BEST_EFFORT reader.
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            for record in self.records.values():
                try:
                    message_class = load_message_class(record.type_name)
                    subscription = self._node.create_subscription(
                        message_class,
                        record.topic,
                        self._make_callback(record),
                        qos,
                    )
                    self._subscriptions.append(subscription)
                except Exception as exc:
                    record.subscribe_error = repr(exc)
            self._thread = threading.Thread(target=self._spin, name="act-probe-camera", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.init_error = repr(exc)

    def _make_callback(self, record: CameraRecord) -> Callable[[Any], None]:
        def callback(msg: Any) -> None:
            arrival = time.monotonic()
            with self._lock:
                if not self._active:
                    return
                relative = arrival - self._window_start
                record.arrivals.append(relative)
                stamp = ros_stamp_seconds(msg)
                if stamp is not None:
                    record.ros_stamps.append(stamp)
                if record.first_summary is not None:
                    return
                record.first_summary = {
                    "width": getattr(msg, "width", None),
                    "height": getattr(msg, "height", None),
                    "encoding": getattr(msg, "encoding", None),
                    "format": getattr(msg, "format", None),
                    "data_len": len(getattr(msg, "data", [])),
                }
                sample_path, sample_error = save_first_image(
                    msg,
                    record.type_name,
                    self.output_dir / f"sample_{record.label}",
                )
                record.sample_path = sample_path
                record.sample_error = sample_error

        return callback

    def _spin(self) -> None:
        assert self._executor is not None
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._executor.spin_once(timeout_sec=0.05)

    def begin_window(self, start_monotonic: float) -> None:
        with self._lock:
            self._window_start = start_monotonic
            self._active = True

    def end_window(self) -> None:
        with self._lock:
            self._active = False

    def close(self) -> None:
        self.end_window()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._node is not None:
            for subscription in self._subscriptions:
                with contextlib.suppress(Exception):
                    self._node.destroy_subscription(subscription)
            if self._executor is not None:
                with contextlib.suppress(Exception):
                    self._executor.remove_node(self._node)
            with contextlib.suppress(Exception):
                self._node.destroy_node()
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown(timeout_sec=1.0)
        if self._rclpy is not None and self._owns_rclpy_context and self._rclpy.ok():
            with contextlib.suppress(Exception):
                self._rclpy.shutdown()


class IntegratedRecordingProbe:
    """Background probe that reuses device clients owned by an experiment."""

    def __init__(self, output_dir: Path, target_fps: float = 30.0):
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.output_dir = output_dir
        self.target_fps = float(target_fps)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.devices: dict[str, Any] = {}
        self.camera_sampler: CameraSampler | None = None
        self.discovery: dict[str, Any] = {}
        self.hand_before: dict[str, Any] = {}
        self.hand_after: dict[str, Any] = {}
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.overrun_periods = 0
        self.started_at = now_text()
        self.start_monotonic: float | None = None
        self.elapsed_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False

    def start(
        self,
        *,
        left_arm: Any,
        right_arm: Any,
        left_hand: Any,
        right_hand: Any,
    ) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("integrated recording probe already started")
        self.devices = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }
        topics, self.discovery = discover_camera_topics()
        self.camera_sampler = CameraSampler(topics, self.output_dir)
        self.camera_sampler.start()

        try:
            initial_state, _ = read_state26(self.devices)
            self.hand_before = read_hand_joint_dimensions(self.devices)
            initial_check = {"ok": True, "dimension": len(initial_state), "error": None}
        except Exception as exc:
            initial_check = {"ok": False, "dimension": None, "error": repr(exc)}
            self.errors.append({"time_s": 0.0, "error": repr(exc), "stage": "INITIAL_STATE"})

        self.start_monotonic = time.monotonic()
        self.camera_sampler.begin_window(self.start_monotonic)
        self._thread = threading.Thread(target=self._sample_loop, name="act-recording-state-probe", daemon=True)
        self._thread.start()
        self._started = True
        return {
            "initial_state": initial_check,
            "camera_count": len(topics),
            "camera_topics": topics,
            "camera_sampler_init_error": self.camera_sampler.init_error,
        }

    def _sample_loop(self) -> None:
        assert self.start_monotonic is not None
        interval = 1.0 / self.target_fps
        deadline = self.start_monotonic
        while not self._stop.is_set():
            now = time.monotonic()
            if now < deadline:
                self._stop.wait(min(deadline - now, 0.005))
                continue

            read_start = time.monotonic()
            try:
                state, components = read_state26(self.devices)
                read_end = time.monotonic()
                self.samples.append(
                    {
                        "time_s": read_end - self.start_monotonic,
                        "read_duration_s": read_end - read_start,
                        "state": state,
                        "components": components,
                    }
                )
            except Exception as exc:
                read_end = time.monotonic()
                self.errors.append(
                    {
                        "time_s": read_end - self.start_monotonic,
                        "read_duration_s": read_end - read_start,
                        "error": repr(exc),
                        "stage": "STATE_SAMPLE",
                    }
                )

            deadline += interval
            if read_end > deadline:
                missed = max(1, int((read_end - deadline) / interval) + 1)
                self.overrun_periods += missed
                deadline += missed * interval

    def stop(self, *, task_status: str) -> dict[str, Any]:
        if self._stopped:
            return {
                "result": "ALREADY_STOPPED",
                "report_json": str(self.output_dir / "probe_report.json"),
                "report_markdown": str(self.output_dir / "probe_report.md"),
            }
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self.errors.append({"time_s": None, "error": "state probe thread did not stop within 3 seconds"})

        if self.start_monotonic is not None:
            self.elapsed_s = time.monotonic() - self.start_monotonic
        if self.camera_sampler is None:
            self.camera_sampler = CameraSampler({}, self.output_dir)
        self.camera_sampler.end_window()
        self.camera_sampler.close()
        if self.devices:
            self.hand_after = read_hand_joint_dimensions(self.devices)

        state_run = {
            "samples": self.samples,
            "errors": self.errors,
            "overrun_periods": self.overrun_periods,
            "elapsed_s": self.elapsed_s,
        }
        report = summarize_probe(
            state_run=state_run,
            camera_sampler=self.camera_sampler,
            target_fps=self.target_fps,
            duration=self.elapsed_s,
            hand_joints_before=self.hand_before,
            hand_joints_after=self.hand_after,
            discovery=self.discovery,
            shutdown_errors=[],
        )
        report["mode"] = "INTEGRATED_WITH_THREE_NUT_V1"
        report["task_status"] = task_status
        report["started_at"] = self.started_at
        report["finished_at"] = now_text()

        json_path = self.output_dir / "probe_report.json"
        markdown_path = self.output_dir / "probe_report.md"
        json_path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        return {
            "result": report["result"],
            "checks": report["checks"],
            "state_summary": report["state"],
            "camera_summary": {
                label: {
                    key: value
                    for key, value in camera.items()
                    if key not in {"arrival_times_s"}
                }
                for label, camera in report["cameras"].items()
            },
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
        }


def init_devices(ids: dict[str, str]) -> dict[str, Any]:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

    factories: dict[str, Callable[[], Any]] = {
        "left_arm": lambda: LinkerArmA7(robot_id=ids["LEFT_ARM_ID"], mode="sim"),
        "right_arm": lambda: LinkerArmA7(robot_id=ids["RIGHT_ARM_ID"], mode="sim"),
        "left_hand": lambda: LinkerHandO6Left(robot_id=ids["LEFT_HAND_ID"], mode="sim"),
        "right_hand": lambda: LinkerHandO6Right(robot_id=ids["RIGHT_HAND_ID"], mode="sim"),
    }
    devices: dict[str, Any] = {}
    try:
        for name, factory in factories.items():
            print(f"初始化只读设备：{name}", flush=True)
            devices[name] = factory()
    except Exception:
        shutdown_devices(devices)
        raise
    return devices


def shutdown_devices(devices: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name in ("left_hand", "right_hand", "left_arm", "right_arm"):
        device = devices.get(name)
        if device is None or not hasattr(device, "shutdown"):
            continue
        try:
            device.shutdown()
        except Exception as exc:
            errors.append(f"{name}: {repr(exc)}")
    return errors


def read_component(device: Any, method_name: str, expected_dim: int) -> list[float]:
    values = float_list(getattr(device, method_name)())
    if len(values) != expected_dim:
        raise ValueError(f"{method_name} expected {expected_dim} values, got {len(values)}")
    return values


def read_state26(devices: dict[str, Any]) -> tuple[list[float], dict[str, list[float]]]:
    components = {
        "left_arm": read_component(devices["left_arm"], "get_joint_angles", 7),
        "right_arm": read_component(devices["right_arm"], "get_joint_angles", 7),
        "left_hand_clench": read_component(devices["left_hand"], "get_clench", 6),
        "right_hand_clench": read_component(devices["right_hand"], "get_clench", 6),
    }
    state = [value for name in EXPECTED_COMPONENT_DIMS for value in components[name]]
    return state, components


def read_hand_joint_dimensions(devices: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("left_hand", "right_hand"):
        try:
            values = float_list(devices[name].get_joint_angles())
            result[name] = {"dimension": len(values), "values": values, "error": None}
        except Exception as exc:
            result[name] = {"dimension": None, "values": None, "error": repr(exc)}
    return result


def vector_ranges(samples: list[list[float]]) -> list[float]:
    if not samples:
        return []
    return [max(column) - min(column) for column in zip(*samples)]


def rate_summary(times: list[float], target_fps: float) -> dict[str, Any]:
    if len(times) < 2:
        return {
            "sample_count": len(times),
            "effective_fps": None,
            "interval_mean_s": None,
            "interval_std_s": None,
            "interval_max_s": None,
            "target_fps": target_fps,
        }
    intervals = [b - a for a, b in zip(times, times[1:])]
    elapsed = times[-1] - times[0]
    return {
        "sample_count": len(times),
        "effective_fps": (len(times) - 1) / elapsed if elapsed > 0 else None,
        "interval_mean_s": statistics.fmean(intervals),
        "interval_std_s": statistics.pstdev(intervals),
        "interval_max_s": max(intervals),
        "target_fps": target_fps,
    }


def nearest_offsets(reference_times: list[float], camera_times: list[float]) -> dict[str, Any]:
    if not reference_times or not camera_times:
        return {"sample_count": 0, "mean_abs_offset_s": None, "max_abs_offset_s": None}
    offsets: list[float] = []
    camera_index = 0
    for reference in reference_times:
        while camera_index + 1 < len(camera_times) and abs(camera_times[camera_index + 1] - reference) <= abs(
            camera_times[camera_index] - reference
        ):
            camera_index += 1
        offsets.append(abs(camera_times[camera_index] - reference))
    return {
        "sample_count": len(offsets),
        "mean_abs_offset_s": statistics.fmean(offsets),
        "max_abs_offset_s": max(offsets),
    }


def sample_states(devices: dict[str, Any], duration: float, target_fps: float) -> dict[str, Any]:
    interval = 1.0 / target_fps
    start = time.monotonic()
    deadline = start
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    overruns = 0

    while True:
        now = time.monotonic()
        if now - start >= duration:
            break
        if now < deadline:
            time.sleep(min(deadline - now, 0.005))
            continue

        read_start = time.monotonic()
        try:
            state, components = read_state26(devices)
            read_end = time.monotonic()
            samples.append(
                {
                    "time_s": read_end - start,
                    "read_duration_s": read_end - read_start,
                    "state": state,
                    "components": components,
                }
            )
        except Exception as exc:
            read_end = time.monotonic()
            errors.append(
                {
                    "time_s": read_end - start,
                    "read_duration_s": read_end - read_start,
                    "error": repr(exc),
                }
            )

        deadline += interval
        if read_end > deadline:
            missed = max(1, int((read_end - deadline) / interval) + 1)
            overruns += missed
            deadline += missed * interval

    return {
        "start_monotonic": start,
        "samples": samples,
        "errors": errors,
        "overrun_periods": overruns,
        "elapsed_s": time.monotonic() - start,
    }


def summarize_probe(
    state_run: dict[str, Any],
    camera_sampler: CameraSampler,
    target_fps: float,
    duration: float,
    hand_joints_before: dict[str, Any],
    hand_joints_after: dict[str, Any],
    discovery: dict[str, Any],
    shutdown_errors: list[str],
) -> dict[str, Any]:
    samples = state_run["samples"]
    state_times = [sample["time_s"] for sample in samples]
    states = [sample["state"] for sample in samples]
    ranges = vector_ranges(states)
    component_ranges: dict[str, list[float]] = {}
    offset = 0
    for name, dimension in EXPECTED_COMPONENT_DIMS.items():
        component_ranges[name] = ranges[offset : offset + dimension]
        offset += dimension

    state_rate = rate_summary(state_times, target_fps)
    read_durations = [sample["read_duration_s"] for sample in samples]
    state_summary = {
        **state_rate,
        "valid_26d_samples": len(samples),
        "derived_next_state_action_pairs": max(0, len(samples) - 1),
        "derived_action_definition": "action[t] = observation.state[t + 1]",
        "read_error_count": len(state_run["errors"]),
        "read_duration_mean_s": statistics.fmean(read_durations) if read_durations else None,
        "read_duration_max_s": max(read_durations) if read_durations else None,
        "overrun_periods": state_run["overrun_periods"],
        "per_dimension_range": ranges,
        "component_ranges": component_ranges,
        "max_state_range": max(ranges) if ranges else None,
    }

    cameras: dict[str, Any] = {}
    for label in KNOWN_CAMERAS:
        record = camera_sampler.records.get(label)
        if record is None:
            cameras[label] = {
                "status": "NOT_FOUND",
                "topic": None,
                "frame_count": 0,
            }
            continue
        camera_rate = rate_summary(record.arrivals, target_fps)
        cameras[label] = {
            "status": "AVAILABLE" if record.arrivals else "NO_FRAMES",
            "topic": record.topic,
            "type": record.type_name,
            "frame_count": len(record.arrivals),
            "effective_fps": camera_rate["effective_fps"],
            "interval_mean_s": camera_rate["interval_mean_s"],
            "interval_std_s": camera_rate["interval_std_s"],
            "interval_max_s": camera_rate["interval_max_s"],
            "ros_stamp_count": len(record.ros_stamps),
            "first_summary": record.first_summary,
            "sample_path": record.sample_path,
            "sample_error": record.sample_error,
            "subscribe_error": record.subscribe_error,
            "state_receive_alignment": nearest_offsets(state_times, record.arrivals),
            "arrival_times_s": record.arrivals,
        }

    effective_fps = state_rate["effective_fps"] or 0.0
    state_ready = bool(samples) and not state_run["errors"] and effective_fps >= target_fps * 0.9
    cameras_available = all(item["status"] == "AVAILABLE" for item in cameras.values())
    camera_rates_ready = all((item.get("effective_fps") or 0.0) >= target_fps * 0.9 for item in cameras.values())
    hand_motion = {
        side: max(component_ranges.get(f"{side}_hand_clench", []) or [0.0]) > 1e-4
        for side in ("left", "right")
    }
    ready_label = f"READY_{target_fps:g}HZ"
    result = ready_label if state_ready and cameras_available and camera_rates_ready else "CHECK"

    return {
        "generated_at": now_text(),
        "read_only": True,
        "requested_duration_s": duration,
        "requested_fps": target_fps,
        "result": result,
        "checks": {
            "state_26d_ready": state_ready,
            "derived_action_pairs_available": len(samples) >= 2,
            "all_three_cameras_available": cameras_available,
            "all_three_camera_rates_ready": camera_rates_ready,
            "left_hand_clench_changed": hand_motion["left"],
            "right_hand_clench_changed": hand_motion["right"],
        },
        "state": state_summary,
        "state_samples": samples,
        "state_errors": state_run["errors"],
        "hand_joint_angles_before": hand_joints_before,
        "hand_joint_angles_after": hand_joints_after,
        "cameras": cameras,
        "camera_discovery": discovery,
        "camera_sampler_init_error": camera_sampler.init_error,
        "shutdown_errors": shutdown_errors,
        "interpretation": {
            "clench_unchanged": "如果探针期间没有执行抓取/释放，这是正常的；执行过仍无变化则需要检查get_clench语义。",
            "alignment": "相机对齐误差按本机接收时间估算，不等同于硬件时间戳精度。",
            "ready": f"{ready_label}要求26维状态及三路相机都达到目标频率的90%。",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    state = report["state"]
    lines = [
        "# ACT 录制源短探针报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 结论：`{report['result']}`",
        f"- 目标：`{report['requested_fps']} Hz`，持续 `{report['requested_duration_s']} s`",
        "- 安全约束：只读；未调用运动、手部控制、场景修改或重置接口。",
        "",
        "## 26维机器人状态",
        "",
        f"- 有效样本：`{state['valid_26d_samples']}`",
        f"- 可生成相邻状态动作对：`{state['derived_next_state_action_pairs']}`",
        f"- 读取错误：`{state['read_error_count']}`",
        f"- 实际频率：`{state['effective_fps']}`",
        f"- 平均读取耗时：`{state['read_duration_mean_s']}` 秒",
        f"- 最大读取耗时：`{state['read_duration_max_s']}` 秒",
        f"- 错过周期：`{state['overrun_periods']}`",
        f"- 左手 clench 有变化：`{report['checks']['left_hand_clench_changed']}`",
        f"- 右手 clench 有变化：`{report['checks']['right_hand_clench_changed']}`",
        "",
        "## RGB相机",
        "",
        "| 相机 | 状态 | 帧数 | 实际FPS | 分辨率/编码 | 最近状态帧最大偏差(s) |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for label, camera in report["cameras"].items():
        summary = camera.get("first_summary") or {}
        image_text = f"{summary.get('width')}x{summary.get('height')} {summary.get('encoding') or summary.get('format')}"
        alignment = camera.get("state_receive_alignment") or {}
        lines.append(
            f"| {label} | {camera.get('status')} | {camera.get('frame_count')} | "
            f"{camera.get('effective_fps')} | {image_text} | {alignment.get('max_abs_offset_s')} |"
        )
    lines += [
        "",
        "## 判断方法",
        "",
        f"- `READY_{report['requested_fps']:g}HZ`：26维状态及三路RGB都达到目标频率的90%。",
        "- `CHECK`：查看JSON中的状态读取错误、相机订阅错误、实际频率和首帧信息。",
        "- 如果探针期间明确执行过抓取和释放，但对应 `clench_changed=false`，暂不应把该6维量作为ACT手部状态。",
        "- 每路相机的首帧样本保存在本报告同目录，可人工确认画面和相机身份。",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读检查ACT采样所需的三路RGB和26维机器人状态。")
    parser.add_argument("--duration", type=float, default=20.0, help="采样持续时间，默认20秒。")
    parser.add_argument("--fps", type=float, default=30.0, help="目标状态采样频率，默认30Hz。")
    parser.add_argument("--start-delay", type=float, default=3.0, help="初始化后倒计时秒数。")
    parser.add_argument("--output-dir", type=Path, help="输出目录；默认按时间创建在outputs/act_recording_probe。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration <= 0 or args.fps <= 0 or args.start_delay < 0:
        raise SystemExit("duration和fps必须大于0，start-delay不能小于0。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / stamp)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = PROJECT_ROOT / "logs" / "ros"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))

    print("ACT录制源短探针（只读）", flush=True)
    print("不会调用运动、夹爪控制、场景修改或重置接口。", flush=True)
    print(f"输出目录：{output_dir}", flush=True)

    camera_topics, discovery = discover_camera_topics()
    print(f"发现RGB相机：{len(camera_topics)}/3", flush=True)
    for label in KNOWN_CAMERAS:
        info = camera_topics.get(label)
        print(f"  {label}: {info['topic'] if info else 'NOT_FOUND'}", flush=True)

    devices: dict[str, Any] = {}
    camera_sampler = CameraSampler(camera_topics, output_dir)
    shutdown_errors: list[str] = []
    state_run: dict[str, Any] = {
        "samples": [],
        "errors": [],
        "overrun_periods": 0,
        "elapsed_s": 0.0,
    }
    hand_before: dict[str, Any] = {}
    hand_after: dict[str, Any] = {}

    try:
        ids = load_robot_ids()
        devices = init_devices(ids)
        initial_state, _ = read_state26(devices)
        print(f"26维状态初检：PASS（dimension={len(initial_state)}）", flush=True)
        hand_before = read_hand_joint_dimensions(devices)

        camera_sampler.start()
        if camera_sampler.init_error:
            print(f"相机订阅初始化失败：{camera_sampler.init_error}", flush=True)

        print("请在另一个终端执行一段包含机械臂运动、抓取和释放的代表动作。", flush=True)
        countdown_deadline = time.monotonic() + args.start_delay
        last_announced: int | None = None
        while True:
            remaining = countdown_deadline - time.monotonic()
            if remaining <= 0:
                break
            shown = int(math.ceil(remaining))
            if shown != last_announced:
                print(f"{shown} 秒后开始采样……", flush=True)
                last_announced = shown
            time.sleep(min(0.1, remaining))

        print(f"开始采样：{args.duration:.1f}秒，目标{args.fps:.1f}Hz", flush=True)
        window_start = time.monotonic()
        camera_sampler.begin_window(window_start)
        state_run = sample_states(devices, args.duration, args.fps)
        camera_sampler.end_window()
        hand_after = read_hand_joint_dimensions(devices)
        print("采样完成，正在生成报告。", flush=True)
    except KeyboardInterrupt:
        print("用户中断：保留已经采到的数据。", flush=True)
        camera_sampler.end_window()
        if devices:
            hand_after = read_hand_joint_dimensions(devices)
    except Exception as exc:
        state_run["errors"].append({"time_s": None, "error": repr(exc)})
        print(f"探针异常：{repr(exc)}", flush=True)
    finally:
        camera_sampler.close()
        shutdown_errors = shutdown_devices(devices)

    report = summarize_probe(
        state_run=state_run,
        camera_sampler=camera_sampler,
        target_fps=args.fps,
        duration=args.duration,
        hand_joints_before=hand_before,
        hand_joints_after=hand_after,
        discovery=discovery,
        shutdown_errors=shutdown_errors,
    )
    json_path = output_dir / "probe_report.json"
    markdown_path = output_dir / "probe_report.md"
    json_path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"结论：{report['result']}", flush=True)
    print(f"JSON：{json_path}", flush=True)
    print(f"报告：{markdown_path}", flush=True)
    return 0 if report["result"] == "READY_30HZ" else 2


if __name__ == "__main__":
    raise SystemExit(main())
