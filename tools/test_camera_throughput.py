#!/usr/bin/env python3
"""Read-only camera throughput comparison diagnostic for the Rabo sim runtime.

Purpose
-------
Localize why the three RGB streams only arrive at ~0.9-2.0Hz instead of the
target 10Hz, without changing any task/algorithm parameter.  See
``deepseek读取文档.md`` section 6 (P0) for the acceptance criteria and the
diagnostic steps this tool implements.

What it does (all read-only w.r.t. the robot/ROS business state)
----------------------------------------------------------------
1. Lists topics and runs ``ros2 topic info -v`` for each camera to record the
   publisher QoS and subscription matching state.
2. Inventories every topic that carries one of the known camera serials, so we
   can see whether a CompressedImage or lower-resolution variant exists.
3. Runs a subscription ladder and measures, per configuration:
   * fixed-window receive FPS (frame_count / actual window duration) plus the
     inter-arrival rhythm,
   * received frames' header-timeline FPS (NOT the publisher rate),
   * whether consecutive header-stamp gaps align to integer multiples of the
     target period (this is ambiguous between a slow source and dropped
     frames, and is labelled as such - never as proof of one specific cause),
   * first-frame latency, received payload bytes and payload MB/s per camera,
   * this process's CPU% and RSS while that configuration is subscribed.
4. Writes a JSON report and a Markdown summary under ``reports/camera_throughput``.

It never publishes, never calls motion/hand/reset APIs, and never modifies
camera parameters.  It does not depend on ``rabo_robocap``.

Usage
-----
    python3 -u tools/test_camera_throughput.py --compare all --duration 15
    python3 -u tools/test_camera_throughput.py --cameras fixed_rgb,left_wrist_rgb
    python3 -u tools/test_camera_throughput.py --rounds 2
    python3 -u tools/test_camera_throughput.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional, only used for CPU/RSS
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "camera_throughput"

# Camera serial suffixes are the only stable identifiers on this platform.
KNOWN_CAMERAS = {
    "fixed_rgb": "r6ef2dc_tp_cam_303d2b1ce0",
    "left_wrist_rgb": "rbd03eb_tp_cam_069a6739f3",
    "right_wrist_rgb": "r412d23_tp_cam_3c67aef2bc",
}
SUPPORTED_IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
# Prefer the raw Image type (what the current pipeline subscribes to, and the
# 6MB/frame payload whose throughput is in question) over CompressedImage.
IMAGE_TYPE_PRIORITY = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

MODE_KINDS = ("single", "dual", "triple")

# Diagnostic outcome states (top-level run_status).
RUN_STATUS_OK = "OK"
RUN_STATUS_PARTIAL = "PARTIAL_NO_FRAMES"
RUN_STATUS_NO_FRAMES = "SUBSCRIBED_BUT_NO_FRAMES"
RUN_STATUS_SUBSCRIBE_FAILED = "SUBSCRIPTION_FAILED"
RUN_STATUS_SAMPLING_ERROR = "SAMPLING_ERROR"
RUN_STATUS_DISCOVERY_FAILED = "DISCOVERY_FAILED"
RUN_STATUS_TOPICS_NOT_FOUND = "TOPICS_NOT_FOUND"
RUN_STATUS_METADATA_FAILED = "METADATA_FAILED"
RUN_STATUS_PARTIAL_METADATA = "PARTIAL_METADATA_FAILED"


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
    return repr(value)


def clean_output(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Pure-math analysis helpers (self-testable without any ROS or hardware)
# --------------------------------------------------------------------------- #


def rate_summary(times: list[float], target_fps: float) -> dict[str, Any]:
    """Summarise a strictly-ordered timestamp series as an inter-arrival rate.

    ``effective_fps`` here is ``(N-1) / (last - first)``: the inter-arrival
    rhythm, NOT ``N / window_duration``.  Fixed-window throughput is computed
    separately in ``summarize_camera`` as ``window_receive_fps``.
    """
    base: dict[str, Any] = {
        "sample_count": len(times),
        "effective_fps": None,
        "interval_mean_s": None,
        "interval_std_s": None,
        "interval_min_s": None,
        "interval_max_s": None,
        "target_fps": target_fps,
    }
    if len(times) < 2:
        return base
    intervals = [b - a for a, b in zip(times, times[1:])]
    elapsed = times[-1] - times[0]
    return {
        **base,
        "effective_fps": (len(times) - 1) / elapsed if elapsed > 0 else None,
        "interval_mean_s": statistics.fmean(intervals),
        "interval_std_s": statistics.pstdev(intervals),
        "interval_min_s": min(intervals),
        "interval_max_s": max(intervals),
    }


def stamp_gap_analysis(stamps: list[float], period_s: float) -> dict[str, Any]:
    """Describe how received header-stamp gaps align to the target period.

    If consecutive *received* header stamps are near-integer multiples of the
    target period, it is *ambiguous* whether the source publishes slowly or
    the subscriber drops intermediate frames - both produce the same pattern.
    This function therefore reports the alignment statistics neutrally and
    never asserts a specific cause.
    """
    empty: dict[str, Any] = {
        "period_s": period_s,
        "delta_count": 0,
        "positive_delta_count": 0,
        "non_monotonic_count": 0,
        "single_period_ratio": None,
        "multi_period_ratio": None,
        "aligned_multiple_distribution": {},
        "median_multiple": None,
        "max_multiple": None,
        # None until an actual gap analysis is available (>= 2 stamps).
        "ambiguous_slow_source_or_dropped_frames": None,
        "median_delta_s": None,
        "max_delta_s": None,
    }
    if len(stamps) < 2 or period_s <= 0:
        return empty

    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    non_monotonic = sum(1 for d in deltas if d <= 0)
    positive = [d for d in deltas if d > 0]

    aligned_dist: dict[int, int] = {}
    single_count = 0
    multi_count = 0
    for d in positive:
        ratio = d / period_s
        k = round(ratio)
        if k >= 1 and abs(ratio - k) <= 0.25:
            aligned_dist[k] = aligned_dist.get(k, 0) + 1
            if k == 1:
                single_count += 1
            else:
                multi_count += 1

    # Weighted median over the actual multiple samples, not the dict keys.
    multiples_flat = sorted(k for k, count in aligned_dist.items() for _ in range(count))

    return {
        "period_s": period_s,
        "delta_count": len(deltas),
        "positive_delta_count": len(positive),
        "non_monotonic_count": non_monotonic,
        "single_period_ratio": single_count / len(positive) if positive else None,
        "multi_period_ratio": multi_count / len(positive) if positive else None,
        "aligned_multiple_distribution": aligned_dist,
        "median_multiple": statistics.median(multiples_flat) if multiples_flat else None,
        "max_multiple": max(aligned_dist) if aligned_dist else None,
        # Cannot be decided from header stamps alone: a slow publisher and a
        # dropping subscriber both yield integer-multiple gaps.
        "ambiguous_slow_source_or_dropped_frames": True,
        "median_delta_s": statistics.median(positive) if positive else None,
        "max_delta_s": max(positive) if positive else None,
    }


def self_test() -> bool:
    """Validate the pure-math helpers without ROS or hardware."""
    ok = True

    times = [i * 0.1 for i in range(101)]
    rate = rate_summary(times, 10.0)
    if abs((rate["effective_fps"] or 0.0) - 10.0) > 1e-6:
        ok = False
        print(f"FAIL rate_summary perfect 10Hz: {rate}", file=sys.stderr)

    empty_rate = rate_summary([], 10.0)
    if empty_rate["effective_fps"] is not None:
        ok = False
        print(f"FAIL rate_summary empty: {empty_rate}", file=sys.stderr)

    # 10Hz source, subscriber receives every 10th frame -> 1Hz gaps of 1.0s.
    stamps = [i * 0.1 for i in range(0, 1001, 10)]
    gaps = stamp_gap_analysis(stamps, 0.1)
    if gaps["max_multiple"] != 10 or (gaps["multi_period_ratio"] or 0.0) < 0.9:
        ok = False
        print(f"FAIL stamp_gap_analysis multi-period: {gaps}", file=sys.stderr)
    if gaps["median_multiple"] != 10:
        ok = False
        print(f"FAIL stamp_gap_analysis weighted median: {gaps}", file=sys.stderr)

    # A genuinely slow 1Hz publisher also yields 10x multiples.  It must NOT be
    # asserted as "subscriber drops" - the analysis stays ambiguous.
    slow = stamp_gap_analysis([float(i) for i in range(11)], 0.1)
    if slow["non_monotonic_count"] != 0 or slow["delta_count"] != 10:
        ok = False
        print(f"FAIL stamp_gap_analysis slow publisher: {slow}", file=sys.stderr)
    if not slow["ambiguous_slow_source_or_dropped_frames"]:
        ok = False
        print(f"FAIL stamp_gap_analysis must stay ambiguous: {slow}", file=sys.stderr)

    # Weighted median regression: 99x k=1 and 1x k=10 must median to 1, not 5.5.
    mixed_stamps = [0.0]
    for _ in range(99):
        mixed_stamps.append(mixed_stamps[-1] + 0.1)
    mixed_stamps.append(mixed_stamps[-1] + 1.0)
    mixed = stamp_gap_analysis(mixed_stamps, 0.1)
    if mixed["median_multiple"] != 1:
        ok = False
        print(f"FAIL stamp_gap_analysis weighted median mixed: {mixed}", file=sys.stderr)

    if ok:
        print("self-test: PASS")
    return ok


# --------------------------------------------------------------------------- #
# ROS discovery helpers
# --------------------------------------------------------------------------- #


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
    except subprocess.TimeoutExpired as exc:
        return {"cmd": cmd, "ok": False, "error": f"timeout: {exc}"}
    except Exception as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def ros2_topic_cmd(args: list[str], timeout: float = 15.0) -> dict[str, Any]:
    result = run_cmd(["ros2", "topic", *args, "--no-daemon"], timeout=timeout)
    text = "\n".join(result.get("stderr", []) + result.get("stdout", []))
    if "unrecognized arguments: --no-daemon" not in text:
        return result
    return run_cmd(["ros2", "topic", *args], timeout=timeout)


def list_topic_types() -> tuple[list[tuple[str, list[str]]], dict[str, Any]]:
    """Return ``(pairs, result)`` from ``ros2 topic list -t``."""
    result = ros2_topic_cmd(["list", "-t"])
    pairs: list[tuple[str, list[str]]] = []
    pattern = re.compile(r"^(?P<name>\S+)\s+\[(?P<types>[^\]]+)\]\s*$")
    for line in result.get("stdout", []):
        match = pattern.match(line.strip())
        if not match:
            continue
        pairs.append(
            (
                match.group("name"),
                [item.strip() for item in match.group("types").split(",") if item.strip()],
            )
        )
    return pairs, result


def pick_image_type(types: list[str]) -> str | None:
    for preferred in IMAGE_TYPE_PRIORITY:
        if preferred in types:
            return preferred
    return next((t for t in types if t in SUPPORTED_IMAGE_TYPES), None)


def discover_camera_topics(
    pairs: list[tuple[str, list[str]]],
) -> dict[str, dict[str, str]]:
    """Map each known camera label to one concrete topic/type, if present."""
    selected: dict[str, dict[str, str]] = {}
    for label, suffix in KNOWN_CAMERAS.items():
        candidates = [
            (name, types)
            for name, types in pairs
            if (name == suffix or name.endswith("/" + suffix))
            and pick_image_type(types) is not None
        ]
        if not candidates:
            continue

        def sort_key(item: tuple[str, list[str]]) -> tuple[int, int, str]:
            name, types = item
            chosen = pick_image_type(types) or ""
            type_rank = (
                IMAGE_TYPE_PRIORITY.index(chosen)
                if chosen in IMAGE_TYPE_PRIORITY
                else len(IMAGE_TYPE_PRIORITY)
            )
            return (type_rank, len(name), name)

        # Prefer the raw Image type over CompressedImage across *different*
        # candidate topic names, then break ties by shortest name.
        name, types = sorted(candidates, key=sort_key)[0]
        chosen = pick_image_type(types)
        if chosen is not None:
            selected[label] = {"topic": name, "type": chosen}
    return selected


def find_camera_alternates(
    pairs: list[tuple[str, list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    """List every topic whose name embeds a known camera serial.

    This surfaces CompressedImage / theora / lower-resolution variants that the
    primary ``Image`` subscription would otherwise miss.
    """
    alternates: dict[str, list[dict[str, Any]]] = {}
    for label, suffix in KNOWN_CAMERAS.items():
        matches = [
            {"topic": name, "types": types}
            for name, types in pairs
            if name == suffix or name.endswith("/" + suffix) or suffix in name
        ]
        alternates[label] = sorted(matches, key=lambda item: len(item["topic"]))
    return alternates


def topic_info_verbose(topic: str) -> dict[str, Any]:
    return ros2_topic_cmd(["info", "-v", topic], timeout=15.0)


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


# --------------------------------------------------------------------------- #
# Subscription sampler
# --------------------------------------------------------------------------- #


@dataclass
class CameraMetrics:
    label: str
    topic: str
    type_name: str
    arrivals: list[float] = field(default_factory=list)
    ros_stamps: list[float] = field(default_factory=list)
    total_bytes: int = 0
    subscribe_monotonic: float | None = None
    first_callback_monotonic: float | None = None
    first_summary: dict[str, Any] | None = None
    subscribe_error: str | None = None


def sample_mode(
    camera_topics: dict[str, dict[str, str]],
    duration: float,
    settle: float,
    node_name: str,
) -> tuple[dict[str, CameraMetrics], dict[str, Any]]:
    """Subscribe to ``camera_topics`` and measure one configuration.

    Returns ``(metrics_by_label, process_stats)``.  The sampler matches the
    verified Rabo publishers' QoS exactly: RELIABLE / VOLATILE / KEEP_LAST
    depth=1 (the same policy used by ``probe_act_recording_sources.py``, which
    the platform confirmed receives frames; BEST_EFFORT receives nothing).

    ``spin_once`` is called on the calling thread, so callbacks run inline and
    no lock is required.  The settle phase spins without recording to drain
    warm-up frames; the measurement window enforces explicit start/end bounds.

    The caller (``main``) is responsible for ``rclpy.init``/``shutdown``
    ownership; this function assumes rclpy is already initialised.
    """
    metrics = {
        label: CameraMetrics(label, info["topic"], info["type"])
        for label, info in camera_topics.items()
    }
    stats: dict[str, Any] = {
        "cpu_percent": [],
        "rss_mb": [],
        "psutil_available": psutil is not None,
        "subscription_count": 0,
        "sampling_error": None,
        "window_start_s": None,
        "window_end_s": None,
        "window_duration_s": None,
    }

    if not camera_topics:
        stats["note"] = "no camera topics to subscribe"
        return metrics, stats

    node = None
    executor = None
    subscriptions: list[Any] = []
    proc: Any = None
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        node = rclpy.create_node(node_name)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Mutable window state captured by the callback closure.
        state = {"active": False, "window_start": 0.0, "window_end": 0.0}

        def make_callback(record: CameraMetrics) -> Callable[[Any], None]:
            def callback(msg: Any) -> None:
                arrival = time.monotonic()
                if record.first_callback_monotonic is None:
                    record.first_callback_monotonic = arrival
                if not state["active"] or arrival < state["window_start"] or arrival > state["window_end"]:
                    return
                record.arrivals.append(arrival)
                stamp = ros_stamp_seconds(msg)
                if stamp is not None:
                    record.ros_stamps.append(stamp)
                with contextlib.suppress(Exception):
                    record.total_bytes += len(msg.data)
                if record.first_summary is None:
                    record.first_summary = {
                        "width": getattr(msg, "width", None),
                        "height": getattr(msg, "height", None),
                        "encoding": getattr(msg, "encoding", None),
                        "format": getattr(msg, "format", None),
                        "data_len": len(getattr(msg, "data", [])),
                    }

            return callback

        for record in metrics.values():
            try:
                message_class = load_message_class(record.type_name)
                subscription = node.create_subscription(
                    message_class,
                    record.topic,
                    make_callback(record),
                    qos,
                )
                subscriptions.append(subscription)
                record.subscribe_monotonic = time.monotonic()
            except Exception as exc:
                record.subscribe_error = repr(exc)

        stats["subscription_count"] = len(subscriptions)
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)  # establish an initial baseline

        # Settle: keep spinning so warm-up frames are consumed, but do not
        # record them.  A settle exception aborts this mode's measurement.
        settle_ok = True
        settle_deadline = time.monotonic() + settle
        while time.monotonic() < settle_deadline:
            remaining = settle_deadline - time.monotonic()
            try:
                executor.spin_once(timeout_sec=max(0.0, min(0.02, remaining)))
            except Exception as exc:
                stats["sampling_error"] = f"settle: {exc!r}"
                settle_ok = False
                break

        if settle_ok:
            # Measurement window.
            window_start = time.monotonic()
            window_end = window_start + duration
            state["window_start"] = window_start
            state["window_end"] = window_end
            state["active"] = True
            if proc is not None:
                # Re-baseline at window_start so the first sample excludes settle.
                proc.cpu_percent(interval=None)

            cpu_sample_at = window_start
            while True:
                now = time.monotonic()
                remaining = window_end - now
                if remaining <= 0:
                    break
                try:
                    executor.spin_once(timeout_sec=min(0.02, remaining))
                except Exception as exc:
                    # A runtime failure must not be reported as "no frames".
                    stats["sampling_error"] = repr(exc)
                    break
                now = time.monotonic()
                if proc is not None and now - cpu_sample_at >= 1.0:
                    stats["cpu_percent"].append(proc.cpu_percent(interval=None))
                    stats["rss_mb"].append(proc.memory_info().rss / (1024 * 1024))
                    cpu_sample_at = now

            # Ensure at least one CPU/RSS sample even for very short windows.
            if proc is not None and not stats["cpu_percent"]:
                stats["cpu_percent"].append(proc.cpu_percent(interval=None))
                stats["rss_mb"].append(proc.memory_info().rss / (1024 * 1024))

            state["active"] = False
            actual_end = time.monotonic()
            # The callback only admits frames with arrival <= window_end.  Clamp
            # the denominator to the planned window so an overshooting final
            # spin_once does not deflate the rate; on a sampling error the loop
            # ends early and the real elapsed time is used instead.
            effective_end = min(actual_end, window_end)
            stats["window_start_s"] = window_start
            stats["window_end_s"] = effective_end
            stats["window_duration_s"] = max(0.0, effective_end - window_start)
    except Exception as exc:
        # Any outer failure (create_node, add_node, psutil, ...) becomes a
        # sampling error rather than an uncaught traceback.
        stats["sampling_error"] = repr(exc)
    finally:
        for subscription in subscriptions:
            if node is not None:
                with contextlib.suppress(Exception):
                    node.destroy_subscription(subscription)
        if executor is not None and node is not None:
            with contextlib.suppress(Exception):
                executor.remove_node(node)
        if node is not None:
            with contextlib.suppress(Exception):
                node.destroy_node()
        if executor is not None:
            with contextlib.suppress(Exception):
                executor.shutdown(timeout_sec=1.0)

    return metrics, stats


# --------------------------------------------------------------------------- #
# Mode ladder and reporting
# --------------------------------------------------------------------------- #


def build_modes(kind: str, camera_labels: list[str]) -> list[list[str]]:
    """Build the list of subscription configurations to measure."""
    if kind == "single":
        return [[label] for label in camera_labels]
    if kind == "dual":
        return [
            [camera_labels[i], camera_labels[j]]
            for i in range(len(camera_labels))
            for j in range(i + 1, len(camera_labels))
        ]
    if kind == "triple":
        return [list(camera_labels)]
    if kind == "all":
        rungs = build_modes("single", camera_labels) + build_modes("dual", camera_labels)
        # Only add the triple rung when all three cameras are actually present,
        # so a 1/2-camera run never fabricates a "triple" out of fewer cameras.
        if len(camera_labels) == len(KNOWN_CAMERAS):
            rungs.append(list(camera_labels))
        return rungs
    raise ValueError(f"unknown compare kind: {kind}")


def mode_name(labels: list[str]) -> str:
    if not labels:
        return "empty"
    if len(labels) == 1:
        return f"single:{labels[0]}"
    if len(labels) == len(KNOWN_CAMERAS):
        return "triple:all"
    return "pair:" + "+".join(labels)


def summarize_camera(
    record: CameraMetrics,
    target_fps: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    period = 1.0 / target_fps
    arrival_rate = rate_summary(record.arrivals, target_fps)
    stamp_rate = rate_summary(record.ros_stamps, target_fps)
    gap = stamp_gap_analysis(record.ros_stamps, period)

    # Non-monotonic header stamps make the timeline FPS meaningless.
    if gap["non_monotonic_count"] > 0:
        stamp_rate["effective_fps"] = None
        stamp_rate["invalid_reason"] = "NON_MONOTONIC_STAMPS"

    window_duration = stats.get("window_duration_s")
    frame_count = len(record.arrivals)
    window_receive_fps = frame_count / window_duration if window_duration and window_duration > 0 else None
    payload_mbps = (
        (record.total_bytes / 1_000_000) / window_duration
        if window_duration and window_duration > 0
        else None
    )
    window_start = stats.get("window_start_s")
    if record.subscribe_error:
        status = "SUBSCRIPTION_FAILED"
    elif record.arrivals:
        status = "AVAILABLE"
    else:
        status = "NO_FRAMES"
    return {
        "status": status,
        "topic": record.topic,
        "type": record.type_name,
        "frame_count": frame_count,
        "window_receive_fps": window_receive_fps,
        "inter_arrival_fps": arrival_rate["effective_fps"],
        "receive": arrival_rate,
        "received_stamp_timeline": stamp_rate,
        "stamp_gap_analysis": gap,
        "first_frame_in_window_s": (
            record.arrivals[0] - window_start
            if record.arrivals and window_start is not None
            else None
        ),
        "first_callback_since_subscribe_s": (
            record.first_callback_monotonic - record.subscribe_monotonic
            if record.first_callback_monotonic is not None and record.subscribe_monotonic is not None
            else None
        ),
        "total_bytes": record.total_bytes,
        "payload_MBps": payload_mbps,
        "first_summary": record.first_summary,
        "subscribe_error": record.subscribe_error,
    }


def summarize_process(stats: dict[str, Any]) -> dict[str, Any]:
    cpu = [float(v) for v in stats.get("cpu_percent", [])]
    rss = [float(v) for v in stats.get("rss_mb", [])]
    return {
        "psutil_available": stats.get("psutil_available"),
        "cpu_percent_mean": statistics.fmean(cpu) if cpu else None,
        "cpu_percent_max": max(cpu) if cpu else None,
        "rss_mb_mean": statistics.fmean(rss) if rss else None,
        "rss_mb_max": max(rss) if rss else None,
        "subscription_count": stats.get("subscription_count"),
        "window_duration_s": stats.get("window_duration_s"),
        "sampling_error": stats.get("sampling_error"),
        "note": stats.get("note"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rabo 相机吞吐对照诊断报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 目标频率：`{report['target_fps']} Hz`",
        f"- 每种配置采样：`{report['duration_per_mode_s']} s`（订阅后稳定 `{report['settle_s']} s`）",
        f"- 轮数：`{report['rounds']}`",
        f"- 运行状态：`{report['run_status']}`",
        "- 安全约束：只读；未调用运动、夹爪、场景或重置接口，未修改相机参数。",
        "",
        "## 相机话题发现",
        "",
        f"- 发现状态：`{report['camera_discovery']['status']}`",
        "",
    ]
    failure_reasons = report.get("failure_reasons")
    if failure_reasons:
        lines.append(f"- failure_reasons：`{', '.join(failure_reasons)}`")
    for label, info in report["camera_discovery"]["selected"].items():
        lines.append(f"- `{label}` -> `{info['topic']}` [{info['type']}]")
    missing = [label for label in KNOWN_CAMERAS if label not in report["camera_discovery"]["selected"]]
    if missing:
        lines.append(f"- 未发现：{', '.join(missing)}")
    info_failed = [
        label
        for label, res in report["camera_discovery"].get("topic_info_verbose", {}).items()
        if not res.get("ok")
    ]
    if info_failed:
        lines.append(f"- `topic info -v` 失败：{', '.join(info_failed)}")
    lines += [
        "",
        "## 各配置吞吐",
        "",
        "| 配置 | 相机 | 状态 | 帧数 | 接收FPS(窗口) | 戳时间线FPS | 最大间隔(s) | 多周期占比 | payload MB/s | CPU%均值 | RSS均值(MB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in report["modes"]:
        proc = mode["process"]
        cpu = proc["cpu_percent_mean"]
        rss = proc["rss_mb_mean"]
        cpu_text = f"{cpu:.1f}" if cpu is not None else ""
        rss_text = f"{rss:.0f}" if rss is not None else ""
        for label in mode["camera_labels"]:
            camera = mode["cameras"].get(label, {})
            gap = camera.get("stamp_gap_analysis", {})
            timeline = camera.get("received_stamp_timeline", {})
            multi = gap.get("multi_period_ratio")
            multi_text = f"{multi:.0%}" if multi is not None else ""
            status_text = camera.get("status", "")
            if camera.get("subscribe_error"):
                status_text += "!subscribe_error"
            lines.append(
                f"| {mode['name']} | {label} | {status_text} | {camera.get('frame_count', 0)} | "
                f"{_fmt(camera.get('window_receive_fps'))} | {_fmt(timeline.get('effective_fps'))} | "
                f"{_fmt(camera.get('receive', {}).get('interval_max_s'))} | {multi_text} | "
                f"{_fmt(camera.get('payload_MBps'))} | {cpu_text} | {rss_text} |"
            )
        if proc.get("sampling_error"):
            err = proc["sampling_error"].replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(f"| {mode['name']} | * | SAMPLING_ERROR: `{err}` | | | | | | | | |")
    lines += [
        "",
        "## 判断方法",
        "",
        "- 接收FPS(窗口) = 帧数 / 实际窗口时长；这是判断是否达标的吞吐量。帧间节奏（inter_arrival_fps）见 JSON。",
        "- 戳时间线FPS = 收到帧在 header 时间轴上的有效频率，**不是**发布器真实发布频率。",
        "- 多周期占比 = 收到帧 header 时间戳差接近目标周期整数倍（k≥2）的比例。该模式**既可能**是发布端低频、**也可能**是订阅端丢帧，单凭时间戳无法区分，请勿据此直接归因。",
        "- payload MB/s = `msg.data` 载荷字节数 / 实际窗口时长，不含 DDS、序列化与传输协议开销。",
        "- 单相机也低频 → 优先排查发布端/桥接层；单相机正常、多相机低频 → 优先排查 Python 反序列化、执行器与内存带宽。",
        "- 各配置按固定顺序串行执行，结果可能受运行顺序影响；建议用 `--rounds 2` 以上交叉复核。",
        "- 完整数值（含每路 camera 的间隔分布、首帧延迟、alternate 话题、topic info -v）见同目录 JSON。",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读相机吞吐对照诊断：单路/双路/三路订阅测FPS、ROS戳、CPU/内存。"
    )
    parser.add_argument(
        "--compare",
        choices=MODE_KINDS + ("all",),
        default="all",
        help="对比模式：single/dual/triple 或 all（默认 all，串行跑完单路+双路+三路）。",
    )
    parser.add_argument(
        "--cameras",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        help="仅测指定相机（逗号分隔：fixed_rgb,left_wrist_rgb,right_wrist_rgb）。与--compare组合使用。",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="每种配置采样秒数，默认15。")
    parser.add_argument("--settle", type=float, default=2.0, help="订阅后稳定等待秒数，默认2。")
    parser.add_argument("--fps", type=float, default=10.0, help="目标频率，默认10Hz。")
    parser.add_argument("--rounds", type=int, default=1, help="整个阶梯重复轮数，默认1。")
    parser.add_argument("--output-dir", type=Path, help="输出目录；默认按时间创建于 reports/camera_throughput。")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="仅运行纯数学自检，不连接ROS或硬件。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if args.duration <= 0 or args.fps <= 0 or args.settle < 0 or args.rounds < 1:
        raise SystemExit("duration/fps必须大于0，settle不能小于0，rounds必须≥1。")

    if args.cameras:
        unknown = [label for label in args.cameras if label not in KNOWN_CAMERAS]
        if unknown:
            raise SystemExit(f"未知相机标签：{unknown}；可选：{', '.join(KNOWN_CAMERAS)}")
        if len(set(args.cameras)) != len(args.cameras):
            raise SystemExit(f"相机标签重复：{args.cameras}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / stamp)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = PROJECT_ROOT / "logs" / "ros"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))

    print("相机吞吐对照诊断（只读）", flush=True)
    print("不会调用运动、夹爪控制、场景修改、重置或相机参数修改接口。", flush=True)
    print(f"输出目录：{output_dir}", flush=True)

    pairs, topic_list_result = list_topic_types()
    selected = discover_camera_topics(pairs)
    alternates = find_camera_alternates(pairs)

    if not topic_list_result.get("ok"):
        # The discovery command itself failed (even if some stdout parsed).
        discovery_status = RUN_STATUS_DISCOVERY_FAILED
    elif not pairs:
        discovery_status = RUN_STATUS_TOPICS_NOT_FOUND
    elif not selected:
        discovery_status = RUN_STATUS_TOPICS_NOT_FOUND
    else:
        discovery_status = "OK"

    print(f"发现RGB相机：{len(selected)}/3（{discovery_status}）", flush=True)
    for label in KNOWN_CAMERAS:
        info = selected.get(label)
        print(f"  {label}: {info['topic'] if info else 'NOT_FOUND'}", flush=True)
        for alt in alternates.get(label, []):
            marker = " <== 主订阅" if info and alt["topic"] == info["topic"] else ""
            print(f"      候选 {alt['topic']} {alt['types']}{marker}", flush=True)

    def write_report(
        run_status: str,
        modes: list[dict[str, Any]],
        missing: list[str],
        extra: dict[str, Any] | None = None,
    ) -> int:
        report = {
            "generated_at": now_text(),
            "read_only": True,
            "target_fps": args.fps,
            "duration_per_mode_s": args.duration,
            "settle_s": args.settle,
            "rounds": args.rounds,
            "run_status": run_status,
            "camera_discovery": {
                "status": discovery_status,
                "selected": selected,
                "missing": missing,
                "alternates": alternates,
                "topic_info_verbose": info_verbose,
                "topic_list": topic_list_result,
            },
            "modes": modes,
        }
        if extra:
            report.update(extra)
        json_path = output_dir / "throughput_report.json"
        markdown_path = output_dir / "throughput_report.md"
        json_path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"JSON：{json_path}", flush=True)
        print(f"报告：{markdown_path}", flush=True)
        return 0 if run_status == RUN_STATUS_OK else 2

    info_verbose = {
        label: topic_info_verbose(info["topic"])
        for label, info in selected.items()
    }

    expected_labels = list(args.cameras) if args.cameras else list(KNOWN_CAMERAS)
    missing = [label for label in expected_labels if label not in selected]

    if discovery_status == RUN_STATUS_DISCOVERY_FAILED:
        # The topic-listing command itself failed; its output is untrustworthy,
        # so even parseable lines must not lead to a false OK.
        return write_report(RUN_STATUS_DISCOVERY_FAILED, [], missing)

    if not selected:
        # Discovery found nothing: write an explicit report and fail.
        return write_report(discovery_status, [], missing)

    if args.cameras and missing:
        # Explicitly requested cameras are absent: report, don't crash.
        print(f"请求的相机未发现：{missing}", flush=True)
        return write_report(RUN_STATUS_TOPICS_NOT_FOUND, [], missing)

    camera_labels = [label for label in expected_labels if label in selected]

    # Compare-kind arity checks: report rather than crash, since missing
    # cameras are a runtime discovery result, not a CLI syntax error.
    if args.compare == "dual" and len(camera_labels) < 2:
        return write_report(
            RUN_STATUS_TOPICS_NOT_FOUND,
            [],
            missing,
            {"requested_compare": args.compare, "arity_error": f"dual 需要至少两路，当前 {len(camera_labels)} 路"},
        )
    if args.compare == "triple" and set(camera_labels) != set(KNOWN_CAMERAS):
        return write_report(
            RUN_STATUS_TOPICS_NOT_FOUND,
            [],
            missing,
            {"requested_compare": args.compare, "arity_error": f"triple 需要三路，当前 {len(camera_labels)} 路"},
        )
    if args.compare == "all" and len(camera_labels) < len(KNOWN_CAMERAS):
        print(f"注：三路相机不全（{len(camera_labels)}/3），跳过 triple 阶梯。", flush=True)

    modes = build_modes(args.compare, camera_labels)
    if not modes:
        return write_report(
            RUN_STATUS_TOPICS_NOT_FOUND,
            [],
            missing,
            {"requested_compare": args.compare, "arity_error": f"没有可执行的配置（{len(camera_labels)} 路）"},
        )

    # rclpy context ownership: only shut down what we started.  Import/init
    # failures must produce a report, not an uncaught traceback.
    try:
        import rclpy
    except Exception as exc:
        print(f"rclpy 导入失败：{exc!r}", flush=True)
        return write_report(
            RUN_STATUS_SAMPLING_ERROR,
            [],
            missing,
            {"init_error": repr(exc), "init_traceback": traceback.format_exc()},
        )

    initialized_here = False
    try:
        initialized_here = not rclpy.ok()
        if initialized_here:
            rclpy.init(args=None)
    except Exception as exc:
        print(f"rclpy 初始化失败：{exc!r}", flush=True)
        if rclpy.ok():
            with contextlib.suppress(Exception):
                rclpy.shutdown()
        return write_report(
            RUN_STATUS_SAMPLING_ERROR,
            [],
            missing,
            {"init_error": repr(exc), "init_traceback": traceback.format_exc()},
        )

    report_modes: list[dict[str, Any]] = []
    try:
        for round_index in range(args.rounds):
            for mode_index, labels in enumerate(modes):
                mode_topics = {label: selected[label] for label in labels}
                display = mode_name(labels) + (f" [r{round_index + 1}]" if args.rounds > 1 else "")
                print(f"测量配置：{display}（{', '.join(labels) or '无'}）…", flush=True)
                node_name = f"rabo_camera_throughput_diag_{os.getpid()}_{round_index}_{mode_index}"
                metrics, stats = sample_mode(mode_topics, args.duration, args.settle, node_name)
                report_modes.append(
                    {
                        "name": display,
                        "round": round_index + 1,
                        "camera_labels": labels,
                        "cameras": {
                            label: summarize_camera(metrics[label], args.fps, stats)
                            for label in labels
                        },
                        "process": summarize_process(stats),
                    }
                )
    finally:
        if initialized_here:
            with contextlib.suppress(Exception):
                rclpy.shutdown()

    # Classify per (mode, camera) cell, plus any default-expected camera that
    # was never discovered, and topic-info metadata failures.  Sampling facts
    # (subscribe / sampling / no-frames / missing) take priority over metadata
    # completeness, so a total metadata failure cannot mask a no-frames result.
    all_cells_have_frames = True
    any_cell_has_frames = False
    any_subscribe_error = False
    any_sampling_error = False
    no_frame_cells: list[str] = []
    for mode in report_modes:
        mode_sampling_error = mode["process"].get("sampling_error")
        if mode_sampling_error:
            any_sampling_error = True
        for label in mode["camera_labels"]:
            camera = mode["cameras"][label]
            if camera.get("frame_count", 0) > 0:
                any_cell_has_frames = True
            else:
                all_cells_have_frames = False
                # Only record a *genuine* no-frames cell when subscription
                # succeeded and this mode did not abort with a sampling error;
                # those cases are represented by their own failure reasons.
                if not camera.get("subscribe_error") and not mode_sampling_error:
                    no_frame_cells.append(f"{mode['name']}/{label}")
            if camera.get("subscribe_error"):
                any_subscribe_error = True

    # Only metadata for the cameras in the *tested* scope affects run_status;
    # extra discovery of other cameras is still reported but must not fail an
    # explicit --cameras subset run.
    info_failed = [
        label
        for label in camera_labels
        if info_verbose.get(label) is not None and not info_verbose[label].get("ok")
    ]

    failure_reasons: list[str] = []
    if any_subscribe_error:
        failure_reasons.append("subscribe_error")
    if any_sampling_error:
        failure_reasons.append("sampling_error")
    if no_frame_cells:
        failure_reasons.append(f"no_frames:{','.join(no_frame_cells)}")
    if missing:
        failure_reasons.append(f"missing:{','.join(missing)}")
    if info_failed:
        failure_reasons.append(f"metadata_failed:{','.join(info_failed)}")

    if any_subscribe_error:
        run_status = RUN_STATUS_SUBSCRIBE_FAILED
    elif any_sampling_error:
        run_status = RUN_STATUS_SAMPLING_ERROR
    elif not any_cell_has_frames:
        run_status = RUN_STATUS_NO_FRAMES
    elif not all_cells_have_frames or missing:
        # Some (mode, camera) cell got no frames, or a default-expected camera
        # was never discovered.
        run_status = RUN_STATUS_PARTIAL
    elif info_failed and len(info_failed) == len(camera_labels):
        # All frames present, but publisher QoS / matching could not be
        # inspected at all.
        run_status = RUN_STATUS_METADATA_FAILED
    elif info_failed:
        # All frames present, but some topic metadata failed to collect.
        run_status = RUN_STATUS_PARTIAL_METADATA
    else:
        run_status = RUN_STATUS_OK

    print(f"运行状态：{run_status}", flush=True)
    return write_report(run_status, report_modes, missing, {"failure_reasons": failure_reasons})


if __name__ == "__main__":
    raise SystemExit(main())
