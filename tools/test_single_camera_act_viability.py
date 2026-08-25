#!/usr/bin/env python3
"""Measure top-only RGB capture and causal alignment to a 5 Hz ACT timeline.

This test never creates robot SDK clients.  It reuses the raw ACT recorder's
ROS discovery, subscription, decode and bounded writer queue, while adding
queue-depth and writer-lag instrumentation for one fixed top camera only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "single_camera_act_viability"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.probe_act_recording_sources import discover_camera_topics  # noqa: E402
from tools.record_act_episode import CameraStats, RawCameraRecorder  # noqa: E402


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def percentile(values: np.ndarray, q: float) -> float | None:
    return None if not len(values) else float(np.percentile(values, q))


def distribution(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float | None]:
    return {
        "mean": None if not len(values) else float(np.mean(values) * scale),
        "median": percentile(values, 50) * scale if len(values) else None,
        "p90": percentile(values, 90) * scale if len(values) else None,
        "p95": percentile(values, 95) * scale if len(values) else None,
        "max": None if not len(values) else float(np.max(values) * scale),
    }


class InstrumentedTopRecorder(RawCameraRecorder):
    """RawCameraRecorder with passive queue-depth/writer-lag measurements."""

    def __init__(self, topics: dict[str, dict[str, str]], episode_dir: Path, *, queue_size: int) -> None:
        super().__init__(topics, episode_dir, queue_size=queue_size)
        self.max_queue_depth = 0
        self.writer_start_lags_s: list[float] = []
        self.writer_complete_lags_s: list[float] = []

    def _make_callback(self, record: CameraStats) -> Callable[[Any], None]:
        base_callback = super()._make_callback(record)

        def callback(msg: Any) -> None:
            base_callback(msg)
            with self._lock:
                self.max_queue_depth = max(self.max_queue_depth, self._queue.qsize())

        return callback

    def _writer_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                packet = self._queue.get(timeout=0.05)
            except Empty:
                continue
            record = self.records[packet.source_label]
            writer_started = time.monotonic() - self._start_monotonic
            try:
                image_bytes, suffix = self._packet_to_image_bytes(packet)
                with self._lock:
                    saved_index = len(record.saved)
                relative_path = Path("cameras") / record.dataset_name / f"frame_{saved_index:06d}{suffix}"
                (self.episode_dir / relative_path).write_bytes(image_bytes)
                writer_finished = time.monotonic() - self._start_monotonic
                start_lag = max(0.0, writer_started - packet.arrival_time_s)
                complete_lag = max(0.0, writer_finished - packet.arrival_time_s)
                with self._lock:
                    self.writer_start_lags_s.append(start_lag)
                    self.writer_complete_lags_s.append(complete_lag)
                    record.saved.append({
                        "path": str(relative_path),
                        "arrival_time_s": packet.arrival_time_s,
                        "ros_timestamp": packet.ros_timestamp,
                        "width": packet.width,
                        "height": packet.height,
                        "encoding": packet.encoding,
                        "writer_start_lag_s": start_lag,
                        "writer_complete_lag_s": complete_lag,
                    })
            except Exception as exc:
                with self._lock:
                    record.write_error_count += 1
                    if len(record.write_errors) < 20:
                        record.write_errors.append(repr(exc))
            finally:
                self._queue.task_done()


def causal_alignment(frame_times: np.ndarray, duration_s: float, dataset_fps: float) -> dict[str, Any]:
    interval = 1.0 / dataset_fps
    tick_times = np.arange(0.0, duration_s + 1e-12, interval, dtype=np.float64)
    selected = np.searchsorted(frame_times, tick_times, side="right") - 1
    valid = selected >= 0
    selected_times = np.full(len(tick_times), np.nan, dtype=np.float64)
    selected_times[valid] = frame_times[selected[valid]]
    ages = tick_times - selected_times
    valid_ages = ages[valid]
    ticks_with_camera = int(np.count_nonzero(valid))
    unique_used = int(len(np.unique(selected[valid]))) if ticks_with_camera else 0
    thresholds_ms = (100, 200, 300, 400, 500)
    freshness_all = {
        f"le_{threshold}_ms": float(np.count_nonzero(valid & (ages <= threshold / 1000.0)) / len(tick_times))
        if len(tick_times) else 0.0
        for threshold in thresholds_ms
    }
    freshness_available = {
        f"le_{threshold}_ms": float(np.count_nonzero(valid_ages <= threshold / 1000.0) / ticks_with_camera)
        if ticks_with_camera else 0.0
        for threshold in thresholds_ms
    }
    return {
        "tick_times": tick_times,
        "selected_indices": selected.astype(np.int64),
        "selected_times": selected_times,
        "ages": ages,
        "valid": valid,
        "report": {
            "selection_policy": "LATEST_PREVIOUS_FRAME",
            "future_frames_allowed": False,
            "tick_count": int(len(tick_times)),
            "ticks_with_camera": ticks_with_camera,
            "causal_coverage_fraction": ticks_with_camera / len(tick_times) if len(tick_times) else 0.0,
            "unique_camera_frames_used": unique_used,
            "camera_frame_reuse_fraction": 1.0 - unique_used / ticks_with_camera if ticks_with_camera else None,
            "first_tick_status": "CAUSAL_FRAME_AVAILABLE" if len(valid) and valid[0] else "NO_CAUSAL_FRAME",
            "frame_age_ms": distribution(valid_ages, scale=1000.0),
            "freshness_fraction_of_all_ticks": freshness_all,
            "freshness_fraction_of_ticks_with_camera": freshness_available,
        },
    }


def assess(report: dict[str, Any]) -> tuple[str, list[str]]:
    alignment = report["causal_alignment"]
    inter = report["inter_frame_dt_s"]
    queue = report["queue"]
    p95_age = alignment["frame_age_ms"]["p95"]
    max_gap = inter["max"]
    coverage = alignment["causal_coverage_fraction"]
    reuse = alignment["camera_frame_reuse_fraction"]
    no_writer_loss = (
        queue["dropped_frames"] == 0
        and report["frames_received"] == report["frames_written"]
        and queue["write_errors"] == 0
    )
    reasons = [
        f"causal_coverage={coverage:.3f}", f"p95_frame_age_ms={p95_age}",
        f"max_inter_frame_gap_s={max_gap}", f"reuse_fraction={reuse}",
        f"writer_loss_free={no_writer_loss}",
    ]
    if (
        no_writer_loss and coverage >= 0.95 and p95_age is not None and p95_age <= 200.0
        and max_gap is not None and max_gap <= 0.5 and reuse is not None and reuse <= 0.20
    ):
        return "GOOD", reasons
    if (
        coverage >= 0.90 and p95_age is not None and p95_age <= 500.0
        and max_gap is not None and max_gap <= 1.0
    ):
        return "MARGINAL", reasons
    return "NOT_USABLE", reasons


def run(args: argparse.Namespace) -> int:
    if args.plan_only:
        print(json.dumps({
            "status": "PLAN_ONLY_NO_ROS_SUBSCRIPTION",
            "camera": "top/fixed_rgb", "robot_commands": False,
            "raw_resize": False, "dataset_fps": args.dataset_fps,
            "selection_policy": "LATEST_PREVIOUS_FRAME",
        }, indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    selected, discovery = discover_camera_topics()
    top_topics = {"fixed_rgb": selected["fixed_rgb"]} if "fixed_rgb" in selected else {}
    recorder = InstrumentedTopRecorder(top_topics, output_dir, queue_size=args.queue_size)
    started_at = now_iso()
    recorder.start()
    start_monotonic = time.monotonic()
    recorder.begin(start_monotonic)
    interrupted = False
    try:
        deadline = start_monotonic + args.duration
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
    actual_duration = time.monotonic() - start_monotonic
    recorder.end()
    recorder.close()
    camera_metadata = recorder.write_metadata()
    record = recorder.records.get("fixed_rgb")
    saved = sorted(record.saved, key=lambda row: row["arrival_time_s"]) if record else []
    frame_times = np.asarray([row["arrival_time_s"] for row in saved], dtype=np.float64)
    ros_times = np.asarray([
        row["ros_timestamp"] if row["ros_timestamp"] is not None else np.nan for row in saved
    ], dtype=np.float64)
    inter = np.diff(frame_times)
    alignment = causal_alignment(frame_times, actual_duration, args.dataset_fps)
    np.save(output_dir / "camera_timestamps.npy", frame_times)
    np.savez_compressed(
        output_dir / "dataset_tick_alignment.npz",
        tick_times=alignment["tick_times"],
        selected_camera_indices=alignment["selected_indices"],
        selected_camera_times=alignment["selected_times"],
        frame_ages=alignment["ages"],
        no_causal_frame=~alignment["valid"],
    )
    finite_ros = ros_times[np.isfinite(ros_times)]
    frames_received = record.callback_count if record else 0
    frames_written = len(saved)
    report: dict[str, Any] = {
        "status": "INTERRUPTED" if interrupted else "COMPLETED",
        "started_at": started_at, "finished_at": now_iso(), "output_dir": str(output_dir),
        "camera": "top", "source_label": "fixed_rgb", "topic": record.topic if record else None,
        "type": record.type_name if record else None,
        "frames_received": frames_received, "frames_written": frames_written,
        "duration_s": actual_duration,
        "effective_camera_fps": (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        if len(frame_times) >= 2 and frame_times[-1] > frame_times[0] else None,
        "window_write_fps": frames_written / actual_duration if actual_duration > 0 else None,
        "inter_frame_dt_s": distribution(inter),
        "queue": {
            "configured_capacity": args.queue_size,
            "max_queue_depth": recorder.max_queue_depth,
            "dropped_frames": record.queue_drop_count if record else 0,
            "write_errors": record.write_error_count if record else 0,
            "writer_lag_ms": distribution(np.asarray(recorder.writer_complete_lags_s), scale=1000.0),
            "writer_start_lag_ms": distribution(np.asarray(recorder.writer_start_lags_s), scale=1000.0),
            "shutdown_error": recorder.writer_shutdown_error,
        },
        "ros_timestamp": {
            "missing_count": int(np.count_nonzero(~np.isfinite(ros_times))),
            "all_present": bool(len(ros_times)) and bool(np.isfinite(ros_times).all()),
            "monotonic_non_decreasing": bool(len(finite_ros)) and bool(np.all(np.diff(finite_ros) >= 0)),
            "strictly_increasing": bool(len(finite_ros)) and bool(np.all(np.diff(finite_ros) > 0)),
        },
        "observed_frames": {
            "widths": sorted({row["width"] for row in saved if row["width"] is not None}),
            "heights": sorted({row["height"] for row in saved if row["height"] is not None}),
            "encodings": sorted({row["encoding"] for row in saved if row["encoding"] is not None}),
            "online_resize_performed": False,
            "expected_raw_resolution": [960, 540],
        },
        "causal_alignment": alignment["report"],
        "camera_init_error": recorder.init_error,
        "camera_metadata": camera_metadata,
        "discovery": discovery,
    }
    assessment, reasons = assess(report)
    report["TOP_CAMERA_5HZ_ASSESSMENT"] = assessment
    report["assessment_reasons"] = reasons
    report["assessment_policy"] = {
        "GOOD": "loss-free writer; coverage>=95%; p95 age<=200ms; max gap<=0.5s; reuse<=20%",
        "MARGINAL": "coverage>=90%; p95 age<=500ms; max gap<=1s",
        "NOT_USABLE": "does not satisfy the multi-metric MARGINAL boundary",
        "note": "not based on average FPS alone",
    }
    metadata = {
        "created_at": now_iso(), "git_commit": git_commit(),
        "duration_requested_s": args.duration, "dataset_fps_requested": args.dataset_fps,
        "camera_requested": "top/fixed_rgb", "camera_count": 1,
        "robot_sdk_clients_created": False, "robot_commands_sent": False,
        "frame_storage": "raw source resolution; no 224 resize",
        "timeline_clock": "local monotonic arrival time",
        "selection_policy": "latest frame with arrival_time <= tick_time; never nearest/future",
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if frames_written > 0 and recorder.init_error is None else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("top",), default="top")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dataset-fps", type=float, default=5.0)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-only", action="store_true", help="print a safe plan without ROS discovery/subscription")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive")
    if not math.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        parser.error("--dataset-fps must be positive")
    if args.queue_size <= 0:
        parser.error("--queue-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
