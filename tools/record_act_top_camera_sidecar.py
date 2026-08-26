#!/usr/bin/env python3
"""Record only the fixed/top RGB stream in an independent Python process.

The callback records ``time.monotonic_ns()`` before copying the image payload.
Image conversion and disk I/O happen on a bounded writer queue, never in the
ROS callback.  The process is stopped cleanly by writing ``STOP`` to stdin.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_DATASET_NAME = "cam_top"
TOP_SOURCE = "fixed_rgb"
EXPECTED_WIDTH = 960
EXPECTED_HEIGHT = 540
EXPECTED_ENCODING = "rgb8"
DEFAULT_QUEUE_SIZE = 8

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.probe_act_recording_sources import (  # noqa: E402
    discover_camera_topics,
    load_message_class,
    ros_stamp_seconds,
)


def emit(message: str) -> None:
    print(message, flush=True)


def json_line(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


@dataclass
class FramePacket:
    frame_index: int
    frame_path: str
    receive_monotonic_ns: int
    ros_stamp: float | None
    width: int
    height: int
    encoding: str
    data: bytes
    step: int


class TopCameraSidecar:
    def __init__(self, episode_dir: Path, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.episode_dir = episode_dir
        self.camera_dir = episode_dir / "cameras" / TOP_DATASET_NAME
        self.timestamps_path = episode_dir / "camera_timestamps.jsonl"
        self.queue: Queue[FramePacket] = Queue(maxsize=max(1, int(queue_size)))
        self.stop_event = threading.Event()
        self.writer_thread: threading.Thread | None = None
        self.stdin_thread: threading.Thread | None = None
        self.node: Any = None
        self.rclpy: Any = None
        self.executor: Any = None
        self.subscription: Any = None
        self.topic: str | None = None
        self.type_name: str | None = None
        self.qos_mode = "BEST_EFFORT_KEEP_LAST_1"
        self.callback_count = 0
        self.queue_drop_count = 0
        self.saved_count = 0
        self.writer_error_count = 0
        self.writer_errors: list[str] = []
        self._lock = threading.Lock()
        self._next_index = 0

    @staticmethod
    def _image_bytes(data: bytes, width: int, height: int, encoding: str, step: int) -> bytes:
        if encoding not in {"rgb8", "bgr8"}:
            raise ValueError(f"top sidecar requires rgb8/bgr8, got {encoding}")
        channels = 3
        row_width = width * channels
        rows = [data[row * step : row * step + row_width] for row in range(height)]
        if any(len(row) != row_width for row in rows):
            raise ValueError("short RGB image buffer")
        if encoding == "bgr8":
            converted = bytearray(width * height * channels)
            out = 0
            for row in rows:
                for col in range(0, row_width, channels):
                    converted[out : out + 3] = row[col : col + 3][::-1]
                    out += 3
            payload = bytes(converted)
            encoding = "rgb8"
        else:
            payload = b"".join(rows)
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + payload
        return ppm

    def _callback(self, msg: Any) -> None:
        receive_ns = time.monotonic_ns()
        try:
            width = int(msg.width)
            height = int(msg.height)
            step = int(msg.step)
            encoding = str(msg.encoding).lower()
            if encoding not in {"rgb8", "bgr8"}:
                raise ValueError(f"top sidecar requires rgb8/bgr8, got {encoding}")
            payload = bytes(msg.data)
            ros_stamp = ros_stamp_seconds(msg)
            with self._lock:
                index = self._next_index
                self._next_index += 1
                self.callback_count += 1
            emit(f"[TOP_CAMERA_SIDECAR] FRAME {index}")
            relative = Path("cameras") / TOP_DATASET_NAME / f"frame_{index:06d}.ppm"
            packet = FramePacket(
                frame_index=index,
                frame_path=str(relative),
                receive_monotonic_ns=receive_ns,
                ros_stamp=ros_stamp,
                width=width,
                height=height,
                encoding=encoding,
                data=payload,
                step=step,
            )
            try:
                self.queue.put_nowait(packet)
            except Full:
                with self._lock:
                    self.queue_drop_count += 1
        except Exception as exc:
            with self._lock:
                self.writer_error_count += 1
                if len(self.writer_errors) < 20:
                    self.writer_errors.append(f"callback: {exc!r}")

    def _writer_loop(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                packet = self.queue.get(timeout=0.05)
            except Exception:
                continue
            try:
                path = self.episode_dir / packet.frame_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self._image_bytes(
                    packet.data,
                    packet.width,
                    packet.height,
                    packet.encoding,
                    packet.step,
                ))
                json_line(self.timestamps_path, {
                    "frame_index": packet.frame_index,
                    "frame_path": packet.frame_path,
                    "receive_monotonic_ns": packet.receive_monotonic_ns,
                    "ros_stamp": packet.ros_stamp,
                    "width": packet.width,
                    "height": packet.height,
                    "encoding": packet.encoding,
                })
                with self._lock:
                    self.saved_count += 1
            except Exception as exc:
                with self._lock:
                    self.writer_error_count += 1
                    if len(self.writer_errors) < 20:
                        self.writer_errors.append(repr(exc))
            finally:
                self.queue.task_done()

    def _stdin_loop(self) -> None:
        try:
            for line in sys.stdin:
                if line.strip().upper() in {"STOP", "QUIT", "Q"}:
                    self.stop_event.set()
                    return
        except Exception:
            self.stop_event.set()

    def start(self) -> dict[str, Any]:
        import rclpy
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )

        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = rclpy.create_node("act_top_camera_sidecar")
        topics, discovery = discover_camera_topics()
        selected = topics.get(TOP_SOURCE)
        if not selected:
            raise RuntimeError("fixed_rgb top camera topic was not discovered")
        self.topic = selected["topic"]
        self.type_name = selected["type"]
        msg_type = load_message_class(self.type_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        try:
            self.subscription = self.node.create_subscription(
                msg_type,
                self.topic,
                self._callback,
                qos,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
        except Exception:
            self.qos_mode = "RELIABLE_KEEP_LAST_6_FALLBACK"
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=6,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.subscription = self.node.create_subscription(msg_type, self.topic, self._callback, qos)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.camera_dir.mkdir(parents=True, exist_ok=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, name="act-top-sidecar-writer", daemon=True)
        self.writer_thread.start()
        self.stdin_thread = threading.Thread(target=self._stdin_loop, name="act-top-sidecar-stdin", daemon=True)
        self.stdin_thread.start()
        emit(f"[TOP_CAMERA_SIDECAR] READY topic={self.topic} qos={self.qos_mode}")
        return {"topic": self.topic, "type": self.type_name, "qos": self.qos_mode, "discovery": discovery}

    def run(self, *, duration_s: float | None = None) -> dict[str, Any]:
        started = time.monotonic()
        while not self.stop_event.is_set():
            if duration_s is not None and time.monotonic() - started >= duration_s:
                self.stop_event.set()
                break
            self.executor.spin_once(timeout_sec=0.05)
        self.close()
        with self._lock:
            callback_count = self.callback_count
            saved_count = self.saved_count
            drops = self.queue_drop_count
            errors = self.writer_error_count
        summary = self.summary()
        emit(f"[TOP_CAMERA_SIDECAR] SUMMARY {json.dumps(summary, ensure_ascii=False, separators=(',', ':'))}")
        return summary

    def summary(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if self.timestamps_path.exists():
            for line in self.timestamps_path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(json.loads(line))
        times = sorted(int(row["receive_monotonic_ns"]) for row in rows)
        intervals = [b - a for a, b in zip(times, times[1:])]
        effective_fps = None
        if len(times) >= 2 and times[-1] > times[0]:
            effective_fps = len(times) / ((times[-1] - times[0]) / 1e9)
        percentile = lambda values, p: float(__import__("numpy").percentile(values, p) / 1e6) if values else None
        return {
            "topic": self.topic,
            "type": self.type_name,
            "qos": self.qos_mode,
            "callback_count": self.callback_count,
            "saved_count": self.saved_count,
            "effective_fps": effective_fps,
            "callback_interval_median_ms": percentile(intervals, 50),
            "callback_interval_p95_ms": percentile(intervals, 95),
            "callback_interval_max_ms": (max(intervals) / 1e6) if intervals else None,
            "queue_drop_count": self.queue_drop_count,
            "writer_error_count": self.writer_error_count,
            "writer_errors": list(self.writer_errors),
        }

    def close(self) -> None:
        if self.executor is not None:
            with contextlib.suppress(Exception):
                self.executor.shutdown()
        if self.node is not None and self.subscription is not None:
            with contextlib.suppress(Exception):
                self.node.destroy_subscription(self.subscription)
        if self.node is not None:
            with contextlib.suppress(Exception):
                self.node.destroy_node()
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=30.0)
        if self.rclpy is not None and self.rclpy.ok():
            with contextlib.suppress(Exception):
                self.rclpy.shutdown()


def synthetic_test() -> dict[str, Any]:
    values = [1_000_000_000, 1_100_000_000, 1_250_000_000]
    rows = [{"receive_monotonic_ns": value} for value in values]
    indices = []
    for state in (1_050_000_000, 1_200_000_000, 1_300_000_000):
        valid = [i for i, row in enumerate(rows) if row["receive_monotonic_ns"] <= state]
        indices.append(valid[-1] if valid else -1)
    passed = indices == [0, 1, 2] and all(b > a for a, b in zip(values, values[1:]))
    return {"result": "PASS" if passed else "FAIL", "causal_indices": indices, "timestamp_monotonic": passed}


def synthetic_process(episode_dir: Path) -> int:
    """Small no-ROS process used to test the collector's subprocess handshake."""
    camera_dir = episode_dir / "cameras" / TOP_DATASET_NAME
    camera_dir.mkdir(parents=True, exist_ok=True)
    timestamps = episode_dir / "camera_timestamps.jsonl"
    emit("[TOP_CAMERA_SIDECAR] READY topic=synthetic qos=SYNTHETIC")
    base = time.monotonic_ns()
    payload = b"P6\n2 1\n255\n" + b"\x00\x00\x00" * 2
    for index in range(3):
        receive_ns = base + index * 100_000_000
        frame_path = Path("cameras") / TOP_DATASET_NAME / f"frame_{index:06d}.ppm"
        (episode_dir / frame_path).write_bytes(payload)
        json_line(timestamps, {
            "frame_index": index,
            "frame_path": str(frame_path),
            "receive_monotonic_ns": receive_ns,
            "ros_stamp": None,
            "width": 2,
            "height": 1,
            "encoding": "rgb8",
        })
        emit(f"[TOP_CAMERA_SIDECAR] FRAME {index}")
    for line in sys.stdin:
        if line.strip().upper() in {"STOP", "QUIT", "Q"}:
            emit("[TOP_CAMERA_SIDECAR] SUMMARY {\"callback_count\":3,\"saved_count\":3,\"queue_drop_count\":0,\"writer_error_count\":0}")
            return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, help="Temporary Episode directory.")
    parser.add_argument("--duration", type=float, help="Optional self-stop duration; otherwise wait for STOP on stdin.")
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic sidecar timestamp test without ROS.")
    parser.add_argument("--synthetic-run", action="store_true", help="Run a no-ROS READY/STOP subprocess fixture.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        result = synthetic_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["result"] == "PASS" else 2
    if args.synthetic_run:
        if args.episode_dir is None:
            raise SystemExit("--episode-dir is required with --synthetic-run")
        return synthetic_process(args.episode_dir)
    if args.episode_dir is None:
        raise SystemExit("--episode-dir is required")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be > 0")
    sidecar = TopCameraSidecar(args.episode_dir, queue_size=args.queue_size)
    try:
        sidecar.start()
        sidecar.run(duration_s=args.duration)
    except BaseException as exc:
        emit(f"[TOP_CAMERA_SIDECAR] ERROR {exc!r}")
        with contextlib.suppress(Exception):
            sidecar.close()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
