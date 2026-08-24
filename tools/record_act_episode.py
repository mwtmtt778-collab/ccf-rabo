#!/usr/bin/env python3
"""Record one raw Rabo ACT episode without commanding the robot.

This recorder is intentionally separate from the Expert process.  It samples
the 26D Rabo state at a fixed rate while subscribing to all three asynchronous
RGB streams.  Run the desired Expert motion in another Rabo terminal during
the recording window.

The output is a transport-oriented raw episode, not a final LeRobotDataset.
Conversion to LeRobotDataset v3 happens locally in ``ccf_act_baseline``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tarfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "act_samples"
CAMERA_QOS_DEPTH = 6
CAMERA_NAMES = {
    "fixed_rgb": "cam_top",
    "left_wrist_rgb": "cam_left_wrist",
    "right_wrist_rgb": "cam_right_wrist",
}
ACTION_NAMES = [
    *(f"left_arm.j{i}" for i in range(1, 8)),
    *(f"right_arm.j{i}" for i in range(1, 8)),
    "left_hand.thumb_rotation",
    "left_hand.thumb_bend",
    "left_hand.index",
    "left_hand.middle",
    "left_hand.ring",
    "left_hand.pinky",
    "right_hand.thumb_rotation",
    "right_hand.thumb_bend",
    "right_hand.index",
    "right_hand.middle",
    "right_hand.ring",
    "right_hand.pinky",
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.probe_act_recording_sources import (  # noqa: E402
    discover_camera_topics,
    init_devices,
    load_message_class,
    load_robot_ids,
    read_state26_timed,
    ros_stamp_seconds,
    shutdown_devices,
)


def now_iso() -> str:
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


@dataclass(frozen=True)
class FramePacket:
    source_label: str
    dataset_name: str
    type_name: str
    arrival_time_s: float
    ros_timestamp: float | None
    width: int | None
    height: int | None
    step: int | None
    encoding: str | None
    compressed_format: str | None
    data: bytes


@dataclass
class CameraStats:
    source_label: str
    dataset_name: str
    topic: str
    type_name: str
    callback_count: int = 0
    unique_count: int = 0
    duplicate_stamp_count: int = 0
    queue_drop_count: int = 0
    write_error_count: int = 0
    subscribe_error: str | None = None
    last_ros_stamp: float | None = None
    saved: list[dict[str, Any]] = field(default_factory=list)
    write_errors: list[str] = field(default_factory=list)


class RawCameraRecorder:
    def __init__(
        self,
        topics: dict[str, dict[str, str]],
        episode_dir: Path,
        *,
        queue_size: int,
    ) -> None:
        self.episode_dir = episode_dir
        self.records = {
            source_label: CameraStats(
                source_label=source_label,
                dataset_name=CAMERA_NAMES[source_label],
                topic=info["topic"],
                type_name=info["type"],
            )
            for source_label, info in topics.items()
            if source_label in CAMERA_NAMES
        }
        for record in self.records.values():
            (episode_dir / "cameras" / record.dataset_name).mkdir(parents=True, exist_ok=True)

        self._queue: Queue[FramePacket] = Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._active = False
        self._start_monotonic = 0.0
        self._stop = threading.Event()
        self._spin_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._node: Any = None
        self._executor: Any = None
        self._rclpy: Any = None
        self._owns_rclpy_context = False
        self._callback_groups: list[Any] = []
        self._subscriptions: list[Any] = []
        self.init_error: str | None = None
        self.writer_shutdown_error: str | None = None

    def start(self) -> None:
        if not self.records:
            self.init_error = "no known RGB camera topics"
            return
        try:
            import rclpy
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

            self._rclpy = rclpy
            self._owns_rclpy_context = not rclpy.ok()
            if self._owns_rclpy_context:
                rclpy.init(args=None)
            self._node = rclpy.create_node("rabo_act_raw_episode_recorder")
            self._executor = MultiThreadedExecutor(num_threads=max(2, len(self.records)))
            self._executor.add_node(self._node)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=CAMERA_QOS_DEPTH,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            for record in self.records.values():
                try:
                    callback_group = MutuallyExclusiveCallbackGroup()
                    subscription = self._node.create_subscription(
                        load_message_class(record.type_name),
                        record.topic,
                        self._make_callback(record),
                        qos,
                        callback_group=callback_group,
                    )
                    self._callback_groups.append(callback_group)
                    self._subscriptions.append(subscription)
                except Exception as exc:
                    record.subscribe_error = repr(exc)

            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="act-raw-jpeg-writer",
                daemon=True,
            )
            self._spin_thread = threading.Thread(
                target=self._spin_loop,
                name="act-raw-camera-spin",
                daemon=True,
            )
            self._writer_thread.start()
            self._spin_thread.start()
        except Exception as exc:
            self.init_error = repr(exc)

    def begin(self, start_monotonic: float) -> None:
        with self._lock:
            self._start_monotonic = start_monotonic
            self._active = True

    def end(self) -> None:
        with self._lock:
            self._active = False

    def _make_callback(self, record: CameraStats) -> Callable[[Any], None]:
        def callback(msg: Any) -> None:
            arrival = time.monotonic()
            stamp = ros_stamp_seconds(msg)
            with self._lock:
                if not self._active:
                    return
                record.callback_count += 1
                if stamp is not None and stamp == record.last_ros_stamp:
                    record.duplicate_stamp_count += 1
                    return
                record.last_ros_stamp = stamp
                record.unique_count += 1
                relative_arrival = arrival - self._start_monotonic

            packet = FramePacket(
                source_label=record.source_label,
                dataset_name=record.dataset_name,
                type_name=record.type_name,
                arrival_time_s=relative_arrival,
                ros_timestamp=stamp,
                width=int(msg.width) if hasattr(msg, "width") else None,
                height=int(msg.height) if hasattr(msg, "height") else None,
                step=int(msg.step) if hasattr(msg, "step") else None,
                encoding=str(msg.encoding) if hasattr(msg, "encoding") else None,
                compressed_format=str(msg.format) if hasattr(msg, "format") else None,
                data=bytes(msg.data),
            )
            try:
                self._queue.put_nowait(packet)
            except Full:
                with self._lock:
                    record.queue_drop_count += 1

        return callback

    @staticmethod
    def _packet_to_image_bytes(packet: FramePacket) -> tuple[bytes, str]:
        if packet.type_name == "sensor_msgs/msg/CompressedImage":
            image_format = (packet.compressed_format or "").lower()
            if "jpeg" in image_format or "jpg" in image_format:
                return packet.data, ".jpg"
            if "png" in image_format:
                return packet.data, ".png"
            raise ValueError(f"unsupported compressed image format: {packet.compressed_format}")

        if packet.width is None or packet.height is None or packet.encoding is None:
            raise ValueError("raw Image is missing width/height/encoding")
        encoding = packet.encoding.lower()
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "8uc1": 1,
        }
        if encoding not in channels_by_encoding:
            raise ValueError(f"unsupported image encoding: {packet.encoding}")
        channels = channels_by_encoding[encoding]
        stride = packet.step or packet.width * channels
        expected = stride * packet.height
        if len(packet.data) < expected:
            raise ValueError(f"short image buffer: {len(packet.data)} < {expected}")
        rows = np.frombuffer(packet.data, dtype=np.uint8, count=expected).reshape(packet.height, stride)
        pixels = rows[:, : packet.width * channels].reshape(packet.height, packet.width, channels)
        if encoding in {"mono8", "8uc1"}:
            payload = np.ascontiguousarray(pixels[:, :, 0]).tobytes()
            return f"P5\n{packet.width} {packet.height}\n255\n".encode("ascii") + payload, ".pgm"
        if encoding in {"bgr8", "bgra8"}:
            pixels = pixels[:, :, [2, 1, 0, 3] if channels == 4 else [2, 1, 0]]
        if channels == 4:
            pixels = pixels[:, :, :3]
        payload = np.ascontiguousarray(pixels).tobytes()
        return f"P6\n{packet.width} {packet.height}\n255\n".encode("ascii") + payload, ".ppm"

    def _writer_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                packet = self._queue.get(timeout=0.05)
            except Empty:
                continue
            record = self.records[packet.source_label]
            try:
                image_bytes, suffix = self._packet_to_image_bytes(packet)
                with self._lock:
                    saved_index = len(record.saved)
                relative_path = Path("cameras") / record.dataset_name / f"frame_{saved_index:06d}{suffix}"
                output_path = self.episode_dir / relative_path
                output_path.write_bytes(image_bytes)
                with self._lock:
                    record.saved.append(
                        {
                            "path": str(relative_path),
                            "arrival_time_s": packet.arrival_time_s,
                            "ros_timestamp": packet.ros_timestamp,
                            "width": packet.width,
                            "height": packet.height,
                            "encoding": packet.encoding,
                        }
                    )
            except Exception as exc:
                with self._lock:
                    record.write_error_count += 1
                    if len(record.write_errors) < 20:
                        record.write_errors.append(repr(exc))
            finally:
                self._queue.task_done()

    def _spin_loop(self) -> None:
        assert self._executor is not None
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._executor.spin_once(timeout_sec=0.05)

    def close(self) -> None:
        self.end()
        self._stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=30.0)
            if self._writer_thread.is_alive():
                self.writer_shutdown_error = "image writer did not drain within 30 seconds"
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown(timeout_sec=1.0)
        if self._node is not None:
            for subscription in self._subscriptions:
                with contextlib.suppress(Exception):
                    self._node.destroy_subscription(subscription)
            if self._executor is not None:
                with contextlib.suppress(Exception):
                    self._executor.remove_node(self._node)
            with contextlib.suppress(Exception):
                self._node.destroy_node()
        if self._rclpy is not None and self._owns_rclpy_context and self._rclpy.ok():
            with contextlib.suppress(Exception):
                self._rclpy.shutdown()

    def write_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for record in self.records.values():
            saved = sorted(record.saved, key=lambda item: item["arrival_time_s"])
            camera_dir = self.episode_dir / "cameras" / record.dataset_name
            np.save(
                camera_dir / "ros_timestamps.npy",
                np.asarray(
                    [item["ros_timestamp"] if item["ros_timestamp"] is not None else np.nan for item in saved],
                    dtype=np.float64,
                ),
            )
            np.save(
                camera_dir / "arrival_times_s.npy",
                np.asarray([item["arrival_time_s"] for item in saved], dtype=np.float64),
            )
            (camera_dir / "frames.json").write_text(
                json.dumps(jsonable(saved), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result[record.dataset_name] = {
                "source_label": record.source_label,
                "topic": record.topic,
                "type": record.type_name,
                "callback_count": record.callback_count,
                "unique_count": record.unique_count,
                "saved_count": len(saved),
                "duplicate_stamp_count": record.duplicate_stamp_count,
                "queue_drop_count": record.queue_drop_count,
                "write_error_count": record.write_error_count,
                "subscribe_error": record.subscribe_error,
                "write_errors": record.write_errors,
            }
        return result


def sample_states(
    devices: dict[str, Any],
    *,
    start_monotonic: float,
    duration_s: float,
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, str]:
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    overrun_periods = 0
    status = "COMPLETED"
    interval = 1.0 / fps
    deadline = start_monotonic
    end_time = start_monotonic + duration_s
    try:
        while time.monotonic() < end_time:
            now = time.monotonic()
            if now < deadline:
                time.sleep(min(deadline - now, 0.005))
                continue
            read_start = time.monotonic()
            try:
                state, components, component_durations = read_state26_timed(devices)
                read_end = time.monotonic()
                samples.append(
                    {
                        "time_s": read_end - start_monotonic,
                        "read_duration_s": read_end - read_start,
                        "state": state,
                        "components": components,
                        "component_read_duration_s": component_durations,
                    }
                )
            except Exception as exc:
                read_end = time.monotonic()
                errors.append(
                    {
                        "time_s": read_end - start_monotonic,
                        "read_duration_s": read_end - read_start,
                        "error": repr(exc),
                    }
                )
            deadline += interval
            if read_end > deadline:
                missed = max(1, int((read_end - deadline) / interval) + 1)
                overrun_periods += missed
                deadline += missed * interval
    except KeyboardInterrupt:
        status = "INTERRUPTED"
    return samples, errors, overrun_periods, status


def write_telemetry(episode_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    source_states = np.asarray([sample["state"] for sample in samples], dtype=np.float32)
    source_timestamps = np.asarray([sample["time_s"] for sample in samples], dtype=np.float64)
    source_read_durations = np.asarray([sample["read_duration_s"] for sample in samples], dtype=np.float64)
    if len(source_states) >= 2:
        qpos = source_states[:-1]
        actions = source_states[1:]
        timestamps = source_timestamps[:-1]
        read_durations = source_read_durations[:-1]
    else:
        qpos = np.empty((0, 26), dtype=np.float32)
        actions = np.empty((0, 26), dtype=np.float32)
        timestamps = np.empty((0,), dtype=np.float64)
        read_durations = np.empty((0,), dtype=np.float64)
    np.savez_compressed(
        episode_dir / "telemetry.npz",
        qpos=qpos,
        actions=actions,
        timestamps=timestamps,
        read_durations=read_durations,
        source_states=source_states,
        source_timestamps=source_timestamps,
        source_read_durations=source_read_durations,
    )
    return {
        "source_state_count": len(source_states),
        "transition_count": len(qpos),
        "qpos_shape": list(qpos.shape),
        "actions_shape": list(actions.shape),
        "timestamps_shape": list(timestamps.shape),
        "finite": bool(
            np.isfinite(qpos).all()
            and np.isfinite(actions).all()
            and np.isfinite(timestamps).all()
        ),
        "timestamps_strictly_increasing": bool(len(timestamps) < 2 or np.all(np.diff(timestamps) > 0)),
    }


def create_archive(episode_dir: Path) -> Path:
    archive_path = episode_dir.with_suffix(".tar.gz")
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive_path}")
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        archive.add(episode_dir, arcname=episode_dir.name, recursive=True)
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record one 5Hz raw ACT episode while an Expert runs in another Rabo terminal."
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Recording window in seconds; default 10.")
    parser.add_argument("--fps", type=float, default=5.0, help="State/action timeline frequency; default 5Hz.")
    parser.add_argument("--start-delay", type=float, default=5.0, help="Countdown before recording; default 5s.")
    parser.add_argument("--episode-id", help="Unique episode directory name; default timestamp-based.")
    parser.add_argument("--task", default="rabo_act_pipeline_smoke_test", help="Task label stored in metadata.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--queue-size", type=int, default=96, help="Bounded image writer queue size.")
    parser.add_argument("--no-archive", action="store_true", help="Keep the folder but do not make tar.gz.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration <= 0 or args.fps <= 0 or args.start_delay < 0:
        raise SystemExit("duration/fps must be > 0 and start-delay must be >= 0")
    if args.queue_size <= 0:
        raise SystemExit("queue-size must be > 0")

    episode_id = args.episode_id or f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    episode_dir = output_root / episode_id
    if episode_dir.exists():
        raise SystemExit(f"refusing to overwrite existing episode: {episode_dir}")
    episode_dir.mkdir(parents=True)
    (PROJECT_ROOT / "logs" / "ros").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))

    print("Rabo ACT raw episode recorder", flush=True)
    print("READ-ONLY: this process never calls robot motion, hand control, or scene reset APIs.", flush=True)
    print(f"Episode directory: {episode_dir}", flush=True)

    started_at = now_iso()
    camera_topics, discovery = discover_camera_topics()
    camera_recorder = RawCameraRecorder(
        camera_topics,
        episode_dir,
        queue_size=args.queue_size,
    )
    devices: dict[str, Any] = {}
    state_samples: list[dict[str, Any]] = []
    state_errors: list[dict[str, Any]] = []
    shutdown_errors: list[str] = []
    overrun_periods = 0
    task_status = "NOT_STARTED"
    actual_duration_s = 0.0

    try:
        devices = init_devices(load_robot_ids())
        initial_state, _, _ = read_state26_timed(devices)
        if len(initial_state) != 26:
            raise RuntimeError(f"expected initial state dimension 26, got {len(initial_state)}")
        camera_recorder.start()
        print(f"Discovered cameras: {len(camera_recorder.records)}/3", flush=True)
        if camera_recorder.init_error:
            print(f"Camera initialization warning: {camera_recorder.init_error}", flush=True)
        print("Run the Expert in another Rabo terminal during the recording window.", flush=True)
        countdown_deadline = time.monotonic() + args.start_delay
        last_announced: int | None = None
        while True:
            remaining_s = countdown_deadline - time.monotonic()
            if remaining_s <= 0:
                break
            shown = int(math.ceil(remaining_s))
            if shown != last_announced:
                print(f"Recording starts in {shown}s...", flush=True)
                last_announced = shown
            time.sleep(min(0.1, remaining_s))

        start_monotonic = time.monotonic()
        camera_recorder.begin(start_monotonic)
        print(f"RECORDING: {args.duration:g}s at state/action timeline {args.fps:g}Hz", flush=True)
        state_samples, state_errors, overrun_periods, task_status = sample_states(
            devices,
            start_monotonic=start_monotonic,
            duration_s=args.duration,
            fps=args.fps,
        )
        actual_duration_s = time.monotonic() - start_monotonic
    except KeyboardInterrupt:
        task_status = "INTERRUPTED"
    except Exception as exc:
        task_status = "ERROR"
        state_errors.append({"time_s": None, "error": repr(exc), "stage": "MAIN"})
        print(f"Recorder error: {repr(exc)}", flush=True)
    finally:
        camera_recorder.close()
        shutdown_errors = shutdown_devices(devices)

    telemetry = write_telemetry(episode_dir, state_samples)
    cameras = camera_recorder.write_metadata()
    expected_cameras = set(CAMERA_NAMES.values())
    saved_cameras = {name for name, item in cameras.items() if item["saved_count"] > 0}
    checks = {
        "all_three_cameras_discovered": set(cameras) == expected_cameras,
        "all_three_cameras_saved": saved_cameras == expected_cameras,
        "at_least_two_transitions": telemetry["transition_count"] >= 2,
        "finite_26d_telemetry": telemetry["finite"],
        "timestamps_strictly_increasing": telemetry["timestamps_strictly_increasing"],
        "no_state_read_errors": not state_errors,
        "camera_writer_drained": camera_recorder.writer_shutdown_error is None,
    }
    result = "READY_FOR_LOCAL_CONVERSION" if all(checks.values()) else "CHECK"
    metadata = {
        "format": "rabo_raw_act_episode_v1",
        "purpose": "PIPELINE_SMOKE_TEST",
        "result": result,
        "episode_id": episode_id,
        "task": args.task,
        "started_at": started_at,
        "finished_at": now_iso(),
        "requested_fps": args.fps,
        "requested_duration_s": args.duration,
        "actual_duration_s": actual_duration_s,
        "action_source": "next_state_proxy",
        "deployable_quality": False,
        "state_action_schema": {
            "name": "rabo_a7_o6_clench_v1",
            "dimension": 26,
            "names": ACTION_NAMES,
            "groups": {
                "left_arm": [0, 7],
                "right_arm": [7, 14],
                "left_hand_clench": [14, 20],
                "right_hand_clench": [20, 26],
            },
            "units": {"arms": "radian", "hands": "normalized_0_1"},
        },
        "camera_name_mapping": CAMERA_NAMES,
        "camera_discovery": discovery,
        "telemetry": telemetry,
        "cameras": cameras,
        "checks": checks,
        "task_status": task_status,
        "state_read_errors": state_errors,
        "state_overrun_periods": overrun_periods,
        "camera_init_error": camera_recorder.init_error,
        "camera_writer_shutdown_error": camera_recorder.writer_shutdown_error,
        "shutdown_errors": shutdown_errors,
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(jsonable(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (episode_dir / "quality_report.json").write_text(
        json.dumps(
            jsonable({"result": result, "checks": checks, "telemetry": telemetry, "cameras": cameras}),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    archive_path: Path | None = None
    if not args.no_archive:
        print("Creating transport archive...", flush=True)
        archive_path = create_archive(episode_dir)

    print(f"Result: {result}", flush=True)
    print(f"Episode: {episode_dir}", flush=True)
    if archive_path is not None:
        print(f"Download this file: {archive_path}", flush=True)
    return 0 if result == "READY_FOR_LOCAL_CONVERSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
