#!/usr/bin/env python3
"""Run one ACT inference from live Rabo state and three RGB cameras.

This program is deliberately read-only: it contains no arm or hand command
calls.  A stale or unsafe prediction is still reported for diagnosis, but is
never sent to the robot.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_INPUTS = {
    "fixed_rgb": "cam_top",
    "left_wrist_rgb": "cam_left_wrist",
    "right_wrist_rgb": "cam_right_wrist",
}
IMAGE_SIZE = 224

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmark_act_onnx import load_manifest, make_session  # noqa: E402
from tools.probe_act_recording_sources import (  # noqa: E402
    discover_camera_topics,
    init_devices,
    load_message_class,
    load_robot_ids,
    read_state26_timed,
    ros_stamp_seconds,
    shutdown_devices,
)


@dataclass(frozen=True)
class LiveFrame:
    image: np.ndarray
    arrival_monotonic: float
    ros_timestamp: float | None
    topic: str
    source_width: int
    source_height: int
    encoding: str


def raw_message_to_rgb(msg: Any) -> tuple[np.ndarray, str]:
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
    }.get(encoding)
    if width <= 0 or height <= 0 or step <= 0 or channels is None:
        raise ValueError(f"unsupported raw image {width}x{height} encoding={encoding!r}")
    minimum_step = width * channels
    payload = memoryview(msg.data)
    if step < minimum_step or len(payload) < step * height:
        raise ValueError(
            f"short image buffer len={len(payload)} step={step} height={height}"
        )
    rows = np.frombuffer(payload, dtype=np.uint8, count=step * height).reshape(height, step)
    pixels = rows[:, :minimum_step].reshape(height, width, channels)
    if channels == 1:
        rgb = np.repeat(pixels, 3, axis=2)
    else:
        rgb = pixels[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            rgb = rgb[:, :, ::-1]
    return np.asarray(rgb, dtype=np.uint8), encoding


def compressed_message_to_rgb(msg: Any) -> tuple[np.ndarray, str]:
    payload = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "compressed camera topic requires OpenCV; select a raw sensor_msgs/Image topic"
        ) from exc
    bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode compressed image format={getattr(msg, 'format', '')!r}")
    return bgr[:, :, ::-1], str(getattr(msg, "format", "compressed"))


def resize_nchw_float(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
    height, width = rgb.shape[:2]
    y_indices = np.minimum(
        (np.arange(IMAGE_SIZE, dtype=np.float64) * height / IMAGE_SIZE).astype(np.int64),
        height - 1,
    )
    x_indices = np.minimum(
        (np.arange(IMAGE_SIZE, dtype=np.float64) * width / IMAGE_SIZE).astype(np.int64),
        width - 1,
    )
    resized = rgb[y_indices[:, None], x_indices[None, :], :]
    nchw = np.transpose(resized, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(nchw, dtype=np.float32) / 255.0


class LiveCameraSnapshot:
    def __init__(self, topics: dict[str, dict[str, str]]) -> None:
        missing = sorted(set(CAMERA_INPUTS) - set(topics))
        if missing:
            raise RuntimeError(f"missing required camera topics: {missing}")
        self.topics = {name: topics[name] for name in CAMERA_INPUTS}
        self.frames: dict[str, LiveFrame] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._subscriptions: list[Any] = []
        self._callback_groups: list[Any] = []
        self._rclpy: Any = None
        self._owns_context = False

    def start(self) -> None:
        import rclpy
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node("rabo_act_onnx_one_shot")
        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self._node)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=6,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        for source_label, info in self.topics.items():
            callback_group = MutuallyExclusiveCallbackGroup()
            subscription = self._node.create_subscription(
                load_message_class(info["type"]),
                info["topic"],
                self._make_callback(source_label, info),
                qos,
                callback_group=callback_group,
            )
            self._callback_groups.append(callback_group)
            self._subscriptions.append(subscription)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _make_callback(
        self, source_label: str, info: dict[str, str]
    ) -> Callable[[Any], None]:
        def callback(msg: Any) -> None:
            try:
                if info["type"] == "sensor_msgs/msg/CompressedImage":
                    rgb, encoding = compressed_message_to_rgb(msg)
                else:
                    rgb, encoding = raw_message_to_rgb(msg)
                image = resize_nchw_float(rgb)
                frame = LiveFrame(
                    image=image,
                    arrival_monotonic=time.monotonic(),
                    ros_timestamp=ros_stamp_seconds(msg),
                    topic=info["topic"],
                    source_width=int(rgb.shape[1]),
                    source_height=int(rgb.shape[0]),
                    encoding=encoding,
                )
                with self._lock:
                    self.frames[source_label] = frame
                    self.errors.pop(source_label, None)
            except Exception as exc:
                with self._lock:
                    self.errors[source_label] = repr(exc)

        return callback

    def _spin(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._executor.spin_once(timeout_sec=0.1)

    def wait(self, timeout_s: float) -> dict[str, LiveFrame]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if set(self.frames) == set(CAMERA_INPUTS):
                    return dict(self.frames)
                errors = dict(self.errors)
            if errors:
                print(json.dumps({"camera_decode_errors": errors}, ensure_ascii=False), flush=True)
            time.sleep(0.05)
        with self._lock:
            received = sorted(self.frames)
            errors = dict(self.errors)
        raise TimeoutError(
            f"camera timeout after {timeout_s}s; received={received}; errors={errors}"
        )

    def close(self) -> None:
        self._stop.set()
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.wake()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown(timeout_sec=5.0)
        if self._node is not None:
            for subscription in self._subscriptions:
                with contextlib.suppress(Exception):
                    self._node.destroy_subscription(subscription)
        if self._executor is not None and self._node is not None:
            with contextlib.suppress(Exception):
                self._executor.remove_node(self._node)
        if self._node is not None:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
        if self._owns_context and self._rclpy is not None:
            with contextlib.suppress(Exception):
                self._rclpy.shutdown()


def finite_float_list(values: np.ndarray) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("array contains NaN/Inf")
    return [float(value) for value in values]


def run_one_shot(
    deployment_dir: Path,
    timeout_s: float,
    freshness_limit_s: float,
    threads: int,
) -> dict[str, Any]:
    manifest = load_manifest(deployment_dir)
    topics, discovery = discover_camera_topics()
    camera_reader = LiveCameraSnapshot(topics)
    devices: dict[str, Any] = {}
    try:
        camera_reader.start()
        frames = camera_reader.wait(timeout_s)
        devices = init_devices(load_robot_ids())
        state_values, components, read_durations = read_state26_timed(devices)
        state = np.asarray(state_values, dtype=np.float32)[None, :]
        if state.shape != (1, 26) or not np.isfinite(state).all():
            raise ValueError(f"invalid live state: {state.shape}")

        inference_started = time.perf_counter()
        session = make_session(deployment_dir / "act_policy.onnx", threads)
        inputs = {"state": state}
        for source_label, input_name in CAMERA_INPUTS.items():
            inputs[input_name] = frames[source_label].image
        chunk = session.run(["action_chunk"], inputs)[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        if chunk.shape != (1, 20, 26) or not np.isfinite(chunk).all():
            raise ValueError(f"invalid ACT output: {chunk.shape}")

        now = time.monotonic()
        camera_ages = {
            source_label: now - frame.arrival_monotonic
            for source_label, frame in frames.items()
        }
        first_action = chunk[0, 0]
        delta = np.abs(first_action - state[0])
        thresholds = manifest["dry_run_safety_thresholds"]
        arm_delta_limit = float(thresholds["max_arm_delta_rad"])
        hand_delta_limit = float(thresholds["max_hand_delta"])
        cameras_fresh = max(camera_ages.values()) <= freshness_limit_s
        arm_delta_safe = float(delta[:14].max()) <= arm_delta_limit
        hand_delta_safe = float(delta[14:].max()) <= hand_delta_limit
        return {
            "format": "rabo_act_onnx_live_one_shot_v1",
            "model": str(deployment_dir / "act_policy.onnx"),
            "provider": session.get_providers()[0],
            "discovered_topics": discovery.get("selected", {}),
            "camera_frames": {
                CAMERA_INPUTS[label]: {
                    "topic": frame.topic,
                    "source_size": [frame.source_width, frame.source_height],
                    "encoding": frame.encoding,
                    "ros_timestamp": frame.ros_timestamp,
                    "age_s": camera_ages[label],
                }
                for label, frame in frames.items()
            },
            "freshness_limit_s": freshness_limit_s,
            "cameras_fresh_for_actuation": cameras_fresh,
            "state_read_ms": {
                name: duration * 1000.0 for name, duration in read_durations.items()
            },
            "state_components": components,
            "inference_ms_including_session_load": inference_ms,
            "action_chunk_shape": list(chunk.shape),
            "first_action_26d": finite_float_list(first_action),
            "first_action_mapping": {
                "left_arm_rad": finite_float_list(first_action[0:7]),
                "right_arm_rad": finite_float_list(first_action[7:14]),
                "left_hand_normalized": finite_float_list(first_action[14:20]),
                "right_hand_normalized": finite_float_list(first_action[20:26]),
            },
            "first_action_delta": {
                "max_arm_rad": float(delta[:14].max()),
                "max_hand": float(delta[14:].max()),
                "arm_limit_rad": arm_delta_limit,
                "hand_limit": hand_delta_limit,
            },
            "prediction_safe_by_dry_run_thresholds": bool(arm_delta_safe and hand_delta_safe),
            "eligible_for_actuation": False,
            "robot_commanded": False,
            "note": "diagnostic inference only; no SDK write method exists in this program",
        }
    finally:
        camera_reader.close()
        if devices:
            shutdown_devices(devices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-dir", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--freshness-limit-s", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.timeout_s <= 0 or args.freshness_limit_s <= 0 or args.threads <= 0:
        raise SystemExit("timeout, freshness limit, and threads must be positive")
    result = run_one_shot(
        args.deployment_dir.resolve(),
        args.timeout_s,
        args.freshness_limit_s,
        args.threads,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="", flush=True)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print("ACT_LIVE_ONE_SHOT_COMPLETE: robot_commanded=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
