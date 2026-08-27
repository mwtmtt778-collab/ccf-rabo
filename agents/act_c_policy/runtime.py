"""Single-camera fixed-point Nut C ACT runtime.

The first deployment stage is deliberately dry-run only.  It receives passive
ROS observations and runs ONNX at 5 Hz, but opens no robot SDK control client.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "act_c_fixed_point_v1.onnx"
DEFAULT_CONTRACT = PROJECT_ROOT / "models" / "act_c_fixed_point_v1.json"
DEFAULT_DRY_RUN_LOG = PROJECT_ROOT / "logs" / "act_c_runtime_dry_run.jsonl"
TOP_RGB_TOPIC = "/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0"
HAND_OPEN = np.zeros(6, dtype=np.float32)
STATE_DIM = 26
ACTION_DIM = 28
CHUNK_SIZE = 10
POLICY_HZ = 5.0
# PassiveArmState.observation() retains subscription order
# [J1,J5,J4,J7,J3,J2,J6].  This is the same verified conversion used by its
# arm_state.jsonl logger to produce training order [J1,J2,J3,J4,J5,J6,J7].
PASSIVE_TO_SDK_INDICES = (0, 5, 4, 2, 1, 6, 3)


class RuntimeFault(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSnapshot:
    image: np.ndarray
    timestamp_monotonic_ns: int
    sequence: int


class TopCameraReader:
    """Independent ROS executor retaining the latest real TOP RGB frame."""

    def __init__(self, rclpy_module: Any, *, topic: str = TOP_RGB_TOPIC) -> None:
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self.rclpy = rclpy_module
        self.topic = topic
        self.node = rclpy_module.create_node("act_c_policy_top_camera")
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.node.create_subscription(Image, topic, self._callback, qos)
        self.lock = threading.Lock()
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self.latest: CameraSnapshot | None = None
        self.error: str | None = None
        self.frame_count = 0
        self.first_frame_ns: int | None = None
        self.last_frame_ns: int | None = None
        self.thread = threading.Thread(target=self._spin, name="act-c-top-camera", daemon=True)
        self.thread.start()

    @staticmethod
    def decode_image(message: Any, cv2_module: Any | None = None) -> np.ndarray:
        """Decode ROS Image by its declared encoding and return RGB NCHW [0,1]."""
        encoding = str(getattr(message, "encoding", "")).lower()
        channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
        if encoding not in channels_by_encoding:
            raise ValueError(f"unsupported TOP Image encoding: {encoding!r}")
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        channels = channels_by_encoding[encoding]
        if height <= 0 or width <= 0 or step < width * channels:
            raise ValueError(f"invalid TOP Image layout: {width}x{height}, step={step}, encoding={encoding}")
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if raw.size < height * step:
            raise ValueError(f"short TOP Image payload: {raw.size} < {height * step}")
        rows = raw[: height * step].reshape(height, step)
        pixels = rows[:, : width * channels].reshape(height, width, channels)
        if encoding == "rgb8":
            rgb = pixels
        elif encoding == "bgr8":
            rgb = pixels[..., ::-1]
        elif encoding == "rgba8":
            rgb = pixels[..., :3]
        elif encoding == "bgra8":
            rgb = pixels[..., [2, 1, 0]]
        else:
            rgb = np.repeat(pixels, 3, axis=2)
        if rgb.shape[:2] != (128, 128):
            if cv2_module is None:
                import cv2 as cv2_module

            rgb = cv2_module.resize(rgb, (128, 128), interpolation=cv2_module.INTER_AREA)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw[None, ...]

    def _callback(self, message: Any) -> None:
        try:
            image = self.decode_image(message)
            now_ns = time.monotonic_ns()
            with self.lock:
                self.frame_count += 1
                if self.first_frame_ns is None:
                    self.first_frame_ns = now_ns
                self.last_frame_ns = now_ns
                self.latest = CameraSnapshot(image, now_ns, self.frame_count)
                self.ready_event.set()
        except BaseException as exc:
            with self.lock:
                self.error = repr(exc)
            self.ready_event.set()

    def _spin(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.executor.spin_once(timeout_sec=0.05)
        except BaseException as exc:
            with self.lock:
                self.error = repr(exc)
            self.ready_event.set()

    def wait_ready(self, timeout_s: float) -> None:
        if not self.ready_event.wait(timeout_s):
            raise RuntimeFault(f"TOP first-frame timeout after {timeout_s:.1f}s: {self.topic}")
        with self.lock:
            if self.error is not None:
                raise RuntimeFault(f"TOP camera failed: {self.error}")
            if self.latest is None:
                raise RuntimeFault("TOP camera signaled ready without a frame")

    def snapshot(self) -> CameraSnapshot:
        with self.lock:
            if self.error is not None:
                raise RuntimeFault(f"TOP camera failed: {self.error}")
            if self.latest is None:
                raise RuntimeFault("TOP camera has no frame")
            return self.latest

    def effective_fps(self) -> float | None:
        with self.lock:
            if self.frame_count < 2 or self.first_frame_ns is None or self.last_frame_ns is None:
                return None
            span = (self.last_frame_ns - self.first_frame_ns) / 1e9
            return (self.frame_count - 1) / span if span > 0 else None

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()


class RuntimeStateReader:
    """state26 from passive arm14 plus command-side hold-last hand12."""

    def __init__(self, passive: Any) -> None:
        self.passive = passive
        # All five accepted training episodes start from Web Reset with both
        # command-side hand vectors open and both force modes inactive.
        self.left_hand = HAND_OPEN.copy()
        self.right_hand = HAND_OPEN.copy()
        self.left_mode = 0
        self.right_mode = 0

    def wait_ready(self, timeout_s: float) -> None:
        self.passive.wait_ready(timeout_s)

    def snapshot(self) -> tuple[np.ndarray, int]:
        left = self.passive.observation("left_arm")
        right = self.passive.observation("right_arm")
        left_sdk = np.asarray(left.state, dtype=np.float32)[list(PASSIVE_TO_SDK_INDICES)]
        right_sdk = np.asarray(right.state, dtype=np.float32)[list(PASSIVE_TO_SDK_INDICES)]
        state = np.concatenate(
            (
                left_sdk,
                right_sdk,
                self.left_hand,
                self.right_hand,
            )
        )[None, ...]
        if state.shape != (1, STATE_DIM) or not np.isfinite(state).all():
            raise RuntimeFault(f"invalid state26: shape={state.shape}, finite={np.isfinite(state).all()}")
        return state, min(int(left.timestamp_monotonic_ns), int(right.timestamp_monotonic_ns))


class OnnxPolicy:
    def __init__(self, model_path: Path, *, threads: int = 2, session: Any | None = None) -> None:
        self.model_path = model_path.resolve()
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.session = session
        inputs = {item.name: (tuple(item.shape), item.type) for item in session.get_inputs()}
        outputs = {item.name: (tuple(item.shape), item.type) for item in session.get_outputs()}
        expected_inputs = {
            "top_image": ((1, 3, 128, 128), "tensor(float)"),
            "state": ((1, 26), "tensor(float)"),
        }
        expected_outputs = {"action_chunk": ((1, 10, 28), "tensor(float)")}
        if inputs != expected_inputs or outputs != expected_outputs:
            raise RuntimeFault(f"ONNX contract mismatch: inputs={inputs}, outputs={outputs}")

    def infer(self, image: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, float]:
        if image.shape != (1, 3, 128, 128) or image.dtype != np.float32 or not np.isfinite(image).all():
            raise RuntimeFault(f"invalid ONNX top_image: {image.shape}/{image.dtype}")
        if state.shape != (1, 26) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise RuntimeFault(f"invalid ONNX state: {state.shape}/{state.dtype}")
        started = time.perf_counter()
        output = np.asarray(
            self.session.run(["action_chunk"], {"top_image": image, "state": state})[0],
            dtype=np.float32,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if output.shape != (1, CHUNK_SIZE, ACTION_DIM) or not np.isfinite(output).all():
            raise RuntimeFault(f"invalid ONNX action_chunk: {output.shape}, finite={np.isfinite(output).all()}")
        return output, latency_ms


class FixedPointCRuntime:
    def __init__(
        self,
        camera: Any,
        state_reader: RuntimeStateReader,
        policy: OnnxPolicy,
        *,
        rate_hz: float = POLICY_HZ,
        monotonic: Any = time.monotonic,
        monotonic_ns: Any = time.monotonic_ns,
        sleep: Any = time.sleep,
    ) -> None:
        self.camera = camera
        self.state_reader = state_reader
        self.policy = policy
        self.rate_hz = rate_hz
        self.monotonic = monotonic
        self.monotonic_ns = monotonic_ns
        self.sleep = sleep

    def wait_ready(self, timeout_s: float) -> None:
        deadline = self.monotonic() + timeout_s
        self.camera.wait_ready(max(0.0, deadline - self.monotonic()))
        self.state_reader.wait_ready(max(0.0, deadline - self.monotonic()))
        state, _ = self.state_reader.snapshot()
        if not np.isfinite(state).all():
            raise RuntimeFault("state26 is not finite at readiness gate")

    def run_dry(self, duration_s: float, log_path: Path) -> dict[str, Any]:
        if duration_s <= 0 or self.rate_hz <= 0:
            raise ValueError("duration/rate must be positive")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        period = 1.0 / self.rate_hz
        start = self.monotonic()
        next_summary = start
        tick_times: list[float] = []
        latencies: list[float] = []
        camera_sequences: set[int] = set()
        tick = 0
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            while True:
                due = start + tick * period
                remaining = due - self.monotonic()
                if remaining > 0:
                    self.sleep(remaining)
                if due >= start + duration_s:
                    break
                tick_started = self.monotonic()
                now_ns = self.monotonic_ns()
                camera = self.camera.snapshot()
                state, state_timestamp_ns = self.state_reader.snapshot()
                chunk, latency_ms = self.policy.infer(camera.image, state)
                action = chunk[0, 0]
                row = {
                    "timestamp_monotonic_ns": now_ns,
                    "tick": tick,
                    "camera_sequence": camera.sequence,
                    "camera_age_ms": (now_ns - camera.timestamp_monotonic_ns) / 1e6,
                    "arm_state_age_ms": (now_ns - state_timestamp_ns) / 1e6,
                    "state26_finite": bool(np.isfinite(state).all()),
                    "onnx_latency_ms": latency_ms,
                    "action_min": float(action.min()),
                    "action_max": float(action.max()),
                    "left_mode": float(action[26]),
                    "right_mode": float(action[27]),
                    "action_finite": bool(np.isfinite(chunk).all()),
                    "robot_commanded": False,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                tick_times.append(tick_started)
                latencies.append(latency_ms)
                camera_sequences.add(camera.sequence)
                if tick_started >= next_summary:
                    print(
                        "[ACT_C_DRY] "
                        f"camera_age_ms={row['camera_age_ms']:.1f} "
                        f"arm_state_age_ms={row['arm_state_age_ms']:.1f} "
                        f"state26_finite={row['state26_finite']} "
                        f"onnx_latency_ms={latency_ms:.2f} "
                        f"action_min/max={row['action_min']:.3f}/{row['action_max']:.3f} "
                        f"left_mode={row['left_mode']:.3f} right_mode={row['right_mode']:.3f}",
                        flush=True,
                    )
                    next_summary = tick_started + 1.0
                tick += 1
        span = tick_times[-1] - tick_times[0] if len(tick_times) > 1 else 0.0
        effective_hz = (len(tick_times) - 1) / span if span > 0 else 0.0
        summary = {
            "status": "PASS" if tick_times and max(latencies) < 200.0 and 0.9 * self.rate_hz <= effective_hz <= 1.1 * self.rate_hz else "FAULT",
            "duration_s": duration_s,
            "tick_count": len(tick_times),
            "loop_effective_hz": effective_hz,
            "camera_effective_fps": self.camera.effective_fps(),
            "camera_unique_frames_used": len(camera_sequences),
            "onnx_latency_ms": {
                "mean": float(np.mean(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "max": float(np.max(latencies)),
            },
            "action_finite": True,
            "robot_commanded": False,
            "log": str(log_path.resolve()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if summary["status"] != "PASS":
            raise RuntimeFault(f"dry-run acceptance failed: {summary}")
        return summary


def validate_contract_file(path: Path) -> None:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("camera_count") != 1 or contract.get("state_dim") != 26:
        raise RuntimeFault("model contract camera/state mismatch")
    if contract.get("action_dim") != 28 or contract.get("chunk_size") != 10:
        raise RuntimeFault("model contract action/chunk mismatch")


def run_live_dry(args: argparse.Namespace) -> dict[str, Any]:
    # PassiveArmState owns rclpy init and its dedicated arm-state executor.
    from tools.run_c_only_serial_expert import PassiveArmState

    validate_contract_file(args.contract.resolve())
    passive = PassiveArmState(history_hz=20.0)
    camera: TopCameraReader | None = None
    try:
        camera = TopCameraReader(passive.rclpy)
        state_reader = RuntimeStateReader(passive)
        policy = OnnxPolicy(args.model, threads=args.threads)
        runtime = FixedPointCRuntime(camera, state_reader, policy, rate_hz=args.rate)
        runtime.wait_ready(args.ready_timeout)
        print("ACT C FIXED-POINT POLICY", flush=True)
        print(f"Model: {args.model.resolve()}", flush=True)
        print("Camera: READY", flush=True)
        print("State: READY", flush=True)
        print(f"Policy rate: {args.rate:g} Hz", flush=True)
        print("Mode: DRY_RUN (no robot SDK control clients)", flush=True)
        return runtime.run_dry(args.duration, args.log)
    finally:
        if camera is not None:
            camera.close()
        passive.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=POLICY_HZ)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log", type=Path, default=DEFAULT_DRY_RUN_LOG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        raise SystemExit(
            "EXECUTE_DISABLED_FAIL_CLOSED: no retained evidence proves dual-arm move_joints can accept "
            "continuous 5 Hz targets, and no verified threshold maps continuous ONNX grasp-mode outputs "
            "to the existing HybridHandExecutor's exact {0,1} modes. No SDK client was created."
        )
    try:
        run_live_dry(args)
        return 0
    except BaseException as exc:
        print(f"FAULT: {exc!r}", file=sys.stderr, flush=True)
        return 1
