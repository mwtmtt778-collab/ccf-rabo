"""Single-camera fixed-point Nut C ACT dry-run and serial execution MVP."""

from __future__ import annotations

import argparse
import contextlib
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
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "act_c_policy.json"
HAND_OPEN = np.zeros(6, dtype=np.float32)
STATE_DIM = 26
ACTION_DIM = 28
CHUNK_SIZE = 10
POLICY_HZ = 5.0
MAX_DELTA_PER_STEP = 0.10
JOINT_LIMIT_MARGIN_RAD = 0.001
MIN_ARM_COMMAND_DELTA = 0.005
MAX_POLICY_STEPS = 150
POST_DONE_ARM_STEPS = 5
MODE_RISE_THRESHOLD = 0.8
MODE_FALL_THRESHOLD = 0.2
MODE_CONFIRM_SAMPLES = 3
# Retained LinkerArmA7 SDK/SDF limits in public SDK J1..J7 order, radians.
ARM_JOINT_LIMITS = np.asarray(
    [
        (-2.18, 3.75),
        (-3.20, 0.07),
        (-2.69, 2.69),
        (-2.05, 2.05),
        (-2.69, 2.69),
        (-1.59, 1.59),
        (-1.59, 1.59),
    ],
    dtype=np.float32,
)
# PassiveArmState.observation() retains subscription order
# [J1,J5,J4,J7,J3,J2,J6].  This is the same verified conversion used by its
# arm_state.jsonl logger to produce training order [J1,J2,J3,J4,J5,J6,J7].
PASSIVE_TO_SDK_INDICES = (0, 5, 4, 2, 1, 6, 3)
IMAGE_TYPE = "sensor_msgs/msg/Image"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
GRASP_FORCE_CONFIG = {"strength": 1.0, "fingers": [1, 3, 4]}


class RuntimeFault(RuntimeError):
    pass


@dataclass(frozen=True)
class ActCConfig:
    repo_root: Path
    path: Path
    camera_topic: str
    left_arm_name: str
    right_arm_name: str
    left_hand_name: str
    right_hand_name: str
    model: Path
    contract: Path
    dry_run_log: Path
    execute_log: Path
    max_delta_rad: float
    joint_limit_margin_rad: float


def _repo_path(repo_root: Path, value: Any, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise RuntimeFault(f"config {field} must be repo-relative: {path}")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeFault(f"config {field} escapes repo root: {path}") from exc
    return resolved


def load_config(
    path: Path = DEFAULT_CONFIG,
    *,
    repo_root: Path = PROJECT_ROOT,
    require_assets: bool = True,
) -> ActCConfig:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeFault(f"ACT C config missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeFault(f"ACT C config unreadable: {exc!r}") from exc
    required = {
        "camera_topic", "left_arm_name", "right_arm_name", "left_hand_name",
        "right_hand_name", "model", "contract", "dry_run_log", "execute_log",
        "max_delta_rad", "joint_limit_margin_rad",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise RuntimeFault(f"ACT C config missing fields: {missing}")
    model = _repo_path(repo_root, raw["model"], "model")
    contract = _repo_path(repo_root, raw["contract"], "contract")
    config = ActCConfig(
        repo_root=repo_root.resolve(),
        path=path,
        camera_topic=str(raw["camera_topic"]),
        left_arm_name=str(raw["left_arm_name"]),
        right_arm_name=str(raw["right_arm_name"]),
        left_hand_name=str(raw["left_hand_name"]),
        right_hand_name=str(raw["right_hand_name"]),
        model=model,
        contract=contract,
        dry_run_log=_repo_path(repo_root, raw["dry_run_log"], "dry_run_log"),
        execute_log=_repo_path(repo_root, raw["execute_log"], "execute_log"),
        max_delta_rad=float(raw["max_delta_rad"]),
        joint_limit_margin_rad=float(raw["joint_limit_margin_rad"]),
    )
    if not config.camera_topic or not all(
        (config.left_arm_name, config.right_arm_name, config.left_hand_name, config.right_hand_name)
    ):
        raise RuntimeFault("ACT C device names and camera_topic must be nonempty")
    if not 0 < config.max_delta_rad <= MAX_DELTA_PER_STEP:
        raise RuntimeFault(f"max_delta_rad must be in (0,{MAX_DELTA_PER_STEP}]")
    half_width = float(np.min(ARM_JOINT_LIMITS[:, 1] - ARM_JOINT_LIMITS[:, 0])) / 2.0
    if not 0 < config.joint_limit_margin_rad < half_width:
        raise RuntimeFault("joint_limit_margin_rad is invalid")
    if require_assets:
        if not config.model.is_file():
            raise RuntimeFault(f"ACT C ONNX model missing: {config.model}")
        if not config.contract.is_file():
            raise RuntimeFault(f"ACT C model contract missing: {config.contract}")
    return config


def _topic_leaf_entity(topic: str, marker: str) -> str | None:
    leaf = topic.rsplit("/", 1)[-1]
    if marker not in leaf:
        return None
    return leaf.split(marker, 1)[0]


def joint_state_topic_groups(topic_pairs: Sequence[tuple[str, Sequence[str]]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, types in topic_pairs:
        if JOINT_STATE_TYPE not in types:
            continue
        entity = _topic_leaf_entity(name, "_tp_ps_")
        if entity is not None:
            groups.setdefault(entity, []).append(name)
    return {entity: sorted(names) for entity, names in groups.items()}


def image_topic_candidates(topic_pairs: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    groups = joint_state_topic_groups(topic_pairs)
    arm_entities = {entity for entity, names in groups.items() if len(names) == 7}
    rgb_topics = sorted(
        name
        for name, types in topic_pairs
        if IMAGE_TYPE in types and _topic_leaf_entity(name, "_tp_cam_") is not None
    )
    # A fixed TOP camera does not share the entity prefix of a discovered 7-DOF arm.
    return [
        name for name in rgb_topics
        if _topic_leaf_entity(name, "_tp_cam_") not in arm_entities
    ]


def select_camera_topic(
    topic_pairs: Sequence[tuple[str, Sequence[str]]],
    configured: str,
) -> str:
    image_topics = sorted(name for name, types in topic_pairs if IMAGE_TYPE in types)
    if configured != "auto":
        if configured not in image_topics:
            raise RuntimeFault(
                f"configured camera topic is unavailable: {configured}; image topics={image_topics}"
            )
        return configured
    candidates = image_topic_candidates(topic_pairs)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeFault(f"camera auto discovery found 0 legal candidates; image topics={image_topics}")
    raise RuntimeFault(
        f"camera auto discovery is ambiguous ({len(candidates)} candidates): {candidates}; "
        "set camera_topic explicitly in config/act_c_policy.json"
    )


class RclpyOwner:
    """Own global rclpy initialization and perform shutdown exactly once, last."""

    def __init__(self, rclpy_module: Any) -> None:
        self.rclpy = rclpy_module
        self.owns_init = not bool(rclpy_module.ok())
        self.closed = False
        if self.owns_init:
            rclpy_module.init(args=None)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.owns_init and self.rclpy.ok():
            self.rclpy.shutdown()


def discover_topic_pairs(rclpy_module: Any, timeout_s: float) -> list[tuple[str, list[str]]]:
    node = rclpy_module.create_node("act_c_policy_graph_discovery")
    try:
        deadline = time.monotonic() + max(0.0, timeout_s)
        pairs: list[tuple[str, list[str]]] = []
        while True:
            pairs = sorted(
                (str(name), [str(value) for value in types])
                for name, types in node.get_topic_names_and_types()
            )
            if any(IMAGE_TYPE in types for _, types in pairs) or time.monotonic() >= deadline:
                return pairs
            time.sleep(0.1)
    finally:
        node.destroy_node()


@dataclass(frozen=True)
class CameraSnapshot:
    image: np.ndarray
    timestamp_monotonic_ns: int
    sequence: int


class TopCameraReader:
    """Independent ROS executor retaining the latest real TOP RGB frame."""

    def __init__(self, rclpy_module: Any, *, topic: str) -> None:
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
        self._close_lock = threading.Lock()
        self._closed = False
        self._executor_closed = False
        self._node_destroyed = False
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
        if self.stop_event.is_set():
            return
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

    def stop_callbacks(self) -> None:
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.executor.wake()
        if self.thread is not threading.current_thread() and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.stop_callbacks()
            if not self._executor_closed:
                with contextlib.suppress(Exception):
                    self.executor.remove_node(self.node)
                try:
                    with contextlib.suppress(Exception):
                        self.executor.shutdown(timeout_sec=1.0)
                finally:
                    self._executor_closed = True
            if not self._node_destroyed:
                try:
                    with contextlib.suppress(Exception):
                        self.node.destroy_node()
                finally:
                    self._node_destroyed = True
            self._closed = True


@dataclass(frozen=True)
class ArmObservation:
    sequence: int
    timestamp_monotonic_ns: int
    state: tuple[float, ...]


def require_sdk_success(value: Any, operation: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            return
        raise RuntimeFault(f"{operation} returned False")
    text = str(value).lower()
    if any(token in text for token in ("error", "failed", "fail", "false", "失败", "错误")):
        raise RuntimeFault(f"{operation} failed: {value!r}")


class SdkDeviceBundle:
    """Portable Rabo SDK clients selected only by config device names."""

    def __init__(self, config: ActCConfig, *, include_hands: bool) -> None:
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

        self.closed = False
        self.left_arm = self.right_arm = self.left_hand = self.right_hand = None
        try:
            self.left_arm = LinkerArmA7(robot_id=config.left_arm_name, mode="sim")
            self.right_arm = LinkerArmA7(robot_id=config.right_arm_name, mode="sim")
            if include_hands:
                self.left_hand = LinkerHandO6Left(robot_id=config.left_hand_name, mode="sim")
                self.right_hand = LinkerHandO6Right(robot_id=config.right_hand_name, mode="sim")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for device in (self.left_hand, self.right_hand, self.left_arm, self.right_arm):
            if device is not None and hasattr(device, "shutdown"):
                with contextlib.suppress(Exception):
                    device.shutdown()


class SdkArmStateSource:
    """Read-only state26 arm source using the public SDK J1..J7 order."""

    sdk_order = True

    def __init__(self, left_arm: Any, right_arm: Any) -> None:
        self.arms = {"left_arm": left_arm, "right_arm": right_arm}
        self.sequences = {"left_arm": 0, "right_arm": 0}

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        last_error: BaseException | None = None
        while time.monotonic() <= deadline:
            try:
                self.observation("left_arm")
                self.observation("right_arm")
                return
            except BaseException as exc:
                last_error = exc
                time.sleep(0.05)
        raise RuntimeFault(f"SDK passive arm state not ready: {last_error!r}")

    def observation(self, arm: str) -> ArmObservation:
        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm}")
        values = np.asarray(self.arms[arm].get_joint_angles(), dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise RuntimeFault(f"invalid {arm} SDK state: shape={values.shape}, values={values}")
        self.sequences[arm] += 1
        return ArmObservation(
            self.sequences[arm], time.monotonic_ns(), tuple(float(value) for value in values)
        )


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
        left_raw = np.asarray(left.state, dtype=np.float32)
        right_raw = np.asarray(right.state, dtype=np.float32)
        if getattr(self.passive, "sdk_order", False):
            left_sdk, right_sdk = left_raw, right_raw
        else:
            left_sdk = left_raw[list(PASSIVE_TO_SDK_INDICES)]
            right_sdk = right_raw[list(PASSIVE_TO_SDK_INDICES)]
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

    def set_hand_position(self, hand: str, target: Sequence[float]) -> None:
        values = np.asarray(target, dtype=np.float32)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise RuntimeFault(f"invalid {hand} hand position target: {values}")
        if hand == "left":
            self.left_hand = values.copy()
        elif hand == "right":
            self.right_hand = values.copy()
        else:
            raise ValueError(f"unknown hand: {hand}")

    def set_hand_mode(self, hand: str, mode: int) -> None:
        if mode not in (0, 1):
            raise ValueError(f"invalid hand mode: {mode}")
        if hand == "left":
            self.left_mode = mode
        elif hand == "right":
            self.right_mode = mode
        else:
            raise ValueError(f"unknown hand: {hand}")


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


@dataclass(frozen=True)
class ArmTarget:
    raw_prediction: np.ndarray
    clipped_target: np.ndarray
    max_raw_delta: float
    max_executed_delta: float


def build_safe_arm_target(
    current_arm14: Sequence[float] | np.ndarray,
    predicted_arm14: Sequence[float] | np.ndarray,
    *,
    max_delta_per_step: float = MAX_DELTA_PER_STEP,
    joint_limit_margin_rad: float = JOINT_LIMIT_MARGIN_RAD,
) -> ArmTarget:
    current = np.asarray(current_arm14, dtype=np.float32)
    predicted = np.asarray(predicted_arm14, dtype=np.float32)
    if current.shape != (14,) or predicted.shape != (14,):
        raise RuntimeFault(f"arm target shape mismatch: current={current.shape}, predicted={predicted.shape}")
    if not np.isfinite(current).all() or not np.isfinite(predicted).all():
        raise RuntimeFault("arm current/prediction contains NaN or Inf")
    if max_delta_per_step <= 0:
        raise ValueError("max_delta_per_step must be positive")
    low = np.tile(ARM_JOINT_LIMITS[:, 0], 2)
    high = np.tile(ARM_JOINT_LIMITS[:, 1], 2)
    safe_low = low + float(joint_limit_margin_rad)
    safe_high = high - float(joint_limit_margin_rad)
    if joint_limit_margin_rad <= 0 or np.any(safe_low >= safe_high):
        raise ValueError("joint_limit_margin_rad is invalid")
    if np.any(current < low) or np.any(current > high):
        raise RuntimeFault("current real arm state is outside SDK hard limits")
    raw_delta = predicted - current
    clipped_delta = np.clip(raw_delta, -max_delta_per_step, max_delta_per_step)
    target = np.clip(current + clipped_delta, safe_low, safe_high).astype(np.float32, copy=False)
    executed_delta = target - current
    if float(np.max(np.abs(executed_delta))) > max_delta_per_step + 1e-6:
        raise RuntimeFault("internal arm delta clip violation")
    assert_arm_target_safe(target, joint_limit_margin_rad=joint_limit_margin_rad)
    return ArmTarget(
        raw_prediction=predicted.copy(),
        clipped_target=target.copy(),
        max_raw_delta=float(np.max(np.abs(raw_delta))),
        max_executed_delta=float(np.max(np.abs(executed_delta))),
    )


def assert_arm_target_safe(
    target_arm14: Sequence[float] | np.ndarray,
    *,
    joint_limit_margin_rad: float = JOINT_LIMIT_MARGIN_RAD,
) -> None:
    target = np.asarray(target_arm14, dtype=np.float32)
    if target.shape not in ((7,), (14,)) or not np.isfinite(target).all():
        raise RuntimeFault(f"invalid final arm target: shape={target.shape}")
    repeats = 1 if target.shape == (7,) else 2
    low = np.tile(ARM_JOINT_LIMITS[:, 0], repeats)
    high = np.tile(ARM_JOINT_LIMITS[:, 1], repeats)
    safe_low = low + float(joint_limit_margin_rad)
    safe_high = high - float(joint_limit_margin_rad)
    if np.any(target < safe_low) or np.any(target > safe_high):
        raise RuntimeFault("final arm target violates joint limit safety margin")
    if np.any(target <= low) or np.any(target >= high):
        raise RuntimeFault("final arm target is not strictly inside SDK hard limits")


@dataclass(frozen=True)
class ModeDecision:
    mode: int
    apply_hand: bool
    transition: str | None


class HybridHandExecutor:
    """Persistent position/force actuator semantics shared by Agent and tool."""

    def __init__(self, clench: Any, grasp_force: Any) -> None:
        self._clench = clench
        self._grasp_force = grasp_force
        self.mode = 0

    def apply(self, target_clench: Sequence[float], target_mode: int, *, blocking: bool) -> str:
        target_mode = int(target_mode)
        if target_mode not in (0, 1):
            raise ValueError(f"invalid grasp mode: {target_mode}")
        if self.mode == 0 and target_mode == 1:
            value = self._grasp_force(blocking)
            require_sdk_success(value, "hand.grasp_force")
            self.mode = 1
            return "GRASP_FORCE_RISING_EDGE"
        if self.mode == 1 and target_mode == 1:
            return "FORCE_MODE_HOLD_NO_HAND_COMMAND"
        values = [float(value) for value in target_clench]
        if self.mode == 1 and target_mode == 0:
            value = self._clench(values, blocking)
            require_sdk_success(value, "hand.clench")
            self.mode = 0
            return "POSITION_MODE_FALLING_EDGE_CLENCH"
        value = self._clench(values, blocking)
        require_sdk_success(value, "hand.clench")
        return "POSITION_MODE_CLENCH"


class ConstrainedHandModeState:
    """Three-sample hysteresis for one POSITION -> FORCE -> DONE sequence."""

    def __init__(
        self,
        *,
        rise_threshold: float = MODE_RISE_THRESHOLD,
        fall_threshold: float = MODE_FALL_THRESHOLD,
        confirm_samples: int = MODE_CONFIRM_SAMPLES,
    ) -> None:
        if not 0 <= fall_threshold < rise_threshold <= 1:
            raise ValueError("mode thresholds must satisfy 0 <= fall < rise <= 1")
        if confirm_samples < 1:
            raise ValueError("confirm_samples must be >= 1")
        self.rise_threshold = float(rise_threshold)
        self.fall_threshold = float(fall_threshold)
        self.confirm_samples = int(confirm_samples)
        self.state = "POSITION"
        self.high_count = 0
        self.low_count = 0

    def update(self, score: float, *, enabled: bool = True) -> ModeDecision:
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeFault("hand mode score contains NaN or Inf")
        if self.state == "DONE":
            return ModeDecision(0, False, None)
        if not enabled:
            if self.state != "POSITION":
                raise RuntimeFault("cannot disable a hand after FORCE has started")
            self.high_count = 0
            return ModeDecision(0, True, None)
        if self.state == "POSITION":
            self.high_count = self.high_count + 1 if score >= self.rise_threshold else 0
            if self.high_count >= self.confirm_samples:
                self.state = "FORCE"
                self.high_count = 0
                self.low_count = 0
                return ModeDecision(1, True, "POSITION_TO_FORCE")
            return ModeDecision(0, True, None)
        if self.state == "FORCE":
            self.low_count = self.low_count + 1 if score <= self.fall_threshold else 0
            if self.low_count >= self.confirm_samples:
                self.state = "DONE"
                self.low_count = 0
                return ModeDecision(0, True, "FORCE_TO_DONE")
            return ModeDecision(1, True, None)
        raise RuntimeFault(f"unknown hand mode state: {self.state}")


class SerialRecedingHorizonMvp:
    def __init__(
        self,
        camera: Any,
        state_reader: RuntimeStateReader,
        policy: OnnxPolicy,
        arm_executor: Any,
        left_hand_executor: Any,
        right_hand_executor: Any,
        *,
        max_policy_steps: int = MAX_POLICY_STEPS,
        post_done_arm_steps: int = POST_DONE_ARM_STEPS,
        max_delta_per_step: float = MAX_DELTA_PER_STEP,
        joint_limit_margin_rad: float = JOINT_LIMIT_MARGIN_RAD,
        no_arm_move_threshold: float = MIN_ARM_COMMAND_DELTA,
        monotonic_ns: Any = time.monotonic_ns,
        sleep: Any = time.sleep,
    ) -> None:
        if max_policy_steps < 1 or post_done_arm_steps < 0:
            raise ValueError("invalid MVP step limits")
        self.camera = camera
        self.state_reader = state_reader
        self.policy = policy
        self.arm_executor = arm_executor
        self.left_hand_executor = left_hand_executor
        self.right_hand_executor = right_hand_executor
        self.max_policy_steps = int(max_policy_steps)
        self.post_done_arm_steps = int(post_done_arm_steps)
        self.max_delta_per_step = float(max_delta_per_step)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.no_arm_move_threshold = float(no_arm_move_threshold)
        self.monotonic_ns = monotonic_ns
        self.sleep = sleep
        self.right_mode = ConstrainedHandModeState()
        self.left_mode = ConstrainedHandModeState()
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def _apply_hand(
        self,
        hand: str,
        executor: Any,
        target: np.ndarray,
        decision: ModeDecision,
    ) -> str:
        if not decision.apply_hand:
            return "DONE_NO_HAND_COMMAND"
        semantic = executor.apply(target.tolist(), decision.mode, blocking=True)
        self.state_reader.set_hand_mode(hand, decision.mode)
        if decision.mode == 0:
            self.state_reader.set_hand_position(hand, target)
        return semantic

    def run(self, log_path: Path) -> dict[str, Any]:
        log_path = log_path.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        post_done_started = False
        post_done_steps = 0
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            for step in range(self.max_policy_steps):
                if self.stop_event.is_set():
                    raise RuntimeFault("policy loop stopped")
                now_ns = int(self.monotonic_ns())
                camera = self.camera.snapshot()
                state, state_timestamp_ns = self.state_reader.snapshot()
                if state.shape != (1, STATE_DIM) or state.dtype != np.float32 or not np.isfinite(state).all():
                    raise RuntimeFault(f"invalid live state26: {state.shape}/{state.dtype}")
                chunk, onnx_ms = self.policy.infer(camera.image, state)
                if chunk.shape != (1, CHUNK_SIZE, ACTION_DIM) or not np.isfinite(chunk).all():
                    raise RuntimeFault("ONNX action chunk contains NaN/Inf or has wrong shape")
                action = chunk[0, 0]
                arm_target = build_safe_arm_target(
                    state[0, :14], action[:14],
                    max_delta_per_step=self.max_delta_per_step,
                    joint_limit_margin_rad=self.joint_limit_margin_rad,
                )
                print(
                    "[ACT_MVP_TARGET] "
                    f"raw_pred_target={arm_target.raw_prediction.tolist()} "
                    f"clipped_target={arm_target.clipped_target.tolist()} "
                    f"max_raw_delta={arm_target.max_raw_delta:.6f} "
                    f"max_executed_delta={arm_target.max_executed_delta:.6f}",
                    flush=True,
                )
                arm_commanded = arm_target.max_executed_delta >= self.no_arm_move_threshold
                settled = True
                if arm_commanded:
                    assert_arm_target_safe(
                        arm_target.clipped_target,
                        joint_limit_margin_rad=self.joint_limit_margin_rad,
                    )
                    settled = bool(
                        self.arm_executor.move_pair(
                            arm_target.clipped_target[:7].tolist(),
                            arm_target.clipped_target[7:14].tolist(),
                            step=step,
                        )
                    )
                    if not settled:
                        raise TimeoutError("passive settle returned false")
                else:
                    self.sleep(0.2)

                right_decision = self.right_mode.update(float(action[27]), enabled=True)
                left_decision = self.left_mode.update(
                    float(action[26]), enabled=self.right_mode.state == "DONE"
                )
                right_semantic = self._apply_hand(
                    "right", self.right_hand_executor, action[20:26], right_decision
                )
                left_semantic = self._apply_hand(
                    "left", self.left_hand_executor, action[14:20], left_decision
                )
                row = {
                    "event": "ACT_MVP_STEP",
                    "step": step,
                    "timestamp_monotonic_ns": now_ns,
                    "camera_age_ms": (now_ns - int(camera.timestamp_monotonic_ns)) / 1e6,
                    "state_age_ms": (now_ns - int(state_timestamp_ns)) / 1e6,
                    "onnx_ms": float(onnx_ms),
                    "raw_pred_target": arm_target.raw_prediction.tolist(),
                    "clipped_target": arm_target.clipped_target.tolist(),
                    "raw_delta_max": arm_target.max_raw_delta,
                    "executed_delta_max": arm_target.max_executed_delta,
                    "right_score": float(action[27]),
                    "right_mode": self.right_mode.state,
                    "right_hand_semantic": right_semantic,
                    "left_score": float(action[26]),
                    "left_mode": self.left_mode.state,
                    "left_hand_semantic": left_semantic,
                    "arm_commanded": arm_commanded,
                    "settled": settled,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    "[ACT_MVP] "
                    f"step={step} camera_age_ms={row['camera_age_ms']:.1f} "
                    f"state_age_ms={row['state_age_ms']:.1f} onnx_ms={onnx_ms:.2f} "
                    f"raw_delta_max={arm_target.max_raw_delta:.6f} "
                    f"executed_delta_max={arm_target.max_executed_delta:.6f} "
                    f"right_score={float(action[27]):.3f} right_mode={self.right_mode.state} "
                    f"left_score={float(action[26]):.3f} left_mode={self.left_mode.state} "
                    f"arm_commanded={arm_commanded} settled={settled}",
                    flush=True,
                )

                both_done = self.right_mode.state == "DONE" and self.left_mode.state == "DONE"
                if post_done_started:
                    post_done_steps += 1
                    if post_done_steps >= self.post_done_arm_steps:
                        print("ACT C MVP FINISHED", flush=True)
                        return {
                            "status": "DONE",
                            "policy_steps": step + 1,
                            "post_done_arm_steps": post_done_steps,
                            "log": str(log_path),
                        }
                elif both_done:
                    post_done_started = True
                    if self.post_done_arm_steps == 0:
                        print("ACT C MVP FINISHED", flush=True)
                        return {
                            "status": "DONE",
                            "policy_steps": step + 1,
                            "post_done_arm_steps": 0,
                            "log": str(log_path),
                        }
        raise RuntimeFault("MAX_POLICY_STEPS_EXCEEDED")


class SerialArmExecutor:
    """Serial move_joints followed by passive SDK state settle detection."""

    def __init__(
        self,
        state_source: SdkArmStateSource,
        left_arm: Any,
        right_arm: Any,
        args: argparse.Namespace,
        *,
        joint_limit_margin_rad: float,
        monotonic: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.state_source = state_source
        self.arms = {"left_arm": left_arm, "right_arm": right_arm}
        self.args = args
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.monotonic = monotonic
        self.sleep = sleep

    def _move_one(self, arm_name: str, target: list[float], *, step: int) -> dict[str, Any]:
        # Validate this 7D target against the same safety margin immediately before SDK send.
        assert_arm_target_safe(target, joint_limit_margin_rad=self.joint_limit_margin_rad)

        baseline = self.state_source.observation(arm_name)
        started = self.monotonic()
        result = self.arms[arm_name].move_joints([float(value) for value in target], blocking=False)
        require_sdk_success(result, f"{arm_name}.move_joints")
        returned = self.monotonic()
        deadline = started + float(self.args.motion_timeout_s)
        previous = np.asarray(baseline.state, dtype=np.float32)
        movement_started = False
        stable_count = 0
        no_motion_stable_count = 0
        max_observed_delta = 0.0
        while self.monotonic() < deadline:
            observation = self.state_source.observation(arm_name)
            current = np.asarray(observation.state, dtype=np.float32)
            adjacent = float(np.max(np.abs(current - previous)))
            baseline_delta = float(np.max(np.abs(current - np.asarray(baseline.state))))
            max_observed_delta = max(max_observed_delta, adjacent, baseline_delta)
            significant = max(adjacent, baseline_delta) > float(self.args.start_threshold_rad)
            if not movement_started and significant:
                movement_started = True
                stable_count = 0
            elif movement_started and adjacent < float(self.args.settle_threshold_rad):
                stable_count += 1
            elif movement_started:
                stable_count = 0
            elif adjacent < float(self.args.settle_threshold_rad):
                no_motion_stable_count += 1
            else:
                no_motion_stable_count = 0
            previous = current
            already_settled = (
                not movement_started
                and no_motion_stable_count >= int(self.args.settle_samples)
                and self.monotonic() - returned >= float(self.args.no_motion_grace_s)
            )
            if (movement_started and stable_count >= int(self.args.settle_samples)) or already_settled:
                settled = {
                    "movement_started": movement_started,
                    "already_settled": already_settled,
                    "max_observed_delta_rad": max_observed_delta,
                    "settle_elapsed_s": self.monotonic() - started,
                }
                print(
                    "[ARM_SERIAL] "
                    f"step={step} arm={arm_name} settled=True "
                    f"movement_started={movement_started} already_settled={already_settled} "
                    f"settle_elapsed_s={settled['settle_elapsed_s']:.3f}",
                    flush=True,
                )
                if self.args.post_settle_s > 0:
                    self.sleep(float(self.args.post_settle_s))
                return settled
            self.sleep(1.0 / float(self.args.observe_hz))
        raise TimeoutError(f"{arm_name} passive settle timeout after {self.args.motion_timeout_s:g}s")

    def move_pair(self, left_target: list[float], right_target: list[float], *, step: int) -> bool:
        self._move_one("left_arm", left_target, step=step)
        self._move_one("right_arm", right_target, step=step)
        return True


def _append_mvp_fault(log_path: Path, exc: BaseException) -> None:
    with contextlib.suppress(Exception):
        log_path = log_path.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "FAULT",
                        "timestamp_monotonic_ns": time.monotonic_ns(),
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class RuntimeResources:
    """Idempotent ordered cleanup: loop/callbacks/executor/node, SDK, rclpy."""

    def __init__(self, rclpy_owner: Any) -> None:
        self.rclpy_owner = rclpy_owner
        self.policy_loop: Any | None = None
        self.camera: Any | None = None
        self.sdk: Any | None = None
        self._close_lock = threading.Lock()
        self.closed = False

    def close(self) -> None:
        with self._close_lock:
            if self.closed:
                return
            self.closed = True
            if self.policy_loop is not None:
                with contextlib.suppress(Exception):
                    self.policy_loop.request_stop()
            if self.camera is not None:
                with contextlib.suppress(Exception):
                    self.camera.close()
            if self.sdk is not None:
                with contextlib.suppress(Exception):
                    self.sdk.close()
            with contextlib.suppress(Exception):
                self.rclpy_owner.close()


def validate_contract_file(path: Path) -> None:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("camera_count") != 1 or contract.get("state_dim") != 26:
        raise RuntimeFault("model contract camera/state mismatch")
    if contract.get("action_dim") != 28 or contract.get("chunk_size") != 10:
        raise RuntimeFault("model contract action/chunk mismatch")


def _initialize_live(
    args: argparse.Namespace,
    config: ActCConfig,
    *,
    include_hands: bool,
) -> tuple[RuntimeResources, TopCameraReader, RuntimeStateReader, OnnxPolicy, SdkDeviceBundle]:
    import rclpy

    validate_contract_file(config.contract)
    policy = OnnxPolicy(config.model, threads=args.threads)
    owner = RclpyOwner(rclpy)
    resources = RuntimeResources(owner)
    try:
        topic_pairs = discover_topic_pairs(rclpy, min(float(args.ready_timeout), 5.0))
        camera_topic = select_camera_topic(topic_pairs, config.camera_topic)
        print(f"Selected TOP: {camera_topic}", flush=True)
        camera = TopCameraReader(rclpy, topic=camera_topic)
        resources.camera = camera
        sdk = SdkDeviceBundle(config, include_hands=include_hands)
        resources.sdk = sdk
        assert sdk.left_arm is not None and sdk.right_arm is not None
        state_source = SdkArmStateSource(sdk.left_arm, sdk.right_arm)
        state_reader = RuntimeStateReader(state_source)
        deadline = time.monotonic() + float(args.ready_timeout)
        camera.wait_ready(max(0.0, deadline - time.monotonic()))
        print("Camera READY", flush=True)
        state_reader.wait_ready(max(0.0, deadline - time.monotonic()))
        state, _ = state_reader.snapshot()
        if not np.isfinite(state).all():
            raise RuntimeFault("state26 is not finite at readiness gate")
        print("State26 READY", flush=True)
        return resources, camera, state_reader, policy, sdk
    except BaseException:
        resources.close()
        raise


def run_live_dry(args: argparse.Namespace, config: ActCConfig) -> dict[str, Any]:
    resources: RuntimeResources | None = None
    try:
        resources, camera, state_reader, policy, _sdk = _initialize_live(
            args, config, include_hands=False
        )
        runtime = FixedPointCRuntime(camera, state_reader, policy, rate_hz=args.rate)
        print("ACT C FIXED-POINT POLICY", flush=True)
        print(f"Model: {config.model}", flush=True)
        print(f"Policy rate: {args.rate:g} Hz", flush=True)
        print("Mode: DRY_RUN (SDK state reads only; no robot commands)", flush=True)
        return runtime.run_dry(args.duration, config.dry_run_log)
    finally:
        if resources is not None:
            resources.close()


def run_live_mvp(args: argparse.Namespace, config: ActCConfig) -> dict[str, Any]:
    resources: RuntimeResources | None = None
    try:
        resources, camera, state_reader, policy, sdk = _initialize_live(
            args, config, include_hands=True
        )
        assert sdk.left_arm is not None and sdk.right_arm is not None
        assert sdk.left_hand is not None and sdk.right_hand is not None
        state_source = state_reader.passive
        arm_executor = SerialArmExecutor(
            state_source,
            sdk.left_arm,
            sdk.right_arm,
            args,
            joint_limit_margin_rad=config.joint_limit_margin_rad,
        )

        def right_clench(target: list[float], blocking: bool) -> Any:
            return sdk.right_hand.clench(*target, blocking=blocking)

        def right_force(blocking: bool) -> Any:
            return sdk.right_hand.grasp_force(**GRASP_FORCE_CONFIG, blocking=blocking)

        def left_clench(target: list[float], blocking: bool) -> Any:
            return sdk.left_hand.clench(*target, blocking=blocking)

        def left_force(blocking: bool) -> Any:
            return sdk.left_hand.grasp_force(**GRASP_FORCE_CONFIG, blocking=blocking)

        print("ACT C SERIAL RECEDING-HORIZON MVP", flush=True)
        print("This is NOT 5Hz streaming.", flush=True)
        print(f"Arm target max delta: {config.max_delta_rad:.2f} rad", flush=True)
        print(f"Joint limit margin: {config.joint_limit_margin_rad:.3f} rad", flush=True)
        print("Mode control: constrained hysteresis", flush=True)
        print("Execution: serial move + passive settle", flush=True)
        runtime = SerialRecedingHorizonMvp(
            camera,
            state_reader,
            policy,
            arm_executor,
            HybridHandExecutor(left_clench, left_force),
            HybridHandExecutor(right_clench, right_force),
            max_delta_per_step=config.max_delta_rad,
            joint_limit_margin_rad=config.joint_limit_margin_rad,
        )
        resources.policy_loop = runtime
        return runtime.run(config.execute_log)
    except BaseException as exc:
        _append_mvp_fault(config.execute_log, exc)
        raise
    finally:
        if resources is not None:
            resources.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--execute-mvp", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=POLICY_HZ)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--observe-hz", type=float, default=5.0)
    parser.add_argument("--start-threshold-rad", type=float, default=0.003)
    parser.add_argument("--settle-threshold-rad", type=float, default=0.002)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument("--post-settle-s", type=float, default=0.2)
    parser.add_argument("--motion-timeout-s", type=float, default=20.0)
    parser.add_argument("--no-motion-grace-s", type=float, default=1.0)
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
        config = load_config(args.config, repo_root=PROJECT_ROOT)
        if args.execute_mvp:
            run_live_mvp(args, config)
        else:
            run_live_dry(args, config)
        return 0
    except BaseException as exc:
        print(f"FAULT: {exc!r}", file=sys.stderr, flush=True)
        return 1
