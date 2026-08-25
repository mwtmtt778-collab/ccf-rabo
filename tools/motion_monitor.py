#!/usr/bin/env python3
"""Optional motion diagnostic monitor for Rabo arm commands.

The monitor is intentionally passive: it only polls read-only SDK methods while
the caller executes the original motion command.  It never changes targets,
timeouts, controller parameters, or robot state.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "motion_monitor"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("motion_monitor_config.json")

DEFAULT_CONFIG: dict[str, float] = {
    "sample_hz": 10.0,
    "joint_error_threshold_rad": 0.15,
    "position_error_threshold_m": 0.03,
    "orientation_error_threshold_deg": 15.0,
    "target_joint_jump_threshold_rad": 1.0,
    "target_position_jump_threshold_m": 0.25,
    "target_orientation_jump_threshold_deg": 45.0,
    "actual_joint_jump_threshold_rad": 0.35,
    "divergence_growth_threshold_rad": 0.10,
    "recent_history_s": 2.0,
    "action_timeout_s": 10.0,
    "ready_joint_error_threshold_rad": 0.10,
    "ready_stability_delta_threshold_rad": 0.02,
    "ready_sample_count": 3.0,
    "ready_sample_interval_s": 0.2,
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return repr(value)


def to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        if all(key in value for key in pose_keys):
            return [float(value[key]) for key in pose_keys]
        joint_keys = ("position", "positions", "joint_angles", "joints", "qpos")
        for key in joint_keys:
            if key in value:
                return to_float_list(value[key])
        return None
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple, dict)):
                return None
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return None
        return out
    return None


def is_numeric_sequence(value: Any, length: int | None = None) -> bool:
    values = to_float_list(value)
    if values is None:
        return False
    return length is None or len(values) == length


def matrix_to_rpy(rotation_matrix: list[list[float]]) -> list[float]:
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]
    sy = math.sqrt(r11 * r11 + r21 * r21)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(r32, r33)
        pitch = math.atan2(-r31, sy)
        yaw = math.atan2(r21, r11)
    else:
        roll = math.atan2(-r23, r22)
        pitch = math.atan2(-r31, sy)
        yaw = 0.0
    return [roll, pitch, yaw]


def parse_robot_pose(value: Any) -> list[float] | None:
    """Parse SDK get_pose result into [x, y, z, roll, pitch, yaw].

    Supported formats:
    - [x, y, z, roll, pitch, yaw]
    - {"x": ..., "y": ..., "z": ..., "roll": ..., "pitch": ..., "yaw": ...}
    - [[x, y, z], [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]]
    - {"position": [x, y, z], "rotation_matrix": [[...], [...], [...]]}
    """
    flat = to_float_list(value)
    if flat is not None and len(flat) >= 6:
        return [float(item) for item in flat[:6]]
    if isinstance(value, dict):
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        if all(key in value for key in pose_keys):
            return [float(value[key]) for key in pose_keys]
        position = value.get("position") or value.get("translation") or value.get("pos")
        rotation = (
            value.get("rotation_matrix")
            or value.get("rotation")
            or value.get("orientation_matrix")
            or value.get("matrix")
        )
        parsed_position = to_float_list(position)
        if parsed_position is not None and len(parsed_position) >= 3 and is_rotation_matrix(rotation):
            return [*parsed_position[:3], *matrix_to_rpy(rotation)]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parsed_position = to_float_list(value[0])
        rotation = value[1]
        if parsed_position is not None and len(parsed_position) >= 3 and is_rotation_matrix(rotation):
            return [*parsed_position[:3], *matrix_to_rpy(rotation)]
    return None


def is_rotation_matrix(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    for row in value:
        if not is_numeric_sequence(row, 3):
            return False
    return True


def read_joint_method(device: Any, names: tuple[str, ...]) -> tuple[list[float] | None, str | None]:
    for name in names:
        if not hasattr(device, name):
            continue
        try:
            value = getattr(device, name)()
            parsed = to_float_list(value)
            if parsed is not None:
                return parsed, None
            return None, f"{name} returned non-numeric value: {jsonable(value)!r}"
        except Exception as exc:
            return None, f"{name} failed: {exc!r}"
    return None, f"no readable method found: {', '.join(names)}"


def read_pose_method(device: Any, names: tuple[str, ...]) -> tuple[list[float] | None, str | None]:
    for name in names:
        if not hasattr(device, name):
            continue
        try:
            value = getattr(device, name)()
            parsed = parse_robot_pose(value)
            if parsed is not None:
                return parsed, None
            return None, f"{name} returned unsupported pose value: {jsonable(value)!r}"
        except Exception as exc:
            return None, f"{name} failed: {exc!r}"
    return None, f"no readable pose method found: {', '.join(names)}"


def extract_joint_target(value: Any, expected_len: int = 7) -> list[float] | None:
    """Extract a target joint vector from SDK planning/pose_check returns.

    This is intentionally conservative: it only accepts numeric vectors that
    are explicitly returned by the SDK/control layer, never generated by the
    monitor itself.
    """
    values = to_float_list(value)
    if values is not None and len(values) == expected_len:
        return values
    if isinstance(value, dict):
        preferred_keys = (
            "target_joint",
            "target_joints",
            "joint_target",
            "joint_targets",
            "joint_angles",
            "joints",
            "qpos",
            "solution",
            "ik_solution",
        )
        for key in preferred_keys:
            if key in value:
                found = extract_joint_target(value[key], expected_len=expected_len)
                if found is not None:
                    return found
        for item in value.values():
            found = extract_joint_target(item, expected_len=expected_len)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = extract_joint_target(item, expected_len=expected_len)
            if found is not None:
                return found
    return None


def max_abs_error(target: list[float] | None, actual: list[float] | None) -> float | None:
    if not target or not actual:
        return None
    count = min(len(target), len(actual))
    if count == 0:
        return None
    return max(abs(float(target[index]) - float(actual[index])) for index in range(count))


def vector_error(target: list[float] | None, actual: list[float] | None) -> list[float] | None:
    if not target or not actual:
        return None
    count = min(len(target), len(actual))
    if count == 0:
        return None
    return [float(target[index]) - float(actual[index]) for index in range(count)]


def euclidean_error(target: list[float] | None, actual: list[float] | None, start: int, end: int) -> float | None:
    if not target or not actual or len(target) < end or len(actual) < end:
        return None
    return math.sqrt(sum((float(target[index]) - float(actual[index])) ** 2 for index in range(start, end)))


def angle_wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def orientation_error_deg(target: list[float] | None, actual: list[float] | None) -> float | None:
    if not target or not actual or len(target) < 6 or len(actual) < 6:
        return None
    diff = [angle_wrap(float(target[index]) - float(actual[index])) for index in range(3, 6)]
    return math.degrees(math.sqrt(sum(item * item for item in diff)))


def joint_velocity(
    prev_joint: list[float] | None,
    actual_joint: list[float] | None,
    dt_s: float,
) -> list[float] | None:
    if not prev_joint or not actual_joint or dt_s <= 0:
        return None
    count = min(len(prev_joint), len(actual_joint))
    if count == 0:
        return None
    return [(float(actual_joint[index]) - float(prev_joint[index])) / dt_s for index in range(count)]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "UNKNOWN_PHASE"


def unique_phase_dir(output_root: Path, phase_name: str) -> Path:
    date_prefix = datetime.now().astimezone().strftime("%Y%m%d")
    base = output_root / f"{date_prefix}_{safe_name(phase_name)}"
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = output_root / f"{date_prefix}_{safe_name(phase_name)}_{index}"
        if not candidate.exists():
            return candidate
    timestamp = datetime.now().astimezone().strftime("%H%M%S_%f")
    return output_root / f"{date_prefix}_{timestamp}_{safe_name(phase_name)}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, float]:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, value in raw.items():
            if key in config:
                config[key] = float(value)
    return config


def max_target_joint_step(samples: Iterable[dict[str, Any]]) -> float | None:
    previous: list[float] | None = None
    max_step: float | None = None
    for sample in samples:
        target = sample.get("target_joint")
        if not isinstance(target, list):
            continue
        if previous is not None:
            step = max_abs_error(previous, target)
            if step is not None:
                max_step = step if max_step is None else max(max_step, step)
        previous = target
    return max_step


class MotionMonitor:
    """Passive 10Hz motion monitor with per-phase JSON output."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_root = output_root
        self.config_path = config_path
        self.config = load_config(config_path)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: dict[str, Any] | None = None
        self._active_arm: Any | None = None
        self._samples: list[dict[str, Any]] = []
        self._failure_buffer: deque[dict[str, Any]] = deque(maxlen=self._ring_buffer_size())

    def _ring_buffer_size(self) -> int:
        sample_hz = max(float(self.config.get("sample_hz", 10.0)), 0.1)
        history_s = max(float(self.config.get("recent_history_s", 2.0)), 0.1)
        return max(1, int(round(sample_hz * history_s)))

    def start_motion_monitor(
        self,
        phase_name: str,
        arm: Any,
        *,
        target_joint: list[float] | None = None,
        target_ee_pose: list[float] | None = None,
        command_method: str = "",
    ) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop_motion_monitor(status="FORCED_STOP_BEFORE_NEW_PHASE")

        phase_dir = unique_phase_dir(self.output_root, phase_name)
        started_at = now_text()
        self._active = {
            "phase_name": phase_name,
            "command_method": command_method,
            "phase_dir": phase_dir,
            "started_at": started_at,
            "started_monotonic_s": time.monotonic(),
            "target_joint": list(target_joint) if target_joint is not None else None,
            "target_joint_source": "caller" if target_joint is not None else "",
            "target_ee_pose": list(target_ee_pose) if target_ee_pose is not None else None,
            "initial_joint": None,
            "initial_ee_pose": None,
        }
        self._samples = []
        self._failure_buffer = deque(maxlen=self._ring_buffer_size())
        self._stop_event.clear()
        self.output_root.mkdir(parents=True, exist_ok=True)
        phase_dir.mkdir(parents=True, exist_ok=False)
        self._active_arm = arm
        first_sample = self._read_sample(arm, None, None, time.monotonic())
        with self._lock:
            self._samples.append(first_sample)
            self._failure_buffer.append(self._snapshot_sample(first_sample))
        self._thread = threading.Thread(target=self._sample_loop, args=(arm,), daemon=True)
        self._thread.start()
        print(f"【MotionMonitor】开始：{phase_name} -> {display_path(phase_dir)}", flush=True)

    def set_target_joint(self, target_joint: list[float] | None, *, source: str = "caller") -> None:
        if not self.enabled or target_joint is None:
            return
        with self._lock:
            if self._active is not None:
                self._active["target_joint"] = [float(value) for value in target_joint]
                self._active["target_joint_source"] = source

    def stop_motion_monitor(self, *, status: str = "COMPLETED", error_info: str = "") -> dict[str, Any] | None:
        if not self.enabled or self._active is None:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._active_arm is not None:
            final_sample = self._read_sample(self._active_arm, None, None, time.monotonic())
            final_sample["final_sample"] = True
            with self._lock:
                self._samples.append(final_sample)
                self._failure_buffer.append(self._snapshot_sample(final_sample))
        self._active_arm = None

        with self._lock:
            samples = list(self._samples)
        active = dict(self._active)
        active["finished_at"] = now_text()
        active["status"] = status
        active["error_info"] = error_info
        diagnosis = self._build_diagnosis(active, samples)

        phase_dir = active["phase_dir"]
        trajectory_path = phase_dir / "trajectory.json"
        diagnosis_path = phase_dir / "diagnosis.json"
        failure_snapshot_path = phase_dir / "failure_snapshot.json"
        trajectory_path.write_text(
            json.dumps(
                jsonable(
                    {
                        "metadata": {
                            key: value
                            for key, value in active.items()
                            if key not in {"phase_dir"}
                        },
                        "config": self.config,
                        "samples": samples,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        diagnosis_path.write_text(
            json.dumps(jsonable(diagnosis), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if status != "COMPLETED":
            failure_snapshot = {
                "phase": active["phase_name"],
                "status": status,
                "diagnosis": diagnosis["diagnosis"],
                "trajectory": list(self._failure_buffer),
            }
            failure_snapshot_path.write_text(
                json.dumps(jsonable(failure_snapshot), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._active = None

        print(
            f"【MotionMonitor】完成：{active['phase_name']} diagnosis={diagnosis['diagnosis']} "
            f"dir={display_path(phase_dir)}",
            flush=True,
        )
        return {
            "phase_name": active["phase_name"],
            "phase_dir": display_path(phase_dir),
            "trajectory_path": display_path(trajectory_path),
            "diagnosis_path": display_path(diagnosis_path),
            "failure_snapshot_path": display_path(failure_snapshot_path) if status != "COMPLETED" else "",
            "diagnosis": diagnosis["diagnosis"],
            "diagnosis_detail": diagnosis,
            "metrics": diagnosis["metrics"],
            "recent_joint_history": list(self._failure_buffer),
        }

    def _sample_loop(self, arm: Any) -> None:
        interval_s = 1.0 / max(float(self.config.get("sample_hz", 10.0)), 0.1)
        prev_joint: list[float] | None = None
        prev_monotonic: float | None = None
        while not self._stop_event.is_set():
            sample_monotonic = time.monotonic()
            sample = self._read_sample(arm, prev_joint, prev_monotonic, sample_monotonic)
            actual_joint = sample.get("actual_joint")
            if isinstance(actual_joint, list):
                prev_joint = actual_joint
                prev_monotonic = sample_monotonic
            with self._lock:
                self._samples.append(sample)
                self._failure_buffer.append(self._snapshot_sample(sample))
            self._stop_event.wait(interval_s)

    def _read_sample(
        self,
        arm: Any,
        prev_joint: list[float] | None,
        prev_monotonic: float | None,
        sample_monotonic: float,
    ) -> dict[str, Any]:
        assert self._active is not None
        target_joint = self._active["target_joint"]
        target_ee_pose = self._active["target_ee_pose"]
        actual_joint, joint_read_error = read_joint_method(arm, ("get_joint_angles", "get_joints", "get_qpos"))
        actual_ee_pose, pose_read_error = read_pose_method(arm, ("get_pose", "get_ee_pose", "get_end_effector_pose"))
        if self._active["initial_joint"] is None and actual_joint is not None:
            self._active["initial_joint"] = list(actual_joint)
        if self._active["initial_ee_pose"] is None and actual_ee_pose is not None:
            self._active["initial_ee_pose"] = list(actual_ee_pose)
        dt_s = None if prev_monotonic is None else sample_monotonic - prev_monotonic
        elapsed_s = sample_monotonic - float(self._active.get("started_monotonic_s", sample_monotonic))
        joint_err = vector_error(target_joint, actual_joint)
        max_joint_err = max_abs_error(target_joint, actual_joint)
        position_err = euclidean_error(target_ee_pose, actual_ee_pose, 0, 3)
        orientation_err = orientation_error_deg(target_ee_pose, actual_ee_pose)
        sample_target_joint = list(target_joint) if isinstance(target_joint, list) else None
        sample_actual_joint = list(actual_joint) if isinstance(actual_joint, list) else None
        sample_target_ee_pose = list(target_ee_pose) if isinstance(target_ee_pose, list) else None
        sample_actual_ee_pose = list(actual_ee_pose) if isinstance(actual_ee_pose, list) else None
        sample = {
            "timestamp": now_text(),
            "time": round(elapsed_s, 4),
            "monotonic_s": sample_monotonic,
            "target_joint": sample_target_joint,
            "actual_joint": sample_actual_joint,
            "joint_error": joint_err,
            "max_joint_error": max_joint_err,
            "target_ee_pose": sample_target_ee_pose,
            "actual_ee_pose": sample_actual_ee_pose,
            "position_error": position_err,
            "orientation_error": orientation_err,
            "joint_velocity": joint_velocity(prev_joint, actual_joint, dt_s or 0.0),
        }
        errors = []
        if joint_read_error:
            errors.append(joint_read_error)
        if pose_read_error:
            errors.append(pose_read_error)
        if errors:
            sample["sample_errors"] = errors
        return sample

    def _snapshot_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "time": sample.get("time"),
            "target_joint": sample.get("target_joint"),
            "actual_joint": sample.get("actual_joint"),
            "joint_error": sample.get("joint_error"),
            "max_joint_error": sample.get("max_joint_error"),
            "ee_error": {
                "target_ee_pose": sample.get("target_ee_pose"),
                "actual_ee_pose": sample.get("actual_ee_pose"),
                "position_error": sample.get("position_error"),
                "orientation_error": sample.get("orientation_error"),
            },
        }

    def _build_diagnosis(self, active: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
        max_joint_error = max(
            (float(sample["max_joint_error"]) for sample in samples if sample.get("max_joint_error") is not None),
            default=None,
        )
        max_position_error = max(
            (float(sample["position_error"]) for sample in samples if sample.get("position_error") is not None),
            default=None,
        )
        max_orientation_error = max(
            (float(sample["orientation_error"]) for sample in samples if sample.get("orientation_error") is not None),
            default=None,
        )
        final_position_error = next(
            (float(sample["position_error"]) for sample in reversed(samples) if sample.get("position_error") is not None),
            None,
        )
        final_orientation_error = next(
            (float(sample["orientation_error"]) for sample in reversed(samples) if sample.get("orientation_error") is not None),
            None,
        )
        final_actual_ee_pose = next(
            (list(sample["actual_ee_pose"]) for sample in reversed(samples) if isinstance(sample.get("actual_ee_pose"), list)),
            None,
        )
        initial_joint = active.get("initial_joint")
        target_joint = active.get("target_joint")
        initial_ee_pose = active.get("initial_ee_pose")
        target_ee_pose = active.get("target_ee_pose")
        target_joint_delta = max_abs_error(target_joint, initial_joint)
        target_position_delta = euclidean_error(target_ee_pose, initial_ee_pose, 0, 3)
        target_orientation_delta = orientation_error_deg(target_ee_pose, initial_ee_pose)
        max_target_step = max_target_joint_step(samples)
        actual_joint_samples = [
            sample["actual_joint"] for sample in samples if isinstance(sample.get("actual_joint"), list)
        ]
        actual_joint_steps = [
            max_abs_error(previous, current)
            for previous, current in zip(actual_joint_samples, actual_joint_samples[1:])
        ]
        max_actual_joint_jump = max((value for value in actual_joint_steps if value is not None), default=None)
        final_actual_joint = actual_joint_samples[-1] if actual_joint_samples else None
        final_joint_error = max_abs_error(target_joint, final_actual_joint)
        joint_error_series = [
            float(sample["max_joint_error"])
            for sample in samples
            if sample.get("max_joint_error") is not None
        ]
        divergence_growth = (
            joint_error_series[-1] - min(joint_error_series)
            if len(joint_error_series) >= 3
            else None
        )

        joint_threshold = float(self.config["joint_error_threshold_rad"])
        position_threshold = float(self.config["position_error_threshold_m"])
        orientation_threshold = float(self.config["orientation_error_threshold_deg"])
        joint_jump_threshold = float(self.config["target_joint_jump_threshold_rad"])
        position_jump_threshold = float(self.config["target_position_jump_threshold_m"])
        orientation_jump_threshold = float(self.config["target_orientation_jump_threshold_deg"])
        actual_joint_jump_threshold = float(self.config["actual_joint_jump_threshold_rad"])
        divergence_growth_threshold = float(self.config["divergence_growth_threshold_rad"])

        sample_count = len(samples)
        samples_with_target_joint = sum(1 for sample in samples if isinstance(sample.get("target_joint"), list))
        samples_with_actual_joint = sum(1 for sample in samples if isinstance(sample.get("actual_joint"), list))
        samples_with_actual_ee = sum(1 for sample in samples if isinstance(sample.get("actual_ee_pose"), list))
        sample_error_count = sum(1 for sample in samples if sample.get("sample_errors"))
        reasons: list[str] = []
        suspect = ""
        diagnosis = "NORMAL"

        command_method = active.get("command_method")
        target_joint_required = command_method in {"execute_checked.move_joints", "move_joints"}
        ee_required = command_method in {"move_checked", "move_to"}
        if (
            sample_count == 0
            or (target_joint_required and samples_with_target_joint == 0)
            or samples_with_actual_joint == 0
        ):
            diagnosis = "MONITOR_DATA_INCOMPLETE"
            suspect = "monitor"
            if sample_count == 0:
                reasons.append("no monitor samples collected")
            if target_joint_required and samples_with_target_joint == 0:
                reasons.append("target_joint missing; SDK did not expose a joint target for this command")
            if samples_with_actual_joint == 0:
                reasons.append("actual_joint missing; get_joint_angles was unavailable or unreadable")
        elif ee_required and samples_with_actual_ee == 0:
            diagnosis = "CARTESIAN_ENDPOINT_FEEDBACK_UNAVAILABLE"
            suspect = "endpoint_feedback"
            reasons.append("get_pose was unavailable or unparseable")
        elif max_actual_joint_jump is not None and max_actual_joint_jump > actual_joint_jump_threshold:
            diagnosis = "ACTUAL_JOINT_JUMP"
            suspect = "controller_or_state"
            reasons.append(
                f"max_actual_joint_jump {max_actual_joint_jump:.4f} rad > "
                f"{actual_joint_jump_threshold:.4f} rad"
            )
        elif max_target_step is not None and max_target_step > joint_jump_threshold:
            diagnosis = "POSSIBLE_IK_OR_PATH_JUMP"
            suspect = "path"
            reasons.append(f"target_joint adjacent step {max_target_step:.4f} rad > {joint_jump_threshold:.4f} rad")
        elif (
            divergence_growth is not None
            and divergence_growth > divergence_growth_threshold
            and final_joint_error is not None
            and final_joint_error > joint_threshold
        ):
            diagnosis = "JOINT_DIVERGENCE"
            suspect = "controller"
            reasons.append(
                f"joint error grew {divergence_growth:.4f} rad and final error "
                f"{final_joint_error:.4f} rad exceeds {joint_threshold:.4f} rad"
            )
        elif final_joint_error is not None and final_joint_error > joint_threshold:
            diagnosis = "JOINT_TRACKING_ERROR"
            suspect = "controller"
            reasons.append(f"final_joint_error {final_joint_error:.4f} rad > {joint_threshold:.4f} rad")
        elif final_position_error is not None and final_position_error > position_threshold:
            diagnosis = "EE_POSITION_ERROR"
            suspect = "controller_or_kinematics"
            reasons.append(f"final_position_error {final_position_error:.4f} m > {position_threshold:.4f} m")
        elif final_orientation_error is not None and final_orientation_error > orientation_threshold:
            diagnosis = "EE_ORIENTATION_ERROR"
            suspect = "controller_or_kinematics"
            reasons.append(f"final_orientation_error {final_orientation_error:.4f} deg > {orientation_threshold:.4f} deg")

        return {
            "phase": active["phase_name"],
            "command_method": active.get("command_method", ""),
            "status": active.get("status", ""),
            "error_info": active.get("error_info", ""),
            "diagnosis": diagnosis,
            "suspect": suspect,
            "reasons": reasons,
            "metrics": {
                "sample_count": sample_count,
                "samples_with_target_joint": samples_with_target_joint,
                "samples_with_actual_joint": samples_with_actual_joint,
                "samples_with_actual_ee_pose": samples_with_actual_ee,
                "sample_error_count": sample_error_count,
                "max_joint_error": max_joint_error,
                "max_position_error": max_position_error,
                "max_orientation_error": max_orientation_error,
                "final_position_error": final_position_error,
                "final_orientation_error": final_orientation_error,
                "final_actual_ee_pose": final_actual_ee_pose,
                "max_target_joint_step": max_target_step,
                "max_actual_joint_jump": max_actual_joint_jump,
                "final_joint_error": final_joint_error,
                "divergence_growth": divergence_growth,
                "diverged": diagnosis == "JOINT_DIVERGENCE",
                "joint_target_available": target_joint is not None,
                "joint_target_based_divergence_available": target_joint is not None,
                "cartesian_endpoint_feedback_available": final_actual_ee_pose is not None,
                "target_joint_delta_from_start": target_joint_delta,
                "target_position_delta_from_start": target_position_delta,
                "target_orientation_delta_from_start": target_orientation_delta,
            },
            "thresholds": {
                "joint_error_rad": joint_threshold,
                "position_error_m": position_threshold,
                "orientation_error_deg": orientation_threshold,
                "target_joint_jump_rad": joint_jump_threshold,
                "target_position_jump_m": position_jump_threshold,
                "target_orientation_jump_deg": orientation_jump_threshold,
                "actual_joint_jump_rad": actual_joint_jump_threshold,
                "divergence_growth_rad": divergence_growth_threshold,
            },
        }


def evaluate_motion_result(
    monitor_result: dict[str, Any] | None,
    *,
    timeout: bool = False,
    sdk_return_ok: bool = True,
) -> dict[str, Any]:
    """Convert the existing monitor diagnosis into a fail-closed motion gate."""
    if monitor_result is None:
        return {
            "pass": False,
            "reason": "motion_monitor_result_missing",
            "sdk_return_ok": bool(sdk_return_ok),
            "timeout": bool(timeout),
            "diverged": False,
            "max_joint_error": None,
            "final_joint_error": None,
            "max_joint_jump": None,
            "diagnosis": "MONITOR_DATA_INCOMPLETE",
        }
    metrics = monitor_result.get("metrics") or {}
    diagnosis = str(monitor_result.get("diagnosis") or "MONITOR_DATA_INCOMPLETE")
    passed = bool(sdk_return_ok) and not timeout and diagnosis == "NORMAL"
    reasons = []
    if not sdk_return_ok:
        reasons.append("sdk_return_failed")
    if timeout:
        reasons.append("action_timeout")
    if diagnosis != "NORMAL":
        reasons.append(diagnosis)
    return {
        "pass": passed,
        "reason": "; ".join(reasons) if reasons else "normal",
        "sdk_return_ok": bool(sdk_return_ok),
        "timeout": bool(timeout),
        "diverged": bool(metrics.get("diverged", diagnosis == "JOINT_DIVERGENCE")),
        "max_joint_error": metrics.get("max_joint_error"),
        "final_joint_error": metrics.get("final_joint_error"),
        "max_joint_jump": metrics.get("max_actual_joint_jump"),
        "diagnosis": diagnosis,
        "metrics": metrics,
    }
