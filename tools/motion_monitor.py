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
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "motion_monitor"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("motion_monitor_config.json")

DEFAULT_CONFIG: dict[str, float] = {
    "sample_hz": 10.0,
    "joint_error_threshold_rad": 0.15,
    "position_error_threshold_m": 0.03,
    "orientation_error_threshold_deg": 15.0,
    "target_joint_jump_threshold_rad": 1.2,
    "target_position_jump_threshold_m": 0.25,
    "target_orientation_jump_threshold_deg": 45.0,
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


def read_method(device: Any, names: tuple[str, ...]) -> tuple[list[float] | None, str | None]:
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
            "target_joint": list(target_joint) if target_joint is not None else None,
            "target_ee_pose": list(target_ee_pose) if target_ee_pose is not None else None,
            "initial_joint": None,
            "initial_ee_pose": None,
        }
        self._samples = []
        self._stop_event.clear()
        self.output_root.mkdir(parents=True, exist_ok=True)
        phase_dir.mkdir(parents=True, exist_ok=False)
        self._active_arm = arm
        first_sample = self._read_sample(arm, None, None, time.monotonic())
        with self._lock:
            self._samples.append(first_sample)
        self._thread = threading.Thread(target=self._sample_loop, args=(arm,), daemon=True)
        self._thread.start()
        print(f"【MotionMonitor】开始：{phase_name} -> {display_path(phase_dir)}", flush=True)

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
            "diagnosis": diagnosis["diagnosis"],
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
        actual_joint, joint_read_error = read_method(arm, ("get_joint_angles", "get_joints", "get_qpos"))
        actual_ee_pose, pose_read_error = read_method(arm, ("get_pose", "get_ee_pose", "get_end_effector_pose"))
        if self._active["initial_joint"] is None and actual_joint is not None:
            self._active["initial_joint"] = list(actual_joint)
        if self._active["initial_ee_pose"] is None and actual_ee_pose is not None:
            self._active["initial_ee_pose"] = list(actual_ee_pose)
        dt_s = None if prev_monotonic is None else sample_monotonic - prev_monotonic
        sample = {
            "timestamp": now_text(),
            "monotonic_s": sample_monotonic,
            "target_joint": target_joint,
            "actual_joint": actual_joint,
            "joint_error": vector_error(target_joint, actual_joint),
            "max_joint_error": max_abs_error(target_joint, actual_joint),
            "target_ee_pose": target_ee_pose,
            "actual_ee_pose": actual_ee_pose,
            "position_error": euclidean_error(target_ee_pose, actual_ee_pose, 0, 3),
            "orientation_error": orientation_error_deg(target_ee_pose, actual_ee_pose),
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
        initial_joint = active.get("initial_joint")
        target_joint = active.get("target_joint")
        initial_ee_pose = active.get("initial_ee_pose")
        target_ee_pose = active.get("target_ee_pose")
        target_joint_delta = max_abs_error(target_joint, initial_joint)
        target_position_delta = euclidean_error(target_ee_pose, initial_ee_pose, 0, 3)
        target_orientation_delta = orientation_error_deg(target_ee_pose, initial_ee_pose)

        joint_threshold = float(self.config["joint_error_threshold_rad"])
        position_threshold = float(self.config["position_error_threshold_m"])
        orientation_threshold = float(self.config["orientation_error_threshold_deg"])
        joint_jump_threshold = float(self.config["target_joint_jump_threshold_rad"])
        position_jump_threshold = float(self.config["target_position_jump_threshold_m"])
        orientation_jump_threshold = float(self.config["target_orientation_jump_threshold_deg"])

        reasons: list[str] = []
        diagnosis = "NORMAL"
        if max_joint_error is not None and max_joint_error > joint_threshold:
            diagnosis = "JOINT_TRACKING_ERROR"
            reasons.append(f"max_joint_error {max_joint_error:.4f} rad > {joint_threshold:.4f} rad")
        elif (
            (target_joint_delta is not None and target_joint_delta > joint_jump_threshold)
            or (target_position_delta is not None and target_position_delta > position_jump_threshold)
            or (target_orientation_delta is not None and target_orientation_delta > orientation_jump_threshold)
        ):
            diagnosis = "POSSIBLE_PATH_OR_IK_PROBLEM"
            reasons.append("target change from initial state exceeded configured jump threshold")
        elif (
            (max_position_error is not None and max_position_error > position_threshold)
            or (max_orientation_error is not None and max_orientation_error > orientation_threshold)
        ):
            diagnosis = "EE_TRACKING_ERROR"
            reasons.append("joint error normal/unknown, but EE pose error exceeded threshold")

        sample_error_count = sum(1 for sample in samples if sample.get("sample_errors"))
        return {
            "phase_name": active["phase_name"],
            "command_method": active.get("command_method", ""),
            "status": active.get("status", ""),
            "error_info": active.get("error_info", ""),
            "diagnosis": diagnosis,
            "reasons": reasons,
            "sample_count": len(samples),
            "sample_error_count": sample_error_count,
            "max_joint_error": max_joint_error,
            "max_position_error": max_position_error,
            "max_orientation_error": max_orientation_error,
            "target_joint_delta_from_start": target_joint_delta,
            "target_position_delta_from_start": target_position_delta,
            "target_orientation_delta_from_start": target_orientation_delta,
            "thresholds": {
                "joint_error_rad": joint_threshold,
                "position_error_m": position_threshold,
                "orientation_error_deg": orientation_threshold,
                "target_joint_jump_rad": joint_jump_threshold,
                "target_position_jump_m": position_jump_threshold,
                "target_orientation_jump_deg": orientation_jump_threshold,
            },
        }
