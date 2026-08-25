#!/usr/bin/env python3
"""Fail-closed Episode reset validation for the Rabo three-Nut task.

This tool does not treat a successful SDK return as proof of reset.  Robot and
hand commands are followed by repeated joint-state reads; Nut SetEntityPose
commands are followed by repeated fresh top-camera point-cloud observations.

The currently documented SDK does not expose a verified rigid-body velocity
reset/read API or a verified object-pose read API.  Those gaps are reported as
SOFT_RESET_LIMITATION and keep ``overall_reset_ready`` false.  This is
intentional: ACT batch collection must not start on an unverified reset.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "reset_validation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    KNOWN_FIXED_NUT_WORLD_POSE,
    KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
    LEFT_PRE_JOINTS,
    NUT_IDS,
    WORLD_ID,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_SCENE_UI  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import RIGHT_OBSERVATION_JOINTS  # noqa: E402
from tools.test_nut_camera_calibration import set_pose_with_retry  # noqa: E402
from tools import test_pointcloud_nut_detection as nut_detector  # noqa: E402


RESET_STATES = (
    "RESET_BEGIN",
    "RESET_ROBOT",
    "RESET_HANDS",
    "RESET_OBJECTS",
    "SETTLE",
    "RESET_VERIFY",
)
BATCH_COLLECTOR_CONTRACT = {
    "order": [
        "reset_episode",
        "verify_reset",
        "start_recording",
        "run_episode",
        "stop_recording",
        "evaluate_episode",
    ],
    "recorder_start_gate": "reset_status == PASS and overall_reset_ready == true",
    "accepted_for_training": (
        "reset_status == PASS and expert_status == PASS and "
        "abnormal_motion == false and task_result_correct == true"
    ),
    "failed_episode_policy": "retain diagnostics; exclude from training set",
}
STORAGE_BOX_ID = str(CONFIRMED_SCENE_UI["storage_box"]["thing_id"])
STORAGE_BOX_POSE = tuple(float(value) for value in CONFIRMED_SCENE_UI["storage_box"]["world_pose"])
LEFT_READY_JOINTS = tuple(tuple(float(value) for value in row) for row in LEFT_PRE_JOINTS)
RIGHT_READY_JOINTS = tuple(tuple(float(value) for value in row) for row in RIGHT_OBSERVATION_JOINTS)


@dataclass(frozen=True)
class ResetThresholds:
    joint_error_rad: float = 0.05
    joint_stability_rad: float = 0.01
    hand_clench_error: float = 0.08
    hand_stability_rad: float = 0.02
    nut_position_error_m: float = 0.025
    nut_stability_m: float = 0.004
    sample_count: int = 3
    sample_interval_s: float = 0.25
    settle_s: float = 2.0


class EpisodeResetError(RuntimeError):
    pass


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return repr(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def float_vector(value: Any) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        for key in ("position", "positions", "joint_angles", "joints", "qpos", "values"):
            if key in value:
                return float_vector(value[key])
        return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if values and all(math.isfinite(item) for item in values) else None


def sdk_command_ok(value: Any) -> tuple[bool, str | None]:
    """Strictly classify only the command acknowledgement, never final state."""
    if value is False or value is None:
        return False, repr(value)
    plain = jsonable(value)
    if isinstance(plain, dict):
        if plain.get("success") is False or plain.get("ok") is False or plain.get("error"):
            return False, str(plain.get("error") or plain)
    if isinstance(plain, str) and any(token in plain.lower() for token in ("error", "fail", "exception")):
        return False, plain
    return True, None


def command(method: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    started = time.monotonic()
    try:
        returned = method(*args, **kwargs)
        accepted, error = sdk_command_ok(returned)
        return {
            "accepted": accepted,
            "return_value": jsonable(returned),
            "error": error,
            "elapsed_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "accepted": False,
            "return_value": None,
            "error": repr(exc),
            "elapsed_s": time.monotonic() - started,
        }


def max_abs_error(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def sample_joint_target(
    device: Any,
    target: Sequence[float],
    *,
    count: int,
    interval_s: float,
    error_limit: float,
    stability_limit: float,
) -> dict[str, Any]:
    samples: list[list[float]] = []
    errors: list[str] = []
    for index in range(count):
        try:
            sample = float_vector(device.get_joint_angles())
            if sample is None:
                errors.append("get_joint_angles returned an unparseable value")
            else:
                samples.append(sample)
        except Exception as exc:
            errors.append(repr(exc))
        if index + 1 < count and interval_s > 0:
            time.sleep(interval_s)

    target_values = [float(value) for value in target]
    target_errors = [max_abs_error(target_values, sample) for sample in samples]
    adjacent_deltas = [max_abs_error(a, b) for a, b in zip(samples, samples[1:])]
    dimension_ok = bool(samples) and all(len(sample) == len(target_values) for sample in samples)
    max_error = max((value for value in target_errors if value is not None), default=None)
    max_delta = max((value for value in adjacent_deltas if value is not None), default=0.0)
    ok = (
        len(samples) == count
        and dimension_ok
        and max_error is not None
        and max_error <= error_limit
        and max_delta <= stability_limit
    )
    return {
        "ok": ok,
        "target": target_values,
        "samples": samples,
        "sample_errors": errors,
        "max_abs_target_error": max_error,
        "max_adjacent_sample_delta": max_delta,
        "thresholds": {"target_error": error_limit, "stability": stability_limit},
    }


def sample_hand_open(
    device: Any,
    *,
    count: int,
    interval_s: float,
    clench_error_limit: float,
    joint_stability_limit: float,
) -> dict[str, Any]:
    """Verify normalized open state and independently verify physical-joint stability."""
    clench_samples: list[list[float]] = []
    joint_samples: list[list[float]] = []
    errors: list[str] = []
    for index in range(count):
        try:
            clench = float_vector(device.get_clench())
            if clench is None:
                errors.append("get_clench returned an unparseable value")
            else:
                clench_samples.append(clench)
        except Exception as exc:
            errors.append(f"get_clench: {exc!r}")
        try:
            joints = float_vector(device.get_joint_angles())
            if joints is None:
                errors.append("get_joint_angles returned an unparseable value")
            else:
                joint_samples.append(joints)
        except Exception as exc:
            errors.append(f"get_joint_angles: {exc!r}")
        if index + 1 < count and interval_s > 0:
            time.sleep(interval_s)

    target = [float(value) for value in HAND_OPEN]
    clench_errors = [max_abs_error(target, sample) for sample in clench_samples]
    clench_deltas = [max_abs_error(a, b) for a, b in zip(clench_samples, clench_samples[1:])]
    joint_deltas = [max_abs_error(a, b) for a, b in zip(joint_samples, joint_samples[1:])]
    max_clench_error = max((value for value in clench_errors if value is not None), default=None)
    max_clench_delta = max((value for value in clench_deltas if value is not None), default=0.0)
    max_joint_delta = max((value for value in joint_deltas if value is not None), default=0.0)
    dimensions_ok = (
        all(len(sample) == len(target) for sample in clench_samples)
        and bool(joint_samples)
        and all(len(sample) == len(joint_samples[0]) for sample in joint_samples)
    )
    ok = (
        len(clench_samples) == count
        and len(joint_samples) == count
        and dimensions_ok
        and max_clench_error is not None
        and max_clench_error <= clench_error_limit
        and max_clench_delta <= joint_stability_limit
        and max_joint_delta <= joint_stability_limit
    )
    return {
        "ok": ok,
        "target_clench": target,
        "target_source": "agents.three_nut_expert.config.HAND_OPEN",
        "clench_samples": clench_samples,
        "joint_angle_samples": joint_samples,
        "sample_errors": errors,
        "max_abs_clench_target_error": max_clench_error,
        "max_adjacent_clench_delta": max_clench_delta,
        "max_adjacent_joint_angle_delta": max_joint_delta,
        "thresholds": {
            "clench_target_error": clench_error_limit,
            "clench_and_joint_stability": joint_stability_limit,
        },
        "semantics": {
            "get_clench": "normalized 6D hand state compared with HAND_OPEN",
            "get_joint_angles": "physical joint readback used for stability only; not compared with normalized HAND_OPEN",
        },
    }


def assign_nut_centers(centers: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    keys = tuple(KNOWN_FIXED_NUT_WORLD_POSE)
    if len(centers) != len(keys):
        raise EpisodeResetError(f"expected exactly {len(keys)} Nut clusters, got {len(centers)}")
    expected = {
        key: np.asarray(pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE[key])[:3], dtype=float)
        for key in keys
    }
    candidates = [np.asarray(center[:3], dtype=float) for center in centers]
    best: tuple[float, tuple[np.ndarray, ...]] | None = None
    for permutation in itertools.permutations(candidates):
        score = sum(float(np.linalg.norm(permutation[index] - expected[key])) for index, key in enumerate(keys))
        if best is None or score < best[0]:
            best = (score, permutation)
    assert best is not None
    return {key: best[1][index].tolist() for index, key in enumerate(keys)}


def capture_nut_centers() -> dict[str, list[float]]:
    """Capture one fresh cloud and reuse the existing verified detector."""
    transform, _source = nut_detector.load_camera_to_world()
    message, _frames = nut_detector.capture_fresh_cloud()
    if message is None:
        raise EpisodeResetError("fresh point-cloud capture timed out")
    points, _cloud_info = nut_detector.cloud_xyz(message)
    finite = points[np.isfinite(points).all(axis=1)]
    normal, offset, _table_inliers = nut_detector.fit_table_plane(finite)
    if float((transform[:3, :3] @ normal)[2]) < 0.0:
        normal, offset = -normal, -offset
    height = finite @ normal + offset
    workspace = nut_detector.configured_nut_workspace()
    trials = [
        nut_detector.detect_at_clearance(finite, height, transform, workspace, clearance)
        for clearance in nut_detector.TABLE_CLEARANCE_CANDIDATES_M
    ]
    exact = [trial for trial in trials if len(trial["eligible"]) == len(KNOWN_FIXED_NUT_WORLD_POSE)]
    if not exact:
        counts = [len(trial["eligible"]) for trial in trials]
        raise EpisodeResetError(f"no clearance produced exactly three Nut clusters; counts={counts}")
    centers = [item["center_world"] for item in exact[0]["eligible"]]
    return assign_nut_centers(centers)


def verify_nuts(
    *,
    capture: Callable[[], dict[str, list[float]]],
    count: int,
    interval_s: float,
    position_limit_m: float,
    stability_limit_m: float,
) -> dict[str, Any]:
    captures: list[dict[str, list[float]]] = []
    errors: list[str] = []
    for index in range(count):
        try:
            captures.append(capture())
        except Exception as exc:
            errors.append(repr(exc))
        if index + 1 < count and interval_s > 0:
            time.sleep(interval_s)

    result: dict[str, Any] = {"captures": captures, "capture_errors": errors}
    for key, target_pose in KNOWN_FIXED_NUT_WORLD_POSE.items():
        target = pose_to_list(target_pose)
        samples = [row[key] for row in captures if key in row]
        position_errors = [math.dist(sample[:3], target[:3]) for sample in samples]
        stability_deltas = [math.dist(a[:3], b[:3]) for a, b in zip(samples, samples[1:])]
        pose_ok = len(samples) == count and max(position_errors, default=math.inf) <= position_limit_m
        stable = len(samples) == count and max(stability_deltas, default=0.0) <= stability_limit_m
        result[key] = {
            "target_pose": target,
            "actual_xyz_samples": samples,
            "actual_pose": [*samples[-1], None, None, None] if samples else None,
            "max_position_error_m": max(position_errors, default=None),
            "max_adjacent_position_delta_m": max(stability_deltas, default=0.0),
            "pose_ok": pose_ok,
            "stable": stable,
            "verified_components": ["x", "y", "z"],
            "unverified_components": ["roll", "pitch", "yaw"],
            "thresholds": {"position_error_m": position_limit_m, "stability_m": stability_limit_m},
        }
    return result


def verify_reset(
    runtime: Any,
    *,
    thresholds: ResetThresholds | None = None,
    nut_capture: Callable[[], dict[str, list[float]]] = capture_nut_centers,
) -> dict[str, Any]:
    """Verify real observable state; never infer readiness from command returns."""
    cfg = thresholds or ResetThresholds()
    right_arm = sample_joint_target(
        runtime.right_arm,
        RIGHT_READY_JOINTS[-1],
        count=cfg.sample_count,
        interval_s=cfg.sample_interval_s,
        error_limit=cfg.joint_error_rad,
        stability_limit=cfg.joint_stability_rad,
    )
    left_arm = sample_joint_target(
        runtime.left_arm,
        LEFT_READY_JOINTS[-1],
        count=cfg.sample_count,
        interval_s=cfg.sample_interval_s,
        error_limit=cfg.joint_error_rad,
        stability_limit=cfg.joint_stability_rad,
    )
    right_hand = sample_hand_open(
        runtime.right_hand,
        count=cfg.sample_count,
        interval_s=cfg.sample_interval_s,
        clench_error_limit=cfg.hand_clench_error,
        joint_stability_limit=cfg.hand_stability_rad,
    )
    left_hand = sample_hand_open(
        runtime.left_hand,
        count=cfg.sample_count,
        interval_s=cfg.sample_interval_s,
        clench_error_limit=cfg.hand_clench_error,
        joint_stability_limit=cfg.hand_stability_rad,
    )
    nuts = verify_nuts(
        capture=nut_capture,
        count=cfg.sample_count,
        interval_s=cfg.sample_interval_s,
        position_limit_m=cfg.nut_position_error_m,
        stability_limit_m=cfg.nut_stability_m,
    )

    operational = all(
        (
            right_arm["ok"],
            left_arm["ok"],
            right_hand["ok"],
            left_hand["ok"],
            *(nuts[key]["pose_ok"] and nuts[key]["stable"] for key in KNOWN_FIXED_NUT_WORLD_POSE),
        )
    )
    limitations = [
        "NO_VERIFIED_WORLD_OR_SCENE_RESET_API",
        "NO_VERIFIED_RIGID_BODY_LINEAR_VELOCITY_RESET_OR_READBACK",
        "NO_VERIFIED_RIGID_BODY_ANGULAR_VELOCITY_RESET_OR_READBACK",
        "NUT_ORIENTATION_NOT_OBSERVABLE_IN_CURRENT_POINTCLOUD_GATE",
        "STORAGE_BOX_POSE_COMMAND_HAS_NO_VERIFIED_READBACK",
        "CONTROLLER_INTERNAL_TARGET_STATE_HAS_NO_VERIFIED_CLEAR_API",
    ]
    return {
        "right_arm_ok": right_arm["ok"],
        "left_arm_ok": left_arm["ok"],
        "right_hand_ok": right_hand["ok"],
        "left_hand_ok": left_hand["ok"],
        "nut_A_pose_ok": nuts["A"]["pose_ok"],
        "nut_B_pose_ok": nuts["B"]["pose_ok"],
        "nut_C_pose_ok": nuts["C"]["pose_ok"],
        "nut_A_stable": nuts["A"]["stable"],
        "nut_B_stable": nuts["B"]["stable"],
        "nut_C_stable": nuts["C"]["stable"],
        "linear_velocity_ok": None,
        "angular_velocity_ok": None,
        "storage_box_pose_ok": None,
        "controller_state_ok": None,
        "operational_observable_state_ready": operational,
        "overall_reset_ready": False,
        "overall_reset_ready_reason": "SOFT_RESET_LIMITATION" if operational else "OBSERVABLE_STATE_VERIFY_FAILED",
        "soft_reset_limitations": limitations,
        "details": {
            "right_arm": right_arm,
            "left_arm": left_arm,
            "right_hand": right_hand,
            "left_hand": left_hand,
            "nuts": nuts,
        },
    }


def reset_robot(runtime: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {"right_arm": [], "left_arm": []}
    for joints in RIGHT_READY_JOINTS:
        rows["right_arm"].append(command(runtime.right_arm.move_joints, list(joints)))
    for joints in LEFT_READY_JOINTS:
        rows["left_arm"].append(command(runtime.left_arm.move_joints, list(joints)))
    return rows


def reset_hands(runtime: Any) -> dict[str, Any]:
    return {
        "right_hand": command(runtime.right_hand.clench, *list(HAND_OPEN)),
        "left_hand": command(runtime.left_hand.clench, list(HAND_OPEN)),
        "target": list(HAND_OPEN),
        "target_source": "agents.three_nut_expert.config.HAND_OPEN",
    }


def reset_objects(runtime: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "pose_source": KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
        "nuts": {},
        "storage_box": {},
        "velocity_reset": {
            "available": False,
            "status": "SOFT_RESET_LIMITATION",
            "reason": "No verified SetEntityVelocity/rigid-body velocity API is available.",
        },
    }
    for key, pose in KNOWN_FIXED_NUT_WORLD_POSE.items():
        target = pose_to_list(pose)
        rows["nuts"][key] = {
            "thing_id": NUT_IDS[key],
            "target_pose": target,
            "command": set_pose_with_retry(runtime.pose_setter, NUT_IDS[key], target),
        }
    rows["storage_box"] = {
        "thing_id": STORAGE_BOX_ID,
        "target_pose": list(STORAGE_BOX_POSE),
        "command": set_pose_with_retry(runtime.pose_setter, STORAGE_BOX_ID, list(STORAGE_BOX_POSE)),
        "readback_verified": False,
    }
    return rows


def reset_episode(
    runtime: Any,
    *,
    thresholds: ResetThresholds | None = None,
    max_attempts: int = 3,
    nut_capture: Callable[[], dict[str, list[float]]] = capture_nut_centers,
) -> dict[str, Any]:
    """Run a bounded reset/verify state machine. Recorder/Expert are not called."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    cfg = thresholds or ResetThresholds()
    attempts: list[dict[str, Any]] = []
    for attempt_id in range(1, max_attempts + 1):
        attempt: dict[str, Any] = {
            "attempt": attempt_id,
            "states": ["RESET_BEGIN"],
            "started_at": datetime.now().astimezone().isoformat(),
        }
        started = time.monotonic()
        attempt["states"].append("RESET_ROBOT")
        attempt["robot_commands"] = reset_robot(runtime)
        attempt["states"].append("RESET_HANDS")
        attempt["hand_commands"] = reset_hands(runtime)
        attempt["states"].append("RESET_OBJECTS")
        attempt["object_commands"] = reset_objects(runtime)
        attempt["states"].append("SETTLE")
        settle_started = time.monotonic()
        if cfg.settle_s > 0:
            time.sleep(cfg.settle_s)
        attempt["settle_duration_s"] = time.monotonic() - settle_started
        attempt["states"].append("RESET_VERIFY")
        attempt["verification"] = verify_reset(runtime, thresholds=cfg, nut_capture=nut_capture)
        attempt["elapsed_s"] = time.monotonic() - started
        if attempt["verification"]["overall_reset_ready"]:
            attempt["states"].append("EPISODE_READY")
            attempts.append(attempt)
            return {"status": "PASS", "overall_reset_ready": True, "attempts": attempts}
        attempt["states"].append("RESET_FAILED")
        attempts.append(attempt)
        if attempt["verification"]["overall_reset_ready_reason"] == "SOFT_RESET_LIMITATION":
            break
    return {
        "status": "RESET_FAILED",
        "overall_reset_ready": False,
        "attempts": attempts,
        "batch_collection_allowed": False,
        "fault": "Reset verification did not establish complete Episode equivalence.",
    }


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def audit_rabo_reset_capabilities() -> dict[str, Any]:
    keywords = (
        "reset", "world_reset", "reset_world", "scene_reset", "reset_scene",
        "reload", "restart", "setentitypose", "setentityvelocity",
        "linear_velocity", "angular_velocity", "rigid_body", "simulation",
    )
    project_calls = {
        "SetEntityPose": [
            "agents/three_nut_expert/expert.py",
            "tools/test_nut_camera_calibration.py",
            "tools/test_three_nut_closed_loop_v2.py",
        ]
    }
    modules: dict[str, Any] = {}
    for module_name in ("rabo_robocap", "rabo_dev_kit"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            modules[module_name] = {
                "importable": False,
                "error": repr(exc),
                "host_permission": "UNVERIFIED_MODULE_UNAVAILABLE",
                "matches": [],
            }
            continue
        matches: list[dict[str, Any]] = []
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            low = name.lower()
            if any(keyword in low for keyword in keywords):
                matches.append({
                    "module": module_name,
                    "name": name,
                    "kind": "class" if inspect.isclass(obj) else "function" if callable(obj) else type(obj).__name__,
                    "signature": safe_signature(obj) if callable(obj) else None,
                    "project_calls": project_calls.get(name, []),
                    "usable_by_script": "IMPORTABLE_ONLY_MUTATING_CALL_NOT_PROBED",
                })
            if inspect.isclass(obj):
                for method_name, method in inspect.getmembers(obj, callable):
                    if method_name.startswith("_"):
                        continue
                    if any(keyword in method_name.lower() for keyword in keywords):
                        matches.append({
                            "module": module_name,
                            "class": name,
                            "name": method_name,
                            "kind": "method",
                            "signature": safe_signature(method),
                            "project_calls": project_calls.get(name, []),
                            "usable_by_script": "IMPORTABLE_ONLY_MUTATING_CALL_NOT_PROBED",
                        })
        source_matches: list[dict[str, Any]] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            root = Path(module_file).resolve().parent
            for path in sorted(root.rglob("*.py")):
                try:
                    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for line_number, line in enumerate(lines, start=1):
                    low = line.lower()
                    if any(keyword in low for keyword in keywords):
                        source_matches.append({
                            "file": str(path),
                            "line": line_number,
                            "text": line.strip()[:300],
                        })
        modules[module_name] = {
            "importable": True,
            "file": module_file,
            "host_permission": "IMPORT_SUCCEEDED_MUTATING_PERMISSION_UNVERIFIED",
            "matches": matches,
            "source_matches": source_matches,
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "searched_install_paths": [
            {
                "path": "/opt/rabo-venvs/agent_system/lib/python3.12/site-packages/rabo_robocap",
                "exists": Path("/opt/rabo-venvs/agent_system/lib/python3.12/site-packages/rabo_robocap").exists(),
            },
            {
                "path": "/opt/rabo-venvs/agent_system/lib/python3.12/site-packages/rabo_dev_kit",
                "exists": Path("/opt/rabo-venvs/agent_system/lib/python3.12/site-packages/rabo_dev_kit").exists(),
            },
        ],
        "modules": modules,
        "verified_project_capabilities": {
            "robot_command": ["LinkerArmA7.move_joints"],
            "robot_readback": ["LinkerArmA7.get_joint_angles"],
            "hand_command": ["LinkerHandO6Right.clench", "LinkerHandO6Left.clench"],
            "hand_readback": ["LinkerHandO6Right.get_joint_angles", "LinkerHandO6Left.get_joint_angles"],
            "object_command": ["rabo_dev_kit.SetEntityPose.set"],
            "object_readback": [],
            "world_scene_reset": [],
            "rigid_body_velocity_reset": [],
        },
        "batch_collector_contract": BATCH_COLLECTOR_CONTRACT,
    }


class RaboResetRuntime:
    def __init__(self) -> None:
        from rabo_dev_kit import SetEntityPose
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

        self.pose_setter = SetEntityPose(world=WORLD_ID)
        self.right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
        self.left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
        self.right_hand = LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim")
        self.left_hand = LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim")

    def shutdown(self) -> None:
        for device in (self.left_hand, self.right_hand, self.left_arm, self.right_arm, self.pose_setter):
            if hasattr(device, "shutdown"):
                try:
                    device.shutdown()
                except Exception:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed full Episode reset validation.")
    parser.add_argument("--audit-only", action="store_true", help="Inspect installed reset APIs without mutating simulation.")
    parser.add_argument("--reset-test", action="store_true", help="Run reset -> verify stress cycles; requires --execute.")
    parser.add_argument("--reset-cycles", type=int, default=20, help="Number of reset/verify cycles (default: 20).")
    parser.add_argument("--max-reset-attempts", type=int, default=3, help="Bounded attempts per cycle (default: 3).")
    parser.add_argument("--execute", action="store_true", help="Allow robot/hand/object reset commands in Rabo simulation.")
    parser.add_argument("--settle-s", type=float, default=ResetThresholds.settle_s)
    parser.add_argument("--sample-interval-s", type=float, default=ResetThresholds.sample_interval_s)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_rabo_reset_capabilities()
    if args.audit_only or not args.reset_test:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        raise SystemExit("--reset-test is mutating; add --execute only in the Rabo simulation workspace")
    if args.reset_cycles < 1 or args.max_reset_attempts < 1:
        raise SystemExit("--reset-cycles and --max-reset-attempts must be >= 1")
    if args.settle_s < 0 or args.sample_interval_s < 0:
        raise SystemExit("settle/sample intervals must be >= 0")

    thresholds = ResetThresholds(settle_s=args.settle_s, sample_interval_s=args.sample_interval_s)
    started = datetime.now().astimezone()
    report: dict[str, Any] = {
        "started_at": started.isoformat(),
        "mode": "RESET_STRESS_TEST_NO_EXPERT_NO_RECORDER",
        "requested_cycles": args.reset_cycles,
        "max_reset_attempts": args.max_reset_attempts,
        "thresholds": asdict(thresholds),
        "capability_audit": audit,
        "batch_collector_contract": BATCH_COLLECTOR_CONTRACT,
        "cycles": [],
        "batch_collection_allowed": False,
    }
    path = REPORT_DIR / f"reset_validation_{started.strftime('%Y%m%d_%H%M%S')}.json"
    runtime: RaboResetRuntime | None = None
    try:
        runtime = RaboResetRuntime()
        for cycle in range(1, args.reset_cycles + 1):
            result = reset_episode(
                runtime,
                thresholds=thresholds,
                max_attempts=args.max_reset_attempts,
            )
            report["cycles"].append({"cycle": cycle, **result})
            write_json(path, report)
            print(f"[RESET_CYCLE] {cycle}/{args.reset_cycles} status={result['status']}", flush=True)
        failed_cycles = [row["cycle"] for row in report["cycles"] if not row["overall_reset_ready"]]
        report["failed_cycles"] = failed_cycles
        report["status"] = "RESET_FAILED" if failed_cycles else "PASS"
        report["overall_reset_ready"] = bool(
            len(report["cycles"]) == args.reset_cycles
            and all(row["overall_reset_ready"] for row in report["cycles"])
        )
        report["batch_collection_allowed"] = report["overall_reset_ready"]
    except Exception as exc:
        report["status"] = "RESET_FAILED"
        report["error"] = repr(exc)
        report["overall_reset_ready"] = False
    finally:
        if runtime is not None:
            runtime.shutdown()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(path, report)
        print(f"report: {path}")
    return 0 if report.get("overall_reset_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
