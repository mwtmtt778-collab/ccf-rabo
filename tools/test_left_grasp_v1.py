#!/usr/bin/env python3
"""Left-hand grasp V1 parameterized experiment tool.

V1 deliberately stays small: CLI pose parameters, pose_check gates before every
Cartesian move, manual observations, and structured JSON/Markdown reports.
It does not use MoveIt, OMPL, RRT, vision, or automatic grasp optimization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "left_grasp_v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    LEFT_GRASP,
    LEFT_GRASP_FORCE,
    LEFT_PRE_JOINTS,
    NUT_IDS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import ActionStep, pose_to_list  # noqa: E402
from expert.transforms import (  # noqa: E402
    pose_orientation_error,
    pose_position_error,
    transform_pose_base_to_world,
    transform_pose_world_to_base,
)


LEFT_TEST_XY = [0.385, 0.038]
LEFT_TEST_XY_STATUS = "VERIFIED_BY_PREVIOUS_RABO_REACHABILITY_TEST"

CALIBRATION_XYZ = [0.43, 0.30, -0.10]
ORIENTATION_CALIBRATION_SAFE_JOINTS = [
    [0, -1.57, 0, 0, 0, 0, 0],
    [-1.57, -0.7, 0, 0, 0, 0, 0],
]
ORIENTATION_CALIBRATION_POSES = [
    ("Pose_A_REFERENCE", "REFERENCE", Pose6(CALIBRATION_XYZ[0], CALIBRATION_XYZ[1], CALIBRATION_XYZ[2], 0.0, 1.3, 1.57)),
    ("Pose_B", "THEORY_TEST", Pose6(CALIBRATION_XYZ[0], CALIBRATION_XYZ[1], CALIBRATION_XYZ[2], 0.0, -1.57, 0.0)),
    ("Pose_C", "POSE_B_YAW_POSITIVE", Pose6(CALIBRATION_XYZ[0], CALIBRATION_XYZ[1], CALIBRATION_XYZ[2], 0.0, -1.57, 1.0)),
    ("Pose_D", "POSE_B_YAW_NEGATIVE", Pose6(CALIBRATION_XYZ[0], CALIBRATION_XYZ[1], CALIBRATION_XYZ[2], 0.0, -1.57, -1.0)),
]

NUT_B_LEFT_TEST_WORLD_POSE = [-0.2966, 0.0420, 0.2806, 0.0, 0.0, 0.5233]
LEFT_TOP_GRASP_ROLL = 0.0
LEFT_TOP_GRASP_PITCH = -1.57
LEFT_DIRECT_GRASP_Z = -0.33
LEFT_DIRECT_PRE_GRASP_HIGH_Z = -0.15
LEFT_DIRECT_DESCENT_Z = [-0.20, -0.25, -0.29, -0.31, -0.33]
LEFT_DIRECT_LIFT_Z = [-0.30, -0.27, -0.23]
LEFT_DIRECT_GRASP_FORCE = {"strength": 1.0, "fingers": [1, 3, 4]}
LEFT_DIRECT_ORIENTATION_SOURCE = "VERIFIED_ORIENTATION_CALIBRATION_20260819"
LEFT_DIRECT_SETTLE_AFTER_POSE_S = 2.5

RIGHT_NUT_B_GRASP_POSE = Pose6(-0.2803, 0.157, -0.33, 0.0, 0.8, 0.0)
RIGHT_NUT_B_GRASP_POSE_STATUS = "VERIFIED_RIGHT_NUT_B_SUCCESS_FROZEN"

LEFT_BASE_WORLD = [-0.6816, 0.0040, 0.7520, 0.0, 0.0, 0.0]
RIGHT_BASE_WORLD = [-0.6816, -0.0040, 0.7520, 0.0, 0.0, 3.14]
BASE_WORLD_STATUS = "VERIFIED_CURRENT_PROJECT_VALUE"

NUT_B_INITIAL_REFERENCE_POSE = [-0.3413, -0.171, 0.2806, 0.0, 0.0, 0.5233]
NUT_B_INITIAL_REFERENCE_STATUS = "PREDICTED_INITIAL_SCENE_REFERENCE_ONLY_NOT_RUNTIME_GROUND_TRUTH"

RIGHT_GRASP_OFFSET_WORLD_XY = [-0.06025, 0.00955]
RIGHT_GRASP_OFFSET_STATUS = "PREDICTED"

DEFAULT_RIGHT_RELEASE_Z_RPY = [-0.20, 0.0, 0.8, 0.0]
RIGHT_RETREAT_DZ = 0.10

DRY_CHOICES = {
    "1": "GOOD",
    "2": "TOO_HIGH",
    "3": "TOO_LOW",
    "4": "PALM_ORIENTATION_WRONG",
    "5": "FINGER_DIRECTION_WRONG",
    "6": "COLLISION_RISK",
    "7": "OTHER",
}

DRY_GRASP_ONLY_CHOICES = {
    "1": "GOOD",
    "2": "PALM_DOWN_GOOD",
    "3": "PALM_UP_WRONG",
    "4": "PALM_REVERSED",
    "5": "PALM_ORIENTATION_WRONG",
    "6": "FINGER_DIRECTION_WRONG",
    "7": "TOO_HIGH",
    "8": "TOO_LOW",
    "9": "TABLE_COLLISION",
    "10": "NEAR_TABLE_COLLISION",
    "11": "COLLISION_RISK",
    "12": "OTHER",
}

PLACE_CHOICES = {
    "1": "GOOD",
    "2": "NUT_X_NEGATIVE_ERROR",
    "3": "NUT_X_POSITIVE_ERROR",
    "4": "NUT_Y_NEGATIVE_ERROR",
    "5": "NUT_Y_POSITIVE_ERROR",
    "6": "NUT_ROLL_OR_BOUNCE_ERROR",
    "7": "UNKNOWN",
    "8": "OTHER",
}

REAL_CHOICES = {
    "1": "SUCCESS",
    "2": "NO_CONTACT",
    "3": "CONTACT_BUT_NOT_GRASPED",
    "4": "GRASPED_BUT_DROPPED_ON_LIFT",
    "5": "NUT_PLACEMENT_BAD",
    "6": "COLLISION_OR_INTERFERENCE",
    "7": "OTHER",
}


class StepExecutionError(RuntimeError):
    def __init__(self, message: str, *, failed_node: str | None = None, failed_stage: str | None = None) -> None:
        super().__init__(message)
        self.failed_node = failed_node
        self.failed_stage = failed_stage


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return repr(value)


def pose_from_list(values: list[float] | tuple[float, ...]) -> Pose6:
    if len(values) != 6:
        raise ValueError(f"pose must have 6 values, got {len(values)}")
    return Pose6(*[float(v) for v in values])


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return asdict(pose)


def pose_list(pose: Pose6 | None) -> list[float] | None:
    return None if pose is None else pose_to_list(pose)


def pose_with_z_delta(pose: Pose6, dz: float) -> Pose6:
    return Pose6(pose.x, pose.y, pose.z + dz, pose.roll, pose.pitch, pose.yaw)


def staged_waypoints(left_grasp: Pose6, staged_z: list[float] | None = None) -> list[tuple[str, Pose6]]:
    z_values = staged_z if staged_z is not None else [left_grasp.z + 0.10, left_grasp.z + 0.06, left_grasp.z + 0.03]
    return [
        ("HIGH", Pose6(left_grasp.x, left_grasp.y, z_values[0], left_grasp.roll, left_grasp.pitch, left_grasp.yaw)),
        ("MID", Pose6(left_grasp.x, left_grasp.y, z_values[1], left_grasp.roll, left_grasp.pitch, left_grasp.yaw)),
        ("LOW", Pose6(left_grasp.x, left_grasp.y, z_values[2], left_grasp.roll, left_grasp.pitch, left_grasp.yaw)),
        ("GRASP", left_grasp),
    ]


def timestamp_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def run_id_text(mode: str) -> str:
    if mode == "orientation-calibration":
        return datetime.now().strftime("orientation_calibration_%Y%m%d_%H%M%S")
    if mode == "left-direct-grasp":
        return datetime.now().strftime("left_direct_grasp_%Y%m%d_%H%M%S")
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{mode.replace('-', '_')}"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"UNKNOWN: {repr(exc)}"


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def normalize_pose_check(raw: Any) -> tuple[str, str, Any]:
    value = jsonable(raw)
    if isinstance(raw, bool):
        return ("PASS" if raw else "FAIL", "reachable" if raw else "pose_check_returned_false", value)
    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        ok = lower.get("success", lower.get("ok", lower.get("reachable", lower.get("result"))))
        reason = lower.get("reason", lower.get("message", lower.get("error", "")))
        if isinstance(ok, bool):
            return ("PASS" if ok else "FAIL", str(reason or ("reachable" if ok else "not_reachable")), value)
    if isinstance(value, list) and value and isinstance(value[0], bool):
        return ("PASS" if value[0] else "FAIL", str(value[1] if len(value) > 1 else ""), value)
    text = str(value)
    low = text.lower()
    if "true" in low or "reachable" in low or "success" in low:
        return "PASS", text, value
    if "false" in low or "fail" in low or "limit" in low or "workspace" in low:
        return "FAIL", text, value
    return "CHECK", text, value


def step_result_failed(result: Any) -> tuple[bool, str]:
    value = jsonable(result)
    if result is None:
        return False, ""
    if isinstance(result, bool):
        return (not result), "returned_false"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
    low = text.lower()
    if any(token in low for token in ("error", "failed", "fail", "false", "ik", "无解", "失败", "错误")):
        return True, text
    return False, ""


def safe_call(label: str, obj: Any, method_name: str) -> Any:
    if obj is None or not hasattr(obj, method_name):
        return None
    try:
        return jsonable(getattr(obj, method_name)())
    except Exception as exc:
        return f"{label}.{method_name} ERROR {repr(exc)}"


def read_device_state(bundle: Any | None, prefix: str) -> dict[str, Any]:
    if bundle is None:
        return {"timestamp": timestamp_text(), "state": "DEVICE_BUNDLE_NOT_INITIALIZED"}
    data: dict[str, Any] = {"timestamp": timestamp_text()}
    for name in ("left_arm", "right_arm", "left_hand", "right_hand"):
        device = getattr(bundle, name, None)
        if device is None:
            continue
        data[f"{name}_pose"] = safe_call(name, device, "get_pose") if name.endswith("_arm") else None
        data[f"{name}_joint_angles"] = safe_call(name, device, "get_joint_angles")
        data[f"{name}_clench"] = safe_call(name, device, "get_clench") if name.endswith("_hand") else None
    data["label"] = prefix
    return data


def shutdown_bundle(bundle: Any | None) -> None:
    if bundle is None:
        return
    if getattr(bundle, "_shutdown_done", False):
        return
    setattr(bundle, "_shutdown_done", True)
    for name in ("left_hand", "right_hand", "left_arm", "right_arm", "pose_setter"):
        device = getattr(bundle, name, None)
        if hasattr(device, "shutdown"):
            try:
                device.shutdown()
            except Exception as exc:
                print(f"shutdown warning: {name}: {repr(exc)}")


def make_bundle(
    include_left: bool,
    include_right: bool,
    *,
    include_left_hand: bool = True,
    include_right_hand: bool = True,
    include_pose_setter: bool = False,
) -> Any:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

    values: dict[str, Any] = {"_shutdown_done": False}
    if include_left:
        values["left_arm"] = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
        if include_left_hand:
            values["left_hand"] = LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim")
    if include_right:
        values["right_arm"] = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
        if include_right_hand:
            values["right_hand"] = LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim")
    if include_pose_setter:
        from rabo_dev_kit import SetEntityPose

        values["pose_setter"] = SetEntityPose(world=WORLD_ID)
    return SimpleNamespace(**values)


class MockArm:
    def __init__(self, result: str, *, pose_check_sequence: list[str] | None = None, move_fail_phase: str | None = None) -> None:
        self.mock_pose_check_result = result
        self.mock_pose_check_sequence = list(pose_check_sequence or [])
        self.mock_move_fail_phase = move_fail_phase
        self.calls: list[dict[str, Any]] = []
        self.current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def pose_check(self, *_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("mock pose_check should be handled by call_pose_check")

    def move_to(self, x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> bool:
        pose = [x, y, z, roll, pitch, yaw]
        self.calls.append({"method": "move_to", "pose": pose})
        if self.mock_move_fail_phase is not None and self.mock_move_fail_phase == getattr(self, "current_phase", None):
            return False
        self.current_pose = pose
        return True

    def move_joints(self, joints: list[float]) -> bool:
        self.calls.append({"method": "move_joints", "joints": joints})
        return True

    def get_pose(self) -> list[float]:
        return list(self.current_pose)

    def get_joint_angles(self) -> list[float]:
        return [0.0] * 7

    def shutdown(self) -> None:
        self.calls.append({"method": "shutdown"})


class MockHand:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def clench(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append({"method": "clench", "args": list(args), "kwargs": kwargs})
        return True

    def open(self) -> bool:
        self.calls.append({"method": "open"})
        return True

    def grasp_force(self, **kwargs: Any) -> bool:
        self.calls.append({"method": "grasp_force", "kwargs": kwargs})
        return True

    def get_joint_angles(self) -> list[float]:
        return [0.0] * 6

    def get_clench(self) -> list[float]:
        return [0.0] * 6

    def shutdown(self) -> None:
        self.calls.append({"method": "shutdown"})


class MockPoseSetter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def set(self, thing_id: str, pose: tuple[float, float, float, float, float, float]) -> bool:
        self.calls.append({"method": "set", "thing_id": thing_id, "pose": list(pose)})
        return True

    def shutdown(self) -> None:
        self.calls.append({"method": "shutdown"})


def make_mock_bundle(
    include_left: bool,
    include_right: bool,
    *,
    include_left_hand: bool,
    include_right_hand: bool,
    result: str,
    include_pose_setter: bool = False,
    pose_check_sequence: list[str] | None = None,
    move_fail_phase: str | None = None,
) -> Any:
    values: dict[str, Any] = {"_shutdown_done": False, "mock": True}
    if include_left:
        values["left_arm"] = MockArm(result, pose_check_sequence=pose_check_sequence, move_fail_phase=move_fail_phase)
        if include_left_hand:
            values["left_hand"] = MockHand()
    if include_right:
        values["right_arm"] = MockArm(result)
        if include_right_hand:
            values["right_hand"] = MockHand()
    if include_pose_setter:
        values["pose_setter"] = MockPoseSetter()
    return SimpleNamespace(**values)


def make_move_step(phase: str, target: str, pose: Pose6, note: str = "") -> ActionStep:
    return ActionStep(phase, target, "move_to", [], pose_kwargs(pose), True, note)


def make_hand_step(phase: str, target: str, method: str, args: list[Any], kwargs: dict[str, Any], note: str) -> ActionStep:
    return ActionStep(phase, target, method, args, kwargs, True, note)


def target_pose_from_step(step: ActionStep) -> Pose6 | None:
    if step.method != "move_to":
        return None
    return Pose6(
        step.kwargs["x"],
        step.kwargs["y"],
        step.kwargs["z"],
        step.kwargs.get("roll", 0.0),
        step.kwargs.get("pitch", 0.0),
        step.kwargs.get("yaw", 0.0),
    )


def call_pose_check(arm: Any, pose: Pose6) -> dict[str, Any]:
    mock_result = getattr(arm, "mock_pose_check_result", None)
    if mock_result is not None:
        sequence = getattr(arm, "mock_pose_check_sequence", [])
        if sequence:
            mock_result = sequence.pop(0)
        status = "PASS" if mock_result == "pass" else "FAIL"
        reason = "mock_reachable" if status == "PASS" else "mock_out_of_workspace"
        return {"status": status, "reason": reason, "raw": {"mock": mock_result}, "pose": pose_to_list(pose)}
    raw = arm.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)
    status, reason, value = normalize_pose_check(raw)
    return {"status": status, "reason": reason, "raw": value, "pose": pose_to_list(pose)}


def call_step(bundle: Any, step: ActionStep) -> Any:
    target = getattr(bundle, step.target)
    method = getattr(target, step.method)
    setattr(target, "current_phase", step.phase)
    if step.method == "move_to":
        return method(**pose_kwargs(target_pose_from_step(step)))
    if step.method == "grasp_force":
        if step.kwargs.get("fingers") is None:
            return method(strength=step.kwargs["strength"])
        return method(strength=step.kwargs["strength"], fingers=step.kwargs["fingers"])
    return method(*step.args, **step.kwargs)


def numeric_pose(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            return [
                float(value["x"]),
                float(value["y"]),
                float(value["z"]),
                float(value.get("roll", 0.0)),
                float(value.get("pitch", 0.0)),
                float(value.get("yaw", 0.0)),
            ]
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 6:
        try:
            return [float(value[i]) for i in range(6)]
        except Exception:
            return None
    return None


def pose_error(target: Pose6 | None, actual_raw: Any) -> dict[str, Any] | None:
    actual = numeric_pose(actual_raw)
    if target is None or actual is None:
        return None
    target_values = pose_to_list(target)
    return {
        "position_error_m": pose_position_error(target_values, actual),
        "orientation_error_rpy_l2_rad": pose_orientation_error(target_values, actual),
    }


def update_report_pose_summary(report: dict[str, Any], *, target: Pose6 | None, actual_raw: Any, joints_raw: Any) -> None:
    report["actual_pose"] = jsonable(actual_raw) if actual_raw is not None else "NOT_APPLICABLE"
    report["actual_joints"] = jsonable(joints_raw) if joints_raw is not None else "NOT_APPLICABLE"
    error = pose_error(target, actual_raw)
    if error is None:
        report["position_error"] = "NOT_APPLICABLE"
        report["orientation_error"] = "NOT_APPLICABLE"
    else:
        report["position_error"] = error["position_error_m"]
        report["orientation_error"] = error["orientation_error_rpy_l2_rad"]


def append_robot_record(report: dict[str, Any], **kwargs: Any) -> None:
    record = {"timestamp": timestamp_text(), **kwargs}
    report["robot_data"].append(jsonable(record))


def prompt_enter_skip_quit(message: str) -> str:
    answer = input(message).strip().lower()
    if answer == "q":
        return "quit"
    if answer == "s":
        return "skip"
    return "enter"


def open_left_hand(left_hand: Any, report: dict[str, Any], *, phase: str = "ORIENTATION_CALIBRATION_OPEN_HAND") -> None:
    if hasattr(left_hand, "open"):
        result = left_hand.open()
        method = "open"
    else:
        result = left_hand.clench(list(HAND_OPEN))
        method = "clench"
    failed, reason = step_result_failed(result)
    append_robot_record(
        report,
        phase=phase,
        target="left_hand",
        method=method,
        return_value=result,
        status="FAILED" if failed else "OK",
    )
    print(f"[LEFT_HAND_OPEN] method={method} return={jsonable(result)}")
    if failed:
        raise StepExecutionError(
            f"{phase}: left_hand.{method} failed: {reason}",
            failed_node=phase,
        )


def execute_orientation_move(bundle: Any, name: str, pose: Pose6, report: dict[str, Any], note: str) -> tuple[bool, Any, Any, Any]:
    setattr(bundle.left_arm, "current_phase", name)
    result = bundle.left_arm.move_to(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)
    failed, reason = step_result_failed(result)
    actual_pose = safe_call("left_arm", bundle.left_arm, "get_pose")
    joint_angles = safe_call("left_arm", bundle.left_arm, "get_joint_angles")
    append_robot_record(
        report,
        phase=name,
        target="left_arm",
        method="move_to",
        note=note,
        target_pose=pose_to_list(pose),
        move_to_return=result,
        actual_get_pose=actual_pose,
        joint_angles=joint_angles,
        pose_error=pose_error(pose, actual_pose),
        status="FAILED" if failed else "OK",
    )
    if failed:
        print(f"[FAIL] {name} move_to failed: {reason}")
    return (not failed), jsonable(result), actual_pose, joint_angles


def left_direct_pose(z: float, yaw: float) -> Pose6:
    return Pose6(LEFT_TEST_XY[0], LEFT_TEST_XY[1], z, LEFT_TOP_GRASP_ROLL, LEFT_TOP_GRASP_PITCH, yaw)


def record_left_direct_waypoint(report: dict[str, Any], bucket: str, record: dict[str, Any]) -> None:
    report.setdefault(bucket, []).append(jsonable(record))


def confirm_or_abort(prompt: str, *, failed_node: str) -> None:
    answer = input(prompt).strip().lower()
    if answer == "q":
        raise StepExecutionError(f"{failed_node}: aborted by user", failed_node=failed_node)


def checked_left_move(
    bundle: Any,
    report: dict[str, Any],
    *,
    phase: str,
    pose: Pose6,
    waypoint_bucket: str | None,
    confirm_prompt: str | None = None,
) -> dict[str, Any]:
    pose_check = call_pose_check(bundle.left_arm, pose)
    record: dict[str, Any] = {
        "phase": phase,
        "target_pose": pose_to_list(pose),
        "pose_check": pose_check["status"] == "PASS",
        "pose_check_status": pose_check["status"],
        "pose_check_reason": pose_check["reason"],
        "pose_check_raw": pose_check["raw"],
        "move_executed": False,
        "move_to_return": None,
        "actual_get_pose": [],
        "actual_joints": [],
    }
    append_robot_record(
        report,
        phase=phase,
        target="left_arm",
        method="pose_check",
        target_pose=pose_to_list(pose),
        pose_check=pose_check,
        status="OK" if pose_check["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
    )
    print(f"[{phase}]")
    print(f"target = {pose_to_list(pose)}")
    print(f"pose_check = {pose_check['status']}")
    print(f"reason = {pose_check['reason']}")
    if pose_check["status"] != "PASS":
        if waypoint_bucket is not None:
            record_left_direct_waypoint(report, waypoint_bucket, record)
        raise StepExecutionError(f"{phase}: left_arm pose_check failed: {pose_check['reason']}", failed_node=phase)
    if confirm_prompt is not None:
        confirm_or_abort(confirm_prompt, failed_node=phase)
    success, move_return, actual_pose, joint_angles = execute_orientation_move(
        bundle,
        phase,
        pose,
        report,
        "left direct grasp checked waypoint",
    )
    record["move_executed"] = True
    record["move_to_return"] = move_return
    record["actual_get_pose"] = jsonable(actual_pose)
    record["actual_joints"] = jsonable(joint_angles)
    record["move_success"] = success
    print("actual get_pose()")
    print(json.dumps(jsonable(actual_pose), ensure_ascii=False))
    print("actual joints")
    print(json.dumps(jsonable(joint_angles), ensure_ascii=False))
    print(f"move_to return value = {jsonable(move_return)}")
    if waypoint_bucket is not None:
        record_left_direct_waypoint(report, waypoint_bucket, record)
    if not success:
        raise StepExecutionError(f"{phase}: left_arm.move_to failed", failed_node=phase)
    return record


def execute_left_hand_command(
    left_hand: Any,
    report: dict[str, Any],
    *,
    phase: str,
    method_name: str,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    args = args or []
    kwargs = kwargs or {}
    method = getattr(left_hand, method_name)
    result = method(*args, **kwargs)
    failed, reason = step_result_failed(result)
    append_robot_record(
        report,
        phase=phase,
        target="left_hand",
        method=method_name,
        args=args,
        kwargs=kwargs,
        return_value=result,
        joint_angles=safe_call("left_hand", left_hand, "get_joint_angles"),
        hand_clench=safe_call("left_hand", left_hand, "get_clench"),
        status="FAILED" if failed else "OK",
    )
    print(f"[{phase}] {method_name} return = {jsonable(result)}")
    if failed:
        raise StepExecutionError(f"{phase}: left_hand.{method_name} failed: {reason}", failed_node=phase)
    return result


def execute_action_step(bundle: Any, step: ActionStep, report: dict[str, Any], *, sleep_after_s: float = 0.0) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": timestamp_text(),
        "phase": step.phase,
        "target": step.target,
        "method": step.method,
        "args": jsonable(step.args),
        "kwargs": jsonable(step.kwargs),
        "note": step.note,
        "target_pose": pose_list(target_pose_from_step(step)),
    }
    target_pose = target_pose_from_step(step)
    if target_pose is not None:
        pose_check = call_pose_check(getattr(bundle, step.target), target_pose)
        record["pose_check"] = pose_check
        if pose_check["status"] != "PASS":
            record["status"] = "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE"
            report["robot_data"].append(record)
            raise StepExecutionError(f"{step.phase}: {step.target} pose_check failed: {pose_check['reason']}")
    result = call_step(bundle, step)
    failed, reason = step_result_failed(result)
    record["move_to_return" if step.method == "move_to" else "return"] = jsonable(result)
    record["status"] = "FAILED" if failed else "OK"
    state = read_device_state(bundle, step.phase)
    record["actual_get_pose"] = state.get(f"{step.target}_pose")
    record["joint_angles"] = state.get(f"{step.target}_joint_angles")
    record["hand_joint_angles"] = state.get(f"{step.target}_joint_angles") if step.target.endswith("_hand") else None
    record["hand_clench"] = state.get(f"{step.target}_clench") if step.target.endswith("_hand") else None
    record["pose_error"] = pose_error(target_pose, record["actual_get_pose"])
    report["robot_data"].append(record)
    if step.target == "left_arm" and step.method == "move_to":
        update_report_pose_summary(report, target=target_pose, actual_raw=record["actual_get_pose"], joints_raw=record["joint_angles"])
    if failed:
        raise StepExecutionError(f"{step.phase}: {step.target}.{step.method} failed: {reason}")
    if sleep_after_s > 0:
        time.sleep(sleep_after_s)
    return record


def execute_group(bundle: Any, name: str, steps: list[ActionStep], report: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"[{name}]")
    for step in steps:
        execute_action_step(bundle, step, report, sleep_after_s=args.step_delay_s)


def preflight_pose_checks(bundle: Any, checks: list[tuple[str, str, Pose6]], report: dict[str, Any]) -> None:
    for label, arm_name, pose in checks:
        result = call_pose_check(getattr(bundle, arm_name), pose)
        update_chain_reachability(report, label, result)
        record = {
            "timestamp": timestamp_text(),
            "phase": label,
            "target": arm_name,
            "method": "pose_check",
            "target_pose": pose_to_list(pose),
            "pose_check": result,
            "status": "OK" if result["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
        }
        report["robot_data"].append(record)
        if result["status"] != "PASS":
            raise StepExecutionError(f"{label}: {arm_name} pose_check failed: {result['reason']}")


def preflight_staged_waypoints(bundle: Any, waypoints: list[tuple[str, Pose6]], report: dict[str, Any]) -> None:
    print("STAGED PREFLIGHT")
    report["staged_preflight"] = []
    report["staged_pose_checks"] = {}
    for stage, pose in waypoints:
        result = call_pose_check(bundle.left_arm, pose)
        if stage == "GRASP":
            update_chain_reachability(report, "LEFT_GRASP", result)
        phase = "LEFT_GRASP" if stage == "GRASP" else f"STAGED_{stage}"
        row = {
            "timestamp": timestamp_text(),
            "phase": phase,
            "stage": stage,
            "target": "left_arm",
            "method": "pose_check",
            "target_pose": pose_to_list(pose),
            "pose_check": result,
            "status": "OK" if result["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
        }
        report["staged_preflight"].append(row)
        report["staged_pose_checks"][phase] = result
        report["robot_data"].append(row)
        print(f"{stage}: {result['status']} ({result['reason']})")
    for row in report["staged_preflight"]:
        if row["pose_check"]["status"] != "PASS":
            raise StepExecutionError(
                f"STAGE_PREFLIGHT_FAILED: {row['stage']} pose_check failed: {row['pose_check']['reason']}",
                failed_node="STAGE_PREFLIGHT_FAILED",
                failed_stage=row["stage"],
            )


def execute_move_without_pose_check(bundle: Any, phase: str, pose: Pose6, report: dict[str, Any], args: argparse.Namespace) -> None:
    step = make_move_step(phase, "left_arm", pose, "staged grasp-only descent waypoint")
    preflight_result = (report.get("staged_pose_checks") or {}).get(phase, "NOT_TESTED")
    record: dict[str, Any] = {
        "timestamp": timestamp_text(),
        "phase": step.phase,
        "target": step.target,
        "method": step.method,
        "args": jsonable(step.args),
        "kwargs": jsonable(step.kwargs),
        "note": step.note,
        "target_pose": pose_to_list(pose),
        "pose_check": preflight_result,
        "pose_check_result": preflight_result.get("status") if isinstance(preflight_result, dict) else "NOT_TESTED",
        "pose_check_reason": preflight_result.get("reason") if isinstance(preflight_result, dict) else "NOT_TESTED",
    }
    result = call_step(bundle, step)
    failed, reason = step_result_failed(result)
    record["move_to_return"] = jsonable(result)
    record["status"] = "FAILED" if failed else "OK"
    state = read_device_state(bundle, step.phase)
    record["actual_get_pose"] = state.get("left_arm_pose")
    record["joint_angles"] = state.get("left_arm_joint_angles")
    record["pose_error"] = pose_error(pose, record["actual_get_pose"])
    report["robot_data"].append(record)
    update_report_pose_summary(report, target=pose, actual_raw=record["actual_get_pose"], joints_raw=record["joint_angles"])
    if failed:
        raise StepExecutionError(f"{phase}: left_arm.move_to failed: {reason}", failed_node=phase)
    if args.step_confirm:
        input(f"{phase} reached. Press Enter to continue...")
    if args.stage_wait > 0:
        time.sleep(args.stage_wait)


def update_chain_reachability(report: dict[str, Any], label: str, result: dict[str, Any]) -> None:
    key = {
        "LEFT_APPROACH": "APPROACH_REACHABLE",
        "LEFT_GRASP": "GRASP_REACHABLE",
        "LEFT_LIFT": "LIFT_REACHABLE",
    }.get(label)
    if key is None:
        return
    report["left_chain_reachability"][key] = {
        "status": result["status"],
        "reachable": result["status"] == "PASS",
        "reason": result["reason"],
        "source": "VERIFIED_BY_CURRENT_RUNTIME",
    }


def derive_values(args: argparse.Namespace) -> dict[str, Any]:
    left_grasp = pose_from_list(args.pose) if getattr(args, "pose", None) is not None else None
    if getattr(args, "mode", None) in ("check", "dry-grasp-only", "orientation-calibration", "left-direct-grasp"):
        left_approach = None
        left_lift = None
    else:
        left_approach = pose_with_z_delta(left_grasp, args.approach_dz) if left_grasp else None
        left_lift = pose_with_z_delta(left_grasp, args.lift_dz) if left_grasp else None

    target_world_xy = None
    right_release_world_pose = None
    right_release_base_pose = None
    right_release_pose_source = None
    if getattr(args, "nut_target_xy", None) is not None:
        target_left_base = [args.nut_target_xy[0], args.nut_target_xy[1], 0.0, 0.0, 0.0, 0.0]
        target_world = transform_pose_base_to_world(target_left_base, LEFT_BASE_WORLD)
        target_world_xy = target_world[:2]

    if getattr(args, "right_release_pose", None) is not None:
        right_release_base_pose = list(args.right_release_pose)
        right_release_world_pose = transform_pose_base_to_world(right_release_base_pose, RIGHT_BASE_WORLD)
        right_release_pose_source = "USER_OVERRIDE_EXPERIMENTAL"
    elif target_world_xy is not None:
        release_world_xy = [
            target_world_xy[0] + RIGHT_GRASP_OFFSET_WORLD_XY[0],
            target_world_xy[1] + RIGHT_GRASP_OFFSET_WORLD_XY[1],
        ]
        right_release_world_pose = [
            release_world_xy[0],
            release_world_xy[1],
            DEFAULT_RIGHT_RELEASE_Z_RPY[0],
            DEFAULT_RIGHT_RELEASE_Z_RPY[1],
            DEFAULT_RIGHT_RELEASE_Z_RPY[2],
            DEFAULT_RIGHT_RELEASE_Z_RPY[3],
        ]
        release_xy_base = transform_pose_world_to_base(
            [release_world_xy[0], release_world_xy[1], RIGHT_BASE_WORLD[2], 0.0, 0.0, 0.0],
            RIGHT_BASE_WORLD,
        )
        right_release_base_pose = [
            release_xy_base[0],
            release_xy_base[1],
            DEFAULT_RIGHT_RELEASE_Z_RPY[0],
            DEFAULT_RIGHT_RELEASE_Z_RPY[1],
            DEFAULT_RIGHT_RELEASE_Z_RPY[2],
            DEFAULT_RIGHT_RELEASE_Z_RPY[3],
        ]
        right_release_pose_source = "PREDICTED_RELEASE_POSE"

    staged = None
    if left_grasp is not None and getattr(args, "staged", False):
        staged = [
            {
                "stage": stage,
                "pose": pose_list(pose),
                "status": "PREDICTED_STAGED_ENTRY_WAYPOINT_NOT_VERIFIED",
            }
            for stage, pose in staged_waypoints(left_grasp, getattr(args, "staged_z", None))
        ]

    return {
        "LEFT_GRASP_POSE": {"value": pose_list(left_grasp), "status": "PREDICTED_EXPERIMENT_SEED" if left_grasp else "NOT_USED"},
        "LEFT_APPROACH_POSE": {"value": pose_list(left_approach), "status": "NOT_APPLICABLE" if getattr(args, "mode", None) in ("check", "dry-grasp-only", "orientation-calibration", "left-direct-grasp") else "PREDICTED"},
        "LEFT_LIFT_POSE": {"value": pose_list(left_lift), "status": "NOT_APPLICABLE" if getattr(args, "mode", None) in ("check", "dry-grasp-only", "orientation-calibration", "left-direct-grasp") else "PREDICTED"},
        "STAGED_WAYPOINTS": {"value": staged, "status": "PREDICTED_STAGED_ENTRY" if staged else "NOT_APPLICABLE"},
        "TARGET_NUT_WORLD_XY": {"value": target_world_xy, "status": "PREDICTED_FROM_LEFT_BASE_TARGET_XY" if target_world_xy else "NOT_USED"},
        "PREDICTED_RIGHT_RELEASE_WORLD_POSE": {"value": right_release_world_pose, "status": right_release_pose_source or "NOT_USED"},
        "PREDICTED_RIGHT_RELEASE_BASE_POSE": {"value": right_release_base_pose, "status": right_release_pose_source or "NOT_USED"},
    }


def make_right_groups(right_release_pose: Pose6) -> list[tuple[str, list[ActionStep]]]:
    right_retreat_pose = pose_with_z_delta(right_release_pose, RIGHT_RETREAT_DZ)
    groups: list[tuple[str, list[ActionStep]]] = [
        (
            "RIGHT_PRE",
            [ActionStep("RIGHT_PRE", "right_arm", "move_joints", [joints], {}, True, "legacy right pre-position") for joints in RIGHT_PRE_JOINTS],
        ),
        ("RIGHT_APPROACH_NUT_B", [make_move_step("RIGHT_GRASP", "right_arm", RIGHT_NUT_B_GRASP_POSE, "frozen verified Nut B grasp pose")]),
        (
            "RIGHT_GRASP_NUT_B",
            [
                make_hand_step("RIGHT_GRASP", "right_hand", "clench", [], {"thumb_rotation": 1.0}, "legacy right thumb tuck"),
                make_hand_step("RIGHT_GRASP", "right_hand", "grasp_force", [], RIGHT_GRASP_FORCE, "legacy right grasp force"),
            ],
        ),
        ("RIGHT_LIFT", [make_move_step("RIGHT_LIFT", "right_arm", pose, "legacy right lift") for pose in RIGHT_LIFT_POSES]),
        ("RIGHT_MOVE_TRANSFER_V1", [make_move_step("RIGHT_TRANSFER", "right_arm", right_release_pose, "predicted/user right release pose")]),
        ("RIGHT_RELEASE_TRANSFER", [make_hand_step("RIGHT_RELEASE", "right_hand", "clench", list(HAND_OPEN), {}, "open right hand to release Nut B")]),
        ("RIGHT_RETREAT_TRANSFER", [make_move_step("RIGHT_RETREAT", "right_arm", right_retreat_pose, "retreat above right release pose")]),
        (
            "RIGHT_RETURN_SAFE",
            [ActionStep("RIGHT_RETREAT", "right_arm", "move_joints", [joints], {}, True, "legacy right safe return") for joints in RIGHT_PRE_JOINTS],
        ),
    ]
    return groups


def right_preflight_checks(right_release_pose: Pose6) -> list[tuple[str, str, Pose6]]:
    right_retreat_pose = pose_with_z_delta(right_release_pose, RIGHT_RETREAT_DZ)
    checks = [("RIGHT_GRASP", "right_arm", RIGHT_NUT_B_GRASP_POSE)]
    checks.extend(("RIGHT_LIFT", "right_arm", pose) for pose in RIGHT_LIFT_POSES)
    checks.append(("RIGHT_TRANSFER", "right_arm", right_release_pose))
    checks.append(("RIGHT_RETREAT", "right_arm", right_retreat_pose))
    return checks


def make_left_groups(left_approach: Pose6, left_grasp: Pose6, left_lift: Pose6, *, include_lift: bool) -> list[tuple[str, list[ActionStep]]]:
    groups: list[tuple[str, list[ActionStep]]] = [
        (
            "LEFT_PRE",
            [ActionStep("LEFT_INITIAL", "left_arm", "move_joints", [joints], {}, True, "legacy left pre-position") for joints in LEFT_PRE_JOINTS],
        ),
        ("LEFT_OPEN", [make_hand_step("LEFT_INITIAL", "left_hand", "clench", list(HAND_OPEN), {}, "open left hand before experiment")]),
        ("LEFT_APPROACH", [make_move_step("LEFT_APPROACH", "left_arm", left_approach, "left approach pose")]),
        ("LEFT_GRASP_POSE", [make_move_step("LEFT_GRASP", "left_arm", left_grasp, "left grasp pose")]),
        (
            "LEFT_CLOSE",
            [
                make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "clench", list(LEFT_GRASP), {}, "legacy left grasp posture"),
                make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, "legacy left grasp force"),
            ],
        ),
    ]
    if include_lift:
        groups.append(("LEFT_LIFT", [make_move_step("LEFT_LIFT", "left_arm", left_lift, "left lift pose")]))
    return groups


def left_preflight_checks(left_approach: Pose6, left_grasp: Pose6, left_lift: Pose6) -> list[tuple[str, str, Pose6]]:
    return [
        ("LEFT_APPROACH", "left_arm", left_approach),
        ("LEFT_GRASP", "left_arm", left_grasp),
        ("LEFT_LIFT", "left_arm", left_lift),
    ]


def build_report(args: argparse.Namespace, derived: dict[str, Any]) -> dict[str, Any]:
    effective_mode = "dry-grasp-only-staged" if args.mode == "dry-grasp-only" and getattr(args, "staged", False) else args.mode
    return {
        "experiment_metadata": {
            "run_id": run_id_text(effective_mode),
            "timestamp": timestamp_text(),
            "mode": effective_mode,
            "git_commit": get_git_commit(),
        },
        "command_inputs": {
            "pose": getattr(args, "pose", None),
            "staged": getattr(args, "staged", False),
            "staged_z": getattr(args, "staged_z", None),
            "stage_wait": getattr(args, "stage_wait", None),
            "step_confirm": getattr(args, "step_confirm", False),
            "nut_target_xy": getattr(args, "nut_target_xy", None),
            "approach_dz": getattr(args, "approach_dz", None),
            "lift_dz": getattr(args, "lift_dz", None),
            "right_release_pose": getattr(args, "right_release_pose", None),
            "yaw": getattr(args, "yaw", None),
            "argv": command_line(),
            "plan_only": args.plan_only,
            "mock_pose_check": getattr(args, "mock_pose_check", None),
        },
        "known_facts": {
            "LEFT_TEST_XY": {"value": LEFT_TEST_XY, "status": LEFT_TEST_XY_STATUS},
            "RIGHT_NUT_B_GRASP_POSE": {"value": pose_to_list(RIGHT_NUT_B_GRASP_POSE), "status": RIGHT_NUT_B_GRASP_POSE_STATUS},
            "LEFT_BASE_WORLD": {"value": LEFT_BASE_WORLD, "status": BASE_WORLD_STATUS},
            "RIGHT_BASE_WORLD": {"value": RIGHT_BASE_WORLD, "status": BASE_WORLD_STATUS},
            "NUT_B_INITIAL_REFERENCE_POSE": {"value": NUT_B_INITIAL_REFERENCE_POSE, "status": NUT_B_INITIAL_REFERENCE_STATUS},
            "RIGHT_GRASP_OFFSET_WORLD_XY": {"value": RIGHT_GRASP_OFFSET_WORLD_XY, "status": RIGHT_GRASP_OFFSET_STATUS},
        },
        "derived_values": derived,
        "final_grasp": derived["LEFT_GRASP_POSE"]["value"] or "NOT_APPLICABLE",
        "staged_waypoints": derived["STAGED_WAYPOINTS"]["value"] or "NOT_APPLICABLE",
        "staged_preflight": "NOT_APPLICABLE",
        "staged_pose_checks": "NOT_APPLICABLE",
        "left_chain_reachability": {
            "APPROACH_REACHABLE": {"status": "NOT_TESTED", "reachable": "NOT_TESTED", "reason": "NOT_TESTED", "source": "NOT_TESTED"},
            "GRASP_REACHABLE": {"status": "NOT_TESTED", "reachable": "NOT_TESTED", "reason": "NOT_TESTED", "source": "NOT_TESTED"},
            "LIFT_REACHABLE": {"status": "NOT_TESTED", "reachable": "NOT_TESTED", "reason": "NOT_TESTED", "source": "NOT_TESTED"},
        },
        "robot_data": [],
        "initial_pose": "NOT_TESTED",
        "initial_joints": "NOT_TESTED",
        "target_pose": derived["LEFT_GRASP_POSE"]["value"] or "NOT_APPLICABLE",
        "actual_pose": "NOT_TESTED",
        "actual_joints": "NOT_TESTED",
        "position_error": "NOT_TESTED",
        "orientation_error": "NOT_TESTED",
        "user_observation": None,
        "diagnosis": {"labels": [], "status": "UNKNOWN", "ambiguous_causes": []},
        "runtime": {"status": "PLANNED", "failed_node": None, "failed_stage": None, "error": None},
    }


def observation_prompt(mode: str) -> tuple[dict[str, str], str]:
    if mode == "dry-grasp-only":
        return DRY_GRASP_ONLY_CHOICES, "Dry grasp-only observation"
    if mode == "dry":
        return DRY_CHOICES, "Dry observation"
    if mode == "place":
        return PLACE_CHOICES, "Nut placement observation"
    return REAL_CHOICES, "Real grasp result"


def ask_user_observation(mode: str) -> dict[str, Any]:
    choices, title = observation_prompt(mode)
    print("")
    print(title)
    for key, label in choices.items():
        print(f"{key} {label}")
    selected = input("Select result: ").strip()
    label = choices.get(selected, "OTHER")
    note = ""
    if label == "OTHER":
        note = input("OTHER note: ").strip()
    return {"source": "USER_OBSERVED", "choice": label, "note": note, "timestamp": timestamp_text()}


def derive_diagnosis(report: dict[str, Any]) -> None:
    labels: list[str] = []
    failed_node = report["runtime"].get("failed_node")
    if failed_node:
        if "LEFT_APPROACH" in failed_node:
            labels.append("LEFT_APPROACH_IK_ERROR")
        elif "LEFT_GRASP" in failed_node:
            labels.append("LEFT_GRASP_IK_ERROR")
        elif "LEFT_LIFT" in failed_node:
            labels.append("LEFT_LIFT_IK_ERROR")
        elif "RIGHT_TRANSFER" in failed_node or "RIGHT_MOVE_TRANSFER" in failed_node:
            labels.append("RIGHT_TRANSFER_IK_ERROR")
        elif failed_node.startswith("LEFT"):
            labels.append("LEFT_EXECUTION_ERROR")
        elif failed_node.startswith("RIGHT"):
            labels.append("RIGHT_TRANSFER_EXECUTION_ERROR")

    observation = (report.get("user_observation") or {}).get("choice")
    mapping = {
        "TOO_HIGH": "LEFT_Z_TOO_HIGH",
        "TOO_LOW": "LEFT_Z_TOO_LOW",
        "PALM_UP_WRONG": "LEFT_PALM_ORIENTATION_ERROR",
        "PALM_REVERSED": "LEFT_PALM_ORIENTATION_ERROR",
        "PALM_ORIENTATION_WRONG": "LEFT_PALM_ORIENTATION_ERROR",
        "FINGER_DIRECTION_WRONG": "LEFT_FINGER_DIRECTION_ERROR",
        "COLLISION_RISK": "LEFT_EXECUTION_ERROR",
        "TABLE_COLLISION": "LEFT_EXECUTION_ERROR",
        "NEAR_TABLE_COLLISION": "LEFT_EXECUTION_ERROR",
        "NUT_X_NEGATIVE_ERROR": "NUT_PLACEMENT_X_ERROR",
        "NUT_X_POSITIVE_ERROR": "NUT_PLACEMENT_X_ERROR",
        "NUT_Y_NEGATIVE_ERROR": "NUT_PLACEMENT_Y_ERROR",
        "NUT_Y_POSITIVE_ERROR": "NUT_PLACEMENT_Y_ERROR",
        "NUT_ROLL_OR_BOUNCE_ERROR": "NUT_SETTLE_ERROR",
        "NO_CONTACT": "GRASP_ALIGNMENT_ERROR",
        "CONTACT_BUT_NOT_GRASPED": "HAND_CLOSURE_ERROR",
        "GRASPED_BUT_DROPPED_ON_LIFT": "GRASP_STABILITY_ERROR",
        "SUCCESS": "SUCCESS",
        "GOOD": "SUCCESS",
    }
    if observation in mapping:
        labels.append(mapping[observation])
    if observation == "NUT_PLACEMENT_BAD":
        report["diagnosis"]["status"] = "AMBIGUOUS"
        report["diagnosis"]["ambiguous_causes"] = ["NUT_PLACEMENT_ERROR", "GRASP_ALIGNMENT_ERROR"]
    elif not labels:
        labels.append("UNKNOWN")
        report["diagnosis"]["status"] = "UNKNOWN"
    else:
        report["diagnosis"]["status"] = "EXPERIMENTAL"
    report["diagnosis"]["labels"] = sorted(set(labels))


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = report["experiment_metadata"]["run_id"]
    json_path = REPORT_DIR / f"{run_id}.json"
    md_path = REPORT_DIR / f"{run_id}.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)
        f.write("\n")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return json_path, md_path


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["experiment_metadata"]
    lines = [
        f"# Left Grasp V1 Report: {meta['run_id']}",
        "",
        "## Experiment Metadata",
        f"- run_id: `{meta['run_id']}`",
        f"- timestamp: `{meta['timestamp']}`",
        f"- mode: `{meta['mode']}`",
        f"- git_commit: `{meta['git_commit']}`",
        "",
        "## Command Inputs",
        "```json",
        json.dumps(report["command_inputs"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Derived Values",
        "```json",
        json.dumps(report["derived_values"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Staged Entry",
        f"- FINAL_GRASP: `{report['final_grasp']}`",
        "```json",
        json.dumps(report["staged_waypoints"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Runtime",
        "```json",
        json.dumps(report["runtime"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## User Observation",
        "```json",
        json.dumps(report["user_observation"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Diagnosis",
        "```json",
        json.dumps(report["diagnosis"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Robot Data",
    ]
    for row in report["robot_data"]:
        lines.extend(
            [
                "",
                f"### {row.get('phase')} / {row.get('target')} / {row.get('method')}",
                "```json",
                json.dumps(row, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes For ChatGPT Analysis",
            "- Do not treat predicted release pose or seed left grasp pose as VERIFIED.",
            "- No actual Nut runtime pose is recorded in V1; use USER_OBSERVED placement/grasp notes only.",
            "- If data cannot distinguish placement error from left grasp alignment error, keep DIAGNOSIS ambiguous.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode in ("check", "dry", "dry-grasp-only", "real") and args.pose is None:
        raise SystemExit("--pose X Y Z R P YAW is required for check, dry, dry-grasp-only, and real")
    if args.mode in ("place", "real") and args.nut_target_xy is None and args.right_release_pose is None:
        raise SystemExit("--nut-target-xy X Y or --right-release-pose X Y Z R P YAW is required for place and real")
    if args.approach_dz <= 0.0:
        raise SystemExit("--approach-dz must be positive")
    if args.lift_dz <= 0.0:
        raise SystemExit("--lift-dz must be positive")
    if getattr(args, "stage_wait", 0.0) < 0.0:
        raise SystemExit("--stage-wait must be >= 0")
    if getattr(args, "staged_z", None) is not None and not getattr(args, "staged", False):
        raise SystemExit("--staged-z requires --staged")
    if getattr(args, "step_confirm", False) and not getattr(args, "staged", False):
        raise SystemExit("--step-confirm requires --staged")
    if getattr(args, "yaw", 0.0) < -3.14 or getattr(args, "yaw", 0.0) > 3.14:
        raise SystemExit("--yaw must be within [-3.14, 3.14]")


def run_right_place(bundle: Any, right_release_pose: Pose6, report: dict[str, Any], args: argparse.Namespace) -> None:
    preflight_pose_checks(bundle, right_preflight_checks(right_release_pose), report)
    groups = make_right_groups(right_release_pose)
    for name, steps in groups[:5]:
        execute_group(bundle, name, steps, report, args)
    print(f"[WAIT_BEFORE_RELEASE] {args.wait_before_release_s:.2f}s")
    if args.wait_before_release_s > 0:
        time.sleep(args.wait_before_release_s)
    execute_group(bundle, groups[5][0], groups[5][1], report, args)
    print(f"[WAIT_OBJECT_SETTLE] {args.settle_after_release_s:.2f}s")
    if args.settle_after_release_s > 0:
        time.sleep(args.settle_after_release_s)
    for name, steps in groups[6:]:
        execute_group(bundle, name, steps, report, args)


def run_left_dry_or_grasp(bundle: Any, left_grasp: Pose6, left_approach: Pose6, left_lift: Pose6, report: dict[str, Any], args: argparse.Namespace) -> None:
    preflight_pose_checks(bundle, left_preflight_checks(left_approach, left_grasp, left_lift), report)
    groups = make_left_groups(left_approach, left_grasp, left_lift, include_lift=args.mode == "real")
    for name, steps in groups:
        execute_group(bundle, name, steps, report, args)


def run_check_mode(bundle: Any, left_grasp: Pose6, report: dict[str, Any]) -> None:
    result = call_pose_check(bundle.left_arm, left_grasp)
    update_chain_reachability(report, "LEFT_GRASP", result)
    report["robot_data"].append(
        {
            "timestamp": timestamp_text(),
            "phase": "LEFT_GRASP",
            "target": "left_arm",
            "method": "pose_check",
            "target_pose": pose_to_list(left_grasp),
            "pose_check": result,
            "status": "OK" if result["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
        }
    )
    print(f"GRASP_POSE: {pose_to_list(left_grasp)}")
    print(f"GRASP_POSE_CHECK: {result['status']}")
    print(f"REASON: {result['reason']}")


def run_dry_grasp_only_mode(bundle: Any, left_grasp: Pose6, report: dict[str, Any], args: argparse.Namespace) -> None:
    if args.staged:
        run_dry_grasp_only_staged_mode(bundle, left_grasp, report, args)
        return
    result = call_pose_check(bundle.left_arm, left_grasp)
    update_chain_reachability(report, "LEFT_GRASP", result)
    report["robot_data"].append(
        {
            "timestamp": timestamp_text(),
            "phase": "LEFT_GRASP",
            "target": "left_arm",
            "method": "pose_check",
            "target_pose": pose_to_list(left_grasp),
            "pose_check": result,
            "status": "OK" if result["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
        }
    )
    print(f"GRASP_POSE: {pose_to_list(left_grasp)}")
    print(f"GRASP_POSE_CHECK: {result['status']}")
    print(f"REASON: {result['reason']}")
    if result["status"] != "PASS":
        raise StepExecutionError(f"LEFT_GRASP: left_arm pose_check failed: {result['reason']}")
    groups = [
        ("LEFT_OPEN", [make_hand_step("LEFT_INITIAL", "left_hand", "clench", list(HAND_OPEN), {}, "open left hand before grasp-only dry run")]),
        ("LEFT_GRASP_POSE", [make_move_step("LEFT_GRASP", "left_arm", left_grasp, "direct grasp-only pose")]),
        (
            "LEFT_CLOSE",
            [
                make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "clench", list(LEFT_GRASP), {}, "legacy left grasp posture"),
                make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, "legacy left grasp force"),
            ],
        ),
    ]
    for name, steps in groups:
        execute_group(bundle, name, steps, report, args)


def run_dry_grasp_only_staged_mode(bundle: Any, left_grasp: Pose6, report: dict[str, Any], args: argparse.Namespace) -> None:
    waypoints = staged_waypoints(left_grasp, args.staged_z)
    preflight_staged_waypoints(bundle, waypoints, report)
    execute_group(
        bundle,
        "LEFT_OPEN",
        [make_hand_step("LEFT_INITIAL", "left_hand", "clench", list(HAND_OPEN), {}, "open left hand before staged grasp-only dry run")],
        report,
        args,
    )
    for stage, pose in waypoints:
        phase = "LEFT_GRASP" if stage == "GRASP" else f"STAGED_{stage}"
        execute_move_without_pose_check(bundle, phase, pose, report, args)
    execute_group(
        bundle,
        "LEFT_CLOSE",
        [
            make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "clench", list(LEFT_GRASP), {}, "legacy left grasp posture"),
            make_hand_step("LEFT_AFTER_CLOSE", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, "legacy left grasp force"),
        ],
        report,
        args,
    )


def run_orientation_calibration_mode(bundle: Any, report: dict[str, Any]) -> None:
    left_arm = bundle.left_arm
    left_hand = bundle.left_hand
    report["calibration_xyz"] = list(CALIBRATION_XYZ)
    report["tests"] = []
    report["orientation_calibration"] = {
        "calibration_xyz": list(CALIBRATION_XYZ),
        "safe_pre_position_joints": ORIENTATION_CALIBRATION_SAFE_JOINTS,
        "reference_pose": pose_to_list(ORIENTATION_CALIBRATION_POSES[0][2]),
        "tests": report["tests"],
    }

    print("==================================================")
    print("ORIENTATION CALIBRATION")
    print("Opening left hand.")
    open_left_hand(left_hand, report)

    for index, joints in enumerate(ORIENTATION_CALIBRATION_SAFE_JOINTS, start=1):
        print(f"[SAFE_PRE_POSITION_{index}] target joints = {joints}")
        result = left_arm.move_joints(joints)
        failed, reason = step_result_failed(result)
        append_robot_record(
            report,
            phase=f"ORIENTATION_CALIBRATION_SAFE_PRE_{index}",
            target="left_arm",
            method="move_joints",
            args=[joints],
            return_value=result,
            status="FAILED" if failed else "OK",
        )
        print(f"[SAFE_PRE_POSITION_{index}] return = {jsonable(result)}")
        if failed:
            print("[FAIL] safe pre-position failed")
            raise StepExecutionError(
                f"ORIENTATION_CALIBRATION_SAFE_PRE_{index}: left_arm.move_joints failed: {reason}",
                failed_node="ORIENTATION_CALIBRATION_SAFE_PRE",
            )

    reference_pose = ORIENTATION_CALIBRATION_POSES[0][2]
    reference_check = call_pose_check(left_arm, reference_pose)
    append_robot_record(
        report,
        phase="ORIENTATION_CALIBRATION_REFERENCE_CHECK",
        target="left_arm",
        method="pose_check",
        target_pose=pose_to_list(reference_pose),
        pose_check=reference_check,
        status="OK" if reference_check["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
    )
    print("Reference safe pose")
    print(f"target = {pose_to_list(reference_pose)}")
    print(f"pose_check = {reference_check['status']}")
    print(f"reason = {reference_check['reason']}")
    if reference_check["status"] != "PASS":
        raise StepExecutionError(
            f"ORIENTATION_CALIBRATION_REFERENCE_CHECK: left_arm pose_check failed: {reference_check['reason']}",
            failed_node="ORIENTATION_CALIBRATION_REFERENCE_CHECK",
        )

    success, move_return, actual_pose, joint_angles = execute_orientation_move(
        bundle,
        "ORIENTATION_CALIBRATION_REFERENCE",
        reference_pose,
        report,
        "enter known high reference pose before fixed-xyz orientation tests",
    )
    if not success:
        raise StepExecutionError(
            "ORIENTATION_CALIBRATION_REFERENCE: left_arm.move_to failed",
            failed_node="ORIENTATION_CALIBRATION_REFERENCE",
        )

    print("")
    print("Reference safe pose reached.")
    print("")
    print("Actual pose:")
    print(f"position/orientation = {jsonable(actual_pose)}")
    print("")
    print("Actual joints:")
    print(json.dumps(jsonable(joint_angles), ensure_ascii=False))
    print(f"move_to return value = {jsonable(move_return)}")
    print("")
    print("Check the simulator.")
    answer = input("Press ENTER to continue.\nType q + ENTER to abort.\n").strip().lower()
    print("==================================================")
    if answer == "q":
        report["runtime"]["status"] = "ABORTED_BY_USER_AFTER_REFERENCE"
        return

    for name, label, pose in ORIENTATION_CALIBRATION_POSES:
        test_record: dict[str, Any] = {
            "name": name,
            "label": label,
            "commanded_pose": pose_to_list(pose),
            "commanded_rpy": [pose.roll, pose.pitch, pose.yaw],
            "pose_check": False,
            "pose_check_reason": "NOT_RUN",
            "move_executed": False,
            "move_success": False,
            "move_to_return": None,
            "actual_pose": [],
            "joint_angles": [],
        }
        report["tests"].append(test_record)

        pose_check = call_pose_check(left_arm, pose)
        test_record["pose_check"] = pose_check["status"] == "PASS"
        test_record["pose_check_reason"] = pose_check["reason"]
        test_record["pose_check_raw"] = pose_check["raw"]
        append_robot_record(
            report,
            phase=name,
            target="left_arm",
            method="pose_check",
            target_pose=pose_to_list(pose),
            pose_check=pose_check,
            status="OK" if pose_check["status"] == "PASS" else "POSE_CHECK_FAILED_STOPPED_BEFORE_MOVE",
        )

        print("")
        print(name.replace("_", " "))
        if label == "REFERENCE":
            print(label)
        print(f"target = {pose_to_list(pose)}")
        print(f"pose_check = {pose_check['status']}")
        print(f"reason = {pose_check['reason']}")

        if pose_check["status"] != "PASS":
            input("Press ENTER to test the next pose.")
            continue

        action = prompt_enter_skip_quit(
            f"Press ENTER to execute {name.replace('_', ' ')}.\n"
            "Type s to skip.\n"
            "Type q to quit.\n"
        )
        if action == "quit":
            report["runtime"]["status"] = "ABORTED_BY_USER"
            break
        if action == "skip":
            test_record["move_skipped_by_user"] = True
            continue

        success, move_return, actual_pose, joint_angles = execute_orientation_move(
            bundle,
            name,
            pose,
            report,
            "fixed-xyz orientation calibration pose",
        )
        test_record["move_executed"] = True
        test_record["move_success"] = success
        test_record["move_to_return"] = move_return
        test_record["actual_pose"] = jsonable(actual_pose)
        test_record["joint_angles"] = jsonable(joint_angles)

        print("")
        print("commanded pose")
        print(pose_to_list(pose))
        print("actual get_pose()")
        print(json.dumps(jsonable(actual_pose), ensure_ascii=False))
        print("actual joint angles")
        print(json.dumps(jsonable(joint_angles), ensure_ascii=False))
        print(f"move_to return value = {jsonable(move_return)}")

        if not success:
            raise StepExecutionError(f"{name}: left_arm.move_to failed", failed_node=name)

        input(
            "Observe:\n"
            "1. Palm direction\n"
            "2. Finger direction\n"
            "3. Whether arm/hand touched the table\n"
            "4. Whether wrist made a large flip\n\n"
            "Press ENTER for next pose.\n"
        )


def run_left_direct_grasp_mode(bundle: Any, report: dict[str, Any], args: argparse.Namespace) -> None:
    yaw = float(args.yaw)
    safe_palm_down_pose = Pose6(CALIBRATION_XYZ[0], CALIBRATION_XYZ[1], CALIBRATION_XYZ[2], LEFT_TOP_GRASP_ROLL, LEFT_TOP_GRASP_PITCH, yaw)
    pre_grasp_high = left_direct_pose(LEFT_DIRECT_PRE_GRASP_HIGH_Z, yaw)
    grasp_pose = left_direct_pose(LEFT_DIRECT_GRASP_Z, yaw)
    descent_poses = [left_direct_pose(z, yaw) for z in LEFT_DIRECT_DESCENT_Z]
    lift_poses = [left_direct_pose(z, yaw) for z in LEFT_DIRECT_LIFT_Z]

    report["mode"] = "left-direct-grasp"
    report["nut_world_pose"] = list(NUT_B_LEFT_TEST_WORLD_POSE)
    report["left_test_xy"] = list(LEFT_TEST_XY)
    report["orientation"] = {
        "roll": LEFT_TOP_GRASP_ROLL,
        "pitch": LEFT_TOP_GRASP_PITCH,
        "yaw": yaw,
        "source": LEFT_DIRECT_ORIENTATION_SOURCE,
    }
    report["grasp_pose"] = pose_to_list(grasp_pose)
    report["safe_palm_down_pose"] = pose_to_list(safe_palm_down_pose)
    report["pre_grasp_high"] = pose_to_list(pre_grasp_high)
    report["descent_waypoints"] = []
    report["lift_waypoints"] = []
    report["grasp_force"] = dict(LEFT_DIRECT_GRASP_FORCE)

    print("==================================================")
    print("LEFT DIRECT GRASP")
    print("Moving Nut B to left-hand direct-grasp test position.")
    result = bundle.pose_setter.set(NUT_IDS["B"], tuple(NUT_B_LEFT_TEST_WORLD_POSE))
    failed, reason = step_result_failed(result)
    append_robot_record(
        report,
        phase="LEFT_DIRECT_SET_NUT_B",
        target="pose_setter",
        method="set",
        args=[NUT_IDS["B"], NUT_B_LEFT_TEST_WORLD_POSE],
        return_value=result,
        status="FAILED" if failed else "OK",
    )
    print("Nut B moved to left-hand direct-grasp test position.")
    print(f"world pose = {NUT_B_LEFT_TEST_WORLD_POSE}")
    print(f"set return = {jsonable(result)}")
    if failed:
        raise StepExecutionError(f"LEFT_DIRECT_SET_NUT_B: pose_setter.set failed: {reason}", failed_node="LEFT_DIRECT_SET_NUT_B")
    time.sleep(LEFT_DIRECT_SETTLE_AFTER_POSE_S)

    print("Opening left hand.")
    open_left_hand(bundle.left_hand, report, phase="LEFT_DIRECT_OPEN_HAND")

    for index, joints in enumerate(ORIENTATION_CALIBRATION_SAFE_JOINTS, start=1):
        print(f"[LEFT_DIRECT_SAFE_PRE_{index}] target joints = {joints}")
        result = bundle.left_arm.move_joints(joints)
        failed, reason = step_result_failed(result)
        append_robot_record(
            report,
            phase=f"LEFT_DIRECT_SAFE_PRE_{index}",
            target="left_arm",
            method="move_joints",
            args=[joints],
            return_value=result,
            status="FAILED" if failed else "OK",
        )
        print(f"[LEFT_DIRECT_SAFE_PRE_{index}] return = {jsonable(result)}")
        if failed:
            print("[FAIL] safe pre-position failed")
            raise StepExecutionError(f"LEFT_DIRECT_SAFE_PRE_{index}: left_arm.move_joints failed: {reason}", failed_node="LEFT_DIRECT_SAFE_PRE")

    checked_left_move(
        bundle,
        report,
        phase="LEFT_DIRECT_SAFE_PALM_DOWN",
        pose=safe_palm_down_pose,
        waypoint_bucket=None,
    )

    checked_left_move(
        bundle,
        report,
        phase="LEFT_DIRECT_PRE_GRASP_HIGH",
        pose=pre_grasp_high,
        waypoint_bucket=None,
    )
    confirm_or_abort(
        "PRE_GRASP_HIGH reached.\nObserve Nut B alignment and clearance.\nPress ENTER to start staged descent.\nq to abort.\n",
        failed_node="LEFT_DIRECT_PRE_GRASP_HIGH",
    )

    for pose in descent_poses:
        checked_left_move(
            bundle,
            report,
            phase=f"LEFT_DIRECT_DESCEND_Z_{pose.z:.2f}",
            pose=pose,
            waypoint_bucket="descent_waypoints",
            confirm_prompt=(
                f"Press ENTER to execute descent z={pose.z:.2f}.\n"
                "q to abort.\n"
            ),
        )

    confirm_or_abort(
        "GRASP POSE REACHED\n\n"
        "Check:\n"
        "- palm is facing down\n"
        "- nut is between fingers\n"
        "- no collision with table\n\n"
        "Press ENTER to close hand.\n"
        "q to abort.\n",
        failed_node="LEFT_DIRECT_GRASP_CONFIRM",
    )

    execute_left_hand_command(
        bundle.left_hand,
        report,
        phase="LEFT_DIRECT_THUMB_TUCK",
        method_name="clench",
        kwargs={"thumb_rotation": 1.0},
    )
    execute_left_hand_command(
        bundle.left_hand,
        report,
        phase="LEFT_DIRECT_GRASP_FORCE",
        method_name="grasp_force",
        kwargs=dict(LEFT_DIRECT_GRASP_FORCE),
    )
    time.sleep(1.0)

    for pose in lift_poses:
        checked_left_move(
            bundle,
            report,
            phase=f"LEFT_DIRECT_LIFT_Z_{pose.z:.2f}",
            pose=pose,
            waypoint_bucket="lift_waypoints",
        )

    input(
        "================================\n"
        "LEFT DIRECT GRASP TEST COMPLETE\n\n"
        "Observe Nut B:\n\n"
        "1. Nut lifted successfully\n"
        "2. Nut slipped / dropped\n"
        "3. Hand missed Nut\n"
        "4. Collision occurred\n\n"
        "Press ENTER to finish.\n"
        "================================\n"
    )
    answer = input("Open hand before shutdown? [y/N] ").strip().lower()
    report["open_hand_before_shutdown"] = answer == "y"
    if answer == "y":
        open_left_hand(bundle.left_hand, report, phase="LEFT_DIRECT_OPTIONAL_OPEN_BEFORE_SHUTDOWN")


def run_experiment(args: argparse.Namespace, report: dict[str, Any]) -> int:
    if args.plan_only:
        report["runtime"]["status"] = "PLAN_ONLY_NO_RABO_MOTION"
        return 0

    include_left = args.mode in ("check", "dry", "dry-grasp-only", "real", "orientation-calibration", "left-direct-grasp")
    include_right = args.mode in ("place", "real")
    include_left_hand = args.mode in ("dry", "dry-grasp-only", "real", "orientation-calibration", "left-direct-grasp")
    include_right_hand = args.mode in ("place", "real")
    include_pose_setter = args.mode == "left-direct-grasp"
    bundle = None
    try:
        if args.mock_pose_check is not None:
            bundle = make_mock_bundle(
                include_left=include_left,
                include_right=include_right,
                include_left_hand=include_left_hand,
                include_right_hand=include_right_hand,
                result=args.mock_pose_check,
                include_pose_setter=include_pose_setter,
                pose_check_sequence=getattr(args, "mock_pose_check_sequence", None),
                move_fail_phase=getattr(args, "mock_move_fail_phase", None),
            )
        else:
            bundle = make_bundle(
                include_left=include_left,
                include_right=include_right,
                include_left_hand=include_left_hand,
                include_right_hand=include_right_hand,
                include_pose_setter=include_pose_setter,
            )
        initial = read_device_state(bundle, "INITIAL")
        report["robot_data"].append(initial)
        report["initial_pose"] = initial.get("left_arm_pose", "NOT_APPLICABLE")
        report["initial_joints"] = initial.get("left_arm_joint_angles", "NOT_APPLICABLE")
        if args.mode == "check":
            left_grasp = pose_from_list(report["derived_values"]["LEFT_GRASP_POSE"]["value"])
            run_check_mode(bundle, left_grasp, report)
        if args.mode in ("place", "real"):
            right_release_values = report["derived_values"]["PREDICTED_RIGHT_RELEASE_BASE_POSE"]["value"]
            run_right_place(bundle, pose_from_list(right_release_values), report, args)
        if args.mode == "dry-grasp-only":
            left_grasp = pose_from_list(report["derived_values"]["LEFT_GRASP_POSE"]["value"])
            run_dry_grasp_only_mode(bundle, left_grasp, report, args)
        if args.mode in ("dry", "real"):
            left_grasp = pose_from_list(report["derived_values"]["LEFT_GRASP_POSE"]["value"])
            left_approach = pose_from_list(report["derived_values"]["LEFT_APPROACH_POSE"]["value"])
            left_lift = pose_from_list(report["derived_values"]["LEFT_LIFT_POSE"]["value"])
            run_left_dry_or_grasp(bundle, left_grasp, left_approach, left_lift, report, args)
        if args.mode == "orientation-calibration":
            run_orientation_calibration_mode(bundle, report)
        if args.mode == "left-direct-grasp":
            run_left_direct_grasp_mode(bundle, report, args)
        if not str(report["runtime"].get("status", "")).startswith("ABORTED_BY_USER"):
            report["runtime"]["status"] = "EXECUTED"
        if args.mode not in ("check", "orientation-calibration", "left-direct-grasp") and not args.no_prompt:
            report["user_observation"] = ask_user_observation(args.mode)
        return 0
    except Exception as exc:
        report["runtime"]["status"] = "FAILED"
        report["runtime"]["error"] = repr(exc)
        if isinstance(exc, StepExecutionError):
            report["runtime"]["failed_node"] = exc.failed_node or str(exc).split(":", 1)[0]
            report["runtime"]["failed_stage"] = exc.failed_stage
        else:
            report["runtime"]["failed_node"] = "EXCEPTION"
        print(f"FAILED: {repr(exc)}")
        return 1
    finally:
        report["robot_data"].append(read_device_state(bundle, "FINAL"))
        shutdown_bundle(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Left-hand grasp V1 parameterized experiment tool.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("check", "dry", "dry-grasp-only", "place", "real", "orientation-calibration", "left-direct-grasp"):
        sub = subparsers.add_parser(mode)
        sub.set_defaults(
            approach_dz=0.05,
            lift_dz=0.05,
            wait_before_release_s=1.5,
            settle_after_release_s=3.0,
            step_delay_s=0.0,
            stage_wait=0.5,
            staged=False,
            staged_z=None,
            step_confirm=False,
            plan_only=False,
            no_prompt=False,
            mock_pose_check=None,
            mock_pose_check_sequence=None,
            mock_move_fail_phase=None,
        )
        if mode in ("check", "dry", "dry-grasp-only", "real"):
            sub.add_argument("--pose", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW"))
        if mode in ("place", "real"):
            sub.add_argument("--nut-target-xy", type=float, nargs=2, metavar=("X", "Y"))
            sub.add_argument("--right-release-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW"))
        if mode in ("dry", "place", "real"):
            sub.add_argument("--approach-dz", type=float, default=0.05)
            sub.add_argument("--lift-dz", type=float, default=0.05)
        if mode in ("place", "real"):
            sub.add_argument("--wait-before-release-s", type=float, default=1.5)
            sub.add_argument("--settle-after-release-s", type=float, default=3.0)
        if mode == "left-direct-grasp":
            sub.add_argument("--yaw", type=float, default=0.0, help="Left top-grasp yaw in radians. Roll/pitch stay fixed at 0.0/-1.57.")
        if mode not in ("check", "orientation-calibration", "left-direct-grasp"):
            sub.add_argument("--step-delay-s", type=float, default=0.0)
        if mode == "dry-grasp-only":
            sub.add_argument("--staged", action="store_true", help="Use high/mid/low staged descent before final grasp pose.")
            sub.add_argument("--staged-z", type=float, nargs=3, metavar=("HIGH_Z", "MID_Z", "LOW_Z"), help="Override staged HIGH/MID/LOW z values.")
            sub.add_argument("--stage-wait", type=float, default=0.5, help="Seconds to wait between staged waypoints.")
            sub.add_argument("--step-confirm", action="store_true", help="Require Enter after each staged waypoint.")
        if mode in ("dry", "place", "real"):
            sub.add_argument("--plan-only", action="store_true", help="Generate derived values and reports without importing or driving Rabo SDK.")
        if mode not in ("check", "orientation-calibration", "left-direct-grasp"):
            sub.add_argument("--no-prompt", action="store_true", help="Skip manual observation prompt after motion.")
        sub.add_argument("--mock-pose-check", choices=("pass", "fail"), help=argparse.SUPPRESS)
        sub.add_argument("--mock-pose-check-sequence", choices=("pass", "fail"), nargs="+", help=argparse.SUPPRESS)
        sub.add_argument("--mock-move-fail-phase", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    derived = derive_values(args)
    report = build_report(args, derived)
    status = run_experiment(args, report)
    derive_diagnosis(report)
    json_path, md_path = write_reports(report)
    print(f"REPORT_JSON: {json_path}")
    print(f"REPORT_MD: {md_path}")
    if report["runtime"]["failed_node"]:
        print(f"FAILED_NODE: {report['runtime']['failed_node']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
