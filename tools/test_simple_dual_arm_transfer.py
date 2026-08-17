#!/usr/bin/env python3
"""MVP Nut B table transfer: right pick -> table transfer -> left pick -> Place B.

This tool intentionally does not modify the existing three_nut_expert flow.
Default mode is --check-only, which initializes arms, runs pose_check for the
manual transfer poses, and prints targets without calling motion or hand APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    HAND_OPEN,
    LEFT_GRASP,
    LEFT_GRASP_FORCE,
    LEFT_PLACE_POSES,
    LEFT_PRE_JOINTS,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import (  # noqa: E402
    RaboDeviceBundle,
    compute_right_grasp_pose,
    pose_to_list,
    ActionStep,
)
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_BASE_FRAMES  # noqa: E402


TABLE_DROP_RELEASE_POSE = Pose6(-0.40, 0.00, -0.20, 0.00, 0.80, 0.00)
LEFT_TABLE_GRASP_RPY_V1 = (0.0, 0.8, 0.0)
LEFT_TABLE_PICK_V1_STATUS = "PROPOSED_FOR_RUNTIME_TEST"
TABLE_NUT_Z_WORLD = NUT_SPECS["B"].nominal_pose.z
LEFT_APPROACH_LIFT_DZ = 0.04
STABLE_RIGHT_VERSION_SOURCE = {
    "right_action_chain_source_commit": "10ed692 Update simple transfer calibration flow",
    "wait_before_release_source_commit": "6c5f55b Add left table pick v1 candidate",
    "current_regression_note": "1344175 changed failure handling and left lift gate; it did not change RIGHT ActionStep poses.",
    "restored_scope": [
        "RIGHT_PRE",
        "RIGHT_APPROACH_NUT_B",
        "RIGHT_GRASP_NUT_B",
        "RIGHT_LIFT",
        "RIGHT_MOVE_TRANSFER",
        "RIGHT_RELEASE_TRANSFER",
    ],
}

RIGHT_RETREAT_POSE = Pose6(
    TABLE_DROP_RELEASE_POSE.x,
    TABLE_DROP_RELEASE_POSE.y,
    TABLE_DROP_RELEASE_POSE.z + 0.10,
    TABLE_DROP_RELEASE_POSE.roll,
    TABLE_DROP_RELEASE_POSE.pitch,
    TABLE_DROP_RELEASE_POSE.yaw,
)

# Manual MVP transfer point. These are LinkerArmA7 base_link target poses, not a
# searched workspace result.
TRANSFER_POINT = {
    "name": "manual_table_transfer_v1",
    "right_release_pose": TABLE_DROP_RELEASE_POSE,
    "right_retreat_pose": RIGHT_RETREAT_POSE,
    "left_approach_pose": None,
    "left_grasp_pose": None,
    "left_lift_pose": None,
    "left_table_pick_v1_status": LEFT_TABLE_PICK_V1_STATUS,
    "nut_transfer_world": None,
    "hand_to_nut_offset_source": None,
    "source_note": "LEFT TABLE PICK V1 mirrored from RIGHT successful table-grasp geometry; proposed for runtime test.",
}

LEFT_GRASP_DIAGNOSIS_CASES = [
    (
        "CASE_A_CURRENT_V1",
        Pose6(0.5205002967, -0.0277446182, -0.33, 0.0, 0.8, 0.0),
    ),
    (
        "CASE_B_FAR_YAW_PI",
        Pose6(0.5205002967, -0.0277446182, -0.33, 0.0, 0.8, 3.14),
    ),
    (
        "CASE_C_NEAR_YAW_ZERO",
        Pose6(0.46, -0.02, -0.33, 0.0, 0.8, 0.0),
    ),
    (
        "CASE_D_NEAR_YAW_PI",
        Pose6(0.46, -0.02, -0.33, 0.0, 0.8, 3.14),
    ),
]


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


def print_json(label: str, payload: Any) -> None:
    print(f"{label}: {json.dumps(jsonable(payload), ensure_ascii=False)}")


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


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return asdict(pose)


def safe_call(label: str, obj: Any, method_name: str) -> Any:
    try:
        method = getattr(obj, method_name)
        return method()
    except Exception as exc:
        return f"{label}.{method_name} ERROR {repr(exc)}"


def read_failure_state(bundle: RaboDeviceBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {"state": "DEVICE_BUNDLE_NOT_INITIALIZED"}
    return {
        "left_arm_pose": safe_call("left_arm", bundle.left_arm, "get_pose"),
        "left_arm_qpos": safe_call("left_arm", bundle.left_arm, "get_joint_angles"),
        "right_arm_pose": safe_call("right_arm", bundle.right_arm, "get_pose"),
        "right_arm_qpos": safe_call("right_arm", bundle.right_arm, "get_joint_angles"),
    }


def read_available_state(bundle: Any | None) -> dict[str, Any]:
    if bundle is None:
        return {"state": "DEVICE_BUNDLE_NOT_INITIALIZED"}
    state = {}
    for label in ("left_arm", "right_arm"):
        arm = getattr(bundle, label, None)
        if arm is None:
            continue
        state[f"{label}_pose"] = safe_call(label, arm, "get_pose")
        state[f"{label}_qpos"] = safe_call(label, arm, "get_joint_angles")
    return state


def shutdown_available(bundle: Any | None) -> None:
    if bundle is None:
        return
    if hasattr(bundle, "shutdown"):
        bundle.shutdown()
        return
    for name in ("left_hand", "right_hand", "left_arm", "right_arm"):
        device = getattr(bundle, name, None)
        if hasattr(device, "shutdown"):
            device.shutdown()


def make_move_step(phase: str, target: str, pose: Pose6, note: str = "") -> ActionStep:
    return ActionStep(phase, target, "move_to", [], pose_kwargs(pose), True, note)


def pose_from_values(values: list[float] | tuple[float, ...]) -> Pose6:
    return Pose6(values[0], values[1], values[2], values[3], values[4], values[5])


def xyz_delta(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def xyz_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def mirror_right_offset_for_left(offset_world_xyz: list[float]) -> list[float]:
    return [-offset_world_xyz[0], -offset_world_xyz[1], offset_world_xyz[2]]


def build_left_table_pick_v1(right_grasp_pose: Pose6) -> dict[str, Any]:
    right_base_world = CONFIRMED_BASE_FRAMES["right_arm_base"]["world_pose"]
    left_base_world = CONFIRMED_BASE_FRAMES["left_arm_base"]["world_pose"]
    nut_world = pose_to_list(NUT_SPECS["B"].nominal_pose)
    right_grasp_world = transform_pose_base_to_world(pose_to_list(right_grasp_pose), right_base_world)
    right_grasp_offset_world_xyz = xyz_delta(right_grasp_world, nut_world)
    right_release_world = transform_pose_base_to_world(pose_to_list(TABLE_DROP_RELEASE_POSE), right_base_world)

    release_nut_xyz = xyz_delta(right_release_world, right_grasp_offset_world_xyz)
    expected_transfer_nut_world = [
        release_nut_xyz[0],
        release_nut_xyz[1],
        TABLE_NUT_Z_WORLD,
        0.0,
        0.0,
        NUT_SPECS["B"].nominal_pose.yaw,
    ]

    left_mirrored_offset_world_xyz = mirror_right_offset_for_left(right_grasp_offset_world_xyz)
    left_grasp_world_xyz = xyz_add(expected_transfer_nut_world, left_mirrored_offset_world_xyz)
    left_grasp_world = [
        left_grasp_world_xyz[0],
        left_grasp_world_xyz[1],
        left_grasp_world_xyz[2],
        *LEFT_TABLE_GRASP_RPY_V1,
    ]
    left_grasp_base = transform_pose_world_to_base(left_grasp_world, left_base_world)
    left_approach_base = [
        left_grasp_base[0],
        left_grasp_base[1],
        left_grasp_base[2] + LEFT_APPROACH_LIFT_DZ,
        left_grasp_base[3],
        left_grasp_base[4],
        left_grasp_base[5],
    ]
    left_lift_base = list(left_approach_base)
    return {
        "status": LEFT_TABLE_PICK_V1_STATUS,
        "base_frame_source": CONFIRMED_BASE_FRAMES["source"],
        "T_world_right_base": right_base_world,
        "T_world_left_base": left_base_world,
        "RIGHT_NUT_B_GRASP_BASE": pose_to_list(right_grasp_pose),
        "RIGHT_GRASP_WORLD": right_grasp_world,
        "RIGHT_GRASP_OFFSET_WORLD_XYZ": right_grasp_offset_world_xyz,
        "TABLE_DROP_RELEASE_WORLD": right_release_world,
        "EXPECTED_TRANSFER_NUT_WORLD": expected_transfer_nut_world,
        "LEFT_MIRRORED_OFFSET_WORLD_XYZ": left_mirrored_offset_world_xyz,
        "LEFT_TABLE_GRASP_RPY_V1": list(LEFT_TABLE_GRASP_RPY_V1),
        "LEFT_GRASP_WORLD_V1": left_grasp_world,
        "LEFT_GRASP_POSE_V1": left_grasp_base,
        "LEFT_APPROACH_POSE_V1": left_approach_base,
        "LEFT_LIFT_POSE_V1": left_lift_base,
        "note": "V1 mirrors the RIGHT successful grasp offset across the left/right symmetric setup; not VERIFIED.",
    }


class StepExecutionError(RuntimeError):
    pass


def pose_step_kwargs(kwargs: dict[str, Any]) -> dict[str, float]:
    return {
        "x": kwargs["x"],
        "y": kwargs["y"],
        "z": kwargs["z"],
        "roll": kwargs.get("roll", 0.0),
        "pitch": kwargs.get("pitch", 0.0),
        "yaw": kwargs.get("yaw", 0.0),
    }


def call_step(bundle: Any, step: ActionStep) -> Any:
    # Reused from agents/three_nut_expert/expert.py execute_step(), with return
    # value checking added for MVP runtime failures.
    if step.method == "set_entity_pose":
        thing_id, pose = step.args
        return bundle.pose_setter.set(thing_id, tuple(pose))

    target = getattr(bundle, step.target)
    method = getattr(target, step.method)
    if step.method == "move_to":
        return method(**pose_step_kwargs(step.kwargs))
    if step.method == "grasp_force":
        if step.kwargs.get("fingers") is None:
            return method(strength=step.kwargs["strength"])
        return method(strength=step.kwargs["strength"], fingers=step.kwargs["fingers"])
    return method(*step.args, **step.kwargs)


def step_result_failed(result: Any) -> tuple[bool, str]:
    value = jsonable(result)
    if result is None:
        return False, ""
    if isinstance(result, bool):
        return (not result), "returned_false"
    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        ok = lower.get("success", lower.get("ok", lower.get("reachable", lower.get("result"))))
        if isinstance(ok, bool):
            return (not ok), str(lower.get("reason", lower.get("message", lower.get("error", "")))
                                   or ("returned_false" if not ok else ""))
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    low = text.lower()
    failure_tokens = ("error", "failed", "fail", "false", "ik", "无解", "失败", "错误")
    if any(token in low for token in failure_tokens):
        return True, text
    return False, ""


def execute_checked_step(bundle: Any, step: ActionStep) -> Any:
    result = call_step(bundle, step)
    failed, reason = step_result_failed(result)
    if failed:
        raise StepExecutionError(f"{step.target}.{step.method} failed: {reason}; return={repr(result)}")
    return result


def make_plan() -> dict[str, Any]:
    nut_pose = NUT_SPECS["B"].nominal_pose
    right_grasp_pose = compute_right_grasp_pose(nut_pose)
    left_pick_v1 = build_left_table_pick_v1(right_grasp_pose)
    right_release_pose = TRANSFER_POINT["right_release_pose"]
    right_retreat_pose = TRANSFER_POINT["right_retreat_pose"]
    left_approach_pose = pose_from_values(left_pick_v1["LEFT_APPROACH_POSE_V1"])
    left_grasp_pose = pose_from_values(left_pick_v1["LEFT_GRASP_POSE_V1"])
    left_lift_pose = pose_from_values(left_pick_v1["LEFT_LIFT_POSE_V1"])
    place_b_pose = LEFT_PLACE_POSES["B"]
    TRANSFER_POINT["left_approach_pose"] = left_approach_pose
    TRANSFER_POINT["left_grasp_pose"] = left_grasp_pose
    TRANSFER_POINT["left_lift_pose"] = left_lift_pose
    TRANSFER_POINT["nut_transfer_world"] = left_pick_v1["EXPECTED_TRANSFER_NUT_WORLD"]

    return {
        "stable_right_version_source": STABLE_RIGHT_VERSION_SOURCE,
        "nut_b_pose": nut_pose,
        "right_grasp_pose": right_grasp_pose,
        "left_pick_v1": left_pick_v1,
        "transfer_point": TRANSFER_POINT,
        "right_release_pose": right_release_pose,
        "right_retreat_pose": right_retreat_pose,
        "left_approach_pose": left_approach_pose,
        "left_grasp_pose": left_grasp_pose,
        "left_lift_pose": left_lift_pose,
        "place_b_pose": place_b_pose,
    }


def make_right_calibration_groups(plan: dict[str, Any]) -> list[tuple[str, list[ActionStep]]]:
    nut_pose = plan["nut_b_pose"]
    right_grasp_pose = plan["right_grasp_pose"]
    right_release_pose = plan["right_release_pose"]
    right_retreat_pose = plan["right_retreat_pose"]
    return [
        (
            "RIGHT_PRE",
            [
                ActionStep("init", "world", "set_entity_pose", [NUT_SPECS["B"].thing_id, pose_to_list(nut_pose)], {}, True, "place Nut B at nominal pose"),
                *[
                    ActionStep("pre_position", "right_arm", "move_joints", [joints], {}, True, "legacy right pre-position")
                    for joints in RIGHT_PRE_JOINTS
                ],
            ],
        ),
        ("RIGHT_APPROACH_NUT_B", [make_move_step("approach", "right_arm", right_grasp_pose, "legacy Nut B approach pose")]),
        (
            "RIGHT_GRASP_NUT_B",
            [
                ActionStep("grasp", "right_hand", "clench", [], {"thumb_rotation": 1.0}, True, "legacy right thumb tuck"),
                ActionStep("grasp", "right_hand", "grasp_force", [], RIGHT_GRASP_FORCE, True, "legacy right grasp force"),
            ],
        ),
        ("RIGHT_LIFT", [make_move_step("lift", "right_arm", pose, "legacy right lift") for pose in RIGHT_LIFT_POSES]),
        ("RIGHT_MOVE_TRANSFER", [make_move_step("transfer", "right_arm", right_release_pose, "TABLE_DROP_RELEASE_POSE")]),
        ("RIGHT_RELEASE_TRANSFER", [ActionStep("release", "right_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open release")]),
        ("RIGHT_RETREAT_TRANSFER", [make_move_step("retreat", "right_arm", right_retreat_pose, "raise 0.10m from table drop pose")]),
        (
            "RIGHT_RETURN_SAFE",
            [
                ActionStep("return_safe", "right_arm", "move_joints", [joints], {}, True, "legacy right pre-position")
                for joints in RIGHT_PRE_JOINTS
            ],
        ),
    ]


def make_left_pick_calibration_groups(plan: dict[str, Any]) -> list[tuple[str, list[ActionStep]]]:
    left_approach_pose = plan["left_approach_pose"]
    left_grasp_pose = plan["left_grasp_pose"]
    return [
        (
            "LEFT_PRE",
            [
                ActionStep("pre_position", "left_arm", "move_joints", [joints], {}, True, "legacy left pre-position")
                for joints in LEFT_PRE_JOINTS
            ],
        ),
        ("LEFT_OPEN_HAND", [ActionStep("open", "left_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open before table pick")]),
        ("LEFT_APPROACH_TRANSFER", [make_move_step("transfer_approach", "left_arm", left_approach_pose, "safe approach above manual transfer point")]),
        ("LEFT_MOVE_GRASP", [make_move_step("transfer_grasp", "left_arm", left_grasp_pose, "manual transfer grasp pose")]),
        (
            "LEFT_GRASP_TRANSFER",
            [
                ActionStep("catch", "left_hand", "clench", list(LEFT_GRASP), {}, True, "legacy left grasp posture"),
                ActionStep("catch", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, True, "legacy left grasp force"),
            ],
        ),
        ("LEFT_LIFT", [make_move_step("lift", "left_arm", plan["left_lift_pose"], "MVP lift back to safe transfer height")]),
    ]


def make_left_calibration_groups(plan: dict[str, Any]) -> list[tuple[str, list[ActionStep]]]:
    return [
        *make_left_pick_calibration_groups(plan),
        ("LEFT_MOVE_PLACE_B", [make_move_step("place", "left_arm", plan["place_b_pose"], "Official Place B")]),
        ("LEFT_RELEASE", [ActionStep("release", "left_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open release")]),
    ]


def pose6_from_list(pose: list[float]) -> Pose6:
    return Pose6(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])


def check_pose(arm: Any, pose: Pose6) -> dict[str, Any]:
    raw = arm.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)
    status, reason, value = normalize_pose_check(raw)
    return {
        "status": status,
        "reason": reason,
        "raw": value,
        "pose": pose_to_list(pose),
    }


def transfer_pose_gates(plan: dict[str, Any]) -> list[tuple[str, str, Pose6]]:
    return [
        ("RIGHT transfer pose_check", "right_arm", plan["right_release_pose"]),
        ("RIGHT retreat pose_check", "right_arm", plan["right_retreat_pose"]),
        ("LEFT approach pose_check", "left_arm", plan["left_approach_pose"]),
        ("LEFT grasp pose_check", "left_arm", plan["left_grasp_pose"]),
        ("LEFT lift pose_check", "left_arm", plan["left_lift_pose"]),
        ("Official Place B pose_check", "left_arm", plan["place_b_pose"]),
    ]


def left_pick_pose_gates(plan: dict[str, Any]) -> list[tuple[str, Pose6]]:
    return [
        ("LEFT approach pose_check", plan["left_approach_pose"]),
        ("LEFT grasp pose_check", plan["left_grasp_pose"]),
        ("LEFT lift pose_check", plan["left_lift_pose"]),
    ]


def run_left_pick_pose_gates(plan: dict[str, Any], left_arm: Any) -> bool:
    passed = True
    for label, pose in left_pick_pose_gates(plan):
        result = check_pose(left_arm, pose)
        print_json(label, result)
        if result["status"] != "PASS":
            passed = False
    print(f"LEFT_PICK_POSE_CHECK_GATES: {'PASS' if passed else 'FAIL'}")
    return passed


def run_pose_gates(plan: dict[str, Any], right_arm: Any, left_arm: Any) -> bool:
    passed = True
    arms = {"right_arm": right_arm, "left_arm": left_arm}
    for label, arm_name, pose in transfer_pose_gates(plan):
        result = check_pose(arms[arm_name], pose)
        print_json(label, result)
        if result["status"] != "PASS":
            passed = False
    print(f"POSE_CHECK_GATES: {'PASS' if passed else 'FAIL'}")
    return passed


def run_check_only(plan: dict[str, Any]) -> int:
    from rabo_robocap import LinkerArmA7
    from agents.three_nut_expert.config import DEVICE_IDS

    print("MODE: CHECK_ONLY")
    print("No move_to, move_joints, SetEntityPose, clench, or grasp_force will be called.")
    print_targets(plan)

    right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
    try:
        if run_pose_gates(plan, right_arm, left_arm):
            return 0
        return 2
    finally:
        for device in (left_arm, right_arm):
            if hasattr(device, "shutdown"):
                device.shutdown()


def execute_named_groups(
    bundle: Any,
    groups: list[tuple[str, list[ActionStep]]],
    *,
    step_delay_s: float,
    settle_after_pose_s: float,
    hold_after_grasp_s: float,
    start_index: int = 1,
) -> int:
    next_index = start_index
    for index, (name, steps) in enumerate(groups, start=start_index):
        print(f"[STEP {index}] {name}")
        next_index = index + 1
        for step in steps:
            print_json("STEP_NAME", name)
            print_json("TARGET_POSE", step_target_payload(step))
            try:
                result = execute_checked_step(bundle, step)
            except Exception as exc:
                print_json("RESULT", {"status": "FAILED", "target": step.target, "method": step.method, "error": repr(exc)})
                print("FAILED_STEP")
                print(name)
                print(f"exception: {repr(exc)}")
                raise
            print_json("RESULT", {"status": "OK", "target": step.target, "method": step.method, "return": jsonable(result)})
            if step.method == "set_entity_pose" and settle_after_pose_s > 0:
                time.sleep(settle_after_pose_s)
            if step.method == "grasp_force" and hold_after_grasp_s > 0:
                time.sleep(hold_after_grasp_s)
            if step_delay_s > 0:
                time.sleep(step_delay_s)
    return next_index


def step_target_payload(step: ActionStep) -> dict[str, Any]:
    if step.method == "move_to":
        return {"target": step.target, "method": step.method, "pose": step.kwargs, "note": step.note}
    if step.method == "move_joints":
        return {"target": step.target, "method": step.method, "joints": step.args[0], "note": step.note}
    if step.method == "set_entity_pose":
        thing_id, pose = step.args
        return {"target": step.target, "method": step.method, "thing_id": thing_id, "pose": pose, "note": step.note}
    return {"target": step.target, "method": step.method, "args": step.args, "kwargs": step.kwargs, "note": step.note}


def make_right_calibration_bundle() -> Any:
    from rabo_dev_kit import SetEntityPose
    from rabo_robocap import LinkerArmA7, LinkerHandO6Right
    from agents.three_nut_expert.config import DEVICE_IDS

    return SimpleNamespace(
        pose_setter=SetEntityPose(world=WORLD_ID),
        right_arm=LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim"),
        right_hand=LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim"),
    )


def make_left_calibration_bundle() -> Any:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left
    from agents.three_nut_expert.config import DEVICE_IDS

    return SimpleNamespace(
        left_arm=LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim"),
        left_hand=LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim"),
    )


def make_left_pose_check_bundle() -> Any:
    from rabo_robocap import LinkerArmA7
    from agents.three_nut_expert.config import DEVICE_IDS

    return SimpleNamespace(left_arm=LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim"))


def run_right_calibration(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: RIGHT_CALIBRATION")
    print_targets(plan)
    bundle = None
    try:
        bundle = make_right_calibration_bundle()
        right_groups = make_right_calibration_groups(plan)
        execute_named_groups(
            bundle,
            right_groups[:5],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
        )
        print(f"[STEP 6] WAIT_BEFORE_RELEASE {args.wait_before_release_s:.2f}s")
        if args.wait_before_release_s > 0:
            time.sleep(args.wait_before_release_s)
        execute_named_groups(
            bundle,
            right_groups[5:6],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
            start_index=7,
        )
        print(f"[STEP 8] WAIT_OBJECT_SETTLE {args.settle_after_release_s:.2f}s")
        if args.settle_after_release_s > 0:
            time.sleep(args.settle_after_release_s)
        execute_named_groups(
            bundle,
            right_groups[6:],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
            start_index=9,
        )
        print("RIGHT_ARM_CLEAR_OF_TRANSFER_ZONE")
        print("STOP")
        print("MANUAL_CALIBRATION_REQUIRED:")
        print("Please inspect Nut B settled position.")
        return 0
    except Exception as exc:
        print("FAILED_STEP")
        print(f"exception: {repr(exc)}")
        print_json("current_state", read_available_state(bundle))
        return 1
    finally:
        try:
            shutdown_available(bundle)
        except Exception as exc:
            print(f"shutdown exception: {repr(exc)}")


def diagnose_left_grasp(results: dict[str, dict[str, Any]]) -> str:
    statuses = {name: item["status"] for name, item in results.items()}
    a_fail = statuses["CASE_A_CURRENT_V1"] == "FAIL"
    b_fail = statuses["CASE_B_FAR_YAW_PI"] == "FAIL"
    c_fail = statuses["CASE_C_NEAR_YAW_ZERO"] == "FAIL"
    d_fail = statuses["CASE_D_NEAR_YAW_PI"] == "FAIL"
    if a_fail and b_fail and not c_fail:
        return "POSITION_LIMIT_DOMINANT"
    if a_fail and not b_fail:
        return "ORIENTATION_DOMINANT"
    if a_fail and b_fail and c_fail and not d_fail:
        return "POSITION_AND_ORIENTATION_COUPLED"
    if a_fail and b_fail and c_fail and d_fail:
        return "CURRENT_TEST_SET_INSUFFICIENT"
    return "INCONCLUSIVE"


def run_left_pose_diagnosis() -> int:
    print("MODE: LEFT_POSE_DIAGNOSIS")
    print("No move_to, move_joints, SetEntityPose, clench, or grasp_force will be called.")
    print_json(
        "LEFT_GRASP_DIAGNOSIS_CASES",
        {name: pose_to_list(pose) for name, pose in LEFT_GRASP_DIAGNOSIS_CASES},
    )
    bundle = None
    try:
        bundle = make_left_pose_check_bundle()
        results = {}
        for name, pose in LEFT_GRASP_DIAGNOSIS_CASES:
            result = check_pose(bundle.left_arm, pose)
            row = {
                "case_name": name,
                "pose": pose_to_list(pose),
                "status": result["status"],
                "reason": result["reason"],
                "raw_result": result["raw"],
            }
            results[name] = row
            print_json(name, row)
        summary = {name: row["status"] for name, row in results.items()}
        print_json("LEFT_GRASP_DIAGNOSIS", summary)
        print(f"DIAGNOSIS = {diagnose_left_grasp(results)}")
        return 0
    except Exception as exc:
        print("FAILED_LEFT_POSE_DIAGNOSIS")
        print(f"exception: {repr(exc)}")
        print_json("current_state", read_available_state(bundle))
        return 1
    finally:
        try:
            shutdown_available(bundle)
        except Exception as exc:
            print(f"shutdown exception: {repr(exc)}")


def run_left_calibration(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: LEFT_CALIBRATION")
    print_targets(plan)
    bundle = None
    try:
        bundle = make_left_calibration_bundle()
        execute_named_groups(
            bundle,
            make_left_calibration_groups(plan),
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
        )
        print("LEFT_CALIBRATION: COMPLETE")
        return 0
    except Exception as exc:
        print("FAILED_STEP")
        print(f"exception: {repr(exc)}")
        print_json("current_state", read_available_state(bundle))
        return 1
    finally:
        try:
            shutdown_available(bundle)
        except Exception as exc:
            print(f"shutdown exception: {repr(exc)}")


def run_left_pick_calibration(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: LEFT_PICK_CALIBRATION")
    print_targets(plan)
    bundle = None
    try:
        bundle = make_left_calibration_bundle()
        if not run_left_pick_pose_gates(plan, bundle.left_arm):
            print("STOP_BEFORE_EXECUTE")
            return 2
        execute_named_groups(
            bundle,
            make_left_pick_calibration_groups(plan),
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
        )
        print("STOP")
        print("LEFT_PICK_CALIBRATION_COMPLETE")
        return 0
    except Exception as exc:
        print("FAILED_STEP")
        print(f"exception: {repr(exc)}")
        print_json("current_state", read_available_state(bundle))
        return 1
    finally:
        try:
            shutdown_available(bundle)
        except Exception as exc:
            print(f"shutdown exception: {repr(exc)}")


def run_execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: EXECUTE")
    print_targets(plan)
    print("NOTE: Full execute should be used after right and left calibration both succeed.")

    bundle: RaboDeviceBundle | None = None
    try:
        bundle = RaboDeviceBundle()
        if not run_pose_gates(plan, bundle.right_arm, bundle.left_arm):
            print("STOP_BEFORE_EXECUTE")
            return 2

        right_groups = make_right_calibration_groups(plan)
        execute_named_groups(
            bundle,
            right_groups[:5],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
        )
        print(f"[STEP 6] WAIT_BEFORE_RELEASE {args.wait_before_release_s:.2f}s")
        if args.wait_before_release_s > 0:
            time.sleep(args.wait_before_release_s)
        execute_named_groups(
            bundle,
            right_groups[5:6],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
            start_index=7,
        )
        print(f"[STEP 8] WAIT_OBJECT_SETTLE {args.settle_after_release_s:.2f}s")
        if args.settle_after_release_s > 0:
            time.sleep(args.settle_after_release_s)
        execute_named_groups(
            bundle,
            right_groups[6:],
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
            start_index=9,
        )
        print("RIGHT_ARM_CLEAR_OF_TRANSFER_ZONE")
        execute_named_groups(
            bundle,
            make_left_calibration_groups(plan),
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
            start_index=11,
        )
        print("[STEP 19] DONE")
        print("MVP_DUAL_ARM_TRANSFER_SUCCESS")
        return 0
    except Exception as exc:
        print("FAILED_STEP")
        print(f"exception: {repr(exc)}")
        print_json("current_state", read_failure_state(bundle))
        return 1
    finally:
        if bundle is not None:
            try:
                bundle.shutdown()
            except Exception as exc:
                print(f"shutdown exception: {repr(exc)}")


def print_targets(plan: dict[str, Any]) -> None:
    print(f"WORLD_ID: {WORLD_ID}")
    print_json("STABLE_RIGHT_VERSION_SOURCE", plan["stable_right_version_source"])
    print_json("Nut B nominal pose", pose_to_list(plan["nut_b_pose"]))
    print_json("Right Nut B grasp pose", pose_to_list(plan["right_grasp_pose"]))
    print_json("RIGHT_GRASP_TEMPLATE", plan["left_pick_v1"])
    print_json("TABLE_DROP_RELEASE_POSE", pose_to_list(plan["right_release_pose"]))
    print_json("Right retreat pose", pose_to_list(plan["right_retreat_pose"]))
    print_json("Left approach pose", pose_to_list(plan["left_approach_pose"]))
    print_json("Left grasp pose", pose_to_list(plan["left_grasp_pose"]))
    print_json("TRANSFER_POINT", plan["transfer_point"])
    print_json("Left transfer lift pose", pose_to_list(plan["left_lift_pose"]))
    print_json("Official Place B", pose_to_list(plan["place_b_pose"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP Nut B right-pick table-transfer left-place test.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Initialize arms, pose_check transfer poses, and print targets only.")
    mode.add_argument("--right-calibration", action="store_true", help="Run right-only Nut B pick and table release, then stop for manual Nut B world-pose recording.")
    mode.add_argument("--left-pick-calibration", action="store_true", help="Run left-only table pick and lift, then stop before Place B.")
    mode.add_argument("--left-pose-diagnosis", action="store_true", help="Run four left grasp pose_check-only diagnosis cases; no robot motion.")
    mode.add_argument("--left-calibration", action="store_true", help="Run left-only table pick from manually configured LEFT_GRASP_POSE to Official Place B.")
    mode.add_argument("--execute", action="store_true", help="Run the full MVP motion sequence after transfer pose_check passes.")
    parser.add_argument("--left-grasp-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"), help="Override manual left_grasp_pose; approach/lift are set 0.04m above it for this MVP.")
    parser.add_argument("--step-delay-s", type=float, default=0.0, help="Sleep after every executed ActionStep.")
    parser.add_argument("--settle-after-pose-s", type=float, default=0.5, help="Sleep after setting Nut B pose.")
    parser.add_argument("--hold-after-grasp-s", type=float, default=0.5, help="Sleep after grasp_force before lifting.")
    parser.add_argument("--wait-before-release-s", type=float, default=1.5, help="Right arm wait after moving to TABLE_DROP_RELEASE_POSE and before opening the hand.")
    parser.add_argument("--settle-after-release-s", type=float, default=3.0, help="Right calibration wait after releasing Nut B at the table transfer point.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = make_plan()
    if args.left_grasp_pose is not None:
        left_grasp_pose = pose6_from_list(args.left_grasp_pose)
        TRANSFER_POINT["left_grasp_pose"] = left_grasp_pose
        TRANSFER_POINT["left_approach_pose"] = Pose6(
            left_grasp_pose.x,
            left_grasp_pose.y,
            left_grasp_pose.z + LEFT_APPROACH_LIFT_DZ,
            left_grasp_pose.roll,
            left_grasp_pose.pitch,
            left_grasp_pose.yaw,
        )
        TRANSFER_POINT["left_lift_pose"] = TRANSFER_POINT["left_approach_pose"]
        plan["left_grasp_pose"] = TRANSFER_POINT["left_grasp_pose"]
        plan["left_approach_pose"] = TRANSFER_POINT["left_approach_pose"]
        plan["left_lift_pose"] = TRANSFER_POINT["left_lift_pose"]
        plan["transfer_point"] = TRANSFER_POINT
    if args.right_calibration:
        return run_right_calibration(args, plan)
    if args.left_pose_diagnosis:
        return run_left_pose_diagnosis()
    if args.left_pick_calibration:
        return run_left_pick_calibration(args, plan)
    if args.left_calibration:
        return run_left_calibration(args, plan)
    if args.execute:
        return run_execute(args, plan)
    return run_check_only(plan)


if __name__ == "__main__":
    raise SystemExit(main())
