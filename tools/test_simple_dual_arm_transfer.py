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
    execute_step,
    pose_to_list,
    ActionStep,
)


# Manual MVP transfer point. These are LinkerArmA7 base_link target poses, not a
# searched workspace result.
#
# RIGHT_RELEASE_POSE starts from the legacy right lift x/y/orientation and lowers
# to a table-transfer height.
# LEFT_APPROACH_POSE/LEFT_GRASP_POSE reuse the legacy left handoff x/y/orientation
# and add a safe approach height before the table grasp.
TRANSFER_POINT = {
    "name": "manual_table_transfer_v1",
    "right_release_pose": Pose6(-0.4, 0.0, -0.20, 0.0, 0.8, 0.0),
    "left_approach_pose": Pose6(0.44, 0.06, -0.10, 0.0, 1.3, 1.57),
    "left_grasp_pose": Pose6(0.44, 0.06, -0.20, 0.0, 1.3, 1.57),
    "nut_transfer_world": None,
    "hand_to_nut_offset_source": None,
    "source_note": "MVP manual table transfer point; adjust x/y by hand if pose_check fails.",
}


STEP_LABELS = {
    1: "RIGHT_APPROACH_NUT_B",
    2: "RIGHT_GRASP_NUT_B",
    3: "RIGHT_LIFT",
    4: "RIGHT_MOVE_TRANSFER",
    5: "RIGHT_RELEASE_TRANSFER",
    6: "LEFT_APPROACH_TRANSFER",
    7: "LEFT_GRASP_TRANSFER",
    8: "LEFT_LIFT",
    9: "LEFT_MOVE_PLACE_B",
    10: "LEFT_RELEASE",
}


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


def step_header(index: int) -> str:
    return f"[STEP {index}] {STEP_LABELS[index]}"


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


def make_plan() -> dict[str, Any]:
    nut_pose = NUT_SPECS["B"].nominal_pose
    right_grasp_pose = compute_right_grasp_pose(nut_pose)
    right_release_pose = TRANSFER_POINT["right_release_pose"]
    left_approach_pose = TRANSFER_POINT["left_approach_pose"]
    left_grasp_pose = TRANSFER_POINT["left_grasp_pose"]
    left_lift_pose = left_approach_pose
    place_b_pose = LEFT_PLACE_POSES["B"]

    # Reused from agents/three_nut_expert/expert.py build_single_nut_plan().
    setup_steps = [
        ActionStep("init", "world", "set_entity_pose", [NUT_SPECS["B"].thing_id, pose_to_list(nut_pose)], {}, True, "place Nut B at nominal pose"),
    ]
    for joints in RIGHT_PRE_JOINTS:
        setup_steps.append(ActionStep("pre_position", "right_arm", "move_joints", [joints], {}, True, "legacy right pre-position"))
    for joints in LEFT_PRE_JOINTS:
        setup_steps.append(ActionStep("pre_position", "left_arm", "move_joints", [joints], {}, True, "legacy left pre-position"))

    step_groups: list[tuple[int, list[ActionStep]]] = [
        (
            1,
            [
                *setup_steps,
                make_move_step("approach", "right_arm", right_grasp_pose, "legacy Nut B approach pose"),
            ],
        ),
        (
            2,
            [
                # Reused from docs/legacy/arm_hand_demo_legacy_snapshot.py.
                ActionStep("grasp", "right_hand", "clench", [], {"thumb_rotation": 1.0}, True, "legacy right thumb tuck"),
                ActionStep("grasp", "right_hand", "grasp_force", [], RIGHT_GRASP_FORCE, True, "legacy right grasp force"),
            ],
        ),
        (3, [make_move_step("lift", "right_arm", pose, "legacy right lift") for pose in RIGHT_LIFT_POSES]),
        (4, [make_move_step("transfer", "right_arm", right_release_pose, "manual table transfer release pose")]),
        (5, [ActionStep("release", "right_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open release")]),
        (
            6,
            [
                make_move_step("transfer_approach", "left_arm", left_approach_pose, "safe approach above manual transfer point"),
                make_move_step("transfer_grasp", "left_arm", left_grasp_pose, "manual transfer grasp pose"),
            ],
        ),
        (
            7,
            [
                ActionStep("catch", "left_hand", "clench", list(LEFT_GRASP), {}, True, "legacy left grasp posture"),
                ActionStep("catch", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, True, "legacy left grasp force"),
            ],
        ),
        (8, [make_move_step("lift", "left_arm", left_lift_pose, "MVP lift back to safe transfer height")]),
        (9, [make_move_step("place", "left_arm", place_b_pose, "Official Place B")]),
        (10, [ActionStep("release", "left_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open release")]),
    ]

    return {
        "nut_b_pose": nut_pose,
        "right_grasp_pose": right_grasp_pose,
        "transfer_point": TRANSFER_POINT,
        "left_lift_pose": left_lift_pose,
        "place_b_pose": place_b_pose,
        "step_groups": step_groups,
    }


def make_right_calibration_groups(plan: dict[str, Any]) -> list[tuple[str, list[ActionStep]]]:
    nut_pose = plan["nut_b_pose"]
    right_grasp_pose = plan["right_grasp_pose"]
    right_release_pose = plan["transfer_point"]["right_release_pose"]
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
        ("RIGHT_MOVE_TRANSFER", [make_move_step("transfer", "right_arm", right_release_pose, "manual table transfer release pose")]),
        ("RIGHT_RELEASE_TRANSFER", [ActionStep("release", "right_hand", "clench", list(HAND_OPEN), {}, True, "legacy hand open release")]),
    ]


def make_left_calibration_groups(plan: dict[str, Any]) -> list[tuple[str, list[ActionStep]]]:
    left_approach_pose = plan["transfer_point"]["left_approach_pose"]
    left_grasp_pose = plan["transfer_point"]["left_grasp_pose"]
    return [
        (
            "LEFT_PRE",
            [
                ActionStep("pre_position", "left_arm", "move_joints", [joints], {}, True, "legacy left pre-position")
                for joints in LEFT_PRE_JOINTS
            ],
        ),
        (
            "LEFT_APPROACH_TRANSFER",
            [
                make_move_step("transfer_approach", "left_arm", left_approach_pose, "safe approach above manual transfer point"),
                make_move_step("transfer_grasp", "left_arm", left_grasp_pose, "manual transfer grasp pose"),
            ],
        ),
        (
            "LEFT_GRASP_TRANSFER",
            [
                ActionStep("catch", "left_hand", "clench", list(LEFT_GRASP), {}, True, "legacy left grasp posture"),
                ActionStep("catch", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, True, "legacy left grasp force"),
            ],
        ),
        ("LEFT_LIFT", [make_move_step("lift", "left_arm", plan["left_lift_pose"], "MVP lift back to safe transfer height")]),
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


def run_check_only(plan: dict[str, Any]) -> int:
    from rabo_robocap import LinkerArmA7
    from agents.three_nut_expert.config import DEVICE_IDS

    print("MODE: CHECK_ONLY")
    print("No move_to, move_joints, SetEntityPose, clench, or grasp_force will be called.")
    print_targets(plan)

    right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
    try:
        right_result = check_pose(right_arm, plan["transfer_point"]["right_release_pose"])
        left_result = check_pose(left_arm, plan["transfer_point"]["left_grasp_pose"])
        print_json("RIGHT transfer pose_check", right_result)
        print_json("LEFT transfer pose_check", left_result)
        if right_result["status"] == "PASS" and left_result["status"] == "PASS":
            print("TRANSFER_CHECK: PASS")
            return 0
        print("TRANSFER_CHECK: FAIL")
        return 2
    finally:
        for device in (left_arm, right_arm):
            if hasattr(device, "shutdown"):
                device.shutdown()


def execute_group(
    bundle: RaboDeviceBundle,
    index: int,
    steps: list[ActionStep],
    *,
    step_delay_s: float,
    settle_after_pose_s: float,
    hold_after_grasp_s: float,
) -> None:
    print(step_header(index))
    for step in steps:
        execute_step(bundle, step)
        if step.method == "set_entity_pose" and settle_after_pose_s > 0:
            time.sleep(settle_after_pose_s)
        if step.method == "grasp_force" and hold_after_grasp_s > 0:
            time.sleep(hold_after_grasp_s)
        if step_delay_s > 0:
            time.sleep(step_delay_s)


def execute_named_groups(
    bundle: Any,
    groups: list[tuple[str, list[ActionStep]]],
    *,
    step_delay_s: float,
    settle_after_pose_s: float,
    hold_after_grasp_s: float,
) -> None:
    for index, (name, steps) in enumerate(groups, start=1):
        print(f"[STEP {index}] {name}")
        for step in steps:
            execute_step(bundle, step)
            if step.method == "set_entity_pose" and settle_after_pose_s > 0:
                time.sleep(settle_after_pose_s)
            if step.method == "grasp_force" and hold_after_grasp_s > 0:
                time.sleep(hold_after_grasp_s)
            if step_delay_s > 0:
                time.sleep(step_delay_s)


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


def run_right_calibration(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: RIGHT_CALIBRATION")
    print_targets(plan)
    bundle = None
    try:
        bundle = make_right_calibration_bundle()
        execute_named_groups(
            bundle,
            make_right_calibration_groups(plan),
            step_delay_s=args.step_delay_s,
            settle_after_pose_s=args.settle_after_pose_s,
            hold_after_grasp_s=args.hold_after_grasp_s,
        )
        print(f"[STEP 7] WAIT_OBJECT_SETTLE {args.settle_after_release_s:.2f}s")
        if args.settle_after_release_s > 0:
            time.sleep(args.settle_after_release_s)
        print("STOP")
        print("MANUAL_CALIBRATION_REQUIRED:")
        print("Please record Nut B world pose after settling.")
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


def run_execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    print("MODE: EXECUTE")
    print_targets(plan)
    print("NOTE: Full execute should be used after right and left calibration both succeed.")

    bundle: RaboDeviceBundle | None = None
    try:
        bundle = RaboDeviceBundle()
        right_result = check_pose(bundle.right_arm, plan["transfer_point"]["right_release_pose"])
        left_result = check_pose(bundle.left_arm, plan["transfer_point"]["left_grasp_pose"])
        print_json("RIGHT transfer pose_check", right_result)
        print_json("LEFT transfer pose_check", left_result)
        if right_result["status"] != "PASS" or left_result["status"] != "PASS":
            print("TRANSFER_CHECK: FAIL")
            print("Execution stopped before motion.")
            return 2

        for index, steps in plan["step_groups"]:
            try:
                execute_group(
                    bundle,
                    index,
                    steps,
                    step_delay_s=args.step_delay_s,
                    settle_after_pose_s=args.settle_after_pose_s,
                    hold_after_grasp_s=args.hold_after_grasp_s,
                )
            except Exception as exc:
                print("FAILED_STEP")
                print(step_header(index))
                print(f"exception: {repr(exc)}")
                print_json("current_state", read_failure_state(bundle))
                return 1
        print("SIMPLE_DUAL_ARM_TRANSFER: COMPLETE")
        return 0
    except Exception as exc:
        print("FAILED_STEP")
        print("INITIALIZATION_OR_PRECHECK")
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
    print_json("Nut B nominal pose", pose_to_list(plan["nut_b_pose"]))
    print_json("Right Nut B grasp pose", pose_to_list(plan["right_grasp_pose"]))
    print_json("TRANSFER_POINT", plan["transfer_point"])
    print_json("Left transfer lift pose", pose_to_list(plan["left_lift_pose"]))
    print_json("Official Place B", pose_to_list(plan["place_b_pose"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP Nut B right-pick table-transfer left-place test.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Initialize arms, pose_check transfer poses, and print targets only.")
    mode.add_argument("--right-calibration", action="store_true", help="Run right-only Nut B pick and table release, then stop for manual Nut B world-pose recording.")
    mode.add_argument("--left-calibration", action="store_true", help="Run left-only table pick from manually configured LEFT_GRASP_POSE to Official Place B.")
    mode.add_argument("--execute", action="store_true", help="Run the full MVP motion sequence after transfer pose_check passes.")
    parser.add_argument("--left-grasp-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"), help="Override manual left_grasp_pose; left_approach_pose is set 0.10m above it.")
    parser.add_argument("--step-delay-s", type=float, default=0.0, help="Sleep after every executed ActionStep.")
    parser.add_argument("--settle-after-pose-s", type=float, default=0.5, help="Sleep after setting Nut B pose.")
    parser.add_argument("--hold-after-grasp-s", type=float, default=0.5, help="Sleep after grasp_force before lifting.")
    parser.add_argument("--settle-after-release-s", type=float, default=3.0, help="Right calibration wait after releasing Nut B at the table transfer point.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.left_grasp_pose is not None:
        left_grasp_pose = pose6_from_list(args.left_grasp_pose)
        TRANSFER_POINT["left_grasp_pose"] = left_grasp_pose
        TRANSFER_POINT["left_approach_pose"] = Pose6(
            left_grasp_pose.x,
            left_grasp_pose.y,
            left_grasp_pose.z + 0.10,
            left_grasp_pose.roll,
            left_grasp_pose.pitch,
            left_grasp_pose.yaw,
        )
    plan = make_plan()
    if args.right_calibration:
        return run_right_calibration(args, plan)
    if args.left_calibration:
        return run_left_calibration(args, plan)
    if args.execute:
        return run_execute(args, plan)
    return run_check_only(plan)


if __name__ == "__main__":
    raise SystemExit(main())
