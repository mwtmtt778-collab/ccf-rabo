#!/usr/bin/env python3
"""Single Nut B dual-arm transfer V1.

Flow:
right pick Nut B -> right hover-release at the verified left-grasp position ->
manual confirmation -> left planner grasp/lift -> left place at Official Place B.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    HAND_OPEN,
    LEFT_PLACE_POSES,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import RaboDeviceBundle, pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import (  # noqa: E402
    LeftNutGraspPlanner,
    jsonable,
    normalize_pose_check,
    result_failed,
)
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_BASE_FRAMES  # noqa: E402
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402


TARGET_NUT_B_WORLD_POSE = [-0.2966, 0.0420, 0.2806, 0.0, 0.0, 0.5233]
RIGHT_RELEASE_Z_BASE = -0.20
RIGHT_RELEASE_RPY_BASE = [0.0, 0.8, 0.0]
RIGHT_RETREAT_DZ = 0.10
DEFAULT_SETTLE_AFTER_RELEASE_S = 3.0
DEFAULT_HOLD_AFTER_GRASP_S = 0.7


class TransferExecutionError(RuntimeError):
    pass


def print_stage(stage: str) -> None:
    print("")
    print(stage)
    print("=" * len(stage))


def print_json(label: str, payload: Any) -> None:
    print(f"{label}: {json.dumps(jsonable(payload), ensure_ascii=False)}")


def pose_from_list(values: list[float] | tuple[float, ...]) -> Pose6:
    if len(values) != 6:
        raise ValueError(f"pose must have 6 values, got {len(values)}")
    return Pose6(*[float(v) for v in values])


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return asdict(pose)


def pose_with_z_delta(pose: Pose6, dz: float) -> Pose6:
    return Pose6(pose.x, pose.y, pose.z + dz, pose.roll, pose.pitch, pose.yaw)


def safe_shutdown(bundle: Any | None) -> None:
    if bundle is None:
        return
    if hasattr(bundle, "shutdown"):
        bundle.shutdown()
        return
    for name in ("left_hand", "right_hand", "left_arm", "right_arm"):
        device = getattr(bundle, name, None)
        if hasattr(device, "shutdown"):
            device.shutdown()


def execute_checked(label: str, target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name)
    result = method(*args, **kwargs)
    failed, reason = result_failed(result)
    print_json(
        label,
        {
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
            "return_value": result,
            "status": "FAILED" if failed else "OK",
            "reason": reason,
        },
    )
    if failed:
        raise TransferExecutionError(f"{label}: {method_name} failed: {reason}")
    return result


def move_checked(label: str, arm: Any, pose: Pose6) -> Any:
    setattr(arm, "current_phase", label)
    print_json(f"{label}_TARGET", pose_to_list(pose))
    return execute_checked(label, arm, "move_to", **pose_kwargs(pose))


def pose_check_checked(label: str, arm: Any, pose: Pose6) -> dict[str, Any]:
    raw = arm.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)
    status, reason, value = normalize_pose_check(raw)
    result = {"status": status, "reason": reason, "raw": value, "pose": pose_to_list(pose)}
    print_json(label, result)
    if status != "PASS":
        raise TransferExecutionError(f"{label}: pose_check did not PASS: {reason}")
    return result


def build_right_release_pose() -> dict[str, Any]:
    right_base_world = CONFIRMED_BASE_FRAMES["right_arm_base"]["world_pose"]
    initial_nut_world = pose_to_list(NUT_SPECS["B"].nominal_pose)
    right_grasp_world = transform_pose_base_to_world(pose_to_list(RIGHT_NUT_B_GRASP_POSE), right_base_world)
    hand_to_nut_offset_world = [
        right_grasp_world[index] - initial_nut_world[index]
        for index in range(3)
    ]

    release_hand_world = [
        TARGET_NUT_B_WORLD_POSE[0] + hand_to_nut_offset_world[0],
        TARGET_NUT_B_WORLD_POSE[1] + hand_to_nut_offset_world[1],
        right_base_world[2] + RIGHT_RELEASE_Z_BASE,
        *right_grasp_world[3:],
    ]
    release_base = transform_pose_world_to_base(release_hand_world, right_base_world)
    release_pose = Pose6(
        release_base[0],
        release_base[1],
        RIGHT_RELEASE_Z_BASE,
        RIGHT_RELEASE_RPY_BASE[0],
        RIGHT_RELEASE_RPY_BASE[1],
        RIGHT_RELEASE_RPY_BASE[2],
    )
    return {
        "right_base_world": right_base_world,
        "initial_nut_world": initial_nut_world,
        "target_nut_world": TARGET_NUT_B_WORLD_POSE,
        "right_grasp_pose_base": pose_to_list(RIGHT_NUT_B_GRASP_POSE),
        "right_grasp_pose_world": right_grasp_world,
        "hand_to_nut_offset_world": hand_to_nut_offset_world,
        "right_release_pose_base": pose_to_list(release_pose),
        "right_retreat_pose_base": pose_to_list(pose_with_z_delta(release_pose, RIGHT_RETREAT_DZ)),
        "release_policy": "SCHEME_B_HOVER_RELEASE_NO_TABLE_PRESS",
    }


def preflight(bundle: Any, release_pose: Pose6, retreat_pose: Pose6) -> None:
    print_stage("START")
    print_json("WORLD_ID", WORLD_ID)
    print_json("TARGET_NUT_B_WORLD_POSE", TARGET_NUT_B_WORLD_POSE)
    print_json("OFFICIAL_PLACE_B", pose_to_list(LEFT_PLACE_POSES["B"]))
    pose_check_checked("RIGHT_GRASP_POSE_CHECK", bundle.right_arm, RIGHT_NUT_B_GRASP_POSE)
    for index, pose in enumerate(RIGHT_LIFT_POSES, start=1):
        pose_check_checked(f"RIGHT_LIFT_{index}_POSE_CHECK", bundle.right_arm, pose)
    pose_check_checked("RIGHT_TRANSFER_POSE_CHECK", bundle.right_arm, release_pose)
    pose_check_checked("RIGHT_RETREAT_POSE_CHECK", bundle.right_arm, retreat_pose)

    planner = LeftNutGraspPlanner(left_arm=bundle.left_arm, left_hand=bundle.left_hand)
    left_plan = planner.build_plan(TARGET_NUT_B_WORLD_POSE[:3], yaw_offset=0.0)
    left_preflight = planner.preflight_grasp_chain(left_plan)
    print_json("LEFT_GRASP_PREFLIGHT", left_preflight)
    if not left_preflight["success"]:
        raise TransferExecutionError(f"LEFT_GRASP preflight failed: {left_preflight['failed_stage']} {left_preflight['reason']}")
    pose_check_checked("LEFT_PLACE_B_POSE_CHECK", bundle.left_arm, LEFT_PLACE_POSES["B"])


def run_right_flow(bundle: Any, release_pose: Pose6, retreat_pose: Pose6, *, hold_after_grasp_s: float) -> None:
    print_stage("RIGHT_GRASP")
    for index, joints in enumerate(RIGHT_PRE_JOINTS, start=1):
        execute_checked(f"RIGHT_PRE_{index}", bundle.right_arm, "move_joints", joints)
    move_checked("RIGHT_APPROACH_NUT_B", bundle.right_arm, RIGHT_NUT_B_GRASP_POSE)
    execute_checked("RIGHT_THUMB_TUCK", bundle.right_hand, "clench", thumb_rotation=1.0)
    execute_checked("RIGHT_GRASP_FORCE", bundle.right_hand, "grasp_force", **RIGHT_GRASP_FORCE)
    if hold_after_grasp_s > 0:
        time.sleep(hold_after_grasp_s)

    print_stage("RIGHT_TRANSFER")
    for index, pose in enumerate(RIGHT_LIFT_POSES, start=1):
        move_checked(f"RIGHT_LIFT_{index}", bundle.right_arm, pose)
    move_checked("RIGHT_TRANSFER_RELEASE_POSE", bundle.right_arm, release_pose)

    print_stage("RIGHT_RELEASE")
    execute_checked("RIGHT_RELEASE_OPEN", bundle.right_hand, "clench", *list(HAND_OPEN))
    move_checked("RIGHT_RETREAT", bundle.right_arm, retreat_pose)


def run_left_flow(bundle: Any) -> None:
    print_stage("LEFT_GRASP")
    planner = LeftNutGraspPlanner(left_arm=bundle.left_arm, left_hand=bundle.left_hand)
    plan = planner.build_plan(TARGET_NUT_B_WORLD_POSE[:3], yaw_offset=0.0)
    preflight = planner.preflight_grasp_chain(plan)
    print_json("LEFT_GRASP_PREFLIGHT", preflight)
    if not preflight["success"]:
        raise TransferExecutionError(f"LEFT_GRASP preflight failed: {preflight['failed_stage']} {preflight['reason']}")

    open_result = planner.open_hand()
    print_json("LEFT_OPEN_HAND", open_result)
    if not open_result["success"]:
        raise TransferExecutionError(f"LEFT_OPEN_HAND failed: {open_result['reason']}")

    safe_rows = planner.execute_safe_pre_joints()
    print_json("LEFT_SAFE_PRE", safe_rows)
    failed_safe = next((row for row in safe_rows if not row["success"]), None)
    if failed_safe is not None:
        raise TransferExecutionError(f"LEFT_SAFE_PRE failed: {failed_safe['stage']} {failed_safe['reason']}")

    for item in plan["path"]:
        if item["stage"].startswith("LIFT_"):
            continue
        row = planner.move_to_pose(item["stage"], item["pose"])
        print_json(f"LEFT_{item['stage']}", row)
        if not row["move_success"]:
            raise TransferExecutionError(f"LEFT_GRASP move failed: {item['stage']} {row['reason']}")

    grasp_action = planner.execute_grasp_action()
    print_json("LEFT_GRASP_ACTION", grasp_action)
    if not grasp_action["success"]:
        raise TransferExecutionError(f"LEFT_GRASP_ACTION failed: {grasp_action}")

    print_stage("LEFT_LIFT")
    for item in plan["lift_waypoints"]:
        row = planner.move_to_pose(item["stage"], item["pose"])
        print_json(f"LEFT_{item['stage']}", row)
        if not row["move_success"]:
            raise TransferExecutionError(f"LEFT_LIFT failed: {item['stage']} {row['reason']}")

    print_stage("LEFT_PLACE_B")
    move_checked("LEFT_PLACE_B", bundle.left_arm, LEFT_PLACE_POSES["B"])
    execute_checked("LEFT_RELEASE_OPEN", bundle.left_hand, "clench", *list(HAND_OPEN))


def run(args: argparse.Namespace) -> int:
    derived = build_right_release_pose()
    release_pose = pose_from_list(derived["right_release_pose_base"])
    retreat_pose = pose_from_list(derived["right_retreat_pose_base"])
    print_json("DERIVED_PLAN", derived)

    if args.plan_only:
        return 0

    bundle = None
    try:
        bundle = RaboDeviceBundle()
        preflight(bundle, release_pose, retreat_pose)
        run_right_flow(bundle, release_pose, retreat_pose, hold_after_grasp_s=args.hold_after_grasp_s)

        print_stage("WAIT_CONFIRM")
        if args.settle_after_release_s > 0:
            print(f"Waiting {args.settle_after_release_s:.1f}s for Nut stability.")
            time.sleep(args.settle_after_release_s)
        input("Right hand placement completed.\nCheck Nut position.\nPress ENTER to start left grasp.")

        run_left_flow(bundle)
        print_stage("DONE")
        return 0
    except KeyboardInterrupt:
        print("")
        print("ABORTED_BY_USER")
        return 130
    except Exception as exc:
        print("")
        print("FAILED")
        print(repr(exc))
        return 1
    finally:
        safe_shutdown(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single Nut B right-to-left transfer V1.")
    parser.add_argument("--plan-only", action="store_true", help="Print derived poses only; do not initialize or move robot devices.")
    parser.add_argument("--settle-after-release-s", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S)
    parser.add_argument("--hold-after-grasp-s", type=float, default=DEFAULT_HOLD_AFTER_GRASP_S)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
