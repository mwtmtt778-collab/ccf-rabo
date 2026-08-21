#!/usr/bin/env python3
"""Run one vision-closed-loop Nut B transfer with both robot arms.

The right arm uses the already validated fixed grasp/lift/release sequence.  It
then retreats before the existing top-camera point-cloud detector is called.
Only the detector's measured world XYZ is passed to LeftNutGraspPlanner; the
configured release target is used solely to select the relevant detector ROI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "dual_closed_loop_v2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    LEFT_PLACE_POSES,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    Pose6,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import LeftNutGraspPlanner, jsonable  # noqa: E402
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    TARGET_NUT_B_WORLD_POSE,
    detect_released_nut,
    execute_checked,
    make_right_bundle,
    move_checked,
    shutdown_bundle,
    shutdown_device,
)


# Frozen parameters from the successful right-release validation.  These are
# intentionally not optimized or recalculated in this closed-loop experiment.
RIGHT_RELEASE_POSE = Pose6(-0.324660708, -0.056070921, -0.25, 0.0, 0.8, 0.0)


class DualClosedLoopError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_left_bundle() -> Any:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left

    return SimpleNamespace(
        left_arm=LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim"),
        left_hand=LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim"),
    )


def shutdown_left_bundle(bundle: Any | None) -> None:
    if bundle is None:
        return
    shutdown_device(getattr(bundle, "left_hand", None))
    shutdown_device(getattr(bundle, "left_arm", None))


def add_phase(
    report: dict[str, Any],
    phase: str,
    success: bool,
    *,
    failure_reason: str | None = None,
    **details: Any,
) -> None:
    row = {
        "phase": phase,
        "success": bool(success),
        "timestamp": now_text(),
        "failure_reason": failure_reason,
        **details,
    }
    report["phases"].append(jsonable(row))
    print(f"\n{phase}: {'PASS' if success else 'FAIL'}")
    if failure_reason:
        print(f"failure_reason: {failure_reason}")


def write_report(report: dict[str, Any], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"dual_closed_loop_v2_{started.strftime('%Y%m%d_%H%M%S')}.json"
    report: dict[str, Any] = {
        "experiment": "dual_closed_loop_v2",
        "timestamp": started.isoformat(timespec="seconds"),
        "status": "RUNNING",
        "config": {
            "right_pre_joints": [list(joints) for joints in RIGHT_PRE_JOINTS],
            "right_grasp_pose": pose_to_list(RIGHT_NUT_B_GRASP_POSE),
            "right_lift_poses": [pose_to_list(pose) for pose in RIGHT_LIFT_POSES],
            "right_release_pose": pose_to_list(RIGHT_RELEASE_POSE),
            "right_retreat": [list(joints) for joints in RIGHT_PRE_JOINTS],
            "settle_after_release_s": float(args.settle_after_release_s),
            "vision_roi_center_world": list(TARGET_NUT_B_WORLD_POSE[:3]),
            "vision_roi_radius_m": float(args.vision_target_radius_m),
            "vision_note": (
                "ROI center selects the released Nut B candidate only; "
                "the left grasp coordinate always comes from PointCloud2 detection."
            ),
            "left_place_pose_b": pose_to_list(LEFT_PLACE_POSES["B"]),
            "headless": bool(args.headless),
        },
        "phases": [],
        "nut_world_xyz": None,
        "left_grasp_pose": None,
        "failure_reason": None,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
    }

    right_bundle = None
    left_bundle = None
    current_phase = "INITIALIZE_RIGHT"
    phase_recorded = False
    try:
        print("================================")
        print("Dual closed loop V2")
        print("================================")
        print(f"headless: {str(bool(args.headless)).lower()}")

        right_bundle = make_right_bundle()
        add_phase(report, current_phase, True)

        current_phase = "RIGHT_PREPARE"
        phase_recorded = False
        execute_checked("RIGHT_HAND_OPEN_INITIAL", right_bundle.right_hand, "clench", *list(HAND_OPEN))
        for index, joints in enumerate(RIGHT_PRE_JOINTS, start=1):
            execute_checked(f"RIGHT_PRE_{index}", right_bundle.right_arm, "move_joints", list(joints))
        add_phase(report, current_phase, True)
        phase_recorded = True

        current_phase = "RIGHT_GRASP"
        phase_recorded = False
        move_checked("RIGHT_NUT_B_GRASP_POSE", right_bundle.right_arm, RIGHT_NUT_B_GRASP_POSE)
        execute_checked("RIGHT_THUMB_TUCK", right_bundle.right_hand, "clench", thumb_rotation=1.0)
        execute_checked("RIGHT_GRASP_FORCE", right_bundle.right_hand, "grasp_force", **RIGHT_GRASP_FORCE)
        if DEFAULT_HOLD_AFTER_GRASP_S > 0:
            time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
        add_phase(report, current_phase, True, grasp_pose=pose_to_list(RIGHT_NUT_B_GRASP_POSE))
        phase_recorded = True

        current_phase = "RIGHT_LIFT"
        phase_recorded = False
        for index, pose in enumerate(RIGHT_LIFT_POSES, start=1):
            move_checked(f"RIGHT_LIFT_{index}", right_bundle.right_arm, pose)
        add_phase(report, current_phase, True, lift_poses=[pose_to_list(pose) for pose in RIGHT_LIFT_POSES])
        phase_recorded = True

        current_phase = "RIGHT_RELEASE"
        phase_recorded = False
        move_checked("RIGHT_RELEASE_POSE", right_bundle.right_arm, RIGHT_RELEASE_POSE)
        execute_checked("RIGHT_RELEASE_OPEN", right_bundle.right_hand, "clench", *list(HAND_OPEN))
        release_timestamp = now_text()
        add_phase(
            report,
            current_phase,
            True,
            release_pose=pose_to_list(RIGHT_RELEASE_POSE),
            release_timestamp=release_timestamp,
        )
        phase_recorded = True

        current_phase = "RIGHT_RETREAT"
        phase_recorded = False
        for index, joints in enumerate(RIGHT_PRE_JOINTS, start=1):
            execute_checked(f"RIGHT_RETREAT_PRE_{index}", right_bundle.right_arm, "move_joints", list(joints))
        add_phase(report, current_phase, True, retreat_pose=[list(joints) for joints in RIGHT_PRE_JOINTS])
        phase_recorded = True

        # Release DDS robot clients before the high-bandwidth reliable camera
        # subscription, matching the validated release-stability experiment.
        shutdown_bundle(right_bundle)
        right_bundle = None

        current_phase = "WAIT_SETTLE"
        phase_recorded = False
        print(f"\nWaiting: {args.settle_after_release_s:g}s")
        time.sleep(args.settle_after_release_s)
        add_phase(report, current_phase, True, seconds=float(args.settle_after_release_s))
        phase_recorded = True

        current_phase = "VISION_DETECT_NUT_B"
        phase_recorded = False
        vision = detect_released_nut(
            tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3]),
            float(args.vision_target_radius_m),
        )
        if not vision.get("detected") or not vision.get("nut_world_xyz"):
            raise DualClosedLoopError(
                f"Nut B point-cloud detection failed: {vision.get('error', 'candidate outside ROI')}"
            )
        nut_world_xyz = [float(value) for value in vision["nut_world_xyz"]]
        report["nut_world_xyz"] = nut_world_xyz
        add_phase(
            report,
            current_phase,
            True,
            nut_world_xyz=nut_world_xyz,
            detector_result=vision,
        )
        phase_recorded = True
        print("Nut world XYZ: " + json.dumps(nut_world_xyz))

        current_phase = "INITIALIZE_LEFT"
        phase_recorded = False
        left_bundle = make_left_bundle()
        planner = LeftNutGraspPlanner(left_arm=left_bundle.left_arm, left_hand=left_bundle.left_hand)
        add_phase(report, current_phase, True)
        phase_recorded = True

        current_phase = "LEFT_GRASP_AND_LIFT"
        phase_recorded = False
        left_result = planner.grasp_world_xyz(nut_world_xyz, execute=True)
        report["left_grasp_pose"] = left_result.get("grasp_pose")
        if not left_result.get("success"):
            reason = f"{left_result.get('failed_stage')}: {left_result.get('reason')}"
            raise DualClosedLoopError(f"left grasp/lift failed: {reason}")
        add_phase(
            report,
            current_phase,
            True,
            nut_world_xyz=nut_world_xyz,
            left_grasp_pose=left_result.get("grasp_pose"),
            planner_result=left_result,
        )
        phase_recorded = True

        current_phase = "LEFT_PLACE_BOX_B"
        phase_recorded = False
        place_pose = pose_to_list(LEFT_PLACE_POSES["B"])
        place_check = planner.pose_check(place_pose)
        if not place_check.get("pass"):
            raise DualClosedLoopError(f"left place pose_check failed: {place_check.get('reason')}")
        move_checked("LEFT_PLACE_B", left_bundle.left_arm, LEFT_PLACE_POSES["B"])
        execute_checked("LEFT_RELEASE_OPEN", left_bundle.left_hand, "clench", *list(HAND_OPEN))
        add_phase(
            report,
            current_phase,
            True,
            place_pose=place_pose,
            pose_check=place_check,
        )
        phase_recorded = True

        report["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        reason = repr(exc)
        if not phase_recorded:
            add_phase(report, current_phase, False, failure_reason=reason)
        report["status"] = "FAILED"
        report["failure_reason"] = f"{current_phase}: {reason}"
        return_code = 1
    finally:
        shutdown_bundle(right_bundle)
        shutdown_left_bundle(left_bundle)
        report["finished_at"] = now_text()
        write_report(report, report_path)
        print(f"\nResult: {report['status']}")
        print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one right-release -> top-camera -> left-grasp -> box closed loop."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI or interactive prompts (the script never opens visualization windows).",
    )
    parser.add_argument(
        "--settle-after-release-s",
        type=float,
        default=DEFAULT_SETTLE_AFTER_RELEASE_S,
        help=f"Physics settling time after retreat (default: {DEFAULT_SETTLE_AFTER_RELEASE_S:g}s).",
    )
    parser.add_argument(
        "--vision-target-radius-m",
        type=float,
        default=DEFAULT_VISION_TARGET_RADIUS_M,
        help=f"Nut B detector selection ROI radius (default: {DEFAULT_VISION_TARGET_RADIUS_M:g}m).",
    )
    args = parser.parse_args()
    if args.settle_after_release_s < 0:
        parser.error("--settle-after-release-s must be >= 0")
    if args.vision_target_radius_m <= 0:
        parser.error("--vision-target-radius-m must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
