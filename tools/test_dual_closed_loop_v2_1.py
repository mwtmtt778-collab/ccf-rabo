#!/usr/bin/env python3
"""Dual-arm Nut B vision closed loop V2.1 with explicit safety states.

V2.1 preserves the verified right release, point-cloud detector, coordinate
transform, and LeftNutGraspPlanner template.  It adds a stable right-arm
observation state before vision and an extra vertical left-arm lift before the
box transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "dual_closed_loop_v2_1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    HAND_OPEN,
    LEFT_PLACE_POSES,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import LeftNutGraspPlanner, jsonable  # noqa: E402
from tools.test_dual_closed_loop_v2 import (  # noqa: E402
    RIGHT_RELEASE_POSE,
    make_left_bundle,
    shutdown_left_bundle,
)
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
)


# V2.1 safety configuration.  The observation path intentionally reuses the
# already verified right-arm preparation joints rather than introducing a new
# searched pose.
RIGHT_OBSERVATION_JOINTS = tuple(tuple(float(value) for value in joints) for joints in RIGHT_PRE_JOINTS)
RIGHT_RELEASE_OPEN_WAIT_S = 1.0
RIGHT_OBSERVATION_STABLE_WAIT_S = 2.0
LEFT_SAFE_LIFT_DELTA_Z_M = 0.18


class DualClosedLoopV21Error(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def add_phase(
    report: dict[str, Any],
    phase: str,
    success: bool,
    *,
    failure_reason: str | None = None,
    **details: Any,
) -> None:
    report["phases"].append(
        jsonable(
            {
                "phase": phase,
                "success": bool(success),
                "timestamp": now_text(),
                "failure_reason": failure_reason,
                **details,
            }
        )
    )
    print(f"\n{phase}: {'PASS' if success else 'FAIL'}")
    if failure_reason:
        print(f"failure_reason: {failure_reason}")


def write_report(report: dict[str, Any], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_lift_pose_from_grasp(grasp_pose: list[float]) -> list[float]:
    if len(grasp_pose) != 6:
        raise DualClosedLoopV21Error(f"left grasp pose must contain 6 values, got {len(grasp_pose)}")
    pose = [float(value) for value in grasp_pose]
    pose[2] += LEFT_SAFE_LIFT_DELTA_Z_M
    return pose


def run(args: argparse.Namespace) -> int:
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"dual_closed_loop_v2_1_{started.strftime('%Y%m%d_%H%M%S')}.json"
    report: dict[str, Any] = {
        "experiment": "dual_closed_loop_v2_1",
        "timestamp": started.isoformat(timespec="seconds"),
        "status": "RUNNING",
        "config": {
            "right_grasp_pose": pose_to_list(RIGHT_NUT_B_GRASP_POSE),
            "right_lift_poses": [pose_to_list(pose) for pose in RIGHT_LIFT_POSES],
            "right_release_pose": pose_to_list(RIGHT_RELEASE_POSE),
            "right_observation_joints": [list(joints) for joints in RIGHT_OBSERVATION_JOINTS],
            "right_release_open_wait_s": RIGHT_RELEASE_OPEN_WAIT_S,
            "right_observation_stable_wait_s": RIGHT_OBSERVATION_STABLE_WAIT_S,
            "minimum_release_to_vision_s": float(args.settle_after_release_s),
            "left_safe_lift_delta_z_m": LEFT_SAFE_LIFT_DELTA_Z_M,
            "left_place_pose_b": pose_to_list(LEFT_PLACE_POSES["B"]),
            "vision_roi_center_world": list(TARGET_NUT_B_WORLD_POSE[:3]),
            "vision_roi_radius_m": float(args.vision_target_radius_m),
            "vision_note": (
                "ROI selects the released Nut B candidate only; the left grasp "
                "coordinate is always the measured PointCloud2 world XYZ."
            ),
            "headless": bool(args.headless),
        },
        "right_grasp_success": False,
        "right_release_success": False,
        "right_observation_success": False,
        "vision_detect": False,
        "nut_world_xyz": None,
        "left_grasp_success": False,
        "left_grasp_pose": None,
        "left_safe_lift_success": False,
        "left_safe_lift_pose": None,
        "left_place_success": False,
        "overall_success": False,
        "failed_phase": None,
        "failure_reason": None,
        "phases": [],
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
    }

    right_bundle = None
    left_bundle = None
    current_phase = "INITIALIZE_RIGHT"
    phase_recorded = False
    try:
        print("================================")
        print("Dual closed loop V2.1")
        print("================================")
        print(f"headless: {str(bool(args.headless)).lower()}")

        right_bundle = make_right_bundle()
        add_phase(report, current_phase, True)
        phase_recorded = True

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
        report["right_grasp_success"] = True
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
        release_monotonic = time.monotonic()
        report["right_release_success"] = True
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
        print(f"\nWaiting after hand open: {RIGHT_RELEASE_OPEN_WAIT_S:g}s")
        time.sleep(RIGHT_RELEASE_OPEN_WAIT_S)
        for index, joints in enumerate(RIGHT_OBSERVATION_JOINTS[:-1], start=1):
            execute_checked(
                f"RIGHT_RETREAT_JOINTS_{index}",
                right_bundle.right_arm,
                "move_joints",
                list(joints),
            )
        add_phase(
            report,
            current_phase,
            True,
            transition_joints=[list(joints) for joints in RIGHT_OBSERVATION_JOINTS[:-1]],
            destination="RIGHT_OBSERVATION",
        )
        phase_recorded = True

        current_phase = "RIGHT_OBSERVATION"
        phase_recorded = False
        observation_joints = list(RIGHT_OBSERVATION_JOINTS[-1])
        execute_checked(
            "RIGHT_OBSERVATION_JOINTS",
            right_bundle.right_arm,
            "move_joints",
            observation_joints,
        )
        print(f"\nWaiting at observation pose: {RIGHT_OBSERVATION_STABLE_WAIT_S:g}s")
        time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
        report["right_observation_success"] = True
        add_phase(
            report,
            current_phase,
            True,
            joints=observation_joints,
            stable_wait_s=RIGHT_OBSERVATION_STABLE_WAIT_S,
        )
        phase_recorded = True

        # Preserve V2's minimum release-to-vision settling interval.  The two
        # new waits count toward it instead of blindly adding another 10 s.
        current_phase = "WAIT_SETTLE"
        phase_recorded = False
        elapsed_since_release = time.monotonic() - release_monotonic
        remaining_settle = max(0.0, float(args.settle_after_release_s) - elapsed_since_release)
        print(f"\nRemaining physical settle: {remaining_settle:.2f}s")
        time.sleep(remaining_settle)
        add_phase(
            report,
            current_phase,
            True,
            minimum_release_to_vision_s=float(args.settle_after_release_s),
            remaining_wait_s=remaining_settle,
        )
        phase_recorded = True

        # Destroy right robot clients only after reaching the observation
        # state, before opening the reliable high-bandwidth camera subscriber.
        shutdown_bundle(right_bundle)
        right_bundle = None

        current_phase = "VISION"
        phase_recorded = False
        vision = detect_released_nut(
            tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3]),
            float(args.vision_target_radius_m),
        )
        if not vision.get("detected") or not vision.get("nut_world_xyz"):
            raise DualClosedLoopV21Error(
                f"Nut B point-cloud detection failed: {vision.get('error', 'candidate outside ROI')}"
            )
        nut_world_xyz = [float(value) for value in vision["nut_world_xyz"]]
        report["vision_detect"] = True
        report["nut_world_xyz"] = nut_world_xyz
        add_phase(report, current_phase, True, nut_world_xyz=nut_world_xyz, detector_result=vision)
        phase_recorded = True
        print("Nut world XYZ: " + json.dumps(nut_world_xyz))

        current_phase = "INITIALIZE_LEFT"
        phase_recorded = False
        left_bundle = make_left_bundle()
        planner = LeftNutGraspPlanner(left_arm=left_bundle.left_arm, left_hand=left_bundle.left_hand)
        add_phase(report, current_phase, True)
        phase_recorded = True

        current_phase = "LEFT_GRASP"
        phase_recorded = False
        left_result = planner.grasp_world_xyz(nut_world_xyz, execute=True)
        grasp_pose = left_result.get("grasp_pose")
        report["left_grasp_pose"] = grasp_pose
        if not left_result.get("success"):
            reason = f"{left_result.get('failed_stage')}: {left_result.get('reason')}"
            raise DualClosedLoopV21Error(f"left grasp/lift failed: {reason}")
        report["left_grasp_success"] = True
        add_phase(
            report,
            current_phase,
            True,
            nut_world_xyz=nut_world_xyz,
            left_grasp_pose=grasp_pose,
            planner_result=left_result,
        )
        phase_recorded = True

        current_phase = "LEFT_SAFE_LIFT"
        phase_recorded = False
        safe_lift_pose = safe_lift_pose_from_grasp(grasp_pose)
        report["left_safe_lift_pose"] = safe_lift_pose
        safe_lift_result = planner.move_to_pose("LEFT_SAFE_LIFT", safe_lift_pose)
        if not safe_lift_result.get("move_success"):
            raise DualClosedLoopV21Error(
                f"left safe lift failed: {safe_lift_result.get('reason', 'move_to failed')}"
            )
        report["left_safe_lift_success"] = True
        add_phase(
            report,
            current_phase,
            True,
            lift_delta_z_m=LEFT_SAFE_LIFT_DELTA_Z_M,
            safe_lift_pose=safe_lift_pose,
            move_result=safe_lift_result,
        )
        phase_recorded = True

        current_phase = "LEFT_PLACE"
        phase_recorded = False
        place_pose = pose_to_list(LEFT_PLACE_POSES["B"])
        place_check = planner.pose_check(place_pose)
        if not place_check.get("pass"):
            raise DualClosedLoopV21Error(f"left place pose_check failed: {place_check.get('reason')}")
        move_checked("LEFT_PLACE_B", left_bundle.left_arm, LEFT_PLACE_POSES["B"])
        execute_checked("LEFT_RELEASE_OPEN", left_bundle.left_hand, "clench", *list(HAND_OPEN))
        report["left_place_success"] = True
        add_phase(report, current_phase, True, place_pose=place_pose, pose_check=place_check)
        phase_recorded = True

        report["overall_success"] = True
        report["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        reason = repr(exc)
        if not phase_recorded:
            add_phase(report, current_phase, False, failure_reason=reason)
        report["status"] = "FAILED"
        report["failed_phase"] = current_phase
        report["failure_reason"] = reason
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
        description=(
            "Run V2.1: right release/observation -> point-cloud detection -> "
            "left grasp/safe-lift -> box."
        )
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI or interactive prompts.",
    )
    parser.add_argument(
        "--settle-after-release-s",
        type=float,
        default=DEFAULT_SETTLE_AFTER_RELEASE_S,
        help=(
            "Minimum total time from release to vision; the 1s release and 2s "
            f"observation waits count toward it (default: {DEFAULT_SETTLE_AFTER_RELEASE_S:g}s)."
        ),
    )
    parser.add_argument(
        "--vision-target-radius-m",
        type=float,
        default=DEFAULT_VISION_TARGET_RADIUS_M,
        help=f"Nut B candidate-selection ROI radius (default: {DEFAULT_VISION_TARGET_RADIUS_M:g}m).",
    )
    args = parser.parse_args()
    if args.settle_after_release_s < RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S:
        parser.error(
            "--settle-after-release-s must be at least "
            f"{RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S:g}s"
        )
    if args.vision_target_radius_m <= 0:
        parser.error("--vision-target-radius-m must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
