#!/usr/bin/env python3
"""Sequential B -> A -> C three-Nut vision closed-loop experiment.

This runner composes the already validated single-Nut components.  It does not
change perception, calibration, release behavior, hand actions, or the left
grasp template.  A failed phase stops the current sequence and is written to a
JSON report before robot clients are shut down.
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
REPORT_DIR = PROJECT_ROOT / "reports" / "three_nut_closed_loop"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    HAND_OPEN,
    LEFT_PLACE_POSES,
    NUT_IDS,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import compute_right_grasp_pose, pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import LeftNutGraspPlanner, jsonable  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import (  # noqa: E402
    LEFT_SAFE_LIFT_DELTA_Z_M,
    RIGHT_OBSERVATION_JOINTS,
    RIGHT_OBSERVATION_STABLE_WAIT_S,
    RIGHT_RELEASE_OPEN_WAIT_S,
    safe_lift_pose_from_grasp,
)
from tools.test_dual_closed_loop_v2_2 import (  # noqa: E402
    RIGHT_RELEASE_POSE,
    RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
    build_right_release_safe_height_pose,
)
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402
from tools.test_nut_camera_calibration import set_pose_with_retry  # noqa: E402
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


NUT_SEQUENCE = ("B", "A", "C")
RELEASE_TARGET_WORLD_XYZ = tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3])

# B remains the frozen successful pose.  A/C come directly from the existing
# three_nut_expert transform and nominal Nut specs; no new grasp search occurs.
RIGHT_GRASP_POSES: dict[str, Pose6] = {
    "B": RIGHT_NUT_B_GRASP_POSE,
    "A": compute_right_grasp_pose(NUT_SPECS["A"].nominal_pose),
    "C": compute_right_grasp_pose(NUT_SPECS["C"].nominal_pose),
}
RIGHT_GRASP_POSE_SOURCES = {
    "B": "tools.test_left_grasp_v1.RIGHT_NUT_B_GRASP_POSE (VERIFIED_FROZEN)",
    "A": "agents.three_nut_expert.expert.compute_right_grasp_pose + NUT_SPECS[A] (EXISTING_STAGED)",
    "C": "agents.three_nut_expert.expert.compute_right_grasp_pose + NUT_SPECS[C] (EXISTING_STAGED)",
}
PLACE_POSE_SOURCES = {
    key: f"agents.three_nut_expert.config.LEFT_PLACE_POSES[{key}]"
    for key in NUT_SEQUENCE
}


class ThreeNutClosedLoopError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_static_contract() -> None:
    if NUT_SEQUENCE != ("B", "A", "C"):
        raise ThreeNutClosedLoopError("Nut sequence must remain B,A,C")
    missing_ids = [key for key in NUT_SEQUENCE if not NUT_IDS.get(key)]
    missing_grasps = [key for key in NUT_SEQUENCE if key not in RIGHT_GRASP_POSES]
    missing_places = [key for key in NUT_SEQUENCE if key not in LEFT_PLACE_POSES]
    if missing_ids:
        raise ThreeNutClosedLoopError(f"missing real entity IDs for: {missing_ids}")
    if missing_grasps:
        raise ThreeNutClosedLoopError(f"missing existing right grasp poses for: {missing_grasps}")
    if missing_places:
        raise ThreeNutClosedLoopError(f"missing existing place poses for: {missing_places}")
    if pose_to_list(RIGHT_GRASP_POSES["B"]) != pose_to_list(RIGHT_NUT_B_GRASP_POSE):
        raise ThreeNutClosedLoopError("Nut B frozen grasp pose was changed")


def make_pose_setter() -> Any:
    from rabo_dev_kit import SetEntityPose

    return SetEntityPose(world=WORLD_ID)


def shutdown_pose_setter(pose_setter: Any | None) -> None:
    if pose_setter is not None and hasattr(pose_setter, "shutdown"):
        try:
            pose_setter.shutdown()
        except Exception as exc:
            print(f"pose setter shutdown warning: {exc!r}")


def new_nut_result(key: str) -> dict[str, Any]:
    return {
        "id": key,
        "entity_id": NUT_IDS[key],
        "reset_success": False,
        "vision_position": None,
        "detected_position": None,
        "right_observation_before_grasp": False,
        "right_grasp": False,
        "lift_success": False,
        "right_release": False,
        "right_safe_retreat": False,
        "right_observation_after_release": False,
        "released_nut_position": None,
        "vision_after_release": False,
        "left_grasp": False,
        "left_safe_lift": False,
        "place": False,
        "success": False,
        "failed_phase": None,
        "failure_code": None,
        "failure_reason": None,
        "phases": [],
    }


def add_phase(
    result: dict[str, Any],
    phase: str,
    success: bool,
    *,
    failure_reason: str | None = None,
    **details: Any,
) -> None:
    result["phases"].append(
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
    print(f"\n[{result['id']}] {phase}: {'PASS' if success else 'FAIL'}")
    if failure_reason:
        print(f"failure_reason: {failure_reason}")


def write_report(report: dict[str, Any], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def detect_nut(target_xyz: tuple[float, float, float], radius: float) -> dict[str, Any]:
    """Use the existing detector; target XYZ is candidate selection only."""
    return detect_released_nut(target_xyz, radius)


def move_right_to_observation(right_bundle: Any, label: str) -> list[list[float]]:
    executed: list[list[float]] = []
    for index, joints in enumerate(RIGHT_OBSERVATION_JOINTS, start=1):
        values = list(joints)
        execute_checked(f"{label}_{index}", right_bundle.right_arm, "move_joints", values)
        executed.append(values)
    return executed


def execute_one_nut(
    key: str,
    pose_setter: Any,
    *,
    settle_after_release_s: float,
    vision_radius_m: float,
) -> dict[str, Any]:
    result = new_nut_result(key)
    right_bundle = None
    left_bundle = None
    current_phase = "RESET"
    phase_recorded = False
    try:
        print("\n================================")
        print(f"Nut {key}")
        print("================================")

        reset_pose = pose_to_list(NUT_SPECS[key].nominal_pose)
        reset = set_pose_with_retry(pose_setter, NUT_IDS[key], reset_pose)
        if not reset.get("ok"):
            raise ThreeNutClosedLoopError(f"SetEntityPose fallback failed: {reset.get('error')}")
        result["reset_success"] = True
        add_phase(
            result,
            current_phase,
            True,
            mode="SET_ENTITY_POSE_FALLBACK",
            entity_id=NUT_IDS[key],
            nominal_pose=reset_pose,
            reset_result=reset,
        )
        phase_recorded = True

        current_phase = "RIGHT_OBSERVATION_BEFORE_GRASP"
        phase_recorded = False
        right_bundle = make_right_bundle()
        execute_checked("RIGHT_HAND_OPEN_INITIAL", right_bundle.right_hand, "clench", *list(HAND_OPEN))
        observation_path = move_right_to_observation(right_bundle, "RIGHT_OBSERVATION_BEFORE_GRASP")
        time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
        result["right_observation_before_grasp"] = True
        add_phase(result, current_phase, True, joints=observation_path)
        phase_recorded = True
        shutdown_bundle(right_bundle)
        right_bundle = None

        current_phase = "VISION_BEFORE_GRASP"
        phase_recorded = False
        nominal_xyz = tuple(float(value) for value in reset_pose[:3])
        initial_vision = detect_nut(nominal_xyz, vision_radius_m)
        if not initial_vision.get("detected") or not initial_vision.get("nut_world_xyz"):
            raise ThreeNutClosedLoopError(
                f"initial point-cloud detection failed: {initial_vision.get('error', 'candidate outside ROI')}"
            )
        detected_position = [float(value) for value in initial_vision["nut_world_xyz"]]
        result["vision_position"] = detected_position
        result["detected_position"] = detected_position
        add_phase(
            result,
            current_phase,
            True,
            detected_position=detected_position,
            detector_result=initial_vision,
        )
        phase_recorded = True

        current_phase = "RIGHT_GRASP"
        phase_recorded = False
        right_bundle = make_right_bundle()
        grasp_pose = RIGHT_GRASP_POSES[key]
        move_checked(f"RIGHT_NUT_{key}_GRASP_POSE", right_bundle.right_arm, grasp_pose)
        execute_checked("RIGHT_THUMB_TUCK", right_bundle.right_hand, "clench", thumb_rotation=1.0)
        execute_checked("RIGHT_GRASP_FORCE", right_bundle.right_hand, "grasp_force", **RIGHT_GRASP_FORCE)
        if DEFAULT_HOLD_AFTER_GRASP_S > 0:
            time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
        result["right_grasp"] = True
        add_phase(
            result,
            current_phase,
            True,
            grasp_pose=pose_to_list(grasp_pose),
            grasp_pose_source=RIGHT_GRASP_POSE_SOURCES[key],
        )
        phase_recorded = True

        current_phase = "RIGHT_LIFT"
        phase_recorded = False
        for index, lift_pose in enumerate(RIGHT_LIFT_POSES, start=1):
            move_checked(f"RIGHT_LIFT_{index}", right_bundle.right_arm, lift_pose)
        result["lift_success"] = True
        add_phase(result, current_phase, True, lift_poses=[pose_to_list(pose) for pose in RIGHT_LIFT_POSES])
        phase_recorded = True

        current_phase = "RIGHT_RELEASE"
        phase_recorded = False
        move_checked("RIGHT_RELEASE_POSE", right_bundle.right_arm, RIGHT_RELEASE_POSE)
        execute_checked("RIGHT_RELEASE_OPEN", right_bundle.right_hand, "clench", *list(HAND_OPEN))
        release_timestamp = now_text()
        release_monotonic = time.monotonic()
        result["right_release"] = True
        add_phase(
            result,
            current_phase,
            True,
            release_pose=pose_to_list(RIGHT_RELEASE_POSE),
            release_target_world_xyz=list(RELEASE_TARGET_WORLD_XYZ),
            release_timestamp=release_timestamp,
        )
        phase_recorded = True

        current_phase = "RIGHT_SAFE_RETREAT"
        phase_recorded = False
        time.sleep(RIGHT_RELEASE_OPEN_WAIT_S)
        safe_height_pose = build_right_release_safe_height_pose()
        move_checked("RIGHT_RELEASE_SAFE_HEIGHT", right_bundle.right_arm, safe_height_pose)
        result["right_safe_retreat"] = True
        add_phase(
            result,
            current_phase,
            True,
            offset_z_m=RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
            safe_height_pose=pose_to_list(safe_height_pose),
        )
        phase_recorded = True

        current_phase = "RIGHT_OBSERVATION_AFTER_RELEASE"
        phase_recorded = False
        observation_path = move_right_to_observation(right_bundle, "RIGHT_OBSERVATION_AFTER_RELEASE")
        time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
        result["right_observation_after_release"] = True
        add_phase(result, current_phase, True, joints=observation_path)
        phase_recorded = True

        current_phase = "WAIT_SETTLE"
        phase_recorded = False
        elapsed = time.monotonic() - release_monotonic
        remaining = max(0.0, settle_after_release_s - elapsed)
        time.sleep(remaining)
        add_phase(
            result,
            current_phase,
            True,
            minimum_release_to_vision_s=settle_after_release_s,
            remaining_wait_s=remaining,
        )
        phase_recorded = True
        shutdown_bundle(right_bundle)
        right_bundle = None

        current_phase = "VISION_AFTER_RELEASE"
        phase_recorded = False
        released_vision = detect_nut(RELEASE_TARGET_WORLD_XYZ, vision_radius_m)
        if not released_vision.get("detected") or not released_vision.get("nut_world_xyz"):
            raise ThreeNutClosedLoopError(
                "VISION_AFTER_RELEASE_FAILED: "
                f"{released_vision.get('error', 'candidate outside ROI')}"
            )
        released_position = [float(value) for value in released_vision["nut_world_xyz"]]
        result["vision_after_release"] = True
        result["released_nut_position"] = released_position
        add_phase(
            result,
            current_phase,
            True,
            released_nut_position=released_position,
            detector_result=released_vision,
        )
        phase_recorded = True

        current_phase = "LEFT_GRASP"
        phase_recorded = False
        left_bundle = make_left_bundle()
        planner = LeftNutGraspPlanner(left_arm=left_bundle.left_arm, left_hand=left_bundle.left_hand)
        left_result = planner.grasp_world_xyz(released_position, execute=True)
        grasp_pose_left = left_result.get("grasp_pose")
        if not left_result.get("success"):
            raise ThreeNutClosedLoopError(
                f"left grasp failed at {left_result.get('failed_stage')}: {left_result.get('reason')}"
            )
        result["left_grasp"] = True
        add_phase(
            result,
            current_phase,
            True,
            nut_world_xyz=released_position,
            left_grasp_pose=grasp_pose_left,
            planner_result=left_result,
        )
        phase_recorded = True

        current_phase = "LEFT_SAFE_LIFT"
        phase_recorded = False
        safe_lift_pose = safe_lift_pose_from_grasp(grasp_pose_left)
        safe_lift_result = planner.move_to_pose("LEFT_SAFE_LIFT", safe_lift_pose)
        if not safe_lift_result.get("move_success"):
            raise ThreeNutClosedLoopError(
                f"left safe lift failed: {safe_lift_result.get('reason', 'move_to failed')}"
            )
        result["left_safe_lift"] = True
        add_phase(
            result,
            current_phase,
            True,
            lift_delta_z_m=LEFT_SAFE_LIFT_DELTA_Z_M,
            safe_lift_pose=safe_lift_pose,
            move_result=safe_lift_result,
        )
        phase_recorded = True

        current_phase = "LEFT_PLACE"
        phase_recorded = False
        place_pose = LEFT_PLACE_POSES[key]
        place_pose_values = pose_to_list(place_pose)
        place_check = planner.pose_check(place_pose_values)
        if not place_check.get("pass"):
            raise ThreeNutClosedLoopError(f"place pose_check failed: {place_check.get('reason')}")
        move_checked(f"LEFT_PLACE_{key}", left_bundle.left_arm, place_pose)
        execute_checked("LEFT_RELEASE_OPEN", left_bundle.left_hand, "clench", *list(HAND_OPEN))
        result["place"] = True
        add_phase(
            result,
            current_phase,
            True,
            place_pose=place_pose_values,
            place_pose_source=PLACE_POSE_SOURCES[key],
            pose_check=place_check,
        )
        phase_recorded = True

        # The right arm already remains at observation while the left arm
        # completes grasp/place; record that invariant at task completion.
        current_phase = "RIGHT_FINAL_OBSERVATION"
        phase_recorded = False
        add_phase(
            result,
            current_phase,
            True,
            note="Right arm remained at RIGHT_OBSERVATION_JOINTS during left grasp/place.",
        )
        phase_recorded = True
        result["success"] = True
    except Exception as exc:
        reason = repr(exc)
        if not phase_recorded:
            add_phase(result, current_phase, False, failure_reason=reason)
        result["failed_phase"] = current_phase
        if current_phase == "VISION_AFTER_RELEASE":
            result["failure_code"] = "VISION_AFTER_RELEASE_FAILED"
        result["failure_reason"] = reason
    finally:
        shutdown_bundle(right_bundle)
        shutdown_left_bundle(left_bundle)
    return result


def run_trial(trial_id: int, pose_setter: Any, args: argparse.Namespace) -> dict[str, Any]:
    trial = {
        "trial_id": trial_id,
        "sequence": list(NUT_SEQUENCE),
        "nut_results": [],
        "success": False,
        "failed_nut": None,
        "failed_phase": None,
        "failure_code": None,
    }
    for key in NUT_SEQUENCE:
        result = execute_one_nut(
            key,
            pose_setter,
            settle_after_release_s=float(args.settle_after_release_s),
            vision_radius_m=float(args.vision_target_radius_m),
        )
        trial["nut_results"].append(result)
        if not result["success"]:
            trial["failed_nut"] = key
            trial["failed_phase"] = result["failed_phase"]
            trial["failure_code"] = result["failure_code"]
            return trial
    trial["success"] = True
    return trial


def run(args: argparse.Namespace) -> int:
    validate_static_contract()
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"three_nut_trial_{started.strftime('%Y%m%d_%H%M%S')}.json"
    report: dict[str, Any] = {
        "experiment": "three_nut_closed_loop_v1",
        "timestamp": started.isoformat(timespec="seconds"),
        "status": "RUNNING",
        "sequence": list(NUT_SEQUENCE),
        "config": {
            "nut_ids": {key: NUT_IDS[key] for key in NUT_SEQUENCE},
            "nut_nominal_poses": {key: pose_to_list(NUT_SPECS[key].nominal_pose) for key in NUT_SEQUENCE},
            "right_grasp_poses": {key: pose_to_list(RIGHT_GRASP_POSES[key]) for key in NUT_SEQUENCE},
            "right_grasp_pose_sources": RIGHT_GRASP_POSE_SOURCES,
            "right_release_pose": pose_to_list(RIGHT_RELEASE_POSE),
            "release_target_world_xyz": list(RELEASE_TARGET_WORLD_XYZ),
            "right_release_safe_height_offset_z_m": RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
            "right_observation_joints": [list(joints) for joints in RIGHT_OBSERVATION_JOINTS],
            "left_safe_lift_delta_z_m": LEFT_SAFE_LIFT_DELTA_Z_M,
            "left_place_poses": {key: pose_to_list(LEFT_PLACE_POSES[key]) for key in NUT_SEQUENCE},
            "place_pose_sources": PLACE_POSE_SOURCES,
            "settle_after_release_s": float(args.settle_after_release_s),
            "vision_target_radius_m": float(args.vision_target_radius_m),
            "headless": True,
            "configuration_warning": "A/C right grasps and place slots are existing staged values, not field-verified.",
        },
        "trials": [],
        "overall_success": False,
        "failed_trial": None,
        "failed_nut": None,
        "failed_phase": None,
        "failure_code": None,
        "failure_reason": None,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
    }

    pose_setter = None
    return_code = 1
    try:
        print("================================")
        print("Three Nut closed loop V1")
        print("Sequence: B -> A -> C")
        print("================================")
        pose_setter = make_pose_setter()
        for trial_id in range(1, args.trials + 1):
            print(f"\n######## Trial {trial_id}/{args.trials} ########")
            trial = run_trial(trial_id, pose_setter, args)
            report["trials"].append(trial)
            write_report(report, report_path)
            if not trial["success"]:
                report["failed_trial"] = trial_id
                report["failed_nut"] = trial["failed_nut"]
                report["failed_phase"] = trial["failed_phase"]
                report["failure_code"] = trial["failure_code"]
                failed_result = trial["nut_results"][-1]
                report["failure_reason"] = failed_result["failure_reason"]
                break
        report["overall_success"] = len(report["trials"]) == args.trials and all(
            trial["success"] for trial in report["trials"]
        )
        report["status"] = "PASS" if report["overall_success"] else "FAILED"
        return_code = 0 if report["overall_success"] else 1
    except Exception as exc:
        report["status"] = "FAILED"
        report["failed_phase"] = report["failed_phase"] or "INITIALIZE"
        report["failure_reason"] = repr(exc)
    finally:
        shutdown_pose_setter(pose_setter)
        report["finished_at"] = now_text()
        write_report(report, report_path)
        print(f"\nResult: {report['status']}")
        print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed B -> A -> C three-Nut visual closed-loop experiment."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run headless (default and only supported mode).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of complete B,A,C trials; stops after the first failed trial (default: 1).",
    )
    parser.add_argument(
        "--settle-after-release-s",
        type=float,
        default=DEFAULT_SETTLE_AFTER_RELEASE_S,
        help=f"Minimum release-to-vision settling time (default: {DEFAULT_SETTLE_AFTER_RELEASE_S:g}s).",
    )
    parser.add_argument(
        "--vision-target-radius-m",
        type=float,
        default=DEFAULT_VISION_TARGET_RADIUS_M,
        help=f"Existing detector candidate-selection radius (default: {DEFAULT_VISION_TARGET_RADIUS_M:g}m).",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    minimum_wait = RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S
    if args.settle_after_release_s < minimum_wait:
        parser.error(f"--settle-after-release-s must be at least {minimum_wait:g}s")
    if args.vision_target_radius_m <= 0:
        parser.error("--vision-target-radius-m must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
