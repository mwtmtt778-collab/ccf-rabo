#!/usr/bin/env python3
"""Vision-feedback right-hand Nut B release stability experiment.

Every trial resets Nut B, executes the verified right-hand grasp/lift/release
sequence, immediately retreats through RIGHT_PRE_JOINTS, waits for physics to
settle, then calls the existing top-camera geometry detector.  No GUI is used.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "right_release_stability"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    NUT_IDS,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import (  # noqa: E402
    LeftNutGraspPlanner,
    jsonable,
    normalize_pose_check,
    result_failed,
)
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_BASE_FRAMES  # noqa: E402
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402
from tools.test_nut_camera_calibration import set_pose_with_retry  # noqa: E402
from tools import test_pointcloud_nut_detection as nut_detector  # noqa: E402


TARGET_NUT_B_WORLD_POSE = [-0.2966, 0.0420, 0.2806, 0.0, 0.0, 0.5233]
RIGHT_RELEASE_Z_BASE = -0.25
RIGHT_RELEASE_RPY_BASE = [0.0, 0.8, 0.0]
DEFAULT_SETTLE_AFTER_RELEASE_S = 10.0
DEFAULT_HOLD_AFTER_GRASP_S = 0.7
DEFAULT_VISION_TARGET_RADIUS_M = 0.08
RESET_SETTLE_S = 2.0


class ReleaseStabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseConfig:
    """One parameterized right-hand release strategy (right-arm base frame)."""

    release_xyz: tuple[float, float, float]
    release_rpy: tuple[float, float, float]
    retreat_pose: tuple[tuple[float, ...], ...]
    settle_time: float
    target_nut_world_xyz: tuple[float, float, float]
    hold_after_grasp_s: float = DEFAULT_HOLD_AFTER_GRASP_S

    @property
    def release_pose(self) -> Pose6:
        return Pose6(*self.release_xyz, *self.release_rpy)


def jsonable_pretty(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2)


def print_json(label: str, payload: Any) -> None:
    print(f"{label}: {json.dumps(jsonable(payload), ensure_ascii=False)}")


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return asdict(pose)


def execute_checked(label: str, target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    result = getattr(target, method_name)(*args, **kwargs)
    failed, reason = result_failed(result)
    print_json(label, {"method": method_name, "return_value": result, "status": "FAILED" if failed else "OK"})
    if failed:
        raise ReleaseStabilityError(f"{label}: {method_name} failed: {reason}")
    return result


def move_checked(label: str, arm: Any, pose: Pose6) -> Any:
    setattr(arm, "current_phase", label)
    print_json(f"{label}_TARGET", pose_to_list(pose))
    return execute_checked(label, arm, "move_to", **pose_kwargs(pose))


def make_right_bundle() -> Any:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Right

    return SimpleNamespace(
        right_arm=LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim"),
        right_hand=LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim"),
    )


def shutdown_device(device: Any | None) -> None:
    if device is not None and hasattr(device, "shutdown"):
        try:
            device.shutdown()
        except Exception as exc:
            print(f"shutdown warning: {exc!r}")


def shutdown_bundle(bundle: Any | None) -> None:
    if bundle is None:
        return
    shutdown_device(getattr(bundle, "right_hand", None))
    shutdown_device(getattr(bundle, "right_arm", None))


def build_default_release_pose() -> Pose6:
    right_base_world = CONFIRMED_BASE_FRAMES["right_arm_base"]["world_pose"]
    initial_nut_world = pose_to_list(NUT_SPECS["B"].nominal_pose)
    right_grasp_world = transform_pose_base_to_world(pose_to_list(RIGHT_NUT_B_GRASP_POSE), right_base_world)
    hand_to_nut_offset_world = [right_grasp_world[index] - initial_nut_world[index] for index in range(3)]
    release_hand_world = [
        TARGET_NUT_B_WORLD_POSE[0] + hand_to_nut_offset_world[0],
        TARGET_NUT_B_WORLD_POSE[1] + hand_to_nut_offset_world[1],
        right_base_world[2] + RIGHT_RELEASE_Z_BASE,
        *right_grasp_world[3:],
    ]
    release_base = transform_pose_world_to_base(release_hand_world, right_base_world)
    return Pose6(
        release_base[0],
        release_base[1],
        RIGHT_RELEASE_Z_BASE,
        RIGHT_RELEASE_RPY_BASE[0],
        RIGHT_RELEASE_RPY_BASE[1],
        RIGHT_RELEASE_RPY_BASE[2],
    )


def build_release_config(args: argparse.Namespace) -> ReleaseConfig:
    default_pose = build_default_release_pose()
    release_xyz = tuple(args.release_xyz) if args.release_xyz is not None else (
        default_pose.x,
        default_pose.y,
        args.release_z,
    )
    return ReleaseConfig(
        release_xyz=tuple(float(value) for value in release_xyz),
        release_rpy=tuple(float(value) for value in args.release_rpy),
        retreat_pose=tuple(tuple(float(value) for value in joints) for joints in RIGHT_PRE_JOINTS),
        settle_time=float(args.settle_time),
        target_nut_world_xyz=tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3]),
        hold_after_grasp_s=float(args.hold_after_grasp),
    )


def try_world_reset() -> dict[str, Any]:
    """Use a platform reset function when exposed; target reset remains mandatory."""
    attempts: list[dict[str, Any]] = []
    names = ("reset", "reset_world", "reset_scene", "reset_env", "reset_episode", "restart")
    for module_name in ("rabo_dev_kit", "rabo_robocap"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempts.append({"source": module_name, "error": repr(exc)})
            continue
        for name in names:
            function = getattr(module, name, None)
            if not callable(function):
                continue
            for call_args, kwargs in (((), {"world": WORLD_ID}), ((WORLD_ID,), {}), ((), {})):
                try:
                    result = function(*call_args, **kwargs)
                    failed, reason = result_failed(result)
                    if not failed:
                        return {
                            "success": True,
                            "source": f"{module_name}.{name}",
                            "return_value": jsonable(result),
                            "attempts": attempts,
                        }
                    attempts.append({"source": f"{module_name}.{name}", "error": reason})
                except Exception as exc:
                    attempts.append({"source": f"{module_name}.{name}", "error": repr(exc)})
    return {"success": False, "source": None, "attempts": attempts, "reason": "NO_WORLD_RESET_API_FOUND"}


def reset_trial_environment(pose_setter: Any) -> dict[str, Any]:
    world_reset = try_world_reset()
    initial_pose = pose_to_list(NUT_SPECS["B"].nominal_pose)
    target_reset = set_pose_with_retry(pose_setter, NUT_IDS["B"], initial_pose)
    result = {
        "success": bool(target_reset.get("ok")),
        "world_reset": world_reset,
        "nut_b_pose_reset": target_reset,
        "nut_b_initial_pose": initial_pose,
        "mode": "WORLD_RESET_PLUS_NUT_B_POSE" if world_reset["success"] else "NUT_B_POSE_RESET_FALLBACK",
    }
    if not result["success"]:
        raise ReleaseStabilityError(f"Nut B reset failed: {target_reset.get('error')}")
    time.sleep(RESET_SETTLE_S)
    return result


def detector_workspace(target_xyz: tuple[float, float, float], radius: float) -> dict[str, Any]:
    return {
        "source": "right_release_target_dynamic_roi",
        "nominal_centers_world": {"Nut B release target": list(target_xyz)},
        "bounds_world_xy": [
            target_xyz[0] - radius,
            target_xyz[0] + radius,
            target_xyz[1] - radius,
            target_xyz[1] + radius,
        ],
        "margin_xy_m": [radius, radius],
        "note": "ROI selects candidates only; target coordinates are not returned as detections.",
    }


def detect_released_nut(target_xyz: tuple[float, float, float], radius: float) -> dict[str, Any]:
    """Call the existing detector pipeline and select the candidate nearest target XY."""
    transform, calibration_source = nut_detector.load_camera_to_world()
    message, frames = nut_detector.capture_fresh_cloud()
    if message is None:
        return {"detected": False, "error": "NO_POINTCLOUD", "pointcloud_frames_received": frames}
    points, cloud_info = nut_detector.cloud_xyz(message)
    finite = points[np.isfinite(points).all(axis=1)]
    normal, offset, table_inliers = nut_detector.fit_table_plane(finite)
    if float((transform[:3, :3] @ normal)[2]) < 0.0:
        normal, offset = -normal, -offset
    height = finite @ normal + offset
    workspace = detector_workspace(target_xyz, radius)
    trials = [
        nut_detector.detect_at_clearance(finite, height, transform, workspace, clearance)
        for clearance in nut_detector.TABLE_CLEARANCE_CANDIDATES_M
    ]
    usable = [trial for trial in trials if trial["eligible"]]
    if not usable:
        return {
            "detected": False,
            "error": "NO_ELIGIBLE_NUT_CLUSTER",
            "pointcloud": cloud_info,
            "pointcloud_frames_received": frames,
            "table_inlier_points": table_inliers,
            "clearance_trials": [nut_detector.clearance_summary(trial) for trial in trials],
        }
    chosen_trial = usable[0]
    candidates = chosen_trial["eligible"]
    target_xy = np.asarray(target_xyz[:2], dtype=float)
    candidate = min(
        candidates,
        key=lambda item: float(np.linalg.norm(np.asarray(item["center_world"][:2]) - target_xy)),
    )
    distance_xy = float(np.linalg.norm(np.asarray(candidate["center_world"][:2]) - target_xy))
    public_candidate = {key: value for key, value in candidate.items() if not key.startswith("_")}
    return {
        "detected": distance_xy <= radius,
        "nut_world_xyz": list(public_candidate["center_world"]),
        "selected_cluster": public_candidate,
        "target_xy_distance_m": distance_xy,
        "selected_table_clearance_m": chosen_trial["clearance_m"],
        "candidate_count": len(candidates),
        "pointcloud": cloud_info,
        "pointcloud_frames_received": frames,
        "table_inlier_points": table_inliers,
        "camera_to_world_source": calibration_source,
        "workspace": workspace,
        "clearance_trials": [nut_detector.clearance_summary(trial) for trial in trials],
    }


def check_left_reachability(nut_world_xyz: list[float]) -> dict[str, Any]:
    from rabo_robocap import LinkerArmA7

    left_arm = None
    try:
        left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
        planner = LeftNutGraspPlanner(left_arm=left_arm, left_hand=None)
        grasp_pose = planner.build_grasp_pose(nut_world_xyz, yaw_offset=0.0)
        raw = left_arm.pose_check(
            grasp_pose[0],
            grasp_pose[1],
            grasp_pose[2],
            roll=grasp_pose[3],
            pitch=grasp_pose[4],
            yaw=grasp_pose[5],
        )
        status, reason, value = normalize_pose_check(raw)
        return {
            "reachable": status == "PASS",
            "status": status,
            "reason": reason,
            "raw": value,
            "left_grasp_pose": grasp_pose,
            "source": "LeftNutGraspPlanner.build_grasp_pose + left_arm.pose_check",
        }
    except Exception as exc:
        return {"reachable": False, "status": "ERROR", "reason": repr(exc), "source": "left_arm.pose_check"}
    finally:
        shutdown_device(left_arm)


def print_trial_header(trial_id: int, total: int) -> None:
    print("\n================================")
    print(f"Trial {trial_id}/{total}")
    print("================================")


def execute_release_trial(
    trial_id: int,
    total_trials: int,
    config: ReleaseConfig,
    pose_setter: Any,
    vision_radius: float,
) -> dict[str, Any]:
    print_trial_header(trial_id, total_trials)
    record: dict[str, Any] = {
        "trial_id": trial_id,
        "grasp_success": False,
        "release_success": False,
        "retreat_success": False,
        "detected": False,
        "reachable": False,
        "nut_world_xyz": None,
        "error_xy_mm": None,
        "error_xyz_mm": None,
        "release_timestamp": None,
        "failed_phase": None,
        "error": None,
    }
    bundle = None
    phase = "RESET"
    try:
        record["reset"] = reset_trial_environment(pose_setter)
        bundle = make_right_bundle()

        phase = "PREPARE"
        execute_checked("RIGHT_HAND_OPEN_INITIAL", bundle.right_hand, "clench", *list(HAND_OPEN))
        for index, joints in enumerate(RIGHT_PRE_JOINTS, start=1):
            execute_checked(f"RIGHT_PRE_{index}", bundle.right_arm, "move_joints", list(joints))

        phase = "GRASP"
        move_checked("RIGHT_NUT_B_GRASP_POSE", bundle.right_arm, RIGHT_NUT_B_GRASP_POSE)
        execute_checked("RIGHT_THUMB_TUCK", bundle.right_hand, "clench", thumb_rotation=1.0)
        execute_checked("RIGHT_GRASP_FORCE", bundle.right_hand, "grasp_force", **RIGHT_GRASP_FORCE)
        record["grasp_success"] = True
        print("\nGrasp:\nPASS")
        if config.hold_after_grasp_s > 0:
            time.sleep(config.hold_after_grasp_s)

        phase = "LIFT"
        for index, pose in enumerate(RIGHT_LIFT_POSES, start=1):
            move_checked(f"RIGHT_LIFT_{index}", bundle.right_arm, pose)

        phase = "RELEASE"
        move_checked("RIGHT_RELEASE_POSE", bundle.right_arm, config.release_pose)
        execute_checked("RIGHT_RELEASE_OPEN", bundle.right_hand, "clench", *list(HAND_OPEN))
        released_wall = datetime.now().astimezone()
        record["release_timestamp"] = released_wall.isoformat()
        record["release_timestamp_epoch_s"] = time.time()
        record["release_success"] = True
        print("\nRelease:\nPASS")

        phase = "RETREAT"
        for index, joints in enumerate(config.retreat_pose, start=1):
            execute_checked(f"RIGHT_RETREAT_PRE_{index}", bundle.right_arm, "move_joints", list(joints))
        record["retreat_success"] = True
        print("\nRetreat:\nPASS")

        # Destroy robot clients before reliable high-bandwidth camera capture.
        shutdown_bundle(bundle)
        bundle = None

        phase = "SETTLE"
        print(f"\nWaiting:\n{config.settle_time:g}s")
        time.sleep(config.settle_time)

        phase = "VISION"
        vision = detect_released_nut(config.target_nut_world_xyz, vision_radius)
        record["vision"] = vision
        record["detected"] = bool(vision.get("detected"))
        if not record["detected"]:
            raise ReleaseStabilityError(f"vision failed: {vision.get('error', 'target candidate outside ROI')}")
        xyz = [float(value) for value in vision["nut_world_xyz"]]
        record["nut_world_xyz"] = xyz
        delta = [xyz[index] - config.target_nut_world_xyz[index] for index in range(3)]
        record["error_xyz_vector_mm"] = [value * 1000.0 for value in delta]
        record["error_xy_mm"] = math.hypot(delta[0], delta[1]) * 1000.0
        record["error_xyz_mm"] = math.sqrt(sum(value * value for value in delta)) * 1000.0
        print("\nVision:\nPASS")
        print("\nNut world:")
        print(f"x={xyz[0]:.6f}")
        print(f"y={xyz[1]:.6f}")
        print(f"z={xyz[2]:.6f}")

        phase = "LEFT_REACHABILITY"
        reachability = check_left_reachability(xyz)
        record["left_reachability"] = reachability
        record["reachable"] = bool(reachability.get("reachable"))
        record["success"] = True
    except Exception as exc:
        record["success"] = False
        record["failed_phase"] = phase
        record["error"] = repr(exc)
        print(f"\n{phase}:\nFAIL\n{exc!r}")
    finally:
        shutdown_bundle(bundle)
    return record


def compute_statistics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [trial for trial in trials if trial.get("detected") and trial.get("nut_world_xyz")]
    reachable = [trial for trial in detected if trial.get("reachable")]
    total = len(trials)
    if not detected:
        return {
            "total_trials": total,
            "detect_success": 0,
            "detect_success_rate": 0.0,
            "mean_xyz": None,
            "std_xyz": None,
            "std_xy": None,
            "std_xy_mm": None,
            "max_xy_error": None,
            "max_xy_error_mm": None,
            "max_xy_deviation_from_mean_mm": None,
            "success_rate": 0.0,
            "reachable_count": 0,
            "reachable_rate": 0.0,
        }
    xyz = [trial["nut_world_xyz"][:3] for trial in detected]
    mean_xyz = [statistics.fmean(row[index] for row in xyz) for index in range(3)]
    std_xyz = [statistics.pstdev(row[index] for row in xyz) if len(xyz) > 1 else 0.0 for index in range(3)]
    std_xy = math.hypot(std_xyz[0], std_xyz[1])
    deviations = [math.hypot(row[0] - mean_xyz[0], row[1] - mean_xyz[1]) for row in xyz]
    complete = [
        trial for trial in trials
        if trial.get("grasp_success") and trial.get("release_success") and trial.get("retreat_success") and trial.get("detected")
    ]
    return {
        "total_trials": total,
        "detect_success": len(detected),
        "detect_success_rate": len(detected) / total if total else 0.0,
        "mean_xyz": mean_xyz,
        "std_xyz": std_xyz,
        "std_xy": std_xy,
        "std_xy_mm": std_xy * 1000.0,
        "max_xy_error": max(float(trial["error_xy_mm"]) for trial in detected),
        "max_xy_error_mm": max(float(trial["error_xy_mm"]) for trial in detected),
        "max_xy_deviation_from_mean_mm": max(deviations) * 1000.0,
        "success_rate": len(complete) / total if total else 0.0,
        "reachable_count": len(reachable),
        "reachable_rate": len(reachable) / len(detected),
    }


def write_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"right_release_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(jsonable_pretty(report) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision-feedback right-hand Nut B release stability test.")
    parser.add_argument("--trials", "--repeat", dest="trials", type=int, default=1, help="Number of independent release trials.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI; only terminal logs and JSON are produced.")
    parser.add_argument("--release-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override right-arm base-frame release XYZ.")
    parser.add_argument("--release-z", type=float, default=RIGHT_RELEASE_Z_BASE, help="Right-arm base-frame release Z when --release-xyz is omitted.")
    parser.add_argument("--release-rpy", nargs=3, type=float, default=RIGHT_RELEASE_RPY_BASE, metavar=("R", "P", "Y"), help="Release orientation in radians.")
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S, help="Seconds to wait after retreat before vision.")
    parser.add_argument("--hold-after-grasp", type=float, default=DEFAULT_HOLD_AFTER_GRASP_S, help="Seconds to hold after grasp_force.")
    parser.add_argument("--vision-target-radius", type=float, default=DEFAULT_VISION_TARGET_RADIUS_M, help="World-XY radius used to select released Nut B.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.settle_time < 0:
        raise SystemExit("--settle-time must be >= 0")
    if args.hold_after_grasp < 0:
        raise SystemExit("--hold-after-grasp must be >= 0")
    if args.vision_target_radius <= 0:
        raise SystemExit("--vision-target-radius must be > 0")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    config = build_release_config(args)
    print("\n================================")
    print("Vision-feedback right release stability")
    print("================================")
    print_json("ReleaseConfig", asdict(config))
    print_json("headless", args.headless)

    pose_setter = None
    trials: list[dict[str, Any]] = []
    try:
        from rabo_dev_kit import SetEntityPose

        pose_setter = SetEntityPose(world=WORLD_ID)
        for trial_id in range(1, args.trials + 1):
            trials.append(
                execute_release_trial(
                    trial_id,
                    args.trials,
                    config,
                    pose_setter,
                    args.vision_target_radius,
                )
            )
    except Exception as exc:
        print(f"SETUP FAILED: {exc!r}", file=sys.stderr)
        if not trials:
            trials.append({
                "trial_id": 1,
                "grasp_success": False,
                "release_success": False,
                "retreat_success": False,
                "detected": False,
                "reachable": False,
                "success": False,
                "failed_phase": "SETUP",
                "error": repr(exc),
            })
    finally:
        shutdown_device(pose_setter)

    statistics_result = compute_statistics(trials)
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "headless": args.headless,
        "requested_trials": args.trials,
        "config": {
            "release_pose": pose_to_list(config.release_pose),
            "release_xyz": list(config.release_xyz),
            "release_rpy": list(config.release_rpy),
            "release_z": config.release_xyz[2],
            "retreat_pose": [list(joints) for joints in config.retreat_pose],
            "settle_time": config.settle_time,
            "target_nut_world_xyz": list(config.target_nut_world_xyz),
            "right_pre_joints": [list(joints) for joints in RIGHT_PRE_JOINTS],
            "right_grasp_pose": pose_to_list(RIGHT_NUT_B_GRASP_POSE),
            "right_lift_poses": [pose_to_list(pose) for pose in RIGHT_LIFT_POSES],
            "vision_detector": "tools.test_pointcloud_nut_detection",
        },
        "trials": trials,
        "statistics": statistics_result,
    }
    path = write_report(report)
    print("\n================================")
    print("Statistics")
    print("================================")
    print_json("statistics", statistics_result)
    print(f"report: {path}")
    return 0 if statistics_result["detect_success"] == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
