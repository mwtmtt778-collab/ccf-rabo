#!/usr/bin/env python3
"""Right-hand Nut B release stability test.

This tool measures where Nut B physically settles after the fixed right-hand
release strategy. It does not move Nut B with SetEntityPose and does not run
the left-hand planner.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "right_release_stability"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    NUT_IDS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import jsonable, result_failed  # noqa: E402
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_BASE_FRAMES  # noqa: E402
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402


TARGET_NUT_B_WORLD_POSE = [-0.2966, 0.0420, 0.2806, 0.0, 0.0, 0.5233]
RIGHT_RELEASE_Z_BASE = -0.25
RIGHT_RELEASE_RPY_BASE = [0.0, 0.8, 0.0]
RIGHT_RETREAT_OFFSET = [-0.05, 0.0, 0.10]
DEFAULT_SETTLE_AFTER_RELEASE_S = 10.0
DEFAULT_HOLD_AFTER_GRASP_S = 0.7


class ReleaseStabilityError(RuntimeError):
    pass


def jsonable_pretty(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2)


def print_json(label: str, payload: Any) -> None:
    print(f"{label}: {json.dumps(jsonable(payload), ensure_ascii=False)}")


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return asdict(pose)


def pose_from_list(values: list[float]) -> Pose6:
    if len(values) != 6:
        raise ValueError(f"pose must have 6 values, got {len(values)}")
    return Pose6(*[float(v) for v in values])


def pose_with_xyz_offset(pose: Pose6, offset: list[float]) -> Pose6:
    return Pose6(
        pose.x + offset[0],
        pose.y + offset[1],
        pose.z + offset[2],
        pose.roll,
        pose.pitch,
        pose.yaw,
    )


def execute_checked(label: str, target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name)
    result = method(*args, **kwargs)
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


def shutdown_bundle(bundle: Any | None) -> None:
    if bundle is None:
        return
    for name in ("right_hand", "right_arm"):
        device = getattr(bundle, name, None)
        if hasattr(device, "shutdown"):
            device.shutdown()


def build_release_plan() -> dict[str, Any]:
    right_base_world = CONFIRMED_BASE_FRAMES["right_arm_base"]["world_pose"]
    initial_nut_world = [-0.3413, -0.1710, 0.2806, 0.0, 0.0, 0.5233]
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
    retreat_pose = pose_with_xyz_offset(release_pose, RIGHT_RETREAT_OFFSET)
    return {
        "target_nut_world_pose": TARGET_NUT_B_WORLD_POSE,
        "right_grasp_pose_base": pose_to_list(RIGHT_NUT_B_GRASP_POSE),
        "right_lift_poses_base": [pose_to_list(pose) for pose in RIGHT_LIFT_POSES],
        "right_release_pose_base": pose_to_list(release_pose),
        "right_retreat_pose_base": pose_to_list(retreat_pose),
        "right_release_z_base": RIGHT_RELEASE_Z_BASE,
        "right_release_rpy_base": RIGHT_RELEASE_RPY_BASE,
        "right_retreat_offset": RIGHT_RETREAT_OFFSET,
        "hand_to_nut_offset_world": hand_to_nut_offset_world,
    }


def parse_pose6(value: Any) -> list[float] | None:
    value = jsonable(value)
    if isinstance(value, dict):
        if "pose" in value:
            parsed = parse_pose6(value["pose"])
            if parsed is not None:
                return parsed
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        if all(key in value for key in keys):
            return [float(value[key]) for key in keys]
        position = value.get("position") or value.get("xyz")
        orientation = value.get("orientation") or value.get("rpy")
        if position is not None:
            xyz = parse_xyz(position)
            rpy = parse_rpy(orientation)
            if xyz is not None:
                return [*xyz, *rpy]
    if isinstance(value, (list, tuple)):
        if len(value) >= 6 and all(isinstance(item, (int, float)) for item in value[:6]):
            return [float(item) for item in value[:6]]
        if len(value) >= 3 and all(isinstance(item, (int, float)) for item in value[:3]):
            return [float(value[0]), float(value[1]), float(value[2]), 0.0, 0.0, 0.0]
    return None


def parse_xyz(value: Any) -> list[float] | None:
    value = jsonable(value)
    if isinstance(value, dict) and all(key in value for key in ("x", "y", "z")):
        return [float(value["x"]), float(value["y"]), float(value["z"])]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def parse_rpy(value: Any) -> list[float]:
    value = jsonable(value)
    if isinstance(value, dict) and all(key in value for key in ("roll", "pitch", "yaw")):
        return [float(value["roll"]), float(value["pitch"]), float(value["yaw"])]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return [0.0, 0.0, 0.0]


def candidate_call_patterns(thing_id: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    return [
        ((thing_id,), {}),
        ((), {"thing_id": thing_id}),
        ((), {"entity_id": thing_id}),
        ((), {"object_id": thing_id}),
        ((), {"id": thing_id}),
    ]


def try_call_reader(obj: Any, method_name: str, thing_id: str) -> tuple[list[float] | None, Any, str | None]:
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return None, None, None
    errors = []
    for args, kwargs in candidate_call_patterns(thing_id):
        try:
            raw = method(*args, **kwargs)
            pose = parse_pose6(raw)
            if pose is not None:
                return pose, raw, None
            errors.append(f"{method_name}{args}{kwargs}: unparsed {repr(jsonable(raw))[:200]}")
        except TypeError as exc:
            errors.append(f"{method_name}{args}{kwargs}: {exc}")
        except Exception as exc:
            errors.append(f"{method_name}{args}{kwargs}: {repr(exc)}")
    return None, None, "; ".join(errors[-3:])


def instantiate_candidate(cls: Any) -> list[Any]:
    instances = []
    for args, kwargs in (
        ((), {"world": WORLD_ID}),
        ((WORLD_ID,), {}),
        ((), {"world_id": WORLD_ID}),
        ((), {}),
    ):
        try:
            instances.append(cls(*args, **kwargs))
        except Exception:
            continue
    return instances


def read_nut_world_pose(thing_id: str) -> tuple[list[float] | None, dict[str, Any]]:
    method_names = [
        "get_entity_pose",
        "get_object_pose",
        "get_thing_pose",
        "get_pose",
        "get_entity_state",
        "get_object_state",
        "read_entity_pose",
        "read_object_pose",
        "query_entity_pose",
        "query_object_pose",
    ]
    class_names = [
        "GetEntityPose",
        "GetObjectPose",
        "EntityPose",
        "ObjectPose",
        "EntityState",
        "ObjectState",
        "WorldState",
        "SceneState",
        "RemoteControl",
    ]
    attempts = []
    for module_name in ("rabo_dev_kit", "rabo_robocap"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempts.append({"source": module_name, "error": repr(exc)})
            continue
        for method_name in method_names:
            pose, raw, error = try_call_reader(module, method_name, thing_id)
            attempts.append({"source": f"{module_name}.{method_name}", "success": pose is not None, "error": error})
            if pose is not None:
                return pose, {"source": f"{module_name}.{method_name}", "raw": jsonable(raw), "attempts": attempts}
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            for instance in instantiate_candidate(cls):
                for method_name in method_names:
                    pose, raw, error = try_call_reader(instance, method_name, thing_id)
                    attempts.append({"source": f"{module_name}.{class_name}.{method_name}", "success": pose is not None, "error": error})
                    if pose is not None:
                        return pose, {"source": f"{module_name}.{class_name}.{method_name}", "raw": jsonable(raw), "attempts": attempts}
    return None, {"source": None, "attempts": attempts, "error": "NO_READABLE_ENTITY_POSE_API_FOUND"}


def try_reset_environment() -> dict[str, Any]:
    names = [
        "reset",
        "reset_world",
        "reset_scene",
        "reset_env",
        "reset_episode",
        "restart",
    ]
    attempts = []
    for module_name in ("rabo_dev_kit", "rabo_robocap"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempts.append({"source": module_name, "error": repr(exc)})
            continue
        for name in names:
            fn = getattr(module, name, None)
            if not callable(fn):
                continue
            for args, kwargs in (((), {"world": WORLD_ID}), ((WORLD_ID,), {}), ((), {})):
                try:
                    result = fn(*args, **kwargs)
                    return {"success": True, "source": f"{module_name}.{name}", "return_value": jsonable(result)}
                except Exception as exc:
                    attempts.append({"source": f"{module_name}.{name}", "args": args, "kwargs": kwargs, "error": repr(exc)})
    return {"success": False, "source": None, "attempts": attempts, "reason": "NO_CONFIRMED_RESET_API_FOUND"}


def execute_release_trial(args: argparse.Namespace, trial: int, plan: dict[str, Any]) -> dict[str, Any]:
    print("")
    print("================================")
    print("右手释放稳定性测试")
    print("")
    print(f"第{trial}次实验")
    print("================================")

    reset_result = try_reset_environment()
    if reset_result["success"]:
        print(f"环境 reset 完成: {reset_result['source']}")
        time.sleep(2.0)
    else:
        print("未找到可确认的环境 reset 接口；本次不直接移动 Nut B。")

    bundle = None
    record: dict[str, Any] = {
        "trial": trial,
        "success": False,
        "target_nut_world_pose": TARGET_NUT_B_WORLD_POSE,
        "nut_world_pose": None,
        "error_xyz": None,
        "reset": reset_result,
        "error": None,
    }
    try:
        bundle = make_right_bundle()
        release_pose = pose_from_list(plan["right_release_pose_base"])
        retreat_pose = pose_from_list(plan["right_retreat_pose_base"])

        for index, joints in enumerate(RIGHT_PRE_JOINTS, start=1):
            execute_checked(f"RIGHT_PRE_{index}", bundle.right_arm, "move_joints", joints)
        move_checked("RIGHT_NUT_B_GRASP_POSE", bundle.right_arm, RIGHT_NUT_B_GRASP_POSE)
        execute_checked("RIGHT_THUMB_TUCK", bundle.right_hand, "clench", thumb_rotation=1.0)
        execute_checked("RIGHT_GRASP_FORCE", bundle.right_hand, "grasp_force", **RIGHT_GRASP_FORCE)
        time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)

        for index, pose in enumerate(RIGHT_LIFT_POSES, start=1):
            move_checked(f"RIGHT_LIFT_{index}", bundle.right_arm, pose)
        move_checked("RIGHT_RELEASE_POSE", bundle.right_arm, release_pose)

        print("释放前等待稳定...")
        time.sleep(DEFAULT_SETTLE_AFTER_RELEASE_S)
        execute_checked("RIGHT_RELEASE_OPEN", bundle.right_hand, "clench", *list(HAND_OPEN))
        print("释放完成")
        print("")
        print("等待物理稳定...")
        time.sleep(DEFAULT_SETTLE_AFTER_RELEASE_S)

        actual_pose, read_meta = read_nut_world_pose(NUT_IDS["B"])
        record["pose_reader"] = read_meta
        if actual_pose is None:
            raise ReleaseStabilityError(read_meta["error"])
        error_xyz = [actual_pose[index] - TARGET_NUT_B_WORLD_POSE[index] for index in range(3)]
        record["nut_world_pose"] = actual_pose
        record["error_xyz"] = error_xyz
        record["success"] = True

        print("")
        print("Nut B 最终位置:")
        print(f"X: {actual_pose[0]:.6f}")
        print(f"Y: {actual_pose[1]:.6f}")
        print(f"Z: {actual_pose[2]:.6f}")
        print("")
        print("误差:")
        print(f"dX: {error_xyz[0]:.6f}")
        print(f"dY: {error_xyz[1]:.6f}")
        print(f"dZ: {error_xyz[2]:.6f}")

        move_checked("RIGHT_RETREAT", bundle.right_arm, retreat_pose)
        return record
    except Exception as exc:
        record["error"] = repr(exc)
        print("本次实验失败:")
        print(repr(exc))
        return record
    finally:
        shutdown_bundle(bundle)


def compute_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row.get("success") and row.get("nut_world_pose")]
    if not successful:
        return {
            "success_count": 0,
            "mean_xyz": None,
            "std_xyz": None,
            "max_error": None,
            "mean_error": None,
            "max_error_norm": None,
        }
    xyz_rows = [row["nut_world_pose"][:3] for row in successful]
    err_rows = [row["error_xyz"] for row in successful]
    norms = [math.sqrt(sum(value * value for value in row)) for row in err_rows]
    mean_xyz = [statistics.fmean(row[index] for row in xyz_rows) for index in range(3)]
    std_xyz = [
        statistics.pstdev(row[index] for row in xyz_rows) if len(xyz_rows) > 1 else 0.0
        for index in range(3)
    ]
    max_error = [max(abs(row[index]) for row in err_rows) for index in range(3)]
    return {
        "success_count": len(successful),
        "mean_xyz": mean_xyz,
        "std_xyz": std_xyz,
        "max_error": max_error,
        "mean_error": statistics.fmean(norms),
        "max_error_norm": max(norms),
    }


def write_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"right_release_stability_{datetime.now().strftime('%Y%m%d')}.json"
    path.write_text(jsonable_pretty(report) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test right-hand Nut B physical release stability.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of release trials.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    plan = build_release_plan()
    print("")
    print("================================")
    print("右手释放稳定性测试")
    print("================================")
    print_json("target_nut_world_pose", TARGET_NUT_B_WORLD_POSE)
    print_json("right_release_pose_base", plan["right_release_pose_base"])
    print_json("right_retreat_pose_base", plan["right_retreat_pose_base"])
    print_json("fixed_parameters", {
        "RIGHT_RELEASE_Z_BASE": RIGHT_RELEASE_Z_BASE,
        "RIGHT_RELEASE_RPY_BASE": RIGHT_RELEASE_RPY_BASE,
        "RIGHT_RETREAT_OFFSET": RIGHT_RETREAT_OFFSET,
        "DEFAULT_SETTLE_AFTER_RELEASE_S": DEFAULT_SETTLE_AFTER_RELEASE_S,
        "DEFAULT_HOLD_AFTER_GRASP_S": DEFAULT_HOLD_AFTER_GRASP_S,
    })

    results = [execute_release_trial(args, trial, plan) for trial in range(1, args.repeat + 1)]
    stats = compute_statistics(results)
    report = {
        "repeat": args.repeat,
        "target_nut_world_pose": TARGET_NUT_B_WORLD_POSE,
        "plan": plan,
        "release_results": results,
        "statistics": stats,
    }
    report_path = write_report(report)
    print("")
    print("================================")
    print("统计结果")
    print("================================")
    print_json("statistics", stats)
    print(f"报告: {report_path}")
    return 0 if stats["success_count"] == args.repeat else 1


if __name__ == "__main__":
    raise SystemExit(main())
