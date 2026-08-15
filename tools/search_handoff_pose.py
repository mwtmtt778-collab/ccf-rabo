#!/usr/bin/env python3
"""Workspace V3: search pose_check-only handoff candidates.

This script is read-only for robot motion. It only uses pose_check against
candidate Cartesian poses; it never calls move_to, move_joints, hand APIs,
SetEntityPose, reset, Recorder, Replay, or ACT.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_workspace_v3"
OUTPUT_JSON = OUTPUT_DIR / "handoff_candidates.json"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_workspace_v3.log"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS, LEFT_HANDOFF_POSE, RIGHT_LIFT_POSES, Pose6  # noqa: E402
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_BASE_FRAMES  # noqa: E402


LEFT_BASE_WORLD = CONFIRMED_BASE_FRAMES["left_arm_base"]["world_pose"]
RIGHT_BASE_WORLD = CONFIRMED_BASE_FRAMES["right_arm_base"]["world_pose"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def log_event(event: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now_text(), "event": event, **payload}, ensure_ascii=False) + "\n")


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
    if isinstance(value, list) and value:
        if isinstance(value[0], bool):
            return ("PASS" if value[0] else "FAIL", str(value[1] if len(value) > 1 else ""), value)
    text = str(value)
    low = text.lower()
    if "true" in low or "reachable" in low or "success" in low:
        return "PASS", text, value
    if "false" in low or "fail" in low or "limit" in low or "workspace" in low:
        return "FAIL", text, value
    return "CHECK", text, value


def make_arm(arm: str) -> Any:
    from rabo_robocap import LinkerArmA7

    return LinkerArmA7(robot_id=DEVICE_IDS[arm], mode="sim")


def call_pose_check(arm_obj: Any, pose: list[float]) -> tuple[str, str, Any]:
    raw = arm_obj.pose_check(pose[0], pose[1], pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5])
    return normalize_pose_check(raw)


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + step * 0.5:
        values.append(round(current, 6))
        current += step
    return values


def xyz_distance(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt(sum((av[i] - bv[i]) ** 2 for i in range(3)))


def default_search_bounds() -> dict[str, Any]:
    left_seed_world = transform_pose_base_to_world(pose_to_list(LEFT_HANDOFF_POSE), LEFT_BASE_WORLD)
    right_seed_worlds = [transform_pose_base_to_world(pose_to_list(pose), RIGHT_BASE_WORLD) for pose in RIGHT_LIFT_POSES]
    seed_points = [left_seed_world, *right_seed_worlds]
    xs = [p[0] for p in seed_points]
    ys = [p[1] for p in seed_points]
    zs = [p[2] for p in seed_points]
    return {
        "x": [round(min(xs) - 0.12, 3), round(max(xs) + 0.12, 3), 0.04],
        "y": [round(min(ys) - 0.10, 3), round(max(ys) + 0.10, 3), 0.04],
        "z": [max(0.36, round(min(zs) - 0.16, 3)), min(0.76, round(max(zs) + 0.08, 3)), 0.04],
        "seed_world_poses": {
            "left_handoff_pose_world": left_seed_world,
            "right_lift_poses_world": right_seed_worlds,
        },
    }


def candidate_orientations() -> list[dict[str, Any]]:
    left_seed_world = transform_pose_base_to_world(pose_to_list(LEFT_HANDOFF_POSE), LEFT_BASE_WORLD)
    right_seed_world = transform_pose_base_to_world(pose_to_list(RIGHT_LIFT_POSES[-1]), RIGHT_BASE_WORLD)
    return [
        {
            "name": "left_handoff_orientation",
            "rpy": left_seed_world[3:],
            "source": "agents/three_nut_expert/config.py LEFT_HANDOFF_POSE transformed to world",
        },
        {
            "name": "right_lift_orientation",
            "rpy": right_seed_world[3:],
            "source": "agents/three_nut_expert/config.py RIGHT_LIFT_POSES[-1] transformed to world",
        },
        {
            "name": "neutral_down_orientation",
            "rpy": [0.0, 1.0, 1.57],
            "source": "Workspace V3 coarse neutral handoff orientation",
        },
    ]


def build_candidates(bounds: dict[str, Any], max_candidates: int | None = None) -> list[dict[str, Any]]:
    x_values = frange(*bounds["x"])
    y_values = frange(*bounds["y"])
    z_values = frange(*bounds["z"])
    orientations = candidate_orientations()
    candidates = []
    index = 0
    for x in x_values:
        for y in y_values:
            for z in z_values:
                for orientation in orientations:
                    world_pose = [x, y, z, *orientation["rpy"]]
                    left_base_pose = transform_pose_world_to_base(world_pose, LEFT_BASE_WORLD)
                    right_base_pose = transform_pose_world_to_base(world_pose, RIGHT_BASE_WORLD)
                    candidates.append(
                        {
                            "candidate_id": f"handoff_{index:05d}",
                            "world_pose": world_pose,
                            "left_base_pose": left_base_pose,
                            "right_base_pose": right_base_pose,
                            "orientation_name": orientation["name"],
                            "orientation_source": orientation["source"],
                            "distance_to_left_arm": xyz_distance(world_pose, LEFT_BASE_WORLD),
                            "distance_to_right_arm": xyz_distance(world_pose, RIGHT_BASE_WORLD),
                        }
                    )
                    index += 1
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates
    return candidates


def evaluate_candidates(
    candidates: list[dict[str, Any]], skip_runtime: bool, progress_every: int
) -> tuple[list[dict[str, Any]], str | None]:
    runtime_error = None
    left_arm = None
    right_arm = None
    if not skip_runtime:
        try:
            left_arm = make_arm("LEFT_ARM")
            right_arm = make_arm("RIGHT_ARM")
        except Exception as exc:
            runtime_error = repr(exc)
            log_event("RUNTIME_INIT_ERROR", {"error": runtime_error})

    total = len(candidates)
    print(f"开始评估 handoff candidates: {total}", flush=True)
    for index, candidate in enumerate(candidates, start=1):
        if skip_runtime:
            candidate.update(
                {
                    "left_result": "CHECK",
                    "left_reason": "SKIPPED_RUNTIME",
                    "left_raw": None,
                    "right_result": "CHECK",
                    "right_reason": "SKIPPED_RUNTIME",
                    "right_raw": None,
                }
            )
        elif runtime_error:
            candidate.update(
                {
                    "left_result": "CHECK",
                    "left_reason": f"RUNTIME_INIT_FAILED: {runtime_error}",
                    "left_raw": None,
                    "right_result": "CHECK",
                    "right_reason": f"RUNTIME_INIT_FAILED: {runtime_error}",
                    "right_raw": None,
                }
            )
        else:
            assert left_arm is not None and right_arm is not None
            left_status, left_reason, left_raw = call_pose_check(left_arm, candidate["left_base_pose"])
            right_status, right_reason, right_raw = call_pose_check(right_arm, candidate["right_base_pose"])
            candidate.update(
                {
                    "left_result": left_status,
                    "left_reason": left_reason,
                    "left_raw": left_raw,
                    "right_result": right_status,
                    "right_reason": right_reason,
                    "right_raw": right_raw,
                }
            )
        candidate["common_reachable"] = candidate["left_result"] == "PASS" and candidate["right_result"] == "PASS"
        candidate["cost"] = candidate["distance_to_right_arm"] + candidate["distance_to_left_arm"]
        log_event("HANDOFF_CANDIDATE_RESULT", candidate)
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(
                "进度 {}/{} | common={} | left={} | right={}".format(
                    index,
                    total,
                    sum(1 for item in candidates[:index] if item.get("common_reachable")),
                    sum(1 for item in candidates[:index] if item.get("left_result") == "PASS"),
                    sum(1 for item in candidates[:index] if item.get("right_result") == "PASS"),
                ),
                flush=True,
            )

    for arm_obj in (left_arm, right_arm):
        if hasattr(arm_obj, "shutdown"):
            try:
                arm_obj.shutdown()
            except Exception as exc:
                log_event("ARM_SHUTDOWN_ERROR", {"error": repr(exc)})
    return candidates, runtime_error


def summarize(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    right_reachable = [item for item in candidates if item.get("right_result") == "PASS"]
    left_reachable = [item for item in candidates if item.get("left_result") == "PASS"]
    common = [item for item in candidates if item.get("common_reachable")]
    top_common = sorted(common, key=lambda item: item["cost"])[:10]
    return {
        "number_of_right_reachable": len(right_reachable),
        "number_of_left_reachable": len(left_reachable),
        "number_of_common_reachable": len(common),
        "top_10_recommended_handoff_poses": top_common,
    }


def write_output(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    defaults = default_search_bounds()
    parser = argparse.ArgumentParser(description="Workspace V3 handoff pose search using pose_check only.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument("--skip-runtime", action="store_true", help="Generate candidates without importing Rabo runtime.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit candidates for quick smoke tests.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N candidates; use 0 to disable.")
    parser.add_argument("--x-min", type=float, default=defaults["x"][0])
    parser.add_argument("--x-max", type=float, default=defaults["x"][1])
    parser.add_argument("--x-step", type=float, default=defaults["x"][2])
    parser.add_argument("--y-min", type=float, default=defaults["y"][0])
    parser.add_argument("--y-max", type=float, default=defaults["y"][1])
    parser.add_argument("--y-step", type=float, default=defaults["y"][2])
    parser.add_argument("--z-min", type=float, default=defaults["z"][0])
    parser.add_argument("--z-max", type=float, default=defaults["z"][1])
    parser.add_argument("--z-step", type=float, default=defaults["z"][2])
    parser.set_defaults(seed_world_poses=defaults["seed_world_poses"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()
    bounds = {
        "x": [args.x_min, args.x_max, args.x_step],
        "y": [args.y_min, args.y_max, args.y_step],
        "z": [args.z_min, args.z_max, args.z_step],
    }
    print("========================================", flush=True)
    print("Workspace V3 Handoff Pose Search", flush=True)
    print("========================================", flush=True)
    print(f"搜索范围：{bounds}", flush=True)
    print(f"skip_runtime：{args.skip_runtime}", flush=True)
    log_event("HANDOFF_SEARCH_START", {"bounds": bounds, "skip_runtime": args.skip_runtime})
    candidates = build_candidates(bounds, max_candidates=args.max_candidates)
    print(f"候选生成完成：{len(candidates)}", flush=True)
    candidates, runtime_error = evaluate_candidates(
        candidates, skip_runtime=args.skip_runtime, progress_every=args.progress_every
    )
    summary = summarize(candidates)
    result = {
        "generated": now_text(),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "phase": "workspace_v3_handoff_pose_search",
        "safety": {
            "allowed_calls": ["pose_check"],
            "forbidden_calls": ["move_to", "move_joints", "home", "SetEntityPose", "clench", "grasp_force", "reset", "Recorder", "Replay", "ACT"],
            "runtime_error": runtime_error,
        },
        "base_frames": {
            "T_world_left_base": LEFT_BASE_WORLD,
            "T_world_right_base": RIGHT_BASE_WORLD,
            "source": "tools.resolve_workspace_coordinates.CONFIRMED_BASE_FRAMES",
        },
        "search_bounds": bounds,
        "seed_world_poses": args.seed_world_poses,
        "candidate_count": len(candidates),
        **summary,
        "candidates": candidates,
    }
    write_output(result, args.output)
    log_event("HANDOFF_SEARCH_END", {**summary, "output": str(args.output)})

    print("========================================")
    print("Workspace V3 Handoff Pose Search Result")
    print("========================================")
    print(f"候选总数：{len(candidates)}")
    print(f"RIGHT reachable：{summary['number_of_right_reachable']}")
    print(f"LEFT reachable：{summary['number_of_left_reachable']}")
    print(f"COMMON reachable：{summary['number_of_common_reachable']}")
    print(f"Top 10：{len(summary['top_10_recommended_handoff_poses'])}")
    print(f"输出：{args.output.relative_to(PROJECT_ROOT) if args.output.is_absolute() and args.output.is_relative_to(PROJECT_ROOT) else args.output}")
    print(f"日志：{LOG_PATH.relative_to(PROJECT_ROOT)}")
    if runtime_error:
        print(f"Runtime：{runtime_error}")
    print("是否进入真实动作：NO")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
