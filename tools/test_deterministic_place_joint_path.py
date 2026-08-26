#!/usr/bin/env python3
"""Physical gate for recorded C/B deterministic place joint paths.

Dry-run is the default.  Execute mode never resets the world and refuses to
move unless the current left-arm joints are already close to the reference
Episode's LEFT_SAFE_LIFT exit / transport entrance sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "deterministic_place_joint_path"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DETERMINISTIC_PLACE_ENTRY_TOLERANCE_RAD,
    LEFT_FIXED_PLACE_JOINT_PATHS,
    LEFT_FIXED_PLACE_REFERENCE_EPISODE,
)
from tools.motion_monitor import MotionMonitor, max_abs_error, read_joint_method  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_three_nut_closed_loop_v2 import (  # noqa: E402
    ExpertStateRunner,
    execute_recorded_joint_path,
    jsonable,
    now_text,
)


START_ANCHOR_TOLERANCE_RAD = DETERMINISTIC_PLACE_ENTRY_TOLERANCE_RAD


def path_contract(reference_root: Path | None = None) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "reference_episode_frozen": (
            LEFT_FIXED_PLACE_REFERENCE_EPISODE == "episode_20260826_144708_105746"
        ),
        "only_c_b": set(LEFT_FIXED_PLACE_JOINT_PATHS) == {"C", "B"},
        "c_has_transport_place_retreat": set(LEFT_FIXED_PLACE_JOINT_PATHS["C"])
        == {"transport", "place", "retreat"},
        "b_has_transport_place_only": set(LEFT_FIXED_PLACE_JOINT_PATHS["B"])
        == {"transport", "place"},
    }
    summary: dict[str, Any] = {}
    source_cache: dict[Path, dict[str, Any]] = {}
    for nut, segments in LEFT_FIXED_PLACE_JOINT_PATHS.items():
        summary[nut] = {}
        for segment, waypoints in segments.items():
            joints = [item.joints for item in waypoints]
            deltas = [
                max(abs(after - before) for before, after in zip(a, b))
                for a, b in zip(joints, joints[1:])
            ]
            segment_checks = {
                "waypoint_count_3_to_6": 3 <= len(waypoints) <= 6,
                "all_7d_finite": all(
                    len(item.joints) == 7 and all(math.isfinite(value) for value in item.joints)
                    for item in waypoints
                ),
                "all_from_motion_monitor_trajectory": all(
                    item.source_trajectory.startswith("reports/motion_monitor/")
                    and item.source_trajectory.endswith("/trajectory.json")
                    for item in waypoints
                ),
                "sample_indices_nonnegative": all(item.sample_index >= 0 for item in waypoints),
                "times_nonnegative": all(item.time_s >= 0.0 for item in waypoints),
                "adjacent_waypoint_delta_lt_1rad": max(deltas, default=0.0) < 1.0,
            }
            if reference_root is not None:
                source_path = reference_root / waypoints[0].source_trajectory
                try:
                    source = source_cache.setdefault(
                        source_path,
                        json.loads(source_path.read_text(encoding="utf-8")),
                    )
                    samples = source["samples"]
                    segment_checks["source_status_completed"] = (
                        source.get("metadata", {}).get("status") == "COMPLETED"
                    )
                    segment_checks["exact_actual_joint_source_match"] = all(
                        list(item.joints) == samples[item.sample_index]["actual_joint"]
                        and item.time_s == samples[item.sample_index]["time"]
                        for item in waypoints
                    )
                except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                    segment_checks["source_status_completed"] = False
                    segment_checks["exact_actual_joint_source_match"] = False
            checks.update({f"{nut}_{segment}_{name}": value for name, value in segment_checks.items()})
            summary[nut][segment] = {
                "waypoint_count": len(waypoints),
                "max_adjacent_waypoint_delta_rad": max(deltas, default=0.0),
                "source_trajectory": waypoints[0].source_trajectory,
                "samples": [
                    {"sample_index": item.sample_index, "time_s": item.time_s}
                    for item in waypoints
                ],
            }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "paths": summary,
        "b_retreat": "UNAVAILABLE_FAILED_REFERENCE; B IS TERMINAL AFTER PLACE/RELEASE",
        "reference_root": str(reference_root) if reference_root is not None else None,
    }


def execute_test(nut: str) -> int:
    report_path = REPORT_DIR / f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_{nut}.json"
    left_bundle = None
    runner = ExpertStateRunner(1, MotionMonitor(enabled=True), report_path, nut=nut)
    report: dict[str, Any] = {
        "experiment": "deterministic_place_joint_path_physical_gate",
        "nut": nut,
        "reference_episode": LEFT_FIXED_PLACE_REFERENCE_EPISODE,
        "status": "RUNNING",
        "world_reset_used": False,
        "start_anchor_tolerance_rad": START_ANCHOR_TOLERANCE_RAD,
        "states": runner.states,
    }
    try:
        left_bundle = make_left_bundle()
        actual, error = read_joint_method(
            left_bundle.left_arm, ("get_joint_angles", "get_joints", "get_qpos")
        )
        if actual is None:
            raise RuntimeError(f"left-arm joint state unavailable: {error}")
        anchor = list(LEFT_FIXED_PLACE_JOINT_PATHS[nut]["transport"][0].joints)
        anchor_error = max_abs_error(anchor, actual)
        report.update({"joint_before": actual, "required_start_anchor": anchor, "start_anchor_error_rad": anchor_error})
        if anchor_error is None or anchor_error > START_ANCHOR_TOLERANCE_RAD:
            raise RuntimeError(
                "unsafe start refused: place the left arm at the reference LEFT_SAFE_LIFT "
                f"exit first; max joint error={anchor_error} rad"
            )

        runner.enter("LEFT_PLACE_ABOVE", nut=nut)
        runner.pass_state({"motion": execute_recorded_joint_path(runner, left_bundle.left_arm, nut, "transport")})
        runner.enter("LEFT_PLACE", nut=nut)
        runner.pass_state({"motion": execute_recorded_joint_path(runner, left_bundle.left_arm, nut, "place")})
        if nut == "C":
            runner.enter("LEFT_SAFE_RETREAT", nut=nut)
            runner.pass_state({"motion": execute_recorded_joint_path(runner, left_bundle.left_arm, nut, "retreat")})
        runner.enter("DONE", nut=nut)
        runner.pass_state({"terminal_at_place": nut == "B", "hand_commands_sent": False})
        report["status"] = "PASS"
        return_code = 0
    except BaseException as exc:
        report.update({"status": "FAILED", "failure_reason": repr(exc), "failed_state": runner.state})
        return_code = 1
    finally:
        shutdown_left_bundle(left_bundle)
        report["states"] = runner.states
        report["finished_at"] = now_text()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Result: {report['status']}")
        print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate one recorded deterministic C/B place joint path.")
    parser.add_argument("--nut", required=True, choices=("C", "B"))
    parser.add_argument("--execute", action="store_true", help="Drive only after the left arm matches the recorded start anchor.")
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional extracted Episode root used to verify every waypoint against samples[*].actual_joint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = path_contract(args.reference_root)
    if contract["result"] != "PASS":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return 2
    if not args.execute:
        print("DRY RUN: no robot, hand, or reset API was invoked.")
        print(json.dumps({
            **contract,
            "selected_nut": args.nut,
            "execute_precondition": (
                "Current left-arm joints must be within 0.10 rad of the selected transport sample 0."
            ),
        }, ensure_ascii=False, indent=2))
        return 0
    return execute_test(args.nut)


if __name__ == "__main__":
    raise SystemExit(main())
