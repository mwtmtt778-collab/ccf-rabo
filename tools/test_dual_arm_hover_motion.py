#!/usr/bin/env python3
"""Run safe hover-only repeatability tests for dual-arm strategy selection.

This tool never closes the hands and never grasps. It only calls get_pose,
get_joint_angles, move_to, and stop on one A7 arm at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_strategy"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_strategy.log"
JSON_PATH = OUTPUT_DIR / "arm_strategy_summary.json"
REPORT_PATH = OUTPUT_DIR / "ARM_STRATEGY_EVALUATION_REPORT.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS, RIGHT_CLEAR_POSE, Pose6  # noqa: E402
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from tools.test_dual_arm_reachability import (  # noqa: E402
    build_arm_targets,
    matrix_rows,
    source_coordinates,
    strategy_status,
)


LEFT_SAFE_POSE = Pose6(0.43, 0.3, -0.1, 0.0, 1.3, 1.57)
RIGHT_SAFE_POSE = RIGHT_CLEAR_POSE


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(event: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    line = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "event": event,
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def load_reachability() -> dict[str, Any]:
    if not JSON_PATH.exists():
        raise SystemExit(
            "Reachability JSON is missing. Run: python3 tools/test_dual_arm_reachability.py first."
        )
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def target_lookup() -> dict[tuple[str, str], Pose6]:
    lookup: dict[tuple[str, str], Pose6] = {}
    for target in build_arm_targets():
        if target.pose is not None:
            lookup[(target.arm, target.target)] = target.pose
    return lookup


def allowed_targets(reachability: dict[str, Any], arms: set[str] | None, target_names: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    for row in reachability.get("reachability", []):
        if row.get("result") != "PASS":
            continue
        if arms and row.get("arm") not in arms:
            continue
        if target_names and row.get("target") not in target_names:
            continue
        rows.append(row)
    return rows


def make_arm(arm: str) -> Any:
    from rabo_robocap import LinkerArmA7

    return LinkerArmA7(robot_id=DEVICE_IDS[arm], mode="sim")


def pose_kwargs(pose: Pose6) -> dict[str, float]:
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "roll": pose.roll,
        "pitch": pose.pitch,
        "yaw": pose.yaw,
    }


def move_to_pose(arm_obj: Any, pose: Pose6) -> Any:
    return arm_obj.move_to(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "range": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def joint_stats(samples: list[list[float]]) -> dict[str, Any]:
    if not samples:
        return {"joint_count": 0, "per_joint": [], "max_joint_range": None}
    count = max(len(sample) for sample in samples)
    per_joint = []
    for idx in range(count):
        vals = [sample[idx] for sample in samples if len(sample) > idx]
        row = stats(vals)
        row["joint"] = idx
        per_joint.append(row)
    ranges = [row["range"] for row in per_joint if row["range"] is not None]
    return {
        "joint_count": count,
        "per_joint": per_joint,
        "max_joint_range": max(ranges) if ranges else None,
    }


def classify_repeatability(rep: dict[str, Any]) -> str:
    max_range = rep.get("joint_stats", {}).get("max_joint_range")
    if max_range is None:
        return "CHECK"
    if max_range <= 0.25:
        return "PASS"
    if max_range <= 0.8:
        return "CHECK"
    return "FAIL"


def run_hover_tests(repeats: int, arms_filter: set[str] | None, targets_filter: set[str] | None) -> dict[str, Any]:
    ensure_dirs()
    reachability = load_reachability()
    lookup = target_lookup()
    rows = allowed_targets(reachability, arms_filter, targets_filter)
    if not rows:
        raise SystemExit("No PASS reachability targets selected. Hover motion is blocked.")

    log_event("HOVER_START", {"repeats": repeats, "targets": rows})
    repeatability: dict[str, Any] = {}

    for row in rows:
        arm_name = row["arm"]
        target_name = row["target"]
        pose = lookup[(arm_name, target_name)]
        safe_pose = LEFT_SAFE_POSE if arm_name == "LEFT_ARM" else RIGHT_SAFE_POSE
        arm_obj = None
        records: list[dict[str, Any]] = []
        try:
            arm_obj = make_arm(arm_name)
            for i in range(repeats):
                started = time.time()
                record: dict[str, Any] = {
                    "repeat": i + 1,
                    "arm": arm_name,
                    "target": target_name,
                    "target_pose": pose_to_list(pose),
                    "safe_pose": pose_to_list(safe_pose),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                }
                log_event("HOVER_REPEAT_START", record)
                try:
                    record["start_qpos"] = jsonable(arm_obj.get_joint_angles())
                    record["start_pose"] = jsonable(arm_obj.get_pose())
                    target_return = move_to_pose(arm_obj, pose)
                    record["move_to_target_return"] = jsonable(target_return)
                    record["target_pose_readback"] = jsonable(arm_obj.get_pose())
                    record["target_qpos"] = jsonable(arm_obj.get_joint_angles())
                    safe_return = move_to_pose(arm_obj, safe_pose)
                    record["move_to_safe_return"] = jsonable(safe_return)
                    record["end_pose"] = jsonable(arm_obj.get_pose())
                    record["end_qpos"] = jsonable(arm_obj.get_joint_angles())
                    record["status"] = "OK"
                except Exception as exc:
                    record["status"] = "ERROR"
                    record["error"] = repr(exc)
                    if hasattr(arm_obj, "stop"):
                        try:
                            arm_obj.stop()
                        except Exception as stop_exc:
                            record["stop_error"] = repr(stop_exc)
                record["duration_s"] = time.time() - started
                records.append(record)
                log_event("HOVER_REPEAT_RESULT", record)
        finally:
            if arm_obj is not None and hasattr(arm_obj, "shutdown"):
                try:
                    arm_obj.shutdown()
                except Exception as exc:
                    log_event("ARM_SHUTDOWN_ERROR", {"arm": arm_name, "error": repr(exc)})

        ok_records = [r for r in records if r.get("status") == "OK"]
        qpos_samples = [r["target_qpos"] for r in ok_records if isinstance(r.get("target_qpos"), list)]
        duration_values = [float(r["duration_s"]) for r in records if isinstance(r.get("duration_s"), (int, float))]
        rep = {
            "arm": arm_name,
            "target": target_name,
            "records": records,
            "joint_stats": joint_stats(qpos_samples),
            "duration_stats": stats(duration_values),
        }
        rep["status"] = classify_repeatability(rep) if len(ok_records) == repeats else "FAIL"
        repeatability[f"{arm_name}:{target_name}"] = rep

    summary = merge_summary(reachability, repeatability)
    write_outputs(summary)
    log_event("HOVER_END", {"overall": summary["overall"]})
    return summary


def merge_summary(reachability: dict[str, Any], repeatability: dict[str, Any]) -> dict[str, Any]:
    strategy = strategy_status(reachability.get("reachability", []))
    unknowns = list(reachability.get("unknowns", []))
    if any(rep.get("status") != "PASS" for rep in repeatability.values()):
        unknowns.append("SOME_HOVER_REPEATABILITY_NOT_STABLE_OR_NOT_COMPLETE")

    overall = "PASS"
    if unknowns or strategy["recommended_strategy"].startswith("UNKNOWN"):
        overall = "CHECK"
    if any(rep.get("status") == "FAIL" for rep in repeatability.values()):
        overall = "FAIL"

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "overall": overall,
        "phase": "hover_motion",
        "source_coordinates": source_coordinates(),
        "reachability": reachability.get("reachability", []),
        "repeatability": repeatability,
        "single_right_feasible": strategy["single_right"] == "PASS",
        "single_left_feasible": strategy["single_left"] == "PASS",
        "dual_independent_feasible": strategy["dual_independent"] == "PASS",
        "handoff_required": strategy["handoff_required"],
        "strategy": strategy,
        "recommended_strategy": strategy["recommended_strategy"],
        "unknowns": unknowns,
    }


def write_outputs(summary: dict[str, Any]) -> None:
    ensure_dirs()
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(summary), encoding="utf-8")


def render_repeatability(summary: dict[str, Any], arm: str) -> list[str]:
    lines = ["| Target | Status | Repeats | Max Joint Range | Duration Mean | Duration Range |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for rep in summary.get("repeatability", {}).values():
        if rep.get("arm") != arm:
            continue
        js = rep.get("joint_stats", {})
        ds = rep.get("duration_stats", {})
        lines.append(
            "| {target} | {status} | {repeats} | {jr} | {dm} | {dr} |".format(
                target=rep.get("target"),
                status=rep.get("status"),
                repeats=len(rep.get("records", [])),
                jr=js.get("max_joint_range"),
                dm=ds.get("mean"),
                dr=ds.get("range"),
            )
        )
    if len(lines) == 2:
        lines.append("| None | NOT_RUN | 0 |  |  |  |")
    return lines


def render_joint_stats(summary: dict[str, Any]) -> list[str]:
    lines = ["| Arm | Target | Joint | Mean | Std | Max-Min |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for rep in summary.get("repeatability", {}).values():
        for row in rep.get("joint_stats", {}).get("per_joint", []):
            lines.append(
                f"| {rep.get('arm')} | {rep.get('target')} | {row.get('joint')} | {row.get('mean')} | {row.get('std')} | {row.get('range')} |"
            )
    if len(lines) == 2:
        lines.append("| None | None |  |  |  |  |")
    return lines


def markdown_report(summary: dict[str, Any]) -> str:
    strategy = summary["strategy"]
    lines = [
        "# Rabo Arm Strategy Evaluation Report",
        "",
        "## 1. Summary",
        "",
        f"Overall: {summary['overall']}",
        "",
        "## 2. Source Coordinates",
        "",
        "- Nut A/B/C: `agents/three_nut_expert/config.py:53-57`",
        "- Right arm base: `agents/arm_hand_demo/__init__.py:25-28` and `agents/three_nut_expert/config.py:59-60`",
        "- Right nut hover formula: `agents/arm_hand_demo/__init__.py:75-78`",
        "- Left staged place poses: `agents/three_nut_expert/config.py:93-97`",
        "- Explicit Box A/B/C target region coordinates: `UNKNOWN`",
        "- Left-arm world-to-base nut transform: `UNKNOWN`",
        "",
        "## 3. Reachability Matrix",
        "",
        *matrix_rows(summary["reachability"]),
        "",
        "## 4. Left Arm Hover Repeatability",
        "",
        *render_repeatability(summary, "LEFT_ARM"),
        "",
        "## 5. Right Arm Hover Repeatability",
        "",
        *render_repeatability(summary, "RIGHT_ARM"),
        "",
        "## 6. Joint Repeatability",
        "",
        *render_joint_stats(summary),
        "",
        "## 7. Motion Duration",
        "",
        "Duration statistics are included in sections 4 and 5.",
        "",
        "## 8. Candidate Strategies",
        "",
        "| Strategy | Feasible | Complexity | ACT Suitability | Reason |",
        "| --- | --- | --- | --- | --- |",
        f"| Single Right | {strategy['single_right']} | LOW | GOOD | Requires right arm to cover all nut and box hovers. |",
        f"| Single Left | {strategy['single_left']} | LOW | GOOD | Requires left arm to cover all nut and box hovers. |",
        f"| Dual Independent | {strategy['dual_independent']} | MEDIUM | GOOD | Requires at least one arm to cover each nut and matching box without handoff. |",
        f"| Right-to-left Handoff | {strategy['handoff_required']} | HIGH | POOR | Only required if no single-arm or independent assignment covers the workspace. |",
        "",
        "## 9. Recommended Strategy",
        "",
        f"FIRST CHOICE: `{strategy['recommended_strategy']}`",
        "",
        f"HANDOFF REQUIRED: `{strategy['handoff_required']}`",
        "",
        "## 10. Remaining Unknowns",
        "",
    ]
    if summary["unknowns"]:
        lines.extend(f"- {item}" for item in summary["unknowns"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## 11. Next Step",
            "",
            "Use the measured workspace strategy only. Do not start grasping, Recorder, or ACT in this phase.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hover-only motion repeatability test for PASS pose_check targets.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arms", help="Comma-separated arms, e.g. LEFT_ARM,RIGHT_ARM")
    parser.add_argument("--targets", help="Comma-separated exact target names, e.g. 'Nut B hover,Box B hover'")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    summary = run_hover_tests(args.repeats, parse_csv_set(args.arms), parse_csv_set(args.targets))
    print("========================================")
    print("RABO ARM STRATEGY EVALUATION")
    print("========================================")
    print()
    print(f"Single Right:\n{summary['strategy']['single_right']}")
    print()
    print(f"Single Left:\n{summary['strategy']['single_left']}")
    print()
    print(f"Dual Independent:\n{summary['strategy']['dual_independent']}")
    print()
    print(f"Handoff Required:\n{summary['strategy']['handoff_required']}")
    print()
    print(f"Recommended Strategy:\n{summary['strategy']['recommended_strategy']}")
    print()
    print(f"Report:\n{REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"JSON:\n{JSON_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Raw Log:\n{LOG_PATH.relative_to(PROJECT_ROOT)}")
    print("========================================")
    return 0 if summary["overall"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
