#!/usr/bin/env python3
"""Evaluate dual-arm reachability with pose_check only.

This tool is intentionally read-only for robot motion. It never calls move_to,
move_joints, hand commands, SetEntityPose, or grasp APIs.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_strategy"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_strategy.log"
JSON_PATH = OUTPUT_DIR / "arm_strategy_summary.json"
REPORT_PATH = OUTPUT_DIR / "ARM_STRATEGY_EVALUATION_REPORT.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    GRASP_TARGET_OFFSETS,
    LEFT_PLACE_POSES,
    NUT_SPECS,
    RIGHT_ARM_BASE_XY,
    Pose6,
)
from agents.three_nut_expert.expert import compute_right_grasp_pose, pose_to_list  # noqa: E402


@dataclass(frozen=True)
class ArmTarget:
    target: str
    kind: str
    arm: str
    pose: Pose6 | None
    source: str
    confidence: str
    note: str


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
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return repr(value)


def normalize_pose_check_result(raw: Any) -> tuple[str, str, Any]:
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


def build_arm_targets() -> list[ArmTarget]:
    targets: list[ArmTarget] = []

    for key, spec in NUT_SPECS.items():
        targets.append(
            ArmTarget(
                target=f"Nut {key} hover",
                kind="nut_hover",
                arm="RIGHT_ARM",
                pose=compute_right_grasp_pose(spec.nominal_pose),
                source=(
                    "agents/three_nut_expert/config.py:53-72 and "
                    "agents/three_nut_expert/expert.py:76-85"
                ),
                confidence="CONFIRMED_FOR_RIGHT_ARM_FORMULA",
                note="Right-arm hover uses the legacy world-to-right-arm formula.",
            )
        )
        targets.append(
            ArmTarget(
                target=f"Nut {key} hover",
                kind="nut_hover",
                arm="LEFT_ARM",
                pose=None,
                source="UNKNOWN",
                confidence="UNKNOWN",
                note="No confirmed world-to-left-arm transform for nut hover was found.",
            )
        )

    for key, pose in LEFT_PLACE_POSES.items():
        targets.append(
            ArmTarget(
                target=f"Box {key} hover",
                kind="box_hover",
                arm="LEFT_ARM",
                pose=pose,
                source="agents/three_nut_expert/config.py:93-97",
                confidence="INFERRED_STAGED_TARGET",
                note="Existing left-arm place pose; target-box coordinate is not independently confirmed.",
            )
        )
        targets.append(
            ArmTarget(
                target=f"Box {key} hover",
                kind="box_hover",
                arm="RIGHT_ARM",
                pose=None,
                source="UNKNOWN",
                confidence="UNKNOWN",
                note="No confirmed right-arm base-frame target-box pose was found.",
            )
        )

    return targets


def make_arm(arm: str) -> Any:
    from rabo_robocap import LinkerArmA7

    return LinkerArmA7(robot_id=DEVICE_IDS[arm], mode="sim")


def call_pose_check(arm_obj: Any, pose: Pose6) -> Any:
    return arm_obj.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)


def run_reachability(skip_runtime: bool = False) -> dict[str, Any]:
    ensure_dirs()
    targets = build_arm_targets()
    results: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    runtime_error: str | None = None

    log_event("REACHABILITY_START", {"skip_runtime": skip_runtime})
    if not skip_runtime:
        try:
            for arm in ("LEFT_ARM", "RIGHT_ARM"):
                arms[arm] = make_arm(arm)
        except Exception as exc:
            runtime_error = repr(exc)
            log_event("RUNTIME_INIT_ERROR", {"error": runtime_error})

    for target in targets:
        row = {
            "target": target.target,
            "kind": target.kind,
            "arm": target.arm,
            "pose": pose_to_list(target.pose) if target.pose else None,
            "source": target.source,
            "confidence": target.confidence,
            "note": target.note,
            "result": "UNKNOWN",
            "reason": "",
            "raw": None,
        }
        if target.pose is None:
            row["result"] = "UNKNOWN"
            row["reason"] = "BLOCKED_MISSING_CONFIRMED_ARM_FRAME_TARGET"
        elif skip_runtime:
            row["result"] = "CHECK"
            row["reason"] = "SKIPPED_RUNTIME"
        elif runtime_error:
            row["result"] = "CHECK"
            row["reason"] = f"RUNTIME_INIT_FAILED: {runtime_error}"
        else:
            started = time.time()
            try:
                raw = call_pose_check(arms[target.arm], target.pose)
                status, reason, value = normalize_pose_check_result(raw)
                row.update(
                    {
                        "result": status,
                        "reason": reason,
                        "raw": value,
                        "duration_s": time.time() - started,
                    }
                )
            except Exception as exc:
                row["result"] = "FAIL"
                row["reason"] = repr(exc)
                row["duration_s"] = time.time() - started
        results.append(row)
        log_event("POSE_CHECK_RESULT", row)

    for arm_obj in arms.values():
        if hasattr(arm_obj, "shutdown"):
            try:
                arm_obj.shutdown()
            except Exception as exc:
                log_event("ARM_SHUTDOWN_ERROR", {"error": repr(exc)})

    summary = build_summary(results, runtime_error)
    write_outputs(summary)
    log_event("REACHABILITY_END", {"overall": summary["overall"]})
    return summary


def _result_for(results: list[dict[str, Any]], arm: str, target: str) -> dict[str, Any] | None:
    return next((r for r in results if r["arm"] == arm and r["target"] == target), None)


def strategy_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    nut_targets = [f"Nut {k} hover" for k in ("A", "B", "C")]
    box_targets = [f"Box {k} hover" for k in ("A", "B", "C")]

    def arm_covers(arm: str) -> str:
        rows = [_result_for(results, arm, t) for t in nut_targets + box_targets]
        if all(row and row["result"] == "PASS" for row in rows):
            return "PASS"
        if any(row is None or row["result"] == "UNKNOWN" for row in rows):
            return "CHECK"
        return "FAIL"

    single_right = arm_covers("RIGHT_ARM")
    single_left = arm_covers("LEFT_ARM")
    assignments: dict[str, list[str]] = {}
    unknown_assignment = False

    for key in ("A", "B", "C"):
        choices = []
        for arm in ("LEFT_ARM", "RIGHT_ARM"):
            nut = _result_for(results, arm, f"Nut {key} hover")
            box = _result_for(results, arm, f"Box {key} hover")
            if nut and box and nut["result"] == "PASS" and box["result"] == "PASS":
                choices.append(arm)
            elif nut is None or box is None or nut["result"] == "UNKNOWN" or box["result"] == "UNKNOWN":
                unknown_assignment = True
        assignments[key] = choices

    if all(assignments[k] for k in ("A", "B", "C")):
        dual = "PASS"
        handoff = "NO"
    elif unknown_assignment:
        dual = "CHECK"
        handoff = "UNKNOWN"
    else:
        dual = "FAIL"
        handoff = "YES"

    if single_right == "PASS":
        recommended = "SINGLE_RIGHT"
    elif single_left == "PASS":
        recommended = "SINGLE_LEFT"
    elif dual == "PASS":
        recommended = "DUAL_INDEPENDENT"
    elif handoff == "YES":
        recommended = "HANDOFF_REQUIRED"
    else:
        recommended = "UNKNOWN_NEEDS_COORDINATES_AND_RUNTIME_DATA"

    return {
        "single_right": single_right,
        "single_left": single_left,
        "dual_independent": dual,
        "handoff_required": handoff,
        "assignments": assignments,
        "recommended_strategy": recommended,
    }


def build_summary(results: list[dict[str, Any]], runtime_error: str | None) -> dict[str, Any]:
    strategy = strategy_status(results)
    unknowns = []
    if runtime_error:
        unknowns.append(f"RUNTIME_INIT_FAILED: {runtime_error}")
    if any(r["result"] == "UNKNOWN" for r in results):
        unknowns.append("BLOCKED_MISSING_CONFIRMED_ARM_FRAME_TARGET")
    if any(r["confidence"].startswith("INFERRED") for r in results):
        unknowns.append("BOX_TARGET_COORDINATES_ARE_INFERRED_FROM_EXISTING_PLACE_POSES")

    overall = "PASS"
    if unknowns or any(r["result"] == "CHECK" for r in results):
        overall = "CHECK"
    if any(r["result"] == "FAIL" for r in results):
        overall = "FAIL"

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "overall": overall,
        "phase": "reachability",
        "source_coordinates": source_coordinates(),
        "reachability": results,
        "repeatability": {},
        "single_right_feasible": strategy["single_right"] == "PASS",
        "single_left_feasible": strategy["single_left"] == "PASS",
        "dual_independent_feasible": strategy["dual_independent"] == "PASS",
        "handoff_required": strategy["handoff_required"],
        "strategy": strategy,
        "recommended_strategy": strategy["recommended_strategy"],
        "unknowns": unknowns,
    }


def source_coordinates() -> dict[str, Any]:
    return {
        "nuts": {
            key: {
                "thing_id": spec.thing_id,
                "nominal_pose": pose_to_list(spec.nominal_pose),
                "source": "agents/three_nut_expert/config.py:53-57",
            }
            for key, spec in NUT_SPECS.items()
        },
        "arm_base": {
            "right_arm_base_xy": {
                "value": list(RIGHT_ARM_BASE_XY),
                "source": "agents/arm_hand_demo/__init__.py:25-28 and agents/three_nut_expert/config.py:59-60",
            },
            "left_arm_base_xy": {
                "value": None,
                "source": "UNKNOWN",
                "status": "BLOCKED_LEFT_ARM_BASE_NOT_FOUND",
            },
        },
        "hover": {
            "right_nut_hover": {
                "offsets": GRASP_TARGET_OFFSETS,
                "source": "agents/arm_hand_demo/__init__.py:75-78 and agents/three_nut_expert/config.py:62-72",
            },
            "left_box_hover": {
                "poses": {key: pose_to_list(pose) for key, pose in LEFT_PLACE_POSES.items()},
                "source": "agents/three_nut_expert/config.py:93-97",
                "status": "INFERRED_STAGED_TARGET",
            },
        },
        "target_regions": {
            "box_A": "UNKNOWN_EXPLICIT_REGION_NOT_FOUND",
            "box_B": "UNKNOWN_EXPLICIT_REGION_NOT_FOUND",
            "box_C": "UNKNOWN_EXPLICIT_REGION_NOT_FOUND",
        },
    }


def matrix_rows(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Target | Left Arm | Right Arm | Left Reason | Right Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for target in [f"Nut {k} hover" for k in ("A", "B", "C")] + [f"Box {k} hover" for k in ("A", "B", "C")]:
        left = _result_for(results, "LEFT_ARM", target) or {}
        right = _result_for(results, "RIGHT_ARM", target) or {}
        lines.append(
            "| {target} | {left_result} | {right_result} | {left_reason} | {right_reason} |".format(
                target=target,
                left_result=left.get("result", "UNKNOWN"),
                right_result=right.get("result", "UNKNOWN"),
                left_reason=str(left.get("reason", "")),
                right_reason=str(right.get("reason", "")),
            )
        )
    return lines


def write_outputs(summary: dict[str, Any]) -> None:
    ensure_dirs()
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(summary), encoding="utf-8")


def markdown_report(summary: dict[str, Any]) -> str:
    strategy = summary["strategy"]
    lines = [
        "# Rabo Arm Strategy Evaluation Report",
        "",
        "## 1. Summary",
        "",
        f"Overall: {summary['overall']}",
        "",
        "This report is generated by the reachability phase. Hover motion repeatability is intentionally not run until reachability data is complete enough to allow it.",
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
        "Not run in this phase. It must be run with `tools/test_dual_arm_hover_motion.py` only after pose_check data is available for the selected targets.",
        "",
        "## 5. Right Arm Hover Repeatability",
        "",
        "Not run in this phase. It must be run with `tools/test_dual_arm_hover_motion.py` only after pose_check data is available for the selected targets.",
        "",
        "## 6. Joint Repeatability",
        "",
        "Not available until hover motion repeatability is executed.",
        "",
        "## 7. Motion Duration",
        "",
        "Not available until hover motion repeatability is executed.",
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
            "Run reachability in the Rabo runtime. If unknown coordinates remain, do not run hover motion for those targets. Do not start grasping, Recorder, or ACT in this phase.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only dual-arm pose_check reachability test.")
    parser.add_argument("--skip-runtime", action="store_true", help="Generate coordinate/report scaffold without importing Rabo runtime.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_reachability(skip_runtime=args.skip_runtime)
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
