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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_workspace_v2"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_workspace_v2.log"
JSON_PATH = OUTPUT_DIR / "工作空间测试V2结果.json"
REPORT_PATH = OUTPUT_DIR / "工作空间测试V2报告.md"
COORD_JSON_PATH = PROJECT_ROOT / "outputs" / "arm_workspace_v2" / "坐标解析结果.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    Pose6,
)
from agents.three_nut_expert.expert import pose_to_list  # noqa: E402
from expert.transforms import transform_pose_base_to_world, transform_pose_world_to_base  # noqa: E402
from tools.resolve_workspace_coordinates import CONFIRMED_SCENE_UI, build_coordinate_db  # noqa: E402


@dataclass(frozen=True)
class ArmTarget:
    target: str
    key: str
    kind: str
    arm: str
    world_task_pose: list[float]
    pose: Pose6
    source: str
    confidence: str
    note: str
    regression_expected_pass: bool


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


def pose6_from_list(pose: list[float]) -> Pose6:
    return Pose6(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])


def coordinate_gate_passed(coordinates: dict[str, Any]) -> bool:
    validation = coordinates.get("validation", {}).get("right_arm_legacy", {})
    official_place = coordinates.get("official_place_verification", {})
    return (
        not coordinates.get("blocked")
        and validation.get("status") == "PASS"
        and validation.get("geometry_transform_validation") == "PASS"
        and validation.get("task_target_validation") == "PASS"
        and official_place.get("status") == "PASS"
    )


def base_pose(coordinates: dict[str, Any], side: str) -> list[float]:
    return list(coordinates["frames"][f"{side}_arm_base"]["world_pose"])


def nut_task_world_poses(coordinates: dict[str, Any]) -> dict[str, list[float]]:
    right_base = base_pose(coordinates, "right")
    rows = coordinates["validation"]["right_arm_legacy"]["rows"]
    return {row["nut"]: transform_pose_base_to_world(list(row["v2_final_target"]), right_base) for row in rows}


def official_place_world_poses(coordinates: dict[str, Any]) -> dict[str, list[float]]:
    places = coordinates["official_place_verification"]["places"]
    return {key: list(item["world_task_pose"]) for key, item in places.items()}


def build_arm_targets(coordinates: dict[str, Any]) -> list[ArmTarget]:
    targets: list[ArmTarget] = []
    bases = {
        "LEFT_ARM": base_pose(coordinates, "left"),
        "RIGHT_ARM": base_pose(coordinates, "right"),
    }
    nut_world = nut_task_world_poses(coordinates)
    place_world = official_place_world_poses(coordinates)

    for key in ("A", "B", "C"):
        for arm in ("RIGHT_ARM", "LEFT_ARM"):
            target_base = transform_pose_world_to_base(nut_world[key], bases[arm])
            targets.append(
                ArmTarget(
                    target=f"Nut {key} hover",
                    key=key,
                    kind="nut_hover",
                    arm=arm,
                    world_task_pose=nut_world[key],
                    pose=pose6_from_list(target_base),
                    source=(
                        "Nut task world pose derived from legacy right-arm final target after "
                        "GEOMETRY_TRANSFORM_VALIDATION and TASK_TARGET_VALIDATION"
                    ),
                    confidence="DERIVED_FROM_VALIDATED_LEGACY_TASK_TARGET",
                    note="World task pose is transformed into each arm's own base_link frame; numeric right target is not copied to left.",
                    regression_expected_pass=arm == "RIGHT_ARM",
                )
            )

    for key in ("A", "B", "C"):
        for arm in ("LEFT_ARM", "RIGHT_ARM"):
            target_base = transform_pose_world_to_base(place_world[key], bases[arm])
            targets.append(
                ArmTarget(
                    target=f"Official Place {key}",
                    key=key,
                    kind="official_place",
                    arm=arm,
                    world_task_pose=place_world[key],
                    pose=pose6_from_list(target_base),
                    source=(
                        "agents/three_nut_expert/config.py:93-97; "
                        "agents/three_nut_expert/expert.py:128,156"
                    ),
                    confidence="CONFIRMED_TASK_TARGETS",
                    note="Official Place is the task target used by left_arm.move_to; it is not a Box geometric center.",
                    regression_expected_pass=arm == "LEFT_ARM",
                )
            )

    return targets


def build_blocked_summary(coordinates: dict[str, Any]) -> dict[str, Any]:
    validation = coordinates.get("validation", {}).get("right_arm_legacy", {})
    official_place = coordinates.get("official_place_verification", {})
    unknowns = list(coordinates.get("unknowns", []))
    if validation.get("status") != "PASS":
        unknowns.append("LEGACY_RIGHT_ARM_CROSS_VALIDATION_NOT_PASS：Legacy 右臂坐标交叉验证未通过")
    if official_place.get("status") != "PASS":
        unknowns.append("OFFICIAL_PLACE_SOURCE_VERIFICATION_NOT_PASS：Official Place 源码确认未通过")
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "overall": "CHECK",
        "phase": "workspace_v2_coordinate_gate",
        "source_coordinates": source_coordinates(coordinates),
        "coordinate_gate": {
            "passed": False,
            "coordinate_json": str(COORD_JSON_PATH.relative_to(PROJECT_ROOT)),
            "legacy_right_arm_validation": validation.get("status", "UNKNOWN"),
            "legacy_geometry_transform_validation": validation.get("geometry_transform_validation", "UNKNOWN"),
            "legacy_task_target_validation": validation.get("task_target_validation", "UNKNOWN"),
            "official_place_source_verification": official_place.get("status", "UNKNOWN"),
            "unknowns": unknowns,
            "confirmed_scene_ui": coordinates.get("confirmed_scene_ui", CONFIRMED_SCENE_UI),
            "storage_box": coordinates.get("storage_box"),
            "box_model_search": coordinates.get("box_model_search"),
        },
        "reachability": [],
        "repeatability": {},
        "single_right_feasible": False,
        "single_left_feasible": False,
        "dual_independent_feasible": False,
        "handoff_required": "UNKNOWN",
        "strategy": {
            "single_right": "CHECK",
            "single_left": "CHECK",
            "dual_independent": "CHECK",
            "handoff_required": "UNKNOWN",
            "assignments": {"A": [], "B": [], "C": []},
            "recommended_strategy": "UNKNOWN_NEEDS_CONFIRMED_COORDINATES",
        },
        "recommended_strategy": "UNKNOWN_NEEDS_CONFIRMED_COORDINATES",
        "unknowns": unknowns,
    }


def make_arm(arm: str) -> Any:
    from rabo_robocap import LinkerArmA7

    return LinkerArmA7(robot_id=DEVICE_IDS[arm], mode="sim")


def call_pose_check(arm_obj: Any, pose: Pose6) -> Any:
    return arm_obj.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)


def run_reachability(skip_runtime: bool = False) -> dict[str, Any]:
    ensure_dirs()
    log_event("WORKSPACE_V2_START", {"skip_runtime": skip_runtime})
    coordinate_gate = build_coordinate_db(include_runtime_tf=False)
    if not coordinate_gate_passed(coordinate_gate):
        summary = build_blocked_summary(coordinate_gate)
        write_outputs(summary)
        log_event("WORKSPACE_V2_BLOCKED_BY_COORDINATES", summary["coordinate_gate"])
        return summary

    targets = build_arm_targets(coordinate_gate)
    results: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    runtime_error: str | None = None

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
            "key": target.key,
            "kind": target.kind,
            "arm": target.arm,
            "world_task_pose": target.world_task_pose,
            "base_link_target": pose_to_list(target.pose),
            "source": target.source,
            "confidence": target.confidence,
            "note": target.note,
            "regression_expected_pass": target.regression_expected_pass,
            "regression": "NOT_RUN",
            "result": "UNKNOWN",
            "reason": "",
            "raw": None,
        }
        if skip_runtime:
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
        if target.regression_expected_pass:
            if row["result"] == "PASS":
                row["regression"] = "PASS"
            elif row["result"] == "CHECK":
                row["regression"] = "CHECK"
            else:
                row["regression"] = "FAIL"
        else:
            row["regression"] = "N/A"
        results.append(row)
        log_event("WORKSPACE_V2_POSE_CHECK_RESULT", row)

    for arm_obj in arms.values():
        if hasattr(arm_obj, "shutdown"):
            try:
                arm_obj.shutdown()
            except Exception as exc:
                log_event("ARM_SHUTDOWN_ERROR", {"error": repr(exc)})

    summary = build_summary(results, runtime_error, coordinate_gate, skip_runtime)
    write_outputs(summary)
    log_event("WORKSPACE_V2_END", {"overall": summary["overall"], "workspace_matrix": summary["workspace_matrix_status"]})
    return summary


def _result_for(results: list[dict[str, Any]], arm: str, target: str) -> dict[str, Any] | None:
    return next((r for r in results if r["arm"] == arm and r["target"] == target), None)


def strategy_status(results: list[dict[str, Any]], allow_final: bool) -> dict[str, Any]:
    nut_targets = [f"Nut {k} hover" for k in ("A", "B", "C")]
    place_targets = [f"Official Place {k}" for k in ("A", "B", "C")]

    if not allow_final:
        return {
            "single_right": "无法判断",
            "single_left": "无法判断",
            "dual_independent": "无法判断",
            "handoff_required": "无法判断",
            "assignments": {"A": [], "B": [], "C": []},
            "recommended_strategy": "BLOCKED_BEFORE_FINAL_STRATEGY",
        }

    def arm_covers(arm: str) -> str:
        rows = [_result_for(results, arm, t) for t in nut_targets + place_targets]
        if all(row and row["result"] == "PASS" for row in rows):
            return "可行"
        return "不可行"

    single_right = arm_covers("RIGHT_ARM")
    single_left = arm_covers("LEFT_ARM")
    assignments: dict[str, list[str]] = {}

    for key in ("A", "B", "C"):
        choices = []
        for arm in ("LEFT_ARM", "RIGHT_ARM"):
            nut = _result_for(results, arm, f"Nut {key} hover")
            place = _result_for(results, arm, f"Official Place {key}")
            if nut and place and nut["result"] == "PASS" and place["result"] == "PASS":
                choices.append(arm)
        assignments[key] = choices

    if all(assignments[k] for k in ("A", "B", "C")):
        dual = "可行"
        handoff = "否"
    else:
        dual = "不可行"
        handoff = "是"

    if single_right == "可行":
        recommended = "SINGLE_RIGHT"
    elif single_left == "可行":
        recommended = "SINGLE_LEFT"
    elif dual == "可行":
        recommended = "DUAL_INDEPENDENT"
    elif handoff == "是":
        recommended = "HANDOFF_REQUIRED"
    else:
        recommended = "NO_GEOMETRIC_STRATEGY_FOUND"

    return {
        "single_right": single_right,
        "single_left": single_left,
        "dual_independent": dual,
        "handoff_required": handoff,
        "assignments": assignments,
        "recommended_strategy": recommended,
    }


def build_summary(
    results: list[dict[str, Any]], runtime_error: str | None, coordinates: dict[str, Any], skip_runtime: bool
) -> dict[str, Any]:
    pose_checks_complete = bool(results) and all(r["result"] in ("PASS", "FAIL") for r in results)
    regression_status = workspace_regression_status(results, skip_runtime)
    matrix_status = workspace_matrix_status(results, skip_runtime)
    validation = coordinates["validation"]["right_arm_legacy"]
    transform_validation = validation.get("geometry_transform_validation", "UNKNOWN")
    task_validation = validation.get("task_target_validation", "UNKNOWN")
    official_place_status = coordinates.get("official_place_verification", {}).get("status", "UNKNOWN")
    base_frame_status = base_frame_status_text(coordinates)
    allow_final = (
        transform_validation == "PASS"
        and task_validation == "PASS"
        and regression_status == "PASS"
        and matrix_status in ("PASS", "FAIL")
        and pose_checks_complete
    )
    strategy = strategy_status(results, allow_final)
    unknowns = []
    if runtime_error:
        unknowns.append(f"RUNTIME_INIT_FAILED: {runtime_error}")
    if skip_runtime:
        unknowns.append("SKIPPED_RUNTIME：未真实执行 12 个 pose_check")
    if regression_status == "FAIL":
        unknowns.append("WORKSPACE_V2_REGRESSION = FAIL")

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
        "source_coordinates": source_coordinates(coordinates),
        "base_frame_evidence": base_frame_status,
        "official_place_source_verification": official_place_status,
        "legacy_geometry_transform_validation": transform_validation,
        "legacy_task_target_validation": task_validation,
        "workspace_v2_regression": regression_status,
        "workspace_matrix_status": matrix_status,
        "coordinate_json": str(COORD_JSON_PATH.relative_to(PROJECT_ROOT)),
        "legacy_validation": validation,
        "reachability": results,
        "repeatability": {},
        "single_right_feasible": strategy["single_right"] == "可行",
        "single_left_feasible": strategy["single_left"] == "可行",
        "dual_independent_feasible": strategy["dual_independent"] == "可行",
        "handoff_required": strategy["handoff_required"],
        "strategy": strategy,
        "recommended_strategy": strategy["recommended_strategy"],
        "unknowns": unknowns,
    }


def base_frame_status_text(coordinates: dict[str, Any]) -> str:
    frames = coordinates.get("frames", {})
    left = frames.get("left_arm_base", {})
    right = frames.get("right_arm_base", {})
    if "CONFIRMED" in str(left.get("confidence", "")) and "CONFIRMED" in str(right.get("confidence", "")):
        return "PASS"
    return "FAIL"


def workspace_regression_status(results: list[dict[str, Any]], skip_runtime: bool) -> str:
    regression_rows = [r for r in results if r.get("regression_expected_pass")]
    if not regression_rows:
        return "BLOCKED"
    if skip_runtime or any(r["result"] == "CHECK" for r in regression_rows):
        return "CHECK"
    if all(r["result"] == "PASS" for r in regression_rows):
        return "PASS"
    return "FAIL"


def workspace_matrix_status(results: list[dict[str, Any]], skip_runtime: bool) -> str:
    if len(results) != 12:
        return "BLOCKED"
    if skip_runtime or any(r["result"] == "CHECK" for r in results):
        return "BLOCKED"
    if all(r["result"] == "PASS" for r in results):
        return "PASS"
    return "FAIL"


def source_coordinates(coordinates: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_facts": {
            "nuts": coordinates.get("nuts", {}),
            "official_place": coordinates.get("official_place_verification", {}),
            "a7_frame": coordinates.get("official_place_verification", {}).get("A7_FRAME_EVIDENCE"),
        },
        "rabo_ui_facts": {
            "frames": coordinates.get("frames", {}),
            "storage_box": coordinates.get("storage_box", CONFIRMED_SCENE_UI["storage_box"]),
        },
        "mathematical_derivation": {
            "box_geometric_center_status": coordinates.get("box_geometric_center_status"),
            "legacy_validation": coordinates.get("validation", {}).get("right_arm_legacy", {}),
        },
    }


def matrix_rows(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Target | Left Arm | Right Arm | Left Reason | Right Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for target in [f"Nut {k} hover" for k in ("A", "B", "C")] + [f"Official Place {k}" for k in ("A", "B", "C")]:
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
    if summary.get("phase") == "workspace_v2_coordinate_gate":
        return markdown_blocked_report(summary)

    strategy = summary["strategy"]
    lines = [
        "# Rabo 工作空间测试 V2 报告",
        "",
        "## 1. 总结",
        "",
        f"Overall: {summary['overall']}",
        f"Base frame evidence：{summary['base_frame_evidence']}",
        f"Official Place source verification：{summary['official_place_source_verification']}",
        f"Legacy geometry transform validation：{summary['legacy_geometry_transform_validation']}",
        f"Legacy task target validation：{summary['legacy_task_target_validation']}",
        f"Workspace V2 regression：{summary['workspace_v2_regression']}",
        f"完整 6×2 Matrix：{summary['workspace_matrix_status']}",
        "",
        "本报告只覆盖 Workspace V2：统一坐标系、legacy 门禁、12 格 pose_check 和几何策略判断；未运行 move_to 稳定性实验。",
        "",
        "## 2. 官方来源事实",
        "",
        "- Nut A/B/C: `agents/three_nut_expert/config.py:53-57`",
        "- Legacy right nut target formula: `docs/legacy/arm_hand_demo_legacy_snapshot.py:75-78`",
        "- Official Place A/B/C: `agents/three_nut_expert/config.py:93-97`",
        "- Official Place enters `left_arm.move_to`: `agents/three_nut_expert/expert.py:128,156`",
        "- A7 API frame: LinkerArmA7 `move_to/get_pose/pose_check` use this node's `base_link` frame.",
        "- `CONFIRMED_TASK_TARGETS` means task targets, not Box geometric centers.",
        "",
        "## 3. Rabo UI 实测事实",
        "",
        f"- LEFT base_link world pose: `{summary['source_coordinates']['rabo_ui_facts']['frames']['left_arm_base']['world_pose']}`",
        f"- RIGHT base_link world pose: `{summary['source_coordinates']['rabo_ui_facts']['frames']['right_arm_base']['world_pose']}`",
        "- TF / omni.usd / root-base runtime check is optional cross-validation, not a hard blocker in this round.",
        "",
        "## 4. 数学推导与工程门限",
        "",
        "- Nut world pose -> pure world->right_base SE(3) -> legacy strategy offset -> V2 final target.",
        "- Official Place left_base target -> world task pose -> right_base equivalent target.",
        "- Position error threshold: <= 0.01 m.",
        "- Orientation error threshold: <= 0.02 rad.",
        "- Thresholds are engineering validation gates, not official specifications.",
        "",
        "### Legacy 右臂验证明细",
        "",
        "| Nut | Position error | Orientation error | Result |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["legacy_validation"].get("rows", []):
        lines.append(
            f"| {row['nut']} | {row['position_error']:.6f} | {row['orientation_error']:.6f} | {row['result']} |"
        )
    lines.extend(
        [
        "",
        "## 5. 完整 6×2 工作空间矩阵",
        "",
        *matrix_rows(summary["reachability"]),
        "",
        "## 6. pose_check 实验结果明细",
        "",
    ]
    )
    for row in summary["reachability"]:
        lines.extend(
            [
                f"### {row['target']} / {row['arm']}",
                "",
                f"- 目标来源：{row['source']}",
                f"- world task pose：`{row['world_task_pose']}`",
                f"- base_link target：`{row['base_link_target']}`",
                f"- pose_check：{row['result']}",
                f"- raw message：`{row['raw']}`",
                f"- confidence：{row['confidence']}",
                f"- regression：{row['regression']}",
                "",
            ]
        )
    lines.extend(
        [
        "## 7. 最终策略判断",
        "",
        f"- 单右臂：{strategy['single_right']}",
        f"- 单左臂：{strategy['single_left']}",
        f"- 双臂独立：{strategy['dual_independent']}",
        f"- handoff 几何必要性：{strategy['handoff_required']}",
        f"- 首选策略：`{strategy['recommended_strategy']}`",
        "",
        "## 8. 剩余未知项",
        "",
        ]
    )
    if summary["unknowns"]:
        lines.extend(f"- {item}" for item in summary["unknowns"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## 9. 安全状态",
            "",
            "未调用 move_to、move_joints、home、SetEntityPose、clench、grasp_force、抓取、handoff、reset、Recorder、Replay 或 ACT。",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_blocked_report(summary: dict[str, Any]) -> str:
    gate = summary["coordinate_gate"]
    lines = [
        "# Rabo 工作空间测试 V2 报告",
        "",
        "## 1. 总结",
        "",
        "Overall: CHECK",
        "",
        "结论：工作空间测试 V2 尚未完成。当前停在坐标门禁阶段，未运行完整 6×2 pose_check。",
        "",
        "## 2. 坐标来源",
        "",
        "- Nut A/B/C：`agents/three_nut_expert/config.py:53-57`",
        "- Legacy 右臂 base XY：`agents/arm_hand_demo/__init__.py:25-28` 和 `agents/three_nut_expert/config.py:59-60`",
        "- Legacy 右臂 world → arm target 公式：`agents/arm_hand_demo/__init__.py:75-78`",
        f"- 已确认左臂 base_link world pose：`{CONFIRMED_SCENE_UI['left_robot_root']['world_pose']}`",
        f"- 已确认右臂 base_link world pose：`{CONFIRMED_SCENE_UI['right_robot_root']['world_pose']}`",
        f"- 已确认收纳盒整体 root UI world pose：`{CONFIRMED_SCENE_UI['storage_box']['world_pose']}`",
        f"- 收纳盒 ID：`{CONFIRMED_SCENE_UI['storage_box']['thing_id']}`",
        "- 当前左臂 staged place pose：`agents/three_nut_expert/config.py:93-97`，不能当作 Box A/B/C 真实世界坐标",
        f"- 坐标解析 JSON：`{gate['coordinate_json']}`",
        "",
        "## 3. 可达性矩阵",
        "",
        "未生成完整 6×2 矩阵。原因：base frame、Official Place 或 Legacy 右臂验证门禁未通过。",
        "",
        "## 4. 左臂 Hover 重复性",
        "",
        "未运行。本轮禁止运动机器人。",
        "",
        "## 5. 右臂 Hover 重复性",
        "",
        "未运行。本轮禁止运动机器人。",
        "",
        "## 6. 关节重复性",
        "",
        "未运行稳定性实验，无数据。",
        "",
        "## 7. 运动耗时",
        "",
        "未运行运动测试，无数据。",
        "",
        "## 8. 候选策略",
        "",
        "| 策略 | 可行性 | 复杂度 | ACT 适配性 | 原因 |",
        "| --- | --- | --- | --- | --- |",
        "| 单右臂 | CHECK | LOW | GOOD | 需要完整 6×2 pose_check 后判断。 |",
        "| 单左臂 | CHECK | LOW | GOOD | 需要完整 6×2 pose_check 后判断。 |",
        "| 双臂独立分工 | CHECK | MEDIUM | GOOD | 需要完整 6×2 pose_check 后才能判断。 |",
        "| 右手到左手交接 | UNKNOWN | HIGH | POOR | 只有前面方案失败后才允许判断是否需要。 |",
        "",
        "## 9. 推荐策略",
        "",
        "第一选择：`UNKNOWN_NEEDS_CONFIRMED_COORDINATES`",
        "",
        "是否需要 handoff：`UNKNOWN`",
        "",
        "## 10. 剩余未知项",
        "",
    ]
    lines.extend(f"- {item}" for item in gate["unknowns"])
    lines.extend(
        [
            "",
            "## 11. 下一步",
            "",
            "先补齐以下信息：",
            "",
            "1. 修正门禁中失败的 base frame、Official Place source 或 Legacy validation 项。",
            "2. 不把 Box A/B/C geometry center 作为硬 blocker。",
            "",
            "补齐后先通过 Legacy 右臂坐标交叉验证，再运行完整 6×2 pose_check。不要提前运行 hover motion、抓取、handoff、Recorder 或 ACT。",
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
    print("Workspace V2 最终验证")
    print("========================================")
    print()
    print(f"Base frame evidence：\n{summary.get('base_frame_evidence', 'FAIL')}")
    print()
    print(f"Official Place source verification：\n{summary.get('official_place_source_verification', 'FAIL')}")
    print()
    print(f"Legacy geometry transform validation：\n{summary.get('legacy_geometry_transform_validation', 'FAIL')}")
    print()
    print(f"Legacy task target validation：\n{summary.get('legacy_task_target_validation', 'FAIL')}")
    print()
    print(f"Workspace V2 regression：\n{summary.get('workspace_v2_regression', 'FAIL')}")
    print()
    print(f"完整 6×2 Matrix：\n{summary.get('workspace_matrix_status', 'BLOCKED')}")
    print()
    print(f"单右臂：\n{summary['strategy']['single_right']}")
    print()
    print(f"单左臂：\n{summary['strategy']['single_left']}")
    print()
    print(f"双臂独立：\n{summary['strategy']['dual_independent']}")
    print()
    print(f"handoff 几何必要性：\n{summary['strategy']['handoff_required']}")
    print()
    print(f"首选策略：\n{summary['strategy']['recommended_strategy']}")
    print()
    allow_move_to = (
        summary.get("legacy_geometry_transform_validation") == "PASS"
        and summary.get("legacy_task_target_validation") == "PASS"
        and summary.get("workspace_v2_regression") == "PASS"
        and summary.get("workspace_matrix_status") in ("PASS", "FAIL")
    )
    print(f"是否允许进入 move_to 稳定性测试：\n{'YES' if allow_move_to else 'NO'}")
    if allow_move_to:
        print()
        print("下一阶段建议代表目标：优先测试首选策略中最小覆盖路径的一个 PASS 目标；本轮不执行。")
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
