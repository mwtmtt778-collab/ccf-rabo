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
COORD_JSON_PATH = PROJECT_ROOT / "outputs" / "arm_workspace_v2" / "坐标解析结果.json"

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
from tools.resolve_workspace_coordinates import CONFIRMED_SCENE_UI, build_coordinate_db  # noqa: E402


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


def load_or_build_coordinate_gate(include_runtime_tf: bool) -> dict[str, Any]:
    if COORD_JSON_PATH.exists():
        try:
            return json.loads(COORD_JSON_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "blocked": True,
                "unknowns": [f"坐标解析结果读取失败：{repr(exc)}"],
                "validation": {"right_arm_legacy": {"status": "BLOCKED"}},
            }
    return build_coordinate_db(include_runtime_tf=include_runtime_tf)


def coordinate_gate_passed(coordinates: dict[str, Any]) -> bool:
    validation = coordinates.get("validation", {}).get("right_arm_legacy", {})
    return not coordinates.get("blocked") and validation.get("status") == "PASS"


def build_blocked_summary(coordinates: dict[str, Any]) -> dict[str, Any]:
    validation = coordinates.get("validation", {}).get("right_arm_legacy", {})
    unknowns = list(coordinates.get("unknowns", []))
    if validation.get("status") != "PASS":
        unknowns.append("LEGACY_RIGHT_ARM_CROSS_VALIDATION_NOT_PASS：Legacy 右臂坐标交叉验证未通过")
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "overall": "CHECK",
        "phase": "workspace_v2_coordinate_gate",
        "source_coordinates": source_coordinates(),
        "coordinate_gate": {
            "passed": False,
            "coordinate_json": str(COORD_JSON_PATH.relative_to(PROJECT_ROOT)),
            "legacy_right_arm_validation": validation.get("status", "UNKNOWN"),
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
    log_event("REACHABILITY_START", {"skip_runtime": skip_runtime})
    coordinate_gate = load_or_build_coordinate_gate(include_runtime_tf=not skip_runtime)
    if not coordinate_gate_passed(coordinate_gate):
        summary = build_blocked_summary(coordinate_gate)
        write_outputs(summary)
        log_event("REACHABILITY_BLOCKED_BY_COORDINATES", summary["coordinate_gate"])
        return summary

    targets = build_arm_targets()
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
                "robot_root_world_pose": CONFIRMED_SCENE_UI["left_robot_root"]["world_pose"],
                "source": "Rabo scene UI, user supplied",
                "status": "ROOT_TO_BASE_LINK_TRANSFORM_UNCONFIRMED",
            },
            "right_robot_root_pose": {
                "value": CONFIRMED_SCENE_UI["right_robot_root"]["world_pose"],
                "source": "Rabo scene UI, user supplied",
                "status": "ROOT_TO_BASE_LINK_TRANSFORM_UNCONFIRMED",
            },
        },
        "storage_box": CONFIRMED_SCENE_UI["storage_box"],
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
            "box_A": "BLOCKED_BOX_ABC_CENTER_POSE",
            "box_B": "BLOCKED_BOX_ABC_CENTER_POSE",
            "box_C": "BLOCKED_BOX_ABC_CENTER_POSE",
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
    if summary.get("phase") == "workspace_v2_coordinate_gate":
        return markdown_blocked_report(summary)

    strategy = summary["strategy"]
    lines = [
        "# Rabo 工作空间测试 V2 报告",
        "",
        "## 1. 总结",
        "",
        f"Overall: {summary['overall']}",
        "",
        "本报告由工作空间可达性阶段生成。只有坐标系和 Box A/B/C 真实坐标已确认，并且 Legacy 右臂坐标交叉验证通过后，才允许运行完整 6×2 pose_check。",
        "",
        "## 2. 坐标来源",
        "",
        "- Nut A/B/C: `agents/three_nut_expert/config.py:53-57`",
        "- Right arm base: `agents/arm_hand_demo/__init__.py:25-28` and `agents/three_nut_expert/config.py:59-60`",
        "- Right nut hover formula: `agents/arm_hand_demo/__init__.py:75-78`",
        "- Left staged place poses: `agents/three_nut_expert/config.py:93-97`",
        "- Box A/B/C 真实目标区坐标：`UNKNOWN`",
        "- 左臂 world → base_link 变换：`UNKNOWN`",
        "",
        "## 3. 可达性矩阵",
        "",
        *matrix_rows(summary["reachability"]),
        "",
        "## 4. 左臂 Hover 重复性",
        "",
        "本轮未运行。工作空间测试 V2 不允许运动机器人。",
        "",
        "## 5. 右臂 Hover 重复性",
        "",
        "本轮未运行。工作空间测试 V2 不允许运动机器人。",
        "",
        "## 6. 关节重复性",
        "",
        "本轮未运行稳定性实验，因此无关节重复性数据。",
        "",
        "## 7. 运动耗时",
        "",
        "本轮未运行运动测试，因此无运动耗时数据。",
        "",
        "## 8. 候选策略",
        "",
        "| 策略 | 可行性 | 复杂度 | ACT 适配性 | 原因 |",
        "| --- | --- | --- | --- | --- |",
        f"| 单右臂 | {strategy['single_right']} | LOW | GOOD | 需要右臂同时覆盖全部螺母 hover 和 Box hover。 |",
        f"| 单左臂 | {strategy['single_left']} | LOW | GOOD | 需要左臂同时覆盖全部螺母 hover 和 Box hover。 |",
        f"| 双臂独立分工 | {strategy['dual_independent']} | MEDIUM | GOOD | 需要每个螺母至少有一只手能同时覆盖抓取区和对应放置区。 |",
        f"| 右手到左手交接 | {strategy['handoff_required']} | HIGH | POOR | 只有单臂或双臂独立分工都无法覆盖工作空间时才需要。 |",
        "",
        "## 9. 推荐策略",
        "",
        f"第一选择：`{strategy['recommended_strategy']}`",
        "",
        f"是否需要 handoff：`{strategy['handoff_required']}`",
        "",
        "## 10. 剩余未知项",
        "",
    ]
    if summary["unknowns"]:
        lines.extend(f"- {item}" for item in summary["unknowns"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## 11. 下一步",
            "",
            "先补齐坐标系和 Box A/B/C 真实坐标，通过 Legacy 右臂坐标交叉验证以后，再运行完整 6×2 pose_check。本阶段不要运行稳定性实验、抓取、Recorder 或 ACT。",
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
        f"- 已确认左机器人 root UI world pose：`{CONFIRMED_SCENE_UI['left_robot_root']['world_pose']}`，但 root→base_link 未确认",
        f"- 已确认右机器人 root UI world pose：`{CONFIRMED_SCENE_UI['right_robot_root']['world_pose']}`，但 root→base_link 未确认",
        f"- 已确认收纳盒整体 root UI world pose：`{CONFIRMED_SCENE_UI['storage_box']['world_pose']}`",
        f"- 收纳盒 ID：`{CONFIRMED_SCENE_UI['storage_box']['thing_id']}`",
        "- 当前左臂 staged place pose：`agents/three_nut_expert/config.py:93-97`，不能当作 Box A/B/C 真实世界坐标",
        f"- 坐标解析 JSON：`{gate['coordinate_json']}`",
        "",
        "## 3. 可达性矩阵",
        "",
        "未生成完整 6×2 矩阵。原因：坐标系和 Box A/B/C 真实坐标尚未补齐，Legacy 右臂坐标交叉验证未通过。",
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
        "| 单右臂 | CHECK | LOW | GOOD | Box A/B/C 真实坐标和完整右臂目标未确认。 |",
        "| 单左臂 | CHECK | LOW | GOOD | 左臂 base_link 世界位姿和螺母 hover 目标未确认。 |",
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
            "1. 左机器人 root -> left base_link 的固定变换，或直接确认 left base_link world pose。",
            "2. 右机器人 root -> right base_link 的固定变换，或直接确认 right base_link world pose。",
            "3. Box A/B/C 三个格子中心的真实 world pose，或可严格计算三格中心的模型/尺寸/局部坐标。",
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
