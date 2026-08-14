#!/usr/bin/env python3
"""实验1：双臂工作空间 pose_check 测试。

本脚本只测试几何/关节可达性，不调用 move_to、move_joints、home、
SetEntityPose、clench、grasp_force，也不进行抓取或 handoff。
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_workspace_test"
REPORT_PATH = OUTPUT_DIR / "工作空间测试报告.md"
JSON_PATH = OUTPUT_DIR / "工作空间测试结果.json"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_workspace_test.log"

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
class WorkspaceTarget:
    name: str
    kind: str
    arm: str
    pose: Pose6 | None
    coordinate_source: str
    transform_source: str
    confidence: str
    note: str


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def log_event(event: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    line = {"time": now_text(), "event": event, **payload}
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
    if isinstance(value, list) and value and isinstance(value[0], bool):
        return ("PASS" if value[0] else "FAIL", str(value[1] if len(value) > 1 else ""), value)
    text = str(value)
    low = text.lower()
    if "out_of_workspace" in low or "joint_limit" in low or "false" in low or "fail" in low:
        return "FAIL", text, value
    if "reachable" in low or "true" in low or "success" in low:
        return "PASS", text, value
    return "CHECK", text, value


def build_targets() -> list[WorkspaceTarget]:
    targets: list[WorkspaceTarget] = []

    for key, spec in NUT_SPECS.items():
        targets.append(
            WorkspaceTarget(
                name=f"Nut {key} 抓取上方安全点",
                kind="nut_hover",
                arm="RIGHT_ARM",
                pose=compute_right_grasp_pose(spec.nominal_pose),
                coordinate_source="agents/three_nut_expert/config.py:53-57",
                transform_source="agents/arm_hand_demo/__init__.py:75-78；agents/three_nut_expert/expert.py:76-85",
                confidence="CONFIRMED",
                note="右臂抓取点沿用官方 Demo 的 world -> right_arm base_link 公式。",
            )
        )
        targets.append(
            WorkspaceTarget(
                name=f"Nut {key} 抓取上方安全点",
                kind="nut_hover",
                arm="LEFT_ARM",
                pose=None,
                coordinate_source="agents/three_nut_expert/config.py:53-57",
                transform_source="UNKNOWN",
                confidence="UNKNOWN",
                note="当前仓库未找到 Nut 世界坐标到左臂 base_link 的确认转换公式。",
            )
        )

    for key, pose in LEFT_PLACE_POSES.items():
        targets.append(
            WorkspaceTarget(
                name=f"Box {key} 放置上方安全点",
                kind="box_hover",
                arm="LEFT_ARM",
                pose=pose,
                coordinate_source="agents/three_nut_expert/config.py:93-97",
                transform_source="已是左臂 move_to 使用的 base_link 目标",
                confidence="INFERRED",
                note="这是现有 Expert 的 staged place pose，尚未由场景盒子中心坐标独立确认。",
            )
        )
        targets.append(
            WorkspaceTarget(
                name=f"Box {key} 放置上方安全点",
                kind="box_hover",
                arm="RIGHT_ARM",
                pose=None,
                coordinate_source="UNKNOWN",
                transform_source="UNKNOWN",
                confidence="UNKNOWN",
                note="当前仓库未找到 Box 目标到右臂 base_link 的确认坐标。",
            )
        )

    return targets


def make_arm(arm: str) -> Any:
    from rabo_robocap import LinkerArmA7

    return LinkerArmA7(robot_id=DEVICE_IDS[arm], mode="sim")


def pose_check(arm_obj: Any, pose: Pose6) -> Any:
    return arm_obj.pose_check(pose.x, pose.y, pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw)


def run_workspace_test(skip_runtime: bool = False, verbose: bool = False) -> dict[str, Any]:
    ensure_dirs()
    targets = build_targets()
    arms: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    runtime_error: str | None = None

    log_event("实验1开始", {"skip_runtime": skip_runtime})
    if not skip_runtime:
        try:
            for arm in ("LEFT_ARM", "RIGHT_ARM"):
                arms[arm] = make_arm(arm)
                if verbose:
                    log_event(
                        "读取机械臂状态",
                        {
                            "arm": arm,
                            "pose": jsonable(arms[arm].get_pose()),
                            "qpos": jsonable(arms[arm].get_joint_angles()),
                        },
                    )
        except Exception as exc:
            runtime_error = repr(exc)
            log_event("运行时初始化失败", {"error": runtime_error})

    for target in targets:
        row: dict[str, Any] = {
            "target": target.name,
            "kind": target.kind,
            "arm": target.arm,
            "pose": pose_to_list(target.pose) if target.pose else None,
            "coordinate_source": target.coordinate_source,
            "transform_source": target.transform_source,
            "confidence": target.confidence,
            "note": target.note,
            "result": "UNKNOWN",
            "reason": "",
            "raw": None,
        }
        if target.pose is None:
            row["result"] = "UNKNOWN"
            row["reason"] = "缺少确认的 base_link 目标坐标"
        elif skip_runtime:
            row["result"] = "CHECK"
            row["reason"] = "本地跳过 Rabo runtime，仅生成报告结构"
        elif runtime_error:
            row["result"] = "CHECK"
            row["reason"] = f"Rabo runtime 初始化失败：{runtime_error}"
        else:
            started = time.time()
            try:
                raw = pose_check(arms[target.arm], target.pose)
                status, reason, value = normalize_pose_check(raw)
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
        log_event("pose_check结果", row)

    for arm, arm_obj in arms.items():
        if hasattr(arm_obj, "shutdown"):
            try:
                arm_obj.shutdown()
            except Exception as exc:
                log_event("shutdown失败", {"arm": arm, "error": repr(exc)})

    summary = build_summary(results, runtime_error)
    write_outputs(summary)
    log_event("实验1结束", {"overall": summary["workspace_mapping"]})
    return summary


def result_for(results: list[dict[str, Any]], arm: str, target: str) -> dict[str, Any] | None:
    return next((r for r in results if r["arm"] == arm and r["target"] == target), None)


def strategy_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("A", "B", "C")

    def covers(arm: str, key: str) -> bool:
        nut = result_for(results, arm, f"Nut {key} 抓取上方安全点")
        box = result_for(results, arm, f"Box {key} 放置上方安全点")
        return bool(nut and box and nut["result"] == "PASS" and box["result"] == "PASS")

    def arm_all(arm: str) -> str:
        rows = [
            result_for(results, arm, f"Nut {key} 抓取上方安全点") for key in keys
        ] + [
            result_for(results, arm, f"Box {key} 放置上方安全点") for key in keys
        ]
        if all(row and row["result"] == "PASS" for row in rows):
            return "通过"
        if any(row is None or row["result"] in {"UNKNOWN", "CHECK"} for row in rows):
            return "检查"
        return "失败"

    assignments: dict[str, list[str]] = {}
    for key in keys:
        assignments[key] = [arm for arm in ("RIGHT_ARM", "LEFT_ARM") if covers(arm, key)]

    single_right = arm_all("RIGHT_ARM")
    single_left = arm_all("LEFT_ARM")
    if all(assignments[key] for key in keys):
        dual = "通过"
        handoff = "否"
    elif any(
        (result_for(results, arm, f"Nut {key} 抓取上方安全点") or {}).get("result") in {"UNKNOWN", "CHECK"}
        or (result_for(results, arm, f"Box {key} 放置上方安全点") or {}).get("result") in {"UNKNOWN", "CHECK"}
        for key in keys
        for arm in ("RIGHT_ARM", "LEFT_ARM")
    ):
        dual = "检查"
        handoff = "未知"
    else:
        dual = "失败"
        handoff = "是"

    if single_right == "通过":
        recommended = "单右臂"
    elif single_left == "通过":
        recommended = "单左臂"
    elif dual == "通过":
        recommended = "双臂独立"
    elif handoff == "是":
        recommended = "handoff"
    else:
        recommended = "暂时无法判断"

    return {
        "single_right": single_right,
        "single_left": single_left,
        "dual_independent": dual,
        "handoff_required": handoff,
        "assignments": assignments,
        "recommended": recommended,
    }


def build_summary(results: list[dict[str, Any]], runtime_error: str | None) -> dict[str, Any]:
    strategy = strategy_from_results(results)
    unknowns: list[str] = []
    if runtime_error:
        unknowns.append(f"Rabo runtime 初始化失败：{runtime_error}")
    if any(r["confidence"] == "UNKNOWN" for r in results):
        unknowns.append("存在缺少确认转换公式的目标坐标")
    if any(r["confidence"] == "INFERRED" for r in results):
        unknowns.append("Box A/B/C 当前只有 staged place pose，未确认真实盒子中心坐标")
    if any(r["result"] == "CHECK" for r in results):
        unknowns.append("存在未完成真实 pose_check 的目标")

    complete_coordinates = not any(r["confidence"] == "UNKNOWN" for r in results)
    all_checked = not any(r["result"] in {"UNKNOWN", "CHECK"} for r in results)
    can_decide = strategy["recommended"] != "暂时无法判断"
    workspace_mapping = "通过" if complete_coordinates and all_checked and can_decide else "检查"
    if any(r["result"] == "FAIL" for r in results) and complete_coordinates and all_checked:
        workspace_mapping = "通过" if can_decide else "失败"

    return {
        "generated": now_text(),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "workspace_mapping": workspace_mapping,
        "runtime_error": runtime_error,
        "source_coordinates": source_coordinates(),
        "results": results,
        "strategy": strategy,
        "unknowns": unknowns,
        "experiment2_status": "未开始。按计划必须先完成实验1并明确机械臂分工后才能开始。",
    }


def source_coordinates() -> dict[str, Any]:
    return {
        "nut": {
            key: {
                "id": spec.thing_id,
                "world_pose": pose_to_list(spec.nominal_pose),
                "source": "agents/three_nut_expert/config.py:53-57",
                "confidence": "A/C 为扩展 staged pose，B 来源于官方 Demo；整体需按报告区分使用。",
            }
            for key, spec in NUT_SPECS.items()
        },
        "right_arm_base_xy": {
            "value": list(RIGHT_ARM_BASE_XY),
            "source": "agents/arm_hand_demo/__init__.py:25-28",
        },
        "right_nut_transform": {
            "formula": "target_x = right_arm_base_x - nut_x + 0.06; target_y = right_arm_base_y - nut_y - 0.01; z=-0.33; rpy=(0,0.8,0)",
            "offsets": GRASP_TARGET_OFFSETS,
            "source": "agents/arm_hand_demo/__init__.py:75-78",
        },
        "left_box_place_poses": {
            key: {
                "pose": pose_to_list(pose),
                "source": "agents/three_nut_expert/config.py:93-97",
                "confidence": "INFERRED",
            }
            for key, pose in LEFT_PLACE_POSES.items()
        },
        "blocked": [
            "未找到左臂 Nut world -> base_link 转换公式",
            "未找到右臂 Box world -> base_link 转换公式",
            "未找到 Box A/B/C 真实中心或分类区域坐标",
        ],
    }


def matrix_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 目标位置 | 右臂 | 左臂 | 右臂原因 | 左臂原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    ordered_names = [f"Nut {key} 抓取上方安全点" for key in ("A", "B", "C")]
    ordered_names += [f"Box {key} 放置上方安全点" for key in ("A", "B", "C")]
    for name in ordered_names:
        right = result_for(results, "RIGHT_ARM", name) or {}
        left = result_for(results, "LEFT_ARM", name) or {}
        lines.append(
            f"| {name} | {right.get('result', 'UNKNOWN')} | {left.get('result', 'UNKNOWN')} | {right.get('reason', '')} | {left.get('reason', '')} |"
        )
    return lines


def target_detail_lines(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in results:
        lines.extend(
            [
                f"### {row['target']} / {row['arm']}",
                "",
                f"- base_link 目标：`{row['pose']}`",
                f"- 坐标来源：{row['coordinate_source']}",
                f"- 转换公式来源：{row['transform_source']}",
                f"- 可信度：{row['confidence']}",
                f"- pose_check 结果：{row['result']}",
                f"- 原因：{row['reason']}",
                f"- 说明：{row['note']}",
                "",
            ]
        )
    return lines


def report_text(summary: dict[str, Any]) -> str:
    strategy = summary["strategy"]
    lines = [
        "# 工作空间测试报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验仅测试几何/关节可达性，不测试轨迹稳定性，不运动机器人。",
        "本实验禁止调用 move_to、move_joints、home、SetEntityPose、clench、grasp_force、抓取、handoff、Recorder 或 ACT。",
        "",
        "## 2. 坐标来源",
        "",
        "- Nut A/B/C 世界坐标：`agents/three_nut_expert/config.py:53-57`。",
        "- 右臂 world -> base_link 抓取上方目标公式：`agents/arm_hand_demo/__init__.py:75-78`。",
        "- 左臂 Box staged place pose：`agents/three_nut_expert/config.py:93-97`，可信度为 INFERRED。",
        "- Box A/B/C 真实中心或分类区域坐标：未找到，记为 BLOCKED_BOX_COORDINATES。",
        "- 左臂 Nut world -> base_link 转换公式：未找到。",
        "- 右臂 Box world -> base_link 转换公式：未找到。",
        "",
        *target_detail_lines(summary["results"]),
        "## 3. 工作空间矩阵",
        "",
        *matrix_lines(summary["results"]),
        "",
        "## 4. 四种策略判定",
        "",
        "| 方案 | 判定 | 原因 |",
        "| --- | --- | --- |",
        f"| 单右臂 | {strategy['single_right']} | 需要右臂同时覆盖 Nut A/B/C 和 Box A/B/C；当前右臂 Box 坐标未确认。 |",
        f"| 单左臂 | {strategy['single_left']} | 需要左臂同时覆盖 Nut A/B/C 和 Box A/B/C；当前左臂 Nut 坐标未确认。 |",
        f"| 双臂独立 | {strategy['dual_independent']} | 需要每个 Nut 的抓取区和对应 Box 由同一只手完整覆盖。 |",
        f"| handoff | {strategy['handoff_required']} | 只有单臂或双臂独立方案不可行时才考虑。 |",
        "",
        "## 5. 实验1最终结论",
        "",
        f"- WORKSPACE_MAPPING：{summary['workspace_mapping']}",
        f"- 推荐策略：{strategy['recommended']}",
        "",
        "原因：",
        "",
    ]
    if summary["unknowns"]:
        lines.extend(f"- {item}" for item in summary["unknowns"])
    else:
        lines.append("- 坐标和 pose_check 信息已足够支撑策略选择。")
    lines.extend(
        [
            "",
            "## 6. 是否进入实验2",
            "",
            "当前脚本只执行实验1。只有在实验1完成并明确选出代表性目标后，才能单独运行实验2。",
            "",
            "## 7. 下一步建议",
            "",
            "先在真实 Rabo runtime 中运行本脚本并查看 `WORKSPACE_MAPPING`。如果仍为“检查”，需要补齐 Box 真实坐标或对应 arm base_link 转换公式；不要提前进入抓取、handoff、Recorder 或 ACT。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any]) -> None:
    ensure_dirs()
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实验1：双臂工作空间 pose_check 测试。")
    parser.add_argument("--verbose", action="store_true", help="额外记录 get_pose/get_joint_angles 只读状态。")
    parser.add_argument("--skip-runtime", action="store_true", help="不导入 Rabo SDK，仅生成中文报告结构。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_workspace_test(skip_runtime=args.skip_runtime, verbose=args.verbose)
    print("========================================")
    print("双臂工作空间测试完成")
    print("========================================")
    print()
    print(f"实验1 工作空间映射：\n{summary['workspace_mapping']}")
    print()
    print(f"推荐机械臂策略：\n{summary['strategy']['recommended']}")
    print()
    print("实验2 move_to 稳定性：\n未开始")
    print()
    print("是否可以进入 ACT 数据采集：\n否")
    print()
    print(f"工作空间报告：\n{REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"JSON：\n{JSON_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"原始日志：\n{LOG_PATH.relative_to(PROJECT_ROOT)}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
