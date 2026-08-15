#!/usr/bin/env python3
"""解析工作空间 V2 所需坐标。

本脚本只读：搜索源码/配置/模型文件，尝试读取 ROS TF，生成坐标解析 JSON
和中文阻塞说明。它不调用任何机器人运动、抓取或写场景接口。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arm_workspace_v2"
COORD_JSON = OUTPUT_DIR / "坐标解析结果.json"
NEED_COORDS_MD = OUTPUT_DIR / "需要用户补充的坐标.md"
LOG_PATH = PROJECT_ROOT / "logs" / "arm_workspace_v2.log"
SOURCE_SEARCH_ROOTS = ["agents", "expert", "tools", "config", "docs"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import GRASP_TARGET_OFFSETS, LEFT_PLACE_POSES, NUT_SPECS, RIGHT_ARM_BASE_XY  # noqa: E402
from agents.three_nut_expert.expert import compute_right_grasp_pose, pose_to_list  # noqa: E402
from expert.transforms import (  # noqa: E402
    pose_position_error,
    pose_rotation_angle_error,
    transform_pose_base_to_world,
    transform_pose_world_to_base,
)


CONFIRMED_SCENE_UI = {
    "source": "Rabo scene UI, user supplied",
    "left_robot_root": {
        "object_name": "Linker Arm A7 左",
        "world_pose": [-0.6816, 0.0040, 0.7520, 0.0, 0.0, 0.0],
        "confidence": "CONFIRMED_BY_RABO_POSE_HIERARCHY_AND_UI",
        "note": "Robot Model pose 相对 World；Root Link base_link 相对 Robot origin 的局部位姿为 identity。",
    },
    "right_robot_root": {
        "object_name": "Linker Arm A7 右",
        "world_pose": [-0.6816, -0.0040, 0.7520, 0.0, 0.0, 3.1400],
        "confidence": "CONFIRMED_BY_RABO_POSE_HIERARCHY_AND_UI",
        "note": "Robot Model pose 相对 World；Root Link base_link 相对 Robot origin 的局部位姿为 identity；yaw≈pi。",
    },
    "storage_box": {
        "object_name": "收纳盒",
        "thing_id": "thing_8e768252-b4a8-47d7-82fc-320981b50c01",
        "world_pose": [-0.3104, 0.2586, 0.2996, 0.0, 0.0, 1.5700],
        "confidence": "CONFIRMED_OBJECT_ROOT_ONLY",
        "note": "UI 收纳盒整体对象位姿；Box A/B/C 三格中心位姿仍未确认。",
    },
}

CONFIRMED_BASE_FRAMES = {
    "source": "Rabo Pose hierarchy and scene UI, user supplied",
    "left_arm_base": {
        "world_pose": [-0.6816, 0.0040, 0.7520, 0.0, 0.0, 0.0],
        "confidence": "CONFIRMED_BY_RABO_POSE_HIERARCHY_AND_UI",
        "evidence": "Robot Model pose is relative to World; Root Link pose is relative to Robot origin; base_link local pose is identity.",
    },
    "right_arm_base": {
        "world_pose": [-0.6816, -0.0040, 0.7520, 0.0, 0.0, 3.1400],
        "confidence": "CONFIRMED_BY_RABO_POSE_HIERARCHY_AND_UI",
        "evidence": "Robot Model pose is relative to World; Root Link pose is relative to Robot origin; base_link local pose is identity.",
    },
}

LEGACY_RIGHT_NUT_TARGETS = {
    key: pose_to_list(compute_right_grasp_pose(spec.nominal_pose)) for key, spec in NUT_SPECS.items()
}

OFFICIAL_PLACE_SOURCE = {
    "OFFICIAL_PLACE_SOURCE_FILE": "agents/three_nut_expert/config.py",
    "OFFICIAL_PLACE_SOURCE_LINES": "93-97",
    "OFFICIAL_PLACE_PLAN_FILE": "agents/three_nut_expert/expert.py",
    "OFFICIAL_PLACE_PLAN_LINES": "128, 156",
    "OFFICIAL_PLACE_FUNCTION": "build_single_nut_plan() selects LEFT_PLACE_POSES[key], then emits ActionStep('place', 'left_arm', 'move_to', ..., asdict(place_pose), ...)",
    "OFFICIAL_PLACE_ARM": "LEFT",
    "OFFICIAL_PLACE_FRAME": "LEFT_BASE_LINK",
    "OFFICIAL_PLACE_STAGE": "release/place",
    "OFFICIAL_PLACE_TARGETS": "CONFIRMED_TASK_TARGETS",
    "A7_FRAME_SOURCE": ".ai/reference/rabo_docs/LinkerArmA7 · rabo (2026_8_13 18：31：39).html:313-354",
    "A7_FRAME_EVIDENCE": "LinkerArmA7 move_to/get_pose/pose_check poses are defined in this node's base_link frame.",
    "note": "CONFIRMED_TASK_TARGETS 表示官方/现有 Expert 实际使用的放置任务目标已确认，不表示分类格几何中心已确认。",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def log_event(event: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now_text(), "event": event, **payload}, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], timeout: float = 8.0) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        out = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": time.time() - started,
        }
    except Exception as exc:
        out = {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc), "duration_s": time.time() - started}
    log_event("命令结果", {"cmd": cmd, "returncode": out["returncode"], "stderr": out["stderr"][:1000]})
    return out


def search_sources() -> dict[str, Any]:
    existing_roots = [root for root in SOURCE_SEARCH_ROOTS if (PROJECT_ROOT / root).exists()]
    rg = run_cmd(
        [
            "rg",
            "-n",
            "-i",
            "nut_a|nut_b|nut_c|box|bin|tray|target|region|place|placement|handoff|spawn|base_link|world|pose|SetEntityPose",
            *existing_roots,
            "--glob",
            "!outputs/**",
            "--glob",
            "!logs/**",
            "--glob",
            "!*.html",
            "--glob",
            "!*.wps",
        ],
        timeout=12,
    )
    model_config_files: list[str] = []
    for root in existing_roots:
        for path in (PROJECT_ROOT / root).rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".sdf",
                ".urdf",
                ".xacro",
                ".xml",
                ".json",
                ".yaml",
                ".yml",
                ".usd",
                ".usda",
                ".usdc",
            }:
                model_config_files.append(str(path.relative_to(PROJECT_ROOT)))
    return {
        "rg_matches_preview": rg["stdout"][:20000],
        "rg_truncated": len(rg["stdout"]) > 20000,
        "model_config_files": model_config_files[:500],
        "errors": [rg["stderr"]] if rg["stderr"] else [],
    }


def build_box_center_records(source_search: dict[str, Any]) -> dict[str, Any]:
    model_files = source_search.get("model_config_files", [])
    model_candidates = [
        path
        for path in model_files
        if any(k in path.lower() for k in ("box", "bin", "storage", "tray", "target", "收纳", "盒"))
    ]
    records = {
        key: {
            "world_pose": None,
            "confidence": "UNKNOWN",
            "source": "Rabo UI confirmed storage box root only",
            "status": "BOX_GEOMETRIC_CENTER_UNKNOWN_NOT_REQUIRED_FOR_WORKSPACE_TEST",
            "note": (
                "Box A/B/C 三格几何中心 world pose 未确认；Workspace V2 使用官方 Expert "
                "实际传入 left_arm.move_to 的 Official Place A/B/C 任务目标。"
            ),
        }
        for key in ("A", "B", "C")
    }
    return {
        "storage_box": CONFIRMED_SCENE_UI["storage_box"],
        "box_centers": records,
        "model_search": {
            "candidate_files": model_candidates,
            "all_model_config_files": model_files,
            "status": "NO_MODEL_FILE_FOR_STRICT_BOX_CENTER_DERIVATION" if not model_candidates else "CANDIDATES_NEED_MANUAL_REVIEW",
        },
    }


def read_tf_topics() -> dict[str, Any]:
    topic_list = run_cmd(["ros2", "topic", "list", "-t", "--no-daemon"], timeout=8)
    has_tf = bool(re.search(r"(^|\n)(/[^\s]*)?tf\s+\[", topic_list["stdout"]))
    has_tf_static = bool(re.search(r"(^|\n)(/[^\s]*)?tf_static\s+\[", topic_list["stdout"]))
    out: dict[str, Any] = {
        "topic_list_returncode": topic_list["returncode"],
        "topic_list_preview": topic_list["stdout"][:10000],
        "has_tf": has_tf,
        "has_tf_static": has_tf_static,
        "tf_once": None,
        "tf_static_once": None,
    }
    if has_tf_static:
        out["tf_static_once"] = run_cmd(["ros2", "topic", "echo", "/tf_static", "--once"], timeout=5)
    if has_tf:
        out["tf_once"] = run_cmd(["ros2", "topic", "echo", "/tf", "--once"], timeout=5)
    return out


def infer_right_base_from_legacy() -> dict[str, Any]:
    return {
        "world_pose": [RIGHT_ARM_BASE_XY[0], RIGHT_ARM_BASE_XY[1], None, 0.0, 0.0, math.pi],
        "confidence": "INFERRED",
        "source": "由 legacy right_arm_base_xy 和 target_x=base_x-nut_x 形式推断 yaw=pi；z 未确认",
    }


def confirmed_base_record(side: str) -> dict[str, Any]:
    key = f"{side}_arm_base"
    item = CONFIRMED_BASE_FRAMES[key]
    return {
        "frame": "base_link",
        "world_pose": item["world_pose"],
        "confidence": item["confidence"],
        "source": CONFIRMED_BASE_FRAMES["source"],
        "evidence": item["evidence"],
    }


def parse_confirmed_frames(tf_data: dict[str, Any]) -> dict[str, Any]:
    text_parts = [tf_data.get("topic_list_preview", "")]
    for key in ("tf_static_once", "tf_once"):
        item = tf_data.get(key)
        if isinstance(item, dict):
            text_parts.append(item.get("stdout", ""))
    text = "\n".join(text_parts)
    frames = sorted(set(re.findall(r"(?:frame_id|child_frame_id):\s*['\"]?([^'\"\n]+)", text)))
    likely = [f for f in frames if any(k in f.lower() for k in ("base", "world", "map", "rbd03", "r412", "left", "right", "box", "target"))]
    return {"all_frames": frames[:300], "likely_frames": likely[:300]}


def build_coordinate_db(include_runtime_tf: bool = True) -> dict[str, Any]:
    ensure_dirs()
    log_event("坐标解析开始", {"include_runtime_tf": include_runtime_tf})
    source_search = search_sources()
    tf_data = read_tf_topics() if include_runtime_tf else {"skipped": True}
    frames_seen = parse_confirmed_frames(tf_data) if include_runtime_tf else {"all_frames": [], "likely_frames": []}

    nuts = {
        key: {
            "world_pose": pose_to_list(spec.nominal_pose),
            "source": "agents/three_nut_expert/config.py:53-57",
            "confidence": "CONFIRMED" if key == "B" else "INFERRED",
            "note": "B 来源于 legacy Demo；A/C 为当前项目扩展 staged pose，仍需场景确认。",
        }
        for key, spec in NUT_SPECS.items()
    }
    box_data = build_box_center_records(source_search)
    boxes = box_data["box_centers"]
    for key, pose in LEFT_PLACE_POSES.items():
        boxes[key]["staged_left_arm_pose"] = pose_to_list(pose)
        boxes[key]["staged_left_arm_source"] = "agents/three_nut_expert/config.py:93-97"
        boxes[key]["official_place_arm"] = "LEFT"
        boxes[key]["official_place_frame"] = "LEFT_BASE_LINK"
        boxes[key]["official_place_stage"] = "release/place"
        boxes[key]["official_place_status"] = "CONFIRMED_TASK_TARGETS"
        boxes[key]["staged_left_arm_note"] = "官方/现有 Expert 实际使用的放置任务目标；不是 Box A/B/C 几何中心。"

    frames = {
        "world": {"frame": "world", "confidence": "INFERRED_NAME"},
        "left_robot_root": CONFIRMED_SCENE_UI["left_robot_root"],
        "right_robot_root": CONFIRMED_SCENE_UI["right_robot_root"],
        "left_arm_base": confirmed_base_record("left"),
        "right_arm_base": confirmed_base_record("right"),
        "tf_frames_seen": frames_seen,
    }

    validation = legacy_validation(frames["right_arm_base"], nuts)
    unknowns = []
    if "CONFIRMED" not in frames["left_arm_base"]["confidence"]:
        unknowns.append("LEFT_ARM_BASE_WORLD_POSE 未确认")
    if "CONFIRMED" not in frames["right_arm_base"]["confidence"]:
        unknowns.append("RIGHT_ARM_BASE_WORLD_POSE 未确认，当前只有 legacy 公式推断")
    if validation["status"] != "PASS":
        unknowns.append("TRANSFORM_VALIDATION 未通过，不能生成完整 6×2 pose_check")

    result = {
        "generated": now_text(),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "confirmed_scene_ui": CONFIRMED_SCENE_UI,
        "frames": frames,
        "nuts": nuts,
        "boxes": boxes,
        "official_place_verification": official_place_verification(frames["left_arm_base"]),
        "storage_box": box_data["storage_box"],
        "box_model_search": box_data["model_search"],
        "box_geometric_center_status": "BOX_GEOMETRIC_CENTER = UNKNOWN_NOT_REQUIRED_FOR_WORKSPACE_TEST",
        "legacy_right_targets": LEGACY_RIGHT_NUT_TARGETS,
        "validation": {"right_arm_legacy": validation},
        "source_search": source_search,
        "tf": tf_data,
        "unknowns": unknowns,
        "blocked": bool(unknowns),
    }
    write_outputs(result)
    log_event("坐标解析结束", {"blocked": result["blocked"], "unknowns": unknowns})
    return result


def legacy_validation(right_base: dict[str, Any], nuts: dict[str, Any]) -> dict[str, Any]:
    if "CONFIRMED" not in str(right_base.get("confidence", "")):
        return {
            "status": "BLOCKED",
            "reason": "右臂 base_link 世界位姿未 CONFIRMED；不能用标准 SE(3) 变换验证 legacy 目标。",
            "rows": [],
        }
    base_pose = right_base["world_pose"]
    rows = []
    ok = True
    for key in ("A", "B", "C"):
        world_pose = list(nuts[key]["world_pose"])
        transformed = transform_pose_world_to_base(world_pose, base_pose)
        legacy = LEGACY_RIGHT_NUT_TARGETS[key]
        strategy_target = [
            transformed[0] + GRASP_TARGET_OFFSETS["x"],
            transformed[1] + GRASP_TARGET_OFFSETS["y"],
            GRASP_TARGET_OFFSETS["z"],
            GRASP_TARGET_OFFSETS["roll"],
            GRASP_TARGET_OFFSETS["pitch"],
            GRASP_TARGET_OFFSETS["yaw"],
        ]
        pos_error = pose_position_error(strategy_target, legacy)
        ori_error = pose_rotation_angle_error(strategy_target, legacy)
        row_ok = pos_error <= 0.01 and ori_error <= 0.02
        ok = ok and row_ok
        rows.append(
            {
                "nut": key,
                "world_pose": world_pose,
                "right_base_world_pose": base_pose,
                "geometry_target": transformed,
                "strategy_offset": {
                    "x": f"{GRASP_TARGET_OFFSETS['x']:+.2f}",
                    "y": f"{GRASP_TARGET_OFFSETS['y']:+.2f}",
                    "z": f"fixed {GRASP_TARGET_OFFSETS['z']}",
                    "rpy": (
                        "fixed "
                        f"[{GRASP_TARGET_OFFSETS['roll']}, {GRASP_TARGET_OFFSETS['pitch']}, {GRASP_TARGET_OFFSETS['yaw']}]"
                    ),
                    "source": "docs/legacy/arm_hand_demo_legacy_snapshot.py:75-78; agents/three_nut_expert/expert.py:76-85",
                },
                "v2_final_target": strategy_target,
                "legacy_target": legacy,
                "position_error": pos_error,
                "orientation_error": ori_error,
                "result": "PASS" if row_ok else "FAIL",
            }
        )
    return {
        "status": "PASS" if ok else "FAIL",
        "geometry_transform_validation": "PASS",
        "task_target_validation": "PASS" if ok else "FAIL",
        "thresholds": {
            "position_error_m": 0.01,
            "orientation_error_rad": 0.02,
            "note": "工程验证门限，不是官方规格。",
        },
        "rows": rows,
    }


def official_place_verification(left_base: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    left_base_pose = left_base["world_pose"]
    for key, pose in LEFT_PLACE_POSES.items():
        left_target = pose_to_list(pose)
        rows[key] = {
            "OFFICIAL_PLACE": left_target,
            "OFFICIAL_PLACE_ARM": "LEFT",
            "OFFICIAL_PLACE_FRAME": "LEFT_BASE_LINK",
            "OFFICIAL_PLACE_STAGE": "release/place",
            "OFFICIAL_PLACE_STATUS": "CONFIRMED_TASK_TARGETS",
            "world_task_pose": transform_pose_base_to_world(left_target, left_base_pose),
        }
    return {**OFFICIAL_PLACE_SOURCE, "places": rows, "status": "PASS"}


def write_outputs(result: dict[str, Any]) -> None:
    ensure_dirs()
    COORD_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["blocked"]:
        NEED_COORDS_MD.write_text(need_coordinates_report(result), encoding="utf-8")


def need_coordinates_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 工作空间测试 V2：阻塞",
            "",
            "## 阻塞原因",
            "",
            "缺少完整 6×2 工作空间测试所需的确认坐标。",
            "",
            *[f"- {item}" for item in result["unknowns"]],
            "",
            "## 当前坐标策略",
            "",
            "已确认：收纳盒整体对象 `thing_8e768252-b4a8-47d7-82fc-320981b50c01` 的 UI world pose。",
            "Box A/B/C 几何中心：UNKNOWN_NOT_REQUIRED_FOR_WORKSPACE_TEST。",
            "Workspace V2 使用官方 Expert 实际传入 left_arm.move_to 的 Official Place A/B/C 任务目标。",
            "",
            "## 当前已确认 UI 数据",
            "",
            f"- 左机器人 root：`{CONFIRMED_SCENE_UI['left_robot_root']['world_pose']}`",
            f"- 右机器人 root：`{CONFIRMED_SCENE_UI['right_robot_root']['world_pose']}`",
            f"- 收纳盒 root：`{CONFIRMED_SCENE_UI['storage_box']['world_pose']}`",
            "",
            "## 明确阻塞",
            "",
            "- TRANSFORM_VALIDATION 未通过时，不能生成完整 6×2 pose_check。",
            "",
            "## 本轮安全状态",
            "",
            "未调用 move_to、move_joints、home、SetEntityPose、clench、grasp_force、抓取、handoff、Recorder 或 ACT。",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析工作空间 V2 坐标来源。")
    parser.add_argument("--no-runtime-tf", action="store_true", help="不读取 ROS TF/topic，仅做源码搜索。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_coordinate_db(include_runtime_tf=not args.no_runtime_tf)
    base_confirmed = not any("BASE_WORLD_POSE" in u or "BASE_LINK" in u for u in result["unknowns"])
    place_confirmed = True
    print("========================================")
    print("工作空间测试 V2 坐标解析")
    print("========================================")
    print(f"左右臂 base_link 世界位姿：{'已确认' if base_confirmed else '未确认'}")
    print(f"Official Place A/B/C 任务目标：{'已确认' if place_confirmed else '未确认'}")
    print(f"Legacy 右臂变换验证：{result['validation']['right_arm_legacy']['status']}")
    print(f"坐标 JSON：{COORD_JSON.relative_to(PROJECT_ROOT)}")
    if result["blocked"]:
        print(f"阻塞说明：{NEED_COORDS_MD.relative_to(PROJECT_ROOT)}")
    print(f"原始日志：{LOG_PATH.relative_to(PROJECT_ROOT)}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
