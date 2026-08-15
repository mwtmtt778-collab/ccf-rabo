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

from agents.three_nut_expert.config import LEFT_PLACE_POSES, NUT_SPECS, RIGHT_ARM_BASE_XY  # noqa: E402
from agents.three_nut_expert.expert import compute_right_grasp_pose, pose_to_list  # noqa: E402
from expert.transforms import pose_orientation_error, pose_position_error, transform_pose_world_to_base  # noqa: E402


LEGACY_RIGHT_NUT_TARGETS = {
    key: pose_to_list(compute_right_grasp_pose(spec.nominal_pose)) for key, spec in NUT_SPECS.items()
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
            if path.is_file() and path.suffix.lower() in {".sdf", ".urdf", ".xacro", ".xml", ".json", ".yaml", ".yml"}:
                model_config_files.append(str(path.relative_to(PROJECT_ROOT)))
    return {
        "rg_matches_preview": rg["stdout"][:20000],
        "rg_truncated": len(rg["stdout"]) > 20000,
        "model_config_files": model_config_files[:500],
        "errors": [rg["stderr"]] if rg["stderr"] else [],
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
    boxes = {
        key: {
            "world_pose": None,
            "staged_left_arm_pose": pose_to_list(pose),
            "source": "agents/three_nut_expert/config.py:93-97",
            "confidence": "INFERRED_ONLY",
            "note": "当前仅找到左臂 staged place pose，不能当作 Box 世界坐标。",
        }
        for key, pose in LEFT_PLACE_POSES.items()
    }

    frames = {
        "world": {"frame": "world", "confidence": "INFERRED_NAME"},
        "left_arm_base": {"frame": None, "world_pose": None, "confidence": "UNKNOWN"},
        "right_arm_base": infer_right_base_from_legacy(),
        "tf_frames_seen": frames_seen,
    }

    validation = legacy_validation(frames["right_arm_base"], nuts)
    unknowns = []
    if frames["left_arm_base"]["confidence"] != "CONFIRMED":
        unknowns.append("LEFT_ARM_BASE_WORLD_POSE 未确认")
    if frames["right_arm_base"]["confidence"] != "CONFIRMED":
        unknowns.append("RIGHT_ARM_BASE_WORLD_POSE 未确认，当前只有 legacy 公式推断")
    if any(item["confidence"] != "CONFIRMED" for item in boxes.values()):
        unknowns.append("BLOCKED_BOX_WORLD_POSE：Box A/B/C 真实世界坐标未确认")
    if validation["status"] != "PASS":
        unknowns.append("TRANSFORM_VALIDATION 未通过，不能生成完整 6×2 pose_check")

    result = {
        "generated": now_text(),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "frames": frames,
        "nuts": nuts,
        "boxes": boxes,
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
    if right_base.get("confidence") != "CONFIRMED":
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
        pos_error = pose_position_error(transformed, legacy)
        ori_error = pose_orientation_error(transformed, legacy)
        row_ok = pos_error <= 0.02 and ori_error <= 0.05
        ok = ok and row_ok
        rows.append(
            {
                "nut": key,
                "legacy_target": legacy,
                "v2_transform_target": transformed,
                "position_error": pos_error,
                "orientation_error": ori_error,
                "result": "PASS" if row_ok else "FAIL",
            }
        )
    return {"status": "PASS" if ok else "FAIL", "rows": rows}


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
            "## 需要用户从 Rabo 场景中提供",
            "",
            "1. 蓝色分类盒模型名称 / ID。",
            "2. 蓝色分类盒模型 world pose。",
            "3. 如果三个格子是独立 link：Box A/B/C 三个 link 的 pose。",
            "4. 如果三个格子只是一个模型内部区域：Box A/B/C 三个目标格中心的 world pose。",
            "5. 左臂 base_link 的 world pose。",
            "6. 右臂 base_link 的 world pose，或可从 TF 中确认的 frame 名称。",
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
    print("========================================")
    print("工作空间测试 V2 坐标解析")
    print("========================================")
    print(f"左右臂 base_link 世界位姿：{'已确认' if not any('BASE_WORLD_POSE' in u for u in result['unknowns']) else '未确认'}")
    print(f"Box A/B/C 世界坐标：{'已确认' if not any('BOX_WORLD_POSE' in u for u in result['unknowns']) else '未确认'}")
    print(f"Legacy 右臂变换验证：{result['validation']['right_arm_legacy']['status']}")
    print(f"坐标 JSON：{COORD_JSON.relative_to(PROJECT_ROOT)}")
    if result["blocked"]:
        print(f"阻塞说明：{NEED_COORDS_MD.relative_to(PROJECT_ROOT)}")
    print(f"原始日志：{LOG_PATH.relative_to(PROJECT_ROOT)}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
