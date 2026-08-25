"""Left-hand Nut grasp planner V1.

This module implements a task-specific Nut grasp template and a staged approach
strategy. It deliberately does not claim collision-free planning: LinkerArmA7
pose_check/move_to provide IK reachability and execution only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


LEFT_BASE_WORLD = [-0.6816, 0.0040, 0.7520, 0.0, 0.0, 0.0]
LEFT_GRASP_OFFSET_BASE = [-0.05829, -0.00924, 0.13681]
LEFT_SLANTED_GRASP_RPY = [0.0, -0.8544, 0.0]

SAFE_PRE_JOINTS_1 = [0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
SAFE_PRE_JOINTS_2 = [-1.57, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
SAFE_PRE_JOINTS = [SAFE_PRE_JOINTS_1, SAFE_PRE_JOINTS_2]

SAFE_HIGH_POSE = [0.43, 0.30, -0.10, 0.0, -0.8544, 0.0]
APPROACH_HEIGHT = 0.12
DESCENT_OFFSETS = [0.08, 0.05, 0.03, 0.015, 0.0]
LIFT_OFFSETS = [0.03, 0.08]
GRASP_FORCE = {"strength": 1.0, "fingers": [1, 3, 4]}

J2_MAX = 0.07
J2_WARNING_THRESHOLD = 0.06


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


def world_xyz_to_left_base_xyz(nut_world_xyz: list[float] | tuple[float, float, float]) -> list[float]:
    if len(nut_world_xyz) != 3:
        raise ValueError(f"nut_world_xyz must have 3 values, got {len(nut_world_xyz)}")
    return [
        float(nut_world_xyz[0]) - LEFT_BASE_WORLD[0],
        float(nut_world_xyz[1]) - LEFT_BASE_WORLD[1],
        float(nut_world_xyz[2]) - LEFT_BASE_WORLD[2],
    ]


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
    if "true" in low or "reachable" in low or "success" in low:
        return "PASS", text, value
    if "false" in low or "fail" in low or "limit" in low or "workspace" in low:
        return "FAIL", text, value
    return "CHECK", text, value


def result_failed(result: Any) -> tuple[bool, str]:
    value = jsonable(result)
    if result is None:
        return False, ""
    if isinstance(result, bool):
        return (not result), "returned_false"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
    low = text.lower()
    if any(token in low for token in ("error", "failed", "fail", "false", "ik", "无解", "失败", "错误")):
        return True, text
    return False, ""


def pose_with_z(pose: list[float], z: float) -> list[float]:
    return [pose[0], pose[1], z, pose[3], pose[4], pose[5]]


@dataclass
class PlannerConfig:
    left_base_world: list[float]
    grasp_offset_base: list[float]
    grasp_rpy: list[float]
    safe_pre_joints: list[list[float]]
    safe_high_pose: list[float]
    approach_height: float
    descent_offsets: list[float]
    lift_offsets: list[float]
    grasp_force: dict[str, Any]


class LeftNutGraspPlanner:
    def __init__(self, left_arm: Any, left_hand: Any, *, config: PlannerConfig | None = None) -> None:
        self.left_arm = left_arm
        self.left_hand = left_hand
        self.config = config or PlannerConfig(
            left_base_world=list(LEFT_BASE_WORLD),
            grasp_offset_base=list(LEFT_GRASP_OFFSET_BASE),
            grasp_rpy=list(LEFT_SLANTED_GRASP_RPY),
            safe_pre_joints=[list(j) for j in SAFE_PRE_JOINTS],
            safe_high_pose=list(SAFE_HIGH_POSE),
            approach_height=APPROACH_HEIGHT,
            descent_offsets=list(DESCENT_OFFSETS),
            lift_offsets=list(LIFT_OFFSETS),
            grasp_force=dict(GRASP_FORCE),
        )

    def world_to_left_base(self, xyz: list[float] | tuple[float, float, float]) -> list[float]:
        return world_xyz_to_left_base_xyz(xyz)

    def build_grasp_pose(self, nut_world_xyz: list[float], yaw_offset: float = 0.0) -> list[float]:
        nut_left = self.world_to_left_base(nut_world_xyz)
        return [
            nut_left[0] + self.config.grasp_offset_base[0],
            nut_left[1] + self.config.grasp_offset_base[1],
            nut_left[2] + self.config.grasp_offset_base[2],
            self.config.grasp_rpy[0],
            self.config.grasp_rpy[1],
            yaw_offset,
        ]

    def build_pregrasp_pose(self, grasp_pose: list[float]) -> list[float]:
        return pose_with_z(grasp_pose, grasp_pose[2] + self.config.approach_height)

    def build_descent_waypoints(self, grasp_pose: list[float]) -> list[dict[str, Any]]:
        rows = []
        for index, offset in enumerate(self.config.descent_offsets, start=1):
            stage = "GRASP" if abs(offset) < 1e-9 else f"DESCENT_{index}"
            rows.append({"stage": stage, "pose": pose_with_z(grasp_pose, grasp_pose[2] + offset), "offset": offset})
        return rows

    def build_lift_waypoints(self, grasp_pose: list[float]) -> list[dict[str, Any]]:
        return [
            {"stage": f"LIFT_{index}", "pose": pose_with_z(grasp_pose, grasp_pose[2] + offset), "offset": offset}
            for index, offset in enumerate(self.config.lift_offsets, start=1)
        ]

    def build_plan(self, nut_world_xyz: list[float], yaw_offset: float = 0.0) -> dict[str, Any]:
        nut_left = self.world_to_left_base(nut_world_xyz)
        grasp_pose = self.build_grasp_pose(nut_world_xyz, yaw_offset=yaw_offset)
        pregrasp_pose = self.build_pregrasp_pose(grasp_pose)
        descent = self.build_descent_waypoints(grasp_pose)
        lift = self.build_lift_waypoints(grasp_pose)
        return {
            "nut_world_xyz": [float(v) for v in nut_world_xyz],
            "nut_left_base_xyz": nut_left,
            "grasp_template": {
                "position_offset_left_base": list(self.config.grasp_offset_base),
                "orientation_rpy": [self.config.grasp_rpy[0], self.config.grasp_rpy[1], yaw_offset],
                "position_source": "VERIFIED_LEFT_NUT_B_GRASP_TEMPLATE",
                "orientation_source": "LEFT_VISUAL_SLANTED_CALIBRATION",
            },
            "safe_pre_joints": [list(j) for j in self.config.safe_pre_joints],
            "safe_high_pose": [self.config.safe_high_pose[0], self.config.safe_high_pose[1], self.config.safe_high_pose[2], self.config.safe_high_pose[3], self.config.safe_high_pose[4], yaw_offset],
            "grasp_pose": grasp_pose,
            "pregrasp_pose": pregrasp_pose,
            "descent_waypoints": descent,
            "lift_waypoints": lift,
            "path": [
                {"stage": "SAFE_HIGH", "pose": [self.config.safe_high_pose[0], self.config.safe_high_pose[1], self.config.safe_high_pose[2], self.config.safe_high_pose[3], self.config.safe_high_pose[4], yaw_offset]},
                {"stage": "PREGRASP", "pose": pregrasp_pose},
                *descent,
                *lift,
            ],
        }

    def pose_check(self, pose: list[float]) -> dict[str, Any]:
        raw = self.left_arm.pose_check(pose[0], pose[1], pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5])
        status, reason, value = normalize_pose_check(raw)
        return {
            "status": status,
            "pass": status == "PASS",
            "reason": reason,
            "raw": value,
            "pose": list(pose),
        }

    def preflight(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for item in plan["path"]:
            check = self.pose_check(item["pose"])
            row = {"stage": item["stage"], "target_pose": list(item["pose"]), **check}
            rows.append(row)
        return rows

    def preflight_grasp_chain(self, plan: dict[str, Any]) -> dict[str, Any]:
        rows = self.preflight(plan)
        failed = next((row for row in rows if not row["pass"]), None)
        return {
            "success": failed is None,
            "checks": rows,
            "failed_stage": None if failed is None else failed["stage"],
            "reason": None if failed is None else failed["reason"],
            "failed_pose": None if failed is None else failed["target_pose"],
        }

    def joint_margin(self, joints: Any) -> dict[str, Any]:
        value = jsonable(joints)
        if not isinstance(value, list) or len(value) < 2:
            return {"available": False, "reason": "joint_angles_unavailable", "raw": value}
        try:
            j2 = float(value[1])
        except Exception:
            return {"available": False, "reason": "j2_not_numeric", "raw": value}
        margin_to_max = J2_MAX - j2
        warning = j2 > J2_WARNING_THRESHOLD
        return {
            "available": True,
            "j2": j2,
            "j2_max_reference": J2_MAX,
            "j2_margin_to_max": margin_to_max,
            "warning": warning,
        }

    def move_to_pose(self, stage: str, pose: list[float]) -> dict[str, Any]:
        precheck = self.pose_check(pose)
        if not precheck["pass"]:
            return {
                "stage": stage,
                "target_pose": list(pose),
                "pose_check": precheck,
                "move_executed": False,
                "move_success": False,
                "reason": precheck["reason"],
            }
        setattr(self.left_arm, "current_phase", stage)
        result = self.left_arm.move_to(pose[0], pose[1], pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5])
        failed, reason = result_failed(result)
        actual_pose = self.safe_call(self.left_arm, "get_pose")
        joints = self.safe_call(self.left_arm, "get_joint_angles")
        margin = self.joint_margin(joints)
        self.print_j2_margin(margin)
        return {
            "stage": stage,
            "target_pose": list(pose),
            "pose_check": precheck,
            "move_executed": True,
            "move_to_return": jsonable(result),
            "move_success": not failed,
            "reason": reason,
            "actual_pose": jsonable(actual_pose),
            "joint_angles": jsonable(joints),
            "joint_margin": margin,
        }

    def safe_call(self, obj: Any, method_name: str) -> Any:
        if obj is None or not hasattr(obj, method_name):
            return None
        try:
            return jsonable(getattr(obj, method_name)())
        except Exception as exc:
            return f"{method_name} ERROR {repr(exc)}"

    def print_j2_margin(self, margin: dict[str, Any]) -> None:
        if not margin.get("available"):
            return
        print("J2 当前值：")
        print(f"{margin['j2']:.4f} rad")
        print("")
        print("距离上限：")
        print(f"{margin['j2_margin_to_max']:.4f} rad")
        if margin.get("warning"):
            print("")
            print("[警告]")
            print("")
            print("当前 J2 已接近上限。")
            print("虽然 Pose 可达，但该构型余量较小。")

    def execute_safe_pre_joints(self) -> list[dict[str, Any]]:
        rows = []
        for index, joints in enumerate(self.config.safe_pre_joints, start=1):
            print(f"[安全预摆 {index}] 目标关节：{joints}")
            result = self.left_arm.move_joints(joints)
            failed, reason = result_failed(result)
            row = {
                "stage": f"SAFE_PRE_JOINTS_{index}",
                "target_joints": list(joints),
                "return_value": jsonable(result),
                "success": not failed,
                "reason": reason,
            }
            rows.append(row)
            print(f"返回值：{jsonable(result)}")
            if failed:
                break
        return rows

    def open_hand(self) -> dict[str, Any]:
        if hasattr(self.left_hand, "open"):
            result = self.left_hand.open()
            method = "open"
        else:
            result = self.left_hand.clench([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            method = "clench"
        failed, reason = result_failed(result)
        return {"method": method, "return_value": jsonable(result), "success": not failed, "reason": reason}

    def execute_thumb_tuck(self) -> dict[str, Any]:
        clench_result = self.left_hand.clench(thumb_rotation=1.0)
        clench_failed, clench_reason = result_failed(clench_result)
        time.sleep(0.7)
        return {
            "method": "clench",
            "kwargs": {"thumb_rotation": 1.0},
            "return_value": jsonable(clench_result),
            "success": not clench_failed,
            "reason": clench_reason,
        }

    def execute_grasp_force(self) -> dict[str, Any]:
        force_result = self.left_hand.grasp_force(
            strength=self.config.grasp_force["strength"],
            fingers=self.config.grasp_force["fingers"],
        )
        force_failed, force_reason = result_failed(force_result)
        return {
            "method": "grasp_force",
            "kwargs": dict(self.config.grasp_force),
            "return_value": jsonable(force_result),
            "success": not force_failed,
            "reason": force_reason,
        }

    def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": False,
            "safe_pre_joints": [],
            "open_hand": None,
            "executed_waypoints": [],
            "grasp_action": {},
            "lift": [],
            "failed_stage": None,
            "reason": None,
        }
        open_result = self.open_hand()
        result["open_hand"] = open_result
        print(f"左手张开返回值：{open_result['return_value']}")
        if not open_result["success"]:
            result["failed_stage"] = "OPEN_HAND"
            result["reason"] = open_result["reason"]
            return result

        safe_rows = self.execute_safe_pre_joints()
        result["safe_pre_joints"] = safe_rows
        failed_safe = next((row for row in safe_rows if not row["success"]), None)
        if failed_safe is not None:
            result["failed_stage"] = failed_safe["stage"]
            result["reason"] = failed_safe["reason"]
            return result

        for item in plan["path"]:
            if item["stage"].startswith("LIFT_"):
                continue
            row = self.move_to_pose(item["stage"], item["pose"])
            result["executed_waypoints"].append(row)
            if not row["move_success"]:
                result["failed_stage"] = item["stage"]
                result["reason"] = row["reason"]
                return result
            if item["stage"] == "PREGRASP":
                thumb_tuck = self.execute_thumb_tuck()
                result["grasp_action"]["thumb_tuck"] = thumb_tuck
                if not thumb_tuck["success"]:
                    result["failed_stage"] = "LEFT_THUMB_TUCK"
                    result["reason"] = thumb_tuck
                    return result

        grasp_force = self.execute_grasp_force()
        result["grasp_action"]["grasp_force"] = grasp_force
        result["grasp_action"]["success"] = grasp_force["success"]
        if not grasp_force["success"]:
            result["failed_stage"] = "LEFT_GRASP_FORCE"
            result["reason"] = grasp_force
            return result

        time.sleep(0.5)
        for item in plan["lift_waypoints"]:
            row = self.move_to_pose(item["stage"], item["pose"])
            result["lift"].append(row)
            if not row["move_success"]:
                result["failed_stage"] = item["stage"]
                result["reason"] = row["reason"]
                return result

        result["success"] = True
        return result

    def grasp_world_xyz(self, nut_world_xyz: list[float], yaw_offset: float = 0.0, *, execute: bool = True) -> dict[str, Any]:
        plan = self.build_plan(nut_world_xyz, yaw_offset=yaw_offset)
        preflight = self.preflight_grasp_chain(plan)
        result = {
            "success": False,
            "nut_world_xyz": plan["nut_world_xyz"],
            "nut_left_base_xyz": plan["nut_left_base_xyz"],
            "grasp_pose": plan["grasp_pose"],
            "pregrasp_pose": plan["pregrasp_pose"],
            "descent_waypoints": plan["descent_waypoints"],
            "lift_waypoints": plan["lift_waypoints"],
            "preflight": preflight["checks"],
            "failed_stage": preflight["failed_stage"],
            "reason": preflight["reason"],
            "plan": plan,
            "execution": None,
        }
        if not preflight["success"] or not execute:
            return result
        execution = self.execute(plan)
        result["execution"] = execution
        result["success"] = execution["success"]
        result["failed_stage"] = execution["failed_stage"]
        result["reason"] = execution["reason"]
        return result
