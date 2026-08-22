#!/usr/bin/env python3
"""Sequential B -> A -> C three-Nut vision closed-loop experiment.

This runner composes the already validated single-Nut components.  It does not
change perception, calibration, release behavior, hand actions, or the left
grasp template.  A failed phase stops the current sequence and is written to a
JSON report before robot clients are shut down.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "three_nut_closed_loop"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    HAND_OPEN,
    LEFT_PLACE_POSES,
    NUT_IDS,
    NUT_SPECS,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    Pose6,
)
from agents.three_nut_expert.expert import compute_right_grasp_pose, pose_to_list  # noqa: E402
from expert.left_nut_grasp_planner import LeftNutGraspPlanner, jsonable  # noqa: E402
from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import (  # noqa: E402
    LEFT_SAFE_LIFT_DELTA_Z_M,
    RIGHT_OBSERVATION_JOINTS,
    RIGHT_OBSERVATION_STABLE_WAIT_S,
    RIGHT_RELEASE_OPEN_WAIT_S,
    safe_lift_pose_from_grasp,
)
from tools.test_dual_closed_loop_v2_2 import (  # noqa: E402
    RIGHT_RELEASE_POSE,
    RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
    build_right_release_safe_height_pose,
)
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE  # noqa: E402
from tools.test_nut_camera_calibration import set_pose_with_retry  # noqa: E402
from tools.motion_monitor import MotionMonitor, extract_joint_target  # noqa: E402
from tools.probe_act_recording_sources import IntegratedRecordingProbe  # noqa: E402
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    TARGET_NUT_B_WORLD_POSE,
    detect_released_nut,
    execute_checked,
    make_right_bundle,
    move_checked,
    shutdown_bundle,
)


NUT_SEQUENCE = ("B", "A", "C")
RELEASE_TARGET_WORLD_XYZ = tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3])

# B remains the frozen successful pose.  A/C come directly from the existing
# three_nut_expert transform and nominal Nut specs; no new grasp search occurs.
RIGHT_GRASP_POSES: dict[str, Pose6] = {
    "B": RIGHT_NUT_B_GRASP_POSE,
    "A": compute_right_grasp_pose(NUT_SPECS["A"].nominal_pose),
    "C": compute_right_grasp_pose(NUT_SPECS["C"].nominal_pose),
}
RIGHT_GRASP_POSE_SOURCES = {
    "B": "tools.test_left_grasp_v1.RIGHT_NUT_B_GRASP_POSE (VERIFIED_FROZEN)",
    "A": "agents.three_nut_expert.expert.compute_right_grasp_pose + NUT_SPECS[A] (EXISTING_STAGED)",
    "C": "agents.three_nut_expert.expert.compute_right_grasp_pose + NUT_SPECS[C] (EXISTING_STAGED)",
}
PLACE_POSE_SOURCES = {
    key: f"agents.three_nut_expert.config.LEFT_PLACE_POSES[{key}]"
    for key in NUT_SEQUENCE
}
RIGHT_ACTION_WARN_AFTER_S = 10.0


class ThreeNutClosedLoopError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def compact_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))


def print_right_action_start(
    *,
    action_cn: str,
    action_en: str,
    target: Any,
    start_time: str,
    call_label: str | None = None,
    method: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    print("\n【右手动作开始】", flush=True)
    print(f"动作名称：{action_cn}", flush=True)
    print(f"动作英文：{action_en}", flush=True)
    if call_label:
        print(f"调用标签：{call_label}", flush=True)
    if method:
        print(f"调用方法：{method}", flush=True)
    if extra:
        for key, value in extra.items():
            print(f"{key}：{compact_json(value)}", flush=True)
    print(f"目标：{compact_json(target)}", flush=True)
    print(f"时间：{start_time}", flush=True)


def print_right_action_done(
    *,
    action_cn: str,
    action_en: str,
    elapsed_s: float,
    start_time: str,
    end_time: str,
    timed_out: bool,
    call_label: str | None = None,
    method: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    print("\n【右手动作完成】", flush=True)
    print(f"动作名称：{action_cn}", flush=True)
    print(f"动作英文：{action_en}", flush=True)
    if call_label:
        print(f"调用标签：{call_label}", flush=True)
    if method:
        print(f"调用方法：{method}", flush=True)
    if extra:
        for key, value in extra.items():
            print(f"{key}：{compact_json(value)}", flush=True)
    print(f"开始：{start_time}", flush=True)
    print(f"结束：{end_time}", flush=True)
    print(f"耗时：{elapsed_s:.2f} 秒", flush=True)
    print(f"是否超时：{'是' if timed_out else '否'}", flush=True)


def print_right_action_failed(
    *,
    action_cn: str,
    action_en: str,
    elapsed_s: float,
    start_time: str,
    end_time: str,
    timed_out: bool,
    error: BaseException,
    call_label: str | None = None,
    method: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    print("\n【右手动作失败】", flush=True)
    print(f"动作名称：{action_cn}", flush=True)
    print(f"动作英文：{action_en}", flush=True)
    if call_label:
        print(f"调用标签：{call_label}", flush=True)
    if method:
        print(f"调用方法：{method}", flush=True)
    if extra:
        for key, value in extra.items():
            print(f"{key}：{compact_json(value)}", flush=True)
    print(f"开始：{start_time}", flush=True)
    print(f"结束：{end_time}", flush=True)
    print(f"耗时：{elapsed_s:.2f} 秒", flush=True)
    print(f"是否超时：{'是' if timed_out else '否'}", flush=True)
    print(f"错误信息：{error!r}", flush=True)


def append_right_debug_trace(
    result: dict[str, Any],
    *,
    action_cn: str,
    action_en: str,
    status: str,
    start_time: str,
    target: Any,
    end_time: str = "",
    elapsed_s: float | str = "",
    error: str = "",
    call_label: str | None = None,
    method: str | None = None,
    timed_out: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "arm": "right",
        "动作": action_cn,
        "动作英文": action_en,
        "状态": status,
        "开始时间": start_time,
        "结束时间": end_time,
        "耗时_s": elapsed_s,
        "目标": target,
        "错误信息": error,
    }
    if call_label:
        entry["调用标签"] = call_label
    if method:
        entry["调用方法"] = method
    if timed_out is not None:
        entry["是否超过10秒"] = bool(timed_out)
    if extra:
        entry.update(extra)
    result.setdefault("debug_trace", []).append(jsonable(entry))


def right_phase_failure_note(phase: str) -> str:
    notes = {
        "RIGHT_GRASP": "右手抓取阶段执行异常，需结合最后一个右手动作判断是移动、拇指预收还是抓取施力未返回",
        "RIGHT_LIFT": "右手抬升阶段执行异常，需检查对应 RIGHT_LIFT 子动作是否阻塞",
        "RIGHT_RELEASE": "右手释放阶段执行异常，需检查释放位移动或张开手指是否未返回",
        "RIGHT_SAFE_RETREAT": "右手移动到释放后安全高度时未返回或执行异常",
        "RIGHT_OBSERVATION_AFTER_RELEASE": "右手释放后返回观察位置时未返回或执行异常",
        "RIGHT_OBSERVATION_BEFORE_GRASP": "右手抓取前返回观察位置时未返回或执行异常",
    }
    return notes.get(phase, f"{phase} 阶段执行异常")


def record_failure_detail(
    result: dict[str, Any],
    *,
    phase: str,
    exc: BaseException,
) -> None:
    result["failure_detail"] = {
        "失败阶段": phase,
        "中文说明": right_phase_failure_note(phase),
        "当前动作": result.get("current_right_action") or phase,
        "最后成功动作": result.get("last_success_right_action") or "",
        "异常时间": now_text(),
        "异常类型": type(exc).__name__,
        "异常信息": repr(exc),
    }


def run_right_action(
    result: dict[str, Any],
    *,
    action_cn: str,
    action_en: str,
    target: Any,
    call_label: str,
    method: str,
    fn: Any,
    extra: dict[str, Any] | None = None,
    monitor: MotionMonitor | None = None,
    monitor_arm: Any | None = None,
    monitor_phase_name: str | None = None,
    monitor_target_joint: list[float] | None = None,
    monitor_target_ee_pose: list[float] | None = None,
) -> Any:
    start_time = now_text()
    start_monotonic = time.monotonic()
    timed_out_event = threading.Event()
    previous_action = result.get("current_right_action")
    result["current_right_action"] = call_label
    append_right_debug_trace(
        result,
        action_cn=action_cn,
        action_en=action_en,
        status="开始",
        start_time=start_time,
        target=target,
        call_label=call_label,
        method=method,
        extra=extra,
    )
    print_right_action_start(
        action_cn=action_cn,
        action_en=action_en,
        target=target,
        start_time=start_time,
        call_label=call_label,
        method=method,
        extra=extra,
    )

    def warn_if_still_running() -> None:
        timed_out_event.set()
        print("\n【警告】", flush=True)
        print("右手动作超过10秒未返回：", flush=True)
        print(f"动作名称：{action_cn}", flush=True)
        print(f"动作英文：{action_en}", flush=True)
        print(f"当前动作：{call_label}", flush=True)
        print("疑似：", flush=True)
        print("- 控制接口阻塞", flush=True)
        print("- 机器人状态未更新", flush=True)
        print("- move命令未完成", flush=True)

    timer = threading.Timer(RIGHT_ACTION_WARN_AFTER_S, warn_if_still_running)
    timer.daemon = True
    timer.start()
    monitor_started = False
    if monitor is not None and monitor.enabled and monitor_arm is not None:
        try:
            monitor.start_motion_monitor(
                monitor_phase_name or call_label,
                monitor_arm,
                target_joint=monitor_target_joint,
                target_ee_pose=monitor_target_ee_pose,
                command_method=method,
            )
            monitor_started = True
        except Exception as monitor_exc:
            print(f"【MotionMonitor警告】启动失败：{monitor_exc!r}", flush=True)
    try:
        if monitor_started and monitor is not None and monitor_target_joint is not None:
            monitor.set_target_joint(monitor_target_joint, source="caller_before_motion")
        value = fn()
    except BaseException as exc:
        timer.cancel()
        end_time = now_text()
        elapsed_s = time.monotonic() - start_monotonic
        timed_out = timed_out_event.is_set() or elapsed_s >= RIGHT_ACTION_WARN_AFTER_S
        done_extra = dict(extra or {})
        monitor_result = None
        if monitor_started and monitor is not None and monitor.enabled:
            try:
                monitor_result = monitor.stop_motion_monitor(status="FAILED", error_info=repr(exc))
            except Exception as monitor_exc:
                done_extra["motion_monitor_error"] = repr(monitor_exc)
                print(f"【MotionMonitor警告】停止失败：{monitor_exc!r}", flush=True)
        if monitor_result:
            done_extra["motion_monitor"] = monitor_result
        if "move_joints" in method:
            done_extra.setdefault("move_joints调用前时间", start_time)
            done_extra["move_joints返回时间"] = end_time
        append_right_debug_trace(
            result,
            action_cn=action_cn,
            action_en=action_en,
            status="失败",
            start_time=start_time,
            end_time=end_time,
            elapsed_s=round(elapsed_s, 3),
            target=target,
            error=repr(exc),
            call_label=call_label,
            method=method,
            timed_out=timed_out,
            extra=done_extra,
        )
        print_right_action_failed(
            action_cn=action_cn,
            action_en=action_en,
            elapsed_s=elapsed_s,
            start_time=start_time,
            end_time=end_time,
            timed_out=timed_out,
            error=exc,
            call_label=call_label,
            method=method,
            extra=done_extra,
        )
        raise
    else:
        timer.cancel()
        end_time = now_text()
        elapsed_s = time.monotonic() - start_monotonic
        timed_out = timed_out_event.is_set() or elapsed_s >= RIGHT_ACTION_WARN_AFTER_S
        done_extra = dict(extra or {})
        monitor_result = None
        if monitor_started and monitor is not None and monitor.enabled:
            try:
                monitor_result = monitor.stop_motion_monitor(status="COMPLETED")
            except Exception as monitor_exc:
                done_extra["motion_monitor_error"] = repr(monitor_exc)
                print(f"【MotionMonitor警告】停止失败：{monitor_exc!r}", flush=True)
        if monitor_result:
            done_extra["motion_monitor"] = monitor_result
        if "move_joints" in method:
            done_extra.setdefault("move_joints调用前时间", start_time)
            done_extra["move_joints返回时间"] = end_time
        append_right_debug_trace(
            result,
            action_cn=action_cn,
            action_en=action_en,
            status="完成",
            start_time=start_time,
            end_time=end_time,
            elapsed_s=round(elapsed_s, 3),
            target=target,
            call_label=call_label,
            method=method,
            timed_out=timed_out,
            extra=done_extra,
        )
        result["last_success_right_action"] = call_label
        result["current_right_action"] = previous_action
        print_right_action_done(
            action_cn=action_cn,
            action_en=action_en,
            elapsed_s=elapsed_s,
            start_time=start_time,
            end_time=end_time,
            timed_out=timed_out,
            call_label=call_label,
            method=method,
            extra=done_extra,
        )
        return value


def move_right_checked_debug(
    result: dict[str, Any],
    *,
    action_cn: str,
    action_en: str,
    label: str,
    right_arm: Any,
    pose: Pose6,
    monitor: MotionMonitor | None = None,
) -> Any:
    pose_values = pose_to_list(pose)
    monitor_target_joint = None
    monitor_extra = None
    if monitor is not None and monitor.enabled:
        monitor_target_joint, monitor_extra = pose_check_target_joint(right_arm, pose)
    return run_right_action(
        result,
        action_cn=action_cn,
        action_en=action_en,
        target=pose_values,
        call_label=label,
        method="move_checked",
        fn=lambda: move_checked(label, right_arm, pose),
        monitor=monitor,
        monitor_arm=right_arm,
        monitor_phase_name=label,
        monitor_target_joint=monitor_target_joint,
        monitor_target_ee_pose=pose_values,
        extra={"motion_monitor_target_joint_lookup": monitor_extra} if monitor_extra else None,
    )


def execute_right_checked_debug(
    result: dict[str, Any],
    action_cn: str,
    action_en: str,
    label: str,
    target: Any,
    device: Any,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return run_right_action(
        result,
        action_cn=action_cn,
        action_en=action_en,
        target=target,
        call_label=label,
        method=f"execute_checked.{method}",
        fn=lambda: execute_checked(label, device, method, *args, **kwargs),
    )


def validate_static_contract(sequence: tuple[str, ...] = NUT_SEQUENCE) -> None:
    if NUT_SEQUENCE != ("B", "A", "C"):
        raise ThreeNutClosedLoopError("Nut sequence must remain B,A,C")
    missing_ids = [key for key in sequence if not NUT_IDS.get(key)]
    missing_grasps = [key for key in sequence if key not in RIGHT_GRASP_POSES]
    missing_places = [key for key in sequence if key not in LEFT_PLACE_POSES]
    if missing_ids:
        raise ThreeNutClosedLoopError(f"missing real entity IDs for: {missing_ids}")
    if missing_grasps:
        raise ThreeNutClosedLoopError(f"missing existing right grasp poses for: {missing_grasps}")
    if missing_places:
        raise ThreeNutClosedLoopError(f"missing existing place poses for: {missing_places}")
    if pose_to_list(RIGHT_GRASP_POSES["B"]) != pose_to_list(RIGHT_NUT_B_GRASP_POSE):
        raise ThreeNutClosedLoopError("Nut B frozen grasp pose was changed")


def parse_sequence_arg(value: str) -> tuple[str, ...]:
    cleaned = value.replace(" ", "").upper()
    if not cleaned:
        raise ThreeNutClosedLoopError("--sequence must not be empty")
    if "," in cleaned:
        sequence = tuple(item for item in cleaned.split(",") if item)
    else:
        sequence = tuple(cleaned)
    invalid = [key for key in sequence if key not in NUT_SEQUENCE]
    if invalid:
        raise ThreeNutClosedLoopError(f"--sequence contains invalid nut IDs: {invalid}")
    return sequence


def pose_check_target_joint(arm: Any, pose: Pose6) -> tuple[list[float] | None, dict[str, Any]]:
    pose_values = pose_to_list(pose)
    if not hasattr(arm, "pose_check"):
        return None, {"ok": False, "reason": "right_arm.pose_check unavailable"}
    try:
        raw = arm.pose_check(
            pose_values[0],
            pose_values[1],
            pose_values[2],
            roll=pose_values[3],
            pitch=pose_values[4],
            yaw=pose_values[5],
        )
    except Exception as exc:
        return None, {"ok": False, "reason": repr(exc), "source": "right_arm.pose_check"}
    target_joint = extract_joint_target(raw)
    return target_joint, {
        "ok": target_joint is not None,
        "source": "right_arm.pose_check",
        "raw": jsonable(raw),
        "target_joint_found": target_joint is not None,
    }


def make_pose_setter() -> Any:
    from rabo_dev_kit import SetEntityPose

    return SetEntityPose(world=WORLD_ID)


def shutdown_pose_setter(pose_setter: Any | None) -> None:
    if pose_setter is not None and hasattr(pose_setter, "shutdown"):
        try:
            pose_setter.shutdown()
        except Exception as exc:
            print(f"pose setter shutdown warning: {exc!r}")


def new_nut_result(key: str) -> dict[str, Any]:
    return {
        "id": key,
        "entity_id": NUT_IDS[key],
        "reset_success": False,
        "vision_position": None,
        "detected_position": None,
        "right_observation_before_grasp": False,
        "right_grasp": False,
        "lift_success": False,
        "right_release": False,
        "right_safe_retreat": False,
        "right_observation_after_release": False,
        "released_nut_position": None,
        "vision_after_release": False,
        "left_grasp": False,
        "left_safe_lift": False,
        "place": False,
        "success": False,
        "failed_phase": None,
        "failure_code": None,
        "failure_reason": None,
        "failure_detail": None,
        "interrupted": False,
        "phases": [],
        "debug_trace": [],
        "current_right_action": None,
        "last_success_right_action": None,
    }


def add_phase(
    result: dict[str, Any],
    phase: str,
    success: bool,
    *,
    failure_reason: str | None = None,
    **details: Any,
) -> None:
    result["phases"].append(
        jsonable(
            {
                "phase": phase,
                "success": bool(success),
                "timestamp": now_text(),
                "failure_reason": failure_reason,
                **details,
            }
        )
    )
    print(f"\n[{result['id']}] {phase}: {'PASS' if success else 'FAIL'}")
    if failure_reason:
        print(f"failure_reason: {failure_reason}")


def write_report(report: dict[str, Any], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def detect_nut(target_xyz: tuple[float, float, float], radius: float) -> dict[str, Any]:
    """Use the existing detector; target XYZ is candidate selection only."""
    return detect_released_nut(target_xyz, radius)


def move_right_to_observation(
    result: dict[str, Any],
    right_bundle: Any,
    label: str,
    *,
    monitor: MotionMonitor | None = None,
) -> list[list[float]]:
    executed: list[list[float]] = []
    for index, joints in enumerate(RIGHT_OBSERVATION_JOINTS, start=1):
        values = list(joints)
        call_label = f"{label}_{index}"
        extra = {
            "第几个关节目标点": index,
            "目标joint数值": values,
            "move_joints调用前时间": now_text(),
        }
        run_right_action(
            result,
            action_cn="右手返回观察位置",
            action_en=label,
            target=values,
            call_label=call_label,
            method="execute_checked.move_joints",
            fn=lambda values=values, call_label=call_label: execute_checked(
                call_label,
                right_bundle.right_arm,
                "move_joints",
                values,
            ),
            extra=extra,
            monitor=monitor,
            monitor_arm=right_bundle.right_arm,
            monitor_phase_name=call_label,
            monitor_target_joint=values,
        )
        executed.append(values)
    return executed


def execute_one_nut(
    key: str,
    pose_setter: Any,
    devices: dict[str, Any],
    *,
    settle_after_release_s: float,
    vision_radius_m: float,
    monitor: MotionMonitor | None = None,
) -> dict[str, Any]:
    result = new_nut_result(key)
    right_bundle = devices["right"]
    current_phase = "RESET"
    phase_recorded = False
    try:
        print("\n================================")
        print(f"Nut {key}")
        print("================================")

        reset_pose = pose_to_list(NUT_SPECS[key].nominal_pose)
        reset = set_pose_with_retry(pose_setter, NUT_IDS[key], reset_pose)
        if not reset.get("ok"):
            raise ThreeNutClosedLoopError(f"SetEntityPose fallback failed: {reset.get('error')}")
        result["reset_success"] = True
        add_phase(
            result,
            current_phase,
            True,
            mode="SET_ENTITY_POSE_FALLBACK",
            entity_id=NUT_IDS[key],
            nominal_pose=reset_pose,
            reset_result=reset,
        )
        phase_recorded = True

        current_phase = "RIGHT_OBSERVATION_BEFORE_GRASP"
        phase_recorded = False
        execute_right_checked_debug(
            result,
            "右手初始张开",
            current_phase,
            "RIGHT_HAND_OPEN_INITIAL",
            list(HAND_OPEN),
            right_bundle.right_hand,
            "clench",
            *list(HAND_OPEN),
        )
        observation_path = move_right_to_observation(
            result,
            right_bundle,
            "RIGHT_OBSERVATION_BEFORE_GRASP",
            monitor=monitor,
        )
        time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
        result["right_observation_before_grasp"] = True
        add_phase(result, current_phase, True, joints=observation_path)
        phase_recorded = True
        current_phase = "VISION_BEFORE_GRASP"
        phase_recorded = False
        nominal_xyz = tuple(float(value) for value in reset_pose[:3])
        initial_vision = detect_nut(nominal_xyz, vision_radius_m)
        if not initial_vision.get("detected") or not initial_vision.get("nut_world_xyz"):
            raise ThreeNutClosedLoopError(
                f"initial point-cloud detection failed: {initial_vision.get('error', 'candidate outside ROI')}"
            )
        detected_position = [float(value) for value in initial_vision["nut_world_xyz"]]
        result["vision_position"] = detected_position
        result["detected_position"] = detected_position
        add_phase(
            result,
            current_phase,
            True,
            detected_position=detected_position,
            detector_result=initial_vision,
        )
        phase_recorded = True

        current_phase = "RIGHT_GRASP"
        phase_recorded = False
        grasp_pose = RIGHT_GRASP_POSES[key]
        move_right_checked_debug(
            result,
            action_cn=f"右手移动到 Nut {key} 抓取位",
            action_en=current_phase,
            label=f"RIGHT_NUT_{key}_GRASP_POSE",
            right_arm=right_bundle.right_arm,
            pose=grasp_pose,
            monitor=monitor,
        )
        execute_right_checked_debug(
            result,
            "右手拇指预收",
            current_phase,
            "RIGHT_THUMB_TUCK",
            {"thumb_rotation": 1.0},
            right_bundle.right_hand,
            "clench",
            thumb_rotation=1.0,
        )
        execute_right_checked_debug(
            result,
            "右手抓取施力",
            current_phase,
            "RIGHT_GRASP_FORCE",
            RIGHT_GRASP_FORCE,
            right_bundle.right_hand,
            "grasp_force",
            **RIGHT_GRASP_FORCE,
        )
        if DEFAULT_HOLD_AFTER_GRASP_S > 0:
            time.sleep(DEFAULT_HOLD_AFTER_GRASP_S)
        result["right_grasp"] = True
        add_phase(
            result,
            current_phase,
            True,
            grasp_pose=pose_to_list(grasp_pose),
            grasp_pose_source=RIGHT_GRASP_POSE_SOURCES[key],
        )
        phase_recorded = True

        current_phase = "RIGHT_LIFT"
        phase_recorded = False
        for index, lift_pose in enumerate(RIGHT_LIFT_POSES, start=1):
            move_right_checked_debug(
                result,
                action_cn=f"右手抬升第 {index} 段",
                action_en=current_phase,
                label=f"RIGHT_LIFT_{index}",
                right_arm=right_bundle.right_arm,
                pose=lift_pose,
                monitor=monitor,
            )
        result["lift_success"] = True
        add_phase(result, current_phase, True, lift_poses=[pose_to_list(pose) for pose in RIGHT_LIFT_POSES])
        phase_recorded = True

        current_phase = "RIGHT_RELEASE"
        phase_recorded = False
        move_right_checked_debug(
            result,
            action_cn="右手移动到释放位",
            action_en=current_phase,
            label="RIGHT_RELEASE_POSE",
            right_arm=right_bundle.right_arm,
            pose=RIGHT_RELEASE_POSE,
            monitor=monitor,
        )
        execute_right_checked_debug(
            result,
            "右手释放张开",
            current_phase,
            "RIGHT_RELEASE_OPEN",
            list(HAND_OPEN),
            right_bundle.right_hand,
            "clench",
            *list(HAND_OPEN),
        )
        release_timestamp = now_text()
        release_monotonic = time.monotonic()
        result["right_release"] = True
        add_phase(
            result,
            current_phase,
            True,
            release_pose=pose_to_list(RIGHT_RELEASE_POSE),
            release_target_world_xyz=list(RELEASE_TARGET_WORLD_XYZ),
            release_timestamp=release_timestamp,
        )
        phase_recorded = True

        current_phase = "RIGHT_SAFE_RETREAT"
        phase_recorded = False
        time.sleep(RIGHT_RELEASE_OPEN_WAIT_S)
        safe_height_pose = build_right_release_safe_height_pose()
        move_right_checked_debug(
            result,
            action_cn="右手释放后移动到安全高度",
            action_en=current_phase,
            label="RIGHT_RELEASE_SAFE_HEIGHT",
            right_arm=right_bundle.right_arm,
            pose=safe_height_pose,
            monitor=monitor,
        )
        result["right_safe_retreat"] = True
        add_phase(
            result,
            current_phase,
            True,
            offset_z_m=RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
            safe_height_pose=pose_to_list(safe_height_pose),
        )
        phase_recorded = True

        current_phase = "RIGHT_OBSERVATION_AFTER_RELEASE"
        phase_recorded = False
        observation_path = move_right_to_observation(
            result,
            right_bundle,
            "RIGHT_OBSERVATION_AFTER_RELEASE",
            monitor=monitor,
        )
        time.sleep(RIGHT_OBSERVATION_STABLE_WAIT_S)
        result["right_observation_after_release"] = True
        add_phase(result, current_phase, True, joints=observation_path)
        phase_recorded = True

        current_phase = "WAIT_SETTLE"
        phase_recorded = False
        elapsed = time.monotonic() - release_monotonic
        remaining = max(0.0, settle_after_release_s - elapsed)
        time.sleep(remaining)
        add_phase(
            result,
            current_phase,
            True,
            minimum_release_to_vision_s=settle_after_release_s,
            remaining_wait_s=remaining,
        )
        phase_recorded = True
        current_phase = "VISION_AFTER_RELEASE"
        phase_recorded = False
        released_vision = detect_nut(RELEASE_TARGET_WORLD_XYZ, vision_radius_m)
        if not released_vision.get("detected") or not released_vision.get("nut_world_xyz"):
            raise ThreeNutClosedLoopError(
                "VISION_AFTER_RELEASE_FAILED: "
                f"{released_vision.get('error', 'candidate outside ROI')}"
            )
        released_position = [float(value) for value in released_vision["nut_world_xyz"]]
        result["vision_after_release"] = True
        result["released_nut_position"] = released_position
        add_phase(
            result,
            current_phase,
            True,
            released_nut_position=released_position,
            detector_result=released_vision,
        )
        phase_recorded = True

        current_phase = "LEFT_GRASP"
        phase_recorded = False
        if devices["left"] is None:
            devices["left"] = make_left_bundle()
        left_bundle = devices["left"]
        planner = LeftNutGraspPlanner(left_arm=left_bundle.left_arm, left_hand=left_bundle.left_hand)
        left_result = planner.grasp_world_xyz(released_position, execute=True)
        grasp_pose_left = left_result.get("grasp_pose")
        if not left_result.get("success"):
            raise ThreeNutClosedLoopError(
                f"left grasp failed at {left_result.get('failed_stage')}: {left_result.get('reason')}"
            )
        result["left_grasp"] = True
        add_phase(
            result,
            current_phase,
            True,
            nut_world_xyz=released_position,
            left_grasp_pose=grasp_pose_left,
            planner_result=left_result,
        )
        phase_recorded = True

        current_phase = "LEFT_SAFE_LIFT"
        phase_recorded = False
        safe_lift_pose = safe_lift_pose_from_grasp(grasp_pose_left)
        safe_lift_result = planner.move_to_pose("LEFT_SAFE_LIFT", safe_lift_pose)
        if not safe_lift_result.get("move_success"):
            raise ThreeNutClosedLoopError(
                f"left safe lift failed: {safe_lift_result.get('reason', 'move_to failed')}"
            )
        result["left_safe_lift"] = True
        add_phase(
            result,
            current_phase,
            True,
            lift_delta_z_m=LEFT_SAFE_LIFT_DELTA_Z_M,
            safe_lift_pose=safe_lift_pose,
            move_result=safe_lift_result,
        )
        phase_recorded = True

        current_phase = "LEFT_PLACE"
        phase_recorded = False
        place_pose = LEFT_PLACE_POSES[key]
        place_pose_values = pose_to_list(place_pose)
        place_check = planner.pose_check(place_pose_values)
        if not place_check.get("pass"):
            raise ThreeNutClosedLoopError(f"place pose_check failed: {place_check.get('reason')}")
        move_checked(f"LEFT_PLACE_{key}", left_bundle.left_arm, place_pose)
        execute_checked("LEFT_RELEASE_OPEN", left_bundle.left_hand, "clench", *list(HAND_OPEN))
        result["place"] = True
        add_phase(
            result,
            current_phase,
            True,
            place_pose=place_pose_values,
            place_pose_source=PLACE_POSE_SOURCES[key],
            pose_check=place_check,
        )
        phase_recorded = True

        # The right arm already remains at observation while the left arm
        # completes grasp/place; record that invariant at task completion.
        current_phase = "RIGHT_FINAL_OBSERVATION"
        phase_recorded = False
        add_phase(
            result,
            current_phase,
            True,
            note="Right arm remained at RIGHT_OBSERVATION_JOINTS during left grasp/place.",
        )
        phase_recorded = True
        result["success"] = True
    except KeyboardInterrupt as exc:
        reason = repr(exc) or "KeyboardInterrupt()"
        if not phase_recorded:
            add_phase(result, current_phase, False, failure_reason=reason)
        record_failure_detail(result, phase=current_phase, exc=exc)
        result["failed_phase"] = current_phase
        result["failure_code"] = "USER_INTERRUPTED"
        result["failure_reason"] = reason
        result["interrupted"] = True
    except Exception as exc:
        reason = repr(exc)
        if not phase_recorded:
            add_phase(result, current_phase, False, failure_reason=reason)
        record_failure_detail(result, phase=current_phase, exc=exc)
        result["failed_phase"] = current_phase
        if current_phase == "VISION_AFTER_RELEASE":
            result["failure_code"] = "VISION_AFTER_RELEASE_FAILED"
        result["failure_reason"] = reason
    return result


def run_trial(
    trial_id: int,
    sequence: tuple[str, ...],
    pose_setter: Any,
    devices: dict[str, Any],
    args: argparse.Namespace,
    monitor: MotionMonitor | None,
) -> dict[str, Any]:
    trial = {
        "trial_id": trial_id,
        "sequence": list(sequence),
        "nut_results": [],
        "success": False,
        "failed_nut": None,
        "failed_phase": None,
        "failure_code": None,
        "failure_detail": None,
        "interrupted": False,
        "debug_trace": [],
    }
    for key in sequence:
        result = execute_one_nut(
            key,
            pose_setter,
            devices,
            settle_after_release_s=float(args.settle_after_release_s),
            vision_radius_m=float(args.vision_target_radius_m),
            monitor=monitor,
        )
        trial["nut_results"].append(result)
        trial["debug_trace"].extend(result.get("debug_trace", []))
        if not result["success"]:
            trial["failed_nut"] = key
            trial["failed_phase"] = result["failed_phase"]
            trial["failure_code"] = result["failure_code"]
            trial["failure_detail"] = result.get("failure_detail")
            trial["interrupted"] = result["interrupted"]
            return trial
    trial["success"] = True
    return trial


def run(args: argparse.Namespace) -> int:
    sequence = parse_sequence_arg(args.sequence)
    validate_static_contract(sequence)
    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"three_nut_trial_{started.strftime('%Y%m%d_%H%M%S')}.json"
    report: dict[str, Any] = {
        "experiment": "three_nut_closed_loop_v1",
        "timestamp": started.isoformat(timespec="seconds"),
        "status": "RUNNING",
        "sequence": list(sequence),
        "config": {
            "nut_ids": {key: NUT_IDS[key] for key in sequence},
            "nut_nominal_poses": {key: pose_to_list(NUT_SPECS[key].nominal_pose) for key in sequence},
            "right_grasp_poses": {key: pose_to_list(RIGHT_GRASP_POSES[key]) for key in sequence},
            "right_grasp_pose_sources": RIGHT_GRASP_POSE_SOURCES,
            "right_release_pose": pose_to_list(RIGHT_RELEASE_POSE),
            "release_target_world_xyz": list(RELEASE_TARGET_WORLD_XYZ),
            "right_release_safe_height_offset_z_m": RIGHT_RELEASE_SAFE_HEIGHT_OFFSET_Z,
            "right_observation_joints": [list(joints) for joints in RIGHT_OBSERVATION_JOINTS],
            "left_safe_lift_delta_z_m": LEFT_SAFE_LIFT_DELTA_Z_M,
            "left_place_poses": {key: pose_to_list(LEFT_PLACE_POSES[key]) for key in sequence},
            "place_pose_sources": PLACE_POSE_SOURCES,
            "settle_after_release_s": float(args.settle_after_release_s),
            "vision_target_radius_m": float(args.vision_target_radius_m),
            "headless": True,
            "configuration_warning": "A/C right grasps and place slots are existing staged values, not field-verified.",
        },
        "trials": [],
        "debug_trace": [],
        "overall_success": False,
        "failed_trial": None,
        "failed_nut": None,
        "failed_phase": None,
        "failure_code": None,
        "failure_detail": None,
        "failure_reason": None,
        "interrupted": False,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
    }

    pose_setter = None
    devices = {"right": None, "left": None}
    monitor = MotionMonitor(enabled=bool(args.monitor)) if args.monitor else None
    recording_probe: IntegratedRecordingProbe | None = None
    recording_probe_started = False
    if args.monitor:
        report["motion_monitor"] = {
            "enabled": True,
            "output_root": "reports/motion_monitor",
            "config_path": "tools/motion_monitor_config.json",
        }
    if args.recording_probe:
        probe_output_dir = (
            PROJECT_ROOT / "reports" / "act_recording_probe" / started.strftime("%Y%m%d_%H%M%S")
        )
        recording_probe = IntegratedRecordingProbe(
            output_dir=probe_output_dir,
            target_fps=float(args.recording_probe_fps),
        )
        report["recording_probe"] = {
            "enabled": True,
            "target_fps": float(args.recording_probe_fps),
            "output_dir": str(probe_output_dir.relative_to(PROJECT_ROOT)),
            "status": "WAITING_FOR_DEVICE_INITIALIZATION",
        }
    return_code = 1
    try:
        print("================================")
        print("Three Nut closed loop V1")
        print(f"Sequence: {' -> '.join(sequence)}")
        print("================================")
        pose_setter = make_pose_setter()
        devices["right"] = make_right_bundle()
        if recording_probe is not None:
            # The normal task initializes the left bundle lazily.  The optional
            # read-only probe needs all 26 state dimensions from its first frame,
            # so initialize the same existing bundle before starting the trial.
            devices["left"] = make_left_bundle()
            probe_start = recording_probe.start(
                left_arm=devices["left"].left_arm,
                right_arm=devices["right"].right_arm,
                left_hand=devices["left"].left_hand,
                right_hand=devices["right"].right_hand,
            )
            recording_probe_started = True
            report["recording_probe"].update(
                {
                    "status": "RUNNING",
                    "start_result": probe_start,
                }
            )
            print("\n【ACT录制源探针】已启动", flush=True)
            print(f"目标状态采样频率：{args.recording_probe_fps:g} Hz", flush=True)
            print(f"发现RGB相机：{probe_start.get('camera_count', 0)}/3", flush=True)
        for trial_id in range(1, args.trials + 1):
            print(f"\n######## Trial {trial_id}/{args.trials} ########")
            trial = run_trial(trial_id, sequence, pose_setter, devices, args, monitor)
            report["trials"].append(trial)
            report["debug_trace"].extend(trial.get("debug_trace", []))
            write_report(report, report_path)
            if not trial["success"]:
                report["failed_trial"] = trial_id
                report["failed_nut"] = trial["failed_nut"]
                report["failed_phase"] = trial["failed_phase"]
                report["failure_code"] = trial["failure_code"]
                report["failure_detail"] = trial.get("failure_detail")
                report["interrupted"] = trial["interrupted"]
                failed_result = trial["nut_results"][-1]
                report["failure_reason"] = failed_result["failure_reason"]
                break
        report["overall_success"] = len(report["trials"]) == args.trials and all(
            trial["success"] for trial in report["trials"]
        )
        if report["overall_success"]:
            report["status"] = "PASS"
            return_code = 0
        elif report["interrupted"]:
            report["status"] = "INTERRUPTED"
            return_code = 130
        else:
            report["status"] = "FAILED"
            return_code = 1
    except KeyboardInterrupt as exc:
        report["status"] = "INTERRUPTED"
        report["interrupted"] = True
        report["failure_code"] = "USER_INTERRUPTED"
        report["failed_phase"] = report["failed_phase"] or "INITIALIZE"
        report["failure_reason"] = repr(exc) or "KeyboardInterrupt()"
        report["failure_detail"] = {
            "失败阶段": report["failed_phase"],
            "中文说明": "初始化或 trial 外层执行被用户中断",
            "当前动作": report["failed_phase"],
            "最后成功动作": "",
            "异常时间": now_text(),
            "异常类型": type(exc).__name__,
            "异常信息": report["failure_reason"],
        }
        return_code = 130
    except Exception as exc:
        report["status"] = "FAILED"
        report["failed_phase"] = report["failed_phase"] or "INITIALIZE"
        report["failure_reason"] = repr(exc)
        report["failure_detail"] = {
            "失败阶段": report["failed_phase"],
            "中文说明": "初始化或 trial 外层执行异常",
            "当前动作": report["failed_phase"],
            "最后成功动作": "",
            "异常时间": now_text(),
            "异常类型": type(exc).__name__,
            "异常信息": report["failure_reason"],
        }
    finally:
        if recording_probe is not None:
            if not recording_probe_started:
                report["recording_probe"].update(
                    {
                        "status": "NOT_STARTED",
                        "reason": "V1 device initialization failed before the probe could start",
                    }
                )
            else:
                try:
                    probe_summary = recording_probe.stop(task_status=report["status"])
                    report["recording_probe"].update(
                        {
                            "status": "COMPLETED",
                            "summary": probe_summary,
                        }
                    )
                    print("\n【ACT录制源探针】采样完成", flush=True)
                    print(f"结论：{probe_summary.get('result')}", flush=True)
                    print(f"报告：{probe_summary.get('report_json')}", flush=True)
                except Exception as exc:
                    report["recording_probe"].update(
                        {
                            "status": "FAILED",
                            "error": repr(exc),
                            "failed_at": now_text(),
                        }
                    )
                    print(f"\n【ACT录制源探针】停止/保存失败：{exc!r}", flush=True)
        shutdown_bundle(devices["right"])
        shutdown_left_bundle(devices["left"])
        shutdown_pose_setter(pose_setter)
        report["finished_at"] = now_text()
        write_report(report, report_path)
        print(f"\nResult: {report['status']}")
        print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed B -> A -> C three-Nut visual closed-loop experiment."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run headless (default and only supported mode).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of complete B,A,C trials; stops after the first failed trial (default: 1).",
    )
    parser.add_argument(
        "--sequence",
        default="B,A,C",
        help="Nut execution sequence, e.g. B,A,C or A (default: B,A,C).",
    )
    parser.add_argument(
        "--settle-after-release-s",
        type=float,
        default=DEFAULT_SETTLE_AFTER_RELEASE_S,
        help=f"Minimum release-to-vision settling time (default: {DEFAULT_SETTLE_AFTER_RELEASE_S:g}s).",
    )
    parser.add_argument(
        "--vision-target-radius-m",
        type=float,
        default=DEFAULT_VISION_TARGET_RADIUS_M,
        help=f"Existing detector candidate-selection radius (default: {DEFAULT_VISION_TARGET_RADIUS_M:g}m).",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="enable motion diagnostic monitor",
    )
    parser.add_argument(
        "--recording-probe",
        "--probe",
        dest="recording_probe",
        action="store_true",
        help="enable the read-only ACT recording-source probe during the original task",
    )
    parser.add_argument(
        "--recording-probe-fps",
        type=float,
        default=30.0,
        help="target state sampling frequency for --recording-probe (default: 30Hz)",
    )
    args = parser.parse_args()
    try:
        parse_sequence_arg(args.sequence)
    except ThreeNutClosedLoopError as exc:
        parser.error(str(exc))
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    minimum_wait = RIGHT_RELEASE_OPEN_WAIT_S + RIGHT_OBSERVATION_STABLE_WAIT_S
    if args.settle_after_release_s < minimum_wait:
        parser.error(f"--settle-after-release-s must be at least {minimum_wait:g}s")
    if args.vision_target_radius_m <= 0:
        parser.error("--vision-target-radius-m must be > 0")
    if args.recording_probe_fps <= 0:
        parser.error("--recording-probe-fps must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
