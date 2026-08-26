#!/usr/bin/env python3
"""A/B, read-only initialization comparison for camera observation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def report_object(obj: object) -> None:
    cls = type(obj)
    print(f"[OBJECT_CREATED] class={cls.__name__} module={cls.__module__}", flush=True)


def official_init() -> list[object]:
    from agents.arm_hand_demo import (  # noqa: F401
        LEFT_ARM_ID, LEFT_HAND_ID, RIGHT_ARM_ID, RIGHT_HAND_ID, WORLD_ID,
    )
    from rabo_dev_kit import SetEntityPose
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

    objects: list[object] = []
    pose_setter = SetEntityPose(world=WORLD_ID)
    objects.append(pose_setter); report_object(pose_setter)
    for obj in (
        LinkerArmA7(robot_id=RIGHT_ARM_ID, mode="sim"),
        LinkerArmA7(robot_id=LEFT_ARM_ID, mode="sim"),
        LinkerHandO6Right(robot_id=RIGHT_HAND_ID, mode="sim"),
        LinkerHandO6Left(robot_id=LEFT_HAND_ID, mode="sim"),
    ):
        objects.append(obj); report_object(obj)
    return objects


def current_init() -> list[object]:
    import tools.test_three_nut_closed_loop_v2 as expert
    from agents.three_nut_expert.config import DEVICE_IDS
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right
    from tools.motion_monitor import MotionMonitor
    from agents.three_nut_expert.execution import ExecutionCoordinator

    objects: list[object] = []
    right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    report_object(right_arm); objects.append(right_arm)
    right_hand = LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim")
    report_object(right_hand); objects.append(right_hand)
    left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
    report_object(left_arm); objects.append(left_arm)
    left_hand = LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim")
    report_object(left_hand); objects.append(left_hand)
    right_bundle = SimpleNamespace(right_arm=right_arm, right_hand=right_hand)
    left_bundle = SimpleNamespace(left_arm=left_arm, left_hand=left_hand)
    report_object(right_bundle); objects.append(right_bundle)
    report_object(left_bundle); objects.append(left_bundle)
    coordinator = ExecutionCoordinator()
    report_object(coordinator); objects.append(coordinator)
    monitor = MotionMonitor(enabled=False)
    report_object(monitor); objects.append(monitor)
    runner = expert.ExpertStateRunner(1, monitor, ROOT / "reports" / "startup_trace" / "probe_report.json", coordinator=coordinator)
    report_object(runner); objects.append(runner)
    return objects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare official/current initialization without robot motion.")
    parser.add_argument("--mode", choices=("official", "current"), required=True)
    args = parser.parse_args(argv)
    started = time.monotonic_ns()
    objects = official_init() if args.mode == "official" else current_init()
    label = "OFFICIAL" if args.mode == "official" else "CURRENT"
    print(f"[{label}_INIT] READY", flush=True)
    print(f"{label.lower()} object count={len(objects)}", flush=True)
    time.sleep(20.0)
    print(f"[{label}_INIT] elapsed_ms={(time.monotonic_ns() - started) / 1e6:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
