#!/usr/bin/env python3
"""Probe the sequential RIGHT_READY_1 -> RIGHT_READY_2 boundary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.three_nut_expert.config import DEVICE_IDS, KNOWN_FIXED_NUT_WORLD_POSE  # noqa: E402
from agents.three_nut_expert.expert import compute_right_approach_pose, compute_right_grasp_pose, pose_to_list  # noqa: E402

RIGHT_READY_1 = [-1.57, -1.5, 0.0, -1.57, 0.0, -1.0, 0.0]
RIGHT_READY_2 = [0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0]


def settle(arm: object, label: str, before: list[float], timeout_s: float = 15.0) -> bool:
    previous = list(before)
    started = time.monotonic()
    movement_started = False
    stable_count = 0
    while time.monotonic() - started < timeout_s:
        time.sleep(0.2)
        try:
            current = [float(v) for v in list(arm.get_joint_angles())[:7]]
        except BaseException as exc:
            print(f"[PROBE] {label}_READ_ERROR {exc!r}", flush=True)
            return False
        delta = max(abs(a - b) for a, b in zip(current, previous))
        if delta > 0.003:
            movement_started = True
            stable_count = 0
        elif delta < 0.002 and (movement_started or time.monotonic() - started >= 1.0):
            stable_count += 1
        else:
            stable_count = 0
        previous = current
        if stable_count >= 3:
            print(f"[PROBE] {label}_SETTLED elapsed={time.monotonic() - started:.3f}", flush=True)
            return True
    print(f"[PROBE] {label}_TIMEOUT", flush=True)
    return False


def run_motion(arm: object, target: list[float], label: str, blocking: bool) -> bool:
    try:
        before = [float(v) for v in list(arm.get_joint_angles())[:7]]
        started = time.monotonic_ns()
        result = arm.move_joints(target, blocking=blocking)
        duration_ms = (time.monotonic_ns() - started) / 1e6
        print(f"[PROBE] {label} call_duration_ms={duration_ms:.3f} result={result!r}", flush=True)
        if result is False:
            print(f"[PROBE] {label}_FAILED returned_false", flush=True)
            return False
        return settle(arm, label, before)
    except BaseException as exc:
        print(f"[PROBE] {label}_FAILED exception={exc!r}", flush=True)
        return False


def run_approach(arm: object, blocking: bool) -> bool:
    nut = KNOWN_FIXED_NUT_WORLD_POSE["C"]
    grasp = compute_right_grasp_pose((nut.x, nut.y, nut.z))
    target = pose_to_list(compute_right_approach_pose(grasp))
    try:
        pose_check = arm.pose_check(*target[:3], roll=target[3], pitch=target[4], yaw=target[5])
        print(f"[PROBE] RIGHT_APPROACH_POSE_CHECK {pose_check!r}", flush=True)
        if isinstance(pose_check, (list, tuple)) and pose_check and pose_check[0] is False:
            return False
        before = [float(v) for v in list(arm.get_joint_angles())[:7]]
        started = time.monotonic_ns()
        result = arm.move_to(*target[:3], roll=target[3], pitch=target[4], yaw=target[5], blocking=blocking)
        print(f"[PROBE] RIGHT_APPROACH call_duration_ms={(time.monotonic_ns() - started) / 1e6:.3f} result={result!r}", flush=True)
        if result is False:
            return False
        return settle(arm, "RIGHT_APPROACH", before)
    except BaseException as exc:
        print(f"[PROBE] RIGHT_APPROACH_FAILED exception={exc!r}", flush=True)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe sequential right-arm ready motions without other devices.")
    parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--blocking", action="store_true", help="Future comparison: use blocking=True.")
    args = parser.parse_args(argv)
    from rabo_robocap import LinkerArmA7

    arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    if not run_motion(arm, RIGHT_READY_1, "RIGHT_READY_1", args.blocking):
        return 1
    if args.stop_after == 1:
        time.sleep(15.0)
        return 0
    print("[PROBE] BEFORE_RIGHT_READY_2", flush=True)
    time.sleep(1.0)
    if not run_motion(arm, RIGHT_READY_2, "RIGHT_READY_2", args.blocking):
        return 1
    if args.stop_after == 2:
        time.sleep(15.0)
        return 0
    print("[PROBE] BEFORE_RIGHT_APPROACH", flush=True)
    time.sleep(1.0)
    if not run_approach(arm, args.blocking):
        return 1
    time.sleep(15.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
