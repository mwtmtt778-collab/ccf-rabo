#!/usr/bin/env python3
"""Probe one verified Cartesian move_to(blocking=False) during camera-rate tests."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS, KNOWN_FIXED_NUT_WORLD_POSE  # noqa: E402
from agents.three_nut_expert.expert import compute_right_approach_pose, compute_right_grasp_pose, pose_to_list  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe non-blocking Cartesian motion and external camera rate.")
    args = parser.parse_args(argv)
    from rabo_robocap import LinkerArmA7

    # This is the exact C-only Expert approach target derived from the
    # verified fixed C scene pose; no new geometry is introduced.
    nut_xyz = KNOWN_FIXED_NUT_WORLD_POSE["C"]
    grasp_pose = compute_right_grasp_pose((nut_xyz.x, nut_xyz.y, nut_xyz.z))
    target = pose_to_list(compute_right_approach_pose(grasp_pose))
    print(f"target={target}", flush=True)
    right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    try:
        try:
            pose_check = right_arm.pose_check(
                *target[:3], roll=target[3], pitch=target[4], yaw=target[5]
            )
            print(f"pose_check={pose_check!r}", flush=True)
        except BaseException as exc:
            print(f"pose_check_exception={exc!r}", flush=True)
            return 1
        result = right_arm.move_to(
            *target[:3], roll=target[3], pitch=target[4], yaw=target[5], blocking=False
        )
        print(f"move_to(blocking=False) return={result!r}", flush=True)
        poll_count = 0
        read_error_count = 0
        deadline = time.monotonic() + 8.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.2, remaining))
            if time.monotonic() >= deadline:
                break
            try:
                right_arm.get_joint_angles()
                poll_count += 1
            except BaseException:
                read_error_count += 1
        print(f"poll_count={poll_count}", flush=True)
        print(f"read_error_count={read_error_count}", flush=True)
    except BaseException as exc:
        print(f"probe failed: {exc!r}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
