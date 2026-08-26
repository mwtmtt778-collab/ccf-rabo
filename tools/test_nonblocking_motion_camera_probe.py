#!/usr/bin/env python3
"""Send one non-blocking right-arm motion for an external camera-rate probe.

After the command is submitted this process deliberately performs no further
arm SDK reads or commands; observe the camera from another terminal for the
eight-second wait window.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS, RIGHT_PRE_JOINTS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe non-blocking motion and optional joint polling.")
    parser.add_argument(
        "--poll-joints",
        action="store_true",
        help="Poll right_arm.get_joint_angles() every 0.2s during the 8s wait.",
    )
    args = parser.parse_args(argv)
    from rabo_robocap import LinkerArmA7

    target = list(RIGHT_PRE_JOINTS[0])
    print(f"target={target}", flush=True)
    right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    try:
        current = list(right_arm.get_joint_angles()[0:7])
        print(f"current_joints={current}", flush=True)
        result = right_arm.move_joints(target, blocking=False)
        print(f"move_joints(blocking=False) return={result!r}", flush=True)
        if not args.poll_joints:
            print("No further arm SDK calls; waiting 8 seconds.", flush=True)
            time.sleep(8.0)
        else:
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
