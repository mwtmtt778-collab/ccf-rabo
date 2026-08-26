#!/usr/bin/env python3
"""Read-only probe for isolating multi-device SDK initialization effects."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS  # noqa: E402

DEVICE_ORDER = ("right_arm", "right_hand", "left_arm", "left_hand")
DEVICE_LABELS = {name: name.upper() for name in DEVICE_ORDER}


def parse_devices(value: str) -> list[str]:
    devices = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("--devices must not be empty")
    invalid = [item for item in devices if item not in DEVICE_ORDER]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown device(s): {invalid}; choose from {DEVICE_ORDER}")
    return devices


def create_device(name: str) -> object:
    from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

    if name == "right_arm":
        return LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
    if name == "left_arm":
        return LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
    if name == "right_hand":
        return LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim")
    return LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe device initialization without robot actuation.")
    parser.add_argument(
        "--devices",
        required=True,
        type=parse_devices,
        help="Comma-separated devices: right_arm,right_hand,left_arm,left_hand.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Wait 8s after each device; wait 15s after the final device.",
    )
    args = parser.parse_args(argv)

    # Keep references alive for the complete observation window. No SDK method
    # is called after construction; the external camera monitor observes each
    # initialization boundary for external observation.
    devices: list[object] = []
    try:
        for index, name in enumerate(args.devices):
            if args.staged and index:
                print("================================", flush=True)
                print(f"NEXT DEVICE: {DEVICE_LABELS[name]}", flush=True)
                print("================================", flush=True)
            label = DEVICE_LABELS[name]
            print(f"[DEVICE_INIT] START device={label}", flush=True)
            started = time.monotonic()
            devices.append(create_device(name))
            elapsed_ms = (time.monotonic() - started) * 1000.0
            print(f"[DEVICE_INIT] DONE device={label} elapsed_ms={elapsed_ms:.1f}", flush=True)
            if args.staged:
                time.sleep(15.0 if index == len(args.devices) - 1 else 8.0)
        if not args.staged:
            time.sleep(15.0)
    except BaseException as exc:
        print(f"[DEVICE_INIT] ERROR exception={exc!r}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
