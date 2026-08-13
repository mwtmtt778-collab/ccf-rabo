#!/usr/bin/env python3
"""Read-only robot state stream test for the Rabo sim runtime."""

from __future__ import annotations

import argparse
import ast
import faulthandler
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_HAND_DEMO = PROJECT_ROOT / "agents" / "arm_hand_demo" / "__init__.py"

REQUIRED_IDS = ("LEFT_ARM_ID", "RIGHT_ARM_ID", "LEFT_HAND_ID", "RIGHT_HAND_ID")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return repr(value)


def _format_value(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False)


def load_robot_ids(path: Path = ARM_HAND_DEMO) -> dict[str, str]:
    """Read robot IDs from source without importing arm_hand_demo."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ids: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in REQUIRED_IDS and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            ids[name] = node.value.value
    missing = [name for name in REQUIRED_IDS if name not in ids]
    if missing:
        raise RuntimeError(f"Missing robot IDs in {path}: {missing}")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only 30 second Rabo robot state stream test.")
    parser.add_argument("--duration", type=float, default=30.0, help="Stream duration in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Print interval in seconds.")
    return parser


def describe_shutdown(label: str, device: Any) -> bool:
    has_shutdown = hasattr(device, "shutdown")
    print(f"{label} has shutdown: {has_shutdown}")
    if not has_shutdown:
        return False
    try:
        signature = inspect.signature(device.shutdown)
        print(f"{label} shutdown signature: {signature}")
    except Exception as exc:
        print(f"{label} shutdown signature: unavailable ({repr(exc)})")
    return True


def init_device(label: str, factory: Callable[[], Any]) -> Any | None:
    print(f"initializing {label}...")
    try:
        device = factory()
        print(f"{label}: CONNECTED")
        describe_shutdown(label, device)
        return device
    except Exception as exc:
        print(f"{label}: INIT_ERROR {repr(exc)}")
        return None


def read_method(label: str, device: Any | None, method_name: str) -> tuple[bool, Any]:
    if device is None:
        return False, "DEVICE_NOT_CONNECTED"
    try:
        method = getattr(device, method_name)
        return True, method()
    except Exception as exc:
        return False, f"ERROR {repr(exc)}"


def print_sample(
    index: int,
    left_arm: Any | None,
    right_arm: Any | None,
    left_hand: Any | None,
    right_hand: Any | None,
) -> int:
    errors = 0
    print()
    print(f"[{index:03d}]")

    ok, value = read_method("LEFT_ARM", left_arm, "get_joint_angles")
    errors += 0 if ok else 1
    print(f"LEFT_ARM qpos={_format_value(value)}")
    ok, value = read_method("LEFT_ARM", left_arm, "get_pose")
    errors += 0 if ok else 1
    print(f"LEFT_ARM pose={_format_value(value)}")
    print()

    ok, value = read_method("RIGHT_ARM", right_arm, "get_joint_angles")
    errors += 0 if ok else 1
    print(f"RIGHT_ARM qpos={_format_value(value)}")
    ok, value = read_method("RIGHT_ARM", right_arm, "get_pose")
    errors += 0 if ok else 1
    print(f"RIGHT_ARM pose={_format_value(value)}")
    print()

    ok, value = read_method("LEFT_HAND", left_hand, "get_joint_angles")
    errors += 0 if ok else 1
    print(f"LEFT_HAND joints={_format_value(value)}")
    ok, value = read_method("LEFT_HAND", left_hand, "get_clench")
    errors += 0 if ok else 1
    print(f"LEFT_HAND clench={_format_value(value)}")
    print()

    ok, value = read_method("RIGHT_HAND", right_hand, "get_joint_angles")
    errors += 0 if ok else 1
    print(f"RIGHT_HAND joints={_format_value(value)}")
    ok, value = read_method("RIGHT_HAND", right_hand, "get_clench")
    errors += 0 if ok else 1
    print(f"RIGHT_HAND clench={_format_value(value)}")

    return errors


def shutdown_device(label: str, device: Any | None) -> bool:
    if device is None:
        print(f"shutting down {label}... SKIPPED")
        return True

    print(f"shutting down {label}...")
    if not hasattr(device, "shutdown"):
        print(f"{label} shutdown UNAVAILABLE")
        return False

    try:
        device.shutdown()
        print(f"{label} shutdown OK")
        return True
    except Exception as exc:
        print(f"{label} shutdown ERROR {repr(exc)}")
        return False


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = build_parser().parse_args(argv)

    ids = load_robot_ids()
    print("Loaded robot IDs from:", ARM_HAND_DEMO)
    for name in REQUIRED_IDS:
        print(f"{name}: {ids[name]}")

    try:
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right
    except Exception as exc:
        print("SDK_IMPORT_ERROR:", repr(exc))
        return 0

    left_arm = None
    right_arm = None
    left_hand = None
    right_hand = None
    total_errors = 0
    clean_shutdown = True
    completed_stream = False

    try:
        left_arm = init_device("LEFT ARM", lambda: LinkerArmA7(robot_id=ids["LEFT_ARM_ID"], mode="sim"))
        right_arm = init_device("RIGHT ARM", lambda: LinkerArmA7(robot_id=ids["RIGHT_ARM_ID"], mode="sim"))
        left_hand = init_device("LEFT HAND", lambda: LinkerHandO6Left(robot_id=ids["LEFT_HAND_ID"], mode="sim"))
        right_hand = init_device("RIGHT HAND", lambda: LinkerHandO6Right(robot_id=ids["RIGHT_HAND_ID"], mode="sim"))

        start = time.monotonic()
        next_sample = start
        sample_index = 1
        while time.monotonic() - start < args.duration:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(0.05, next_sample - now))
                continue

            total_errors += print_sample(sample_index, left_arm, right_arm, left_hand, right_hand)
            sample_index += 1
            next_sample += args.interval

        completed_stream = total_errors == 0
        if completed_stream:
            print()
            print("STATE_STREAM: PASS")
        else:
            print()
            print(f"STATE_STREAM: FAIL errors={total_errors}")

    except KeyboardInterrupt:
        print()
        print("Interrupted")
    finally:
        print()
        print("BEGIN CLEAN SHUTDOWN")
        shutdown_results = [
            shutdown_device("LEFT HAND", left_hand),
            shutdown_device("RIGHT HAND", right_hand),
            shutdown_device("LEFT ARM", left_arm),
            shutdown_device("RIGHT ARM", right_arm),
        ]
        clean_shutdown = all(shutdown_results)

    if completed_stream and total_errors == 0:
        print("ROBOT STATE STREAM TEST: PASS")
    else:
        print(f"ROBOT STATE STREAM TEST: FAIL errors={total_errors}")

    if clean_shutdown:
        print("CLEAN SHUTDOWN: PASS")
    else:
        print("CLEAN SHUTDOWN: FAIL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
