#!/usr/bin/env python3
"""Read-only robot state smoke test for the Rabo sim runtime.

This script intentionally does not call any motion, hand actuation, entity-pose,
or reset API. It only instantiates the configured devices and calls read methods.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


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


def print_section(title: str) -> None:
    print()
    print("=" * 32)
    print(title)
    print("=" * 32)


def call_readonly(label: str, factory, reads: tuple[str, ...]) -> dict[str, Any]:
    print_section(label)
    result: dict[str, Any] = {"label": label, "status": "UNKNOWN", "reads": {}, "errors": []}
    try:
        device = factory()
        result["status"] = "CONNECTED"
        print("status: CONNECTED")
    except Exception as exc:
        result["status"] = "INIT_ERROR"
        result["errors"].append({"where": "init", "error": repr(exc)})
        print("status: INIT_ERROR")
        print("init error:", repr(exc))
        return result

    for method_name in reads:
        try:
            method = getattr(device, method_name)
            value = method()
            result["reads"][method_name] = _jsonable(value)
            print(f"{method_name}: {json.dumps(_jsonable(value), ensure_ascii=False)}")
        except Exception as exc:
            result["reads"][method_name] = None
            result["errors"].append({"where": method_name, "error": repr(exc)})
            print(f"{method_name}: ERROR {repr(exc)}")

    return result


def main() -> int:
    ids = load_robot_ids()
    print("Loaded robot IDs from:", ARM_HAND_DEMO)
    for name in REQUIRED_IDS:
        print(f"{name}: {ids[name]}")

    try:
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right
    except Exception as exc:
        for label in ("LEFT ARM", "RIGHT ARM", "LEFT HAND", "RIGHT HAND"):
            print_section(label)
            print("status: SDK_IMPORT_ERROR")
            print("import error:", repr(exc))
        print()
        print("=" * 32)
        print("ROBOT STATE TEST SUMMARY")
        print("=" * 32)
        for label in ("LEFT ARM", "RIGHT ARM", "LEFT HAND", "RIGHT HAND"):
            print(f"{label}: SDK_IMPORT_ERROR errors=1")
        return 0

    results = []
    results.append(
        call_readonly(
            "LEFT ARM",
            lambda: LinkerArmA7(robot_id=ids["LEFT_ARM_ID"], mode="sim"),
            ("get_joint_angles", "get_pose"),
        )
    )
    results.append(
        call_readonly(
            "RIGHT ARM",
            lambda: LinkerArmA7(robot_id=ids["RIGHT_ARM_ID"], mode="sim"),
            ("get_joint_angles", "get_pose"),
        )
    )
    results.append(
        call_readonly(
            "LEFT HAND",
            lambda: LinkerHandO6Left(robot_id=ids["LEFT_HAND_ID"], mode="sim"),
            ("get_joint_angles", "get_clench"),
        )
    )
    results.append(
        call_readonly(
            "RIGHT HAND",
            lambda: LinkerHandO6Right(robot_id=ids["RIGHT_HAND_ID"], mode="sim"),
            ("get_joint_angles", "get_clench"),
        )
    )

    print()
    print("=" * 32)
    print("ROBOT STATE TEST SUMMARY")
    print("=" * 32)
    for result in results:
        print(f"{result['label']}: {result['status']} errors={len(result['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
