#!/usr/bin/env python3
"""Read-only deployment checker for the portable fixed-point C Agent."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def import_check(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        return True, f"PASS{f' ({version})' if version else ''}"
    except BaseException as exc:
        return False, f"FAIL ({exc!r})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true", help="print all discovered image/state candidates")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    print(f"Repo root: {'PASS' if (PROJECT_ROOT / 'agents/act_c_policy/runtime.py').is_file() else 'FAIL'} ({PROJECT_ROOT})")
    checks["repo"] = (PROJECT_ROOT / "agents/act_c_policy/runtime.py").is_file()
    checks["python"] = sys.version_info >= (3, 10)
    print(f"Python: {'PASS' if checks['python'] else 'FAIL'} ({sys.version.split()[0]})")
    for label, module_name in (
        ("NumPy", "numpy"),
        ("ONNXRuntime", "onnxruntime"),
        ("rclpy", "rclpy"),
        ("Rabo SDK", "rabo_robocap"),
    ):
        ok, detail = import_check(module_name)
        checks[module_name] = ok
        print(f"{label}: {detail}")

    config_path = PROJECT_ROOT / "config" / "act_c_policy.json"
    checks["config"] = config_path.is_file()
    print(f"Config: {'PASS' if checks['config'] else 'FAIL'} ({config_path})")
    raw: dict[str, object] = {}
    if checks["config"]:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except BaseException as exc:
            checks["config"] = False
            print(f"Config parse: FAIL ({exc!r})")
    model = PROJECT_ROOT / str(raw.get("model", "models/act_c_fixed_point_v1.onnx"))
    checks["model"] = model.is_file()
    print(f"Model: {'PASS' if checks['model'] else 'FAIL'} ({model})")

    selected = None
    topic_pairs: list[tuple[str, list[str]]] = []
    state_groups: dict[str, list[str]] = {}
    if checks.get("numpy") and checks.get("rclpy") and checks["config"]:
        from agents.act_c_policy.runtime import (
            IMAGE_TYPE,
            RclpyOwner,
            RuntimeFault,
            discover_topic_pairs,
            joint_state_topic_groups,
            select_camera_topic,
        )
        import rclpy

        owner = None
        try:
            owner = RclpyOwner(rclpy)
            topic_pairs = discover_topic_pairs(rclpy, 5.0)
            state_groups = joint_state_topic_groups(topic_pairs)
            image_topics = sorted(name for name, types in topic_pairs if IMAGE_TYPE in types)
            print(f"Image topics: {image_topics}")
            try:
                selected = select_camera_topic(topic_pairs, str(raw.get("camera_topic", "auto")))
                checks["camera"] = True
                print(f"Selected TOP: {selected}")
            except RuntimeFault as exc:
                checks["camera"] = False
                print(f"Selected TOP: FAIL ({exc})")
            arm_groups = {name: topics for name, topics in state_groups.items() if len(topics) == 7}
            checks["state"] = len(arm_groups) >= 2
            print(f"State: {'PASS' if checks['state'] else 'FAIL'} ({len(arm_groups)} seven-joint groups)")
            if args.discover:
                print("Arm/state topic groups:")
                for name, topics in sorted(state_groups.items()):
                    print(f"  {name}: {len(topics)} topics")
                    for topic in topics:
                        print(f"    {topic}")
        except BaseException as exc:
            checks["camera"] = False
            checks["state"] = False
            print(f"Image topics: discovery failed ({exc!r})")
            print("Selected TOP: FAIL")
            print("State: FAIL")
        finally:
            if owner is not None:
                owner.close()
    else:
        checks["camera"] = False
        checks["state"] = False
        print("Image topics: discovery unavailable")
        print("Selected TOP: FAIL")
        print("State: FAIL")

    ready = all(
        checks.get(key, False)
        for key in ("repo", "python", "numpy", "onnxruntime", "rclpy", "rabo_robocap", "config", "model", "camera", "state")
    )
    print(f"Ready for dry-run: {'YES' if ready else 'NO'}")
    print("Robot commands sent: NO")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
