#!/usr/bin/env python3
"""Read-only runtime check for robot root -> base_link transforms.

This script inspects the live USD/Omniverse stage when available. It does not
publish ROS messages, call robot motion APIs, change entity poses, or modify
scene state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert.transforms import invert_transform, matrix_to_pose6, matmul


LEFT_ROOT_UI_POSE = [-0.6816, 0.0040, 0.7520, 0.0, 0.0, 0.0]
RIGHT_ROOT_UI_POSE = [-0.6816, -0.0040, 0.7520, 0.0, 0.0, 3.1400]

LEFT_HINTS = ("linker arm a7 左", "linkerarma7left", "left", "rbd03eb", "左")
RIGHT_HINTS = ("linker arm a7 右", "linkerarma7right", "right", "r412d23", "右")
BASE_HINTS = ("base_link", "baselink", "基本链接", "基址")

TRANSLATION_EPS = 1e-4
ROTATION_EPS = 1e-4


@dataclass
class PrimRecord:
    path: str
    name: str
    display_name: str
    type_name: str

    @property
    def text(self) -> str:
        return " ".join([self.path, self.name, self.display_name, self.type_name]).lower()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def load_stage() -> Any:
    try:
        import omni.usd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"omni.usd is not importable in this Python runtime: {exc!r}") from exc

    ctx = omni.usd.get_context()
    stage = ctx.get_stage() if ctx else None
    if stage is None:
        raise RuntimeError("omni.usd context has no active stage")
    return stage


def prim_record(prim: Any) -> PrimRecord:
    display = ""
    try:
        display = prim.GetDisplayName() or ""
    except Exception:
        pass
    return PrimRecord(
        path=str(prim.GetPath()),
        name=str(prim.GetName()),
        display_name=str(display),
        type_name=str(prim.GetTypeName()),
    )


def all_prims(stage: Any) -> list[Any]:
    return [prim for prim in stage.Traverse() if prim.IsValid()]


def score_root(record: PrimRecord, hints: tuple[str, ...]) -> int:
    text = record.text
    score = 0
    for hint in hints:
        compact = hint.replace(" ", "").lower()
        if hint.lower() in text:
            score += 10
        if compact and compact in text.replace(" ", ""):
            score += 6
    if any(base in text for base in BASE_HINTS):
        score -= 100
    return score


def find_root(candidates: list[Any], hints: tuple[str, ...]) -> Any | None:
    scored = [(score_root(prim_record(prim), hints), prim) for prim in candidates]
    scored = [(score, prim) for score, prim in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(str(item[1].GetPath()))), reverse=True)
    return scored[0][1]


def find_base_link(root: Any) -> Any | None:
    best: tuple[int, Any] | None = None
    for prim in root.GetDescendants():
        record = prim_record(prim)
        text = record.text
        score = sum(10 for hint in BASE_HINTS if hint in text)
        if score <= 0:
            continue
        if best is None or score > best[0] or (score == best[0] and len(record.path) < len(str(best[1].GetPath()))):
            best = (score, prim)
    return best[1] if best else None


def matrix_to_list(matrix: Any) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def world_matrix(prim: Any) -> list[list[float]]:
    from pxr import UsdGeom  # type: ignore

    cache = UsdGeom.XformCache()
    return matrix_to_list(cache.GetLocalToWorldTransform(prim))


def relative_pose(root_world: list[list[float]], base_world: list[list[float]]) -> list[float]:
    return matrix_to_pose6(matmul(invert_transform(root_world), base_world))


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def is_identity_pose(pose: list[float]) -> bool:
    return vector_norm(pose[:3]) <= TRANSLATION_EPS and vector_norm(pose[3:]) <= ROTATION_EPS


def inspect_side(label: str, stage: Any, hints: tuple[str, ...], ui_root_pose: list[float]) -> dict[str, Any]:
    prims = all_prims(stage)
    root = find_root(prims, hints)
    if root is None:
        return {
            "label": label,
            "status": "ROOT_NOT_FOUND",
            "ui_root_pose": ui_root_pose,
            "root_hints": hints,
        }
    base = find_base_link(root)
    if base is None:
        return {
            "label": label,
            "status": "BASE_LINK_NOT_FOUND_UNDER_ROOT",
            "ui_root_pose": ui_root_pose,
            "root": _jsonable(prim_record(root).__dict__),
        }

    root_world = world_matrix(root)
    base_world = world_matrix(base)
    root_pose = matrix_to_pose6(root_world)
    base_pose = matrix_to_pose6(base_world)
    root_to_base = relative_pose(root_world, base_world)
    identity = is_identity_pose(root_to_base)
    return {
        "label": label,
        "status": "IDENTITY_CONFIRMED" if identity else "ROOT_TO_BASE_NOT_IDENTITY",
        "ui_root_pose": ui_root_pose,
        "root": _jsonable(prim_record(root).__dict__),
        "base_link": _jsonable(prim_record(base).__dict__),
        "root_world_pose": root_pose,
        "base_link_world_pose": base_pose,
        "root_to_base_pose": root_to_base,
        "translation_norm": vector_norm(root_to_base[:3]),
        "rotation_norm": vector_norm(root_to_base[3:]),
        "thresholds": {"translation": TRANSLATION_EPS, "rotation": ROTATION_EPS},
    }


def print_side(prefix: str, result: dict[str, Any]) -> None:
    print(f"{prefix}_STATUS = {result['status']}")
    print(f"{prefix}_ROOT_PRIM = {result.get('root', {}).get('path')}")
    print(f"{prefix}_BASE_LINK_PRIM = {result.get('base_link', {}).get('path')}")
    print(f"{prefix}_ROOT_WORLD_POSE = {result.get('root_world_pose')}")
    print(f"{prefix}_BASE_LINK_WORLD_POSE = {result.get('base_link_world_pose')}")
    print(f"{prefix}_ROOT_TO_BASE = {result.get('root_to_base_pose')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Linker Arm A7 root -> base_link transform in live stage.")
    parser.add_argument("--json", action="store_true", help="Also print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stage = load_stage()
        left = inspect_side("LEFT", stage, LEFT_HINTS, LEFT_ROOT_UI_POSE)
        right = inspect_side("RIGHT", stage, RIGHT_HINTS, RIGHT_ROOT_UI_POSE)
        result = {"left": left, "right": right}
    except Exception as exc:
        result = {"error": repr(exc), "status": "STAGE_UNAVAILABLE"}
        print("ROOT_BASE_RUNTIME_CHECK = STAGE_UNAVAILABLE")
        print(f"ERROR = {exc!r}")
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    print("========================================")
    print("ROOT -> BASE_LINK RUNTIME CHECK")
    print("========================================")
    print_side("LEFT", left)
    print()
    print_side("RIGHT", right)
    print("========================================")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    ok = left["status"] == "IDENTITY_CONFIRMED" and right["status"] == "IDENTITY_CONFIRMED"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
