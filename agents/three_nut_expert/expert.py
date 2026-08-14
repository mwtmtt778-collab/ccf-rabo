"""Parameterized expert built from the legacy single-nut demo."""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    DEVICE_IDS,
    GRASP_TARGET_OFFSETS,
    HAND_OPEN,
    LEFT_COARSE_POSE,
    LEFT_GRASP,
    LEFT_GRASP_FORCE,
    LEFT_HANDOFF_POSE,
    LEFT_PLACE_POSES,
    LEFT_PRE_JOINTS,
    LEFT_PRE_OPEN,
    NUT_SPECS,
    RIGHT_ARM_BASE_XY,
    RIGHT_CLEAR_POSE,
    RIGHT_GRASP_FORCE,
    RIGHT_LIFT_POSES,
    RIGHT_PRE_JOINTS,
    WORLD_ID,
    NutSpec,
    Pose6,
)


@dataclass
class ActionStep:
    phase: str
    target: str
    method: str
    args: list[Any]
    kwargs: dict[str, Any]
    legacy: bool
    note: str = ""


@dataclass
class ExpertRunResult:
    mode: str
    execute: bool
    success: bool
    started_at: str
    duration_s: float
    nut_keys: list[str]
    steps_planned: int
    steps_executed: int
    error: str | None
    samples: dict[str, Any]


class ExpertExecutionError(RuntimeError):
    pass


def pose_to_list(pose: Pose6) -> list[float]:
    return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]


def jittered_nut_pose(spec: NutSpec, rng: random.Random, enable_jitter: bool = True) -> Pose6:
    dx = rng.uniform(-spec.jitter_xy, spec.jitter_xy) if enable_jitter else 0.0
    dy = rng.uniform(-spec.jitter_xy, spec.jitter_xy) if enable_jitter else 0.0
    p = spec.nominal_pose
    return Pose6(p.x + dx, p.y + dy, p.z, p.roll, p.pitch, p.yaw)


def compute_right_grasp_pose(nut_pose: Pose6) -> Pose6:
    base_x, base_y = RIGHT_ARM_BASE_XY
    return Pose6(
        base_x - nut_pose.x + GRASP_TARGET_OFFSETS["x"],
        base_y - nut_pose.y + GRASP_TARGET_OFFSETS["y"],
        GRASP_TARGET_OFFSETS["z"],
        GRASP_TARGET_OFFSETS["roll"],
        GRASP_TARGET_OFFSETS["pitch"],
        GRASP_TARGET_OFFSETS["yaw"],
    )


def build_demo_contract() -> dict[str, Any]:
    return {
        "world_id": WORLD_ID,
        "device_ids": DEVICE_IDS,
        "source_of_truth": "agents/arm_hand_demo/__init__.py and docs/legacy/arm_hand_demo_legacy_snapshot.py",
        "runtime_decision": "Use SDK get_joint_angles for qpos; do not assemble full robot state from per-joint ROS topics.",
        "single_nut_contract": {
            "proven_nut": "B",
            "right_arm_base_xy": list(RIGHT_ARM_BASE_XY),
            "grasp_transform": "target_x = base_x - nut_x + 0.06; target_y = base_y - nut_y - 0.01; z=-0.33; rpy=(0,0.8,0)",
            "right_grasp_force": RIGHT_GRASP_FORCE,
            "left_grasp_force": LEFT_GRASP_FORCE,
        },
        "gates": [
            "dry-run plan renders without importing Rabo SDK",
            "cloud single B execute succeeds 5-10 times",
            "only then run A/C single validation",
            "only then run three-nut sequence",
            "camera gate must pass before Recorder",
        ],
        "unknowns": [
            "A/C staged spawn/place poses are not proven",
            "RGB camera frames are unstable in latest runtime map",
            "no read-only object pose API has been confirmed",
        ],
    }


def print_contract(contract: dict[str, Any]) -> None:
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def build_single_nut_plan(nut_key: str, seed: int | None = None, enable_jitter: bool = True) -> tuple[Pose6, list[ActionStep]]:
    key = nut_key.upper()
    if key not in NUT_SPECS:
        raise ValueError(f"unknown nut key: {nut_key}")

    rng = random.Random(seed)
    nut_pose = jittered_nut_pose(NUT_SPECS[key], rng, enable_jitter=enable_jitter)
    grasp_pose = compute_right_grasp_pose(nut_pose)
    place_pose = LEFT_PLACE_POSES[key]

    steps: list[ActionStep] = [
        ActionStep("init", "world", "set_entity_pose", [NUT_SPECS[key].thing_id, pose_to_list(nut_pose)], {}, True, "place known nut pose"),
    ]
    for joints in RIGHT_PRE_JOINTS:
        steps.append(ActionStep("pre_position", "right_arm", "move_joints", [joints], {}, True))
    for joints in LEFT_PRE_JOINTS:
        steps.append(ActionStep("pre_position", "left_arm", "move_joints", [joints], {}, True))

    steps.extend(
        [
            ActionStep("approach", "right_arm", "move_to", [], asdict(grasp_pose), True, "legacy world-to-arm transform"),
            ActionStep("grasp", "right_hand", "clench", [], {"thumb_rotation": 1.0}, True),
            ActionStep("grasp", "right_hand", "grasp_force", [], RIGHT_GRASP_FORCE, True),
        ]
    )
    for pose in RIGHT_LIFT_POSES:
        steps.append(ActionStep("handoff", "right_arm", "move_to", [], asdict(pose), True))
    steps.extend(
        [
            ActionStep("handoff", "left_arm", "move_to", [], asdict(LEFT_COARSE_POSE), True),
            ActionStep("handoff", "left_arm", "move_to", [], asdict(LEFT_HANDOFF_POSE), True),
            ActionStep("handoff", "left_hand", "clench", list(LEFT_PRE_OPEN), {}, True),
            ActionStep("release", "right_hand", "clench", list(HAND_OPEN), {}, True),
            ActionStep("clear", "right_arm", "move_to", [], asdict(RIGHT_CLEAR_POSE), True),
            ActionStep("catch", "left_hand", "clench", list(LEFT_GRASP), {}, True),
            ActionStep("catch", "left_hand", "grasp_force", [], {"strength": LEFT_GRASP_FORCE["strength"]}, True),
            ActionStep("place", "left_arm", "move_to", [], asdict(place_pose), key == "B", "B is legacy release; A/C are staged slots"),
            ActionStep("place", "left_hand", "clench", list(HAND_OPEN), {}, True),
        ]
    )
    return nut_pose, steps


def plan_to_dict(nut_key: str, nut_pose: Pose6, steps: list[ActionStep]) -> dict[str, Any]:
    return {
        "nut": nut_key.upper(),
        "nut_pose": pose_to_list(nut_pose),
        "steps": [asdict(step) for step in steps],
    }


class RaboDeviceBundle:
    def __init__(self) -> None:
        from rabo_dev_kit import SetEntityPose
        from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right

        self.pose_setter = SetEntityPose(world=WORLD_ID)
        self.right_arm = LinkerArmA7(robot_id=DEVICE_IDS["RIGHT_ARM"], mode="sim")
        self.left_arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")
        self.right_hand = LinkerHandO6Right(robot_id=DEVICE_IDS["RIGHT_HAND"], mode="sim")
        self.left_hand = LinkerHandO6Left(robot_id=DEVICE_IDS["LEFT_HAND"], mode="sim")

    def shutdown(self) -> None:
        for device in (self.left_hand, self.right_hand, self.left_arm, self.right_arm):
            if hasattr(device, "shutdown"):
                device.shutdown()

    def read_state(self) -> dict[str, Any]:
        return {
            "left_arm_qpos": self.left_arm.get_joint_angles(),
            "right_arm_qpos": self.right_arm.get_joint_angles(),
            "left_hand_qpos": self.left_hand.get_joint_angles(),
            "right_hand_qpos": self.right_hand.get_joint_angles(),
        }


def _pose_kwargs(kwargs: dict[str, Any]) -> dict[str, float]:
    return {
        "x": kwargs["x"],
        "y": kwargs["y"],
        "z": kwargs["z"],
        "roll": kwargs.get("roll", 0.0),
        "pitch": kwargs.get("pitch", 0.0),
        "yaw": kwargs.get("yaw", 0.0),
    }


def execute_step(bundle: RaboDeviceBundle, step: ActionStep) -> None:
    if step.method == "set_entity_pose":
        thing_id, pose = step.args
        bundle.pose_setter.set(thing_id, tuple(pose))
        return

    target = getattr(bundle, step.target)
    method = getattr(target, step.method)
    if step.method == "move_to":
        method(**_pose_kwargs(step.kwargs))
    elif step.method == "grasp_force":
        if step.kwargs.get("fingers") is None:
            method(strength=step.kwargs["strength"])
        else:
            method(strength=step.kwargs["strength"], fingers=step.kwargs["fingers"])
    else:
        method(*step.args, **step.kwargs)


def execute_single_nut(
    nut_key: str = "B",
    *,
    seed: int | None = None,
    execute: bool = False,
    enable_jitter: bool = True,
    stop_on_failure: bool = True,
) -> tuple[ExpertRunResult, dict[str, Any]]:
    started = time.time()
    started_text = time.strftime("%Y-%m-%d %H:%M:%S %z")
    nut_pose, steps = build_single_nut_plan(nut_key, seed=seed, enable_jitter=enable_jitter)
    plan = plan_to_dict(nut_key, nut_pose, steps)
    samples: dict[str, Any] = {}
    executed = 0
    error = None
    success = True

    if execute:
        bundle = None
        try:
            bundle = RaboDeviceBundle()
            samples["before"] = bundle.read_state()
            for step in steps:
                execute_step(bundle, step)
                executed += 1
            samples["after"] = bundle.read_state()
        except Exception as exc:
            error = repr(exc)
            success = False
            if stop_on_failure:
                pass
        finally:
            if bundle is not None:
                try:
                    bundle.shutdown()
                except Exception as exc:
                    samples["shutdown_error"] = repr(exc)
    else:
        executed = 0

    result = ExpertRunResult(
        mode="single",
        execute=execute,
        success=success if execute else True,
        started_at=started_text,
        duration_s=time.time() - started,
        nut_keys=[nut_key.upper()],
        steps_planned=len(steps),
        steps_executed=executed,
        error=error,
        samples=samples,
    )
    return result, plan


def execute_three_nut(
    *,
    order: list[str] | None = None,
    seed: int | None = None,
    execute: bool = False,
    enable_jitter: bool = True,
) -> tuple[ExpertRunResult, dict[str, Any]]:
    order = [x.upper() for x in (order or ["A", "B", "C"])]
    started = time.time()
    started_text = time.strftime("%Y-%m-%d %H:%M:%S %z")
    all_plans = []
    samples: dict[str, Any] = {"single_results": []}
    total_steps = 0
    executed_steps = 0
    success = True
    error = None

    for index, key in enumerate(order):
        single_seed = None if seed is None else seed + index
        result, plan = execute_single_nut(
            key,
            seed=single_seed,
            execute=execute,
            enable_jitter=enable_jitter,
            stop_on_failure=True,
        )
        all_plans.append(plan)
        total_steps += result.steps_planned
        executed_steps += result.steps_executed
        samples["single_results"].append(asdict(result))
        if not result.success:
            success = False
            error = f"{key} failed: {result.error}"
            break

    result = ExpertRunResult(
        mode="three",
        execute=execute,
        success=success,
        started_at=started_text,
        duration_s=time.time() - started,
        nut_keys=order,
        steps_planned=total_steps,
        steps_executed=executed_steps,
        error=error,
        samples=samples,
    )
    return result, {"order": order, "plans": all_plans}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
