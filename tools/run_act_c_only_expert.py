#!/usr/bin/env python3
"""Minimal single-Nut C orchestration for ACT collection."""

from __future__ import annotations

from typing import Any

from agents.three_nut_expert.expert import pose_to_list
from agents.three_nut_expert.config import KNOWN_FIXED_NUT_WORLD_POSE, KNOWN_FIXED_NUT_WORLD_POSE_STATUS


def run_act_c_only_expert(
    runner: Any,
    right_bundle: Any,
    left_bundle: Any,
    *,
    settle_after_release_s: float,
    vision_target_radius_m: float,
    deterministic_place_path: bool = True,
) -> list[str]:
    """Run only the already validated C path and stop at terminal release."""
    from tools.test_three_nut_closed_loop_v2 import (
        execute_left_pick_place,
        execute_right_transfer,
        go_left_initial_ready,
        go_right_ready,
    )

    runner.enter("EPISODE_INIT")
    runner.pass_state({"use_scene_initial_pose": True, "set_entity_pose_on_episode_init": False, "world_reset_used": False})
    go_right_ready(runner, right_bundle.right_arm)
    go_left_initial_ready(runner, left_bundle.left_arm)
    runner.enter("READY_CHECK")
    runner.pass_state({"right_ready": True, "left_ready": True})
    runner.enter("NUT_C", nut="C")
    runtime_xyz = pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE["C"])[:3]
    released_xyz = execute_right_transfer(
        runner, "C", right_bundle,
        runtime_nut_world_xyz=runtime_xyz,
        nut_xyz_source=KNOWN_FIXED_NUT_WORLD_POSE_STATUS,
        vision_radius_m=vision_target_radius_m,
        settle_after_release_s=settle_after_release_s,
    )
    execute_left_pick_place(
        runner, "C", left_bundle, released_xyz,
        deterministic_place_path=deterministic_place_path,
        terminal_after_release=True,
    )
    runner.enter("DONE")
    runner.pass_state({"completed_nuts": ["C"]})
    return ["C"]
