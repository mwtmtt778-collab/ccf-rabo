#!/usr/bin/env python3
"""No-Rabo regression tests for the Expert V2 Cartesian/joint motion gates."""

from __future__ import annotations

import tempfile
import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import tools.test_three_nut_closed_loop_v2 as expert_v2
from tools.motion_monitor import MotionMonitor


TARGET_POSE = [0.10, 0.20, -0.10, 0.0, 0.8, 0.0]


class MockArm:
    def __init__(
        self,
        *,
        endpoint_offset_x: float = 0.0,
        joint_after: float = 0.10,
        move_delay_s: float = 0.0,
    ) -> None:
        self.endpoint_offset_x = endpoint_offset_x
        self.joint_after = joint_after
        self.move_delay_s = move_delay_s
        self.joints = [0.0] * 7
        self.pose = [0.0] * 6
        self.move_to_calls = 0

    def pose_check(self, *_args: object, **_kwargs: object) -> list[object]:
        return [True, "reachable"]

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        *,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> bool:
        self.move_to_calls += 1
        if self.move_delay_s:
            time.sleep(self.move_delay_s)
        self.pose = [x + self.endpoint_offset_x, y, z, roll, pitch, yaw]
        self.joints = [self.joint_after] * 7
        return True

    def move_joints(self, target: list[float]) -> bool:
        self.joints = list(target)
        return True

    def get_joint_angles(self) -> list[float]:
        return list(self.joints)

    def get_pose(self) -> list[float]:
        return list(self.pose)


class MockArmWithoutPoseFeedback(MockArm):
    def get_pose(self) -> list[float]:
        raise RuntimeError("get_pose unavailable")


class CartesianGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=expert_v2.PROJECT_ROOT)
        self.root = Path(self.temp_dir.name)
        self.original_report_dir = expert_v2.REPORT_DIR
        expert_v2.REPORT_DIR = self.root / "faults"

    def tearDown(self) -> None:
        expert_v2.REPORT_DIR = self.original_report_dir
        self.temp_dir.cleanup()

    def runner(self, *, timeout_s: float = 10.0) -> expert_v2.ExpertStateRunner:
        monitor = MotionMonitor(enabled=True, output_root=self.root / "motion")
        monitor.config["action_timeout_s"] = timeout_s
        runner = expert_v2.ExpertStateRunner(1, monitor, self.root / "report.json")
        runner.enter("MOCK_MOTION", nut="C")
        return runner

    def test_pose_check_reachable_without_target_joint(self) -> None:
        arm = MockArm()
        target_joint, result = expert_v2.pose_check_cartesian(arm, TARGET_POSE)
        self.assertIsNone(target_joint)
        self.assertTrue(result["pose_check_reachable"])
        self.assertFalse(result["pose_check_target_joint_available"])

    def test_cartesian_endpoint_near_target_passes(self) -> None:
        arm = MockArm()
        result = expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_PASS")
        self.assertEqual(arm.move_to_calls, 1)
        self.assertTrue(result["gate"]["pass"])
        self.assertFalse(result["pose_check_target_joint_available"])
        self.assertFalse(result["joint_target_based_divergence_available"])
        self.assertAlmostEqual(result["position_error_m"], 0.0)
        self.assertAlmostEqual(result["orientation_error_deg"], 0.0)

    def test_cartesian_endpoint_error_faults(self) -> None:
        arm = MockArm(endpoint_offset_x=0.10)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_BAD_ENDPOINT")
        self.assertEqual(caught.exception.context["motion_monitor"]["diagnosis"], "EE_POSITION_ERROR")

    def test_cartesian_actual_joint_jump_faults(self) -> None:
        arm = MockArm(joint_after=0.80)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_JUMP")
        self.assertEqual(caught.exception.context["motion_monitor"]["diagnosis"], "ACTUAL_JOINT_JUMP")

    def test_cartesian_timeout_faults_after_sdk_returns(self) -> None:
        arm = MockArm(move_delay_s=0.03)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(timeout_s=0.01), arm, TARGET_POSE, "MOCK_CARTESIAN_TIMEOUT")
        self.assertTrue(caught.exception.context["timeout"])
        self.assertTrue(caught.exception.context["sdk_returned_after_timeout"])

    def test_cartesian_endpoint_feedback_unavailable_faults(self) -> None:
        arm = MockArmWithoutPoseFeedback()
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_NO_GET_POSE")
        self.assertEqual(
            caught.exception.context["motion_monitor"]["diagnosis"],
            "CARTESIAN_ENDPOINT_FEEDBACK_UNAVAILABLE",
        )

    def test_move_joints_gate_regression_passes(self) -> None:
        arm = MockArm()
        target = [0.10] * 7
        result = self.runner().arm_action(
            label="MOCK_MOVE_JOINTS_PASS",
            arm=arm,
            command_method="move_joints",
            target_joint=target,
            fn=lambda: arm.move_joints(target),
        )
        self.assertTrue(result["gate"]["pass"])
        self.assertAlmostEqual(result["final_joint_error"], 0.0)
        self.assertTrue(result["joint_target_based_divergence_available"])

    def test_default_plan_uses_scene_initial_pose_without_reset_or_randomization(self) -> None:
        args = expert_v2.parse_args([])
        plan = expert_v2.plan_summary(expert_v2.NUT_SEQUENCE, args.reset_to_nominal)
        self.assertTrue(plan["use_scene_initial_pose"])
        self.assertFalse(plan["randomize_nuts"])
        self.assertFalse(plan["set_entity_pose_on_episode_init"])

    def test_multiple_trials_require_explicit_deterministic_reset(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            expert_v2.parse_args(["--trials", "2"])
        args = expert_v2.parse_args(["--trials", "2", "--reset-to-nominal"])
        self.assertTrue(args.reset_to_nominal)

    def test_reset_to_nominal_is_fixed_and_deterministic(self) -> None:
        calls: list[tuple[str, list[float]]] = []
        original = expert_v2.set_pose_with_retry
        expert_v2.set_pose_with_retry = lambda _setter, entity_id, pose: (
            calls.append((entity_id, list(pose))) or {"ok": True}
        )
        try:
            first = expert_v2.reset_nuts_to_nominal(object())
            first_calls = list(calls)
            calls.clear()
            second = expert_v2.reset_nuts_to_nominal(object())
        finally:
            expert_v2.set_pose_with_retry = original
        self.assertEqual(first, second)
        self.assertEqual(first_calls, calls)
        self.assertEqual([row["nut"] for row in first], ["A", "B", "C"])

    def test_v2_has_no_randomized_nut_entry_point(self) -> None:
        source = Path(expert_v2.__file__).read_text(encoding="utf-8")
        for forbidden in ("random.uniform", "np.random", "numpy.random", "jittered_nut_pose"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
