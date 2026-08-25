#!/usr/bin/env python3
"""No-Rabo regression tests for the Expert V2 Cartesian/joint motion gates."""

from __future__ import annotations

import inspect
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import expert.left_nut_grasp_planner as left_planner_module
import tools.test_three_nut_closed_loop_v2 as expert_v2
from agents.three_nut_expert.config import (
    KNOWN_FIXED_NUT_WORLD_POSE,
    RIGHT_APPROACH_HEIGHT,
    RIGHT_ARM_BASE_WORLD_Z,
    RIGHT_ARM_BASE_XY,
)
from agents.three_nut_expert.expert import (
    build_right_safe_lift_pose,
    compute_right_approach_pose,
    compute_right_grasp_pose,
    pose_to_list,
)
from expert.left_nut_grasp_planner import LeftNutGraspPlanner
from tools.test_left_grasp_v1 import RIGHT_NUT_B_GRASP_POSE
from tools.motion_monitor import MotionMonitor


TARGET_POSE = [0.10, 0.20, -0.10, 0.0, 0.8, 0.0]


def grasp_base_to_ideal_world_xyz(pose: object) -> list[float]:
    """Use the verified right-base axis convention (world yaw = pi)."""
    return [
        RIGHT_ARM_BASE_XY[0] - pose.x,
        RIGHT_ARM_BASE_XY[1] - pose.y,
        RIGHT_ARM_BASE_WORLD_Z + pose.z,
    ]


class RightGraspTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        b = KNOWN_FIXED_NUT_WORLD_POSE["B"]
        self.b_xyz = [b.x, b.y, b.z]
        self.b_grasp = compute_right_grasp_pose(self.b_xyz)

    def test_b_fixed_scene_pose_reproduces_verified_grasp(self) -> None:
        for actual, expected in zip(pose_to_list(self.b_grasp), pose_to_list(RIGHT_NUT_B_GRASP_POSE)):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_nut_xyz_translation_is_one_to_one_in_hand_world(self) -> None:
        base_hand_world = grasp_base_to_ideal_world_xyz(self.b_grasp)
        for axis in range(3):
            for delta in (-0.01, 0.01):
                shifted_nut = list(self.b_xyz)
                shifted_nut[axis] += delta
                shifted_hand_world = grasp_base_to_ideal_world_xyz(compute_right_grasp_pose(shifted_nut))
                for output_axis in range(3):
                    expected_delta = delta if output_axis == axis else 0.0
                    self.assertAlmostEqual(
                        shifted_hand_world[output_axis] - base_hand_world[output_axis],
                        expected_delta,
                        places=9,
                    )

    def test_right_approach_differs_only_by_positive_z(self) -> None:
        approach = compute_right_approach_pose(self.b_grasp)
        self.assertEqual(
            [approach.x, approach.y, approach.roll, approach.pitch, approach.yaw],
            [self.b_grasp.x, self.b_grasp.y, self.b_grasp.roll, self.b_grasp.pitch, self.b_grasp.yaw],
        )
        self.assertAlmostEqual(approach.z - self.b_grasp.z, RIGHT_APPROACH_HEIGHT, places=9)

    def test_all_verified_fixed_scene_targets(self) -> None:
        expected = {
            "A": [-0.3930, 0.0859, -0.3291, 0.0, 0.8, 0.0],
            "B": [-0.2803, 0.1570, -0.3300, 0.0, 0.8, 0.0],
            "C": [-0.3241, 0.0387, -0.3238, 0.0, 0.8, 0.0],
        }
        for key, fixed_pose in KNOWN_FIXED_NUT_WORLD_POSE.items():
            xyz = [fixed_pose.x, fixed_pose.y, fixed_pose.z]
            grasp = compute_right_grasp_pose(xyz)
            approach = compute_right_approach_pose(grasp)
            safe_lift = build_right_safe_lift_pose(grasp)
            for actual, wanted in zip(pose_to_list(grasp), expected[key]):
                self.assertAlmostEqual(actual, wanted, places=9)
            self.assertAlmostEqual(approach.z - grasp.z, 0.10, places=9)
            self.assertEqual(pose_to_list(safe_lift), pose_to_list(approach))


class MockArm:
    def __init__(
        self,
        *,
        endpoint_offset_x: float = 0.0,
        joint_after: float = 0.10,
        move_delay_s: float = 0.0,
        move_result: bool = True,
    ) -> None:
        self.endpoint_offset_x = endpoint_offset_x
        self.joint_after = joint_after
        self.move_delay_s = move_delay_s
        self.move_result = move_result
        self.joints = [0.0] * 7
        self.pose = [0.0] * 6
        self.move_to_calls = 0
        self.move_to_history: list[list[float]] = []

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
        self.move_to_history.append([x, y, z, roll, pitch, yaw])
        if self.move_delay_s:
            time.sleep(self.move_delay_s)
        self.pose = [x + self.endpoint_offset_x, y, z, roll, pitch, yaw]
        self.joints = [self.joint_after] * 7
        return self.move_result

    def move_joints(self, target: list[float]) -> bool:
        self.joints = list(target)
        return True

    def get_joint_angles(self) -> list[float]:
        return list(self.joints)

    def get_pose(self) -> list[float]:
        return list(self.pose)


class MockArmWithoutPoseFeedback(MockArm):
    def get_pose(self) -> None:
        return None


class MockHand:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def clench(self, *_args: object, **kwargs: object) -> bool:
        self.calls.append(("clench", dict(kwargs)))
        return True

    def grasp_force(self, **kwargs: object) -> bool:
        self.calls.append(("grasp_force", dict(kwargs)))
        return True


class TrackingLeftHand(MockHand):
    def __init__(self, arm: MockArm) -> None:
        super().__init__()
        self.arm = arm
        self.thumb_tuck_move_count: int | None = None
        self.grasp_force_move_count: int | None = None

    def clench(self, *_args: object, **kwargs: object) -> bool:
        if kwargs.get("thumb_rotation") == 1.0:
            self.thumb_tuck_move_count = self.arm.move_to_calls
        return super().clench(*_args, **kwargs)

    def grasp_force(self, **kwargs: object) -> bool:
        self.grasp_force_move_count = self.arm.move_to_calls
        return super().grasp_force(**kwargs)


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

    def test_cartesian_actual_joint_jump_faults_even_without_endpoint_feedback(self) -> None:
        arm = MockArmWithoutPoseFeedback(joint_after=0.80)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_JUMP")
        self.assertEqual(caught.exception.context["motion_monitor"]["diagnosis"], "ACTUAL_JOINT_JUMP")

    def test_cartesian_timeout_faults_after_sdk_returns(self) -> None:
        arm = MockArmWithoutPoseFeedback(move_delay_s=0.03)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(timeout_s=0.01), arm, TARGET_POSE, "MOCK_CARTESIAN_TIMEOUT")
        self.assertTrue(caught.exception.context["timeout"])
        self.assertTrue(caught.exception.context["sdk_returned_after_timeout"])

    def test_cartesian_endpoint_feedback_unavailable_is_nonfatal(self) -> None:
        arm = MockArmWithoutPoseFeedback()
        result = expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_NO_GET_POSE")
        self.assertTrue(result["gate"]["pass"])
        self.assertEqual(
            result["motion_gate_status"],
            "PASS_WITHOUT_CARTESIAN_ENDPOINT_FEEDBACK",
        )
        self.assertFalse(result["cartesian_endpoint_feedback_available"])
        self.assertEqual(result["endpoint_verification_status"], "UNAVAILABLE_NOT_FATAL")
        self.assertIsNone(result["position_error_m"])
        self.assertIsNone(result["orientation_error_deg"])

    def test_cartesian_sdk_false_faults_without_endpoint_feedback(self) -> None:
        arm = MockArmWithoutPoseFeedback(move_result=False)
        with self.assertRaises(expert_v2.StateFault) as caught:
            expert_v2.move_pose(self.runner(), arm, TARGET_POSE, "MOCK_CARTESIAN_SDK_FALSE")
        self.assertIn("SDK return failed", str(caught.exception))

    def test_right_pick_continues_to_thumb_tuck_without_endpoint_feedback(self) -> None:
        arm = MockArmWithoutPoseFeedback()
        hand = MockHand()
        bundle = SimpleNamespace(right_arm=arm, right_hand=hand)
        monitor = MotionMonitor(enabled=True, output_root=self.root / "motion")
        runner = expert_v2.ExpertStateRunner(1, monitor, self.root / "report.json")
        for state in ("EPISODE_INIT", "RIGHT_READY", "LEFT_READY", "READY_CHECK"):
            runner.enter(state, nut="C" if state == "READY_CHECK" else None)
            runner.pass_state({"mock": True})
        original_go_right_ready = expert_v2.go_right_ready
        original_detect = expert_v2.detect_released_nut
        original_hold = expert_v2.DEFAULT_HOLD_AFTER_GRASP_S
        original_release_wait = expert_v2.RIGHT_RELEASE_OPEN_WAIT_S
        ready_calls = []
        expert_v2.go_right_ready = lambda _runner, _arm: ready_calls.append("RIGHT_READY")
        expert_v2.detect_released_nut = lambda *_args, **_kwargs: {
            "detected": True,
            "nut_world_xyz": [0.1, 0.2, 0.3],
        }
        expert_v2.DEFAULT_HOLD_AFTER_GRASP_S = 0.0
        expert_v2.RIGHT_RELEASE_OPEN_WAIT_S = 0.0
        try:
            output = StringIO()
            with redirect_stdout(output):
                expert_v2.execute_right_transfer(
                    runner,
                    "C",
                    bundle,
                    runtime_nut_world_xyz=[-0.2975, -0.0527, 0.2868],
                    nut_xyz_source="VERIFIED_SCENE_FIXED_POSE",
                    vision_radius_m=0.1,
                    settle_after_release_s=0.0,
                )
        finally:
            expert_v2.go_right_ready = original_go_right_ready
            expert_v2.detect_released_nut = original_detect
            expert_v2.DEFAULT_HOLD_AFTER_GRASP_S = original_hold
            expert_v2.RIGHT_RELEASE_OPEN_WAIT_S = original_release_wait
        self.assertGreaterEqual(len(hand.calls), 2)
        self.assertEqual(ready_calls, ["RIGHT_READY"])
        trace = output.getvalue()
        self.assertIn("[NUT_RUNTIME_TARGET]", trace)
        self.assertIn("RIGHT_APPROACH_C =", trace)
        self.assertIn("RIGHT_PICK_C =", trace)
        self.assertIn("RIGHT_SAFE_LIFT_C =", trace)
        self.assertEqual(hand.calls[0], ("clench", {"thumb_rotation": 1.0}))
        self.assertEqual(hand.calls[1][0], "grasp_force")
        right_grasp = next(row for row in runner.states if row["state"] == "RIGHT_GRASP")
        right_approach = next(row for row in runner.states if row["state"] == "RIGHT_APPROACH")
        self.assertEqual(right_approach["status"], "STATE_PASS")
        state_names = [row["state"] for row in runner.states]
        ready_check_index = state_names.index("READY_CHECK")
        self.assertEqual(state_names[ready_check_index + 1], "RIGHT_APPROACH")
        approach_index = state_names.index("RIGHT_APPROACH")
        self.assertEqual(
            state_names[approach_index:approach_index + 5],
            ["RIGHT_APPROACH", "RIGHT_THUMB_TUCK", "RIGHT_GRASP", "RIGHT_GRASP_FORCE", "RIGHT_LIFT"],
        )
        self.assertEqual(
            right_grasp["details"]["move"]["motion_gate_status"],
            "PASS_WITHOUT_CARTESIAN_ENDPOINT_FEEDBACK",
        )
        expected_c_safe_lift = [-0.3241, 0.0387, -0.2238, 0.0, 0.8, 0.0]
        for actual, expected in zip(arm.move_to_history[2], expected_c_safe_lift):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertNotIn([-0.4, 0.12, -0.03, 0.0, 0.8, 0.0], arm.move_to_history)
        self.assertNotIn([-0.4, 0.0, -0.03, 0.0, 0.8, 0.0], arm.move_to_history)

    def test_left_pick_has_no_entry_ready_repeat(self) -> None:
        source = inspect.getsource(expert_v2.execute_left_pick_place)
        self.assertEqual(source.count("go_left_ready("), 1)
        self.assertGreater(source.index("go_left_ready("), source.index('runner.enter("LEFT_SAFE_RETREAT")'))

    def test_left_planner_tucks_after_pregrasp_before_descent(self) -> None:
        arm = MockArm()
        hand = TrackingLeftHand(arm)
        planner = LeftNutGraspPlanner(left_arm=arm, left_hand=hand)
        plan = planner.build_plan([0.1, 0.2, 0.3])
        original_sleep = left_planner_module.time.sleep
        left_planner_module.time.sleep = lambda _seconds: None
        try:
            result = planner.execute(plan)
        finally:
            left_planner_module.time.sleep = original_sleep
        self.assertTrue(result["success"])
        self.assertEqual(hand.thumb_tuck_move_count, 2)
        self.assertEqual(hand.grasp_force_move_count, 7)

    def test_left_v2_reports_safe_thumb_tuck_sequence(self) -> None:
        arm = MockArm()
        hand = MockHand()
        bundle = SimpleNamespace(left_arm=arm, left_hand=hand)
        runner = self.runner()
        original_go_left_ready = expert_v2.go_left_ready
        original_sleep = expert_v2.time.sleep
        ready_calls = []
        expert_v2.go_left_ready = lambda _runner, _arm: ready_calls.append("LEFT_READY")
        expert_v2.time.sleep = lambda _seconds: None
        try:
            expert_v2.execute_left_pick_place(runner, "C", bundle, [0.1, 0.2, 0.3])
        finally:
            expert_v2.go_left_ready = original_go_left_ready
            expert_v2.time.sleep = original_sleep
        state_names = [row["state"] for row in runner.states]
        approach_index = state_names.index("LEFT_APPROACH")
        self.assertEqual(
            state_names[approach_index:approach_index + 6],
            ["LEFT_APPROACH", "LEFT_THUMB_TUCK", "LEFT_DESCENT", "LEFT_GRASP", "LEFT_GRASP_FORCE", "LEFT_SAFE_LIFT"],
        )
        self.assertEqual(ready_calls, ["LEFT_READY"])

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
        self.assertNotIn("RIGHT_VISION", plan["per_nut_template"])
        self.assertIn("LEFT_VISION", plan["per_nut_template"])
        self.assertFalse(plan["right_pick_uses_vision"])
        self.assertTrue(plan["left_pick_uses_post_release_vision"])

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
