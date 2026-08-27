#!/usr/bin/env python3
"""Mock tests for fixed-point Nut C ACT runtime; no ROS or robot SDK required."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.act_c_policy.runtime import (
    ARM_JOINT_LIMITS,
    CameraSnapshot,
    ConstrainedHandModeState,
    FixedPointCRuntime,
    OnnxPolicy,
    RuntimeFault,
    RuntimeStateReader,
    SerialRecedingHorizonMvp,
    TopCameraReader,
    build_safe_arm_target,
    main,
)
from tools.test_hybrid_action_replay import HybridHandExecutor


class FakeClock:
    def __init__(self) -> None:
        self.value = 1.0

    def monotonic(self) -> float:
        return self.value

    def monotonic_ns(self) -> int:
        return int(self.value * 1e9)

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, seconds)


class FakeCamera:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sequence = 1

    def wait_ready(self, timeout_s: float) -> None:
        return None

    def snapshot(self) -> CameraSnapshot:
        # Same real frame is intentionally held across all 5 Hz ticks.
        return CameraSnapshot(np.zeros((1, 3, 128, 128), dtype=np.float32), int(1e9), self.sequence)

    def effective_fps(self):
        return 1.25


class FakePassive:
    def wait_ready(self, timeout_s: float) -> None:
        return None

    def observation(self, arm: str):
        base = 0.0 if arm == "left_arm" else 10.0
        return SimpleNamespace(state=tuple(base + i for i in range(7)), timestamp_monotonic_ns=int(1e9))


class FakePolicy:
    def infer(self, image, state):
        output = np.zeros((1, 10, 28), dtype=np.float32)
        output[0, 0, :26] = state[0]
        output[0, 0, 26:] = [0.1, 0.9]
        return output, 5.0


class FakeMvpStateReader:
    def __init__(self) -> None:
        self.state = np.zeros((1, 26), dtype=np.float32)

    def snapshot(self):
        return self.state.copy(), int(1e9)

    def set_hand_position(self, hand, target) -> None:
        section = slice(14, 20) if hand == "left" else slice(20, 26)
        self.state[0, section] = np.asarray(target, dtype=np.float32)

    def set_hand_mode(self, hand, mode) -> None:
        return None


class SequencePolicy:
    def __init__(self, actions) -> None:
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.index = 0

    def infer(self, image, state):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        output = np.zeros((1, 10, 28), dtype=np.float32)
        output[0, 0] = action
        return output, 1.0


class FakeArmExecutor:
    def __init__(self, error=None) -> None:
        self.calls = []
        self.error = error

    def move_pair(self, left_target, right_target, *, step):
        self.calls.append((left_target, right_target, step))
        if self.error is not None:
            raise self.error
        return True


def fake_hand_executor(calls, name):
    return HybridHandExecutor(
        lambda target, blocking: calls.append((name, "clench", target, blocking)) or True,
        lambda blocking: calls.append((name, "force", blocking)) or True,
    )


def action_with(*, arm=0.0, left_score=0.0, right_score=0.0):
    action = np.zeros(28, dtype=np.float32)
    action[:14] = arm
    action[26] = left_score
    action[27] = right_score
    return action


class FakeIo:
    def __init__(self, name, shape, type_name="tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_name


class FakeSession:
    def get_inputs(self):
        return [FakeIo("top_image", [1, 3, 128, 128]), FakeIo("state", [1, 26])]

    def get_outputs(self):
        return [FakeIo("action_chunk", [1, 10, 28])]

    def run(self, names, inputs):
        return [np.zeros((1, 10, 28), dtype=np.float32)]


class RuntimeTests(unittest.TestCase):
    def test_ros_rgb_and_bgr_encoding_are_explicit(self) -> None:
        rgb_pixel = bytes([1, 2, 3]) * (128 * 128)
        rgb = TopCameraReader.decode_image(SimpleNamespace(encoding="rgb8", height=128, width=128, step=384, data=rgb_pixel))
        bgr = TopCameraReader.decode_image(SimpleNamespace(encoding="bgr8", height=128, width=128, step=384, data=rgb_pixel))
        self.assertEqual(rgb.shape, (1, 3, 128, 128))
        np.testing.assert_allclose(rgb[0, :, 0, 0], np.asarray([1, 2, 3]) / 255.0)
        np.testing.assert_allclose(bgr[0, :, 0, 0], np.asarray([3, 2, 1]) / 255.0)

    def test_state26_uses_passive_sdk_order_and_open_command_state(self) -> None:
        state, timestamp = RuntimeStateReader(FakePassive()).snapshot()
        self.assertEqual(state.shape, (1, 26))
        np.testing.assert_array_equal(state[0, :7], np.asarray([0, 5, 4, 2, 1, 6, 3]))
        np.testing.assert_array_equal(state[0, 7:14], np.asarray([10, 15, 14, 12, 11, 16, 13]))
        np.testing.assert_array_equal(state[0, 14:], np.zeros(12))
        self.assertEqual(timestamp, int(1e9))

    def test_onnx_contract_and_finite_output(self) -> None:
        policy = OnnxPolicy(Path("unused.onnx"), session=FakeSession())
        output, latency = policy.infer(
            np.zeros((1, 3, 128, 128), dtype=np.float32),
            np.zeros((1, 26), dtype=np.float32),
        )
        self.assertEqual(output.shape, (1, 10, 28))
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(latency, 0.0)

    def test_five_hz_dry_run_holds_latest_frame_and_never_commands(self) -> None:
        clock = FakeClock()
        runtime = FixedPointCRuntime(
            FakeCamera(clock), RuntimeStateReader(FakePassive()), FakePolicy(),
            monotonic=clock.monotonic, monotonic_ns=clock.monotonic_ns, sleep=clock.sleep,
        )
        runtime.wait_ready(1.0)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dry.jsonl"
            report = runtime.run_dry(2.0, log)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["loop_effective_hz"], 5.0)
        self.assertEqual(report["camera_unique_frames_used"], 1)
        self.assertTrue(all(row["robot_commanded"] is False for row in rows))
        self.assertTrue(all(row["action_finite"] for row in rows))

    def test_existing_hybrid_hand_rising_edge_semantics_are_reused(self) -> None:
        calls = []
        executor = HybridHandExecutor(
            lambda target, blocking: calls.append(("clench", target, blocking)) or True,
            lambda blocking: calls.append(("force", blocking)) or True,
        )
        self.assertEqual(executor.apply([0] * 6, 1, blocking=False), "GRASP_FORCE_RISING_EDGE")
        self.assertEqual(executor.apply([1] * 6, 1, blocking=False), "FORCE_MODE_HOLD_NO_HAND_COMMAND")
        self.assertEqual(executor.apply([0] * 6, 0, blocking=False), "POSITION_MODE_FALLING_EDGE_CLENCH")
        self.assertEqual(sum(call[0] == "force" for call in calls), 1)

    def test_execute_fails_closed_before_sdk_creation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "EXECUTE_DISABLED_FAIL_CLOSED"):
            main(["--execute"])

    def test_mvp_one_rad_prediction_is_clipped_to_point_one(self) -> None:
        target = build_safe_arm_target(np.zeros(14), np.ones(14))
        self.assertAlmostEqual(target.max_raw_delta, 1.0)
        self.assertLessEqual(target.max_executed_delta, 0.100001)
        self.assertTrue(np.all(np.abs(target.clipped_target) <= 0.100001))

    def test_mvp_joint_limit_clamp(self) -> None:
        current = np.zeros(14, dtype=np.float32)
        current[1] = 0.06
        predicted = current.copy()
        predicted[1] = 1.0
        target = build_safe_arm_target(current, predicted)
        self.assertAlmostEqual(float(target.clipped_target[1]), float(ARM_JOINT_LIMITS[1, 1]), places=6)
        self.assertLessEqual(target.max_executed_delta, 0.100001)

    def test_right_mode_completes_zero_one_zero(self) -> None:
        mode = ConstrainedHandModeState()
        states = [mode.update(score).mode for score in [0.9, 0.9, 0.9, 0.5, 0.1, 0.1, 0.1]]
        self.assertEqual(states, [0, 0, 1, 1, 1, 1, 0])
        self.assertEqual(mode.state, "DONE")

    def test_left_force_is_gated_until_right_done(self) -> None:
        right = ConstrainedHandModeState()
        left = ConstrainedHandModeState()
        for _ in range(3):
            right.update(0.0)
            decision = left.update(1.0, enabled=right.state == "DONE")
            self.assertEqual(decision.mode, 0)
            self.assertEqual(left.state, "POSITION")
        for score in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]:
            right.update(score)
            left.update(1.0, enabled=right.state == "DONE")
        self.assertEqual(right.state, "DONE")
        self.assertEqual(left.state, "POSITION")
        left.update(1.0, enabled=True)
        left.update(1.0, enabled=True)
        decision = left.update(1.0, enabled=True)
        self.assertEqual(decision.mode, 1)
        self.assertEqual(left.state, "FORCE")

    def test_mvp_force_rising_edge_calls_once(self) -> None:
        calls = []
        executor = fake_hand_executor(calls, "right")
        mode = ConstrainedHandModeState()
        for score in [0.9, 0.9, 0.9, 0.9, 0.9]:
            decision = mode.update(score)
            executor.apply([0.0] * 6, decision.mode, blocking=True)
        self.assertEqual(sum(call[1] == "force" for call in calls), 1)

    def test_mvp_nan_action_fails(self) -> None:
        calls = []
        runtime = SerialRecedingHorizonMvp(
            FakeCamera(FakeClock()), FakeMvpStateReader(),
            SequencePolicy([action_with(arm=np.nan)]), FakeArmExecutor(),
            fake_hand_executor(calls, "left"), fake_hand_executor(calls, "right"),
            max_policy_steps=1, sleep=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeFault, "NaN/Inf"):
                runtime.run(Path(directory) / "mvp.jsonl")

    def test_mvp_settle_timeout_fails(self) -> None:
        calls = []
        runtime = SerialRecedingHorizonMvp(
            FakeCamera(FakeClock()), FakeMvpStateReader(),
            SequencePolicy([action_with(arm=0.1)]),
            FakeArmExecutor(TimeoutError("settle timeout")),
            fake_hand_executor(calls, "left"), fake_hand_executor(calls, "right"),
            max_policy_steps=1, sleep=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TimeoutError, "settle timeout"):
                runtime.run(Path(directory) / "mvp.jsonl")

    def test_mvp_max_policy_steps_fails(self) -> None:
        calls = []
        runtime = SerialRecedingHorizonMvp(
            FakeCamera(FakeClock()), FakeMvpStateReader(),
            SequencePolicy([action_with()]), FakeArmExecutor(),
            fake_hand_executor(calls, "left"), fake_hand_executor(calls, "right"),
            max_policy_steps=2, sleep=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeFault, "MAX_POLICY_STEPS_EXCEEDED"):
                runtime.run(Path(directory) / "mvp.jsonl")

    def test_mvp_finishes_after_five_additional_predictions(self) -> None:
        calls = []
        actions = []
        for step in range(16):
            right_score = 0.9 if step < 3 else 0.1
            left_score = 0.9 if 5 <= step < 8 else 0.1
            actions.append(action_with(left_score=left_score, right_score=right_score))
        runtime = SerialRecedingHorizonMvp(
            FakeCamera(FakeClock()), FakeMvpStateReader(), SequencePolicy(actions),
            FakeArmExecutor(), fake_hand_executor(calls, "left"),
            fake_hand_executor(calls, "right"), max_policy_steps=20, sleep=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            report = runtime.run(Path(directory) / "mvp.jsonl")
        self.assertEqual(report["status"], "DONE")
        self.assertEqual(report["policy_steps"], 16)
        self.assertEqual(report["post_done_arm_steps"], 5)


if __name__ == "__main__":
    unittest.main()
