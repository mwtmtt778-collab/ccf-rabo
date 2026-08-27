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
    CameraSnapshot,
    FixedPointCRuntime,
    OnnxPolicy,
    RuntimeStateReader,
    TopCameraReader,
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


if __name__ == "__main__":
    unittest.main()
