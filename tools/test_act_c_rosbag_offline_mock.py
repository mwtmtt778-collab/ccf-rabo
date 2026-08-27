#!/usr/bin/env python3
"""Synthetic checks for the offline Nut C rosbag alignment contract."""

from __future__ import annotations

import unittest
import threading
import time
from types import SimpleNamespace

import numpy as np

from tools.convert_act_c_rosbag_episode import build_aligned_episode
from tools.run_c_only_serial_expert import ArmObservation, SerialExpert


class FakeEvents:
    def __init__(self) -> None:
        self.rows = []

    def write(self, phase, device, command, target, result, *, event, **fields) -> None:
        self.rows.append({"phase": phase, "device": device, "command": command, "target": target, "result": result, "event": event, **fields})


class FakePassiveArmState:
    def __init__(self, initial: list[float]) -> None:
        self.condition = threading.Condition()
        self.rows = [ArmObservation(0, time.monotonic_ns(), tuple(initial))]

    def push(self, state: list[float]) -> None:
        with self.condition:
            self.rows.append(ArmObservation(len(self.rows), time.monotonic_ns(), tuple(state)))
            self.condition.notify_all()

    def observation(self, _arm: str) -> ArmObservation:
        with self.condition:
            return self.rows[-1]

    def history_since(self, _arm: str, sequence: int) -> list[ArmObservation]:
        with self.condition:
            return [row for row in self.rows if row.sequence > sequence]

    def wait_for_update(self, _arm: str, sequence: int, timeout_s: float) -> None:
        with self.condition:
            if self.rows[-1].sequence <= sequence:
                self.condition.wait(timeout_s)

    def spin(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def serial_args(**overrides):
    values = {
        "start_threshold_rad": 0.003,
        "settle_threshold_rad": 0.002,
        "settle_samples": 3,
        "motion_timeout_s": 3.0,
        "no_motion_grace_s": 0.1,
        "observe_hz": 20.0,
        "post_settle_s": 0.0,
        "hand_dwell_s": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ActCRosbagOfflineMockTest(unittest.TestCase):
    def test_motion_during_two_second_sdk_call_is_not_lost(self) -> None:
        passive = FakePassiveArmState([0.0] * 7)
        events = FakeEvents()
        expert = SerialExpert(events, passive, None, None, serial_args())

        def sdk_call() -> bool:
            def publish_motion() -> None:
                for state in ([0.1] * 7, [0.2] * 7, [0.2] * 7, [0.2] * 7, [0.2] * 7):
                    time.sleep(0.1)
                    passive.push(list(state))
            publisher = threading.Thread(target=publish_motion, daemon=True)
            publisher.start()
            time.sleep(2.0)
            publisher.join()
            return True

        expert.arm_command("MOCK_BLOCKING_CALL", "left_arm", "move_joints", [0.2] * 7, sdk_call)
        settled = next(row["result"] for row in events.rows if row["event"] == "motion_settled")
        self.assertTrue(settled["movement_observed_during_call"])
        self.assertTrue(settled["movement_started"])
        self.assertFalse(settled["already_settled"])
        self.assertGreaterEqual(settled["command_call_ms"], 1900.0)

    def test_no_significant_motion_becomes_already_settled(self) -> None:
        passive = FakePassiveArmState([0.0] * 7)
        events = FakeEvents()
        expert = SerialExpert(events, passive, None, None, serial_args(motion_timeout_s=1.0))

        def sdk_call() -> bool:
            def publish_stable() -> None:
                for _ in range(5):
                    time.sleep(0.03)
                    passive.push([0.0] * 7)
            threading.Thread(target=publish_stable, daemon=True).start()
            return True

        started = time.monotonic()
        expert.arm_command("MOCK_ALREADY_SETTLED", "right_arm", "move_joints", [0.0] * 7, sdk_call)
        elapsed = time.monotonic() - started
        settled = next(row["result"] for row in events.rows if row["event"] == "motion_settled")
        self.assertFalse(settled["movement_started"])
        self.assertTrue(settled["already_settled"])
        self.assertLess(elapsed, 0.8)

    def test_12hz_camera_passive_arms_and_hand_commands_align_to_5hz(self) -> None:
        second = 1_000_000_000
        start = 1_800_000_000_000_000_000
        end = start + 2 * second
        camera = [start - 100_000_000 + index * (second // 12) for index in range(27)]
        arm_samples = {
            "left_arm": [
                (start - 50_000_000 + index * 50_000_000, [index * 0.01 + joint for joint in range(7)])
                for index in range(42)
            ],
            "right_arm": [
                (start - 50_000_000 + index * 50_000_000, [-index * 0.01 - joint for joint in range(7)])
                for index in range(42)
            ],
        }
        right_thumb = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        events = [
            {
                "event": "command_start", "timestamp_wall_ns": start + 300_000_000,
                "device": "RIGHT_HAND", "command": "clench",
                "target": {"thumb_rotation": 1.0, "clench6_after": right_thumb},
            },
            {
                "event": "command_start", "timestamp_wall_ns": start + 500_000_000,
                "device": "RIGHT_HAND", "command": "grasp_force",
                "target": {"strength": 1.0, "fingers": [1, 3, 4], "clench6_after": None, "grasp_mode_after": 1},
            },
            {
                "event": "command_start", "timestamp_wall_ns": start + 1_300_000_000,
                "device": "RIGHT_HAND", "command": "clench",
                "target": {"value": [0.0] * 6, "clench6_after": [0.0] * 6, "grasp_mode_after": 0},
            },
        ]

        arrays = build_aligned_episode(camera, arm_samples, events, start_ns=start, end_ns=end)

        self.assertEqual(arrays["states"].shape, (11, 26))
        self.assertEqual(arrays["hybrid_actions"].shape, (10, 28))
        np.testing.assert_array_equal(arrays["hybrid_actions"][:, :26], arrays["states"][1:])
        np.testing.assert_array_equal(arrays["hybrid_actions"][:, 26:28], arrays["grasp_modes_at_state"][1:])
        np.testing.assert_allclose(np.diff(arrays["state_timestamps"]), 0.2)
        self.assertLessEqual(float(np.max(arrays["camera_age_s"])), 1.0 / 24.0 + 1e-6)
        self.assertTrue(np.isfinite(arrays["states"]).all())
        self.assertTrue(np.any(arrays["states"][:, 21] == 1.0))
        self.assertTrue(np.any(arrays["grasp_modes_at_state"][:, 1] == 1.0))


if __name__ == "__main__":
    unittest.main()
