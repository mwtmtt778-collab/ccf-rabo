#!/usr/bin/env python3
"""Synthetic checks for the offline Nut C rosbag alignment contract."""

from __future__ import annotations

import unittest
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.convert_act_c_rosbag_episode import build_aligned_episode
from tools.run_c_only_serial_expert import ArmObservation, SerialExpert
from tools import test_nut_camera_calibration as camera_calibration


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


class FakeRosNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.context = object()
        self.subscriptions = []
        self.destroyed = False

    def create_subscription(self, _message_type, topic, callback, _qos):
        subscription = SimpleNamespace(topic=topic, callback=callback)
        self.subscriptions.append(subscription)
        return subscription

    def destroy_subscription(self, subscription) -> None:
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeSingleThreadedExecutor:
    instances = []

    def __init__(self, *, context=None) -> None:
        self.context = context
        self.node = None
        self.shutdown_called = False
        self.spin_count = 0
        self.__class__.instances.append(self)

    def add_node(self, node) -> bool:
        self.node = node
        return True

    def remove_node(self, node) -> bool:
        if self.node is node:
            self.node = None
            return True
        return False

    def spin_once(self, timeout_sec=None) -> None:
        self.spin_count += 1
        node = self.node
        if node is not None:
            for subscription in list(node.subscriptions):
                if "points" in subscription.topic:
                    message = SimpleNamespace(frame=self.spin_count)
                else:
                    message = SimpleNamespace(position=[self.spin_count * 0.001])
                subscription.callback(message)
        time.sleep(min(float(timeout_sec or 0.0), 0.001))

    def shutdown(self, timeout_sec=None) -> bool:
        self.shutdown_called = True
        return True


class ActCRosbagOfflineMockTest(unittest.TestCase):
    def test_passive_and_pointcloud_executors_coexist(self) -> None:
        import rclpy
        import rclpy.executors

        FakeSingleThreadedExecutor.instances = []
        nodes = []

        def create_node(name):
            node = FakeRosNode(name)
            nodes.append(node)
            return node

        with (
            mock.patch.object(rclpy, "init"),
            mock.patch.object(rclpy, "ok", return_value=True),
            mock.patch.object(rclpy, "shutdown"),
            mock.patch.object(rclpy, "create_node", side_effect=create_node),
            mock.patch.object(rclpy.executors, "SingleThreadedExecutor", FakeSingleThreadedExecutor),
            mock.patch.object(camera_calibration, "SETTLE_SECONDS", 0.01),
            mock.patch.object(camera_calibration, "POINT_TIMEOUT_SECONDS", 0.1),
        ):
            from tools.run_c_only_serial_expert import PassiveArmState

            passive = PassiveArmState(history_hz=500.0)
            try:
                passive.wait_ready(0.5)
                before = passive.observation("left_arm").sequence
                cloud, frames = camera_calibration.capture_fresh_cloud()
                after = passive.observation("left_arm").sequence
                self.assertIsNotNone(cloud)
                self.assertGreater(frames, 0)
                self.assertGreater(after, before)
                self.assertEqual(len(FakeSingleThreadedExecutor.instances), 2)
                passive_executor, cloud_executor = FakeSingleThreadedExecutor.instances
                self.assertIsNot(passive_executor, cloud_executor)
                self.assertFalse(passive_executor.shutdown_called)
                self.assertTrue(cloud_executor.shutdown_called)
                self.assertTrue(nodes[1].destroyed)
                self.assertFalse(nodes[0].destroyed)
            finally:
                passive.close()
            self.assertTrue(FakeSingleThreadedExecutor.instances[0].shutdown_called)
            self.assertTrue(nodes[0].destroyed)

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
