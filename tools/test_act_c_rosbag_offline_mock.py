#!/usr/bin/env python3
"""Synthetic checks for the offline Nut C rosbag alignment contract."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.collect_act_c_rosbag import ROSBAG_TOPICS
from tools.convert_act_c_rosbag_episode import (
    build_aligned_episode,
    quality_report,
    read_arm_states,
    wall_to_monotonic_ns,
)
from tools.run_c_only_serial_expert import (
    ArmObservation,
    PASSIVE_TO_SDK_INDICES,
    PassiveArmState,
    SerialExpert,
    STATE_TOPICS,
    TOP_RGB_TOPIC,
)
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
        try:
            import rclpy
            import rclpy.executors
        except ImportError as exc:
            self.skipTest(f"rclpy unavailable: {exc}")

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

    def test_top_only_bag_with_arm_jsonl_aligns_state26_and_hybrid28(self) -> None:
        second = 1_000_000_000
        start_mono = 5_000_000_000
        start_wall = 1_800_000_000_000_000_000
        end_mono = start_mono + 2 * second
        camera_wall = [start_wall - 100_000_000 + index * (second // 12) for index in range(27)]
        camera = wall_to_monotonic_ns(camera_wall, start_wall, start_mono)
        self.assertEqual(int(camera[0]), start_mono - 100_000_000)
        arm_rows = [
            {
                "timestamp_monotonic_ns": start_mono - 25_000_000 + index * 50_000_000,
                "timestamp_wall_ns": start_wall - 25_000_000 + index * 50_000_000,
                "left_arm": [index * 0.01 + joint for joint in range(7)],
                "right_arm": [-index * 0.01 - joint for joint in range(7)],
            }
            for index in range(42)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            arm_path = Path(temporary) / "arm_state.jsonl"
            arm_path.write_text("".join(json.dumps(row) + "\n" for row in arm_rows), encoding="utf-8")
            arm_samples, arm_state_ns = read_arm_states(arm_path)
        right_thumb = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        events = [
            {
                "event": "command_start", "timestamp_monotonic_ns": start_mono + 300_000_000,
                "device": "RIGHT_HAND", "command": "clench",
                "target": {"thumb_rotation": 1.0, "clench6_after": right_thumb},
            },
            {
                "event": "command_start", "timestamp_monotonic_ns": start_mono + 500_000_000,
                "device": "RIGHT_HAND", "command": "grasp_force",
                "target": {"strength": 1.0, "fingers": [1, 3, 4], "clench6_after": None, "grasp_mode_after": 1},
            },
            {
                "event": "command_start", "timestamp_monotonic_ns": start_mono + 1_300_000_000,
                "device": "RIGHT_HAND", "command": "clench",
                "target": {"value": [0.0] * 6, "clench6_after": [0.0] * 6, "grasp_mode_after": 0},
            },
        ]

        arrays = build_aligned_episode(camera, arm_samples, events, start_ns=start_mono, end_ns=end_mono)
        quality = quality_report(
            arrays,
            np.asarray(camera, dtype=np.int64),
            arm_state_ns,
            events,
            {"expert_status": "PASS", "raw_arm_topics_recorded": False, "raw_hand_topics_recorded": False},
        )

        self.assertEqual(ROSBAG_TOPICS, (TOP_RGB_TOPIC,))
        self.assertEqual(len(ROSBAG_TOPICS), 1)
        self.assertTrue(set(ROSBAG_TOPICS).isdisjoint(STATE_TOPICS["left_arm"]))
        self.assertTrue(set(ROSBAG_TOPICS).isdisjoint(STATE_TOPICS["right_arm"]))
        self.assertTrue(set(ROSBAG_TOPICS).isdisjoint(STATE_TOPICS["left_hand"]))
        self.assertTrue(set(ROSBAG_TOPICS).isdisjoint(STATE_TOPICS["right_hand"]))
        self.assertEqual(arrays["states"].shape, (11, 26))
        self.assertEqual(arrays["hybrid_actions"].shape, (10, 28))
        np.testing.assert_array_equal(arrays["hybrid_actions"][:, :26], arrays["states"][1:])
        np.testing.assert_array_equal(arrays["hybrid_actions"][:, 26:28], arrays["grasp_modes_at_state"][1:])
        np.testing.assert_allclose(np.diff(arrays["state_timestamps"]), 0.2)
        np.testing.assert_allclose(arrays["states"][0, :7], arm_rows[0]["left_arm"])
        np.testing.assert_allclose(arrays["states"][0, 7:14], arm_rows[0]["right_arm"])
        np.testing.assert_allclose(arrays["states"][1, :7], arm_rows[4]["left_arm"])
        self.assertLessEqual(float(np.max(arrays["camera_age_s"])), 1.0 / 24.0 + 1e-6)
        self.assertTrue(np.isfinite(arrays["states"]).all())
        self.assertTrue(np.any(arrays["states"][:, 21] == 1.0))
        self.assertTrue(np.any(arrays["grasp_modes_at_state"][:, 1] == 1.0))
        self.assertEqual(quality["camera_raw_count"], 27)
        self.assertAlmostEqual(quality["camera_raw_fps"], 12.0, places=5)
        self.assertEqual(quality["arm_state_count"], 42)
        self.assertAlmostEqual(quality["arm_state_effective_hz"], 20.0)
        self.assertEqual(quality["arm_state_source"], "passive_arm_state_jsonl")
        self.assertLessEqual(quality["arm_age_max_ms"], 50.0)
        self.assertEqual(quality["hand_state_source"], "command_hold_last")
        self.assertFalse(quality["raw_arm_topics_recorded"])
        self.assertFalse(quality["raw_hand_topics_recorded"])
        self.assertTrue(quality["accepted"])
        self.assertEqual(quality["task_profile"], "fixed_point_c_baseline")
        self.assertEqual(quality["visual_mode"], "async_latest_hold")
        self.assertFalse(quality["requires_visual_generalization"])

        low_rate_camera = np.asarray(camera[::12], dtype=np.int64)
        stale_camera_arrays = dict(arrays)
        stale_camera_arrays["camera_age_s"] = np.full(len(arrays["states"]), 0.6, dtype=np.float64)
        fixed_quality = quality_report(
            stale_camera_arrays,
            low_rate_camera,
            arm_state_ns,
            events,
            {
                "expert_status": "PASS",
                "task_profile": "fixed_point_c_baseline",
                "visual_mode": "async_latest_hold",
                "requires_visual_generalization": False,
                "hand_state_source": "command_hold_last",
            },
        )
        self.assertFalse(fixed_quality["checks"]["camera_raw_fps_gte_5"])
        self.assertFalse(fixed_quality["checks"]["camera_p95_age_lte_0_2s"])
        self.assertNotIn("camera_raw_fps_gte_5", fixed_quality["hard_acceptance_checks"])
        self.assertNotIn("camera_p95_age_lte_0_2s", fixed_quality["hard_acceptance_checks"])
        self.assertTrue(fixed_quality["accepted_for_training"])

        general_quality = quality_report(
            stale_camera_arrays,
            low_rate_camera,
            arm_state_ns,
            events,
            {"expert_status": "PASS", "task_profile": "generalization", "hand_state_source": "command_hold_last"},
        )
        self.assertFalse(general_quality["accepted_for_training"])

        invalid_camera_quality = quality_report(
            stale_camera_arrays,
            low_rate_camera[::-1],
            arm_state_ns,
            events,
            {"expert_status": "PASS", "task_profile": "fixed_point_c_baseline", "hand_state_source": "command_hold_last"},
        )
        self.assertFalse(invalid_camera_quality["checks"]["camera_timestamps_monotonic_and_valid"])
        self.assertFalse(invalid_camera_quality["accepted_for_training"])

    def test_passive_arm_logger_decimates_and_writes_sdk_order(self) -> None:
        passive = PassiveArmState.__new__(PassiveArmState)
        passive.latest = {"left_arm": {}, "right_arm": {}}
        passive.history = {"left_arm": [], "right_arm": []}
        passive.sequences = {"left_arm": 0, "right_arm": 0}
        passive.last_history_ns = {"left_arm": 0, "right_arm": 0}
        passive.history_period_ns = 200_000_000
        passive.arm_state_log_path = Path("mock_arm_state.jsonl")
        passive.arm_state_log_period_ns = 50_000_000
        passive.arm_state_log_last_ns = 0
        passive.arm_state_log_count = 0
        passive.arm_state_log_error = None
        passive.arm_state_log_stream = io.StringIO()
        passive.lock = threading.Lock()
        passive.condition = threading.Condition(passive.lock)
        clock = {"mono": 1_000_000_000, "wall": 1_800_000_000_000_000_000}

        def publish_all(offset: float) -> None:
            for arm in ("left_arm", "right_arm"):
                for index in range(7):
                    passive._callback(arm, index, SimpleNamespace(position=[offset + index]))

        with (
            mock.patch("tools.run_c_only_serial_expert.time.monotonic_ns", side_effect=lambda: clock["mono"]),
            mock.patch("tools.run_c_only_serial_expert.time.time_ns", side_effect=lambda: clock["wall"]),
        ):
            publish_all(0.0)
            clock["mono"] += 25_000_000
            clock["wall"] += 25_000_000
            publish_all(10.0)
            self.assertEqual(passive.arm_state_log_count, 1)
            clock["mono"] += 25_000_000
            clock["wall"] += 25_000_000
            with passive.condition:
                passive.latest = {
                    arm: {index: 20.0 + index for index in range(7)}
                    for arm in ("left_arm", "right_arm")
                }
            passive._callback("left_arm", 0, SimpleNamespace(position=[20.0]))

        rows = [json.loads(line) for line in passive.arm_state_log_stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["timestamp_monotonic_ns"] - rows[0]["timestamp_monotonic_ns"], 50_000_000)
        expected = [20.0 + index for index in PASSIVE_TO_SDK_INDICES]
        self.assertEqual(rows[1]["left_arm"], expected)
        self.assertEqual(rows[1]["right_arm"], expected)

        class BrokenStream:
            def write(self, _value):
                raise OSError("mock disk failure")

        passive.arm_state_log_stream = BrokenStream()
        clock["mono"] += 50_000_000
        with mock.patch("tools.run_c_only_serial_expert.time.monotonic_ns", return_value=clock["mono"]):
            passive._callback("left_arm", 0, SimpleNamespace(position=[99.0]))
        self.assertEqual(passive.latest["left_arm"][0], 99.0)
        self.assertIsNotNone(passive.arm_state_log_error)


if __name__ == "__main__":
    unittest.main()
