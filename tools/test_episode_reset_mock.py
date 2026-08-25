#!/usr/bin/env python3
"""No-Rabo regression tests for the fail-closed Episode reset gate."""

from __future__ import annotations

import unittest

from agents.three_nut_expert.config import HAND_OPEN, KNOWN_FIXED_NUT_WORLD_POSE
from agents.three_nut_expert.expert import pose_to_list
from tools.test_episode_reset import ResetThresholds, reset_episode, verify_reset


class MockJointDevice:
    def __init__(self, joints: list[float]) -> None:
        self.joints = list(joints)

    def move_joints(self, target: list[float]) -> bool:
        self.joints = list(target)
        return True

    def clench(self, *target: object) -> bool:
        values = target[0] if len(target) == 1 and isinstance(target[0], list) else target
        self.joints = [float(value) for value in values]
        return True

    def get_joint_angles(self) -> list[float]:
        return list(self.joints)

    def get_clench(self) -> list[float]:
        return list(self.joints)


class MockPoseSetter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def set(self, thing_id: str, pose: tuple[float, ...]) -> bool:
        self.calls.append((thing_id, pose))
        return True


class MockRuntime:
    def __init__(self) -> None:
        self.right_arm = MockJointDevice([9.0] * 7)
        self.left_arm = MockJointDevice([9.0] * 7)
        self.right_hand = MockJointDevice([1.0] * len(HAND_OPEN))
        self.left_hand = MockJointDevice([1.0] * len(HAND_OPEN))
        self.pose_setter = MockPoseSetter()


def nominal_capture() -> dict[str, list[float]]:
    return {
        key: pose_to_list(pose)[:3]
        for key, pose in KNOWN_FIXED_NUT_WORLD_POSE.items()
    }


class EpisodeResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ResetThresholds(
            sample_count=3,
            sample_interval_s=0.0,
            settle_s=0.0,
        )

    def test_observable_state_passes_but_complete_reset_fails_closed(self) -> None:
        runtime = MockRuntime()
        result = reset_episode(
            runtime,
            thresholds=self.thresholds,
            max_attempts=1,
            nut_capture=nominal_capture,
        )
        verification = result["attempts"][0]["verification"]
        self.assertTrue(verification["operational_observable_state_ready"])
        self.assertFalse(verification["overall_reset_ready"])
        self.assertEqual(result["status"], "RESET_FAILED")
        self.assertFalse(result["batch_collection_allowed"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertIn("RESET_FAILED", result["attempts"][0]["states"])
        self.assertNotIn("EPISODE_READY", result["attempts"][0]["states"])

    def test_all_three_nuts_and_storage_box_receive_pose_commands(self) -> None:
        runtime = MockRuntime()
        reset_episode(
            runtime,
            thresholds=self.thresholds,
            max_attempts=1,
            nut_capture=nominal_capture,
        )
        self.assertEqual(len(runtime.pose_setter.calls), 4)

    def test_nut_drift_fails_observable_gate(self) -> None:
        runtime = MockRuntime()

        def drifted_capture() -> dict[str, list[float]]:
            result = nominal_capture()
            result["C"][0] += 0.10
            return result

        verification = verify_reset(
            runtime,
            thresholds=self.thresholds,
            nut_capture=drifted_capture,
        )
        self.assertFalse(verification["nut_C_pose_ok"])
        self.assertFalse(verification["operational_observable_state_ready"])
        self.assertFalse(verification["overall_reset_ready"])

    def test_unstable_nut_fails_stability_gate(self) -> None:
        runtime = MockRuntime()
        count = 0

        def unstable_capture() -> dict[str, list[float]]:
            nonlocal count
            result = nominal_capture()
            result["B"][1] += 0.006 * count
            count += 1
            return result

        verification = verify_reset(
            runtime,
            thresholds=self.thresholds,
            nut_capture=unstable_capture,
        )
        self.assertFalse(verification["nut_B_stable"])
        self.assertFalse(verification["operational_observable_state_ready"])

    def test_observable_failure_uses_only_bounded_retries(self) -> None:
        runtime = MockRuntime()

        def missing_clusters() -> dict[str, list[float]]:
            raise RuntimeError("NO_POINTCLOUD")

        result = reset_episode(
            runtime,
            thresholds=self.thresholds,
            max_attempts=3,
            nut_capture=missing_clusters,
        )
        self.assertEqual(len(result["attempts"]), 3)
        self.assertTrue(all("RESET_FAILED" in row["states"] for row in result["attempts"]))
        self.assertFalse(result["batch_collection_allowed"])


if __name__ == "__main__":
    unittest.main()
