#!/usr/bin/env python3
"""No-Rabo tests for grasp experiment reset policy and summary semantics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.run_grasp_success_trials import evaluate_reset_gate, physical_proxy_success, summarize
from tools.test_right_release_stability import ReleaseConfig, execute_release_trial


def reset_result(*, strict: bool, observable: bool) -> dict:
    return {
        "attempts": [{
            "verification": {
                "overall_reset_ready": strict,
                "operational_observable_state_ready": observable,
                "soft_reset_limitations": ["NO_WORLD_RESET"] if observable and not strict else [],
            }
        }]
    }


class GraspSuccessTrialsTests(unittest.TestCase):
    def test_strict_policy_rejects_observable_only_reset(self) -> None:
        gate = evaluate_reset_gate(reset_result(strict=False, observable=True), "strict")
        self.assertFalse(gate["allowed"])
        self.assertEqual(gate["classification"], "OBSERVABLE_ONLY_DIAGNOSTIC")

    def test_observable_policy_allows_diagnostic_but_not_sampling(self) -> None:
        gate = evaluate_reset_gate(reset_result(strict=False, observable=True), "observable")
        self.assertTrue(gate["allowed"])
        self.assertFalse(gate["sampling_ready"])

    def test_proxy_requires_entire_visual_chain(self) -> None:
        complete = {key: True for key in ("grasp_success", "release_success", "retreat_success", "detected")}
        self.assertTrue(physical_proxy_success(complete))
        complete["detected"] = False
        self.assertFalse(physical_proxy_success(complete))

    def test_observable_trials_never_authorize_sampling(self) -> None:
        rows = [{
            "reset_gate": evaluate_reset_gate(reset_result(strict=False, observable=True), "observable"),
            "experiment": {},
            "physical_success_proxy": True,
        } for _ in range(10)]
        result = summarize(rows, 10, "observable")
        self.assertEqual(result["success_rate_of_requested"], 1.0)
        self.assertFalse(result["sampling_batch_allowed"])

    def test_release_trial_reuses_caller_bundle_and_skips_left_reconnect(self) -> None:
        class Device:
            def clench(self, *args, **kwargs):
                return True

            def grasp_force(self, **kwargs):
                return True

            def move_joints(self, joints):
                return True

            def move_to(self, **kwargs):
                return True

            def shutdown(self):
                return None

        class Bundle:
            right_arm = Device()
            right_hand = Device()

        config = ReleaseConfig(
            release_xyz=(0.1, 0.2, -0.25),
            release_rpy=(0.0, 0.8, 0.0),
            retreat_pose=((0.0,) * 7,),
            settle_time=0.0,
            target_nut_world_xyz=(1.0, 2.0, 3.0),
            hold_after_grasp_s=0.0,
        )
        vision = {"detected": True, "nut_world_xyz": [1.0, 2.0, 3.0]}
        with patch("tools.test_right_release_stability.make_right_bundle") as factory, patch(
            "tools.test_right_release_stability.detect_released_nut", return_value=vision
        ), patch("tools.test_right_release_stability.check_left_reachability") as reachability:
            result = execute_release_trial(
                1,
                1,
                config,
                None,
                0.08,
                reset_before_trial=False,
                execution_bundle=Bundle(),
                check_reachability_after_release=False,
                shutdown_bundle_before_vision=False,
                shutdown_execution_bundle=False,
            )
        factory.assert_not_called()
        reachability.assert_not_called()
        self.assertTrue(result["success"])
        self.assertIsNone(result["reachable"])


if __name__ == "__main__":
    unittest.main()
