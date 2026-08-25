#!/usr/bin/env python3
"""No-Rabo tests for grasp experiment reset policy and summary semantics."""

from __future__ import annotations

import unittest

from tools.run_grasp_success_trials import evaluate_reset_gate, physical_proxy_success, summarize


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


if __name__ == "__main__":
    unittest.main()
