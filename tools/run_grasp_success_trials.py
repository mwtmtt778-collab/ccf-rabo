#!/usr/bin/env python3
"""Run a reset-gated 10-trial Nut B grasp success diagnostic.

The physical success proxy is intentionally stricter than an SDK return: the
right hand must complete grasp/lift/release/retreat and the top point cloud must
find Nut B at the release target.  Every trial is preceded by a reset command
sequence and observable-state verification.

``strict`` reset policy is the sampling-safe default.  It currently fails
closed because no verified world/scene, rigid-body velocity, or controller
state reset/readback API is known.  ``observable`` may be selected explicitly
for a diagnostic experiment; it is never reported as sampling-ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "grasp_success_trials"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.test_episode_reset import (  # noqa: E402
    BATCH_COLLECTOR_CONTRACT,
    RaboResetRuntime,
    ResetThresholds,
    audit_rabo_reset_capabilities,
    jsonable,
    reset_episode,
)
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    build_release_config,
    compute_statistics,
    execute_release_trial,
)


RESET_POLICIES = ("strict", "observable")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def last_verification(reset_result: dict[str, Any]) -> dict[str, Any]:
    attempts = reset_result.get("attempts") or []
    if not attempts:
        return {}
    return attempts[-1].get("verification") or {}


def evaluate_reset_gate(reset_result: dict[str, Any], policy: str) -> dict[str, Any]:
    verification = last_verification(reset_result)
    strict_ready = bool(verification.get("overall_reset_ready"))
    observable_ready = bool(verification.get("operational_observable_state_ready"))
    allowed = strict_ready if policy == "strict" else observable_ready
    return {
        "policy": policy,
        "allowed": allowed,
        "strict_reset_ready": strict_ready,
        "observable_state_ready": observable_ready,
        "sampling_ready": strict_ready,
        "classification": (
            "FULL_RESET_VERIFIED"
            if strict_ready
            else "OBSERVABLE_ONLY_DIAGNOSTIC" if observable_ready else "RESET_NOT_READY"
        ),
        "soft_reset_limitations": verification.get("soft_reset_limitations", []),
    }


def physical_proxy_success(trial: dict[str, Any]) -> bool:
    return all(
        bool(trial.get(key))
        for key in ("grasp_success", "release_success", "retreat_success", "detected")
    )


def summarize(trials: list[dict[str, Any]], requested: int, reset_policy: str) -> dict[str, Any]:
    attempted = [row for row in trials if row.get("experiment") is not None]
    successes = [row for row in attempted if row.get("physical_success_proxy")]
    strict_ready = all(row.get("reset_gate", {}).get("strict_reset_ready") for row in trials)
    completed = len(attempted) == requested
    return {
        "requested_trials": requested,
        "reset_passed_trials": sum(bool(row.get("reset_gate", {}).get("allowed")) for row in trials),
        "attempted_trials": len(attempted),
        "physical_proxy_successes": len(successes),
        "success_rate_of_attempted": len(successes) / len(attempted) if attempted else 0.0,
        "success_rate_of_requested": len(successes) / requested,
        "all_requested_trials_completed": completed,
        "reset_policy": reset_policy,
        "all_trials_strict_reset_ready": strict_ready and len(trials) == requested,
        "sampling_batch_allowed": completed and strict_ready and len(successes) == requested,
        "success_definition": (
            "grasp/lift/release/retreat commands completed and top-camera point cloud "
            "detected Nut B at the release target"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reset-gated 10-trial Nut B grasp success diagnostic.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--reset-policy", choices=RESET_POLICIES, default="strict")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually reset and move the Rabo simulation; without this flag only print the contract/audit.",
    )
    parser.add_argument("--max-reset-attempts", type=int, default=3)
    parser.add_argument("--reset-settle-s", type=float, default=ResetThresholds.settle_s)
    parser.add_argument("--reset-sample-interval-s", type=float, default=ResetThresholds.sample_interval_s)
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S)
    parser.add_argument("--hold-after-grasp", type=float, default=DEFAULT_HOLD_AFTER_GRASP_S)
    parser.add_argument("--vision-target-radius", type=float, default=DEFAULT_VISION_TARGET_RADIUS_M)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1 or args.max_reset_attempts < 1:
        raise SystemExit("--trials and --max-reset-attempts must be >= 1")
    if min(args.reset_settle_s, args.reset_sample_interval_s, args.settle_time, args.hold_after_grasp) < 0:
        raise SystemExit("settle/sample/hold durations must be >= 0")
    if args.vision_target_radius <= 0:
        raise SystemExit("--vision-target-radius must be > 0")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    audit = audit_rabo_reset_capabilities()
    contract = {
        "requested_trials": args.trials,
        "reset_policy": args.reset_policy,
        "strict_policy_note": "Required for sampling; fails closed unless complete Episode equivalence is verified.",
        "observable_policy_note": "Explicit diagnostic override only; never authorizes training-data collection.",
        "batch_collector_contract": BATCH_COLLECTOR_CONTRACT,
        "capability_audit": audit,
    }
    if not args.execute:
        print(json.dumps(jsonable(contract), ensure_ascii=False, indent=2))
        print("\nDRY RUN: add --execute only in the Rabo simulation workspace.")
        return 0

    started = datetime.now().astimezone()
    report_path = REPORT_DIR / f"grasp_success_trials_{started.strftime('%Y%m%d_%H%M%S')}.json"
    thresholds = ResetThresholds(
        settle_s=args.reset_settle_s,
        sample_interval_s=args.reset_sample_interval_s,
    )
    release_args = argparse.Namespace(
        release_xyz=None,
        release_z=-0.25,
        release_rpy=[0.0, 0.8, 0.0],
        settle_time=args.settle_time,
        hold_after_grasp=args.hold_after_grasp,
    )
    release_config = build_release_config(release_args)
    report: dict[str, Any] = {
        "started_at": started.isoformat(),
        "mode": "RESET_GATED_GRASP_SUCCESS_DIAGNOSTIC",
        "contract": contract,
        "thresholds": asdict(thresholds),
        "release_config": asdict(release_config),
        "trials": [],
    }

    runtime: RaboResetRuntime | None = None
    try:
        # Keep one SDK bundle for the whole experiment.  Rabo uses fixed ROS
        # node names per device; destroying and immediately recreating them can
        # leave duplicate publishers/clients and block the next motion call.
        runtime = RaboResetRuntime()
        for trial_id in range(1, args.trials + 1):
            print(f"\n[TRIAL {trial_id}/{args.trials}] reset -> verify", flush=True)
            reset_result = reset_episode(
                runtime,
                thresholds=thresholds,
                max_attempts=args.max_reset_attempts,
            )
            gate = evaluate_reset_gate(reset_result, args.reset_policy)
            row: dict[str, Any] = {
                "trial_id": trial_id,
                "reset": reset_result,
                "reset_gate": gate,
                "experiment": None,
                "physical_success_proxy": False,
            }
            report["trials"].append(row)
            write_report(report_path, report)
            print(f"[RESET_GATE] {gate['classification']} allowed={gate['allowed']}", flush=True)
            if not gate["allowed"]:
                report["stopped_early"] = True
                report["stop_reason"] = f"trial {trial_id} reset gate failed under {args.reset_policy} policy"
                break

            experiment = execute_release_trial(
                trial_id,
                args.trials,
                release_config,
                None,
                args.vision_target_radius,
                reset_before_trial=False,
                execution_bundle=runtime,
                check_reachability_after_release=False,
                shutdown_bundle_before_vision=False,
                shutdown_execution_bundle=False,
            )
            row["experiment"] = experiment
            row["physical_success_proxy"] = physical_proxy_success(experiment)
            write_report(report_path, report)
    except Exception as exc:
        report["stopped_early"] = True
        report["stop_reason"] = repr(exc)
    finally:
        if runtime is not None:
            runtime.shutdown()
        experiments = [row["experiment"] for row in report["trials"] if row.get("experiment") is not None]
        report["experiment_statistics"] = compute_statistics(experiments)
        report["summary"] = summarize(report["trials"], args.trials, args.reset_policy)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write_report(report_path, report)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if report["summary"]["all_requested_trials_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
