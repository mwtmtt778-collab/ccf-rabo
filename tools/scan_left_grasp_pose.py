#!/usr/bin/env python3
"""Pose-check-only 4D scan for left-hand table grasp candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "left_pose_scan"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import DEVICE_IDS  # noqa: E402


DEFAULT_XY = [0.385, 0.038]
LEFT_TEST_XY_STATUS = "VERIFIED_BY_RABO_REACHABILITY_TEST"
Z_PRIOR = -0.33
Z_PRIOR_STATUS = "PREDICTED_FROM_RIGHT_SUCCESSFUL_GRASP"
RIGHT_NUT_B_GRASP_POSE = [-0.2803, 0.157, -0.33, 0.0, 0.8, 0.0]
RIGHT_NUT_B_GRASP_POSE_STATUS = "VERIFIED_RIGHT_NUT_B_SUCCESS_FROZEN"


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def run_id_text(suffix: str = "scan") -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + f"_{suffix}"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"UNKNOWN: {repr(exc)}"


def normalize_pose_check(raw: Any) -> tuple[str, str, Any]:
    value = jsonable(raw)
    if isinstance(raw, bool):
        return ("PASS" if raw else "FAIL", "reachable" if raw else "pose_check_returned_false", value)
    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        ok = lower.get("success", lower.get("ok", lower.get("reachable", lower.get("result"))))
        reason = lower.get("reason", lower.get("message", lower.get("error", "")))
        if isinstance(ok, bool):
            return ("PASS" if ok else "FAIL", str(reason or ("reachable" if ok else "not_reachable")), value)
    if isinstance(value, list) and value and isinstance(value[0], bool):
        return ("PASS" if value[0] else "FAIL", str(value[1] if len(value) > 1 else ""), value)
    text = str(value)
    low = text.lower()
    if "true" in low or "reachable" in low or "success" in low:
        return "PASS", text, value
    if "false" in low or "fail" in low or "limit" in low or "workspace" in low:
        return "FAIL", text, value
    return "CHECK", text, value


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    values: list[float] = []
    current = start
    while current <= stop + step * 0.5:
        values.append(round(current, 6))
        current += step
    return values


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angular_delta(a: float, b: float) -> float:
    return normalize_angle(a - b)


def angular_distance(a: float, b: float) -> float:
    return abs(angular_delta(a, b))


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(v), 6) for v in values})


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def default_pitch_values(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.pitch_positive:
        values.extend(frange(0.4, 1.2, args.pitch_step))
    if args.pitch_negative:
        values.extend(frange(-1.2, -0.4, args.pitch_step))
    return unique_sorted(values)


def build_scan_values(args: argparse.Namespace) -> dict[str, list[float]]:
    if args.fine:
        if args.center is None:
            raise SystemExit("--fine requires --center X Y Z R P YAW")
        center = args.center
        return {
            "z": frange(center[2] - args.fine_z_radius, center[2] + args.fine_z_radius, args.fine_z_step),
            "roll": frange(center[3] - args.fine_angle_radius, center[3] + args.fine_angle_radius, args.fine_angle_step),
            "pitch": frange(center[4] - args.fine_angle_radius, center[4] + args.fine_angle_radius, args.fine_angle_step),
            "yaw": frange(center[5] - args.fine_angle_radius, center[5] + args.fine_angle_radius, args.fine_angle_step),
        }
    pitch_values = unique_sorted(args.pitch_values) if args.pitch_values is not None else default_pitch_values(args)
    return {
        "z": frange(args.z_min, args.z_max, args.z_step),
        "roll": frange(args.roll_min, args.roll_max, args.roll_step),
        "pitch": pitch_values,
        "yaw": frange(args.yaw_min, args.yaw_max, args.yaw_step),
    }


def build_orientation_local_values(args: argparse.Namespace) -> dict[str, list[float]]:
    if args.center is None:
        raise SystemExit("--orientation-local requires --center X Y Z R P YAW")
    center = args.center
    args.xy = [center[0], center[1]]
    return {
        "z": frange(center[2] - args.local_z_radius, center[2] + args.local_z_radius, args.local_z_step),
        "roll": frange(center[3] - args.local_roll_radius, center[3] + args.local_roll_radius, args.local_roll_step),
        "pitch": frange(args.local_pitch_min, args.local_pitch_max, args.local_pitch_step),
        "yaw": local_yaw_values(args.local_yaw_step),
    }


def progressive_level_values(center: list[float], level: int) -> dict[str, list[float]]:
    if level == 1:
        return {
            "z": [round(center[2] - 0.01, 6), round(center[2], 6)],
            "roll": frange(-0.4, 0.4, 0.2),
            "pitch": frange(0.6, 1.0, 0.2),
            "yaw": [-round(math.pi, 6), -3.0, -2.8, 2.8, 3.0, round(math.pi, 6)],
        }
    if level == 2:
        return {
            "z": [round(center[2] - 0.01, 6), round(center[2], 6)],
            "roll": frange(-0.6, 0.6, 0.2),
            "pitch": frange(0.4, 1.2, 0.2),
            "yaw": [-round(math.pi, 6), -3.0, -2.8, -2.6, 2.6, 2.8, 3.0, round(math.pi, 6)],
        }
    return {
        "z": [round(center[2] - 0.01, 6), round(center[2], 6), round(center[2] + 0.01, 6)],
        "roll": frange(-0.8, 0.8, 0.2),
        "pitch": frange(0.2, 1.4, 0.2),
        "yaw": local_yaw_values(0.2),
    }


def local_yaw_values(step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("local yaw step must be positive")
    negative = [-round(math.pi, 6)]
    current = -3.0
    while current <= -2.2 + step * 0.1:
        negative.append(round(current, 6))
        current += step
    positive = []
    current = 2.2
    while current <= 3.0 + step * 0.1:
        positive.append(round(current, 6))
        current += step
    positive.append(round(math.pi, 6))
    return unique_sorted([*negative, *positive])


def make_pose(xy: list[float], z: float, roll: float, pitch: float, yaw: float) -> list[float]:
    return [round(xy[0], 6), round(xy[1], 6), round(z, 6), round(roll, 6), round(pitch, 6), round(yaw, 6)]


def approach_lift_pose(grasp_pose: list[float], dz: float) -> list[float]:
    return [grasp_pose[0], grasp_pose[1], round(grasp_pose[2] + dz, 6), grasp_pose[3], grasp_pose[4], grasp_pose[5]]


class PoseChecker:
    def __init__(self, mock: str | None) -> None:
        self.mock = mock
        self.arm = None
        self.attempts: dict[tuple[float, ...], int] = {}
        if mock is None:
            from rabo_robocap import LinkerArmA7

            self.arm = LinkerArmA7(robot_id=DEVICE_IDS["LEFT_ARM"], mode="sim")

    def shutdown(self) -> None:
        if self.arm is not None and hasattr(self.arm, "shutdown"):
            self.arm.shutdown()

    def check(self, pose: list[float]) -> tuple[str, str, Any]:
        if self.mock is not None:
            key = tuple(round(v, 6) for v in pose)
            self.attempts[key] = self.attempts.get(key, 0) + 1
            return self.mock_check(pose, self.attempts[key])
        raw = self.arm.pose_check(pose[0], pose[1], pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5])
        return normalize_pose_check(raw)

    def mock_check(self, pose: list[float], attempt: int = 1) -> tuple[str, str, Any]:
        _x, _y, z, roll, pitch, yaw = pose
        table_chain_z = -0.345 <= z <= -0.265
        if self.mock == "all-pass":
            ok = True
        elif self.mock == "progressive":
            stable = (
                -0.34 <= z <= -0.33
                and abs(roll) <= 0.2
                and 0.6 <= pitch <= 1.0
                and angular_distance(yaw, math.pi) <= 0.35
            )
            flaky = (
                -0.34 <= z <= -0.33
                and abs(roll) <= 0.4
                and 0.6 <= pitch <= 1.0
                and angular_distance(yaw, math.pi) <= 0.55
            )
            ok = stable or (flaky and attempt == 1)
        elif self.mock == "orientation-local":
            ok = (
                -0.34 <= z <= -0.32
                and -0.65 <= roll <= 0.65
                and 0.35 <= pitch <= 1.25
                and (angular_distance(yaw, math.pi) <= 0.75 or angular_distance(yaw, -2.4) <= 0.25)
            )
        elif self.mock == "pitch-sign":
            ok = table_chain_z and abs(roll) <= 0.25 and 0.55 <= pitch <= 1.05 and abs(yaw) <= 0.45
        else:
            positive = table_chain_z and abs(roll) <= 0.25 and 0.55 <= pitch <= 1.05 and abs(yaw) <= 0.45
            negative = table_chain_z and abs(roll) <= 0.2 and -1.0 <= pitch <= -0.6 and abs(yaw) <= 0.25
            ok = positive or negative
        return ("PASS" if ok else "FAIL", "mock_reachable" if ok else "mock_outside_stable_region", {"mock": self.mock, "ok": ok, "attempt": attempt})


def candidate_id(index: int) -> str:
    return f"left_grasp_{index:05d}"


def scan_candidates(args: argparse.Namespace, values: dict[str, list[float]]) -> list[dict[str, Any]]:
    checker = PoseChecker(args.mock)
    candidates: list[dict[str, Any]] = []
    index = 0
    total = math.prod(len(v) for v in values.values())
    started = time.time()
    full_chain_pass_count = 0
    try:
        for iz, z in enumerate(values["z"]):
            for ir, roll in enumerate(values["roll"]):
                for ip, pitch in enumerate(values["pitch"]):
                    for iy, yaw in enumerate(values["yaw"]):
                        grasp = make_pose(args.xy, z, roll, pitch, yaw)
                        approach = approach_lift_pose(grasp, args.approach_dz)
                        lift = approach_lift_pose(grasp, args.lift_dz)
                        approach_status, approach_reason, approach_raw = checker.check(approach)
                        grasp_status, grasp_reason, grasp_raw = checker.check(grasp)
                        lift_status, lift_reason, lift_raw = checker.check(lift)
                        full_chain_pass = approach_status == "PASS" and grasp_status == "PASS" and lift_status == "PASS"
                        candidates.append(
                            {
                                "candidate_id": candidate_id(index),
                                "grid_index": [iz, ir, ip, iy],
                                "grasp_pose": grasp,
                                "approach_pose": approach,
                                "lift_pose": lift,
                                "approach_reachable": approach_status,
                                "grasp_reachable": grasp_status,
                                "lift_reachable": lift_status,
                                "approach_reason": approach_reason,
                                "grasp_reason": grasp_reason,
                                "lift_reason": lift_reason,
                                "approach_raw": approach_raw,
                                "grasp_raw": grasp_raw,
                                "lift_raw": lift_raw,
                                "full_chain": "PASS" if full_chain_pass else "FAIL",
                                "result_status": "VERIFIED_BY_CURRENT_RUNTIME" if args.mock is None else "MOCK_VERIFIED_FOR_LOCAL_LOGIC_TEST_ONLY",
                                "robustness_score": 0.0,
                                "neighbor_count": 0,
                                "neighbor_full_chain_pass_count": 0,
                                "region_id": None,
                                "distance_to_edge": 0,
                                "orientation_prior_distance": orientation_prior_distance(grasp),
                            }
                        )
                        if full_chain_pass:
                            full_chain_pass_count += 1
                        index += 1
                        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == total):
                            print_progress(index, total, full_chain_pass_count, started)
        return candidates
    finally:
        checker.shutdown()


def print_progress(done: int, total: int, pass_count: int, started: float) -> None:
    elapsed = max(0.001, time.time() - started)
    rate = done / elapsed
    remaining = max(0, total - done)
    eta_s = remaining / rate if rate > 0 else 0.0
    percent = 100.0 * done / total if total else 100.0
    print(
        f"SCAN_PROGRESS: {done}/{total} ({percent:.1f}%) "
        f"FULL_CHAIN_PASS={pass_count} elapsed={elapsed:.1f}s eta={eta_s:.1f}s",
        flush=True,
    )


class ProgressDisplay:
    def __init__(self, total: int, interval: int) -> None:
        self.total = total
        self.interval = max(1, interval)
        self.started = time.monotonic()

    def stats(self, done: int) -> tuple[float, float, float]:
        elapsed = max(0.001, time.monotonic() - self.started)
        rate = done / elapsed if done else 0.0
        remaining = max(0, self.total - done)
        eta = remaining / rate if rate > 0 else 0.0
        percent = 100.0 * done / self.total if self.total else 100.0
        return elapsed, eta, percent

    def update(self, done: int, pass_count: int, fail_count: int) -> None:
        elapsed, eta, percent = self.stats(done)
        width = 24
        filled = int(width * done / self.total) if self.total else width
        bar = "█" * filled + "-" * (width - filled)
        print(
            f"Scan [{bar}] {percent:5.1f}% | {done}/{self.total} | "
            f"PASS {pass_count} | FAIL {fail_count} | elapsed {human_duration(elapsed)} | ETA {human_duration(eta)}",
            end="\r",
            flush=True,
        )

    def checkpoint(self, done: int, pass_count: int, fail_count: int, current_pose: list[float]) -> None:
        elapsed, eta, _percent = self.stats(done)
        print("")
        print("[checkpoint]")
        print(f"processed={done}/{self.total}")
        print(f"grasp_pass={pass_count}")
        print(f"grasp_fail={fail_count}")
        print(f"elapsed={human_duration(elapsed)}")
        print(f"eta={human_duration(eta)}")
        print(f"current_pose={current_pose}")

    def complete_line(self) -> None:
        print("")


def orientation_distance(a: list[float], b: list[float]) -> float:
    dz = (a[2] - b[2]) / 0.01
    dr = angular_distance(a[3], b[3])
    dp = angular_distance(a[4], b[4])
    dy = angular_distance(a[5], b[5])
    return math.sqrt(0.05 * dz * dz + dr * dr + dp * dp + dy * dy)


def orientation_deltas(pose: list[float], center: list[float]) -> dict[str, float]:
    return {
        "delta_z": round(pose[2] - center[2], 6),
        "delta_roll": round(angular_delta(pose[3], center[3]), 6),
        "delta_pitch": round(angular_delta(pose[4], center[4]), 6),
        "delta_yaw_wrapped": round(angular_delta(pose[5], center[5]), 6),
    }


def print_orientation_scan_start(center: list[float], total: int) -> None:
    print("=" * 60)
    print("LEFT GRASP LOCAL ORIENTATION SCAN")
    print("=" * 60)
    print("")
    print("CENTER:")
    print(center)
    print("")
    print("TOTAL CANDIDATES:")
    print(total)
    print("")
    print("MODE:")
    print("GRASP_POSE_CHECK_ONLY")
    print("")


def scan_orientation_local(args: argparse.Namespace, values: dict[str, list[float]]) -> tuple[list[dict[str, Any]], str, float]:
    checker = PoseChecker(args.mock)
    candidates: list[dict[str, Any]] = []
    center = [round(v, 6) for v in args.center]
    total = math.prod(len(v) for v in values.values())
    progress = ProgressDisplay(total, args.progress_interval)
    index = 0
    pass_count = 0
    fail_count = 0
    status = "COMPLETE"
    print_orientation_scan_start(center, total)
    try:
        for iz, z in enumerate(values["z"]):
            for ir, roll in enumerate(values["roll"]):
                for ip, pitch in enumerate(values["pitch"]):
                    for iy, yaw in enumerate(values["yaw"]):
                        pose = make_pose(args.xy, z, roll, pitch, yaw)
                        grasp_status, grasp_reason, grasp_raw = checker.check(pose)
                        is_pass = grasp_status == "PASS"
                        if is_pass:
                            pass_count += 1
                        else:
                            fail_count += 1
                        deltas = orientation_deltas(pose, center)
                        candidates.append(
                            {
                                "candidate_id": candidate_id(index),
                                "grid_index": [iz, ir, ip, iy],
                                "pose": pose,
                                "grasp_pose": pose,
                                "approach_pose": "NOT_APPLICABLE",
                                "lift_pose": "NOT_APPLICABLE",
                                "grasp_reachable": grasp_status,
                                "grasp_reason": grasp_reason,
                                "grasp_raw": grasp_raw,
                                "approach_reachable": "NOT_TESTED",
                                "lift_reachable": "NOT_TESTED",
                                "approach_reason": "NOT_TESTED",
                                "lift_reason": "NOT_TESTED",
                                "full_chain": "NOT_APPLICABLE_GRASP_ONLY",
                                "local_grasp_pass": is_pass,
                                "result_status": "VERIFIED_BY_CURRENT_RUNTIME" if args.mock is None else "MOCK_VERIFIED_FOR_LOCAL_LOGIC_TEST_ONLY",
                                **deltas,
                                "orientation_distance_to_center": round(orientation_distance(pose, center), 6),
                                "robustness_score": 0.0,
                                "neighbor_count": 0,
                                "neighbor_grasp_pass_count": 0,
                                "distance_to_edge": 0,
                                "selection_label": "IK_REACHABLE_ORIENTATION_CANDIDATE" if is_pass else "GRASP_POSE_CHECK_FAIL",
                            }
                        )
                        index += 1
                        if args.progress_interval > 0 and (index == 1 or index % args.progress_interval == 0 or index == total):
                            progress.update(index, pass_count, fail_count)
                            if index % args.progress_interval == 0 or index == total:
                                progress.checkpoint(index, pass_count, fail_count, pose)
                        if args.mock_interrupt_after is not None and index >= args.mock_interrupt_after:
                            raise KeyboardInterrupt
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        progress.complete_line()
        print("SCAN INTERRUPTED")
        print(f"processed_candidates={index}")
        print(f"total_candidates={total}")
        print(f"grasp_pass={pass_count}")
        print(f"grasp_fail={fail_count}")
    finally:
        progress.complete_line()
        elapsed = time.monotonic() - progress.started
        checker.shutdown()
    return candidates, status, elapsed


def print_phase(name: str) -> None:
    print("")
    print("=" * 60)
    print(name)
    print("=" * 60)


def total_level_candidates(values: dict[str, list[float]]) -> int:
    return math.prod(len(v) for v in values.values())


def print_progressive_workload(level: int, values: dict[str, list[float]], args: argparse.Namespace) -> None:
    total = total_level_candidates(values)
    print(f"LEVEL {level} CANDIDATES:\n{total}")
    print(f"INITIAL POSE_CHECK CALLS:\n{total * args.initial_checks}")
    print(f"MAX ADDITIONAL RECHECKS:\n{total * args.repeat_checks}")
    print(f"MAX TOTAL:\n{total * (args.initial_checks + args.repeat_checks)}")


def progressive_pose_items(args: argparse.Namespace, values: dict[str, list[float]], seen_poses: set[tuple[float, ...]]) -> list[tuple[list[int], list[float]]]:
    items = []
    for iz, z in enumerate(values["z"]):
        for ir, roll in enumerate(values["roll"]):
            for ip, pitch in enumerate(values["pitch"]):
                for iy, yaw in enumerate(values["yaw"]):
                    pose = make_pose(args.xy, z, roll, pitch, yaw)
                    pose_key = tuple(pose)
                    if pose_key in seen_poses:
                        continue
                    items.append(([iz, ir, ip, iy], pose))
    return items


def print_progressive_unique_workload(raw_total: int, unique_total: int, args: argparse.Namespace) -> None:
    if unique_total != raw_total:
        print(f"UNIQUE NEW CANDIDATES:\n{unique_total}")
        print(f"SKIPPED PREVIOUSLY CHECKED:\n{raw_total - unique_total}")
        print(f"UNIQUE INITIAL POSE_CHECK CALLS:\n{unique_total * args.initial_checks}")
        print(f"UNIQUE MAX ADDITIONAL RECHECKS:\n{unique_total * args.repeat_checks}")


def run_pose_check_attempts(checker: PoseChecker, pose: list[float], count: int, start_attempt: int) -> list[dict[str, Any]]:
    records = []
    for offset in range(count):
        attempt_id = start_attempt + offset
        status, reason, raw = checker.check(pose)
        records.append(
            {
                "attempt_id": attempt_id,
                "pose": list(pose),
                "arm": "LEFT_ARM",
                "frame": "left_arm_base_link",
                "result": status,
                "reason": reason,
                "raw": raw,
                "timestamp": now_text(),
            }
        )
    return records


def pose_inputs_identical(attempts: list[dict[str, Any]]) -> bool:
    if not attempts:
        return True
    first = attempts[0]["pose"]
    return all(row["pose"] == first for row in attempts)


def make_progressive_candidate(index: int, level: int, grid_index: list[int], pose: list[float], center: list[float]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(index),
        "scan_level": level,
        "grid_index": grid_index,
        "pose": pose,
        "grasp_pose": pose,
        "approach_pose": "NOT_APPLICABLE",
        "lift_pose": "NOT_APPLICABLE",
        "approach_reachable": "NOT_TESTED",
        "lift_reachable": "NOT_TESTED",
        "approach_reason": "NOT_TESTED",
        "lift_reason": "NOT_TESTED",
        "full_chain": "NOT_APPLICABLE_GRASP_ONLY",
        **orientation_deltas(pose, center),
        "orientation_distance_to_center": round(orientation_distance(pose, center), 6),
        "initial_checks": [],
        "final_checks": [],
        "total_pass": 0,
        "total_checks": 0,
        "ik_repeatability": 0.0,
        "ik_repeatability_text": "0/0 = 0.0",
        "pose_inputs_identical": "NOT_TESTED",
        "grasp_reachable": "NOT_TESTED",
        "grasp_reason": "NOT_TESTED",
        "local_grasp_pass": False,
        "neighbor_robustness": 0.0,
        "robustness_score": 0.0,
        "neighbor_count": 0,
        "neighbor_grasp_pass_count": 0,
        "distance_to_edge": 0,
        "selection_label": "NOT_TESTED",
    }


def finalize_progressive_candidates(candidates: list[dict[str, Any]], values: dict[str, list[float]], args: argparse.Namespace) -> None:
    for candidate in candidates:
        attempts = [*candidate["initial_checks"], *candidate["final_checks"]]
        total_pass = sum(1 for row in attempts if row["result"] == "PASS")
        total_checks = len(attempts)
        candidate["total_pass"] = total_pass
        candidate["total_checks"] = total_checks
        candidate["ik_repeatability"] = round(total_pass / total_checks, 6) if total_checks else 0.0
        candidate["ik_repeatability_text"] = f"{total_pass}/{total_checks} = {candidate['ik_repeatability']}"
        candidate["pose_inputs_identical"] = pose_inputs_identical(attempts)
        candidate["local_grasp_pass"] = candidate["ik_repeatability"] >= args.stable_threshold
        if candidate["local_grasp_pass"]:
            candidate["grasp_reachable"] = "PASS"
            candidate["selection_label"] = "IK_STABLE_CANDIDATE"
        elif total_pass > 0:
            candidate["grasp_reachable"] = "UNSTABLE"
            candidate["selection_label"] = "UNSTABLE_IK_CANDIDATE"
        else:
            candidate["grasp_reachable"] = "FAIL"
            candidate["selection_label"] = "UNSTABLE_IK_CANDIDATE"
        candidate["grasp_reason"] = attempts[-1]["reason"] if attempts else "NOT_TESTED"
    analyze_orientation_local(candidates, values)
    for candidate in candidates:
        candidate["neighbor_robustness"] = candidate["robustness_score"]


def scan_progressive_orientation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], str, float]:
    center = [round(v, 6) for v in args.center]
    args.xy = [center[0], center[1]]
    checker = PoseChecker(args.mock)
    started = time.monotonic()
    all_candidates: list[dict[str, Any]] = []
    status = "COMPLETE"
    stop_reason = "MAX_LEVEL_REACHED"
    selected_level = 0
    total_pose_check_calls = 0
    seen_poses: set[tuple[float, ...]] = set()
    try:
        print_phase("PHASE A: candidate generation")
        for level in range(1, args.max_level + 1):
            values = progressive_level_values(center, level)
            raw_total = total_level_candidates(values)
            pose_items = progressive_pose_items(args, values, seen_poses)
            total = len(pose_items)
            print(f"\nLEVEL {level}")
            print_progressive_workload(level, values, args)
            print_progressive_unique_workload(raw_total, total, args)
            progress = ProgressDisplay(total, args.progress_interval)
            level_candidates: list[dict[str, Any]] = []
            initial_pass_count = 0
            stable_2_of_2 = 0
            rejected_unstable = 0
            index_base = len(all_candidates)
            try:
                print_phase("PHASE B: initial 2x pose_check")
                processed = 0
                for grid_index, pose in pose_items:
                    seen_poses.add(tuple(pose))
                    candidate = make_progressive_candidate(index_base + processed, level, grid_index, pose, center)
                    candidate["initial_checks"] = run_pose_check_attempts(checker, pose, args.initial_checks, 1)
                    total_pose_check_calls += args.initial_checks
                    initial_passes = sum(1 for row in candidate["initial_checks"] if row["result"] == "PASS")
                    if initial_passes == args.initial_checks:
                        initial_pass_count += 1
                        stable_2_of_2 += 1
                    else:
                        rejected_unstable += 1
                        candidate["selection_label"] = "UNSTABLE_IK_CANDIDATE"
                    processed += 1
                    level_candidates.append(candidate)
                    if args.progress_interval > 0 and (processed == 1 or processed % args.progress_interval == 0 or processed == total):
                        progress.update(processed, stable_2_of_2, rejected_unstable)
                        if processed % args.progress_interval == 0 or processed == total:
                            progress.checkpoint(processed, stable_2_of_2, rejected_unstable, pose)
                    if args.mock_interrupt_after is not None and processed >= args.mock_interrupt_after:
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                status = "INTERRUPTED"
                progress.complete_line()
                print("SCAN INTERRUPTED")
                finalize_progressive_candidates(level_candidates, values, args)
                all_candidates.extend(level_candidates)
                selected_level = level
                stop_reason = "INTERRUPTED"
                break
            finally:
                progress.complete_line()

            print_phase("PHASE C: repeatability 3x re-check")
            verify_pool = [c for c in level_candidates if sum(1 for row in c["initial_checks"] if row["result"] == "PASS") == args.initial_checks]
            verify_started = time.monotonic()
            for idx, candidate in enumerate(verify_pool, start=1):
                elapsed_verify = max(0.001, time.monotonic() - verify_started)
                rate = idx / elapsed_verify
                remaining = max(0, len(verify_pool) - idx)
                eta = remaining / rate if rate > 0 else 0.0
                if args.progress_interval > 0 and (idx == 1 or idx % args.progress_interval == 0 or idx == len(verify_pool)):
                    print(
                        f"Verify stable candidates {idx}/{len(verify_pool)} | "
                        f"repeatability checks {args.initial_checks + 1}-{args.initial_checks + args.repeat_checks}/"
                        f"{args.initial_checks + args.repeat_checks} | "
                        f"elapsed {human_duration(elapsed_verify)} | ETA {human_duration(eta)}",
                        end="\r",
                        flush=True,
                    )
                    if idx % args.progress_interval == 0 or idx == len(verify_pool):
                        print("")
                        print("[checkpoint]")
                        print(f"verify_candidate={idx}/{len(verify_pool)}")
                        print(f"elapsed={human_duration(elapsed_verify)}")
                        print(f"eta={human_duration(eta)}")
                        print(f"current_pose={candidate['pose']}")
                candidate["final_checks"] = run_pose_check_attempts(checker, candidate["pose"], args.repeat_checks, args.initial_checks + 1)
                total_pose_check_calls += args.repeat_checks
            if verify_pool:
                print("")
            finalize_progressive_candidates(level_candidates, values, args)
            print_phase("PHASE D: ranking and diversity selection")
            top = select_representative_orientations(level_candidates, center, args.top_k)
            stable_count = sum(1 for c in level_candidates if c["selection_label"] == "IK_STABLE_CANDIDATE")
            print(f"LEVEL {level} COMPLETE")
            print(f"stable candidates = {stable_count}")
            all_candidates.extend(level_candidates)
            selected_level = level
            if stable_count >= args.min_stable_candidates and len(top) >= min(args.top_k, args.min_stable_candidates):
                stop_reason = "ENOUGH_STABLE_CANDIDATES"
                print("STOPPING:\nenough stable candidates found")
                break
            if level == 2 and args.max_level < 3:
                stop_reason = "MAX_LEVEL_REACHED"
        print_phase("PHASE E: report generation")
    finally:
        checker.shutdown()
    elapsed = time.monotonic() - started
    meta = {
        "selected_level": selected_level,
        "stop_reason": stop_reason,
        "total_pose_check_calls": total_pose_check_calls,
        "total_unique_candidates": len(all_candidates),
    }
    return all_candidates, meta, status, elapsed


def orientation_prior_distance(pose: list[float]) -> float:
    _x, _y, z, roll, pitch, yaw = pose
    pitch_distance = min(abs(pitch - 0.8), abs(pitch + 0.8))
    z_distance = abs(z - Z_PRIOR) / 0.01
    return math.sqrt(roll * roll + pitch_distance * pitch_distance + yaw * yaw + 0.05 * z_distance * z_distance)


def neighbor_offsets_4d() -> list[tuple[int, int, int, int]]:
    offsets = []
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            for dp in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (dz, dr, dp, dy) != (0, 0, 0, 0):
                        offsets.append((dz, dr, dp, dy))
    return offsets


def axis_offsets_4d() -> list[tuple[int, int, int, int]]:
    return [
        (-1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 1, 0),
        (0, 0, 0, -1),
        (0, 0, 0, 1),
    ]


def add_index(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def in_bounds(key: tuple[int, int, int, int], shape: tuple[int, int, int, int]) -> bool:
    return all(0 <= key[i] < shape[i] for i in range(4))


def analyze_grid(candidates: list[dict[str, Any]], values: dict[str, list[float]]) -> list[dict[str, Any]]:
    shape = (len(values["z"]), len(values["roll"]), len(values["pitch"]), len(values["yaw"]))
    by_key = {tuple(c["grid_index"]): c for c in candidates}
    pass_keys = {key for key, c in by_key.items() if c["full_chain"] == "PASS"}

    all_offsets = neighbor_offsets_4d()
    for key, candidate in by_key.items():
        neighbors = [add_index(key, offset) for offset in all_offsets]
        valid_neighbors = [n for n in neighbors if in_bounds(n, shape)]
        pass_count = sum(1 for n in valid_neighbors if n in pass_keys)
        candidate["neighbor_count"] = len(valid_neighbors)
        candidate["neighbor_full_chain_pass_count"] = pass_count
        candidate["robustness_score"] = round(pass_count / len(valid_neighbors), 6) if valid_neighbors else 0.0

    regions: list[dict[str, Any]] = []
    visited: set[tuple[int, int, int, int]] = set()
    for start in sorted(pass_keys):
        if start in visited:
            continue
        queue: deque[tuple[int, int, int, int]] = deque([start])
        visited.add(start)
        keys: list[tuple[int, int, int, int]] = []
        while queue:
            key = queue.popleft()
            keys.append(key)
            for offset in axis_offsets_4d():
                nxt = add_index(key, offset)
                if nxt in pass_keys and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        region_id = f"Region_{len(regions) + 1}"
        for key in keys:
            by_key[key]["region_id"] = region_id
        regions.append(build_region(region_id, keys, by_key, values, shape, pass_keys))

    distance_to_edge(by_key, regions, pass_keys, shape)
    regions.sort(key=lambda r: (r["candidate_count"], r["mean_robustness"], r["best_candidate"]["robustness_score"]), reverse=True)
    for index, region in enumerate(regions, start=1):
        new_id = f"Region_{index}"
        region["region_id"] = new_id
        for key in region["keys"]:
            by_key[tuple(key)]["region_id"] = new_id
    return regions


def analyze_orientation_local(candidates: list[dict[str, Any]], values: dict[str, list[float]]) -> None:
    z_step = infer_step(values["z"])
    roll_step = infer_step(values["roll"])
    pitch_step = infer_step(values["pitch"])
    yaw_step = infer_wrapped_yaw_step(values["yaw"])
    for candidate in candidates:
        pose = candidate["pose"]
        neighbors = [
            other
            for other in candidates
            if other is not candidate
            and abs(other["pose"][2] - pose[2]) <= z_step * 1.1
            and angular_distance(other["pose"][3], pose[3]) <= roll_step * 1.1
            and angular_distance(other["pose"][4], pose[4]) <= pitch_step * 1.1
            and angular_distance(other["pose"][5], pose[5]) <= yaw_step * 1.1
        ]
        pass_count = sum(1 for other in neighbors if other["local_grasp_pass"])
        candidate["neighbor_count"] = len(neighbors)
        candidate["neighbor_grasp_pass_count"] = pass_count
        candidate["robustness_score"] = round(pass_count / len(neighbors), 6) if neighbors else 0.0
        candidate["distance_to_edge"] = 1 if candidate["local_grasp_pass"] and candidate["robustness_score"] >= 0.99 else 0


def infer_step(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1) if abs(values[i + 1] - values[i]) > 1e-9]
    return min(diffs) if diffs else 1.0


def infer_wrapped_yaw_step(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = []
    for i, value in enumerate(values):
        for other in values[i + 1:]:
            diff = angular_distance(value, other)
            if diff > 1e-9:
                diffs.append(diff)
    return min(diffs) if diffs else 1.0


def local_distance_to_edge(
    key: tuple[int, int, int, int],
    pass_keys: set[tuple[int, int, int, int]],
    shape: tuple[int, int, int, int],
) -> int:
    min_steps = None
    for offset in axis_offsets_4d():
        steps = 0
        current = key
        while True:
            nxt = add_index(current, offset)
            if not in_bounds(nxt, shape) or nxt not in pass_keys:
                break
            steps += 1
            current = nxt
        min_steps = steps if min_steps is None else min(min_steps, steps)
    return int(min_steps or 0)


def select_representative_orientations(candidates: list[dict[str, Any]], center: list[float], top_k: int) -> list[dict[str, Any]]:
    passed = [c for c in candidates if c["local_grasp_pass"]]
    if not passed:
        return []
    ik_stable = [c for c in passed if c.get("selection_label") == "IK_STABLE_CANDIDATE"]
    if ik_stable:
        passed = ik_stable
    passed = sorted(
        passed,
        key=lambda c: (c.get("ik_repeatability", 0.0), c.get("neighbor_robustness", c.get("robustness_score", 0.0)), c.get("distance_to_edge", 0)),
        reverse=True,
    )
    best_repeatability = passed[0].get("ik_repeatability", 0.0)
    priority_pool = [c for c in passed if c.get("ik_repeatability", 0.0) == best_repeatability]
    if len(priority_pool) < top_k:
        priority_pool = passed
    passed = priority_pool
    stable = [c for c in passed if c.get("neighbor_robustness", c.get("robustness_score", 0.0)) > 0.0]
    pool = stable or passed
    first = max(pool, key=lambda c: (c.get("ik_repeatability", 0.0), c.get("neighbor_robustness", c.get("robustness_score", 0.0)), -c["orientation_distance_to_center"]))
    selected = [first]
    remaining = [c for c in pool if c is not first]
    while remaining and len(selected) < top_k:
        def score(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
            min_distance = min(orientation_distance(candidate["pose"], chosen["pose"]) for chosen in selected)
            return (
                candidate.get("ik_repeatability", 0.0),
                min_distance,
                candidate.get("neighbor_robustness", candidate.get("robustness_score", 0.0)),
                candidate["distance_to_edge"],
            )

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining = [c for c in remaining if c is not chosen]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected


def distance_to_edge(
    by_key: dict[tuple[int, int, int, int], dict[str, Any]],
    regions: list[dict[str, Any]],
    pass_keys: set[tuple[int, int, int, int]],
    shape: tuple[int, int, int, int],
) -> None:
    for region in regions:
        region_keys = {tuple(key) for key in region["keys"]}
        for key in region_keys:
            min_steps = None
            for offset in axis_offsets_4d():
                steps = 0
                current = key
                while True:
                    nxt = add_index(current, offset)
                    if not in_bounds(nxt, shape) or nxt not in pass_keys or nxt not in region_keys:
                        break
                    steps += 1
                    current = nxt
                min_steps = steps if min_steps is None else min(min_steps, steps)
            by_key[key]["distance_to_edge"] = int(min_steps or 0)


def build_region(
    region_id: str,
    keys: list[tuple[int, int, int, int]],
    by_key: dict[tuple[int, int, int, int], dict[str, Any]],
    values: dict[str, list[float]],
    shape: tuple[int, int, int, int],
    pass_keys: set[tuple[int, int, int, int]],
) -> dict[str, Any]:
    candidates = [by_key[key] for key in keys]
    z_values = [values["z"][key[0]] for key in keys]
    roll_values = [values["roll"][key[1]] for key in keys]
    pitch_values = [values["pitch"][key[2]] for key in keys]
    yaw_values = [values["yaw"][key[3]] for key in keys]
    best = sorted(
        candidates,
        key=lambda c: (c["robustness_score"], c["neighbor_full_chain_pass_count"], -c["orientation_prior_distance"]),
        reverse=True,
    )[0]
    edge_touch_count = 0
    for key in keys:
        if any(not in_bounds(add_index(key, offset), shape) or add_index(key, offset) not in pass_keys for offset in axis_offsets_4d()):
            edge_touch_count += 1
    return {
        "region_id": region_id,
        "keys": [list(key) for key in keys],
        "candidate_count": len(keys),
        "center_pose": best["grasp_pose"],
        "mean_robustness": round(sum(c["robustness_score"] for c in candidates) / len(candidates), 6),
        "edge_touch_count": edge_touch_count,
        "z_range": [min(z_values), max(z_values)],
        "roll_range": [min(roll_values), max(roll_values)],
        "pitch_range": [min(pitch_values), max(pitch_values)],
        "yaw_range": [min(yaw_values), max(yaw_values)],
        "best_candidate": {
            "candidate_id": best["candidate_id"],
            "grasp_pose": best["grasp_pose"],
            "robustness_score": best["robustness_score"],
            "orientation_prior_distance": best["orientation_prior_distance"],
        },
    }


def rank_candidates(candidates: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    passed = [c for c in candidates if c["full_chain"] == "PASS"]
    ranked = sorted(
        passed,
        key=lambda c: (
            c["robustness_score"],
            c["distance_to_edge"],
            c["neighbor_full_chain_pass_count"],
            -c["orientation_prior_distance"],
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked[:limit], start=1):
        candidate["rank"] = rank
    return ranked[:limit]


def build_summary(args: argparse.Namespace, values: dict[str, list[float]], candidates: list[dict[str, Any]], regions: list[dict[str, Any]], top: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for c in candidates if c["full_chain"] == "PASS")
    return {
        "metadata": {
            "run_id": run_id_text(),
            "timestamp": now_text(),
            "git_commit": get_git_commit(),
            "argv": [sys.executable, *sys.argv],
            "mock": args.mock,
        },
        "status_labels": {
            "LEFT_TEST_XY": {"value": args.xy, "status": LEFT_TEST_XY_STATUS},
            "Z_SEARCH_CENTER": {"value": Z_PRIOR, "status": Z_PRIOR_STATUS},
            "RIGHT_NUT_B_GRASP_POSE": {"value": RIGHT_NUT_B_GRASP_POSE, "status": RIGHT_NUT_B_GRASP_POSE_STATUS},
            "POSE_CHECK_RESULT": "VERIFIED_BY_CURRENT_RUNTIME" if args.mock is None else "MOCK_VERIFIED_FOR_LOCAL_LOGIC_TEST_ONLY",
        },
        "command_inputs": vars(args),
        "scan_values": values,
        "counts": {
            "candidate_count": len(candidates),
            "full_chain_pass_count": pass_count,
            "full_chain_fail_count": len(candidates) - pass_count,
            "region_count": len(regions),
        },
        "regions": [{k: v for k, v in region.items() if k != "keys"} for region in regions],
        "top_candidates": [compact_candidate(c) for c in top],
        "recommended_left_grasp_pose": {
            "value": top[0]["grasp_pose"] if top else None,
            "status": "POSE_CHECK_ROBUST_CANDIDATE" if top else "NO_FULL_CHAIN_PASS",
            "robustness": top[0]["robustness_score"] if top else None,
            "region": top[0]["region_id"] if top else None,
        },
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rank",
        "candidate_id",
        "grasp_pose",
        "approach_pose",
        "lift_pose",
        "full_chain",
        "robustness_score",
        "neighbor_count",
        "neighbor_full_chain_pass_count",
        "region_id",
        "distance_to_edge",
        "orientation_prior_distance",
        "approach_reason",
        "grasp_reason",
        "lift_reason",
    ]
    return {key: candidate.get(key) for key in keys}


def compact_orientation_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rank",
        "candidate_id",
        "pose",
        "grasp_reachable",
        "grasp_reason",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw_wrapped",
        "orientation_distance_to_center",
        "neighbor_robustness",
        "robustness_score",
        "ik_repeatability",
        "ik_repeatability_text",
        "total_pass",
        "total_checks",
        "initial_checks",
        "final_checks",
        "pose_inputs_identical",
        "neighbor_count",
        "neighbor_grasp_pass_count",
        "distance_to_edge",
        "selection_label",
    ]
    return {key: candidate.get(key) for key in keys}


def write_reports(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = summary["metadata"]["run_id"]
    json_path = REPORT_DIR / f"{run_id}.json"
    csv_path = REPORT_DIR / f"{run_id}.csv"
    md_path = REPORT_DIR / f"{run_id}.md"
    payload = {**summary, "candidates": candidates}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_csv(csv_path, candidates)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_markdown(summary))
    return json_path, csv_path, md_path


def build_orientation_summary(
    args: argparse.Namespace,
    values: dict[str, list[float]],
    candidates: list[dict[str, Any]],
    top: list[dict[str, Any]],
    scan_status: str,
    elapsed_s: float,
    progressive_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pass_count = sum(1 for c in candidates if c["local_grasp_pass"])
    processed = len(candidates)
    total = progressive_meta.get("total_unique_candidates", processed) if progressive_meta else math.prod(len(v) for v in values.values())
    total_checks = sum(c.get("total_checks", 0) for c in candidates)
    stable = [c for c in candidates if c.get("selection_label") == "IK_STABLE_CANDIDATE"]
    five_of_five = sum(1 for c in candidates if c.get("total_pass") == 5 and c.get("total_checks") == 5)
    four_of_five = sum(1 for c in candidates if c.get("total_pass") == 4 and c.get("total_checks") == 5)
    initial_full_pass = sum(1 for c in candidates if c.get("initial_checks") and all(row["result"] == "PASS" for row in c["initial_checks"]))
    unstable_rejected = sum(1 for c in candidates if c.get("selection_label") == "UNSTABLE_IK_CANDIDATE")
    suffix = "progressive_orientation" if progressive_meta else "orientation_local"
    return {
        "metadata": {
            "run_id": run_id_text(suffix),
            "timestamp": now_text(),
            "git_commit": get_git_commit(),
            "argv": [sys.executable, *sys.argv],
            "mock": args.mock,
            "scan_status": scan_status,
            "elapsed_s": elapsed_s,
            "elapsed_text": human_duration(elapsed_s),
            "progressive": bool(progressive_meta),
        },
        "status_labels": {
            "CENTER_POSE": {
                "value": args.center,
                "status": "RABO_EXECUTION_VERIFIED_BUT_USER_OBSERVED_PALM_REVERSED",
            },
            "ROLL_PI_BRANCH": {
                "status": "TESTED_ROLL_PI_CANDIDATES_FAILED",
                "tested_candidates": [
                    [0.385, 0.038, -0.33, 3.1416, 0.8, 3.1416],
                    [0.385, 0.038, -0.33, 3.1416, -0.8, 0.0],
                ],
                "reason": "out_of_workspace",
                "scope_note": "Does not prove every roll≈pi pose is unreachable.",
            },
            "POSE_CHECK_RESULT": "VERIFIED_BY_CURRENT_RUNTIME" if args.mock is None else "MOCK_VERIFIED_FOR_LOCAL_LOGIC_TEST_ONLY",
            "OUTPUT_LABEL": "IK_REACHABLE_ORIENTATION_CANDIDATE",
        },
        "command_inputs": vars(args),
        "search_range": {
            "x": [args.center[0], args.center[0]],
            "y": [args.center[1], args.center[1]],
            "z": [min(values["z"]), max(values["z"]), args.local_z_step],
            "roll": [min(values["roll"]), max(values["roll"]), args.local_roll_step],
            "pitch": [min(values["pitch"]), max(values["pitch"]), args.local_pitch_step],
            "yaw": {
                "values": values["yaw"],
                "note": "Yaw comparisons use wrapped angular distance.",
            },
        },
        "scan_values": values,
        "counts": {
            "total_candidates": total,
            "processed_candidates": processed,
            "grasp_pass_count": pass_count,
            "grasp_fail_count": processed - pass_count,
            "pass_rate": round(pass_count / processed, 6) if processed else 0.0,
            "representative_candidate_count": len(top),
            "total_pose_check_calls": progressive_meta.get("total_pose_check_calls") if progressive_meta else total,
            "initial_full_pass_count": initial_full_pass,
            "unstable_rejected_count": unstable_rejected,
            "ik_stable_candidate_count": len(stable),
            "five_of_five_candidate_count": five_of_five,
            "four_of_five_candidate_count": four_of_five,
        },
        "progressive": progressive_meta or {},
        "representative_orientation_candidates": [compact_orientation_candidate(c) for c in top],
        "dry_grasp_only_commands": [dry_grasp_only_command(c["pose"]) for c in top],
    }


def dry_grasp_only_command(pose: list[float]) -> str:
    values = " ".join(str(v) for v in pose)
    return f"python3 tools/test_left_grasp_v1.py dry-grasp-only --pose {values}"


def write_orientation_reports(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = summary["metadata"]["run_id"]
    json_path = REPORT_DIR / f"{run_id}.json"
    csv_path = REPORT_DIR / f"{run_id}.csv"
    md_path = REPORT_DIR / f"{run_id}.md"
    payload = {**summary, "candidates": candidates}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_orientation_csv(csv_path, candidates)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_orientation_markdown(summary))
    return json_path, csv_path, md_path


def write_orientation_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "pose",
        "z",
        "roll",
        "pitch",
        "yaw",
        "grasp_reachable",
        "grasp_reason",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw_wrapped",
        "orientation_distance_to_center",
        "ik_repeatability",
        "ik_repeatability_text",
        "total_pass",
        "total_checks",
        "robustness_score",
        "neighbor_robustness",
        "neighbor_count",
        "neighbor_grasp_pass_count",
        "distance_to_edge",
        "selection_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            pose = c["pose"]
            writer.writerow(
                {
                    "candidate_id": c["candidate_id"],
                    "pose": json.dumps(pose),
                    "z": pose[2],
                    "roll": pose[3],
                    "pitch": pose[4],
                    "yaw": pose[5],
                    "grasp_reachable": c["grasp_reachable"],
                    "grasp_reason": c["grasp_reason"],
                    "delta_z": c["delta_z"],
                    "delta_roll": c["delta_roll"],
                    "delta_pitch": c["delta_pitch"],
                    "delta_yaw_wrapped": c["delta_yaw_wrapped"],
                    "orientation_distance_to_center": c["orientation_distance_to_center"],
                    "ik_repeatability": c.get("ik_repeatability", 0.0),
                    "ik_repeatability_text": c.get("ik_repeatability_text", ""),
                    "total_pass": c.get("total_pass", 0),
                    "total_checks": c.get("total_checks", 0),
                    "robustness_score": c["robustness_score"],
                    "neighbor_robustness": c.get("neighbor_robustness", c["robustness_score"]),
                    "neighbor_count": c["neighbor_count"],
                    "neighbor_grasp_pass_count": c["neighbor_grasp_pass_count"],
                    "distance_to_edge": c["distance_to_edge"],
                    "selection_label": c["selection_label"],
                }
            )


def write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "z",
        "roll",
        "pitch",
        "yaw",
        "grasp_pose",
        "approach_pose",
        "lift_pose",
        "approach_reachable",
        "grasp_reachable",
        "lift_reachable",
        "full_chain",
        "robustness_score",
        "neighbor_count",
        "neighbor_full_chain_pass_count",
        "region_id",
        "distance_to_edge",
        "orientation_prior_distance",
        "approach_reason",
        "grasp_reason",
        "lift_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "candidate_id": c["candidate_id"],
                    "z": c["grasp_pose"][2],
                    "roll": c["grasp_pose"][3],
                    "pitch": c["grasp_pose"][4],
                    "yaw": c["grasp_pose"][5],
                    "grasp_pose": json.dumps(c["grasp_pose"]),
                    "approach_pose": json.dumps(c["approach_pose"]),
                    "lift_pose": json.dumps(c["lift_pose"]),
                    "approach_reachable": c["approach_reachable"],
                    "grasp_reachable": c["grasp_reachable"],
                    "lift_reachable": c["lift_reachable"],
                    "full_chain": c["full_chain"],
                    "robustness_score": c["robustness_score"],
                    "neighbor_count": c["neighbor_count"],
                    "neighbor_full_chain_pass_count": c["neighbor_full_chain_pass_count"],
                    "region_id": c["region_id"],
                    "distance_to_edge": c["distance_to_edge"],
                    "orientation_prior_distance": c["orientation_prior_distance"],
                    "approach_reason": c["approach_reason"],
                    "grasp_reason": c["grasp_reason"],
                    "lift_reason": c["lift_reason"],
                }
            )


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Left Pose Scan Report: {summary['metadata']['run_id']}",
        "",
        "## Metadata",
        f"- timestamp: `{summary['metadata']['timestamp']}`",
        f"- git_commit: `{summary['metadata']['git_commit']}`",
        f"- pose_check_status: `{summary['status_labels']['POSE_CHECK_RESULT']}`",
        "",
        "## Status Labels",
        "```json",
        json.dumps(summary["status_labels"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Counts",
        "```json",
        json.dumps(summary["counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Robust Regions",
    ]
    if not summary["regions"]:
        lines.append("- No FULL_CHAIN_PASS regions found.")
    for region in summary["regions"]:
        lines.extend(
            [
                "",
                f"### {region['region_id']}",
                f"- center: `{region['center_pose']}`",
                f"- candidate_count: `{region['candidate_count']}`",
                f"- mean_robustness: `{region['mean_robustness']}`",
                f"- z range: `{region['z_range']}`",
                f"- roll range: `{region['roll_range']}`",
                f"- pitch range: `{region['pitch_range']}`",
                f"- yaw range: `{region['yaw_range']}`",
            ]
        )
    lines.extend(["", "## TOP 10 ROBUST LEFT GRASP POSES"])
    if not summary["top_candidates"]:
        lines.append("- No FULL_CHAIN_PASS candidates found.")
    for c in summary["top_candidates"]:
        lines.extend(
            [
                "",
                f"### #{c['rank']}",
                "",
                "POSE:",
                f"`{c['grasp_pose']}`",
                "",
                "FULL_CHAIN:",
                f"`{c['full_chain']}`",
                "",
                "ROBUSTNESS:",
                f"`{c['robustness_score']}`",
                "",
                "REGION:",
                f"`{c['region_id']}`",
                "",
                "DISTANCE_TO_EDGE:",
                f"`{c['distance_to_edge']}`",
                "",
                "APPROACH / GRASP / LIFT:",
                f"`{c['approach_reason']}` / `{c['grasp_reason']}` / `{c['lift_reason']}`",
                "",
                "------------------------------------------------",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended Left Grasp Pose",
            "```json",
            json.dumps(summary["recommended_left_grasp_pose"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Notes",
            "- This scan only verifies pose_check reachability. It does not verify real grasp success.",
            "- Use the recommended or Top Candidate pose with `tools/test_left_grasp_v1.py dry --pose ...` before place/real tests.",
            "",
        ]
    )
    return "\n".join(lines)


def render_orientation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Left Local Orientation Scan Report: {summary['metadata']['run_id']}",
        "",
        "## Metadata",
        f"- timestamp: `{summary['metadata']['timestamp']}`",
        f"- git_commit: `{summary['metadata']['git_commit']}`",
        f"- scan_status: `{summary['metadata']['scan_status']}`",
        f"- pose_check_status: `{summary['status_labels']['POSE_CHECK_RESULT']}`",
        "",
        "## CENTER POSE",
        "```json",
        json.dumps(summary["status_labels"]["CENTER_POSE"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## SEARCH RANGE",
        "```json",
        json.dumps(summary["search_range"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## CURRENT EXPERIMENT EXCLUSIONS",
        "```json",
        json.dumps(summary["status_labels"]["ROLL_PI_BRANCH"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## PASS / FAIL",
        f"- SCAN LEVEL USED: `{summary.get('progressive', {}).get('selected_level', 'NOT_APPLICABLE')}`",
        f"- STOP REASON: `{summary.get('progressive', {}).get('stop_reason', 'NOT_APPLICABLE')}`",
        f"- TOTAL CANDIDATES: `{summary['counts']['total_candidates']}`",
        f"- PROCESSED CANDIDATES: `{summary['counts']['processed_candidates']}`",
        f"- TOTAL POSE_CHECK CALLS: `{summary['counts']['total_pose_check_calls']}`",
        f"- GRASP PASS COUNT: `{summary['counts']['grasp_pass_count']}`",
        f"- GRASP FAIL COUNT: `{summary['counts']['grasp_fail_count']}`",
        f"- PASS RATE: `{summary['counts']['pass_rate']}`",
        f"- 2/2 INITIAL PASS: `{summary['counts']['initial_full_pass_count']}`",
        f"- UNSTABLE REJECTED: `{summary['counts']['unstable_rejected_count']}`",
        f"- IK STABLE CANDIDATES: `{summary['counts']['ik_stable_candidate_count']}`",
        f"- 5/5 CANDIDATES: `{summary['counts']['five_of_five_candidate_count']}`",
        f"- 4/5 CANDIDATES: `{summary['counts']['four_of_five_candidate_count']}`",
        "",
        "## REPRESENTATIVE ORIENTATION CANDIDATES",
    ]
    if not summary["representative_orientation_candidates"]:
        lines.append("- No GRASP pose_check PASS candidates found.")
    for candidate in summary["representative_orientation_candidates"]:
        lines.extend(
            [
                "",
                f"### #{candidate['rank']}",
                "",
                "POSE:",
                f"`{candidate['pose']}`",
                "",
                "GRASP:",
                f"`{candidate['grasp_reachable']}`",
                "",
                "DELTA FROM CENTER:",
                f"`z={candidate['delta_z']}, roll={candidate['delta_roll']}, pitch={candidate['delta_pitch']}, yaw_wrapped={candidate['delta_yaw_wrapped']}`",
                "",
                "ORIENTATION_DISTANCE_TO_CENTER:",
                f"`{candidate['orientation_distance_to_center']}`",
                "",
                "IK_REPEATABILITY:",
                f"`{candidate.get('ik_repeatability_text', 'NOT_APPLICABLE')}`",
                "",
                "NEIGHBOR_ROBUSTNESS:",
                f"`{candidate.get('neighbor_robustness', candidate.get('robustness_score'))}`",
                "",
                "DISTANCE_TO_EDGE:",
                f"`{candidate['distance_to_edge']}`",
                "",
                "LABEL:",
                f"`{candidate['selection_label']}`",
                "",
                "--------------------------------",
            ]
        )
    lines.extend(["", "## dry-grasp-only Commands"])
    for index, command in enumerate(summary["dry_grasp_only_commands"], start=1):
        lines.extend(
            [
                "",
                f"### CANDIDATE_{index:02d}_COMMAND",
                "```bash",
                command,
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- This scan only checks GRASP pose_check. It does not check Approach or Lift.",
            "- These are IK_REACHABLE_ORIENTATION_CANDIDATE poses, not confirmed correct grasp poses.",
            "- Correct palm/finger direction must be verified by dry-grasp-only and USER_OBSERVED notes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Left-hand 4D grasp pose scan using pose_check only.")
    parser.add_argument("--orientation-local", action="store_true", help="Scan local z/roll/pitch/yaw around --center with GRASP pose_check only.")
    parser.add_argument("--progressive", action="store_true", help="Use small progressive local scan with repeated IK pose_check verification.")
    parser.add_argument("--xy", type=float, nargs=2, default=list(DEFAULT_XY), metavar=("X", "Y"))
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=-0.31)
    parser.add_argument("--z-step", type=float, default=0.01)
    parser.add_argument("--roll-min", type=float, default=-0.4)
    parser.add_argument("--roll-max", type=float, default=0.4)
    parser.add_argument("--roll-step", type=float, default=0.2)
    parser.add_argument("--yaw-min", type=float, default=-0.8)
    parser.add_argument("--yaw-max", type=float, default=0.8)
    parser.add_argument("--yaw-step", type=float, default=0.2)
    parser.add_argument("--pitch-step", type=float, default=0.2)
    parser.add_argument("--pitch-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pitch-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pitch-values", type=parse_csv_floats, help="Comma-separated explicit pitch values, e.g. 0.6,0.8,1.0,-0.8")
    parser.add_argument("--approach-dz", type=float, default=0.05)
    parser.add_argument("--lift-dz", type=float, default=0.05)
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--center", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW"))
    parser.add_argument("--local-z-radius", type=float, default=0.01)
    parser.add_argument("--local-z-step", type=float, default=0.01)
    parser.add_argument("--local-roll-radius", type=float, default=0.8)
    parser.add_argument("--local-roll-step", type=float, default=0.2)
    parser.add_argument("--local-pitch-min", type=float, default=0.2)
    parser.add_argument("--local-pitch-max", type=float, default=1.4)
    parser.add_argument("--local-pitch-step", type=float, default=0.2)
    parser.add_argument("--local-yaw-step", type=float, default=0.2)
    parser.add_argument("--fine-z-radius", type=float, default=0.015)
    parser.add_argument("--fine-z-step", type=float, default=0.005)
    parser.add_argument("--fine-angle-radius", type=float, default=0.20)
    parser.add_argument("--fine-angle-step", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=6, help="Representative candidate count for --orientation-local.")
    parser.add_argument("--initial-checks", type=int, default=2)
    parser.add_argument("--repeat-checks", type=int, default=3)
    parser.add_argument("--stable-threshold", type=float, default=0.8)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--min-stable-candidates", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=25, help="Print scan progress every N candidates; 0 disables progress output.")
    parser.add_argument("--progress-interval", type=int, default=25, help="Checkpoint interval for --orientation-local progress.")
    parser.add_argument("--mock", choices=("islands", "pitch-sign", "all-pass", "orientation-local", "progressive"), help="Local logic test mode; does not initialize Rabo SDK.")
    parser.add_argument("--mock-interrupt-after", type=int, help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.approach_dz <= 0.0 or args.lift_dz <= 0.0:
        raise SystemExit("--approach-dz and --lift-dz must be positive")
    if args.orientation_local:
        if args.center is None:
            raise SystemExit("--orientation-local requires --center X Y Z R P YAW")
        if args.top_k <= 0:
            raise SystemExit("--top-k must be positive")
        if args.progress_interval < 0:
            raise SystemExit("--progress-interval must be >= 0")
        if args.progressive:
            if args.initial_checks <= 0 or args.repeat_checks < 0:
                raise SystemExit("--initial-checks must be positive and --repeat-checks must be >= 0")
            if args.max_level not in (1, 2, 3):
                raise SystemExit("--max-level must be 1, 2, or 3")
            if not 0.0 <= args.stable_threshold <= 1.0:
                raise SystemExit("--stable-threshold must be between 0 and 1")
    if args.fine and args.center is not None:
        args.xy = [args.center[0], args.center[1]]
    if not args.fine and args.pitch_values is None and not args.pitch_positive and not args.pitch_negative:
        raise SystemExit("at least one pitch region must be enabled")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.orientation_local:
        if args.progressive:
            candidates, progressive_meta, scan_status, elapsed_s = scan_progressive_orientation(args)
            # Recompute neighbor robustness over the accumulated scan points for final reporting.
            if progressive_meta.get("selected_level"):
                values = progressive_level_values([round(v, 6) for v in args.center], progressive_meta["selected_level"])
            else:
                values = progressive_level_values([round(v, 6) for v in args.center], 1)
            top = select_representative_orientations(candidates, [round(v, 6) for v in args.center], args.top_k)
            summary = build_orientation_summary(args, values, candidates, top, scan_status, elapsed_s, progressive_meta)
            json_path, csv_path, md_path = write_orientation_reports(summary, candidates)
            print("=" * 60)
            print("SCAN COMPLETE" if scan_status == "COMPLETE" else "SCAN INTERRUPTED")
            print("=" * 60)
            print(f"SCAN LEVEL USED:\n{summary['progressive'].get('selected_level')}")
            print(f"STOP REASON:\n{summary['progressive'].get('stop_reason')}")
            print(f"TOTAL UNIQUE CANDIDATES:\n{summary['counts']['processed_candidates']}")
            print(f"TOTAL POSE_CHECK CALLS:\n{summary['counts']['total_pose_check_calls']}")
            print(f"2/2 INITIAL PASS:\n{summary['counts']['initial_full_pass_count']}")
            print(f"UNSTABLE REJECTED:\n{summary['counts']['unstable_rejected_count']}")
            print(f"5/5 CANDIDATES:\n{summary['counts']['five_of_five_candidate_count']}")
            print(f"4/5 CANDIDATES:\n{summary['counts']['four_of_five_candidate_count']}")
            print(f"REPRESENTATIVE CANDIDATES:\n{summary['counts']['representative_candidate_count']}")
            print(f"REPORT:\n{md_path}")
            print(f"REPORT_JSON: {json_path}")
            print(f"REPORT_CSV: {csv_path}")
            print(f"REPORT_MD: {md_path}")
            return 0 if candidates else 1
        values = build_orientation_local_values(args)
        candidates, scan_status, elapsed_s = scan_orientation_local(args, values)
        analyze_orientation_local(candidates, values)
        top = select_representative_orientations(candidates, [round(v, 6) for v in args.center], args.top_k)
        summary = build_orientation_summary(args, values, candidates, top, scan_status, elapsed_s)
        json_path, csv_path, md_path = write_orientation_reports(summary, candidates)
        print("=" * 60)
        print("SCAN COMPLETE" if scan_status == "COMPLETE" else "SCAN INTERRUPTED")
        print("=" * 60)
        print(f"TOTAL:\n{summary['counts']['total_candidates']}")
        print(f"PROCESSED:\n{summary['counts']['processed_candidates']}")
        print(f"GRASP PASS:\n{summary['counts']['grasp_pass_count']}")
        print(f"GRASP FAIL:\n{summary['counts']['grasp_fail_count']}")
        print(f"PASS RATE:\n{summary['counts']['pass_rate'] * 100:.2f}%")
        print(f"ELAPSED:\n{summary['metadata']['elapsed_text']}")
        print(f"REPRESENTATIVE CANDIDATES:\n{summary['counts']['representative_candidate_count']}")
        print(f"REPORT:\n{md_path}")
        print(f"REPORT_JSON: {json_path}")
        print(f"REPORT_CSV: {csv_path}")
        print(f"REPORT_MD: {md_path}")
        return 0 if candidates else 1
    values = build_scan_values(args)
    expected = math.prod(len(v) for v in values.values())
    print(f"SCAN_CANDIDATES: {expected}")
    print("ALLOWED_RUNTIME_CALLS: pose_check only")
    candidates = scan_candidates(args, values)
    regions = analyze_grid(candidates, values)
    top = rank_candidates(candidates, limit=args.top_n)
    summary = build_summary(args, values, candidates, regions, top)
    json_path, csv_path, md_path = write_reports(summary, candidates)
    print(f"FULL_CHAIN_PASS: {summary['counts']['full_chain_pass_count']}")
    print(f"REGIONS: {summary['counts']['region_count']}")
    if top:
        print(f"RECOMMENDED_LEFT_GRASP_POSE: {top[0]['grasp_pose']}")
    print(f"REPORT_JSON: {json_path}")
    print(f"REPORT_CSV: {csv_path}")
    print(f"REPORT_MD: {md_path}")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
