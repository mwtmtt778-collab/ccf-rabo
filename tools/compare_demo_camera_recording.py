#!/usr/bin/env python3
"""Offline-style A/B camera recording around official or minimal execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_frames(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return [row for row in rows if row.get("receive_monotonic_ns") is not None]


def run_minimal() -> None:
    from tools.run_act_c_only_expert import run_act_c_only_expert
    from tools.test_right_release_stability import make_right_bundle, shutdown_bundle
    from tools.test_dual_closed_loop_v2 import make_left_bundle, shutdown_left_bundle
    from tools.test_three_nut_closed_loop_v2 import ExpertStateRunner
    from tools.motion_monitor import MotionMonitor
    import tempfile

    right = make_right_bundle(); left = make_left_bundle()
    try:
        runner = ExpertStateRunner(1, MotionMonitor(enabled=False), Path(tempfile.mktemp()), nonblocking_motion=True)
        run_act_c_only_expert(runner, right, left, settle_after_release_s=2.0, vision_target_radius_m=0.08)
    finally:
        shutdown_bundle(right); shutdown_left_bundle(left)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare official/minimal execution with identical camera recording.")
    parser.add_argument("--mode", choices=("official", "minimal"), required=True)
    parser.add_argument("--camera-topic", choices=("top", "right_wrist", "left_wrist"), default="top")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "reports" / "camera_ab" / f"{args.mode}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    sidecar = ROOT / "tools" / "record_act_top_camera_sidecar.py"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(sidecar), "--episode-dir", str(out), "--camera-topic", args.camera_topic],
        stdin=subprocess.PIPE, stdout=log_path.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True,
    )
    execution_start_ns = 0
    execution_end_ns = 0
    try:
        time.sleep(2.0)
        execution_start_ns = time.monotonic_ns()
        if args.mode == "official":
            from agents.arm_hand_demo import run
            run()
        else:
            run_minimal()
        execution_end_ns = time.monotonic_ns()
    except BaseException as exc:
        (out / "run_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        execution_end_ns = time.monotonic_ns()
    finally:
        time.sleep(5.0)
        if proc.stdin is not None:
            proc.stdin.write("STOP\n"); proc.stdin.flush()
        proc.wait(timeout=30.0)
    frames = read_frames(out / "camera_timestamps.jsonl")
    times = [int(row["receive_monotonic_ns"]) for row in frames]
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    during = [value for value in times if execution_start_ns <= value <= execution_end_ns]
    summary = {
        "mode": args.mode, "camera_topic": args.camera_topic, "frame_count": len(frames),
        "first_frame_ns": times[0] if times else None, "last_frame_ns": times[-1] if times else None,
        "effective_fps": (len(times) - 1) / ((times[-1] - times[0]) / 1e9) if len(times) > 1 and times[-1] > times[0] else None,
        "median_gap_ms": sorted(gaps)[len(gaps)//2] if gaps else None,
        "p95_gap_ms": sorted(gaps)[min(len(gaps)-1, int(len(gaps)*0.95))] if gaps else None,
        "max_gap_ms": max(gaps) if gaps else None,
        "execution_start_ns": execution_start_ns, "execution_end_ns": execution_end_ns,
        "frames_before_execution": sum(value < execution_start_ns for value in times),
        "frames_during_execution": len(during), "frames_after_execution": sum(value > execution_end_ns for value in times),
        "camera_during_execution_status": {"frame_count": len(during), "execution_duration_s": (execution_end_ns - execution_start_ns) / 1e9, "effective_fps": (len(during) - 1) / ((during[-1] - during[0]) / 1e9) if len(during) > 1 and during[-1] > during[0] else None, "longest_gap_ms": max(gaps) if gaps else None},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
