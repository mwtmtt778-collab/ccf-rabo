#!/usr/bin/env python3
"""Collect one raw Nut C episode with native ros2 bag recording."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_c_only_serial_expert import STATE_TOPICS, TOP_RGB_TOPIC  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "act_rosbag_raw"
ROSBAG_TOPICS = (TOP_RGB_TOPIC,)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_episode_dir(root: Path) -> tuple[str, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    episode_id = f"episode_{stamp}"
    path = root / episode_id
    path.mkdir(parents=True, exist_ok=False)
    return episode_id, path


def read_logger_status(path: Path) -> dict[str, Any] | None:
    try:
        rows = (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        statuses = [row.get("result") for row in rows if row.get("event") == "logger_status"]
        return statuses[-1] if statuses and isinstance(statuses[-1], dict) else None
    except BaseException:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record TOP RGB only and run the serial Nut C Expert with passive arm JSONL logging.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--recorder-start-wait-s", type=float, default=2.0)
    parser.add_argument("--post-expert-record-s", type=float, default=1.5)
    parser.add_argument("--rosbag-stop-timeout-s", type=float, default=15.0)
    parser.add_argument("--expert-arg", action="append", default=[], help="extra argument passed to the Expert; repeatable")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode_id, episode_dir = new_episode_dir(args.output_root.resolve())
    bag_path = episode_dir / "bag"
    arm_state_path = episode_dir / "arm_state.jsonl"
    action_events = episode_dir / "action_events.jsonl"
    expert_log_path = episode_dir / "expert.log"
    action_events.touch()
    arm_state_path.touch()
    expert_log_path.touch()
    start_wall_ns = time.time_ns()
    start_monotonic_ns = time.monotonic_ns()
    topics = list(ROSBAG_TOPICS)
    rosbag_command = ["ros2", "bag", "record", "-s", "mcap", "-o", str(bag_path), *topics]
    expert_command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "tools" / "run_c_only_serial_expert.py"),
        "--action-events",
        str(action_events),
        "--arm-state-log",
        str(arm_state_path),
        *args.expert_arg,
    ]
    recorder: subprocess.Popen[Any] | None = None
    expert_returncode: int | None = None
    recorder_returncode: int | None = None
    failure_reason: str | None = None
    try:
        recorder = subprocess.Popen(rosbag_command, cwd=PROJECT_ROOT, start_new_session=True)
        time.sleep(args.recorder_start_wait_s)
        if recorder.poll() is not None:
            raise RuntimeError(f"ros2 bag record exited early with {recorder.returncode}")
        with expert_log_path.open("w", encoding="utf-8") as expert_log:
            completed = subprocess.run(
                expert_command,
                cwd=PROJECT_ROOT,
                stdout=expert_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        expert_returncode = completed.returncode
        if args.post_expert_record_s > 0:
            time.sleep(args.post_expert_record_s)
    except BaseException as exc:
        failure_reason = repr(exc)
    finally:
        if recorder is not None and recorder.poll() is None:
            os.killpg(recorder.pid, signal.SIGINT)
            try:
                recorder_returncode = recorder.wait(timeout=args.rosbag_stop_timeout_s)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder_returncode = recorder.wait(timeout=5.0)
                failure_reason = failure_reason or "ros2 bag record did not close after SIGINT"
        elif recorder is not None:
            recorder_returncode = recorder.returncode

    end_wall_ns = time.time_ns()
    end_monotonic_ns = time.monotonic_ns()
    expert_status = "PASS" if expert_returncode == 0 and failure_reason is None else "FAULT"
    logger_status = read_logger_status(action_events)
    metadata = {
        "format": "rabo_act_c_rosbag_raw_v1",
        "episode_id": episode_id,
        "sequence": "C",
        "task_profile": "fixed_point_c_baseline",
        "visual_mode": "async_latest_hold",
        "requires_visual_generalization": False,
        "start_wall_ns": start_wall_ns,
        "end_wall_ns": end_wall_ns,
        "start_monotonic_ns": start_monotonic_ns,
        "end_monotonic_ns": end_monotonic_ns,
        "expert_status": expert_status,
        "expert_returncode": expert_returncode,
        "failure_reason": failure_reason,
        "bag_path": "bag",
        "bag_storage": "mcap",
        "rosbag_topic_count": len(topics),
        "rosbag_topics": topics,
        "rosbag_returncode": recorder_returncode,
        "camera_recording": "rosbag_top_only",
        "camera": {
            "name": "top",
            "topic": TOP_RGB_TOPIC,
            "expected_raw_rate_hz": 12.32,
            "expected_rate_basis": "measured TOP-only rosbag result: 739 messages / 59.973 s",
        },
        "raw_arm_topics_recorded": False,
        "raw_hand_topics_recorded": False,
        "arm_state_source": "passive_arm_state_jsonl",
        "arm_state_path": "arm_state.jsonl",
        "arm_state_joint_order": "SDK_J1_TO_J7",
        "arm_state_target_hz": 20.0,
        "arm_state_logger_status": logger_status,
        "passive_arm_state_topics": {name: STATE_TOPICS[name] for name in ("left_arm", "right_arm")},
        "passive_arm_state_message_type": "sensor_msgs/msg/JointState",
        "hand_state_source": "command_hold_last",
        "action_events_path": "action_events.jsonl",
        "expert_log_path": "expert.log",
        "recorder_start_wait_s": args.recorder_start_wait_s,
        "post_expert_record_s": args.post_expert_record_s,
        "commands": {"rosbag": rosbag_command, "expert": expert_command},
    }
    write_json(episode_dir / "episode_meta.json", metadata)
    print(str(episode_dir), flush=True)
    print(f"Expert status: {expert_status}", flush=True)
    return 0 if expert_status == "PASS" and recorder_returncode in (0, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
