#!/usr/bin/env python3
"""Collect one raw Nut C episode with native ros2 bag recording."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import signal
import shutil
import subprocess
import sys
import tarfile
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_c_only_serial_expert import STATE_TOPICS, TOP_RGB_TOPIC  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "act_rosbag_raw"
DEFAULT_ARCHIVE_ROOT = Path("/workspace/agent_system")
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


class AbortKeyMonitor:
    """Non-blocking single-key reader that restores the terminal on exit."""

    def __init__(self, stream: TextIO = sys.stdin) -> None:
        self.stream = stream
        self.fd: int | None = None
        self.previous_terminal: list[Any] | None = None

    def __enter__(self) -> "AbortKeyMonitor":
        try:
            self.fd = self.stream.fileno()
            if self.stream.isatty():
                self.previous_terminal = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
        except (AttributeError, OSError, termios.error):
            self.fd = None
        return self

    def __exit__(self, *_: Any) -> None:
        if self.fd is not None and self.previous_terminal is not None:
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_terminal)

    def abort_requested(self, timeout_s: float) -> bool:
        if self.fd is None:
            time.sleep(timeout_s)
            return False
        ready, _, _ = select.select([self.fd], [], [], timeout_s)
        if not ready:
            return False
        data = os.read(self.fd, 1)
        if not data:
            self.fd = None
            time.sleep(timeout_s)
            return False
        return data.decode(errors="ignore").lower() == "q"


def wait_for_expert(
    expert: subprocess.Popen[Any],
    monitor_factory: Callable[[], AbortKeyMonitor] = AbortKeyMonitor,
) -> bool:
    """Return True only when the user presses q before Expert exits."""
    with monitor_factory() as monitor:
        while expert.poll() is None:
            if monitor.abort_requested(0.1):
                return True
    return False


def stop_process_group(process: subprocess.Popen[Any] | None, timeout_s: float) -> int | None:
    """Ask a child process group to exit through SIGINT, then escalate if stuck."""
    if process is None or process.poll() is not None:
        return None if process is None else process.returncode
    for sig, wait_s in ((signal.SIGINT, timeout_s), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            return process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


def prompt_keep(input_func: Callable[[str], str] | None = None) -> bool:
    reader = input if input_func is None else input_func
    while True:
        answer = reader("Keep this episode? [yes/no]: ").strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Please enter yes/y or no/n.", flush=True)


def discard_episode(episode_dir: Path) -> None:
    if not episode_dir.name.startswith("episode_"):
        raise ValueError(f"refusing to delete unexpected episode path: {episode_dir}")
    if episode_dir.exists():
        shutil.rmtree(episode_dir)


@dataclass(frozen=True)
class ConversionOutcome:
    accepted: bool | None
    error: str | None
    returncode: int | None


def convert_episode(episode_dir: Path) -> ConversionOutcome:
    print("Converting...", flush=True)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "tools" / "convert_act_c_rosbag_episode.py"),
        "--episode",
        str(episode_dir),
    ]
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except BaseException as exc:
        return ConversionOutcome(None, repr(exc), None)
    quality_path = episode_dir / "act_5hz" / "quality_report.json"
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        accepted = bool(quality["accepted_for_training"])
    except BaseException as exc:
        return ConversionOutcome(None, f"quality report unavailable: {exc!r}", completed.returncode)
    if completed.returncode not in (0, 1):
        return ConversionOutcome(accepted, f"converter exited with {completed.returncode}", completed.returncode)
    return ConversionOutcome(accepted, None, completed.returncode)


def create_episode_archive(episode_dir: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = (archive_root / f"{episode_dir.name}.tar.gz").resolve()
    archive_path.unlink(missing_ok=True)
    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            for child in sorted(episode_dir.iterdir(), key=lambda path: path.name):
                archive.add(child, arcname=child.name)
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


@dataclass(frozen=True)
class RetentionOutcome:
    kept: bool
    accepted: bool | None
    archive_path: Path | None
    conversion_error: str | None
    archive_error: str | None


def handle_retention(
    episode_dir: Path,
    expert_status: str,
    keep: bool,
    archive_root: Path,
    *,
    converter: Callable[[Path], ConversionOutcome] = convert_episode,
    archiver: Callable[[Path, Path], Path] = create_episode_archive,
) -> RetentionOutcome:
    if not keep:
        discard_episode(episode_dir)
        print("Episode rejected by user.", flush=True)
        print("Data discarded.", flush=True)
        return RetentionOutcome(False, None, None, None, None)

    conversion = converter(episode_dir)
    if conversion.error is not None:
        print(f"Converter failure: {conversion.error}", flush=True)
    archive_path = None
    archive_error = None
    print("Creating archive...", flush=True)
    try:
        archive_path = archiver(episode_dir, archive_root)
    except BaseException as exc:
        archive_error = repr(exc)
        print(f"Archive failure: {archive_error}", flush=True)

    accepted_text = str(conversion.accepted is True).lower()
    print("\nSAVED EPISODE:", flush=True)
    print(str(episode_dir), flush=True)
    print("\nARCHIVE:", flush=True)
    print(str(archive_path) if archive_path is not None else "NOT CREATED", flush=True)
    print(f"\nExpert status: {expert_status}", flush=True)
    print(f"Accepted for training: {accepted_text}", flush=True)
    return RetentionOutcome(True, conversion.accepted, archive_path, conversion.error, archive_error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record TOP RGB only and run the serial Nut C Expert with passive arm JSONL logging.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--recorder-start-wait-s", type=float, default=2.0)
    parser.add_argument("--post-expert-record-s", type=float, default=1.5)
    parser.add_argument("--rosbag-stop-timeout-s", type=float, default=15.0)
    parser.add_argument("--expert-stop-timeout-s", type=float, default=10.0)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT, help=argparse.SUPPRESS)
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
    expert: subprocess.Popen[Any] | None = None
    expert_returncode: int | None = None
    recorder_returncode: int | None = None
    failure_reason: str | None = None
    abort_requested = False
    print("Starting episode...", flush=True)
    print("Press q to abort and discard.", flush=True)
    try:
        recorder = subprocess.Popen(rosbag_command, cwd=PROJECT_ROOT, start_new_session=True)
        time.sleep(args.recorder_start_wait_s)
        if recorder.poll() is not None:
            raise RuntimeError(f"ros2 bag record exited early with {recorder.returncode}")
        with expert_log_path.open("w", encoding="utf-8") as expert_log:
            print("\n[Expert running...]", flush=True)
            expert = subprocess.Popen(
                expert_command,
                cwd=PROJECT_ROOT,
                stdout=expert_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            abort_requested = wait_for_expert(expert)
            if abort_requested:
                expert_returncode = stop_process_group(expert, args.expert_stop_timeout_s)
            else:
                expert_returncode = expert.wait()
        if not abort_requested and args.post_expert_record_s > 0:
            time.sleep(args.post_expert_record_s)
    except BaseException as exc:
        failure_reason = repr(exc)
        if expert is not None and expert.poll() is None:
            expert_returncode = stop_process_group(expert, args.expert_stop_timeout_s)
    finally:
        recorder_returncode = stop_process_group(recorder, args.rosbag_stop_timeout_s)

    if abort_requested:
        discard_episode(episode_dir)
        print("Episode aborted by user.", flush=True)
        print("Data discarded.", flush=True)
        return 0

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
    print(f"Expert status: {expert_status}", flush=True)
    print("\nEpisode finished.", flush=True)
    try:
        keep = prompt_keep()
    except (EOFError, KeyboardInterrupt):
        print("No keep/discard decision received; raw episode retained without conversion or archive.", flush=True)
        return 1
    outcome = handle_retention(
        episode_dir,
        expert_status,
        keep,
        args.archive_root.resolve(),
    )
    if not outcome.kept:
        return 0
    if outcome.conversion_error is not None or outcome.archive_error is not None:
        return 1
    return 0 if expert_status == "PASS" and recorder_returncode in (0, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
