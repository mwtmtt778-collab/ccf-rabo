#!/usr/bin/env python3
"""Mock-only tests for interactive Nut C episode retention."""

from __future__ import annotations

import contextlib
import io
import signal
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import collect_act_c_rosbag as collector


def make_episode(root: Path, name: str = "episode_test") -> Path:
    episode = root / name
    (episode / "bag").mkdir(parents=True)
    (episode / "bag" / "bag_0.mcap").write_bytes(b"mcap")
    (episode / "arm_state.jsonl").write_text("{}\n", encoding="utf-8")
    (episode / "action_events.jsonl").write_text("{}\n", encoding="utf-8")
    (episode / "episode_meta.json").write_text("{}\n", encoding="utf-8")
    (episode / "expert.log").write_text("PASS\n", encoding="utf-8")
    return episode


def successful_converter(accepted: bool = True):
    calls: list[Path] = []

    def convert(episode: Path) -> collector.ConversionOutcome:
        calls.append(episode)
        output = episode / "act_5hz"
        output.mkdir()
        (output / "quality_report.json").write_text(
            '{"accepted_for_training": ' + str(accepted).lower() + "}\n",
            encoding="utf-8",
        )
        return collector.ConversionOutcome(accepted, None, 0 if accepted else 1)

    return calls, convert


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -signal.SIGINT
        return self.returncode


class ImmediateQMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def abort_requested(self, _timeout_s: float) -> bool:
        return True


class InteractiveCollectorTest(unittest.TestCase):
    def test_normal_pass_yes_retains_converts_and_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = make_episode(root)
            archive_root = root / "archives"
            calls, converter = successful_converter(True)
            result = collector.handle_retention(
                episode, "PASS", True, archive_root, converter=converter
            )
            self.assertEqual(calls, [episode])
            self.assertTrue(episode.is_dir())
            self.assertTrue(result.archive_path and result.archive_path.is_file())
            with tarfile.open(result.archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("bag", names)
            self.assertIn("arm_state.jsonl", names)
            self.assertIn("action_events.jsonl", names)
            self.assertIn("episode_meta.json", names)
            self.assertIn("expert.log", names)
            self.assertIn("act_5hz", names)

    def test_normal_pass_no_discards_without_converter_or_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = make_episode(root)
            converter = mock.Mock()
            archiver = mock.Mock()
            result = collector.handle_retention(
                episode, "PASS", False, root / "archives", converter=converter, archiver=archiver
            )
            self.assertFalse(result.kept)
            self.assertFalse(episode.exists())
            converter.assert_not_called()
            archiver.assert_not_called()
            self.assertFalse((root / "archives").exists())

    def test_fault_yes_retains_raw_and_creates_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = make_episode(root)
            calls, converter = successful_converter(False)
            result = collector.handle_retention(
                episode, "FAULT", True, root / "archives", converter=converter
            )
            self.assertEqual(calls, [episode])
            self.assertFalse(result.accepted)
            self.assertTrue(episode.exists())
            self.assertTrue(result.archive_path and result.archive_path.exists())

    def test_q_abort_interrupts_expert_and_rosbag_then_discards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expert = FakeProcess(101)
            recorder = FakeProcess(202)
            episode = make_episode(root)
            archive_path = root / "archives" / f"{episode.name}.tar.gz"
            self.assertTrue(
                collector.wait_for_expert(expert, monitor_factory=ImmediateQMonitor)
            )
            signals = []
            with mock.patch.object(collector.os, "killpg", side_effect=lambda pid, sig: signals.append((pid, sig))):
                collector.stop_process_group(expert, 1.0)
                collector.stop_process_group(recorder, 1.0)
            collector.discard_episode(episode)
            self.assertIn((101, signal.SIGINT), signals)
            self.assertIn((202, signal.SIGINT), signals)
            self.assertFalse(episode.exists())
            self.assertFalse(archive_path.exists())

    def test_converter_failure_yes_retains_and_still_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = make_episode(root)

            def failed_converter(_episode: Path) -> collector.ConversionOutcome:
                return collector.ConversionOutcome(None, "mock converter failure", 2)

            result = collector.handle_retention(
                episode, "PASS", True, root / "archives", converter=failed_converter
            )
            self.assertTrue(episode.exists())
            self.assertEqual(result.conversion_error, "mock converter failure")
            self.assertTrue(result.archive_path and result.archive_path.exists())

    def test_tar_failure_retains_raw_and_prints_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = make_episode(root)
            _, converter = successful_converter(True)

            def failed_archiver(_episode: Path, _root: Path) -> Path:
                raise OSError("mock tar failure")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = collector.handle_retention(
                    episode,
                    "PASS",
                    True,
                    root / "archives",
                    converter=converter,
                    archiver=failed_archiver,
                )
            self.assertTrue(episode.exists())
            self.assertIsNone(result.archive_path)
            self.assertIn("Archive failure", output.getvalue())
            self.assertIn("mock tar failure", output.getvalue())

    def test_yes_no_inputs_are_case_insensitive(self) -> None:
        for value in ("yes", "y", "YES", "Y", "Yes"):
            with self.subTest(value=value):
                self.assertTrue(collector.prompt_keep(lambda _prompt, value=value: value))
        for value in ("no", "n", "NO", "N", "No"):
            with self.subTest(value=value):
                self.assertFalse(collector.prompt_keep(lambda _prompt, value=value: value))

    def test_stop_process_group_escalates_only_after_timeout(self) -> None:
        process = FakeProcess(303)
        waits = [subprocess.TimeoutExpired("mock", 1.0), -signal.SIGTERM]

        def wait(timeout=None):
            result = waits.pop(0)
            if isinstance(result, BaseException):
                raise result
            process.returncode = result
            return result

        process.wait = wait
        signals = []
        with mock.patch.object(collector.os, "killpg", side_effect=lambda pid, sig: signals.append((pid, sig))):
            result = collector.stop_process_group(process, 1.0)
        self.assertEqual(result, -signal.SIGTERM)
        self.assertEqual(signals, [(303, signal.SIGINT), (303, signal.SIGTERM)])


if __name__ == "__main__":
    unittest.main()
