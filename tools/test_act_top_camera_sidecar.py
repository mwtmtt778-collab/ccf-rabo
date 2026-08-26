#!/usr/bin/env python3
"""No-ROS tests for the top-camera sidecar protocol and causal alignment."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.collect_act_episode import causal_camera_alignment_ns  # noqa: E402


class TopCameraSidecarTests(unittest.TestCase):
    def test_absolute_monotonic_causal_alignment(self) -> None:
        indices, ages = causal_camera_alignment_ns(
            __import__("numpy").array([1_050_000_000, 1_200_000_000, 1_300_000_000], dtype="int64"),
            __import__("numpy").array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype="int64"),
        )
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(ages.tolist(), [0.05, 0.1, 0.05])

    def test_subprocess_ready_stop_flushes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            episode_dir = Path(directory) / "episode"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "tools/record_act_top_camera_sidecar.py"),
                    "--synthetic-run",
                    "--episode-dir",
                    str(episode_dir),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdin is not None
            process.stdin.write("STOP\n")
            process.stdin.flush()
            output, _ = process.communicate(timeout=5.0)
            self.assertEqual(process.returncode, 0)
            self.assertIn("READY", output)
            self.assertIn("SUMMARY", output)
            rows = [
                json.loads(line)
                for line in (episode_dir / "camera_timestamps.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2])
            self.assertTrue(all("receive_monotonic_ns" in row for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
