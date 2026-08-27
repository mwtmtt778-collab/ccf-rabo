#!/usr/bin/env python3
"""Mock tests for the read-only Rabo instance probe."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from tools.probe_rabo_instance import IMAGE_TYPES, UNRESOLVED, analyze_snapshot, print_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT = "sensor_msgs/msg/JointState"
IMAGE = next(iter(IMAGE_TYPES))


def entity_topics(entity: str, count: int) -> list[tuple[str, list[str]]]:
    return [(f"/gs_new/{entity}_tp_ps_joint{i}", [JOINT]) for i in range(count)]


class ProbeTests(unittest.TestCase):
    def test_single_a7_candidate_is_found_without_guessing_side(self) -> None:
        topics = entity_topics("rarmonly", 7)
        report = analyze_snapshot({"topics": topics})
        self.assertEqual(len(report["devices"]["arm_candidates"]), 1)
        self.assertEqual(report["devices"]["arm_candidates"][0]["generated_id"], "rarmonly")
        self.assertEqual(report["devices"]["left_arm"]["status"], UNRESOLVED)
        self.assertEqual(report["devices"]["right_arm"]["status"], UNRESOLVED)

    def test_full_device_id_can_be_read_from_related_graph_name(self) -> None:
        full_id = "r" + "a" * 32
        topics = entity_topics("raaaaaaa", 7)
        nodes = [(f"left_arm_{full_id}", "/")]
        report = analyze_snapshot({"topics": topics, "nodes": nodes})
        self.assertEqual(report["devices"]["left_arm"]["device_id"], full_id)
        self.assertEqual(report["devices"]["left_arm"]["device_id_source"], "ROS_GRAPH_NAME")

    def test_single_devices_are_discovered_with_explicit_graph_roles(self) -> None:
        left_arm = "r" + "1" * 32
        right_arm = "r" + "2" * 32
        left_hand = "r" + "3" * 32
        right_hand = "r" + "4" * 32
        topics = (
            entity_topics(left_arm, 7)
            + entity_topics(right_arm, 7)
            + entity_topics(left_hand, 11)
            + entity_topics(right_hand, 11)
            + [("/gs_new/topcam_tp_cam_rgb", [IMAGE])]
        )
        nodes = [
            (f"left_arm_{left_arm}", "/"),
            (f"right_arm_{right_arm}", "/"),
            (f"left_hand_{left_hand}", "/"),
            (f"right_hand_{right_hand}", "/"),
        ]
        report = analyze_snapshot({"topics": topics, "nodes": nodes})
        self.assertEqual(report["devices"]["left_arm"]["device_id"], left_arm)
        self.assertEqual(report["devices"]["right_arm"]["device_id"], right_arm)
        self.assertEqual(report["devices"]["left_hand"]["device_id"], left_hand)
        self.assertEqual(report["devices"]["right_hand"]["device_id"], right_hand)

    def test_multiple_candidates_are_not_guessed(self) -> None:
        topics = entity_topics("raaaaaaa", 7) + entity_topics("rbbbbbbb", 7)
        report = analyze_snapshot({"topics": topics})
        self.assertEqual(report["devices"]["left_arm"]["status"], UNRESOLVED)
        self.assertEqual(report["devices"]["right_arm"]["status"], UNRESOLVED)
        self.assertEqual(len(report["devices"]["arm_candidates"]), 2)

    def test_missing_devices_are_unresolved(self) -> None:
        report = analyze_snapshot({"topics": []})
        for key in ("left_arm", "right_arm", "left_hand", "right_hand"):
            self.assertEqual(report["migration"][key]["new"], UNRESOLVED)
            self.assertIn(key, report["unresolved"])

    def test_multiple_cameras_are_all_preserved_and_top_is_unresolved(self) -> None:
        topics = entity_topics("rarm0001", 7) + [
            ("/gs_new/rarm0001_tp_cam_wrist", [IMAGE]),
            ("/gs_new/camera_a_tp_cam_rgb", [IMAGE]),
            ("/gs_new/camera_b_tp_cam_rgb", [IMAGE]),
        ]
        report = analyze_snapshot({"topics": topics})
        self.assertEqual(len(report["cameras"]["all"]), 3)
        self.assertEqual(report["cameras"]["top"]["status"], UNRESOLVED)
        self.assertEqual(len(report["cameras"]["top"]["candidates"]), 2)
        print_report(report)

    def test_report_has_required_json_sections(self) -> None:
        report = analyze_snapshot({"topics": []})
        encoded = json.loads(json.dumps(report))
        for key in ("scene", "devices", "cameras", "state_topics", "migration", "unresolved"):
            self.assertIn(key, encoded)

    def test_probe_source_contains_no_motion_sdk_calls(self) -> None:
        path = PROJECT_ROOT / "tools" / "probe_rabo_instance.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = {
            "move_joints", "move_to", "home", "stop", "grasp_force",
            "set_joint_positions", "clench", "set_pose",
        }
        called: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
        self.assertFalse(forbidden.intersection(called), forbidden.intersection(called))
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("LinkerArmA7(", source)
        self.assertNotIn("LinkerHandO6Left(", source)
        self.assertNotIn("LinkerHandO6Right(", source)


if __name__ == "__main__":
    unittest.main()
