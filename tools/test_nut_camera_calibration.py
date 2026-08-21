#!/usr/bin/env python3
"""Calibrate the fixed top-camera PointCloud2 frame against Rabo world coordinates.

This is one destructive simulation experiment: it moves Nut A and Nut C out of
the scene, then places Nut B at four known world positions.  No robot motion or
gripper command is issued.  The output is always written to
``reports/nut_camera_calibration/result.json`` (including failure details).

Run from the project root:

    python3 tools/test_nut_camera_calibration.py
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "nut_camera_calibration" / "result.json"

WORLD_ID = "ww4b1e6f8392584c89902a6783a9f53075"
SET_POSE_SERVICE = f"/world/{WORLD_ID}/set_pose"
# Fixed/top camera (r6ef2dc), not either wrist camera.
POINTCLOUD_TOPIC = "/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/points"

NUT_IDS = {
    "A": "thing_3eb64241-4166-4c4e-9144-edf733c885f1",
    "B": "thing_e937f633-faed-4b2e-b609-a7b80825a64a",
    "C": "thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7",
}
WORLD_POSITIONS = [
    [-0.30, 0.04, 0.28],
    [-0.40, 0.04, 0.28],
    [-0.30, 0.15, 0.28],
    [-0.40, 0.15, 0.28],
]
# SetEntityPose is the verified client for the supplied /world/.../set_pose
# endpoint.  The SDK currently exposes no delete-entity API, so A/C are placed
# far below and outside the table instead of being irreversibly deleted.
OFF_SCENE_POSE = [10.0, 10.0, -10.0, 0.0, 0.0, 0.0]
NUT_RPY = [0.0, 0.0, 0.5233]
SETTLE_SECONDS = 3.0
POINT_TIMEOUT_SECONDS = 5.0
WORLD_FRAME_TOLERANCE_M = 0.03
SET_POSE_RETRIES = 3
SET_POSE_RETRY_DELAY_SECONDS = 1.0
SET_POSE_COOLDOWN_SECONDS = 2.0


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return repr(value)


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def result_failed(value: Any) -> tuple[bool, str | None]:
    """Conservatively interpret the heterogeneous SetEntityPose return values."""
    if value is False or value is None:
        return True, repr(value)
    if isinstance(value, dict):
        if value.get("success") is False or value.get("ok") is False or value.get("error"):
            return True, str(value.get("error") or value)
    if isinstance(value, str) and any(word in value.lower() for word in ("error", "fail", "exception")):
        return True, value
    return False, None


def set_pose_with_retry(pose_setter: Any, entity_id: str, pose: list[float]) -> dict[str, Any]:
    """Retry transient Rabo set-pose service timeouts, preserving every result."""
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, SET_POSE_RETRIES + 1):
        result = pose_setter.set(entity_id, tuple(pose))
        failed, reason = result_failed(result)
        attempts.append({"attempt": attempt, "return": jsonable(result), "ok": not failed, "error": reason})
        if not failed:
            return {"ok": True, "attempts": attempts}
        if attempt < SET_POSE_RETRIES:
            time.sleep(SET_POSE_RETRY_DELAY_SECONDS)
    return {"ok": False, "attempts": attempts, "error": attempts[-1].get("error") if attempts else "no attempts"}


@dataclass
class CloudReceiver:
    node: Any
    subscription: Any
    owns_rclpy_context: bool
    latest: Any = None
    latest_received_at: float = 0.0
    frames: int = 0

    def spin_for(self, seconds: float) -> None:
        import rclpy

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))

    def next_after(self, marker: float, timeout: float) -> Any | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_for(0.1)
            if self.latest is not None and self.latest_received_at >= marker:
                return copy.deepcopy(self.latest)
        return None

    def close(self) -> None:
        import rclpy

        with contextlib.suppress(Exception):
            self.node.destroy_subscription(self.subscription)
        with contextlib.suppress(Exception):
            self.node.destroy_node()
        if self.owns_rclpy_context:
            with contextlib.suppress(Exception):
                rclpy.shutdown()


def start_receiver() -> CloudReceiver:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2

    os.environ.setdefault("ROS_LOG_DIR", str(PROJECT_ROOT / "logs" / "ros"))
    # In the platform's agent runtime rclpy may already be initialized by the
    # controller.  Reuse that context; calling init() again raises
    # "Context.init() must only be called once".
    owns_rclpy_context = not rclpy.ok()
    if owns_rclpy_context:
        rclpy.init(args=None)
    node = rclpy.create_node("nut_camera_calibration")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        # The fixed top camera publishes RELIABLE/VOLATILE.  Match it exactly:
        # the platform's DDS bridge does not reliably deliver this PointCloud2
        # stream to a BEST_EFFORT subscriber even though discovery succeeds.
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    receiver = CloudReceiver(node=node, subscription=None, owns_rclpy_context=owns_rclpy_context)

    def callback(msg: PointCloud2) -> None:
        receiver.latest = msg
        receiver.latest_received_at = time.monotonic()
        receiver.frames += 1

    receiver.subscription = node.create_subscription(PointCloud2, POINTCLOUD_TOPIC, callback, qos)
    return receiver


def cloud_xyz(msg: Any) -> tuple[Any, dict[str, Any]]:
    """Read x/y/z from PointCloud2 without retaining RGB or other fields."""
    import numpy as np
    from sensor_msgs_py import point_cloud2

    fields = [field.name for field in msg.fields]
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PointCloud2 is missing x/y/z fields; fields={fields}")
    if hasattr(point_cloud2, "read_points_numpy"):
        points = np.asarray(
            point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True),
            dtype=float,
        )
    else:
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if getattr(getattr(raw, "dtype", None), "names", None):
            points = np.column_stack([raw[name] for name in ("x", "y", "z")]).astype(float)
        else:
            points = np.asarray(list(raw), dtype=float)
    if points.size == 0:
        points = np.empty((0, 3), dtype=float)
    points = points.reshape((-1, 3))
    points = points[np.isfinite(points).all(axis=1)]
    summary = {
        "frame_id": str(msg.header.frame_id),
        "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
        "width": int(msg.width), "height": int(msg.height), "point_step": int(msg.point_step),
        "fields": fields, "finite_xyz_count": int(len(points)),
    }
    return points, summary


def fit_table_plane(points: Any) -> tuple[Any, float, int]:
    """RANSAC a dominant plane; it is the tabletop in the intended setup."""
    import numpy as np

    if len(points) < 100:
        raise ValueError(f"too few finite points: {len(points)}")
    rng = np.random.default_rng(20260821)
    sample = points if len(points) <= 12000 else points[rng.choice(len(points), 12000, replace=False)]
    best_normal = None
    best_offset = 0.0
    best_count = -1
    threshold = 0.006
    for _ in range(400):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal /= length
        offset = -float(normal @ a)
        count = int(np.count_nonzero(np.abs(sample @ normal + offset) <= threshold))
        if count > best_count:
            best_normal, best_offset, best_count = normal, offset, count
    if best_normal is None:
        raise ValueError("could not fit a tabletop plane")
    inliers = points[np.abs(points @ best_normal + best_offset) <= threshold]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    return normal, offset, int(len(inliers))


def largest_voxel_component(points: Any, voxel_m: float = 0.012) -> Any:
    """Keep the largest spatially connected protrusion; no object classifier."""
    import numpy as np

    if len(points) == 0:
        return points
    cells = np.floor(points / voxel_m).astype(int)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)
    visited: set[tuple[int, int, int]] = set()
    best: list[int] = []
    for start in buckets:
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        component: list[int] = []
        while queue:
            cell = queue.pop()
            component.extend(buckets[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        if neighbor in buckets and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        if len(component) > len(best):
            best = component
    return points[np.asarray(best, dtype=int)] if best else np.empty((0, 3), dtype=float)


def extract_nut_center(points: Any) -> tuple[Any, dict[str, Any]]:
    """Table-plane removal -> elevated region -> centroid, deliberately simple."""
    import numpy as np

    normal, offset, table_inliers = fit_table_plane(points)
    signed = points @ normal + offset
    # The tabletop normal has arbitrary sign.  The nut is the smaller near-table
    # elevated set, so choose the side containing fewer candidates.
    positive = points[(signed >= 0.008) & (signed <= 0.12)]
    negative = points[(signed <= -0.008) & (signed >= -0.12)]
    # One side is normally empty because the plane normal is arbitrary.  When
    # both sides contain geometry, prefer the smaller elevated set (the nut
    # rather than an attached structure).
    if len(positive) >= 8 and len(negative) < 8:
        candidates, side = positive, "positive"
    elif len(negative) >= 8 and len(positive) < 8:
        candidates, side = negative, "negative"
    elif len(positive) <= len(negative):
        candidates, side = positive, "positive"
    else:
        candidates, side = negative, "negative"
    if len(candidates) < 8:
        raise ValueError(f"no usable protruding region; positive={len(positive)} negative={len(negative)}")
    component = largest_voxel_component(candidates)
    if len(component) < 8:
        raise ValueError(f"largest protruding component too small: {len(component)}")
    center = component.mean(axis=0)
    diagnostics = {
        "table_plane": [float(normal[0]), float(normal[1]), float(normal[2]), float(offset)],
        "table_inlier_count": table_inliers,
        "positive_protrusion_count": int(len(positive)),
        "negative_protrusion_count": int(len(negative)),
        "selected_side": side,
        "selected_component_count": int(len(component)),
        "selected_component_bbox_min": component.min(axis=0).tolist(),
        "selected_component_bbox_max": component.max(axis=0).tolist(),
    }
    return center, diagnostics


def rigid_transform(camera: Any, world: Any) -> Any:
    """Kabsch/SVD least-squares rigid transform T_world_camera."""
    import numpy as np

    camera_mean = camera.mean(axis=0)
    world_mean = world.mean(axis=0)
    u, _, vt = np.linalg.svd((camera - camera_mean).T @ (world - world_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = world_mean - rotation @ camera_mean
    return transform


def main() -> int:
    report: dict[str, Any] = {
        "experiment": "nut_camera_calibration",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "set_pose_service": SET_POSE_SERVICE,
        "pointcloud_topic": POINTCLOUD_TOPIC,
        "nut_b_id": NUT_IDS["B"],
        "other_nuts": {"ids": {"A": NUT_IDS["A"], "C": NUT_IDS["C"]}, "action": "MOVE_OFF_SCENE", "pose": OFF_SCENE_POSE},
        "settle_seconds": SETTLE_SECONDS,
        "set_pose_retries": SET_POSE_RETRIES,
        "set_pose_cooldown_seconds": SET_POSE_COOLDOWN_SECONDS,
        "world_frame_tolerance_m": WORLD_FRAME_TOLERANCE_M,
        "experiments": [],
        "status": "RUNNING",
    }
    receiver: CloudReceiver | None = None
    pointcloud_frames_received = 0
    try:
        import numpy as np
        from rabo_dev_kit import SetEntityPose

        pose_setter = SetEntityPose(world=WORLD_ID)

        for label in ("A", "C"):
            set_result = set_pose_with_retry(pose_setter, NUT_IDS[label], OFF_SCENE_POSE)
            report["other_nuts"].setdefault("results", {})[label] = set_result
            if not set_result["ok"]:
                raise RuntimeError(f"failed to move Nut {label} off scene after {SET_POSE_RETRIES} attempts: {set_result['error']}")
            # Keep the reliable, high-bandwidth point-cloud stream disconnected
            # while the set-pose service and simulator settle.
            time.sleep(SET_POSE_COOLDOWN_SECONDS)

        for index, xyz in enumerate(WORLD_POSITIONS, start=1):
            pose = [*xyz, *NUT_RPY]
            trial: dict[str, Any] = {"index": index, "world_xyz": xyz, "nut_b_pose": pose}
            trial["set_pose"] = set_pose_with_retry(pose_setter, NUT_IDS["B"], pose)
            if not trial["set_pose"]["ok"]:
                trial["status"] = "SET_POSE_FAILED"
                report["experiments"].append(trial)
                raise RuntimeError(f"Nut B set_pose failed at trial {index}: {trial['set_pose']['error']}")

            # Subscribe only after set_pose has returned, receive a fresh cloud,
            # then tear down the subscription before the next service call.
            receiver = start_receiver()
            marker = time.monotonic()
            try:
                receiver.spin_for(SETTLE_SECONDS)
                msg = receiver.next_after(marker, POINT_TIMEOUT_SECONDS)
            finally:
                pointcloud_frames_received += receiver.frames
                receiver.close()
                receiver = None
            if msg is None:
                trial["status"] = "POINTCLOUD_TIMEOUT"
                report["experiments"].append(trial)
                raise RuntimeError(f"no PointCloud2 received after trial {index}")
            points, cloud_info = cloud_xyz(msg)
            trial["pointcloud"] = cloud_info
            center, segmentation = extract_nut_center(points)
            trial["camera_xyz"] = center.tolist()
            trial["pointcloud_center"] = center.tolist()
            trial["segmentation"] = segmentation
            trial["raw_camera_to_world_error_m"] = float(np.linalg.norm(center - np.asarray(xyz)))
            trial["status"] = "OK"
            report["experiments"].append(trial)

        valid = [item for item in report["experiments"] if item.get("status") == "OK"]
        if len(valid) != len(WORLD_POSITIONS):
            raise RuntimeError(f"only {len(valid)}/{len(WORLD_POSITIONS)} trials completed")
        camera = np.asarray([item["camera_xyz"] for item in valid], dtype=float)
        world = np.asarray([item["world_xyz"] for item in valid], dtype=float)
        raw_errors = np.linalg.norm(camera - world, axis=1)
        report["raw_world_frame_errors_m"] = raw_errors.tolist()
        report["raw_world_frame_rmse_m"] = float(np.sqrt(np.mean(raw_errors ** 2)))
        if float(np.max(raw_errors)) <= WORLD_FRAME_TOLERANCE_M:
            report["frame_decision"] = "POINTCLOUD_IS_WORLD_FRAME"
            report["needs_transform"] = False
            report["transform_matrix_camera_to_world"] = np.eye(4).tolist()
            report["transform_fit_rmse_m"] = report["raw_world_frame_rmse_m"]
        else:
            transform = rigid_transform(camera, world)
            mapped = (transform[:3, :3] @ camera.T).T + transform[:3, 3]
            errors = np.linalg.norm(mapped - world, axis=1)
            report["frame_decision"] = "POINTCLOUD_NEEDS_CAMERA_TO_WORLD_TRANSFORM"
            report["needs_transform"] = True
            report["transform_matrix_camera_to_world"] = transform.tolist()
            report["transform_fit_errors_m"] = errors.tolist()
            report["transform_fit_rmse_m"] = float(np.sqrt(np.mean(errors ** 2)))
        report["pointcloud_frames_received"] = pointcloud_frames_received
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        print(f"FAILED: {exc}", file=sys.stderr)
    finally:
        if receiver is not None:
            receiver.close()
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_report(report)
        print(f"result: {relative(REPORT_PATH)}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
