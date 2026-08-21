#!/usr/bin/env python3
"""Detect the three tabletop Nut candidates from the fixed top PointCloud2.

This is a read-only geometry experiment.  It does not move robots or entities.
It reuses the verified camera topic/capture helpers and the camera_to_world
matrix produced by ``tools/test_nut_camera_calibration.py``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.test_nut_camera_calibration import (  # noqa: E402
    POINTCLOUD_TOPIC,
    REPORT_PATH as CALIBRATION_REPORT_PATH,
    capture_fresh_cloud,
    cloud_xyz,
    fit_table_plane,
)
from agents.three_nut_expert.config import NUT_SPECS  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "reports" / "nut_pointcloud_detection"
RESULT_PATH = OUTPUT_DIR / "result.json"

EXPECTED_NUTS = 3
TABLE_INLIER_THRESHOLD_M = 0.006  # documented RANSAC threshold in calibration helper
MIN_OBJECT_HEIGHT_M = 0.006
MAX_OBJECT_HEIGHT_M = 0.090
TABLE_CLEARANCE_CANDIDATES_M = (0.006, 0.010, 0.014, 0.018, 0.022, 0.026)
VOXEL_SIZE_M = 0.003
CLUSTER_EPS_M = 0.008
MIN_CLUSTER_VOXELS = 8
MIN_CLUSTER_POINTS = 40
MIN_NUT_XY_EXTENT_M = 0.012
MAX_NUT_XY_EXTENT_M = 0.100
MAX_NUT_Z_EXTENT_M = 0.080
NUT_WORKSPACE_MARGIN_X_M = 0.16
NUT_WORKSPACE_MARGIN_Y_M = 0.10
MAX_VISUALIZATION_POINTS = 15000

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def configured_nut_workspace() -> dict[str, Any]:
    nominal = {
        key: [float(spec.nominal_pose.x), float(spec.nominal_pose.y), float(spec.nominal_pose.z)]
        for key, spec in NUT_SPECS.items()
    }
    xs = [value[0] for value in nominal.values()]
    ys = [value[1] for value in nominal.values()]
    return {
        "source": "agents.three_nut_expert.config.NUT_SPECS",
        "nominal_centers_world": nominal,
        "bounds_world_xy": [
            min(xs) - NUT_WORKSPACE_MARGIN_X_M,
            max(xs) + NUT_WORKSPACE_MARGIN_X_M,
            min(ys) - NUT_WORKSPACE_MARGIN_Y_M,
            max(ys) + NUT_WORKSPACE_MARGIN_Y_M,
        ],
        "margin_xy_m": [NUT_WORKSPACE_MARGIN_X_M, NUT_WORKSPACE_MARGIN_Y_M],
        "note": "Loose ROI only; nominal coordinates are not used as detected centers.",
    }


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return repr(value)


def write_report(report: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_camera_to_world() -> tuple[np.ndarray, dict[str, Any]]:
    if not CALIBRATION_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"missing calibration report: {CALIBRATION_REPORT_PATH}; "
            "run python3 tools/test_nut_camera_calibration.py first"
        )
    calibration = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
    if calibration.get("status") != "PASS":
        raise ValueError(f"calibration status is not PASS: {calibration.get('status')!r}")
    matrix = np.asarray(calibration.get("transform_matrix_camera_to_world"), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("calibration report has no valid 4x4 camera_to_world matrix")
    source = {
        "path": str(CALIBRATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
        "frame_decision": calibration.get("frame_decision"),
        "transform_fit_rmse_m": calibration.get("transform_fit_rmse_m"),
        "matrix": matrix.tolist(),
    }
    return matrix, source


def camera_points_to_world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the already-calibrated camera_to_world matrix; no re-fitting."""
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def voxel_downsample(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / VOXEL_SIZE_M).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3), dtype=float)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None], counts.astype(np.int64)


def euclidean_components(points: np.ndarray) -> list[np.ndarray]:
    """DBSCAN-like Euclidean connected components using a spatial hash."""
    if len(points) == 0:
        return []
    cells = np.floor(points / CLUSTER_EPS_M).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(index)

    visited = np.zeros(len(points), dtype=bool)
    components: list[np.ndarray] = []
    epsilon_squared = CLUSTER_EPS_M * CLUSTER_EPS_M
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        members: list[int] = []
        while queue:
            current = queue.pop()
            members.append(current)
            cell = cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_cell = (int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz))
                        for neighbor in buckets.get(neighbor_cell, []):
                            if visited[neighbor]:
                                continue
                            delta = points[neighbor] - points[current]
                            if float(delta @ delta) <= epsilon_squared:
                                visited[neighbor] = True
                                queue.append(neighbor)
        if len(members) >= MIN_CLUSTER_VOXELS:
            components.append(np.asarray(members, dtype=np.int64))
    return components


def cluster_metrics(
    voxel_points_camera: np.ndarray,
    voxel_weights: np.ndarray,
    component: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    camera = voxel_points_camera[component]
    weights = voxel_weights[component]
    world = camera_points_to_world(camera, transform)
    center_camera = np.average(camera, axis=0, weights=weights)
    center_world = camera_points_to_world(center_camera.reshape(1, 3), transform)[0]
    bbox_min = world.min(axis=0)
    bbox_max = world.max(axis=0)
    size = bbox_max - bbox_min
    points = int(weights.sum())
    rejection_reasons: list[str] = []
    if points < MIN_CLUSTER_POINTS:
        rejection_reasons.append("TOO_FEW_POINTS")
    if float(size[0]) < MIN_NUT_XY_EXTENT_M:
        rejection_reasons.append("X_EXTENT_TOO_SMALL")
    if float(size[1]) < MIN_NUT_XY_EXTENT_M:
        rejection_reasons.append("Y_EXTENT_TOO_SMALL")
    if float(size[0]) > MAX_NUT_XY_EXTENT_M:
        rejection_reasons.append("X_EXTENT_TOO_LARGE")
    if float(size[1]) > MAX_NUT_XY_EXTENT_M:
        rejection_reasons.append("Y_EXTENT_TOO_LARGE")
    if float(size[2]) > MAX_NUT_Z_EXTENT_M:
        rejection_reasons.append("Z_EXTENT_TOO_LARGE")
    eligible = not rejection_reasons
    return {
        "points": points,
        "voxel_points": int(len(component)),
        "center_camera": center_camera.tolist(),
        "center_world": center_world.tolist(),
        "bbox": [*bbox_min.tolist(), *bbox_max.tolist()],
        "bbox_min_world": bbox_min.tolist(),
        "bbox_max_world": bbox_max.tolist(),
        "size": size.tolist(),
        "eligible_nut_geometry": eligible,
        "rejection_reasons": rejection_reasons,
        "_camera_points": camera,
        "_world_points": world,
    }


def detect_at_clearance(
    finite_camera: np.ndarray,
    height: np.ndarray,
    transform: np.ndarray,
    workspace: dict[str, Any],
    clearance_m: float,
) -> dict[str, Any]:
    height_mask = (height >= clearance_m) & (height <= MAX_OBJECT_HEIGHT_M)
    above_camera = finite_camera[height_mask]
    above_world = camera_points_to_world(above_camera, transform)
    x_min, x_max, y_min, y_max = workspace["bounds_world_xy"]
    roi_mask = (
        (above_world[:, 0] >= x_min)
        & (above_world[:, 0] <= x_max)
        & (above_world[:, 1] >= y_min)
        & (above_world[:, 1] <= y_max)
    )
    roi_camera = above_camera[roi_mask]
    if len(roi_camera) == 0:
        voxels_camera = np.empty((0, 3), dtype=float)
        voxel_weights = np.empty((0,), dtype=np.int64)
        components: list[np.ndarray] = []
    else:
        voxels_camera, voxel_weights = voxel_downsample(roi_camera)
        components = euclidean_components(voxels_camera)
    metrics = [cluster_metrics(voxels_camera, voxel_weights, component, transform) for component in components]
    eligible = [item for item in metrics if item["eligible_nut_geometry"]]
    return {
        "clearance_m": clearance_m,
        "above_camera": above_camera,
        "above_world": above_world,
        "roi_camera": roi_camera,
        "voxels_camera": voxels_camera,
        "voxel_weights": voxel_weights,
        "components": components,
        "metrics": metrics,
        "eligible": eligible,
    }


def clearance_summary(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "clearance_m": trial["clearance_m"],
        "points_after_table_filter": int(len(trial["above_camera"])),
        "points_in_nut_workspace_roi": int(len(trial["roi_camera"])),
        "voxel_points": int(len(trial["voxels_camera"])),
        "raw_cluster_num": int(len(trial["components"])),
        "eligible_cluster_num": int(len(trial["eligible"])),
        "raw_clusters": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in trial["metrics"]
        ],
    }


def sample_rows(points: np.ndarray, limit: int = MAX_VISUALIZATION_POINTS) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def save_svg_projection(
    path: Path,
    point_groups: list[np.ndarray],
    colors: list[str],
    title: str,
) -> None:
    """Save a dependency-free top-down world XY point-cloud visualization."""
    width, height, margin = 1000, 760, 45
    sampled = [sample_rows(group) for group in point_groups if len(group)]
    if not sampled:
        raise ValueError(f"no points for visualization {path.name}")
    all_points = np.vstack(sampled)
    x_low, x_high = np.percentile(all_points[:, 0], [1.0, 99.0])
    y_low, y_high = np.percentile(all_points[:, 1], [1.0, 99.0])
    if x_high - x_low < 1e-6:
        x_low, x_high = x_low - 0.5, x_high + 0.5
    if y_high - y_low < 1e-6:
        y_low, y_high = y_low - 0.5, y_high + 0.5

    def screen(point: np.ndarray) -> tuple[float, float]:
        x = margin + (float(point[0]) - x_low) / (x_high - x_low) * (width - 2 * margin)
        y = height - margin - (float(point[1]) - y_low) / (y_high - y_low) * (height - 2 * margin)
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101418"/>',
        f'<text x="{margin}" y="28" fill="white" font-family="sans-serif" font-size="18">{title}</text>',
        f'<text x="{margin}" y="{height - 12}" fill="#aaa" font-family="sans-serif" font-size="12">world X/Y top view</text>',
    ]
    for group_index, group in enumerate(sampled):
        color = colors[group_index % len(colors)]
        for point in group:
            x, y = screen(point)
            if margin <= x <= width - margin and margin <= y <= height - margin:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1" fill="{color}" fill-opacity="0.72"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "RUNNING",
        "pointcloud_topic": POINTCLOUD_TOPIC,
        "expected_cluster_num": EXPECTED_NUTS,
        "parameters": {
            "table_method": "RANSAC_DOMINANT_PLANE",
            "table_inlier_threshold_m": TABLE_INLIER_THRESHOLD_M,
            "object_height_range_m": [MIN_OBJECT_HEIGHT_M, MAX_OBJECT_HEIGHT_M],
            "table_clearance_candidates_m": list(TABLE_CLEARANCE_CANDIDATES_M),
            "voxel_size_m": VOXEL_SIZE_M,
            "cluster_eps_m": CLUSTER_EPS_M,
            "min_cluster_voxels": MIN_CLUSTER_VOXELS,
            "min_cluster_points": MIN_CLUSTER_POINTS,
            "nut_bbox_limits_m": {
                "min_xy_extent": MIN_NUT_XY_EXTENT_M,
                "max_xy_extent": MAX_NUT_XY_EXTENT_M,
                "max_z_extent": MAX_NUT_Z_EXTENT_M,
            },
        },
        "clusters": [],
    }
    try:
        transform, calibration_source = load_camera_to_world()
        report["camera_to_world_source"] = calibration_source
        workspace = configured_nut_workspace()
        report["nut_workspace_roi"] = workspace
        message, frames = capture_fresh_cloud()
        report["pointcloud_frames_received"] = frames
        if message is None:
            raise RuntimeError("no top-camera PointCloud2 received")
        points, cloud_info = cloud_xyz(message)
        finite = points[np.isfinite(points).all(axis=1)]
        report["pointcloud"] = cloud_info
        report["pointcloud_frame"] = cloud_info["frame_id"]
        report["raw_finite_points"] = int(len(finite))
        if len(finite) < 100:
            raise RuntimeError(f"too few finite points: {len(finite)}")

        normal, offset, table_inliers = fit_table_plane(finite)
        # Orient the plane normal toward +world Z using the verified rotation.
        if float((transform[:3, :3] @ normal)[2]) < 0.0:
            normal = -normal
            offset = -offset
        height = finite @ normal + offset
        report["table_plane_camera"] = [float(normal[0]), float(normal[1]), float(normal[2]), float(offset)]
        report["table_inlier_points"] = table_inliers
        clearance_trials = [
            detect_at_clearance(finite, height, transform, workspace, clearance)
            for clearance in TABLE_CLEARANCE_CANDIDATES_M
        ]
        exact = [trial for trial in clearance_trials if len(trial["eligible"]) == EXPECTED_NUTS]
        if exact:
            chosen = exact[0]
            selection_reason = "LOWEST_CLEARANCE_WITH_EXPECTED_CLUSTER_COUNT"
        else:
            chosen = min(
                clearance_trials,
                key=lambda trial: (
                    abs(len(trial["eligible"]) - EXPECTED_NUTS),
                    -len(trial["eligible"]),
                    trial["clearance_m"],
                ),
            )
            selection_reason = "CLOSEST_CLUSTER_COUNT_NO_EXACT_MATCH"
        report["clearance_trials"] = [clearance_summary(trial) for trial in clearance_trials]
        report["selected_table_clearance_m"] = chosen["clearance_m"]
        report["clearance_selection_reason"] = selection_reason
        above_table_camera = chosen["above_camera"]
        above_table_world = chosen["above_world"]
        objects_camera = chosen["roi_camera"]
        voxels_camera = chosen["voxels_camera"]
        components = chosen["components"]
        metrics = chosen["metrics"]
        eligible = chosen["eligible"]
        report["points_after_table_filter"] = int(len(above_table_camera))
        report["points_in_nut_workspace_roi"] = int(len(objects_camera))
        if len(objects_camera) < MIN_CLUSTER_POINTS:
            raise RuntimeError(f"too few above-table points: {len(objects_camera)}")
        eligible.sort(key=lambda item: item["points"], reverse=True)
        # Stable IDs are spatially ordered in world X then Y, not DBSCAN traversal order.
        selected = sorted(eligible, key=lambda item: (item["center_world"][0], item["center_world"][1]))

        report["voxel_points"] = int(len(voxels_camera))
        report["raw_cluster_num"] = int(len(components))
        report["raw_clusters"] = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in metrics
        ]
        report["eligible_cluster_num"] = int(len(eligible))
        report["cluster_num"] = int(len(selected))
        report["warning"] = None if len(selected) == EXPECTED_NUTS else f"Expected 3 clusters, got {len(selected)}"
        for cluster_id, item in enumerate(selected):
            report_item = {key: value for key, value in item.items() if not key.startswith("_")}
            report_item["id"] = cluster_id
            report["clusters"].append(report_item)

        raw_world = camera_points_to_world(finite, transform)
        objects_world = camera_points_to_world(objects_camera, transform)
        save_svg_projection(OUTPUT_DIR / "raw_pointcloud.svg", [raw_world], ["#8da0cb"], "Raw top-camera point cloud")
        save_svg_projection(
            OUTPUT_DIR / "objects_no_table.svg",
            [above_table_world],
            ["#66c2a5"],
            "Above-table points",
        )
        save_svg_projection(
            OUTPUT_DIR / "nut_workspace_roi.svg",
            [objects_world],
            ["#ffd92f"],
            "Above-table points inside Nut workspace ROI",
        )
        cluster_world_groups = [item["_world_points"] for item in selected]
        if cluster_world_groups:
            save_svg_projection(OUTPUT_DIR / "clusters.svg", cluster_world_groups, COLORS, "Nut candidate clusters")
        report["visualizations"] = {
            "raw": "reports/nut_pointcloud_detection/raw_pointcloud.svg",
            "objects_no_table": "reports/nut_pointcloud_detection/objects_no_table.svg",
            "nut_workspace_roi": "reports/nut_pointcloud_detection/nut_workspace_roi.svg",
            "clusters": "reports/nut_pointcloud_detection/clusters.svg" if cluster_world_groups else None,
        }
        report["status"] = "PASS" if len(selected) == EXPECTED_NUTS else "CHECK"

        print(f"Detected clusters: {len(selected)}")
        if report["warning"]:
            print(f"WARNING: Expected 3 clusters, got {len(selected)}")
        for item in report["clusters"]:
            print(f"\nNut candidate {item['id']}:")
            print(f"  points: {item['points']}")
            print(f"  camera frame: {item['center_camera']}")
            print(f"  world frame:  {item['center_world']}")
            print(f"  size world:   {item['size']}")
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        print(f"FAILED: {exc}", file=sys.stderr)
    finally:
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_report(report)
        print(f"result: {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if report["status"] in {"PASS", "CHECK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
