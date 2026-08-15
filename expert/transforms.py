"""SE(3) pose utilities for Rabo workspace analysis.

Pose format is [x, y, z, roll, pitch, yaw], using radians and a right-handed
coordinate convention. Rotation order is Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

from __future__ import annotations

import math
from typing import Iterable


Matrix4 = list[list[float]]
Pose6 = list[float]


def _pose6(pose: Iterable[float]) -> Pose6:
    values = [float(v) for v in pose]
    if len(values) != 6:
        raise ValueError(f"pose must have 6 values, got {len(values)}")
    return values


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return matmul(matmul(rz, ry), rx)


def matrix_to_rpy(r: list[list[float]]) -> tuple[float, float, float]:
    pitch = math.atan2(-r[2][0], math.sqrt(r[0][0] ** 2 + r[1][0] ** 2))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-r[0][1], r[1][1])
    return roll, pitch, yaw


def pose6_to_matrix(pose: Iterable[float]) -> Matrix4:
    x, y, z, roll, pitch, yaw = _pose6(pose)
    r = rpy_to_matrix(roll, pitch, yaw)
    return [
        [r[0][0], r[0][1], r[0][2], x],
        [r[1][0], r[1][1], r[1][2], y],
        [r[2][0], r[2][1], r[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_pose6(t: Matrix4) -> Pose6:
    r = [row[:3] for row in t[:3]]
    roll, pitch, yaw = matrix_to_rpy(r)
    return [float(t[0][3]), float(t[1][3]), float(t[2][3]), float(roll), float(pitch), float(yaw)]


def invert_transform(t: Matrix4) -> Matrix4:
    r = [row[:3] for row in t[:3]]
    p = [t[0][3], t[1][3], t[2][3]]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    inv_p = [-sum(rt[i][j] * p[j] for j in range(3)) for i in range(3)]
    return [
        [rt[0][0], rt[0][1], rt[0][2], inv_p[0]],
        [rt[1][0], rt[1][1], rt[1][2], inv_p[1]],
        [rt[2][0], rt[2][1], rt[2][2], inv_p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def compose_transform(t_a_b: Matrix4, t_b_c: Matrix4) -> Matrix4:
    return matmul(t_a_b, t_b_c)


def transform_pose_world_to_base(target_world: Iterable[float], base_world: Iterable[float]) -> Pose6:
    t_world_target = pose6_to_matrix(target_world)
    t_world_base = pose6_to_matrix(base_world)
    t_base_target = compose_transform(invert_transform(t_world_base), t_world_target)
    return matrix_to_pose6(t_base_target)


def transform_pose_base_to_world(target_base: Iterable[float], base_world: Iterable[float]) -> Pose6:
    t_base_target = pose6_to_matrix(target_base)
    t_world_base = pose6_to_matrix(base_world)
    t_world_target = compose_transform(t_world_base, t_base_target)
    return matrix_to_pose6(t_world_target)


def pose_position_error(a: Iterable[float], b: Iterable[float]) -> float:
    pa = _pose6(a)[:3]
    pb = _pose6(b)[:3]
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def pose_orientation_error(a: Iterable[float], b: Iterable[float]) -> float:
    pa = _pose6(a)[3:]
    pb = _pose6(b)[3:]
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def pose_rotation_angle_error(a: Iterable[float], b: Iterable[float]) -> float:
    ra = rpy_to_matrix(*_pose6(a)[3:])
    rb = rpy_to_matrix(*_pose6(b)[3:])
    r_delta = matmul([[ra[j][i] for j in range(3)] for i in range(3)], rb)
    trace = r_delta[0][0] + r_delta[1][1] + r_delta[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cos_angle)
