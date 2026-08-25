"""Configuration for the Rabo three-nut expert.

These values are extracted from the working legacy demo first, then extended
conservatively. The B-nut path is the legacy-proven path. A/C placement poses
are staged for cloud validation and should not be treated as proven until the
single-nut gate passes repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass


WORLD_ID = "ww4b1e6f8392584c89902a6783a9f53075"

NUT_IDS = {
    "A": "thing_3eb64241-4166-4c4e-9144-edf733c885f1",
    "B": "thing_e937f633-faed-4b2e-b609-a7b80825a64a",
    "C": "thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7",
}

DEVICE_IDS = {
    "RIGHT_ARM": "r412d237980e3167577d7aece10f7aedb",
    "RIGHT_HAND": "rcd72e2daf71f064c29aa45d4eeceeca9",
    "LEFT_ARM": "rbd03ebf4ebf83c6a6a64754454bc520a",
    "LEFT_HAND": "r136d7b4b6e527ea3875679b4bf7eeb7d",
}


@dataclass(frozen=True)
class Pose6:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class NutSpec:
    key: str
    thing_id: str
    nominal_pose: Pose6
    jitter_xy: float = 0.02


# Current Rabo scene object poses, manually verified in the UI property panel.
KNOWN_FIXED_NUT_WORLD_POSE_STATUS = "VERIFIED_SCENE_FIXED_POSE"
KNOWN_FIXED_NUT_WORLD_POSE = {
    "A": Pose6(-0.2286, -0.0999, 0.2815, 0.0, 0.0, 0.5233),
    "B": Pose6(-0.3413, -0.1710, 0.2806, 0.0, 0.0, 0.5233),
    "C": Pose6(-0.2975, -0.0527, 0.2868, 0.0, 0.0, 0.5233),
}

# Historical staged A/C coordinates. Never use these as current-scene targets.
OBSOLETE_STAGED_NUT_WORLD_POSE_STATUS = "OBSOLETE_STAGED_INCORRECT_FOR_CURRENT_SCENE"
OBSOLETE_STAGED_NUT_WORLD_POSE = {
    "A": Pose6(-0.3413, -0.2310, 0.2806, 0.0, 0.0, 0.5233),
    "C": Pose6(-0.3413, -0.1110, 0.2806, 0.0, 0.0, 0.5233),
}

NUT_SPECS = {
    key: NutSpec(key, NUT_IDS[key], pose)
    for key, pose in KNOWN_FIXED_NUT_WORLD_POSE.items()
}

# Legacy right arm base in table/world x-y convention.
RIGHT_ARM_BASE_XY = (-0.6816, -0.004)
RIGHT_ARM_BASE_WORLD_Z = 0.752

# Verified Nut B grasp relation, expressed in world Z.  X/Y retain the
# legacy right-base conversion below so the frozen B command stays exact.
RIGHT_GRASP_OFFSET_WORLD_Z = 0.14140
RIGHT_APPROACH_HEIGHT = 0.10

# Legacy-proven right-arm grasp target transform:
# target_x = right_arm_base_x - nut_world_x + 0.06
# target_y = right_arm_base_y - nut_world_y - 0.01
GRASP_TARGET_OFFSETS = {
    "x": 0.06,
    "y": -0.01,
    # Compatibility/reference value for the verified B pose only.
    # compute_right_grasp_pose derives Z from Nut world Z instead of this key.
    "z": -0.33,
    "roll": 0.0,
    "pitch": 0.8,
    "yaw": 0.0,
}

RIGHT_PRE_JOINTS = [
    [-1.57, -1.5, 0.0, -1.57, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0],
]

LEFT_PRE_JOINTS = [
    [0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.57, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
]

# Legacy Nut B absolute transfer poses. Do not use these as the first lift
# after a dynamic A/B/C grasp; that lift must be derived from the grasp pose.
RIGHT_LIFT_POSES = [
    Pose6(-0.4, 0.12, -0.03, 0.0, 0.8, 0.0),
    Pose6(-0.4, 0.0, -0.03, 0.0, 0.8, 0.0),
]

LEFT_COARSE_POSE = Pose6(0.43, 0.3, -0.1, 0.0, 1.3, 1.57)
LEFT_HANDOFF_POSE = Pose6(0.44, 0.06, -0.2, 0.0, 1.3, 1.57)
RIGHT_CLEAR_POSE = Pose6(-0.4, 0.0, 0.12, 0.0, 0.8, 0.0)

LEFT_PLACE_POSES = {
    "A": Pose6(0.39, 0.20, -0.2, 0.0, 0.0, 0.0),
    "B": Pose6(0.43, 0.25, -0.2, 0.0, 0.0, 0.0),
    "C": Pose6(0.47, 0.30, -0.2, 0.0, 0.0, 0.0),
}

LEFT_PRE_OPEN = (0.3, 0.0, 0.3, 0.3, 0.3, 0.3)
LEFT_GRASP = (1.0, 0.0, 0.3, 0.3, 0.3, 0.3)
HAND_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

RIGHT_GRASP_FORCE = {"strength": 1.0, "fingers": [1, 3, 4]}
LEFT_GRASP_FORCE = {"strength": 0.5, "fingers": None}
