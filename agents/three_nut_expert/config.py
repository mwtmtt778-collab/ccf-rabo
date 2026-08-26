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


@dataclass(frozen=True)
class RecordedJointWaypoint:
    """One unmodified actual_joint sample from a recorded successful motion."""

    joints: tuple[float, float, float, float, float, float, float]
    source_trajectory: str
    sample_index: int
    time_s: float


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
    "B": Pose6(0.33, 0.25, -0.20, 0.0, 0.0, 0.0),
    "C": Pose6(0.43, 0.25, -0.20, 0.0, 0.0, 0.0),
}

# [VERIFIED] Extracted only from samples[*].actual_joint in the visually
# successful fixed-map C -> B episode below.  These are sparse recorded poses,
# not IK results or interpolated/smoothed joints.  B has no retreat path: its
# Cartesian retreat returned false without moving, so deterministic C -> B
# treats B Place + release as terminal success.
LEFT_FIXED_PLACE_REFERENCE_EPISODE = "episode_20260826_144708_105746"
DETERMINISTIC_PLACE_ENTRY_ELIGIBILITY_RAD = 0.25
DETERMINISTIC_PLACE_ENTRY_FINAL_TOLERANCE_RAD = 0.05
LEFT_FIXED_PLACE_JOINT_PATHS = {
    "C": {
        "transport": (
            RecordedJointWaypoint(
                (-1.6728717609560564, -0.004959265970217467, 0.6101380784178074, -1.8443547098017632, -0.6223608418873436, 0.8239798649495675, -0.9540226078577456),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 0, 0.0026,
            ),
            RecordedJointWaypoint(
                (-1.687376415407574, -0.04819958705857103, 0.5450732445276057, -1.6794260052489243, -0.5568917513689916, 0.5163014491757174, -0.7958028819452164),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 13, 1.7701,
            ),
            RecordedJointWaypoint(
                (-1.702362405191888, -0.09288058977879952, 0.4778202154790521, -1.508929615207862, -0.4892071708507591, 0.19801043432793358, -0.6321791173604087),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 20, 2.7523,
            ),
            RecordedJointWaypoint(
                (-1.718043679575431, -0.14020973713107915, 0.4728613775347352, -1.3285258812944907, -0.4174482808433397, -0.12749425346677248, -0.45158426468740454),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 29, 4.0916,
            ),
            RecordedJointWaypoint(
                (-1.7340894385729213, -0.1875084246932481, 0.40082136427810544, -1.1480123536308244, -0.3457817871552649, -0.46452698935678993, -0.2762759841502463),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 40, 5.5632,
            ),
            RecordedJointWaypoint(
                (-1.7503662027507585, -0.23605426358793388, 0.3277461731829667, -0.9627443050002092, -0.272228381341416, -0.8104248069498955, -0.09846039902719511),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_C_5/trajectory.json", 57, 7.8301,
            ),
        ),
        "place": (
            RecordedJointWaypoint(
                (-1.750521477006989, -0.23651731006120177, 0.3270492075039012, -0.9609774240545258, -0.2715269547329822, -0.8143555421107874, -0.09676473964328502),
                "reports/motion_monitor/20260826_LEFT_PLACE_C_18/trajectory.json", 0, 0.0035,
            ),
            RecordedJointWaypoint(
                (-1.7514995598720726, -0.2417600688582314, 0.3172345029126499, -0.9601250830212762, -0.27477739092630266, -0.819945422095821, -0.09035816467305036),
                "reports/motion_monitor/20260826_LEFT_PLACE_C_18/trajectory.json", 2, 0.1904,
            ),
            RecordedJointWaypoint(
                (-1.7545560688393673, -0.2581436901192904, 0.2835878520120282, -0.9574615173624611, -0.28493500679184547, -0.8474329220255362, -0.11150418220407034),
                "reports/motion_monitor/20260826_LEFT_PLACE_C_18/trajectory.json", 4, 0.398,
            ),
        ),
        "retreat": (
            RecordedJointWaypoint(
                (-1.755167370308227, -0.26142041440376396, 0.2761737947745301, -0.9568222622413998, -0.2873728374430741, -0.8666385417857809, -0.1337248346536176),
                "reports/motion_monitor/20260826_LEFT_SAFE_RETREAT_C_15/trajectory.json", 0, 0.003,
            ),
            RecordedJointWaypoint(
                (-1.8487957815228768, -0.2640940540500515, 0.2754605304749604, -0.9774121617678968, -0.26483756131027375, -0.9315090417858851, -0.1476220759181255),
                "reports/motion_monitor/20260826_LEFT_SAFE_RETREAT_C_15/trajectory.json", 3, 0.343,
            ),
            RecordedJointWaypoint(
                (-2.0828668096160574, -0.2706685777719912, 0.2736654820520874, -1.0301083452181126, -0.20849937091824658, -1.097533541786872, -0.1831895916828203),
                "reports/motion_monitor/20260826_LEFT_SAFE_RETREAT_C_15/trajectory.json", 8, 0.9258,
            ),
        ),
    },
    "B": {
        "transport": (
            RecordedJointWaypoint(
                (-0.3662448692619467, 0.06996240252444824, -0.47631579449584, 1.7282889430480983, 0.8389607619274174, -1.0424430919994088, -1.0613933063324588),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 0, 0.0002,
            ),
            RecordedJointWaypoint(
                (-0.4269753988219561, -0.09457831689791628, -0.5244357958568999, 1.6532759977657203, 1.189989777597729, -1.0772641479496823, -1.1721799053156083),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 12, 1.6646,
            ),
            RecordedJointWaypoint(
                (-0.48010889461523076, -0.2385764306557788, -0.5665546760455382, 1.587619638573194, 1.4950901972107333, -1.1086965350161357, -1.2699103551875117),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 19, 2.5772,
            ),
            RecordedJointWaypoint(
                (-0.5356326153359474, -0.38904096851307457, -0.609409392073834, 1.5349763641208214, 1.8116954409434471, -1.0776911384178092, -1.3124952899001017),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 27, 3.6706,
            ),
            RecordedJointWaypoint(
                (-0.5964536302208985, -0.5538073253521614, -0.657632523565246, 1.4583135529721138, 2.1603186472583182, -1.1037820593340546, -1.3506713133342294),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 39, 5.2843,
            ),
            RecordedJointWaypoint(
                (-0.6575336530387141, -0.7193466545995756, -0.706050387852744, 1.3828214719019933, 2.5111639810276607, -1.13778838521321, -1.4492796847163085),
                "reports/motion_monitor/20260826_LEFT_PLACE_ABOVE_B/trajectory.json", 53, 7.1652,
            ),
        ),
        "place": (
            RecordedJointWaypoint(
                (-0.6575336530158759, -0.7193466545955933, -0.7060503879142797, 1.382821471862824, 2.511163981227528, -1.137788384970716, -1.4492796846609386),
                "reports/motion_monitor/20260826_LEFT_PLACE_B_5/trajectory.json", 0, 0.0025,
            ),
            RecordedJointWaypoint(
                (-0.6741218461286648, -0.7310564273213634, -0.73156390409461, 1.389983756500358, 2.5211182984593705, -1.1398359190708585, -1.46892234476096),
                "reports/motion_monitor/20260826_LEFT_PLACE_B_5/trajectory.json", 3, 0.3165,
            ),
            RecordedJointWaypoint(
                (-0.7346687510475797, -0.7737970978173025, -0.8246882382107452, 1.4161260954710653, 2.5574515521922088, -1.1473094192336792, -1.5491858442787902),
                "reports/motion_monitor/20260826_LEFT_PLACE_B_5/trajectory.json", 7, 0.8943,
            ),
        ),
    },
}

LEFT_PRE_OPEN = (0.3, 0.0, 0.3, 0.3, 0.3, 0.3)
LEFT_GRASP = (1.0, 0.0, 0.3, 0.3, 0.3, 0.3)
HAND_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

RIGHT_GRASP_FORCE = {"strength": 1.0, "fingers": [1, 3, 4]}
LEFT_GRASP_FORCE = {"strength": 0.5, "fingers": None}
