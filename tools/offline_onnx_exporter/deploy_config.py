"""The one deployment configuration this exporter supports.

HU_L04_01 running the closed-loop ``hu_l04_parallel`` description, with a policy whose observation
ends in ``projected_gravity``. Everything the export needs - network shape, hardware metadata,
reference layout and the geometry of the three achilles linkages - is written down here rather
than read from a spec file, so an export cannot be run against a configuration nobody checked.

Every number here is derived from the pinned ``luna-description`` submodule - the gains from its
armature table, the joint travel from its URDF and MJCF - so the two stay in step. The robot's own
``config.yaml`` is a separate artifact and is not guaranteed to agree; see the note above
``REFERENCE_JOINT_LIMITS``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

POLICY_ACTIVATION = "elu"
OBSERVATION_DIM = 141
ACTION_DIM = 27
NORMALIZER_EPSILON = 0.01
ONNX_OPSET_VERSION = 11

# 1 / (decimation * sim.dt) of TrackingEnvCfg. DanceLibrary advances one reference row per control
# step and ignores any rate stored alongside it, so the TXT has to be written at this rate.
REFERENCE_FPS = 50
REFERENCE_COLUMN_COUNT = 102

# Observation terms in policy order, all unstacked. This is what makes the export gravity-only:
# a 138-wide checkpoint trained without the last term will be rejected on width.
OBSERVATION_NAMES = (
    "command",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "projected_gravity",
)

_OBSERVATION_TERM_WIDTHS = {
    "command": 2 * ACTION_DIM,
    "base_ang_vel": 3,
    "joint_pos": ACTION_DIM,
    "joint_vel": ACTION_DIM,
    "actions": ACTION_DIM,
    "projected_gravity": 3,
}


def observation_width() -> int:
    return sum(_OBSERVATION_TERM_WIDTHS[name] for name in OBSERVATION_NAMES)


@dataclass(frozen=True)
class Linkage:
    """One achilles loop: two motors moving one roll/pitch pair through two rods.

    ``motor_names`` are the deployment slot names the two solutions land in, in that order.
    ``negate_and_cross`` reproduces the negate-and-cross step the hardware wraps around the waist
    solver but not around the ankles.
    """

    name: str
    roll_joint_name: str
    pitch_joint_name: str
    motor_names: tuple[str, str]
    negate_and_cross: bool
    r_a0: tuple[tuple[float, float, float], tuple[float, float, float]]
    r_b0: tuple[tuple[float, float, float], tuple[float, float, float]]
    r_c0: tuple[tuple[float, float, float], tuple[float, float, float]]
    pitch_offset: float


# Geometry copied from the deployment's linkage configuration for HU_L04_01. The motor names are
# the hardware slots, which keep the ankle and waist names even though they hold achilles motors -
# see HU_L04_PARALLEL_DEPLOY_JOINT_NAME_ALIASES in the robot config and the sweep in
# scripts/verify_parallel_motor_convention.py, which shows this is a rename and nothing more.
MECHANISMS = (
    Linkage(
        name="left_ankle",
        roll_joint_name="left_ankle_roll_joint",
        pitch_joint_name="left_ankle_pitch_joint",
        motor_names=("left_ankle_pitch_joint", "left_ankle_roll_joint"),
        negate_and_cross=False,
        r_a0=((0.02895, 0.02555, 0.30061), (0.02793, -0.02555, 0.21217)),
        r_b0=((-0.02405, 0.02555, 0.30061), (-0.02507, -0.02555, 0.21217)),
        r_c0=((-0.053, 0.02555, 0.0), (-0.053, -0.02555, 0.0)),
        pitch_offset=0.0,
    ),
    Linkage(
        name="right_ankle",
        roll_joint_name="right_ankle_roll_joint",
        pitch_joint_name="right_ankle_pitch_joint",
        motor_names=("right_ankle_pitch_joint", "right_ankle_roll_joint"),
        negate_and_cross=False,
        r_a0=((0.02895, -0.02555, 0.30061), (0.02793, 0.02555, 0.21217)),
        r_b0=((-0.02405, -0.02555, 0.30061), (-0.02507, 0.02555, 0.21217)),
        r_c0=((-0.053, -0.02555, 0.0), (-0.053, 0.02555, 0.0)),
        pitch_offset=0.0,
    ),
    Linkage(
        name="waist",
        roll_joint_name="waist_roll_joint",
        pitch_joint_name="waist_pitch_joint",
        motor_names=("waist_roll_joint", "waist_pitch_joint"),
        negate_and_cross=True,
        r_a0=((0.0, 0.06, 0.07), (0.0, -0.06, 0.07)),
        r_b0=((-0.068, 0.06, 0.07), (-0.068, -0.06, 0.07)),
        r_c0=((-0.068, 0.06, 0.0), (-0.068, -0.06, 0.0)),
        pitch_offset=-0.0,
    ),
)

_LINKAGE_MOTOR_NAMES = tuple(name for mechanism in MECHANISMS for name in mechanism.motor_names)

# The 27 reference joints in DanceLibrary order. Six of them are achilles motors the retargeter
# never produces; the rest are read straight out of the motion by name.
REFERENCE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
)

# The joint axis in the motion.npz consumed by training, copied from scripts/npy_to_npz.py for the
# HU_L04 parallel articulation. That file is intentionally not imported here so the exporter keeps
# working as a standalone tool.
TRAINING_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_A_joint",
    "waist_B_joint",
    "left_A_achilles_joint",
    "left_B_achilles_joint",
    "left_ankle_pitch_joint",
    "right_A_achilles_joint",
    "right_B_achilles_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_A_rod_joint:0",
    "waist_A_rod_joint:1",
    "waist_A_rod_joint:2",
    "waist_B_rod_joint:0",
    "waist_B_rod_joint:1",
    "waist_B_rod_joint:2",
    "left_A_achilles_rod_joint:0",
    "left_A_achilles_rod_joint:1",
    "left_A_achilles_rod_joint:2",
    "left_B_achilles_rod_joint:0",
    "left_B_achilles_rod_joint:1",
    "left_B_achilles_rod_joint:2",
    "left_ankle_roll_joint",
    "right_A_achilles_rod_joint:0",
    "right_A_achilles_rod_joint:1",
    "right_A_achilles_rod_joint:2",
    "right_B_achilles_rod_joint:0",
    "right_B_achilles_rod_joint:1",
    "right_B_achilles_rod_joint:2",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# Six deployment reference slots are named after ankle/waist joints but carry raw achilles motor
# angles in the training motion.
_REFERENCE_TO_TRAINING_JOINT_NAMES = {
    "left_ankle_pitch_joint": "left_A_achilles_joint",
    "left_ankle_roll_joint": "left_B_achilles_joint",
    "right_ankle_pitch_joint": "right_A_achilles_joint",
    "right_ankle_roll_joint": "right_B_achilles_joint",
    "waist_roll_joint": "waist_A_joint",
    "waist_pitch_joint": "waist_B_joint",
}
REFERENCE_TRAINING_JOINT_NAMES = tuple(
    _REFERENCE_TO_TRAINING_JOINT_NAMES.get(name, name) for name in REFERENCE_JOINT_NAMES
)

DIRECT_JOINT_NAMES = tuple(name for name in REFERENCE_JOINT_NAMES if name not in _LINKAGE_MOTOR_NAMES)

# The joints the retargeted motion has to carry: the ones copied through, plus the roll/pitch pairs
# the linkages are solved from.
SOURCE_JOINT_NAMES = DIRECT_JOINT_NAMES + tuple(
    name for mechanism in MECHANISMS for name in (mechanism.roll_joint_name, mechanism.pitch_joint_name)
)

BASE_BODY_NAME = "base_link"

# Head, left foot, right foot, left hand, right hand, in the order DanceLibrary reads them.
KEY_BODY_NAMES = (
    "head_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

# The body axis in the motion.npz consumed by training, copied from scripts/npy_to_npz.py for the
# HU_L04 parallel articulation. This file has to stand alone outside the repository, so the list is
# duplicated rather than imported; test_deploy_config_tracks_description.py pins the two together
# and to the description's own body order.
TRAINING_BODY_NAMES = (
    "base_link",
    # Fixed-joint IMU mount, present on HU_L04 but not on HU_L03. It holds a slot in the
    # articulation and therefore in the npz, even though the retargeter never fills it.
    "base_imu",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "waist_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "waist_A_link",
    "waist_B_link",
    "left_A_achilles_link",
    "left_B_achilles_link",
    "left_ankle_pitch_link",
    "right_A_achilles_link",
    "right_B_achilles_link",
    "right_ankle_pitch_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "waist_A_rod_link",
    "waist_B_rod_link",
    "left_A_achilles_rod_link",
    "left_B_achilles_rod_link",
    "left_ankle_roll_link",
    "right_A_achilles_rod_link",
    "right_B_achilles_rod_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "contact_foot_center_L",
    "contact_foot_heel_L",
    "contact_foot_tip_L",
    "contact_foot_center_R",
    "contact_foot_heel_R",
    "contact_foot_tip_R",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_hand_contact",
    "left_hand_manip",
    "right_hand_contact",
    "right_hand_manip",
)

# Hardware slot order, from the robot's config.yaml. The deployed controller permutes the
# policy's actions into this order, so it is not the reference order above.
DEPLOY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# Travel of each reference joint, in REFERENCE_JOINT_NAMES order, taken from the description this
# repository pins: HU_L04_description/urdf/HU_L04_01.urdf for the twenty-one serial joints, and
# HU_L04_description/xml/HU_L04_01.xml for the other six. Those six columns carry achilles motors
# rather than the ankle and waist angles they are named after, and only the MJCF declares a range
# for a motor. The distinction matters at the waist, where the motor travel is the joint's
# reflected: waist_pitch_joint is (-0.5236, 0.7854) but waist_B_joint is (-0.7854, 0.5236).
#
# The description is the authority here rather than the robot's own config.yaml, which is not
# published for HU_L04. The two are not guaranteed to agree - on HU_L03 they disagreed on hip roll
# by 10 degrees of offset and gave the shoulders considerably more travel than the URDF does.
# Confirm this table against the hardware's config.yaml before trusting an export on a real robot;
# where they differ, the robot wins.
REFERENCE_JOINT_LIMITS = (
    (-3.0543, 1.5708),      # left_hip_pitch_joint
    (-1.0472, 1.5708),      # left_hip_roll_joint
    (-1.5708, 1.5708),      # left_hip_yaw_joint
    (-0.087266, 2.4435),    # left_knee_joint
    (-1.0123, 0.61086),     # left_ankle_pitch_joint (left_A_achilles_joint in motor space)
    (-1.0123, 0.61086),     # left_ankle_roll_joint (left_B_achilles_joint in motor space)
    (-3.0543, 1.5708),      # right_hip_pitch_joint
    (-1.5708, 1.0472),      # right_hip_roll_joint
    (-1.5708, 1.5708),      # right_hip_yaw_joint
    (-0.087266, 2.4435),    # right_knee_joint
    (-1.0123, 0.61086),     # right_ankle_pitch_joint (right_A_achilles_joint in motor space)
    (-1.0123, 0.61086),     # right_ankle_roll_joint (right_B_achilles_joint in motor space)
    (-1.5708, 1.5708),      # waist_yaw_joint
    (-0.7854, 0.5236),      # waist_roll_joint (waist_A_joint in motor space)
    (-0.7854, 0.5236),      # waist_pitch_joint (waist_B_joint in motor space)
    (-0.7854, 0.7854),      # head_yaw_joint
    (-0.5236, 0.5236),      # head_pitch_joint
    (-3.0543, 1.0472),      # left_shoulder_pitch_joint
    (-0.087266, 1.9199),    # left_shoulder_roll_joint
    (-2.5307, 1.5708),      # left_shoulder_yaw_joint
    (-2.3911, 0.5236),      # left_elbow_joint
    (-1.5708, 1.5708),      # left_wrist_yaw_joint
    (-3.0543, 1.0472),      # right_shoulder_pitch_joint
    (-1.9199, 0.087266),    # right_shoulder_roll_joint
    (-1.5708, 2.5307),      # right_shoulder_yaw_joint
    (-2.3911, 0.5236),      # right_elbow_joint
    (-1.5708, 1.5708),      # right_wrist_yaw_joint
)  # fmt: skip

# Small linkage-workspace misses are accepted by training and settle at the nearest motor stop.
# Permit the offline closed-form path to do the same, but reference_converter clips every accepted
# value back to the hard limit before publishing it. Larger misses still indicate incompatible data.
JOINT_LIMIT_TOLERANCE = 2.0e-2

# Gains in DEPLOY_JOINT_NAMES order. Each one is derived rather than measured:
# ``stiffness = armature * (2*pi*10)**2`` and ``damping = 2 * 2.0 * armature * (2*pi*10)`` over the
# armature table in HU_L04_description/urdf/HU_L04_01.srdf, matching the derivation in
# whole_body_tracking.robots.hu_l04_parallel. All four leg joints share one armature on HU_L04, so
# hip pitch and hip yaw carry the same numbers as hip roll and knee.
DEPLOY_JOINT_STIFFNESS = (
    288.763, 288.763, 70.193, 288.763, 288.763, 288.763, 288.763, 288.763, 288.763,
    64.576, 70.193, 70.193, 70.193, 70.193, 70.193, 70.193, 70.193, 70.193,
    64.576, 70.193, 70.193, 70.193, 70.193, 50.401, 50.401, 64.576, 64.576,
)  # fmt: skip

DEPLOY_JOINT_DAMPING = (
    18.383, 18.383, 4.469, 18.383, 18.383, 18.383, 18.383, 18.383, 18.383,
    4.111, 4.469, 4.469, 4.469, 4.469, 4.469, 4.469, 4.469, 4.469,
    4.111, 4.469, 4.469, 4.469, 4.469, 3.209, 3.209, 4.111, 4.111,
)  # fmt: skip

DEPLOY_DEFAULT_JOINT_POS = (0.0,) * ACTION_DIM

# ``0.25 * effort_limit / stiffness``, the same expression that builds
# HU_L04_PARALLEL_ACTION_SCALE.
DEPLOY_ACTION_SCALE = (
    0.173, 0.173, 0.249, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173,
    0.061, 0.249, 0.249, 0.249, 0.249, 0.249, 0.249, 0.249, 0.249,
    0.061, 0.249, 0.249, 0.249, 0.249, 0.149, 0.149, 0.061, 0.061,
)  # fmt: skip

ANCHOR_BODY_NAME = "waist_pitch_link"

# The bodies the tracking command was defined over, which the controller reports against.
DEPLOY_BODY_NAMES = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "head_pitch_link",
)

DEFAULT_RUN_PATH = "none"


def onnx_metadata(run_path: str = DEFAULT_RUN_PATH) -> dict[str, object]:
    """The metadata the deployed controller parses out of the model."""
    return {
        "run_path": run_path,
        "joint_names": list(DEPLOY_JOINT_NAMES),
        "joint_stiffness": list(DEPLOY_JOINT_STIFFNESS),
        "joint_damping": list(DEPLOY_JOINT_DAMPING),
        "default_joint_pos": list(DEPLOY_DEFAULT_JOINT_POS),
        "command_names": ["motion"],
        "observation_names": list(OBSERVATION_NAMES),
        "observation_history_lengths": [1] * len(OBSERVATION_NAMES),
        "action_scale": list(DEPLOY_ACTION_SCALE),
        "anchor_body_name": ANCHOR_BODY_NAME,
        "body_names": list(DEPLOY_BODY_NAMES),
    }


def _rot_x(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(*c.shape, 3, 3)


def _rot_y(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(*c.shape, 3, 3)


# The loop closure reduces to ``a*cos(angle) + b*sin(angle) = c``, in metres. A solution is only
# the right one if it satisfies that, which is what separates a real answer from the arcsin branch
# the solver would otherwise return past a quarter turn.
LOOP_CLOSURE_TOLERANCE = 1.0e-9


def solve_motor_angles(mechanism: Linkage, roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """The hardware's closed-form inverse kinematics, plus the slot mapping wrapped around it.

    Returns ``(frames, 2)``: the value each of ``mechanism.motor_names`` holds when the loop is
    closed on the given roll and pitch. The loop is planar in xz, so y drops out of both lengths.
    """
    roll = np.asarray(roll, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    rotation = _rot_y(pitch) @ _rot_x(roll)

    motors = np.zeros(roll.shape + (2,))
    for index in range(2):
        r_a0 = np.asarray(mechanism.r_a0[index])
        r_b0 = np.asarray(mechanism.r_b0[index])
        r_c0 = np.asarray(mechanism.r_c0[index])
        length_bar = float(np.hypot(r_b0[0] - r_a0[0], r_b0[2] - r_a0[2]))
        length_rod = float(np.hypot(r_b0[0] - r_c0[0], r_b0[2] - r_c0[2]))

        delta = np.einsum("...ij,j->...i", rotation, r_c0) - r_a0
        a = delta[..., 0]
        b = -delta[..., 2]
        c = (length_rod**2 - length_bar**2 - np.sum(delta * delta, axis=-1)) / (2.0 * length_bar)
        discriminant = b * b * c * c - (a * a + b * b) * (c * c - a * a)
        _reject_unreachable(mechanism, discriminant >= 0.0, "the linkage cannot close on that pose")

        angle = np.arcsin((b * c + np.sqrt(np.maximum(discriminant, 0.0))) / (a * a + b * b))
        residual = np.abs(a * np.cos(angle) + b * np.sin(angle) - c)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.abs(c))
        _reject_unreachable(
            mechanism,
            np.isfinite(angle) & (residual <= LOOP_CLOSURE_TOLERANCE * np.maximum(scale, 1.0)),
            "the joint travels past the quarter turn the closed-form solution is defined on",
        )
        motors[..., index] = angle + mechanism.pitch_offset

    if mechanism.negate_and_cross:
        motors = -motors[..., ::-1]
    return motors


def _reject_unreachable(mechanism: Linkage, acceptable: np.ndarray, reason: str) -> None:
    if bool(np.all(acceptable)):
        return
    first = int(np.argmin(np.asarray(acceptable).reshape(-1)))
    raise ValueError(
        f"{mechanism.name}: the motion asks for a pose outside the linkage workspace of the deployment "
        f"model at frame {first} - {reason}."
    )
