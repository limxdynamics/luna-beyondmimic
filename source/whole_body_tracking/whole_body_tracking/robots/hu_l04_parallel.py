"""HU_L04_01 with the ankle and waist achilles linkages simulated as real closed loops.

The description's serial model - ``HU_L04_description/usd/`` and the URDF beside it - drives
``*_ankle_pitch/roll_joint`` and ``waist_roll/pitch_joint`` directly, which fixes their stiffness
at a constant. On hardware those four axes are moved by a pair of achilles rods, so the stiffness
the joint sees is ``J(q)^T K J(q)`` and varies over the workspace. That mismatch is what breaks
sim2sim: a policy trained on the serial model stands but loses the feet as soon as the ankles
start moving.

Here the loop is simulated instead of approximated. Six ``*_connect_joint`` spherical joints carry
``physics:excludeFromArticulation``, which is how PhysX expresses a kinematic loop: the tree stays
a tree and the closures are solved as constraints. The articulation therefore exposes 51 DoF
rather than 27:

  27 driven   21 ordinary serial joints + 6 achilles motors
   6 passive  ``*_ankle_pitch/roll_joint``, ``waist_roll/pitch_joint`` - moved only by the loop
  18 linkage  6 in-tree spherical rod joints, expanded by PhysX into ``<name>:0/:1/:2``

The action space is the 27 driven joints, so its width is unchanged, but six of them are achilles
motor angles rather than ankle and waist angles. Tracking rewards are unaffected: the passive
joints still exist and are still observable, so the policy learns to reach a commanded ankle angle
through the motors without any explicit linkage inverse kinematics.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import luna_description_path

# ``rotor_mass * gear_ratio ** 2`` for each actuator, read off the rotor table in
# ``HU_L04_description/urdf/HU_L04_01.srdf``. The ankle and waist numbers are the achilles
# actuator's rather than the joint's, which is the motor that actually exists in this model, so
# each number sits on the joint it was measured for.
ARMATURE_HIP_PITCH_YAW = 0.0731446
ARMATURE_HIP_ROLL_KNEE = 0.0731446
ARMATURE_ANKLE_WAIST_SHOULDER = 0.0177800
ARMATURE_ELBOW = 0.0127666
ARMATURE_HEAD_WRIST = 0.0163574

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP_PITCH_YAW = ARMATURE_HIP_PITCH_YAW * NATURAL_FREQ**2
STIFFNESS_HIP_ROLL_KNEE = ARMATURE_HIP_ROLL_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE_WAIST_SHOULDER = ARMATURE_ANKLE_WAIST_SHOULDER * NATURAL_FREQ**2
STIFFNESS_ELBOW = ARMATURE_ELBOW * NATURAL_FREQ**2
STIFFNESS_HEAD_WRIST = ARMATURE_HEAD_WRIST * NATURAL_FREQ**2

DAMPING_HIP_PITCH_YAW = 2.0 * DAMPING_RATIO * ARMATURE_HIP_PITCH_YAW * NATURAL_FREQ
DAMPING_HIP_ROLL_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_ROLL_KNEE * NATURAL_FREQ
DAMPING_ANKLE_WAIST_SHOULDER = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_WAIST_SHOULDER * NATURAL_FREQ
DAMPING_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_ELBOW * NATURAL_FREQ
DAMPING_HEAD_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_HEAD_WRIST * NATURAL_FREQ

# Joints the loop moves. They keep the USD's zero drive gains and infinite force limit; giving
# them any gain would put an actuator in series with the constraint that already determines them.
PASSIVE_JOINTS = [
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# The three DoF PhysX expands each in-tree spherical rod joint into. Their pose follows from the
# motor angle and the loop closure, so they are undriven for the same reason as the passive
# joints. The trailing ':' is what distinguishes them from the revolute joints.
LINKAGE_JOINTS = [".*_rod_joint:.*"]

# The six achilles motors, in place of the ankle and waist joints of the serial model.
ANKLE_MOTOR_JOINTS = [".*_A_achilles_joint", ".*_B_achilles_joint"]
WAIST_MOTOR_JOINTS = ["waist_A_joint", "waist_B_joint"]

HU_L04_PARALLEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_parallel/ is a sibling of the description's serial usd/ tree rather than a
        # replacement, so both remain available. The entry file is a 1.5 kB stub; the payload
        # lives in usd_parallel/configuration/ and is resolved relative to it.
        usd_path=luna_description_path("HU_L04_description", "usd_parallel", "HU_L04_01.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # The default 8/4 solver iterations already hold the six loop closures without drift, and
        # the loop costs about 1.5x throughput on its own, so there is no headroom to spend on
        # raising them.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.84),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=18.3,
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_PITCH_YAW,
                ".*_hip_roll_joint": STIFFNESS_HIP_ROLL_KNEE,
                ".*_hip_yaw_joint": STIFFNESS_HIP_PITCH_YAW,
                ".*_knee_joint": STIFFNESS_HIP_ROLL_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH_YAW,
                ".*_hip_roll_joint": DAMPING_HIP_ROLL_KNEE,
                ".*_hip_yaw_joint": DAMPING_HIP_PITCH_YAW,
                ".*_knee_joint": DAMPING_HIP_ROLL_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP_PITCH_YAW,
                ".*_hip_roll_joint": ARMATURE_HIP_ROLL_KNEE,
                ".*_hip_yaw_joint": ARMATURE_HIP_PITCH_YAW,
                ".*_knee_joint": ARMATURE_HIP_ROLL_KNEE,
            },
        ),
        # 70 Nm is the maxForce the USD gives these drives, and matches the limit the serial
        # config puts on the ankle joints they replace.
        "ankle_motors": ImplicitActuatorCfg(
            joint_names_expr=ANKLE_MOTOR_JOINTS,
            effort_limit_sim=70.0,
            velocity_limit_sim=24.5,
            stiffness=STIFFNESS_ANKLE_WAIST_SHOULDER,
            damping=DAMPING_ANKLE_WAIST_SHOULDER,
            armature=ARMATURE_ANKLE_WAIST_SHOULDER,
        ),
        "waist_motors": ImplicitActuatorCfg(
            joint_names_expr=WAIST_MOTOR_JOINTS,
            effort_limit_sim=70.0,
            velocity_limit_sim=24.5,
            stiffness=STIFFNESS_ANKLE_WAIST_SHOULDER,
            damping=DAMPING_ANKLE_WAIST_SHOULDER,
            armature=ARMATURE_ANKLE_WAIST_SHOULDER,
        ),
        # waist_yaw is the one waist axis with a direct drive on hardware, so it stays serial.
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=70.0,
            velocity_limit_sim=24.5,
            stiffness=STIFFNESS_ANKLE_WAIST_SHOULDER,
            damping=DAMPING_ANKLE_WAIST_SHOULDER,
            armature=ARMATURE_ANKLE_WAIST_SHOULDER,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=15.8,
            velocity_limit_sim=17.5,
            stiffness=STIFFNESS_HEAD_WRIST,
            damping=DAMPING_HEAD_WRIST,
            armature=ARMATURE_HEAD_WRIST,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            effort_limit_sim=70.0,
            velocity_limit_sim=24.5,
            stiffness=STIFFNESS_ANKLE_WAIST_SHOULDER,
            damping=DAMPING_ANKLE_WAIST_SHOULDER,
            armature=ARMATURE_ANKLE_WAIST_SHOULDER,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=26.9,
            stiffness=STIFFNESS_ELBOW,
            damping=DAMPING_ELBOW,
            armature=ARMATURE_ELBOW,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_yaw_joint"],
            effort_limit_sim=15.8,
            velocity_limit_sim=17.5,
            stiffness=STIFFNESS_HEAD_WRIST,
            damping=DAMPING_HEAD_WRIST,
            armature=ARMATURE_HEAD_WRIST,
        ),
        # The two groups below exist only so that every one of the 51 DoF belongs to exactly one
        # actuator. Articulation._process_actuators_cfg writes each group's gains into PhysX, so
        # leaving these joints out of the config is not equivalent to giving them zero gains -
        # and covering them with a catch-all '.*' double-counts the joints matched above, which
        # IsaacLab reports as "Total number of actuated joints not equal to number of joints
        # available". Effort, velocity and armature are left unset so the USD values survive.
        "passive": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINTS,
            stiffness=0.0,
            damping=0.0,
        ),
        "linkage": ImplicitActuatorCfg(
            joint_names_expr=LINKAGE_JOINTS,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

# The joints the policy commands. Everything else is either passive or a linkage DoF, and the
# action term must not reach them.
HU_L04_PARALLEL_ACTUATED_GROUPS = (
    "legs",
    "ankle_motors",
    "waist_motors",
    "waist_yaw",
    "head",
    "shoulders",
    "elbows",
    "wrists",
)
HU_L04_PARALLEL_ACTION_JOINTS = [
    name for group in HU_L04_PARALLEL_ACTUATED_GROUPS for name in HU_L04_PARALLEL_CFG.actuators[group].joint_names_expr
]

# What the six achilles motors are called on the robot.
#
# The deployed controller runs kinematic_projection in parallel_solver_mode 0, which bypasses the
# linkage solver in both directions. The six hardware slots that mode 1 would fill with ankle and
# waist angles then carry the raw achilles motor angles instead, but they keep the names the
# solver's joint space gave them - slot 4 is still called left_ankle_pitch_joint while holding the
# left A motor. There is no rename table on that side: the deployed controller requires the
# metadata's joint_names to be exactly the set in the robot's config.yaml. So the rename has to
# happen here, on the way out.
#
# scripts/verify_parallel_motor_convention.py checks that this is only a rename. It servos each
# loop over a grid of poses and compares the USD motor angle against what the hardware's own
# closed-form inverse kinematics puts in that slot: all six agree to ~1e-5 rad with no sign flip,
# no A/B swap and no offset. Note this holds for the waist too even though the hardware wraps its
# solver in a negate-and-cross - waist_A_joint is named for the slot it lands in, not for the link
# it drives, so the two cancel.
HU_L04_PARALLEL_DEPLOY_JOINT_NAME_ALIASES = {
    "left_A_achilles_joint": "left_ankle_pitch_joint",
    "left_B_achilles_joint": "left_ankle_roll_joint",
    "right_A_achilles_joint": "right_ankle_pitch_joint",
    "right_B_achilles_joint": "right_ankle_roll_joint",
    "waist_A_joint": "waist_roll_joint",
    "waist_B_joint": "waist_pitch_joint",
}

HU_L04_PARALLEL_ACTION_SCALE = {}
for group in HU_L04_PARALLEL_ACTUATED_GROUPS:
    a = HU_L04_PARALLEL_CFG.actuators[group]
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            HU_L04_PARALLEL_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
