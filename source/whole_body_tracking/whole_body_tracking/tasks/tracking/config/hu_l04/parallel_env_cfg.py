"""Tracking environment for the supported closed-loop HU_L04 description.

The articulation has 51 DoF, only 27 of which may be commanded, and six of those are achilles
motor angles rather than ankle and waist angles.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.hu_l04_parallel import (
    HU_L04_PARALLEL_ACTION_JOINTS,
    HU_L04_PARALLEL_ACTION_SCALE,
    HU_L04_PARALLEL_CFG,
    HU_L04_PARALLEL_DEPLOY_JOINT_NAME_ALIASES,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg, TrackingEnvCfg


@configclass
class HuL04ParallelFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = HU_L04_PARALLEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The base config commands every joint. Here that would reach the six passive joints and
        # the eighteen rod DoF, whose position targets have to stay inert for the loop closures
        # to be the only thing deciding where they go.
        self.actions.joint_pos.joint_names = HU_L04_PARALLEL_ACTION_JOINTS
        self.actions.joint_pos.scale = HU_L04_PARALLEL_ACTION_SCALE

        # The achilles cranks and rods are internal to the mechanism and the retargeter has no
        # reference for them.
        self.commands.motion.anchor_body_name = "waist_pitch_link"
        self.commands.motion.body_names = [
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
        ]
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_pitch_link")

        # Perturbing the default position of a joint the policy does not command is not a
        # calibration error the policy can be robust to - it just shifts the zero of an
        # observation. Restrict the randomization to the 27 commanded joints.
        self.events.add_joint_default_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=HU_L04_PARALLEL_ACTION_JOINTS
        )
        # Likewise for the joint-limit penalty. The rod joints are spherical and their travel is
        # set by the loop, not by the policy, so charging the policy for reaching it would
        # penalize the mechanism working as designed.
        self.rewards.joint_limit.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=HU_L04_PARALLEL_ACTION_JOINTS
        )


@configclass
class HuL04ParallelWoStateEstimationEnvCfg(HuL04ParallelFlatEnvCfg):
    """Policy observations restricted to what the hardware can measure."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class HuL04ParallelPolicyWithGravityCfg(ObservationsCfg.PolicyCfg):
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))


@configclass
class HuL04ParallelPrivilegedWithGravityCfg(ObservationsCfg.PrivilegedCfg):
    projected_gravity = ObsTerm(func=mdp.projected_gravity)


@configclass
class HuL04ParallelWoSeGravityEnvCfg(HuL04ParallelWoStateEstimationEnvCfg):
    """Wo-state-estimation observations plus projected gravity."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = HuL04ParallelPolicyWithGravityCfg()
        self.observations.critic = HuL04ParallelPrivilegedWithGravityCfg()
        # Re-apply the removals: assigning a fresh group above resets them to their defaults.
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


# Terms the robot senses about itself. Only these get stacked: the command term is the reference
# trajectory looked up by time step, so its past frames carry nothing the policy cannot already
# derive from the current one.
PROPRIOCEPTIVE_TERMS = (
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "projected_gravity",
)

OBS_HISTORY_LENGTH = 6


@configclass
class HuL04ParallelWoSeGravityHistoryEnvCfg(HuL04ParallelWoSeGravityEnvCfg):
    """Adds a proprioceptive observation history so the policy can infer its own motion.

    Without base_lin_vel a single frame gives the policy no measure of horizontal velocity, and
    on the serial model that produced a policy which shuffled instead of stepping. Stacking
    frames lets it difference them, which is how the deployed controller supplies the same
    information. The critic stays at one frame - it observes base_lin_vel directly.

    The closed-loop model makes this more useful rather than less: joint_pos now also carries the
    passive ankle and waist angles, so the stacked history is the only place the policy can see
    how the linkage responded to what it commanded.
    """

    def __post_init__(self):
        super().__post_init__()
        for name in PROPRIOCEPTIVE_TERMS:
            term = getattr(self.observations.policy, name)
            term.history_length = OBS_HISTORY_LENGTH
            term.flatten_history_dim = True


def restrict_policy_to_hardware(policy: ObservationsCfg.PolicyCfg) -> None:
    """Cut the policy group down to what the robot can actually produce at run time.

    Three things go, on top of what the wo-state-estimation variant already drops:

    ``motion_anchor_ori_b`` needs the anchor body's world orientation, which on hardware only
    exists as a forward-kinematics estimate. The controller does compute one, but in
    parallel_solver_mode 0 the joint vector it feeds Pinocchio holds motor angles where the model
    expects ankle and waist angles, so the estimate is wrong. It is only ever published as a
    diagnostic - it reaches neither the command nor the e-stop - which is what makes dropping the
    observation the right fix rather than trusting a wrong number.

    ``joint_pos`` and ``joint_vel`` are narrowed to the driven joints. The other twenty-four are
    the six loop-driven ankle and waist axes, which the robot could report but which are a
    function of the motors it already sees, and the eighteen rod DoF, for which it has no sensor
    at all and no way to reconstruct one - the spherical joints' axial DoF is not determined by
    the two anchor points.
    """
    policy.motion_anchor_pos_b = None
    policy.motion_anchor_ori_b = None
    policy.base_lin_vel = None
    # Separate SceneEntityCfg objects: the manager resolves them in place, and a shared one would
    # be resolved twice and then fail its own consistency check.
    policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=HU_L04_PARALLEL_ACTION_JOINTS)}
    policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=HU_L04_PARALLEL_ACTION_JOINTS)}


@configclass
class HuL04ParallelDeployEnvCfg(HuL04ParallelWoStateEstimationEnvCfg):
    """Closed-loop tracking with an observation the deployed controller can assemble.

    138 = command(54) + base_ang_vel(3) + joint_pos(27) + joint_vel(27) + actions(27), every joint
    term over the 27 driven joints and in the same order, which is the one thing the deployed
    controller cannot work around: it carries a single joint count and a single permutation for
    the reference, the measurement, the gains and the actions alike.

    The critic is left alone. It runs only during training, so privileged terms and the full 51
    DoF cost nothing here.
    """

    def __post_init__(self):
        super().__post_init__()
        restrict_policy_to_hardware(self.observations.policy)
        self.commands.motion.observed_joint_names = HU_L04_PARALLEL_ACTION_JOINTS
        self.export_actuated_joints_only = True
        self.export_joint_name_aliases = HU_L04_PARALLEL_DEPLOY_JOINT_NAME_ALIASES


@configclass
class HuL04ParallelDeployGravityEnvCfg(HuL04ParallelDeployEnvCfg):
    """Deployable observation plus projected gravity: 141 instead of 138.

    Gravity is free on hardware - it is the IMU reading the controller already has - and it is the
    only direct measure of trunk tilt left once motion_anchor_ori_b is gone.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = HuL04ParallelPolicyWithGravityCfg()
        self.observations.critic = HuL04ParallelPrivilegedWithGravityCfg()
        # Assigning a fresh group above resets it to the class defaults, so the restrictions have
        # to be applied again.
        restrict_policy_to_hardware(self.observations.policy)
