"""Check that an achilles motor in the closed-loop USD is the same motor, with the same sign and
zero, as the hardware slot the deployed controller writes it to.

A policy trained on the closed-loop articulation commands six achilles motors. The deployed
controller runs ``kinematic_projection`` in ``parallel_solver_mode = 0``, which bypasses the
linkage solver in both directions, so those six actions land straight on the six achilles motors
of the robot - but they land there through hardware slots whose *names* still say ankle pitch,
ankle roll, waist roll and waist pitch. Whether that is the right thing to do depends on a
question the names cannot answer: is ``left_A_achilles_joint`` in the USD the motor that hardware
slot 4 drives, turning the same way, measured from the same zero?

The hardware has no forward model of the linkage that can be queried directly, but its inverse
kinematics is closed form and its geometry is three YAML files. Reimplementing it here gives a
hardware-side prediction "at this ankle pose the motors read this much", which can be held against
what the USD's own loop closures produce.

The comparison runs pose by pose:

1. Servo the six achilles motors in the USD over a grid and let the loops settle. The pose that
   results is whatever the USD geometry says; nothing is assumed about it.
2. Read the passive ankle/waist angles back. Those are the joint-space coordinates the hardware
   solver takes as input, and both descriptions define them with the same URDF joint.
3. Run the hardware inverse kinematics on them, then push the answer through the same index and
   sign juggling the hardware applies, to get the value each hardware slot would hold.
4. Compare against the USD motor angles, mapped to slots by the alias table the ONNX exporter
   uses.

Agreement means the alias table is a rename and nothing else. Disagreement is characterized rather
than corrected: the four candidate relations (identity, A/B swap, negation, both) are each fitted
with a free offset and reported, so the failure mode is named instead of guessed at.

.. code-block:: bash

    python scripts/verify_parallel_motor_convention.py \
        --config_dir /path/to/HU_L04_01/linkage-config
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the achilles motor sign and zero conventions.")
parser.add_argument(
    "--config_dir",
    type=str,
    default=None,
    help="Required directory holding the deployment linkage geometry YAMLs.",
)
parser.add_argument("--amplitude", type=float, default=0.3, help="Half-width of the motor sweep, rad.")
parser.add_argument("--samples", type=int, default=7, help="Grid points per motor; the sweep is samples^2 poses.")
parser.add_argument("--settle_steps", type=int, default=120, help="Simulation steps per ramp stage.")
parser.add_argument("--ramp_stages", type=int, default=8, help="Stages the motor targets are ramped over.")
parser.add_argument("--tolerance", type=float, default=5.0e-4, help="Pass threshold on the motor angle, rad.")
parser.add_argument(
    "--limit_margin",
    type=float,
    default=1.0e-3,
    help="A pose whose passive joint sits this close to its stop is jammed, and is not measured.",
)
parser.add_argument("--dump_grid", action="store_true", help="Print every swept pose, not just the worst.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.config_dir is None:
    parser.error("--config_dir is required")
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import sys
import torch
import yaml
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from whole_body_tracking.robots.hu_l04_parallel import HU_L04_PARALLEL_CFG

# Gains for the sweep only, as in solve_parallel_linkage.py: this is a kinematics measurement, so
# the motors are made far stiffer than hardware to reach steady state quickly, and the undriven
# DoF get a little damping to bleed off the energy each step injects.
SWEEP_MOTOR_STIFFNESS = 4000.0
SWEEP_MOTOR_DAMPING = 200.0
SWEEP_MOTOR_EFFORT = 1.0e4
SWEEP_PASSIVE_DAMPING = 0.2


@dataclass(frozen=True)
class Mechanism:
    """One achilles loop, described from both sides.

    ``slot_from_model`` is the whole of the hardware-side convention: it turns the two motor
    angles the solver returns for links A and B of ``model_file`` into the values that land in the
    two hardware slots, reproducing what the hardware does around the solver call.
    ``usd_motors`` are listed in the order the alias table maps them onto ``slot_names``.
    """

    name: str
    model_file: str
    usd_motors: tuple[str, str]
    usd_roll_joint: str
    usd_pitch_joint: str
    usd_proximal_body: str
    usd_distal_body: str
    slot_names: tuple[str, str]
    slot_from_model: str  # "AB" or "-BA", see below
    source: str


# The ankle pair's hardware mapping takes link A into slot index_A and link B into slot index_B,
# untouched. The waist's hardware mapping negates and crosses the pair, so the slot named
# waist_roll holds -(link B) and the slot named waist_pitch holds -(link A).
MECHANISMS = (
    Mechanism(
        name="left_ankle",
        model_file="left_ankle_model.yaml",
        usd_motors=("left_A_achilles_joint", "left_B_achilles_joint"),
        usd_roll_joint="left_ankle_roll_joint",
        usd_pitch_joint="left_ankle_pitch_joint",
        usd_proximal_body="left_knee_link",
        usd_distal_body="left_ankle_roll_link",
        slot_names=("left_ankle_pitch_joint", "left_ankle_roll_joint"),
        slot_from_model="AB",
        source="hardware ankle mapping, slots 4/5",
    ),
    Mechanism(
        name="right_ankle",
        model_file="right_ankle_model.yaml",
        usd_motors=("right_A_achilles_joint", "right_B_achilles_joint"),
        usd_roll_joint="right_ankle_roll_joint",
        usd_pitch_joint="right_ankle_pitch_joint",
        usd_proximal_body="right_knee_link",
        usd_distal_body="right_ankle_roll_link",
        slot_names=("right_ankle_pitch_joint", "right_ankle_roll_joint"),
        slot_from_model="AB",
        source="hardware ankle mapping, slots 10/11",
    ),
    Mechanism(
        name="waist",
        model_file="inverse_waist_model.yaml",
        usd_motors=("waist_A_joint", "waist_B_joint"),
        usd_roll_joint="waist_roll_joint",
        usd_pitch_joint="waist_pitch_joint",
        usd_proximal_body="waist_yaw_link",
        usd_distal_body="waist_pitch_link",
        slot_names=("waist_roll_joint", "waist_pitch_joint"),
        slot_from_model="-BA",
        source="hardware waist mapping (negate+cross), slots 13/14",
    ),
)


def load_model(path: str) -> dict:
    """Read one linkage geometry and derive the two lengths the hardware uses."""
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    model = {
        "r_a0": [np.array(raw["A_r_a0"], dtype=float), np.array(raw["B_r_a0"], dtype=float)],
        "r_b0": [np.array(raw["A_r_b0"], dtype=float), np.array(raw["B_r_b0"], dtype=float)],
        "r_c0": [np.array(raw["A_r_c0"], dtype=float), np.array(raw["B_r_c0"], dtype=float)],
        "pitch_offset": float(raw["pitch_offset"]),
    }
    # The linkage is planar in xz, so y drops out of both lengths.
    model["l_bar"] = [
        float(np.hypot(model["r_b0"][i][0] - model["r_a0"][i][0], model["r_b0"][i][2] - model["r_a0"][i][2]))
        for i in range(2)
    ]
    model["l_rod"] = [
        float(np.hypot(model["r_b0"][i][0] - model["r_c0"][i][0], model["r_b0"][i][2] - model["r_c0"][i][2]))
        for i in range(2)
    ]
    return model


def rot_x(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(*c.shape, 3, 3)


def rot_y(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(*c.shape, 3, 3)


def inverse_kinematics(model: dict, roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """The hardware's closed-form inverse kinematics, vectorized over poses.

    Returns the motor angle of link A and link B, in that order, for each (roll, pitch).
    """
    x_rot = rot_y(pitch) @ rot_x(roll)
    out = np.zeros(roll.shape + (2,))
    for i in range(2):
        r_c = np.einsum("...ij,j->...i", x_rot, model["r_c0"][i])
        delta = r_c - model["r_a0"][i]
        a = delta[..., 0]
        b = -delta[..., 2]
        c = (model["l_rod"][i] ** 2 - model["l_bar"][i] ** 2 - np.sum(delta * delta, axis=-1)) / (2 * model["l_bar"][i])
        discriminant = b * b * c * c - (a * a + b * b) * (c * c - a * a)
        if np.any(discriminant < 0.0):
            raise ValueError("The requested pose is outside the linkage workspace of the deployment model.")
        out[..., i] = np.arcsin((b * c + np.sqrt(discriminant)) / (a * a + b * b)) + model["pitch_offset"]
    return out


def model_to_slots(mechanism: Mechanism, motor: np.ndarray) -> np.ndarray:
    """Apply the index and sign juggling the hardware wraps around the solver."""
    if mechanism.slot_from_model == "AB":
        return motor
    if mechanism.slot_from_model == "-BA":
        return -motor[..., ::-1]
    raise ValueError(mechanism.slot_from_model)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """wxyz quaternions to rotation matrices."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*w.shape, 3, 3)  # fmt: skip


def decompose_ry_rx(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a rotation as Ry(pitch) @ Rx(roll), the composition the deployment solver assumes."""
    roll = -np.arcsin(np.clip(rotation[..., 1, 2], -1.0, 1.0))
    pitch = np.arctan2(-rotation[..., 2, 0], rotation[..., 0, 0])
    return roll, pitch


def build_scene(num_envs: int) -> tuple[SimulationContext, Articulation]:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.005, gravity=(0.0, 0.0, 0.0)))

    cfg = HU_L04_PARALLEL_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Robot"
    cfg.spawn.activate_contact_sensors = False
    # A mesh must not be what decides where a joint settles, and the root must not drift: the
    # measurement is of the linkage alone.
    cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.spawn.articulation_props.fix_root_link = True

    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class SweepSceneCfg(InteractiveSceneCfg):
        robot = cfg

    scene = InteractiveScene(SweepSceneCfg(num_envs=num_envs, env_spacing=3.0))
    sim.reset()
    return sim, scene["robot"]


def settle(sim: SimulationContext, robot: Articulation, targets: torch.Tensor, steps: int) -> None:
    robot.set_joint_position_target(targets)
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())


def fit_relation(predicted: np.ndarray, measured: np.ndarray, swap: bool, sign: float) -> tuple[float, np.ndarray]:
    """Residual of ``predicted ~= sign * measured[perm] + offset`` with offset fitted per slot."""
    candidate = sign * (measured[:, ::-1] if swap else measured)
    offset = (predicted - candidate).mean(axis=0)
    return float(np.abs(predicted - candidate - offset).max()), offset


def main() -> int:
    grid = np.linspace(-args_cli.amplitude, args_cli.amplitude, args_cli.samples)
    poses = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    num_envs = poses.shape[0]

    models = {m.name: load_model(os.path.join(args_cli.config_dir, m.model_file)) for m in MECHANISMS}
    print(f"[INFO] Deployment geometry from {args_cli.config_dir}")
    for mechanism in MECHANISMS:
        model = models[mechanism.name]
        print(
            f"       {mechanism.name:<12} l_bar={model['l_bar'][0]:.5f}/{model['l_bar'][1]:.5f}"
            f"  l_rod={model['l_rod'][0]:.5f}/{model['l_rod'][1]:.5f}"
            f"  pitch_offset={model['pitch_offset']:+.5f}"
        )

    sim, robot = build_scene(num_envs)
    index = {name: i for i, name in enumerate(robot.joint_names)}
    body_index = {name: i for i, name in enumerate(robot.body_names)}

    motor_ids = [index[name] for mechanism in MECHANISMS for name in mechanism.usd_motors]
    passive_ids = [
        index[name] for mechanism in MECHANISMS for name in (mechanism.usd_roll_joint, mechanism.usd_pitch_joint)
    ]
    linkage_ids = [i for i, name in enumerate(robot.joint_names) if ":" in name]
    undriven_ids = passive_ids + linkage_ids
    # Everything that is neither a motor nor part of a loop is welded at zero. The linkage pushes
    # hard enough to move the shank against its nominal gains, and a shank that has drifted breaks
    # the body-frame reading of the pose even though the joint angles stay right.
    held_ids = [i for i in range(len(robot.joint_names)) if i not in set(motor_ids + undriven_ids)]

    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    effort = robot.data.joint_effort_limits.clone()
    stiffness[:, motor_ids + held_ids] = SWEEP_MOTOR_STIFFNESS
    damping[:, motor_ids + held_ids] = SWEEP_MOTOR_DAMPING
    effort[:, motor_ids + held_ids] = SWEEP_MOTOR_EFFORT
    stiffness[:, undriven_ids] = 0.0
    damping[:, undriven_ids] = SWEEP_PASSIVE_DAMPING
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.write_joint_effort_limit_to_sim(effort)

    zeros = torch.zeros_like(robot.data.joint_pos)
    robot.write_joint_state_to_sim(zeros.clone(), zeros.clone())
    settle(sim, robot, zeros, args_cli.settle_steps)

    goal = torch.zeros_like(robot.data.joint_pos)
    for mechanism in MECHANISMS:
        goal[:, index[mechanism.usd_motors[0]]] = torch.tensor(poses[:, 0], dtype=torch.float32, device=sim.device)
        goal[:, index[mechanism.usd_motors[1]]] = torch.tensor(poses[:, 1], dtype=torch.float32, device=sim.device)
    lower = robot.data.joint_pos_limits[..., 0]
    upper = robot.data.joint_pos_limits[..., 1]
    clamped = int((goal[:, motor_ids] != goal[:, motor_ids].clamp(lower[:, motor_ids], upper[:, motor_ids])).sum())
    if clamped:
        print(f"[WARN] {clamped} motor targets sit outside the USD joint limits and will not be reached.")

    # Ramped rather than teleported: writing the motors to their goal while the passive joints are
    # still at zero opens every loop closure at once, and PhysX resolves that as an impulse large
    # enough to leave the mechanism somewhere the geometry did not put it.
    for stage in range(1, args_cli.ramp_stages + 1):
        settle(sim, robot, goal * (stage / args_cli.ramp_stages), args_cli.settle_steps)

    joint_pos = robot.data.joint_pos.cpu().numpy()
    limits = robot.data.joint_pos_limits.cpu().numpy()
    held_drift = float(np.abs(joint_pos[:, held_ids]).max())
    print(f"[INFO] Worst drift on the joints held at zero: {held_drift:.2e} rad.")
    failures = []
    for mechanism in MECHANISMS:
        model = models[mechanism.name]
        # The achieved motor angle, not the commanded one: any servo error then cancels out of the
        # comparison instead of being read as a convention mismatch.
        measured_motor = np.stack([joint_pos[:, index[n]] for n in mechanism.usd_motors], axis=1)
        roll = joint_pos[:, index[mechanism.usd_roll_joint]]
        pitch = joint_pos[:, index[mechanism.usd_pitch_joint]]

        # A passive joint on its stop is not a pose the linkage produced: the stop is holding the
        # loop open against the motors, so there is nothing there for a kinematic model of the
        # closed loop to predict. The ankle roll limit is +-25 deg and the motors reach past it.
        free = np.ones(num_envs, dtype=bool)
        for joint in (mechanism.usd_roll_joint, mechanism.usd_pitch_joint):
            value = joint_pos[:, index[joint]]
            lower, upper = limits[:, index[joint], 0], limits[:, index[joint], 1]
            free &= (value > lower + args_cli.limit_margin) & (value < upper - args_cli.limit_margin)

        # The solver builds the pose as Ry(pitch) @ Rx(roll). Whether that is how the two passive
        # joints actually compose is a property of the description, not of the motors, so the
        # comparison is run twice: once on the joint angles, which is what the hardware solver
        # would be handed, and once on the pose the bodies actually take, which removes the
        # composition from the picture entirely and leaves only the motor convention.
        relative = np.einsum(
            "...ji,...jk->...ik",
            quat_to_matrix(robot.data.body_quat_w[:, body_index[mechanism.usd_proximal_body]].cpu().numpy()),
            quat_to_matrix(robot.data.body_quat_w[:, body_index[mechanism.usd_distal_body]].cpu().numpy()),
        )
        body_roll, body_pitch = decompose_ry_rx(relative)
        composition_error = max(np.abs(body_roll - roll).max(), np.abs(body_pitch - pitch).max())

        from_joints = model_to_slots(mechanism, inverse_kinematics(model, roll, pitch))
        from_pose = model_to_slots(mechanism, inverse_kinematics(model, body_roll, body_pitch))
        error = np.abs(from_joints - measured_motor)

        print()
        print(f"{'=' * 96}")
        print(f"{mechanism.name}  ({mechanism.source})")
        print(f"{'=' * 96}")
        print(f"  USD motor       {mechanism.usd_motors[0]:<28} -> hardware slot {mechanism.slot_names[0]}")
        print(f"                  {mechanism.usd_motors[1]:<28} -> hardware slot {mechanism.slot_names[1]}")
        print(
            f"  poses           {int(free.sum())} free of {num_envs} swept over"
            f" [{-args_cli.amplitude:+.2f}, {args_cli.amplitude:+.2f}] rad; {int((~free).sum())} jammed on a stop"
        )
        print(
            f"  reached         roll {roll[free].min():+.4f}..{roll[free].max():+.4f},"
            f" pitch {pitch[free].min():+.4f}..{pitch[free].max():+.4f}"
        )
        print(f"  worst |IK(joint angles) - USD motor|: {error[free, 0].max():.3e} / {error[free, 1].max():.3e} rad")
        # Same comparison run on the pose the bodies actually take. It differs from the line above
        # only where Ry(pitch)Rx(roll) is not how the two passive joints compose, which is a
        # property of the deployment model, not of the motors.
        pose_error = np.abs(from_pose - measured_motor)
        print(
            f"  worst |IK(actual pose)  - USD motor|: {pose_error[free, 0].max():.3e}"
            f" / {pose_error[free, 1].max():.3e} rad"
        )
        print(f"  Ry(pitch)Rx(roll) vs the USD chain: worst {composition_error:.2e} rad on (roll, pitch)")

        print(f"  {'relation':<30}{'worst residual':>16}{'fitted offset A':>18}{'fitted offset B':>18}")
        best = None
        for label, swap, sign in (
            ("identity (assumed)", False, 1.0),
            ("A/B swapped", True, 1.0),
            ("negated", False, -1.0),
            ("negated and swapped", True, -1.0),
        ):
            residual, offset = fit_relation(from_joints[free], measured_motor[free], swap, sign)
            print(f"  {label:<30}{residual:>16.3e}{offset[0]:>18.5f}{offset[1]:>18.5f}")
            if best is None or residual < best[1]:
                best = (label, residual)

        worst = error[free].max()
        if worst > args_cli.tolerance:
            failures.append((mechanism.name, worst, best))
            print(f"  [FAIL] worst error {worst:.3e} rad exceeds {args_cli.tolerance:.1e}; best fit is '{best[0]}'.")
        else:
            print(f"  [PASS] worst error {worst:.3e} rad is within {args_cli.tolerance:.1e}.")

        masked = np.where(free, error.max(axis=1), -1.0)
        rows = range(num_envs) if args_cli.dump_grid else [int(masked.argmax())]
        print(
            f"  {'usd motor A':>12}{'usd motor B':>12}{'roll':>10}{'pitch':>10}"
            f"{'ik slot 1':>12}{'ik slot 2':>12}{'err 1':>11}{'err 2':>11}  state"
        )
        for row in rows:
            print(
                f"  {measured_motor[row, 0]:>12.5f}{measured_motor[row, 1]:>12.5f}{roll[row]:>10.5f}"
                f"{pitch[row]:>10.5f}{from_joints[row, 0]:>12.5f}{from_joints[row, 1]:>12.5f}"
                f"{from_joints[row, 0] - measured_motor[row, 0]:>11.2e}"
                f"{from_joints[row, 1] - measured_motor[row, 1]:>11.2e}"
                f"  {'free' if free[row] else 'jammed'}"
            )

    print()
    if failures:
        print(f"[FAIL] {len(failures)} of {len(MECHANISMS)} mechanisms disagree:")
        for name, worst, best in failures:
            print(f"       {name}: worst {worst:.3e} rad, best-fitting relation '{best[0]}' at {best[1]:.3e} rad")
        return 1
    print(f"[PASS] All {len(MECHANISMS)} mechanisms agree to within {args_cli.tolerance:.1e} rad.")
    print("       The alias table is a rename: same motor, same direction, same zero.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # simulation_app.close() hangs indefinitely on a scene built straight from InteractiveScene.
    # Everything is reported by now, so leave teardown to the OS.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
