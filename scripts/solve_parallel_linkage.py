"""Solve the achilles linkage DoF that a serial HU_L04 motion does not carry.

The retargeter produces the 27 joints of the serial description. The closed-loop articulation has
51: the same 27 names, plus the 6 achilles motors that move the ankles and waist, plus the 18 DoF
of the 6 in-tree spherical rod joints. Those 24 extra values are not free - the loop closures fix
them once the ankle and waist angles are chosen - but they cannot be written down analytically
without the linkage geometry, and ``MotionCommand`` writes the whole joint vector at every reset.
Leaving them at zero would start each episode with the closures violated by as much as the ankle
travel itself, which PhysX would then resolve as an impulse.

This script recovers them by letting PhysX solve the mechanism. Each frame becomes one
environment; the achilles motors are servoed until the passive joints land on the values the
motion asks for, and the settled configuration is read back. Gravity is off, the root is pinned
and every joint outside the linkages is left at zero, because each loop is closed entirely within
one parent link and so does not depend on the rest of the pose.

The output feeds ``npy_to_npz.py --robot hu_l04_parallel --linkage_file ...``.

.. code-block:: bash

    python scripts/solve_parallel_linkage.py \
        --input_file motion_npy/motion.npy --output_file /tmp/linkage.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve the HU_L04 achilles linkage DoF for a motion.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the serial .npy motion.")
parser.add_argument("--output_file", type=str, required=True, help="Where to write the linkage .npz.")
parser.add_argument("--chunk_size", type=int, default=1024, help="Frames solved concurrently, one per environment.")
parser.add_argument("--max_frames", type=int, default=None, help="Stop after this many frames. For smoke tests only.")
parser.add_argument("--settle_steps", type=int, default=120, help="Simulation steps per servo iteration.")
parser.add_argument("--iterations", type=int, default=6, help="Servo iterations per chunk.")
parser.add_argument("--tolerance", type=float, default=1e-3, help="Reported pass threshold on the passive joints, rad.")
parser.add_argument(
    "--max_unreachable_error",
    type=float,
    default=0.1,
    help="How far outside the linkage workspace a frame may sit before the run is rejected, rad.",
)
parser.add_argument(
    "--allow_unreachable_motion",
    action="store_true",
    help="Report residuals but do not reject the output when they exceed --max_unreachable_error.",
)
parser.add_argument("--report_file", type=str, default=None, help="Optional linkage quality report JSON output.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import sys
import torch

from linkage_quality import build_linkage_report, write_linkage_report

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from whole_body_tracking.robots.hu_l04_parallel import HU_L04_PARALLEL_CFG

# Each linkage is one closed loop: two motors driving two passive axes through two rods. The
# names are what ties the solve to the articulation; nothing else here is robot specific.
LINKAGES = (
    (
        "left_ankle",
        ("left_A_achilles_joint", "left_B_achilles_joint"),
        ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
    ),
    (
        "right_ankle",
        ("right_A_achilles_joint", "right_B_achilles_joint"),
        ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    ),
    ("waist", ("waist_A_joint", "waist_B_joint"), ("waist_pitch_joint", "waist_roll_joint")),
)

# Gains used for the solve only. This is a kinematics problem, so the motors are made far stiffer
# than the hardware values in the robot config to reach steady state in few steps, and the
# undriven DoF are given a little damping to bleed off the energy the corrections inject. Damping
# vanishes at steady state, so neither changes the configuration that is read back.
SOLVE_MOTOR_STIFFNESS = 4000.0
SOLVE_MOTOR_DAMPING = 200.0
SOLVE_MOTOR_EFFORT = 1.0e4
SOLVE_PASSIVE_DAMPING = 0.2

# Amplitude of the calibration sweep, in radians. Large enough that the response is well above
# solver noise, small enough to stay inside every joint limit involved.
CALIBRATION_ANGLE = 0.15


def build_scene(num_envs: int) -> tuple[SimulationContext, Articulation]:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005, gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)

    cfg = HU_L04_PARALLEL_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Robot"
    cfg.spawn.activate_contact_sensors = False
    # Self collisions would let a mesh, rather than the linkage, decide where a joint settles.
    cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.spawn.articulation_props.fix_root_link = True

    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class SolveSceneCfg(InteractiveSceneCfg):
        robot = cfg

    scene = InteractiveScene(SolveSceneCfg(num_envs=num_envs, env_spacing=3.0))
    sim.reset()
    return sim, scene["robot"]


def apply_solve_gains(robot: Articulation, motor_ids: list[int], undriven_ids: list[int]) -> None:
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    effort = robot.data.joint_effort_limits.clone()

    stiffness[:, motor_ids] = SOLVE_MOTOR_STIFFNESS
    damping[:, motor_ids] = SOLVE_MOTOR_DAMPING
    effort[:, motor_ids] = SOLVE_MOTOR_EFFORT
    stiffness[:, undriven_ids] = 0.0
    damping[:, undriven_ids] = SOLVE_PASSIVE_DAMPING

    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.write_joint_effort_limit_to_sim(effort)


def settle(sim: SimulationContext, robot: Articulation, targets: torch.Tensor, steps: int) -> None:
    robot.set_joint_position_target(targets)
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())


def calibrate(sim: SimulationContext, robot: Articulation, index: dict[str, int]) -> dict[str, np.ndarray]:
    """Measure d(passive)/d(motor) for every linkage.

    Each loop is independent, so all three are swept at once. Four probes give a central
    difference for both motors, which is enough for a 2x2 Jacobian; the map is close enough to
    linear that this doubles as the initial guess for the servo below.
    """
    probes = [(CALIBRATION_ANGLE, 0.0), (-CALIBRATION_ANGLE, 0.0), (0.0, CALIBRATION_ANGLE), (0.0, -CALIBRATION_ANGLE)]
    response = {name: np.zeros((len(probes), 2)) for name, _, _ in LINKAGES}

    for probe, (a, b) in enumerate(probes):
        targets = torch.zeros_like(robot.data.joint_pos)
        for _, motors, _ in LINKAGES:
            targets[:, index[motors[0]]] = a
            targets[:, index[motors[1]]] = b
        robot.write_joint_state_to_sim(targets.clone(), torch.zeros_like(targets))
        settle(sim, robot, targets, args_cli.settle_steps)
        for name, _, passives in LINKAGES:
            response[name][probe] = [robot.data.joint_pos[0, index[p]].item() for p in passives]

    jacobians = {}
    for name, _, passives in LINKAGES:
        r = response[name]
        d_da = (r[0] - r[1]) / (2.0 * CALIBRATION_ANGLE)
        d_db = (r[2] - r[3]) / (2.0 * CALIBRATION_ANGLE)
        jacobians[name] = np.stack([d_da, d_db], axis=1)  # rows: passive, cols: motor
        print(f"[INFO] {name}: d({passives[0]},{passives[1]})/d(A,B) =")
        print(f"       [[{jacobians[name][0, 0]:+.4f} {jacobians[name][0, 1]:+.4f}]")
        print(f"        [{jacobians[name][1, 0]:+.4f} {jacobians[name][1, 1]:+.4f}]]")
    return jacobians


def solve_chunk(
    sim: SimulationContext,
    robot: Articulation,
    index: dict[str, int],
    inverses: dict[str, torch.Tensor],
    reference: torch.Tensor,
    num_active: int,
    motor_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Servo the motors until the passive joints match ``reference``.

    Returns the settled joint state and the per-frame residual on the passive joints. A residual
    that stops shrinking is not necessarily a solver failure: the coupled workspace of a linkage
    is smaller than the box of independent limits the retargeter worked against, so some frames
    simply have no solution and the motors end up against their stops.
    """
    lower = robot.data.joint_pos_limits[..., 0]
    upper = robot.data.joint_pos_limits[..., 1]

    targets = torch.zeros_like(robot.data.joint_pos)
    # Seed the motors from the linear model, then correct: the map is only linear to a percent or
    # so, and the ankles in these motions run out to 0.7 rad.
    for name, motors, passives in LINKAGES:
        goal = torch.stack([reference[:, index[p]] for p in passives], dim=1)
        motor = goal @ inverses[name].T
        targets[:, index[motors[0]]] = motor[:, 0]
        targets[:, index[motors[1]]] = motor[:, 1]

    residual = torch.zeros(robot.num_instances, device=reference.device)
    for iteration in range(args_cli.iterations):
        # Without this the correction winds up far past the stop on unreachable frames, which
        # changes nothing physically but hides how far out of reach they were.
        targets[:, motor_ids] = targets[:, motor_ids].clamp(lower[:, motor_ids], upper[:, motor_ids])
        robot.write_joint_state_to_sim(targets.clone(), torch.zeros_like(targets))
        settle(sim, robot, targets, args_cli.settle_steps)

        residual = torch.zeros(robot.num_instances, device=reference.device)
        for name, motors, passives in LINKAGES:
            goal = torch.stack([reference[:, index[p]] for p in passives], dim=1)
            actual = torch.stack([robot.data.joint_pos[:, index[p]] for p in passives], dim=1)
            error = goal - actual
            residual = torch.maximum(residual, error.abs().amax(dim=1))
            correction = error @ inverses[name].T
            targets[:, index[motors[0]]] += correction[:, 0]
            targets[:, index[motors[1]]] += correction[:, 1]
        worst = residual[:num_active].max().item()
        print(f"[INFO]   iteration {iteration}: worst passive-joint error {worst:.3e} rad")
        if worst < args_cli.tolerance:
            break

    return robot.data.joint_pos.clone(), residual


def main() -> int:
    data = np.load(args_cli.input_file, allow_pickle=True).item()
    dof_names = list(data["dof_names"])
    dof_pos = np.asarray(data["dof_pos_vel"], dtype=np.float64)[:, :, 0]
    if args_cli.max_frames is not None:
        dof_pos = dof_pos[: args_cli.max_frames]
    num_frames = dof_pos.shape[0]
    print(f"[INFO] Loaded {args_cli.input_file}: {num_frames} frames, {len(dof_names)} joints.")

    num_envs = min(num_frames, args_cli.chunk_size)
    sim, robot = build_scene(num_envs)
    index = {name: i for i, name in enumerate(robot.joint_names)}

    missing = [n for n in dof_names if n not in index]
    if missing:
        print(f"[FAIL] The motion names joints the articulation does not have: {missing}")
        return 1

    motor_names = [m for _, motors, _ in LINKAGES for m in motors]
    passive_names = [p for _, _, passives in LINKAGES for p in passives]
    linkage_names = [n for n in robot.joint_names if ":" in n]
    solved_names = motor_names + linkage_names
    if len(solved_names) != len(robot.joint_names) - len(dof_names):
        print(f"[FAIL] Expected {len(robot.joint_names) - len(dof_names)} unsolved DoF, found {len(solved_names)}.")
        return 1

    motor_ids = [index[n] for n in motor_names]
    undriven_ids = [index[n] for n in passive_names + linkage_names]
    apply_solve_gains(robot, motor_ids, undriven_ids)

    jacobians = calibrate(sim, robot, index)
    inverses = {}
    for name, jacobian in jacobians.items():
        if abs(np.linalg.det(jacobian)) < 1e-6:
            print(f"[FAIL] {name} transmission is singular: {jacobian}")
            return 1
        inverses[name] = torch.tensor(np.linalg.inv(jacobian), dtype=torch.float32, device=sim.device)

    solved = np.zeros((num_frames, len(solved_names)), dtype=np.float64)
    solved_ids = [index[n] for n in solved_names]
    residuals = np.zeros(num_frames)
    for start in range(0, num_frames, num_envs):
        stop = min(start + num_envs, num_frames)
        print(f"[INFO] Frames {start}-{stop - 1}:")
        reference = torch.zeros(num_envs, len(robot.joint_names), device=sim.device)
        for i, name in enumerate(dof_names):
            reference[: stop - start, index[name]] = torch.tensor(dof_pos[start:stop, i], dtype=torch.float32)
        state, residual = solve_chunk(sim, robot, index, inverses, reference, stop - start, motor_ids)
        solved[start:stop] = state[: stop - start, solved_ids].cpu().numpy()
        residuals[start:stop] = residual[: stop - start].cpu().numpy()

    np.savez(args_cli.output_file, joint_names=np.array(solved_names), joint_pos=solved.astype(np.float32))
    print(f"\n[INFO] Wrote {args_cli.output_file}: {solved.shape[0]} frames x {solved.shape[1]} solved DoF.")
    print("[INFO] Solved DoF ranges (rad):")
    for i, name in enumerate(solved_names):
        print(f"       {name:<32}{solved[:, i].min():+.4f} .. {solved[:, i].max():+.4f}")

    unreachable = residuals > args_cli.tolerance
    report = build_linkage_report(
        residuals,
        residual_tolerance=args_cli.tolerance,
        max_unreachable_error=args_cli.max_unreachable_error,
        strict_check=not args_cli.allow_unreachable_motion,
    )
    if args_cli.report_file is not None:
        report_path = write_linkage_report(args_cli.report_file, report)
        print(f"\n[INFO] Wrote linkage quality report: {report_path}")
    if unreachable.any():
        print(
            f"\n[WARN] {unreachable.sum()} of {num_frames} frames ({100 * unreachable.mean():.2f}%) ask for an "
            f"ankle or waist pose the linkage cannot reach; worst miss {residuals.max():.4f} rad at frame "
            f"{int(residuals.argmax())}. The motors are at their stops there and the motion is followed as "
            "closely as the mechanism allows. This is a property of the reference, not of the solve: the "
            "retargeter treats these joints as independent, but the achilles pair couples them into a "
            "smaller workspace."
        )
    if residuals.max() > args_cli.max_unreachable_error:
        message = f"Worst residual {residuals.max():.4f} rad exceeds {args_cli.max_unreachable_error:.4f}."
        if not args_cli.allow_unreachable_motion:
            print(f"\n[FAIL] {message}")
            return 1
        print(f"\n[WARN] {message} Continuing because the caller selected warning-only motion quality.")
    else:
        print(f"\n[PASS] Worst passive-joint residual {residuals.max():.4e} rad over {num_frames} frames.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # simulation_app.close() hangs indefinitely on a scene built straight from InteractiveScene.
    # Everything is written and reported by now, so leave teardown to the OS.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
