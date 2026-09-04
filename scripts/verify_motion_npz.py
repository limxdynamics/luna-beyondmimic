"""Cross-check a motion.npz against Isaac Sim forward kinematics.

``npy_to_npz.py`` keeps the articulation ordering as a static constant, and a stale constant is
not detectable from the npz alone. This script closes that gap: it writes the npz root and joint
state into the articulation, lets Isaac Sim recompute the body poses, and compares them with the
poses the retargeter stored. A wrong joint order, body order, quaternion convention or root pose
convention all surface as a large per-body discrepancy.

Run it after regenerating the ordering constants, after changing the description, and after
upgrading IsaacLab.

On the position tolerance: the default is loose enough to accept a motion retargeted against a
neighbouring revision of the description. The bundled ``motions/HU_L04_01_WAKAWAKA_an03.npy`` is
one - its link lengths differ from this description's by up to 2.5 mm, which accumulates to a few
millimetres along the arm and leg chains and would fail a 1 mm gate for a reason that has nothing
to do with what this script is looking for. An ordering or convention error is off by orders of
magnitude more than that, so the looser default still catches it. Pass ``--pos_tol 1e-3`` for a
motion you retargeted against this exact description.

.. code-block:: bash

    python scripts/verify_motion_npz.py --motion_file /tmp/motion.npz --robot hu_l04
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cross-check a motion.npz against Isaac Sim forward kinematics.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion.npz.")
parser.add_argument(
    "--robot",
    type=str,
    default="hu_l04_parallel",
    choices=["hu_l04_parallel"],
    help="Supported articulation (fixed to the Luna closed-loop model).",
)
parser.add_argument("--num_samples", type=int, default=8, help="How many frames to sample across the motion.")
parser.add_argument(
    "--pos_tol",
    type=float,
    default=5e-3,
    help=(
        "Per-body position tolerance in metres. The default absorbs the link-length difference"
        " between description revisions; use 1e-3 for a motion retargeted against this one."
    ),
)
parser.add_argument("--rot_tol", type=float, default=1e-2, help="Per-body rotation tolerance in radians.")
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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.hu_l04_parallel import HU_L04_PARALLEL_CFG

ROBOT_CFGS = {"hu_l04_parallel": HU_L04_PARALLEL_CFG}


@configclass
class VerifySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(VerifySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    device = sim.device
    motion = np.load(args_cli.motion_file)

    num_frames = motion["joint_pos"].shape[0]
    if motion["body_pos_w"].shape[1] != len(robot.body_names):
        print(
            f"[FAIL] npz has {motion['body_pos_w'].shape[1]} bodies but the articulation has {len(robot.body_names)}."
        )
        return 1

    frames = np.linspace(0, num_frames - 1, args_cli.num_samples, dtype=int)
    pos_err = np.zeros((len(frames), len(robot.body_names)))
    rot_err = np.zeros((len(frames), len(robot.body_names)))

    for i, frame in enumerate(frames):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(motion["body_pos_w"][frame, 0], device=device)
        root_state[:, 3:7] = torch.tensor(motion["body_quat_w"][frame, 0], device=device)
        root_state[:, 7:10] = torch.tensor(motion["body_lin_vel_w"][frame, 0], device=device)
        root_state[:, 10:] = torch.tensor(motion["body_ang_vel_w"][frame, 0], device=device)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            torch.tensor(motion["joint_pos"][frame], device=device).unsqueeze(0),
            torch.tensor(motion["joint_vel"][frame], device=device).unsqueeze(0),
        )
        scene.write_data_to_sim()
        sim.render()  # forward kinematics only, no physics stepping
        scene.update(sim_cfg.dt)

        sim_pos = robot.data.body_pos_w[0].cpu().numpy()
        sim_quat = robot.data.body_quat_w[0].cpu().numpy()
        pos_err[i] = np.linalg.norm(sim_pos - motion["body_pos_w"][frame], axis=-1)
        # Quaternion double cover: q and -q are the same rotation.
        dot = np.abs((sim_quat * motion["body_quat_w"][frame]).sum(-1)).clip(max=1.0)
        rot_err[i] = 2.0 * np.arccos(dot)

    # Slots the converter zero-filled because the retargeter does not produce that body. They
    # are never read at training time, so comparing them against forward kinematics is
    # meaningless - the sim reports where the link actually is, the npz holds a placeholder.
    padded = {i for i in range(len(robot.body_names)) if np.allclose(motion["body_pos_w"][:, i], 0.0, atol=1e-9)}

    print(f"\n{'=' * 72}")
    print(f"{'body':<28}{'pos err (m)':>14}{'rot err (rad)':>16}{'':>6}")
    print(f"{'=' * 72}")
    failed = []
    for i, name in enumerate(robot.body_names):
        p, r = pos_err[:, i].max(), rot_err[:, i].max()
        if i in padded:
            print(f"{name:<28}{'-':>14}{'-':>16}{'  (zero-filled)':>6}")
            continue
        bad = p > args_cli.pos_tol or r > args_cli.rot_tol
        failed.append(name) if bad else None
        print(f"{name:<28}{p:>14.5f}{r:>16.5f}{'  <-- FAIL' if bad else '':>6}")
    checked = [i for i in range(len(robot.body_names)) if i not in padded]
    print(f"{'=' * 72}")
    print(f"{'max over checked bodies':<28}{pos_err[:, checked].max():>14.5f}{rot_err[:, checked].max():>16.5f}")

    if failed:
        print(f"\n[FAIL] {len(failed)} bodies exceed tolerance: {failed}")
        print("Most likely cause: the ordering constants in npy_to_npz.py no longer match the")
        print("articulation. Re-run scripts/dump_articulation_order.py and update them.")
        return 1

    print(f"\n[PASS] {len(checked)} bodies within tolerance over {len(frames)} sampled frames")
    print(f"       ({len(padded)} zero-filled slots skipped).")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # simulation_app.close() hangs indefinitely when the scene was built directly from
    # InteractiveScene rather than through a gym env. The verdict is already printed, and the
    # exit code has to stay usable from scripts, so leave process teardown to the OS.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
