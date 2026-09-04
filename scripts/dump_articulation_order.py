"""Print the joint and body ordering that Isaac Sim assigns to a robot articulation.

The ordering depends on IsaacLab's fixed-joint merging and PhysX's link traversal, so it cannot
be derived from the URDF. ``npy_to_npz.py`` keeps it as a static constant; run this script and
paste the output there whenever the description or IsaacLab version changes.

The per-joint drive gains are printed alongside, because for the closed-loop description they are
part of what has to be checked: the joints the achilles linkages move must reach PhysX with zero
stiffness and damping, otherwise their drive fights the loop constraint.

.. code-block:: bash

    python scripts/dump_articulation_order.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the articulation joint/body ordering.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.hu_l04_parallel import HU_L04_PARALLEL_CFG


@configclass
class InspectSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = HU_L04_PARALLEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def format_list(name: str, values: list[str]) -> str:
    lines = [f"{name} = ("]
    lines += [f'    "{v}",' for v in values]
    lines.append(")")
    return "\n".join(lines)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(InspectSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    robot_name = "hu_l04_parallel"
    prefix = robot_name.upper()

    print(f"\n{'=' * 78}")
    print(f"{robot_name}: {len(robot.joint_names)} joints, {len(robot.body_names)} bodies")
    print(f"{'=' * 78}\n")
    print(format_list(f"{prefix}_JOINT_ORDER", robot.joint_names))
    print()
    print(format_list(f"{prefix}_BODY_ORDER", robot.body_names))
    print()

    group_of = {name: group for group, act in robot.actuators.items() for name in act.joint_names}
    stiffness = robot.data.joint_stiffness[0].cpu().tolist()
    damping = robot.data.joint_damping[0].cpu().tolist()
    print(f"{'=' * 78}")
    print(f"{'index':<7}{'joint':<34}{'actuator':<14}{'stiffness':>11}{'damping':>11}")
    print(f"{'=' * 78}")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:<7}{name:<34}{group_of.get(name, '<none>'):<14}{stiffness[i]:>11.4f}{damping[i]:>11.4f}")
    print()


if __name__ == "__main__":
    main()
    # simulation_app.close() hangs indefinitely when the scene was built directly from
    # InteractiveScene rather than through a gym env. Everything is already printed, so leave
    # process teardown to the OS.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
