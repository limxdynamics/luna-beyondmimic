"""Validate the target policy-only DanceDeploy ONNX contract offline.

The current deployment controller still implements the legacy two-input/seven-output contract.
This checker describes the target ``obs -> actions`` interface that the deployment-side change must
implement. A pass proves the policy and metadata are internally consistent; it does not mean an
unmodified controller can load the model or that the policy is safe on hardware.

.. code-block:: bash

    python scripts/check_deploy_onnx.py policy.onnx \
        --hardware_config /path/to/HU_L04_01/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

import onnx

# The metadata keys the deployed controller requires.
REQUIRED_KEYS = (
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "action_scale",
    "anchor_body_name",
    "body_names",
    "observation_names",
    "observation_history_lengths",
)

PER_JOINT_KEYS = ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale")

# Observation term widths, in policy order. "dof" and "bodies" are resolved from the metadata.
OBS_TERM_DIM = {
    "command": ("dof", 2),
    "motion_anchor_pos_b": ("const", 3),
    "motion_anchor_ori_b": ("const", 6),
    "base_lin_vel": ("const", 3),
    "base_ang_vel": ("const", 3),
    "projected_gravity": ("const", 3),
    "joint_pos": ("dof", 1),
    "joint_vel": ("dof", 1),
    "actions": ("dof", 1),
}

# Policy-only contract. Reference trajectories are loaded from a separate DanceLibrary-format TXT.
EXPECTED_INPUTS = ("obs",)
EXPECTED_OUTPUTS = ("actions",)

# Terms the controller can only fill from a source it does not have on this robot.
TERMS_NEEDING_WORLD_STATE = ("motion_anchor_pos_b", "base_lin_vel")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}{(': ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(what)
        return ok

    def warn(self, what: str) -> None:
        print(f"  [WARN] {what}")
        self.warnings.append(what)


def element_count(value_info) -> int:
    count = 1
    for dim in value_info.type.tensor_type.shape.dim:
        count *= dim.dim_value if dim.dim_value > 0 else 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline DanceDeploy acceptance check.")
    parser.add_argument("policy", type=str, help="Path to the exported policy.onnx.")
    parser.add_argument(
        "--hardware_config",
        type=str,
        required=True,
        help="The robot's config.yaml, whose joints_name the metadata has to match.",
    )
    args = parser.parse_args()

    hardware_config = args.hardware_config

    if not os.path.isfile(args.policy):
        print(f"[FAIL] No such file: {args.policy}")
        return 1
    if not os.path.isfile(hardware_config):
        print(f"[FAIL] No such hardware config: {hardware_config}")
        return 1
    model = onnx.load(args.policy)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    report = Report()

    print(f"\npolicy   {args.policy}")
    print(f"run_path {metadata.get('run_path', '<absent>')}")

    print("\n-- metadata keys --")
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if not report.check(not missing, "all required metadata keys present", f"missing {missing}" if missing else ""):
        return 1

    joint_names = metadata["joint_names"].split(",")
    observation_names = metadata["observation_names"].split(",")
    body_names = metadata["body_names"].split(",")
    numeric = {key: [float(x) for x in metadata[key].split(",")] for key in PER_JOINT_KEYS}
    history = [float(x) for x in metadata["observation_history_lengths"].split(",")]
    dof = len(joint_names)

    print("\n-- per-joint metadata --")
    report.check(dof > 0, "joint_names is not empty", f"{dof} joints")
    for key in PER_JOINT_KEYS:
        report.check(len(numeric[key]) == dof, f"{key} has one entry per joint", f"{len(numeric[key])} vs {dof}")
    report.check(len(body_names) > 0, "body_names is not empty", f"{len(body_names)} bodies")

    print("\n-- observation history --")
    report.check(
        len(history) == len(observation_names),
        "observation_history_lengths has one entry per term",
        f"{len(history)} vs {len(observation_names)}",
    )
    stacked = [n for n, h in zip(observation_names, history) if abs(h - 1.0) > 1e-6]
    report.check(not stacked, "no observation term is stacked", f"stacked: {stacked}" if stacked else "all 1.0")

    print("\n-- observation layout --")
    unknown = [name for name in observation_names if name not in OBS_TERM_DIM]
    if not report.check(not unknown, "every observation term is one the controller knows", f"unknown {unknown}"):
        return 1
    print(f"  {'offset':>8}  {'width':>6}  term")
    offset = 0
    for name in observation_names:
        kind, factor = OBS_TERM_DIM[name]
        width = factor * dof if kind == "dof" else factor
        print(f"  {offset:>8}  {width:>6}  {name}")
        offset += width
    print(f"  {offset:>8}  {'':>6}  total")

    print("\n-- ONNX signature --")
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    report.check(tuple(input_names) == EXPECTED_INPUTS, "input is [obs]", str(input_names))
    report.check(tuple(output_names) == EXPECTED_OUTPUTS, "output is [actions]", str(output_names))
    outputs = {o.name: element_count(o) for o in model.graph.output}
    obs_dim = element_count(model.graph.input[0])
    report.check(obs_dim == offset, "obs input width equals the sum of the declared terms", f"{obs_dim} vs {offset}")
    if "actions" in outputs:
        report.check(outputs["actions"] == dof, f"output 'actions' is {dof} wide", f"{outputs['actions']}")

    print("\n-- joint permutation --")
    with open(hardware_config) as handle:
        hardware_joints = list(yaml.safe_load(handle)["joints_name"])
    print(f"  reference: {hardware_config}")
    report.check(
        dof == len(hardware_joints),
        "joint count matches the robot's config.yaml",
        f"{dof} vs {len(hardware_joints)}",
    )
    only_model = sorted(set(joint_names) - set(hardware_joints))
    only_hardware = sorted(set(hardware_joints) - set(joint_names))
    report.check(
        not only_model and not only_hardware,
        "joint_names is a permutation of the robot's joints",
        f"only in model {only_model}; only on robot {only_hardware}" if (only_model or only_hardware) else "",
    )
    duplicates = sorted({n for n in joint_names if joint_names.count(n) > 1})
    report.check(not duplicates, "joint_names has no duplicates", str(duplicates) if duplicates else "")
    permutation = [hardware_joints.index(n) for n in joint_names] if not (only_model or only_hardware) else []
    if permutation:
        print(f"  model index -> hardware slot: {permutation}")

    print("\n-- base state --")
    needed = [name for name in TERMS_NEEDING_WORLD_STATE if name in observation_names]
    if needed:
        report.warn(
            f"{needed} need a world-frame base state, so base_state_source.mode must be"
            " 'ground_truth'; 'imu_only' will be refused."
        )
    else:
        print("  [PASS] no term needs a world-frame base state; base_state_source.mode 'imu_only' is enough")

    print("\n-- summary --")
    print(f"  dof {dof}, obs {obs_dim}, bodies {len(body_names)}, anchor {metadata['anchor_body_name']}")
    if report.failures:
        print(f"\n[FAIL] {len(report.failures)} check(s) failed:")
        for failure in report.failures:
            print(f"       {failure}")
        return 1
    print(f"\n[PASS] Policy-only DanceDeploy contract is valid. {len(report.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
