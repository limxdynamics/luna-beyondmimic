# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyOnlyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxPolicyOnlyExporter(_OnnxPolicyExporter):
    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def resolve_export_joint_ids(env: ManagerBasedRLEnv) -> list[int]:
    """Articulation indices of the joints the exported metadata describes, in metadata order.

    The action term is the single source of truth for the restricted case: the deployed controller
    turns an action into a target for joint ``i`` of its own permutation, so anything it indexes
    with that permutation - gains, default position, action scale, and the reference and
    measurement vectors - has to be the same joints in the same order.
    """
    robot = env.scene["robot"]
    every_joint = list(range(len(robot.data.joint_names)))
    if not getattr(env.cfg, "export_actuated_joints_only", False):
        return every_joint

    # Both of these are slice(None) rather than an index list when they cover the articulation,
    # which is the shape they take on a serial robot.
    action_ids = env.action_manager.get_term("joint_pos")._joint_ids
    joint_ids = every_joint if isinstance(action_ids, slice) else list(action_ids)
    command_ids = env.command_manager.get_term("motion").observed_joint_ids
    command_ids = every_joint if isinstance(command_ids, slice) else command_ids.cpu().tolist()
    if command_ids != joint_ids:
        raise ValueError(
            "The motion command publishes a different joint set than the action term drives, so"
            " one exported metadata permutation cannot describe both."
        )

    # The failure this guards against is silent and unrecoverable on hardware: one permutation
    # applied to two differently ordered vectors sends a joint the target meant for another.
    manager = env.observation_manager
    for name, term_cfg in zip(manager.active_terms["policy"], manager._group_obs_term_cfgs["policy"]):
        if name not in ("joint_pos", "joint_vel"):
            continue
        asset_cfg = term_cfg.params.get("asset_cfg")
        observed = every_joint if asset_cfg is None or isinstance(asset_cfg.joint_ids, slice) else asset_cfg.joint_ids
        if list(observed) != joint_ids:
            raise ValueError(
                f"The policy's '{name}' observation covers different joints than the action term"
                " drives, so one exported metadata permutation cannot describe both."
            )
    return joint_ids


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    robot = env.scene["robot"]
    joint_ids = resolve_export_joint_ids(env)
    # Applied last and only to the names. A robot whose hardware slot names no longer describe
    # what the slot holds still has to be addressed by those names, and the simulation must not
    # hear about it - see HU_L04_PARALLEL_DEPLOY_JOINT_NAME_ALIASES.
    aliases = getattr(env.cfg, "export_joint_name_aliases", {})
    joint_names = [aliases.get(robot.data.joint_names[i], robot.data.joint_names[i]) for i in joint_ids]

    metadata = {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": robot.data.joint_stiffness[0, joint_ids].cpu().tolist(),
        "joint_damping": robot.data.joint_damping[0, joint_ids].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos_nominal[joint_ids].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
