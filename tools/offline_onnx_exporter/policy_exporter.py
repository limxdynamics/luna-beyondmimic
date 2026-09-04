"""Export an RSL-RL checkpoint to the policy-only DanceDeploy ONNX model.

The checkpoint carries weights and nothing else: not the activation, not the observation layout,
not the gains the controller needs. Those come from ``deploy_config``, which describes the one
configuration this exporter supports, so the checkpoint is only ever checked against it.
"""

from __future__ import annotations

import numpy as np
import os
import re
import stat
import tempfile
import torch
from pathlib import Path
from torch import nn
from typing import Any

import onnx

try:
    from . import deploy_config
except ImportError:
    import deploy_config

ACTIVATION_MODULES: dict[str, type[nn.Module]] = {
    "crelu": nn.CELU,
    "elu": nn.ELU,
    "identity": nn.Identity,
    "lrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# How far the ONNX graph may drift from the module it was traced from before the export is
# refused. Both run in float32 on the same weights, so anything larger is a broken graph rather
# than accumulated arithmetic error.
GRAPH_AGREEMENT_TOLERANCE = 1.0e-5


class FrozenNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, epsilon: float):
        super().__init__()
        self.register_buffer("_mean", mean.detach().clone())
        self.register_buffer("_std", std.detach().clone())
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean) / (self._std + self.epsilon)


class PolicyExportModule(nn.Module):
    def __init__(self, actor: nn.Module, normalizer: nn.Module):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(observations))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix != ".pt":
        raise ValueError(f"Checkpoint {path} must be a .pt file.")
    if not path.is_file():
        raise ValueError(f"No such checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not an RSL-RL checkpoint with model_state_dict.")
    return checkpoint


def _reconstruct_actor(model_state: dict[str, torch.Tensor]) -> nn.Sequential:
    layer_pattern = re.compile(r"^actor\.(\d+)\.weight$")
    layer_ids = sorted(int(match.group(1)) for key in model_state if (match := layer_pattern.match(key)))
    if not layer_ids:
        raise ValueError("Checkpoint model_state_dict contains no actor linear layers.")
    expected_layer_ids = list(range(0, 2 * len(layer_ids), 2))
    if layer_ids != expected_layer_ids:
        raise ValueError(
            f"Checkpoint actor layers {layer_ids} are not a standard RSL-RL MLP topology {expected_layer_ids}."
        )
    expected_actor_keys = {
        key for layer_id in layer_ids for key in (f"actor.{layer_id}.weight", f"actor.{layer_id}.bias")
    }
    actual_actor_keys = {key for key in model_state if key.startswith("actor.")}
    if actual_actor_keys != expected_actor_keys:
        unexpected = sorted(actual_actor_keys - expected_actor_keys)
        missing = sorted(expected_actor_keys - actual_actor_keys)
        raise ValueError(
            "Checkpoint actor is not a standard RSL-RL MLP: "
            f"unexpected actor parameters {unexpected}, missing {missing}."
        )

    modules: list[nn.Module] = []
    previous_output = None
    for position, layer_id in enumerate(layer_ids):
        weight = model_state[f"actor.{layer_id}.weight"]
        bias = model_state.get(f"actor.{layer_id}.bias")
        if weight.ndim != 2 or bias is None or bias.shape != (weight.shape[0],):
            raise ValueError(f"Checkpoint actor layer {layer_id} is not a supported Linear layer.")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError(f"Checkpoint actor layer {layer_id} contains non-finite values.")
        if previous_output is not None and weight.shape[1] != previous_output:
            raise ValueError(f"Checkpoint actor layer {layer_id} is not connected to the previous layer.")
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        modules.append(linear)
        previous_output = weight.shape[0]
        if position != len(layer_ids) - 1:
            modules.append(ACTIVATION_MODULES[deploy_config.POLICY_ACTIVATION]())

    actor = nn.Sequential(*modules)
    if actor[0].in_features != deploy_config.OBSERVATION_DIM:
        raise ValueError(
            f"Checkpoint actor expects {actor[0].in_features} observations; this exporter only supports the "
            f"{deploy_config.OBSERVATION_DIM}-wide HU_L04 observation ending in projected_gravity."
        )
    if actor[-1].out_features != deploy_config.ACTION_DIM:
        raise ValueError(
            f"Checkpoint actor produces {actor[-1].out_features} actions, expected {deploy_config.ACTION_DIM}."
        )
    actor.eval()
    return actor


def _build_normalizer(checkpoint: dict[str, Any]) -> nn.Module:
    state = checkpoint.get("obs_norm_state_dict")
    if not isinstance(state, dict) or "_mean" not in state or "_std" not in state:
        raise ValueError("Checkpoint has no obs_norm_state_dict; the deployed policy is trained with normalization.")
    mean = state["_mean"]
    std = state["_std"]
    expected_shape = (1, deploy_config.OBSERVATION_DIM)
    if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
        raise ValueError(
            f"Checkpoint normalizer shape is {tuple(mean.shape)}/{tuple(std.shape)}, expected {expected_shape}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Checkpoint normalizer mean and std must be finite.")
    if (std < 0).any():
        raise ValueError("Checkpoint normalizer std must be non-negative.")
    if (std + deploy_config.NORMALIZER_EPSILON <= 0).any():
        raise ValueError("Checkpoint normalizer requires std + epsilon to be strictly positive.")
    normalizer = FrozenNormalizer(mean, std, deploy_config.NORMALIZER_EPSILON)
    normalizer.eval()
    return normalizer


def build_export_module(checkpoint_path: str | Path) -> PolicyExportModule:
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    actor = _reconstruct_actor(checkpoint["model_state_dict"])
    normalizer = _build_normalizer(checkpoint)
    module = PolicyExportModule(actor, normalizer)
    module.eval()
    return module


def _metadata_value(value: Any) -> str:
    if not isinstance(value, list):
        return str(value)
    return ",".join(f"{item:.3f}" if isinstance(item, (int, float)) else str(item) for item in value)


def _tensor_shape(value_info) -> list[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def _validate_model(model: onnx.ModelProto, module: PolicyExportModule, metadata: dict[str, Any]) -> None:
    """Check the written graph against the contract before it is allowed to become the output."""
    onnx.checker.check_model(model)

    opsets = {opset.domain: opset.version for opset in model.opset_import}
    if opsets.get("") != deploy_config.ONNX_OPSET_VERSION:
        raise ValueError(f"Exported model uses opset {opsets}, expected {deploy_config.ONNX_OPSET_VERSION}.")

    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    if inputs != ["obs"] or outputs != ["actions"]:
        raise ValueError(f"Exported graph is {inputs} -> {outputs}, expected ['obs'] -> ['actions'].")
    expected_input = [1, deploy_config.OBSERVATION_DIM]
    expected_output = [1, deploy_config.ACTION_DIM]
    if _tensor_shape(model.graph.input[0]) != expected_input:
        raise ValueError(f"Exported obs shape is {_tensor_shape(model.graph.input[0])}, expected {expected_input}.")
    if _tensor_shape(model.graph.output[0]) != expected_output:
        raise ValueError(
            f"Exported actions shape is {_tensor_shape(model.graph.output[0])}, expected {expected_output}."
        )
    float_type = onnx.TensorProto.FLOAT
    if model.graph.input[0].type.tensor_type.elem_type != float_type:
        raise ValueError("Exported obs input must be float32.")
    if model.graph.output[0].type.tensor_type.elem_type != float_type:
        raise ValueError("Exported actions output must be float32.")

    written = {entry.key: entry.value for entry in model.metadata_props}
    expected_metadata = {key: _metadata_value(value) for key, value in metadata.items()}
    if written != expected_metadata:
        differing = sorted(
            key for key in set(written) | set(expected_metadata) if written.get(key) != expected_metadata.get(key)
        )
        raise ValueError(f"Exported metadata does not match the deployment configuration: {differing}.")

    # A graph that checks out structurally can still compute the wrong thing, so it is run.
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    generator = np.random.default_rng(0)
    probes = [
        np.zeros((1, deploy_config.OBSERVATION_DIM), dtype=np.float32),
        generator.normal(size=(1, deploy_config.OBSERVATION_DIM)).astype(np.float32),
    ]
    for probe in probes:
        with torch.no_grad():
            expected_actions = module(torch.from_numpy(probe)).numpy()
        actual_actions = np.asarray(evaluator.run(None, {"obs": probe})[0])
        if actual_actions.shape != expected_actions.shape:
            raise ValueError(f"Exported graph returns shape {actual_actions.shape}, expected {expected_actions.shape}.")
        if not np.isfinite(actual_actions).all():
            raise ValueError("Exported graph returns non-finite actions.")
        deviation = float(np.max(np.abs(actual_actions - expected_actions)))
        if deviation > GRAPH_AGREEMENT_TOLERANCE:
            raise ValueError(f"Exported graph disagrees with the checkpoint by {deviation:.3e}.")


def export_policy(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    run_path: str = deploy_config.DEFAULT_RUN_PATH,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if output_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output path must differ from the checkpoint path.")

    module = build_export_module(checkpoint_path)
    metadata = deploy_config.onnx_metadata(run_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_mode = stat.S_IMODE(output_path.stat().st_mode)
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    observations = torch.zeros(1, deploy_config.OBSERVATION_DIM)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".onnx", dir=output_path.parent, delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.onnx.export(
            module,
            observations,
            temporary_path,
            export_params=True,
            opset_version=deploy_config.ONNX_OPSET_VERSION,
            verbose=False,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
        model = onnx.load(temporary_path)
        for key, value in metadata.items():
            entry = onnx.StringStringEntryProto()
            entry.key = key
            entry.value = _metadata_value(value)
            model.metadata_props.append(entry)
        _validate_model(model, module, metadata)
        onnx.save(model, temporary_path)
        temporary_path.chmod(output_mode)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
