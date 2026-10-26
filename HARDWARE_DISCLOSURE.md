# HU_L04 hardware disclosure record

## Decision

On 2026-09-04, the project owner approved public disclosure of all HU_L04 hardware and deployment
information currently included in this repository. This supersedes the 2026-09-01 approval, which
predated `tools/offline_onnx_exporter/param.yaml` becoming a complete controller configuration:
the last four rows below entered the repository after that date and are covered by this approval.

The approved scope is:

| Information | Public location | Decision |
| --- | --- | --- |
| Closed-loop robot geometry | [luna-description](https://github.com/limxdynamics/luna-description), `HU_L04_description/usd_parallel/` | Approved |
| Achilles linkage geometry | `tools/offline_onnx_exporter/deploy_config.py` | Approved |
| Joint names, ordering, and motor-space aliases | `hu_l04_parallel.py`, `deploy_config.py`, ONNX metadata | Approved |
| Joint and achilles-motor travel limits | `deploy_config.py`, read from the description's URDF and MJCF | Approved |
| Stiffness, damping, armature, and action scale | `hu_l04_parallel.py`, `deploy_config.py`, ONNX metadata | Approved |
| `motor` joint-space mode and parallel solver mapping | `deploy_config.py`, `tools/offline_onnx_exporter/param.yaml`, exporter metadata | Approved |
| Per-joint torque budget | `tools/offline_onnx_exporter/param.yaml` | Approved |
| IMU offset interface | `tools/offline_onnx_exporter/param.yaml` | Approved |
| Policy/control rates and action transition | `tools/offline_onnx_exporter/param.yaml` | Approved |
| Deployment topics and configured path conventions | `tools/offline_onnx_exporter/param.yaml` | Approved |

## Why these values are included

The fixed deployment exporter is intended to be reproducible. Joint order, gains, action scale,
linkage geometry, and the motor-space convention are part of the contract between training, ONNX
metadata, the reference trajectory, and the controller. Removing or replacing them with
placeholders would make the published artifacts impossible to validate against the supported
configuration.

`tools/offline_onnx_exporter/param.yaml` is the controller configuration the exporter copies
alongside each policy. It is a complete `LimxDanceSDKHumanoidController` configuration rather than
a path-only stub, which is why the four rows above are in scope. The robot's own `config.yaml` is
a separate artifact and is not part of this repository.

The two deployment cross-check scripts still require caller-supplied independent configuration:

```bash
python scripts/check_deploy_onnx.py policy.onnx \
  --hardware_config /path/to/HU_L04_01/config.yaml

python scripts/verify_parallel_motor_convention.py \
  --config_dir /path/to/HU_L04_01/linkage-config
```

The independent configuration is not duplicated in this repository because validating a published
copy against itself would not be an effective cross-check.

## Safety boundary

Permission to disclose these values does not make them universal safe limits and does not authorize
their use on a different robot or controller version. See [`SAFETY.md`](SAFETY.md).
