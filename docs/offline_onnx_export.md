# ONNX and Reference Export

Use `tools/offline_onnx_exporter/export_policy_and_reference.py` for both platform conversion
modes. With only a checkpoint it writes `policy.onnx`. With a trusted retargeted `.npy` it also
invokes the repository motion-preparation pipeline and writes:

- `policy.onnx`: the deployed policy graph, `obs -> actions`.
- `reference.txt`: the 102-column DanceLibrary reference trajectory.
- `param.yaml`: the minimal deployment parameter template that points the controller at the policy
  and reference.
- `linkage_report.json`: machine-readable linkage residual summary and contiguous bad segments.

## Supported Configuration

The exporter no longer consumes an external `export_spec.json`. The only supported configuration is
encoded in `tools/offline_onnx_exporter/deploy_config.py`:

- robot: `HU_L04_01`
- articulation: closed-loop `hu_l04_parallel`
- task: `Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`
- policy observation width: `141`
- policy action width: `27`
- reference rate: `50 Hz`

The checkpoint alone is still insufficient: activation, observation layout, joint names, gains,
action scale, tracked bodies, and deployment metadata all come from `deploy_config.py`.

## Inputs

The exporter always uses:

- `model_*.pt`: RSL-RL checkpoint with actor weights and empirical observation normalizer.

`motion.npy` is optional. Omit it for platform policy-only conversion. When supplied, it must be
the same trusted retargeted motion passed to training; `param.yaml` is then copied from the tool
directory.

The `.npy` must be a retargeter-produced pickled dictionary containing `body_states`,
`dof_pos_vel`, `body_names`, `dof_names`, and `fps`. It is prepared with the same closed-loop
solve, 50 Hz conversion, final-pose hold, and FK verification used by `train.py`. Unknown or
untrusted `.npy` inputs are unsupported because pickle loading can execute code before validation.

## Usage

Run from the LimxMimic repository root. Policy-only conversion does not launch Isaac Sim:

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --output_path /absolute/path/to/exported
```

Supply a motion for the complete deployment bundle. This path launches Isaac Sim for linkage
solving and FK verification:

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path /absolute/path/to/exported
```

Linkage residuals are warnings by default during model conversion and are recorded in
`linkage_report.json`. Add `--strict-motion-check` to reject residuals above 0.1 rad. An explicit
`--motion-max-unreachable-error` also enables strict mode with that limit. Use the same
`--motion-tail-seconds` as training. Optionally use `--run-path organization/project/run` to
record the checkpoint source in ONNX metadata.

The output directory will contain:

```text
exported/
├── linkage_report.json
├── policy.onnx
├── reference.txt
└── param.yaml
```

`param.yaml` is copied in full from `LimxDanceSDKHumanoidController` rather than generated as a
minimal stub. Its deployment-side paths remain:

```yaml
HumanoidRobotCfg:
  policy_dir: "/policy/test"
  ref_dir: "motion/test.txt"
```

## Validation

The policy is validated and published first. If no motion is supplied, the command returns zero
with only `policy.onnx`. If motion preparation or reference conversion fails, the command returns
non-zero but keeps the valid policy; stale reference, parameter, and report outputs are
removed before the attempt.

With a motion, the exporter validates its schema, the temporary training `.npz`, FK consistency,
the 102-column reference layout, and the final TXT write. Warning-only linkage quality does not
disable the tail or FK gates.

To inspect or retain the intermediate `.npz` separately:

```bash
python scripts/verify_motion_npz.py \
  --motion_file motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel
```

Optionally compare the exported joint metadata against an independent deployment configuration:

```bash
python scripts/check_deploy_onnx.py exported/policy.onnx \
  --hardware_config /path/to/HU_L04_01/config.yaml
```

The hardware config is intentionally supplied by the caller. The public repository does not
contain a developer-specific default or duplicate the independent deployment source.
