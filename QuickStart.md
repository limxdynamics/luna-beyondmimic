# LimxMimic: training-to-export quick start

[English](QuickStart.md) | [中文](QuickStart.zh-CN.md)

This is the shortest supported path from a retargeted Luna `.npy` motion to the three deployment
files:

```text
retargeted motion.npy
  -> train.py [temporary motion.npz] -> model.pt

model.pt + the same motion.npy
  -> export_policy_and_reference.py [temporary motion.npz]
  -> policy.onnx + reference.txt + param.yaml
```

The one supported task is:

```text
Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0
```

## 1. Environment

The supported stack is Python 3.10.15, PyTorch 2.5.1, Isaac Sim 4.5.0, and Isaac Lab 2.1.0 on
Ubuntu 22.04. Follow the full [installation guide](README.md#1-installation) or the official
[Isaac Lab 2.1 pip installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).

Activate the environment and install this repository as an editable source checkout:

```bash
conda activate limxmimic
cd /path/to/LimxMimic
python -m pip install -e source/whole_body_tracking
```

Wheel, sdist, and PyPI distribution are not supported.

## 2. Trusted motion input

The input is a retargeted `.npy` containing a pickled dictionary with:

- `body_states`
- `dof_pos_vel`
- `body_names`
- `dof_names`
- `fps`

> Only load a motion generated or verified inside LimX Dynamics. The preparation scripts use
> `allow_pickle=True`; a malicious file can execute code before schema validation starts. Unknown
> or third-party `.npy` files are outside the supported trust boundary.

## 3. Optional: inspect the intermediate motion

Training and combined export perform this step automatically. Run it explicitly the first time you
see a new motion to inspect the warnings, or when you need to retain an `.npz` for replay/debugging:

```bash
python scripts/prepare_motion.py \
  --input /absolute/path/to/your_motion.npy \
  --output motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel
```

This performs the closed-loop solve, converts to the 51-DoF/50-body layout at 50 Hz, adds a
10-second held final pose, and runs the forward-kinematics check.

If the linkage gate reports an unreachable reference pose, inspect the affected-frame count and
worst residual. Only after deciding it is acceptable should you repeat with a higher explicit
limit, for example:

```bash
--max_unreachable_error 0.2
```

Do not train when the final forward-kinematics verification fails.

## 4. Train

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /absolute/path/to/your_motion.npy \
  --num_envs 4096 \
  --headless \
  --max_iterations 15000 \
  --logger tensorboard
```

Checkpoints are written to:

```text
logs/rsl_rl/hu_l04_parallel_deploy_gravity/<timestamp>/model_*.pt
```

Keep the source `.npy` and the preparation settings with the checkpoint. Combined export accepts
that same `.npy` and regenerates the intermediate `.npz` through the same pipeline. Retain the
explicit `.npz` from section 3 only when you need replay or debugging.

## 5. Replay

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /absolute/path/to/your_motion.npy \
  --checkpoint /absolute/path/to/model_14999.pt \
  --num_envs 2 \
```

Playback prepares the same temporary `.npz` as training. An existing checkpoint path is loaded
directly; bare checkpoint names or regexes still use the legacy `--load_run` lookup.

## 6. Export

Use the same script for both conversion modes. Omit `--motion` to export only `policy.onnx`
without launching Isaac Sim:

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint logs/rsl_rl/hu_l04_parallel_deploy_gravity/<timestamp>/model_14999.pt \
  --output_path exported/your_motion
```

Supply the training `.npy` for the complete bundle. This path requires the LimxMimic checkout,
Isaac Sim, and Isaac Lab:

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint logs/rsl_rl/hu_l04_parallel_deploy_gravity/<timestamp>/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path exported/your_motion \
  --run-path <traceable-task-or-run-id>
```

Outputs:

```text
exported/your_motion/policy.onnx
exported/your_motion/reference.txt
exported/your_motion/param.yaml
exported/your_motion/linkage_report.json
```

The policy must have one `obs[1,141]` input and one `actions[1,27]` output. The reference must be
50 Hz with 102 columns. The exported `param.yaml` is the full controller configuration rather than
a minimal stub. Linkage residuals are warnings by default; add `--strict-motion-check` to reject
residuals above 0.1 rad.

## 7. Before real-hardware use

Read:

- [`SAFETY.md`](SAFETY.md)
- [`HARDWARE_DISCLOSURE.md`](HARDWARE_DISCLOSURE.md)
- [`SECURITY.md`](SECURITY.md)

Interface, simulation, and unit-test passes do not constitute hardware safety approval.
