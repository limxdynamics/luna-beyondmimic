<div align="center">

<img src="docs/limxmimic_banner.png" alt="LimxMimic: Luna humanoid dance poses rendered from the bundled motion" width="100%"/>

**English** | [中文](README.zh-CN.md)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Upstream code](https://img.shields.io/badge/upstream_code-MIT-yellow.svg)](LICENCE)
[![LimX modifications](https://img.shields.io/badge/LimX_modifications-Apache--2.0-blue.svg)](LICENSES/Apache-2.0.txt)
[![Robot description](https://img.shields.io/badge/robot_description-Apache--2.0-blue.svg)](https://github.com/limxdynamics/luna-description)

</div>

# LimxMimic — Motion Tracking for Luna (HU_L04)

Motion tracking training for the Luna (HU_L04) humanoid, built on
[BeyondMimic](https://beyondmimic.github.io/)
([paper](https://arxiv.org/abs/2508.08241), [video](https://youtu.be/RS_MtKVIAzY)).

**Scope.** The closed-loop Luna model, with the task fixed to
`Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`. Retargeting itself and training-platform job
submission are out of scope. No other robot or training task is part of the public package.

## Highlights

- **BeyondMimic, adapted to Luna.** Whole-body motion tracking for the 27-DoF Luna humanoid on Isaac Lab, trained with PPO.
- **Closed-loop linkages handled for you.** Every motion is converted automatically: the ankle and waist linkages are solved, the motion is mapped to the 51-DoF closed-loop training layout, and forward kinematics is checked before training starts.
- **One `.npy` from training to deployment.** Training, replay and export all take the same retargeted motion file.
- **Deployment-ready export.** `policy.onnx` (141-dim observation, 27 actions), a 50 Hz reference trajectory and a complete `param.yaml`, plus scripts that cross-check them against the deployment configuration.
- **Try it end to end.** Ships with a 22.9 s dance motion collected by LimX Dynamics.

## Pipeline

```text
retargeted motion.npy  (27 DoF, 50 Hz)
        │
        │  prepared automatically: solve ankle + waist linkages ─► 51-DoF closed-loop layout
        │                          ─► hold final pose ─► forward-kinematics check
        ▼
  train.py   Isaac Lab + PPO ─────────────────────────► model.pt
        │
        ├──► play.py   replay the policy in Isaac Sim
        │
        ▼
  export_policy_and_reference.py ─► policy.onnx + reference.txt + param.yaml
```

## Documentation

This README covers four things: setting up the environment, reading the repository layout, knowing
which parameters are safe to tune, and taking a retargeted `.npy` all the way to a policy you can
replay. Every gotcha we have actually hit is collected in [section 5](#5-gotchas).

| Document | What it covers |
| --- | --- |
| [QuickStart.md](QuickStart.md) | Shortest path from a retargeted `.npy` to the deployment files |
| [Install.md](Install.md) | A longer, blow-by-blow record of the same install (in Chinese) |
| [1. Installation](#1-installation) | Pinned versions, preflight checks and a 7-step install |
| [2. Repository layout](#2-repository-layout) | Where everything lives and where to change what |
| [3. Tunable parameters](#3-tunable-parameters) | What is fixed and what is safe to tune |
| [4. From a retargeted `.npy` to `play`](#4-from-a-retargeted-npy-to-play) | Motion conversion, training, checkpoints and convergence |
| [5. Gotchas](#5-gotchas) | Every issue we have actually hit, with fixes |
| [SAFETY.md](SAFETY.md) | The approved operating boundary before real-hardware use |
| [motions/README.md](motions/README.md) | Bundled motion data: schema, provenance and license |

## On the real robot

<p align="center">
  <img src="docs/luna_dance_figure.png" alt="LimX Luna humanoid performing a whole-body dance motion learned with LimxMimic" width="100%"/>
</p>

---

## 1. Installation

> **Source-only distribution.** This project supports an editable source checkout only:
> `python -m pip install -e source/whole_body_tracking`. Building, publishing, or installing a
> wheel/sdist/PyPI package is not supported. The runtime uses the large HU_L04 assets in the source
> tree, which are not packaged into a wheel.

### 1.1 Versions

These versions are bound to each other — do not upgrade any one of them alone. Isaac Sim 4.X only
supports Python 3.10, and Isaac Lab 2.1.0 pins `torch==2.5.1` in its `setup.py`.

| Component | Version |
| --- | --- |
| Ubuntu | 22.04.5 LTS |
| Python | 3.10.15 |
| PyTorch | 2.5.1+cu118 / torchvision 0.20.1+cu118 |
| Isaac Sim | 4.5.0.0 (pip) |
| Isaac Lab | v2.1.0 (from source) |
| conda env | `limxmimic` |

### 1.2 Preflight

```bash
lsb_release -a                    # Ubuntu 22.04.5 LTS
ldd --version | head -n 1         # needs GLIBC >= 2.35
dpkg -l build-essential cmake     # if missing: sudo apt install cmake build-essential
df -h /                           # keep 20 GB free
nvidia-smi                        # driver healthy?
```

The Isaac Sim pip package needs **GLIBC 2.35+**. Ubuntu 20.04 ships 2.31 and cannot use it; install
Isaac Sim from the binary release instead.

### 1.3 Install

Seven steps, in order. **The setuptools downgrade in step 2 has to happen before Isaac Lab is
installed** — see [gotcha 2](#gotcha-2-setuptools-drops-pkg_resources-and-isaaclab-fails-silently).

```bash
# 1/7  create the environment and bake in two variables
conda create -n limxmimic python=3.10.15 -y
conda env config vars set PYTHONNOUSERSITE=1 -n limxmimic
conda env config vars set OMNI_KIT_ACCEPT_EULA=Y -n limxmimic
conda activate limxmimic
echo "$PYTHONNOUSERSITE $OMNI_KIT_ACCEPT_EULA"     # expect: 1 Y

# 2/7  downgrade setuptools
pip install --upgrade pip
pip install "setuptools==75.8.0" wheel
python -c "import pkg_resources; print('pkg_resources OK')"

# 3/7  PyTorch
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 4/7  Isaac Sim, ~4 GB download and ~13 GB installed
pip install --no-cache-dir "isaacsim[all,extscache]==4.5.0" \
    --extra-index-url https://pypi.nvidia.com

# 5/7  Isaac Lab v2.1.0
git clone --branch v2.1.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-2.1.0
cd IsaacLab-2.1.0
echo "setuptools<81" > ~/isaaclab-build-constraints.txt
export TERM=xterm
export PIP_CONSTRAINT=~/isaaclab-build-constraints.txt
./isaaclab.sh --install
pip list | grep -i "^isaaclab"                     # all five packages must be present

# 6/7  install this repository
cd /path/to/whole_body_tracking
python -m pip install -e source/whole_body_tracking

# 7/7  verify
cd /path/to/IsaacLab-2.1.0
python scripts/tutorials/00_sim/create_empty.py --headless
```

> **Do not skip the `pip list` check in step 5.** `isaaclab.sh` returns success even when the core
> package failed to install. You should see all five of `isaaclab`, `isaaclab_assets`,
> `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks`.

Step 4 downgrades numpy from 2.x to 1.26.4. That is expected.

Disk: the conda environment is ~17 GB, the shallow Isaac Lab clone ~90 MB, and installation needs
another ~4 GB of scratch space. `pip cache purge` frees space if you are tight.

`Install.md` keeps a longer, blow-by-blow record of the same install.

---

## 2. Repository layout

Two halves: `source/` is the Python package that holds the training code, `scripts/` holds the
command line tools you actually run. Configuration lives in the former, workflow in the latter.

```
whole_body_tracking/
├── scripts/                         command line tools
│   ├── rsl_rl/
│   │   ├── train.py                 start training
│   │   ├── play.py                  replay a policy, also exports ONNX
│   │   └── cli_args.py              shared train/play arguments
│   ├── prepare_motion.py            the four steps below, in one command
│   ├── solve_parallel_linkage.py    solve the closed-loop DoF
│   ├── npy_to_npz.py                retargeted .npy -> trainable .npz
│   ├── append_motion_tail.py        hold the final pose for N seconds
│   ├── verify_motion_npz.py         forward-kinematics check (hard gate)
│   ├── linkage_quality.py           per-frame residual report for a linkage solve
│   ├── dump_articulation_order.py   print the canonical joint/body order
│   ├── replay_npz.py                replay a reference motion in Isaac Sim
│   ├── export_motion_policy.py      entry point for the offline ONNX exporter
│   ├── check_deploy_onnx.py         cross-check ONNX metadata vs a hardware config
│   └── verify_parallel_motor_convention.py   cross-check motor sign and zero
│
├── source/whole_body_tracking/whole_body_tracking/
│   ├── tasks/tracking/
│   │   ├── tracking_env_cfg.py      rewards, terminations, randomization, observations
│   │   ├── mdp/
│   │   │   ├── commands.py          motion loading, error terms, adaptive sampling
│   │   │   ├── rewards.py           reward implementations
│   │   │   ├── terminations.py      termination implementations
│   │   │   ├── events.py            domain randomization implementations
│   │   │   └── observations.py      observation implementations
│   │   └── config/hu_l04/
│   │       ├── parallel_env_cfg.py  closed-loop environment configuration
│   │       └── agents/rsl_rl_ppo_cfg.py   PPO hyperparameters and network
│   ├── robots/hu_l04_parallel.py    actuator gains, armature, action scale
│   ├── assets/__init__.py           resolves the description submodule path
│   └── utils/exporter.py            ONNX export and metadata
│
├── robot_descriptions/luna-description/   submodule: the HU_L04 USD, URDF, MJCF and meshes
├── motions/                         retargeted .npy reference motions
├── logs/rsl_rl/                     checkpoints, TensorBoard, exported ONNX
├── tools/offline_onnx_exporter/     policy-only offline export + combined NPY pipeline
└── tests/                           pytest suite for the scripts and the exporter
```

### Where to change what

| To change | Edit |
| --- | --- |
| Rewards, termination thresholds, domain randomization | `tasks/tracking/tracking_env_cfg.py` |
| Curriculum sampling | `MotionCommandCfg` in `tasks/tracking/mdp/commands.py` |
| Add a reward function | `tasks/tracking/mdp/rewards.py`, then register it in `RewardsCfg` |
| PPO hyperparameters, network width | `config/hu_l04/agents/rsl_rl_ppo_cfg.py` |
| Actuator stiffness and damping | `robots/hu_l04_parallel.py` |
| Observation terms | `config/hu_l04/parallel_env_cfg.py` — **but don't**, see [3.1](#31-hard-rule-observations-and-actions-are-fixed) |

---

## 3. Tunable parameters

### 3.1 Hard rule: observations and actions are fixed

**Do not add, remove, or reorder observation terms, and do not change the action width.** This is
not a style preference — a policy that breaks either one cannot be deployed.

| Term | Why it is frozen |
| --- | --- |
| **Observation** | Each term and its order is written into the exported ONNX metadata, and the onboard controller assembles its input vector against that fixed layout. One term more, one fewer, or a different order and the vector no longer means what the policy expects. |
| **Action** | 27 motor commands, each mapped to a specific hardware slot. Changing the width or the order sends torque to the wrong joint. |

The variant used here, `Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`, has a **141**-dimensional
observation and a **27**-dimensional action:

```
command(54) + base_ang_vel(3) + projected_gravity(3)
            + joint_pos(27) + joint_vel(27) + actions(27) = 141
```

**Deploy** in the name means everything the onboard controller cannot produce has already been
dropped — global position, the linear velocity that would need state estimation, and the linkage
DoF that have no sensor. **Gravity** means the IMU's gravity direction is kept, which after
dropping state estimation is the only remaining direct measure of trunk tilt.

### 3.2 Rewards — free to add, remove, and retune

The safest knob: nothing here touches the deployment interface. Defined in `RewardsCfg` in
`tracking_env_cfg.py`.

| Reward | Weight | std | Effect of increasing |
| --- | --- | --- | --- |
| `motion_global_anchor_pos` | 0.5 | 0.3 | Trunk world position tracks harder, at the cost of local pose freedom |
| `motion_global_anchor_ori` | 0.5 | 0.4 | Trunk heading tracks harder |
| `motion_body_pos` | 1.0 | 0.3 | Limb positions track harder; the main driver of whether the motion looks right |
| `motion_body_ori` | 1.0 | 0.4 | Limb orientations track harder |
| `motion_body_lin_vel` | 1.0 | 1.0 | Timing tracks harder, motion drags less |
| `motion_body_ang_vel` | 1.0 | 3.14 | Rotational timing tracks harder |
| `action_rate_l2` | −0.1 | — | Smoother, less chatter, but blunter response — fast motions fall behind |
| `joint_limit` | −10.0 | — | Stays further from the limits. Fights the tracking terms when the reference itself sits on a limit |
| `undesired_contacts` | −0.1 | — | Less willing to touch the ground with anything but feet and hands |

> **`std` is usually the better knob.** All six tracking terms are `exp(−error² / std²)`, so a
> **smaller `std` is a stricter demand** — the reward collapses toward zero for a slightly larger
> error. A larger one is more forgiving but flattens the gradient. It directly encodes "how much
> error still counts as good".

**Adding a term.** `rewards.py` already contains `feet_contact_time`, written but not wired into
any config, as a worked example. Write a function there and register a `RewTerm` in `RewardsCfg`.

**Removing a term.** Be careful: the six `motion_*` terms *are* the task definition, and dropping
any one of them leaves that dimension unconstrained. The three penalties are the ones you can
genuinely turn off.

### 3.3 Terminations

Defined in `TerminationsCfg`. Easy to overlook, but often more consequential than reward weights,
**because they decide whether the policy ever gets to see the end of a motion**.

| Term | Default | Meaning |
| --- | --- | --- |
| `time_out` | 10 s (500 steps) | Normal end, not a failure |
| `anchor_pos` | 0.25 m | Trunk anchor **height** deviates from the reference by more than this |
| `anchor_ori` | 0.8 | Trunk tilt deviates from the reference (difference of projected-gravity z) |
| `ee_body_pos` | 0.25 m | Any of the four end effectors (both ankles, both wrists) deviates in **height** |

> **Is the threshold right for your motion?** Estimate the tolerance window as
> `threshold ÷ peak vertical speed of that end effector`. At a 2 m/s peak and a 0.25 m threshold,
> the policy dies after falling 0.125 s behind. References with a big kick, a jump, or a get-up
> narrow that window until the policy never gets a chance to learn, which shows up as **a
> completion rate that will not climb and failures almost entirely from `ee_body_pos`**. Raising
> the threshold to 0.4–0.5 m usually fixes it outright.

### 3.4 Domain randomization

Defined in `EventCfg`. More randomization means a more robust policy that transfers better, at the
cost of slower training and lower tracking accuracy in simulation.

| Parameter | Default | Effect |
| --- | --- | --- |
| `physics_material` static friction | 0.3 – 1.6 | Ground friction range; keep the lower bound low if the real floor is slippery |
| `physics_material` dynamic friction | 0.3 – 1.2 | As above |
| `physics_material` restitution | 0.0 – 0.5 | Bounce on contact |
| `add_joint_default_pos` | ±0.01 rad | Joint zero-calibration error |
| `base_com` | x ±0.025 m, y/z ±0.05 m | Trunk centre-of-mass uncertainty; matters a lot for balance |
| `push_robot` interval | 1 – 3 s | Disturbance frequency |
| `push_robot` velocity | xy ±0.5 m/s, yaw ±0.78 rad/s | Disturbance strength; raising it buys robustness but shoves fine motions off course |

### 3.5 Curriculum sampling

Defined in `MotionCommandCfg`. Which second of the motion each reset starts from is chosen by a
failure-driven adaptive sampler.

| Parameter | Default | Effect |
| --- | --- | --- |
| **`adaptive_kernel_size`** | 1 | **The one worth knowing about.** At 1 a failure is credited only to the second it died in, and the sampler then restarts episodes there, so the policy never practises *entering* the hard part. When a long motion stalls at one spot, 5 spreads the weight back over the preceding four seconds |
| `adaptive_uniform_ratio` | 0.1 | Uniform floor. Raising it guarantees every second gets some practice instead of the budget collapsing onto one spot |
| `adaptive_alpha` | 0.001 | EMA coefficient of the failure histogram. Higher reacts faster but is noisier |
| `pose_range` / `velocity_range` | see config | Initial-state noise at reset; wider is more robust |

### 3.6 PPO and network

Defined in `config/hu_l04/agents/rsl_rl_ppo_cfg.py`. The defaults work across the motions we have
trained; **do not change them without evidence**.

| Parameter | Default | Note |
| --- | --- | --- |
| Network | `[512, 256, 128]`, elu | Only widen if the observation grows substantially |
| `num_steps_per_env` | 24 | Rollout length per environment |
| `learning_rate` | 1e-3, adaptive | Adaptively scheduled; the effective rate is driven by `desired_kl`, so changing the initial value achieves little |
| **`desired_kl`** | 0.01 | The real step-size knob. Smaller is more conservative and stable, but slower |
| `entropy_coef` | 0.005 | More exploration when raised; too high and the motion chatters |
| `gamma` / `lam` | 0.99 / 0.95 | Discount and GAE |
| `max_iterations` | 30000 | Overridable on the command line; 15000 is usually converged in practice |

### 3.7 Actuator gains

Defined in `robots/hu_l04_parallel.py`. Stiffness and damping are not hard-coded — they are derived
from two constants:

```python
NATURAL_FREQ  = 10 * 2 * pi   # 10 Hz
DAMPING_RATIO = 2.0

stiffness = armature * NATURAL_FREQ ** 2
damping   = 2 * DAMPING_RATIO * armature * NATURAL_FREQ
```

`armature` comes from the vendor MJCF. **Change the two constants rather than per-joint numbers**,
so the joints keep their relative proportions.

- **Raising `NATURAL_FREQ`** tracks harder and more accurately, but excites chatter on hardware.
- **Raising `DAMPING_RATIO`** is steadier but blunts the response.
- **`soft_joint_pos_limit_factor`** (default 0.9) sets how close to the limit the `joint_limit`
  penalty starts biting. Raise it when the reference genuinely needs to sit near a limit.

---

## 4. From a retargeted `.npy` to `play`

Assuming you already have a `.npy` retargeted onto Luna. Run everything from the repository root.

> **Trusted input boundary.** These `.npy` files contain a pickled Python dictionary and are loaded
> with `allow_pickle=True`. Loading a malicious file can execute code before schema validation
> starts. The project owner accepts this risk for the supported workflow, which is limited to
> trusted motion files generated or verified inside LimX Dynamics. Do not process `.npy` files from
> unknown or untrusted sources.

```
retargeted .npy  ->  solve_parallel_linkage  ->  npy_to_npz  ->  append_motion_tail
                 ->  verify_motion_npz (gate) ->  train.py   ->  play.py
```

### 4.1 Convert the motion

The user-facing input to training, playback, and combined export is always the trusted retargeted
`.npy`. All three commands prepare a temporary `.npz` internally because the closed-loop model has
51 degrees of freedom while the retargeter emits 27 joints — the extra 24 are the six achilles
motors and the 18 DoF of the six spherical rod joints.

> Those 24 solved DoF exist **only to give simulation resets a configuration that does not violate
> the closed-loop constraint**. They never enter the policy's observation or action; the interface
> stays 27-dimensional motor space throughout.

**The short way.** `train.py` accepts a `.npy` directly and prepares it first, so the single
command in [4.2](#42-train) takes you from retargeted data all the way to a running job — there is
no separate preparation step to run. Preparation adds about a minute, because two of its stages
each launch their own Isaac Sim. `--motion_tail_seconds` (default 10) and
`--motion_max_unreachable_error` tune that preparation.
To inspect or retain the intermediate `.npz`, run the same pipeline on its own — worth doing the
first time you see a motion, so you read the warnings before committing to a long job:

```bash
python scripts/prepare_motion.py \
  --input /path/to/your_motion.npy \
  --output motions/your_motion_parallel_tail10.npz
```

**The long way**, if you want to run the steps yourself:

```bash
# 1  solve the linkage (launches Isaac Sim, ~35 s)
python scripts/solve_parallel_linkage.py \
  --input_file /path/to/your_motion.npy \
  --output_file /tmp/linkage.npz

# 2  convert to npz
python scripts/npy_to_npz.py \
  --input_file /path/to/your_motion.npy \
  --robot hu_l04_parallel \
  --linkage_file /tmp/linkage.npz \
  --output_file /tmp/your_motion_parallel.npz

# 3  hold the final pose
python scripts/append_motion_tail.py \
  --input /tmp/your_motion_parallel.npz \
  --output motions/your_motion_parallel_tail10.npz \
  --tail_seconds 10

# 4  verify (mandatory, launches Isaac Sim, ~10 s)
python scripts/verify_motion_npz.py \
  --motion_file motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel
```

> All four commands will run to completion, but **silence is not the same as success**. Steps 1, 3
> and 4 each emit a class of warning you have to read. Ignoring them costs you the several hours it
> takes to notice the data was wrong. See [5.2](#52-preparing-motion-data).

**Why the tail.** On hardware the controller **repeats the last reference frame** once the motion
ends, until an operator switches controller. Training never asks for that: the moment the clip runs
out, the environment is teleported to a fresh random start time, so the final pose exists for a
frame or two and is never held. The result is a robot that cannot stand still after the motion
finishes. A 10 s tail makes the policy practise holding it. Ten seconds matches the episode length,
so a reset can land inside the tail and spend a whole episode doing nothing but standing; longer
adds little. The tail freezes the final pose and **zeroes every velocity**. Deployment uses the
same `.npy`, but the combined exporter appends a 2-second tail by default. Pass
`--motion-tail-seconds 10` during export when matching the training length is required.

### 4.2 Train

```bash
conda activate limxmimic
cd /path/to/whole_body_tracking

python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /path/to/your_motion.npy \
  --num_envs 4096 --headless \
  --max_iterations 15000 \
  --logger tensorboard
```

Output lands in `logs/rsl_rl/hu_l04_parallel_deploy_gravity/<timestamp>/`, one checkpoint every 500
iterations. `hu_l04_parallel_deploy_gravity` is the `experiment_name` this task variant is pinned
to, defined in `rsl_rl_ppo_cfg.py`.

### 4.3 Where the checkpoint and the motion have to live

**The checkpoint — give it a path.** A relative or absolute file path is used as given, wherever
the file happens to sit:

```bash
--checkpoint /path/to/model_14999.pt
```

A bare name or a regex is the legacy form: it is resolved under `--load_run` inside the experiment
directory, whose name the task variant fixes:

```
whole_body_tracking/
└── logs/rsl_rl/
    └── hu_l04_parallel_deploy_gravity/   <- fixed by the task variant
        └── 2026-08-14_18-07-36/          <- any directory name
            └── model_14999.pt            <- must be model_<number>.pt
```

**The motion source** can live anywhere. Pass the same trusted `.npy` used for training; `play.py`
runs the shared preparation pipeline before launching the playback environment. Use
`scripts/prepare_motion.py` only when you need to retain the intermediate `.npz` for diagnostics
or `replay_npz.py`.

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /path/to/your_motion.npy \
  --checkpoint /path/to/model_14999.pt \
  --num_envs 2
```

Add `--video --video_length 500` to record one full episode into that run's `videos/play/`.

Every `play` also exports the policy to
`logs/rsl_rl/hu_l04_parallel_deploy_gravity/<run>/exported/policy.onnx`. That is expected. For
deployment, prefer the combined exporter, which accepts the same source `.npy` as training and
writes `policy.onnx`, `reference.txt`, `param.yaml`, and `linkage_report.json` together. Omitting
the motion exports only `policy.onnx`; see
[ONNX and reference export](docs/offline_onnx_export.md).

#### Optional deployment cross-checks

These checks are not part of training. They deliberately require independent deployment-side
configuration supplied by the caller; the public repository does not guess a developer machine
path or duplicate hardware configuration.

```bash
# Compare ONNX joint metadata with the deployment robot config.
python scripts/check_deploy_onnx.py exported/policy.onnx \
  --hardware_config /path/to/HU_L04_01/config.yaml

# Compare closed-loop USD motor sign/zero conventions with the deployment linkage models.
python scripts/verify_parallel_motor_convention.py \
  --config_dir /path/to/HU_L04_01/linkage-config
```

The linkage directory must contain the approved left-ankle, right-ankle, and inverse-waist model
YAML files. A pass is evidence of interface consistency, not hardware safety approval.

### 4.4 Reading convergence

| Metric | How to read it |
| --- | --- |
| `Metrics/motion/error_body_pos`, `error_body_rot` | **The primary signal** — the task objective itself. Independent of the sampling distribution. Converged once the moving average stops falling for a couple of thousand iterations |
| `Episode_Termination/*` | Look at `time_out` as a share of the sum, i.e. the fraction of episodes that ran the full 10 s. The sum equals `num_envs ÷ mean episode length`, so smaller is better. Whichever term is largest names the bottleneck |
| `Metrics/motion/sampling_entropy` | How spread out the curriculum is, in `[0, 1]`. Real convergence pushes it toward 1. **Sitting low for a long time means the policy is not out of capacity, it is stuck on one hard passage.** Pair it with `sampling_top1_bin` to find the exact second |

> **Do not judge convergence by `Train/mean_reward`.** It is the episode return, so it is coupled to
> episode length, and it moves with the curriculum's sampling distribution. When the sampler shifts
> to a different hard spot the reward spikes or collapses — that reflects a changed task
> distribution, not a better or worse policy. It is especially misleading across runs with different
> termination thresholds or sampler settings.

---

## 5. Gotchas

### 5.1 Installation

#### Gotcha 1: user site-packages contaminate the environment

`~/.local/lib/python3.10/site-packages` is added to every conda environment's `sys.path`, **ahead of
the environment's own site-packages**. Two consequences: packages get shadowed at run time, and pip
decides a dependency is already satisfied and skips it, leaving the environment short at install
time. `PYTHONNOUSERSITE=1` (step 1 above) fixes it; re-run the install command afterwards to pick up
whatever was skipped.

#### Gotcha 2: setuptools drops `pkg_resources` and isaaclab fails silently

`isaaclab` depends on `flatdict`, an old-style sdist whose `setup.py` imports `pkg_resources`, which
setuptools ≥ 82 removed.

> **The failure mode is the problem.** The core package's install is rolled back, but
> `./isaaclab.sh --install` still exits 0 and the other four packages install normally. Nothing
> tells you until you run `pip list | grep -i "^isaaclab"` yourself.

Fix **both** places: downgrade to `setuptools==75.8.0` inside the environment, and point
`PIP_CONSTRAINT` at a file containing `setuptools<81` so pip's isolated build environment uses it too.

#### Gotcha 3: `isaaclab.sh` exits immediately under `TERM=dumb`

The `tabs 4` call in the script fails with `terminal type 'dumb' cannot reset tabs` and aborts.
`export TERM=xterm` first.

#### Gotcha 4: pip dependency-conflict warnings are safe to ignore

Installing this repository may report stale onnx and protobuf version pins from Isaac Lab. Both
have been verified against this repository's exporter and TensorBoard path.

#### Gotcha 5: imports fail before `SimulationApp` starts

`ModuleNotFoundError: No module named 'isaacsim.core'` is **normal** for the pip build, not a broken
install. Kit extensions only join `sys.path` after `SimulationApp` starts, so every script has to go
through `AppLauncher` first. Do not use a bare `python -c "import isaaclab_tasks"` to decide whether
the install worked.

#### Gotcha 6: NVIDIA driver mismatch

`nvidia-smi` reporting `Driver/library version mismatch` with `torch.cuda.is_available()` false
means the userspace driver was upgraded while the loaded kernel module is still the old one. A long
running X session pins the module so it cannot be reloaded — **the machine has to be rebooted**.
This is a system-level issue unrelated to this install.

### 5.2 Preparing motion data

| Source | Warning | What it means, what to do |
| --- | --- | --- |
| `solve_parallel_linkage` | `[FAIL] Worst residual … exceeds …` | The reference asks for ankle/waist poses the achilles linkage **physically cannot reach**. The file is still written — it holds the closest reachable configuration — and remains usable, but those frames carry a tracking ceiling that **more training iterations cannot close**. Above roughly 10% of frames, go back and fix the retarget |
| `append_motion_tail` | Refuses to run, reports the final frame is still moving | The motion still has speed at its last frame, so freezing it asks the robot to stop dead. **Usually a retargeting artifact at the clip boundary** — trim the last few frames with `npy_to_npz.py --frame_range 1 N`. Only when the warning is confirmed to be high-fps linkage-solver noise and all 27 motor speeds are settled may train/play use `--motion_force_tail` or combined export use `--motion-force-tail` |
| `verify_motion_npz` | `[FAIL]`, some bodies out of tolerance | **A hard gate — do not train on it.** Usually means the joint/body ordering constants no longer match the model |

### 5.3 Training and replay

#### Direct checkpoint paths avoid ambiguous run selection

Prefer `--checkpoint /path/to/model.pt`. If you use the legacy log-directory lookup instead, always
pass `--load_run` when an experiment contains more than one run. Without it, `play` silently takes
the **alphabetically last** run directory and says nothing about it. A smoke-test checkpoint from a
handful of iterations, or a policy for an entirely different motion, will load and look like a
success. Note also that **submission order and directory timestamp order do not have to agree** — a
job that queued later carries the larger timestamp, so guessing gets it wrong.

#### Isaac Sim ignores SIGTERM

Quit `play` with **Ctrl+C**. `timeout` and friends send signals it will not act on; use `kill -9`
if you have to.

#### `mean_reward` does not tell you whether training is going well

See [4.4](#44-reading-convergence). It is coupled to episode length and moves with the sampling
distribution, so comparing it across configurations gives outright wrong conclusions.

---

## 6. License

This repository is a modified fork of an MIT-licensed upstream project, so more than one license
applies. In short:

| What | License | Text |
| --- | --- | --- |
| Upstream code this fork started from | MIT, Copyright (c) 2024 The Isaac Lab Project Developers | [`LICENCE`](LICENCE) |
| LimX Dynamics' modifications and additions | Apache-2.0, Copyright 2026 LimX Dynamics | [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| Files carrying their own header (`scripts/rsl_rl/train.py`, `utils/exporter.py`) | BSD-3-Clause | [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt) |
| Motion data under `motions/` | Apache-2.0 | [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| Robot description (submodule, not distributed here) | Apache-2.0 | [luna-description](https://github.com/limxdynamics/luna-description) |

The upstream MIT notice is preserved verbatim and must stay that way in any redistribution. LimX
Dynamics licenses its own contributions under Apache-2.0; where a file carries its own license
header, that header governs the file. MIT and Apache-2.0 are compatible, and a recipient of the
combined work satisfies both by retaining `LICENCE`, `LICENSES/Apache-2.0.txt` and
[`NOTICE`](NOTICE).

The bundled motion data carries no separate terms. It is LimX Dynamics' own material, released
under the same Apache-2.0 license as the rest of its contributions, so no additional permission is
needed to use it - commercially or otherwise.

Fork provenance, file-level attributions and the full inventory are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## About this repository

### Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Do not open a public issue for
an unpatched vulnerability or unsafe real-hardware behavior; follow [SECURITY.md](SECURITY.md) and
report it privately to `opensource@limxdynamics.com`. The approved normal-operation boundary is in
[SAFETY.md](SAFETY.md), and publicly disclosed robot/deployment parameters are inventoried in
[HARDWARE_DISCLOSURE.md](HARDWARE_DISCLOSURE.md).

### Bundled motion-data provenance

The source motion represented by the file under `motions/` was **collected internally by LimX
Dynamics**. The `.npy` is a processed, retargeted robot-kinematics trajectory rather than a raw
capture recording. The project owner confirms that the represented data has been authorized for
use and has approved its publication here under Apache-2.0. See the
[motion-data inventory and data-use status](motions/README.md).
