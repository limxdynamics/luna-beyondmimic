# Luna 训练到导出快速上手

[English](QuickStart.md) | [中文](QuickStart.zh-CN.md)

这份手册只介绍一条最推荐的路径：从已经重定向好的 `.npy` 动作文件开始，准备训练用 `.npz`，启动一次
HU_L04 parallel deploy gravity 训练，最后用离线导出工具同时生成部署需要的 `policy.onnx`、
`reference.txt` 和 `param.yaml`。

如果你只是想完成一次从数据到部署文件的闭环，按本文从上到下执行即可。更细的原理、参数解释和排错细节
可以之后再看 `README.zh-CN.md`。

## 0. 整体流程

```text
重定向 motion.npy
  -> train.py [内部临时 motion.npz] -> model.pt

model.pt + 同一个 motion.npy
  -> export_policy_and_reference.py [内部临时 motion.npz]
  -> policy.onnx + reference.txt + param.yaml
```

本文默认任务固定为：

```text
Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0
```

这对应 Luna `HU_L04_01` 的闭环 `hu_l04_parallel` 训练配置，也是当前推荐用于部署的配置。

## 1. 安装和激活环境

推荐 conda 环境名使用 `limxmimic`。如果你已经安装好环境，可以跳到本节最后的激活命令。

> 本项目只支持源码 checkout 后执行 editable install，不支持 wheel、sdist 或 PyPI 分发。不要把
> `pip wheel` 构建成功当作可发布证据；wheel 不包含运行需要的完整 Python 子包和 HU_L04 大型资产。

### 1.1 创建 conda 环境

```bash
conda create -n limxmimic python=3.10.15 -y
conda activate limxmimic
```

设置两个环境变量。`PYTHONNOUSERSITE=1` 用于避免用户级 Python 包污染 conda 环境，
`OMNI_KIT_ACCEPT_EULA=Y` 用于接受 Omniverse EULA，避免安装或启动时卡在交互确认。

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n limxmimic
conda env config vars set OMNI_KIT_ACCEPT_EULA=Y -n limxmimic
conda deactivate
conda activate limxmimic
```

### 1.2 安装 PyTorch 和 Isaac Sim

Isaac Lab 2.1.0 需要 Python 3.10 和 PyTorch 2.5.1。当前推荐版本组合是 Isaac Sim 4.5.0、
Isaac Lab 2.1.0、PyTorch 2.5.1。

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools==75.8.0" wheel

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-cache-dir "isaacsim[all,extscache]==4.5.0" \
  --extra-index-url https://pypi.nvidia.com
```

### 1.3 安装 Isaac Lab

把 `/path/to` 换成你想放 Isaac Lab 的目录。

```bash
cd /path/to
git clone --branch v2.1.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-2.1.0
cd IsaacLab-2.1.0

echo "setuptools<81" > ~/isaaclab-build-constraints.txt
export PIP_CONSTRAINT=~/isaaclab-build-constraints.txt
export TERM=xterm
./isaaclab.sh --install
```

安装后简单验证：

```bash
python scripts/tutorials/00_sim/create_empty.py --headless
```

### 1.4 安装本仓库

把 `/path/to/LimXMimic` 换成你的仓库根目录路径。

```bash
cd /path/to/LimXMimic
python -m pip install -e source/whole_body_tracking
```

之后每次开始训练前，进入仓库并激活环境：

```bash
cd /path/to/LimXMimic
conda activate limxmimic
```

## 2. 输入文件要求

你需要先拿到一个重定向后的 `.npy` 动作文件。它通常来自 xsens/fzmotion 到 Luna 串联模型的重定向流程。

> **只处理可信文件。** `.npy` 内含 pickled Python dictionary，转换脚本使用 `allow_pickle=True`
> 加载。恶意文件可能在后续字段、形状和有限值校验开始前执行代码。项目负责人接受内部工作流的这项
> 风险，但支持范围仅限 LimX Dynamics 内部生成或验证过的动作文件；不要使用网络下载、第三方发送或
> 其他未知来源的 `.npy`。

这个 `.npy` 需要满足：

- 是 `numpy` 保存的 pickled dictionary。
- 包含 `body_states`、`dof_pos_vel`、`body_names`、`dof_names`、`fps`。
- `body_states` 形状是 `(frames, bodies, 13)`。
- `dof_pos_vel` 形状是 `(frames, joints, 2)`。
- `body_names` 和 `dof_names` 与数组第二维顺序一致。
- `fps` 是正数。
- 数组里不能有 `NaN` 或 `Inf`。

训练和组合导出的用户输入都固定为这个 `.npy`。两个入口会在内部生成临时 `.npz`，因为闭环
`hu_l04_parallel` 模型有 51 个 DoF，而重定向 `.npy` 只给出 27 个关节状态；还需要补出跟腱电机和
连杆相关状态。

## 3. 检查或保留中间 `.npz`

训练和组合导出会自动执行本节，不要求用户提供 `.npz`。第一次处理新动作时仍建议显式运行一次，以便
提前阅读并联可达性和 FK 告警；需要 play 或调试时也可以保留该中间文件。

假设原始文件是：

```text
/absolute/path/to/your_motion.npy
```

输出训练文件到：

```text
motions/your_motion_parallel_tail10.npz
```

执行：

```bash
python scripts/prepare_motion.py \
  --input /absolute/path/to/your_motion.npy \
  --output motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel
```

生成的 `.npz` 需要包含：

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

其中 `fps` 应该是 50。`joint_pos/joint_vel` 是闭环模型的 51 DoF 顺序，`body_*` 是闭环模型的 50 body
顺序。不要手工改这些数组顺序。

## 4. 校验中间轨迹

`prepare_motion.py` 已在最后自动执行该检查。也可以单独重跑：

```bash
python scripts/verify_motion_npz.py \
  --motion_file motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel
```

这一步会启动 Isaac Sim，用正向运动学检查 `.npz` 是否和机器人模型一致。不要只看命令是否退出，也要读
终端里的 warning。轨迹顺序错、闭环状态不一致、四元数不对，都可能在这里暴露。

## 5. 启动训练

训练入口只接受可信重定向 `.npy`，内部临时 `.npz` 不再是用户输入：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /absolute/path/to/your_motion.npy \
  --num_envs 4096 \
  --headless \
  --max_iterations 15000 \
  --logger tensorboard
```

训练日志和 checkpoint 会写到：

```text
logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/
```

常用 checkpoint 文件名类似：

```text
model_14999.pt
```

训练完成后，后续导出需要两个输入文件：

- `logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_14999.pt`
- `/absolute/path/to/your_motion.npy`

这两个输入必须对应同一次训练，并使用相同的尾帧和并联可达误差参数。不要用一个动作训练出来的
checkpoint 去配另一个 `.npy`。

## 6. 模型转换

只使用 `tools/offline_onnx_exporter/export_policy_and_reference.py`。

平台不提供 motion 时，仅导出 `policy.onnx`：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_14999.pt \
  --output_path exported/your_motion
```

提供训练使用的 NPY 时，生成完整部署包：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path exported/your_motion
```

该命令会先调用与训练相同的 `prepare_motion.py` 生成临时 `.npz`，因此需要在完整 LimxMimic 源码
checkout 和 Isaac Sim / Isaac Lab 环境中运行。临时 `.npz` 在导出结束后删除。

导出成功后会得到：

```text
exported/your_motion/policy.onnx
exported/your_motion/reference.txt
exported/your_motion/param.yaml
exported/your_motion/linkage_report.json
```

`--output_path` 是目录，不是文件名。如果目录不存在，脚本会自动创建。如果这个路径已经是普通文件，脚本会
报错，请换一个新目录名。

其中 `param.yaml` 来自 `tools/offline_onnx_exporter/param.yaml`，是
`LimxDanceSDKHumanoidController` 的完整配置，不再是仅含两个路径的简化模板。配置中的部署路径保持
`policy_dir: "/policy/test"` 和 `ref_dir: "motion/test.txt"`。

并联残差默认只告警并写入 `linkage_report.json`。如需严格拦截，增加
`--strict-motion-check`；显式设置 `--motion-max-unreachable-error` 也会启用严格模式。reference
阶段失败时命令返回非 0，但已成功导出的 `policy.onnx` 会保留。

## 7. 导出输入要求

### checkpoint 要求

- 必须是 RSL-RL 的 `model_*.pt`。
- 必须来自 `Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`。
- actor 输入必须是 141 维，输出必须是 27 维。
- checkpoint 里必须包含 observation normalizer。

如果拿 138 维的旧策略来导出，脚本会明确报错。

### motion `.npy` 要求

- 必须是训练时传给 `--motion_file` 的同一个可信重定向 `.npy`。
- 必须满足第 2 节的字段、形状、有限值和信任边界要求。
- 导出时的 `--motion-tail-seconds` 必须与训练一致。
- 生成的临时 `.npz` 固定为 50 Hz、51 DoF、50 bodies，并通过 FK 校验后才用于 reference。

## 8. 最常见问题

### `Motion ... must be a trusted retargeted .npy file`

你传入了旧的 `.npz` 或其他格式。训练和组合导出现在只接受可信重定向 `.npy`；`.npz` 是内部中间
产物。

### `Training motion must be 50 Hz`

内部中间 `.npz` 没有按标准准备流程生成。检查 `prepare_motion.py` 输出和代码版本，不要手工拼接
中间文件。

### `Checkpoint actor expects ... observations`

checkpoint 和当前导出器支持的任务不匹配。请确认训练任务是
`Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`。

### `output_path must be a directory`

`--output_path` 指向了一个已有文件。换一个目录，例如：

```bash
--output_path exported/your_motion_v2
```

## 9. 最小命令汇总

把下面两个路径换成你自己的：

- `/absolute/path/to/your_motion.npy`
- `logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_14999.pt`

```bash
# 可选：首次处理时检查并保留中间 npz
python scripts/prepare_motion.py \
  --input /absolute/path/to/your_motion.npy \
  --output motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel

python scripts/verify_motion_npz.py \
  --motion_file motions/your_motion_parallel_tail10.npz \
  --robot hu_l04_parallel

python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /absolute/path/to/your_motion.npy \
  --num_envs 4096 \
  --headless \
  --max_iterations 15000 \
  --logger tensorboard

python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path exported/your_motion
```
