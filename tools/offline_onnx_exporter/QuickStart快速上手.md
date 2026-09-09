# 模型转换快速上手

只使用 `tools/offline_onnx_exporter/export_policy_and_reference.py`：

- 不传 `--motion`：仅生成 `policy.onnx`。
- 传 `--motion`：生成完整的模型转换产物。

## 适用场景

这个工具只面向当前固定配置：

- 机器人：`HU_L04_01`
- 训练任务：`Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`
- 机器人模型：闭环 `hu_l04_parallel`
- policy 输入：141 维 observation，最后包含 `projected_gravity`
- policy 输出：27 维 action
- reference 输出：102 列 TXT，50 Hz，每行一帧

如果你的 checkpoint 不是这个任务训练出来的，导出器会拒绝导出，而不是生成一个可能无法部署的 ONNX。

## 需要准备的文件

### 1. 训练 checkpoint

传给 `--checkpoint`，通常叫：

```text
model_xxx.pt
```

这个文件来自训练目录，例如：

```text
logs/rsl_rl/hu_l04_parallel_deploy_gravity/<训练时间戳>/model_xxx.pt
```

要求：

- 必须是 RSL-RL 保存的 `model_*.pt` checkpoint。
- 必须包含 actor 权重和 observation normalizer。
- actor 输入维度必须是 141，输出维度必须是 27。
- 不能把没有 `projected_gravity` 的 138 维旧策略拿来导出。

### 2. 可选：训练使用的重定向 motion `.npy`

平台只需要策略时不填写 motion。需要完整部署包时，`--motion` 必须是训练时传给
`train.py --motion_file` 的同一个可信 `.npy`：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0 \
  --motion_file /path/to/your_motion.npy \
  --num_envs 4096 --headless \
  --max_iterations 15000
```

组合导出会调用同一个 `scripts/prepare_motion.py`，生成临时 50 Hz / 51 DoF / 50 bodies `.npz`，
增加默认 2 秒尾帧并执行 FK 校验，然后再生成 reference。该 `.npz` 是内部中间产物，导出结束后删除。

并联残差默认作为告警写入报告，不阻断模型转换。需要严格拦截时使用
`--strict-motion-check`；显式设置 `--motion-max-unreachable-error` 也会启用严格模式。
训练默认尾帧仍为 10 秒；如需导出相同长度，传 `--motion-tail-seconds 10`。
只有确认末帧告警来自高帧率连杆求解噪声、27 个电机已经稳定时，才可传
`--motion-force-tail` 绕过静止门禁。

只处理 LimX Dynamics 内部生成或验证过的可信 `.npy`；文件使用 pickle，未知来源输入可能执行代码。

### 3. 输出目录

传给 `--output_path`。这是一个目录，不是文件名。

导出成功后，目录下会固定生成：

```text
linkage_report.json
policy.onnx
reference.txt
param.yaml
```

未传 `--motion` 时目录中只有 `policy.onnx`。

如果目录不存在，脚本会自动创建。如果 `--output_path` 已经是一个普通文件，脚本会报错。

## 推荐命令

两种模式都使用同一个脚本，并从 LimxMimic 仓库根目录运行。

仅导出策略：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --output_path /absolute/path/to/exported
```

完整部署包会调用 Isaac Sim 动作准备流程：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path /absolute/path/to/exported
```

导出成功后会看到类似输出：

```text
[OK] policy.onnx exported to /absolute/path/to/exported/policy.onnx
[OK] reference.txt exported to /absolute/path/to/exported/reference.txt
[OK] param.yaml copied to /absolute/path/to/exported/param.yaml
[OK] linkage_report.json written to /absolute/path/to/exported/linkage_report.json
```

## 输出文件怎么用

`param.yaml` 是完整版控制器配置，不再是只含路径的简化模板；其中部署路径保持 `/policy/test` 和
`motion/test.txt`。

同一组 `policy.onnx`、`reference.txt` 和 `param.yaml` 应来自同一次训练配置和同一份源 `.npy`。
不要把 A 动作训练出来的 policy 和 B 动作的 reference 混在一起测试。

如果 reference 阶段失败，命令返回非 0，但已经通过校验的 `policy.onnx` 会保留。

## 常见错误

### `Motion ... must be a trusted retargeted .npy file`

你传入了旧 `.npz` 或其他格式。组合导出现在只接受训练使用的可信重定向 `.npy`。

### `Training motion must be 50 Hz`

内部准备流程没有生成标准 50 Hz 中间 `.npz`。检查准备日志和代码版本，不要手工拼接中间文件。

### `Checkpoint actor expects ... observations`

checkpoint 的 observation 维度不匹配。当前导出器只支持
`Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0` 的 141 维策略。

### `output_path must be a directory`

`--output_path` 应该传目录。如果你之前把 `output` 当作文件生成过，请换一个新目录名，例如：

```bash
--output_path /absolute/path/to/exported_wakawaka
```
