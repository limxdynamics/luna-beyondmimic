# HU_L04 策略与参考轨迹导出

只使用 `tools/offline_onnx_exporter/export_policy_and_reference.py`。该入口固定支持
HU_L04_01 闭环 `hu_l04_parallel`、141 维带 `projected_gravity` 的观测和 27 维动作。

## 仅导出策略

平台不提供动作文件时，只生成 `policy.onnx`，不会启动 Isaac Sim：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --output_path /absolute/path/to/exported
```

成功时输出目录只包含 `policy.onnx`。它包含 observation normalizer、Actor 和部署 metadata，接口为
`obs[1,141] -> actions[1,27]`。

## 导出完整部署包

在完整 LimxMimic checkout 和 Isaac Sim / Isaac Lab 环境中运行：

```bash
python tools/offline_onnx_exporter/export_policy_and_reference.py \
  --checkpoint /absolute/path/to/model_14999.pt \
  --motion /absolute/path/to/your_motion.npy \
  --output_path /absolute/path/to/exported
```

这里的 `--motion` 是训练时传给 `scripts/rsl_rl/train.py --motion_file` 的同一个可信重定向 `.npy`。
脚本生成临时 50 Hz / 51 DoF / 50 bodies `.npz`，默认加 2 秒尾帧并执行 FK 校验，再用它
生成 TXT。临时 `.npz` 在导出结束后删除。

并联不可达残差在模型转换中默认只告警，并写入 `linkage_report.json`。如需严格拦截，追加
`--strict-motion-check`；显式传入 `--motion-max-unreachable-error` 也会开启严格模式并使用该阈值。
训练默认加 10 秒尾帧，导出默认加 2 秒；如需导出相同长度，显式传
`--motion-tail-seconds 10`。
如果末帧静止门禁仅由高帧率连杆求解噪声触发，在确认 27 个电机速度已经稳定后，可显式传
`--motion-force-tail`；真实未停止的动作不应使用该参数绕过门禁。

成功后得到：

```text
exported/
├── linkage_report.json
├── policy.onnx
├── reference.txt
└── param.yaml
```

`param.yaml` 是 `LimxDanceSDKHumanoidController` 的完整配置，不再是简化模板；其中部署路径保持
`/policy/test` 和 `motion/test.txt`。`reference.txt` 为 50 Hz、无表头、每帧 102 列：

```text
joint_pos(27) + joint_vel(27) + base_pose(7) + base_vel(6)
+ head/left-foot/right-foot/left-hand/right-hand relative pose(5*7)
```

## 失败与部分成功

脚本先校验并发布 `policy.onnx`，再处理 motion：

- 策略失败：退出码非 0，不发布新 policy。
- 策略成功、reference 失败：退出码非 0，但保留 `policy.onnx`；日志明确标记 reference 失败。
- 全部成功：退出码 0，生成四个文件。

`linkage_report.json` 包含不可达帧数、占比、最差残差与帧号，以及连续问题分段。尾帧静止性和 FK
校验始终是硬门禁，不受 warning-only 并联残差模式影响。

尾帧静止门限分别为：关节速度 `0.1 rad/s`、刚体线速度 `0.1 m/s`、刚体角速度 `0.2 rad/s`。
三者分开判断，避免视觉上已经稳定的 30 Hz 手腕或头部末帧被过度拦截。

只处理 LimX Dynamics 内部生成或验证过的可信 `.npy`。该格式使用 pickle，未知来源文件可能在字段
校验前执行代码。
