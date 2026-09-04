# LimxMimic 环境安装记录

本文档记录 `limxmimic` conda 环境的完整安装流程，参考
[Isaac Lab 2.1 官方 pip 安装指南](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)，
并按 Isaac Sim 4.5 / Isaac Lab 2.1.0 的版本要求做了调整。

> 本仓库只支持源码 checkout 后使用 `pip install -e` 安装；不支持 wheel、sdist 或 PyPI 分发。

## 目标版本

| 组件 | 版本 |
| --- | --- |
| Ubuntu | 22.04.5 LTS |
| Python | 3.10.15 |
| PyTorch | 2.5.1+cu118 / torchvision 0.20.1+cu118 |
| Isaac Sim | 4.5.0.0（pip 安装） |
| Isaac Lab | v2.1.0（源码安装） |
| conda 环境名 | `limxmimic` |

安装路径：

- Isaac Lab 仓库：`/path/to/IsaacLab-2.1.0`
- Mimic 仓库：`/path/to/LimXMimic`
- conda 环境：`/path/to/miniconda3/envs/limxmimic`

> 机器上如果另有一份旧版 Isaac Lab，例如 `/path/to/IsaacLab-old`，
> 本次安装未改动它。

## 0. 安装前检查

```bash
lsb_release -a                    # Ubuntu 22.04.5 LTS
ldd --version | head -n 1         # 需要 GLIBC >= 2.35，本机 2.35
dpkg -l build-essential cmake     # robomimic 编译需要，本机已装
df -h /                           # 需预留 20 GB 以上（环境 17 GB + 临时下载）
nvidia-smi                        # 驱动是否正常
```

Isaac Sim 的 pip 包要求 **GLIBC 2.35+**。若为 Ubuntu 20.04（GLIBC 2.31），
应改用 Isaac Sim 二进制安装方式。

`cmake` 与 `build-essential` 若缺失：

```bash
sudo apt install cmake build-essential
```

## 1. 创建 conda 环境

Isaac Sim 4.X 要求 Python 3.10（5.X 才是 3.11）。

```bash
conda create -n limxmimic python=3.10.15 -y
```

### 设置环境变量

两个变量都通过 `conda env config vars` 写入环境，激活时自动生效，无需每次手动 export：

```bash
# 隔离用户级 site-packages，见「坑位一」
conda env config vars set PYTHONNOUSERSITE=1 -n limxmimic
# 接受 NVIDIA Omniverse EULA，避免安装/启动时卡在交互式提问
conda env config vars set OMNI_KIT_ACCEPT_EULA=Y -n limxmimic

conda activate limxmimic
```

验证变量已生效，且 `.local` 不在搜索路径里：

```bash
echo "$PYTHONNOUSERSITE $OMNI_KIT_ACCEPT_EULA"          # 应输出：1 Y
python -c "import sys; print([p for p in sys.path if '.local' in p])"   # 应输出：[]
```

## 2. 准备 pip 与 setuptools

```bash
pip install --upgrade pip
# 降级 setuptools，见「坑位二」。必须在装 Isaac Lab 之前做
pip install "setuptools==75.8.0" wheel
python -c "import pkg_resources; print('pkg_resources OK')"
```

## 3. 安装 PyTorch

Isaac Lab 2.1.0 在 `source/isaaclab/setup.py` 中固定 `torch==2.5.1`，默认索引为 cu118：

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
# 2.5.1+cu118 0.20.1+cu118
```

## 4. 安装 Isaac Sim 4.5.0

约 4 GB 下载、13 GB 安装体积。`--no-cache-dir` 用于节省磁盘（否则 pip 缓存会再占一份）。

```bash
pip install --no-cache-dir "isaacsim[all,extscache]==4.5.0" \
    --extra-index-url https://pypi.nvidia.com

python -c "import isaacsim; print('isaacsim OK')"
```

这一步会把 `numpy` 自动从 2.x 降到 1.26.4（Isaac Sim 要求 `numpy<2`），属正常行为。

## 5. 安装 Isaac Lab v2.1.0

```bash
cd /path/to
git clone --branch v2.1.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-2.1.0
cd IsaacLab-2.1.0
```

创建构建约束文件，让 pip 的隔离构建环境也使用带 `pkg_resources` 的 setuptools：

```bash
echo "setuptools<81" > ~/isaaclab-build-constraints.txt
```

执行安装（默认装齐 rl_games、rsl_rl、sb3、skrl、robomimic）：

```bash
export TERM=xterm                                          # 见「坑位三」
export PIP_CONSTRAINT=~/isaaclab-build-constraints.txt     # 见「坑位二」
./isaaclab.sh --install
```

**务必核对 5 个包全部装上**，`isaaclab.sh` 即使核心包失败也可能返回成功：

```bash
pip list | grep -i "^isaaclab"
```

期望输出：

```
isaaclab            0.36.21   .../IsaacLab-2.1.0/source/isaaclab
isaaclab_assets     0.2.2     .../IsaacLab-2.1.0/source/isaaclab_assets
isaaclab_mimic      1.0.7     .../IsaacLab-2.1.0/source/isaaclab_mimic
isaaclab_rl         0.1.4     .../IsaacLab-2.1.0/source/isaaclab_rl
isaaclab_tasks      0.10.31   .../IsaacLab-2.1.0/source/isaaclab_tasks
```

安装末尾出现的 `[WARN] Could not find Isaac Sim VSCode settings` 只影响 VSCode 代码补全，可忽略。

## 6. 安装 LimxMimic

```bash
cd /path/to/LimXMimic
python -m pip install -e source/whole_body_tracking
```

HU_L04 闭链 USD 已随仓库提供，不需要另外下载机器人资产。

## 7. 验证

```bash
conda activate limxmimic
cd /path/to/IsaacLab-2.1.0
python scripts/tutorials/00_sim/create_empty.py --headless
```

因为 Isaac Sim 用的是 `extscache` 变体（扩展已打包进 wheel），首次启动不需要从
注册表拉扩展，约 10 秒即可完成，不会出现文档中提到的「首次运行 10 分钟」。

版本总览：

```bash
python -c "
import importlib.metadata as md
for p in ['isaacsim','isaaclab','isaaclab_tasks','torch','torchvision','numpy','rsl-rl-lib','whole_body_tracking']:
    print(f'{p:22s} {md.version(p)}')
"
```

## 已知坑位

### 坑位一：用户级 site-packages 污染环境

`~/.local/lib/python3.10/site-packages` 下如果已有很多包（`cv2`、`numba`、
`llvmlite`、`jinja2`、`cmeel` 等）。所有 Python 3.10 的 conda 环境都会把它挂进
`sys.path`，而且**排在环境自己的 site-packages 前面**，造成两类问题：

1. 运行时被外部包覆盖（例如 `.local` 的 opencv 会盖掉 Isaac Sim 需要的版本）；
2. 安装时 pip 误判依赖「已满足」而跳过，导致环境内实际缺包。

本次安装第一遍装 PyTorch 时，`jinja2`、`networkx`、`mpmath` 就因此被跳过。

处理：设置 `PYTHONNOUSERSITE=1`（已写入环境变量），然后重跑一次安装命令补齐被跳过的依赖。

### 坑位二：setuptools 83 移除 pkg_resources，导致 isaaclab 核心包静默安装失败

`isaaclab` 依赖 `flatdict==4.0.1`，这是个老式 sdist，其 `setup.py` 需要
`pkg_resources`。而新版 setuptools（>= 82）已将其移除，构建时报：

```
ModuleNotFoundError: No module named 'pkg_resources'
ERROR: Failed to build 'flatdict' when getting requirements to build wheel
```

后果比较隐蔽：**`isaaclab` 核心包整个安装被回滚，但 `./isaaclab.sh --install`
仍然返回 exit code 0**，后续的 `isaaclab_assets` / `tasks` / `rl` / `mimic` 照常装上，
只有核心包缺失，直到 `pip show isaaclab` 才能发现。

处理，两处都要做：

- 环境内降级：`pip install "setuptools==75.8.0"`
- pip 隔离构建环境：`PIP_CONSTRAINT` 指向含 `setuptools<81` 的约束文件

### 坑位三：isaaclab.sh 在 TERM=dumb 下直接退出

`isaaclab.sh` 第 16 行的 `tabs 4` 在非交互终端会报
`tabs: terminal type 'dumb' cannot reset tabs` 并使脚本中止。
执行前 `export TERM=xterm` 即可。

### 坑位四：editable install 不应升级 Isaac Lab 基础依赖

当前 `pip install -e source/whole_body_tracking` 只补充 `psutil`，不会安装 `onnxscript`，也不会升级
镜像内的 NumPy、ONNX 或 protobuf。安装后应保持：

```text
numpy 1.26.4
onnx 1.16.1
protobuf 3.20.3
```

如果安装日志出现 `onnxscript 0.7.x`，继而把 NumPy 升到 2.x、ONNX 升到 1.22 或 protobuf 升到
7.x，说明使用的是旧版仓库或旧压缩包；这些版本会破坏 Isaac Lab 的依赖约束，不应忽略。

### 坑位五：未启动 SimulationApp 时无法 import

以下报错是 pip 版 Isaac Sim 的**正常现象**，不是安装问题：

```
ModuleNotFoundError: No module named 'isaacsim.core'
ModuleNotFoundError: No module named 'omni.kit'
```

Kit 扩展要等 `SimulationApp` 启动后才会加入 `sys.path`。所有脚本都必须先走
`AppLauncher`，再 import `isaaclab_tasks` / `whole_body_tracking`。
不要用裸 `python -c "import isaaclab_tasks"` 来判断安装是否成功。

### 坑位六：NVIDIA 驱动版本不匹配

现象：

```
nvidia-smi                        # Failed to initialize NVML: Driver/library version mismatch
torch.cuda.is_available()         # False
# Isaac Sim: CUDA error 804: cudaErrorCompatNotSupportedOnDevice
```

原因是用户态驱动已升级（`nvidia-driver-580` → 580.173.02），但当前加载的内核模块
仍是旧版（580.159.03）。由于 X 会话长期占用，`nvidia` 模块引用计数很高，无法热重载，
**只能重启机器**。重启与本套环境安装无关，是系统层面的独立问题。

## 磁盘占用参考

| 项目 | 体积 |
| --- | --- |
| conda 环境 `limxmimic` | 17 GB |
| IsaacLab-2.1.0 仓库（`--depth 1` 浅克隆） | 90 MB |
| HU_L04 闭链 USD | 约 82 MB |

安装过程另需约 4 GB 临时下载空间。

若空间紧张，可用 `pip cache purge` 清理 pip 缓存。
