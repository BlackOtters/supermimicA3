# SuperMimic A3

独立的 A3 Ultra SuperMimic 训练仓库，当前为 **v9 四卡刚性版本**，基于 v8 / e5 更新。
包含训练、世界模型、A3 仿真适配、机器人 MJCF/mesh、SwanLab、评测与策略导出。
不需要原来的 MimicLite 工作区，不包含训练数据、checkpoint 或凭据。

This is a standalone snapshot of the A3 Ultra adaptation of
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced).
It reproduces the training recipe; convergence and hardware transfer are not claimed.

## 安装

Linux x86-64，Python **3.11**，支持 CUDA 12.8 的 NVIDIA 驱动。
默认四卡配置面向每卡 48 GB 显存。单卡可用于小规模验证。

```bash
git clone https://github.com/BlackOtters/supermimicA3.git
cd supermimicA3
bash scripts/install.sh
source .venv/bin/activate
a3-sm-check
```

安装脚本创建独立 `.venv`，从 PyTorch cu128 源安装 torch 2.10.0，
然后安装本项目与固定版本依赖。`requirements.txt` 记录原训练环境的关键版本：
mjlab 1.3.0、MuJoCo 3.11.0、mujoco-warp 3.8.1、Warp 1.12.0。
需要在本地已有环境中使用时：`python -m pip install -e .`。
本包自带 `active_adaptation` 和 `mimic_lite` 的所需代码，建议使用独立环境以避免同名包混用。

## 四卡训练

数据由使用者准备，设置 `DATA_DIR` 即可；支持本文后面的 NPZ 或 packed-library 格式。
默认启用 SwanLab。使用自己的凭据文件，并限制文件读取权限：

```bash
export DATA_DIR=/path/to/a3/library
export SWANLAB_API_KEY_FILE=/path/to/private/swanlab_api_key
chmod 600 "$SWANLAB_API_KEY_FILE"

# 不启动仿真，检查超参数、A3 模型和日志服务。
bash scripts/train.sh --check-config
bash scripts/train.sh --check-tracking

# DLC 中可以直接将以下命令作为 bash 入口。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 ENVS_PER_GPU=4096 \
  ACTOR_BATCH_PER_GPU=2048 WM_BATCH_PER_GPU=2048 \
  bash scripts/train.sh
```

也接受 `SWANLAB_API_KEY` 环境变量。不要将 key 写入源码或命令行参数。
自定义项目名使用 `A3_SM_SWANLAB_PROJECT`；不要把普通字符串写入 SDK 的 `SWANLAB_PROJECT`。
`A3_SM_SWANLAB_MODE=offline` 保留离线 SwanLab 日志，`disabled` 只记录 JSONL/TensorBoard。
仅 rank 0 创建和上传实验；每卡独立采样，梯度通过 all-reduce 平均。

`RUN_NAME`、`LOG_ROOT`、`TOTAL_ITERATIONS`、`PYTHON` 均可覆盖；额外参数直接传给 tyro：

```bash
A3_SM_SWANLAB_MODE=disabled NPROC_PER_NODE=1 ENVS_PER_GPU=128 \
  ACTOR_BATCH_PER_GPU=128 WM_BATCH_PER_GPU=128 \
  bash scripts/train.sh --schedule.total-iterations 2000
```

改变卡数、batch 或环境数会改变有效训练配比；上例只是小规模运行示例。
四卡默认配方：

| 参数 | 值 |
|---|---:|
| GPU / 每卡环境数 / 全局环境数 | 4 / 4096 / 16384 |
| Actor、WM 每卡 batch / 全局 batch | 2048 / 8192 |
| UTD / 每轮 WM batch 数 / actor 更新数 | 0.4 / 10 / 1 |
| WM-only 预训练 | 500 轮 WM 更新；每轮 10 个 minibatch，actor/critic 冻结 |
| WM 序列长度 k / 想象长度 H / 未来参考 K | 8 / 32 / 10 |
| 基础 LR / 四卡有效初始 LR | 2e-4 / 4e-4（乘 √4） |
| 总 iteration / 日志间隔 / 保存间隔 | 320000 / 50 / 500 |
| Seed | 42，各 rank 加 1000 × rank |

在 DLC 中前台执行，由平台管理生命周期；普通远程机器建议在命名 tmux 中执行。
入口会先检查 GPU 占用和 NCCL 通信，再验证数据/模型并开始训练。
默认设置 `NCCL_P2P_DISABLE=1`、`NCCL_CUMEM_HOST_ENABLE=0`，与该 DLC 配方一致；可按机器覆盖。
`--check-distributed` 仅测试多卡通信，`--print-command` 只打印最终训练命令。

## 模型与算法

- 原生刚性 29 轴 A3 Ultra，头部/手指固定中立位，无弹性踝转子、动作延迟。
  保留标称质量、惯量、碰撞几何、PD、力矩限制和动作缩放。
- 物理 200 Hz，控制 50 Hz，平地，Newton solver / 50 iterations，`mjlab_njmax=1024`。
- 状态 71、动力学预测 70、context 33、动作 29；历史输入 930、目标输入 686（680 维参考序列＋6D 相对姿态）。
- e5 原始 `tanh(MLP(goal × scale))` 编码器、32 维 latent；无 LayerNorm 或辅助重构目标，actor 仅优化 `-J/H`。
- 奖励和目标反馈锚点均为 `torso_Link`；终止检查根部高度偏差 >0.25 m 或重力 z 分量差 >0.8，根部为 `pelvis_link`。
- 目标反馈只包含参考 torso 相对机器人 torso 的旋转矩阵前两列，按行展开；不包含机器人位置误差。
  真机需要躯干姿态估计、IMU 外参和初始航向对齐。参考序列仍包含参考高度/速度，critic 仍使用训练特权状态。
- 想象终止为两个 sigmoid 失败概率的软 OR，高度/倾斜软化宽度分别为 0.02 m / 0.04；真实仿真仍按硬阈值终止。
  WM 不再训练终止 MLP/BCE，只拟合多步动力学状态误差。
- 先采集 40 个环境步填充 replay，再执行 500 轮 WM-only 更新（每轮 10 个 minibatch）；第 501 轮起更新 actor/critic。
  预训练期间继续采集动作探索数据；UTD 保持 0.4。SwanLab 记录 `train/wm_only`、`train/updates`、`train/policy_updates`。
- WM hidden `(2048,2048,2048)`；encoder/decoder hidden `(2048,1024,512,256)`。
- 随机化：启动时躯干 COM、reset 时脚底摩擦与编码器偏置；保留动作/观测噪声和 reset 位姿/速度扰动。无运行中推力。
- 真实环境从参考动作 reset；actor 想象起点来自 replay。Reset 采样为 50% 均匀、40% 当前困难权重、10% 历史困难权重。

A3 与 G1 的机械参数、关节/身体映射不同；并非所有 G1 配置逐项相同。
关节速度观测噪声 ±0.5 rad/s；reset 软限位为腰部 0.8、其他策略关节 0.9。
具体参数以 `src/a3_supermimic/config.py`、`assets/task_base.yaml` 和保存的 `contract.json` 为准。

机器人文件随包提供。可通过 `DEPSIC_A3_ULTRA_MODEL_ROOT` 显式覆盖模型目录；换模型会改变 checkpoint 合约。
来源、版本与独立化修改见 [PROVENANCE.md](PROVENANCE.md)。

## 数据接口

不提供动作数据。每条 NPZ 包含以下字段，50 Hz，单位 m/rad/s，z-up：

| 字段 | 形状 / 含义 |
|---|---|
| `fps` | 标量 50 |
| `joint_names` | 31 个名字：29 策略关节 + 两个头部关节 |
| `joint_pos`, `joint_vel` | `(T,31)`，按名字映射 |
| `root_pos` | `(T,3)`，世界系 |
| `root_quat_wxyz` | `(T,4)`，单位四元数 |
| `root_lin_vel_w` | `(T,3)`，世界系根部线速度 |
| `root_ang_vel_b` | `(T,3)`，根部局部系角速度 |

默认刚性模型读取其中 29 个策略关节。也支持 `library.json` + `root_q.npy`、
`root_v.npy`、`joint_q.npy`、`joint_v.npy` 打包布局，适合大规模多卡加载。
读取器验证关节映射、形状、有限值、四元数和数据哈希。已有 MimicLite qpos 数据可用转换器：

```bash
python -m a3_supermimic.data.convert_corpus convert /path/to/mimiclite /path/to/library --workers 16
a3-sm-data validate /path/to/library
```

最新训练使用筛选后的数据；本仓库只保留转换/筛选代码与数据接口，不隐式下载或绑定数据集。

仅使用筛选后的 LAFAN 训练：见 [数据提取与四卡启动说明](docs/lafan_training.md)。

## 日志、恢复与评测

每次训练保存在 `logs/<RUN_NAME>/`：`config.json`、`contract.json`、`metrics.jsonl`、
`tb/`、`swanlab.json`、`ckpt_latest.pt` 和编号 checkpoint。
SwanLab 回报为多卡聚合，WM/actor/critic 和编码器诊断标量来自 rank 0。

```bash
tensorboard --logdir logs

# 恢复同一数据/物理/网络合约；如需接续同一 SwanLab 实验，沿用原 RUN_NAME 和 LOG_ROOT。
RESUME=/path/to/ckpt_latest.pt RUN_NAME=original_run TOTAL_ITERATIONS=400000 \
  bash scripts/train.sh

a3-sm-eval --checkpoint /path/to/ckpt_latest.pt --npz-dir /path/to/eval_npz \
  --physics rigid --output outputs/evaluation.json --export-policy outputs/actor.pt
```

恢复包含网络、optimizer、采样统计和进度；仿真/replay 会重新初始化并 warmup。
合约版本为 v9，旧 v8 及更早 checkpoint 不兼容；数据、机器人或任务配置不一致也会拒绝恢复。
恢复会保留已完成 WM 更新和策略更新次数；重新填充 replay 不会重新执行已经完成的 500 轮 WM-only。
评测关闭动作/观测/reset 噪声和 domain randomization，并分别报告完整动作长度归一化回报与执行步归一化回报。

## 自检

先检查 `nvidia-smi`，确认有空闲 GPU，再运行：

```bash
python -m pytest tests -q --disable-warnings
python scripts/smoke.py --output /tmp/supermimic_a3_smoke_unique --compile
```

Smoke 生成带明确标记的合成站立数据，测试刚性仿真、模型更新、保存/恢复、评测和 TorchScript 导出。
它仅验证执行路径，不衡量动作跟踪效果。发布验证记录见 [VALIDATION.md](VALIDATION.md)。
