# MARS：PC + AGX Orin 64GB 的真实 CUDA / SmolVLA 测试指南

> 先核对版本：NVIDIA 官方没有 **JetPack 7.1.2**。AGX Orin 的 JetPack 7 支持从
> **7.2 / L4T 39.2** 开始；本文目标是 **JetPack 7.2.1 / L4T 39.2.1**。
> 如果 `/etc/nv_tegra_release` 不是 `R39, REVISION: 2.1`，不要把口头版本号当作通过。
> 先完成 [CPU/GPU 混合任务指南](hardware_validation_zh.md) 的原生 CUDA 闭环，再做本页的
> PyTorch 和模型测试。[NVIDIA JetPack 版本归档](https://developer.nvidia.com/embedded/jetpack-archive)

这份指南新增两项真实 GPU 测试：先运行 CUDA 矩阵计算，确认 GPU 与跨机传输正常；再把公开数据集中的真实相机画面、机器人状态和文字指令送入 **SmolVLA 预训练模型**，在 Orin GPU 上生成动作序列，传回 PC 验证。

SmolVLA 测试的闭环是 **PC 读取观测 → MARS 调度 → Orin CUDA 推理 → PC 检查输出**。它使用真实模型权重和真实记录数据，但不连接机械臂、不执行动作，也不证明模型完成了抓取任务。`smolvla_base` 是用于后续微调的基础模型；这里验收计算和通信，而非机械臂控制效果。[模型说明](https://huggingface.co/lerobot/smolvla_base)

**本文提供待执行的硬件验收流程；代码测试通过不代表已经在你的 AGX Orin 上运行通过。** 原来的 [CPU 导航测试](hardware_cpu_validation_zh.md) 仍可单独使用。

## 1. 运行位置与终端

| 终端 | 机器 | 工作 | Python 环境 |
| --- | --- | --- | --- |
| 准备终端 | Orin | 安装 GPU 依赖、下载模型、导出观测 | `.venv-vla` |
| O1 | Orin | GPU 执行 Agent，接单后自动启动推理子进程 | Agent 用 `.venv`，子进程用 `.venv-vla` |
| P1 | PC | 读取观测、验证返回结果 | `.venv` |
| P2 | PC | MARS 调度器，发起一次测试并保存报告 | `.venv` |

PC 不需要 NVIDIA GPU、PyTorch 或 LeRobot。运行测试时只需保持 O1、P1 两个 Agent，P2 每次运行结束就退出。

```text
PC：真实记录观测 ── 图像 + 6 维状态 + 指令 ──> Orin：SmolVLA CUDA 推理
        │                                                  │
        └──────────────> PC：验证 <──── 50 × 6 动作序列 + GPU 记录
```

示例假设两台机器都是 Linux，仓库位于 `$HOME/mars-hardware`，PC IP 为 `192.168.1.10`，Orin IP 为 `192.168.1.20`。执行前替换全部示例 IP 和 SSH 用户名。两端 TCP `50051` 必须互通；沿用 CPU 指南中的可信局域网要求。

## 2. 两台机器：检查代码，建立 Agent 环境

两台都必须使用**包含本指南和 `scripts/vla_loop.py` 的同一提交**。仅有最初 CPU 版本的 PR 分支还不够。

首次克隆：

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
cd "$HOME/mars-hardware"
git rev-parse HEAD
```

已有仓库时进入实际目录，更新到包含 GPU/VLA 改动的提交；保留已有修改。确认两台 `git rev-parse HEAD` 完全一致，并存在 `scripts/vla_loop.py`。

两台分别建立轻量 Agent 环境；如果 CPU 测试已经建好 `.venv`，直接使用即可：

```bash
cd "$HOME/mars-hardware"
python3 -m venv .venv
.venv/bin/python -m pip install -r agent/requirements-hardware.txt
.venv/bin/python -m agent.main --help
.venv/bin/python -m scripts.vla_loop --help
```

JetPack 7.2.1 的系统 Python 是 3.12；PC 也建议使用 3.12。Agent 与 VLA worker 仍须使用两个独立环境。

## 3. Orin：确认实际平台，再建立 VLA 环境

AGX Orin 64GB 的内存满足本测试，但版本必须以设备输出为准。在 Orin 执行：

```bash
tr -d '\000' < /proc/device-tree/model
printf '\n'
cat /etc/nv_tegra_release
cat /etc/os-release
dpkg-query -W -f='${db:Status-Status} ${Version}\n' nvidia-l4t-core nvidia-jetpack 2>&1 || true
python3 --version
/usr/local/cuda/bin/nvcc --version
uname -m
```

继续本页前应看到：设备型号含 `Jetson AGX Orin`，L4T 为 `R39`、`REVISION: 2.1`，Ubuntu 24.04，Python 3.12，CUDA 编译器为 13.2 系列，架构为 `aarch64`。`nvidia-jetpack` 元包可能未安装，所以它只作辅助；L4T 是本页的平台事实来源。`nvidia-smi`、软件包候选版本或手写的“7.1.2”都不能替代这些输出。

NVIDIA 的 JetPack 7.2.1 页面给出的系统栈是 L4T 39.2.1、Ubuntu 24.04 和 CUDA 13.2.1，并明确列出 Orin Family。[JetPack 7.2.1 版本说明](https://developer.nvidia.com/embedded/jetpack/downloads)

### 3.1 兼容组合与支持边界

本文保持已经过代码级核对的 SmolVLA API，并给出以下可复现候选组合；下方安装命令会固定安装这些版本：

| 组件 | 本页命令安装值 |
| --- | --- |
| Python | `3.12.x` |
| PyTorch / TorchVision | `2.10.0` / `0.25.0`，CUDA 13.0 aarch64 构建 |
| LeRobot | `0.4.4` |
| Transformers | `4.57.1` |
| 目标 GPU | AGX Orin，compute capability `8.7`，Torch 构建含 `sm_87` |

LeRobot 0.4.4 要求 Torch `<2.11`、TorchVision `<0.26`，所以不能让安装器升级成 Torch 2.11 或更高。[LeRobot 0.4.4 依赖](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

截至本文更新日，NVIDIA 的 PyTorch for Jetson 兼容表尚未列出 JetPack 7.2/7.2.1 的正式 wheel 组合，NVIDIA wheel 一栏也没有 7.x wheel。因此下方使用 **Jetson AI Lab 社区索引**的 `sbsa/cu130` aarch64 构建，属于需要在本机通过严格门禁的候选组合，不能只凭安装成功称为 NVIDIA 官方认证。[NVIDIA PyTorch for Jetson 兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)

### 3.2 新建独立 VLA 环境并执行 GPU 门禁

```bash
cd "$HOME/mars-hardware"
python3 -m venv .venv-vla
.venv-vla/bin/python -m pip install --upgrade pip
.venv-vla/bin/python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
.venv-vla/bin/python -m scripts.install_vla \
  --install \
  --require-python 3.12 \
  --require-compute-capability 8.7
```

最后一条会在安装 LeRobot 前后各执行真实 CUDA 矩阵运算和 TorchVision CUDA NMS，要求设备能力为 `[8, 7]`、Torch 架构列表含 `sm_87`，并用约束文件保持已安装的 Torch/TorchVision 不被替换。任一步失败都应停止；不要改用 CPU wheel、删掉版本门禁或把错误结果写成 GPU 通过。

**不要把 `agent/requirements-hardware.txt` 和 `agent/requirements-vla.txt` 安装到同一个环境。** MARS 当前使用 protobuf 7；LeRobot 0.4.4 的依赖树要求较旧 protobuf。O1 的 Agent 与 GPU worker 通过受限的本机标准输入输出通信，因此两个环境可以各自保持依赖。

已有独立 CUDA 环境时，也必须使用它的 Python 执行同一条 `scripts.install_vla` 检查，并把 O1 的 `--worker-python` 指向该解释器。该环境可以使用 LeRobot 0.4.4 声明范围内的其他 Torch/TorchVision 版本，但报告必须保存实际版本，不能把它描述为上表的精确基线。不要在 Agent 的 `.venv` 中安装模型依赖。

### 3.3 再次只读检查

```bash
.venv-vla/bin/python -m scripts.install_vla \
  --require-python 3.12 \
  --require-compute-capability 8.7
```

期望 JSON 同时包含 `"status": "ready"`、`"python": "3.12..."`、`"compute_capability": [8, 7]`、`"sm_87"`、实际 GPU 名称、Torch/CUDA 版本、`"lerobot": "0.4.4"` 和 `"transformers": "4.57.1"`。这一步不下载模型；完整权重加载和动作推理仍由后续闭环完成。

如果这一候选 PyTorch 构建在你的实际系统上没有通过门禁，原生 CUDA 混合闭环仍可独立执行，但不得把 SmolVLA 标记为已就绪。保留完整报错，以便换用后来获得实机验证的 PyTorch 环境。

## 4. Orin：下载固定模型，导出真实观测

**准备阶段需要互联网；Agent 的推理子进程只读取本地文件，运行期间禁止自动下载。** 首次模型下载约 0.91GB，加少量 VLM 配置与 tokenizer；样本源视频约 470MB。建议为依赖、缓存与报告预留数 GB 可用磁盘空间。

```bash
cd "$HOME/mars-hardware"
.venv-vla/bin/python -m scripts.prepare_vla model \
  --output .mars-vla/model
.venv-vla/bin/python -m scripts.prepare_vla sample \
  --output .mars-vla/observation.json \
  --cache .mars-vla/datasets
```

模型准备命令会下载并记录每个文件的 SHA256。运行时会检查文件、清单和模型版本，再严格加载全部权重；不使用随机初始化权重替代缺失权重。使用以下固定版本：

| 内容 | Hugging Face 仓库 | 固定提交 |
| --- | --- | --- |
| SmolVLA 策略 | `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| VLM 配置与 tokenizer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| 真实观测数据 | `lerobot/svla_so100_pickplace` | `728583b5eaf9e739a7f119e2def466fa1d552402` |

样本使用公开 SO100 数据集的第 0 个 episode、第 0 帧。它保留真实的 6 维关节状态与原始文字任务，把两个实际相机画面按比例缩小到最长边 256 像素并保存为 PNG。因为数据集把多个 episode 合并到视频文件中，导出一帧仍需下载约 470MB 源文件。[数据集](https://huggingface.co/datasets/lerobot/svla_so100_pickplace)

相机映射为 `top → observation.images.camera1`、`wrist → observation.images.camera2`。基础模型配置另有 `camera3`；样本没有第三个视角，因此报告会记录缺失，而不会复制或生成第三张图片。LeRobot 在 `empty_cameras=0` 时支持只使用实际提供的相机画面。[相机处理实现](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)

6 维状态顺序为肩部旋转、肩部抬升、肘部、腕部俯仰、腕部旋转、夹爪；具体源字段名写在观测 JSON 的 `provenance.state_joint_order`。这些数值不会被替换成零或随机状态。

### 4.1 把观测文件复制到 PC

在 **PC 本地终端**执行，先替换 Orin 用户名和地址：

```bash
cd "$HOME/mars-hardware"
mkdir -p .mars-vla
scp YOUR_ORIN_USER@192.168.1.20:~/mars-hardware/.mars-vla/observation.json \
  .mars-vla/observation.json
```

只有观测 JSON 需要复制到 PC。模型目录和原始视频保留在 Orin。复制目标已有观测时，先确认这是你要替换的输入；报告输出请始终使用新的文件名。

## 5. O1：启动 Orin GPU Agent

先停止占用 `50051` 的旧 CPU 导航 Agent。然后在 Orin 执行：

```bash
cd "$HOME/mars-hardware"
.venv/bin/python -m agent.main \
  --executor vla-cuda \
  --agent-id robot_1 \
  --kind robot \
  --listen 0.0.0.0:50051 \
  --peer edge_pc=192.168.1.10:50051 \
  --worker-python "$PWD/.venv-vla/bin/python" \
  --model-dir .mars-vla/model \
  --task-timeout 300 \
  --artifact-dir .mars-vla/robot_1
```

启动时会在 `--worker-python` 中执行 CUDA 求和、TorchVision CUDA NMS，并导入固定版本的 LeRobot/Transformers；同时验证模型清单，全部成功后才公布 GPU/VLA 能力。期望出现 `REAL CUDA VLA`。完整权重仍在收到推理任务时严格加载；任一检查失败都不会降级到 CPU。

`--peer` 必须填 **PC 的局域网 IP**。`--worker-python` 必须指向 **Orin 上的 GPU 环境**。**保持 O1 运行。**

如果暂时只想验收 CUDA 矩阵计算，可以省略 `--model-dir`，并跳过第 4 节的模型与样本准备；这样的 Agent 不会接受 SmolVLA 任务。

## 6. P1：启动 PC 输入与验证 Agent

在 PC 新开本地终端：

```bash
cd "$HOME/mars-hardware"
.venv/bin/python -m agent.main \
  --executor vla-io \
  --agent-id edge_pc \
  --kind edge \
  --listen 0.0.0.0:50051 \
  --peer robot_1=192.168.1.20:50051 \
  --observation-file .mars-vla/observation.json \
  --artifact-dir .mars-vla/edge_pc
```

期望出现 `REAL CPU VLA input/validation`。`--peer` 必须填 **Orin 的局域网 IP**。**保持 P1 运行。** 只跑 CUDA 矩阵测试时，可以省略 `--observation-file`。

## 7. P2：先运行 CUDA，再运行 SmolVLA

在 PC 再开一个本地终端。先运行没有模型依赖的 CUDA 测试：

```bash
cd "$HOME/mars-hardware"
.venv/bin/python -m scripts.vla_loop \
  --workload cuda \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/cuda-01.json \
  --require-distinct-hosts \
  --require-hardware \
  --require-jetpack 7.2.1
```

Orin 实际执行完整 FP32 矩阵乘法，检查全部结果，再把结果样本传回 PC；PC 用独立公式验证样本。CUDA 通过后运行真实 VLA：

```bash
.venv/bin/python -m scripts.vla_loop \
  --workload smolvla \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/smolvla-01.json \
  --require-distinct-hosts \
  --require-hardware \
  --require-jetpack 7.2.1
```

调度器运行在 PC，因此这里 `edge_pc=127.0.0.1` 正确；O1 的 `--peer` 仍应使用 PC 局域网 IP。

默认先预热 1 次，再测量 3 次完整推理。每次都会调用 `predict_action_chunk` 重新计算 50 个动作，每个动作 6 维；不会把 `select_action` 缓存里取出的动作算成一次 GPU 推理。最后一次的动作经过模型配套的反归一化处理后传回 PC。[推理接口](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py) · [处理器](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/processor_smolvla.py)

每次接单都会新建子进程并加载模型。因此，这是可重复的首次加载与推理验收，不是常驻模型服务的最高吞吐测试。默认单任务完成等待 300 秒、工作流等待 600 秒；需要扩大时，同时调整 O1 的 `--task-timeout` 以及 P2 的 `--task-completion-timeout`、`--workflow-timeout`。例如分别设为 `600`、`600`、`1200`。

后续运行使用新报告名，例如 `smolvla-02.json`；程序保留旧报告，不会覆盖失败或成功记录。

## 8. 如何判定成功

```bash
.venv/bin/python -m json.tool .mars-vla/cuda-01.json
.venv/bin/python -m json.tool .mars-vla/smolvla-01.json
```

不要只看“Agent 启动成功”或日志里出现 `cuda`。完整验收应同时满足：

- 总报告 `status` 为 `succeeded`、`hardware_smoke_passed` 和 `gpu_tested` 均为 `true`，最终验证 `valid` 为 `true`。
- `execution_evidence_kind` 必须是 `trusted_agent_report`；`hardware_gate_failures` 必须为空。软件 fixture 即使业务和传输成功，也只能得到 `test_fixture`，不能再被计作真实 GPU。
- 两台机器必须有不同的 `/etc/machine-id` 哈希、相同提交和相同运行源码指纹；GPU 主机必须是 aarch64 AGX Orin、compute capability 8.7、L4T 39.2.1。
- Worker 必须是 Python 3.12；Torch/TorchVision 必须在 LeRobot 0.4.4 的受支持范围内，且启动预检记录的版本必须与实际推理完全一致。
- CUDA 测试中 `smoke` 在 `robot_1`，`validate` 在 `edge_pc`；矩阵参考检查通过。
- SmolVLA 中 `observe`、`validate` 在 `edge_pc`，`infer` 在 `robot_1`；`infer` 和 `validate` 都有大于零的远程输入字节，分别对应观测传入和动作传回。
- GPU 结果记录包含真实设备名称、Torch/CUDA 版本、输入和原始输出 tensor 的 `cuda:0` 位置、正数的 `cuda_event_ms`、`synchronized_wall_ms`、`peak_memory_allocated_bytes`。
- VLA 结果包含固定模型提交、通过验证的权重、模型参数 CUDA 位置、实际参数 dtype、50 × 6 个有限动作值，以及与输入观测对应的 SHA256。
- `missing_camera_keys` 记录缺失的 `camera3`；`physical_actuation` / `robot_control_executed` 为 `false`，符合本测试的计算范围。

报告中的具体位置如下，便于直接查找：

| JSON 位置 | 双机测试的期望 |
| --- | --- |
| `status` / `scope` | `succeeded` / `cross_host_cuda_execution` |
| `hardware_smoke_passed` / `gpu_tested` | `true` / `true` |
| `execution_evidence_kind` / `executing_host_count` | `trusted_agent_report` / `2` |
| `hardware_gate_failures` | `[]` |
| `jetpack_profile_evidence.passed` | `true`，`profile_scope` 为 `l4t_only`，观测到 L4T R39 revision 2.1；系统 CUDA Runtime 已由前置原生 CUDA 闭环另行验收 |
| `validation.valid` | `true` |
| `gpu_execution.agent_id` | 默认 `robot_1` |
| `gpu_execution.measurement` | GPU 名称、设备、CUDA event、同步耗时、内存分配 |
| `gpu_execution.action_shape` | SmolVLA 为 `[50, 6]` |
| `gpu_execution.model` | 固定提交、权重校验与严格加载记录 |
| `executions` | 各任务主机、执行模式与 `remote_input_bytes` |
| `observation_source` | SmolVLA 数据集提交、episode、帧和相机映射 |
| `artifacts[].envelope.payload` | 完整观测、动作或验证内容；动作包含 `missing_camera_keys` |
| `physical_actuation` / `control_success_tested` | 均为 `false` |

CUDA event 记录 GPU stream 上的时间，`synchronized_wall_ms` 记录同步后的整次调用时间；模型加载及预处理时间单独记录。它们不包含完整跨机工作流耗时。PyTorch 的显存分配数也不是 Jetson 全机内存占用，不能直接拿来当系统总内存。

报告中的调度 profile 是启动测试用的估计值；本次 GPU event、同步耗时和分配内存来自实际执行。此流程不测量能耗，也不证明某种调度策略优于另一种。

## 9. 常见问题

| 现象 | 处理方法 |
| --- | --- |
| `torch.cuda.is_available()` 为 false | 确认检查的是 `.venv-vla`；按实际 JetPack 安装 CUDA 版本的 Torch。不要用普通 CPU wheel 替代。 |
| TorchVision CUDA 算子失败 | Torch/TorchVision 构建不匹配；在独立 VLA 环境修复配套版本，再执行 `scripts.install_vla`。 |
| protobuf 依赖冲突 | Agent 和 LeRobot 装进了同一环境；重新建立独立 `.venv-vla`，O1 通过 `--worker-python` 调用。 |
| `worker package lookup is not isolated` | 当前 shell 设置了 `PYTHONHOME`、`PYTHONPATH` 或启用了用户 site-packages；先 `unset PYTHONHOME PYTHONPATH`，确认使用 `.venv-vla/bin/python`，再重跑安装门禁。 |
| 模型文件缺失、哈希不匹配、离线下载报错 | 在联网准备终端重新准备完整模型目录，检查磁盘；不要手改清单或跳过校验。 |
| 样本导出时 AV1 / PyAV 解码失败 | 在 `.venv-vla` 中修复 `av` 与 FFmpeg 解码支持；导出成功前不要继续 VLA 测试。 |
| P1 找不到观测文件 | 确认观测 JSON 已复制到 PC，路径相对于 P1 启动时的仓库目录。 |
| GPU 任务没有可用节点 | 确认 O1 使用 `vla-cuda`；SmolVLA 还需有效的 `--model-dir`。普通导航 Agent 不提供这些能力。 |
| 等待结果超时 | 检查 O1 原始报错、模型加载耗时；需要时一起扩大 Agent、任务完成和工作流超时。 |
| `JetPack 7.1.2 is not an NVIDIA release` | 读取 `/etc/nv_tegra_release`；若是 `R39 / REVISION: 2.1`，命令应写 `--require-jetpack 7.2.1`。若不是，按实际官方版本重新选择流程。 |
| 两台主机检查失败 | 确认 O1 真正在 Orin、P1/P2 真正在 PC，两边 `git rev-parse HEAD` 完全相同；容器运行 Agent 时还要正确提供宿主的机器标识。不要删除硬件验收参数。 |
| `no_test_fixtures` 失败 | 当前结果来自软件测试样例，不能作为 GPU 实机证据；必须在真实 Orin worker 上重跑。 |
| `cuda_preflight_matches_execution` 失败 | 启动预检与实际任务使用了不同设备或 Torch/CUDA/Python/模型依赖；核对 `--worker-python` 和运行环境后重启 O1。 |
| `worker_python_for_jetpack` / `smolvla_framework_versions` 失败 | 使用 Python 3.12 的独立 worker，并恢复 LeRobot 0.4.4 支持的 Torch/TorchVision 组合；重新执行第 3 节门禁。 |
| 想使用 PC GPU | PC 也需独立 VLA/CUDA 环境和模型；交换两端 `vla-cuda` / `vla-io` 角色及输入所在位置，P2 加 `--gpu-agent edge_pc`。 |

运行时可在另一个 Orin 终端用 `tegrastats` 辅助观察设备负载，但短任务可能被采样间隔漏掉；以任务生成的 CUDA event 和输出验证为主要验收记录。

完成后，在 O1、P1 各按一次 `Ctrl+C` 正常停止 Agent。保留 `.mars-vla` 中的报告、模型清单、观测来源和任务产物，方便比较后续改动。
