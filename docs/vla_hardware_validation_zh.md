# MARS：PC + AGX Orin 64GB 的真实 CUDA / SmolVLA 测试指南

这份指南新增两项真实 GPU 测试：先运行 CUDA 矩阵计算，确认 GPU 与跨机传输正常；再把公开数据集中的真实相机画面、机器人状态和文字指令送入 **SmolVLA 预训练模型**，在 Orin GPU 上生成动作序列，传回 PC 验证。

SmolVLA 测试的闭环是 **PC 读取观测 → MARS 调度 → Orin CUDA 推理 → PC 检查输出**。它使用真实模型权重和真实记录数据，但不连接机械臂、不执行动作，也不证明模型完成了抓取任务。`smolvla_base` 是用于后续微调的基础模型；这里验收计算和通信，而非机械臂控制效果。[模型说明](https://huggingface.co/lerobot/smolvla_base)

**本文提供待执行的硬件验收流程；代码测试通过不代表已经在你的 AGX Orin 上运行通过。** 原来的 [CPU 导航测试](hardware_validation_zh.md) 仍可单独使用。

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
python3.10 -m venv .venv
.venv/bin/python -m pip install -r agent/requirements-hardware.txt
.venv/bin/python -m agent.main --help
.venv/bin/python -m scripts.vla_loop --help
```

PC 可以使用 Python 3.10 以上版本，替换命令中的解释器即可。下节 Orin 原生安装示例专门使用 Python 3.10。

## 3. Orin：先确定 JetPack，再安装 GPU 环境

AGX Orin 的 64GB 内存足以作为本流程的目标配置，但 **JetPack、CUDA、Python、PyTorch 必须匹配**。目前尚未确认你的 JetPack 版本，不能只凭机器型号选择安装包。

在 Orin 准备终端检查：

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack
python3.10 --version
uname -m
```

`nvidia-jetpack` 元包未安装时，第二条可能没有结果；继续依据 L4T 版本和实际 CUDA 环境判断。不要以 `nvidia-smi` 是否存在作为 Jetson GPU 能否使用的唯一依据。

下面的原生安装组合针对 **JetPack 6.2 系列 / CUDA 12.6、Ubuntu 22.04、Linux aarch64、Python 3.10**：

| 组件 | 本指南使用的版本 |
| --- | --- |
| PyTorch / TorchVision | `2.8.0` / `0.23.0`，Jetson AI Lab 的 `jp6/cu126` 构建 |
| LeRobot | `0.4.4` |
| Transformers | `4.57.1` |

PyTorch 的官方 Torch-TensorRT 文档列出了这个 JetPack 6.2 安装源与 Torch/TorchVision 组合；LeRobot 0.4.4 的依赖范围支持它。[JetPack 安装源](https://docs.pytorch.org/TensorRT/getting_started/jetpack.html) · [LeRobot 0.4.4 依赖](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

如果你的 JetPack 不属于上述组合，先使用 [NVIDIA 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 和 [兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) 准备匹配的 CUDA PyTorch；不要直接套用下方 wheel 安装命令，也不要为此替换系统 Python。

### 3.1 新建独立的 VLA 环境

```bash
cd "$HOME/mars-hardware"
python3.10 -m venv .venv-vla
.venv-vla/bin/python -m pip install --upgrade pip
.venv-vla/bin/python -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
.venv-vla/bin/python -m scripts.install_vla --install
```

最后一条会先实际运行 CUDA 运算和 TorchVision CUDA 算子，再安装固定版本的 LeRobot/Transformers，并锁定已经安装的 Torch/TorchVision，防止依赖解析把 Jetson 版本替换掉。遇到不匹配会报错停止。

**不要把 `agent/requirements-hardware.txt` 和 `agent/requirements-vla.txt` 安装到同一个环境。** MARS 当前使用 protobuf 7；LeRobot 依赖的 WandB 0.24.x 使用 protobuf `<7`。O1 的 Agent 与 GPU 子进程通过本机输入输出交换数据，两个环境可以保持各自的依赖。

已有可用的独立 CUDA 环境时，可以使用它的 Python 执行 `scripts.install_vla --install`，随后把 O1 的 `--worker-python` 改成该解释器绝对路径。不要在 `.venv` 中执行此安装命令。

### 3.2 检查准备结果

```bash
.venv-vla/bin/python -m scripts.install_vla
```

期望输出包含 `"status": "ready"`、实际 GPU 名称、CUDA 版本、`"lerobot": "0.4.4"`、`"transformers": "4.57.1"`。这一步不下载模型，也不替代后续 SmolVLA 测试。

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

启动时会真正执行一次 CUDA 运算，验证模型文件，成功后才公布 GPU/VLA 能力。期望出现 `REAL CUDA VLA`。如果模型缺失、CUDA 不可用或解释器错误，应该修复报错；不会自动降级为 CPU。

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
  --require-distinct-hosts
```

Orin 实际执行完整 FP32 矩阵乘法，检查全部结果，再把结果样本传回 PC；PC 用独立公式验证样本。CUDA 通过后运行真实 VLA：

```bash
.venv/bin/python -m scripts.vla_loop \
  --workload smolvla \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/smolvla-01.json \
  --require-distinct-hosts
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

- 总报告 `status` 为 `succeeded`，最终验证 `valid` 为 `true`，并确认执行来自两个不同主机。
- CUDA 测试中 `smoke` 在 `robot_1`，`validate` 在 `edge_pc`；矩阵参考检查通过。
- SmolVLA 中 `observe`、`validate` 在 `edge_pc`，`infer` 在 `robot_1`；`infer` 和 `validate` 都有大于零的远程输入字节，分别对应观测传入和动作传回。
- GPU 结果记录包含真实设备名称、Torch/CUDA 版本、输入和原始输出 tensor 的 `cuda:0` 位置、正数的 `cuda_event_ms`、`synchronized_wall_ms`、`peak_memory_allocated_bytes`。
- VLA 结果包含固定模型提交、通过验证的权重、模型参数 CUDA 位置、实际参数 dtype、50 × 6 个有限动作值，以及与输入观测对应的 SHA256。
- `missing_camera_keys` 记录缺失的 `camera3`；`physical_actuation` / `robot_control_executed` 为 `false`，符合本测试的计算范围。

报告中的具体位置如下，便于直接查找：

| JSON 位置 | 双机测试的期望 |
| --- | --- |
| `status` / `scope` | `succeeded` / `cross_host_cuda_execution` |
| `gpu_tested` / `executing_host_count` | `true` / `2` |
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
| 模型文件缺失、哈希不匹配、离线下载报错 | 在联网准备终端重新准备完整模型目录，检查磁盘；不要手改清单或跳过校验。 |
| 样本导出时 AV1 / PyAV 解码失败 | 在 `.venv-vla` 中修复 `av` 与 FFmpeg 解码支持；导出成功前不要继续 VLA 测试。 |
| P1 找不到观测文件 | 确认观测 JSON 已复制到 PC，路径相对于 P1 启动时的仓库目录。 |
| GPU 任务没有可用节点 | 确认 O1 使用 `vla-cuda`；SmolVLA 还需有效的 `--model-dir`。普通导航 Agent 不提供这些能力。 |
| 等待结果超时 | 检查 O1 原始报错、模型加载耗时；需要时一起扩大 Agent、任务完成和工作流超时。 |
| 两台主机检查失败 | 确认 O1 真正在 Orin、P1/P2 真正在 PC，且主机名有区别。不要用去掉验收参数掩盖错误部署。 |
| 想使用 PC GPU | PC 也需独立 VLA/CUDA 环境和模型；交换两端 `vla-cuda` / `vla-io` 角色及输入所在位置，P2 加 `--gpu-agent edge_pc`。 |

运行时可在另一个 Orin 终端用 `tegrastats` 辅助观察设备负载，但短任务可能被采样间隔漏掉；以任务生成的 CUDA event 和输出验证为主要验收记录。

完成后，在 O1、P1 各按一次 `Ctrl+C` 正常停止 Agent。保留 `.mars-vla` 中的报告、模型清单、观测来源和任务产物，方便比较后续改动。
