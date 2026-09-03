# MARS：PC + Jetson Orin 硬件闭环操作指南

本页保留轻量 CPU 导航的完整步骤。**新增真实 GPU 测试**请接着使用
[AGX Orin 64GB：CUDA 与 SmolVLA 操作指南](vla_hardware_validation_zh.md)：
先验证 CUDA 矩阵运算，再运行公开 SmolVLA 权重与真实机器人采集样例，回传动作结果。
GPU 流程使用独立的模型环境；不要把 LeRobot 装进本页的通信环境。

本流程让 **PC 上的 MARS 真正派单，PC/Orin 真正计算，再把实际结果交给后续任务**。
不是“等待一下再假装完成”。传感器观测与位置是合成的，建图、规划、碰撞检查、数据传输和耗时是真实的。

本页使用仓库自带的轻量 CPU 导航业务，不需要 DDS、ROS 2、网页、LLM、模型权重或单独的业务服务。
它不是 YOLO/VLA/GPU 测试，也不会驱动电机或控制真实机器人。文档提供待执行的硬件验收步骤，
不代表已经在你的 PC 和 Orin 上实测通过。

## 1. 先看清楚需要几个终端

| 终端 | 在哪里运行 | 用途 | 是否保持运行 |
| --- | --- | --- | --- |
| O1 | Orin | 启动 Orin 的执行 Agent | 是 |
| P1 | PC | 启动 PC 的 Edge Agent | 是 |
| P2 | PC | 启动 MARS 调度器并运行一次测试 | 跑完后自动结束 |

Agent 接单后会自动启动业务子进程，不用再手动打开一个“业务终端”。
每个 Agent 同时最多执行一个任务。

默认 `split` 流程如下，所有箭头都对应实际数据依赖：

```text
Orin：合成测距观测 ──观测数据──> PC：占用栅格建图 ──地图──> PC：轨迹规划
         │                            │                         │
         │ 场景真值                   │ 地图                    │ 轨迹
         └────────────────────────────┴─────────────────────────┘
                                      │
                                      v
                               Orin：独立验证结果
```

建图只读取测距观测，不偷看场景真值；规划使用实际生成的地图；验证检查轨迹是否到达目标、
是否碰撞、速度等是否超限。验证不通过会让任务失败。

## 2. 两台机器分别准备环境，只需首次执行

以下假设两台都是 Linux，仓库放在 `$HOME/mars-hardware`，并已安装 **Python 3.10 或更新版本**。
Ubuntu 20.04 自带的 Python 可能太旧；先准备独立的合适解释器，不要替换 Orin 的系统 Python。

两台机器必须使用同一个已检查的硬件闭环提交。新分支名为 `codex/grpc-hardware-loop`；
可在两台机器分别执行：

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
```

已有仓库就进入实际目录，不要覆盖已有修改。合并以后，改用包含该实现的合并目标分支；
仅有旧 `feature/grpc-runtime` 代码还不够。

在 **PC 和 Orin 各自的准备终端**执行：

```bash
cd "$HOME/mars-hardware"
git rev-parse HEAD
uname -m
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r agent/requirements-hardware.txt
python -m agent.main --help
python -m scripts.hardware_loop --help
```

确认两边 `git rev-parse HEAD` 输出一致。如果使用 Python 3.11/3.12 等，把上面的 `python3.10`
换成实际解释器。安装失败时先解决依赖，不要跳过错误继续启动。
这份依赖已经覆盖本次 Agent、业务和调度器，不需要安装整个网页后端、CUDA 或 TensorRT。

之后每开一个新终端，都需要重新进入仓库并激活 `.venv`。

## 3. 确认局域网地址

以下命令使用示例 IP，执行前全部替换为你的真实地址：

| 机器 | 示例 IP | Agent 名称 |
| --- | --- | --- |
| PC | `192.168.1.10` | `edge_pc` |
| Orin | `192.168.1.20` | `robot_1` |

两台分别查看网络地址：

```bash
ip -br addr
```

PC 测试能否找到 Orin：

```bash
ping -c 3 192.168.1.20
```

Orin 测试能否找到 PC：

```bash
ping -c 3 192.168.1.10
```

两边必须能互相访问 TCP `50051` 端口；启用防火墙时，只允许对方局域网地址访问该端口。
能 ping 通不等于端口已开放。PC 可以同时通过 Wi-Fi 上网、通过网线连接 Orin。

**当前通信未加密、没有身份认证，只能放在可信局域网。不要做公网端口转发。**

## 4. Orin 终端 O1：启动执行 Agent

在 Orin 打开终端。也可以在 PC 新开一个终端，通过 SSH 登录 Orin：

```bash
ssh YOUR_ORIN_USER@192.168.1.20
```

先把 `YOUR_ORIN_USER` 换成 Orin 用户名；登录后，以下命令实际运行在 Orin：

```bash
cd "$HOME/mars-hardware"
source .venv/bin/activate
python -m agent.main \
  --executor navigation \
  --agent-id robot_1 \
  --kind robot \
  --listen 0.0.0.0:50051 \
  --peer edge_pc=192.168.1.10:50051 \
  --artifact-dir .mars-hil/robot_1
```

确认出现 `REAL CPU navigation`，不是 `MOCK`。这里不要再加旧模拟配置的 `--config`。
`--peer` 告诉 Orin 去哪里获取 PC 生成的数据，所以必须填写 **PC 的局域网 IP**。

**保持 O1 运行，不要关闭。**

## 5. PC 终端 P1：启动 Edge Agent

在 PC **另开一个本地终端**，不要误用已经登录 Orin 的 SSH 窗口：

```bash
cd "$HOME/mars-hardware"
source .venv/bin/activate
python -m agent.main \
  --executor navigation \
  --agent-id edge_pc \
  --kind edge \
  --listen 0.0.0.0:50051 \
  --peer robot_1=192.168.1.20:50051 \
  --artifact-dir .mars-hil/edge_pc
```

确认出现 `REAL CPU navigation`。这里的 `--peer` 填 **Orin 的局域网 IP**。
两台不同机器都使用 `50051` 不会冲突。

**保持 P1 运行，不要关闭。**

## 6. PC 终端 P2：启动调度器，运行一次闭环

在 PC **再开一个本地终端**：

```bash
cd "$HOME/mars-hardware"
source .venv/bin/activate
python -m scripts.hardware_loop \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --placement split \
  --seed 19 \
  --output .mars-hil/run-19.json \
  --require-distinct-hosts
```

此处 `edge_pc=127.0.0.1` 正确：调度器就在 PC，直接连接本机 Agent。
但 **O1 的 peer 仍必须用 PC 局域网 IP**；Orin 上的 `127.0.0.1` 指 Orin 自己。

这条命令直接运行 `CentralCoordinator`：派单、接收实际完成结果、推动后续任务，最后保存报告。
不需要启动网页或另外再开一个 Coordinator 进程。

`--require-distinct-hosts` 会比较实际执行任务所报告的主机名和架构，防止把“两台机器测试”
误跑成“本机两个 Agent”。它是配置检查，不是密码学硬件认证。

本次结束前让 O1、P1 持续运行。下次测试必须换一个未存在的输出文件名，如 `run-20.json`；
程序不会覆盖旧报告，包括失败报告。

## 7. 查看结果，判断是否真正跑通

P2 运行结束后执行：

```bash
python -m json.tool .mars-hil/run-19.json
```

至少确认：

- `status` 是 `succeeded`，`error` 是 `null`。
- `scope` 是 `cross_host_cpu_execution`，`executing_host_count` 是 `2`。
- `executions` 中，`sense`/`validate` 在 `robot_1`，`map`/`plan` 在 `edge_pc`。
- `validation.valid` 是 `true`，输入哈希对应本次地图、轨迹和真值。
- `remote_input_bytes` 大于零；`map` 和 `validate` 的记录都有远程输入字节数，证明两个方向都传过真实数据。
- `worker_elapsed_ms` 等耗时是实测值，`artifacts` 包含真实结果和校验和。

业务输出保存在各自 Agent 的 `.mars-hil/robot_1`、`.mars-hil/edge_pc` 中；
调度器收集的证据保存在 PC 的 `.mars-hil/received-artifacts/` 中。

读数注意：

- `executions[].host_observations.before` / `after` 是任务前后的 **整台机器** CPU/内存观测，不是业务独占用量。
- CPU 按至少 100 毫秒的窗口采样；更密集的请求复用带原始时间戳的读数，因此短任务前后数值可能相同。实际窗口见 `cpu_sample_window_ms`，并非整项任务的平均或峰值。
- `final_node_observations` 保存最后一次反馈给调度器的节点状态；即使任务因资源不足没有启动，也可用它检查 CPU/内存情况。
- `worker_elapsed_ms` 包含业务子进程启动、输入输出和计算；`total_wall_elapsed_ms` 还包含收集证据等步骤。
- 规划出的“运动时长”是数学结果，不表示机器人真的动了那么久。
- 链路带宽、调度性能配置仍有估计值，见 `planning_assumptions`。
- 当前未测 GPU、功耗或能耗；`energy_j: null` 表示没测，不是零耗电。不要把旧协议占位零值当成实测。

可选：若想实时看 Orin 状态，在 Orin **另开终端 O2**：

```bash
tegrastats --interval 1000
```

如果系统提供该工具，就会持续显示机器状态；它不是测试必需项，也不会自动写入本次 MARS 报告。

## 8. 换一种任务分配方式再测

保持 O1、P1 运行，在 P2 重新执行第 6 步，改变 `--placement` 并使用新的输出文件名：

| 选项 | 实际分配 |
| --- | --- |
| `split` | Orin 生成观测/验证，PC 建图/规划 |
| `orin` | 四个任务全部由 Orin 执行 |
| `edge` | 四个任务全部由 PC 执行 |
| `auto` | 观测/验证仍在 Orin，建图/规划由调度器选择 |

**使用 `orin` 或 `edge` 时，删除 `--require-distinct-hosts` 参数**，因为你有意只用一台机器执行。
按本指南继续登记两个 Agent 即可；`orin` 也支持只登记 `robot_1`，但 `edge` 仍需要登记源节点 `robot_1`。

`auto` 不保证一定用到 PC；如果允许它全部放在 Orin，也删除 `--require-distinct-hosts`。
固定分配跑通，只证明执行和数据闭环可用，不证明该分配方案更快或更省电。

如果暂时没有 Orin，可以先按[英文指南中的本机三终端步骤](hardware_validation.md#8-repeat-placements-or-test-locally-first)
使用不同端口跑两个 Agent。那种结果会是 `same_host_cpu_execution`，不能当成 PC + Orin 已实测通过。

## 9. 失败时检查什么

| 现象 | 优先检查 |
| --- | --- |
| 显示 `MOCK` | 是否加了 `--executor navigation`；是否误用旧分支。 |
| 拒绝连接或连接超时 | O1/P1 是否还在运行、实际 IP、`50051` 端口、防火墙。 |
| 能接单但拿不到输入 | 两边 `--peer` 是否指向对方的局域网地址。 |
| 主机数量检查失败 | O1 是否真的运行在 Orin，不要靠删除参数假装硬件验证成功。 |
| 类型/端口/schema 不匹配 | 两边提交是否一致；是否误用了网页生成的模拟工作流。 |
| 业务计算或轨迹验证失败 | 保留报告及 O1/P1 日志，查看具体错误，不要改成假成功。 |
| `no_feasible_agent` | 查看 `final_node_observations`、机器负载和任务约束；真实满载仍可能让测试无法派单。 |
| 输出文件已存在 | 换新的 `--output` 文件名。 |
| `attempt_history_full_restart_agent` | Agent 已达到 1,024 次尝试的内存记录上限；结束本轮后重启 Agent。 |

Agent 每任务默认限制 30 秒，可由 `--task-timeout` 调整；调度器整轮默认限制 120 秒，
参数为 `--workflow-timeout`，单次等待完成还受 60 秒上限约束。先看日志确定原因，不要反复提交或直接放大所有超时。

## 10. 结束测试

1. 正常情况先等 P2 跑完，保存报告。
2. 在 P1 按 `Ctrl+C`，停止 PC Agent。
3. 在 O1 按 `Ctrl+C`，停止 Orin Agent。
4. 若开了监控，在 O2 按 `Ctrl+C`。

Agent 正常退出会取消自己尚未结束的业务子进程。如果中途打断 P2，也检查并停止两个 Agent，
不要把“关闭网页或终端”当作远程任务已经取消。保留 `.mars-hil` 数据供分析，不要在 Agent 运行时删除它。

本次验收范围是 **两台真实机器上的调度、计算、传输和反馈闭环**，不包括真实传感器、物理控制、
生产安全认证或调度优越性证明。算法边界及参考来源见[英文指南](hardware_validation.md#design-references-and-scope)。
