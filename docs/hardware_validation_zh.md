# MARS：PC + AGX Orin 64GB 的 CPU / 原生 CUDA 硬件闭环指南

[English](hardware_validation.md) · [原 CPU-only 中文指南](hardware_cpu_validation_zh.md) · [原 CPU-only 英文指南](hardware_cpu_validation.md)

本指南从以下**现有状态**开始：你有原版 **Jetson AGX Orin 64GB Developer Kit**，已成功刷入 **JetPack 7.2.1**；PC 是 **Ubuntu 24.04、x86_64**；两台连接同一 Wi-Fi，没有网线。接下来是在已安装的系统中配置并测试，**不需要再启动 SDK Manager、不需要进入 Recovery、不需要重新刷机**。

NVIDIA 官方下载页列出的本次基线是 **JetPack 7.2.1 / Jetson Linux（L4T）39.2.1 / Ubuntu 24.04 / CUDA 13.2.1**，支持 Orin 系列。实际安装版本仍须在 Orin 上检查。[NVIDIA JetPack 下载与版本说明](https://developer.nvidia.com/embedded/jetpack/downloads)

**本文是待在你的两台机器上执行的操作与验收指南，不是已经在你的 Orin 上通过测试的记录。** 只有实际生成的报告、任务产物及执行日志满足后文验收要求，才能记录“本次硬件测试通过”。

本次不安装 ROS、模型、PyTorch、LeRobot，不连接电机，也不运行车辆控制。只需要轻量 Python 通信环境和 Orin 已安装的原生 CUDA 工具链。旧 [CUDA / SmolVLA 指南](vla_hardware_validation_zh.md) 可以留作**之后单独进行模型推理**的参考；其中 JetPack 6.2 / CUDA 12.6 / Python 3.10 / PyTorch 安装组合不适用于本页的 JetPack 7.2.1，不能直接照搬，也不是本次测试的前置步骤。归档 CPU 指南保留了旧版本说明，当前环境以本页为准。

## 1. 先理解这次实际运行什么

### 1.1 五个任务、六份产物、八条数据边

MARS 的协调器在 PC 上派单，两个 Agent 收单后启动真实业务子进程。固定分工如下；`robot_1` 和 `edge_pc` 是程序标识，不是 Linux 登录用户名。

| 顺序 / 任务 ID | 执行机器 | 实际工作 | 输出产物 |
| --- | --- | --- | --- |
| 1 `sense` | Orin CPU | 在已知位置进行合成二维测距巡测，生成带有限噪声的观测及独立场景真值 | `observations`、`truth` |
| 2 `map` | PC CPU | 仅用观测建立自由、占据、未知栅格，不读取真值障碍物 | `map` |
| 3 `inflate` | Orin GPU，原生 CUDA | 按机器人半径、噪声余量和栅格几何，对占据及未知区域计算完整膨胀禁行掩码 | `inflated` |
| 4 `plan` | PC CPU | **直接消费收到的 GPU 掩码**进行四邻接 A*，生成停车、旋转、直线运动轨迹 | `trajectory` |
| 5 `validate` | Orin CPU | 独立重算完整 CPU 掩码，逐格比较 GPU 结果，再检查轨迹全程连续几何碰撞与运动限制 | `validation` |

主链为：

```text
Orin CPU sense → PC CPU map → Orin CUDA inflate → PC CPU plan → Orin CPU validate
```

主链之外还有后续任务需要的输入。完整八条边如下，一份产物可以被多个任务消费，因此不是八份不同产物：

| 边 | 产物来源 → 消费者 | 是否跨 Wi-Fi |
| --- | --- | --- |
| 1 | `sense.observations` → `map.observations` | Orin → PC |
| 2 | `map.map` → `inflate.map` | PC → Orin |
| 3 | `map.map` → `plan.map` | PC 本机 |
| 4 | `inflate.inflated` → `plan.inflated` | Orin → PC |
| 5 | `map.map` → `validate.map` | PC → Orin |
| 6 | `inflate.inflated` → `validate.inflated` | Orin 本机 |
| 7 | `plan.trajectory` → `validate.trajectory` | PC → Orin |
| 8 | `sense.truth` → `validate.truth` | Orin 本机 |

每份产物都有内容校验值；远端消费者通过另一台 Agent 下载并校验真实 JSON 数据。两台机器都叫 `~/mars-hardware` **不意味着共享目录**，无需设置共享磁盘或 Dropbox 同步。边上传输的是实际产物，不是把另一台机器的文件路径直接当作本地路径。

### 1.2 哪些是模拟，哪些是真实

- 模拟的是传感器输入和已知位姿：当前场景为 12 × 8 米、96 × 64 栅格；20 个已知巡测点各产生 256 条测距射线。没有接入摄像头、激光雷达或真实定位；它不是 SLAM，也没有真实移动采集过程。
- 真实的是 CPU 建图、CUDA 内核、A* 搜索、CPU 独立校验、两台机器之间的网络传输和执行计时。PC 规划不能在 GPU 失败后用 CPU 掩码替换结果来通过。
- 最终检查包括完整 **6,144 格** GPU/CPU 掩码一致、输入来源一致、起终点、位姿和时间连续、沿整段轨迹的连续几何碰撞、地图已观测空间间隙、速度、加速度及偏航速率。不是只检查几个路径点。
- 协调器收齐产物后还在 PC 上独立复核返回的验证内容。这是证据复核，不额外增加第六个 DAG 业务任务。
- 通过说明这一套真实 CPU/GPU 跨机计算链条正确工作；不说明机器人完成导航，不证明调度最优、GPU 性能领先或工业安全认证。

## 2. 接线、开机、登录和终端位置

### 2.1 Orin 放在桌上正常开机

1. 将开发套件放在通风、平稳的桌面，让散热器和风扇周围留出空间；接好套件合适的电源。
2. 如使用本机桌面，在 Orin 的显示输出接好兼容显示器，将 USB 键盘和鼠标插到 Orin；显示器选择对应输入源。具体接口以你的套件接口标识为准。
3. 若机器已进入 Ubuntu 桌面，直接继续。若关机，使用正常电源键开机。**不要按 Recovery 键，不要进行刷机按键组合。**
4. 用刷机后首次开机设置的 Orin 用户名和密码登录 Ubuntu。PC 的登录密码与 Orin 的密码可能不同。
5. 在 Orin 桌面右上角网络菜单连接与 PC 相同的 Wi-Fi。也在 PC 确认连接成功。全程可以只使用 Wi-Fi；刷机时用的 USB 数据线不承担本指南的业务网络。
6. 在 Orin 桌面按 `Ctrl+Alt+T` 打开终端，将它视为 **O0（Orin 准备终端）**。在 PC 同样打开终端，视为 **P0（PC 准备终端）**。

如果 Orin 没有可用无线接口，先解决该设备的 Wi-Fi 连接，必要时使用已兼容并正常工作的无线适配器；不要把 USB 虚拟网络的地址当作无线地址。

### 2.2 先记住终端分工

| 标签 | 命令实际执行在哪里 | 用途 | 测试时是否保持运行 |
| --- | --- | --- | --- |
| O0 | Orin | 查看系统、网络、安装、编译；之后也可检查 PC 端口 | 按需使用 |
| P0 | PC | 查看网络、安装；可另开 SSH 连接 | 按需使用 |
| O1 | **Orin**，本机终端或从 PC SSH 登录进去的终端 | `robot_1` Agent，CPU + CUDA | **保持运行** |
| P1 | **PC 本机**，不是 SSH 会话 | `edge_pc` Agent，CPU 建图 / 规划 | **保持运行** |
| P2 | **PC 本机**，新的终端 | 一次提交三轮测试、查看报告 | 命令结束后返回提示符 |

如果用 PC SSH 操作 Orin，PC 屏幕上会有三个主要窗口：O1（SSH）、P1（本机）、P2（本机）。**窗口在 PC 上显示，不代表命令在 PC 上执行。** `hostname` 和 `uname -m` 用于确认位置。

### 2.3 本指南的复制规则：bash 和 zsh 都能用

- 命令块中的 `bash` 是语法高亮；这些命令也适用于 zsh，不需要先切换 shell。本文不使用 `read -p`，也不依靠另一个终端里曾经设置过的变量。
- 在终端通常用 `Ctrl+Shift+V` 粘贴，按 Enter 执行。不要复制提示符、输出文字或 Markdown 围栏。
- 首次操作按一个命令块一次执行。**出现错误先停在该步**，不要继续启动后面的服务。
- `sudo` 要求的是当前机器的登录密码，输入时没有字符或星号回显是正常现象。输完按 Enter。
- 多行命令末尾的 `\` 表示下一行仍属于同一条命令；不要在它后面加空格。若终端一直显示续行提示符而未运行，按 `Ctrl+C`，重新完整粘贴。
- 后文所有 `192.168.1.10`、`192.168.1.20` 都只是示例。**先查到真实 Wi-Fi IPv4 地址，再替换每个变量设置块中的两个地址**。不要把尖括号占位符直接粘贴到 shell。
- 不要求 `source .venv-hil/bin/activate`。始终使用 `.venv-hil/bin/python`，避免不同窗口意外使用不同 Python。

## 3. O0 / P0：识别同一 Wi-Fi 上的真实 IP

### 3.1 两台分别执行

**O0：Orin 本机终端。**

```bash
hostname
uname -m
nmcli device status
ip -4 -br addr
```

逐行含义：第一条显示主机名，第二条应显示 `aarch64`（报告也接受 `arm64`），第三条列出网卡类型和连接状态，第四条只列 IPv4 地址。

**P0：PC 本机终端。**

```bash
hostname
uname -m
nmcli device status
ip -4 -br addr
```

PC 的 `uname -m` 应是 `x86_64`。在两台输出中各找到 `TYPE` 为 `wifi`、`STATE` 为 `connected` 的接口，再到第四条输出中找**同名接口**的 IPv4 地址。

例如接口可能叫 `wlan0`、`wlp2s0` 或其他名字，不要照抄示例接口名。地址可能形如 `192.168.1.20/24`；只取 `/24` 前面的 `192.168.1.20`。不要选：

- `127.0.0.1`：只能到达本机；
- `192.168.55.1`：通常是 Jetson 的 USB 虚拟网络，本次不用；
- Docker、虚拟机、VPN、桥接接口的地址；
- 没有 `connected` 的接口或 IPv6 地址。

将真实地址记在纸上或文本编辑器：`PC_IP` 是 PC 的 Wi-Fi IPv4，`ORIN_IP` 是 Orin 的 Wi-Fi IPv4。两台同连一个 SSID 仍可能被访客网络、校园网或 AP 隔离阻断；后面的双向检查才是依据。

### 3.2 两边分别测试路由和可达性

**P0：PC；先将以下两行中的示例地址改为真实值。**

```bash
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
printf 'PC=%s  Orin=%s\n' "$PC_IP" "$ORIN_IP"
ip -4 route get "$ORIN_IP"
ping -c 3 "$ORIN_IP"
```

`route get` 应选择前面识别的真实 Wi-Fi 接口，其 `src` 应对应 PC 无线地址。`ping` 有回复表明基本连通；100% 丢包时先检查地址、Wi-Fi 和网络隔离。部分网络禁 ICMP，因此最终仍要看 TCP 检查。

**O0：Orin；同样重新设置两个真实地址，不继承 P0 的变量。**

```bash
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
printf 'PC=%s  Orin=%s\n' "$PC_IP" "$ORIN_IP"
ip -4 route get "$PC_IP"
ping -c 3 "$PC_IP"
```

路由应经过 Orin 的真实 Wi-Fi 接口，`src` 是 Orin 无线地址。若访问 PC 失败，之后 Orin 下载地图和轨迹也会失败；不能因为 PC 能连接 Orin 就跳过反方向。

### 3.3 可选：从 PC SSH 操作 Orin

如果一直使用 Orin 显示器和键盘，可以跳过本节。若要 SSH，首次仍可在 Orin 本机完成安装和网络检查。

**O0：仅在 Orin 需要启用 SSH 且尚未启用时执行。**

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable --now ssh
systemctl status ssh --no-pager
```

这会安装并启动 SSH 服务；应看到服务处于 `active (running)`。如果软件包下载报证书或“尚未生效”错误，先按第 4 节修正网络时间再重试。

如果 Orin 的 UFW 已启用，按第 5 节的方式查看状态，并仅允许实际 PC 地址访问 SSH 的 TCP 22；若 UFW 未启用，不要为了此测试额外启用或禁用防火墙。

**O0：仅在确认 UFW 为 active 且 SSH 被阻断时执行，替换 PC 地址。**

```bash
export PC_IP='192.168.1.10'
sudo ufw allow from "$PC_IP" to any port 22 proto tcp
```

**PC 新终端：从 PC 发起 SSH，替换地址和 Orin 登录用户名。**

```bash
export ORIN_IP='192.168.1.20'
export ORIN_USER='your_orin_login_name'
ssh "${ORIN_USER}@${ORIN_IP}"
```

用户名填你在 Orin Ubuntu 登录时使用的用户，不能填 `robot_1` 来代替，除非它碰巧就是该 Linux 用户。首次连接若询问主机指纹，应与 Orin 本机 `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` 的指纹核对后输入 `yes`；随后输入 **Orin** 密码。

**刚连接成功的 SSH 窗口：以下已在 Orin 执行。**

```bash
hostname
uname -m
whoami
```

确认是 Orin 主机、`aarch64` 和 Orin 用户后，才能将此窗口用作 O0 或 O1。不要在该窗口启动 P1/P2。关闭 SSH 窗口可能使前台 Agent 退出；运行期间保留它。

## 4. O0 / P0：核对系统、时间和基础依赖

### 4.1 Orin：核对已刷好的系统

**O0：Orin。**

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
tr -d '\000' < /proc/device-tree/model
printf '\n'
dpkg-query -W nvidia-l4t-core nvidia-jetpack
python3 --version
/usr/local/cuda/bin/nvcc --version
```

对应检查：

1. `/etc/os-release` 的版本应是 Ubuntu 24.04。
2. `/etc/nv_tegra_release` 应显示 `R39`、`REVISION: 2.1`，对应 L4T 39.2.1。
3. 设备树型号应包含 `Jetson AGX Orin`；内存容量不能单独证明套件型号，系统可用内存也不必精确显示 64 GiB。
4. 软件包查询用于辅助核对。若只缺 `nvidia-jetpack` 元包，不要仅因此判定系统损坏；继续检查 L4T、已安装包和实际 CUDA 探针。元包存在也不能代替内核执行证据。
5. 系统 `python3` 应是 3.12.x；本页不替换系统 Python。
6. `nvcc` 应报告 CUDA 13.2 系列，完整包版本与官方 13.2.1 基线对应。`nvcc` 版本文本和 CUDA Runtime 整数不一定表达补丁号；例如 `13020` 表示 13.2，不等于字符串 `13.2.1`。

此时 `nvcc --version` 仅证明编译器能运行，**尚未证明 GPU 运行成功**。本指南不执行 Ubuntu 通用 `nvidia-driver-*` 安装、`ubuntu-drivers` 自动安装或重装另一套 CUDA；不要用这些方法覆盖 JetPack 的驱动栈。工具链缺失时见故障表。

### 4.2 PC：核对系统

**P0：PC 本机。**

```bash
cat /etc/os-release
uname -m
python3 --version
```

应为 Ubuntu 24.04、`x86_64`、Python 3.12.x。PC 不需要 NVIDIA GPU，也不需要 `nvcc`。

### 4.3 两台：先纠正开机后的网络时间

**O0 和 P0：分别在各自机器执行。**

```bash
date -Iseconds
timedatectl status
```

查看日期是否合理，以及是否同步。Orin 重启后若回到 1970 年，可能导致 HTTPS、Git 和软件包下载失败。这通常是时钟问题，不应先重刷系统。

**只在对应机器时间明显不正确或 NTP 尚未启用时执行：**

```bash
sudo timedatectl set-ntp true
timedatectl status
timedatectl timesync-status
```

等待网络校时后再次执行 `date -Iseconds` 和 `timedatectl status`，确认 `System clock synchronized: yes`。如果系统由其他 NTP 服务管理，`timesync-status` 可能不适用，依据已安装服务的状态检查；不要安装多个互相冲突的校时服务。若时间仍错误，检查互联网/DNS、NTP 是否被网络限制，必要时换到允许两机互通及校时的网络。**不要把文档编写日硬编码进 `date -s`，也不要通过关闭 TLS 校验绕过证书错误。**

### 4.4 两台：安装基础工具

**O0：Orin；本块执行一次。**

```bash
sudo apt update
sudo apt install python3-venv git build-essential
```

**P0：PC；同样执行一次。**

```bash
sudo apt update
sudo apt install python3-venv git build-essential
```

`python3-venv` 提供系统 Python 的隔离环境支持；`git` 用于取得同一版代码；`build-essential` 包括 GCC/G++、make 等工具，**Orin 的 `nvcc` 编译需要本机 C++ 编译器**。按提示确认软件包安装并等待完成。这一步不安装 CUDA、不下载模型，也不进行系统发行版升级。

## 5. O0 / P0：按需开放双向 Agent 端口

两台都要能被对方连接 TCP `50051`。使用可信 Wi-Fi；目前的 gRPC 没有加密和登录认证，不做路由器公网端口映射。

**O0 和 P0：分别检查本机，不能只检查一台。**

```bash
if command -v ufw >/dev/null 2>&1; then
  sudo ufw status verbose
else
  printf 'UFW is not installed; check any other active firewall if connectivity fails.\n'
fi
```

- 显示 `Status: inactive`：不添加 UFW 规则，继续。
- 显示 `Status: active`：如没有适用规则，按下面对应机器的命令，只允许另一台主机的真实地址。
- 没有 UFW：不代表绝对没有其他防火墙；若 TCP 仍不通，检查当前实际使用的防火墙或网络管理策略。

**O0：仅在 Orin 的 UFW 为 active 时执行，地址是 PC 的真实无线地址。**

```bash
export PC_IP='192.168.1.10'
sudo ufw allow from "$PC_IP" to any port 50051 proto tcp
sudo ufw status numbered
```

**P0：仅在 PC 的 UFW 为 active 时执行，地址是 Orin 的真实无线地址。**

```bash
export ORIN_IP='192.168.1.20'
sudo ufw allow from "$ORIN_IP" to any port 50051 proto tcp
sudo ufw status numbered
```

不要执行 `ufw disable`，不要开放整个网段或任意来源。重连 Wi-Fi 后若 DHCP 地址改变，更新两边命令、Agent 参数及对应规则。此时 Agent 还没有启动，所以暂时连接 `50051` 被拒绝不说明配置失败；第 10 节再测试实际监听。

## 6. O0 / P0：取得同一提交的代码

下面的操作在 **Orin 和 PC 各做一次**。目标路径统一为 `~/mars-hardware`。`~` 是当前机器、当前登录用户的 home 目录；SSH 到 Orin 后与 PC 的 home 不是同一个目录。

### 6.1 首次安装：仅当目标目录不存在

**O0：Orin；P0：PC。两台分别执行同一块。**

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
```

成功后仓库应在当前机器的 `~/mars-hardware`。如果提示目录已存在，不要删除它、不要覆盖，改用下一节检查已有仓库。网络/认证失败先修复，再继续。

### 6.2 已有仓库：只允许干净工作区、正确分支、快进更新

先结束所有正在进行的测试并停止 O1/P1，再更新代码。先执行下面的只读检查，确认 `origin` 对应上面的 `wangshiwen-ai-hku/capstone-simulator` 仓库；若不是，停在这里，不运行后面的更新块，也不盲目改写已有 remote。

**O0：Orin；P0：PC。若你使用其他已有目录，应把此页所有后续 `cd` 同步改为那个目录。**

```bash
cd "$HOME/mars-hardware"
git remote -v
git status --short
git branch --show-current
```

**两台分别核对以上输出后，才在各自准备终端执行下面的更新块。** 外层括号使失败只退出本次更新步骤，不关闭你的终端。

```bash
(
  set -e
  cd "$HOME/mars-hardware"
  if [ -n "$(git status --porcelain)" ]; then
    printf 'STOP: local changes or untracked files exist; preserve them before updating.\n' >&2
    exit 1
  fi
  if [ "$(git branch --show-current)" != 'codex/grpc-hardware-loop' ]; then
    printf 'STOP: review the existing branch before switching or updating.\n' >&2
    exit 1
  fi
  git fetch origin codex/grpc-hardware-loop
  git merge --ff-only origin/codex/grpc-hardware-loop
  git rev-parse HEAD
)
```

`fetch` 下载分支更新，`merge --ff-only` 仅接受无需覆盖或合并冲突的快进。

如果提示本地修改、分支不同或无法快进，先保留并审阅现有工作，再安排更新或另选新目录克隆。**不要用 `reset --hard`、`clean -fd`、强制 checkout 或自动 stash 来抹去/隐藏修改。** 已有代码只是分支名相同，也可能尚未包含本次功能。

### 6.3 两台各自确认文件与提交

**O0：Orin；P0：PC。**

```bash
cd "$HOME/mars-hardware"
git status --short
git rev-parse HEAD
ls scripts/mixed_smoke.py scripts/build_cuda_smoke.py
ls examples/mixed_workloads/inflate.cu agent/requirements-hardware.txt
```

记录两台完整 `git rev-parse HEAD` 输出，必须完全相同。若上述文件不存在，说明所取提交还没有完整混合流程，先取得包含它们的同一提交，不改回旧 `hardware_loop` 命令冒充新测试。两次下载之间分支可能更新，所以需要比较完整提交号，不能只比较分支名。

`.venv-hil/` 与 `.mars-hil/` 是忽略的本地产物目录，一般不使 Git 工作区变脏。测试期间不要更新、编辑运行代码，也不要替换 CUDA 二进制。

## 7. O0 / P0：各自创建轻量 Python 3.12 环境

### 7.1 两台分别执行同一套安装

若已有 `.venv-hil`，先检查 `.venv-hil/bin/python --version` 和其用途。仅复用本项目的兼容 Python 3.12 环境；不要向不明用途或其他 Python 版本的环境直接安装。保留不兼容的旧环境后，再建立本指南需要的环境。

**O0：Orin；P0：PC。每台执行一次。**

```bash
cd "$HOME/mars-hardware"
python3 --version
python3 -m venv .venv-hil
.venv-hil/bin/python --version
.venv-hil/bin/python -m pip install -r agent/requirements-hardware.txt
.venv-hil/bin/python -m pip check
```

第一条 Python 版本应是 3.12.x；创建后隔离环境也应是 3.12.x。依赖安装只进入当前机器的 `.venv-hil`，不能从 PC 拷贝整个虚拟环境到 Orin。`pip check` 应报告依赖没有冲突。

**O0 和 P0：分别确认命令入口。**

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python -m agent.main --help
.venv-hil/bin/python -m scripts.mixed_smoke --help
```

Agent 帮助应包含 `mixed-orin`、`mixed-pc`、`--cuda-binary`、`--cuda-repeats`。Runner 帮助应包含 `--runs`、`--workflow-timeout`、`--task-completion-timeout`、`--evidence-timeout`、`--allow-same-host`、`--require-jetpack721`。如果仍只有旧 CPU 流程，回到代码检查。

### 7.2 比较实际运行源码指纹，而不只比较 Git 标签

**O0 和 P0：分别执行，保留两份输出。**

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python - <<'PY'
import json
import agent.telemetry
from agent.telemetry import _runtime_identity
print('loaded telemetry:', agent.telemetry.__file__)
print(json.dumps(_runtime_identity(), indent=2, ensure_ascii=False))
PY
```

输出中的模块路径应属于当前机器的 `~/mars-hardware/agent/telemetry.py`。`git_revision` 必须相同；`runtime_source_sha256` 也必须相同。后者实际读取 `agent`、`examples`、`interfaces`、`mars`、`scripts` 内的 `.py`、`.cu`、`.proto` 文件内容计算，能发现相同提交号下仍有不同源码的情况。

`machine_id_sha256` 应两台不同；它是系统机器标识的哈希，不是直接显示标识原文。Orin 的 `jetson_model`、`jetson_linux` 应有本机信息；普通 PC 对这两项没有值正常。

本步是人工预检。最终报告还要检查**实际执行任务的 Agent** 所报告的主机、提交与源码指纹，不能用两张准备终端截图替代。身份在 Agent 启动时采集，因此改过代码后必须停止并重新启动两台 Agent，再进行新一轮测试。这些是可信局域网部署证据，不是防恶意主机伪造的密码学认证。

## 8. O0：只在 Orin 编译并实际探测 CUDA

PC 不执行本节，不要将 PC 编译出的 x86_64 文件拷贝到 Orin 使用。确保 O0 的 `uname -m` 是 `aarch64`。

**O0：Orin。**

```bash
cd "$HOME/mars-hardware"
g++ --version
/usr/local/cuda/bin/nvcc --version
.venv-hil/bin/python -m scripts.build_cuda_smoke --output .mars-hil/bin/inflate_cuda
```

第一条确认 `build-essential` 提供的本机 C++ 编译器存在。第二条检查 JetPack CUDA 编译器。第三条用默认 `sm_87` 编译 AGX Orin 目标，并且**默认立即在本机 GPU 运行真实内核探针**：实际分配设备内存、执行内核、取回结果、比较完整独立参考、检查有限且大于零的计时。没有 GPU 或内核失败会报错，不会静默降级到 CPU。

成功时最后一条输出 JSON。必须同时看到：

- `compiled: true`；
- `runtime_verified: true`；
- `gpu_info.available: true`、`gpu_info.kernel_execution_verified: true`、`backend: "cuda_runtime"`；
- AGX Orin 对应的设备信息与 compute capability `[8, 7]`；
- `measurement.cuda_event_ms` 和 `measurement.synchronized_wall_ms` 的每个值都是有限正数。

`compiled: true` **单独不构成 GPU 通过**。`--compile-only` 只适合构建检查，会留下 `runtime_verified: false`；本页硬件流程不用这个选项。`nvcc --version`、能够导入包、能列出显卡也都不能替代探针。

生成的文件位于 Orin：

| 文件 | 用途 |
| --- | --- |
| `.mars-hil/bin/inflate_cuda` | 本机原生 CUDA 可执行文件 |
| `.mars-hil/bin/inflate_cuda.manifest.json` | 绑定源文件、二进制 SHA-256、编译参数、编译器和构建主机的记录 |

保留终端中的编译/探针输出。Agent 启动和后续执行会校验本地二进制及其 manifest 与当前 `.cu` 源码是否一致。源码或二进制改变时，先停止 Agent、重新在 Orin 运行上面的构建命令，再重启；不能手工改 manifest 来绕过不一致。构建成功后还没有完成跨机闭环，继续下面三终端操作。

## 9. 打开并保持 O1 / P1 两个 Agent

不要在同一个终端连续启动两个前台服务。以下两个 Agent 会一直占用各自窗口，这是正常状态；光标停在那里不代表卡死。先让两边都出现监听成功信息，再运行 P2。

### 9.1 O1：Orin 的 CPU + CUDA Agent

在 Orin 桌面新开终端，或在 PC 新开终端并按第 3.3 节 SSH 登录 Orin。该窗口标记为 **O1**。

**O1：命令实际在 Orin 执行。先单独执行准备块；重新填写两个真实地址。**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
printf 'O1 on Orin; PC peer=%s:50051; Orin Wi-Fi=%s\n' "$PC_IP" "$ORIN_IP"
.venv-hil/bin/python --version
```

确认架构为 `aarch64`、Python 为 3.12.x，PC peer 是 PC 的真实 Wi-Fi 地址。

**O1：仍在 Orin，启动服务。**

```bash
.venv-hil/bin/python -m agent.main \
  --executor mixed-orin \
  --agent-id robot_1 \
  --kind robot \
  --listen 0.0.0.0:50051 \
  --peer "edge_pc=$PC_IP:50051" \
  --cuda-binary .mars-hil/bin/inflate_cuda \
  --cuda-repeats 3 \
  --task-timeout 90 \
  --artifact-dir .mars-hil/robot_1
```

参数含义：`mixed-orin` 接受测距、CUDA 膨胀和最终 CPU 校验；`--listen` 让另一台机器能通过无线地址连接；`--peer` 指向真实 PC，用于取回地图和轨迹；`--cuda-binary` 是本机刚编译的文件；`--cuda-repeats 3` 是每个 GPU 任务的三次测量；`--task-timeout 90` 是 Agent 单任务超时，单位秒；`--artifact-dir` 保存本机任务产物。

启动会再执行 CUDA 预检。应看到包含以下文字的监听提示：

```text
REAL CPU + native CUDA mixed navigation
robot_1 listening on 0.0.0.0:50051
```

实际输出可在同一行且带产物路径。若是 `MOCK`、旧 `REAL CPU navigation` 或直接退出，不能继续。不要添加旧 `--config` 配置，也不要另起 YOLO、模型或业务服务器。**保持 O1 运行；不要在这个窗口输入后面的检查命令。**

### 9.2 P1：PC 的 CPU 建图 / 规划 Agent

在 **PC 本机**新开终端，标记为 P1。不要使用刚才 SSH 到 Orin 的窗口。

**P1：PC 本机；重新填写两个真实地址。**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
printf 'P1 on PC; Orin peer=%s:50051; PC Wi-Fi=%s\n' "$ORIN_IP" "$PC_IP"
.venv-hil/bin/python --version
```

确认 `x86_64`、Python 3.12.x。

**P1：仍在 PC，启动服务。**

```bash
.venv-hil/bin/python -m agent.main \
  --executor mixed-pc \
  --agent-id edge_pc \
  --kind edge \
  --listen 0.0.0.0:50051 \
  --peer "robot_1=$ORIN_IP:50051" \
  --task-timeout 90 \
  --artifact-dir .mars-hil/edge_pc
```

应看到 `REAL CPU mixed mapping/planning` 及 `edge_pc listening on 0.0.0.0:50051`。PC 的 Agent 不需要 CUDA 二进制。两台不同机器都使用 50051 不冲突。**保持 P1 与 O1 同时运行。**

## 10. O0 / P2：服务启动后检查 TCP 的两个方向

### 10.1 Orin → PC

回到空闲 **O0**，或新开一个 Orin 本机终端/额外 SSH 终端。**不要在仍运行 Agent 的 O1 输入本块。**

**O0：Orin，重新填写 PC 的真实地址。**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
.venv-hil/bin/python - <<'PY'
import os
import socket
peer = os.environ['PC_IP']
with socket.create_connection((peer, 50051), timeout=5):
    print(f'OK: Orin can reach PC {peer}:50051')
PY
```

只有打印 `OK` 才说明这次 TCP 连接成功。`ConnectionRefusedError` 常见于 P1 没启动、监听地址/端口错；超时常见于地址、路由、防火墙或 Wi-Fi 隔离问题。

### 10.2 PC → Orin，以及 PC → 本机 Agent

在 **PC 本机再新开一个终端 P2**，作为后续协调器终端。

**P2：PC 本机；重新填写两台真实地址。**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
.venv-hil/bin/python - <<'PY'
import os
import socket
for label, host in [('Orin', os.environ['ORIN_IP']), ('PC local Agent', '127.0.0.1')]:
    with socket.create_connection((host, 50051), timeout=5):
        print(f'OK: PC can reach {label} at {host}:50051')
PY
```

确认这里是 `x86_64`，且两个 TCP 检查都成功。`127.0.0.1` **只在 P2 连接 PC 本机 Agent 时正确**；O1 的 `--peer edge_pc=...` 必须是 PC 无线地址，不能改成 localhost。

TCP 连接成功只证明端口可达；还要执行整个 DAG 并检查产物才是闭环验收。

## 11. P2：运行种子 19 / 20 / 21 的三轮完整测试

### 11.1 为此次尝试选择不会覆盖旧证据的路径

保持 O1/P1 运行。在 **同一个 P2** 执行下面准备块。即使上一步已设置变量，这里仍给出完整设置，便于重开终端后使用。

**P2：PC 本机；替换真实地址。**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
mkdir -p .mars-hil/reports
export HIL_REPORT=".mars-hil/reports/mixed-$(.venv-hil/bin/python -c 'from datetime import datetime, timezone; from uuid import uuid4; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex)').json"
printf 'Save this report path: %s\n' "$HIL_REPORT"
```

路径包含 UTC 时间和随机 UUID；后者也避免时间重复产生同名文件。**生成的是路径，不提前创建报告文件**。把输出路径保存下来。Runner 以独占创建方式写入报告；目标已存在时拒绝覆盖。不要用 `> "$HIL_REPORT"` 重定向命令输出，这会先创建文件并导致拒绝。

### 11.2 执行协调器

**P2：PC 本机，仍在刚才的目录及窗口。**

```bash
.venv-hil/bin/python -m scripts.mixed_smoke \
  --agent "robot_1=$ORIN_IP:50051" \
  --agent edge_pc=127.0.0.1:50051 \
  --seed 19 \
  --runs 3 \
  --require-jetpack721 \
  --output "$HIL_REPORT"
HIL_EXIT_CODE=$?
printf 'mixed_smoke exit code: %s\n' "$HIL_EXIT_CODE"
```

此命令通过 `CentralCoordinator` 和 `GrpcRuntimeAdapter` 进行派单、依赖推进、产物传输与证据收集；不是在 P2 直接调用五个业务函数。`--seed 19 --runs 3` 依次运行 **19、20、21**，每轮都是完整五任务；首次失败会停止后续轮次并保留已完成部分，不会跳过失败凑出三次成功。

默认硬件模式要求真实两台主机、PC `x86_64`、Orin `aarch64`/`arm64`、AGX Orin 设备树型号、GPU compute capability `[8, 7]`、不同机器标识、相同提交及实际源码指纹。CLI 的 JetPack 版本开关默认不启用；**本页硬件命令全部显式带 `--require-jetpack721`**，要求 L4T `R39 / REVISION: 2.1` 和 CUDA Runtime 13.2（整数 `13020`）。省略该开关不能代表通过本页的严格版本验收；它也不代替第 4 节对 Ubuntu、Python 和 CUDA 工具链的检查。

**不要添加 `--allow-same-host`。** 它只用于开发阶段的单机/传输调试，即使 `status` 显示 `succeeded`，也永远不能取得 `hardware_smoke_passed: true`。两个 Agent 名字不同、两个端口不同或编译 CI 成功，都不等于两台实物主机通过。

### 11.3 超时与重复次数的区别

| 参数 | 本页数值 | 管什么 |
| --- | --- | --- |
| O1/P1 `--task-timeout` | 90 秒，启动命令显式设置 | Agent 的单次任务执行上限 |
| P2 `--workflow-timeout` | 默认 180 秒 | 每轮工作流阶段上限 |
| P2 `--task-completion-timeout` | 默认 120 秒 | 协调器等待任务完成上限，不能大于 workflow timeout |
| P2 `--evidence-timeout` | 默认 60 秒 | 每轮产物收集与独立复核阶段上限 |
| O1 `--cuda-repeats` | 3 | 每个 GPU 任务内测量三次，另有一次 warmup |
| P2 `--runs` | 3 | 三轮独立工作流，种子依次递增 |

180 秒不是三轮加在一起的总上限，90 秒也不是所有任务运行耗时之和。原生 CUDA 子进程和编译还各有自身保护超时；仅调大 P2 超时不一定改变内核或 Agent 限制。出现超时先查失败阶段、Agent 日志、网络和负载，再有依据地调整。

等待 P2 完成，不要同时再启动第二个协调器，不要关闭 Agent，也不要在三轮之间更新源码、重新编译或重启服务；报告会检查跨轮实际执行主机/运行身份一致。

## 12. P2：按真实报告验收，而不是只看“进程没报错”

### 12.1 打开完整 JSON

**P2：PC 本机，使用刚才同一终端中的路径。**

```bash
printf 'Report: %s\n' "$HIL_REPORT"
.venv-hil/bin/python -m json.tool "$HIL_REPORT" | less
```

按空格向下翻页，输入 `/hardware_smoke_passed` 后按 Enter 搜索，按 `n` 查下一个，按 `q` 退出查看器。报告保留全部业务产物，内容较长是正常的。

如果重开过 P2，先执行 `cd "$HOME/mars-hardware"`，再用 `export HIL_REPORT='实际已保存的报告相对或绝对路径'` 恢复刚才记录的路径；**不要重新生成一个不存在的随机路径来查看旧报告**。

### 12.2 必须满足的验收内容

| 层级 / 字段 | 必须看到什么 |
| --- | --- |
| 进程及汇总 | 退出码 0，顶层 `status: "succeeded"`、`error: null`、`hardware_smoke_passed: true`、`gpu_tested: true` |
| 汇总范围 | `scope: "cross_host_cpu_native_cuda_execution"`；`allow_same_host: false`；本命令的 `require_jetpack721: true` |
| 轮次完整 | `requested_runs: 3`、`completed_runs: 3`；`runs` 三项的种子依次是 19、20、21；每项都 succeeded、硬件通过，不能只看第一项 |
| 每轮结构 | `executions` 正好五项、`artifacts` 正好六项、`edges` 正好八项；分配与第 1 节相同 |
| 实际硬件 | `hosts.edge_pc.architecture` 是 x86_64；`hosts.robot_1.architecture` 是 aarch64/arm64；Orin 型号/版本正确；`executing_host_count: 2`、`host_count_basis: "machine_id_sha256"` |
| 代码来源 | 每轮 `checks.matching_git_revision`、`checks.matching_runtime_source` 为 true；两个实际 host 的 `git_revision` 和 `runtime_source_sha256` 分别完全一致 |
| CUDA 实算 | `gpu_execution.task_id: "inflate"`、`agent_id: "robot_1"`；`measurement.backend: "cuda_runtime"`；真实 source/binary 哈希与启动预检一致 |
| CUDA 计时 | `gpu_execution.measurement.cuda_event_ms` 与 `synchronized_wall_ms` 均为三项，每项有限且 > 0；不是空数组、0、NaN 或估算值 |
| 全图独立参考 | `validation.valid: true`、`gpu_full_reference_match: true`、`gpu_cells_checked: 6144`；`checks.independent_validation: true` |
| GPU 输出确实被使用 | `validation.checks` 包含 `gpu_full_grid_cpu_reference` 和 `gpu_output_used_by_planner`；轨迹的 `inflation_sha256` 对应本轮 `inflated` 业务内容哈希 |
| 双向真实字节 | `map` 与 `plan` 的执行记录有正的 `remote_input_bytes`（Orin → PC）；`inflate` 与 `validate` 有正值（PC → Orin）；八条边的 `remote_bytes` 与第 1 节的本机/跨机关系一致 |
| 输入/输出身份 | `checks.artifact_ports`、`execution_placement`、`host_identity_consistent`、`edge_transfers`、`source_lineage`、`cuda_measurement` 都是 true；`hardware_gate_failures` 是空数组 |

`artifacts[].reference.checksum` 是整份传输 envelope 的哈希；业务内容哈希在 `payload_sha256` 和各产物的来源字段里。二者覆盖内容不同，不能要求它们相等。`validation.source_hashes` 标明实际消费的 `map`、`inflated`、`trajectory`、`truth`。不要只检查 GPU 设备名或一项 `valid: true` 就跳过整条来源链。

PC 独立复核时，GPU 整数栅格、哈希、结构和布尔结果要求完全一致；距离、规划运动时长等派生浮点指标采用 `rel_tol=1e-9`、`abs_tol=1e-9`，允许 x86_64 与 ARM64 数学库的末位差异。实际容差记录在 `validation_float_tolerance`；它不放宽碰撞或 GPU 栅格正确性检查。

### 12.3 可复制的报告核对命令

下面只读取已保存的 JSON，并在条件不满足时明确退出；不会改写报告，也不会创造 GPU 成功证据。Runner 已负责更完整的产物、输入边、校验值和独立算法复核，本块用于方便人工验收。

**P2：PC 本机，`HIL_REPORT` 仍指向刚才的真实文件。**

```bash
.venv-hil/bin/python - "$HIL_REPORT" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    report = json.load(stream)

def require(condition, message):
    if not condition:
        raise SystemExit('NOT ACCEPTED: ' + message)

print('status:', report.get('status'), 'hardware_smoke_passed:', report.get('hardware_smoke_passed'))
print('error:', report.get('error'))
require(report.get('status') == 'succeeded' and report.get('error') is None, 'aggregate failed')
require(report.get('hardware_smoke_passed') is True and report.get('gpu_tested') is True, 'no hardware/GPU acceptance')
require(report.get('scope') == 'cross_host_cpu_native_cuda_execution', 'wrong scope')
require(report.get('allow_same_host') is False and report.get('require_jetpack721') is True, 'strict run required')
require(report.get('requested_runs') == 3 and report.get('completed_runs') == 3, 'three runs required')
runs = report.get('runs', [])
require([run.get('seed') for run in runs] == [19, 20, 21], 'expected seeds 19/20/21')
placements = {'sense': 'robot_1', 'map': 'edge_pc', 'inflate': 'robot_1', 'plan': 'edge_pc', 'validate': 'robot_1'}
required_checks = ('artifact_ports', 'execution_placement', 'host_identity_consistent', 'edge_transfers',
                   'source_lineage', 'cuda_measurement', 'independent_validation', 'distinct_machine_ids',
                   'matching_runtime_source', 'matching_git_revision', 'target_architectures',
                   'jetson_agx_orin', 'orin_compute_capability', 'jetpack721', 'no_test_fixtures', 'native_binary_identity')
for run in runs:
    require(run.get('status') == 'succeeded' and run.get('hardware_smoke_passed') is True, 'failed run')
    require(run.get('error') is None and run.get('hardware_gate_failures') == [], 'run errors/gates')
    require(run.get('executing_host_count') == 2 and run.get('host_count_basis') == 'machine_id_sha256', 'two actual hosts required')
    require(all(run.get('checks', {}).get(key) is True for key in required_checks), 'missing/failed evidence check')
    records = run.get('executions', [])
    require(len(records) == 5 and {item['task_id']: item['agent_id'] for item in records} == placements, 'task placement/count')
    require(len(run.get('artifacts', [])) == 6 and len(run.get('edges', [])) == 8, 'artifact/edge count')
    for item in records:
        if item['task_id'] in ('map', 'inflate', 'plan', 'validate'):
            require(item.get('remote_input_bytes', 0) > 0, 'missing bidirectional input bytes')
    gpu = run['gpu_execution']['measurement']
    require(gpu.get('backend') == 'cuda_runtime' and gpu.get('repeats') == 3, 'native CUDA repeats')
    for key in ('cuda_event_ms', 'synchronized_wall_ms'):
        values = gpu.get(key, [])
        require(len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), 'invalid ' + key)
    valid = run['validation']
    require(valid.get('valid') is True and valid.get('gpu_full_reference_match') is True, 'independent full-mask validation')
    require(valid.get('gpu_cells_checked') == 96 * 64, 'incomplete grid check')
    require({'gpu_full_grid_cpu_reference', 'gpu_output_used_by_planner'}.issubset(valid.get('checks', [])), 'GPU mask consumption/reference')
    print('seed:', run['seed'], 'cells:', valid['gpu_cells_checked'], 'remote bytes:', run['remote_input_bytes'])
    print('  CUDA event ms:', gpu['cuda_event_ms'], 'synchronized wall ms:', gpu['synchronized_wall_ms'])
    for node, host in run['hosts'].items():
        print(' ', node, host['hostname'], host['architecture'], host['git_revision'], host['runtime_source_sha256'])
print('ACCEPTED: this saved report satisfies the documented three-run hardware checks.')
PY
```

只有这次真实运行的原始报告满足条件才会输出最后一行。若打印 `NOT ACCEPTED`、缺少字段或异常，保留原始文件和错误信息，按下节排查；不要手工把 JSON 中的 false 改成 true，也不要复制单元测试/CI 的 fixture 作为硬件报告。

## 13. 正确解释耗时、GPU 利用率和能量

- `gpu_execution.measurement.cuda_event_ms`：CUDA event 包围膨胀内核的计时；不包含网络、Python 启动、设备内存分配或主机/设备数据传输。
- `synchronized_wall_ms`：主机侧围绕内核启动、event 和同步的墙钟耗时；它与纯内核时间含义不同，不能互相替代。warmup 在正式三次测量之前。
- `executions[].worker_elapsed_ms`：业务子进程启动、输入输出及计算的实际耗时；`input_fetch_ms` 单列获取输入的耗时。每轮 `worker_elapsed_ms` 为五个任务的累计，不是仅内核时间。
- 每轮 `workflow_wall_elapsed_ms`、`evidence_wall_elapsed_ms`、`total_wall_elapsed_ms` 分别反映工作流、证据收集和完整执行范围。顶层 `total_wall_elapsed_ms` 是这次多轮调用的实际总体耗时。
- `remote_input_bytes` / `edges[].remote_bytes` 计数的是跨节点消费者获取的产物 envelope 字节。它不包含所有 gRPC/TCP/Wi-Fi 开销，也不把协调器最后下载证据混入业务边流量；同一产物被不同任务消费时可被多次计数。
- `host_observations.before/after` 是整台机器的 CPU/内存采样，不是专属于这个任务的峰值或平均值。CPU 样本使用至少约 100 ms 的真实计数窗口，短任务前后可能复用同一样本。各 Agent 的 monotonic 时间不能跨主机当作同步日历时间比较。
- 这个 6,144 格小内核只用来证明计算路径与正确性。启动和网络开销可能远大于内核时间，**不是 GPU 跑分或压力测试**，也不承诺 GPU 比 CPU 快。
- 空闲桌面或低频监控里看到 GPU 利用率 0% 很正常，小内核可能在采样之间完成。仓库遥测中的 GPU 利用率旧字段还可能是明确标记为未测量的零占位值；通过与否看真实内核输出、计时和独立参考，不看一张利用率截图。
- `allocated_device_bytes` 是该程序显式分配的设备字节，不是整个 GPU 或进程的峰值显存；AGX Orin 采用共享内存架构，不能按独立显卡显存占用简单解释。
- **能量未测量**：`energy_j: null`，功率和温度等未测项应保留“不可用”含义。`planning_assumptions` 中的带宽、计算代价、能量或旧 Proto 的零只是调度假设/占位值，不能写成实测 0 J 或节能结论。
- 轨迹的 `planned_motion_duration_s` 是规划的运动时长，不是真实设备运动耗时；本次没有电机执行。

## 14. 按症状排查：先保留报告和两边终端输出

| 症状 / 报告关键字 | 具体操作 |
| --- | --- |
| 开机日期 1970；证书无效；软件包“尚未生效” | 在发生问题的机器执行第 4.3 节检查并启用网络校时；日期合理后再下载。不要重刷或关闭证书校验。 |
| `nmcli` 无 connected 的 wifi 接口 | 在该机 Ubuntu 网络设置连接 Wi-Fi；检查适配器是否被系统识别。选真实无线地址，不选 USB 的 192.168.55.1。 |
| Ping 或 TCP 超时 | 重新检查两台 Wi-Fi IPv4、`ip -4 route get`、是否访客网络/AP 隔离、UFW scoped 规则。换网络后重新设置所有窗口中的 IP 并重启两个 Agent。 |
| `Connection refused` / `UNAVAILABLE` | 看 O1/P1 是否仍在运行；检查监听 0.0.0.0:50051。在发生问题的机器空闲终端运行 `ss -ltnp 'sport = :50051'`；无监听就先修复 Agent 启动错误。 |
| 注册成功但 `remote inputs` 失败 | 第 10 节两个方向都测；O1 的 PC peer 不能是 127.0.0.1，P1 的 Orin peer 不能是 PC 地址。确认对方仍有产物且目录未被删。 |
| `No module named scripts.mixed_smoke` / 缺少 mixed executor | 当前提交不完整或目录错误。两台 `cd` 到正确仓库，检查必需文件和完整提交号，停止 Agent 后快进到同一兼容提交。 |
| `ensurepip` / `venv` 不可用 | 当前机器安装 `python3-venv`，确认系统 Python 3.12，再创建 `.venv-hil`；不要使用旧指南的 python3.10。 |
| `grpc` / `protobuf` 导入错误或版本冲突 | 确认执行的是 `.venv-hil/bin/python`，从本仓库 `agent/requirements-hardware.txt` 安装并 `pip check`。本环境不装 LeRobot/PyTorch。 |
| `nvcc` 不存在 | 仅在 Orin 查 `/usr/local/cuda/bin/nvcc`、`ls -ld /usr/local/cuda*`、`dpkg-query -W 'cuda-nvcc*'`。对照已安装 JetPack 7.2.1 的 NVIDIA 软件包修复缺失组件；不要安装 Ubuntu 通用驱动或随意重装 CUDA。若只是实际路径不同，核实版本后用构建工具 `--nvcc` 指向已安装的正确编译器。 |
| `g++` 不存在 / host compiler failed | Orin 安装 `build-essential`，核对 `g++ --version`；保留编译诊断。不要以 `--compile-only` 把失败转写成 GPU 通过。 |
| `unsupported gpu architecture` / `no kernel image` / `invalid device function` | 确认在 AGX Orin 上用正确 CUDA 工具链默认 `sm_87` 编译，未从 PC 拷贝 x86_64 二进制；本机重建并实际探针。不要为隐藏错误随意改目标架构。 |
| `Exec format error` | 二进制主机架构不对，通常是拷贝了 PC 构建产物。停止 O1，在 Orin 重新运行本机构建命令。 |
| `compiled: true` 但 runtime probe failed | 编译阶段成功，GPU 尚未通过。检查原始 CUDA 错误、设备访问与 JetPack 安装，恢复真实探针成功后再启动 O1。不要当作通过。 |
| `CUDA binary provenance check failed` / `source_sha256 mismatch` / `binary_sha256 mismatch` | 保留错误；停止 O1，核对源码版本，再在 Orin 重新构建及运行探针，随后重启。不编辑 manifest 或将错误哈希手工换成期望值。 |
| `matching_git_revision` / `matching_runtime_source` 失败 | 查看报告内实际两个 host 的值，核对两台源码差异及是否有旧 Agent 正在运行；保留修改，统一同一提交与源码后重启两边。只改分支名不够。 |
| `distinct_machine_ids` / `target_architectures` 失败 | O1 是否真的 SSH 在 Orin？P1/P2 是否真的在 x86_64 PC？检查机器身份，不加 `--allow-same-host` 来放行硬件测试。 |
| `jetson_agx_orin` / `jetpack721` 失败 | 看报告中的设备树型号和 L4T release，回到 Orin 第 4.1 节核对。缺文件、错误型号或错误版本都应定位实际环境，不伪造文件或跳过严格检查。 |
| `native_binary_identity` / CUDA preflight identity 不一致 | 检查本轮 `gpu_execution.measurement` 与 Orin host 的 `cuda_device`；测试中是否重建过二进制或换了源码？停止、在一致版本上重建并重启，用新文件名再试。 |
| `missing positive measured` / 计时为 0、非有限值 | 该 GPU 测量无效，保留原始值及设备信息。排查实际内核及同步错误，不用估算或最小常数替代。 |
| `GPU inflation differs from the independent full CPU reference` | 真实正确性失败。保留地图、掩码、种子及源码/二进制哈希；不能在 PC 重算掩码替换 GPU 输出后称原测试通过。 |
| `planner did not consume this GPU inflation output` / `source_lineage` / checksum 错误 | 保留本轮六份产物和完整输入引用，核对两边代码、scene ID 和边对应关系。不要删除来源检查。 |
| `TimeoutError` 且 `phase: workflow` | 先看 O1/P1 的业务错误、无线传输、CPU 负载和 90 秒任务上限；再评估 workflow 180 / completion 120。 |
| `TimeoutError` 且 `phase: evidence` | 五任务可能已结束，但产物下载或 PC 独立复核未完成；保持两边 Agent，检查 Wi-Fi、产物服务和 PC 负载。未收齐证据仍不通过。 |
| 只运行 1 或 2 轮 | runner 在首次失败后停止。查看最后一轮 `error`、`phase`、`hardware_gate_failures`，修复后换新报告路径重跑三轮。 |
| `output already exists` | 原文件受到保护，重新执行第 11.1 节生成新路径，再运行。不要删除旧报告来复用名字，也不要预先创建目标文件。 |
| `Address already in use` | 当前机器已有服务占用该端口。在空闲终端检查监听，先确认进程身份，只停止自己之前启动的 Agent；不要盲目杀其他服务。 |
| `attempt_history_full_restart_agent` | 等当前协调器结束/停止，重启对应 Agent 后以新路径重试；不要在三轮正常测试中途重启。 |

任何软件或来源检查失败都属于待修复的可定位问题。保存完整 `error` 和 `phase` 比只描述“不能运行”更有帮助。默认硬件门槛、CPU 全图参考和证据检查必须保留。

## 15. 保存证据、正常停止和下次启动

1. 等 P2 有最终结果和报告路径，完成第 12 节核对。不要把终端的一行 succeeded 摘出来而丢掉原始 JSON。
2. 保留 PC 上 `.mars-hil/reports/` 的独立报告及其 `received-artifacts/` 子目录；保留 PC `.mars-hil/edge_pc/` 和 Orin `.mars-hil/robot_1/`。报告内也包含所收集产物与执行记录。
3. 同时保留 Orin 的 CUDA 二进制、相邻 manifest、编译探针输出，及 O1/P1 的启动和失败诊断。可复制终端文本保存；需要共享时同时说明报告路径、种子、两台提交号及源码指纹。不要把开发 CI 日志替代这两台设备的日志。
4. 在 **P1** 按一次 `Ctrl+C`，等 PC Agent 返回提示符；再在 **O1** 按一次 `Ctrl+C`，等 Orin Agent 返回提示符。这是结束前台进程，不是关机。
5. O1 如果是 SSH，会话内在 Agent 停止后执行 `exit` 才返回 PC 本地 shell。要关 Orin 时，使用 Ubuntu 正常关机流程，等待关机后再断电。
6. 若中途按 `Ctrl+C` 停止 P2，也要检查并停止 O1/P1，不能假设协调器窗口关闭就已取消全部远程工作。中断可能尚未写出完整汇总报告，已有诊断和产物仍应保留。
7. 不要在 Agent 运行时删除 `.mars-hil`。下次运行先检查 Wi-Fi 地址、时间、代码与二进制是否一致；再按 O1 → P1 → 双向 TCP → P2 的顺序启动，生成新报告路径。

完成本页后可以据真实日志记录“在这两台设备、此提交和环境上，种子 19/20/21 的 CPU + 原生 CUDA 混合闭环通过”。未拿到这样的证据之前，只能记录已完成的准备、编译或软件测试。后续若要评估模型推理，再单独准备匹配 JetPack 7.2.1 的模型环境，不把旧 VLA 安装步骤混入本次验收。
