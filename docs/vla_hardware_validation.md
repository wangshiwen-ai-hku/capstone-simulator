# Real CUDA and SmolVLA validation: PC + Jetson AGX Orin 64GB

> Check the version first: NVIDIA has no release named **JetPack 7.1.2**. AGX Orin
> support in JetPack 7 starts at **7.2 / L4T 39.2**; this guide targets
> **JetPack 7.2.1 / L4T 39.2.1**. Do not accept a remembered version string if
> `/etc/nv_tegra_release` is not `R39, REVISION: 2.1`. Complete the native CUDA
> loop in the [mixed CPU/GPU runbook](hardware_validation.md) before this PyTorch
> and model test. [NVIDIA JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive)

This workflow adds real CUDA matrix multiplication and pretrained SmolVLA inference to MARS. A PC supplies a recorded robot observation; MARS schedules inference on the Orin GPU; the PC receives and validates the returned actions. The [existing CPU navigation workflow](hardware_cpu_validation.md) remains available.

The test verifies computation and cross-host transport. It does not actuate a robot or establish pick-and-place success. `lerobot/smolvla_base` is a foundation checkpoint intended for task-specific fine-tuning. Hardware execution on your devices remains to be verified by following this guide. [Model card](https://huggingface.co/lerobot/smolvla_base)

For the complete Chinese instructions, see [中文 GPU/VLA 操作指南](vla_hardware_validation_zh.md).

## 1. Hosts and environments

Assume Linux on both hosts, checkout at `$HOME/mars-hardware`, PC `192.168.1.10`, and Orin `192.168.1.20`. Replace addresses and SSH usernames throughout. Both hosts need the same commit containing `scripts/vla_loop.py`, with mutual access to TCP `50051` on the trusted LAN described in the CPU guide.

| Terminal | Host | Responsibility |
| --- | --- | --- |
| Preparation | Orin | Install the separate ML environment and prepare assets |
| O1 | Orin | GPU Agent; starts its own ML subprocess |
| P1 | PC | Observation and validation Agent |
| P2 | PC | Coordinator and one-run report |

The PC requires no GPU or ML libraries. Each host uses `.venv-hil` for the lightweight MARS Agent. Orin additionally uses `.venv-vla` for CUDA/LeRobot. These must remain separate: MARS requires protobuf 7, while LeRobot's WandB dependency requires protobuf below 7.

## 2. Common Agent setup

For a new checkout on each host:

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
cd "$HOME/mars-hardware"
git rev-parse HEAD
python3 -m venv .venv-hil
.venv-hil/bin/python -m pip install -r agent/requirements-hardware.txt
.venv-hil/bin/python -m agent.main --help
.venv-hil/bin/python -m scripts.vla_loop --help
```

An existing checkout must be updated to the GPU/VLA implementation, preserving local changes. Compare commit IDs on both hosts. Reuse the CPU/native-CUDA workflow's `.venv-hil` if already prepared. JetPack 7.2.1 supplies Python 3.12; Python 3.12 is also recommended on the PC. Keep the lightweight Agent and VLA worker in separate environments.

## 3. Verify the Orin platform and create the VLA environment

Run these checks on Orin before installing any ML packages:

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

Continue only when the model contains `Jetson AGX Orin`, L4T reports `R39` and `REVISION: 2.1`, the OS is Ubuntu 24.04, Python is 3.12, CUDA compiler is from the 13.2 series, and the architecture is `aarch64`. The `nvidia-jetpack` meta-package may be absent, so L4T is the platform source of truth. `nvidia-smi`, a package candidate, or a handwritten `7.1.2` string is insufficient. NVIDIA lists JetPack 7.2.1 as L4T 39.2.1, Ubuntu 24.04, CUDA 13.2.1, with Orin Family support. [JetPack 7.2.1 release information](https://developer.nvidia.com/embedded/jetpack/downloads)

The documented install path uses this reproducible candidate combination:

| Component | Value installed by this guide |
| --- | --- |
| Python | `3.12.x` |
| PyTorch / TorchVision | `2.10.0` / `0.25.0`, CUDA 13.0 aarch64 builds |
| LeRobot | `0.4.4` |
| Transformers | `4.57.1` |
| Target GPU | AGX Orin, compute capability `8.7`, Torch build includes `sm_87` |

LeRobot 0.4.4 requires Torch below 2.11 and TorchVision below 0.26, so the installer must not upgrade Torch to 2.11 or later. [LeRobot 0.4.4 requirements](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

At this guide's update date, NVIDIA's PyTorch for Jetson matrix does not list a JetPack 7.2/7.2.1 wheel combination, and its NVIDIA wheel column contains no JetPack 7 wheel. The commands below therefore use the **Jetson AI Lab community index** `sbsa/cu130`. This is a candidate that must pass the on-device gates below; installation alone is not NVIDIA certification. [NVIDIA PyTorch for Jetson matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)

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

The last command runs real CUDA matrix and TorchVision CUDA NMS checks before and after installing LeRobot. It requires device capability `[8, 7]`, requires the Torch architecture list to contain `sm_87`, and constrains Torch/TorchVision so dependency resolution cannot replace them. Stop on any failure; do not substitute a CPU wheel or remove a gate.

Do not install `agent/requirements-hardware.txt` and `agent/requirements-vla.txt` in the same environment. MARS uses protobuf 7, while the LeRobot 0.4.4 dependency tree needs an older protobuf. O1 runs the GPU worker through a restricted local standard-input/output protocol, so each environment can keep its own dependencies.

An existing separate CUDA environment may be used instead; run the same check through its interpreter and give that absolute interpreter path to O1's `--worker-python`. Such an environment may use another version within LeRobot 0.4.4's declared Torch/TorchVision ranges, so preserve the reported versions with the test evidence rather than describing it as the exact baseline above:

```bash
.venv-vla/bin/python -m scripts.install_vla \
  --require-python 3.12 \
  --require-compute-capability 8.7
```

The JSON must include `status: ready`, Python `3.12`, capability `[8, 7]`, `sm_87`, the actual GPU and Torch/CUDA versions, LeRobot `0.4.4`, and Transformers `4.57.1`. If this community build fails a gate on the actual system, the native CUDA mixed loop remains independently usable, but SmolVLA is not ready.

## 4. Prepare fixed assets on Orin

Preparation requires Internet. The inference worker operates offline and cannot fetch missing files. The policy download is about 0.91GB plus small VLM configuration/tokenizer files. The recorded sample's source videos total about 470MB. Allow additional disk space for dependencies and caches.

```bash
cd "$HOME/mars-hardware"
.venv-vla/bin/python -m scripts.prepare_vla model \
  --output .mars-vla/model
.venv-vla/bin/python -m scripts.prepare_vla sample \
  --output .mars-vla/observation.json \
  --cache .mars-vla/datasets
```

The bundle records SHA256 hashes and immutable revisions; execution verifies the files and strictly loads the full policy weights. The embedded VLM weights come from the policy checkpoint; only the backbone configuration and tokenizer need separate files.

| Asset | Repository | Revision |
| --- | --- | --- |
| Policy | `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| VLM configuration/tokenizer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Recorded observation | `lerobot/svla_so100_pickplace` | `728583b5eaf9e739a7f119e2def466fa1d552402` |

The exporter uses episode 0, frame 0, preserving the recorded six joint states and task text. Actual camera pixels are resized with preserved aspect ratio to at most 256 pixels on the longest edge and encoded as PNG. The consolidated dataset videos explain why a one-frame export still downloads about 470MB. [Dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace)

`top` maps to `observation.images.camera1`; `wrist` maps to `camera2`. The checkpoint also describes `camera3`, but the sample has no third camera. LeRobot's `empty_cameras=0` inference path uses the supplied views; the result explicitly records the missing third view. No zero state, duplicate camera, or generated picture is substituted. [Image handling](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)

Copy only the observation JSON to the PC. Run on the **PC**:

```bash
cd "$HOME/mars-hardware"
mkdir -p .mars-vla
scp YOUR_ORIN_USER@192.168.1.20:~/mars-hardware/.mars-vla/observation.json \
  .mars-vla/observation.json
```

Check an existing destination before replacing it. The model and source videos remain on Orin.

## 5. O1: GPU Agent on Orin

Stop any old navigation Agent using port `50051`, then run:

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python -m agent.main \
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

Startup runs a CUDA sum, TorchVision CUDA NMS, and imports the fixed LeRobot/Transformers versions through `--worker-python`; it also validates the model manifest. Expect `REAL CUDA VLA` only after every check passes. Full weights still load strictly when the inference task arrives. Keep O1 running. The peer is the PC's LAN address; the worker interpreter must exist on Orin. CUDA unavailability is an error; there is no CPU fallback.

For a CUDA-only check, omit `--model-dir` and skip asset preparation. That Agent will not advertise SmolVLA support.

## 6. P1: Input and validation Agent on PC

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python -m agent.main \
  --executor vla-io \
  --agent-id edge_pc \
  --kind edge \
  --listen 0.0.0.0:50051 \
  --peer robot_1=192.168.1.20:50051 \
  --observation-file .mars-vla/observation.json \
  --artifact-dir .mars-vla/edge_pc
```

Expect `REAL CPU VLA input/validation`. Keep P1 running. CUDA-only testing can omit `--observation-file`.

## 7. P2: Execute both workflows from PC

First check real CUDA multiplication and its independently checked result:

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python -m scripts.vla_loop \
  --workload cuda \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/cuda-01.json \
  --require-distinct-hosts \
  --require-hardware \
  --require-jetpack 7.2.1
```

Then execute pretrained SmolVLA:

```bash
.venv-hil/bin/python -m scripts.vla_loop \
  --workload smolvla \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/smolvla-01.json \
  --require-distinct-hosts \
  --require-hardware \
  --require-jetpack 7.2.1
```

The coordinator is on the PC, so its PC endpoint uses localhost. O1 must still use the PC's LAN address.

The default is one warm-up followed by three measured full `predict_action_chunk` calls. Every call performs new inference, producing 50 six-dimensional actions; cached `select_action` pops are not counted as model calls. The final chunk is unnormalized by the checkpoint's postprocessor before return. [Policy API](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py), [Processors](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/processor_smolvla.py)

Each task launches a fresh process and loads the model. This measures a repeatable load/inference workflow, not a persistent inference server's maximum throughput. Default task-completion and workflow timeouts are 300 and 600 seconds. If necessary, raise O1 `--task-timeout`, P2 `--task-completion-timeout`, and P2 `--workflow-timeout` together, for example to 600, 600, and 1200 seconds.

Use new output names for subsequent runs. Existing reports, including failed runs, are not overwritten.

## 8. Acceptance evidence and interpretation

```bash
.venv-hil/bin/python -m json.tool .mars-vla/smolvla-01.json
```

Confirm all of the following:

- Overall `status: succeeded`, `hardware_smoke_passed: true`, `gpu_tested: true`, final `valid: true`, and execution on two distinct physical hosts.
- `execution_evidence_kind` is `trusted_agent_report` and `hardware_gate_failures` is empty. A software fixture can validate business logic and transport, but is labeled `test_fixture` and cannot count as real GPU evidence.
- The two hosts have distinct machine IDs, the same commit and runtime source fingerprint; the GPU host is an aarch64 AGX Orin with compute capability 8.7 and L4T 39.2.1.
- The worker uses Python 3.12; Torch/TorchVision are within LeRobot 0.4.4's supported ranges, and startup preflight versions exactly match task execution.
- CUDA `smoke` executes on `robot_1` and validation on `edge_pc`, with matrix reference checks passing.
- SmolVLA `observe` and `validate` execute on `edge_pc`, `infer` on `robot_1`, with nonzero remote inputs for both inference and validation.
- Measurements identify the GPU, Torch/CUDA versions, actual CUDA input/output tensors, positive CUDA-event and synchronized-wall times, and positive allocated GPU memory.
- VLA output records immutable model revisions, verified and strictly loaded weights, CUDA parameter devices, actual parameter dtypes, 50 × 6 finite actions, and the observation/action hashes.
- Missing `camera3` is explicit; `physical_actuation` / `robot_control_executed` remains false.

| Report location | Expected value or content |
| --- | --- |
| `status` / `scope` | `succeeded` / `cross_host_cuda_execution` |
| `hardware_smoke_passed` / `gpu_tested` | `true` / `true` |
| `execution_evidence_kind` / `executing_host_count` | `trusted_agent_report` / `2` |
| `hardware_gate_failures` | `[]` |
| `jetpack_profile_evidence.passed` | `true`, `profile_scope` is `l4t_only`, observed L4T R39 revision 2.1; the prerequisite native CUDA loop separately accepts the system CUDA Runtime |
| `validation.valid` | `true` |
| `gpu_execution.agent_id` | `robot_1` by default |
| `gpu_execution.measurement` | Device identity, CUDA events, synchronized times, allocations |
| `gpu_execution.action_shape` | `[50, 6]` for SmolVLA |
| `gpu_execution.model` | Fixed revisions, verified weights, strict loading |
| `executions` | Task/host records and `remote_input_bytes` |
| `observation_source` | Dataset revision, episode/frame, camera mapping |
| `artifacts[].envelope.payload` | Full observation, action, and validation payloads |
| `physical_actuation` / `control_success_tested` | Both `false` |

CUDA-event timing measures the GPU stream. Synchronized wall timing measures the completed inference call. Model loading/preprocessing is recorded separately; neither measure is the entire network workflow duration. PyTorch allocator memory is not total Jetson system memory. Scheduling profiles are bootstrap estimates; reported CUDA timings and allocations come from execution. Energy and scheduling superiority are not established by this test.

For failures, inspect O1 first. Common causes are a CPU-only Torch install, incompatible TorchVision, a worker path pointing at `.venv-hil`, missing or corrupt model files, failed AV1 decoding during sample preparation, and a missing observation on the PC. `worker package lookup is not isolated` means `PYTHONHOME`, `PYTHONPATH`, or user site-packages can contaminate the installer; unset the overrides and rerun through `.venv-vla/bin/python`. `JetPack 7.1.2 is not an NVIDIA release` means the runner rejected that profile name; read `/etc/nv_tegra_release` and use `--require-jetpack 7.2.1` only when it says R39 revision 2.1. A `no_test_fixtures` failure means the result is developer test data rather than hardware evidence. A `cuda_preflight_matches_execution` failure means startup and task execution used different devices or software stacks. `worker_python_for_jetpack` or `smolvla_framework_versions` means the worker is not Python 3.12 or its Torch stack is outside the supported range. Correct the underlying environment before rerunning; do not remove verification to conceal it.

`tegrastats` can provide an additional view of Orin load, but sampling can miss short tasks. Keep the task's CUDA measurements and result checks as the primary records.

To place inference on a PC GPU, prepare its separate CUDA/VLA environment and model, exchange the `vla-cuda`/`vla-io` roles and observation location, then add `--gpu-agent edge_pc` on P2. Stop O1 and P1 with `Ctrl+C` when finished and retain the reports, model manifest, input provenance, and artifacts.
