# Real CUDA and SmolVLA validation: PC + Jetson AGX Orin 64GB

This workflow adds real CUDA matrix multiplication and pretrained SmolVLA inference to MARS. A PC supplies a recorded robot observation; MARS schedules inference on the Orin GPU; the PC receives and validates the returned actions. The [existing CPU navigation workflow](hardware_validation.md) remains available.

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

The PC requires no GPU or ML libraries. Each host uses `.venv` for the lightweight MARS Agent. Orin additionally uses `.venv-vla` for CUDA/LeRobot. These must remain separate: MARS requires protobuf 7, while LeRobot's WandB dependency requires protobuf below 7.

## 2. Common Agent setup

For a new checkout on each host:

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
cd "$HOME/mars-hardware"
git rev-parse HEAD
python3.10 -m venv .venv
.venv/bin/python -m pip install -r agent/requirements-hardware.txt
.venv/bin/python -m agent.main --help
.venv/bin/python -m scripts.vla_loop --help
```

An existing checkout must be updated to the GPU/VLA implementation, preserving local changes. Compare commit IDs on both hosts. Reuse the CPU workflow's `.venv` if already prepared. PC Python may be 3.10 or newer; the following native Jetson wheel example specifically uses Python 3.10.

## 3. Orin CUDA environment

The user's JetPack version is not yet known. Inspect it before selecting packages:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack
python3.10 --version
uname -m
```

The JetPack meta-package may be absent; use the L4T version and installed CUDA stack as additional evidence. Do not rely solely on the availability of `nvidia-smi` on Jetson.

The commands below target **JetPack 6.2 / CUDA 12.6, Ubuntu 22.04, aarch64, Python 3.10**, using Torch 2.8.0, TorchVision 0.23.0, LeRobot 0.4.4, and Transformers 4.57.1. The Torch-TensorRT project documents this JetPack package source, and LeRobot 0.4.4 supports these versions. [JetPack package instructions](https://docs.pytorch.org/TensorRT/getting_started/jetpack.html), [LeRobot requirements](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

For another JetPack release, first obtain a matching CUDA-enabled Torch/TorchVision build using the [NVIDIA installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and [compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html). Do not replace the system Python.

On Orin, for the stated combination:

```bash
cd "$HOME/mars-hardware"
python3.10 -m venv .venv-vla
.venv-vla/bin/python -m pip install --upgrade pip
.venv-vla/bin/python -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
.venv-vla/bin/python -m scripts.install_vla --install
.venv-vla/bin/python -m scripts.install_vla
```

The installer executes CUDA and TorchVision CUDA checks, pins the installed Torch/TorchVision versions while installing LeRobot, and checks the resulting imports. The final read-only check should report `status: ready`, the GPU/CUDA identity, LeRobot `0.4.4`, and Transformers `4.57.1`. It rejects an environment containing the MARS protobuf 7 stack.

An existing separate CUDA environment may be used instead; run the installer through its interpreter and give that absolute interpreter path to O1's `--worker-python`.

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

Expect `REAL CUDA VLA` after the CUDA probe and asset validation. Keep O1 running. The peer is the PC's LAN address; the worker interpreter must exist on Orin. CUDA unavailability is an error; there is no CPU fallback.

For a CUDA-only check, omit `--model-dir` and skip asset preparation. That Agent will not advertise SmolVLA support.

## 6. P1: Input and validation Agent on PC

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

Expect `REAL CPU VLA input/validation`. Keep P1 running. CUDA-only testing can omit `--observation-file`.

## 7. P2: Execute both workflows from PC

First check real CUDA multiplication and its independently checked result:

```bash
cd "$HOME/mars-hardware"
.venv/bin/python -m scripts.vla_loop \
  --workload cuda \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/cuda-01.json \
  --require-distinct-hosts
```

Then execute pretrained SmolVLA:

```bash
.venv/bin/python -m scripts.vla_loop \
  --workload smolvla \
  --agent robot_1=192.168.1.20:50051 \
  --agent edge_pc=127.0.0.1:50051 \
  --output .mars-vla/smolvla-01.json \
  --require-distinct-hosts
```

The coordinator is on the PC, so its PC endpoint uses localhost. O1 must still use the PC's LAN address.

The default is one warm-up followed by three measured full `predict_action_chunk` calls. Every call performs new inference, producing 50 six-dimensional actions; cached `select_action` pops are not counted as model calls. The final chunk is unnormalized by the checkpoint's postprocessor before return. [Policy API](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py), [Processors](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/processor_smolvla.py)

Each task launches a fresh process and loads the model. This measures a repeatable load/inference workflow, not a persistent inference server's maximum throughput. Default task-completion and workflow timeouts are 300 and 600 seconds. If necessary, raise O1 `--task-timeout`, P2 `--task-completion-timeout`, and P2 `--workflow-timeout` together, for example to 600, 600, and 1200 seconds.

Use new output names for subsequent runs. Existing reports, including failed runs, are not overwritten.

## 8. Acceptance evidence and interpretation

```bash
.venv/bin/python -m json.tool .mars-vla/smolvla-01.json
```

Confirm all of the following:

- Overall `status: succeeded`, final `valid: true`, and execution on two distinct hosts.
- CUDA `smoke` executes on `robot_1` and validation on `edge_pc`, with matrix reference checks passing.
- SmolVLA `observe` and `validate` execute on `edge_pc`, `infer` on `robot_1`, with nonzero remote inputs for both inference and validation.
- Measurements identify the GPU, Torch/CUDA versions, actual CUDA input/output tensors, positive CUDA-event and synchronized-wall times, and positive allocated GPU memory.
- VLA output records immutable model revisions, verified and strictly loaded weights, CUDA parameter devices, actual parameter dtypes, 50 × 6 finite actions, and the observation/action hashes.
- Missing `camera3` is explicit; `physical_actuation` / `robot_control_executed` remains false.

| Report location | Expected value or content |
| --- | --- |
| `status` / `scope` | `succeeded` / `cross_host_cuda_execution` |
| `gpu_tested` / `executing_host_count` | `true` / `2` |
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

For failures, inspect O1 first. Common causes are a CPU-only Torch install, incompatible TorchVision, a worker path pointing at `.venv`, missing or corrupt model files, failed AV1 decoding during sample preparation, and a missing observation on the PC. Correct these errors before rerunning; do not remove verification to conceal them.

`tegrastats` can provide an additional view of Orin load, but sampling can miss short tasks. Keep the task's CUDA measurements and result checks as the primary records.

To place inference on a PC GPU, prepare its separate CUDA/VLA environment and model, exchange the `vla-cuda`/`vla-io` roles and observation location, then add `--gpu-agent edge_pc` on P2. Stop O1 and P1 with `Ctrl+C` when finished and retain the reports, model manifest, input provenance, and artifacts.
