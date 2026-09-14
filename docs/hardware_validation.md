# PC + AGX Orin 64GB: real CPU / native CUDA HIL guide

[Detailed Chinese instructions](hardware_validation_zh.md) · [Preserved CPU-only guide](hardware_cpu_validation.md) · [Preserved CPU-only Chinese guide](hardware_cpu_validation_zh.md)

This starts from an original **Jetson AGX Orin 64GB Developer Kit already successfully flashed with JetPack 7.2.1**, and an **Ubuntu 24.04 x86_64 PC**. Both use the same Wi-Fi; Ethernet is unnecessary. Continue within the installed systems: **SDK Manager is no longer needed; do not press Recovery or reflash** for this test.

The official baseline is **JetPack 7.2.1 / Jetson Linux (L4T) 39.2.1 / Ubuntu 24.04 / CUDA 13.2.1**. Verify the actual installation below. [NVIDIA JetPack downloads and release information](https://developer.nvidia.com/embedded/jetpack/downloads)

**NVIDIA has no release named JetPack 7.1.2.** JetPack 7.1 is L4T 38.4 and does not list Orin; Orin support in JetPack 7 starts at 7.2 / L4T 39.2. This guide reads `/etc/nv_tegra_release` and accepts 7.2.1 only for `R39, REVISION: 2.1`. [NVIDIA JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive)

**This is an execution and acceptance procedure, not a claim that your physical PC/Orin test has passed.** Software tests, CUDA fixtures, and an ARM64 compile-only result cannot establish real GPU execution. Hardware acceptance requires the actual reports and execution evidence described here.

No motors, ROS, web backend, model weights, PyTorch, or LeRobot are needed. The updated [VLA guide](vla_hardware_validation.md) is a separate later model step for JetPack 7.2.1 and is not a dependency. The CPU archives retain their historical setup instructions; use this page for the current environment.

## 1. Five tasks, six artifacts, eight data edges

| Task ID | Host / processor | Actual computation | Output |
| --- | --- | --- | --- |
| `sense` | Orin CPU | Seeded, simulated range survey from known poses; separate scene truth | `observations`, `truth` |
| `map` | PC CPU | Occupancy mapping from observations, without truth obstacles | `map` |
| `inflate` | Orin native CUDA | Full occupancy/unknown-space inflation for the robot footprint | `inflated` |
| `plan` | PC CPU | Four-connected A* **consuming the returned GPU mask**, then bounded trajectory construction | `trajectory` |
| `validate` | Orin CPU | Independent full-mask CPU reference, then continuous geometric trajectory validation | `validation` |

```text
Orin CPU sense → PC CPU map → Orin CUDA inflate → PC CPU plan → Orin CPU validate
```

The complete dependency edges are:

| Producer → consumer | Transport |
| --- | --- |
| `sense.observations` → `map.observations` | Orin → PC |
| `map.map` → `inflate.map` | PC → Orin |
| `map.map` → `plan.map` | PC local |
| `inflate.inflated` → `plan.inflated` | Orin → PC |
| `map.map` → `validate.map` | PC → Orin |
| `inflate.inflated` → `validate.inflated` | Orin local |
| `plan.trajectory` → `validate.trajectory` | PC → Orin |
| `sense.truth` → `validate.truth` | Orin local |

Each artifact is content-addressed JSON fetched and checksum-verified from its producing Agent. Identically named directories on the hosts are not shared storage. One artifact may serve multiple edges.

The fixed synthetic workload uses a 12 × 8 m scene, a 96 × 64 grid, and 20 known survey poses with 256 rays each. Sensor data and localization are synthetic; CPU/GPU computation, socket transfers, and timing are real. Validation compares all **6,144 cells**, verifies that planning consumed this GPU output, and checks continuous collision along trajectory segments, observed-space clearance, start/goal, pose/time continuity, speed, acceleration, and yaw rate. It is not SLAM or physical robot control. The coordinator additionally rechecks validation on PC during evidence collection; this is not a sixth DAG task.

## 2. Physical setup, login, terminals, and Wi-Fi

Connect suitable power to the ventilated Orin kit. If using its desktop, connect a compatible monitor to its display output and USB keyboard/mouse, select the display input, and boot normally with the power button if needed. Log in with the Ubuntu account created after flashing. Connect both machines to the same Wi-Fi through Ubuntu network settings. Do not use Recovery or flashing button combinations.

Open terminals with `Ctrl+Alt+T`; paste with `Ctrl+Shift+V`. `sudo` uses the current host's login password; invisible password input is normal. Commands work in **bash and zsh** without interactive `read` prompts. Stop at an error instead of continuing. Keep multiline backslashes as shown, without trailing spaces. No virtual-environment activation is required.

| Terminal | Actual host | Purpose |
| --- | --- | --- |
| O0 | Orin local terminal or SSH session | Setup and diagnostics |
| P0 | PC local terminal | Setup |
| O1 | **Orin**, local or reached by SSH | Keep the CPU/CUDA Agent running |
| P1 | **PC local**, not SSH | Keep the PC Agent running |
| P2 | **PC local**, separate terminal | Coordinator and reports |

**O0 on Orin and P0 on PC: execute on each host.**

```bash
hostname
uname -m
nmcli device status
ip -4 -br addr
```

Find the connected device whose type is `wifi`, then its matching IPv4 address. Interface names vary; do not assume `wlan0`. Strip the `/24` or other prefix length. Record both real addresses. **No actual addresses have been supplied: every `192.168.1.10` / `192.168.1.20` below is an example that must be replaced.** Do not select localhost, Docker/VPN bridges, or Jetson's USB virtual address `192.168.55.1`. If Orin has no working wireless interface, establish Wi-Fi first. Same-SSID guest/AP isolation can still prevent peer traffic.

**P0 on PC: set real addresses, inspect the Wi-Fi route, and test Orin.**

```bash
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
ip -4 route get "$ORIN_IP"
ping -c 3 "$ORIN_IP"
```

**O0 on Orin: set real addresses independently and test PC.**

```bash
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
ip -4 route get "$PC_IP"
ping -c 3 "$PC_IP"
```

The route should select the actual Wi-Fi interface and source address. Ping is diagnostic, not proof that TCP 50051 works; some networks block ICMP.

Optional SSH: **on Orin O0**, install/start SSH if absent:

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable --now ssh
systemctl status ssh --no-pager
```

If Orin UFW is active and blocks SSH, allow TCP 22 **only from the real PC IP** using the same scoped rule pattern as section 4. Then **in a new PC terminal**, replace the real address and Orin Ubuntu login name:

```bash
export ORIN_IP='192.168.1.20'
export ORIN_USER='your_orin_login_name'
ssh "${ORIN_USER}@${ORIN_IP}"
```

Verify a first-connection fingerprint against `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` **on Orin**, then accept and enter the Orin password. **Inside the SSH session**, run `hostname`, `uname -m`, and `whoami`; it must be the Orin account on aarch64. This session can be O0/O1, never P1/P2. `robot_1` is an Agent ID, not necessarily a login account.

## 3. Check installed software, time, and prerequisites

**O0 on Orin:**

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
tr -d '\000' < /proc/device-tree/model
printf '\n'
dpkg-query -W nvidia-l4t-core nvidia-jetpack
python3 --version
/usr/local/cuda/bin/nvcc --version
```

Expect Ubuntu 24.04, `R39` with `REVISION: 2.1`, a model containing `Jetson AGX Orin`, Python 3.12.x, and the installed CUDA 13.2 toolchain matching the official 13.2.1 baseline. A missing `nvidia-jetpack` metapackage alone is not decisive; inspect L4T and actual CUDA execution. CUDA version text/runtime integers may omit patch releases (`13020` denotes 13.2). Compiler availability alone is not GPU verification.

**P0 on PC:**

```bash
cat /etc/os-release
uname -m
python3 --version
```

Expect Ubuntu 24.04, x86_64, Python 3.12.x. PC needs neither NVIDIA GPU nor CUDA.

**O0 and P0, each host: check time before downloads.**

```bash
date -Iseconds
timedatectl status
```

If the clock is wrong, especially an Orin reboot to 1970, or NTP is disabled, **on the affected host**:

```bash
sudo timedatectl set-ntp true
timedatectl status
timedatectl timesync-status
```

Wait and recheck the date and synchronization. If another NTP service is installed, inspect that service; `timesync-status` may not apply. Resolve network/DNS/NTP restrictions instead of hardcoding today's date, disabling TLS verification, or reflashing.

**O0 and P0, each host: install prerequisites.**

```bash
sudo apt update
sudo apt install python3-venv git build-essential
```

`build-essential` supplies GCC/G++, required by `nvcc` on Orin. Do not replace system Python or install Ubuntu generic NVIDIA drivers, run driver autoinstall, or reinstall a different CUDA stack over JetPack.

## 4. Conditional firewall rules

Both hosts must accept TCP 50051 from the other. Use a trusted LAN; current gRPC is unencrypted/unauthenticated. Do not forward this port to the Internet.

**O0 and P0, each host:**

```bash
if command -v ufw >/dev/null 2>&1; then
  sudo ufw status verbose
else
  printf 'UFW not installed; inspect other active firewalls if needed.\n'
fi
```

If inactive, leave it unchanged. Only if UFW is **active** and a rule is needed:

**O0 on Orin, real PC address:**

```bash
export PC_IP='192.168.1.10'
sudo ufw allow from "$PC_IP" to any port 50051 proto tcp
sudo ufw status numbered
```

**P0 on PC, real Orin address:**

```bash
export ORIN_IP='192.168.1.20'
sudo ufw allow from "$ORIN_IP" to any port 50051 proto tcp
sudo ufw status numbered
```

For optional SSH only, **Orin O0 with active UFW** may use `sudo ufw allow from "$PC_IP" to any port 22 proto tcp` after setting the real PC address. Do not disable the firewall or allow all sources. Recheck IPs/rules after DHCP changes. Port connection refusal before Agents start is expected.

## 5. Same clean code and Python environment on both hosts

Perform this section **once on Orin O0 and once on PC P0**. Use `~/mars-hardware` on each host. Preserve existing changes; stop Agents before updating.

For a fresh checkout, only when the destination does not already exist:

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
```

For an existing checkout, first inspect its remote and confirm `origin` is the repository above:

```bash
cd "$HOME/mars-hardware"
git remote -v
```

Then update only a clean checkout already on the intended branch:

```bash
(
  set -e
  cd "$HOME/mars-hardware"
  git status --short
  if [ -n "$(git status --porcelain)" ]; then
    printf 'STOP: preserve and review local changes before updating.\n' >&2
    exit 1
  fi
  if [ "$(git branch --show-current)" != 'codex/grpc-hardware-loop' ]; then
    printf 'STOP: review the existing branch before updating.\n' >&2
    exit 1
  fi
  git fetch origin codex/grpc-hardware-loop
  git merge --ff-only origin/codex/grpc-hardware-loop
)
```

Do not force-reset, clean, overwrite, or automatically stash changes. If the branch diverged, preserve/review it or use a separate fresh directory and adjust all subsequent paths.

**Both hosts, respective setup terminal:**

```bash
cd "$HOME/mars-hardware"
git status --short
git rev-parse HEAD
ls scripts/mixed_smoke.py scripts/build_cuda_smoke.py
ls examples/mixed_workloads/inflate.cu agent/requirements-hardware.txt
```

Compare the complete commit IDs: they must match. Missing files mean this checkout does not yet contain the mixed workflow. Branch-name equality alone is insufficient.

Check any existing `.venv-hil` before reuse; preserve an incompatible or unrelated environment instead of overwriting it. With system Python 3.12 and a new/compatible environment, **on each host**:

```bash
cd "$HOME/mars-hardware"
python3 --version
python3 -m venv .venv-hil
.venv-hil/bin/python --version
.venv-hil/bin/python -m pip install -r agent/requirements-hardware.txt
.venv-hil/bin/python -m pip check
.venv-hil/bin/python -m agent.main --help
.venv-hil/bin/python -m scripts.mixed_smoke --help
```

Both interpreters should be 3.12.x. Do not copy a venv between architectures. Always invoke `.venv-hil/bin/python`; do not add the VLA dependencies.

**Both hosts: compare actual source content as well as the commit.**

```bash
cd "$HOME/mars-hardware"
.venv-hil/bin/python - <<'PY'
import json
import agent.telemetry
from agent.telemetry import _runtime_identity
print('loaded telemetry:', agent.telemetry.__file__)
print(json.dumps(_runtime_identity(), indent=2))
PY
```

The imported module must belong to this checkout. `git_revision` and `runtime_source_sha256` must match across hosts; `machine_id_sha256` must differ. The source fingerprint hashes actual `.py`, `.cu`, and `.proto` files under `agent`, `examples`, `interfaces`, `mars`, and `scripts`. Final acceptance checks the identities reported by the **Agents that actually executed**, not just these setup outputs. Restart both Agents after code changes; identity is sampled at startup. Do not modify code/binaries during a run. This is trusted-host deployment evidence, not adversarial hardware attestation.

## 6. Orin only: build and execute a real CUDA probe

**O0 on Orin, never PC:**

```bash
cd "$HOME/mars-hardware"
g++ --version
/usr/local/cuda/bin/nvcc --version
.venv-hil/bin/python -m scripts.build_cuda_smoke --output .mars-hil/bin/inflate_cuda
```

The default target is **`sm_87`**. The build writes the binary and adjacent `inflate_cuda.manifest.json`, binding source/binary hashes, compiler, flags, and build host. By default it then runs a **real GPU kernel probe**, compares the entire output to an independent reference, and requires finite positive timing measurements. There is no CPU fallback.

Require `compiled: true` **and** `runtime_verified: true`, `gpu_info.available: true`, `gpu_info.kernel_execution_verified: true`, backend `cuda_runtime`, the correct device (AGX Orin compute capability `[8, 7]`), and positive `measurement.cuda_event_ms` / `synchronized_wall_ms` samples. **`--compile-only` explicitly leaves `runtime_verified: false` and cannot establish GPU success**, whether local or in CI.

Retain probe output and the manifest. Stale/changed source or binary hashes fail provenance checks. Stop the Agent and rebuild locally after changes; never edit a manifest to bypass validation or copy a PC x86_64 binary to Orin.

## 7. O1 and P1: start both Agents and leave them running

**O1 on Orin**: open a local Orin terminal or SSH into it. Replace both IPs in this new terminal, verify aarch64, then start:

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
printf 'O1 Orin; PC peer=%s:50051\n' "$PC_IP"
```

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

Startup performs CUDA preflight again. Require `REAL CPU + native CUDA mixed navigation` and `robot_1 listening on 0.0.0.0:50051`, not MOCK or the old navigation executor. Keep O1 open/running.

**P1 on PC local**: open another terminal, replace both IPs independently, verify x86_64, then start:

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
printf 'P1 PC; Orin peer=%s:50051\n' "$ORIN_IP"
```

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

Require `REAL CPU mixed mapping/planning` and `edge_pc listening on 0.0.0.0:50051`. Keep P1 and O1 running. Port 50051 on two different machines does not conflict. Do not supply mock `--config`; business subprocesses start automatically.

## 8. Verify TCP in both directions

**Orin O0, or another Orin/SSH terminal; do not type into the running O1:**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
.venv-hil/bin/python - <<'PY'
import os
import socket
with socket.create_connection((os.environ['PC_IP'], 50051), timeout=5):
    print('OK: Orin -> PC:50051')
PY
```

**New PC local terminal P2:**

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
hostname
uname -m
.venv-hil/bin/python - <<'PY'
import os
import socket
for host in (os.environ['ORIN_IP'], '127.0.0.1'):
    with socket.create_connection((host, 50051), timeout=5):
        print(f'OK: PC -> {host}:50051')
PY
```

All checks must succeed. Localhost is correct for **P2 → PC Agent only**, never for Orin's PC peer. TCP reachability is preliminary, not business acceptance.

## 9. P2: run three sequential hardware workflows

**P2 on PC local**: set real addresses again so this block works in a fresh terminal. Generate a unique report path; do not create the file in advance.

```bash
cd "$HOME/mars-hardware"
export PC_IP='192.168.1.10'
export ORIN_IP='192.168.1.20'
mkdir -p .mars-hil/reports
export HIL_REPORT=".mars-hil/reports/mixed-$(.venv-hil/bin/python -c 'from datetime import datetime, timezone; from uuid import uuid4; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex)').json"
printf 'Save this report path: %s\n' "$HIL_REPORT"
```

**P2, same terminal, with O1/P1 still running:**

```bash
.venv-hil/bin/python -m scripts.mixed_smoke \
  --agent "robot_1=$ORIN_IP:50051" \
  --agent edge_pc=127.0.0.1:50051 \
  --seed 19 \
  --runs 3 \
  --require-jetpack 7.2.1 \
  --output "$HIL_REPORT"
HIL_EXIT_CODE=$?
printf 'mixed_smoke exit code: %s\n' "$HIL_EXIT_CODE"
```

This uses `CentralCoordinator` through `GrpcRuntimeAdapter`. Seeds are **19, 20, 21**, each a complete five-task workflow; the first failure stops later runs. Output creation is exclusive: an existing filename is refused, never overwritten. Do not redirect stdout to the report path with `>`.

Normal hardware mode already requires two distinct hosts, the PC/Orin architectures, AGX Orin model, compute capability `[8, 7]`, and matching commit/runtime provenance. **Every primary hardware command here additionally supplies `--require-jetpack 7.2.1`, requiring L4T `R39 / REVISION: 2.1` and CUDA Runtime 13.2 (`13020`).** The CLI's version flag is opt-in; omitting it does not establish this guide's strict version acceptance. Manual Ubuntu/Python/CUDA inspection remains necessary.

**Do not add `--allow-same-host`**: it is development transport mode and can never produce hardware acceptance, even if its overall status succeeds. Fixtures and compile-only evidence also do not prove GPU execution.

| Timeout / repetition | Value in this guide |
| --- | --- |
| Agent `--task-timeout` | Explicit 90 seconds per task |
| Runner `--workflow-timeout` | Default 180 seconds per workflow phase |
| Runner `--task-completion-timeout` | Default 120 seconds; cannot exceed workflow timeout |
| Runner `--evidence-timeout` | Default 60 seconds per evidence phase |
| Orin `--cuda-repeats` | Three measured kernel runs, plus one warmup |
| Runner `--runs` | Three full workflows, with sequential seeds |

The 180-second limit is not a total bound for all three runs. Diagnose the error/phase before increasing limits; native subprocesses have their own bounds. Do not run concurrent coordinators, rebuild, change source, or restart Agents during the three runs.

## 10. Inspect and accept the actual report

**P2 on PC:**

```bash
printf 'Report: %s\n' "$HIL_REPORT"
.venv-hil/bin/python -m json.tool "$HIL_REPORT" | less
```

Use Space to page, `/hardware_smoke_passed` to search, `n` for the next match, and `q` to quit. In a new terminal, first enter the checkout and restore `HIL_REPORT` to the saved actual path; do not generate a new UUID to read an old report.

Require all of the following:

1. Exit code 0; aggregate `status: "succeeded"`, `error: null`, `hardware_smoke_passed: true`, `gpu_tested: true`, and `scope: "cross_host_cpu_native_cuda_execution"`. `allow_same_host` is false and `required_jetpack: "7.2.1"`.
2. `requested_runs` and `completed_runs` are 3; all three `runs` have seeds 19/20/21, succeeded status, no error, and hardware acceptance. A partial report is not a three-run pass.
3. Each run contains five `executions`, six `artifacts`, and eight `edges`, with exactly the placements in section 1. All source/attempt/input identities and checksums must pass the runner's checks.
4. Each run's `hosts` and `executions[].host` identify the actual x86_64 PC and aarch64/arm64 AGX Orin; `executing_host_count` is 2 with `host_count_basis: "machine_id_sha256"`. Both hosts have matching `git_revision` and `runtime_source_sha256`; machine hashes differ. `checks.jetpack_profile`, `jetson_agx_orin`, `target_architectures`, `distinct_machine_ids`, `matching_git_revision`, and `matching_runtime_source` are true.
5. `gpu_execution` identifies `inflate` on `robot_1`. Its `measurement.backend` is `cuda_runtime`, source/binary hashes agree with the Orin preflight, and both `cuda_event_ms` and `synchronized_wall_ms` have three finite values **greater than zero**.
6. Each `validation` is valid, `gpu_full_reference_match` is true, and `gpu_cells_checked` is 6144. Its checks include `gpu_full_grid_cpu_reference` and `gpu_output_used_by_planner`; the trajectory's `inflation_sha256` matches this GPU payload. `checks.independent_validation` is true.
7. Both directions carry real bytes: `map`/`plan` execution records have positive `remote_input_bytes` from Orin, and `inflate`/`validate` have positive bytes from PC. All eight `edges[].remote_bytes` follow the local/remote distinction above.
8. `checks.artifact_ports`, `execution_placement`, `host_identity_consistent`, `edge_transfers`, `source_lineage`, `cuda_measurement`, `no_test_fixtures`, and `native_binary_identity` are true; `hardware_gate_failures` is empty.

`artifacts[].reference.checksum` hashes the transported envelope, while `payload_sha256` and business `source_hashes` hash payload contents; these are different scopes. Validation identifies the consumed map, GPU mask, trajectory, and truth. The [Chinese guide's report reader](hardware_validation_zh.md#123-可复制的报告核对命令) provides a copyable read-only check for these acceptance conditions.

Independent validation requires exact integer masks, hashes, structures, and booleans. Derived floating-point metrics use `rel_tol=1e-9` and `abs_tol=1e-9` to allow last-bit differences between x86_64 and ARM64 math libraries. The report records this as `validation_float_tolerance`; collision and full-mask checks remain unchanged.

## 11. Measurement limits

CUDA event timing brackets the inflation kernel only; synchronized wall timing includes launch/event/synchronization overhead. Neither includes allocation, host/device transfer, networking, or Python startup. Worker timing includes subprocess startup, computation, and I/O; `input_fetch_ms` separately records input retrieval. Per-run workflow/evidence/total wall times and aggregate total wall time have different scopes.

Remote-byte metrics count consumed artifact envelopes, not total Wi-Fi traffic or the coordinator's final evidence downloads; repeated consumers can count the same artifact again. CPU/memory observations describe the whole host, using actual CPU-counter sample windows of at least about 100 ms; short tasks may reuse a sample. Agent monotonic timestamps cannot be compared as synchronized cross-host wall clocks.

This tiny 6,144-cell kernel is a correctness and execution-path smoke test, **not a GPU benchmark**. An idle or sparsely sampled GPU reading of 0% is normal and may miss the kernel entirely. Legacy GPU-utilization fields may also be explicitly unmeasured zero placeholders. Device allocation bytes are explicit helper allocations, not peak process/device memory. **Energy is unmeasured (`energy_j: null`)**; scheduling profiles/link bandwidth and legacy zero energy values are assumptions/placeholders, not measurements. Planned trajectory duration is mathematical output; no robot moves. Fixed placement does not prove scheduling optimality or GPU speedup.

## 12. Troubleshoot, retain evidence, and stop

| Failure | Action |
| --- | --- |
| 1970 clock / TLS or package-date errors | Fix NTP on the affected host; do not hardcode a date or bypass TLS. |
| Connection refused | Check O1/P1 startup and listen address. In an idle terminal on the affected host, use `ss -ltnp 'sport = :50051'`. |
| Timeout / remote-input failure | Recheck both real Wi-Fi addresses, routes, peer maps, scoped firewall rules, and AP isolation. Both TCP directions must work. |
| Wrong/missing mixed CLI or imports | Correct checkout and `.venv-hil/bin/python`; obtain the same complete commit, install hardware requirements, and run `pip check`. |
| Missing nvcc / g++ | On Orin inspect `/usr/local/cuda/bin/nvcc`, `ls -ld /usr/local/cuda*`, and `dpkg-query -W 'cuda-nvcc*'`; install `build-essential` for GCC. Repair missing components using the matching NVIDIA JetPack package source; do not install generic Ubuntu drivers or a different CUDA stack. Use `--nvcc` only for a verified alternate installed compiler path. |
| Unsupported architecture / no kernel image / exec format | Build locally on AGX Orin with the correct CUDA toolkit and default `sm_87`; do not copy PC binaries or arbitrarily change architecture flags. |
| Compiled but probe failed | GPU execution remains unverified. Inspect the real CUDA diagnostic/device access and resolve it; compile-only is not a workaround for acceptance. |
| Source/binary provenance mismatch | Preserve diagnostics, stop O1, align source, rebuild/probe locally, and restart. Never edit manifest hashes to bypass the error. |
| Commit/runtime-source/host identity gate failed | Inspect the actual report hosts, local changes, wrong SSH destinations, and stale running Agents. Preserve work, align deployments, restart both, and rerun with a new filename. |
| AGX model / `jetpack_profile` gate failed | Verify device-tree model and L4T R39 revision 2.1 on Orin; do not falsify identity files or remove the flag. |
| Nonpositive/nonfinite GPU timing | Treat evidence as invalid; preserve original values and diagnose kernel/synchronization errors. Do not substitute estimates. |
| CPU full-reference / planner-consumption / source-lineage failure | Preserve all six artifacts, seed, source/binary hashes, and exact error. Do not replace the GPU output with a CPU result or bypass validation. |
| Workflow timeout | Inspect Agent logs, Wi-Fi and load; distinguish Agent 90 seconds from workflow 180/completion 120. |
| Evidence timeout | Output retrieval or independent PC verification did not finish; keep Agents available and inspect networking/PC load. Finished tasks without complete evidence do not pass. |
| Only one/two runs | Inspect the last run's `error`, `phase`, and `hardware_gate_failures`; fix and rerun all three to a fresh report path. |
| Output exists | Generate a new UUID path; do not overwrite/delete prior evidence or pre-create the destination. |
| Port already used / attempt history full | Identify your earlier Agent before stopping it. Restart only after a completed/stopped coordinator, not during the three-run sequence. |

After P2 finishes, retain the report plus PC `.mars-hil/reports/received-artifacts/`, PC `.mars-hil/edge_pc/`, Orin `.mars-hil/robot_1/`, native binary/manifest, probe output, and both Agent terminal diagnostics. The report also embeds collected artifacts and execution records.

Press `Ctrl+C` in **P1**, then **O1**, and wait for prompts. If O1 was SSH, `exit` returns to the PC after its Agent stops. Shut down Orin through Ubuntu before removing power. If you interrupt P2, also inspect/stop both Agents; a closed coordinator window is not proof that remote work was cancelled, and a complete report may not yet have been written. Never delete live artifact directories.

For the next attempt, recheck IPs, time, source identity, and CUDA build; restart O1/P1 and use a new report filename. Only real accepted three-run evidence supports recording that this particular hardware/software deployment passed.
