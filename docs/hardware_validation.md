# PC + Jetson Orin hardware validation

This page retains the CPU navigation procedure. For real CUDA and pretrained
SmolVLA inference with a recorded robot observation, continue with the
[GPU/VLA runbook](vla_hardware_validation.md). The ML worker uses a separate
environment so its dependencies do not replace the MARS communication stack.

This runbook runs actual CPU computations through MARS on a PC and an Orin.
The PC runs the central coordinator; both machines run an execution Agent.
Each accepted task launches a bounded workload subprocess, so there is no
separate business server to start. FastAPI, Vite, DDS, ROS 2, and an LLM are
not needed for this test.

The business workload is an original, small navigation pipeline. It is not
YOLO, VLA, SLAM, a CUDA benchmark, or a vehicle controller. Sensor observations
and localization are synthetic; mapping, graph search, trajectory construction,
collision checking, network transfer, and execution timing are real. No physical
actuation is performed. Passing this test establishes a distributed computation
closed loop, not production readiness or a scheduling-performance advantage.

## 1. Understand what will run

The default `split` placement produces this dependency graph:

```text
Orin: hil_sensor -- observations --> PC: hil_mapping -- map --> PC: hil_planning
         |                                  |                          |
         | truth                            | map                      | trajectory
         +----------------------------------+--------------------------+
                                            |
                                            v
                                  Orin: hil_validation
```

- `hil_sensor` ray-casts a seeded, static 2D scene from known survey poses,
  producing noisy range observations and separate ground truth. These are
  generated inputs, not captured sensor data.
- `hil_mapping` integrates the actual observations into free, occupied, and
  unknown grid cells. It does not receive the ground-truth obstacles.
- `hil_planning` inflates blocked and unknown space for the circular robot
  footprint, performs four-connected A* search, and constructs stop/turn/translate
  segments with bounded speed, acceleration, and yaw rate.
- `hil_validation` consumes the actual map, trajectory, and independent truth.
  It checks input identities, start/goal, continuous geometric collision,
  observed-space clearance, and motion limits. An invalid result fails the task.

Every data edge carries a real content-addressed JSON artifact. Remote inputs
are fetched from configured Agent peers and checked against their SHA-256
digest; a local file path on another machine is not treated as shared storage.
The sensor seed defines the fixture, not a fabricated completion time.

## 2. Prepare both machines once

The commands below assume Linux, a working Python 3.10 or newer installation,
and this repository checked out as `~/mars-hardware` on both hosts. Use the
same reviewed hardware-loop revision on each machine. Until merged, the
implementation branch is `codex/grpc-hardware-loop`; the original
`feature/grpc-runtime` revision alone only provides mock business execution.

A fresh checkout can be obtained on each host:

```bash
git clone --branch codex/grpc-hardware-loop \
  https://github.com/wangshiwen-ai-hku/capstone-simulator.git \
  "$HOME/mars-hardware"
```

For an existing checkout, use its actual directory and verify its revision;
do not overwrite local changes. After merge, use the branch containing the
merged implementation instead of the temporary branch name.

On **each machine**, open a setup terminal:

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

Replace `python3.10` with your installed supported interpreter if necessary.
Do not replace Orin's system Python. Ubuntu 20.04's default Python may be too
old; resolve the interpreter/dependency installation before starting services.
This CPU workload needs no model weights, CUDA, TensorRT, or pretrained data.
Both the Agent and command-line coordinator use the lightweight hardware
requirements; installing the web backend is unnecessary.

## 3. Check the LAN

The remaining commands use these example addresses; substitute the real ones:

| Host | LAN address | Agent ID | Agent port |
| --- | --- | --- | --- |
| PC | `192.168.1.10` | `edge_pc` | `50051` |
| Orin | `192.168.1.20` | `robot_1` | `50051` |

On each host, inspect addresses with `ip -br addr`. From the PC:

```bash
ping -c 3 192.168.1.20
```

From the Orin:

```bash
ping -c 3 192.168.1.10
```

Both machines must be able to reach the other's TCP port `50051`. If a firewall
is enabled, permit that port only from the other host's LAN address. Ping alone
does not prove the Agent port is reachable. The PC may retain Internet access
over Wi-Fi while using Ethernet to reach the Orin.

The MVP uses unencrypted, unauthenticated gRPC. Run it only on an isolated or
trusted LAN. Do not forward its port through a router, expose it to the Internet,
or point Agents at untrusted peers. The peer list limits artifact destinations;
it is not authentication for incoming connections.

## 4. Orin terminal O1: start the execution Agent

Open an Orin terminal, either locally or using a new PC terminal with
`ssh YOUR_ORIN_USER@192.168.1.20`. Commands after SSH execute on the Orin.

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

Keep this terminal running. Confirm the startup message says `REAL CPU
navigation`, not `MOCK`. Do not supply the old synthetic `--config` file in
navigation mode. CPU and memory capacity are detected from the host.

The Agent starts business workers on demand. No separate YOLO process or
`business-worker` server is needed for this bundled workload.
Each Agent permits one active attempt at a time. It retains at most 1,024
accepted attempts in memory; restart it between completed test runs if that
limit is reached. Attempt history is not recovered after restart.

## 5. PC terminal P1: start the edge execution Agent

Open a **new PC terminal**, separate from any SSH terminal used for O1:

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

Keep this terminal running. Port `50051` on two different machines does not
conflict. The PC Agent's peer must be the Orin's LAN address, not localhost.

## 6. PC terminal P2: run the coordinator and collect evidence

Open a **second PC terminal**. This is a finite command, not a third permanent
service:

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

Here, `127.0.0.1` is correct for the coordinator's connection to the PC Agent.
It is not correct for the peer address that the Orin uses to fetch PC results.
Leave both Agent terminals running until the command completes.

The runner invokes `CentralCoordinator` through `GrpcRuntimeAdapter`, rather
than calling the four business functions directly. Task completion makes its
downstream tasks ready; inventory refreshes supply actual CPU/memory observations
to scheduling. `--require-distinct-hosts` compares the hostname/architecture
pairs reported by the tasks that actually executed, catching accidentally
running both Agents on one machine. It is a deployment sanity check, not
cryptographic proof of hardware identity.

Inspect the saved evidence:

```bash
python -m json.tool .mars-hil/run-19.json
```

The output path must not already exist: the CLI refuses to overwrite an earlier
report. Choose a new filename for every attempt, including failed runs.
The coordinator also saves fetched artifacts in
`.mars-hil/received-artifacts/` beside this report.

## 7. Check success and interpret measurements

Confirm all of the following, not just a successful gRPC connection:

1. The runner exits with code 0, `status` is `succeeded`, and `error` is null.
   All four tasks completed through the coordinator.
2. `scope` is `cross_host_cpu_execution`, `executing_host_count` is 2, and
   `executions[].host` matches the PC and Orin. `robot_1` and `edge_pc` must not
   merely be two Agent IDs for one host.
3. `executions` places task IDs `sense`/`validate` on `robot_1` and `map`/`plan`
   on `edge_pc` for the split run.
4. `validation.valid` is true. `validation.source_hashes` identifies the actual
   map, trajectory, and truth payloads consumed by validation.
5. `remote_input_bytes` is positive. Both the `map` and `validate` execution
   records have positive remote input bytes, proving data moved in both
   directions. `artifacts[].reference.checksum` identifies each transferred
   envelope; it is different from a hash of the business payload alone.
6. `worker_elapsed_ms`, `workflow_wall_elapsed_ms`, and `total_wall_elapsed_ms`
   contain measured durations, not just estimates from a scheduling plan.

`executions[].host_observations.before` and `.after` preserve actual host-wide
CPU/memory samples before input fetching and after the business worker returns.
The same samples are retained in each output artifact's `envelope.execution`.
`cpu_utilization_ratio` and `memory_utilization_ratio` range from 0 to 1;
`memory_total_bytes` and `memory_available_bytes` are byte counts.
`sampled_at_ms` is milliseconds on that Agent's monotonic elapsed clock, not a
timestamp comparable between PC and Orin. CPU readings use independent raw
host-counter deltas over at least 100 ms. `cpu_sample_window_start_ms` and
`cpu_sample_window_ms` identify that interval. Faster calls reuse the cached
observation with its original timestamps; a short task's before/after values
can therefore be identical. The Agent waits for a real initial sample before
serving state. These samples are neither a whole-task average nor a peak;
genuine 100% host saturation is retained. Memory is sampled at the interval's
end. These observations describe all activity on the host, not exclusively the
workload process. Failed or timed-out invocations retain an after-observation in the Agent's local
execution log, but do not produce successful output artifacts.

`final_node_observations` retains the last runtime-inventory CPU/memory ratios
and online state for each node, including runs rejected before business work
starts. These are also host-wide observations, not process-exclusive usage.

`executions[].worker_elapsed_ms` includes subprocess startup, input/output
communication, and computation. The top-level `worker_elapsed_ms` sums those
four durations. `workflow_wall_elapsed_ms` covers coordinator execution;
`total_wall_elapsed_ms` also includes collection of the output evidence.
`remote_input_bytes` counts application payload bytes consumed by the business
tasks, not all wire traffic or the coordinator's final evidence downloads.

Scheduled compute/link costs remain the `planning_assumptions`, not calibration
results. Scheduler timestamps are logical anchors plus measured elapsed time,
not synchronized wall clocks from the two machines. In particular, energy or
zero-valued legacy Proto placeholders in `coordinator_report` are not hardware
measurements. Top-level and execution `energy_j` are null; each execution's
`host.unavailable` explicitly lists power, energy, GPU utilization, and temperature.
Do not read unavailable readings as measured zero consumption.

The trajectory's planned motion duration is mathematical output, not elapsed
execution time: no robot moves for that duration. Repeat runs can produce the
same business results from a seed, but measured timing naturally varies.

## 8. Repeat placements or test locally first

For placement checks, repeat the P2 command with another output filename and:

- `--placement orin`: all four tasks pinned to Orin.
- `--placement edge`: all four tasks pinned to PC.
- `--placement split`: sensor/validation on Orin, mapping/planning on PC.
- `--placement auto`: keep sensor/validation on Orin and let the scheduler
  place mapping/planning among eligible nodes.

Remove `--require-distinct-hosts` for `orin` and `edge`: those intentionally
execute on a single host and the CLI rejects that combination. Keep both Agents
registered in this runbook; `orin` mode can also run with only the `robot_1`
endpoint. `edge` mode still needs the source `robot_1` registered.

A pinned placement verifies where work runs; it does not show that the optimizer
selected an optimal split. Auto mode is not guaranteed to use both hosts or
transfer data across the LAN. Omit `--require-distinct-hosts` for auto mode if a
legitimate all-Orin placement should pass.

For a **single-machine development smoke test**, open three local terminals.
In each, first enter the repository and activate `.venv`. Use separate ports
and directories, and do not use `--require-distinct-hosts`:

Terminal L1:

```bash
python -m agent.main --executor navigation --agent-id robot_1 --kind robot \
  --listen 127.0.0.1:50051 --peer edge_pc=127.0.0.1:50052 \
  --artifact-dir .mars-hil/local-robot
```

Terminal L2:

```bash
python -m agent.main --executor navigation --agent-id edge_pc --kind edge \
  --listen 127.0.0.1:50052 --peer robot_1=127.0.0.1:50051 \
  --artifact-dir .mars-hil/local-edge
```

Terminal L3:

```bash
python -m scripts.hardware_loop \
  --agent robot_1=127.0.0.1:50051 \
  --agent edge_pc=127.0.0.1:50052 \
  --placement split --seed 19 --output .mars-hil/local-run-19.json
```

This runs real business subprocesses and socket transfers, but is not evidence
of successful PC-to-Orin deployment: the report should say
`scope: same_host_cpu_execution`. Stop any Agent already using a selected local
port before starting this variant.

## 9. Troubleshoot failures and stop

| Symptom | Check/action |
| --- | --- |
| Startup says `MOCK` | Restart with `--executor navigation`; do not use mock JSON config. |
| Connection refused or deadline exceeded | Agent process, actual LAN IP, listen address, matching port, and firewall. |
| Registration works but remote inputs fail | Both `--peer` maps; Orin must reach the PC LAN address. Preserve the detailed checksum/type error if present. |
| Distinct-host check fails | Ensure O1 really runs on Orin; do not remove the check to label a local test hardware validation. |
| Unsupported task or schema | Both hosts must use the same compatible hardware-loop revision; use this CLI DAG, not a web-generated mock workflow. |
| Worker fails or validation rejects a trajectory | Preserve the report and Agent logs; investigate the actual exception rather than accepting a simulated completion. |
| A legitimate worker exceeds its limit | Inspect load and logs first. The Agent's `--task-timeout SECONDS` controls its bounded task lifetime; the default is 30 seconds. |
| Whole workflow times out | The runner's `--workflow-timeout SECONDS` defaults to 120; individual completion waits are also capped at 60 seconds. Diagnose before increasing limits. |
| Output already exists | Choose a new `--output` filename; never overwrite an earlier test's evidence. |
| `attempt_history_full_restart_agent` | Finish/stop the current run, then restart that Agent; the in-memory limit is 1,024 accepted attempts. |
| Port already in use | Stop the earlier process you started on that port, or choose matching alternative ports in Agent and coordinator commands. |

Do not repeatedly submit a run while another coordinator is still executing.
To exercise failure handling deliberately, start a test run and stop one Agent
with `Ctrl+C`; the run must fail, not report successful missing work. Run duration
may be short, so a single manual attempt is not a substitute for automated
failure-path tests.

Normally, wait for P2 to finish, then press `Ctrl+C` in P1 and O1. Agent shutdown
cancels its active workload subprocesses. If interrupting a coordinator run,
also stop the Agents and check their terminals; do not assume closing a browser
or a shell has cancelled remote work. Keep `.mars-hil` reports and artifacts
for inspection. Do not delete an artifact directory while an Agent uses it.

## Design references and scope

Autra and Pony source snapshots were inspected only for the boundary pattern
of observations -> perception/world representation -> planning -> verification.
No proprietary source excerpts or runtime dependencies are included here.
The lightweight workload is an original implementation using standard algorithms.
Its public design references are:

- [ROS 2 OccupancyGrid](https://github.com/ros2/common_interfaces/blob/rolling/nav_msgs/msg/OccupancyGrid.msg): explicit occupancy and row-major spatial semantics.
- [Nav2 inflation layer](https://ros-navigation.github.io/mkdocs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/inflation/): robot-footprint clearance around obstacles.
- [OMPL state/motion validation](https://ompl.kavrakilab.org/stateValidation.html): validate motions between states, not only isolated waypoints.

ROS/Nav2/OMPL are references, not installation dependencies. This MVP does not
provide production authentication, persistent execution recovery, real sensor
capture, GPU inference, physical safety certification, or a measured energy
comparison. The existing web UI still authors synthetic workflows; the CLI above
is the supported entry point for this hardware workload.
