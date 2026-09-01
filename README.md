# MARS

MARS is a runtime-neutral central scheduler for multi-robot edge workflows.
It contains the DAG and scheduling contracts, one asynchronous runtime
contract, one coordinator event loop, an in-process simulation adapter, a
FastAPI adapter, and a React interface.

The runnable architecture is:

```text
React UI --> FastAPI adapter
               +-- Web simulation request --+
               +-- runtime request ----------+--> CentralCoordinator
                                                    +-- Snapshot + Policy + SolveLimits
                                                    |     +-- SchedulingProblem
                                                    |           +-- Optimizer
                                                    |                 +-- validated SchedulingPlan
                                                    +-- RuntimePort
                                                          +-- InProcessRuntime
                                                                +-- Simulated Robot Agent(s)
                                                                +-- Simulated Edge Agent(s)

CoordinatorReport --> immutable RunArtifact (inputs + raw run evidence)
                         +--> evals post-run evaluation
                         +--> Web response projector / runtime result
mars.engine -------> compatibility wrapper over the same artifact path
```

The central runtime uses virtual time rather than wall-clock model execution.
It performs agent registration, heartbeats, capability checks, resource
reservation, assignment, typed Artifact transfer costing, completion, resource
release, and retry. The same seed produces a repeatable run.

Dependency direction is one way: `backend` imports `mars`; MARS does not import
the web application. `CentralCoordinator` depends only on the aggregate,
asynchronous `RuntimePort`. The in-process simulator is one implementation of
that port.

## Implemented capabilities

- Atomic DAG validation with cycle, reference, port, and message-type checks.
- `BLOCKED -> READY -> RUNNING -> terminal` task lifecycle.
- Named `DataPort` and `DataEdge` contracts separate data flow from ordering.
- One output may fan out to multiple consumers without duplicating its Artifact.
- One task may publish multiple typed output Artifacts.
- Transfer cost includes only the output ports selected by downstream DataEdges.
- Optional `TaskClass` reporting cohorts are separate from declarative placement constraints.
- Directed `LinkSpec` and `LinkSnapshot` topology with multi-hop transfer estimates.
- Every planning iteration captures runtime facts in an immutable
  `SchedulingSnapshot`.
- `SchedulingProblem = SchedulingSnapshot + SchedulingPolicy + SolveLimits`;
  this is the solver-independent input contract shared by every formulation.
- A policy declares objectives and constraints. A formulation compiles those
  declarations and captured facts into a versioned decision domain; an
  optimizer is the replaceable search strategy over a compatible formulation.
- `MetricDefinition` is the immutable built-in catalog behind stable
  `ObjectiveMetric` wire IDs. It versions each metric's unit, scope, canonical
  plan evaluator, and optional candidate proxy without putting executable
  behavior into a policy or Proto message.
- `candidate_proxy_key` is explicitly a local heuristic ranking aid; registry
  fidelity marks it as exact, proxy, or unsupported. Committed Plans are always
  scored again with the canonical plan evaluator.
- Plans are validated against candidates, node capacity, concurrency, and link
  reservations before commit; shared evaluation recomputes policy objectives
  and constraints rather than trusting solver-reported values.
- Runtime dispatch carries the exact validated Assignment and its matching node
  and link reservation fragment.
- Invalid plug-in plans are rejected and may be re-solved by the configured
  fallback optimizer. The fallback first preserves the requested formulation;
  if it must use its own default domain, the Plan records the relaxation and
  both effective formulation identities remain visible in the solve trace.
- Critical-path, deadline, load, locality, bandwidth, and energy-aware built-in policies.
- Central scheduler with scene-defined simulated Orin and edge Agents.
- Explicit registration, heartbeat, reservation/release, attempts, and
  contract-safe retry.
- Replaceable synthetic workload profiles for local development and integration testing.
- Web views for DAGs, typed data flow, assignments, attempts, Artifacts, metrics, and events.

## Task placement and reporting cohorts

| Reporting cohort | Typical work | Legacy default placement |
|---|---|---|
| `local_safety` | obstacle avoidance, emergency stop, local control | Must run on its safety-capable source robot |
| `realtime_offloadable` | localization, environment understanding, detection, segmentation, local planning | May run on its source robot or edge |
| `edge_heavy` | 7B/10B local models and map fusion | Prefer edge; local fallback only when enabled |

`task_type` identifies the concrete work. `task_class` is optional compatibility
metadata used for aggregate reporting; it is not a placement rule. New workloads
use `PlacementConstraints` for pinning, allowed node kinds, required
capabilities, source/peer permissions, safety requirements, fallback,
statefulness, idempotence, splitting, replication, and ordered node-kind
preferences. Scenes without explicit constraints are mapped from the legacy
reporting cohorts for backward compatibility.

## Scheduling pipeline

```text
READY task batch
  -> hard placement filtering
  -> compute and directed-link candidate estimates
  -> immutable SchedulingSnapshot
  + SchedulingPolicy
  + SolveLimits
  -> SchedulingProblem
  -> configurable Formulation
  -> CompiledFormulation
  -> Optimizer
  -> SchedulingPlan
  -> shared objective/constraint evaluation and plan validation
  -> caller-owned OptimizerSolveState trace
  -> optional fallback solve using the same Problem and Policy
  -> node and link reservations
  -> runtime commit
```

The snapshot contains only observed or derived planning facts: epoch state,
node and link state, exact consumer-port artifact bindings, feasible candidates,
current reservations, availability, and critical-path estimates. The policy
contains the optimization intent:
ordered weighted objectives plus policy-level constraints. Solve limits bound
solver work without changing either facts or intent. These objects are kept
separate so the same captured state and policy can be compiled and searched by
different formulation/optimizer pairs.

`snapshot_id` fingerprints all captured facts. `problem_id` additionally
fingerprints the complete versioned Policy, referenced metric semantics, and
SolveLimits. Formulation is deliberately not part of `problem_id`: two model
encodings can solve the same Problem. Instead, `solve_request_id` fingerprints
the Problem, formulation/materializer contract, and optimizer version/config,
so Plans, traces, continuations, and dispatches remain exactly correlated.

The first concrete implementation is `one_hot_placement` v1. It selects exactly
one feasible candidate for each ready task and decodes the selection with the
serial-transfer/earliest-resource materializer. Drop, defer, split,
replication, alternate task ordering, and free start-time decisions are outside
this formulation version. Therefore `OPTIMAL` means globally optimal within
the Plan's recorded formulation domain, not across every schedule representable
by the general `SchedulingPlan` contract.

The v1 compiled model is a black-box discrete decision domain: it is directly
usable by enumerative and heuristic searches through canonical plan scoring.
A future MILP, CP-SAT, or ADMM plug-in will additionally need either a typed
solver-family encoder or a solver-neutral expression IR supplied by the
formulation; it must not reimplement Policy metric semantics inside the
optimizer. The canonical evaluator and Plan validator remain the final truth.

The DAG manager, not an optimizer, owns dependency satisfaction and task
lifecycle. Each rolling-horizon Problem contains the currently ready batch plus
critical-tail look-ahead; it is not a one-shot formulation of every remaining
task in the workflow.

For weighted-sum policies, `SchedulingPlan.objective_key` has one value. For
lexicographic policies it is the ordered vector of priority-group scores,
including soft-constraint penalties. Compare Plans by this key;
`objective_value` is only the first component for compatibility and reporting.
Plans also report solver status, version, elapsed time, iteration count, and a
termination reason so future MILP, ADMM, or primal-dual implementations share
the same result envelope. `INFEASIBLE` and `ERROR` Plans are never committable;
time- or iteration-limited Plans must still contain a fully validated feasible
incumbent for every assignment they return.
Coordinator `total_solver_time_ms` and `max_solver_time_ms` measure the complete
planning orchestration wall time, including compilation, rejected attempts,
and fallback attempts; a Plan's own elapsed field describes its effective
optimizer attempt.
`solve_budget_ms` is one shared, cooperative deadline for that orchestration:
compiled optimizers receive the absolute deadline and must stop promptly, and
the scheduler rejects every result returned after it. A synchronous in-process
plug-in cannot be forcibly interrupted while its Python call is running;
strict execution preemption requires a future worker-process boundary.

The coordinator owns one `OptimizerSolveState` per workflow and carries it
across rolling-horizon epochs. Every solver is traced through the common
started/completed/failed/validated/rejected/fallback lifecycle; a solver may additionally
implement the stateful formulated interface to record incumbents or keep a
typed, versioned continuation for warm starts. Formulated continuations are
isolated by Problem/Snapshot schema, Policy and metric semantics,
formulation/materializer, optimizer version/config, and the deterministic/
random-seed contract; they cannot be resumed under an incompatible
mathematical or search model. Optimizer
instances remain stateless and safe to reuse. The complete recorded trace and
continuation history are included in the coordinator scheduling report; each
solver controls checkpoint cadence, so an enumerative solver can record
incumbent changes without logging every rejected combination.
If a workflow run raises before a report can be returned, the same state remains
available through `CentralCoordinator.optimizer_solve_state` for diagnosis.

`CentralCoordinator` owns the only scheduling event loop. Its optimizer sees
the complete ready batch. The coordinator commits validated assignments through
`RuntimePort`, consumes correlated completions, updates DAG state, and replans
after completion or retry. `FAIL_FAST` workflows use a single rolling commit so
that a failure cannot leave sibling work in flight.

Both Web simulation and runtime workflow submission use this path with
`InProcessRuntime`. `mars.engine` is a compatibility wrapper and report
projector for callers that still consume `SimulationReport`; it does not own a
second scheduler or event loop.

`RunArtifact` is the immutable, pre-evaluation record of a completed run: it
captures declared workflow/topology/profile inputs, run configuration, retained
scheduling Plans, and the raw `CoordinatorReport`. Post-run metric definitions,
observations, aggregation, and reusable benchmark packages live in top-level
`evals`. Evaluation consumes artifacts only after execution; benchmark packages
drive the production engine externally, then aggregate and report its artifacts.
Neither changes scheduling or runtime semantics. `SimulationReport` remains a
compatibility projection rather than the canonical run record.

The built-in optimizer IDs are `heuristic` and `binary_offload`. The existing
API values `dag_deadline`, `rule_based`, `local_first`, `edge_first`, and
`greedy_cost` continue to be accepted as policy aliases. Each resolves to the
`heuristic` optimizer and the policy with the same name. Additional solvers
implement the `Optimizer` protocol and are registered through
`OptimizerRegistry`; they consume the same `SchedulingProblem` and do not change
the coordinator, task model, or runtime interface. Solvers that need incumbent
tracing or cross-frame warm starts additionally implement the optional
`StatefulOptimizer` protocol; existing stateless plug-ins continue to use
`solve(problem)` unchanged.

## Quick start

The CI runtime is Python 3.12.

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

### Optional LLM scene generation

LLM integration generates a candidate scene and typed DAG; it does not replace
the scheduler or optimizer. To use DeepSeek locally, copy the ignored
environment file and add a private API key:

```bash
cp backend/.env.example backend/.env
```

```dotenv
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=<private-api-key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
LLM_TIMEOUT_SECONDS=300
LLM_MAX_RETRIES=1
LLM_STREAM_RESPONSES=1
```

The left control surface also includes MARS Agent and MARS Templates. Agent
models are routed through APIYI and currently allow `deepseek-v4-flash` and
`gemini-3.1-flash-lite`. Configure `APIYI_KEY`; optional settings are documented in
`backend/.env.example`. Without a key, the Agent remains usable in a local,
deterministic structured-draft mode. Saved templates contain a complete,
validated `BenchmarkScene` and can be imported directly into Studio. The Web
client creates a cryptographically random template-workspace capability, keeps
it in browser local storage, and sends it only to template endpoints. Template
list/read/delete operations are isolated to that capability instead of exposing
one global library. This is lightweight isolation for the demo, not user-account
authentication: anyone who obtains the capability can access that workspace,
and clearing site data loses the browser's reference to it. Export important
templates as JSON backups.

MARS Agent uses an incremental conversation: discovery, atomic-task planning,
review, then schema compilation after confirmation. Model calls are bounded by
`MARS_AGENT_MODEL_TIMEOUT_SECONDS` (35 seconds by default). Retrieval creates
its TLS context from the packaged `certifi` CA bundle, so a normal backend
startup does not require manually exporting `SSL_CERT_FILE`.

Generated accelerator resources use absolute sparse INT8 TOPS: Jetson Orin
Nano/NX/AGX capacities are 67/157/275 TOPS, while each workload declares one
fixed `accelerator_demand_tops` independent of board, difficulty, seed, and
utilization. Every `BenchmarkScene` must declare
`resource_contract_version: "mars.resources.absolute.v1"`; under that contract,
CPU values are physical cores and accelerator values are sparse INT8 TOPS.
Unversioned scenes are rejected because normalized GPU fractions cannot be
distinguished safely from absolute demands. Synthetic scenes run a
schedulability preflight before they are returned so random background load
cannot silently leave a task without an execution candidate.

Restart the backend and confirm that `GET /api/health` reports
`"provider": "deepseek"` and `"llm_configured": true`. Enable **Use LLM** in
the Web interface when generating a scene. Provider credentials remain in the
backend; they are not returned by the API or sent to the browser. Invalid model
output falls back to the deterministic scene generator.

### Fly template storage

`fly.toml` stores templates at `/data/mars-templates` on the `mars_data`
persistent volume. Create that volume once in the app's primary region before
the first deployment containing the mount, then deploy normally:

```bash
fly volumes create mars_data \
  --app capstone-simulator-backend \
  --region sin \
  --size 1
fly deploy --app capstone-simulator-backend
```

Fly volumes attach to one Machine and are not shared filesystems. This app is
currently configured as a single always-running Machine; create one volume per
Machine or move templates to shared object/database storage before scaling out.
Files previously written to the container-local `tmp/mars-templates` directory
cannot be assigned safely to a browser workspace and are intentionally not
served by the capability-scoped store.

### Optional API trace archive

Off by default. Set in `backend/.env`:

```dotenv
MARS_TRACE_ARCHIVE=1
MARS_TRACE_DIR=tmp/mars-traces
```

When enabled, the backend logs a **warning** on startup. Each generated scene
owns one timestamped root, and later simulation/runtime calls are attached to
that root through the scene's opaque `trace_id`. Calls are grouped by execution
path and record the effective solver in their directory name. The v3 layout is:

```text
tmp/mars-traces/
  YYYYMMDDTHHMMSS.ffffff_<scene-id>/
    scene/
      meta.json
      request.json
      response.json
    llm/
      meta.json               # timing, summary, and exception chain
      request.json            # prompts and safe request metadata
      response.json           # full raw content, when received
    calls/
      simulate/
        YYYYMMDDTHHMMSS.ffffff_<solver>_<call-id>/
          meta.json
          request.json
          response.json
      runtime/
        YYYYMMDDTHHMMSS.ffffff_<solver>_<call-id>/
          meta.json
          request.json
          accepted.json
          response.json
          status.json
```

If a scheduler call uses an older/imported scene without a known `trace_id`, a
new root marked `status: imported` is created instead of silently losing the
call. Files are written atomically. `GET /api/health` includes the archive
status, layout, and schema version without exposing the server filesystem path.
Credentials are redacted from archived prompts, responses, and metadata.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The default scene has two Orin nodes, one edge
node, explicit per-task placement constraints, and a localization Artifact that
fans out to environment understanding and planning.

Both interface actions use the same coordinator and RuntimePort execution path:

- **Run Scheduling Simulation** returns the `SimulationReport` representation of
  the evaluated `RunArtifact` produced from the completed coordinator run.
- **Submit to Agent Runtime** stores the coordinator report asynchronously and
  retains its `RunArtifact` while exposing the compatible raw result and run
  events. Failure injection is available through the runtime API and is
  disabled by default.

### Tests

```bash
pip install -r backend/requirements-dev.txt
python -m ruff check backend evals mars scripts tests
python -m compileall -q backend evals mars scripts tests
python -m pytest -q
cd frontend && npm test && npm run build
```

With the backend virtual environment active, install Chromium once and run the
full-stack smoke test against the production frontend bundle:

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

The smoke test starts FastAPI and the Vite preview server, loads the UI in a
real browser, generates a deterministic scene, submits a workflow, and verifies
that the terminal runtime result renders. It does not require an LLM API key.

The binary-offload experiment is implemented as the importable
`evals.benchmarks.binary_offload` package. Tests import that package directly;
`scripts/run_binary_offload_benchmark.py` is only the command-line entry point
that runs the fixed matrix and writes its seven compatibility artifacts to
`doc/`.

## Synthetic workloads

`configs/mars/workloads.synthetic.json` defines synthetic Orin and edge profiles
for:

- obstacle avoidance, emergency stop, and local control;
- localization, environment understanding, object detection, semantic segmentation, and local planning;
- 7B/10B local model inference, data compression, result verification, and map
  fusion.

Each target profile includes p50/p95/p99 latency, CPU/GPU/memory demand,
input/output size ranges, energy, failure rate, accuracy, and maximum
concurrency. `SyntheticWorkloadCatalog.register_dict(...)` can add or replace a
synthetic workload definition from a dictionary or JSON object.

The values are explicitly synthetic. Deployment profiles require the following
measured metadata:

- exact Orin/edge hardware, power mode, and runtime versions;
- model artifact, precision, batch size, and input shape;
- input/output size distributions;
- warm-up method and p50/p95/p99 latency;
- throughput at concurrency 1/2/4;
- peak host/device memory;
- average/peak power or joules per task;
- failure rate and output quality for each hardware target.

`configs/mars/profiles.synthetic.json` contains compatibility profiles for
older task labels.

## Project layout

```text
mars/
  domain/
    task.py                    task declarations, instances, placement, task state
    workflow.py                DAG edges, workflow declarations, lifecycle progress
    artifact.py                artifact references and input-port bindings
    topology.py                node/link declarations and dynamic snapshots
    transfer.py                transfer estimates and reservations
    execution.py               assignments, resource demand, task completion
  models.py                    compatibility re-exports for mars.domain
  dag.py                       validation, readiness, results, failure propagation
  network.py                   directed topology and transfer estimation
  scheduler.py                 candidate generation and planning orchestration
  optimizers/base.py           snapshot, problem, plan, registry, invariant validation
  optimizers/policy.py         objectives, constraints, solve limits, policy presets
  optimizers/evaluation.py     shared objective and constraint evaluation
  optimizers/formulation.py    formulation, solve-request, and registry contracts
  optimizers/formulations/     concrete compiled decision domains
  optimizers/materialization.py shared candidate timing and reservation construction
  optimizers/state.py          cross-frame solve trace and continuation state
  optimizers/heuristics.py     built-in heuristic optimizer
  optimizers/binary_offload.py exhaustive ready-batch placement optimizer
  coordinator.py               central runtime orchestration, attempts, retry, report
  runtime/base.py              sole asynchronous control-plane runtime contract
  runtime/inprocess.py         process-local simulated runtime adapter
  run_artifact.py              immutable inputs and raw evidence for one run
  engine.py                    compatibility wrapper and SimulationReport projector
  synthetic_workloads.py       replaceable synthetic workload registry and sampler
  profiling.py                 execution-profile catalog
evals/
  contracts.py                 versioned post-run metric and aggregation contracts
  workflow.py                  canonical RunArtifact workflow evaluation
  benchmarks/binary_offload/   benchmark definition, runner, audit, and reporting
backend/app/
  main.py                      FastAPI endpoints
  runtime.py                   background local-runtime service and run store
  mars_adapter.py              web schema to MARS domain conversion
  scene_generator.py           deterministic typed-DAG generation
frontend/                      React benchmark and Agent runtime UI
interfaces/proto/mars/v1/      versioned cross-module data contracts
configs/mars/                  synthetic workload and profile configuration
tests/                         core, runtime contract, adapter, and API tests
scripts/                       thin command-line and demonstration entry points
```

The Proto files define versioned data messages for workflows, topology,
profiling, scheduling problems and plans, and runtime commands/events. They are
the language-neutral interface source; Python domain classes remain the
in-process implementation model.

Current scope does not include generated Proto bindings, RPC service
definitions, a gRPC or DDS network adapter, deployment middleware integration,
or production optimizer implementations such as MILP, ADMM, or primal-dual
solvers. Those components can be added against the defined Problem, Plan, and
RuntimePort boundaries.

## API

Web and inspection API:

- `GET /api/health`
- `GET /api/architecture`
- `GET /api/workload-catalog`
- `POST /api/validate-workflow`
- `POST /api/generate-scene`
- `POST /api/simulate`

Central runtime:

- `POST /api/runtime/bootstrap`
- `GET /api/runtime`
- `GET /api/agents`
- `POST /api/runtime/workflows`
- `GET /api/runtime/workflows/{run_id}`
- `GET /api/runtime/workflows/{run_id}/events?after_sequence=N`

## Runtime boundary

`RuntimePort` is the only coordinator-facing contract. It supplies global node
inventory and heartbeats, accepts attempt-scoped dispatch commands, returns
dispatch-correlated completions, supports cancellation, and reports runtime
state. Each dispatch contains the unmodified validated Assignment plus its
matching resource and transfer reservations and exact consumer-port input
bindings. Command validation rejects an inconsistent plan fragment before an
adapter receives it, and preserves the Problem, Snapshot, and Policy
correlation identifiers for replay and audit.
`InProcessRuntime` implements the contract with virtual time. Networked or
deployment-specific adapters implement the same contract; the coordinator does
not depend on their communication mechanism.

---

# MARS（中文说明）

MARS 是一个面向多机器人边缘工作流、与运行时无关的中央调度器。
它包含有向无环图（DAG）与调度契约、一个异步运行时契约、一个协调器事件循环、一个
进程内仿真适配器、一个 FastAPI 适配器以及一个 React 界面。

可运行的架构如下：

```text
React 界面 --> FastAPI 适配器
               +-- Web 仿真请求 --------+
               +-- 运行时请求 ----------+--> CentralCoordinator
                                             +-- Snapshot + Policy + SolveLimits
                                             |     +-- SchedulingProblem
                                             |           +-- Optimizer
                                             |                 +-- 已校验的 SchedulingPlan
                                             +-- RuntimePort
                                                   +-- InProcessRuntime
                                                         +-- 仿真机器人 Agent
                                                         +-- 仿真边缘 Agent

CoordinatorReport --> 不可变 RunArtifact（输入 + 原始运行证据）
                             +--> evals 运行后评估
                             +--> Web 响应投影器 / 运行时结果
mars.engine -------> 同一制品路径上的兼容性封装
```

中央运行时采用虚拟时间，而不是执行模型时的真实墙钟时间。它会执行 Agent
注册、心跳、能力检查、资源预留、任务分派、带类型的制品（Artifact）传输成本计算、
任务完成、资源释放和重试。使用相同的随机种子可得到可复现的运行结果。

依赖方向是单向的：`backend` 导入 `mars`，MARS 不导入 Web 应用。
`CentralCoordinator` 只依赖聚合式异步接口 `RuntimePort`。进程内仿真器是该
接口的一种实现。

## 已实现的能力

- 对 DAG 执行原子化校验，包括环、引用、端口和消息类型检查。
- `BLOCKED -> READY -> RUNNING -> terminal` 任务生命周期。
- 具名的 `DataPort` 和 `DataEdge` 契约将数据流与执行顺序分离。
- 一个输出可以扇出至多个消费者，而无需复制其制品。
- 一个任务可以发布多个带类型的输出制品。
- 传输成本仅计入下游 `DataEdge` 所选择的输出端口。
- 可选的 `TaskClass` 报告分组与声明式放置约束相互独立。
- 使用有向 `LinkSpec` 和 `LinkSnapshot` 拓扑进行多跳传输估算。
- 每次规划迭代都会将运行时事实记录在不可变的 `SchedulingSnapshot` 中。
- `SchedulingProblem = SchedulingSnapshot + SchedulingPolicy + SolveLimits`；
  这是所有建模方式共用、与求解器无关的输入契约。
- 策略（Policy）声明目标和约束。模型表述（Formulation）将这些声明及捕获到的事实
  编译为带版本的决策域；优化器（Optimizer）则是在兼容模型表述上可替换的搜索策略。
- `MetricDefinition` 是稳定 `ObjectiveMetric` 序列化 ID 背后的不可变内置目录。
  它为每个指标的单位、作用域、规范调度方案评估器和可选候选代理赋予版本，且不会
  将可执行行为放入策略或 Proto 消息中。
- `candidate_proxy_key` 明确只用于本地启发式排序；注册表中的 fidelity 会将其
  标记为 `exact`、`proxy` 或 `unsupported`。已提交的调度方案始终会由规范调度方案评估器
  再次评分。
- 调度方案在提交前会根据候选项、节点容量、并发度和链路预留进行校验；共享评估会
  重新计算策略目标和约束，而不是信任求解器报告的值。
- 运行时分派携带经过校验的原始 `Assignment`，以及与其匹配的节点和链路预留片段。
- 无效的插件调度方案会被拒绝，并可由配置的回退优化器重新求解。回退过程首先保留
  请求的模型表述；若必须使用自身的默认决策域，调度方案会记录这一放宽，且两个实际
  使用的模型表述身份都会保留在求解轨迹中。
- 内置策略支持关键路径、截止时间、负载、局部性、带宽和能耗感知。
- 中央调度器支持场景定义的仿真 Orin 和边缘 Agent。
- 显式的注册、心跳、资源预留/释放、尝试以及契约安全重试。
- 可替换的合成工作负载配置，用于本地开发和集成测试。
- 提供 DAG、带类型数据流、`Assignment`、尝试、制品、指标和事件的 Web 视图。

## 任务放置与报告分组

| 报告分组 | 典型工作 | 旧版默认放置规则 |
|---|---|---|
| `local_safety` | 避障、紧急停止、本地控制 | 必须在具备安全能力的源机器人上运行 |
| `realtime_offloadable` | 定位、环境理解、检测、分割、本地规划 | 可在源机器人或边缘节点上运行 |
| `edge_heavy` | 7B/10B 本地模型和地图融合 | 优先放置于边缘；仅在启用时允许本地回退 |

`task_type` 标识具体工作。`task_class` 是用于聚合报告的可选兼容性元数据，
并非放置规则。新的工作负载通过 `PlacementConstraints` 声明固定节点、允许的
节点类型、所需能力、源节点/对等节点权限、安全要求、回退、状态性、幂等性、
拆分、复制以及有序节点类型偏好。为保持向后兼容，未显式指定约束的场景会由旧版
报告分组映射得到约束。

## 调度流水线

```text
READY 任务批次
  -> 硬性放置筛选
  -> 计算与有向链路候选估算
  -> 不可变 SchedulingSnapshot
  + SchedulingPolicy
  + SolveLimits
  -> SchedulingProblem
  -> 可配置 Formulation
  -> CompiledFormulation
  -> Optimizer
  -> SchedulingPlan
  -> 共享目标/约束评估与调度方案校验
  -> 调用方持有的 OptimizerSolveState 轨迹
  -> 可选：使用同一 Problem 和 Policy 进行回退求解
  -> 节点与链路资源预留
  -> 提交到运行时
```

快照只包含观察到或推导出的规划事实：调度轮次状态、节点与链路状态、精确到消费者
端口的制品绑定、可行候选项、当前预留、可用性以及关键路径估算。策略包含优化意图：
按顺序排列的加权目标与策略级约束。`SolveLimits` 限制求解器的工作量，而不改变事实
或意图。三者相互分离，因此同一份捕获状态和策略可以由不同的模型表述/优化器组合
进行编译和搜索。

`snapshot_id` 为所有捕获到的事实生成指纹。`problem_id` 还会为完整的带版本策略、
所引用的指标语义和 `SolveLimits` 生成指纹。模型表述被有意排除在 `problem_id`
之外：两种模型编码可以求解同一个调度问题。与之相对，`solve_request_id` 会为调度
问题、模型表述/物化器契约以及优化器版本/配置生成指纹，从而让调度方案、轨迹、续接
状态和分派保持精确关联。

首个具体模型表述实现是 `one_hot_placement` v1。它为每个处于 `READY` 状态的任务
恰好选择一个可行候选项，并通过串行传输/最早资源物化器对选择进行解码。丢弃、延后、
拆分、复制、采用其他任务排序方式以及自由开始时间决策均不在此版模型表述的范围内。
因此，
`OPTIMAL` 表示在调度方案所记录的模型表述决策域内达到全局最优，而非在通用
`SchedulingPlan` 契约可表达的所有调度中达到全局最优。

v1 编译模型是一个黑盒离散决策域，可以通过规范调度方案评分直接用于枚举式和启发式
搜索。未来的 MILP、CP-SAT 或 ADMM 插件还需要模型表述提供带类型的求解器族编码器
或求解器无关的表达式 IR；它不得在优化器内部重新实现策略的指标语义。规范评估器和
调度方案校验器仍是最终判定依据。

DAG 管理器而非优化器负责依赖满足与任务生命周期。每个滚动时域调度问题包含当前
就绪任务批次以及关键尾部前瞻；它并不是对工作流中所有剩余任务进行一次性建模。

对于加权和策略，`SchedulingPlan.objective_key` 只有一个值。对于字典序策略，它是
按优先级分组排列的得分向量，其中包括软约束惩罚。应使用该键比较调度方案；
`objective_value` 仅作为第一个分量，用于兼容和报告。调度方案还会报告求解器状态、
版本、耗时、迭代次数和终止原因，以便未来的 MILP、ADMM 或原始-对偶实现共用同一
结果封装。`INFEASIBLE` 和 `ERROR` 调度方案永远不能提交；受到时间或迭代次数限制的
调度方案仍必须为其返回的每个 `Assignment` 提供经过完整校验的当前最优可行解。
协调器的 `total_solver_time_ms` 和 `max_solver_time_ms` 衡量完整规划编排所用的墙钟
时间，包括编译、被拒绝的尝试和回退尝试；调度方案自身的耗时字段描述其实际优化器
尝试。
`solve_budget_ms` 是整个编排过程共用的协作式截止时间：编译后的优化器会收到绝对
截止时间并且必须及时停止，调度器会拒绝所有在截止时间之后返回的结果。同步进程内
插件在其 Python 调用运行期间无法被强制中断；严格的执行抢占需要未来引入工作进程
边界。

协调器为每个工作流维护一个 `OptimizerSolveState`，并使其贯穿滚动时域内的各个调度
轮次。每个求解器都会通过统一的
`started/completed/failed/validated/rejected/fallback` 生命周期留下轨迹；求解器还可
实现有状态、基于模型表述的接口，以记录当前
最优解或保留带类型、带版本的续接状态，用于热启动。续接状态会根据调度问题/快照的
模式、策略和指标语义、模型表述/物化器、优化器版本/配置以及确定性/随机种子契约彼此
隔离；它们不能在不兼容的数学模型或搜索模型下恢复。优化器实例保持无状态，因此可以
安全复用。完整记录的轨迹和续接状态历史会包含在协调器调度报告中；每个求解器自行
控制检查点频率，因此枚举求解器可以记录当前最优解的变化，而无需记录每个被拒绝的
组合。
如果工作流运行在报告返回前抛出异常，同一状态仍可通过
`CentralCoordinator.optimizer_solve_state` 获取以进行诊断。

`CentralCoordinator` 拥有唯一的调度事件循环。其优化器可以看到完整的就绪任务
批次。协调器通过 `RuntimePort` 提交经过校验的任务分派（`Assignment`），接收已关联的
完成结果，更新 DAG 状态，并在完成或重试后重新规划。`FAIL_FAST`
工作流采用单次滚动提交，避免任务失败后仍有同批任务在运行。

Web 仿真和运行时工作流提交都通过 `InProcessRuntime` 使用这一路径。
`mars.engine` 是面向仍使用 `SimulationReport` 的调用方所提供的兼容性封装和报告
投影器；它不拥有第二个调度器或事件循环。

`RunArtifact` 是一次已完成运行在评估前的不可变记录：它捕获声明的工作流/拓扑/
性能配置输入、运行配置、保留的调度方案以及原始 `CoordinatorReport`。运行后指标
定义、观测、聚合和可复用基准包位于顶层 `evals` 中。评估仅在执行完成后使用制品；
基准包从外部驱动生产引擎，随后聚合并报告其制品。两者都不会改变调度或运行时语义。
`SimulationReport` 仍是兼容性投影，而非规范运行记录。

内置优化器 ID 为 `heuristic` 和 `binary_offload`。现有 API 值
`dag_deadline`、`rule_based`、`local_first`、`edge_first` 和 `greedy_cost`
仍可作为策略别名使用。每个别名都会解析为 `heuristic` 优化器和同名策略。其他求解器
通过实现 `Optimizer` 协议并注册到 `OptimizerRegistry` 来接入；它们使用同一个
`SchedulingProblem`，且不会改变协调器、任务模型或运行时接口。需要当前最优解轨迹
或跨帧热启动的求解器还可实现可选的
`StatefulOptimizer` 协议；现有无状态插件仍可原样使用 `solve(problem)`。

## 快速开始

CI 运行环境使用 Python 3.12。

### 后端

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

### 可选的 LLM 场景生成

LLM 集成会生成候选场景和带类型的 DAG；它不会取代调度器或优化器。
如需在本地使用 DeepSeek，请复制被忽略的环境文件并加入私有 API 密钥：

```bash
cp backend/.env.example backend/.env
```

```dotenv
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=<private-api-key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
LLM_TIMEOUT_SECONDS=300
LLM_MAX_RETRIES=1
LLM_STREAM_RESPONSES=1
```

重启后端，并确认 `GET /api/health` 报告的内容包含
`"provider": "deepseek"` 和 `"llm_configured": true`。在 Web 界面生成场景时启用
**Use LLM**。模型服务商凭证只保留在后端，不会由 API 返回或发送至浏览器。无效的
模型输出会回退到确定性场景生成器。

### 可选的 API 轨迹归档

默认关闭。请在 `backend/.env` 中设置：

```dotenv
MARS_TRACE_ARCHIVE=1
MARS_TRACE_DIR=tmp/mars-traces
```

启用后，后端会在启动时记录一条 **warning**（警告）。每个生成的场景都拥有一个带时间戳的
根目录，后续仿真/运行时调用通过该场景的不透明 `trace_id` 挂接到此根目录。调用按
执行路径分组，并在目录名中记录实际使用的求解器。v3 布局如下：

```text
tmp/mars-traces/
  YYYYMMDDTHHMMSS.ffffff_<scene-id>/
    scene/
      meta.json
      request.json
      response.json
    llm/
      meta.json               # 耗时、摘要和异常链
      request.json            # 提示词和安全请求元数据
      response.json           # 收到时的完整原始内容
    calls/
      simulate/
        YYYYMMDDTHHMMSS.ffffff_<solver>_<call-id>/
          meta.json
          request.json
          response.json
      runtime/
        YYYYMMDDTHHMMSS.ffffff_<solver>_<call-id>/
          meta.json
          request.json
          accepted.json
          response.json
          status.json
```

如果调度器调用使用的是没有已知 `trace_id` 的旧场景或导入场景，系统会创建
一个标记为 `status: imported` 的新根目录，而不会静默丢失该调用。文件以原子方式
写入。`GET /api/health` 会包含归档状态、布局和模式版本，但不会暴露服务器文件系统
路径。归档的提示词、响应和元数据中的凭证会被脱敏。

### 前端

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。默认场景包含两个 Orin 节点、一个边缘节点、显式的
逐任务放置约束，以及一个扇出至环境理解和规划任务的定位制品。

界面中的两个操作都使用同一条协调器和 RuntimePort 执行路径：

- **Run Scheduling Simulation** 返回由已完成协调器运行生成并经过评估的
  `RunArtifact` 所对应的 `SimulationReport` 表示。
- **Submit to Agent Runtime** 异步存储协调器报告并保留其 `RunArtifact`，
  同时公开兼容的原始结果与运行事件。故障注入可通过运行时 API 使用，默认关闭。

### 测试

```bash
pip install -r backend/requirements-dev.txt
python -m ruff check backend evals mars scripts tests
python -m compileall -q backend evals mars scripts tests
python -m pytest -q
cd frontend && npm test && npm run build
```

激活后端虚拟环境后，安装一次 Chromium，并针对生产前端构建产物运行全栈冒烟测试：

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

该冒烟测试会启动 FastAPI 和 Vite 预览服务器，在真实浏览器中加载 UI，生成
确定性场景、提交工作流，并验证终态运行结果是否渲染。它不需要 LLM API 密钥。

二元卸载实验以可导入的 `evals.benchmarks.binary_offload` 包实现。测试会直接导入
该包；`scripts/run_binary_offload_benchmark.py` 只是命令行入口，用于运行固定
矩阵并将七个兼容性制品写入 `doc/`。

## 合成工作负载

`configs/mars/workloads.synthetic.json` 为以下任务定义了合成 Orin 和边缘性能配置：

- 避障、紧急停止和本地控制；
- 定位、环境理解、目标检测、语义分割和本地规划；
- 7B/10B 本地模型推理、数据压缩、结果验证和地图融合。

每个目标性能配置包含 p50/p95/p99 延迟、CPU/GPU/内存需求、输入/输出大小范围、
能耗、故障率、准确率和最大并发度。`SyntheticWorkloadCatalog.register_dict(...)`
可以从字典或 JSON 对象添加或替换合成工作负载定义。

这些值明确属于合成数据。部署性能配置需要以下实测元数据：

- 准确的 Orin/边缘硬件、功耗模式和运行时版本；
- 模型制品、精度、批大小和输入形状；
- 输入/输出大小分布；
- 预热方法和 p50/p95/p99 延迟；
- 并发度为 1/2/4 时的吞吐量；
- 主机/设备峰值内存；
- 平均/峰值功耗或单任务焦耳数；
- 每种硬件目标上的故障率和输出质量。

`configs/mars/profiles.synthetic.json` 包含面向旧任务标签的兼容性性能配置。

## 项目布局

```text
mars/
  domain/
    task.py                    任务声明、实例、放置和状态
    workflow.py                DAG 边、工作流声明和生命周期进度
    artifact.py                制品引用与输入端口绑定
    topology.py                节点/链路声明与动态快照
    transfer.py                传输估算与资源预留
    execution.py               任务分配、资源需求和任务完成
  models.py                    mars.domain 的兼容性重导出
  dag.py                       校验、就绪、结果与失败传播
  network.py                   有向拓扑与传输估算
  scheduler.py                 候选生成与规划编排
  optimizers/base.py           快照、问题、计划、注册表与不变量校验
  optimizers/policy.py         目标、约束、求解限制与策略预设
  optimizers/evaluation.py     共享目标与约束评估
  optimizers/formulation.py    模型表述、求解请求与注册表契约
  optimizers/formulations/     具体的已编译决策域
  optimizers/materialization.py 共享候选计时与资源预留构造
  optimizers/state.py          跨帧求解轨迹与续接状态
  optimizers/heuristics.py     内置启发式优化器
  optimizers/binary_offload.py 穷举式就绪批次放置优化器
  coordinator.py               中央运行时编排、尝试、重试与报告
  runtime/base.py              唯一的异步控制平面运行时契约
  runtime/inprocess.py         进程内仿真运行时适配器
  run_artifact.py              一次运行的不可变输入与原始证据
  engine.py                    兼容性封装与 SimulationReport 投影器
  synthetic_workloads.py       可替换的合成工作负载注册表与采样器
  profiling.py                 执行性能配置目录
evals/
  contracts.py                 带版本的运行后指标与聚合契约
  workflow.py                  规范 RunArtifact 工作流评估
  benchmarks/binary_offload/   基准定义、运行器、审计与报告
backend/app/
  main.py                      FastAPI 端点
  runtime.py                   后台本地运行时服务与运行存储
  mars_adapter.py              Web 模式到 MARS 领域模型的转换
  scene_generator.py           确定性的带类型 DAG 生成
frontend/                      React 基准与 Agent 运行时 UI
interfaces/proto/mars/v1/      带版本的跨模块数据契约
configs/mars/                  合成工作负载与性能配置
tests/                         核心、运行时契约、适配器与 API 测试
scripts/                       精简命令行与演示入口
```

Proto 文件为工作流、拓扑、性能剖析、调度问题与调度方案，以及运行时命令/事件定义
带版本的数据消息。它们是与语言无关的接口源；Python 领域类仍作为进程内实现模型。

当前范围不包括生成的 Proto 绑定、RPC 服务定义、gRPC 或 DDS 网络适配器、部署中间件
集成，或 MILP、ADMM、原始-对偶求解器等生产级优化器实现。上述组件可以基于已定义的
调度问题、调度方案和 `RuntimePort` 边界添加。

## API

Web 与检查 API：

- `GET /api/health`
- `GET /api/architecture`
- `GET /api/workload-catalog`
- `POST /api/validate-workflow`
- `POST /api/generate-scene`
- `POST /api/simulate`

中央运行时：

- `POST /api/runtime/bootstrap`
- `GET /api/runtime`
- `GET /api/agents`
- `POST /api/runtime/workflows`
- `GET /api/runtime/workflows/{run_id}`
- `GET /api/runtime/workflows/{run_id}/events?after_sequence=N`

## 运行时边界

`RuntimePort` 是唯一面向协调器的契约。它提供全局节点清单和心跳，接收以尝试为作用域
的分派命令，返回与分派关联的完成结果，支持取消，并报告运行时状态。每次分派都包含
未经修改且已校验的 `Assignment`、与之匹配的资源和传输预留，以及精确到消费者端口的
输入绑定。命令校验会在适配器收到不一致的调度方案片段之前将其拒绝，并保留调度问题、
快照和策略的关联 ID，用于重放和审计。
`InProcessRuntime` 以虚拟时间实现该契约。网络化或部署专用适配器实现同一契约；协调器
不依赖其通信机制。
