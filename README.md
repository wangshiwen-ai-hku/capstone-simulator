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

Restart the backend and confirm that `GET /api/health` reports
`"provider": "deepseek"` and `"llm_configured": true`. Enable **Use LLM** in
the Web interface when generating a scene. Provider credentials remain in the
backend; they are not returned by the API or sent to the browser. Invalid model
output falls back to the deterministic scene generator.

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
