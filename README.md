# MARS

MARS is a runtime-neutral central scheduler for multi-robot edge workflows.
It contains the DAG model and placement policies, one asynchronous runtime
contract, an in-process simulation adapter, a deterministic benchmark engine,
a FastAPI adapter, and a React interface.

The runnable architecture is:

```text
React UI ──► FastAPI adapter ──► CentralCoordinator ──► RuntimePort
                                                        └── InProcessRuntime
                                                            ├── Simulated Orin 1
                                                            ├── Simulated Orin 2
                                                            └── Simulated edge

                         └────► deterministic benchmark engine
```

The central runtime uses virtual time, so it does not wait for wall-clock model
execution. It still performs agent registration, heartbeats, capability checks,
resource reservation, assignment, typed Artifact transfer costing, completion,
resource release, and retry. The same seed produces a repeatable run.

Dependency direction is one way: `backend` imports `mars`; MARS does not import
the web application. `CentralCoordinator` depends only on the aggregate,
asynchronous `RuntimePort`. The in-process simulator is one implementation of
that port.

## Implemented capabilities

- Atomic DAG validation with cycle, reference, port, and message-type checks.
- `BLOCKED → READY → RUNNING → terminal` task lifecycle.
- Named `DataPort` and `DataEdge` contracts separate data flow from ordering.
- One output may fan out to multiple consumers without duplicating its Artifact.
- One task may publish multiple typed output Artifacts.
- Transfer cost includes only the output ports selected by downstream DataEdges.
- Critical-path, deadline, load, locality, bandwidth, and energy-aware placement.
- Three workload classes enforced as hard placement rules.
- Central scheduler with two simulated Orin Agents and one simulated edge Agent.
- Explicit registration, heartbeat, reservation/release, attempts, and retry.
- Replaceable synthetic workload profiles for local development without business code.
- Web views for DAGs, typed data flow, assignments, attempts, Artifacts, metrics, and events.

## The three task classes

| Class | Typical work | Placement contract |
|---|---|---|
| `local_safety` | obstacle avoidance, emergency stop, local control | Must run on its safety-capable source robot |
| `realtime_offloadable` | localization, environment understanding, detection, segmentation, local planning | May run on its source robot or edge |
| `edge_heavy` | 7B/10B local models and map fusion | Prefer edge; local fallback only when enabled |

Task types describe business capabilities. `task_class` is the stable placement
contract used by the scheduler.

## Quick start

Python 3.10–3.13 is recommended.

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The default scene has two Orin nodes, one edge
node, all three task classes, and a localization Artifact that fans out to
environment understanding and planning.

Use either execution path:

- **Run simulation / evaluation** compares scheduling algorithms in the deterministic engine.
- **Run 2 Orin + 1 Edge demo** runs the central Agent lifecycle and injects one recoverable failure.

### Tests

```bash
pip install -r backend/requirements-dev.txt
python -m pytest -q
cd frontend && npm run build
```

## Synthetic workloads

`configs/mars/workloads.synthetic.json` defines synthetic Orin and edge profiles
for:

- obstacle avoidance, emergency stop, and local control;
- localization, environment understanding, object detection, semantic segmentation, and local planning;
- 7B/10B local model inference and map fusion.

Each target profile includes p50/p95/p99 latency, CPU/GPU/memory demand,
input/output size ranges, energy, failure rate, accuracy, and maximum
concurrency. `SyntheticWorkloadCatalog.register_dict(...)` can add or replace a
fake task directly from a dictionary or JSON object.

The values are explicitly synthetic. When partner measurements become
available, request and record:

- exact Orin/edge hardware, power mode, and runtime versions;
- model artifact, precision, batch size, and input shape;
- input/output size distributions;
- warm-up method and p50/p95/p99 latency;
- throughput at concurrency 1/2/4;
- peak host/device memory;
- average/peak power or joules per task;
- failure rate and output quality for each hardware target.

`configs/mars/profiles.synthetic.json` contains compact benchmark-engine
profiles for older task labels.

## Project layout

```text
mars/
  models.py                    tasks, ports, data edges, artifacts, nodes, assignments
  dag.py                       validation, readiness, results, failure propagation
  scheduler.py                 placement constraints, costing, locality, critical path
  coordinator.py               central runtime orchestration, attempts, retry, report
  runtime/base.py              sole asynchronous control-plane runtime contract
  runtime/inprocess.py         process-local simulated runtime adapter
  engine.py                    deterministic algorithm benchmark engine
  synthetic_workloads.py       replaceable fake workload registry and sampler
  profiling.py                 compact execution-profile catalog
backend/app/
  main.py                      FastAPI endpoints
  runtime.py                   background local-runtime service and run store
  mars_adapter.py              web schema to MARS domain conversion
  scene_generator.py           deterministic typed-DAG generation
frontend/                      React benchmark and Agent runtime UI
configs/mars/                  synthetic workload and profile configuration
tests/                         core, runtime contract, adapter, and API tests
```

## API

Benchmark path:

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
state. `InProcessRuntime` implements the contract with virtual time. Future
gRPC, DDS, or partner adapters implement the same contract; the coordinator
does not depend on their communication mechanism.
