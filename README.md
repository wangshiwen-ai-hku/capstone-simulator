# Unified EdgeSched Simulator

The project combines a web benchmark with a transport-neutral edge-scheduling
core for multi-robot workflows.

The runnable shape is:

```text
Web UI / REST API
        │
        ▼
Workflow + DAG Manager ──► DAG-aware Scheduler
        │                         │
        └── in-memory simulation ─┘
```

## Implemented in v0.2

- Atomic DAG validation: unique IDs, known parents, no self-dependencies and no cycles.
- Authoritative `BLOCKED → READY → RUNNING → terminal` lifecycle.
- Multi-parent release, idempotent completion, descendant skipping and fail-fast policies.
- Critical-path/deadline-aware scheduling with intermediate-artifact locality.
- Three workload classes enforced as hard placement rules.
- Configurable synthetic profiling catalogue for runs without workload artifacts.
- Updated React UI with workflow status, task-class metrics and a DAG stage view.
- Legacy sequential simulation and isolated-task scheduling modules retained only as deprecated import paths.

## The three task classes

| Class | Typical work | Placement contract |
|---|---|---|
| `local_safety` | obstacle avoidance, emergency control | Must run on its source Orin |
| `realtime_offloadable` | YOLO, segmentation, path planning, verification | May run on source Orin or edge |
| `edge_heavy` | VLA/LLM, map fusion, compression and heavy planning | Prefer edge; explicit local fallback |

Task categories remain detailed benchmark labels. `task_class` is the stable
placement contract.

## Quick start

Python 3.10–3.13 is recommended.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Select `dag_deadline` for the new scheduler.

### Tests

The core test suite uses the standard library and can run before installing the
web dependencies:

```bash
PYTHONPATH=backend python -m unittest discover -s backend -p 'test_*.py' -v
```

## Profiles when real tasks are unavailable

`configs/profiles.synthetic.json` contains clearly labelled placeholder rows.
The simulator loads them automatically and falls back to a compute-demand model
when a row is missing. Replace rows with measurements while keeping the schema.

For each model/hardware pair, request:

- exact Orin/PC hardware and power mode;
- JetPack, CUDA, TensorRT, driver and runtime versions;
- model artifact, precision, batch size and input shape;
- input/output size distributions;
- warm-up method and p50/p95/p99 latency;
- throughput at concurrency 1/2/4;
- peak host/device memory;
- average/peak power or joules per task;
- quality/accuracy for local and edge variants.

## Project layout

```text
backend/
  app/                         FastAPI and web-facing schemas
  edgesched/
    models.py                  transport-neutral domain model
    dag.py                     validation, readiness and failure propagation
    scheduler.py               constraints, cost, locality and critical path
    engine.py                  deterministic event-driven simulator
    profiling.py               replaceable profiling catalogue
    transports/                transport protocol and in-memory adapter
  tests/
frontend/                      React benchmark and DAG UI
configs/profiles.synthetic.json
proto/edgesched_v2.proto       optional wire contract
```

## API additions

- `GET /api/health`: core and model-provider status.
- `GET /api/architecture`: active contracts and deprecated paths.
- `POST /api/validate-workflow`: validate and return topology without running.
- `POST /api/generate-scene`: generate a valid DAG benchmark.
- `POST /api/simulate`: run the unified event-driven scheduler.

## Deprecations and compatibility

- `backend.app.simulator` keeps its old function name but now delegates to the unified core.
- `backend.app.schedulers` is the v1 isolated-task implementation and is no longer called by the API.
- The v1 external HTTP callback does not receive topology or artifact locations. The API accepts the option but routes it to `dag_deadline` until a workflow-aware v2 adapter is supplied.
- The separate top-level `edgesched` repository contains the legacy v1 gRPC experiment runner.
