# MARS

MARS is the transport-neutral scheduling core for multi-robot edge workflows.
It defines middleware-level domain models, DAG lifecycle, placement policies,
and a transport interface. The repository also contains a FastAPI adapter and a
React benchmark interface for generating, validating, simulating, and
inspecting workflows.

The runnable shape is:

```text
React UI ──► FastAPI adapter ──► MARS scheduling core
                                    ├── DAG manager
                                    ├── scheduler + profiles
                                    └── deterministic engine

Future executors ──► SchedulerTransport interface
```

The dependency direction is one way: `backend` imports `mars`; MARS does not
depend on the web application. This keeps the scheduling/control-plane layer
available to future node agents, hardware executors, and transport adapters.
The current runnable path is an in-process simulator, not a live distributed
control plane.

## Current scope

- Atomic DAG validation: unique IDs, known parents, no self-dependencies and no cycles.
- Authoritative `BLOCKED → READY → RUNNING → terminal` lifecycle.
- Multi-parent release, idempotent completion, descendant skipping and fail-fast policies.
- Critical-path/deadline-aware scheduling with intermediate-artifact locality.
- Three workload classes enforced as hard placement rules.
- Configurable synthetic profiling catalogue for runs without workload artifacts.
- Updated React UI with workflow status, task-class metrics and a DAG stage view.

## The three task classes

| Class | Typical work | Placement contract |
|---|---|---|
| `local_safety` | obstacle avoidance, emergency control | Must run on its safety-capable source robot |
| `realtime_offloadable` | YOLO, segmentation, path planning, verification | May run on its source robot or edge |
| `edge_heavy` | VLA/LLM, map fusion, compression and heavy planning | Prefer edge; local fallback only when `allow_local_fallback=true` |

Task categories remain detailed benchmark labels. `task_class` is the stable
placement contract.

## Quick start

Python 3.10–3.13 is recommended.

### Python environment and backend

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

Open `http://localhost:5173`. Select `dag_deadline` for the new scheduler.

### Tests

Install the development requirements, then run the core, transport, and web
adapter tests from the repository root:

```bash
pip install -r backend/requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Profiles when real tasks are unavailable

`configs/mars/profiles.synthetic.json` contains clearly labelled placeholder rows.
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
mars/                          scheduling/control-plane middleware
  models.py                    transport-neutral domain model
  dag.py                       validation, readiness, failure propagation
  scheduler.py                 constraints, cost, locality, critical path
  engine.py                    deterministic event-driven simulator
  profiling.py                 replaceable profiling catalogue
  transports/                  transport protocol and in-memory adapter
backend/
  app/                         FastAPI adapter and web-facing schemas
frontend/                      React benchmark and DAG UI
configs/mars/                  runtime and profiling configuration
tests/                         middleware tests
```

## API additions

- `GET /api/health`: MARS and model-provider status.
- `GET /api/architecture`: active core, runtime, transport interfaces, and task classes.
- `POST /api/validate-workflow`: validate and return topology without running.
- `POST /api/generate-scene`: generate a valid DAG benchmark.
- `POST /api/simulate`: run the MARS event-driven scheduler.

## Transport boundary

The deterministic web simulator runs in process. `SchedulerTransport` is the
Python interface for future executors, and `InMemoryTransport` exercises that
boundary in tests. A network wire protocol should be added together with its
first real transport adapter so the two contracts cannot drift.
