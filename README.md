# Capstone MARS Benchmark

MARS = Multi-Agent Robot Scheduling benchmark. This is a runnable full-stack prototype for generating benchmark scenes and stress-test tasks for a multi-robot cloud-edge-device scheduling platform.

It matches the Capstone direction: robot nodes submit physical-AI workloads, the control plane observes node resources, schedules local/edge execution, and evaluates latency, energy, success rate, deadline misses, bandwidth and robustness.

## What is included

- **Backend**: Python + FastAPI.
- **Frontend**: React + TypeScript + plain CSS.
- **LLM scene generator**: OpenAI-compatible chat-completions client, configurable through `.env` for OpenAI / Doubao / GLM / Gemini / custom compatible endpoints.
- **Simulation engine**: deterministic fallback generator, workload abstraction, rule-based scheduler, local-first, edge-first, greedy-cost, and external algorithm callback support.
- **Algorithm integration**: point the UI/API to an external scheduler HTTP endpoint and compare it with built-in baselines.

## Directory

```text
backend/
  app/
    main.py
    config.py
    schemas.py
    llm_client.py
    generators.py
    schedulers.py
    simulator.py
  requirements.txt
  .env.example
frontend/
  package.json
  index.html
  tsconfig.json
  vite.config.ts
  src/
    App.tsx
    api.ts
    types.ts
    main.tsx
    styles.css
```

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Environment configuration

The backend uses one OpenAI-compatible client. Choose provider with `MODEL_PROVIDER` and fill the matching key/model/base URL.

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

For Doubao / GLM / Gemini, set provider-specific variables in `backend/.env`. Endpoint examples are intentionally left editable because each lab/company account may route through a gateway.

```env
MODEL_PROVIDER=doubao
DOUBAO_API_KEY=...
DOUBAO_BASE_URL=https://your-compatible-endpoint/v1
DOUBAO_MODEL=...

MODEL_PROVIDER=glm
GLM_API_KEY=...
GLM_BASE_URL=https://your-compatible-endpoint/v1
GLM_MODEL=...

MODEL_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://your-compatible-endpoint/v1
GEMINI_MODEL=...
```

When no key is configured, the backend automatically uses a deterministic rule generator, so the demo still runs offline.

## External scheduling algorithm interface

If you want to plug in a custom scheduler, expose an HTTP endpoint and put its URL into the frontend field **External Scheduler URL** or API request field `external_scheduler_url`.

Your service receives:

```json
{
  "scene": { "...": "full scene json" },
  "task": { "...": "current workload" },
  "resources": {
    "robot_1": { "cpu_util": 0.3, "gpu_util": 0.1, "memory_util": 0.2, "temperature_c": 52 },
    "edge_pc": { "cpu_util": 0.4, "gpu_util": 0.6, "memory_util": 0.5, "temperature_c": 61 }
  },
  "candidates": ["robot_1", "edge_pc", "robot_3"]
}
```

Return:

```json
{
  "target_node_id": "edge_pc",
  "mode": "edge",
  "reason": "lower expected finish time under current GPU load"
}
```

If the external algorithm fails or returns an invalid node, the simulator falls back to greedy-cost scheduling and records the fallback in logs.

## Suggested benchmark scenarios

- Similar-task conflict: multiple robots request the same edge model simultaneously.
- Priority queue: high-priority safety task preempts lower-priority perception/VLA tasks.
- Long-chain task: perception → planning → VLA inference → verification with dependencies.
- Network degradation: bandwidth and latency fluctuate during offloading.
- Edge overload: edge PC receives too many heavy requests.
- Local fallback: control plane unavailable or safety-critical task requires local execution.
