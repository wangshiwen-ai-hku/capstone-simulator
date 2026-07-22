import type {
  Algorithm,
  BenchmarkScene,
  GenerateSceneRequest,
  RuntimeStatus,
  RuntimeWorkflowRun,
  SimulationResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function health() {
  return request<{
    status: string;
    provider: string;
    model: string;
    llm_configured: boolean;
    system: string;
    mars_version: string;
  }>('/api/health');
}

export function generateScene(payload: GenerateSceneRequest) {
  return request<BenchmarkScene>('/api/generate-scene', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function simulate(scene: BenchmarkScene, algorithm: Algorithm) {
  return request<SimulationResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify({
      scene,
      algorithm,
      network_jitter: 0.12,
      resource_noise: 0.05,
      seed: 7,
    }),
  });
}

export function bootstrapRuntime() {
  return request<RuntimeStatus>('/api/runtime/bootstrap', { method: 'POST' });
}

export function submitRuntimeWorkflow(
  scene: BenchmarkScene,
  algorithm: Algorithm,
  seed: number,
) {
  return request<{ run_id: string; workflow_id: string; status: string }>(
    '/api/runtime/workflows',
    {
      method: 'POST',
      body: JSON.stringify({
        scene,
        algorithm,
        seed,
        max_attempts: 2,
        inject_first_failure: true,
        failure_task_type: 'local_llm_7b',
        deterministic: true,
      }),
    },
  );
}

export function getRuntimeWorkflow(runId: string) {
  return request<RuntimeWorkflowRun>(`/api/runtime/workflows/${runId}`);
}
