import type { Algorithm, BenchmarkScene, GenerateSceneRequest, SimulationResponse } from './types';

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
    architecture: string;
    edgesched_version: string;
  }>('/api/health');
}

export function generateScene(payload: GenerateSceneRequest) {
  return request<BenchmarkScene>('/api/generate-scene', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function simulate(scene: BenchmarkScene, algorithm: Algorithm, externalSchedulerUrl?: string) {
  return request<SimulationResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify({
      scene,
      algorithm,
      external_scheduler_url: externalSchedulerUrl || null,
      network_jitter: 0.12,
      resource_noise: 0.05,
      seed: 7,
    }),
  });
}
