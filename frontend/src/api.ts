import type {
  Algorithm,
  ArchitectureResponse,
  BenchmarkScene,
  GenerateSceneRequest,
  RuntimeWorkflowRun,
  SchedulerRunOptions,
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
    trace_archive?: {
      enabled: boolean;
      layout?: string;
      schema_version?: string;
    };
  }>('/api/health');
}

export function getArchitecture() {
  return request<ArchitectureResponse>('/api/architecture');
}

export function generateScene(payload: GenerateSceneRequest) {
  return request<BenchmarkScene>('/api/generate-scene', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function simulate(
  scene: BenchmarkScene,
  algorithm: Algorithm,
  seed: number,
  options: SchedulerRunOptions = {},
) {
  const optimizerOptions = serializedOptimizerOptions(algorithm, options);
  return request<SimulationResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify({
      scene,
      algorithm,
      network_jitter: 0.12,
      resource_noise: 0.05,
      seed,
      ...(optimizerOptions ? { optimizer_options: optimizerOptions } : {}),
    }),
  });
}

export function submitRuntimeWorkflow(
  scene: BenchmarkScene,
  algorithm: Algorithm,
  seed: number,
  options: SchedulerRunOptions = {},
) {
  const optimizerOptions = serializedOptimizerOptions(algorithm, options);
  return request<{ run_id: string; workflow_id: string; status: string }>(
    '/api/runtime/workflows',
    {
      method: 'POST',
      body: JSON.stringify({
        scene,
        algorithm,
        seed,
        ...(optimizerOptions ? { optimizer_options: optimizerOptions } : {}),
        max_attempts: 2,
        inject_first_failure: false,
        deterministic: true,
      }),
    },
  );
}

export function getRuntimeWorkflow(runId: string) {
  return request<RuntimeWorkflowRun>(`/api/runtime/workflows/${runId}`);
}

function serializedOptimizerOptions(
  algorithm: Algorithm,
  options: SchedulerRunOptions,
) {
  if (algorithm !== 'binary_offload' || options.communicationWeight === undefined) {
    return undefined;
  }
  if (!Number.isFinite(options.communicationWeight)) {
    throw new RangeError('Communication weight must be a finite number.');
  }
  return { communication_weight: options.communicationWeight };
}
