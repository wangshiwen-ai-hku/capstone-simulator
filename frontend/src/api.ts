import type {
  Algorithm,
  AgentChatResponse,
  ArchitectureResponse,
  BenchmarkScene,
  BenchmarkTemplate,
  GenerateSceneRequest,
  MarsAgentModel,
  RuntimeWorkflowRun,
  SchedulerRunOptions,
  SimulationResponse,
} from './types';
import { DEFAULT_BINARY_FORMULATION } from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
const TEMPLATE_WORKSPACE_STORAGE_KEY = 'mars.template.workspace-token.v1';
const TEMPLATE_WORKSPACE_HEADER = 'X-MARS-Workspace-Token';
const TEMPLATE_WORKSPACE_PATTERN = /^[a-f0-9]{64}$/;

function templateWorkspaceToken(): string {
  try {
    const stored = window.localStorage.getItem(TEMPLATE_WORKSPACE_STORAGE_KEY);
    if (stored && TEMPLATE_WORKSPACE_PATTERN.test(stored)) return stored;

    const bytes = new Uint8Array(32);
    window.crypto.getRandomValues(bytes);
    const generated = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    window.localStorage.setItem(TEMPLATE_WORKSPACE_STORAGE_KEY, generated);
    return generated;
  } catch {
    throw new Error('Template workspace storage is unavailable in this browser.');
  }
}

function templateWorkspaceHeaders(): Record<string, string> {
  return { [TEMPLATE_WORKSPACE_HEADER]: templateWorkspaceToken() };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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

export function chatWithAgent(payload: {
  thread_id?: string;
  message: string;
  model: MarsAgentModel;
  enable_web_search: boolean;
  current_scene?: BenchmarkScene;
  action?: 'message' | 'confirm' | 'restart';
}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 50_000);
  return request<AgentChatResponse>('/api/agent/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeout));
}

export function listTemplates() {
  return request<{ templates: BenchmarkTemplate[] }>('/api/templates', {
    headers: templateWorkspaceHeaders(),
  });
}

export function createTemplate(payload: {
  name: string;
  description: string;
  tags: string[];
  scene: BenchmarkScene;
}) {
  return request<BenchmarkTemplate>('/api/templates', {
    method: 'POST',
    headers: templateWorkspaceHeaders(),
    body: JSON.stringify(payload),
  });
}

export async function deleteTemplate(templateId: string) {
  const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(templateId)}`, {
    method: 'DELETE',
    headers: templateWorkspaceHeaders(),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
}

export function simulate(
  scene: BenchmarkScene,
  algorithm: Algorithm,
  seed: number,
  options: SchedulerRunOptions = {},
) {
  const optimizerOptions = serializedOptimizerOptions(algorithm, options);
  const formulation = serializedFormulation(algorithm, options);
  return request<SimulationResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify({
      scene,
      algorithm,
      network_jitter: 0.12,
      resource_noise: 0.05,
      seed,
      ...(formulation ? { formulation } : {}),
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
  const formulation = serializedFormulation(algorithm, options);
  return request<{ run_id: string; workflow_id: string; status: string }>(
    '/api/runtime/workflows',
    {
      method: 'POST',
      body: JSON.stringify({
        scene,
        algorithm,
        seed,
        ...(formulation ? { formulation } : {}),
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

function serializedFormulation(
  algorithm: Algorithm,
  options: SchedulerRunOptions,
) {
  const formulation = options.formulation
    ?? (algorithm === 'binary_offload' ? DEFAULT_BINARY_FORMULATION : undefined);
  if (formulation === undefined) return undefined;
  if (!formulation.trim()) {
    throw new RangeError('Scheduling formulation must be non-blank.');
  }
  return formulation.trim();
}
