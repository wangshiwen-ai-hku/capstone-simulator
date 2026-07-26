// @vitest-environment jsdom

import { render, screen, waitFor } from '@testing-library/react';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { BenchmarkScene } from './types';

const { scene } = vi.hoisted(() => ({ scene: {
  id: 'scene-test',
  title: 'Warehouse test scene',
  natural_language_description: 'A minimal scheduler UI fixture.',
  scenario_type: 'warehouse',
  difficulty: 'medium',
  nodes: [
    {
      id: 'robot_1',
      kind: 'robot',
      display_name: 'Robot 1',
      architecture: 'jetson-orin-nx',
      cpu_capacity: 2,
      gpu_capacity: 2,
      memory_gb: 16,
      bandwidth_mbps: 100,
      base_latency_ms: 2,
      safety_capable: true,
      capabilities: ['localization'],
      supported_models: [],
      max_concurrency: 2,
    },
  ],
  initial_resources: [
    {
      node_id: 'robot_1',
      cpu_util: 0,
      gpu_util: 0,
      memory_util: 0,
      temperature_c: 40,
      power_w: 15,
      network_latency_ms: 2,
      online: true,
    },
  ],
  tasks: [
    {
      id: 'task_1',
      name: 'Localization',
      source_robot_id: 'robot_1',
      task_type: 'localization',
      task_class: 'realtime_offloadable',
      priority: 5,
      compute_demand: 1,
      gpu_demand: 0.5,
      latency_budget_ms: 100,
      safety_level: 1,
      model_requirement: '',
      data_size_mb: 1,
      output_size_mb: 0.1,
      bandwidth_requirement_mbps: 10,
      energy_budget_j: 10,
      allow_local_fallback: true,
      result_verification: '',
      arrival_time_ms: 0,
      deadline_ms: 500,
      dependencies: [],
      stage_index: 0,
      expected_accuracy: 1,
      input_ports: [],
      output_ports: [],
    },
  ],
  data_edges: [],
  workflow_id: 'workflow-test',
  workflow_deadline_ms: 1000,
  failure_policy: 'fail_fast',
  stressors: [],
  success_criteria: [],
  generation_source: 'deterministic',
  generation_note: '',
} as BenchmarkScene }));

vi.mock('./api', () => ({
  health: vi.fn().mockResolvedValue({
    status: 'ok',
    provider: 'test',
    model: 'test',
    llm_configured: false,
    system: 'MARS',
    mars_version: 'test',
  }),
  generateScene: vi.fn().mockResolvedValue(scene),
  getRuntimeWorkflow: vi.fn(),
  submitRuntimeWorkflow: vi.fn(),
}));

beforeAll(() => {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

describe('MARS Studio', () => {
  it('mounts the generated graph and runtime controls without a render loop', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Warehouse test scene')).toBeTruthy();
      expect(screen.getByDisplayValue('Localization')).toBeTruthy();
    });

    expect(screen.getByRole('button', { name: 'Run' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Reset' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Continue' })).toBeTruthy();
  });
});
