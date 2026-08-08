// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import App, { slotUtilization, taskPlayback } from './App';
import {
  generateScene,
  getRuntimeWorkflow,
  health,
  submitRuntimeWorkflow,
} from './api';
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

beforeEach(() => {
  vi.mocked(health).mockReset().mockResolvedValue({
    status: 'ok',
    provider: 'test',
    model: 'test',
    llm_configured: false,
    system: 'MARS',
    mars_version: 'test',
  });
  vi.mocked(generateScene).mockReset().mockResolvedValue(scene);
  vi.mocked(getRuntimeWorkflow).mockReset();
  vi.mocked(submitRuntimeWorkflow).mockReset();
});

afterEach(() => cleanup());

describe('MARS Studio', () => {
  it('mounts the generated graph and runtime controls without a render loop', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Warehouse test scene')).toBeTruthy();
      expect(screen.getAllByText('Localization').length).toBeGreaterThan(0);
    });

    expect(screen.getByRole('button', { name: 'Run' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Reset' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Continue' })).toBeTruthy();
    expect(
      (screen.getByRole('checkbox', { name: 'Use LLM scene generation' }) as HTMLInputElement).disabled,
    ).toBe(true);
  });

  it('passes the backend-owned LLM selection into scene generation', async () => {
    vi.mocked(health).mockResolvedValue({
      status: 'ok',
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      llm_configured: true,
      system: 'MARS',
      mars_version: 'test',
    });
    render(<App />);

    const llm = await screen.findByRole('checkbox', { name: 'Use LLM scene generation' });
    await waitFor(() => expect((llm as HTMLInputElement).disabled).toBe(false));
    fireEvent.click(llm);
    fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }));

    await waitFor(() => {
      expect(generateScene).toHaveBeenLastCalledWith(
        expect.objectContaining({ use_llm: true }),
      );
    });
    expect(screen.getByText('deepseek / deepseek-v4-flash')).toBeTruthy();
  });

  it('renders a failed workflow report instead of treating it as a transport error', async () => {
    vi.mocked(submitRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-failed',
      workflow_id: scene.workflow_id,
      status: 'accepted',
    });
    vi.mocked(getRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-failed',
      workflow_id: scene.workflow_id,
      status: 'failed',
      error: '',
      result: {
        workflow: {
          workflow_id: scene.workflow_id,
          state: 'failed',
          failure_policy: 'fail_fast',
          state_counts: { failed: 1 },
          critical_path: ['task_1'],
          topological_order: ['task_1'],
          levels: { task_1: 0 },
        },
        metrics: { makespan_ms: 0 },
        task_results: [{
          task_id: 'task_1',
          task_name: 'Localization',
          task_type: 'localization',
          task_class: 'realtime_offloadable',
          state: 'failed',
          source_node_id: 'robot_1',
          target_node_id: 'robot_1',
          mode: 'local',
          dependencies: [],
          attempt_count: 0,
          attempts: [],
          outputs: [],
        }],
        agents: [],
        data_edges: [],
        events: [],
        logs: [],
      },
    });
    render(<App />);

    await screen.findByText('Warehouse test scene');
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(screen.getAllByText('failed').length).toBeGreaterThan(0);
    });
    expect(screen.queryByRole('alert')).toBeNull();
  });
});

describe('slotUtilization', () => {
  it('caps active-slot demand at one for display', () => {
    expect(slotUtilization(5, 2)).toBe(1);
    expect(slotUtilization(1, 2)).toBe(0.5);
  });
});

describe('taskPlayback', () => {
  it('projects a no-attempt terminal task at workflow completion', () => {
    const task = { ...scene.tasks[0], arrival_time_ms: 500 };
    const playback = taskPlayback(task, {
      task_id: task.id,
      task_name: task.name,
      task_type: task.task_type,
      state: 'skipped',
      source_node_id: task.source_robot_id,
      target_node_id: '',
      mode: '',
      dependencies: task.dependencies,
      attempt_count: 0,
      attempts: [],
      outputs: [],
    }, 100, 100);

    expect(playback.state).toBe('skipped');
    expect(playback.progress).toBe(1);
  });
});
