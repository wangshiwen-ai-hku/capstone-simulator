// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import App, { slotUtilization, taskPlayback } from './App';
import {
  generateScene,
  getArchitecture,
  getRuntimeWorkflow,
  health,
  submitRuntimeWorkflow,
} from './api';
import type { BenchmarkScene, SchedulingAlgorithmCapability } from './types';

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

const legacyBinaryCapability = {
  id: 'binary_offload',
  label: 'Binary offload',
  kind: 'optimizer',
  stability: 'beta',
  execution_paths: ['simulation', 'runtime'],
  parameters: {
    communication_weight: {
      type: 'number',
      label: 'Communication weight',
      default: 0.25,
      minimum: 0,
      maximum: 2,
      step: 0.05,
      description: 'Balances expected success against communication time.',
    },
  },
  compatibility: {
    supported_node_kinds: ['robot', 'edge'],
    supports_multiple_nodes: true,
    requires_source_candidate: false,
    max_ready_tasks: 32,
  },
} satisfies SchedulingAlgorithmCapability;

const binaryCapability = {
  ...legacyBinaryCapability,
  default_formulation: 'one_hot_placement',
  supported_formulations: ['one_hot_placement'],
} satisfies SchedulingAlgorithmCapability;

const formulatedDeadlineCapability = {
  ...legacyBinaryCapability,
  id: 'dag_deadline',
  label: 'DAG deadline',
  kind: 'policy_alias',
  stability: 'stable',
  parameters: {},
  default_formulation: 'deadline_aware',
  supported_formulations: ['one_hot_placement', 'deadline_aware'],
} satisfies SchedulingAlgorithmCapability;

const optionalFormulationDeadlineCapability = {
  ...formulatedDeadlineCapability,
  default_formulation: null,
  supported_formulations: ['one_hot_placement'],
} satisfies SchedulingAlgorithmCapability;

vi.mock('./api', () => ({
  chatWithAgent: vi.fn(),
  createTemplate: vi.fn(),
  deleteTemplate: vi.fn(),
  listTemplates: vi.fn().mockResolvedValue({ templates: [] }),
  health: vi.fn().mockResolvedValue({
    status: 'ok',
    provider: 'test',
    model: 'test',
    llm_configured: false,
    system: 'MARS',
    mars_version: 'test',
  }),
  generateScene: vi.fn().mockResolvedValue(scene),
  getArchitecture: vi.fn().mockResolvedValue({}),
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
  vi.mocked(getArchitecture).mockReset().mockResolvedValue({});
  vi.mocked(getRuntimeWorkflow).mockReset();
  vi.mocked(submitRuntimeWorkflow).mockReset();
});

afterEach(() => cleanup());

describe('MARS Studio', () => {
  it('switches among Studio, Agent, and Templates and expands the Agent', async () => {
    render(<App />);
    await screen.findByText('Warehouse test scene');

    fireEvent.click(screen.getByRole('button', { name: 'Agent' }));
    expect(screen.getAllByText('Modelling copilot')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Expand MARS Agent' }));
    expect(document.querySelector('.studio-shell')?.className).toContain('agent-expanded');

    fireEvent.click(screen.getByRole('button', { name: 'Templates' }));
    expect(screen.getAllByText('Benchmark library')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Studio' }));
    expect(screen.getByLabelText('Scheduling method')).toBeTruthy();
  });

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
    expect(screen.getByLabelText('Scheduling method')).toBeTruthy();
  });

  it('keeps the five stable methods when capability discovery fails', async () => {
    vi.mocked(getArchitecture).mockRejectedValue(new Error('old backend'));
    render(<App />);

    await screen.findByText('Warehouse test scene');
    const method = screen.getByLabelText('Scheduling method') as HTMLSelectElement;
    expect(method.options).toHaveLength(5);
    expect(screen.queryByRole('option', { name: 'Binary offload' })).toBeNull();
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

  it('shows when LLM generation falls back to a deterministic scene', async () => {
    vi.mocked(health).mockResolvedValue({
      status: 'ok',
      provider: 'apiyi',
      model: 'deepseek-v4-flash',
      llm_configured: true,
      system: 'MARS',
      mars_version: 'test',
      trace_archive: {
        enabled: true,
      },
    });
    vi.mocked(generateScene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce({
        ...scene,
        generation_source: 'deterministic_fallback',
        generation_note: 'LLM apiyi failed; deterministic fallback used. Trace: trace-1.',
      });
    render(<App />);

    await screen.findByText('Warehouse test scene');
    const llm = await screen.findByRole('checkbox', { name: 'Use LLM scene generation' });
    fireEvent.click(llm);
    fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }));

    expect((await screen.findByRole('status')).textContent).toContain(
      'LLM apiyi failed; deterministic fallback used. Trace: trace-1.',
    );
    expect(screen.getByText('apiyi / deepseek-v4-flash | trace archive on')).toBeTruthy();
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

  it('discovers binary offload, uses its advertised default, and renders available metrics', async () => {
    vi.mocked(getArchitecture).mockResolvedValue({
      scheduling_capabilities: {
        schema_version: '1',
        algorithms: [binaryCapability],
      },
    });
    vi.mocked(submitRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-binary',
      workflow_id: scene.workflow_id,
      status: 'accepted',
    });
    vi.mocked(getRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-binary',
      workflow_id: scene.workflow_id,
      status: 'succeeded',
      error: '',
      result: {
        workflow: {
          workflow_id: scene.workflow_id,
          state: 'succeeded',
          failure_policy: 'fail_fast',
          state_counts: { succeeded: 1 },
          critical_path: ['task_1'],
          topological_order: ['task_1'],
          levels: { task_1: 0 },
          scheduling: {
            requested_algorithm: 'binary_offload',
            effective_optimizers: { binary_offload: 5, heuristic: 1 },
            effective_policies: { binary_offload: 6 },
            solve_statuses: { optimal: 5, time_limit: 1 },
            fallback_count: 1,
          },
        },
        metrics: {
          makespan_ms: 0,
          success_rate: 1,
          required_task_on_time_rate: 0.75,
          executed_deadline_miss_rate: 0.25,
          skipped_task_count: 1,
          total_solver_time_ms: 3.5,
          scheduling_epoch_count: 2,
          expected_success_reward: 4,
          workflow_evaluation_objective: -0.75,
        },
        task_results: [{
          task_id: 'task_1',
          task_name: 'Localization',
          task_type: 'localization',
          task_class: 'realtime_offloadable',
          state: 'succeeded',
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

    const method = await screen.findByLabelText('Scheduling method');
    await screen.findByRole('option', { name: 'Binary offload' });
    fireEvent.change(method, { target: { value: 'binary_offload' } });

    const weight = await screen.findByLabelText('Communication weight');
    expect((weight as HTMLInputElement).value).toBe('0.25');
    expect(screen.queryByLabelText('Formulation')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(submitRuntimeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({ id: scene.id }),
        'binary_offload',
        7,
        {
          communicationWeight: 0.25,
          formulation: 'one_hot_placement',
        },
      );
    });
    expect(await screen.findByLabelText('Runtime metrics')).toBeTruthy();
    expect(screen.getByText('Total solve')).toBeTruthy();
    expect(screen.getByText('3.50 ms')).toBeTruthy();
    expect(screen.getByText('Expected reward')).toBeTruthy();
    expect(screen.getByText('4.000')).toBeTruthy();
    expect(screen.getByText('Required tasks on time')).toBeTruthy();
    expect(screen.getByText('75.0%')).toBeTruthy();
    expect(screen.getByText('Executed deadline misses')).toBeTruthy();
    expect(screen.getByText('25.0%')).toBeTruthy();
    expect(screen.getByText('Skipped tasks')).toBeTruthy();
    expect(screen.getByText('Scheduling audit')).toBeTruthy();
    expect(screen.getByText('binary offload x 5, heuristic x 1')).toBeTruthy();
    expect(screen.getByText('Fallbacks')).toBeTruthy();
    expect(screen.queryByText('Longest solve')).toBeNull();
  });

  it('uses legacy one-hot defaults when formulation capabilities are absent', async () => {
    vi.mocked(getArchitecture).mockResolvedValue({
      scheduling_capabilities: {
        schema_version: '1',
        algorithms: [legacyBinaryCapability],
      },
    });
    vi.mocked(submitRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-legacy-binary',
      workflow_id: scene.workflow_id,
      status: 'accepted',
    });
    vi.mocked(getRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-legacy-binary',
      workflow_id: scene.workflow_id,
      status: 'failed',
      result: null,
      error: 'expected test stop',
    });
    render(<App />);

    const method = await screen.findByLabelText('Scheduling method');
    await screen.findByRole('option', { name: 'Binary offload' });
    fireEvent.change(method, { target: { value: 'binary_offload' } });
    expect(screen.queryByLabelText('Formulation')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(submitRuntimeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({ id: scene.id }),
        'binary_offload',
        7,
        {
          communicationWeight: 0.25,
          formulation: 'one_hot_placement',
        },
      );
    });
  });

  it('shows multiple formulations and selects the advertised default', async () => {
    vi.mocked(getArchitecture).mockResolvedValue({
      scheduling_capabilities: {
        schema_version: '1',
        algorithms: [{
          ...binaryCapability,
          default_formulation: 'relaxed_placement',
          supported_formulations: ['one_hot_placement', 'relaxed_placement'],
        }],
      },
    });
    render(<App />);

    const method = await screen.findByLabelText('Scheduling method');
    await screen.findByRole('option', { name: 'Binary offload' });
    fireEvent.change(method, { target: { value: 'binary_offload' } });

    await waitFor(() => {
      expect((screen.getByLabelText('Formulation') as HTMLSelectElement).value).toBe(
        'relaxed_placement',
      );
    });
  });

  it('drives formulation selection from the selected policy capability', async () => {
    vi.mocked(getArchitecture).mockResolvedValue({
      scheduling_capabilities: {
        schema_version: '1',
        algorithms: [formulatedDeadlineCapability],
      },
    });
    vi.mocked(submitRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-formulated-policy',
      workflow_id: scene.workflow_id,
      status: 'accepted',
    });
    vi.mocked(getRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-formulated-policy',
      workflow_id: scene.workflow_id,
      status: 'failed',
      result: null,
      error: 'expected test stop',
    });
    render(<App />);

    const selector = await screen.findByLabelText('Formulation');
    expect((selector as HTMLSelectElement).value).toBe('deadline_aware');
    fireEvent.change(selector, { target: { value: 'one_hot_placement' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(submitRuntimeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({ id: scene.id }),
        'dag_deadline',
        7,
        { formulation: 'one_hot_placement' },
      );
    });
  });

  it('does not implicitly select an optional policy formulation', async () => {
    vi.mocked(getArchitecture).mockResolvedValue({
      scheduling_capabilities: {
        schema_version: '1',
        algorithms: [optionalFormulationDeadlineCapability],
      },
    });
    vi.mocked(submitRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-unformulated-policy',
      workflow_id: scene.workflow_id,
      status: 'accepted',
    });
    vi.mocked(getRuntimeWorkflow).mockResolvedValue({
      run_id: 'run-unformulated-policy',
      workflow_id: scene.workflow_id,
      status: 'failed',
      result: null,
      error: 'expected test stop',
    });
    render(<App />);

    const selector = await screen.findByLabelText('Formulation');
    expect((selector as HTMLSelectElement).value).toBe('');
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(submitRuntimeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({ id: scene.id }),
        'dag_deadline',
        7,
        {},
      );
    });
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
