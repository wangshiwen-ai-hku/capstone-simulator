import { afterEach, describe, expect, it, vi } from 'vitest';
import { simulate, submitRuntimeWorkflow } from './api';
import type { BenchmarkScene } from './types';

const scene = { id: 'scene-wire-test' } as BenchmarkScene;

function response(payload: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => payload,
    text: async () => '',
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('scheduler API payloads', () => {
  it('serializes the binary formulation independently from optimizer_options', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ algorithm: 'binary_offload' }))
      .mockResolvedValueOnce(response({ run_id: 'run-1' }));
    vi.stubGlobal('fetch', fetchMock);

    await simulate(scene, 'binary_offload', 11, {
      communicationWeight: 0.4,
      formulation: 'one_hot_placement',
    });
    await submitRuntimeWorkflow(scene, 'binary_offload', 11, {
      communicationWeight: 0.4,
      formulation: 'one_hot_placement',
    });

    for (const [, init] of fetchMock.mock.calls) {
      const payload = JSON.parse(String(init.body));
      expect(payload.formulation).toBe('one_hot_placement');
      expect(payload.optimizer_options).toEqual({ communication_weight: 0.4 });
      expect(payload).not.toHaveProperty('beta');
    }
  });

  it('defaults binary requests to the legacy one-hot formulation', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ run_id: 'run-default' }));
    vi.stubGlobal('fetch', fetchMock);

    await submitRuntimeWorkflow(scene, 'binary_offload', 7);

    const payload = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(payload.formulation).toBe('one_hot_placement');
  });

  it('serializes an explicitly selected formulation for any algorithm', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ run_id: 'run-policy' }));
    vi.stubGlobal('fetch', fetchMock);

    await submitRuntimeWorkflow(scene, 'dag_deadline', 7, {
      formulation: ' deadline_aware ',
    });

    const payload = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(payload.formulation).toBe('deadline_aware');
    expect(payload).not.toHaveProperty('optimizer_options');
  });

  it('omits binary optimizer options for stable heuristic methods', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ run_id: 'run-2' }));
    vi.stubGlobal('fetch', fetchMock);

    await submitRuntimeWorkflow(scene, 'dag_deadline', 7, { communicationWeight: 9 });

    const payload = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(payload).not.toHaveProperty('formulation');
    expect(payload).not.toHaveProperty('optimizer_options');
    expect(payload).not.toHaveProperty('beta');
  });
});
