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
  it('serializes the binary communication weight under optimizer_options', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ algorithm: 'binary_offload' }))
      .mockResolvedValueOnce(response({ run_id: 'run-1' }));
    vi.stubGlobal('fetch', fetchMock);

    await simulate(scene, 'binary_offload', 11, { communicationWeight: 0.4 });
    await submitRuntimeWorkflow(scene, 'binary_offload', 11, { communicationWeight: 0.4 });

    for (const [, init] of fetchMock.mock.calls) {
      const payload = JSON.parse(String(init.body));
      expect(payload.optimizer_options).toEqual({ communication_weight: 0.4 });
      expect(payload).not.toHaveProperty('beta');
    }
  });

  it('omits binary optimizer options for stable heuristic methods', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ run_id: 'run-2' }));
    vi.stubGlobal('fetch', fetchMock);

    await submitRuntimeWorkflow(scene, 'dag_deadline', 7, { communicationWeight: 9 });

    const payload = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(payload).not.toHaveProperty('optimizer_options');
    expect(payload).not.toHaveProperty('beta');
  });
});
