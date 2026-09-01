// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createTemplate,
  deleteTemplate,
  listTemplates,
  simulate,
  submitRuntimeWorkflow,
} from './api';
import type { BenchmarkScene } from './types';

const scene = { id: 'scene-wire-test' } as BenchmarkScene;
const STORAGE_KEY = 'mars.template.workspace-token.v1';
const WORKSPACE_HEADER = 'X-MARS-Workspace-Token';

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
  window.localStorage.clear();
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

describe('template workspace capability', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ templates: [] }),
      text: async () => '',
    }));
  });

  it('persists one strong token and sends it on template list/create/delete', async () => {
    await listTemplates();
    await createTemplate({
      name: 'Saved benchmark',
      description: '',
      tags: [],
      scene: {} as BenchmarkScene,
    });
    await deleteTemplate('template_0123456789ab');

    const fetchMock = vi.mocked(fetch);
    const workspaceTokens = fetchMock.mock.calls.map(([, init]) => (
      new Headers(init?.headers).get(WORKSPACE_HEADER)
    ));
    expect(workspaceTokens).toHaveLength(3);
    expect(workspaceTokens[0]).toMatch(/^[a-f0-9]{64}$/);
    expect(new Set(workspaceTokens).size).toBe(1);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(workspaceTokens[0]);
  });

  it('replaces an invalid stored token instead of sending it', async () => {
    window.localStorage.setItem(STORAGE_KEY, '../../another-workspace');

    await listTemplates();

    const [, init] = vi.mocked(fetch).mock.calls[0];
    const sent = new Headers(init?.headers).get(WORKSPACE_HEADER);
    expect(sent).toMatch(/^[a-f0-9]{64}$/);
    expect(sent).not.toBe('../../another-workspace');
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(sent);
  });
});
