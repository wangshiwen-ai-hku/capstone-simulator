// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MarsModePanel from './MarsModePanel';

vi.mock('./api', () => ({
  chatWithAgent: vi.fn(),
  createTemplate: vi.fn(),
  deleteTemplate: vi.fn(),
  listTemplates: vi.fn().mockResolvedValue({ templates: [] }),
}));

afterEach(() => cleanup());

describe('MARS Agent retrieval', () => {
  it('requires an explicit opt-in before querying arXiv', () => {
    render(
      <MarsModePanel
        mode="agent"
        studio={null}
        scene={null}
        expanded={false}
        onExpandedChange={vi.fn()}
        onImportScene={vi.fn()}
      />,
    );

    const retrieval = screen.getByRole('checkbox', {
      name: 'Retrieval via arXiv',
    }) as HTMLInputElement;
    expect(retrieval.checked).toBe(false);
    expect(retrieval.closest('label')?.title).toContain('generic workflow keywords');
  });
});
