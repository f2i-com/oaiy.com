// Model files in words for a person: a download's state as a word ("Downloading…", "Paused", "Done"), not the
// downloader's ("active", "queued"); no note about the JSON file the quick-add list comes from; sizes in decimal
// units, as the engines' catalogue gives them; and OAIY not answering said once, with a way to ask again.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ list: vi.fn(), downloads: vi.fn(), catalog: vi.fn(), push: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, models: { ...real.models, list: m.list, downloads: m.downloads, catalog: m.catalog } };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: m.push }) }));

import ModelsPanel from './ModelsPanel';
import { ApiError, NOT_ANSWERING, type DownloadProgress } from './api';
import { invalidate } from './useCached';

let host: HTMLDivElement;
let root: Root;

async function settle() {
  for (let i = 0; i < 6; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

const download = (id: string, status: DownloadProgress['status'], url = `https://example.com/${id}.gguf`): DownloadProgress => ({
  id,
  url,
  filename: `${id}.gguf`,
  subdir: 'llm',
  destPath: `/Users/sam/models/llm/${id}.gguf`,
  status,
  bytesDownloaded: 1e9,
  bytesTotal: 4_920_739_232,
  startedAt: '2026-10-19T00:00:00Z',
  finishedAt: null,
  error: null,
  resumable: true,
  speedBps: null,
  etaSecs: null,
});

beforeEach(() => {
  invalidate('modelsSnapshot');
  for (const f of Object.values(m)) f.mockReset();
  m.list.mockResolvedValue({ rootDir: '/Users/sam/models', models: [], total: 0, offset: 0, limit: 200, freeBytes: 212e9 });
  m.downloads.mockResolvedValue([]);
  m.catalog.mockResolvedValue({
    sourcePath: '/Users/sam/Library/Application Support/com.oaiy.app/model-catalog.json',
    catalog: {
      categories: [
        {
          id: 'llm',
          name: 'LLM (GGUF)',
          description: 'Local chat models.',
          models: [{ id: 'a-model', name: 'A Model 8B (Q4_K_M)', description: 'A chat model.', url: 'https://example.com/busy.gguf', filename: 'busy.gguf', subdir: 'llm', sizeBytes: 4_920_739_232 }],
        },
      ],
    },
  });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

async function render() {
  await act(async () => {
    root.render(<ModelsPanel />);
  });
  await settle();
}

describe('Model files', () => {
  it('says each download’s state in a word, and sizes in decimal units', async () => {
    m.downloads.mockResolvedValue([download('busy', 'active', 'https://example.com/busy.gguf'), download('waits', 'queued'), download('held', 'paused'), download('got', 'completed')]);
    await render();
    const badges = Array.from(host.querySelectorAll('.dl-head .badge')).map((b) => b.textContent);
    expect(badges).toEqual(['Downloading…', 'Waiting…', 'Paused', 'Done']);
    expect(host.textContent).not.toMatch(/\b(active|queued)…/);
    expect(host.textContent).not.toContain('GiB');
  });

  it('is what is on disk and what is downloading: no second catalogue of models, and no file to edit', async () => {
    await render();
    // Getting a model is the Engines page's Get models, the one place for it.
    expect(host.textContent).not.toContain('Quick add');
    expect(host.querySelector('.catalog-card')).toBeNull();
    expect(host.querySelector('.models-what')).toBeNull();
    // One file by its address is still there for a program of the person's own, closed until asked for.
    expect(host.querySelector('details.models-by-address')!.hasAttribute('open')).toBe(false);
    expect(host.querySelector('details.models-by-address > summary')!.textContent).toBe('Download one file by its address');
    expect(host.textContent).not.toContain('model-catalog.json');
    expect(host.textContent).not.toContain('edit it to add your own');
  });

  it('says OAIY is not answering, once, with a way to ask again', async () => {
    m.list.mockRejectedValue(new ApiError(0, NOT_ANSWERING));
    await render();
    const banner = host.querySelector('.load-error')!;
    expect(banner.textContent).toContain(NOT_ANSWERING);
    expect(banner.textContent).not.toContain('Couldn');
    expect(banner.querySelector('button')!.textContent).toContain('Try again');
  });
});
