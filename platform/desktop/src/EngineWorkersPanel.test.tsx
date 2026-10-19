import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  network: { status: vi.fn(), nodes: vi.fn(), setDefault: vi.fn() },
  engines: { catalog: vi.fn() },
}));
vi.mock('./api', () => api);
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import EngineWorkersPanel from './EngineWorkersPanel';

let container: HTMLDivElement;
let root: Root;

const primary = { enabled: true, role: 'primary', uses: true, lends: false, name: 'Mac', defaults: {} };
const rig = {
  nodeId: 'w1',
  name: 'Rig',
  role: 'worker',
  online: true,
  lastError: null,
  devices: [{ id: 'gpu0', name: 'RTX 5090', memory_total_mb: 32768, memory_used_mb: 4096, load: 12 }],
  models: [
    { id: 'engine:llm:node.w1/qwen', kind: 'llm', model: 'qwen', state: 'ready', device: 'gpu0', pinned: false },
    { id: 'engine:image:node.w1/flux', kind: 'image', model: 'flux', state: 'not loaded', device: null, pinned: false },
    { id: 'engine:image:node.w1/moss', kind: 'image', model: 'moss', state: 'kept', device: 'gpu0', pinned: false },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  api.network.nodes.mockResolvedValue({ nodes: [rig] });
  api.engines.catalog.mockResolvedValue({ running: true, defaults: { llm: 'local-llm', image: 'local-image' } });
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});
async function mount(onOpenNetworking = () => {}) {
  await act(async () => root.render(<EngineWorkersPanel onOpenNetworking={onOpenNetworking} />));
  await act(async () => {});
}
const rows = (testId: string) => Array.from(container.querySelectorAll(`[data-testid="${testId}"]`));

describe('EngineWorkersPanel', () => {
  it('says where to switch OAIY Network on, and asks for nothing else while it is off', async () => {
    api.network.status.mockResolvedValue({ ...primary, enabled: false, role: 'standalone', uses: false });
    const open = vi.fn();
    await mount(open);
    expect(container.textContent).toContain('It is off until you switch on Use other computers');
    expect(api.network.nodes).not.toHaveBeenCalled();
    await act(async () => Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.includes('Open Networking'))!.click());
    expect(open).toHaveBeenCalled();
  });

  it("lists each kind's default as this computer's own, and a worker's models by kind, each to make the default", async () => {
    api.network.status.mockResolvedValue(primary);
    api.network.setDefault.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/moss', nodeId: 'w1', nodeName: 'Rig', model: 'moss' } } });
    await mount();
    const defaults = rows('engine-default').map((r) => r.textContent);
    expect(defaults).toEqual(['Textlocal-llmthis computer', 'Picturelocal-imagethis computer']);
    const worker = rows('engine-worker')[0];
    expect(worker.textContent).toContain('RTX 5090');
    const moss = Array.from(worker.querySelectorAll('.network-model')).find((r) => r.textContent?.includes('moss'))!;
    await act(async () => moss.querySelector('button')!.click());
    expect(api.network.setDefault).toHaveBeenCalledWith('image', 'engine:image:node.w1/moss');
  });

  it("shows a worker's model as the default (its language model, whichever it runs now), and gives it back to this computer", async () => {
    api.network.status.mockResolvedValue({
      ...primary,
      defaults: {
        image: { id: 'engine:image:node.w1/moss', nodeId: 'w1', nodeName: 'Rig', model: 'moss' },
        llm: { id: 'engine:llm:node.w1/old-llm', nodeId: 'w1', nodeName: 'Rig', model: 'old-llm' },
      },
    });
    api.network.setDefault.mockResolvedValue(primary);
    await mount();
    const [text, picture] = rows('engine-default');
    expect(text.textContent).toContain('qwen');
    expect(text.textContent).toContain('on Rig');
    expect(picture.textContent).toContain('moss');
    const worker = rows('engine-worker')[0];
    const chosen = Array.from(worker.querySelectorAll('.network-model.is-default')).map((r) => r.querySelector('code')!.textContent);
    expect(chosen).toEqual(['qwen', 'moss']);
    await act(async () => picture.querySelector('button')!.click());
    expect(api.network.setDefault).toHaveBeenCalledWith('image', null);
  });

  it("gives back a default whose model the worker no longer has, though no model of its kind is listed", async () => {
    api.network.status.mockResolvedValue({ ...primary, defaults: { video: { id: 'engine:video:node.w1/ltx', nodeId: 'w1', nodeName: 'Rig', model: 'ltx' } } });
    api.network.setDefault.mockResolvedValue(primary);
    await mount();
    const video = rows('engine-default').find((r) => r.textContent?.includes('ltx'))!;
    expect(video.textContent).toContain('Rig no longer has this model');
    await act(async () => video.querySelector('button')!.click());
    expect(api.network.setDefault).toHaveBeenCalledWith('video', null);
  });

  it('says a default on a worker that does not answer is offline, not run elsewhere', async () => {
    api.network.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/moss', nodeId: 'w1', nodeName: 'Rig', model: 'moss' } } });
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, online: false, lastError: 'connection refused' }] });
    await mount();
    expect(rows('engine-default')[1].textContent).toContain('offline');
    expect(rows('engine-worker')[0].textContent).toContain('connection refused');
  });

  it.each([
    ['not_heard', 'Not in effect: Rig has not been heard from since this computer started; this computer’s own model runs meanwhile.'],
    ['engines_down', 'Not in effect: Rig’s engines are not running; this computer’s own model runs meanwhile.'],
    ['model_gone', 'Not in effect: Rig no longer has this model; this computer’s own model runs meanwhile.'],
  ])('says a default the paired computer cannot run now is not in effect (%s), and why', async (waiting, said) => {
    api.network.status.mockResolvedValue({
      ...primary,
      defaults: { image: { id: 'engine:image:node.w1/moss', nodeId: 'w1', nodeName: 'Rig', model: 'moss', inEffect: false, waiting } },
    });
    await mount();
    const picture = rows('engine-default').find((r) => r.textContent?.includes('moss'))!;
    expect(picture.querySelector('[data-testid="engine-default-waiting"]')?.textContent?.trim()).toBe(said);
    // (beside the model on its computer too: not a plain "Default")
    const moss = Array.from(rows('engine-worker')[0].querySelectorAll('.network-model')).find((r) => r.textContent?.includes('moss'))!;
    expect(moss.querySelector('.engine-default-chip')?.textContent?.trim()).toBe('Default, not in effect');
    expect(moss.querySelector('.engine-default-chip')?.getAttribute('title')).toBe(said);
  });

  it('says nothing of the kind for a default in effect', async () => {
    api.network.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/moss', nodeId: 'w1', nodeName: 'Rig', model: 'moss', inEffect: true, waiting: null } } });
    await mount();
    expect(container.querySelector('[data-testid="engine-default-waiting"]')).toBeNull();
    expect(container.textContent).not.toContain('no longer has this model');
    expect(container.querySelector('.network-model.is-default .engine-default-chip')?.textContent?.trim()).toBe('Default');
  });

  it('cannot make an offline computer’s model the default, and says so', async () => {
    api.network.status.mockResolvedValue(primary);
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, online: false, lastError: 'connection refused (os error 61)', lastSeenMs: Date.now() - 3 * 3_600_000 }] });
    await mount();
    const worker = rows('engine-worker')[0];
    const make = Array.from(worker.querySelectorAll('button')).filter((b) => b.textContent?.includes('Make default'));
    expect(make.length).toBeGreaterThan(0);
    for (const b of make) {
      expect(b.disabled).toBe(true);
      expect(b.title).toBe('Rig is offline: it can be made the default once it answers again');
    }
    expect(worker.querySelector('.card-warn')?.textContent?.trim()).toBe('connection refused (os error 61). Last heard from 3 h ago.');
  });

  it('points to “Use this computer’s” to make this computer’s own the default again, and calls the computers paired', async () => {
    api.network.status.mockResolvedValue(primary);
    api.network.nodes.mockResolvedValue({ nodes: [] });
    await mount();
    expect(container.textContent).toContain('“Use this computer’s”, beside a default, makes this computer’s own the default again.');
    expect(container.textContent).not.toContain('Choosing a default in the Engines tab');
    expect(container.textContent).toContain('Paired computers');
    expect(container.textContent).not.toMatch(/worker/i);
  });
});
