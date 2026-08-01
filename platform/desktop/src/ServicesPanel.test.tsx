// Services panel load behaviour. The user-visible symptom was "slow loading":
// the panel started from null on every mount, so switching to Services always
// showed "Loading services…" and a round trip, even though Overview polls the
// same endpoint and had the answer cached.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listMock, ollamaMock, ggufMock, configMock, gpusMock, pushMock, autostartMock } = vi.hoisted(() => ({
  listMock: vi.fn(),
  ollamaMock: vi.fn(),
  ggufMock: vi.fn(),
  configMock: vi.fn(),
  gpusMock: vi.fn(),
  pushMock: vi.fn(),
  autostartMock: vi.fn(),
}));

vi.mock('./api', () => ({
  services: {
    list: listMock,
    start: vi.fn(),
    stop: vi.fn(),
    install: vi.fn(),
    uninstall: vi.fn(),
    cancelInstall: vi.fn(),
    repair: vi.fn(),
    setAutostart: autostartMock,
    delete: vi.fn(),
    add: vi.fn(),
    import: vi.fn(),
    export: vi.fn(),
    logs: vi.fn(),
  },
  appConfig: {
    listOllamaModels: ollamaMock,
    listGgufModels: ggufMock,
    get: configMock,
    listGpus: gpusMock,
    setServiceGpu: vi.fn(),
    setLlamaModel: vi.fn(),
    setOllamaModel: vi.fn(),
  },
  openExternal: vi.fn(),
}));
vi.mock('./Toasts', () => ({ useToast: () => ({ push: pushMock }) }));
vi.mock('./LogsViewer', () => ({ default: () => null }));

import ServicesPanel from './ServicesPanel';
import { invalidate, put } from './useCached';

const service = (over: Record<string, unknown> = {}) => ({
  id: 'ollama',
  name: 'Ollama',
  category: 'LLM',
  status: 'stopped',
  port: 11434,
  defaultPort: 11434,
  installed: true,
  gpu: null,
  error: null,
  ...over,
});

const SNAPSHOT = { services: [service()], dataDir: 'C:\\data' };

let host: HTMLDivElement;
let root: Root;

async function mount() {
  await act(async () => {
    root.render(<ServicesPanel />);
  });
}

const text = () => host.textContent ?? '';

beforeEach(() => {
  invalidate();
  listMock.mockReset();
  ollamaMock.mockReset();
  ggufMock.mockReset();
  configMock.mockReset();
  gpusMock.mockReset();
  autostartMock.mockReset();
  autostartMock.mockResolvedValue(undefined);
  listMock.mockResolvedValue(SNAPSHOT);
  ollamaMock.mockResolvedValue(['qwen2.5:0.5b']);
  ggufMock.mockResolvedValue([]);
  configMock.mockResolvedValue({ ollamaModel: null, llamaModel: null });
  gpusMock.mockResolvedValue([]);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('ServicesPanel loading', () => {
  it('paints from the cache instead of flashing "Loading services…"', async () => {
    put('servicesSnapshot', SNAPSHOT);
    // Never resolves: proves the first paint owes nothing to the network.
    listMock.mockReturnValue(new Promise(() => {}));

    await act(async () => {
      root.render(<ServicesPanel />);
    });

    expect(text()).not.toContain('Loading services');
    expect(text()).toContain('Ollama');
  });

  it('still shows the loading state on a genuinely cold start', async () => {
    listMock.mockReturnValue(new Promise(() => {}));
    await act(async () => {
      root.render(<ServicesPanel />);
    });
    // Nothing cached and nothing fetched yet — saying so is correct here.
    expect(text()).toContain('Loading services');
  });

  it('fills the cache Overview reads, so arriving there is instant too', async () => {
    await mount();
    const { peek } = await import('./useCached');
    expect(peek('servicesSnapshot')).toEqual(SNAPSHOT);
    expect(peek('services')).toEqual(SNAPSHOT.services);
  });

  it('does not probe a stopped Ollama for its model list', async () => {
    await mount();
    // A stopped server cannot answer, so the request is a guaranteed wait for
    // a connection that never completes usefully — on every visit.
    expect(ollamaMock).not.toHaveBeenCalled();
    expect(text()).toContain('Start Ollama');
  });

  it('does probe Ollama once it is actually running', async () => {
    listMock.mockResolvedValue({ ...SNAPSHOT, services: [service({ status: 'running' })] });
    await mount();
    expect(ollamaMock).toHaveBeenCalled();
  });
});

// "Start with the app" is a stored preference, not a second Start button. The
// distinction matters on both edges: ticking must not launch the service now,
// and a failed write must not leave a ticked box that will be gone at the next
// launch (the service would simply never come up, with nothing to explain it).
describe('ServicesPanel start-with-the-app', () => {
  const box = () => host.querySelector<HTMLInputElement>('input[type="checkbox"]');

  it('reflects the stored preference rather than whether it is running', async () => {
    listMock.mockResolvedValue({
      ...SNAPSHOT,
      services: [service({ status: 'stopped', autostart: true })],
    });
    await mount();
    expect(box()?.checked).toBe(true);
  });

  it('is unticked for a snapshot from a desktop build that has no such field', async () => {
    await mount();
    expect(box()).not.toBeNull();
    expect(box()?.checked).toBe(false);
  });

  it('saves the preference without starting the service', async () => {
    await mount();
    const el = box()!;
    // el.click() lets jsdom toggle `checked` and fire the click React maps to
    // onChange; setting `checked` by hand first makes React see no change.
    await act(async () => {
      el.click();
    });
    expect(autostartMock).toHaveBeenCalledWith('ollama', true);
    // The only lever that starts a service is Start.
    const { services: api } = await import('./api');
    expect(api.start).not.toHaveBeenCalled();
  });

  it('rolls the tick back when the preference could not be saved', async () => {
    autostartMock.mockRejectedValue(new Error('disk full'));
    await mount();
    const el = box()!;
    // el.click() lets jsdom toggle `checked` and fire the click React maps to
    // onChange; setting `checked` by hand first makes React see no change.
    await act(async () => {
      el.click();
    });
    expect(box()?.checked).toBe(false);
    expect(text()).toContain('disk full');
  });
});
