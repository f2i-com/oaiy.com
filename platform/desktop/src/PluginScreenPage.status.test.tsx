import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { list, start, trust, ask, toast } = vi.hoisted(() => ({ list: vi.fn(), start: vi.fn(), trust: vi.fn(), ask: vi.fn(), toast: { push: vi.fn() } }));
vi.mock('./api', () => ({
  API_BASE: 'http://127.0.0.1:17972',
  plugins: { list, start, trust }, bridge: {}, companion: {},
}));
vi.mock('./Toasts', () => ({ useToast: () => toast }));
vi.mock('./ask', () => ({ ask }));

import PluginScreenPage from './PluginScreenPage';
import { testMessageEvent, testPorts } from './pluginScreenRpc.testTransport';
const transports = new WeakMap<HTMLIFrameElement, ReturnType<typeof testPorts>>();

const plugin = (state: string, reason?: string) => ({
  id: 'aokie', state, reason,
  manifest: {
    name: 'Aokie', version: '1.0',
    ui: {
      nav: [{ id: 'calls', screen: 'receptionist' }],
      screens: [{ id: 'receptionist', title: 'Receptionist', entry: 'index.html', files: ['index.html'] }],
    },
  },
});

let container: HTMLDivElement;
let root: Root;
let fetchAssets: ReturnType<typeof vi.fn>;

beforeEach(() => {
  list.mockReset();
  start.mockReset().mockResolvedValue(undefined);
  trust.mockReset().mockResolvedValue({});
  ask.mockReset();
  list.mockResolvedValue({ plugins: [plugin('unhealthy', 'LLM was unavailable')] });
  fetchAssets = vi.fn().mockResolvedValue({ ok: true, text: async () => '<div id="transcript"></div>' });
  vi.stubGlobal('fetch', fetchAssets);
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

/** Fetches of the screen's own files (the host's font loader fetches too). */
const assetFetches = () => fetchAssets.mock.calls.filter(([url]) => String(url).includes('/api/plugins/')).length;

async function mount() {
  await act(async () => root.render(<PluginScreenPage pluginId="aokie" navId="calls" />));
  return container.querySelector('iframe')!;
}

function snapshot(frame: HTMLIFrameElement, id: string) {
  const nonce = JSON.parse(frame.srcdoc.match(/window\.__oaiyDocumentNonce=("[^"]+")/)![1]);
  let ports = transports.get(frame);
  if (!ports) {
    ports = testPorts(); transports.set(frame, ports);
    window.dispatchEvent(testMessageEvent({ source: frame.contentWindow, ports: [ports.port2 as unknown as MessagePort], data: { __pluginHost: 1, documentNonce: nonce, method: 'document.connect' } }));
  }
  ports.port1.postMessage({ __pluginHost: 1, documentNonce: nonce, id, method: 'snapshot', args: [] });
}

describe('plugin screen runtime status', () => {
  it('clears a recovered health warning without reloading the iframe or its transcript', async () => {
    const frame = await mount();
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('LLM was unavailable');
    const source = frame.srcdoc;
    frame.contentDocument!.body.innerHTML = '<p id="saved-turn">Caller: Friday afternoon please.</p>';
    const transcript = frame.contentDocument!.getElementById('saved-turn');
    // A real API refresh returns a fresh manifest object too. Updating the
    // asset record here would refetch srcdoc and erase the frame's live state.
    list.mockResolvedValue({ plugins: [plugin('running')] });
    await act(async () => snapshot(frame, 'healthy'));
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.querySelector('iframe')).toBe(frame);
    expect(frame.srcdoc).toBe(source);
    expect(frame.contentDocument!.getElementById('saved-turn')).toBe(transcript);
    expect(assetFetches()).toBe(1);
  });

  it('does not let an older snapshot put a recovered plugin back into unhealthy state', async () => {
    const frame = await mount();
    let resolveOld!: (value: unknown) => void;
    list.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
    list.mockResolvedValueOnce({ plugins: [plugin('running')] });
    await act(async () => { snapshot(frame, 'old'); snapshot(frame, 'new'); });
    expect(container.querySelector('[role="alert"]')).toBeNull();
    await act(async () => resolveOld({ plugins: [plugin('unhealthy', 'outdated probe')] }));
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.querySelector('iframe')).toBe(frame);
    expect(assetFetches()).toBe(1);
  });

  it('says a plugin that is not running in words, with the button that starts it, not a trip to Plugins', async () => {
    list.mockResolvedValue({ plugins: [plugin('installed', 'Not started yet.')] });
    await mount();
    const note = container.querySelector('.setup-run-note')!;
    expect(note.textContent).toContain('Aokie is not running. This screen works once it runs.');
    expect(note.textContent).not.toContain('installed');
    expect(note.textContent).not.toContain('..');
    expect(note.textContent).not.toContain('from Plugins');
    list.mockResolvedValue({ plugins: [plugin('running')] });
    await act(async () => note.querySelector<HTMLButtonElement>('button')!.click());
    expect(start).toHaveBeenCalledWith('aokie');
    // (read again at once: the note goes as it runs)
    expect(container.querySelector('.setup-run-note')).toBeNull();
  });

  it('loads the screen again from the start on a Refresh, so a message it still showed is gone', async () => {
    list.mockResolvedValue({ plugins: [plugin('running')] });
    const frame = await mount();
    const nonce = (f: HTMLIFrameElement) => f.srcdoc.match(/window\.__oaiyDocumentNonce=("[^"]+")/)![1];
    const first = nonce(frame);
    frame.contentDocument!.body.innerHTML = '<p id="stale-error">The radio did not answer.</p>';
    await act(async () => root.render(<PluginScreenPage pluginId="aokie" navId="calls" reload={1} />));
    const fresh = container.querySelector('iframe')!;
    expect(fresh).not.toBe(frame);
    expect(nonce(fresh)).not.toBe(first);
    expect(fresh.contentDocument!.getElementById('stale-error')).toBeNull();
    // (the screen's files read again, as opening it does)
    expect(assetFetches()).toBe(2);
    // The same value again is no reload.
    await act(async () => root.render(<PluginScreenPage pluginId="aokie" navId="calls" reload={1} />));
    expect(container.querySelector('iframe')).toBe(fresh);
    expect(assetFetches()).toBe(2);
  });

  it('offers to trust a plugin nobody signed, as the one thing that gets it running', async () => {
    list.mockResolvedValue({ plugins: [{ ...plugin('stopped', 'Not started. Not signed, and you have not trusted it yet.'), trust: { state: 'unsigned', reason: 'Not signed, and you have not trusted it yet.' } }] });
    ask.mockResolvedValue(true);
    await mount();
    const note = container.querySelector('.setup-run-note')!;
    expect(note.textContent).toContain('Aokie is not signed, and you have not trusted it yet, so it does not run.');
    const button = note.querySelector<HTMLButtonElement>('button')!;
    expect(button.textContent).toContain('Trust this plugin');
    await act(async () => button.click());
    expect(trust).toHaveBeenCalledWith('aokie');
    expect(start).toHaveBeenCalledWith('aokie');
  });
});
