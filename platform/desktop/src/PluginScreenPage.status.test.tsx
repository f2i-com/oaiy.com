import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { list, toast } = vi.hoisted(() => ({ list: vi.fn(), toast: { push: vi.fn() } }));
vi.mock('./api', () => ({
  API_BASE: 'http://127.0.0.1:17972',
  plugins: { list }, bridge: {}, companion: {},
}));
vi.mock('./Toasts', () => ({ useToast: () => toast }));

import PluginScreenPage from './PluginScreenPage';

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

async function mount() {
  await act(async () => root.render(<PluginScreenPage pluginId="aokie" navId="calls" />));
  return container.querySelector('iframe')!;
}

function snapshot(frame: HTMLIFrameElement, id: string) {
  window.dispatchEvent(new MessageEvent('message', {
    source: frame.contentWindow,
    data: { __pluginHost: 1, id, method: 'snapshot', args: [] },
  }));
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
    expect(fetchAssets).toHaveBeenCalledTimes(1);
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
    expect(fetchAssets).toHaveBeenCalledTimes(1);
  });
});
