import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const h = vi.hoisted(() => ({
  navigate: null as null | ((target: { kind: 'view'; view: string }) => void),
  openSetup: null as null | ((target: { plugin?: string }) => void),
  carryGuideDismissal: vi.fn().mockResolvedValue(false),
  waiting: vi.fn(() => []),
  unread: vi.fn(() => 0),
  embedded: vi.fn(() => null),
  overview: vi.fn(() => null),
  setup: vi.fn(() => null),
  pairing: vi.fn(() => null),
  ring: vi.fn((_props: { on: boolean }) => null),
  phoneCalls: vi.fn().mockResolvedValue([]),
}));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    API_BASE: 'http://127.0.0.1:48123',
    DESKTOP_LAUNCH: Object.freeze({ apiBase: 'http://127.0.0.1:48123', isolated: true }),
    phone: { ...real.phone, calls: h.phoneCalls },
  };
});
vi.mock('./navigate', () => ({ onNavigate: (cb: typeof h.navigate) => { h.navigate = cb; return () => undefined; } }));
vi.mock('./useSetupState', () => ({
  // An unfinished first run must not auto-open the live setup wizard.
  useSetupState: () => ({ firstRun: { finished: false, skipped: [], chosenPlugins: [] }, plugins: {} }),
  onOpenSetup: (cb: typeof h.openSetup) => { h.openSetup = cb; return () => undefined; },
  openSetup: vi.fn(),
  carryGuideDismissal: h.carryGuideDismissal,
  guideDismissed: () => false,
}));
vi.mock('./useModules', () => ({
  useModules: () => ({
    modules: [{ id: 'phone', enabled: true }, { id: 'calendar', enabled: true }],
    contributions: { sections: [{
      id: 'plugin:example:editor', builtin: false, label: 'Example', group: 'Work', pluginId: 'example', pluginName: 'Example',
      pages: [{ view: 'plugin:example:editor', pluginId: 'example', pluginName: 'Example', navId: 'editor', label: 'Editor', screen: 'editor' }],
    }] },
  }),
  moduleOn: () => true,
  refetchModules: vi.fn(),
}));
vi.mock('./calendarRequests', () => ({ useWaitingRequests: h.waiting }));
vi.mock('./unreadMessages', () => ({ useUnreadMessages: h.unread }));
vi.mock('./EmbeddedPage', () => ({ default: h.embedded }));
vi.mock('./OverviewPanel', () => ({ default: h.overview }));
vi.mock('./SetupPage', () => ({ default: h.setup }));
vi.mock('./PairingPrompt', () => ({ default: h.pairing }));
vi.mock('./RingDialog', () => ({ default: h.ring }));
vi.mock('./PluginsPanel', () => ({ default: () => <p>Plugin manager</p> }));
vi.mock('./PluginScreenPage', () => ({ default: ({ navId }: { navId: string }) => <p>Plugin screen {navId}</p> }));

import App from './App';

let host: HTMLDivElement;
let root: Root;
let fetch: ReturnType<typeof vi.fn>;

beforeEach(async () => {
  vi.clearAllMocks();
  window.localStorage.clear();
  fetch = vi.fn(async () => new Response(JSON.stringify({ status: 'ok', product: 'oaiy-desktop', version: '0.1.0' }), { status: 200 }));
  vi.stubGlobal('fetch', fetch);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => root.render(<App />));
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('isolated local qualification navigation', () => {
  it('starts on Plugins, announces its scope, and does not open unfinished setup', () => {
    expect(host.querySelector('.view-plugins')).not.toBeNull();
    expect(host.textContent).toContain('Plugin manager');
    expect(host.querySelector('[role="status"]')?.textContent).toContain('Isolated local qualification.');
    expect(host.querySelector('[role="status"]')?.textContent).toContain('Only plugins and their screens are available.');
    expect(h.overview).not.toHaveBeenCalled();
    expect(h.setup).not.toHaveBeenCalled();
    expect(h.carryGuideDismissal).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:48123/api/health');
  });

  it('does not activate account, calendar, phone or message background UI', () => {
    expect(h.pairing).not.toHaveBeenCalled();
    expect(h.waiting).toHaveBeenCalledWith(false);
    expect(h.unread).toHaveBeenCalledWith(false);
    expect(h.ring.mock.calls.every(([props]) => props.on === false)).toBe(true);
    expect(h.phoneCalls).not.toHaveBeenCalled();
  });

  it.each(['agent', 'flows', 'engines', 'overview', 'connections', 'providers', 'settings', 'setup', 'agent-settings', 'services', 'runs', 'models', 'python'])(
    'refuses the %s page without mounting live integration panels',
    async (view) => {
      await act(async () => h.navigate!({ kind: 'view', view }));
      expect(host.textContent).toContain('This page is unavailable during isolated local qualification.');
      expect(h.embedded).not.toHaveBeenCalled();
      expect(h.overview).not.toHaveBeenCalled();
      expect(h.setup).not.toHaveBeenCalled();
      expect(fetch.mock.calls.every(([url]) => url === 'http://127.0.0.1:48123/api/health')).toBe(true);
      const back = [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'Open Plugins')!;
      await act(async () => back.click());
      expect(host.textContent).toContain('Plugin manager');
    },
  );

  it('refuses a plugin setup request while keeping contributed screens available', async () => {
    await act(async () => h.openSetup!({ plugin: 'example' }));
    expect(host.textContent).toContain('This page is unavailable during isolated local qualification.');
    expect(h.setup).not.toHaveBeenCalled();
    await act(async () => h.navigate!({ kind: 'view', view: 'plugin:example:editor' }));
    expect(host.textContent).toContain('Plugin screen editor');
    expect(host.textContent).not.toContain('This page is unavailable during isolated local qualification.');
  });
});
