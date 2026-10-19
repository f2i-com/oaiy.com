// The dashboard's own frame around a page: a request to connect asked once (not again over the Connections page,
// which lists it under "Waiting for you"); the setup page under one heading, the wizard's, in the top bar; and the
// narrow sidebar's right arrow, which opens a sub-menu's first page (its pages show there only while one is open).
//
// Same convention as the other tests: react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModulesSnapshot } from './api';

const h = vi.hoisted(() => ({
  navigate: null as null | ((target: { kind: 'view'; view: string } | { kind: 'setup' }) => void),
  modulesList: vi.fn(),
}));

vi.mock('./navigate', () => ({
  onNavigate: (cb: typeof h.navigate) => {
    h.navigate = cb;
    return () => undefined;
  },
}));
vi.mock('./useSetupState', () => ({
  useSetupState: () => null,
  onOpenSetup: () => () => undefined,
  openSetup: vi.fn(),
  carryGuideDismissal: vi.fn().mockResolvedValue(false),
  guideDismissed: () => true,
}));
vi.mock('./EmbeddedPage', () => ({ default: () => null }));
vi.mock('./PairingPrompt', () => ({ default: () => <p className="app-pairing-prompt">An app wants to connect</p> }));
vi.mock('./ConnectionsPanel', () => ({ default: () => <p>The Connections page</p> }));
vi.mock('./OverviewPanel', () => ({ default: () => <p>The Overview</p> }));
vi.mock('./PluginScreenPage', () => ({ default: ({ navId }: { navId: string }) => <p>Plugin screen {navId}</p> }));
// The wizard as far as the frame goes: its heading and progress, given to the frame.
vi.mock('./SetupPage', async () => {
  const { WizardFrame } = await import('./SetupParts');
  return {
    default: () => (
      <WizardFrame title="Set up OAIY" subtitle="Choose your AI and what the Agent may do." progress={{ done: 0, total: 2 }} rail={[]} current={0} railLabel="Setup steps" onLeave={() => undefined}>
        <span className="setup-kicker">Step 2 of 4</span>
      </WizardFrame>
    ),
  };
});
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    modules: { list: (...a: unknown[]) => h.modulesList(...a) },
    phone: { ...real.phone, calls: vi.fn().mockResolvedValue([]) },
    calendar: { ...real.calendar, get: vi.fn().mockResolvedValue({ settings: null, appointments: [], now: '2026-10-19T09:00' }) },
    messages: { ...real.messages, list: vi.fn().mockResolvedValue({ messages: [], total: 0, unread: 0 }) },
    network: { ...real.network, status: vi.fn().mockRejectedValue(new Error('off')) },
  };
});

import App from './App';
import { ToastProvider } from './Toasts';
import { resetModules } from './useModules';
import { resetRequests } from './calendarRequests';

const aokie = { pluginId: 'aokie', name: 'Aokie Phone Bridge', state: 'running' as const, declared: true };
const receptionist = {
  view: 'plugin:aokie:receptionist',
  pluginId: 'aokie',
  pluginName: 'Aokie Phone Bridge',
  navId: 'receptionist',
  label: 'AI Receptionist',
  screen: 'receptionist-home',
  icon: 'phone',
  module: 'phone',
};
const snapshot: { snapshot: ModulesSnapshot; etag: string } = {
  snapshot: {
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: true, builtin: true, provider: aokie },
      { id: 'calendar', name: 'Calendar', enabled: false, builtin: true, provider: aokie },
    ],
    contributions: {
      sections: [{ id: receptionist.view, builtin: false, pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', label: 'AI Receptionist', icon: 'phone', group: 'Work' as const, module: 'phone', pages: [receptionist] }],
    },
    warnings: [],
  },
  etag: '"1"',
};

let host: HTMLDivElement;
let root: Root;
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
async function go(view: string) {
  await act(async () => h.navigate!({ kind: 'view', view }));
  await settle();
}
const title = () => host.querySelector('.top-title h1')?.textContent;

beforeEach(async () => {
  resetModules();
  resetRequests();
  window.localStorage.clear();
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ status: 'ok', product: 'oaiy-desktop', version: '0.1.0' }), { status: 200 })));
  h.modulesList.mockResolvedValue(snapshot);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root.render(
      <ToastProvider>
        <App />
      </ToastProvider>,
    );
  });
  await settle();
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('the dashboard’s frame', () => {
  it('asks to connect an app wherever the person is, but not over the Connections page, which lists the request itself', async () => {
    expect(host.querySelector('.app-pairing-prompt')).not.toBeNull();
    await go('connections');
    expect(host.textContent).toContain('The Connections page');
    expect(host.querySelector('.app-pairing-prompt')).toBeNull();
    await go('overview');
    expect(host.querySelector('.app-pairing-prompt')).not.toBeNull();
  });

  it('heads the setup page once, with the wizard’s own heading in the top bar, and one count', async () => {
    await go('setup');
    expect(title()).toBe('Set up OAIY');
    expect(host.querySelector('.top-kicker')).toBeNull();
    expect(host.querySelector('.top-title p')?.textContent).toBe('Choose your AI and what the Agent may do.');
    expect(host.querySelectorAll('h1')).toHaveLength(1);
    expect(host.querySelector('.setup-hero h1')).toBeNull();
    // The step's "Step 2 of 4" is the count; the bar beside "Continue later" says the rest to a screen reader only.
    expect(host.textContent).not.toMatch(/\d of \d done/);
    expect(host.querySelector('.setup-progress')?.getAttribute('aria-valuetext')).toBe('0 of 2 done');
  });

  it('in the narrow sidebar, opens a sub-menu’s first page on the right arrow, and steps onto it', async () => {
    // jsdom lays nothing out. The narrow sidebar as calendar.css draws it: no chevron, and a sub-menu's pages only
    // while one of them is open.
    const rects = vi.spyOn(HTMLElement.prototype, 'getClientRects').mockImplementation(function (this: HTMLElement) {
      const hidden = this.classList.contains('nav-toggle') || !!this.closest('.nav-sub:not(.has-active) .nav-children');
      return (hidden ? [] : [{}]) as unknown as DOMRectList;
    });
    try {
      const parent = host.querySelector<HTMLButtonElement>('nav .nav-parent')!;
      parent.focus();
      await act(async () => parent.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
      await settle();
      for (let i = 0; i < 3; i++) await act(async () => new Promise<void>((r) => requestAnimationFrame(() => r())));
      expect(title()).toBe('Phone');
      expect(host.textContent).toContain('Plugin screen receptionist');
      expect(document.activeElement?.classList.contains('nav-child')).toBe(true);
      expect(document.activeElement?.textContent).toContain('Phone');
    } finally {
      rects.mockRestore();
    }
  });
});
