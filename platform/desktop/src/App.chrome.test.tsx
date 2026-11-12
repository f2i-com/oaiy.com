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
// (pages the frame is looked at around: their own reads are not this file's)
vi.mock('./ServicesPanel', () => ({ default: () => <p>The Services page</p> }));
vi.mock('./ModelsPanel', () => ({ default: () => <p>The model files page</p> }));
vi.mock('./RunsPanel', () => ({ default: () => <p>Run history</p> }));
vi.mock('./SettingsPanel', () => ({ default: () => <p>General settings</p>, BackupSettings: () => <p>Backup settings</p>, StorageSettings: () => <p>Storage settings</p> }));
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
const title = () => host.querySelector('.top-heading h1')?.textContent;

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
    expect(host.querySelector('.top-copy')?.textContent).toBe('Choose your AI and what the Agent may do.');
    expect(host.querySelectorAll('h1')).toHaveLength(1);
    expect(host.querySelector('.setup-hero h1')).toBeNull();
    // The step's "Step 2 of 4" is the count, and the step list shows what is done: no bar over the step, and no row
    // of its own for "Continue later" (it is in the step's footer).
    expect(host.textContent).not.toMatch(/\d of \d done/);
    expect(host.querySelector('.setup-hero')).toBeNull();
    expect(host.querySelector('.setup-page > .setup-layout')).not.toBeNull();
    expect(host.querySelector('.setup-foot .setup-leave')?.textContent).toBe('Continue later');
    // (the setup page keeps the full header: it is not one of the sections with a page in a webview)
    expect(host.querySelector('.topbar')!.classList.contains('is-compact')).toBe(false);
  });

  it('gives a section with a page in a webview a one-line header: its name, its tabs beside it, no group and no line under', async () => {
    for (const view of ['agent', 'flows', 'engines']) {
      await go(view);
      const bar = host.querySelector('.topbar')!;
      expect(bar.classList.contains('is-compact')).toBe(true);
      expect(host.querySelector('.workspace')!.classList.contains('is-compact')).toBe(true);
      expect(bar.querySelector('.top-kicker')).toBeNull();
      expect(bar.querySelector('.top-copy')).toBeNull();
      // What the page is for is still said: the name's tooltip, and its description for a screen reader.
      expect(bar.querySelector('h1')!.getAttribute('title')).toBeTruthy();
      expect(bar.querySelector('h1')!.getAttribute('aria-description')).toBe(bar.querySelector('h1')!.getAttribute('title'));
    }
    // Engines: its tabs are on the header's own line, not in a strip under it.
    expect(title()).toBe('Engines');
    const tabs = Array.from(host.querySelectorAll('.topbar [role=tab]'));
    expect(tabs.map((t) => t.textContent)).toEqual(['This computer', 'Other computers', 'Model files']);
    expect(host.querySelector('.view .section-tabs')).toBeNull();
    // Each tab is named whatever is drawn of it (a narrow bar shows the open tab's word and the others' icons).
    expect(tabs.map((t) => t.getAttribute('aria-label'))).toEqual(['This computer', 'Other computers', 'Model files']);
    // Its other pages keep the one line too: the tabs stay where they are from one to the next.
    await go('models');
    expect(host.querySelector('.topbar')!.classList.contains('is-compact')).toBe(true);
    expect(host.querySelector('.topbar [role=tab][aria-selected=true]')!.textContent).toBe('Model files');
    await go('runs');
    expect(title()).toBe('Flows');
    expect(host.querySelector('.topbar')!.classList.contains('is-compact')).toBe(true);
    // A page of the dashboard's own, in a section with none in a webview: the full header, its three lines (the
    // group, the name with the section's tabs beside it, and what the page is for), and no strip of tabs under it.
    await go('services');
    const full = host.querySelector('.topbar')!;
    expect(full.classList.contains('is-compact')).toBe(false);
    expect(Array.from(full.children).map((c) => c.className)).toEqual(['top-row', 'top-copy']);
    expect(Array.from(full.querySelector('.top-row')!.children).map((c) => c.className)).toEqual(['top-heading', 'top-actions']);
    expect(full.querySelector('.top-kicker')).toBeNull();
    expect(full.querySelector('.top-heading > h1')!.textContent).toBe('Services');
    expect(Array.from(full.querySelectorAll('.top-heading [role=tab]')).map((t) => t.textContent)).toEqual(['Services', 'Python']);
    expect(full.querySelector('.top-copy')!.textContent).toBe('Start, stop and install the local AI services this computer offers OAIY.');
    expect(host.querySelector('.view .section-tabs')).toBeNull();
    // (the line is on the page, so it is not said a second time as the name's description)
    expect(full.querySelector('h1')!.getAttribute('aria-description')).toBeNull();
  });

  it('has no bar at the foot: whether OAIY answers is at the foot of the sidebar, above Settings, and opens Services', async () => {
    expect(host.querySelector('.endpoint-dock, footer')).toBeNull();
    expect(host.textContent).not.toContain('http://127.0.0.1:17972');
    expect(host.textContent).not.toContain('Keys stay on this computer');
    const status = host.querySelector<HTMLButtonElement>('.sidebar .api-status')!;
    expect(status.textContent).toBe('Running');
    expect(status.classList.contains('is-up')).toBe(true);
    expect(status.getAttribute('aria-label')).toBe('OAIY is running. Open Services.');
    // Above Settings, after the sections.
    expect(status.nextElementSibling).toBe(host.querySelector('.sidebar > .settings'));
    await act(async () => status.click());
    await settle();
    expect(title()).toBe('Services');
  });

  it('splits Settings by what a person comes for: General, Backup, Storage, Agent', async () => {
    await go('settings');
    expect(title()).toBe('Settings');
    expect(Array.from(host.querySelectorAll('[role=tab]')).map((t) => t.textContent)).toEqual(['General', 'Backup', 'Storage', 'Agent']);
    expect(host.querySelector('[role=tab][aria-selected=true]')!.textContent).toBe('General');
    await go('storage');
    expect(host.querySelector('[role=tab][aria-selected=true]')!.textContent).toBe('Storage');
    await go('backup');
    expect(host.querySelector('[role=tab][aria-selected=true]')!.textContent).toBe('Backup');
  });

  it('names the theme button for what it does, in plain words', async () => {
    const button = () => host.querySelector<HTMLButtonElement>('.top-actions .icon-button')!;
    const was = button().getAttribute('aria-label')!;
    expect(['Switch to the light theme', 'Switch to the dark theme']).toContain(was);
    expect(['Light theme', 'Dark theme']).toContain(button().title);
    await act(async () => button().click());
    expect(button().getAttribute('aria-label')).toBe(was === 'Switch to the light theme' ? 'Switch to the dark theme' : 'Switch to the light theme');
    // (the design system's own names for the two themes are not a person's words)
    expect(host.innerHTML).not.toMatch(/Paper Circuit|Prism Lab/);
  });
});

describe('the dashboard’s frame while OAIY is not answering', () => {
  it('says so at the foot of the sidebar, and not a second time in the top bar', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    await act(async () => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    const status = host.querySelector<HTMLButtonElement>('.sidebar .api-status')!;
    expect(status.textContent).toBe('Not answering');
    expect(status.classList.contains('is-down')).toBe(true);
    expect(status.getAttribute('aria-label')).toBe('OAIY is not answering. Open Services.');
    // The page says it once, with its own "Try again" (the Overview's banner): the top bar has no chip for it.
    expect(host.querySelector('.top-alert')).toBeNull();
    expect(host.querySelector('.topbar')!.textContent).not.toMatch(/unreachable|not answering/i);
  });

  it('in the narrow sidebar, folds a sub-menu into the header’s tabs: the right arrow opens its first page and steps onto its tab', async () => {
    // The window is narrow enough for the sidebar to be icons alone (jsdom has no media queries of its own).
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === '(max-width: 1240px)', media: query, addEventListener() {}, removeEventListener() {} }));
    await act(async () => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    // jsdom lays nothing out. The narrow sidebar as sidebar.css draws it: no chevron, and none of a sub-menu's pages.
    const rects = vi.spyOn(HTMLElement.prototype, 'getClientRects').mockImplementation(function (this: HTMLElement) {
      const hidden = this.classList.contains('nav-toggle') || !!this.closest('.nav-children');
      return (hidden ? [] : [{}]) as unknown as DOMRectList;
    });
    try {
      // On a page outside the sub-menu the header has no tabs of it.
      expect(host.querySelector('.topbar .seg-tabs.is-folded')).toBeNull();
      const parent = host.querySelector<HTMLButtonElement>('nav .nav-parent')!;
      parent.focus();
      await act(async () => parent.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
      await settle();
      for (let i = 0; i < 3; i++) await act(async () => new Promise<void>((r) => requestAnimationFrame(() => r())));
      // The section is the name now, and its pages are the tabs beside it (the first is open).
      expect(title()).toBe('AI Receptionist');
      expect(host.textContent).toContain('Plugin screen receptionist');
      const tabs = Array.from(host.querySelectorAll('.topbar .seg-tabs.is-folded [role=tab]'));
      expect(tabs.length).toBeGreaterThan(1);
      expect(tabs[0].getAttribute('aria-selected')).toBe('true');
      expect(tabs.map((t) => t.getAttribute('aria-label')!.split(',')[0])).toContain('Transfers');
      expect(document.activeElement).toBe(tabs[0]);
      // (the group over the name is the sidebar's, as for any section: the section itself is the name)
      expect(host.querySelector('.top-kicker')).toBeNull();
    } finally {
      rects.mockRestore();
    }
  });
});
