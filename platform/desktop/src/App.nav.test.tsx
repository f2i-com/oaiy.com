// The sidebar's sub-menu and the pages' ids, in the whole dashboard: while
// Aokie provides the calendar, Calendar and Hours & Services are under the AI
// Receptionist; the ids anything may open them by (`calendar`, `hours`, from
// the desktop's oaiy://navigate, a plugin screen, the setup wizard) land on
// them; the sub-menu opens and closes, remembered for this viewer; and the
// requests waiting show as a count. Same convention as the other tests:
// react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Appointment, ModulesSnapshot } from './api';

const h = vi.hoisted(() => ({
  navigate: null as null | ((target: { kind: 'view'; view: string; contact?: string } | { kind: 'setup' }) => void),
  modulesList: vi.fn(),
  calendarGet: vi.fn(),
  contactsList: vi.fn(),
  messagesList: vi.fn(),
  ringSettings: vi.fn(),
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
vi.mock('./PairingPrompt', () => ({ default: () => null }));
vi.mock('./OverviewPanel', () => ({ default: () => <p>The Overview</p> }));
vi.mock('./PluginScreenPage', () => ({ default: ({ navId }: { navId: string }) => <p>Plugin screen {navId}</p> }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    modules: { list: (...a: unknown[]) => h.modulesList(...a) },
    phone: { ...real.phone, calls: vi.fn().mockResolvedValue([]) },
    calendar: {
      ...real.calendar,
      get: (...a: unknown[]) => h.calendarGet(...a),
      syncStatus: vi.fn().mockResolvedValue({ linked: false, at: null, pulled: 0, pushed: 0, error: null }),
      free: vi.fn().mockResolvedValue({ minutes: 60, days: [] }),
    },
    voices: { ...real.voices, list: vi.fn().mockResolvedValue({ voices: [], chosen: null }) },
    contacts: { ...real.contacts, list: (...a: unknown[]) => h.contactsList(...a) },
    messages: { ...real.messages, list: (...a: unknown[]) => h.messagesList(...a) },
    ring: { ...real.ring, settings: (...a: unknown[]) => h.ringSettings(...a) },
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
const snapshot = (phone: boolean, calendar: boolean): { snapshot: ModulesSnapshot; etag: string } => ({
  snapshot: {
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: phone, builtin: true, provider: aokie },
      { id: 'calendar', name: 'Calendar', enabled: calendar, builtin: true, provider: aokie },
    ],
    contributions: {
      // The desktop leaves out Aokie's page while the phone is off.
      sections: phone
        ? [{ id: receptionist.view, builtin: false, pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', label: 'AI Receptionist', icon: 'phone', group: 'Work' as const, module: 'phone', pages: [receptionist] }]
        : [],
    },
    warnings: [],
  },
  etag: `"${phone}-${calendar}"`,
});

const request = (id: string, start: string): Appointment => ({
  id,
  service: 'Lawn mowing',
  start,
  minutes: 60,
  status: 'requested',
  name: 'Lanes',
  phone: '0412 345 678',
  notes: '',
  source: 'call',
  createdAt: '2026-09-29T00:00:00Z',
  updatedAt: '2026-09-29T00:00:00Z',
});
const SETTINGS = {
  business: 'Green Lawns',
  hours: [[{ open: '08:00', close: '17:00' }], [{ open: '08:00', close: '17:00' }], [], [], [], [], []],
  services: [{ id: 'mow', name: 'Lawn mowing', minutes: 60 }],
  slotMinutes: 30,
  noticeMinutes: 60,
  horizonDays: 30,
  textConfirmations: true,
};

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';
const title = () => host.querySelector('.top-title h1')?.textContent;
const kicker = () => host.querySelector('.top-kicker')?.textContent;
const nav = () => host.querySelector('nav[aria-label="Primary"]')!;
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
async function go(view: string) {
  await act(async () => h.navigate!({ kind: 'view', view }));
  await settle();
}

beforeEach(async () => {
  vi.clearAllMocks();
  resetModules();
  resetRequests();
  window.localStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ status: 'ok', product: 'oaiy-desktop', version: '0.1.0' }), { status: 200 })),
  );
  h.modulesList.mockResolvedValue(snapshot(true, true));
  h.contactsList.mockResolvedValue({
    contacts: [
      { key: '491570006', number: '+61491570006', name: 'Liam', nameBy: 'owner', notes: '', facts: [], createdAt: '2026-09-29T00:00:00Z', updatedAt: '2026-09-29T00:00:00Z' },
      { key: '400000001', number: '', name: 'Sam', nameBy: 'agent', notes: '', facts: [], createdAt: '2026-09-29T00:00:00Z', updatedAt: '2026-09-29T00:00:00Z' },
    ],
    total: 2,
  });
  h.messagesList.mockResolvedValue({ messages: [], total: 0, unread: 0 });
  h.ringSettings.mockResolvedValue({
    settings: {
      enabled: false,
      takeMessages: false,
      initiative: 'on_request',
      urgentPhrases: [],
      ringSeconds: 40,
      phoneRing: 'when_away',
      desktopRing: 'auto',
      away: 'auto',
      awayUntil: null,
      desktopActiveSeconds: 120,
      quietHours: { enabled: false, start: '21:00', end: '07:00', days: 127, allowUrgent: false, allowVip: true },
      vipNumbers: [],
      limits: { perCall: 2, gapSeconds: 60, perCallerHour: 3, globalHour: 10 },
      windowsCompanions: [],
      excludedDevices: [],
    },
    features: { transfer: false, messages: false },
  });
  h.calendarGet.mockResolvedValue({
    settings: SETTINGS,
    appointments: [request('r1', '2026-10-01T10:00'), request('r2', '2026-10-02T09:00')],
    now: '2026-09-29T09:00',
  });
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

describe('the AI Receptionist’s sub-menu', () => {
  it('holds its own page, the Calendar, Contacts and Hours & Services, with the requests waiting counted', () => {
    const sub = nav().querySelector('.nav-sub')!;
    expect(sub.querySelector('.nav-parent')?.getAttribute('aria-label')).toBe('AI Receptionist');
    const children = [...sub.querySelectorAll('.nav-child')].map((b) => b.querySelector('span')?.textContent);
    expect(children).toEqual(['Phone', 'Calendar', 'Contacts', 'Messages', 'Hours & Services', 'Transfers']);
    // No Calendar or Contacts entry of their own any more.
    expect(nav().querySelector('button[aria-label="Calendar"]')).toBeNull();
    expect(nav().querySelector('button[aria-label="Contacts"]:not(.nav-child)')).toBeNull();
    const calendarLink = [...sub.querySelectorAll('.nav-child')].find((b) => b.textContent?.startsWith('Calendar'))!;
    expect(calendarLink.querySelector('.nav-count')?.textContent).toBe('2');
    expect(calendarLink.getAttribute('aria-label')).toBe('Calendar, 2 requests waiting');
  });

  it('counts the messages nobody has looked at, and lands `messages` and `transfers` on their pages under the AI Receptionist', async () => {
    const left = (id: string) => ({ id, at: '2026-09-30T02:15:03Z', callId: 'c', from: '+61491570006', name: 'Alex', callback: '+61491570006', message: 'Ring me.', urgency: 'normal', wantsCallback: true, state: 'new', seenAt: null, handledAt: null, handledBy: null });
    h.messagesList.mockResolvedValue({ messages: [left('m1'), left('m2')], total: 2, unread: 2 });
    resetModules();
    act(() => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    const sub = nav().querySelector('.nav-sub')!;
    const link = [...sub.querySelectorAll('.nav-child')].find((b) => b.textContent?.startsWith('Messages'))!;
    expect(link.querySelector('.nav-count')?.textContent).toBe('2');
    expect(link.getAttribute('aria-label')).toBe('Messages, 2 new messages');
    // With the two requests waiting, the parent adds both up.
    expect(sub.querySelector('.nav-count-parent')?.textContent).toBe('4');
    expect(sub.querySelector('.nav-count-parent')?.getAttribute('title')).toBe('4 waiting for you');

    await go('messages');
    expect(title()).toBe('Messages');
    expect(kicker()).toBe('AI Receptionist');
    expect(host.querySelector('.messages-page')).not.toBeNull();
    await go('transfers');
    expect(title()).toBe('Transfers');
    expect(kicker()).toBe('AI Receptionist');
    expect(text()).toContain('Transfer calls to me');
    expect(text()).toContain('Take messages');
  });

  it('lands the old id `calendar`, and `hours`, on their pages, with the AI Receptionist over them', async () => {
    await go('calendar');
    expect(title()).toBe('Calendar');
    expect(kicker()).toBe('AI Receptionist');
    expect(host.querySelector('.cal-page')).not.toBeNull();
    // The sub-menu is in the sidebar: no tabs under the header too.
    expect(host.querySelector('.section-tabs')).toBeNull();
    const current = nav().querySelector('[aria-current="page"]');
    expect(current?.textContent).toContain('Calendar');

    await go('hours');
    expect(title()).toBe('Hours & Services');
    expect(kicker()).toBe('AI Receptionist');
    expect(host.querySelector('.hours-form')).not.toBeNull();
    expect(text()).toContain('Opening hours');
    expect(nav().querySelector('[aria-current="page"]')?.textContent).toBe('Hours & Services');

    await go('plugin:aokie:receptionist');
    expect(title()).toBe('Phone');
    expect(text()).toContain('Plugin screen receptionist');
  });

  it('lands `contacts` on Contacts, and the Agent’s contact key opens that person', async () => {
    await go('contacts');
    expect(title()).toBe('Contacts');
    expect(kicker()).toBe('AI Receptionist');
    expect(nav().querySelector('[aria-current="page"]')?.textContent).toBe('Contacts');
    expect([...host.querySelectorAll('.contact-row strong')].map((s) => s.textContent)).toEqual(['Liam', 'Sam']);
    expect(host.querySelector('.contacts-side')).toBeNull();

    await go('overview');
    await act(async () => h.navigate!({ kind: 'view', view: 'contacts', contact: '400000001' }));
    await settle();
    expect(title()).toBe('Contacts');
    expect(host.querySelector('#contacts-side-title')?.textContent).toBe('Sam');
    expect(host.querySelector<HTMLInputElement>('.contact-name input')?.value).toBe('Sam');
  });

  it('opens a link in it directly, and the entry opens the page last open in it', async () => {
    const child = (label: string) => [...nav().querySelectorAll<HTMLButtonElement>('.nav-child')].find((b) => b.textContent?.startsWith(label))!;
    await act(async () => child('Hours & Services').click());
    await settle();
    expect(title()).toBe('Hours & Services');
    await go('overview');
    expect(text()).toContain('The Overview');
    await act(async () => nav().querySelector<HTMLButtonElement>('.nav-parent')!.click());
    await settle();
    expect(title()).toBe('Hours & Services');
  });

  it('closes and opens with its chevron, remembered for this viewer; closed, the count moves to the entry', async () => {
    const sub = () => nav().querySelector('.nav-sub')!;
    const toggle = () => sub().querySelector<HTMLButtonElement>('.nav-toggle')!;
    expect(sub().classList.contains('is-expanded')).toBe(true);
    expect(toggle().getAttribute('aria-expanded')).toBe('true');
    await act(async () => toggle().click());
    expect(sub().classList.contains('is-expanded')).toBe(false);
    expect(toggle().getAttribute('aria-expanded')).toBe('false');
    expect(JSON.parse(window.localStorage.getItem('oaiy.navExpanded')!)).toEqual({ 'plugin:aokie:receptionist': false });
    expect(sub().querySelector('.nav-count-parent')?.textContent).toBe('2');
    // Opening one of its pages from elsewhere opens it again.
    await go('calendar');
    expect(sub().classList.contains('is-expanded')).toBe(true);
  });

  it('works from the keyboard: up and down between entries, right to open and step in, left to step out and close', async () => {
    // jsdom lays nothing out: an entry counts as shown unless it is in a closed sub-menu.
    const rects = vi.spyOn(HTMLElement.prototype, 'getClientRects').mockImplementation(function (this: HTMLElement) {
      const hidden = !!this.closest('.nav-sub:not(.is-expanded) .nav-children');
      return (hidden ? [] : [{}]) as unknown as DOMRectList;
    });
    const key = async (el: Element, k: string) => {
      await act(async () => el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true })));
      await act(async () => new Promise<void>((r) => requestAnimationFrame(() => r())));
    };
    const parent = nav().querySelector<HTMLButtonElement>('.nav-parent')!;
    const flows = nav().querySelector<HTMLButtonElement>('button[aria-label="Flows"]')!;
    flows.focus();
    await key(flows, 'ArrowDown');
    expect(document.activeElement).toBe(parent);
    await key(parent, 'ArrowLeft');
    expect(nav().querySelector('.nav-sub')!.classList.contains('is-expanded')).toBe(false);
    // Closed, down skips its pages.
    await key(parent, 'ArrowDown');
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Engines');
    parent.focus();
    await key(parent, 'ArrowRight');
    expect(nav().querySelector('.nav-sub')!.classList.contains('is-expanded')).toBe(true);
    await key(parent, 'ArrowRight');
    expect(document.activeElement?.textContent).toBe('Phone');
    await key(document.activeElement!, 'ArrowDown');
    expect(document.activeElement?.textContent).toContain('Calendar');
    await key(document.activeElement!, 'ArrowLeft');
    expect(document.activeElement).toBe(parent);
    rects.mockRestore();
  });

  it('keeps working when the stored state cannot be read', async () => {
    window.localStorage.setItem('oaiy.navExpanded', '{not json');
    act(() => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    expect(nav().querySelector('.nav-sub')?.classList.contains('is-expanded')).toBe(true);
  });
});

describe('without the AI Receptionist', () => {
  it('keeps the Calendar top-level, with Calendar and Hours & Services as its tabs', async () => {
    h.modulesList.mockResolvedValue(snapshot(false, true));
    resetModules();
    act(() => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    expect(nav().querySelector('.nav-sub')).toBeNull();
    expect(nav().querySelector('button[aria-label="Calendar"]')).not.toBeNull();
    await go('hours');
    expect(title()).toBe('Calendar');
    expect([...host.querySelectorAll('.section-tabs [role="tab"]')].map((t) => t.textContent)).toEqual(['Calendar', 'Hours & Services']);
    expect(host.querySelector('.hours-form')).not.toBeNull();
  });

  it('hides the Calendar’s pages while the calendar is off, and an open one goes to the Overview', async () => {
    h.modulesList.mockResolvedValue(snapshot(true, false));
    resetModules();
    act(() => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    // Contacts are the phone's: the AI Receptionist keeps them, and its Phone.
    const children = [...nav().querySelectorAll('.nav-sub .nav-child')].map((b) => b.querySelector('span')?.textContent);
    expect(children).toEqual(['Phone', 'Contacts', 'Messages', 'Transfers']);
    expect(text()).not.toContain('Hours & Services');
    await go('hours');
    expect(text()).toContain('The Overview');
    expect(host.querySelector('.hours-form')).toBeNull();
  });

  it('hides Contacts while the phone is off, and an open one goes to the Overview', async () => {
    h.modulesList.mockResolvedValue(snapshot(false, true));
    resetModules();
    act(() => root.unmount());
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ToastProvider>
          <App />
        </ToastProvider>,
      );
    });
    await settle();
    expect(nav().querySelector('button[aria-label="Contacts"]')).toBeNull();
    expect(text()).not.toContain('Contacts');
    await go('contacts');
    expect(text()).toContain('The Overview');
    expect(host.querySelector('.contacts-page')).toBeNull();
  });
});
