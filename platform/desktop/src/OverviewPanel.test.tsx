// Overview panel. The behaviour worth pinning here is a regression I created:
// the setup guide hides the "Next steps" section (they cover the same ground),
// and the "N runs have failed" warning had been added INSIDE that section — so
// on a fresh install, when the guide is open by default and a failure is most
// likely, the machine said nothing about it.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { servicesMock, pluginsMock, providersMock, pairedMock, statusMock, nodeInstallMock } =
  vi.hoisted(() => ({
    servicesMock: vi.fn(),
    pluginsMock: vi.fn(),
    providersMock: vi.fn(),
    pairedMock: vi.fn(),
    statusMock: vi.fn(),
    nodeInstallMock: vi.fn(),
  }));

const calendarMock = vi.hoisted(() => vi.fn());
const syncMock = vi.hoisted(() => vi.fn());
const linkMock = vi.hoisted(() => vi.fn());
vi.mock('./api', () => ({
  services: { list: servicesMock },
  plugins: { list: pluginsMock },
  aiProviders: { list: providersMock },
  codex: { status: vi.fn().mockResolvedValue({available:false,connected:false}) },
  pairing: { paired: pairedMock },
  bridge: { status: statusMock },
  nodeRuntime: { install: nodeInstallMock },
  openExternal: vi.fn(),
  // Today's tiles (TodayPanel): nothing to show.
  phone: { status: vi.fn().mockRejectedValue(new Error('no phone')), calls: vi.fn().mockRejectedValue(new Error('no phone')) },
  calendar: { get: (...a: unknown[]) => calendarMock(...a), syncStatus: (...a: unknown[]) => syncMock(...a) },
  link: { status: (...a: unknown[]) => linkMock(...a) },
  engines: { status: vi.fn().mockRejectedValue(new Error('no engines')) },
}));

import OverviewPanel from './OverviewPanel';
import { invalidate, peek } from './useCached';
import { dismissGuide, reopenGuide } from './setupGuide';

const runtime = (failed: number) => ({
  ready: true,
  deviceId: 'dev_1',
  flowRuntime: { cliResolved: true, cliKind: 'node', detail: null },
  runs: { queued: 0, known: 10, failed },
  nodeRuntime: { available: true, installing: false, installsVersion: '24.19.0', source: 'system' },
  plugins: { serving: 1, total: 1 },
});

let host: HTMLDivElement;
let root: Root;
const onNavigate = vi.fn();

async function mount() {
  await act(async () => {
    root.render(<OverviewPanel onNavigate={onNavigate} onOpenPluginScreen={vi.fn()} />);
  });
}

const text = () => host.textContent ?? '';
const button = (label: string) =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));

beforeEach(() => {
  vi.clearAllMocks();
  invalidate();
  onNavigate.mockReset();
  servicesMock.mockResolvedValue({ services: [], dataDir: 'C:\\d' });
  pluginsMock.mockResolvedValue({ plugins: [] });
  providersMock.mockResolvedValue({
    providers: [{ id: 'p', enabled: true, hasKey: true, allowLocal: false }],
  });
  pairedMock.mockResolvedValue({ paired: [] });
  statusMock.mockResolvedValue(runtime(3));
  calendarMock.mockRejectedValue(new Error('no calendar'));
  syncMock.mockRejectedValue(new Error('no calendar'));
  linkMock.mockResolvedValue({ linked: false, attempt: { phase: 'idle' }, available: [] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  reopenGuide();
});

describe("Today's tiles", () => {
  it('shows the phone and its calendar only while the phone receptionist is installed', async () => {
    calendarMock.mockResolvedValue({ available: true, settings: {}, appointments: [], now: '' });
    await mount();
    expect(text()).toContain('The phone');
    expect(text()).toContain('Next appointment');
    act(() => root.unmount());
    root = createRoot(host);
    calendarMock.mockResolvedValue({ available: false, settings: {}, appointments: [], now: '' });
    await mount();
    expect(text()).not.toContain('The phone');
    expect(text()).not.toContain('Next appointment');
    expect(text()).toContain('Language model');
  });

  it('says nothing about FormLogic while unlinked', async () => {
    await mount();
    expect(text()).not.toContain('FormLogic:');
  });

  it('tells the truth when FormLogic cannot be reached: offline, when it last synced, and what waits', async () => {
    const lastSync = new Date(Date.now() - 12 * 60_000).toISOString();
    syncMock.mockResolvedValue({ linked: true, state: 'offline', at: new Date().toISOString(), lastSuccessAt: lastSync, pulled: 0, pushed: 0, error: 'formlogic.com can’t be reached: it did not answer in time', pending: { creates: 1, updates: 1, deletes: 0, total: 2 } });
    linkMock.mockResolvedValue({ linked: true, heartbeatError: 'formlogic.com can’t be reached', outbox: { waiting: 3, lastError: 'formlogic.com can’t be reached' }, attempt: { phase: 'idle' }, available: [] });
    await mount();
    expect(text()).toContain('FormLogic: Offline');
    expect(text()).toContain('last synced 12 min ago');
    expect(text()).toContain('5 changes waiting');
    await act(async () => button('FormLogic: Offline')!.click());
    expect(onNavigate).toHaveBeenCalledWith('connections');
  });

  it('says when it last synced when all is well', async () => {
    syncMock.mockResolvedValue({ linked: true, state: 'synced', at: new Date().toISOString(), lastSuccessAt: new Date().toISOString(), pulled: 0, pushed: 0, error: null, pending: { creates: 0, updates: 0, deletes: 0, total: 0 } });
    linkMock.mockResolvedValue({ linked: true, lastHeartbeatAt: new Date().toISOString(), outbox: { waiting: 0 }, attempt: { phase: 'idle' }, available: [] });
    await mount();
    expect(text()).toContain('FormLogic: Synced');
    expect(text()).toContain('just now');
    expect(text()).not.toContain('waiting');
  });
});

describe('OverviewPanel failure reporting', () => {
  it('surfaces partial refresh failures and recovers on demand', async () => {
    servicesMock.mockRejectedValueOnce(new Error('offline'));
    await mount();
    expect(text()).toContain("Couldn't refresh services");
    await act(async () => button('Refresh status')!.click());
    expect(text()).not.toContain("Couldn't refresh services");
    expect(servicesMock).toHaveBeenCalledTimes(2);
  });
  it('reports failed runs even while the setup guide is open', async () => {
    reopenGuide(); // guide auto-opens, as on a fresh install
    await mount();

    // The guide is up (it reads "You're set up" here because the fixture has a
    // ready runtime and a keyed provider — either way the panel is rendered).
    expect(host.querySelector('.setup-guide')).not.toBeNull();
    // The regression: "Next steps" yields to the guide, and this warning used
    // to live inside it — so a fresh install stayed silent about real failures.
    expect(text()).toContain('3 runs have failed');
  });

  it('still reports them once the guide is dismissed', async () => {
    dismissGuide();
    await mount();
    expect(host.querySelector('.setup-guide')).toBeNull();
    expect(text()).toContain('3 runs have failed');
  });

  it('takes the user to the run history', async () => {
    dismissGuide();
    await mount();
    await act(async () => {
      button('See why')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(onNavigate).toHaveBeenCalledWith('runs');
  });

  it('says why flows cannot run when the CLI resolved but its engine did not', async () => {
    // The CLI is there and Node is there — what is missing is the ZIPP engine
    // the CLI runs flows on. The readiness route puts the probe's reason in
    // `detail`; the banner must show THAT, not a generic "unavailable".
    dismissGuide();
    statusMock.mockResolvedValue({
      ...runtime(0),
      ready: false,
      flowRuntime: {
        cliResolved: true,
        cliKind: 'node',
        engine: { name: null, release: null, status: 'unavailable', reason: 'ZIPP engine unavailable: zipp/zipp_wasm_bg.wasm is missing' },
        detail: 'the OAIY CLI does not run user logic on ZIPP: ZIPP engine unavailable: zipp/zipp_wasm_bg.wasm is missing. Reinstall OAIY Desktop.',
      },
    });
    await mount();
    expect(text()).toContain('Flows cannot run on this machine');
    expect(text()).toContain('zipp_wasm_bg.wasm is missing');
    expect(text()).toContain('Reinstall OAIY Desktop');
  });

  it('fills the cache key ServicesPanel actually seeds from', async () => {
    // The app opens on Overview, so Overview's poll is what has a chance to warm
    // the Services panel. Writing only the `services` ARRAY meant ServicesPanel —
    // which seeds from the whole `servicesSnapshot` — still found nothing, and
    // the very first visit to Services showed "Loading services…" anyway. The
    // reported symptom, surviving its own fix on the one path everybody takes.
    dismissGuide();
    await mount();
    // Compared against the fixture itself, not a re-typed literal.
    expect(peek('servicesSnapshot')).toEqual(await servicesMock.mock.results[0].value);
    expect(peek('services')).toEqual([]);
  });

  it('says nothing when nothing has failed', async () => {
    statusMock.mockResolvedValue(runtime(0));
    dismissGuide();
    await mount();
    // A permanent "0 failed" line is noise that trains people to ignore the row.
    expect(text()).not.toContain('failed on this machine');
  });

  it('uses singular wording for one failure', async () => {
    statusMock.mockResolvedValue(runtime(1));
    dismissGuide();
    await mount();
    expect(text()).toContain('1 run has failed');
  });
});
