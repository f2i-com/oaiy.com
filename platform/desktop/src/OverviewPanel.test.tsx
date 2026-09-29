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
const modulesMock = vi.hoisted(() => vi.fn());
const connectorMock = vi.hoisted(() => vi.fn());
const updateStatusMock = vi.hoisted(() => vi.fn());
const backupStatusMock = vi.hoisted(() => vi.fn());
vi.mock('./api', () => ({
  // The update banner: nothing to install unless a test says so.
  updates: { status: (...a: unknown[]) => updateStatusMock(...a) },
  backup: { status: (...a: unknown[]) => backupStatusMock(...a) },
  modules: { list: (...a: unknown[]) => modulesMock(...a) },
  services: { list: servicesMock },
  plugins: { list: pluginsMock },
  aiProviders: { list: providersMock },
  codex: { status: vi.fn().mockResolvedValue({available:false,connected:false}) },
  pairing: { paired: pairedMock },
  bridge: { status: statusMock, connectorRequest: (...a: unknown[]) => connectorMock(...a) },
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
import { resetModules } from './useModules';

/** The desktop's modules: the phone and the calendar on or off. */
const modulesOn = (phone: boolean, calendar: boolean) => ({
  snapshot: {
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: phone, builtin: true, provider: null },
      { id: 'calendar', name: 'Calendar', enabled: calendar, builtin: true, provider: null },
    ],
    contributions: {},
    warnings: [],
  },
  etag: '"1-x"',
});
import { dismissGuide, reopenGuide } from './setupGuide';
import { describeLastBackup } from './BackupPanel';
import type { BackupStatus } from './api';

const runtime = (failed: number) => ({
  ready: true,
  deviceId: 'dev_1',
  flowRuntime: { cliResolved: true, cliKind: 'node', detail: null },
  runs: { queued: 0, known: 10, failed },
  nodeRuntime: { available: true, installing: false, installsVersion: '24.19.0', source: 'system' },
  plugins: { serving: 1, total: 1 },
});

/** The update status when there is nothing to install. */
const noUpdate = {
  state: 'upToDate', currentVersion: '0.1.0', channel: 'stable', latestVersion: '0.1.0', notes: null, publishedAt: null,
  lastCheckedAt: null, error: null, failedDuring: null, progress: null, note: null, canAutoUpdate: true, manualReason: null,
  blockers: [], manualUrl: 'https://github.com/f2i-com/oaiy.com/releases/latest', autoCheck: true, nextCheckIn: null,
};

let host: HTMLDivElement;
let root: Root;
const onNavigate = vi.fn();
const onOpenPluginScreen = vi.fn();

async function mount() {
  await act(async () => {
    root.render(<OverviewPanel onNavigate={onNavigate} onOpenPluginScreen={onOpenPluginScreen} />);
  });
}

const text = () => host.textContent ?? '';
const button = (label: string) =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));
/** The FormLogic tile, found by the name it is read out with ("FormLogic: Synced. …"). */
const formlogicTile = (headline = '') =>
  host.querySelector<HTMLButtonElement>(`button[aria-label^="FormLogic: ${headline}"]`);

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
  // No backup line unless a test asks for one.
  backupStatusMock.mockRejectedValue(new Error('no backup status'));
  linkMock.mockResolvedValue({ linked: false, attempt: { phase: 'idle' }, available: [] });
  updateStatusMock.mockResolvedValue(noUpdate);
  localStorage.clear();
  resetModules();
  modulesMock.mockResolvedValue(modulesOn(true, true));
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
  it('shows the phone and its calendar only while a plugin provides them', async () => {
    calendarMock.mockResolvedValue({ available: true, settings: {}, appointments: [], now: '' });
    await mount();
    expect(text()).toContain('The phone');
    expect(text()).toContain('Next appointment');
    act(() => root.unmount());
    root = createRoot(host);
    resetModules();
    modulesMock.mockResolvedValue(modulesOn(false, false));
    await mount();
    expect(text()).not.toContain('The phone');
    expect(text()).not.toContain('Next appointment');
    expect(text()).toContain('Language model');
  });

  it('says nothing about FormLogic while unlinked', async () => {
    await mount();
    expect(formlogicTile()).toBeNull();
  });

  it('tells the truth when FormLogic cannot be reached: offline, when it last synced, and what waits', async () => {
    const lastSync = new Date(Date.now() - 12 * 60_000).toISOString();
    syncMock.mockResolvedValue({ linked: true, state: 'offline', at: new Date().toISOString(), lastSuccessAt: lastSync, pulled: 0, pushed: 0, error: 'formlogic.com can’t be reached: it did not answer in time', pending: { creates: 1, updates: 1, deletes: 0, total: 2 } });
    linkMock.mockResolvedValue({ linked: true, heartbeatError: 'formlogic.com can’t be reached', outbox: { waiting: 3, lastError: 'formlogic.com can’t be reached' }, attempt: { phase: 'idle' }, available: [] });
    await mount();
    expect(formlogicTile('Offline')?.textContent).toContain('Offline');
    expect(text()).toContain('last synced 12 min ago');
    expect(text()).toContain('5 changes waiting');
    await act(async () => formlogicTile('Offline')!.click());
    expect(onNavigate).toHaveBeenCalledWith('connections');
  });

  it('says when it last synced when all is well', async () => {
    syncMock.mockResolvedValue({ linked: true, state: 'synced', at: new Date().toISOString(), lastSuccessAt: new Date().toISOString(), pulled: 0, pushed: 0, error: null, pending: { creates: 0, updates: 0, deletes: 0, total: 0 } });
    linkMock.mockResolvedValue({ linked: true, lastHeartbeatAt: new Date().toISOString(), outbox: { waiting: 0 }, attempt: { phase: 'idle' }, available: [] });
    await mount();
    expect(formlogicTile('Synced')?.textContent).toContain('Synced');
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

describe("Plugins' own cards", () => {
  /** The desktop's snapshot with the phone on and what Aokie contributes to the Overview. */
  const withCards = () => {
    const on = modulesOn(true, true);
    return {
      ...on,
      snapshot: {
        ...on.snapshot,
        contributions: {
          sections: [],
          overview: [
            {
              pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', id: 'aokie-hero', kind: 'hero', title: 'Aokie receptionist', icon: 'phone', module: 'phone',
              bind: { headline: '$health.status', body: '$health.detail' },
              cta: { label: 'Open AI Receptionist', view: 'plugin:aokie:receptionist' },
            },
            {
              pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', id: 'outbox', kind: 'tile', title: 'Waiting to send', icon: 'cloud',
              bind: { value: '$poll.data-delivery.outbox.pending' }, view: 'plugin:aokie:receptionist',
            },
            {
              pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', id: 'radio', kind: 'status', title: 'Radio',
              bind: { value: '$poll.data-delivery.radio.state', detail: 'The Bluetooth adapter' },
            },
          ],
          polls: [{ pluginId: 'aokie', id: 'data-delivery', connector: 'aokie', command: 'dongle.diagnostics', intervalMs: 10000 }],
          agent: { tools: [] },
          setup: [],
        },
      },
    };
  };
  const aokie = (state: string, extra: Record<string, unknown> = {}) => ({
    plugins: [{ id: 'aokie', state, dir: 'C:\\p\\aokie', userDisabled: false, restartAttempts: 0, manifest: { name: 'Aokie Phone Bridge', version: '1.0.0' }, lastHealth: { status: 'ok', detail: 'Phone connected' }, ...extra }],
  });
  const settle = async () => {
    for (let i = 0; i < 4; i++) await act(async () => {});
  };

  it("fills a hero from the plugin's health and a tile from its poll, and opens its page", async () => {
    dismissGuide();
    pluginsMock.mockResolvedValue(aokie('running'));
    modulesMock.mockResolvedValue(withCards());
    connectorMock.mockResolvedValue({ ok: true, result: { ok: true, data: { outbox: { pending: 4 }, radio: { state: 'ready' } } } });
    await mount();
    await settle();
    expect(text()).toContain('Aokie receptionist');
    expect(text()).toContain('Phone connected');
    // Its health status reads as one pill, not "running" beside a bare "ok".
    expect(host.querySelector('.overview-hero .badge')?.textContent).toBe('Healthy');
    expect(host.querySelector('.overview-hero-headline')).toBeNull();
    expect(text()).not.toContain('$health');
    expect(text()).not.toContain('$poll');
    // Sent with no payload, through the gated connector route.
    expect(connectorMock).toHaveBeenCalledWith('aokie', 'dongle.diagnostics', undefined, expect.any(String));
    const tile = host.querySelector<HTMLButtonElement>('button[aria-label^="Waiting to send"]')!;
    expect(tile.getAttribute('aria-label')).toBe('Waiting to send: 4');
    expect(tile.textContent).toContain('4');
    expect(host.querySelector('.overview-status')?.textContent).toContain('ready');
    await act(async () => tile.click());
    expect(onOpenPluginScreen).toHaveBeenCalledWith('aokie', 'receptionist');
    await act(async () => button('Open AI Receptionist')!.click());
    expect(onOpenPluginScreen).toHaveBeenCalledTimes(2);
  });

  it("shows an em dash while its poll fails, and a stopped plugin's reason rather than its old health", async () => {
    dismissGuide();
    pluginsMock.mockResolvedValue(aokie('crashed', { reason: 'It stopped: the dongle was unplugged.' }));
    modulesMock.mockResolvedValue(withCards());
    connectorMock.mockRejectedValue(new Error('not running'));
    await mount();
    await settle();
    expect(text()).not.toContain('Phone connected');
    expect(text()).toContain('It stopped: the dongle was unplugged.');
    const tile = host.querySelector<HTMLButtonElement>('button[aria-label^="Waiting to send"]')!;
    expect(tile.getAttribute('aria-label')).toBe('Waiting to send: not known');
    expect(tile.textContent).toContain('—');
  });

  it('shows no plugin cards when the desktop contributes none', async () => {
    dismissGuide();
    pluginsMock.mockResolvedValue(aokie('running'));
    await mount();
    await settle();
    expect(text()).not.toContain('Aokie receptionist');
    expect(connectorMock).not.toHaveBeenCalled();
  });
});

describe('The update banner', () => {
  it('says nothing when there is nothing to install', async () => {
    dismissGuide();
    await mount();
    expect(host.querySelector('.update-banner')).toBeNull();
  });

  it('says a newer version is available, opens Settings, and can be dismissed', async () => {
    dismissGuide();
    updateStatusMock.mockResolvedValue({ ...noUpdate, state: 'available', latestVersion: '0.2.0' });
    await mount();
    expect(host.querySelector('.update-banner')?.textContent).toContain('OAIY 0.2.0 is available.');
    await act(async () => button('See what is new')!.click());
    expect(onNavigate).toHaveBeenCalledWith('settings');
    await act(async () => host.querySelector<HTMLButtonElement>('button[aria-label="Dismiss until the next version"]')!.click());
    expect(host.querySelector('.update-banner')).toBeNull();
  });
});

// The "Last backup" line. describeLastBackup is pure, so the cases are checked with a fixed clock.
const NOW = new Date('2026-09-30T12:00:00Z');
const daysAgo = (n: number) => new Date(NOW.getTime() - n * 86_400_000).toISOString();
const backupStatus = (over: Partial<BackupStatus> = {}): BackupStatus => ({
  lastBackupAt: daysAgo(3),
  lastBackupOk: true,
  lastBackupSize: 1234,
  pendingRestore: null,
  lastRestore: null,
  undoAvailable: false,
  undoKind: null,
  running: null,
  ...over,
});

describe('describeLastBackup', () => {
  it('says how long ago the last backup was, plainly', () => {
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(0) }), NOW)).toEqual({ text: 'Last backup: today', nudge: false, kind: 'recent' });
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(1) }), NOW).text).toBe('Last backup: yesterday');
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(3) }), NOW)).toEqual({ text: 'Last backup: 3 days ago', nudge: false, kind: 'recent' });
  });

  it('is not a nudge at 29 or 30 days, and is one from 31', () => {
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(29) }), NOW)).toMatchObject({ text: 'Last backup: 29 days ago', nudge: false, kind: 'recent' });
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(30) }), NOW).nudge).toBe(false);
    const stale = describeLastBackup(backupStatus({ lastBackupAt: daysAgo(31) }), NOW);
    expect(stale).toMatchObject({ nudge: true, kind: 'stale' });
    expect(stale.text).toBe('It has been 31 days since your last backup. A backup takes a minute.');
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(42) }), NOW).text).toContain('42 days');
  });

  it('counts months once it has been a while', () => {
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(95) }), NOW).text).toBe('It has been 3 months since your last backup. A backup takes a minute.');
  });

  it('nudges gently when there has never been one, or the date cannot be read', () => {
    expect(describeLastBackup(backupStatus({ lastBackupAt: null, lastBackupOk: null, lastBackupSize: null }), NOW)).toEqual({
      text: 'You have not made a backup yet.',
      nudge: true,
      kind: 'never',
    });
    expect(describeLastBackup(backupStatus({ lastBackupAt: 'not a date' }), NOW).kind).toBe('never');
  });

  it('says so gently when the last backup did not finish', () => {
    const line = describeLastBackup(backupStatus({ lastBackupOk: false }), NOW);
    expect(line).toMatchObject({ nudge: true, kind: 'failed' });
    expect(line.text).toContain('did not finish');
  });

  it('puts a restore that waits for a restart first', () => {
    const line = describeLastBackup(
      backupStatus({ lastBackupAt: daysAgo(90), pendingRestore: { id: 'r1', kind: 'restore', stagedAt: daysAgo(0), expiresAt: daysAgo(-1), expired: false, files: 4, agentStorage: false, classes: [] } }),
      NOW,
    );
    expect(line).toEqual({ text: 'A restore is waiting for a restart', nudge: false, kind: 'pending' });
  });

  it('never shows a date in the future as negative days', () => {
    expect(describeLastBackup(backupStatus({ lastBackupAt: daysAgo(-2) }), NOW).text).toBe('Last backup: today');
  });
});

describe('the Overview line about backups', () => {
  it('shows the last backup and its button opens Settings', async () => {
    dismissGuide();
    backupStatusMock.mockResolvedValue(backupStatus({ lastBackupAt: new Date(Date.now() - 3 * 86_400_000).toISOString() }));
    await mount();
    expect(text()).toContain('Last backup: 3 days ago');
    await act(async () => button('Back up')!.click());
    expect(onNavigate).toHaveBeenCalledWith('settings');
  });

  it('turns into a nudge after 30 days, with a button that opens Settings', async () => {
    dismissGuide();
    backupStatusMock.mockResolvedValue(backupStatus({ lastBackupAt: new Date(Date.now() - 42 * 86_400_000).toISOString() }));
    await mount();
    expect(text()).toContain('It has been 42 days since your last backup');
    await act(async () => button('Back up now')!.click());
    expect(onNavigate).toHaveBeenCalledWith('settings');
  });

  it('says a restore waits for a restart', async () => {
    dismissGuide();
    backupStatusMock.mockResolvedValue(backupStatus({ pendingRestore: { id: 'r1', kind: 'restore', stagedAt: new Date().toISOString(), expiresAt: new Date().toISOString(), expired: false, files: 2, agentStorage: false, classes: [] } }));
    await mount();
    expect(text()).toContain('A restore is waiting for a restart');
    await act(async () => button('Open Settings')!.click());
    expect(onNavigate).toHaveBeenCalledWith('settings');
  });

  it('shows no line, and does not break the panel, when the status cannot be read', async () => {
    dismissGuide();
    backupStatusMock.mockRejectedValue(new Error('404: not found'));
    await mount();
    expect(backupStatusMock).toHaveBeenCalled();
    expect(text()).not.toContain('backup');
    expect(text()).toContain('This machine');
  });
});
