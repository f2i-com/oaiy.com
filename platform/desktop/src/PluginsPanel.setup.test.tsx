// The live Aokie is fully set up (consent given, phone paired) but was never
// taken through the wizard, so no setup version is recorded for it, and its
// plugin card said "Finish setting up". A plugin whose steps with `done`
// checks all pass counts as set up: the card says so, with no nudge.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => {
  const push = vi.fn();
  // One object, as the real useToast's memoised context value is: a new one each render would re-create the panel's poll.
  return { list: vi.fn(), setupGet: vi.fn(), check: vi.fn(), push, toast: { push } };
});

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    plugins: { ...real.plugins, list: m.list },
    serviceDefinitions: { list: vi.fn().mockResolvedValue({ definitions: [] }) },
    setup: { ...real.setup, get: m.setupGet, check: m.check },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => m.toast }));

import PluginsPanel from './PluginsPanel';
import { invalidate } from './useCached';
import { resetSetupState } from './useSetupState';
import type { PluginRecord } from './api';

/** Aokie's schemaVersion 4 setup: three steps with done checks, the dongle's shown only for a dongle. */
const AOKIE: PluginRecord = {
  id: 'aokie',
  state: 'running',
  dir: 'x',
  userDisabled: false,
  restartAttempts: 0,
  manifest: {
    name: 'Aokie Phone Bridge',
    version: '0.1.0',
    connectors: [{ id: 'aokie', commands: ['consent.get', 'settings.get', 'dongle.diagnostics', 'phone.status'] }],
    setup: {
      version: 1,
      title: 'Set up the AI Receptionist',
      steps: [
        { id: 'consent', kind: 'screen', title: 'Consent', screen: 'receptionist-home', view: 'consent', done: { command: 'consent.get', path: 'mode', equals: 'enforce' } },
        { id: 'speech', kind: 'requirements', title: 'Hearing and speaking', requires: [{ kind: 'service', id: 'oaiy-voice' }] },
        {
          id: 'dongle', kind: 'screen', title: 'Bluetooth dongle', screen: 'receptionist-home', view: 'dongle',
          when: { command: 'settings.get', path: 'settings.transportMode', notIn: ['native', 'auto'] },
          done: { command: 'dongle.diagnostics', path: 'radio.initialized', equals: true },
        },
        { id: 'pair', kind: 'screen', title: 'Pair your phone', screen: 'receptionist-home', view: 'phone', done: { command: 'phone.status', path: 'connected', equals: true } },
        { id: 'answer', kind: 'host', action: 'phone.answerWithOaiy', title: 'Answer calls and texts with OAIY' },
      ],
    },
  } as unknown as PluginRecord['manifest'],
};

let host: HTMLDivElement;
let root: Root;

async function render() {
  await act(async () => {
    root.render(<PluginsPanel />);
  });
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

beforeEach(() => {
  resetSetupState();
  invalidate('pluginsSnapshot');
  m.list.mockReset().mockResolvedValue({ root: 'C:/plugins', plugins: [AOKIE] });
  // No setup version recorded for it: the wizard never ran on this desktop.
  m.setupGet.mockReset().mockResolvedValue({ firstRun: { finished: true, skipped: [], chosenPlugins: [] }, plugins: {} });
  m.check.mockReset();
  m.push.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('a plugin card’s setup, from live checks', () => {
  it('Aokie, all set up by hand, shows as set up and is not nudged', async () => {
    // Consent given, paired; the phone is reached natively, so the dongle step does not show.
    m.check.mockImplementation(async (_id: string, step: string, which: string) =>
      which === 'when' ? { passed: false, detail: 'settings.transportMode is "native"' } : { passed: step !== 'dongle', detail: '' },
    );
    await render();
    expect(host.textContent).toContain('Aokie Phone Bridge');
    expect(host.querySelector('.card-head .badge-ok[title="Set up the AI Receptionist: done"]')?.textContent).toBe('set up');
    expect(host.textContent).not.toContain('Finish setting up');
    expect(m.check).toHaveBeenCalledWith('aokie', 'consent', 'done');
    expect(m.check).toHaveBeenCalledWith('aokie', 'pair', 'done');
    expect(m.check).toHaveBeenCalledWith('aokie', 'dongle', 'when');
  });

  it('still nudges while a check says not yet (the phone is not paired)', async () => {
    m.check.mockImplementation(async (_id: string, step: string, which: string) =>
      which === 'when' ? { passed: false, detail: '' } : { passed: step !== 'pair', detail: step === 'pair' ? 'phone.status: connected is false' : '' },
    );
    await render();
    expect(host.textContent).toContain('Finish setting up Aokie Phone Bridge.');
    expect(host.querySelector('.card-head .badge-ok[title]')).toBeNull();
  });

  it('does not nudge on a guess while the checks are asked', async () => {
    m.check.mockImplementation(() => new Promise(() => {}));
    await render();
    expect(host.textContent).not.toContain('Finish setting up');
    expect(host.textContent).toContain('Set up…');
  });
});

// A fresh Aokie is unhealthy until its setup records the person's consent: it says so, and that is the next step,
// not a fault. The card says "needs setup" (not a red "unhealthy"), and turning unhealthy then is not toasted.
describe('a plugin unhealthy only because its setup is not done', () => {
  const unhealthy: PluginRecord = { ...AOKIE, state: 'unhealthy', reason: 'consent required: no consent has been recorded for this device' };
  const unhealthyToast = () => m.push.mock.calls.filter(([t]) => String(t?.title ?? '').includes('is unhealthy'));

  async function turnUnhealthy() {
    // Running until the card has settled, then unhealthy at the panel's next poll (every 2 s).
    let now: PluginRecord = AOKIE;
    m.list.mockReset().mockImplementation(async () => ({ root: 'C:/plugins', plugins: [now] }));
    await render();
    // The setup record and the live checks are answered before the plugin turns unhealthy, as they are in the
    // window (Aokie turns unhealthy only after three 10 s health probes): React applies them before the next poll.
    for (let i = 0; i < 100 && !host.textContent?.includes('Finish setting up') && !host.querySelector('.card-head .badge-ok[title]'); i++) {
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });
    }
    now = unhealthy;
    await act(async () => {
      await new Promise((r) => setTimeout(r, 2300));
    });
    await render();
  }

  it('shows "needs setup", with its reason, and is not toasted as unhealthy', async () => {
    m.check.mockImplementation(async (_id: string, _step: string, which: string) => ({ passed: false, detail: which === 'when' ? '' : 'not yet' }));
    await turnUnhealthy();
    const badge = host.querySelector('.card-head > .badge:not([data-trust])');
    expect(badge?.textContent).toBe('needs setup');
    expect(badge?.className).toContain('badge-pending');
    expect(host.querySelector('.service-card')?.className).toContain('service-card-starting');
    expect(host.textContent).toContain('Finish setting up Aokie Phone Bridge.');
    expect(host.textContent).toContain('consent required: no consent has been recorded for this device');
    expect(unhealthyToast()).toEqual([]);
  }, 10_000);

  it('is still unhealthy, in red and toasted, once its setup is done (the positive control)', async () => {
    m.setupGet.mockReset().mockResolvedValue({ firstRun: { finished: true, skipped: [], chosenPlugins: [] }, plugins: { aokie: { version: 1 } } });
    await turnUnhealthy();
    const badge = host.querySelector('.card-head > .badge:not([data-trust])');
    expect(badge?.textContent).toBe('unhealthy');
    expect(badge?.className).toContain('badge-err');
    expect(unhealthyToast()).toHaveLength(1);
  }, 10_000);
});
