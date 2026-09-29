// The live Aokie is fully set up (consent given, phone paired) but was never
// taken through the wizard, so no setup version is recorded for it, and its
// plugin card said "Finish setting up". A plugin whose steps with `done`
// checks all pass counts as set up: the card says so, with no nudge.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ list: vi.fn(), setupGet: vi.fn(), check: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    plugins: { ...real.plugins, list: m.list },
    serviceDefinitions: { list: vi.fn().mockResolvedValue({ definitions: [] }) },
    setup: { ...real.setup, get: m.setupGet, check: m.check },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: vi.fn() }) }));

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
