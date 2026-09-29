// The Overview's setup card: "Continue setup" while the first-run wizard is not
// finished, a nudge for each plugin whose setup is not finished, and the live
// checks a click away. setupGuide.test.ts and setupFlow.test.ts pin WHICH
// steps are done; this pins what the card puts on screen.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const setupState = vi.hoisted(() => ({ value: null as unknown }));
const openSetup = vi.hoisted(() => vi.fn());
vi.mock('./useSetupState', () => ({
  useSetupState: () => setupState.value,
  openSetup: (...a: unknown[]) => openSetup(...a),
}));
// No desktop in a test: the engines' catalog is not there.
vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./api')>()),
  engines: { catalog: vi.fn().mockRejectedValue(new Error('no engines')) },
}));

import SetupGuidePanel from './SetupGuidePanel';
import type { SetupInput } from './setupGuide';
import type { PluginRecord, RuntimeStatus, SetupState } from './api';

const READY = { ready: true, flowRuntime: { cliResolved: true, cliKind: 'node' } } as RuntimeStatus;
const BROKEN = {
  ready: false,
  flowRuntime: { cliResolved: false, cliKind: 'missing', detail: 'Node is not installed.' },
} as RuntimeStatus;

const input: SetupInput = {
  runtime: BROKEN,
  providers: [],
  services: [],
  plugins: [],
  connected: [],
};

const record = (finished: boolean, plugins: SetupState['plugins'] = {}): SetupState => ({
  firstRun: { finished, skipped: [], chosenPlugins: [], position: 'engine' },
  plugins,
});

const aokie: PluginRecord = {
  id: 'aokie',
  state: 'crashed',
  dir: 'x',
  userDisabled: false,
  restartAttempts: 0,
  manifest: {
    name: 'Aokie Phone Bridge',
    version: '0.1.0',
    setup: { version: 1, title: 'Set up the AI Receptionist', steps: [] },
  } as unknown as PluginRecord['manifest'],
};

let host: HTMLDivElement;
let root: Root;

function render(props: Partial<React.ComponentProps<typeof SetupGuidePanel>> = {}) {
  act(() => {
    root.render(<SetupGuidePanel {...input} onNavigate={vi.fn()} onDismiss={vi.fn()} {...props} />);
  });
}

const rows = () => Array.from(host.querySelectorAll('.setup-step'));
const rowFor = (title: string) => rows().find((li) => li.querySelector('strong')?.textContent?.includes(title))!;
const button = (label: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));

beforeEach(() => {
  setupState.value = null;
  openSetup.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('the setup card', () => {
  it('says where first-run setup got to, and continues it', () => {
    setupState.value = record(false);
    render();
    expect(host.querySelector('.setup-guide')).not.toBeNull();
    expect(host.querySelector('.section-title')?.textContent).toMatch(/^Continue setup · 0 of \d$/);
    expect(host.textContent).toContain('Next: The engine');
    act(() => button('Continue setup')!.click());
    expect(openSetup).toHaveBeenCalledWith();
  });

  it('nudges about a plugin dropped in by hand, and opens its own wizard', () => {
    setupState.value = record(true);
    render({ plugins: [aokie] });
    expect(host.querySelector('.section-title')?.textContent).toBe('Finish setting up');
    expect(host.textContent).toContain('Finish setting up Aokie Phone Bridge');
    expect(host.textContent).toContain('Set up the AI Receptionist');
    act(() => button('Set up…')!.click());
    expect(openSetup).toHaveBeenCalledWith({ plugin: 'aokie' });
  });

  it('no nudge once its setup version is finished, or while it is turned off', () => {
    setupState.value = record(true, { aokie: { version: 1, done: [], skipped: [] } });
    render({ plugins: [aokie] });
    expect(host.textContent).not.toContain('Finish setting up');
    expect(host.textContent).toContain('Setup is finished.');
    setupState.value = record(true);
    render({ plugins: [{ ...aokie, userDisabled: true }] });
    expect(host.textContent).not.toContain('Finish setting up');
  });

  it('keeps the live checks a click away: the backend’s blocker, a caller’s action, done rows quiet', () => {
    setupState.value = record(true);
    const onNavigate = vi.fn();
    render({ onNavigate, actions: { runtime: <button className="install-node">Install Node</button> } });
    expect(rows()).toHaveLength(4);
    expect(rowFor('flow runtime').textContent).toContain('Node is not installed.');
    expect(rowFor('flow runtime').querySelector('.install-node')).not.toBeNull();
    expect(rowFor('flow runtime').textContent).not.toContain('Fix the runtime');
    act(() => rowFor('a model').querySelector('button')!.click());
    expect(onNavigate).toHaveBeenCalledWith('providers');
    render({ runtime: READY });
    expect(rowFor('flow runtime').className).toContain('is-done');
    expect(rowFor('flow runtime').querySelector('button')).toBeNull();
  });

  it('with an older desktop (no setup record) it is the old guide, with setup a click away', () => {
    render();
    expect(host.querySelector('.section-title')?.textContent).toContain('0 of 2');
    act(() => button('Open setup')!.click());
    expect(openSetup).toHaveBeenCalled();
    render({ runtime: READY, providers: [{ id: 'p', enabled: true, hasKey: true, allowLocal: false } as never] });
    expect(host.textContent).toContain('You’re set up');
  });

  it('reports dismissal to its caller', () => {
    const onDismiss = vi.fn();
    render({ onDismiss });
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="Dismiss the setup guide"]')!.click());
    expect(onDismiss).toHaveBeenCalled();
  });
});
