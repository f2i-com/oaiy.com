// The Overview's setup card: "Continue setup" while the first-run wizard is not
// finished, a nudge for each plugin whose setup is not finished, and the live
// checks a click away. setupGuide.test.ts and setupFlow.test.ts pin WHICH
// steps are done; this pins what the card puts on screen.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const setupState = vi.hoisted(() => ({ value: null as unknown, error: null as string | null }));
const openSetup = vi.hoisted(() => vi.fn());
const refetchSetup = vi.hoisted(() => vi.fn());
const catalogMock = vi.hoisted(() => vi.fn());
vi.mock('./useSetupState', () => ({
  useSetupState: () => setupState.value,
  useSetupError: () => setupState.error,
  refetchSetup: (...a: unknown[]) => refetchSetup(...a),
  openSetup: (...a: unknown[]) => openSetup(...a),
}));
// No desktop in a test: the engines' catalog is not there unless a test says so.
vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./api')>()),
  engines: { catalog: (...a: unknown[]) => catalogMock(...a) },
  agentPreferences: { get: vi.fn().mockResolvedValue(null) },
  link: { status: vi.fn().mockResolvedValue({ linked: false }) },
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
  setupState.error = null;
  openSetup.mockReset();
  refetchSetup.mockReset();
  catalogMock.mockReset().mockRejectedValue(new Error('no engines'));
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
    // Left by an older dashboard at "engine": the essentials' "Your AI" now.
    expect(host.textContent).toContain('Next: Your AI');
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

  it('says what needs a look once setup is finished and something it set up is not true any more', async () => {
    setupState.value = record(true);
    // A language model chosen in Engines counts as the flows' model; the runtime is broken.
    catalogMock.mockResolvedValue({ running: true, defaults: { llm: 'qwen3.5-9b' } });
    await act(async () => {
      root.render(<SetupGuidePanel {...input} onNavigate={vi.fn()} onDismiss={vi.fn()} />);
    });
    await act(async () => {});
    expect(host.querySelector('.section-title')?.textContent).toBe('Something needs a look');
    expect(host.querySelector('.setup-guide')!.className).not.toContain('is-complete');
    expect(host.textContent).not.toContain('Setup is finished.');
    // Outside the collapsed list: the runtime alone (the model is chosen in Engines).
    const outside = Array.from(host.querySelectorAll('.setup-guide > .setup-list .setup-step')).map((li) => li.querySelector('strong')?.textContent);
    expect(outside).toEqual(['Get the flow runtime ready']);
  });

  it('says setup could not be read, with a way to ask again, rather than a wizard that never loads', () => {
    setupState.error = 'the desktop is busy';
    render();
    expect(host.textContent).toContain('Setup could not be read: the desktop is busy');
    expect(button('Open setup')).toBeUndefined();
    act(() => button('Try again')!.click());
    expect(refetchSetup).toHaveBeenCalled();
  });

  it('reports dismissal to its caller', () => {
    const onDismiss = vi.fn();
    render({ onDismiss });
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="Dismiss the setup guide"]')!.click());
    expect(onDismiss).toHaveBeenCalled();
  });
});
