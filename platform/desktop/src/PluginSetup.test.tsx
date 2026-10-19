// A plugin's own wizard (PluginSetup.tsx), its settings step: a required one says it is saved to go on, Next saves
// what is in the form before moving on (edits are never dropped), and a number left blank is not saved as 0.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => {
  const push = vi.fn();
  return { list: vi.fn(), setupGet: vi.fn(), detail: vi.fn(), markStep: vi.fn(), finish: vi.fn(), request: vi.fn(), trust: vi.fn(), start: vi.fn(), ask: vi.fn(), push, toast: { push } };
});

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    plugins: { ...real.plugins, list: m.list, trust: m.trust, start: m.start },
    bridge: { ...real.bridge, connectorRequest: m.request },
    setup: { ...real.setup, get: m.setupGet, plugin: m.detail, markStep: m.markStep, finish: m.finish, check: vi.fn().mockResolvedValue({ passed: false, detail: '' }) },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => m.toast }));
vi.mock('./useModules', () => ({ refetchModules: vi.fn() }));
vi.mock('./ask', () => ({ ask: m.ask }));

import PluginWizard from './PluginSetup';
import { resetSetupState } from './useSetupState';
import type { PluginRecord, SetupState } from './api';

const DEMO: PluginRecord = {
  id: 'demo',
  state: 'running',
  dir: 'x',
  userDisabled: false,
  restartAttempts: 0,
  manifest: {
    name: 'Demo',
    version: '0.1.0',
    connectors: [{ id: 'demo', commands: ['settings.get', 'settings.set'] }],
    setup: {
      version: 1,
      title: 'Set up Demo',
      steps: [
        {
          id: 'general',
          kind: 'settings',
          title: 'General',
          fields: [
            { key: 'greeting', label: 'Greeting', type: 'text' },
            { key: 'limit', label: 'Limit', type: 'number' },
          ],
        },
      ],
    },
  } as unknown as PluginRecord['manifest'],
};

const state = (done: string[]): SetupState => ({
  firstRun: { finished: true, position: 'done', skipped: [], chosenPlugins: [] },
  plugins: { demo: { version: 0, done, skipped: [], permissionsAccepted: [] } },
});

let host: HTMLDivElement;
let root: Root;
const onFinished = vi.fn();

async function settle() {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

const footer = (label: string) => Array.from(host.querySelectorAll<HTMLButtonElement>('.setup-foot button')).find((b) => b.textContent?.trim() === label);
const field = (label: string) =>
  Array.from(host.querySelectorAll<HTMLLabelElement>('.setup-settings label')).find((l) => l.textContent?.startsWith(label))!.querySelector('input')!;
const type = async (input: HTMLInputElement, value: string) => {
  await act(async () => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
    set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

beforeEach(() => {
  resetSetupState();
  for (const f of [m.list, m.setupGet, m.detail, m.markStep, m.finish, m.request, m.trust, m.start, m.ask, m.push, onFinished]) f.mockReset();
  m.trust.mockResolvedValue({});
  m.start.mockResolvedValue(undefined);
  m.list.mockResolvedValue({ root: 'x', plugins: [DEMO] });
  m.setupGet.mockResolvedValue(state([]));
  m.detail.mockResolvedValue({ pluginId: 'demo', name: 'Demo', setup: null, capabilities: [], permissionsAccepted: true, needsSetup: true, state: state([]).plugins.demo });
  m.markStep.mockResolvedValue(state(['general']));
  m.finish.mockResolvedValue(state(['general']));
  m.request.mockImplementation(async (_connector: string, command: string) =>
    command === 'settings.get' ? { ok: true, result: { ok: true, data: { settings: { greeting: 'Hi', limit: 5 } } } } : { ok: true, result: { ok: true, data: {} } },
  );
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('a settings step', () => {
  it('says it is saved to go on, and Next saves what is in the form, a blank number left as it is', async () => {
    await act(async () => {
      root.render(<PluginWizard pluginId="demo" initialStep="general" layout="page" onFinished={onFinished} onLeave={vi.fn()} onNavigate={vi.fn()} />);
    });
    await settle();
    expect(host.querySelector('.setup-foot-status')!.textContent).toBe('Save to go on');
    expect(footer('Save and continue')).toBeTruthy();
    await type(field('Greeting'), 'Hello');
    await type(field('Limit'), '');
    await act(async () => footer('Save and continue')!.click());
    await settle();
    // Saved first: the greeting edited, the blank number not sent as 0.
    expect(m.request).toHaveBeenCalledWith('demo', 'settings.set', { greeting: 'Hello' }, expect.any(String));
    expect(m.markStep).toHaveBeenCalledWith('demo', 'general', 'done');
    // The last step: then finished.
    expect(m.finish).toHaveBeenCalledWith('demo');
    expect(onFinished).toHaveBeenCalled();
  });

  it('stays, saying why, when the save fails', async () => {
    m.request.mockImplementation(async (_connector: string, command: string) =>
      command === 'settings.get' ? { ok: true, result: { ok: true, data: { settings: { greeting: 'Hi', limit: 5 } } } } : { ok: true, result: { ok: false, error: 'the plugin is busy' } },
    );
    await act(async () => {
      root.render(<PluginWizard pluginId="demo" initialStep="general" layout="page" onFinished={onFinished} onLeave={vi.fn()} onNavigate={vi.fn()} />);
    });
    await settle();
    await act(async () => footer('Save and continue')!.click());
    await settle();
    expect(host.textContent).toContain('Not saved: the plugin is busy');
    expect(m.finish).not.toHaveBeenCalled();
    expect(onFinished).not.toHaveBeenCalled();
  });
});

// A package nobody signed does not start until the person trusts it: the wizard offers that, and not a Start that fails.
describe('a plugin not signed and not trusted yet', () => {
  const UNSIGNED: PluginRecord = {
    ...DEMO,
    state: 'stopped',
    reason: 'Not started. Not signed, and you have not trusted it yet.',
    trust: { state: 'unsigned', reason: 'Not signed, and you have not trusted it yet.' },
    manifest: { ...DEMO.manifest!, publisher: 'Somebody Famous' },
  };
  const render = async (initialStep?: string) => {
    await act(async () => {
      root.render(<PluginWizard pluginId="demo" initialStep={initialStep} layout="page" onFinished={onFinished} onLeave={vi.fn()} onNavigate={vi.fn()} />);
    });
    await settle();
  };

  it('says so on a step that talks to it, and its one button trusts it, then starts it', async () => {
    m.list.mockResolvedValue({ root: 'x', plugins: [UNSIGNED] });
    m.ask.mockResolvedValue(true);
    await render('general');
    const note = host.querySelector('.setup-run-note')!;
    expect(note.textContent).toContain('Demo is not signed, and you have not trusted it yet, so it does not run.');
    expect(note.textContent).not.toContain('Start it');
    const button = note.querySelector('button')!;
    expect(button.textContent).toContain('Trust this plugin');
    expect(button.className).toContain('btn-primary');
    await act(async () => button.click());
    await settle();
    expect(m.ask).toHaveBeenCalledWith(expect.stringContaining('Trust the "Demo" plugin?'));
    expect(m.trust).toHaveBeenCalledWith('demo');
    expect(m.start).toHaveBeenCalledWith('demo');
  });

  it('trusts nothing when the person says no', async () => {
    m.list.mockResolvedValue({ root: 'x', plugins: [UNSIGNED] });
    m.ask.mockResolvedValue(false);
    await render('general');
    await act(async () => host.querySelector<HTMLButtonElement>('.setup-run-note button')!.click());
    await settle();
    expect(m.trust).not.toHaveBeenCalled();
    expect(m.start).not.toHaveBeenCalled();
  });

  it('makes trusting it the first step’s main action, and names no publisher an unsigned manifest claims', async () => {
    m.list.mockResolvedValue({ root: 'x', plugins: [UNSIGNED] });
    m.detail.mockResolvedValue({ pluginId: 'demo', name: 'Demo', setup: null, capabilities: ['flow.run'], legacyCapabilities: [['flow.run', 'oaiy.flow.run']], permissionsAccepted: false, needsSetup: true, state: state([]).plugins.demo });
    m.ask.mockResolvedValue(true);
    await render();
    expect(host.textContent).not.toContain('Published by');
    expect(host.textContent).not.toContain('Somebody Famous');
    expect(host.textContent).toContain('It is not signed, so OAIY cannot tell who made it');
    // (the names a plugin used before OAIY's are not for the person)
    expect(host.textContent).not.toContain('older names');
    expect(host.querySelector('.setup-foot-status')!.textContent).toBe('Trust it to go on');
    await act(async () => footer('Trust it and continue')!.click());
    await settle();
    expect(m.trust).toHaveBeenCalledWith('demo');
    expect(m.start).toHaveBeenCalledWith('demo');
    expect(m.markStep).toHaveBeenCalledWith('demo', 'permissions', 'done');
  });

  it('names the publisher of a signed, checked package', async () => {
    m.list.mockResolvedValue({ root: 'x', plugins: [{ ...DEMO, trust: { state: 'verified', publisher: 'Demo Labs' } }] });
    m.detail.mockResolvedValue({ pluginId: 'demo', name: 'Demo', setup: null, capabilities: ['flow.run'], permissionsAccepted: false, needsSetup: true, state: state([]).plugins.demo });
    await render();
    expect(host.textContent).toContain('Published by Demo Labs, signed and checked.');
    expect(footer('Accept and continue')).toBeTruthy();
  });
});
