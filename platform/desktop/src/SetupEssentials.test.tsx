// The first-run wizard is the essentials: welcome, your AI (local or ChatGPT,
// the recommended one first), the Agent (one switch), and "Continue with the
// Agent", which records setup as finished, opens the Agent and sends it the
// setupWithAgent intent. "Set up the rest myself" goes on to Plugins. Against
// a desktop without the new routes (404) and one with them.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({
  recommendation: vi.fn(),
  prefsGet: vi.fn(),
  prefsSet: vi.fn(),
  controlGet: vi.fn(),
  controlSet: vi.fn(),
  catalog: vi.fn(),
  codexStatus: vi.fn(),
  setupGet: vi.fn(),
  putFirstRun: vi.fn(),
  push: vi.fn(),
  providers: vi.fn(),
  providerModels: vi.fn(),
}));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    engineRecommendation: m.recommendation,
    agentPreferences: { get: m.prefsGet, set: m.prefsSet },
    control: { settings: m.controlGet, setSettings: m.controlSet, log: vi.fn().mockResolvedValue(null) },
    engines: { ...real.engines, catalog: m.catalog, download: vi.fn() },
    codex: { ...real.codex, status: m.codexStatus },
    pairing: { ...real.pairing, paired: vi.fn().mockResolvedValue({ paired: [] }) },
    link: { ...real.link, status: vi.fn().mockResolvedValue({ linked: false }) },
    plugins: { ...real.plugins, list: vi.fn().mockResolvedValue({ root: 'x', plugins: [] }) },
    bridge: { ...real.bridge, status: vi.fn().mockResolvedValue({ ready: true, flowRuntime: { cliResolved: true, cliKind: 'node' } }) },
    setup: { ...real.setup, get: m.setupGet, putFirstRun: m.putFirstRun, catalog: vi.fn().mockResolvedValue({ plugins: [] }), check: vi.fn() },
    aiProviders: { ...real.aiProviders, list: m.providers, models: m.providerModels },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: m.push }) }));

import SetupPage from './SetupPage';
import { resetSetupState } from './useSetupState';
import type { EngineCatalog, FirstRunState, SetupState } from './api';

const catalog = (chosen: string | null): EngineCatalog => ({
  running: true,
  groups: [{ id: 'llm', name: 'Chat' }],
  models: [{ id: 'catalog-llm', group: 'llm', name: 'Catalog LLM', sizeGb: 5.7, vramGb: 8, recommended: true, needs: [], installed: false, partial: false, download: null }],
  defaults: { llm: chosen },
});

const record = (firstRun: Partial<FirstRunState> = {}): SetupState => ({
  firstRun: { finished: false, position: 'welcome', skipped: [], chosenPlugins: [], ...firstRun },
  plugins: {},
});

let host: HTMLDivElement;
let root: Root;
const onNavigate = vi.fn<(view: string) => void>();
const invoke = vi.fn<(cmd: string, args: unknown) => Promise<unknown>>();

async function settle() {
  for (let i = 0; i < 6; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render(props: Partial<React.ComponentProps<typeof SetupPage>> = {}) {
  await act(async () => {
    root.render(<SetupPage pluginId={null} onExit={vi.fn()} onNavigate={onNavigate} {...props} />);
  });
  await settle();
}

const rail = () => Array.from(host.querySelectorAll('.setup-rail-text strong')).map((e) => e.textContent);
/** A button of the step's footer (the step list has buttons of the same names). */
const button = (label: string) => Array.from(host.querySelectorAll<HTMLButtonElement>('.setup-foot button')).find((b) => b.textContent?.trim() === label);
const heading = () => host.querySelector('.setup-step-head h2')?.firstChild?.textContent;
const choices = () => Array.from(host.querySelectorAll('.ai-choice'));
const click = async (el: Element | undefined | null) => {
  expect(el).toBeTruthy();
  await act(async () => (el as HTMLElement).click());
  await settle();
};

beforeEach(() => {
  resetSetupState();
  for (const f of Object.values(m)) f.mockReset();
  m.setupGet.mockResolvedValue(record());
  m.putFirstRun.mockImplementation(async (fr: FirstRunState) => ({ firstRun: fr, plugins: {} }));
  // An older desktop: none of the new routes (the client turns their 404 into null).
  m.recommendation.mockResolvedValue(null);
  m.prefsGet.mockResolvedValue(null);
  m.controlGet.mockResolvedValue(null);
  m.catalog.mockResolvedValue(catalog(null));
  m.codexStatus.mockResolvedValue({ available: true, connected: false });
  onNavigate.mockReset();
  invoke.mockReset().mockResolvedValue(undefined);
  (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = { invoke };
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
});

describe('first run, on a desktop without the new routes', () => {
  it('is the essentials only, and falls back to ChatGPT first when Engines has nothing chosen', async () => {
    await render();
    // A desktop that keeps no Agent model cannot keep a provider as one: two choices.
    expect(host.querySelector('.ai-choices.is-trio')).toBeNull();
    expect(rail()).toEqual(['Welcome', 'Your AI', 'The Agent', 'Continue with the Agent']);
    expect(heading()).toBe('Welcome to OAIY');
    await click(button('Get started'));
    expect(heading()).toBe('Your AI');
    expect(choices()[0].textContent).toContain('ChatGPT');
    expect(choices()[0].textContent).toContain('Recommended');
    expect(choices()[1].textContent).toContain('On this computer');
    expect(choices()[1].textContent).toContain('The catalog recommends Catalog LLM (5.7 GB download · needs 8 GB of GPU memory).');
    // ChatGPT chosen: its sign-in below both; not signed in, so Next waits and Skip is offered.
    expect(host.textContent).toContain('Sign in with ChatGPT');
    expect(host.textContent).not.toContain('Download Catalog LLM (5.7 GB)');
    expect(button('Next')!.disabled).toBe(true);
    expect(button('Skip')).toBeTruthy();
    // The other one stays choosable: local shows the engine step's card, the download offer.
    await click(choices()[1].querySelector('input'));
    expect(host.querySelector('.ai-choice.is-selected')!.textContent).toContain('On this computer');
    expect(host.querySelector('.setup-reqs')!.textContent).toContain('The catalog recommends:');
    expect(host.querySelector('.setup-reqs')!.textContent).toContain('Download Catalog LLM (5.7 GB)');
  });

  it('puts local first when Engines has a model chosen, and names it', async () => {
    m.catalog.mockResolvedValue(catalog('Picked-In-Engines'));
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    await render();
    expect(heading()).toBe('Your AI');
    expect(choices()[0].textContent).toContain('On this computer');
    expect(choices()[0].textContent).toContain('Recommended');
    expect(choices()[0].textContent).toContain('Uses Picked-In-Engines, chosen in Engines.');
    expect(button('Next')!.disabled).toBe(false);
    await click(button('Next'));
    // No preference kept on this desktop: nothing to write.
    expect(m.prefsSet).not.toHaveBeenCalled();
    expect(heading()).toBe('The Agent');
  });

  it('shows the switch as on, and says it cannot be changed yet', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'agent' }));
    await render();
    const sw = host.querySelector<HTMLInputElement>('input.switch')!;
    expect(sw.checked).toBe(true);
    expect(sw.disabled).toBe(true);
    expect(host.textContent).toContain('Let the Agent set up and change OAIY for you');
    expect(host.textContent).toContain('This OAIY can’t change it yet');
  });

  it('continues with the Agent: finished, the Agent opened, and asked to set up OAIY', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'handoff' }));
    await render();
    expect(heading()).toBe('Continue with the Agent');
    await click(button('Continue with the Agent'));
    expect(m.putFirstRun).toHaveBeenLastCalledWith(expect.objectContaining({ finished: true, position: 'handoff' }));
    expect(onNavigate).toHaveBeenCalledWith('agent');
    expect(invoke).toHaveBeenCalledWith('agent_intent', { intent: 'setupWithAgent' });
  });

  it('says so when the Agent does not take the request, rather than that it took it from here', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'handoff' }));
    // (outside the desktop's window the Agent page cannot be asked)
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
    await render();
    await click(button('Continue with the Agent'));
    expect(onNavigate).toHaveBeenCalledWith('agent');
    expect(m.push).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', title: 'The Agent did not start setting up OAIY' }));
    expect(m.push).not.toHaveBeenCalledWith(expect.objectContaining({ title: 'The essentials are set up' }));
  });

  it('keeps every plugin ticked quickly, each on the newest record', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'plugins' }));
    let release: () => void = () => {};
    // The first save waits: the second is sent after it, on what it answered.
    m.putFirstRun.mockImplementationOnce(
      (fr: FirstRunState) => new Promise((resolve) => (release = () => resolve({ firstRun: fr, plugins: {} }))),
    );
    const { saveFirstRun } = await import('./useSetupState');
    await render();
    const first = saveFirstRun((now) => ({ chosenPlugins: [...now.chosenPlugins, 'a'] }));
    const second = saveFirstRun((now) => ({ chosenPlugins: [...now.chosenPlugins, 'b'] }));
    await settle();
    // (the first is out, the second waits for it)
    expect(m.putFirstRun).toHaveBeenCalledTimes(1);
    release();
    await act(async () => {
      await first;
      await second;
    });
    expect(m.putFirstRun).toHaveBeenLastCalledWith(expect.objectContaining({ chosenPlugins: ['a', 'b'] }));
  });

  it('“Set up the rest myself” opens the full flow at Plugins, and keeps it on the way back', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'handoff' }));
    await render();
    await click(button('Set up the rest myself'));
    expect(heading()).toBe('Plugins');
    expect(rail()).toEqual(['Welcome', 'Your AI', 'The Agent', 'Continue with the Agent', 'Plugins', 'Connect an app', 'Done']);
    expect(m.putFirstRun).toHaveBeenLastCalledWith(expect.objectContaining({ position: 'plugins', finished: false }));
    await click(button('Back'));
    expect(heading()).toBe('Continue with the Agent');
    expect(rail()).toContain('Plugins');
  });

  it('opens where a navigation asked, in the rest too', async () => {
    await render({ step: 'plugins' });
    expect(heading()).toBe('Plugins');
    expect(rail()).toContain('Connect an app');
  });
});

describe('first run, with the desktop’s new routes', () => {
  beforeEach(() => {
    m.recommendation.mockResolvedValue({
      recommend: 'chatgpt',
      local: { ok: false, reason: 'The largest GPU has 6 GB; the recommended model needs 8 GB.', gpus: [{ name: 'Laptop GPU', totalGb: 6, freeGb: 3 }], chosen: null, suggested: null },
      chatgpt: { signedIn: true },
    });
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' } });
    m.prefsSet.mockImplementation(async (p: unknown) => p);
    m.controlGet.mockResolvedValue({ agentMayChange: true });
    m.controlSet.mockImplementation(async (s: unknown) => s);
    m.codexStatus.mockResolvedValue({ available: true, connected: true, email: 'owner@example.com' });
  });

  it('offers an AI provider as a third choice, and sets the Agent to its model on Next', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    m.providers.mockResolvedValue({
      providers: [{ id: 'lm-studio', name: 'LM Studio', protocol: 'openai', baseUrl: 'http://localhost:1234/v1', capabilities: [], enabled: true, allowLocal: true, hasKey: false }],
    });
    m.providerModels.mockResolvedValue(['qwen3.5-9b']);
    await render();
    expect(choices()).toHaveLength(3);
    expect(choices()[2].textContent).toContain('An AI provider');
    expect(host.querySelector('.ai-choices.is-trio')).toBeTruthy();
    await click(choices()[2].querySelector('input'));
    // The picker below: LM Studio, set up already, with the models it lists; one is chosen by the person (a provider
    // merely looked at does not change the Agent's model).
    expect(host.querySelector('.ai-provider')).toBeTruthy();
    expect(m.providerModels).toHaveBeenCalledWith('lm-studio');
    expect(button('Next')!.disabled).toBe(true);
    const model = host.querySelector<HTMLSelectElement>('select[aria-label="Model"]')!;
    await act(async () => {
      model.value = 'qwen3.5-9b';
      model.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    expect(choices()[2].textContent).toContain('LM Studio · qwen3.5-9b');
    expect(button('Next')!.disabled).toBe(false);
    await click(button('Next'));
    expect(m.prefsSet).toHaveBeenCalledWith({ model: { source: 'provider', provider: 'lm-studio', model: 'qwen3.5-9b' } });
    expect(heading()).toBe('The Agent');
  });

  it('says when the Agent’s settings cannot be read, asks again, and saves nothing on a guess', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    m.catalog.mockResolvedValue(catalog('Picked-In-Engines'));
    m.prefsGet.mockRejectedValueOnce(new Error('500: the desktop is busy'));
    await render();
    expect(host.textContent).toContain('The Agent’s settings could not be read: the desktop is busy');
    // (not taken for an older desktop: no "can't choose" and nothing saved; Retry asks again)
    await click(Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes('Try again')));
    expect(host.textContent).not.toContain('could not be read');
    expect(m.prefsGet).toHaveBeenCalledTimes(2);
  });

  it('orders by the recommendation, says why, and sets the Agent to ChatGPT on Next', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    await render();
    expect(choices()[0].textContent).toContain('ChatGPT');
    expect(choices()[0].textContent).toContain('Signed in as owner@example.com.');
    expect(choices()[1].textContent).toContain('The largest GPU has 6 GB; the recommended model needs 8 GB.');
    expect(choices()[1].textContent).toContain('Laptop GPU · 6 GB');
    await click(button('Next'));
    expect(m.prefsSet).toHaveBeenCalledWith({ model: { source: 'chatgpt' } });
    expect(heading()).toBe('The Agent');
  });

  it('the switch writes control.json, and the hand-off says what off means', async () => {
    m.setupGet.mockResolvedValue(record({ position: 'agent' }));
    await render();
    const sw = host.querySelector<HTMLInputElement>('input.switch')!;
    expect(sw.checked).toBe(true);
    expect(sw.disabled).toBe(false);
    await click(sw);
    expect(m.controlSet).toHaveBeenCalledWith({ agentMayChange: false });
    expect(host.querySelector<HTMLInputElement>('input.switch')!.checked).toBe(false);
    await click(button('Next'));
    expect(heading()).toBe('Continue with the Agent');
    expect(host.textContent).toContain('Off: it will tell you what to do instead of doing it.');
  });
});

describe('first run, as a first-timer meets it', () => {
  beforeEach(() => {
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' } });
    m.controlGet.mockResolvedValue({ agentMayChange: true });
  });

  it('names the model Engines chose by its catalogue name, and says it once', async () => {
    m.catalog.mockResolvedValue(catalog('catalog-llm'));
    m.recommendation.mockResolvedValue({
      recommend: 'engine',
      local: { ok: true, reason: 'A language model is chosen in Engines: catalog-llm.', gpus: [{ name: 'Apple M5 Pro', totalGb: 24, freeGb: 20 }], chosen: 'catalog-llm', suggested: null },
      chatgpt: { signedIn: false, available: true },
    });
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    await render();
    const local = choices()[0];
    expect(local.textContent).toContain('Uses Catalog LLM, chosen in Engines.');
    expect(local.textContent).not.toContain('catalog-llm');
    expect(local.textContent).not.toContain('A language model is chosen in Engines');
    expect(host.querySelector('.setup-chosen strong')!.textContent).toBe('Catalog LLM');
  });

  it('neither recommends nor opens on ChatGPT where Codex is not installed, and says so in words', async () => {
    // (a desktop that recommends ChatGPT for a GPU too small, and does not say whether Codex is there)
    m.recommendation.mockResolvedValue({
      recommend: 'chatgpt',
      local: { ok: false, reason: 'The largest GPU has 6 GB; the recommended model needs 8 GB.', gpus: [], chosen: null, suggested: null },
      chatgpt: { signedIn: false },
    });
    m.codexStatus.mockResolvedValue({ available: false, connected: false, detail: "The Codex app isn't installed on this computer." });
    m.setupGet.mockResolvedValue(record({ position: 'ai' }));
    await render();
    expect(choices()[0].textContent).toContain('On this computer');
    expect(host.querySelector('.ai-choice.is-selected')!.textContent).toContain('On this computer');
    expect(host.querySelectorAll('.ai-choice .badge-pending')).toHaveLength(0);
    await click(choices()[1].querySelector('input'));
    expect(host.textContent).toContain('The Codex app isn’t installed on this computer.');
    expect(host.textContent).not.toContain('os error');
  });

  it('holds Next while a ticked plugin is not installed, says why, and shows one install box', async () => {
    m.catalog.mockResolvedValue(catalog('catalog-llm'));
    const aokie = { id: 'aokie', name: 'AI Receptionist', source: { kind: 'folder', path: null, note: 'For all.', notes: { macos: 'For the Mac.', windows: 'For Windows.', linux: 'For Linux.' } }, installed: false, installedVersion: null };
    const { setup } = await import('./api');
    vi.mocked(setup.catalog).mockResolvedValue({ plugins: [aokie] } as Awaited<ReturnType<typeof setup.catalog>>);
    m.setupGet.mockResolvedValue(record({ position: 'plugins', chosenPlugins: ['aokie'] }));
    await render();
    expect(heading()).toBe('Plugins');
    expect(button('Next')!.disabled).toBe(true);
    expect(host.querySelector('.setup-foot-status')!.textContent).toBe('AI Receptionist is ticked but not installed: install it, or untick it, to go on.');
    // Its own install box, and not a second one for any other plugin meanwhile.
    expect(host.querySelectorAll('.setup-folder')).toHaveLength(1);
    expect(host.querySelector('.setup-other-plugin')).toBeNull();
    // Unticked: Next goes on, and the other box is there, closed.
    await click(host.querySelector('.setup-plugin-tick input'));
    m.setupGet.mockResolvedValue(record({ position: 'plugins', chosenPlugins: [] }));
    await settle();
    expect(button('Next')!.disabled).toBe(false);
    expect(host.querySelector('.setup-other-plugin')).toBeTruthy();
    expect(host.querySelector<HTMLDetailsElement>('.setup-other-plugin')!.open).toBe(false);
  });
});
