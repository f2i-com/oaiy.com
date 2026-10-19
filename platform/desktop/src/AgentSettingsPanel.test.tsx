// Settings → Agent: the switch, the Agent's model (the engine's, or ChatGPT
// with Codex's catalogue as the picker), and what the Agent changed, newest
// first. On a desktop without the routes: the switch on and not changeable,
// the model not choosable yet, and the log "Nothing yet".
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({
  controlGet: vi.fn(),
  controlSet: vi.fn(),
  log: vi.fn(),
  prefsGet: vi.fn(),
  prefsSet: vi.fn(),
  recommendation: vi.fn(),
  catalog: vi.fn(),
  codexStatus: vi.fn(),
  codexModels: vi.fn(),
  nodes: vi.fn(),
  setDefault: vi.fn(),
}));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    control: { settings: m.controlGet, setSettings: m.controlSet, log: m.log },
    agentPreferences: { get: m.prefsGet, set: m.prefsSet },
    engineRecommendation: m.recommendation,
    engines: { ...real.engines, catalog: m.catalog },
    codex: { ...real.codex, status: m.codexStatus },
    codexModels: m.codexModels,
    network: { ...real.network, nodes: m.nodes, setDefault: m.setDefault },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: vi.fn() }) }));

import AgentSettingsPanel from './AgentSettingsPanel';

let host: HTMLDivElement;
let root: Root;

async function settle() {
  for (let i = 0; i < 6; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render() {
  await act(async () => {
    root.render(<AgentSettingsPanel onOpenEngines={vi.fn()} />);
  });
  await settle();
}

const section = (title: string) => Array.from(host.querySelectorAll('section')).find((s) => s.querySelector('.section-title')?.textContent === title)!;

beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset();
  m.controlGet.mockResolvedValue(null);
  m.log.mockResolvedValue(null);
  m.prefsGet.mockResolvedValue(null);
  m.recommendation.mockResolvedValue(null);
  m.catalog.mockResolvedValue({ running: true, models: [], defaults: { llm: 'Picked-In-Engines' } });
  m.codexStatus.mockResolvedValue({ available: true, connected: false });
  m.codexModels.mockResolvedValue([]);
  m.nodes.mockResolvedValue({ nodes: [] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('Settings → Agent on a desktop without the routes', () => {
  it('shows the switch on and not changeable, the model not choosable yet, and nothing in the log', async () => {
    await render();
    const sw = host.querySelector<HTMLInputElement>('input.switch')!;
    expect(sw.checked).toBe(true);
    expect(sw.disabled).toBe(true);
    expect(section('What the Agent may do').textContent).toContain('This OAIY can’t change it yet');
    expect(section('The Agent’s model').textContent).toContain('This OAIY can’t choose the Agent’s model yet');
    expect(section('The Agent’s model').textContent).toContain('Uses Picked-In-Engines, chosen in Engines.');
    expect(section('What the Agent changed').textContent).toContain('Nothing yet.');
  });
});

describe('Settings → Agent with the routes', () => {
  beforeEach(() => {
    m.controlGet.mockResolvedValue({ agentMayChange: true });
    m.controlSet.mockImplementation(async (s: unknown) => s);
    m.prefsGet.mockResolvedValue({ model: { source: 'chatgpt', model: 'gpt-5.5' } });
    m.prefsSet.mockImplementation(async (p: unknown) => p);
    m.codexStatus.mockResolvedValue({ available: true, connected: true, email: 'owner@example.com' });
    m.codexModels.mockResolvedValue([
      { id: 'gpt-5.5', displayName: 'GPT-5.5', isDefault: true },
      { id: 'gpt-5.6-luna', displayName: 'Luna' },
    ]);
    m.log.mockResolvedValue([
      { at: '2026-09-29T09:00:00Z', tool: 'plugin_install', args: { source: 'C:/plugins/aokie' }, session: 'setup', ok: true },
      { at: '2026-09-29T10:00:00Z', tool: 'plugin_settings_set', args: { pluginId: 'aokie', settings: { autoAnswer: true } }, session: 'setup', ok: true, summary: 'Turned on answering calls by itself for Aokie' },
      { at: '2026-09-29T09:30:00Z', tool: 'service_install', args: { id: 'oaiy-voice' }, session: 'project', ok: false },
    ]);
  });

  it('lists what the Agent changed, newest first and in words', async () => {
    await render();
    const rows = Array.from(section('What the Agent changed').querySelectorAll('.agent-changes li'));
    expect(rows.map((r) => r.querySelector('strong')!.textContent)).toEqual([
      'Turned on answering calls by itself for Aokie',
      'Installed a serviceoaiy-voice',
      'Installed a pluginC:/plugins/aokie',
    ]);
    expect(rows[1].className).toBe('is-failed');
    expect(rows[1].textContent).toContain('In a chat · did not work');
    expect(rows[2].textContent).toContain('Setting up OAIY');
    expect(m.log).toHaveBeenCalledWith(100);
  });

  it('the switch writes control.json', async () => {
    await render();
    await act(async () => host.querySelector<HTMLInputElement>('input.switch')!.click());
    await settle();
    expect(m.controlSet).toHaveBeenCalledWith({ agentMayChange: false });
    expect(section('What the Agent may do').textContent).toContain('Off: the Agent can still look at OAIY');
  });

  it("offers a paired worker's language model (OAIY Network) only where there is one, and saves it as the engine's", async () => {
    await render();
    expect(section('The Agent’s model').textContent).not.toContain('On a paired computer');
    m.nodes.mockResolvedValue({
      nodes: [{ nodeId: 'w1', name: 'Rig', role: 'worker', online: true, models: [{ id: 'engine:llm:node.w1/qwen', kind: 'llm', model: 'qwen' }, { id: 'engine:image:node.w1/flux', kind: 'image', model: 'flux' }] }],
    });
    act(() => root.unmount());
    root = createRoot(host);
    await render();
    const choice = Array.from(section('The Agent’s model').querySelectorAll('.ai-choice')).find((c) => c.textContent?.includes('On a paired computer'))!;
    await act(async () => choice.querySelector<HTMLInputElement>('input[type="radio"]')!.click());
    await settle();
    const select = choice.querySelector('select')!;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(['Choose a model…', 'qwen · Rig']);
    await act(async () => {
      select.value = 'engine:llm:node.w1/qwen';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    expect(m.prefsSet).toHaveBeenCalledWith({ model: { source: 'engine', model: 'engine:llm:node.w1/qwen' } });
  });

  it('names the paired computer’s model the engine’s default is, as the choice in use, and makes this computer’s own the default again', async () => {
    // (the engine's default is a paired computer's, Engines → Other computers: the desktop says which beside the preference)
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' }, engineDefault: 'engine:llm:node.w1/qwen3.8-flash-next', engineDefaultName: 'Desktop' });
    m.nodes.mockResolvedValue({ nodes: [{ nodeId: 'w1', name: 'Desktop', role: 'worker', online: true, models: [{ id: 'engine:llm:node.w1/qwen3.8-flash-next', kind: 'llm', model: 'qwen3.8-flash-next' }] }] });
    m.setDefault.mockResolvedValue({});
    await render();
    const choices = Array.from(section('The Agent’s model').querySelectorAll('.ai-choice'));
    const paired = choices.find((c) => c.textContent?.includes('On a paired computer'))!;
    expect(paired.className).toContain('is-selected');
    expect(paired.querySelector<HTMLInputElement>('input[type="radio"]')!.checked).toBe(true);
    expect(paired.textContent).toContain('Uses qwen3.8-flash-next on Desktop (Engines → Other computers).');
    expect(paired.querySelector('select')!.value).toBe('engine:llm:node.w1/qwen3.8-flash-next');
    // (not "On this computer … Picked-In-Engines" as if that were in use)
    const local = choices.find((c) => c.textContent?.includes('On this computer'))!;
    expect(local.className).not.toContain('is-selected');
    expect(local.textContent).toContain('Choosing it makes this computer’s own model the engine’s default again.');
    const back = Array.from(paired.querySelectorAll('button')).find((b) => b.textContent?.includes('Use this computer’s model again'))!;
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' } });
    await act(async () => back.click());
    await settle();
    expect(m.setDefault).toHaveBeenCalledWith('llm', null);
    expect(m.prefsSet).not.toHaveBeenCalled();
  });

  it('choosing “On this computer” while the engine’s default is a paired computer’s takes the default back', async () => {
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' }, engineDefault: 'engine:llm:node.w1/qwen', engineDefaultName: 'Rig' });
    m.setDefault.mockResolvedValue({});
    await render();
    const local = Array.from(section('The Agent’s model').querySelectorAll('.ai-choice')).find((c) => c.textContent?.includes('On this computer'))!;
    await act(async () => local.querySelector<HTMLInputElement>('input[type="radio"]')!.click());
    await settle();
    expect(m.setDefault).toHaveBeenCalledWith('llm', null);
    expect(m.prefsSet).not.toHaveBeenCalled();
  });

  it('picks the ChatGPT model from Codex’s catalogue, or goes back to the engine', async () => {
    await render();
    const select = section('The Agent’s model').querySelector('select')!;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(['Codex’s default (GPT-5.5)', 'GPT-5.5', 'Luna']);
    expect(select.value).toBe('gpt-5.5');
    await act(async () => {
      select.value = 'gpt-5.6-luna';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    expect(m.prefsSet).toHaveBeenCalledWith({ model: { source: 'chatgpt', model: 'gpt-5.6-luna' } });
    await act(async () => {
      select.value = '';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    // Codex's default: no model named.
    expect(m.prefsSet).toHaveBeenLastCalledWith({ model: { source: 'chatgpt' } });
    const engine = section('The Agent’s model').querySelectorAll<HTMLInputElement>('input[type=radio]')[0];
    await act(async () => engine.click());
    await settle();
    expect(m.prefsSet).toHaveBeenLastCalledWith({ model: { source: 'engine' } });
  });
});

describe('Settings → Agent when a read fails', () => {
  it('says the Agent’s settings and its changes could not be read, and reads them again when asked', async () => {
    m.prefsGet.mockRejectedValueOnce(new Error('500: the desktop is busy'));
    m.log.mockRejectedValueOnce(new Error('500: the log is busy'));
    await render();
    const model = section('The Agent’s model');
    expect(model.textContent).toContain('The Agent’s settings could not be read: the desktop is busy');
    expect(model.textContent).not.toContain('Checking…');
    const log = section('What the Agent changed');
    expect(log.textContent).toContain('What the Agent changed could not be read: the log is busy');
    m.prefsGet.mockResolvedValue({ model: { source: 'engine' } });
    await act(async () => model.querySelector<HTMLButtonElement>('.setup-retry button')!.click());
    await settle();
    expect(section('The Agent’s model').textContent).not.toContain('could not be read');
    m.log.mockResolvedValue([]);
    await act(async () => section('What the Agent changed').querySelector<HTMLButtonElement>('.setup-retry button')!.click());
    await settle();
    expect(section('What the Agent changed').textContent).toContain('Nothing yet');
  });
});
