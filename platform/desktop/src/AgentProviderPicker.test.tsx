// An AI provider for the Agent (AgentProviderPicker.tsx): the providers already set up that it can think with
// (OpenAI-compatible and on), presets to add one (LM Studio, Ollama, an API key), the model from the list it gives,
// and what is said when it does not answer. The setup wizard and Settings → Agent both use it.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ list: vi.fn(), upsert: vi.fn(), setKey: vi.fn(), models: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, aiProviders: { ...real.aiProviders, list: m.list, upsert: m.upsert, setKey: m.setKey, models: m.models } };
});

import { AgentProviderPicker, agentCapable, freeId, looksLocal, startingModel, type ProviderPick } from './AgentProviderPicker';
import type { AiProviderPublic } from './api';

const provider = (over: Partial<AiProviderPublic>): AiProviderPublic => ({
  id: 'x',
  name: 'X',
  protocol: 'openai',
  baseUrl: 'http://localhost:1234/v1',
  capabilities: [],
  enabled: true,
  allowLocal: true,
  hasKey: false,
  ...over,
});

let host: HTMLDivElement;
let root: Root;
const onPick = vi.fn<(p: ProviderPick) => void>();
let saved: AiProviderPublic[] = [];

async function settle() {
  for (let i = 0; i < 6; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render(value: ProviderPick | null = null) {
  await act(async () => {
    root.render(<AgentProviderPicker value={value} onPick={onPick} />);
  });
  await settle();
}

const select = (label: string) => host.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;
const input = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
const change = async (el: HTMLSelectElement | HTMLInputElement, value: string) => {
  await act(async () => {
    const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(el, value);
    el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? 'change' : 'input', { bubbles: true }));
  });
  await settle();
};
const button = (label: RegExp) => Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((b) => label.test((b.textContent ?? '').trim()));
const click = async (el: Element | undefined | null) => {
  expect(el).toBeTruthy();
  await act(async () => (el as HTMLElement).click());
  await settle();
};

beforeEach(() => {
  saved = [];
  for (const f of Object.values(m)) f.mockReset();
  m.list.mockImplementation(async () => ({ providers: saved }));
  m.upsert.mockImplementation(async (input: AiProviderPublic) => {
    saved = [...saved.filter((p) => p.id !== input.id), { ...provider({}), ...input, hasKey: false }];
    return { id: input.id };
  });
  m.setKey.mockResolvedValue(undefined);
  onPick.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('choosing an AI provider for the Agent', () => {
  it('adds LM Studio from its preset, on this computer, and starts on the first model it lists', async () => {
    m.models.mockResolvedValue(['qwen3.5-9b', 'gemma-4-e4b']);
    await render();
    // Nothing set up yet: LM Studio's preset, with its address filled in and no key asked.
    expect(select('Provider').value).toBe('preset:lm-studio');
    expect(input('Address').value).toBe('http://localhost:1234/v1');
    expect(host.querySelector('input[aria-label="API key"]')).toBeNull();
    await click(button(/Connect/));
    expect(m.upsert).toHaveBeenCalledWith({
      id: 'lm-studio',
      name: 'LM Studio',
      protocol: 'openai',
      baseUrl: 'http://localhost:1234/v1',
      capabilities: ['chat'],
      enabled: true,
      allowLocal: true,
    });
    expect(m.setKey).not.toHaveBeenCalled();
    expect(m.models).toHaveBeenCalledWith('lm-studio');
    expect(onPick).toHaveBeenLastCalledWith({ provider: 'lm-studio', model: 'qwen3.5-9b', name: 'LM Studio' });
    expect(select('Provider').value).toBe('lm-studio');
    expect(Array.from(select('Model').options).map((o) => o.value)).toEqual(['qwen3.5-9b', 'gemma-4-e4b']);
  });

  it('asks an API provider for its key, keeps it with the desktop, and not on the open web', async () => {
    m.models.mockResolvedValue(['gpt-5.5']);
    await render();
    await change(select('Provider'), 'preset:openai');
    expect(input('Address').value).toBe('https://api.openai.com/v1');
    await click(button(/Connect/));
    expect(host.querySelector('[role="alert"]')!.textContent).toContain('Enter your OpenAI API key');
    expect(m.upsert).not.toHaveBeenCalled();
    await change(input('API key'), ' sk-test ');
    await click(button(/Connect/));
    expect(m.upsert).toHaveBeenCalledWith(expect.objectContaining({ id: 'openai', baseUrl: 'https://api.openai.com/v1', allowLocal: false }));
    expect(m.setKey).toHaveBeenCalledWith('openai', 'sk-test');
    expect(onPick).toHaveBeenLastCalledWith({ provider: 'openai', model: 'gpt-5.5', name: 'OpenAI' });
  });

  it('offers the providers already set up that the Agent can think with, and the model it is on', async () => {
    saved = [
      provider({ id: 'ollama', name: 'Ollama', baseUrl: 'http://localhost:11434/v1', model: 'gemma4' }),
      provider({ id: 'claude', name: 'Claude', protocol: 'anthropic', baseUrl: 'https://api.anthropic.com', allowLocal: false }),
      provider({ id: 'off', name: 'Off', enabled: false }),
    ];
    m.models.mockResolvedValue(['qwen3.5', 'gemma4', 'llama3.3']);
    await render({ provider: 'ollama', model: 'llama3.3', name: 'Ollama' });
    const names = Array.from(select('Provider').querySelectorAll('optgroup')[0].querySelectorAll('option')).map((o) => o.textContent);
    expect(names).toEqual(['Ollama']);
    expect(select('Provider').value).toBe('ollama');
    // The model it is on stays chosen.
    expect(onPick).toHaveBeenLastCalledWith({ provider: 'ollama', model: 'llama3.3', name: 'Ollama' });
    await change(select('Model'), 'gemma4');
    expect(onPick).toHaveBeenLastCalledWith({ provider: 'ollama', model: 'gemma4', name: 'Ollama' });
  });

  it('says a local server is not running, and tries again', async () => {
    saved = [provider({ id: 'lm-studio', name: 'LM Studio' })];
    m.models.mockRejectedValueOnce(new Error('502: request failed: connection refused'));
    await render();
    const alert = host.querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('LM Studio did not answer at http://localhost:1234/v1: is its server running, with a model loaded?');
    expect(onPick).not.toHaveBeenCalled();
    m.models.mockResolvedValue(['qwen3.5-9b']);
    await click(button(/Try again/));
    expect(onPick).toHaveBeenLastCalledWith({ provider: 'lm-studio', model: 'qwen3.5-9b', name: 'LM Studio' });
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('gives another OpenAI-compatible server an id of its own, named by its address', async () => {
    saved = [provider({ id: 'openai-compatible', name: 'first' })];
    m.models.mockResolvedValue(['m']);
    await render({ provider: 'openai-compatible', model: 'm', name: 'first' });
    await change(select('Provider'), 'preset:openai-compatible');
    await change(input('Address'), 'http://gpu-box.lan:8000/v1');
    await click(button(/Connect/));
    expect(m.upsert).toHaveBeenLastCalledWith(expect.objectContaining({ id: 'openai-compatible-2', name: 'gpu-box.lan:8000', allowLocal: true }));
  });
});

describe('the helpers', () => {
  it('tell a local address from one on the internet', () => {
    for (const url of ['http://localhost:1234/v1', 'http://127.0.0.1:11434', 'http://[::1]:8000', 'http://192.168.1.20:1234', 'http://10.0.0.5', 'http://box.local:8080', 'http://172.20.0.2'])
      expect(looksLocal(url), url).toBe(true);
    for (const url of ['https://api.openai.com/v1', 'https://openrouter.ai/api/v1', 'http://172.32.0.1', 'not a url']) expect(looksLocal(url), url).toBe(false);
  });

  it('pick a free id, the model to start on, and the providers the Agent can use', () => {
    expect(freeId('a', [])).toBe('a');
    expect(freeId('a', ['a', 'a-2'])).toBe('a-3');
    expect(startingModel(['a', 'b', 'c'], 'c', 'b')).toBe('c');
    expect(startingModel(['a', 'b'], 'gone', 'b')).toBe('b');
    expect(startingModel(['a', 'b'], null, null)).toBe('a');
    expect(startingModel([], 'a', 'b')).toBeNull();
    const list = [provider({ id: 'a' }), provider({ id: 'b', protocol: 'anthropic' }), provider({ id: 'c', enabled: false }), provider({ id: 'd', capabilities: ['transcription'] }), provider({ id: 'e', capabilities: ['chat', 'speech'] })];
    expect(agentCapable(list).map((p) => p.id)).toEqual(['a', 'e']);
  });
});
