// The wizard's engine model card (SetupParts.tsx): with nothing chosen in Engines it lists the catalog's models for
// the group, the recommended one selected, says which the Agent can use its tools with and whether each fits the GPU,
// downloads the one chosen, and (for a language model, in the desktop's window) adds a file the computer already has.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ download: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, engines: { ...real.engines, download: m.download } };
});

import { EngineModelCard } from './SetupParts';
import type { EngineCatalog, EngineCatalogModel } from './api';

const model = (over: Partial<EngineCatalogModel>): EngineCatalogModel => ({
  id: 'x',
  group: 'llm',
  name: 'X',
  recommended: false,
  needs: [],
  installed: false,
  partial: false,
  download: null,
  ...over,
});

const catalog = (chosen: string | null, models?: EngineCatalogModel[]): EngineCatalog => ({
  running: true,
  groups: [{ id: 'llm', name: 'Chat' }],
  models: models ?? [
    model({ id: 'llama-3.2-3b', name: 'Llama 3.2 3B', sizeGb: 2, vramGb: 3, license: 'Llama 3.2' }),
    model({ id: 'qwen3.5-9b', name: 'Qwen3.5 9B', sizeGb: 5.7, vramGb: 8, recommended: true, agentTools: true, about: 'Strong.' }),
    model({ id: 'qwen3.8-27b', name: 'Qwen3.8 27B', sizeGb: 16.5, vramGb: 20, agentTools: true }),
    model({ id: 'z-image', group: 'image', name: 'Z Image', sizeGb: 6 }),
  ],
  defaults: { llm: chosen },
});

let host: HTMLDivElement;
let root: Root;
const invoke = vi.fn<(cmd: string, args: unknown) => Promise<unknown>>();
const onChanged = vi.fn();

async function settle() {
  for (let i = 0; i < 4; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render(props: Partial<React.ComponentProps<typeof EngineModelCard>> = {}) {
  await act(async () => {
    root.render(<EngineModelCard group="llm" catalog={catalog(null)} onChanged={onChanged} onOpenEngines={vi.fn()} {...props} />);
  });
  await settle();
}

const options = () => Array.from(host.querySelectorAll<HTMLLabelElement>('.setup-model-option'));
const button = (label: RegExp) => Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((b) => label.test((b.textContent ?? '').trim()));
const click = async (el: Element | undefined | null) => {
  expect(el).toBeTruthy();
  await act(async () => (el as HTMLElement).click());
  await settle();
};
const type = async (input: HTMLInputElement, value: string) => {
  await act(async () => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
    set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

beforeEach(() => {
  m.download.mockReset().mockResolvedValue({ running: true, downloads: [] });
  invoke.mockReset();
  onChanged.mockReset();
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

describe('the engine model card, with nothing chosen in Engines', () => {
  it('lists the group’s models, the recommended one first and selected, with what each can do and whether it fits', async () => {
    await render({ gpuGb: 16 });
    const names = options().map((o) => o.querySelector('strong')!.firstChild!.textContent!.trim());
    expect(names).toEqual(['Qwen3.5 9B', 'Llama 3.2 3B', 'Qwen3.8 27B']);
    expect(options()[0].querySelector('input')!.checked).toBe(true);
    expect(options()[0].textContent).toContain('Recommended');
    expect(options()[0].textContent).toContain('Agent tools');
    expect(options()[1].textContent).toContain('Chat only');
    expect(options()[0].textContent).toContain('5.7 GB download · needs 8 GB of GPU memory · fits your GPU');
    expect(options()[2].textContent).toContain('needs 20 GB of GPU memory · more than your GPU’s 16 GB');
    // Not the image model: another group's.
    expect(host.textContent).not.toContain('Z Image');
    expect(button(/^Download/)!.textContent).toContain('Download (5.7 GB)');
  });

  it('downloads the one chosen', async () => {
    await render();
    await click(options()[2].querySelector('input'));
    expect(options()[2].classList.contains('is-selected')).toBe(true);
    await click(button(/^Download/));
    expect(m.download).toHaveBeenCalledWith('qwen3.8-27b');
    expect(onChanged).toHaveBeenCalled();
  });

  it('holds the choice on a download under way, and shows how far it is', async () => {
    const downloading = catalog(null);
    downloading.models![0] = { ...downloading.models![0], download: { id: 'llama-3.2-3b', status: 'downloading', done: 1024, total: 4096 } };
    await render({ catalog: downloading });
    expect(options()[1].querySelector('input')!.checked).toBe(true);
    expect(options().every((o) => o.querySelector('input')!.disabled)).toBe(true);
    expect(host.querySelector('[role="progressbar"]')!.getAttribute('aria-label')).toBe('Downloading Llama 3.2 3B');
    expect(button(/^Download/)).toBeUndefined();
  });

  it('adds a model file the computer already has, through the desktop’s window', async () => {
    invoke.mockImplementation(async (cmd) => {
      if (cmd === 'pick_model_file') return 'E:\\models\\gemma-3-4b.gguf';
      if (cmd === 'add_engine_model') return { name: 'gemma-3-4b', architecture: 'gemma3', tools: false };
      throw new Error(cmd);
    });
    await render();
    const input = host.querySelector<HTMLInputElement>('input[aria-label="Model file"]')!;
    expect(button(/Use this file/)!.disabled).toBe(true);
    await click(button(/Choose…/));
    expect(input.value).toBe('E:\\models\\gemma-3-4b.gguf');
    await click(button(/Use this file/));
    expect(invoke).toHaveBeenCalledWith('add_engine_model', { path: 'E:\\models\\gemma-3-4b.gguf' });
    expect(onChanged).toHaveBeenCalled();
    // The engines chose it: the card says so, and that the Agent only chats with it.
    await render({ catalog: catalog('gemma-3-4b') });
    expect(host.textContent).toContain('Chosen in Engines');
    expect(host.querySelector('.setup-chat-only')!.textContent).toContain('cannot use its tools with it');
  });

  it('says why a file was refused, and keeps what was typed', async () => {
    invoke.mockRejectedValue('phi.gguf is a phi3 model, which OAIY’s engine does not run yet.');
    await render();
    const input = host.querySelector<HTMLInputElement>('input[aria-label="Model file"]')!;
    await type(input, '"D:\\phi.gguf"');
    await click(button(/Use this file/));
    expect(invoke).toHaveBeenCalledWith('add_engine_model', { path: '"D:\\phi.gguf"' });
    expect(host.querySelector('.setup-own-model [role="alert"]')!.textContent).toContain('does not run yet');
    expect(input.value).toBe('"D:\\phi.gguf"');
    expect(onChanged).not.toHaveBeenCalled();
  });

  it('says what a model of several GPUs needs on each, and marks no model for one kind of card', async () => {
    const two: EngineCatalog = catalog(null, [
      model({ id: 'deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash', sizeGb: 510, vramGb: 32, ramGb: 192, gpuCount: 2, recommended: true, agentTools: true }),
      model({ id: 'qwen3.5-9b', name: 'Qwen3.5 9B', sizeGb: 5.7, vramGb: 8, agentTools: true }),
    ]);
    await render({ catalog: two, gpuGb: 32 });
    const [deepseek, qwen] = options();
    // On each of its GPUs, and no word on fitting: only the largest GPU is known.
    expect(deepseek.textContent).toContain('needs 32 GB of GPU memory on each of 2 GPUs · 192 GB of RAM');
    expect(deepseek.textContent).not.toContain('fits your GPU');
    expect(qwen.textContent).toContain('fits your GPU');
    // One engine runs every model, on any GPU: the recommended one is chosen, and either can be.
    expect(host.textContent).not.toContain('NVIDIA');
    expect(deepseek.querySelector('input')!.checked).toBe(true);
    expect(deepseek.querySelector('input')!.disabled).toBe(false);
    expect(qwen.querySelector('input')!.disabled).toBe(false);
    await click(button(/^Download/));
    expect(m.download).toHaveBeenCalledWith('deepseek-v4.1-flash');
  });

  it('offers no file outside the desktop’s window, nor for another group', async () => {
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
    await render();
    expect(host.querySelector('.setup-own-model')).toBeNull();
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = { invoke };
    await render({ group: 'image' });
    expect(host.querySelector('.setup-own-model')).toBeNull();
    // One model: shown, with nothing to choose between.
    expect(options()).toHaveLength(1);
    expect(options()[0].querySelector('input')).toBeNull();
    expect(host.textContent).toContain('The catalog recommends:');
    expect(host.textContent).not.toContain('Agent tools');
  });
});
