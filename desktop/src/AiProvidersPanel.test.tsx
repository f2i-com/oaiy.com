// AI Providers panel — regression cover for the bugs that shipped in the first
// cut of this panel. Each test here maps to a real defect a UI review caught:
//   (a) saving an edit dropped category/capabilities and re-enabled a disabled
//       provider, (b) adding a duplicate id silently overwrote it, (c) the
//   destructive actions had no confirmation, (d) a failed first load hung on
//   "Loading providers…", (e) a partial save reported the wrong outcome.
//
// Convention copied from the FormLogic web app: raw react-dom/client + act, no
// Testing Library, mocks declared with vi.hoisted.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listMock, upsertMock, setKeyMock, deleteMock, testMock, pushMock } = vi.hoisted(() => ({
  listMock: vi.fn(),
  upsertMock: vi.fn(),
  setKeyMock: vi.fn(),
  deleteMock: vi.fn(),
  testMock: vi.fn(),
  pushMock: vi.fn(),
}));

vi.mock('./api', () => ({
  aiProviders: {
    list: listMock,
    upsert: upsertMock,
    setKey: setKeyMock,
    delete: deleteMock,
    test: testMock,
  },
}));
// Toasts pulls in @tauri-apps/plugin-notification at module scope; mocking it
// keeps that out of the test module graph AND gives us the push spy.
vi.mock('./Toasts', () => ({ useToast: () => ({ push: pushMock }) }));

import AiProvidersPanel from './AiProvidersPanel';

const PROVIDER = {
  id: 'openai',
  name: 'OpenAI',
  category: 'cloud',
  protocol: 'openai' as const,
  baseUrl: 'https://api.openai.com',
  model: 'gpt-x',
  capabilities: ['chat', 'embeddings'] as Array<'chat' | 'embeddings'>,
  enabled: false,
  allowLocal: false,
  hasKey: true,
};

let container: HTMLDivElement;
let root: Root;

async function mount() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(AiProvidersPanel));
  });
  await flush();
}

/** Let effects, the awaited list() and the focus setTimeout(…, 0) settle. */
async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function byLabel(label: string): HTMLElement {
  const el = container.querySelector<HTMLElement>(`[aria-label="${label}"]`);
  if (!el) throw new Error(`no element with aria-label "${label}"`);
  return el;
}

/** The <input> inside the .form-row whose <span> label matches `re`. */
function field(re: RegExp): HTMLInputElement {
  for (const row of Array.from(container.querySelectorAll('label.form-row'))) {
    const span = row.querySelector('span');
    if (span && re.test(span.textContent ?? '')) {
      const input = row.querySelector('input');
      if (input) return input as HTMLInputElement;
    }
  }
  throw new Error(`no field matching ${re}`);
}

/** Set a controlled input the way React sees it (native setter + input event),
 *  inside act() so the resulting state update is flushed. */
async function setValue(el: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
  await act(async () => {
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function click(el: HTMLElement) {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
  await flush();
}

async function submitForm() {
  const form = container.querySelector('form')!;
  await act(async () => {
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
  await flush();
}

beforeEach(() => {
  listMock.mockResolvedValue({ providers: [PROVIDER] });
  upsertMock.mockResolvedValue({ id: 'openai' });
  setKeyMock.mockResolvedValue(undefined);
  deleteMock.mockResolvedValue(undefined);
  testMock.mockResolvedValue({ ok: true });
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe('editing a provider (the fields the form does not show)', () => {
  it('preserves category, capabilities and the disabled state', async () => {
    await mount();
    await click(byLabel('Edit OpenAI'));
    await setValue(field(/Display name/), 'OpenAI Prod');
    await submitForm();

    expect(upsertMock).toHaveBeenCalledTimes(1);
    expect(upsertMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'openai',
        name: 'OpenAI Prod',
        category: 'cloud',
        capabilities: ['chat', 'embeddings'],
        enabled: false,
      }),
    );
    // A blank key field must keep the stored key rather than clearing it.
    expect(setKeyMock).not.toHaveBeenCalled();
  });

  it('locks the id (read-only) so an edit targets the same record', async () => {
    await mount();
    await click(byLabel('Edit OpenAI'));
    const id = field(/^ID/);
    expect(id.readOnly).toBe(true);
    expect(id.value).toBe('openai');

    await submitForm();
    expect(upsertMock).toHaveBeenCalledWith(expect.objectContaining({ id: 'openai' }));
  });

  it('sends the whole record with only enabled flipped when toggling from the card', async () => {
    await mount();
    await click(byLabel('Enable OpenAI'));
    expect(upsertMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'openai',
        category: 'cloud',
        capabilities: ['chat', 'embeddings'],
        model: 'gpt-x',
        enabled: true,
      }),
    );
  });
});

describe('adding a provider', () => {
  it('refuses a duplicate id before any API call', async () => {
    await mount();
    await click(container.querySelector<HTMLElement>('button.btn-secondary')!); // + Add provider
    await setValue(field(/^ID/), 'openai');
    await setValue(field(/Display name/), 'Dupe');
    await setValue(field(/Base URL/), 'https://x.example');
    await submitForm();

    expect(upsertMock).not.toHaveBeenCalled();
    expect(pushMock).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', title: expect.stringContaining('already exists') }),
    );
    // The form stays open so the user can correct the id.
    expect(container.querySelector('form')).not.toBeNull();
  });
});

describe('destructive actions', () => {
  it('Remove asks first, and cancelling deletes nothing', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    await mount();
    await click(byLabel('Remove OpenAI'));
    expect(deleteMock).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    await click(byLabel('Remove OpenAI'));
    expect(deleteMock).toHaveBeenCalledWith('openai');
    // The prompt warns that the stored key goes with it.
    const asked = confirmSpy.mock.calls.map((c) => String(c[0])).join(' ');
    expect(asked).toContain('OpenAI');
    expect(asked).toContain('stored API key is deleted');
  });

  it('Clear key asks first, and cancelling keeps the key', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    await mount();
    await click(byLabel('Clear the stored API key for OpenAI'));
    expect(setKeyMock).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    await click(byLabel('Clear the stored API key for OpenAI'));
    expect(setKeyMock).toHaveBeenCalledWith('openai', null);
  });

  it('removing the provider being edited closes the form so a save cannot resurrect it', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    await mount();
    await click(byLabel('Edit OpenAI'));
    expect(container.querySelector('form')).not.toBeNull();
    await click(byLabel('Remove OpenAI'));
    expect(container.querySelector('form')).toBeNull();
  });
});

describe('load failure', () => {
  it('offers a retry instead of hanging on "Loading providers…"', async () => {
    listMock.mockRejectedValueOnce(new Error('500: gateway down'));
    await mount();

    expect(container.textContent).not.toContain('Loading providers…');
    expect(container.textContent).toContain("Couldn't load providers.");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('500: gateway down');

    listMock.mockResolvedValue({ providers: [PROVIDER] });
    const retry = Array.from(container.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('Try again'),
    )!;
    await click(retry);
    expect(container.textContent).toContain('OpenAI');
  });
});

describe('save outcomes are reported honestly', () => {
  it('a provider that saves but whose key write fails reports a PARTIAL save', async () => {
    setKeyMock.mockRejectedValueOnce(new Error('401: invalid key'));
    await mount();
    await click(container.querySelector<HTMLElement>('button.btn-secondary')!);
    await setValue(field(/^ID/), 'newone');
    await setValue(field(/Display name/), 'New One');
    await setValue(field(/Base URL/), 'https://api.example.com');
    await setValue(field(/API key/), 'sk-abc');
    await submitForm();

    const alert = container.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('but its API key could not be stored');
    expect(alert).toContain('401: invalid key');
    expect(alert).not.toContain('Could not save');
  });

  it('a save that fails outright reports failure and keeps the form open', async () => {
    upsertMock.mockRejectedValueOnce(new Error('400: bad base url'));
    await mount();
    await click(container.querySelector<HTMLElement>('button.btn-secondary')!);
    await setValue(field(/^ID/), 'newone');
    await setValue(field(/Display name/), 'New One');
    await setValue(field(/Base URL/), 'https://api.example.com');
    await submitForm();

    const alert = container.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('Could not save');
    expect(alert).toContain('400: bad base url');
    expect(alert).not.toContain('but its API key');
    expect(setKeyMock).not.toHaveBeenCalled();
    expect(container.querySelector('form')).not.toBeNull();
    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
  });
});
