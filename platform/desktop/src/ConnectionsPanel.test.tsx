// Connections panel — the Companion relay section.
//
// The relay holds a credential and decides whether an approved phone can join a
// call at all, so the cases worth pinning are the ones where getting it wrong is
// invisible: a key echoed back into the DOM, a disconnect with no confirmation,
// and a rejected address that leaves the user with no idea why.
//
// Convention as elsewhere in this app: raw react-dom/client + act, no Testing
// Library, mocks declared with vi.hoisted.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { relayStatusMock, setRelayMock, clearRelayMock, pendingMock, pairedMock, pushMock } =
  vi.hoisted(() => ({
    relayStatusMock: vi.fn(),
    setRelayMock: vi.fn(),
    clearRelayMock: vi.fn(),
    pendingMock: vi.fn(),
    pairedMock: vi.fn(),
    pushMock: vi.fn(),
  }));

vi.mock('./api', () => ({
  companion: {
    relayStatus: relayStatusMock,
    setRelay: setRelayMock,
    clearRelay: clearRelayMock,
  },
  pairing: {
    pending: pendingMock,
    paired: pairedMock,
    approve: vi.fn(),
    deny: vi.fn(),
    revoke: vi.fn(),
  },
}));
vi.mock('./Toasts', () => ({ useToast: () => ({ push: pushMock }) }));
vi.mock('./useCached', () => ({ peek: () => null, put: vi.fn() }));

import ConnectionsPanel from './ConnectionsPanel';

let container: HTMLDivElement;
let root: Root;

async function mount() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<ConnectionsPanel />);
  });
}

/** Click the first button whose visible text matches. */
async function click(text: string) {
  const button = Array.from(container.querySelectorAll('button')).find(
    (b) => b.textContent?.trim() === text,
  );
  if (!button) throw new Error(`no button "${text}" — have: ${
    Array.from(container.querySelectorAll('button')).map((b) => b.textContent?.trim()).join(', ')
  }`);
  await act(async () => {
    button.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
}

async function type(placeholder: string, value: string) {
  const input = container.querySelector<HTMLInputElement>(`input[placeholder*="${placeholder}"]`);
  if (!input) throw new Error(`no input matching "${placeholder}"`);
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  await act(async () => {
    setter.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  pendingMock.mockResolvedValue({ pending: [] });
  pairedMock.mockResolvedValue({ paired: [] });
  relayStatusMock.mockResolvedValue({ configured: false, hasToken: false });
  vi.stubGlobal('confirm', vi.fn().mockReturnValue(true));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('ConnectionsPanel · Companion relay', () => {
  it('says a phone cannot join a call when no relay is connected', async () => {
    await mount();
    // The honest consequence, not just "not configured" — a user who paired a
    // phone and saw nothing happen needs to be told why.
    expect(container.textContent).toContain('cannot join a call');
  });

  it('sends the typed relay settings and clears the key from the form', async () => {
    setRelayMock.mockResolvedValue({
      configured: true,
      baseUrl: 'https://formlogic.com/api/v1',
      appId: 'app_1',
      hasToken: true,
    });
    await mount();
    await type('https://formlogic.com', 'https://formlogic.com/api/v1');
    await type('the key your relay issued', 'flk_supersecret');
    await type('used when the plugin', 'app_1');
    await click('Connect relay');

    expect(setRelayMock).toHaveBeenCalledWith({
      baseUrl: 'https://formlogic.com/api/v1',
      token: 'flk_supersecret',
      appId: 'app_1',
    });
    // The key must not survive in the DOM after saving. It is write-only: the
    // server hands back only whether one is held.
    expect(container.innerHTML).not.toContain('flk_supersecret');
    expect(container.textContent).toContain('https://formlogic.com/api/v1');
  });

  it('omits an app id the user left blank rather than sending an empty one', async () => {
    // An empty string would be stored as a default app id and then sent on
    // every admission, scoping it to an app that does not exist.
    setRelayMock.mockResolvedValue({ configured: true, baseUrl: 'https://r.example', hasToken: true });
    await mount();
    await type('https://formlogic.com', 'https://r.example');
    await type('the key your relay issued', 'k');
    await click('Connect relay');
    expect(setRelayMock).toHaveBeenCalledWith({
      baseUrl: 'https://r.example',
      token: 'k',
      appId: undefined,
    });
  });

  it('shows the server’s reason when the address is rejected', async () => {
    setRelayMock.mockRejectedValue(new Error('400: the upstream base URL must be https'));
    await mount();
    await type('https://formlogic.com', 'http://relay.example');
    await type('the key your relay issued', 'k');
    await click('Connect relay');
    // Without this the form just… does nothing, and the user retypes the same
    // address forever.
    expect(container.textContent).toContain('must be https');
  });

  it('asks before disconnecting, and says approved phones survive it', async () => {
    relayStatusMock.mockResolvedValue({
      configured: true,
      baseUrl: 'https://formlogic.com/api/v1',
      hasToken: true,
    });
    clearRelayMock.mockResolvedValue({ configured: false, hasToken: false });
    await mount();
    await click('Disconnect');

    const message = (globalThis.confirm as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(message).toContain('stay approved');
    expect(clearRelayMock).toHaveBeenCalled();
    expect(container.textContent).toContain('cannot join a call');
  });

  it('does not disconnect when the confirm is declined', async () => {
    relayStatusMock.mockResolvedValue({ configured: true, baseUrl: 'https://r.example', hasToken: true });
    vi.stubGlobal('confirm', vi.fn().mockReturnValue(false));
    await mount();
    await click('Disconnect');
    expect(clearRelayMock).not.toHaveBeenCalled();
  });

  it('flags a stored relay that has lost its credential', async () => {
    // `configured` without `hasToken` means the file survived but the key did
    // not — every admission will fail, and nothing else on screen would say so.
    relayStatusMock.mockResolvedValue({
      configured: true,
      baseUrl: 'https://r.example',
      hasToken: false,
    });
    await mount();
    expect(container.textContent).toContain('reconnect');
  });
});
