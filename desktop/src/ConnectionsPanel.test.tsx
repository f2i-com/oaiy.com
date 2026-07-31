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

const {
  relayStatusMock, setRelayMock, clearRelayMock,
  linkStatusMock, linkStartMock, unlinkMock,
  pendingMock, pairedMock, pushMock,
} = vi.hoisted(() => ({
  relayStatusMock: vi.fn(),
  setRelayMock: vi.fn(),
  clearRelayMock: vi.fn(),
  linkStatusMock: vi.fn(),
  linkStartMock: vi.fn(),
  unlinkMock: vi.fn(),
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
  link: {
    status: linkStatusMock,
    start: linkStartMock,
    unlink: unlinkMock,
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

const CONNECTOR = {
  id: 'acme',
  name: 'Acme Cloud',
  defaultBaseUrl: 'https://acme.example',
  scopes: ['data:read', 'data:write'],
};
const IDLE = { linked: false, attempt: { phase: 'idle' }, available: [CONNECTOR] };

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  pendingMock.mockResolvedValue({ pending: [] });
  pairedMock.mockResolvedValue({ paired: [] });
  relayStatusMock.mockResolvedValue({ configured: false, hasToken: false });
  linkStatusMock.mockResolvedValue(IDLE);
  linkStartMock.mockResolvedValue({ authorizeUrl: 'https://acme.example/authorize?x=1' });
  unlinkMock.mockResolvedValue(IDLE);
  vi.stubGlobal('confirm', vi.fn().mockReturnValue(true));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('ConnectionsPanel · Linked account', () => {
  it('offers whatever providers the host reports, with no hardcoded list', async () => {
    // The whole point of the descriptor design: a new provider is a JSON file,
    // and it must reach the picker without a frontend change.
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'idle' },
      available: [CONNECTOR, { id: 'other', name: 'Other Co', scopes: ['x'] }],
    });
    await mount();
    const options = Array.from(container.querySelectorAll('option')).map((o) => o.textContent);
    expect(options).toEqual(['Acme Cloud', 'Other Co']);
  });

  it('prefills the provider address and shows the scopes it will ask for', async () => {
    await mount();
    const input = container.querySelector('input[placeholder*="provider.example"]');
    expect((input as HTMLInputElement)?.value).toBe('https://acme.example');
    // Consent is meaningless if the user cannot see what is being granted.
    expect(container.textContent).toContain('data:read, data:write');
  });

  it('starts the ceremony with the chosen provider and address', async () => {
    await mount();
    await click('Link account');
    expect(linkStartMock).toHaveBeenCalledWith('acme', 'https://acme.example');
  });

  it('offers a manual link when the browser did not open', async () => {
    // A launcher can silently do nothing; without this the user is stuck on a
    // spinner with no way forward.
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'awaitingBrowser', authorizeUrl: 'https://acme.example/authorize?x=1' },
      available: [CONNECTOR],
    });
    await mount();
    const a = container.querySelector('a[href^="https://acme.example/authorize"]');
    expect(a).toBeTruthy();
    expect(container.textContent).toContain('Waiting for you to approve');
  });

  it('surfaces a failed attempt rather than looking idle', async () => {
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'failed', message: 'security check failed (state mismatch)' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('state mismatch');
  });

  it('shows the linked account and its granted scopes, never a credential', async () => {
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorId: 'acme',
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      accountName: 'Reception PC',
      grantedScopes: 'data:read',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('Acme Cloud');
    expect(container.textContent).toContain('Reception PC');
    expect(container.textContent).toContain('data:read');
  });

  it('warns that disconnecting is local-only before unlinking', async () => {
    // Telling the user the key is dead when it still works at the provider
    // would be worse than saying nothing.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    await click('Disconnect');
    const msg = (globalThis.confirm as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(msg).toContain('does not revoke it at the provider');
    expect(unlinkMock).toHaveBeenCalled();
  });
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
