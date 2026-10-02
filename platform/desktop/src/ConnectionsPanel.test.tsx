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
  linkStatusMock, linkStartMock, unlinkMock, removeCopiesMock, cancelLinkMock,
  pendingMock, pairedMock, pushMock,
} = vi.hoisted(() => ({
  relayStatusMock: vi.fn(),
  setRelayMock: vi.fn(),
  clearRelayMock: vi.fn(),
  linkStatusMock: vi.fn(),
  linkStartMock: vi.fn(),
  unlinkMock: vi.fn(),
  removeCopiesMock: vi.fn(),
  cancelLinkMock: vi.fn(),
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
    removeCopies: removeCopiesMock,
    cancel: cancelLinkMock,
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
  removeCopiesMock.mockResolvedValue(IDLE);
  cancelLinkMock.mockResolvedValue({ ...IDLE, attempt: { phase: 'cancelled' } });
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

  it('lets the user cancel while waiting on the browser', async () => {
    // Without this the only way out of a link they changed their mind about is
    // to wait out the five-minute timeout.
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'awaitingBrowser', authorizeUrl: 'https://acme.example/authorize?x=1' },
      available: [CONNECTOR],
    });
    await mount();
    await click('Cancel');
    expect(cancelLinkMock).toHaveBeenCalled();
    // Cancelling is not an error — it must not be dressed as one.
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.textContent).toContain('Linking cancelled');
    // Cancelling closes the loopback port the provider redirects to, so an
    // approval page left open leads nowhere. Approving it yields a bare
    // ERR_CONNECTION_REFUSED that reads as a broken app.
    expect(container.textContent).toContain('close it');
  });

  it('does not describe the wait with mangled escape text', async () => {
    // Regression: a patch script wrote literal \\u2026 / \\u2019 sequences into
    // the JSX, so the panel rendered "browser\\u2026" and "Didn\\u2019t".
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'awaitingBrowser', authorizeUrl: 'https://acme.example/authorize' },
      available: [CONNECTOR],
    });
    await mount();
    const text = container.textContent ?? '';
    expect(text).not.toMatch(/\\u[0-9a-fA-F]{4}/);
    expect(text).toContain('Waiting for you to approve this in your browser');
    expect(text).toContain('No tab opened?');
  });

  it('offers a way to dismiss a stale failure', async () => {
    // Otherwise the error banner has no exit short of a successful link.
    linkStatusMock.mockResolvedValue({
      linked: false,
      attempt: { phase: 'failed', message: 'the provider refused the link' },
      available: [CONNECTOR],
    });
    await mount();
    await click('Dismiss');
    expect(cancelLinkMock).toHaveBeenCalled();
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

  it('says when the desktop last checked in with the provider', async () => {
    // Presence is judged by the provider from how recently we spoke, so this is
    // the one number that explains a link looking fine here and offline there.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      lastHeartbeatAt: '2026-07-31T13:14:57Z',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('Checked in');
  });

  it('explains a failing check-in rather than just looking healthy', async () => {
    // The exact confusion this fixes: the panel said "linked" while the
    // provider showed "No Desktop", with nothing connecting the two.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      heartbeatError: 'the provider no longer accepts this desktop’s key — link again',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('Not checking in');
    expect(container.textContent).toContain('link again');
  });

  it('explains a failing command lane rather than just looking linked', async () => {
    // The heartbeat's twin, and the one that fails more confusingly. Checking in
    // only says this machine is HERE; everything the user then asks for on the
    // provider's website travels the command lane. A lane that is erroring shows
    // up nowhere on this machine and, over there, only as "no desktop picked it
    // up in time" — which reads as a broken connection and sends them looking at
    // their network instead of at the reason.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      lastHeartbeatAt: '2026-08-01T00:14:57Z',
      heartbeatSupported: true,
      relaySupported: true,
      relayError: 'claim refused: HTTP 500',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    // Both lanes are reported, and independently: checking in is fine here.
    expect(container.textContent).toContain('Checked in');
    expect(container.textContent).toContain('Not receiving commands');
    expect(container.textContent).toContain('HTTP 500');
  });

  it('says the command lane is alive, so “no error” is not just silence', async () => {
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      relaySupported: true,
      lastRelayAt: '2026-08-01T00:14:57Z',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('Listening for commands');
  });

  it('claims nothing about lanes the provider does not have', async () => {
    // A connector with no relay must read as ABSENT, not as pending — otherwise
    // it sits under a "connecting…" that is never going to resolve.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      heartbeatSupported: false,
      relaySupported: false,
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    const text = container.textContent ?? '';
    expect(text).not.toContain('command lane');
    expect(text).not.toContain('Listening for commands');
    expect(text).not.toContain('Waiting for the first check-in');
  });

  it('keeps asking, so a lane that breaks after the screen opens still says so', async () => {
    // Without polling the whole thing is decoration: the status is fetched once
    // at mount, so a link that was healthy when the panel opened goes on looking
    // healthy through every failure that follows.
    const healthy = {
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      relaySupported: true,
      lastRelayAt: '2026-08-01T00:14:57Z',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    };
    linkStatusMock.mockResolvedValue(healthy);
    await mount();
    expect(container.textContent).toContain('Listening for commands');

    linkStatusMock.mockResolvedValue({
      ...healthy,
      lastRelayAt: undefined,
      relayError: 'could not reach the relay',
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3500);
    });
    expect(container.textContent).toContain('Not receiving commands');
  });

  it('does not leave a momentary fetch failure on screen forever', async () => {
    // The regression polling would otherwise introduce: one failed poll paints a
    // banner with no dismiss, and it outlives the problem it described.
    linkStatusMock.mockRejectedValueOnce(new Error('failed to fetch'));
    await mount();
    expect(container.textContent).toContain('failed to fetch');

    linkStatusMock.mockResolvedValue(IDLE);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3500);
    });
    expect(container.textContent).not.toContain('failed to fetch');
  });

  it('shows the storage-node fingerprint the way the provider shows it', async () => {
    // The approval ceremony is the user comparing the two strings by eye. The
    // provider's site groups the first 24 hex characters in fours; formatting
    // them differently here makes the check tedious enough to skip, which is
    // the only thing standing between an approval and approving a wrong key.
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      dataNodeSupported: true,
      dataNode: {
        fingerprint: 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90',
        status: 'pending',
        approved: false,
        keyGeneration: 1,
      },
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).toContain('a1b2 c3d4 e5f6 0718 293a 4b5c');
    // …and it must say what the user has to DO, not just report a state.
    expect(container.textContent).toContain('waiting for you to approve');
  });

  it('says nothing about a storage node the provider does not offer', async () => {
    linkStatusMock.mockResolvedValue({
      linked: true,
      connectorName: 'Acme Cloud',
      baseUrl: 'https://acme.example',
      attempt: { phase: 'linked' },
      available: [CONNECTOR],
    });
    await mount();
    expect(container.textContent).not.toContain('Storage node');
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

  const LINKED = {
    linked: true,
    connectorName: 'Acme Cloud',
    baseUrl: 'https://acme.example',
    accountName: 'Reception PC',
    attempt: { phase: 'linked' },
    available: [CONNECTOR],
  };

  it('says Disconnected only when the key was forgotten', async () => {
    linkStatusMock.mockResolvedValue(LINKED);
    unlinkMock.mockResolvedValue(IDLE);
    await mount();
    await click('Disconnect');
    expect(pushMock).toHaveBeenCalledWith({ kind: 'success', title: 'Disconnected Acme Cloud' });
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });

  it('does not say Disconnected when the host could not remove the stored link, and keeps the account on the card', async () => {
    // A program holding account.json without delete sharing (a scanner, an indexer) makes the removal fail. The
    // host answers with the link still in place and says why; a green "Disconnected" would tell the owner the
    // key is gone when it is on the disk and links this desktop again at the next start.
    const why = 'the link could not be forgotten: link/account.json could not be removed (Access is denied. (os error 5)). It is still linked here.';
    linkStatusMock.mockResolvedValue(LINKED);
    unlinkMock.mockResolvedValue({ ...LINKED, linkError: { file: 'link/account.json', message: why, kind: 'notForgotten' } });
    await mount();
    await click('Disconnect');

    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
    expect(pushMock.mock.calls.some(([t]) => /^Disconnected /.test(t.title) && !/but not all/.test(t.title))).toBe(false);
    expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', title: 'Could not disconnect Acme Cloud', body: why }));
    // The reason is said once, in the banner for what the owner just did: the status carries it too, and a second
    // banner with the same words under the first would be noise (and one that says it without the other is a bug).
    const alerts = container.querySelectorAll('[role="alert"]');
    expect(alerts).toHaveLength(1);
    expect(alerts[0].textContent).toContain('could not be removed');
    expect(container.querySelector('[data-link-error]')).toBeNull();
    // Still linked: the card, its account and its Disconnect button are there to try again.
    expect(container.textContent).toContain('Reception PC');
    expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Disconnect')).toBe(true);
    expect(container.querySelector('[data-copies-left]')).toBeNull();
  });

  it('does not say Disconnected when the host answers with the account still linked and gives no reason', async () => {
    // Whatever the host leaves out, an answer that still has the account linked is not a disconnect: the owner is
    // told so in a sentence of the panel's own, and nothing green is shown.
    linkStatusMock.mockResolvedValue(LINKED);
    unlinkMock.mockResolvedValue(LINKED);
    await mount();
    await click('Disconnect');

    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
    expect(pushMock).toHaveBeenCalledWith({ kind: 'error', title: 'Could not disconnect Acme Cloud', body: 'Acme Cloud is still linked on this computer.' });
    const alerts = container.querySelectorAll('[role="alert"]');
    expect(alerts).toHaveLength(1);
    expect(alerts[0].textContent).toBe('Acme Cloud is still linked on this computer.');
    expect(container.textContent).toContain('Reception PC');
  });

  it('does not say Disconnected when the file could not be moved aside, though no account is shown for it', async () => {
    // A stored link that could not be read was held by a program that also kept it from being moved: it was not forgotten, and
    // nothing is shown linked because the file could not be read. The host says what it is (`notForgotten`); the panel does not
    // take "not linked" for "forgotten".
    const why = 'the link could not be forgotten: link/account.json could not be moved aside (Access is denied. (os error 5)); it has not been changed. It is still stored here, and would be linked again at the next start. Close whatever has it open and try again.';
    linkStatusMock.mockResolvedValue(LINKED);
    unlinkMock.mockResolvedValue({ ...IDLE, linkError: { file: 'link/account.json', message: why, kind: 'notForgotten' } });
    await mount();
    await click('Disconnect');

    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
    expect(pushMock).toHaveBeenCalledWith({ kind: 'error', title: 'Could not disconnect Acme Cloud', body: why });
    expect(pushMock.mock.calls.some(([t]) => /but not all/.test(t.title))).toBe(false);
    const alerts = container.querySelectorAll('[role="alert"]');
    expect(alerts).toHaveLength(1);
    expect(alerts[0].textContent).toContain('could not be moved aside');
    expect(container.querySelector('[data-copies-left]')).toBeNull();
  });

  it('calls a link that could not be changed that, with no "nothing was deleted" and no account shown for it', async () => {
    const why = 'the link could not be forgotten: link/account.json could not be moved aside (held); it has not been changed.';
    linkStatusMock.mockResolvedValue({ ...IDLE, linkError: { file: 'link/account.json', message: why, kind: 'notForgotten' } });
    await mount();
    const alert = container.querySelector('[data-link-error]');
    expect(alert?.textContent).toContain('The link could not be changed.');
    expect(alert?.textContent).toContain(why);
    expect(alert?.textContent).not.toContain('The stored link could not be used.');
    expect(alert?.textContent).not.toContain('Nothing was deleted');
  });

  it('says so, and offers Remove copies, when the link was forgotten and a copy of its key could not be removed', async () => {
    const why = 'Disconnected, but a copy of the old key could not be removed: account.json.corrupt (Access is denied. (os error 5)). Close the program that holds it and press Remove copies.';
    linkStatusMock.mockResolvedValue(LINKED);
    unlinkMock.mockResolvedValue({ ...IDLE, linkError: { file: 'link/account.json', message: why, kind: 'copiesLeft' } });
    await mount();
    await click('Disconnect');
    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
    expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', title: 'Disconnected Acme Cloud, but not all of it', body: why }));

    // One banner, its own: the words, and the button that tries again. Not the "stored link could not be used"
    // one (nothing is stored and nothing is to be read again), and not a second one with the same words and no button.
    const alerts = container.querySelectorAll('[role="alert"]');
    expect(alerts).toHaveLength(1);
    const banner = container.querySelector('[data-copies-left]')!;
    expect(banner).toBe(alerts[0]);
    expect(banner.textContent).toContain('a copy of the old key could not be removed: account.json.corrupt');
    expect(banner.textContent).not.toContain('The stored link could not be used');
    expect(banner.textContent).not.toContain('Nothing was deleted');
    expect(container.querySelector('[data-link-error]')).toBeNull();
    expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Remove copies')).toBe(true);
    // It is disconnected: the account can be linked again, and the card is gone.
    expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Link account')).toBe(true);
    expect(container.textContent).not.toContain('Reception PC');
  });

  it('removes the copies on Remove copies, asks for nothing else to be forgotten, and says so when it worked', async () => {
    const left = { file: 'link/account.json', message: 'Disconnected, but a copy of the old key could not be removed: account.json.corrupt (held). Close the program that holds it and press Remove copies.', kind: 'copiesLeft' };
    linkStatusMock.mockResolvedValue({ ...IDLE, linkError: left });
    removeCopiesMock.mockResolvedValue(IDLE);
    await mount();
    expect(container.querySelector('[data-copies-left]')).not.toBeNull();

    await click('Remove copies');
    // The copies only: not the call that forgets a link, and nothing asked of the owner first.
    expect(removeCopiesMock).toHaveBeenCalledTimes(1);
    expect(unlinkMock).not.toHaveBeenCalled();
    expect(globalThis.confirm).not.toHaveBeenCalled();
    expect(pushMock).toHaveBeenCalledWith({ kind: 'success', title: 'Copies of the old key removed' });
    // And what was said is gone with them, the plain panel of a desktop nobody linked is what is left.
    expect(container.querySelector('[data-copies-left]')).toBeNull();
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Remove copies')).toBe(false);
  });

  it('keeps saying it when a copy is still held, and lets the owner press Remove copies again', async () => {
    const left = { file: 'link/account.json', message: 'Disconnected, but a copy of the old key could not be removed: account.json.corrupt (held). Close the program that holds it and press Remove copies.', kind: 'copiesLeft' };
    linkStatusMock.mockResolvedValue({ ...IDLE, linkError: left });
    removeCopiesMock.mockResolvedValue({ ...IDLE, linkError: left });
    await mount();

    await click('Remove copies');
    expect(pushMock).toHaveBeenCalledWith({ kind: 'error', title: 'A copy of the old key is still there', body: left.message });
    expect(pushMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }));
    expect(container.querySelector('[data-copies-left]')?.textContent).toContain('account.json.corrupt (held)');
    expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
    // Not left "Removing…" and disabled.
    const button = Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Remove copies');
    expect(button?.disabled).toBe(false);

    // Let go of it, press again: it is gone.
    removeCopiesMock.mockResolvedValue(IDLE);
    await click('Remove copies');
    expect(removeCopiesMock).toHaveBeenCalledTimes(2);
    expect(container.querySelector('[data-copies-left]')).toBeNull();
  });

  it('does not send Remove copies again while it is running', async () => {
    const left = { file: 'link/account.json', message: 'Disconnected, but a copy of the old key could not be removed: account.json.corrupt (held). Close the program that holds it and press Remove copies.', kind: 'copiesLeft' };
    linkStatusMock.mockResolvedValue({ ...IDLE, linkError: left });
    let release: (status: unknown) => void = () => {};
    removeCopiesMock.mockReturnValue(new Promise((resolve) => { release = resolve; }));
    await mount();

    await click('Remove copies');
    const running = Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Removing…');
    expect(running?.disabled).toBe(true);
    await act(async () => {
      running?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(removeCopiesMock).toHaveBeenCalledTimes(1);

    await act(async () => release(IDLE));
    expect(container.querySelector('[data-copies-left]')).toBeNull();
  });

  it('shows the host’s answer when Remove copies cannot even be asked, and keeps the button', async () => {
    const left = { file: 'link/account.json', message: 'Disconnected, but a copy of the old key could not be removed: account.json.corrupt (held). Close the program that holds it and press Remove copies.', kind: 'copiesLeft' };
    linkStatusMock.mockResolvedValue({ ...IDLE, linkError: left });
    removeCopiesMock.mockRejectedValue(new Error('the desktop did not answer'));
    await mount();
    await click('Remove copies');
    const texts = Array.from(container.querySelectorAll('[role="alert"]')).map((a) => a.textContent);
    expect(texts.some((t) => t?.includes('the desktop did not answer'))).toBe(true);
    expect(container.querySelector('[data-copies-left]')).not.toBeNull();
    const button = Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Remove copies');
    expect(button?.disabled).toBe(false);
  });

  it('offers Remove copies for a copy left behind and for nothing else', async () => {
    const files = { file: 'link/account.json', message: 'x' };
    for (const [what, status] of [
      ['an unusable file', { ...IDLE, linkError: { ...files, kind: 'unusable' } }],
      ['a link not forgotten', { ...LINKED, linkError: { ...files, kind: 'notForgotten' } }],
      ['a link', LINKED],
      ['no link', IDLE],
    ] as const) {
      linkStatusMock.mockResolvedValue(status);
      await mount();
      expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Remove copies'), what).toBe(false);
      await act(async () => root.unmount());
      container.remove();
    }
  });

  it('shows why a stored link is not in use, rather than looking as if nobody had linked this desktop', async () => {
    // An unreadable or newer-shaped account.json used to read as "not linked" with no word said: the link was on
    // the disk and gone from the screen. The file is kept, and the panel says what is wrong with it.
    linkStatusMock.mockResolvedValue({
      ...IDLE,
      linkError: { file: 'link/account.json', message: 'link/account.json is not a link this version of OAIY understands (an unknown field, line 1, column 99). It has not been changed.', kind: 'unusable' },
    });
    await mount();
    const alert = container.querySelector('[data-link-error]');
    expect(alert?.getAttribute('role')).toBe('alert');
    expect(alert?.textContent).toContain('The stored link could not be used.');
    expect(alert?.textContent).toContain('an unknown field');
    expect(alert?.textContent).toContain('Nothing was deleted');
    // It can still be linked again from here.
    expect(Array.from(container.querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Link account')).toBe(true);
  });

  it('says that a link that could not be changed is still there, in words that are not the stored link one', async () => {
    // A linked desktop with an error of its own (the status says it after a forget that failed and the panel was
    // opened again): its heading is not "stored link could not be used", and "Nothing was deleted. Link again" is
    // not said to someone whose link is working.
    const why = 'the link could not be forgotten: link/account.json could not be removed (Access is denied. (os error 5)). It is still linked here.';
    linkStatusMock.mockResolvedValue({ ...LINKED, linkError: { file: 'link/account.json', message: why, kind: 'notForgotten' } });
    await mount();
    const alert = container.querySelector('[data-link-error]');
    expect(alert?.textContent).toContain('The link could not be changed.');
    expect(alert?.textContent).toContain(why);
    expect(alert?.textContent).not.toContain('The stored link could not be used.');
    expect(alert?.textContent).not.toContain('Nothing was deleted');
    expect(container.querySelector('[data-copies-left]')).toBeNull();
  });

  it('says nothing about a stored link that is fine', async () => {
    linkStatusMock.mockResolvedValue(LINKED);
    await mount();
    expect(container.querySelector('[data-link-error]')).toBeNull();
    expect(container.querySelector('[data-copies-left]')).toBeNull();
    expect(container.textContent).not.toContain('could not be used');
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
