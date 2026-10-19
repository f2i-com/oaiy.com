// The ChatGPT sign-in tells its caller when a sign-in started here finishes:
// the first-run wizard then sets the Agent to ChatGPT (CONTROL_API.md §3).
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ status: vi.fn(), startLogin: vi.fn(), push: vi.fn(), open: vi.fn() }));
vi.mock('./api', () => ({
  codex: { status: m.status, startLogin: m.startLogin, cancelLogin: vi.fn(), logout: vi.fn() },
  openExternal: m.open,
}));
vi.mock('./Toasts', () => ({ useToast: () => ({ push: m.push }) }));

import ChatGptConnector from './ChatGptConnector';

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.useFakeTimers();
  for (const f of Object.values(m)) f.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.useRealTimers();
});

describe('the ChatGPT sign-in', () => {
  it('calls onConnected once the account appears, and has no heading of its own when bare', async () => {
    m.status.mockResolvedValue({ available: true, connected: false });
    m.startLogin.mockResolvedValue({ loginId: 'l1', verificationUrl: 'https://auth.openai.com/device', userCode: 'ABCD-1234' });
    const onConnected = vi.fn();
    await act(async () => root.render(<ChatGptConnector bare onConnected={onConnected} />));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(host.querySelector('.section-title')).toBeNull();
    const signIn = Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes('Sign in with ChatGPT'))!;
    await act(async () => signIn.click());
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(m.open).toHaveBeenCalledWith('https://auth.openai.com/device');
    expect(host.textContent).toContain('ABCD-1234');
    m.status.mockResolvedValue({ available: true, connected: true, email: 'owner@example.com' });
    await act(async () => vi.advanceTimersByTimeAsync(3100));
    expect(onConnected).toHaveBeenCalledWith(expect.objectContaining({ connected: true, email: 'owner@example.com' }));
    expect(host.textContent).toContain('owner@example.com');
  });

  it('says the Codex app is not installed in words, with what to do, never the system’s error', async () => {
    m.status.mockResolvedValue({ available: false, connected: false, detail: "The Codex app isn't installed on this computer." });
    await act(async () => root.render(<ChatGptConnector bare />));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(host.textContent).toContain('The Codex app isn’t installed on this computer.');
    expect(host.textContent).toContain('Install Codex with OpenAI’s guide');
    expect(host.textContent).not.toContain('os error');
    expect(Array.from(host.querySelectorAll('button')).map((b) => b.textContent?.trim())).toEqual(['How to install Codex', 'Check again']);
  });

  it('shows another reason Codex cannot start as it came', async () => {
    m.status.mockResolvedValue({ available: false, connected: false, detail: 'cannot run the codex CLI: Permission denied (os error 13)' });
    await act(async () => root.render(<ChatGptConnector bare />));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(host.textContent).toContain('The Codex app could not be started on this computer.');
    expect(host.textContent).toContain('Permission denied (os error 13)');
  });
});
