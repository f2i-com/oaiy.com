import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ network: { requests: vi.fn(), approve: vi.fn(), deny: vi.fn() } }));
const toast = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock('./api', () => api);
vi.mock('./Toasts', () => ({ useToast: () => toast }));
import NetworkPairingPrompt from './NetworkPairingPrompt';

let container: HTMLDivElement;
let root: Root;
const request = { id: 'r1', nodeId: 'p1', name: 'Mac mini', address: '192.168.1.5', code: '123 456', fingerprint: 'aaaa', secondsLeft: 280 };

beforeEach(() => {
  vi.clearAllMocks();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});
const button = (label: string) => Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;

describe('NetworkPairingPrompt', () => {
  it('asks nothing while no primary is asking', async () => {
    await act(async () => root.render(<NetworkPairingPrompt waiting={0} onOpen={() => {}} />));
    expect(api.network.requests).not.toHaveBeenCalled();
    expect(container.textContent).toBe('');
  });

  it("shows a primary's request wherever the person is, with its code, and approves it there", async () => {
    api.network.requests.mockResolvedValueOnce({ requests: [request] }).mockResolvedValue({ requests: [] });
    api.network.approve.mockResolvedValue({ status: 'approved' });
    const open = vi.fn();
    await act(async () => root.render(<NetworkPairingPrompt waiting={1} onOpen={open} />));
    await act(async () => {});
    expect(container.textContent).toContain('Mac mini asks to use this computer');
    expect(container.textContent).toContain('123 456');
    await act(async () => button('Open Networking').click());
    expect(open).toHaveBeenCalled();
    await act(async () => button('Approve').click());
    expect(api.network.approve).toHaveBeenCalledWith('r1');
    expect(toast.push).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Approved Mac mini' }));
    expect(container.textContent).toBe('');
  });
});
