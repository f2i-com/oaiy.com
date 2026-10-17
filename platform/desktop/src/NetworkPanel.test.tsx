import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  status: vi.fn(),
  setSettings: vi.fn(),
  discover: vi.fn(),
  startPairing: vi.fn(),
  confirmPairing: vi.fn(),
  requests: vi.fn(),
  approve: vi.fn(),
  deny: vi.fn(),
  nodes: vi.fn(),
  unpair: vi.fn(),
  pin: vi.fn(),
  unpin: vi.fn(),
}));
vi.mock('./api', () => ({ network: api }));
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import NetworkPanel from './NetworkPanel';

let container: HTMLDivElement;
let root: Root;

const base = {
  nodeId: 'n1',
  name: 'Mac mini',
  fingerprint: 'abcd 1234 ef56 7890',
  listening: null,
  addresses: [],
  discovery: true,
  protocol: 1,
  waiting: 0,
  found: 0,
  paired: 0,
};
const off = { ...base, enabled: false, role: 'standalone', nodeId: null, fingerprint: null, settings: { enabled: false, role: 'standalone', name: '', port: 17975, bind: '' } };
const worker = { ...base, enabled: true, role: 'worker', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'], settings: { enabled: true, role: 'worker', name: 'Rig', port: 17975, bind: '' } };
const primary = { ...base, enabled: true, role: 'primary', settings: { enabled: true, role: 'primary', name: '', port: 17975, bind: '' } };
const node = {
  nodeId: 'w1',
  name: 'Rig',
  role: 'worker',
  address: '192.168.1.20:17975',
  fingerprint: '1111 2222 3333 4444',
  pairedAtMs: 1,
  online: true,
  lastSeenMs: 2,
  lastError: null,
  node: { os: 'windows' },
  enginesError: null,
  devices: [{ id: 'gpu0', name: 'RTX 5090', memory_total_mb: 32768, memory_used_mb: 4096 }, { id: 'gpu1', name: 'RTX 3070', memory_total_mb: null }],
  models: [{ id: 'engine:image:node.w1/flux', kind: 'image', model: 'flux', state: 'kept', device: 'gpu0', pinned: false }],
};

beforeEach(() => {
  vi.clearAllMocks();
  api.nodes.mockResolvedValue({ nodes: [] });
  api.requests.mockResolvedValue({ requests: [] });
  api.discover.mockResolvedValue({ discovery: true, found: [] });
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
});
async function mount() {
  await act(async () => root.render(<NetworkPanel />));
  await act(async () => {});
}
const button = (label: string) => Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.trim() === label)!;

describe('NetworkPanel', () => {
  it('is off until switched on, and asks for nothing else while off', async () => {
    api.status.mockResolvedValue(off);
    await mount();
    expect(container.textContent).toContain('Off');
    expect(button('Switch on')).toBeTruthy();
    expect(api.nodes).not.toHaveBeenCalled();
    expect(api.discover).not.toHaveBeenCalled();
    api.setSettings.mockResolvedValue(primary);
    api.status.mockResolvedValue(primary);
    await act(async () => button('Switch on').click());
    expect(api.setSettings).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, role: 'primary' }));
  });

  it("on a worker, shows a primary's request with its code and approves only that one", async () => {
    api.status.mockResolvedValue(worker);
    api.requests.mockResolvedValue({ requests: [{ id: 'r1', nodeId: 'p1', name: 'Mac mini', address: '192.168.1.5', code: '123 456', fingerprint: 'aaaa', secondsLeft: 280 }] });
    api.approve.mockResolvedValue({ status: 'approved' });
    await mount();
    expect(container.textContent).toContain('192.168.1.20:17975');
    const card = container.querySelector('[data-testid="network-request"]')!;
    expect(card.textContent).toContain('123 456');
    await act(async () => button('Approve').click());
    expect(api.approve).toHaveBeenCalledWith('r1');
    expect(api.deny).not.toHaveBeenCalled();
  });

  it('on a primary, pairs by address showing the code, the worker approving and then this side', async () => {
    vi.useFakeTimers();
    api.status.mockResolvedValue(primary);
    api.startPairing.mockResolvedValue({ pairingId: 'x', code: '654 321', status: 'pending', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.20:17975', fingerprint: '1111' } });
    api.confirmPairing
      .mockResolvedValueOnce({ status: 'pending', code: '654 321' })
      .mockResolvedValueOnce({ status: 'approved_there', code: '654 321' })
      .mockResolvedValueOnce({ status: 'approved', node });
    await mount();
    const input = container.querySelector('input[aria-label="Worker address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => input.closest('form')!.requestSubmit());
    expect(api.startPairing).toHaveBeenCalledWith({ address: '192.168.1.20' });
    expect(container.querySelector('[data-testid="network-pairing"]')!.textContent).toContain('654 321');
    api.nodes.mockResolvedValue({ nodes: [node] });
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2100); });
    expect(api.confirmPairing).toHaveBeenCalledTimes(2);
    // approved there: nothing is paired until this side says the worker showed the same code
    expect(container.querySelector('[data-testid="network-pairing"]')!.textContent).toContain('Did its screen show this same code?');
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.confirmPairing).toHaveBeenCalledTimes(2);
    await act(async () => button('Yes, pair').click());
    expect(api.confirmPairing).toHaveBeenLastCalledWith('x', true);
    expect(container.textContent).toContain('Paired with Rig');
    const card = container.querySelector('[data-testid="network-node"]')!;
    expect(card.textContent).toContain('Online');
    const devices = Array.from(card.querySelectorAll('.network-device')).map((d) => d.textContent);
    expect(devices[0]).toContain('RTX 5090');
    expect(devices[0]).toContain('4.0 of 32.0 GB in use');
    expect(devices[1]).toContain('RTX 3070');
    expect(devices[1]).toContain('memory unknown');
    // the address typed is cleared once it is paired
    expect(input.value).toBe('');
    expect(card.textContent).toContain('flux');
  });

  it("pins a worker's model by its id, shows the pin loading, and unpins it by its name there", async () => {
    api.status.mockResolvedValue(primary);
    const pinned = { ...node.models[0], id: 'engine:image:node.w1/moss', model: 'moss', state: 'not loaded', device: null, pinned: true, pinState: 'warming' };
    const loose = { id: 'engine:image:node.w1/klein', kind: 'image', model: 'klein', state: 'not loaded', device: null, pinned: false };
    api.nodes.mockResolvedValue({ nodes: [{ ...node, models: [node.models[0], pinned, loose] }] });
    api.pin.mockResolvedValue({});
    api.unpin.mockResolvedValue({});
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    const rows = Array.from(card.querySelectorAll('.network-model'));
    const moss = rows.find((r) => r.textContent?.includes('moss'))!;
    expect(moss.textContent).toContain('Not loaded');
    expect(moss.textContent).toContain('Pinned · loading');
    // (a pinned model is listed among the loaded ones, the rest folded away)
    expect(card.querySelector('details')!.textContent).toContain('klein');
    const buttonOf = (model: string) => rows.find((r) => r.textContent?.includes(model))!.querySelector('button')!;
    const asked = api.nodes.mock.calls.length;
    await act(async () => buttonOf('moss').click());
    expect(api.unpin).toHaveBeenCalledWith('w1', 'moss');
    // (the page asks again straight away, not at the next poll)
    expect(api.nodes.mock.calls.length).toBeGreaterThan(asked);
    await act(async () => buttonOf('klein').click());
    expect(api.pin).toHaveBeenCalledWith('w1', 'engine:image:node.w1/klein');
  });

  it('withdraws a pairing it stops waiting for, so the worker no longer shows it', async () => {
    vi.useFakeTimers();
    api.status.mockResolvedValue(primary);
    api.startPairing.mockResolvedValue({ pairingId: 'x', code: '654 321', status: 'pending', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.20:17975', fingerprint: '1111' } });
    api.confirmPairing.mockResolvedValue({ status: 'declined' });
    await mount();
    const input = container.querySelector('input[aria-label="Worker address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => input.closest('form')!.requestSubmit());
    await act(async () => button('Stop waiting').click());
    expect(api.confirmPairing).toHaveBeenCalledWith('x', false);
    expect(container.querySelector('[data-testid="network-pairing"]')).toBeNull();
    // and nothing asks after it any more
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.confirmPairing).toHaveBeenCalledTimes(1);
  });

  it("saves only a change, and a worker's port only when it is one", async () => {
    api.status.mockResolvedValue(worker);
    await mount();
    expect(button('Save').disabled).toBe(true);
    const port = Array.from(container.querySelectorAll('label')).find((l) => l.textContent?.startsWith('Port'))!.querySelector('input')!;
    const type = async (value: string) =>
      act(async () => {
        const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
        set.call(port, value);
        port.dispatchEvent(new Event('input', { bubbles: true }));
      });
    await type('80');
    expect(button('Save').disabled).toBe(true);
    expect(container.textContent).toContain('A port from 1024 to 65535');
    await type('18000');
    expect(button('Save').disabled).toBe(false);
    api.setSettings.mockResolvedValue(worker);
    await act(async () => button('Save').click());
    expect(api.setSettings).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, role: 'worker', port: 18000 }));
  });

  it('shows an offline worker with why, and unpairs it only when confirmed', async () => {
    api.status.mockResolvedValue(primary);
    api.nodes.mockResolvedValue({ nodes: [{ ...node, online: false, lastError: 'connection refused' }] });
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    expect(card.textContent).toContain('Offline');
    expect(card.textContent).toContain('connection refused');
    // what it last said is shown as that, its models' states left out (not "Loaded" while it does not answer)
    expect(card.textContent).toContain('Last report');
    expect(card.querySelector('.network-stale')).not.toBeNull();
    expect(card.textContent).not.toContain('Loaded');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
    await act(async () => button('Unpair').click());
    expect(api.unpair).not.toHaveBeenCalled();
    api.unpair.mockResolvedValue(undefined);
    await act(async () => button('Unpair').click());
    expect(api.unpair).toHaveBeenCalledWith('w1');
    confirm.mockRestore();
  });
});
