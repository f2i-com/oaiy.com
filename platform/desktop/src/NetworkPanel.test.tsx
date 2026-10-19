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
vi.mock('./api', () => ({ network: api, NOT_ANSWERING: 'OAIY isn’t answering. It may still be starting, or it has stopped.' }));
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import NetworkPanel, { defaultDown, notInEffect } from './NetworkPanel';

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
const off = { ...base, enabled: false, role: 'standalone', uses: false, lends: false, nodeId: null, fingerprint: null, settings: { enabled: false, role: 'standalone', name: '', port: 17975, bind: '' } };
const worker = { ...base, enabled: true, role: 'worker', uses: false, lends: true, listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'], settings: { enabled: true, role: 'worker', name: 'Rig', port: 17975, bind: '' } };
const primary = { ...base, enabled: true, role: 'primary', uses: true, lends: false, settings: { enabled: true, role: 'primary', name: '', port: 17975, bind: '' } };
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
  sessionStorage.clear();
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
    expect(api.setSettings).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, uses: true, lends: false }));
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
    // (asked once: nothing more said)
    expect(container.querySelector('[data-testid="network-request-tries"]')).toBeNull();
  });

  it("says when a request's address asked again and again, each time with a new code", async () => {
    api.status.mockResolvedValue(worker);
    api.requests.mockResolvedValue({ requests: [{ id: 'r2', nodeId: 'p1', name: 'Mac mini', address: '192.168.1.5', code: '123 456', fingerprint: 'aaaa', secondsLeft: 280, tries: 4 }] });
    await mount();
    const tries = container.querySelector('[data-testid="network-request-tries"]')!;
    expect(tries.textContent).toContain('Asked 4 times from 192.168.1.5');
    expect(tries.textContent).toContain('deny it');
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
    const input = container.querySelector('input[aria-label="The other computer’s address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => input.closest('form')!.requestSubmit());
    expect(api.startPairing).toHaveBeenCalledWith({ address: '192.168.1.20', bothWays: false });
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
    expect(api.confirmPairing).toHaveBeenLastCalledWith('x', true, false);
    expect(container.textContent).toContain('Paired with Rig');
    const card = container.querySelector('[data-testid="network-node"]')!;
    expect(card.textContent).toContain('Online');
    const devices = Array.from(card.querySelectorAll('.network-device')).map((d) => d.textContent);
    expect(devices[0]).toContain('RTX 5090');
    expect(devices[0]).toContain('4.0 of 32.0 GB VRAM in use');
    expect(devices[1]).toContain('RTX 3070');
    expect(devices[1]).toContain('memory unknown');
    // the address typed is cleared once it is paired
    expect(input.value).toBe('');
    expect(card.textContent).toContain('flux');
  });

  it("shows a worker's processor, memory and GPUs, and its queue: what it makes and what waits", async () => {
    api.status.mockResolvedValue(primary);
    api.discover.mockResolvedValue({
      discovery: true,
      found: [{ nodeId: 'w9', name: 'Laptop', addresses: ['192.168.1.30:17975'], protocol: 2, hardware: 'RTX 4070 (8 GB VRAM) · 32 GB RAM · 16 cores', seenMs: 1, paired: false, compatible: true }],
    });
    api.nodes.mockResolvedValue({
      nodes: [
        {
          ...node,
          machine: { cpu: { name: 'AMD Ryzen 9 9950X', cores: 32, busy: 18 }, ram: { total_mb: 65536, available_mb: 49152 }, gpus: [{ name: 'RTX 5090', temperature: 61 }] },
          jobs: [
            { kind: 'image', model: 'flux', status: 'running', progress: 40 },
            { kind: 'flow', flow: 'logo', runId: 'r2', status: 'waiting', mine: true },
            { kind: 'video', model: 'ltx', status: 'waiting' },
          ],
        },
      ],
    });
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    const tiles = Array.from(card.querySelectorAll('.network-device')).map((d) => d.textContent);
    expect(tiles[0]).toContain('Processor');
    expect(tiles[0]).toContain('AMD Ryzen 9 9950X · 32 cores');
    expect(tiles[0]).toContain('18% busy');
    expect(tiles[1]).toContain('16.0 of 64.0 GB in use');
    expect(tiles[2]).toContain('RTX 5090');
    expect(tiles[2]).toContain('61°C');
    const queue = card.querySelector('[data-testid="network-queue"]')!.textContent;
    expect(queue).toContain('1 running, 2 waiting');
    expect(queue).toContain('Making a picture');
    expect(queue).toContain('40%');
    expect(queue).toContain('Waiting: a flow run');
    expect(queue).toContain('this computer’s');
    // (one found on the network says what it has, to choose which to pair with)
    expect(container.querySelector('[data-testid="network-found"]')!.textContent).toContain('RTX 4070 (8 GB VRAM) · 32 GB RAM · 16 cores');
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

  it("marks a worker's model that is this computer's default, listed with the loaded ones", async () => {
    api.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/klein', nodeId: 'w1', nodeName: 'Rig', model: 'klein' } } });
    const klein = { id: 'engine:image:node.w1/klein', kind: 'image', model: 'klein', state: 'not loaded', device: null, pinned: false };
    api.nodes.mockResolvedValue({ nodes: [{ ...node, models: [node.models[0], klein] }] });
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    const row = Array.from(card.querySelectorAll('.network-model')).find((r) => r.textContent?.includes('klein'))!;
    expect(row.textContent).toContain('Default');
    expect(row.closest('details')).toBeNull();
  });

  it('shows a pairing started before the page was left again, at the step it is at', async () => {
    vi.useFakeTimers();
    api.status.mockResolvedValue({ ...primary, pairings: [{ pairingId: 'x', code: '654 321', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.20:17975', fingerprint: '1111' } }] });
    await mount();
    const card = container.querySelector('[data-testid="network-pairing"]')!;
    expect(card.textContent).toContain('654 321');
    expect(card.textContent).toContain('Did its screen show this same code?');
    expect(card.querySelector('.network-progress .is-now')!.textContent).toBe('Codes matched here');
  });

  it('says how it goes while it is off', async () => {
    api.status.mockResolvedValue(off);
    await mount();
    expect(container.querySelectorAll('[data-testid="network-steps"] li')).toHaveLength(3);
  });

  it('uses other computers and lends itself at once, either switched on or both', async () => {
    api.status.mockResolvedValue(primary);
    api.setSettings.mockResolvedValue({ ...primary, uses: true, lends: true, role: 'both' });
    await mount();
    const lend = Array.from(container.querySelectorAll('[role="checkbox"]')).find((b) => b.textContent?.includes('Lend this computer')) as HTMLButtonElement;
    expect(lend.getAttribute('aria-checked')).toBe('false');
    await act(async () => lend.click());
    expect(lend.getAttribute('aria-checked')).toBe('true');
    await act(async () => button('Save').click());
    expect(api.setSettings).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, uses: true, lends: true, port: 17975 }));
    // both off: nothing to save
    const use = Array.from(container.querySelectorAll('[role="checkbox"]')).find((b) => b.textContent?.includes('Use other computers')) as HTMLButtonElement;
    await act(async () => use.click());
    await act(async () => lend.click());
    expect(container.textContent).toContain('Choose one, or both');
  });

  it('pairs both ways at once where this computer lends itself too, and shows a computer paired both ways on one card', async () => {
    vi.useFakeTimers();
    const both = { ...primary, uses: true, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] };
    api.status.mockResolvedValue({ ...both, pairings: [{ pairingId: 'x', code: '654 321', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.21:17975', fingerprint: '1111' } }] });
    api.confirmPairing.mockResolvedValue({ status: 'approved', node });
    await mount();
    const tick = container.querySelector('.network-both-ways input') as HTMLInputElement;
    expect(tick.checked).toBe(true);
    expect(container.querySelector('.network-both-ways')!.textContent).toContain('Also let Rig use this computer (one pairing, both ways)');
    api.status.mockResolvedValue(both);
    api.nodes.mockResolvedValue({ nodes: [{ ...node, uses: true, usesThis: true }] });
    await act(async () => button('Yes, pair').click());
    expect(api.confirmPairing).toHaveBeenLastCalledWith('x', true, true);
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    const card = container.querySelector('[data-testid="network-node"]')!;
    expect(card.textContent).toContain('This computer uses it');
    expect(card.textContent).toContain('Uses this computer');
  });

  it('lets the Agent set it up with a tick, saved at once, and asks this side for the code where the other approved by itself', async () => {
    api.status.mockResolvedValue(primary);
    api.setSettings.mockResolvedValue({ ...primary, agent: true });
    api.startPairing.mockResolvedValue({ pairingId: 'x', status: 'approved_there', code: '111 222', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.20:17975', fingerprint: '1111' } });
    await mount();
    const tick = container.querySelector('[data-testid="network-agent"] input') as HTMLInputElement;
    expect(tick.checked).toBe(false);
    await act(async () => tick.click());
    expect(api.setSettings).toHaveBeenCalledWith({ agent: true });
    const input = container.querySelector('input[aria-label="The other computer’s address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => input.closest('form')!.requestSubmit());
    // (approved there at once, as anyone answering for it could say: paired here only on this side's word)
    const card = container.querySelector('[data-testid="network-pairing"]')!;
    expect(card.textContent).toContain('111 222');
    expect(card.textContent).toContain('Did its screen show this same code?');
    expect(container.textContent).not.toContain('Paired with Rig');
    expect(api.confirmPairing).not.toHaveBeenCalled();
  });

  it('opens pairing by itself for five minutes where the Agent may set it up, and says who paired so, with the code', async () => {
    const lending = { ...primary, lends: true, role: 'both', agent: true, pairingOpenUntilMs: null, pairedByItself: [{ name: 'Laptop', at: Date.now(), code: '345 678' }] };
    api.status.mockResolvedValue(lending);
    api.setSettings.mockResolvedValue({ ...lending, pairingOpenUntilMs: Date.now() + 5 * 60_000 });
    await mount();
    expect(container.textContent).toContain('Laptop paired with this computer by itself');
    // (the code the computer that asked compares before it pairs on its side)
    expect(container.querySelector('.network-by-itself')!.textContent).toContain('345 678');
    const window_ = () => container.querySelector('[data-testid="network-window"]')!;
    expect(window_().textContent).toContain('Closed');
    api.status.mockResolvedValue({ ...lending, pairingOpenUntilMs: Date.now() + 5 * 60_000 });
    await act(async () => (window_().querySelector('button') as HTMLButtonElement).click());
    expect(api.setSettings).toHaveBeenCalledWith({ openPairingMinutes: 5 });
    expect(window_().textContent).toContain('Pairing by itself is open until');
  });

  it('lends its model files while lending itself, until its person unticks it', async () => {
    api.status.mockResolvedValue({ ...primary, lends: true, role: 'both', shareModels: true });
    api.setSettings.mockResolvedValue({ ...primary, lends: true, role: 'both', shareModels: false });
    await mount();
    const tick = container.querySelector('.network-share-models input') as HTMLInputElement;
    expect(tick.checked).toBe(true);
    await act(async () => tick.click());
    expect(api.setSettings).toHaveBeenCalledWith({ shareModels: false });
  });

  it('withdraws a pairing it stops waiting for, so the worker no longer shows it', async () => {
    vi.useFakeTimers();
    api.status.mockResolvedValue(primary);
    api.startPairing.mockResolvedValue({ pairingId: 'x', code: '654 321', status: 'pending', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.20:17975', fingerprint: '1111' } });
    api.confirmPairing.mockResolvedValue({ status: 'declined' });
    await mount();
    const input = container.querySelector('input[aria-label="The other computer’s address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => input.closest('form')!.requestSubmit());
    await act(async () => button('Stop waiting').click());
    expect(api.confirmPairing).toHaveBeenCalledWith('x', false, false);
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
    expect(api.setSettings).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, uses: false, lends: true, port: 18000 }));
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

  it('says so when a pairing asked for both ways came about one way only, and why', async () => {
    const both = { ...primary, uses: true, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] };
    api.status.mockResolvedValue({ ...both, pairings: [{ pairingId: 'x', code: '654 321', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.21:17975', fingerprint: '1111' } }] });
    api.confirmPairing.mockResolvedValue({ status: 'approved', node, bothWays: false, bothWaysError: 'Rig could not reach this computer at 192.168.1.20:17975 (connection timed out).' });
    await mount();
    api.status.mockResolvedValue(both);
    await act(async () => button('Yes, pair').click());
    const said = Array.from(container.querySelectorAll('.banner')).find((b) => b.textContent?.startsWith('Paired with'))!;
    expect(said.textContent).toBe(
      'Paired with Rig, one way only: this computer uses Rig, but Rig cannot use this one: Rig could not reach this computer at 192.168.1.20:17975 (connection timed out). To let it, pair again from Rig’s Networking page.',
    );
    expect(said.className).not.toContain('banner-ok');
  });

  it('says plainly “Paired” where both ways came about, or was not asked for', async () => {
    const both = { ...primary, uses: true, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] };
    api.status.mockResolvedValue({ ...both, pairings: [{ pairingId: 'x', code: '654 321', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.21:17975', fingerprint: '1111' } }] });
    api.confirmPairing.mockResolvedValue({ status: 'approved', node, bothWays: true, bothWaysError: null });
    await mount();
    api.status.mockResolvedValue(both);
    await act(async () => button('Yes, pair').click());
    const said = Array.from(container.querySelectorAll('.banner')).find((b) => b.textContent?.startsWith('Paired with'))!;
    expect(said.textContent).toBe('Paired with Rig.');
    expect(said.className).toContain('banner-ok');
  });

  it('offers both ways before pairing in plain words', async () => {
    api.status.mockResolvedValue({ ...primary, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] });
    await mount();
    expect(container.querySelector('.network-both-ways')!.textContent).toBe('Also let the computer you pair with use this one (one pairing, both ways)');
    expect(container.querySelector('input[aria-label="The other computer’s address"]')).not.toBeNull();
    expect(container.textContent).not.toMatch(/\bworker/i);
  });

  it('where the other computer was not reached, lists what to check, this Mac’s Local Network permission on a Mac', async () => {
    const agent = vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15');
    try {
      api.status.mockResolvedValue(primary);
      api.startPairing.mockRejectedValue(new Error('Could not reach 192.168.1.99:17975: connection refused (os error 61)'));
      await mount();
      const input = container.querySelector('input[aria-label="The other computer’s address"]') as HTMLInputElement;
      await act(async () => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, '192.168.1.99:17975');
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await act(async () => (container.querySelector('form.network-add button[type="submit"]') as HTMLButtonElement).click());
      const items = Array.from(container.querySelectorAll('.network-checklist li')).map((li) => li.textContent);
      expect(items).toHaveLength(4);
      expect(items[2]).toContain('on Windows the network must be set as Private, or the firewall must allow OAIY on Public networks');
      expect(items[3]).toBe('This Mac lets OAIY reach the local network: System Settings → Privacy & Security → Local Network, with OAIY switched on.');
      agent.mockReturnValue('Mozilla/5.0 (Windows NT 10.0; Win64; x64)');
      await act(async () => root.render(<NetworkPanel key="again" />));
      await act(async () => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(container.querySelector('input[aria-label="The other computer’s address"]'), '192.168.1.99:17975');
        container.querySelector('input[aria-label="The other computer’s address"]')!.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await act(async () => (container.querySelector('form.network-add button[type="submit"]') as HTMLButtonElement).click());
      expect(container.querySelectorAll('.network-checklist li')).toHaveLength(3);
    } finally {
      agent.mockRestore();
    }
  });

  it('lending, says the other computer must reach this one, and on Windows that the network must be Private', async () => {
    const agent = vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Mozilla/5.0 (Windows NT 10.0; Win64; x64)');
    try {
      api.status.mockResolvedValue(worker);
      await mount();
      const reach = container.querySelector('[data-testid="network-reach"]')!.textContent;
      expect(reach).toContain('must reach it at one of these addresses, on port 17975');
      expect(reach).toContain('the network must be set as Private, or the firewall must allow OAIY on Public networks');
    } finally {
      agent.mockRestore();
    }
  });

  it('says what is not known of an offline computer rather than guessing it', async () => {
    api.status.mockResolvedValue(primary);
    api.nodes.mockResolvedValue({
      nodes: [{ ...node, online: false, lastError: 'connection refused', devices: [{ id: 'gpu0', name: 'RTX 3090', memory_total_mb: 24576, memory_used_mb: null }], models: [{ id: 'engine:image:node.w1/moss', kind: 'image', model: 'moss', state: 'not loaded', device: null, pinned: false }] }],
    });
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    expect(card.querySelector('.network-device-figure')?.textContent).toBe('24.0 GB VRAM, in use not known while it is offline');
    expect(card.textContent).toContain('Which models are loaded on it is not known while it is offline.');
    expect(card.textContent).not.toContain('No model is loaded on it now.');
    expect(card.textContent).not.toContain('?');
  });

  it('offers “Use here” only for a service not in use here already', async () => {
    const voice = { id: 'oaiy-voice', name: 'OAIY Voice', installed: true, status: 'running' };
    api.nodes.mockResolvedValue({ nodes: [{ ...node, services: [voice] }] });
    api.status.mockResolvedValue({ ...primary, services: {} });
    const onNavigate = vi.fn();
    await act(async () => root.render(<NetworkPanel onNavigate={onNavigate} />));
    await act(async () => {});
    expect(button('Use here')).toBeTruthy();
    api.status.mockResolvedValue({ ...primary, services: { 'oaiy-voice': { nodeId: 'w1', nodeName: 'Rig' } } });
    await act(async () => root.render(<NetworkPanel key="used" onNavigate={onNavigate} />));
    await act(async () => {});
    expect(container.textContent).toContain('OAIY Voice (in use here)');
    expect(button('Use here')).toBeUndefined();
    await act(async () => button('Open Services').click());
    expect(onNavigate).toHaveBeenCalledWith('services');
  });

  it('marks a default its computer cannot run now as not in effect, with why', async () => {
    api.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/flux', nodeId: 'w1', nodeName: 'Rig', model: 'flux', inEffect: false, waiting: 'engines_down' } } });
    api.nodes.mockResolvedValue({ nodes: [node] });
    await mount();
    const chip = container.querySelector('[data-testid="network-node"] .engine-default-chip')!;
    expect(chip.textContent?.trim()).toBe('Default, not in effect');
    expect(chip.getAttribute('title')).toBe('Not in effect: Rig’s engines are not running; this computer’s own model runs meanwhile.');
  });

  it('says a default still in effect whose computer’s engines are not answering fails meanwhile, on its chip and its card', async () => {
    api.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/flux', nodeId: 'w1', nodeName: 'Rig', model: 'flux', inEffect: true, waiting: 'engines_down' } } });
    api.nodes.mockResolvedValue({ nodes: [node] });
    await mount();
    const card = container.querySelector('[data-testid="network-node"]')!;
    const chip = card.querySelector('.engine-default-chip')!;
    expect(chip.textContent?.trim()).toBe('Default, not answering');
    expect(chip.getAttribute('title')).toBe('Rig’s engines aren’t answering: what uses it fails until they do.');
    // (not "not in effect": nothing of this computer's runs in its place)
    expect(card.textContent).not.toContain('not in effect');
    expect(card.querySelector('[data-testid="network-default-down"]')!.textContent?.trim()).toBe('flux is this computer’s default. Rig’s engines aren’t answering: what uses it fails until they do.');
  });

  it('says a language model default whose computer’s engines are not answering fails chats meanwhile', () => {
    const d = { id: 'engine:llm:node.w1/qwen3.5-9b', nodeId: 'w1', nodeName: 'Rig', model: 'qwen3.5-9b', inEffect: true, waiting: 'engines_down' as const };
    expect(defaultDown(d, 'Rig')).toBe('Rig’s engines aren’t answering: chats that use it fail until they do.');
    // Not in effect, engines down keeps its old meaning: this computer's own model runs meanwhile.
    expect(defaultDown({ ...d, inEffect: false }, 'Rig')).toBeNull();
    expect(notInEffect({ ...d, inEffect: false }, 'Rig')).toBe('Not in effect: Rig’s engines are not running; this computer’s own model runs meanwhile.');
    expect(notInEffect(d, 'Rig')).toBeNull();
  });

  it('says nothing more of a default in effect whose computer answers', async () => {
    api.status.mockResolvedValue({ ...primary, defaults: { image: { id: 'engine:image:node.w1/flux', nodeId: 'w1', nodeName: 'Rig', model: 'flux', inEffect: true, waiting: null } } });
    api.nodes.mockResolvedValue({ nodes: [node] });
    await mount();
    expect(container.querySelector('[data-testid="network-node"] .engine-default-chip')!.textContent?.trim()).toBe('Default');
    expect(container.querySelector('[data-testid="network-default-down"]')).toBeNull();
  });

  it('keeps a pairing’s both-ways choice when the page is left and opened again while it waits', async () => {
    const both = { ...primary, uses: true, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] };
    const waiting = { pairingId: 'p7', code: '123 456', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.21:17975', fingerprint: '1111' } };
    api.status.mockResolvedValue(both);
    api.startPairing.mockResolvedValue(waiting);
    await mount();
    const input = container.querySelector('input[aria-label="The other computer’s address"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, '192.168.1.21');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    // One way only: unticked before pairing.
    await act(async () => (container.querySelector('.network-both-ways input') as HTMLInputElement).click());
    await act(async () => input.closest('form')!.requestSubmit());
    expect(api.startPairing).toHaveBeenCalledWith({ address: '192.168.1.21', bothWays: false });
    // Left, and opened again: the pairing is shown again with its tick as it was, and answered as it was asked.
    await act(async () => root.unmount());
    root = createRoot(container);
    api.status.mockResolvedValue({ ...both, pairings: [waiting] });
    api.confirmPairing.mockResolvedValue({ status: 'approved', node, bothWays: false, bothWaysError: null });
    await mount();
    expect((container.querySelector('.network-both-ways input') as HTMLInputElement).checked).toBe(false);
    api.status.mockResolvedValue(both);
    await act(async () => button('Yes, pair').click());
    expect(api.confirmPairing).toHaveBeenLastCalledWith('p7', true, false);
    // (one way was what was asked: plainly paired)
    expect(Array.from(container.querySelectorAll('.banner')).find((b) => b.textContent?.startsWith('Paired with'))!.textContent).toBe('Paired with Rig.');
  });

  it('says a pairing ended one way only when the page is opened again, where the answer came after it was left', async () => {
    const both = { ...primary, uses: true, lends: true, role: 'both', listening: '0.0.0.0:17975', addresses: ['192.168.1.20:17975'] };
    api.status.mockResolvedValue({ ...both, pairings: [{ pairingId: 'x', code: '654 321', status: 'approved_there', worker: { nodeId: 'w1', name: 'Rig', address: '192.168.1.21:17975', fingerprint: '1111' } }] });
    let answer!: (v: unknown) => void;
    api.confirmPairing.mockReturnValue(new Promise((done) => (answer = done)));
    await mount();
    api.status.mockResolvedValue(both);
    await act(async () => button('Yes, pair').click());
    expect(api.confirmPairing).toHaveBeenLastCalledWith('x', true, true);
    // Left before the answer came; it comes; the page is opened again.
    await act(async () => root.unmount());
    await act(async () => answer({ status: 'approved', node, bothWays: false, bothWaysError: 'Rig could not reach this computer.' }));
    root = createRoot(container);
    await mount();
    const said = () => Array.from(container.querySelectorAll('.banner')).find((b) => b.textContent?.startsWith('Paired with'));
    expect(said()!.textContent).toBe('Paired with Rig, one way only: this computer uses Rig, but Rig cannot use this one: Rig could not reach this computer. To let it, pair again from Rig’s Networking page.');
    // Said once: not again the next time.
    await act(async () => root.unmount());
    root = createRoot(container);
    await mount();
    expect(said()).toBeUndefined();
  });

  it('says OAIY is not answering, with a way to ask again, rather than a page with nothing on it', async () => {
    api.status.mockRejectedValueOnce(new Error('OAIY isn’t answering. It may still be starting, or it has stopped.'));
    await mount();
    const banner = container.querySelector('.load-error')!;
    expect(banner.textContent).toContain('OAIY isn’t answering. It may still be starting, or it has stopped.');
    expect(banner.textContent).not.toContain('could not be read');
    api.status.mockResolvedValue(off);
    await act(async () => button('Try again').click());
    expect(container.querySelector('.load-error')).toBeNull();
    expect(container.querySelector('[data-testid="network-panel"]')).not.toBeNull();
  });

  it('says what could not be read when OAIY answered with an error', async () => {
    api.status.mockRejectedValueOnce(new Error('the network settings file is damaged'));
    await mount();
    expect(container.querySelector('.load-error')!.textContent).toContain('OAIY Network’s status could not be read: the network settings file is damaged');
  });
});
