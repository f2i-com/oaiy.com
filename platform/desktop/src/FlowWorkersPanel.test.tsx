import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ network: { status: vi.fn(), nodes: vi.fn(), runFlow: vi.fn(), run: vi.fn(), cancelRun: vi.fn(), picture: vi.fn() } }));
vi.mock('./api', () => api);
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import FlowWorkersPanel, { forgetFollowedRuns } from './FlowWorkersPanel';

let container: HTMLDivElement;
let root: Root;
const flow = { flowId: 'logo', name: 'Make a logo', description: 'A logo from a prompt', inputs: [{ name: 'Prompt', default: 'a fox' }] };
const rig = { nodeId: 'w1', name: 'Rig', role: 'worker', online: true, devices: [], models: [], flows: [flow] };
const picture = 'http://127.0.0.1:17972/api/network/nodes/w1/gateway/files/images/a.png';

beforeEach(() => {
  forgetFollowedRuns();
  vi.clearAllMocks();
  vi.useFakeTimers();
  api.network.status.mockResolvedValue({ enabled: true, role: 'primary', uses: true, lends: false });
  api.network.nodes.mockResolvedValue({ nodes: [rig] });
  api.network.picture.mockResolvedValue(new Blob(['x']));
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: () => 'blob:picture' });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: () => {} });
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
  await act(async () => root.render(<FlowWorkersPanel onOpenNetworking={() => {}} />));
  await act(async () => {});
}
const button = (label: string) => Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;

describe('FlowWorkersPanel', () => {
  it("runs a worker's shared flow there with its inputs, follows it, and shows its picture and words", async () => {
    api.network.runFlow.mockResolvedValue({ runId: 'r1', status: 'queued' });
    api.network.run
      .mockResolvedValueOnce({ runId: 'r1', status: 'running' })
      .mockResolvedValueOnce({ runId: 'r1', status: 'succeeded', output: { output: { image: picture, caption: 'A fox in a circle' } } });
    await mount();
    const card = container.querySelector('[data-testid="flow-run-card"]')!;
    expect(card.textContent).toContain('Make a logo');
    const prompt = card.querySelector('textarea')!;
    expect(prompt.value).toBe('a fox');
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!;
      set.call(prompt, 'a lighthouse');
      prompt.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => button('Run on Rig').click());
    expect(api.network.runFlow).toHaveBeenCalledWith('w1', 'logo', { Prompt: 'a lighthouse' });
    expect(card.textContent).toContain('Waiting in its queue');
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(card.textContent).toContain('Running');
    expect(button('Stop')).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(card.textContent).toContain('Done');
    expect(card.textContent).toContain('A fox in a circle');
    expect(api.network.picture).toHaveBeenCalledWith(picture);
    expect(card.querySelector('img')!.getAttribute('src')).toBe('blob:picture');
  });

  it('queues another run while one runs there, each followed on its own', async () => {
    api.network.runFlow.mockResolvedValueOnce({ runId: 'r1', status: 'running' }).mockResolvedValueOnce({ runId: 'r2', status: 'queued' });
    api.network.run.mockImplementation(async (_n: string, id: string) => ({ runId: id, status: id === 'r1' ? 'running' : 'queued' }));
    await mount();
    const card = container.querySelector('[data-testid="flow-run-card"]')!;
    await act(async () => button('Run on Rig').click());
    await act(async () => button('Queue another on Rig').click());
    expect(api.network.runFlow).toHaveBeenCalledTimes(2);
    const runs = Array.from(card.querySelectorAll('[data-testid="flow-run"]')).map((r) => r.textContent);
    expect(runs[0]).toContain('Waiting in its queue');
    expect(runs[1]).toContain('Running');
    expect(card.textContent).toContain('2 in its queue');
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(api.network.run).toHaveBeenCalledWith('w1', 'r1');
    expect(api.network.run).toHaveBeenCalledWith('w1', 'r2');
  });

  it('still follows a run after the page was left and opened again, and lets go of one that stops answering', async () => {
    api.network.runFlow.mockResolvedValue({ runId: 'r1', status: 'running' });
    api.network.run.mockRejectedValue(new Error('503: Rig did not answer'));
    await mount();
    await act(async () => button('Run on Rig').click());
    await act(async () => root.unmount());
    root = createRoot(container);
    await mount();
    const card = () => container.querySelector('[data-testid="flow-run-card"]')!;
    expect(card().querySelectorAll('[data-testid="flow-run"]').length).toBe(1);
    expect(card().textContent).toContain('Running');
    // asked less and less often, then let go, saying so
    for (let i = 0; i < 12; i++) await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(card().textContent).toContain('Lost track of this run');
    const asked = api.network.run.mock.calls.length;
    expect(asked).toBeLessThanOrEqual(8);
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(api.network.run.mock.calls.length).toBe(asked);
    expect(card().querySelector('button[title="Stop"], .flow-run-item-head .btn-tiny')).toBeNull();
  });

  it('says how a computer shares a flow, where a worker shares none', async () => {
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, flows: [] }] });
    await mount();
    expect(container.textContent).toContain('Rig shares no flow yet');
    expect(container.textContent).toContain('Let apps run it');
  });

  it('says a computer that is not answering is off, rather than that it shares nothing', async () => {
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, online: false, flows: [], lastSeenMs: null, lastError: 'connection refused' }] });
    await mount();
    expect(container.textContent).toContain('Rig is not answering: connection refused');
    expect(container.textContent).toContain('The flows it shares show here when it answers again');
    expect(container.textContent).not.toContain('shares no flow yet');
  });

  it('follows a run this computer asked for there that this page did not start (the Agent’s, or from before a restart)', async () => {
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, jobs: [{ kind: 'flow', flow: 'logo', runId: 'r9', status: 'running', mine: true }, { kind: 'flow', status: 'waiting', mine: false }] }] });
    api.network.run.mockResolvedValue({ runId: 'r9', status: 'succeeded', output: { output: 'A fox, found', }, startedAt: '2026-10-18T01:00:00Z', finishedAt: '2026-10-18T01:00:12Z' });
    await mount();
    const card = container.querySelector('[data-testid="flow-run-card"]')!;
    expect(card.querySelectorAll('[data-testid="flow-run"]')).toHaveLength(1);
    expect(card.textContent).toContain('Running');
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(api.network.run).toHaveBeenCalledWith('w1', 'r9');
    expect(card.textContent).toContain('Done');
    expect(card.textContent).toContain('A fox, found');
    // (its time is the worker's own: start to end there)
    expect(card.textContent).toContain('12 s');
  });

  it('asks before stopping a run there', async () => {
    api.network.runFlow.mockResolvedValue({ runId: 'r1', status: 'running' });
    api.network.run.mockResolvedValue({ runId: 'r1', status: 'running' });
    api.network.cancelRun.mockResolvedValue({});
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
    await mount();
    await act(async () => button('Run on Rig').click());
    await act(async () => button('Stop').click());
    expect(api.network.cancelRun).not.toHaveBeenCalled();
    api.network.run.mockResolvedValue({ runId: 'r1', status: 'cancelled' });
    await act(async () => button('Stop').click());
    expect(api.network.cancelRun).toHaveBeenCalledWith('w1', 'r1');
    expect(container.textContent).toContain('Stopped');
    confirm.mockRestore();
  });

  it('points to Networking while OAIY Network is off', async () => {
    api.network.status.mockResolvedValue({ enabled: false, role: 'standalone', uses: false, lends: false });
    await mount();
    expect(container.textContent).toContain('It is off until you switch on Use other computers');
    expect(api.network.nodes).not.toHaveBeenCalled();
  });
});
