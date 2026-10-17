import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ network: { status: vi.fn(), nodes: vi.fn(), runFlow: vi.fn(), run: vi.fn(), cancelRun: vi.fn(), picture: vi.fn() } }));
vi.mock('./api', () => api);
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import FlowWorkersPanel from './FlowWorkersPanel';

let container: HTMLDivElement;
let root: Root;
const flow = { flowId: 'logo', name: 'Make a logo', description: 'A logo from a prompt', inputs: [{ name: 'Prompt', default: 'a fox' }] };
const rig = { nodeId: 'w1', name: 'Rig', role: 'worker', online: true, devices: [], models: [], flows: [flow] };
const picture = 'http://127.0.0.1:17972/api/network/nodes/w1/gateway/files/images/a.png';

beforeEach(() => {
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
    expect(card.textContent).toContain('Waiting to start');
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(card.textContent).toContain('Running');
    expect(button('Stop')).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(card.textContent).toContain('Done');
    expect(card.textContent).toContain('A fox in a circle');
    expect(api.network.picture).toHaveBeenCalledWith(picture);
    expect(card.querySelector('img')!.getAttribute('src')).toBe('blob:picture');
  });

  it('says how a computer shares a flow, where a worker shares none', async () => {
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, flows: [] }] });
    await mount();
    expect(container.textContent).toContain('Rig shares no flow yet');
    expect(container.textContent).toContain('Let apps run it');
  });

  it('points to Networking while OAIY Network is off', async () => {
    api.network.status.mockResolvedValue({ enabled: false, role: 'standalone', uses: false, lends: false });
    await mount();
    expect(container.textContent).toContain('It is off until you switch on Use other computers');
    expect(api.network.nodes).not.toHaveBeenCalled();
  });
});
