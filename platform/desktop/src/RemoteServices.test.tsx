import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ network: { status: vi.fn(), nodes: vi.fn(), useService: vi.fn() } }));
vi.mock('./api', () => api);
vi.mock('./useVisiblePoll', () => ({ useVisiblePoll: () => {} }));
import RemoteServices from './RemoteServices';

let container: HTMLDivElement;
let root: Root;
const rig = { nodeId: 'w1', name: 'Rig', role: 'worker', online: true, devices: [], models: [], services: [{ id: 'oaiy-voice', name: 'OAIY Voice', installed: true, status: 'running' }] };

beforeEach(() => {
  vi.clearAllMocks();
  api.network.nodes.mockResolvedValue({ nodes: [rig] });
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});
async function mount() {
  await act(async () => root.render(<RemoteServices />));
  await act(async () => {});
}
const button = (label: string) => Array.from(container.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;

describe('RemoteServices', () => {
  it("uses a worker's OAIY Voice here with a press, and gives it back", async () => {
    api.network.status.mockResolvedValue({ enabled: true, role: 'primary', uses: true, services: {} });
    api.network.useService.mockResolvedValueOnce({ enabled: true, role: 'primary', uses: true, services: { 'oaiy-voice': { nodeId: 'w1', nodeName: 'Rig' } } });
    api.network.useService.mockResolvedValueOnce({ enabled: true, role: 'primary', uses: true, services: {} });
    await mount();
    const row = container.querySelector('[data-testid="remote-service"]')!;
    expect(row.textContent).toContain('OAIY Voice');
    expect(row.textContent).toContain('on Rig');
    expect(row.textContent).toContain('Running');
    await act(async () => button('Use it here').click());
    expect(api.network.useService).toHaveBeenCalledWith('oaiy-voice', 'w1');
    expect(container.textContent).toContain('In use here');
    await act(async () => button('Use this computer').click());
    expect(api.network.useService).toHaveBeenLastCalledWith('oaiy-voice', null);
  });

  it('shows nothing where no paired computer lends a service', async () => {
    api.network.status.mockResolvedValue({ enabled: true, role: 'primary', uses: true });
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, services: [] }] });
    await mount();
    expect(container.textContent).toBe('');
  });

  it('keeps a service in use on a computer not heard from since this one started, with the way back to this computer’s own', async () => {
    // (as the desktop lists it: in use here, not heard of since it started)
    api.network.status.mockResolvedValue({ enabled: true, role: 'primary', uses: true, services: { 'oaiy-voice': { nodeId: 'w1', nodeName: 'Rig', heard: false, online: false } } });
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, online: false, services: [{ id: 'oaiy-voice', name: 'OAIY Voice', installed: false, status: null, heard: false }] }] });
    api.network.useService.mockResolvedValue({ enabled: true, role: 'primary', uses: true, services: {} });
    await mount();
    const row = container.querySelector('[data-testid="remote-service"]')!;
    expect(row.textContent).toContain('Not heard from yet');
    expect(row.textContent).toContain('In use here');
    expect(row.textContent).toContain('Rig has not been heard from since this computer started.');
    await act(async () => button('Use this computer').click());
    expect(api.network.useService).toHaveBeenCalledWith('oaiy-voice', null);
  });

  it('keeps one in use on a computer that lists no services at all (an older desktop’s answer)', async () => {
    api.network.status.mockResolvedValue({ enabled: true, role: 'primary', uses: true, services: { 'oaiy-voice': { nodeId: 'w1', nodeName: 'Rig' } } });
    api.network.nodes.mockResolvedValue({ nodes: [{ ...rig, online: false, services: [] }] });
    await mount();
    const row = container.querySelector('[data-testid="remote-service"]')!;
    expect(row.textContent).toContain('OAIY Voice');
    expect(row.textContent).toContain('on Rig');
    expect(row.textContent).toContain('Offline');
    expect(button('Use this computer')).toBeTruthy();
  });
});
