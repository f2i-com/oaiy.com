import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({
  list: vi.fn(), command: vi.fn(), events: vi.fn(),
  toast: { push: vi.fn() },
  ai: [] as { dispose: ReturnType<typeof vi.fn>; complete: ReturnType<typeof vi.fn> }[],
  voice: [] as { dispose: ReturnType<typeof vi.fn>; status: ReturnType<typeof vi.fn>; open: ReturnType<typeof vi.fn>; emit: (value: unknown) => void }[],
}));
vi.mock('./api', () => ({ API_BASE: 'http://mock.invalid', plugins: { list: mock.list }, bridge: { connectorRequest: mock.command, events: mock.events }, companion: {} }));
vi.mock('./Toasts', () => ({ useToast: () => mock.toast }));
vi.mock('./useModules', () => ({ useModules: () => null, moduleOn: () => false }));
vi.mock('./PluginVoiceControls', () => ({ PluginVoiceControls: () => null }));
vi.mock('./pluginAi', () => ({ PluginAiSession: class {
  dispose = vi.fn(); complete = vi.fn();
  constructor() { mock.ai.push(this); }
} }));
vi.mock('./pluginVoice', () => ({ PluginVoiceSession: class {
  dispose = vi.fn(); status = vi.fn().mockResolvedValue({ sttReady: true, ttsReady: true, reason: null }); open = vi.fn();
  constructor(_plugin: string, _api: string, readonly emit: (value: unknown) => void) { mock.voice.push(this); }
} }));
import PluginScreenPage, { HOST_BOOTSTRAP } from './PluginScreenPage';
import { testMessageEvent, installTestChannel, testPorts } from './pluginScreenRpc.testTransport';

let root: Root, container: HTMLDivElement;
const plugin = { id: 'mock-plugin', state: 'running', manifest: { name: 'MOCK only', capabilities: ['oaiy.voice.session', 'oaiy.ai.complete'], ui: { nav: [{ id: 'screen', screen: 'screen' }], screens: [{ id: 'screen', entry: 'index.html', files: ['index.html'] }] } } };
beforeEach(() => {
  mock.list.mockReset().mockResolvedValue({ plugins: [plugin] }); mock.command.mockReset(); mock.events.mockReset(); mock.ai.length = 0; mock.voice.length = 0;
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, text: async () => '<p>MOCK plugin</p>', arrayBuffer: async () => new ArrayBuffer(0) }));
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
async function mount(connectDirect = true) {
  await act(async () => root.render(<PluginScreenPage pluginId="mock-plugin" navId="screen"/>));
  const frame = container.querySelector('iframe')!;
  const nonce = JSON.parse(frame.srcdoc.match(/window\.__oaiyDocumentNonce=("[^"]+")/)![1]);
  const source = frame.contentWindow!;
  const ports = testPorts();
  const reply = vi.spyOn(ports.port2, 'postMessage');
  if (connectDirect) window.dispatchEvent(testMessageEvent({ source, ports: [ports.port2 as unknown as MessagePort], data: { __pluginHost: 1, documentNonce: nonce, method: 'document.connect' } }));
  const send = (method: string, documentNonce: unknown = nonce, id = 'test') => ports.port1.postMessage({ __pluginHost: 1, documentNonce, id, method, args: method === 'command' ? ['read'] : method === 'aiComplete' ? [{}] : [] });
  // jsdom emits its initial iframe load; the next dispatched load models a
  // replacement document without fabricating a different WindowProxy.
  return { frame, nonce, source, reply, send };
}
describe('mounted plugin document replacement security', () => {
  it('legitimate first load works; replacement load disposes AI/voice and cannot replay the old nonce or source', async () => {
    const { frame, source, nonce, reply, send } = await mount();
    await act(async () => send('voice.status'));
    expect(mock.voice[0].status).toHaveBeenCalledTimes(1);
    expect(reply).toHaveBeenCalledWith(expect.objectContaining({ documentNonce: nonce, ok: true }));
    reply.mockClear();
    await act(async () => frame.dispatchEvent(new Event('load')));
    expect(frame.contentWindow).toBe(source);
    expect(mock.ai[0].dispose).toHaveBeenCalledTimes(1); expect(mock.voice[0].dispose).toHaveBeenCalledTimes(1);
    for (const token of [nonce, 'new-document-guess', undefined]) await act(async () => { send('voice.open', token); send('aiComplete', token); send('command', token); });
    mock.voice[0].emit({ sessionId: 'old', type: 'transcript', text: 'PRIVATE old voice result' });
    expect(reply).not.toHaveBeenCalled(); expect(mock.voice[0].open).not.toHaveBeenCalled(); expect(mock.ai[0].complete).not.toHaveBeenCalled(); expect(mock.command).not.toHaveBeenCalled();
    expect(container.textContent).toContain('pending deliveries were revoked');
  });
  it('retirement suppresses delayed command and completion replies without automatically reissuing actions', async () => {
    const { frame, reply, send } = await mount();
    let command!: (value: unknown) => void, complete!: (value: unknown) => void;
    mock.command.mockReturnValue(new Promise(resolve => { command = resolve; }));
    mock.ai[0].complete.mockReturnValue(new Promise(resolve => { complete = resolve; }));
    await act(async () => { send('command', undefined, 'command'); send('aiComplete', undefined, 'ai'); });
    await act(async () => frame.dispatchEvent(new Event('load')));
    await act(async () => { command({ ok: true, result: { ok: true, data: 'PRIVATE command' } }); complete({ text: 'PRIVATE model' }); });
    expect(reply).not.toHaveBeenCalled(); expect(mock.command).toHaveBeenCalledTimes(1); expect(mock.ai[0].complete).toHaveBeenCalledTimes(1);
  });
  it('the injected SDK tags calls/replies and retires on pagehide before a replacement can load', async () => {
    const { frame, nonce } = await mount(false), source = frame.contentWindow!;
    Object.defineProperty(source, '__oaiyDocumentNonce', { configurable: true, value: nonce });
    const ports = installTestChannel(source);
    vi.spyOn(window, 'postMessage').mockImplementation((data: unknown, _origin?: string | WindowPostMessageOptions, transfer?: Transferable[]) => { window.dispatchEvent(testMessageEvent({ source, data, ports: transfer as MessagePort[] })); });
    new Function('window', 'parent', 'document', HOST_BOOTSTRAP)(source, window, frame.contentDocument);
    const requests = vi.spyOn(ports().port1, 'postMessage'), reply = vi.spyOn(ports().port2, 'postMessage');
    const host = (source as unknown as { PluginHost: { command(name: string): Promise<unknown> } }).PluginHost;
    mock.command.mockReturnValue(new Promise(() => {}));
    const pending = host.command('read').catch(error => error);
    expect(requests).toHaveBeenCalledWith(expect.objectContaining({ documentNonce: nonce, method: 'command' }));
    await act(async () => source.dispatchEvent(new Event('pagehide')));
    expect((await pending as Error).message).toContain('Pending delivery was cancelled');
    await expect(host.command('read')).rejects.toThrow('no longer current');
    window.dispatchEvent(testMessageEvent({ source, data: { __pluginHost: 1, documentNonce: nonce, id: 'replacement', method: 'voice.open', args: [] } }));
    expect(mock.voice[0].dispose).toHaveBeenCalledTimes(1); expect(mock.voice[0].open).not.toHaveBeenCalled(); expect(reply).not.toHaveBeenCalled();
  });
});
