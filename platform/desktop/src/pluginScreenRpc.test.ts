import { describe, expect, it, vi } from 'vitest';
import { PluginScreenDocument } from './pluginScreenRpc';
import { testMessageEvent, testPorts } from './pluginScreenRpc.testTransport';

describe('document-bound opaque iframe channel', () => {
  const nonce = '11111111-1111-4111-8111-111111111111';
  it('requires the owned WindowProxy and exact own nonce, without reading accessors', () => {
    const frame = document.createElement('iframe'); document.body.append(frame);
    const channel = new PluginScreenDocument(nonce, frame.contentWindow, vi.fn());
    const event = (data: unknown, source: Window | null = frame.contentWindow) => testMessageEvent({ source, data });
    expect(channel.accepts(event({ documentNonce: nonce }))).toBe(true);
    for (const data of [{}, { documentNonce: 'another-document' }, Object.create({ documentNonce: nonce }), Object.defineProperty({}, 'documentNonce', { get() { throw Error('Do not read accessors'); } })]) expect(channel.accepts(event(data))).toBe(false);
    expect(channel.accepts(event({ documentNonce: nonce }, null))).toBe(false);
    frame.remove(); channel.revoke();
  });
  it('preserves one initial load, then permanently revokes even when the WindowProxy and nonce are replayed', () => {
    const source = { postMessage: vi.fn() } as unknown as Window, revoke = vi.fn();
    const channel = new PluginScreenDocument(nonce, source, revoke);
    const ports = testPorts(), receive = vi.fn();
    channel.connect(testMessageEvent({ source, data: { documentNonce: nonce }, ports: [ports.port2 as unknown as MessagePort] }), receive);
    const reply = vi.spyOn(ports.port2, 'postMessage');
    channel.loaded();
    expect(channel.signal.aborted).toBe(false);
    channel.post({ data: 'private-reply' });
    expect(reply).toHaveBeenCalledWith({ __pluginHost: 1, documentNonce: nonce, data: 'private-reply' });
    channel.loaded(); channel.loaded(); channel.revoke();
    expect(channel.signal.aborted).toBe(true); expect(revoke).toHaveBeenCalledTimes(1);
    expect(channel.accepts(testMessageEvent({ source, data: { documentNonce: nonce } }))).toBe(false);
    channel.post({ voiceEvent: { text: 'old private transcript' } });
    expect(reply).toHaveBeenCalledTimes(1); expect(source.postMessage).not.toHaveBeenCalled();
  });
  it('explicit close revokes before later load and no absent window can gain a capability', () => {
    const revoke = vi.fn(), channel = new PluginScreenDocument(nonce, null, revoke);
    channel.revoke(); channel.loaded();
    expect(revoke).toHaveBeenCalledTimes(1);
    expect(channel.accepts(testMessageEvent({ data: { documentNonce: nonce } }))).toBe(false);
  });
  it('binds only the original port, even when a second handshake copies the exact nonce and WindowProxy', () => {
    const source = {} as Window, channel = new PluginScreenDocument(nonce, source, vi.fn());
    const original = testPorts(), copied = testPorts(), receive = vi.fn();
    const connect = (ports: ReturnType<typeof testPorts>) => testMessageEvent({ source, data: { documentNonce: nonce }, ports: [ports.port2 as unknown as MessagePort] });
    expect(channel.connect(connect(original), receive)).toBe(true);
    expect(channel.connect(connect(copied), receive)).toBe(false);
    copied.port1.postMessage({ documentNonce: nonce, method: 'voice.open' });
    expect(receive).not.toHaveBeenCalled();
    original.port1.postMessage({ documentNonce: nonce, method: 'snapshot' });
    expect(receive).toHaveBeenCalledTimes(1);
    const reply = vi.spyOn(copied.port2, 'postMessage');
    channel.post({ voiceEvent: { text: 'private original callback' } });
    expect(reply).not.toHaveBeenCalled();
    channel.revoke();
  });
});
