import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { connectorRequest, list, toast } = vi.hoisted(() => ({
  connectorRequest: vi.fn(), list: vi.fn(), toast: { push: vi.fn() },
}));
vi.mock('./api', () => ({
  API_BASE: 'http://127.0.0.1:17972', plugins: { list }, bridge: { connectorRequest }, companion: {},
}));
vi.mock('./Toasts', () => ({ useToast: () => toast }));
vi.mock('./useModules', () => ({ useModules: () => null, moduleOn: () => false }));

import PluginScreenPage, { HOST_BOOTSTRAP } from './PluginScreenPage';

const details = { code: 'revision_conflict', message: 'The saved revision has changed.', version: 1 };
const typedRefusal = { ok: true, result: { ok: false, data: { version: 1, error: { code: details.code, message: details.message } } } };
const plugin = {
  id: 'test-plugin', state: 'running', manifest: { name: 'Test plugin', version: '1.0', ui: {
    nav: [{ id: 'settings', screen: 'settings' }, { id: 'other', screen: 'settings' }],
    screens: [{ id: 'settings', entry: 'index.html', files: ['index.html'] }],
  } },
};

let container: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.useFakeTimers();
  connectorRequest.mockReset();
  list.mockReset().mockResolvedValue({ plugins: [plugin] });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true, text: async () => '<p>Plugin settings</p>', arrayBuffer: async () => new ArrayBuffer(0),
  }));
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function mount(navId = 'settings') {
  await act(async () => root.render(<PluginScreenPage pluginId="test-plugin" navId={navId} />));
  const frame = container.querySelector('iframe')!;
  expect(frame).not.toBeNull();
  expect(frame.srcdoc).toContain(HOST_BOOTSTRAP);
  return frame;
}

/** Execute the production bootstrap inside the mounted frame. jsdom does not
 * execute srcdoc scripts or supply real postMessage source identities, so the
 * two message transports below provide those identities. Both real message
 * listeners, command dispatch, envelope handling and serialization still run. */
function connect(frame: HTMLIFrameElement) {
  const frameWindow = frame.contentWindow!;
  const receive = (data: unknown, source: Window | null = window) => {
    frameWindow.dispatchEvent(new MessageEvent('message', { source, data }));
  };
  const requests = vi.spyOn(window, 'postMessage').mockImplementation((data: unknown) => {
    window.dispatchEvent(new MessageEvent('message', { source: frameWindow, data }));
  });
  const replies = vi.spyOn(frameWindow, 'postMessage').mockImplementation((data: unknown) => receive(data));
  new Function('window', 'parent', 'document', 'setTimeout', 'clearTimeout', HOST_BOOTSTRAP)(
    frameWindow, window, frame.contentDocument, setTimeout, clearTimeout,
  );
  const host = (frameWindow as unknown as { PluginHost: {
    command: (name: string, payload?: unknown) => Promise<unknown>;
    aiComplete: (request: unknown) => Promise<unknown>;
    aiCancel: (requestId: string) => Promise<unknown>;
    aiSources: () => Promise<unknown>;
  } }).PluginHost;
  return { host, requests, replies, receive };
}

describe('plugin command RPC through the mounted host and actual bootstrap', () => {
  it('carries a real bounded AI request through the bootstrap and mounted plugin-owned route', async () => {
    list.mockResolvedValue({plugins:[{...plugin,manifest:{...plugin.manifest,capabilities:['oaiy.ai.complete']}}]});
    const {host,replies} = connect(await mount());
    const request = {requestId:'screen-request',sourceId:'provider:test',prompt:'Synthetic evidence only.',maxOutputChars:256};
    const result = {requestId:request.requestId,sourceId:request.sourceId,text:'Grounded answer',model:'test-model'};
    const fetch = vi.mocked(globalThis.fetch).mockResolvedValue(new Response(JSON.stringify(result)));
    await expect(host.aiComplete(request)).resolves.toEqual(result);
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:17972/api/plugins/test-plugin/ai/complete',expect.objectContaining({body:JSON.stringify(request)}));
    expect(replies).toHaveBeenCalledExactlyOnceWith({__pluginHost:1,id:'r1',ok:true,data:result},'*');
  });

  it('delivers a typed provider refusal through the production AI bootstrap', async () => {
    const {host} = connect(await mount());
    vi.mocked(globalThis.fetch).mockResolvedValue(new Response(JSON.stringify({error:{code:'no_provider',message:'No scoped provider is configured.'}}), {status:404}));
    await expect(host.aiComplete({requestId:'none',sourceId:'provider:test',prompt:'Synthetic evidence only.',maxOutputChars:256}))
      .rejects.toMatchObject({name:'PluginCommandError',code:'no_provider',message:'No scoped provider is configured.'});
  });

  it('closing a screen aborts its completion and suppresses a delayed RPC reply', async () => {
    const {host,replies} = connect(await mount());
    let finish!: (value: Response) => void;
    let signal!: AbortSignal;
    vi.mocked(globalThis.fetch).mockImplementation((url,options) => {
      if (String(url).endsWith('/ai/complete')) {
        signal=options!.signal!;
        return new Promise((resolve) => {finish=resolve;});
      }
      return Promise.resolve(new Response(JSON.stringify({requestId:'closing',cancelled:true})));
    });
    void host.aiComplete({requestId:'closing',sourceId:'provider:test',prompt:'Synthetic evidence only.',maxOutputChars:256}).catch(() => {});
    await act(async () => root.render(<PluginScreenPage pluginId="test-plugin" navId="other" />));
    expect(signal.aborted).toBe(true);
    finish(new Response(JSON.stringify({requestId:'closing',sourceId:'provider:test',text:'stale'})));
    await act(async () => {await Promise.resolve();});
    expect(replies).not.toHaveBeenCalled();
  });

  it.each([
    ['inner versioned refusal', typedRefusal],
    ['inner SDK refusal', { ok: true, result: { ok: false, error: details } }],
    ['outer refusal', { ok: false, error: details }],
  ])('rejects a typed %s with the same bounded fields', async (_name, response) => {
    const { host, replies } = connect(await mount());
    connectorRequest.mockResolvedValue(response);
    const timers = vi.getTimerCount();
    const success = vi.fn();
    const error = await host.command('settings.save', { revision: 1 }).then(success, (error: unknown) => error);
    expect(error).toBeInstanceOf(Error);
    expect(error).toMatchObject({ name: 'PluginCommandError', ...details });
    expect(success).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(timers);
    expect(connectorRequest).toHaveBeenCalledWith('test-plugin', 'settings.save', { revision: 1 }, expect.stringMatching(/^ui-settings\.save-/));
    expect(replies).toHaveBeenCalledExactlyOnceWith({
      __pluginHost: 1, id: 'r1', ok: false, error: details.message, errorDetails: details,
    }, '*');
  });

  it.each([
    ['string', { ok: true, result: { ok: false, error: 'Not connected.' } }, 'Not connected.'],
    ['object', { ok: true, result: { ok: false, error: { message: 'Still busy.', retryable: true } } }, 'Still busy.'],
    ['unknown shape', { ok: true, result: { ok: false, error: { privateData: 'not forwarded' } } }, 'The plugin could not complete this action.'],
    ['missing error', { ok: false, result: { ok: true, data: 'not success' } }, 'The desktop could not complete this plugin request.'],
    ['invalid version', { ok: true, result: { ok: false, error: { ...details, version: '1' } } }, details.message],
  ])('keeps malformed/legacy %s refusals rejected and string-compatible', async (_name, response, message) => {
    const { host, replies } = connect(await mount());
    connectorRequest.mockResolvedValue(response);
    const error = await host.command('settings.save').catch((error: unknown) => error);
    expect(error).toBeInstanceOf(Error);
    expect(error).toMatchObject({ name: 'Error', message });
    expect(error).not.toHaveProperty('code');
    expect(replies).toHaveBeenCalledExactlyOnceWith({ __pluginHost: 1, id: 'r1', ok: false, error: message }, '*');
  });

  it.each([
    ['SDK data', { ok: true, data: { saved: true } }, { saved: true }],
    ['legacy scalar', 'ready', 'ready'],
    ['legacy array', [1, 2], [1, 2]],
    ['handled domain error', { ok: true, data: { version: 1, error: details } }, { version: 1, error: details }],
  ])('preserves successful %s', async (_name, result, expected) => {
    const { host, replies } = connect(await mount());
    connectorRequest.mockResolvedValue({ ok: true, result });
    await expect(host.command('settings.read')).resolves.toEqual(expected);
    expect(replies).toHaveBeenCalledExactlyOnceWith({ __pluginHost: 1, id: 'r1', ok: true, data: expected }, '*');
  });

  it('allows an explicit retry after rejection with a new correlation and idempotency key', async () => {
    const { host, replies } = connect(await mount());
    connectorRequest.mockResolvedValueOnce(typedRefusal).mockResolvedValueOnce({ ok: true, result: { ok: true, data: { saved: true } } });
    await expect(host.command('settings.save')).rejects.toMatchObject({ code: details.code });
    vi.setSystemTime(Date.now() + 1);
    await expect(host.command('settings.save')).resolves.toEqual({ saved: true });
    expect(replies.mock.calls.map(([reply]) => reply)).toEqual([
      { __pluginHost: 1, id: 'r1', ok: false, error: details.message, errorDetails: details },
      { __pluginHost: 1, id: 'r2', ok: true, data: { saved: true } },
    ]);
    expect(connectorRequest.mock.calls[0][3]).not.toBe(connectorRequest.mock.calls[1][3]);
  });

  it('ignores requests from another frame and malformed correlation or method fields', async () => {
    const frame = await mount();
    const foreign = document.createElement('iframe');
    container.appendChild(foreign);
    const request = { __pluginHost: 1, id: 'r1', method: 'command', args: ['settings.save'] };
    window.dispatchEvent(new MessageEvent('message', { source: foreign.contentWindow, data: request }));
    for (const data of [
      { ...request, __pluginHost: 2 }, { ...request, id: {} }, { ...request, id: '' },
      { ...request, id: 'x'.repeat(129) }, { ...request, method: {} }, { ...request, args: {} },
    ]) window.dispatchEvent(new MessageEvent('message', { source: frame.contentWindow, data }));
    expect(connectorRequest).not.toHaveBeenCalled();
    foreign.remove();
  });

  it('drops completion from an unmounted screen and leaves a new screen independent', async () => {
    let finish!: (response: unknown) => void;
    connectorRequest.mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    const first = connect(await mount());
    // A removed document may retain its promise until its own deadline, but no
    // reply can be delivered into its replacement (whose sequence starts at r1).
    void first.host.command('settings.save').catch(() => {});
    first.requests.mockRestore();
    const second = connect(await mount('other'));
    connectorRequest.mockResolvedValueOnce({ ok: true, result: { ok: true, data: 'new screen' } });
    await expect(second.host.command('settings.read')).resolves.toBe('new screen');
    await act(async () => finish(typedRefusal));
    expect(first.replies).not.toHaveBeenCalled();
    expect(second.replies).toHaveBeenCalledTimes(1);
  });
});

describe('actual bootstrap reply validation and recovery', () => {
  it.each([false, undefined, null, 'true', 1])('requires literal success, rejecting ok=%j', async (ok) => {
    const { host, requests, receive } = connect(await mount());
    requests.mockImplementation(() => {});
    const result = host.command('settings.save');
    const rejected = expect(result).rejects.toMatchObject({ name: 'Error', message: 'Refused.' });
    receive({ __pluginHost: 1, id: 'r1', ok, data: 'must not resolve', error: 'Refused.' });
    await rejected;
  });

  it.each([
    ['extra property', { ...details, privateData: { secret: true } }],
    ['prototype', Object.assign(Object.create({ privateData: true }), details)],
    ['accessor', Object.defineProperty({ ...details }, 'code', { get: () => { throw new Error('must not execute'); } })],
    ['symbol', { ...details, [Symbol('private')]: true }],
    ['missing message', { code: details.code }],
    ['invalid code', { ...details, code: '<script>' }],
    ['oversized code', { ...details, code: 'x'.repeat(65) }],
    ['empty message', { ...details, message: '\u0000\n' }],
    ['invalid version', { ...details, version: 1.5 }],
    ['undefined version', { ...details, version: undefined }],
  ])('rejects a reply with malformed %s metadata without trusting it', async (_name, errorDetails) => {
    const { host, requests, receive } = connect(await mount());
    requests.mockImplementation(() => {});
    const result = host.command('settings.save').catch((error: unknown) => error);
    receive({ __pluginHost: 1, id: 'r1', ok: false, error: 'Legacy failure.', errorDetails });
    const error = await result;
    expect(error).toMatchObject({ name: 'Error', message: 'Legacy failure.' });
    expect(error).not.toHaveProperty('code');
    expect(error).not.toHaveProperty('privateData');
  });

  it('constructs a local Error with only bounded typed metadata', async () => {
    const { host, requests, receive } = connect(await mount());
    requests.mockImplementation(() => {});
    const result = host.command('settings.save').catch((error: unknown) => error);
    receive({ __pluginHost: 1, id: 'r1', ok: false, error: 'Legacy failure.', errorDetails: {
      code: 'new_vendor_code', message: '\u0000 Cannot save.\n' + 'x'.repeat(2000),
    } });
    const error = await result as Error;
    expect(error).toBeInstanceOf(Error);
    expect(error).toMatchObject({ name: 'PluginCommandError', code: 'new_vendor_code' });
    expect(error.message.length).toBeLessThanOrEqual(1024);
    expect(error.message).not.toMatch(/[\u0000-\u001f\u007f]/);
    expect(error).not.toHaveProperty('version');
    expect(Object.keys(error).sort()).toEqual(['code', 'name']);
  });

  it('ignores wrong-frame, forged-marker and wrong-id replies without consuming a pending request', async () => {
    const { host, requests, receive } = connect(await mount());
    requests.mockImplementation(() => {});
    const settled = vi.fn();
    const result = host.command('settings.save').then(settled, (error: unknown) => error);
    receive({ __pluginHost: 1, id: 'r1', ok: true, data: 'forged' }, null);
    for (const reply of [
      { __pluginHost: 2, id: 'r1' }, { __pluginHost: 1, id: 'r2' },
      { __pluginHost: 1, id: '__proto__' }, { __pluginHost: 1, id: 'constructor' },
      { __pluginHost: 1, id: { toString: () => 'r1' } },
    ]) receive({ ...reply, ok: true, data: 'forged' });
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();
    receive({ __pluginHost: 1, id: 'r1', ok: false, error: details.message, errorDetails: details });
    expect(await result).toMatchObject({ name: 'PluginCommandError', ...details });
  });

  it('correlates concurrent replies and ignores a late completion after a timeout and retry', async () => {
    const { host, requests, receive } = connect(await mount());
    requests.mockImplementation(() => {});
    const first = host.command('settings.save').catch((error: unknown) => error);
    const second = host.command('settings.read');
    receive({ __pluginHost: 1, id: 'r2', ok: true, data: 'second' });
    expect(await second).toBe('second');
    await vi.advanceTimersByTimeAsync(20000);
    expect(await first).toMatchObject({ name: 'Error', message: expect.stringContaining('outcome may be unknown') });
    const retry = host.command('settings.save');
    const settled = vi.fn();
    void retry.then(settled);
    receive({ __pluginHost: 1, id: 'r1', ok: true, data: 'old result' });
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();
    receive({ __pluginHost: 1, id: 'r3', ok: true, data: 'retried' });
    expect(await retry).toBe('retried');
    expect(requests).toHaveBeenCalledTimes(3);
  });
});
