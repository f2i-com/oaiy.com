// The key OAIY's own window sends with every request to OAIY's local API (windowKey.ts).
//
// On Windows the window's origin (`http://tauri.localhost`) is a name any page on this computer's port 80 has too, so
// the API believes it only with the key the desktop hands this page. These hold the page's half: the key is asked of
// the desktop once, goes on every request to the API and nowhere else, never into an address, and a window that
// cannot be given it says so in words for a person instead of going quiet.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, API_BASE, contacts, modules, network, plugins, voices } from './api';
import { PluginAiSession } from './pluginAi';
import {
  forgetWindowKey,
  KEY_REFUSAL,
  retryWindowKey,
  watchWindowKeyProblem,
  WINDOW_KEY_ERROR,
  windowFetch,
  windowKey,
  windowKeyProblem,
  withWindowKey,
} from './windowKey';

const KEY = '5f3c9a0e1b7d4c2f8a6e0d9b3c1f7a5e2d4b6c8e0f1a3b5c7d9e1f2a4b6c8d0e';
type Invoke = (cmd: string, args?: unknown) => Promise<unknown>;
const bridged = window as unknown as { __TAURI_INTERNALS__?: { invoke: Invoke } };

/** OAIY's own window: the native bridge is there, and answers `window_token` as `answer` says. */
function inOaiysWindow(answer: Invoke = async () => KEY) {
  const invoke = vi.fn<Invoke>(answer);
  bridged.__TAURI_INTERNALS__ = { invoke };
  return invoke;
}

const ok = (body: unknown = {}) => new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
const refused = () => new Response(JSON.stringify({ error: KEY_REFUSAL }), { status: 403, headers: { 'Content-Type': 'application/json' } });

/** `fetch`, answering each request with what `answer` makes for it. */
function api(answer: () => Response = ok) {
  const fetch = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(async () => answer());
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

const sent = (fetch: ReturnType<typeof api>, call = 0) => (fetch.mock.calls[call][1]?.headers ?? {}) as Record<string, string>;

/** One request to the API, made with the window's key, that the API answers: what takes a problem away. */
const answeredOnce = async () => {
  api();
  await windowFetch(`${API_BASE}/api/config`);
};

beforeEach(async () => {
  delete bridged.__TAURI_INTERNALS__;
  forgetWindowKey();
  // (a problem left by the test before is gone once the window has its key and is answered)
  if (windowKeyProblem()) {
    inOaiysWindow();
    await answeredOnce();
    delete bridged.__TAURI_INTERNALS__;
    forgetWindowKey();
  }
});

afterEach(() => {
  delete bridged.__TAURI_INTERNALS__;
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('in a browser tab (no native bridge)', () => {
  it('has no key, and a request goes exactly as it was given', async () => {
    const fetch = api();
    expect(await windowKey()).toBeNull();
    const init = { headers: { 'Content-Type': 'application/json' }, method: 'POST', body: '{}' };
    await windowFetch(`${API_BASE}/api/services`, init);
    expect(fetch).toHaveBeenCalledWith(`${API_BASE}/api/services`, init);
    await windowFetch(`${API_BASE}/api/health`);
    expect(fetch).toHaveBeenLastCalledWith(`${API_BASE}/api/health`);
    expect(fetch.mock.calls.some(([, i]) => 'Authorization' in ((i?.headers ?? {}) as object))).toBe(false);
  });

  it('is not told the window is refused when the API refuses it: it is no window of OAIY’s', async () => {
    api(refused);
    await windowFetch(`${API_BASE}/api/config`);
    expect(windowKeyProblem()).toBeNull();
  });
});

describe('in OAIY’s own window', () => {
  it('asks the desktop for the key once, and sends it on every request to the API', async () => {
    const invoke = inOaiysWindow();
    const fetch = api();
    await windowFetch(`${API_BASE}/api/config`, { headers: { 'Content-Type': 'application/json' } });
    await windowFetch(`${API_BASE}/api/health`);
    await Promise.all([windowFetch(`${API_BASE}/api/plugins`), windowFetch(`${API_BASE}/api/modules`, { headers: { 'If-None-Match': '"3"' } })]);
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(invoke).toHaveBeenCalledWith('window_token', {});
    expect(fetch).toHaveBeenCalledTimes(4);
    for (let call = 0; call < 4; call++) expect(sent(fetch, call).Authorization).toBe(`Bearer ${KEY}`);
    // The caller's own headers are kept beside it, and the rest of the request is as it was given.
    expect(sent(fetch, 0)).toEqual({ 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` });
    expect(sent(fetch, 3)['If-None-Match']).toBe('"3"');
  });

  it('asks before the first request goes, however long the desktop takes to answer', async () => {
    let hand!: (key: string) => void;
    inOaiysWindow(() => new Promise<string>((resolve) => { hand = resolve; }));
    const fetch = api();
    const asked = windowFetch(`${API_BASE}/api/config`);
    await Promise.resolve();
    expect(fetch).not.toHaveBeenCalled();
    hand(KEY);
    await asked;
    expect(sent(fetch).Authorization).toBe(`Bearer ${KEY}`);
  });

  it('never puts the key in an address, and sends it to no address but the API’s own', async () => {
    const invoke = inOaiysWindow();
    const fetch = api();
    const elsewhere = [
      'https://example.com/api/config',
      'http://127.0.0.1:17973/api/config',
      'http://localhost:17972/api/config',
      'http://127.0.0.1.evil.example:17972/api/config',
      'http://tauri.localhost/api/config',
      '/fonts/inter.woff2',
    ];
    for (const url of elsewhere) await windowFetch(url, { headers: { Accept: '*/*' } });
    expect(invoke).not.toHaveBeenCalled();
    expect(fetch.mock.calls.map(([, init]) => init?.headers)).toEqual(elsewhere.map(() => ({ Accept: '*/*' })));
    await windowFetch(`${API_BASE}/api/config?since=4`);
    for (const [url] of fetch.mock.calls) expect(url).not.toContain(KEY);
    expect(fetch.mock.lastCall![0]).toBe(`${API_BASE}/api/config?since=4`);
  });

  it('replaces an Authorization of the caller’s own, in whatever shape the headers came', () => {
    const want = { Accept: 'text/plain', Authorization: `Bearer ${KEY}` };
    expect(withWindowKey({ headers: { Accept: 'text/plain', authorization: 'Bearer other' } }, KEY).headers).toEqual(want);
    expect(withWindowKey({ headers: [['Accept', 'text/plain'], ['AUTHORIZATION', 'Bearer other']] }, KEY).headers).toEqual(want);
    expect(withWindowKey({ headers: new Headers({ Accept: 'text/plain', Authorization: 'Bearer other' }) }, KEY).headers).toEqual({ accept: 'text/plain', Authorization: `Bearer ${KEY}` });
    expect(withWindowKey(undefined, KEY)).toEqual({ headers: { Authorization: `Bearer ${KEY}` } });
    const kept = withWindowKey({ method: 'PUT', body: '{}', cache: 'no-store' }, KEY);
    expect(kept).toMatchObject({ method: 'PUT', body: '{}', cache: 'no-store' });
  });

  it('sends it from every place the dashboard asks the API', async () => {
    inOaiysWindow();
    const keyed = async (ask: () => Promise<unknown>, answer: () => Response = ok) => {
      const fetch = api(answer);
      await ask();
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(String(fetch.mock.calls[0][0]).startsWith(`${API_BASE}/api/`)).toBe(true);
      return sent(fetch).Authorization;
    };
    const want = `Bearer ${KEY}`;
    // The JSON requests every page makes (api.ts's `request`), and the four that read something else.
    expect(await keyed(() => plugins.list(), () => ok([]))).toBe(want);
    expect(await keyed(() => modules.list('"1"'), () => ok({ revision: 1, modules: [], contributions: {}, warnings: [] }))).toBe(want);
    expect(await keyed(() => voices.hear('amy'), () => new Response('RIFF', { status: 200 }))).toBe(want);
    expect(await keyed(() => contacts.exportCsv(), () => new Response('name,number', { status: 200 }))).toBe(want);
    expect(await keyed(() => network.picture('/api/ai/engine/gateway/files/a.png'), () => new Response('png', { status: 200 }))).toBe(want);
    // What a plugin's screen asks for is asked here, with the dashboard's key: the screen is given none.
    const completion = { requestId: 'r1', sourceId: 'provider:test', prompt: 'Say hello.', maxOutputChars: 64 };
    const answer = () => ok({ requestId: 'r1', sourceId: 'provider:test', text: 'Hello' });
    expect(await keyed(() => new PluginAiSession('probe', API_BASE).complete(completion), answer)).toBe(want);
  });
});

describe('a window that cannot be given its key', () => {
  it.each([
    ['is refused', async () => { throw new Error('Only OAIY’s own window can do that.'); }],
    ['is given nothing', async () => undefined],
    ['is given something that is no key', async () => 'short'],
    ['is given a key with a line break in it', async () => `${KEY}\nX-Other: 1`],
    ['is given an object', async () => ({ token: KEY })],
  ] as Array<[string, Invoke]>)('still sends its request, without one, when it %s', async (_how, answer) => {
    const invoke = inOaiysWindow(answer);
    const fetch = api();
    expect(await windowKey()).toBeNull();
    const response = await windowFetch(`${API_BASE}/api/config`, { headers: { Accept: '*/*' } });
    expect(response.status).toBe(200);
    expect(fetch).toHaveBeenCalledWith(`${API_BASE}/api/config`, { headers: { Accept: '*/*' } });
    // Answered (an origin no page can copy, a Mac's and Linux's): nothing is wrong, and nothing is said.
    expect(windowKeyProblem()).toBeNull();
    // It asks the desktop again at the next request, not once for good.
    await windowFetch(`${API_BASE}/api/config`);
    expect(invoke.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('does not wait for a desktop that never answers', async () => {
    vi.useFakeTimers();
    inOaiysWindow(() => new Promise(() => {}));
    const fetch = api();
    const asked = windowFetch(`${API_BASE}/api/config`);
    await vi.advanceTimersByTimeAsync(4_000);
    expect(fetch).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(2_000);
    await asked;
    expect(fetch).toHaveBeenCalledWith(`${API_BASE}/api/config`);
  });

  it('says so, in words for a person, once the API refuses it for the lack of the key', async () => {
    inOaiysWindow(async () => { throw new Error('refused'); });
    const told = vi.fn();
    const stop = watchWindowKeyProblem(told);
    api(refused);
    expect(windowKeyProblem()).toBeNull();
    // A page's own request fails with the same words, not with the API's.
    const failed = await plugins.list().catch((e: unknown) => e);
    expect(failed).toBeInstanceOf(ApiError);
    expect((failed as ApiError).status).toBe(403);
    expect((failed as ApiError).message).toBe(WINDOW_KEY_ERROR);
    expect(windowKeyProblem()).toBe(WINDOW_KEY_ERROR);
    expect(told).toHaveBeenCalledTimes(1);
    // Said once, however many requests are refused.
    await windowFetch(`${API_BASE}/api/config`);
    expect(told).toHaveBeenCalledTimes(1);
    stop();
  });

  it('is not mistaken for any other refusal', async () => {
    inOaiysWindow(async () => { throw new Error('refused'); });
    for (const other of [
      new Response(JSON.stringify({ error: 'origin not allowed' }), { status: 403 }),
      new Response(JSON.stringify({ error: { code: 'forbidden', message: KEY_REFUSAL } }), { status: 403 }),
      new Response(JSON.stringify({ error: KEY_REFUSAL }), { status: 409 }),
      new Response('not json', { status: 403 }),
    ]) {
      api(() => other);
      await windowFetch(`${API_BASE}/api/config`);
      expect(windowKeyProblem()).toBeNull();
    }
  });

  it('is well again when the desktop gives the key and the API answers it', async () => {
    let give = false;
    const invoke = inOaiysWindow(async () => { if (!give) throw new Error('refused'); return KEY; });
    let fetch = api(refused);
    await windowFetch(`${API_BASE}/api/config`);
    expect(windowKeyProblem()).toBe(WINDOW_KEY_ERROR);
    expect(sent(fetch).Authorization).toBeUndefined();
    // "Try again", while the desktop still gives none: the problem stays.
    await retryWindowKey();
    expect(windowKeyProblem()).toBe(WINDOW_KEY_ERROR);
    give = true;
    await retryWindowKey();
    expect(windowKeyProblem()).toBeNull();
    fetch = api();
    await windowFetch(`${API_BASE}/api/config`);
    expect(sent(fetch).Authorization).toBe(`Bearer ${KEY}`);
    expect(invoke.mock.calls.every(([cmd]) => cmd === 'window_token')).toBe(true);
  });

  it('asks for the key afresh when one it sent is not taken', async () => {
    const invoke = inOaiysWindow();
    api(refused);
    await windowFetch(`${API_BASE}/api/config`);
    expect(windowKeyProblem()).toBe(WINDOW_KEY_ERROR);
    api();
    await windowFetch(`${API_BASE}/api/config`);
    expect(invoke).toHaveBeenCalledTimes(2);
    expect(windowKeyProblem()).toBeNull();
  });
});
