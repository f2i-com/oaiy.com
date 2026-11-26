import { afterEach, describe, expect, it, vi } from 'vitest';
import { addEnginesGateway, forgetEnginesKeyForTests, isEnginesKey, useEnginesKey, withEnginesKey } from '../../src/desktop/enginesKey';
import { EMPTY_MEDIA, discoverOaiy, generateSpeech } from '../../src/agent/media';

/** The key the desktop gives the page (`__OAIY_DESKTOP__.enginesKey`, or `/api/engines`' `gatewayKey`). */
const KEY = '7e3a91c0'.repeat(8);
const authOf = (init: RequestInit | undefined): string | null => new Headers(init?.headers).get('authorization');

afterEach(() => {
  forgetEnginesKeyForTests();
  vi.unstubAllGlobals();
});

/** A `fetch` stand-in that keeps what each request went with, and the host it is installed on. */
function recordingHost() {
  const sent: Array<{ url: string; auth: string | null }> = [];
  const host = {
    fetch: vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      sent.push({ url, auth: new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined)).get('authorization') });
      return new Response('{}', { headers: { 'Content-Type': 'application/json' } });
    }) as unknown as typeof fetch,
  };
  return { host, sent };
}

describe("the engines' key", () => {
  it('goes with every request to the engines gateway on this computer, and nowhere else', () => {
    useEnginesKey(KEY, recordingHost().host);
    for (const url of ['http://127.0.0.1:8080/v1/chat/completions', 'http://localhost:8080/v1/discovery', 'http://127.0.0.1:8080/files/images/a.png']) {
      expect(authOf(withEnginesKey(url, {})), url).toBe(`Bearer ${KEY}`);
    }
    for (const url of [
      'https://api.openai.com/v1/chat/completions', // another service
      'http://192.168.1.20:8080/v1/chat/completions', // another computer's OAIY
      'http://localhost:1234/v1/chat/completions', // another server on this computer (LM Studio)
      'http://127.0.0.1:17972/api/engines', // the desktop's own API, which has the desktop's token
      'http://127.0.0.1:8081/v1/models', // a port the desktop did not name
      '/v1/models', // the page's own address
      'not a url',
    ]) {
      expect(authOf(withEnginesKey(url, {})), url).toBeNull();
    }
  });

  it('follows the gateway where the desktop says it moved, on this computer alone', () => {
    useEnginesKey(KEY, recordingHost().host);
    expect(authOf(withEnginesKey('http://127.0.0.1:8081/v1/models'))).toBeNull();
    addEnginesGateway('http://127.0.0.1:8081');
    addEnginesGateway('http://192.168.1.20:8080'); // never off this computer
    addEnginesGateway('https://oaiy.com');
    addEnginesGateway('not a url');
    addEnginesGateway(null);
    expect(authOf(withEnginesKey('http://127.0.0.1:8081/v1/models'))).toBe(`Bearer ${KEY}`);
    expect(authOf(withEnginesKey('http://192.168.1.20:8080/v1/models'))).toBeNull();
    expect(authOf(withEnginesKey('https://oaiy.com/v1/models'))).toBeNull();
  });

  it("leaves a request that has an Authorization of its own (the person's API key) as it is", () => {
    useEnginesKey(KEY, recordingHost().host);
    const mine = { headers: { Authorization: 'Bearer sk-mine', 'Content-Type': 'application/json' } };
    expect(withEnginesKey('http://127.0.0.1:8080/v1/chat/completions', mine)).toBe(mine);
    const lower = { headers: [['authorization', 'Bearer sk-mine']] as [string, string][] };
    expect(withEnginesKey('http://127.0.0.1:8080/v1/chat/completions', lower)).toBe(lower);
    // The request's own headers are kept beside the key.
    const added = withEnginesKey('http://127.0.0.1:8080/v1/chat/completions', { method: 'POST', headers: { 'X-OAIY-Incognito': '1' } });
    expect(added?.method).toBe('POST');
    expect(new Headers(added?.headers).get('x-oaiy-incognito')).toBe('1');
    expect(authOf(added)).toBe(`Bearer ${KEY}`);
  });

  it('is added to a Request given to fetch, keeping its own headers', () => {
    useEnginesKey(KEY, recordingHost().host);
    const request = new Request('http://127.0.0.1:8080/v1/models', { headers: { Accept: 'application/json' } });
    const init = withEnginesKey(request);
    expect(authOf(init)).toBe(`Bearer ${KEY}`);
    expect(new Headers(init?.headers).get('accept')).toBe('application/json');
  });

  it('is nothing until the desktop gives one, and a value that is not a key is none', () => {
    expect(withEnginesKey('http://127.0.0.1:8080/v1/models', {})).toEqual({});
    for (const value of [undefined, null, '', 'short', 42, `${'a'.repeat(20)} b`, `${'a'.repeat(20)}\nX-Evil: 1`, { key: KEY }]) {
      expect(isEnginesKey(value), String(value)).toBe(false);
      useEnginesKey(value, recordingHost().host);
      expect(authOf(withEnginesKey('http://127.0.0.1:8080/v1/models', {})), String(value)).toBeNull();
    }
    expect(isEnginesKey(KEY)).toBe(true);
  });

  it("is sent by the page's own fetch once given, through every caller that uses it", async () => {
    const { host, sent } = recordingHost();
    useEnginesKey(KEY, host);
    useEnginesKey(KEY, host); // (wrapped once)
    await host.fetch('http://127.0.0.1:8080/v1/models');
    await host.fetch('https://api.openai.com/v1/models', { headers: { Authorization: 'Bearer sk-openai' } });
    await host.fetch(new Request('http://localhost:8080/v1/discovery'));
    expect(sent).toEqual([
      { url: 'http://127.0.0.1:8080/v1/models', auth: `Bearer ${KEY}` },
      { url: 'https://api.openai.com/v1/models', auth: 'Bearer sk-openai' },
      { url: 'http://localhost:8080/v1/discovery', auth: `Bearer ${KEY}` },
    ]);
  });

  it("goes with the media service's requests and OAIY's discovery, where no API key was given", async () => {
    const { host, sent } = recordingHost();
    vi.stubGlobal('fetch', host.fetch);
    useEnginesKey(KEY, globalThis);
    await discoverOaiy('http://127.0.0.1:8080', '', undefined, 2000);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      sent.push({ url: String(input), auth: authOf(init) });
      return new Response(new Uint8Array([1, 2]), { headers: { 'Content-Type': 'audio/mpeg' } });
    }));
    // (the stub above stands where the page's fetch is: the key is added by the wrapper around it)
    forgetEnginesKeyForTests();
    useEnginesKey(KEY, globalThis);
    await generateSpeech({ ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080/v1', speechModel: 'tts' }, { input: 'hi', format: 'mp3' });
    await generateSpeech({ ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080/v1', apiKey: 'sk-person', speechModel: 'tts' }, { input: 'hi', format: 'mp3' });
    expect(sent.find((s) => s.url.endsWith('/v1/discovery'))?.auth).toBe(`Bearer ${KEY}`);
    const speech = sent.filter((s) => s.url.endsWith('/audio/speech'));
    expect(speech.map((s) => s.auth)).toEqual([`Bearer ${KEY}`, 'Bearer sk-person']);
  });
});
