import { afterEach, describe, expect, it, vi } from 'vitest';
import { addOaiyOrigin, incognitoHeaderFor, setIncognito } from '../../src/privacy';
import { EMPTY_MEDIA, generateSpeech } from '../../src/agent/media';

afterEach(() => {
  setIncognito(false);
  vi.unstubAllGlobals();
});

describe('incognito', () => {
  it('tells OAIY, and only OAIY, to keep nothing, and only while incognito', () => {
    addOaiyOrigin('http://127.0.0.1:8080/v1');
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/images/generations')).toEqual({});
    setIncognito(true);
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/images/generations')).toEqual({ 'X-OAIY-Incognito': '1' });
    // Another service is not sent a header it may refuse the request for.
    expect(incognitoHeaderFor('https://api.openai.com/v1/chat/completions')).toEqual({});
    expect(incognitoHeaderFor('not a url')).toEqual({});
  });

  it('names the incognito session to OAIY, so its next request reuses what OAIY read', () => {
    addOaiyOrigin('http://127.0.0.1:8080/v1');
    setIncognito(true, 'project-1');
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/chat/completions')).toEqual({ 'X-OAIY-Incognito': '1', 'X-OAIY-Session': 'project-1' });
    expect(incognitoHeaderFor('https://api.openai.com/v1/chat/completions')).toEqual({});
    // Out of incognito, no session is named.
    setIncognito(false, 'project-1');
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/chat/completions')).toEqual({});
  });

  it("is sent with the media service's requests", async () => {
    addOaiyOrigin('http://127.0.0.1:8080');
    setIncognito(true);
    const headers: Array<Record<string, string>> = [];
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
      headers.push(init?.headers as Record<string, string>);
      return new Response(new Uint8Array([1, 2]), { headers: { 'Content-Type': 'audio/mpeg' } });
    }));
    await generateSpeech({ ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', speechModel: 'tts' }, { input: 'hi', format: 'mp3' });
    expect(headers[0]['X-OAIY-Incognito']).toBe('1');
  });
});
