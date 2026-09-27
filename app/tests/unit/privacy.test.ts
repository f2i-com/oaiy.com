import { afterEach, describe, expect, it, vi } from 'vitest';
import { addNrobOrigin, incognitoHeaderFor, setIncognito } from '../../src/privacy';
import { EMPTY_MEDIA, generateSpeech } from '../../src/agent/media';

afterEach(() => {
  setIncognito(false);
  vi.unstubAllGlobals();
});

describe('incognito', () => {
  it('tells nrob, and only nrob, to keep nothing, and only while incognito', () => {
    addNrobOrigin('http://127.0.0.1:8080/v1');
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/images/generations')).toEqual({});
    setIncognito(true);
    expect(incognitoHeaderFor('http://127.0.0.1:8080/v1/images/generations')).toEqual({ 'X-NROB-Incognito': '1' });
    // Another service is not sent a header it may refuse the request for.
    expect(incognitoHeaderFor('https://api.openai.com/v1/chat/completions')).toEqual({});
    expect(incognitoHeaderFor('not a url')).toEqual({});
  });

  it("is sent with the media service's requests", async () => {
    addNrobOrigin('http://127.0.0.1:8080');
    setIncognito(true);
    const headers: Array<Record<string, string>> = [];
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
      headers.push(init?.headers as Record<string, string>);
      return new Response(new Uint8Array([1, 2]), { headers: { 'Content-Type': 'audio/mpeg' } });
    }));
    await generateSpeech({ ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', speechModel: 'tts' }, { input: 'hi', format: 'mp3' });
    expect(headers[0]['X-NROB-Incognito']).toBe('1');
  });
});
