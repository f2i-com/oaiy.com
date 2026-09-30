import { describe, expect, it } from 'vitest';
import { EMPTY_MEDIA, type MediaSettings } from '../../src/agent/media';
import { addressOf, forgetOaiy } from '../../src/ui/settings';

const FOUND = 'http://192.168.1.20:8080';

/**
 * The review's F1: clearing the Address and saving used to keep `media.discovered`, so every load asked the old address and sent the
 * typed key to it as a Bearer token. An OAIY that was found can be forgotten: by clearing the address, or by Forget OAIY.
 */
describe('forgetting an OAIY that was found', () => {
  const found: MediaSettings = {
    ...EMPTY_MEDIA,
    baseUrl: `${FOUND}/v1`,
    apiKey: 'sk-oaiy',
    enabled: false,
    imageModel: 'sdxl',
    imageModels: [{ id: 'sdxl' }],
    endpoints: { images: `${FOUND}/v1/images/generations` },
    discovered: { service: 'oaiy-studio', version: '0.1.0', origin: FOUND, at: 1 },
  };

  it('leaves no address, no key, nothing it said and nothing that says it was found; whether the agent may use media stays as chosen', () => {
    const after = forgetOaiy(found);
    expect(after.discovered).toBeUndefined();
    expect(after.endpoints).toBeUndefined();
    expect(after.baseUrl).toBe('');
    expect(after.apiKey).toBe('');
    expect(after.imageModels).toEqual([]);
    expect(after.imageModel).toBeUndefined();
    expect(after.enabled).toBe(false);
  });

  it('does not change what it was given, and shares no list with the empty settings', () => {
    const before = JSON.stringify(found);
    const a = forgetOaiy(found);
    const b = forgetOaiy(found);
    expect(JSON.stringify(found)).toBe(before);
    a.imageModels.push({ id: 'x' });
    expect(b.imageModels).toEqual([]);
    expect(EMPTY_MEDIA.imageModels).toEqual([]);
  });

  it('an address typed is the found OAIY only when it is at the origin that was found: an empty one, another one, and one that is not an address are not', () => {
    expect(addressOf(`${FOUND}/v1`, FOUND)).toBe(true);
    expect(addressOf('192.168.1.20:8080', FOUND)).toBe(true);
    expect(addressOf('', FOUND)).toBe(false);
    expect(addressOf('   ', FOUND)).toBe(false);
    expect(addressOf('http://192.168.1.21:8080/v1', FOUND)).toBe(false);
    expect(addressOf('http://192.168.1.20:9090', FOUND)).toBe(false);
    expect(addressOf('not an address', FOUND)).toBe(false);
    expect(addressOf('http://', FOUND)).toBe(false);
  });
});
