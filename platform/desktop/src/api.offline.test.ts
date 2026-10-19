// OAIY not answering at all (its local API stopped, or still starting) is said for a person, the same on every page:
// not the browser's "Failed to fetch". An answer OAIY gave, an error with its own words, is kept as it was.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, formatBytes, isNotAnswering, NOT_ANSWERING, plugins, statusOf } from './api';
import { loadErrorText } from './LoadError';

afterEach(() => vi.unstubAllGlobals());

describe('a request OAIY did not answer', () => {
  it('fails with an ApiError saying OAIY isn’t answering, status 0', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    const failed = await plugins.list().catch((e: unknown) => e);
    expect(failed).toBeInstanceOf(ApiError);
    expect((failed as ApiError).message).toBe(NOT_ANSWERING);
    expect(statusOf(failed)).toBe(0);
    expect(isNotAnswering(failed)).toBe(true);
  });

  it('keeps an answer OAIY gave as it was', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: 'aokie was not started' }), { status: 409 })));
    const failed = await plugins.list().catch((e: unknown) => e);
    expect(statusOf(failed)).toBe(409);
    expect(isNotAnswering(failed)).toBe(false);
    expect((failed as Error).message).toBe('aokie was not started');
  });

  it('is said alone in a page’s banner, and any other failure with what could not be read', () => {
    expect(loadErrorText(NOT_ANSWERING, 'The plugins')).toBe(NOT_ANSWERING);
    expect(loadErrorText('the registry is busy', 'The plugins')).toBe('The plugins could not be read: the registry is busy');
    expect(loadErrorText('the registry is busy')).toBe('the registry is busy');
  });
});

describe('sizes', () => {
  it('are in decimal units, as the engines’ catalogue gives them', () => {
    expect(formatBytes(6.1e9)).toBe('6.1 GB');
    expect(formatBytes(4_920_739_232)).toBe('4.9 GB');
    expect(formatBytes(24e6)).toBe('24.0 MB');
    expect(formatBytes(300e3)).toBe('300.0 kB');
    expect(formatBytes(999)).toBe('999 B');
    expect(formatBytes(null)).toBe('—');
  });
});
