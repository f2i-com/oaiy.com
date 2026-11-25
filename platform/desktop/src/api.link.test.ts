// The requests the Connections panel's two ways of forgetting make, as the host sees them.
//
// `DELETE /api/link` forgets the link. `DELETE /api/link?copiesOnly=true` only takes the copies of an earlier key that a forget could
// not remove, and leaves a link that is there: it is what "Remove copies" sends. A button that sent the plain request would be a
// disconnect that is not asked for, and one that sent `copiesOnly=false` the same, so the panel's tests (which mock the client) are
// not enough: these read the request itself.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, link, openExternal } from './api';

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockImplementation(async () => new Response(JSON.stringify({ linked: false, attempt: { phase: 'idle' }, available: [] }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('api · link', () => {
  it('forgets only the copies with Remove copies: a DELETE with copiesOnly=true', async () => {
    await link.removeCopies();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/link?copiesOnly=true`);
    expect(init.method).toBe('DELETE');
  });

  it('forgets the link with Disconnect: a plain DELETE, with no query', async () => {
    await link.unlink();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/link`);
    expect(init.method).toBe('DELETE');
  });

  it('reads the status with a GET and no query', async () => {
    await link.status();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit | undefined];
    expect(url).toBe(`${API_BASE}/api/link`);
    expect(init?.method ?? 'GET').toBe('GET');
  });
});

describe('api · openExternal', () => {
  afterEach(() => {
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  it('rejects non-http(s) and invalid URLs before calling Tauri or window.open', () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const invoke = vi.fn().mockResolvedValue(undefined);
    (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = { invoke };

    for (const bad of ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,hi', 'not a url', '']) {
      openExternal(bad);
    }
    expect(invoke).not.toHaveBeenCalled();
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });

  it('does not fall back to window.open when Tauri open_url rejects', async () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const invoke = vi.fn().mockRejectedValue(new Error('rejected by host'));
    (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = { invoke };

    openExternal('https://provider.example/oauth?a=1&b=2');
    await new Promise((r) => setTimeout(r, 0));
    expect(invoke).toHaveBeenCalledWith('open_url', { url: 'https://provider.example/oauth?a=1&b=2' });
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });

  it('opens valid http(s) URLs with window.open when outside Tauri', () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    openExternal('https://oaiy.com/docs?a=1&b=2');
    expect(openSpy).toHaveBeenCalledWith('https://oaiy.com/docs?a=1&b=2', '_blank', 'noopener,noreferrer');
    openSpy.mockRestore();
  });
});
