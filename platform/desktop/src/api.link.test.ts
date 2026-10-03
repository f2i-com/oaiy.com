// The requests the Connections panel's two ways of forgetting make, as the host sees them.
//
// `DELETE /api/link` forgets the link. `DELETE /api/link?copiesOnly=true` only takes the copies of an earlier key that a forget could
// not remove, and leaves a link that is there: it is what "Remove copies" sends. A button that sent the plain request would be a
// disconnect that is not asked for, and one that sent `copiesOnly=false` the same, so the panel's tests (which mock the client) are
// not enough: these read the request itself.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, link } from './api';

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
