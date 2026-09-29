// The client call behind "Trust this plugin". The route it reaches reads only the plugin's
// id, so what this call sends is all a caller can ask to have trusted: a POST to the
// plugin's own path, the id encoded into it, and no body. (PluginsPanel.trust.test.tsx
// replaces the call with a mock; this is the test of the call itself.)
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, plugins, type PluginRecord } from './api';

type Call = { url: string; method: string; hasBody: boolean };
let calls: Call[];
let reply: { status: number; body: unknown };

const TRUSTED: PluginRecord = {
  id: 'aokie',
  state: 'installed',
  dir: 'C:/plugins/aokie',
  userDisabled: false,
  restartAttempts: 0,
  trust: { state: 'trusted-local', trustedAt: '2026-09-29T10:00:00Z', reason: 'You trusted this exact package.' },
};

beforeEach(() => {
  calls = [];
  reply = { status: 200, body: TRUSTED };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method ?? 'GET', hasBody: init?.body !== undefined && init?.body !== null });
      return new Response(JSON.stringify(reply.body), { status: reply.status, headers: { 'Content-Type': 'application/json' } });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('plugins.trust', () => {
  it('posts to the plugin’s own trust route and sends no body', async () => {
    await plugins.trust('aokie');
    expect(calls).toEqual([{ url: `${API_BASE}/api/plugins/aokie/trust`, method: 'POST', hasBody: false }]);
  });

  it('puts the id in the path, encoded, so it can never name another route', async () => {
    await plugins.trust('a/b c');
    expect(calls[0].url).toBe(`${API_BASE}/api/plugins/a%2Fb%20c/trust`);
    await plugins.trust('../install');
    expect(calls[1].url).toBe(`${API_BASE}/api/plugins/..%2Finstall/trust`);
    expect(calls.every((c) => c.method === 'POST' && !c.hasBody)).toBe(true);
  });

  it('hands back the record the route answers with', async () => {
    const record = await plugins.trust('aokie');
    expect(record.trust?.state).toBe('trusted-local');
    expect(record.id).toBe('aokie');
  });

  it('turns a refusal into an error that carries the desktop’s own words', async () => {
    reply = {
      status: 409,
      body: { error: { code: 'invalid_request', message: 'aokie carries a signature (package-manifest.json). A signed package is judged by its signature alone, so it cannot be trusted by hand.' } },
    };
    await expect(plugins.trust('aokie')).rejects.toThrow(/409: aokie carries a signature.*cannot be trusted by hand/);
  });
});
