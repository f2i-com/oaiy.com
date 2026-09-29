// The typed client for the routes two other branches build (CONTROL_API.md §1
// and §2), against fixtures in the pinned shapes: /api/control/settings,
// /api/control/log, /api/agent/preferences, /api/engines/recommendation, and
// Codex's model catalogue. A 404 (an older desktop, or before those branches
// merge) is `null`, never an error, so every view can degrade.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  agentIntent,
  agentPreferences,
  API_BASE,
  codexModels,
  control,
  engineRecommendation,
  logEntries,
  optional,
  type ControlLogEntry,
  type EngineRecommendation,
} from './api';

type Call = { url: string; method: string; body: unknown };
let calls: Call[];
let routes: Record<string, { status: number; body?: unknown }>;

function json(status: number, body?: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

beforeEach(() => {
  calls = [];
  routes = {};
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const route = routes[`${method} ${url.replace(API_BASE, '')}`];
      if (!route) return json(404, { error: 'no such route' });
      return json(route.status, route.body);
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// Fixtures in the pinned shapes.
const SETTINGS = { agentMayChange: true };
const LOG: ControlLogEntry[] = [
  { at: '2026-09-29T10:20:00Z', tool: 'plugin_setup_step_done', args: { pluginId: 'aokie', stepId: 'pair' }, session: 'setup', ok: true, summary: 'Marked Aokie’s pairing done' },
  { at: '2026-09-29T10:15:00Z', tool: 'plugin_install', args: { source: 'C:/plugins/aokie', token: '[redacted]' }, session: 'setup', ok: true, summary: null },
];
const PREFS_CHATGPT = { model: { source: 'chatgpt', model: 'gpt-5.5' } };
const RECOMMENDATION: EngineRecommendation = {
  recommend: 'chatgpt',
  local: { ok: false, reason: 'The largest GPU has 6 GB; the recommended model needs 8 GB.', gpus: [{ name: 'Laptop GPU', totalGb: 6, freeGb: 3.2 }], chosen: null, suggested: { id: 'some-llm', name: 'Some LLM', vramGb: 8 } },
  chatgpt: { signedIn: false },
};

describe('the control API client', () => {
  it('reads and writes the switch', async () => {
    routes['GET /api/control/settings'] = { status: 200, body: SETTINGS };
    routes['PUT /api/control/settings'] = { status: 200, body: { agentMayChange: false } };
    expect(await control.settings()).toEqual(SETTINGS);
    expect(await control.setSettings({ agentMayChange: false })).toEqual({ agentMayChange: false });
    expect(calls.at(-1)).toEqual({ url: `${API_BASE}/api/control/settings`, method: 'PUT', body: { agentMayChange: false } });
  });

  it('a PUT answered with no body keeps what was sent', async () => {
    routes['PUT /api/control/settings'] = { status: 204 };
    expect(await control.setSettings({ agentMayChange: true })).toEqual({ agentMayChange: true });
  });

  it('an older desktop (404): no switch, no log', async () => {
    expect(await control.settings()).toBeNull();
    expect(await control.log()).toBeNull();
  });

  it('any other failure still throws', async () => {
    routes['GET /api/control/settings'] = { status: 500, body: { error: 'control.json could not be read' } };
    await expect(control.settings()).rejects.toThrow('500: control.json could not be read');
  });

  it('reads the log newest first, asking for 100 by default', async () => {
    routes['GET /api/control/log?limit=100'] = { status: 200, body: LOG };
    expect(await control.log()).toEqual(LOG);
    routes['GET /api/control/log?limit=5'] = { status: 200, body: { entries: LOG } };
    expect(await control.log(5)).toEqual(LOG);
  });

  it('takes the log as a list or wrapped, and drops rows it cannot read', () => {
    expect(logEntries(LOG)).toEqual(LOG);
    expect(logEntries({ log: LOG })).toEqual(LOG);
    expect(logEntries({ items: [LOG[0], null, { at: 1 }, 'x'] })).toEqual([LOG[0]]);
    expect(logEntries(null)).toEqual([]);
    expect(logEntries({ nothing: true })).toEqual([]);
  });
});

describe('the Agent’s model', () => {
  it('reads and writes the preference', async () => {
    routes['GET /api/agent/preferences'] = { status: 200, body: PREFS_CHATGPT };
    routes['PUT /api/agent/preferences'] = { status: 200, body: { model: { source: 'engine' } } };
    expect(await agentPreferences.get()).toEqual(PREFS_CHATGPT);
    expect(await agentPreferences.set({ model: { source: 'engine' } })).toEqual({ model: { source: 'engine' } });
    expect(calls.at(-1)?.body).toEqual({ model: { source: 'engine' } });
  });

  it('a desktop that keeps none (404): null', async () => {
    expect(await agentPreferences.get()).toBeNull();
  });

  it('reads the hardware recommendation, or null from a desktop without it', async () => {
    expect(await engineRecommendation()).toBeNull();
    routes['GET /api/engines/recommendation'] = { status: 200, body: RECOMMENDATION };
    expect(await engineRecommendation()).toEqual(RECOMMENDATION);
  });

  it('lists Codex’s catalogue from the ChatGPT connector’s models route, skipping nameless rows', async () => {
    routes['GET /api/ai/providers/openai-codex-agent/v1/models'] = {
      status: 200,
      body: { object: 'list', data: [{ id: 'gpt-5.5', object: 'model', displayName: 'GPT-5.5', isDefault: true }, { id: '' }, { id: 'gpt-5.6-luna', object: 'model' }] },
    };
    expect((await codexModels()).map((m) => m.id)).toEqual(['gpt-5.5', 'gpt-5.6-luna']);
  });

  it('optional() passes answers through and turns only a 404 into null', async () => {
    expect(await optional(Promise.resolve(3))).toBe(3);
    expect(await optional(Promise.reject(new Error('404: gone')))).toBeNull();
    await expect(optional(Promise.reject(new Error('502: upstream')))).rejects.toThrow('502');
  });
});

describe('the setupWithAgent intent', () => {
  afterEach(() => {
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  it('is sent by name through the agent_intent command', async () => {
    const invoke = vi.fn().mockResolvedValue(undefined);
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = { invoke };
    await agentIntent('setupWithAgent');
    expect(invoke).toHaveBeenCalledWith('agent_intent', { intent: 'setupWithAgent' });
  });

  it('is refused plainly outside OAIY’s window', async () => {
    await expect(agentIntent('setupWithAgent')).rejects.toThrow('Not running in the OAIY desktop app.');
  });
});
