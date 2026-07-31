import { describe, expect, it } from 'vitest';
import { deriveSetupSteps, setupComplete, setupProgress, type SetupInput } from './setupGuide';
import type { AiProviderPublic, PluginRecord, RuntimeStatus, ServiceSnapshot } from './api';

// The checklist is derived from live state rather than stored, so these tests
// pin the rule that a row can only claim "done" when it is actually true.

const runtimeReady = { ready: true, flowRuntime: { cliResolved: true, cliKind: 'node' } } as RuntimeStatus;
const runtimeBroken = {
  ready: false,
  flowRuntime: { cliResolved: false, cliKind: 'missing', detail: 'Install the `oaiy` CLI…' },
} as RuntimeStatus;

function provider(over: Partial<AiProviderPublic> = {}): AiProviderPublic {
  return {
    id: 'p', name: 'P', protocol: 'openai', baseUrl: 'https://x', capabilities: [],
    enabled: true, allowLocal: false, hasKey: true, category: null, model: null, ...over,
  } as AiProviderPublic;
}
const svc = (over: Partial<ServiceSnapshot> = {}) =>
  ({ id: 's', name: 'S', category: 'LLM', status: 'running', port: 1, defaultPort: 1, ...over }) as ServiceSnapshot;
const plugin = (over: Partial<PluginRecord> = {}) => ({ id: 'a', state: 'running', ...over }) as PluginRecord;

const base: SetupInput = { runtime: runtimeReady, providers: [], services: [], plugins: [], connected: [] };
const step = (i: SetupInput, id: string) => deriveSetupSteps(i).find((s) => s.id === id)!;

describe('runtime step', () => {
  it('is done when the runtime reports ready', () => {
    expect(step(base, 'runtime').done).toBe(true);
  });

  it('carries the endpoint\'s own reason when it is not', () => {
    const s = step({ ...base, runtime: runtimeBroken }, 'runtime');
    expect(s.done).toBe(false);
    // Not re-derived here — the backend already worked out which part is missing.
    expect(s.blocker).toContain('oaiy` CLI');
  });

  it('is never optional — nothing runs without it', () => {
    expect(step(base, 'runtime').optional).toBeUndefined();
  });
});

describe('AI step', () => {
  it('is done with a keyed, enabled provider', () => {
    expect(step({ ...base, providers: [provider()] }, 'ai').done).toBe(true);
  });

  it('is NOT done when the only provider is disabled or keyless', () => {
    expect(step({ ...base, providers: [provider({ enabled: false })] }, 'ai').done).toBe(false);
    expect(step({ ...base, providers: [provider({ hasKey: false })] }, 'ai').done).toBe(false);
  });

  it('accepts a local endpoint with no key', () => {
    expect(step({ ...base, providers: [provider({ hasKey: false, allowLocal: true })] }, 'ai').done).toBe(true);
  });

  it('also accepts a RUNNING local model service instead of a provider', () => {
    expect(step({ ...base, services: [svc()] }, 'ai').done).toBe(true);
    // A stopped one does not count — it cannot answer a flow.
    expect(step({ ...base, services: [svc({ status: 'stopped' })] }, 'ai').done).toBe(false);
    // Nor does a non-LLM service.
    expect(step({ ...base, services: [svc({ category: 'Browser' })] }, 'ai').done).toBe(false);
  });

  it('distinguishes "none configured" from "all unusable"', () => {
    expect(step(base, 'ai').blocker).toContain('No provider');
    expect(step({ ...base, providers: [provider({ enabled: false })] }, 'ai').blocker).toContain('disabled');
  });
});

describe('plugin + connect steps', () => {
  it('needs a plugin that is actually running, not merely installed', () => {
    expect(step({ ...base, plugins: [plugin({ state: 'installed' })] }, 'plugins').done).toBe(false);
    expect(step({ ...base, plugins: [plugin({ state: 'installed' })] }, 'plugins').blocker).toContain('not running');
    expect(step({ ...base, plugins: [plugin()] }, 'plugins').done).toBe(true);
  });

  it('marks plugins and connect optional — flows run without them', () => {
    expect(step(base, 'plugins').optional).toBe(true);
    expect(step(base, 'connect').optional).toBe(true);
  });
});

describe('completion', () => {
  it('is complete once the REQUIRED steps pass, even with optional ones outstanding', () => {
    const steps = deriveSetupSteps({ ...base, providers: [provider()] });
    expect(setupComplete(steps)).toBe(true);
    expect(steps.filter((s) => s.optional && !s.done).length).toBeGreaterThan(0);
  });

  it('is not complete while the runtime is broken, however much else is set up', () => {
    const steps = deriveSetupSteps({
      runtime: runtimeBroken,
      providers: [provider()],
      services: [svc()],
      plugins: [plugin()],
      connected: [{ id: 't', product: 'formlogic' } as never],
    });
    expect(setupComplete(steps)).toBe(false);
  });

  it('counts only required steps in progress', () => {
    expect(setupProgress(deriveSetupSteps(base))).toEqual({ done: 1, total: 2 });
    expect(setupProgress(deriveSetupSteps({ ...base, providers: [provider()] }))).toEqual({ done: 2, total: 2 });
  });

  it('treats unknown (still loading) state as not-done rather than guessing', () => {
    const steps = deriveSetupSteps({ runtime: null, providers: null, services: null, plugins: null, connected: null });
    expect(steps.every((s) => !s.done)).toBe(true);
    // …and offers no misleading blocker text while data is still in flight.
    expect(steps.find((s) => s.id === 'ai')!.blocker).toBeUndefined();
  });
});
