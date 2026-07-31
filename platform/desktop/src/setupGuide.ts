import type {
  AiProviderPublic,
  PairedApp,
  PluginRecord,
  RuntimeStatus,
  ServiceSnapshot,
} from './api';

/**
 * The first-run checklist.
 *
 * Deliberately DERIVED from the state the app already polls rather than stored:
 * a checklist with its own persisted "done" flags drifts the moment someone
 * removes a provider or a service dies, and then it is lying. Every row here is
 * recomputed from live data, so it can only ever say what is actually true.
 *
 * Order is the order a user should work in — the runtime has to be able to
 * execute anything before an AI provider is worth configuring.
 */

export type StepId = 'runtime' | 'ai' | 'plugins' | 'connect';

/** Where a step's "Take me there" hands off to. */
export type StepTarget = 'overview' | 'services' | 'plugins' | 'providers' | 'connections';

export interface SetupStep {
  id: StepId;
  title: string;
  /** What this gets the user — not a restatement of the title. */
  detail: string;
  done: boolean;
  /** A step that isn't required for flows to run, just useful. */
  optional?: boolean;
  target: StepTarget;
  cta: string;
  /** Shown when not done and we know something specific about why. */
  blocker?: string;
}

export interface SetupInput {
  runtime: RuntimeStatus | null;
  providers: AiProviderPublic[] | null;
  services: ServiceSnapshot[] | null;
  plugins: PluginRecord[] | null;
  connected: PairedApp[] | null;
}

/** A provider is usable when it's on and either has a key or is a local endpoint. */
function providerReady(p: AiProviderPublic): boolean {
  return p.enabled && (p.hasKey || p.allowLocal);
}

/** A running local model service counts as an AI source too. */
function hasLocalModel(services: ServiceSnapshot[] | null): boolean {
  return (services ?? []).some(
    (s) => s.status === 'running' && (s.category ?? '').toLowerCase().includes('llm'),
  );
}

export function deriveSetupSteps(input: SetupInput): SetupStep[] {
  const { runtime, providers, services, plugins, connected } = input;

  const runtimeReady = runtime?.ready === true;
  const aiReady = (providers ?? []).some(providerReady) || hasLocalModel(services);
  const pluginRunning = (plugins ?? []).some((p) => p.state === 'running');
  const anyConnected = (connected ?? []).length > 0;

  return [
    {
      id: 'runtime',
      title: 'Get the flow runtime ready',
      detail: 'OAIY runs flows through a bundled engine. Without it nothing can execute.',
      done: runtimeReady,
      target: 'overview',
      cta: 'Fix the runtime',
      // The endpoint already works out WHICH part is missing; pass it through
      // rather than guessing a second time here.
      blocker: runtimeReady ? undefined : (runtime?.flowRuntime.detail ?? undefined),
    },
    {
      id: 'ai',
      title: 'Give your flows a model',
      detail:
        'Sign in with ChatGPT, add a provider key, or run a local model — whichever you prefer.',
      done: aiReady,
      target: 'providers',
      cta: 'Set up AI',
      blocker:
        aiReady || providers === null
          ? undefined
          : providers.length === 0
            ? 'No provider is configured yet.'
            : 'Every provider is disabled or missing a key.',
    },
    {
      id: 'plugins',
      title: 'Add a plugin',
      detail:
        'Plugins are where connectors and device events come from — a phone bridge, for example.',
      done: pluginRunning,
      optional: true,
      target: 'plugins',
      cta: 'Open Plugins',
      blocker:
        pluginRunning || plugins === null
          ? undefined
          : plugins.length === 0
            ? 'Nothing installed yet.'
            : 'Installed, but not running.',
    },
    {
      id: 'connect',
      title: 'Connect an app',
      detail: 'Let a site like FormLogic run flows on this machine. It asks; you approve.',
      done: anyConnected,
      optional: true,
      target: 'connections',
      cta: 'Open Connections',
    },
  ];
}

/** Everything that actually blocks flows is done. Optional steps don't count. */
export function setupComplete(steps: SetupStep[]): boolean {
  return steps.every((s) => s.done || s.optional);
}

export function setupProgress(steps: SetupStep[]): { done: number; total: number } {
  const required = steps.filter((s) => !s.optional);
  return { done: required.filter((s) => s.done).length, total: required.length };
}

const DISMISS_KEY = 'oaiy.setupGuide.dismissed';

/** Whether to auto-open the guide (first run, and not yet dismissed). */
export function shouldAutoOpen(): boolean {
  try {
    return localStorage.getItem(DISMISS_KEY) !== '1';
  } catch {
    // Storage can be unavailable in a webview; never let that block the app.
    return false;
  }
}

export function dismissGuide(): void {
  try {
    localStorage.setItem(DISMISS_KEY, '1');
  } catch {
    /* ignore */
  }
}

export function reopenGuide(): void {
  try {
    localStorage.removeItem(DISMISS_KEY);
  } catch {
    /* ignore */
  }
}
