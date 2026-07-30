/**
 * useDynamicOptions — resolve a `PropertyDefinition.dynamicOptionsSource`
 * URI to a runtime list of `PropertyOption`s.
 *
 * Why a hook in `oaiy-ui-components` rather than `oaiy-core`: UI fields
 * live here, the Tauri broker that `oaiy-core/runtime.ts` exposes is for
 * compiled workflow code only (see ALLOWED_TAURI_COMMANDS) — UI
 * components reach Tauri via `window.__TAURI_INTERNALS__.invoke`
 * directly, the same way `PropertyField`'s file picker does.
 *
 * Source scheme is `<provider>:<verb>:<arg>...`. Initial providers:
 *
 *   diffusion:list-models-by-task:<task>
 *     → invokes `plugin:oaiy-diffusion|diffusion_list_models`, filters
 *       by `m.task === <task>` (use `*` for "all"), maps to options.
 *       Used by the diffusion_image / diffusion_video node modelId
 *       dropdowns so externally-registered checkpoints appear without
 *       a node-JSON edit.
 *
 * Cache: module-level, by source string. One in-flight promise per
 * source, deduped across concurrent mounts. `refetch()` evicts and
 * re-resolves; call it from a re-scan / file-picker handler.
 */

import { useCallback, useEffect, useState } from 'react';
import type { PropertyOption } from 'oaiy-core';

interface UseDynamicOptionsResult {
  options: PropertyOption[];
  loading: boolean;
  error: string | undefined;
  refetch: () => void;
}

const optionsCache = new Map<string, PropertyOption[]>();
const inflight = new Map<string, Promise<PropertyOption[]>>();

// Subscribers fire when external code (a Tauri event listener, a
// settings panel, etc.) calls `invalidateDynamicOptions()`. Each
// useDynamicOptions instance registers a re-fetch hook so a single
// invalidation refreshes every mounted dropdown in one pass.
const invalidationSubscribers = new Set<() => void>();

// Host-app-registered resolvers for additional `dynamicOptionsSource`
// schemes. The built-in `diffusion:list-models-by-task:` is hardcoded
// in resolveSource (tied to plugin-oaiy-diffusion); anything else lives
// here, registered from the consuming app's bootstrap.
//
// Used by oaiy-web for `service:list` (the Service-node preset dropdown,
// reading the localStorage service registry). Could equally well wrap a
// REST call or a static map — the resolver just returns options.
type DynamicOptionsResolver = (
  rest: string,
) => PropertyOption[] | Promise<PropertyOption[]>;
const customResolvers = new Map<string, DynamicOptionsResolver>();

// Tauri event subscription is initialised lazily on the first hook
// mount that finds a Tauri runtime. Plugins emit
// `diffusion://models-changed` after add/rescan/register; we
// invalidate the cache + retrigger every active hook so dropdowns
// reflect the new registry without a manual reload.
let tauriListenerInitialised = false;

interface DiffusionModelDef {
  id: string;
  name: string;
  family: string;
  task: string;
  precision: string;
}

function getInvoke(): ((cmd: string, args?: unknown) => Promise<unknown>) | undefined {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  return w?.__TAURI__?.core?.invoke ?? w?.__TAURI_INTERNALS__?.invoke;
}

// Type-safe view of `window.__TAURI_INTERNALS__.transformCallback`,
// the same back-door the OaiyLocalModelPicker uses to listen for
// `llm://download-progress` events without going through
// `@tauri-apps/api/event` (which is blocked for plugin code by
// pluginCompiler.ts). Mirroring the technique keeps the hook usable
// from inside compiled plugin bundles.
interface TauriInternals {
  invoke: (cmd: string, args?: unknown) => Promise<unknown>;
  transformCallback?: (cb: (response: unknown) => void, once?: boolean) => number;
}

function getInternals(): TauriInternals | null {
  if (typeof window === 'undefined') return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const i = (window as any).__TAURI_INTERNALS__;
  return i && typeof i.invoke === 'function' ? (i as TauriInternals) : null;
}

function setupTauriInvalidationOnce(): void {
  if (tauriListenerInitialised) return;
  const internals = getInternals();
  if (!internals?.transformCallback) return;
  tauriListenerInitialised = true;
  const handlerId = internals.transformCallback((event) => {
    const payload = (event as { payload?: unknown } | null)?.payload;
    // Don't try to react beyond invalidation — the actual filter logic
    // (which dropdowns to refresh) lives in resolveSource. Wholesale
    // invalidate + retrigger is correct since the typical workflow has
    // 1-2 dropdowns mounted at any one time.
    void payload;
    optionsCache.clear();
    invalidationSubscribers.forEach((fn) => fn());
  }, false);
  // Fire-and-forget; we don't unsubscribe because this is process-wide.
  internals.invoke('plugin:event|listen', {
    event: 'diffusion://models-changed',
    target: { kind: 'Any' },
    handler: handlerId,
  }).catch(() => {
    // Tauri 1 or non-Tauri runtime — silently disable real-time refresh.
    tauriListenerInitialised = false;
  });
}

async function resolveSource(source: string): Promise<PropertyOption[]> {
  // Host-registered resolvers first (longest prefix wins so more-specific
  // schemes shadow generic ones). The prefix may be the full source
  // (exact-match, e.g. `service:list`) or a prefix-with-rest like
  // `service:list:custom` — the resolver receives the trailing remainder
  // (or '' for exact match).
  if (customResolvers.size > 0) {
    const prefixes = [...customResolvers.keys()].sort((a, b) => b.length - a.length);
    for (const p of prefixes) {
      if (source === p) {
        return Array.from(await Promise.resolve(customResolvers.get(p)!('')));
      }
      if (source.startsWith(p + ':') || (p.endsWith(':') && source.startsWith(p))) {
        const rest = source.slice(p.length).replace(/^:/, '');
        return Array.from(await Promise.resolve(customResolvers.get(p)!(rest)));
      }
    }
  }

  const invoke = getInvoke();
  if (!invoke) return [];

  const prefix = 'diffusion:list-models-by-task:';
  if (source.startsWith(prefix)) {
    const raw = source.slice(prefix.length);
    // Comma-separated whitelist; `*` short-circuits to "all tasks".
    const tasks = raw === '*' ? null : new Set(raw.split(',').map((s) => s.trim()).filter(Boolean));
    const list = (await invoke('plugin:oaiy-diffusion|diffusion_list_models')) as DiffusionModelDef[];
    if (!Array.isArray(list)) return [];
    return list
      .filter((m) => tasks === null || tasks.has(m.task))
      .map((m) => ({
        value: m.id,
        label: m.name,
        description: `${m.family} · ${m.task} · ${m.precision}`,
      }));
  }

  // Unknown scheme — return empty so the dropdown silently falls back
  // to the static options. The PropertyField caller decides whether to
  // warn; keeping resolveSource itself silent avoids spamming the
  // console on every render.
  return [];
}

export function useDynamicOptions(source: string | undefined): UseDynamicOptionsResult {
  const [tick, setTick] = useState(0);
  const [options, setOptions] = useState<PropertyOption[]>(() =>
    source ? optionsCache.get(source) ?? [] : [],
  );
  const [loading, setLoading] = useState<boolean>(() =>
    !!source && !optionsCache.has(source),
  );
  const [error, setError] = useState<string | undefined>(undefined);

  const refetch = useCallback(() => {
    if (source) optionsCache.delete(source);
    setTick((t) => t + 1);
  }, [source]);

  // Subscribe to global invalidation events so the Tauri listener can
  // refresh every mounted dropdown when the model registry changes.
  useEffect(() => {
    if (!source) return;
    const onInvalidate = () => setTick((t) => t + 1);
    invalidationSubscribers.add(onInvalidate);
    setupTauriInvalidationOnce();
    return () => {
      invalidationSubscribers.delete(onInvalidate);
    };
  }, [source]);

  useEffect(() => {
    if (!source) {
      setOptions([]);
      setLoading(false);
      setError(undefined);
      return;
    }
    let cancelled = false;
    const cached = optionsCache.get(source);
    if (cached) {
      setOptions(cached);
      setLoading(false);
      setError(undefined);
      return;
    }
    setLoading(true);
    setError(undefined);
    let p = inflight.get(source);
    if (!p) {
      p = resolveSource(source).finally(() => inflight.delete(source));
      inflight.set(source, p);
    }
    p.then((opts) => {
      optionsCache.set(source, opts);
      if (!cancelled) {
        setOptions(opts);
        setLoading(false);
      }
    }).catch((e) => {
      if (!cancelled) {
        setOptions([]);
        setLoading(false);
        setError(String((e as { message?: string })?.message ?? e));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [source, tick]);

  return { options, loading, error, refetch };
}

/**
 * Register a resolver for a `dynamicOptionsSource` scheme prefix.
 *
 * The hook ships with one hardcoded scheme (`diffusion:list-models-by-task:`
 * — tied to plugin-oaiy-diffusion). Host apps add their own schemes via
 * this API: oaiy-web registers `service:list` to surface its localStorage
 * service registry on the Service Call node's preset dropdown.
 *
 * Lookup is longest-prefix-wins; the resolver receives the part of the
 * source AFTER its prefix (or '' for an exact match). Re-registering a
 * prefix replaces the previous handler and invalidates the options
 * cache so already-mounted dropdowns refresh immediately.
 *
 * @example
 *   registerDynamicOptionsResolver('service:list', () =>
 *     listServices().map((s) => ({ value: s.id, label: s.name }))
 *   );
 */
export function registerDynamicOptionsResolver(
  prefix: string,
  resolver: DynamicOptionsResolver,
): void {
  customResolvers.set(prefix, resolver);
  invalidateDynamicOptions();
}

/**
 * Subscribe to dynamic-options invalidation events.
 *
 * Returns an unsubscribe function. Used by places that derive UI from
 * the same registries the options resolvers read (e.g. the node
 * palette's service-injection re-runs when the user adds a service in
 * Settings). The Services registry mutator already calls
 * `invalidateDynamicOptions()` on every save/delete, so a subscriber
 * here will refresh whenever the underlying state changes.
 */
export function subscribeToDynamicOptionsInvalidation(fn: () => void): () => void {
  invalidationSubscribers.add(fn);
  return () => { invalidationSubscribers.delete(fn); };
}

/**
 * Evict every cached source. Useful from places that just changed the
 * underlying registry (added an external dir, picked a custom file)
 * and want every mounted dropdown to re-resolve on next render. Also
 * notifies every active hook so the refetch happens immediately
 * rather than on next mount.
 */
export function invalidateDynamicOptions(): void {
  optionsCache.clear();
  invalidationSubscribers.forEach((fn) => fn());
}
