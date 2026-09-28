/**
 * OAIY Desktop service → flow palette bridge.
 *
 * When OAIY Desktop is running it offers two kinds of services, and this
 * module polls both while the desktop is available:
 *
 *   - `GET /api/services`: the local services it manages (Ollama, llama.cpp,
 *     Python rigs …), ids `companion:<id>`. Only INSTALLED ones are listed —
 *     a stopped one is still pickable (the runtime asks the desktop to start
 *     it), one that is not installed cannot run and is left out.
 *   - `GET /api/ai/engine/services` (in OAIY's window only): OAIY's engine's
 *     models whose files are here, per kind, ids `engine:<kind>:<model>`, and
 *     OAIY Voice's transcription (`voice:transcribe`), each with its call
 *     contract (see bundled-modules/core-service/contract.ts). Asking starts
 *     nothing on the desktop. Listed only in OAIY's window, whose origin (and
 *     token) may run them; a plain browser tab could list but not call them.
 *
 * Both are mapped to the `CustomService` shape and published together into a
 * dedicated localStorage key (`oaiy.desktopServices`) — kept SEPARATE from the
 * user's own `oaiy.customServices` so we never touch their saved data.
 *
 * Readers of the list:
 *   - the `service:list` dynamic-options resolver (main.tsx) → every node's
 *     Service dropdown, grouped "OAIY engine" / "Your services" / "Custom";
 *   - the palette (hooks/useModuleNodes.tsx + lib/nodeAvailability.ts): in
 *     OAIY's window a media node is shown only when something installed can
 *     run it, and each service is a draggable entry;
 *   - the compilers (core-service/contract.ts) → picking one and running the
 *     flow resolves the right endpoint / model / contract at run time.
 *
 * Lifecycle: the poll starts/stops in lockstep with desktop availability.
 * When the desktop goes away the published list is cleared.
 * `desktopServicesLoaded()` says whether the list reflects a poll that has
 * answered (so the palette does not hide nodes before it has been asked).
 */
import { invalidateDynamicOptions } from 'oaiy-ui-components';
import type { CustomService, ServiceNodeTag } from 'oaiy-core/modules/core-service/examples';
import {
  DESKTOP_API_BASE,
  refreshDesktopStatus,
  subscribeDesktopStatus,
} from './desktopDetection';

/** Shared contract with the compilers (core-service/contract.ts reads this key). */
export const DESKTOP_SERVICE_STORAGE_KEY = 'oaiy.desktopServices';

const POLL_INTERVAL_MS = 10_000;
const FETCH_TIMEOUT_MS = 1500;
/** The engine list asks the engines' gateway through the desktop: a little slower. */
const ENGINE_TIMEOUT_MS = 6000;

/** A template-declared call contract (companion ServiceSnapshot.node). */
interface DesktopNodeSpec {
  endpoint?: string | null;
  method?: string | null;
  bodyTemplate?: string | null;
  responsePath?: string | null;
  apiFormat?: string | null;
  icon?: string | null;
}

/** Subset of OAIY Desktop's ServiceSnapshot we actually use. */
export interface DesktopServiceSnapshot {
  id: string;
  name: string;
  description: string;
  category: string;
  status: string;
  port: number;
  defaultPort: number;
  docsUrl: string | null;
  /** Whether its program is on disk. Not installed → not listed. */
  installed?: boolean;
  /** How to call this service as a node, declared by its template (optional). */
  node?: DesktopNodeSpec | null;
}

/** One entry of `GET /api/ai/engine/services` (camelCase, like CustomService). */
export interface EngineServiceEntry {
  id: string;
  name: string;
  kind?: string;
  model?: string;
  default?: boolean;
  description?: string;
  icon?: string;
  nodeTypes?: string[];
  endpoint: string;
  method?: string;
  bodyTemplate?: string;
  responseType?: string;
  responsePath?: string;
  apiFormat?: string;
  output?: string;
  outputFormat?: string;
  requestFormat?: string;
  inputs?: CustomService['inputs'];
  job?: {
    idPath: string;
    statusUrl: string;
    statusPath: string;
    progressPath?: string;
    done: string[];
    failed: string[];
    errorPath?: string;
    contentUrl?: string;
    contentFallbackUrl?: string | null;
    cancelUrl?: string | null;
  } | null;
}

/** The desktop, as this page reaches it: OAIY's window gives its origin and token. */
function desktopAccess(): { origin: string; token: string | null; inOaiy: boolean } {
  const given = (typeof window !== 'undefined'
    ? (window as unknown as { __OAIY_DESKTOP__?: { origin?: unknown; token?: unknown } }).__OAIY_DESKTOP__
    : undefined);
  // In OAIY's window only with its token too (as oaiyDesktop() decides).
  if (given && typeof given.origin === 'string' && given.origin && typeof given.token === 'string' && given.token) {
    return { origin: given.origin.replace(/\/+$/, ''), token: given.token, inOaiy: true };
  }
  return { origin: DESKTOP_API_BASE.replace(/\/+$/, ''), token: null, inOaiy: false };
}

/**
 * Map one desktop service to a `CustomService`. Running and stopped ones are
 * surfaced (a stopped one auto-starts when a flow using it runs); one that is
 * not installed is not (null). The status is folded into the label.
 *
 * LLM-category services are assumed OpenAI-compatible on
 * `/v1/chat/completions`. We deliberately leave `bodyTemplate` EMPTY so the
 * AI LLM node uses its standard OpenAI request path (which handles vision +
 * streaming) rather than the generic custom-template branch.
 */
export function mapToCustomService(s: DesktopServiceSnapshot): CustomService | null {
  if (s.installed === false) return null;
  const port = s.port || s.defaultPort;
  if (!port) return null;
  const cat = (s.category || '').toLowerCase();
  const base = `http://127.0.0.1:${port}`;
  const id = `companion:${s.id}`;
  const running = s.status === 'running';
  const statusTag = running ? 'running' : s.status;
  // Keep the node/palette description to a tidy one-liner — OAIY Desktop can
  // report a long paragraph (krea2's is), which overflows the node body.
  const rawDesc = (s.description || `Managed by the OAIY Desktop on port ${port}.`).trim();
  const firstSentence = rawDesc.split('. ')[0];
  const baseDesc =
    firstSentence.length > 90 ? `${firstSentence.slice(0, 87).trimEnd()}…` : firstSentence;
  const desc = running
    ? baseDesc
    : `${baseDesc} (currently ${s.status} — auto-starts when the flow runs)`;
  const common = { id, description: desc, headers: '{}', responseType: 'json' as const, isBuiltIn: false, group: 'desktop' as const };

  // 1. An explicit call contract declared by the template WINS — the service
  // tells us exactly how to call it, so even a custom/third-party service
  // surfaces ready-to-run with no per-service knowledge baked in here.
  const node = s.node;
  if (node) {
    if (node.apiFormat === 'openai') {
      return {
        ...common,
        name: `${s.name} (${statusTag})`,
        endpoint: `${base}${node.endpoint || '/v1/chat/completions'}`,
        method: (node.method || 'POST') as CustomService['method'],
        bodyTemplate: node.bodyTemplate ?? '',
        responsePath: node.responsePath || 'choices.0.message.content',
        icon: node.icon || '🤖',
        nodeTypes: ['ai_llm', 'service_call'],
        apiFormat: 'openai',
        model: '',
        installHint: `Managed by the OAIY Desktop (port ${port}). Start/stop it from OAIY Desktop's Services tab.`,
      };
    }
    return {
      ...common,
      name: `${s.name} (${statusTag})`,
      endpoint: node.endpoint ? `${base}${node.endpoint}` : base,
      method: (node.method || 'POST') as CustomService['method'],
      bodyTemplate: node.bodyTemplate ?? '{{inputRaw}}',
      responsePath: node.responsePath ?? '',
      icon: node.icon || '🧩',
      nodeTypes: ['service_call'],
      installHint: `Managed by the OAIY Desktop (port ${port}).`,
    };
  }

  // 2. No declared contract → fall back to a best-effort category convention.
  // Browser-automation services speak a bespoke session API — not surfaced.
  if (cat === 'browser') return null;

  if (cat === 'llm') {
    return {
      ...common,
      name: `${s.name} (${statusTag})`,
      endpoint: `${base}/v1/chat/completions`,
      method: 'POST',
      // Empty → AI node uses the standard OpenAI body, not custom-template.
      bodyTemplate: '',
      responsePath: 'choices.0.message.content',
      icon: '🤖',
      nodeTypes: ['ai_llm', 'service_call'],
      apiFormat: 'openai',
      model: '',
      installHint: `Managed by the OAIY Desktop (port ${port}). Start/stop it from OAIY Desktop's Services tab.`,
    };
  }

  // OAIY-managed image/video services expose POST /generate {prompt,...} and
  // return an image/video URL — wire that contract so a dragged node runs as-is
  // (otherwise it POSTs to the bare base URL and gets a 405 Method Not Allowed).
  const isVideo = cat === 'video generation';
  if (isVideo || cat === 'image generation') {
    return {
      ...common,
      name: `${s.name} (${statusTag})`,
      endpoint: `${base}/generate`,
      method: 'POST',
      bodyTemplate: '{"prompt": {{input}}}',
      responsePath: isVideo ? 'videoUrl' : 'imageUrl',
      icon: isVideo ? '🎬' : '🎨',
      nodeTypes: ['service_call'],
      installHint: `Managed by the OAIY Desktop (port ${port}).`,
    };
  }

  return {
    ...common,
    name: `${s.name} (${statusTag})`,
    endpoint: base,
    method: 'POST',
    bodyTemplate: '{{inputRaw}}',
    responsePath: '',
    icon: '🧩',
    nodeTypes: ['service_call'],
    installHint: `Managed by the OAIY Desktop (port ${port}).`,
  };
}

const KNOWN_OUTPUTS = new Set(['image', 'video', 'audio', 'model3d', 'text']);

/**
 * Map one engine entry to a `CustomService`: its paths made absolute on the
 * desktop (`origin`), grouped as OAIY's engine. Null for an entry without an
 * id or an endpoint.
 */
export function mapEngineService(e: EngineServiceEntry, origin: string): CustomService | null {
  if (!e || typeof e.id !== 'string' || !e.id || typeof e.endpoint !== 'string' || !e.endpoint) return null;
  const at = (u: string | null | undefined) => (u ? (u.startsWith('/') ? `${origin}${u}` : u) : u);
  const out: CustomService = {
    id: e.id,
    name: e.name || e.id,
    description: e.description || '',
    endpoint: at(e.endpoint)!,
    method: ((e.method || 'POST').toUpperCase()) as CustomService['method'],
    headers: '{}',
    bodyTemplate: e.bodyTemplate ?? '',
    responseType: (e.responseType === 'binary' || e.responseType === 'text' ? e.responseType : 'json'),
    responsePath: e.responsePath ?? '',
    isBuiltIn: false,
    icon: e.icon || '⚙️',
    nodeTypes: (e.nodeTypes ?? []) as ServiceNodeTag[],
    model: e.model || '',
    group: 'engine',
    kind: e.kind,
    default: e.default === true,
    installHint: "OAIY's engine: add or change its models in OAIY → Engines.",
  };
  if (e.apiFormat === 'openai') out.apiFormat = 'openai';
  if (e.output && KNOWN_OUTPUTS.has(e.output)) out.output = e.output as CustomService['output'];
  if (e.outputFormat) out.outputFormat = e.outputFormat;
  if (e.requestFormat === 'wav16k') out.requestFormat = 'wav16k';
  if (e.inputs?.length) out.inputs = e.inputs;
  if (e.job) {
    out.job = {
      ...e.job,
      statusUrl: at(e.job.statusUrl)!,
      contentUrl: at(e.job.contentUrl) ?? undefined,
      contentFallbackUrl: at(e.job.contentFallbackUrl) ?? null,
      cancelUrl: at(e.job.cancelUrl) ?? null,
    };
  }
  return out;
}

/**
 * The published list from both answers: the engine's entries first (defaults
 * first within a kind, as the desktop orders them), then installed desktop
 * services, running ones first.
 */
export function combineDesktopServices(
  services: DesktopServiceSnapshot[],
  engine: EngineServiceEntry[],
  origin: string,
): CustomService[] {
  const ranked = [...services].sort((a, b) => (a.status === 'running' ? 0 : 1) - (b.status === 'running' ? 0 : 1));
  return [
    ...engine.map((e) => mapEngineService(e, origin)).filter((x): x is CustomService => x !== null),
    ...ranked.map(mapToCustomService).filter((x): x is CustomService => x !== null),
  ];
}

// ---------------------------------------------------------------------------
// Publishing
// ---------------------------------------------------------------------------

let loaded = false;
const loadListeners = new Set<() => void>();

/** Whether the published list reflects an answered poll (the palette waits for one before hiding nodes). */
export function desktopServicesLoaded(): boolean {
  return loaded;
}

function setLoaded(next: boolean): void {
  if (loaded === next) return;
  loaded = next;
  for (const fn of loadListeners) fn();
  invalidateDynamicOptions();
}

/** Called when `desktopServicesLoaded()` changes. Returns an unsubscribe. */
export function subscribeDesktopServicesLoaded(fn: () => void): () => void {
  loadListeners.add(fn);
  return () => loadListeners.delete(fn);
}

function publish(list: CustomService[]): void {
  const next = JSON.stringify(list);
  const prev = localStorage.getItem(DESKTOP_SERVICE_STORAGE_KEY) ?? '[]';
  if (next === prev) return; // no change — skip the write + dropdown churn
  if (list.length === 0) {
    localStorage.removeItem(DESKTOP_SERVICE_STORAGE_KEY);
  } else {
    localStorage.setItem(DESKTOP_SERVICE_STORAGE_KEY, next);
  }
  // Refresh any mounted service dropdown (and the palette) so the change shows immediately.
  invalidateDynamicOptions();
}

/**
 * Synchronous snapshot of the current desktop-service list. Used by the
 * `service:list` resolver and the palette; the compilers read the same key.
 */
export function listDesktopServices(): CustomService[] {
  try {
    const raw = localStorage.getItem(DESKTOP_SERVICE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}

let pollTimer: number | null = null;
// Bumped on every startPoll/stopPoll so an in-flight pollOnce that resolves
// AFTER a teardown can't re-publish a stale companion-service list.
let pollGen = 0;

async function getJson<T>(url: string, timeoutMs: number, token: string | null): Promise<T | null> {
  const controller = new AbortController();
  const t = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
      credentials: 'omit',
      cache: 'no-store',
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (!resp.ok) return null;
    return (await resp.json().catch(() => null)) as T | null;
  } catch {
    return null;
  } finally {
    window.clearTimeout(t);
  }
}

async function pollOnce(): Promise<void> {
  const gen = pollGen;
  const desktop = desktopAccess();
  const [services, engine] = await Promise.all([
    getJson<{ services?: DesktopServiceSnapshot[] }>(`${desktop.origin}/api/services`, FETCH_TIMEOUT_MS, desktop.token),
    desktop.inOaiy
      ? getJson<{ services?: EngineServiceEntry[] }>(`${desktop.origin}/api/ai/engine/services`, ENGINE_TIMEOUT_MS, desktop.token)
      : Promise.resolve(null),
  ]);
  // Drop a result that arrives after a stop/start happened mid-flight, so a
  // late-resolving fetch can't restore a list the teardown already cleared.
  if (gen !== pollGen) return;
  publish(combineDesktopServices(services?.services ?? [], engine?.services ?? [], desktop.origin));
  setLoaded(services !== null || engine !== null);
}

function startPoll(): void {
  if (pollTimer !== null) return;
  pollGen++;
  void pollOnce();
  pollTimer = window.setInterval(pollOnce, POLL_INTERVAL_MS);
}

function stopPoll(): void {
  pollGen++; // invalidate any in-flight pollOnce so it can't re-publish
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
  publish([]); // authoritative clear, runs under the new generation
  setLoaded(false);
}

/**
 * Begin syncing desktop services (and OAIY's engine) into the palette. Polls
 * only while the desktop is available, and clears the list when it's not.
 * Safe to call once at app boot (after startDesktopDetection()).
 */
export function startDesktopServiceSync(): void {
  // The list kept from the last session stays until the first probe has
  // answered (a flow compiled meanwhile still finds its services); the
  // desktop missing then, or going away later, clears it.
  subscribeDesktopStatus((info) => {
    if (info.available) startPoll();
    else if (pollTimer !== null) stopPoll();
  });
  void refreshDesktopStatus()
    .then((info) => {
      if (!info.available) stopPoll();
    })
    .catch(() => stopPoll());
}

/** Ask again now (after adding a service or a model), rather than at the next tick. */
export function refreshDesktopServices(): Promise<void> {
  return pollOnce();
}
