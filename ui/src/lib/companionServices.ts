/**
 * Companion service → flow palette bridge (Phase 3).
 *
 * When the OAIY Companion is running it manages local AI services
 * (Ollama, llama.cpp, custom Python rigs) and exposes them at
 * `GET /api/services`. This module polls that endpoint while the
 * companion is available, maps each RUNNING service to the oaiy
 * `CustomService` shape, and publishes the result into a dedicated
 * localStorage key (`oaiy.companionServices`) — kept SEPARATE from the
 * user's own `oaiy.customServices` so we never touch their saved data.
 *
 * Three readers merge this list:
 *   - the `service:list` dynamic-options resolver (main.tsx) → companion
 *     services appear in every AI node's "Service Preset" dropdown.
 *   - core-ai/compiler.ts + core-service/compiler.ts → picking one and
 *     running the flow resolves the right endpoint / model at run time.
 *
 * Lifecycle: the poll starts/stops in lockstep with companion
 * availability (it subscribes to the detection probe). When the
 * companion goes away the published list is cleared, so a service you
 * stop in the companion disappears from the dropdowns within a tick.
 *
 * Why localStorage and not just an in-memory list? The compilers run in
 * the renderer and read service config synchronously from localStorage
 * (same way they read `oaiy.customServices`). Publishing to a key lets
 * them resolve companion services without importing app-level modules,
 * keeping the bundled modules self-contained.
 */
import { invalidateDynamicOptions } from 'oaiy-ui-components';
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import {
  COMPANION_API_BASE,
  subscribeCompanionStatus,
} from './companionDetection';

/** Shared contract with the two compilers (which read this key inline). */
export const COMPANION_SERVICE_STORAGE_KEY = 'oaiy.companionServices';

const POLL_INTERVAL_MS = 10_000;
const FETCH_TIMEOUT_MS = 1500;

/** A template-declared call contract (companion ServiceSnapshot.node). */
interface CompanionNodeSpec {
  endpoint?: string | null;
  method?: string | null;
  bodyTemplate?: string | null;
  responsePath?: string | null;
  apiFormat?: string | null;
  icon?: string | null;
}

/** Subset of the companion's ServiceSnapshot we actually use. */
interface CompanionServiceSnapshot {
  id: string;
  name: string;
  description: string;
  category: string;
  status: string;
  port: number;
  defaultPort: number;
  docsUrl: string | null;
  /** How to call this service as a node, declared by its template (optional). */
  node?: CompanionNodeSpec | null;
}

/**
 * Map one companion service to a `CustomService`. Both running AND
 * stopped services are surfaced — a stopped one is still pickable, and
 * the AI/HTTP runtime asks the companion to start it (ensure-by-port)
 * the moment a flow that uses it runs. The status is folded into the
 * dropdown label so the user knows what they're picking.
 *
 * LLM-category services are assumed OpenAI-compatible on
 * `/v1/chat/completions` — the universal convention for llama.cpp,
 * Ollama, LM Studio, vLLM and TGI. We deliberately leave `bodyTemplate`
 * EMPTY so the AI LLM node uses its standard OpenAI request path (which
 * handles vision + streaming) rather than the generic custom-template
 * branch. Everything else is tagged `service_call` only.
 *
 * Returns null only when there's no usable port.
 */
function mapToCustomService(s: CompanionServiceSnapshot): CustomService | null {
  const port = s.port || s.defaultPort;
  if (!port) return null;
  const cat = (s.category || '').toLowerCase();
  const base = `http://127.0.0.1:${port}`;
  const id = `companion:${s.id}`;
  const running = s.status === 'running';
  const statusTag = running ? 'running' : s.status;
  // Keep the node/palette description to a tidy one-liner — a companion can
  // report a long paragraph (krea2's is), which overflows the node body.
  const rawDesc = (s.description || `Managed by the OAIY Companion on port ${port}.`).trim();
  const firstSentence = rawDesc.split('. ')[0];
  const baseDesc =
    firstSentence.length > 90 ? `${firstSentence.slice(0, 87).trimEnd()}…` : firstSentence;
  const desc = running
    ? baseDesc
    : `${baseDesc} (currently ${s.status} — auto-starts when the flow runs)`;

  // 1. An explicit call contract declared by the template WINS — the service
  // tells us exactly how to call it, so even a custom/third-party service
  // surfaces ready-to-run with no per-service knowledge baked in here.
  const node = s.node;
  if (node) {
    if (node.apiFormat === 'openai') {
      return {
        id,
        name: `${s.name} (companion · ${statusTag})`,
        description: desc,
        endpoint: `${base}${node.endpoint || '/v1/chat/completions'}`,
        method: (node.method || 'POST') as CustomService['method'],
        headers: '{}',
        bodyTemplate: node.bodyTemplate ?? '',
        responseType: 'json',
        responsePath: node.responsePath || 'choices.0.message.content',
        isBuiltIn: false,
        icon: node.icon || '🤖',
        nodeTypes: ['ai_llm', 'service_call'],
        apiFormat: 'openai',
        model: '',
        installHint: `Managed by the OAIY Companion (port ${port}). Start/stop it from the companion's Services tab.`,
      };
    }
    return {
      id,
      name: `${s.name} (companion · ${statusTag})`,
      description: desc,
      endpoint: node.endpoint ? `${base}${node.endpoint}` : base,
      method: (node.method || 'POST') as CustomService['method'],
      headers: '{}',
      bodyTemplate: node.bodyTemplate ?? '{{inputRaw}}',
      responseType: 'json',
      responsePath: node.responsePath ?? '',
      isBuiltIn: false,
      icon: node.icon || '🧩',
      nodeTypes: ['service_call'],
      installHint: `Managed by the OAIY Companion (port ${port}).`,
    };
  }

  // 2. No declared contract → fall back to a best-effort category convention.
  // Browser-automation services speak a bespoke session API — not surfaced.
  if (cat === 'browser') return null;
  const isLlm = cat === 'llm';

  if (isLlm) {
    return {
      id,
      name: `${s.name} (companion · ${statusTag})`,
      description: desc,
      endpoint: `${base}/v1/chat/completions`,
      method: 'POST',
      headers: '{}',
      // Empty → AI node uses the standard OpenAI body, not custom-template.
      bodyTemplate: '',
      responseType: 'json',
      responsePath: 'choices.0.message.content',
      isBuiltIn: false,
      icon: '🤖',
      nodeTypes: ['ai_llm', 'service_call'],
      apiFormat: 'openai',
      model: '',
      installHint: `Managed by the OAIY Companion (port ${port}). Start/stop it from the companion's Services tab.`,
    };
  }

  // OAIY-managed image/video services expose POST /generate {prompt,...} and
  // return an image/video URL — wire that contract so a dragged node runs as-is
  // (otherwise it POSTs to the bare base URL and gets a 405 Method Not Allowed).
  const isVideo = cat === 'video generation';
  if (isVideo || cat === 'image generation') {
    return {
      id,
      name: `${s.name} (companion · ${statusTag})`,
      description: desc,
      endpoint: `${base}/generate`,
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{"prompt": {{input}}}',
      responseType: 'json',
      responsePath: isVideo ? 'videoUrl' : 'imageUrl',
      isBuiltIn: false,
      icon: isVideo ? '🎬' : '🎨',
      nodeTypes: ['service_call'],
      installHint: `Managed by the OAIY Companion (port ${port}).`,
    };
  }

  return {
    id,
    name: `${s.name} (companion · ${statusTag})`,
    description: desc,
    endpoint: base,
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{{inputRaw}}',
    responseType: 'json',
    responsePath: '',
    isBuiltIn: false,
    icon: '🧩',
    nodeTypes: ['service_call'],
    installHint: `Managed by the OAIY Companion (port ${port}).`,
  };
}

function publish(list: CustomService[]): void {
  const next = JSON.stringify(list);
  const prev = localStorage.getItem(COMPANION_SERVICE_STORAGE_KEY) ?? '[]';
  if (next === prev) return; // no change — skip the write + dropdown churn
  if (list.length === 0) {
    localStorage.removeItem(COMPANION_SERVICE_STORAGE_KEY);
  } else {
    localStorage.setItem(COMPANION_SERVICE_STORAGE_KEY, next);
  }
  // Refresh any mounted service dropdown so the change shows immediately.
  invalidateDynamicOptions();
}

/**
 * Synchronous snapshot of the current companion-service list. Used by
 * the `service:list` resolver; the compilers read the same key inline.
 */
export function listCompanionServices(): CustomService[] {
  try {
    const raw = localStorage.getItem(COMPANION_SERVICE_STORAGE_KEY);
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

async function pollOnce(): Promise<void> {
  const gen = pollGen;
  // Drop a result that arrives after a stop/start happened mid-flight, so a
  // late-resolving fetch can't restore a list the teardown already cleared.
  const commit = (list: CustomService[]) => {
    if (gen === pollGen) publish(list);
  };
  try {
    const controller = new AbortController();
    const t = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    const resp = await fetch(`${COMPANION_API_BASE}/api/services`, {
      method: 'GET',
      signal: controller.signal,
      credentials: 'omit',
      cache: 'no-store',
    });
    window.clearTimeout(t);
    if (!resp.ok) {
      commit([]);
      return;
    }
    const body = (await resp.json().catch(() => null)) as {
      services?: CompanionServiceSnapshot[];
    } | null;
    const services = body?.services ?? [];
    // Running services first so the immediately-usable ones sit at the
    // top of every dropdown; stable within each group otherwise.
    const ranked = [...services].sort((a, b) => {
      const ar = a.status === 'running' ? 0 : 1;
      const br = b.status === 'running' ? 0 : 1;
      return ar - br;
    });
    const mapped = ranked
      .map(mapToCustomService)
      .filter((x): x is CustomService => x !== null);
    commit(mapped);
  } catch {
    // Companion went away mid-poll, timeout, CORS — clear the list.
    commit([]);
  }
}

function startPoll(): void {
  if (pollTimer !== null) return;
  pollGen++;
  pollOnce();
  pollTimer = window.setInterval(pollOnce, POLL_INTERVAL_MS);
}

function stopPoll(): void {
  pollGen++; // invalidate any in-flight pollOnce so it can't re-publish
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
  publish([]); // authoritative clear, runs under the new generation
}

/**
 * Begin syncing companion services into the palette. Wires itself to the
 * companion detection probe — it polls `/api/services` only while the
 * companion is available, and clears the list when it's not. Safe to
 * call once at app boot (after startCompanionDetection()).
 */
export function startCompanionServiceSync(): void {
  subscribeCompanionStatus((info) => {
    if (info.available) startPoll();
    else stopPoll();
  });
}
