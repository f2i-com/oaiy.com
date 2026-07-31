/**
 * Typed wrapper around OAIY Desktop's localhost HTTP API.
 *
 * One module per concern (services / models / python) so the components
 * each pull a focused slice. All requests are JSON in/out and surface
 * non-2xx as thrown errors with the body's `error` field when present.
 */

export const API_BASE = 'http://127.0.0.1:17972';

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  // Bound every call so a wedged OAIY Desktop handler (TCP accepted but no response)
  // can't leave the promise pending forever and stack up under the 1.5-2s pollers.
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 15000);
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
      signal: ac.signal,
    });
    if (!resp.ok) {
      // Try to surface the API's error message — falls back to status text.
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        // Two error shapes in this API: the plain `{ error: "msg" }` the
        // services/models/python routes use, and the taxonomy `{ error: { code,
        // message } }` the bridge + AI gateway routes use. Surface either.
        if (typeof body?.error === 'string') detail = body.error;
        else if (typeof body?.error?.message === 'string') detail = body.error.message;
      } catch {
        /* not JSON — ignore */
      }
      throw new Error(`${resp.status}: ${detail}`);
    }
    // Empty body → no JSON to parse. Covers 204 No Content AND 202 Accepted
    // (fire-and-forget endpoints like /api/python/install return an empty
    // 202). Reading text first avoids "Unexpected end of JSON input" that
    // `resp.json()` throws on an empty body.
    const text = await resp.text();
    return (text ? JSON.parse(text) : undefined) as T;
  } finally {
    clearTimeout(timer);
  }
}

// ----- services -----

export type ServiceStatus =
  | 'stopped'
  | 'installing'
  | 'starting'
  | 'running'
  | 'errored';

export interface ServiceSnapshot {
  id: string;
  name: string;
  description: string;
  category: string;
  status: ServiceStatus;
  error: string | null;
  port: number;
  defaultPort: number;
  pid: number | null;
  startedAt: string | null;
  lastStatusChange: string;
  docsUrl: string | null;
  installable: boolean;
  /** True when the service declares an `uninstall` spec — show an Uninstall button. */
  uninstallable: boolean;
  /** True when the run executable exists on disk. Drives a single Install/Uninstall toggle
   *  button (Install when not installed, Uninstall when installed) instead of two buttons. */
  installed: boolean;
  /** GPU index this service is pinned to (CUDA_VISIBLE_DEVICES), or null for default
   *  placement. Set via the GPU picker. */
  gpu: number | null;
  /** Consecutive automatic restarts since it last ran healthily. */
  restartAttempts?: number;
  /** Why it died last time — survives the restart that replaces its log buffer. */
  lastCrash?: { code: number; at: string; detail?: string | null } | null;
  /** Automatic recovery gave up; a human needs to look (offer Repair). */
  needsRepair?: boolean;
}

/** A CUDA GPU present on the machine (from nvidia-smi). */
export interface GpuInfo {
  index: number;
  name: string;
}

export interface RegistrySnapshot {
  services: ServiceSnapshot[];
  dataDir: string;
}

export interface LogLine {
  timestamp: string;
  stream: 'stdout' | 'stderr';
  text: string;
}

/**
 * Service template — same shape as the on-disk JSON, used both for
 * snapshot replies AND POST /api/services bodies.
 */
export interface ServiceTemplateInput {
  id: string;
  name: string;
  description: string;
  category: string;
  defaultPort: number;
  install?: { kind: 'none' } | {
    kind: 'script';
    windows?: string;
    unix?: string;
  };
  run: {
    command: string;
    args: string[];
    env: Record<string, string>;
    cwd?: string | null;
  };
  health?: {
    url: string;
    timeoutSecs: number;
  };
  docsUrl?: string | null;
  /**
   * Bundled scripts (filename → contents) that make this a self-contained,
   * plug-and-play package. Written into the scripts dir on load so install/run
   * commands resolve. Empty for built-ins; populated by Export + on Import.
   */
  files?: Record<string, string>;
}

export const services = {
  list: () => request<RegistrySnapshot>('/api/services'),
  add: (template: ServiceTemplateInput) =>
    request<void>('/api/services', {
      method: 'POST',
      body: JSON.stringify(template),
    }),
  /** Import a self-contained service package (same shape as add). */
  import: (pkg: ServiceTemplateInput) =>
    request<void>('/api/services', {
      method: 'POST',
      body: JSON.stringify(pkg),
    }),
  /** Export a service as a self-contained package (template + bundled scripts). */
  export: (id: string) =>
    request<ServiceTemplateInput>(
      `/api/services/${encodeURIComponent(id)}/export`,
    ),
  delete: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  start: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/start`, {
      method: 'POST',
    }),
  /** Clear a tripped crash breaker and start from a clean slate. */
  repair: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/repair`, { method: 'POST' }),
  stop: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/stop`, {
      method: 'POST',
    }),
  install: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/install`, {
      method: 'POST',
    }),
  /** Remove a service's installed files so it can be cleanly reinstalled. */
  uninstall: (id: string) =>
    request<{ removed: number }>(
      `/api/services/${encodeURIComponent(id)}/uninstall`,
      { method: 'POST' },
    ),
  cancelInstall: (id: string) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/cancel-install`, {
      method: 'POST',
    }),
  logs: (id: string, tail = 200) =>
    request<LogLine[]>(
      `/api/services/${encodeURIComponent(id)}/logs?tail=${tail}`,
    ),
};

// ----- models / downloads -----

export interface ModelFile {
  name: string;
  path: string;
  sizeBytes: number;
  modified: string | null;
}

export interface ModelsSnapshot {
  rootDir: string;
  models: ModelFile[];
  /** Free space on the drive holding the models dir (null if unknown). */
  freeBytes: number | null;
}

export type DownloadStatus =
  | 'queued'
  | 'active'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface DownloadProgress {
  id: string;
  url: string;
  filename: string;
  subdir: string | null;
  destPath: string;
  status: DownloadStatus;
  bytesDownloaded: number;
  bytesTotal: number | null;
  startedAt: string;
  finishedAt: string | null;
  error: string | null;
  resumable: boolean | null;
  speedBps: number | null;
  etaSecs: number | null;
  /** SHA-256 of what was written, computed as it streamed. */
  sha256?: string | null;
  expectedSha256?: string | null;
  /** true matched, false did not (the file was deleted), null nothing to check
   *  against. Three-valued on purpose — "unverified" is not "bad". */
  verified?: boolean | null;
}

export interface CatalogModel {
  id: string;
  name: string;
  description: string;
  url: string;
  filename: string;
  subdir: string | null;
  sizeBytes: number;
}

export interface CatalogCategory {
  id: string;
  name: string;
  description: string;
  models: CatalogModel[];
}

export interface CatalogSnapshot {
  sourcePath: string;
  catalog: {
    categories: CatalogCategory[];
  };
}

export const models = {
  list: () => request<ModelsSnapshot>('/api/models'),
  catalog: () => request<CatalogSnapshot>('/api/models/catalog'),
  download: (url: string, filename?: string, subdir?: string) =>
    request<{ downloadId: string }>('/api/models/download', {
      method: 'POST',
      body: JSON.stringify({ url, filename, subdir }),
    }),
  downloads: () => request<DownloadProgress[]>('/api/models/downloads'),
  pause: (id: string) =>
    request<void>(`/api/models/downloads/${encodeURIComponent(id)}/pause`, {
      method: 'POST',
    }),
  resume: (id: string) =>
    request<void>(`/api/models/downloads/${encodeURIComponent(id)}/resume`, {
      method: 'POST',
    }),
  cancel: (id: string) =>
    request<void>(`/api/models/downloads/${encodeURIComponent(id)}/cancel`, {
      method: 'POST',
    }),
  delete: (name: string) =>
    request<void>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
};

// ----- python -----

export interface VenvInfo {
  name: string;
  path: string;
  pythonExecutable: string | null;
  sizeBytes: number;
  created: string | null;
  boundServices: string[];
}

export type PythonJobKind = 'installruntime' | 'createvenv';

export interface PythonJobStatus {
  kind: PythonJobKind;
  target: string;
  startedAt: string;
  finishedAt: string | null;
  exitCode: number | null;
  error: string | null;
}

export interface PythonSnapshot {
  installed: boolean;
  runtimeDir: string;
  interpreterPath: string | null;
  venvsDir: string;
  venvs: VenvInfo[];
  currentJob: PythonJobStatus | null;
}

export const python = {
  status: () => request<PythonSnapshot>('/api/python'),
  install: () => request<void>('/api/python/install', { method: 'POST' }),
  logs: (tail = 200) => request<LogLine[]>(`/api/python/logs?tail=${tail}`),
  createVenv: (name: string, requirements: string[] = []) =>
    request<{ path: string }>('/api/python/venvs', {
      method: 'POST',
      body: JSON.stringify({ name, requirements }),
    }),
  deleteVenv: (name: string) =>
    request<void>(`/api/python/venvs/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
};

// ----- plugins (Bridge Protocol) -----

export type PluginState =
  | 'installed'
  | 'stopped'
  | 'starting'
  | 'running'
  | 'unhealthy'
  | 'crashed'
  | 'disabled';

/** One plugin as the registry reports it (`GET /api/plugins`). */
export interface PluginRecord {
  id: string;
  state: PluginState;
  /** Present for every state that is not `running`. */
  reason?: string;
  dir: string;
  manifest?: {
    name: string;
    version: string;
    publisher?: string;
    description?: string;
    connectors?: Array<{ id: string; commands: string[] }>;
    events?: string[];
  };
  /** Capability names rewritten from a pre-OAIY spelling, for a UI nudge. */
  legacyCapabilities?: Array<[string, string]>;
  /** Declared capabilities that grant nothing (a typo, or a name OAIY has no
   *  equivalent for). Worth surfacing so a mystery denial has a cause. */
  unknownCapabilities?: string[];
  userDisabled: boolean;
  restartAttempts: number;
}

export interface PluginsSnapshot {
  plugins: PluginRecord[];
  root: string;
  scan: { added: number; unchanged: number; invalid: number };
}

// ----- pairing (a consumer earning a bearer token) -----

export interface PendingPairing {
  pairingId: string;
  product: string;
  label?: string;
  origin?: string;
  code: string;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  createdAtMs: number;
}

export interface PairedApp {
  id: string;
  product: string;
  label?: string;
  /** The browser origin this token was granted to. `product` and `label` are
   *  both supplied by the consumer and neither is unique, so this is the only
   *  field that says WHICH site holds the grant. Absent for a native caller. */
  origin?: string | null;
  createdAtMs: number;
}

export const pairing = {
  /** Pending requests awaiting the user's approval (privileged — the webview). */
  pending: () => request<{ pending: PendingPairing[] }>('/api/bridge/pairing'),
  approve: (id: string) =>
    request<void>(`/api/bridge/pairing/${encodeURIComponent(id)}/approve`, { method: 'POST' }),
  deny: (id: string) =>
    request<void>(`/api/bridge/pairing/${encodeURIComponent(id)}/deny`, { method: 'POST' }),
  /** Apps currently paired (secret-free). */
  paired: () => request<{ paired: PairedApp[] }>('/api/bridge/pairings'),
  revoke: (id: string) =>
    request<void>(`/api/bridge/pairings/${encodeURIComponent(id)}`, { method: 'DELETE' }),
};

export const plugins = {
  list: () => request<PluginsSnapshot>('/api/plugins'),
  start: (id: string) =>
    request<void>(`/api/plugins/${encodeURIComponent(id)}/start`, { method: 'POST' }),
  stop: (id: string) =>
    request<void>(`/api/plugins/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
  setEnabled: (id: string, enabled: boolean) =>
    request<PluginRecord>(`/api/plugins/${encodeURIComponent(id)}/enabled`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),
  logs: (id: string, tail = 200) =>
    request<{ lines: LogLine[] }>(
      `/api/plugins/${encodeURIComponent(id)}/logs?tail=${tail}`,
    ),
  /** Install (or replace) a plugin from a path on this machine — a plugin folder
   *  or a .tar.gz of one. Installing native code, so it's Desktop-window only. */
  install: (source: string) =>
    request<{ id: string; name: string; version: string; replaced: boolean }>(
      '/api/plugins/install',
      { method: 'POST', body: JSON.stringify({ source }) },
    ),
  /** Stop a plugin and remove it from disk. */
  uninstall: (id: string) =>
    request<void>(`/api/plugins/${encodeURIComponent(id)}`, { method: 'DELETE' }),
};

/** An invocable action surface a plugin contributes (`manifest.serviceDefinitions`). */
export interface ServiceDefinitionAction {
  id: string;
  title?: string;
  description?: string;
  sideEffects?: string;
  timeoutMs?: number;
  transport: { kind: string; command?: string };
  inputSchema?: unknown;
}

export interface ServiceDefinition {
  id: string;
  name: string;
  version?: string;
  description?: string;
  category?: string;
  actions: ServiceDefinitionAction[];
  /** Which plugin contributed it — stamped by the host, not self-declared. */
  pluginId: string;
}

export const serviceDefinitions = {
  list: () => request<{ definitions: ServiceDefinition[] }>('/api/services/definitions'),
  invoke: (definitionId: string, actionId: string, input?: unknown, idempotencyKey?: string) =>
    request<{ ok: boolean; result?: unknown }>(
      `/api/services/actions/${encodeURIComponent(definitionId)}/${encodeURIComponent(actionId)}/invoke`,
      { method: 'POST', body: JSON.stringify({ input, idempotencyKey }) },
    ),
};

// ----- bridge (connector commands + plugin events) -----

/** What a plugin-contributed screen needs from the host to be useful. */
/** The Node runtime the bundled CLI runs under. */
export interface NodeSnapshot {
  available: boolean;
  source: 'portable' | 'system' | 'none';
  path?: string | null;
  version?: string | null;
  installing: boolean;
  installsVersion: string;
}

export const nodeRuntime = {
  status: () => request<NodeSnapshot>('/api/node'),
  /** Download the pinned portable Node; progress streams via logs. */
  install: () => request<void>('/api/node/install', { method: 'POST' }),
  logs: (tail = 200) => request<LogLine[]>(`/api/node/logs?tail=${tail}`),
};

export interface RuntimeStatus {
  ready: boolean;
  deviceId: string;
  flowRuntime: { cliResolved: boolean; cliKind: string; detail?: string | null };
  runs: { queued: number; known: number; failed?: number };
  nodeRuntime?: NodeSnapshot | null;
  plugins: { serving: number; total: number };
}

export type RunStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'timed_out'
  | 'cancelled';

/** One row of run history. Mirrors the ledger's `RunRecord` wire shape. */
export interface RunRecord {
  runId: string;
  status: RunStatus;
  callerProduct: string;
  flowId?: string;
  correlationId: string;
  mode: string;
  runtime?: string;
  error?: {
    code: string;
    message: string;
    detail?: string;
    nodeId?: string;
    capability?: string;
    retryable?: boolean;
  };
  reservedAt: string;
  startedAt?: string;
  finishedAt?: string;
  triggerEvent?: string;
}

export interface RunHistory {
  runs: RunRecord[];
  total: number;
  byStatus?: Partial<Record<RunStatus, number>>;
}

/** An event that arrived and produced no work. */
export interface DeadLetter {
  id: string;
  source: string;
  event: string;
  reason: { kind: 'shed' | 'not_reserved'; detail?: string };
  envelope: unknown;
  recordedAtMs: number;
  attempts: number;
  lastAttemptMs?: number;
  lastOutcome?: string;
}

export const bridge = {
  /** Can this runtime actually run a flow? (health only asserts identity.) */
  status: () => request<RuntimeStatus>('/api/bridge/status'),
  /** Run history, newest first. `statuses` empty means every state — omitting
   *  the filter entirely is the worker's queued-only poll, not what a UI wants. */
  runs: (statuses: RunStatus[] | 'all' = 'all', limit = 50) =>
    request<RunHistory>(
      `/api/bridge/runs?status=${encodeURIComponent(
        statuses === 'all' ? 'all' : statuses.join(','),
      )}&limit=${limit}`,
    ),
  /** Invoke a connector command on a plugin, exactly as a flow would.
   *  `idempotencyKey` is required by the gateway for the plugin's journalled
   *  (physically side-effecting) commands, so callers should always pass one. */
  connectorRequest: (
    connectorId: string,
    command: string,
    payload?: unknown,
    idempotencyKey?: string,
  ) =>
    request<{ ok: boolean; result?: unknown }>(
      `/api/bridge/connectors/${encodeURIComponent(connectorId)}/request`,
      { method: 'POST', body: JSON.stringify({ command, payload, idempotencyKey }) },
    ),
  /** Poll plugin events after `since` (0 = from the current tail). */
  events: (since: number, limit = 100) =>
    request<{ events: Array<{ seq: number; envelope: Record<string, unknown> }>; next: number }>(
      `/api/bridge/events?since=${since}&limit=${limit}`,
    ),
  /** Events that arrived and produced no work, newest first. */
  deadLetters: (limit = 100) =>
    request<{ deadLetters: DeadLetter[]; total: number }>(
      `/api/bridge/deadletters?limit=${limit}`,
    ),
  /** Re-dispatch one against the CURRENT bindings. `reserved` says whether it
   *  finally produced a run; a redrive that fails again is not an error. */
  redrive: (id: string) =>
    request<{ reserved: boolean; outcomes: string[] }>(
      `/api/bridge/deadletters/${encodeURIComponent(id)}/redrive`,
      { method: 'POST' },
    ),
  dismissDeadLetter: (id: string) =>
    request<void>(`/api/bridge/deadletters/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  /** The AI sources union (local services + configured providers). */
  aiSources: () => request<{ sources: unknown[] }>('/api/ai/sources'),
};

// ----- AI providers (the local AI gateway) -----

export type AiProtocol = 'openai' | 'anthropic';
export type AiCapability = 'chat' | 'transcription' | 'speech' | 'embeddings' | 'realtime';

/** A configured AI provider, secret-free: the API key never leaves the device,
 *  so the wire carries only `hasKey`. */
export interface AiProviderPublic {
  id: string;
  name: string;
  category?: string | null;
  protocol: AiProtocol;
  baseUrl: string;
  model?: string | null;
  capabilities: AiCapability[];
  enabled: boolean;
  allowLocal: boolean;
  hasKey: boolean;
}

/** Upsert body — no key (the key is set separately via `setKey`; an edit that
 *  omits it preserves the existing key). */
export interface AiProviderInput {
  id: string;
  name: string;
  category?: string;
  protocol: AiProtocol;
  baseUrl: string;
  model?: string;
  capabilities?: AiCapability[];
  enabled?: boolean;
  allowLocal?: boolean;
}

/** The ChatGPT connector: a managed `codex` child that owns its own OAuth. */
export interface CodexStatus {
  available: boolean;
  connected: boolean;
  email?: string | null;
  planType?: string | null;
  accountType?: string | null;
  detail?: string | null;
}

export interface CodexLogin {
  loginId?: string | null;
  authUrl?: string | null;
  verificationUrl?: string | null;
  userCode?: string | null;
}

export const codex = {
  status: () => request<CodexStatus>('/api/ai/codex/status'),
  /** Begin sign-in. `deviceCode` shows a code to type instead of a redirect. */
  startLogin: (deviceCode = true) =>
    request<CodexLogin>('/api/ai/codex/login', {
      method: 'POST',
      body: JSON.stringify({ deviceCode }),
    }),
  cancelLogin: () => request<void>('/api/ai/codex/login', { method: 'DELETE' }),
  logout: () => request<void>('/api/ai/codex/logout', { method: 'POST' }),
};

export const aiProviders = {
  list: () => request<{ providers: AiProviderPublic[] }>('/api/ai/providers'),
  upsert: (input: AiProviderInput) =>
    request<{ id: string }>('/api/ai/providers', { method: 'POST', body: JSON.stringify(input) }),
  delete: (id: string) =>
    request<void>(`/api/ai/providers/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  /** Set the plaintext key (stored on-device), or clear it with `null`. */
  setKey: (id: string, key: string | null) =>
    request<void>(`/api/ai/providers/${encodeURIComponent(id)}/key`, {
      method: 'POST',
      body: JSON.stringify({ key }),
    }),
  /** A real authenticated round trip. Resolves on reachable+authorized; throws
   *  (the API returns 502) otherwise. */
  test: (id: string) =>
    request<{ ok: boolean }>(`/api/ai/providers/${encodeURIComponent(id)}/test`, { method: 'POST' }),
};

// ----- formatting helpers used by multiple components -----

export function formatBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return '';
  return new Date(iso).toLocaleString();
}

export function formatSpeed(bps: number | null | undefined): string {
  if (bps == null || bps === 0) return '';
  // `formatBytes` is already per-unit; just append /s.
  return `${formatBytes(bps)}/s`;
}

export function formatEta(secs: number | null | undefined): string {
  if (secs == null || secs < 0) return '';
  if (secs < 60) return `${secs}s left`;
  if (secs < 3600) {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return s > 0 ? `${m}m ${s}s left` : `${m}m left`;
  }
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return m > 0 ? `${h}h ${m}m left` : `${h}h left`;
}

/**
 * Call a Tauri command on OAIY Desktop's Rust side. Inside the
 * OAIY Desktop webview the `__TAURI_INTERNALS__` global is always present;
 * in a plain browser tab it's absent, so we reject with a clear message.
 */
export function tauriInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const tauri = (window as unknown as {
    __TAURI_INTERNALS__?: { invoke: (cmd: string, args: unknown) => Promise<unknown> };
  }).__TAURI_INTERNALS__;
  if (!tauri) {
    return Promise.reject(new Error('Not running in the OAIY desktop app.'));
  }
  return tauri.invoke(cmd, args ?? {}) as Promise<T>;
}

/** Whether we're running inside OAIY Desktop's Tauri webview. */
export function isTauri(): boolean {
  return !!(window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
}

/** Open a folder or file in the OS file manager. */
export function openInExplorer(path: string): Promise<void> {
  return tauriInvoke<void>('open_path', { path });
}

/**
 * Open an external URL in the system default browser. In the Tauri app this
 * invokes the native `open_url` command; in a plain dev browser (vite :17973)
 * it falls back to window.open so the link works there too.
 */
export function openExternal(url: string): void {
  const internals = (
    window as unknown as {
      __TAURI_INTERNALS__?: { invoke: (cmd: string, args: unknown) => Promise<unknown> };
    }
  ).__TAURI_INTERNALS__;
  if (internals?.invoke) {
    internals.invoke('open_url', { url }).catch(() => window.open(url, '_blank', 'noopener,noreferrer'));
  } else {
    window.open(url, '_blank', 'noopener,noreferrer');
  }
}

// ----- config (data dir) — Tauri commands, desktop-only -----

export interface DesktopConfig {
  /** The dir the running app is actually using right now. */
  activeDir: string;
  /** OS default (what "Reset" returns to). */
  defaultDir: string;
  /** Override written to the pointer file, if any. */
  configuredDir: string | null;
  /** A custom dir is configured (differs from default). */
  isCustom: boolean;
  /** A change is pending — app must restart to apply it. */
  restartRequired: boolean;

  // ----- models dir (separate override; defaults to <dataDir>/models) -----
  /** The models dir the running app is actually using. */
  modelsActiveDir: string;
  /** Where the models dir falls back to (`<activeDataDir>/models`). */
  modelsDefaultDir: string;
  /** The `modelsDir` override written to the pointer, if any. */
  modelsConfiguredDir: string | null;
  /** A custom models dir is configured. */
  modelsIsCustom: boolean;
  /** A models-dir change is pending — restart to apply. */
  modelsRestartRequired: boolean;

  /** The GGUF a single-model server (llama.cpp) is set to load, if the user
   * picked one in its Model selector (null = the `model.gguf` default). */
  llamaModel: string | null;

  /** The model name the Ollama node uses, if the user picked one in its Model
   * selector (null = the pre-pulled default qwen2.5:0.5b). */
  ollamaModel: string | null;
}

/** What a data-folder migration would move (old → pending folder). */
export interface MigratePlan {
  oldDir: string;
  newDir: string;
  fileCount: number;
  totalBytes: number;
  /** Migratable subdirs present in the old folder (models/templates/bin). */
  subdirs: string[];
  /** True when there's a pending change AND something to move. */
  canMigrate: boolean;
}

/** Live migration progress (polled while a copy/move runs). */
export interface MigrationProgress {
  running: boolean;
  mode: string;
  filesTotal: number;
  filesDone: number;
  bytesTotal: number;
  bytesDone: number;
  current: string;
  done: boolean;
  error: string | null;
}

export const appConfig = {
  get: () => tauriInvoke<DesktopConfig>('get_config'),
  setDataDir: (path: string) => tauriInvoke<void>('set_data_dir', { path }),
  /** Set (or reset, with '') the models folder — separate from the data dir. */
  setModelsDir: (path: string) => tauriInvoke<void>('set_models_dir', { path }),
  pickFolder: () => tauriInvoke<string | null>('pick_folder'),
  restart: () => tauriInvoke<void>('restart_app'),
  migrationPlan: () => tauriInvoke<MigratePlan>('migration_plan'),
  startMigration: (mode: 'copy' | 'move') =>
    tauriInvoke<void>('start_migration', { mode }),
  migrationStatus: () => tauriInvoke<MigrationProgress>('migration_status'),
  /** Whether a HuggingFace token is saved (never returns the token itself). */
  getHfTokenStatus: () => tauriInvoke<boolean>('get_hf_token_status'),
  /** Save (or clear, with '') the HuggingFace token for gated downloads. */
  setHfToken: (token: string) => tauriInvoke<void>('set_hf_token', { token }),

  // ----- additional model folders (extra search roots beyond the primary) -----
  /** The extra model folders registered beyond the primary models dir. */
  listModelDirs: () => tauriInvoke<string[]>('list_model_dirs'),
  /** Register an extra (read-only) model folder; returns the updated list. */
  addModelDir: (path: string) => tauriInvoke<string[]>('add_model_dir', { path }),
  /** Remove a registered extra model folder; returns the updated list. */
  removeModelDir: (path: string) => tauriInvoke<string[]>('remove_model_dir', { path }),

  // ----- single-model server (llama.cpp) model selection -----
  /** Loadable GGUFs discovered across the model folders — the picker options. */
  listGgufModels: () => tauriInvoke<string[]>('list_gguf_models'),
  /** Set (or reset, with '') which GGUF the llama.cpp server loads. Applies to
   * the next start of the service — no app restart needed. */
  setLlamaModel: (path: string) => tauriInvoke<void>('set_llama_model', { path }),

  // ----- multi-model server (Ollama) model selection -----
  /** Models pulled into the running Ollama server — the Ollama picker options
   * (empty + throws when Ollama isn't running). */
  listOllamaModels: () => tauriInvoke<string[]>('list_ollama_models'),
  /** Set (or reset, with '') the model NAME the Ollama node uses. Applies to
   * the next flow run — no app restart needed. */
  setOllamaModel: (model: string) => tauriInvoke<void>('set_ollama_model', { model }),

  // ----- per-service GPU pinning -----
  /** CUDA GPUs present (index + name). Empty on a box without an NVIDIA GPU. */
  listGpus: () => tauriInvoke<GpuInfo[]>('list_gpus'),
  /** Where the desktop app writes its log, if logging is attached. */
  logPath: () => tauriInvoke<string | null>('log_path'),
  /** Pin a service to a GPU index (CUDA_VISIBLE_DEVICES), or pass null to clear. Applies to
   * the service's next start — no app restart needed. */
  setServiceGpu: (id: string, gpu: number | null) =>
    tauriInvoke<void>('set_service_gpu', { id, gpu }),
};
