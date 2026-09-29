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
  /** The user ticked "start with the app" for this service. Independent of
   *  whether it is running right now. Optional so a snapshot from an older
   *  desktop build (no such field) reads as unticked rather than undefined. */
  autostart?: boolean;
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
  /** Tick/untick "start this service when the app starts". A stored preference
   *  only — it never starts or stops the service now. */
  setAutostart: (id: string, enabled: boolean) =>
    request<void>(`/api/services/${encodeURIComponent(id)}/autostart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }),
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
  /** One PAGE of the library — see `total`. */
  models: ModelFile[];
  /** Files in the whole library, so the count shown is never a page length. */
  total: number;
  offset: number;
  limit: number;
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
  /** One page of the library. The panel polls this, so a big library must not
   *  be serialised in full every tick. */
  list: (limit = 200, offset = 0) =>
    request<ModelsSnapshot>(`/api/models?offset=${offset}&limit=${limit}`),
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

/**
 * What is known about a plugin's package (`plugins/trust.rs`).
 *
 * `verified`: signed by a publisher this OAIY trusts, every file as signed.
 * `quarantined`: it carries a signature that does not check out; never started.
 * `unsigned`: no signature, in a release build, and not trusted by the person; never
 * started until they trust this exact package.
 * `unsigned-dev`: no signature; it runs because this is a developer build.
 * `trusted-local`: no signature; the person trusted this exact package.
 */
export type PackageTrustState = 'verified' | 'quarantined' | 'unsigned' | 'unsigned-dev' | 'trusted-local';

export interface PackageTrust {
  state: PackageTrustState;
  /** Who signed it (`verified` only). */
  publisher?: string;
  /** The pinned key that verified it (`verified` only). */
  keyId?: string;
  /** The release the signer wrote into the signed payload (`verified` only). */
  version?: string;
  /** Why, for every state but `verified`. */
  reason?: string;
  /** When the person trusted it (`trusted-local` only). */
  trustedAt?: string;
}

/** One plugin as the registry reports it (`GET /api/plugins`). */
export interface PluginRecord {
  id: string;
  state: PluginState;
  /** Present for every state that is not `running`. */
  reason?: string;
  dir: string;
  /** Absent only when the manifest could not be loaded. */
  trust?: PackageTrust;
  manifest?: {
    name: string;
    version: string;
    publisher?: string;
    description?: string;
    connectors?: Array<{ id: string; commands: string[] }>;
    events?: string[];
    /** Commands with effects a retry must not repeat. */
    commands?: { journalled?: string[] };
    /** schemaVersion 4: the built-in modules it provides (the phone, the calendar). */
    modules?: { provides: string[]; connector?: string };
    /** schemaVersion 4: its service-definition actions offered to the agent. */
    agentTools?: Array<{ action: string; name: string; description?: string; audience?: string[]; confirm?: string }>;
    /** schemaVersion 4: its setup wizard, as the desktop validated it (version and title filled in). */
    setup?: SetupDeclJson;
    /** Its screens, pages and cards (presentation only; see sections.ts). */
    ui?: { screens?: Array<{ id: string; title?: string; entry?: string; files?: string[] }> } & Record<string, unknown>;
  };
  /** Capability names rewritten from a pre-OAIY spelling, for a UI nudge. */
  legacyCapabilities?: Array<[string, string]>;
  /** Declared capabilities that grant nothing (a typo, or a name OAIY has no
   *  equivalent for). Worth surfacing so a mystery denial has a cause. */
  unknownCapabilities?: string[];
  userDisabled: boolean;
  restartAttempts: number;
  /** Supervisor health report; absent before the first probe or after failure. */
  lastHealth?: { status: string; detail?: string; components?: Record<string, unknown> };
  lastHealthAt?: string;
  lastHealthError?: string;
}

/** One test of a command's answer: a dot-separated `path` and exactly one operator. A missing path equals null. */
export interface SetupConditionJson {
  path: string;
  equals?: unknown;
  present?: boolean;
  in?: unknown[];
  notIn?: unknown[];
}

/** A step's `done` or `when`: a read-only command sent with no payload, and one condition inline or `all` of a list. */
export type SetupCheckJson = { command: string; all?: SetupConditionJson[] } & Partial<SetupConditionJson>;

export type SetupRequirementJson =
  | { kind: 'service'; id: string; why?: string }
  /** Met by the model chosen in Engines for the group: a plugin never names a model. */
  | { kind: 'engineModel'; group: string; why?: string };

export interface SetupFieldJson {
  key: string;
  label: string;
  type: 'bool' | 'choice' | 'text' | 'number';
  options?: Array<{ value: string | number | boolean; label: string }>;
  help?: string;
}

/** One step of a plugin's setup (the manifest's `setup.steps[]`). */
export type SetupStepJson = {
  id: string;
  title: string;
  description?: string;
  optional?: boolean;
  when?: SetupCheckJson;
} & (
  | { kind: 'permissions' }
  | { kind: 'requirements'; requires: SetupRequirementJson[] }
  /** `read` absent: `settings.get`, the settings under `settings`; a `read` with no path: the whole answer. */
  | { kind: 'settings'; fields: SetupFieldJson[]; read?: { command: string; path?: string }; write?: { command: string } }
  | { kind: 'screen'; screen: string; view: string; done?: SetupCheckJson }
  | { kind: 'host'; action: string }
);

/** A plugin's setup (schemaVersion 4), as its record carries it. */
export interface SetupDeclJson {
  version: number;
  title: string;
  steps: SetupStepJson[];
}

export interface PluginsSnapshot {
  plugins: PluginRecord[];
  root: string;
  scan: { added: number; unchanged: number; invalid: number };
}

// ----- modules (the parts of OAIY a plugin brings: the phone, the calendar) -----

/** One module as `GET /api/modules` reports it. */
export interface ModuleRecord {
  id: string;
  name: string;
  /** On while a plugin provides it (a crashed or stopped provider keeps it on). */
  enabled: boolean;
  builtin: boolean;
  /** Why it is off. */
  reason?: string;
  provider: { pluginId: string; name: string; state: PluginState; connector?: string; declared: boolean } | null;
  leases?: string[];
  uses?: string[];
  store?: string[];
}

/** One plugin screen in the dashboard (`ui.nav[]`). */
export interface PageContribution {
  /** `plugin:<pluginId>:<navId>`. */
  view: string;
  pluginId: string;
  pluginName: string;
  navId: string;
  label: string;
  /** The `ui.screens` id it shows. */
  screen: string;
  icon?: string;
  badge?: string;
  module?: string;
}

/**
 * A sidebar section plugins add or extend. `id` is a built-in section's id
 * (its pages go after that section's own tabs), `plugin-section:<pluginId>:<id>`
 * for a plugin's own (`ui.sections`), or a lone page's own view id.
 */
export interface SectionContribution {
  id: string;
  builtin: boolean;
  pluginId?: string;
  pluginName?: string;
  label?: string;
  icon?: string;
  /** A plugin's own section only; a built-in keeps its own group. */
  group?: 'Home' | 'Work' | 'Setup';
  badge?: string;
  module?: string;
  pages: PageContribution[];
}

/** An Overview card a plugin contributes (`ui.overview[]`). */
export interface OverviewContribution {
  pluginId: string;
  pluginName: string;
  id: string;
  kind: 'hero' | 'status' | 'tile';
  title: string;
  icon?: string;
  module?: string;
  /** Texts by name: plain, `$health.<path>` or `$poll.<statusCardId>.<path>` (looked up by bind.ts, never evaluated). */
  bind: Record<string, string>;
  cta?: { label: string; view: string };
  /** Where a click on a tile goes. */
  view?: string;
}

/** A read-only plugin command polled for `$poll` bindings (`ui.statusCards[]`). */
export interface PollContribution {
  pluginId: string;
  /** The status card's id, as `$poll.<id>` names it. */
  id: string;
  connector: string;
  command: string;
  /** At least 5000. */
  intervalMs: number;
}

/** A plugin's service-definition action offered to the agent (`agentTools[]`). */
export interface AgentToolContribution {
  pluginId: string;
  name: string;
  action: string;
  definition: string;
  actionId: string;
  description: string;
  inputSchema: Record<string, unknown>;
  sideEffects?: string;
  audience: string[];
  confirm?: string;
  timeoutMs?: number;
}

/** What the enabled plugins add (a turned-off plugin's, and a module that is off's, are left out). */
export interface Contributions {
  sections: SectionContribution[];
  overview: OverviewContribution[];
  polls: PollContribution[];
  agent: { tools: AgentToolContribution[] };
  setup: Array<{ pluginId: string; title: string; version: number; steps: number }>;
}

export interface ModulesSnapshot {
  /** Moves only when something here changes (it is the ETag). */
  revision: number;
  modules: ModuleRecord[];
  /** Partial from an older desktop (step 1's was `{}`): read each list with `?? []`. */
  contributions: Partial<Contributions>;
  warnings: string[];
}

export const modules = {
  /** The snapshot and its ETag; `null` when it has not changed since `etag` (304). */
  list: async (etag?: string | null): Promise<{ snapshot: ModulesSnapshot; etag: string | null } | null> => {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 15000);
    try {
      const resp = await fetch(`${API_BASE}/api/modules`, {
        headers: etag ? { 'If-None-Match': etag } : {},
        cache: 'no-store',
        signal: ac.signal,
      });
      if (resp.status === 304) return null;
      if (!resp.ok) throw new Error(`${resp.status}: ${resp.statusText}`);
      return { snapshot: (await resp.json()) as ModulesSnapshot, etag: resp.headers.get('ETag') };
    } finally {
      clearTimeout(timer);
    }
  },
};

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

export interface CompanionEndpointKey {
  kty: string;
  crv: string;
  publicKey: string;
  thumbprint: string;
}
export interface CompanionApproved {
  deviceId: string;
  displayName: string;
  endpointKey: CompanionEndpointKey;
  approvedAt: string;
}
export interface CompanionPending {
  id: string;
  deviceId: string;
  displayName: string;
  thumbprint: string;
  /** Grouped hex the operator reads aloud against the phone's screen. */
  fingerprint: string;
  endpointKey: CompanionEndpointKey;
  receivedAt: string;
}
export interface CompanionStatus {
  available: boolean;
  protectionLabel?: string;
  endpointKey?: CompanionEndpointKey;
  rosterRevision: number;
  rosterHash: string;
  approvedMobiles: CompanionApproved[];
  pendingApprovals: CompanionPending[];
  remoteAccessReady: boolean;
  warning?: string;
}
export interface CompanionOfferRequest {
  appId?: string;
  workspaceId?: string;
  desktopConnectionId?: string;
}
export interface CompanionOffer {
  requestId: string;
  payload: Record<string, unknown>;
  /** The exact JSON to paste into the phone when a camera isn't an option. */
  encodedPayload: string;
  qrSvg: string;
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

/**
 * Companion device trust, scoped to the broker plugin that asked.
 *
 * The plugin id is a path segment rather than an implicit "the companion
 * plugin" because the host gates each route on that plugin declaring
 * `oaiy.companion.admission` — the id IS the authorisation subject, so it has
 * to travel with the call.
 */
export interface CompanionRelayStatus {
  configured: boolean;
  baseUrl?: string;
  appId?: string;
  /** Whether a credential is held. The credential itself never comes back. */
  hasToken: boolean;
}
export interface CompanionRelayRequest {
  baseUrl: string;
  token: string;
  appId?: string;
}

export const companion = {
  /** The relay is per-machine, not per-plugin: it is the user's account with a
   *  relay deployment, and two broker plugins reach the same one. */
  relayStatus: () => request<CompanionRelayStatus>('/api/companion/relay'),
  setRelay: (body: CompanionRelayRequest) =>
    request<CompanionRelayStatus>('/api/companion/relay', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  clearRelay: () => request<CompanionRelayStatus>('/api/companion/relay', { method: 'DELETE' }),
  status: (plugin: string) => request<CompanionStatus>(`/api/companion/${encodeURIComponent(plugin)}/pairing`),
  createOffer: (plugin: string, body: CompanionOfferRequest) =>
    request<CompanionOffer>(`/api/companion/${encodeURIComponent(plugin)}/pairing/offers`, {
      method: 'POST',
      body: JSON.stringify(body ?? {}),
    }),
  receiveResponse: (plugin: string, response: unknown) =>
    request<CompanionPending>(`/api/companion/${encodeURIComponent(plugin)}/pairing/responses`, {
      method: 'POST',
      body: JSON.stringify(response),
    }),
  approve: (plugin: string, id: string) =>
    request<CompanionApproved>(
      `/api/companion/${encodeURIComponent(plugin)}/pairing/approvals/${encodeURIComponent(id)}/approve`,
      { method: 'POST' },
    ),
  deny: (plugin: string, id: string) =>
    request<void>(
      `/api/companion/${encodeURIComponent(plugin)}/pairing/approvals/${encodeURIComponent(id)}/deny`,
      { method: 'POST' },
    ),
  revoke: (plugin: string, thumbprint: string) =>
    request<void>(
      `/api/companion/${encodeURIComponent(plugin)}/mobiles/${encodeURIComponent(thumbprint)}`,
      { method: 'DELETE' },
    ),
  rotate: (plugin: string) =>
    request<CompanionStatus>(`/api/companion/${encodeURIComponent(plugin)}/identity/rotate`, {
      method: 'POST',
    }),
};

export interface LinkConnector {
  id: string;
  name: string;
  description?: string;
  docsUrl?: string;
  defaultBaseUrl?: string;
  scopes: string[];
}
export type LinkPhase =
  | { phase: 'idle' }
  | { phase: 'awaitingBrowser'; authorizeUrl: string }
  | { phase: 'exchanging' }
  | { phase: 'linked' }
  | { phase: 'failed'; message: string }
  | { phase: 'cancelled' };
export interface LinkStatus {
  linked: boolean;
  connectorId?: string;
  connectorName?: string;
  baseUrl?: string;
  accountName?: string;
  accountId?: string;
  grantedScopes?: string;
  linkedAt?: string;
  /** When the provider was last told this desktop is here. */
  lastHeartbeatAt?: string;
  /** Why the last heartbeat failed, if it did. */
  heartbeatError?: string;
  /** When the command lane last polled cleanly. On a long poll, a clean return
   *  is the only evidence it is alive — without it, "never started" and
   *  "running fine" look identical. */
  lastRelayAt?: string;
  /** Why the command lane stopped. Its failures are invisible on this machine:
   *  they show up on the provider's website as "no desktop picked it up in
   *  time", which reads as a broken connection rather than a lane erroring. */
  relayError?: string;
  /** When this desktop last looked at the account's queued flow runs, and why
   *  it stopped if it did (a run it claimed and could not report on). */
  lastFlowRunAt?: string;
  flowRunError?: string;
  /** Which lanes the linked connector declares at all, so the UI can tell a
   *  lane that is broken from one this provider never had. Absent from an older
   *  host, hence optional. */
  heartbeatSupported?: boolean;
  relaySupported?: boolean;
  sealedFlowsSupported?: boolean;
  lastSealedFlowAt?: string;
  sealedFlowError?: string;
  /** This desktop's storage-node enrolment. The fingerprint is what the owner
   *  compares against the one their browser shows before approving — the whole
   *  ceremony rests on the two matching, so it has to be visible here. */
  dataNode?: {
    fingerprint: string;
    /** The provider's vocabulary: pending | approved | revoked. */
    status: string;
    /** Approved AND holding an unexpired certificate — a node can be approved
     *  with an expired one and have no authority. */
    approved: boolean;
    keyGeneration: number;
    displayName?: string;
  };
  dataNodeError?: string;
  dataNodeSupported?: boolean;
  /** Plugin events kept for the account until FormLogic can take them. */
  outbox?: { waiting: number; oldestAt?: string | null; lastError?: string | null; lastSentAt?: string | null; nextAttemptAt?: string | null };
  attempt: LinkPhase;
  /** Every provider this build can link to — the UI hardcodes no list. */
  available: LinkConnector[];
}

/** The one outbound account link. Generic over the provider: see
 *  `resources/connectors/*.json`. */
export const link = {
  status: () => request<LinkStatus>('/api/link'),
  start: (connectorId: string, baseUrl: string) =>
    request<{ authorizeUrl: string }>('/api/link/start', {
      method: 'POST',
      body: JSON.stringify({ connectorId, baseUrl }),
    }),
  unlink: () => request<LinkStatus>('/api/link', { method: 'DELETE' }),
  /** Abandon an attempt in flight. Distinct from unlink, which throws away a
   *  credential we already hold. */
  cancel: () => request<LinkStatus>('/api/link/cancel', { method: 'POST' }),
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
   *  or a .tar.gz of one. Installing native code, so it's Desktop-window only.
   *  `setup` is there when the plugin declares a setup wizard; `trust` is what its
   *  package was found to be (a signed package that fails is not installed at all). */
  install: (source: string) =>
    request<{ id: string; name: string; version: string; replaced: boolean; trust?: PackageTrust; setup?: { version: number; title: string } }>(
      '/api/plugins/install',
      { method: 'POST', body: JSON.stringify({ source }) },
    ),
  /** Trust this exact, unsigned package (bound to a digest of its files: a change to
   *  any file needs it again). Takes the plugin's id and nothing else; Desktop-window
   *  only, like install. */
  trust: (id: string) =>
    request<PluginRecord>(`/api/plugins/${encodeURIComponent(id)}/trust`, { method: 'POST' }),
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
  source: 'bundled' | 'portable' | 'system' | 'none';
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
  flowRuntime: {
    cliResolved: boolean;
    cliKind: string;
    /** What the CLI runs flows on. `ready` only when it answered that it is ZIPP; `unknown` until it has been asked. */
    engine?: { name: string | null; release: string | null; status: 'ready' | 'unavailable' | 'unknown'; reason?: string | null };
    detail?: string | null;
  };
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
  reason: { kind: 'shed' | 'not_reserved' | 'not_delivered'; detail?: string };
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
  /** Forget finished runs. Queued and running work is kept — the host refuses
   *  to delete a record for work still in flight. Returns how many went. */
  clearRuns: () =>
    request<{ cleared: number; total: number }>('/api/bridge/runs', { method: 'DELETE' }),
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

// ----- updates -----

export type UpdateState = 'idle' | 'checking' | 'upToDate' | 'available' | 'downloading' | 'ready' | 'installing' | 'failed';

/** One reason an update cannot be installed now, in words for a person. */
export interface UpdateBlocker {
  /**
   * call (OAIY's own line), phoneCall (a phone plugin reports one), callUnknown (a phone plugin cannot say), agentTask, download,
   * mediaJob, enginesUnknown (the engines cannot say), installing, migration or starting. The window shows the message and keys on the code.
   */
  code: string;
  message: string;
}

/** `GET /api/update/status`: what OAIY knows about a newer release. */
export interface UpdateStatus {
  state: UpdateState;
  currentVersion: string;
  /** Always the stable releases. */
  channel: string;
  latestVersion: string | null;
  notes: string | null;
  publishedAt: string | null;
  lastCheckedAt: string | null;
  /** The last thing that went wrong, in words for a person. */
  error: string | null;
  failedDuring: 'check' | 'download' | 'install' | null;
  progress: { downloaded: number; total: number | null } | null;
  /** Something to say beside the state (no release for this platform, say). */
  note: string | null;
  /** This copy can download and install an update itself. */
  canAutoUpdate: boolean;
  /** When it cannot, why. */
  manualReason: string | null;
  /** What stops an install now; empty when nothing does. */
  blockers: UpdateBlocker[];
  /** The releases page, for a manual download. */
  manualUrl: string;
  /** Whether OAIY looks for updates by itself (a little after it starts, then daily). */
  autoCheck: boolean;
  /** Seconds until a check is allowed again; null when now. */
  nextCheckIn: number | null;
}

export const updates = {
  status: () => request<UpdateStatus>('/api/update/status'),
  /** Look for a newer release now (at most once every 30 seconds). Inside the desktop app it is the window's own command. */
  check: () => (isTauri() ? tauriInvoke<UpdateStatus>('update_check') : request<UpdateStatus>('/api/update/check', { method: 'POST' })),
  /** Download and verify the release the last check found. Answers the status; follow the progress in `status`. */
  download: () => tauriInvoke<UpdateStatus>('update_download'),
  /** Restart to update: stops OAIY, hands over to the installer, and OAIY starts again. Takes no argument. */
  install: () => tauriInvoke<void>('update_install'),
  /** Whether OAIY looks for updates by itself. A check asked for here works either way. */
  setAutoCheck: (enabled: boolean) => tauriInvoke<void>('set_update_auto_check', { enabled }),
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

// ----- calendar -----

/** Open from `open` to `close` (`HH:MM`). */
export interface CalendarSpan {
  open: string;
  close: string;
}

export interface CalendarService {
  id?: string;
  name: string;
  minutes: number;
  description?: string;
  price?: string;
}

export interface CalendarSettings {
  business: string;
  /** The name the receptionist calls itself on calls and texts; empty: the phone's own (Aokie). */
  receptionist: string;
  /** Seven days, Monday first; an empty day is closed. */
  hours: CalendarSpan[][];
  services: CalendarService[];
  slotMinutes: number;
  noticeMinutes: number;
  horizonDays: number;
  textConfirmations: boolean;
}

export type AppointmentStatus = 'requested' | 'confirmed' | 'declined' | 'cancelled' | 'done';

export interface Appointment {
  id: string;
  service: string;
  /** Local time, `YYYY-MM-DDTHH:MM`. */
  start: string;
  minutes: number;
  status: AppointmentStatus;
  name: string;
  phone: string;
  notes: string;
  source: string;
  requestId?: string;
  callId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface NewAppointment {
  service: string;
  date: string;
  time: string;
  minutes?: number;
  status?: AppointmentStatus;
  name?: string;
  phone?: string;
  notes?: string;
  source?: string;
}

/** How the calendar's sync with FormLogic stands. */
export interface CalendarSync {
  linked: boolean;
  /** `offline`: FormLogic could not be reached; `error`: it answered, and refused.
   *  Absent from an older desktop. */
  state?: 'unlinked' | 'waiting' | 'syncing' | 'synced' | 'offline' | 'busy' | 'error';
  /** When the last sync was tried. */
  at: string | null;
  /** When a sync last went through. */
  lastSuccessAt?: string | null;
  nextAttemptAt?: string | null;
  pulled: number;
  pushed: number;
  removed?: number;
  /** Changes made here that FormLogic has not had yet. */
  pending?: { creates: number; updates: number; deletes: number; total: number };
  error: string | null;
  /** Appointments FormLogic would not take, and why (sent again once changed here). */
  problems?: { id: string; message: string }[];
}

export const calendar = {
  /** The settings and the appointments from `from` to before `to`; `receptionistName` is the name the receptionist goes by (the one set, or Aokie's). */
  get: (from?: string, to?: string) =>
    request<{ available?: boolean; receptionistName?: string; settings: CalendarSettings; appointments: Appointment[]; now: string }>(
      `/api/calendar${from || to ? `?${new URLSearchParams({ ...(from ? { from } : {}), ...(to ? { to } : {}) })}` : ''}`,
    ),
  saveSettings: (s: CalendarSettings) => request<CalendarSettings>('/api/calendar/settings', { method: 'PUT', body: JSON.stringify(s) }),
  /** Free times from `from` for `days` days, for `service` (its length) or `minutes` long. */
  free: (from: string, days: number, service?: string, minutes?: number) =>
    request<{ minutes: number; days: { date: string; times: string[] }[] }>(
      `/api/calendar/free?${new URLSearchParams({ from, days: String(days), ...(service ? { service } : {}), ...(minutes ? { minutes: String(minutes) } : {}) })}`,
    ),
  create: (a: NewAppointment) => request<Appointment>('/api/calendar/appointments', { method: 'POST', body: JSON.stringify(a) }),
  update: (id: string, change: Partial<Pick<Appointment, 'status' | 'start' | 'minutes' | 'service' | 'name' | 'phone' | 'notes'>>) =>
    request<Appointment>(`/api/calendar/appointments/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(change) }),
  remove: (id: string) => request<void>(`/api/calendar/appointments/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  /** How the last FormLogic sync went (`linked` false when this desktop is not linked). */
  syncStatus: () => request<CalendarSync>('/api/calendar/sync'),
  syncNow: () => request<CalendarSync>('/api/calendar/sync', { method: 'POST' }),
  /** Text someone through the phone (Aokie). */
  text: (to: string, body: string) =>
    request<unknown>('/api/bridge/connectors/aokie/request', {
      method: 'POST',
      body: JSON.stringify({ command: 'sms.send', payload: { to, body }, idempotencyKey: `oaiy-calendar-${Date.now()}-${Math.random().toString(36).slice(2)}` }),
    }),
};

// ----- the voice calls are answered in -----

/** A voice: a clip of someone speaking, kept on this machine by name. */
export interface VoiceClip {
  name: string;
  file: string;
  bytes: number;
  /** What it says is written beside it (else OAIY's speech server hears it). */
  written: boolean;
}

/**
 * How a call is answered, kept with the voices (`/api/voice/settings`).
 * `greetingDelayMs`: how long the greeting waits after a call connects, 0 to
 * 5000 (sooner if the caller speaks first; calls the agent places wait for the
 * other person's hello whatever it is).
 */
export interface CallSettings {
  greetingDelayMs: number;
}

export const voices = {
  /** `greetingDelayMs` is there on a desktop that has `/api/voice/settings`. */
  list: () => request<{ voices: VoiceClip[]; chosen: string | null; greetingDelayMs?: number }>('/api/voice/voices'),
  choose: (voice: string) => request<{ chosen: string }>('/api/voice/voices/chosen', { method: 'PUT', body: JSON.stringify({ voice }) }),
  /** A 404 on an older desktop (see `optional`). */
  settings: () => request<CallSettings>('/api/voice/settings'),
  /** The desktop keeps it to 0 to 5000 and answers with what it kept; it applies from the next call. */
  saveSettings: (s: CallSettings) => request<CallSettings>('/api/voice/settings', { method: 'PUT', body: JSON.stringify(s) }),
  /** Keep a clip (MP3, WAV, ...) as the voice `name`; `words` is what it says, when known. */
  add: (name: string, file: File, words?: string, choose?: boolean) =>
    request<{ voice: VoiceClip; chosen: string | null }>(
      `/api/voice/voices?${new URLSearchParams({ name, file: file.name, ...(words?.trim() ? { words: words.trim() } : {}), ...(choose ? { choose: 'true' } : {}) })}`,
      { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file },
    ),
  remove: (name: string) => request<{ chosen: string | null }>(`/api/voice/voices/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  /** A line spoken in a voice, as audio. The speech server starts if it has to (up to a minute the first time). */
  hear: async (name: string, text?: string): Promise<Blob> => {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 90_000);
    try {
      const resp = await fetch(`${API_BASE}/api/voice/voices/${encodeURIComponent(name)}/try`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(text ? { text } : {}),
        signal: ac.signal,
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        throw new Error(body?.error?.message ?? `${resp.status}: ${resp.statusText}`);
      }
      return await resp.blob();
    } finally {
      clearTimeout(timer);
    }
  },
};

// ----- contacts: the people who ring and text -----

/** Who named a contact (or wrote a fact): the person (`owner`), or the receptionist on a call (`agent`). */
export type NamedBy = 'owner' | 'agent';

/** Something the receptionist remembered about a contact. */
export interface ContactFact {
  text: string;
  at: string;
  by: NamedBy;
}

export interface Contact {
  /** The last nine digits of their number: one person however the number is written. */
  key: string;
  /** The number last seen for them (a call's, or one given); empty when only the key is known. */
  number: string;
  /** One word is a whole name ("Lance"). */
  name: string;
  /** Who named them; `null` while they have no name. The receptionist never renames an `owner` name. */
  nameBy: NamedBy | null;
  /** The person's notes, which the receptionist reads on every call and text with them. */
  notes: string;
  facts: ContactFact[];
  createdAt: string;
  updatedAt: string;
}

/** A change to a contact: only the fields given change; a name given is the person's. */
export interface ContactChange {
  name?: string;
  notes?: string;
  number?: string;
}

/** One row of an import's file, as it goes. */
export interface ImportRow {
  /** Its line in the file. */
  row: number;
  action: 'add' | 'update' | 'unchanged' | 'skip';
  name: string;
  number: string;
  key?: string;
  /** Why it is skipped: no_number, not_a_phone_number, hidden, duplicate, notes_too_long. */
  reason?: string;
  why?: string;
  /** The person's own name, kept over the file's. */
  keptName?: string;
  notes: boolean;
}

/** What an import did, or (a preview) would do. */
export interface ImportReport {
  preview: boolean;
  country: string;
  headerRow: number | null;
  /** The headings read. */
  columns: string[];
  rows: number;
  added: number;
  updated: number;
  unchanged: number;
  skipped: number;
  reasons: Record<string, number>;
  skips: ImportRow[];
  sample: ImportRow[];
}

export interface ImportRequest {
  csv: string;
  /** Whose local numbers they are (`0491 570 006`): AU unless given. */
  country: string;
  /** Let the file's names replace the ones the person set. */
  replaceNames: boolean;
  preview: boolean;
}

const contactPath = (number: string) => `/api/contacts/${encodeURIComponent(number)}`;

export const contacts = {
  /** Everyone, by name (a 404 on a desktop older than contacts, see `isNotFound`). */
  list: () => request<{ contacts: Contact[]; total: number }>('/api/contacts'),
  get: (number: string) => request<Contact>(contactPath(number)),
  /** Made when there is none. */
  save: (number: string, change: ContactChange) => request<Contact>(contactPath(number), { method: 'PUT', body: JSON.stringify(change) }),
  remove: (number: string) => request<void>(contactPath(number), { method: 'DELETE' }),
  /** Forget a fact by its place; refused (409) when the fact there no longer says `text`. */
  forgetFact: (number: string, index: number, text: string) =>
    request<{ contact: Contact; forgotten: ContactFact }>(`${contactPath(number)}/facts/${index}?${new URLSearchParams({ text })}`, { method: 'DELETE' }),
  importCsv: (body: ImportRequest) => request<ImportReport>('/api/contacts/import', { method: 'POST', body: JSON.stringify(body) }),
  /** The CSV file as the desktop wrote it: its bytes, byte-order mark and all (reading it as text would drop the mark). */
  exportCsv: async (): Promise<Blob> => {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 30_000);
    try {
      const resp = await fetch(`${API_BASE}/api/contacts/export.csv`, { signal: ac.signal });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        throw new Error(`${resp.status}: ${body?.error?.message ?? resp.statusText}`);
      }
      return new Blob([await resp.arrayBuffer()], { type: 'text/csv;charset=utf-8' });
    } finally {
      clearTimeout(timer);
    }
  },
};

// ----- messages callers leave for the owner -----

export type MessageState = 'new' | 'seen' | 'handled';

/** A message the receptionist took because the owner could not be reached. */
export interface CallerMessage {
  id: string;
  /** When it was taken (RFC 3339). */
  at: string;
  callId: string;
  /** The number the call came from ('' for a hidden number). */
  from: string;
  /** The name the caller gave ('' when none). */
  name: string;
  /** Where to ring them back. */
  callback: string;
  message: string;
  urgency: 'normal' | 'urgent';
  wantsCallback: boolean;
  state: MessageState;
  seenAt: string | null;
  handledAt: string | null;
  handledBy: string | null;
}

const messagePath = (id: string) => `/api/messages/${encodeURIComponent(id)}`;

export const messages = {
  /** Newest first: all, or those of one `state`, or those `q` finds in names, numbers and words. */
  list: (state?: MessageState, q = '') => {
    const params = new URLSearchParams({ ...(state ? { state } : {}), ...(q.trim() ? { q: q.trim() } : {}) }).toString();
    return request<{ messages: CallerMessage[]; total: number; unread: number; notice?: string | null }>(`/api/messages${params ? `?${params}` : ''}`);
  },
  /** Mark one `new`, `seen` or `handled`. */
  mark: (id: string, state: MessageState) => request<CallerMessage>(messagePath(id), { method: 'PATCH', body: JSON.stringify({ state }) }),
  remove: (id: string) => request<void>(messagePath(id), { method: 'DELETE' }),
};

// ----- putting callers through to the owner (settings, and the ring that is going now) -----

export type Initiative = 'on_request' | 'on_request_or_urgent';
export type PhoneRing = 'when_away' | 'always' | 'never';
export type DesktopRing = 'auto' | 'always' | 'never';
export type AwayMode = 'auto' | 'on' | 'off';

export interface QuietHours {
  enabled: boolean;
  start: string;
  end: string;
  /** One bit a day the window starts on, Sunday (bit 0) to Saturday (bit 6). */
  days: number;
  allowUrgent: boolean;
  allowVip: boolean;
}

/** The owner's settings for transferring calls to them and taking messages (`<data>/ring.json`). */
export interface RingSettings {
  /** "Transfer calls to me". */
  enabled: boolean;
  /** "Take messages": on whenever transfers are. */
  takeMessages: boolean;
  initiative: Initiative;
  urgentPhrases: string[];
  ringSeconds: number;
  phoneRing: PhoneRing;
  desktopRing: DesktopRing;
  away: AwayMode;
  awayUntil: number | null;
  desktopActiveSeconds: number;
  quietHours: QuietHours;
  vipNumbers: string[];
  limits: { perCall: number; gapSeconds: number; perCallerHour: number; globalHour: number };
  windowsCompanions: string[];
  excludedDevices: string[];
}

export interface RingFeatures {
  transfer: boolean;
  messages: boolean;
}

/** What the owner may do with a ring going now: decline it and have the receptionist take a message. (Taking the call is the Companion's.) */
export type RingAction = 'decline' | 'message';

/** One caller the receptionist is trying to reach the owner for. */
export interface ActiveRing {
  /** The transfer request's id. */
  id: string;
  callId: string;
  callerName: string;
  callerNumber: string;
  /** What the caller last said (their own words, as the desktop heard them). */
  said: string[];
  /** When it began and when it stops ringing (Unix milliseconds). */
  startedAt: number;
  expiresAt: number;
  /** The desktop's clock now, in the same unit: the countdown does not trust the window's clock. */
  now: number;
  /** Who else is rung: a phone or Companion (their names), or nobody but this computer. */
  devices: string[];
  /** The owner declined and the phone is being asked to withdraw the request: the dialog waits for its answer. */
  stopping: boolean;
  /** The last thing the owner asked of it here, and what came of it. */
  note: string;
}

/** Somebody asked for the owner and nobody could be rung, for want of a device set up to take a transfer. */
export interface RingNotice {
  id: string;
  callId: string;
  callerName: string;
  callerNumber: string;
  /** Unix milliseconds, by the desktop's clock. */
  at: number;
  text: string;
}

export const ring = {
  settings: () => request<{ settings: RingSettings; features: RingFeatures }>('/api/ring/settings'),
  /** Change some settings (any of them; `quietHours` and `limits` member by member). */
  save: (change: Partial<Omit<RingSettings, 'quietHours' | 'limits'>> & { quietHours?: Partial<QuietHours>; limits?: Partial<RingSettings['limits']> }) =>
    request<{ settings: RingSettings; features: RingFeatures }>('/api/ring/settings', { method: 'PUT', body: JSON.stringify(change) }),
  /** The rings going now (none most of the time), and the callers who asked for the owner when nothing could be rung. */
  active: async () => {
    const r = await request<{ rings?: ActiveRing[]; notices?: RingNotice[] }>('/api/ring/active');
    return { rings: r.rings ?? [], notices: r.notices ?? [] };
  },
  /** The owner has read a notice. */
  dismissNotice: (id: string) => request<{ ok: boolean }>(`/api/ring/notices/${encodeURIComponent(id)}/dismiss`, { method: 'POST' }),
  /** Decline one: the phone is asked to withdraw the request, and the caller is offered a message. */
  respond: (id: string, action: RingAction) =>
    request<{ ok: boolean; note: string }>(`/api/ring/active/${encodeURIComponent(id)}/respond`, { method: 'POST', body: JSON.stringify({ action }) }),
};

// ----- the engines and the phone, for Overview -----

export interface EnginesStatus {
  running: boolean;
  uiUrl?: string;
  llm?: { state: string | null; resident: string | null; models: string[] | null; loadSeconds: number | null };
  gpus?: Array<{ index: number; name: string; memory_used_mb: number; memory_total_mb: number }> | null;
}

/** A download from the engines' catalog, as it stands. */
export interface EngineDownload {
  id: string;
  /** `queued`, `downloading`, `adding`, `done`, `paused`, `cancelled`, `failed`. */
  status: string;
  done: number;
  total: number;
  file?: string;
  filesDone?: number;
  filesTotal?: number;
  /** MB/s. */
  speed?: number;
  error?: string | null;
}

/** One model of the engines' catalog. */
export interface EngineCatalogModel {
  id: string;
  group: string;
  name: string;
  about?: string;
  license?: string;
  sizeGb?: number;
  vramGb?: number;
  recommended: boolean;
  needs: string[];
  installed: boolean;
  partial: boolean;
  download: EngineDownload | null;
}

/** The engines' catalog, and the model chosen in Engines for each group (`null`: none). */
export interface EngineCatalog {
  running: boolean;
  error?: string;
  dir?: string;
  free?: number | null;
  groups?: Array<{ id: string; name: string; about?: string }>;
  models?: EngineCatalogModel[];
  defaults?: Record<string, string | null>;
}

export const engines = {
  status: () => request<EnginesStatus>('/api/engines'),
  /** The catalog and the models chosen in Engines. */
  catalog: () => request<EngineCatalog>('/api/engines/catalog'),
  downloads: () => request<{ running: boolean; downloads: EngineDownload[]; error?: string }>('/api/engines/downloads'),
  /** Download a catalog model into the engines' own folder. */
  download: (id: string) =>
    request<{ running: boolean; downloads: EngineDownload[] }>('/api/engines/downloads', { method: 'POST', body: JSON.stringify({ id }) }),
};

// ----- setup (the wizard's record, on the desktop) -----

export interface FirstRunState {
  finished: boolean;
  /** The step it is on. */
  position?: string | null;
  skipped: string[];
  chosenPlugins: string[];
  /** Why a desktop already in use was recorded as set up. */
  migrated?: string;
}

export interface PluginSetupState {
  /** The setup version last finished (0: never). */
  version: number;
  done: string[];
  skipped: string[];
  permissionsAccepted?: string[];
}

export interface SetupState {
  firstRun: FirstRunState;
  plugins: Record<string, PluginSetupState>;
}

/** A plugin's setup as the desktop reads it, its capabilities, and its record. */
export interface SetupPluginDetail {
  pluginId: string;
  name: string;
  setup: SetupDeclJson | null;
  /** Its capabilities, wildcards expanded. */
  capabilities: string[];
  legacyCapabilities?: Array<[string, string]>;
  unknownCapabilities?: string[];
  /** The person accepted everything it asks for now. */
  permissionsAccepted: boolean;
  needsSetup: boolean;
  state: PluginSetupState;
}

export interface CheckOutcome {
  passed: boolean;
  detail: string;
}

/** A plugin OAIY knows how to install (the bundled catalog). */
export interface CatalogPlugin {
  id: string;
  name: string;
  plugin?: string;
  publisher?: string;
  description?: string;
  provides?: string[];
  needs?: string;
  /** Installed from a folder on this machine; `path` is where one was found (null: the person chooses). */
  source: { kind: 'folder'; note?: string; path: string | null };
  installed: boolean;
  installedVersion: string | null;
}

export type StepMark = 'done' | 'skipped' | 'todo';

export const setup = {
  get: () => request<SetupState>('/api/setup'),
  /** Replace the first-run part of the record. */
  putFirstRun: (firstRun: Omit<FirstRunState, 'migrated'>) =>
    request<SetupState>('/api/setup', { method: 'PUT', body: JSON.stringify({ firstRun }) }),
  catalog: () => request<{ plugins: CatalogPlugin[] }>('/api/setup/catalog'),
  plugin: (id: string) => request<SetupPluginDetail>(`/api/setup/plugins/${encodeURIComponent(id)}`),
  /** Record a step done, skipped, or neither. `permissions` done records what is accepted now. */
  markStep: (id: string, step: string, status: StepMark = 'done') =>
    request<SetupState>(`/api/setup/plugins/${encodeURIComponent(id)}/steps/${encodeURIComponent(step)}`, {
      method: 'POST',
      body: JSON.stringify({ status }),
    }),
  finish: (id: string) => request<SetupState>(`/api/setup/plugins/${encodeURIComponent(id)}/finish`, { method: 'POST' }),
  /** Run a step's `done` (or `when`) check on the desktop. */
  check: (id: string, step: string, which: 'done' | 'when' = 'done') =>
    request<CheckOutcome>(
      `/api/setup/plugins/${encodeURIComponent(id)}/check/${encodeURIComponent(step)}${which === 'when' ? '?check=when' : ''}`,
      { method: 'POST' },
    ),
};

/**
 * What the dashboard may ask the Agent (its page in OAIY's window) to do, by
 * name only (embed.rs `AGENT_INTENTS`): answer calls and texts (its settings
 * live in its own storage), or set up the rest of OAIY with the person, in a
 * "Set up OAIY" conversation.
 */
export type AgentIntent = 'answerWithOaiy' | 'setupWithAgent';

/** Ask the Agent to do something only it can. Fails when the Agent's page is not made yet. */
export function agentIntent(intent: AgentIntent): Promise<void> {
  return tauriInvoke<void>('agent_intent', { intent });
}

// ----- the Agent's access to OAIY, what it changed, and its model (CONTROL_API.md §1, §2) -----

/** A 404 from `request()`: a route this desktop does not have (an older desktop). */
export function isNotFound(e: unknown): boolean {
  return e instanceof Error && /^404: /.test(e.message);
}

/** A route this desktop may not have yet: `null` on a 404, its answer otherwise (any other failure still throws). */
export async function optional<T>(pending: Promise<T>): Promise<T | null> {
  try {
    return await pending;
  } catch (e) {
    if (isNotFound(e)) return null;
    throw e;
  }
}

/** `<data>/control.json`: while `agentMayChange` is false the Agent's change tools refuse, and its read tools still work. */
export interface ControlSettings {
  agentMayChange: boolean;
}

/** Who asked, as the Agent app says with `X-OAIY-Session`. */
export type AgentSession = 'project' | 'setup' | 'runner' | 'call' | 'sms' | 'task';

/** One change the Agent made through the MCP API (`<data>/control-log.jsonl`), secrets redacted. */
export interface ControlLogEntry {
  /** When: an ISO time, or seconds or milliseconds since 1970. */
  at: string | number;
  /** The MCP tool, e.g. `plugin_install`. */
  tool: string;
  args?: unknown;
  session?: AgentSession | string | null;
  ok: boolean;
  /** A sentence about what it did, when the tool wrote one. */
  summary?: string | null;
}

/** The log's answer, as a list or in an object that holds one (the route's wrapper is not pinned). */
export function logEntries(body: unknown): ControlLogEntry[] {
  const list = Array.isArray(body)
    ? body
    : body && typeof body === 'object'
      ? ((['entries', 'log', 'items', 'changes'] as const).map((k) => (body as Record<string, unknown>)[k]).find(Array.isArray) ?? [])
      : [];
  return (list as unknown[]).filter(
    (e): e is ControlLogEntry => !!e && typeof e === 'object' && typeof (e as ControlLogEntry).tool === 'string',
  );
}

export const control = {
  /** `null`: this desktop has no control API yet (the switch shows as on, and cannot be changed). */
  settings: () => optional(request<ControlSettings>('/api/control/settings')),
  setSettings: async (s: ControlSettings) =>
    (await request<ControlSettings | undefined>('/api/control/settings', { method: 'PUT', body: JSON.stringify(s) })) ?? s,
  /** Newest first. `null`: this desktop keeps no log yet. */
  log: async (limit = 100): Promise<ControlLogEntry[] | null> => {
    const body = await optional(request<unknown>(`/api/control/log?limit=${limit}`));
    return body === null ? null : logEntries(body);
  },
};

/** What the Agent thinks with: the engine's language model, or ChatGPT (Codex's default model when `model` is left out). */
export type AgentModelSource = 'engine' | 'chatgpt';
export interface AgentModelPreference {
  source: AgentModelSource;
  model?: string | null;
}
export interface AgentPreferences {
  model: AgentModelPreference;
}

export const agentPreferences = {
  /** `null`: this desktop does not keep the Agent's model yet. */
  get: () => optional(request<AgentPreferences>('/api/agent/preferences')),
  set: async (prefs: AgentPreferences) =>
    (await request<AgentPreferences | undefined>('/api/agent/preferences', { method: 'PUT', body: JSON.stringify(prefs) })) ?? prefs,
};

/** One GPU, as Engines reports it. */
export interface GpuMemory {
  name: string;
  totalGb: number;
  freeGb?: number | null;
}

/** The catalog's recommended language model, as the recommendation names it (the catalog's own entry). */
export type SuggestedModel = Partial<EngineCatalogModel> & { id: string; name?: string };

/** Local or ChatGPT for the Agent, from this computer's hardware and what Engines has chosen. */
export interface EngineRecommendation {
  recommend: AgentModelSource;
  local: {
    ok: boolean;
    reason?: string | null;
    gpus?: GpuMemory[];
    /** The language model chosen in Engines, or null. */
    chosen?: string | null;
    suggested?: SuggestedModel | null;
  };
  chatgpt: { signedIn: boolean };
}

/** `null`: this desktop does not recommend yet (the wizard falls back to "local if Engines has a model chosen"). */
export function engineRecommendation(): Promise<EngineRecommendation | null> {
  return optional(request<EngineRecommendation>('/api/engines/recommendation'));
}

/** The ChatGPT connector's provider id (ai/codex.rs `CODEX_PROVIDER_ID`): a route, not a model. */
export const CODEX_PROVIDER_ID = 'openai-codex-agent';

/** One model of Codex's catalogue: `id` is what to send back as the model. */
export interface CodexModel {
  id: string;
  displayName?: string | null;
  isDefault?: boolean;
}

/** The models this ChatGPT account can use (fails while signed out). */
export async function codexModels(): Promise<CodexModel[]> {
  const r = await request<{ data?: CodexModel[] }>(`/api/ai/providers/${CODEX_PROVIDER_ID}/v1/models`);
  return (r?.data ?? []).filter((m) => !!m && typeof m.id === 'string' && m.id !== '');
}

export const phone = {
  /** Whether Aokie has the phone connected. */
  status: async () => {
    const r = await request<{ result?: { data?: { connected?: boolean } } }>('/api/bridge/connectors/aokie/request', {
      method: 'POST',
      body: JSON.stringify({ command: 'phone.status', payload: {}, idempotencyKey: `oaiy-overview-${Date.now()}` }),
    });
    return { connected: !!r?.result?.data?.connected };
  },
  /** The calls live now. */
  calls: async () => (await request<{ calls: unknown[] }>('/api/voice/calls')).calls ?? [],
};
