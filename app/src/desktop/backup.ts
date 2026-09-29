/**
 * The Agent's part of OAIY Desktop's backup (Settings, "Backup and restore").
 *
 * The Agent's conversations and projects live in this page's own storage (OPFS
 * and IndexedDB), which only this page can read, so the desktop asks the page
 * for them and the page hands them over as a ZIP, in parts, over the desktop's
 * local API. Nothing here writes a backup file or knows a passphrase: the
 * desktop puts what it is sent into the one encrypted file.
 *
 *  - EXPORT: the desktop evaluates `window.__oaiyBackup.export({ id, token, includeKeys })`
 *    into the page. The page builds a ZIP with fflate's streaming `Zip` (never the
 *    whole archive in memory) and posts it in parts of at most 4 MiB to
 *    `/api/backup/agent/<id>/part?seq=<n>`, then reports to `.../done`.
 *  - IMPORT: a restore is applied at the page's next start, before anything is
 *    opened. The page asks `/api/backup/agent-import` whether one waits, downloads
 *    it in parts, checks its SHA-256, takes an undo copy of what it holds now
 *    (uploaded to the desktop, which keeps it for "Undo the last restore"), and
 *    only then writes. It overwrites what the backup holds and deletes nothing.
 *
 * What is never exported: an incognito project (it is designed never to leave its
 * folder), the paired desktop's token, the sealed encryption key of the API keys,
 * and, unless the person chose "Include my API provider keys", the API keys.
 *
 * The logic works on `AgentStorage` and an injected `fetch`, so it is tested without a
 * browser; `opfsStorage()` is the real one.
 */
import { Zip, ZipDeflate, strToU8, unzipSync } from 'fflate';
import type { ProviderConfig } from '../agent/providers/types';
import type { MediaSettings } from '../agent/media';
import type { NetGateSettings } from '../gate/netgate';
import { DEFAULT_AGENT_SETTINGS, DEFAULT_MESSAGE_SETTINGS, loadSettings, saveAgentSettings, saveGate, saveLastKeptProject, saveLastProject, saveMedia, saveMessages, saveProviders, type AgentSettings, type MessageSettings, type Settings } from '../settings';

/** What is sent to the desktop in one request. */
export const PART_BYTES = 4 * 1024 * 1024;
/** One file over this is left out, and named in the warnings. */
export const MAX_FILE_BYTES = 64 * 1024 * 1024;
/** Once this much is in the archive, no more files are added. */
export const MAX_TOTAL_BYTES = 512 * 1024 * 1024;
/** A restore refuses an archive of more entries than this. */
export const MAX_ENTRIES = 200_000;
/** ...or one that unpacks to more than this. */
export const MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024;
/** A single entry of an archive being restored over this is left out. */
export const MAX_ENTRY_BYTES = 256 * 1024 * 1024;
/** Unpacked together: keeps a restore's memory to about this and one entry. */
const BATCH_BYTES = 64 * 1024 * 1024;
const BATCH_ENTRIES = 2000;
/** Input is fed to the compressor in pieces of this, so the output can be sent as it comes. */
const SLICE_BYTES = 1024 * 1024;

export interface Limits {
  part: number;
  file: number;
  total: number;
  entries: number;
  unpacked: number;
  entry: number;
}

export const DEFAULT_LIMITS: Limits = { part: PART_BYTES, file: MAX_FILE_BYTES, total: MAX_TOTAL_BYTES, entries: MAX_ENTRIES, unpacked: MAX_UNPACKED_BYTES, entry: MAX_ENTRY_BYTES };

/** The warning the dashboard shows when the archive was cut short by the total limit. */
const TOTAL_WARNING = (mib: number, left: number) => `The Agent's storage is bigger than ${mib} MiB, so ${left} more file${left === 1 ? ' was' : 's were'} not included.`;

// ---------------------------------------------------------------------------
// Storage

/** One file in the Agent's storage, read only when it is wanted. */
export interface StoredFile {
  /** Where it is, from the storage's root (`projects/<id>/chat.json`, `front-desk/brief.md`), forward slashes. */
  path: string;
  size: number;
  read(): Promise<Uint8Array>;
}

/** The part of the settings a backup carries. */
export type BackupSettings = Pick<Settings, 'providers' | 'activeProviderId' | 'gate' | 'lastProjectId' | 'lastKeptProjectId' | 'agent' | 'media' | 'messages'>;

/** What the Agent keeps in this browser (see vfs/projects.ts and settings.ts). */
export interface AgentStorage {
  /** The ids of the projects (the folders under `projects/`). */
  projectIds(): Promise<string[]>;
  /** Every file under `root` (`projects/<id>` or `front-desk`); none when the folder is not there. */
  walk(root: string): AsyncIterable<StoredFile>;
  /** A file's bytes, or null when it is not there. */
  readFile(path: string): Promise<Uint8Array | null>;
  /** A file written, its folders made, whatever stood in the way replaced. */
  writeFile(path: string, data: Uint8Array): Promise<void>;
  /** The settings as the page loads them (the API keys already opened). */
  readSettings(): Promise<Settings>;
  writeSettings(settings: BackupSettings): Promise<void>;
}

/** The desktop, as the page knows it inside OAIY's window. */
export interface DesktopRef {
  origin: string;
  token: string;
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

// ---------------------------------------------------------------------------
// Reports

export interface ExportCounts {
  projects: number;
  incognitoSkipped: number;
  conversations: number;
  files: number;
  bytes: number;
}

/** What the page tells the desktop when it has sent (or given up sending) the archive. */
export interface DoneBody {
  ok: boolean;
  error?: string;
  parts: number;
  counts: ExportCounts;
  warnings: string[];
}

export interface ExportReport extends DoneBody {
  /** Files that were left out (for the undo copy: a restore must not replace one of them). Never sent. */
  skipped: string[];
}

export interface ExportOptions {
  /** Include the API keys (the person chose it, or this is the undo copy). */
  includeKeys: boolean;
  /** Write out what is pending first (the open project, the conversation). */
  flush?: () => Promise<void>;
  now?: () => Date;
  limits?: Partial<Limits>;
}

/** Where the parts go. */
export interface PartTarget {
  part(seq: number, bytes: Uint8Array): Promise<void>;
  done(body: DoneBody): Promise<void>;
}

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------------------
// Export

/** Small pieces gathered into parts of at most `size` bytes, sent in order. */
class PartSink {
  private chunks: Uint8Array[] = [];
  private length = 0;
  private ready: Uint8Array[] = [];
  /** How many parts have been sent. */
  parts = 0;

  constructor(private readonly target: PartTarget, private readonly size: number) {}

  write(chunk: Uint8Array): void {
    if (!chunk.length) return;
    this.chunks.push(chunk);
    this.length += chunk.length;
    if (this.length < this.size) return;
    const all = new Uint8Array(this.length);
    let at = 0;
    for (const c of this.chunks) {
      all.set(c, at);
      at += c.length;
    }
    let from = 0;
    while (this.length - from >= this.size) {
      this.ready.push(all.subarray(from, from + this.size));
      from += this.size;
    }
    this.chunks = from < this.length ? [all.subarray(from)] : [];
    this.length -= from;
  }

  /** Send the parts that are full. */
  async drain(): Promise<void> {
    while (this.ready.length) await this.send(this.ready.shift()!);
  }

  /** Send everything that is left. */
  async finish(): Promise<void> {
    await this.drain();
    if (!this.length) return;
    const rest = new Uint8Array(this.length);
    let at = 0;
    for (const c of this.chunks) {
      rest.set(c, at);
      at += c.length;
    }
    this.chunks = [];
    this.length = 0;
    await this.send(rest);
  }

  private async send(bytes: Uint8Array): Promise<void> {
    await this.target.part(this.parts, bytes);
    this.parts++;
  }
}

/** A conversation: a project's chat, or one of its texts and calls (`sessions/<thread>.json`, not the list). */
const CONVERSATION = /^(?:projects\/[^/]+|front-desk)\/(?:chat\.json|sessions\/(?!index\.json$)[^/]+\.json)$/;

async function isIncognito(storage: AgentStorage, projectPath: string): Promise<boolean> {
  try {
    const bytes = await storage.readFile(`${projectPath}/project.json`);
    return bytes ? incognitoFlag(bytes) : false;
  } catch {
    return false;
  }
}

function incognitoFlag(bytes: Uint8Array): boolean {
  try {
    return (JSON.parse(new TextDecoder().decode(bytes)) as { incognito?: unknown })?.incognito === true;
  } catch {
    return false;
  }
}

/**
 * Build the archive of the Agent's storage and send it, in parts. Never throws: a failure is
 * the returned report (and the `done` sent to the target says so), and what could not be read
 * is named in `warnings`, not fatal.
 */
export async function exportAgentStorage(storage: AgentStorage, target: PartTarget, options: ExportOptions): Promise<ExportReport> {
  const limits = { ...DEFAULT_LIMITS, ...options.limits };
  const warnings: string[] = [];
  const skipped: string[] = [];
  const counts: ExportCounts = { projects: 0, incognitoSkipped: 0, conversations: 0, files: 0, bytes: 0 };
  const sink = new PartSink(target, limits.part);
  let failure: string | null = null;
  try {
    if (options.flush) {
      try {
        await options.flush();
      } catch {
        warnings.push('The Agent could not save its latest changes before the backup, so they may be missing from it.');
      }
    }
    let zipError: unknown = null;
    const zip = new Zip((err, chunk) => {
      if (err) zipError = err;
      else sink.write(chunk);
    });
    const add = async (name: string, bytes: Uint8Array): Promise<void> => {
      const entry = new ZipDeflate(name, { level: 6 });
      zip.add(entry);
      if (!bytes.length) entry.push(bytes, true);
      for (let at = 0; at < bytes.length; at += SLICE_BYTES) {
        entry.push(bytes.subarray(at, at + SLICE_BYTES), at + SLICE_BYTES >= bytes.length);
        if (zipError) throw zipError;
        await sink.drain();
      }
      if (zipError) throw zipError;
      await sink.drain();
    };
    const now = options.now ?? (() => new Date());
    await add('agent-manifest.json', strToU8(JSON.stringify({ v: 1, kind: 'oaiy-agent-storage', createdAt: now().toISOString(), includesKeys: options.includeKeys })));

    let full = false;
    let notAdded = 0;
    const addTree = async (root: string): Promise<void> => {
      for await (const file of storage.walk(root)) {
        if (file.path.endsWith('.crswap')) continue;
        if (full || counts.bytes >= limits.total) {
          full = true;
          notAdded++;
          skipped.push(file.path);
          continue;
        }
        if (file.size > limits.file) {
          warnings.push(`${file.path} was left out: it is bigger than ${Math.round(limits.file / 2 ** 20)} MiB.`);
          skipped.push(file.path);
          continue;
        }
        let bytes: Uint8Array;
        try {
          bytes = await file.read();
        } catch (e) {
          warnings.push(`${file.path} could not be read (${message(e)}) and was left out.`);
          skipped.push(file.path);
          continue;
        }
        if (bytes.length > limits.file) {
          warnings.push(`${file.path} was left out: it is bigger than ${Math.round(limits.file / 2 ** 20)} MiB.`);
          skipped.push(file.path);
          continue;
        }
        await add(`opfs/${file.path}`, bytes);
        counts.files++;
        counts.bytes += bytes.length;
        if (CONVERSATION.test(file.path)) counts.conversations++;
      }
    };

    const incognito = new Set<string>();
    for (const id of await storage.projectIds()) {
      if (await isIncognito(storage, `projects/${id}`)) {
        incognito.add(id);
        counts.incognitoSkipped++;
        continue;
      }
      counts.projects++;
      await addTree(`projects/${id}`);
    }
    await addTree('front-desk');
    if (full) warnings.push(TOTAL_WARNING(Math.round(limits.total / 2 ** 20), notAdded));

    try {
      const s = await storage.readSettings();
      const inIncognito = (id: string | null) => (id !== null && incognito.has(id) ? null : id);
      const keys = options.includeKeys;
      const settings: BackupSettings = {
        providers: s.providers.map((p) => ({ ...p, apiKey: keys ? p.apiKey : '' })),
        activeProviderId: s.activeProviderId,
        gate: s.gate,
        lastProjectId: inIncognito(s.lastProjectId),
        lastKeptProjectId: inIncognito(s.lastKeptProjectId),
        agent: s.agent,
        media: { ...s.media, apiKey: keys ? s.media.apiKey : '' },
        messages: s.messages,
      };
      await add('idb/settings.json', strToU8(JSON.stringify(settings)));
    } catch (e) {
      warnings.push(`The Agent's settings could not be read (${message(e)}) and were left out.`);
    }
    zip.end();
    if (zipError) throw zipError;
    await sink.finish();
  } catch (e) {
    failure = message(e);
  }
  const body: DoneBody = { ok: failure === null, ...(failure === null ? {} : { error: failure }), parts: sink.parts, counts, warnings };
  try {
    await target.done(body);
  } catch (e) {
    if (body.ok) {
      body.ok = false;
      body.error = `The desktop could not be told the archive was complete (${message(e)}).`;
    }
  }
  return { ...body, skipped };
}

// ---------------------------------------------------------------------------
// Talking to the desktop

function bearer(desktop: DesktopRef, session: string): Record<string, string> {
  return { Authorization: `Bearer ${desktop.token}`, 'X-Backup-Token': session };
}

function base(desktop: DesktopRef): string {
  return desktop.origin.replace(/\/+$/, '');
}

function timeout(ms: number): AbortSignal | undefined {
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(ms) : undefined;
}

/** Parts and the final report posted to `<prefix>-part` and `<prefix>-done` style URLs. */
function uploadTo(desktop: DesktopRef, session: string, partUrl: string, doneUrl: string, fetchImpl: FetchLike): PartTarget {
  const headers = bearer(desktop, session);
  const post = async (url: string, init: RequestInit): Promise<void> => {
    const response = await fetchImpl(url, { method: 'POST', ...init, headers: { ...headers, ...(init.headers as Record<string, string> | undefined) }, signal: timeout(60_000) });
    if (!response.ok) throw new Error(`the desktop answered ${response.status}`);
  };
  return {
    part: (seq, bytes) => post(`${partUrl}?seq=${seq}`, { headers: { 'Content-Type': 'application/octet-stream' }, body: bytes as unknown as BodyInit }),
    done: (body) => post(doneUrl, { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  };
}

/** A session id or token as the desktop makes them: nothing that could change a URL or a header. */
const SAFE_ID = /^[A-Za-z0-9_-]{1,128}$/;

export interface ExportJob {
  id: string;
  token: string;
  includeKeys: boolean;
}

function validJob(job: unknown): job is ExportJob {
  const j = job as Partial<ExportJob> | null;
  return !!j && typeof j.id === 'string' && SAFE_ID.test(j.id) && typeof j.token === 'string' && SAFE_ID.test(j.token) && typeof j.includeKeys === 'boolean';
}

/** The window the desktop calls into. */
export interface BackupWindow {
  __oaiyBackup?: { export: (job: unknown) => Promise<void> };
}

export interface BackupHooks {
  desktop: DesktopRef;
  storage: AgentStorage;
  /** Write out what is pending first (the open project and its conversation). */
  flush?: () => Promise<void>;
  fetch?: FetchLike;
  limits?: Partial<Limits>;
}

/**
 * Answer the desktop's request for a backup: `window.__oaiyBackup.export(job)`. One at a time.
 * The promise it returns never rejects (the desktop evaluates it and does not look).
 * Returns a function that takes it away again.
 */
export function installBackupHooks(hooks: BackupHooks, w: BackupWindow = window as unknown as BackupWindow): () => void {
  const fetchImpl: FetchLike = hooks.fetch ?? ((input, init) => fetch(input, init));
  let running: Promise<void> | null = null;
  const run = (job: unknown): Promise<void> => {
    if (!validJob(job) || running) return Promise.resolve();
    const { id, token, includeKeys } = job;
    const url = `${base(hooks.desktop)}/api/backup/agent/${encodeURIComponent(id)}`;
    const target = uploadTo(hooks.desktop, token, `${url}/part`, `${url}/done`, fetchImpl);
    running = exportAgentStorage(hooks.storage, target, { includeKeys, flush: hooks.flush, limits: hooks.limits })
      .then(() => undefined)
      .catch(() => undefined)
      .finally(() => {
        running = null;
      });
    return running;
  };
  const hook = { export: run };
  w.__oaiyBackup = hook;
  return () => {
    if (w.__oaiyBackup === hook) delete w.__oaiyBackup;
  };
}

// ---------------------------------------------------------------------------
// Import

export interface ImportOutcome {
  ok: boolean;
  error?: string;
  applied: { projects: number; files: number; settings: boolean };
  warnings: string[];
}

interface PendingImport {
  id: string;
  token: string;
  kind: 'restore' | 'undo';
  size: number;
  sha256: string;
  parts: number;
  partSize: number;
}

export interface RestoreOptions {
  fetch?: FetchLike;
  limits?: Partial<Limits>;
  /** Tries to reach the desktop again after a refused connection (it may still be starting), and how long to wait. */
  retries?: number;
  sleep?: (ms: number) => Promise<void>;
}

export type EntryCheck = { kind: 'file'; where: 'manifest' | 'settings' | 'opfs'; path: string } | { kind: 'dir' } | { kind: 'skip'; why: string };

/**
 * Whether an archive entry may be restored, and where it goes (a path from the storage's root).
 * Only the Agent's own storage: `agent-manifest.json`, `idb/settings.json`, `opfs/projects/<id>/...`
 * and `opfs/front-desk/...`; and only plain names inside it.
 */
export function checkEntry(name: string): EntryCheck {
  if (name.endsWith('/')) return { kind: 'dir' };
  if (!name || name.length > 1024) return { kind: 'skip', why: 'its name is empty or too long' };
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(name)) return { kind: 'skip', why: 'its name has a control character' };
  if (name.includes('\\')) return { kind: 'skip', why: 'its name has a backslash' };
  if (name.startsWith('/')) return { kind: 'skip', why: 'its name is an absolute path' };
  if (/^[A-Za-z]:/.test(name)) return { kind: 'skip', why: 'its name starts with a drive letter' };
  const segments = name.split('/');
  if (segments.some((s) => s === '' || s === '.' || s === '..')) return { kind: 'skip', why: 'its name goes out of the folder it is in' };
  if (name === 'agent-manifest.json') return { kind: 'file', where: 'manifest', path: name };
  if (name === 'idb/settings.json') return { kind: 'file', where: 'settings', path: name };
  if (segments[0] === 'opfs' && segments[1] === 'projects' && segments.length >= 4) return { kind: 'file', where: 'opfs', path: segments.slice(1).join('/') };
  if (segments[0] === 'opfs' && segments[1] === 'front-desk' && segments.length >= 3) return { kind: 'file', where: 'opfs', path: segments.slice(1).join('/') };
  return { kind: 'skip', why: 'it is not part of the Agent’s storage' };
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

function pendingFrom(value: unknown, limits: Limits): PendingImport | null {
  const v = value as Partial<PendingImport> | null;
  if (!v || typeof v !== 'object') return null;
  const whole = (n: unknown, max: number) => typeof n === 'number' && Number.isInteger(n) && n >= 0 && n <= max;
  if (typeof v.id !== 'string' || !SAFE_ID.test(v.id) || typeof v.token !== 'string' || !SAFE_ID.test(v.token)) return null;
  if (v.kind !== 'restore' && v.kind !== 'undo') return null;
  if (!whole(v.size, limits.unpacked) || !whole(v.parts, 1_000_000) || !whole(v.partSize, 1024 * 1024 * 1024)) return null;
  if (typeof v.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(v.sha256)) return null;
  return { id: v.id, token: v.token, kind: v.kind, size: v.size as number, sha256: v.sha256, parts: v.parts as number, partSize: v.partSize as number };
}

/** What a restore does with the settings: the backup's, but never an empty key over a key that is kept now. */
export function mergeSettings(current: Settings, backup: Partial<BackupSettings>, blocked: ReadonlySet<string>): BackupSettings {
  const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
  const providers: ProviderConfig[] = current.providers.map((p) => ({ ...p }));
  for (const incoming of Array.isArray(backup.providers) ? backup.providers : []) {
    if (!object(incoming) || typeof incoming.id !== 'string') continue;
    const at = providers.findIndex((p) => p.id === incoming.id);
    const kept = at >= 0 ? providers[at].apiKey : '';
    const apiKey = typeof incoming.apiKey === 'string' && incoming.apiKey ? incoming.apiKey : kept;
    const next = { ...(incoming as unknown as ProviderConfig), apiKey };
    if (at >= 0) providers[at] = next;
    else providers.push(next);
  }
  const known = (id: unknown): id is string => typeof id === 'string' && !blocked.has(id);
  let media: MediaSettings = current.media;
  if (object(backup.media)) {
    const incoming = backup.media as MediaSettings;
    media = { ...current.media, ...incoming, apiKey: typeof incoming.apiKey === 'string' && incoming.apiKey ? incoming.apiKey : current.media.apiKey };
  }
  return {
    providers,
    activeProviderId: typeof backup.activeProviderId === 'string' && providers.some((p) => p.id === backup.activeProviderId) ? backup.activeProviderId : current.activeProviderId,
    gate: object(backup.gate) ? ({ ...current.gate, ...(backup.gate as unknown as NetGateSettings) }) : current.gate,
    lastProjectId: known(backup.lastProjectId) ? backup.lastProjectId : current.lastProjectId,
    lastKeptProjectId: known(backup.lastKeptProjectId) ? backup.lastKeptProjectId : current.lastKeptProjectId,
    agent: object(backup.agent) ? { ...DEFAULT_AGENT_SETTINGS, ...current.agent, ...(backup.agent as unknown as Partial<AgentSettings>) } : current.agent,
    media,
    messages: object(backup.messages) ? { ...DEFAULT_MESSAGE_SETTINGS, ...current.messages, ...(backup.messages as unknown as Partial<MessageSettings>) } : current.messages,
  };
}

/** A restore waiting on the desktop, or null when there is none (or the desktop cannot be reached). */
async function fetchPending(desktop: DesktopRef, fetchImpl: FetchLike, limits: Limits, retries: number, sleep: (ms: number) => Promise<void>): Promise<PendingImport | null> {
  for (let attempt = 0; ; attempt++) {
    let response: Response;
    try {
      response = await fetchImpl(`${base(desktop)}/api/backup/agent-import`, { headers: { Authorization: `Bearer ${desktop.token}` }, signal: timeout(5000) });
    } catch {
      // The desktop's server may still be starting when this page loads first.
      if (attempt >= retries) return null;
      await sleep(400 * 2 ** attempt);
      continue;
    }
    if (!response.ok) return null;
    try {
      const body = (await response.json()) as { pending?: unknown };
      return body?.pending === true ? pendingFrom(body, limits) : null;
    } catch {
      return null;
    }
  }
}

/**
 * Put a restore the desktop has staged into this page's storage. Call it early in the start, before
 * anything is opened. Never throws; null when nothing waits (or the desktop is unreachable).
 */
export async function applyPendingRestore(desktop: DesktopRef, storage: AgentStorage, options: RestoreOptions = {}): Promise<ImportOutcome | null> {
  const fetchImpl: FetchLike = options.fetch ?? ((input, init) => fetch(input, init));
  const limits = { ...DEFAULT_LIMITS, ...options.limits };
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  try {
    const pending = await fetchPending(desktop, fetchImpl, limits, options.retries ?? 3, sleep);
    if (!pending) return null;
    const url = `${base(desktop)}/api/backup/agent-import/${encodeURIComponent(pending.id)}`;
    const outcome = await doImport(desktop, storage, pending, url, fetchImpl, limits).catch(
      (e): ImportOutcome => ({ ok: false, error: message(e), applied: { projects: 0, files: 0, settings: false }, warnings: [] }),
    );
    try {
      await fetchImpl(`${url}/done`, { method: 'POST', headers: { ...bearer(desktop, pending.token), 'Content-Type': 'application/json' }, body: JSON.stringify(outcome), signal: timeout(30_000) });
    } catch {
      /* the desktop keeps the restore pending and asks again at the next start */
    }
    return outcome;
  } catch {
    return null;
  }
}

async function download(desktop: DesktopRef, pending: PendingImport, url: string, fetchImpl: FetchLike): Promise<Uint8Array> {
  const expected = pending.partSize > 0 ? Math.ceil(pending.size / pending.partSize) : 0;
  if (pending.parts !== expected) throw new Error('the desktop described the archive in a way that does not add up');
  const bytes = new Uint8Array(pending.size);
  let at = 0;
  for (let i = 0; i < pending.parts; i++) {
    const response = await fetchImpl(`${url}/part/${i}`, { headers: bearer(desktop, pending.token), signal: timeout(60_000) });
    if (!response.ok) throw new Error(`the desktop answered ${response.status} for part ${i}`);
    const part = new Uint8Array(await response.arrayBuffer());
    if (part.length > pending.partSize || at + part.length > pending.size) throw new Error(`part ${i} is bigger than the desktop said`);
    bytes.set(part, at);
    at += part.length;
  }
  if (at !== pending.size) throw new Error('the archive came short of the size the desktop said');
  if ((await sha256Hex(bytes)) !== pending.sha256) throw new Error('the archive does not match its checksum, so it was not restored');
  return bytes;
}

interface Listed {
  name: string;
  path: string;
  where: 'manifest' | 'settings' | 'opfs';
  size: number;
}

async function doImport(desktop: DesktopRef, storage: AgentStorage, pending: PendingImport, url: string, fetchImpl: FetchLike, limits: Limits): Promise<ImportOutcome> {
  const warnings: string[] = [];
  const applied = { projects: 0, files: 0, settings: false };
  const archive = await download(desktop, pending, url, fetchImpl);

  // What is in it, checked before anything is unpacked (fflate reports each entry's sizes first).
  const listed: Listed[] = [];
  const seen = new Set<string>();
  let count = 0;
  let total = 0;
  unzipSync(archive, {
    filter: (file) => {
      count++;
      if (count > limits.entries) throw new Error(`the archive holds more than ${limits.entries.toLocaleString()} entries`);
      total += file.originalSize;
      if (total > limits.unpacked) throw new Error(`the archive unpacks to more than ${Math.round(limits.unpacked / 2 ** 20)} MiB`);
      const check = checkEntry(file.name);
      if (check.kind === 'dir') return false;
      if (check.kind === 'skip') {
        warnings.push(`${file.name.slice(0, 120)} was not restored: ${check.why}.`);
        return false;
      }
      if (seen.has(file.name)) {
        warnings.push(`${file.name.slice(0, 120)} is in the archive twice: the first was restored.`);
        return false;
      }
      seen.add(file.name);
      if (file.originalSize > limits.entry) {
        warnings.push(`${file.name.slice(0, 120)} was not restored: it is bigger than ${Math.round(limits.entry / 2 ** 20)} MiB.`);
        return false;
      }
      listed.push({ name: file.name, path: check.path, where: check.where, size: file.originalSize });
      return false;
    },
  });

  // Its own manifest says what it is; one of another kind, or of a newer format, is not restored.
  const manifest = listed.find((l) => l.where === 'manifest');
  let manifestOk = false;
  if (manifest) {
    try {
      const m = JSON.parse(new TextDecoder().decode(unzipSync(archive, { filter: (f) => f.name === manifest.name })[manifest.name])) as { v?: unknown; kind?: unknown };
      manifestOk = m?.kind === 'oaiy-agent-storage' && m?.v === 1;
    } catch {
      manifestOk = false;
    }
  }
  if (!manifestOk) throw new Error('this is not an archive of the Agent’s storage that this version can restore');

  // Projects that must not be written to: the archive's or this page's own says incognito.
  const blocked = new Set<string>();
  const idOf = (path: string) => (path.startsWith('projects/') ? path.split('/')[1] : null);
  const metaNames = new Set(listed.filter((l) => l.where === 'opfs' && /^projects\/[^/]+\/project\.json$/.test(l.path) && l.size <= 1024 * 1024).map((l) => l.name));
  if (metaNames.size) {
    const metas = unzipSync(archive, { filter: (f) => metaNames.has(f.name) });
    for (const [name, bytes] of Object.entries(metas)) if (incognitoFlag(bytes)) blocked.add(name.split('/')[2]);
  }
  for (const id of new Set(listed.map((l) => idOf(l.path)).filter((id): id is string => id !== null))) {
    if (!blocked.has(id) && (await isIncognito(storage, `projects/${id}`))) blocked.add(id);
  }
  for (const id of blocked) warnings.push(`Project ${id} was not restored: it is (or was) an incognito project, which is never kept.`);
  const writes = listed.filter((l) => l.where === 'opfs' && !blocked.has(idOf(l.path) ?? ''));

  // The undo copy comes first, so that nothing is replaced before there is a way back.
  if (pending.kind === 'restore') {
    const target = uploadTo(desktop, pending.token, `${url}/undo-part`, `${url}/undo-done`, fetchImpl);
    // Without the API keys: they were sealed here with a key the browser will not let out, and the undo copy is
    // kept by the desktop as plain files. The merge rule below never replaces a stored key with an empty one, so
    // an undo does not need them (it also does not roll back keys that a restore brought in).
    const copy = await exportAgentStorage(storage, target, { includeKeys: false, limits });
    if (!copy.ok) throw new Error(`the undo copy could not be made, so nothing was restored (${copy.error ?? 'unknown reason'})`);
    const left = new Set(copy.skipped);
    const clash = writes.filter((w) => left.has(w.path));
    if (clash.length) throw new Error(`the undo copy could not include ${clash.length} file${clash.length === 1 ? '' : 's'} that this restore would replace, so nothing was restored`);
    if (copy.warnings.length) warnings.push(...copy.warnings.map((w) => `Undo copy: ${w}`));
  }

  // Files, a few at a time, so that only a part of the archive is unpacked at once.
  const written = new Set<string>();
  let failed = 0;
  for (let from = 0; from < writes.length; ) {
    let to = from;
    let bytes = 0;
    while (to < writes.length && to - from < BATCH_ENTRIES && (to === from || bytes + writes[to].size <= BATCH_BYTES)) bytes += writes[to++].size;
    const names = new Set(writes.slice(from, to).map((w) => w.name));
    const files = unzipSync(archive, { filter: (f) => names.has(f.name) });
    for (const item of writes.slice(from, to)) {
      const data = files[item.name];
      if (!data) {
        failed++;
        continue;
      }
      try {
        await storage.writeFile(item.path, data);
        applied.files++;
        const id = idOf(item.path);
        if (id) written.add(id);
      } catch (e) {
        failed++;
        warnings.push(`${item.path} could not be written: ${message(e)}.`);
      }
    }
    from = to;
  }
  applied.projects = written.size;

  const settingsEntry = listed.find((l) => l.where === 'settings');
  if (settingsEntry) {
    try {
      const raw = unzipSync(archive, { filter: (f) => f.name === settingsEntry.name })[settingsEntry.name];
      const parsed = JSON.parse(new TextDecoder().decode(raw)) as Partial<BackupSettings>;
      await storage.writeSettings(mergeSettings(await storage.readSettings(), parsed, blocked));
      applied.settings = true;
    } catch (e) {
      failed++;
      warnings.push(`The settings could not be restored: ${message(e)}.`);
    }
  }
  return failed ? { ok: false, error: `${failed} item${failed === 1 ? '' : 's'} could not be restored`, applied, warnings } : { ok: true, applied, warnings };
}

// ---------------------------------------------------------------------------
// The real storage: OPFS and IndexedDB

type Dir = FileSystemDirectoryHandle;

function entriesOf(dir: Dir): AsyncIterable<[string, FileSystemHandle]> {
  return dir as unknown as AsyncIterable<[string, FileSystemHandle]>;
}

async function dirAt(path: string, create: boolean): Promise<Dir> {
  let dir: Dir = await navigator.storage.getDirectory();
  for (const seg of path.split('/').filter(Boolean)) {
    try {
      dir = await dir.getDirectoryHandle(seg, { create });
    } catch (error) {
      // A file where a folder now belongs: replace it.
      if (!create || (error as DOMException).name !== 'TypeMismatchError') throw error;
      await dir.removeEntry(seg);
      dir = await dir.getDirectoryHandle(seg, { create: true });
    }
  }
  return dir;
}

async function* walkDir(dir: Dir, prefix: string): AsyncGenerator<StoredFile> {
  for await (const [name, handle] of entriesOf(dir)) {
    const path = `${prefix}/${name}`;
    if (handle.kind === 'directory') {
      yield* walkDir(handle as Dir, path);
      continue;
    }
    let file: File | null = null;
    let failure: unknown = null;
    try {
      file = await (handle as FileSystemFileHandle).getFile();
    } catch (e) {
      failure = e;
    }
    yield {
      path,
      size: file?.size ?? 0,
      read: async () => {
        if (!file) throw failure;
        return new Uint8Array(await file.arrayBuffer());
      },
    };
  }
}

/** The Agent's storage in this browser: the Origin Private File System and IndexedDB (see vfs/projects.ts, settings.ts). */
export function opfsStorage(): AgentStorage {
  return {
    async projectIds() {
      let dir: Dir;
      try {
        dir = await dirAt('projects', false);
      } catch {
        return [];
      }
      const ids: string[] = [];
      for await (const [name, handle] of entriesOf(dir)) if (handle.kind === 'directory') ids.push(name);
      return ids;
    },
    async *walk(root) {
      let dir: Dir;
      try {
        dir = await dirAt(root, false);
      } catch {
        return;
      }
      yield* walkDir(dir, root);
    },
    async readFile(path) {
      try {
        const slash = path.lastIndexOf('/');
        const dir = await dirAt(path.slice(0, slash), false);
        return new Uint8Array(await (await (await dir.getFileHandle(path.slice(slash + 1))).getFile()).arrayBuffer());
      } catch {
        return null;
      }
    },
    async writeFile(path, data) {
      const slash = path.lastIndexOf('/');
      const dir = await dirAt(path.slice(0, slash), true);
      const name = path.slice(slash + 1);
      // A folder where a file now belongs: replace it.
      await dir.getDirectoryHandle(name).then(() => dir.removeEntry(name, { recursive: true }), () => {});
      const writable = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await writable.write(new Blob([data as BlobPart]));
      await writable.close();
    },
    readSettings: () => loadSettings(),
    async writeSettings(s) {
      await saveProviders(s.providers, s.activeProviderId);
      await saveMedia(s.media);
      await saveGate(s.gate);
      await saveAgentSettings(s.agent);
      await saveMessages(s.messages);
      await saveLastProject(s.lastProjectId);
      await saveLastKeptProject(s.lastKeptProjectId);
    },
  };
}
