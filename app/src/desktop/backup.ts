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
 *    opened. The page asks `/api/backup/agent-import` whether one waits (with the
 *    `backupToken` the desktop gave it in `window.__OAIY_DESKTOP__`, sent as
 *    `X-Backup-Token` on every import request), downloads it in parts, checks its
 *    SHA-256, takes an undo copy of what it holds now (uploaded to the desktop, which
 *    keeps it for "Undo the last restore", for an undo as well as a restore), and
 *    only then writes. It deletes only what the desktop says the restore being undone
 *    had added. It reports every outcome, and the files it added, with `POST .../done`.
 *
 * WHAT IT WRITES IS DECIDED BY THE ARCHIVE'S RECORD, AND NOTHING ELSE. The desktop does
 * not hand back the archive the page exported: it reads its directory, classifies every
 * name against its table (`backup/table.json`: default-deny, so a name it does not know
 * is not restored), and builds a new archive of only what may come back and only what
 * the person ticked. Its record, `agent-manifest.json` (`v: 2`), names every item and how
 * it is brought back:
 *
 *    replace         write the file, over the one of the same name
 *    union           the numbers not to be contacted: added to the list here, none of
 *                    ours is ever taken away or replaced
 *    campaign        an outreach campaign, which arrives PAUSED: a file of the same
 *                    name that is here already is left exactly as it is (local wins),
 *                    and one that is written is never running
 *    campaign-index  the list of campaigns: joined to ours, and only for campaigns that
 *                    are here
 *    settings        the settings, only the keys the desktop let through, by the merge
 *                    rule below
 *
 * The page refuses the whole import, writing nothing, when the record is missing, is not
 * version 2 of this kind, names a name twice or one the archive does not hold, names
 * something outside the Agent's own storage or with an unsafe name, or a name in a mode
 * it may not have, and when the archive's own directory lists a name twice (the desktop
 * would have refused it; a program reading the other one would see other bytes). An entry
 * the record does not name is never written.
 *
 * What the settings may change is what the person ticked: the desktop leaves them out of
 * the archive otherwise (`apply.keys` still says whether a key may come over). A provider
 * of the backup that points somewhere else than the one kept now never gets the kept key:
 * it arrives beside it, without one. An UNDO is the one exception to "nothing is taken away": the
 * providers become exactly the list the undo copy holds (the one from before the restore), so a
 * provider the restore added goes again; a key that is kept stays with its provider, at the same address.
 *
 * What is never exported: an incognito project (it is designed never to leave its
 * folder), a project whose project.json cannot be read (it is skipped, not guessed
 * at), the paired desktop's token, the sealed encryption key of the API keys,
 * and, unless the person chose "Include my API provider keys", the API keys.
 *
 * The logic works on `AgentStorage` and an injected `fetch`, so it is tested without a
 * browser; `opfsStorage()` is the real one.
 */
import { Zip, ZipDeflate, strToU8, unzipSync } from 'fflate';
import type { ProviderConfig } from '../agent/providers/types';
import type { MediaSettings } from '../agent/media';
import type { NetGateSettings } from '../gate/netgate';
import { PersonIndex } from '../phoneNumbers';
import { DEFAULT_AGENT_SETTINGS, DEFAULT_MESSAGE_SETTINGS, loadSettings, saveAgentSettings, saveGate, saveLastKeptProject, saveLastProject, saveMedia, saveMessages, saveProviders, type AgentSettings, type MessageSettings, type Settings } from '../settings';

/** What is sent to the desktop in one request. */
export const PART_BYTES = 4 * 1024 * 1024;
/** One file over this is left out, and named in the warnings. */
export const MAX_FILE_BYTES = 64 * 1024 * 1024;
/** Once this much is in the archive, no more files are added. */
export const MAX_TOTAL_BYTES = 512 * 1024 * 1024;
/**
 * An export stops adding files at this many entries (the record and the settings included), and a restore refuses
 * an archive of more. 50,000: a real storage is thousands of files, and the desktop refuses more than this.
 */
export const MAX_ENTRIES = 50_000;
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

/** The warning when the archive was cut short by the number of entries it may hold. */
const COUNT_WARNING = (entries: number, left: number) => `The Agent's storage holds more than ${entries.toLocaleString('en-US')} files, so ${left} more file${left === 1 ? ' was' : 's were'} not included.`;

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
  /** A file's bytes, or null when it is NOT THERE; any other trouble reading it throws (a caller that must not guess treats that as unreadable). */
  readFile(path: string): Promise<Uint8Array | null>;
  /** A file written, its folders made, whatever stood in the way replaced. */
  writeFile(path: string, data: Uint8Array): Promise<void>;
  /** Whether a file is there (a folder is not a file). */
  exists(path: string): Promise<boolean>;
  /** A file removed (never a folder): true when it was there, false when it was not. */
  removeFile(path: string): Promise<boolean>;
  /** The settings as the page loads them (the API keys already opened). */
  readSettings(): Promise<Settings>;
  writeSettings(settings: BackupSettings): Promise<void>;
}

/** The desktop, as the page knows it inside OAIY's window. */
export interface DesktopRef {
  origin: string;
  token: string;
  /** What the desktop gave this page for the restore's own requests (`window.__OAIY_DESKTOP__.backupToken`). */
  backupToken?: string;
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

/**
 * Whether a project is an incognito one, as its project.json says. It fails CLOSED: a project.json that
 * cannot be read, or is not a JSON object, is `unreadable`, and the project is left alone rather than
 * guessed to be an ordinary one. A project with no project.json at all is not a project (`no`).
 */
type Incognito = 'yes' | 'no' | 'unreadable';

/** A project.json larger than this is not read (a real one is a few hundred bytes): unreadable. */
const MAX_PROJECT_JSON = 1024 * 1024;

function incognitoOf(bytes: Uint8Array): Incognito {
  if (bytes.length > MAX_PROJECT_JSON) return 'unreadable';
  try {
    const meta = JSON.parse(new TextDecoder().decode(bytes)) as unknown;
    if (!meta || typeof meta !== 'object' || Array.isArray(meta)) return 'unreadable';
    return (meta as { incognito?: unknown }).incognito === true ? 'yes' : 'no';
  } catch {
    return 'unreadable';
  }
}

async function projectState(storage: AgentStorage, projectPath: string): Promise<Incognito> {
  try {
    const bytes = await storage.readFile(`${projectPath}/project.json`);
    return bytes ? incognitoOf(bytes) : 'no';
  } catch {
    return 'unreadable';
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

    // Room for files: the record and the settings are entries too, and the desktop refuses an archive of more than the
    // limit, so a backup the Agent makes is never refused for its count.
    const room = Math.max(0, limits.entries - 2);
    let full: 'size' | 'count' | null = null;
    let notAdded = 0;
    const addTree = async (root: string): Promise<void> => {
      for await (const file of storage.walk(root)) {
        if (file.path.endsWith('.crswap')) continue;
        if (full || counts.bytes >= limits.total || counts.files >= room) {
          full ??= counts.bytes >= limits.total ? 'size' : 'count';
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

    // Projects that are left out: an incognito one, and one whose project.json cannot be read (never a guess).
    const incognito = new Set<string>();
    for (const id of await storage.projectIds()) {
      const state = await projectState(storage, `projects/${id}`);
      if (state === 'yes') {
        incognito.add(id);
        counts.incognitoSkipped++;
        continue;
      }
      if (state === 'unreadable') {
        incognito.add(id);
        warnings.push(`Project ${id} was skipped: its project.json could not be read.`);
        continue;
      }
      counts.projects++;
      await addTree(`projects/${id}`);
    }
    await addTree('front-desk');
    if (full === 'size') warnings.push(TOTAL_WARNING(Math.round(limits.total / 2 ** 20), notAdded));
    else if (full === 'count') warnings.push(COUNT_WARNING(limits.entries, notAdded));

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
  applied: { projects: number; files: number; settings: boolean; removed: number };
  warnings: string[];
  /** The zip entry names of the files this import wrote that were not there before (at most MAX_ADDED): what an undo takes away. */
  added: string[];
}

/** How many added files are listed to the desktop: more are dropped, and the warnings say so. */
export const MAX_ADDED = 20_000;

/** What the person chose at restore time, as the desktop tells the page (nothing here is the backup's own say-so). */
export interface RestoreApply {
  /** Take the settings (the network gate, how calls and texts are answered, the AI providers) from the backup. */
  settings: boolean;
  /** Bring API keys over (only ever where none is kept, and only for the same address). */
  keys: boolean;
}

interface PendingImport {
  id: string;
  kind: 'restore' | 'undo';
  size: number;
  sha256: string;
  parts: number;
  partSize: number;
  apply: RestoreApply;
  /** Zip entry names of files the restore being undone had added: they are taken away again. */
  remove: string[];
}

export interface RestoreOptions {
  fetch?: FetchLike;
  limits?: Partial<Limits>;
  /** Tries to reach the desktop again after a refused connection (it may still be starting), and how long to wait. */
  retries?: number;
  sleep?: (ms: number) => Promise<void>;
}

/** How an item of the archive is brought back (see the header). */
export type ItemMode = 'replace' | 'union' | 'campaign' | 'campaign-index' | 'settings';
const MODES: readonly ItemMode[] = ['replace', 'union', 'campaign', 'campaign-index', 'settings'];

/** The front desk's outreach files, as the archive names them (see vfs/projects.ts). */
const OUTREACH = 'opfs/front-desk/outreach/';
/** The numbers not to be called or texted again. */
export const DO_NOT_CONTACT = `${OUTREACH}do-not-contact.json`;
/** The most numbers a restore adds to the list of numbers not to be contacted (the desktop hands over no more than this, and the page holds to it). */
export const MAX_DO_NOT_CONTACT_ADDED = 5000;
/** The list of campaigns. */
export const CAMPAIGN_INDEX = `${OUTREACH}index.json`;
/** The archive's own record of what it brings back. */
export const RECORD = 'agent-manifest.json';
/** The most the record may be: a few thousand items of a few hundred bytes. */
const MAX_RECORD_BYTES = 16 * 1024 * 1024;

export type NameCheck = { ok: true; where: 'settings' | 'opfs'; path: string; mode: ItemMode } | { ok: false; why: string };

/**
 * Whether a name may be an item of the archive, where it goes (a path from the storage's root), and the ONE mode it
 * may be brought back in. Only the Agent's own storage: `idb/settings.json`, `opfs/projects/<id>/...` and
 * `opfs/front-desk/...`; and only plain names inside it. (The record, `agent-manifest.json`, is not an item.)
 */
export function checkName(name: string): NameCheck {
  const no = (why: string): NameCheck => ({ ok: false, why });
  if (!name || name.length > 1024) return no('its name is empty or too long');
  if (name.endsWith('/')) return no('it is a folder, not a file');
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(name)) return no('its name has a control character');
  if (name.includes('\\')) return no('its name has a backslash');
  if (name.startsWith('/')) return no('its name is an absolute path');
  if (name.includes(':')) return no('its name has a colon (a drive)');
  const segments = name.split('/');
  if (segments.some((s) => s === '' || s === '.' || s === '..')) return no('its name goes out of the folder it is in');
  if (name === 'idb/settings.json') return { ok: true, where: 'settings', path: name, mode: 'settings' };
  const inside = segments[0] === 'opfs' && ((segments[1] === 'projects' && segments.length >= 4) || (segments[1] === 'front-desk' && segments.length >= 3));
  if (!inside) return no('it is not part of the Agent\u2019s storage');
  const path = segments.slice(1).join('/');
  // A few names are not plain files: what they are decides how they come back, and nothing else may be said of them.
  if (name === DO_NOT_CONTACT) return { ok: true, where: 'opfs', path, mode: 'union' };
  if (name === CAMPAIGN_INDEX) return { ok: true, where: 'opfs', path, mode: 'campaign-index' };
  if (name.startsWith(OUTREACH) && segments.length === 4 && name.endsWith('.json')) return { ok: true, where: 'opfs', path, mode: 'campaign' };
  return { ok: true, where: 'opfs', path, mode: 'replace' };
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** What the desktop's description of a waiting restore comes to: something to act on, or (when it has an id) something to refuse aloud. */
type PendingParse = { ok: true; pending: PendingImport } | { ok: false; id: string | null; why: string };

function pendingFrom(value: unknown, limits: Limits): PendingParse {
  const v = value as Record<string, unknown> | null;
  if (!v || typeof v !== 'object') return { ok: false, id: null, why: 'no description' };
  if (typeof v.id !== 'string' || !SAFE_ID.test(v.id)) return { ok: false, id: null, why: 'no usable id' };
  const id = v.id;
  const refuse = (why: string): PendingParse => ({ ok: false, id, why });
  if (v.kind !== 'restore' && v.kind !== 'undo') return refuse('the desktop described the restore in a way this version does not understand');
  const whole = (n: unknown, max: number): n is number => typeof n === 'number' && Number.isInteger(n) && n >= 0 && n <= max;
  if (typeof v.size === 'number' && Number.isInteger(v.size) && v.size > limits.unpacked) {
    return refuse(`the Agent’s storage in the backup is bigger than this version restores (${Math.round(limits.unpacked / 2 ** 20)} MiB), so it was not restored`);
  }
  if (!whole(v.size, limits.unpacked) || !whole(v.parts, 1_000_000) || !whole(v.partSize, 1024 * 1024 * 1024)) return refuse('the desktop described the archive in a way that does not add up');
  if (typeof v.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(v.sha256)) return refuse('the desktop described the archive in a way that does not add up');
  const apply = v.apply as Partial<RestoreApply> | undefined;
  const flags: RestoreApply = { settings: apply?.settings === true, keys: apply?.keys === true };
  if (v.remove !== undefined && !Array.isArray(v.remove)) return refuse('the list of files to take away is not one this version accepts');
  const remove: unknown[] = Array.isArray(v.remove) ? v.remove : [];
  if (remove.length > limits.entries || remove.some((n) => typeof n !== 'string')) return refuse('the list of files to take away is not one this version accepts');
  return { ok: true, pending: { id, kind: v.kind, size: v.size, sha256: v.sha256, parts: v.parts, partSize: v.partSize, apply: flags, remove: remove as string[] } };
}

/** How two provider records are told to point at the same place: the same kind of server at the same address. */
function sameEndpoint(a: { type?: unknown; baseUrl?: unknown }, b: { type?: unknown; baseUrl?: unknown }): boolean {
  const address = (u: unknown) => (typeof u === 'string' ? u.trim().replace(/\/+$/, '') : '');
  return a.type === b.type && address(a.baseUrl) === address(b.baseUrl);
}

export interface MergeOptions {
  /** Bring API keys over: only where none is kept, only for a provider at the same address. Default: no. */
  keys?: boolean;
  /** Where to say what was kept or set apart. */
  notes?: string[];
  /**
   * An undo: the providers become exactly the list of the backup (which is the undo copy, made before the restore that is
   * being undone), so a provider that restore added goes again, and one that is not in the copy does not stay. A restore
   * merges and never takes one away; only this does. A key that is kept stays with its provider only at the same address, and
   * the copy holds none. (A copy that holds no providers key is a copy of a list that was empty: the desktop leaves an empty
   * list out.)
   */
  exact?: boolean;
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const text = (v: unknown, max = 4096): string | undefined => (typeof v === 'string' && v.length <= max ? v : undefined);
const flag = (v: unknown): boolean | undefined => (typeof v === 'boolean' ? v : undefined);
const whole = (v: unknown, min: number, max: number): number | undefined => (typeof v === 'number' && Number.isInteger(v) && v >= min && v <= max ? v : undefined);
const PROVIDER_TYPES = ['anthropic', 'openai', 'local', 'custom'];

/** Only the fields of a provider that are settings, of the right kind (an unknown field is not carried in). */
function pickProvider(v: unknown): ProviderConfig | null {
  if (!isObject(v)) return null;
  const id = text(v.id, 128);
  const type = text(v.type, 32);
  if (!id || !type || !PROVIDER_TYPES.includes(type)) return null;
  const out: Record<string, unknown> = { id, type, name: text(v.name, 200) ?? id, apiKey: text(v.apiKey) ?? '' };
  const optional: Record<string, unknown> = {
    baseUrl: text(v.baseUrl, 2048),
    modelId: text(v.modelId, 200),
    followEngine: flag(v.followEngine),
    orgId: text(v.orgId, 200),
    serverKind: text(v.serverKind, 32),
    contextTokens: whole(v.contextTokens, 0, 100_000_000),
    parallelAgents: whole(v.parallelAgents, 1, 64),
  };
  for (const [k, value] of Object.entries(optional)) if (value !== undefined) out[k] = value;
  return out as unknown as ProviderConfig;
}

/** The parts of the media service that are not its address or its key: whether the Agent may use it, and the models it names. */
function pickMedia(v: Record<string, unknown>): Partial<MediaSettings> {
  const out: Record<string, unknown> = {};
  const enabled = flag(v.enabled);
  if (enabled !== undefined) out.enabled = enabled;
  for (const k of ['imageModel', 'videoModel', 'speechModel', 'musicModel', 'soundModel', 'model3dModel']) {
    const value = text(v[k], 200);
    if (value !== undefined) out[k] = value;
  }
  return out as Partial<MediaSettings>;
}

/** Only the message settings the Agent has, of the right kind. */
function pickMessages(v: Record<string, unknown>): Partial<MessageSettings> {
  const out: Record<string, unknown> = {};
  for (const k of ['answer', 'calls', 'callBack']) if (flag(v[k]) !== undefined) out[k] = v[k];
  for (const [k, max] of [['instructions', 20000], ['callInstructions', 20000], ['callBackLine', 2000], ['country', 8]] as const) if (text(v[k], max) !== undefined) out[k] = v[k];
  if (v.callBackFilter === 'answered' || v.callBackFilter === 'au' || v.callBackFilter === 'any') out.callBackFilter = v.callBackFilter;
  return out as Partial<MessageSettings>;
}

/** The network gate: its mode and its two lists. */
function pickGate(v: Record<string, unknown>): Partial<NetGateSettings> {
  const out: Record<string, unknown> = {};
  if (v.mode === 'open' || v.mode === 'allowlist' || v.mode === 'blocked') out.mode = v.mode;
  for (const k of ['allow', 'deny']) {
    const list = v[k];
    if (Array.isArray(list)) out[k] = list.filter((x): x is string => typeof x === 'string' && x.length <= 253).slice(0, 500);
  }
  return out as Partial<NetGateSettings>;
}

/** How the Agent manages its work: two numbers, within their limits. */
function pickAgent(v: Record<string, unknown>): Partial<AgentSettings> {
  const out: Record<string, unknown> = {};
  if (typeof v.compactAt === 'number' && Number.isFinite(v.compactAt) && v.compactAt >= 0.05 && v.compactAt <= 1) out.compactAt = v.compactAt;
  const tokens = whole(v.subAgentTokens, 1000, 10_000_000);
  if (tokens !== undefined) out.subAgentTokens = tokens;
  return out as Partial<AgentSettings>;
}

/** The providers of an undo: exactly the list of the copy, in its order. Says which of the ones here it took away. */
function exactProviders(current: readonly ProviderConfig[], incoming: unknown, notes: string[]): ProviderConfig[] {
  const out: ProviderConfig[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(incoming) ? incoming : []) {
    const theirs = pickProvider(raw);
    if (!theirs || seen.has(theirs.id)) continue;
    seen.add(theirs.id);
    const kept = current.find((p) => p.id === theirs.id);
    out.push({ ...theirs, apiKey: kept && sameEndpoint(kept, theirs) ? kept.apiKey : '' });
  }
  const gone = current.filter((p) => !seen.has(p.id));
  if (gone.length) {
    const named = gone.slice(0, 10).map((p) => `“${clip(p.name || p.id)}”`);
    notes.push(`${gone.length === 1 ? 'A provider that was' : `${gone.length} providers that were`} not there before the restore ${gone.length === 1 ? 'was' : 'were'} taken away: ${named.join(', ')}${gone.length > named.length ? ` and ${gone.length - named.length} more` : ''}.`);
  }
  return out;
}

/**
 * What a restore does with the settings the desktop let through. What is applied is only what this page has a
 * setting for, of the right kind (a key that is not a setting, or a value that is not one, is not carried in), and
 * a key that the backup does not have leaves what is kept as it is.
 *
 * A key never moves to another address. A provider of the backup that has the id of one kept now, but
 * another type or address, does not replace it and never gets its key: the kept one stays as it is, and
 * the backup's arrives beside it (`<id>-from-backup`) without a key. A key from the backup is used only
 * where the kept provider (at the same address) has none, and only when the person asked for keys.
 * The media service follows the same rule. The project that was open last belongs to the computer it was
 * open on, and is never taken from a backup (`blocked` is what the caller knows to keep out of it).
 */
export function mergeSettings(current: Settings, backup: Partial<BackupSettings>, _blocked: ReadonlySet<string>, options: MergeOptions = {}): BackupSettings {
  const keys = options.keys === true;
  const notes = options.notes ?? [];
  const exact = options.exact === true;
  const providers: ProviderConfig[] = exact ? exactProviders(current.providers, backup.providers, notes) : current.providers.map((p) => ({ ...p }));
  for (const incoming of exact ? [] : Array.isArray(backup.providers) ? backup.providers : []) {
    const theirs = pickProvider(incoming);
    if (!theirs) continue;
    const backupKey = keys ? theirs.apiKey : '';
    const at = providers.findIndex((p) => p.id === theirs.id);
    if (at < 0) {
      providers.push({ ...theirs, apiKey: backupKey });
      continue;
    }
    const kept = providers[at];
    if (sameEndpoint(kept, theirs)) {
      // The same place: the backup's settings for it, with the key that is kept (or, where there is none, the backup's, if asked for).
      providers[at] = { ...kept, ...theirs, apiKey: kept.apiKey || backupKey };
      continue;
    }
    // Another place under the same id: what is kept stays exactly as it is, and this one is set beside it, keyless.
    const label = theirs.name || theirs.id;
    let id = `${theirs.id}-from-backup`;
    for (let n = 2; providers.some((p) => p.id === id && !sameEndpoint(p, theirs)); n++) id = `${theirs.id}-from-backup-${n}`;
    const beside = providers.findIndex((p) => p.id === id);
    const record: ProviderConfig = { ...theirs, id, name: `${label} (from backup)`, apiKey: beside >= 0 ? providers[beside].apiKey : '' };
    if (beside >= 0) providers[beside] = record;
    else providers.push(record);
    notes.push(`The backup’s provider “${label}” points somewhere else than yours, so yours was kept as it is and the backup’s was added as “${record.name}”, without a key.`);
  }
  let media: MediaSettings = current.media;
  if (isObject(backup.media)) {
    const theirs = backup.media;
    const safe = pickMedia(theirs);
    const address = text(theirs.baseUrl, 2048);
    const backupKey = keys ? text(theirs.apiKey) ?? '' : '';
    if (address !== undefined) {
      if (!current.media.baseUrl.trim()) {
        // Nothing is kept: the backup's service, and its key only if asked for (a key kept for no address is not sent to a new one).
        media = { ...current.media, ...safe, baseUrl: address, apiKey: backupKey };
      } else if (sameEndpoint({ baseUrl: current.media.baseUrl }, { baseUrl: address })) {
        media = { ...current.media, ...safe, baseUrl: address, apiKey: current.media.apiKey || backupKey };
      } else {
        notes.push('The backup’s image, video and audio service points somewhere else than yours, so yours was kept as it is.');
      }
    } else if (Object.keys(safe).length) {
      // No address in the backup: what it says of the service is put on the one that is kept.
      media = { ...current.media, ...safe };
    }
  }
  const named = typeof backup.activeProviderId === 'string' && providers.some((p) => p.id === backup.activeProviderId) ? backup.activeProviderId : current.activeProviderId;
  return {
    providers,
    // (An undo that took the active provider away leaves the first one that is left.)
    activeProviderId: exact && !providers.some((p) => p.id === named) ? (providers[0]?.id ?? null) : named,
    gate: isObject(backup.gate) ? { ...current.gate, ...pickGate(backup.gate) } : current.gate,
    lastProjectId: current.lastProjectId,
    lastKeptProjectId: current.lastKeptProjectId,
    agent: isObject(backup.agent) ? { ...DEFAULT_AGENT_SETTINGS, ...current.agent, ...pickAgent(backup.agent) } : current.agent,
    media,
    messages: isObject(backup.messages) ? { ...DEFAULT_MESSAGE_SETTINGS, ...current.messages, ...pickMessages(backup.messages) } : current.messages,
  };
}

function refused(why: string): ImportOutcome {
  return { ok: false, error: why, applied: { projects: 0, files: 0, settings: false, removed: 0 }, warnings: [], added: [] };
}

/** A restore waiting on the desktop: null when there is none (or the desktop cannot be reached). */
async function fetchPending(desktop: DesktopRef, token: string, fetchImpl: FetchLike, limits: Limits, retries: number, sleep: (ms: number) => Promise<void>): Promise<PendingParse | null> {
  for (let attempt = 0; ; attempt++) {
    let response: Response;
    try {
      response = await fetchImpl(`${base(desktop)}/api/backup/agent-import`, { headers: bearer(desktop, token), signal: timeout(5000) });
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
 * anything is opened. Never throws; null when nothing waits (or the desktop is unreachable, or did not
 * give this page its `backupToken`). Every restore it does not do is reported to the desktop with a
 * reason, so nothing is left waiting for ever.
 */
export async function applyPendingRestore(desktop: DesktopRef, storage: AgentStorage, options: RestoreOptions = {}): Promise<ImportOutcome | null> {
  const fetchImpl: FetchLike = options.fetch ?? ((input, init) => fetch(input, init));
  const limits = { ...DEFAULT_LIMITS, ...options.limits };
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  try {
    const token = desktop.backupToken;
    if (!token || !SAFE_ID.test(token)) return null;
    const found = await fetchPending(desktop, token, fetchImpl, limits, options.retries ?? 3, sleep);
    if (!found) return null;
    const id = found.ok ? found.pending.id : found.id;
    if (!id) return null;
    const url = `${base(desktop)}/api/backup/agent-import/${encodeURIComponent(id)}`;
    const outcome = found.ok
      ? await doImport(desktop, token, storage, found.pending, url, fetchImpl, limits).catch((e): ImportOutcome => refused(message(e)))
      : refused(found.why);
    try {
      await fetchImpl(`${url}/done`, { method: 'POST', headers: { ...bearer(desktop, token), 'Content-Type': 'application/json' }, body: JSON.stringify(outcome), signal: timeout(30_000) });
    } catch {
      /* the desktop keeps the restore pending and asks again at the next start */
    }
    return outcome;
  } catch {
    return null;
  }
}

async function download(desktop: DesktopRef, token: string, pending: PendingImport, url: string, fetchImpl: FetchLike): Promise<Uint8Array> {
  const expected = pending.partSize > 0 ? Math.ceil(pending.size / pending.partSize) : 0;
  if (pending.parts !== expected) throw new Error('the desktop described the archive in a way that does not add up');
  const bytes = new Uint8Array(pending.size);
  let at = 0;
  for (let i = 0; i < pending.parts; i++) {
    const response = await fetchImpl(`${url}/part/${i}`, { headers: bearer(desktop, token), signal: timeout(60_000) });
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

/** One item the archive's record names, checked. */
interface Item {
  /** Its name in the archive. */
  name: string;
  /** Where it goes: a path from the storage's root (`projects/<id>/chat.json`), or `idb/settings.json`. */
  path: string;
  where: 'settings' | 'opfs';
  mode: ItemMode;
  size: number;
}

/** A campaign's id as a file name (the front desk keeps `outreach/<id>.json`). */
const CAMPAIGN_ID = /^[A-Za-z0-9_-]{1,128}$/;

const clip = (name: string) => (name.length > 120 ? `${name.slice(0, 119)}…` : name);

/**
 * What the archive's record says it brings back, checked against the directory of the archive itself. Anything that
 * does not add up throws, and the whole import is refused with nothing written: the record is what the desktop
 * decided (see the header), and one that is not well formed is not one this page acts on.
 */
function itemsOf(record: unknown, directory: ReadonlyMap<string, number>, limits: Limits): Item[] {
  if (!isObject(record) || record.kind !== 'oaiy-agent-storage') throw new Error('this is not an archive of the Agent’s storage that this version can restore');
  if (record.v !== 2) throw new Error('this archive was not prepared by OAIY Desktop for this version of the Agent (its record is not version 2), so it was not restored');
  if (!Array.isArray(record.items)) throw new Error('this archive does not say what it brings back, so it was not restored');
  if (record.items.length > limits.entries) throw new Error(`the archive names more than ${limits.entries.toLocaleString()} items, so it was not restored`);
  const seen = new Set<string>();
  const items: Item[] = [];
  for (const named of record.items) {
    if (!isObject(named) || typeof named.name !== 'string' || typeof named.mode !== 'string' || !MODES.includes(named.mode as ItemMode)) {
      throw new Error('the archive names an item in a way this version does not accept, so it was not restored');
    }
    const name = named.name;
    if (seen.has(name)) throw new Error(`the archive names ${clip(name)} twice, so it was not restored`);
    seen.add(name);
    const check = checkName(name);
    if (!check.ok) throw new Error(`the archive names ${clip(name)}, which this restore may not write (${check.why}), so it was not restored`);
    if (check.mode !== named.mode) throw new Error(`the archive brings ${clip(name)} back as “${named.mode}”, which is not how it may come back, so it was not restored`);
    const size = directory.get(name);
    if (size === undefined) throw new Error(`the archive names ${clip(name)} and does not hold it, so it was not restored`);
    items.push({ name, path: check.path, where: check.where, mode: check.mode, size });
  }
  return items;
}

/** The numbers not to be contacted: what is here, and what the backup adds that is not here already (by person, not by spelling). */
async function unionDoNotContact(storage: AgentStorage, item: Item, data: Uint8Array): Promise<{ data?: Uint8Array; warn?: string; failed?: boolean }> {
  let incoming: unknown;
  try {
    incoming = JSON.parse(new TextDecoder().decode(data));
  } catch {
    incoming = null;
  }
  if (!Array.isArray(incoming)) return { warn: `${item.name} was not restored: it is not a list of numbers.` };
  let held: Uint8Array | null;
  try {
    held = await storage.readFile(item.path);
  } catch {
    return { failed: true, warn: `${item.path} could not be read, so the numbers in the backup were not added to it.` };
  }
  let list: unknown[] = [];
  if (held) {
    try {
      const parsed: unknown = JSON.parse(new TextDecoder().decode(held));
      if (!Array.isArray(parsed)) throw new Error('not a list');
      list = parsed;
    } catch {
      // What is here cannot be read as a list: it is not written over, so nothing of it is lost.
      return { failed: true, warn: `${item.path} is not a list this page can read, so the numbers in the backup were not added to it.` };
    }
  }
  const before = list.length;
  // Who is here already, found in constant time (asking `samePerson` of each entry of the list, for each number that comes, took
  // 40 s for 8,000 numbers, in a page that waits for this before it opens anything).
  const here = new PersonIndex();
  for (const have of list) if (isObject(have) && typeof have.number === 'string') here.add(have.number);
  const coming = incoming.length > MAX_DO_NOT_CONTACT_ADDED ? incoming.slice(0, MAX_DO_NOT_CONTACT_ADDED) : incoming;
  const warn = incoming.length > coming.length ? `${(incoming.length - coming.length).toLocaleString()} of the numbers not to be contacted in the backup were left out: at most ${MAX_DO_NOT_CONTACT_ADDED.toLocaleString()} are added at once.` : undefined;
  for (const entry of coming) {
    if (!isObject(entry)) continue;
    const number = text(entry.number, 40);
    if (!number || [...number].some((c) => c < ' ')) continue;
    if (here.has(number)) continue;
    const at = typeof entry.at === 'number' && Number.isFinite(entry.at) && entry.at >= 0 ? entry.at : 0;
    list.push({ number, at, why: text(entry.why, 300) ?? '' });
    here.add(number);
  }
  // Nothing to add: the file that is here is left as it is.
  if (list.length === before) return warn ? { warn } : {};
  return { data: new TextEncoder().encode(JSON.stringify(list)), ...(warn ? { warn } : {}) };
}

/** A campaign as it is written: never running, and nothing it waits for (whatever the archive says). */
function pausedCampaign(data: Uint8Array): Uint8Array | null {
  let campaign: unknown;
  try {
    campaign = JSON.parse(new TextDecoder().decode(data));
  } catch {
    return null;
  }
  if (!isObject(campaign) || typeof campaign.id !== 'string' || !CAMPAIGN_ID.test(campaign.id)) return null;
  if (campaign.state !== 'paused' && campaign.state !== 'done' && campaign.state !== 'stopped') campaign.state = 'paused';
  campaign.waitingFor = '';
  // It was never approved on this computer: it is started by a person, and only then.
  campaign.approvedAt = 0;
  return new TextEncoder().encode(JSON.stringify(campaign));
}

/** The list of campaigns: what is here, then the backup's, and only those whose campaign is here. */
async function joinCampaignIndex(storage: AgentStorage, item: Item, data: Uint8Array): Promise<{ data?: Uint8Array; warn?: string; failed?: boolean }> {
  let incoming: unknown;
  try {
    incoming = JSON.parse(new TextDecoder().decode(data));
  } catch {
    incoming = null;
  }
  if (!Array.isArray(incoming)) return { warn: `${item.name} was not restored: it is not a list of campaigns.` };
  let held: Uint8Array | null;
  try {
    held = await storage.readFile(item.path);
  } catch {
    return { failed: true, warn: `${item.path} could not be read, so the campaigns in the backup were not added to it.` };
  }
  let ids: string[] = [];
  if (held) {
    try {
      const parsed: unknown = JSON.parse(new TextDecoder().decode(held));
      if (Array.isArray(parsed)) ids = parsed.filter((id): id is string => typeof id === 'string');
    } catch {
      /* an index that cannot be read lists nothing, as the engine reads it */
    }
  }
  const before = ids.length;
  for (const id of incoming) {
    if (typeof id !== 'string' || !CAMPAIGN_ID.test(id) || ids.includes(id)) continue;
    if (!(await storage.exists(`front-desk/outreach/${id}.json`))) continue;
    ids.push(id);
  }
  if (ids.length === before) return {};
  return { data: new TextEncoder().encode(JSON.stringify(ids)) };
}

async function doImport(desktop: DesktopRef, token: string, storage: AgentStorage, pending: PendingImport, url: string, fetchImpl: FetchLike, limits: Limits): Promise<ImportOutcome> {
  const warnings: string[] = [];
  const applied = { projects: 0, files: 0, settings: false, removed: 0 };
  const added: string[] = [];
  const archive = await download(desktop, token, pending, url, fetchImpl);

  // What is in it, checked before anything is unpacked (fflate reports each entry's sizes first). It lists every record of
  // the archive's directory, so a name that is listed twice is seen twice, and refused: which of the two a program reads
  // is not the same for every program.
  const directory = new Map<string, number>();
  let count = 0;
  let total = 0;
  unzipSync(archive, {
    filter: (file) => {
      count++;
      if (count > limits.entries) throw new Error(`the archive holds more than ${limits.entries.toLocaleString()} entries`);
      total += file.originalSize;
      if (total > limits.unpacked) throw new Error(`the archive unpacks to more than ${Math.round(limits.unpacked / 2 ** 20)} MiB`);
      if (directory.has(file.name)) throw new Error(`the archive lists ${clip(file.name)} twice, so it was not restored`);
      directory.set(file.name, file.originalSize);
      return false;
    },
  });

  // Its own record says what it is and what it brings back; without one of this version, nothing is restored.
  const recordSize = directory.get(RECORD);
  if (recordSize === undefined) throw new Error('this is not an archive of the Agent’s storage that this version can restore: it has no record of what it brings back');
  if (recordSize > MAX_RECORD_BYTES) throw new Error('the archive’s record is larger than this version reads, so it was not restored');
  let record: unknown;
  try {
    record = JSON.parse(new TextDecoder().decode(unzipSync(archive, { filter: (f) => f.name === RECORD })[RECORD]));
  } catch {
    throw new Error('the archive’s record is not one this version can read, so it was not restored');
  }
  const named = itemsOf(record, directory, limits);

  // Only what the record names is ever written. Anything else in the archive is left where it is.
  const nameSet = new Set(named.map((i) => i.name));
  const strays = [...directory.keys()].filter((n) => n !== RECORD && !n.endsWith('/') && !nameSet.has(n));
  for (const stray of strays.slice(0, 5)) warnings.push(`${clip(stray)} was in the archive and not named by its record, so it was not restored.`);
  if (strays.length > 5) warnings.push(`${strays.length - 5} more files in the archive were not named by its record, so they were not restored.`);
  const items: Item[] = [];
  for (const item of named) {
    if (item.size > limits.entry) warnings.push(`${clip(item.name)} was not restored: it is bigger than ${Math.round(limits.entry / 2 ** 20)} MiB.`);
    else items.push(item);
  }

  // Files to take away (what the restore being undone added), checked like an item that comes in. The numbers not to be
  // contacted are never taken away: they only ever grow.
  const removals: Array<{ name: string; path: string }> = [];
  for (const name of pending.remove) {
    const check = checkName(name);
    if (check.ok && check.where === 'opfs' && name === DO_NOT_CONTACT) warnings.push(`${DO_NOT_CONTACT} was kept: the numbers not to be contacted are never taken away.`);
    else if (check.ok && check.where === 'opfs') removals.push({ name, path: check.path });
    else warnings.push(`${clip(name)} was not taken away: ${check.ok ? 'it is not a file of the Agent’s projects or front desk' : check.why}.`);
  }

  // Projects that must not be written to or taken from: the archive's or this page's own project.json says incognito, or
  // cannot be read (never a guess: a project.json that cannot be read, or is too large to be one, keeps its project out).
  const blocked = new Map<string, Incognito>();
  const block = (id: string, state: Incognito) => {
    if (state !== 'no' && !blocked.has(id)) blocked.set(id, state);
  };
  const idOf = (path: string) => (path.startsWith('projects/') ? path.split('/')[1] : null);
  const opfs = items.filter((i) => i.where === 'opfs');
  const metaEntries = opfs.filter((l) => /^projects\/[^/]+\/project\.json$/.test(l.path));
  const small = metaEntries.filter((l) => l.size <= MAX_PROJECT_JSON);
  for (const l of metaEntries) if (l.size > MAX_PROJECT_JSON) block(l.path.split('/')[1], 'unreadable');
  if (small.length) {
    const names = new Set(small.map((l) => l.name));
    const metas = unzipSync(archive, { filter: (f) => names.has(f.name) });
    for (const l of small) block(l.path.split('/')[1], metas[l.name] ? incognitoOf(metas[l.name]) : 'unreadable');
  }
  const touched = new Set([...opfs.map((l) => idOf(l.path)), ...removals.map((r) => idOf(r.path))].filter((id): id is string => id !== null));
  for (const id of touched) if (!blocked.has(id)) block(id, await projectState(storage, `projects/${id}`));
  for (const [id, state] of blocked) {
    warnings.push(state === 'yes' ? `Project ${id} was not restored: it is (or was) an incognito project, which is never kept.` : `Project ${id} was skipped: its project.json could not be read.`);
  }
  const isBlocked = (path: string) => blocked.has(idOf(path) ?? '');
  let writes = opfs.filter((l) => !isBlocked(l.path));
  const takeAway = removals.filter((r) => !isBlocked(r.path));

  // A campaign that is here already is kept exactly as it is: an older backup of your own must never bring one back to
  // life that you have since paused or finished. (Asked before anything is copied or written.)
  const keptCampaigns = new Set<string>();
  for (const item of writes) {
    if (item.mode === 'campaign' && (await storage.exists(item.path))) {
      keptCampaigns.add(item.name);
      warnings.push(`${clip(item.name)} was not restored: a campaign of the same id is already here, and is kept exactly as it is.`);
    }
  }
  writes = writes.filter((l) => !keptCampaigns.has(l.name));

  // The undo copy comes first, for an undo as well as a restore (an undo can be undone), so that nothing is
  // replaced or taken away before there is a way back. (If a copy of an earlier try is there, the desktop keeps that one.)
  {
    const target = uploadTo(desktop, token, `${url}/undo-part`, `${url}/undo-done`, fetchImpl);
    // Without the API keys: they were sealed here with a key the browser will not let out, and the undo copy is
    // kept by the desktop as plain files. The merge rule never replaces a stored key with an empty one, so
    // an undo does not need them (it also does not roll back keys that a restore brought in).
    const copy = await exportAgentStorage(storage, target, { includeKeys: false, limits });
    if (!copy.ok) throw new Error(`the undo copy could not be made, so nothing was restored (${copy.error ?? 'unknown reason'})`);
    const left = new Set(copy.skipped);
    const clash = [...writes, ...takeAway].filter((w) => left.has(w.path));
    if (clash.length) throw new Error(`the undo copy could not include ${clash.length} file${clash.length === 1 ? '' : 's'} that this would replace or take away, so nothing was changed`);
    if (copy.warnings.length) warnings.push(...copy.warnings.map((w) => `Undo copy: ${w}`));
  }

  // What the restore had added is taken away (files only, never a folder).
  for (const item of takeAway) {
    try {
      if (await storage.removeFile(item.path)) applied.removed++;
    } catch (e) {
      warnings.push(`${item.path} could not be taken away: ${message(e)}.`);
    }
  }

  // Files, a few at a time, so that only a part of the archive is unpacked at once. The list of campaigns goes last: it
  // names only campaigns that are here.
  const ordered = [...writes.filter((w) => w.mode !== 'campaign-index'), ...writes.filter((w) => w.mode === 'campaign-index')];
  const written = new Set<string>();
  let failed = 0;
  let unlisted = 0;
  for (let from = 0; from < ordered.length; ) {
    let to = from;
    let bytes = 0;
    while (to < ordered.length && to - from < BATCH_ENTRIES && (to === from || bytes + ordered[to].size <= BATCH_BYTES)) bytes += ordered[to++].size;
    const names = new Set(ordered.slice(from, to).map((w) => w.name));
    const files = unzipSync(archive, { filter: (f) => names.has(f.name) });
    for (const item of ordered.slice(from, to)) {
      let data: Uint8Array | undefined = files[item.name];
      if (!data) {
        failed++;
        continue;
      }
      // How it comes back is the record's, in the one way its name may.
      if (item.mode === 'union' || item.mode === 'campaign-index') {
        const made = item.mode === 'union' ? await unionDoNotContact(storage, item, data) : await joinCampaignIndex(storage, item, data);
        if (made.warn) warnings.push(made.warn);
        if (made.failed) failed++;
        if (!made.data) continue;
        data = made.data;
      } else if (item.mode === 'campaign') {
        const paused = pausedCampaign(data);
        if (!paused) {
          warnings.push(`${clip(item.name)} was not restored: it is not a campaign.`);
          continue;
        }
        data = paused;
      }
      try {
        const existed = await storage.exists(item.path);
        await storage.writeFile(item.path, data);
        applied.files++;
        if (!existed) {
          if (added.length < MAX_ADDED) added.push(item.name);
          else unlisted++;
        }
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
  if (unlisted) warnings.push(`${unlisted} added file${unlisted === 1 ? ' is' : 's are'} not listed, so an undo will not take ${unlisted === 1 ? 'it' : 'them'} away.`);

  // The settings only when the record names them (the desktop names them only for what the person ticked, and only the
  // keys that may come back), and then with the merge rule above.
  const settingsItem = items.find((l) => l.mode === 'settings');
  if (settingsItem) {
    try {
      const raw = unzipSync(archive, { filter: (f) => f.name === settingsItem.name })[settingsItem.name];
      const parsed: unknown = JSON.parse(new TextDecoder().decode(raw));
      if (!isObject(parsed)) throw new Error('it is not a settings file');
      const blockedIds = new Set(blocked.keys());
      await storage.writeSettings(mergeSettings(await storage.readSettings(), parsed as Partial<BackupSettings>, blockedIds, { keys: pending.apply.keys, notes: warnings, exact: pending.kind === 'undo' }));
      applied.settings = true;
    } catch (e) {
      failed++;
      warnings.push(`The settings could not be restored: ${message(e)}.`);
    }
  }
  return failed ? { ok: false, error: `${failed} item${failed === 1 ? '' : 's'} could not be restored`, applied, warnings, added } : { ok: true, applied, warnings, added };
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
      // Only "not there" is null: a file that cannot be read throws, so a caller that must not guess (the incognito check) does not.
      try {
        const slash = path.lastIndexOf('/');
        const dir = await dirAt(path.slice(0, slash), false);
        return new Uint8Array(await (await (await dir.getFileHandle(path.slice(slash + 1))).getFile()).arrayBuffer());
      } catch (error) {
        if ((error as DOMException | undefined)?.name === 'NotFoundError') return null;
        throw error;
      }
    },
    async exists(path) {
      try {
        const slash = path.lastIndexOf('/');
        const dir = await dirAt(path.slice(0, slash), false);
        await dir.getFileHandle(path.slice(slash + 1));
        return true;
      } catch {
        return false;
      }
    },
    async removeFile(path) {
      const slash = path.lastIndexOf('/');
      let dir: Dir;
      try {
        dir = await dirAt(path.slice(0, slash), false);
      } catch (error) {
        if ((error as DOMException | undefined)?.name === 'NotFoundError') return false;
        throw error;
      }
      try {
        await dir.getFileHandle(path.slice(slash + 1));
      } catch (error) {
        // Not there, or a folder (never removed): nothing removed.
        const name = (error as DOMException | undefined)?.name;
        if (name === 'NotFoundError' || name === 'TypeMismatchError') return false;
        throw error;
      }
      await dir.removeEntry(path.slice(slash + 1));
      return true;
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
