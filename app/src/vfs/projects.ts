/**
 * Projects on this device, kept in the browser's Origin Private File System:
 *
 *   /projects/<id>/project.json   name, created, updated
 *   /projects/<id>/files/...      the project's files, mirrored
 *   /projects/<id>/chat.json      the conversation
 *
 * The in-memory Vfs is the source of truth while a project is open; every
 * change is written behind to OPFS (a short debounce, one write per path).
 *
 * An incognito project is a temporary one: its files and conversation are kept
 * only in its own folder here (so it survives a refresh or a restart), and the
 * folder is deleted when incognito is cleared or turned off. Nothing of it goes
 * anywhere else; exporting it is the only way to keep it.
 *
 * The front desk is kept beside the projects, not among them (`/front-desk/`,
 * the same layout): the phone's conversations (each call and text thread, and
 * flows' tasks) and the files they read. It stays open whatever project the
 * person works in, and it is not in the project menu, so it cannot be renamed,
 * deleted or left behind by incognito.
 */
import type { Callback } from '../callbacks';
import type { Campaign, DoNotContact } from '../outreach';
import type { Turn } from '../agent/protocol';
import { Vfs, type VfsChange } from './vfs';

export interface ProjectMeta {
  id: string;
  name: string;
  created: number;
  updated: number;
  /** A temporary project: nothing of it is kept (see the module note). */
  incognito?: boolean;
}

async function root(): Promise<FileSystemDirectoryHandle> {
  const opfs = await navigator.storage.getDirectory();
  return opfs.getDirectoryHandle('projects', { create: true });
}

async function dirAt(base: FileSystemDirectoryHandle, path: string, create: boolean): Promise<FileSystemDirectoryHandle> {
  let dir = base;
  for (const seg of path.split('/').filter(Boolean)) {
    try {
      dir = await dir.getDirectoryHandle(seg, { create });
    } catch (error) {
      // A file where a folder now belongs (its removal never reached the disk): replace it.
      if (!create || (error as DOMException).name !== 'TypeMismatchError') throw error;
      await dir.removeEntry(seg);
      dir = await dir.getDirectoryHandle(seg, { create: true });
    }
  }
  return dir;
}

async function readJson<T>(dir: FileSystemDirectoryHandle, name: string): Promise<T | null> {
  try {
    const file = await (await dir.getFileHandle(name)).getFile();
    return JSON.parse(await file.text()) as T;
  } catch {
    return null;
  }
}

async function writeBytes(dir: FileSystemDirectoryHandle, name: string, data: Uint8Array | string): Promise<void> {
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  await writable.write(typeof data === 'string' ? data : new Blob([data as BlobPart]));
  await writable.close();
}

/**
 * A conversation of a project besides its own chat: a text-message thread with
 * one person (`key` their number), or a call. Its turns are kept beside it.
 */
export interface SessionInfo {
  id: string;
  /** A text thread, a call, or a flow's tasks for the agent. */
  kind: 'sms' | 'call' | 'task';
  /** Who it is with: a phone number (a flow's tasks: the flow's name). */
  key: string;
  /** Their name, when the phone knows it; else the number. */
  title: string;
  /** When it last heard or said something (ms). */
  lastAt: number;
  /** Messages the person has not looked at yet. */
  unread: number;
  /** The phone's handles of the texts it has had (the phone may deliver one again, when it reconnects). */
  handles?: string[];
  /**
   * The conversation it is part of: a person's calls and texts are one (its
   * turns kept together, in `sessions/<thread>.json`); a flow's tasks are
   * their own. None: kept before a person's were one (its turns in
   * `sessions/<id>.json`, merged into their conversation on the next start).
   */
  thread?: string;
  /** A call from a hidden number: its own conversation, never another caller's. */
  hidden?: boolean;
  /** The person took this sender up (wrote in the conversation, or texted them for an outreach): they are answered whatever their first text carried. */
  vouched?: boolean;
}

/**
 * What the phone's agents know about one person who calls or texts: their
 * calls and texts share it (their numbers agree by the last nine digits).
 */
export interface CallerNote {
  /** Their number, as the phone gave it. */
  number: string;
  name?: string;
  /** Short facts worth knowing next time, oldest first: what the receptionist (or the runner) remembered. */
  facts: string[];
  /** When it last changed (ms). */
  updatedAt: number;
  /**
   * The rest is their contact on OAIY Desktop as it was last read (contacts.ts):
   * who named them there (the person's own name, `owner`, is never replaced
   * by one the agents learn), the person's notes for the receptionist, and
   * the facts the person wrote themselves.
   */
  nameBy?: 'owner' | 'agent';
  notes?: string;
  ownerFacts?: string[];
  /** Facts remembered while OAIY Desktop could not be reached: sent to it when it can be. */
  unsent?: string[];
}

/**
 * The facts in callers.json moved to OAIY Desktop's contacts, once: when, how
 * many were sent, how many it had already, those it refused (and why), and
 * callers.json as it was before.
 */
export interface ContactsMoved {
  at: number;
  sent: number;
  there: number;
  skipped: Array<{ number: string; fact: string; why: string }>;
  callers: CallerNote[];
}

/** A session id as a file name. */
function safeName(id: string): string {
  return id.replace(/[^\w.-]+/g, '_');
}

/** The front desk's place and name (see the module note). */
export const FRONT_DESK = { id: 'front-desk', name: 'Front desk' } as const;
/** What the front desk is for, in its files. */
export const FRONT_DESK_NOTE = '/knowledge/README.md';
/** The runner's direction to the phone's agents: every call, text and task reads it before each reply. */
export const FRONT_DESK_BRIEF = '/brief.md';
const FRONT_DESK_BRIEF_TEXT = `# The brief

- Nothing special for now.
`;
const FRONT_DESK_README = `# The front desk

The phone's agents work here: each call, text-message thread and flow task is
answered by a sub-agent in a conversation of its own, and they all take their
direction from \`/brief.md\`, which the Front desk's main agent keeps (open the
Front desk in the project list and tell it what the phone should know, like
"this week, tell callers we're booked until Friday" or "don't quote prices for
jobs over a day"; it writes that into the brief, which is only ever what the
phone goes by right now).

Put what they should know about your business in this \`knowledge\` folder, as
plain text or Markdown files: what you offer and what it costs, your area,
directions, what to say about common questions, what not to promise. It reads
these files when a call or a text needs them; callers and texters cannot
change them. Your hours and services come from the Calendar.
`;

/**
 * "Set up OAIY": the person's conversation with the Agent about OAIY itself
 * (what it should do, and setting that up through OAIY Desktop's control
 * API). A project among the others, found by its fixed id, so its
 * conversation is kept like any project's; it is not renamed or deleted.
 */
export const SETUP_PROJECT = { id: 'oaiy-setup', name: 'Set up OAIY' } as const;
/** What the "Set up OAIY" project is for, in its files. */
export const SETUP_NOTE = '/README.md';
export const SETUP_README = `# Set up OAIY

Here the Agent sets OAIY up with you: tell it what you want OAIY to do (answer
your business phone, run flows, make pictures and music…), and it checks what
this computer has, recommends the next steps and does them, through OAIY
Desktop's control API.

- What you must do yourself (accept what a plugin may do, pair a phone with a
  code) it shows you in OAIY's dashboard, then checks it is done.
- Every change it makes is listed in OAIY's Settings → Agent → What the Agent
  changed. The switch there ("Let the Agent set up and change OAIY for you")
  turns its changes off; it can still look.
`;

/** The "Set up OAIY" project: made on first use. */
export async function setupProject(): Promise<ProjectMeta> {
  const dir = await (await root()).getDirectoryHandle(SETUP_PROJECT.id, { create: true });
  let meta = await readJson<ProjectMeta>(dir, 'project.json');
  if (!meta) {
    meta = { ...SETUP_PROJECT, created: Date.now(), updated: Date.now() };
    await dir.getDirectoryHandle('files', { create: true });
    await writeBytes(dir, 'project.json', JSON.stringify(meta));
  }
  return { ...meta, id: SETUP_PROJECT.id };
}

export function newId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export async function listProjects(): Promise<ProjectMeta[]> {
  const base = await root();
  const out: ProjectMeta[] = [];
  for await (const [name, handle] of (base as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
    if (handle.kind !== 'directory') continue;
    const meta = await readJson<ProjectMeta>(handle as FileSystemDirectoryHandle, 'project.json');
    if (meta) out.push({ ...meta, id: name });
  }
  return out.sort((a, b) => b.updated - a.updated);
}

export async function createProject(name: string, incognito = false): Promise<ProjectMeta> {
  const base = await root();
  const meta: ProjectMeta = { id: newId(), name, created: Date.now(), updated: Date.now(), ...(incognito ? { incognito: true } : {}) };
  const dir = await base.getDirectoryHandle(meta.id, { create: true });
  await dir.getDirectoryHandle('files', { create: true });
  await writeBytes(dir, 'project.json', JSON.stringify(meta));
  return meta;
}

export async function deleteProject(id: string): Promise<void> {
  await (await root()).removeEntry(id, { recursive: true });
}

/** Delete the incognito projects left behind (a closed app, a crash), all but `keep`. */
export async function clearIncognito(keep?: string): Promise<void> {
  for (const meta of await listProjects()) {
    if (meta.incognito && meta.id !== keep) await deleteProject(meta.id).catch(() => {});
  }
}

export async function renameProject(meta: ProjectMeta, name: string): Promise<ProjectMeta> {
  const dir = await (await root()).getDirectoryHandle(meta.id);
  const next = { ...meta, name, updated: Date.now() };
  await writeBytes(dir, 'project.json', JSON.stringify(next));
  return next;
}

async function readTree(dir: FileSystemDirectoryHandle, prefix: string, files: Array<[string, Uint8Array]>, dirs: string[]): Promise<void> {
  for await (const [name, handle] of (dir as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'directory') {
      dirs.push(path);
      await readTree(handle as FileSystemDirectoryHandle, path, files, dirs);
    } else {
      const file = await (handle as FileSystemFileHandle).getFile();
      files.push([path, new Uint8Array(await file.arrayBuffer())]);
    }
  }
}

/** An open project: its files in memory, written behind to OPFS. */
export class OpenProject {
  readonly vfs = new Vfs();
  private pending = new Map<string, VfsChange>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private flushing: Promise<void> = Promise.resolve();
  private unsubscribe: (() => void) | null = null;
  /** Errors writing behind, for the UI. */
  onError: (message: string) => void = () => {};

  private constructor(public meta: ProjectMeta, private readonly dir: FileSystemDirectoryHandle) {}

  static async open(meta: ProjectMeta): Promise<OpenProject> {
    return OpenProject.openAt(meta, await (await root()).getDirectoryHandle(meta.id, { create: true }));
  }

  /**
   * The front desk (see the module note), made on first use with a note on
   * what to put in it, and given the phone conversations a project held
   * before it existed (see `adoptSessions`).
   */
  static async openFrontDesk(): Promise<OpenProject> {
    const opfs = await navigator.storage.getDirectory();
    const dir = await opfs.getDirectoryHandle(FRONT_DESK.id, { create: true });
    const meta = (await readJson<ProjectMeta>(dir, 'project.json')) ?? { ...FRONT_DESK, created: Date.now(), updated: Date.now() };
    const fresh = !(await readJson<ProjectMeta>(dir, 'project.json'));
    if (fresh) await writeBytes(dir, 'project.json', JSON.stringify(meta));
    const desk = await OpenProject.openAt(meta, dir);
    if (fresh || !desk.vfs.exists(FRONT_DESK_NOTE)) desk.vfs.writeFile(FRONT_DESK_NOTE, FRONT_DESK_README, { parents: true });
    if (!desk.vfs.exists(FRONT_DESK_BRIEF)) desk.vfs.writeFile(FRONT_DESK_BRIEF, FRONT_DESK_BRIEF_TEXT, { parents: true });
    await desk.flush();
    if (fresh) await desk.adoptSessions();
    return desk;
  }

  /**
   * Phone conversations kept in a project before the front desk existed come
   * here (from the project that has them, the latest if several do). They are
   * copied: the project's own copies stay where they were, unread.
   */
  private async adoptSessions(): Promise<void> {
    if ((await this.loadSessions()).length) return;
    for (const meta of await listProjects()) {
      const from = await (await root()).getDirectoryHandle(meta.id).catch(() => null);
      const sessions = from && (await dirAt(from, 'sessions', false).catch(() => null));
      const index = sessions && (await readJson<SessionInfo[]>(sessions, 'index.json'));
      if (!sessions || !index?.length) continue;
      const to = await dirAt(this.dir, 'sessions', true);
      for await (const [name, handle] of (sessions as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
        if (handle.kind !== 'file') continue;
        const file = await (handle as FileSystemFileHandle).getFile();
        await writeBytes(to, name, new Uint8Array(await file.arrayBuffer()));
      }
      return;
    }
  }

  private static async openAt(meta: ProjectMeta, dir: FileSystemDirectoryHandle): Promise<OpenProject> {
    const project = new OpenProject(meta, dir);
    const files: Array<[string, Uint8Array]> = [];
    const dirs: string[] = [];
    await readTree(await dir.getDirectoryHandle('files', { create: true }), '', files, dirs);
    project.vfs.load(files, dirs);
    project.unsubscribe = project.vfs.onChange((change) => project.queue(change));
    return project;
  }

  private queue(change: VfsChange): void {
    if (change.type === 'reset') {
      // A whole new tree (an import): rewrite everything.
      this.pending.clear();
      this.pending.set('\u0000reset', change);
    } else {
      this.pending.set(change.path, change);
    }
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => void this.flush(), 300);
  }

  /** Whether changes are waiting to be written. */
  get dirty(): boolean {
    return this.pending.size > 0 || this.timer !== null;
  }

  private retries = 0;
  private lastErrorAt = 0;

  /** Write everything pending; resolves when it is on disk (or has failed and is queued again). */
  flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const batch = [...this.pending.values()];
    this.pending.clear();
    if (!batch.length) return this.flushing;
    this.flushing = this.flushing.then(async () => {
      const failed = await this.write(batch);
      if (!failed.length) {
        this.retries = 0;
        return;
      }
      // Keep what did not reach the disk (unless something newer replaced it) and try again shortly.
      for (const { change } of failed) if (change.type === 'reset' || !this.pending.has(change.path)) this.pending.set(change.type === 'reset' ? '\u0000reset' : change.path, change);
      this.retries++;
      if (this.retries > 3 && Date.now() - this.lastErrorAt < 60_000) return;
      this.lastErrorAt = Date.now();
      this.onError(`Saving ${failed.length === 1 ? failed[0].label : `${failed.length} files`} to this browser's storage failed: ${failed[0].error}.${this.retries <= 3 ? ' Trying again.' : ' Export the project to keep a copy.'}`);
      if (this.retries <= 3 && !this.timer) this.timer = setTimeout(() => void this.flush(), 2000 * this.retries);
    });
    return this.flushing;
  }

  /**
   * Bring the disk in line with the files in memory for each changed path.
   * Each path is written as it is NOW (a file, a folder, or gone), whatever
   * the change said, so a change that was replaced before it was written
   * cannot leave the disk out of step. Returns the paths that failed.
   */
  private async write(batch: VfsChange[]): Promise<Array<{ change: VfsChange; label: string; error: string }>> {
    const failed: Array<{ change: VfsChange; label: string; error: string }> = [];
    const files = await this.dir.getDirectoryHandle('files', { create: true });
    for (const change of batch) {
      try {
        if (change.type === 'reset') {
          await this.dir.removeEntry('files', { recursive: true }).catch(() => {});
          const fresh = await this.dir.getDirectoryHandle('files', { create: true });
          for (const [path, data] of this.vfs.files()) {
            const slash = path.lastIndexOf('/');
            await writeBytes(await dirAt(fresh, slash < 0 ? '' : path.slice(0, slash), true), path.slice(slash + 1), data);
          }
          for (const e of this.vfs.walk('/', { includeIgnored: true }).entries) if (e.type === 'dir') await dirAt(fresh, e.path, true);
          continue;
        }
        const slash = change.path.lastIndexOf('/');
        const parentPath = slash < 0 ? '' : change.path.slice(0, slash);
        const name = change.path.slice(slash + 1);
        const now = this.vfs.stat(`/${change.path}`);
        if (!now) {
          const parent = await dirAt(files, parentPath, false).catch(() => null);
          await parent?.removeEntry(name, { recursive: true }).catch(() => {});
        } else if (now.type === 'dir') {
          await dirAt(files, change.path, true);
        } else {
          const parent = await dirAt(files, parentPath, true);
          // A folder where a file now belongs: replace it.
          await parent.getDirectoryHandle(name).then(() => parent.removeEntry(name, { recursive: true }), () => {});
          await writeBytes(parent, name, this.vfs.readBytes(`/${change.path}`));
        }
      } catch (error) {
        failed.push({ change, label: change.type === 'reset' ? 'the project' : change.path, error: (error as Error).message || String(error) });
      }
    }
    this.meta = { ...this.meta, updated: Date.now() };
    await writeBytes(this.dir, 'project.json', JSON.stringify(this.meta)).catch(() => {});
    return failed;
  }

  async loadChat(): Promise<Turn[]> {
    return (await readJson<Turn[]>(this.dir, 'chat.json')) ?? [];
  }

  /** The conversation, kept with the project (an incognito one's is deleted with it). */
  async saveChat(turns: Turn[]): Promise<void> {
    await writeBytes(this.dir, 'chat.json', JSON.stringify(turns));
  }

  /** Missed calls to ring back, and the last few settled (the front desk's). */
  async loadCallbacks(): Promise<Callback[]> {
    return (await readJson<Callback[]>(this.dir, 'callbacks.json')) ?? [];
  }

  async saveCallbacks(list: Callback[]): Promise<void> {
    await writeBytes(this.dir, 'callbacks.json', JSON.stringify(list));
  }

  /**
   * Outreach campaigns (the front desk's): one file each in `outreach/`, with
   * the list of them in `outreach/index.json`. Kept beside the files, not
   * among them (their results are written into the files, under /outreach).
   */
  async loadOutreach(): Promise<Campaign[]> {
    const dir = await dirAt(this.dir, 'outreach', false).catch(() => null);
    if (!dir) return [];
    const ids = (await readJson<string[]>(dir, 'index.json')) ?? [];
    const out: Campaign[] = [];
    for (const id of ids) {
      const c = await readJson<Campaign>(dir, `${safeName(id)}.json`);
      if (c?.id) out.push(c);
    }
    return out;
  }

  async saveOutreach(campaign: Campaign, ids: string[]): Promise<void> {
    const dir = await dirAt(this.dir, 'outreach', true);
    await writeBytes(dir, `${safeName(campaign.id)}.json`, JSON.stringify(campaign));
    await writeBytes(dir, 'index.json', JSON.stringify(ids));
  }

  /** Numbers not to be called or texted again (they asked): outreach and call backs keep off them. */
  async loadDoNotContact(): Promise<DoNotContact[]> {
    const dir = await dirAt(this.dir, 'outreach', false).catch(() => null);
    return (dir && (await readJson<DoNotContact[]>(dir, 'do-not-contact.json'))) ?? [];
  }

  async saveDoNotContact(list: DoNotContact[]): Promise<void> {
    await writeBytes(await dirAt(this.dir, 'outreach', true), 'do-not-contact.json', JSON.stringify(list));
  }

  /** What the phone's agents know about the people who call and text (the front desk's). */
  async loadCallers(): Promise<CallerNote[]> {
    return (await readJson<CallerNote[]>(this.dir, 'callers.json')) ?? [];
  }

  async saveCallers(list: CallerNote[]): Promise<void> {
    await writeBytes(this.dir, 'callers.json', JSON.stringify(list));
  }

  /** Whether (and how) the facts in callers.json were moved to OAIY Desktop's contacts: null until they were. */
  async loadContactsMoved(): Promise<ContactsMoved | null> {
    return readJson<ContactsMoved>(this.dir, 'contacts-moved.json');
  }

  async saveContactsMoved(mark: ContactsMoved): Promise<void> {
    await writeBytes(this.dir, 'contacts-moved.json', JSON.stringify(mark));
  }

  /** The project's other conversations (a text-message thread, a call): what each is, newest first. */
  async loadSessions(): Promise<SessionInfo[]> {
    const dir = await dirAt(this.dir, 'sessions', false).catch(() => null);
    return (dir && (await readJson<SessionInfo[]>(dir, 'index.json'))) ?? [];
  }

  async saveSessions(list: SessionInfo[]): Promise<void> {
    await writeBytes(await dirAt(this.dir, 'sessions', true), 'index.json', JSON.stringify(list));
  }

  async loadSessionChat(id: string): Promise<Turn[]> {
    const dir = await dirAt(this.dir, 'sessions', false).catch(() => null);
    return (dir && (await readJson<Turn[]>(dir, `${safeName(id)}.json`))) ?? [];
  }

  async saveSessionChat(id: string, turns: Turn[]): Promise<void> {
    await writeBytes(await dirAt(this.dir, 'sessions', true), `${safeName(id)}.json`, JSON.stringify(turns));
  }

  /** A conversation's turns gone (kept in a backup first, when they matter). */
  async removeSessionChat(id: string): Promise<void> {
    const dir = await dirAt(this.dir, 'sessions', false).catch(() => null);
    await dir?.removeEntry(`${safeName(id)}.json`).catch(() => {});
  }

  /**
   * A copy of the conversations as they are (the list, each one's turns, what
   * is known about callers, the calls to ring back), in `<name>/` beside them,
   * before they are changed. A copy made already under that name is kept: the
   * new one goes under `<name>-2` (and so on). Answers with the name used.
   */
  async backupSessions(name: string): Promise<string> {
    let used = name;
    for (let n = 2; await this.dir.getDirectoryHandle(used).then(() => true, () => false); n++) used = `${name}-${n}`;
    const backup = await this.dir.getDirectoryHandle(used, { create: true });
    const copy = async (from: FileSystemDirectoryHandle, to: FileSystemDirectoryHandle, file: string) => {
      const handle = await from.getFileHandle(file).catch(() => null);
      if (handle) await writeBytes(to, file, new Uint8Array(await (await handle.getFile()).arrayBuffer()));
    };
    for (const file of ['callers.json', 'callbacks.json']) await copy(this.dir, backup, file);
    const sessions = await dirAt(this.dir, 'sessions', false).catch(() => null);
    if (sessions) {
      const to = await backup.getDirectoryHandle('sessions', { create: true });
      for await (const [file, handle] of (sessions as unknown as AsyncIterable<[string, FileSystemHandle]>)) if (handle.kind === 'file') await copy(sessions, to, file);
    }
    return used;
  }

  async close(): Promise<void> {
    await this.flush();
    this.unsubscribe?.();
  }
}
