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
 */
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
  kind: 'sms' | 'call';
  /** Who it is with: a phone number. */
  key: string;
  /** Their name, when the phone knows it; else the number. */
  title: string;
  /** When it last heard or said something (ms). */
  lastAt: number;
  /** Messages the person has not looked at yet. */
  unread: number;
}

/** A session id as a file name. */
function safeName(id: string): string {
  return id.replace(/[^\w.-]+/g, '_');
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
    const dir = await (await root()).getDirectoryHandle(meta.id, { create: true });
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

  async close(): Promise<void> {
    await this.flush();
    this.unsubscribe?.();
  }
}
