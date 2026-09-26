/**
 * Projects on this device, kept in the browser's Origin Private File System:
 *
 *   /projects/<id>/project.json   name, created, updated
 *   /projects/<id>/files/...      the project's files, mirrored
 *   /projects/<id>/chat.json      the conversation
 *
 * The in-memory Vfs is the source of truth while a project is open; every
 * change is written behind to OPFS (a short debounce, one write per path).
 */
import type { Turn } from '../agent/protocol';
import { Vfs, type VfsChange } from './vfs';

export interface ProjectMeta {
  id: string;
  name: string;
  created: number;
  updated: number;
}

async function root(): Promise<FileSystemDirectoryHandle> {
  const opfs = await navigator.storage.getDirectory();
  return opfs.getDirectoryHandle('projects', { create: true });
}

async function dirAt(base: FileSystemDirectoryHandle, path: string, create: boolean): Promise<FileSystemDirectoryHandle> {
  let dir = base;
  for (const seg of path.split('/').filter(Boolean)) dir = await dir.getDirectoryHandle(seg, { create });
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

export async function createProject(name: string): Promise<ProjectMeta> {
  const base = await root();
  const meta: ProjectMeta = { id: newId(), name, created: Date.now(), updated: Date.now() };
  const dir = await base.getDirectoryHandle(meta.id, { create: true });
  await dir.getDirectoryHandle('files', { create: true });
  await writeBytes(dir, 'project.json', JSON.stringify(meta));
  return meta;
}

export async function deleteProject(id: string): Promise<void> {
  await (await root()).removeEntry(id, { recursive: true });
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

  /** Write everything pending; resolves when it is on disk. */
  flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const batch = [...this.pending.values()];
    this.pending.clear();
    if (!batch.length) return this.flushing;
    this.flushing = this.flushing.then(() => this.write(batch)).catch((error: unknown) => this.onError(`Saving the project failed: ${(error as Error).message}`));
    return this.flushing;
  }

  private async write(batch: VfsChange[]): Promise<void> {
    const files = await this.dir.getDirectoryHandle('files', { create: true });
    for (const change of batch) {
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
      if (change.type === 'remove') {
        const parent = await dirAt(files, parentPath, false).catch(() => null);
        await parent?.removeEntry(name, { recursive: true }).catch(() => {});
      } else if (change.type === 'mkdir') {
        await dirAt(files, change.path, true);
      } else if (this.vfs.stat(`/${change.path}`)?.type === 'file') {
        await writeBytes(await dirAt(files, parentPath, true), name, this.vfs.readBytes(`/${change.path}`));
      }
    }
    this.meta = { ...this.meta, updated: Date.now() };
    await writeBytes(this.dir, 'project.json', JSON.stringify(this.meta));
  }

  async loadChat(): Promise<Turn[]> {
    return (await readJson<Turn[]>(this.dir, 'chat.json')) ?? [];
  }

  async saveChat(turns: Turn[]): Promise<void> {
    await writeBytes(this.dir, 'chat.json', JSON.stringify(turns));
  }

  async close(): Promise<void> {
    await this.flush();
    this.unsubscribe?.();
  }
}
