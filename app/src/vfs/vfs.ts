/**
 * The project's virtual filesystem: an in-memory tree the agent, the editor
 * and the sandbox all share. Paths are POSIX, rooted at "/" (the project
 * root). There is nothing outside the tree, so `..` above the root simply
 * stays at the root: confinement by construction, not by checking.
 *
 * Persistence is someone else's job (see opfs.ts): it listens for changes
 * and writes them behind.
 */

export type EntryKind = 'file' | 'dir';

export interface Stat {
  type: EntryKind;
  size: number;
  mtimeMs: number;
}

export interface DirEntry extends Stat {
  name: string;
}

export interface WalkEntry extends Stat {
  /** Project-relative, without a leading slash. */
  path: string;
}

export type VfsChange =
  | { type: 'write'; path: string }
  | { type: 'mkdir'; path: string }
  | { type: 'remove'; path: string; kind: EntryKind }
  | { type: 'reset' };

interface FileNode {
  kind: 'file';
  data: Uint8Array;
  mtimeMs: number;
  /** Bumped on every write; the agent's read-before-edit rule uses it. */
  version: number;
}

interface DirNode {
  kind: 'dir';
  mtimeMs: number;
}

type Node = FileNode | DirNode;

export class VfsError extends Error {
  constructor(readonly code: 'ENOENT' | 'EEXIST' | 'EISDIR' | 'ENOTDIR' | 'ENOTEMPTY' | 'EINVAL' | 'EFBIG', message: string) {
    super(`${code}: ${message}`);
  }
}

/** Directories a project walk skips unless asked for them. */
export const IGNORED_DIRS = new Set(['.git', 'node_modules', 'target', 'dist', 'build', '.venv', 'venv', '__pycache__', '.cache', '.next', '.turbo']);

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });

/** Normalize to a key: no leading slash, `.`/`..` resolved, clamped at the root. */
export function normalizePath(path: string, cwd = '/'): string {
  const raw = String(path).replace(/\\/g, '/');
  if (raw.includes('\0')) throw new VfsError('EINVAL', 'path contains a NUL character');
  const joined = raw.startsWith('/') ? raw : `${cwd.replace(/\/+$/, '')}/${raw}`;
  const out: string[] = [];
  for (const seg of joined.split('/')) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..') {
      out.pop();
      continue;
    }
    out.push(seg);
  }
  return out.join('/');
}

export function parentOf(key: string): string {
  const i = key.lastIndexOf('/');
  return i < 0 ? '' : key.slice(0, i);
}

export function baseName(key: string): string {
  const i = key.lastIndexOf('/');
  return i < 0 ? key : key.slice(i + 1);
}

export class Vfs {
  private nodes = new Map<string, Node>([['', { kind: 'dir', mtimeMs: Date.now() }]]);
  private children = new Map<string, Set<string>>([['', new Set()]]);
  private listeners = new Set<(change: VfsChange) => void>();
  /** Every write gets the next number, so a version is never reused. */
  private clock = 0;

  onChange(listener: (change: VfsChange) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(change: VfsChange): void {
    for (const listener of this.listeners) listener(change);
  }

  private node(key: string): Node | undefined {
    return this.nodes.get(key);
  }

  private link(key: string): void {
    const parent = parentOf(key);
    let set = this.children.get(parent);
    if (!set) {
      set = new Set();
      this.children.set(parent, set);
    }
    set.add(key);
  }

  private unlink(key: string): void {
    this.children.get(parentOf(key))?.delete(key);
  }

  exists(path: string): boolean {
    return this.nodes.has(normalizePath(path));
  }

  stat(path: string): Stat | null {
    const node = this.node(normalizePath(path));
    if (!node) return null;
    return { type: node.kind, size: node.kind === 'file' ? node.data.byteLength : 0, mtimeMs: node.mtimeMs };
  }

  version(path: string): number {
    const node = this.node(normalizePath(path));
    return node?.kind === 'file' ? node.version : 0;
  }

  readBytes(path: string): Uint8Array {
    const key = normalizePath(path);
    const node = this.node(key);
    if (!node) throw new VfsError('ENOENT', `no such file or directory, '/${key}'`);
    if (node.kind === 'dir') throw new VfsError('EISDIR', `illegal operation on a directory, read '/${key}'`);
    return node.data;
  }

  /** UTF-8 text, or an error for binary content. */
  readText(path: string): string {
    const bytes = this.readBytes(path);
    try {
      return decoder.decode(bytes);
    } catch {
      throw new VfsError('EINVAL', `'/${normalizePath(path)}' is not UTF-8 text`);
    }
  }

  isText(path: string): boolean {
    try {
      decoder.decode(this.readBytes(path).subarray(0, 8192));
      return true;
    } catch {
      return false;
    }
  }

  /** Create parents as needed when `parents`, else require the parent. */
  writeFile(path: string, data: string | Uint8Array, options: { parents?: boolean; append?: boolean } = {}): void {
    const key = normalizePath(path);
    if (key === '') throw new VfsError('EISDIR', 'illegal operation on a directory, write \'/\'');
    const existing = this.node(key);
    if (existing?.kind === 'dir') throw new VfsError('EISDIR', `illegal operation on a directory, write '/${key}'`);
    const parent = parentOf(key);
    const parentNode = this.node(parent);
    if (!parentNode) {
      if (!options.parents) throw new VfsError('ENOENT', `no such directory for '/${key}' (create it first, e.g. mkdir -p)`);
      this.mkdir(`/${parent}`, true);
    } else if (parentNode.kind !== 'dir') {
      throw new VfsError('ENOTDIR', `not a directory, '/${parent}'`);
    }
    let bytes = typeof data === 'string' ? encoder.encode(data) : data;
    if (options.append && existing?.kind === 'file') {
      const joined = new Uint8Array(existing.data.byteLength + bytes.byteLength);
      joined.set(existing.data, 0);
      joined.set(bytes, existing.data.byteLength);
      bytes = joined;
    }
    const version = ++this.clock;
    this.nodes.set(key, { kind: 'file', data: bytes, mtimeMs: Date.now(), version });
    this.link(key);
    this.emit({ type: 'write', path: key });
  }

  mkdir(path: string, recursive = false): void {
    const key = normalizePath(path);
    if (key === '') {
      if (!recursive) throw new VfsError('EEXIST', 'file already exists, mkdir \'/\'');
      return;
    }
    const existing = this.node(key);
    if (existing) {
      if (existing.kind === 'dir' && recursive) return;
      throw new VfsError('EEXIST', `file already exists, mkdir '/${key}'`);
    }
    const parent = parentOf(key);
    const parentNode = this.node(parent);
    if (!parentNode) {
      if (!recursive) throw new VfsError('ENOENT', `no such file or directory, mkdir '/${key}'`);
      this.mkdir(`/${parent}`, true);
    } else if (parentNode.kind !== 'dir') {
      throw new VfsError('ENOTDIR', `not a directory, '/${parent}'`);
    }
    this.nodes.set(key, { kind: 'dir', mtimeMs: Date.now() });
    this.children.set(key, this.children.get(key) ?? new Set());
    this.link(key);
    this.emit({ type: 'mkdir', path: key });
  }

  list(path: string): DirEntry[] {
    const key = normalizePath(path);
    const node = this.node(key);
    if (!node) throw new VfsError('ENOENT', `no such file or directory, scandir '/${key}'`);
    if (node.kind !== 'dir') throw new VfsError('ENOTDIR', `not a directory, scandir '/${key}'`);
    const out: DirEntry[] = [];
    for (const child of this.children.get(key) ?? []) {
      const c = this.nodes.get(child);
      if (!c) continue;
      out.push({ name: baseName(child), type: c.kind, size: c.kind === 'file' ? c.data.byteLength : 0, mtimeMs: c.mtimeMs });
    }
    return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  }

  /** Every entry below `path`, depth-first in name order. */
  walk(path = '/', options: { includeIgnored?: boolean; limit?: number } = {}): { entries: WalkEntry[]; truncated: boolean } {
    const start = normalizePath(path);
    const node = this.node(start);
    if (!node) throw new VfsError('ENOENT', `no such file or directory, '/${start}'`);
    const limit = options.limit ?? 50_000;
    const entries: WalkEntry[] = [];
    if (node.kind === 'file') {
      entries.push({ path: start, type: 'file', size: node.data.byteLength, mtimeMs: node.mtimeMs });
      return { entries, truncated: false };
    }
    const stack = [start];
    let truncated = false;
    while (stack.length) {
      const dir = stack.pop()!;
      const kids = [...(this.children.get(dir) ?? [])].sort();
      const subdirs: string[] = [];
      for (const child of kids) {
        const c = this.nodes.get(child);
        if (!c) continue;
        if (c.kind === 'dir' && !options.includeIgnored && IGNORED_DIRS.has(baseName(child))) continue;
        if (entries.length >= limit) {
          truncated = true;
          break;
        }
        entries.push({ path: child, type: c.kind, size: c.kind === 'file' ? c.data.byteLength : 0, mtimeMs: c.mtimeMs });
        if (c.kind === 'dir') subdirs.push(child);
      }
      if (truncated) break;
      for (let i = subdirs.length - 1; i >= 0; i--) stack.push(subdirs[i]);
    }
    return { entries, truncated };
  }

  remove(path: string, recursive = false): void {
    const key = normalizePath(path);
    if (key === '') throw new VfsError('EINVAL', 'refusing to remove the project root');
    const node = this.node(key);
    if (!node) throw new VfsError('ENOENT', `no such file or directory, '/${key}'`);
    if (node.kind === 'dir') {
      const kids = this.children.get(key) ?? new Set();
      if (kids.size && !recursive) throw new VfsError('ENOTEMPTY', `directory not empty, '/${key}'`);
      for (const child of [...kids]) this.remove(`/${child}`, true);
      this.children.delete(key);
    }
    this.nodes.delete(key);
    this.unlink(key);
    this.emit({ type: 'remove', path: key, kind: node.kind });
  }

  rename(from: string, to: string): void {
    const a = normalizePath(from);
    const b = normalizePath(to);
    if (a === '') throw new VfsError('EINVAL', 'refusing to move the project root');
    if (b === a) return;
    if (b.startsWith(`${a}/`)) throw new VfsError('EINVAL', `cannot move '/${a}' into itself`);
    this.copy(from, to);
    this.remove(from, true);
  }

  copy(from: string, to: string): void {
    const a = normalizePath(from);
    const b = normalizePath(to);
    const node = this.node(a);
    if (!node) throw new VfsError('ENOENT', `no such file or directory, '/${a}'`);
    if (b === a || b.startsWith(`${a}/`)) throw new VfsError('EINVAL', `cannot copy '/${a}' into itself`);
    if (node.kind === 'file') {
      this.writeFile(`/${b}`, node.data.slice(), { parents: false });
      return;
    }
    this.mkdir(`/${b}`, true);
    for (const entry of this.walk(`/${a}`, { includeIgnored: true }).entries) {
      const target = `/${b}${entry.path.slice(a.length)}`;
      if (entry.type === 'dir') this.mkdir(target, true);
      else this.writeFile(target, this.readBytes(`/${entry.path}`).slice());
    }
  }

  /** Replace the whole tree (loading a project). Emits one `reset`. */
  load(files: Iterable<[string, Uint8Array]>, dirs: Iterable<string> = []): void {
    this.nodes = new Map([['', { kind: 'dir', mtimeMs: Date.now() }]]);
    this.children = new Map([['', new Set()]]);
    const quiet = this.listeners;
    this.listeners = new Set();
    try {
      for (const dir of dirs) this.mkdir(`/${dir}`, true);
      for (const [path, data] of files) this.writeFile(`/${path}`, data, { parents: true });
    } finally {
      this.listeners = quiet;
    }
    this.emit({ type: 'reset' });
  }

  /** Every file, for export. */
  files(): Array<[string, Uint8Array]> {
    const out: Array<[string, Uint8Array]> = [];
    for (const [key, node] of this.nodes) if (node.kind === 'file') out.push([key, node.data]);
    return out.sort((x, y) => (x[0] < y[0] ? -1 : 1));
  }

  totalBytes(): number {
    let total = 0;
    for (const node of this.nodes.values()) if (node.kind === 'file') total += node.data.byteLength;
    return total;
  }
}
