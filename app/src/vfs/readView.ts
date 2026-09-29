/**
 * A view of a project's files with some of them hidden: the front desk's
 * outreach results (`/outreach`, one customer's answers) are the runner's, and
 * the agents that talk with callers, texters and flows must not read them to
 * anyone else. The view answers as if the hidden paths were not there, and
 * refuses to write under them; everything else reads and writes through to
 * the files themselves (a flow's task may still write its own files).
 */
import { VfsError, normalizePath, type DirEntry, type Stat, type Vfs, type WalkEntry } from './vfs';

/** The front desk's outreach results: hidden from its calls, texts and tasks. */
export const isOutreachPath = (key: string): boolean => key === 'outreach' || key.startsWith('outreach/');

export function readView(vfs: Vfs, hidden: (key: string) => boolean = isOutreachPath): Vfs {
  const key = (path: string) => normalizePath(path);
  const gone = (path: string, what = 'no such file or directory'): never => {
    throw new VfsError('ENOENT', `${what}, '/${key(path)}'`);
  };
  const refuse = (path: string): never => {
    throw new VfsError('EINVAL', `'/${key(path)}' is private to the runner: it cannot be changed from here`);
  };
  const view: Partial<Record<keyof Vfs, unknown>> = {
    exists: (path: string) => !hidden(key(path)) && vfs.exists(path),
    stat: (path: string): Stat | null => (hidden(key(path)) ? null : vfs.stat(path)),
    version: (path: string) => (hidden(key(path)) ? 0 : vfs.version(path)),
    readBytes: (path: string) => (hidden(key(path)) ? gone(path) : vfs.readBytes(path)),
    readText: (path: string) => (hidden(key(path)) ? gone(path) : vfs.readText(path)),
    isText: (path: string) => !hidden(key(path)) && vfs.isText(path),
    list: (path: string): DirEntry[] => {
      const dir = key(path);
      if (hidden(dir)) gone(path, 'no such file or directory, scandir');
      return vfs.list(path).filter((e) => !hidden(dir ? `${dir}/${e.name}` : e.name));
    },
    walk: (path = '/', options: { includeIgnored?: boolean; limit?: number } = {}): { entries: WalkEntry[]; truncated: boolean } => {
      if (hidden(key(path))) gone(path);
      const limit = options.limit ?? 50_000;
      const all = vfs.walk(path, { ...options, limit: Number.MAX_SAFE_INTEGER }).entries.filter((e) => !hidden(e.path));
      return { entries: all.slice(0, limit), truncated: all.length > limit };
    },
    files: () => vfs.files().filter(([path]) => !hidden(path)),
    writeFile: (path: string, data: string | Uint8Array, options?: { parents?: boolean; append?: boolean }) => (hidden(key(path)) ? refuse(path) : vfs.writeFile(path, data, options)),
    mkdir: (path: string, recursive?: boolean) => (hidden(key(path)) ? refuse(path) : vfs.mkdir(path, recursive)),
    remove: (path: string, recursive?: boolean) => (hidden(key(path)) ? refuse(path) : vfs.remove(path, recursive)),
    rename: (from: string, to: string) => (hidden(key(from)) ? refuse(from) : hidden(key(to)) ? refuse(to) : vfs.rename(from, to)),
    copy: (from: string, to: string) => (hidden(key(from)) ? gone(from) : hidden(key(to)) ? refuse(to) : vfs.copy(from, to)),
    load: () => {
      throw new VfsError('EINVAL', 'the files cannot be replaced from here');
    },
  };
  return new Proxy(vfs, {
    get(target, prop) {
      if (typeof prop === 'string' && prop in view) return view[prop as keyof Vfs];
      const value = Reflect.get(target, prop) as unknown;
      return typeof value === 'function' ? (value as (...args: unknown[]) => unknown).bind(target) : value;
    },
  });
}
