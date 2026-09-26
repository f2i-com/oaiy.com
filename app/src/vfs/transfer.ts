/**
 * Getting files in and out: a folder from disk (File System Access API, or a
 * directory <input> where that is missing), a .zip, and a .zip download.
 */
import { unzipSync, zipSync } from 'fflate';
import { IGNORED_DIRS } from './vfs';

export const MAX_IMPORT_BYTES = 256 << 20;
export const MAX_IMPORT_FILES = 20_000;

export interface Imported {
  name: string;
  files: Array<[string, Uint8Array]>;
  skipped: string[];
}

function budget() {
  let bytes = 0;
  let count = 0;
  return (size: number) => {
    if (count + 1 > MAX_IMPORT_FILES || bytes + size > MAX_IMPORT_BYTES) return false;
    count++;
    bytes += size;
    return true;
  };
}

export function canPickFolder(): boolean {
  return typeof (window as unknown as { showDirectoryPicker?: unknown }).showDirectoryPicker === 'function';
}

/** Read a folder the user picks, skipping dependency/build folders. */
export async function importFolder(): Promise<Imported | null> {
  const picker = (window as unknown as { showDirectoryPicker: (o?: unknown) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker;
  let dir: FileSystemDirectoryHandle;
  try {
    dir = await picker({ mode: 'read' });
  } catch {
    return null;
  }
  const take = budget();
  const files: Array<[string, Uint8Array]> = [];
  const skipped: string[] = [];
  const walk = async (handle: FileSystemDirectoryHandle, prefix: string) => {
    for await (const [name, child] of (handle as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
      const path = prefix ? `${prefix}/${name}` : name;
      if (child.kind === 'directory') {
        if (IGNORED_DIRS.has(name)) {
          skipped.push(`${path}/`);
          continue;
        }
        await walk(child as FileSystemDirectoryHandle, path);
      } else {
        const file = await (child as FileSystemFileHandle).getFile();
        if (!take(file.size)) {
          skipped.push(path);
          continue;
        }
        files.push([path, new Uint8Array(await file.arrayBuffer())]);
      }
    }
  };
  await walk(dir, '');
  return { name: dir.name, files, skipped };
}

/** The same from an <input type=file webkitdirectory> selection. */
export async function importFileList(list: FileList): Promise<Imported> {
  const take = budget();
  const files: Array<[string, Uint8Array]> = [];
  const skipped: string[] = [];
  let name = 'project';
  for (const file of Array.from(list)) {
    const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    const parts = rel.split('/');
    if (parts.length > 1) name = parts[0];
    const inner = parts.length > 1 ? parts.slice(1) : parts;
    if (inner.slice(0, -1).some((seg) => IGNORED_DIRS.has(seg))) continue;
    if (!take(file.size)) {
      skipped.push(inner.join('/'));
      continue;
    }
    files.push([inner.join('/'), new Uint8Array(await file.arrayBuffer())]);
  }
  return { name, files, skipped };
}

export async function importZip(file: File): Promise<Imported> {
  const entries = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const names = Object.keys(entries).filter((n) => !n.endsWith('/'));
  // A zip of one folder: drop that folder from every path.
  const firstSeg = names[0]?.split('/')[0];
  const common = firstSeg && names.every((n) => n.startsWith(`${firstSeg}/`)) ? `${firstSeg}/` : '';
  const take = budget();
  const files: Array<[string, Uint8Array]> = [];
  const skipped: string[] = [];
  for (const name of names) {
    const path = name.slice(common.length);
    if (!path || path.split('/').some((seg) => seg === '..' || seg === '')) continue;
    if (path.split('/').slice(0, -1).some((seg) => IGNORED_DIRS.has(seg))) continue;
    if (!take(entries[name].byteLength)) {
      skipped.push(path);
      continue;
    }
    files.push([path, entries[name]]);
  }
  return { name: (common.replace(/\/$/, '') || file.name.replace(/\.zip$/i, '')) || 'project', files, skipped };
}

export function downloadZip(name: string, files: Array<[string, Uint8Array]>): void {
  const tree: Record<string, Uint8Array> = {};
  for (const [path, data] of files) tree[`${name}/${path}`] = data;
  const zipped = zipSync(tree, { level: 6 });
  const url = URL.createObjectURL(new Blob([zipped as BlobPart], { type: 'application/zip' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name}.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
