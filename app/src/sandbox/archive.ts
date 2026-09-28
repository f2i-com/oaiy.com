/**
 * Archives for the sandbox shell's zip, unzip, tar, gzip and gunzip: the
 * page does the (de)compression with fflate, over the project's files, and
 * answers the guest's `fs.archive` host call. Paths never leave the project
 * (the VFS normalises every one), and unpacking is checked against the
 * import limits before anything is inflated.
 */
import { gunzipSync, gzipSync, strFromU8, strToU8, zipSync } from 'fflate';
import { unzipChecked } from '../vfs/transfer';
import { normalizePath, type Vfs } from '../vfs/vfs';

export interface ArchiveRequest {
  op: 'zip' | 'unzip' | 'tar' | 'untar' | 'gzip' | 'gunzip';
  /** The archive (or the file to compress), absolute in the project. */
  file: string;
  /** zip and tar: what to pack, relative to `base` (a folder in the project). */
  paths?: string[];
  base?: string;
  /** unzip and untar: where to unpack. */
  dest?: string;
  /** unzip, untar: only list; unzip: only these entries. */
  list?: boolean;
  only?: string[];
  overwrite?: boolean;
  /** tar: gzip it too (tar -z), or detect on untar. */
  gzip?: boolean;
  /** gzip/gunzip: keep the input. */
  keep?: boolean;
  /** gzip/gunzip: the result as text (to stdout), nothing written. */
  stdout?: boolean;
  /** zip, tar: follow folders (always, here). */
  recursive?: boolean;
}

export interface ArchiveResult {
  files: Array<{ path: string; size: number }>;
  /** gzip/gunzip -c: the result as base64. */
  data?: string;
  skipped?: string[];
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/** The files under `paths` (each a file or folder, relative to `base`), keyed as given. */
function collect(vfs: Vfs, base: string, paths: string[]): Array<[string, Uint8Array]> {
  const out: Array<[string, Uint8Array]> = [];
  for (const raw of paths) {
    const rel = raw.replace(/\/+$/, '');
    const abs = `/${normalizePath(`${base}/${rel}`)}`;
    const st = vfs.stat(abs);
    if (!st) throw new Error(`${raw}: No such file or directory`);
    if (st.type === 'file') {
      out.push([normalizePath(rel), vfs.readBytes(abs)]);
      continue;
    }
    const prefix = normalizePath(rel);
    for (const e of vfs.walk(abs, { includeIgnored: true }).entries) {
      const inside = e.path.slice(normalizePath(abs).length).replace(/^\//, '');
      const key = prefix ? `${prefix}/${inside}` : inside;
      if (e.type === 'file') out.push([key, vfs.readBytes(`/${e.path}`)]);
      else out.push([`${key}/`, new Uint8Array(0)]);
    }
  }
  return out;
}

/* ---------- tar (ustar) ---------- */

function tarPack(entries: Array<[string, Uint8Array]>): Uint8Array {
  const blocks: Uint8Array[] = [];
  const now = Math.floor(Date.now() / 1000);
  for (const [name, data] of entries) {
    const dir = name.endsWith('/');
    const header = new Uint8Array(512);
    const put = (text: string, at: number, len: number) => header.set(strToU8(text).subarray(0, len), at);
    let path = name;
    let prefix = '';
    if (path.length > 100) {
      const cut = path.lastIndexOf('/', path.length - 101 >= 0 ? 155 : path.length);
      prefix = path.slice(0, cut);
      path = path.slice(cut + 1);
    }
    put(path, 0, 100);
    put(dir ? '0000755\0' : '0000644\0', 100, 8);
    put('0001750\0', 108, 8);
    put('0001750\0', 116, 8);
    put(`${(dir ? 0 : data.byteLength).toString(8).padStart(11, '0')}\0`, 124, 12);
    put(`${now.toString(8).padStart(11, '0')}\0`, 136, 12);
    put('        ', 148, 8);
    header[156] = dir ? 53 : 48;
    put('ustar\0', 257, 6);
    put('00', 263, 2);
    put('sandbox', 265, 32);
    put('sandbox', 297, 32);
    put(prefix, 345, 155);
    let sum = 0;
    for (const b of header) sum += b;
    put(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 8);
    blocks.push(header);
    if (!dir) {
      blocks.push(data);
      const pad = (512 - (data.byteLength % 512)) % 512;
      if (pad) blocks.push(new Uint8Array(pad));
    }
  }
  blocks.push(new Uint8Array(1024));
  const total = blocks.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const b of blocks) {
    out.set(b, at);
    at += b.byteLength;
  }
  return out;
}

function tarUnpack(bytes: Uint8Array): Array<[string, Uint8Array]> {
  const out: Array<[string, Uint8Array]> = [];
  const text = (at: number, len: number) => strFromU8(bytes.subarray(at, at + len)).replace(/\0.*$/s, '');
  let at = 0;
  let longName: string | null = null;
  let total = 0;
  while (at + 512 <= bytes.byteLength) {
    const name = text(at, 100);
    if (!name && bytes[at] === 0) break;
    const size = parseInt(text(at + 124, 12).trim() || '0', 8);
    const type = String.fromCharCode(bytes[at + 156] || 48);
    const prefix = text(at + 345, 155);
    const data = bytes.subarray(at + 512, at + 512 + size);
    total += size;
    if (total > 512 << 20) throw new Error('the archive unpacks to more than 512 MB');
    let full = longName ?? (prefix ? `${prefix}/${name}` : name);
    longName = null;
    if (type === 'L') longName = strFromU8(data).replace(/\0.*$/s, '');
    else if (type === '5') out.push([full.endsWith('/') ? full : `${full}/`, new Uint8Array(0)]);
    else if (type === '0' || type === '\0' || type === '7') out.push([full, data.slice()]);
    full = '';
    at += 512 + Math.ceil(size / 512) * 512;
  }
  return out;
}

function safe(path: string): string | null {
  const clean = path.replace(/^\.\//, '');
  if (!clean || clean.split('/').includes('..') || clean.startsWith('/')) return null;
  return clean;
}

function extract(vfs: Vfs, entries: Array<[string, Uint8Array]>, dest: string, overwrite: boolean, only?: string[]): ArchiveResult {
  const files: ArchiveResult['files'] = [];
  const skipped: string[] = [];
  for (const [raw, data] of entries) {
    const name = safe(raw);
    if (!name) {
      skipped.push(raw);
      continue;
    }
    if (only?.length && !only.some((o) => name === o || name.startsWith(`${o.replace(/\/$/, '')}/`))) continue;
    const target = `/${normalizePath(`${dest}/${name}`)}`;
    if (name.endsWith('/')) {
      vfs.mkdir(target, true);
      continue;
    }
    if (!overwrite && vfs.exists(target)) {
      skipped.push(name);
      continue;
    }
    vfs.writeFile(target, data, { parents: true });
    files.push({ path: target, size: data.byteLength });
  }
  return { files, skipped };
}

export function archive(vfs: Vfs, req: ArchiveRequest): ArchiveResult {
  const base = req.base ?? '/';
  switch (req.op) {
    case 'zip': {
      const entries = collect(vfs, base, req.paths ?? []);
      const tree: Record<string, Uint8Array> = {};
      for (const [name, data] of entries) tree[name] = data;
      vfs.writeFile(req.file, zipSync(tree, { level: 6 }), { parents: true });
      return { files: entries.filter(([n]) => !n.endsWith('/')).map(([path, data]) => ({ path, size: data.byteLength })) };
    }
    case 'unzip': {
      const entries = Object.entries(unzipChecked(vfs.readBytes(req.file)));
      if (req.list) return { files: entries.map(([path, data]) => ({ path, size: data.byteLength })) };
      return extract(vfs, entries, req.dest ?? base, req.overwrite !== false, req.only);
    }
    case 'tar': {
      const entries = collect(vfs, base, req.paths ?? []);
      const tar = tarPack(entries);
      vfs.writeFile(req.file, req.gzip ? gzipSync(tar) : tar, { parents: true });
      return { files: entries.map(([path, data]) => ({ path, size: data.byteLength })) };
    }
    case 'untar': {
      let bytes = vfs.readBytes(req.file);
      if (req.gzip || (bytes[0] === 0x1f && bytes[1] === 0x8b)) bytes = gunzipSync(bytes);
      const entries = tarUnpack(bytes);
      if (req.list) return { files: entries.map(([path, data]) => ({ path, size: data.byteLength })) };
      return extract(vfs, entries, req.dest ?? base, req.overwrite !== false, req.only);
    }
    case 'gzip': {
      const data = gzipSync(vfs.readBytes(req.file), { level: 6, filename: req.file.split('/').pop() });
      if (req.stdout) return { files: [], data: bytesToBase64(data) };
      const target = `${req.file}.gz`;
      if (vfs.exists(target) && req.overwrite === false) throw new Error(`${target} already exists`);
      vfs.writeFile(target, data);
      if (!req.keep) vfs.remove(req.file);
      return { files: [{ path: target, size: data.byteLength }] };
    }
    case 'gunzip': {
      const data = gunzipSync(vfs.readBytes(req.file));
      if (data.byteLength > 512 << 20) throw new Error('the file unpacks to more than 512 MB');
      if (req.stdout) return { files: [], data: bytesToBase64(data) };
      const target = req.file.replace(/\.(gz|tgz|z)$/i, (m) => (m.toLowerCase() === '.tgz' ? '.tar' : ''));
      if (target === req.file) throw new Error(`${req.file}: unknown suffix -- ignored`);
      vfs.writeFile(target, data);
      if (!req.keep) vfs.remove(req.file);
      return { files: [{ path: target, size: data.byteLength }] };
    }
    default:
      throw new Error(`unknown archive operation ${(req as { op: string }).op}`);
  }
}
