/**
 * SoftN apps in an OAIY project: finding them, starting one, checking
 * one, importing a .softn, and bundling one as a .softn file.
 *
 * A SoftN app is a folder of text files — manifest.json, its pages, its logic,
 * xdb/*.xdb seed records, assets/ — and a .softn file is that folder zipped
 * flat, manifest.json at the root. A new app is written as HTML: `index.html`
 * (and `pages/*.html`) naming its code with `<script src="main.js">` (or
 * `main.py`). An app made before has `ui/*.ui` pages and `logic/*.logic` (or
 * `.py`) files, and runs the same: the runtime finds a page or its logic by
 * what names it (`main`, `<import from>`, `<script src>`), not by its
 * extension. A project may hold several apps, each in its own folder (the
 * project root is a folder too): any folder whose manifest.json's `main` is a
 * page. Paths inside an app are relative to its folder. The rules here follow
 * softn.com's Studio (exportBundle.ts planBundle, sourceKinds.ts, validator.ts)
 * and core's composer (source-composer.ts, module-syntax.ts), Apache-2.0.
 */
import { zipSync } from 'fflate';
import { unzipChecked } from '../vfs/transfer';
import { IGNORED_DIRS, normalizePath, type Vfs } from '../vfs/vfs';
import { SOFTN_GUIDE_JAVASCRIPT, SOFTN_GUIDE_PYTHON } from './guide.generated';
import { NetGate } from '../gate/netgate';
import { SandboxHost } from '../sandbox/host';
import { runInSandbox } from '../sandbox/runner';

export interface SoftnManifest {
  name?: unknown;
  version?: unknown;
  main?: unknown;
  entry?: unknown;
  files?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Finding {
  level: 'error' | 'warning';
  file: string;
  message: string;
}

const TEXT_EXT = /\.(ui|html?|logic|js|py|xdb|json|md|txt|css|csv|svg)$/i;
const CAPABILITIES = ['net', 'camera', 'mic', 'files', 'qr', 'ai', 'gpu', 'sync', 'storage', 'accel'];

/** A new app's main page, where a site's first page is (`ui/main.ui` in an app made before). */
export const MAIN_PAGE = 'index.html';

/**
 * A page of an app: `.ui`, or `.html` / `.htm` as a page written the way an HTML page is written, outside `assets/`
 * (a file there is one the app shows or serves, so `assets/help.html` is no page).
 */
export function isAppPagePath(path: string): boolean {
  return /\.(ui|html?)$/i.test(path) && !/^assets\//i.test(path);
}

/** A `<script type>` that is the app's code: none, or JavaScript's or Python's (core's scriptLanguage). */
const CODE_TYPES = new Set(['', 'module', 'javascript', 'text/javascript', 'application/javascript', 'text/ecmascript', 'application/ecmascript', 'python', 'text/python', 'text/x-python', 'application/python', 'application/x-python']);

/** An attribute's value in a tag's attribute text, quoted or not; undefined when it is not written. */
function attributeValue(attributes: string, name: string): string | undefined {
  const m = new RegExp(String.raw`(?:^|\s)${name}\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>\x60]+))`, 'i').exec(attributes);
  return m ? (m[1] ?? m[2] ?? m[3]) : undefined;
}

/**
 * The files of code a page names, as written, in order: `<script src="main.js">`, and `<logic src>` as a page made
 * before writes it. Passed over: what is in an HTML comment, a script that is data (`type="application/json"`), one
 * from another site, and the runtime's own loader (`softn.js`).
 */
export function pageLogicSources(page: string): string[] {
  const out: string[] = [];
  const text = page.replace(/<!--[\s\S]*?(?:-->|$)/g, '');
  for (const m of text.matchAll(/<(script|logic)(?=[\s/>])((?:[^>"']|"[^"]*"|'[^']*')*)>/gi)) {
    const src = attributeValue(m[2], 'src')?.trim();
    if (!src) continue;
    if (m[1].toLowerCase() === 'script' && (!CODE_TYPES.has((attributeValue(m[2], 'type') ?? '').trim().toLowerCase()) || /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(src) || /(?:^|\/)softn\.js$/i.test(src.split(/[?#]/)[0]))) continue;
    out.push(src);
  }
  return out;
}

/** `relative` from the file at `base`, as a path of the app; null when it would leave the app or names none. */
function resolveInApp(base: string, relative: string): string | null {
  const rel = relative.replace(/\\/g, '/');
  if (rel.startsWith('/') || /^[A-Za-z]:\//.test(rel)) return null;
  const parts = base.split('/').slice(0, -1);
  for (const part of rel.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      if (!parts.length) return null;
      parts.pop();
    } else parts.push(part);
  }
  return parts.length ? parts.join('/') : null;
}

/**
 * Logic written as a JavaScript module, read as the script SoftN runs (core's moduleSyntaxAsScript): an app's logic is
 * one script, so `export` before a declaration is dropped, `export { a }` is nothing, and `import { a } from "./b.js"`
 * is `import "./b.js"`, which places that file's code. Line for line, so a line is still the author's.
 */
export function moduleSyntaxAsScript(code: string): string {
  if (!/\b(?:export|import)\b/.test(code)) return code;
  return code
    .replace(/^([ \t]*)export[ \t]+default[ \t]+(?=(?:async[ \t]+)?function[ \t]*\*?[ \t]*[A-Za-z_$]|class[ \t]+[A-Za-z_$])/gm, '$1')
    .replace(/^([ \t]*)export[ \t]+(?=(?:async[ \t]+)?function\b|class\b|const\b|let\b|var\b)/gm, '$1')
    .replace(/^([ \t]*)export[ \t]+default[ \t]+[A-Za-z_$][\w$]*[ \t]*;?[ \t]*$/gm, '$1')
    .replace(/^([ \t]*)export[ \t]*\{[^}\n]*\}[ \t]*;?[ \t]*$/gm, '$1')
    .replace(/^([ \t]*)(?:import|export)[ \t]*\{([^}\n]*)\}[ \t]*from[ \t]*(["'])((?:\.{1,2})?\/[^"'\n]+)\3[ \t]*;?[ \t]*$/gm, (line, indent: string, names: string, quote: string, path: string) =>
      /\bas\b/.test(names) ? line : `${indent}import ${quote}${path}${quote}`,
    );
}

/** A whole-file `import "./x.js"` line of logic; group 2 is the path as written. */
const LOGIC_IMPORT_LINE = /^[ \t]*import[ \t]+(["'])([^"'\r\n]+)\1[ \t]*;?[ \t]*\r?$/gm;

/** The app paths the logic file at `path` imports: from that file for `./` and `../`, from the app's root otherwise. */
export function logicImportPaths(source: string, path: string): string[] {
  const out: string[] = [];
  for (const m of moduleSyntaxAsScript(source).matchAll(LOGIC_IMPORT_LINE)) {
    const raw = m[2];
    if (/^https?:\/\//i.test(raw)) continue;
    const resolved = raw.startsWith('./') || raw.startsWith('../') ? resolveInApp(path, raw) : resolveInApp('x', raw);
    if (resolved) out.push(resolved);
  }
  return out;
}

/**
 * The `.js` files that are the app's code: those its pages link with `<script src>` or `<logic src>`, and what those
 * import, followed. A `.js` nothing links is one the app carries (a library a page loads for itself, a service worker),
 * not code it runs (Studio's linkedScripts).
 */
export function linkedScripts(texts: ReadonlyMap<string, string>): Set<string> {
  const linked = new Set<string>();
  const follow = (path: string): void => {
    if (linked.has(path) || !/\.js$/i.test(path)) return;
    const text = texts.get(path);
    if (text === undefined) return;
    linked.add(path);
    for (const imported of logicImportPaths(text, path)) follow(imported);
  };
  for (const [path, text] of texts) {
    if (!isAppPagePath(path)) continue;
    for (const src of pageLogicSources(text)) {
      const resolved = resolveInApp(path, src);
      if (resolved) follow(resolved);
    }
  }
  return linked;
}

/** The app's pages and `.js` files as text, keyed by their path in the app (what linkedScripts reads). */
function scriptTexts(files: Array<[string, Uint8Array]>): Map<string, string> {
  const texts = new Map<string, string>();
  const decoder = new TextDecoder();
  for (const [path, data] of files) if (isAppPagePath(path) || /\.js$/i.test(path)) texts.set(path, decoder.decode(data));
  return texts;
}

/** The manifest's paths under `group`, as strings. */
function listedIn(manifest: SoftnManifest | null, group: string): string[] {
  const list = manifest?.files?.[group];
  return Array.isArray(list) ? list.filter((p): p is string => typeof p === 'string') : [];
}

type Group = 'ui' | 'logic' | 'server' | 'xdb' | 'assets';

/**
 * Each file's group in the manifest, as Studio's export files it: a page is `ui` (unless the manifest lists it under
 * `assets`); `.logic` and `.py` are `logic`, and a `.js` when a page links it, logic imports it or the manifest lists
 * it there; server logic is `server`; seed records `xdb`; and `assets` is the assets/ folder and what the manifest
 * lists there. Anything else is in no group.
 */
function fileGroups(files: Array<[string, Uint8Array]>, manifest: SoftnManifest | null): Map<string, Group> {
  const linked = linkedScripts(scriptTexts(files));
  const listedAssets = new Set(listedIn(manifest, 'assets'));
  const listedLogic = new Set(listedIn(manifest, 'logic'));
  const groups = new Map<string, Group>();
  for (const [path] of files) {
    const server = path.startsWith('server/');
    let group: Group | null = null;
    if (server && /\.(logic|py)$/i.test(path)) group = 'server';
    else if (isAppPagePath(path) && !listedAssets.has(path)) group = 'ui';
    else if (!server && (linked.has(path) || (!listedAssets.has(path) && (/\.(logic|py)$/i.test(path) || (/\.js$/i.test(path) && listedLogic.has(path)))))) group = 'logic';
    else if (/\.xdb$/i.test(path)) group = 'xdb';
    else if (path.startsWith('assets/') || listedAssets.has(path)) group = 'assets';
    if (group) groups.set(path, group);
  }
  return groups;
}

/** An app folder as a key: '' is the project root, else `a/b` (no slashes at the ends). */
export function appKey(folder: string | undefined | null): string {
  return normalizePath(folder ?? '/');
}

function inApp(root: string, rel: string): string {
  return `/${root ? `${root}/` : ''}${rel}`;
}

/** How an app folder reads in messages. */
export function appLabel(root: string): string {
  return root ? `${root}/` : 'the project root';
}

export function readManifest(vfs: Vfs, root = ''): SoftnManifest | null {
  try {
    const parsed = JSON.parse(vfs.readText(inApp(root, 'manifest.json'))) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as SoftnManifest) : null;
  } catch {
    return null;
  }
}

/** Is this folder a SoftN app? (A manifest.json whose `main` is a page: `index.html`, or a .ui page made before.) */
export function isSoftnApp(vfs: Vfs, root = ''): boolean {
  const manifest = readManifest(vfs, root);
  return !!manifest && typeof manifest.main === 'string' && isAppPagePath(manifest.main);
}

/** Every SoftN app folder in the project, the root first. */
export function findApps(vfs: Vfs): string[] {
  const roots: string[] = [];
  if (isSoftnApp(vfs, '')) roots.push('');
  for (const entry of vfs.walk('/', { limit: 20_000 }).entries) {
    if (entry.type !== 'file' || !entry.path.endsWith('/manifest.json')) continue;
    const root = entry.path.slice(0, -'/manifest.json'.length);
    if (isSoftnApp(vfs, root)) roots.push(root);
  }
  return roots;
}

export function isSoftnProject(vfs: Vfs): boolean {
  return findApps(vfs).length > 0;
}

/**
 * The app whose page the project file at `path` is (no leading slash), or null: an app's `index.html` is a page of the
 * app, shown as the app, not a web page of the project. The innermost app holding the file is its app, and a file it
 * keeps out of itself (assets/, builder/, the project's uploads/) is no page of it.
 */
export function softnAppOf(vfs: Vfs, path: string, apps = findApps(vfs)): string | null {
  const file = path.replace(/^\/+/, '');
  const root = apps.filter((r) => r === '' || file.startsWith(`${r}/`)).sort((a, b) => b.length - a.length)[0];
  if (root === undefined) return null;
  const rel = root ? file.slice(root.length + 1) : file;
  const first = rel.split('/')[0];
  if (first === 'builder' || (root === '' && first === 'uploads')) return null;
  return isAppPagePath(rel) ? root : null;
}

/** `text` as a sentence starts: its first letter a capital. */
const sentence = (text: string): string => text.charAt(0).toUpperCase() + text.slice(1);

/**
 * The app a request means: the one named, or the only one there is. Why there is none is a sentence of its own, as the
 * chat shows it after /softn or Export .softn (and a tool's error says it).
 */
export function resolveApp(vfs: Vfs, requested?: unknown): { ok: true; root: string } | { ok: false; reason: string } {
  const apps = findApps(vfs);
  if (typeof requested === 'string' && requested.trim()) {
    const root = appKey(requested.replace(/\/manifest\.json$/i, ''));
    if (isSoftnApp(vfs, root)) return { ok: true, root };
    return { ok: false, reason: sentence(`${appLabel(root)} is not a SoftN app (it needs a manifest.json whose "main" is its main page, such as "index.html"). Apps here: ${apps.length ? apps.map(appLabel).join(', ') : 'none'}`) };
  }
  if (apps.length === 1) return { ok: true, root: apps[0] };
  if (!apps.length) return { ok: false, reason: 'There is no SoftN app in this project yet (a folder whose manifest.json names its main page, such as "index.html", as "main")' };
  return { ok: false, reason: `This project has several SoftN apps; say which with \`app\`: ${apps.map(appLabel).join(', ')}` };
}

/** Python logic or JavaScript logic: one app is one or the other. */
export function usesPython(vfs: Vfs, root = ''): boolean {
  return appFiles(vfs, root).some(([p]) => /\.py$/i.test(p) && !p.startsWith('server/'));
}

export function guideFor(vfs: Vfs, root = ''): string {
  return usesPython(vfs, root) ? SOFTN_GUIDE_PYTHON : SOFTN_GUIDE_JAVASCRIPT;
}

/**
 * The app's files, relative to its folder: everything below it but editor
 * state, dependency folders, dotfiles, and other apps nested inside it.
 */
export function appFiles(vfs: Vfs, root = ''): Array<[string, Uint8Array]> {
  const prefix = root ? `${root}/` : '';
  const nested = findApps(vfs).filter((r) => r !== root && (root === '' || r.startsWith(prefix)));
  return vfs
    .files()
    .filter(([path]) => path.startsWith(prefix) && !nested.some((n) => path.startsWith(`${n}/`)))
    .map(([path, data]) => [path.slice(prefix.length), data] as [string, Uint8Array])
    .filter(([path]) => {
      const segs = path.split('/');
      if (segs[0] === 'builder') return false;
      // Uploads to the project are not the app's (an app at the root would otherwise ship them).
      if (root === '' && segs[0] === 'uploads') return false;
      return !segs.some((s, i) => s.startsWith('.') || (i < segs.length - 1 && IGNORED_DIRS.has(s)));
    });
}

/** The manifest as a bundle carries it: `main`, `version` and `files` made true. */
export function normalizedManifest(vfs: Vfs, root = ''): SoftnManifest {
  const read = readManifest(vfs, root);
  const manifest: SoftnManifest = { ...(read ?? {}) };
  const appFileList = appFiles(vfs, root);
  const files = appFileList.map(([p]) => p);
  if (typeof manifest.main !== 'string' || !manifest.main) {
    manifest.main = typeof manifest.entry === 'string' && manifest.entry ? manifest.entry : [MAIN_PAGE, 'ui/main.ui'].find((p) => files.includes(p)) ?? files.find(isAppPagePath) ?? MAIN_PAGE;
  }
  delete manifest.entry;
  if (typeof manifest.version !== 'string' || !manifest.version) manifest.version = '1.0.0';
  if (typeof manifest.name !== 'string' || !manifest.name.trim()) manifest.name = root ? root.split('/').pop()! : 'SoftN app';
  const groups: Record<string, string[]> = { ui: [], logic: [], xdb: [], assets: [] };
  const listedLogic = listedIn(read, 'logic');
  for (const [path, group] of fileGroups(appFileList, read)) (groups[group] ??= []).push(path);
  // Logic keeps the order the author gave it (helpers before the entry).
  groups.logic.sort((a, b) => {
    const ia = listedLogic.indexOf(a);
    const ib = listedLogic.indexOf(b);
    return (ia < 0 ? Infinity : ia) - (ib < 0 ? Infinity : ib) || (a < b ? -1 : 1);
  });
  if (!groups.server?.length) delete groups.server;
  manifest.files = { ...(typeof manifest.files === 'object' && manifest.files ? manifest.files : {}), ...groups };
  return manifest;
}

/** The .softn bytes: a flat zip, manifest.json at the root. */
export function buildSoftn(vfs: Vfs, root = ''): Uint8Array {
  const tree: Record<string, Uint8Array> = {};
  for (const [path, data] of appFiles(vfs, root)) tree[path] = data;
  tree['manifest.json'] = new TextEncoder().encode(`${JSON.stringify(normalizedManifest(vfs, root), null, 2)}\n`);
  return zipSync(tree, { level: 6 });
}

export function softnFileName(vfs: Vfs, root: string, fallback: string): string {
  const manifest = normalizedManifest(vfs, root);
  return `${String(manifest.name || fallback).replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'app'}.softn`;
}

export function downloadSoftn(vfs: Vfs, root: string, fallbackName: string): string {
  const name = softnFileName(vfs, root, fallbackName);
  const url = URL.createObjectURL(new Blob([buildSoftn(vfs, root) as BlobPart], { type: 'application/zip' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return name;
}

/**
 * Unpack a .softn into a new folder of the project and return that folder.
 * A bundle is flat (manifest.json at its root); one wrapped in a single
 * folder is accepted too. Paths that would leave the folder are dropped.
 */
export function importSoftn(vfs: Vfs, bytes: Uint8Array, fileName: string, parent = ''): { root: string; files: number; name: string } {
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipChecked(bytes);
  } catch (error) {
    throw new Error(`${fileName} is not a .softn (zip) file: ${(error as Error).message}`);
  }
  const names = Object.keys(entries).filter((n) => !n.endsWith('/'));
  let prefix = '';
  if (!names.includes('manifest.json')) {
    const wrapped = names.find((n) => /^[^/]+\/manifest\.json$/.test(n));
    if (!wrapped) throw new Error(`${fileName} has no manifest.json: it is not a SoftN app`);
    prefix = wrapped.slice(0, -'manifest.json'.length);
  }
  let manifestName = '';
  try {
    manifestName = String((JSON.parse(new TextDecoder().decode(entries[`${prefix}manifest.json`])) as SoftnManifest).name ?? '');
  } catch {
    throw new Error(`${fileName}: manifest.json is not valid JSON`);
  }
  const base = (manifestName || fileName.replace(/\.softn$/i, '')).replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'app';
  const parentKey = appKey(parent);
  let root = parentKey ? `${parentKey}/${base}` : base;
  for (let n = 2; vfs.exists(`/${root}`); n++) root = parentKey ? `${parentKey}/${base}-${n}` : `${base}-${n}`;
  let count = 0;
  for (const name of names) {
    if (!name.startsWith(prefix)) continue;
    const rel = normalizePath(name.slice(prefix.length));
    if (!rel || name.slice(prefix.length).split('/').includes('..')) continue;
    vfs.writeFile(`/${root}/${rel}`, entries[name], { parents: true });
    count++;
  }
  return { root, files: count, name: manifestName || base };
}

/**
 * A short account of an app for the agent: its name, main page and files,
 * so it can decide what to read before changing or recreating it.
 */
export function describeApp(vfs: Vfs, root: string): string {
  const manifest = readManifest(vfs, root);
  const files = appFiles(vfs, root);
  const shown = files.slice(0, 60).map(([path, data]) => `  ${path} (${data.byteLength.toLocaleString()} bytes)`);
  if (files.length > shown.length) shown.push(`  … and ${files.length - shown.length} more`);
  const name = typeof manifest?.name === 'string' ? manifest.name : appLabel(root);
  const main = typeof manifest?.main === 'string' ? manifest.main : '(none)';
  return [`SoftN app "${name}" in ${appLabel(root)}: main ${main}, ${files.length} files`, ...shown].join('\n');
}

/** What Studio's validator would say before the app is even rendered. Paths are the app's own. */
export function checkProject(vfs: Vfs, root = ''): Finding[] {
  const out: Finding[] = [];
  const err = (file: string, message: string) => out.push({ level: 'error', file, message });
  const warn = (file: string, message: string) => out.push({ level: 'warning', file, message });
  let manifest: SoftnManifest | null = null;
  if (!vfs.exists(inApp(root, 'manifest.json'))) {
    err('manifest.json', `missing: a SoftN app needs manifest.json at the root of ${appLabel(root)}`);
  } else {
    try {
      manifest = JSON.parse(vfs.readText(inApp(root, 'manifest.json'))) as SoftnManifest;
    } catch (error) {
      err('manifest.json', `not valid JSON: ${(error as Error).message}`);
    }
  }
  const files = appFiles(vfs, root);
  const present = new Set(files.map(([p]) => p));
  const groups = fileGroups(files, manifest && typeof manifest === 'object' ? manifest : null);
  if (manifest) {
    if (typeof manifest.name !== 'string' || !manifest.name.trim()) err('manifest.json', '`name` must be a non-empty string');
    if (typeof manifest.version !== 'string') warn('manifest.json', '`version` is missing; export will use "1.0.0"');
    if ('entry' in manifest) warn('manifest.json', 'remove `entry`: the runtime reads `main`, never `entry`');
    if (typeof manifest.main !== 'string') err('manifest.json', `\`main\` must name the app's main page, e.g. "${MAIN_PAGE}"`);
    else if (!present.has(manifest.main)) err('manifest.json', `\`main\` names ${manifest.main}, which does not exist`);
    else if (!isAppPagePath(manifest.main)) err('manifest.json', `\`main\` names ${manifest.main}, which is not a page: it names the app's main page, e.g. "${MAIN_PAGE}" (or a .ui page)`);
    const listed = new Set<string>();
    if (manifest.files && typeof manifest.files === 'object') {
      for (const [group, list] of Object.entries(manifest.files)) {
        if (!Array.isArray(list)) {
          err('manifest.json', `\`files.${group}\` must be an array of paths`);
          continue;
        }
        for (const path of list) {
          if (typeof path !== 'string') continue;
          listed.add(path);
          if (!present.has(path)) err('manifest.json', `\`files.${group}\` lists ${path}, which does not exist`);
        }
      }
    } else {
      err('manifest.json', '`files` must list the app\'s files by group: { "ui": [...], "logic": [...] }');
    }
    for (const [path, group] of groups) {
      if (group !== 'assets' && !listed.has(path)) warn(path, `not listed in manifest.json \`files.${group}\` (export adds it)`);
    }
  }
  const logic = [...groups].filter(([, group]) => group === 'logic').map(([path]) => path);
  const js = logic.some((p) => /\.(logic|js)$/i.test(p));
  const py = logic.some((p) => /\.py$/i.test(p));
  if (js && py) err(logic.find((p) => /\.py$/i.test(p))!, 'an app\'s logic is JavaScript (.js or .logic) or Python (.py), never both');
  for (const path of present) {
    if (/\.(json|xdb)$/i.test(path) && path !== 'manifest.json') {
      try {
        JSON.parse(vfs.readText(inApp(root, path)));
      } catch (error) {
        err(path, `not valid JSON: ${(error as Error).message}`);
      }
    }
    if (TEXT_EXT.test(path) && vfs.stat(inApp(root, path))?.size === 0) warn(path, 'empty file');
  }
  if (present.has('permission.json')) {
    try {
      const perms = (JSON.parse(vfs.readText(inApp(root, 'permission.json'))) as { permissions?: Record<string, { enabled?: unknown }> }).permissions ?? {};
      for (const [cap, value] of Object.entries(perms)) {
        if (!CAPABILITIES.includes(cap)) warn('permission.json', `unknown capability \`${cap}\` (known: ${CAPABILITIES.join(', ')})`);
        else if (value && value.enabled !== undefined && value.enabled !== true && value.enabled !== false) err('permission.json', `\`${cap}.enabled\` must be the boolean true or false`);
      }
    } catch {
      /* reported above */
    }
  }
  return out;
}

/** An event handler written on a component in a page: `<Text @click={…}>` is Text, click, and its line. */
export interface Handler {
  tag: string;
  event: string;
  line: number;
}

/**
 * The event handlers a page writes on its components (tags that begin with a
 * capital): each `@name` among an opening tag's attributes, outside its strings and
 * its `{…}` values.
 */
export function handlersIn(source: string): Handler[] {
  const out: Handler[] = [];
  const open = /<([A-Z][A-Za-z0-9]*)(?=[\s/>])/g;
  for (let m = open.exec(source); m; m = open.exec(source)) {
    const line = source.slice(0, m.index).split('\n').length;
    let depth = 0;
    let quote = '';
    let i = open.lastIndex;
    for (; i < source.length; i++) {
      const c = source[i];
      if (quote) {
        if (c === '\\') i++;
        else if (c === quote) quote = '';
      } else if (c === '"' || c === "'" || c === '`') quote = c;
      else if (c === '{') depth++;
      else if (c === '}') depth = Math.max(0, depth - 1);
      else if (depth === 0 && c === '>') break;
      else if (depth === 0 && c === '@') {
        const name = /^[A-Za-z][A-Za-z0-9]*/.exec(source.slice(i + 1));
        if (name) out.push({ tag: m[1], event: name[0], line });
      }
    }
    open.lastIndex = i;
  }
  return out;
}

/**
 * Syntax errors in the app's JavaScript logic, found by compiling each logic
 * file (`.logic`, and each `.js` the app's pages link) on the Zipp sandbox
 * (the same engine the app runs on). Each is read as the runtime reads it: as a
 * script, its module syntax as what it means there, and its whole-file imports,
 * which the runtime replaces with the files they name, left out (each of those
 * is compiled as a file of its own); line for line, so a line is the author's.
 * The hosted runtime shows such an error inside the preview without reporting
 * it, so this is how the page and the agent learn of it.
 */
export async function checkLogicSyntax(vfs: Vfs, root: string, run: (source: string) => Promise<string | undefined>): Promise<Finding[]> {
  const sources: Record<string, string> = {};
  const files = appFiles(vfs, root);
  const linked = linkedScripts(scriptTexts(files));
  for (const [path, data] of files) {
    if (!/\.logic$/i.test(path) && !linked.has(path)) continue;
    try {
      sources[path] = moduleSyntaxAsScript(new TextDecoder('utf-8', { fatal: true }).decode(data)).replace(LOGIC_IMPORT_LINE, '');
    } catch {
      /* not text: reported elsewhere */
    }
  }
  if (!Object.keys(sources).length) return [];
  const program = `var sources = ${JSON.stringify(sources)}; var out = {};
for (var path in sources) { try { Function(sources[path]); } catch (e) { out[path] = String(e && e.message ? e.message : e); } }
JSON.stringify(out)`;
  const value = await run(program);
  let failures: Record<string, string> = {};
  try {
    failures = JSON.parse(value ?? '{}');
  } catch {
    return [];
  }
  return Object.entries(failures).map(([file, message]) => ({ level: 'error' as const, file, message: `syntax error: ${message}` }));
}

/** Compile-check the app's logic on the Zipp sandbox (no network, no writes). */
export function logicSyntax(vfs: Vfs, root = ''): Promise<Finding[]> {
  return checkLogicSyntax(vfs, root, async (source) => {
    const host = new SandboxHost(vfs, new NetGate({ mode: 'blocked' }), 'softn_check', 15_000);
    const outcome = await runInSandbox({ lang: 'js', source, limits: { maxSteps: 500_000_000 } }, host, { timeoutMs: 15_000 });
    return outcome.result?.value;
  });
}

export function formatFindings(findings: Finding[]): string {
  if (!findings.length) return 'no problems found in the files';
  return findings.map((f) => `${f.level === 'error' ? 'ERROR' : 'warning'} ${f.file}: ${f.message}`).join('\n');
}

/** The smallest SoftN app: a page, its logic, and a manifest that lists them. */
export const SOFTN_BLANK: Array<[string, string]> = [
  ['manifest.json', `${JSON.stringify({ name: 'App', version: '1.0.0', main: MAIN_PAGE, files: { ui: [MAIN_PAGE], logic: ['main.js'] } }, null, 2)}\n`],
  [
    MAIN_PAGE,
    `<script src="main.js"></script>

<App theme="dark">
  <Container size="md">
    <Stack direction="vertical" gap="lg" padding="xl">
      <Heading level={1}>{title}</Heading>
      <Text variant="muted">Ask the agent what to build here.</Text>
    </Stack>
  </Container>
</App>
`,
  ],
  ['main.js', 'let title = "Hello"\n'],
];

export const SOFTN_STARTER: Array<[string, string]> = [
  ['manifest.json', `${JSON.stringify({ name: 'Tasks', version: '1.0.0', main: MAIN_PAGE, files: { ui: [MAIN_PAGE], logic: ['main.js'] } }, null, 2)}\n`],
  [
    MAIN_PAGE,
    `<script src="main.js"></script>

<App theme="dark">
  <Container size="sm">
    <Stack direction="vertical" gap="lg" padding="xl">
      <Heading level={1}>Tasks</Heading>

      <Stack direction="horizontal" gap="sm">
        <Input :bind={newTask} placeholder="What needs to be done?" />
        <Button variant="primary" @click={() => addTask()}>Add</Button>
      </Stack>

      #each (task in tasks)
        <Card>
          <Stack direction="horizontal" gap="md" align="center">
            <Checkbox checked={task.done} @change={() => toggleTask(task.id)} />
            <Text style={{ flex: 1, textDecoration: task.done ? "line-through" : "none" }}>{task.title}</Text>
            <Button variant="ghost" size="sm" @click={() => deleteTask(task.id)}>Delete</Button>
          </Stack>
        </Card>
      #empty
        <EmptyState title="No tasks" description="Add a task to get started" />
      #end

      <Text size="sm" variant="muted">{remaining() + " remaining"}</Text>
    </Stack>
  </Container>
</App>
`,
  ],
  [
    'main.js',
    `let tasks = []
let newTask = ""
let nextId = 1

function addTask() {
  const title = newTask.trim()
  if (title === "") return
  tasks = tasks.concat([{ id: String(nextId), title: title, done: false }])
  nextId = nextId + 1
  newTask = ""
}

function toggleTask(id) {
  tasks = tasks.map((t) => (t.id === id ? { id: t.id, title: t.title, done: !t.done } : t))
}

function deleteTask(id) {
  tasks = tasks.filter((t) => t.id !== id)
}

function remaining() {
  return tasks.filter((t) => !t.done).length
}
`,
  ],
];
