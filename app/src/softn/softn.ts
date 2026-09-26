/**
 * SoftN apps in a bot.computer project: finding them, starting one, checking
 * one, importing a .softn, and bundling one as a .softn file.
 *
 * A SoftN app is a folder of text files — manifest.json, ui/*.ui pages,
 * logic/*.logic (JavaScript) or logic/*.py (Python), data/*.xdb seed records,
 * assets/ — and a .softn file is that folder zipped flat, manifest.json at the
 * root. A project may hold several apps, each in its own folder (the project
 * root is a folder too): any folder whose manifest.json has a .ui `main`.
 * Paths inside an app are relative to its folder. The rules here follow
 * softn.com's Studio (exportBundle.ts planBundle, validator.ts), Apache-2.0.
 */
import { unzipSync, zipSync } from 'fflate';
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

const TEXT_EXT = /\.(ui|logic|py|xdb|json|md|txt|css|csv|svg)$/i;
const CAPABILITIES = ['net', 'camera', 'mic', 'files', 'qr', 'ai', 'gpu', 'sync', 'storage', 'accel'];

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

/** Is this folder a SoftN app? (A manifest.json whose `main` is a .ui page.) */
export function isSoftnApp(vfs: Vfs, root = ''): boolean {
  const manifest = readManifest(vfs, root);
  return !!manifest && typeof manifest.main === 'string' && manifest.main.endsWith('.ui');
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

/** The app a request means: the one named, or the only one there is. */
export function resolveApp(vfs: Vfs, requested?: unknown): { ok: true; root: string } | { ok: false; reason: string } {
  const apps = findApps(vfs);
  if (typeof requested === 'string' && requested.trim()) {
    const root = appKey(requested.replace(/\/manifest\.json$/i, ''));
    if (isSoftnApp(vfs, root)) return { ok: true, root };
    return { ok: false, reason: `${appLabel(root)} is not a SoftN app (it needs a manifest.json whose "main" is a .ui page). Apps here: ${apps.length ? apps.map(appLabel).join(', ') : 'none'}` };
  }
  if (apps.length === 1) return { ok: true, root: apps[0] };
  if (!apps.length) return { ok: false, reason: 'there is no SoftN app in this project yet (a folder whose manifest.json has a "main" .ui page)' };
  return { ok: false, reason: `this project has several SoftN apps; say which with \`app\`: ${apps.map(appLabel).join(', ')}` };
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
      return !segs.some((s, i) => s.startsWith('.') || (i < segs.length - 1 && IGNORED_DIRS.has(s)));
    });
}

function groupOf(path: string): 'ui' | 'logic' | 'server' | 'xdb' | 'assets' | null {
  if (path.startsWith('server/') && /\.(logic|py)$/i.test(path)) return 'server';
  if (/\.ui$/i.test(path)) return 'ui';
  if (/\.(logic|py)$/i.test(path)) return 'logic';
  if (/\.xdb$/i.test(path)) return 'xdb';
  if (path.startsWith('assets/')) return 'assets';
  return null;
}

/** The manifest as a bundle carries it: `main`, `version` and `files` made true. */
export function normalizedManifest(vfs: Vfs, root = ''): SoftnManifest {
  const manifest: SoftnManifest = { ...(readManifest(vfs, root) ?? {}) };
  const files = appFiles(vfs, root).map(([p]) => p);
  if (typeof manifest.main !== 'string' || !manifest.main) {
    manifest.main = typeof manifest.entry === 'string' && manifest.entry ? manifest.entry : files.includes('ui/main.ui') ? 'ui/main.ui' : files.find((p) => p.endsWith('.ui')) ?? 'ui/main.ui';
  }
  delete manifest.entry;
  if (typeof manifest.version !== 'string' || !manifest.version) manifest.version = '1.0.0';
  if (typeof manifest.name !== 'string' || !manifest.name.trim()) manifest.name = root ? root.split('/').pop()! : 'SoftN app';
  const groups: Record<string, string[]> = { ui: [], logic: [], xdb: [], assets: [] };
  const listedLogic = Array.isArray(manifest.files?.logic) ? (manifest.files!.logic as unknown[]).filter((p): p is string => typeof p === 'string') : [];
  for (const path of files) {
    const group = groupOf(path);
    if (group) (groups[group] ??= []).push(path);
  }
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
    entries = unzipSync(bytes);
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
  const present = new Set(appFiles(vfs, root).map(([p]) => p));
  if (manifest) {
    if (typeof manifest.name !== 'string' || !manifest.name.trim()) err('manifest.json', '`name` must be a non-empty string');
    if (typeof manifest.version !== 'string') warn('manifest.json', '`version` is missing; export will use "1.0.0"');
    if ('entry' in manifest) warn('manifest.json', 'remove `entry`: the runtime reads `main`, never `entry`');
    if (typeof manifest.main !== 'string') err('manifest.json', '`main` must name the entry .ui page, e.g. "ui/main.ui"');
    else if (!present.has(manifest.main)) err('manifest.json', `\`main\` names ${manifest.main}, which does not exist`);
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
    for (const path of present) {
      const group = groupOf(path);
      if (group && group !== 'assets' && !listed.has(path)) warn(path, `not listed in manifest.json \`files.${group}\` (export adds it)`);
    }
  }
  const js = [...present].some((p) => /\.logic$/i.test(p) && !p.startsWith('server/'));
  const py = [...present].some((p) => /\.py$/i.test(p) && !p.startsWith('server/'));
  if (js && py) err('logic/', 'an app\'s logic is JavaScript (.logic) or Python (.py), never both');
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

/**
 * Syntax errors in the app's JavaScript logic, found by compiling each .logic
 * file on the Zipp sandbox (the same engine the app runs on). The hosted
 * runtime shows such an error inside the preview without reporting it, so
 * this is how the page and the agent learn of it.
 */
export async function checkLogicSyntax(vfs: Vfs, root: string, run: (source: string) => Promise<string | undefined>): Promise<Finding[]> {
  const sources: Record<string, string> = {};
  for (const [path, data] of appFiles(vfs, root)) {
    if (!/\.logic$/i.test(path)) continue;
    try {
      sources[path] = new TextDecoder('utf-8', { fatal: true }).decode(data);
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

export const SOFTN_STARTER: Array<[string, string]> = [
  ['manifest.json', `${JSON.stringify({ name: 'Tasks', version: '1.0.0', main: 'ui/main.ui', files: { ui: ['ui/main.ui'], logic: ['logic/main.logic'] } }, null, 2)}\n`],
  [
    'ui/main.ui',
    `<logic src="../logic/main.logic" />

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
    'logic/main.logic',
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
