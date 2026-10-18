/**
 * Shared Node-host state: where files live on a real disk, how the runtime's
 * virtual paths map onto it, and a tiny secrets store.
 *
 * The browser host keeps everything in an in-memory VFS / OPFS; the Node host
 * uses the actual filesystem. Runtime code addresses files with a handful of
 * virtual roots (`/oaiy`, `/tmp`, `/downloads`) plus genuine absolute paths, so
 * `resolvePath` is the single place that maps those onto real directories.
 */
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';

/** Root for app data (databases, app-state). `OAIY_DATA_DIR` overrides. */
export const DATA_DIR = path.resolve(
  process.env.OAIY_DATA_DIR || path.join(os.homedir(), '.oaiy'),
);

/** Scratch space for intermediate flow artifacts (frames, transcodes, …). */
export const TMP_ROOT = path.resolve(
  process.env.OAIY_TMP_DIR || path.join(DATA_DIR, 'tmp'),
);

/** Where user-facing outputs that the web build would "download" land. */
export const DOWNLOADS_DIR = path.resolve(
  process.env.OAIY_DOWNLOADS_DIR || path.join(DATA_DIR, 'downloads'),
);

export function ensureDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
}

const stripLead = (s: string) => s.replace(/^[/\\]+/, '');

/**
 * Join `rel` under `root` and assert the result stays inside `root` — blocks
 * `..` traversal out of a virtual sandbox (the desktop Rust host validates
 * paths the same way; without this the Node adapter would be strictly weaker).
 * `path.join` collapses `..`, so we canonicalise and check the prefix with a
 * trailing separator (so a sibling dir sharing the name prefix can't match).
 */
export function within(root: string, rel: string): string {
  const base = path.resolve(root);
  const real = path.resolve(path.join(base, rel));
  if (real !== base && !real.startsWith(base + path.sep)) {
    throw new Error(`path escapes the sandbox root (${root}): ${rel}`);
  }
  // Under worker confinement, never let a flow fs op reach the secret store — it lives inside
  // DATA_DIR (an allowed root), so `/oaiy/secrets.json` would otherwise resolve here and leak
  // every stored API key. get_secret/store_secret read it DIRECTLY (not via this), so app-level
  // secret access is unaffected; desktop / `oaiy run` (not confined) is unchanged.
  if (confineEnabled() && real === SECRETS_FILE) {
    throw new Error('access to the secret store is not allowed');
  }
  return real;
}

const ALLOWED_ROOTS = [DATA_DIR, TMP_ROOT, DOWNLOADS_DIR].map((r) => path.resolve(r));

/**
 * Server-mode confinement for ABSOLUTE paths. When `oaiy worker` drives an
 * untrusted run queue, a flow input carrying an absolute path would otherwise read
 * any file the process can (e.g. /etc/shadow, SSH keys), overwrite a cron/startup
 * file (→ RCE), or delete arbitrary data — `resolvePath` confines only the virtual
 * roots, not absolute paths. When confinement is on, reject absolute paths outside
 * the allowed roots. OFF by default so single-user `oaiy run` keeps full-disk parity
 * with the desktop host; `oaiy worker` sets OAIY_FS_CONFINE=1, and OAIY_FS_ALLOW_ABSOLUTE=1
 * overrides. Read from env at call time so the CLI can set it before any fs op runs.
 */
function confineEnabled(): boolean {
  return process.env.OAIY_FS_CONFINE === '1' && process.env.OAIY_FS_ALLOW_ABSOLUTE !== '1';
}

export function confineAbsolute(p: string): string {
  if (!confineEnabled()) return p; // desktop / `oaiy run` parity unchanged
  const real = path.resolve(p);
  // An absolute path to the secret store (e.g. ffmpeg `-i /home/u/.oaiy/secrets.json`) is inside
  // an allowed root, so block it here too — same reason as `within`.
  if (real === SECRETS_FILE) {
    throw new Error('access to the secret store is not allowed');
  }
  if (ALLOWED_ROOTS.some((root) => real === root || real.startsWith(root + path.sep))) {
    return real;
  }
  throw new Error(`absolute path is outside the server fs sandbox: ${p}`);
}

/**
 * Whether a run may reach `hostPort` on this computer or its network. A run with
 * outside input (`oaiy worker`'s queue, the Desktop's runs for webhooks, plugins
 * and apps: OAIY_FS_CONFINE=1) reaches only the Desktop that started it, its
 * engines, voice and flows, at OAIY_SERVER_URL: a URL an outside caller gave
 * could otherwise reach the cloud's metadata address, a router, or another
 * program's port of this computer. `OAIY_HTTP_ALLOW_LOCAL=1` is the operator's
 * way out. An unconfined run (`oaiy run` of the person's own flow) reaches all.
 */
export function localNetworkAllowed(hostPort: string, env: NodeJS.ProcessEnv = process.env): boolean {
  const confined = env.OAIY_FS_CONFINE === '1' && env.OAIY_FS_ALLOW_ABSOLUTE !== '1';
  if (!confined || env.OAIY_HTTP_ALLOW_LOCAL === '1') return true;
  let desktop: URL;
  try {
    desktop = new URL(env.OAIY_SERVER_URL || '');
  } catch {
    return false;
  }
  const loopback = (h: string) => ['127.0.0.1', 'localhost', '[::1]', '::1'].includes(h.toLowerCase());
  const at = hostPort.lastIndexOf(':');
  const host = at > 0 ? hostPort.slice(0, at) : hostPort;
  const port = at > 0 ? hostPort.slice(at + 1) : '';
  const desktopPort = desktop.port || (desktop.protocol === 'https:' ? '443' : '80');
  if (port !== desktopPort) return false;
  return host.toLowerCase() === desktop.hostname.toLowerCase() || (loopback(host) && loopback(desktop.hostname));
}

/** True when remote-protocol media args (ffmpeg `-i http://…`) should be blocked. */
export function fsConfinementOn(): boolean {
  return confineEnabled();
}

/**
 * Map a runtime/virtual path to a real filesystem path, confined to its root.
 *   /oaiy, /oaiy/…          → DATA_DIR/…       (traversal-checked)
 *   /tmp, /tmp/…, tmp/…   → TMP_ROOT/…       (traversal-checked)
 *   /downloads, …         → DOWNLOADS_DIR/…  (traversal-checked)
 *   absolute real path    → unchanged (or root-confined under `oaiy worker`)
 *   anything else (rel)   → under TMP_ROOT    (traversal-checked)
 */
export function resolvePath(p: string): string {
  if (!p) return TMP_ROOT;
  const norm = p.replace(/\\/g, '/');
  if (norm === '/oaiy' || norm.startsWith('/oaiy/')) {
    return within(DATA_DIR, stripLead(norm.slice('/oaiy'.length)));
  }
  if (norm === '/tmp' || norm.startsWith('/tmp/')) {
    return within(TMP_ROOT, stripLead(norm.slice('/tmp'.length)));
  }
  if (norm.startsWith('tmp/')) {
    return within(TMP_ROOT, stripLead(norm.slice('tmp'.length)));
  }
  if (norm === '/downloads' || norm.startsWith('/downloads/')) {
    return within(DOWNLOADS_DIR, stripLead(norm.slice('/downloads'.length)));
  }
  if (path.isAbsolute(p)) return confineAbsolute(p);
  return within(TMP_ROOT, stripLead(norm));
}

// --- Secrets -------------------------------------------------------------
// Engine API-key resolution normally flows through `projectConstants`
// (passed to createEngine). These helpers back the `get_secrets`/`store_secret`
// app-level commands for parity; sources are env first, then a JSON file.

const SECRETS_FILE = path.join(DATA_DIR, 'secrets.json');

function readSecretsFile(): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(SECRETS_FILE, 'utf-8')) as Record<string, string>;
  } catch {
    return {};
  }
}

export function getSecret(key: string): string | undefined {
  if (process.env[key]) return process.env[key];
  return readSecretsFile()[key];
}

export function storeSecret(key: string, value: string): void {
  const all = readSecretsFile();
  if (value) all[key] = value;
  else delete all[key];
  ensureDir(DATA_DIR);
  fs.writeFileSync(SECRETS_FILE, JSON.stringify(all, null, 2), 'utf-8');
}
