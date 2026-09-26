/**
 * The sandbox's host: every file and network operation a guest program asks
 * for is decided here, on the page. Files live in the project's virtual
 * filesystem, which has no outside to escape to; network requests pass the
 * network gate first. The same host calls coder-cli's runner answers, so the
 * guest scripts (prelude.js, shell.js) are shared between the two.
 */
import type { NetGate } from '../gate/netgate';
import { IGNORED_DIRS, baseName, normalizePath, type Vfs } from '../vfs/vfs';
import type { Lang, RunRequest } from './protocol';
import { runInSandbox, type HostHandler, type RunOutcome } from './runner';

const MAX_FETCH_BYTES = 8 << 20;
const MAX_DOWNLOAD_BYTES = 128 << 20;
const MAX_GREP_FILE_BYTES = 4 << 20;
const MAX_NESTING = 3;
const MAX_SLEEP_MS = 10_000;
const PRELOAD_FILES = 1_500;
const PRELOAD_FILE_BYTES = 2 << 20;
const PRELOAD_TOTAL_BYTES = 16 << 20;
export const DEFAULT_MAX_STEPS = 2_000_000_000;

const FORBIDDEN_HEADERS = new Set(['host', 'connection', 'content-length', 'transfer-encoding', 'cookie', 'origin', 'referer', 'proxy-authorization', 'upgrade']);

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function base64ToBytes(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** A host as the gate compares it: lower case, IPv6 without brackets. */
function hostKey(host: string): string {
  return host.toLowerCase().replace(/^\[|\]$/g, '');
}

/**
 * An IPv6 address that carries an IPv4 one (::ffff:7f00:1, ::ffff:127.0.0.1,
 * ::127.0.0.1) as the IPv4 address, so it is judged as what it reaches.
 */
function embeddedIPv4(h: string): string | null {
  const dotted = /^::(?:ffff:(?:0:)?)?(\d+\.\d+\.\d+\.\d+)$/.exec(h);
  if (dotted) return dotted[1];
  const hex = /^::(?:ffff:(?:0:)?)?([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(h);
  if (!hex) return null;
  const hi = parseInt(hex[1], 16);
  const lo = parseInt(hex[2], 16);
  return `${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}`;
}

/** Hosts that name this machine or its network: reachable only when allowed by name. */
export function isLocalHost(host: string): boolean {
  let h = hostKey(host);
  const v4 = h.includes(':') ? embeddedIPv4(h) : null;
  if (v4) h = v4;
  if (h === 'localhost' || h.endsWith('.localhost') || h.endsWith('.local') || h.endsWith('.internal') || h === 'metadata.google.internal') return true;
  if (h.includes(':')) return /^[0-9a-f:.]+$/.test(h) && (h === '::1' || h === '::' || /^fe[89ab]/.test(h) || /^f[cd]/.test(h));
  const m = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(h);
  if (!m) return false;
  const [a, b] = [Number(m[1]), Number(m[2])];
  return a === 127 || a === 10 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
}

export interface ChangeLog {
  written: Set<string>;
  deleted: Set<string>;
}

export class SandboxHost implements HostHandler {
  readonly changes: ChangeLog = { written: new Set(), deleted: new Set() };
  /** Stop: ends every program this host (and its nested ones) runs. */
  signal?: AbortSignal;
  private readonly deadline: number;

  constructor(
    private readonly vfs: Vfs,
    private readonly gate: NetGate,
    private readonly via: string,
    timeoutMs: number,
    private readonly depth = 0,
    changes?: ChangeLog,
    deadline?: number,
  ) {
    this.deadline = deadline ?? Date.now() + timeoutMs;
    if (changes) this.changes = changes;
  }

  remainingMs(): number {
    return Math.max(0, this.deadline - Date.now());
  }

  private nested(): SandboxHost {
    const nested = new SandboxHost(this.vfs, this.gate, this.via, 0, this.depth + 1, this.changes, this.deadline);
    nested.signal = this.signal;
    return nested;
  }

  private wrote(path: string): void {
    const key = normalizePath(path);
    this.changes.deleted.delete(key);
    this.changes.written.add(key);
  }

  private removed(path: string): void {
    const key = normalizePath(path);
    this.changes.written.delete(key);
    this.changes.deleted.add(key);
  }

  async call(kind: string, args: string[]): Promise<string> {
    const arg = (i: number, what: string): string => {
      if (args[i] === undefined) throw new TypeError(`missing ${what}`);
      return args[i];
    };
    const flag = (i: number) => args[i] === 'true' || args[i] === '1';
    const vfs = this.vfs;
    switch (kind) {
      case 'fs.read':
        return vfs.readText(arg(0, 'path'));
      case 'fs.readb64':
        return bytesToBase64(vfs.readBytes(arg(0, 'path')));
      case 'fs.write':
        vfs.writeFile(arg(0, 'path'), arg(1, 'data'));
        this.wrote(args[0]);
        return '';
      case 'fs.append':
        vfs.writeFile(arg(0, 'path'), arg(1, 'data'), { append: true });
        this.wrote(args[0]);
        return '';
      case 'fs.writeb64':
        vfs.writeFile(arg(0, 'path'), base64ToBytes(arg(1, 'data')));
        this.wrote(args[0]);
        return '';
      case 'fs.stat': {
        const st = vfs.stat(arg(0, 'path'));
        return st ? JSON.stringify({ type: st.type, size: st.size, mtime_ms: st.mtimeMs }) : 'null';
      }
      case 'fs.list':
        return JSON.stringify(vfs.list(arg(0, 'path')).map((e) => ({ name: e.name, type: e.type, size: e.size, mtime_ms: e.mtimeMs })));
      case 'fs.walk': {
        let includeIgnored = false;
        try {
          includeIgnored = !!JSON.parse(args[1] ?? '{}')?.include_ignored;
        } catch {
          /* default */
        }
        const walked = vfs.walk(arg(0, 'path'), { includeIgnored });
        return JSON.stringify({ entries: walked.entries.map((e) => ({ path: e.path, type: e.type, size: e.size, mtime_ms: e.mtimeMs })), truncated: walked.truncated });
      }
      case 'fs.grep':
        return JSON.stringify(this.grep(JSON.parse(arg(0, 'request'))));
      case 'fs.mkdir':
        vfs.mkdir(arg(0, 'path'), flag(1));
        return '';
      case 'fs.remove':
        vfs.remove(arg(0, 'path'), flag(1));
        this.removed(args[0]);
        return '';
      case 'fs.rename':
        vfs.rename(arg(0, 'source'), arg(1, 'target'));
        this.removed(args[0]);
        this.wrote(args[1]);
        return '';
      case 'fs.copy':
        vfs.copy(arg(0, 'source'), arg(1, 'target'));
        this.wrote(args[1]);
        return '';
      case 'net.fetch':
        return JSON.stringify(await this.fetch(JSON.parse(arg(0, 'request'))));
      case 'proc.run':
        return JSON.stringify(await this.procRun(JSON.parse(arg(0, 'request'))));
      case 'sys.sleep': {
        const ms = Math.min(Number(args[0]) || 0, MAX_SLEEP_MS, this.remainingMs());
        await new Promise((r) => setTimeout(r, ms));
        return '';
      }
      default:
        throw new TypeError(`\`${kind}\` is not available in the bot.computer sandbox`);
    }
  }

  private grep(req: { pattern: string; path?: string; ignore_case?: boolean; glob?: string | null; exclude_dir?: string | null; max_results?: number; files_only?: boolean }) {
    let regex: RegExp;
    try {
      regex = new RegExp(req.pattern, req.ignore_case ? 'i' : '');
    } catch (error) {
      throw new SyntaxError(`invalid regular expression: ${(error as Error).message}`);
    }
    const glob = req.glob ? globRegex(req.glob) : null;
    const max = Math.min(Math.max(req.max_results ?? 500, 1), 20_000);
    const walked = this.vfs.walk(req.path ?? '/');
    const matches: Array<{ path: string; line: number; text: string }> = [];
    const files: string[] = [];
    let scanned = 0;
    let truncated = walked.truncated;
    outer: for (const entry of walked.entries) {
      if (entry.type !== 'file' || entry.size > MAX_GREP_FILE_BYTES) continue;
      if (glob && !glob.test(baseName(entry.path)) && !glob.test(entry.path)) continue;
      if (req.exclude_dir && entry.path.split('/').includes(req.exclude_dir)) continue;
      let text: string;
      try {
        text = this.vfs.readText(`/${entry.path}`);
      } catch {
        continue;
      }
      scanned++;
      const lines = text.split('\n');
      for (let n = 0; n < lines.length; n++) {
        if (!regex.test(lines[n])) continue;
        if (req.files_only) {
          files.push(entry.path);
          continue outer;
        }
        matches.push({ path: entry.path, line: n + 1, text: lines[n].slice(0, 1000) });
        if (matches.length >= max) {
          truncated = true;
          break outer;
        }
      }
    }
    return { matches, files, truncated, files_scanned: scanned };
  }

  private async fetch(req: { url: string; method?: string; headers?: Record<string, string>; body?: string | null; save_to?: string }) {
    let url: URL;
    try {
      url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(req.url) ? req.url : `https://${req.url}`);
    } catch {
      throw new TypeError(`fetch failed: invalid URL ${req.url}`);
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new TypeError('fetch failed: only http and https URLs are allowed');
    const host = url.hostname;
    const settings = this.gate.getSettings();
    if (isLocalHost(host) && !settings.allow.some((h) => hostKey(h) === hostKey(host))) {
      this.gate.check(host, this.via);
      throw new TypeError(`fetch failed: \`${host}\` is on this machine or its local network; the user can allow it with \`/internet allow ${host}\``);
    }
    const decision = this.gate.check(host, this.via);
    if (!decision.ok) throw new TypeError(`fetch failed: ${decision.reason}`);
    const headers = new Headers();
    for (const [name, value] of Object.entries(req.headers ?? {})) {
      if (!FORBIDDEN_HEADERS.has(name.toLowerCase())) headers.set(name, String(value));
    }
    // Only an open gate may follow redirects: the browser cannot let the gate
    // look at a redirect's new host before it goes there.
    const open = settings.mode === 'open' && settings.deny.length === 0;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.max(1000, Math.min(this.remainingMs(), 120_000)));
    let response: Response;
    try {
      response = await fetch(url, {
        method: (req.method ?? 'GET').toUpperCase(),
        headers,
        body: req.body ?? undefined,
        redirect: open ? 'follow' : 'error',
        credentials: 'omit',
        signal: controller.signal,
      });
      // A redirect the gate did not see: judge where it ended up before anything is read.
      if (response.redirected && response.url) {
        const final = new URL(response.url).hostname;
        if (isLocalHost(final) && !settings.allow.some((h) => hostKey(h) === hostKey(final))) {
          clearTimeout(timer);
          void response.body?.cancel().catch(() => {});
          throw new TypeError(`fetch failed: ${host} redirected to \`${final}\`, which is on this machine or its local network; not followed`);
        }
        const after = this.gate.check(final, this.via);
        if (!after.ok) {
          clearTimeout(timer);
          void response.body?.cancel().catch(() => {});
          throw new TypeError(`fetch failed: ${host} redirected to ${final}: ${after.reason}`);
        }
      }
    } catch (error) {
      clearTimeout(timer);
      if (error instanceof TypeError && error.message.startsWith('fetch failed:')) throw error;
      const why = controller.signal.aborted
        ? 'timed out'
        : `${(error as Error).message}. From a browser this usually means the site does not allow cross-origin requests (CORS)${open ? '' : ', or it redirected while the gate is restricted'}`;
      throw new TypeError(`fetch failed: ${why}`);
    }
    const limit = req.save_to ? MAX_DOWNLOAD_BYTES : MAX_FETCH_BYTES;
    const reader = response.body?.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    let truncated = false;
    if (reader) {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (total + value.byteLength > limit) {
          chunks.push(value.subarray(0, limit - total));
          total = limit;
          truncated = true;
          await reader.cancel();
          break;
        }
        chunks.push(value);
        total += value.byteLength;
      }
    }
    clearTimeout(timer);
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) {
      bytes.set(c, offset);
      offset += c.byteLength;
    }
    const outHeaders: Record<string, string> = {};
    response.headers.forEach((value, name) => {
      outHeaders[name] = value;
    });
    const out: Record<string, unknown> = {
      status: response.status,
      status_text: response.statusText,
      url: response.url || url.href,
      redirected: response.redirected,
      headers: outHeaders,
      truncated,
    };
    if (req.save_to) {
      if (truncated) throw new RangeError(`EFBIG: download exceeds ${MAX_DOWNLOAD_BYTES} bytes`);
      this.vfs.writeFile(req.save_to, bytes);
      this.wrote(req.save_to);
      out.saved_bytes = bytes.byteLength;
      out.body = '';
    } else {
      out.body = new TextDecoder().decode(bytes);
    }
    return out;
  }

  private async procRun(req: { lang: string; file?: string; source?: string; argv?: string[]; stdin?: string; cwd?: string }) {
    if (this.depth >= MAX_NESTING) throw new Error('programs may only be nested three deep in the sandbox');
    const lang: Lang = /^py/.test(req.lang) ? 'python' : 'js';
    const cwd = `/${normalizePath(req.cwd ?? '/')}`;
    let source = req.source ?? '';
    let fileName: string | undefined;
    if (req.file !== undefined) {
      source = this.vfs.readText(req.file);
      fileName = relativeTo(cwd, req.file);
    }
    const outcome = await this.nested().runProgram({ lang, source, fileName, argv: req.argv ?? [], stdin: req.stdin ?? '', cwd });
    const { stdout, stderr, exitCode } = summarize(outcome);
    return { stdout, stderr, exit_code: exitCode };
  }

  /** Run a JS or Python program; Python's writes come back into the project. */
  async runProgram(request: Omit<RunRequest, 'limits' | 'files'> & { preload?: string[] }): Promise<RunOutcome> {
    const remaining = this.remainingMs();
    if (remaining < 200) throw new Error('the sandbox time budget is spent');
    const cwd = `/${normalizePath(request.cwd ?? '/')}`;
    const full: RunRequest = { ...request, cwd, limits: { maxSteps: DEFAULT_MAX_STEPS } };
    if (request.lang === 'python') full.files = this.preload(cwd, request.preload);
    const outcome = await runInSandbox(full, this, { timeoutMs: remaining, signal: this.signal });
    if (request.lang === 'python' && outcome.result?.vfsChanges) {
      for (const change of outcome.result.vfsChanges) {
        const target = `${cwd === '/' ? '' : cwd}/${change.path}`;
        if (change.deleted) {
          if (this.vfs.exists(target)) {
            this.vfs.remove(target, false);
            this.removed(target);
          }
        } else if (change.base64 !== undefined) {
          this.vfs.writeFile(target, base64ToBytes(change.base64), { parents: true });
          this.wrote(target);
        }
      }
    }
    return outcome;
  }

  /** The files a Python program's `open()` sees, relative to its directory. */
  private preload(cwd: string, patterns?: string[]): Record<string, string | { base64: string }> {
    const base = normalizePath(cwd);
    const matchers = patterns?.map((p) => {
      const clean = p.trim().replace(/^\.?\//, '');
      return /[*?[]/.test(clean) ? globRegex(clean, true) : new RegExp(`^${escapeRegex(clean.replace(/\/$/, ''))}(/.*)?$`);
    });
    const files: Record<string, string | { base64: string }> = {};
    let count = 0;
    let total = 0;
    const walked = this.vfs.walk(`/${base}`, { includeIgnored: !!patterns });
    for (const entry of walked.entries) {
      if (entry.type !== 'file') continue;
      const inside = base ? entry.path.slice(base.length + 1) : entry.path;
      if (matchers && !matchers.some((m) => m.test(inside))) continue;
      if (!matchers && (entry.size > PRELOAD_FILE_BYTES || count >= PRELOAD_FILES || total + entry.size > PRELOAD_TOTAL_BYTES)) continue;
      const bytes = this.vfs.readBytes(`/${entry.path}`);
      total += bytes.byteLength;
      count++;
      let text: string | null = null;
      try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      } catch {
        text = null;
      }
      files[inside] = text !== null ? text : { base64: bytesToBase64(bytes) };
    }
    return files;
  }
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
}

/** A shell-style glob as a RegExp; `**` crosses directories when `paths`. */
export function globRegex(glob: string, paths = false): RegExp {
  let re = '';
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === '*') {
      if (glob[i + 1] === '*') {
        re += '.*';
        i++;
        if (glob[i + 1] === '/') {
          re = `${re.slice(0, -2)}(?:.*/)?`;
          i++;
        }
      } else re += paths ? '[^/]*' : '.*';
    } else if (c === '?') re += paths ? '[^/]' : '.';
    else re += escapeRegex(c);
  }
  return new RegExp(`^${re}$`);
}

export function relativeTo(cwd: string, file: string): string {
  const abs = normalizePath(file, cwd);
  const base = normalizePath(cwd);
  return base && abs.startsWith(`${base}/`) ? abs.slice(base.length + 1) : abs;
}

/** stdout, stderr and an exit code from a finished (or failed) run. */
export function summarize(outcome: RunOutcome): { stdout: string; stderr: string; exitCode: number } {
  const lines = outcome.result?.console ?? [];
  let stdout = lines.filter((l) => !l.err).map((l) => `${l.text}\n`).join('');
  let stderr = lines.filter((l) => l.err).map((l) => `${l.text}\n`).join('');
  let exitCode = 0;
  const result = outcome.result;
  if (outcome.timedOut) {
    stderr += '[stopped: the sandbox time limit was reached]\n';
    exitCode = 124;
  }
  if (outcome.failure) {
    stderr += `[sandbox failed: ${outcome.failure}]\n`;
    exitCode = 125;
  }
  if (result) {
    if (result.error) {
      stderr += `${result.error}\n`;
      exitCode = 1;
    }
    if (result.limit) {
      stderr += `[stopped: ${result.limit}]\n`;
      exitCode = 137;
    }
    if (typeof result.exit === 'number') exitCode = result.exit;
    else if (typeof result.exit === 'string' && result.exit) {
      stderr += `${result.exit}\n`;
      exitCode = 1;
    }
  } else if (exitCode === 0) {
    exitCode = 1;
  }
  stdout = stdout.replace(/\n$/, '\n');
  return { stdout, stderr, exitCode };
}

export { IGNORED_DIRS };
