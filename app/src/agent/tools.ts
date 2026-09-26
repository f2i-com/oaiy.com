/**
 * The agent's tools. Everything happens inside the browser tab: files are the
 * project's virtual filesystem, code runs on the Zipp VM in a Worker, and
 * network requests go through the network gate.
 *
 * Editing rules adapted from softn Studio (Apache-2.0): a file must be read
 * before it is edited or replaced, an edit must match exactly once (or say
 * replace_all), and a file that changed since it was read is refused.
 */
import type { NetGate } from '../gate/netgate';
import { SandboxHost, globRegex, summarize } from '../sandbox/host';
import { runInSandbox } from '../sandbox/runner';
import { VfsError, normalizePath, type Vfs } from '../vfs/vfs';
import type { ToolCall, ToolResult, ToolSpec } from './protocol';
import { checkProject, formatFindings, guideFor, isSoftnProject, logicSyntax } from '../softn/softn';
import type { PreviewResult } from '../softn/preview';

const READ_LINES = 400;
const READ_CHARS = 40_000;
const OUTPUT_CHARS = 30_000;
const DEFAULT_TIMEOUT_S = 30;
const MAX_TIMEOUT_S = 300;

export interface ToolContext {
  vfs: Vfs;
  gate: NetGate;
  /** Versions of the files read in this conversation: path -> version when read. */
  reads: Map<string, number>;
  /** The emulated shell's state, kept between calls. */
  shell: { cwd: string; env: Record<string, string> };
  /** The live SoftN preview, when the page has one. */
  softn?: { check(): Promise<PreviewResult> };
  signal?: AbortSignal;
}

const str = { type: 'string' };
const int = { type: 'integer' };

export const TOOLS: ToolSpec[] = [
  {
    name: 'list_files',
    description: 'List the project\'s files and folders under a path ("/" is the project root). Skips dependency and build folders unless include_ignored.',
    parameters: { type: 'object', properties: { path: str, depth: { ...int, description: 'How many levels (default 3)' }, include_ignored: { type: 'boolean' } } },
  },
  {
    name: 'read_file',
    description: `Read a text file with numbered lines (at most ${READ_LINES} lines per call; use offset to page). Read a file before editing or replacing it.`,
    parameters: { type: 'object', required: ['path'], properties: { path: str, offset: { ...int, description: 'First line (1-based)' }, limit: int } },
  },
  {
    name: 'write_file',
    description: 'Create a file, or replace one you have read in full. Parent folders are created.',
    parameters: { type: 'object', required: ['path', 'content'], properties: { path: str, content: str } },
  },
  {
    name: 'edit_file',
    description: 'Replace exact text in a file you have read. old_string must match exactly once unless replace_all is true; include enough surrounding lines to make it unique.',
    parameters: { type: 'object', required: ['path', 'old_string', 'new_string'], properties: { path: str, old_string: str, new_string: str, replace_all: { type: 'boolean' } } },
  },
  {
    name: 'delete_file',
    description: 'Delete a file or folder (folders recursively).',
    parameters: { type: 'object', required: ['path'], properties: { path: str } },
  },
  {
    name: 'grep',
    description: 'Search file contents with a JavaScript regular expression. Returns path:line: text matches.',
    parameters: { type: 'object', required: ['pattern'], properties: { pattern: str, path: str, glob: { ...str, description: 'Only files whose name or path matches, e.g. *.ts' }, ignore_case: { type: 'boolean' }, max_results: int } },
  },
  {
    name: 'glob',
    description: 'Find files by glob, e.g. src/**/*.ts.',
    parameters: { type: 'object', required: ['pattern'], properties: { pattern: str } },
  },
  {
    name: 'code_run',
    description:
      'Run JavaScript or Python on the Zipp VM, sandboxed in a Worker, to compute, transform data or answer with code. Paths are project-relative ("/" is the root). ' +
      'JavaScript: Node-style require("fs"|"path"), fs.readFileSync/writeFileSync/readdirSync/statSync, fs.walkSync, fs.grepSync, global fetch(), process.argv; top-level await works; the last expression\'s value is returned as `result`. ' +
      'Python 3 subset (math, json, re, collections, itertools, dataclasses, statistics, fractions, hashlib; no pip, no csv module): open() sees the project files matched by `files` (default: the project\'s files up to 2 MiB each, 16 MiB in all) and files it writes are saved back. ' +
      'Network requests go through the /internet gate and only reach sites that allow cross-origin requests from a browser.',
    parameters: {
      type: 'object',
      properties: {
        language: { type: 'string', enum: ['javascript', 'python'] },
        code: str,
        file: { ...str, description: 'Run this project file instead of code' },
        args: { type: 'array', items: str },
        files: { type: 'array', items: str, description: 'Python only: globs of files open() should see' },
        stdin: str,
        timeout_secs: { ...int, minimum: 1, maximum: MAX_TIMEOUT_S },
      },
    },
  },
  {
    name: 'sandbox_shell',
    description:
      'Run shell commands in bot.computer\'s emulated POSIX-style shell (on the Zipp VM, confined to the project; "/" is the project root). ' +
      'Pipes, && || ;, redirects, heredocs, $VAR/$(...)/$((...)), globs, if/for/while/case, and built-in ls cat head tail grep find sed awk sort uniq cut tr wc diff mkdir cp mv rm touch tree xargs tee printf echo, curl/wget (through the /internet gate; CORS applies), node/js and python on Zipp. ' +
      'There are no real processes: git, npm, compilers are not available. The working directory and exported variables persist between calls. Run `help` for details.',
    parameters: { type: 'object', required: ['command'], properties: { command: str, timeout_secs: { ...int, minimum: 1, maximum: MAX_TIMEOUT_S } } },
  },
  {
    name: 'softn_docs',
    description: 'The SoftN writing guide (from SoftN Studio): the app files, the .ui page language, .logic/.py logic, XDB data, capabilities, common mistakes and a complete example. Read it before writing or changing a SoftN app. Optional `section` returns only the sections whose heading contains it (e.g. "ui page", "logic", "mistakes", "example").',
    parameters: { type: 'object', properties: { section: str } },
  },
  {
    name: 'softn_check',
    description: 'Check the SoftN app in this project: the files (manifest.json, listed files, JSON, permissions) and a real render in the live preview, returning load and render errors. Run it after every change to a SoftN app, and fix what it reports.',
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'web_fetch',
    description: 'Fetch a URL as text through the /internet gate. From a browser only sites that allow cross-origin requests (CORS) answer.',
    parameters: { type: 'object', required: ['url'], properties: { url: str, max_chars: int } },
  },
];

function cut(text: string, max = OUTPUT_CHARS): string {
  if (text.length <= max) return text;
  const head = text.slice(0, Math.floor(max * 0.7));
  const tail = text.slice(-Math.floor(max * 0.25));
  return `${head}\n\n[... ${text.length - head.length - tail.length} characters omitted ...]\n\n${tail}`;
}

function need(input: Record<string, unknown>, key: string): string {
  const value = input[key];
  if (typeof value !== 'string') throw new Error(`expected \`${key}\` as a string`);
  return value;
}

function timeoutOf(input: Record<string, unknown>): number {
  const secs = typeof input.timeout_secs === 'number' ? input.timeout_secs : DEFAULT_TIMEOUT_S;
  return Math.min(Math.max(Math.round(secs), 1), MAX_TIMEOUT_S) * 1000;
}

/** A few lines of the file that most resemble `needle`, for a failed edit. */
function nearestLines(text: string, needle: string): string {
  const first = needle.split('\n').find((l) => l.trim()) ?? needle;
  const grams = (s: string) => {
    const t = s.trim().toLowerCase();
    const set = new Set<string>();
    for (let i = 0; i < t.length - 1; i++) set.add(t.slice(i, i + 2));
    return set;
  };
  const target = grams(first);
  const lines = text.split('\n');
  let best = -1;
  let score = 0;
  lines.forEach((line, i) => {
    const g = grams(line);
    let shared = 0;
    for (const x of g) if (target.has(x)) shared++;
    const s = (2 * shared) / (g.size + target.size || 1);
    if (s > score) {
      score = s;
      best = i;
    }
  });
  if (best < 0 || score < 0.3) return '';
  const from = Math.max(0, best - 2);
  return lines
    .slice(from, best + 3)
    .map((l, i) => `${from + i + 1}\t${l}`)
    .join('\n');
}

function requireFreshRead(ctx: ToolContext, key: string, what: string): void {
  const seen = ctx.reads.get(key);
  if (seen === undefined) throw new Error(`read /${key} before you ${what} it`);
  if (seen !== ctx.vfs.version(`/${key}`)) throw new Error(`/${key} changed since you read it; read it again before you ${what} it`);
}

async function execute(call: ToolCall, ctx: ToolContext): Promise<string> {
  const input = call.input;
  const vfs = ctx.vfs;
  switch (call.name) {
    case 'list_files': {
      const start = normalizePath(typeof input.path === 'string' ? input.path : '/');
      const depth = typeof input.depth === 'number' ? Math.max(1, input.depth) : 3;
      const walked = vfs.walk(`/${start}`, { includeIgnored: input.include_ignored === true, limit: 2000 });
      const lines: string[] = [];
      for (const e of walked.entries) {
        const rel = start ? e.path.slice(start.length + 1) : e.path;
        if (rel.split('/').length > depth) continue;
        lines.push(e.type === 'dir' ? `${rel}/` : `${rel}  (${e.size} B)`);
      }
      if (!lines.length) return `/${start} is empty`;
      return lines.join('\n') + (walked.truncated ? '\n[listing truncated]' : '');
    }
    case 'read_file': {
      const key = normalizePath(need(input, 'path'));
      const text = vfs.readText(`/${key}`);
      const lines = text.split('\n');
      const offset = typeof input.offset === 'number' ? Math.max(1, Math.floor(input.offset)) : 1;
      const limit = typeof input.limit === 'number' ? Math.min(Math.max(1, input.limit), READ_LINES) : READ_LINES;
      const slice = lines.slice(offset - 1, offset - 1 + limit);
      let body = slice.map((l, i) => `${offset + i}\t${l}`).join('\n');
      if (body.length > READ_CHARS) body = `${body.slice(0, READ_CHARS)}\n[cut at ${READ_CHARS} characters]`;
      const whole = offset === 1 && slice.length === lines.length && body.length <= READ_CHARS;
      // Editing needs a read; replacing needs the whole file read.
      ctx.reads.set(key, vfs.version(`/${key}`));
      if (!whole) ctx.reads.set(`${key}#partial`, 1);
      else ctx.reads.delete(`${key}#partial`);
      const more = offset - 1 + slice.length < lines.length ? `\n[lines ${offset}-${offset - 1 + slice.length} of ${lines.length}; continue with offset ${offset + slice.length}]` : '';
      return body + more;
    }
    case 'write_file': {
      const key = normalizePath(need(input, 'path'));
      const content = need(input, 'content');
      if (vfs.exists(`/${key}`)) {
        requireFreshRead(ctx, key, 'replace');
        if (ctx.reads.has(`${key}#partial`)) throw new Error(`read all of /${key} before replacing it, or use edit_file`);
      }
      vfs.writeFile(`/${key}`, content, { parents: true });
      ctx.reads.set(key, vfs.version(`/${key}`));
      return `wrote /${key} (${content.split('\n').length} lines)`;
    }
    case 'edit_file': {
      const key = normalizePath(need(input, 'path'));
      const oldString = need(input, 'old_string');
      const newString = need(input, 'new_string');
      requireFreshRead(ctx, key, 'edit');
      const text = vfs.readText(`/${key}`);
      if (!oldString) throw new Error('old_string is empty');
      const count = text.split(oldString).length - 1;
      if (count === 0) {
        const near = nearestLines(text, oldString);
        throw new Error(`old_string was not found in /${key}${near ? `. The closest lines are:\n${near}` : ''}`);
      }
      if (count > 1 && input.replace_all !== true) throw new Error(`old_string matches ${count} places in /${key}; add surrounding lines to make it unique, or set replace_all`);
      const next = input.replace_all === true ? text.split(oldString).join(newString) : text.replace(oldString, () => newString);
      vfs.writeFile(`/${key}`, next);
      ctx.reads.set(key, vfs.version(`/${key}`));
      return `edited /${key} (${count} replacement${count === 1 ? '' : 's'})`;
    }
    case 'delete_file': {
      const key = normalizePath(need(input, 'path'));
      vfs.remove(`/${key}`, true);
      ctx.reads.delete(key);
      return `deleted /${key}`;
    }
    case 'grep': {
      const host = new SandboxHost(vfs, ctx.gate, 'grep', 10_000);
      const found = JSON.parse(
        await host.call('fs.grep', [
          JSON.stringify({ pattern: need(input, 'pattern'), path: typeof input.path === 'string' ? input.path : '/', glob: input.glob ?? null, ignore_case: input.ignore_case === true, max_results: typeof input.max_results === 'number' ? input.max_results : 200 }),
        ]),
      ) as { matches: Array<{ path: string; line: number; text: string }>; truncated: boolean };
      if (!found.matches.length) return 'no matches';
      return found.matches.map((m) => `${m.path}:${m.line}: ${m.text}`).join('\n') + (found.truncated ? '\n[more matches not shown]' : '');
    }
    case 'glob': {
      const pattern = need(input, 'pattern').replace(/^\.?\//, '');
      const re = globRegex(pattern, true);
      const hits = vfs.walk('/', { includeIgnored: /node_modules|\.git|dist|build|target/.test(pattern) }).entries.filter((e) => e.type === 'file' && re.test(e.path));
      return hits.length ? hits.slice(0, 1000).map((e) => e.path).join('\n') : 'no files match';
    }
    case 'code_run': {
      const timeoutMs = timeoutOf(input);
      const host = new SandboxHost(vfs, ctx.gate, 'code_run', timeoutMs);
      host.signal = ctx.signal;
      const file = typeof input.file === 'string' ? normalizePath(input.file) : undefined;
      const language = typeof input.language === 'string' ? input.language : file?.endsWith('.py') ? 'python' : 'javascript';
      const lang = /^py/i.test(language) ? 'python' : 'js';
      const source = file !== undefined ? vfs.readText(`/${file}`) : need(input, 'code');
      const outcome = await host.runProgram({
        lang,
        source,
        fileName: file ?? (lang === 'python' ? 'main.py' : 'main.js'),
        argv: Array.isArray(input.args) ? input.args.map(String) : [],
        stdin: typeof input.stdin === 'string' ? input.stdin : '',
        preload: Array.isArray(input.files) ? input.files.map(String) : undefined,
      });
      const { stdout, stderr, exitCode } = summarize(outcome);
      const report: Record<string, unknown> = { ok: exitCode === 0, exit_code: exitCode, stdout: cut(stdout), stderr: cut(stderr), elapsed_ms: outcome.elapsedMs };
      if (outcome.result?.value !== undefined) report.result = cut(outcome.result.value);
      if (outcome.timedOut) report.timed_out = true;
      if (host.changes.written.size) report.files_written = [...host.changes.written];
      if (host.changes.deleted.size) report.files_deleted = [...host.changes.deleted];
      return JSON.stringify(report, null, 1);
    }
    case 'sandbox_shell': {
      const timeoutMs = timeoutOf(input);
      const host = new SandboxHost(vfs, ctx.gate, 'sandbox_shell', timeoutMs);
      host.signal = ctx.signal;
      const cwd = vfs.stat(ctx.shell.cwd)?.type === 'dir' ? ctx.shell.cwd : '/';
      const outcome = await runInSandbox({ lang: 'shell', source: need(input, 'command'), cwd, env: ctx.shell.env, limits: { maxSteps: 2_000_000_000 } }, host, { timeoutMs, signal: ctx.signal });
      const shell = outcome.result?.shell;
      const report: Record<string, unknown> = {};
      if (shell && !outcome.timedOut && !outcome.result?.error) {
        ctx.shell.cwd = shell.cwd;
        ctx.shell.env = shell.env;
        Object.assign(report, { exit_code: shell.exit_code, stdout: cut(shell.stdout), stderr: cut(shell.stderr), cwd: shell.cwd });
      } else {
        const { stdout, stderr, exitCode } = summarize(outcome);
        Object.assign(report, { exit_code: exitCode || 1, stdout: cut(stdout), stderr: cut(stderr), cwd });
        if (outcome.timedOut) report.timed_out = true;
      }
      if (host.changes.written.size) report.files_written = [...host.changes.written];
      if (host.changes.deleted.size) report.files_deleted = [...host.changes.deleted];
      return JSON.stringify(report, null, 1);
    }
    case 'softn_docs': {
      const guide = guideFor(vfs);
      const want = typeof input.section === 'string' ? input.section.trim().toLowerCase() : '';
      if (!want) return guide;
      const sections = guide.split(/\n(?=## )/);
      const hits = sections.filter((s) => s.split('\n')[0].toLowerCase().includes(want));
      if (hits.length) return hits.join('\n\n');
      return `No section heading contains "${want}". Sections: ${sections.map((s) => s.split('\n')[0].replace(/^#+\s*/, '')).join('; ')}`;
    }
    case 'softn_check': {
      const findings = [...checkProject(vfs), ...(await logicSyntax(vfs).catch(() => []))];
      const lines = [`Files: ${formatFindings(findings)}`];
      if (!isSoftnProject(vfs)) return `${lines[0]}\nNot rendered: this is not a SoftN app until manifest.json names a "main" .ui page.`;
      if (findings.some((f) => f.level === 'error')) {
        lines.push('Render: skipped until the file errors above are fixed.');
      } else if (!ctx.softn) {
        lines.push('Render: no live preview in this session.');
      } else {
        const result = await ctx.softn.check();
        lines.push(result.ok ? 'Render: the app loaded and rendered without reported errors (the user sees it in the preview).' : `Render errors:\n${result.errors.map((e) => `- ${e}`).join('\n')}`);
      }
      return lines.join('\n');
    }
    case 'web_fetch': {
      const host = new SandboxHost(vfs, ctx.gate, 'web_fetch', 60_000);
      const response = JSON.parse(await host.call('net.fetch', [JSON.stringify({ url: need(input, 'url') })])) as { status: number; url: string; headers: Record<string, string>; body: string; truncated: boolean };
      const max = typeof input.max_chars === 'number' ? Math.min(Math.max(1000, input.max_chars), 100_000) : 20_000;
      let body = response.body;
      if (/html/i.test(response.headers['content-type'] ?? '')) body = htmlToText(body);
      return `HTTP ${response.status} ${response.url}\n\n${cut(body, max)}`;
    }
    default:
      throw new Error(`unknown tool \`${call.name}\``);
  }
}

function htmlToText(html: string): string {
  try {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('script,style,noscript,svg').forEach((n) => n.remove());
    return (doc.body?.innerText || doc.body?.textContent || '').replace(/\n{3,}/g, '\n\n').trim();
  } catch {
    return html.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, '').replace(/<[^>]+>/g, ' ').replace(/\s+\n/g, '\n');
  }
}

export async function runTool(call: ToolCall, ctx: ToolContext): Promise<ToolResult> {
  if (call.parseError) return { id: call.id, name: call.name, content: `Error: the tool call's arguments could not be read: ${call.parseError}`, isError: true };
  try {
    const content = await execute(call, ctx);
    let isError = false;
    if ((call.name === 'code_run' || call.name === 'sandbox_shell') && /"exit_code": (?!0\b)-?\d+/.test(content)) isError = true;
    return { id: call.id, name: call.name, content, isError };
  } catch (error) {
    const message = error instanceof VfsError || error instanceof Error ? error.message : String(error);
    return { id: call.id, name: call.name, content: `Error: ${message}`, isError: true };
  }
}
