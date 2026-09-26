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
import { appLabel, checkProject, describeApp, findApps, formatFindings, guideFor, importSoftn, logicSyntax, resolveApp } from '../softn/softn';
import type { PageReport, PreviewAction, PreviewResult } from '../softn/preview';
import { describeExample, docsMap, installExample, listExamples, lookupComponents, readTopic, searchKnowledge } from '../softn/knowledge';
import { DEFAULT_VIEW_SIZE, MAX_VIEW_SIZE, imageMimeFor, imageSize, viewImage, type ImagePart } from './images';

const READ_LINES = 400;
const READ_CHARS = 40_000;
/** A line longer than this is cut in a line read; char_start reads the rest. */
const LINE_CHARS = 2_000;
const SEARCH_RESULTS = 100;
const OUTPUT_CHARS = 30_000;
const DEFAULT_TIMEOUT_S = 30;
const MAX_TIMEOUT_S = 300;

/** The live SoftN preview, as the tools use it. */
export interface SoftnHost {
  check(root?: string): Promise<PreviewResult>;
  inspect(root?: string): Promise<PageReport>;
  act(root: string | undefined, actions: PreviewAction[]): Promise<PageReport>;
}

export interface ToolContext {
  vfs: Vfs;
  gate: NetGate;
  /** Versions of the files read in this conversation: path -> version when read. */
  reads: Map<string, number>;
  /** The emulated shell's state, kept between calls. */
  shell: { cwd: string; env: Record<string, string> };
  /** The live SoftN preview, when the page has one. */
  softn?: SoftnHost;
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
    description: `Read a text file with numbered lines (at most ${READ_LINES} lines per call; page with offset). Lines over ${LINE_CHARS} characters are cut, with where they continue; char_start/char_count read raw characters instead (for minified or single-line files). For a large file, use file_info first and search_file to find the part you need. Read a file before editing or replacing it. Images: use view_image.`,
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        offset: { ...int, description: 'First line (1-based)' },
        limit: int,
        char_start: { ...int, description: 'Read raw characters from this offset (0-based) instead of lines' },
        char_count: { ...int, description: `How many characters with char_start (default and max ${READ_CHARS})` },
      },
    },
  },
  {
    name: 'file_info',
    description: 'Size and shape of a file before reading it: bytes, text or binary, line count, longest line, the first lines; for an image, its dimensions. Use it to plan reading a large file.',
    parameters: { type: 'object', required: ['path'], properties: { path: str } },
  },
  {
    name: 'search_file',
    description: 'Search inside one file (any size) with a JavaScript regular expression; returns each match with its line number, column, character offset and surrounding lines. Use it to navigate a large file, then read_file around the match.',
    parameters: {
      type: 'object',
      required: ['path', 'pattern'],
      properties: {
        path: str,
        pattern: str,
        context: { ...int, description: 'Lines of context around each match (default 2)' },
        ignore_case: { type: 'boolean' },
        literal: { type: 'boolean', description: 'Treat pattern as plain text' },
        max_results: int,
      },
    },
  },
  {
    name: 'present_file',
    description: 'Show the user a file from the project in the chat: an image is displayed, audio and video get a player, anything else a link that opens it. Use it to hand over something you made or found (a generated image or sound, a report). It does not show you the file: use view_image or read_file for that.',
    parameters: { type: 'object', required: ['path'], properties: { path: str, caption: str } },
  },
  {
    name: 'view_image',
    description:
      `Look at an image in the project (png, jpg, gif, webp, svg, bmp, avif). The whole image is shown scaled to fit ${DEFAULT_VIEW_SIZE} px (max_size up to ${MAX_VIEW_SIZE}). ` +
      'To see detail, zoom: pass a region x, y, width, height in the ORIGINAL image\'s pixels and that region is shown at up to max_size, so a smaller region shows more real detail. ' +
      'grid: true overlays labelled coordinates (original pixels) to aim the next zoom. The reply states the original size, the region shown and the scale.',
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        max_size: { ...int, minimum: 64, maximum: MAX_VIEW_SIZE },
        grid: { type: 'boolean' },
      },
    },
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
    description:
      'The SoftN reference: how to write SoftN apps. With no arguments: the map of everything there is (the writing guide\'s sections, the published guides, every component, the example apps). ' +
      '`topic` reads one: "guide" for the whole writing guide (read it before your first app), "guide#<words from a heading>" for one section (e.g. "guide#mistakes"), or a published guide\'s slug, optionally "#section" (e.g. "xdb-data", "state-events#events"). ' +
      '`search` finds words across the guide, the guides, the component reference and the example apps\' source, best first, each with a snippet and how to open it: use it when you need to know how something is done (e.g. "timer interval", "@change Select", "save to storage").',
    parameters: { type: 'object', properties: { topic: str, search: str, app: { ...str, description: 'The app folder, so the guide matches its language (JavaScript or Python logic)' } } },
  },
  {
    name: 'softn_components',
    description: 'The exact reference for SoftN components you are about to use: every prop with its type and allowed values, events (@name), children, and a usage example. Generated from the component sources, so it is authoritative: check here instead of guessing a prop.',
    parameters: { type: 'object', required: ['names'], properties: { names: { type: 'array', items: { type: 'string' }, description: 'Component names, e.g. ["Table", "Tabs"]; up to 12' } } },
  },
  {
    name: 'softn_examples',
    description:
      'Complete, working SoftN apps to learn from (a notes app with storage, a game, a component and chart showcase, 3D, WebGPU, device permissions). ' +
      'No arguments: the list. `name`: an app\'s files and manifest. `name` + `file`: read one of its files. `name` + `install_to`: copy the whole app into that folder of the project (it then shows in the preview; start from it or take parts of it).',
    parameters: { type: 'object', properties: { name: str, file: str, install_to: str } },
  },
  {
    name: 'softn_check',
    description: 'Check a SoftN app: its files (manifest.json, listed files, JSON, permissions, logic syntax) and a real render in the live preview, which switches to show it, returning load and render errors. Run it after every change to a SoftN app, and fix what it reports. With several apps in the project, name the one with `app` (its folder; "/" for the project root).',
    parameters: { type: 'object', properties: { app: { ...str, description: 'The app\'s folder, e.g. "apps/tasks"; optional when the project has one app' } } },
  },
  {
    name: 'softn_inspect',
    description: 'Describe what a SoftN app shows in the live preview right now, as text: headings, text, buttons, inputs with their values, checkboxes, selects, links, images and canvases, plus any errors the app reported. Use it to confirm the page looks as intended.',
    parameters: { type: 'object', properties: { app: { ...str, description: 'The app\'s folder; optional when the project has one app' } } },
  },
  {
    name: 'softn_interact',
    description:
      'Use a SoftN app in the live preview the way a person would, to test that it works: a list of actions, run in order, then the page as text and any errors the app raised. ' +
      'Actions: {"click": "<button or link text or label>"}, {"fill": "<input label or placeholder>", "value": "..."}, {"select": "<select label>", "value": "<option>"}, {"key": "Enter" | "ArrowUp" | "a" …}, {"wait": 500}. Add "nth": 2 to pick the second match. ' +
      'State carries over between calls until the app is changed or re-rendered.',
    parameters: {
      type: 'object',
      required: ['actions'],
      properties: { app: { ...str, description: 'The app\'s folder; optional when the project has one app' }, actions: { type: 'array', items: { type: 'object' } } },
    },
  },
  {
    name: 'softn_import',
    description: 'Unpack a .softn file (a zipped SoftN app) that is in the project into a new folder of its own, named after the app (never over existing files), and list what it holds. Uploaded .softn files are unpacked already; use this for one that is not, or to get a fresh copy of the original to compare with or start again from. Then change the unpacked app in place, or read it and write a new app in another folder, as the user asks.',
    parameters: { type: 'object', required: ['path'], properties: { path: { ...str, description: 'The .softn (or .zip) file, e.g. "uploads/Tasks.softn"' }, parent: { ...str, description: 'Folder to unpack under (default: the project root)' } } },
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

/** Images a tool produced for the model, collected alongside its text. */
let imagesOut: ImagePart[] = [];
/** Files present_file shows the person, collected the same way. */
let filesOut: string[] = [];

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
      if (imageMimeFor(key) && !/\.svg$/i.test(key)) throw new Error(`/${key} is an image: look at it with view_image`);
      const text = vfs.readText(`/${key}`);
      if (typeof input.char_start === 'number') {
        const start = Math.max(0, Math.floor(input.char_start));
        const count = Math.min(Math.max(1, typeof input.char_count === 'number' ? Math.floor(input.char_count) : READ_CHARS), READ_CHARS);
        const slice = text.slice(start, start + count);
        const lineNo = text.slice(0, start).split('\n').length;
        return `[characters ${start}-${start + slice.length} of ${text.length}, starting on line ${lineNo}]\n${slice}${start + slice.length < text.length ? `\n[continue with char_start ${start + slice.length}]` : ''}`;
      }
      const lines = text.split('\n');
      const offset = typeof input.offset === 'number' ? Math.max(1, Math.floor(input.offset)) : 1;
      const limit = typeof input.limit === 'number' ? Math.min(Math.max(1, input.limit), READ_LINES) : READ_LINES;
      const slice = lines.slice(offset - 1, offset - 1 + limit);
      let charAt = 0;
      for (let i = 0; i < offset - 1; i++) charAt += lines[i].length + 1;
      let body = slice
        .map((l, i) => {
          const at = charAt;
          charAt += l.length + 1;
          return l.length > LINE_CHARS ? `${offset + i}\t${l.slice(0, LINE_CHARS)} [line continues: ${l.length - LINE_CHARS} more characters; char_start ${at + LINE_CHARS}]` : `${offset + i}\t${l}`;
        })
        .join('\n');
      if (body.length > READ_CHARS) body = `${body.slice(0, READ_CHARS)}\n[cut at ${READ_CHARS} characters]`;
      const whole = offset === 1 && slice.length === lines.length && body.length <= READ_CHARS;
      // Editing needs a read; replacing needs the whole file read.
      ctx.reads.set(key, vfs.version(`/${key}`));
      if (!whole) ctx.reads.set(`${key}#partial`, 1);
      else ctx.reads.delete(`${key}#partial`);
      const more = offset - 1 + slice.length < lines.length ? `\n[lines ${offset}-${offset - 1 + slice.length} of ${lines.length}; continue with offset ${offset + slice.length}]` : '';
      return body + more;
    }
    case 'file_info': {
      const key = normalizePath(need(input, 'path'));
      const st = vfs.stat(`/${key}`);
      if (!st) throw new Error(`ENOENT: no such file or directory, '/${key}'`);
      if (st.type === 'dir') {
        const kids = vfs.list(`/${key}`);
        return `/${key}: folder, ${kids.length} entries (${kids.filter((k) => k.type === 'dir').length} folders)`;
      }
      const head = `/${key}: ${st.size.toLocaleString()} bytes`;
      const mime = imageMimeFor(key);
      if (mime) {
        try {
          const size = await imageSize(vfs.readBytes(`/${key}`), mime);
          return `${head}, image ${mime}, ${size.width}×${size.height} px. Look at it with view_image (zoom with a region).`;
        } catch (error) {
          return `${head}, ${mime}, but it could not be decoded: ${(error as Error).message}`;
        }
      }
      if (!vfs.isText(`/${key}`)) return `${head}, binary (not UTF-8 text).`;
      const text = vfs.readText(`/${key}`);
      const lines = text.split('\n');
      let longest = 0;
      let longestAt = 0;
      lines.forEach((l, i) => {
        if (l.length > longest) {
          longest = l.length;
          longestAt = i + 1;
        }
      });
      const preview = lines.slice(0, 5).map((l, i) => `${i + 1}\t${l.length > 200 ? `${l.slice(0, 200)}…` : l}`).join('\n');
      return `${head}, text, ${text.length.toLocaleString()} characters, ${lines.length.toLocaleString()} lines; longest line ${longest.toLocaleString()} characters (line ${longestAt}).${longest > LINE_CHARS ? ' Long lines: read them with char_start.' : ''}\nFirst lines:\n${preview}`;
    }
    case 'search_file': {
      const key = normalizePath(need(input, 'path'));
      const text = vfs.readText(`/${key}`);
      const raw = need(input, 'pattern');
      let re: RegExp;
      try {
        re = new RegExp(input.literal === true ? raw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : raw, input.ignore_case === true ? 'gi' : 'g');
      } catch (error) {
        throw new Error(`invalid regular expression: ${(error as Error).message}`);
      }
      const context = typeof input.context === 'number' ? Math.min(Math.max(0, Math.floor(input.context)), 20) : 2;
      const max = typeof input.max_results === 'number' ? Math.min(Math.max(1, input.max_results), 1000) : SEARCH_RESULTS;
      const lines = text.split('\n');
      const starts: number[] = [];
      let at = 0;
      for (const l of lines) {
        starts.push(at);
        at += l.length + 1;
      }
      const lineOf = (offset: number) => {
        let lo = 0;
        let hi = starts.length - 1;
        while (lo < hi) {
          const mid = (lo + hi + 1) >> 1;
          if (starts[mid] <= offset) lo = mid;
          else hi = mid - 1;
        }
        return lo;
      };
      const out: string[] = [];
      let count = 0;
      let m: RegExpExecArray | null;
      while ((m = re.exec(text)) !== null) {
        if (m[0] === '') re.lastIndex++;
        count++;
        if (count > max) continue;
        const li = lineOf(m.index);
        const col = m.index - starts[li];
        const clip = (l: string) => (l.length > 300 ? `${l.slice(Math.max(0, col - 120), col + 180)}…` : l);
        const from = Math.max(0, li - context);
        const to = Math.min(lines.length - 1, li + context);
        const block: string[] = [`match ${count}: line ${li + 1}, column ${col + 1}, char ${m.index}`];
        for (let j = from; j <= to; j++) block.push(`${j === li ? '>' : ' '}${j + 1}\t${clip(lines[j])}`);
        out.push(block.join('\n'));
        if (count > 100_000) break;
      }
      if (!count) return `no matches in /${key} (${lines.length.toLocaleString()} lines)`;
      return `${count.toLocaleString()} match${count === 1 ? '' : 'es'} in /${key}${count > max ? ` (showing the first ${max})` : ''}\n\n${out.join('\n\n')}`;
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
    case 'view_image': {
      const key = normalizePath(need(input, 'path'));
      const mime = imageMimeFor(key);
      if (!mime) throw new Error(`/${key} is not an image this tool can show (png, jpg, gif, webp, svg, bmp, avif)`);
      const num = (k: string) => (typeof input[k] === 'number' ? (input[k] as number) : undefined);
      const view = await viewImage(vfs.readBytes(`/${key}`), mime, {
        x: num('x'),
        y: num('y'),
        width: num('width'),
        height: num('height'),
        maxSize: num('max_size'),
        grid: input.grid === true,
        label: key,
      });
      imagesOut.push(view.image);
      const r = view.region;
      const whole = r.x === 0 && r.y === 0 && r.width === view.width && r.height === view.height;
      const scale = view.shownWidth / r.width;
      return `/${key}: ${view.width}×${view.height} px. Showing ${whole ? 'the whole image' : `region x=${r.x}, y=${r.y}, ${r.width}×${r.height}`} at ${view.shownWidth}×${view.shownHeight} (scale ${scale >= 1 ? scale.toFixed(2) : `1/${(1 / scale).toFixed(1)}`}).${scale < 0.9 ? ' Zoom into a region to see more detail.' : ''}`;
    }
    case 'softn_docs': {
      const target = typeof input.app === 'string' ? resolveApp(vfs, input.app) : null;
      const guide = guideFor(vfs, target?.ok ? target.root : (findApps(vfs)[0] ?? ''));
      if (typeof input.search === 'string' && input.search.trim()) return searchKnowledge(input.search, guide);
      // `section` is the older way to ask for part of the writing guide.
      const topic = typeof input.topic === 'string' && input.topic.trim() ? input.topic : typeof input.section === 'string' && input.section.trim() ? `guide#${input.section}` : '';
      if (!topic) return docsMap(guide);
      const read = await readTopic(topic, guide);
      if (!read.ok) throw new Error(read.text);
      return read.text;
    }
    case 'softn_components': {
      const names = Array.isArray(input.names) ? input.names.map(String) : typeof input.names === 'string' ? input.names.split(/[\s,]+/).filter(Boolean) : [];
      if (!names.length) throw new Error('give the component names, e.g. names: ["Button", "Table"]');
      const found = await lookupComponents(names);
      if (!found.ok) throw new Error(found.text);
      return found.text;
    }
    case 'softn_examples': {
      const name = typeof input.name === 'string' ? input.name.trim() : '';
      if (!name) return listExamples();
      if (typeof input.install_to === 'string') {
        const installed = await installExample(vfs, name, normalizePath(input.install_to));
        return `Copied the example into ${appLabel(installed.root)} (${installed.files} files).\n${describeApp(vfs, installed.root)}\nRun softn_check with app "${installed.root}" to see it in the preview.`;
      }
      const shown = await describeExample(name, typeof input.file === 'string' && input.file.trim() ? input.file : undefined);
      if (!shown.ok) throw new Error(shown.text);
      return shown.text;
    }
    case 'softn_check': {
      const target = resolveApp(vfs, input.app);
      if (!target.ok) {
        const root = typeof input.app === 'string' ? input.app : '/';
        const findings = checkProject(vfs, normalizePath(root));
        return `Not a SoftN app yet: ${target.reason}\nFiles in ${root}: ${formatFindings(findings)}`;
      }
      return (await checkApp(ctx, target.root)).text;
    }
    case 'softn_inspect': case 'softn_interact': {
      const target = resolveApp(vfs, input.app);
      if (!target.ok) throw new Error(target.reason);
      if (!ctx.softn) throw new Error('there is no live preview in this session');
      const report = call.name === 'softn_inspect' ? await ctx.softn.inspect(target.root) : await ctx.softn.act(target.root, Array.isArray(input.actions) ? (input.actions as PreviewAction[]) : []);
      const lines = [`App: ${appLabel(target.root)}`];
      if (report.done?.length) lines.push(`Done: ${report.done.join('; ')}`);
      if (report.error) lines.push(`Could not: ${report.error}`);
      lines.push(formatProblems(report.problems) || 'Errors: none reported.');
      lines.push('The page now shows:', report.page || '(nothing)');
      const text = cut(lines.join('\n'));
      if (report.error || report.problems.some((p) => p.level === 'error')) throw new Error(text);
      return text;
    }
    case 'present_file': {
      const path = normalizePath(need(input, 'path'));
      const stat = vfs.stat(`/${path}`);
      if (!stat) throw new Error(`/${path} does not exist`);
      if (stat.type !== 'file') throw new Error(`/${path} is a folder; present a file`);
      filesOut.push(path);
      const caption = typeof input.caption === 'string' && input.caption.trim() ? ` (${input.caption.trim()})` : '';
      return `Shown to the user in the chat: /${path}, ${stat.size.toLocaleString()} bytes${caption}.`;
    }
    case 'softn_import': {
      const path = normalizePath(need(input, 'path'));
      const imported = importSoftn(vfs, vfs.readBytes(path), path.split('/').pop()!, typeof input.parent === 'string' ? normalizePath(input.parent) : '');
      return `Unpacked /${path} into ${appLabel(imported.root)} (${imported.files} files).\n${describeApp(vfs, imported.root)}\nRun softn_check with app "${imported.root}" to see it in the preview.`;
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

function formatProblems(problems: PageReport['problems']): string {
  if (!problems.length) return '';
  const errors = problems.filter((p) => p.level === 'error');
  const warnings = problems.filter((p) => p.level === 'warning');
  return [
    errors.length ? `Errors the app raised (from the running app):\n${errors.slice(0, 8).map((p) => `- ${p.message}`).join('\n')}` : '',
    warnings.length ? `Warnings:\n${warnings.slice(0, 8).map((p) => `- ${p.message}`).join('\n')}` : '',
  ].filter(Boolean).join('\n');
}

/**
 * Check a SoftN app: the files (manifest, listed files, JSON, permissions,
 * logic syntax) and a real render in the live preview. Used by softn_check
 * and by the automatic check after the agent changes an app.
 */
export async function checkApp(ctx: ToolContext, root: string): Promise<{ ok: boolean; text: string; signature: string }> {
  const findings = [...checkProject(ctx.vfs, root), ...(await logicSyntax(ctx.vfs, root).catch(() => []))];
  const lines = [`App: ${appLabel(root)}`, `Files: ${formatFindings(findings)}`];
  const problems: string[] = findings.filter((f) => f.level === 'error').map((f) => `${f.file}: ${f.message}`);
  if (problems.length) {
    lines.push('Render: skipped until the file errors above are fixed.');
  } else if (!ctx.softn) {
    lines.push('Render: no live preview in this session.');
  } else {
    const result = await ctx.softn.check(root);
    problems.push(...result.errors);
    lines.push(result.ok ? 'Render: the app loaded and rendered without reported errors (the user sees it in the preview).' : `Render errors:\n${result.errors.map((e) => `- ${e}`).join('\n')}`);
    if (result.warnings?.length) lines.push(`Warnings while it loaded (often a handler or name the logic does not define):\n${[...new Set(result.warnings)].slice(0, 8).map((w) => `- ${w}`).join('\n')}`);
  }
  // The same errors again mean the fixes are not working.
  const signature = problems.map((p) => p.replace(/\d+/g, '#')).sort().join('|');
  return { ok: !problems.length, text: lines.join('\n'), signature };
}

export async function runTool(call: ToolCall, ctx: ToolContext): Promise<ToolResult> {
  if (call.parseError) return { id: call.id, name: call.name, content: `Error: the tool call's arguments could not be read: ${call.parseError}`, isError: true };
  imagesOut = [];
  filesOut = [];
  try {
    const content = await execute(call, ctx);
    let isError = false;
    if ((call.name === 'code_run' || call.name === 'sandbox_shell') && /"exit_code": (?!0\b)-?\d+/.test(content)) isError = true;
    const result: ToolResult = { id: call.id, name: call.name, content, isError };
    if (imagesOut.length) result.images = imagesOut;
    if (filesOut.length) result.files = filesOut;
    imagesOut = [];
    filesOut = [];
    return result;
  } catch (error) {
    const message = error instanceof VfsError || error instanceof Error ? error.message : String(error);
    return { id: call.id, name: call.name, content: `Error: ${message}`, isError: true };
  }
}
