// OAIY Runtime - Executes compiled workflow scripts (JavaScript) with agentic capabilities
import { OAIYCompiler } from './compiler.js';
import { extractDeepValue } from './value-utils.js';
import { AbortError } from './errors.js';
import type { WorkflowGraph, StreamCallback, LogCallback, ImageCallback, NodeStatusCallback, Flow, DatabaseCallback, WorkflowInputs, LocalNetworkPermissionCallback, ProjectSettings } from './types.js';
import type { RuntimeModule, RuntimeContext, RuntimeMethod, ModuleRegistry, LoadedModule } from './module-types.js';
import { MODULE_CLEANUP } from './module-types.js';
import { BoundedMap } from './runtime/BoundedMap.js';
import {
  isLocalNetworkUrl as isLocalNetworkUrlUtil,
  getHostPort as getHostPortUtil,
  isUrlWhitelisted as isUrlWhitelistedUtil,
} from './runtime/NetworkUtils.js';
import {
  createAbortModule,
  createUtilityModule,
  createAgentModule,
} from './runtime/BuiltinModules.js';
import { runtimeLogger, redactSensitive, formatPathForLog, type LogPathPolicy } from './logger.js';
import { spawnUntrustedWorker } from './untrusted-executor.js';
import { parse as acornParse } from 'acorn';

export interface RuntimeConfig {
  callbacks?: {
    onToken?: StreamCallback;
    onLog?: LogCallback;
    onImage?: ImageCallback;
    onNodeStatus?: NodeStatusCallback;
    onDatabase?: DatabaseCallback;
    onLocalNetworkPermission?: LocalNetworkPermissionCallback;
  };
  abortSignal?: AbortSignal;
  moduleRegistry?: ModuleRegistry;
  flows?: Flow[];
  packageMacros?: Flow[];
  projectSettings?: ProjectSettings;
  moduleSettings?: Record<string, Record<string, unknown>>;
  /**
   * When true (default), workflows running inside a package context are
   * executed inside a Web Worker so that even a successful sandbox-escape
   * inside the workflow only reaches the worker's pristine realm — it
   * does not have access to the main thread's Tauri `invoke`, DOM, or
   * file handles. Set to `false` only for environments without `Worker`
   * (e.g. unit tests in Node) or to debug a regression.
   */
  useWorkerForUntrusted?: boolean;
}

declare global {
  interface Window {
    __TAURI__?: {
      core: {
        invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
      };
    };
  }
}

const AGENT_MEMORY_TABLE = '_agent_memory';

/**
 * Central capability map: each `Module.method` is matched against the
 * permission the active package needs in order to call it.
 *
 * For trusted local flows (no `currentPackageId` set) every method is
 * allowed — these maps only matter when an installed package is running.
 *
 * For package contexts the broker is **fail-closed**: a method is callable
 * only if it is named in *one of these two maps*. A new module method
 * that touches the network/FS/shell must be classified explicitly, either
 * with the permission it needs (here) or as safe-no-permission (below).
 * Anything not classified is denied with a clear "method not
 * permission-declared" error so the omission is loud rather than silent.
 *
 * Keys use the same `Module.method` shape that `host.call` receives.
 */
const REQUIRED_PERMISSIONS_BY_KIND: Readonly<Record<string, string>> = {
  // === Utility — network, services, code execution ========================
  'Utility.httpRequest': 'network',
  'Utility.comfyuiFreeMemory': 'network:local',
  'Utility.getServiceUrl': 'service:start',
  'Utility.ensureService': 'service:start',
  'Utility.ensureServiceByPort': 'service:start',
  // (Utility.logicBlock handler removed — it was a dead main-thread new Function bridge; the
  // compiler inlines logic_block into the worker script, gated by requireCodeExecute.)
  'Utility.requireCodeExecute': 'code:execute',
  // === Network module =====================================================
  'Network.httpRequest': 'network',
  'Network.fetch': 'network',
  // === Filesystem =========================================================
  'Filesystem.listFolder': 'filesystem:read',
  'Filesystem.readFile': 'filesystem:read',
  'Filesystem.calculateFileChunks': 'filesystem:read',
  'Filesystem.readChunkContent': 'filesystem:read',
  'Filesystem.writeFile': 'filesystem',
  'Filesystem.copyFile': 'filesystem',
  // Spawning OS processes is far more dangerous than reading/writing
  // files; it gets its own permission so granting `filesystem` no
  // longer implies the right to execute arbitrary commands. Round 5
  // reviewer #4.
  'Filesystem.runCommand': 'process:execute',
  // === Browser automation =================================================
  'Browser.create': 'browser',
  'Browser.createSession': 'browser',
  'Browser.navigate': 'browser',
  'Browser.evaluate': 'browser',
  'Browser.screenshot': 'browser',
  'Browser.getHtml': 'browser',
  'Browser.action': 'browser',
  'Browser.close': 'browser',
  'Browser.closeSession': 'browser',
  'Browser.request': 'browser',
  'Browser.extract': 'browser',
  'Browser.control': 'browser',
  'Browser.ensureServiceReadyByPort': 'service:start',
  'Browser.httpRequest': 'browser',
  // === Diffusion module (image+video generation) ==========================
  'Diffusion.getStatus': 'diffusion',
  'Diffusion.getDevices': 'diffusion',
  'Diffusion.listModels': 'diffusion',
  'Diffusion.registerModel': 'diffusion',
  'Diffusion.registerFromPath': 'diffusion',
  'Diffusion.inspectCheckpoint': 'diffusion',
  'Diffusion.loadModel': 'diffusion',
  'Diffusion.unloadModel': 'diffusion',
  'Diffusion.getLoaded': 'diffusion',
  'Diffusion.imageGenerate': 'diffusion',
  'Diffusion.videoGenerate': 'diffusion',
  'Diffusion.getJob': 'diffusion',
  'Diffusion.cancelJob': 'diffusion',
  // === Terminal automation ================================================
  'Terminal.create': 'terminal',
  'Terminal.createSession': 'terminal',
  'Terminal.aiControl': 'terminal',
  'Terminal.runCommand': 'terminal',
  'Terminal.sendKeys': 'terminal',
  'Terminal.screenshot': 'terminal',
  'Terminal.readOutput': 'terminal',
  'Terminal.close': 'terminal',
  'Terminal.closeSession': 'terminal',
  'Terminal.showWindow': 'terminal',
  // === AI module — calls external providers ===============================
  'AI.chat': 'network',
  'AI.vision': 'network',
  'AI.request': 'network',
  // === Audio — cloud APIs + filesystem ====================================
  'Audio.textToSpeech': 'network',
  'Audio.generateMusic': 'network',
  'Audio.speechToText': 'network',
  'Audio.saveAudio': 'filesystem',
  'Audio.grabAudio': 'filesystem:read',
  'Audio.appendAudio': 'filesystem',
  'Audio.getAudioDuration': 'filesystem:read',
  'Audio.fadeAudio': 'filesystem',
  'Audio.resolveServiceUrl': 'service:start',
  // === Image — generation + filesystem ====================================
  'Image.generate': 'network',
  'Image.resize': 'filesystem:read',
  'Image.save': 'filesystem',
  // === Video — services + heavy filesystem ================================
  'Video.getInfo': 'filesystem:read',
  'Video.extract': 'filesystem:read',
  'Video.extractAtTimestamps': 'filesystem:read',
  'Video.extractLastFrame': 'filesystem:read',
  'Video.extractBatch': 'filesystem:read',
  'Video.generate': 'network',
  'Video.generateVideoWan2GP': 'network',
  'Video.generateAvatar': 'network',
  'Video.save': 'filesystem',
  'Video.saveVideo': 'filesystem',
  'Video.extendVideos': 'filesystem',
  'Video.appendVideos': 'filesystem',
  'Video.mixAudio': 'filesystem',
  'Video.videoPip': 'filesystem',
  'Video.videoCaptions': 'filesystem',
  'Video.downloadVideo': 'network',
  // === Database — local SQLite =============================================
  'Database.execute': 'filesystem',
  // === Input — read-only file access via OS picker =========================
  'Input.readInputFile': 'filesystem:read',
  // === Vectorize — reads source images from disk ==========================
  'Vectorize.convert': 'filesystem:read',
};

/**
 * Methods that an installed package may call without holding any specific
 * permission. These either have no side effects (pure transforms) or
 * operate only on package-local state (memory scoped to `${packageId}`).
 *
 * Anything not in either this set or `REQUIRED_PERMISSIONS_BY_KIND` is
 * refused for package contexts.
 */
const NO_PERMISSION_REQUIRED_KINDS: ReadonlySet<string> = new Set([
  // Abort — control plane only.
  'Abort.check',
  'Abort.checkThrow',
  'Abort.getSignalId',
  // Utility — pure transformations + local in-memory bag.
  'Utility.template',
  'Utility.memoryRead',
  'Utility.memoryWrite',
  'Utility.memoryClear',
  // Agent — persisted memory is namespaced per package/flow scope.
  'Agent.get',
  'Agent.set',
  'Agent.memory',
  // FlowControl — control-flow primitives, no I/O.
  'FlowControl.execute',
  'FlowControl.executeMacro',
  'FlowControl.checkAborted',
  // Input — UI file pickers gate themselves on user interaction.
  'Input.pickFile',
  'Input.pickFolder',
  // Filesystem path lookups that don't open any file.
  'Filesystem.getDownloadsPath',
]);

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/**
 * Rewrite the keyword `await` to `yield` in the compiled workflow script
 * **without** touching string literals, template literals, comments, or
 * regex literals.
 *
 * The compiled workflow runs as a generator (so the host loop can pump
 * suspending operations through `host.call`); rewriting `await` to
 * `yield` is what hands control back to that loop. We must NOT touch any
 * `await` that lives inside user-visible content — strings, prompts,
 * comments, regex sources, or template-literal raw text — or we'd
 * silently corrupt prompts and JSON payloads.
 *
 * Implementation: parse the script with Acorn and rewrite every
 * `AwaitExpression` (and the `await` keyword inside `for await (…)`) by
 * its source range. The previous state-machine scanner handled the
 * common cases but the reviewer's round-4 #16 flagged that hand-rolled
 * JS parsing accumulates edge cases. Acorn is a real ES parser, so the
 * transform is correct against every construct the spec defines.
 *
 * If parsing fails (broken syntax in something the OAIY compiler emitted,
 * which would be a separate bug), we fall back to the original
 * state-machine scanner so the workflow still runs. The scanner is
 * exported as `rewriteAwaitToYieldScanner` for the regression-test
 * suite that locks down its 5 historical scenarios.
 */
export function rewriteAwaitToYield(script: string): string {
  try {
    return rewriteAwaitToYieldAst(script);
  } catch {
    // Acorn rejected the script. The fallback scanner is intentionally
    // forgiving so a downstream codegen bug doesn't take the runtime
    // out — but a parse failure here is still notable and worth a log
    // line in development. We deliberately don't throw.
    return rewriteAwaitToYieldScanner(script);
  }
}

/**
 * AST-based rewriter — preferred path. Handles every construct the JS
 * spec defines, including:
 *   * `await` inside template-literal `${…}` interpolations,
 *   * `for await (…)` async iteration loops,
 *   * arrow / function / class / object-method bodies,
 *   * any future JS syntax Acorn knows about.
 */
export function rewriteAwaitToYieldAst(script: string): string {
  const ast = acornParse(script, {
    ecmaVersion: 'latest',
    sourceType: 'script',
    allowAwaitOutsideFunction: true,
    allowReturnOutsideFunction: true,
    allowImportExportEverywhere: true,
  });

  type Replacement = { start: number; end: number };
  const replacements: Replacement[] = [];

  const visit = (node: unknown): void => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) {
      for (const child of node) visit(child);
      return;
    }
    const n = node as Record<string, unknown> & { type?: string; start?: number; end?: number };
    if (typeof n.type !== 'string') return;

    if (n.type === 'AwaitExpression' && typeof n.start === 'number') {
      // The `await` keyword spans node.start..node.start+5. The argument
      // expression starts after the keyword + whitespace.
      replacements.push({ start: n.start, end: n.start + 5 });
    } else if (
      n.type === 'ForOfStatement' &&
      (n as { await?: boolean }).await === true &&
      typeof n.start === 'number'
    ) {
      // `for await (…)` — the `await` token sits between `for` and the
      // opening paren of the loop. Find it by scanning the source slice.
      const left = (n as { left?: { start?: number } }).left;
      const searchEnd = typeof left?.start === 'number' ? left.start : (n.end ?? script.length);
      const slice = script.slice(n.start, searchEnd);
      const idx = slice.search(/\bawait\b/);
      if (idx !== -1) {
        const absStart = n.start + idx;
        replacements.push({ start: absStart, end: absStart + 5 });
      }
    }

    // Recurse into every child object/array property. We skip metadata
    // keys Acorn adds to nodes that don't contain syntax (`loc`, `range`).
    for (const key of Object.keys(n)) {
      if (key === 'loc' || key === 'range' || key === 'start' || key === 'end' || key === 'type')
        continue;
      visit(n[key]);
    }
  };

  visit(ast);

  // Apply in reverse order so positions don't shift.
  replacements.sort((a, b) => b.start - a.start);
  let out = script;
  for (const r of replacements) {
    out = out.slice(0, r.start) + 'yield' + out.slice(r.end);
  }
  return out;
}

/**
 * Original state-machine scanner. Kept as a fallback for the (unlikely)
 * case where the AST parser rejects the input, and as the test target
 * for the round-3 regression suite that locks down its 5 scenarios.
 */
export function rewriteAwaitToYieldScanner(script: string): string {
  // Identifier-character lookup (ASCII-only, sufficient for our generated code).
  const isIdent = (ch: number | undefined) =>
    ch !== undefined &&
    ((ch >= 65 && ch <= 90) || // A-Z
      (ch >= 97 && ch <= 122) || // a-z
      (ch >= 48 && ch <= 57) || // 0-9
      ch === 95 || // _
      ch === 36); // $

  // After these tokens, `/` starts a regex; after others, it's division.
  const REGEX_PRECEDED_BY = new Set([
    '(', ',', '=', ':', '[', '!', '&', '|', '?', '+', '-', '*', '/', '%', '^',
    '<', '>', '~', '{', '}', ';', 'return', 'typeof', 'delete', 'in',
    'instanceof', 'new', 'throw', 'void', 'yield', 'await', 'case', 'do',
    'else',
  ]);

  // Track the last non-whitespace, non-comment token to decide if `/` is a
  // regex or a division operator.
  let lastToken = ';';

  const out: string[] = [];
  let i = 0;
  const n = script.length;

  // Frame stack so we can re-enter "code mode" for `${...}` interpolations.
  // - tmpl: between backticks, walk characters until ` or ${
  // - interp: code inside ${...}, with brace depth tracking so we exit on the
  //   matching `}` — the contents are scanned with the full code-mode rules
  //   (so `await` inside an interpolation becomes `yield`).
  type Frame = { kind: 'tmpl' } | { kind: 'interp'; depth: number };
  const stack: Frame[] = [];
  const top = (): Frame | null => (stack.length === 0 ? null : stack[stack.length - 1]);

  while (i < n) {
    const frame = top();

    // Inside template-literal raw text: walk chars, watch for ` and ${.
    if (frame && frame.kind === 'tmpl') {
      const cc = script[i];
      if (cc === '\\') {
        out.push(script.slice(i, i + 2));
        i += 2;
        continue;
      }
      if (cc === '`') {
        out.push('`');
        i++;
        stack.pop();
        lastToken = 'string';
        continue;
      }
      if (cc === '$' && script[i + 1] === '{') {
        out.push('${');
        i += 2;
        stack.push({ kind: 'interp', depth: 1 });
        lastToken = '{';
        continue;
      }
      out.push(cc);
      i++;
      continue;
    }

    // Code mode (top-level OR inside `${...}` with frame.kind === 'interp').
    const c = script[i];
    const c2 = script[i + 1];

    // ---- Line comment ----
    if (c === '/' && c2 === '/') {
      const end = script.indexOf('\n', i + 2);
      const stop = end === -1 ? n : end;
      out.push(script.slice(i, stop));
      i = stop;
      continue;
    }

    // ---- Block comment ----
    if (c === '/' && c2 === '*') {
      const end = script.indexOf('*/', i + 2);
      const stop = end === -1 ? n : end + 2;
      out.push(script.slice(i, stop));
      i = stop;
      continue;
    }

    // ---- Regex literal? ----
    if (c === '/' && REGEX_PRECEDED_BY.has(lastToken)) {
      let j = i + 1;
      let inClass = false;
      while (j < n) {
        const cj = script[j];
        if (cj === '\\') { j += 2; continue; }
        if (cj === '[') inClass = true;
        else if (cj === ']') inClass = false;
        else if (cj === '/' && !inClass) { j++; break; }
        else if (cj === '\n') { break; }
        j++;
      }
      while (j < n && /[a-z]/i.test(script[j])) j++;
      out.push(script.slice(i, j));
      lastToken = 'regex';
      i = j;
      continue;
    }

    // ---- String literals: ' or " ----
    if (c === '"' || c === "'") {
      const quote = c;
      let j = i + 1;
      while (j < n) {
        const cj = script[j];
        if (cj === '\\') { j += 2; continue; }
        if (cj === quote) { j++; break; }
        if (cj === '\n') break;
        j++;
      }
      out.push(script.slice(i, j));
      lastToken = 'string';
      i = j;
      continue;
    }

    // ---- Template-literal opener: enter `tmpl` frame ----
    if (c === '`') {
      out.push('`');
      i++;
      stack.push({ kind: 'tmpl' });
      continue;
    }

    // ---- Brace tracking inside `${...}` interpolations ----
    if (frame && frame.kind === 'interp') {
      if (c === '{') {
        frame.depth++;
        out.push('{');
        lastToken = '{';
        i++;
        continue;
      }
      if (c === '}') {
        frame.depth--;
        out.push('}');
        i++;
        if (frame.depth === 0) {
          stack.pop(); // back into surrounding template literal
        } else {
          lastToken = '}';
        }
        continue;
      }
    }

    // ---- Identifier / keyword ----
    if (isIdent(c.charCodeAt(0)) && !(c >= '0' && c <= '9')) {
      let j = i;
      while (j < n && isIdent(script.charCodeAt(j))) j++;
      const word = script.slice(i, j);
      if (word === 'await') {
        out.push('yield');
      } else {
        out.push(word);
      }
      lastToken = word;
      i = j;
      continue;
    }

    // ---- Numeric literal ----
    if (c >= '0' && c <= '9') {
      let j = i;
      while (j < n && /[0-9a-fA-FxXoObBeE._+-]/.test(script[j])) j++;
      out.push(script.slice(i, j));
      lastToken = 'number';
      i = j;
      continue;
    }

    // ---- Whitespace ----
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      out.push(c);
      i++;
      continue;
    }

    // ---- Punctuation ----
    out.push(c);
    lastToken = c;
    i++;
  }

  return out.join('');
}

/**
 * The global-shadow preamble emitted at the top of every hardened-mode
 * workflow. We declare wide swathes of dangerous globals as `var` locals
 * so plain identifier references resolve to the shadow `undefined`, not
 * the real global. The list errs on the side of generosity: anything that
 * could leak the host realm (network, storage, navigation, timers, code
 * execution, Tauri internals) gets shadowed.
 *
 * Notes:
 *  - `eval` and `Function` are intentionally *not* shadowed here because
 *    redeclaring them as `var` is illegal in strict mode. They're enforced
 *    instead at the central permission gate (logic blocks require
 *    `code:execute`) and via the Tauri `invoke` denylist.
 *  - `globalThis` cannot be shadowed via `var` either (it's specced as a
 *    non-configurable property of the realm); the escape-pattern scanner
 *    blocks any *reference* to it before the script runs.
 */
export const HARDENED_SHADOW_PREAMBLE: string = `
// === Hardened mode: shadow dangerous globals ============================
// Active because this workflow is running inside a package context. Any
// network/FS/code-execute capability must go through the gated module
// methods registered with the host broker.
var fetch = undefined;
var XMLHttpRequest = undefined;
var WebSocket = undefined;
var EventSource = undefined;
var BroadcastChannel = undefined;
var document = undefined;
var window = undefined;
var self = undefined;
var parent = undefined;
var top = undefined;
var frames = undefined;
var location = undefined;
var history = undefined;
var localStorage = undefined;
var sessionStorage = undefined;
var indexedDB = undefined;
var caches = undefined;
var navigator = undefined;
var importScripts = undefined;
var Worker = undefined;
var SharedWorker = undefined;
var ServiceWorker = undefined;
var __TAURI__ = undefined;
var __TAURI_INVOKE__ = undefined;
var __TAURI_INTERNALS__ = undefined;
var require = undefined;
var process = undefined;
var Buffer = undefined;
// Timers are routed through the host broker if needed; raw access lets
// untrusted code spin or schedule work outside the workflow lifecycle.
var setTimeout = undefined;
var setInterval = undefined;
var clearTimeout = undefined;
var clearInterval = undefined;
var setImmediate = undefined;
var queueMicrotask = undefined;
var requestAnimationFrame = undefined;
var requestIdleCallback = undefined;
// =========================================================================
`;

/**
 * Source-level patterns that indicate the script is trying to reach the
 * realm global by a route the global-shadow preamble cannot block.
 *
 * BEST-EFFORT ONLY — NOT a security boundary. Because we scan the
 * comment-and-string-STRIPPED source, an escape route assembled from string
 * fragments (\`['cons'+'tructor']\`) or reached by computed/bracket access
 * (\`[]['constructor']['constructor']('return this')()\`) evades every pattern
 * below. The real isolation boundary is the Web Worker realm (separate realm,
 * no DOM, no __TAURI_INTERNALS__, brokered host bridge); this scanner is only a
 * second layer that blocks naive/literal forms. The first matching label is
 * returned for the error message; \`null\` means "no literal escape token found"
 * (NOT "proven safe").
 */
const SANDBOX_ESCAPE_PATTERNS: Array<[RegExp, string]> = [
  [/\bglobalThis\b/, 'globalThis'],
  [/\bFunction\s*\(/, 'Function constructor'],
  [/\beval\s*\(/, 'eval()'],
  [/\bimport\s*\(/, 'dynamic import()'],
  [/\bimport\s*\.\s*meta\b/, 'import.meta'],
  // `(function(){return this})()` and minor variants.
  [/\(\s*function\s*\([^)]*\)\s*\{\s*return\s+this\s*;?\s*\}\s*\)\s*\(/, '(function(){return this})()'],
  // `()=>this` — arrow-function workflow trick: arrows inherit `this` so a
  // bare `()=>this` evaluated at top level would yield the realm global.
  // Strict-mode wrapping turns this into `undefined`, but we still flag it
  // because it indicates intent to escape.
  [/\(\s*\)\s*=>\s*this\b/, '() => this arrow'],
  // `constructor.constructor("…")()` — string-evaluating via the Function
  // constructor reachable from any object's prototype.
  [/\.constructor\s*\.\s*constructor\b/, 'constructor.constructor chain'],
];

/**
 * Returns the label of the first sandbox-escape pattern that the script
 * matches, or `null` if it looks safe to run in hardened mode.
 *
 * The scan reuses the comment-and-string stripper from
 * `verify-dist-fresh.cjs` (re-implemented here so oaiy-core has no script
 * dependency): it walks the source character-by-character, stripping line
 * comments, block comments, and string/template literals so the regexes
 * only match against actual code.
 */
export function detectSandboxEscape(script: string): string | null {
  const stripped = stripCommentsAndStringsFromCode(script);
  for (const [re, label] of SANDBOX_ESCAPE_PATTERNS) {
    if (re.test(stripped)) {
      return label;
    }
  }
  return null;
}

/**
 * Strip line/block comments and the bodies of string and template literals
 * so an escape-pattern regex only matches actual executable code.
 */
function stripCommentsAndStringsFromCode(src: string): string {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const c2 = src[i + 1];
    if (c === '/' && c2 === '/') {
      // A // comment ends at ANY ECMAScript LineTerminator (LF, CR, U+2028 LS, U+2029 PS) — the
      // engine resumes executing code after one, so the scanner must stop consuming the comment
      // there too; otherwise a separator hidden inside a comment lets a globalThis/Function()
      // payload slip past the escape-pattern regexes (defense-in-depth with the compiler's
      // allowlisting of values that reach generated comments).
      while (i < n) {
        const cc = src.charCodeAt(i);
        // LF, CR, U+2028 (LS), U+2029 (PS) all end a // comment in JS — stop at any of them.
        if (cc === 10 || cc === 13 || cc === 0x2028 || cc === 0x2029) break;
        i++;
      }
      continue;
    }
    if (c === '/' && c2 === '*') {
      i += 2;
      while (i < n - 1 && !(src[i] === '*' && src[i + 1] === '/')) i++;
      i += 2;
      continue;
    }
    if (c === '"' || c === "'" || c === '`') {
      const quote = c;
      out += c + c;
      i++;
      while (i < n) {
        const cc = src[i];
        if (cc === '\\') { i += 2; continue; }
        i++;
        if (cc === quote) break;
      }
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

/**
 * Returns true when the compiled-script debug dump should be written to disk.
 *
 * The debug dump can leak prompts, file paths and inlined constants — so we
 * gate it behind two conditions that must both be true:
 *   1. We're in a development build (`process.env.NODE_ENV !== 'production'`,
 *      or Vite's `import.meta.env.DEV === true`).
 *   2. The user explicitly opted in via `OAIY_DEBUG_DUMP_SCRIPT=1`.
 *
 * Production binaries never write the dump, regardless of env vars.
 */
function shouldDumpCompiledScript(): boolean {
  // We deliberately avoid `import.meta` here — ts-jest's CommonJS preset
  // can't parse it. Vite sets `NODE_ENV=production` for production builds,
  // and that's the same signal we want.
  if (typeof process !== 'undefined' && process.env?.NODE_ENV === 'production') {
    return false;
  }

  // Hard opt-in switch. The flag must be present and truthy.
  const flag =
    typeof process !== 'undefined' ? process.env?.OAIY_DEBUG_DUMP_SCRIPT : undefined;
  return flag === '1' || flag === 'true';
}

/**
 * Redact obvious secrets before writing the compiled script to disk.
 * This is *not* a substitute for not writing secrets in the first place — the
 * compiled script normally references constants by name and does not inline
 * their values, but a defensive redact still helps with the long tail.
 */
function redactCompiledScript(script: string): string {
  const patterns: Array<[RegExp, string]> = [
    [/("(?:sk|pk|xai|gho|ghp|ghu|ghs|ghr|hf|sk-ant)_[A-Za-z0-9_-]{8,})"/g, '"<redacted-token>"'],
    [/(Bearer\s+)[A-Za-z0-9._\-]{16,}/g, '$1<redacted-token>'],
    [/("Authorization"\s*:\s*")[^"]+(")/g, '$1<redacted>$2'],
    [/("[Xx]-[Aa]pi-[Kk]ey"\s*:\s*")[^"]+(")/g, '$1<redacted>$2'],
    [/("apiKey"\s*:\s*")[^"]+(")/g, '$1<redacted>$2'],
    [/("password"\s*:\s*")[^"]+(")/g, '$1<redacted>$2'],
  ];
  let out = script;
  for (const [re, repl] of patterns) {
    out = out.replace(re, repl);
  }
  return out;
}

function formatLogArgs(args: unknown[]): string {
  return args
    .map(a => (typeof a === 'object' && a !== null ? safeStringify(a) : String(a)))
    .join(' ');
}

export class OAIYRuntime {
  private onToken: StreamCallback | null = null;
  private onLog: LogCallback | null = null;
  private onImage: ImageCallback | null = null;
  private onNodeStatus: NodeStatusCallback | null = null;
  private onDatabase: DatabaseCallback | null = null;
  private onLocalNetworkPermission: LocalNetworkPermissionCallback | null = null;
  private abortSignal: AbortSignal | null = null;
  private agentMemory: BoundedMap<string, string | number | boolean | object> = new BoundedMap({
    maxEntries: 1000,
    maxValueSize: 1024 * 1024,
  });
  /**
   * Per-scope tracking for `loadPersistedMemory`. The previous boolean
   * was shared across every package/flow scope this runtime instance ever
   * saw — so once any scope had loaded, switching to a new package or
   * flow would skip loading its persisted entries entirely. This Set
   * records each scope identifier that has been hydrated.
   */
  private loadedMemoryScopes: Set<string> = new Set();
  private moduleRegistry: ModuleRegistry | null = null;
  private loadedRuntimeModules: Map<string, RuntimeModule> = new Map();
  private moduleSettings: Record<string, Record<string, unknown>> = {};
  private availableFlows: Flow[] = [];
  private packageMacros: Flow[] = [];
  private projectSettings: ProjectSettings = {};
  /**
   * Project constants (constant key → value), including rehydrated secret
   * values such as API keys. Wired into `ctx.getConstant` so module runtimes
   * can resolve an `apiKeyConstant` NAME to its actual value at run time.
   * Empty until the host calls `setProjectConstants`; `getConstant` then
   * returns undefined for unknown names, which every module already guards.
   */
  private projectConstants: Record<string, string> = {};

  /**
   * Subset of `projectConstants` keys that are SECRET (API keys etc.). Used by `getConstant`
   * to decide which permission an untrusted package needs to resolve a given constant —
   * `secrets:read` (high-risk) for these, `constants:read` for the rest.
   */
  private secretConstantNames: Set<string> = new Set();

  private currentFlowId: string | null = null;
  private currentPackageId: string | null = null;

  private useClaudeForAI: boolean = false;
  private currentJobId: string | null = null;
  private onYieldForAI: ((request: any) => Promise<string>) | null = null;

  // Store module functions for the host loop
  private moduleFunctionHandlers: Record<string, (...args: any[]) => any> = {};

  // Per-job cross-module accessor (ctx.modules). Cached on the instance so method
  // identity is stable and a module's `ctx.modules.X` references THIS runtime's
  // handlers. See buildModulesProxy().
  private modulesProxy?: Record<string, Record<string, (...args: any[]) => any>>;
  private readonly moduleMethodProxies = new Map<string, Record<string, (...args: any[]) => any>>();

  /**
   * Whether to run hardened (package-context) workflows in a Web Worker.
   * Defaults to `true`; the in-thread fallback is selected automatically
   * when `Worker` is unavailable (e.g. Node-based unit tests).
   */
  private useWorkerForUntrusted: boolean = true;

  constructor(configOrOnToken?: RuntimeConfig | StreamCallback, ...legacyArgs: unknown[]) {
    if (configOrOnToken && typeof configOrOnToken === 'object' && !('call' in configOrOnToken)) {
      const config = configOrOnToken as RuntimeConfig;
      this.onToken = config.callbacks?.onToken || null;
      this.onLog = config.callbacks?.onLog || null;
      this.onImage = config.callbacks?.onImage || null;
      this.onNodeStatus = config.callbacks?.onNodeStatus || null;
      this.onDatabase = config.callbacks?.onDatabase || null;
      this.onLocalNetworkPermission = config.callbacks?.onLocalNetworkPermission || null;
      this.abortSignal = config.abortSignal || null;
      this.moduleRegistry = config.moduleRegistry || null;
      this.availableFlows = config.flows || [];
      this.packageMacros = config.packageMacros || [];
      this.projectSettings = config.projectSettings || {};
      this.moduleSettings = config.moduleSettings || {};
      if (config.useWorkerForUntrusted !== undefined) {
        this.useWorkerForUntrusted = config.useWorkerForUntrusted;
      }
    } else {
      this.onToken = (configOrOnToken as StreamCallback) || null;
      this.onLog = (legacyArgs[0] as LogCallback) || null;
      this.onImage = (legacyArgs[1] as ImageCallback) || null;
      this.onNodeStatus = (legacyArgs[2] as NodeStatusCallback) || null;
      this.abortSignal = (legacyArgs[3] as AbortSignal) || null;
      this.onDatabase = (legacyArgs[4] as DatabaseCallback) || null;
      this.onLocalNetworkPermission = (legacyArgs[5] as LocalNetworkPermissionCallback) || null;
    }

    this.registerBuiltinModules();
  }

  setProjectSettings(settings: ProjectSettings): void {
    this.projectSettings = settings;
  }

  /**
   * Provide project constants (constant key → value) to the runtime so module
   * code can resolve API-key constants via `ctx.getConstant(name)`. Pass the
   * rehydrated values (secrets included); callers that have no constants may
   * omit this entirely.
   */
  setProjectConstants(constants: Record<string, string>, secretNames?: Set<string>): void {
    this.projectConstants = constants || {};
    this.secretConstantNames = secretNames ?? new Set();
  }

  /**
   * Granted capability set for the active package context. The desktop
   * shell calls `setPackagePermissions` with the trust-level + permissions
   * for an installed package right before running its workflow.
   *
   * For trusted local flows (no package context), every permission check
   * returns true.
   */
  private packagePermissions: Set<string> = new Set();

  setPackagePermissions(permissions: string[]): void {
    this.packagePermissions = new Set(permissions ?? []);
  }

  /**
   * Explicit Tauri `invoke` function provided by the desktop shell. The
   * runtime no longer reaches into `window.__TAURI__` — callers must wire
   * this in (the desktop entry point does so during boot).
   */
  private invokeFn: ((cmd: string, args?: Record<string, unknown>) => Promise<unknown>) | null = null;

  setTauriInvoke(invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>): void {
    this.invokeFn = invoke;
  }

  /**
   * Tauri commands that workflow / module code is *allowed* to call.
   * The broker is **fail-closed**: a command not in this set is refused.
   *
   * The list is the union of every command currently used by built-in
   * modules (ai/audio/browser/database/filesystem/image/input/network/
   * terminal/utility/vectorize/video). Adding a new module method that
   * needs a new Tauri command means adding the command here — which is
   * a deliberate code-review trigger so privileged commands don't ship
   * reachable from untrusted package code by accident.
   *
   * Defence-in-depth: each underlying Rust command also performs its own
   * authorisation (path canonicalisation against the user-directory
   * allow-list, run_command bundled-binary resolution, signed-package
   * trust checks, etc.). The allowlist here is a tight upper bound; the
   * Rust side is the lower bound.
   */
  private static readonly ALLOWED_TAURI_COMMANDS: ReadonlySet<string> = new Set([
    // === Service orchestration ============================================
    'ensure_service_ready',
    'ensure_service_ready_by_port',
    'get_service_port',
    // === Media playback URL minting (still goes via media server) =========
    'get_media_url',
    // Temp-dir cleanup a module triggers after writing intermediate media. The
    // node-host/desktop scope it to the temp root. Was wrongly denylisted.
    'cleanup_temp_dir',
    // === Image plugin (native desktop / sharp node-host; canvas fallback in web) =
    'resize_image',
    // === HTTP gateway (policy-checked downstream) =========================
    'http_request',
    // === Video plugin (FFmpeg-backed) =====================================
    'get_video_info',
    'extract_video_frames',
    'extract_video_frames_batch',
    // ffmpeg/ffprobe exec for the audio/video modules. EVERY host restricts this to
    // ffmpeg+ffprobe ONLY (tauri-shim → ffmpeg.wasm, node-host → native ffmpeg, the
    // desktop Rust plugin resolves it to the bundled binary — "Rust side is the lower
    // bound"), so it is NOT arbitrary command execution. It was wrongly on the
    // DENYLIST, which silently broke audio_fade/append/mixer/save_audio +
    // video_append/pip/captions/extend on every host.
    'plugin:oaiy-filesystem|run_command',
    // === Filesystem helpers (path-validated downstream) ===================
    'get_downloads_path',
    'plugin:oaiy-filesystem|get_app_data_dir',
    'plugin:oaiy-filesystem|get_temp_dir',
    'plugin:oaiy-filesystem|get_downloads_path',
    'plugin:oaiy-filesystem|list_folder',
    'plugin:oaiy-filesystem|read_file',
    'plugin:oaiy-filesystem|read_chunk_content',
    'plugin:oaiy-filesystem|write_file',
    'plugin:oaiy-filesystem|native_copy_file',
    'plugin:oaiy-filesystem|delete_file',
    'plugin:oaiy-filesystem|pick_file',
    'plugin:oaiy-filesystem|pick_folder',
    // === Browser plugin ===================================================
    'plugin:oaiy-browser|webview_create',
    'plugin:oaiy-browser|webview_navigate',
    'plugin:oaiy-browser|webview_action',
    'plugin:oaiy-browser|webview_screenshot',
    'plugin:oaiy-browser|webview_evaluate',
    'plugin:oaiy-browser|webview_close',
    // V2 unified Chromium-CDP / WebView / HTTP backend (commands.rs)
    'plugin:oaiy-browser|browser_v2_create_session',
    'plugin:oaiy-browser|browser_v2_close_session',
    'plugin:oaiy-browser|browser_v2_list_sessions',
    'plugin:oaiy-browser|browser_v2_goto',
    'plugin:oaiy-browser|browser_v2_evaluate_json',
    'plugin:oaiy-browser|browser_v2_get_html',
    'plugin:oaiy-browser|browser_v2_get_text',
    'plugin:oaiy-browser|browser_v2_get_title',
    'plugin:oaiy-browser|browser_v2_get_url',
    'plugin:oaiy-browser|browser_v2_wait_for_selector',
    'plugin:oaiy-browser|browser_v2_screenshot',
    'plugin:oaiy-browser|browser_v2_get_cookies',
    'plugin:oaiy-browser|browser_v2_set_cookies',
    'plugin:oaiy-browser|browser_v2_http_request',
    // Engine-update / managed-Chromium control surface
    'plugin:oaiy-browser|browser_engine_get_status',
    'plugin:oaiy-browser|browser_engine_check_updates',
    'plugin:oaiy-browser|browser_engine_set_custom_path',
    'plugin:oaiy-browser|browser_engine_use_bundled',
    // === Terminal plugin ==================================================
    'plugin:oaiy-terminal|terminal_create',
    'plugin:oaiy-terminal|terminal_send_keys',
    'plugin:oaiy-terminal|terminal_close',
    'plugin:oaiy-terminal|terminal_screenshot',
    'plugin:oaiy-terminal|terminal_read_output',
    'plugin:oaiy-terminal|terminal_show_window',
    // plugin-diffusion + plugin-llm (native Rust inference) removed 2026-05-20
    // (flow-builder / local-AI-only pivot). Image generation now goes through
    // ComfyUI / Wan2GP HTTP (image_gen node); LLM through external OpenAI-
    // compatible engines (AI LLM node). Source backed up at repo-root external/.

    // plugin-tts (DramaBox) removed 2026-05-20 — TTS handled externally
    // via HTTP (text_to_speech node External mode). Native command surface
    // gone; backed up at repo-root external/plugin-tts.

    // plugin-agent: LLM-driven assistant with tool access to flows and
    // node outputs. Phase 1 surface: status + list_tools + call_tool +
    // set_api_base. The chat panel calls these directly (not from
    // sandboxed workflow code), but they're on the allowlist so future
    // workflow nodes that want to invoke the agent (e.g. an
    // "ask-the-agent" node) can do so too.
    'plugin:oaiy-agent|agent_get_status',
    'plugin:oaiy-agent|agent_list_tools',
    'plugin:oaiy-agent|agent_call_tool',
    'plugin:oaiy-agent|agent_set_api_base',
  ]);

  /**
   * Tauri commands that workflow/module code is *never* allowed to call,
   * regardless of trust level. The defence-in-depth layer behind the
   * allowlist: even if a future refactor of `ALLOWED_TAURI_COMMANDS`
   * accidentally lists one of these names, this still refuses.
   */
  private static readonly DENYLISTED_TAURI_COMMANDS: ReadonlySet<string> = new Set([
    'terminal_set_approved',
    'set_package_trust',
    'install_package',
    'uninstall_package',
    'set_api_config',
    'rotate_api_key',
    'get_api_key',
    'set_install_config',
    'store_secret',
    'store_secrets',
    'get_secret',
    'get_secrets',
    'delete_secret',
    'restart_app',
    'exit_app',
    'recompile_packages',
    'clear_cache',
    'reload_macros',
    'write_cli_output',
    // NOTE: run_command + cleanup_temp_dir moved to the ALLOWLIST — both are
    // invoked by built-in audio/video/image modules and are host-constrained
    // (run_command → ffmpeg/ffprobe only; cleanup_temp_dir → temp root only).
    // Denylisting them silently broke those nodes (the deny check runs first).
  ]);

  /**
   * Build the per-module Tauri invoke broker. The broker:
   *   1. Refuses denylisted commands outright (defence in depth).
   *   2. Refuses commands not in the allowlist (fail-closed).
   *   3. For allowed commands, delegates to the explicit invoke function
   *      provided by the desktop shell (no `window.__TAURI__` lookups).
   *
   * `_moduleId` is reserved for future audit-logging.
   */
  private buildModuleInvokeBroker(_moduleId: string): { invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T> } | undefined {
    if (!this.invokeFn) return undefined;
    const deny = OAIYRuntime.DENYLISTED_TAURI_COMMANDS;
    const allow = OAIYRuntime.ALLOWED_TAURI_COMMANDS;
    const inner = this.invokeFn;
    const log = this.onLog;
    return {
      invoke: <T,>(cmd: string, args?: Record<string, unknown>): Promise<T> => {
        if (deny.has(cmd)) {
          const message = `Refusing module invoke '${cmd}': command is on the denylist`;
          log?.({ source: 'Runtime', message, type: 'error' });
          return Promise.reject(new Error(message));
        }
        if (!allow.has(cmd)) {
          const message =
            `Refusing module invoke '${cmd}': command is not on the allowlist. ` +
            `Add it to ALLOWED_TAURI_COMMANDS in runtime.ts if it's needed by a built-in module.`;
          log?.({ source: 'Runtime', message, type: 'error' });
          return Promise.reject(new Error(message));
        }
        return inner(cmd, args) as Promise<T>;
      },
    };
  }

  hasPackagePermission(permission: string): boolean {
    if (!this.currentPackageId) {
      // Trusted, locally-authored flow — no gate.
      return true;
    }
    if (this.packagePermissions.has(permission)) return true;
    // `filesystem` implies `filesystem:read`; `network` implies `network:local`.
    if (permission === 'filesystem:read' && this.packagePermissions.has('filesystem')) return true;
    if (permission === 'network:local' && this.packagePermissions.has('network')) return true;
    return false;
  }

  setFlowContext(flowId: string | null, packageId?: string | null): void {
    const previousScope = this.currentMemoryScope();
    this.currentFlowId = flowId;
    this.currentPackageId = packageId || null;
    const newScope = this.currentMemoryScope();
    // When the scope changes, the in-memory bag still holds entries from
    // the previous package/flow — those would otherwise leak into the new
    // scope on read. Clear it; the next `Agent.memory` call hydrates the
    // new scope from persistent storage.
    if (previousScope !== newScope) {
      this.agentMemory.clear();
      // The bag is shared across scopes, so wiping it also "unloads" every scope's persisted
      // data — clear the load-tracking too, or a REVISITED scope would read as already-loaded
      // and stay empty (loadPersistedMemory would skip the re-hydrate).
      this.loadedMemoryScopes.clear();
    }
    if (flowId) {
      runtimeLogger.debug(`Flow context set: flowId=${flowId}${packageId ? `, packageId=${packageId}` : ''}`);
    }
  }

  setClaudeAsAI(enabled: boolean, jobId: string | null, yieldCallback: any | null): void {
    this.useClaudeForAI = enabled;
    this.currentJobId = jobId;
    this.onYieldForAI = yieldCallback;
  }

  getFlowContext(): { flowId: string | null; packageId: string | null } {
    return { flowId: this.currentFlowId, packageId: this.currentPackageId };
  }

  addToLocalNetworkWhitelist(hostPort: string): void {
    const currentWhitelist = this.projectSettings.localNetworkWhitelist || [];
    if (!currentWhitelist.includes(hostPort)) {
      this.projectSettings = {
        ...this.projectSettings,
        localNetworkWhitelist: [...currentWhitelist, hostPort],
      };
    }
  }

  private isLocalNetworkUrl(url: string): boolean {
    return isLocalNetworkUrlUtil(url);
  }

  private getHostPort(url: string): string {
    return getHostPortUtil(url);
  }

  private isUrlWhitelisted(url: string): boolean {
    return isUrlWhitelistedUtil(
      url,
      this.projectSettings.localNetworkWhitelist || [],
      this.projectSettings.allowAllLocalNetwork
    );
  }

  /**
   * Module-facing fetch. Routes every request through the same policy as
   * `Utility.httpRequest` (local-network permission prompt, SSRF guard,
   * DNS pinning) and returns a `Response`-shaped object so module code that
   * was written against the Web Fetch API keeps working.
   *
   * The `_meta` parameter is reserved for future audit logging — modules
   * should pass `{ moduleId, nodeId, purpose }` so we can surface which
   * node initiated a request to the user when prompting.
   */
  private async policyFetch(
    url: string,
    options?: RequestInit,
    _meta?: { moduleId?: string; nodeId?: string; purpose?: string }
  ): Promise<Response> {
    const method = (options?.method || 'GET').toUpperCase();

    // Normalise headers from any of the three valid Fetch shapes.
    const headers: Record<string, string> = {};
    const h = options?.headers;
    if (h instanceof Headers) {
      h.forEach((v, k) => { headers[k] = v; });
    } else if (Array.isArray(h)) {
      for (const [k, v] of h) headers[k] = String(v);
    } else if (h && typeof h === 'object') {
      for (const [k, v] of Object.entries(h)) headers[k] = String(v);
    }

    // Normalise body to a string (sufficient for the Tauri http_request
    // command). Binary/multipart bodies aren't supported here — those
    // workflows should go through dedicated Tauri commands.
    let body: string | undefined;
    if (options?.body !== undefined && options.body !== null) {
      if (typeof options.body === 'string') {
        body = options.body;
      } else if (typeof Blob !== 'undefined' && options.body instanceof Blob) {
        body = await options.body.text();
      } else {
        try { body = JSON.stringify(options.body); } catch { body = String(options.body); }
      }
    }

    const result = await this.performHttpRequest(url, method, headers, body);

    // Build a minimal Response-shaped object. Headers are case-insensitive,
    // so we normalise to lower-case keys before rebuilding.
    const respHeaders = new Headers();
    for (const [k, v] of Object.entries(result.headers)) {
      try { respHeaders.set(k, v); } catch { /* ignore invalid header names */ }
    }

    const status = result.status;
    const bodyText = result.body;
    // The Tauri http_request native command base64-encodes binary
    // responses (images, audio, video, octet-stream, etc.) and signals
    // that with `bodyIsBase64: true`. The previous wrapper ignored this
    // flag and treated every body as text, so `arrayBuffer()` / `blob()`
    // would re-encode the base64 string as UTF-8 bytes — corrupting any
    // image or binary download routed through `Network.fetch`.
    //
    // Now: when the flag is set we decode once into a Uint8Array and
    // serve every binary accessor from that decoded buffer. `text()` on
    // a binary response surfaces the raw base64 string (callers that
    // care about correctness should use `arrayBuffer()` or `blob()`
    // instead).
    const isBinary = result.bodyIsBase64 === true;
    let binaryBytes: Uint8Array | null = null;
    const decodeBinary = (): Uint8Array => {
      if (binaryBytes) return binaryBytes;
      // atob is available in browsers, Tauri webview, Node 16+, and Workers.
      const bin = (typeof atob !== 'undefined') ? atob(bodyText) : Buffer.from(bodyText, 'base64').toString('binary');
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i) & 0xff;
      binaryBytes = u8;
      return u8;
    };

    return {
      status,
      ok: status >= 200 && status < 300,
      statusText: '',
      headers: respHeaders,
      url,
      redirected: false,
      type: 'basic',
      body: null,
      bodyUsed: false,
      clone() { return this; },
      async text() { return bodyText; },
      async json() { return JSON.parse(bodyText); },
      async arrayBuffer() {
        if (isBinary) {
          const u8 = decodeBinary();
          // Slice into a fresh ArrayBuffer so callers can mutate without
          // touching our internal cache.
          const out = new ArrayBuffer(u8.byteLength);
          new Uint8Array(out).set(u8);
          return out;
        }
        const enc = new TextEncoder();
        return enc.encode(bodyText).buffer;
      },
      async blob() {
        if (isBinary) {
          const u8 = decodeBinary();
          const ct = respHeaders.get('content-type') || 'application/octet-stream';
          // Slice the underlying buffer so the Blob holds an independent copy.
          return new Blob([u8.slice(0)], { type: ct });
        }
        return new Blob([bodyText]);
      },
      async formData() {
        throw new Error('formData() is not supported by the OAIY policy fetch');
      },
      async bytes() {
        if (isBinary) {
          // Return a defensive copy of the decoded buffer.
          return decodeBinary().slice(0);
        }
        const enc = new TextEncoder();
        return enc.encode(bodyText);
      },
    } as unknown as Response;
  }

  private async performHttpRequest(
    url: string,
    method: string,
    headers: Record<string, string>,
    body?: string
  ): Promise<{ status: number; headers: Record<string, string>; body: string; bodyIsBase64?: boolean }> {
    const isLocalNetwork = this.isLocalNetworkUrl(url);

    if (isLocalNetwork && !this.isUrlWhitelisted(url)) {
      if (this.onLocalNetworkPermission) {
        const hostPort = this.getHostPort(url);
        const response = await this.onLocalNetworkPermission({
          url,
          hostPort,
          purpose: 'HTTP request from workflow',
        });

        if (!response.allowed) {
          throw new Error(`Local network access denied for ${hostPort}`);
        }

        if (response.remember) {
          this.addToLocalNetworkWhitelist(hostPort);
        }
      } else {
        throw new Error(`Local network access to ${this.getHostPort(url)} is not allowed`);
      }
    }

    if (this.invokeFn) {
      const result = (await this.invokeFn('http_request', {
        request: {
          url,
          method,
          headers,
          body: body || null,
          follow_redirects: true,
          max_redirects: 10,
          allow_private_networks: isLocalNetwork,
        },
      })) as {
        status: number;
        headers: Record<string, string>;
        body: string;
        url: string;
        bodyIsBase64?: boolean;
      };

      return {
        status: result.status,
        headers: result.headers,
        body: result.body,
        bodyIsBase64: result.bodyIsBase64,
      };
    }

    // Non-Tauri fallback (used in unit tests / standalone Node). The
    // browser/Node `fetch` does not perform DNS pinning, so for any
    // hostname-based local-network URL we have already obtained explicit
    // permission above. We additionally refuse public-network requests
    // outside the Tauri shell so that misconfigured hosts don't leak
    // requests through an unprotected path in production builds.
    const isProductionBuild =
      (typeof process !== 'undefined' && process.env?.NODE_ENV === 'production');
    if (isProductionBuild && !isLocalNetwork) {
      throw new Error(
        'OAIY runtime is not embedded in the Tauri shell — refusing to make ' +
          'public-network requests through the browser fetch fallback. ' +
          'Run inside the desktop app or use the Tauri http_request command.'
      );
    }

    const fetchOptions: RequestInit = {
      method,
      headers,
    };

    if (body) {
      fetchOptions.body = body;
    }

    if (this.abortSignal) {
      fetchOptions.signal = this.abortSignal;
    }

    const response = await fetch(url, fetchOptions);
    const responseBody = await response.text();

    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    return {
      status: response.status,
      headers: responseHeaders,
      body: responseBody,
    };
  }

  /**
   * Build the scope-prefix for the active workflow's agent memory.
   *
   * Memory entries are stored with a `scope` field so two workflows in
   * different packages — or two flows in the same package — see
   * isolated views even though they share the underlying
   * `_agent_memory` collection. Cross-flow memory was a frequent
   * surprise vector in pre-fix builds, so we now require an explicit
   * `Agent.shareAcrossScopes` opt-in (not yet wired) before crossing
   * boundaries.
   */
  private currentMemoryScope(): string {
    const pkg = this.currentPackageId ?? '_global';
    const flow = this.currentFlowId ?? '_default';
    return `${pkg}::${flow}`;
  }

  private async loadPersistedMemory(): Promise<void> {
    if (!this.onDatabase) {
      return;
    }
    const scope = this.currentMemoryScope();
    if (this.loadedMemoryScopes.has(scope)) {
      return;
    }
    // Mark the scope loaded eagerly so concurrent calls don't double-load.
    // If the query throws, we leave the marker set — the error path logs
    // and the next caller will see "loaded" with whatever entries we got
    // before failure. This matches the previous single-boolean behaviour.
    this.loadedMemoryScopes.add(scope);
    try {
      const result = await this.onDatabase({
        operation: 'query',
        storageType: 'collection',
        collectionName: AGENT_MEMORY_TABLE,
        filter: { scope },
      });

      if (result.success && result.data && Array.isArray(result.data)) {
        for (const doc of result.data as Array<{ key: string; value: unknown; scope?: string }>) {
          // Defensive: skip rows that somehow got here from a different
          // scope (older oaiy builds didn't filter at write time).
          if (doc.scope && doc.scope !== scope) continue;
          if (doc.key && doc.value !== undefined) {
            try {
              const value = typeof doc.value === 'string' ? JSON.parse(doc.value) : doc.value;
              this.agentMemory.set(doc.key, value as string | number | boolean | object);
            } catch {
              this.agentMemory.set(doc.key, doc.value as string | number | boolean | object);
            }
          }
        }
        if (this.onLog) {
          this.onLog({
            source: 'Agent',
            message: `Loaded ${result.data.length} persisted memory entries (scope: ${scope})`,
            type: 'info',
          });
        }
      }
    } catch (err) {
      if (this.onLog) {
        this.onLog({ source: 'Agent', message: `Failed to load persisted memory: ${err}`, type: 'error' });
      }
    }
  }

  private async persistMemoryEntry(key: string, value: unknown): Promise<void> {
    if (!this.onDatabase) return;
    const scope = this.currentMemoryScope();
    try {
      const serializedValue = JSON.stringify(value);
      // Remove any existing rows for this (scope,key) FIRST (by id — see
      // deleteMemoryRows), then insert. The collection `delete` op only matches
      // filter.id, so the old `{scope,key}` delete here was a silent no-op and rows
      // accumulated unbounded with ambiguous load order.
      await this.deleteMemoryRows(scope, key);
      await this.onDatabase({
        operation: 'insert',
        storageType: 'collection',
        collectionName: AGENT_MEMORY_TABLE,
        data: { scope, key, value: serializedValue },
      });
    } catch (err) {
      if (this.onLog) {
        this.onLog({ source: 'Agent', message: `Failed to persist memory entry '${key}': ${err}`, type: 'error' });
      }
    }
  }

  private async deletePersistedMemoryEntry(key: string): Promise<void> {
    if (!this.onDatabase) return;
    const scope = this.currentMemoryScope();
    try {
      await this.deleteMemoryRows(scope, key);
    } catch (err) {
      if (this.onLog) {
        this.onLog({ source: 'Agent', message: `Failed to delete persisted memory entry '${key}': ${err}`, type: 'error' });
      }
    }
  }

  /**
   * Delete persisted agent-memory rows matching (scope,key). The collection `delete`
   * operation only matches `filter.id`, so we query for matching rows first and delete
   * each by id. Without this, persist/delete were no-ops and rows grew without bound.
   */
  private async deleteMemoryRows(scope: string, key: string): Promise<void> {
    if (!this.onDatabase) return;
    const existing = await this.onDatabase({
      operation: 'query',
      storageType: 'collection',
      collectionName: AGENT_MEMORY_TABLE,
      filter: { scope, key },
    });
    if (existing.success && Array.isArray(existing.data)) {
      for (const row of existing.data) {
        const id = (row as { id?: unknown; _id?: unknown }).id ?? (row as { _id?: unknown })._id;
        if (id != null) {
          await this.onDatabase({
            operation: 'delete',
            storageType: 'collection',
            collectionName: AGENT_MEMORY_TABLE,
            filter: { id: id as string },
          });
        }
      }
    }
  }

  private registerModuleInternally(name: string, methods: Record<string, (...args: any[]) => any>) {
    const methodNames: string[] = [];
    for (const [methodName, fn] of Object.entries(methods)) {
      this.moduleFunctionHandlers[`${name}.${methodName}`] = fn;
      methodNames.push(methodName);
    }
    // NOTE: module methods are intentionally NOT published to globalThis[name].
    // That used to back module-internal cross-module lookup (a package's runtime
    // calling Browser.*), but a single shared global is last-writer-wins across
    // concurrent jobs — the exact hazard that pinned maxConcurrency=1. Module
    // runtime code must reach other modules through the per-job `ctx.modules`
    // accessor (buildModulesProxy) or `ctx.callModule(...)`, both of which dispatch
    // through THIS job's handlers. The compiled-workflow path already resolves
    // `Browser.foo()` through the per-job getModulesShim → moduleFunctionHandlers,
    // so nothing in the engine depends on the old global.
    console.log(`[OAIYRuntime] Registered module '${name}' with methods: ${methodNames.join(', ')}`);
  }

  /**
   * Per-job cross-module accessor backing `ctx.modules`. A two-level Proxy that
   * resolves `ctx.modules.<Module>.<method>` to THIS runtime's per-job handler in
   * `moduleFunctionHandlers` AT CALL TIME — never the shared `globalThis[name]`.
   *
   * Why this is parallel-safe where globalThis is not: the Proxy closes over this
   * OAIYRuntime instance, and each job has its OWN OAIYRuntime with its OWN
   * `moduleFunctionHandlers` (populated by per-job `createMethods(ctx)`). So
   * `jobA.ctx.modules.Browser.x` always dispatches into job A's handlers — there is
   * no shared mutable cell, and an interleaved job B cannot redirect it. This is the
   * same per-job dispatch the compiled-workflow shim already uses (getModulesShim →
   * host.call → moduleFunctionHandlers).
   *
   * Resolution is at CALL TIME (not a snapshot), so a module registered AFTER a
   * sibling's ctx was built still resolves. The inner method-proxies are cached so
   * a given method's function reference is stable across reads (identity-safe).
   */
  private buildModulesProxy(): Record<string, Record<string, (...args: any[]) => any>> {
    if (!this.modulesProxy) {
      this.modulesProxy = new Proxy(
        {} as Record<string, Record<string, (...args: any[]) => any>>,
        {
          get: (_target, modName) => {
            if (typeof modName !== 'string') return undefined;
            let methodProxy = this.moduleMethodProxies.get(modName);
            if (!methodProxy) {
              methodProxy = new Proxy({} as Record<string, (...args: any[]) => any>, {
                get: (_t, methodName) => {
                  if (typeof methodName !== 'string') return undefined;
                  const handler = this.moduleFunctionHandlers[`${modName}.${methodName}`];
                  if (handler) return handler;
                  // Unknown method: return a thunk that throws a CLEAR error when called
                  // (vs a cryptic "undefined is not a function"). Feature-detection should
                  // use `'method' in ctx.modules.X` (the has-trap below), not typeof.
                  return (..._args: unknown[]) => {
                    throw new Error(`ctx.modules: module method not registered: ${modName}.${String(methodName)}`);
                  };
                },
                has: (_t, methodName) =>
                  typeof methodName === 'string' &&
                  `${modName}.${methodName}` in this.moduleFunctionHandlers,
                ownKeys: () =>
                  Object.keys(this.moduleFunctionHandlers)
                    .filter((k) => k.startsWith(`${modName}.`))
                    .map((k) => k.slice(modName.length + 1)),
                getOwnPropertyDescriptor: () => ({ configurable: true, enumerable: true }),
              });
              this.moduleMethodProxies.set(modName, methodProxy);
            }
            return methodProxy;
          },
        },
      );
    }
    return this.modulesProxy;
  }

  private registerBuiltinModules(): void {
    const abortMod = createAbortModule({ abortSignal: this.abortSignal });
    const utilMod = createUtilityModule({ performHttpRequest: this.performHttpRequest.bind(this) });
    const agentMod = createAgentModule({
      agentMemory: this.agentMemory,
      loadPersistedMemory: this.loadPersistedMemory.bind(this),
      persistMemoryEntry: this.persistMemoryEntry.bind(this),
      deletePersistedMemoryEntry: this.deletePersistedMemoryEntry.bind(this),
      onLog: this.onLog,
    });

    this.registerModuleInternally('Abort', abortMod);
    this.registerModuleInternally('Utility', utilMod);
    this.registerModuleInternally('Agent', agentMod);
  }

  setModuleRegistry(registry: ModuleRegistry): void {
    this.moduleRegistry = registry;
  }

  setModuleSettings(settings: Record<string, Record<string, unknown>>): void {
    this.moduleSettings = settings;
  }

  setAvailableFlows(flows: Flow[]): void {
    this.availableFlows = flows;
  }

  setPackageMacros(macros: Flow[]): void {
    this.packageMacros = macros;
  }

  clearPackageMacros(): void {
    this.packageMacros = [];
  }

  private findFlowOrMacro(flowId: string): Flow | undefined {
    const packageMacro = this.packageMacros.find(f => f.id === flowId);
    if (packageMacro) return packageMacro;
    return this.availableFlows.find(f => f.id === flowId);
  }

  private async executeSubflow(flowId: string, inputs: Record<string, unknown>): Promise<unknown> {
    const flow = this.findFlowOrMacro(flowId);
    if (!flow) {
      throw new Error(`Subflow not found: ${flowId}`);
    }

    const compiler = new OAIYCompiler();
    compiler.setAvailableFlows(this.availableFlows);
    if (this.packageMacros.length > 0) {
      compiler.setPackageMacros(this.packageMacros);
    }
    if (this.moduleRegistry) {
      compiler.setModuleRegistry(this.moduleRegistry);
    }
    compiler.setProjectSettings(this.projectSettings);

    let script = compiler.compile(flow.graph, inputs as WorkflowInputs);
    // Resolve any deferred (off-realm) node compilations before executing.
    script = await compiler.resolveDeferred(script);

    try {
      const result = await this.executeScript(script);
      return extractDeepValue(result);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      throw new Error(`Subflow execution failed: ${error}`);
    }
  }

  async registerDynamicModule(loadedModule: LoadedModule): Promise<void> {
    const runtimeModule = loadedModule.runtime;
    if (!runtimeModule) return;

    const context: RuntimeContext = {
      log: (level, message) => {
        if (this.onLog) {
          const typeMap: Record<string, 'info' | 'error' | 'success' | 'node'> = {
            info: 'info', warn: 'info', error: 'error', success: 'success',
          };
          this.onLog({
            source: loadedModule.manifest.name,
            message: redactSensitive(message),
            type: typeMap[level] || 'info',
          });
        }
      },
      settings: this.moduleSettings[loadedModule.manifest.id] || {},
      getModuleSetting: (key) => {
        const moduleSettings = this.moduleSettings[loadedModule.manifest.id] || {};
        return moduleSettings[key];
      },
      // Resolve a project-constant NAME (e.g. an `apiKeyConstant` like
      // OPENAI_API_KEY) to its value. Undefined for unknown names — modules
      // fall back to their own heuristics (literal-key detection, etc.).
      // PERMISSION GATE: untrusted package code (currentPackageId set) must hold the right
      // permission — high-risk `secrets:read` for a SECRET constant (API key), `constants:read`
      // for a plain one — or this returns undefined. Without it a package granted only the
      // benign `network` permission could resolve ANY service's key via getConstant(thatName)
      // (e.g. an ai_llm node with apiKeyConstant pointed at another service + a custom endpoint)
      // and exfiltrate it. hasPackagePermission returns true for a trusted local flow (no
      // packageId), so the user's own flows are unaffected.
      getConstant: (name: string) => {
        const value = this.projectConstants[name];
        if (value === undefined) return undefined;
        const perm = this.secretConstantNames.has(name) ? 'secrets:read' : 'constants:read';
        if (!this.hasPackagePermission(perm)) return undefined;
        return value;
      },
      abortSignal: this.abortSignal || undefined,
      // ctx.fetch and ctx.secureFetch *both* go through the policy-enforced
      // gateway. Modules must never see the raw browser/Node `fetch` because
      // it bypasses the local-network permission flow, the SSRF guard and
      // (in the Rust backend) the DNS-pinning protections.
      //
      // Returns a Response-shaped wrapper so callers can keep using the
      // standard `.text()`, `.json()`, `.headers.forEach(...)` patterns.
      fetch: (url, options) =>
        this.policyFetch(url as string, options as RequestInit | undefined, {
          moduleId: loadedModule.manifest.id,
          purpose: 'module fetch',
        }),
      secureFetch: (url, options) =>
        this.policyFetch(url as string, options as RequestInit | undefined, {
          moduleId: loadedModule.manifest.id,
          nodeId: (options as { nodeId?: string } | undefined)?.nodeId,
          purpose: (options as { purpose?: string } | undefined)?.purpose ?? 'module fetch',
        }),
      onStreamToken: this.onToken ? (nodeId, token) => this.onToken?.(nodeId, token) : undefined,
      onImage: this.onImage ? (nodeId, imageUrl) => this.onImage?.(nodeId, imageUrl) : undefined,
      onNodeStatus: this.onNodeStatus ? (nodeId, status) => this.onNodeStatus?.(nodeId, status) : undefined,
      runSubflow: (flowId, inputs) => this.executeSubflow(flowId, inputs),
      useClaudeForAI: this.useClaudeForAI,
      currentJobId: this.currentJobId || undefined,
      yieldForAI: this.onYieldForAI || undefined,
      currentPackageId: this.currentPackageId ?? undefined,
      hasPermission: (perm: string) => this.hasPackagePermission(perm),
      formatPath: (p: string) =>
        formatPathForLog(p, (this.projectSettings.logPathPolicy ?? 'tilde') as LogPathPolicy),
      // Modules call into Tauri exclusively through this filtered broker.
      // Direct access to `window.__TAURI__` is no longer wired here; with
      // `withGlobalTauri: false` it isn't available at runtime anyway, but
      // exposing only an explicit invoke also makes auditing trivial —
      // search for `ctx.tauri.invoke` to enumerate every call.
      tauri: this.buildModuleInvokeBroker(loadedModule.manifest.id),
      // Per-job cross-module access. `ctx.modules.Browser.createSessionV2(...)` and
      // `ctx.callModule('Browser','createSessionV2',...)` both dispatch through THIS
      // job's handlers (never the shared globalThis[name]), so module-internal
      // cross-module calls are safe under parallel execution. See buildModulesProxy.
      modules: this.buildModulesProxy(),
      callModule: (modName: string, methodName: string, ...args: unknown[]) => {
        const handler = this.moduleFunctionHandlers[`${modName}.${methodName}`];
        if (!handler) {
          throw new Error(`callModule: module method not registered: ${modName}.${methodName}`);
        }
        return handler(...args);
      },
      // Per-job collection store for the Database module. Routes through the host
      // databaseHandler with THIS job's flowId/packageId, so concurrent jobs hit their
      // own per-flow database file (the handler falls back to the shared db only when
      // flowId is absent). Without this wiring ctx.database was undefined and every
      // Database node returned "Database not available".
      database: this.onDatabase
        ? {
            insertDocument: async (collection: string, data: Record<string, unknown>) => {
              const r = await this.onDatabase!({
                operation: 'insert',
                storageType: 'collection',
                collectionName: collection,
                data,
                flowId: this.currentFlowId ?? undefined,
                packageId: this.currentPackageId ?? undefined,
              });
              if (!r.success) throw new Error(r.error || 'Database insert failed');
              return String(r.insertedId ?? '');
            },
            findDocuments: async (collection: string, filter?: Record<string, unknown>) => {
              const r = await this.onDatabase!({
                operation: 'query',
                storageType: 'collection',
                collectionName: collection,
                filter,
                flowId: this.currentFlowId ?? undefined,
                packageId: this.currentPackageId ?? undefined,
              });
              if (!r.success) throw new Error(r.error || 'Database query failed');
              // The handler returns each row flattened as { id, _id, _created, ...fields }.
              // The ctx.database contract wants `data` to be just the document fields, so
              // strip the id/timestamp metadata back out (callers re-add id/_created).
              return (r.data ?? []).map((doc) => {
                const { id, _id, _created, created_at, ...fields } = doc as Record<string, unknown>;
                return {
                  id: String(id ?? _id ?? ''),
                  data: fields,
                  created_at: String(_created ?? created_at ?? ''),
                };
              });
            },
            updateDocument: async (id: string, data: Record<string, unknown>) => {
              const r = await this.onDatabase!({
                operation: 'update',
                storageType: 'collection',
                data,
                filter: { id },
                flowId: this.currentFlowId ?? undefined,
                packageId: this.currentPackageId ?? undefined,
              });
              if (!r.success) throw new Error(r.error || 'Database update failed');
              return (r.rowsAffected ?? 0) > 0;
            },
            deleteDocument: async (id: string) => {
              const r = await this.onDatabase!({
                operation: 'delete',
                storageType: 'collection',
                filter: { id },
                flowId: this.currentFlowId ?? undefined,
                packageId: this.currentPackageId ?? undefined,
              });
              if (!r.success) throw new Error(r.error || 'Database delete failed');
              return (r.rowsAffected ?? 0) > 0;
            },
          }
        : undefined,
    };

    // Runtime-only plugins (e.g. plugin-llm, plugin-tts) ship a Tauri
    // command surface and a helper class but no `RuntimeModule` shape.
    // Their bundle ends up with the helper *class* as the `runtime`
    // default export, which lacks `.methods` / `.name` / `.init`.
    // Skip registration cleanly instead of throwing in
    // `registerModuleInternally(undefined.methods)` and stalling the
    // entire job processor.
    const hasRuntimeShape =
      typeof runtimeModule === 'object' &&
      runtimeModule !== null &&
      typeof (runtimeModule as { name?: unknown }).name === 'string' &&
      typeof (runtimeModule as { methods?: unknown }).methods === 'object';
    if (!hasRuntimeShape) {
      return;
    }

    // Per-job methods. Prefer the createMethods factory: it returns FRESH closures
    // bound to THIS job's `context`, so two jobs running concurrently never share a
    // module-level ctx singleton (the bug that forced maxConcurrency=1). Fall back to
    // the legacy init + shared methods for any module that hasn't migrated.
    let methods: Record<string, RuntimeMethod>;
    let perJobCleanup: (() => Promise<void>) | undefined;
    if (runtimeModule.createMethods) {
      methods = runtimeModule.createMethods(context);
      // A migrated module may attach a per-job cleanup under the MODULE_CLEANUP
      // symbol (Object.entries skips it, so it's never registered as a node method).
      perJobCleanup = (methods as Record<symbol, unknown>)[MODULE_CLEANUP] as
        | (() => Promise<void>)
        | undefined;
    } else {
      if (runtimeModule.init) {
        await runtimeModule.init(context);
      }
      methods = runtimeModule.methods;
    }

    this.registerModuleInternally(runtimeModule.name, methods as any);
    // Store a per-job entry so cleanupDynamicModules runs a cleanup bound to THIS
    // job's state (createMethods modules close over per-job state the singleton
    // cleanup can't see); fall back to the module's own cleanup otherwise.
    this.loadedRuntimeModules.set(
      loadedModule.manifest.id,
      perJobCleanup ? { ...runtimeModule, cleanup: perJobCleanup } : runtimeModule,
    );

    if (this.onLog) {
      this.onLog({ source: 'ModuleLoader', message: `Registered dynamic module: ${runtimeModule.name}`, type: 'info' });
    }
  }

  async cleanupDynamicModules(): Promise<void> {
    // Tear down in REVERSE registration order, so a module registered later (which may
    // depend on an earlier one) cleans up before its dependency.
    const modules = Array.from(this.loadedRuntimeModules.values()).reverse();
    for (const runtimeModule of modules) {
      if (runtimeModule.cleanup) {
        try {
          await runtimeModule.cleanup();
        } catch (error) {
          runtimeLogger.error(`Error cleaning up module ${runtimeModule.name}: ${error}`);
        }
      }
    }
    this.loadedRuntimeModules.clear();
  }

  private log(type: 'info' | 'error' | 'success', message: string) {
    if (this.onLog) {
      // Run every workflow-side log line through the central redaction
      // layer so API keys/tokens emitted by `console.log(apiKey)` from a
      // logic block don't end up in the UI log panel verbatim.
      this.onLog({ source: 'System', message: redactSensitive(message), type });
    }
  }

  private getModulesShim(): string {
    let shimCode = `

`;

    const allModules = ['Abort', 'Utility', 'Agent', ...Array.from(this.loadedRuntimeModules.values()).map(m => m.name)];

    for (const modName of new Set(allModules)) {
      shimCode += `let ${modName} = {};\n`;

      const methods = Object.keys(this.moduleFunctionHandlers).filter(k => k.startsWith(`${modName}.`)).map(k => k.split('.')[1]);

      for (const methodName of methods) {
        shimCode += `${modName}["${methodName}"] = function(...args) {\n`;
        shimCode += `  let jsonArgs = JSON.stringify(args);\n`;
        shimCode += `  return { _kind: "${modName}.${methodName}", _args: [jsonArgs] };\n`;
        shimCode += `};\n`;
      }
    }

    return shimCode;
  }

  private async executeScript(script: string): Promise<unknown> {
    return await new Promise<unknown>((resolve, reject) => {
      let isDone = false;
      let abortListener: (() => void) | null = null;

      // The abort signal is shared across the runtime and may have many
      // long-running scripts attached. We always remove our listener in
      // `finalize` so we don't leak references after the script completes.
      const detachAbort = () => {
        if (abortListener && this.abortSignal) {
          this.abortSignal.removeEventListener('abort', abortListener);
          abortListener = null;
        }
      };

      const finish = (value: unknown) => {
        if (isDone) return;
        isDone = true;
        detachAbort();
        resolve(value);
      };
      const fail = (err: unknown) => {
        if (isDone) return;
        isDone = true;
        detachAbort();
        reject(err instanceof Error ? err : new Error(String(err)));
      };

      const consoleProxy = {
        log: (...args: unknown[]) => this.log('info', `[VM] ${formatLogArgs(args)}`),
        warn: (...args: unknown[]) => this.log('info', `[VM] WARN: ${formatLogArgs(args)}`),
        error: (...args: unknown[]) => this.log('error', `[VM] ${formatLogArgs(args)}`),
        debug: () => {},
        info: (...args: unknown[]) => this.log('info', `[VM] ${formatLogArgs(args)}`),
      };

      const host = {
        call: (kind: string, args: unknown[], callback: (res: unknown) => void) => {
          if (isDone) return;

          if (this.abortSignal?.aborted) {
            fail(new AbortError());
            return;
          }

          if (kind === '__system.finish') {
            const raw = args[0];
            let value: unknown = null;
            try { value = typeof raw === 'string' && raw ? JSON.parse(raw) : raw; } catch { value = raw; }
            finish(value);
            return;
          }
          if (kind === '__system.finish_error') {
            fail(new Error(String(args[0])));
            return;
          }

          const handler = this.moduleFunctionHandlers[kind];
          if (!handler) {
            callback({ __error__: `Unknown method ${kind}` });
            return;
          }

          // Central permission gate. Two checks layer here:
          //
          //   1. **Fail-closed gate for packages.** When a package context
          //      is active, the kind MUST be classified — either in
          //      `REQUIRED_PERMISSIONS_BY_KIND` (needs a specific permission)
          //      or in `NO_PERMISSION_REQUIRED_KINDS` (safe by construction).
          //      Anything else is refused with a "method not permission-
          //      declared" error so a new module method shipping without
          //      a classification is loud, not silent.
          //
          //   2. **Permission check.** If the method needs a permission and
          //      the active package isn't granted it, refuse.
          //
          // Trusted local flows (no `currentPackageId`) skip both checks.
          const required = REQUIRED_PERMISSIONS_BY_KIND[kind];
          if (this.currentPackageId) {
            const isClassified =
              kind in REQUIRED_PERMISSIONS_BY_KIND ||
              NO_PERMISSION_REQUIRED_KINDS.has(kind);
            if (!isClassified) {
              callback({
                __error__:
                  `Refusing to call ${kind}: method is not permission-declared. ` +
                  `Add an entry to REQUIRED_PERMISSIONS_BY_KIND or NO_PERMISSION_REQUIRED_KINDS in runtime.ts.`,
              });
              return;
            }
          }
          if (required && !this.hasPackagePermission(required)) {
            callback({
              __error__:
                `Refusing to call ${kind}: the active package does not have ` +
                `the '${required}' permission.`,
            });
            return;
          }

          const argsStr = args[0];
          let realArgs: unknown[] = [];
          try {
            realArgs = typeof argsStr === 'string' && argsStr ? JSON.parse(argsStr) : [];
          } catch (e) {
            callback({ __error__: `Failed to parse args for ${kind}: ${String(e)}` });
            return;
          }

          Promise.resolve()
            .then(() => handler(...realArgs))
            .then(res => { if (!isDone) callback(res); })
            .catch(err => { if (!isDone) callback({ __error__: String(err) }); });
        },
      };

      if (this.abortSignal) {
        if (this.abortSignal.aborted) {
          fail(new AbortError());
          return;
        }
        abortListener = () => fail(new AbortError());
        // `once: true` guarantees the listener auto-removes after firing,
        // but we still call detachAbort() on the success path so we don't
        // pin a closure on the AbortSignal for the lifetime of the runtime.
        this.abortSignal.addEventListener('abort', abortListener, { once: true });
      }

      // ===================================================================
      // Hardened-mode runtime isolation
      // ===================================================================
      //
      // When running an *untrusted package* workflow we layer four defences
      // on top of the central permission broker:
      //
      //   1. Escape-pattern scanner — refuse to run scripts that name known
      //      ways back to the global object (globalThis, (function(){return
      //      this})(), constructor.constructor, import.meta/import(), etc.).
      //   2. Strict-mode preamble — promotes silent footguns (e.g.
      //      assigning to undeclared identifiers) to errors so a script
      //      can't just `delete fetch` to undo a shadow.
      //   3. Global shadowing — declare a wide list of dangerous globals as
      //      local `undefined` at function scope, so plain references
      //      resolve to the shadow rather than the real global.
      //   4. `null`-prototype `this` — the wrapper is invoked with
      //      `Object.create(null)` so a stray `this` in user code points
      //      at a sealed empty object rather than the realm's global.
      //
      // **None of this is a true sandbox.** A determined adversary can
      // still construct functions via the same realm's prototype chain.
      // For the untrusted/hardened path we therefore prefer the Web
      // Worker route (`runInWorker` below, implemented in
      // `untrusted-executor.ts`); the worker runs in its own JS realm so
      // a recovered `globalThis` only reaches the worker's pristine
      // globals, not the main thread's Tauri `invoke`, DOM, or our
      // private state. The hardened-shadow preamble above is still
      // applied so the in-thread fallback (Node test envs without
      // `Worker`, or callers that opted out via
      // `useWorkerForUntrusted: false`) gets the same belt-and-braces
      // protection.
      const isHardened = !!this.currentPackageId;
      if (isHardened) {
        const escape = detectSandboxEscape(script);
        if (escape) {
          fail(
            new Error(
              `Refusing to run script: matched sandbox-escape pattern '${escape}'. ` +
                `Untrusted packages cannot reference the global object directly.`,
            ),
          );
          return;
        }
      }

      const hardenedShadows = isHardened ? HARDENED_SHADOW_PREAMBLE : '';
      const strictPreamble = isHardened ? `'use strict';\n` : '';

      const fullScript = `
${strictPreamble}${this.getModulesShim()}
${hardenedShadows}
function* __run_workflow() {
   let __res = null;
   try {
       ${rewriteAwaitToYield(script)}
       __res = workflow_context;
   } catch (e) {
       host.call("__system.finish_error", ["" + e], function(r){});
       return;
   }
   return __res;
}

let __gen = __run_workflow();
function __step(val) {
   try {
       let item = val === undefined ? __gen.next() : __gen.next(val);
       if (item.done) {
          host.call("__system.finish", [JSON.stringify(item.value)], function(){});
          return;
       }
       if (item.value && item.value._kind) {
          host.call(item.value._kind, item.value._args, function(res) {
             if (res && typeof res === 'object' && res.__error__) {
                 try {
                     __gen.throw(new Error(res.__error__));
                 } catch(e) {
                     host.call("__system.finish_error", ["" + e], function(){});
                 }
             } else {
                 __step(res);
             }
          });
       } else {
          __step();
       }
   } catch(e) {
       host.call("__system.finish_error", ["" + e], function(){});
   }
}
__step();
`;

      // ----- Worker path (mandatory for hardened mode when available) -----
      //
      // For untrusted package workflows we run the compiled script in a
      // Web Worker so a recovered `globalThis` only points at the
      // worker's pristine realm — it cannot reach the main thread's
      // Tauri `invoke`, DOM, file handles, or our private state. Every
      // `host.call` from the worker is round-tripped over `postMessage`
      // and serviced by the same broker the in-thread path uses, so
      // capability gating, permission checks, and aborts behave
      // identically.
      //
      // We fall back to the in-thread path when:
      //   * the workflow is not hardened (trusted local flow, no package
      //     context — there's nothing to isolate from);
      //   * the runtime opted out via `useWorkerForUntrusted: false`
      //     (mostly unit tests in Node);
      //   * the Worker constructor isn't available (Node < 22 without
      //     `--experimental-worker`, jsdom test envs without globals).
      const workerAvailable =
        typeof Worker !== 'undefined' &&
        typeof Blob !== 'undefined' &&
        typeof URL !== 'undefined' &&
        typeof URL.createObjectURL === 'function';
      const canUseWorker = isHardened && this.useWorkerForUntrusted && workerAvailable;

      if (canUseWorker) {
        try {
          this.runInWorker(fullScript, host, consoleProxy, fail);
        } catch (e) {
          // Anything that breaks during worker spin-up should fail the
          // execution; we deliberately do *not* fall back to in-thread
          // for hardened code, because that would silently weaken the
          // sandbox the user is relying on.
          fail(e);
        }
        return;
      }

      // FAIL CLOSED: a hardened (untrusted-package) flow that WANTED worker isolation but the
      // Worker constructor is unavailable here must NOT silently downgrade to in-thread
      // execution in the host realm — on a Node host a recovered realm global yields
      // process/require → RCE. This mirrors the spin-up-failure handling above (which already
      // fails rather than downgrading). It never fires in the current wiring (browser/Tauri
      // always have Worker; the Node CLI never sets a packageId) — it guards a future
      // headless-package path. The explicit opt-out useWorkerForUntrusted:false (Node unit
      // tests) still reaches the in-thread path below by design.
      if (isHardened && this.useWorkerForUntrusted && !workerAvailable) {
        fail(
          new Error(
            'Refusing to run untrusted package code in-thread: Web Worker isolation is unavailable in this host',
          ),
        );
        return;
      }

      try {
        const fn = new Function('host', 'console', fullScript);
        // Hardened mode (in-thread fallback): invoke with a null-prototype
        // `this` so a bare `this` in user code does not resolve to the
        // realm global.
        if (isHardened) {
          fn.call(Object.create(null), host, consoleProxy);
        } else {
          fn(host, consoleProxy);
        }
      } catch (e) {
        fail(e);
      }
    });
  }

  /**
   * Spin up a Worker, post the compiled script, and bridge `host_call`
   * messages from the worker into the same `host.call` broker the
   * in-thread path uses. Console messages are forwarded to `consoleProxy`
   * so logs appear identically.
   *
   * The worker is terminated on:
   *   * a `finish` / `finish_error` message,
   *   * an `abort` on the runtime's `abortSignal`,
   *   * any `error` on the worker (we rethrow as a workflow failure).
   */
  private runInWorker(
    fullScript: string,
    host: { call: (kind: string, args: unknown[], cb: (res: unknown) => void) => void },
    consoleProxy: { log: (...a: unknown[]) => void; warn: (...a: unknown[]) => void; error: (...a: unknown[]) => void; debug: (...a: unknown[]) => void; info: (...a: unknown[]) => void },
    fail: (err: unknown) => void,
  ): void {
    const { worker, cleanup } = spawnUntrustedWorker();

    let terminated = false;
    const tearDown = () => {
      if (terminated) return;
      terminated = true;
      // Detach the abort listener on every teardown path (success / worker error / abort) — with
      // `once:true` it only auto-removes when it FIRES, so on the normal finish path it would
      // otherwise linger on a shared abortSignal (executeSubflow reuses the parent's signal across
      // iterations → unbounded listener growth). Mirrors the in-thread detachAbort.
      this.abortSignal?.removeEventListener('abort', onAbort);
      try { cleanup(); } catch { /* swallow */ }
    };

    // Propagate aborts.
    const onAbort = () => {
      tearDown();
      fail(new AbortError());
    };
    if (this.abortSignal && !this.abortSignal.aborted) {
      this.abortSignal.addEventListener('abort', onAbort, { once: true });
    }

    worker.addEventListener('message', (ev: MessageEvent) => {
      if (terminated) return;
      const msg = ev.data;
      if (!msg || typeof msg !== 'object') return;

      switch (msg.type) {
        case 'finish':
        case 'finish_error': {
          // Both are routed back through the host broker so the existing
          // finish/fail bookkeeping (memory, callbacks, etc.) fires once.
          if (msg.type === 'finish') {
            host.call('__system.finish', [JSON.stringify(msg.value)], () => {});
          } else {
            host.call('__system.finish_error', [String(msg.message)], () => {});
          }
          tearDown();
          return;
        }
        case 'console': {
          const level = String(msg.level || 'log');
          const args = Array.isArray(msg.args) ? msg.args : [];
          const fn = (consoleProxy as Record<string, (...a: unknown[]) => void>)[level];
          if (typeof fn === 'function') {
            fn(...args);
          } else {
            consoleProxy.log(...args);
          }
          return;
        }
        case 'host_call': {
          const id = msg.id;
          host.call(String(msg.kind || ''), Array.isArray(msg.args) ? msg.args : [], (result: unknown) => {
            if (terminated) return;
            try {
              worker.postMessage({ type: 'host_result', id, result });
            } catch (e) {
              // Posting can fail with non-cloneable values; surface as a
              // host_result with an error string rather than silently
              // hanging the worker's pending callback.
              try {
                worker.postMessage({
                  type: 'host_result',
                  id,
                  result: { __error__: `Worker bridge could not serialize result: ${String(e)}` },
                });
              } catch { /* nothing more we can do */ }
            }
          });
          return;
        }
      }
    });

    worker.addEventListener('error', (ev: ErrorEvent) => {
      if (terminated) return;
      tearDown();
      fail(new Error(`Untrusted-worker error: ${ev.message || String(ev)}`));
    });
    worker.addEventListener('messageerror', () => {
      if (terminated) return;
      tearDown();
      fail(new Error('Untrusted-worker message-deserialization error'));
    });

    // Kick off execution.
    worker.postMessage({ type: 'init', script: fullScript });
  }

  async runWorkflow(graph: WorkflowGraph, availableFlows?: Flow[], inputs?: WorkflowInputs): Promise<any> {
    if (availableFlows) {
      this.availableFlows = availableFlows;
    }

    const compiler = new OAIYCompiler();
    if (availableFlows) compiler.setAvailableFlows(availableFlows);
    if (this.packageMacros.length > 0) compiler.setPackageMacros(this.packageMacros);
    if (this.moduleRegistry) compiler.setModuleRegistry(this.moduleRegistry);
    compiler.setProjectSettings(this.projectSettings);

    let script = compiler.compile(graph, inputs);
    // Resolve any deferred (off-realm) node compilations before executing.
    script = await compiler.resolveDeferred(script);

    // Optional debug-only dump of the compiled script for inspection.
    // Disabled by default and never written from production builds.
    // Enable by setting OAIY_DEBUG_DUMP_SCRIPT=1 in a dev environment.
    if (shouldDumpCompiledScript()) {
      try {
        const fs = await import('fs');
        const path = await import('path');
        const dir = path.join(process.env.APPDATA || process.env.HOME || '', 'oaiy', 'debug');
        await fs.promises.mkdir(dir, { recursive: true });
        const debugPath = path.join(dir, 'debug_compiled_script.js');
        const redacted = redactCompiledScript(script);
        await fs.promises.writeFile(debugPath, redacted, { encoding: 'utf8', mode: 0o600 });
        this.log('info', `[DEBUG] Compiled script saved to: ${debugPath} (${redacted.length} chars, redacted)`);
      } catch {
        // Best-effort; never fail the workflow because of a debug dump.
      }
    }

    this.log('info', '--- Starting Workflow Execution ---');

    try {
      const result = await this.executeScript(script);
      this.log('success', '--- Workflow Completed Successfully ---');
      return result;
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      const isAbort = error.includes('__ABORT__') || error.includes('aborted');
      if (!isAbort) {
        this.log('error', `Workflow Error: ${error}`);
      }
      throw e;
    }
  }

  clearMemory() {
    this.agentMemory.clear();
    this.log('info', '[Memory] Cleared');
  }

  getMemorySnapshot(): Record<string, string | number | boolean | object> {
    return Object.fromEntries(this.agentMemory);
  }

  convertResultToJs(result: unknown): unknown {
    return extractDeepValue(result);
  }
}

export function createRuntime(config: RuntimeConfig): OAIYRuntime;
export function createRuntime(
  onToken?: StreamCallback,
  onLog?: LogCallback,
  onImage?: ImageCallback,
  onNodeStatus?: NodeStatusCallback,
  abortSignal?: AbortSignal,
  onDatabase?: DatabaseCallback,
  onLocalNetworkPermission?: LocalNetworkPermissionCallback
): OAIYRuntime;
export function createRuntime(
  configOrOnToken?: RuntimeConfig | StreamCallback,
  onLog?: LogCallback,
  onImage?: ImageCallback,
  onNodeStatus?: NodeStatusCallback,
  abortSignal?: AbortSignal,
  onDatabase?: DatabaseCallback,
  onLocalNetworkPermission?: LocalNetworkPermissionCallback
): OAIYRuntime {
  if (configOrOnToken && typeof configOrOnToken === 'object' && !('call' in configOrOnToken)) {
    return new OAIYRuntime(configOrOnToken as RuntimeConfig);
  }
  return new OAIYRuntime(
    configOrOnToken as StreamCallback,
    onLog,
    onImage,
    onNodeStatus,
    abortSignal,
    onDatabase,
    onLocalNetworkPermission
  );
}
