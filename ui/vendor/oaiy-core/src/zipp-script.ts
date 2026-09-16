/**
 * The leaf-script envelope: running ONE script — a logic_block body, a
 * condition expression, an app-logic entry function, a Python project — on the
 * Zipp engine without a workflow graph around it.
 *
 * `zipp-executor.ts` drives a compiled OAIY workflow: a generator trampoline,
 * a pump loop, host calls. A leaf script has none of that. It is source plus
 * data in, one value or one error out, and it is what `oaiy script`, the
 * Desktop's warm script host and any product that embeds the CLI speak. The
 * shapes here are the wire contract; `protocol/v1/script-*.schema.json` is the
 * same contract as data, for hosts that validate before they send.
 *
 * Engine-neutral by construction: OAIY names no product's files, entry points
 * or wrappers. A requester that wants a Python contract supplies every file,
 * the entry module and the function to call; a requester that wants a JS
 * prelude supplies it as a `profile` preamble with its digest. What OAIY owns
 * is the boundary — how data crosses, what runs where, and what each failure
 * is called.
 *
 * # How data crosses
 *
 * User data never becomes program text. `globals`, `args` and `source` reach
 * the guest only as `JSON.stringify` literals, parsed or compiled INSIDE the
 * engine, so a value can never close a string and continue as code. Globals
 * are installed by a bootstrap that keeps only own keys that are identifiers,
 * do not start with `__` and are not `__proto__`/`constructor`/`prototype`.
 * The result comes back through `evalInContext`'s JSON projection and is then
 * sanitised (depth 8, the same three keys dropped, non-JSON → null), so what a
 * host receives is inert data whatever the guest built.
 *
 * # Fail closed
 *
 * A request with ANY malformed job is refused whole, before an Engine exists,
 * with one error naming the job. Every job that does run gets a fresh Engine,
 * disposed in `finally`, so nothing a job did — a global it set, a budget it
 * spent, a ceiling it hit — is visible to the next. `lastErrorKind` is read
 * only inside `catch` (it is stale after a success). A WebAssembly trap is
 * terminal for the engine; it is reported and the host is told through
 * `onEngineTrap` so it can retire the instance. Python needs a python-capable
 * engine: on any other, a Python job is `unsupported` and never attempted.
 */
import {
  ZIPP_GUEST_SHIMS,
  ZIPP_MAX_INSTRUCTION_BUDGET_STEPS,
  ZIPP_PREAMBLE_GLOBALS,
} from './zipp-executor.ts';

export const SCRIPT_PROTOCOL_VERSION = 1;

export const SCRIPT_JS_MODES = ['program', 'body', 'auto', 'entry', 'parse'] as const;
export type ScriptJsMode = (typeof SCRIPT_JS_MODES)[number];
export type ScriptMode = ScriptJsMode | 'python-project';
export type ScriptLanguage = 'javascript' | 'python';

/**
 * Every way a job can fail, on the wire.
 *
 *   * `guest`       the script threw, or would not parse (its own fault)
 *   * `resource`    an engine ceiling: instruction budget, heap, dynamic-code
 *                   source size; also a WebAssembly trap
 *   * `timeout`     a wall-clock deadline (set by a host watchdog, never here)
 *   * `host`        the host or engine misbehaved: no reply, a conversion or
 *                   usage error, a budget the engine would not take
 *   * `prepare`     the `prepare` hook is missing, not a function, or threw
 *   * `source`      the program failed to compile or its top level threw
 *                   before any call (Python init; a JS profile preamble)
 *   * `unsupported` a language or mode this engine does not run
 */
export const SCRIPT_ERROR_KINDS = ['guest', 'resource', 'timeout', 'host', 'prepare', 'source', 'unsupported'] as const;
export type ScriptErrorKind = (typeof SCRIPT_ERROR_KINDS)[number];

export interface ScriptProfile {
  v: 1;
  /** Program text emitted before every JS job's source, after the guest shims. */
  preamble: string;
  /** Lower-case hex sha256 of `preamble`, verified before anything runs. */
  preambleSha256: string;
  instructionSteps?: number;
  /** Per-lane `prepare` hook names; read by the requester, not by the runner. */
  hooks?: Record<string, { prepare: string }>;
  /** A Python contract the requester will unfold into jobs. Opaque here. */
  python?: { contract: string; files: Record<string, string>; entry: string; call: string };
}

interface ScriptJobBase {
  id: string;
  /** Wall-clock hint for a host watchdog, 1..60000. Not enforced by the runner. */
  budgetMs?: number;
  /** Instruction budget for this job, 1..ZIPP_MAX_INSTRUCTION_BUDGET_STEPS. Unset: the engine default. */
  instructionSteps?: number;
}

export interface ScriptJsJob extends ScriptJobBase {
  language?: 'javascript';
  /**
   *   * `program`  `(0, eval)(source)`: completion value; a top-level `return` is a SyntaxError
   *   * `body`     `new Function(source)()`: a function body, `return` gives the value
   *   * `auto`     decided by PARSING: a script takes the `program` path, else a function body
   *   * `entry`    `source` declares `entry`; it is called with `args`
   *   * `parse`    `new Function("return (" + source + ")")`, never invoked
   */
  mode: ScriptJsMode;
  source: string;
  /** Installed as guest globals (filtered), for every mode. */
  globals?: Record<string, unknown>;
  /** `entry` mode only: the function `source` declares. */
  entry?: string;
  /** `entry` mode only. */
  args?: unknown[];
  /** A preamble function that maps the parsed globals (`program`/`body`/`auto`) or the args array (`entry`) before the source runs. */
  prepare?: string;
}

export interface ScriptPythonJob extends ScriptJobBase {
  language: 'python';
  mode: 'python-project';
  /** Module name (or `name.py`) → source. The requester supplies every file. */
  files: Record<string, string>;
  /** The module whose top level runs. */
  entry: string;
  /** A top-level function of `entry`, called with `args`. */
  call: string;
  args?: unknown[];
  /** A second file set tried once when the first fails to compile or its top level raises. */
  fallbackOnSourceError?: { files: Record<string, string> };
}

export type ScriptJob = ScriptJsJob | ScriptPythonJob;

export interface ScriptRequest {
  v: 1;
  profile?: ScriptProfile;
  jobs: ScriptJob[];
}

/** PR2a's `ZippEngineIdentity` minus the bundle fields — what a requester needs to trust a result. */
export interface ScriptEngineIdentity {
  name: 'zipp';
  release: string;
  version: string;
  revision: string;
  wasmSha256: string;
  languages: string[];
}

export type ScriptJobResult =
  | { id: string; ok: true; value?: unknown }
  | { id: string; ok: false; errorKind: ScriptErrorKind; error: string };

export interface ScriptResponse {
  v: 1;
  engine: ScriptEngineIdentity;
  results: ScriptJobResult[];
}

export interface ScriptRefusal {
  v: 1;
  error: { code: 'invalid_request'; message: string };
}

export type ScriptValidation =
  | { ok: true; request: ScriptRequest }
  | { ok: false; error: ScriptRefusal['error'] };

/** The subset of Zipp's `Engine` a leaf job drives. Python methods are absent on a JS-only build. */
export interface ScriptEngine {
  initScript(source: string): unknown;
  evalInContext(expr: string): unknown;
  lastErrorKind(): string;
  setInstructionBudget?(steps: number): boolean;
  renewInstructionBudget?(): boolean;
  dispose(): void;
  initPythonProject?(files: Record<string, string>, entry: string, argv: string[]): unknown;
  pythonCall?(name: string, args: unknown[]): unknown;
}

export type ScriptEngineCtor = new () => ScriptEngine;

export interface ScriptRunOptions {
  profile?: ScriptProfile;
  /** The engine's `zippProfile().languages`; decides `unsupported` before any Engine is built. */
  languages: readonly string[];
  /** A WebAssembly trap poisoned the instance: retire the Worker/instance, not just the Engine. */
  onEngineTrap?: (error: unknown) => void;
}

export interface ScriptHostOptions {
  engine: ScriptEngineIdentity;
  /** Verifies `profile.preambleSha256`. Without it a request carrying a profile is refused. */
  sha256?: (text: string) => string;
  onEngineTrap?: (error: unknown) => void;
}

/** Output depth kept; deeper levels become null. */
export const SCRIPT_MAX_OUTPUT_DEPTH = 8;

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const PYTHON_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SHA256_HEX = /^[0-9a-f]{64}$/;
const DANGEROUS_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
/** ES reserved words: an `entry`/`prepare` of this shape would be a SyntaxError in the program. */
const RESERVED_WORDS = new Set([
  'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else',
  'enum', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'let',
  'new', 'null', 'return', 'static', 'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void',
  'while', 'with', 'yield',
]);
/** Names the envelope's own program binds; a preamble redeclaring one would break the reply channel. */
export const SCRIPT_ENVELOPE_GLOBALS: readonly string[] = ['__replies', '__emit', '__out', '__ctx', '__args', '__k', '__fn', '__asBody'];
const MAX_BUDGET_MS = 60_000;

// ---------------------------------------------------------------------------
// Validation: the whole request, before any Engine exists.
// ---------------------------------------------------------------------------

class Invalid extends Error {}

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

function noExtraKeys(obj: Record<string, unknown>, allowed: readonly string[], where: string): void {
  for (const key of Object.keys(obj)) {
    if (!allowed.includes(key)) throw new Invalid(`${where}: unknown field ${JSON.stringify(key)}`);
  }
}

function requireString(obj: Record<string, unknown>, key: string, where: string, max = Infinity, min = 0): string {
  const v = obj[key];
  if (typeof v !== 'string') throw new Invalid(`${where}: ${key} must be a string`);
  if (v.length < min) throw new Invalid(`${where}: ${key} must not be empty`);
  if (v.length > max) throw new Invalid(`${where}: ${key} is longer than ${max} characters`);
  return v;
}

function optionalInt(obj: Record<string, unknown>, key: string, where: string, min: number, max: number): number | undefined {
  const v = obj[key];
  if (v === undefined) return undefined;
  if (!Number.isInteger(v) || (v as number) < min || (v as number) > max) {
    throw new Invalid(`${where}: ${key} must be an integer from ${min} to ${max}, not ${String(v)}`);
  }
  return v as number;
}

function requireIdentifier(obj: Record<string, unknown>, key: string, where: string, re = IDENTIFIER): string {
  const v = requireString(obj, key, where, 128);
  if (!re.test(v) || (re === IDENTIFIER && RESERVED_WORDS.has(v))) {
    throw new Invalid(`${where}: ${key} must be an identifier, not ${JSON.stringify(v)}`);
  }
  return v;
}

function requireJson(value: unknown, what: string): void {
  try {
    JSON.stringify(value);
  } catch (e) {
    throw new Invalid(`${what} is not JSON: ${errText(e)}`);
  }
}

function requireFiles(v: unknown, where: string): Record<string, string> {
  if (!isPlainObject(v) || Object.keys(v).length === 0) throw new Invalid(`${where}: files must be a non-empty object of name → source`);
  for (const [name, text] of Object.entries(v)) {
    if (name.length === 0 || name.length > 128) throw new Invalid(`${where}: file name ${JSON.stringify(name)} must be 1 to 128 characters`);
    if (typeof text !== 'string') throw new Invalid(`${where}: files[${JSON.stringify(name)}] must be a string`);
  }
  return v as Record<string, string>;
}

function validateProfile(raw: unknown, sha256: ScriptHostOptions['sha256']): ScriptProfile {
  const where = 'profile';
  if (!isPlainObject(raw)) throw new Invalid(`${where} must be an object`);
  noExtraKeys(raw, ['v', 'preamble', 'preambleSha256', 'instructionSteps', 'hooks', 'python'], where);
  if (raw.v !== 1) throw new Invalid(`${where}: v must be 1`);
  const preamble = requireString(raw, 'preamble', where);
  const digest = requireString(raw, 'preambleSha256', where);
  if (!SHA256_HEX.test(digest)) throw new Invalid(`${where}: preambleSha256 must be 64 lower-case hex characters`);
  if (!sha256) throw new Invalid(`${where}: this host cannot verify a preamble (no sha256)`);
  if (sha256(preamble) !== digest) throw new Invalid(`${where}: preambleSha256 does not match the preamble`);
  const collision = detectPreambleCollision(preamble);
  if (collision) throw new Invalid(`${where}: the preamble declares ${JSON.stringify(collision)}, a name the engine or the envelope binds`);
  const profile: ScriptProfile = { v: 1, preamble, preambleSha256: digest };
  const steps = optionalInt(raw, 'instructionSteps', where, 1, ZIPP_MAX_INSTRUCTION_BUDGET_STEPS);
  if (steps !== undefined) profile.instructionSteps = steps;
  if (raw.hooks !== undefined) {
    if (!isPlainObject(raw.hooks)) throw new Invalid(`${where}: hooks must be an object`);
    const hooks: Record<string, { prepare: string }> = {};
    for (const [lane, hook] of Object.entries(raw.hooks)) {
      const at = `${where}.hooks[${JSON.stringify(lane)}]`;
      if (!isPlainObject(hook)) throw new Invalid(`${at} must be an object`);
      noExtraKeys(hook, ['prepare'], at);
      hooks[lane] = { prepare: requireIdentifier(hook, 'prepare', at) };
    }
    profile.hooks = hooks;
  }
  if (raw.python !== undefined) {
    const at = `${where}.python`;
    if (!isPlainObject(raw.python)) throw new Invalid(`${at} must be an object`);
    noExtraKeys(raw.python, ['contract', 'files', 'entry', 'call'], at);
    profile.python = {
      contract: requireString(raw.python, 'contract', at, 128, 1),
      files: requireFiles(raw.python.files, at),
      entry: requireString(raw.python, 'entry', at, 128, 1),
      call: requireString(raw.python, 'call', at, 128, 1),
    };
  }
  return profile;
}

/**
 * A top-level declaration in `preamble` of a name Zipp's preamble or this
 * envelope binds: `let`/`class` after the engine's `var` is a SyntaxError that
 * takes the whole program, and a redeclared `__emit` would forge replies.
 * A declaration scan, not a parse — a full scan belongs to the CLI, which has a
 * parser; this catches what a text scan can and refuses rather than guesses.
 */
export function detectPreambleCollision(preamble: string): string | null {
  const decl = /^[ \t]*(?:var|let|const|class|async[ \t]+function|function)\b[ \t*]*([A-Za-z_$][A-Za-z0-9_$]*)/gm;
  const bound = new Set([...ZIPP_PREAMBLE_GLOBALS, ...SCRIPT_ENVELOPE_GLOBALS]);
  for (const m of preamble.matchAll(decl)) {
    if (bound.has(m[1])) return m[1];
  }
  return null;
}

function validateJob(raw: unknown, index: number): ScriptJob {
  let where = `jobs[${index}]`;
  if (!isPlainObject(raw)) throw new Invalid(`${where} must be an object`);
  if (typeof raw.id === 'string') where += ` (id ${JSON.stringify(raw.id)})`;
  const id = requireString(raw, 'id', where, 128);
  if (id.length === 0) throw new Invalid(`${where}: id must not be empty`);
  const mode = raw.mode;
  const language = raw.language === undefined ? 'javascript' : raw.language;
  if (language !== 'javascript' && language !== 'python') throw new Invalid(`${where}: unknown language ${JSON.stringify(raw.language)}`);

  const base: ScriptJobBase = { id };
  const budgetMs = optionalInt(raw, 'budgetMs', where, 1, MAX_BUDGET_MS);
  if (budgetMs !== undefined) base.budgetMs = budgetMs;
  const steps = optionalInt(raw, 'instructionSteps', where, 1, ZIPP_MAX_INSTRUCTION_BUDGET_STEPS);
  if (steps !== undefined) base.instructionSteps = steps;

  if (mode === 'python-project') {
    if (language !== 'python') throw new Invalid(`${where}: mode "python-project" needs language "python"`);
    noExtraKeys(raw, ['id', 'language', 'mode', 'files', 'entry', 'call', 'args', 'fallbackOnSourceError', 'budgetMs', 'instructionSteps'], where);
    const job: ScriptPythonJob = {
      ...base,
      language: 'python',
      mode,
      files: requireFiles(raw.files, where),
      entry: requireIdentifier(raw, 'entry', where, PYTHON_IDENTIFIER),
      call: requireIdentifier(raw, 'call', where, PYTHON_IDENTIFIER),
    };
    if (raw.args !== undefined) {
      if (!Array.isArray(raw.args)) throw new Invalid(`${where}: args must be an array`);
      requireJson(raw.args, `${where}: args`);
      job.args = raw.args;
    }
    if (raw.fallbackOnSourceError !== undefined) {
      const at = `${where}.fallbackOnSourceError`;
      if (!isPlainObject(raw.fallbackOnSourceError)) throw new Invalid(`${at} must be an object`);
      noExtraKeys(raw.fallbackOnSourceError, ['files'], at);
      job.fallbackOnSourceError = { files: requireFiles(raw.fallbackOnSourceError.files, at) };
    }
    return job;
  }

  if (typeof mode !== 'string' || !(SCRIPT_JS_MODES as readonly string[]).includes(mode)) {
    throw new Invalid(`${where}: unknown mode ${JSON.stringify(mode)}`);
  }
  if (language !== 'javascript') throw new Invalid(`${where}: mode ${JSON.stringify(mode)} runs JavaScript, not ${JSON.stringify(language)}`);
  noExtraKeys(raw, ['id', 'language', 'mode', 'source', 'globals', 'entry', 'args', 'prepare', 'budgetMs', 'instructionSteps'], where);
  const job: ScriptJsJob = { ...base, mode: mode as ScriptJsMode, source: requireString(raw, 'source', where) };
  if (raw.language !== undefined) job.language = 'javascript';
  if (raw.globals !== undefined) {
    if (!isPlainObject(raw.globals)) throw new Invalid(`${where}: globals must be a plain object`);
    requireJson(raw.globals, `${where}: globals`);
    job.globals = raw.globals;
  }
  if (mode === 'entry') {
    job.entry = requireIdentifier(raw, 'entry', where);
    if (raw.args !== undefined) {
      if (!Array.isArray(raw.args)) throw new Invalid(`${where}: args must be an array`);
      requireJson(raw.args, `${where}: args`);
      job.args = raw.args;
    }
  } else {
    if (raw.entry !== undefined) throw new Invalid(`${where}: entry is for mode "entry" only`);
    if (raw.args !== undefined) throw new Invalid(`${where}: args is for mode "entry" only`);
  }
  if (raw.prepare !== undefined) {
    if (mode === 'parse') throw new Invalid(`${where}: prepare has no meaning for mode "parse"`);
    job.prepare = requireIdentifier(raw, 'prepare', where);
  }
  return job;
}

/**
 * Check a request structurally and semantically. Any fault refuses the WHOLE
 * request with one message naming the job (4-2): a host either runs every job
 * or none, and a caller reading `results` never has to wonder whether a
 * missing entry is a refusal or a crash.
 */
export function validateScriptRequest(input: unknown, opts: { sha256?: ScriptHostOptions['sha256'] } = {}): ScriptValidation {
  try {
    if (!isPlainObject(input)) throw new Invalid('request must be an object');
    noExtraKeys(input, ['v', 'profile', 'jobs'], 'request');
    if (input.v !== SCRIPT_PROTOCOL_VERSION) throw new Invalid(`request: v must be ${SCRIPT_PROTOCOL_VERSION}, not ${JSON.stringify(input.v)}`);
    if (!Array.isArray(input.jobs)) throw new Invalid('request: jobs must be an array');
    const request: ScriptRequest = { v: 1, jobs: input.jobs.map(validateJob) };
    if (input.profile !== undefined) request.profile = validateProfile(input.profile, opts.sha256);
    return { ok: true, request };
  } catch (e) {
    if (e instanceof Invalid) return { ok: false, error: { code: 'invalid_request', message: e.message } };
    throw e;
  }
}

// ---------------------------------------------------------------------------
// The JS program.
// ---------------------------------------------------------------------------

/** The reply channel, plus stubs so a guest cannot write anywhere the host reads. */
const EMIT_PREAMBLE = `var __replies = [];
function __emit(o) { __replies.push(o); }
globalThis.print = function () {};
globalThis.console = { log: function(){}, warn: function(){}, error: function(){}, info: function(){}, debug: function(){} };
`;

/** Installs `__ctx`'s safe own keys as globals. Same rule as the module comment states. */
const INSTALL_GLOBALS = `  for (var __k in __ctx) {
    if (Object.prototype.hasOwnProperty.call(__ctx, __k)
        && /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(__k)
        && __k !== "__proto__" && __k !== "constructor" && __k !== "prototype"
        && __k.indexOf("__") !== 0) {
      globalThis[__k] = __ctx[__k];
    }
  }
`;

function modeBody(job: ScriptJsJob): string {
  const src = JSON.stringify(job.source);
  switch (job.mode) {
    case 'program':
      return `  __out = {ok: true, value: (0, eval)(${src})};`;
    case 'body':
      return `  __out = {ok: true, value: new Function(${src})()};`;
    case 'auto':
      // The style is decided by PARSING, so exactly one path executes the
      // author's code: `throw 0` is the probe's first statement, so a script
      // that parses runs nothing and takes the eval path; source that only
      // parses as a function body runs as one. Source that parses as neither
      // gets the error that fits it — the body error when it mentions `return`.
      return `  var __asBody = (function (src) {
    try { eval("throw 0;\\n" + src); return null; } catch (e) { if (e === 0) return null; }
    try { return new Function(src); } catch (e) { if (/\\breturn\\b/.test(src)) throw e; return null; }
  })(${src});
  __out = {ok: true, value: __asBody ? __asBody() : (0, eval)(${src})};`;
    case 'entry':
      return `  var __fn = new Function(${src} + "\\nreturn typeof ${job.entry} === 'function' ? ${job.entry} : null;")();
  __out = {ok: true, value: (typeof __fn === 'function') ? __fn.apply(undefined, __args) : undefined};`;
    case 'parse':
      // Parsed standalone and never invoked: an unbalanced `}` is a
      // SyntaxError, not an escape, and a side-effecting expression stays inert.
      return `  new Function("return (" + ${src} + "\\n);");
  __out = {ok: true, value: null};`;
  }
}

/**
 * Assemble the program for one JS job: reply channel → guest shims (top
 * level, so a body compiled at run time meets the stubs, as PR1c placed them
 * for trusted flows) → profile preamble → data bootstrap → `prepare` → mode
 * body. Only `source`, `globals` and `args` are the requester's, and each is
 * a JSON literal.
 */
export function buildScriptProgram(job: ScriptJsJob, profile?: ScriptProfile): string {
  const globals = JSON.stringify(JSON.stringify(job.globals ?? {}));
  const args = JSON.stringify(JSON.stringify(job.args ?? []));
  const target = job.mode === 'entry' ? '__args' : '__ctx';
  const prepare = job.prepare
    ? `  if (typeof ${job.prepare} !== 'function') {
    __out = {ok: false, kind: 'prepare', error: ${JSON.stringify(`prepare hook '${job.prepare}' is not a function`)}};
  } else {
    try { ${target} = ${job.prepare}(${target}); } catch (e) { __out = {ok: false, kind: 'prepare', error: String((e && e.message) || e)}; }
  }
`
    : '';
  return `${EMIT_PREAMBLE}${ZIPP_GUEST_SHIMS}
${profile?.preamble ?? ''}
var __out;
try {
  var __ctx; try { __ctx = JSON.parse(${globals}); } catch (e) { __ctx = {}; }
  var __args; try { __args = JSON.parse(${args}); } catch (e) { __args = []; }
${prepare}  if (!__out) {
    if (__ctx === null || typeof __ctx !== 'object') __ctx = {};
    if (!Array.isArray(__args)) __args = [__args];
${INSTALL_GLOBALS}${modeBody(job)}
  }
} catch (e) { __out = {ok: false, error: String((e && e.message) || e)}; }
__emit(__out);
`;
}

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

/** Data coming OUT of the sandbox, made inert: depth capped, polluting keys dropped, non-JSON → null. */
export function sanitizeScriptValue(value: unknown, depth = 0): unknown {
  if (value === null || value === undefined) return value;
  const t = typeof value;
  if (t === 'number' || t === 'string' || t === 'boolean') return value;
  if (depth >= SCRIPT_MAX_OUTPUT_DEPTH) return null;
  if (Array.isArray(value)) return value.map((v) => sanitizeScriptValue(v, depth + 1));
  if (t === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (DANGEROUS_KEYS.has(k)) continue;
      out[k] = sanitizeScriptValue(v, depth + 1);
    }
    return out;
  }
  return null;
}

function errText(e: unknown): string {
  // The engine throws strings, not Errors; both shapes cross the boundary.
  if (typeof e === 'string') return e;
  if (e instanceof Error) return e.message || String(e);
  return String(e);
}

const isTrap = (e: unknown): boolean =>
  typeof WebAssembly !== 'undefined' && e instanceof WebAssembly.RuntimeError;

/** The engine's own classification of a throw, in the envelope's words. */
function classify(engine: ScriptEngine | undefined): ScriptErrorKind {
  let kind = '';
  try {
    if (engine) kind = engine.lastErrorKind();
  } catch {
    // nothing to classify with: a host error
  }
  if (kind === 'guest' || kind === 'resource' || kind === 'source') return kind;
  return 'host';
}

type Attempt = { ok: true; value: unknown } | { ok: false; kind: ScriptErrorKind; error: string; phase: 'init' | 'run' };

/**
 * One fresh Engine: budget, init, one entry, dispose. `run` does the phase's
 * work and returns the raw value; everything around it is the same for JS and
 * Python. The Engine is built inside the guard, so a trap while constructing
 * it is reported like one mid-run.
 */
function attempt(
  Engine: ScriptEngineCtor,
  steps: number | undefined,
  init: (engine: ScriptEngine) => void,
  run: (engine: ScriptEngine) => unknown,
  onEngineTrap?: (error: unknown) => void,
): Attempt {
  let engine: ScriptEngine | undefined;
  let trapped = false;
  let phase: 'init' | 'run' = 'init';
  try {
    engine = new Engine();
    // Sized BEFORE init so the allowance governs the top level too. An engine
    // that cannot take the budget, or refuses it, fails the job rather than
    // running on another (the ZippSession rule).
    if (steps !== undefined) {
      if (typeof engine.setInstructionBudget !== 'function') {
        return { ok: false, kind: 'host', error: 'This Zipp engine cannot take an instruction budget (no setInstructionBudget)', phase };
      }
      if (!engine.setInstructionBudget(steps)) {
        return { ok: false, kind: 'host', error: 'Zipp refused the instruction budget', phase };
      }
    }
    init(engine);
    phase = 'run';
    // The call is a re-entry: it gets the job's whole budget, not what init left.
    engine.renewInstructionBudget?.();
    return { ok: true, value: run(engine) };
  } catch (e) {
    if (isTrap(e)) {
      trapped = true;
      onEngineTrap?.(e);
      return { ok: false, kind: 'resource', error: `The Zipp instance trapped: ${errText(e)}`, phase };
    }
    // lastErrorKind describes the throw being handled; after a success it is stale.
    return { ok: false, kind: classify(engine), error: errText(e), phase };
  } finally {
    // A `source`/`resource` failure has already torn the engine down and a
    // trapped instance has nothing left to dispose; neither may fail the job.
    if (engine && !trapped) {
      try {
        engine.dispose();
      } catch {
        // already gone
      }
    }
  }
}

function runJs(Engine: ScriptEngineCtor, job: ScriptJsJob, steps: number | undefined, profile: ScriptProfile | undefined, onEngineTrap?: (e: unknown) => void): ScriptJobResult {
  const program = buildScriptProgram(job, profile);
  const out = attempt(
    Engine,
    steps,
    (engine) => { engine.initScript(program); },
    (engine) => engine.evalInContext('__replies.length ? __replies[__replies.length - 1] : null'),
    onEngineTrap,
  );
  if (!out.ok) return { id: job.id, ok: false, errorKind: out.kind, error: out.error };
  const reply = out.value;
  if (!reply || typeof reply !== 'object') {
    return { id: job.id, ok: false, errorKind: 'host', error: 'The script produced no result' };
  }
  const outcome = reply as { ok?: unknown; kind?: unknown; value?: unknown; error?: unknown };
  if (outcome.ok !== true) {
    return {
      id: job.id,
      ok: false,
      errorKind: outcome.kind === 'prepare' ? 'prepare' : 'guest',
      error: typeof outcome.error === 'string' && outcome.error ? outcome.error : 'evaluation failed',
    };
  }
  const result: ScriptJobResult = { id: job.id, ok: true };
  if (outcome.value !== undefined) result.value = sanitizeScriptValue(outcome.value);
  return result;
}

function runPython(Engine: ScriptEngineCtor, job: ScriptPythonJob, steps: number | undefined, onEngineTrap?: (e: unknown) => void): ScriptJobResult {
  // The JSON view, exactly as the JS path parses it in the guest: an undefined
  // member drops out and a Date becomes its string, so both languages see the
  // same values.
  const args = JSON.parse(JSON.stringify(job.args ?? [])) as unknown[];
  const tryFiles = (files: Record<string, string>): Attempt =>
    attempt(
      Engine,
      steps,
      (engine) => { engine.initPythonProject!(files, job.entry, []); },
      (engine) => engine.pythonCall!(job.call, args),
      onEngineTrap,
    );
  let out = tryFiles(job.files);
  // The engine reports a compile failure or a top-level raise as `source`
  // while the project initialises, having run none of the requester's call.
  // The fallback is tried ONCE, on a fresh Engine.
  if (!out.ok && out.phase === 'init' && out.kind === 'source' && job.fallbackOnSourceError) {
    out = tryFiles(job.fallbackOnSourceError.files);
  }
  if (!out.ok) return { id: job.id, ok: false, errorKind: out.kind, error: out.error };
  const result: ScriptJobResult = { id: job.id, ok: true };
  if (out.value !== undefined) result.value = sanitizeScriptValue(out.value);
  return result;
}

/** Whether the Engine class carries the Python frontend — read off the prototype, so no instance is built to find out. */
function hasPython(Engine: ScriptEngineCtor): boolean {
  const proto = Engine.prototype as Partial<ScriptEngine> | undefined;
  return typeof proto?.initPythonProject === 'function' && typeof proto?.pythonCall === 'function';
}

/**
 * Run one validated job on a fresh Engine. Never throws for anything the job
 * did: every failure is a result with an `errorKind`.
 */
export function runScriptJob(Engine: ScriptEngineCtor, job: ScriptJob, opts: ScriptRunOptions): ScriptJobResult {
  const steps = job.instructionSteps ?? opts.profile?.instructionSteps;
  if (job.mode === 'python-project') {
    // Decided from the profile AND the class, before any Engine exists: a
    // profile that names python on a build without the methods is a mismatch,
    // and the answer is the same — never attempted.
    if (!opts.languages.includes('python') || !hasPython(Engine)) {
      return { id: job.id, ok: false, errorKind: 'unsupported', error: 'This Zipp engine does not run Python' };
    }
    return runPython(Engine, job, steps, opts.onEngineTrap);
  }
  return runJs(Engine, job, steps, opts.profile, opts.onEngineTrap);
}

/**
 * Validate and run a whole request. A refusal runs nothing; otherwise every
 * job runs, in order, each on its own Engine, and `results` has one entry per
 * job in the same order.
 */
export function runScriptRequest(Engine: ScriptEngineCtor, input: unknown, opts: ScriptHostOptions): ScriptResponse | ScriptRefusal {
  const checked = validateScriptRequest(input, { sha256: opts.sha256 });
  if (!checked.ok) return { v: 1, error: checked.error };
  const { profile, jobs } = checked.request;
  // A trap poisons the WASM instance, not just the Engine: the rest of the
  // batch is not attempted on it. The host hears once and retires the instance.
  let trapped = false;
  const run: ScriptRunOptions = {
    profile,
    languages: opts.engine.languages,
    onEngineTrap: (e) => {
      trapped = true;
      opts.onEngineTrap?.(e);
    },
  };
  const results = jobs.map((job): ScriptJobResult => trapped
    ? { id: job.id, ok: false, errorKind: 'host', error: 'Not attempted: the Zipp instance trapped on an earlier job in this batch' }
    : runScriptJob(Engine, job, run));
  return { v: 1, engine: opts.engine, results };
}

/**
 * The identity a response carries, from the running engine's `zippProfile()`
 * plus what only the install record knows (the release tag and the .wasm
 * digest). Never a literal.
 */
export function scriptEngineIdentity(profileJson: string, source: { release: string; wasmSha256: string }): ScriptEngineIdentity {
  const profile = JSON.parse(profileJson) as { version?: unknown; source?: { sha?: unknown }; languages?: unknown };
  if (typeof profile.version !== 'string' || typeof profile.source?.sha !== 'string' || !Array.isArray(profile.languages)) {
    throw new Error('zippProfile() did not report version, source.sha and languages');
  }
  return {
    name: 'zipp',
    release: source.release,
    version: profile.version,
    revision: profile.source.sha,
    wasmSha256: source.wasmSha256,
    languages: profile.languages.filter((l): l is string => typeof l === 'string'),
  };
}
