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
 * # Modes
 *
 * A Python contract usually wraps the author's text differently depending on
 * what is being run, and then has to UNDO that wrapping to say which line the
 * author should look at. A profile may therefore carry `python.modes`: named
 * wrappings, each a few files, a block file, the text before and after the
 * author's source, and how many lines that puts in front of it. A job names
 * modes instead of carrying files, and the runner unfolds one, runs it, and
 * subtracts the wrapper's lines from every location the engine reports in the
 * block file before the result leaves.
 *
 * The runner has no modes of its own and no opinion about any of them. It
 * cannot: every byte it emits came from the profile the requester sent with
 * the request, a mode name is a lookup key and nothing else, and a job naming
 * a mode the profile does not define is a malformed request. Templating and
 * arithmetic are OAIY's (a wrapper's line count is arithmetic, and Zipp's
 * location grammar is OAIY's boundary with Zipp); what a wrapping is FOR
 * stays with the requester that wrote it.
 *
 * # How data crosses
 *
 * User data never becomes program text. `globals`, `args` and `source` reach
 * the guest only as `JSON.stringify` literals, parsed or compiled INSIDE the
 * engine, so a value can never close a string and continue as code. Globals
 * are installed by a bootstrap that keeps only own keys that are identifiers,
 * do not start with `__` and are not `__proto__`/`constructor`/`prototype`;
 * a key that would shadow a name the envelope's own program uses
 * (`SCRIPT_REFUSED_GLOBALS`) is refused as `invalid_request` rather than
 * failing the job as the script's fault.
 * The result comes back through `evalInContext`'s JSON projection and is then
 * sanitised (depth 8, the same three keys dropped, non-JSON → null), so what a
 * host receives is inert data whatever the guest built.
 *
 * # What a result is evidence OF
 *
 * `ok`, `value` and `error` are what the guest program put in `__replies`, and
 * `__replies` is an ordinary array at the guest's global scope. A script can
 * push `{ok: true, value: "forged"}` and then throw, and the host reports
 * success; it can replace `__emit` outright. `evalInContext` offers no slot
 * the guest cannot reach, so there is nowhere better to put the channel.
 *
 * That is a script lying about ITSELF, inside the sandbox it already owns, so
 * it is not a boundary being crossed: a requester that supplied the source
 * cannot be misled about the source by the source. It matters for a requester
 * that treats `ok: false` as a trustworthy signal about something else — a
 * condition lane reading "this condition was false" out of a failure, say.
 * That reading is not supported. What IS host-side, and cannot be forged from
 * the guest: `errorKind` (from `lastErrorKind`, a trap, or the envelope's own
 * classification), the engine identity, and the refusal of a malformed
 * request.
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
  /**
   * A Python contract. `contract`, `files`, `entry` and `call` are the project
   * every evaluation carries; `modes` are the wrappings a job may name (see
   * `ScriptPythonMode`). Every string in it is the requester's.
   */
  python?: {
    contract: string;
    files: Record<string, string>;
    entry: string;
    call: string;
    modes?: ScriptPythonMode[];
  };
}

/**
 * ONE wrapping of a requester's source into a runnable project — the piece a
 * `files`/`entry`/`call` job cannot carry, because it is per-wrapping rather
 * than per-project.
 *
 * A mode is a template plus arithmetic and nothing else. The runner merges
 * `files` over `python.files`, writes `before + source + after` to `block`,
 * runs `python.entry`/`python.call` as any other Python job, and maps
 * locations the engine reports in `block` back onto the author's lines. It
 * never reads `name` for meaning, never has a mode of its own, and never
 * learns what a wrapping is FOR: which mode to run is the requester's choice,
 * named in the job.
 */
export interface ScriptPythonMode {
  /** How a job names this wrapping. Unique within a profile; opaque to the runner. */
  name: string;
  /** Files this wrapping adds to `python.files` (the entry module among them). May not shadow one. */
  files: Record<string, string>;
  /** The file the wrapped source is written to. Not a key of `python.files` or of `files`. */
  block: string;
  /** Text before the author's source in `block`. */
  before: string;
  /** Text after the author's source in `block`. */
  after: string;
  /**
   * Generated lines before the author's first line — the number of newlines in
   * `before`. Engine line N in `block` is author line N − `lineOffset`.
   */
  lineOffset: number;
  /**
   * The function to call instead of `python.call`. A wrapping decides what
   * shape the entry module has and so what there is to call: a contract whose
   * every mode answers the same way needs none of these, and one with a mode
   * that only compiles the block needs exactly one.
   */
  call?: string;
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
   *   * `parse`    one expression, never invoked. Parsed by the host's
   *                  `parseExpression` when it has one, and compiled by the
   *                  engine as `new Function("return (" + source + ")")`
   *                  either way
   */
  mode: ScriptJsMode;
  source: string;
  /** Installed as guest globals (filtered), for every mode. `SCRIPT_REFUSED_GLOBALS` are not allowed. */
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

/**
 * The same Python project, named rather than spelled out: the profile's
 * `python` contract plus one of its `modes`, with only the author's `source`
 * carried here.
 *
 * `modes` is tried in order and only a `source` failure moves on — the same
 * rule as `fallbackOnSourceError`, said once per phase instead of once per
 * file set. The two shapes never mix: a job carries files OR modes.
 */
export interface ScriptPythonModeJob extends ScriptJobBase {
  language: 'python';
  mode: 'python-project';
  /** Profile mode names, tried in order; the next is tried only after a `source` failure. */
  modes: string[];
  /** The author's text, wrapped by whichever mode is running. */
  source: string;
  args?: unknown[];
}

export type ScriptJob = ScriptJsJob | ScriptPythonJob | ScriptPythonModeJob;

/** Whether a Python job names profile modes rather than carrying its own file set. */
export function isScriptPythonModeJob(job: ScriptJob): job is ScriptPythonModeJob {
  return job.mode === 'python-project' && Array.isArray((job as ScriptPythonModeJob).modes);
}

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
  /** Drains the engine's console buffer. Called once per job, for its emptying side effect only. */
  takeOutput?(): unknown;
  dispose(): void;
  initPythonProject?(files: Record<string, string>, entry: string, argv: string[]): unknown;
  pythonCall?(name: string, args: unknown[]): unknown;
}

export type ScriptEngineCtor = new () => ScriptEngine;

export interface ScriptRunOptions {
  /** The JS preamble for a JS job, and the Python contract and modes a mode job is unfolded from. */
  profile?: ScriptProfile;
  /** The engine's `zippProfile().languages`; decides `unsupported` before any Engine is built. */
  languages: readonly string[];
  /** A WebAssembly trap poisoned the instance: retire the Worker/instance, not just the Engine. */
  onEngineTrap?: (error: unknown) => void;
  /**
   * Parse `source` as ONE expression and throw if it is not; mode `parse`
   * only.
   *
   * Mode `parse` answers "is this a valid expression?", and the engine-side
   * wrapper cannot answer it: the source is concatenated into
   * `return (` + source + `)`, so `1); globalThis.pwned = (1` closes the
   * parenthesis, adds a statement and compiles — `{ok: true, value: null}` for
   * source that is not an expression at all. (Inert: the function is never
   * invoked. Wrong, all the same, and a condition lane reading `ok` as "this
   * expression is well-formed" is reading something that was never checked.)
   * No purely textual wrapper fixes it — a comma survives every bracket — so
   * the answer is a parser.
   *
   * A host that has one supplies it (`oaiy script` does, from `acorn`); a host
   * that does not gets the engine's compile alone, and the doc for mode
   * `parse` says which check it got. A throw is the JOB's failure, `guest`,
   * never a refusal of the request: answering "no" is what this mode is FOR.
   */
  parseExpression?: (source: string) => void;
}

export interface ScriptHostOptions {
  engine: ScriptEngineIdentity;
  /** Verifies `profile.preambleSha256`. Without it a request carrying a profile is refused. */
  sha256?: (text: string) => string;
  onEngineTrap?: (error: unknown) => void;
  /** Passed through to every job: see `ScriptRunOptions.parseExpression`. */
  parseExpression?: (source: string) => void;
}

/** Output depth kept; deeper levels become null. */
export const SCRIPT_MAX_OUTPUT_DEPTH = 8;

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const PYTHON_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
/** How a profile mode may be named, and how a job may name one. Opaque to the runner. */
const MODE_NAME = /^[A-Za-z][A-Za-z0-9_-]*$/;
/** Modes one profile may define, and modes one job may list. Bounds, not judgements. */
export const SCRIPT_MAX_PROFILE_MODES = 32;
export const SCRIPT_MAX_JOB_MODES = 8;
/** The largest `lineOffset` a mode may declare: a wrapper taller than this is not a wrapper. */
export const SCRIPT_MAX_LINE_OFFSET = 1_000_000;
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
export const SCRIPT_ENVELOPE_GLOBALS: readonly string[] = [
  '__replies', '__emit', '__out', '__ctx', '__args', '__k', '__fn', '__asBody',
  // The intrinsics the envelope snapshots before any requester code runs — see
  // EMIT_PREAMBLE.
  '__jsonParse', '__isArray', '__hasOwn',
];

/**
 * `globals` keys a request may NOT use.
 *
 * `globals` are installed as guest globals, so a key that names something the
 * envelope's OWN program uses does not shadow it for the script alone — it
 * shadows it for the envelope. That failed the job as `guest`, and `guest`
 * means the script's own fault (see the `errorKind` list above), so a
 * requester was told its script was broken when its REQUEST was:
 *
 *   * `{JSON: 1}` — "undefined is not a function", because the bootstrap
 *     parses `globals` and `args` with it;
 *   * `{eval: 1}`, `{Function: 1}` — the mode body cannot compile the source;
 *   * `{Object: 1, b: 2}` — the install loop's own `hasOwnProperty` died
 *     mid-loop, so `b` was never installed either. `{b: 2, Object: 1}`
 *     survived only because `Object` happened to be installed last: whether
 *     the job ran at all depended on key order.
 *
 * So they are refused as `invalid_request`, before any Engine exists, in the
 * same breath as every other malformed request — and the envelope stops
 * depending on names a guest could reach anyway (EMIT_PREAMBLE's snapshots),
 * so a `prepare` hook or a profile preamble cannot do it either.
 *
 * `protocol/v1/script-request.schema.json` carries the same list, and
 * `ui/tests/zipp-script.mjs` fails if the two drift.
 */
export const SCRIPT_REFUSED_GLOBALS: readonly string[] = [
  // The envelope's own program depends on these.
  'Array', 'Function', 'JSON', 'Object', 'RegExp', 'String', 'eval', 'globalThis',
  // The engine's preamble and the guest shims bind these.
  'console', 'db', 'host', 'localStorage', 'navigator', 'print', 'window',
];
/** The largest `budgetMs` a job may ask a host watchdog for (the schema's ceiling). */
export const SCRIPT_MAX_BUDGET_MS = 60_000;

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

function requireInt(obj: Record<string, unknown>, key: string, where: string, min: number, max: number): number {
  const v = obj[key];
  if (!Number.isInteger(v) || (v as number) < min || (v as number) > max) {
    throw new Invalid(`${where}: ${key} must be an integer from ${min} to ${max}, not ${JSON.stringify(v) ?? String(v)}`);
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
    noExtraKeys(raw.python, ['contract', 'files', 'entry', 'call', 'modes'], at);
    const files = requireFiles(raw.python.files, at);
    const entry = requireString(raw.python, 'entry', at, 128, 1);
    profile.python = {
      contract: requireString(raw.python, 'contract', at, 128, 1),
      files,
      entry,
      call: requireString(raw.python, 'call', at, 128, 1),
    };
    if (raw.python.modes !== undefined) profile.python.modes = validateModes(raw.python.modes, at, files, entry);
  }
  return profile;
}

/**
 * The profile's wrappings. Every field is checked here because a mode is
 * SPENT by the runner without a second look: a `block` that collides with a
 * contract file would silently replace it, a `lineOffset` that is not a whole
 * count would name a line that is not a line, and two modes of one name would
 * make "which wrapping ran" depend on iteration order. Each is refused, and
 * the refusal names the mode.
 *
 * `entry` is held to a Python identifier only HERE, where modes are present:
 * the runner passes it to `initPythonProject` for a mode job, so a profile
 * that names a module Python could not import is unusable rather than merely
 * odd — and a profile without modes keeps the shape it was accepted with.
 */
function validateModes(raw: unknown, at: string, contractFiles: Record<string, string>, entry: string): ScriptPythonMode[] {
  if (!Array.isArray(raw) || raw.length === 0) throw new Invalid(`${at}.modes must be a non-empty array of modes`);
  if (raw.length > SCRIPT_MAX_PROFILE_MODES) throw new Invalid(`${at}.modes has ${raw.length} modes; at most ${SCRIPT_MAX_PROFILE_MODES}`);
  if (!PYTHON_IDENTIFIER.test(entry)) {
    throw new Invalid(`${at}: entry must be a Python module name to unfold a mode, not ${JSON.stringify(entry)}`);
  }
  const modes: ScriptPythonMode[] = [];
  const seen = new Set<string>();
  raw.forEach((item, index) => {
    let here = `${at}.modes[${index}]`;
    if (!isPlainObject(item)) throw new Invalid(`${here} must be an object`);
    if (typeof item.name === 'string' && item.name !== '') here += ` (name ${JSON.stringify(item.name)})`;
    noExtraKeys(item, ['name', 'files', 'block', 'before', 'after', 'lineOffset', 'call'], here);
    const name = requireString(item, 'name', here, 64, 1);
    if (!MODE_NAME.test(name)) throw new Invalid(`${here}: name must be a letter followed by letters, digits, "_" or "-", not ${JSON.stringify(name)}`);
    if (seen.has(name)) throw new Invalid(`${at}.modes: two modes are named ${JSON.stringify(name)}`);
    seen.add(name);
    const files = requireFiles(item.files, here);
    const block = requireString(item, 'block', here, 128, 1);
    for (const file of Object.keys(files)) {
      if (Object.prototype.hasOwnProperty.call(contractFiles, file)) {
        throw new Invalid(`${here}: files[${JSON.stringify(file)}] would replace a file of ${at}.files; give it another name`);
      }
    }
    if (Object.prototype.hasOwnProperty.call(files, block) || Object.prototype.hasOwnProperty.call(contractFiles, block)) {
      throw new Invalid(`${here}: block ${JSON.stringify(block)} is already a file of this project; the wrapped source would replace it`);
    }
    const before = requireString(item, 'before', here);
    const after = requireString(item, 'after', here);
    const lineOffset = requireInt(item, 'lineOffset', here, 0, SCRIPT_MAX_LINE_OFFSET);
    // The entry module has to BE in the project: `entry` or `entry.py`. This is
    // the fault a mode exists to make impossible — a contract that names an
    // entry no file provides runs nothing on any host but the one that wrote it.
    const provides = (file: string) => Object.prototype.hasOwnProperty.call(files, file) || Object.prototype.hasOwnProperty.call(contractFiles, file);
    if (!provides(entry) && !provides(`${entry}.py`)) {
      throw new Invalid(`${here}: no file is the entry module ${JSON.stringify(entry)} (expected ${JSON.stringify(entry)} or ${JSON.stringify(`${entry}.py`)})`);
    }
    const mode: ScriptPythonMode = { name, files, block, before, after, lineOffset };
    if (item.call !== undefined) mode.call = requireString(item, 'call', here, 128, 1);
    modes.push(mode);
  });
  return modes;
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

/**
 * A Python job that names profile modes. It carries the author's `source` and
 * nothing else about the project: `files`, `entry`, `call` and a second file
 * set all come from the profile, so passing any of them here is two answers to
 * one question and is refused rather than resolved (decision 5-1). Whether the
 * names it lists EXIST is checked once the profile is known — see
 * `validateScriptRequest`.
 */
function validatePythonModeJob(raw: Record<string, unknown>, base: ScriptJobBase, where: string): ScriptPythonModeJob {
  for (const key of ['files', 'entry', 'call', 'fallbackOnSourceError']) {
    if (raw[key] !== undefined) {
      throw new Invalid(`${where}: modes and ${key} are two ways to build one project; pass one or the other`);
    }
  }
  noExtraKeys(raw, ['id', 'language', 'mode', 'modes', 'source', 'args', 'budgetMs', 'instructionSteps'], where);
  if (!Array.isArray(raw.modes) || raw.modes.length === 0) throw new Invalid(`${where}: modes must be a non-empty array of mode names`);
  if (raw.modes.length > SCRIPT_MAX_JOB_MODES) throw new Invalid(`${where}: modes lists ${raw.modes.length} modes; at most ${SCRIPT_MAX_JOB_MODES}`);
  const modes = raw.modes.map((name, i) => {
    if (typeof name !== 'string' || name.length === 0 || name.length > 64 || !MODE_NAME.test(name)) {
      throw new Invalid(`${where}: modes[${i}] must be a mode name, not ${JSON.stringify(name)}`);
    }
    return name;
  });
  const job: ScriptPythonModeJob = {
    ...base,
    language: 'python',
    mode: 'python-project',
    modes,
    source: requireString(raw, 'source', where),
  };
  if (raw.args !== undefined) {
    if (!Array.isArray(raw.args)) throw new Invalid(`${where}: args must be an array`);
    requireJson(raw.args, `${where}: args`);
    job.args = raw.args;
  }
  return job;
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
  const budgetMs = optionalInt(raw, 'budgetMs', where, 1, SCRIPT_MAX_BUDGET_MS);
  if (budgetMs !== undefined) base.budgetMs = budgetMs;
  const steps = optionalInt(raw, 'instructionSteps', where, 1, ZIPP_MAX_INSTRUCTION_BUDGET_STEPS);
  if (steps !== undefined) base.instructionSteps = steps;

  if (mode === 'python-project') {
    if (language !== 'python') throw new Invalid(`${where}: mode "python-project" needs language "python"`);
    if (raw.modes !== undefined) return validatePythonModeJob(raw, base, where);
    if (raw.source !== undefined) throw new Invalid(`${where}: source is for a job that names modes; a job that carries its own files puts the source in one of them`);
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
    for (const key of Object.keys(raw.globals)) {
      if (SCRIPT_REFUSED_GLOBALS.includes(key)) {
        throw new Invalid(`${where}: globals may not shadow ${JSON.stringify(key)}, which the guest program itself uses`);
      }
    }
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
 * Every mode a job names must be defined by the request's own profile. A name
 * with no definition is a malformed REQUEST, not a job that fails: there is
 * nothing to run and no source of truth to guess from, so it is refused whole,
 * naming the job and the mode.
 */
function checkModeReferences(request: ScriptRequest): void {
  const defined = new Set((request.profile?.python?.modes ?? []).map((m) => m.name));
  request.jobs.forEach((job, index) => {
    if (!isScriptPythonModeJob(job)) return;
    const where = `jobs[${index}] (id ${JSON.stringify(job.id)})`;
    for (const name of job.modes) {
      if (!request.profile?.python) {
        throw new Invalid(`${where}: names the mode ${JSON.stringify(name)}, and this request carries no profile python contract to define it`);
      }
      if (!defined.has(name)) {
        throw new Invalid(`${where}: names the mode ${JSON.stringify(name)}, which the profile's python.modes does not define`);
      }
    }
  });
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
    // Third pass, once both halves are known: a job may only name a mode this
    // request also carries the definition of. The order matters — jobs, then
    // profile, then the join — so a malformed job is still reported as a
    // malformed job whatever the profile is.
    checkModeReferences(request);
    return { ok: true, request };
  } catch (e) {
    if (e instanceof Invalid) return { ok: false, error: { code: 'invalid_request', message: e.message } };
    throw e;
  }
}

// ---------------------------------------------------------------------------
// The JS program.
// ---------------------------------------------------------------------------

/**
 * The reply channel, the intrinsics the envelope will need later, and the
 * output stubs.
 *
 * `print` IS stubbed: a call reaches this function and writes nothing.
 * `console` is NOT, whatever this assignment looks like. Measured on v0.0.19:
 * after it, `String(console.log)` reads back as the stub, and
 * `console.log("x")` still puts "x" in the engine's own output buffer, where
 * `takeOutput()` finds it. `zipp-executor.ts`'s module comment says the same
 * of the intrinsic from the other side. So the assignment is kept for what it
 * does do — a guest reading or replacing `console` sees the stub, not Zipp's
 * object — and NOT described as something it does not.
 *
 * Nothing crosses to the host either way: the envelope never calls
 * `takeOutput()` for a result, and `attempt` drains and discards the buffer
 * once the job has run. What a chatty script DOES hit is the engine's
 * `lifetimeOutputBytes` (8 MiB on v0.0.19), which draining does not reset —
 * measured, not assumed: nine 1 MiB rounds with a drain after each still ended
 * in `RangeError: script exceeded its output budget`, reported as `resource`.
 * Each job gets a fresh Engine, so the ceiling is per job and a job that hits
 * it fails on its own account.
 *
 * The snapshots are taken HERE, first, before the guest shims, before a
 * profile preamble, before a `prepare` hook and long before `globals` are
 * installed — so the bootstrap that parses the request's data and the loop
 * that installs it hold their own references rather than reading a global
 * name that anything downstream could have replaced. `SCRIPT_REFUSED_GLOBALS`
 * already refuses the request that did this; this is what makes the envelope
 * not depend on that refusal being complete.
 */
const EMIT_PREAMBLE = `var __replies = [];
function __emit(o) { __replies.push(o); }
var __jsonParse = JSON.parse;
var __isArray = Array.isArray;
var __hasOwn = Object.prototype.hasOwnProperty;
globalThis.print = function () {};
globalThis.console = { log: function(){}, warn: function(){}, error: function(){}, info: function(){}, debug: function(){} };
`;

/** Installs `__ctx`'s safe own keys as globals. Same rule as the module comment states. */
const INSTALL_GLOBALS = `  for (var __k in __ctx) {
    if (__hasOwn.call(__ctx, __k)
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
      // Compiled by the engine and never invoked: an unbalanced `}` is a
      // SyntaxError, not an escape, and a side-effecting expression stays
      // inert. It does NOT prove the source is one expression — the source is
      // concatenated into this wrapper, so `1); globalThis.pwned = (1` becomes
      // two perfectly legal statements and compiles. That is what
      // `ScriptRunOptions.parseExpression` is for, and it runs before this.
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
  var __ctx; try { __ctx = __jsonParse(${globals}); } catch (e) { __ctx = {}; }
  var __args; try { __args = __jsonParse(${args}); } catch (e) { __args = []; }
${prepare}  if (!__out) {
    if (__ctx === null || typeof __ctx !== 'object') __ctx = {};
    if (!__isArray(__args)) __args = [__args];
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
    // Drop whatever the job wrote to the engine's console, and do it HERE:
    // a JS job's program IS its source, so the guest has already run by the
    // time `init` returns (measured), and nothing downstream ever reads the
    // buffer. This does not raise the engine's 8 MiB output ceiling — that is
    // a lifetime total a drain does not reset — it keeps a job's output from
    // outliving the job on an engine a host chose to keep.
    try {
      engine.takeOutput?.();
    } catch {
      // an engine without one, or one already torn down: not the job's problem
    }
    phase = 'run';
    // The call is a re-entry: it gets the job's whole budget, not what init
    // left. The engine ANSWERS that request — `false` for a budget already
    // spent, or a disposed engine — and the answer used to be dropped. The
    // job would then have run on whatever was left, and failed `resource` on
    // an entry that had every right to succeed. A renewal that did not happen
    // is the host's problem and is reported as one.
    if (engine.renewInstructionBudget && !engine.renewInstructionBudget()) {
      return { ok: false, kind: 'host', error: 'Zipp would not renew the instruction budget for this call', phase };
    }
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

function runJs(
  Engine: ScriptEngineCtor,
  job: ScriptJsJob,
  steps: number | undefined,
  profile: ScriptProfile | undefined,
  onEngineTrap?: (e: unknown) => void,
  parseExpression?: (source: string) => void,
): ScriptJobResult {
  // Mode `parse`'s real check, when the host has a parser. Before the Engine,
  // because a source that is not one expression has already failed and there
  // is nothing to compile it for.
  if (job.mode === 'parse' && parseExpression) {
    try {
      parseExpression(job.source);
    } catch (e) {
      return { id: job.id, ok: false, errorKind: 'guest', error: errText(e) };
    }
  }
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

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * The engine's message with locations in the mode's block file moved onto the
 * author's own lines.
 *
 * Zipp writes a location in a project file four ways, all measured on v0.0.19
 * web-python: `(block.py:N)`, `(block.py:N:C)` (also bare, as
 * `Python: block.py:N:C: …`), `File "block.py", line N` in a traceback, and
 * `block.py: … (at offset K)` for a construct the parser rejects, where K
 * counts UTF-16 units into the file. Each is renumbered; NOTHING else is
 * touched. The file keeps its name, frames in other files keep their lines,
 * and no text is added, dropped or reworded — the message stays the engine's,
 * with the arithmetic done.
 *
 * Which line an engine line means:
 *
 *   * BELOW the wrapper (N > lineOffset): `N − lineOffset`, the author's line,
 *     capped at the author's last line. A location past the end can only be in
 *     `after`, and `after` is fixed text that compiles on its own — it is
 *     reached because of what the author wrote, so the author's last line is
 *     the honest answer.
 *   * ON the splice line (N === lineOffset), when `before` ends in a newline:
 *     line 1. Python reports a multi-line statement at the line that OPENS it,
 *     and a wrapper whose last line opens one (`return (`) is reported there
 *     for anything inside the author's expression — measured, including an
 *     error two author lines down. So this is the author's first line, not a
 *     clamp. When `before` does NOT end in a newline the author's first line
 *     IS line lineOffset + 1 (the plain subtraction already reaches it) and
 *     line lineOffset is wrapper prologue, so it falls to the case below.
 *   * ABOVE it: left exactly as the engine wrote it. Those lines fail on their
 *     own account — an import, a def header — and nothing the author typed can
 *     be blamed for them. The runner reports no line rather than the wrong one,
 *     and never 0 or a negative.
 *
 * Columns are left as the engine reported them. A `before` that does not end
 * in a newline shifts the author's first line sideways, so a column on that
 * line is the block's; `lineOffset` is the only knob and it is a line count.
 */
export function mapAuthorLines(text: string, mode: ScriptPythonMode, source: string): string {
  const block = escapeRegExp(mode.block);
  const authorLines = Math.max(1, source.split(/\r\n|\r|\n/).length);
  // The author's text begins on its own line only when `before` ends with one.
  const splice = mode.before.endsWith('\n') ? mode.lineOffset : mode.lineOffset + 1;
  /** null: the location is inside the wrapper and is not the author's to own. */
  const line = (engineLine: number): number | null =>
    engineLine < splice ? null : Math.min(authorLines, Math.max(1, engineLine - mode.lineOffset));
  return text
    .replace(new RegExp(`(^|\\n)(${block}: .*?) \\(at offset (\\d+)\\)`, 'g'), (whole, lead: string, head: string, k: string) => {
      const at = Number(k) - mode.before.length;
      return at < 0 ? whole : `${lead}${head} (at offset ${Math.min(source.length, at)})`;
    })
    .replace(new RegExp(`File "${block}", line (\\d+)`, 'g'), (whole, n: string) => {
      const at = line(Number(n));
      return at === null ? whole : `File "${mode.block}", line ${at}`;
    })
    .replace(new RegExp(`${block}:(\\d+)(?::(\\d+))?`, 'g'), (whole, n: string, col?: string) => {
      const at = line(Number(n));
      return at === null ? whole : `${mode.block}:${at}${col ? `:${col}` : ''}`;
    });
}

/**
 * A Python job that names profile modes: unfold, run, and answer in the
 * author's line numbers.
 *
 * Each mode is one whole attempt on its own fresh Engine — the project is the
 * contract's files, the mode's files over them, and the author's source
 * wrapped into the mode's block file. The next mode is tried ONLY when the
 * engine said `source` while the project was initialising, which is the
 * engine's own word for "nothing of yours ran": the same rule
 * `fallbackOnSourceError` follows, said per phase instead of per file set.
 *
 * What the runner does NOT do: it does not know what any mode MEANS, does not
 * read `name` for anything but lookup, has no mode of its own, and adds
 * nothing to the project the profile did not carry. The failure it reports is
 * the LAST attempt's, with that attempt's own `lineOffset` applied — the
 * phases have different wrappers, so attributing one phase's error with
 * another's arithmetic is exactly the bug this closes.
 */
function runPythonModes(
  Engine: ScriptEngineCtor,
  job: ScriptPythonModeJob,
  steps: number | undefined,
  profile: ScriptProfile | undefined,
  onEngineTrap?: (e: unknown) => void,
): ScriptJobResult {
  const contract = profile?.python;
  // validateScriptRequest refuses this; runScriptJob is also called directly
  // (the CLI's worker does), so it fails closed here as the HOST's fault —
  // the job is unrunnable through no fault of the source.
  if (!contract?.modes?.length) {
    return { id: job.id, ok: false, errorKind: 'host', error: 'This job names script modes and the runner was given no profile that defines any' };
  }
  if (!job.modes.length) {
    return { id: job.id, ok: false, errorKind: 'host', error: 'This job names no script mode to run' };
  }
  const args = JSON.parse(JSON.stringify(job.args ?? [])) as unknown[];
  let last: { out: Extract<Attempt, { ok: false }>; mode: ScriptPythonMode } | undefined;
  for (const name of job.modes) {
    const mode = contract.modes.find((m) => m.name === name);
    if (!mode) {
      return { id: job.id, ok: false, errorKind: 'host', error: `The profile defines no script mode named ${JSON.stringify(name)}` };
    }
    const files = { ...contract.files, ...mode.files, [mode.block]: mode.before + job.source + mode.after };
    const out = attempt(
      Engine,
      steps,
      (engine) => { engine.initPythonProject!(files, contract.entry, []); },
      (engine) => engine.pythonCall!(mode.call ?? contract.call, args),
      onEngineTrap,
    );
    if (out.ok) {
      const result: ScriptJobResult = { id: job.id, ok: true };
      if (out.value !== undefined) result.value = sanitizeScriptValue(out.value);
      return result;
    }
    last = { out, mode };
    if (!(out.phase === 'init' && out.kind === 'source')) break;
  }
  const { out, mode } = last!;
  return { id: job.id, ok: false, errorKind: out.kind, error: mapAuthorLines(out.error, mode, job.source) };
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
    return isScriptPythonModeJob(job)
      ? runPythonModes(Engine, job, steps, opts.profile, opts.onEngineTrap)
      : runPython(Engine, job, steps, opts.onEngineTrap);
  }
  return runJs(Engine, job, steps, opts.profile, opts.onEngineTrap, opts.parseExpression);
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
    parseExpression: opts.parseExpression,
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
