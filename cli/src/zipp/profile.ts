/**
 * `run --profile <file>`: a script profile (`protocol/v1/script-profile.schema.json`,
 * `zipp-script.ts`'s `ScriptProfile`) applied to a whole workflow run — its
 * `preamble` at the program top level of every flow script, its
 * `instructionSteps` as the run's budget.
 *
 * Everything here happens BEFORE the flow file is read or the engine loads,
 * and every fault is a refusal (exit 1, no result file), never a run on a
 * different prelude or budget than the caller named:
 *
 *   * the document must be the profile shape exactly — checked by the same
 *     validator `oaiy script` runs on a request's `profile`, so the two
 *     commands cannot drift on what a profile is (the digest is verified and
 *     a preamble redeclaring an engine or envelope name is refused there).
 *     That validator rules on the WHOLE document, `hooks` and a `python`
 *     contract with its `modes` included, so one file is a requester's whole
 *     prelude for both commands: a run takes the preamble and the budget from
 *     it, and `oaiy script` unfolds its modes. Nothing here reads either —
 *     they are carried, checked and passed on;
 *   * the preamble must PARSE, and none of its top-level declarations may be
 *     a name the engine's preamble, the envelope or the workflow wrapper binds.
 *     The envelope's own check is a text scan (it has no parser); the CLI has
 *     one, so here it is the syntax tree: destructured `const {host} = …`
 *     counts, a `host` inside a string or a nested function does not;
 *   * a `let`/`const`/`class` redeclaring one of the guest shims (`setTimeout`,
 *     `fetch`, …) is refused too: at program top level after the shims' `var`s
 *     it is a SyntaxError that would take every flow with it. A `var` or a
 *     `function` of the same name is legal JavaScript and REPLACES the stub —
 *     a requester's deliberate choice, allowed;
 *   * `instructionSteps` and `--instruction-budget` together are refused
 *     (decision 5-1): two budgets is an ambiguity, and picking one silently
 *     would run the flow on a budget the caller did not intend.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import * as acorn from 'acorn';
import { SCRIPT_ENVELOPE_GLOBALS, validateScriptRequest, type ScriptProfile } from 'oaiy-core/src/zipp-script';
import { ZIPP_GUEST_SHIMS, ZIPP_PREAMBLE_GLOBALS } from 'oaiy-core/src/zipp-executor';

/** A profile that cannot be applied. The message is for the caller; the run never starts. */
export class ScriptProfileError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ScriptProfileError';
  }
}

/** Lower-case hex sha256 of UTF-8 text — how `preambleSha256` is computed on both sides. */
export function sha256Hex(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

/** Names `buildZippScript` binds at program top level beside the preamble. */
const WRAPPER_GLOBALS: readonly string[] = ['__oaiyConsole'];

type Node = acorn.Node & Record<string, unknown>;

/** The identifiers a binding pattern introduces (`a`, `{a, b: c}`, `[a, ...rest]`, `a = 1`). */
function bindingNames(pattern: unknown, out: string[]): void {
  const node = pattern as Node | null | undefined;
  if (!node || typeof node.type !== 'string') return;
  switch (node.type) {
    case 'Identifier':
      out.push(node.name as string);
      return;
    case 'ObjectPattern':
      for (const prop of node.properties as Node[]) {
        bindingNames(prop.type === 'RestElement' ? prop.argument : prop.value, out);
      }
      return;
    case 'ArrayPattern':
      for (const element of node.elements as (Node | null)[]) bindingNames(element, out);
      return;
    case 'RestElement':
      bindingNames(node.argument, out);
      return;
    case 'AssignmentPattern':
      bindingNames(node.left, out);
      return;
    default:
      return;
  }
}

export interface PreambleDeclarations {
  /** Every top-level declared name, in source order. */
  names: string[];
  /** The subset declared with `let`, `const` or `class` (lexical: a redeclaration is a SyntaxError). */
  lexical: string[];
}

/** Parse `source` as a classic script and list its top-level declarations. Throws `ScriptProfileError` when it does not parse. */
export function scanTopLevelDeclarations(source: string, what = 'the preamble'): PreambleDeclarations {
  let program: acorn.Program;
  try {
    program = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'script' });
  } catch (e) {
    throw new ScriptProfileError(`${what} does not parse: ${e instanceof Error ? e.message : String(e)}`);
  }
  const names: string[] = [];
  const lexical: string[] = [];
  for (const statement of program.body as unknown as Node[]) {
    if (statement.type === 'VariableDeclaration') {
      const found: string[] = [];
      for (const declarator of statement.declarations as Node[]) bindingNames(declarator.id, found);
      names.push(...found);
      if (statement.kind !== 'var') lexical.push(...found);
    } else if (statement.type === 'FunctionDeclaration' && (statement.id as Node | null)?.name) {
      names.push((statement.id as Node).name as string);
    } else if (statement.type === 'ClassDeclaration' && (statement.id as Node | null)?.name) {
      const name = (statement.id as Node).name as string;
      names.push(name);
      lexical.push(name);
    }
  }
  return { names, lexical };
}

let shimNames: Set<string> | null = null;
/** The `var`s the guest shims declare, read from the shims' own text once. */
function guestShimNames(): Set<string> {
  if (!shimNames) shimNames = new Set(scanTopLevelDeclarations(ZIPP_GUEST_SHIMS, 'ZIPP_GUEST_SHIMS').names);
  return shimNames;
}

/**
 * Check a parsed profile document (the JSON, not yet trusted) and return the
 * profile. Structure and digest are the envelope validator's verdict — a
 * profile is wrapped in an empty request so the ONE validator rules on both
 * commands — then the preamble is parsed and its declarations checked.
 */
export function parseScriptProfile(raw: unknown): ScriptProfile {
  const checked = validateScriptRequest({ v: 1, profile: raw, jobs: [] }, { sha256: sha256Hex });
  if (!checked.ok) throw new ScriptProfileError(checked.error.message);
  const profile = checked.request.profile;
  if (!profile) throw new ScriptProfileError('profile: the document carries no profile');

  const { names, lexical } = scanTopLevelDeclarations(profile.preamble);
  const reserved = new Set([...ZIPP_PREAMBLE_GLOBALS, ...SCRIPT_ENVELOPE_GLOBALS, ...WRAPPER_GLOBALS]);
  for (const name of names) {
    if (reserved.has(name)) {
      throw new ScriptProfileError(
        `profile: the preamble declares ${JSON.stringify(name)}, a name the engine, the envelope or the workflow wrapper binds`,
      );
    }
  }
  const shims = guestShimNames();
  for (const name of lexical) {
    if (shims.has(name)) {
      throw new ScriptProfileError(
        `profile: the preamble redeclares the guest shim ${JSON.stringify(name)} with let/const/class, a SyntaxError at program top level (a var or function of that name replaces the shim instead)`,
      );
    }
  }
  return profile;
}

/** Read, parse and check a profile file. Every failure is a `ScriptProfileError` naming the file. */
export function loadScriptProfile(file: string): ScriptProfile {
  let text: string;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (e) {
    throw new ScriptProfileError(`--profile ${file}: cannot be read (${(e as NodeJS.ErrnoException).code ?? String(e)})`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new ScriptProfileError(`--profile ${file}: not JSON (${e instanceof Error ? e.message : String(e)})`);
  }
  try {
    return parseScriptProfile(raw);
  } catch (e) {
    if (e instanceof ScriptProfileError) throw new ScriptProfileError(`--profile ${file}: ${e.message}`);
    throw e;
  }
}

/**
 * The run's instruction budget: the profile's when the profile carries one,
 * else `--instruction-budget`, else `undefined` (the engine's default). BOTH
 * present is refused — never a silent pick (decision 5-1).
 */
export function resolveInstructionBudget(profile: ScriptProfile | undefined, cliBudget: number | undefined): number | undefined {
  if (profile?.instructionSteps !== undefined && cliBudget !== undefined) {
    throw new ScriptProfileError(
      `--profile sets instructionSteps (${profile.instructionSteps}) and --instruction-budget was also given (${cliBudget}); pass one or the other`,
    );
  }
  return profile?.instructionSteps ?? cliBudget;
}
