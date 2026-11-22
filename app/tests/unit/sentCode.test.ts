// Code the model sends as text (code_run's `code`, python -c) that fails. Nanbeige 4.3 33B, testing a snake game's
// main.py, sent code_run a script that loaded it with exec(compile(...)), which the sandbox's Python does not have, six
// times over: each time it was told to "read the file the error points to and edit it", and the error was in its script.
// What code_run's Python has is tried here on Zipp's own engine (public/zipp, from `npm install`), handed the project as
// SandboxHost.runProgram hands it (the sandbox's Worker stood in for by this process); the rest on Zipp's words for it.
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ConsoleLine, RunRequest, RunResult } from '../../src/sandbox/protocol';
import type { RunOutcome } from '../../src/sandbox/runner';

const sandbox = vi.hoisted(() => ({ answer: null as null | ((request: RunRequest) => RunOutcome | Promise<RunOutcome>) }));
vi.mock('../../src/sandbox/runner', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../src/sandbox/runner')>()),
  runInSandbox: async (request: RunRequest) => {
    if (!sandbox.answer) throw new Error('the test gave the sandbox no answer');
    return sandbox.answer(request);
  },
}));

import { Agent, sentCodeError } from '../../src/agent/agent';
import { TOOLS, runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import { PACKAGE_NAME, buildPackage } from '../../src/sandbox/pystd/package';
import { Vfs } from '../../src/vfs/vfs';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => {
  sandbox.answer = null;
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------------------------------------------------
// Zipp's engine, in this process
// ---------------------------------------------------------------------------------------------------------------------

// (by a name in a variable: Node's fs has no types in this project, and Zipp's script is not one of its modules)
const load = (specifier: string): Promise<unknown> => import(/* @vite-ignore */ specifier);
const fs = (await load('node:fs')) as { existsSync(path: URL): boolean; readFileSync(path: URL): Uint8Array };
const ZIPP = new URL('../../public/zipp/', import.meta.url);
const HAVE_ZIPP = fs.existsSync(new URL('zipp_wasm_bg.wasm', ZIPP));

interface ZippEngine {
  setAppBridge?(bridge: unknown): void;
  setInstructionBudget(steps: number): boolean;
  initPythonProject(files: unknown, entry: string, argv: unknown): unknown;
  pythonCall(name: string, args: unknown): unknown;
  takeConsole(): unknown;
  takeFailedConsole(): unknown;
  dispose(): void;
  free(): void;
}
interface ZippGlue {
  initSync(module: { module: Uint8Array }): unknown;
  Engine: new () => ZippEngine;
  addPythonPackage(archive: Uint8Array, kernels: unknown): string;
  pythonPackages(): string;
}

let glue: Promise<ZippGlue> | null = null;
function zipp(): Promise<ZippGlue> {
  glue ??= (async () => {
    const g = (await load(new URL('zipp_wasm.js', ZIPP).href)) as ZippGlue;
    g.initSync({ module: fs.readFileSync(new URL('zipp_wasm_bg.wasm', ZIPP)) });
    const info = JSON.parse(g.pythonPackages()) as { engineAbi: string; installed: Array<{ name: string }> };
    if (!info.installed.some((p) => p.name === PACKAGE_NAME)) g.addPythonPackage(await buildPackage(info.engineAbi), null);
    return g;
  })();
  return glue;
}

/** The console as worker.ts reads it. */
function consoleLines(raw: unknown): ConsoleLine[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry: unknown) => {
    if (Array.isArray(entry)) return { err: /err/i.test(String(entry[0])), text: String(entry[1] ?? '') };
    if (entry && typeof entry === 'object') {
      const e = entry as Record<string, unknown>;
      return { err: /err|warn/i.test(String(e.stream ?? e.level ?? '')), text: String(e.text ?? e.line ?? e.message ?? '') };
    }
    return { err: false, text: String(entry) };
  });
}

/** A Python program run as worker.ts runs one, in this process: it asks the page nothing (its files come with it). */
async function onZipp(request: RunRequest): Promise<RunOutcome> {
  if (request.lang !== 'python') throw new Error(`only Python runs on Zipp here, not ${request.lang}`);
  const engine = new (await zipp()).Engine();
  engine.setAppBridge?.({ call: (kind: string) => ({ err: `TypeError: \`${kind}\` is not available in this test` }) });
  engine.setInstructionBudget(request.limits.maxSteps);
  const result: RunResult = { console: [] };
  const entry = request.fileName ?? 'main.py';
  try {
    engine.initPythonProject({ ...(request.files ?? {}), [entry]: request.source }, entry, request.argv ?? []);
  } catch (error) {
    result.error = error instanceof Error ? error.message : String(error);
  }
  for (let i = 0; i < 1000; i++) {
    const page = consoleLines(result.error === undefined ? engine.takeConsole() : engine.takeFailedConsole());
    if (!page.length) break;
    result.console.push(...page);
  }
  try {
    const status = engine.pythonCall('__zipp_py_exit_status', []);
    if (typeof status === 'number' || typeof status === 'bigint') result.exit = Number(status);
    else if (typeof status === 'string' && status) result.exit = status;
  } catch {
    /* no status */
  }
  engine.dispose();
  engine.free();
  return { result, timedOut: false, elapsedMs: 1, hostCalls: 0 };
}

// ---------------------------------------------------------------------------------------------------------------------
// The project, the script, and Zipp's words
// ---------------------------------------------------------------------------------------------------------------------

const SNAKE = `"""Snake game logic."""
state = "ready"
score = 0


def start():
    global state, score
    state = "playing"
    score = 0


def step(dt):
    global score
    if state == "playing":
        score += 1
`;

/** As Nanbeige 4.3 33B sent it to code_run (its line 8 is the exec). */
const SNAKE_TEST = `class M: pass

spec_path = "/apps/snake/main.py"
with open(spec_path) as f:
    code = f.read()
m = M()
m.__dict__["__file__"] = spec_path
exec(compile(code, spec_path, "exec"), m.__dict__)

m.start()
print("state:", m.state)
`;

// As Zipp says them (the tests on its engine below see the first): code given as text is __main__.py.
const EXEC_ERROR = `NameError: name 'exec' is not defined (__main__.py:8)\nTraceback (most recent call last):\n  File "__main__.py", line 8, in <module>\n    exec(compile(code, spec_path, "exec"), m.__dict__)`;
const MODULE_ERROR = `NameError: name 'undefined_name' is not defined (apps/snake/main.py:5)\nTraceback (most recent call last):\n  File "__main__.py", line 3, in <module>\n    main.boom()\n  File "apps/snake/main.py", line 5, in boom\n    return undefined_name + x`;
const JS_ERROR = 'ReferenceError: foo is not defined (in <script>)\n    at <anonymous>\n    at <anonymous>:446:17\n    at <anonymous>:481:1';
const JS_MODULE_ERROR = 'ReferenceError: missingThing is not defined (in boom)\n    at boom (<anonymous>)\n    at <anonymous>:3:5\n    at <anonymous>:446:17\n    at <anonymous>:481:1';

let runs = 0;
/** A program that ended on `error`, as the Worker reports one; each run a little slower, as real runs are. */
const failedWith = (error: string) => (): RunOutcome => ({ result: { console: [], error }, timedOut: false, elapsedMs: 90 + runs++, hostCalls: 0 });
/** A shell command whose program failed so, as the shell reports it. */
const shellFailedWith = (stderr: string) => (): RunOutcome => ({ result: { console: [], shell: { stdout: '', stderr: `${stderr}\n`, exit_code: 1, cwd: '/', env: {} } }, timedOut: false, elapsedMs: 120 + runs++, hostCalls: 1 });

function project(): Vfs {
  const vfs = new Vfs();
  vfs.writeFile('/apps/snake/main.py', SNAKE, { parents: true });
  return vfs;
}
const context = (vfs: Vfs) => ({ vfs, gate: new NetGate(), reads: new Map<string, number>(), shell: { cwd: '/', env: {} } });
type Report = { ok?: boolean; exit_code: number; stdout: string; stderr: string; hint?: string };
async function report(vfs: Vfs, name: 'code_run' | 'sandbox_shell', input: Record<string, unknown>): Promise<Report> {
  const result = await runTool({ id: 'c1', name, input }, context(vfs));
  return JSON.parse(result.content) as Report;
}
const codeRun = (vfs: Vfs, input: Record<string, unknown>) => report(vfs, 'code_run', { language: 'python', ...input });

// ---------------------------------------------------------------------------------------------------------------------

describe.skipIf(!HAVE_ZIPP)("code_run's Python, on Zipp's engine, handed the project as code_run hands it", () => {
  it('has no exec, eval or compile, nor importlib.util or runpy, and code_run says what to do instead', async () => {
    sandbox.answer = onZipp;
    const vfs = project();
    // Nanbeige's script, as it ran in the user's run
    const sent = await codeRun(vfs, { code: SNAKE_TEST });
    expect(sent.ok).toBe(false);
    expect(sent.stderr).toMatch(/^NameError: name 'exec' is not defined \(__main__\.py:8\)\nTraceback \(most recent call last\):\n {2}File "__main__\.py", line 8, in <module>/);
    expect(sent.hint).toBe(
      "exec, eval and compile do not exist in this sandbox's Python (nor do importlib.util and runpy), so a script cannot run another file's code from its text. " +
        "Import the module instead: code given as text runs at the project's root, where a project file imports by its path (from apps.snake import main loads /apps/snake/main.py; changing sys.path does not change where imports are found). " +
        "Or write the test as a file beside the module, named test_*.py (/apps/snake/test_main.py, which can import main), and run it with code_run's file: its test_ functions run as pytest runs them.",
    );
    for (const [code, name] of [['print(eval("1 + 2"))', 'eval'], ['c = compile("1 + 2", "<s>", "eval")', 'compile']] as const) {
      const said = await codeRun(vfs, { code });
      expect(said.stderr, name).toContain(`NameError: name '${name}' is not defined (__main__.py:1)`);
      expect(said.hint, name).toMatch(/^exec, eval and compile do not exist in this sandbox's Python/);
    }
    for (const module of ['importlib.util', 'runpy']) {
      expect((await codeRun(vfs, { code: `import ${module}` })).stderr, module).toContain(`ModuleNotFoundError: No module named '${module}'`);
    }
    // called in a file of the project, not in the code sent: that file and its line
    vfs.writeFile('/apps/snake/rules.py', 'def score(expr):\n    return eval(expr)\n');
    const inFile = await codeRun(vfs, { code: 'from apps.snake import rules\nrules.score("1 + 2")' });
    expect(inFile.stderr).toContain("NameError: name 'eval' is not defined (apps/snake/rules.py:2)");
    expect(inFile.hint).toBe(
      "exec, eval and compile do not exist in this sandbox's Python (nor do importlib.util and runpy), and apps/snake/rules.py calls eval on its line 2: write that part without it. " +
        "To load another file's code, import it as a module: code given as text runs at the project's root, where a project file imports by its path (from apps.snake import rules loads /apps/snake/rules.py; changing sys.path does not change where imports are found). " +
        "Or write the test as a file beside the module, named test_*.py (/apps/snake/test_rules.py, which can import rules), and run it with code_run's file: its test_ functions run as pytest runs them.",
    );
  });

  it('imports a project module by its path from code given as text, and one beside a script run as a file, and runs a test_*.py file as pytest does, as its description says', async () => {
    sandbox.answer = onZipp;
    const vfs = project();
    const description = TOOLS.find((t) => t.name === 'code_run')!.description;
    expect(description).toContain('no pip, and no exec, eval, compile, importlib.util or runpy');
    expect(description).toContain('Python `code` runs at the project root, where a project module imports by its path (from apps.game import main loads /apps/game/main.py; sys.path changes nothing)');
    expect(description).toContain('a script run with `file` imports the modules beside it by name, and one named test_*.py runs its test_ functions as pytest does');
    expect(description).not.toContain('no csv module');

    // `code`, at the root: by its path; sys.path changes nothing
    const byPath = await codeRun(vfs, { code: 'from apps.snake import main\nmain.start()\nmain.step(0.1)\nprint(main.state, main.score)' });
    expect(byPath).toMatchObject({ ok: true, stdout: 'playing 1\n' });
    const viaPath = await codeRun(vfs, { code: 'import sys\nsys.path.insert(0, "/apps/snake")\nimport main' });
    expect(viaPath.stderr).toContain("ModuleNotFoundError: No module named 'main' (__main__.py:3)");
    // `file`: the modules beside it, by name
    vfs.writeFile('/apps/snake/check.py', 'import main\nmain.start()\nprint(main.state)\n');
    expect(await codeRun(vfs, { file: 'apps/snake/check.py' })).toMatchObject({ ok: true, stdout: 'playing\n' });
    // a file named test_*.py: its test_ functions, as pytest runs them
    vfs.writeFile('/apps/snake/test_main.py', 'import main\n\n\ndef test_start():\n    main.start()\n    assert main.state == "playing"\n\n\ndef test_score():\n    main.start()\n    main.step(0.1)\n    assert main.score == 2, "score is " + str(main.score)\n');
    const tested = await codeRun(vfs, { file: 'apps/snake/test_main.py' });
    expect(tested.exit_code).toBe(1);
    expect(tested.stdout).toContain('test_start PASSED');
    expect(tested.stdout).toContain('FAILED test_score - AssertionError: score is 1');
    expect(tested.stdout).toContain('1 failed, 1 passed');
    // and the csv module is there
    expect(await codeRun(vfs, { code: 'import csv, io\nprint(next(csv.reader(io.StringIO("a,b"))))' })).toMatchObject({ ok: true, stdout: "['a', 'b']\n" });
  });
});

describe('what a run that reached for exec, eval or compile is told', () => {
  it("tells sandbox_shell's python the same, in the shell's terms, and python -m py_compile that it cannot run here; nothing of other errors", async () => {
    const vfs = project();
    sandbox.answer = shellFailedWith("NameError: name 'exec' is not defined (__main__.py:1)\nTraceback (most recent call last):\n  File \"__main__.py\", line 1, in <module>\n    exec(open('apps/snake/main.py').read())");
    const shell = await report(vfs, 'sandbox_shell', { command: `python -c "exec(open('apps/snake/main.py').read())"` });
    expect(shell.hint).toBe(
      "exec, eval and compile do not exist in this sandbox's Python (nor do importlib.util and runpy), so a script cannot run another file's code from its text. " +
        "Import the module instead: python FILE imports the modules beside FILE by name, and python -c imports those of the working folder (cd /apps/snake, then import main); from the project's root a project file imports by its path (from apps.snake import main). " +
        'A file named test_*.py run with python runs its test_ functions as pytest runs them.',
    );
    // (the shell's own py_compile calls compile())
    sandbox.answer = shellFailedWith("NameError: name 'compile' is not defined (__main__.py:3)\nTraceback (most recent call last):\n  File \"__main__.py\", line 3, in <module>\n    compile(open(f).read(), f, 'exec')");
    expect((await report(vfs, 'sandbox_shell', { command: 'python -m py_compile apps/snake/main.py' })).hint).toBe(
      "python -m py_compile needs compile(), which this sandbox's Python does not have, so it cannot run here. To check that a file parses, import it (a syntax error is reported with its file and line) or run it.",
    );
    // code_run with no path in it: a path for the example
    sandbox.answer = failedWith("NameError: name 'eval' is not defined (__main__.py:1)");
    expect((await codeRun(vfs, { code: 'print(eval("1"))' })).hint).toContain('(from apps.game import main loads /apps/game/main.py;');
    // other names, and other errors: no hint
    for (const error of ["NameError: name 'execute' is not defined (__main__.py:1)", "NameError: name 'evaluate' is not defined (__main__.py:1)", MODULE_ERROR]) {
      sandbox.answer = failedWith(error);
      expect((await codeRun(vfs, { code: 'execute()' })).hint, error).toBeUndefined();
      sandbox.answer = shellFailedWith(error);
      expect((await report(vfs, 'sandbox_shell', { command: 'python -c "execute()"' })).hint, error).toBeUndefined();
    }
  });
});

describe('a step repeated with the same error in the code it sent', () => {
  const OLD_WORDS = 'If they report errors, fix them now: read the file the error points to and edit it.';
  /**
   * The same call three times, then a reply in words: the third's result, where the note goes. (A shell's report, the
   * same to the letter each time, also stops the run there, as the same failure three times in a row; code_run's says
   * how long it took, and goes on to the sixth.)
   */
  async function repeated(call: { name: string; input: Record<string, unknown> }, answer: () => RunOutcome): Promise<string> {
    sandbox.answer = answer;
    fakeProvider('openai', [...Array.from({ length: 3 }, () => ({ calls: [call] })), { text: 'I will change it.' }]);
    const agent = new Agent({ vfs: project(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'p' });
    await agent.run('create a snake game', () => {});
    const steps = agent.turns.filter((t) => t.role === 'tool');
    expect(steps).toHaveLength(3);
    const third = steps[2];
    const told = third.role === 'tool' ? third.results[0].content : '';
    expect(told, 'the note came').toContain('[OAIY] This step is the same as the 2 before it: the same calls (numbers aside), and the same results.');
    return told;
  }

  it('is told plainly that the error is in the code it sent to code_run, quoting its line, and to change that code', async () => {
    const told = await repeated({ name: 'code_run', input: { code: SNAKE_TEST, language: 'python' } }, failedWith(EXEC_ERROR));
    expect(told).toContain(
      "The error is in the code you sent to code_run, not in a file of the project: line 8: NameError: name 'exec' is not defined. Running that code again gives the same error, so change that code before you run it again (the hint in its result says what to use instead). If you cannot, stop and tell the user what is wrong.",
    );
    expect(told).not.toContain(OLD_WORDS);
  });

  it("says the same of python -c in sandbox_shell, and of JavaScript sent to code_run that loads nothing", async () => {
    const shell = await repeated({ name: 'sandbox_shell', input: { command: `python - <<'EOF'\n${SNAKE_TEST}EOF` } }, shellFailedWith(EXEC_ERROR));
    expect(shell).toContain("The error is in the code you sent to sandbox_shell, not in a file of the project: line 8: NameError: name 'exec' is not defined. Running that code again gives the same error");
    expect(shell).not.toContain(OLD_WORDS);
    const js = await repeated({ name: 'code_run', input: { code: 'const a = 1;\nfoo(a);' } }, failedWith(JS_ERROR));
    expect(js).toContain('The error is in the code you sent to code_run, not in a file of the project: ReferenceError: foo is not defined. Running that code again gives the same error, so change that code before you run it again. If you cannot');
    expect(js).not.toContain(OLD_WORDS);
  });

  it('keeps its words where the error is in a file of the project, the code loads one, or the shell ran a module of its own', async () => {
    expect(await repeated({ name: 'code_run', input: { code: 'from apps.snake import main\nprint("before")\nmain.boom()', language: 'python' } }, failedWith(MODULE_ERROR))).toContain(OLD_WORDS);
    expect(await repeated({ name: 'code_run', input: { code: 'const lib = require("./lib.js");\nlib.boom();' } }, failedWith(JS_MODULE_ERROR))).toContain(OLD_WORDS);
    // (python -m json.tool is the shell's own text, run as __main__.py: the fault is in the file it reads)
    expect(await repeated({ name: 'sandbox_shell', input: { command: 'python -m json.tool data/levels.json' } }, shellFailedWith('JSONDecodeError: Expecting value: line 1 column 1 (char 0) (__main__.py:3)'))).toContain(OLD_WORDS);
  });

  it('reads the error out of a report with a flow\'s note before it and notes after it', () => {
    const content = `[Your flow "lint" ran before code_run.]\n${JSON.stringify({ ok: false, exit_code: 1, stdout: '', stderr: `${EXEC_ERROR}\n`, elapsed_ms: 95 }, null, 1)}\n\n[OAIY] Planned: ...`;
    const call = { id: 'c1', name: 'code_run', input: { code: SNAKE_TEST, language: 'python' } };
    expect(sentCodeError(call, { id: 'c1', name: 'code_run', content, isError: true })).toBe("line 8: NameError: name 'exec' is not defined");
    // what the program wrote to stderr before it does not count; an error without words is named alone (as Zipp says them)
    const said = (stderr: string) => ({ id: 'c1', name: 'code_run', content: JSON.stringify({ ok: false, exit_code: 1, stdout: '', stderr, elapsed_ms: 9 }, null, 1), isError: true });
    expect(sentCodeError(call, said('warn (x.py:3)\nRuntimeError: boom: (1) (__main__.py:3)\nTraceback (most recent call last):\n'))).toBe('line 3: RuntimeError: boom: (1)');
    expect(sentCodeError(call, said('ValueError:  (__main__.py:2)\n'))).toBe('line 2: ValueError');
    expect(sentCodeError(call, said("NameError: name 'exec' is not defined (__oaiy_main__.py:1)\n"))).toBe("line 1: NameError: name 'exec' is not defined");
    // a run that did not fail, or another tool, is not read
    expect(sentCodeError(call, { id: 'c1', name: 'code_run', content, isError: false })).toBeNull();
    expect(sentCodeError({ ...call, name: 'read_file' }, { id: 'c1', name: 'read_file', content, isError: true })).toBeNull();
  });
});
