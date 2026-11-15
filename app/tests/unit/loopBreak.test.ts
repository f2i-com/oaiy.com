import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import { variedFields } from '../../src/agent/protocol';
import { moduleHint, runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { checkProject, guideFor } from '../../src/softn/softn';
import { SOFTN_GUIDE_JAVASCRIPT, SOFTN_GUIDE_PYTHON } from '../../src/softn/guide.generated';
import { LOCAL, OPENAI, fakeProvider } from './fakeProvider';
import type { ProviderConfig } from '../../src/agent/providers/types';

// Nanbeige 4.3 33B (its thinking off), asked to "create a snake game" unattended: it wrote the game's logic in Python and
// tested it with code_run, then sent the same failing script again and again, each step's result saying what was wrong,
// until the run stopped for a person.

afterEach(() => vi.unstubAllGlobals());

const OAIY: ProviderConfig = { id: 'o', type: 'custom', serverKind: 'oaiy', name: 'OAIY', apiKey: '', modelId: 'nanbeige4.3-33b', baseUrl: 'http://127.0.0.1:18452' };

function setup(provider: ProviderConfig) {
  const vfs = new Vfs();
  vfs.writeFile('/src/app.js', 'const greeting = "hello";\n', { parents: true });
  const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'Project: demo' });
  return { vfs, agent };
}

describe('a new SoftN app', () => {
  it("has its logic in JavaScript, in Python only when the user asks for Python", async () => {
    const system = (body: Record<string, unknown>) => String((body.messages as Array<{ content: unknown }>)[0].content);
    const fake = fakeProvider('openai', [{ calls: [{ name: 'guide', input: { topic: 'app' } }] }, { text: 'ok' }]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '' });
    await agent.run('make a game', () => {});
    const guide = system(fake.bodies[1]);
    expect(guide).toContain('its logic in JavaScript in main.js named with <script src> (in Python, main.py, only when the user asks for Python');
    expect(guide).not.toContain('main.js (or main.py)');
  });

  it("is given SoftN's JavaScript guide, whatever .py files the project has; an app whose logic is Python, the Python guide", () => {
    // a new project's starter files, and no app yet: its hello.py chose the Python guide for every first app
    const vfs = new Vfs();
    vfs.writeFile('/hello.py', 'print("hello")\n');
    vfs.writeFile('/hello.js', 'console.log("hello")\n');
    expect(guideFor(vfs, '')).toBe(SOFTN_GUIDE_JAVASCRIPT);
    // an app whose logic is Python keeps its own
    vfs.writeFile('/apps/snake/manifest.json', JSON.stringify({ main: 'index.html' }), { parents: true });
    vfs.writeFile('/apps/snake/index.html', '<App><script src="main.py"></script></App>\n');
    vfs.writeFile('/apps/snake/main.py', 'score = 0\n');
    expect(guideFor(vfs, 'apps/snake')).toBe(SOFTN_GUIDE_PYTHON);
    // and one whose logic is JavaScript its own
    vfs.writeFile('/apps/tetris/manifest.json', JSON.stringify({ main: 'index.html' }), { parents: true });
    vfs.writeFile('/apps/tetris/index.html', '<App><script src="main.js"></script></App>\n');
    vfs.writeFile('/apps/tetris/main.js', 'let score = 0;\n');
    expect(guideFor(vfs, 'apps/tetris')).toBe(SOFTN_GUIDE_JAVASCRIPT);
  });
});

describe("a page's value in quotes with {…} in it", () => {
  it('is an error the check names, with the expression to write', () => {
    // the snake game's page: its score's badges showed "Best {best}" and "Speed {level}/10"
    const vfs = new Vfs();
    vfs.writeFile('/apps/snake/manifest.json', JSON.stringify({ name: 'Snake', version: '1.0.0', main: 'index.html', files: { ui: ['index.html'], logic: ['main.js'] } }), { parents: true });
    vfs.writeFile('/apps/snake/main.js', 'let best = 0\nlet level = 1\nlet score = 0\n');
    vfs.writeFile('/apps/snake/index.html', '<App>\n  <script src="main.js"></script>\n  <Badge content={score} />\n  <Badge variant="warning" content="Best {best}" />\n  <Badge content="Speed {level}/10" />\n</App>\n');
    const said = checkProject(vfs, 'apps/snake').filter((f) => f.level === 'error').map((f) => `${f.file}: ${f.message}`);
    expect(said).toEqual([
      'index.html: line 4: content="Best {best}" is shown as it is written: a value in quotes is text, and {…} is read only outside quotes. Write content={"Best " + best}',
      'index.html: line 5: content="Speed {level}/10" is shown as it is written: a value in quotes is text, and {…} is read only outside quotes. Write content={"Speed " + level + "/10"}',
    ]);
  });

  it('says nothing of values in braces, nor of braces that are no name', () => {
    const vfs = new Vfs();
    vfs.writeFile('/apps/a/manifest.json', JSON.stringify({ name: 'A', version: '1.0.0', main: 'index.html', files: { ui: ['index.html'] } }), { parents: true });
    vfs.writeFile('/apps/a/index.html', '<App>\n  <Badge content={score} />\n  <Input pattern="[0-9]{2,3}" />\n  <Text>Best {best}</Text>\n</App>\n');
    expect(checkProject(vfs, 'apps/a').filter((f) => f.level === 'error')).toEqual([]);
  });
});

describe('code_run given both code and a file', () => {
  it('says it runs one of them, and how to run the test that was sent', async () => {
    // Nanbeige 4.3 33B sent a test of its game as code with file: its main.py, which ran instead, printed nothing and
    // said ok: 28 tests over, none of them ever ran
    const vfs = new Vfs();
    vfs.writeFile('/apps/snake/main.py', 'score = 0\n', { parents: true });
    const ctx = { vfs, gate: new NetGate(), reads: new Map<string, number>(), shell: { cwd: '/', env: {} } };
    const result = await runTool({ id: 'c1', name: 'code_run', input: { language: 'python', file: '/apps/snake/main.py', code: 'print(score)' } }, ctx as never);
    expect(result.isError).toBe(true);
    expect(result.content).toContain('code_run takes code or file, not both: given both, it runs the file and never reads the code');
    expect(result.content).toContain('from apps.snake import main');
  });
});

describe('a step after the same step again and again', () => {
  it('is sampled anew once, with a temperature of 1 and a seed of its own, by OAIY and other local servers', async () => {
    const same = { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] };
    const fake = fakeProvider('openai', [same, same, same, same, { text: 'done' }]);
    await setup(OAIY).agent.run('look at the app', () => {});
    // told from the third same step on (the request after it, the fourth, is sampled anew)
    expect(JSON.stringify(fake.bodies[3].messages)).toContain('This step is the same as the');
    expect(fake.bodies.slice(0, 3).map((b) => b.temperature)).toEqual([undefined, undefined, undefined]);
    expect(fake.bodies[3].temperature).toBe(1);
    expect(Number.isInteger(fake.bodies[3].seed)).toBe(true);
    // the fourth was the same again: told again, sampled anew again; one that is not is not
    expect(fake.bodies[4].temperature).toBe(1);
    expect(fake.bodies).toHaveLength(5);
  });

  it('asks nothing of an API, which may refuse a temperature', async () => {
    const same = { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] };
    const fake = fakeProvider('openai', [same, same, same, { text: 'done' }]);
    await setup(OPENAI).agent.run('look at the app', () => {});
    expect(JSON.stringify(fake.bodies[3].messages)).toContain('This step is the same as the');
    expect(fake.bodies.map((b) => [b.temperature, b.seed])).toEqual(fake.bodies.map(() => [undefined, undefined]));
  });

  it('asks it of a request only where it is asked to', () => {
    expect(variedFields(OAIY, {})).toEqual({});
    expect(variedFields(OAIY, { varied: true })).toMatchObject({ temperature: 1, seed: expect.any(Number) });
    expect(variedFields(LOCAL, { varied: true })).toMatchObject({ temperature: 1 });
    expect(variedFields(OPENAI, { varied: true })).toEqual({});
  });
});

describe('a module code given as text cannot import', () => {
  const files = (paths: string[]) => (file: string) => paths.filter((p) => p === file || p.endsWith(`/${file}`));
  const failed = "ModuleNotFoundError: No module named 'main' (__main__.py:4)\nTraceback (most recent call last):\n  File \"__main__.py\", line 4, in <module>\n";
  const script = 'import sys\nsys.path.insert(0, "/apps/snake")\nimport importlib\nm = importlib.import_module("main")\n';

  it('is named as the project file it is, with the line that imports it by its path', () => {
    const hint = moduleHint(failed, script, 'code_run', files(['apps/other/main.py', 'apps/snake/main.py']));
    // the one sys.path was pointed at, of the two
    expect(hint).toContain("main is the project's /apps/snake/main.py");
    expect(hint).toContain('from apps.snake import main');
    expect(hint).toContain('changing sys.path does not change where imports are found here');
    expect(hint).toContain("/apps/snake/test_main.py, which imports main by name) and run that file with code_run's file");
  });

  it('in a folder whose name is no Python name, is tested from a file beside it', () => {
    const hint = moduleHint(failed, 'import main', 'code_run', files(['apps/my-game/main.py']));
    expect(hint).not.toContain('from apps');
    expect(hint).toContain('/apps/my-game/test_main.py');
  });

  it("tells the shell's python -c to change to its folder", () => {
    const hint = moduleHint(failed, 'python -c "import main"', 'sandbox_shell', files(['apps/snake/main.py']));
    expect(hint).toContain('cd /apps/snake first, then import main');
  });

  it('says nothing of a module the project has no file of, nor of an import a project file failed', () => {
    expect(moduleHint(failed, script, 'code_run', files([]))).toBeNull();
    expect(moduleHint("ModuleNotFoundError: No module named 'helper' (apps/snake/main.py:2)", script, 'code_run', files(['apps/snake/helper.py']))).toBeNull();
    expect(moduleHint('NameError: name x is not defined', script, 'code_run', files(['apps/snake/main.py']))).toBeNull();
  });
});
