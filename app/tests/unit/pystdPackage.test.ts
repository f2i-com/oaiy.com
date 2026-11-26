// OAIY's Python package and a Zipp that already has one of its modules. Zipp refuses the whole package and names the
// module in a sentence of its own; the installer leaves that module out and tries again. The sentence is read here as
// Zipp v0.0.41 words it (and on that engine itself, public/zipp, from `npm install`), with other marks around the names
// and with none, and when it names nothing the installer can leave out.
import { describe, expect, it } from 'vitest';
import { PACKAGE_NAME, installPythonPackage, packageModules, providedModule } from '../../src/sandbox/pystd/package';

const TAIL = 'is provided by the engine or another package already';

/** One refusal, worded with these marks around the two names. */
type Wording = (module: string, pkg: string) => string;
const WORDINGS: Array<[string, Wording]> = [
  ['backticks, as Zipp v0.0.41 has them', (m, p) => `addPythonPackage: module \`${m}\` of package \`${p}\` ${TAIL}`],
  ['single quotes', (m, p) => `addPythonPackage: module '${m}' of package '${p}' ${TAIL}`],
  ['double quotes', (m, p) => `addPythonPackage: module "${m}" of package "${p}" ${TAIL}`],
  ['no marks at all', (m, p) => `addPythonPackage: module ${m} of package ${p} ${TAIL}`],
  ['curly quotes', (m, p) => `module “${m}” of package “${p}” ${TAIL}.`],
  ['angle quotes and a version after the package', (m, p) => `module «${m}» of package «${p}» (version 1) ${TAIL}`],
  ['marks around one name only', (m, p) => `module \`${m}\` of package ${p} ${TAIL}`],
];

interface Manifest {
  name: string;
  runtime: boolean;
  modules: string[];
}

/** The manifest of an archive, as Zipp reads it: the lines up to `end`. */
function manifestOf(archive: Uint8Array): Manifest {
  const lines = new TextDecoder().decode(archive).split('\nend\n')[0].split('\n');
  const field = (key: string) => lines.filter((l) => l.startsWith(`${key} `)).map((l) => l.split(' ')[1]);
  return { name: field('name')[0], runtime: field('runtime').length === 1, modules: field('module') };
}

/** A glue that already has `provided`, and refuses a package that brings one of them in the given wording. */
function fakeGlue(provided: string[], wording: Wording, thrown: (message: string) => unknown = (message) => message) {
  const seen: Manifest[] = [];
  const installed: Array<{ name: string }> = [];
  return {
    seen,
    installed,
    pythonPackages: () => JSON.stringify({ engineAbi: 'abi0', installed }),
    addPythonPackage: (archive: Uint8Array, _kernels: unknown): string => {
      const manifest = manifestOf(archive);
      seen.push(manifest);
      const clash = manifest.modules.find((m) => provided.includes(m));
      if (clash !== undefined) throw thrown(wording(clash, manifest.name));
      installed.push({ name: manifest.name });
      return JSON.stringify({ name: manifest.name });
    },
  };
}

const ALL = packageModules().map((m) => m.name);

describe("the module Zipp's refusal names", () => {
  it.each(WORDINGS)('is read with %s', (_label, wording) => {
    expect(providedModule(wording('csv', PACKAGE_NAME))).toBe('csv');
    expect(providedModule(wording('unittest.mock', 'some-package'))).toBe('unittest.mock');
    expect(providedModule(wording('_private_2', PACKAGE_NAME))).toBe('_private_2');
  });

  it('is not read from another refusal, nor from one that names nothing', () => {
    expect(providedModule('addPythonPackage: a different package `oaiy-stdlib` is installed already')).toBeUndefined();
    expect(providedModule('addPythonPackage: package `oaiy-stdlib` was built for engine ABI 1, this engine is 2: use the package built with this engine')).toBeUndefined();
    expect(providedModule(`addPythonPackage: module \`\` of package \`\` ${TAIL}`)).toBeUndefined();
    expect(providedModule(`addPythonPackage: a module of package \`oaiy-stdlib\` ${TAIL}`)).toBeUndefined();
    expect(providedModule('')).toBeUndefined();
  });
});

describe("installing OAIY's Python package into a Zipp that has some of its modules", () => {
  it('has modules to leave out', () => {
    expect(ALL).toContain('csv');
    expect(ALL).toContain('uuid');
    expect(ALL.length).toBeGreaterThan(2);
  });

  it.each(WORDINGS)('leaves out each module Zipp names, with %s, and installs the rest', async (_label, wording) => {
    const glue = fakeGlue(['uuid', 'csv'], wording);
    expect(await installPythonPackage(glue)).toBeUndefined();
    // One try per module Zipp had, and the one that went in.
    expect(glue.seen.length).toBe(3);
    const last = glue.seen[glue.seen.length - 1];
    expect(last.name).toBe(PACKAGE_NAME);
    expect(last.runtime).toBe(true);
    expect(last.modules).toEqual(ALL.filter((m) => m !== 'uuid' && m !== 'csv'));
    expect(glue.installed).toEqual([{ name: PACKAGE_NAME }]);
  });

  it('reads the refusal from an Error as from a string (Zipp throws a string)', async () => {
    const glue = fakeGlue(['csv'], WORDINGS[0][1], (message) => new Error(message));
    expect(await installPythonPackage(glue)).toBeUndefined();
    expect(glue.seen[glue.seen.length - 1].modules).toEqual(ALL.filter((m) => m !== 'csv'));
  });

  it('installs the whole package when Zipp has none of its modules, and nothing when it is installed already', async () => {
    const glue = fakeGlue([], WORDINGS[0][1]);
    expect(await installPythonPackage(glue)).toBeUndefined();
    expect(glue.seen.map((m) => m.modules)).toEqual([ALL]);
    expect(await installPythonPackage(glue)).toBeUndefined();
    expect(glue.seen.length).toBe(1);
  });

  it.each<[string, Wording]>([
    ['a sentence whose words around the name changed', (m, p) => `addPythonPackage: ${p} brings ${m}, which ${TAIL}`],
    ['no name', (_m, p) => `addPythonPackage: a module of package \`${p}\` ${TAIL}`],
    ['the name of a module the package does not have', (_m, p) => `addPythonPackage: module \`not_ours\` of package \`${p}\` ${TAIL}`],
  ])('leaves every module out and says why when the refusal has %s: stdin and the working folder still load', async (_label, wording) => {
    const glue = fakeGlue(['csv'], wording);
    const refusal = await installPythonPackage(glue);
    expect(refusal).toBe(wording('csv', PACKAGE_NAME));
    expect(glue.seen.length).toBe(2);
    expect(glue.seen[1]).toEqual({ name: PACKAGE_NAME, runtime: true, modules: [] });
    expect(glue.installed).toEqual([{ name: PACKAGE_NAME }]);
  });

  it('gives up on a refusal about anything else at the first try', async () => {
    let tries = 0;
    const other = 'addPythonPackage: a different package `oaiy-stdlib` is installed already';
    const glue = {
      pythonPackages: () => JSON.stringify({ engineAbi: 'abi0', installed: [] }),
      addPythonPackage: (): string => {
        tries += 1;
        throw other;
      },
    };
    await expect(installPythonPackage(glue)).rejects.toBe(other);
    expect(tries).toBe(1);
  });

  it('gives up when Zipp refuses even the package with no module in it, and does not try for ever', async () => {
    let tries = 0;
    const always = `addPythonPackage: module \`csv\` of package \`${PACKAGE_NAME}\` ${TAIL}`;
    const glue = {
      pythonPackages: () => JSON.stringify({ engineAbi: 'abi0', installed: [] }),
      addPythonPackage: (): string => {
        tries += 1;
        throw always;
      },
    };
    await expect(installPythonPackage(glue)).rejects.toBe(always);
    // csv left out, then every module left out, then nothing more to leave out.
    expect(tries).toBe(3);
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// Zipp's engine, in this process: its own words for the refusal
// ---------------------------------------------------------------------------------------------------------------------

// (by a name in a variable: Node's fs has no types in this project, and Zipp's script is not one of its modules)
const load = (specifier: string): Promise<unknown> => import(/* @vite-ignore */ specifier);
const fs = (await load('node:fs')) as { existsSync(path: URL): boolean; readFileSync(path: URL): Uint8Array };
const ZIPP = new URL('../../public/zipp/', import.meta.url);
const HAVE_ZIPP = fs.existsSync(new URL('zipp_wasm_bg.wasm', ZIPP));

interface ZippGlue {
  initSync(module: { module: Uint8Array }): unknown;
  addPythonPackage(archive: Uint8Array, kernels: unknown): string;
  pythonPackages(): string;
}

/** Another package, in the format package.ts describes, whose modules are one line each. */
async function otherPackage(name: string, engineAbi: string, modules: string[]): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];
  let offset = 0;
  const add = async (text: string) => {
    const bytes = encoder.encode(text);
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
    const entry = `${offset} ${bytes.byteLength} ${Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('')}`;
    parts.push(bytes);
    offset += bytes.byteLength;
    return entry;
  };
  const manifest = ['zipp-python-package 1', `name ${name}`, 'version 1', `engine-abi ${engineAbi}`, 'kernels 0'];
  manifest.push(`runtime runtime.js ${await add('// nothing\n')}`);
  for (const m of modules) manifest.push(`module ${m} ${m}.py ${await add(`NAME = '${m}'\n`)} -`);
  manifest.push('end');
  const head = encoder.encode(`${manifest.join('\n')}\n`);
  const out = new Uint8Array(head.byteLength + offset);
  out.set(head, 0);
  let at = head.byteLength;
  for (const p of parts) {
    out.set(p, at);
    at += p.byteLength;
  }
  return out;
}

describe.skipIf(!HAVE_ZIPP)("Zipp's own refusal of a module it already has", () => {
  it('names the module in words the installer reads, and the package goes in without the modules another package brought', async () => {
    const glue = (await load(new URL('zipp_wasm.js', ZIPP).href)) as ZippGlue;
    glue.initSync({ module: fs.readFileSync(new URL('zipp_wasm_bg.wasm', ZIPP)) });
    const info = JSON.parse(glue.pythonPackages()) as { engineAbi: string; installed: Array<{ name: string }> };
    expect(info.installed).toEqual([]);
    glue.addPythonPackage(await otherPackage('someone-elses', info.engineAbi, ['uuid', 'csv']), null);

    // The sentence itself, from the engine: a package that brings a module the first one brought.
    let refusal = '';
    try {
      glue.addPythonPackage(await otherPackage('a-third', info.engineAbi, ['csv']), null);
    } catch (error) {
      refusal = error instanceof Error ? error.message : String(error);
    }
    expect(refusal).toContain(TAIL);
    expect(providedModule(refusal)).toBe('csv');

    const tried: number[] = [];
    const result = await installPythonPackage({
      pythonPackages: () => glue.pythonPackages(),
      addPythonPackage: (archive, kernels) => {
        tried.push(manifestOf(archive).modules.length);
        return glue.addPythonPackage(archive, kernels);
      },
    });
    // Nothing to report: each module was named and left out, and the rest went in.
    expect(result).toBeUndefined();
    expect(tried).toEqual([ALL.length, ALL.length - 1, ALL.length - 2]);
    const after = JSON.parse(glue.pythonPackages()) as { installed: Array<{ name: string; modules?: number }> };
    const ours = after.installed.find((p) => p.name === PACKAGE_NAME);
    expect(ours).toBeDefined();
    if (typeof ours?.modules === 'number') expect(ours.modules).toBe(ALL.length - 2);
  });
});
