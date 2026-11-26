/**
 * OAIY's Python package for Zipp: standard-library modules Zipp's
 * Python does not have (written in Python, in this folder) and the runtime
 * additions in runtime.js, packed in the `zipp-python-package 1` format
 * Zipp's addPythonPackage takes (see zipp-vm/src/python_packages.rs):
 *
 *   zipp-python-package 1
 *   name <name> / version <v> / engine-abi <abi> / kernels 0
 *   runtime <file> <offset> <length> <sha256>
 *   module <name> <file> <offset> <length> <sha256> <imports>
 *   end
 *   <the files' bytes; offsets count from here>
 *
 * A module's file is `<dotted name>.py`; its imports are listed as Zipp's
 * build lists its bundled modules' (`i|a.b,c` or `<level>|<module>|x,y`,
 * `;` between statements, `-` for none).
 *
 * Zipp takes a module named like one of its own without refusing it, and a
 * program then imports the package's, not Zipp's. So a module belongs here
 * only while Zipp has none, or a smaller one: `shutil` stays (Zipp v0.0.41's
 * has no which, disk_usage, make_archive or ignore_patterns), and `types`
 * went when Zipp's own had more in it (ModuleType, GenericAlias, ...).
 */
import RUNTIME from './runtime.js?raw';

const SOURCES = import.meta.glob('./*.py', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export const PACKAGE_NAME = 'oaiy-stdlib';

/** The modules: dotted name and source. */
export function packageModules(): Array<{ name: string; source: string }> {
  return Object.entries(SOURCES)
    .map(([file, source]) => ({ name: file.replace(/^.*\//, '').replace(/\.py$/, ''), source }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** A module's import statements in Zipp's encoding. */
export function importsField(source: string): string {
  const out: string[] = [];
  let inString: string | null = null;
  for (const raw of source.split('\n')) {
    // Skip the insides of triple-quoted strings (docstrings).
    const quotes: string[] = raw.match(/"""|'''/g) ?? [];
    if (inString) {
      if (quotes.includes(inString) && quotes.length % 2 === 1) inString = null;
      continue;
    }
    if (quotes.length % 2 === 1) {
      inString = quotes[0] ?? null;
      continue;
    }
    const line = raw.replace(/#.*$/, '').trim();
    let m = /^import\s+(.+)$/.exec(line);
    if (m) {
      const names = m[1].split(',').map((p) => p.trim().split(/\s+/)[0]).filter(Boolean);
      if (names.length) out.push(`i|${names.join(',')}`);
      continue;
    }
    m = /^from\s+(\.*)([\w.]*)\s+import\s+\(?(.+?)\)?$/.exec(line);
    if (m) {
      const names = m[3].split(',').map((p) => p.trim().split(/\s+/)[0]).filter(Boolean);
      out.push(`${m[1].length}|${m[2]}|${names.join(',')}`);
    }
  }
  return out.length ? out.join(';') : '-';
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** The package archive, for an engine with this ABI. */
export async function buildPackage(engineAbi: string, exclude?: ReadonlySet<string>): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];
  const manifest = ['zipp-python-package 1', `name ${PACKAGE_NAME}`, 'version 1', `engine-abi ${engineAbi}`, 'kernels 0'];
  let offset = 0;
  const add = async (bytes: Uint8Array) => {
    const at = offset;
    parts.push(bytes);
    offset += bytes.byteLength;
    return `${at} ${bytes.byteLength} ${await sha256(bytes)}`;
  };
  manifest.push(`runtime runtime.js ${await add(encoder.encode(RUNTIME))}`);
  for (const m of packageModules()) {
    if (exclude?.has(m.name)) continue;
    manifest.push(`module ${m.name} ${m.name}.py ${await add(encoder.encode(m.source))} ${importsField(m.source)}`);
  }
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

// Zipp v0.0.41 refuses a module it already has in these words (read from its build and from the engine itself):
//   addPythonPackage: module `unittest.mock` of package `oaiy-stdlib` is provided by the engine or another package already
// The sentence is Zipp's own and nothing promises its quoting, so the module's name is read with any marks around the
// two names, or none.
const ALREADY_PROVIDED = 'is provided by the engine or another package already';
const ALREADY_PROVIDED_RE = /module\s+[^\w.]*([\w.]+?)[^\w.]*\s+of\s+package\s[\s\S]*?is provided by the engine or another package already/;

/** The module a refusal of addPythonPackage says Zipp already has, when the refusal is that one and names it. */
export function providedModule(message: string): string | undefined {
  return ALREADY_PROVIDED_RE.exec(message)?.[1];
}

/**
 * Install OAIY's Python package into a Zipp WASM glue instance once.
 * If a newer Zipp release already provides one of the shimmed modules natively,
 * exclude that module and retry so upgrades never break Python execution.
 *
 * Resolves to nothing when the package went in whole, or without the modules
 * Zipp named. When Zipp refuses a module and the refusal names none that is
 * still in the package (its wording changed), every module is left out so that
 * the runtime additions (stdin, the working folder) still load, and the
 * refusal is what this resolves to, for the caller to show.
 */
export async function installPythonPackage(glue: {
  addPythonPackage: (archive: Uint8Array, kernels: unknown) => string;
  pythonPackages: () => string;
}): Promise<string | undefined> {
  const info = JSON.parse(glue.pythonPackages()) as { engineAbi: string; installed: Array<{ name: string }> };
  if (info.installed.some((p) => p.name === PACKAGE_NAME)) return undefined;
  const names = packageModules().map((m) => m.name);
  const excluded = new Set<string>();
  let refusal: string | undefined;
  // Every turn that does not end here leaves out a module that was still in, so there are at most names.length + 1.
  for (;;) {
    try {
      glue.addPythonPackage(await buildPackage(info.engineAbi, excluded), null);
      return refusal;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      // Another refusal, or this one when no module is left to leave out, is final.
      if (!msg.includes(ALREADY_PROVIDED) || excluded.size >= names.length) throw error;
      const named = providedModule(msg);
      if (named !== undefined && names.includes(named) && !excluded.has(named)) {
        excluded.add(named);
      } else {
        for (const name of names) excluded.add(name);
        refusal = msg;
      }
    }
  }
}
