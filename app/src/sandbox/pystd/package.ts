/**
 * bot.computer's Python package for Zipp: standard-library modules Zipp's
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
 */
import RUNTIME from './runtime.js?raw';

const SOURCES = import.meta.glob('./*.py', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export const PACKAGE_NAME = 'botcomputer-stdlib';

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
export async function buildPackage(engineAbi: string): Promise<Uint8Array> {
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
  for (const m of packageModules()) manifest.push(`module ${m.name} ${m.name}.py ${await add(encoder.encode(m.source))} ${importsField(m.source)}`);
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
