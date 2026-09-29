#!/usr/bin/env node
// Prove that every path the GitHub workflows name exists in the tree.
//
//   node platform/scripts/check-workflow-paths.mjs [--root <repository>] [<workflow.yml> ...]
//
// With no workflow named, every file in .github/workflows is read. The repository
// is the one this script is in (platform/scripts/../..) unless --root says other.
//
// The workflows moved once already (platform/.github → .github, with every path in
// them written for a repository whose root was platform/), and nothing failed
// until a release was tagged: a wrong path in a `run:` or a `working-directory:`
// is only found when the step runs. This reads them instead, and checks:
//
//   working-directory:            (a step's) a folder, from the repository's root
//   run:                          the paths a script names, from the working
//                                 directory in effect: the step's, then any `cd`
//   with: path, cache-dependency-path, workspaces, files, body_path
//                                 from the root, as the actions resolve them
//   uses: ./...                   a local workflow or action
//
// A path passes when it exists, or when the build makes it: release/, artifacts/
// and the like at the root, and anything under a dist, target or node_modules (and
// the staged resources and installed engines, GENERATED below). Even then its
// folder has to exist: platform/ui/dist/x passes, ui/dist/x (the old root's
// spelling) does not. What is read as a path in a script is a word with a slash
// that names something in the tree or ends like a file of it, the argument of a
// `cd` or `tar -C`, and the script `node` runs; words that hold a variable, URLs,
// heredoc bodies and comments are left alone. What a command substitution
// `$( … )` runs is read as a line of its own, in the folder in effect where it
// stands (a `cd` in it stays in it): `resolved="$(node platform/scripts/x.mjs)"`
// names a script, and the ZIPP release every job waits for is resolved in one. A
// pair of backticks is not read: in a `node -e` script it is a template literal.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const posix = path.posix;
const DYNAMIC = '\u0001';
/** As many DYNAMIC as `text` has characters: what is not read is blanked to the same length, so a position in the blanked line is a position in the line. */
const blank = (text) => DYNAMIC.repeat(text.length);

/** `with:` inputs that name paths, and are read from the repository's root whatever the step's working directory. */
export const WITH_PATHS = ['path', 'cache-dependency-path', 'workspaces', 'files', 'body_path'];

/** Made in the checkout's root by the jobs themselves: what they collect, download, unpack, sign and describe. */
export const SCRATCH = ['release', 'artifacts', 'signatures', 'headless-dist', 'headless-verified', 'notes.md', 'cli-asset-summary.json'];

/** Named in any folder: build output and installed dependencies. */
export const BUILD_FOLDERS = ['dist', 'target', 'node_modules'];

/** Made by a script of the repository (from the root): the ZIPP engines fetch-zipp-release.mjs installs, and what sync-cli.mjs and stage-pages.mjs stage. */
export const GENERATED = [
  'platform/ui/vendor/zipp-wasm',
  'platform/ui/vendor/zipp-wasm-python',
  'platform/.zipp-release',
  'platform/desktop/src-tauri/resources/cli',
  'platform/desktop/src-tauri/resources/app',
  'platform/desktop/src-tauri/resources/flows',
];

/** A word with a slash is a path in a script when it ends like one of the repository's files. */
const FILE_LIKE = /\.(?:mjs|cjs|js|ts|tsx|json|ya?ml|md|toml|lock|sh|ps1)$/i;

const isDir = (target) => {
  try {
    return fs.statSync(target).isDirectory();
  } catch {
    return false;
  }
};
const exists = (target) => fs.existsSync(target);

/** Whether `index` in `line` is inside a "double" or 'single' quoted string. */
function quoted(line, index) {
  let quote = null;
  for (let i = 0; i < index; i++) {
    const c = line[i];
    if (quote) {
      if (c === '\\' && quote === '"') i++;
      else if (c === quote) quote = null;
    } else if (c === '"' || c === "'") quote = c;
  }
  return quote !== null;
}

/**
 * The index of the `)` that closes a `(` whose text starts at `from` (a `$(` or a
 * bare one); quoted strings and parentheses inside it are passed over. -1 when the
 * line ends first.
 */
function closing(line, from) {
  let depth = 1;
  for (let i = from; i < line.length; i++) {
    const c = line[i];
    if (c === '\\') i++;
    else if (c === "'") {
      i = line.indexOf("'", i + 1);
      if (i < 0) return -1;
    } else if (c === '"') {
      i = endOfString(line, i + 1);
      if (i < 0) return -1;
    } else if (c === '(') depth++;
    else if (c === ')' && --depth === 0) return i;
  }
  return -1;
}

/** The index of the `"` that ends a double-quoted string whose text starts at `from`; a `$( … )` in it may hold quotes of its own. -1 when the line ends first. */
function endOfString(line, from) {
  for (let i = from; i < line.length; i++) {
    if (line[i] === '\\') i++;
    else if (line[i] === '"') return i;
    else if (line[i] === '$' && line[i + 1] === '(') {
      i = closing(line, i + 2);
      if (i < 0) return -1;
    }
  }
  return -1;
}

/**
 * The command substitutions `$( … )` in one line, the outermost of each, as
 * `{ start, end, inner }` (`end` is after the closing parenthesis). One inside
 * single quotes, or escaped, is text and not found; nor is one that does not end on
 * the line (it goes on to the next, and those lines are read as they come).
 */
export function substitutions(line) {
  const spans = [];
  let quote = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '\\' && quote !== "'") i++;
    else if (quote === "'") {
      if (c === "'") quote = null;
    } else if (c === '"') quote = quote === '"' ? null : '"';
    else if (c === "'" && quote === null) quote = "'";
    else if (c === '$' && line[i + 1] === '(') {
      const end = closing(line, i + 2);
      if (end < 0) continue;
      spans.push({ start: i, end: end + 1, inner: line.slice(i + 2, end) });
      i = end;
    }
  }
  return spans;
}

/**
 * Words with a slash in one line of a script, and the arguments of cd, tar -C and
 * node, as `{ text, explicit, at, scope }`: `at` is where it stands in the line, and
 * `scope` is '' for the line itself and names the command substitution a word is in
 * (`12` for the one at index 12, `12/5` for one inside that), which is read as a
 * line of its own. What is not written out (expansions, URLs) and what is prose (a
 * string with a space in it, an echo, a label) is left alone.
 */
export function wordsOf(rawLine) {
  let line = rawLine.replace(/\$\{\{[\s\S]*?\}\}/g, blank);
  const inner = substitutions(line);
  for (const span of inner) line = line.slice(0, span.start) + blank(line.slice(span.start, span.end)) + line.slice(span.end);
  line = line.replace(/`[^`]*`/g, blank);
  line = line.replace(/\$\{[^}]*\}/g, blank);
  line = line.replace(/\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9@#?!*$-]/g, blank);
  line = line.replace(/\b[A-Za-z][A-Za-z0-9+.-]*:\/\/[^\s"'`)>]*/g, blank);
  line = line.replace(/"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g, (string) => (/\s/.test(string) ? blank(string) : string));

  const words = [];
  const argument = (word) => word.replace(/^["']|["',;]+$/g, '');
  const command = (pattern, explicit) => {
    for (const match of line.matchAll(pattern)) {
      if (quoted(line, match.index)) continue;
      words.push({ text: argument(match[match.length - 1]), explicit, at: match.index, scope: '' });
    }
  };
  command(/(?<![\w.-])cd\s+("[^"]*"|'[^']*'|[^\s;&|()]+)/g, 'cd');
  command(/\s-C\s+("[^"]*"|'[^']*'|[^\s;&|()]+)/g, 'tar -C');
  // `node <script>`: not `node -e <code>` or `node -p <code>`, whose argument is code.
  command(/(?<![\w.-])node(?:\s+(?!-[ep]\b|--eval|--print)-[-\w=.:]+)*\s+([^\s;&|()'"`-][^\s;&|()]*)/g, 'node');
  const seen = new Set(words.map((word) => word.text));
  for (const match of line.matchAll(/(?<![\w.@+~*%-])[\w.@+~*%-]+(?:\/[\w.@+~*%-]*)+/g)) {
    if (line[match.index - 1] === DYNAMIC) continue;
    const text = match[0].replace(/[.,:;@]+$/, '');
    if (text === '' || seen.has(text)) continue;
    words.push({ text, explicit: null, at: match.index, scope: '' });
  }
  const own = words.filter((word) => word.text !== '' && !word.text.includes(DYNAMIC));
  for (const span of inner) {
    for (const word of wordsOf(span.inner)) {
      own.push({ ...word, at: span.start + 2 + word.at, scope: word.scope === '' ? String(span.start) : `${span.start}/${word.scope}` });
    }
  }
  return own;
}

/**
 * The paths a shell script names, as `{ text, explicit, cwd, line }`: `line` is the
 * 0-based line of the script, `cwd` where a relative path is taken from (the step's
 * working directory, changed by a `cd` for what follows it).
 */
export function scriptPaths(script, cwd) {
  const found = [];
  let here = cwd;
  let heredoc = null;
  const lines = script.replace(/\r\n?/g, '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const first = i;
    let text = lines[i];
    while (text.endsWith('\\') && i + 1 < lines.length) text = text.slice(0, -1) + ' ' + lines[++i].trim();
    if (heredoc !== null) {
      if (text.trim() === heredoc) heredoc = null;
      continue;
    }
    if (/^\s*#/.test(text)) continue;
    const start = /(?<!<)<<(?!<)-?\s*(["']?)([A-Za-z_]\w*)\1/.exec(text);
    if (start) heredoc = start[2];
    // Where the line, and each command substitution in it, has got to: a `cd` moves only its own,
    // and what a substitution runs starts where the one around it stands.
    const dirs = new Map([['', here]]);
    const dirOf = (scope) => {
      if (!dirs.has(scope)) dirs.set(scope, dirOf(scope.includes('/') ? scope.slice(0, scope.lastIndexOf('/')) : ''));
      return dirs.get(scope);
    };
    const words = wordsOf(start ? text.slice(0, start.index) : text).sort((a, b) => a.at - b.at);
    for (const word of words) {
      const there = dirOf(word.scope);
      found.push({ text: word.text, explicit: word.explicit, cwd: there, line: first });
      if (word.explicit === 'cd') {
        dirs.set(word.scope, posix.normalize(posix.join(there || '.', word.text)));
        // A subshell — (cd x && …) — ends with the line; a bare cd carries on. One in a substitution ends with it.
        if (word.scope === '' && !/\(\s*cd\b/.test(text)) here = dirs.get('');
      }
    }
  }
  return found;
}

/** The paths of an input that takes a list, one to a line: `{ text, index }` with the 0-based line of the value it is on (`!` lines exclude, and are not paths). */
const argumentList = (value) =>
  String(value ?? '')
    .split('\n')
    .map((item, index) => ({ text: item.trim(), index }))
    .filter(({ text }) => text !== '' && !text.startsWith('!'));

/**
 * Every path a workflow names, as `{ file, line, where, kind, text, base, explicit, folder }`:
 * `base` is the folder a relative `text` is taken from, '' being the repository's root.
 */
export function workflowPaths(file, source) {
  const entries = [];
  const document = parseYaml(source);
  for (const [job, spec] of Object.entries(document?.jobs ?? {})) {
    const defaults = spec.defaults?.run;
    if (defaults?.['working-directory']) {
      entries.push({ file, line: defaults.lines['working-directory'], where: `job ${job}`, kind: 'defaults.run.working-directory', text: defaults['working-directory'], base: '', explicit: true, folder: true });
    }
  }
  for (const step of workflowSteps(document)) {
    const where = step.step === null ? `job ${step.job}` : `job ${step.job}, step "${step.name}"`;
    // A step's own working directory replaces the job's default; both are taken from the workspace's root.
    const base = posix.normalize(step.workingDirectory ?? (step.defaultDirectory || '.')).replace(/^\.$/, '');
    const add = (entry) => entries.push({ file, where, ...entry });
    if (step.workingDirectory !== undefined && step.workingDirectory !== null) {
      add({ line: step.keyLines['working-directory'] ?? step.line, kind: 'working-directory', text: step.workingDirectory, base: '', explicit: true, folder: true });
    }
    if (typeof step.uses === 'string' && step.uses.startsWith('./')) {
      add({ line: step.keyLines.uses ?? step.line, kind: 'uses', text: step.uses.slice(2), base: '', explicit: true, folder: false });
    }
    for (const input of WITH_PATHS) {
      if (step.with[input] === undefined) continue;
      const first = step.withContent?.[input] ?? step.withLines[input] ?? step.line;
      for (const { text: item, index } of argumentList(step.with[input])) {
        // rust-cache: `<folder> -> <target folder>`.
        const text = input === 'workspaces' ? item.split(/\s+->\s+/)[0] : item;
        add({ line: first + index, kind: input, text, base: '', explicit: true, folder: input === 'workspaces' });
      }
    }
    if (typeof step.run === 'string') {
      for (const word of scriptPaths(step.run, base)) {
        add({ line: (step.runLine ?? step.line) + word.line, kind: word.explicit ? `run (${word.explicit})` : 'run', text: word.text, base: word.cwd, explicit: word.explicit !== null, folder: word.explicit === 'cd' || word.explicit === 'tar -C' });
      }
    }
  }
  return entries;
}

/** What the repository has at its top and in platform/ (and in a working directory): the names a path in a script may start with. */
function treeNames(root, base) {
  const names = new Set();
  for (const folder of ['', 'platform', base]) {
    try {
      for (const name of fs.readdirSync(path.join(root, ...folder.split('/').filter(Boolean)))) names.add(name);
    } catch {
      /* not there */
    }
  }
  return names;
}

/**
 * Whether `relative` (from the root, normalised) is made by the build, with what
 * of it has to exist: `null` when it is not, else the folder that has to be there.
 */
function madeByTheBuild(segments) {
  if (SCRATCH.includes(segments[0])) return [];
  for (let k = 0; k < segments.length; k++) {
    const head = segments.slice(0, k + 1).join('/');
    if (GENERATED.includes(head) || BUILD_FOLDERS.includes(segments[k])) return segments.slice(0, k);
  }
  return null;
}

/** `null` when the entry's path exists (or is made by the build), else what is wrong with it. */
export function checkEntry(entry, root, names = treeNames(root, entry.base)) {
  const joined = posix.normalize(posix.join(entry.base || '.', entry.text.replace(/\\/g, '/')));
  if (joined === '..' || joined.startsWith('../') || posix.isAbsolute(joined)) return `"${entry.text}" leaves the repository`;
  let segments = joined === '.' ? [] : joined.split('/');
  const glob = segments.findIndex((segment) => segment.includes('*'));
  if (glob >= 0) segments = segments.slice(0, glob);
  if (segments.length === 0) return null;
  const strict = entry.explicit || FILE_LIKE.test(segments[segments.length - 1]) || names.has(segments[0]);
  if (!strict) return null;
  const here = entry.base ? ` (from ${entry.base})` : '';
  const made = madeByTheBuild(segments);
  if (made !== null) {
    // Made by the build: only the folder it is made in has to be in the tree.
    if (made.length === 0 || isDir(path.join(root, ...made))) return null;
    return `"${entry.text}"${here} is made by the build in ${made.join('/')}, which does not exist${hint(segments, root)}`;
  }
  const target = path.join(root, ...segments);
  if (!exists(target)) return `"${entry.text}"${here} does not exist${hint(segments, root)}`;
  if (entry.folder && !isDir(target)) return `"${entry.text}"${here} is not a folder`;
  return null;
}

/** A first folder that is under platform/ and not at the root was written for a repository whose root was platform/. */
function hint(segments, root) {
  if (segments[0] !== 'platform' && !exists(path.join(root, segments[0])) && exists(path.join(root, 'platform', segments[0]))) {
    return `; platform/${segments[0]} exists: the path predates the merge (it was written for a root of platform/)`;
  }
  return '';
}

/** Check workflow files (paths) against the repository at `root`: `{ entries, problems }`. */
export function checkWorkflowFiles(files, root) {
  const entries = [];
  const problems = [];
  for (const file of files) {
    const shown = path.relative(root, file).replace(/\\/g, '/') || file;
    for (const entry of workflowPaths(shown, fs.readFileSync(file, 'utf8'))) {
      entries.push(entry);
      const message = checkEntry(entry, root);
      if (message) problems.push({ ...entry, message });
    }
  }
  return { entries, problems };
}

function main(argv) {
  const args = [...argv];
  let root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
  const at = args.indexOf('--root');
  if (at >= 0) {
    if (!args[at + 1]) {
      console.error('usage: check-workflow-paths.mjs [--root <repository>] [<workflow.yml> ...]');
      return 2;
    }
    root = path.resolve(args[at + 1]);
    args.splice(at, 2);
  }
  const folder = path.join(root, '.github', 'workflows');
  const files = args.length > 0 ? args.map((file) => path.resolve(file)) : fs.existsSync(folder) ? fs.readdirSync(folder).filter((name) => /\.ya?ml$/.test(name)).sort().map((name) => path.join(folder, name)) : [];
  if (files.length === 0) {
    console.error(`check-workflow-paths: no workflows in ${folder}`);
    return 2;
  }
  let result;
  try {
    result = checkWorkflowFiles(files, root);
  } catch (error) {
    console.error(`check-workflow-paths: ${error.message}`);
    return 2;
  }
  for (const problem of result.problems) {
    console.error(`${problem.file}:${problem.line}: ${problem.where}: ${problem.kind} ${problem.message}`);
  }
  if (result.problems.length > 0) {
    console.error(`check-workflow-paths: ${result.problems.length} of ${result.entries.length} paths in ${files.length} workflow${files.length === 1 ? '' : 's'} are wrong`);
    return 1;
  }
  console.log(`check-workflow-paths: ${result.entries.length} paths in ${files.length} workflow${files.length === 1 ? '' : 's'} (${files.map((file) => path.basename(file)).join(', ')}) all exist or are made by the build`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
