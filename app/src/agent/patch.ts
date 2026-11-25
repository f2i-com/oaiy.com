/**
 * Multi-file atomic patch parser and fuzzy applier, adapted from DeepSeek Harness
 * (`packages/fs/fs-patch`: Codex-style `*** Begin Patch` / `*** End Patch` grammar
 * with 4-pass whitespace & Unicode punctuation matching).
 *
 * All operations in a patch are validated and computed in memory before any VFS
 * write is performed, guaranteeing atomic all-or-nothing application.
 */
import { normalizePath, type Vfs } from '../vfs/vfs';

const BEGIN = '*** Begin Patch';
const END = '*** End Patch';
const EOF = '*** End of File';
const MOVE = '*** Move to: ';
const ADD = '*** Add File: ';
const DELETE = '*** Delete File: ';
const UPDATE = '*** Update File: ';

const WHITE_SPACE = /\p{White_Space}/u;

/** The times in a row a part that opens more brackets than it closes may come in what a tool writes ([`runawayNesting`]). */
const NESTED_RUN = 32;

function opensMore(text: string): boolean {
  let depth = 0;
  for (const c of text) depth += c === '(' || c === '[' || c === '{' ? 1 : c === ')' || c === ']' || c === '}' ? -1 : 0;
  return depth > 0;
}

/**
 * A part of `text` that nests a call or an index inside itself again and again (`dirs.indexOf(dirs.indexOf(…`).
 */
export function runawayNesting(text: string): { unit: string; times: number; line: number } | null {
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c === 40 || c === 91 || c === 123) depth++;
    else if (c === 41 || c === 93 || c === 125) depth--;
  }
  if (depth < NESTED_RUN) return null;
  for (const m of text.matchAll(new RegExp(`([^\\n]{1,32}?)\\1{${NESTED_RUN - 1},}`, 'g'))) {
    const unit = m[1]!;
    if (opensMore(unit)) return { unit, times: m[0].length / unit.length, line: text.slice(0, m.index).split('\n').length };
  }
  return null;
}

/** Refuses what a write or patch would put in a file where it nests a call in itself without end ([`runawayNesting`]). */
export function refuseRunaway(text: string): void {
  const run = runawayNesting(text);
  if (run) {
    throw new Error(`nothing was written: what you wrote repeats ${JSON.stringify(run.unit)} ${run.times} times in a row on its line ${run.line}, a call or an index written inside itself again and again, which nothing after it can close (a reply that ran away). Write that part another way: in steps, with a variable or a loop. Write a large file in parts (write_file with the first part, then append_file for each next part).`);
  }
}

function lineBigrams(line: string): Set<string> {
  const t = line.trim().toLowerCase();
  const set = new Set<string>();
  if (t.length === 1) {
    set.add(t);
    return set;
  }
  for (let i = 0; i < t.length - 1; i++) set.add(t.slice(i, i + 2));
  return set;
}

function bigramDice(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let shared = 0;
  for (const x of a) if (b.has(x)) shared++;
  return (2 * shared) / (a.size + b.size);
}

/**
 * Find the closest matching multi-line block in `lines` for `pattern` using bigram Dice similarity
 * across all non-empty pattern lines. Returns numbered verbatim lines from the file, or '' when no
 * plausible match exists.
 */
export function nearestBlockHint(lines: readonly string[], pattern: readonly string[], maxDisplayLines = 8): string {
  const nonEmpty = pattern
    .map((l, idx) => ({ idx, trimmed: l.trim(), grams: lineBigrams(l) }))
    .filter((p) => p.trimmed.length > 0);
  if (!nonEmpty.length || !lines.length) return '';

  const lineGrams = lines.map(lineBigrams);
  const winLen = Math.max(1, Math.min(pattern.length, lines.length));
  const maxStart = Math.max(0, lines.length - winLen);

  let bestStart = -1;
  let bestScore = 0;

  if (nonEmpty.length === 1) {
    const target = nonEmpty[0]!.grams;
    for (let i = 0; i < lines.length; i++) {
      const s = bigramDice(lineGrams[i]!, target);
      if (s > bestScore) {
        bestScore = s;
        bestStart = i;
      }
    }
  } else {
    const sample = nonEmpty.slice(0, 8);
    for (let start = 0; start <= maxStart; start++) {
      let sum = 0;
      for (const p of sample) {
        const targetIdx = Math.min(lines.length - 1, start + Math.min(p.idx, winLen - 1));
        let localBest = bigramDice(lineGrams[targetIdx]!, p.grams);
        if (localBest < 0.9 && targetIdx > start) {
          localBest = Math.max(localBest, bigramDice(lineGrams[targetIdx - 1]!, p.grams));
        }
        if (localBest < 0.9 && targetIdx + 1 < start + winLen) {
          localBest = Math.max(localBest, bigramDice(lineGrams[targetIdx + 1]!, p.grams));
        }
        sum += localBest;
      }
      const avg = sum / sample.length;
      if (avg > bestScore) {
        bestScore = avg;
        bestStart = start;
      }
    }
  }

  if (bestStart < 0 || bestScore < 0.3) return '';
  const from = Math.max(0, bestStart - 1);
  const to = Math.min(lines.length, bestStart + Math.min(winLen + 1, maxDisplayLines));
  return lines
    .slice(from, to)
    .map((l, i) => `${from + i + 1}\t${l.replace(/\r$/, '')}`)
    .join('\n');
}

export interface PatchHunk {
  lineNumber: number;
  context?: string;
  oldLines: string[];
  newLines: string[];
  isEndOfFile: boolean;
}

export type PatchOperation =
  | { kind: 'add'; path: string; lineNumber: number; content: string }
  | { kind: 'delete'; path: string; lineNumber: number }
  | { kind: 'update'; path: string; moveTo?: string; lineNumber: number; hunks: PatchHunk[] };

export interface HunkMatchFailure {
  hunkLineNumber: number;
  stage: 'context' | 'oldLines';
  searchStart: number;
  pattern: string[];
}

export type ApplyHunksResult =
  | { ok: true; after: string }
  | { ok: false; failure: HunkMatchFailure };

export class PatchParseError extends Error {
  constructor(readonly lineNumber: number, readonly reason: string) {
    super(`Invalid patch at line ${lineNumber}: ${reason}`);
    this.name = 'PatchParseError';
  }
}

function trimEdge(value: string, leading: boolean): string {
  let start = 0;
  let end = value.length;
  if (leading) {
    while (start < end && WHITE_SPACE.test(value.charAt(start))) start++;
  }
  while (end > start && WHITE_SPACE.test(value.charAt(end - 1))) end--;
  return value.slice(start, end);
}

function normalizePunctuation(value: string): string {
  return trimEdge(value, true)
    .replace(/[\u2010-\u2015\u2212]/gu, '-')
    .replace(/[\u2018-\u201B]/gu, "'")
    .replace(/[\u201C-\u201F]/gu, '"')
    .replace(/[\u00A0\u2002-\u200A\u202F\u205F\u3000]/gu, ' ');
}

const MATCH_PASSES: ReadonlyArray<(value: string) => string> = [
  (value) => value,
  (value) => trimEdge(value, false),
  (value) => trimEdge(value, true),
  normalizePunctuation,
];

function seekSequence(lines: readonly string[], pattern: readonly string[], start: number, eof: boolean): number | undefined {
  if (pattern.length === 0) return start;
  if (pattern.length > lines.length) return undefined;
  const last = lines.length - pattern.length;
  if (eof && last < start) return undefined;
  const first = eof ? last : start;
  for (const normalize of MATCH_PASSES) {
    const expected = pattern.map(normalize);
    for (let index = first; index <= last; index++) {
      if (expected.every((line, offset) => normalize(lines[index + offset] as string) === line)) {
        return index;
      }
    }
  }
  return undefined;
}

interface InputLine {
  text: string;
  number: number;
}

function patchLines(patch: string): InputLine[] {
  const trimmedHead = patch.replace(/^\p{White_Space}+/u, '');
  const leading = patch.slice(0, patch.length - trimmedHead.length);
  const offset = leading.split('\n').length - 1;
  let lines = trimEdge(patch, true)
    .split(/\r?\n/u)
    .map((text, index) => ({ text, number: offset + index + 1 }));
  if (lines.length < 2) {
    throw new PatchParseError(1, `Expected '${BEGIN}' and '${END}'`);
  }
  const first = lines[0]!;
  const last = lines[lines.length - 1]!;
  if (['<<EOF', "<<'EOF'", '<<"EOF"'].includes(first.text) && last.text.endsWith('EOF') && lines.length >= 4) {
    lines = lines.slice(1, -1);
  }
  const start = lines[0]!;
  const end = lines[lines.length - 1]!;
  if (trimEdge(start.text, true) !== BEGIN) {
    throw new PatchParseError(start.number, `Expected '${BEGIN}'`);
  }
  if (trimEdge(end.text, true) !== END) {
    throw new PatchParseError(end.number, `Expected '${END}'`);
  }
  return lines.map((line, index) => (index < lines.length - 1 ? { ...line, text: line.text.replace(/\r$/u, '') } : line));
}

function emptyHunk(hunk: PatchHunk): boolean {
  return hunk.oldLines.length === 0 && hunk.newLines.length === 0;
}

function ensureUpdate(operation: PatchOperation | undefined, lineNumber: number): void {
  if (operation?.kind !== 'update') return;
  const last = operation.hunks[operation.hunks.length - 1];
  if (!last) {
    throw new PatchParseError(operation.lineNumber, `Update for '${operation.path}' requires at least one hunk`);
  }
  if (emptyHunk(last)) {
    throw new PatchParseError(lineNumber, 'Update hunk must contain context, added, or removed lines');
  }
}

function newHunk(lineNumber: number): PatchHunk {
  return { lineNumber, oldLines: [], newLines: [], isEndOfFile: false };
}

function readUpdate(operation: Extract<PatchOperation, { kind: 'update' }>, line: InputLine): void {
  const text = trimEdge(line.text, false);
  const previous = operation.hunks[operation.hunks.length - 1];
  const marker = text === '@@' || text.startsWith('@@ ');
  if (previous?.isEndOfFile) {
    if (text === '') return;
    if (!marker) throw new PatchParseError(line.number, `Expected '@@' after '${EOF}'`);
  }
  if (previous === undefined && operation.moveTo === undefined && text.startsWith(MOVE)) {
    operation.moveTo = text.slice(MOVE.length).trim();
    return;
  }
  if (marker) {
    if (previous !== undefined && emptyHunk(previous)) {
      throw new PatchParseError(line.number, 'Update hunk must contain context, added, or removed lines');
    }
    const hunk = newHunk(line.number);
    if (text.startsWith('@@ ')) hunk.context = text.slice(3);
    operation.hunks.push(hunk);
    return;
  }
  if (text === EOF) {
    if (previous !== undefined) {
      if (emptyHunk(previous)) {
        throw new PatchParseError(line.number, 'Update hunk must contain context, added, or removed lines');
      }
      previous.isEndOfFile = true;
    }
    return;
  }
  const prefix = line.text[0];
  if (line.text !== '' && prefix !== ' ' && prefix !== '+' && prefix !== '-') {
    throw new PatchParseError(line.number, "Expected a context (' '), added ('+'), removed ('-') line, or '@@'");
  }
  const hunk = previous ?? newHunk(line.number);
  if (previous === undefined) operation.hunks.push(hunk);
  const content = line.text.slice(1);
  if (prefix === '+') hunk.newLines.push(content);
  else if (prefix === '-') hunk.oldLines.push(content);
  else {
    hunk.oldLines.push(content);
    hunk.newLines.push(content);
  }
}

/** Parse a Codex-style multi-file patch without touching the filesystem. */
export function parsePatch(patch: string): PatchOperation[] {
  const lines = patchLines(patch);
  const operations: PatchOperation[] = [];
  let ended = false;
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i]!;
    const isLast = i === lines.length - 1;
    const operation = operations[operations.length - 1];
    const header = operation?.kind === 'update' && !isLast ? trimEdge(line.text, false) : trimEdge(line.text, true);
    if (ended && !isLast) {
      if (trimEdge(line.text, true) !== '') throw new PatchParseError(line.number, `Unexpected content after '${END}'`);
      continue;
    }
    if (header === END) {
      ensureUpdate(operation, line.number);
      ended = true;
    } else if (header.startsWith(ADD)) {
      ensureUpdate(operation, line.number);
      operations.push({ kind: 'add', path: header.slice(ADD.length).trim(), lineNumber: line.number, content: '' });
    } else if (header.startsWith(DELETE)) {
      ensureUpdate(operation, line.number);
      operations.push({ kind: 'delete', path: header.slice(DELETE.length).trim(), lineNumber: line.number });
    } else if (header.startsWith(UPDATE)) {
      ensureUpdate(operation, line.number);
      operations.push({ kind: 'update', path: header.slice(UPDATE.length).trim(), lineNumber: line.number, hunks: [] });
    } else if (operation?.kind === 'add' && line.text.startsWith('+')) {
      operation.content += `${line.text.slice(1)}\n`;
    } else if (operation?.kind === 'update') {
      readUpdate(operation, line);
    } else {
      throw new PatchParseError(line.number, 'Expected an Add File, Delete File, or Update File header; added content must start with +');
    }
  }
  return operations;
}

/** Apply parsed hunks to LF-normalized text with 4-pass fuzzy matching. */
export function applyHunks(before: string, hunks: readonly PatchHunk[]): ApplyHunksResult {
  const original = before.split('\n');
  if (original[original.length - 1] === '') original.pop();
  const replacements: Array<{ start: number; oldLength: number; newLines: readonly string[] }> = [];
  let lineIndex = 0;

  for (const hunk of hunks) {
    if (hunk.context !== undefined) {
      const pattern = [hunk.context];
      const found = seekSequence(original, pattern, lineIndex, false);
      if (found === undefined) {
        return { ok: false, failure: { hunkLineNumber: hunk.lineNumber, stage: 'context', searchStart: lineIndex, pattern } };
      }
      lineIndex = found + 1;
    }
    if (hunk.oldLines.length === 0) {
      const start =
        hunk.context !== undefined
          ? lineIndex
          : original[original.length - 1] === ''
            ? original.length - 1
            : original.length;
      replacements.push({ start, oldLength: 0, newLines: hunk.newLines });
      lineIndex = start;
      continue;
    }

    let pattern = hunk.oldLines;
    let newLines = hunk.newLines;
    let found = seekSequence(original, pattern, lineIndex, hunk.isEndOfFile);
    if (found === undefined && pattern[pattern.length - 1] === '') {
      pattern = pattern.slice(0, -1);
      if (newLines[newLines.length - 1] === '') newLines = newLines.slice(0, -1);
      found = seekSequence(original, pattern, lineIndex, hunk.isEndOfFile);
    }
    if (found === undefined) {
      return {
        ok: false,
        failure: { hunkLineNumber: hunk.lineNumber, stage: 'oldLines', searchStart: lineIndex, pattern: [...hunk.oldLines] },
      };
    }
    replacements.push({ start: found, oldLength: pattern.length, newLines });
    lineIndex = found + pattern.length;
  }

  replacements.sort((a, b) => a.start - b.start);
  for (let i = 1; i < replacements.length; i++) {
    const prev = replacements[i - 1]!;
    const curr = replacements[i]!;
    if (curr.start < prev.start + prev.oldLength) {
      return {
        ok: false,
        failure: {
          hunkLineNumber: hunks[Math.min(i, hunks.length - 1)]?.lineNumber ?? 1,
          stage: 'oldLines',
          searchStart: prev.start + prev.oldLength,
          pattern: [...(hunks[Math.min(i, hunks.length - 1)]?.oldLines ?? [])],
        },
      };
    }
  }
  let lines = original;
  for (let i = replacements.length - 1; i >= 0; i--) {
    const r = replacements[i]!;
    lines = [...lines.slice(0, r.start), ...r.newLines, ...lines.slice(r.start + r.oldLength)];
  }
  if (lines[lines.length - 1] !== '') lines.push('');
  return { ok: true, after: lines.join('\n') };
}

export interface AppliedPatchFile {
  kind: 'add' | 'update' | 'delete';
  path: string;
  moveTo?: string;
  addedLines: number;
  removedLines: number;
}

export interface ApplyPatchOutcome {
  files: AppliedPatchFile[];
  summary: string;
}

/**
 * Atomically validate and apply a multi-file patch to `vfs`, enforcing OAIY's
 * read-before-edit & stale-file safety checks for updates and preserving CRLF
 * line endings when present in the original file.
 */
export function applyPatchToVfs(
  patchText: string,
  vfs: Vfs,
  reads: Map<string, number>,
): ApplyPatchOutcome {
  const ops = parsePatch(patchText);
  if (!ops.length) throw new Error('patch contains no file operations');

  // Stage all mutations against a virtual overlay so repeated ops in the same patch compose
  // cleanly and any failure leaves the real VFS completely untouched.
  const overlay = new Map<string, string | null>(); // null => deleted
  const createdDirs = new Set<string>();
  const pending: Array<() => void> = [];
  const results: AppliedPatchFile[] = [];

  const noteParents = (key: string): void => {
    const parts = key.split('/');
    let ancestor = '';
    for (let i = 0; i < parts.length - 1; i++) {
      ancestor = ancestor ? `${ancestor}/${parts[i]}` : parts[i]!;
      createdDirs.add(ancestor);
    }
  };

  const hasFileAt = (key: string): boolean => {
    if (overlay.has(key)) return overlay.get(key) !== null;
    return vfs.stat(`/${key}`)?.type === 'file';
  };

  const hasDirAt = (key: string): boolean => {
    if (createdDirs.has(key)) return true;
    const prefix = `${key}/`;
    for (const [k, v] of overlay.entries()) {
      if (v !== null && k.startsWith(prefix)) return true;
    }
    const st = vfs.stat(`/${key}`);
    if (st?.type !== 'dir') return false;
    // A directory still exists if any non-deleted VFS file resides under it or it is already a dir in VFS.
    return true;
  };

  const assertWritablePath = (key: string, verb: string): void => {
    if (hasDirAt(key)) {
      throw new Error(`Cannot ${verb} /${key}: path is a directory`);
    }
    const parts = key.split('/');
    let ancestor = '';
    for (let i = 0; i < parts.length - 1; i++) {
      ancestor = ancestor ? `${ancestor}/${parts[i]}` : parts[i]!;
      if (hasFileAt(ancestor)) {
        throw new Error(`Cannot ${verb} /${key}: parent /${ancestor} is a file`);
      }
    }
  };

  const readCurrent = (key: string): string | null => {
    if (overlay.has(key)) return overlay.get(key) ?? null;
    const st = vfs.stat(`/${key}`);
    if (!st) return null;
    if (st.type === 'dir') {
      throw new Error(`Cannot patch /${key}: path is a directory`);
    }
    return vfs.readText(`/${key}`);
  };

  for (const op of ops) {
    const key = normalizePath(op.path);
    if (!key) throw new Error(`Invalid file path at patch line ${op.lineNumber}`);

    if (op.kind === 'add') {
      assertWritablePath(key, 'add');
      if (readCurrent(key) !== null) {
        throw new Error(`Cannot add /${key}: file already exists (use Update File instead)`);
      }
      refuseRunaway(op.content);
      overlay.set(key, op.content);
      noteParents(key);
      const addedLines = op.content ? op.content.replace(/\n$/, '').split('\n').length : 0;
      pending.push(() => {
        vfs.writeFile(`/${key}`, op.content, { parents: true });
        reads.set(key, vfs.version(`/${key}`));
        reads.delete(`${key}#partial`);
      });
      results.push({ kind: 'add', path: key, addedLines, removedLines: 0 });
    } else if (op.kind === 'delete') {
      const existing = readCurrent(key);
      if (existing === null) {
        throw new Error(`Cannot delete /${key}: file does not exist`);
      }
      overlay.set(key, null);
      const removedLines = existing ? existing.replace(/\r?\n$/, '').split(/\r?\n/).length : 0;
      pending.push(() => {
        vfs.remove(`/${key}`, true);
        reads.delete(key);
        reads.delete(`${key}#partial`);
      });
      results.push({ kind: 'delete', path: key, addedLines: 0, removedLines });
    } else {
      const existing = readCurrent(key);
      if (existing === null) {
        throw new Error(`Cannot update /${key}: file does not exist`);
      }
      // Enforce read-before-edit & stale check if the file existed before this patch.
      if (!overlay.has(key)) {
        const seen = reads.get(key) ?? reads.get(`${key}#partial`);
        const currentVersion = vfs.version(`/${key}`);
        if (!seen) {
          throw new Error(`read /${key} with read_file before updating it with apply_patch`);
        }
        if (seen !== currentVersion) {
          throw new Error(`/${key} changed since you read it; call read_file again before patching`);
        }
      }
      const hadCrlf = existing.includes('\r\n');
      const normalized = hadCrlf ? existing.replace(/\r\n/g, '\n') : existing;
      const applied = applyHunks(normalized, op.hunks);
      if (!applied.ok) {
        const f = applied.failure;
        const snippet = f.pattern.slice(0, 3).join('\n');
        const origLines = normalized.split('\n');
        if (origLines[origLines.length - 1] === '') origLines.pop();
        const earlier = f.searchStart > 0 ? seekSequence(origLines, f.pattern, 0, false) : undefined;
        const orderHint =
          earlier !== undefined && earlier < f.searchStart
            ? `\nNote: those lines appear earlier at line ${earlier + 1} (before line ${f.searchStart + 1}); put hunks in top-to-bottom file order.`
            : '';
        const near = orderHint ? '' : nearestBlockHint(origLines, f.pattern);
        const nearHint = near ? `\nClosest lines in /${key}:\n${near}` : '';
        throw new Error(
          `Patch hunk at line ${f.hunkLineNumber} failed to match ${f.stage === 'context' ? 'context header' : 'target lines'} in /${key} (searched from line ${f.searchStart + 1}):\n${snippet}${orderHint}${nearHint}`,
        );
      }
      const nextText = hadCrlf ? applied.after.replace(/\n/g, '\r\n') : applied.after;
      refuseRunaway(nextText);
      let addedLines = 0;
      let removedLines = 0;
      for (const h of op.hunks) {
        // Count net non-context additions/deletions approximately via hunk line delta.
        let shared = 0;
        const oldBag = new Map<string, number>();
        for (const l of h.oldLines) oldBag.set(l, (oldBag.get(l) ?? 0) + 1);
        for (const l of h.newLines) {
          const n = oldBag.get(l) ?? 0;
          if (n > 0) {
            oldBag.set(l, n - 1);
            shared++;
          }
        }
        addedLines += h.newLines.length - shared;
        removedLines += h.oldLines.length - shared;
      }
      const destKey = op.moveTo !== undefined ? normalizePath(op.moveTo) : key;
      if (!destKey) {
        throw new Error(`Invalid Move to path at patch line ${op.lineNumber}`);
      }
      if (destKey !== key) {
        assertWritablePath(destKey, `move /${key} to`);
        if (readCurrent(destKey) !== null) {
          throw new Error(`Cannot move /${key} to /${destKey}: destination already exists`);
        }
        overlay.set(key, null);
        overlay.set(destKey, nextText);
        noteParents(destKey);
        pending.push(() => {
          const wasPartial = reads.has(`${key}#partial`);
          vfs.writeFile(`/${destKey}`, nextText, { parents: true });
          vfs.remove(`/${key}`, true);
          reads.delete(key);
          reads.delete(`${key}#partial`);
          reads.set(destKey, vfs.version(`/${destKey}`));
          if (wasPartial) reads.set(`${destKey}#partial`, 1);
          else reads.delete(`${destKey}#partial`);
        });
        results.push({ kind: 'update', path: key, moveTo: destKey, addedLines, removedLines });
      } else {
        overlay.set(key, nextText);
        pending.push(() => {
          vfs.writeFile(`/${key}`, nextText);
          reads.set(key, vfs.version(`/${key}`));
        });
        results.push({ kind: 'update', path: key, addedLines, removedLines });
      }
    }
  }

  for (const commit of pending) commit();

  const lines = results.map((f) => {
    if (f.kind === 'add') return `A /${f.path} (+${f.addedLines})`;
    if (f.kind === 'delete') return `D /${f.path} (-${f.removedLines})`;
    const target = f.moveTo ? `/${f.path} -> /${f.moveTo}` : `/${f.path}`;
    return `M ${target} (+${f.addedLines}/-${f.removedLines})`;
  });
  return {
    files: results,
    summary: `Applied patch to ${results.length} file${results.length === 1 ? '' : 's'}:\n${lines.join('\n')}`,
  };
}
