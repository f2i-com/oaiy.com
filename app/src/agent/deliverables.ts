/**
 * Turn deliverables & workspace-changes unified diff tracker adapted from DeepSeek Harness
 * (`@deepseek-ai/dsh-workspace-changes` & `@deepseek-ai/dsh-ui-deliverables`).
 *
 * Captures a lightweight snapshot of project VFS files at the start of an agent turn, then
 * compares against the end-of-turn VFS to produce:
 *   - Exact net change kind (`created`, `modified`, `deleted`) — no-op round-trips are omitted.
 *   - Exact line-level `+added` / `-deleted` counts.
 *   - Compact 3-line-context unified diff hunks for interactive UI inspection.
 */
import { IGNORED_DIRS, parentOf, type Vfs } from '../vfs/vfs';

export type ChangeKind = 'created' | 'modified' | 'deleted';

export interface DiffLine {
  kind: 'context' | 'add' | 'del';
  text: string;
}

export interface DiffHunk {
  header: string;
  lines: DiffLine[];
}

export interface DeliverableFile {
  /** Project-relative path without leading slash (e.g. `src/app.js`). */
  path: string;
  kind: ChangeKind;
  binary: boolean;
  sizeBytes: number;
  added: number;
  deleted: number;
  hunks: DiffHunk[];
  truncatedDiff?: boolean;
}

export interface FileSnapshotEntry {
  version: number;
  size: number;
  /** Immutable VFS byte reference at snapshot time (zero-copy; VFS replaces Uint8Array on every write). */
  bytes: Uint8Array;
}

export type VfsSnapshot = Map<string, FileSnapshotEntry>;

export interface CheckpointEntry {
  /** Project-relative path without leading slash. */
  path: string;
  kind: ChangeKind;
  /** Original pre-turn bytes, or `null` when the file did not exist before the turn (was created). */
  beforeBytes: Uint8Array | null;
}

export interface TurnCheckpoint {
  id: string;
  at: number;
  label: string;
  files: CheckpointEntry[];
}

const MAX_DIFF_BYTES = 256 * 1024;
const MAX_LCS_CELLS = 220_000;
const MAX_HUNKS = 12;
const MAX_HUNK_LINES = 240;
const CONTEXT_LINES = 3;

const textDecoder = new TextDecoder('utf-8', { fatal: true });

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a === b) return true;
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isLikelyBinary(bytes: Uint8Array): boolean {
  const probe = Math.min(bytes.length, 4096);
  for (let i = 0; i < probe; i++) {
    if (bytes[i] === 0) return true;
  }
  return false;
}

function decodeSnapshotText(bytes: Uint8Array): { binary: boolean; text: string | null } {
  const binary = isLikelyBinary(bytes);
  if (binary) return { binary: true, text: null };
  if (bytes.byteLength > MAX_DIFF_BYTES) return { binary: false, text: null };
  try {
    return { binary: false, text: textDecoder.decode(bytes) };
  } catch {
    return { binary: true, text: null };
  }
}

function shouldSkipPath(relPath: string): boolean {
  const parts = relPath.split('/');
  for (let i = 0; i < parts.length - 1; i++) {
    if (IGNORED_DIRS.has(parts[i])) return true;
  }
  return parts[0] === '.oaiy' || parts[0] === '.git';
}

/** Zero-copy snapshot of all non-ignored files currently in `vfs`. */
export function snapshotVfs(vfs: Vfs): VfsSnapshot {
  const snap: VfsSnapshot = new Map();
  const walked = vfs.walk('/', { limit: 10_000 });
  for (const entry of walked.entries) {
    if (entry.type !== 'file') continue;
    if (shouldSkipPath(entry.path)) continue;
    try {
      const bytes = vfs.readBytes(`/${entry.path}`);
      snap.set(entry.path, {
        version: vfs.version(`/${entry.path}`),
        size: bytes.byteLength,
        bytes,
      });
    } catch {
      /* file vanished concurrently */
    }
  }
  return snap;
}

function splitLines(text: string): string[] {
  if (!text) return [];
  const normalized = text.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

interface RawOp {
  kind: 'context' | 'add' | 'del';
  text: string;
  oldLine: number;
  newLine: number;
}

function computeLineOps(oldLines: string[], newLines: string[]): { ops: RawOp[]; added: number; deleted: number } {
  // Trim common prefix and suffix first so large files with small edits stay O(delta).
  let prefix = 0;
  const minLen = Math.min(oldLines.length, newLines.length);
  while (prefix < minLen && oldLines[prefix] === newLines[prefix]) prefix++;

  let suffix = 0;
  while (
    suffix < oldLines.length - prefix &&
    suffix < newLines.length - prefix &&
    oldLines[oldLines.length - 1 - suffix] === newLines[newLines.length - 1 - suffix]
  ) {
    suffix++;
  }

  const a = oldLines.slice(prefix, oldLines.length - suffix);
  const b = newLines.slice(prefix, newLines.length - suffix);

  const ops: RawOp[] = [];
  for (let i = 0; i < prefix; i++) {
    ops.push({ kind: 'context', text: oldLines[i], oldLine: i + 1, newLine: i + 1 });
  }

  let added = 0;
  let deleted = 0;

  if (a.length === 0) {
    for (let j = 0; j < b.length; j++) {
      ops.push({ kind: 'add', text: b[j], oldLine: prefix, newLine: prefix + j + 1 });
      added++;
    }
  } else if (b.length === 0) {
    for (let i = 0; i < a.length; i++) {
      ops.push({ kind: 'del', text: a[i], oldLine: prefix + i + 1, newLine: prefix });
      deleted++;
    }
  } else if (a.length * b.length <= MAX_LCS_CELLS) {
    // Exact Myers/LCS DP on the changed middle region.
    const m = a.length;
    const n = b.length;
    const cols = n + 1;
    const dp = new Uint32Array((m + 1) * cols);
    for (let i = m - 1; i >= 0; i--) {
      const row = i * cols;
      const next = (i + 1) * cols;
      for (let j = n - 1; j >= 0; j--) {
        dp[row + j] = a[i] === b[j] ? dp[next + j + 1] + 1 : Math.max(dp[next + j], dp[row + j + 1]);
      }
    }
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (a[i] === b[j]) {
        ops.push({ kind: 'context', text: a[i], oldLine: prefix + i + 1, newLine: prefix + j + 1 });
        i++;
        j++;
      } else if (dp[(i + 1) * cols + j] >= dp[i * cols + (j + 1)]) {
        ops.push({ kind: 'del', text: a[i], oldLine: prefix + i + 1, newLine: prefix + j });
        deleted++;
        i++;
      } else {
        ops.push({ kind: 'add', text: b[j], oldLine: prefix + i, newLine: prefix + j + 1 });
        added++;
        j++;
      }
    }
    while (i < m) {
      ops.push({ kind: 'del', text: a[i], oldLine: prefix + i + 1, newLine: prefix + j });
      deleted++;
      i++;
    }
    while (j < n) {
      ops.push({ kind: 'add', text: b[j], oldLine: prefix + i, newLine: prefix + j + 1 });
      added++;
      j++;
    }
  } else {
    // Fallback for huge middle diffs: emit whole changed block cleanly without stalling the UI thread.
    for (let i = 0; i < a.length; i++) {
      ops.push({ kind: 'del', text: a[i], oldLine: prefix + i + 1, newLine: prefix });
      deleted++;
    }
    for (let j = 0; j < b.length; j++) {
      ops.push({ kind: 'add', text: b[j], oldLine: prefix + a.length, newLine: prefix + j + 1 });
      added++;
    }
  }

  for (let k = 0; k < suffix; k++) {
    const oi = oldLines.length - suffix + k;
    const ni = newLines.length - suffix + k;
    ops.push({ kind: 'context', text: oldLines[oi], oldLine: oi + 1, newLine: ni + 1 });
  }

  return { ops, added, deleted };
}

function buildHunks(ops: RawOp[]): { hunks: DiffHunk[]; truncated: boolean } {
  const changedIndices: number[] = [];
  for (let i = 0; i < ops.length; i++) {
    if (ops[i].kind !== 'context') changedIndices.push(i);
  }
  if (!changedIndices.length) return { hunks: [], truncated: false };

  const ranges: Array<[number, number]> = [];
  for (const idx of changedIndices) {
    const start = Math.max(0, idx - CONTEXT_LINES);
    const end = Math.min(ops.length - 1, idx + CONTEXT_LINES);
    const prev = ranges[ranges.length - 1];
    if (prev && start <= prev[1] + 1) {
      prev[1] = Math.max(prev[1], end);
    } else {
      ranges.push([start, end]);
    }
  }

  const hunks: DiffHunk[] = [];
  let totalLines = 0;
  let truncated = false;

  for (const [start, end] of ranges) {
    if (hunks.length >= MAX_HUNKS || totalLines >= MAX_HUNK_LINES) {
      truncated = true;
      break;
    }
    const slice = ops.slice(start, end + 1);
    const oldCount = slice.filter((o) => o.kind !== 'add').length;
    const newCount = slice.filter((o) => o.kind !== 'del').length;
    const firstOld = slice.find((o) => o.kind !== 'add')?.oldLine ?? 0;
    const firstNew = slice.find((o) => o.kind !== 'del')?.newLine ?? 0;
    const header = `@@ -${oldCount ? firstOld : 0},${oldCount} +${newCount ? firstNew : 0},${newCount} @@`;
    const budget = MAX_HUNK_LINES - totalLines;
    const kept = slice.length > budget ? slice.slice(0, budget) : slice;
    if (kept.length < slice.length) truncated = true;
    hunks.push({
      header,
      lines: kept.map((o) => ({ kind: o.kind, text: o.text })),
    });
    totalLines += kept.length;
  }

  return { hunks, truncated };
}

/**
 * Compute net file deliverables and unified diff hunks between `before` and `after` snapshots.
 * Unchanged files are skipped in O(1) via VFS version comparison; UTF-8 decoding and LCS diff
 * only run on files whose contents actually changed.
 */
export function diffSnapshots(before: VfsSnapshot, after: VfsSnapshot): DeliverableFile[] {
  const paths = new Set<string>([...before.keys(), ...after.keys()]);
  const files: DeliverableFile[] = [];

  for (const path of [...paths].sort()) {
    const prev = before.get(path);
    const next = after.get(path);

    if (!prev && next) {
      // Created
      const decoded = decodeSnapshotText(next.bytes);
      if (decoded.binary || decoded.text === null) {
        files.push({ path, kind: 'created', binary: decoded.binary, sizeBytes: next.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const newLines = splitLines(decoded.text);
      const { ops, added, deleted } = computeLineOps([], newLines);
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'created', binary: false, sizeBytes: next.size, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    } else if (prev && !next) {
      // Deleted
      const decoded = decodeSnapshotText(prev.bytes);
      if (decoded.binary || decoded.text === null) {
        files.push({ path, kind: 'deleted', binary: decoded.binary, sizeBytes: prev.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const oldLines = splitLines(decoded.text);
      const { ops, added, deleted } = computeLineOps(oldLines, []);
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'deleted', binary: false, sizeBytes: 0, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    } else if (prev && next) {
      // Fast O(1) unchanged check via VFS version, plus exact byte equality for write-then-revert no-ops.
      if (prev.version === next.version) continue;
      if (prev.size === next.size && bytesEqual(prev.bytes, next.bytes)) continue;
      const oldDec = decodeSnapshotText(prev.bytes);
      const newDec = decodeSnapshotText(next.bytes);
      if (oldDec.binary || newDec.binary || oldDec.text === null || newDec.text === null) {
        files.push({ path, kind: 'modified', binary: oldDec.binary || newDec.binary, sizeBytes: next.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const oldLines = splitLines(oldDec.text);
      const newLines = splitLines(newDec.text);
      const { ops, added, deleted } = computeLineOps(oldLines, newLines);
      if (added === 0 && deleted === 0) continue;
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'modified', binary: false, sizeBytes: next.size, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    }
  }

  return files;
}

/**
 * Capture a compact turn-level undo checkpoint storing only the pre-turn bytes of net-changed files
 * (or `null` for files created during the turn). Returns `null` if no files changed.
 */
export function createCheckpoint(
  id: string,
  label: string,
  before: VfsSnapshot,
  after: VfsSnapshot,
): TurnCheckpoint | null {
  const paths = new Set<string>([...before.keys(), ...after.keys()]);
  const files: CheckpointEntry[] = [];
  for (const path of [...paths].sort()) {
    const prev = before.get(path);
    const next = after.get(path);
    if (!prev && next) {
      files.push({ path, kind: 'created', beforeBytes: null });
    } else if (prev && !next) {
      files.push({ path, kind: 'deleted', beforeBytes: prev.bytes });
    } else if (prev && next) {
      if (prev.version === next.version) continue;
      if (prev.size === next.size && bytesEqual(prev.bytes, next.bytes)) continue;
      files.push({ path, kind: 'modified', beforeBytes: prev.bytes });
    }
  }
  if (!files.length) return null;
  return { id, at: Date.now(), label, files };
}

/**
 * Revert a turn checkpoint on `vfs`, restoring pre-turn file contents and removing files created
 * during that turn. Updates/clears `reads` so subsequent agent edits see the restored versions.
 */
export function revertCheckpoint(
  vfs: Vfs,
  checkpoint: TurnCheckpoint,
  reads?: Map<string, number>,
): { restored: string[]; deleted: string[] } {
  const restored: string[] = [];
  const deleted: string[] = [];
  const clearReadsFor = (path: string, subtree = false) => {
    if (!reads) return;
    reads.delete(path);
    reads.delete(`${path}#partial`);
    if (subtree) {
      const prefix = `${path}/`;
      for (const k of [...reads.keys()]) {
        if (k.startsWith(prefix)) reads.delete(k);
      }
    }
  };
  const pruneEmptyParents = (path: string) => {
    let dir = parentOf(path);
    while (dir) {
      const st = vfs.stat(`/${dir}`);
      if (st?.type !== 'dir') break;
      if (vfs.list(`/${dir}`).length > 0) break;
      vfs.remove(`/${dir}`, false);
      dir = parentOf(dir);
    }
  };

  // Phase 1: remove files created during the turn first, and clean up newly empty parent directories
  // so restoring a pre-turn file that was replaced by a directory (or vice versa) does not hit EISDIR/ENOTDIR.
  for (const entry of checkpoint.files) {
    if (entry.beforeBytes !== null) continue;
    const st = vfs.stat(`/${entry.path}`);
    if (st) {
      vfs.remove(`/${entry.path}`, true);
      deleted.push(entry.path);
      pruneEmptyParents(entry.path);
    }
    clearReadsFor(entry.path, st?.type === 'dir');
  }

  // Phase 2: restore pre-turn file contents, clearing any conflicting directory at the target path
  // or conflicting file along its ancestor chain first.
  for (const entry of checkpoint.files) {
    if (entry.beforeBytes === null) continue;
    if (vfs.stat(`/${entry.path}`)?.type === 'dir') {
      vfs.remove(`/${entry.path}`, true);
      clearReadsFor(entry.path, true);
    }
    let anc = parentOf(entry.path);
    while (anc) {
      if (vfs.stat(`/${anc}`)?.type === 'file') {
        vfs.remove(`/${anc}`, true);
        clearReadsFor(anc);
      }
      anc = parentOf(anc);
    }
    vfs.writeFile(`/${entry.path}`, entry.beforeBytes, { parents: true });
    restored.push(entry.path);
    clearReadsFor(entry.path);
  }

  return { restored, deleted };
}
