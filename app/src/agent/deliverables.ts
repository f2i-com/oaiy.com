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
import { IGNORED_DIRS, type Vfs } from '../vfs/vfs';

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
  size: number;
  hash: number;
  binary: boolean;
  text: string | null;
}

export type VfsSnapshot = Map<string, FileSnapshotEntry>;

const MAX_DIFF_BYTES = 256 * 1024;
const MAX_LCS_CELLS = 220_000;
const MAX_HUNKS = 12;
const MAX_HUNK_LINES = 240;
const CONTEXT_LINES = 3;

const textDecoder = new TextDecoder('utf-8', { fatal: true });

/** Fast non-cryptographic 32-bit FNV-1a hash for change detection. */
function hashBytes(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function isLikelyBinary(bytes: Uint8Array): boolean {
  const probe = Math.min(bytes.length, 4096);
  for (let i = 0; i < probe; i++) {
    if (bytes[i] === 0) return true;
  }
  return false;
}

function shouldSkipPath(relPath: string): boolean {
  const parts = relPath.split('/');
  for (let i = 0; i < parts.length - 1; i++) {
    if (IGNORED_DIRS.has(parts[i])) return true;
  }
  return parts[0] === '.oaiy' || parts[0] === '.git';
}

/** Snapshot all non-ignored files currently in `vfs`. */
export function snapshotVfs(vfs: Vfs): VfsSnapshot {
  const snap: VfsSnapshot = new Map();
  const walked = vfs.walk('/', { limit: 10_000 });
  for (const entry of walked.entries) {
    if (entry.type !== 'file') continue;
    if (shouldSkipPath(entry.path)) continue;
    try {
      const bytes = vfs.readBytes(`/${entry.path}`);
      const binary = isLikelyBinary(bytes);
      let text: string | null = null;
      if (!binary && bytes.byteLength <= MAX_DIFF_BYTES) {
        try {
          text = textDecoder.decode(bytes);
        } catch {
          text = null;
        }
      }
      snap.set(entry.path, {
        size: bytes.byteLength,
        hash: hashBytes(bytes),
        binary: binary || (text === null && bytes.byteLength <= MAX_DIFF_BYTES),
        text,
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
 */
export function diffSnapshots(before: VfsSnapshot, after: VfsSnapshot): DeliverableFile[] {
  const paths = new Set<string>([...before.keys(), ...after.keys()]);
  const files: DeliverableFile[] = [];

  for (const path of [...paths].sort()) {
    const prev = before.get(path);
    const next = after.get(path);

    if (!prev && next) {
      // Created
      if (next.binary || next.text === null) {
        files.push({ path, kind: 'created', binary: next.binary, sizeBytes: next.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const newLines = splitLines(next.text);
      const { ops, added, deleted } = computeLineOps([], newLines);
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'created', binary: false, sizeBytes: next.size, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    } else if (prev && !next) {
      // Deleted
      if (prev.binary || prev.text === null) {
        files.push({ path, kind: 'deleted', binary: prev.binary, sizeBytes: prev.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const oldLines = splitLines(prev.text);
      const { ops, added, deleted } = computeLineOps(oldLines, []);
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'deleted', binary: false, sizeBytes: 0, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    } else if (prev && next) {
      // Unchanged (including write-then-revert no-ops!)
      if (prev.size === next.size && prev.hash === next.hash && prev.text === next.text) continue;
      if (prev.binary || next.binary || prev.text === null || next.text === null) {
        files.push({ path, kind: 'modified', binary: prev.binary || next.binary, sizeBytes: next.size, added: 0, deleted: 0, hunks: [] });
        continue;
      }
      const oldLines = splitLines(prev.text);
      const newLines = splitLines(next.text);
      const { ops, added, deleted } = computeLineOps(oldLines, newLines);
      if (added === 0 && deleted === 0) continue;
      const { hunks, truncated } = buildHunks(ops);
      files.push({ path, kind: 'modified', binary: false, sizeBytes: next.size, added, deleted, hunks, ...(truncated ? { truncatedDiff: true } : {}) });
    }
  }

  return files;
}
