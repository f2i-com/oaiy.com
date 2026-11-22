/**
 * Verbatim tool-output spill storage adapted from DeepSeek Harness (`dsh-spill` / `dsh-spill-policy`).
 *
 * Instead of silently discarding large tool outputs past the inline budget, the full verbatim output
 * is written under `/.oaiy/spill/<callId>-<tool>.txt` inside the project VFS, and the model receives
 * a surrogate-pair-safe head + tail preview together with the exact VFS path and retrieval hints
 * (`read_file` with `offset`/`limit`, or `search_file` / `grep`).
 */
import type { Vfs } from '../vfs/vfs';

export const SPILL_DIR = '/.oaiy/spill';
/** Inline character threshold above which tool output spills verbatim to VFS. */
export const SPILL_MAX_INLINE_CHARS = 16_000;
/** Inline line threshold above which tool output spills verbatim to VFS. */
export const SPILL_MAX_INLINE_LINES = 280;
/** Maximum characters retained in the inline head+tail preview when spilled. */
export const SPILL_PREVIEW_CHARS = 8_000;
/** Keep at most this many spill files per project so OPFS never bloats. */
const MAX_SPILL_FILES = 40;

/** Tools whose outputs already carry pagination or must stay intact. */
const SKIP_SPILL_TOOLS = new Set([
  'read_file',
  'softn_docs',
  'update_plan',
  'guide',
  'give_verdict',
  'find_tools',
  'ask_user_question',
  'exit_plan_mode',
]);

const encoder = new TextEncoder();

export interface SpillOutcome {
  /** Bounded inline text to return to the model (original text if not spilled). */
  text: string;
  /** Project VFS path where full verbatim output was stored, if spilled. */
  spillPath?: string;
  /** Exact UTF-8 bytes omitted from the inline preview. */
  omittedBytes: number;
  /** Exact line count of the full output. */
  totalLines: number;
}

/** Surrogate-pair-safe head or tail slice (never splits a UTF-16 code point). */
export function safeTextSlice(text: string, maxUnits: number, fromTail: boolean): string {
  if (maxUnits <= 0) return '';
  if (maxUnits >= text.length) return text;
  let cut = fromTail ? text.length - maxUnits : maxUnits;
  const prev = text.charCodeAt(cut - 1);
  const curr = text.charCodeAt(cut);
  if (prev >= 0xd800 && prev <= 0xdbff && curr >= 0xdc00 && curr <= 0xdfff) {
    cut += fromTail ? 1 : -1;
  }
  return fromTail ? text.slice(cut) : text.slice(0, cut);
}

/** Count Unicode code points (matching DeepSeek Harness's code-point measurement). */
export function codePointLength(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    n++;
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) i++;
    }
  }
  return n;
}

function sanitizeSegment(raw: string, fallback: string, maxLen = 32): string {
  const clean = raw.replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, maxLen);
  return clean || fallback;
}

function countLines(text: string): number {
  if (!text) return 0;
  let n = 1;
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++;
  return n;
}

function pruneOldSpills(vfs: Vfs): void {
  try {
    if (!vfs.exists(SPILL_DIR)) return;
    const entries = vfs.list(SPILL_DIR).filter((e) => e.type === 'file');
    if (entries.length <= MAX_SPILL_FILES) return;
    entries.sort((a, b) => a.mtimeMs - b.mtimeMs);
    const excess = entries.length - MAX_SPILL_FILES;
    for (let i = 0; i < excess; i++) {
      try {
        vfs.remove(`${SPILL_DIR}/${entries[i].name}`);
      } catch {
        /* ignore cleanup race */
      }
    }
  } catch {
    /* ignore cleanup failure */
  }
}

export interface SpillOptions {
  maxInlineChars?: number;
  maxInlineLines?: number;
  previewChars?: number;
}

/**
 * Bound a tool output with recoverable verbatim spill storage in VFS.
 * Returns the original text untouched if it is within budget, excluded, or if VFS write fails.
 */
export function maybeSpillOutput(
  vfs: Vfs,
  toolName: string,
  callId: string,
  content: string,
  options: SpillOptions = {},
): SpillOutcome {
  const totalLines = countLines(content);
  const maxChars = options.maxInlineChars ?? SPILL_MAX_INLINE_CHARS;
  const maxLines = options.maxInlineLines ?? SPILL_MAX_INLINE_LINES;
  const previewBudget = Math.min(options.previewChars ?? SPILL_PREVIEW_CHARS, maxChars);

  if (SKIP_SPILL_TOOLS.has(toolName)) {
    return { text: content, omittedBytes: 0, totalLines };
  }
  if (content.length <= maxChars && totalLines <= maxLines) {
    return { text: content, omittedBytes: 0, totalLines };
  }

  const safeId = sanitizeSegment(callId, String(Date.now()), 28);
  const safeTool = sanitizeSegment(toolName, 'tool', 28);
  const spillPath = `${SPILL_DIR}/${safeId}-${safeTool}.txt`;

  try {
    vfs.writeFile(spillPath, content, { parents: true });
    pruneOldSpills(vfs);
  } catch {
    // Storage failure must never hide valid tool output.
    return { text: content, omittedBytes: 0, totalLines };
  }

  // Keep symmetric head + tail preview bounded by both characters and lines.
  const headBudget = Math.ceil(previewBudget / 2);
  const tailBudget = Math.floor(previewBudget / 2);
  let head = safeTextSlice(content, headBudget, false);
  let tail = safeTextSlice(content, tailBudget, true);

  // Align to clean line boundaries when the preview spans multiple lines.
  const headMaxLines = Math.max(20, Math.floor(maxLines / 4));
  const tailMaxLines = Math.max(20, Math.floor(maxLines / 4));
  const headLines = head.split('\n');
  if (headLines.length > headMaxLines) head = headLines.slice(0, headMaxLines).join('\n');
  else if (head.length < content.length) {
    const nl = head.lastIndexOf('\n');
    if (nl > head.length * 0.6) head = head.slice(0, nl);
  }

  const tailLines = tail.split('\n');
  if (tailLines.length > tailMaxLines) tail = tailLines.slice(-tailMaxLines).join('\n');
  else if (tail.length < content.length) {
    const nl = tail.indexOf('\n');
    if (nl >= 0 && nl < tail.length * 0.4) tail = tail.slice(nl + 1);
  }

  // Avoid overlap if content was line-heavy but char-short.
  if (head.length + tail.length >= content.length) {
    const half = Math.floor(content.length * 0.35);
    head = safeTextSlice(content, half, false);
    tail = safeTextSlice(content, half, true);
  }

  const totalBytes = encoder.encode(content).byteLength;
  const keptBytes = encoder.encode(head).byteLength + encoder.encode(tail).byteLength;
  const omittedBytes = Math.max(0, totalBytes - keptBytes);
  const notice =
    `[Full ${toolName} output (${totalLines.toLocaleString()} lines, ${totalBytes.toLocaleString()} bytes; ` +
    `${omittedBytes.toLocaleString()} bytes omitted above) saved verbatim to ${spillPath}. ` +
    `Use read_file({"path": "${spillPath}", "offset": 1, "limit": 200}) or search_file({"path": "${spillPath}", "pattern": "..."}) to inspect specific parts.]`;

  const text = `${head}\n\n[...]\n\n${tail}\n\n${notice}`;
  return { text, spillPath, omittedBytes, totalLines };
}
