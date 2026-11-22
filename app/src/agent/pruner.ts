/**
 * Stage-1 deterministic, model-free tool-result pruner adapted from DeepSeek Harness
 * (`@deepseek-ai/dsh-compaction-tool-result-pruner`).
 *
 * Runs BEFORE LLM summarization (`Agent.compact`) whenever prompt usage approaches the
 * compaction threshold:
 *   - Leaves recent turns (inside the protected tail) completely verbatim.
 *   - Replaces the middle of oversized older tool results using Unicode code-point-safe
 *     slicing (`headChars` + marker + `tailChars`), never splitting UTF-16 surrogate pairs
 *     and preserving any spill-path retrieval footer in the tail.
 *   - Strictly preserves assistant tool-call / tool-result pairing (never drops a tool result).
 */
import type { ToolResult, Turn } from './protocol';
import { codePointLength } from './spill';

export const PRUNE_MARKER = '\n\n[... older tool output pruned to save context ...]\n\n';

export interface PruneConfig {
  /** Minimum code-point length before a tool result is pruned (default 1,600). */
  thresholdChars?: number;
  /** Code points retained at the beginning of a pruned result (default 600). */
  headChars?: number;
  /** Code points retained at the end of a pruned result (default 600 — preserves spill pointers). */
  tailChars?: number;
  /** Number of most-recent turns protected from pruning (default 6). */
  protectRecentTurns?: number;
}

export const PRUNE_DEFAULTS: Required<PruneConfig> = {
  thresholdChars: 8_192,
  headChars: 4_096,
  tailChars: 1_024,
  protectRecentTurns: 6,
};

export interface PruneTurnResult {
  turns: Turn[];
  prunedResults: number;
  charsRemoved: number;
}

/**
 * Deterministically prune one string by Unicode code points, preserving head and tail.
 * Returns `null` if the text is already within `thresholdChars` or pruning would not shrink it.
 */
export function pruneToolText(text: string, config: PruneConfig = {}): string | null {
  const threshold = config.thresholdChars ?? PRUNE_DEFAULTS.thresholdChars;
  const headChars = config.headChars ?? PRUNE_DEFAULTS.headChars;
  const tailChars = config.tailChars ?? PRUNE_DEFAULTS.tailChars;

  const total = codePointLength(text);
  if (total <= threshold) return null;

  const points = Array.from(text);
  const markerLen = codePointLength(PRUNE_MARKER);
  const maxHead = Math.min(headChars, Math.max(0, threshold - markerLen));
  const maxTail = Math.min(tailChars, Math.max(0, threshold - markerLen - maxHead));

  if (maxHead + maxTail >= points.length) return null;

  const head = points.slice(0, maxHead).join('');
  const tail = maxTail > 0 ? points.slice(points.length - maxTail).join('') : '';
  const next = `${head}${PRUNE_MARKER}${tail}`;
  const after = codePointLength(next);
  if (after >= total || after > threshold) return null;
  return next;
}

/**
 * Prune oversized older tool results in `turns[viewStart .. turns.length - protectRecentTurns]`
 * without mutating the input array or breaking any assistant tool-call / tool-result pair.
 */
export function pruneOldToolResults(
  turns: readonly Turn[],
  viewStartIndex: number,
  config: PruneConfig = {},
): PruneTurnResult {
  const protect = config.protectRecentTurns ?? PRUNE_DEFAULTS.protectRecentTurns;
  const cutoff = Math.max(viewStartIndex, turns.length - protect);

  let prunedResults = 0;
  let charsRemoved = 0;
  let changed = false;

  const out = turns.map((turn, idx): Turn => {
    if (idx < viewStartIndex || idx >= cutoff || turn.role !== 'tool') return turn;
    let turnChanged = false;
    const nextResults = turn.results.map((res): ToolResult => {
      const pruned = pruneToolText(res.content, config);
      if (pruned === null) return res;
      turnChanged = true;
      prunedResults++;
      charsRemoved += res.content.length - pruned.length;
      return { ...res, content: pruned };
    });
    if (!turnChanged) return turn;
    changed = true;
    return { ...turn, results: nextResults };
  });

  return {
    turns: changed ? out : [...turns],
    prunedResults,
    charsRemoved,
  };
}
