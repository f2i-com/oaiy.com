/**
 * Session tail repair & cooperative per-tool timeout policy, ported from DeepSeek Harness
 * (`packages/core/session/src/repair.ts` and `packages/guard/timeout-policy/src/index.ts`).
 *
 * - `repairDanglingToolCalls`: guarantees every assistant turn with `calls` is followed by
 *   matching tool results so interrupted/reloaded sessions never trigger 400 API errors.
 * - `withToolTimeout`: arms a scoped cooperative deadline on `AbortSignal` and converts only
 *   its own expiry into a structured `TOOL_TIMEOUT` error without racing abandoned promises.
 */
import type { ToolCall, ToolResult, Turn } from './protocol';

export const TOOL_NOT_STARTED = 'TOOL_NOT_STARTED';
export const TOOL_OUTCOME_UNKNOWN = 'TOOL_OUTCOME_UNKNOWN';
export const TOOL_TIMEOUT = 'TOOL_TIMEOUT';

export type RepairCause = 'interrupted' | 'forked';

const REPAIR_MESSAGES: Record<RepairCause, { started: string; notStarted: string }> = {
  interrupted: {
    started:
      `Error [${TOOL_OUTCOME_UNKNOWN}]: The tool call was interrupted before its result was recorded. Its outcome is unknown. ` +
      'Retry only if the operation is read-only or idempotent; if it may have side effects, inspect current state first.',
    notStarted:
      `Error [${TOOL_NOT_STARTED}]: The tool call was interrupted before it started. Retry it if it is still needed.`,
  },
  forked: {
    started:
      `Error [${TOOL_OUTCOME_UNKNOWN}]: This branch inherited a started tool call without its result. Verify current state before retrying.`,
    notStarted:
      `Error [${TOOL_NOT_STARTED}]: This branch inherited an unstarted tool call without its result. Retry it if still needed.`,
  },
};

export interface RepairReport {
  turns: Turn[];
  repairedCalls: number;
}

/**
 * Scan `turns` and supply deterministic synthetic error `ToolResult`s for any
 * assistant tool calls lacking a matching result in the immediately following
 * tool turn. Returns a new array only when repairs were needed (preserving
 * referential equality for clean histories).
 */
export function repairDanglingToolCalls(
  turns: readonly Turn[],
  cause: RepairCause = 'interrupted',
  startedCallIds?: ReadonlySet<string>,
): RepairReport {
  let repairedCalls = 0;
  let mutated = false;
  const out: Turn[] = [];

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i]!;
    out.push(turn);
    if (turn.role !== 'assistant' || !turn.calls.length) continue;

    const next = turns[i + 1];
    if (next?.role === 'tool') {
      const answered = new Set(next.results.map((r) => r.id));
      const missing = turn.calls.filter((c) => !answered.has(c.id));
      if (!missing.length) continue;
      mutated = true;
      repairedCalls += missing.length;
      const patchedResults: ToolResult[] = [
        ...next.results,
        ...missing.map((call) => makeRepairResult(call, cause, startedCallIds?.has(call.id) ?? false)),
      ];
      out.push({ role: 'tool', results: patchedResults });
      i++; // consumed next
    } else {
      mutated = true;
      repairedCalls += turn.calls.length;
      out.push({
        role: 'tool',
        results: turn.calls.map((call) => makeRepairResult(call, cause, startedCallIds?.has(call.id) ?? false)),
      });
    }
  }

  return {
    turns: mutated ? out : (turns as Turn[]),
    repairedCalls,
  };
}

function makeRepairResult(call: ToolCall, cause: RepairCause, wasStarted: boolean): ToolResult {
  const msg = REPAIR_MESSAGES[cause];
  return {
    id: call.id,
    name: call.name,
    content: wasStarted ? msg.started : msg.notStarted,
    isError: true,
  };
}

export class ToolTimeoutError extends Error {
  readonly code = TOOL_TIMEOUT;
  constructor(readonly timeoutMs: number, readonly toolName: string) {
    super(`Error [${TOOL_TIMEOUT}]: ${toolName} timed out after ${timeoutMs}ms`);
    this.name = 'ToolTimeoutError';
  }
}

/**
 * Run `fn` with a scoped cooperative deadline linked to `upstream`.
 * Only fires `ToolTimeoutError` when THIS timer (not `upstream`) caused the abort.
 */
export async function withToolTimeout<T>(
  toolName: string,
  timeoutMs: number | undefined,
  upstream: AbortSignal | undefined,
  fn: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  if (!timeoutMs || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return fn(upstream ?? new AbortController().signal);
  }
  const controller = new AbortController();
  let timedOut = false;
  const onUpstreamAbort = () => {
    controller.abort(upstream?.reason);
  };
  if (upstream) {
    if (upstream.aborted) {
      controller.abort(upstream.reason);
    } else {
      upstream.addEventListener('abort', onUpstreamAbort, { once: true });
    }
  }
  const timer = setTimeout(() => {
    if (!controller.signal.aborted) {
      timedOut = true;
      controller.abort(new ToolTimeoutError(timeoutMs, toolName));
    }
  }, timeoutMs);

  try {
    const value = await fn(controller.signal);
    if (timedOut) throw new ToolTimeoutError(timeoutMs, toolName);
    return value;
  } catch (err) {
    if (timedOut) throw new ToolTimeoutError(timeoutMs, toolName);
    throw err;
  } finally {
    clearTimeout(timer);
    upstream?.removeEventListener('abort', onUpstreamAbort);
  }
}
