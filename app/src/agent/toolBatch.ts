/**
 * Bounded rolling parallel tool-batch scheduler with exclusive mutation barriers,
 * ported from DeepSeek Harness (`packages/core/tools/src/batch.ts`).
 *
 * Read-only / inspection tools may execute concurrently up to `maxParallel` while
 * mutating or interactive tools act as exclusive barriers. Results are ALWAYS
 * finalized and emitted in strict model-call order, and aborts drain in-flight
 * calls cleanly while synthesizing deterministic results for skipped calls.
 */
import type { ToolCall, ToolResult } from './protocol';

/** Read-only tools safe to overlap concurrently within the same assistant step. */
export const PARALLEL_SAFE_TOOLS = new Set([
  'list_files',
  'read_file',
  'file_info',
  'search_file',
  'grep',
  'glob',
  'view_image',
  'media_info',
  'softn_docs',
  'softn_components',
  'softn_examples',
  'sqlite_list_databases',
  'sqlite_schema',
  'sqlite_query',
  'web_fetch',
  'find_tools',
]);

export const DEFAULT_MAX_PARALLEL_TOOLS = 6;

export function isParallelSafeTool(name: string): boolean {
  return PARALLEL_SAFE_TOOLS.has(name);
}

/** Call-aware parallel-safety check (treats mutating options like `softn_examples.install_to` as exclusive barriers). */
export function isParallelSafeCall(call: Pick<ToolCall, 'name' | 'input'>): boolean {
  if (!PARALLEL_SAFE_TOOLS.has(call.name)) return false;
  if (call.name === 'softn_examples' && typeof call.input?.install_to === 'string' && call.input.install_to.trim().length > 0) {
    return false;
  }
  return true;
}

export interface BatchCallSpec {
  index: number;
  call: ToolCall;
  parallel: boolean;
}

export interface ExecuteBatchOptions<TOutcome> {
  /** Maximum concurrent parallel-safe calls in flight (default 6). */
  maxParallel?: number;
  signal?: AbortSignal;
  /** Execute a single call; may run concurrently for consecutive `parallel: true` specs. */
  dispatch: (spec: BatchCallSpec) => Promise<TOutcome>;
  /** Called in strict model order as soon as each call settles. */
  onCommit: (spec: BatchCallSpec, outcome: TOutcome) => Promise<void> | void;
  /** Synthesize an outcome for calls skipped after an abort. */
  onSkipped: (spec: BatchCallSpec) => TOutcome;
}

/**
 * Schedule `specs` into maximal contiguous parallel pools separated by exclusive
 * barriers. Overlapping dispatches settle into model-order commits.
 */
export async function executeToolBatch<TOutcome>(
  specs: readonly BatchCallSpec[],
  options: ExecuteBatchOptions<TOutcome>,
): Promise<{ outcomes: TOutcome[]; aborted: boolean }> {
  const maxParallel = Math.max(1, Math.floor(options.maxParallel ?? DEFAULT_MAX_PARALLEL_TOOLS));
  const outcomes: TOutcome[] = new Array(specs.length);
  let cursor = 0;
  let aborted = options.signal?.aborted === true;

  while (cursor < specs.length) {
    if (aborted) {
      for (let i = cursor; i < specs.length; i++) {
        const spec = specs[i]!;
        const skipped = options.onSkipped(spec);
        outcomes[i] = skipped;
        await options.onCommit(spec, skipped);
      }
      return { outcomes, aborted: true };
    }

    const first = specs[cursor]!;
    if (!first.parallel) {
      const result = await options.dispatch(first);
      outcomes[cursor] = result;
      await options.onCommit(first, result);
      cursor++;
      if (options.signal?.aborted) aborted = true;
      continue;
    }

    // Gather maximal contiguous parallel-safe group.
    let end = cursor + 1;
    while (end < specs.length && specs[end]!.parallel) end++;
    const group = specs.slice(cursor, end);

    const settled: Array<{ hasValue: boolean; value?: TOutcome }> = group.map(() => ({ hasValue: false }));
    const inFlight = new Map<number, Promise<number>>();
    let nextToStart = 0;
    let committed = 0;
    let firstError: unknown = undefined;
    let hasError = false;

    const commitReady = async () => {
      while (committed < group.length && settled[committed]!.hasValue) {
        const spec = group[committed]!;
        const val = settled[committed]!.value as TOutcome;
        outcomes[cursor + committed] = val;
        await options.onCommit(spec, val);
        committed++;
      }
    };

    const fillPool = () => {
      while (!aborted && !hasError && nextToStart < group.length && inFlight.size < maxParallel) {
        if (options.signal?.aborted) {
          aborted = true;
          break;
        }
        const slotIndex = nextToStart++;
        const spec = group[slotIndex]!;
        const p = options
          .dispatch(spec)
          .then(
            (val) => {
              settled[slotIndex] = { hasValue: true, value: val };
              return slotIndex;
            },
            (err) => {
              if (!hasError) {
                hasError = true;
                firstError = err;
              }
              return slotIndex;
            },
          );
        inFlight.set(slotIndex, p);
      }
    };

    fillPool();
    while (inFlight.size > 0) {
      const doneIdx = await Promise.race(inFlight.values());
      inFlight.delete(doneIdx);
      if (hasError) {
        await Promise.allSettled(inFlight.values());
        await commitReady();
        throw firstError;
      }
      await commitReady();
      if (options.signal?.aborted) aborted = true;
      fillPool();
    }
    await commitReady();

    if (aborted && nextToStart < group.length) {
      for (let i = nextToStart; i < group.length; i++) {
        const spec = group[i]!;
        const skipped = options.onSkipped(spec);
        outcomes[cursor + i] = skipped;
        await options.onCommit(spec, skipped);
      }
      cursor = end;
      continue;
    }

    cursor = end;
  }

  return { outcomes, aborted };
}

/** Standard synthetic result for calls skipped when a step is aborted mid-batch. */
export function abortedToolResult(call: ToolCall): ToolResult {
  return {
    id: call.id,
    name: call.name,
    content: 'Error [TOOL_ABORTED_BEFORE_DISPATCH]: Tool call skipped because the turn was cancelled before dispatch.',
    isError: true,
  };
}
