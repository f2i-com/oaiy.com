/**
 * Escalating repeat-tool loop guard adapted from DeepSeek Harness (`@deepseek-ai/dsh-repeat-tool-reminder`).
 *
 * Tracks consecutive identical tool calls using deep key-sorted canonical JSON argument serialization
 * so reordered JSON properties still match identically. Injects escalating advisory reminders at
 * configurable thresholds (`[3, 5, 8]` by default):
 *   - First threshold (`3`): concise gentle reminder to analyze the previous result and vary approach.
 *   - Later thresholds (`5`, `8`): detailed reminder quoting tool name, consecutive count, and
 *     bounded (500-char) canonical argument preview.
 */
import type { ToolCall } from './protocol';

export interface RepeatGuardConfig {
  /** Consecutive-repeat counts that trigger escalating reminders (default `[3, 5, 8]`). */
  thresholds?: number[];
  /** Maximum characters of canonical arguments quoted in the detailed reminder (default 500). */
  argumentsPreviewChars?: number;
  /** Tool names transparent to the chain (neither increment nor reset). */
  exclude?: string[];
}

export const REPEAT_GUARD_DEFAULTS = {
  thresholds: [3, 5, 8] as const,
  argumentsPreviewChars: 500,
};

const GENTLE_REMINDER =
  'You are repeating the exact same tool call with identical arguments. ' +
  'Carefully analyze the previous result before calling again: if the task is ' +
  'not complete, try a different approach or different arguments instead of ' +
  'repeating the call.';

function detailedReminder(toolName: string, count: number, canonicalPreview: string): string {
  return (
    'Repeated tool call detected:\n' +
    `- tool: ${toolName}\n` +
    `- consecutive_calls: ${count}\n` +
    `- arguments: ${canonicalPreview}\n` +
    'The repeated calls are not making progress. Do not call this tool with ' +
    'these exact arguments again. Inspect the latest result and choose a ' +
    'different action, different arguments, or finish the task if enough ' +
    'evidence has been gathered.'
  );
}

/** Deep key-sort of parsed JSON so objects differing only in key order canonicalize identically. */
export function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonValue);
  if (value !== null && typeof value === 'object') {
    const rec = value as Record<string, unknown>;
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(rec).sort()) {
      sorted[key] = sortJsonValue(rec[key]);
    }
    return sorted;
  }
  return value;
}

/** Canonical string form of a tool call's arguments. */
export function canonicalizeArgs(args: unknown): string {
  try {
    return JSON.stringify(sortJsonValue(args));
  } catch {
    return String(args);
  }
}

function previewArguments(canonical: string, cap: number): string {
  if (canonical.length <= cap) return canonical;
  return `${canonical.slice(0, cap)}… (+${canonical.length - cap} more chars)`;
}

export interface RepeatReminder {
  tier: 'gentle' | 'detailed';
  kind?: 'consecutive' | 'cycle' | 'error-repeat';
  hardStop?: boolean;
  count: number;
  toolName: string;
  message: string;
}

const READ_ONLY_OBSERVE_TOOLS = new Set([
  'read_file',
  'list_files',
  'file_info',
  'search_file',
  'grep',
  'glob',
  'view_image',
  'softn_docs',
  'softn_components',
  'softn_inspect',
  'page_inspect',
  'sqlite_list_databases',
  'sqlite_schema',
  'sqlite_query',
]);

const RECENT_RING_LIMIT = 15;

function normalizeErrorSig(text: string): string {
  return text
    .slice(0, 240)
    .replace(/\d+/g, '#')
    .replace(/\s+/g, ' ')
    .trim();
}

export class RepeatToolGuard {
  private readonly thresholds: number[];
  private readonly thresholdSet: Set<number>;
  private readonly previewChars: number;
  private readonly excluded: Set<string>;
  private lastKey = '';
  private count = 0;
  private recent: Array<{ key: string; name: string; canonical: string }> = [];
  private errorStreaks = new Map<string, number>();

  constructor(config: RepeatGuardConfig = {}) {
    const raw = config.thresholds ?? [...REPEAT_GUARD_DEFAULTS.thresholds];
    if (raw.length === 0) throw new Error('RepeatToolGuard: thresholds must not be empty');
    for (const v of raw) {
      if (!Number.isInteger(v) || v < 2) {
        throw new Error(`RepeatToolGuard: invalid threshold ${v} (must be integer >= 2)`);
      }
    }
    if (new Set(raw).size !== raw.length) {
      throw new Error('RepeatToolGuard: thresholds must not contain duplicates');
    }
    this.thresholds = [...raw].sort((a, b) => a - b);
    this.thresholdSet = new Set(this.thresholds);
    this.previewChars = config.argumentsPreviewChars ?? REPEAT_GUARD_DEFAULTS.argumentsPreviewChars;
    this.excluded = new Set(config.exclude ?? ['update_plan']);
  }

  /** Reset consecutive chain, cycle ring buffer, and error streaks (e.g. when a new user request starts). */
  reset(): void {
    this.lastKey = '';
    this.count = 0;
    this.recent = [];
    this.errorStreaks.clear();
  }

  /**
   * Observe one executed tool call and return an escalating reminder if a repeat or oscillating cycle threshold was crossed.
   */
  observe(call: Pick<ToolCall, 'name' | 'input'>, outcome?: { isError?: boolean; errorText?: string }): RepeatReminder | null {
    if (this.excluded.has(call.name)) return null;
    const canonical = canonicalizeArgs(call.input);
    const key = `${call.name}\0${canonical}`;
    if (key === this.lastKey) {
      this.count++;
    } else {
      this.lastKey = key;
      this.count = 1;
    }

    this.recent.push({ key, name: call.name, canonical });
    if (this.recent.length > RECENT_RING_LIMIT) this.recent.shift();

    // 1. Exact consecutive identical calls (`A, A, A, ...`).
    if (this.thresholdSet.has(this.count)) {
      if (this.count === this.thresholds[0]) {
        return {
          tier: 'gentle',
          kind: 'consecutive',
          count: this.count,
          toolName: call.name,
          message: `[OAIY] ${GENTLE_REMINDER}`,
        };
      }
      return {
        tier: 'detailed',
        kind: 'consecutive',
        count: this.count,
        toolName: call.name,
        message: `[OAIY] ${detailedReminder(call.name, this.count, previewArguments(canonical, this.previewChars))}`,
      };
    }

    // 2. Repeated identical failing call across intervening read-only inspection calls (e.g. edit_file fail -> read_file -> edit_file fail).
    if (outcome?.isError) {
      const errKey = `${key}\0${normalizeErrorSig(outcome.errorText ?? '')}`;
      const errCount = (this.errorStreaks.get(errKey) ?? 0) + 1;
      this.errorStreaks.set(errKey, errCount);
      if (errCount === 3 || errCount >= 5) {
        return {
          tier: 'detailed',
          kind: 'error-repeat',
          hardStop: errCount >= 5,
          count: errCount,
          toolName: call.name,
          message:
            `[OAIY] Repeated failing tool call detected across steps:\n` +
            `- tool: ${call.name}\n` +
            `- failed_repeats: ${errCount}\n` +
            `- arguments: ${previewArguments(canonical, this.previewChars)}\n` +
            `The same call keeps failing with the same error. Stop repeating it and change your approach or arguments based on the error message.`,
        };
      }
    } else if (outcome && !outcome.isError && !READ_ONLY_OBSERVE_TOOLS.has(call.name)) {
      // A non-read-only tool succeeded — clear prior error streaks so fresh work starts clean.
      this.errorStreaks.clear();
    }

    // 3. Oscillating Period-2 (`ABABAB`) and Period-3 (`ABCABCABC`) tool call cycles.
    for (const period of [2, 3] as const) {
      const n = this.recent.length;
      if (n < period * 3) continue;
      const tail = this.recent.slice(n - period);
      // Skip degenerate all-equal tails (already handled by consecutive detector).
      if (new Set(tail.map((e) => e.key)).size < 2) continue;
      let cycles = 1;
      while (n - (cycles + 1) * period >= 0) {
        const start = n - (cycles + 1) * period;
        let match = true;
        for (let i = 0; i < period; i++) {
          if (this.recent[start + i]!.key !== tail[i]!.key) {
            match = false;
            break;
          }
        }
        if (!match) break;
        cycles++;
      }
      const hardThreshold = period === 2 ? 5 : 4;
      if (cycles === 3 || cycles >= hardThreshold) {
        const seq = tail.map((e) => e.name).join(' -> ');
        return {
          tier: 'detailed',
          kind: 'cycle',
          hardStop: cycles >= hardThreshold,
          count: cycles,
          toolName: call.name,
          message:
            `[OAIY] Oscillating tool loop detected (${cycles} repeated cycles of: ${seq}):\n` +
            `You are alternating between the exact same tool calls without making progress. ` +
            `Break the cycle: change your arguments, inspect the underlying cause differently, or report what is blocking you.`,
        };
      }
    }

    return null;
  }
}
