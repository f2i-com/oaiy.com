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
  count: number;
  toolName: string;
  message: string;
}

export class RepeatToolGuard {
  private readonly thresholds: number[];
  private readonly thresholdSet: Set<number>;
  private readonly previewChars: number;
  private readonly excluded: Set<string>;
  private lastKey = '';
  private count = 0;

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

  /** Reset consecutive chain (e.g. when a new user request starts). */
  reset(): void {
    this.lastKey = '';
    this.count = 0;
  }

  /**
   * Observe one executed tool call and return an escalating reminder if a threshold was crossed.
   */
  observe(call: Pick<ToolCall, 'name' | 'input'>): RepeatReminder | null {
    if (this.excluded.has(call.name)) return null;
    const canonical = canonicalizeArgs(call.input);
    const key = `${call.name}\0${canonical}`;
    if (key === this.lastKey) {
      this.count++;
    } else {
      this.lastKey = key;
      this.count = 1;
    }
    if (!this.thresholdSet.has(this.count)) return null;
    if (this.count === this.thresholds[0]) {
      return {
        tier: 'gentle',
        count: this.count,
        toolName: call.name,
        message: `[OAIY] ${GENTLE_REMINDER}`,
      };
    }
    return {
      tier: 'detailed',
      count: this.count,
      toolName: call.name,
      message: `[OAIY] ${detailedReminder(call.name, this.count, previewArguments(canonical, this.previewChars))}`,
    };
  }
}
