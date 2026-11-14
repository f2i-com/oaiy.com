/**
 * How the chat says how much the model has thought, as OAIY's engine counts
 * it (`oaiy_thinking`): while it thinks, "Thinking… 812 / 2,048 tokens"; after,
 * "What the model thought · 812 tokens". And what the thinking chip's choice
 * reaches of the model.
 */
import type { ThinksAs } from '../../agent/protocol';
import type { AgentThinking } from '../../settings';

/** How far a thought has gone: tokens so far, and the most it may take (null: no limit). */
export interface ThinkingCount {
  used: number;
  budget: number | null;
}

const words = (v: number) => Math.max(0, Math.round(v)).toLocaleString('en-US');

/**
 * "812 / 2,048 tokens", or "812 tokens" with no limit. (A thought its budget ended counts the token that closed it too:
 * never more than the budget is said.)
 */
export function thinkingTokens(count: ThinkingCount): string {
  return count.budget === null ? `${words(count.used)} tokens` : `${words(Math.min(count.used, count.budget))} / ${words(count.budget)} tokens`;
}

/** The thinking box's label: while it thinks, how far it has gone of its budget; after, what it thought and how long that was. */
export function thinkingLabel(count: ThinkingCount | undefined, streaming: boolean): string {
  if (streaming) return count ? `Thinking… ${thinkingTokens(count)}` : 'Thinking…';
  if (!count) return 'What the model thought';
  return `What the model thought · ${thinkingTokens({ used: count.budget === null ? count.used : Math.min(count.used, count.budget), budget: null })}`;
}

/** The chat's status while the model thinks: "thinking… 812 / 2,048 tokens" once its engine has counted, else "thinking…". */
export function thinkingStatus(count: ThinkingCount | undefined): string {
  return count ? `thinking… ${thinkingTokens(count)}` : 'thinking…';
}

/**
 * What a click on the thinking chip says it chose, where the model runs (`reach`: what its server takes of it, OAIY's
 * engine all of it, this computer's or a paired one's; `name`, as the provider chip says it): on OAIY's engine the
 * choice and `limit` (how long a step may think: "at most 2,048 tokens"); elsewhere, honestly, how little of it reaches
 * the model.
 */
export function thinkingChoiceWords(choice: AgentThinking, reach: ThinksAs, name: string, limit: string): string {
  if (reach === 'oaiy') {
    return choice === 'on'
      ? `Thinking on: the model thinks briefly before each step from the next one on, ${limit} (Agent settings).`
      : choice === 'off'
        ? 'Thinking off: the model answers each step at once.'
        : `Thinking as OAIY's engine is set (Engines → Think by default), ${limit} where it thinks.`;
  }
  if (reach === 'template' && choice === 'off') return `Thinking off: ${name} is asked, through its chat template, to answer each step at once.`;
  if (reach === 'effort' && choice === 'off') return `Thinking off: ${name} is asked to answer each step at once.`;
  if (reach === 'lmstudio' && choice === 'off') return `Thinking off: ${name} is asked to answer each step at once, but LM Studio may not take that from a request. If it still thinks, turn thinking off for the model in LM Studio.`;
  return `Thinking ${choice}: ${thinkingReachNote(reach, name)}`;
}

/** How much of the chip's choice reaches the model, where not all of it does ('' on OAIY's engine). */
export function thinkingReachNote(reach: ThinksAs, name: string): string {
  if (reach === 'oaiy') return '';
  if (reach === 'template' || reach === 'effort') return `${name} is not OAIY's engine, and is asked only to turn thinking off: otherwise it thinks as it is set.`;
  if (reach === 'lmstudio') return `${name} is not OAIY's engine, and is asked only to turn thinking off, which LM Studio may not take from a request: otherwise it thinks as it is set in LM Studio.`;
  if (reach === 'none') return `${name} is not asked (OAIY's engine is, and another server on this computer only to turn thinking off): it thinks as it is set.`;
  return `OAIY Desktop does not say what kind of server ${name} is, so it is not asked: it thinks as it is set.`;
}
