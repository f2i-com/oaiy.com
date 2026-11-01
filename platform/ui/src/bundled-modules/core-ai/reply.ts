/**
 * What a chat reply says: its answer, or why it has none.
 *
 * On its own so that everything that takes an answer from a model goes by the same checks: the AI LLM node here, and
 * the connector's chat, a linked provider's AI step on this desktop (connector-module/runtime.ts).
 */

/**
 * Strip reasoning-model `<think>...</think>` (and `<thinking>...</thinking>`)
 * blocks from a chat completion before handing it back to the workflow.
 *
 * Why this matters: Qwen 3.5 thinking variants, DeepSeek R1, and the
 * various Reasoner-style local + cloud LLMs interleave their chain of
 * thought into the same response stream as the final answer. Without
 * stripping, the user-visible text starts with several paragraphs of
 * scratchpad ("Let me think about this... Actually, the user wants...")
 * before the actual answer — which then propagates downstream into any
 * node that consumes the AI output (Save File, another LLM, a diffusion
 * prompt, etc.) and produces nonsense results.
 *
 * Cleanup:
 *   - Removes well-formed `<think>...</think>` / `<thinking>...</thinking>`
 *     blocks. Case-insensitive, multiline-safe, non-greedy so multiple
 *     blocks in the same response each get stripped.
 *   - If an opening tag has no matching close (model hit max_tokens
 *     mid-thought), drops everything from that tag to the end — the
 *     half-finished reasoning isn't useful as an answer.
 *   - Trims leading whitespace so the visible text starts on a real
 *     character rather than the gap the stripped block used to occupy.
 *
 * Applied at every return path of `chat()` that produces user-visible
 * text. Models that don't emit think tags are unaffected (the regex
 * just doesn't match anything).
 */
export function stripThinkTags(text: string): string {
  if (!text) return text;
  let out = text.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '');
  // Trailing unclosed-thought drop. `i` flag handles either case mix.
  const openTail = out.match(/<think(?:ing)?>[\s\S]*$/i);
  if (openTail && typeof openTail.index === 'number') {
    out = out.slice(0, openTail.index);
  }
  return out.replace(/^\s+/, '');
}

export const NO_MESSAGE = 'The AI service answered without a message.';
const LIMIT_SPENT_THINKING = 'The model reached its token limit before it answered: its thinking used it up. Raise Max Tokens.';

/**
 * `stripThinkTags`, for a reply that has to hold an answer: one that was thought and nothing else fails the step. Empty
 * text would go on as the step's answer, into whatever the flow does with it next.
 */
export function answerOf(text: string): string {
  const answer = stripThinkTags(text);
  if (!answer && /<think(?:ing)?>/i.test(text)) {
    throw new Error(/<\/think(?:ing)?>/i.test(text) ? NO_MESSAGE : LIMIT_SPENT_THINKING);
  }
  return answer;
}

/**
 * The `error` a provider's 200 carries, if it does: of the reply, or of its first choice as some gateways put it. An
 * empty one says nothing: a gateway's `error: {}` beside a good answer is no error.
 */
export function errorIn(data: any): unknown {
  const said = (error: unknown) => (error && typeof error === 'object' ? Object.keys(error).length > 0 : !!error);
  return [data?.error, data?.choices?.[0]?.error].find(said);
}

/**
 * A reply that is the answer without being text: a turn that calls tools (no text, `tool_calls`, or Anthropic's
 * `tool_use` blocks), or an image (no text, `images`). The flow reads its JSON. Text it does not have is not a missing
 * answer. A finish that says tools were called, and no call behind it, is nothing to read.
 */
export function holdsOtherThanText(data: any): boolean {
  const choice = data?.choices?.[0];
  const some = (list: unknown) => Array.isArray(list) && list.length > 0;
  const toolBlocks = Array.isArray(data?.content) && data.content.some((b: { type?: unknown }) => b?.type === 'tool_use');
  return some(choice?.message?.tool_calls) || toolBlocks || some(choice?.message?.images);
}

/**
 * Whether what stands beside such a reply's calls or pictures says nothing: no text, or only thinking or white space. A
 * reasoning model's turn that calls a tool still has its thinking in `content`, and is that turn all the same.
 */
export function turnWithoutText(text: unknown, data: any): boolean {
  return holdsOtherThanText(data) && (!text || (typeof text === 'string' && !stripThinkTags(text)));
}

/**
 * A list of `choices` that is a reply's choices. A custom endpoint may have a `choices` of its own: words, numbers, or
 * objects of its own (`{label, score}`), and a plain completions endpoint's are `{text}`. Those are the flow's to read.
 */
export function holdsChoices(data: any): boolean {
  if (!Array.isArray(data?.choices)) return false;
  if (data.choices.length === 0) return true;
  const first = data.choices[0];
  return !!first && typeof first === 'object' && ('message' in first || ('delta' in first && 'finish_reason' in first));
}

/** The reply's text, whatever the shape: by what it is, since the address says nothing (a proxy with `claude` in its URL speaks OpenAI). */
export function replyText(data: any): string {
  const said = data?.choices?.[0]?.message?.content || data?.message?.content || data?.response;
  if (said) return said;
  // (every text block, in order: the answer can come in several, after a thinking block that has no text)
  const texts: string[] = Array.isArray(data?.content)
    ? data.content.filter((b: { type?: unknown; text?: unknown }) => b?.type === 'text' && typeof b.text === 'string').map((b: { text: string }) => b.text)
    : [];
  return texts.join('');
}

/** `NO_MESSAGE`, with how the reply says it ended where it does. */
export function noMessage(data: any): string {
  const finish = data?.choices?.[0]?.finish_reason;
  if (finish) return `The AI service answered without a message (finish_reason: ${finish}).`;
  return data?.stop_reason ? `The AI service answered without a message (stop_reason: ${data.stop_reason}).` : NO_MESSAGE;
}

/**
 * What a provider's 200 says about why it holds no answer, where it says. A status of 200 is not an answer: a body with
 * an `error` in it, a refusal, or a reasoning model's that used its tokens thinking, otherwise reached the flow as text.
 */
export function whyNoAnswer(data: any): string | null {
  const failure: any = errorIn(data);
  if (failure) {
    const said = typeof failure === 'string' ? failure : typeof failure.message === 'string' ? failure.message : JSON.stringify(failure);
    return `The AI service answered with an error: ${said}`;
  }
  const choice = data?.choices?.[0];
  if (typeof choice?.message?.refusal === 'string' && choice.message.refusal) {
    return `The model refused: ${choice.message.refusal}${choice.finish_reason ? ` (finish_reason: ${choice.finish_reason})` : ''}`;
  }
  if (choice?.finish_reason === 'content_filter' || data?.stop_reason === 'refusal') return noMessage(data);
  return choice?.finish_reason === 'length' || data?.stop_reason === 'max_tokens' ? LIMIT_SPENT_THINKING : null;
}

/** How a reply ends when what it holds is whole, an `error` beside it or not. */
const WHOLE = new Set(['stop', 'tool_calls', 'function_call', 'end_turn', 'stop_sequence', 'tool_use']);

/**
 * A reply that holds the start of an answer and says it failed there is a failure, not that start: it ended in an error,
 * or was stopped (`content_filter`, a `refusal`), or carries an `error` and does not say it finished. A reply of a
 * shape of its own (a template's) has no such end to go by, and an `error` beside its answer is its own business.
 */
export function endedInError(data: any): boolean {
  const finish = data?.choices?.[0]?.finish_reason ?? data?.stop_reason;
  if (finish === 'error' || finish === 'content_filter' || finish === 'refusal') return true;
  return !!errorIn(data) && (holdsChoices(data) || !!data?.stop_reason) && !WHOLE.has(finish);
}

/**
 * The answer of a reply in a provider's usual shape, or the failure that says why there is none. The reply's own JSON
 * where it is a turn with no text: not cleaned as text, since a thought left open in it would cut the JSON short.
 */
export function answerIn(data: any): string {
  const text = replyText(data);
  if (turnWithoutText(text, data)) return JSON.stringify(data);
  if (!text || endedInError(data)) throw new Error(whyNoAnswer(data) ?? noMessage(data));
  const answer = answerOf(text);
  if (!answer) throw new Error(whyNoAnswer(data) ?? noMessage(data));
  return answer;
}
