/*
 * Adapted from softn.com (apps/softn-studio/src/lib/agent/stream.ts),
 * Copyright f2i-com, licensed under the Apache License, Version 2.0.
 */
/**
 * Streamed replies, read as they arrive: Anthropic's Messages events and the
 * OpenAI-compatible Chat Completions chunks (OpenAI, Ollama, LM Studio and
 * other servers that speak it). Each assembler takes the server-sent events
 * one by one, tells a sink what appeared — text, a tool call's name, a
 * fragment of its arguments — and at the end hands back the reply in the
 * same shape the non-streamed API returns, so one decoder (protocol.ts)
 * reads both and a streamed reply cannot be read differently from a whole one.
 *
 * Three kinds of trouble are kept apart. An event that is not JSON is a
 * fault in the stream itself (`streamFault`): the request is worth sending
 * again without streaming. An error event is the provider's own failure,
 * reported as any other. A stream that simply stops before its end is a
 * dropped connection (`network`), which pauses the run for Resume.
 */

import { AIProviderError, isRecord, type SSEEvent } from './aiProvider';
import { CHATGPT_SIGN_IN, signInNeeded } from './chatgpt';

/** Where a streamed reply's pieces go as they arrive. */
export interface StreamSink {
  text(delta: string): void;
  /** Reasoning the model shows (a thinking block, `reasoning_content`): not part of the reply's text. */
  thinking?(delta: string): void;
  /** A tool call began: its name is known. */
  toolStart(index: number, id: string, name: string): void;
  /** A fragment of a call's JSON arguments, exactly as it came. */
  toolArgs(index: number, delta: string): void;
  /** OAIY: the tool call as the model writes it, raw (`start` begins a new one). */
  draft?(text: string, start: boolean): void;
  /** OAIY: how far the model has read the prompt (`done` of `total` tokens), `reused` read on from what it kept. */
  reading?(done: number, total: number, reused: number | null): void;
  /** OAIY: the tokens the model has thought so far, of the most it may (`budget`; null: no limit), and whether its thought is over. */
  thinkingCount?(used: number, budget: number | null, done: boolean): void;
}

function parseEvent(event: SSEEvent): Record<string, unknown> | null {
  const data = event.data.trim();
  if (data === '[DONE]') return null;
  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    throw new AIProviderError('invalid-response', `The provider's stream carried an event that is not JSON: ${data.slice(0, 80)}`, { streamFault: true });
  }
  if (!isRecord(payload)) throw new AIProviderError('invalid-response', 'The provider\'s stream carried an event that is not an object.', { streamFault: true });
  return payload;
}

/** A provider's error, reported inside a stream that had already started. */
function streamError(error: unknown): AIProviderError {
  const record = isRecord(error) ? error : {};
  const type = typeof record.type === 'string' ? record.type : typeof record.code === 'string' ? record.code : '';
  const message = typeof record.message === 'string' && record.message ? record.message : typeof error === 'string' ? error : 'The provider reported an error mid-reply.';
  if (/overloaded|rate_limit/i.test(type)) return new AIProviderError('rate-limited', `The provider is overloaded: ${message}`);
  if (signInNeeded(type)) return new AIProviderError('http', CHATGPT_SIGN_IN, { detail: message });
  // Its code as the server gives it, so the error is read as one sent before the stream: a number is the status it would
  // have answered with (an older llama.cpp's 400, for a prompt past its context), a word the code (context_length_exceeded).
  const status = typeof record.code === 'number' && Number.isInteger(record.code) ? record.code : undefined;
  const code = typeof record.code === 'string' && record.code.trim() ? record.code.trim().slice(0, 80) : undefined;
  return new AIProviderError('http', `The provider stopped its reply with an error: ${message}`, { detail: message, status, code });
}

const closedEarly = () => new AIProviderError('network', 'The connection closed before the reply was complete.');

/** Strip commas immediately preceding `}` or `]` outside quoted JSON strings. */
function stripTrailingJsonCommas(text: string): string {
  let out = '';
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      continue;
    }
    if (ch === ',') {
      let j = i + 1;
      while (j < text.length && /\s/.test(text[j]!)) j++;
      if (text[j] === '}' || text[j] === ']') continue;
    }
    out += ch;
  }
  return out;
}

/**
 * Parse a tool call's JSON arguments with safe, deterministic recovery for common local-model quirks:
 *   1. Markdown code fences wrapping a JSON object (```json ... ```)
 *   2. Trailing commas before `}` or `]` outside strings
 *   3. Single double-encoded JSON string (`"{\"path\":\"a.ts\"}"`)
 *   4. Single spurious trailing `}` (`{"path":"a.ts"}}`)
 * Truncated JSON and non-object values (arrays, primitives) are never fabricated and return a clear `parseError`.
 */
export function parseToolArgs(raw: string): { input: Record<string, unknown>; parseError?: string } {
  const trimmed = raw.trim();
  if (!trimmed) return { input: {} };

  const accept = (value: unknown, allowUnwrap: boolean): { input: Record<string, unknown>; parseError?: string } => {
    if (isRecord(value)) return { input: value };
    if (allowUnwrap && typeof value === 'string') {
      const inner = value.trim();
      if (inner.startsWith('{') && inner.endsWith('}')) {
        try {
          const unwrapped: unknown = JSON.parse(inner);
          if (isRecord(unwrapped)) return { input: unwrapped };
        } catch {
          try {
            const unwrapped: unknown = JSON.parse(stripTrailingJsonCommas(inner));
            if (isRecord(unwrapped)) return { input: unwrapped };
          } catch {
            /* fall through to non-object error */
          }
        }
      }
    }
    return { input: {}, parseError: 'arguments are not a JSON object' };
  };

  let firstError: unknown;
  try {
    return accept(JSON.parse(trimmed), true);
  } catch (err) {
    firstError = err;
  }

  // Strip optional markdown fence wrapper.
  const fenced = /^```(?:json)?\s*\r?\n?([\s\S]*?)\r?\n?```$/i.exec(trimmed);
  const candidate = (fenced ? fenced[1]! : trimmed).trim();

  const tryRecover = (text: string): { input: Record<string, unknown>; parseError?: string } | null => {
    try {
      const res = accept(JSON.parse(text), true);
      if (!res.parseError) return res;
    } catch {
      /* try next pass */
    }
    const noTrailing = stripTrailingJsonCommas(text);
    if (noTrailing !== text) {
      try {
        const res = accept(JSON.parse(noTrailing), true);
        if (!res.parseError) return res;
      } catch {
        /* try next pass */
      }
    }
    return null;
  };

  const direct = tryRecover(candidate);
  if (direct) return direct;

  // Recover a single spurious trailing brace emitted by some local quantized models (e.g. `{"a":1}}`).
  if (candidate.endsWith('}}')) {
    const trimmedBrace = tryRecover(candidate.slice(0, -1));
    if (trimmedBrace) return trimmedBrace;
  }

  const msg = firstError instanceof Error ? firstError.message : String(firstError);
  return { input: {}, parseError: `arguments are not valid JSON (${msg})` };
}

/** Upper bound on streamed block / tool-call indices so malformed or hostile streams cannot cause unbounded growth. */
const MAX_STREAM_INDEX = 4096;

// ---------------------------------------------------------------------------
// Anthropic Messages
// ---------------------------------------------------------------------------

export class AnthropicStream {
  private blocks = new Map<number, Record<string, unknown>>();
  private json = new Map<number, string>();
  private finalized = new Set<number>();
  private nextIndex = 0;
  private lastIndex = -1;
  private fallbackIdSeq = 0;
  private stopReason: string | null = null;
  private usage = { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 };
  private stopped = false;
  private delta = false;
  /** Per tool_use id, why its arguments could not be read. */
  readonly parseErrors = new Map<string, string>();

  constructor(private readonly sink?: StreamSink) {}

  private allocateIndex(): number | null {
    while (this.blocks.has(this.nextIndex) && this.nextIndex < MAX_STREAM_INDEX) this.nextIndex++;
    if (this.nextIndex >= MAX_STREAM_INDEX) return null;
    return this.nextIndex++;
  }

  private resolveIndex(raw: unknown, isStart: boolean): number | null {
    if (raw === undefined || raw === null) {
      return isStart ? this.allocateIndex() : this.lastIndex >= 0 ? this.lastIndex : null;
    }
    if (typeof raw !== 'number' || !Number.isInteger(raw)) return null;
    if (raw < 0) {
      return isStart ? this.allocateIndex() : this.lastIndex >= 0 ? this.lastIndex : null;
    }
    if (raw >= MAX_STREAM_INDEX) return null;
    return raw;
  }

  accept(event: SSEEvent): void {
    const payload = parseEvent(event);
    if (!payload) return;
    const type = typeof payload.type === 'string' ? payload.type : event.event;
    switch (type) {
      case 'message_start': {
        const message = isRecord(payload.message) ? payload.message : {};
        this.readUsage(message.usage);
        break;
      }
      case 'content_block_start': {
        const index = this.resolveIndex(payload.index, true);
        if (index === null) break;
        this.lastIndex = index;
        if (index >= this.nextIndex) this.nextIndex = index + 1;
        const block: Record<string, unknown> = isRecord(payload.content_block) ? { ...payload.content_block } : { type: 'unknown' };
        if (block.type === 'tool_use') {
          const rawId = typeof block.id === 'string' ? block.id.trim() : '';
          const id = rawId || `call_${Date.now().toString(36)}_${++this.fallbackIdSeq}_${index}`;
          block.id = id;
          block.input = {};
          this.json.set(index, '');
          this.finalized.delete(index);
          this.sink?.toolStart(index, id, typeof block.name === 'string' ? block.name : '');
        } else if (block.type === 'text') {
          block.text = typeof block.text === 'string' ? block.text : '';
          if (block.text) this.sink?.text(block.text as string);
        }
        this.blocks.set(index, block);
        break;
      }
      case 'content_block_delta': {
        const index = this.resolveIndex(payload.index, false);
        if (index === null) break;
        const block = this.blocks.get(index);
        const delta = isRecord(payload.delta) ? payload.delta : {};
        if (!block) break;
        if (delta.type === 'text_delta' && typeof delta.text === 'string') {
          block.text = `${typeof block.text === 'string' ? block.text : ''}${delta.text}`;
          this.sink?.text(delta.text);
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          this.json.set(index, (this.json.get(index) ?? '') + delta.partial_json);
          this.sink?.toolArgs(index, delta.partial_json);
        } else if (delta.type === 'thinking_delta' && typeof delta.thinking === 'string') {
          block.thinking = `${typeof block.thinking === 'string' ? block.thinking : ''}${delta.thinking}`;
          this.sink?.thinking?.(delta.thinking);
        } else if (delta.type === 'signature_delta' && typeof delta.signature === 'string') {
          block.signature = `${typeof block.signature === 'string' ? block.signature : ''}${delta.signature}`;
        }
        break;
      }
      case 'content_block_stop': {
        const index = this.resolveIndex(payload.index, false);
        if (index !== null) this.finalizeToolBlock(index);
        break;
      }
      case 'message_delta': {
        const delta = isRecord(payload.delta) ? payload.delta : {};
        if (typeof delta.stop_reason === 'string') this.stopReason = delta.stop_reason;
        this.readUsage(payload.usage);
        this.delta = true;
        break;
      }
      case 'message_stop':
        this.stopped = true;
        break;
      case 'error':
        throw streamError(payload.error);
      default:
        // ping, and event types added after this was written.
        break;
    }
  }

  private finalizeToolBlock(index: number): void {
    if (this.finalized.has(index)) return;
    this.finalized.add(index);
    const block = this.blocks.get(index);
    if (block?.type !== 'tool_use') return;
    const raw = (this.json.get(index) ?? '').trim();
    const parsed = parseToolArgs(raw);
    block.input = parsed.input;
    if (parsed.parseError) {
      this.parseErrors.set(String(block.id), `The ${parsed.parseError}. Send the call again with a JSON object.`);
    }
  }

  /** The reply as the non-streamed API returns it. */
  result(): Record<string, unknown> {
    if (!this.stopped && !this.delta) throw closedEarly();
    for (const index of this.json.keys()) this.finalizeToolBlock(index);
    const content = [...this.blocks.entries()].sort((a, b) => a[0] - b[0]).map(([, block]) => block);
    return {
      type: 'message',
      role: 'assistant',
      content,
      stop_reason: this.stopReason,
      usage: { ...this.usage },
    };
  }

  private readUsage(usage: unknown): void {
    if (!isRecord(usage)) return;
    // output_tokens in message_delta is the running total, not an increment.
    // The cache counts come in message_start, and again (the same totals) in a later message_delta.
    for (const key of ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'] as const) {
      if (typeof usage[key] === 'number') this.usage[key] = usage[key];
    }
  }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible Chat Completions
// ---------------------------------------------------------------------------

interface CallInProgress {
  id: string;
  name: string;
  args: string;
  announced: boolean;
}

export class OpenAIStream {
  private text = '';
  private refusal = '';
  /** What the model thought (`reasoning_content`): kept with the reply, as a reply sent whole keeps it. */
  private reasoning = '';
  private calls = new Map<number, CallInProgress>();
  /** A server's index whose call was followed by another whole call at the same index: where that one went. */
  private moved = new Map<number, number>();
  private finishReason: string | null = null;
  private usage: Record<string, unknown> | null = null;
  private done = false;
  private nextIndex = 0;
  private lastIndex = -1;

  constructor(private readonly sink?: StreamSink) {}

  private allocateCallIndex(): number | null {
    while (this.calls.has(this.nextIndex) && this.nextIndex < MAX_STREAM_INDEX) this.nextIndex++;
    if (this.nextIndex >= MAX_STREAM_INDEX) return null;
    return this.nextIndex++;
  }

  accept(event: SSEEvent): void {
    const payload = parseEvent(event);
    if (!payload) {
      this.done = true;
      return;
    }
    if (payload.error !== undefined) throw streamError(payload.error);
    // OAIY streams a tool call's text as it is written, before the call is whole.
    if (isRecord(payload.oaiy_tool_preview)) {
      const p = payload.oaiy_tool_preview;
      if (typeof p.text === 'string' && p.text) this.sink?.draft?.(p.text, p.start === true);
    }
    // OAIY says how far it has read the prompt (a local model's first words can be a minute away).
    if (isRecord(payload.oaiy_progress)) {
      const p = payload.oaiy_progress;
      if (typeof p.prompt_done === 'number' && typeof p.prompt_total === 'number') this.sink?.reading?.(p.prompt_done, p.prompt_total, typeof p.cached_tokens === 'number' ? p.cached_tokens : null);
    }
    // OAIY counts the model's thought against its budget as it thinks, and says when the thought is over.
    if (isRecord(payload.oaiy_thinking)) {
      const t = payload.oaiy_thinking;
      if (typeof t.used === 'number' && Number.isFinite(t.used)) this.sink?.thinkingCount?.(t.used, typeof t.budget === 'number' && Number.isFinite(t.budget) ? t.budget : null, t.done === true);
    }
    // The usage chunk (stream_options.include_usage) has an empty choices list.
    if (isRecord(payload.usage)) this.usage = payload.usage;
    const choice = Array.isArray(payload.choices) ? payload.choices.find((c) => isRecord(c) && (c.index === undefined || c.index === 0)) : undefined;
    if (!isRecord(choice)) return;
    const delta = isRecord(choice.delta) ? choice.delta : isRecord(choice.message) ? choice.message : {};
    if (typeof delta.content === 'string' && delta.content) {
      this.text += delta.content;
      this.sink?.text(delta.content);
    }
    if (typeof delta.refusal === 'string') this.refusal += delta.refusal;
    const reasoning = typeof delta.reasoning_content === 'string' ? delta.reasoning_content : typeof delta.reasoning === 'string' ? delta.reasoning : '';
    if (reasoning) {
      this.reasoning += reasoning;
      this.sink?.thinking?.(reasoning);
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const raw of delta.tool_calls) if (isRecord(raw)) this.readCall(raw);
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason) this.finishReason = choice.finish_reason;
  }

  private readCall(raw: Record<string, unknown>): void {
    const id = typeof raw.id === 'string' ? raw.id : '';
    const fn = isRecord(raw.function) ? raw.function : {};
    const hasRawIndex = raw.index !== undefined && raw.index !== null;
    let rawIndex: number | null = null;
    if (hasRawIndex) {
      if (typeof raw.index !== 'number' || !Number.isInteger(raw.index) || raw.index >= MAX_STREAM_INDEX) return;
      if (raw.index >= 0) rawIndex = raw.index;
    }
    // Calls are told apart by `index`. A server that leaves it out (or sends a negative index) sends each
    // call whole, or continues an existing call by id / the last one: look up by id first.
    let index: number | null = rawIndex !== null ? (this.moved.get(rawIndex) ?? rawIndex) : null;
    if (index === null) {
      if (id) {
        for (const [k, c] of this.calls) {
          if (c.id === id) {
            index = k;
            break;
          }
        }
        if (index === null) index = this.allocateCallIndex();
      } else {
        index = this.lastIndex >= 0 ? this.lastIndex : this.allocateCallIndex();
      }
    }
    if (index === null || index < 0 || index >= MAX_STREAM_INDEX) return;
    // A call's first fragment (its name, with an id of its own) at an index whose call already has a name and another
    // id: a server that sends each call whole at the same index. A new call, and the index's later fragments are its own
    // (the two were merged into one call, its arguments two objects run together, and neither ran).
    const there = this.calls.get(index);
    if (rawIndex !== null && id && typeof fn.name === 'string' && fn.name && there?.name && there.id && there.id !== id) {
      const allocated = this.allocateCallIndex();
      if (allocated === null) return;
      index = allocated;
      this.moved.set(rawIndex, index);
    }
    if (index >= this.nextIndex) this.nextIndex = index + 1;
    this.lastIndex = index;
    let call = this.calls.get(index);
    if (!call) {
      call = { id: '', name: '', args: '', announced: false };
      this.calls.set(index, call);
    }
    if (id) call.id = id;
    // A name arrives whole, in the call's first fragment; later fragments carry arguments only.
    if (typeof fn.name === 'string' && fn.name && !call.name) call.name = fn.name;
    if (!call.announced && call.name) {
      call.announced = true;
      this.sink?.toolStart(index, call.id, call.name);
    }
    if (typeof fn.arguments === 'string' && fn.arguments) {
      call.args += fn.arguments;
      this.sink?.toolArgs(index, fn.arguments);
    } else if (isRecord(fn.arguments)) {
      const whole = JSON.stringify(fn.arguments);
      call.args = whole;
      this.sink?.toolArgs(index, whole);
    }
  }

  /** The reply as the non-streamed API returns it. */
  result(): Record<string, unknown> {
    if (!this.done && this.finishReason === null) throw closedEarly();
    const calls = [...this.calls.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, c]) => c)
      .filter((c) => c.name);
    return {
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: this.text,
            ...(this.refusal ? { refusal: this.refusal } : {}),
            ...(this.reasoning ? { reasoning_content: this.reasoning } : {}),
            ...(calls.length > 0 ? { tool_calls: calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } })) } : {}),
          },
          finish_reason: this.finishReason,
        },
      ],
      ...(this.usage ? { usage: this.usage } : {}),
    };
  }
}

/**
 * The text a live view shows of a reply that calls tools as text: the reply
 * with its <tool_call> blocks taken out — including one still being
 * written — and the names of the calls begun so far, in order.
 */
export function liveTextCalls(raw: string): { text: string; calls: Array<{ name: string; body: string }> } {
  const calls: Array<{ name: string; body: string }> = [];
  const opening = /<tool_call\s+name\s*=\s*["']([A-Za-z_][\w-]*)["']\s*>/g;
  let match: RegExpExecArray | null;
  while ((match = opening.exec(raw)) !== null) {
    const rest = raw.slice(match.index + match[0].length);
    const close = rest.search(/<\/tool_call\s*>/);
    calls.push({ name: match[1], body: close < 0 ? rest : rest.slice(0, close) });
  }
  let text = raw.replace(/<tool_call[\s\S]*?(<\/tool_call\s*>|$)/g, '');
  // A tag still being written ("<tool_c") is not text yet.
  text = text.replace(/<[A-Za-z_/][^>]*$/, '');
  return { text: text.replace(/\n{3,}/g, '\n\n').trim(), calls };
}

/**
 * What a call in progress is about — its path, name, page or topic — read
 * from arguments that may still be arriving: a JSON fragment, or the text
 * protocol's <arg> elements. Only a value that is already complete is read.
 */
export function liveSubject(args: string): string {
  const json = /"(path|from|name|page|topic|question)"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(args);
  if (json) {
    try {
      return String(JSON.parse(`"${json[2]}"`)).slice(0, 120);
    } catch {
      return json[2].slice(0, 120);
    }
  }
  const arg = /<arg\s+name\s*=\s*["'](path|from|name|page|topic|question)["']\s*>\s*([^<\n]+?)\s*<\/arg/.exec(args);
  return arg ? arg[2].slice(0, 120) : '';
}
