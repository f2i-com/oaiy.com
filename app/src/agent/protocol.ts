/**
 * One agent request, in either wire shape: Anthropic Messages, or the
 * OpenAI-compatible Chat Completions that OpenAI, Ollama, LM Studio,
 * llama.cpp and most gateways speak. Always streamed; a server that ignores
 * `stream` and answers with JSON is read the same way.
 */
import { AIProviderError, isOpenAIHost, isRecord, postProviderStream, sanitizeCount, type SSEEvent } from './providers/aiProvider';
import { providerEndpoints } from './providers/providerConnection';
import { AnthropicStream, OpenAIStream, type StreamSink } from './providers/stream';
import type { ProviderConfig } from './providers/types';
import type { ImagePart } from './images';

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
  /** Why the arguments could not be read, when they could not. */
  parseError?: string;
}

export interface ToolResult {
  id: string;
  name: string;
  content: string;
  isError: boolean;
  /** Images the tool shows the model (view_image). */
  images?: ImagePart[];
  /** Project files the tool shows the person in the chat (present_file); not sent to the model. */
  files?: string[];
  /** The outcome of softn_check, for the agent's record of failing apps; not sent to the model. */
  check?: { root: string; ok: boolean; text: string };
  /** Pictures generate_image made for a scripted video, for the agent to have reviewed; not sent to the model. */
  review?: FrameReview[];
  /** The tool started calls (SessionTool.startsCalls, asked of what it answered itself, before any flow's note); not sent to the model. */
  startedCalls?: boolean;
}

/** A picture made for a scripted video, and what it is checked against. */
export interface FrameReview {
  path: string;
  kind?: 'start' | 'end' | 'background' | 'character' | 'prop';
  shot?: string;
  scene?: string;
  /** The reference images it was made from. */
  references: string[];
  /** The story's folder (where script.md is; '' for the project root). */
  story: string;
  /** What the try before it was sent back for, to check it is fixed. */
  fix?: string;
  /** The user flagged it as wrong: what they said, or true when they said nothing. */
  flagged?: string | true;
}

/** A file attached to a user message, as saved in the project; for the chat only. */
export interface Attachment {
  name: string;
  /** Where it was saved, without a leading slash. */
  path: string;
  /** For a .softn: the folder it was unpacked into. */
  app?: string;
}

/**
 * Where a turn of a person's conversation came from, and when (the phone's
 * conversations keep a person's calls and texts in one order): kept with the
 * turn, never sent to a model.
 */
export interface TurnPlace {
  /** When it happened (ms). */
  at?: number;
  /** Which of the person's agents it belongs to: their calls' or their texts'. */
  via?: 'call' | 'sms';
}

export type Turn =
  | ({ role: 'user'; text: string; images?: ImagePart[]; attachments?: Attachment[]; /** Written by OAIY (a nudge to finish), not the person. */ automatic?: boolean; /** A summary of everything before it: the model reads from the latest one on. */ summary?: boolean; /** A fresh start (a new phone call): the model reads from here on, what came before kept for the chat and for searching. */ fresh?: boolean; /** Guides read with this request (it plainly asked for that kind of work). */ guides?: string[] } & TurnPlace)
  | ({ role: 'assistant'; text: string; calls: ToolCall[]; anthropicContent?: unknown[]; /** What the model thought first: shown in the chat, never sent back. */ thinking?: string } & TurnPlace)
  | ({ role: 'tool'; results: ToolResult[] } & TurnPlace);

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
}

export interface Reply {
  text: string;
  thinking: string;
  calls: ToolCall[];
  stopReason: string | null;
  truncated: boolean;
  usage: Usage;
  anthropicContent?: unknown[];
}

export interface RequestOptions {
  signal?: AbortSignal;
  sink?: StreamSink;
  maxOutputTokens?: number;
  timeoutMs?: number;
  /**
   * How hard the model thinks first (OAIY's `reasoning_effort`). `none`: not at
   * all (a phone call cannot wait), also asked of other local servers through
   * the chat template; APIs are not asked.
   */
  reasoning?: 'none' | 'low' | 'medium' | 'high' | 'max';
}

function anthropicMessages(turns: Turn[]): unknown[] {
  const out: Array<{ role: string; content: unknown[] }> = [];
  const push = (role: string, blocks: unknown[]) => {
    const last = out[out.length - 1];
    if (last && last.role === role) last.content.push(...blocks);
    else out.push({ role, content: blocks });
  };
  for (const turn of turns) {
    if (turn.role === 'user') push('user', [{ type: 'text', text: turn.text }, ...(turn.images ?? []).map(anthropicImage)]);
    else if (turn.role === 'assistant') {
      if (turn.anthropicContent?.length) push('assistant', turn.anthropicContent);
      else {
        const blocks: unknown[] = [];
        if (turn.text) blocks.push({ type: 'text', text: turn.text });
        for (const call of turn.calls) blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.input });
        if (blocks.length) push('assistant', blocks);
      }
    } else {
      push('user', turn.results.map((r) => ({
        type: 'tool_result',
        tool_use_id: r.id,
        content: r.images?.length ? [{ type: 'text', text: r.content }, ...r.images.map(anthropicImage)] : r.content,
        is_error: r.isError,
      })));
    }
  }
  // Cache the conversation up to its last message.
  const last = out[out.length - 1];
  const block = last?.content[last.content.length - 1];
  if (isRecord(block)) block.cache_control = { type: 'ephemeral' };
  return out;
}

function anthropicImage(image: ImagePart): unknown {
  return { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } };
}

function openAIImage(image: ImagePart): unknown {
  return { type: 'image_url', image_url: { url: `data:${image.mediaType};base64,${image.data}` } };
}

function openAIMessages(system: string, turns: Turn[]): unknown[] {
  const out: unknown[] = [{ role: 'system', content: system }];
  for (const turn of turns) {
    if (turn.role === 'user') {
      out.push({ role: 'user', content: turn.images?.length ? [{ type: 'text', text: turn.text }, ...turn.images.map(openAIImage)] : turn.text });
      continue;
    }
    else if (turn.role === 'assistant') {
      // Content may be empty only beside tool calls; some servers refuse null.
      const message: Record<string, unknown> = { role: 'assistant', content: turn.text || (turn.calls.length ? null : '') };
      if (turn.calls.length) {
        message.tool_calls = turn.calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } }));
      }
      out.push(message);
    } else {
      for (const r of turn.results) out.push({ role: 'tool', tool_call_id: r.id, content: r.content });
      // A tool message carries text only; its images follow as a user message.
      const images = turn.results.flatMap((r) => (r.images ?? []).map((image) => ({ image, tool: r.name })));
      if (images.length) {
        out.push({
          role: 'user',
          content: [{ type: 'text', text: `The image${images.length === 1 ? '' : 's'} returned by ${[...new Set(images.map((i) => i.tool))].join(', ')} above:` }, ...images.map((i) => openAIImage(i.image))],
        });
      }
    }
  }
  return out;
}

function decodeAnthropic(data: Record<string, unknown>, parseErrors?: Map<string, string>): Reply {
  const content = Array.isArray(data.content) ? data.content : [];
  let text = '';
  let thinking = '';
  const calls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'thinking' && typeof block.thinking === 'string') thinking += block.thinking;
    else if (block.type === 'tool_use') {
      const id = String(block.id ?? `call_${calls.length}`);
      calls.push({ id, name: String(block.name ?? ''), input: isRecord(block.input) ? block.input : {}, parseError: parseErrors?.get(id) });
    }
  }
  const usage = isRecord(data.usage) ? data.usage : {};
  const stopReason = typeof data.stop_reason === 'string' ? data.stop_reason : null;
  return {
    text,
    thinking,
    calls,
    stopReason,
    truncated: stopReason === 'max_tokens',
    usage: {
      inputTokens: sanitizeCount(usage.input_tokens) + sanitizeCount(usage.cache_creation_input_tokens) + sanitizeCount(usage.cache_read_input_tokens),
      outputTokens: sanitizeCount(usage.output_tokens),
      cacheReadTokens: sanitizeCount(usage.cache_read_input_tokens),
    },
    anthropicContent: content,
  };
}

function decodeOpenAI(data: Record<string, unknown>): Reply {
  const choice = Array.isArray(data.choices) && isRecord(data.choices[0]) ? data.choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const calls: ToolCall[] = [];
  const rawCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const [i, raw] of rawCalls.entries()) {
    if (!isRecord(raw)) continue;
    const fn = isRecord(raw.function) ? raw.function : {};
    let input: Record<string, unknown> = {};
    let parseError: string | undefined;
    const args = fn.arguments;
    if (isRecord(args)) input = args;
    else if (typeof args === 'string' && args.trim()) {
      try {
        const parsed: unknown = JSON.parse(args);
        if (isRecord(parsed)) input = parsed;
        else parseError = 'arguments are not a JSON object';
      } catch (error) {
        parseError = `arguments are not valid JSON (${(error as Error).message})`;
      }
    }
    calls.push({ id: typeof raw.id === 'string' && raw.id ? raw.id : `call_${Date.now().toString(36)}_${i}`, name: String(fn.name ?? ''), input, parseError });
  }
  const reasoning = typeof message.reasoning_content === 'string' ? message.reasoning_content : typeof message.reasoning === 'string' ? message.reasoning : '';
  const usage = isRecord(data.usage) ? data.usage : {};
  const details = isRecord(usage.prompt_tokens_details) ? usage.prompt_tokens_details : {};
  const finish = typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
  return {
    text: typeof message.content === 'string' ? message.content : '',
    thinking: reasoning,
    calls,
    stopReason: finish,
    truncated: finish === 'length',
    usage: {
      inputTokens: sanitizeCount(usage.prompt_tokens),
      outputTokens: sanitizeCount(usage.completion_tokens),
      cacheReadTokens: sanitizeCount(details.cached_tokens),
    },
  };
}

/** Send the conversation and read the whole reply, streaming it to `sink`. */
export async function sendTurn(
  provider: ProviderConfig,
  system: string,
  turns: Turn[],
  tools: ToolSpec[],
  options: RequestOptions = {},
): Promise<Reply> {
  const model = provider.modelId?.trim();
  if (!model) throw new AIProviderError('no-model', `${provider.name} has no model chosen. Open settings and pick one of its models.`);
  const url = providerEndpoints(provider).chat;
  const maxTokens = options.maxOutputTokens ?? 16_384;
  if (provider.type === 'anthropic') {
    const body = {
      model,
      max_tokens: maxTokens,
      stream: true,
      system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
      tools: tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters })),
      messages: anthropicMessages(turns),
    };
    const stream = new AnthropicStream(options.sink);
    const outcome = await postProviderStream(provider, url, body, {
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      onEvent: (event: SSEEvent) => stream.accept(event),
    });
    return decodeAnthropic(outcome.streamed ? stream.result() : outcome.data, stream.parseErrors);
  }
  const body: Record<string, unknown> = {
    // Following OAIY's Engines: no model named, so the engine uses the one chosen there (naming the last one seen would load it back).
    ...(provider.followEngine ? {} : { model }),
    stream: true,
    stream_options: { include_usage: true },
    messages: openAIMessages(system, turns),
  };
  if (tools.length) body.tools = tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  // OpenAI's reasoning models take max_completion_tokens; other servers read max_tokens.
  if (isOpenAIHost(url)) body.max_completion_tokens = maxTokens;
  else body.max_tokens = maxTokens;
  if (options.reasoning && provider.serverKind === 'oaiy') body.reasoning_effort = options.reasoning;
  else if (options.reasoning === 'none' && provider.type === 'local') body.chat_template_kwargs = { enable_thinking: false };
  const stream = new OpenAIStream(options.sink);
  const outcome = await postProviderStream(provider, url, body, {
    signal: options.signal,
    timeoutMs: options.timeoutMs,
    onEvent: (event: SSEEvent) => stream.accept(event),
  });
  return decodeOpenAI(outcome.streamed ? stream.result() : outcome.data);
}
