/**
 * The agent loop: send the conversation, run the tools the model asks for,
 * send their results back, until it answers without asking for anything.
 *
 * The structure follows softn Studio's runAgent (Apache-2.0): rate limits and
 * timeouts are retried, a run has a step cap, the same failure three times in
 * a row stops it, and old tool output is trimmed as the conversation grows.
 */
import type { NetGate } from '../gate/netgate';
import type { Vfs } from '../vfs/vfs';
import { AIProviderError } from './providers/aiProvider';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { TOOLS, runTool, type ToolContext } from './tools';

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  | { type: 'tool_start'; index: number; name: string }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; result: ToolResult }
  | { type: 'status'; message: string }
  | { type: 'usage'; usage: Usage }
  | { type: 'done'; text: string; steps: number }
  | { type: 'error'; message: string };

export const MAX_STEPS = 60;
const TRIM_OVER_CHARS = 240_000;
const KEEP_RECENT_TURNS = 8;

export const SYSTEM_PROMPT = `You are bot.computer, a coding agent that runs entirely inside the user's web browser.

The user's project lives in a virtual filesystem in the browser; "/" is the project root and there is nothing outside it. Work with the tools:
- list_files, read_file, grep, glob to look around; read a file before you edit or replace it.
- edit_file for changes to existing files (exact, unique matches), write_file for new files or full rewrites.
- code_run to compute, test ideas, or process data in JavaScript or Python (a Zipp VM sandbox in a Web Worker).
- sandbox_shell for shell-style work (an emulated POSIX shell on the same sandbox). There is no real operating system: git, npm, pip, compilers and other native programs do not exist. Do not pretend to run them.
- web_fetch, curl, fetch() go through the user's network gate (/internet) and, from a browser, only reach sites that allow cross-origin requests. If the gate refuses a host, say so; the user decides whether to allow it.

Work in small, verified steps. Prefer running code to check a claim over guessing. When you are done, say briefly what you changed and what you verified.`;

export interface AgentOptions {
  vfs: Vfs;
  gate: NetGate;
  provider: () => ProviderConfig | null;
  /** A short description of the project, given to the model with the first request. */
  projectSummary: () => string;
}

function estimateChars(turns: Turn[]): number {
  let n = 0;
  for (const t of turns) {
    if (t.role === 'user') n += t.text.length;
    else if (t.role === 'assistant') n += t.text.length + JSON.stringify(t.calls).length;
    else for (const r of t.results) n += r.content.length;
  }
  return n;
}

/** Old tool output shrinks first; the conversation's shape never changes. */
function trimmed(turns: Turn[]): Turn[] {
  if (estimateChars(turns) <= TRIM_OVER_CHARS) return turns;
  const cutoff = turns.length - KEEP_RECENT_TURNS;
  return turns.map((t, i) => {
    if (i >= cutoff || t.role !== 'tool') return t;
    return {
      role: 'tool',
      results: t.results.map((r) => (r.content.length > 600 ? { ...r, content: `${r.content.slice(0, 500)}\n[older output trimmed to save context]` } : r)),
    };
  });
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new AIProviderError('cancelled', 'Stopped.'));
    }, { once: true });
  });

export class Agent {
  turns: Turn[] = [];
  readonly toolContext: ToolContext;
  running = false;

  constructor(private readonly options: AgentOptions) {
    this.toolContext = { vfs: options.vfs, gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} } };
  }

  reset(): void {
    this.turns = [];
    this.toolContext.reads.clear();
    this.toolContext.shell = { cwd: '/', env: {} };
  }

  private async request(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<Reply> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        return await sendTurn(provider, SYSTEM_PROMPT, trimmed(this.turns), TOOLS, {
          signal,
          sink: {
            text: (delta) => emit({ type: 'text', delta }),
            thinking: (delta) => emit({ type: 'thinking', delta }),
            toolStart: (index, _id, name) => emit({ type: 'tool_start', index, name }),
            toolArgs: () => {},
          },
        });
      } catch (error) {
        lastError = error;
        if (!(error instanceof AIProviderError)) throw error;
        if (error.kind === 'rate-limited' && attempt < 3) {
          const wait = Math.min(error.retryAfterMs ?? 5000 * (attempt + 1), 60_000);
          emit({ type: 'status', message: `The provider is busy; retrying in ${Math.round(wait / 1000)} s` });
          await sleep(wait, signal);
          continue;
        }
        if ((error.kind === 'timeout' || error.kind === 'network') && attempt < 1) {
          emit({ type: 'status', message: 'The request failed; retrying once' });
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  /** Run one user request to completion. */
  async run(prompt: string, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const provider = this.options.provider();
    if (!provider) {
      emit({ type: 'error', message: 'No AI provider is set up yet. Open Settings to connect a local server (Ollama, LM Studio) or an API.' });
      return;
    }
    this.running = true;
    this.toolContext.signal = signal;
    const first = this.turns.length === 0;
    this.turns.push({ role: 'user', text: first ? `<project>\n${this.options.projectSummary()}\n</project>\n\n${prompt}` : prompt });
    let failures = 0;
    let lastFailure = '';
    try {
      for (let step = 1; step <= MAX_STEPS; step++) {
        signal?.throwIfAborted();
        const reply = await this.request(provider, emit, signal);
        emit({ type: 'usage', usage: reply.usage });
        this.turns.push({ role: 'assistant', text: reply.text, calls: reply.calls, anthropicContent: provider.type === 'anthropic' ? reply.anthropicContent : undefined });
        if (!reply.calls.length) {
          if (reply.truncated) emit({ type: 'status', message: 'The reply was cut off at the output limit.' });
          emit({ type: 'done', text: reply.text, steps: step });
          return;
        }
        const results: ToolResult[] = [];
        for (const call of reply.calls) {
          signal?.throwIfAborted();
          emit({ type: 'tool_call', call });
          const result = await runTool(call, this.toolContext);
          results.push(result);
          emit({ type: 'tool_result', result });
          if (result.isError && result.content === lastFailure) failures++;
          else failures = result.isError ? 1 : 0;
          lastFailure = result.isError ? result.content : '';
        }
        this.turns.push({ role: 'tool', results });
        if (failures >= 3) {
          emit({ type: 'error', message: 'The same step failed three times in a row, so the run stopped. Say how to proceed.' });
          return;
        }
      }
      emit({ type: 'error', message: `The run reached its ${MAX_STEPS}-step limit. Say "continue" to keep going.` });
    } catch (error) {
      if (signal?.aborted || (error instanceof AIProviderError && error.kind === 'cancelled')) {
        emit({ type: 'status', message: 'Stopped.' });
      } else {
        emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
      }
      // A turn left waiting for tool results cannot be sent again.
      const last = this.turns[this.turns.length - 1];
      if (last?.role === 'assistant' && last.calls.length) {
        this.turns.push({ role: 'tool', results: last.calls.map((c) => ({ id: c.id, name: c.name, content: 'Error: the run stopped before this tool ran', isError: true })) });
      }
    } finally {
      this.running = false;
    }
  }
}
