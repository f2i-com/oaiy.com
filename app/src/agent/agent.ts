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
import { sendTurn, type Attachment, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { TOOLS, checkApp, runTool, type SoftnHost, type ToolContext } from './tools';
import { appLabel, findApps, resolveApp } from '../softn/softn';
import type { ImagePart } from './images';

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  | { type: 'tool_start'; index: number; name: string }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; result: ToolResult }
  | { type: 'status'; message: string }
  | { type: 'usage'; usage: Usage }
  /** The automatic check of a SoftN app after the agent changed it: running, then its outcome. */
  | { type: 'check'; id: string; root: string; state: 'running' | 'ok' | 'failed'; text?: string }
  | { type: 'done'; text: string; steps: number }
  | { type: 'error'; message: string };

export const MAX_STEPS = 60;
const TRIM_OVER_CHARS = 240_000;
const KEEP_RECENT_TURNS = 8;
/** The same automatic-check errors this many times in a row stop the run. */
const SAME_CHECK_LIMIT = 3;
/** Images stay in the conversation for this many image-bearing turns. */
const KEEP_IMAGE_TURNS = 3;

export const SYSTEM_PROMPT = `You are bot.computer, a coding agent that runs entirely inside the user's web browser.

The user's project lives in a virtual filesystem in the browser; "/" is the project root and there is nothing outside it. Work with the tools:
- list_files, read_file, grep, glob to look around; read a file before you edit or replace it.
- edit_file for changes to existing files (exact, unique matches), write_file for new files or full rewrites.
- code_run to compute, test ideas, or process data in JavaScript or Python (a Zipp VM sandbox in a Web Worker).
- sandbox_shell for shell-style work (an emulated POSIX shell on the same sandbox). There is no real operating system: git, npm, pip, compilers and other native programs do not exist. Do not pretend to run them.
- SoftN apps: a SoftN app is a folder whose manifest.json names a .ui page as "main" (with ui/*.ui pages and logic/*.logic or .py). A project can hold several, each in its own folder: to rebuild or learn from an existing app, read its files and write the new one in another folder. A .softn the user attaches is unpacked into its own folder (the original stays in uploads/, and softn_import unpacks any .softn in the project): when they ask for changes, edit that folder; when they ask to recreate, redo or base something on it, write a new app in a new folder and leave the original as it is. The SoftN reference is in your tools, so do not guess the language: softn_docs with no arguments gives the map, topic "guide" is the writing guide (read it before your first app), search finds how something is done across the guides, the components and the example apps; softn_components gives exact props and events; softn_examples has complete working apps to read or copy. Keep manifest.json true. After each step that changes an app, bot.computer checks it automatically (its files, then a real render) and adds the outcome to that step's result: when it reports errors, fix them before anything else. softn_check checks on demand; softn_inspect shows what the page displays; softn_interact uses the app like a person (click, fill, select, press keys) and reports errors the app raises, so test that the app works, not just that it renders. The user watches the app in a live preview as you build it, and can export any app folder as a .softn file.
- web_fetch, curl, fetch() go through the user's network gate (/internet) and, from a browser, only reach sites that allow cross-origin requests. If the gate refuses a host, say so; the user decides whether to allow it.

Work in small, verified steps. Prefer running code to check a claim over guessing. When you are done, say briefly what you changed and what you verified.`;

export interface AgentOptions {
  vfs: Vfs;
  gate: NetGate;
  provider: () => ProviderConfig | null;
  /** A short description of the project, given to the model with the first request. */
  projectSummary: () => string;
  /** The live SoftN preview: softn_check, softn_inspect, softn_interact and the automatic check. */
  softn?: SoftnHost;
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

function hasImages(turn: Turn): boolean {
  return (turn.role === 'user' && !!turn.images?.length) || (turn.role === 'tool' && turn.results.some((r) => r.images?.length));
}

/** Older images become a note: each costs the model as much as a page of text. */
function withoutOldImages(turns: Turn[], keep = KEEP_IMAGE_TURNS): Turn[] {
  let seen = 0;
  const out = [...turns];
  for (let i = out.length - 1; i >= 0; i--) {
    const turn = out[i];
    if (!hasImages(turn)) continue;
    if (++seen <= keep) continue;
    if (turn.role === 'user') out[i] = { role: 'user', text: `${turn.text}\n[${turn.images!.length} image(s) shown earlier, no longer attached; look again with view_image if needed]` };
    else if (turn.role === 'tool') out[i] = { role: 'tool', results: turn.results.map((r) => (r.images?.length ? { ...r, images: undefined, content: `${r.content}\n[image no longer attached; call view_image again to see it]` } : r)) };
  }
  return out;
}

/** Old tool output shrinks first; the conversation's shape never changes. */
function trimmed(allTurns: Turn[], images = true): Turn[] {
  const turns = withoutOldImages(allTurns, images ? KEEP_IMAGE_TURNS : 0);
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
    this.toolContext = { vfs: options.vfs, gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} }, softn: options.softn };
  }

  reset(): void {
    this.turns = [];
    this.toolContext.reads.clear();
    this.toolContext.shell = { cwd: '/', env: {} };
  }

  /** Apps whose last check failed, with what it said: the run is not done while any are here. */
  readonly failingApps = new Map<string, string>();
  private sameCheck = { signature: '', count: 0 };

  /**
   * After a step: check each SoftN app the step changed, unless the step
   * already ran softn_check on it after its last change. The outcome goes
   * into the step's last result, where every provider carries it.
   */
  private async autoCheck(calls: ToolCall[], results: ToolResult[], changes: Array<{ path: string; index: number }>, emit: (e: AgentEvent) => void): Promise<string | null> {
    if (!changes.length) return null;
    const vfs = this.options.vfs;
    const apps = findApps(vfs);
    const rootOf = (path: string) => apps.filter((r) => r === '' || path === r || path.startsWith(`${r}/`)).sort((a, b) => b.length - a.length)[0];
    const lastChange = new Map<string, number>();
    for (const change of changes) {
      const root = rootOf(change.path);
      if (root !== undefined) lastChange.set(root, Math.max(lastChange.get(root) ?? -1, change.index));
    }
    const lastCheck = new Map<string, number>();
    calls.forEach((call, index) => {
      if (call.name !== 'softn_check') return;
      const target = resolveApp(vfs, call.input.app);
      if (target.ok) lastCheck.set(target.root, index);
    });
    let stop: string | null = null;
    for (const [root, changed] of lastChange) {
      if ((lastCheck.get(root) ?? -1) > changed) continue;
      const id = `check-${Date.now()}-${root}`;
      emit({ type: 'check', id, root, state: 'running' });
      const check = await checkApp(this.toolContext, root);
      emit({ type: 'check', id, root, state: check.ok ? 'ok' : 'failed', text: check.text });
      const last = results[results.length - 1];
      last.content += `\n\n[Automatic check of ${appLabel(root)} after this step]\n${check.text}${check.ok ? '' : '\nFix these errors before going on (read the files involved; softn_docs search helps).'}`;
      if (check.ok) {
        this.failingApps.delete(root);
        this.sameCheck = { signature: '', count: 0 };
        continue;
      }
      this.failingApps.set(root, check.text);
      this.sameCheck = check.signature === this.sameCheck.signature ? { signature: check.signature, count: this.sameCheck.count + 1 } : { signature: check.signature, count: 1 };
      if (this.sameCheck.count >= SAME_CHECK_LIMIT) stop = `The same errors in ${appLabel(root)} came back ${SAME_CHECK_LIMIT} times after the agent's fixes, so the run stopped rather than keep trying the same thing. Say "continue" to let it try again, or say how to fix it.`;
    }
    return stop;
  }

  /** False once the model has refused images: they are left out from then on. */
  private imagesAccepted = true;

  private async request(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<Reply> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        return await sendTurn(provider, SYSTEM_PROMPT, trimmed(this.turns, this.imagesAccepted), TOOLS, {
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
        // A model without vision refuses image content: carry on in text.
        if (this.imagesAccepted && error.kind === 'http' && /image|vision|multimodal|image_url|content.*array/i.test(`${error.message} ${error.detail ?? ''}`) && this.turns.some(hasImages)) {
          this.imagesAccepted = false;
          emit({ type: 'status', message: 'This model does not take images; continuing with text only' });
          continue;
        }
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
  async run(prompt: string, emit: (e: AgentEvent) => void, signal?: AbortSignal, images: ImagePart[] = [], attachments: Attachment[] = []): Promise<void> {
    const provider = this.options.provider();
    if (!provider) {
      emit({ type: 'error', message: 'No AI provider is set up yet. Open Settings to connect a local server (Ollama, LM Studio) or an API.' });
      return;
    }
    this.running = true;
    this.toolContext.signal = signal;
    const first = this.turns.length === 0;
    const text = first ? `<project>\n${this.options.projectSummary()}\n</project>\n\n${prompt}` : prompt;
    this.turns.push({ role: 'user', text, ...(images.length ? { images } : {}), ...(attachments.length ? { attachments } : {}) });
    let failures = 0;
    let lastFailure = '';
    // What each step changes, by the index of the call that changed it.
    let callIndex = -1;
    let changes: Array<{ path: string; index: number }> = [];
    const unwatch = this.options.vfs.onChange((change) => {
      if (change.type !== 'reset') changes.push({ path: change.path.replace(/^\/+/, ''), index: callIndex });
    });
    this.sameCheck = { signature: '', count: 0 };
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
        changes = [];
        for (const [index, call] of reply.calls.entries()) {
          signal?.throwIfAborted();
          callIndex = index;
          emit({ type: 'tool_call', call });
          const result = await runTool(call, this.toolContext);
          results.push(result);
          emit({ type: 'tool_result', result });
          if (result.isError && result.content === lastFailure) failures++;
          else failures = result.isError ? 1 : 0;
          lastFailure = result.isError ? result.content : '';
        }
        callIndex = reply.calls.length;
        const stop = await this.autoCheck(reply.calls, results, changes, emit);
        this.turns.push({ role: 'tool', results });
        if (stop) {
          emit({ type: 'error', message: stop });
          return;
        }
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
      unwatch();
      this.running = false;
    }
  }
}
