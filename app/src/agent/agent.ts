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
import { DEFAULT_COMPACT_AT, budgetFor, contextWindow, formatTokens, overflowWindow } from './context';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Attachment, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { TOOLS, checkApp, readPlan, runTool, type Plan, type SoftnHost, type ToolContext } from './tools';
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
  /** How full the model's context is: the prompt about to be sent, in tokens, and the window. */
  | { type: 'context'; used: number; window: number }
  /** Older turns were summarized to make room. */
  | { type: 'compact'; turns: number; before: number; after: number; how: 'summary' | 'trimmed' }
  /** The checklist changed (update_plan). */
  | { type: 'plan'; plan: Plan }
  /** bot.computer asked the model to carry on (open plan items, a failing app). */
  | { type: 'nudge'; message: string }
  /** The automatic check of a SoftN app after the agent changed it: running, then its outcome. */
  | { type: 'check'; id: string; root: string; state: 'running' | 'ok' | 'failed'; text?: string }
  | { type: 'done'; text: string; steps: number }
  | { type: 'error'; message: string };

export const MAX_STEPS = 60;
const KEEP_RECENT_TURNS = 8;
/** Characters per token until the provider's own counts say otherwise. */
const DEFAULT_CHARS_PER_TOKEN = 3.5;
/** What an image costs, in tokens (about what vision models charge for one of ~1 megapixel). */
const IMAGE_TOKENS = 1600;

const SUMMARIZER_PROMPT = `You compress the conversation of a coding agent (bot.computer) so it can carry on with less context. Write a summary the agent can continue from as if it had read everything, with these sections (leave out empty ones):
Goal: what the user asked for, in their words where it matters, including the request being worked on now.
Decisions and constraints: what was agreed or ruled out, and why.
Files: the files read, and the files created or changed with what changed in each.
Plan: the steps and which are done.
Current state: what works, errors still open, and what the agent was about to do next.
Facts to remember: names, values, paths, commands, ids, anything that would be expensive to find again.
Be specific and complete. Keep code only where it is essential. At most about 1200 words. Write only the summary.`;
/** A run that stops short of its goal is asked to carry on at most this many times. */
const MAX_NUDGES = 2;
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

Work toward the user's goal. For anything that takes several steps (building or changing an app, a feature, a fix across files), first call update_plan with the goal and 3 to 8 concrete steps, and update it as each step starts and finishes: the user watches that checklist. You are done when every step is done and the result is checked (for an app: it renders without errors and softn_interact shows it working), not before.

Work in small, verified steps. Prefer running code to check a claim over guessing. When you are done, say briefly what you changed and what you verified.`;

export interface AgentOptions {
  vfs: Vfs;
  gate: NetGate;
  provider: () => ProviderConfig | null;
  /** A short description of the project, given to the model with the first request. */
  projectSummary: () => string;
  /** The live SoftN preview: softn_check, softn_inspect, softn_interact and the automatic check. */
  softn?: SoftnHost;
  /** The share of the context window a prompt may fill before older turns are summarized (default 0.75). */
  compactAt?: () => number;
  /** A cap on the window this agent works in (a sub-agent gets a smaller one). */
  maxContext?: number;
  /** The server stated the window (in an overflow error): remember it for this provider and model. */
  onWindow?: (tokens: number) => void;
}

function turnChars(t: Turn, charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  if (t.role === 'user') return t.text.length + (t.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken;
  if (t.role === 'assistant') return t.text.length + JSON.stringify(t.calls).length;
  return t.results.reduce((n, r) => n + r.content.length + (r.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken, 0);
}

function estimateChars(turns: Turn[], charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  return turns.reduce((n, t) => n + turnChars(t, charsPerToken), 0);
}

/** A turn as plain text, for the summarizer: long tool output is cut, the gist is kept. */
function transcript(turn: Turn): string {
  const cutText = (text: string, max: number) => (text.length > max ? `${text.slice(0, max)} […${text.length - max} more characters]` : text);
  if (turn.role === 'user') {
    if (turn.summary) return `Summary of what came before:\n${turn.text.replace(/^\[bot\.computer\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}`;
    const text = turn.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
    return `${turn.automatic ? 'bot.computer' : 'User'}: ${cutText(text, 6000)}${turn.images?.length ? ` [${turn.images.length} image(s)]` : ''}`;
  }
  if (turn.role === 'assistant') {
    const calls = turn.calls.map((c) => `  → ${c.name}(${cutText(JSON.stringify(c.input), 400)})`).join('\n');
    return `Assistant: ${cutText(turn.text, 3000)}${calls ? `\n${calls}` : ''}`;
  }
  return turn.results.map((r) => `  ← ${r.name}${r.isError ? ' (error)' : ''}: ${cutText(r.content, 1500)}`).join('\n');
}

/** Where the model's view starts: the latest summary, or the beginning. */
function viewStart(turns: Turn[]): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.role === 'user' && t.summary) return i;
  }
  return 0;
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

/**
 * The last resort when a summary cannot make room (or is not enough): old
 * tool output shrinks, then everything but the recent turns; the
 * conversation's shape never changes.
 */
function trimmed(allTurns: Turn[], images: boolean, maxChars: number): Turn[] {
  const turns = withoutOldImages(allTurns, images ? KEEP_IMAGE_TURNS : 0);
  if (estimateChars(turns) <= maxChars) return turns;
  const cutoff = turns.length - KEEP_RECENT_TURNS;
  const shrink = (limit: number) =>
    turns.map((t, i): Turn => {
      if (i >= cutoff) return t;
      if (t.role === 'tool') return { role: 'tool', results: t.results.map((r) => (r.content.length > limit ? { ...r, content: `${r.content.slice(0, limit)}\n[older output trimmed to save context]` } : r)) };
      if (t.role === 'user' && t.text.length > limit * 4 && !t.summary) return { ...t, text: `${t.text.slice(0, limit * 4)}\n[trimmed to save context]` };
      return t;
    });
  let out = shrink(500);
  if (estimateChars(out) > maxChars) out = shrink(120);
  return out;
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
  /** The whole conversation, as the chat shows it; the model reads from the latest summary on. */
  turns: Turn[] = [];
  /** Measured from the provider's token counts; used to estimate the next prompt. */
  private charsPerToken = DEFAULT_CHARS_PER_TOKEN;
  /** A window the server stated in an overflow error, smaller than the one configured, for that provider and model. */
  private windowOverride: { key: string; tokens: number } | null = null;
  readonly toolContext: ToolContext;
  running = false;

  constructor(private readonly options: AgentOptions) {
    this.toolContext = { vfs: options.vfs, gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} }, softn: options.softn };
  }

  reset(): void {
    this.turns = [];
    this.plan = null;
    this.failingApps.clear();
    this.toolContext.reads.clear();
    this.toolContext.shell = { cwd: '/', env: {} };
  }

  /** The checklist from the latest update_plan. */
  plan: Plan | null = null;
  /** Apps checked during this run, by the automatic check or softn_check. */
  private checkedRoots = new Set<string>();

  /**
   * Why the run is not done yet, if the model stopped early: plan items it
   * set this run and did not finish, or an app whose check still fails.
   */
  private unfinished(planThisRun: boolean): string | null {
    // Only apps checked in this run count: an old failure should not hold up something else.
    const failing = [...this.failingApps.entries()].find(([root]) => this.checkedRoots.has(root));
    if (failing) return `The last automatic check of ${appLabel(failing[0])} still reports errors:\n${failing[1]}\nFix them and check the app again before you finish. If you cannot, say what is wrong.`;
    const open = planThisRun && this.plan ? this.plan.items.filter((i) => i.status !== 'done') : [];
    if (open.length) return `Your plan still has ${open.length} open step${open.length > 1 ? 's' : ''}: ${open.map((i) => `"${i.text}"`).join(', ')}. Carry on with ${open.length > 1 ? 'them' : 'it'}. If a step is no longer needed, or already done, call update_plan to say so. Then finish with a short summary.`;
    return null;
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
      this.checkedRoots.add(root);
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

  /** What the model reads: the latest summary and everything after it. */
  view(): Turn[] {
    return this.turns.slice(viewStart(this.turns));
  }

  private window(provider: ProviderConfig): number {
    const configured = contextWindow(provider).tokens;
    const override = this.windowOverride?.key === `${provider.id}|${provider.modelId}` ? this.windowOverride.tokens : null;
    const window = override ? Math.min(configured, override) : configured;
    return this.options.maxContext ? Math.min(window, this.options.maxContext) : window;
  }

  /** The prompt's fixed part: the system prompt and the tool definitions. */
  private fixedChars(): number {
    return SYSTEM_PROMPT.length + JSON.stringify(TOOLS).length;
  }

  private budget(provider: ProviderConfig): { window: number; fixed: number; prompt: number; reply: number } {
    const window = this.window(provider);
    const fixed = Math.ceil(this.fixedChars() / this.charsPerToken);
    return { window, fixed, ...budgetFor(window, fixed) };
  }

  /** Tokens the conversation part of the next prompt will take. */
  private estimate(turns: Turn[]): number {
    return Math.ceil(estimateChars(withoutOldImages(turns, this.imagesAccepted ? KEEP_IMAGE_TURNS : 0), this.charsPerToken) / this.charsPerToken);
  }

  /**
   * Before a request: if the prompt would pass the compaction threshold,
   * summarize the older turns. `force` compacts regardless (the server said
   * the prompt was too long).
   */
  private async fit(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal, force = false): Promise<void> {
    const b = this.budget(provider);
    const used = this.estimate(this.view());
    emit({ type: 'context', used: used + b.fixed, window: b.window });
    const threshold = b.prompt * (this.options.compactAt?.() ?? DEFAULT_COMPACT_AT);
    if (!force && used <= threshold) return;
    await this.compact(provider, b, emit, signal);
  }

  /**
   * Summarize everything before the recent turns into one summary turn. The
   * recent turns (about a third of the room) stay word for word, starting on
   * a user or assistant turn so every tool result keeps the call it answers.
   */
  private async compact(provider: ProviderConfig, b: { window: number; fixed: number; prompt: number; reply: number }, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const start = viewStart(this.turns);
    const view = this.turns.slice(start);
    const before = this.estimate(view) + b.fixed;
    const tailChars = b.prompt * 0.3 * this.charsPerToken;
    let keepFrom = view.length;
    let chars = 0;
    for (let i = view.length - 1; i >= 1; i--) {
      chars += turnChars(view[i], this.charsPerToken);
      if (chars > tailChars && keepFrom < view.length) break;
      keepFrom = i;
    }
    while (keepFrom > 0 && view[keepFrom]?.role === 'tool') keepFrom--;
    const old = view.slice(0, keepFrom);
    // Nothing old enough to summarize (one huge recent step, or only the last summary): trimming handles it.
    if (!old.length || (old.length === 1 && old[0].role === 'user' && old[0].summary)) return;
    emit({ type: 'status', message: `Summarizing ${old.length} earlier turns to make room in the context…` });
    let summary: string;
    let how: 'summary' | 'trimmed' = 'summary';
    try {
      summary = await this.summarize(old, provider, b, signal);
    } catch (error) {
      if (signal?.aborted) throw error;
      summary = this.mechanicalSummary(old);
      how = 'trimmed';
    }
    const note: Turn = {
      role: 'user',
      automatic: true,
      summary: true,
      text: `[bot.computer] The conversation before this point (${old.length} turns) was summarized to fit the model's context.\n<project>\n${this.options.projectSummary()}\n</project>\n\n${summary}`,
    };
    this.turns.splice(start + keepFrom, 0, note);
    const after = this.estimate(this.view()) + b.fixed;
    emit({ type: 'compact', turns: old.length, before, after, how });
  }

  /** The model writes the summary, a chunk at a time when the old part is bigger than it can read. */
  private async summarize(old: Turn[], provider: ProviderConfig, b: { window: number; reply: number }, signal?: AbortSignal): Promise<string> {
    const room = Math.max(2000, (b.window - Math.min(4096, b.reply) - 2000) * 0.8 * this.charsPerToken);
    const chunks: string[] = [];
    let current = '';
    for (const turn of old) {
      let text = transcript(turn);
      if (text.length > room * 0.9) text = `${text.slice(0, room * 0.9)} […]`;
      if (current && current.length + text.length > room * 0.6) {
        chunks.push(current);
        current = '';
      }
      current += `${text}\n\n`;
    }
    if (current) chunks.push(current);
    let summary = '';
    for (const chunk of chunks) {
      const prompt = `${summary ? `The summary so far:\n${summary}\n\n` : ''}The conversation to ${summary ? 'add to it' : 'summarize'}:\n${chunk}\nWrite the ${summary ? 'updated ' : ''}summary.`;
      const reply = await sendTurn(provider, SUMMARIZER_PROMPT, [{ role: 'user', text: prompt }], [], { signal, maxOutputTokens: Math.min(4096, b.reply), sink: { text: () => {}, thinking: () => {}, toolStart: () => {}, toolArgs: () => {} } });
      if (!reply.text.trim()) throw new Error('the summary came back empty');
      summary = reply.text.trim();
    }
    return summary;
  }

  /** Without the model: the requests, the files touched, the plan. */
  private mechanicalSummary(old: Turn[]): string {
    const requests = old.filter((t): t is Extract<Turn, { role: 'user' }> => t.role === 'user' && !t.automatic).map((t) => `- ${t.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '').slice(0, 300)}`);
    const touched = new Set<string>();
    for (const t of old) if (t.role === 'assistant') for (const c of t.calls) if (typeof c.input.path === 'string' && /write|edit|delete/.test(c.name)) touched.add(c.input.path);
    const earlier = old.find((t) => t.role === 'user' && t.summary) as Extract<Turn, { role: 'user' }> | undefined;
    return [
      earlier ? `Earlier summary:\n${earlier.text.replace(/^\[bot\.computer\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}` : '',
      requests.length ? `Requests:\n${requests.join('\n')}` : '',
      touched.size ? `Files changed: ${[...touched].join(', ')}` : '',
      this.plan ? `Plan: ${this.plan.items.map((i) => `[${i.status}] ${i.text}`).join('; ')}` : '',
      '(A model-written summary could not be made; read files again where details matter.)',
    ].filter(Boolean).join('\n\n');
  }

  private async request(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<Reply> {
    let lastError: unknown;
    let overflowRetried = false;
    for (let attempt = 0; attempt < 4; attempt++) {
      const b = this.budget(provider);
      const sent = trimmed(this.view(), this.imagesAccepted, b.prompt * this.charsPerToken);
      try {
        const reply = await sendTurn(provider, SYSTEM_PROMPT, sent, TOOLS, {
          maxOutputTokens: b.reply,
          signal,
          sink: {
            text: (delta) => emit({ type: 'text', delta }),
            thinking: (delta) => emit({ type: 'thinking', delta }),
            toolStart: (index, _id, name) => emit({ type: 'tool_start', index, name }),
            toolArgs: () => {},
          },
        });
        // The provider's own count calibrates the next estimate.
        if (reply.usage.inputTokens > 200) {
          const ratio = (this.fixedChars() + estimateChars(sent, this.charsPerToken)) / reply.usage.inputTokens;
          if (ratio > 1 && ratio < 10) this.charsPerToken = ratio;
          emit({ type: 'context', used: reply.usage.inputTokens + reply.usage.outputTokens, window: b.window });
        }
        return reply;
      } catch (error) {
        lastError = error;
        if (!(error instanceof AIProviderError)) throw error;
        // Too long for the server: it often says how long it can take. Compact and try once more.
        const overflow = error.kind === 'http' && (error.status === 400 || error.status === 413 || error.status === 422 || error.status === 500) ? overflowWindow(`${error.message} ${error.detail ?? ''}`) : { overflow: false, tokens: null };
        if (overflow.overflow && !overflowRetried) {
          overflowRetried = true;
          if (overflow.tokens && overflow.tokens < b.window) {
            this.windowOverride = { key: `${provider.id}|${provider.modelId}`, tokens: overflow.tokens };
            this.options.onWindow?.(overflow.tokens);
          }
          emit({ type: 'status', message: `The prompt was too long for the model${overflow.tokens ? ` (its window is ${formatTokens(overflow.tokens)} tokens)` : ''}; compacting and retrying` });
          await this.compact(provider, this.budget(provider), emit, signal);
          continue;
        }
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
    let planThisRun = false;
    this.checkedRoots.clear();
    let nudges = 0;
    try {
      for (let step = 1; step <= MAX_STEPS; step++) {
        signal?.throwIfAborted();
        await this.fit(provider, emit, signal);
        const reply = await this.request(provider, emit, signal);
        emit({ type: 'usage', usage: reply.usage });
        this.turns.push({ role: 'assistant', text: reply.text, calls: reply.calls, anthropicContent: provider.type === 'anthropic' ? reply.anthropicContent : undefined });
        if (!reply.calls.length) {
          if (reply.truncated) emit({ type: 'status', message: 'The reply was cut off at the output limit.' });
          // Stopping short of the goal: ask once or twice to carry on.
          const unfinished = this.unfinished(planThisRun);
          if (unfinished && nudges < MAX_NUDGES && !reply.truncated) {
            nudges++;
            emit({ type: 'nudge', message: unfinished.split('\n')[0] });
            this.turns.push({ role: 'user', text: `[bot.computer] ${unfinished}`, automatic: true });
            continue;
          }
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
          // softn_check keeps the record of failing apps current, as the automatic check does.
          if (result.check) {
            this.checkedRoots.add(result.check.root);
            if (result.check.ok) this.failingApps.delete(result.check.root);
            else this.failingApps.set(result.check.root, result.check.text);
          }
          if (call.name === 'update_plan' && !result.isError) {
            this.plan = readPlan(call.input);
            planThisRun = true;
            emit({ type: 'plan', plan: this.plan });
          }
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
