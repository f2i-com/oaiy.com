/**
 * The agent loop: send the conversation, run the tools the model asks for,
 * send their results back, until it answers without asking for anything.
 *
 * The structure follows softn Studio's runAgent (Apache-2.0): rate limits and
 * timeouts are retried, a run has a step cap, the same failure three times in
 * a row stops it, and old tool output is trimmed as the conversation grows.
 */
import type { NetGate } from '../gate/netgate';
import { AIProviderError } from './providers/aiProvider';
import { DEFAULT_COMPACT_AT, budgetFor, contextWindow, formatTokens, outputLimit, overflowWindow } from './context';
import { normalizePath, type Vfs } from '../vfs/vfs';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Attachment, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { MAIN_AGENT_ONLY, TOOLS, checkApp, readPlan, readTasks, runTool, type Plan, type SoftnHost, type ToolContext } from './tools';
import { queueFor } from './queue';
import type { ToolSpec } from './protocol';
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
  /** A sub-agent's task: waiting in the queue, running (with what it is doing), done or failed. */
  | { type: 'agent_task'; callId: string; id: string; title: string; state: 'queued' | 'running' | 'done' | 'failed'; activity?: string; result?: string }
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

/** Model requests one run may make: enough to see a big plan through. */
export const MAX_STEPS = 200;
/** A sub-agent's step limit: a task is smaller than a request. */
const SUB_AGENT_STEPS = 40;
/** Sub-agent tasks one run may start. */
const MAX_TASKS_PER_RUN = 24;

/** For the agent that plans (the main one): plan before changing anything. */
const PLAN_GUIDE = `

Plan first: when you are given a task (anything that will change files: building or changing an app, a feature, a fix across files), call update_plan with the goal and 3 to 8 concrete steps that break the task down before you change anything, and update it as each step starts and finishes; the user watches that checklist. Then carry the plan out step by step. The task is done when every step is done and checked.`;

const DELEGATE_GUIDE = `

Sub-agents: for a big task that splits into independent parts, hand the parts to sub-agents with delegate. Each gets a fresh, smaller context and only the instructions you give it, so write each task to stand on its own (what to do, which files it may change, what to report) and keep tasks on separate files. Plan first, give each task the plan step it completes, then check and join the results yourself. Small tasks are quicker to do directly.`;

const SUB_AGENT_ROLE = `

You are a sub-agent: the main agent gave you one task, below. Do that task and nothing else; other agents may be changing other files at the same time, so change only the files your task is about. You cannot ask the user anything: decide sensibly and say what you assumed. When the task is done (and checked, for an app), reply with a short report for the main agent: what you did, the files you changed, how you checked it, and anything unfinished or wrong.`;
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
/** A run that stops short of its goal is asked to carry on at most this many times… */
const MAX_NUDGES = 6;
/** …and not again after this many in a row with no progress (no step done, no file changed). */
const MAX_IDLE_NUDGES = 2;
/** The same automatic-check errors this many times in a row stop the run. */
const SAME_CHECK_LIMIT = 3;
/** Images stay in the conversation for this many image-bearing turns. */
const KEEP_IMAGE_TURNS = 3;

export const SYSTEM_PROMPT = `You are bot.computer, a coding agent that runs entirely inside the user's web browser.

The user's project lives in a virtual filesystem in the browser; "/" is the project root and there is nothing outside it. Work with the tools:
- list_files, read_file, grep, glob to look around; read a file before you edit or replace it.
- edit_file for changes to existing files (exact, unique matches), write_file for new files or full rewrites.
- code_run to compute, test ideas, or process data in JavaScript or Python (a Zipp VM sandbox in a Web Worker).
- sandbox_shell for shell-style work (an emulated bash-like shell on the same sandbox): run project scripts with node or python, search and transform files (grep, find, sed, awk, jq, diff/patch), pack and unpack archives (tar, zip, gzip), and keep history with git (a local repository in .git/; no remotes, so no push, pull or clone). There is no real operating system: npm install, pip install, compilers and other native programs do not exist. Do not pretend to run them.
- SoftN apps: a SoftN app is a folder whose manifest.json names a .ui page as "main" (with ui/*.ui pages and logic/*.logic or .py). A project can hold several, each in its own folder: to rebuild or learn from an existing app, read its files and write the new one in another folder. A .softn the user attaches is unpacked into its own folder (the original stays in uploads/, and softn_import unpacks any .softn in the project): when they ask for changes, edit that folder; when they ask to recreate, redo or base something on it, write a new app in a new folder and leave the original as it is. The SoftN reference is in your tools, so do not guess the language: softn_docs with no arguments gives the map, topic "guide" is the writing guide (read it before your first app), search finds how something is done across the guides, the components and the example apps; softn_components gives exact props and events; softn_examples has complete working apps to read or copy. Keep manifest.json true. After each step that changes an app, bot.computer checks it automatically (its files, then a real render) and adds the outcome to that step's result: when it reports errors, fix them before anything else. softn_check checks on demand; softn_inspect shows what the page displays; softn_interact uses the app like a person (click, fill, select, press keys) and reports errors the app raises, so test that the app works, not just that it renders. The user watches the app in a live preview as you build it, and can export any app folder as a .softn file.
- web_fetch, curl, fetch() go through the user's network gate (/internet) and, from a browser, only reach sites that allow cross-origin requests. If the gate refuses a host, say so; the user decides whether to allow it.

Work toward the goal on your own until it is reached, without stopping to ask for permission; ask only when you truly cannot decide something yourself. You are done when the work is done and checked (for an app: it renders without errors and softn_interact shows it working), not before.

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
  /** The tools this agent may use (default: all of them). */
  tools?: ToolSpec[];
  /** Added to the system prompt: a sub-agent's instructions. */
  role?: string;
  /** Model requests one run may make (default MAX_STEPS). */
  maxSteps?: number;
  /** How sub-agents work: the context each gets, and how many share the model at once. */
  subAgents?: () => { contextTokens: number; parallel: number };
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

/** A file that belongs to an app at the project root (its manifest, pages, logic, data, assets). */
function isAppFile(path: string): boolean {
  return /^(manifest\.json|permission\.json|ui\/|logic\/|xdb\/|assets\/|server\/)/.test(path);
}

/**
 * The conversation in a shape every provider accepts: each tool call answered
 * (a run stopped mid-step leaves calls without results), no empty assistant
 * turns, and no two user turns in a row (strict chat templates refuse them).
 */
export function wellFormed(turns: Turn[]): Turn[] {
  const out: Turn[] = [];
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (turn.role === 'assistant' && !turn.text.trim() && !turn.calls.length && !turn.anthropicContent?.length) continue;
    if (turn.role === 'tool') {
      const prev = out[out.length - 1];
      if (prev?.role !== 'assistant' || !prev.calls.length) continue;
      const byId = new Map(turn.results.map((r) => [r.id, r]));
      out.push({ role: 'tool', results: prev.calls.map((c) => byId.get(c.id) ?? { id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call finished)', isError: true }) });
      continue;
    }
    const prev = out[out.length - 1];
    if (prev?.role === 'assistant' && prev.calls.length) {
      out.push({ role: 'tool', results: prev.calls.map((c) => ({ id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call ran)', isError: true })) });
    }
    const last = out[out.length - 1];
    if (turn.role === 'user' && last?.role === 'user') {
      out[out.length - 1] = { ...last, text: `${last.text}\n\n${turn.text}`, images: [...(last.images ?? []), ...(turn.images ?? [])], summary: last.summary || turn.summary };
      continue;
    }
    out.push(turn);
  }
  const last = out[out.length - 1];
  if (last?.role === 'assistant' && last.calls.length) out.push({ role: 'tool', results: last.calls.map((c) => ({ id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call ran)', isError: true })) });
  return out;
}

/** The file system as one agent sees it: its own writes are reported to it, nobody else's. */
function trackedVfs(vfs: Vfs, record: (path: string) => void): Vfs {
  return new Proxy(vfs, {
    get(target, prop) {
      const value = Reflect.get(target, prop, target) as unknown;
      if (typeof value !== 'function') return value;
      const fn = value as (...args: unknown[]) => unknown;
      if (prop === 'writeFile' || prop === 'mkdir' || prop === 'remove') {
        return (path: string, ...rest: unknown[]) => {
          const result = fn.call(target, path, ...rest);
          record(normalizePath(path));
          return result;
        };
      }
      if (prop === 'copy') {
        return (from: string, to: string, ...rest: unknown[]) => {
          const result = fn.call(target, from, to, ...rest);
          record(normalizePath(to));
          return result;
        };
      }
      if (prop === 'rename') {
        return (from: string, to: string) => {
          const result = fn.call(target, from, to);
          record(normalizePath(from));
          record(normalizePath(to));
          return result;
        };
      }
      return fn.bind(target);
    },
  });
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
  // Still over: the recent turns hold something huge (a long read, a big search). Cut their tool output too, the latest least.
  for (const limit of [8000, 3000, 1000]) {
    if (estimateChars(out) <= maxChars) break;
    out = out.map((t, i): Turn => (t.role === 'tool' && i < out.length - 1 ? { role: 'tool', results: t.results.map((r) => (r.content.length > limit ? { ...r, content: `${r.content.slice(0, limit)}\n[cut to fit the model's context]` } : r)) } : t));
    const lastTool = out[out.length - 1];
    if (estimateChars(out) > maxChars && lastTool?.role === 'tool') out[out.length - 1] = { role: 'tool', results: lastTool.results.map((r) => (r.content.length > limit * 2 ? { ...r, content: `${r.content.slice(0, limit * 2)}\n[cut to fit the model's context: read less at a time]` } : r)) };
  }
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
  /** An output limit the server stated, for that provider and model. */
  private outputCap: { key: string; tokens: number } | null = null;
  /** Models (provider|model) that refused images. */
  private noImages = new Set<string>();
  readonly toolContext: ToolContext;
  /** Set during a run: records a path this agent's tools wrote. */
  private onWrite: ((path: string) => void) | null = null;
  running = false;

  constructor(private readonly options: AgentOptions) {
    // Writes made through this agent's tools are its changes; another agent's (or the person's) are not.
    this.toolContext = { vfs: trackedVfs(options.vfs, (path) => this.onWrite?.(path)), gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} }, softn: options.softn };
  }

  reset(): void {
    this.turns = [];
    this.noImages.clear();
    this.toolContext.images = true;
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

  /** False when the current model has refused images: they are left out. */
  private get imagesAccepted(): boolean {
    const p = this.options.provider();
    return !p || !this.noImages.has(`${p.id}|${p.modelId}`);
  }

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

  private get tools(): ToolSpec[] {
    return this.options.tools ?? TOOLS;
  }

  private get canPlan(): boolean {
    return this.tools.some((t) => t.name === 'update_plan');
  }

  private get canDelegate(): boolean {
    return this.tools.some((t) => t.name === 'delegate');
  }

  private get systemPrompt(): string {
    return `${SYSTEM_PROMPT}${this.canPlan ? PLAN_GUIDE : ''}${this.canDelegate ? DELEGATE_GUIDE : ''}${this.options.role ?? ''}`;
  }

  /** The prompt's fixed part: the system prompt and the tool definitions. */
  private fixedChars(): number {
    return this.systemPrompt.length + JSON.stringify(this.tools).length;
  }

  /** Apps checked in the latest run, and whether each passed its last check. */
  checkedApps(): Array<{ root: string; failing: string | null }> {
    return [...this.checkedRoots].map((root) => ({ root, failing: this.failingApps.get(root) ?? null }));
  }

  private tasksThisRun = 0;

  /**
   * Run a delegate call: each task in its own sub-agent, through the
   * provider's queue, and one report back. Plan steps named by the tasks
   * follow them (active, then done), and what the sub-agents found about
   * apps joins this agent's record.
   */
  private async delegate(call: ToolCall, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    const provider = this.options.provider();
    let tasks: ReturnType<typeof readTasks>;
    try {
      tasks = readTasks(call.input);
      if (!provider) throw new Error('no AI provider');
      if (this.tasksThisRun + tasks.length > MAX_TASKS_PER_RUN) throw new Error(`this request has already started ${this.tasksThisRun} tasks; at most ${MAX_TASKS_PER_RUN} per request. Do the rest directly.`);
    } catch (error) {
      return { id: call.id, name: call.name, content: `Error: ${(error as Error).message}`, isError: true };
    }
    this.tasksThisRun += tasks.length;
    const settings = () => this.options.subAgents?.() ?? { contextTokens: 32_000, parallel: provider.type === 'local' ? 1 : 3 };
    const queue = queueFor(`${provider.id}|${provider.modelId}`, () => settings().parallel);
    const setStep = (step: number | undefined, status: 'active' | 'done') => {
      if (!step || !this.plan || !this.plan.items[step - 1] || this.plan.items[step - 1].status === 'done') return;
      this.plan = { ...this.plan, items: this.plan.items.map((item, i) => (i === step - 1 ? { ...item, status } : item)) };
      emit({ type: 'plan', plan: this.plan });
    };
    const outcomes = await Promise.all(
      tasks.map(async (task, i) => {
        const id = `${call.id}#${i}`;
        const base = { type: 'agent_task' as const, callId: call.id, id, title: task.title };
        emit({ ...base, state: 'queued' });
        try {
          const outcome = await queue.run(
            () => this.runSubAgent(task, provider, settings().contextTokens, (activity) => emit({ ...base, state: 'running', activity }), emit, signal),
            signal,
            () => {
              emit({ ...base, state: 'running', activity: 'starting' });
              setStep(task.planStep, 'active');
            },
          );
          emit({ ...base, state: outcome.ok ? 'done' : 'failed', result: outcome.text });
          if (outcome.ok) setStep(task.planStep, 'done');
          return { task, ...outcome };
        } catch (error) {
          const text = signal?.aborted ? 'stopped before it finished' : (error as Error).message;
          emit({ ...base, state: 'failed', result: text });
          return { task, ok: false, text, files: [] as string[] };
        }
      }),
    );
    const done = outcomes.filter((o) => o.ok).length;
    const report = outcomes.map((o, i) => [
      `### Task ${i + 1}: ${o.task.title} (${o.ok ? 'done' : 'not finished'})`,
      o.text.length > 4000 ? `${o.text.slice(0, 4000)}\n[report cut]` : o.text || '(no report)',
      o.files.length ? `Files changed: ${o.files.join(', ')}` : 'Files changed: none',
    ].join('\n')).join('\n\n');
    const failing = [...this.failingApps.keys()].filter((root) => this.checkedRoots.has(root));
    return {
      id: call.id,
      name: call.name,
      isError: done === 0,
      content: `${tasks.length} task${tasks.length > 1 ? 's' : ''}: ${done} done${done < tasks.length ? `, ${tasks.length - done} not finished` : ''}.\n\n${report}${failing.length ? `\n\nApps still failing their check: ${failing.join(', ')}.` : ''}\n\nCheck the results fit together before you finish.`,
    };
  }

  /** One task in a fresh agent with a smaller context; its report, and the files it changed. */
  private async runSubAgent(
    task: { title: string; instructions: string },
    provider: ProviderConfig,
    contextTokens: number,
    activity: (text: string) => void,
    emit: (e: AgentEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ ok: boolean; text: string; files: string[] }> {
    const child = new Agent({
      ...this.options,
      provider: () => provider,
      tools: this.tools.filter((t) => !MAIN_AGENT_ONLY.has(t.name)),
      role: `${SUB_AGENT_ROLE}\n\nYour task: ${task.title}`,
      maxSteps: SUB_AGENT_STEPS,
      maxContext: contextTokens,
      subAgents: undefined,
    });
    const calls = new Map<string, ToolCall>();
    const files = new Set<string>();
    let final = '';
    let failure = '';
    const goal = this.plan?.goal ? `\n\n(The main agent's goal, for context: ${this.plan.goal})` : '';
    await child.run(`${task.instructions}${goal}`, (e) => {
      if (e.type === 'tool_call') {
        calls.set(e.call.id, e.call);
        const arg = ['path', 'command', 'pattern', 'app', 'url'].map((k) => e.call.input[k]).find((v) => typeof v === 'string') as string | undefined;
        activity(`${e.call.name}${arg ? ` ${arg.split('\n')[0].slice(0, 80)}` : ''}`);
      } else if (e.type === 'tool_result') {
        const c = calls.get(e.result.id);
        if (c && !e.result.isError && /^(write_file|edit_file|delete_file)$/.test(c.name) && typeof c.input.path === 'string') files.add(c.input.path.replace(/^\/+/, ''));
      } else if (e.type === 'check') {
        emit(e);
      } else if (e.type === 'compact') {
        activity(`compacted its context (${e.turns} turns)`);
      } else if (e.type === 'done') {
        final = e.text;
      } else if (e.type === 'error') {
        failure = e.message;
      }
    }, signal);
    // What the sub-agent learned about apps is now this agent's to act on.
    for (const app of child.checkedApps()) {
      this.checkedRoots.add(app.root);
      if (app.failing) this.failingApps.set(app.root, app.failing);
      else this.failingApps.delete(app.root);
    }
    if (signal?.aborted) return { ok: false, text: 'stopped before it finished', files: [...files] };
    return { ok: !failure, text: failure ? `${final ? `${final}\n` : ''}It stopped: ${failure}` : final, files: [...files] };
  }

  private budget(provider: ProviderConfig): { window: number; fixed: number; prompt: number; reply: number } {
    const window = this.window(provider);
    const fixed = Math.ceil(this.fixedChars() / this.charsPerToken);
    const b = { window, fixed, ...budgetFor(window, fixed) };
    const cap = this.outputCap?.key === `${provider.id}|${provider.modelId}` ? this.outputCap.tokens : null;
    if (cap && cap < b.reply) {
      b.prompt += b.reply - cap;
      b.reply = cap;
    }
    return b;
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
    // Too small to hold the instructions and tools with room to work: say so, rather than fail in circles.
    if (b.prompt < 1024) throw new Error(`The model's context window (${formatTokens(b.window)} tokens) is too small for bot.computer: its instructions and tools alone take about ${formatTokens(b.fixed)}. Give the model a bigger window (Ollama: OLLAMA_CONTEXT_LENGTH=16384 or more; then Detect in Settings), or set the size in Settings if the detected one is wrong.`);
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
    // One turn on its own (a huge first message): there is nothing before it to summarize, and
    // summarizing it would replace the request itself; trimming handles it.
    if (keepFrom >= view.length) return;
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
      const sent = wellFormed(trimmed(this.view(), this.imagesAccepted, b.prompt * this.charsPerToken));
      try {
        const reply = await sendTurn(provider, this.systemPrompt, sent, this.tools, {
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
        const said = `${error.message} ${error.detail ?? ''}`;
        // A max_tokens the model cannot give: it usually says what it can.
        const limit = error.kind === 'http' ? outputLimit(said) : null;
        if (limit && limit < b.reply) {
          this.outputCap = { key: `${provider.id}|${provider.modelId}`, tokens: limit };
          emit({ type: 'status', message: `The model writes at most ${formatTokens(limit)} tokens per reply; retrying with that` });
          continue;
        }
        // A model without vision refuses image content: carry on in text (an image that is too big is not that).
        if (this.imagesAccepted && error.kind === 'http' && /image|vision|multimodal|image_url|content.*array/i.test(said) && !/too (large|big)|exceeds?|dimension|resolution|megapixel/i.test(said) && this.turns.some(hasImages)) {
          this.noImages.add(`${provider.id}|${provider.modelId}`);
          this.toolContext.images = false;
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
    this.onWrite = (path) => changes.push({ path, index: callIndex });
    this.toolContext.images = this.imagesAccepted;
    const unwatch = () => {
      this.onWrite = null;
    };
    // The results of the step in progress: kept if the run stops part-way.
    let stepResults: ToolResult[] = [];
    this.sameCheck = { signature: '', count: 0 };
    let planThisRun = false;
    let planNoted = false;
    // Progress, for deciding whether asking to carry on is still worth it.
    const changedThisRun = new Set<string>();
    let idleNudges = 0;
    let progressAtNudge = { done: -1, changed: -1 };
    this.checkedRoots.clear();
    this.tasksThisRun = 0;
    let nudges = 0;
    const maxSteps = this.options.maxSteps ?? MAX_STEPS;
    try {
      for (let step = 1; step <= maxSteps; step++) {
        signal?.throwIfAborted();
        await this.fit(provider, emit, signal);
        const reply = await this.request(provider, emit, signal);
        emit({ type: 'usage', usage: reply.usage });
        this.turns.push({ role: 'assistant', text: reply.text, calls: reply.calls, anthropicContent: provider.type === 'anthropic' ? reply.anthropicContent : undefined });
        if (!reply.calls.length) {
          if (reply.truncated) emit({ type: 'status', message: 'The reply was cut off at the output limit.' });
          // Stopping short of the goal: ask once or twice to carry on.
          const unfinished = this.unfinished(planThisRun);
          const progress = { done: this.plan?.items.filter((i) => i.status === 'done').length ?? 0, changed: changedThisRun.size };
          idleNudges = progress.done > progressAtNudge.done || progress.changed > progressAtNudge.changed ? 0 : idleNudges + 1;
          progressAtNudge = progress;
          if (unfinished && nudges < MAX_NUDGES && idleNudges < MAX_IDLE_NUDGES && !reply.truncated) {
            nudges++;
            emit({ type: 'nudge', message: unfinished.split('\n')[0] });
            this.turns.push({ role: 'user', text: `[bot.computer] ${unfinished}`, automatic: true });
            continue;
          }
          emit({ type: 'done', text: reply.text, steps: step });
          return;
        }
        const results: ToolResult[] = [];
        stepResults = results;
        changes = [];
        // A reply cut off at the output limit mid-call: the call is incomplete, so say why rather than run it.
        if (reply.truncated && reply.calls.some((c) => c.parseError)) {
          for (const call of reply.calls) {
            const result: ToolResult = { id: call.id, name: call.name, content: `Error: your reply was cut off at the output limit (about ${formatTokens(this.budget(provider).reply)} tokens) before this call was complete, so it did not run. Keep each call smaller: write a large file in parts (write_file with the first part, then edit_file to add the rest), and keep reasoning short.`, isError: true };
            results.push(result);
            emit({ type: 'tool_call', call });
            emit({ type: 'tool_result', result });
          }
          this.turns.push({ role: 'tool', results });
          continue;
        }
        for (const [index, call] of reply.calls.entries()) {
          signal?.throwIfAborted();
          callIndex = index;
          emit({ type: 'tool_call', call });
          const allowed = this.tools.some((t) => t.name === call.name);
          const result = !allowed
            ? { id: call.id, name: call.name, content: `Error: ${call.name} is not one of your tools`, isError: true }
            : call.name === 'delegate'
              ? await this.delegate(call, emit, signal)
              : await runTool(call, this.toolContext);
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
        for (const c of changes) changedThisRun.add(c.path);
        // A task: plan before going further (once, and not for a one-file fix).
        if (!planThisRun && !planNoted && this.canPlan && results.length) {
          const apps = findApps(this.options.vfs);
          const touchesApp = [...changedThisRun].some((p) => apps.some((r) => r === '' ? isAppFile(p) : p.startsWith(`${r}/`)));
          if (changedThisRun.size >= 2 || touchesApp) {
            planNoted = true;
            results[results.length - 1].content += '\n\n[bot.computer] This is a task with several steps, and there is no plan yet. Call update_plan now with the goal and the steps that break it down (mark what is already done), then carry on.';
          }
        }
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
      emit({ type: 'error', message: `The run reached its ${maxSteps}-step limit. Say "continue" to keep going.` });
    } catch (error) {
      if (signal?.aborted || (error instanceof AIProviderError && error.kind === 'cancelled')) {
        emit({ type: 'status', message: 'Stopped.' });
      } else {
        emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
      }
      // A turn left waiting for tool results cannot be sent again: keep the results
      // that came in (their changes happened) and mark the rest as not run.
      const last = this.turns[this.turns.length - 1];
      if (last?.role === 'assistant' && last.calls.length) {
        const done = new Map(stepResults.map((r) => [r.id, r]));
        this.turns.push({ role: 'tool', results: last.calls.map((c) => done.get(c.id) ?? { id: c.id, name: c.name, content: 'Error: the run stopped before this tool ran', isError: true }) });
      }
    } finally {
      unwatch();
      this.running = false;
    }
  }
}
