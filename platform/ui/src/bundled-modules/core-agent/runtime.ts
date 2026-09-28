/**
 * "Ask the Agent": the task goes to OAIY Desktop (`POST /api/agent/tasks`),
 * which hands it to the agent in OAIY's window and answers with its reply.
 * In OAIY's flow editor the page's own desktop address and token reach it;
 * run by the desktop (the CLI), the host's `ask_agent` does, with the
 * credential the desktop gave it.
 */
import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

/** How long the first request waits for the answer before the task is asked after. */
const FIRST_WAIT = 240;

interface Outcome {
  id?: string;
  status?: string;
  reply?: string;
  error?: string | { message?: string };
}

/** The task as the agent reads it: the input (or the setting, with {{input}} filled), and the context. */
export function taskText(input: unknown, written: string, context: unknown): string {
  const text = (v: unknown) => (v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v, null, 1));
  const given = text(input).trim();
  const task = written.trim() ? written.replace(/\{\{\s*input\s*\}\}/g, given).trim() : given;
  const extra = text(context).trim();
  return extra ? `${task}\n\nWhat the flow gives with it:\n${extra}` : task;
}

function createAgentMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {
  async function ask(input: unknown, written: string, context: unknown, conversation: string, waitSeconds: number, nodeId: string): Promise<string> {
    const task = taskText(input, written, context);
    if (!task) throw new Error('Ask the Agent: the task is empty (connect the Task input, or write it in the node)');
    ctx.onNodeStatus?.(nodeId, 'running');
    ctx.log('info', `[Agent] Asking the agent (${conversation}): ${task.slice(0, 120)}${task.length > 120 ? '…' : ''}`);
    const body = { task, from: conversation, waitSeconds };
    let outcome: Outcome;
    const desktop = (globalThis as { __OAIY_DESKTOP__?: { origin?: string; token?: string } }).__OAIY_DESKTOP__;
    if (desktop?.origin && desktop.token) {
      const headers = { authorization: `Bearer ${desktop.token}`, 'content-type': 'application/json' };
      const until = Date.now() + waitSeconds * 1000;
      // The request waits a few minutes at most (clients give up on headers after five); then the task is asked after.
      const resp = await fetch(`${desktop.origin}/api/agent/tasks`, { method: 'POST', headers, body: JSON.stringify({ ...body, waitSeconds: Math.min(waitSeconds, FIRST_WAIT) }), signal: ctx.abortSignal });
      outcome = (await resp.json().catch(() => ({}))) as Outcome;
      if (!resp.ok) throw new Error(`the agent did not take the task: ${typeof outcome.error === 'object' ? outcome.error?.message : outcome.error ?? `HTTP ${resp.status}`}`);
      while (outcome.status === 'pending' && outcome.id && Date.now() < until) {
        await new Promise((r) => setTimeout(r, 2000));
        ctx.abortSignal?.throwIfAborted();
        const again = await fetch(`${desktop.origin}/api/agent/tasks/${encodeURIComponent(outcome.id)}`, { headers, signal: ctx.abortSignal });
        if (again.ok) outcome = (await again.json().catch(() => outcome)) as Outcome;
      }
    } else if (ctx.tauri) {
      outcome = await ctx.tauri.invoke<Outcome>('ask_agent', body);
    } else {
      throw new Error("Ask the Agent runs in the OAIY app, whose agent does the task");
    }
    if (outcome.status === 'done') {
      ctx.onNodeStatus?.(nodeId, 'completed');
      return String(outcome.reply ?? '');
    }
    ctx.onNodeStatus?.(nodeId, 'error');
    if (outcome.status === 'pending') throw new Error(`the agent had not answered after ${Math.round(waitSeconds / 60)} min (task ${outcome.id ?? ''}); it may still be working on it`);
    const why = typeof outcome.error === 'object' ? outcome.error?.message : outcome.error;
    throw new Error(`the agent could not do the task: ${why || 'no reason given'}`);
  }

  return { ask: ask as RuntimeMethod };
}

const CoreAgentRuntime: RuntimeModule = {
  name: 'AgentTask',
  createMethods: createAgentMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default CoreAgentRuntime;
