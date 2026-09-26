import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { TaskQueue } from '../../src/agent/queue';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { OPENAI, fakeProvider, type Step } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const tick = () => new Promise((r) => setTimeout(r, 5));

describe('the task queue', () => {
  it('runs in order, at most `limit` at a time', async () => {
    let limit = 1;
    const queue = new TaskQueue(() => limit);
    const log: string[] = [];
    const job = (name: string) => async () => {
      log.push(`start ${name}`);
      await tick();
      log.push(`end ${name}`);
      return name;
    };
    expect(await Promise.all([queue.run(job('a')), queue.run(job('b'))])).toEqual(['a', 'b']);
    expect(log).toEqual(['start a', 'end a', 'start b', 'end b']);
    limit = 2;
    log.length = 0;
    await Promise.all([queue.run(job('c')), queue.run(job('d'))]);
    expect(log.slice(0, 2)).toEqual(['start c', 'start d']);
  });

  it('a stopped run takes its waiting tasks out of the queue', async () => {
    const queue = new TaskQueue(() => 1);
    const controller = new AbortController();
    let release!: () => void;
    const first = queue.run(() => new Promise<string>((r) => (release = () => r('first'))));
    const second = queue.run(async () => 'second', controller.signal);
    controller.abort();
    await expect(second).rejects.toThrow('Stopped');
    release();
    expect(await first).toBe('first');
    await tick();
    expect(queue.size).toEqual({ running: 0, waiting: 0 });
  });
});

/**
 * A model that answers the main agent and each sub-agent by who is asking
 * (the sub-agent's system prompt names its task).
 */
function routed(main: Step[], sub: (task: string, n: number, body: Record<string, unknown>) => Step) {
  let m = 0;
  const counts = new Map<string, number>();
  const bodies: Array<{ who: string; body: Record<string, unknown> }> = [];
  const respond = (body: Record<string, unknown>): Step => {
    const system = JSON.stringify((body.messages as Array<{ role: string; content: unknown }>)[0]);
    const task = /Your task: ([^"\\]+)/.exec(system)?.[1];
    bodies.push({ who: task ?? 'main', body });
    if (!task) return main[m++] ?? { text: 'main: out of script' };
    const n = (counts.get(task) ?? 0) + 1;
    counts.set(task, n);
    return sub(task, n, body);
  };
  return { script: Array.from({ length: 60 }, () => respond), bodies };
}

function setup(provider: ProviderConfig, parallel: number) {
  const vfs = new Vfs();
  const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'Project: two pages', subAgents: () => ({ contextTokens: 16_000, parallel }) });
  const events: AgentEvent[] = [];
  return { vfs, agent, events, emit: (e: AgentEvent) => events.push(e) };
}

const plan = { goal: 'Two pages', items: [{ text: 'Write page A', status: 'pending' }, { text: 'Write page B', status: 'pending' }, { text: 'Check both', status: 'pending' }] };
const delegate = {
  name: 'delegate',
  input: {
    tasks: [
      { title: 'Page A', instructions: 'Write pages/a.txt saying A.', plan_step: 1 },
      { title: 'Page B', instructions: 'Write pages/b.txt saying B.', plan_step: 2 },
    ],
  },
};
const worker = (task: string, n: number): Step =>
  n === 1 ? { calls: [{ name: 'write_file', input: { path: `pages/${task.slice(-1).toLowerCase()}.txt`, content: task.slice(-1) } }] } : { text: `Wrote ${task}.` };

describe('sub-agents', () => {
  it('run each task in a fresh agent with fewer tools, through the queue, and report back', async () => {
    const provider: ProviderConfig = { ...OPENAI, id: 'sub-1' };
    const { script, bodies } = routed(
      [
        { calls: [{ name: 'update_plan', input: plan }, delegate] },
        { calls: [{ name: 'update_plan', input: { ...plan, items: plan.items.map((i) => ({ ...i, status: 'done' })) } }] },
        { text: 'Both pages are written.' },
      ],
      worker,
    );
    fakeProvider('openai', script);
    const { vfs, agent, events, emit } = setup(provider, 1);
    await agent.run('write two pages', emit);
    expect(vfs.readText('/pages/a.txt')).toBe('A');
    expect(vfs.readText('/pages/b.txt')).toBe('B');
    const tasks = events.filter((e) => e.type === 'agent_task') as Array<Extract<AgentEvent, { type: 'agent_task' }>>;
    // One at a time: B starts after A is done.
    const order = tasks.filter((t) => t.state !== 'running' || t.activity === 'starting').map((t) => `${t.title} ${t.state}`);
    expect(order).toEqual(['Page A queued', 'Page A running', 'Page B queued', 'Page A done', 'Page B running', 'Page B done']);
    expect(tasks.some((t) => t.state === 'running' && t.activity === 'write_file pages/a.txt')).toBe(true);
    // The plan followed the tasks.
    const plans = events.filter((e) => e.type === 'plan') as Array<Extract<AgentEvent, { type: 'plan' }>>;
    expect(plans.some((p) => p.plan.items[0].status === 'done' && p.plan.items[1].status === 'active')).toBe(true);
    // A sub-agent gets its task, the project, and no delegate or update_plan.
    const sub = bodies.find((b) => b.who === 'Page A')!.body;
    const tools = (sub.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
    expect(tools).toContain('write_file');
    expect(tools).not.toContain('delegate');
    expect(tools).not.toContain('update_plan');
    expect(JSON.stringify(sub.messages)).toContain('Write pages/a.txt saying A.');
    // Its instructions do not mention tools it does not have; the main agent's do.
    const system = (b: Record<string, unknown>) => JSON.stringify((b.messages as unknown[])[0]);
    expect(system(sub)).not.toMatch(/update_plan|delegate/);
    expect(system(bodies.find((b) => b.who === 'main')!.body)).toContain('Plan first');
    expect(JSON.stringify(sub.messages)).toContain('Project: two pages');
    // The main agent's context holds the reports, not the sub-agents' work.
    const report = (agent.turns.find((t) => t.role === 'tool' && t.results.some((r) => r.name === 'delegate')) as Extract<(typeof agent.turns)[number], { role: 'tool' }>).results.find((r) => r.name === 'delegate')!;
    expect(report.content).toContain('2 tasks: 2 done.');
    expect(report.content).toContain('### Task 1: Page A (done)\nWrote Page A.\nFiles changed: pages/a.txt');
    expect(JSON.stringify(agent.turns)).not.toContain('"write_file"');
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Both pages are written.' });
  });

  it('run side by side when the provider allows more than one', async () => {
    const provider: ProviderConfig = { ...OPENAI, id: 'sub-2' };
    const { script } = routed([{ calls: [delegate] }, { text: 'done' }], worker);
    fakeProvider('openai', script);
    const { events, agent, emit } = setup(provider, 2);
    await agent.run('write two pages', emit);
    const states = (events.filter((e) => e.type === 'agent_task') as Array<Extract<AgentEvent, { type: 'agent_task' }>>).filter((t) => t.activity === 'starting' || t.state === 'done').map((t) => `${t.title} ${t.state}`);
    expect(states.slice(0, 2)).toEqual(['Page A running', 'Page B running']);
  });

  it('a sub-agent cannot delegate, and a bad task list is refused', async () => {
    const provider: ProviderConfig = { ...OPENAI, id: 'sub-3' };
    const { script } = routed(
      [{ calls: [{ name: 'delegate', input: { tasks: [] } }] }, { calls: [{ name: 'delegate', input: { tasks: [{ title: 'Nest', instructions: 'Try to delegate.' }] } }] }, { text: 'ok' }],
      (_task, n) => (n === 1 ? { calls: [{ name: 'delegate', input: { tasks: [{ title: 'x', instructions: 'y' }] } }] } : { text: 'I could not delegate.' }),
    );
    fakeProvider('openai', script);
    const { events, agent, emit } = setup(provider, 1);
    await agent.run('go', emit);
    const results = (events.filter((e) => e.type === 'tool_result') as Array<Extract<AgentEvent, { type: 'tool_result' }>>).map((e) => e.result);
    expect(results[0].isError).toBe(true);
    expect(results[0].content).toContain('tasks is empty');
    expect(results[1].content).toContain('I could not delegate.');
  });
});
