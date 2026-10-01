// A run that started calls is not pushed on by its plan while they are going on: the plan's "carry on with step 4" nudge and the
// status check it brings were each a request the call's replies waited behind (seen live on 1 Oct 2026). Only a run that
// started calls, only while they are going on, and only when they really started (not declined, refused, or texts). A run that
// ends with no words is still asked to say what it started.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent, type SessionTool, type ToolHook } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import mainSource from '../../src/main.ts?raw';
import { outreachTools } from '../../src/outreachTools';
import { setLocalCountry } from '../../src/phoneNumbers';
import { Vfs } from '../../src/vfs/vfs';
import { CAMPAIGN, JANE, agent, engine, runnerOf, world } from './callWorld';
import { fakeProvider } from './fakeProvider';

beforeAll(() => setLocalCountry('AU'));
afterEach(() => vi.unstubAllGlobals());

const PLAN = { goal: 'Confirm with Jane', items: [{ text: 'Call Jane', status: 'done' }, { text: 'Report her reply', status: 'active' }] };
const openai = { id: 'o', type: 'openai' as const, name: 'OpenAI', apiKey: 'k', modelId: 'gpt' };
const GOING = 'Your last reply had no words. The calls you started are going on';

/** A runner with a tool that starts calls (`startsCalls` says whether what it answered did), and a flag for whether its call is going on. */
function runnerWith(waiting: () => boolean, tool: Partial<SessionTool> & { name?: string } = {}, toolHooks: ToolHook[] = [], more: SessionTool[] = []) {
  const name = tool.name ?? 'start_outreach';
  return new Agent({
    vfs: new Vfs(),
    gate: new NetGate(),
    provider: () => openai,
    projectSummary: () => '',
    waitingOnCall: waiting,
    toolHooks,
    sessionTools: [{ spec: { name, description: 'Starts calls.', parameters: { type: 'object', properties: {} } }, run: async () => 'Started "x" (outreach out-1).', startsCalls: () => true, ...tool }, ...more],
  });
}
const startAndPlan = (name = 'start_outreach') => ({ calls: [{ name: 'update_plan', input: PLAN }, { name, input: {} }] });
const nudges = (events: AgentEvent[]) => events.filter((e): e is Extract<AgentEvent, { type: 'nudge' }> => e.type === 'nudge');

describe("a run waiting for its call's result is not pushed on", () => {
  it('it started calls, which are going on, with a plan step open: the run ends with no request more, and the result is one more request, when it comes', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'The call is going out now; I will report when it ends.' }]);
    const runner = runnerWith(() => true);
    const events: AgentEvent[] = [];
    await runner.run('Call Jane back about the price.', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'The call is going out now; I will report when it ends.' });
    // Its plan still has the step: the result, when it comes, is one more request, which carries on from it.
    expect(runner.plan?.items.map((i) => i.status)).toEqual(['done', 'active']);
    fakeProvider('openai', [{ text: 'Jane agreed.' }]);
    const next: string[] = [];
    await runner.run('[OAIY] Outreach "Jane" is finished.', (e) => e.type === 'done' && next.push(e.text));
    expect(next).toEqual(['Jane agreed.']);
    expect(JSON.stringify(runner.turns)).toContain('still has 1 open step');
  });

  it('the same without a plan to push, or one that is done: nothing to push either way', async () => {
    const fake = fakeProvider('openai', [{ calls: [{ name: 'start_outreach', input: {} }] }, { text: 'Calling now.' }]);
    await runnerWith(() => true).run('Call Jane.', () => {});
    expect(fake.bodies).toHaveLength(2);
  });

  it('it said what it would do next (a reply that announces work): not pushed to do it, while the call is going on', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: "I'll check on the call in a moment." }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane back about the price.', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
  });

  it('it started calls, but they are over: the plan pushes it on as before', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'Calling now.' }, { text: 'Still waiting.' }, { text: 'Nothing yet.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => false).run('Call Jane back about the price.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });

  it("another run's plan is its own: calls going on that this run did not start do not end it with steps open", async () => {
    const own = { goal: 'Tidy the notes', items: [{ text: 'Rename files', status: 'active' }, { text: 'Update the index', status: 'pending' }] };
    const fake = fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: own }] }, { text: 'Renamed them.' }, { text: 'Index updated.' }, { text: 'Done now.' }]);
    // The call is going on, but this run did not start it (and a tool of its that does not start calls does not make it so).
    const runner = runnerWith(() => true, { name: 'outreach_status', startsCalls: undefined });
    await runner.run('Tidy my notes while the call is going.', () => {});
    expect(fake.bodies.length).toBeGreaterThan(2);
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 2 open steps');
  });

  it('checking on a campaign (outreach_status) is not starting calls: the plan pushes on, and the check goes on being pushed', async () => {
    const fake = fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: PLAN }, { name: 'outreach_status', input: {} }] }, { text: 'The call is still going.' }, { text: 'Checking.' }, { text: 'Still.' }]);
    const runner = runnerWith(() => true, { name: 'outreach_status', startsCalls: undefined });
    await runner.run('How is the call going?', () => {});
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });

  it('a start that did not start calls (it said so: declined, refused, a list of texts) is not waited for', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'It did not start.' }, { text: 'Trying.' }, { text: 'Not yet.' }]);
    // The tool answered, but not that calls are going out.
    const runner = runnerWith(() => true, { startsCalls: () => false });
    await runner.run('Call Jane.', () => {});
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });

  it('a start that failed (the tool threw) is not waited for', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'It did not start.' }, { text: 'Trying.' }, { text: 'Not yet.' }]);
    const runner = runnerWith(() => true, { run: async () => { throw new Error('the phone is not set up'); } });
    await runner.run('Call Jane.', () => {});
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });

  it('the tool is asked what it started with what it was given and what it answered', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'start_outreach', input: { kind: 'call' } }] }, { text: 'Calling.' }]);
    const asked: Array<[Record<string, unknown>, string]> = [];
    await runnerWith(() => true, { run: async () => 'Started "y".', startsCalls: (input, result) => (asked.push([input, result]), true) }).run('Call.', () => {});
    expect(asked).toEqual([[{ kind: 'call' }, 'Started "y".']]);
  });

  it('waiting for the call skips the plan only: an app whose check fails still pushes the run on', () => {
    const runner = runnerWith(() => true) as unknown as { unfinished(plan: boolean, announced: boolean, waiting: boolean): string | null; failingApps: Map<string, string>; checkedRoots: Set<string>; plan: unknown };
    runner.plan = PLAN;
    runner.failingApps.set('/app', 'boom');
    runner.checkedRoots.add('/app');
    expect(runner.unfinished(true, false, true)).toContain('still reports errors');
    runner.failingApps.clear();
    expect(runner.unfinished(true, false, true)).toBeNull();
    expect(runner.unfinished(true, false, false)).toContain('open step');
  });
});

describe('a run that started calls and ends with no words says what it started', () => {
  it('a reply with no words, a plan step open, the call going on: one nudge to say what was started (not the plan), then it speaks and the run ends with words', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: '' }, { text: 'I have started the call to Jane; I will report when it ends.' }]);
    const runner = runnerWith(() => true);
    const events: AgentEvent[] = [];
    await runner.run('Call Jane back about the price.', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(3);
    expect(nudges(events)).toHaveLength(1);
    expect(nudges(events)[0].message).toContain(GOING);
    expect(JSON.stringify(fake.bodies[2])).not.toContain('Your plan still has');
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'I have started the call to Jane; I will report when it ends.' });
  });

  it('the same with no plan at all', async () => {
    const fake = fakeProvider('openai', [{ calls: [{ name: 'start_outreach', input: {} }] }, { text: '' }, { text: 'Calling Jane now.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane.', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(3);
    expect(nudges(events)).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Calling Jane now.' });
  });

  it('a model that alternates a tool call with an empty reply is asked at most twice in all, however much it does in between, and then the run ends', async () => {
    const status = { calls: [{ name: 'outreach_status', input: {} }] };
    const checker: SessionTool = { spec: { name: 'outreach_status', description: 'How the lists are going.', parameters: { type: 'object', properties: {} } }, run: async () => 'Hedge price answer: Jane is on the call.' };
    for (const first of [startAndPlan(), { calls: [{ name: 'start_outreach', input: {} }] }]) {
      const fake = fakeProvider('openai', [first, { text: '' }, status, { text: '' }, status, { text: '' }, status, { text: '' }, status, { text: '' }, status, { text: '' }, status, { text: '' }]);
      const events: AgentEvent[] = [];
      await runnerWith(() => true, {}, [], [checker]).run('Call Jane.', (e) => events.push(e));
      expect(nudges(events)).toHaveLength(2);
      expect(nudges(events).every((n) => n.message.startsWith(GOING))).toBe(true);
      // The call, an empty reply (the first ask), a check, an empty reply (the second ask), a check, an empty reply: then it ends.
      expect(fake.bodies).toHaveLength(6);
      expect(events.at(-1)).toMatchObject({ type: 'done', text: '' });
    }
  });

  it('the ask is accurate about what was said: one that spoke in an earlier reply is told its last reply had no words, and nothing about nothing said yet', async () => {
    fakeProvider('openai', [{ text: 'Starting the call to Jane now.', calls: [{ name: 'start_outreach', input: {} }] }, { text: '' }, { text: 'Done: Jane is being called.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane.', (e) => events.push(e));
    expect(nudges(events)).toHaveLength(1);
    expect(nudges(events)[0].message).toContain('Your last reply had no words');
    expect(nudges(events)[0].message).not.toMatch(/yet|nothing/i);
  });
  it('a reply of only spaces has no words either', async () => {
    fakeProvider('openai', [startAndPlan(), { text: '  \n ' }, { text: 'Calling Jane now.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane.', (e) => events.push(e));
    expect(nudges(events)).toHaveLength(1);
  });

  it('a model that stays silent is asked as often as main would have asked (the same bound), and the run then ends', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: '' }, { text: '' }, { text: '' }, { text: '' }, { text: '' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThanOrEqual(1);
    expect(nudges(events).length).toBeLessThanOrEqual(2);
    expect(fake.bodies.length).toBe(2 + nudges(events).length);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: '' });
  });

  it('words said, a plan step open, the call going on: no nudge at all', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'Calling Jane now.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true).run('Call Jane.', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
  });

  it('the call is over: a reply with no words is pushed by the plan, as on main', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: '' }, { text: 'Still waiting.' }, { text: 'Nothing yet.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => false).run('Call Jane.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(nudges(events)[0].message).not.toContain(GOING);
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });

  it('no call started (a plan and an empty reply): nothing of this, and as on main the run ends where main ends it', async () => {
    const own = { goal: 'Tidy the notes', items: [{ text: 'Rename files', status: 'active' }] };
    const fake = fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: own }] }, { text: '' }, { text: 'Renamed.' }, { text: 'Done.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true, { name: 'outreach_status', startsCalls: undefined }).run('Tidy my notes.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(nudges(events).map((n) => n.message).join(' ')).not.toContain(GOING);
    expect(fake.bodies.length).toBeGreaterThan(2);
  });
});

describe('the calls this run started are this run\'s', () => {
  it('a second, unrelated run on the same agent while the call is still live is pushed by its own plan, as on main', async () => {
    const own = { goal: 'Tidy the notes', items: [{ text: 'Rename files', status: 'active' }, { text: 'Update the index', status: 'pending' }] };
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'The call is going out now.' }]);
    const runner = runnerWith(() => true);
    await runner.run('Call Jane back about the price.', () => {});
    expect(fake.bodies).toHaveLength(2);
    const second = fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: own }] }, { text: 'Renamed them.' }, { text: 'Index updated.' }, { text: 'Done now.' }]);
    const events: AgentEvent[] = [];
    await runner.run('Tidy my notes while the call is going.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(JSON.stringify(second.bodies[2])).toContain('Your plan still has 2 open steps');
  });

  it('a tool that says whether it started calls but throws is taken to have started none (the start itself still succeeded)', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'Calling now.' }, { text: 'Still waiting.' }, { text: 'Nothing yet.' }]);
    const runner = runnerWith(() => true, { startsCalls: () => { throw new Error('cannot tell'); } });
    const events: AgentEvent[] = [];
    await runner.run('Call Jane.', (e) => events.push(e));
    const results = runner.turns.flatMap((t) => (t.role === 'tool' ? t.results : [])).filter((r) => r.name === 'start_outreach');
    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ isError: false, content: 'Started "x" (outreach out-1).' });
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });
});
describe("a flow of the person's in front of the tool", () => {
  const before: ToolHook = { tool: 'start_outreach', mode: 'before', flowName: 'Check the list', run: async () => '' };
  const instead: ToolHook = { tool: 'start_outreach', mode: 'instead', flowName: 'Our own start', run: async () => 'Started "Our own start" (a flow ran it).' };

  it('one that runs before it puts its note in front of the result, and the run still knows calls started', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'The call is going out now.' }]);
    const events: AgentEvent[] = [];
    const runner = runnerWith(() => true, {}, [before]);
    await runner.run('Call Jane.', (e) => events.push(e));
    expect(JSON.stringify(runner.turns)).toContain('Your flow \\"Check the list\\" ran before start_outreach');
    expect(fake.bodies).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
  });

  it('one that runs in its place: the tool did not run, so nothing is taken to have started, and the plan pushes as on main', async () => {
    const fake = fakeProvider('openai', [startAndPlan(), { text: 'Done.' }, { text: 'Still.' }, { text: 'Yes.' }]);
    const events: AgentEvent[] = [];
    await runnerWith(() => true, {}, [instead]).run('Call Jane.', (e) => events.push(e));
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(JSON.stringify(fake.bodies[2])).toContain('Your plan still has 1 open step');
  });
});

describe('what says that calls were started (outreachTools)', () => {
  function tools(w: ReturnType<typeof world>, approve = true, ready = '') {
    return outreachTools({
      engine: () => w.outreach,
      origin: () => ({ kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' }),
      ready: async () => ready,
      screening: async () => null,
      approve: async () => approve,
    });
  }
  const TEXTS = { ...CAMPAIGN, kind: 'text', name: 'Reminders', textTemplate: "Hi {first_name}, it's {receptionist} from {business}.", people: [{ name: 'Tom', number: '0498 765 432' }] };
  /** Runs `name` with `input` the way a run would, and says what the tool answered and whether it says calls started. */
  async function run(list: SessionTool[], name: string, input: Record<string, unknown>) {
    const tool = list.find((t) => t.spec.name === name)!;
    const result = await tool.run(input).catch((e: Error) => `Error: ${e.message}`);
    return { result, starts: !!tool.startsCalls?.(input, result) };
  }

  it('starting a list of calls does; a declined one, a refused one and a list of texts do not', async () => {
    engine(() => ({ text: 'Hi.' }));
    const w = world();
    await w.sessions.load();
    const started = await run(tools(w), 'start_outreach', CAMPAIGN);
    expect(started.result).toMatch(/^Started "Hedge price answer"/);
    expect(started.starts).toBe(true);
    expect((await run(tools(w, false), 'start_outreach', { ...CAMPAIGN, name: 'Declined' })).starts).toBe(false);
    expect((await run(tools(w, true, 'The phone is not on here'), 'start_outreach', { ...CAMPAIGN, name: 'Refused' })).starts).toBe(false);
    const texts = await run(tools(w), 'start_outreach', TEXTS);
    expect(texts.result).toMatch(/^Started "Reminders"/);
    expect(texts.starts).toBe(false);
  });

  it('resuming a paused list of calls does; a list of texts, one that is not paused and one that is not there do not', async () => {
    engine(() => ({ text: 'Hi.' }));
    const w = world({ phone: { holdsCalls: false } });
    await w.sessions.load();
    const list = tools(w);
    const plan = w.outreach.plan(CAMPAIGN, null);
    const textPlan = w.outreach.plan(TEXTS, null);
    if (typeof plan === 'string' || typeof textPlan === 'string') throw new Error('no plan');
    const calls = await w.outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    const texts = await w.outreach.create(textPlan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    expect((await run(list, 'outreach_resume', { id: calls.id })).starts).toBe(false);
    w.outreach.pause(calls.id);
    w.outreach.pause(texts.id);
    const resumedCalls = await run(list, 'outreach_resume', { id: calls.id });
    expect(resumedCalls.result).toMatch(/^Resumed "Hedge price answer"/);
    expect(resumedCalls.starts).toBe(true);
    const resumedTexts = await run(list, 'outreach_resume', { id: texts.id });
    expect(resumedTexts.result).toMatch(/^Resumed "Reminders"/);
    expect(resumedTexts.starts).toBe(false);
    expect((await run(list, 'outreach_resume', { id: 'out-nobody' })).starts).toBe(false);
  });

  it('nothing else of outreach says it started calls: checking, pausing, stopping and reading the results do not', () => {
    const w = world();
    expect(tools(w).filter((t) => t.startsCalls).map((t) => t.spec.name)).toEqual(['start_outreach', 'outreach_resume']);
  });
});

describe('a call this conversation started is going on (Outreach.callLiveFor)', () => {
  it('while it dials, rings or is on the call, and has just ended with its result not recorded: for that conversation only, and not for texts', async () => {
    engine(() => ({ text: 'Hi.' }));
    const w = world();
    await w.sessions.load();
    expect(w.outreach.callLiveFor('front-desk')).toBe(false);
    const plan = w.outreach.plan(CAMPAIGN, null);
    if (typeof plan === 'string') throw new Error(plan);
    const c = await w.outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    await w.outreach.tick();
    expect(w.outreach.callLiveFor('front-desk')).toBe(true);
    expect(w.outreach.callLiveFor('another-project')).toBe(false);
    const callId = w.outreach.campaigns[0].people[0].attempt!.callId!;
    await w.sessions.callEvent({ type: 'call.started', callId, from: JANE, direction: 'outbound' });
    expect(w.outreach.callLiveFor('front-desk')).toBe(true);
    for (const state of ['dialling', 'ringing', 'on_call', 'ended'] as const) {
      c.people[0].state = state;
      expect(w.outreach.callLiveFor('front-desk'), state).toBe(true);
    }
    for (const state of ['done', 'queued', 'waiting', 'skipped', 'sending', 'awaiting_reply'] as const) {
      c.people[0].state = state;
      expect(w.outreach.callLiveFor('front-desk'), state).toBe(false);
    }
    c.people[0].state = 'on_call';
    c.kind = 'text';
    expect(w.outreach.callLiveFor('front-desk')).toBe(false);
  });
});

describe('the runner, with its real tools and campaigns', () => {
  const planStep = { calls: [{ name: 'update_plan', input: PLAN }, { name: 'start_outreach', input: CAMPAIGN }] };

  it('starts a list of calls with a step open: no push, no status check, one request more when the result comes', async () => {
    const log: string[] = [];
    let step = 0;
    engine((_b, asker) => (asker === 'runner' ? [planStep, { text: 'Going out now; I will report when it ends.' }, { text: 'Jane agreed.' }][step++] ?? { text: 'Done.' } : { text: 'Hi.' }), log);
    const w = world();
    await w.sessions.load();
    const { agent: runner, events, emit } = runnerOf(w);
    await runner.run('Ring Jane and tell her the hedge trimming price.', emit);
    expect(log.filter((l) => l === 'runner')).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
    expect(w.outreach.callLiveFor('front-desk')).toBe(true);
  });

  it('starts one and then says nothing: asked once to say what it started, and then it ends with words', async () => {
    const log: string[] = [];
    let step = 0;
    engine((_b, asker) => (asker === 'runner' ? [planStep, { text: '' }, { text: 'Calling Jane now.' }][step++] ?? { text: 'Done.' } : { text: 'Hi.' }), log);
    const w = world();
    await w.sessions.load();
    const { agent: runner, events, emit } = runnerOf(w);
    await runner.run('Ring Jane and tell her the hedge trimming price.', emit);
    expect(log.filter((l) => l === 'runner')).toHaveLength(3);
    expect(nudges(events)).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Calling Jane now.' });
  });

  it("a flow of the person's runs before start_outreach and puts its note in front of the answer: the run still knows its calls started", async () => {
    const log: string[] = [];
    let step = 0;
    engine((_b, asker) => (asker === 'runner' ? [planStep, { text: 'Going out now; I will report when it ends.' }, { text: 'Jane agreed.' }][step++] ?? { text: 'Done.' } : { text: 'Hi.' }), log);
    const w = world();
    await w.sessions.load();
    const before: ToolHook = { tool: 'start_outreach', mode: 'before', flowName: 'Check the list', run: async () => '' };
    const { agent: runner, events, emit } = runnerOf(w, { toolHooks: [before] });
    await runner.run('Ring Jane and tell her the hedge trimming price.', emit);
    expect(JSON.stringify(runner.turns)).toContain('Your flow \\"Check the list\\" ran before start_outreach');
    expect(log.filter((l) => l === 'runner')).toHaveLength(2);
    expect(nudges(events)).toEqual([]);
  });
  it('is declined (the person said no): nothing started, so the plan pushes on as before', async () => {
    const log: string[] = [];
    let step = 0;
    engine((_b, asker) => (asker === 'runner' ? [planStep, { text: 'The person said no.' }, { text: 'Checking.' }, { text: 'Done.' }][step++] ?? { text: 'Done.' } : { text: 'Hi.' }), log);
    const w = world();
    await w.sessions.load();
    const tools = outreachTools({ engine: () => w.outreach, origin: () => ({ kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' }), ready: async () => '', screening: async () => null, approve: async () => false });
    const { agent: runner, events, emit } = agent('runner', { sessionTools: tools, waitingOnCall: () => true });
    await runner.run('Ring Jane.', emit);
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(w.outreach.campaigns).toHaveLength(0);
  });

  it('a call going on that this run did not start, and only a check on it (outreach_status): the plan pushes on as before', async () => {
    const log: string[] = [];
    let step = 0;
    const answers = [{ calls: [{ name: 'update_plan', input: PLAN }, { name: 'outreach_status', input: {} }] }, { text: 'The call is still going.' }, { text: 'Checking.' }, { text: 'Still going.' }];
    engine((_b, asker) => (asker === 'runner' ? answers[step++] ?? { text: 'Done.' } : { text: 'Hi.' }), log);
    const w = world();
    await w.sessions.load();
    // A call the Front desk started earlier (another run) is going on.
    const plan = w.outreach.plan(CAMPAIGN, null);
    if (typeof plan === 'string') throw new Error(plan);
    await w.outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    await w.outreach.tick();
    expect(w.outreach.callLiveFor('front-desk')).toBe(true);
    const { agent: runner, events, emit } = runnerOf(w);
    await runner.run('How is the call going?', emit);
    expect(nudges(events).length).toBeGreaterThan(0);
    expect(log.filter((l) => l === 'runner').length).toBeGreaterThan(2);
  });
});

describe('the page wires it', () => {
  // (main.ts is a page's worth of closures: it is read, not run, as caps.test.ts reads it.)
  it("the runner's and a project's conversations are told whether the call they started is going on, for their own place: the one their campaigns are started from", () => {
    expect(mainSource).toContain("waitingOnCall: () => (kind() === 'runner' || kind() === 'project') && !!outreach?.callLiveFor(place().meta.id),");
    expect(mainSource).toContain('projectId: place().meta.id, projectName: place().meta.name');
  });
});
