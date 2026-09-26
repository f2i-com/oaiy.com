import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, LOCAL, OPENAI, fakeProvider } from './fakeProvider';
import type { ProviderConfig } from '../../src/agent/providers/types';

afterEach(() => vi.unstubAllGlobals());

function setup(provider: ProviderConfig) {
  const vfs = new Vfs();
  vfs.writeFile('/src/app.js', 'const greeting = "hello";\nconsole.log(greeting);\n', { parents: true });
  const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'Project: demo' });
  const events: AgentEvent[] = [];
  return { vfs, agent, events, emit: (e: AgentEvent) => events.push(e) };
}

describe.each([
  ['anthropic', ANTHROPIC],
  ['openai', OPENAI],
] as const)('the agent loop over %s', (wire, provider) => {
  it('reads, edits, and answers', async () => {
    const fake = fakeProvider(wire, [
      { text: 'Let me look.', calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] },
      { calls: [{ name: 'edit_file', input: { path: '/src/app.js', old_string: '"hello"', new_string: '"hi there"' } }] },
      { text: 'Changed the greeting.' },
    ]);
    const { vfs, agent, events, emit } = setup(provider);
    await agent.run('change the greeting', emit);
    expect(vfs.readText('/src/app.js')).toContain('"hi there"');
    const done = events.find((e) => e.type === 'done');
    expect(done).toMatchObject({ type: 'done', text: 'Changed the greeting.', steps: 3 });
    expect(events.filter((e) => e.type === 'text').map((e) => (e as { delta: string }).delta).join('')).toContain('Let me look.');
    // The first request carries the project summary; tool results go back.
    expect(JSON.stringify(fake.bodies[0])).toContain('Project: demo');
    expect(JSON.stringify(fake.bodies[1])).toContain('1\\tconst greeting');
    expect(fake.bodies[0].stream).toBe(true);
    if (wire === 'anthropic') {
      expect(fake.headers[0]['anthropic-dangerous-direct-browser-access']).toBe('true');
      expect(fake.urls[0]).toContain('/v1/messages');
    } else {
      expect(fake.urls[0]).toContain('/chat/completions');
      expect(fake.bodies[0].max_completion_tokens).toBeDefined();
    }
  });

  it('refuses an edit to a file it has not read, and the model recovers', async () => {
    fakeProvider(wire, [
      { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: 'hello', new_string: 'bye' } }] },
      (body) => {
        expect(JSON.stringify(body)).toContain('read /src/app.js before you edit it');
        return { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] };
      },
      { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: 'hello', new_string: 'bye' } }] },
      { text: 'done' },
    ]);
    const { vfs, agent, emit } = setup(provider);
    await agent.run('say bye', emit);
    expect(vfs.readText('/src/app.js')).toContain('"bye"');
  });
});

describe('tool rules', () => {
  it('an ambiguous edit is refused with a count, a missing one shows the nearest lines', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] },
      { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: 'greeting', new_string: 'x' } }] },
      (body) => {
        expect(JSON.stringify(body)).toContain('matches 2 places');
        return { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: 'const greting = ', new_string: 'x' } }] };
      },
      (body) => {
        expect(JSON.stringify(body)).toContain('closest lines');
        return { text: 'ok' };
      },
    ]);
    const { agent, emit } = setup(OPENAI);
    await agent.run('go', emit);
  });

  it('a local OpenAI-compatible server gets max_tokens, not max_completion_tokens', async () => {
    const fake = fakeProvider('openai', [{ text: 'hi' }]);
    const { agent, emit } = setup(LOCAL);
    await agent.run('hello', emit);
    expect(fake.urls[0]).toBe('http://localhost:11434/v1/chat/completions');
    expect(fake.bodies[0].max_tokens).toBeDefined();
    expect(fake.bodies[0].max_completion_tokens).toBeUndefined();
  });

  it('write_file needs a full read before it replaces an existing file', async () => {
    fakeProvider('anthropic', [
      { calls: [{ name: 'write_file', input: { path: 'src/app.js', content: 'gone' } }] },
      (body) => {
        expect(JSON.stringify(body)).toContain('before you replace it');
        return { calls: [{ name: 'write_file', input: { path: 'notes/new.md', content: '# new' } }] };
      },
      { text: 'ok' },
    ]);
    const { vfs, agent, emit } = setup(ANTHROPIC);
    await agent.run('go', emit);
    expect(vfs.readText('/src/app.js')).toContain('hello');
    expect(vfs.readText('/notes/new.md')).toBe('# new');
  });

  it('with no provider the run says what to do', async () => {
    const vfs = new Vfs();
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => null, projectSummary: () => '' });
    const events: AgentEvent[] = [];
    await agent.run('hi', (e) => events.push(e));
    expect(events[0]).toMatchObject({ type: 'error' });
  });
});

describe('the plan and the goal', () => {
  const plan = (statuses: string[]) => ({ goal: 'A greeting that says hi', items: statuses.map((status, i) => ({ text: `step ${i + 1}`, status })) });

  it('shows the plan, and a run that stops with open steps is asked to carry on', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'update_plan', input: plan(['active', 'pending']) }] },
      { text: 'I think that is it.' },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('Your plan still has 2 open steps');
        return { calls: [{ name: 'update_plan', input: plan(['done', 'done']) }] };
      },
      { text: 'All done.' },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('greet', emit);
    expect(fake.bodies).toHaveLength(4);
    const plans = events.filter((e) => e.type === 'plan');
    expect(plans).toHaveLength(2);
    expect(agent.plan?.items.every((i) => i.status === 'done')).toBe(true);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'All done.' });
    expect(agent.turns.some((t) => t.role === 'user' && t.automatic)).toBe(true);
  });

  it('asks at most twice, then lets the run end', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'update_plan', input: plan(['active', 'pending']) }] },
      { text: 'stopping' },
      { text: 'still stopping' },
      { text: 'really stopping' },
      { text: 'out of script' },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('greet', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(2);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'really stopping' });
  });

  it('a plan from an earlier request does not hold up a new one', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'update_plan', input: plan(['active']) }] },
      { text: 'a' },
      { text: 'b' },
      { text: 'c' },
      { text: 'answer to the question' },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('greet', emit);
    events.length = 0;
    await agent.run('what is 2 + 2?', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(0);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'answer to the question' });
  });

  it('refuses an empty plan', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: { items: [] } }] }, { text: 'ok' }]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('go', emit);
    const result = events.find((e) => e.type === 'tool_result') as Extract<AgentEvent, { type: 'tool_result' }>;
    expect(result.result.isError).toBe(true);
    expect(agent.plan).toBeNull();
  });
});
