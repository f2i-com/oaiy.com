import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { EMPTY_MEDIA } from '../../src/agent/media';
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
    // No media service: no media instructions to pay for in a small window.
    expect(JSON.stringify(fake.bodies[0])).not.toContain('create_voice');
  });

  it('a small context window gets the core tools without the video editing ones', async () => {
    const names = (body: Record<string, unknown>) => (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
    const small = fakeProvider('openai', [{ text: 'hi' }]);
    await setup({ ...LOCAL, contextTokens: 12_000 }).agent.run('hello', () => {});
    expect(names(small.bodies[0])).toContain('read_file');
    expect(names(small.bodies[0])).not.toContain('media_compose');
    const large = fakeProvider('openai', [{ text: 'hi' }]);
    await setup({ ...LOCAL, contextTokens: 32_000 }).agent.run('hello', () => {});
    expect(names(large.bodies[0])).toEqual(expect.arrayContaining(['media_info', 'video_frames', 'video_split', 'media_compose']));
  });

  it('a small window with a video service gets the scripted way to a story in brief, and room to work', async () => {
    const fake = fakeProvider('openai', [{ text: 'hi' }]);
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video', speechModel: 'speech' };
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 12_000 }), projectSummary: () => '', media: () => media });
    const events: AgentEvent[] = [];
    await agent.run('make a short film', (e) => events.push(e));
    expect(events.find((e) => e.type === 'error')).toBeUndefined();
    const system = JSON.stringify(fake.bodies[0].messages);
    expect(system).toContain('is made from a script, written first in video/NAME/script.md');
    expect(system).not.toContain('Review each frame');
  });

  it('with a video service, a story is scripted shot by shot before anything is made', async () => {
    const fake = fakeProvider('openai', [{ text: 'hi' }]);
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video', speechModel: 'speech' };
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => media });
    await agent.run('make a short film', () => {});
    const system = JSON.stringify(fake.bodies[0].messages);
    expect(system).toContain('video/NAME/) with script.md');
    for (const line of ['Premise:', 'Characters:', 'Props:', 'Scenes:', 'Start frame:', 'End frame:', 'Video:', 'Dialogue:', 'Sound:']) expect(system).toContain(line);
    expect(system).toMatch(/read it all back and revise it with edit_file until: the story holds together/);
    // Reference images with neutral faces on blank white; the frames give the expressions, and props are made first.
    expect(system).toContain('facing the camera with a neutral expression, on a blank white background');
    expect(system).toContain('never left neutral');
    expect(system).toContain("a. Each character's reference image (neutral expression, blank white background) and saved voice, and each prop's reference image");
    // Each scene's empty background, reviewed; frames made new from it and reviewed, again when off.
    expect(system).toContain('a Background line: the empty place');
    expect(system).toMatch(/b\. Each scene's background.*Review it.*If not, make it again\./);
    expect(system).toMatch(/c\. Then shot by shot, in order: its start and end frames.*each made new with generate_image from the scene's background.*Never use a character's or prop's reference image itself as a frame\..*Review each frame.*make it again/);
    expect(system).toContain("d. Then that shot's clip, from its two frames and its Video and Dialogue lines, before the next shot.");
    // Every shot ends on its own end frame, and its Video line tells the motion between the two.
    expect(system).toContain('Every shot has one, and it differs from the start frame.');
    expect(system).toContain('Video: the motion from the start frame to the end frame, in order');
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

  it('append_file writes a long document in parts, and the parts can then be edited', async () => {
    fakeProvider('anthropic', [
      { calls: [{ name: 'append_file', input: { path: 'script/draft.md', content: '# Act one\nGary cooks.' } }] },
      { calls: [{ name: 'append_file', input: { path: 'script/draft.md', content: '# Act two\nGary rests.\n' } }] },
      // Appending to a file it wrote keeps it known, so it can edit without reading again.
      { calls: [{ name: 'edit_file', input: { path: 'script/draft.md', old_string: 'Gary rests.', new_string: 'Gary sleeps.' } }] },
      // An existing file it never read can be added to, not edited.
      { calls: [{ name: 'append_file', input: { path: 'src/app.js', content: '// more' } }] },
      (body) => {
        expect(JSON.stringify(body)).toContain('now 3 lines');
        return { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: '// more', new_string: '' } }] };
      },
      (body) => {
        expect(JSON.stringify(body)).toContain('before you edit it');
        return { text: 'ok' };
      },
    ]);
    const { vfs, agent, emit } = setup(ANTHROPIC);
    await agent.run('go', emit);
    expect(vfs.readText('/script/draft.md')).toBe('# Act one\nGary cooks.\n# Act two\nGary sleeps.\n');
    expect(vfs.readText('/src/app.js')).toMatch(/\n\/\/ more$/);
  });

  it('asks once more when the server could not read a tool call, then gives up', async () => {
    const unreadable = { error: { status: 422, body: JSON.stringify({ error: { message: 'tool_contract_error: missing function tag; no tool from this batch was executed', code: 'tool_contract_error' } }) } };
    fakeProvider('openai', [unreadable, { text: 'read it the second time' }, unreadable, unreadable]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('go', emit);
    expect(events.some((e) => e.type === 'status' && /could not read/.test(e.message))).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'read it the second time' });
    events.length = 0;
    await agent.run('again', emit);
    expect(events.at(-1)).toMatchObject({ type: 'error' });
  });

  it('an unreadable view_image call is not the model refusing images', async () => {
    const quoted = { error: { status: 422, body: JSON.stringify({ error: { message: 'tool_contract_error: missing function tag; no tool from this batch was executed. The model wrote: "<tool_call>\\n{\\"name\\": \\"view_image\\"}"', code: 'tool_contract_error' } }) } };
    const fake = fakeProvider('openai', [quoted, quoted, { text: 'fine' }]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('look', emit, undefined, [{ mediaType: 'image/png', data: 'iVBORw0KGgo=' }]);
    expect(events.some((e) => e.type === 'status' && /does not take images/.test(e.message))).toBe(false);
    expect(JSON.stringify(fake.bodies.at(-1))).toContain('image_url');
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

describe('the record of failing apps', () => {
  function appSetup() {
    const s = setup(OPENAI);
    for (const root of ['a', 'b']) {
      s.vfs.writeFile(`/${root}/manifest.json`, JSON.stringify({ name: root, version: '1.0.0', main: 'ui/main.ui', files: { ui: ['ui/main.ui'] } }), { parents: true });
      s.vfs.writeFile(`/${root}/ui/main.ui`, '<App><Text>hi</Text></App>\n', { parents: true });
    }
    return s;
  }
  const breakA = { name: 'edit_file', input: { path: 'a/manifest.json', old_string: '["ui/main.ui"]', new_string: '["ui/main.ui","ui/x.ui"]' } };

  it('a fix checked with softn_check clears the failure: no nudge about errors already gone', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'read_file', input: { path: 'a/manifest.json' } }] },
      { calls: [breakA] },
      (body) => {
        expect(JSON.stringify((body.messages as unknown[]).at(-1))).toContain('lists ui/x.ui, which does not exist');
        return { calls: [{ name: 'write_file', input: { path: 'a/ui/x.ui', content: '<App />\n' } }, { name: 'softn_check', input: { app: 'a' } }] };
      },
      { text: 'Fixed and checked.' },
    ]);
    const { agent, events, emit } = appSetup();
    await agent.run('add a page', emit);
    expect(events.filter((e) => e.type === 'check').map((e) => (e as { state: string }).state)).toEqual(['running', 'failed']);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(0);
    expect(agent.failingApps.size).toBe(0);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Fixed and checked.' });
  });

  it('a failure left in one app does not nudge a later run that works on another', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'read_file', input: { path: 'a/manifest.json' } }] },
      { calls: [breakA] },
      { text: 'left it' },
      { text: 'still left' },
      { text: 'giving up' },
      { calls: [{ name: 'read_file', input: { path: 'b/ui/main.ui' } }] },
      { calls: [{ name: 'edit_file', input: { path: 'b/ui/main.ui', old_string: 'hi', new_string: 'hello' } }] },
      { text: 'Changed b.' },
    ]);
    const { agent, events, emit } = appSetup();
    await agent.run('break a', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(2);
    expect(agent.failingApps.has('a')).toBe(true);
    events.length = 0;
    await agent.run('change b', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(0);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Changed b.' });
  });
});

describe('plan first, and carry on while there is progress', () => {
  it('a task that changes several files without a plan is asked for one, once', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'write_file', input: { path: 'a.txt', content: 'a' } }, { name: 'write_file', input: { path: 'b.txt', content: 'b' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('there is no plan yet. Call update_plan now');
        return { calls: [{ name: 'write_file', input: { path: 'c.txt', content: 'c' } }] };
      },
      { text: 'Wrote three files.' },
    ]);
    const { agent, emit } = setup(OPENAI);
    await agent.run('write three files', emit);
    expect(JSON.stringify(fake.bodies.at(-1)).match(/there is no plan yet/g)).toHaveLength(1);
  });

  it('a one-file fix is not asked for a plan', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] },
      { calls: [{ name: 'edit_file', input: { path: 'src/app.js', old_string: 'hello', new_string: 'hi' } }] },
      { text: 'Fixed.' },
    ]);
    const { agent, emit } = setup(OPENAI);
    await agent.run('fix the greeting', emit);
    expect(JSON.stringify(fake.bodies)).not.toContain('there is no plan yet');
  });

  it('keeps asking to carry on while steps get done, beyond two times', async () => {
    const step = (done: number) => ({ goal: 'three things', items: [0, 1, 2].map((i) => ({ text: `thing ${i + 1}`, status: i < done ? 'done' : 'pending' })) });
    fakeProvider('openai', [
      { calls: [{ name: 'update_plan', input: step(0) }] },
      { text: 'pausing' },
      { calls: [{ name: 'update_plan', input: step(1) }] },
      { text: 'pausing again' },
      { calls: [{ name: 'update_plan', input: step(2) }] },
      { text: 'and again' },
      { calls: [{ name: 'update_plan', input: step(3) }] },
      { text: 'All three done.' },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('do three things', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(3);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'All three done.' });
  });
});
