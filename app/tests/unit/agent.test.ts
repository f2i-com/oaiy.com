import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, announcesWork, type AgentEvent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { OpenAIStream } from '../../src/agent/providers/stream';
import { flagPicture } from '../../src/agent/review';
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
    expect(system).toContain('Every video is made from a plan (update_plan first) and a script, however short or vague the request');
    expect(system).toContain('The script is written first in video/NAME/script.md');
    expect(system).not.toContain('Review each frame');
  });

  it('with a video service, a story is scripted shot by shot before anything is made', async () => {
    const fake = fakeProvider('openai', [{ text: 'hi' }]);
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video', speechModel: 'speech' };
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => media });
    await agent.run('make a short film', () => {});
    const system = JSON.stringify(fake.bodies[0].messages);
    expect(system).toContain('video/NAME/) with script.md');
    // Every video is planned and scripted; a vague request becomes a short scene of several shots.
    expect(system).toContain('Every video the user asks for is made from a plan and a script, however short or vague the request');
    expect(system).toContain('told in 3 to 6 shots of varied framing');
    expect(system).toContain('Only when the user asks for exactly one clip is it a single shot.');
    // Each shot is a cut or continuous (the motion flowing on from the clip before), the agent's choice.
    expect(system).toContain('Joins: cut (a new framing, place or moment), or continuous from shot 2');
    // One art style for the whole video, carried by every picture's prompt and checked by the reviewer.
    expect(system).toContain('## Style: the look of the whole video');
    // Anything alive, and any part of it, is a character; props are non-living objects.
    expect(system).toContain('any part of one seen on its own (a hand reaching in, a paw) belongs to its character');
    expect(system).toContain('## Props: every non-living object');
    // Characters, props and places from the user's pictures are made from them.
    expect(system).toContain('A character from a picture the user gave says so (From: uploads/NAME.jpg)');
    expect(system).toContain("the same person, recognisably, in the video's Style.");
    expect(system).toContain("Every picture's prompt ends with that Style line, word for word");
    expect(system).toContain("Every generate_image prompt below ends with the script's Style line.");
    expect(system).toContain('Cut or continuous is your choice, shot by shot.');
    expect(system).toContain('A continuous shot instead starts on the last frame the clip it continues really ended on');
    expect(system).toContain('trim the first frame of each continuous clip (start: 0.04)');
    for (const line of ['## Premise:', '## Characters:', '## Props:', '## Scene 2:', 'Start frame:', 'End frame:', 'Video:', 'Dialogue:', 'Sound:']) expect(system).toContain(line);
    expect(system).toMatch(/read it all back and revise it with edit_file until: the story holds together/);
    // Reference images with neutral faces on blank white; the frames give the expressions, and props are made first.
    expect(system).toContain('facing the camera with a neutral expression, on a blank white background');
    expect(system).toContain('never left neutral');
    expect(system).toContain("a. Each character's reference image (frame: character; neutral expression, blank white background) and saved voice, and each prop's reference image");
    // Each scene's empty background; every picture reviewed as it is made, and made again when sent back.
    expect(system).toContain('a Background line: the empty place');
    expect(system).toContain("b. Each scene's background (frame: background, scene: its number)");
    expect(system).toMatch(/Every picture saved in the video's folder is reviewed as soon as it is made, before anything else.*A picture sent back is made again at the same path/);
    // The start frame made new from the background and references; the end frame an edit of the start frame.
    expect(system).toMatch(/c\. Then shot by shot, in order\. Its start frame \(frame: start, shot: its number\).*made new with generate_image from the scene's background.*Never use a character's or prop's reference image itself as a frame\./);
    expect(system).toContain('Then its end frame (frame: end, shot: its number), from its End frame line, made by editing the start frame: give the start frame as the first reference image');
    expect(system).toContain("d. Then that shot's clip, from its two passed frames and its Video and Dialogue lines, before the next shot.");
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

  it('reads a message the user sent while it worked at its next step, and is not done until it has read it', async () => {
    const { agent, events, emit } = setup(OPENAI);
    expect(agent.interject('not now')).toBe(false);
    const said = (body: Record<string, unknown>) => JSON.stringify(body.messages);
    fakeProvider('openai', [
      () => {
        expect(agent.interject('Also make the greeting shout.')).toBe(true);
        return { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] };
      },
      (body) => {
        expect(said(body)).toContain('[The user sent this while you were working. Read it now and act on it as soon as you can');
        expect(said(body)).toContain('Also make the greeting shout.');
        // One more, as it gives its final answer.
        agent.interject('And say hi.');
        return { text: 'All done.' };
      },
      (body) => {
        expect(said(body)).toContain('And say hi.');
        return { text: 'Hi! Done, with a shout.' };
      },
    ]);
    await agent.run('change the greeting', emit);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Hi! Done, with a shout.' });
    expect(agent.takeUnread()).toEqual([]);
  });

  it('thinks harder while it works on a video script, and only asks nrob for that', async () => {
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video' };
    const script = { goal: 'a short film', items: [{ text: 'Write the script in video/film/script.md', status: 'active' }, { text: 'Make the clips', status: 'pending' }] };
    for (const [serverKind, want] of [['nrob', 'high'], ['ollama', undefined]] as const) {
      const fake = fakeProvider('openai', [
        { calls: [{ name: 'update_plan', input: script }] },
        { calls: [{ name: 'append_file', input: { path: 'video/film/script.md', content: '## Premise\nA cat.' } }] },
        { text: 'done' },
      ]);
      const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...LOCAL, serverKind, contextTokens: 32_000 }), projectSummary: () => '', media: () => media });
      await agent.run('make a short film', () => {});
      // Not before there is a plan; while its step is the script, and after writing the script.
      expect(fake.bodies.slice(0, 3).map((b) => b.reasoning_effort)).toEqual([undefined, want, want]);
    }
  });

  it("shows nrob's tool call as it is written", () => {
    const drafts: Array<[string, boolean]> = [];
    const stream = new OpenAIStream({ text: () => {}, toolStart: () => {}, toolArgs: () => {}, draft: (t, s) => drafts.push([t, s]) });
    for (const chunk of [{ nrob_tool_preview: { text: '<tool_call>\n<function=append_file>', start: true } }, { nrob_tool_preview: { text: '\n<parameter=content>\n## Scene 1', start: false } }]) {
      stream.accept({ event: 'message', data: JSON.stringify(chunk) });
    }
    expect(drafts).toEqual([['<tool_call>\n<function=append_file>', true], ['\n<parameter=content>\n## Scene 1', false]]);
  });

  it('is asked, once, to start when it only says what it will do', async () => {
    const said = (body: Record<string, unknown>) => JSON.stringify(body.messages);
    fakeProvider('openai', [
      { text: "Right away! I'll plan the script, make the character, then animate the clip." },
      (body) => {
        expect(said(body)).toContain('You said what you will do, but no tool has run yet, so nothing has started.');
        return { calls: [{ name: 'read_file', input: { path: 'src/app.js' } }] };
      },
      // Having started, a plain closing answer ends the run: the nudge is only for not starting.
      { text: "Let me know if you'd like changes." },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('make a video', emit);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'done' });
  });

  it('stops, rather than loops, when the model keeps giving the same reply and does nothing', async () => {
    const same = { text: "Oh, you bet! Let's make it." };
    const fake = fakeProvider('openai', [
      same,
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('no tool has run yet, so nothing has started');
        return same;
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('You gave the same reply again, and still no tool has run.');
        return same;
      },
    ]);
    const { agent, events, emit } = setup(OPENAI);
    await agent.run('make a video', emit);
    expect(fake.bodies).toHaveLength(3);
    expect(events.filter((e) => e.type === 'nudge')).toHaveLength(2);
    expect(events.some((e) => e.type === 'status' && /kept giving the same reply/.test(e.message))).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: 'done' });
  });

  it('saves the conversation without the pictures tools showed long ago, but with the user\'s own', () => {
    const { agent } = setup(OPENAI);
    const pixel = { mediaType: 'image/png' as const, data: 'iVBORw0KGgo=' };
    agent.turns = [
      { role: 'user', text: 'look at this', images: [pixel] },
      ...Array.from({ length: 5 }, (_, i) => ({ role: 'tool' as const, results: [{ id: String(i), name: 'view_image', content: `look ${i}`, isError: false, images: [pixel] }] })),
    ];
    const saved = agent.savedTurns();
    const kept = saved.filter((t) => t.role === 'tool' && t.results[0].images?.length);
    expect(kept.map((t) => (t.role === 'tool' ? t.results[0].content : ''))).toEqual(['look 2', 'look 3', 'look 4']);
    expect(saved[1]).toMatchObject({ role: 'tool', results: [{ content: 'look 0\n[image no longer attached; call view_image again to see it]' }] });
    expect(saved[0]).toMatchObject({ role: 'user', images: [pixel] });
    // The conversation itself is untouched.
    expect(agent.turns.filter((t) => t.role === 'tool' && t.results[0].images?.length)).toHaveLength(5);
  });

  it('fixes flagged pictures first, one at a time, then goes back to its work', async () => {
    const said = (body: Record<string, unknown>) => JSON.stringify(body.messages);
    // What each request lacked (a failed expect inside the fake would only show as a network error).
    const missing: string[] = [];
    const want = (body: Record<string, unknown>, step: number, text: string, present = true) => {
      if (said(body).includes(text) !== present) missing.push(`request ${step} ${present ? 'lacks' : 'has'}: ${text}`);
    };
    const plan = { goal: 'a film', items: [{ text: 'Make shot 2', status: 'active' }] };
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'update_plan', input: plan }] },
      // The first flag alone, with the other waiting.
      (body) => {
        want(body, 2, 'The user flagged /pics/a.png as wrong: \\"three arms\\". Fix it before anything else, and only it (1 more flagged picture will follow');
        want(body, 2, '/pics/b.png as wrong', false);
        return { calls: [{ name: 'append_file', input: { path: 'pics/a.png', content: 'made again' } }] };
      },
      // Fixed: the next one.
      (body) => {
        want(body, 3, 'The user flagged /pics/b.png as wrong, without saying why');
        return { calls: [{ name: 'append_file', input: { path: 'pics/b.png', content: 'made again' } }] };
      },
      // All fixed: back to the plan.
      (body) => {
        want(body, 4, 'The flagged pictures are fixed. Now go back to the work you were doing');
        want(body, 4, '\\"Make shot 2\\"');
        return { calls: [{ name: 'update_plan', input: { ...plan, items: [{ text: 'Make shot 2', status: 'done' }] } }] };
      },
      { text: 'Done.' },
    ]);
    const { vfs, agent, events, emit } = setup(OPENAI);
    for (const p of ['pics/a.png', 'pics/b.png']) {
      vfs.writeFile(`/${p}`, 'first try', { parents: true });
      flagPicture(vfs, p, p.endsWith('a.png') ? 'three arms' : '');
    }
    // Flagged while idle: queued, and taken when the run starts.
    expect(agent.flag('pics/a.png', 'three arms')).toBe(false);
    agent.flag('pics/b.png', '');
    await agent.run('make the film', emit);
    expect(missing).toEqual([]);
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(fake.bodies).toHaveLength(5);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Done.' });
    expect(events.filter((e) => e.type === 'status' && /Fixed the flagged/.test(e.message))).toHaveLength(2);
  });

  it('announces work only with a promise, not a question', () => {
    expect(announcesWork("Okay! Let's make it. I'll plan the script first.")).toBe(true);
    expect(announcesWork('Right away!')).toBe(true);
    expect(announcesWork('Shall I make it 10 seconds or 20? Let me know?')).toBe(false);
    expect(announcesWork('The greeting now says hi.')).toBe(false);
    expect(announcesWork("It says hi now. Let me know if you'd like changes.")).toBe(false);
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
