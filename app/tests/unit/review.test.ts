import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { EMPTY_MEDIA } from '../../src/agent/media';
import { runTool, type ToolContext } from '../../src/agent/tools';
import { REVIEWS_FILE, checklist, contentHash, countProblems, flagPicture, readReviews, scriptExcerpt, storyFolder, writeReviews } from '../../src/agent/review';
import { LOCAL, fakeProvider } from './fakeProvider';

// No canvas here: a look at a picture gives a stand-in image of the region asked for.
vi.mock('../../src/agent/images', async (original) => {
  const real = await original<typeof import('../../src/agent/images')>();
  return {
    ...real,
    imageSize: async () => ({ width: 100, height: 100 }),
    viewImage: async (_bytes: Uint8Array, _mime: string, o: { x?: number; y?: number; width?: number; height?: number } = {}) => {
      const region = { x: o.x ?? 0, y: o.y ?? 0, width: o.width ?? 100, height: o.height ?? 100 };
      return { image: { mediaType: 'image/png', data: 'iVBORw0KGgo=' }, width: 100, height: 100, region, shownWidth: region.width, shownHeight: region.height };
    },
  };
});

afterEach(() => vi.unstubAllGlobals());

const SCRIPT = [
  '# The Soup',
  '',
  '## Premise',
  'Gary, a tired chef, must win back his restaurant with one last soup.',
  '',
  '## Style',
  'Soft 2D anime look. Style: 2D anime, clean line art, cel shading, soft pastel palette',
  '',
  '## Characters',
  'Gary: fifties, grey stubble, red chef jacket. Voice: gruff.',
  '',
  '## Scene 1: The kitchen at dawn',
  'Gary alone in his kitchen before opening.',
  'Background: a small steel kitchen, dawn light through one window, no people.',
  '',
  '### Shot 1 (scene 1, 4 s)',
  'Start frame: Gary at the stove, close-up, tired eyes, steam rising.',
  'End frame: Gary tastes the soup, eyes widening.',
  'Video: he lifts the spoon, blows, tastes, his eyes widen.',
  'Dialogue: none',
  'Sound: a simmering pot.',
  '',
  '### Shot 2 (scene 1, 3 s)',
  'Start frame: the pot, close-up.',
  '',
  '## Scene 2: The dining room',
  'Background: an empty dining room at noon.',
].join('\n');

const MEDIA = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video', discovered: { origin: 'http://127.0.0.1:8080' } } as never;

/** The chat model's script for every request but the image service's, which makes a new picture each time. */
let videoAsked = 0;

function fakeServices(script: Parameters<typeof fakeProvider>[1]) {
  const chat = fakeProvider('openai', script);
  const llm = globalThis.fetch;
  let made = 0;
  videoAsked = 0;
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (String(url).includes('/videos')) {
      videoAsked++;
      return new Response(JSON.stringify({ error: { message: 'the video worker is down' } }), { status: 503, headers: { 'Content-Type': 'application/json' } });
    }
    if (!String(url).includes('/images')) return llm(url, init);
    made++;
    const bytes = new Uint8Array([137, 80, 78, 71, made, made * 7, 3]);
    return new Response(JSON.stringify({ data: [{ b64_json: btoa(String.fromCharCode(...bytes)) }] }), { headers: { 'Content-Type': 'application/json' } });
  });
  return chat;
}

function film(): Vfs {
  const vfs = new Vfs();
  vfs.writeFile('/video/film/script.md', SCRIPT, { parents: true });
  vfs.writeFile('/video/film/scene1-background.png', new Uint8Array([137, 80, 78, 71, 1]));
  vfs.writeFile('/video/film/gary.png', new Uint8Array([137, 80, 78, 71, 2]));
  return vfs;
}

const lastToolText = (body: Record<string, unknown>) => {
  const messages = body.messages as Array<{ role: string; content: unknown }>;
  return String([...messages].reverse().find((m) => m.role === 'tool')?.content ?? '');
};

describe('the script a picture is checked against', () => {
  it('is its premise, its scene without the other shots, and its shot', () => {
    const excerpt = scriptExcerpt(SCRIPT, 'start', '1');
    expect(excerpt).toContain('Gary, a tired chef');
    // Every picture is checked against the video's art style.
    expect(excerpt).toContain('Style: 2D anime, clean line art');
    expect(scriptExcerpt(SCRIPT, 'background', undefined, '2')).toContain('Style: 2D anime');
    for (const kind of ['start', 'end', 'background', 'character', 'prop'] as const) expect(checklist(kind)).toContain("It is in the script's Style");
    // A character from the user's picture of a person must still be them.
    expect(checklist('character')).toContain('it is recognisably that person, restyled but not replaced');
    expect(checklist('prop')).toContain('a picture of a body part (a hand, a face), a person or an animal is not a prop');
    expect(excerpt).toContain('Background: a small steel kitchen');
    expect(excerpt).toContain('Start frame: Gary at the stove');
    expect(excerpt).not.toContain('Start frame: the pot');
    expect(excerpt).not.toContain('dining room');
    expect(scriptExcerpt(SCRIPT, 'background', undefined, '2')).toContain('an empty dining room at noon');
    expect(scriptExcerpt(SCRIPT, 'character')).toContain('red chef jacket');
    expect(scriptExcerpt(SCRIPT, 'start', '9')).toContain('No "### Shot 9" heading');
  });

  it("belongs to the nearest folder with a script", () => {
    const vfs = film();
    expect(storyFolder(vfs, 'video/film/frames/shot1.png')).toBe('video/film');
    expect(storyFolder(vfs, 'pictures/cat.png')).toBeNull();
  });
});

describe('a picture made for a scripted video', () => {
  it('is reviewed by a reviewer that reads and looks as it likes, sent back, made again and passed; only then animated', async () => {
    const vfs = film();
    const frame = { path: 'video/film/shot1-start.png', frame: 'start', shot: '1', prompt: 'Gary tastes', reference_images: ['video/film/scene1-background.png', 'video/film/gary.png'] };
    fakeServices([
      { calls: [{ name: 'generate_image', input: frame }] },
      // The reviewer: told what the picture is, its references and its part of the script; free to look further.
      (body) => {
        const text = JSON.stringify(body.messages);
        expect(text).toContain('You are a reviewer');
        expect(text).toContain("a picture not in the script's art style");
        expect(text).toContain("It is in the script's Style");
        expect(text).toContain('/video/film/shot1-start.png');
        expect(text).toContain('the start frame of shot 1');
        expect(text).toContain('- /video/film/gary.png');
        expect(text).toContain('Start frame: Gary at the stove');
        const tools = (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
        expect(tools).toEqual(expect.arrayContaining(['read_file', 'view_image', 'list_files', 'give_verdict']));
        expect(tools).not.toContain('write_file');
        expect(tools).not.toContain('generate_image');
        return { calls: [{ name: 'read_file', input: { path: 'video/film/script.md' } }] };
      },
      { calls: [{ name: 'list_files', input: { path: 'video/film' } }] },
      { calls: [{ name: 'give_verdict', input: { verdict: 'redo', people: [{ who: 'Gary', heads: 1, arms: 3, hands: 3, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'Gary has a third arm from his left shoulder.' } }] },
      // The main agent hears the verdict, and cannot animate the frame.
      (body) => {
        expect(lastToolText(body)).toContain('sent back (try 1 of 3). Gary has a third arm from his left shoulder. (Counted: Gary: 1 head, 3 arms, 3 hands, 2 legs.)');
        return { calls: [{ name: 'generate_video', input: { prompt: 'he tastes', path: 'video/film/shot1.mp4', start_image: frame.path, end_image: frame.path } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain('was sent back by its review (Gary has a third arm');
        return { calls: [{ name: 'generate_image', input: { ...frame, prompt: 'Gary tastes, two arms' } }] };
      },
      // The remake's review is told what the last try was sent back for, and gets the picture and its references
      // with its task, small, for a once-over; counting three arms and still passing is refused.
      (body) => {
        const messages = body.messages as Array<{ role: string; content: unknown }>;
        expect(JSON.stringify(messages)).toContain('The try before this one was sent back for: Gary has a third arm');
        expect(JSON.stringify(messages)).toContain('Attached, in order:\\n1. the picture, /video/film/shot1-start.png\\n2. reference /video/film/scene1-background.png\\n3. reference /video/film/gary.png');
        const task = messages.find((m) => m.role === 'user' && Array.isArray(m.content)) as { content: Array<{ type: string }> };
        expect(task.content.filter((c) => c.type === 'image_url')).toHaveLength(3);
        return { calls: [{ name: 'give_verdict', input: { verdict: 'pass', people: [{ who: 'Gary', heads: 1, arms: 3, hands: 2, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'Fine.' } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain('it cannot pass with what you found: Gary has 3 arms');
        return { calls: [{ name: 'give_verdict', input: { verdict: 'pass', people: [{ who: 'Gary', heads: 1, arms: 2, hands: 2, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'Face, jacket and kitchen match; two arms.' } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain('passed. Face, jacket and kitchen match; two arms.');
        return { calls: [{ name: 'generate_video', input: { prompt: 'he tastes', path: 'video/film/shot1.mp4', start_image: frame.path, end_image: frame.path } }] };
      },
      // Passed: the frame goes to the video service (here, one that is down).
      (body) => {
        expect(lastToolText(body)).not.toContain('review');
        expect(videoAsked).toBe(1);
        return { text: 'done' };
      },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('make shot 1', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'done' });
    const review = readReviews(vfs)[frame.path];
    expect(review).toMatchObject({ verdict: 'pass', redos: 1, kind: 'start', shot: '1' });
    // The chat shows each review as a sub-agent under the picture's card.
    expect(events.some((e) => e.type === 'agent_task' && e.title === `review /${frame.path}` && e.state === 'done')).toBe(true);
  });

  it('is made to decide after a few looks, is tried again without a verdict, and then waits for the agent to look or remake', async () => {
    const vfs = film();
    const path = 'video/film/scene1-bg.png';
    const toolNames = (body: Record<string, unknown>) => (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
    // A reviewer that only ever looks: three looks, then give_verdict is its only tool, until its steps run out.
    const looking = (n: number) =>
      Array.from({ length: 8 }, (_, i) => (body: Record<string, unknown>) => {
        if (i < 3) {
          expect(toolNames(body)).toContain('view_image');
          return { calls: [{ name: 'view_image', input: { path } }] };
        }
        expect(toolNames(body)).toEqual(['give_verdict']);
        expect(JSON.stringify(body.messages)).toContain('You have looked enough. Give your verdict now with give_verdict');
        return { text: `try ${n}: still unsure` };
      });
    fakeServices([
      { calls: [{ name: 'generate_image', input: { path, frame: 'background', scene: '1', prompt: 'the kitchen' } }] },
      ...looking(1),
      ...looking(2),
      (body) => {
        expect(lastToolText(body)).toContain(`Review of /${path}: no verdict after 2 tries`);
        expect(lastToolText(body)).toContain('Look at it yourself with view_image: if it is right, take it with review_frame and accept');
        return { calls: [{ name: 'generate_image', input: { path: 'video/film/other.png', prompt: 'anything' } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain(`/${path} is still waiting for review`);
        return { calls: [{ name: 'review_frame', input: { path, accept: true, notes: 'looked right to me' } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain(`Took /${path} as it is.`);
        return { text: 'ok' };
      },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('make the background', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
    expect(readReviews(vfs)[path]).toMatchObject({ verdict: 'pass', failures: 2 });
    // The review's report in the chat says what each try did.
    const failed = events.find((e) => e.type === 'agent_task' && e.state === 'failed') as { result?: string } | undefined;
    expect(failed?.result).toContain('Try 1: The run reached its 8-step limit');
    expect(failed?.result).toContain(`view_image ${path}`);
  });

  it('ends as soon as the reviewer gives its verdict: no closing reply is asked for', async () => {
    const vfs = film();
    const chat = fakeServices([
      { calls: [{ name: 'generate_image', input: { path: 'video/film/gary-2.png', frame: 'character', prompt: 'Gary' } }] },
      { calls: [{ name: 'give_verdict', input: { verdict: 'pass', people: [{ who: 'Gary', heads: 1, arms: 2, hands: 2, legs: 2, matches: 'no reference' }], out_of_place: 'nothing', notes: 'Gary as described.' } }] },
      (body) => {
        expect(lastToolText(body)).toContain('passed. Gary as described.');
        return { text: 'ok' };
      },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('make Gary', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
    expect(chat.bodies).toHaveLength(3);
  });
});

describe("a reviewer's counts", () => {
  it('cannot pass a person with an extra head, arm, hand or leg, or one unlike their reference', () => {
    expect(countProblems([{ who: 'Gary', heads: 1, arms: 2, hands: 2, legs: 2, matches: 'yes' }])).toEqual([]);
    // Limbs out of frame are not counted: fewer is fine.
    expect(countProblems([{ who: 'Gary', heads: 1, arms: 1, hands: 1, legs: 0, matches: 'no reference' }])).toEqual([]);
    expect(countProblems([{ who: 'Gary', heads: 1, arms: 3, hands: 2, legs: 2, matches: 'yes' }])).toEqual(['Gary has 3 arms']);
    expect(countProblems([{ who: 'Ann', heads: 2, arms: 2, hands: 3, legs: 3, matches: 'no' }])).toEqual([
      'Ann has 2 heads',
      'Ann has 3 hands',
      'Ann has more hands (3) than arms (2)',
      'Ann has 3 legs',
      'Ann does not match their reference',
    ]);
  });
});

describe('a picture the user flagged', () => {
  const looks = (path: string) => ({ calls: [{ name: 'view_image', input: { path } }, { name: 'view_image', input: { path, x: 10, y: 10, width: 50, height: 50 } }] });

  it('without a comment: cannot be animated or taken as it is, and its review must find what is wrong', async () => {
    const vfs = film();
    const path = 'video/film/shot1-start.png';
    vfs.writeFile(`/${path}`, new Uint8Array([137, 80, 78, 71, 9]));
    writeReviews(vfs, { [path]: { hash: contentHash(vfs.readBytes(`/${path}`)), verdict: 'pass', redos: 0, kind: 'start', shot: '1', references: [] } });
    flagPicture(vfs, path, '');
    fakeServices([
      { calls: [{ name: 'generate_video', input: { prompt: 'he tastes', path: 'video/film/shot1.mp4', start_image: path, end_image: path } }] },
      (body) => {
        expect(lastToolText(body)).toContain('was sent back by its review (flagged by the user, without saying why)');
        return { calls: [{ name: 'review_frame', input: { path, accept: true } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain(`the user flagged /${path}`);
        return { calls: [{ name: 'review_frame', input: { path } }] };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The user flagged this picture as wrong, without saying why. Something in it is wrong: find what');
        return looks(path);
      },
      { calls: [{ name: 'give_verdict', input: { verdict: 'pass', people: [{ who: 'Gary', heads: 1, arms: 2, hands: 2, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'I see nothing wrong.' } }] },
      (body) => {
        expect(lastToolText(body)).toContain('the user flagged this picture as wrong, so it cannot pass');
        return { calls: [{ name: 'give_verdict', input: { verdict: 'redo', people: [{ who: 'Gary', heads: 1, arms: 2, hands: 2, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'His left hand has six fingers.' } }] };
      },
      (body) => {
        expect(lastToolText(body)).toContain('sent back (try 1 of 3). His left hand has six fingers.');
        return { text: 'ok' };
      },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('I flagged it', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
    expect(videoAsked).toBe(0);
    expect(readReviews(vfs)[path]).toMatchObject({ verdict: 'redo', flagged: true, notes: expect.stringContaining('six fingers') });
  });

  it("with a comment: its review is told what the user said", async () => {
    const vfs = film();
    const path = 'video/film/gary.png';
    flagPicture(vfs, path, 'he has three arms');
    fakeServices([
      { calls: [{ name: 'review_frame', input: { path } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The user flagged this picture as wrong: \\"he has three arms\\"');
        return { calls: [{ name: 'give_verdict', input: { verdict: 'redo', people: [{ who: 'Gary', heads: 1, arms: 3, hands: 3, legs: 2, matches: 'yes' }], out_of_place: 'nothing', notes: 'A third arm behind his back.' } }] };
      },
      { text: 'ok' },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('I flagged it', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
  });
});

describe('a picture sent back again and again', () => {
  it('can be taken as it is after three tries, saying what is still off', async () => {
    const vfs = film();
    const path = 'video/film/scene1-background.png';
    writeReviews(vfs, { [path]: { hash: contentHash(vfs.readBytes(`/${path}`)), verdict: 'redo', redos: 3, kind: 'background', scene: '1', notes: 'the window is on the wrong wall' } });
    fakeServices([
      { calls: [{ name: 'review_frame', input: { path, accept: true, notes: 'the window is on the left' } }] },
      (body) => {
        expect(lastToolText(body)).toContain(`Took /${path} as it is. Tell the user what is still off in it: the window is on the left.`);
        return { text: 'ok' };
      },
    ]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...LOCAL, contextTokens: 32_000 }), projectSummary: () => '', media: () => MEDIA, guides: ['video'] });
    const events: AgentEvent[] = [];
    await agent.run('go on', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
    expect(readReviews(vfs)[path]).toMatchObject({ verdict: 'pass', notes: 'taken as it is (3 tries, 0 reviews without a verdict); still off: the window is on the left' });
  });
});

describe('generate_video with a start frame', () => {
  it("takes the frame's shape when the size asked for has another", async () => {
    const vfs = film();
    // The stand-in viewer says every picture is 100×100; a square frame.
    vfs.writeFile('/video/film/square.png', new Uint8Array([137, 80, 78, 71, 3]));
    writeReviews(vfs, {});
    const sent: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url).endsWith('/content')) return new Response(new Uint8Array([0, 0, 0, 24]));
      if (init?.method === 'POST') sent.push(JSON.parse(String(init.body)));
      return new Response(JSON.stringify({ id: 'v1', status: 'completed', model: 'ltx', size: '640x640' }), { headers: { 'Content-Type': 'application/json' } });
    }));
    const ctx: ToolContext = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => MEDIA };
    const run = (size: string) => runTool({ id: size, name: 'generate_video', input: { prompt: 'she smiles', path: `video/film/${size}.mp4`, size, start_image: 'video/film/square.png' } }, ctx);
    const landscape = await run('768x512');
    expect(sent[0].size).toBeUndefined();
    expect(landscape.content).toContain("The size 768x512 did not have the start frame's shape (100×100)");
    const square = await run('640x640');
    expect(sent[1].size).toBe('640x640');
    expect(square.content).not.toContain('did not have');
  });
});

describe('generate_image for a scripted video', () => {
  const ctx = (vfs: Vfs, images?: boolean): ToolContext => ({ vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => MEDIA, images });

  it("makes an end frame from its start frame, not from a character's reference image", async () => {
    const vfs = film();
    writeReviews(vfs, { 'video/film/gary.png': { hash: contentHash(vfs.readBytes('/video/film/gary.png')), verdict: 'pass', redos: 0, kind: 'character' } });
    fakeServices([]);
    const result = await runTool({ id: '1', name: 'generate_image', input: { path: 'video/film/shot1-end.png', frame: 'end', shot: '1', prompt: 'he tastes', reference_images: ['video/film/gary.png'] } }, ctx(vfs));
    expect(result.content).toContain("an end frame is its shot's start frame edited: give the start frame as the first reference image");
  });

  it('is not reviewed outside a scripted video, or for a model that cannot see pictures', async () => {
    const vfs = film();
    fakeServices([]);
    const elsewhere = await runTool({ id: '1', name: 'generate_image', input: { path: 'pictures/cat.png', prompt: 'a cat' } }, ctx(vfs));
    expect(elsewhere.isError).toBe(false);
    expect(elsewhere.review).toBeUndefined();
    const blind = await runTool({ id: '2', name: 'generate_image', input: { path: 'video/film/shot1-start.png', frame: 'start', shot: '1', prompt: 'Gary' } }, ctx(vfs, false));
    expect(blind.isError).toBe(false);
    expect(blind.review).toBeUndefined();
    expect(vfs.exists(REVIEWS_FILE)).toBe(false);
  });
});
