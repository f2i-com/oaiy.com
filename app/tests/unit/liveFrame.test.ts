// The preview's side of SoftN's frame protocol past the first `init` (src/preview/liveFrame.ts), against a
// stand-in frame that speaks it over a real MessageChannel: `shown`, an update taken in place and each of its
// answers, the deadline, and what falls back to a new frame. The runtime installed today (v0.0.32) announces
// neither `shown` nor `update`; the e2e (tests/e2e/softn.mjs) holds the old path on it.
import { afterEach, describe, expect, it } from 'vitest';
import { KEPT_STATUS, LiveFrame, PHONE_LAYOUT_MAX, UPDATE_DEADLINE_MS, engineMismatch, fitNote, frameSays, isLoadSyntaxError, plainProblem, readUpdateAnswer, updateSaid, type UpdateAnswer } from '../../src/preview/liveFrame';
import AGENT from '../../src/agent/agent.ts?raw';
import PREVIEW from '../../src/preview/preview.ts?raw';

const BROKEN = 'function broken( {';
const PROBLEM = 'main.js: expected a property name, at the end of the logic: a brace, bracket or string opened before it is never closed';

/**
 * A frame that takes updates as SoftN's does: it holds the last version that started, and answers each
 * `formlogic:update` with what became of it. `silent` never answers (a frame that hangs).
 */
class StandInFrame {
  readonly ready = { type: 'formlogic:ready', nativeProtocol: 1, storageProtocol: 1, shown: 1, update: 1 };
  /** The version on screen: its text files, and its media. */
  running: Record<string, string>;
  runningMedia: Record<string, string>;
  /** The media as the parent last sent it (in `init` or an update): what an update that leaves `assets` out means. */
  sentMedia: Record<string, string>;
  received: Array<{ type?: string; id?: number; client?: Record<string, string>; assets?: Record<string, string> | null }> = [];
  silent = false;
  private readonly channel = new MessageChannel();
  readonly parentPort = this.channel.port1;

  constructor(client: Record<string, string>, assets: Record<string, string> = {}) {
    this.running = { ...client };
    this.runningMedia = { ...assets };
    this.sentMedia = { ...assets };
    this.channel.port2.onmessage = (event) => {
      const data = event.data as StandInFrame['received'][number];
      this.received.push(data);
      if (data.type !== 'formlogic:update' || this.silent) return;
      // Absent or null: as last sent. `{}`: no media. Media sent with a version that is kept back is not lost.
      if (data.assets !== undefined && data.assets !== null) this.sentMedia = { ...data.assets };
      this.channel.port2.postMessage({ type: 'updated', id: data.id, ...this.take(data.client ?? {}) });
    };
  }

  private take(client: Record<string, string>): UpdateAnswer {
    const differs = (a: Record<string, string>, b: Record<string, string>) => [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((path) => a[path] !== b[path]).sort();
    const files = differs(client, this.running);
    const media = differs(this.sentMedia, this.runningMedia).length > 0;
    if (!files.length && !media) return { outcome: 'unchanged', changed: [] };
    // Another main page is another app to this frame: it asks for a new one.
    if (files.includes('manifest.json')) return { outcome: 'reopen', reason: 'the manifest names another main page' };
    // A version that does not start: the last one stays, on its own files and media.
    if (Object.values(client).some((text) => text.includes(BROKEN))) return { outcome: 'kept', changed: [], problem: PROBLEM };
    this.running = { ...client };
    this.runningMedia = { ...this.sentMedia };
    if (files.some((path) => /\.(js|logic|py)$/.test(path))) return { outcome: 'restarted', changed: ['logic'], reason: 'the logic changed', carried: true };
    return { outcome: 'updated', changed: [...(files.length ? ['page'] : []), ...(media ? ['media'] : [])] };
  }

  close(): void {
    this.channel.port1.close();
    this.channel.port2.close();
  }
}

const APP = { 'manifest.json': '{"main":"index.html"}', 'index.html': '<App><Heading>Tasks</Heading></App>', 'main.js': 'let tasks = []\n' };
const open: StandInFrame[] = [];
afterEach(() => {
  for (const frame of open.splice(0)) frame.close();
});

const LOGO = { 'assets/logo.png': 'AAAA' };

/** A frame with the app open in it (`init` carried `assets`) and the parent's link to it, wired as preview.ts wires them. */
function opened(says = { shown: true, update: true }, assets: Record<string, string> = {}) {
  const frame = new StandInFrame(APP, assets);
  open.push(frame);
  const link = new LiveFrame('', frame.parentPort, says, assets);
  frame.parentPort.onmessage = (event) => {
    link.take(event.data);
  };
  link.drawn = true;
  return { frame, link };
}

describe('what a frame announces in formlogic:ready', () => {
  it('is read from `shown: 1` and `update: 1`, and a frame that says neither (v0.0.32) is taken as doing neither', () => {
    expect(frameSays({ type: 'formlogic:ready', shown: 1, update: 1 })).toEqual({ shown: true, update: true });
    expect(frameSays({ type: 'formlogic:ready', shown: 1 })).toEqual({ shown: true, update: false });
    // what the installed runtime posts
    expect(frameSays({ type: 'formlogic:ready', nativeProtocol: 1, storageProtocol: 1, zipp: {}, engines: {} })).toEqual({ shown: false, update: false });
    // only the number 1 is a yes
    expect(frameSays({ shown: true, update: '1' })).toEqual({ shown: false, update: false });
    expect(frameSays(null)).toEqual({ shown: false, update: false });
  });
});

describe('an update sent to the open frame', () => {
  it('posts every text file on the port, with an id, and resolves with the answer that carries that id', async () => {
    const { frame, link } = opened();
    const next = { ...APP, 'index.html': '<App><Heading>Groceries</Heading></App>' };
    const answer = await link.update(next, {});
    expect(answer).toEqual({ outcome: 'updated', changed: ['page'] });
    // (an app with no media, as `init` had it: nothing to say of media, so `assets` is left out)
    expect(frame.received).toEqual([{ type: 'formlogic:update', id: 1, client: next }]);
    // the next one has the next id, and its own answer
    expect(await link.update(next, {})).toEqual({ outcome: 'unchanged', changed: [] });
    expect(frame.received[1].id).toBe(2);
  });

  it('leaves the media out while it is as the frame last had it, and sends all of it when any of it differs', async () => {
    const { frame, link } = opened(undefined, LOGO);
    const next = { ...APP, 'index.html': '<App><Heading>Groceries</Heading></App>' };
    // the same media as `init` carried (an equal copy, as the preview makes one for every change): left out
    await link.update(next, { ...LOGO });
    expect('assets' in frame.received[0]).toBe(false);
    expect(frame.runningMedia).toEqual(LOGO);
    // a picture added: the whole set, the old one with it
    const two = { ...LOGO, 'assets/photo.jpg': 'BBBB' };
    expect(await link.update(next, two)).toEqual({ outcome: 'updated', changed: ['media'] });
    expect(frame.received[1].assets).toEqual(two);
    // the same again: left out, and the frame keeps both
    await link.update(next, { ...two });
    expect('assets' in frame.received[2]).toBe(false);
    expect(frame.runningMedia).toEqual(two);
    // a picture changed in place (the same path, other bytes) is a difference
    await link.update(next, { ...two, 'assets/photo.jpg': 'CCCC' });
    expect(frame.received[3].assets).toEqual({ ...LOGO, 'assets/photo.jpg': 'CCCC' });
    // all of it removed: `{}`, which is "no media", not "as before"
    expect(await link.update(next, {})).toEqual({ outcome: 'updated', changed: ['media'] });
    expect(frame.received[4].assets).toEqual({});
    expect(frame.runningMedia).toEqual({});
  });

  it('media sent with a version that is kept back comes with the next version, which leaves it out', async () => {
    const { frame, link } = opened(undefined, LOGO);
    const two = { ...LOGO, 'assets/photo.jpg': 'BBBB' };
    const broken = { ...APP, 'main.js': `${APP['main.js']}${BROKEN}\n` };
    expect((await link.update(broken, two))?.outcome).toBe('kept');
    expect(frame.received[0].assets).toEqual(two);
    // still the last working version, on its own media
    expect(frame.runningMedia).toEqual(LOGO);
    // the fix: the media has not changed since it was sent, so it is not sent again, and the frame has it
    const fixed = { ...APP, 'index.html': '<App><Heading>Photos</Heading></App>' };
    expect((await link.update(fixed, { ...two }))?.outcome).toBe('updated');
    expect('assets' in frame.received[1]).toBe(false);
    expect(frame.runningMedia).toEqual(two);
  });

  it('a change to the logic starts the app again, and says why and whether its values were carried', async () => {
    const { link } = opened();
    const answer = await link.update({ ...APP, 'main.js': 'let tasks = []\nlet title = "x"\n' }, {});
    expect(answer).toEqual({ outcome: 'restarted', changed: ['logic'], reason: 'the logic changed', carried: true });
    expect(updateSaid(answer!, '14:02:07')).toMatchObject({ status: 'live · started again 14:02:07', kind: 'ok', ok: true });
    expect(updateSaid(answer!, '14:02:07').title).toContain('the logic changed');
    expect(updateSaid(answer!, '14:02:07').title).toContain('Its values were carried over');
  });

  it('a version that does not start is kept out: the frame keeps the last working one, and the problem comes back', async () => {
    const { frame, link } = opened();
    const answer = await link.update({ ...APP, 'main.js': `${APP['main.js']}${BROKEN}\n` }, {});
    expect(answer).toEqual({ outcome: 'kept', changed: [], problem: PROBLEM });
    // what is on screen is still the version from before
    expect(frame.running).toEqual(APP);
    const said = updateSaid(answer!, '14:02:07');
    expect(said).toMatchObject({ status: KEPT_STATUS, kind: 'error', ok: false });
    // the frame is still the one to send the fix to
    expect(link.takes('')).toBe(true);
    expect(await link.update(APP, {})).toEqual({ outcome: 'unchanged', changed: [] });
  });

  it('`reopen` comes back as an answer the caller opens a new frame for, and so does an outcome this page does not know', async () => {
    const { link } = opened();
    expect(await link.update({ ...APP, 'manifest.json': '{"main":"home.html"}' }, {})).toEqual({ outcome: 'reopen', reason: 'the manifest names another main page' });
    expect(readUpdateAnswer({ type: 'updated', id: 4, outcome: 'morphed' })).toEqual({ id: 4, outcome: 'reopen' });
  });

  it("reads the frame's answer as it sends it: `changed` as given, whatever it names; an answer with no usable id is no answer", () => {
    expect(readUpdateAnswer({ type: 'updated', id: 2, outcome: 'updated', changed: ['page', 'styles', 'media', 'theme', 'something-new'] })).toEqual({ id: 2, outcome: 'updated', changed: ['page', 'styles', 'media', 'theme', 'something-new'] });
    expect(readUpdateAnswer({ type: 'updated', id: 3, outcome: 'restarted', changed: ['logic'], reason: 'the logic changed', carried: false })).toEqual({ id: 3, outcome: 'restarted', changed: ['logic'], reason: 'the logic changed', carried: false });
    // (the frame answers a message whose id it cannot use with `id: null`: this side's ids are numbers, so that is
    // never one of its own, and the update it was for, if any, runs to its deadline and a new frame)
    expect(readUpdateAnswer({ type: 'updated', id: null, outcome: 'reopen', reason: 'no usable id' })).toBeNull();
  });

  it('no answer within the deadline resolves with null (the caller opens a new frame), and a late answer is dropped', async () => {
    const { frame, link } = opened();
    frame.silent = true;
    const started = Date.now();
    expect(await link.update(APP, {}, 60)).toBeNull();
    expect(Date.now() - started).toBeGreaterThanOrEqual(50);
    // the answer to that update arriving afterwards changes nothing and is not mistaken for another's
    expect(link.take({ type: 'updated', id: 1, outcome: 'updated' })).toBe(true);
    frame.silent = false;
    expect(await link.update(APP, {})).toEqual({ outcome: 'unchanged', changed: [] });
    expect(UPDATE_DEADLINE_MS).toBe(8000);
  });

  it('a frame that is closed while an update waits answers it as unanswered, and takes no more', async () => {
    const { frame, link } = opened();
    frame.silent = true;
    const pending = link.update(APP, {});
    link.close();
    expect(await pending).toBeNull();
    expect(link.takes('')).toBe(false);
    expect(await link.update(APP, {})).toBeNull();
    // (only the first reached the frame: nothing is posted once the link is closed)
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frame.received.length).toBe(1);
  });

  it('a port that cannot be posted to resolves with null instead of throwing', async () => {
    const link = new LiveFrame('', { postMessage: () => { throw new Error('closed'); } }, { shown: true, update: true });
    link.drawn = true;
    expect(await link.update(APP, {})).toBeNull();
  });
});

describe('when a change is sent as an update, and when a new frame is opened as before', () => {
  it('only a frame that announced `update`, has the app drawn, did not fail to open, and holds the same app', () => {
    const { link } = opened();
    expect(link.takes('')).toBe(true);
    // another app folder is another app
    expect(link.takes('apps/cat')).toBe(false);
    // the installed runtime: it never announced it, so every change opens a new frame, exactly as today
    expect(opened({ shown: false, update: false }).link.takes('')).toBe(false);
    expect(opened({ shown: true, update: false }).link.takes('')).toBe(false);
    // not before the app is drawn, and not in a frame whose app did not open
    const early = opened().link;
    early.drawn = false;
    expect(early.takes('')).toBe(false);
    const failed = opened().link;
    failed.failed = true;
    expect(failed.takes('')).toBe(false);
  });

  it('other messages on the port are not taken for answers', () => {
    const { link } = opened();
    for (const message of [{ type: 'shown' }, { type: 'error', reason: 'x', phase: 'runtime' }, { type: 'call', id: 1 }, { type: 'updated' }, { type: 'updated', id: 'one', outcome: 'updated' }, null, 'updated']) {
      expect(link.take(message), JSON.stringify(message)).toBe(false);
    }
  });

  it('preview.ts takes the update path only then, keeps the port for it, and falls back on `reopen` and on no answer', () => {
    // (the wiring is DOM code, held here by its text; the e2e runs it against the real runtime)
    expect(PREVIEW).toContain("link.takes(target.root)");
    expect(PREVIEW).toContain("if (!answer || answer.outcome === 'reopen') {");
    expect(PREVIEW).toMatch(/if \(!answer \|\| answer\.outcome === 'reopen'\) \{\s+this\.leaveFrame\(\);\s+return this\.renderApp\(root, generation\);/);
    expect(PREVIEW).toContain('new LiveFrame(root, channel.port1, frameSays(event.data), assets)');
    // Reload, another project and another theme never update in place: the frame cannot see those, so it is not asked
    expect(PREVIEW).toContain('onclick: () => this.render(true)');
    // not before `shown`: the link takes nothing until the app is drawn (the frame would answer `reopen`)
    expect(PREVIEW).toMatch(/if \(data\?\.type === 'shown'\) \{\s+if \(link\.drawn\) return;\s+link\.drawn = true;/);
    // a manifest that is there and unreadable goes to the frame, which keeps the last working version
    expect(PREVIEW).toContain('(isSoftnApp(this.vfs, target.root) || this.vfs.exists(manifestPath(target.root)))');
  });
});

describe('how an answer reads in the preview bar', () => {
  it('says updated and started again with the time, why in the tooltip, and that nothing changed', () => {
    expect(updateSaid({ outcome: 'updated' }, '14:02:07')).toMatchObject({ status: 'live · updated 14:02:07', kind: 'ok', ok: true });
    // The time, so that two restarts in a row do not read the same; the reason is too long for the bar.
    expect(updateSaid({ outcome: 'restarted' }, '14:02:07')).toMatchObject({ status: 'live · started again 14:02:07', title: 'Its logic changed.' });
    expect(updateSaid({ outcome: 'restarted' }, '14:02:09').status).not.toBe(updateSaid({ outcome: 'restarted' }, '14:02:07').status);
    expect(updateSaid({ outcome: 'restarted', reason: 'its permissions changed', carried: false }, '14:02:07')).toMatchObject({ status: 'live · started again 14:02:07', ok: true });
    expect(updateSaid({ outcome: 'restarted', reason: 'its permissions changed', carried: false }, '14:02:07').title).toContain('its permissions changed');
    expect(updateSaid({ outcome: 'restarted', carried: false }, '14:02:07').title).toContain('could not be carried over');
    // The runtime's own sentence already says what became of the values: it is not said twice.
    expect(updateSaid({ outcome: 'restarted', reason: 'Its logic changed. Its variables were carried over, and its start-up ran again.', carried: true }, '1').title).toBe('Its logic changed. Its variables were carried over, and its start-up ran again.');
    expect(updateSaid({ outcome: 'unchanged' }, '14:02:07')).toMatchObject({ status: 'live · 14:02:07', ok: true });
    // every status of a working app says "live", every other says "error": what the bar's colour and the tests go by
    for (const outcome of ['updated', 'restarted', 'unchanged'] as const) expect(updateSaid({ outcome }, '1').status).toMatch(/^live/);
    expect(KEPT_STATUS).toBe('error · last version kept');
    expect(updateSaid({ outcome: 'kept', problem: 'x' }, '1').title).toContain('still shows the last one that worked');
  });
});

describe("`shown` and an error's `phase`", () => {
  it('preview.ts declares the app live on `shown` where the frame announced it, and after the wait where it did not', () => {
    expect(PREVIEW).toContain("if (data?.type === 'shown') {");
    expect(PREVIEW).toMatch(/this\.settleTimer = link\.says\.shown\s+\? setTimeout\(/);
    expect(PREVIEW).toContain('}, SETTLE_MS);');
    expect(PREVIEW).toContain('the app did not finish opening within');
  });

  it("an error with phase 'runtime' is a problem of an app that is drawn, and any other is an app that did not open", () => {
    const runtime = PREVIEW.indexOf("data?.type === 'error' && data.phase === 'runtime'");
    const load = PREVIEW.indexOf("} else if (data?.type === 'error') {");
    expect(runtime).toBeGreaterThan(0);
    expect(load).toBeGreaterThan(runtime);
    // runtime: the banner and "Fix with agent" (fromApp), the app still drawn and still updatable
    expect(PREVIEW.slice(runtime, load)).toContain('link.drawn = true;');
    expect(PREVIEW.slice(runtime, load)).toContain('this.fromApp(');
    expect(PREVIEW.slice(runtime, load)).not.toContain('link.failed');
    // load: nothing in this frame to update
    expect(PREVIEW.slice(load, load + 400)).toContain('link.failed = true;');
  });
});

describe('a problem as the banner says it (B1)', () => {
  it('leads with the file and the part a person can act on, without the runtime\'s prefixes', () => {
    // the words the runtime gives for a brace that is never closed, as the preview showed them cut off twice
    const raw = '[SoftN] Error loading script: SyntaxError: expected a property name, at the end of the logic: a brace, bracket or string opened before it is never closed';
    expect(plainProblem(raw, 'main.js')).toBe('main.js: a brace, bracket or string opened before it is never closed.');
    // (with no file known, it is a sentence of its own)
    expect(plainProblem(raw)).toBe('A brace, bracket or string opened before it is never closed.');
    // the same from the syntax check, which names the file itself
    expect(plainProblem(`main.js: syntax error: ${raw.slice(raw.indexOf('expected'))}`)).toBe('main.js: a brace, bracket or string opened before it is never closed.');
    // and from a frame that kept the last version
    expect(plainProblem(PROBLEM)).toBe('main.js: a brace, bracket or string opened before it is never closed.');
  });

  it('keeps what says what is wrong, and only the first line', () => {
    expect(plainProblem('[SoftN] Error loading script: ReferenceError: nothere is not defined (in <script>)')).toBe('ReferenceError: nothere is not defined (in <script>)');
    expect(plainProblem('[SoftN] Parse error at line 32:1: A "," is missing before "let": arguments are separated by commas')).toBe('Parse error at line 32:1: A "," is missing before "let": arguments are separated by commas.');
    expect(plainProblem('Error: boom from addTask\n    at addTask (main.js:5)')).toBe('Error: boom from addTask.');
    expect(plainProblem('pages/home.html: the page is missing')).toBe('pages/home.html: the page is missing.');
    expect(plainProblem('')).toBe('Something went wrong.');
  });

  it('the file is named for an error of the logic failing to compile, and not for any other', () => {
    expect(isLoadSyntaxError('[SoftN] Error loading script: SyntaxError: expected a property name')).toBe(true);
    expect(isLoadSyntaxError('SyntaxError: unexpected token')).toBe(true);
    expect(isLoadSyntaxError('[SoftN] Error loading script: ReferenceError: nothere is not defined')).toBe(false);
    expect(isLoadSyntaxError('Error: boom from addTask')).toBe(false);
  });

  it('the status beside the size picker says "error" and no more; the banner carries the message', () => {
    expect(PREVIEW).toContain("this.setStatus(this.kept ? KEPT_STATUS : 'error', 'error'");
    expect(PREVIEW).not.toMatch(/setStatus\(`error: /);
  });
});

describe('the note beside the size picker when the pane is phone-sized (R2)', () => {
  it('says the width and that it is the phone layout at 768 px and under, and nothing above', () => {
    expect(PHONE_LAYOUT_MAX).toBe(768);
    expect(fitNote(651)).toBe('651 px wide: phone layout');
    expect(fitNote(768)).toBe('768 px wide: phone layout');
    expect(fitNote(650.6)).toBe('651 px wide: phone layout');
    expect(fitNote(769)).toBe('');
    expect(fitNote(1366)).toBe('');
    // a pane that is not on screen has no width, and no note
    expect(fitNote(0)).toBe('');
  });

  it('offers the Laptop size there, and the agent is told the same in both of its guides', () => {
    expect(PREVIEW).toContain('this.setViewport(VIEWPORTS.laptop)');
    const sentence = 'The preview pane is usually under 768 px wide, where a Sidebar is a Menu button and a wide Table is cards: use preview_viewport (laptop) to check the wide layout, and open Menu before looking for a page\'s link.';
    expect(AGENT.split(sentence).length - 1).toBe(2);
    expect(AGENT.slice(AGENT.indexOf('const APP_GUIDE = `'), AGENT.indexOf('const WEB_GUIDE = `'))).toContain(sentence);
    expect(AGENT.slice(AGENT.indexOf('const APP_BRIEF = `'), AGENT.indexOf('const APP_3D_BRIEF = `'))).toContain(sentence);
  });
});

describe("the engine SoftN's runtime needs (R1)", () => {
  const manifest = { zipp: { version: '0.0.35', sha256: 'a'.repeat(64), release: 'v0.0.35' }, files: {} };
  const own = (digest: string, release = 'v0.0.33') => ({ release, files: { 'zipp_wasm_bg.wasm': digest } });

  it('is the Agent\'s own when the runtime names the same module: nothing to say', () => {
    expect(engineMismatch(manifest, own('a'.repeat(64), 'v0.0.35'))).toBeNull();
  });

  it('is said in words a person can act on when the runtime was built with another ZIPP, naming both releases', () => {
    expect(engineMismatch(manifest, own('b'.repeat(64)))).toBe("SoftN's runtime was built with ZIPP v0.0.35; this Agent has v0.0.33. Run npm run fetch:zipp and npm run fetch:softn.");
    // a manifest from before `release` was recorded names its version
    expect(engineMismatch({ zipp: { version: '0.0.31', sha256: 'c'.repeat(64) } }, own('b'.repeat(64)))).toContain('built with ZIPP v0.0.31; this Agent has v0.0.33.');
  });

  it('is not guessed at where one side does not say which module it has', () => {
    expect(engineMismatch({}, own('b'.repeat(64)))).toBeNull();
    expect(engineMismatch(manifest, null)).toBeNull();
    expect(engineMismatch(null, null)).toBeNull();
  });

  it('there is no fallback to an engine the runtime carries: it carries none', () => {
    expect(PREVIEW).not.toContain('zipp_wasm_bg-');
    expect(PREVIEW).not.toContain("SoftN's Zipp engine could not be loaded");
    expect(PREVIEW).toContain('const mismatch = engineMismatch(manifest, own);');
    // and the message is shown where the app would be
    expect(PREVIEW).toMatch(/this\.failStatus\(message\);\s+this\.showMessage\(message\);/);
  });
});
