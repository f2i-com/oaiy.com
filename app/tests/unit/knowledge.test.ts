import { describe, expect, it } from 'vitest';
import { runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import { checkEvents, componentEvents } from '../../src/softn/knowledge';
import { findApps, handlersIn } from '../../src/softn/softn';
import { Vfs } from '../../src/vfs/vfs';

const ctx = () => ({ vfs: new Vfs(), gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} } });
const call = async (c: ReturnType<typeof ctx>, name: string, input: Record<string, unknown> = {}) => runTool({ id: 'c', name, input }, c);

describe('the SoftN reference harness', () => {
  it('maps what there is: guide sections, published guides, components, examples', async () => {
    const map = await call(ctx(), 'softn_docs');
    expect(map.isError).toBe(false);
    for (const want of ['## The writing guide', 'xdb-data:', 'state-events:', 'Button', 'Table', 'twenty48:', 'notes:']) expect(map.content).toContain(want);
  });

  it('reads a topic: the writing guide, a section of it, a published guide and one of its sections', async () => {
    const c = ctx();
    expect((await call(c, 'softn_docs', { topic: 'guide' })).content.length).toBeGreaterThan(10_000);
    const section = await call(c, 'softn_docs', { topic: 'guide#mistakes' });
    expect(section.isError).toBe(false);
    expect(section.content.toLowerCase()).toContain('mistake');
    const page = await call(c, 'softn_docs', { topic: 'xdb-data' });
    expect(page.content).toMatch(/^# Local app data with XDB/);
    const missing = await call(c, 'softn_docs', { topic: 'no-such-guide' });
    expect(missing.isError).toBe(true);
    expect(missing.content).toContain('xdb-data');
    // The older way still works.
    expect((await call(c, 'softn_docs', { section: 'mistakes' })).content).toBe(section.content);
  });

  it('looks up components by name, forgiving case and angle brackets', async () => {
    const found = await call(ctx(), 'softn_components', { names: ['button', '<Table>'] });
    expect(found.content).toContain('## Button');
    expect(found.content).toContain('## Table');
    const unknown = await call(ctx(), 'softn_components', { names: ['Blink'] });
    expect(unknown.isError).toBe(true);
    expect(unknown.content).toContain('The registered components are');
  });

  it('searches the guides, the components and the examples, and says how to open each hit', async () => {
    const hits = await call(ctx(), 'softn_docs', { search: 'setInterval timer' });
    expect(hits.isError).toBe(false);
    expect(hits.content).toMatch(/^\d+ places mention/);
    expect(hits.content).toMatch(/open: softn_(docs|components|examples) /);
    const storage = await call(ctx(), 'softn_docs', { search: 'xdb collection' });
    expect(storage.content).toContain('softn_docs topic "xdb-data');
    expect((await call(ctx(), 'softn_docs', { search: 'zzqqxx' })).content).toContain('Nothing matches');
  });

  it('lists, reads and installs example apps', async () => {
    const c = ctx();
    expect((await call(c, 'softn_examples')).content).toContain('twenty48');
    const files = await call(c, 'softn_examples', { name: 'three-demo' });
    expect(files.content).toContain('ui/main.ui');
    expect(files.content).toContain('manifest.json:');
    const read = await call(c, 'softn_examples', { name: 'three-demo', file: 'logic/main.logic' });
    expect(read.content).toMatch(/^three-demo\/logic\/main\.logic \(\d+ lines\)/);
    const installed = await call(c, 'softn_examples', { name: 'notes', install_to: 'examples' });
    expect(installed.content).toContain('Copied the example into examples/Notes/');
    expect(findApps(c.vfs)).toEqual(['examples/Notes']);
    expect((await call(c, 'softn_examples', { name: 'nope' })).isError).toBe(true);
  });
});

describe('handlers on components that do not take them', () => {
  const app = (ui: string) => {
    const vfs = new Vfs();
    vfs.writeFile('/apps/game/manifest.json', JSON.stringify({ name: 'Game', version: '1.0.0', main: 'ui/main.ui', files: { ui: ['ui/main.ui'], logic: ['logic/main.logic'] } }), { parents: true });
    vfs.writeFile('/apps/game/logic/main.logic', 'let muted = false\nfunction toggleMute() { muted = !muted }\n', { parents: true });
    vfs.writeFile('/apps/game/ui/main.ui', ui, { parents: true });
    return vfs;
  };

  it('finds each @name on a component\'s opening tag, whatever its values hold', () => {
    const ui = [
      '<logic src="../logic/main.logic" />',
      '<App>',
      '  <Text className={muted ? "a > b" : "c"} @click={toggleMute}>{muted ? "Off" : "On"}</Text>',
      '  <Box',
      '    @click={() => (score > 3 ? win() : lose())}',
      '    @keyDown={onKey} title="mail me @home">',
      '    <Button @click={start} label="x > y">Go</Button>',
      '  </Box>',
      '  <p @click={nothing}>a lower-case tag is the page\'s own</p> 3 < 4 and <Text>plain</Text>',
      '</App>',
    ].join('\n');
    expect(handlersIn(ui)).toEqual([
      { tag: 'Text', event: 'click', line: 3 },
      { tag: 'Box', event: 'click', line: 4 },
      { tag: 'Box', event: 'keyDown', line: 4 },
      { tag: 'Button', event: 'click', line: 7 },
    ]);
  });

  it('knows each component\'s events from its reference', async () => {
    const events = await componentEvents();
    expect(events.get('Button')?.has('click')).toBe(true);
    expect(events.get('Box')?.has('click') && events.get('Box')?.has('keydown')).toBe(true);
    expect(events.get('Text')?.size).toBe(0);
    expect(events.get('Stack')?.size).toBe(0);
  });

  it('says so of a click on a Text, which the runtime drops without a word; softn_check reports it', async () => {
    const vfs = app('<logic src="../logic/main.logic" />\n<App>\n  <Text className="btn" @click={toggleMute}>Sound on</Text>\n  <Box @click={toggleMute} @hover={toggleMute}><Text>ok</Text></Box>\n  <Button @click={toggleMute}>Mute</Button>\n</App>\n');
    const found = await checkEvents(vfs, 'apps/game');
    expect(found).toEqual([
      { level: 'error', file: 'ui/main.ui', message: 'line 3: <Text> takes no events, so its @click never runs. Use <Button>, or put it inside a <Box @click={…}>.' },
      expect.objectContaining({ level: 'warning', file: 'ui/main.ui', message: expect.stringMatching(/^line 4: <Box> has no @hover, so it never runs; its events: @click, /) }),
    ]);
    const checked = await runTool({ id: 'c', name: 'softn_check', input: { app: 'apps/game' } }, { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} } });
    expect(checked.content).toContain('ERROR ui/main.ui: line 3: <Text> takes no events, so its @click never runs');
    expect(checked.content).toContain('warning ui/main.ui: line 4: <Box> has no @hover');
    // a page that keeps to what its components take has nothing said of it
    expect(await checkEvents(app('<App>\n  <Button @click={toggleMute}>Mute</Button>\n  <Card @click={toggleMute}>x</Card>\n</App>\n'), 'apps/game')).toEqual([]);
  });

  it('finds none in the example apps', async () => {
    const c = ctx();
    for (const slug of ['notes', 'twenty48']) {
      expect((await call(c, 'softn_examples', { name: slug, install_to: `examples/${slug}` })).isError).toBe(false);
    }
    for (const root of findApps(c.vfs)) expect(await checkEvents(c.vfs, root)).toEqual([]);
  });
});

