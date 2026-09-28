import { describe, expect, it } from 'vitest';
import { runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import { findApps } from '../../src/softn/softn';
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
