import { describe, expect, it } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { readView } from '../../src/vfs/readView';
import { runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';

function desk() {
  const vfs = new Vfs();
  vfs.writeFile('/brief.md', '# The brief\n', { parents: true });
  vfs.writeFile('/knowledge/services.md', 'Lawn mowing\n', { parents: true });
  vfs.writeFile('/outreach/confirm-friday/results.md', '| Jane Smith | completed | coming: yes |\n', { parents: true });
  vfs.writeFile('/outreach/confirm-friday/results.csv', 'name,number\nJane Smith,+61412345678\n', { parents: true });
  return vfs;
}

describe("the front desk's files as its calls, texts and tasks see them", () => {
  it('hides /outreach from every read: exists, stat, list, walk and reading', () => {
    const vfs = desk();
    const view = readView(vfs);
    expect(view.exists('/outreach')).toBe(false);
    expect(view.exists('/outreach/confirm-friday/results.md')).toBe(false);
    expect(view.stat('/outreach/confirm-friday/results.csv')).toBeNull();
    expect(view.list('/').map((e) => e.name)).toEqual(['brief.md', 'knowledge']);
    expect(view.walk('/').entries.map((e) => e.path)).toEqual(['brief.md', 'knowledge', 'knowledge/services.md']);
    expect(() => view.readText('/outreach/confirm-friday/results.md')).toThrow(/ENOENT/);
    expect(() => view.list('/outreach')).toThrow(/ENOENT/);
    expect(view.files().map(([p]) => p)).toEqual(['brief.md', 'knowledge/services.md']);
    // The rest reads as ever, and the runner's own view has it all.
    expect(view.readText('/knowledge/services.md')).toBe('Lawn mowing\n');
    expect(vfs.exists('/outreach/confirm-friday/results.md')).toBe(true);
  });

  it('refuses to write under /outreach, and writes anything else through', () => {
    const vfs = desk();
    const view = readView(vfs);
    expect(() => view.writeFile('/outreach/x.md', 'mine')).toThrow(/private to the runner/);
    expect(() => view.remove('/outreach', true)).toThrow(/private to the runner/);
    expect(() => view.rename('/knowledge/services.md', '/outreach/services.md')).toThrow(/private to the runner/);
    expect(vfs.exists('/outreach/confirm-friday/results.md')).toBe(true);
    view.writeFile('/notes/task.md', 'done', { parents: true });
    expect(vfs.readText('/notes/task.md')).toBe('done');
  });

  it("a call's read_file, grep and glob find nothing of it", async () => {
    const view = readView(desk());
    const ctx = { vfs: view, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} } };
    const read = await runTool({ id: 'r', name: 'read_file', input: { path: '/outreach/confirm-friday/results.md' } }, ctx as never);
    expect(read.isError).toBe(true);
    const grep = await runTool({ id: 'g', name: 'grep', input: { pattern: 'Jane' } }, ctx as never);
    expect(grep.content).not.toContain('Jane');
    const glob = await runTool({ id: 'l', name: 'glob', input: { pattern: '**/*.csv' } }, ctx as never);
    expect(glob.content).not.toContain('results.csv');
  });
});
