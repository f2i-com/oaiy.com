import { describe, expect, it } from 'vitest';
import { NetGate, normalizeHost } from '../../src/gate/netgate';
import { Vfs, normalizePath } from '../../src/vfs/vfs';
import { globRegex, relativeTo } from '../../src/sandbox/host';

describe('the network gate', () => {
  it('normalizes hosts from URLs and patterns', () => {
    expect(normalizeHost('https://User@Docs.RS:443/x?y')).toBe('docs.rs');
    expect(normalizeHost('*.github.com')).toBe('github.com');
    expect(normalizeHost('[::1]:8080')).toBe('::1');
  });

  it('decides by mode and lists together', () => {
    const gate = new NetGate();
    expect(gate.check('example.com', 't').ok).toBe(true);
    gate.denyHost('evil.example');
    expect(gate.check('a.evil.example', 't').ok).toBe(false);
    gate.setMode('allowlist');
    expect(gate.check('example.com', 't').ok).toBe(false);
    gate.allowHost('https://docs.rs/');
    expect(gate.check('static.docs.rs', 't').ok).toBe(true);
    expect(gate.check('xdocs.rs', 't').ok).toBe(false);
    gate.setMode('blocked');
    expect(gate.check('docs.rs', 't').ok).toBe(false);
    const status = gate.status();
    expect(status.allowed).toBe(2);
    expect(status.blocked).toBe(4);
    expect(status.recentBlocked.at(-1)).toEqual({ host: 'docs.rs', via: 't' });
  });
});

describe('the virtual filesystem', () => {
  it('cannot be left: .. stops at the root', () => {
    expect(normalizePath('../../etc/passwd')).toBe('etc/passwd');
    expect(normalizePath('C:\\Windows\\win.ini')).toBe('C:/Windows/win.ini');
    expect(normalizePath('a/./b/../c', '/x')).toBe('x/a/c');
    expect(normalizePath('b', '/x/y')).toBe('x/y/b');
  });

  it('writes, lists, walks, copies, renames and removes', () => {
    const vfs = new Vfs();
    const changes: string[] = [];
    vfs.onChange((c) => changes.push(`${c.type}:${'path' in c ? c.path : ''}`));
    vfs.writeFile('/src/a.ts', 'a', { parents: true });
    vfs.writeFile('/node_modules/x/index.js', 'x', { parents: true });
    expect(() => vfs.writeFile('/missing/b.ts', 'b')).toThrow(/ENOENT/);
    expect(vfs.list('/').map((e) => e.name)).toEqual(['node_modules', 'src']);
    expect(vfs.walk('/').entries.map((e) => e.path)).toEqual(['src', 'src/a.ts']);
    expect(vfs.walk('/', { includeIgnored: true }).entries).toHaveLength(5);
    vfs.copy('/src', '/lib');
    expect(vfs.readText('/lib/a.ts')).toBe('a');
    vfs.rename('/lib', '/lib2');
    expect(vfs.exists('/lib')).toBe(false);
    expect(() => vfs.remove('/src')).toThrow(/ENOTEMPTY/);
    vfs.remove('/src', true);
    expect(vfs.exists('/src/a.ts')).toBe(false);
    expect(() => vfs.remove('/')).toThrow(/project root/);
    expect(changes).toContain('write:src/a.ts');
    const v = vfs.version('/lib2/a.ts');
    vfs.writeFile('/lib2/a.ts', 'more', { append: true });
    expect(vfs.readText('/lib2/a.ts')).toBe('amore');
    expect(vfs.version('/lib2/a.ts')).toBe(v + 1);
  });

  it('refuses binary as text', () => {
    const vfs = new Vfs();
    vfs.writeFile('/b.bin', new Uint8Array([0xff, 0xfe, 0x00]));
    expect(() => vfs.readText('/b.bin')).toThrow(/not UTF-8/);
    expect(vfs.isText('/b.bin')).toBe(false);
  });
});

describe('sandbox path helpers', () => {
  it('globs and relative paths', () => {
    expect(globRegex('src/**/*.ts', true).test('src/a/b/c.ts')).toBe(true);
    expect(globRegex('src/**/*.ts', true).test('src/c.ts')).toBe(true);
    expect(globRegex('*.ts', true).test('src/c.ts')).toBe(false);
    expect(relativeTo('/src', '/src/x/y.js')).toBe('x/y.js');
    expect(relativeTo('/', 'a.py')).toBe('a.py');
  });
});

describe('softn_import', () => {
  it('unpacks a .softn from the project into a new folder, never over existing files', async () => {
    const { zipSync, strToU8 } = await import('fflate');
    const { runTool } = await import('../../src/agent/tools');
    const vfs = new Vfs();
    vfs.writeFile('/uploads/Tasks.softn', zipSync({
      'manifest.json': strToU8(JSON.stringify({ name: 'Tasks', version: '1.0.0', main: 'ui/main.ui' })),
      'ui/main.ui': strToU8('<App><Text>hi</Text></App>'),
      '../escape.txt': strToU8('no'),
    }), { parents: true });
    vfs.writeFile('/uploads/notes.softn', 'not a zip', { parents: true });
    const ctx = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} } };
    const run = (input: Record<string, unknown>) => runTool({ id: 'c', name: 'softn_import', input }, ctx);

    const first = await run({ path: 'uploads/Tasks.softn' });
    expect(first.isError).toBe(false);
    expect(first.content).toContain('into Tasks/ (2 files)');
    expect(first.content).toContain('main ui/main.ui');
    const second = await run({ path: '/uploads/Tasks.softn', parent: 'copies' });
    expect(second.content).toContain('into copies/Tasks/');
    const third = await run({ path: 'uploads/Tasks.softn' });
    expect(third.content).toContain('into Tasks-2/');
    expect(vfs.exists('/escape.txt')).toBe(false);
    expect((await run({ path: 'uploads/notes.softn' })).isError).toBe(true);
  });
});

describe('local addresses', () => {
  it('are recognised however they are written', async () => {
    const { isLocalHost } = await import('../../src/sandbox/host');
    for (const h of ['localhost', '127.0.0.1', '[::1]', '::ffff:127.0.0.1', '[::ffff:7f00:1]', '::ffff:c0a8:101', '10.1.2.3', '192.168.1.1', '169.254.169.254', 'fe80::1', 'feb0::1', 'fd00::1', 'printer.local', 'x.internal']) expect(isLocalHost(h)).toBe(true);
    for (const h of ['example.com', '8.8.8.8', '[2606:4700::1111]', '::ffff:808:808', 'fe00.example']) expect(isLocalHost(h)).toBe(false);
  });
});
