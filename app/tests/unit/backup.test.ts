// The Agent's part of OAIY Desktop's backup (src/desktop/backup.ts): what it hands the desktop, in parts,
// when the desktop asks for a backup, and what it does with a restore the desktop staged. No browser:
// an in-memory stand-in for the page's storage and a stand-in for the desktop's local API.
import { strToU8, unzipSync, zipSync } from 'fflate';
import { describe, expect, it, vi } from 'vitest';
import {
  PART_BYTES,
  applyPendingRestore,
  checkEntry,
  exportAgentStorage,
  installBackupHooks,
  mergeSettings,
  type AgentStorage,
  type BackupWindow,
  type DoneBody,
  type FetchLike,
  type PartTarget,
  type StoredFile,
} from '../../src/desktop/backup';
import { DEFAULT_AGENT_SETTINGS, DEFAULT_MESSAGE_SETTINGS, type Settings } from '../../src/settings';
import { EMPTY_MEDIA } from '../../src/agent/media';

const enc = new TextEncoder();
const dec = new TextDecoder();
const bytes = (s: string) => enc.encode(s);

/** Bytes that do not compress (so a part of the archive is as big as the file it holds). */
function noise(n: number, seed = 12345): Uint8Array {
  const out = new Uint8Array(n);
  let x = seed >>> 0 || 1;
  for (let i = 0; i < n; i++) {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    out[i] = x & 255;
  }
  return out;
}

const sha = async (data: Uint8Array) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data as unknown as BufferSource)), (b) => b.toString(16).padStart(2, '0')).join('');

const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);

function joined(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

const KEY = 'sk-test-provider-key-1';
const MEDIA_KEY = 'media-api-token-2';
const DESKTOP_TOKEN = 'desktop-token-not-for-backups';

function settings(over: Partial<Settings> = {}): Settings {
  return {
    providers: [{ id: 'p1', type: 'openai', name: 'Main', apiKey: KEY, modelId: 'model-a' }],
    activeProviderId: 'p1',
    gate: { mode: 'open', allow: [], deny: [] },
    lastProjectId: 'p-kept',
    lastKeptProjectId: 'p-kept',
    agent: { ...DEFAULT_AGENT_SETTINGS },
    media: { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080/v1', apiKey: MEDIA_KEY },
    desktop: { origin: 'http://127.0.0.1:17972', token: DESKTOP_TOKEN },
    messages: { ...DEFAULT_MESSAGE_SETTINGS },
    ...over,
  };
}

class FakeStorage implements AgentStorage {
  files = new Map<string, Uint8Array>();
  settings: Settings = settings();
  written: Settings | null = null;
  unreadable = new Set<string>();
  failWrites = new Set<string>();

  constructor(public events: string[] = []) {}

  put(path: string, data: string | Uint8Array): this {
    this.files.set(path, typeof data === 'string' ? bytes(data) : data);
    return this;
  }

  text(path: string): string | null {
    const f = this.files.get(path);
    return f ? dec.decode(f) : null;
  }

  async projectIds(): Promise<string[]> {
    return [...new Set([...this.files.keys()].filter((p) => p.startsWith('projects/')).map((p) => p.split('/')[1]))];
  }

  async *walk(root: string): AsyncIterable<StoredFile> {
    for (const [path, data] of [...this.files]) {
      if (path !== root && !path.startsWith(`${root}/`)) continue;
      yield {
        path,
        size: data.length,
        read: async () => {
          if (this.unreadable.has(path)) throw new Error('locked');
          return data;
        },
      };
    }
  }

  async readFile(path: string) {
    return this.files.get(path) ?? null;
  }

  async writeFile(path: string, data: Uint8Array) {
    if (this.failWrites.has(path)) throw new Error('disk full');
    this.events.push(`write ${path}`);
    this.files.set(path, data);
  }

  async readSettings() {
    return structuredClone(this.settings);
  }

  async writeSettings(next: Parameters<AgentStorage['writeSettings']>[0]) {
    this.events.push('write settings');
    this.written = { ...this.settings, ...structuredClone(next) };
    this.settings = this.written;
  }
}

/** A storage with a kept project, an incognito one, and the front desk. */
function populated(events: string[] = []): FakeStorage {
  return new FakeStorage(events)
    .put('projects/p-kept/project.json', JSON.stringify({ id: 'p-kept', name: 'Kept', created: 1, updated: 2 }))
    .put('projects/p-kept/chat.json', '[{"role":"user","text":"hello"}]')
    .put('projects/p-kept/files/notes.md', '# notes')
    .put('projects/p-kept/sessions/index.json', '[]')
    .put('projects/p-kept/sessions/person-1.json', '[]')
    .put('projects/p-secret/project.json', JSON.stringify({ id: 'p-secret', name: 'Hidden', incognito: true }))
    .put('projects/p-secret/chat.json', '[{"role":"user","text":"incognito words"}]')
    .put('projects/p-secret/files/private.txt', 'incognito file')
    .put('front-desk/project.json', JSON.stringify({ id: 'front-desk', name: 'Front desk' }))
    .put('front-desk/brief.md', '# the brief')
    .put('front-desk/sessions/person-0491570006.json', '[{"role":"user","text":"call"}]')
    .put('front-desk/sessions/index.json', '[]');
}

class Collector implements PartTarget {
  parts: Uint8Array[] = [];
  seqs: number[] = [];
  doneBody: DoneBody | null = null;
  failPartAt: number | null = null;
  failDone = false;
  attempts = 0;

  async part(seq: number, data: Uint8Array) {
    this.attempts++;
    if (this.failPartAt === seq) throw new Error('the desktop answered 500');
    this.seqs.push(seq);
    this.parts.push(data);
  }

  async done(body: DoneBody) {
    if (this.failDone) throw new Error('the desktop answered 500');
    this.doneBody = body;
  }

  get zip(): Uint8Array {
    return joined(this.parts);
  }

  get entries(): Record<string, Uint8Array> {
    return unzipSync(this.zip);
  }
}

const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status, headers: { 'Content-Type': 'application/json' } });

interface FakeDesktopOptions {
  kind?: 'restore' | 'undo';
  partSize?: number;
  sha256?: string;
  pending?: boolean;
  undoStatus?: number;
  events?: string[];
  meta?: Record<string, unknown>;
}

/** The desktop's side of a restore waiting for the page: what it serves and what the page posts back. */
function fakeDesktop(archive: Uint8Array, options: FakeDesktopOptions = {}) {
  const partSize = options.partSize ?? 700;
  const events = options.events ?? [];
  const posted = { undoParts: [] as Uint8Array[], undoDone: null as DoneBody | null, done: null as Record<string, unknown> | null };
  const calls: string[] = [];
  const headers: Array<Record<string, string>> = [];
  const meta = {
    pending: options.pending ?? true,
    id: 'imp1',
    token: 'sessiontok1',
    kind: options.kind ?? 'restore',
    size: archive.length,
    sha256: options.sha256 ?? '',
    parts: Math.ceil(archive.length / partSize),
    partSize,
    ...options.meta,
  };
  const fetch: FetchLike = async (url, init) => {
    const u = new URL(url);
    const method = init?.method ?? 'GET';
    calls.push(`${method} ${u.pathname}${u.search}`);
    headers.push((init?.headers ?? {}) as Record<string, string>);
    const body = init?.body;
    const raw = typeof body === 'string' ? bytes(body) : (body as Uint8Array | undefined);
    if (u.pathname === '/api/backup/agent-import') return json(meta.pending ? { ...meta, sha256: meta.sha256 || (await sha(archive)) } : { pending: false });
    const part = u.pathname.match(/\/part\/(\d+)$/);
    if (part) return new Response(archive.slice(Number(part[1]) * partSize, (Number(part[1]) + 1) * partSize));
    if (u.pathname.endsWith('/undo-part')) {
      if ((options.undoStatus ?? 200) !== 200) return new Response('no', { status: options.undoStatus });
      events.push('undo-part');
      posted.undoParts.push(raw!);
      return json({ ok: true });
    }
    if (u.pathname.endsWith('/undo-done')) {
      events.push('undo-done');
      posted.undoDone = JSON.parse(dec.decode(raw!));
      return json({ ok: true });
    }
    if (u.pathname.endsWith('/done')) {
      events.push('done');
      posted.done = JSON.parse(dec.decode(raw!));
      return json({ ok: true });
    }
    return new Response('not found', { status: 404 });
  };
  return { fetch, posted, calls, headers, meta };
}

const DESKTOP = { origin: 'http://127.0.0.1:17972', token: 'bearer-token' };

/** An archive of `storage`, as the page makes it. */
async function archiveOf(storage: AgentStorage, includeKeys = true): Promise<Uint8Array> {
  const target = new Collector();
  const report = await exportAgentStorage(storage, target, { includeKeys });
  expect(report.ok).toBe(true);
  return target.zip;
}

function craft(entries: Record<string, string>): Uint8Array {
  const manifest = { 'agent-manifest.json': strToU8(JSON.stringify({ v: 1, kind: 'oaiy-agent-storage', includesKeys: false })) };
  const files: Record<string, Uint8Array> = { ...manifest };
  for (const [name, text] of Object.entries(entries)) files[name] = strToU8(text);
  return zipSync(files);
}

describe('exporting the Agent storage', () => {
  it('sends a zip whose first entry is the manifest, then the projects, the front desk and the settings', async () => {
    const target = new Collector();
    const report = await exportAgentStorage(populated(), target, { includeKeys: false, now: () => new Date('2026-09-30T10:00:00Z') });
    expect(report.ok).toBe(true);
    const names = Object.keys(target.entries);
    expect(names[0]).toBe('agent-manifest.json');
    expect(JSON.parse(dec.decode(target.entries['agent-manifest.json']))).toEqual({ v: 1, kind: 'oaiy-agent-storage', createdAt: '2026-09-30T10:00:00.000Z', includesKeys: false });
    expect(names).toEqual(
      expect.arrayContaining([
        'opfs/projects/p-kept/project.json',
        'opfs/projects/p-kept/chat.json',
        'opfs/projects/p-kept/files/notes.md',
        'opfs/projects/p-kept/sessions/person-1.json',
        'opfs/front-desk/brief.md',
        'opfs/front-desk/sessions/person-0491570006.json',
        'idb/settings.json',
      ]),
    );
    expect(dec.decode(target.entries['opfs/projects/p-kept/files/notes.md'])).toBe('# notes');
    // The projects, the conversations (a project's chat and each thread, not the lists), the files and their size.
    expect(target.doneBody).toMatchObject({ ok: true, parts: target.parts.length, warnings: [], counts: { projects: 1, incognitoSkipped: 1, conversations: 3, files: 9 } });
    expect(target.doneBody!.counts.bytes).toBeGreaterThan(0);
  });

  it('never includes an incognito project, and does not name one as the last project', async () => {
    const storage = populated();
    storage.settings = settings({ lastProjectId: 'p-secret', lastKeptProjectId: 'p-secret' });
    const target = new Collector();
    await exportAgentStorage(storage, target, { includeKeys: false });
    const names = Object.keys(target.entries);
    expect(names.filter((n) => n.includes('p-secret'))).toEqual([]);
    const all = names.map((n) => dec.decode(target.entries[n])).join('\n');
    expect(all).not.toContain('incognito words');
    expect(all).not.toContain('incognito file');
    const exported = JSON.parse(dec.decode(target.entries['idb/settings.json']));
    expect(exported.lastProjectId).toBeNull();
    expect(exported.lastKeptProjectId).toBeNull();
    expect(target.doneBody!.counts.incognitoSkipped).toBe(1);
  });

  it('leaves out the browser\u2019s temporary write files', async () => {
    const storage = populated().put('projects/p-kept/files/a.txt.crswap', 'half written');
    const target = new Collector();
    await exportAgentStorage(storage, target, { includeKeys: false });
    expect(Object.keys(target.entries).some((n) => n.endsWith('.crswap'))).toBe(false);
  });

  it('leaves out a file over the limit and says which, and keeps going', async () => {
    const storage = populated().put('projects/p-kept/files/huge.bin', noise(5000));
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false, limits: { file: 4000 } });
    expect(report.ok).toBe(true);
    expect(Object.keys(target.entries)).not.toContain('opfs/projects/p-kept/files/huge.bin');
    expect(Object.keys(target.entries)).toContain('opfs/projects/p-kept/chat.json');
    expect(target.doneBody!.warnings.join('\n')).toContain('projects/p-kept/files/huge.bin was left out');
    expect(report.skipped).toContain('projects/p-kept/files/huge.bin');
  });

  it('stops adding files once the total is reached, says so, and still sends the settings', async () => {
    const storage = new FakeStorage()
      .put('projects/a/project.json', '{"id":"a","name":"A"}')
      .put('projects/a/files/one.bin', noise(3000, 1))
      .put('projects/a/files/two.bin', noise(3000, 2))
      .put('projects/a/files/three.bin', noise(3000, 3))
      .put('projects/a/files/four.bin', noise(3000, 4));
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false, limits: { total: 5000 } });
    expect(report.ok).toBe(true);
    const names = Object.keys(target.entries);
    expect(names).toContain('idb/settings.json');
    expect(names.filter((n) => n.endsWith('.bin')).length).toBeLessThan(4);
    expect(target.doneBody!.warnings.some((w) => /bigger than \d+ MiB, so \d+ more files? w/.test(w))).toBe(true);
  });

  it('names a file that cannot be read and goes on', async () => {
    const storage = populated();
    storage.unreadable.add('projects/p-kept/files/notes.md');
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false });
    expect(report.ok).toBe(true);
    expect(Object.keys(target.entries)).not.toContain('opfs/projects/p-kept/files/notes.md');
    expect(Object.keys(target.entries)).toContain('opfs/projects/p-kept/chat.json');
    expect(report.warnings.join('\n')).toContain('projects/p-kept/files/notes.md could not be read');
  });

  it('sends parts of at most 4 MiB, numbered from 0 without a gap, that add up to the archive', async () => {
    const big = noise(9 * 1024 * 1024 + 123);
    const storage = populated().put('projects/p-kept/files/big.bin', big);
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false });
    expect(report.ok).toBe(true);
    expect(target.parts.length).toBeGreaterThanOrEqual(3);
    for (const p of target.parts) expect(p.length).toBeLessThanOrEqual(PART_BYTES);
    expect(target.seqs).toEqual(target.parts.map((_, i) => i));
    expect(target.doneBody!.parts).toBe(target.parts.length);
    expect(same(target.entries['opfs/projects/p-kept/files/big.bin'], big)).toBe(true);
  }, 60_000);

  it('stops sending when a part is refused, and reports the failure', async () => {
    const storage = populated().put('projects/p-kept/files/a.bin', noise(9000, 1)).put('projects/p-kept/files/b.bin', noise(9000, 2));
    const target = new Collector();
    target.failPartAt = 1;
    const report = await exportAgentStorage(storage, target, { includeKeys: false, limits: { part: 4000 } });
    expect(report.ok).toBe(false);
    expect(report.error).toContain('500');
    expect(target.attempts).toBe(2);
    expect(target.seqs).toEqual([0]);
    expect(target.doneBody).toMatchObject({ ok: false });
  });

  it('is not ok when the desktop cannot be told it is finished', async () => {
    const target = new Collector();
    target.failDone = true;
    const report = await exportAgentStorage(populated(), target, { includeKeys: false });
    expect(report.ok).toBe(false);
  });

  it('writes out what is pending first, and a failure to is a warning, not the end', async () => {
    const flush = vi.fn().mockResolvedValue(undefined);
    const first = await exportAgentStorage(populated(), new Collector(), { includeKeys: false, flush });
    expect(flush).toHaveBeenCalledTimes(1);
    expect(first.warnings).toEqual([]);
    const target = new Collector();
    const second = await exportAgentStorage(populated(), target, { includeKeys: false, flush: () => Promise.reject(new Error('busy')) });
    expect(second.ok).toBe(true);
    expect(second.warnings.join('\n')).toContain('latest changes');
    expect(target.doneBody!.ok).toBe(true);
  });

  it('carries no API key unless asked, and never the paired desktop\u2019s token or the sealed key', async () => {
    for (const includeKeys of [false, true]) {
      const storage = populated();
      // A provider, a second one, and the media service: all with keys.
      storage.settings = settings({ providers: [{ id: 'p1', type: 'openai', name: 'Main', apiKey: KEY }, { id: 'p2', type: 'anthropic', name: 'Other', apiKey: 'sk-second-key' }] });
      const target = new Collector();
      await exportAgentStorage(storage, target, { includeKeys });
      const all = Object.values(target.entries).map((e) => dec.decode(e)).join('\n');
      const exported = JSON.parse(dec.decode(target.entries['idb/settings.json']));
      expect(all).not.toContain(DESKTOP_TOKEN);
      expect(all).not.toContain('secret-key');
      expect(exported).not.toHaveProperty('desktop');
      expect(JSON.parse(dec.decode(target.entries['agent-manifest.json'])).includesKeys).toBe(includeKeys);
      if (includeKeys) {
        expect(exported.providers.map((p: { apiKey: string }) => p.apiKey)).toEqual([KEY, 'sk-second-key']);
        expect(exported.media.apiKey).toBe(MEDIA_KEY);
      } else {
        expect(all).not.toContain(KEY);
        expect(all).not.toContain('sk-second-key');
        expect(all).not.toContain(MEDIA_KEY);
        expect(exported.providers.map((p: { apiKey: string }) => p.apiKey)).toEqual(['', '']);
        expect(exported.media.apiKey).toBe('');
        // The rest of a provider is kept.
        expect(exported.providers[0]).toMatchObject({ id: 'p1', name: 'Main' });
      }
    }
  });
});

describe('answering the desktop\u2019s request for a backup', () => {
  it('posts the parts and the report to the job\u2019s address with both credentials', async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const fetchStub: FetchLike = async (url, init) => {
      calls.push({ url, init: init! });
      return json({ ok: true });
    };
    const w: BackupWindow = {};
    installBackupHooks({ desktop: DESKTOP, storage: populated(), fetch: fetchStub }, w);
    await w.__oaiyBackup!.export({ id: 'job-1', token: 'tok_2', includeKeys: false });
    const parts = calls.filter((c) => c.url.includes('/part?seq='));
    expect(parts.map((c) => c.url)).toEqual(parts.map((_, i) => `http://127.0.0.1:17972/api/backup/agent/job-1/part?seq=${i}`));
    const done = calls[calls.length - 1];
    expect(done.url).toBe('http://127.0.0.1:17972/api/backup/agent/job-1/done');
    expect(JSON.parse(done.init.body as string)).toMatchObject({ ok: true, parts: parts.length });
    for (const c of calls) {
      expect(c.init.method).toBe('POST');
      expect(c.init.headers).toMatchObject({ Authorization: 'Bearer bearer-token', 'X-Backup-Token': 'tok_2' });
    }
  });

  it('ignores a request that is not a well-formed job, and does not run two at once', async () => {
    const fetchStub = vi.fn<FetchLike>(async () => json({ ok: true }));
    const w: BackupWindow = {};
    installBackupHooks({ desktop: DESKTOP, storage: populated(), fetch: fetchStub }, w);
    await w.__oaiyBackup!.export(null);
    await w.__oaiyBackup!.export({ id: '../x', token: 't', includeKeys: false });
    await w.__oaiyBackup!.export({ id: 'ok', token: 'bad token', includeKeys: false });
    await w.__oaiyBackup!.export({ id: 'ok', token: 't', includeKeys: 'yes' });
    expect(fetchStub).not.toHaveBeenCalled();
    const a = w.__oaiyBackup!.export({ id: 'one', token: 't', includeKeys: false });
    const b = w.__oaiyBackup!.export({ id: 'two', token: 't', includeKeys: false });
    await Promise.all([a, b]);
    expect(fetchStub.mock.calls.every(([url]) => url.includes('/agent/one/'))).toBe(true);
  });

  it('never rejects into the desktop, whatever the network does, and can be taken away', async () => {
    const w: BackupWindow = {};
    const stop = installBackupHooks({ desktop: DESKTOP, storage: populated(), fetch: () => Promise.reject(new Error('down')) }, w);
    await expect(w.__oaiyBackup!.export({ id: 'job', token: 't', includeKeys: false })).resolves.toBeUndefined();
    stop();
    expect(w.__oaiyBackup).toBeUndefined();
  });
});

describe('what an archive entry may be', () => {
  it.each([
    ['agent-manifest.json', 'manifest'],
    ['idb/settings.json', 'settings'],
    ['opfs/projects/p1/chat.json', 'opfs'],
    ['opfs/projects/p1/files/a/b.txt', 'opfs'],
    ['opfs/front-desk/brief.md', 'opfs'],
  ])('%s is restored', (name, where) => {
    expect(checkEntry(name)).toMatchObject({ kind: 'file', where });
  });

  it('gives the path from the storage\u2019s root', () => {
    expect(checkEntry('opfs/projects/p1/chat.json')).toEqual({ kind: 'file', where: 'opfs', path: 'projects/p1/chat.json' });
    expect(checkEntry('opfs/front-desk/brief.md')).toEqual({ kind: 'file', where: 'opfs', path: 'front-desk/brief.md' });
  });

  it.each([
    '../evil.txt',
    'opfs/projects/p1/../../x',
    'opfs/projects/../../x.json',
    '/opfs/projects/p1/chat.json',
    'opfs\\projects\\p1\\chat.json',
    'C:/Windows/x',
    'c:evil',
    'opfs/projects//chat.json',
    'opfs/projects/./p1/chat.json',
    'opfs/projects/p1\u0000/chat.json',
    'opfs/projects/chat.json',
    'opfs/other/x.json',
    'idb/desktop.json',
    'secret-key',
    '',
  ])('%j is skipped', (name) => {
    expect(checkEntry(name).kind).toBe('skip');
  });

  it('a folder entry is not a file and not a warning', () => {
    expect(checkEntry('opfs/projects/p1/files/')).toEqual({ kind: 'dir' });
  });
});

describe('restoring into the Agent storage', () => {
  it('puts a backup made by one page into another, and reports what it did', async () => {
    const source = populated();
    const archive = await archiveOf(source);
    const target = new FakeStorage();
    target.settings = settings({ providers: [], activeProviderId: null, lastProjectId: null, lastKeptProjectId: null, media: { ...EMPTY_MEDIA } });
    const desk = fakeDesktop(archive);
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: true, applied: { projects: 1, settings: true }, warnings: [] });
    expect(target.text('projects/p-kept/chat.json')).toBe('[{"role":"user","text":"hello"}]');
    expect(target.text('projects/p-kept/files/notes.md')).toBe('# notes');
    expect(target.text('front-desk/brief.md')).toBe('# the brief');
    // The incognito project was never in the archive.
    expect([...target.files.keys()].some((p) => p.includes('p-secret'))).toBe(false);
    expect(target.settings.providers[0]).toMatchObject({ id: 'p1', apiKey: KEY });
    expect(target.settings.activeProviderId).toBe('p1');
    expect(desk.posted.done).toMatchObject({ ok: true, applied: { projects: 1, settings: true } });
    expect((desk.posted.done as { applied: { files: number } }).applied.files).toBe(outcome!.applied.files);
    // The token in the request is the session's, beside the bearer.
    expect(desk.headers.find((h) => h['X-Backup-Token'])).toMatchObject({ Authorization: 'Bearer bearer-token', 'X-Backup-Token': 'sessiontok1' });
  });

  it('refuses an archive that does not match its checksum: nothing is written, not even an undo copy', async () => {
    const archive = await archiveOf(populated());
    const events: string[] = [];
    const target = new FakeStorage(events);
    const desk = fakeDesktop(archive, { sha256: await sha(bytes('something else')), events });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: false });
    expect(outcome!.error).toContain('checksum');
    expect(events).toEqual(['done']);
    expect(target.files.size).toBe(0);
  });

  it('refuses an archive that comes short of the size it was said to have', async () => {
    const archive = await archiveOf(populated());
    const target = new FakeStorage();
    const desk = fakeDesktop(archive, { meta: { size: archive.length + 10, parts: Math.ceil((archive.length + 10) / 700) } });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(false);
    expect(target.files.size).toBe(0);
  });

  it('skips names that would leave the storage and restores the rest', async () => {
    const archive = craft({
      'opfs/projects/p1/chat.json': 'good',
      '../evil.txt': 'evil',
      'opfs/projects/p1/../../evil2.txt': 'evil',
      '/abs.txt': 'evil',
      'C:/win.txt': 'evil',
      'opfs\\projects\\p1\\back.txt': 'evil',
      'opfs/projects/p1/files/../../../x': 'evil',
      'opfs/elsewhere/x.txt': 'evil',
      'opfs/front-desk/brief.md': 'brief',
    });
    const target = new FakeStorage();
    const desk = fakeDesktop(archive);
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    expect([...target.files.keys()].sort()).toEqual(['front-desk/brief.md', 'projects/p1/chat.json']);
    expect(outcome!.warnings.length).toBe(7);
    expect(outcome!.warnings.join('\n')).toContain('was not restored');
  });

  it('refuses an archive of another kind or none, and one with too many entries', async () => {
    const wrong = zipSync({ 'agent-manifest.json': strToU8('{"v":1,"kind":"something-else"}'), 'opfs/projects/p/chat.json': strToU8('x') });
    const none = zipSync({ 'opfs/projects/p/chat.json': strToU8('x') });
    const newer = zipSync({ 'agent-manifest.json': strToU8('{"v":2,"kind":"oaiy-agent-storage"}') });
    for (const archive of [wrong, none, newer]) {
      const target = new FakeStorage();
      const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch });
      expect(outcome!.ok).toBe(false);
      expect(target.files.size).toBe(0);
    }
    const many = craft({ 'opfs/projects/p/a.txt': '1', 'opfs/projects/p/b.txt': '2', 'opfs/projects/p/c.txt': '3' });
    const target = new FakeStorage();
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(many).fetch, limits: { entries: 3 } });
    expect(outcome!.ok).toBe(false);
    expect(outcome!.error).toContain('more than 3 entries');
    expect(target.files.size).toBe(0);
  });

  it('leaves out an entry over the limit, and refuses an archive that unpacks to too much', async () => {
    const archive = craft({ 'opfs/projects/p/small.txt': 'ok', 'opfs/projects/p/big.txt': 'x'.repeat(500) });
    const target = new FakeStorage();
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch, limits: { entry: 100 } });
    expect(outcome!.ok).toBe(true);
    expect([...target.files.keys()]).toEqual(['projects/p/small.txt']);
    const bomb = craft({ 'opfs/projects/p/a.txt': 'y'.repeat(3000) });
    const refused = await applyPendingRestore(DESKTOP, new FakeStorage(), { fetch: fakeDesktop(bomb).fetch, limits: { unpacked: 1000 } });
    expect(refused!.ok).toBe(false);
  });

  it('takes the undo copy of what is there and finishes uploading it before the first write', async () => {
    const events: string[] = [];
    const archive = await archiveOf(populated());
    const target = new FakeStorage(events).put('projects/p-old/chat.json', '[]').put('projects/p-old/project.json', '{"id":"p-old"}');
    const desk = fakeDesktop(archive, { events });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    const firstWrite = events.findIndex((e) => e.startsWith('write '));
    expect(events.indexOf('undo-part')).toBeGreaterThanOrEqual(0);
    expect(events.indexOf('undo-part')).toBeLessThan(firstWrite);
    expect(events.indexOf('undo-done')).toBeLessThan(firstWrite);
    expect(events[events.length - 1]).toBe('done');
    // The copy holds what the page had, not what the restore brought, and never an API key: the desktop keeps
    // the copy as plain files, and the keys were sealed here with a key the browser does not let out.
    const copy = unzipSync(joined(desk.posted.undoParts));
    expect(Object.keys(copy)).toContain('opfs/projects/p-old/chat.json');
    expect(Object.keys(copy)).not.toContain('opfs/projects/p-kept/chat.json');
    expect(JSON.parse(dec.decode(copy['agent-manifest.json'])).includesKeys).toBe(false);
    const undoSettings = JSON.parse(dec.decode(copy['idb/settings.json']));
    expect(undoSettings.providers.map((p: { apiKey: string }) => p.apiKey)).toEqual(['']);
    expect(undoSettings.media.apiKey).toBe('');
    const everything = dec.decode(joined(desk.posted.undoParts.map((p) => p))) + JSON.stringify(undoSettings);
    expect(everything).not.toContain(KEY);
    expect(everything).not.toContain(MEDIA_KEY);
    expect(desk.posted.undoDone).toMatchObject({ ok: true });
    expect(desk.calls.some((c) => c.startsWith('POST /api/backup/agent-import/imp1/undo-part?seq=0'))).toBe(true);
  });

  it('restores nothing when the undo copy cannot be made', async () => {
    const events: string[] = [];
    const archive = await archiveOf(populated());
    const target = new FakeStorage(events).put('projects/p-old/chat.json', '[]');
    const desk = fakeDesktop(archive, { events, undoStatus: 500 });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: false, applied: { files: 0, projects: 0, settings: false } });
    expect(outcome!.error).toContain('undo copy');
    expect(events.filter((e) => e.startsWith('write'))).toEqual([]);
    expect([...target.files.keys()]).toEqual(['projects/p-old/chat.json']);
    expect(desk.posted.done).toMatchObject({ ok: false });
  });

  it('restores nothing when a file the restore would replace could not go into the undo copy', async () => {
    const archive = await archiveOf(populated());
    const target = new FakeStorage().put('projects/p-kept/chat.json', '[{"role":"user","text":"newer"}]');
    target.unreadable.add('projects/p-kept/chat.json');
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch });
    expect(outcome!.ok).toBe(false);
    expect(outcome!.error).toContain('undo copy');
    expect(target.text('projects/p-kept/chat.json')).toContain('newer');
  });

  it('takes no undo copy when it is itself an undo', async () => {
    const events: string[] = [];
    const archive = await archiveOf(populated());
    const target = new FakeStorage(events);
    const desk = fakeDesktop(archive, { kind: 'undo', events });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    expect(events).not.toContain('undo-part');
    expect(desk.calls.some((c) => c.includes('/undo-'))).toBe(false);
    expect(target.files.size).toBeGreaterThan(0);
  });

  it('overwrites the files the backup holds and leaves every other file alone', async () => {
    const archive = await archiveOf(populated());
    const target = new FakeStorage()
      .put('projects/p-kept/chat.json', 'newer than the backup')
      .put('projects/p-kept/files/only-here.txt', 'mine')
      .put('projects/p-other/chat.json', 'another project')
      .put('front-desk/extra.md', 'kept');
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch });
    expect(outcome!.ok).toBe(true);
    expect(target.text('projects/p-kept/chat.json')).toBe('[{"role":"user","text":"hello"}]');
    expect(target.text('projects/p-kept/files/only-here.txt')).toBe('mine');
    expect(target.text('projects/p-other/chat.json')).toBe('another project');
    expect(target.text('front-desk/extra.md')).toBe('kept');
  });

  it('never restores into an incognito project, whichever side says so', async () => {
    const archive = craft({
      'opfs/projects/inc-in-archive/project.json': JSON.stringify({ id: 'x', incognito: true }),
      'opfs/projects/inc-in-archive/chat.json': 'archive says incognito',
      'opfs/projects/inc-here/chat.json': 'this page says incognito',
      'opfs/projects/fine/project.json': JSON.stringify({ id: 'fine' }),
      'opfs/projects/fine/chat.json': 'fine',
    });
    const target = new FakeStorage().put('projects/inc-here/project.json', '{"id":"inc-here","incognito":true}').put('projects/inc-here/chat.json', 'mine');
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive, { kind: 'undo' }).fetch });
    expect(outcome!.ok).toBe(true);
    expect(target.text('projects/inc-here/chat.json')).toBe('mine');
    expect(target.files.has('projects/inc-in-archive/chat.json')).toBe(false);
    expect(target.text('projects/fine/chat.json')).toBe('fine');
    expect(outcome!.warnings.filter((w) => w.includes('incognito')).length).toBe(2);
  });

  it('does not let an empty key in the backup wipe a key that is kept, and leaves the paired desktop alone', async () => {
    const source = populated();
    const archive = await archiveOf(source, false);
    const target = new FakeStorage();
    target.settings = settings({ providers: [{ id: 'p1', type: 'openai', name: 'Old name', apiKey: 'sk-kept-here' }, { id: 'p9', type: 'local', name: 'Local', apiKey: 'sk-local' }], media: { ...EMPTY_MEDIA, apiKey: 'media-kept' } });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch });
    expect(outcome!.ok).toBe(true);
    const after = target.settings;
    expect(after.providers.find((p) => p.id === 'p1')).toMatchObject({ name: 'Main', apiKey: 'sk-kept-here' });
    expect(after.providers.find((p) => p.id === 'p9')).toMatchObject({ apiKey: 'sk-local' });
    expect(after.media).toMatchObject({ baseUrl: 'http://127.0.0.1:8080/v1', apiKey: 'media-kept' });
    expect(after.desktop).toEqual({ origin: 'http://127.0.0.1:17972', token: DESKTOP_TOKEN });
  });

  it('puts a key in the backup where none is kept now, and follows its active provider only if it exists', async () => {
    const archive = await archiveOf(populated(), true);
    const target = new FakeStorage();
    target.settings = settings({ providers: [], activeProviderId: null, media: { ...EMPTY_MEDIA } });
    await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive).fetch });
    expect(target.settings.providers[0].apiKey).toBe(KEY);
    expect(target.settings.media.apiKey).toBe(MEDIA_KEY);
    const merged = mergeSettings(settings(), { activeProviderId: 'no-such-provider', lastProjectId: 'p-secret' }, new Set(['p-secret']));
    expect(merged.activeProviderId).toBe('p1');
    expect(merged.lastProjectId).toBe('p-kept');
  });

  it('does nothing when no restore waits', async () => {
    const target = new FakeStorage();
    const desk = fakeDesktop(bytes('unused'), { pending: false });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toBeNull();
    expect(desk.calls).toEqual(['GET /api/backup/agent-import']);
    expect(target.files.size).toBe(0);
  });

  it('ignores a description of a restore it cannot trust', async () => {
    const archive = craft({ 'opfs/projects/p/chat.json': 'x' });
    for (const meta of [{ id: '../x' }, { kind: 'other' }, { token: 'a b' }, { sha256: 'abc' }, { size: -1 }, { parts: 99 }]) {
      const target = new FakeStorage();
      const desk = fakeDesktop(archive, { meta });
      const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
      // A description that is not well formed is not acted on; one that does not add up is reported and nothing is written.
      expect(target.files.size).toBe(0);
      expect(outcome === null || outcome.ok === false).toBe(true);
    }
  });

  it('does not throw when the desktop is not there: it tries again a few times, then does nothing', async () => {
    const target = new FakeStorage();
    const fetchStub = vi.fn<FetchLike>(() => Promise.reject(new Error('connection refused')));
    const sleep = vi.fn().mockResolvedValue(undefined);
    await expect(applyPendingRestore(DESKTOP, target, { fetch: fetchStub, retries: 2, sleep })).resolves.toBeNull();
    expect(fetchStub).toHaveBeenCalledTimes(3);
    expect(sleep).toHaveBeenCalledTimes(2);
    expect(target.files.size).toBe(0);
  });

  it('does not throw when the network fails part way through, and says so', async () => {
    const archive = await archiveOf(populated());
    const desk = fakeDesktop(archive);
    const target = new FakeStorage();
    const failing: FetchLike = (url, init) => (url.includes('/part/1') ? Promise.reject(new Error('reset')) : desk.fetch(url, init));
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: failing });
    expect(outcome).toMatchObject({ ok: false });
    expect(target.files.size).toBe(0);
    expect(desk.posted.done).toMatchObject({ ok: false });
  });

  it('reports files it could not write, and carries on with the others', async () => {
    const archive = await archiveOf(populated());
    const target = new FakeStorage();
    target.failWrites.add('projects/p-kept/chat.json');
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive, { kind: 'undo' }).fetch });
    expect(outcome!.ok).toBe(false);
    expect(outcome!.error).toContain('1 item');
    expect(outcome!.warnings.join('\n')).toContain('projects/p-kept/chat.json could not be written');
    expect(target.text('projects/p-kept/files/notes.md')).toBe('# notes');
  });
});
