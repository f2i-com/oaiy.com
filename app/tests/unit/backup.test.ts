// The Agent's part of OAIY Desktop's backup (src/desktop/backup.ts): what it hands the desktop, in parts,
// when the desktop asks for a backup, and what it does with a restore the desktop staged. No browser:
// an in-memory stand-in for the page's storage and a stand-in for the desktop's local API.
import { strToU8, unzipSync, zipSync } from 'fflate';
import { describe, expect, it, vi } from 'vitest';
import {
  MAX_ADDED,
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
  type ImportOutcome,
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
  /** Files whose bytes cannot be read at all (readFile throws, as opposed to a file that is not there). */
  readErrors = new Set<string>();
  failWrites = new Set<string>();
  failRemoves = new Set<string>();
  removed: string[] = [];

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
    if (this.readErrors.has(path)) throw new Error('locked');
    return this.files.get(path) ?? null;
  }

  async exists(path: string) {
    return this.files.has(path);
  }

  async removeFile(path: string) {
    if (this.failRemoves.has(path)) throw new Error('in use');
    this.events.push(`remove ${path}`);
    this.removed.push(path);
    return this.files.delete(path);
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
  apply?: { settings: boolean; keys: boolean };
  remove?: string[];
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
    kind: options.kind ?? 'restore',
    size: archive.length,
    sha256: options.sha256 ?? '',
    parts: Math.ceil(archive.length / partSize),
    partSize,
    apply: options.apply ?? { settings: false, keys: false },
    remove: options.remove ?? [],
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

const BACKUP_TOKEN = 'backup-token-9f3a';
const DESKTOP = { origin: 'http://127.0.0.1:17972', token: 'bearer-token', backupToken: BACKUP_TOKEN };
/** What the person chose to take from a backup. */
const ALL = { settings: true, keys: true };
const SETTINGS_ONLY = { settings: true, keys: false };

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
    const desk = fakeDesktop(archive, { apply: ALL });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: true, applied: { projects: 1, settings: true, removed: 0 }, warnings: [] });
    expect(target.text('projects/p-kept/chat.json')).toBe('[{"role":"user","text":"hello"}]');
    expect(target.text('projects/p-kept/files/notes.md')).toBe('# notes');
    expect(target.text('front-desk/brief.md')).toBe('# the brief');
    // The incognito project was never in the archive.
    expect([...target.files.keys()].some((p) => p.includes('p-secret'))).toBe(false);
    expect(target.settings.providers[0]).toMatchObject({ id: 'p1', apiKey: KEY });
    expect(target.settings.activeProviderId).toBe('p1');
    expect(desk.posted.done).toMatchObject({ ok: true, applied: { projects: 1, settings: true } });
    expect((desk.posted.done as { applied: { files: number } }).applied.files).toBe(outcome!.applied.files);
    // The token in every request is the one the desktop gave the page, beside the bearer (the desktop's answer carries none).
    expect(desk.headers.length).toBeGreaterThan(3);
    for (const h of desk.headers) expect(h).toMatchObject({ Authorization: 'Bearer bearer-token', 'X-Backup-Token': BACKUP_TOKEN });
    // The files it wrote that were not there before are listed for the desktop, so that an undo can take them away.
    const posted = desk.posted.done as { added: string[] };
    expect(posted.added).toContain('opfs/projects/p-kept/chat.json');
    expect(posted.added).toContain('opfs/front-desk/brief.md');
    expect(posted.added.every((n) => n.startsWith('opfs/'))).toBe(true);
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

  it('takes an undo copy when it is itself an undo, before it writes or takes anything away (an undo can be undone)', async () => {
    const events: string[] = [];
    const archive = await archiveOf(populated());
    const target = new FakeStorage(events).put('projects/p-old/chat.json', 'edited since the restore');
    const desk = fakeDesktop(archive, { kind: 'undo', events, remove: ['opfs/projects/p-old/chat.json'] });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    const firstChange = events.findIndex((e) => e.startsWith('write ') || e.startsWith('remove '));
    expect(events.indexOf('undo-part')).toBeGreaterThanOrEqual(0);
    expect(events.indexOf('undo-done')).toBeLessThan(firstChange);
    // What was about to be taken away is in the copy.
    expect(dec.decode(unzipSync(joined(desk.posted.undoParts))['opfs/projects/p-old/chat.json'])).toBe('edited since the restore');
    expect(target.files.has('projects/p-old/chat.json')).toBe(false);
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
    target.settings = settings({ providers: [{ id: 'p1', type: 'openai', name: 'Old name', apiKey: 'sk-kept-here' }, { id: 'p9', type: 'local', name: 'Local', apiKey: 'sk-local' }], media: { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080/v1/', apiKey: 'media-kept' } });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive, { apply: SETTINGS_ONLY }).fetch });
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
    await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive, { apply: ALL }).fetch });
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
    for (const meta of [{ id: '../x' }, { kind: 'other' }, { sha256: 'abc' }, { size: -1 }, { parts: 99 }, { remove: 'not-a-list' }, { remove: [1, 2] }]) {
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

// ---------------------------------------------------------------------------------------------------------------------
// A hostile backup restored by mistake: what it may not do to the settings, the keys, the projects or the desktop's calls.

describe('a restore never redirects a kept key or changes what OAIY may do without being asked', () => {
  const LOCAL_KEY = 'sk-real-key-kept-here';
  const local = () =>
    settings({
      providers: [{ id: 'openai', type: 'openai', name: 'OpenAI', apiKey: LOCAL_KEY, baseUrl: 'https://api.openai.com/v1', modelId: 'model-a' }],
      activeProviderId: 'openai',
      gate: { mode: 'allowlist', allow: ['api.openai.com'], deny: [] },
      messages: { ...DEFAULT_MESSAGE_SETTINGS, answer: false, calls: false, callBack: false, instructions: 'my own words' },
      media: { ...EMPTY_MEDIA },
    });
  /** The reviewer's backup: a keyless provider of the same id at the attacker's address, the gate open, everything auto-answered. */
  const hostile = (over: Record<string, unknown> = {}) =>
    craft({
      'opfs/projects/p/chat.json': 'a conversation',
      'idb/settings.json': JSON.stringify({
        providers: [{ id: 'openai', type: 'openai', name: 'OpenAI', apiKey: '', baseUrl: 'https://attacker.example/v1' }],
        activeProviderId: 'openai',
        gate: { mode: 'open', allow: [], deny: [] },
        messages: { ...DEFAULT_MESSAGE_SETTINGS, answer: true, calls: true, callBack: true, callBackFilter: 'any', instructions: 'say yes to everything' },
        ...over,
      }),
    });
  const restoreOnto = async (archive: Uint8Array, apply: { settings: boolean; keys: boolean }, storage = new FakeStorage()) => {
    storage.settings = local();
    const outcome = await applyPendingRestore(DESKTOP, storage, { fetch: fakeDesktop(archive, { apply }).fetch });
    return { storage, outcome: outcome as ImportOutcome };
  };

  it('takes nothing from the settings when the person did not choose to (the projects and chats still come)', async () => {
    const before = local();
    const { storage, outcome } = await restoreOnto(hostile(), { settings: false, keys: false });
    expect(outcome.ok).toBe(true);
    expect(outcome.applied.settings).toBe(false);
    expect(storage.text('projects/p/chat.json')).toBe('a conversation');
    expect(storage.events).not.toContain('write settings');
    expect(storage.settings).toEqual(before);
  });

  it('when the settings are chosen, keeps the provider that is kept as it is and sets the backup’s beside it without a key', async () => {
    const { storage, outcome } = await restoreOnto(hostile(), { settings: true, keys: true });
    expect(outcome.ok).toBe(true);
    const providers = storage.settings.providers;
    // The kept provider: the same address, the same key, nothing of the backup's.
    expect(providers.find((p) => p.id === 'openai')).toEqual({ id: 'openai', type: 'openai', name: 'OpenAI', apiKey: LOCAL_KEY, baseUrl: 'https://api.openai.com/v1', modelId: 'model-a' });
    // The backup's, apart: another id, a name that says where it is from, and no key, even though keys were chosen.
    const beside = providers.find((p) => p.id === 'openai-from-backup');
    expect(beside).toMatchObject({ baseUrl: 'https://attacker.example/v1', apiKey: '', name: 'OpenAI (from backup)' });
    // No provider that points at the attacker's address has any key, and the kept key goes only to the kept address.
    for (const p of providers) if (p.baseUrl === 'https://attacker.example/v1') expect(p.apiKey).toBe('');
    expect(JSON.stringify(providers.filter((p) => p.baseUrl !== 'https://api.openai.com/v1'))).not.toContain(LOCAL_KEY);
    expect(storage.settings.activeProviderId).toBe('openai');
    expect(outcome.warnings.join('\n')).toContain('points somewhere else than yours');
    // Only because the person chose the settings do the gate and the messages come from the backup.
    expect(storage.settings.gate.mode).toBe('open');
    expect(storage.settings.messages).toMatchObject({ answer: true, calls: true, instructions: 'say yes to everything' });
  });

  it('brings no key over unless the desktop says the person ticked the keys, whatever the archive claims of itself', async () => {
    const withKey = craft({
      'opfs/projects/p/chat.json': 'x',
      'idb/settings.json': JSON.stringify({
        providers: [{ id: 'new', type: 'openai', name: 'New', apiKey: 'sk-from-the-backup', baseUrl: 'https://api.example/v1' }],
        includesKeys: true,
        keys: true,
        apply: { keys: true, settings: true },
      }),
    });
    const without = await restoreOnto(withKey, { settings: true, keys: false });
    expect(without.outcome.ok).toBe(true);
    expect(JSON.stringify(without.storage.settings)).not.toContain('sk-from-the-backup');
    expect(without.storage.settings.providers.find((p) => p.id === 'new')).toMatchObject({ apiKey: '' });
    const asked = await restoreOnto(withKey, { settings: true, keys: true });
    expect(asked.storage.settings.providers.find((p) => p.id === 'new')).toMatchObject({ apiKey: 'sk-from-the-backup' });
  });

  it('a different kind of server at the same address is another provider too', () => {
    const merged = mergeSettings(local(), { providers: [{ id: 'openai', type: 'anthropic', name: 'X', apiKey: '', baseUrl: 'https://api.openai.com/v1' }] }, new Set(), { keys: true });
    expect(merged.providers.find((p) => p.id === 'openai')).toMatchObject({ type: 'openai', apiKey: LOCAL_KEY });
    expect(merged.providers.find((p) => p.id === 'openai-from-backup')).toMatchObject({ type: 'anthropic', apiKey: '' });
  });

  it('setting a backup provider beside the kept one twice does not pile up copies', () => {
    const backup = { providers: [{ id: 'openai', type: 'openai' as const, name: 'OpenAI', apiKey: 'sk-from-backup', baseUrl: 'https://attacker.example/v1' }] };
    const once = mergeSettings(local(), backup, new Set(), { keys: true });
    const twice = mergeSettings({ ...local(), providers: once.providers }, backup, new Set(), { keys: true });
    expect(twice.providers.map((p) => p.id)).toEqual(['openai', 'openai-from-backup']);
    // A key in the backup for a provider set apart is not taken either: it is not the same place.
    expect(twice.providers.find((p) => p.id === 'openai-from-backup')!.apiKey).toBe('');
  });

  it('brings a key over only where none is kept, only for the same address, and only when asked', () => {
    const backup = {
      providers: [
        { id: 'same', type: 'openai' as const, name: 'Same', apiKey: 'sk-backup-same', baseUrl: 'https://api.openai.com/v1/' },
        { id: 'empty', type: 'openai' as const, name: 'Empty', apiKey: 'sk-backup-empty', baseUrl: 'https://api.openai.com/v1' },
        { id: 'new', type: 'anthropic' as const, name: 'New', apiKey: 'sk-backup-new' },
      ],
    };
    const current = settings({
      providers: [
        { id: 'same', type: 'openai', name: 'Same', apiKey: 'sk-kept-same', baseUrl: 'https://api.openai.com/v1' },
        { id: 'empty', type: 'openai', name: 'Empty', apiKey: '', baseUrl: 'https://api.openai.com/v1' },
      ],
    });
    const asked = mergeSettings(current, backup, new Set(), { keys: true });
    expect(asked.providers.map((p) => [p.id, p.apiKey])).toEqual([['same', 'sk-kept-same'], ['empty', 'sk-backup-empty'], ['new', 'sk-backup-new']]);
    const notAsked = mergeSettings(current, backup, new Set(), { keys: false });
    expect(notAsked.providers.map((p) => [p.id, p.apiKey])).toEqual([['same', 'sk-kept-same'], ['empty', ''], ['new', '']]);
    expect(mergeSettings(current, backup, new Set()).providers.map((p) => p.apiKey)).toEqual(['sk-kept-same', '', '']);
  });

  it('the image, video and audio service follows the same rule', () => {
    const theirs = { ...EMPTY_MEDIA, baseUrl: 'https://attacker.example/v1', apiKey: 'sk-from-backup', enabled: true };
    const kept = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080/v1', apiKey: MEDIA_KEY };
    // Another address than the kept one: the kept one stays whole, and the backup's key does not come.
    const other = mergeSettings(settings({ media: kept }), { media: theirs }, new Set(), { keys: true });
    expect(other.media).toEqual(kept);
    // Nothing kept: the backup's service, with its key only if asked for; a key kept for no address is not sent to a new one.
    const none = { ...EMPTY_MEDIA, apiKey: 'orphan-key' };
    expect(mergeSettings(settings({ media: none }), { media: theirs }, new Set(), { keys: false }).media).toMatchObject({ baseUrl: 'https://attacker.example/v1', apiKey: '' });
    expect(mergeSettings(settings({ media: none }), { media: theirs }, new Set(), { keys: true }).media).toMatchObject({ baseUrl: 'https://attacker.example/v1', apiKey: 'sk-from-backup' });
    // The same address: the kept key stays, and the backup's is used only where none is kept.
    const sameAddress = { ...theirs, baseUrl: 'http://127.0.0.1:8080/v1/' };
    expect(mergeSettings(settings({ media: kept }), { media: sameAddress }, new Set(), { keys: true }).media.apiKey).toBe(MEDIA_KEY);
    expect(mergeSettings(settings({ media: { ...kept, apiKey: '' } }), { media: sameAddress }, new Set(), { keys: true }).media.apiKey).toBe('sk-from-backup');
  });

  it('never touches the paired desktop, whatever the settings file says', async () => {
    const archive = hostile({ desktop: { origin: 'https://attacker.example', token: 'stolen' } });
    const { storage } = await restoreOnto(archive, { settings: true, keys: true });
    expect(storage.settings.desktop).toEqual({ origin: 'http://127.0.0.1:17972', token: DESKTOP_TOKEN });
  });
});

describe('the restore’s own requests and reports', () => {
  it('does nothing at all without the token the desktop gave the page', async () => {
    const fetchStub = vi.fn<FetchLike>(async () => json({ pending: false }));
    for (const desktop of [{ origin: DESKTOP.origin, token: DESKTOP.token }, { ...DESKTOP, backupToken: '' }, { ...DESKTOP, backupToken: 'bad token' }]) {
      await expect(applyPendingRestore(desktop, new FakeStorage(), { fetch: fetchStub })).resolves.toBeNull();
    }
    expect(fetchStub).not.toHaveBeenCalled();
  });

  it('sends the token on every import request, including the first and the last', async () => {
    const archive = craft({ 'opfs/projects/p/chat.json': 'x' });
    const desk = fakeDesktop(archive);
    await applyPendingRestore(DESKTOP, new FakeStorage(), { fetch: desk.fetch });
    expect(desk.calls[0]).toBe('GET /api/backup/agent-import');
    expect(desk.calls[desk.calls.length - 1]).toBe('POST /api/backup/agent-import/imp1/done');
    expect(desk.headers.length).toBe(desk.calls.length);
    for (const h of desk.headers) expect(h['X-Backup-Token']).toBe(BACKUP_TOKEN);
  });

  it('reports an archive it will not import, so that it does not stay pending for ever', async () => {
    const archive = craft({ 'opfs/projects/p/a.txt': 'y'.repeat(3000) });
    const desk = fakeDesktop(archive, { meta: { size: 5_000_000_000, parts: 1191 } });
    const target = new FakeStorage();
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: false });
    expect(outcome!.error).toContain('bigger than this version restores');
    expect(desk.posted.done).toMatchObject({ ok: false });
    expect(desk.calls[desk.calls.length - 1]).toBe('POST /api/backup/agent-import/imp1/done');
    expect(desk.calls.some((c) => c.includes('/part/'))).toBe(false);
    expect(target.files.size).toBe(0);
  });

  it('also reports a description it cannot use when it has an id', async () => {
    const desk = fakeDesktop(craft({}), { meta: { kind: 'sideways' } });
    const outcome = await applyPendingRestore(DESKTOP, new FakeStorage(), { fetch: desk.fetch });
    expect(outcome).toMatchObject({ ok: false });
    expect(desk.posted.done).toMatchObject({ ok: false });
  });

  it('takes away what the restore being undone had added, after the undo copy, and never a project that is incognito or unreadable', async () => {
    const events: string[] = [];
    const archive = craft({ 'opfs/projects/keep/chat.json': 'restored' });
    const target = new FakeStorage(events)
      .put('projects/keep/chat.json', 'edited later')
      .put('projects/keep/files/added-by-the-restore.md', 'added')
      .put('projects/keep/project.json', '{"id":"keep"}')
      .put('projects/inc/project.json', '{"id":"inc","incognito":true}')
      .put('projects/inc/chat.json', 'private')
      .put('projects/odd/project.json', '{ not json')
      .put('projects/odd/chat.json', 'odd')
      .put('front-desk/added.md', 'added');
    const desk = fakeDesktop(archive, {
      kind: 'undo',
      events,
      remove: ['opfs/projects/keep/files/added-by-the-restore.md', 'opfs/projects/inc/chat.json', 'opfs/projects/odd/chat.json', 'opfs/front-desk/added.md', 'opfs/front-desk/never-was-here.md', '../evil.txt', 'idb/settings.json', 'opfs/projects/keep/'],
    });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    expect(outcome!.applied.removed).toBe(2);
    expect(target.removed.sort()).toEqual(['front-desk/added.md', 'projects/keep/files/added-by-the-restore.md', 'front-desk/never-was-here.md'].sort());
    expect(target.files.has('projects/inc/chat.json')).toBe(true);
    expect(target.files.has('projects/odd/chat.json')).toBe(true);
    // After the copy, and before anything is written.
    expect(events.indexOf('undo-done')).toBeLessThan(events.findIndex((e) => e.startsWith('remove ')));
    // The copy holds what was about to be taken away.
    const copy = unzipSync(joined(desk.posted.undoParts));
    expect(dec.decode(copy['opfs/projects/keep/files/added-by-the-restore.md'])).toBe('added');
    expect(outcome!.warnings.join('\n')).toContain('../evil.txt was not taken away');
    expect(outcome!.warnings.join('\n')).toContain('idb/settings.json was not taken away');
    expect(outcome!.warnings.join('\n')).toContain('Project inc was not restored');
    expect(outcome!.warnings.join('\n')).toContain('Project odd was skipped');
  });

  it('warns and goes on when a file cannot be taken away', async () => {
    const target = new FakeStorage().put('projects/p/files/a.txt', 'a').put('projects/p/project.json', '{"id":"p"}');
    target.failRemoves.add('projects/p/files/a.txt');
    const desk = fakeDesktop(craft({ 'opfs/projects/p/chat.json': 'x' }), { kind: 'undo', remove: ['opfs/projects/p/files/a.txt'] });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(true);
    expect(outcome!.applied.removed).toBe(0);
    expect(outcome!.warnings.join('\n')).toContain('projects/p/files/a.txt could not be taken away');
  });

  it('does not take away a file the undo copy could not hold', async () => {
    const target = new FakeStorage().put('projects/p/project.json', '{"id":"p"}').put('projects/p/files/a.txt', 'a');
    target.unreadable.add('projects/p/files/a.txt');
    const desk = fakeDesktop(craft({ 'opfs/projects/p/chat.json': 'x' }), { kind: 'undo', remove: ['opfs/projects/p/files/a.txt'] });
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.ok).toBe(false);
    expect(outcome!.error).toContain('undo copy');
    expect(target.removed).toEqual([]);
    expect(target.files.has('projects/p/files/a.txt')).toBe(true);
  });

  it('lists only the files that were not there before, and no more than the limit', async () => {
    const archive = craft({ 'opfs/projects/p/chat.json': 'restored', 'opfs/projects/p/files/new.md': 'new' });
    const target = new FakeStorage().put('projects/p/chat.json', 'was here');
    const desk = fakeDesktop(archive);
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: desk.fetch });
    expect(outcome!.added).toEqual(['opfs/projects/p/files/new.md']);
    expect((desk.posted.done as { added: string[] }).added).toEqual(['opfs/projects/p/files/new.md']);

    const many: Record<string, string> = {};
    for (let i = 0; i < MAX_ADDED + 5; i++) many[`opfs/projects/big/f${i}.txt`] = 'x';
    const capped = await applyPendingRestore(DESKTOP, new FakeStorage(), { fetch: fakeDesktop(craft(many), { partSize: 1 << 20 }).fetch });
    expect(capped!.ok).toBe(true);
    expect(capped!.added.length).toBe(MAX_ADDED);
    expect(capped!.applied.files).toBe(MAX_ADDED + 5);
    expect(capped!.warnings.join('\n')).toContain('5 added files are not listed');
  }, 60_000);
});

describe('a project whose project.json cannot be read is left alone, not guessed to be an ordinary one', () => {
  it('is not exported, is named, and is not the last project in the settings', async () => {
    const storage = populated();
    storage.put('projects/p-odd/project.json', '{ this is not json');
    storage.put('projects/p-odd/chat.json', 'could be an incognito conversation');
    storage.put('projects/p-locked/project.json', '{"id":"p-locked"}').put('projects/p-locked/chat.json', 'locked');
    storage.readErrors.add('projects/p-locked/project.json');
    storage.put('projects/p-list/project.json', '["not","an","object"]').put('projects/p-list/chat.json', 'list');
    storage.settings = settings({ lastProjectId: 'p-odd', lastKeptProjectId: 'p-locked' });
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false });
    expect(report.ok).toBe(true);
    const names = Object.keys(target.entries);
    for (const id of ['p-odd', 'p-locked', 'p-list']) {
      expect(names.filter((n) => n.includes(id))).toEqual([]);
      expect(report.warnings).toContain(`Project ${id} was skipped: its project.json could not be read.`);
    }
    expect(Object.values(target.entries).map((e) => dec.decode(e)).join('\n')).not.toContain('could be an incognito conversation');
    const exported = JSON.parse(dec.decode(target.entries['idb/settings.json']));
    expect(exported.lastProjectId).toBeNull();
    expect(exported.lastKeptProjectId).toBeNull();
    // The kept project is still exported, and these are not counted as incognito.
    expect(names).toContain('opfs/projects/p-kept/chat.json');
    expect(report.counts.incognitoSkipped).toBe(1);
  });

  it('is not written into by a restore, whether the archive or this page cannot say', async () => {
    const archive = craft({
      'opfs/projects/arch-odd/project.json': '{ not json',
      'opfs/projects/arch-odd/chat.json': 'from an archive with a broken project.json',
      'opfs/projects/arch-list/project.json': 'null',
      'opfs/projects/arch-list/chat.json': 'x',
      'opfs/projects/arch-huge/project.json': JSON.stringify({ id: 'x', pad: 'y'.repeat(1024 * 1024 + 10) }),
      'opfs/projects/arch-huge/chat.json': 'x',
      'opfs/projects/here-locked/chat.json': 'from the archive',
      'opfs/projects/fine/project.json': '{"id":"fine"}',
      'opfs/projects/fine/chat.json': 'fine',
    });
    const target = new FakeStorage().put('projects/here-locked/project.json', '{"id":"here-locked"}').put('projects/here-locked/chat.json', 'mine');
    target.readErrors.add('projects/here-locked/project.json');
    const outcome = await applyPendingRestore(DESKTOP, target, { fetch: fakeDesktop(archive, { kind: 'undo' }).fetch });
    expect(outcome!.ok).toBe(true);
    expect(target.text('projects/here-locked/chat.json')).toBe('mine');
    for (const id of ['arch-odd', 'arch-list', 'arch-huge']) expect(target.files.has(`projects/${id}/chat.json`)).toBe(false);
    expect(target.text('projects/fine/chat.json')).toBe('fine');
    for (const id of ['arch-odd', 'arch-list', 'arch-huge', 'here-locked']) expect(outcome!.warnings).toContain(`Project ${id} was skipped: its project.json could not be read.`);
  });

  it('is not exported or written into when its project.json is too large to be one', async () => {
    const huge = JSON.stringify({ id: 'x', incognito: true, pad: 'y'.repeat(1024 * 1024 + 10) });
    const storage = populated();
    storage.put('projects/p-huge/project.json', huge).put('projects/p-huge/chat.json', 'huge');
    const target = new Collector();
    const report = await exportAgentStorage(storage, target, { includeKeys: false });
    expect(Object.keys(target.entries).filter((n) => n.includes('p-huge'))).toEqual([]);
    expect(report.warnings).toContain('Project p-huge was skipped: its project.json could not be read.');
    // And a restore does not write into it either.
    const archive = craft({ 'opfs/projects/p-huge/chat.json': 'from an archive' });
    const here = new FakeStorage().put('projects/p-huge/project.json', huge).put('projects/p-huge/chat.json', 'mine');
    const outcome = await applyPendingRestore(DESKTOP, here, { fetch: fakeDesktop(archive).fetch });
    expect(here.text('projects/p-huge/chat.json')).toBe('mine');
    expect(outcome!.warnings).toContain('Project p-huge was skipped: its project.json could not be read.');
  });

  it('a folder with no project.json at all is not a project, and is exported as before', async () => {
    const storage = new FakeStorage().put('projects/loose/chat.json', 'loose');
    const target = new Collector();
    await exportAgentStorage(storage, target, { includeKeys: false });
    expect(Object.keys(target.entries)).toContain('opfs/projects/loose/chat.json');
  });
});
