import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, backup, inPlainSight, type BackupStatus, type RestorePreview, type StagedRestore } from './api';
import { isInvisible, visibleText } from './visibleText';

const TAGS = Array.from({ length: 25 }, (_, i) => String.fromCodePoint(0xe0041 + (i % 26))).join('');

const preview = (over: Partial<RestorePreview> = {}): RestorePreview => ({
  inspectId: 'i1',
  fileName: 'a.oaiybackup',
  createdAt: '2026-09-30T00:00:00Z',
  appVersion: '0.1.0',
  platform: 'windows',
  includesKeys: false,
  categories: [],
  lacks: [],
  partial: [],
  excluded: [],
  redo: [],
  totalFiles: 1,
  totalBytes: 10,
  classes: [],
  items: [],
  notRestored: [],
  keys: { inBackup: false },
  notes: [],
  ...over,
});

describe('the characters a person cannot see are made visible', () => {
  it('says each run of them as what it is and how many there are', () => {
    expect(visibleText(`Ask how the visit went${TAGS}`)).toBe('Ask how the visit went[25 invisible characters: U+E0041 U+E0042 U+E0043 U+E0044 U+E0045 U+E0046 …]');
    expect(visibleText('a​b‮c')).toBe('a[1 invisible character: U+200B]b[1 invisible character: U+202E]c');
    expect(visibleText('a​‍‌b')).toBe('a[3 invisible characters: U+200B U+200D U+200C]b');
  });

  it('leaves a text that has none as it is, and a line break, a tab and ordinary text with them', () => {
    for (const text of ['', 'Prefers texts', 'line one\nline two\tand a tab\r\n', 'café 你好 \u{1F468}', 'a b']) expect(visibleText(text)).toBe(text);
    expect(isInvisible(0x09) || isInvisible(0x0a) || isInvisible(0x0d)).toBe(false);
    expect(isInvisible(0x200b) && isInvisible(0xe0041) && isInvisible(0x202e) && isInvisible(0x2028) && isInvisible(0xfe0f)).toBe(true);
  });

  it('is the same on every text of a dry run that came from a backup, whoever sent it', () => {
    const seen = inPlainSight(
      preview({
        lacks: [`lack${TAGS}`],
        partial: ['partial​'],
        redo: ['redo‮'],
        notes: [`note${TAGS}`],
        items: [{ class: 'agentData', name: `agent/brief${TAGS}`, title: `The brief${TAGS}`, what: `Says: "Ask how the visit went${TAGS}"` }],
        notRestored: [{ name: `n${TAGS}`, why: `w${TAGS}` }],
        excluded: [{ pattern: `p${TAGS}`, reason: `r${TAGS}`, redo: `d${TAGS}` }],
      }),
    );
    const all = JSON.stringify(seen);
    expect([...all].some((ch) => isInvisible(ch.codePointAt(0) ?? 0) && ch !== '\n')).toBe(false);
    expect(seen.items[0].what).toContain('[25 invisible characters: U+E0041');
    expect(seen.items[0].title).toContain('The brief[25 invisible characters');
    expect(seen.excluded[0].redo).toContain('[25 invisible characters');
    expect(seen.notRestored[0].why).toContain('[25 invisible characters');
    expect(seen.redo[0]).toBe('redo[1 invisible character: U+202E]');
  });
});

describe('the same rule, at both ends of every range the desktop has', () => {
  // The ranges of `is_invisible` in backup/parts.rs, each with the code point just below and just above it (a test of the desktop holds
  // the two copies of the rule together for every code point).
  const invisible = [
    0x00, 0x08, 0x0b, 0x0c, 0x0e, 0x1f, 0x7f, 0x9f, 0xad, 0x034f, 0x061c, 0x115f, 0x1160, 0x17b4, 0x17b5, 0x180b, 0x180f, 0x200b, 0x200f, 0x2028, 0x202e, 0x2060, 0x206f, 0x3164,
    0xfe00, 0xfe0f, 0xfeff, 0xffa0, 0xfff0, 0xfff8, 0x1bca0, 0x1bca3, 0x1d173, 0x1d17a, 0xe0000, 0xe0fff,
  ];
  const drawn = [
    0x09, 0x0a, 0x0d, 0x20, 0x41, 0x7e, 0xa0, 0xac, 0xae, 0x034e, 0x0350, 0x061b, 0x061d, 0x115e, 0x1161, 0x17b3, 0x17b6, 0x180a, 0x1810, 0x200a, 0x2010, 0x2027, 0x202f, 0x205f, 0x2070,
    0x3163, 0x3165, 0xfdff, 0xfe10, 0xfefe, 0xff00, 0xffef, 0xfff9, 0x1bc9f, 0x1bca4, 0x1d172, 0x1d17b, 0xe1000, 0x1f600,
  ];

  it('makes each end of each range visible, and the code points beside them not', () => {
    expect(invisible.filter((c) => !isInvisible(c)).map((c) => c.toString(16))).toEqual([]);
    expect(drawn.filter((c) => isInvisible(c)).map((c) => c.toString(16))).toEqual([]);
  });

  it('names a code point with at least four digits, in capitals', () => {
    expect(visibleText('a\u0007b')).toBe('a[1 invisible character: U+0007]b');
    expect(visibleText('\u{1d173}')).toBe('[1 invisible character: U+1D173]');
    expect(visibleText('\ufeff')).toBe('[1 invisible character: U+FEFF]');
  });
});

describe('the calls that bring a dry run or a result to the screen say the characters a person cannot see', () => {
  const TAGS = String.fromCodePoint(0xe0041, 0xe0042, 0xe0043);
  const SAID = '[3 invisible characters: U+E0041 U+E0042 U+E0043]';

  beforeEach(() => {
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = undefined;
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  const invokes = (answer: unknown) => {
    const invoke = vi.fn().mockResolvedValue(answer);
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = { invoke };
    return invoke;
  };

  it('the dry run of a backup that is looked at', async () => {
    const found: RestorePreview = {
      inspectId: 'i1', fileName: 'a.oaiybackup', createdAt: '2026-09-30T00:00:00Z', appVersion: '0.1.0', platform: 'windows', includesKeys: false, categories: [], lacks: [], partial: [], excluded: [], redo: [],
      totalFiles: 1, totalBytes: 10, classes: [], items: [{ class: 'agentData', name: 'n', title: 't', what: `Says: "hi${TAGS}"` }], notRestored: [], keys: { inBackup: false }, notes: [],
    };
    invokes(found);
    const seen = await backup.inspectRestore('a passphrase');
    expect(seen?.items[0].what).toBe(`Says: "hi${SAID}"`);
    invokes(null);
    expect(await backup.inspectRestore('a passphrase')).toBeNull();
  });

  it('what a prepared restore, and an undo, say to do again and say they left out', async () => {
    const made: StagedRestore = { id: 's1', kind: 'restore', files: 1, bytes: 1, agentStorage: false, redo: [`redo${TAGS}`], skipped: [`skipped${TAGS}`] };
    invokes(made);
    const staged = await backup.stageRestore('i1', 'a passphrase', { classes: [], keys: false });
    expect(staged.redo).toEqual([`redo${SAID}`]);
    expect(staged.skipped).toEqual([`skipped${SAID}`]);
    const undone = await backup.undo();
    expect(undone.redo).toEqual([`redo${SAID}`]);
    expect(undone.skipped).toEqual([`skipped${SAID}`]);
  });

  it('how the last restore went', async () => {
    const status: BackupStatus = {
      lastBackupAt: null, lastBackupOk: null, lastBackupSize: null, pendingRestore: null, undoAvailable: false, undoKind: null,
      lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T00:00:00Z', ok: false, error: `error${TAGS}`, redo: [`redo${TAGS}`], notes: [`note${TAGS}`], agentStorage: 'none' },
    } as BackupStatus;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(status), { status: 200, headers: { 'Content-Type': 'application/json' } })));
    const seen = await backup.status();
    expect(seen.lastRestore?.error).toBe(`error${SAID}`);
    expect(seen.lastRestore?.redo).toEqual([`redo${SAID}`]);
    expect(seen.lastRestore?.notes).toEqual([`note${SAID}`]);
    expect((fetch as unknown as { mock: { calls: string[][] } }).mock.calls[0][0]).toBe(`${API_BASE}/api/backup/status`);
    // No restore yet: the status is as it is.
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ ...status, lastRestore: null }), { status: 200, headers: { 'Content-Type': 'application/json' } })));
    expect((await backup.status()).lastRestore).toBeNull();
  });
});
