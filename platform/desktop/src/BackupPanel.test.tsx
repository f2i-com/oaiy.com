// Backup and restore, in Settings. What matters here is what a person could get wrong or lose:
// a passphrase that is too short or mistyped must not start a backup, the passphrase must not
// outlive the command that used it (not in the page, not in storage, not in the console), a
// backup that left something out must say so, and a restore must never look finished before the
// restart that applies it.
//
// Same convention as the other panels' tests: raw react-dom/client + act, mocks by vi.hoisted.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const h = vi.hoisted(() => ({
  status: vi.fn(),
  create: vi.fn(),
  inspect: vi.fn(),
  stage: vi.fn(),
  undo: vi.fn(),
  discard: vi.fn(),
  restart: vi.fn(),
  open: vi.fn(),
}));

vi.mock('./api', () => ({
  backup: {
    status: (...a: unknown[]) => h.status(...a),
    create: (...a: unknown[]) => h.create(...a),
    inspectRestore: (...a: unknown[]) => h.inspect(...a),
    stageRestore: (...a: unknown[]) => h.stage(...a),
    undo: (...a: unknown[]) => h.undo(...a),
    discardPending: (...a: unknown[]) => h.discard(...a),
    restartToApply: (...a: unknown[]) => h.restart(...a),
  },
  formatBytes: (n: number) => `${n} bytes`,
  formatTimestamp: (iso: string) => `[${iso}]`,
  openInExplorer: (...a: unknown[]) => h.open(...a),
}));

import { BackupSection, RESTORE_TIMEOUT_MS, agoWords, undoConfirmText } from './BackupPanel';
import type { BackupCreateResult, BackupStatus, RestoreClass, RestorePreview, ReviewItem, StagedRestore } from './api';

const PASS = 'correct horse battery';
const PASS2 = 'another long passphrase';

const status = (over: Partial<BackupStatus> = {}): BackupStatus => ({
  lastBackupAt: null,
  lastBackupOk: null,
  lastBackupSize: null,
  pendingRestore: null,
  lastRestore: null,
  undoAvailable: false,
  undoKind: null,
  running: null,
  ...over,
});

/** A prepared restore, as the status reports it. */
const pendingOf = (over: Partial<NonNullable<BackupStatus['pendingRestore']>> = {}): NonNullable<BackupStatus['pendingRestore']> => ({
  id: 'r1',
  kind: 'restore',
  stagedAt: '2026-09-30T00:00:00Z',
  expiresAt: '2026-10-01T00:00:00Z',
  expired: false,
  files: 5,
  agentStorage: false,
  classes: [],
  ...over,
});

const staged = (over: Partial<StagedRestore> = {}): StagedRestore => ({ id: 'r1', kind: 'restore', files: 5, bytes: 100, agentStorage: false, redo: [], skipped: [], ...over });

const created = (over: Partial<BackupCreateResult> = {}): BackupCreateResult => ({
  path: 'C:\\Backups\\oaiy-2026-09-30.oaiybackup',
  fileName: 'oaiy-2026-09-30.oaiybackup',
  size: 4096,
  createdAt: '2026-09-30T02:00:00Z',
  counts: { files: 12, bytes: 9000, agentProjects: 2, agentConversations: 5, agentFiles: 30 },
  includesKeys: false,
  partial: [],
  excluded: [],
  verified: true,
  ...over,
});

const preview = (over: Partial<RestorePreview> = {}): RestorePreview => ({
  inspectId: 'insp-1',
  fileName: 'old.oaiybackup',
  createdAt: '2026-08-01T00:00:00Z',
  appVersion: '0.1.0',
  platform: 'windows',
  includesKeys: false,
  categories: [
    { id: 'contacts', label: 'Contacts', added: 3, replaced: 1, unchanged: 0, leftAlone: 2 },
    { id: 'calendar', label: 'Calendar', added: 0, replaced: 4, unchanged: 5, leftAlone: 0 },
  ],
  lacks: ['Agent conversations and projects'],
  partial: ['Agent conversations and projects were not included: open the Agent and try again'],
  excluded: [{ pattern: 'link/account.json', reason: 'the FormLogic link credential', redo: 'Link FormLogic again' }],
  redo: ['Link FormLogic again', 'Pair your phone again'],
  totalFiles: 14,
  totalBytes: 2048,
  classes: [],
  items: [],
  keys: { inBackup: false },
  notes: [],
  ...over,
});

const CLASSES: RestoreClass[] = [
  { id: 'templates', label: 'Service templates', description: 'Each one can run a program on this computer.', count: 2 },
  { id: 'flows', label: 'Flows and triggers', description: 'They can send messages and call your AI providers.', count: 1 },
  { id: 'providers', label: 'AI providers', description: 'Where OAIY sends your prompts.', count: 1 },
];
const ITEMS: ReviewItem[] = [
  { class: 'templates', name: 'templates/evil.json', title: 'Evil helper', what: 'Runs: cmd.exe /c calc.exe' },
  { class: 'templates', name: 'templates/oaiy-voice.json', title: 'OAIY Voice', what: 'Runs: oaiy-voice --port 8090' },
  { class: 'flows', name: 'flows/greeting.json', title: 'Greeting', what: 'A flow with 4 steps' },
  { class: 'providers', name: 'ai/providers.json#openai', title: 'openai', what: 'https://attacker.example/v1, has a key' },
];
/** A backup that holds things that can run. */
const risky = (over: Partial<RestorePreview> = {}): RestorePreview => preview({ classes: CLASSES, items: ITEMS, keys: { inBackup: true }, ...over });

let host: HTMLDivElement;
let root: Root;

const text = () => host.textContent ?? '';
const buttonWith = (label: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label)) as HTMLButtonElement | undefined;
const buttonExact = (label: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.trim() === label) as HTMLButtonElement | undefined;
const input = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
const checkbox = () => host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

function setValue(el: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  setter.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

const flush = () =>
  act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });

async function mount() {
  await act(async () => {
    root.render(<BackupSection />);
  });
  await flush();
}

async function typeBoth(pass: string, again = pass) {
  await act(async () => setValue(input('Backup passphrase'), pass));
  await act(async () => setValue(input('Backup passphrase again'), again));
}

async function click(el: HTMLElement | undefined) {
  expect(el).toBeDefined();
  await act(async () => {
    el!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await flush();
}

/** Everything a passphrase could leak into. */
const spies = () => ({
  log: vi.spyOn(console, 'log').mockImplementation(() => undefined),
  info: vi.spyOn(console, 'info').mockImplementation(() => undefined),
  warn: vi.spyOn(console, 'warn').mockImplementation(() => undefined),
  error: vi.spyOn(console, 'error').mockImplementation(() => undefined),
  debug: vi.spyOn(console, 'debug').mockImplementation(() => undefined),
});
const seen = (s: ReturnType<typeof spies>, secret: string) =>
  Object.values(s).some((spy) => spy.mock.calls.some((call) => JSON.stringify(call).includes(secret)));

beforeEach(() => {
  vi.clearAllMocks();
  h.status.mockResolvedValue(status());
  h.create.mockResolvedValue(created());
  h.inspect.mockResolvedValue(preview());
  h.stage.mockResolvedValue(staged());
  h.undo.mockResolvedValue(staged({ id: 'u1', kind: 'undo' }));
  h.discard.mockResolvedValue(undefined);
  h.restart.mockResolvedValue(undefined);
  h.open.mockResolvedValue(undefined);
  localStorage.clear();
  sessionStorage.clear();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  vi.useRealTimers();
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  // The setup file's default: the person confirmed.
  vi.stubGlobal('confirm', () => true);
});

describe('the words a person reads first', () => {
  it('says plainly that a lost passphrase is a lost backup, and what is never included', async () => {
    await mount();
    expect(text()).toContain('If you lose the passphrase, the backup is lost');
    expect(text()).toContain('Nobody, including OAIY, can open it');
    expect(text()).toContain('FormLogic');
    expect(text()).toContain('Hugging Face');
    expect(text()).toContain('are never in a backup');
    expect(text()).toContain('with PINs, keys and sealed values left out');
  });
});

describe('making a backup: the passphrase rules', () => {
  it('will not start with 11 characters, and says why', async () => {
    await mount();
    await typeBoth('elevenchars');
    expect(buttonWith('Create backup')!.disabled).toBe(true);
    expect(text()).toContain('Use at least 12 characters');
    await click(buttonWith('Create backup'));
    expect(h.create).not.toHaveBeenCalled();
  });

  it('will not start when the two passphrases differ, and says so', async () => {
    await mount();
    await typeBoth(PASS, PASS2);
    expect(buttonWith('Create backup')!.disabled).toBe(true);
    expect(text()).toContain('The two passphrases are not the same');
    await click(buttonWith('Create backup'));
    expect(h.create).not.toHaveBeenCalled();
  });

  it('starts with 12 characters typed twice', async () => {
    await mount();
    await typeBoth('twelve chars');
    expect(buttonWith('Create backup')!.disabled).toBe(false);
    expect(text()).not.toContain('Use at least 12 characters');
    expect(text()).not.toContain('not the same');
  });

  it('counts characters as a person does, not UTF-16 units: 11 characters with two emoji are still too short', async () => {
    await mount();
    await typeBoth('ninechars😀😀');
    expect(buttonWith('Create backup')!.disabled).toBe(true);
    await typeBoth('elevenchars😀');
    expect(buttonWith('Create backup')!.disabled).toBe(false);
  });
});

describe('making a backup', () => {
  it('passes the passphrase and, by default, no keys', async () => {
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(h.create).toHaveBeenCalledWith(PASS, false);
  });

  it('passes includeKeys when the box is ticked, with the warning that it is as sensitive as the keys', async () => {
    await mount();
    await typeBoth(PASS);
    expect(text()).toContain('Include my API provider keys');
    await act(async () => checkbox().click());
    expect(text()).toContain('as sensitive as the keys themselves');
    expect(text()).toContain('When you restore, they come back only if you tick them again');
    expect(text()).toContain('the keys OAIY’s own AI gateway holds');
    await click(buttonWith('Create backup'));
    expect(h.create).toHaveBeenCalledWith(PASS, true);
  });

  it('clears the passphrase and the tick as soon as it is done, and never shows, stores or logs it', async () => {
    const s = spies();
    await mount();
    await typeBoth(PASS);
    await act(async () => checkbox().click());
    await click(buttonWith('Create backup'));
    expect(input('Backup passphrase').value).toBe('');
    expect(input('Backup passphrase again').value).toBe('');
    expect(checkbox().checked).toBe(false);
    expect(host.innerHTML).not.toContain(PASS);
    expect(text()).not.toContain(PASS);
    expect(JSON.stringify({ ...localStorage })).not.toContain(PASS);
    expect(JSON.stringify({ ...sessionStorage })).not.toContain(PASS);
    expect(seen(s, PASS)).toBe(false);
  });

  it('clears it when the backup fails, too', async () => {
    h.create.mockRejectedValue('Something failed');
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(input('Backup passphrase').value).toBe('');
    expect(host.innerHTML).not.toContain(PASS);
  });

  it('shows the file, its size, what it holds and that it was checked', async () => {
    h.create.mockResolvedValue(created({ includesKeys: true }));
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(text()).toContain('Backup created');
    expect(text()).toContain('C:\\Backups\\oaiy-2026-09-30.oaiybackup');
    expect(text()).toContain('4096 bytes');
    expect(text()).toContain('12 files');
    expect(text()).toContain('2 projects');
    expect(text()).toContain('5 conversations');
    expect(text()).toContain('Verified: the file was decrypted and every item checked after writing.');
    expect(text()).toContain('includes your API provider keys');
    await click(buttonWith('Show'));
    expect(h.open).toHaveBeenCalledWith('C:\\Backups\\oaiy-2026-09-30.oaiybackup');
  });

  it('lists what could not be included, in the desktop’s own words', async () => {
    const warning = 'Agent conversations and projects were not included: open the Agent and try again';
    h.create.mockResolvedValue(created({ partial: [warning, 'One file could not be read'] }));
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(text()).toContain('Not everything was included');
    expect(text()).toContain(warning);
    expect(text()).toContain('One file could not be read');
  });

  it('shows nothing when the save dialog was closed, and lets the person try again', async () => {
    h.create.mockResolvedValue(null);
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(text()).not.toContain('Backup created');
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(input('Backup passphrase').disabled).toBe(false);
    await typeBoth(PASS);
    expect(buttonWith('Create backup')!.disabled).toBe(false);
  });

  it('shows the desktop’s refusal while it is busy', async () => {
    h.create.mockRejectedValue('OAIY is busy: a call is in progress. Try again when it has finished.');
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(host.querySelector('.banner-err')?.textContent).toContain('OAIY is busy: a call is in progress');
  });

  it('shows progress while it runs, from the desktop’s status, and disables the fields', async () => {
    let finish!: (r: BackupCreateResult) => void;
    h.create.mockReturnValue(new Promise<BackupCreateResult>((r) => (finish = r)));
    h.status.mockResolvedValue(status({ running: { phase: 'encrypting', label: 'Encrypting the backup' } }));
    await mount();
    await typeBoth(PASS);
    await click(buttonWith('Create backup'));
    expect(host.querySelector('[role="progressbar"]')).not.toBeNull();
    expect(text()).toContain('Encrypting the backup');
    expect(input('Backup passphrase').disabled).toBe(true);
    h.status.mockResolvedValue(status());
    await act(async () => finish(created()));
    await flush();
    expect(host.querySelector('[role="progressbar"]')).toBeNull();
    expect(text()).toContain('Backup created');
  });
});

describe('restoring: the check', () => {
  it('shows what would happen, per kind of data, what the backup lacks and what to do again', async () => {
    await mount();
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    expect(h.inspect).toHaveBeenCalledWith(PASS);
    expect(text()).toContain('old.oaiybackup');
    expect(text()).toContain('OAIY 0.1.0');
    expect(text()).toContain('Contacts');
    expect(text()).toContain('3 added · 1 replaced · 0 unchanged · 2 left alone');
    expect(text()).toContain('Calendar');
    expect(text()).toContain('0 added · 4 replaced · 5 unchanged · 0 left alone');
    expect(text()).toContain('This backup does not have: Agent conversations and projects.');
    expect(text()).toContain('Warnings from when it was made');
    expect(text()).toContain('link/account.json: the FormLogic link credential');
    expect(text()).toContain('After restoring you will need to:');
    expect(text()).toContain('Pair your phone again');
    expect(text()).toContain('Only restore a backup you made yourself');
    expect(h.stage).not.toHaveBeenCalled();
  });

  it('shows nothing and keeps nothing when the open dialog was closed', async () => {
    h.inspect.mockResolvedValue(null);
    await mount();
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    expect(host.querySelector('[aria-label="What restoring would do"]')).toBeNull();
    expect(input('Passphrase of the backup to restore').value).toBe('');
  });

  it('shows a wrong passphrase plainly and clears it', async () => {
    h.inspect.mockRejectedValue('That passphrase did not open the backup, or the file is damaged.');
    await mount();
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    expect(host.querySelector('.banner-err')?.textContent).toContain('did not open the backup');
    expect(input('Passphrase of the backup to restore').value).toBe('');
    expect(host.innerHTML).not.toContain(PASS);
  });

  it('needs a passphrase before it will look', async () => {
    await mount();
    expect(buttonWith('Choose backup file and check it')!.disabled).toBe(true);
  });

  it('cancel drops the summary and the passphrase', async () => {
    await mount();
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    expect(text()).toContain('old.oaiybackup');
    await click(buttonExact('Cancel'));
    expect(text()).not.toContain('old.oaiybackup');
    expect(input('Passphrase of the backup to restore').value).toBe('');
    expect(h.stage).not.toHaveBeenCalled();
  });
});

describe('restoring: the restart', () => {
  async function prepare() {
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    await click(buttonWith('Prepare restore'));
  }

  it('stages with the same passphrase, clears it, and then asks for the restart', async () => {
    const s = spies();
    h.status.mockResolvedValueOnce(status()).mockResolvedValue(
      status({ pendingRestore: pendingOf() }),
    );
    await mount();
    await prepare();
    expect(h.stage).toHaveBeenCalledWith('insp-1', PASS, { classes: [], keys: false });
    expect(input('Passphrase of the backup to restore').value).toBe('');
    expect(host.innerHTML).not.toContain(PASS);
    expect(JSON.stringify({ ...localStorage })).not.toContain(PASS);
    expect(seen(s, PASS)).toBe(false);
    expect(text()).toContain('A restore is ready (5 files). Restart OAIY to finish restoring.');
    expect(buttonWith('Restart to finish restoring')).toBeDefined();
    expect(buttonWith('Cancel restore')).toBeDefined();
    // The changes have not been applied: the summary is gone and nothing says "restored".
    expect(text()).not.toContain('Restored on');
  });

  it('shows the pending banner even before the status answers again', async () => {
    h.status.mockResolvedValue(status());
    await mount();
    await prepare();
    expect(text()).toContain('A restore is ready (5 files)');
  });

  it('shows why a restart was refused, and keeps the button', async () => {
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ files: 2, agentStorage: true }) }));
    h.restart.mockRejectedValue('OAIY is busy: a call is in progress. Try again when it has finished.');
    await mount();
    expect(text()).toContain('A restore is ready (2 files)');
    expect(text()).toContain('conversations and projects are put back when the Agent opens');
    await click(buttonWith('Restart to finish restoring'));
    expect(h.restart).toHaveBeenCalledTimes(1);
    expect(text()).toContain('OAIY is busy: a call is in progress');
    expect(buttonWith('Restart to finish restoring')).toBeDefined();
    expect(buttonWith('Restart to finish restoring')!.disabled).toBe(false);
  });

  it('cancels a restore that waits', async () => {
    h.status.mockResolvedValueOnce(status({ pendingRestore: pendingOf({ files: 2 }) })).mockResolvedValue(status());
    await mount();
    expect(text()).toContain('A restore is ready');
    await click(buttonWith('Cancel restore'));
    expect(h.discard).toHaveBeenCalledTimes(1);
    expect(text()).not.toContain('A restore is ready');
  });

  it('says undo, not restore, for a staged undo', async () => {
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ id: 'u1', kind: 'undo', files: 3 }) }));
    await mount();
    expect(text()).toContain('An undo is ready (3 files). Restart OAIY to finish undoing the last restore.');
    expect(buttonWith('Restart to finish the undo')).toBeDefined();
    expect(buttonWith('Cancel undo')).toBeDefined();
  });
});

describe('after the restart', () => {
  it('reports a restore that finished, what to do again and the Agent’s part', async () => {
    h.status.mockResolvedValue(
      status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: true, redo: ['Link FormLogic again'], notes: [], agentStorage: 'pending' } }),
    );
    await mount();
    expect(text()).toContain('Restored on [2026-09-30T01:00:00Z].');
    expect(text()).toContain('You still need to:');
    expect(text()).toContain('Link FormLogic again');
    expect(text()).toContain('restored when the Agent opens');
  });

  it('reports a restore that did not finish, as an error, with its reason', async () => {
    h.status.mockResolvedValue(
      status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: false, error: 'contacts: access denied', redo: [], notes: [], agentStorage: 'none' } }),
    );
    await mount();
    const banner = host.querySelector('.banner-err');
    expect(banner?.textContent).toContain('The restore did not finish');
    expect(banner?.textContent).toContain('contacts: access denied');
    // The reason is the desktop's own: the panel does not claim the files were put back.
    expect(banner?.textContent).not.toContain('were put back');
    expect(text()).not.toContain('Restored on');
  });
});

describe('undo', () => {
  const withUndo = () => h.status.mockResolvedValue(status({ undoAvailable: true }));

  it('asks first, and does nothing when the person says no', async () => {
    withUndo();
    vi.stubGlobal('confirm', () => false);
    await mount();
    await click(buttonWith('Undo the last restore'));
    expect(h.undo).not.toHaveBeenCalled();
  });

  it('stages the undo once confirmed, then waits for the restart like a restore', async () => {
    h.status.mockResolvedValueOnce(status({ undoAvailable: true })).mockResolvedValue(
      status({ undoAvailable: true, undoKind: 'restore', pendingRestore: pendingOf({ id: 'u1', kind: 'undo' }) }),
    );
    vi.stubGlobal('confirm', () => true);
    await mount();
    await click(buttonWith('Undo the last restore'));
    expect(h.undo).toHaveBeenCalledTimes(1);
    expect(text()).toContain('An undo is ready (5 files)');
    expect(buttonWith('Undo the last restore')).toBeUndefined();
  });

  it('is not offered when there is nothing to undo', async () => {
    await mount();
    expect(buttonWith('Undo the last restore')).toBeUndefined();
  });

  it('shows why an undo could not be prepared', async () => {
    withUndo();
    vi.stubGlobal('confirm', () => true);
    h.undo.mockRejectedValue('There is no restore to undo.');
    await mount();
    await click(buttonWith('Undo the last restore'));
    expect(host.querySelector('.banner-err')?.textContent).toContain('There is no restore to undo.');
  });
});

// ---------------------------------------------------------------------------------------------
// A hostile backup restored by mistake: what can run or reconfigure things is listed by name and
// brought back only when it is ticked.
// ---------------------------------------------------------------------------------------------

describe('restoring: what can run or change settings needs a tick', () => {
  const classBox = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="Bring back: ${label}"]`)!;
  const keysBox = () => host.querySelector<HTMLInputElement>('input[aria-label="Bring back the API keys that are in this backup"]');

  async function check(over: Partial<RestorePreview> = {}) {
    h.inspect.mockResolvedValue(risky(over));
    await mount();
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
  }

  it('lists every kind and every item by name, with what it does, before anything is ticked', async () => {
    await check();
    const group = host.querySelector('[aria-label="What can run or change settings"]')!;
    expect(group).not.toBeNull();
    expect(group.textContent).toContain('can run programs, send messages, point OAIY at other servers');
    expect(group.textContent).toContain('tick only what you recognise as yours');
    for (const c of CLASSES) {
      expect(group.textContent).toContain(c.label);
      expect(group.textContent).toContain(c.description);
    }
    for (const i of ITEMS) {
      expect(group.textContent).toContain(i.name);
      expect(group.textContent).toContain(i.title);
      expect(group.textContent).toContain(i.what);
    }
    // The reviewer's case: the template that would run cmd.exe is shown with its command.
    expect(group.textContent).toContain('templates/evil.json');
    expect(group.textContent).toContain('cmd.exe /c calc.exe');
    expect(group.textContent).toContain('https://attacker.example/v1');
  });

  it('shows all of a long list, never a silent part of it', async () => {
    const many: ReviewItem[] = Array.from({ length: 75 }, (_, n) => ({ class: 'flows', name: `flows/flow-${n}.json`, title: `Flow ${n}`, what: 'A flow' }));
    await check({ classes: [{ id: 'flows', label: 'Flows and triggers', description: 'They can send messages.', count: 75 }], items: many });
    for (let n = 0; n < 75; n++) expect(text()).toContain(`flows/flow-${n}.json`);
    expect(text()).toContain('The 75 items');
  });

  it('starts with nothing ticked, and says only the data comes back', async () => {
    await check();
    for (const c of CLASSES) expect(classBox(c.label).checked).toBe(false);
    expect(keysBox()!.checked).toBe(false);
    expect(text()).toContain('Nothing is ticked, so only your data comes back: contacts, calendar, conversations, voices and history.');
    await click(buttonWith('Prepare restore'));
    expect(h.stage).toHaveBeenCalledWith('insp-1', PASS, { classes: [], keys: false });
  });

  it('skips a kind the backup holds none of', async () => {
    await check({ classes: [...CLASSES, { id: 'connections', label: 'Connections', description: 'Where OAIY is linked.', count: 0 }] });
    expect(host.querySelector('input[aria-label="Bring back: Connections"]')).toBeNull();
  });

  it('ticks a kind on its own click, and only that one', async () => {
    await check();
    await act(async () => classBox('Flows and triggers').click());
    expect(classBox('Flows and triggers').checked).toBe(true);
    expect(classBox('Service templates').checked).toBe(false);
    expect(classBox('AI providers').checked).toBe(false);
    expect(text()).not.toContain('Nothing is ticked');
    await act(async () => classBox('Flows and triggers').click());
    expect(classBox('Flows and triggers').checked).toBe(false);
  });

  it('“Select all of my own backup” is an explicit click that ticks every kind but not the keys', async () => {
    await check();
    await click(buttonWith('Select all of my own backup'));
    for (const c of CLASSES) expect(classBox(c.label).checked).toBe(true);
    expect(keysBox()!.checked).toBe(false);
    await click(buttonWith('Prepare restore'));
    expect(h.stage).toHaveBeenCalledWith('insp-1', PASS, { classes: ['templates', 'flows', 'providers'], keys: false });
  });

  it('offers the keys only when the file has some, unticked, with the words that they come back only if ticked', async () => {
    await check({ keys: { inBackup: false } });
    expect(keysBox()).toBeNull();
    act(() => root.unmount());
    root = createRoot(host);
    await check({ keys: { inBackup: true } });
    expect(keysBox()).not.toBeNull();
    expect(keysBox()!.checked).toBe(false);
    expect(text()).toContain('The keys come back only when this is ticked, and only for the provider lists you tick above');
  });

  it('carries exactly the ticked kinds, in the desktop’s order, and the keys flag, to the desktop', async () => {
    await check();
    await act(async () => classBox('AI providers').click());
    await act(async () => classBox('Service templates').click());
    await act(async () => keysBox()!.click());
    await click(buttonWith('Prepare restore'));
    expect(h.stage).toHaveBeenCalledTimes(1);
    expect(h.stage).toHaveBeenCalledWith('insp-1', PASS, { classes: ['templates', 'providers'], keys: true });
  });

  it('brings no keys when the keys box is left alone, whatever else is ticked', async () => {
    await check();
    await click(buttonWith('Select all of my own backup'));
    await click(buttonWith('Prepare restore'));
    expect(h.stage.mock.calls[0][2].keys).toBe(false);
  });

  it('forgets the ticks when the panel is cancelled or another file is checked', async () => {
    await check();
    await act(async () => classBox('Flows and triggers').click());
    await act(async () => keysBox()!.click());
    await click(buttonExact('Cancel'));
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
    await click(buttonWith('Choose backup file and check it'));
    for (const c of CLASSES) expect(classBox(c.label).checked).toBe(false);
    expect(keysBox()!.checked).toBe(false);
  });

  it('clears the passphrase after preparing, and shows what the desktop left out', async () => {
    h.stage.mockResolvedValue(staged({ skipped: ['Autostart entries without a template were skipped: evil', 'Flows were not ticked, so they were left out.'] }));
    h.status.mockResolvedValue(status());
    await check();
    await act(async () => classBox('Service templates').click());
    await click(buttonWith('Prepare restore'));
    expect(input('Passphrase of the backup to restore').value).toBe('');
    expect(host.innerHTML).not.toContain(PASS);
    expect(text()).toContain('You ticked: Service templates and what starts with OAIY.');
    expect(text()).toContain('Autostart entries without a template were skipped: evil');
    expect(text()).toContain('Flows were not ticked, so they were left out.');
  });

  it('says plainly that only data comes back when nothing was ticked', async () => {
    h.status.mockResolvedValue(status());
    await check();
    await click(buttonWith('Prepare restore'));
    expect(text()).toContain('Only your data is brought back: nothing that can run or change settings was ticked.');
  });
});

describe('a prepared restore', () => {
  it('says how long ago it was prepared and when it is thrown away', async () => {
    const stagedAt = new Date(Date.now() - 3 * 3_600_000).toISOString();
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ stagedAt, expiresAt: '2026-10-01T00:00:00Z', classes: ['flows', 'providers'] }) }));
    await mount();
    expect(text()).toContain('Prepared 3 hours ago; it is discarded, not applied, at the next start after [2026-10-01T00:00:00Z].');
    expect(text()).toContain('You ticked: Flows, triggers and runs, AI providers.');
  });

  it('says it will not be applied when it has waited more than a day, and does not offer the restart', async () => {
    const stagedAt = new Date(Date.now() - 30 * 3_600_000).toISOString();
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ stagedAt, expired: true }) }));
    await mount();
    expect(text()).toContain('that is more than a day, so it will be discarded, not applied, at the next start.');
    expect(text()).not.toContain('it is discarded, not applied, at the next start after');
    expect(buttonWith('Restart to finish restoring')!.disabled).toBe(true);
    expect(buttonWith('Cancel restore')!.disabled).toBe(false);
  });

  it('offers the restart while it is still good', async () => {
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ stagedAt: new Date(Date.now() - 3_600_000).toISOString() }) }));
    await mount();
    expect(buttonWith('Restart to finish restoring')!.disabled).toBe(false);
  });

  it('does not list ticks for an undo', async () => {
    h.status.mockResolvedValue(status({ pendingRestore: pendingOf({ kind: 'undo', id: 'u1' }) }));
    await mount();
    expect(text()).toContain('An undo is ready');
    expect(text()).not.toContain('You ticked');
    expect(text()).not.toContain('Only your data is brought back');
  });

  it('words how long ago as a person says it', () => {
    const now = Date.parse('2026-09-30T12:00:00Z');
    expect(agoWords('2026-09-30T11:59:40Z', now)).toBe('just now');
    expect(agoWords('2026-09-30T11:59:00Z', now)).toBe('1 minute ago');
    expect(agoWords('2026-09-30T11:15:00Z', now)).toBe('45 minutes ago');
    expect(agoWords('2026-09-30T11:00:00Z', now)).toBe('1 hour ago');
    expect(agoWords('2026-09-30T09:00:00Z', now)).toBe('3 hours ago');
    expect(agoWords('2026-09-27T12:00:00Z', now)).toBe('3 days ago');
    expect(agoWords('not a date', now)).toBe('a while ago');
  });
});

describe('what the last restore did, said plainly', () => {
  it('shows a restore that was thrown away because it waited too long, in the desktop’s words', async () => {
    const error = 'The restore prepared on 2026-09-28 was not applied because it waited more than 24 hours. Prepare it again.';
    h.status.mockResolvedValue(status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: false, error, redo: [], notes: [], agentStorage: 'none' } }));
    await mount();
    const banner = host.querySelector('.banner-err')!;
    expect(banner.textContent).toContain(error);
    expect(banner.textContent).not.toContain('were put back');
  });

  it('shows the notes of a restore that finished, and of one that did not', async () => {
    h.status.mockResolvedValue(
      status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: true, redo: [], notes: ['Autostart entries without a template were skipped: evil'], agentStorage: 'none' } }),
    );
    await mount();
    expect(text()).toContain('Left out or changed:');
    expect(text()).toContain('Autostart entries without a template were skipped: evil');
    act(() => root.unmount());
    root = createRoot(host);
    h.status.mockResolvedValue(
      status({
        lastRestore: { id: 'r2', kind: 'undo', at: '2026-09-30T02:00:00Z', ok: false, error: 'These files could not be put back: callers.json. Their originals are in restore/undo-1.', redo: [], notes: ['Kept for you to look at'], agentStorage: 'none' },
      }),
    );
    await mount();
    expect(host.querySelector('.banner-err')!.textContent).toContain('These files could not be put back: callers.json');
    expect(host.querySelector('.banner-err')!.textContent).toContain('Kept for you to look at');
  });
});

describe('undo and redo say what they really do', () => {
  it('asks first with the plain words: it removes what the restore added, and what was changed since goes too', async () => {
    h.status.mockResolvedValue(status({ undoAvailable: true, undoKind: 'restore' }));
    const asked: string[] = [];
    vi.stubGlobal('confirm', (m: string) => {
      asked.push(m);
      return false;
    });
    await mount();
    await click(buttonWith('Undo the last restore'));
    expect(asked).toHaveLength(1);
    expect(asked[0]).toBe(undoConfirmText('restore'));
    expect(asked[0]).toContain('puts back the files the last restore replaced and REMOVES the files it added');
    expect(asked[0]).toContain('including anything you changed or added in them since');
    expect(asked[0]).toContain('saved first');
    expect(asked[0]).toContain('Redo');
    expect(h.undo).not.toHaveBeenCalled();
  });

  it('is a redo after an undo: another label, and words that fit', async () => {
    h.status.mockResolvedValue(status({ undoAvailable: true, undoKind: 'undo' }));
    const asked: string[] = [];
    vi.stubGlobal('confirm', (m: string) => {
      asked.push(m);
      return true;
    });
    await mount();
    expect(buttonWith('Undo the last restore')).toBeUndefined();
    await click(buttonWith('Redo: put back what the last undo took away'));
    expect(asked[0]).toBe(undoConfirmText('undo'));
    expect(asked[0]).toContain('puts back what the last undo took away');
    expect(asked[0]).toContain('including anything you changed in them since');
    expect(h.undo).toHaveBeenCalledTimes(1);
  });
});

describe('the panel is never left stuck', () => {
  async function typeRestorePass() {
    await act(async () => setValue(input('Passphrase of the backup to restore'), PASS));
  }

  it('stops waiting for a check that never answers, says so, and can be used again', async () => {
    h.inspect.mockReturnValue(new Promise(() => undefined));
    await mount();
    await typeRestorePass();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await act(async () => {
      buttonWith('Choose backup file and check it')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(input('Passphrase of the backup to restore').disabled).toBe(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RESTORE_TIMEOUT_MS + 1000);
    });
    vi.useRealTimers();
    expect(text()).toContain('That took too long: the check was stopped, try again');
    expect(input('Passphrase of the backup to restore').disabled).toBe(false);
    expect(input('Passphrase of the backup to restore').value).toBe('');
    await typeRestorePass();
    expect(buttonWith('Choose backup file and check it')!.disabled).toBe(false);
  });

  it('stops waiting for a restore that is never prepared, and drops the summary so it can be started again', async () => {
    h.inspect.mockResolvedValue(risky());
    h.stage.mockReturnValue(new Promise(() => undefined));
    await mount();
    await typeRestorePass();
    await click(buttonWith('Choose backup file and check it'));
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await act(async () => {
      buttonWith('Prepare restore')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(buttonWith('Prepare restore')!.disabled).toBe(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RESTORE_TIMEOUT_MS + 1000);
    });
    vi.useRealTimers();
    expect(text()).toContain('That took too long: the restore was not prepared, try again');
    expect(host.querySelector('[aria-label="What restoring would do"]')).toBeNull();
    expect(input('Passphrase of the backup to restore').value).toBe('');
    expect(host.innerHTML).not.toContain(PASS);
    await typeRestorePass();
    expect(buttonWith('Choose backup file and check it')!.disabled).toBe(false);
  });

  it('a check that fails leaves the panel ready for another, with the passphrase gone', async () => {
    h.inspect.mockRejectedValue('The file is damaged.');
    await mount();
    await typeRestorePass();
    await click(buttonWith('Choose backup file and check it'));
    expect(text()).toContain('The file is damaged.');
    expect(input('Passphrase of the backup to restore').disabled).toBe(false);
    expect(input('Passphrase of the backup to restore').value).toBe('');
    h.inspect.mockResolvedValue(preview());
    await typeRestorePass();
    await click(buttonWith('Choose backup file and check it'));
    expect(text()).toContain('old.oaiybackup');
  });

  it('a restore that fails to prepare shows why and drops the summary', async () => {
    h.inspect.mockResolvedValue(risky());
    h.stage.mockRejectedValue('There is not enough free space to prepare this restore.');
    await mount();
    await typeRestorePass();
    await click(buttonWith('Choose backup file and check it'));
    await click(buttonWith('Prepare restore'));
    expect(host.querySelector('.banner-err')!.textContent).toContain('not enough free space');
    expect(host.querySelector('[aria-label="What restoring would do"]')).toBeNull();
    expect(input('Passphrase of the backup to restore').value).toBe('');
  });
});
