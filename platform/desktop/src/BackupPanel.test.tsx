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

import { BackupSection } from './BackupPanel';
import type { BackupCreateResult, BackupStatus, RestorePreview } from './api';

const PASS = 'correct horse battery';
const PASS2 = 'another long passphrase';

const status = (over: Partial<BackupStatus> = {}): BackupStatus => ({
  lastBackupAt: null,
  lastBackupOk: null,
  lastBackupSize: null,
  pendingRestore: null,
  lastRestore: null,
  undoAvailable: false,
  running: null,
  ...over,
});

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
  ...over,
});

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
  h.stage.mockResolvedValue({ id: 'r1', kind: 'restore', files: 5, bytes: 100, agentStorage: false, redo: [] });
  h.undo.mockResolvedValue({ id: 'u1', kind: 'undo', files: 5, bytes: 100, agentStorage: false, redo: [] });
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
      status({ pendingRestore: { id: 'r1', kind: 'restore', stagedAt: '2026-09-30T00:00:00Z', files: 5, agentStorage: false } }),
    );
    await mount();
    await prepare();
    expect(h.stage).toHaveBeenCalledWith('insp-1', PASS);
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
    h.status.mockResolvedValue(status({ pendingRestore: { id: 'r1', kind: 'restore', stagedAt: '2026-09-30T00:00:00Z', files: 2, agentStorage: true } }));
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
    h.status.mockResolvedValueOnce(status({ pendingRestore: { id: 'r1', kind: 'restore', stagedAt: '2026-09-30T00:00:00Z', files: 2, agentStorage: false } })).mockResolvedValue(status());
    await mount();
    expect(text()).toContain('A restore is ready');
    await click(buttonWith('Cancel restore'));
    expect(h.discard).toHaveBeenCalledTimes(1);
    expect(text()).not.toContain('A restore is ready');
  });

  it('says undo, not restore, for a staged undo', async () => {
    h.status.mockResolvedValue(status({ pendingRestore: { id: 'u1', kind: 'undo', stagedAt: '2026-09-30T00:00:00Z', files: 3, agentStorage: false } }));
    await mount();
    expect(text()).toContain('An undo is ready (3 files). Restart OAIY to finish undoing the last restore.');
    expect(buttonWith('Restart to finish the undo')).toBeDefined();
    expect(buttonWith('Cancel undo')).toBeDefined();
  });
});

describe('after the restart', () => {
  it('reports a restore that finished, what to do again and the Agent’s part', async () => {
    h.status.mockResolvedValue(
      status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: true, redo: ['Link FormLogic again'], agentStorage: 'pending' } }),
    );
    await mount();
    expect(text()).toContain('Restored on [2026-09-30T01:00:00Z].');
    expect(text()).toContain('You still need to:');
    expect(text()).toContain('Link FormLogic again');
    expect(text()).toContain('restored when the Agent opens');
  });

  it('reports a restore that did not finish, as an error, with its reason', async () => {
    h.status.mockResolvedValue(
      status({ lastRestore: { id: 'r1', kind: 'restore', at: '2026-09-30T01:00:00Z', ok: false, error: 'contacts: access denied', redo: [], agentStorage: 'none' } }),
    );
    await mount();
    const banner = host.querySelector('.banner-err');
    expect(banner?.textContent).toContain('The restore did not finish and your files were put back: contacts: access denied');
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
      status({ undoAvailable: true, pendingRestore: { id: 'u1', kind: 'undo', stagedAt: '2026-09-30T00:00:00Z', files: 5, agentStorage: false } }),
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
