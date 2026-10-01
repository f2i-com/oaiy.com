// Settings holds two features that arrived separately and must live on one page: "About and updates"
// (a newer OAIY, and installing it) and "Backup and restore" (one encrypted file of the setup). This
// checks they are both there, once each, in that order, that each asks only for its own status, and
// that a control of one does not reach the other. Same convention as the other panels' tests: raw
// react-dom/client + act, mocks by vi.hoisted.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BackupStatus, UpdateStatus } from './api';

const h = vi.hoisted(() => ({
  updateStatus: vi.fn(),
  updateCheck: vi.fn(),
  updateDownload: vi.fn(),
  updateInstall: vi.fn(),
  setAutoCheck: vi.fn(),
  backupStatus: vi.fn(),
  backupCreate: vi.fn(),
  backupInspect: vi.fn(),
  backupUndo: vi.fn(),
  appConfigGet: vi.fn(),
}));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    isTauri: () => true,
    updates: {
      status: (...a: unknown[]) => h.updateStatus(...a),
      check: (...a: unknown[]) => h.updateCheck(...a),
      download: (...a: unknown[]) => h.updateDownload(...a),
      install: (...a: unknown[]) => h.updateInstall(...a),
      setAutoCheck: (...a: unknown[]) => h.setAutoCheck(...a),
    },
    backup: {
      status: (...a: unknown[]) => h.backupStatus(...a),
      create: (...a: unknown[]) => h.backupCreate(...a),
      inspectRestore: (...a: unknown[]) => h.backupInspect(...a),
      stageRestore: vi.fn(),
      undo: (...a: unknown[]) => h.backupUndo(...a),
      discardPending: vi.fn(),
      restartToApply: vi.fn(),
    },
    appConfig: {
      ...real.appConfig,
      get: (...a: unknown[]) => h.appConfigGet(...a),
      logPath: () => Promise.resolve(null),
      getHfTokenStatus: () => Promise.resolve(false),
      migrationPlan: () => Promise.reject(new Error('none')),
    },
    setup: { ...real.setup, get: () => Promise.reject(new Error('no setup record in this test')) },
  };
});

import SettingsPanel from './SettingsPanel';
import { ToastProvider } from './Toasts';
import { invalidate } from './useCached';

const RELEASES = 'https://github.com/f2i-com/oaiy.com/releases/latest';

const updateStatus = (over: Partial<UpdateStatus> = {}): UpdateStatus => ({
  state: 'idle',
  currentVersion: '0.1.0',
  channel: 'stable',
  latestVersion: null,
  notes: null,
  publishedAt: null,
  lastCheckedAt: null,
  error: null,
  failedDuring: null,
  progress: null,
  note: null,
  canAutoUpdate: true,
  manualReason: null,
  blockers: [],
  manualUrl: RELEASES,
  autoCheck: true,
  nextCheckIn: null,
  ...over,
});

const backupStatus = (over: Partial<BackupStatus> = {}): BackupStatus => ({
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

let host: HTMLDivElement;
let root: Root;

async function mount() {
  await act(async () => {
    root.render(
      <ToastProvider>
        <SettingsPanel />
      </ToastProvider>,
    );
  });
  // The panels ask for their own status when they mount.
  await act(async () => {
    await Promise.resolve();
  });
}

const text = () => host.textContent ?? '';
const headings = () => Array.from(host.querySelectorAll('h3.section-title')).map((h3) => h3.textContent?.trim());
const button = (label: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));

beforeEach(() => {
  vi.clearAllMocks();
  invalidate();
  h.appConfigGet.mockResolvedValue({
    activeDir: 'C:/data',
    defaultDir: 'C:/data',
    configuredDir: null,
    isCustom: false,
    restartRequired: false,
    modelsActiveDir: 'C:/data/models',
    modelsDefaultDir: 'C:/data/models',
    modelsConfiguredDir: null,
    modelsIsCustom: false,
    modelsRestartRequired: false,
  });
  h.updateStatus.mockResolvedValue(updateStatus());
  h.backupStatus.mockResolvedValue(backupStatus());
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('Settings holds both About and updates and Backup and restore', () => {
  it('shows each section once, the updates first and the backup after it, each with its own controls', async () => {
    await mount();
    const found = headings();
    expect(found.filter((t) => t === 'About and updates')).toHaveLength(1);
    expect(found.filter((t) => t === 'Backup and restore')).toHaveLength(1);
    expect(found.indexOf('About and updates')).toBeLessThan(found.indexOf('Backup and restore'));
    expect(button('Check for updates')).toBeTruthy();
    expect(button('Create backup')).toBeTruthy();
    // Each read its own status, and only its own.
    expect(h.updateStatus).toHaveBeenCalled();
    expect(h.backupStatus).toHaveBeenCalled();
  });

  it('a check for updates does not touch the backup, and a backup control does not touch the updater', async () => {
    h.updateCheck.mockResolvedValue(updateStatus({ state: 'upToDate', lastCheckedAt: '2026-09-30T00:00:00Z' }));
    await mount();
    await act(async () => {
      (button('Check for updates') as HTMLButtonElement).click();
    });
    expect(h.updateCheck).toHaveBeenCalledTimes(1);
    expect(h.backupCreate).not.toHaveBeenCalled();
    expect(h.backupInspect).not.toHaveBeenCalled();
    // "Create backup" needs its passphrase first: pressing it starts nothing, and the updater is not asked either.
    expect((button('Create backup') as HTMLButtonElement).disabled).toBe(true);
    expect(h.updateDownload).not.toHaveBeenCalled();
    expect(h.updateInstall).not.toHaveBeenCalled();
  });

  it('one section failing to read its status leaves the other working', async () => {
    h.backupStatus.mockRejectedValue(new Error('404: not found'));
    await mount();
    expect(headings()).toContain('About and updates');
    expect(headings()).toContain('Backup and restore');
    expect(text()).toContain('OAIY');
    expect(button('Check for updates')).toBeTruthy();
    // And the other way about.
    await act(async () => root.unmount());
    root = createRoot(host);
    invalidate();
    h.backupStatus.mockResolvedValue(backupStatus());
    h.updateStatus.mockRejectedValue(new Error('503'));
    await mount();
    expect(headings()).toContain('Backup and restore');
    expect(button('Create backup')).toBeTruthy();
  });

  it('a restore waiting for its restart is shown while an update is ready, and neither hides the other', async () => {
    h.updateStatus.mockResolvedValue(updateStatus({ state: 'ready', latestVersion: '0.2.0', blockers: [{ code: 'call', message: 'A phone call is in progress on OAIY’s own line.' }] }));
    h.backupStatus.mockResolvedValue(
      backupStatus({ pendingRestore: { id: 'r1', kind: 'restore', stagedAt: new Date().toISOString(), expiresAt: new Date(Date.now() + 3_600_000).toISOString(), expired: false, files: 3, agentStorage: false, classes: [] } }),
    );
    await mount();
    expect(text()).toContain('Version 0.2.0 is downloaded');
    expect(text()).toContain('A phone call is in progress on OAIY’s own line.');
    expect(text()).toContain('A restore is ready');
    expect(button('Restart to update')).toBeTruthy();
    expect(button('Restart to finish restoring')).toBeTruthy();
  });
});
