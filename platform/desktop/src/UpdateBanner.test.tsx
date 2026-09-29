// The Overview's word that a newer OAIY exists: only when one is available or ready, dismissible
// until there is a NEWER version. Same convention as the other tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { UpdateStatus } from './api';

const h = vi.hoisted(() => ({ status: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, updates: { status: (...a: unknown[]) => h.status(...a) } };
});

import UpdateBanner, { bannerVersion, DISMISSED_KEY } from './UpdateBanner';
import { invalidate } from './useCached';
import { pollInterval, POLL_FAST_MS, POLL_SLOW_MS } from './useUpdateStatus';

const status = (over: Partial<UpdateStatus> = {}): UpdateStatus => ({
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
  manualUrl: 'https://github.com/f2i-com/oaiy.com/releases/latest',
  autoCheck: true,
  nextCheckIn: null,
  ...over,
});

let host: HTMLDivElement;
let root: Root;
const onOpenSettings = vi.fn();

async function mount(s: UpdateStatus) {
  h.status.mockResolvedValue(s);
  await act(async () => {
    root.render(<UpdateBanner onOpenSettings={onOpenSettings} />);
  });
}

const text = () => host.textContent ?? '';

beforeEach(() => {
  vi.clearAllMocks();
  invalidate();
  localStorage.clear();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('the Overview banner', () => {
  it('says nothing while there is nothing to install', async () => {
    for (const s of [
      status(),
      status({ state: 'checking' }),
      status({ state: 'upToDate', latestVersion: '0.1.0' }),
      status({ state: 'downloading', latestVersion: '0.2.0' }),
      status({ state: 'installing', latestVersion: '0.2.0' }),
      status({ state: 'failed', latestVersion: '0.2.0', failedDuring: 'download', error: 'The download failed.' }),
    ]) {
      invalidate();
      await act(async () => root.unmount());
      root = createRoot(host);
      await mount(s);
      expect(host.querySelector('.update-banner'), s.state).toBeNull();
    }
  });

  it('says a newer version is available, and opens Settings', async () => {
    await mount(status({ state: 'available', latestVersion: '0.2.0' }));
    expect(text()).toContain('OAIY 0.2.0 is available.');
    const open = Array.from(host.querySelectorAll('button')).find((b) => b.textContent === 'See what is new');
    await act(async () => open!.click());
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  it('says a version is downloaded and ready, and points at Settings to restart', async () => {
    await mount(status({ state: 'ready', latestVersion: '0.2.0' }));
    expect(text()).toContain('OAIY 0.2.0 is downloaded and ready to install.');
    expect(text()).toContain('Open Settings to restart');
  });

  it('is dismissed until there is a newer version than the one dismissed', async () => {
    await mount(status({ state: 'available', latestVersion: '0.2.0' }));
    const dismiss = host.querySelector<HTMLButtonElement>('button[aria-label="Dismiss until the next version"]');
    await act(async () => dismiss!.click());
    expect(host.querySelector('.update-banner')).toBeNull();
    expect(localStorage.getItem(DISMISSED_KEY)).toBe('0.2.0');

    // Back on the page later: still dismissed, also when the update is now downloaded (the same version).
    await act(async () => root.unmount());
    root = createRoot(host);
    invalidate();
    await mount(status({ state: 'ready', latestVersion: '0.2.0' }));
    expect(host.querySelector('.update-banner')).toBeNull();

    // A newer version brings it back.
    await act(async () => root.unmount());
    root = createRoot(host);
    invalidate();
    await mount(status({ state: 'available', latestVersion: '0.3.0' }));
    expect(text()).toContain('OAIY 0.3.0 is available.');
  });

  it('still works with no storage at all', async () => {
    const broken = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const failing = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    await mount(status({ state: 'available', latestVersion: '0.2.0' }));
    const dismiss = host.querySelector<HTMLButtonElement>('button[aria-label="Dismiss until the next version"]');
    await act(async () => dismiss!.click());
    expect(host.querySelector('.update-banner')).toBeNull();
    broken.mockRestore();
    failing.mockRestore();
  });
});

describe('which status gets a banner', () => {
  it('only an available or ready update, with a version, that was not dismissed', () => {
    expect(bannerVersion(null, null)).toBeNull();
    expect(bannerVersion({ state: 'available', latestVersion: '0.2.0' }, null)).toBe('0.2.0');
    expect(bannerVersion({ state: 'ready', latestVersion: '0.2.0' }, null)).toBe('0.2.0');
    expect(bannerVersion({ state: 'available', latestVersion: '0.2.0' }, '0.2.0')).toBeNull();
    expect(bannerVersion({ state: 'available', latestVersion: '0.2.0' }, '0.1.5')).toBe('0.2.0');
    expect(bannerVersion({ state: 'available', latestVersion: null }, null)).toBeNull();
    for (const state of ['idle', 'checking', 'upToDate', 'downloading', 'installing', 'failed']) {
      expect(bannerVersion({ state, latestVersion: '0.2.0' }, null), state).toBeNull();
    }
  });
});

describe('how often the status is read', () => {
  it('quickly while something moves, slowly otherwise', () => {
    expect(pollInterval(null)).toBe(POLL_SLOW_MS);
    for (const state of ['checking', 'downloading', 'installing'] as const) expect(pollInterval(status({ state }))).toBe(POLL_FAST_MS);
    for (const state of ['idle', 'upToDate', 'available', 'ready', 'failed'] as const) expect(pollInterval(status({ state }))).toBe(POLL_SLOW_MS);
  });
});
