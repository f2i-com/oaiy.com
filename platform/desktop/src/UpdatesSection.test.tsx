// Settings > About and updates: which OAIY this is, whether a newer one exists, and the steps of
// getting it (Check, Download, Restart to update), with what is in the way named beside the last
// button. Same convention as the other panels' tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { UpdateStatus } from './api';

const h = vi.hoisted(() => ({
  status: vi.fn(),
  check: vi.fn(),
  download: vi.fn(),
  install: vi.fn(),
  setAutoCheck: vi.fn(),
  openExternal: vi.fn(),
}));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    openExternal: (...a: unknown[]) => h.openExternal(...a),
    updates: {
      status: (...a: unknown[]) => h.status(...a),
      check: (...a: unknown[]) => h.check(...a),
      download: (...a: unknown[]) => h.download(...a),
      install: (...a: unknown[]) => h.install(...a),
      setAutoCheck: (...a: unknown[]) => h.setAutoCheck(...a),
    },
  };
});

import UpdatesSection, { describeProgress } from './UpdatesSection';
import { invalidate } from './useCached';

const RELEASES = 'https://github.com/f2i-com/oaiy.com/releases/latest';

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
  manualUrl: RELEASES,
  autoCheck: true,
  nextCheckIn: null,
  ...over,
});

let host: HTMLDivElement;
let root: Root;

async function mount(initial: UpdateStatus) {
  h.status.mockResolvedValue(initial);
  await act(async () => {
    root.render(<UpdatesSection />);
  });
}

const text = () => host.textContent ?? '';
const button = (label: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));
const click = async (b: HTMLButtonElement | undefined) => {
  expect(b, 'the button is there').toBeTruthy();
  await act(async () => {
    b!.click();
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  invalidate();
  vi.stubGlobal('confirm', () => true);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('About and updates', () => {
  it('says which OAIY this is, that it follows the stable releases, and that nothing has been checked', async () => {
    await mount(status());
    expect(host.querySelector('h3')?.textContent).toBe('About and updates');
    expect(text()).toContain('OAIY 0.1.0');
    expect(text()).toContain('stable releases');
    expect(text()).toContain('Not checked yet');
    expect(text()).toContain('has not looked for updates yet');
    expect(button('Check for updates')).toBeTruthy();
    expect(button('Download')?.textContent).toBeUndefined();
    expect(button('Restart to update')).toBeUndefined();
  });

  it('shows the time of the last check', async () => {
    await mount(status({ state: 'upToDate', lastCheckedAt: '2026-10-01T02:03:04Z', latestVersion: '0.1.0' }));
    expect(text()).not.toContain('Not checked yet');
    expect(text()).toContain(new Date('2026-10-01T02:03:04Z').toLocaleString());
    expect(text()).toContain('OAIY is up to date.');
  });

  it('Check for updates asks the desktop and shows what it found', async () => {
    await mount(status());
    h.check.mockResolvedValue(status({ state: 'upToDate', lastCheckedAt: '2026-10-01T02:03:04Z', latestVersion: '0.1.0', nextCheckIn: 30 }));
    await click(button('Check for updates'));
    expect(h.check).toHaveBeenCalledTimes(1);
    expect(text()).toContain('OAIY is up to date.');
    // And it cannot be asked again straight away: the limit is the desktop's, the button shows it.
    expect(button('Check for updates')?.disabled).toBe(true);
    expect(button('Check for updates')?.title).toContain('30 seconds');
  });

  it('shows a newer version with its date and notes, and Download starts the download', async () => {
    await mount(status({ state: 'available', latestVersion: '0.2.0', publishedAt: '2026-10-01T02:03:04Z', notes: 'Calls keep their audio.', lastCheckedAt: '2026-10-01T03:00:00Z' }));
    expect(text()).toContain('Version 0.2.0 is available.');
    expect(text()).toContain('OAIY 0.2.0');
    expect(text()).toContain(`published ${new Date('2026-10-01T02:03:04Z').toLocaleDateString()}`);
    expect(text()).toContain('Calls keep their audio.');
    const downloading = status({ state: 'downloading', latestVersion: '0.2.0', progress: { downloaded: 0, total: null } });
    h.download.mockResolvedValue(downloading);
    // (The desktop's status says the same from now on: the faster poll that starts reads it again.)
    h.status.mockResolvedValue(downloading);
    await click(button('Download 0.2.0'));
    expect(h.download).toHaveBeenCalledTimes(1);
    expect(h.download.mock.calls[0]).toEqual([]);
    expect(text()).toContain('Downloading version 0.2.0…');
    expect(host.querySelector('[role="progressbar"]')).toBeTruthy();
    expect(button('Download 0.2.0')).toBeUndefined();
  });

  it('shows the progress of a download in bytes and percent, and just the bytes when the size is not known', async () => {
    const mib = 1024 * 1024;
    await mount(status({ state: 'downloading', latestVersion: '0.2.0', progress: { downloaded: 24 * mib, total: 96 * mib } }));
    expect(text()).toContain('24.0 MiB of 96.0 MiB (25%)');
    expect(host.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('25');
    expect(describeProgress({ downloaded: 5 * mib, total: null })).toBe('5.0 MiB');
    expect(describeProgress({ downloaded: 200 * mib, total: 100 * mib })).toContain('(100%)');
    expect(button('Check for updates')?.disabled).toBe(true);
  });

  it('says a download is ready, checked, and offers Restart to update', async () => {
    await mount(status({ state: 'ready', latestVersion: '0.2.0' }));
    expect(text()).toContain('Version 0.2.0 is downloaded and its signature is checked.');
    expect(button('Restart to update')?.disabled).toBe(false);
    expect(button('Download')).toBeUndefined();
  });

  it('turns Restart to update off while anything is in the way, and names each thing beside it', async () => {
    const blockers = [
      { code: 'call', message: "A phone call is in progress on OAIY's own line." },
      { code: 'download', message: '2 models or files are downloading.' },
    ];
    await mount(status({ state: 'ready', latestVersion: '0.2.0', blockers }));
    const restart = button('Restart to update');
    expect(restart?.disabled).toBe(true);
    const list = host.querySelector('ul[aria-label="Why OAIY cannot restart now"]');
    expect(Array.from(list!.querySelectorAll('li')).map((li) => li.textContent)).toEqual(blockers.map((b) => b.message));
    await click(restart);
    expect(h.install).not.toHaveBeenCalled();
  });

  it('names a call the phone plugin reports, a plugin that cannot say and engines that cannot say, each as it is worded', async () => {
    // The codes the desktop sends besides the first few: a call the plugin knows of (OAIY's own line sees none), and the two "can't tell" reasons.
    const blockers = [
      { code: 'phoneCall', message: 'Aokie Phone Bridge reports a phone call (ringing, in progress or on hold).' },
      { code: 'callUnknown', message: "OAIY can't tell whether a phone call is live: Aokie Phone Bridge did not give an answer (it did not answer within 3 s). It does not restart while it can't tell; stopping that plugin (Connections, Plugins) lets it." },
      { code: 'enginesUnknown', message: "OAIY can't tell whether the engines are busy: no answer within 2 s. It does not restart while it can't tell." },
    ];
    await mount(status({ state: 'ready', latestVersion: '0.2.0', blockers }));
    expect(button('Restart to update')?.disabled).toBe(true);
    const list = host.querySelector('ul[aria-label="Why OAIY cannot restart now"]');
    expect(Array.from(list!.querySelectorAll('li')).map((li) => li.textContent)).toEqual(blockers.map((b) => b.message));
  });

  it('shows no reasons and an enabled button once nothing is in the way', async () => {
    await mount(status({ state: 'ready', latestVersion: '0.2.0', blockers: [] }));
    expect(host.querySelector('ul.update-blockers')).toBeNull();
    expect(button('Restart to update')?.disabled).toBe(false);
  });

  it('asks before restarting, and installs only when told yes', async () => {
    await mount(status({ state: 'ready', latestVersion: '0.2.0' }));
    h.install.mockResolvedValue(undefined);
    const asked: string[] = [];
    vi.stubGlobal('confirm', (question: string) => {
      asked.push(question);
      return false;
    });
    await click(button('Restart to update'));
    expect(asked[0]).toContain('0.2.0');
    expect(asked[0]).toContain('will not answer calls or texts');
    expect(h.install).not.toHaveBeenCalled();
    vi.stubGlobal('confirm', () => true);
    await click(button('Restart to update'));
    expect(h.install).toHaveBeenCalledTimes(1);
    // The command takes no argument: what is installed is what was downloaded and checked.
    expect(h.install.mock.calls[0]).toEqual([]);
  });

  it('says why a copy cannot update itself and offers the releases page instead', async () => {
    const manualReason = 'This copy of OAIY was installed from the MSI package, which cannot update itself.';
    await mount(status({ state: 'available', latestVersion: '0.2.0', canAutoUpdate: false, manualReason }));
    expect(text()).toContain(manualReason);
    expect(button('Download 0.2.0')).toBeUndefined();
    expect(button('Restart to update')).toBeUndefined();
    await click(button('Download manually'));
    expect(h.openExternal).toHaveBeenCalledWith(RELEASES);
  });

  it('does not offer a manual download where the copy can update itself and nothing failed', async () => {
    await mount(status({ state: 'available', latestVersion: '0.2.0' }));
    expect(button('Download manually')).toBeUndefined();
  });

  it('shows a failure in plain words, with the step that failed, and can try the download again', async () => {
    await mount(status({ state: 'failed', latestVersion: '0.2.0', failedDuring: 'download', error: 'The downloaded update does not match its signature, so it was thrown away. It may be damaged, or not made by OAIY.' }));
    const alert = host.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('The update was not downloaded.');
    expect(alert?.textContent).toContain('does not match its signature');
    expect(button('Restart to update')).toBeUndefined();
    expect(button('Download manually')).toBeTruthy();
    h.download.mockResolvedValue(status({ state: 'downloading', latestVersion: '0.2.0', progress: { downloaded: 0, total: null } }));
    await click(button('Try downloading again'));
    expect(h.download).toHaveBeenCalledTimes(1);
  });

  it('shows a failed check as such, and a failed install', async () => {
    await mount(status({ state: 'failed', failedDuring: 'check', error: 'Could not reach the update server. Check the internet connection and try again.' }));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('OAIY could not check for updates.');
    expect(button('Try downloading again')).toBeUndefined();
    await act(async () => root.unmount());
    root = createRoot(host);
    invalidate();
    await mount(status({ state: 'failed', latestVersion: '0.2.0', failedDuring: 'install', error: 'The update could not be installed: the installer could not be started (no permission). OAIY started again what it had stopped, and is running as before.' }));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('The update was not installed.');
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('running as before');
  });

  it('shows an error a command answers with, and reads the status again', async () => {
    await mount(status({ state: 'ready', latestVersion: '0.2.0' }));
    h.install.mockRejectedValue(new Error('OAIY cannot restart now: A phone call is in progress.'));
    h.status.mockResolvedValue(status({ state: 'ready', latestVersion: '0.2.0', blockers: [{ code: 'call', message: 'A phone call is in progress.' }] }));
    await click(button('Restart to update'));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('OAIY cannot restart now');
    expect(h.status.mock.calls.length).toBeGreaterThan(1);
    expect(button('Restart to update')?.disabled).toBe(true);
  });

  it('says it is installing, and offers nothing to press meanwhile', async () => {
    await mount(status({ state: 'installing', latestVersion: '0.2.0' }));
    expect(text()).toContain('Installing… OAIY is closing and will open again by itself.');
    expect(button('Check for updates')?.disabled).toBe(true);
    expect(button('Restart to update')).toBeUndefined();
  });

  it('does not check again while a check runs', async () => {
    await mount(status({ state: 'checking' }));
    expect(text()).toContain('Looking for a newer version…');
    expect(button('Check for updates')?.disabled).toBe(true);
  });

  it('switches the automatic check off and on', async () => {
    await mount(status({ autoCheck: true }));
    const box = host.querySelector<HTMLInputElement>('.update-auto input');
    expect(box?.checked).toBe(true);
    h.setAutoCheck.mockResolvedValue(undefined);
    h.status.mockResolvedValue(status({ autoCheck: false }));
    await act(async () => {
      box!.click();
    });
    expect(h.setAutoCheck).toHaveBeenCalledWith(false);
    expect(host.querySelector<HTMLInputElement>('.update-auto input')?.checked).toBe(false);
  });

  it('says so when the status cannot be read at all', async () => {
    h.status.mockRejectedValue(new Error('Failed to fetch'));
    await act(async () => {
      root.render(<UpdatesSection />);
    });
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('Could not read the update status');
  });
});
