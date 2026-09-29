// A release build does not start a plugin nobody signed until the person trusts that exact
// package. The card says so, and offers "Trust this plugin" for that and nothing else: a
// package that carries a signature is verified or quarantined by it alone.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ list: vi.fn(), trust: vi.fn(), start: vi.fn(), push: vi.fn(), install: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    plugins: { ...real.plugins, list: m.list, trust: m.trust, start: m.start, install: m.install },
    serviceDefinitions: { list: vi.fn().mockResolvedValue({ definitions: [] }) },
    setup: {
      ...real.setup,
      get: vi.fn().mockResolvedValue({ firstRun: { finished: true, skipped: [], chosenPlugins: [] }, plugins: {} }),
      check: vi.fn(),
    },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: m.push }) }));

import PluginsPanel from './PluginsPanel';
import { invalidate } from './useCached';
import { resetSetupState } from './useSetupState';
import type { PackageTrust, PluginRecord } from './api';

/** A plugin the host is holding back: no manifest, disabled, with why. */
function held(trust: PackageTrust, extra: Partial<PluginRecord> = {}): PluginRecord {
  return {
    id: 'aokie',
    state: 'disabled',
    reason: `Not started. ${trust.reason ?? ''}`,
    dir: 'C:\\plugins\\aokie',
    userDisabled: false,
    restartAttempts: 0,
    trust,
    ...extra,
  };
}

const UNSIGNED: PackageTrust = {
  state: 'unsigned',
  reason: 'Not signed by a publisher this OAIY trusts. If you built it yourself or know where it came from, you can trust this exact package.',
};
const QUARANTINED: PackageTrust = { state: 'quarantined', reason: 'Quarantined: digest mismatch: aokie-plugin.exe.' };

let host: HTMLDivElement;
let root: Root;

async function settle() {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render(records: PluginRecord[]) {
  m.list.mockReset().mockResolvedValue({ root: 'C:/plugins', plugins: records });
  await act(async () => {
    root.render(<PluginsPanel />);
  });
  await settle();
}

const button = (text: string): HTMLButtonElement | undefined =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as HTMLButtonElement | undefined;

beforeEach(() => {
  resetSetupState();
  invalidate('pluginsSnapshot');
  m.trust.mockReset().mockResolvedValue({});
  m.start.mockReset();
  m.push.mockReset();
  m.install.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  vi.restoreAllMocks();
  act(() => root.unmount());
  host.remove();
});

describe('an unsigned plugin in a release build', () => {
  it('shows the unsigned badge and its reason, cannot be started, and offers to be trusted', async () => {
    await render([held(UNSIGNED)]);
    expect(host.querySelector('.card-head .badge[data-trust="unsigned"]')?.textContent).toBe('unsigned');
    expect(host.querySelector('.card-reason')?.textContent).toContain('you can trust this exact package');
    const start = button('Start')!;
    expect(start.disabled).toBe(true);
    expect(start.title).toContain('Trust it first');
    expect(button('Trust this plugin')).toBeDefined();
  });

  it('asks before it trusts, and trusts only the plugin it was asked about', async () => {
    await render([held(UNSIGNED)]);
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    await act(async () => {
      button('Trust this plugin')!.click();
    });
    await settle();
    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(confirmSpy.mock.calls[0][0]).toContain('It is not signed');
    expect(confirmSpy.mock.calls[0][0]).toContain('exact package');
    expect(m.trust).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    m.list.mockClear();
    await act(async () => {
      button('Trust this plugin')!.click();
    });
    await settle();
    expect(m.trust).toHaveBeenCalledTimes(1);
    expect(m.trust).toHaveBeenCalledWith('aokie');
    expect(m.list).toHaveBeenCalled();
    expect(m.push).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Trusted aokie' }));
  });

  it('says why when trusting it fails, and leaves the card as it was', async () => {
    await render([held(UNSIGNED)]);
    m.trust.mockRejectedValue(new Error('aokie cannot be trusted: a symbolic link is present: link'));
    await act(async () => {
      button('Trust this plugin')!.click();
    });
    await settle();
    expect(m.push).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', title: 'Action failed for "aokie"', body: expect.stringContaining('symbolic link') }),
    );
    expect(button('Trust this plugin')).toBeDefined();
  });

  it('after it is trusted, the badge says so and Start is there to use', async () => {
    await render([
      {
        id: 'aokie',
        state: 'installed',
        reason: 'Not started yet.',
        dir: 'C:\\plugins\\aokie',
        userDisabled: false,
        restartAttempts: 0,
        manifest: { name: 'Aokie Phone Bridge', version: '0.1.0' },
        trust: { state: 'trusted-local', reason: 'You trusted this exact package on 2026-09-29.', trustedAt: '2026-09-29T01:02:03Z' },
      },
    ]);
    expect(host.querySelector('.card-head .badge[data-trust="trusted-local"]')?.textContent).toBe('trusted by you');
    expect(button('Trust this plugin')).toBeUndefined();
    expect(button('Start')!.disabled).toBe(false);
  });
});

describe('the trust action is for an unsigned package and nothing else', () => {
  const manifest = { name: 'Aokie Phone Bridge', version: '0.1.0' };
  const loaded = (trust: PackageTrust | undefined): PluginRecord => ({
    id: 'aokie',
    state: 'installed',
    reason: 'Not started yet.',
    dir: 'C:\\plugins\\aokie',
    userDisabled: false,
    restartAttempts: 0,
    manifest,
    trust,
  });

  it.each([
    ['verified', { state: 'verified', publisher: 'Aokie', keyId: 'fl-aokie-2026a' } as PackageTrust],
    ['unsigned-dev', { state: 'unsigned-dev', reason: 'Not signed. It runs because this is a developer build.' } as PackageTrust],
    ['trusted-local', { state: 'trusted-local', reason: 'You trusted this exact package.' } as PackageTrust],
    ['no verdict (the manifest could not be loaded)', undefined],
  ])('is not offered for %s', async (_name, trust) => {
    await render([loaded(trust)]);
    expect(button('Trust this plugin')).toBeUndefined();
  });

  it('is not offered for a quarantined package: a signature that fails is not something a click fixes', async () => {
    await render([held(QUARANTINED)]);
    expect(host.querySelector('.card-head .badge[data-trust="quarantined"]')?.textContent).toBe('quarantined');
    expect(host.querySelector('.card-reason')?.textContent).toContain('digest mismatch: aokie-plugin.exe');
    expect(button('Trust this plugin')).toBeUndefined();
    const start = button('Start')!;
    expect(start.disabled).toBe(true);
    expect(start.title).toContain('failed its signature check');
  });

  it('is not offered while the plugin is running, but a running plugin whose folder stopped verifying says so', async () => {
    await render([{ ...loaded(QUARANTINED), state: 'running', reason: undefined }]);
    expect(button('Trust this plugin')).toBeUndefined();
    expect(host.querySelector('.card-reason')?.textContent).toContain('Quarantined: digest mismatch');
    expect(button('Stop')).toBeDefined();
  });
});

describe('installing a plugin that will not start until it is trusted', () => {
  it('says so, instead of "Click Start to run it"', async () => {
    await render([]);
    m.install.mockResolvedValue({ id: 'aokie', name: 'Aokie Phone Bridge', version: '0.1.0', replaced: false, trust: UNSIGNED });
    const input = host.querySelector('input[type="text"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(input, 'C:\\build\\aokie');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      button('Install')!.click();
    });
    await settle();
    expect(m.install).toHaveBeenCalledWith('C:\\build\\aokie');
    expect(m.push).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Installed Aokie Phone Bridge v0.1.0', body: 'It is not signed, so it will not start until you trust it.' }),
    );
  });
});
