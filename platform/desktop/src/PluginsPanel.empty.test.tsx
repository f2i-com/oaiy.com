// The Plugins page in words for a person: with no plugin installed it shows the ones OAIY knows (the bundled
// catalogue), not how a plugin is built; it does not list the names a plugin used before OAIY's own; and with OAIY not
// answering it says so, once, with a way to ask again.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ list: vi.fn(), catalog: vi.fn(), push: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    plugins: { ...real.plugins, list: m.list },
    serviceDefinitions: { list: vi.fn().mockResolvedValue({ definitions: [] }) },
    setup: {
      ...real.setup,
      get: vi.fn().mockResolvedValue({ firstRun: { finished: true, skipped: [], chosenPlugins: [] }, plugins: {} }),
      catalog: m.catalog,
      check: vi.fn(),
    },
  };
});
vi.mock('./Toasts', () => ({ useToast: () => ({ push: m.push }) }));

import PluginsPanel from './PluginsPanel';
import { ApiError, NOT_ANSWERING, type PluginRecord } from './api';
import { invalidate } from './useCached';
import { resetSetupState } from './useSetupState';

let host: HTMLDivElement;
let root: Root;

async function settle() {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function render() {
  await act(async () => {
    root.render(<PluginsPanel />);
  });
  await settle();
}

const aokie = (path: string | null) => ({
  id: 'aokie',
  name: 'AI Receptionist',
  description: 'Answers your business phone’s calls and texts.',
  needs: 'A Bluetooth USB adapter, and the phone to pair with it.',
  source: { kind: 'folder' as const, path, note: 'For every computer.', notes: { macos: 'For the Mac.', windows: 'For Windows.', linux: 'For Linux.' } },
  installed: false,
  installedVersion: null,
});

beforeEach(() => {
  resetSetupState();
  invalidate('pluginsSnapshot');
  for (const f of Object.values(m)) f.mockReset();
  m.catalog.mockResolvedValue({ plugins: [aokie(null)] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('the Plugins page', () => {
  it('with none installed, shows the plugins OAIY knows, and how to get one on this computer', async () => {
    m.list.mockResolvedValue({ root: '/plugins', plugins: [] });
    await render();
    const empty = host.querySelector('.plugin-catalog')!;
    expect(empty.textContent).toContain('No plugins installed yet.');
    expect(empty.textContent).toContain('AI Receptionist');
    expect(empty.textContent).toContain('Needs: A Bluetooth USB adapter');
    // (the jsdom window names no system: the note for every computer)
    expect(empty.textContent).toContain('For every computer.');
    expect(empty.textContent).not.toContain('manifest.json');
    expect(empty.textContent).not.toContain('Bridge Protocol');
  });

  it('puts a copy found on this computer in the install box with one press', async () => {
    m.list.mockResolvedValue({ root: '/plugins', plugins: [] });
    m.catalog.mockResolvedValue({ plugins: [aokie('/Users/sam/aokie/bundle')] });
    await render();
    const use = Array.from(host.querySelectorAll('button')).find((b) => b.textContent === 'Use this folder')!;
    await act(async () => use.click());
    expect(host.querySelector<HTMLInputElement>('.dl-form input')!.value).toBe('/Users/sam/aokie/bundle');
  });

  it('does not list the names a plugin used before OAIY’s own', async () => {
    const legacy: PluginRecord = {
      id: 'aokie',
      state: 'running',
      dir: '/plugins/aokie',
      userDisabled: false,
      restartAttempts: 0,
      manifest: { name: 'Aokie Phone Bridge', version: '0.0.3' },
      legacyCapabilities: [['flow.run', 'oaiy.flow.run'], ['companion.admission', 'oaiy.companion.admission']],
    } as PluginRecord;
    m.list.mockResolvedValue({ root: '/plugins', plugins: [legacy] });
    await render();
    expect(host.textContent).toContain('Aokie Phone Bridge');
    expect(host.textContent).not.toContain('legacy');
    expect(host.textContent).not.toContain('flow.run');
  });

  it('says OAIY is not answering, once, with a way to ask again', async () => {
    m.list.mockRejectedValue(new ApiError(0, NOT_ANSWERING));
    await render();
    const banner = host.querySelector('.load-error')!;
    expect(banner.textContent).toContain(NOT_ANSWERING);
    expect(banner.textContent).not.toContain('could not be read');
    expect(banner.textContent).not.toContain('Failed to fetch');
    expect(host.textContent).not.toContain('Loading plugins…');
    m.list.mockResolvedValue({ root: '/plugins', plugins: [] });
    await act(async () => banner.querySelector<HTMLButtonElement>('button')!.click());
    await settle();
    expect(host.querySelector('.load-error')).toBeNull();
  });
});
