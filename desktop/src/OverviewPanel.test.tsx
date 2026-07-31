// Overview panel. The behaviour worth pinning here is a regression I created:
// the setup guide hides the "Next steps" section (they cover the same ground),
// and the "N runs have failed" warning had been added INSIDE that section — so
// on a fresh install, when the guide is open by default and a failure is most
// likely, the machine said nothing about it.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { servicesMock, pluginsMock, providersMock, pairedMock, statusMock, nodeInstallMock } =
  vi.hoisted(() => ({
    servicesMock: vi.fn(),
    pluginsMock: vi.fn(),
    providersMock: vi.fn(),
    pairedMock: vi.fn(),
    statusMock: vi.fn(),
    nodeInstallMock: vi.fn(),
  }));

vi.mock('./api', () => ({
  services: { list: servicesMock },
  plugins: { list: pluginsMock },
  aiProviders: { list: providersMock },
  pairing: { paired: pairedMock },
  bridge: { status: statusMock },
  nodeRuntime: { install: nodeInstallMock },
  openExternal: vi.fn(),
}));

import OverviewPanel from './OverviewPanel';
import { invalidate } from './useCached';
import { dismissGuide, reopenGuide } from './setupGuide';

const runtime = (failed: number) => ({
  ready: true,
  deviceId: 'dev_1',
  flowRuntime: { cliResolved: true, cliKind: 'node', detail: null },
  runs: { queued: 0, known: 10, failed },
  nodeRuntime: { available: true, installing: false, installsVersion: '22.14.0', source: 'system' },
  plugins: { serving: 1, total: 1 },
});

let host: HTMLDivElement;
let root: Root;
const onNavigate = vi.fn();

async function mount() {
  await act(async () => {
    root.render(<OverviewPanel onNavigate={onNavigate} onOpenPluginScreen={vi.fn()} />);
  });
}

const text = () => host.textContent ?? '';
const button = (label: string) =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label));

beforeEach(() => {
  invalidate();
  onNavigate.mockReset();
  servicesMock.mockResolvedValue({ services: [], dataDir: 'C:\\d' });
  pluginsMock.mockResolvedValue({ plugins: [] });
  providersMock.mockResolvedValue({
    providers: [{ id: 'p', enabled: true, hasKey: true, allowLocal: false }],
  });
  pairedMock.mockResolvedValue({ paired: [] });
  statusMock.mockResolvedValue(runtime(3));
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  reopenGuide();
});

describe('OverviewPanel failure reporting', () => {
  it('reports failed runs even while the setup guide is open', async () => {
    reopenGuide(); // guide auto-opens, as on a fresh install
    await mount();

    // The guide is up (it reads "You're set up" here because the fixture has a
    // ready runtime and a keyed provider — either way the panel is rendered).
    expect(host.querySelector('.setup-guide')).not.toBeNull();
    // The regression: "Next steps" yields to the guide, and this warning used
    // to live inside it — so a fresh install stayed silent about real failures.
    expect(text()).toContain('3 runs have failed');
  });

  it('still reports them once the guide is dismissed', async () => {
    dismissGuide();
    await mount();
    expect(host.querySelector('.setup-guide')).toBeNull();
    expect(text()).toContain('3 runs have failed');
  });

  it('takes the user to the run history', async () => {
    dismissGuide();
    await mount();
    await act(async () => {
      button('See why')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(onNavigate).toHaveBeenCalledWith('runs');
  });

  it('says nothing when nothing has failed', async () => {
    statusMock.mockResolvedValue(runtime(0));
    dismissGuide();
    await mount();
    // A permanent "0 failed" line is noise that trains people to ignore the row.
    expect(text()).not.toContain('failed on this machine');
  });

  it('uses singular wording for one failure', async () => {
    statusMock.mockResolvedValue(runtime(1));
    dismissGuide();
    await mount();
    expect(text()).toContain('1 run has failed');
  });
});
