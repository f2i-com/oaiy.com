// Services panel load behaviour. The user-visible symptom was "slow loading":
// the panel started from null on every mount, so switching to Services always
// showed "Loading services…" and a round trip, even though Overview polls the
// same endpoint and had the answer cached.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listMock, configMock, gpusMock, pushMock, autostartMock } = vi.hoisted(() => ({
  listMock: vi.fn(),
  configMock: vi.fn(),
  gpusMock: vi.fn(),
  pushMock: vi.fn(),
  autostartMock: vi.fn(),
}));

vi.mock('./api', () => ({
  services: {
    list: listMock,
    start: vi.fn(),
    stop: vi.fn(),
    install: vi.fn(),
    uninstall: vi.fn(),
    cancelInstall: vi.fn(),
    repair: vi.fn(),
    setAutostart: autostartMock,
    delete: vi.fn(),
    add: vi.fn(),
    import: vi.fn(),
    export: vi.fn(),
    logs: vi.fn(),
  },
  appConfig: {
    get: configMock,
    listGpus: gpusMock,
    setServiceGpu: vi.fn(),
  },
  openExternal: vi.fn(),
  NOT_ANSWERING: 'OAIY isn’t answering. It may still be starting, or it has stopped.',
}));
vi.mock('./Toasts', () => ({ useToast: () => ({ push: pushMock }) }));
vi.mock('./LogsViewer', () => ({ default: () => null }));

import ServicesPanel, { categoryName, crashWords } from './ServicesPanel';
import { invalidate, put } from './useCached';

const service = (over: Record<string, unknown> = {}) => ({
  id: 'playwright-browser',
  name: 'Playwright Browser',
  category: 'Browser',
  status: 'stopped',
  port: 17880,
  defaultPort: 17880,
  installed: true,
  gpu: null,
  error: null,
  ...over,
});

const SNAPSHOT = { services: [service()], dataDir: 'C:\\data' };

let host: HTMLDivElement;
let root: Root;

async function mount() {
  await act(async () => {
    root.render(<ServicesPanel />);
  });
}

const text = () => host.textContent ?? '';

beforeEach(() => {
  invalidate();
  listMock.mockReset();
  configMock.mockReset();
  gpusMock.mockReset();
  autostartMock.mockReset();
  autostartMock.mockResolvedValue(undefined);
  listMock.mockResolvedValue(SNAPSHOT);
  configMock.mockResolvedValue({});
  gpusMock.mockResolvedValue([]);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('ServicesPanel loading', () => {
  it('paints from the cache instead of flashing "Loading services…"', async () => {
    put('servicesSnapshot', SNAPSHOT);
    // Never resolves: proves the first paint owes nothing to the network.
    listMock.mockReturnValue(new Promise(() => {}));

    await act(async () => {
      root.render(<ServicesPanel />);
    });

    expect(text()).not.toContain('Loading services');
    expect(text()).toContain('Playwright Browser');
  });

  it('says the services could not be read, with a way to ask again, and OAIY not answering as that alone', async () => {
    listMock.mockRejectedValueOnce(new Error('the services file is damaged'));
    await mount();
    expect(host.querySelector('.load-error')!.textContent).toContain('The services could not be read: the services file is damaged');
    const again = Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes('Try again'))!;
    await act(async () => again.click());
    expect(listMock).toHaveBeenCalledTimes(2);
    expect(host.querySelector('.load-error')).toBeNull();
    act(() => root.unmount());
    root = createRoot(host);
    listMock.mockRejectedValueOnce(new Error('OAIY isn’t answering. It may still be starting, or it has stopped.'));
    await mount();
    expect(host.querySelector('.load-error')!.textContent?.trim()).toBe('OAIY isn’t answering. It may still be starting, or it has stopped. Try again');
  });

  it('still shows the loading state on a genuinely cold start', async () => {
    listMock.mockReturnValue(new Promise(() => {}));
    await act(async () => {
      root.render(<ServicesPanel />);
    });
    // Nothing cached and nothing fetched yet — saying so is correct here.
    expect(text()).toContain('Loading services');
  });

  it('fills the cache Overview reads, so arriving there is instant too', async () => {
    await mount();
    const { peek } = await import('./useCached');
    expect(peek('servicesSnapshot')).toEqual(SNAPSHOT);
    expect(peek('services')).toEqual(SNAPSHOT.services);
  });

  it('offers no model picker, even for a template of its own named like a retired one', async () => {
    // Models are the engine's (OAIY → Engines). A person's own llama.cpp or
    // Ollama template is still listed, as any service is, but OAIY no longer
    // picks its model.
    listMock.mockResolvedValue({
      ...SNAPSHOT,
      services: [
        service({ id: 'llama-cpp', name: 'My llama.cpp', category: 'LLM', status: 'running' }),
        service({ id: 'ollama', name: 'My Ollama', category: 'LLM' }),
      ],
    });
    await mount();
    expect(text()).toContain('My llama.cpp');
    expect(text()).toContain('My Ollama');
    expect(host.querySelector('.llama-model, .ollama-model')).toBeNull();
    expect(text()).not.toContain('Select a model');
    expect(text()).not.toContain('Custom name');
  });
});

// "Start with the app" is a stored preference, not a second Start button. The
// distinction matters on both edges: ticking must not launch the service now,
// and a failed write must not leave a ticked box that will be gone at the next
// launch (the service would simply never come up, with nothing to explain it).
describe('ServicesPanel start-with-the-app', () => {
  const box = () => host.querySelector<HTMLInputElement>('input[type="checkbox"]');

  it('reflects the stored preference rather than whether it is running', async () => {
    listMock.mockResolvedValue({
      ...SNAPSHOT,
      services: [service({ status: 'stopped', autostart: true })],
    });
    await mount();
    expect(box()?.checked).toBe(true);
  });

  it('is unticked for a snapshot from a desktop build that has no such field', async () => {
    await mount();
    expect(box()).not.toBeNull();
    expect(box()?.checked).toBe(false);
  });

  it('saves the preference without starting the service', async () => {
    await mount();
    const el = box()!;
    // el.click() lets jsdom toggle `checked` and fire the click React maps to
    // onChange; setting `checked` by hand first makes React see no change.
    await act(async () => {
      el.click();
    });
    expect(autostartMock).toHaveBeenCalledWith('playwright-browser', true);
    // The only lever that starts a service is Start.
    const { services: api } = await import('./api');
    expect(api.start).not.toHaveBeenCalled();
  });

  it('rolls the tick back when the preference could not be saved', async () => {
    autostartMock.mockRejectedValue(new Error('disk full'));
    await mount();
    const el = box()!;
    // el.click() lets jsdom toggle `checked` and fire the click React maps to
    // onChange; setting `checked` by hand first makes React see no change.
    await act(async () => {
      el.click();
    });
    expect(box()?.checked).toBe(false);
    expect(text()).toContain('disk full');
  });
});

// The page as its owner reads it (A7): groups named in words, a row with the buttons pressed every day and the rest
// behind "More", a stopped service's trouble said once, and where services are kept at the foot, closed, for
// someone who writes one.
describe('ServicesPanel in words for its owner', () => {
  const menuItems = () => Array.from(document.body.querySelectorAll<HTMLButtonElement>('.more-menu [role=menuitem]'));
  const rowButtons = () => Array.from(host.querySelectorAll<HTMLButtonElement>('.service-actions > button')).map((b) => b.textContent?.trim() || b.getAttribute('aria-label'));
  const gaveUp = {
    id: 'reader', name: 'Invoice reader', category: 'custom', status: 'errored', needsRepair: true, restartAttempts: 3, uninstallable: true, installable: true,
    error: 'It stopped 3 times in a row: ModuleNotFoundError: No module named \'pdfplumber\'',
    lastCrash: { code: 1, at: '2026-11-12T08:00:00Z', detail: 'ModuleNotFoundError: No module named \'pdfplumber\'' },
  };

  it('heads each group with its name in words, one group whatever case the category is written in', async () => {
    listMock.mockResolvedValue({
      ...SNAPSHOT,
      services: [service({ id: 'a', name: 'A', category: 'speech' }), service({ id: 'b', name: 'B', category: 'Speech' }), service({ id: 'c', name: 'C', category: 'custom' }), service({ id: 'd', name: 'D', category: 'automation' }), service({ id: 'e', name: 'E', category: 'weather' })],
    });
    await mount();
    expect(Array.from(host.querySelectorAll('.service-section > h3.section-title')).map((h) => h.textContent)).toEqual(['Automation', 'Speech', 'Weather', 'Your own', 'Add a service']);
    expect(categoryName('custom')).toBe('Your own');
    expect(categoryName('LLM')).toBe('Language models');
    expect(categoryName('')).toBe('Other');
  });

  it('keeps Start and Logs on a stopped service’s row, with Uninstall, Export and Remove under More', async () => {
    listMock.mockResolvedValue({ ...SNAPSHOT, services: [service({ uninstallable: true, installable: true })] });
    await mount();
    expect(rowButtons()).toEqual(['Start', 'Logs', 'More for Playwright Browser']);
    expect(host.querySelector('.service-actions .btn-danger')).toBeNull();
    const more = host.querySelector<HTMLButtonElement>('.more-button')!;
    expect(more.getAttribute('aria-haspopup')).toBe('menu');
    expect(more.getAttribute('aria-expanded')).toBe('false');
    await act(async () => more.click());
    expect(more.getAttribute('aria-expanded')).toBe('true');
    expect(menuItems().map((b) => b.firstElementChild?.nextElementSibling?.firstChild?.textContent ?? b.textContent)).toEqual(['Uninstall', 'Export', 'Remove']);
    expect(menuItems()[2].classList.contains('is-danger')).toBe(true);
    // Escape closes it and gives the button the keyboard back.
    await act(async () => menuItems()[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(menuItems()).toEqual([]);
    expect(document.activeElement).toBe(more);
    // Export asks the desktop for the package.
    const { services: api } = await import('./api');
    vi.mocked(api.export).mockRejectedValue(new Error('not in this test'));
    await act(async () => more.click());
    await act(async () => menuItems()[1].click());
    expect(api.export).toHaveBeenCalledWith('playwright-browser');
    expect(menuItems()).toEqual([]);
  });

  it('offers Stop for a running service, and Remove under More only once it is stopped', async () => {
    listMock.mockResolvedValue({ ...SNAPSHOT, services: [service({ status: 'running', uninstallable: true })] });
    await mount();
    expect(rowButtons()).toEqual(['Stop', 'Logs', 'More for Playwright Browser']);
    await act(async () => host.querySelector<HTMLButtonElement>('.more-button')!.click());
    const remove = menuItems().find((b) => b.textContent?.startsWith('Remove'))!;
    expect(remove.disabled).toBe(true);
    expect(remove.textContent).toContain('Stop it first');
    expect(menuItems().some((b) => b.textContent?.startsWith('Uninstall'))).toBe(false);
  });

  it('says what went wrong once, then what OAIY did about it and the way on', async () => {
    listMock.mockResolvedValue({ ...SNAPSHOT, services: [service(gaveUp)] });
    await mount();
    const card = host.querySelector('.service-card')!;
    expect(card.textContent!.split('No module named').length - 1).toBe(1);
    expect(card.querySelector('.service-crash')!.textContent).toBe('It stopped 3 times and OAIY gave up. Repair starts it afresh.');
    expect(card.textContent).not.toMatch(/Exited with code|restart attempt|needs Repair —/);
    expect(rowButtons()).toEqual(['Start', 'Repair', 'Logs', 'More for Invoice reader']);
    // What it said as it died is added only where the error does not carry it already; a service still being
    // started again says so, with its exit code.
    expect(crashWords({ status: 'errored', error: 'It would not start', lastCrash: { code: 2, at: '', detail: 'port 8801 is taken' }, restartAttempts: 1, needsRepair: false })).toEqual({
      cause: 'It would not start',
      after: 'It stopped (exit code 2), and OAIY is starting it again: try 1. It said: port 8801 is taken',
    });
    expect(crashWords({ status: 'running', error: null, lastCrash: { code: 1, at: '', detail: 'old news' }, restartAttempts: 0, needsRepair: false })).toEqual({ cause: null, after: null });
    expect(crashWords({ status: 'stopped', error: null, lastCrash: null, restartAttempts: 0, needsRepair: false })).toEqual({ cause: null, after: null });
  });

  it('puts where services are kept at the foot, closed, beside Add custom service and Import package', async () => {
    await mount();
    const sections = Array.from(host.querySelectorAll('.panel > section'));
    const foot = sections[sections.length - 1];
    expect(foot.querySelector('h3')!.textContent).toBe('Add a service');
    expect(Array.from(foot.querySelectorAll('.form-actions > button')).map((b) => b.textContent?.trim())).toEqual(['Add custom service', 'Import package']);
    const authors = foot.querySelector<HTMLDetailsElement>('details.services-authors')!;
    expect(authors.open).toBe(false);
    expect(authors.querySelector('summary')!.textContent).toBe('For service authors');
    expect(authors.textContent).toContain('C:\\data');
    // Nothing for a package's author stands before the services themselves.
    expect(host.querySelector('.panel')!.firstElementChild!.textContent).not.toContain('templates');
    expect(host.querySelector('.panel > .datadir-note')).toBeNull();
  });
});
