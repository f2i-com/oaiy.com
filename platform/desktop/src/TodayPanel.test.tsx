// Today's tiles follow the desktop's modules: the phone's tile (and its
// phone.status poll, a command to the Aokie plugin) only while a plugin
// provides the phone, the calendar's tiles and polls only while one provides
// the calendar. Same convention as OverviewPanel.test.tsx: react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { modulesMock, phoneStatus, phoneCalls, calendarGet, calendarSync, networkStatus, enginesStatus, enginesCatalog } = vi.hoisted(() => ({
  networkStatus: vi.fn(),
  enginesStatus: vi.fn(),
  enginesCatalog: vi.fn(),
  modulesMock: vi.fn(),
  phoneStatus: vi.fn(),
  phoneCalls: vi.fn(),
  calendarGet: vi.fn(),
  calendarSync: vi.fn(),
}));

vi.mock('./api', () => ({
  modules: { list: (...a: unknown[]) => modulesMock(...a) },
  phone: { status: (...a: unknown[]) => phoneStatus(...a), calls: (...a: unknown[]) => phoneCalls(...a) },
  calendar: { get: (...a: unknown[]) => calendarGet(...a), syncStatus: (...a: unknown[]) => calendarSync(...a) },
  engines: { status: (...a: unknown[]) => enginesStatus(...a), catalog: (...a: unknown[]) => enginesCatalog(...a) },
  network: { status: (...a: unknown[]) => networkStatus(...a) },
  link: { status: vi.fn().mockResolvedValue({ linked: false, attempt: { phase: 'idle' }, available: [] }) },
}));

import TodayPanel from './TodayPanel';
import { resetModules } from './useModules';

const snapshot = (phone: boolean, calendar: boolean) => ({
  snapshot: {
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: phone, builtin: true, provider: null, ...(phone ? {} : { reason: 'Aokie Phone Bridge is turned off in Plugins.' }) },
      { id: 'calendar', name: 'Calendar', enabled: calendar, builtin: true, provider: null },
    ],
    contributions: {},
    warnings: [],
  },
  etag: '"1-x"',
});

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';

async function mount() {
  await act(async () => {
    root.render(<TodayPanel onNavigate={vi.fn()} />);
  });
  // The modules land, then the tiles ask again with them.
  await act(async () => {});
}

beforeEach(() => {
  vi.clearAllMocks();
  resetModules();
  phoneStatus.mockResolvedValue({ connected: true });
  phoneCalls.mockResolvedValue([]);
  calendarGet.mockResolvedValue({ available: true, settings: {}, appointments: [], now: '' });
  calendarSync.mockResolvedValue({ linked: false, state: 'unlinked' });
  networkStatus.mockRejectedValue(new Error('no network'));
  enginesStatus.mockResolvedValue({ running: false });
  enginesCatalog.mockResolvedValue({ running: true, models: [] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe("Today's tiles and the desktop's modules", () => {
  it('with the phone off there is no phone tile, and the phone is never asked', async () => {
    modulesMock.mockResolvedValue(snapshot(false, true));
    await mount();
    expect(text()).not.toContain('The phone');
    expect(phoneStatus).not.toHaveBeenCalled();
    expect(phoneCalls).not.toHaveBeenCalled();
    // The calendar is its own module: still there.
    expect(text()).toContain('Next appointment');
    expect(calendarGet).toHaveBeenCalled();
  });

  it('with the calendar off there are no calendar tiles, and the calendar is never asked', async () => {
    modulesMock.mockResolvedValue(snapshot(true, false));
    await mount();
    expect(text()).toContain('The phone');
    expect(text()).toContain('Connected');
    expect(text()).not.toContain('Next appointment');
    expect(text()).not.toContain('Requests to confirm');
    expect(calendarGet).not.toHaveBeenCalled();
    expect(calendarSync).not.toHaveBeenCalled();
  });

  it('until the desktop has said, neither is shown nor asked', async () => {
    modulesMock.mockReturnValue(new Promise(() => {}));
    await mount();
    expect(text()).not.toContain('The phone');
    expect(text()).not.toContain('Next appointment');
    expect(phoneStatus).not.toHaveBeenCalled();
    expect(calendarGet).not.toHaveBeenCalled();
    expect(text()).toContain('Language model');
  });
});

describe('the language model’s tile', () => {
  const desktop = { id: 'engine:llm:node.w1/qwen3.8-flash-next', nodeId: 'w1', nodeName: 'Desktop', model: 'qwen3.8-flash-next' };
  const tile = () => Array.from(host.querySelectorAll('.overview-tile')).find((t) => t.textContent?.includes('qwen') || t.textContent?.includes('minicpm'))!;
  beforeEach(() => enginesStatus.mockResolvedValue({ running: true, llm: { state: 'ready', resident: 'minicpm-v-4.7' } }));

  it('names this computer’s model where the engine’s default is its own', async () => {
    modulesMock.mockResolvedValue(snapshot(false, false));
    await mount();
    expect(tile().textContent).toBe('Model readyminicpm-v-4.7');
  });

  it('names the paired computer and its model where the engine’s default is a paired computer’s, and opens Other computers', async () => {
    modulesMock.mockResolvedValue(snapshot(false, false));
    networkStatus.mockResolvedValue({ enabled: true, uses: true, defaults: { llm: { ...desktop, inEffect: true, waiting: null } } });
    const onNavigate = vi.fn();
    await act(async () => {
      root.render(<TodayPanel onNavigate={onNavigate} />);
    });
    await act(async () => {});
    expect(tile().textContent).toBe('On Desktopqwen3.8-flash-next · paired computer');
    await act(async () => (tile() as HTMLButtonElement).click());
    expect(onNavigate).toHaveBeenCalledWith('engine-workers');
  });

  it('names the model the paired computer runs now, as the desktop says it (`model`), not one its id names', async () => {
    modulesMock.mockResolvedValue(snapshot(false, false));
    networkStatus.mockResolvedValue({
      enabled: true,
      uses: true,
      defaults: { llm: { ...desktop, id: 'engine:llm:node.w1/qwen3.5-9b', insteadOf: 'minicpm-v-4.7', model: 'qwen3.8-flash-next', inEffect: true, waiting: null } },
    });
    await mount();
    expect(tile().textContent).toBe('On Desktopqwen3.8-flash-next · paired computer');
    expect(tile().textContent).not.toContain('qwen3.5-9b');
  });

  it('does not call a paired default ready while its computer’s engines are not answering', async () => {
    modulesMock.mockResolvedValue(snapshot(false, false));
    networkStatus.mockResolvedValue({ enabled: true, uses: true, defaults: { llm: { ...desktop, inEffect: true, waiting: 'engines_down' } } });
    await mount();
    expect(tile().textContent).toBe('Not answeringqwen3.8-flash-next · on Desktop');
    expect(tile().querySelector('strong')?.className).toBe('warn');
    expect(tile().getAttribute('title')).toContain('Desktop’s engines aren’t answering');
  });

  it('names a model by its name in the engines’ catalogue, read once and again only when the model changes', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      modulesMock.mockResolvedValue(snapshot(false, false));
      enginesStatus.mockResolvedValue({ running: true, llm: { state: 'ready', resident: 'minicpm-v-4.7' } });
      enginesCatalog.mockResolvedValue({ running: true, models: [{ id: 'minicpm-v-4.7', name: 'MiniCPM-V 4.7' }, { id: 'qwen3.5-9b', name: 'Qwen3.5 9B' }] });
      await mount();
      await act(async () => {});
      const llmTile = () => Array.from(host.querySelectorAll('.overview-tile')).find((t) => t.textContent?.includes('Model ready'))!;
      expect(llmTile().textContent).toBe('Model readyMiniCPM-V 4.7');
      const asked = enginesCatalog.mock.calls.length;
      expect(asked).toBeGreaterThan(0);
      // The next poll, the same model: the catalogue is not asked again.
      await act(async () => {
        vi.advanceTimersByTime(10_000);
      });
      await act(async () => {});
      expect(enginesCatalog.mock.calls.length).toBe(asked);
      // Another model: asked again, and named.
      enginesStatus.mockResolvedValue({ running: true, llm: { state: 'ready', resident: 'qwen3.5-9b' } });
      await act(async () => {
        vi.advanceTimersByTime(10_000);
      });
      await act(async () => {});
      expect(enginesCatalog.mock.calls.length).toBe(asked + 1);
      expect(llmTile().textContent).toBe('Model readyQwen3.5 9B');
    } finally {
      vi.useRealTimers();
    }
  });

  it('names this computer’s own while the paired computer’s is not in effect', async () => {
    modulesMock.mockResolvedValue(snapshot(false, false));
    networkStatus.mockResolvedValue({ enabled: true, uses: true, defaults: { llm: { ...desktop, inEffect: false, waiting: 'not_heard' } } });
    await mount();
    expect(tile().textContent).toBe('Model readyminicpm-v-4.7');
  });
});
