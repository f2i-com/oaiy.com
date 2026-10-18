// Run history panel. The point of this screen is that a failure explains itself
// without anyone knowing a run id, so these pin: the failure-first default, the
// reason showing on the row rather than behind a click, and — the one that would
// be actively harmful to get wrong — an empty list saying whether nothing FAILED
// or nothing RAN.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { runsMock, deadMock, clearRunsMock, pushMock, runMock, cancelMock, againMock, flowsMock, pictureMock } = vi.hoisted(() => ({
  runsMock: vi.fn(),
  deadMock: vi.fn(),
  clearRunsMock: vi.fn(),
  pushMock: vi.fn(),
  runMock: vi.fn(),
  cancelMock: vi.fn(),
  againMock: vi.fn(),
  flowsMock: vi.fn(),
  pictureMock: vi.fn(),
}));
// The panel embeds DeadLetters, which polls on mount. Stubbed empty so these
// tests are about the run list; DeadLetters.test.tsx covers the queue itself.
vi.mock('./api', () => ({
  bridge: {
    runs: runsMock,
    clearRuns: clearRunsMock,
    run: runMock,
    cancelRun: cancelMock,
    runAgain: againMock,
    flows: flowsMock,
    deadLetters: deadMock,
    redrive: vi.fn(),
    dismissDeadLetter: vi.fn(),
  },
  network: { nodes: vi.fn().mockResolvedValue({ nodes: [] }), picture: pictureMock },
}));
vi.mock('./Toasts', () => ({ useToast: () => ({ push: pushMock }) }));

import RunsPanel from './RunsPanel';
import { invalidate, put } from './useCached';

const FAILED = {
  runId: 'run_a1',
  status: 'failed' as const,
  callerProduct: 'formlogic',
  flowId: 'caller-lookup',
  correlationId: 'call_abc',
  mode: 'async',
  runtime: 'desktop',
  error: {
    code: 'capability_unavailable',
    message: 'Ollama is not running.',
    detail: 'start it from Services',
    nodeId: 'node_3',
  },
  reservedAt: '2026-07-30T09:00:00.000Z',
  startedAt: '2026-07-30T09:00:01.000Z',
  finishedAt: '2026-07-30T09:00:04.500Z',
};

let host: HTMLDivElement;
let root: Root;

/** Render and let the initial fetch settle. */
async function mount() {
  await act(async () => {
    root.render(<RunsPanel />);
  });
}

const text = () => host.textContent ?? '';
const click = async (el: Element) => {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
};
const button = (label: string) =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;
/** The clear button is icon-only; its accessible name is the only handle. */
const clearButton = () =>
  host.querySelector<HTMLButtonElement>('button[aria-label="Clear finished runs"]')!;

beforeEach(() => {
  invalidate();
  localStorage.clear();
  runMock.mockReset();
  cancelMock.mockReset();
  againMock.mockReset();
  pictureMock.mockReset();
  pictureMock.mockResolvedValue(new Blob(['x']));
  flowsMock.mockReset();
  flowsMock.mockResolvedValue({ flows: [{ flowId: 'caller-lookup', name: 'Look up a caller' }] });
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: () => 'blob:picture' });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: () => {} });
  runsMock.mockReset();
  clearRunsMock.mockReset();
  clearRunsMock.mockResolvedValue({ cleared: 11, total: 1 });
  pushMock.mockReset();
  vi.stubGlobal('confirm', vi.fn().mockReturnValue(true));
  deadMock.mockReset();
  deadMock.mockResolvedValue({ deadLetters: [], total: 0 });
  runsMock.mockResolvedValue({ runs: [FAILED], total: 12, byStatus: { failed: 1, succeeded: 11 } });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('RunsPanel', () => {
  it('clears finished runs and refetches, so the list cannot show deleted rows', async () => {
    await mount();
    runsMock.mockResolvedValue({ runs: [], total: 1, byStatus: {} });
    await click(clearButton());

    expect(clearRunsMock).toHaveBeenCalled();
    // A refetch after the delete, not an optimistic splice: the host decides
    // what survived (it keeps work still in flight), so only it can say.
    expect(runsMock.mock.calls.length).toBeGreaterThan(1);
    expect(pushMock).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', title: 'Cleared 11 finished runs' }),
    );
  });

  it('warns that a retried request will run again, not just that rows disappear', async () => {
    // The non-obvious consequence: clearing drops idempotency keys, so an app
    // retrying a request from before the clear gets a fresh run instead of the
    // recorded result. Rows vanishing is the part users already expect.
    await mount();
    await click(clearButton());
    const message = (globalThis.confirm as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(message).toContain('run it again');
    expect(message).toContain('Queued and running work is kept');
  });

  it('does not clear when the confirm is declined', async () => {
    vi.stubGlobal('confirm', vi.fn().mockReturnValue(false));
    await mount();
    await click(clearButton());
    expect(clearRunsMock).not.toHaveBeenCalled();
  });

  it('says so plainly when there was nothing to clear', async () => {
    // "Cleared 0 finished runs" reads like a bug. It is a real outcome: every
    // remaining run is still in flight.
    clearRunsMock.mockResolvedValue({ cleared: 0, total: 3 });
    await mount();
    await click(clearButton());
    expect(pushMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Nothing to clear' }),
    );
  });

  it('surfaces a failed clear instead of pretending it worked', async () => {
    clearRunsMock.mockRejectedValue(new Error('403: origin not allowed'));
    await mount();
    await click(clearButton());
    expect(text()).toContain('origin not allowed');
  });

  it('disables the button when there is nothing recorded at all', async () => {
    runsMock.mockResolvedValue({ runs: [], total: 0, byStatus: {} });
    await mount();
    expect(clearButton().disabled).toBe(true);
  });

  it('opens on failures, because a list of successes buries the one row you came for', async () => {
    await mount();
    expect(runsMock).toHaveBeenCalledWith(['failed', 'timed_out'], 50);
    expect(text()).toContain('Failed runs');
  });

  it('puts the reason on the row, not behind a click', async () => {
    await mount();
    expect(text()).toContain('capability_unavailable');
    expect(text()).toContain('Ollama is not running.');
    // The run id is detail, so it stays collapsed until asked for.
    expect(text()).not.toContain('run_a1');
  });

  it('offers no "Show all" when the reason fits in three lines', async () => {
    await mount();
    expect(host.querySelector('.run-error-text.clamped')!.textContent).toBe('Ollama is not running.');
    expect(text()).not.toContain('Show all');
  });

  it('holds a long reason to three lines, with the rest a click away', async () => {
    // jsdom has no layout, so stand in for it: the clamped text is taller
    // inside than the three lines it shows.
    const long = `Unknown node types: ${Array.from({ length: 60 }, (_, i) => `node_type_${i}`).join(', ')}.`;
    runsMock.mockResolvedValue({
      runs: [{ ...FAILED, error: { ...FAILED.error, message: long } }],
      total: 12,
      byStatus: { failed: 1, succeeded: 11 },
    });
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
      configurable: true,
      get(this: HTMLElement) { return this.classList.contains('run-error-text') ? 400 : 0; },
    });
    Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
      configurable: true,
      get(this: HTMLElement) { return this.classList.contains('clamped') ? 57 : this.classList.contains('run-error-text') ? 400 : 0; },
    });
    try {
      await mount();
      const message = host.querySelector('.run-error-text')!;
      expect(message.classList.contains('clamped')).toBe(true);
      expect(message.getAttribute('title')).toBe(long);

      await click(button('Show all'));
      expect(message.classList.contains('clamped')).toBe(false);
      expect(message.textContent).toBe(long);
      expect(button('Show less').getAttribute('aria-expanded')).toBe('true');

      await click(button('Show less'));
      expect(message.classList.contains('clamped')).toBe(true);
      expect(button('Show all').getAttribute('aria-expanded')).toBe('false');
    } finally {
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollHeight;
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientHeight;
    }
  });

  it('expands to the ids and node a support conversation needs', async () => {
    await mount();
    await click(host.querySelector('.run-head')!);
    expect(text()).toContain('run_a1');
    expect(text()).toContain('call_abc');
    expect(text()).toContain('node_3');
    expect(text()).toContain('start it from Services');
  });

  it('shows how long the run took, from the timestamps already recorded', async () => {
    await mount();
    expect(text()).toContain('3.5 s');
  });

  it('refetches with no filter when switching to All', async () => {
    await mount();
    await click(button('All'));
    expect(runsMock).toHaveBeenLastCalledWith('all', 50);
  });

  it('counts the states the failed filter fetches, the ones Overview warns of; a cancelled run is not a failure', async () => {
    // The badge was summed from a hand-written subset while the query used
    // FAILURE_STATES, so the tab said "Failures (2)" above five rows. Both now
    // come from one constant, and it is the count Overview's warning gives: a
    // cancelled run (someone stopped it) is under All.
    runsMock.mockResolvedValue({
      runs: [FAILED],
      total: 12,
      byStatus: { failed: 1, timed_out: 1, cancelled: 3, succeeded: 7 },
    });
    await mount();
    expect(text()).toContain('Failed (2)');
    expect(runsMock).toHaveBeenCalledWith(['failed', 'timed_out'], 50);
    // Looked at here: Overview warns only of failures after these (by when the newest ended, and how many).
    expect(JSON.parse(localStorage.getItem('oaiy.overview.failedSeen')!)).toEqual({ count: 2, at: FAILED.finishedAt });
  });

  it('names a run’s flow by its name, and who asked for it in words', async () => {
    runsMock.mockResolvedValue({
      runs: [FAILED, { ...FAILED, runId: 'run_b', callerProduct: 'oaiy-trigger:aokie', triggerEvent: 'aokie.sms.received' }, { ...FAILED, runId: 'run_c', callerProduct: 'oaiy-app' }],
      total: 3,
      byStatus: { failed: 3 },
    });
    await mount();
    expect(text()).toContain('Look up a caller');
    expect(text()).toContain('FormLogic');
    expect(text()).toContain('On aokie.sms.received');
    expect(text()).toContain('The Agent');
  });

  it('shows what a run was asked with and what it answered, reading its pictures at this desktop alone', async () => {
    const picture = 'http://127.0.0.1:17999/api/ai/engine/gateway/files/images/fox.png';
    runsMock.mockResolvedValue({
      runs: [{
        ...FAILED,
        status: 'succeeded',
        error: undefined,
        input: { Prompt: 'a fox in a circle' },
        output: { engine: 'zipp', output: { image: picture, elsewhere: 'https://example.com/tracker.png', caption: 'A fox, drawn in one line' }, results: { make: 'done' } },
      }],
      total: 1,
      byStatus: { succeeded: 1 },
    });
    await mount();
    await click(button('All'));
    await click(host.querySelector('.run-head')!);
    expect(text()).toContain('a fox in a circle');
    expect(text()).toContain('A fox, drawn in one line');
    expect(text()).toContain('Each step’s result');
    expect(pictureMock).toHaveBeenCalledWith(picture);
    expect(pictureMock).not.toHaveBeenCalledWith('https://example.com/tracker.png');
    expect(host.querySelector('img')!.getAttribute('src')).toBe('blob:picture');
  });

  it('runs a finished run again with its inputs, once asked, and says how it went', async () => {
    vi.useFakeTimers();
    try {
      runsMock.mockResolvedValue({ runs: [{ ...FAILED, input: { Name: 'Sam' }, timeoutMs: 30000 }], total: 1, byStatus: { failed: 1 } });
      againMock.mockResolvedValue({ ...FAILED, runId: 'run_new', status: 'queued', error: undefined });
      runMock.mockResolvedValue({ ...FAILED, runId: 'run_new', status: 'succeeded', error: undefined });
      await mount();
      await click(host.querySelector('.run-head')!);
      vi.stubGlobal('confirm', vi.fn().mockReturnValueOnce(false));
      await click(button('Run again'));
      expect(againMock).not.toHaveBeenCalled();
      const asked = vi.fn().mockReturnValue(true);
      vi.stubGlobal('confirm', asked);
      await click(button('Run again'));
      // The run, by its id: the desktop takes its inputs, time limit and confinement from its record, not from here.
      expect(againMock).toHaveBeenCalledWith(expect.objectContaining({ runId: 'run_a1', flowId: 'caller-lookup' }));
      // An app's run had outside input: run again, it is held in as it was, and the question says so.
      expect(String(asked.mock.calls[0][0])).toContain('held in as the first was');
      expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ title: 'Running Look up a caller again' }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1600);
      });
      expect(runMock).toHaveBeenCalledWith('run_new');
      expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Look up a caller ran again and succeeded' }));
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not offer to run again a run whose flow is no longer here', async () => {
    flowsMock.mockResolvedValue({ flows: [] });
    await mount();
    await click(host.querySelector('.run-head')!);
    expect(button('Run again').disabled).toBe(true);
    expect(text()).toContain('cannot be run again');
  });

  it('cancels a queued run once asked, and stops a running one', async () => {
    runsMock.mockResolvedValue({
      runs: [
        { ...FAILED, runId: 'run_q', status: 'queued', error: undefined, startedAt: undefined, finishedAt: undefined },
        { ...FAILED, runId: 'run_r', status: 'running', error: undefined, finishedAt: undefined },
      ],
      total: 2,
      byStatus: { queued: 1, running: 1 },
    });
    cancelMock.mockResolvedValueOnce({ runId: 'run_q', status: 'cancelled' }).mockResolvedValueOnce({ runId: 'run_r', status: 'running', cancelRequested: true });
    await mount();
    await click(button('In progress'));
    await click(button('Cancel'));
    expect(cancelMock).toHaveBeenCalledWith('run_q');
    expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ title: 'Cancelled Look up a caller' }));
    await click(button('Stop'));
    expect(cancelMock).toHaveBeenCalledWith('run_r');
    expect(pushMock).toHaveBeenCalledWith(expect.objectContaining({ title: 'Asked Look up a caller to stop' }));
  });

  it('asks the desktop to search, over every run it keeps, once the typing rests', async () => {
    const printer = { ...FAILED, runId: 'run_b', flowId: 'other', callerProduct: 'oaiy-app', error: { code: 'node_failed', message: 'The printer is out of paper.' } };
    runsMock.mockImplementation(async (_states: unknown, _limit: unknown, page?: { q?: string }) =>
      page?.q === 'printer'
        ? { runs: [printer], total: 900, byStatus: { failed: 2 }, more: false }
        : page?.q
          ? { runs: [], total: 900, byStatus: { failed: 2 }, more: false }
          : { runs: [FAILED, printer], total: 900, byStatus: { failed: 2 } },
    );
    await mount();
    const search = host.querySelector<HTMLInputElement>('input[type="search"]')!;
    const type = async (value: string) => {
      await act(async () => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(search, value);
        search.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 350));
      });
    };
    await type('printer');
    expect(runsMock).toHaveBeenLastCalledWith(['failed', 'timed_out'], 50, { q: 'printer' });
    expect(host.querySelectorAll('.run-row')).toHaveLength(1);
    expect(text()).toContain('The printer is out of paper.');
    expect(text()).toContain('1 failed run matches.');
    await type('nothing like this');
    expect(text()).toContain('No failed run matches “nothing like this”');
    await click(button('Clear the search'));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(host.querySelectorAll('.run-row')).toHaveLength(2);
    expect(runsMock).toHaveBeenLastCalledWith(['failed', 'timed_out'], 50);
  });

  it('pages back past the 200 newest, each older page from the last run shown', async () => {
    const run = (i: number) => ({ ...FAILED, runId: `run_${String(i).padStart(4, '0')}`, reservedAt: new Date(Date.UTC(2026, 6, 30, 9, 0, 0) - i * 1000).toISOString() });
    const all = Array.from({ length: 260 }, (_, i) => run(i));
    runsMock.mockImplementation(async (_states: unknown, limit: number, page?: { before?: { runId: string } }) => {
      const from = page?.before ? all.findIndex((r) => r.runId === page.before!.runId) + 1 : 0;
      const runs = all.slice(from, from + limit);
      return { runs, total: 260, byStatus: { failed: 260 }, more: from + limit < all.length };
    });
    await mount();
    for (let shown = 50; shown < 200; shown += 50) {
      expect(host.querySelectorAll('.run-row')).toHaveLength(shown);
      await click(button('Show 50 more'));
    }
    expect(host.querySelectorAll('.run-row')).toHaveLength(200);
    expect(runsMock).toHaveBeenLastCalledWith(['failed', 'timed_out'], 200);
    // Past what one answer holds: from the last run shown.
    await click(button('Show 50 more'));
    expect(runsMock).toHaveBeenLastCalledWith(['failed', 'timed_out'], 50, { before: { reservedAt: all[199].reservedAt, runId: all[199].runId } });
    expect(host.querySelectorAll('.run-row')).toHaveLength(250);
    await click(button('Show 10 more'));
    expect(host.querySelectorAll('.run-row')).toHaveLength(260);
    expect(new Set([...host.querySelectorAll('[data-run-id]')].map((el) => el.getAttribute('data-run-id'))).size).toBe(260);
    expect(text()).not.toContain('more');
  });

  it('keeps the rows newer runs push out of the newest page, once older pages are read', async () => {
    const run = (i: number) => ({ ...FAILED, runId: `run_${String(i + 10).padStart(4, '0')}`, reservedAt: new Date(Date.UTC(2026, 6, 30, 9, 0, 0) - i * 1000).toISOString() });
    let all = Array.from({ length: 260 }, (_, i) => run(i));
    runsMock.mockImplementation(async (_states: unknown, limit: number, page?: { before?: { runId: string } }) => {
      const from = page?.before ? all.findIndex((r) => r.runId === page.before!.runId) + 1 : 0;
      return { runs: all.slice(from, from + limit), total: all.length, byStatus: { failed: all.length }, more: from + limit < all.length };
    });
    await mount();
    for (let shown = 50; shown <= 200; shown += 50) await click(button('Show 50 more'));
    expect(host.querySelectorAll('.run-row')).toHaveLength(250);
    // Five runs fail meanwhile: the newest page's five oldest are pushed past its end, and newer than the older pages.
    all = [...Array.from({ length: 5 }, (_, i) => run(-1 - i)), ...all];
    await click(host.querySelector('button[aria-label="Refresh now"]')!);
    const ids = [...host.querySelectorAll('[data-run-id]')].map((el) => el.getAttribute('data-run-id'));
    expect(ids).toHaveLength(255);
    expect(new Set(ids).size).toBe(255);
    for (const i of [195, 196, 197, 198, 199]) expect(ids).toContain(run(i).runId);
    // In history's order still.
    expect(ids).toEqual([...all.slice(0, 255)].map((r) => r.runId));
  });

  it('puts aside an older page that lands after a switch to another filter', async () => {
    const run = (i: number, status: 'succeeded' | 'failed') => ({
      ...FAILED,
      runId: `run_${status}_${String(i).padStart(4, '0')}`,
      status,
      error: status === 'failed' ? FAILED.error : undefined,
      reservedAt: new Date(Date.UTC(2026, 6, 30, 9, 0, 0) - i * 1000).toISOString(),
    });
    const ok = Array.from({ length: 260 }, (_, i) => run(i, 'succeeded'));
    let land: (v: unknown) => void = () => {};
    runsMock.mockImplementation((states: unknown, limit: number, page?: { before?: unknown }) => {
      if (states !== 'all') return Promise.resolve({ runs: [run(0, 'failed')], total: 261, byStatus: { failed: 1, succeeded: 260 } });
      if (page?.before) return new Promise((r) => (land = r));
      return Promise.resolve({ runs: ok.slice(0, limit), total: 261, byStatus: { failed: 1, succeeded: 260 }, more: limit < 260 });
    });
    await mount();
    await click(button('All'));
    for (let shown = 50; shown < 200; shown += 50) await click(button('Show 50 more'));
    expect(host.querySelectorAll('.run-row')).toHaveLength(200);
    // Older succeeded runs are asked for; before they come, the person switches to Failed.
    await click(button('Show 50 more'));
    await click(button('Failed'));
    await act(async () => land({ runs: ok.slice(200, 250), total: 261, byStatus: { failed: 1, succeeded: 260 }, more: true }));
    const ids = [...host.querySelectorAll('[data-run-id]')].map((el) => el.getAttribute('data-run-id'));
    expect(ids).toEqual(['run_failed_0000']);
  });

  it('says who asked in the caller’s own words, and opens that run’s flow in the editor', async () => {
    const opened = vi.fn();
    runsMock.mockResolvedValue({ runs: [{ ...FAILED, caller: { product: 'formlogic', label: 'acme reception' } }], total: 1, byStatus: { failed: 1 } });
    await act(async () => {
      root.render(<RunsPanel onOpenEditor={opened} />);
    });
    expect(text()).toContain('Acme reception');
    await click(host.querySelector('.run-head')!);
    expect(text()).toContain('formlogic');
    await click(button('Open the editor'));
    expect(opened).toHaveBeenCalledWith('caller-lookup');
  });

  it('says which flow in the picker is a macro', async () => {
    flowsMock.mockResolvedValue({ flows: [{ flowId: 'caller-lookup', name: 'Look up a caller' }, { flowId: 'user-macro-1', name: 'Shout', macro: true }] });
    runsMock.mockResolvedValue({ runs: [FAILED, { ...FAILED, runId: 'run_m', flowId: 'user-macro-1' }], total: 2, byStatus: { failed: 2 } });
    await mount();
    const options = [...host.querySelectorAll('select.run-flow-pick option')].map((o) => o.textContent);
    expect(options).toEqual(['Every flow', 'Look up a caller', 'Shout (macro)']);
  });

  it('keeps a run open below after it leaves the filter, until it is closed', async () => {
    const going = { ...FAILED, runId: 'run_r', status: 'running' as const, error: undefined, finishedAt: undefined };
    runsMock.mockResolvedValue({ runs: [going], total: 1, byStatus: { running: 1 } });
    await mount();
    await click(button('In progress'));
    await click(host.querySelector('.run-head')!);
    runsMock.mockResolvedValue({ runs: [], total: 1, byStatus: { succeeded: 1 } });
    runMock.mockResolvedValue({ ...going, status: 'succeeded', finishedAt: '2026-07-30T09:00:04.500Z' });
    await click(host.querySelector('button[aria-label="Refresh now"]')!);
    await act(async () => {});
    expect(text()).toContain('No longer among the runs in progress');
    expect(text()).toContain('run_r');
    await click(host.querySelector('.run-head')!);
    expect(host.querySelectorAll('.run-row')).toHaveLength(0);
  });

  it('distinguishes "nothing failed" from "nothing ran"', async () => {
    runsMock.mockResolvedValue({ runs: [], total: 8, byStatus: { succeeded: 8 } });
    await mount();
    // Saying "no runs" here would be a lie — 8 ran and all of them worked.
    expect(text()).toContain('all 8 recorded runs finished cleanly');

    await act(async () => root.unmount());
    runsMock.mockResolvedValue({ runs: [], total: 0, byStatus: {} });
    root = createRoot(host);
    await mount();
    expect(text()).toContain('No flow has run on this machine yet');
  });

  it('does not call a queued run finished', async () => {
    // The count came from `total` — the whole ledger — so three runs stuck in
    // the queue read as "all 12 recorded runs finished cleanly", an all-clear
    // over work that had not started. Nothing ages a queued run out, so that
    // sentence stays wrong for as long as the runs are stuck.
    runsMock.mockResolvedValue({ runs: [], total: 12, byStatus: { succeeded: 9, queued: 3 } });
    await mount();
    expect(text()).toContain('9 finished cleanly, 3 still queued or running');
    expect(text()).not.toContain('all 12 recorded runs');
  });

  it('never paints one filter’s rows under the other filter’s heading', async () => {
    // Regression: both filters shared one cache key, so browsing All and then
    // revisiting the panel showed green `succeeded` rows beneath the heading
    // "Failed runs" — and if the refetch then failed, it stayed that way behind
    // an error banner that said nothing about the list being wrong.
    put('runHistory:all', {
      runs: [{ ...FAILED, runId: 'run_ok', status: 'succeeded', error: undefined }],
      total: 12,
      byStatus: { succeeded: 12 },
    });
    // The refetch never resolves, so whatever paints must come from the cache.
    runsMock.mockReturnValue(new Promise(() => {}));

    await mount();

    expect(text()).toContain('Failed runs');
    expect(text()).not.toContain('succeeded');
    expect(text()).not.toContain('run_ok');
  });

  it('says a run’s error once, not wrapped in "Error: Error:"', async () => {
    runsMock.mockResolvedValue({
      runs: [{ ...FAILED, error: { ...FAILED.error, message: 'Error: Error: Image Gen: HTTP 503 — Rig did not answer' } }],
      total: 1,
      byStatus: { failed: 1 },
    });
    await mount();
    expect(text()).toContain('Image Gen: HTTP 503 — Rig did not answer');
    expect(text()).not.toContain('Error: Image Gen');
  });

  it('shows the answer to the filter last asked for, an older answer landing late put aside', async () => {
    // The failures fetch answers only after the switch to All has been answered.
    let late: (v: unknown) => void = () => {};
    runsMock.mockReturnValueOnce(new Promise((r) => (late = r)));
    await mount();
    runsMock.mockResolvedValue({ runs: [{ ...FAILED, runId: 'run_ok', status: 'succeeded', error: undefined }], total: 12, byStatus: { succeeded: 12 } });
    await act(async () => button('All').click());
    await act(async () => {});
    expect(text()).toContain('All runs');
    await act(async () => late({ runs: [FAILED], total: 12, byStatus: { failed: 1 } }));
    expect(text()).not.toContain('Ollama is not running.');
  });

  it('surfaces a fetch failure instead of pretending there is no history', async () => {
    runsMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    expect(text()).toContain('bridge is down');
  });
});
