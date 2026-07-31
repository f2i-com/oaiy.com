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

const { runsMock, deadMock, clearRunsMock, pushMock } = vi.hoisted(() => ({
  runsMock: vi.fn(),
  deadMock: vi.fn(),
  clearRunsMock: vi.fn(),
  pushMock: vi.fn(),
}));
// The panel embeds DeadLetters, which polls on mount. Stubbed empty so these
// tests are about the run list; DeadLetters.test.tsx covers the queue itself.
vi.mock('./api', () => ({
  bridge: {
    runs: runsMock,
    clearRuns: clearRunsMock,
    deadLetters: deadMock,
    redrive: vi.fn(),
    dismissDeadLetter: vi.fn(),
  },
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
    expect(runsMock).toHaveBeenCalledWith(['failed', 'timed_out', 'cancelled'], 50);
    expect(text()).toContain('Failed runs');
  });

  it('puts the reason on the row, not behind a click', async () => {
    await mount();
    expect(text()).toContain('capability_unavailable');
    expect(text()).toContain('Ollama is not running.');
    // The run id is detail, so it stays collapsed until asked for.
    expect(text()).not.toContain('run_a1');
  });

  it('expands to the ids and node a support conversation needs', async () => {
    await mount();
    await click(button('Details'));
    expect(text()).toContain('run_a1');
    expect(text()).toContain('call_abc');
    expect(text()).toContain('node_3');
    expect(text()).toContain('start it from Services');
  });

  it('shows how long the run took, from the timestamps already recorded', async () => {
    await mount();
    expect(text()).toContain('3.5s');
  });

  it('refetches with no filter when switching to All', async () => {
    await mount();
    await click(button('All'));
    expect(runsMock).toHaveBeenLastCalledWith('all', 50);
  });

  it('counts every state the failures filter fetches, cancelled included', async () => {
    // The badge was summed from a hand-written subset while the query used
    // FAILURE_STATES, so the tab said "Failures (2)" above five rows.
    runsMock.mockResolvedValue({
      runs: [FAILED],
      total: 12,
      byStatus: { failed: 1, timed_out: 1, cancelled: 3, succeeded: 7 },
    });
    await mount();
    expect(text()).toContain('Failures (5)');
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

  it('surfaces a fetch failure instead of pretending there is no history', async () => {
    runsMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    expect(text()).toContain('bridge is down');
  });
});
