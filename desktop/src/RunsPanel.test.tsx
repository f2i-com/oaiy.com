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

const { runsMock, deadMock } = vi.hoisted(() => ({ runsMock: vi.fn(), deadMock: vi.fn() }));
// The panel embeds DeadLetters, which polls on mount. Stubbed empty so these
// tests are about the run list; DeadLetters.test.tsx covers the queue itself.
vi.mock('./api', () => ({
  bridge: { runs: runsMock, deadLetters: deadMock, redrive: vi.fn(), dismissDeadLetter: vi.fn() },
}));

import RunsPanel from './RunsPanel';

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

beforeEach(() => {
  runsMock.mockReset();
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
});

describe('RunsPanel', () => {
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

  it('surfaces a fetch failure instead of pretending there is no history', async () => {
    runsMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    expect(text()).toContain('bridge is down');
  });
});
