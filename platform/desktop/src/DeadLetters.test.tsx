// Dead-letter queue. The behaviours worth pinning are the ones that decide
// whether someone trusts this list: it must vanish when empty (an always-present
// "0 problems" box trains people to ignore it), a redrive that legitimately
// fails again must SAY so rather than look like an error, and a redrive that
// works must take the entry away so nobody redrives it again tomorrow.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listMock, redriveMock, dismissMock } = vi.hoisted(() => ({
  listMock: vi.fn(),
  redriveMock: vi.fn(),
  dismissMock: vi.fn(),
}));
vi.mock('./api', () => ({
  bridge: { deadLetters: listMock, redrive: redriveMock, dismissDeadLetter: dismissMock },
}));

import DeadLetters from './DeadLetters';

const SHED = {
  id: 'dl_1',
  source: 'aokie',
  event: 'aokie.call.incoming',
  reason: { kind: 'shed' as const },
  envelope: { name: 'aokie.call.incoming' },
  recordedAtMs: 1_785_000_000_000,
  attempts: 0,
};

const GUARDED = {
  id: 'dl_2',
  source: 'aokie',
  event: 'flow.succeeded',
  reason: { kind: 'not_reserved' as const, detail: 'b1: a loop guard refused it: depth 17' },
  envelope: { name: 'flow.succeeded' },
  recordedAtMs: 1_785_000_100_000,
  attempts: 2,
  lastOutcome: 'b1: skipped — a loop guard refused it: depth 17',
};

let host: HTMLDivElement;
let root: Root;

async function mount() {
  await act(async () => {
    root.render(<DeadLetters />);
  });
}

const text = () => host.textContent ?? '';
const button = (label: string) =>
  Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;
const click = async (el: Element) => {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
};

beforeEach(() => {
  listMock.mockReset();
  redriveMock.mockReset();
  dismissMock.mockReset();
  listMock.mockResolvedValue({ deadLetters: [SHED, GUARDED], total: 2 });
  redriveMock.mockResolvedValue({ reserved: true, outcomes: ['b1: reserved run_9'] });
  dismissMock.mockResolvedValue(undefined);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('DeadLetters', () => {
  it('renders nothing at all when the queue is empty', async () => {
    listMock.mockResolvedValue({ deadLetters: [], total: 0 });
    await mount();
    // A permanent "no problems" panel is noise that teaches people to skip it.
    expect(host.innerHTML).toBe('');
  });

  it('says which of the two ways an event was lost', async () => {
    await mount();
    expect(text()).toContain('dropped');
    expect(text()).toContain('queue was full');
    expect(text()).toContain('not run');
    expect(text()).toContain('a loop guard refused it: depth 17');
  });

  it('shows prior attempts, so a repeatedly-failing entry is obvious', async () => {
    await mount();
    expect(text()).toContain('2 redrives');
    expect(text()).toContain('Last redrive:');
  });

  it('refetches after a successful redrive, so the entry goes away', async () => {
    await mount();
    listMock.mockResolvedValue({ deadLetters: [GUARDED], total: 1 });
    await click(button('Redrive'));

    expect(redriveMock).toHaveBeenCalledWith('dl_1');
    // Names the event: the row it belonged to is gone, so the note has to
    // stand on its own.
    expect(text()).toContain('Redrive reserved a run for aokie.call.incoming.');
    // The backend drops a redriven entry; the panel must reflect that rather
    // than leaving a row someone redrives again tomorrow.
    expect(text()).not.toContain('queue was full');
  });

  it('reports a redrive that legitimately failed again, with the reason', async () => {
    redriveMock.mockResolvedValue({
      reserved: false,
      outcomes: ['b1: skipped — a loop guard refused it: depth 17'],
    });
    await mount();
    await click(button('Redrive'));
    // Not an error — the guard doing its job. Saying "failed" would send
    // someone hunting for a bug that isn't there.
    expect(text()).toContain('is still not running');
    expect(text()).toContain('loop guard refused');
  });

  it('surfaces a redrive that actually errored', async () => {
    redriveMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    await click(button('Redrive'));
    expect(text()).toContain('bridge is down');
  });

  it('dismisses an entry the user decides does not matter', async () => {
    await mount();
    listMock.mockResolvedValue({ deadLetters: [GUARDED], total: 1 });
    await click(button('Dismiss'));
    expect(dismissMock).toHaveBeenCalledWith('dl_1');
    expect(text()).not.toContain('queue was full');
  });
});
