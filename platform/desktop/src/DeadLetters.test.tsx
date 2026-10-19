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

  it('says which of the two ways an event was lost, in plain words, the codes kept for its details', async () => {
    await mount();
    expect(text()).toContain('Events that did not start a flow (2)');
    expect(text()).toContain('set aside');
    expect(text()).toContain('Too many events came at once');
    expect(text()).toContain('no flow ran');
    // What happened, in words: not the event's name, and none of the jargon the list used to lead with.
    expect(text()).toContain('A call came in');
    expect(text()).toContain('A flow finished');
    for (const word of ['aokie.call.incoming', 'redrive', 'Redrive', 'binding', 'loop guard', 'dispatch']) expect(text()).not.toContain(word);
    // Opened, an entry has the event's name and why OAIY recorded it.
    await click(host.querySelectorAll('.run-head')[1]);
    expect(text()).toContain('flow.succeeded');
    expect(text()).toContain('a loop guard refused it: depth 17');
  });

  it('shows prior attempts, so a repeatedly-failing entry is obvious, and what the last one did in its details', async () => {
    await mount();
    expect(text()).toContain('tried again 2 times');
    expect(text()).not.toContain('skipped — a loop guard');
    await click(host.querySelectorAll('.run-head')[1]);
    expect(text()).toContain('What that did');
    expect(text()).toContain('b1: skipped — a loop guard refused it: depth 17');
  });

  it('refetches after a successful redrive, so the entry goes away', async () => {
    await mount();
    listMock.mockResolvedValue({ deadLetters: [GUARDED], total: 1 });
    await click(button('Try again'));

    expect(redriveMock).toHaveBeenCalledWith('dl_1');
    // Names the event: the row it belonged to is gone, so the note has to
    // stand on its own.
    expect(text()).toContain('“A call came in” started a flow');
    // The backend drops a redriven entry; the panel must reflect that rather
    // than leaving a row someone redrives again tomorrow.
    expect(text()).not.toContain('Too many events came at once');
  });

  it('reports a redrive that legitimately failed again, with the reason', async () => {
    redriveMock.mockResolvedValue({
      reserved: false,
      outcomes: ['b1: skipped — a loop guard refused it: depth 17'],
    });
    await mount();
    await click(button('Try again'));
    // Not an error — the guard doing its job. Saying "failed" would send
    // someone hunting for a bug that isn't there.
    expect(text()).toContain('still did not start a flow');
    expect(text()).toContain('loop guard refused');
  });

  it('surfaces a redrive that actually errored', async () => {
    redriveMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    await click(button('Try again'));
    expect(text()).toContain('bridge is down');
  });

  it('dismisses an entry the user decides does not matter, once they say so', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
    await mount();
    await click(button('Dismiss'));
    expect(dismissMock).not.toHaveBeenCalled();
    listMock.mockResolvedValue({ deadLetters: [GUARDED], total: 1 });
    await click(button('Dismiss'));
    expect(dismissMock).toHaveBeenCalledWith('dl_1');
    expect(text()).not.toContain('Too many events came at once');
    confirm.mockRestore();
  });

  it('keeps an entry waiting on its own redrive while another is dismissed, and counts all of them', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    listMock.mockResolvedValue({ deadLetters: [SHED, GUARDED], total: 140 });
    let release!: (v: { reserved: boolean; outcomes: string[] }) => void;
    redriveMock.mockReturnValue(new Promise((r) => (release = r)));
    await mount();
    // The newest hundred are listed; the heading says how many there are in all.
    expect(text()).toContain('Events that did not start a flow (2 of 140)');
    const rows = () => [...host.querySelectorAll('.run-row')];
    const inRow = (i: number, label: string) => [...rows()[i].querySelectorAll('button')].find((b) => b.textContent?.includes(label))!;
    await click(inRow(0, 'Try again'));
    await click(inRow(1, 'Dismiss'));
    // The second's answer frees the second alone: the first, still being redriven, cannot be pressed again.
    expect(inRow(0, 'Trying again…').disabled).toBe(true);
    await act(async () => release({ reserved: true, outcomes: [] }));
    expect(inRow(0, 'Try again').disabled).toBe(false);
    confirm.mockRestore();
  });

  it('does not put a redriven entry back when a list asked before the redrive is answered after it', async () => {
    let late: (v: unknown) => void = () => {};
    vi.useFakeTimers();
    try {
      await mount();
      // The next poll goes out, and is slow; the redrive and its own read answer first.
      listMock.mockReturnValueOnce(new Promise((r) => (late = r)));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(6100);
      });
      listMock.mockResolvedValue({ deadLetters: [GUARDED], total: 1 });
      await click(button('Try again'));
      expect(text()).not.toContain('Too many events came at once');
      await act(async () => late({ deadLetters: [SHED, GUARDED], total: 2 }));
      expect(text()).not.toContain('Too many events came at once');
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows the event as it arrived, to copy, when an entry is opened', async () => {
    await mount();
    expect(text()).not.toContain('The event as it arrived');
    await click(host.querySelector('.run-head')!);
    expect(text()).toContain('The event as it arrived');
    expect(text()).toContain('dl_1');
  });

  it('says a dismiss that failed, and keeps the last redrive said after its entry, the last, has gone', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    dismissMock.mockRejectedValue(new Error('bridge is down'));
    await mount();
    await click(button('Dismiss'));
    expect(text()).toContain('“A call came in” could not be dismissed: bridge is down');
    listMock.mockResolvedValue({ deadLetters: [], total: 0 });
    await click(button('Try again'));
    expect(text()).toContain('“A call came in” started a flow');
    await click(button('Close'));
    expect(host.innerHTML).toBe('');
    confirm.mockRestore();
  });
});
