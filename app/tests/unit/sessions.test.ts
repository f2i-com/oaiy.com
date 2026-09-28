import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { SessionInfo } from '../../src/vfs/projects';
import { DesktopEvents, Sessions, TEST_NUMBER, textMessage } from '../../src/sessions';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import type { MessageSettings } from '../../src/settings';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

/** A project's session storage, in memory. */
function fakeProject() {
  const chats = new Map<string, Turn[]>();
  let index: SessionInfo[] = [];
  return {
    chats,
    get index() {
      return index;
    },
    project: {
      loadSessions: async () => index,
      saveSessions: async (list: SessionInfo[]) => {
        index = list;
      },
      loadSessionChat: async (id: string) => chats.get(id) ?? [],
      saveSessionChat: async (id: string, turns: Turn[]) => {
        chats.set(id, turns);
      },
    },
  };
}

/** A desktop that records the commands it is given. */
function fakeDesktop() {
  const commands: Array<{ connector: string; command: string; payload: Record<string, unknown> }> = [];
  const desktop = {
    commands,
    command: async (connector: string, command: string, payload: Record<string, unknown>) => {
      commands.push({ connector, command, payload });
      return { messageId: `m${commands.length}` };
    },
  };
  return desktop;
}

function setup(partial: Partial<MessageSettings> & { answer: boolean; instructions: string }, desktop = fakeDesktop()) {
  // The same object the test may change later.
  const messages = Object.assign(partial, { calls: partial.calls ?? false, callInstructions: partial.callInstructions ?? '' }) as MessageSettings;
  const store = fakeProject();
  const vfs = new Vfs();
  const events: Array<{ id: string; event: AgentEvent }> = [];
  let changes = 0;
  const sessions = new Sessions(
    store.project as never,
    (extra) => new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'Project: phone', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => changes++, event: (session, event) => events.push({ id: session.id, event }) },
  );
  return { sessions, store, desktop, events, changes: () => changes };
}

/** Wait until nothing is working (the queue drained). */
async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 200 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  expect(sessions.busy).toBe(false);
}

describe('text-message conversations', () => {
  it('a text opens a conversation for its sender, and the agent replies through the phone', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('Text message from Lance (+61491570006):\\nAre you open Saturday?');
        expect(said).toContain('text-message thread with Lance (+61491570006)');
        expect(said).toContain('Be brief.');
        return { calls: [{ name: 'send_text_message', input: { body: 'Yes, 9 to 1 on Saturday.' } }] };
      },
      { text: 'Told them our Saturday hours.' },
    ]);
    const { sessions, desktop, store } = setup({ answer: true, instructions: 'Be brief.' });
    const session = await sessions.textArrived('+61 491 570 006', 'Lance', 'Are you open Saturday?');
    await settled(sessions);
    expect(session.id).toBe('sms-61491570006');
    expect(desktop.commands).toEqual([{ connector: 'aokie', command: 'sms.send', payload: { to: '+61491570006', body: 'Yes, 9 to 1 on Saturday.' } }]);
    expect(fake.bodies).toHaveLength(2);
    // The conversation is kept, and listed with its unread count.
    expect(store.chats.get(session.id)?.length).toBeGreaterThan(2);
    expect(store.index).toMatchObject([{ id: 'sms-61491570006', kind: 'sms', key: '+61491570006', title: 'Lance', unread: 1 }]);
    // The same sender's next text continues it.
    fakeProvider('openai', [{ text: 'Nothing to reply to a thank-you.' }]);
    const again = await sessions.textArrived('+61491570006', 'Lance', 'Thanks!');
    await settled(sessions);
    expect(again).toBe(session);
    expect(sessions.list).toHaveLength(1);
  });

  it('with answering off, texts are kept; turning it on answers the recent ones', async () => {
    const messages = { answer: false, instructions: '' };
    const { sessions, desktop } = setup(messages);
    const session = await sessions.textArrived('+61400000001', '', 'Hello');
    expect(sessions.busy).toBe(false);
    expect(session.agent.turns).toEqual([{ role: 'user', text: textMessage('+61400000001', '+61400000001', 'Hello') }]);
    fakeProvider('openai', [{ calls: [{ name: 'send_text_message', input: { body: 'Hi! How can I help?' } }] }, { text: 'Replied.' }]);
    messages.answer = true;
    expect(sessions.answerWaiting()).toBe(1);
    await settled(sessions);
    expect(desktop.commands.map((c) => c.payload.body)).toEqual(['Hi! How can I help?']);
    // The text is in the conversation once, not twice.
    expect(session.agent.turns.filter((t) => t.role === 'user' && t.text.includes('Hello'))).toHaveLength(1);
  });

  it('a pretend text is answered but never sent', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'send_text_message', input: { body: 'We are open.' } }] }, { text: 'done' }]);
    const { sessions, desktop, events } = setup({ answer: false, instructions: '' });
    const session = await sessions.textArrived(TEST_NUMBER, 'Test', 'Open today?');
    await settled(sessions);
    expect(desktop.commands).toEqual([]);
    const result = events.find((e) => e.event.type === 'tool_result')?.event as Extract<AgentEvent, { type: 'tool_result' }>;
    expect(result.result.content).toBe('Not sent (a test conversation): "We are open."');
    expect(session.title).toBe('Test');
  });

  it('a text that comes while its conversation works reaches it at its next step', async () => {
    const script = [
      { calls: [{ name: 'list_files', input: { path: '/' } }] },
      (body: Record<string, unknown>) => {
        expect(JSON.stringify(body.messages)).toContain('Actually, make it Sunday');
        return { calls: [{ name: 'send_text_message', input: { body: 'Sunday it is.' } }] };
      },
      { text: 'Booked for Sunday.' },
    ];
    let first = true;
    const fake = fakeProvider('openai', script.map((step) => (body: Record<string, unknown>) => {
      if (first) {
        first = false;
        // The second text arrives while the first is being answered.
        void sessions.textArrived('+61400000002', 'Sam', 'Actually, make it Sunday');
      }
      return typeof step === 'function' ? step(body) : step;
    }));
    const { sessions, desktop } = setup({ answer: true, instructions: '' });
    await sessions.textArrived('+61400000002', 'Sam', 'Book me in for Saturday');
    await settled(sessions);
    expect(desktop.commands.map((c) => c.payload.body)).toEqual(['Sunday it is.']);
    expect(fake.bodies).toHaveLength(3);
  });

  it('refuses an empty or overlong reply, and says when the desktop is gone', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'send_text_message', input: { body: '' } }] },
      { calls: [{ name: 'send_text_message', input: { body: 'x'.repeat(1700) } }] },
      { text: 'gave up' },
    ]);
    const { sessions, events } = setup({ answer: true, instructions: '' });
    await sessions.textArrived('+61400000003', '', 'hi');
    await settled(sessions);
    const errors = events.filter((e) => e.event.type === 'tool_result').map((e) => (e.event as Extract<AgentEvent, { type: 'tool_result' }>).result.content);
    expect(errors[0]).toContain('body is empty');
    expect(errors[1]).toContain('keep it under 1600');
  });

  it('keeps its conversations: a new page loads them back', async () => {
    fakeProvider('openai', [{ text: 'noted' }]);
    const first = setup({ answer: true, instructions: '' });
    await first.sessions.textArrived('+61400000004', 'Kim', 'See you at 3');
    await settled(first.sessions);
    const again = new Sessions(
      first.store.project as never,
      (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }),
      () => ({ answer: true, instructions: '', calls: false, callInstructions: '' }),
      () => null,
      { changed: () => {}, event: () => {} },
    );
    await again.load();
    expect(again.list.map((s) => [s.title, s.key])).toEqual([['Kim', '+61400000004']]);
    expect(again.list[0].agent.turns[0]).toMatchObject({ role: 'user' });
  });
});

describe('a text the phone delivers again', () => {
  it("is not answered twice: the phone's handle says it is the same text", async () => {
    fakeProvider('openai', [{ calls: [{ name: 'send_text_message', input: { body: 'Hi!' } }] }, { text: 'done' }]);
    const { sessions, desktop, store } = setup({ answer: true, instructions: '' });
    const hello = { seq: 1, name: 'aokie.sms.received', source: 'aokie', correlationId: 'c1', idempotencyKey: 'k1', occurredAt: '', data: { from: '+61491570006', name: 'Lance', body: 'Hello', handle: '040000000000002F' } };
    await sessions.desktopEvent(hello);
    await settled(sessions);
    // The phone reconnects and delivers it again, under a new correlation.
    expect(await sessions.desktopEvent({ ...hello, seq: 7, correlationId: 'c7', idempotencyKey: 'k7' })).toBeNull();
    await settled(sessions);
    expect(desktop.commands).toHaveLength(1);
    expect(store.index[0].handles).toEqual(['040000000000002F']);
  });
});

describe("following the desktop's events", () => {
  function ringOf(events: DesktopEvent[]) {
    return {
      events: async (since: number) => {
        const after = events.filter((e) => e.seq > since);
        return { events: after.slice(0, 2), next: after.slice(0, 2).at(-1)?.seq ?? since };
      },
    };
  }
  const event = (seq: number, name = 'aokie.sms.received'): DesktopEvent => ({ seq, name, source: 'aokie', correlationId: `c${seq}`, idempotencyKey: `k${seq}`, occurredAt: '', data: { from: '+1', body: `m${seq}` } });

  it('acts only on what comes after it starts looking, however long the ring is', async () => {
    const ring = [event(1), event(2), event(3), event(4), event(5)];
    const seen: number[] = [];
    const follower = new DesktopEvents(() => ringOf(ring) as unknown as Desktop, (e) => void seen.push(e.seq), () => {}, 60_000);
    await follower.tick();
    expect(seen).toEqual([]);
    ring.push(event(6), event(7));
    await follower.tick();
    expect(seen).toEqual([6, 7]);
    follower.stop();
  });

  it('says when the desktop cannot be reached, and when it can again', async () => {
    const said: string[] = [];
    let down = true;
    const desktop = {
      events: async () => {
        if (down) throw new Error('OAIY Desktop did not answer');
        return { events: [], next: 0 };
      },
    };
    const follower = new DesktopEvents(() => desktop as unknown as Desktop, () => {}, (p) => said.push(p), 60_000);
    await follower.tick();
    down = false;
    await follower.tick();
    expect(said).toEqual(['OAIY Desktop did not answer', '']);
    follower.stop();
  });
});

describe("a flow's tasks for the agent", () => {
  it('each task is its own run in the flow\'s conversation, answered with what the agent said last', async () => {
    const asked: string[] = [];
    fakeProvider('openai', [
      (body) => {
        asked.push(JSON.stringify(body.messages));
        return { text: 'Welcome aboard, Sam!' };
      },
      (body) => {
        asked.push(JSON.stringify(body.messages));
        return { text: 'Welcome aboard, Priya!' };
      },
    ]);
    const { sessions } = setup({ answer: false, instructions: '' });
    const [sam, priya] = await Promise.all([sessions.task('Welcome', 'Write a welcome line for Sam'), sessions.task('Welcome', 'Now one for Priya')]);
    expect([sam, priya]).toEqual(['Welcome aboard, Sam!', 'Welcome aboard, Priya!']);
    expect(asked[0]).toContain('Your flow \\"Welcome\\" asks: Write a welcome line for Sam');
    expect(asked[0]).toContain('your last reply is handed back to the flow as its output');
    expect(asked[1]).toContain('asks: Now one for Priya');
    const flows = sessions.list.filter((s) => s.kind === 'task');
    expect(flows.map((s) => s.title)).toEqual(['Welcome']);
    await settled(sessions);
  });

  it("a message of the person's in the flow's conversation does not take a task's answer", async () => {
    // Each reply answers what was asked last.
    const answer = (body: Record<string, unknown>) => {
      const last = JSON.stringify((body.messages as unknown[]).at(-1));
      return { text: last.includes('Hello, how is it going?') ? 'Reply to the person.' : last.includes('Second') ? 'The second task done.' : 'The first task done.' };
    };
    fakeProvider('openai', [answer, answer, answer]);
    const { sessions } = setup({ answer: false, instructions: '' });
    expect(await sessions.task('Nightly', 'First')).toBe('The first task done.');
    const session = sessions.list.find((s) => s.kind === 'task')!;
    // The person writes in the flow's tab, and a task comes while that is answered.
    sessions.say(session, 'Hello, how is it going?');
    const second = sessions.task('Nightly', 'Second');
    expect(await second).toBe('The second task done.');
    await settled(sessions);
  });

  it('a task the agent cannot answer fails, and the flow is told why', async () => {
    fakeProvider('openai', [{ error: { status: 400, body: '{"error":{"message":"the model is away"}}' } }]);
    const { sessions } = setup({ answer: false, instructions: '' });
    await expect(sessions.task('Nightly report', 'Summarise the day')).rejects.toThrow();
    await settled(sessions);
  });
});

describe('lanes', () => {
  it("a caller's words are answered while a flow's task is still being worked on", async () => {
    const store = fakeProject();
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const order: string[] = [];
    // A stand-in agent: a task's run holds until released; a call's answers at once.
    const makeAgent = () =>
      ({
        turns: [] as Turn[],
        run: async (prompt: string, emit: (e: AgentEvent) => void) => {
          order.push(`start ${prompt.slice(0, 30)}`);
          if (prompt.includes('asks:')) await held;
          order.push(`end ${prompt.slice(0, 30)}`);
          emit({ type: 'done', text: 'ok', steps: 1 } as AgentEvent);
        },
        interject: () => false,
        takeUnread: () => [],
        savedTurns: () => [],
        warm: async () => {},
      }) as unknown as Agent;
    const sessions = new Sessions(store.project as never, makeAgent, () => ({ answer: false, calls: true, instructions: '', callInstructions: '' }) as MessageSettings, () => null, { changed: () => {}, event: () => {} });
    const task = sessions.task('Nightly', 'Summarise the day');
    await new Promise((r) => setTimeout(r, 10));
    await sessions.callEvent({ type: 'call.started', callId: 'c1', from: '+61400000000' });
    await sessions.callEvent({ type: 'call.caller', callId: 'c1', text: 'Are you open today?' });
    await new Promise((r) => setTimeout(r, 10));
    expect(order).toContain('end Caller: Are you open today?');
    expect(order).not.toContain('end [OAIY] Your flow "Nightly" as');
    release();
    await task;
  });
});

describe('what a stranger can reach', () => {
  it("a text thread's agent reads the front desk's files and replies, but writes, runs and fetches nothing", async () => {
    let offered: string[] = [];
    fakeProvider('openai', [
      (body) => {
        offered = ((body.tools as Array<{ function?: { name: string }; name?: string }>) ?? []).map((t) => t.function?.name ?? t.name ?? '');
        return { calls: [{ name: 'send_text_message', input: { body: 'Hi!' } }] };
      },
      { text: '' },
    ]);
    const { sessions } = setup({ answer: true, instructions: '' });
    await sessions.textArrived('+61400000001', 'Sam', 'Hello?');
    await settled(sessions);
    expect(offered).toEqual(expect.arrayContaining(['read_file', 'grep', 'send_text_message']));
    for (const tool of ['write_file', 'edit_file', 'delete_file', 'web_fetch', 'code_run', 'sandbox_shell', 'generate_image']) expect(offered).not.toContain(tool);
  });
});

describe("the runner's direction", () => {
  it("every call and text reads the front desk's brief afresh, before anything the caller asks", async () => {
    let brief = 'We are fully booked until Friday.';
    const bodies: string[] = [];
    fakeProvider('openai', [
      (body) => {
        bodies.push(JSON.stringify(body));
        return { text: 'The earliest is Monday.' };
      },
      (body) => {
        bodies.push(JSON.stringify(body));
        return { text: 'Sure, Wednesday works.' };
      },
    ]);
    const store = fakeProject();
    const sessions = new Sessions(
      store.project as never,
      (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }),
      () => ({ answer: false, calls: true, instructions: '', callInstructions: '' }) as MessageSettings,
      () => null,
      { changed: () => {}, event: () => {} },
      () => brief,
    );
    await sessions.callEvent({ type: 'call.started', callId: 'c1', from: '+61400000002' });
    await sessions.callEvent({ type: 'call.caller', callId: 'c1', text: 'Can you come Wednesday?' });
    await settled(sessions);
    expect(bodies[0]).toContain('the brief wins');
    expect(bodies[0]).toContain('We are fully booked until Friday.');
    brief = 'Wednesdays are open again.';
    await sessions.callEvent({ type: 'call.caller', callId: 'c1', text: 'What about next week?' });
    await settled(sessions);
    expect(bodies[1]).toContain('Wednesdays are open again.');
    expect(bodies[1]).not.toContain('fully booked until Friday');
  });
});
