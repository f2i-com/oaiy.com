// Reaching the owner for a caller who asks for a person, and taking a message: what the model is offered (nothing until
// the owner allows it, and then the same for every call), what it is told when a request comes out, what the caller is
// never told (a transfer that has not happened), and how a call the owner took goes on or ends. The desktop decides
// whether anything rings; these are about what the receptionist's agent does with what the desktop says.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { NO_CALL_FEATURES, Sessions, TRANSFER_NOTES, callInstructions, ownerInstructions } from '../../src/sessions';
import { NO_IDENTITY } from '../../src/identity';
import type { Desktop } from '../../src/desktop/bridge';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

type Sent = Array<[string, string, unknown]>;

/** What a project keeps of its conversations: a page that is reloaded reads it again (`setup(overrides, store)`). */
interface Store {
  chats: Map<string, Turn[]>;
  index: SessionInfo[];
  callers: CallerNote[];
}
const newStore = (): Store => ({ chats: new Map(), index: [], callers: [] });

/** Sessions on a fake desktop that records what it is asked and answers a request to reach the owner as `ringing`. */
function setup(overrides: Record<string, unknown> = {}, store: Store = newStore()) {
  const { chats } = store;
  const project = {
    loadSessions: async () => store.index,
    saveSessions: async (list: SessionInfo[]) => {
      store.index = list;
    },
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => {
      chats.set(id, turns);
    },
    loadCallers: async () => store.callers,
    saveCallers: async (list: CallerNote[]) => {
      store.callers = list;
    },
  };
  const sent: Sent = [];
  const said: string[] = [];
  const desktop = {
    say: async (_callId: string, text: string) => void said.push(text),
    finishCall: async (callId: string, goodbye: string) => {
      sent.push(['finish', callId, goodbye]);
      return { ok: true, output: { accepted: true } };
    },
    callTool: async (callId: string, name: string, args: unknown) => {
      sent.push([name, callId, args]);
      if (name === 'transfer_to_owner') return { ok: true, output: { status: 'ringing', requestId: 'assist_1', ringSeconds: 30, instruction: 'The owner is being rung.' } };
      return { ok: true, output: { recorded: true } };
    },
    takeMessage: async (callId: string, body: unknown) => {
      sent.push(['take_message', callId, body]);
      return { recorded: true, id: 'msg_1', notified: true };
    },
    ...overrides,
  };
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: false, instructions: '', calls: true, callInstructions: 'Be kind.' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
  );
  return { sessions, sent, said, chats, store };
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
}

const ALLOWED = { type: 'voice.features', transfer: true, messages: true };
const START = { type: 'call.started', callId: 'call_o', from: '+61491570006', name: 'Alex', allowTransfer: true, greeting: 'Thanks for calling!' };
const toolNames = (body: Record<string, unknown>) => (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
const lastUser = (turns: Turn[]) => [...turns].reverse().find((t) => t.role === 'user') as { text: string } | undefined;

describe('what the model is offered about reaching the owner', () => {
  it('is nothing until the owner allows it: the instructions are byte for byte what they were, and there are no tools for it', async () => {
    const before = callInstructions('Be kind.');
    expect(callInstructions('Be kind.', false, NO_IDENTITY, NO_CALL_FEATURES)).toBe(before);
    expect(ownerInstructions(NO_CALL_FEATURES)).toBe('');
    expect(before).not.toMatch(/transfer_to_owner|take_message|speak to the owner/i);
    const fake = fakeProvider('openai', [{ text: 'Hello!' }]);
    const { sessions } = setup();
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    const names = toolNames(fake.bodies[0]);
    expect(names).toEqual(expect.arrayContaining(['end_call', 'request_appointment', 'lookup_business_data']));
    expect(names).not.toContain('transfer_to_owner');
    expect(names).not.toContain('take_message');
    expect(JSON.stringify(fake.bodies[0].messages)).not.toContain('transfer_to_owner');
    // ...and the start note says nothing of it either (a call started as ever).
    expect(JSON.stringify(fake.bodies[0].messages)).not.toContain('The owner can be reached');
  });

  it('is both tools and the words for them once it does, the same for every call and every caller', async () => {
    const text = callInstructions('Be kind.', false, NO_IDENTITY, { transfer: true, messages: true });
    expect(text).toContain('transfer_to_owner');
    expect(text).toContain('take_message');
    // What it may say: trying, never transferred or connected until told; nothing promised; no owner number.
    expect(text).toContain('Until you are told the owner has accepted');
    expect(text).toContain('never say or imply that the call is being transferred, connected, put through, handed over or on hold');
    // The words the desktop would drop anyway, named, so the model does not spend a turn saying them.
    for (const words of ['"connecting you"', '"transferring you"', '"putting you through"']) expect(text).toContain(`not ${words}`);
    expect(text).toContain('I\'ll try to reach them');
    expect(text).toContain('never promise a callback time or say why');
    expect(text).toContain("You do not have the owner's number, and never give one.");
    expect(text).toContain('Say the owner will be told only when take_message says so; otherwise say the message is saved.');
    // The same for every call: no name, number or time in it (the model's prompt cache keeps it).
    expect(text).not.toMatch(/\d{4}|Alex|Liam/);
    expect(callInstructions('Be kind.', false, NO_IDENTITY, { transfer: true, messages: true })).toBe(text);
    // Messages alone (transfers off): a message for a caller who asks for the owner, never a person coming to the phone.
    const only = ownerInstructions({ transfer: false, messages: true });
    expect(only).toContain('take_message');
    expect(only).not.toContain('transfer_to_owner');
    expect(only).toContain('never say a person will come to the phone');

    const fake = fakeProvider('openai', [{ text: 'One.' }, { text: 'Two.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hi' });
    await settled(sessions);
    expect(toolNames(fake.bodies[0])).toEqual(expect.arrayContaining(['transfer_to_owner', 'take_message']));
    // Another caller's call: the model reads the very same instructions and tool list.
    await sessions.callEvent({ type: 'call.ended', callId: 'call_o' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_p', from: '+61400000002', name: 'Sam', allowTransfer: false });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_p', text: 'Hi' });
    await settled(sessions);
    const system = (b: Record<string, unknown>) => JSON.stringify((b.messages as Array<{ role: string; content: unknown }>).filter((m) => m.role === 'system'));
    expect(system(fake.bodies[1])).toBe(system(fake.bodies[0]));
    expect(toolNames(fake.bodies[1])).toEqual(toolNames(fake.bodies[0]));
  });

  it('follows what the desktop says as it connects and when the owner changes it; an older desktop says nothing and nothing is offered', async () => {
    const { sessions } = setup();
    expect(sessions.features).toEqual({ transfer: false, messages: false });
    await sessions.callEvent({ type: 'hello', calls: [] });
    expect(sessions.features).toEqual({ transfer: false, messages: false });
    await sessions.callEvent({ type: 'hello', calls: [], features: { transfer: true, messages: true } });
    expect(sessions.features).toEqual({ transfer: true, messages: true });
    await sessions.callEvent({ type: 'voice.features', transfer: false, messages: true });
    expect(sessions.features).toEqual({ transfer: false, messages: true });
    await sessions.callEvent({ type: 'voice.features', transfer: false, messages: false });
    expect(sessions.features).toEqual({ transfer: false, messages: false });
    // Transfers imply messages, even from a desktop that forgot to say (the fallback of a ring nobody answers).
    await sessions.callEvent({ type: 'voice.features', transfer: true });
    expect(sessions.features).toEqual({ transfer: true, messages: true });
    // Nonsense is off.
    await sessions.callEvent({ type: 'voice.features', transfer: 'yes', messages: 1 });
    expect(sessions.features).toEqual({ transfer: false, messages: false });
  });

  it("the call's own note says whether the owner can be rung on it, and only while transfers are allowed", async () => {
    const fake = fakeProvider('openai', [{ text: 'A.' }, { text: 'B.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hi' });
    await settled(sessions);
    expect(JSON.stringify(fake.bodies[0].messages)).toContain(TRANSFER_NOTES.available);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_o' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_q', from: '+61400000003', allowTransfer: false });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_q', text: 'Hi' });
    await settled(sessions);
    const second = JSON.stringify(fake.bodies[1].messages);
    expect(second).toContain(TRANSFER_NOTES.unavailable);
    expect(second).not.toContain(TRANSFER_NOTES.available);
  });
});

describe('a request to reach the owner', () => {
  it('goes to the desktop as exactly the reason, and the model is told it rings, not that anyone is coming', async () => {
    const fake = fakeProvider('openai', [
      { text: "I'll try to reach them, please stay with me.", calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked', note: 'tell them the owner said yes', number: '+61400000000' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('ringing');
        expect(said).toContain('The owner is being rung.');
        return { text: 'May I take your name while I wait?' };
      },
    ]);
    const { sessions, sent, said } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    expect(sent).toEqual([['transfer_to_owner', 'call_o', { reason: 'caller_asked' }]]);
    expect(said).toEqual(["I'll try to reach them, please stay with me.", 'May I take your name while I wait?']);
    expect(said.join(' ')).not.toMatch(/transferr|connect|put you through|on hold/i);
    expect(fake.bodies).toHaveLength(2);
  });

  it('a refusal from the desktop (the caller did not ask, quiet hours, a limit) is the model’s to read, and nothing is promised', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('Do not say why');
        return { text: "I'm sorry, they are not available. Can I take a message?" };
      },
    ]);
    const { sessions, said } = setup({
      callTool: async () => ({ ok: false, output: { status: 'unavailable', reason: 'quiet_hours', instruction: 'The owner cannot be reached right now. Tell the caller kindly and offer to take a message (take_message). Do not say why, and do not promise a callback time.' } }),
    });
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Put me through to the owner' });
    await settled(sessions);
    expect(said.join(' ')).toBe("I'm sorry, they are not available. Can I take a message?");
    expect(fake.bodies).toHaveLength(2);
    // No ring is going: no clock waits for an outcome.
    expect(sessions.list[0].transferRequest).toBeUndefined();
  });

  it('when the owner accepts the agent stops and says nothing more, and what the caller says is kept and not answered', async () => {
    const fake = fakeProvider('openai', [{ calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] }, { text: 'Please hold.' }]);
    const { sessions, said } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    const before = fake.bodies.length;
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome: 'accepted', source: 'phone' });
    expect(call.handingOver).toBe(true);
    expect(call.transferRequest).toBeUndefined();
    expect(lastUser(call.agent.turns)?.text).toBe(TRANSFER_NOTES.accepted);
    // The caller talks while the owner connects: heard, kept, not answered.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hello? Is anyone there?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(before);
    expect(said.filter((s) => s.includes('anyone'))).toEqual([]);
    // What was said is in the record, after the note: nothing of it was answered.
    expect(call.agent.turns.at(-1)).toMatchObject({ role: 'user', text: TRANSFER_NOTES.accepted });
    expect(call.aside).toHaveLength(1);
  });

  it('a decline, a timeout, a takeover that failed and a request withdrawn each tell the model what is true and to offer a message', async () => {
    for (const [outcome, expected] of [
      ['declined', TRANSFER_NOTES.declined],
      ['expired', TRANSFER_NOTES.nobody],
      ['unavailable', TRANSFER_NOTES.nobody],
      ['cancelled', TRANSFER_NOTES.cancelled],
    ] as const) {
      const fake = fakeProvider('openai', [
        { text: "I'll try.", calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
        { text: 'One moment.' },
        (body) => {
          expect(JSON.stringify(body.messages)).toContain(expected.replace(/"/g, '\\"'));
          return { text: "I'm sorry, they can't come to the phone. Would you like to leave a message?" };
        },
      ]);
      const { sessions, said } = setup();
      await sessions.callEvent(ALLOWED);
      await sessions.callEvent(START);
      await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'I want to speak to the owner' });
      await settled(sessions);
      await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome, source: 'phone' });
      await settled(sessions);
      expect(said.join(' '), outcome).toContain("I'm sorry, they can't come to the phone. Would you like to leave a message?");
      expect(sessions.list[0].handingOver, outcome).toBe(false);
      expect(said.join(' '), outcome).not.toMatch(/connect|transferr|put you through/i);
      expect(fake.bodies).toHaveLength(3);
      vi.unstubAllGlobals();
    }
  });

  it('takes how a request came out even when this page never saw it ring: a session that took the call over is told, and so is the model', async () => {
    // The line moved to a new session while a request was ringing: this page saw no `ringing` for it (it began in the session before), and the
    // desktop tells it how it came out all the same.
    const fake = fakeProvider('openai', [
      (body) => {
        expect(JSON.stringify(body.messages)).toContain(TRANSFER_NOTES.declined.replace(/"/g, '\\"'));
        return { text: "I'm sorry, they can't come to the phone. Would you like to leave a message?" };
      },
    ]);
    const { sessions, said } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    expect(call.transferRequest).toBeUndefined();
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_9', outcome: 'declined', source: 'phone' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
    expect(said.join(' ')).toContain("I'm sorry, they can't come to the phone. Would you like to leave a message?");
    expect(call.handingOver).toBe(false);
    // ...and an acceptance for one it never saw ring stops the agent and says nothing more.
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_10', outcome: 'accepted', source: 'phone' });
    expect(call.handingOver).toBe(true);
    expect(lastUser(call.agent.turns)?.text).toBe(TRANSFER_NOTES.accepted);
  });

  it('the owner’s own words for the caller are relayed as words, with no promise added', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
      { text: 'Hold on.' },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain(TRANSFER_NOTES.declinedWith('Back after three, please leave a message').replace(/"/g, '\\"'));
        return { text: 'They will be back after three. Would you like to leave a message?' };
      },
    ]);
    const { sessions, said } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome: 'declined', message: 'Back after three, please leave a message' });
    await settled(sessions);
    expect(said.join(' ')).toContain('They will be back after three. Would you like to leave a message?');
    expect(fake.bodies).toHaveLength(3);
  });

  it('is told, with the caller’s next words, that a line which promised a transfer was not said, and what they heard instead', async () => {
    const wanted = 'Putting you through now.';
    const said = "I'll try to reach them.";
    const fake = fakeProvider('openai', [{ text: wanted }, { text: 'Sorry, I am still trying to reach them.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
    // The desktop swapped the line for its own: nothing yet for the model to answer, and a note kept for the caller's next words.
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_o', wanted, said });
    expect(fake.bodies).toHaveLength(1);
    expect(call.aside).toEqual([TRANSFER_NOTES.replaced(wanted, said)]);
    // The caller never heard it: it is not among what the agent had said that a cut would list as unsaid.
    expect(call.speech?.reply ?? []).not.toContain(wanted);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hello? Are you still there?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    const read = JSON.stringify(fake.bodies[1].messages);
    expect(read).toContain(TRANSFER_NOTES.replaced(wanted, said).replace(/"/g, '\\"'));
    expect(read).toContain('Hello? Are you still there?');
    // The note says what is true: nothing is being put through, and the caller is not to be told it is.
    expect(TRANSFER_NOTES.replaced(wanted, said)).toContain('nobody has accepted the call');
    expect(TRANSFER_NOTES.replaced(wanted, said)).toContain(said);
    // It is read once, with those words, and not again.
    expect(call.aside).toEqual([]);
  });

  it('a line that was not said counts as never heard: said again, it is sent again (the desktop decides what is said)', async () => {
    const wanted = 'Putting you through now.';
    // (A sentence said before is held back when the reply has something new; one never heard is not held back.)
    fakeProvider('openai', [{ text: wanted }, { text: `${wanted} Please give me a moment.` }]);
    const { sessions, said } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    expect(said.filter((s) => s === wanted)).toHaveLength(1);
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_o', wanted, said: "I'll try to reach them." });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hello?' });
    await settled(sessions);
    expect(said.filter((s) => s === wanted)).toHaveLength(2);
  });

  it('a note of a line not said that names no line, or no words in its place, or comes for a call it does not follow, is nothing', async () => {
    const fake = fakeProvider('openai', [{ text: 'Hello.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_o', wanted: 'Putting you through.' });
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_o', said: 'I will try.' });
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_o', wanted: '  ', said: 'I will try.' });
    await sessions.callEvent({ type: 'call.line_replaced', callId: 'call_x', wanted: 'Putting you through.', said: 'I will try.' });
    expect(call.aside ?? []).toEqual([]);
    expect(fake.bodies).toHaveLength(0);
  });

  it('a ring the desktop never reports the end of is ended for the model too, after its time and a grace', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
      { text: 'One moment.' },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain(TRANSFER_NOTES.nobody);
        return { text: "I'm sorry, I couldn't reach them. Can I take a message?" };
      },
    ]);
    const { sessions, said } = setup({ callTool: async () => ({ ok: true, output: { status: 'ringing', requestId: 'assist_9', ringSeconds: 0.01, instruction: 'Ringing.' } }) });
    sessions.transferGraceMs = 250;
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    expect(call.transferRequest).toBe('assist_9');
    await new Promise((r) => setTimeout(r, 450));
    await settled(sessions);
    expect(said.join(' ')).toContain("I'm sorry, I couldn't reach them. Can I take a message?");
    expect(call.transferRequest).toBeUndefined();
    expect(fake.bodies).toHaveLength(3);
  });

  it('a call that ends stops the clock, so nothing is said to a caller who has gone', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] }, { text: 'One moment.' }]);
    const { sessions, said } = setup({ callTool: async () => ({ ok: true, output: { status: 'ringing', requestId: 'assist_9', ringSeconds: 0.01, instruction: 'Ringing.' } }) });
    sessions.transferGraceMs = 250;
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_o', reason: 'hung up' });
    const spoken = said.length;
    await new Promise((r) => setTimeout(r, 450));
    expect(said).toHaveLength(spoken);
    expect(call.transferRequest).toBeUndefined();
  });
});

describe('a call the owner takes', () => {
  it('is not a call that ended: nothing is written as its end, and it goes on in the same conversation when handed back', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] }, { text: 'Please hold.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome: 'accepted' });
    await sessions.callEvent({ type: 'call.handoff', callId: 'call_o', phase: 'to_human', reason: 'handoff:takeover' });
    expect(call.callId).toBe('call_o');
    expect(call.handoff).toMatchObject({ reason: 'handoff:takeover' });
    expect(call.agent.turns.some((t) => t.role === 'user' && t.text.includes('The call ended'))).toBe(false);
    expect(lastUser(call.agent.turns)?.text).toBe(TRANSFER_NOTES.handoff);
    // The owner's hello after a restart of the stream: a call in handoff is still going on.
    await sessions.callEvent({ type: 'hello', calls: ['call_o'], features: { transfer: true, messages: true } });
    expect(call.callId).toBe('call_o');

    // Handed back: the same call begins again. Nothing of it starts over, and the model is told not to greet again.
    const starts = () => call.agent.turns.filter((t) => t.role === 'user' && t.text.startsWith('[OAIY] 📞 A call from')).length;
    const before = starts();
    const back = await sessions.callEvent({ type: 'call.started', callId: 'call_o', from: '+61491570006', name: 'Alex', allowTransfer: true, resume: { afterHandoff: true, handoffSeconds: 42, via: 'return' } });
    expect(back).toBe(call);
    expect(call.handoff).toBeUndefined();
    expect(call.handingOver).toBe(false);
    expect(starts()).toBe(before);
    expect(lastUser(call.agent.turns)?.text).toBe(TRANSFER_NOTES.back('0:42'));
    expect(lastUser(call.agent.turns)?.text).toContain('do not greet the caller again');
  });

  it('ends as any call does when the phone says it ended while the owner had it', async () => {
    fakeProvider('openai', [{ text: 'Hello.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    await sessions.callEvent({ type: 'call.handoff', callId: 'call_o', phase: 'to_human', reason: 'handoff:takeover' });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_o', reason: 'ended_during_handoff' });
    expect(call.callId).toBeUndefined();
    expect(call.handoff).toBeUndefined();
    expect(call.agent.turns.at(-1)).toMatchObject({ text: '[OAIY] 📞 The call ended.' });
  });

  /** A call the owner has taken, and this page reloaded meanwhile: what a new page reads of what the old one kept. */
  async function reloadedDuringHandoff(script: Parameters<typeof fakeProvider>[1]) {
    const fake = fakeProvider('openai', script);
    const first = setup();
    await first.sessions.callEvent(ALLOWED);
    await first.sessions.callEvent(START);
    await first.sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(first.sessions);
    await first.sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome: 'accepted' });
    await first.sessions.callEvent({ type: 'call.handoff', callId: 'call_o', phase: 'to_human', reason: 'handoff:takeover' });
    await settled(first.sessions);
    // The page is closed and opened again: a new Sessions over what the project kept. Nothing else is remembered.
    const page = setup({}, first.store);
    await page.sessions.load();
    return { fake, page, sessions: page.sessions, said: page.said, before: fake.bodies.length };
  }
  const BACK = { type: 'call.started', callId: 'call_o', from: '+61491570006', name: 'Alex', allowTransfer: true, instructions: 'Be kind.', greeting: 'Thank you for waiting.', resume: { afterHandoff: true, handoffSeconds: 42, via: 'return' } };
  const callStarts = (turns: Turn[]) => turns.filter((t) => t.role === 'user' && t.text.startsWith('[OAIY] 📞 A call from')).length;

  it('goes on when this page was reloaded while the owner had it, in the conversation it saved: not greeted again, not dropped, and the caller is answered', async () => {
    const { fake, sessions, said, before } = await reloadedDuringHandoff([
      { calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
      { text: 'Please hold.' },
      (body) => {
        const seen = JSON.stringify(body.messages);
        // It has the conversation from before the handoff, that the owner took the call, and that it is back and not to greet.
        expect(seen).toContain('Can I speak to the owner?');
        expect(seen).toContain(TRANSFER_NOTES.handoff);
        expect(seen).toContain('do not greet the caller again');
        expect(seen).toContain('handed the call back after 0:42');
        return { text: 'Anything else I can help with?' };
      },
    ]);
    expect(before).toBe(2);
    await sessions.callEvent({ type: 'hello', calls: ['call_o'], features: { transfer: true, messages: true } });
    const back = await sessions.callEvent(BACK);
    // The conversation is the one from before: not a new call's start.
    expect(back).not.toBeNull();
    expect(sessions.list.filter((s) => s.kind === 'call')).toHaveLength(1);
    expect(callStarts(back!.agent.turns)).toBe(1);
    expect(back!.callId).toBe('call_o');
    expect(back!.canTransfer).toBe(true);
    expect(back!.handingOver).toBe(false);
    expect(lastUser(back!.agent.turns)?.text).toBe(TRANSFER_NOTES.back('0:42'));
    // It says nothing until the caller does (the phone said its own line as they came back).
    await settled(sessions);
    expect(fake.bodies).toHaveLength(before);
    expect(said).toEqual([]);
    // The caller speaks: the call is answered, and can still reach the owner or take a message.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Thanks, one more thing.' });
    await settled(sessions);
    expect(said.join(' ')).toBe('Anything else I can help with?');
    expect(fake.bodies).toHaveLength(before + 1);
    expect(toolNames(fake.bodies[before])).toEqual(expect.arrayContaining(['transfer_to_owner', 'take_message']));
  });

  it('is told by the phone alone, on a page that still has the call but did not see the handoff, and not by anything that is not exactly resume.afterHandoff', async () => {
    fakeProvider('openai', [{ text: 'Hello.' }, { text: 'Hello again.' }, { text: 'Hello again.' }]);
    const { sessions } = setup();
    await sessions.callEvent(ALLOWED);
    const call = (await sessions.callEvent(START))!;
    expect(call.handoff).toBeUndefined();
    const starts = callStarts(call.agent.turns);
    // What was played before, by the first run's clock, and when that clock began.
    await sessions.callEvent({ type: 'call.said', callId: 'call_o', text: 'Thanks for calling!', startMs: 0, endMs: 2_000 });
    expect(call.played).toHaveLength(1);
    const zero = call.clockZero!;
    await new Promise((r) => setTimeout(r, 15));
    // (A page that saw the phone begin the call, and not the handoff: the phone's word is enough.)
    expect(await sessions.callEvent(BACK)).toBe(call);
    expect(callStarts(call.agent.turns)).toBe(starts);
    expect(lastUser(call.agent.turns)?.text).toBe(TRANSFER_NOTES.back('0:42'));
    // The desktop's clock for the call began again as it went on: what was played by the old one means nothing on it.
    expect(call.played).toEqual([]);
    expect(call.clockZero).toBeGreaterThan(zero);
    // Anything else is a call beginning: its start is written, as ever (a resume that is not after a handoff, a flag that is not true).
    for (const resume of [{ afterHandoff: false }, { afterHandoff: 'true' }, {}, null, 'yes']) {
      await sessions.callEvent({ type: 'call.ended', callId: 'call_o' });
      const again = (await sessions.callEvent({ ...BACK, resume }))!;
      expect(callStarts(again.agent.turns)).toBeGreaterThan(starts);
      expect(lastUser(again.agent.turns)?.text).toContain('A call from');
    }
  });

  it('is ended as any call is when it ends while the owner had it and this page was reloaded: nothing of it opens here again', async () => {
    const { sessions, said, before, fake } = await reloadedDuringHandoff([
      { calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked' } }] },
      { text: 'Please hold.' },
    ]);
    await sessions.callEvent({ type: 'hello', calls: [], features: { transfer: true, messages: true } });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_o', reason: 'ended_during_handoff' });
    // The caller's last words, after the end, are kept in their conversation and not answered.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hello? Anyone?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(before);
    expect(said).toEqual([]);
    expect(sessions.list.filter((s) => s.kind === 'call' && s.callId)).toHaveLength(0);
  });
});

describe('a caller who asks for the owner and leaves a message instead', () => {
  it('runs both call tools through the Agent against a desktop that declines: the request goes as exactly {reason}, the decline is read as fact, the message is taken with what the desktop is meant to take, and nothing untrue is said', async () => {
    const fake = fakeProvider('openai', [
      // The caller asks; the model asks to reach the owner, with an argument it was not meant to add.
      { text: "I'll try to reach them, please stay with me.", calls: [{ name: 'transfer_to_owner', input: { reason: 'caller_asked', note: 'tell them yes', number: '+61400000000' } }] },
      { text: 'And what is it about?' },
      // The desktop says the owner declined: the model is told what is true and to offer a message.
      (body) => {
        const seen = JSON.stringify(body.messages);
        expect(seen).toContain('The owner cannot take the call now');
        expect(seen).toContain('offer to take a message (take_message)');
        return { text: "I'm sorry, they can't come to the phone. Would you like to leave a message?" };
      },
      // The caller says yes and what: the model takes it, with a number of its own invention that must not go.
      { calls: [{ name: 'take_message', input: { message: 'Please ring about the gate.', callerName: 'Alex', callbackNumber: '0491 570 006', wantsCallback: true, from: '+61400000000' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The owner has been told a message is waiting');
        return { text: "I've passed that on." };
      },
    ]);
    const { sessions, sent, said } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can I speak to the owner?' });
    await settled(sessions);
    expect(sent).toEqual([['transfer_to_owner', 'call_o', { reason: 'caller_asked' }]]);
    await sessions.callEvent({ type: 'call.transfer', callId: 'call_o', requestId: 'assist_1', outcome: 'declined', source: 'phone' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Yes please, ring me about the gate' });
    await settled(sessions);
    expect(sent).toEqual([
      ['transfer_to_owner', 'call_o', { reason: 'caller_asked' }],
      ['take_message', 'call_o', { message: 'Please ring about the gate.', callerName: 'Alex', callbackNumber: '0491 570 006', wantsCallback: true }],
    ]);
    expect(said.join(' ')).toBe("I'll try to reach them, please stay with me. And what is it about? I'm sorry, they can't come to the phone. Would you like to leave a message? I've passed that on.");
    expect(said.join(' ')).not.toMatch(/connect|transferr|put you through/i);
    expect(fake.bodies).toHaveLength(5);
  });
});

describe('taking a message', () => {
  it('keeps what the caller said, and says the owner will be told only when the desktop says so', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'take_message', input: { message: 'Ring me about Friday.', callerName: 'Sam', callbackNumber: '0491 570 156', urgency: 'urgent', wantsCallback: true, from: '+61400000000' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('The owner has been told a message is waiting');
        expect(said).toContain('Do not promise a callback time');
        return { text: "I've passed that on." };
      },
    ]);
    const { sessions, sent, said } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Yes, tell them to ring me about Friday' });
    await settled(sessions);
    // Only what the desktop is meant to take: never a number of the model's own choosing as the caller's.
    expect(sent).toEqual([['take_message', 'call_o', { message: 'Ring me about Friday.', callerName: 'Sam', callbackNumber: '0491 570 156', urgency: 'urgent', wantsCallback: true }]]);
    expect(said).toEqual(["I've passed that on."]);
    expect(fake.bodies).toHaveLength(2);
  });

  it('says it is saved, not that the owner will be told, when nobody could be told', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'take_message', input: { message: 'Ring me.' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('saved for the owner');
        expect(said).not.toContain('The owner has been told');
        return { text: 'I have saved that for them.' };
      },
    ]);
    const { sessions, said } = setup({ takeMessage: async () => ({ recorded: true, id: 'msg_2', notified: false }) });
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Just tell them to ring me' });
    await settled(sessions);
    expect(said).toEqual(['I have saved that for them.']);
  });

  it('never says it was kept when it was not: a limit, or messages off, is told as it is', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'take_message', input: { message: 'Another.' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('NOT recorded');
        expect(said).toContain('3 messages have been taken on this call');
        return { text: "I'm sorry, I couldn't take that one." };
      },
    ]);
    const { sessions, said } = setup({
      takeMessage: async () => {
        throw new Error('3 messages have been taken on this call: no more can be kept');
      },
    });
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'One more message please' });
    await settled(sessions);
    expect(said).toEqual(["I'm sorry, I couldn't take that one."]);
  });

  it('asks for the words when there are none, and keeps a long one to the limit', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'take_message', input: { message: '   ' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('message is empty');
        return { calls: [{ name: 'take_message', input: { message: 'x'.repeat(601) } }] };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('keep it to 600');
        return { text: 'What would you like me to tell them?' };
      },
    ]);
    const { sessions, sent } = setup();
    await sessions.callEvent(ALLOWED);
    await sessions.callEvent(START);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Can you take a message?' });
    await settled(sessions);
    expect(sent).toEqual([]);
  });
});
