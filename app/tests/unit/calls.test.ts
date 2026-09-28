import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { SessionInfo } from '../../src/vfs/projects';
import { Sessions, Speech, callInstructions, earlierWords, sameNumber, spoken } from '../../src/sessions';
import type { Desktop } from '../../src/desktop/bridge';
import type { MessageSettings } from '../../src/settings';
import { LOCAL, OPENAI, fakeProvider } from './fakeProvider';
import type { ProviderConfig } from '../../src/agent/providers/types';

afterEach(() => vi.unstubAllGlobals());

describe('speaking what the agent writes', () => {
  it('says each sentence as it is complete, in order, cleaned of markdown and emoji', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => {
      await new Promise((r) => setTimeout(r, 5));
      said.push(text);
    });
    speech.begin();
    for (const delta of ['Sure! We are **open**', ' from 9 to 5. ', 'Would you like', ' to book? 😊\n', 'See [our hours](http://x) too']) speech.push(delta);
    expect(said).toEqual([]);
    speech.flush();
    await speech.done;
    expect(said).toEqual(['Sure!', 'We are open from 9 to 5.', 'Would you like to book?', 'See our hours too']);
  });

  it('hushed, the rest of the reply is dropped until the next one', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => void said.push(text));
    speech.begin();
    speech.push('One. Two ');
    speech.hush();
    speech.push('three. Four.');
    speech.flush();
    speech.begin();
    speech.push('Next reply.');
    speech.flush();
    await speech.done;
    expect(said).toEqual(['One.', 'Next reply.']);
  });

  it('a sentence already said on this call is not said again; a short one may be, and a new call starts over', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => void said.push(text));
    const reply = (text: string) => {
      speech.begin();
      speech.push(text);
      speech.flush();
    };
    reply("Sure! I've recorded the request for Thursday at ten.");
    reply("Okay. I've recorded the request for Thursday, at ten!");
    reply('Sure! Anything else?');
    await speech.done;
    expect(said).toEqual(['Sure!', "I've recorded the request for Thursday at ten.", 'Okay.', 'Sure!', 'Anything else?']);
    speech.newCall();
    reply("I've recorded the request for Thursday at ten.");
    await speech.done;
    expect(said.at(-1)).toBe("I've recorded the request for Thursday at ten.");
  });

  it('a long run of words is spoken at a comma rather than waited on', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => void said.push(text));
    speech.begin();
    speech.push(`${'word '.repeat(30)}, and ${'more '.repeat(30)}`);
    await speech.done;
    expect(said.length).toBeGreaterThan(0);
    expect(said[0].endsWith(',')).toBe(true);
    expect(spoken('# Hi `there` > you')).toBe('Hi there you');
  });

  it('its instructions put the phone first: spoken words, the call tools, the brief', () => {
    const text = callInstructions('Lance', '+61491570006', 'You are Aokie, a warm receptionist.', 'Never quote prices.');
    expect(text).toContain('live phone call with Lance (+61491570006)');
    expect(text).toContain('Everything you write is spoken aloud');
    expect(text).toContain('end_call');
    expect(text).toContain('The receptionist brief:\nYou are Aokie');
    expect(text).toContain('for calls:\nNever quote prices.');
  });
});

function setup(provider: ProviderConfig = OPENAI, toolWait?: Promise<void>) {
  const chats = new Map<string, Turn[]>();
  let index: SessionInfo[] = [];
  const project = {
    loadSessions: async () => index,
    saveSessions: async (list: SessionInfo[]) => {
      index = list;
    },
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => {
      chats.set(id, turns);
    },
  };
  const calls: Array<[string, string, unknown]> = [];
  const desktop = {
    say: async (callId: string, text: string) => void calls.push(['say', callId, text]),
    finishCall: async (callId: string, goodbye: string) => {
      calls.push(['finish', callId, goodbye]);
      return { ok: true, output: { accepted: true } };
    },
    callTool: async (callId: string, name: string, args: unknown) => {
      calls.push([name, callId, args]);
      // A tool that takes a while (the phone looking something up).
      if (toolWait) await toolWait;
      return { ok: true, output: name === 'lookup_business_data' ? { answer: 'Open Saturday 8 to 2.' } : { recorded: true, status: 'requested' } };
    },
  };
  const messages: MessageSettings = { answer: false, instructions: '', calls: true, callInstructions: 'Be kind.' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
  );
  return { sessions, calls, chats };
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
}

describe('a phone call answered by the agent', () => {
  it('the caller speaks, the agent answers aloud, and ends the call when they are done', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('A call from Lance (+61491570006) began. You greeted them: \\"Thanks for calling!\\"');
        expect(said).toContain('Caller: Are you open on Saturday?');
        expect(JSON.stringify(body)).toContain('The receptionist brief:\\nYou are Aokie.');
        return { text: 'Yes, we are open on Saturday from nine. Anything else?' };
      },
      { calls: [{ name: 'end_call', input: { goodbye: 'Thanks, bye!' } }] },
      { text: '' },
    ]);
    const { sessions, calls } = setup();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '+61 491 570 006', name: 'Lance', instructions: 'You are Aokie.', greeting: 'Thanks for calling!' });
    expect(call?.kind).toBe('call');
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Are you open on Saturday?' });
    await settled(sessions);
    expect(calls).toEqual([
      ['say', 'call_1', 'Yes, we are open on Saturday from nine.'],
      ['say', 'call_1', 'Anything else?'],
    ]);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: "No, that's all, thanks." });
    await settled(sessions);
    expect(calls.at(-1)).toEqual(['finish', 'call_1', 'Thanks, bye!']);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_1', reason: 'hung up' });
    expect(call?.callId).toBeUndefined();
    expect(call?.agent.turns.at(-1)).toMatchObject({ text: '[OAIY] 📞 The call ended.' });
    expect(fake.bodies.length).toBeGreaterThanOrEqual(2);
    // The same caller's next call continues the conversation.
    const again = await sessions.callEvent({ type: 'call.started', callId: 'call_2', from: '+61491570006', name: 'Lance' });
    expect(again).toBe(call);
    expect(sessions.list.filter((s) => s.kind === 'call')).toHaveLength(1);
  });

  it('on a call the model does not think first, and has a short list of tools', async () => {
    const fake = fakeProvider('openai', [{ text: 'Hello!' }]);
    const { sessions } = setup(LOCAL);
    await sessions.callEvent({ type: 'call.started', callId: 'call_7', from: '+61400000007' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_7', text: 'Hi' });
    await settled(sessions);
    const body = fake.bodies[0] as { chat_template_kwargs?: unknown; tools: Array<{ function: { name: string } }> };
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
    const names = body.tools.map((t) => t.function.name);
    expect(names).toEqual(expect.arrayContaining(['end_call', 'request_appointment', 'lookup_business_data', 'read_file']));
    expect(names).not.toContain('sandbox_shell');
    expect(names).not.toContain('update_plan');
  });

  it('an appointment request goes to the phone, and its answer back to the agent', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'request_appointment', input: { callerName: 'Sam', service: 'Haircut', date: '2026-10-03', time: '10:00', agreementPhrase: 'yes ten works' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('requested');
        return { text: 'I have noted a request for Saturday at ten; staff will confirm it.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_9', from: '+61400000009', name: '' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_9', text: 'Yes ten works' });
    await settled(sessions);
    expect(calls[0]).toEqual(['request_appointment', 'call_9', { callerName: 'Sam', service: 'Haircut', date: '2026-10-03', time: '10:00', agreementPhrase: 'yes ten works' }]);
    expect(calls.at(-1)?.[0]).toBe('say');
  });

  it("a call takes the name the caller's text conversation has (the caller id gives none)", async () => {
    const { sessions } = setup();
    const texts = await sessions.conversationWith('+61491570006', 'Lance', 'sms');
    expect(texts.title).toBe('Lance');
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_n', from: '0491570006' });
    expect(call?.title).toBe('Lance');
    expect(sameNumber('+61 491 570 006', '0491570006')).toBe(true);
    expect(sameNumber('0491570006', '0491570157')).toBe(false);
    expect(sameNumber('110', '110')).toBe(false);
  });

  it('its words are its answer: never asked to start work, so nothing is said twice', async () => {
    // Live 28 Sept 2026: "Let me check what's open on the calendar." with no tool call drew the coding
    // nudges ("You said what you will do…"), and each nudge made it say the line again, aloud.
    const fake = fakeProvider('openai', [{ text: "Let me check what's open on the calendar." }, { text: 'We have Friday at two.' }]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_3', from: '+61400000003' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_3', text: "What's your availability?" });
    await settled(sessions);
    expect(calls).toEqual([['say', 'call_3', "Let me check what's open on the calendar."]]);
    expect(fake.bodies).toHaveLength(1);
    const rules = callInstructions('', '+61400000003', '', '');
    expect(rules).toContain('Never say you will check without doing it');
    expect(rules).toContain('Never make up availability');
    expect(rules).toContain('When the caller says goodbye or is done, call end_call');
  });

  it("no records to ask: the agent is told plainly, so it does not make up a time", async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'lookup_business_data', input: { question: 'What is free on Thursday?' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('Do not guess times, availability');
        return { text: "I can't check the calendar right now." };
      },
    ]);
    const { sessions, calls } = setup();
    const desktopCalls = calls;
    (sessions as unknown as { desktop: () => { callTool: unknown } }).desktop().callTool = async (callId: string, name: string, args: unknown) => {
      desktopCalls.push([name, callId, args]);
      return { ok: true, output: { answer: 'LOOKUP UNAVAILABLE (no result)' } };
    };
    await sessions.callEvent({ type: 'call.started', callId: 'call_4', from: '+61400000004' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_4', text: "What's free on Thursday?" });
    await settled(sessions);
    expect(calls[0]).toEqual(['lookup_business_data', 'call_4', { question: 'What is free on Thursday?' }]);
    expect(calls.at(-1)).toEqual(['say', 'call_4', "I can't check the calendar right now."]);
  });

  it("a new call starts afresh: only the caller's last words carry over, never a line said again and again", () => {
    const check = "Let me check what's open on the calendar.";
    const turns: Turn[] = [
      { role: 'user', text: '[OAIY] 📞 A call from Lance began.', automatic: true },
      { role: 'user', text: 'Caller: Hi, can you hear me?' },
      { role: 'assistant', text: 'Yes, I can hear you!', calls: [] },
      { role: 'user', text: "Caller: What's your availability?" },
      { role: 'assistant', text: check, calls: [] },
      { role: 'user', text: '[OAIY] You said what you will do…', automatic: true },
      { role: 'assistant', text: check, calls: [] },
      { role: 'assistant', text: '', calls: [{ id: 't1', name: 'lookup_business_data', input: {} }] },
      { role: 'tool', results: [{ id: 't1', name: 'lookup_business_data', content: '{}', isError: false }] },
      { role: 'user', text: 'Caller: Okay, bye.' },
    ];
    expect(earlierWords(turns)).toEqual([
      { role: 'user', text: 'Caller: Hi, can you hear me?' },
      { role: 'assistant', text: 'Yes, I can hear you!', calls: [] },
      { role: 'user', text: "Caller: What's your availability?" },
      { role: 'user', text: 'Caller: Okay, bye.' },
    ]);
    expect(earlierWords(turns, 2)).toEqual([
      { role: 'user', text: "Caller: What's your availability?" },
      { role: 'user', text: 'Caller: Okay, bye.' },
    ]);
  });

  it('when the caller speaks over it, the rest of that reply is not said', async () => {
    let interrupt!: () => void;
    fakeProvider('openai', [
      () => {
        interrupt();
        return { text: 'This part is long. And this part should not be said.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_5', from: '+61400000005' });
    interrupt = () => void sessions.callEvent({ type: 'call.interrupted', callId: 'call_5', itemId: 'out_1' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_5', text: 'Hello?' });
    await settled(sessions);
    expect(calls.filter((c) => c[0] === 'say')).toEqual([]);
  });

  it('a caller speaking while a tool works does not stop it: the result and their words reach the agent, and the answer is spoken', async () => {
    let lookedUp!: () => void;
    const lookup = new Promise<void>((r) => (lookedUp = r));
    const fake = fakeProvider('openai', [
      { text: 'Let me check.', calls: [{ name: 'lookup_business_data', input: { question: 'Saturday hours' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('Open Saturday 8 to 2.');
        expect(said).toContain('Also, do you mow on Sundays?');
        return { text: "We're open Saturday from eight to two. We don't mow on Sundays." };
      },
    ]);
    const { sessions, calls } = setup(OPENAI, lookup);
    await sessions.callEvent({ type: 'call.started', callId: 'call_9', from: '+61400000009' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_9', text: 'What are your Saturday hours?' });
    for (let i = 0; i < 100 && !calls.some((c) => c[0] === 'lookup_business_data'); i++) await new Promise((r) => setTimeout(r, 10));
    // The caller speaks over the check: the words stop, the lookup goes on.
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_9', itemId: 'out_1' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_9', text: 'Also, do you mow on Sundays?' });
    lookedUp();
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['Let me check.', "We're open Saturday from eight to two.", "We don't mow on Sundays."]);
  });

  it('a tool that takes a while gets a short line said, once, when nothing has been said', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const said: string[] = [];
      const speech = new Speech(async (text) => void said.push(text));
      speech.begin();
      speech.hold('One moment, let me check.');
      speech.hold('One moment, let me check.');
      await speech.done;
      expect(said).toEqual(['One moment, let me check.']);
      speech.begin();
      speech.push('Let me look. ');
      speech.hold('One moment, let me check.');
      await speech.done;
      expect(said).toEqual(['One moment, let me check.', 'Let me look.']);
    } finally {
      vi.useRealTimers();
    }
  });
});
