import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Sessions, Speech, callInstructions, callerNotesTool, promisesBooking, sameNumber, spoken, tellAgentTool, tidyReplies } from '../../src/sessions';
import type { Desktop } from '../../src/desktop/bridge';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
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

  it('a reply that only repeats what was said is still said: the caller is never met with silence', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => void said.push(text));
    const reply = (text: string) => {
      speech.begin();
      speech.push(text);
      speech.flush();
    };
    reply('I will note the request as Tuesday at one. What name should I use?');
    reply('I will note the request as Tuesday at one. What name should I use?');
    reply('I will note the request as Tuesday at one. Anything else?');
    await speech.done;
    expect(said).toEqual([
      'I will note the request as Tuesday at one.',
      'What name should I use?',
      'I will note the request as Tuesday at one.',
      'What name should I use?',
      'Anything else?',
    ]);
    // What the reply said, for when the caller speaks over it.
    expect(speech.reply).toEqual(['Anything else?']);
  });

  it('one filler word opens a reply at most, never a run of them, and none after a tool', async () => {
    const said: string[] = [];
    const speech = new Speech(async (text) => void said.push(text));
    speech.begin();
    speech.push('Great! Let me look that up. Sure thing! ');
    speech.flush();
    speech.begin(true);
    speech.push('Happy to! We have Friday free.');
    speech.flush();
    await speech.done;
    expect(said).toEqual(['Great!', 'Let me look that up.', 'We have Friday free.']);
  });

  it("the call's replies are read back as the caller heard them: no line twice, no run of fillers", () => {
    const turns: Turn[] = [
      { role: 'user', text: 'Caller: A lawn mowing appointment.' },
      { role: 'assistant', text: 'Great! Let me check what days we have open.', calls: [{ id: 't', name: 'lookup_business_data', input: {} }] },
      { role: 'tool', results: [{ id: 't', name: 'lookup_business_data', content: 'unavailable', isError: false }] },
      { role: 'assistant', text: 'Good! Let me check what days we have open. Sure thing! I can\'t check the calendar right now.', calls: [] },
      { role: 'user', text: 'Caller: Okay.' },
      { role: 'assistant', text: 'Let me check what days we have open.', calls: [] },
    ];
    tidyReplies(turns);
    expect(turns.filter((t) => t.role === 'assistant').map((t) => (t as { text: string }).text)).toEqual([
      'Great! Let me check what days we have open.',
      "I can't check the calendar right now.",
      // Nothing new in it: it was said again, so it stays.
      'Let me check what days we have open.',
    ]);
  });

  it('a booking counts as promised only when told as done or doing, not offered on a condition', () => {
    expect(promisesBooking("Sure! I'll request Tuesday 29 September at 1 p.m. under Lance.")).toBe(true);
    expect(promisesBooking("I've noted that down for Tuesday.")).toBe(true);
    expect(promisesBooking("I can't check right now, so tell me the day and time you'd like and I'll take it as a request.")).toBe(false);
    expect(promisesBooking("I'll request that. What name should I use?")).toBe(false);
    expect(promisesBooking('Let me check what days we have open.')).toBe(false);
  });

  it('a reply in quotation marks is said without them', () => {
    expect(spoken('"Of course! What service are you looking for today?"')).toBe('Of course! What service are you looking for today?');
    expect(spoken('\u201cTuesday at 1 p.m.\u201d')).toBe('Tuesday at 1 p.m.');
    expect(spoken("We don't mow on Sundays.")).toBe("We don't mow on Sundays.");
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
    const text = callInstructions('You are Aokie, a warm receptionist.', 'Never quote prices.');
    expect(text).toContain('This conversation is a live phone call');
    // Nothing in them changes during a call or from call to call (the model's prompt cache keeps them).
    expect(text).not.toMatch(/\d{4}|Lance/);
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
    callers: [] as CallerNote[],
    loadCallers: async () => project.callers,
    saveCallers: async (list: CallerNote[]) => {
      project.callers = list;
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
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: false, instructions: '', calls: true, callInstructions: 'Be kind.' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
  );
  return { sessions, calls, chats, project };
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
        expect(said).toMatch(/A call from Lance \(\+61491570006\) began, [^.]+\. You greeted them: \\"Thanks for calling!\\"\\nToday is \w+ \d+ \w+ \d{4}\.\\nNothing is saved about them yet/);
        expect(said).toContain('Caller: Are you open on Saturday?');
        expect(JSON.stringify(body)).toContain('The receptionist brief:\\nYou are Aokie.');
        return { text: 'Yes, we are open on Saturday from nine. Anything else?' };
      },
      { calls: [{ name: 'end_call', input: { goodbye: 'Thanks, bye!' } }] },
      { text: 'Done.' },
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
    // Nothing is said after the goodbye (a model may write "Done." after end_call).
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

  it("the caller's last words, heard after the call ended, are kept and not answered: the call is not taken up again", async () => {
    const fake = fakeProvider('openai', [{ text: 'Bye now.' }]);
    const { sessions, calls } = setup();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_e', from: '+61400000017' });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_e', reason: 'hung up' });
    // Still being transcribed as they hung up: the desktop sends it after the end.
    expect(await sessions.callEvent({ type: 'call.caller', callId: 'call_e', text: 'Thanks, bye.', startMs: 61_000, endMs: 61_800 })).toBeNull();
    await settled(sessions);
    expect(fake.bodies).toHaveLength(0);
    expect(calls).toEqual([]);
    expect(call?.callId).toBeUndefined();
    const turns = call!.agent.turns;
    expect(turns.filter((t) => t.role === 'user' && t.text.startsWith('[OAIY] 📞 A call from'))).toHaveLength(1);
    expect(turns.at(-1)).toMatchObject({ role: 'user', text: 'Caller [1:01]: Thanks, bye.' });
    // A call this page did not follow is not opened by its last words either.
    await sessions.callEvent({ type: 'call.ended', callId: 'call_f' });
    expect(await sessions.callEvent({ type: 'call.caller', callId: 'call_f', text: 'Hello?' })).toBeNull();
    expect(sessions.list).toHaveLength(1);
    expect(sessions.list.some((s) => s.callId)).toBe(false);
  });

  it('a call whose end was missed (the desktop restarted) ends when the desktop says which calls go on', async () => {
    const { sessions } = setup();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_h', from: '+61400000018' });
    // The stream opens again with the call still going on: nothing changes.
    await sessions.callEvent({ type: 'hello', calls: ['call_h'] });
    expect(call?.callId).toBe('call_h');
    // It opens again after the desktop restarted: the call is gone.
    expect(await sessions.callEvent({ type: 'hello', calls: [] })).toBeNull();
    expect(call?.callId).toBeUndefined();
    expect(call?.agent.turns.at(-1)).toMatchObject({ text: '[OAIY] 📞 The call ended.' });
  });

  it('nothing is said after the goodbye, even when another tool of the same reply answers after end_call', async () => {
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'end_call', input: { goodbye: 'Thanks, bye!' } }, { name: 'remember', input: { fact: 'Mows fortnightly' } }] },
      { text: 'Done. I saved that for next time.' },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_g', from: '+61400000020' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_g', text: "That's all, bye." });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(calls).toEqual([['finish', 'call_g', 'Thanks, bye!']]);
  });

  it('a reply that speaks and ends the call says one goodbye: its own words, then the phone hangs up after them', async () => {
    // Live 29 Sept 2026: "You're very welcome, Lance." then end_call's "You're welcome, Lance, have a great day!".
    const fake = fakeProvider('openai', [
      { text: "You're very welcome, Lance.", calls: [{ name: 'end_call', input: { goodbye: "You're welcome, Lance, have a great day!" } }] },
      { text: 'Done.' },
    ]);
    const { sessions, calls } = setup();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_bye', from: '+61491570006', name: 'Lance' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_bye', text: 'Thanks so much, bye.' });
    await settled(sessions);
    expect(calls).toEqual([['finish', 'call_bye', "You're very welcome, Lance."]]);
    expect(fake.bodies).toHaveLength(2);
    const result = call!.agent.turns.find((t) => t.role === 'tool');
    expect(JSON.stringify(result)).toContain('was not said (one goodbye, not two)');
  });

  it('the earlier sentences of that reply are said as it is written, and its last one is the goodbye (a trailing line break too)', async () => {
    fakeProvider('openai', [{ text: 'No problem at all! You are very welcome, Lance.\n', calls: [{ name: 'end_call', input: { goodbye: 'Bye now!' } }] }, { text: '' }]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_bye2', from: '+61400000031' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_bye2', text: "That's everything." });
    await settled(sessions);
    expect(calls).toEqual([
      ['say', 'call_bye2', 'No problem at all!'],
      ['finish', 'call_bye2', 'You are very welcome, Lance.'],
    ]);
  });

  it("a call asks the calendar's free times itself, answered at once, and a check the same as one a moment ago says so, turn after turn", async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const tools = (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
        expect(tools).toEqual(expect.arrayContaining(['calendar_free_times', 'lookup_business_data', 'end_call']));
        expect(JSON.stringify(body)).toContain('calendar_free_times (what is free, answered at once');
        return { calls: [{ name: 'calendar_free_times', input: { from: '2026-10-01', days: 1 } }] };
      },
      { text: 'Thursday has room from ten.' },
      { calls: [{ name: 'calendar_free_times', input: { from: '2026-10-01', days: 1 } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('the same as your check a moment ago');
        return { text: 'Still from ten on Thursday.' };
      },
    ]);
    const { sessions, calls } = setup();
    const desktop = (sessions as unknown as { desktop: () => Record<string, unknown> }).desktop();
    let asked = 0;
    desktop.calendar = async () => ({ available: true, settings: {}, appointments: [], now: '2026-09-29T09:00' });
    desktop.calendarFree = async () => {
      asked++;
      return { minutes: 60, service: null, days: [{ date: '2026-10-01', times: ['10:00', '11:00'] }] };
    };
    await sessions.callEvent({ type: 'call.started', callId: 'call_cal', from: '+61400000033' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_cal', text: 'Anything Thursday?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_cal', text: 'Sorry, Thursday again?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(4);
    expect(asked).toBeGreaterThanOrEqual(2);
    // Not through the phone's lookup: the calendar answers the agent itself.
    expect(calls.filter((c) => c[0] === 'lookup_business_data')).toEqual([]);
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['Thursday has room from ten.', 'Still from ten on Thursday.']);
  });

  it('a reply with only end_call says the goodbye it gives', async () => {
    fakeProvider('openai', [{ calls: [{ name: 'remember', input: { fact: 'Books monthly' } }, { name: 'end_call', input: { goodbye: 'Thanks, Sam. Bye!' } }] }, { text: '' }]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_bye3', from: '+61400000032' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_bye3', text: 'Bye.' });
    await settled(sessions);
    expect(calls).toEqual([['finish', 'call_bye3', 'Thanks, Sam. Bye!']]);
  });

  it('what the caller said as the call ended is kept, and no reply is written for a call that has ended', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. And we close at five.', hold: { at: 10, until: held } },
      { text: 'Are you still there?' },
    ]);
    const { sessions } = setup();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_u', from: '+61400000021' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_u', text: 'When are you open?' });
    for (let i = 0; i < 100 && !fake.bodies.length; i++) await new Promise((r) => setTimeout(r, 10));
    // They speak while the reply is written, and hang up.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_u', text: 'Actually, never mind.', startMs: 5_000 });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_u' });
    release();
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
    expect(call!.agent.turns.at(-1)).toMatchObject({ role: 'user', text: 'Caller [0:05]: Actually, never mind.' });
  });

  it('the next caller is answered when the last call ended while its agent was still writing', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. And we close at five.', hold: { at: 10, until: held } },
      { text: 'Yes, we are open until five.' },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_q1', from: '+61400000022' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_q1', text: 'When are you open?' });
    for (let i = 0; i < 100 && !fake.bodies.length; i++) await new Promise((r) => setTimeout(r, 10));
    await sessions.callEvent({ type: 'call.ended', callId: 'call_q1' });
    // Another caller gets through, and speaks, before that agent has stopped.
    await sessions.callEvent({ type: 'call.started', callId: 'call_q2', from: '+61400000023' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_q2', text: 'Are you open today?' });
    release();
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(calls).toEqual([['say', 'call_q2', 'Yes, we are open until five.']]);
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
    const rules = callInstructions('', '');
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

  it("a new call starts afresh: the agent reads only this call (and a few lines of their last contact), the earlier ones stay, and it can look them up", async () => {
    const fake = fakeProvider('openai', [
      { text: 'We mow on Tuesdays.' },
      (body) => {
        const sent = JSON.stringify(body.messages);
        // Not the earlier call's turns: a short summary of it, in the note that starts this one.
        expect(sent).not.toContain('Caller: Do you mow on Tuesdays?');
        expect(sent).not.toContain('"content":"We mow on Tuesdays."');
        expect(sent).toContain('Their last contact, ');
        expect(sent).toContain('- They said: \\"Do you mow on Tuesdays?\\"\\n- You said: \\"We mow on Tuesdays.\\"');
        expect(sent).toContain('Caller: Hi again.');
        return { text: '', calls: [{ name: 'earlier_conversations', input: { words: 'tuesdays' } }] };
      },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Caller: Do you mow on Tuesdays?');
        expect(sent).toContain('Agent: We mow on Tuesdays.');
        return { text: 'Welcome back! Still after a Tuesday?' };
      },
    ]);
    const { sessions, chats } = setup();
    const first = await sessions.callEvent({ type: 'call.started', callId: 'call_a', from: '+61400000010' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_a', text: 'Do you mow on Tuesdays?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_a' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_b', from: '0400000010' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_b', text: 'Hi again.' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(3);
    // Both calls are kept, for the chat.
    const kept = JSON.stringify(chats.get(first!.thread));
    expect(kept).toContain('Do you mow on Tuesdays?');
    expect(kept).toContain('Welcome back!');
    // Only another caller's words are never found.
    const other = await sessions.callEvent({ type: 'call.started', callId: 'call_c', from: '+61499999999' });
    expect(sessions.earlierWith(other!, 'tuesdays')).toContain('Nothing earlier');
  });

  it("what the agent learns about a caller reaches their next call and texts, and the phone learns their name", async () => {
    const fake = fakeProvider('openai', [
      { text: '', calls: [{ name: 'remember', input: { name: 'Lance', fact: 'Has a big back lawn' } }] },
      { text: 'Thanks, Lance!' },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Name: Lance');
        expect(sent).toContain('- Has a big back lawn');
        return { text: 'Hi Lance!' };
      },
    ]);
    const named: CallerNote[] = [];
    const { sessions, project } = setup();
    (sessions as unknown as { hooks: { named: (n: CallerNote) => void } }).hooks.named = (n) => void named.push({ ...n });
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_n', from: '+61400000011' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_n', text: "It's Lance, I've got a big back lawn." });
    await settled(sessions);
    expect(call?.title).toBe('Lance');
    expect(named.at(-1)).toMatchObject({ number: '+61400000011', name: 'Lance', facts: ['Has a big back lawn'] });
    expect(project.callers).toHaveLength(1);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_n' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_o', from: '0400000011' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_o', text: 'Hello?' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(3);
    // The runner reads and changes the same note.
    const notes = callerNotesTool(() => sessions);
    expect(await notes.run({}, new AbortController().signal)).toContain('Lance');
    expect(await notes.run({ number: '0400000011', remove: 'lawn', add: 'Prefers mornings' }, new AbortController().signal)).toContain('- Prefers mornings');
    expect(sessions.callerNote('+61400000011')?.facts).toEqual(['Prefers mornings']);
    // A name cleared: the phone forgets it, and the conversations go back to the number.
    await notes.run({ number: '0400000011', name: '' }, new AbortController().signal);
    expect(named.at(-1)?.name).toBeUndefined();
    expect(sessions.list.filter((s) => s.key === '+61400000011' || s.key === '0400000011').every((s) => s.title === s.key)).toBe(true);
  });

  it('the runner passes a note to a call going on now: read with its next reply, or acted on at once', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('A note from the runner (the main agent your person talks to): Offer them 10% off.');
        return { text: 'We can do ten percent off.' };
      },
      { text: 'Sorry to cut in, we can also do Sunday.' },
    ]);
    const { sessions, calls } = setup();
    const tell = tellAgentTool(() => sessions);
    const signal = new AbortController().signal;
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_t', from: '+61400000012' });
    expect(await tell.run({ id: call!.id, note: 'Offer them 10% off.' }, signal)).toContain('before its next reply');
    expect(fake.bodies).toHaveLength(0);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_t', text: 'How much is a mow?' });
    await settled(sessions);
    expect(await tell.run({ id: call!.id, note: 'Tell them Sunday is free too.', now: true }, signal)).toContain('acts on it now');
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toContain('Sorry to cut in, we can also do Sunday.');
    await sessions.callEvent({ type: 'call.ended', callId: 'call_t' });
    expect(await tell.run({ id: call!.id, note: 'Anything.' }, signal)).toContain('not on a call now');
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

  it('a reply the caller speaks over stays in the conversation as far as they heard it', async () => {
    let interrupt!: () => void;
    let sayNow!: () => void;
    const spokenYet = new Promise<void>((r) => (sayNow = r));
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. And on Sundays we are open too.', hold: { at: 24, until: spokenYet.then(() => interrupt()) } },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('We are open from nine.…');
        expect(sent).not.toContain('Sundays');
        return { text: 'Sure, go ahead.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_6', from: '+61400000006' });
    interrupt = () => void sessions.callEvent({ type: 'call.interrupted', callId: 'call_6', itemId: 'out_1' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_6', text: 'When are you open?' });
    for (let i = 0; i < 100 && !calls.length; i++) await new Promise((r) => setTimeout(r, 10));
    sayNow();
    await settled(sessions);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_6', text: 'Sorry, one more thing.' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
  });

  it('a lookup answers later: the agent talks with the caller meanwhile, and gives the answer when it comes', async () => {
    let lookedUp!: () => void;
    const lookup = new Promise<void>((r) => (lookedUp = r));
    const fake = fakeProvider('openai', [
      { text: 'Let me check.', calls: [{ name: 'lookup_business_data', input: { question: 'Saturday hours' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The answer comes to you in a message of its own');
        return { text: '' };
      },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Also, do you mow on Sundays?');
        expect(sent).not.toContain('Open Saturday 8 to 2.');
        return { text: "We don't mow on Sundays. Still checking Saturday." };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The answer to your lookup \\"Saturday hours\\":');
        return { text: "We're open Saturday from eight to two." };
      },
    ]);
    const { sessions, calls } = setup(OPENAI, lookup);
    await sessions.callEvent({ type: 'call.started', callId: 'call_9', from: '+61400000009' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_9', text: 'What are your Saturday hours?' });
    await settled(sessions);
    // The lookup still going, the caller asks something else: answered now.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_9', text: 'Also, do you mow on Sundays?' });
    await settled(sessions);
    lookedUp();
    await new Promise((r) => setTimeout(r, 20));
    await settled(sessions);
    expect(fake.bodies).toHaveLength(4);
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['Let me check.', "We don't mow on Sundays.", 'Still checking Saturday.', "We're open Saturday from eight to two."]);
  });

  it('an answer that comes while the caller speaks waits for their words, and goes with them', async () => {
    let lookedUp!: () => void;
    const lookup = new Promise<void>((r) => (lookedUp = r));
    const fake = fakeProvider('openai', [
      { text: '', calls: [{ name: 'lookup_business_data', input: { question: 'Saturday hours' } }] },
      { text: 'Checking.' },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toMatch(/Caller \[0:12\]: Sorry, and Sunday\?\\n\[OAIY\] The answer to your lookup/);
        return { text: 'Saturday eight to two, and no Sundays.' };
      },
    ]);
    const { sessions } = setup(OPENAI, lookup);
    await sessions.callEvent({ type: 'call.started', callId: 'call_w', from: '+61400000014' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_w', text: 'Saturday hours?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.speech_started', callId: 'call_w', atMs: 12_000, over: false });
    lookedUp();
    await new Promise((r) => setTimeout(r, 20));
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_w', text: 'Sorry, and Sunday?', startMs: 12_000, endMs: 13_000, over: false, cut: false, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(3);
  });

  it('an "mm-hmm" over the agent does not start a reply: it goes with the caller\'s next words, with when each was said', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Caller [0:03, over you as you said \\"We mow on Tuesdays and Fridays.\\"]: Mm-hmm.');
        expect(sent).toContain('Caller [0:06]: Friday then.');
        return { text: 'Friday it is.' };
      },
    ]);
    const { sessions } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_m', from: '+61400000015' });
    await sessions.callEvent({ type: 'call.said', callId: 'call_m', itemId: 'out_1', text: 'We mow on Tuesdays and Fridays.', startMs: 2_000, endMs: 4_500 });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_m', text: 'Mm-hmm.', startMs: 3_100, endMs: 3_400, over: true, cut: false, backchannel: true });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(0);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_m', text: 'Friday then.', startMs: 6_000, endMs: 7_000, over: false, cut: false, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
  });

  it('a reply cut off keeps only the sentences that had begun playing when the caller cut in', async () => {
    let cut!: () => void;
    const cutNow = new Promise<void>((r) => (cut = r));
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. We close at five. And on Sundays we rest.', hold: { at: 42, until: cutNow } },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('We are open from nine.…');
        expect(sent).not.toContain('We close at five.');
        expect(sent).toContain('Caller [0:09, cutting in as you said \\"We are open from nine.\\"]: Wait, Saturday?');
        return { text: 'Saturday too.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_x', from: '+61400000016' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_x', text: 'When are you open?' });
    for (let i = 0; i < 100 && calls.filter((c) => c[0] === 'say').length < 2; i++) await new Promise((r) => setTimeout(r, 10));
    // The first sentence has played from 0:08; the second was sent but had not begun when they cut in.
    await sessions.callEvent({ type: 'call.said', callId: 'call_x', itemId: 'out_1', text: 'We are open from nine.', startMs: 8_000, endMs: 9_600 });
    await sessions.callEvent({ type: 'call.said', callId: 'call_x', itemId: 'out_1', text: 'We close at five.', startMs: 9_600, endMs: 11_000 });
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_x', itemId: 'out_1', atMs: 9_400 });
    cut();
    await settled(sessions);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_x', text: 'Wait, Saturday?', startMs: 9_000, endMs: 9_900, over: true, cut: true, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
  });

  it('a reply written already, and cut off as it played, stays as far as it was heard, and the rest may be said again', async () => {
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. We close at five on weekdays.' },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('We are open from nine.…');
        expect(sent).not.toContain('We close at five on weekdays.');
        return { text: 'Sure. We close at five on weekdays.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_p', from: '+61400000024' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_p', text: 'When are you open?' });
    await settled(sessions);
    // Written at once, spoken over seconds: the caller cuts in before the second sentence plays.
    await sessions.callEvent({ type: 'call.said', callId: 'call_p', itemId: 'out_1', text: 'We are open from nine.', startMs: 8_000, endMs: 9_600 });
    await sessions.callEvent({ type: 'call.said', callId: 'call_p', itemId: 'out_1', text: 'We close at five on weekdays.', startMs: 9_600, endMs: 11_500 });
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_p', itemId: 'out_1', atMs: 9_400 });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_p', text: 'Sorry, and closing?', startMs: 9_000, endMs: 10_000, over: true, cut: true, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    // Never heard, so not held back as said before.
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['We are open from nine.', 'We close at five on weekdays.', 'Sure.', 'We close at five on weekdays.']);
  });

  it('a sentence cut off before it played, while the reply was still written, may be said again', async () => {
    let cut!: () => void;
    const cutNow = new Promise<void>((r) => (cut = r));
    fakeProvider('openai', [
      { text: 'We are open from nine. We close at five on weekdays. And on Sundays we rest.', hold: { at: 54, until: cutNow } },
      { text: 'Sure. We close at five on weekdays.' },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_r', from: '+61400000025' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_r', text: 'When are you open?' });
    for (let i = 0; i < 100 && calls.filter((c) => c[0] === 'say').length < 2; i++) await new Promise((r) => setTimeout(r, 10));
    await sessions.callEvent({ type: 'call.said', callId: 'call_r', itemId: 'out_1', text: 'We are open from nine.', startMs: 8_000, endMs: 9_600 });
    await sessions.callEvent({ type: 'call.said', callId: 'call_r', itemId: 'out_1', text: 'We close at five on weekdays.', startMs: 9_600, endMs: 11_500 });
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_r', itemId: 'out_1', atMs: 9_400 });
    cut();
    await settled(sessions);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_r', text: 'And closing?', startMs: 9_000, endMs: 9_900, over: true, cut: true, backchannel: false });
    await settled(sessions);
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['We are open from nine.', 'We close at five on weekdays.', 'Sure.', 'We close at five on weekdays.']);
  });

  it('a reply cut off by a "yeah, sure" and taken up again by the desktop is whole in the conversation, and no turn starts', async () => {
    let cut!: () => void;
    const cutNow = new Promise<void>((r) => (cut = r));
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. We close at five. And on Sundays we rest.', hold: { at: 42, until: cutNow } },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('We are open from nine. We close at five. And on Sundays we rest.');
        expect(sent).not.toContain('We close at five.…');
        expect(sent).toContain('Caller [0:10, over you as you said \\"We close at five.\\"]: Yeah, sure.\\nCaller [0:14]: Saturday too?');
        return { text: 'Saturday too.' };
      },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_y', from: '+61400000026' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_y', text: 'When are you open?' });
    for (let i = 0; i < 100 && calls.filter((c) => c[0] === 'say').length < 2; i++) await new Promise((r) => setTimeout(r, 10));
    await sessions.callEvent({ type: 'call.said', callId: 'call_y', itemId: 'out_1', text: 'We are open from nine.', startMs: 8_000, endMs: 9_600 });
    await sessions.callEvent({ type: 'call.said', callId: 'call_y', itemId: 'out_1', text: 'We close at five.', startMs: 9_600, endMs: 11_000 });
    // "Yeah, sure" from 0:09.95 stops it in its second line; the run stops with it.
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_y', itemId: 'out_1', atMs: 10_550 });
    cut();
    await settled(sessions);
    // Only an acknowledgement: the desktop says the rest itself, from the line cut off midway.
    await sessions.callEvent({ type: 'call.resumed', callId: 'call_y', itemId: 'out_1', fromSentence: 1, sentences: ['We close at five.', 'And on Sundays we rest.'] });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_y', text: 'Yeah, sure.', startMs: 9_950, endMs: 10_750, over: true, cut: false, backchannel: true, resumed: true });
    await sessions.callEvent({ type: 'call.said', callId: 'call_y', itemId: 'out_2', text: 'We close at five.', startMs: 11_100, endMs: 12_500 });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_y', text: 'Saturday too?', startMs: 14_000, endMs: 15_000, over: false, cut: false, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    // The app said its reply as far as it was written; the desktop said the rest again.
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(['We are open from nine.', 'We close at five.', 'Saturday too.']);
  });

  it('a reply written already, cut off in its first line by an acknowledgement and taken up again, is whole in the conversation', async () => {
    const fake = fakeProvider('openai', [
      { text: 'We are open from nine. We close at five on weekdays.' },
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('We are open from nine. We close at five on weekdays.');
        expect(sent).not.toContain('from nine.…');
        return { text: 'Saturday too.' };
      },
    ]);
    const { sessions } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_z', from: '+61400000027' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_z', text: 'When are you open?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.said', callId: 'call_z', itemId: 'out_1', text: 'We are open from nine.', startMs: 8_000, endMs: 9_600 });
    await sessions.callEvent({ type: 'call.said', callId: 'call_z', itemId: 'out_1', text: 'We close at five on weekdays.', startMs: 9_600, endMs: 11_500 });
    await sessions.callEvent({ type: 'call.interrupted', callId: 'call_z', itemId: 'out_1', atMs: 9_400 });
    await sessions.callEvent({ type: 'call.resumed', callId: 'call_z', itemId: 'out_1', fromSentence: 0, sentences: ['We are open from nine.', 'We close at five on weekdays.'] });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_z', text: 'Of course.', startMs: 8_800, endMs: 9_600, over: true, cut: false, backchannel: true, resumed: true });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_z', text: 'And Saturday?', startMs: 14_000, endMs: 15_000, over: false, cut: false, backchannel: false });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
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
      // The reply's words end at its tool call: said then, so the line is not.
      speech.push('Let me look. ');
      speech.flush();
      speech.hold('One moment, let me check.');
      await speech.done;
      expect(said).toEqual(['One moment, let me check.', 'Let me look.']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('a call that failed says why in its record; one that just ended does not', async () => {
    fakeProvider('openai', []);
    const { sessions } = setup();
    const a = await sessions.callEvent({ type: 'call.started', callId: 'call_f1', from: '+61400000031' });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_f1', reason: 'realtime voice failed: the websocket closed' });
    expect(a?.agent.turns.at(-1)).toMatchObject({ text: '[OAIY] 📞 The call ended: realtime voice failed: the websocket closed.' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_f2', from: '+61400000031' });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_f2', reason: 'hung up' });
    expect(a?.agent.turns.at(-1)).toMatchObject({ text: '[OAIY] 📞 The call ended.' });
  });

  it('a caller who hides their number gets a conversation of their own, named as such', async () => {
    fakeProvider('openai', []);
    const { sessions } = setup();
    const a = await sessions.callEvent({ type: 'call.started', callId: 'call_h1', from: '' });
    await sessions.callEvent({ type: 'call.ended', callId: 'call_h1' });
    const b = await sessions.callEvent({ type: 'call.started', callId: 'call_h2' });
    expect(a?.title).toBe('Hidden number');
    expect(b?.title).toBe('Hidden number');
    expect(a).not.toBe(b);
  });

  it('a booking promised but not requested: the agent is told once, and requests it', async () => {
    const fake = fakeProvider('openai', [
      { text: "Thanks, Lance! I'll request Tuesday at one for a mow." },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('request_appointment was not called');
        return { text: '', calls: [{ name: 'request_appointment', input: { callerName: 'Lance', service: 'mowing', date: '2026-09-29', time: '13:00', agreementPhrase: 'Tuesday at one' } }] };
      },
      { text: "It's requested." },
      { text: "Sure, I'll note that too." },
    ]);
    const { sessions, calls } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_b1', from: '+61400000013' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_b1', text: "It's Lance, Tuesday at one is great." });
    await settled(sessions);
    expect(calls.map((c) => c[0])).toContain('request_appointment');
    expect(calls.filter((c) => c[0] === 'say').map((c) => c[2])).toEqual(["Thanks, Lance!", "I'll request Tuesday at one for a mow.", "It's requested."]);
    // Once a call: the next promise is left alone.
    await sessions.callEvent({ type: 'call.caller', callId: 'call_b1', text: 'And my gate code is 1234.' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(4);
  });
});
describe("what the agents remember about a customer is written for the business's Contacts", () => {
  it('remember and caller_notes say how: a short plain note about the customer, the business by name or "the owner", never "your person"', async () => {
    // Live 29 Sept 2026: Contacts showed "…Your person noticed the Thursday one…" and "…waiting on your person to confirm."
    type Spec = { name: string; description: string; parameters: { properties: Record<string, { description?: string }> } };
    let remember: Spec | undefined;
    fakeProvider('openai', [
      (body) => {
        remember = (body.tools as Array<{ function: Spec }>).find((t) => t.function.name === 'remember')?.function;
        return { text: 'Okay.' };
      },
    ]);
    const { sessions } = setup();
    sessions.identity = () => ({ business: 'Green Lawns', receptionist: 'Aokie' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_f', from: '+61412345678' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_f', text: 'Hi.' });
    await settled(sessions);
    const notes = callerNotesTool(() => sessions).spec;
    for (const description of [remember!.description, notes.description]) {
      expect(description).toContain('Write each fact as a short plain note about the customer, e.g. "Wants to keep the Fri 2 Oct 1 pm booking".');
      expect(description).toContain('or as "the owner", never "your person"');
      expect(description).toContain('no internal wording');
    }
    // A call's agent knows the business's name.
    expect(remember!.description).toContain('Refer to the business by its name ("Green Lawns")');
    expect(remember!.parameters.properties.fact.description).toContain('"Wants to keep the Fri 2 Oct 1 pm booking"');
    expect(notes.parameters.properties).toMatchObject({ add: { description: expect.stringContaining('a short plain note about the customer') } });
  });
});
