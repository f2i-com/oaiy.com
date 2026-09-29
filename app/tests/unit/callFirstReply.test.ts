// A call's first reply comes sooner: its prompt starts the same on every call and every turn (the engine
// keeps it read), the engine reads it as the phone rings or a dial goes out (and lets go cleanly at the
// model's first word), a short hold word covers a slow reply, and none of it slows ChatGPT's live-call route.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { Agent, WARM_TOKENS } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { Callbacks, type Callback } from '../../src/callbacks';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import { chatgptProvider } from '../../src/desktop/agentModel';
import { NetGate } from '../../src/gate/netgate';
import { Outreach, type Campaign, type DoNotContact } from '../../src/outreach';
import { PhoneLine } from '../../src/phoneLine';
import { setLocalCountry } from '../../src/phoneNumbers';
import { HOLD_WORD_AFTER_MS, HOLD_WORDS, Sessions, Speech } from '../../src/sessions';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';

beforeAll(() => setLocalCountry('AU'));
afterEach(() => vi.unstubAllGlobals());

const ENGINE: ProviderConfig = { id: 'oaiy', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:8080', modelId: 'Qwen3.8-Flash-Next', followEngine: true };
const GREEN = { business: 'Green Lawns', receptionist: 'Aokie' };
const JANE = '+61412345678';
const TOM = '+61498765432';

interface Answer {
  text?: string;
  calls?: Array<{ name: string; input: Record<string, unknown> }>;
  /** Wait this long before the first word. */
  delayMs?: number;
  /** After the first word, write nothing more until the request is let go. */
  hold?: boolean;
}

interface Sent {
  body: Record<string, unknown>;
  signal: AbortSignal | null | undefined;
  /** The stream was let go by the app (reader.cancel). */
  cancelled: boolean;
}

/** OAIY's engine behind a fake fetch: each request answered by `answer`, as a stream of its own. */
function engine(answer: (body: Record<string, unknown>, n: number) => Answer, log: string[] = []): Sent[] {
  const sent: Sent[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    const record: Sent = { body, signal: init.signal, cancelled: false };
    sent.push(record);
    log.push(body.max_tokens === WARM_TOKENS ? 'warm' : 'turn');
    const a = answer(body, sent.length - 1);
    const events: unknown[] = [];
    for (const piece of a.text?.match(/.{1,7}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
    for (const [i, call] of (a.calls ?? []).entries()) {
      events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${sent.length}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
    }
    const end = [{ choices: [{ index: 0, delta: {}, finish_reason: a.calls?.length ? 'tool_calls' : 'stop' }] }, { choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }];
    const sse = (list: unknown[]) => new TextEncoder().encode(list.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(''));
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const go = () => {
          if (record.cancelled) return;
          if (a.hold) {
            controller.enqueue(sse(events.slice(0, 1)));
            return;
          }
          controller.enqueue(sse([...events, ...end]));
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();
        };
        if (a.delayMs) setTimeout(go, a.delayMs);
        else go();
      },
      cancel() {
        record.cancelled = true;
      },
    });
    return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  });
  return sent;
}

function world(provider: ProviderConfig = ENGINE) {
  let index: SessionInfo[] = [];
  const chats = new Map<string, Turn[]>();
  let callers: CallerNote[] = [
    { number: JANE, name: 'Jane Smith', nameBy: 'owner', notes: 'Side gate code 4411.', ownerFacts: ['Fortnightly mow'], facts: ['Prefers mornings'], updatedAt: 0 },
    { number: TOM, name: 'Tom Nguyen', nameBy: 'agent', facts: ['Large yard in Annandale'], updatedAt: 0 },
  ];
  const project = {
    loadSessions: async () => index,
    saveSessions: async (list: SessionInfo[]) => void (index = list),
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => void chats.set(id, turns),
    loadCallers: async () => callers,
    saveCallers: async (list: CallerNote[]) => void (callers = list),
  };
  const log: string[] = [];
  const said: string[] = [];
  let dialN = 0;
  const desktop = {
    say: async (_callId: string, text: string) => void said.push(text),
    finishCall: async (_callId: string, goodbye: string) => {
      said.push(`(goodbye) ${goodbye}`);
      return { ok: true, output: {} };
    },
    callTool: async () => ({ ok: true, output: { recorded: true } }),
    command: async (_c: string, command: string) => {
      log.push(command);
      if (command === 'call.dial') return { callId: `call_out_${++dialN}`, operationId: `op_${dialN}`, dialsToday: dialN, maxDailyDials: 20 };
      return { accepted: true };
    },
    calendarFree: async () => 'Fri 2 Oct: open 8–5; free 8–12, 1–5.',
  };
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: false, calls: true, instructions: '', callInstructions: 'Never quote a price for hedges.' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
    () => '# Green Lawns\nWe mow, edge and tidy lawns in the inner west. Bookings are requests: staff confirm them by text.',
  );
  sessions.identity = () => GREEN;
  const saved = new Map<string, Campaign>();
  let dnc: DoNotContact[] = [];
  const line = new PhoneLine();
  const outreach = new Outreach({
    store: { loadOutreach: async () => [...saved.values()], saveOutreach: async (c: Campaign) => void saved.set(c.id, JSON.parse(JSON.stringify(c)) as Campaign), loadDoNotContact: async () => dnc, saveDoNotContact: async (l: DoNotContact[]) => void (dnc = l) },
    files: () => new Vfs(),
    desktop: () => desktop as unknown as Desktop,
    phone: () => ({ holdsCalls: true, holdsTexts: true, connected: true }),
    line,
    callbacks: () => null,
    screening: async () => null,
    callsToOaiy: async () => true,
    rules: async () => ({ quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true }),
    sessions: () => sessions.forOutreach(),
    post: () => true,
    report: () => {},
    identity: () => GREEN,
    now: () => new Date(2026, 8, 29, 14, 0).getTime(),
  });
  sessions.outreach = outreach;
  return { sessions, outreach, line, log, said };
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
}

const until = async (ok: () => boolean) => {
  for (let i = 0; i < 200 && !ok(); i++) await new Promise((r) => setTimeout(r, 5));
};

/** A turn's system message and tools: what the engine keeps read from call to call. */
const fixed = (b: Record<string, unknown>) => JSON.stringify({ system: (b.messages as Array<{ content: unknown }>)[0], tools: b.tools });
const system = (b: Record<string, unknown>) => String((b.messages as Array<{ role: string; content: unknown }>)[0].content);
const firstNote = (b: Record<string, unknown>) => String((b.messages as Array<{ role: string; content: unknown }>)[1].content);
const turns = (sent: Sent[]) => sent.filter((s) => s.body.max_tokens !== WARM_TOKENS).map((s) => s.body);
const warms = (sent: Sent[]) => sent.filter((s) => s.body.max_tokens === WARM_TOKENS).map((s) => s.body);

const CAMPAIGN = {
  kind: 'call',
  name: 'Confirm Friday booking',
  objective: 'Confirm they still want their booking on Friday 2 October.',
  openingLine: "Hi {first_name}, it's {receptionist} from {business} about your booking this Friday. Have you got a minute?",
  collect: [{ key: 'keeping', question: 'Still keeping the Friday booking?', type: 'yes_no' }],
  people: [
    { name: 'Jane Smith', number: '0412 345 678', notes: 'Booked Fri 2 Oct 10 am', fields: { booking: 'Fri 2 Oct, 10:00' } },
    { name: 'Tom Nguyen', number: '0498 765 432', notes: 'Booked Fri 2 Oct 1 pm', fields: { booking: 'Fri 2 Oct, 13:00' } },
  ],
};

describe("a call's prompt: the same instructions and tools on every call, what is the call's own after them", () => {
  it('two inbound calls from different people share their whole system message and tools; the brief, who and what is known are in the note', async () => {
    const sent = engine(() => ({ text: 'Sure, Friday morning works.' }));
    const { sessions } = world();
    await sessions.load();
    for (const [callId, from, brief] of [['call_a', JANE, 'You are Aokie. Jane rang last week.'], ['call_b', TOM, 'You are Aokie, the receptionist.']] as const) {
      await sessions.callEvent({ type: 'call.started', callId, from, instructions: brief, greeting: 'Thanks for calling Green Lawns.' });
      await sessions.callEvent({ type: 'call.caller', callId, text: 'Can I get a mow on Friday?' });
      await settled(sessions);
      await sessions.callEvent({ type: 'call.ended', callId });
    }
    const [jane, tom] = turns(sent);
    // The engine keeps a prompt read up to the end of its system message: the same, byte for byte.
    expect(fixed(jane)).toBe(fixed(tom));
    expect(system(jane).length).toBeGreaterThan(3000);
    for (const own of ['Jane', 'Tom', 'Side gate code 4411', 'Prefers mornings', 'The receptionist brief', 'rang last week', 'Today is ']) expect(system(jane)).not.toContain(own);
    expect(firstNote(jane)).toContain('The receptionist brief:\nYou are Aokie. Jane rang last week.');
    expect(firstNote(jane)).toContain('Side gate code 4411');
    expect(firstNote(tom)).toContain('The receptionist brief:\nYou are Aokie, the receptionist.');
    // The warm as the call began read the same system message and tools.
    for (const warm of warms(sent)) expect(fixed(warm)).toBe(fixed(jane));
  });

  it("two people of one outreach list: the same system message and tools; who they are and why you rang are in each call's note", async () => {
    const sent = engine(() => ({ text: 'Great, thanks.' }));
    const { sessions, outreach, line } = world();
    await sessions.load();
    const plan = outreach.plan(CAMPAIGN, null);
    if (typeof plan === 'string') throw new Error(plan);
    const c = await outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    for (let i = 0; i < 2; i++) {
      await outreach.tick();
      const p = outreach.get(c.id)!.people[i];
      const callId = p.attempt!.callId!;
      await sessions.callEvent({ type: 'call.started', callId, from: p.number, direction: 'outbound', instructions: 'You are Aokie.', greeting: `Hi ${p.name.split(' ')[0]}, it's Aokie.` });
      await sessions.callEvent({ type: 'call.caller', callId, text: 'Yeah, sure. Yeah, of course, yes.' });
      await settled(sessions);
      await sessions.callEvent({ type: 'call.ended', callId });
      const ended: DesktopEvent = { seq: 1, name: 'aokie.call.ended', source: 'aokie', correlationId: callId, idempotencyKey: '', occurredAt: '', data: { callId, outcome: 'completed', direction: 'outbound' } };
      line.event(ended);
      line.calmAt = 0;
      await outreach.event(ended);
      await settled(sessions);
    }
    const jane = turns(sent).find((b) => firstNote(b).startsWith('[OAIY] 📞 You rang Jane'))!;
    const tom = turns(sent).find((b) => firstNote(b).startsWith('[OAIY] 📞 You rang Tom'))!;
    expect(fixed(jane)).toBe(fixed(tom));
    expect((jane.tools as Array<{ function: { name: string } }>).map((t) => t.function.name)).toContain('record_result');
    for (const own of ['Jane', 'Tom', 'Booked Fri', 'This is a call YOU placed', 'Why you rang']) expect(system(jane)).not.toContain(own);
    expect(firstNote(jane)).toContain('This is a call YOU placed: you are calling on behalf of Green Lawns, as Aokie');
    expect(firstNote(jane)).toContain('Who: Jane Smith (0412 345 678). Booked Fri 2 Oct 10 am.');
    expect(firstNote(jane)).toContain('Why you rang: Confirm they still want their booking on Friday 2 October.');
    expect(firstNote(tom)).toContain('Who: Tom Nguyen (0498 765 432).');
    // Confirming what you rang about needs no availability check.
    expect(firstNote(jane)).toContain('needs only record_result, no availability check');
    // The warm as each dial went out read the same system message and tools as the calls' turns.
    expect(warms(sent).length).toBeGreaterThanOrEqual(2);
    for (const warm of warms(sent)) expect(fixed(warm)).toBe(fixed(jane));
  });
});

describe('the engine reads a call before it is answered', () => {
  const incoming = (data: Record<string, unknown>, occurredAt = new Date().toISOString()): DesktopEvent => ({ seq: 1, name: 'aokie.call.incoming', source: 'aokie', correlationId: String(data.callId ?? ''), idempotencyKey: '', occurredAt, data });

  it('as it rings in: the same system message and tools as its first reply, before the call begins', async () => {
    const sent = engine(() => ({ text: 'Hi Jane! Friday morning is free.' }));
    const { sessions } = world();
    await sessions.load();
    await sessions.desktopEvent(incoming({ callId: 'call_r', from: JANE, name: 'Jane' }));
    await until(() => sent.length === 1);
    expect(sent).toHaveLength(1);
    expect(sent[0].body.max_tokens).toBe(WARM_TOKENS);
    // Nothing kept or listed for it: the call's own conversation is made as it begins.
    expect(sessions.list).toHaveLength(0);
    await sessions.callEvent({ type: 'call.started', callId: 'call_r', from: JANE, instructions: 'You are Aokie.', greeting: 'Thanks for calling.' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_r', text: 'Hi, is Friday morning free?' });
    await settled(sessions);
    const [turn] = turns(sent);
    expect(fixed(sent[0].body)).toBe(fixed(turn));
  });

  it('not while this page does not answer the calls, a call is going on, or the ring was heard of late; one that ends unanswered lets go', async () => {
    // The engine is slow to read it: the warm is still going when the call ends.
    const sent = engine(() => ({ text: 'Hello.', delayMs: 5_000 }));
    const { sessions } = world();
    await sessions.load();
    sessions.answersCalls = () => false;
    await sessions.desktopEvent(incoming({ callId: 'call_x', from: JANE }));
    sessions.answersCalls = () => true;
    await sessions.desktopEvent(incoming({ callId: 'call_y', from: JANE }, new Date(Date.now() - 60_000).toISOString()));
    await new Promise((r) => setTimeout(r, 20));
    expect(sent).toHaveLength(0);
    // Rings, and is missed: the warm is let go.
    await sessions.desktopEvent(incoming({ callId: 'call_z', from: TOM }));
    await until(() => sent.length === 1);
    expect(sent[0].signal?.aborted).toBe(false);
    await sessions.desktopEvent({ seq: 2, name: 'aokie.call.ended', source: 'aokie', correlationId: 'call_z', idempotencyKey: '', occurredAt: '', data: { callId: 'call_z', outcome: 'missed' } });
    await until(() => !!sent[0].signal?.aborted);
    expect(sent[0].signal?.aborted).toBe(true);
    // A call going on here: a second ringing in (call waiting) is not warmed.
    await sessions.callEvent({ type: 'call.started', callId: 'call_live', from: JANE });
    await new Promise((r) => setTimeout(r, 20));
    const before = sent.length;
    await sessions.desktopEvent(incoming({ callId: 'call_w', from: TOM }));
    await new Promise((r) => setTimeout(r, 20));
    expect(sent).toHaveLength(before);
  });

  it("as an outreach dial goes out: warmed as the dial is made, with the call's own tools (record_result, a silent end_call)", async () => {
    const { sessions, outreach, log } = world();
    const sent = engine(() => ({ text: 'Hi.' }));
    const warmCall = sessions.warmCall.bind(sessions);
    sessions.warmCall = (call) => {
      log.push('warm');
      warmCall(call);
    };
    await sessions.load();
    const plan = outreach.plan(CAMPAIGN, null);
    if (typeof plan === 'string') throw new Error(plan);
    await outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    await outreach.tick();
    expect(log).toEqual(['warm', 'call.dial']);
    await until(() => sent.length === 1);
    expect(sent[0].body.max_tokens).toBe(WARM_TOKENS);
    expect(firstNote(sent[0].body)).toContain('Who: Jane Smith');
    const tools = (sent[0].body.tools as Array<{ function: { name: string; parameters: { properties: Record<string, unknown> } } }>).map((t) => t.function);
    expect(tools.map((t) => t.name)).toContain('record_result');
    expect(Object.keys(tools.find((t) => t.name === 'end_call')!.parameters.properties)).toEqual(['goodbye', 'silent']);
  });

  it('as a call back is dialled: the hook is told the number before the dial', async () => {
    let saved: Callback[] = [];
    const order: string[] = [];
    const desktop = { command: async (_c: string, command: string) => void order.push(command) };
    const settings: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, callBack: true, callBackFilter: 'any' };
    const callbacks = new Callbacks({ loadCallbacks: async () => saved, saveCallbacks: async (l: Callback[]) => void (saved = l) } as never, () => settings, () => desktop as unknown as Desktop, () => true, async () => null, () => {}, async () => true, () => false, (number, purpose) => order.push(`warm ${number}: ${purpose.split(':')[0]}`));
    const t0 = Date.now();
    await callbacks.missed('0412345678', t0);
    await callbacks.tick(t0 + 120_000);
    expect(order).toEqual([expect.stringMatching(/^warm 0412345678: Returning their missed call from/), 'call.dial']);
  });

  it('a warm lets go at the model\'s first word (the engine then ends it cleanly), and asks for a few words, not one', async () => {
    // Live 29 Sept 2026: a one-token warm whose token was <tool_call> failed in OAIY's engine, which then
    // forgot the prompt it had read: the first reply read all 3,900 tokens again (6.7 s).
    const sent = engine(() => ({ text: 'Great', hold: true }));
    const { sessions } = world();
    await sessions.load();
    await sessions.callEvent({ type: 'call.started', callId: 'call_w', from: JANE, instructions: 'You are Aokie.' });
    await until(() => sent[0]?.cancelled === true);
    expect(sent).toHaveLength(1);
    expect(WARM_TOKENS).toBeGreaterThan(1);
    expect(sent[0].body.max_tokens).toBe(WARM_TOKENS);
    expect(sent[0].cancelled).toBe(true);
    expect(sent[0].signal?.aborted).toBe(true);
  });
});

describe('on ChatGPT (the live-call route), nothing is warmed', () => {
  it('no request at ring, dial or the call beginning; the hold word still covers a slow reply', async () => {
    const chatgpt = chatgptProvider({ origin: 'http://127.0.0.1:17972', token: 't' }, 'call', null);
    const sent = engine(() => ({ text: 'Friday is free.', delayMs: 120 }));
    const { sessions, outreach, said } = world(chatgpt);
    sessions.holdWordAfterMs = 30;
    await sessions.load();
    await sessions.desktopEvent({ seq: 1, name: 'aokie.call.incoming', source: 'aokie', correlationId: 'call_c', idempotencyKey: '', occurredAt: new Date().toISOString(), data: { callId: 'call_c', from: JANE } });
    const plan = outreach.plan({ ...CAMPAIGN, people: CAMPAIGN.people.slice(1) }, null);
    if (typeof plan === 'string') throw new Error(plan);
    await outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    await outreach.tick();
    await sessions.callEvent({ type: 'call.started', callId: 'call_c', from: JANE, greeting: 'Thanks for calling.' });
    await new Promise((r) => setTimeout(r, 30));
    expect(sent).toHaveLength(0);
    await sessions.callEvent({ type: 'call.said', callId: 'call_c', text: 'Thanks for calling.', startMs: 1500, endMs: 2600 });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_c', text: 'Is Friday free?', startMs: 3000, endMs: 4000 });
    await settled(sessions);
    expect(sent).toHaveLength(1);
    expect(said).toEqual([HOLD_WORDS[0], 'Friday is free.']);
  });
});

describe('the hold word', () => {
  /** A call begun, its greeting played; the caller's words answered by `answer`. */
  async function call(answer: Answer | Answer[], opts: { greeted?: boolean; callId?: string } = {}) {
    const script = Array.isArray(answer) ? answer : [answer];
    const sent = engine((_b, n) => script[Math.min(n, script.length - 1)]);
    const w = world({ ...ENGINE, type: 'openai', serverKind: undefined });
    w.sessions.holdWordAfterMs = 40;
    await w.sessions.load();
    const callId = opts.callId ?? 'call_h';
    await w.sessions.callEvent({ type: 'call.started', callId, from: JANE, greeting: 'Thanks for calling Green Lawns.' });
    if (opts.greeted !== false) await w.sessions.callEvent({ type: 'call.said', callId, text: 'Thanks for calling Green Lawns.', startMs: 1500, endMs: 3200 });
    const say = async (text: string, startMs = 4000) => {
      await w.sessions.callEvent({ type: 'call.caller', callId, text, startMs, endMs: startMs + 900 });
      await settled(w.sessions);
    };
    return { ...w, sent, say, callId };
  }

  it('the constant: about a second and a half, a few short words to rotate', () => {
    expect(HOLD_WORD_AFTER_MS).toBe(1_500);
    expect(HOLD_WORDS.length).toBeGreaterThanOrEqual(3);
    for (const w of HOLD_WORDS) expect(w.split(' ').length).toBeLessThanOrEqual(2);
  });

  it('a slow reply gets one, once; a filler that opens the reply after it is not said; the next turn takes the next word', async () => {
    const { say, said } = await call([{ text: 'Sure! Friday morning is free. Would ten suit?', delayMs: 150 }, { text: 'Ten on Friday it is.', delayMs: 150 }]);
    await say('Is Friday morning free?');
    expect(said).toEqual([HOLD_WORDS[0], 'Friday morning is free.', 'Would ten suit?']);
    await say('Yes, ten is good.', 9000);
    expect(said).toEqual([HOLD_WORDS[0], 'Friday morning is free.', 'Would ten suit?', HOLD_WORDS[1], 'Ten on Friday it is.']);
  });

  it('a quick reply gets none', async () => {
    const { say, said } = await call({ text: 'Friday morning is free.' });
    await say('Is Friday morning free?');
    expect(said).toEqual(['Friday morning is free.']);
  });

  it('none on the greeting: words said before it had played, or over it', async () => {
    const early = await call({ text: 'Hi there, how can I help?', delayMs: 150 }, { greeted: false });
    await early.say('Hello?', 500);
    expect(early.said).toEqual(['Hi there, how can I help?']);
    const over = await call({ text: 'Hi there, how can I help?', delayMs: 150 });
    await over.say('Hello? Hello?', 2000);
    expect(over.said).toEqual(['Hi there, how can I help?']);
  });

  it('none when a tool comes first (its own "one moment" covers the wait), and none after end_call', async () => {
    const tool = await call([{ calls: [{ name: 'remember', input: { fact: 'Prefers mornings' } }] }, { text: 'Noted. Anything else?', delayMs: 150 }]);
    await tool.say('Mornings suit me best.');
    expect(tool.said).toEqual(['Noted.', 'Anything else?']);
    const bye = await call([{ calls: [{ name: 'end_call', input: { goodbye: 'Bye, Jane!' } }] }, { text: 'Done.', delayMs: 150 }]);
    await bye.say("That's all, thanks.");
    expect(bye.said).toEqual(['(goodbye) Bye, Jane!']);
  });

  it('none once the caller speaks again, or after the call has ended', async () => {
    const again = await call({ text: 'Friday is free.', delayMs: 150 });
    await again.sessions.callEvent({ type: 'call.caller', callId: again.callId, text: 'Is Friday free?', startMs: 4000, endMs: 4900 });
    await again.sessions.callEvent({ type: 'call.speech_started', callId: again.callId, atMs: 5000 });
    await settled(again.sessions);
    expect(again.said).toEqual(['Friday is free.']);
    const gone = await call({ text: 'Friday is free.', delayMs: 150 });
    await gone.sessions.callEvent({ type: 'call.caller', callId: gone.callId, text: 'Is Friday free?', startMs: 4000, endMs: 4900 });
    await gone.sessions.callEvent({ type: 'call.ended', callId: gone.callId });
    await settled(gone.sessions);
    expect(gone.said).toEqual([]);
  });

  it('Speech: once a turn, before the reply has said anything, not its own words (the tool line may follow)', async () => {
    const out: string[] = [];
    const speech = new Speech(async (t) => void out.push(t));
    speech.begin();
    expect(speech.holdWord('Okay —')).toBe(true);
    expect(speech.holdWord('Sure,')).toBe(false);
    // Not the reply's words: what the caller heard of the reply, when they cut in, is the model's own.
    expect(speech.reply).toEqual([]);
    // A tool's line still follows it.
    speech.hold('One moment, let me check.');
    speech.begin(true);
    expect(speech.holdWord('Sure,')).toBe(false);
    speech.push('We have Friday free.');
    speech.flush();
    // A new turn: one again, and a filler that opens the reply after it is not said.
    speech.begin();
    expect(speech.holdWord('Mm, right.')).toBe(true);
    speech.push('Sure! Ten it is.');
    speech.flush();
    // Nothing after the reply has begun.
    speech.begin();
    speech.push('Done.');
    speech.flush();
    expect(speech.holdWord('Okay —')).toBe(false);
    await speech.done;
    expect(out).toEqual(['Okay —', 'One moment, let me check.', 'We have Friday free.', 'Mm, right.', 'Ten it is.', 'Done.']);
  });
});

