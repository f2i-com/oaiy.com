// Outreach and the phone's conversations together: an outreach call's agent
// has the objective and record_result, cannot end the call before recording
// (once), hangs up on a voicemail without a word, and records the result
// after a call that dropped (nothing spoken); a text's reply is answered with
// the same, even while answering is off, and STOP is kept, never answered.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import { Outreach, type Campaign, type DoNotContact } from '../../src/outreach';
import { PhoneLine } from '../../src/phoneLine';
import { setLocalCountry } from '../../src/phoneNumbers';
import { Sessions } from '../../src/sessions';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';
import { OPENAI, fakeProvider } from './fakeProvider';

beforeAll(() => setLocalCountry('AU'));
afterEach(() => vi.unstubAllGlobals());

const T0 = new Date(2026, 8, 29, 10, 0).getTime();
const ev = (name: string, data: Record<string, unknown> = {}, correlationId = ''): DesktopEvent => ({ seq: 1, name, source: 'aokie', correlationId, idempotencyKey: '', occurredAt: '', data });

function setup() {
  const chats = new Map<string, Turn[]>();
  let index: SessionInfo[] = [];
  let callers: CallerNote[] = [];
  const project = {
    loadSessions: async () => index,
    saveSessions: async (list: SessionInfo[]) => void (index = list),
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => void chats.set(id, turns),
    loadCallers: async () => callers,
    saveCallers: async (list: CallerNote[]) => void (callers = list),
  };
  const calls: Array<[string, string, unknown]> = [];
  const commands: Array<{ command: string; payload: Record<string, unknown> }> = [];
  let dialN = 0;
  const desktop = {
    say: async (callId: string, text: string) => void calls.push(['say', callId, text]),
    finishCall: async (callId: string, goodbye: string) => {
      calls.push(['finish', callId, goodbye]);
      return { ok: true, output: {} };
    },
    callTool: async () => ({ ok: true, output: {} }),
    command: async (_c: string, command: string, payload: Record<string, unknown>) => {
      commands.push({ command, payload });
      if (command === 'call.dial') return { callId: `call_${++dialN}`, operationId: `op_${dialN}`, dialsToday: dialN, maxDailyDials: 20 };
      if (command === 'sms.send') return { messageId: payload.messageId ?? 'm', status: 'queued' };
      return { accepted: true };
    },
  };
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: false, calls: true, instructions: '', callInstructions: '' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
  );
  const saved = new Map<string, Campaign>();
  let dnc: DoNotContact[] = [];
  const phone = { holdsCalls: true, holdsTexts: true, connected: true };
  const outreach = new Outreach({
    store: {
      loadOutreach: async () => [...saved.values()],
      saveOutreach: async (c: Campaign) => void saved.set(c.id, JSON.parse(JSON.stringify(c)) as Campaign),
      loadDoNotContact: async () => dnc,
      saveDoNotContact: async (list: DoNotContact[]) => void (dnc = list),
    },
    files: () => new Vfs(),
    desktop: () => desktop as unknown as Desktop,
    phone: () => phone,
    line: new PhoneLine(),
    callbacks: () => null,
    screening: async () => null,
    callsToOaiy: async () => true,
    rules: async () => ({ quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true }),
    sessions: () => sessions.forOutreach(),
    post: () => true,
    report: () => {},
    identity: () => ({ business: 'Greenleaf Lawns', receptionist: 'Aokie' }),
    now: () => T0,
  });
  sessions.outreach = outreach;
  sessions.identity = () => ({ business: 'Greenleaf Lawns', receptionist: 'Aokie' });
  const start = async (input: Record<string, unknown>) => {
    const plan = outreach.plan(input, null);
    if (typeof plan === 'string') throw new Error(plan);
    const c = await outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
    await outreach.tick();
    return c;
  };
  return { sessions, outreach, calls, commands, messages, start, chats };
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
}

const CALL = {
  kind: 'call',
  name: 'Confirm Friday bookings',
  objective: 'Confirm they are still coming on Friday.',
  openingLine: "Hi {first_name}, it's Greenleaf Lawns about Friday. Have you got a minute?",
  collect: [{ key: 'coming', question: 'Still coming Friday?', type: 'yes_no' }],
  people: [{ name: 'Jane Smith', number: '0412 345 678' }],
};

const TEXT = {
  kind: 'text',
  name: 'Friday reminders',
  objective: 'Check they are still coming on Friday.',
  textTemplate: 'Hi {first_name}, {business} here: still right for Friday? Reply YES or NO.',
  collect: [{ key: 'coming', question: 'Still coming?', type: 'yes_no' }],
  people: [{ name: 'Jane Smith', number: '0412 345 678' }],
};

describe('an outreach call', () => {
  it('its agent has the objective and record_result; end_call is refused once until the result is recorded, the words held for it said', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const tools = (body.tools as Array<{ function: { name: string; parameters: { properties: Record<string, unknown> } } }>).map((t) => t.function);
        expect(tools.map((t) => t.name)).toEqual(expect.arrayContaining(['record_result', 'end_call']));
        expect(Object.keys(tools.find((t) => t.name === 'end_call')!.parameters.properties)).toEqual(['goodbye', 'silent']);
        const sent = JSON.stringify(body);
        expect(sent).toContain("This is a call YOU placed: you are calling on behalf of Greenleaf Lawns, as Aokie, for your person's outreach \\\"Confirm Friday bookings\\\"");
        // Every agent a customer talks to says who it is, for whom.
        expect(sent).toContain('You are Aokie, the receptionist for Greenleaf Lawns.');
        expect(sent).toContain('You rang Jane Smith (+61412345678); they answered');
        expect(sent).toContain('Why you rang: Confirm they are still coming on Friday.');
        return { text: 'Great, see you Friday.', calls: [{ name: 'end_call', input: { goodbye: 'Bye!' } }] };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('Not yet: record_result first');
        return { calls: [{ name: 'record_result', input: { outcome: 'completed', answers: { coming: true }, summary: 'Still coming Friday.' } }, { name: 'end_call', input: { goodbye: 'Thanks Jane, bye!' } }] };
      },
      { text: '' },
    ]);
    const { sessions, outreach, calls, start } = setup();
    const c = await start(CALL);
    expect(c.people[0].state).toBe('dialling');
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '+61412345678', direction: 'outbound', greeting: "Hi Jane, it's Greenleaf Lawns about Friday. Have you got a minute?" });
    expect(call?.outreach?.campaignId).toBe(c.id);
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Yes, still coming.' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(3);
    expect(calls).toEqual([['say', 'call_1', 'Great, see you Friday.'], ['finish', 'call_1', 'Thanks Jane, bye!']]);
    await outreach.event(ev('aokie.call.ended', { callId: 'call_1', outcome: 'completed', reason: 'agent_hangup', direction: 'outbound' }, 'call_1'));
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: true }, summary: 'Still coming Friday.' });
  });

  it('a voicemail with no message: hung up without a word, only once voicemail is recorded', async () => {
    fakeProvider('openai', [
      { calls: [{ name: 'record_result', input: { outcome: 'completed', summary: 'x' } }, { name: 'end_call', input: { silent: true } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('silent is only for a voicemail with no message');
        return { calls: [{ name: 'record_result', input: { outcome: 'voicemail', summary: 'Their voicemail greeting.' } }, { name: 'end_call', input: { silent: true } }] };
      },
      { text: '' },
    ]);
    const { sessions, outreach, calls, commands, start } = setup();
    const c = await start(CALL);
    await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '+61412345678', direction: 'outbound' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Hi, you have reached Jane. Leave a message after the tone.' });
    await settled(sessions);
    expect(commands.filter((x) => x.command === 'call.hangup')).toEqual([{ command: 'call.hangup', payload: { callId: 'call_1' } }]);
    expect(calls).toEqual([]);
    await outreach.event(ev('aokie.call.ended', { callId: 'call_1', outcome: 'completed', reason: 'local_hangup' }, 'call_1'));
    // Tries left: rung again later.
    expect(c.people[0]).toMatchObject({ state: 'waiting', outcome: undefined });
    expect(c.people[0].attempt?.hungUp).toBe(true);
  });

  it('a call that dropped before the result: its agent records it from what was said, and nothing is spoken', async () => {
    const fake = fakeProvider('openai', [
      { text: 'Great, thanks Jane!' },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('The call ended before you recorded the result');
        return { calls: [{ name: 'record_result', input: { outcome: 'completed', answers: { coming: 'yes' }, summary: 'Said yes, then the line dropped.' } }] };
      },
      { text: 'Recorded.' },
    ]);
    const { sessions, outreach, calls, start } = setup();
    const c = await start(CALL);
    await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '+61412345678', direction: 'outbound' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: "Yes, I'll be there." });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_1' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(3);
    expect(calls).toEqual([['say', 'call_1', 'Great, thanks Jane!']]);
    await outreach.event(ev('aokie.call.ended', { callId: 'call_1', outcome: 'completed', reason: 'remote_hangup' }, 'call_1'));
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: true } });
  });

  it('a call that is not an outreach has no record_result, and end_call no silent', async () => {
    fakeProvider('openai', [
      (body) => {
        const tools = (body.tools as Array<{ function: { name: string; parameters: { properties: Record<string, unknown> } } }>).map((t) => t.function);
        expect(tools.map((t) => t.name)).not.toContain('record_result');
        expect(Object.keys(tools.find((t) => t.name === 'end_call')!.parameters.properties)).toEqual(['goodbye']);
        return { text: 'Hello!' };
      },
    ]);
    const { sessions } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'call_x', from: '+61400000001' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_x', text: 'Hi' });
    await settled(sessions);
  });
});

describe('an outreach text', () => {
  it('lands in their conversation, and their reply is answered (while answering is off) with record_result', async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const sent = JSON.stringify(body);
        expect(sent).toContain("You texted them on behalf of Greenleaf Lawns, as Aokie, for your person's outreach \\\"Friday reminders\\\"");
        expect(sent).toContain('You are Aokie, the receptionist for Greenleaf Lawns.');
        expect(sent).toContain('[OAIY] Outreach \\"Friday reminders\\": you texted them');
        expect((body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name)).toContain('record_result');
        return { calls: [{ name: 'record_result', input: { outcome: 'completed', answers: { coming: 'yes' }, summary: 'Yes, Friday.' } }, { name: 'send_text_message', input: { body: 'Thanks Jane, see you Friday!' } }] };
      },
      { text: 'Recorded and replied.' },
    ]);
    const { sessions, outreach, commands, start } = setup();
    const c = await start(TEXT);
    const texts = () => commands.filter((x) => x.command === 'sms.send').map((x) => x.payload.body);
    expect(texts()).toEqual(['Hi Jane, Greenleaf Lawns here: still right for Friday? Reply YES or NO.']);
    const thread = sessions.threads().find((t) => t.key === '+61412345678')!;
    expect(c.people[0].thread).toBe(thread.id);
    await outreach.event(ev('aokie.sms.sent', { messageId: `oaiy-out.${c.id}.p1.1` }));
    await sessions.textArrived('0412345678', 'Jane', 'Yes, see you then');
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(texts()).toEqual(['Hi Jane, Greenleaf Lawns here: still right for Friday? Reply YES or NO.', 'Thanks Jane, see you Friday!']);
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: true } });
    // One conversation for her: the outreach note, her reply and the answer.
    expect(sessions.threads().filter((t) => t.key === '+61412345678')).toHaveLength(1);
  });

  it('STOP is kept, never answered, and they are opted out', async () => {
    const fake = fakeProvider('openai', []);
    const { sessions, outreach, commands, start } = setup();
    const c = await start(TEXT);
    const session = await sessions.textArrived('+61412345678', 'Jane', 'STOP');
    await settled(sessions);
    expect(fake.bodies).toHaveLength(0);
    expect(commands.filter((x) => x.command === 'sms.send')).toHaveLength(1);
    expect(session.agent.turns.at(-1)).toMatchObject({ role: 'user', text: 'Text message from Jane Smith (+61412345678):\nSTOP' });
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'opted_out' });
    expect(outreach.doNotContact.map((d) => d.number)).toEqual(['+61412345678']);
  });

  it("someone not on an outreach is not answered while answering is off", async () => {
    const fake = fakeProvider('openai', []);
    const { sessions, start } = setup();
    await start(TEXT);
    await sessions.textArrived('+61499999999', 'Someone', 'Hello?');
    await settled(sessions);
    expect(fake.bodies).toHaveLength(0);
  });
});
