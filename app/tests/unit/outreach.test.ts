import { beforeAll, describe, expect, it } from 'vitest';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import type { Screening } from '../../src/callbacks';
import { setLocalCountry } from '../../src/phoneNumbers';
import { PhoneLine } from '../../src/phoneLine';
import { Vfs } from '../../src/vfs/vfs';
import {
  AFTER_CALL_MS, DIAL_LOST_MS, OUTREACH_AFTER_CALL, Outreach, SMS_ACK_MS, TEXT_GAP_MS, csvCell, fill, mdCell, planOutreach,
  resultsCsv, resultsMarkdown, type Campaign, type DoNotContact, type PhoneRules,
} from '../../src/outreach';

beforeAll(() => setLocalCountry('AU'));

/** Tuesday 29 September 2026, 10:00 local: inside the calling window. */
const T0 = new Date(2026, 8, 29, 10, 0).getTime();
const MIN = 60_000;

const ev = (name: string, data: Record<string, unknown> = {}, correlationId = ''): DesktopEvent => ({ seq: 1, name, source: 'aokie', correlationId, idempotencyKey: '', occurredAt: '', data });

const CALLS = {
  kind: 'call',
  name: 'Confirm Friday bookings',
  objective: "Confirm they're still coming on Friday 2 Oct, or find a better time.",
  openingLine: "Hi {first_name}, it's Greenleaf Lawns about your {service} on {appointment}. Have you got a minute?",
  collect: [
    { key: 'coming', question: 'Still coming Friday?', type: 'yes_no' },
    { key: 'new_time', question: 'If not, what day and time suits?', optional: true },
  ],
  people: [
    { name: 'Jane Smith', number: '0412 345 678', fields: { service: 'lawn mow', appointment: 'Fri 2 Oct 10:30' } },
    { name: 'Bob Jones', number: '+61 413 000 111', fields: { service: 'hedge trim', appointment: 'Fri 2 Oct 1:00' } },
    { name: 'Cara Lee', number: '0414 222 333', fields: { service: 'lawn mow', appointment: 'Fri 2 Oct 3:00' } },
  ],
};

const TEXTS = {
  kind: 'text',
  name: 'Friday reminders',
  objective: 'Check they are still coming on Friday.',
  textTemplate: 'Hi {first_name}, Greenleaf Lawns here: still right for Friday? Reply YES or NO.',
  collect: [{ key: 'coming', question: 'Still coming?', type: 'yes_no' }],
  people: [
    { name: 'Jane Smith', number: '0412345678' },
    { name: 'Bob Jones', number: '0413000111' },
  ],
};

function rig(o: { screening?: Screening | null; rules?: Partial<PhoneRules>; refuse?: string; route?: boolean | null } = {}) {
  let clock = T0;
  const saved = new Map<string, Campaign>();
  let ids: string[] = [];
  let dnc: DoNotContact[] = [];
  const store = {
    loadOutreach: async () => ids.map((id) => JSON.parse(JSON.stringify(saved.get(id))) as Campaign),
    saveOutreach: async (c: Campaign, list: string[]) => {
      saved.set(c.id, JSON.parse(JSON.stringify(c)) as Campaign);
      ids = list;
    },
    loadDoNotContact: async () => [...dnc],
    saveDoNotContact: async (list: DoNotContact[]) => void (dnc = [...list]),
  };
  const commands: Array<{ command: string; payload: Record<string, unknown>; key: string }> = [];
  const state = { refuse: o.refuse ?? '', dialN: 0 };
  const desktop = {
    command: async (_c: string, command: string, payload: Record<string, unknown>, key: string) => {
      if (state.refuse && command === 'call.dial') throw new Error(state.refuse);
      commands.push({ command, payload, key });
      if (command === 'call.dial') {
        state.dialN++;
        return { accepted: true, callId: `call_${state.dialN}`, operationId: `op_${state.dialN}`, dialsToday: state.dialN, maxDailyDials: 20 };
      }
      if (command === 'sms.send') return { messageId: payload.messageId, status: 'queued' };
      return { accepted: true };
    },
  };
  const phone = { holdsCalls: true, holdsTexts: true, connected: true as boolean | null };
  const line = new PhoneLine();
  const callbacks = { calling: false, due: false };
  const files = new Vfs();
  const posted: string[] = [];
  const reports: Campaign[] = [];
  const texted: Array<{ number: string; note: string }> = [];
  const asked: string[] = [];
  const heard = new Map<string, string[]>();
  const live = new Set<string>();
  const rules: PhoneRules = { quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true, ...o.rules };
  const deps = {
    store,
    files: () => files,
    desktop: () => desktop as unknown as Desktop,
    phone: () => phone,
    line,
    callbacks: () => ({ ringing: () => callbacks.calling, due: () => callbacks.due }),
    screening: async () => (o.screening === undefined ? { acceptPattern: '', blockedNumbers: '', rejectPrivate: false } : o.screening),
    callsToOaiy: async () => (o.route === undefined ? true : o.route),
    rules: async () => rules,
    sessions: () => ({
      openText: async (number: string, _name: string, note: string) => {
        texted.push({ number, note });
        return `person-${number.replace(/\W/g, '')}`;
      },
      liveCall: (callId: string) => live.has(callId),
      heardSince: (number: string) => heard.get(number) ?? [],
      askForResult: (number: string) => void asked.push(number),
    }),
    post: (_c: Campaign, text: string) => {
      posted.push(text);
      return true;
    },
    report: (c: Campaign) => void reports.push(c),
    now: () => clock,
  };
  const outreach = new Outreach(deps);
  const start = async (input: Record<string, unknown>) => {
    const plan = outreach.plan(input, await deps.screening());
    if (typeof plan === 'string') throw new Error(plan);
    return outreach.create(plan, { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' });
  };
  const at = (ms: number) => (clock = ms);
  const later = (ms: number) => (clock += ms);
  const dials = () => commands.filter((c) => c.command === 'call.dial');
  /** A dial's whole call, as the phone tells it. */
  const ended = (callId: string, outcome: string, reason = 'remote_hangup') => outreach.event(ev('aokie.call.ended', { callId, outcome, reason, direction: 'outbound' }, callId));
  const endOnLine = (callId: string, outcome: string, reason?: string) => {
    line.event(ev('aokie.call.ended', { callId, outcome }, callId), clock);
    return ended(callId, outcome, reason);
  };
  return { outreach, deps, store, commands, dials, phone, line, callbacks, files, posted, reports, texted, asked, heard, live, rules, state, start, at, later, ended, endOnLine, now: () => clock, dnc: () => dnc };
}

/** The calm after a call, and a bit. */
const CALM = 61_000;

describe('planning an outreach', () => {
  it('reads each number as a person, merges the same person, and leaves out who may not be contacted, saying why', () => {
    const plan = planOutreach({
      ...CALLS,
      people: [
        { name: 'Jane Smith', number: '0412 345 678', fields: { service: 'lawn mow', appointment: 'Fri' } },
        { name: 'Jane S', number: '+61412345678', fields: { gate: '4821' } },
        { name: 'Short', number: '0412 000', fields: { service: 'x', appointment: 'y' } },
        { name: 'Blocked', number: '0400 000 004', fields: { service: 'x', appointment: 'y' } },
        { name: 'Asked', number: '0400 000 005', fields: { service: 'x', appointment: 'y' } },
        { name: 'Abroad', number: '+1 415 555 0100', fields: { service: 'x', appointment: 'y' } },
      ],
    }, {
      screening: { acceptPattern: '^\\s*(\\+?61|\\(?0[1-9])', blockedNumbers: '0400000004', rejectPrivate: false },
      doNotContact: [{ number: '+61400000005', at: 0, why: 'texted STOP' }],
      inTextCampaign: () => null,
      slugs: new Set(),
    });
    if (typeof plan === 'string') throw new Error(plan);
    expect(plan.people.map((p) => [p.name, p.number, p.fields])).toEqual([['Jane Smith', '+61412345678', { service: 'lawn mow', appointment: 'Fri', gate: '4821' }]]);
    expect(plan.merged).toBe(1);
    expect(plan.skipped.map((s) => [s.name, s.why])).toEqual([
      ['Short', 'not a full phone number'],
      ['Blocked', "on the phone's blocked list"],
      ['Asked', 'asked not to be contacted'],
      ['Abroad', 'not a number the phone answers (its filter)'],
    ]);
    expect(plan.slug).toBe('confirm-friday-bookings');
    expect(plan.resultsPath).toBe('/outreach/confirm-friday-bookings/results.md');
    expect(plan.retries).toEqual({ times: 2, gapMinutes: 60 });
    expect(plan.window).toEqual({ from: '09:00', to: '19:00' });
  });

  it('refuses a placeholder that has no value for someone, an opening line that is too long, and results outside /outreach', () => {
    const ctx = { screening: null, doNotContact: [], inTextCampaign: () => null, slugs: new Set<string>() };
    expect(planOutreach({ ...CALLS, people: [{ name: 'Jane', number: '0412345678', fields: { service: 'mow' } }] }, ctx)).toMatch(/\{appointment\} has no value for Jane/);
    expect(planOutreach({ ...CALLS, openingLine: `Hi {first_name}, ${'very '.repeat(120)}long.`, people: [{ name: 'Jane', number: '0412345678' }] }, ctx)).toMatch(/under 500/);
    expect(planOutreach({ ...CALLS, resultsPath: '/knowledge/results.md' }, ctx)).toMatch(/under \/outreach/);
    expect(planOutreach({ ...CALLS, openingLine: '' }, ctx)).toMatch(/openingLine is needed/);
    expect(planOutreach({ ...CALLS, people: [{ name: 'X', number: 'hello' }] }, ctx)).toMatch(/No one left to call: X: not a full phone number/);
    expect(fill('Hi {first_name}, your {Service}', { name: 'Jane Smith', fields: { service: 'mow' } })).toEqual({ text: 'Hi Jane, your mow', missing: [] });
  });

  it('a text campaign leaves out someone still waiting on a reply to another', async () => {
    const r = rig();
    await r.start(TEXTS);
    const plan = r.outreach.plan({ ...TEXTS, name: 'Another', people: [{ name: 'Jane', number: '+61412345678' }, { name: 'Dee', number: '0415000999' }] }, null);
    if (typeof plan === 'string') throw new Error(plan);
    expect(plan.people.map((p) => p.name)).toEqual(['Dee']);
    expect(plan.skipped[0].why).toBe('already texted in "Friday reminders", waiting for their reply');
  });
});

describe('calling down a list', () => {
  it('dials one at a time, with the filled opening line and a key that makes a repeat harmless', async () => {
    const r = rig();
    const c = await r.start(CALLS);
    await r.outreach.tick();
    expect(r.dials()).toEqual([{ command: 'call.dial', key: `oaiy:outreach:${c.id}:p1:1`, payload: { number: '+61412345678', openingLine: "Hi Jane, it's Greenleaf Lawns about your lawn mow on Fri 2 Oct 10:30. Have you got a minute?", purpose: expect.stringContaining('Confirm Friday bookings: Confirm') } }]);
    // The line is ours until it ends: no second dial.
    r.later(30_000);
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(1);
    expect(c.people[0]).toMatchObject({ state: 'dialling', tries: 1, attempt: { callId: 'call_1', operationId: 'op_1' } });
  });

  it('waits while the line is busy, a call goes on here, or a missed call is ringing back or due, and a minute after any call ends', async () => {
    const r = rig();
    r.line.event(ev('aokie.call.incoming', { from: '+61499999999' }, 'call_in'), r.now());
    await r.start(CALLS);
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    r.line.event(ev('aokie.call.ended', { callId: 'call_in' }, 'call_in'), r.now());
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    r.later(CALM);
    r.callbacks.due = true;
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    expect(r.outreach.campaigns[0].waitingFor).toBe('a missed call being rung back first');
    r.callbacks.due = false;
    r.callbacks.calling = true;
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    r.callbacks.calling = false;
    r.phone.holdsCalls = false;
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    expect(r.outreach.campaigns[0].waitingFor).toBe('this page to answer the calls');
    r.phone.holdsCalls = true;
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(1);
  });

  it("waits out Aokie's quiet hours and its window, and says what it waits for", async () => {
    const r = rig({ rules: { quietStart: 21, quietEnd: 8 } });
    r.at(new Date(2026, 8, 29, 7, 30).getTime());
    await r.start(CALLS);
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(0);
    expect(r.outreach.campaigns[0].waitingFor).toBe('quiet hours until 8:00');
    r.at(new Date(2026, 8, 29, 8, 30).getTime());
    await r.outreach.tick();
    expect(r.outreach.campaigns[0].waitingFor).toBe('its window, from 9:00');
    r.at(new Date(2026, 8, 29, 9, 1).getTime());
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(1);
  });

  it("what the phone's refusal says decides the next try: quiet hours, the daily cap and a busy line cost nothing; outbound off pauses; a bad number is done", async () => {
    const cases: Array<[string, (c: Campaign) => void]> = [
      ['quiet hours: automated calls are not placed between 21:00 and 8:00 local time', (c) => expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0 })],
      ['daily dial cap reached (20/20 today) — raise maxDailyDials or try tomorrow', (c) => {
        expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0 });
        expect(c.people[0].nextAt).toBeGreaterThan(T0 + 12 * 60 * MIN);
      }],
      ['a call is already in progress — outbound dialing needs an idle line', (c) => expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0, nextAt: T0 + 60_000 })],
      ['outbound calling is OFF — the operator must set outboundEnabled: true', (c) => expect(c).toMatchObject({ state: 'paused', pausedWhy: 'outbound calling is off on the phone' })],
      ['no phone is connected — reconnect the phone before dialing', (c) => expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0 })],
      ['number: must be 3-32 digits', (c) => expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'invalid_number', tries: 1 })],
      ['something else broke', (c) => expect(c.people[0]).toMatchObject({ state: 'waiting', tries: 1, nextAt: T0 + 60 * MIN })],
    ];
    for (const [refusal, expectation] of cases) {
      const r = rig({ refuse: refusal });
      const c = await r.start(CALLS);
      await r.outreach.tick();
      expectation(c);
      // The line is free again after a refusal.
      expect(r.line.placed).toBeNull();
    }
  });

  it('a dial the radio dropped (a call came in first) is tried again shortly, and is not a try', async () => {
    const r = rig();
    const c = await r.start(CALLS);
    await r.outreach.tick();
    await r.outreach.event(ev('aokie.hardware.error', { code: 'control_failed', action: 'call.dial', operationId: 'op_1' }));
    expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0, nextAt: T0 + 60_000 });
  });
});

describe('how each call ends', () => {
  it('no answer: tried again after the gap, and after the last try left as no answer', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1), retries: { times: 1, gapMinutes: 30 } });
    await r.outreach.tick();
    await r.endOnLine('call_1', 'no_answer');
    expect(c.people[0]).toMatchObject({ state: 'waiting', tries: 1, nextAt: T0 + 30 * MIN });
    r.later(20 * MIN);
    await r.outreach.tick();
    expect(r.dials()).toHaveLength(1);
    r.later(11 * MIN);
    await r.outreach.tick();
    expect(r.dials().map((d) => d.key.split(':').at(-1))).toEqual(['1', '2']);
    await r.endOnLine('call_2', 'no_answer');
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'no_answer', tries: 2 });
    expect(r.posted.at(-1)).toBe('[OAIY] Outreach "Confirm Friday bookings" · Jane Smith: no answer. (1 of 1)');
  });

  it('failed: once more, then unreachable', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1) });
    await r.outreach.tick();
    await r.endOnLine('call_1', 'failed', 'setup_failed');
    expect(c.people[0]).toMatchObject({ state: 'waiting', failed: true });
    r.later(61 * MIN);
    await r.outreach.tick();
    await r.endOnLine('call_2', 'failed', 'setup_failed');
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'unreachable' });
  });

  it("the phone ending the dial itself (OAIY's voice did not start) twice in a row pauses it, and costs no try", async () => {
    const r = rig();
    const c = await r.start(CALLS);
    await r.outreach.tick();
    await r.endOnLine('call_1', 'no_answer', 'cancelled');
    expect(c.people[0]).toMatchObject({ state: 'queued', tries: 0 });
    expect(c.state).toBe('running');
    r.later(CALM);
    await r.outreach.tick();
    await r.endOnLine('call_2', 'no_answer', 'cancelled');
    expect(c).toMatchObject({ state: 'paused', pausedWhy: "OAIY's voice did not start for the call, twice" });
    expect(r.posted.at(-1)).toMatch(/is paused: OAIY's voice did not start/);
  });

  it('answered, with a result recorded on the call: done, with the answers in the results files', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1) });
    await r.outreach.tick();
    await r.outreach.event(ev('aokie.call.answered', { at: '' }, 'call_1'));
    expect(c.people[0].state).toBe('on_call');
    const link = r.outreach.forCall('call_1', '+61412345678')!;
    expect(link.instructions()).toContain('This is a call YOU placed for your person\'s outreach "Confirm Friday bookings". You already said: "Hi Jane,');
    expect(link.instructions().length).toBeLessThan(1600);
    const saved = await link.resultTool().run({ outcome: 'completed', answers: { coming: 'yes', nope: 'dropped' }, summary: 'Confirmed, will be there at 10:30.' });
    expect(saved).toBe('Saved: completed, coming = yes. Still to find out: nothing.');
    expect(link.recorded()).toBe(true);
    expect(c.people[0].state).toBe('on_call');
    await r.endOnLine('call_1', 'completed', 'agent_hangup');
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: true } });
    // Everyone done: the results, and the report once.
    await r.outreach.tick();
    await r.outreach.tick();
    expect(c.state).toBe('done');
    expect(r.reports).toHaveLength(1);
    expect(c.report.text).toContain('Jane Smith | completed | coming: yes | Confirmed, will be there at 10:30.');
    expect(r.files.readText('/outreach/confirm-friday-bookings/results.md')).toContain('| Jane Smith | 0412 345 678 | completed | yes |  | Confirmed, will be there at 10:30. | 1 |');
    expect(r.files.readText('/outreach/confirm-friday-bookings/results.csv').split('\n')[1]).toMatch(/^Jane Smith,'\+61412345678,completed,yes,,"Confirmed, will be there at 10:30.",1,/);
    expect(JSON.parse(r.files.readText('/outreach/confirm-friday-bookings/results.json')).people[0].answers).toEqual({ coming: true });
  });

  it('answered but no result: its agent is asked, and with none after two minutes it is unclear, from the last words', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1) });
    await r.outreach.tick();
    const link = r.outreach.forCall('call_1', '+61412345678')!;
    expect(r.outreach.callEnded(link, ['Caller: Hang on, who is this?'])).toBe(true);
    await r.endOnLine('call_1', 'completed');
    expect(c.people[0].state).toBe('ended');
    r.later(AFTER_CALL_MS + 1000);
    await r.outreach.tick();
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'unclear' });
    expect(c.people[0].summary).toContain('Caller: Hang on, who is this?');
    expect(OUTREACH_AFTER_CALL).toMatch(/call record_result now\. Write nothing else\./);
  });

  it('answered by someone other than OAIY (never on the call): unclear, at once', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1) });
    await r.outreach.tick();
    await r.endOnLine('call_1', 'completed');
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'unclear', summary: 'Answered, but OAIY was not on the call.' });
  });

  it('voicemail with tries left is rung again; a call back asked for is rung then, once, without using a try', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 2) });
    await r.outreach.tick();
    let link = r.outreach.forCall('call_1', '+61412345678')!;
    await link.resultTool().run({ outcome: 'voicemail', summary: 'Voicemail greeting.' });
    expect(link.voicemailRecorded()).toBe(true);
    await r.endOnLine('call_1', 'completed');
    expect(c.people[0]).toMatchObject({ state: 'waiting', tries: 1, nextAt: T0 + 60 * MIN });
    r.later(CALM);
    await r.outreach.tick();
    link = r.outreach.forCall('call_2', '+61413000111')!;
    await link.resultTool().run({ outcome: 'callback_requested', summary: 'Busy, call after lunch.', callBackAt: '2026-09-29 13:30' });
    await r.endOnLine('call_2', 'completed');
    expect(c.people[1]).toMatchObject({ state: 'waiting', tries: 0, callBackUsed: true, nextAt: new Date(2026, 8, 29, 13, 30).getTime() });
  });

  it('opted out: on the do-not-contact list, and a later campaign leaves them out', async () => {
    const r = rig();
    await r.start({ ...CALLS, people: CALLS.people.slice(0, 1) });
    await r.outreach.tick();
    const link = r.outreach.forCall('call_1', '+61412345678')!;
    await link.resultTool().run({ outcome: 'opted_out', summary: "Please don't call again." });
    expect(r.dnc().map((d) => d.number)).toEqual(['+61412345678']);
    const plan = r.outreach.plan({ ...CALLS, name: 'Next week', people: CALLS.people.slice(0, 2) }, null);
    if (typeof plan === 'string') throw new Error(plan);
    expect(plan.skipped).toEqual([{ name: 'Jane Smith', number: '0412 345 678', why: 'asked not to be contacted' }]);
  });

  it('someone on the list who rings in gets the context, and their result counts', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 2) });
    r.phone.holdsCalls = false;
    const link = r.outreach.forCall('call_in', '0413000111')!;
    expect(link.inbound).toBe(true);
    expect(link.instructions()).toMatch(/^They rang you, and they are on your person's outreach list/);
    await link.resultTool().run({ outcome: 'completed', answers: { coming: false, new_time: 'Monday 9am' }, summary: 'Rang in to move it to Monday.' });
    expect(c.people[1]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: false, new_time: 'Monday 9am' } });
  });
});

describe('a reload in the middle', () => {
  it("a call's end that came while the page reloaded is settled from the backlog: no second dial", async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 1), retries: { times: 0, gapMinutes: 60 } });
    await r.outreach.tick();
    // The page reloads: a new engine reads what was kept.
    const again = new Outreach(r.deps);
    await again.load();
    expect(again.campaigns[0].people[0]).toMatchObject({ state: 'dialling', attempt: { callId: 'call_1' } });
    await again.backlog([ev('aokie.sms.received', { from: '+61400000000' }), ev('aokie.call.ended', { callId: 'call_1', outcome: 'no_answer', reason: 'remote_hangup' }, 'call_1')]);
    expect(again.campaigns[0].people[0]).toMatchObject({ state: 'done', outcome: 'no_answer' });
    await again.tick();
    expect(r.dials()).toHaveLength(1);
    expect(c.id).toBe(again.campaigns[0].id);
  });

  it('a dial never heard to end: with words from them, its agent is asked for the result; with none, no answer', async () => {
    const r = rig();
    const c = await r.start({ ...CALLS, people: CALLS.people.slice(0, 2), retries: { times: 0, gapMinutes: 60 } });
    await r.outreach.tick();
    r.heard.set('+61412345678', ['Caller: Yes, still coming.']);
    r.later(DIAL_LOST_MS + 1000);
    await r.outreach.tick();
    expect(c.people[0].state).toBe('ended');
    expect(r.asked).toEqual(['+61412345678']);
    // The line let the lost dial go: the second is rung.
    expect(c.people[1].state).toBe('dialling');
    r.later(AFTER_CALL_MS + 1000);
    await r.outreach.tick();
    expect(c.people[0].outcome).toBe('unclear');
    // The second, never answered and never heard of again.
    r.later(DIAL_LOST_MS);
    await r.outreach.tick();
    expect(c.people[1]).toMatchObject({ state: 'done', outcome: 'no_answer' });
    expect(r.asked).toEqual(['+61412345678']);
  });
});

describe('texting down a list', () => {
  it('texts one at a time, paced, with its own message id; notes it in their conversation; replies settle it', async () => {
    const r = rig();
    const c = await r.start(TEXTS);
    await r.outreach.tick();
    const sends = () => r.commands.filter((x) => x.command === 'sms.send');
    expect(sends()).toEqual([{ command: 'sms.send', key: `oaiy:outreach-sms:oaiy-out.${c.id}.p1.1`, payload: { to: '+61412345678', body: 'Hi Jane, Greenleaf Lawns here: still right for Friday? Reply YES or NO.', messageId: `oaiy-out.${c.id}.p1.1` } }]);
    expect(r.texted[0].note).toMatch(/^\[OAIY\] Outreach "Friday reminders": you texted them \(.+\): "Hi Jane, Greenleaf Lawns here/);
    r.later(5_000);
    await r.outreach.tick();
    expect(sends()).toHaveLength(1);
    r.later(TEXT_GAP_MS);
    await r.outreach.tick();
    expect(sends()).toHaveLength(2);
    await r.outreach.event(ev('aokie.sms.sent', { messageId: `oaiy-out.${c.id}.p1.1` }));
    expect(c.people[0].state).toBe('awaiting_reply');
    const link = r.outreach.forText('+61412345678')!;
    expect(link.instructions()).toContain('You texted them for your person\'s outreach "Friday reminders"');
    await link.resultTool().run({ outcome: 'completed', answers: { coming: 'yes' }, summary: 'Yes, see you Friday.' });
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'completed', answers: { coming: true } });
    // Their context stays for a while, for a late reply.
    expect(r.outreach.forText('+61412345678')).toBeTruthy();
    r.phone.holdsTexts = false;
    expect(r.outreach.forText('+61412345678')).toBeUndefined();
  });

  it('a text the phone refused for its number is not a working number; another failure is tried once more', async () => {
    const r = rig();
    const c = await r.start(TEXTS);
    await r.outreach.tick();
    r.later(TEXT_GAP_MS);
    await r.outreach.tick();
    await r.outreach.event(ev('aokie.sms.failed', { messageId: `oaiy-out.${c.id}.p1.1`, reason: 'recipient: not a number', refused: true }));
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'invalid_number' });
    await r.outreach.event(ev('aokie.sms.failed', { messageId: `oaiy-out.${c.id}.p2.1`, reason: 'MAS PUT failed' }));
    expect(c.people[1]).toMatchObject({ state: 'queued', nextAt: r.now() + 5 * MIN });
    r.later(5 * MIN);
    await r.outreach.tick();
    await r.outreach.event(ev('aokie.sms.failed', { messageId: `oaiy-out.${c.id}.p2.2`, reason: 'MAS PUT failed' }));
    expect(c.people[1]).toMatchObject({ state: 'done', outcome: 'unreachable' });
  });

  it('a text never said to have gone is taken as sent after ten minutes, and never sent again', async () => {
    const r = rig();
    const c = await r.start({ ...TEXTS, people: TEXTS.people.slice(0, 1) });
    await r.outreach.tick();
    r.later(SMS_ACK_MS + 1000);
    await r.outreach.tick();
    expect(c.people[0]).toMatchObject({ state: 'awaiting_reply', unconfirmed: true });
    r.later(60 * MIN);
    await r.outreach.tick();
    expect(r.commands.filter((x) => x.command === 'sms.send')).toHaveLength(1);
  });

  it('STOP is read by code: they are opted out and on the list, and nothing is sent back', async () => {
    const r = rig();
    const c = await r.start(TEXTS);
    await r.outreach.tick();
    expect(r.outreach.stopWord('+61412345678', 'Yes please')).toBe(false);
    expect(r.outreach.stopWord('+61412345678', ' stop. ')).toBe(true);
    expect(c.people[0]).toMatchObject({ state: 'done', outcome: 'opted_out' });
    expect(r.dnc().map((d) => d.number)).toEqual(['+61412345678']);
    // Someone not texted for a campaign: not this engine's to decide.
    expect(r.outreach.stopWord('+61499999999', 'STOP')).toBe(false);
    expect(r.outreach.forText('+61412345678')).toBeUndefined();
  });

  it('no reply by the deadline is no reply; the report comes once, when everyone is done', async () => {
    const r = rig();
    const c = await r.start({ ...TEXTS, replyDeadlineHours: 2 });
    await r.outreach.tick();
    r.later(TEXT_GAP_MS);
    await r.outreach.tick();
    r.later(2 * 60 * MIN + 1000);
    await r.outreach.tick();
    expect(c.people.map((p) => p.outcome)).toEqual(['no_reply', 'no_reply']);
    expect(c.state).toBe('done');
    await r.outreach.tick();
    expect(r.reports).toHaveLength(1);
    expect(c.report.text).toMatch(/^\[OAIY\] Outreach "Friday reminders" is finished: 2 people/);
    expect(c.report.text).toContain('Reached 0 of 2. 2 no reply.');
  });

  it('stopping marks who was not texted yet stopped and who was texted no reply, then reports', async () => {
    const r = rig();
    const c = await r.start(TEXTS);
    await r.outreach.tick();
    await r.outreach.event(ev('aokie.sms.sent', { messageId: `oaiy-out.${c.id}.p1.1` }));
    await r.outreach.end(c.id);
    expect(c.people.map((p) => p.outcome)).toEqual(['no_reply', 'stopped']);
    expect(c.state).toBe('stopped');
    expect(r.reports).toHaveLength(1);
  });
});

describe('the results files', () => {
  it('escape what would break a table or a spreadsheet, and put a leading = + - @ out of harm', () => {
    expect(csvCell('=HYPERLINK("x")')).toBe(`"'=HYPERLINK(""x"")"`);
    expect(csvCell('+61412345678')).toBe("'+61412345678");
    expect(csvCell('a, b')).toBe('"a, b"');
    expect(csvCell('line\nbreak')).toBe('"line\nbreak"');
    expect(mdCell('a | b\nc')).toBe('a \\| b c');
    const c = {
      id: 'out-1', kind: 'call', name: 'X', slug: 'x', objective: 'Y', collect: [{ key: 'note', question: 'Q', type: 'text' }],
      createdAt: T0, endedAt: null, state: 'running', resultsPath: '/outreach/x/results.md', skipped: [],
      people: [{ id: 'p1', name: '@evil', number: '+61412345678', raw: '', fields: {}, state: 'done', tries: 1, nextAt: 0, answers: { note: '-1 | x' }, history: [], outcome: 'completed', summary: 'Said "hi", then left' }],
    } as unknown as Campaign;
    expect(resultsCsv(c).split('\n')[1]).toBe(`'@evil,'+61412345678,completed,'-1 | x,"Said ""hi"", then left",1,,`);
    expect(resultsMarkdown(c)).toContain('| @evil | 0412 345 678 | completed | -1 \\| x | Said "hi", then left | 1 |  |');
  });
});
