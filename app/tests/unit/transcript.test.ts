import { describe, expect, it } from 'vitest';
import { callStartNote, callerLine, textMessage } from '../../src/sessions';
import {
  ago,
  clock,
  conversationKind,
  dayLabel,
  formatNumber,
  initials,
  isAcknowledgement,
  parseCallEnd,
  parseCallStart,
  parseCallTurn,
  parseCallerLine,
  parseFlowAsk,
  parseTextTurn,
  readWhen,
  splitWho,
} from '../../src/ui/chat/transcript';

describe("a call's words, as the chat shows them", () => {
  it('drops the "Caller" label and keeps when they spoke', () => {
    expect(parseCallerLine(callerLine('Hi, I was hoping to book a mow.', { startMs: 42_000 }))).toEqual({ kind: 'caller', text: 'Hi, I was hoping to book a mow.', atMs: 42_000 });
    expect(parseCallerLine(callerLine('Hello?', {}))).toEqual({ kind: 'caller', text: 'Hello?' });
    expect(clock(42_000)).toBe('0:42');
    expect(clock(605_400)).toBe('10:05');
  });

  it('marks words said over the agent, and cutting it off, with what it was saying', () => {
    const over = parseCallerLine(callerLine('Tuesday would be better', { startMs: 9_100, over: true }, 'We mow on Mondays and Fridays.'));
    expect(over).toMatchObject({ text: 'Tuesday would be better', atMs: 9_000, over: true, during: 'We mow on Mondays and Fridays.' });
    expect(over?.backchannel).toBeUndefined();
    const cut = parseCallerLine(callerLine('Wait, sorry', { startMs: 61_000, cut: true }, 'Which would you like?'));
    expect(cut).toMatchObject({ text: 'Wait, sorry', atMs: 61_000, cut: true, during: 'Which would you like?' });
    // No time, only how it fell.
    expect(parseCallerLine(callerLine('yes', { over: true }))).toMatchObject({ text: 'yes', over: true, backchannel: true });
  });

  it('infers an acknowledgement said over the agent (the label does not carry it)', () => {
    expect(parseCallerLine(callerLine('mm-hmm', { startMs: 3_000, over: true }, 'Sure thing.'))?.backchannel).toBe(true);
    expect(parseCallerLine(callerLine('Yeah, okay.', { startMs: 3_000, over: true }))?.backchannel).toBe(true);
    // The same words not over the agent are an answer, not a backchannel.
    expect(parseCallerLine(callerLine('yeah', { startMs: 3_000 }))?.backchannel).toBeUndefined();
    expect(isAcknowledgement('I see')).toBe(true);
    expect(isAcknowledgement('yes please, Tuesday')).toBe(false);
    expect(isAcknowledgement('')).toBe(false);
  });

  it("splits a turn of several lines: the caller's, OAIY's notes (a lookup's answer over several lines), and the person's own", () => {
    const turn = [
      callerLine('mm-hmm', { startMs: 9_100, over: true }, 'Sure thing.'),
      callerLine('Next Tuesday?', { startMs: 11_800 }),
      '[OAIY] The answer to your lookup "Free times":',
      '{',
      ' "free": ["13:00"]',
      '}',
      '',
      'Keep it short, please.',
    ].join('\n');
    const parts = parseCallTurn(turn);
    expect(parts.map((p) => p.kind)).toEqual(['caller', 'caller', 'note', 'plain']);
    expect(parts[0]).toMatchObject({ text: 'mm-hmm', backchannel: true });
    expect(parts[1]).toMatchObject({ text: 'Next Tuesday?', atMs: 12_000 });
    expect(parts[2].text).toBe('The answer to your lookup "Free times":\n{\n "free": ["13:00"]\n}');
    expect(parts[3]).toEqual({ kind: 'plain', text: 'Keep it short, please.' });
  });

  it('leaves what it does not recognise as it is', () => {
    expect(parseCallTurn('Please tell them we close at 5')).toEqual([{ kind: 'plain', text: 'Please tell them we close at 5' }]);
    expect(parseCallerLine('Callers: many')).toBeNull();
    expect(parseCallerLine('The Caller: said hi')).toBeNull();
    // A label it cannot read keeps the words, without a time.
    expect(parseCallerLine('Caller [soon]: hi')).toEqual({ kind: 'caller', text: 'hi' });
  });

  it("reads the note that opens a call: who, when, which way, and the phone's greeting", () => {
    const now = new Date(2026, 8, 29, 18, 0);
    const start = parseCallStart(callStartNote('Lance (+61491570006)', 'Hi, thanks for calling Greenline Gardens.', 'Nothing is saved about them yet.', new Date(2026, 8, 29, 10, 17)), now);
    expect(start).toMatchObject({ name: 'Lance', number: '+61491570006', direction: 'in', greeting: 'Hi, thanks for calling Greenline Gardens.' });
    expect(start?.at?.getTime()).toBe(new Date(2026, 8, 29, 10, 17).getTime());
    const unknown = parseCallStart(callStartNote('+61400111222', '', 'x', new Date(2026, 8, 28, 9, 5)), now);
    expect(unknown).toMatchObject({ name: '+61400111222', direction: 'in' });
    expect(unknown?.number).toBeUndefined();
    expect(unknown?.greeting).toBeUndefined();
    const back = parseCallStart(callStartNote('Dave (+61455666777)', 'Hi Dave, returning your call.', 'x', new Date(2026, 8, 29, 11, 0), new Date(2026, 8, 29, 9, 0).getTime()), now);
    expect(back).toMatchObject({ name: 'Dave', direction: 'back', greeting: 'Hi Dave, returning your call.' });
    expect(back?.at?.getHours()).toBe(11);
    const out = parseCallStart(callStartNote('Mia (+61488999000)', '', 'x', new Date(2026, 8, 29, 12, 30), undefined, { purpose: 'Confirm Saturday' }), now);
    expect(out).toMatchObject({ name: 'Mia', direction: 'out' });
    expect(parseCallStart('[OAIY] Not finished yet, so the agent carries on')).toBeNull();
  });

  it("reads a call's end, with why when it failed", () => {
    expect(parseCallEnd('[OAIY] 📞 The call ended.')).toBe('');
    expect(parseCallEnd('[OAIY] 📞 The call ended: the voice link failed.')).toBe('the voice link failed');
    expect(parseCallEnd('The call ended.')).toBeNull();
  });

  it('reads the time of day the note says, in the formats en-AU comes in, as the latest such day', () => {
    const now = new Date(2026, 8, 29, 18, 0);
    expect(readWhen('Tue 29 Sep, 10:17 am', now)?.getTime()).toBe(new Date(2026, 8, 29, 10, 17).getTime());
    expect(readWhen('Tue, 29 Sept, 10:17 pm', now)?.getTime()).toBe(new Date(2026, 8, 29, 22, 17).getTime());
    expect(readWhen('Mon, 28 Sept, 12:05 a.m.', now)?.getTime()).toBe(new Date(2026, 8, 28, 0, 5).getTime());
    // December, seen in January: last year.
    expect(readWhen('Wed 30 Dec, 9:00 am', new Date(2027, 0, 2))?.getFullYear()).toBe(2026);
    expect(readWhen('sometime', now)).toBeUndefined();
  });
});

describe("a text thread's words, as the chat shows them", () => {
  it('drops the "Text message from" label: the name and number are said once, in the header', () => {
    expect(parseTextTurn(textMessage('Lance', '+61491570006', 'Hello'))).toEqual([{ kind: 'text', name: 'Lance', number: '+61491570006', text: 'Hello' }]);
    expect(parseTextTurn(textMessage('+61400333444', '+61400333444', 'Is this Greenline?'))).toEqual([{ kind: 'text', number: '+61400333444', text: 'Is this Greenline?' }]);
    expect(parseTextTurn(textMessage('Test', 'test', 'Hi'))).toEqual([{ kind: 'text', name: 'Test', number: 'test', text: 'Hi' }]);
  });

  it('splits texts answered together, and keeps a text of several paragraphs whole', () => {
    const joined = [textMessage('Lance', '+61491570006', 'Yes please'), textMessage('Lance', '+61491570006', 'First line\n\nSecond paragraph')].join('\n\n');
    expect(parseTextTurn(joined)).toEqual([
      { kind: 'text', name: 'Lance', number: '+61491570006', text: 'Yes please' },
      { kind: 'text', name: 'Lance', number: '+61491570006', text: 'First line\n\nSecond paragraph' },
    ]);
  });

  it("keeps the person's own words, and a name with brackets, as they are", () => {
    expect(parseTextTurn('Tell him the invoice is attached.')).toEqual([{ kind: 'plain', text: 'Tell him the invoice is attached.' }]);
    expect(splitWho('Lance (work) (+61491570006)')).toEqual({ name: 'Lance (work)', number: '+61491570006' });
    expect(splitWho('+61491570006')).toEqual({ number: '+61491570006' });
    expect(parseTextTurn('Text message from Lance: hi')).toEqual([{ kind: 'plain', text: 'Text message from Lance: hi' }]);
  });
});

describe('the rest of what the chat reads from the words', () => {
  it("reads a flow's task", () => {
    expect(parseFlowAsk('[OAIY] Your flow "Morning summary" asks: Sum it up.\nIn three points.')).toEqual({ kind: 'flow', flow: 'Morning summary', text: 'Sum it up.\nIn three points.' });
    expect(parseFlowAsk('Your flow asks')).toBeNull();
  });

  it("tells a conversation's kind from its words", () => {
    expect(conversationKind([{ role: 'user', automatic: true, text: callStartNote('Lance (+61491570006)', '', 'x') }])).toBe('call');
    expect(conversationKind([{ role: 'user', text: textMessage('Lance', '+61491570006', 'Hi') }])).toBe('sms');
    expect(conversationKind([{ role: 'user', text: '[OAIY] Your flow "X" asks: y' }])).toBe('task');
    expect(conversationKind([{ role: 'user', text: 'Build me a site' }, { role: 'assistant' }])).toBe('own');
  });

  it('says days, times ago, numbers and initials as a person reads them', () => {
    const now = new Date(2026, 8, 29, 18, 0);
    expect(dayLabel(new Date(2026, 8, 29, 9, 0), now)).toBe('Today');
    expect(dayLabel(new Date(2026, 8, 28, 23, 0), now)).toBe('Yesterday');
    expect(ago(now.getTime() - 20 * 60_000, now.getTime())).toBe('20m');
    expect(ago(now.getTime() - 3 * 60 * 60_000, now.getTime())).toBe('3h');
    expect(ago(now.getTime() - 10_000, now.getTime())).toBe('now');
    expect(formatNumber('+61491570006')).toBe('+61 491 570 006');
    expect(formatNumber('0491570006')).toBe('0491 570 006');
    expect(formatNumber('+61298765432')).toBe('+61 2 9876 5432');
    expect(formatNumber('test')).toBe('test');
    expect(initials('Lance')).toBe('L');
    expect(initials('Priya Shah')).toBe('PS');
    expect(initials('+61400333444')).toBe('');
  });
});
