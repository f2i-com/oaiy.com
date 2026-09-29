/**
 * The Front desk's stored conversations as the app kept them before a
 * person's calls and texts were one (the live desk of 29 Sept 2026): Lance's
 * calls in one conversation keyed by his number as the phone's caller id gave
 * it (0491570006, national), his texts in another keyed as the texts gave it
 * (+61491570006, international), five of them with no reply (answering texts
 * was off, and no times were kept then), and what the agents knew about him in
 * two notes. Written with the app's own text builders, in the shapes of
 * `sessions/index.json`, `sessions/<id>.json` and `callers.json`, so the times
 * are this computer's local ones (as the live app wrote them).
 */
import type { ToolCall, Turn } from '../../src/agent/protocol';
import { LOOKUP_ASKED, callStartNote, callerLine, knownText, textMessage } from '../../src/sessions';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';

export interface StoredDesk {
  index: SessionInfo[];
  chats: Record<string, Turn[]>;
  callers: CallerNote[];
}

const GREETING = 'Hi, thanks for calling Greenline Gardens. How can I help?';

/** The live desk, on Tuesday 29 September 2026 (a call yesterday afternoon, texts this morning, a call after them). */
export function liveFrontDesk(): StoredDesk {
  const yesterdayCall = new Date(2026, 8, 28, 16, 5);
  const firstText = new Date(2026, 8, 29, 9, 52);
  const lastText = new Date(2026, 8, 29, 10, 1, 30);
  const todayCall = new Date(2026, 8, 29, 10, 17);
  const callers: CallerNote[] = [
    // Saved by a call's agent (remember): under the number as the caller id gave it.
    { number: '0491570006', name: 'Lance', facts: ['Lawn mowing, fortnightly'], updatedAt: yesterdayCall.getTime() + 60_000 },
    // Saved by the runner (caller_notes), under the texts' number, later: a newer name.
    { number: '+61491570006', name: 'Lance Smith', facts: ['Prefers afternoons', 'lawn mowing, fortnightly'], updatedAt: todayCall.getTime() - 5 * 60_000 },
    { number: '+61400111222', name: 'Priya Shah', facts: [], updatedAt: yesterdayCall.getTime() },
  ];
  const known = knownText(callers[0]);
  const said = (text: string, calls: ToolCall[] = []): Turn => ({ role: 'assistant', text, calls });
  const result = (id: string, name: string, content: string): Turn => ({ role: 'tool', results: [{ id, name, content, isError: false }] });
  const ended: Turn = { role: 'user', automatic: true, text: '[OAIY] 📞 The call ended.' };
  const callTurns: Turn[] = [
    { role: 'user', automatic: true, fresh: true, text: callStartNote('Lance (0491570006)', GREETING, known, yesterdayCall) },
    { role: 'user', text: callerLine('Hi, just checking you got my message about the mowing?', { startMs: 3_400 }) },
    said('Yes, we did. The team has you down for next week.'),
    { role: 'user', text: callerLine('Great, thanks. Bye.', { startMs: 11_900 }) },
    said('', [{ id: 'y1', name: 'end_call', input: { goodbye: 'Bye Lance!' } }]),
    result('y1', 'end_call', 'The goodbye is being said, then the call ends. Write nothing more.'),
    ended,
    { role: 'user', automatic: true, fresh: true, text: callStartNote('Lance (0491570006)', GREETING, known, todayCall) },
    { role: 'user', text: callerLine('Hi there, I was hoping to book a lawn mow.', { startMs: 4_200 }) },
    said('Let me check.', [{ id: 'l1', name: 'lookup_business_data', input: { question: 'Free times next Tuesday afternoon' } }]),
    result('l1', 'lookup_business_data', LOOKUP_ASKED),
    { role: 'user', text: '[OAIY] The answer to your lookup "Free times next Tuesday afternoon":\n{"free": ["13:00", "15:30"]}' },
    said('Tuesday at 1 pm or 3:30 pm are free. Which would you like?'),
    { role: 'user', text: callerLine("One o'clock is great. Oh, and it's Lance, by the way.", { startMs: 24_000, cut: true }, 'Which would you like?') },
    said('', [{ id: 'l2', name: 'remember', input: { name: 'Lance' } }, { id: 'l3', name: 'request_appointment', input: { callerName: 'Lance', service: 'Lawn mowing', date: '2026-10-06', time: '13:00', agreementPhrase: "One o'clock is great" } }]),
    { role: 'tool', results: [{ id: 'l2', name: 'remember', content: 'Saved.', isError: false }, { id: 'l3', name: 'request_appointment', content: '{\n "requestId": "apt_31",\n "status": "requested"\n}', isError: false }] },
    said("Thanks, Lance. I've requested Tuesday the 6th at 1 pm for a lawn mow; the team will confirm by text."),
    { role: 'user', text: callerLine('Perfect, thanks. Bye!', { startMs: 41_000 }) },
    said('', [{ id: 'l4', name: 'end_call', input: { goodbye: 'Bye Lance, have a great day!' } }]),
    result('l4', 'end_call', 'The goodbye is being said, then the call ends. Write nothing more.'),
    ended,
  ];
  // Five texts, kept and not answered (answering was off): no times, no replies.
  const textTurns: Turn[] = ['Testing', 'Hello', 'Hello', 'Hello', 'Hello'].map((body) => ({ role: 'user', text: textMessage('Lance', '+61491570006', body) }));
  const priya: Turn[] = [
    { role: 'user', automatic: true, fresh: true, text: callStartNote('Priya Shah (+61400111222)', GREETING, knownText(callers[2]), yesterdayCall) },
    { role: 'user', text: callerLine('Do you do hedges on the north shore?', { startMs: 2_900 }) },
    said('We do, from $60 a hedge. Would you like a time?'),
    ended,
  ];
  const task: Turn[] = [{ role: 'user', text: '[OAIY] Your flow "Morning summary" asks: Summarise yesterday.' }, said('- Lance asked about his mowing.')];
  return {
    index: [
      { id: 'call-0491570006', kind: 'call', key: '0491570006', title: 'Lance', lastAt: todayCall.getTime() + 45_000, unread: 3 },
      { id: 'sms-61491570006', kind: 'sms', key: '+61491570006', title: 'Lance', lastAt: lastText.getTime(), unread: 5, handles: ['0400000000000030', '0400000000000031', '0400000000000032', '0400000000000033', '0400000000000034'] },
      { id: 'task-morning-summary-mg1', kind: 'task', key: 'Morning summary', title: 'Morning summary', lastAt: firstText.getTime() - 3 * 60 * 60_000, unread: 0 },
      { id: 'call-61400111222', kind: 'call', key: '+61400111222', title: 'Priya Shah', lastAt: yesterdayCall.getTime() + 20_000, unread: 0 },
      { id: 'call-8841', kind: 'call', key: 'call_8841', title: 'Hidden number', lastAt: yesterdayCall.getTime() - 60 * 60_000, unread: 0 },
    ],
    chats: {
      'call-0491570006': callTurns,
      'sms-61491570006': textTurns,
      'task-morning-summary-mg1': task,
      'call-61400111222': priya,
      'call-8841': [{ role: 'user', automatic: true, fresh: true, text: callStartNote('Hidden number (call_8841)', GREETING, knownText(undefined), new Date(yesterdayCall.getTime() - 60 * 60_000)) }, { role: 'user', text: callerLine('Sorry, wrong number.', { startMs: 1_500 }) }, ended],
    },
    callers,
  };
}
