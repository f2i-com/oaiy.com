import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import type { Desktop } from '../../src/desktop/bridge';
import { NetGate } from '../../src/gate/netgate';
import { setLocalCountry } from '../../src/phoneNumbers';
import { Sessions, isCallStart } from '../../src/sessions';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { laneTimes, mergeCallers, mergeLanes, regroup, threadOrder } from '../../src/threads';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';
import { liveFrontDesk, type StoredDesk } from '../fixtures/liveFrontDesk';
import { OPENAI, fakeProvider } from './fakeProvider';

beforeEach(() => setLocalCountry('AU'));
afterEach(() => vi.unstubAllGlobals());

/** The Front desk's storage in memory: its files by path, as OPFS keeps them (`sessions/…`, `callers.json`, a backup's folder). */
function fakeDesk(desk?: StoredDesk) {
  const files = new Map<string, string>();
  if (desk) {
    files.set('sessions/index.json', JSON.stringify(desk.index));
    for (const [id, turns] of Object.entries(desk.chats)) files.set(`sessions/${id}.json`, JSON.stringify(turns));
    files.set('callers.json', JSON.stringify(desk.callers));
    files.set('callbacks.json', '[]');
  }
  const backups: string[] = [];
  const read = <T>(path: string, empty: T): T => (files.has(path) ? (JSON.parse(files.get(path)!) as T) : empty);
  const project = {
    loadSessions: async () => read<SessionInfo[]>('sessions/index.json', []),
    saveSessions: async (list: SessionInfo[]) => void files.set('sessions/index.json', JSON.stringify(list)),
    loadSessionChat: async (id: string) => read<Turn[]>(`sessions/${id}.json`, []),
    saveSessionChat: async (id: string, turns: Turn[]) => void files.set(`sessions/${id}.json`, JSON.stringify(turns)),
    removeSessionChat: async (id: string) => void files.delete(`sessions/${id}.json`),
    loadCallers: async () => read<CallerNote[]>('callers.json', []),
    saveCallers: async (list: CallerNote[]) => void files.set('callers.json', JSON.stringify(list)),
    backupSessions: async (name: string) => {
      let used = name;
      for (let n = 2; [...files.keys()].some((k) => k.startsWith(`${used}/`)); n++) used = `${name}-${n}`;
      for (const [path, text] of [...files]) if (!path.startsWith('.backup')) files.set(`${used}/${path}`, text);
      backups.push(used);
      return used;
    },
  };
  return { files, project, backups };
}

/** A desktop that speaks on calls and sends texts, recording both. */
function fakePhone() {
  const said: Array<[string, string]> = [];
  const sent: Array<{ to: unknown; body: unknown }> = [];
  return {
    said,
    sent,
    say: async (callId: string, text: string) => void said.push([callId, text]),
    finishCall: async () => ({ ok: true, output: {} }),
    callTool: async () => ({ ok: true, output: { recorded: true } }),
    command: async (_c: string, command: string, payload: Record<string, unknown>) => {
      if (command === 'sms.send') sent.push({ to: payload.to, body: payload.body });
      return { messageId: `m${sent.length}` };
    },
  };
}

function sessionsOn(project: unknown, messages: Partial<MessageSettings> = {}, desktop: unknown = fakePhone()) {
  const settings: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, instructions: '', calls: true, ...messages };
  return new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'Front desk', ...extra }),
    () => settings,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
  );
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
  await new Promise((r) => setTimeout(r, 0));
}

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
const textOf = (t: Turn) => (t.role === 'tool' ? t.results.map((r) => r.content).join('|') : t.text);

describe("the live desk's calls and texts with one person, merged once", () => {
  it('Liam\'s Calls (0491570006) and Texts (+61491570006) become one conversation, in the order they happened, keeping every turn', async () => {
    const desk = liveFrontDesk();
    const store = fakeDesk(desk);
    const sessions = sessionsOn(store.project);
    await sessions.load();

    const liam = sessions.threads().find((t) => t.title.startsWith('Liam'))!;
    expect(sessions.threads().filter((t) => t.title.startsWith('Liam'))).toHaveLength(1);
    expect(liam).toMatchObject({ id: 'person-61491570006', kind: 'person', key: '+61491570006', ways: ['call', 'sms'], unread: 8 });
    // Named by them: the newest name they were given.
    expect(liam.title).toBe('Liam Smith');
    expect(liam.call).toMatchObject({ id: 'call-0491570006', key: '+61491570006', thread: 'person-61491570006' });
    expect(liam.sms).toMatchObject({ id: 'sms-61491570006', key: '+61491570006', thread: 'person-61491570006', handles: desk.index[1].handles });

    // Yesterday's call, this morning's texts, then the call after them: each lane's own order kept.
    const turns = sessions.turnsOf(liam.id);
    const calls = desk.chats['call-0491570006'];
    const texts = desk.chats['sms-61491570006'];
    expect(turns).toHaveLength(calls.length + texts.length);
    expect(turns.map(textOf)).toEqual([...calls.slice(0, 7), ...texts, ...calls.slice(7)].map(textOf));
    expect(turns.map((t) => t.via)).toEqual([...Array(7).fill('call'), ...Array(5).fill('sms'), ...Array(14).fill('call')]);

    // Each agent still has its own: the call's starts fresh at the last call, the texts' reads its texts.
    expect(liam.call!.agent.turns.map(textOf)).toEqual(calls.map(textOf));
    expect(liam.call!.agent.view()[0]).toMatchObject({ fresh: true });
    expect(isCallStart(liam.call!.agent.view()[0])).toBe(true);
    expect(liam.call!.agent.view()).toHaveLength(14);
    expect(liam.sms!.agent.turns.map(textOf)).toEqual(texts.map(textOf));

    // One file for the conversation; the two it came from are gone from the list, and kept in the backup.
    const backup = `.backup-${today()}`;
    expect(store.backups).toEqual([backup]);
    expect(JSON.parse(store.files.get('sessions/person-61491570006.json')!)).toHaveLength(26);
    expect(store.files.has('sessions/call-0491570006.json')).toBe(false);
    expect(store.files.has('sessions/sms-61491570006.json')).toBe(false);
    expect(JSON.parse(store.files.get(`${backup}/sessions/call-0491570006.json`)!)).toEqual(desk.chats['call-0491570006']);
    expect(JSON.parse(store.files.get(`${backup}/sessions/sms-61491570006.json`)!)).toEqual(desk.chats['sms-61491570006']);
    expect(JSON.parse(store.files.get(`${backup}/sessions/index.json`)!)).toEqual(desk.index);
    expect(JSON.parse(store.files.get(`${backup}/callers.json`)!)).toEqual(desk.callers);

    // What was known about him, as one note under his E.164 number: the newer name, every fact once.
    expect(sessions.callers.filter((c) => c.name?.startsWith('Liam'))).toEqual([
      { number: '+61491570006', name: 'Liam Smith', facts: ['Lawn mowing, fortnightly', 'Prefers afternoons'], updatedAt: desk.callers[1].updatedAt },
    ]);
    expect(sessions.callerNote('0491 570 006')?.name).toBe('Liam Smith');

    // The others: Priya's calls are hers (a conversation of one lane), a hidden caller's stays its own, a flow's tasks are as they were.
    const index = JSON.parse(store.files.get('sessions/index.json')!) as SessionInfo[];
    expect(index.find((i) => i.id === 'call-61400111222')).toMatchObject({ key: '+61400111222', thread: 'person-61400111222' });
    expect(index.find((i) => i.id === 'call-8841')).toMatchObject({ key: 'call_8841', thread: 'person-call_8841', hidden: true, title: 'Hidden number' });
    expect(index.find((i) => i.kind === 'task')).toMatchObject({ id: 'task-morning-summary-mg1', thread: 'task-morning-summary-mg1' });
    expect(store.files.get('sessions/task-morning-summary-mg1.json')).toBe(JSON.stringify(desk.chats['task-morning-summary-mg1']));
    expect(sessions.threads()).toHaveLength(4);
  });

  it('is done once: the next start finds nothing to merge, changes no file and makes no backup', async () => {
    const store = fakeDesk(liveFrontDesk());
    await sessionsOn(store.project).load();
    const after = new Map(store.files);
    const again = sessionsOn(store.project);
    await again.load();
    expect(store.backups).toHaveLength(1);
    expect(store.files).toEqual(after);
    const liam = again.thread('person-61491570006')!;
    expect(again.turnsOf(liam.id).map((t) => t.via)).toEqual([...Array(7).fill('call'), ...Array(5).fill('sms'), ...Array(14).fill('call')]);
    // The rule itself, on what it returned: no change.
    const stored = { infos: JSON.parse(after.get('sessions/index.json')!), chats: new Map([...after].filter(([k]) => k.startsWith('sessions/') && k !== 'sessions/index.json').map(([k, v]) => [k.slice(9, -5), JSON.parse(v)])), callers: JSON.parse(after.get('callers.json')!) };
    expect(regroup(stored, 'AU')).toMatchObject({ changed: false, rewritten: [], stale: [] });
  });

  it('keeps the order the merge chose when the texts go on (the old texts do not move after the new ones)', async () => {
    const store = fakeDesk(liveFrontDesk());
    const sessions = sessionsOn(store.project);
    await sessions.load();
    await sessions.textArrived('+61491570006', 'Liam', 'Are you still coming Tuesday?');
    const turns = sessionsOn(store.project);
    await turns.load();
    const order = turns.turnsOf('person-61491570006').map((t) => t.via);
    expect(order).toEqual([...Array(7).fill('call'), ...Array(5).fill('sms'), ...Array(14).fill('call'), 'sms']);
  });
});

describe('the merge rule', () => {
  const at = (h: number, m: number, s = 0) => new Date(2026, 8, 29, h, m, s).getTime();

  it("orders by the times turns carry: a call's start, its caller's words (from the start), a turn's own time, and a lane's last turn by when it was last heard", () => {
    const call: Turn[] = [
      { role: 'user', automatic: true, text: '[OAIY] 📞 A call from Liam (0491570006) began, Tue 29 Sep, 10:17 am.\nToday is Tuesday 29 September 2026.' },
      { role: 'user', text: 'Caller [0:04]: Hi.' },
      { role: 'assistant', text: 'Hello!', calls: [] },
      { role: 'user', text: 'Caller [1:10]: Bye.' },
    ];
    const sms: Turn[] = [
      { role: 'user', text: 'Text message from Liam (+61491570006):\nOn my way', at: at(10, 17, 30) },
      { role: 'user', text: 'Text message from Liam (+61491570006):\nHello' },
    ];
    expect(laneTimes({ turns: call, lastAt: at(10, 19) })).toEqual([at(10, 17), at(10, 17, 4), at(10, 17, 4), at(10, 18, 10)]);
    // The last text has no time of its own: the lane's last-heard time.
    expect(laneTimes({ turns: sms, lastAt: at(10, 20) })).toEqual([at(10, 17, 30), at(10, 20)]);
    const merged = mergeLanes([{ via: 'call', turns: call, lastAt: at(10, 19) }, { via: 'sms', turns: sms, lastAt: at(10, 20) }]);
    expect(merged.map((t) => (t.role === 'user' ? t.text.split('\n').at(-1) : 'reply'))).toEqual(['Today is Tuesday 29 September 2026.', 'Caller [0:04]: Hi.', 'reply', 'On my way', 'Caller [1:10]: Bye.', 'Hello']);
  });

  it('a run with no times before any known time goes right before it; one after, right after the turn before it', () => {
    const lane: Turn[] = [{ role: 'user', text: 'a' }, { role: 'user', text: 'b', at: 50 }, { role: 'assistant', text: 'c', calls: [] }, { role: 'user', text: 'd', at: 40 }];
    // Known times never go back within a lane (d is kept after c).
    expect(laneTimes({ turns: lane })).toEqual([50, 50, 50, 50]);
    expect(laneTimes({ turns: [{ role: 'user', text: 'x' }] })).toEqual([-Infinity]);
  });

  it("once kept, the order is the conversation's: new turns go after it by their times, a summary put into a lane before the turn after it", () => {
    const a1: Turn = { role: 'user', text: 'a1', via: 'call' };
    const b1: Turn = { role: 'user', text: 'b1', via: 'sms' };
    const a2: Turn = { role: 'user', text: 'a2', via: 'call' };
    const previous = [b1, a1, a2];
    const b2: Turn = { role: 'user', text: 'b2', at: 20 };
    const a3: Turn = { role: 'user', text: 'a3', at: 10 };
    const a4: Turn = { role: 'assistant', text: 'a4', calls: [], at: 30 };
    const summary: Turn = { role: 'user', text: 'summary', summary: true };
    const order = threadOrder(previous, [{ via: 'call', turns: [a1, summary, a2, a3, a4] }, { via: 'sms', turns: [b1, b2] }]);
    expect(order.map((t) => (t.role === 'tool' ? '' : t.text))).toEqual(['b1', 'a1', 'summary', 'a2', 'a3', 'b2', 'a4']);
    expect(order.map((t) => t.via)).toEqual(['sms', 'call', 'call', 'call', 'call', 'sms', 'call']);
    // A lane let go of its oldest turns, or started again: they leave the conversation.
    expect(threadOrder(order, [{ via: 'call', turns: [a3, a4] }, { via: 'sms', turns: [] }]).map((t) => (t.role === 'tool' ? '' : t.text))).toEqual(['a3', 'a4']);
  });

  it("notes for one person, in any format, become one: the newest name, every fact once, the newest time", () => {
    const { callers, changed } = mergeCallers([
      { number: '+61491570006', name: 'Liam Smith', facts: ['Prefers afternoons'], updatedAt: 20 },
      { number: '0491 570 006', name: 'Liam', facts: ['Big back lawn', 'prefers afternoons'], updatedAt: 10 },
      { number: '+61400000001', facts: [], updatedAt: 5 },
      { number: 'test', name: 'Test', facts: [], updatedAt: 1 },
    ], 'AU');
    expect(changed).toBe(true);
    expect(callers).toEqual([
      { number: 'test', name: 'Test', facts: [], updatedAt: 1 },
      { number: '+61400000001', facts: [], updatedAt: 5 },
      { number: '+61491570006', name: 'Liam Smith', facts: ['Big back lawn', 'prefers afternoons'], updatedAt: 20 },
    ]);
    // A newer note without a name keeps the older name.
    expect(mergeCallers([{ number: '0491570006', name: 'Liam', facts: [], updatedAt: 1 }, { number: '+61491570006', facts: ['x'], updatedAt: 2 }], 'AU').callers).toEqual([{ number: '+61491570006', name: 'Liam', facts: ['x'], updatedAt: 2 }]);
    expect(mergeCallers(callers, 'AU').changed).toBe(false);
  });
});

describe('one conversation per person, however the phone writes their number', () => {
  it('a call from 0491570006 and a text from +61491570006 land in one conversation, in the order they came', async () => {
    const fake = fakeProvider('openai', [{ text: 'Yes, Tuesday at ten is free.' }]);
    const store = fakeDesk();
    const phone = fakePhone();
    const sessions = sessionsOn(store.project, { answer: false }, phone);
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '0491570006', name: '' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Is Tuesday at ten free?' });
    await settled(sessions);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_1' });
    const text = await sessions.textArrived('+61491570006', 'Liam', 'Thanks, see you then');
    expect(fake.bodies).toHaveLength(1);
    expect(text.thread).toBe(call!.thread);
    expect(call).toMatchObject({ id: 'call-61491570006', key: '+61491570006', thread: 'person-61491570006' });
    expect(text).toMatchObject({ id: 'sms-61491570006', key: '+61491570006' });
    const threads = sessions.threads();
    expect(threads).toHaveLength(1);
    expect(threads[0]).toMatchObject({ title: 'Liam', ways: ['call', 'sms'], lastWay: 'sms' });
    // The call took the name the texts brought.
    expect(call!.title).toBe('Liam');
    const turns = sessions.turnsOf(threads[0].id);
    expect(turns.map((t) => (t.role !== 'user' ? t.role : isCallStart(t) ? 'the call began' : t.text.split('\n')[0]))).toEqual(['the call began', 'Caller: Is Tuesday at ten free?', 'assistant', '[OAIY] 📞 The call ended.', 'Text message from Liam (+61491570006):']);
    expect(turns[0].role === 'user' && turns[0].text).toContain('A call from +61491570006 began');
    expect(turns.every((t) => typeof t.at === 'number')).toBe(true);
    // Kept in one file, the conversation's.
    expect(JSON.parse(store.files.get('sessions/person-61491570006.json')!)).toHaveLength(5);
    expect(JSON.parse(store.files.get('sessions/index.json')!).map((i: SessionInfo) => [i.id, i.thread])).toEqual([['sms-61491570006', 'person-61491570006'], ['call-61491570006', 'person-61491570006']]);
    // Another caller's is their own.
    await sessions.callEvent({ type: 'call.started', callId: 'call_2', from: '+44 20 7946 0958' });
    expect(sessions.threads()).toHaveLength(2);
    // A hidden caller, however the phone says it, never joins anyone.
    await sessions.callEvent({ type: 'call.started', callId: 'call_3', from: 'Private' });
    await sessions.callEvent({ type: 'call.started', callId: 'call_4', from: '' });
    expect(sessions.threads().filter((t) => t.hidden).map((t) => t.title)).toEqual(['Hidden number', 'Hidden number']);
  });

  it("a caller who texted first: the call's agent starts fresh, and its note has what they texted", async () => {
    const fake = fakeProvider('openai', [{ text: 'Hi Liam, yes we got your text.' }]);
    const sessions = sessionsOn(fakeDesk().project, { answer: false });
    await sessions.textArrived('+61491570006', 'Liam', 'Can you come a bit earlier on Tuesday?');
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '0491570006' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Did you get my text?' });
    await settled(sessions);
    const sent = JSON.stringify(fake.bodies[0]);
    expect(sent).toContain('A call from Liam (+61491570006) began');
    expect(sent).toContain('Their last contact, ');
    expect(sent).toContain('- They texted: \\"Can you come a bit earlier on Tuesday?\\"');
    // Only its own turns (the text is not a turn of the call's).
    expect(call!.agent.turns.filter((t) => t.role === 'user' && t.text.startsWith('Text message'))).toHaveLength(0);
    // earlier_conversations reaches the texts too, in order.
    expect(sessions.earlierWith(call!, 'earlier')).toContain('Their text messages');
  });

  it('a text during a live call lands in the same conversation without breaking the call', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const asked: string[] = [];
    let texts = 0;
    const route = (body: Record<string, unknown>) => {
      const system = JSON.stringify((body.messages as Array<{ role: string }>)[0]);
      if (/live phone call/.test(system)) {
        asked.push('call');
        return { text: 'Tuesday at ten works. Anything else?', hold: { at: 8, until: held } };
      }
      asked.push('sms');
      return texts++ === 0 ? { calls: [{ name: 'send_text_message', input: { body: 'Got it, thanks!' } }] } : { text: 'Replied.' };
    };
    fakeProvider('openai', [route, route, route, route]);
    const phone = fakePhone();
    const sessions = sessionsOn(fakeDesk().project, { answer: true }, phone);
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '0491570006', name: 'Liam' });
    await sessions.callEvent({ type: 'call.caller', callId: 'call_1', text: 'Can you do Tuesday at ten?' });
    for (let i = 0; i < 100 && !asked.length; i++) await new Promise((r) => setTimeout(r, 5));
    // The call's reply is being written: a text comes, and is answered at once by the texts' agent.
    const text = await sessions.textArrived('+61491570006', 'Liam', "I'll send the gate code by text");
    for (let i = 0; i < 200 && !phone.sent.length; i++) await new Promise((r) => setTimeout(r, 5));
    expect(phone.sent).toEqual([{ to: '+61491570006', body: 'Got it, thanks!' }]);
    expect(call!.callId).toBe('call_1');
    expect(call!.running).not.toBeNull();
    expect(text.thread).toBe(call!.thread);
    release();
    await settled(sessions);
    expect(phone.said.map(([id]) => id)).toEqual(['call_1', 'call_1']);
    expect(phone.said.map(([, words]) => words).join(' ')).toBe('Tuesday at ten works. Anything else?');
    // One conversation, in the order things happened: the call, the text and its answer, then the call's reply.
    const turns = sessions.turnsOf(call!.thread);
    expect(turns.map((t) => `${t.via}:${t.role}`)).toEqual(['call:user', 'call:user', 'sms:user', 'sms:assistant', 'sms:tool', 'sms:assistant', 'call:assistant']);
    // Neither agent reads the other's turns.
    expect(call!.agent.turns.every((t) => t.via === 'call')).toBe(true);
    expect(text.agent.turns.every((t) => t.via === 'sms')).toBe(true);
    expect(sessions.thread(call!.thread)).toMatchObject({ live: call, ways: ['call', 'sms'] });
  });

  it("the runner's note and the person's own message go to the call going on, else to the texts", async () => {
    const sessions = sessionsOn(fakeDesk().project, { answer: false });
    const call = await sessions.callEvent({ type: 'call.started', callId: 'call_1', from: '0491570006', name: 'Liam' });
    const thread = sessions.thread(call!.thread)!;
    expect(await sessions.laneFor(thread)).toBe(call);
    await sessions.callEvent({ type: 'call.ended', callId: 'call_1' });
    // They only ever rang: their texts' agent is made for it (and texts them).
    const texts = await sessions.laneFor(sessions.thread(call!.thread)!);
    expect(texts).toMatchObject({ kind: 'sms', key: '+61491570006', thread: call!.thread, title: 'Liam' });
    expect(sessions.threads()).toHaveLength(1);
  });
});
