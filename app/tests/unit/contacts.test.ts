// The phone's agents and OAIY Desktop's contacts: a person's contact read
// into what a call's and a text's agent know (the business's notes first, and
// winning), what they remember written there, the Front desk's facts moved
// there once, and what is known here when the desktop cannot be reached.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { CallerNote, ContactsMoved, SessionInfo } from '../../src/vfs/projects';
import { Sessions, callerNotesTool, callStartNote, knownText } from '../../src/sessions';
import { contactKey, mirrorContact, moveFacts, type ContactsApi } from '../../src/contacts';
import { DesktopError, type Contact, type Desktop } from '../../src/desktop/bridge';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const signal = new AbortController().signal;

function contact(number: string, c: Partial<Contact> = {}): Contact {
  return { key: contactKey(number), number, name: '', nameBy: null, notes: '', facts: [], createdAt: '', updatedAt: '', ...c };
}
const fact = (text: string, by: 'owner' | 'agent' = 'agent') => ({ text, at: '2026-09-29T00:00:00Z', by });

/**
 * A fake of the desktop's contacts: kept by key (the last nine digits), a fact
 * kept once, `offline` refusing every request as a desktop out of reach does,
 * and `hold` keeping reads waiting.
 */
function fakeDesk(initial: Contact[] = []) {
  const book = new Map(initial.map((c) => [c.key, structuredClone(c)]));
  const requests: string[] = [];
  const state = { offline: false, hold: null as Promise<void> | null };
  const reach = async () => {
    if (state.hold) await state.hold;
    if (state.offline) throw new TypeError('Failed to fetch');
  };
  const api: ContactsApi = {
    get: async (number) => {
      requests.push(`get ${number}`);
      await reach();
      const c = book.get(contactKey(number));
      return c ? structuredClone(c) : null;
    },
    list: async () => {
      requests.push('list');
      await reach();
      return [...book.values()].map((c) => structuredClone(c));
    },
    addFact: async (number, text) => {
      requests.push(`add ${number} ${text}`);
      await reach();
      const key = contactKey(number);
      if (!key) throw new DesktopError('a hidden or withheld number never becomes a contact', 400, 'hidden');
      const c = book.get(key) ?? contact(number);
      book.set(key, c);
      if (c.facts.some((f) => f.text.toLowerCase() === text.toLowerCase())) return { contact: structuredClone(c), added: false };
      c.facts.push(fact(text));
      return { contact: structuredClone(c), added: true };
    },
    forgetFact: async (number, index, text) => {
      requests.push(`forget ${number} ${index} ${text}`);
      await reach();
      const c = book.get(contactKey(number))!;
      expect(c.facts[index]?.text).toBe(text);
      c.facts.splice(index, 1);
      return structuredClone(c);
    },
  };
  return { api, book, requests, state };
}

/** Sessions over an in-memory Front desk (its callers.json and the mark), with the fake contacts. */
function setup(desk = fakeDesk(), callers: CallerNote[] = [], mark: ContactsMoved | null = null) {
  const chats = new Map<string, Turn[]>();
  let index: SessionInfo[] = [];
  const project = {
    callers,
    mark,
    saves: 0,
    loadSessions: async () => index,
    saveSessions: async (list: SessionInfo[]) => {
      index = list;
    },
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => {
      chats.set(id, turns);
    },
    loadCallers: async () => project.callers,
    saveCallers: async (list: CallerNote[]) => {
      project.callers = structuredClone(list);
      project.saves++;
    },
    loadContactsMoved: async () => project.mark,
    saveContactsMoved: async (m: ContactsMoved) => {
      project.mark = structuredClone(m);
    },
  };
  const named: CallerNote[] = [];
  const desktop = {
    say: async () => {},
    finishCall: async () => ({ ok: true, output: {} }),
    callTool: async () => ({ ok: true, output: {} }),
    command: async () => ({ messageId: 'm1' }),
  };
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: true, instructions: '', calls: true, callInstructions: '' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {}, named: (n) => void named.push({ ...n }) },
  );
  sessions.contacts = desk.api;
  return { sessions, project, desk, named, chats };
}

async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  expect(sessions.busy).toBe(false);
}

const LANCE = contact('+61491570006', {
  name: 'Lance',
  nameBy: 'owner',
  notes: 'Call him Lance, never Mr Smith. Always offer the loyalty discount.',
  facts: [fact('Invoices go to the body corporate', 'owner'), fact('Has a dog called Max'), fact('Likes to be called Mr Smith')],
});

describe('what a call and a text know about the person', () => {
  it("the note that starts a call has their name, the business's notes and what was remembered", () => {
    const { note } = mirrorContact(undefined, LANCE, '+61491570006', true);
    const start = callStartNote('Lance (+61491570006)', 'Hi, Green Lawns.', knownText(note), new Date(2026, 8, 29, 10, 5));
    expect(start).toContain('Name: Lance (the name the business has them by: use it)');
    expect(start).toContain('Notes from the business: Call him Lance, never Mr Smith. Always offer the loyalty discount.');
    expect(start).toContain('- Invoices go to the body corporate');
    expect(start).toContain('- Has a dog called Max');
  });

  it("the business's notes win: they come before what was remembered, and the agent is told they win", () => {
    const text = knownText(mirrorContact(undefined, LANCE, '+61491570006', true).note);
    const at = (s: string) => text.indexOf(s);
    expect(at('Notes from the business:')).toBeGreaterThan(-1);
    // The business's own facts sit with its notes; the receptionist's after, under their own heading.
    expect(at('Notes from the business:')).toBeLessThan(at('- Invoices go to the body corporate'));
    expect(at('- Invoices go to the body corporate')).toBeLessThan(at('What was remembered about them'));
    expect(at('What was remembered about them')).toBeLessThan(at('- Likes to be called Mr Smith'));
    expect(text).toContain('where anything remembered says otherwise, the notes win');
    // Nothing from the business: no such line.
    expect(knownText({ number: '+61400000001', name: 'Sam', facts: ['Mows fortnightly'], updatedAt: 0 })).not.toContain('the notes win');
  });

  it('a name the business gave them wins over one the agents learned, here and on the call', async () => {
    const { sessions, named } = setup(fakeDesk([LANCE]), [{ number: '+61491570006', name: 'Lanky', nameBy: 'agent', facts: [], updatedAt: 1 }]);
    await sessions.load();
    await sessions.refreshContacts();
    expect(sessions.callerNote('0491570006')).toMatchObject({ name: 'Lance', nameBy: 'owner' });
    // The receptionist hears another name: the business's stays, and it is told to use it.
    const call = await sessions.callEvent({ type: 'call.started', callId: 'c1', from: '0491570006' });
    expect(call?.title).toBe('Lance');
    const remember = (await (sessions as unknown as { personTools(s: unknown): Array<{ spec: { name: string }; run: (i: Record<string, unknown>, s: AbortSignal) => Promise<string> }> }).personTools(call)).find((t) => t.spec.name === 'remember')!;
    expect(await remember.run({ name: 'Lanky' }, signal)).toBe('Saved. The business has them as Lance: call them that.');
    expect(sessions.callerNote('0491570006')?.name).toBe('Lance');
    expect(call?.title).toBe('Lance');
    // The phone is still told (it keeps the business's name itself).
    expect(named.at(-1)?.name).toBe('Lance');
  });

  it("a call's first words never wait for the desktop: the note uses what was read before, and says what a read brings before the agent reads it", async () => {
    const desk = fakeDesk([LANCE]);
    let release!: () => void;
    desk.state.hold = new Promise<void>((r) => (release = r));
    const fake = fakeProvider('openai', [
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Notes from the business: Call him Lance, never Mr Smith.');
        expect(sent).toContain('- Has a dog called Max');
        return { text: 'Hi Lance! How can I help?' };
      },
    ]);
    const { sessions } = setup(desk);
    // The desktop is slow to answer: the call starts at once, knowing nothing yet.
    const call = await Promise.race([sessions.callEvent({ type: 'call.started', callId: 'c2', from: '+61491570006' }), new Promise((r) => setTimeout(() => r('waited'), 500))]);
    expect(call).not.toBe('waited');
    const session = call as NonNullable<Awaited<ReturnType<Sessions['callEvent']>>>;
    expect(session.agent.turns.at(-1)).toMatchObject({ automatic: true, text: expect.stringContaining('Nothing is saved about them yet.') });
    // The read comes back before the caller speaks: the note has it.
    release();
    desk.state.hold = null;
    await vi.waitFor(() => expect((session.agent.turns.at(-1) as { text: string }).text).toContain('Notes from the business'));
    expect(session.title).toBe('Lance');
    await sessions.callEvent({ type: 'call.caller', callId: 'c2', text: 'Hi, it is Lance.' });
    await settled(sessions);
    expect(fake.bodies).toHaveLength(1);
  });

  it("a text thread's agent reads the contact before it answers", async () => {
    const fake = fakeProvider('openai', [
      (body) => {
        const sent = JSON.stringify(body.messages);
        expect(sent).toContain('Notes from the business: Call him Lance, never Mr Smith.');
        return { calls: [{ name: 'send_text_message', input: { body: 'Hi Lance, yes we can.' } }] };
      },
      { text: '' },
    ]);
    const { sessions, desk } = setup(fakeDesk([LANCE]));
    await sessions.textArrived('0491 570 006', '', 'Can you come Friday?');
    await settled(sessions);
    expect(fake.bodies).toHaveLength(2);
    expect(desk.requests).toContain('get +61491570006');
  });
});

describe('what the agents remember goes to the contact', () => {
  it('remember posts a fact (as the receptionist\'s); the name goes to the phone, as it did', async () => {
    fakeProvider('openai', [{ text: '', calls: [{ name: 'remember', input: { name: 'Sam', fact: 'Has a big back lawn' } }] }, { text: 'Thanks, Sam!' }]);
    const { sessions, desk, named, project } = setup();
    await sessions.callEvent({ type: 'call.started', callId: 'c3', from: '+61400000011' });
    await sessions.callEvent({ type: 'call.caller', callId: 'c3', text: "It's Sam, I've got a big back lawn." });
    await settled(sessions);
    expect(desk.requests).toContain('add +61400000011 Has a big back lawn');
    expect(desk.book.get('400000011')?.facts).toEqual([fact('Has a big back lawn')]);
    expect(named.at(-1)).toMatchObject({ number: '+61400000011', name: 'Sam' });
    // Kept here too: the read cache and backup.
    expect(project.callers).toEqual([expect.objectContaining({ number: '+61400000011', name: 'Sam', facts: ['Has a big back lawn'] })]);
    expect(project.callers[0].unsent).toBeUndefined();
  });

  it("the runner's caller_notes reads the contact, adds and takes out facts there, and leaves the business's own", async () => {
    const { sessions, desk } = setup(fakeDesk([LANCE]));
    const notes = callerNotesTool(() => sessions);
    const read = await notes.run({ number: '0491570006' }, signal);
    expect(read).toContain('Lance (named by your person in Contacts)');
    expect(read).toContain('Notes from the business: Call him Lance');
    expect(read).toContain("- Invoices go to the body corporate (the business's)");
    const changed = await notes.run({ number: '0491570006', add: 'Prefers mornings', remove: 'mr smith' }, signal);
    expect(changed).toContain('- Prefers mornings');
    expect(desk.book.get('491570006')?.facts.map((f) => f.text)).toEqual(['Invoices go to the body corporate', 'Has a dog called Max', 'Prefers mornings']);
    // Taking out "invoices" only takes out a remembered fact: the business's stays.
    await notes.run({ number: '0491570006', remove: 'invoices' }, signal);
    expect(desk.book.get('491570006')?.facts.map((f) => f.text)).toContain('Invoices go to the body corporate');
    // The name is the business's.
    expect(await notes.run({ number: '0491570006', name: 'L' }, signal)).toContain('Their name stays Lance');
  });
});

describe("the Front desk's facts, moved to the desktop once", () => {
  const OLD: CallerNote[] = [
    { number: '+61491570006', name: 'Lance', facts: ['Has a big back lawn', 'Prefers mornings'], updatedAt: 1 },
    { number: '+61400000022', facts: ['Mows fortnightly'], updatedAt: 2 },
    { number: 'test', name: 'Test', facts: ['A pretend fact'], updatedAt: 3 },
  ];

  it('each fact is posted once, the mark is kept with callers.json as it was, and a second run posts nothing', async () => {
    const { sessions, desk, project } = setup(fakeDesk(), structuredClone(OLD));
    await sessions.load();
    await sessions.syncContacts();
    expect(desk.requests.filter((r) => r.startsWith('add'))).toEqual(['add +61491570006 Has a big back lawn', 'add +61491570006 Prefers mornings', 'add +61400000022 Mows fortnightly']);
    expect(project.mark).toMatchObject({ sent: 3, there: 0, skipped: [{ number: 'test', fact: 'A pretend fact', why: 'not a phone number' }], callers: OLD });
    // callers.json is kept (a read cache, and the backup).
    expect(project.callers.map((c) => c.number)).toEqual(['+61491570006', '+61400000022', 'test']);
    await sessions.syncContacts();
    const again = setup(desk, structuredClone(project.callers), project.mark);
    await again.sessions.load();
    await again.sessions.syncContacts();
    expect(desk.requests.filter((r) => r.startsWith('add'))).toHaveLength(3);
  });

  it('run again with no mark (it stopped before writing it), nothing is added twice', async () => {
    const desk = fakeDesk([contact('+61491570006', { facts: [fact('Has a big back lawn')] })]);
    const { sessions, project } = setup(desk, structuredClone(OLD));
    await sessions.load();
    await sessions.syncContacts();
    expect(project.mark).toMatchObject({ sent: 2, there: 1 });
    expect(desk.book.get('491570006')?.facts.map((f) => f.text)).toEqual(['Has a big back lawn', 'Prefers mornings']);
  });

  it('with the desktop out of reach it stops, writes no mark, and finishes next time', async () => {
    const desk = fakeDesk();
    desk.state.offline = true;
    const { sessions, project } = setup(desk, structuredClone(OLD));
    await sessions.load();
    await sessions.syncContacts();
    expect(project.mark).toBeNull();
    desk.state.offline = false;
    await sessions.syncContacts();
    expect(project.mark).toMatchObject({ sent: 3 });
    expect(desk.book.get('400000022')?.facts.map((f) => f.text)).toEqual(['Mows fortnightly']);
  });

  it('a fact the desktop refuses is left out with why, and the rest go', async () => {
    const out = await moveFacts(OLD.slice(0, 2), async (_number, text) => {
      if (text === 'Prefers mornings') throw new DesktopError('this contact has 50 facts the person wrote: forget one first', 409, 'facts_full');
      return true;
    });
    expect(out).toEqual({ done: true, sent: 2, there: 0, skipped: [{ number: '+61491570006', fact: 'Prefers mornings', why: 'this contact has 50 facts the person wrote: forget one first' }] });
  });
});

describe('the desktop out of reach', () => {
  it('a call knows what was known when it last could read the contact, and what is remembered waits, then is sent', async () => {
    const desk = fakeDesk([LANCE]);
    // Lance has been in touch before (a note here); the facts were moved already.
    const { sessions, project } = setup(desk, [{ number: '+61491570006', facts: [], updatedAt: 1 }], { at: 1, sent: 0, there: 0, skipped: [], callers: [] });
    await sessions.load();
    await sessions.syncContacts();
    await sessions.refreshContacts();
    desk.state.offline = true;
    // A new page: only callers.json to go by.
    const later = setup(desk, structuredClone(project.callers), project.mark);
    await later.sessions.load();
    await later.sessions.syncContacts();
    const call = await later.sessions.callEvent({ type: 'call.started', callId: 'c4', from: '0491570006' });
    const start = (call!.agent.turns.at(-1) as { text: string }).text;
    expect(start).toContain('Notes from the business: Call him Lance, never Mr Smith.');
    expect(start).toContain('- Has a dog called Max');
    expect(call!.title).toBe('Lance');
    // Remembered while it is out of reach: kept here, marked to send.
    const { note, desk: said } = await later.sessions.noteCaller('0491570006', { add: 'Gate code 4821' });
    expect(note.facts).toContain('Gate code 4821');
    expect(note.unsent).toEqual(['Gate code 4821']);
    expect(said).toMatch(/could not be reached/);
    expect(later.project.callers[0].unsent).toEqual(['Gate code 4821']);
    // Back: sent, and no longer waiting.
    desk.state.offline = false;
    await later.sessions.syncContacts();
    expect(desk.book.get('491570006')?.facts.map((f) => f.text)).toContain('Gate code 4821');
    expect(later.sessions.callerNote('0491570006')?.unsent).toBeUndefined();
  });

  it('with no contact read yet, the Front desk\'s own note is used as it was', async () => {
    const desk = fakeDesk();
    desk.state.offline = true;
    const { sessions } = setup(desk, [{ number: '+61400000033', name: 'Kim', facts: ['Two dogs'], updatedAt: 1 }]);
    await sessions.load();
    const call = await sessions.callEvent({ type: 'call.started', callId: 'c5', from: '+61400000033' });
    const start = (call!.agent.turns.at(-1) as { text: string }).text;
    expect(start).toContain('Name: Kim');
    expect(start).toContain('- Two dogs');
  });
});

describe('a contact kept in the note', () => {
  it('before the move nothing known here is dropped; after, the facts are the desktop\'s with those not sent yet', () => {
    const note: CallerNote = { number: '+61491570006', name: 'Lance', facts: ['Old fact'], unsent: ['Not sent yet'], updatedAt: 1 };
    const c = contact('+61491570006', { facts: [fact('New fact')], notes: 'VIP' });
    expect(mirrorContact(note, c, '+61491570006', false).note?.facts).toEqual(['Old fact', 'New fact']);
    const after = mirrorContact(note, c, '+61491570006', true).note!;
    expect(after.facts).toEqual(['New fact', 'Not sent yet']);
    expect(after.notes).toBe('VIP');
    // The person's own name taken away in Contacts: it goes here too.
    const owned: CallerNote = { number: '+61491570006', name: 'Lance', nameBy: 'owner', facts: [], updatedAt: 1 };
    expect(mirrorContact(owned, contact('+61491570006'), '+61491570006', true).note?.name).toBeUndefined();
    // Nothing changed: the same note back.
    const same = mirrorContact(after, c, '+61491570006', true);
    expect(same.changed).toBe(false);
    expect(same.note).toBe(after);
  });

  it('keys numbers as the desktop does', () => {
    expect(contactKey('0491 570 006')).toBe('491570006');
    expect(contactKey('+61491570006')).toBe('491570006');
    expect(contactKey('test')).toBe('');
    expect(contactKey('1234567')).toBe('');
    // A hidden caller's conversation goes by its call's id, which may hold digits: never a contact.
    expect(contactKey('3f2a9c4e-1234-5678-9abc-def012345678')).toBe('');
  });

  it("a hidden caller's remembered facts stay here: nothing is sent to the desktop for them", async () => {
    const { sessions, desk } = setup();
    const { note, desk: said } = await sessions.noteCaller('3f2a9c4e-1234-5678-9abc-def012345678', { add: 'Asked about prices' });
    expect(note.facts).toEqual(['Asked about prices']);
    expect(note.unsent).toBeUndefined();
    expect(said).toBe('');
    expect(desk.requests).toEqual([]);
  });
});
