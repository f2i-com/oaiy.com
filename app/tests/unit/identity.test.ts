// Who answers the phone, and for whom: the receptionist's name ("Aokie"
// unless the person set one) and the business's, from the desktop's calendar
// settings, in every agent a customer talks to, filled into outreach's
// templates, and never "OAIY" or "your person" in what a customer hears.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Sessions, callInstructions, smsInstructions } from '../../src/sessions';
import { DEFAULT_RECEPTIONIST, IdentityCache, NO_IDENTITY, identityFrom, identityInstructions, identityNote, type Identity } from '../../src/identity';
import { fill, outreachCallInstructions, outreachTextInstructions, planOutreach, speaksAs, type Campaign, type Person } from '../../src/outreach';
import { setLocalCountry } from '../../src/phoneNumbers';
import type { Desktop } from '../../src/desktop/bridge';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());
setLocalCountry('AU');

const GREEN: Identity = { business: 'Green Lawns', receptionist: 'Aokie' };
const NEVER = 'never "OAIY", never "your person"';

describe("the desktop's names", () => {
  it('reads the receptionist\'s name the desktop gives, and "Aokie" when it gives none', () => {
    expect(identityFrom({ settings: { business: 'Green Lawns' }, receptionistName: 'Mia' })).toEqual({ business: 'Green Lawns', receptionist: 'Mia' });
    // A desktop from before it kept a name: no receptionistName at all.
    expect(identityFrom({ settings: { business: 'Green Lawns' } })).toEqual({ business: 'Green Lawns', receptionist: 'Aokie' });
    expect(identityFrom({ settings: {}, receptionistName: '  ' })).toEqual({ business: '', receptionist: DEFAULT_RECEPTIONIST });
    expect(identityFrom(null)).toEqual(NO_IDENTITY);
  });

  it('is kept as last read: at once, asked again in the background, and kept while the desktop is out of reach', async () => {
    let answer: unknown = { settings: { business: 'Green Lawns' }, receptionistName: 'Aokie' };
    let asked = 0;
    const desktop = { calendar: async () => {
      asked++;
      if (answer instanceof Error) throw answer;
      return answer;
    } } as unknown as Desktop;
    const cache = new IdentityCache(() => desktop, 0);
    expect(cache.get()).toEqual(NO_IDENTITY);
    await cache.refresh();
    expect(cache.get()).toEqual(GREEN);
    answer = new TypeError('Failed to fetch');
    await cache.refresh();
    expect(cache.get()).toEqual(GREEN);
    answer = { settings: { business: 'Blue Pools' }, receptionistName: 'Mia' };
    await cache.refresh();
    expect(cache.get()).toEqual({ business: 'Blue Pools', receptionist: 'Mia' });
    expect(asked).toBeGreaterThanOrEqual(3);
  });
});

describe('every agent a customer talks to says who it is, for whom', () => {
  it('a call, a text thread, an outreach call and an outreach text', () => {
    const who = 'You are Aokie, the receptionist for Green Lawns.';
    expect(callInstructions('', '', false, GREEN)).toContain(who);
    expect(callInstructions('', '', false, GREEN)).toContain(NEVER);
    expect(smsInstructions('Jane', '+61412345678', '', false, '', new Date(), GREEN)).toContain(who);
    expect(smsInstructions('Jane', '+61412345678', '', false, '', new Date(), GREEN)).toContain(NEVER);
    const c = { name: 'Confirm Friday', objective: 'Confirm Friday.', openingLine: "Hi {first_name}, it's {receptionist} from {business} about Friday.", voicemail: 'no_message', voicemailMessage: '', collect: [], identity: GREEN } as unknown as Campaign;
    const p = { name: 'Jane Smith', number: '+61412345678', fields: {} } as unknown as Person;
    expect(outreachCallInstructions(c, p)).toContain('you are calling on behalf of Green Lawns, as Aokie');
    expect(outreachCallInstructions(c, p)).toContain(`You already said: "Hi Jane, it's Aokie from Green Lawns about Friday."`);
    expect(outreachTextInstructions(c, p)).toContain('You texted them on behalf of Green Lawns, as Aokie');
  });

  it('with no business name yet it says so, and the receptionist is still Aokie', () => {
    expect(identityInstructions(NO_IDENTITY)).toMatch(/^You are Aokie, the business's receptionist \(its name is not set yet/);
  });

  it('the runner and a project\'s agent know both names, and to write the placeholders', () => {
    const note = identityNote(GREEN);
    expect(note).toContain('The phone is answered as Aokie, the receptionist for Green Lawns.');
    expect(note).toContain('{receptionist} and {business}');
    expect(note).toContain('never "OAIY" or "your person"');
  });

  it("a call's and a text's agent are sent it", async () => {
    const chats = new Map<string, Turn[]>();
    let index: SessionInfo[] = [];
    const project = {
      loadSessions: async () => index,
      saveSessions: async (list: SessionInfo[]) => void (index = list),
      loadSessionChat: async (id: string) => chats.get(id) ?? [],
      saveSessionChat: async (id: string, turns: Turn[]) => void chats.set(id, turns),
      loadCallers: async () => [] as CallerNote[],
      saveCallers: async () => {},
    };
    const desktop = { say: async () => {}, command: async () => ({ messageId: 'm1' }), callTool: async () => ({ ok: true, output: {} }), finishCall: async () => ({ ok: true, output: {} }) };
    const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: true, instructions: '', calls: true, callInstructions: '' };
    const sessions = new Sessions(project as never, (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...extra }), () => messages, () => desktop as unknown as Desktop, { changed: () => {}, event: () => {} });
    sessions.identity = () => ({ business: 'Green Lawns', receptionist: 'Mia' });
    const systems: string[] = [];
    const system = (body: Record<string, unknown>) => String((body.messages as Array<{ role: string; content: unknown }>).find((m) => m.role === 'system')?.content ?? '');
    fakeProvider('openai', [
      (body) => {
        systems.push(system(body));
        return { text: 'Hi, Mia here at Green Lawns.' };
      },
      (body) => {
        systems.push(system(body));
        return { calls: [{ name: 'send_text_message', input: { body: 'Hi, Mia from Green Lawns.' } }] };
      },
      { text: '' },
    ]);
    await sessions.callEvent({ type: 'call.started', callId: 'c1', from: '+61400000001' });
    await sessions.callEvent({ type: 'call.caller', callId: 'c1', text: 'Hello?' });
    for (let i = 0; i < 200 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
    await sessions.textArrived('+61400000002', '', 'Hi');
    for (let i = 0; i < 200 && (sessions.busy || systems.length < 2); i++) await new Promise((r) => setTimeout(r, 10));
    expect(systems).toHaveLength(2);
    for (const s of systems) expect(s).toContain('You are Mia, the receptionist for Green Lawns.');
  });
});

describe("outreach's templates", () => {
  const ctx = (identity: Identity = GREEN) => ({ identity, screening: null, doNotContact: [], inTextCampaign: () => null, slugs: new Set<string>() });
  const CALLS = { kind: 'call', name: 'Confirm Friday', objective: 'Confirm Friday.', people: [{ name: 'Jane Smith', number: '0412 345 678' }] };

  it('fills {receptionist} and {business}, as well as {first_name}', () => {
    expect(fill("Hi {first_name}, it's {receptionist} from {business}.", { name: 'Jane Smith', fields: {} }, GREEN)).toEqual({ text: "Hi Jane, it's Aokie from Green Lawns.", missing: [] });
    const plan = planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's {receptionist} from {business} about Friday." }, ctx());
    expect(typeof plan).toBe('object');
    expect(typeof plan === 'object' && plan.identity).toEqual(GREEN);
    const texts = planOutreach({ ...CALLS, kind: 'text', textTemplate: 'Hi {first_name}, {business} here: still right for Friday?' }, ctx());
    expect(typeof texts).toBe('object');
  });

  it('refuses an opening that says "OAIY" or "your person", and tells the runner to use the placeholders', () => {
    const oaiy = planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's OAIY calling for your person, just wanted to check one thing about your bookings." }, ctx());
    expect(oaiy).toMatch(/^The opening line says "OAIY": customers hear the business and its receptionist/);
    expect(oaiy).toContain("{receptionist} and {business} (they are filled in), e.g. \"Hi {first_name}, it's {receptionist} from {business} about …\"");
    expect(planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's Aokie from Green Lawns for your person." }, ctx())).toMatch(/says "your person"/);
    expect(planOutreach({ ...CALLS, kind: 'text', textTemplate: 'Hi {first_name}, OAIY here: still right for Friday?' }, ctx())).toMatch(/^The text says "OAIY"/);
    // A voicemail message is heard too.
    expect(planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's {receptionist} from {business}.", voicemail: 'leave_message', voicemailMessage: 'Hi, calling for your person.' }, ctx())).toMatch(/voicemail message says "your person"/);
  });

  it('refuses one that names neither the business nor the receptionist', () => {
    expect(planOutreach({ ...CALLS, openingLine: 'Hi {first_name}, just checking about Friday.' }, ctx())).toMatch(/^The opening line names neither "Aokie" nor "Green Lawns": say who is calling with \{receptionist\} and \{business\}/);
    // Either name is enough.
    expect(typeof planOutreach({ ...CALLS, openingLine: 'Hi {first_name}, Green Lawns here about Friday.' }, ctx())).toBe('object');
    expect(speaksAs("Hi Jane, it's Aokie.", GREEN, 'opening line')).toBe('');
  });

  it('with no business name set, {business} is refused and the receptionist\'s name is enough', () => {
    expect(planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's {receptionist} from {business}." }, ctx(NO_IDENTITY))).toMatch(/^\{business\} has no value: the business's name is not set/);
    expect(typeof planOutreach({ ...CALLS, openingLine: "Hi {first_name}, it's {receptionist} about Friday." }, ctx(NO_IDENTITY))).toBe('object');
    expect(planOutreach({ ...CALLS, openingLine: 'Hi {first_name}, about Friday.' }, ctx(NO_IDENTITY))).toMatch(/does not say "Aokie"/);
  });
});
