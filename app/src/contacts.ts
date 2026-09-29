/**
 * The people who call and text, as OAIY Desktop's Contacts keep them: the
 * person names them (one word is a whole name), leaves notes the receptionist
 * reads on every call and text with them, and sees what it remembered. The
 * phone's agents read a person's contact into what they know about them, and
 * what they remember is written there (as the receptionist's, `by: agent`).
 *
 * The Front desk keeps a copy of each person's contact as it was last read in
 * its `callers.json` (a `CallerNote` a person): what the agents go by when
 * the desktop cannot be reached, and the backup of what they knew. The facts
 * kept there before the desktop had contacts are moved to it once
 * (`moveFacts`), and a fact remembered while it was out of reach waits in
 * the note (`unsent`) until it can be sent.
 */
import { DesktopError, type Contact, type Desktop } from './desktop/bridge';
import type { CallerNote } from './vfs/projects';

/** How long a contact read from the desktop is taken as current: read again after that (in the background on a call). */
export const CONTACT_FRESH_MS = 30_000;
/** The longest a text thread's agent waits for a person's contact before it answers (a call's never waits). */
export const TEXT_CONTACT_WAIT_MS = 1_500;

/**
 * A number's key as the desktop keys contacts: its last nine digits, so
 * "0491 570 006" and "+61491570006" are one person. '' for no phone number
 * (a hidden caller, the pretend conversation, fewer than eight digits).
 */
export function contactKey(number: string): string {
  const digits = number.replace(/\D/g, '');
  if (digits.length < 8 || /^0+$/.test(digits)) return '';
  return digits.slice(-9);
}

/** What the phone's conversations need of the desktop's contacts. Each throws when the desktop cannot be reached. */
export interface ContactsApi {
  /** The contact for a number written any way; null when there is none. */
  get(number: string): Promise<Contact | null>;
  /** Every contact. */
  list(): Promise<Contact[]>;
  /** Something the receptionist remembered (`by: agent`): kept once. */
  addFact(number: string, text: string): Promise<{ contact: Contact | null; added: boolean }>;
  /** Forget a fact by its place, while it still says `text`. */
  forgetFact(number: string, index: number, text: string): Promise<Contact | null>;
}

/** The contacts of the desktop `desktop()` gives now (none: out of reach). Each request gives up after `timeoutMs`. */
export function desktopContacts(desktop: () => Desktop | null, timeoutMs = 5_000): ContactsApi {
  const d = () => {
    const now = desktop();
    if (!now) throw new DesktopError('OAIY Desktop is not connected');
    return now;
  };
  return {
    get: (number) => d().contact(number, AbortSignal.timeout(timeoutMs)),
    list: () => d().contacts('', AbortSignal.timeout(timeoutMs)),
    addFact: (number, text) => d().addContactFact(number, text, 'agent', AbortSignal.timeout(timeoutMs)),
    forgetFact: (number, index, text) => d().forgetContactFact(number, index, text, AbortSignal.timeout(timeoutMs)),
  };
}

/**
 * Whether a failure is the desktop refusing what was sent (a hidden number,
 * a fact too long, a full list of the person's own facts), as opposed to not
 * being reached (or being a desktop from before it kept contacts): sending
 * it again would be refused again.
 */
export function refused(error: unknown): boolean {
  return error instanceof DesktopError && !!error.code && error.status >= 400 && error.status < 500 && ![401, 403, 404, 408, 429].includes(error.status);
}

/** Whether two facts say the same (in any case). */
export const sameFact = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
/** `list` with each of `more` not in it already (in any case) after it. */
export const unionFacts = (list: readonly string[], more: readonly string[]) => {
  const out = [...list];
  for (const m of more) if (!out.some((x) => sameFact(x, m))) out.push(m);
  return out;
};
const equal = (a: readonly string[] | undefined, b: readonly string[] | undefined) => (a ?? []).length === (b ?? []).length && (a ?? []).every((x, i) => x === b![i]);

/** The name a person goes by, and who gave it: the person's own (in Contacts) first; else the agents' latest here (the desktop is given it); else the desktop's. */
function nameOf(note: CallerNote | undefined, contact: Contact | null, moved: boolean): { name?: string; nameBy?: 'owner' | 'agent' } {
  if (contact?.name && contact.nameBy === 'owner') return { name: contact.name, nameBy: 'owner' };
  // The person's own name was changed or taken away there: theirs goes here too.
  if (contact && note?.nameBy === 'owner') return contact.name ? { name: contact.name, ...(contact.nameBy ? { nameBy: contact.nameBy } : {}) } : {};
  if (note?.name) return { name: note.name, ...(note.nameBy && !(moved && !contact && note.nameBy === 'owner') ? { nameBy: note.nameBy } : {}) };
  return contact?.name ? { name: contact.name, ...(contact.nameBy ? { nameBy: contact.nameBy } : {}) } : {};
}

/**
 * A person's note as their contact says (`contact` null: they have none): the
 * name (see `nameOf`: a name the person gave them in Contacts always wins),
 * who gave it, the person's notes and own facts, and the receptionist's
 * facts. Before the facts kept here were moved to the desktop (`moved`
 * false) none of them is dropped; after, the facts are the desktop's (one
 * forgotten there is forgotten here) with those not sent yet. `note` is left
 * as it was; `changed` says whether the one returned differs.
 */
export function mirrorContact(note: CallerNote | undefined, contact: Contact | null, number: string, moved: boolean, now = Date.now()): { note: CallerNote | undefined; changed: boolean } {
  if (!contact && !note) return { note: undefined, changed: false };
  const unsent = note?.unsent ?? [];
  const agentFacts = (contact?.facts ?? []).filter((f) => f.by !== 'owner').map((f) => f.text);
  const ownerFacts = (contact?.facts ?? []).filter((f) => f.by === 'owner').map((f) => f.text);
  const next: CallerNote = {
    number: note?.number ?? number,
    facts: moved ? unionFacts(agentFacts, unsent) : unionFacts(note?.facts ?? [], agentFacts),
    updatedAt: note?.updatedAt ?? now,
  };
  const { name, nameBy } = nameOf(note, contact, moved);
  if (name) next.name = name;
  if (nameBy) next.nameBy = nameBy;
  const notes = contact ? contact.notes.trim() : moved ? '' : note?.notes ?? '';
  if (notes) next.notes = notes;
  const owned = contact ? ownerFacts : moved ? [] : note?.ownerFacts ?? [];
  if (owned.length) next.ownerFacts = owned;
  if (unsent.length) next.unsent = [...unsent];
  const changed = !note || note.name !== next.name || note.nameBy !== next.nameBy || (note.notes ?? '') !== (next.notes ?? '') || !equal(note.facts, next.facts) || !equal(note.ownerFacts, next.ownerFacts);
  if (changed) next.updatedAt = now;
  return { note: changed ? next : note, changed };
}

/** What moving the facts did: whether it finished (false: the desktop could not be reached, so it is tried again later). */
export interface Moved {
  done: boolean;
  sent: number;
  there: number;
  skipped: Array<{ number: string; fact: string; why: string }>;
}

/**
 * The facts of `notes` sent to the desktop's contacts, one request a fact
 * (`add` answers whether it was new): the desktop keeps each once, so sending
 * one it has changes nothing. A note of no phone number (the pretend
 * conversation, a hidden caller's) and a fact the desktop refuses are left
 * out, with why. It stops at the first failure to reach the desktop.
 */
export async function moveFacts(notes: readonly CallerNote[], add: (number: string, text: string) => Promise<boolean>): Promise<Moved> {
  const out: Moved = { done: false, sent: 0, there: 0, skipped: [] };
  for (const note of notes) {
    const facts = unionFacts(note.facts, note.unsent ?? []);
    if (!contactKey(note.number)) {
      for (const fact of facts) out.skipped.push({ number: note.number, fact, why: 'not a phone number' });
      continue;
    }
    for (const fact of facts) {
      try {
        if (await add(note.number, fact)) out.sent++;
        else out.there++;
      } catch (error) {
        if (!refused(error)) return out;
        out.skipped.push({ number: note.number, fact, why: (error as Error).message });
      }
    }
  }
  out.done = true;
  return out;
}
