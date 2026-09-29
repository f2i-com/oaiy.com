import type { Contact, ContactFact } from './api';

/**
 * The Contacts page's words and numbers, kept pure so they are tested alone:
 * a number as a person reads it, the search the list filters by (the same
 * rules as the desktop's `GET /api/contacts?q=`), the export's file name, and
 * the import's countries and reasons.
 */

/** Only digits. */
const digitsOf = (s: string) => s.replace(/\D/g, '');

/**
 * A number as a person reads it. An Australian one the local way: a mobile
 * "0491 570 006", a landline "(02) 9876 5432". Any other as it was kept. A
 * contact from before numbers were kept has only its key (the last nine
 * digits), read as Australian, as the desktop assumes.
 */
export function readableNumber(number: string, key = ''): string {
  const raw = (number ?? '').trim();
  const digits = digitsOf(raw);
  let local: string | null = null;
  if (raw.startsWith('+')) {
    if (digits.startsWith('61') && digits.length === 11) local = `0${digits.slice(2)}`;
  } else if (digits.length === 10 && digits.startsWith('0')) {
    local = digits;
  } else if (!raw && /^[2-8]\d{8}$/.test(key)) {
    local = `0${key}`;
  }
  if (!local) return raw || key;
  if (/^0[45]/.test(local)) return `${local.slice(0, 4)} ${local.slice(4, 7)} ${local.slice(7)}`;
  if (/^0[2378]/.test(local)) return `(${local.slice(0, 2)}) ${local.slice(2, 6)} ${local.slice(6)}`;
  return local;
}

/** What a contact is called on the page: their name, else their number. */
export function contactLabel(c: Pick<Contact, 'name' | 'number' | 'key'>): string {
  return c.name.trim() || readableNumber(c.number, c.key);
}

/** One or two letters for a contact's badge: "Lance" is L, "Lance Smith" is LS. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter((w) => /\p{L}/u.test(w));
  const letter = (w: string) => [...w].find((ch) => /\p{L}/u.test(ch))?.toUpperCase() ?? '';
  if (!words.length) return '';
  return words.length === 1 ? letter(words[0]) : letter(words[0]) + letter(words[words.length - 1]);
}

/** A number's key, as the desktop makes it: its last nine digits (null when it has fewer than eight). */
export function numberKey(number: string): string | null {
  const d = digitsOf(number);
  return d.length < 8 ? null : d.slice(-9);
}

/** What a phone shows for a caller who hides their number. */
const HIDDEN = ['unknown', 'private', 'withheld', 'anonymous', 'restricted', 'blocked', 'unavailable', 'hidden', 'no caller', 'payphone'];

/** Why a number typed for a new contact cannot be one, or null when it can. */
export function numberProblem(number: string): string | null {
  const t = number.trim();
  if (!t) return 'Give their phone number.';
  const lower = t.toLowerCase();
  const d = digitsOf(t);
  if (HIDDEN.some((w) => lower.includes(w)) || (d && /^0+$/.test(d))) return 'A hidden number cannot be a contact.';
  if (d.length < 8) return 'That is not a phone number: it needs at least 8 digits.';
  return null;
}

/**
 * Whether `q` finds `c`: its name, notes, what was remembered or its number,
 * in any case; a number typed any way (its digits, with or without the leading 0).
 */
export function matchesContact(c: Contact, q: string): boolean {
  const query = q.trim();
  if (!query) return true;
  const lower = query.toLowerCase();
  if ([c.name, c.notes, c.number].some((w) => w.toLowerCase().includes(lower)) || c.facts.some((f) => f.text.toLowerCase().includes(lower))) return true;
  if (!/^[\d\s+\-().]+$/.test(query)) return false;
  const d = digitsOf(query);
  if (!d) return false;
  const local = d.replace(/^0+/, '');
  const number = digitsOf(c.number);
  return [d, local].filter(Boolean).some((x) => number.includes(x) || c.key.includes(x));
}

/** A fact's identity while the page holds it (its text and when it was remembered). */
export const factId = (f: ContactFact) => `${f.at}\u0000${f.text}`;

/** The export's file name: `oaiy-contacts-2026-09-29.csv`, by this computer's date. */
export function csvFileName(now = new Date()): string {
  const two = (n: number) => String(n).padStart(2, '0');
  return `oaiy-contacts-${now.getFullYear()}-${two(now.getMonth() + 1)}-${two(now.getDate())}.csv`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** When something was remembered, shortly: "today", "yesterday", "12 Sep", "12 Sep 2025". */
export function saidWhen(iso: string, now = new Date()): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return '';
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(now) - day(at)) / 86_400_000);
  if (days === 0) return 'today';
  if (days === 1) return 'yesterday';
  const month = MONTHS[at.getMonth()];
  return at.getFullYear() === now.getFullYear() ? `${at.getDate()} ${month}` : `${at.getDate()} ${month} ${at.getFullYear()}`;
}

/** The countries an import's local numbers may be from (the desktop's `COUNTRIES`); Australia unless chosen. */
export const IMPORT_COUNTRIES = [
  { code: 'AU', name: 'Australia' },
  { code: 'NZ', name: 'New Zealand' },
  { code: 'GB', name: 'United Kingdom' },
  { code: 'IE', name: 'Ireland' },
  { code: 'US', name: 'United States' },
  { code: 'CA', name: 'Canada' },
  { code: 'ZA', name: 'South Africa' },
  { code: 'IN', name: 'India' },
  { code: 'SG', name: 'Singapore' },
] as const;

/** Why an import skips a row, as a count reads: "2 with no number". */
export const SKIP_REASONS: Record<string, [one: string, many: string]> = {
  no_number: ['with no number', 'with no number'],
  not_a_phone_number: ['that is not a phone number', 'that are not phone numbers'],
  hidden: ['with a hidden number', 'with hidden numbers'],
  duplicate: ['with the same number as an earlier row', 'with the same number as an earlier row'],
  notes_too_long: ['with notes over 2,000 characters', 'with notes over 2,000 characters'],
};

/** "2 with no number", "1 that is not a phone number". */
export function skipReason(reason: string, n: number): string {
  const words = SKIP_REASONS[reason];
  return `${n} ${words ? words[n === 1 ? 0 : 1] : reason.replace(/_/g, ' ')}`;
}
