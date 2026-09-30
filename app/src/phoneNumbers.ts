/**
 * Phone numbers as people. One person's number, however a phone writes it
 * ("0491 570 006", "+61491570006", "0011 61 491 570 006", "(04) 9157-0006"),
 * is one key: its E.164 form, "+61491570006". A number written without its
 * country ("0491570006", as the phone's caller id gives a local call) is read
 * with the country for local numbers (the Phone dialog's setting; by default
 * the computer's own). A caller who hides their number, a short code
 * ("13 22 11") and a sender id of letters ("Telstra") are not phone numbers:
 * each stays its own, never merged with a number or with each other.
 */

export interface Country {
  /** ISO 3166 code: "AU". */
  code: string;
  name: string;
  /** Its calling code, without the "+": "61". */
  dial: string;
  /** What a number dialled within the country starts with before the rest ("0"; "1" in North America, where it may be left out; "" where there is none). */
  trunk: string;
  /** How a call abroad starts from there ("0011" in Australia, "011" in North America): read before the "00" most countries use. */
  exit: string;
  /** How many digits a number has after the calling code (without the trunk): the fewest and the most. */
  digits: [number, number];
  /** What that part starts with: a local number that does not (a short code, a 13 or 1300 number) is left as it is. */
  lead: RegExp;
}

const C = (code: string, name: string, dial: string, trunk: string, exit: string, digits: [number, number], lead = /^[1-9]/): Country => ({ code, name, dial, trunk, exit, digits, lead });

/** The countries a local number can be read for (the Phone dialog lists them). */
export const COUNTRIES: Country[] = [
  C('AU', 'Australia', '61', '0', '0011', [9, 9], /^[2-478]/),
  C('NZ', 'New Zealand', '64', '0', '00', [8, 10], /^[2-9]/),
  C('GB', 'United Kingdom', '44', '0', '00', [9, 10]),
  C('IE', 'Ireland', '353', '0', '00', [7, 9]),
  C('US', 'United States', '1', '1', '011', [10, 10], /^[2-9]\d\d[2-9]/),
  C('CA', 'Canada', '1', '1', '011', [10, 10], /^[2-9]\d\d[2-9]/),
  C('IN', 'India', '91', '0', '00', [10, 10]),
  C('SG', 'Singapore', '65', '', '000', [8, 8], /^[3689]/),
  C('HK', 'Hong Kong', '852', '', '001', [8, 8], /^[2-9]/),
  C('MY', 'Malaysia', '60', '0', '00', [8, 10]),
  C('PH', 'Philippines', '63', '0', '00', [8, 10], /^[2-9]/),
  C('ID', 'Indonesia', '62', '0', '001', [8, 12]),
  C('TH', 'Thailand', '66', '0', '001', [8, 9], /^[2-9]/),
  C('VN', 'Vietnam', '84', '0', '00', [9, 10]),
  C('CN', 'China', '86', '0', '00', [9, 11]),
  C('JP', 'Japan', '81', '0', '010', [9, 10]),
  C('KR', 'South Korea', '82', '0', '001', [8, 10]),
  C('TW', 'Taiwan', '886', '0', '002', [8, 9], /^[2-9]/),
  C('ZA', 'South Africa', '27', '0', '00', [9, 9]),
  C('AE', 'United Arab Emirates', '971', '0', '00', [8, 9], /^[2-9]/),
  C('IL', 'Israel', '972', '0', '00', [8, 9], /^[2-9]/),
  C('DE', 'Germany', '49', '0', '00', [6, 11]),
  C('FR', 'France', '33', '0', '00', [9, 9]),
  C('IT', 'Italy', '39', '', '00', [6, 11], /^[03]/),
  C('ES', 'Spain', '34', '', '00', [9, 9], /^[6-9]/),
  C('NL', 'Netherlands', '31', '0', '00', [9, 9]),
  C('BE', 'Belgium', '32', '0', '00', [8, 9]),
  C('CH', 'Switzerland', '41', '0', '00', [9, 9]),
  C('AT', 'Austria', '43', '0', '00', [4, 13]),
  C('SE', 'Sweden', '46', '0', '00', [7, 9]),
  C('NO', 'Norway', '47', '', '00', [8, 8], /^[2-9]/),
  C('DK', 'Denmark', '45', '', '00', [8, 8], /^[2-9]/),
  C('FI', 'Finland', '358', '0', '00', [5, 10]),
  C('PL', 'Poland', '48', '', '00', [9, 9]),
  C('PT', 'Portugal', '351', '', '00', [9, 9], /^[29]/),
  C('GR', 'Greece', '30', '', '00', [10, 10], /^[26]/),
  C('BR', 'Brazil', '55', '0', '00', [10, 11]),
  C('MX', 'Mexico', '52', '', '00', [10, 10]),
  C('AR', 'Argentina', '54', '0', '00', [10, 10]),
  C('FJ', 'Fiji', '679', '', '00', [7, 7], /^[2-9]/),
  C('PG', 'Papua New Guinea', '675', '', '00', [7, 8]),
];

const BY_CODE = new Map(COUNTRIES.map((c) => [c.code, c]));
/** The country this business is in when nothing says otherwise. */
export const FALLBACK_COUNTRY = 'AU';

export function countryOf(code: string | Country | undefined): Country {
  if (code && typeof code === 'object') return code;
  return BY_CODE.get(String(code ?? '').toUpperCase()) ?? BY_CODE.get(FALLBACK_COUNTRY)!;
}

/** Time zones that say which country the computer is in (the region of a language like "en-US" often does not: Windows in Australia is often in US English). */
const ZONES: Array<[RegExp, string]> = [
  [/^Australia\//, 'AU'],
  [/^Pacific\/(Auckland|Chatham)$/, 'NZ'],
  [/^Europe\/(London|Belfast)$/, 'GB'],
  [/^Europe\/Dublin$/, 'IE'],
  [/^America\/(Toronto|Vancouver|Edmonton|Winnipeg|Halifax|St_Johns|Regina|Montreal|Moncton|Whitehorse|Yellowknife|Iqaluit)$/, 'CA'],
  [/^(America\/(New_York|Chicago|Denver|Los_Angeles|Phoenix|Anchorage|Detroit|Boise|Juneau|Indiana\/.+|Kentucky\/.+|North_Dakota\/.+)|Pacific\/Honolulu)$/, 'US'],
  [/^Asia\/(Kolkata|Calcutta)$/, 'IN'],
  [/^Asia\/Singapore$/, 'SG'],
  [/^Asia\/Hong_Kong$/, 'HK'],
  [/^Asia\/(Kuala_Lumpur|Kuching)$/, 'MY'],
  [/^Asia\/Manila$/, 'PH'],
  [/^Asia\/(Jakarta|Makassar|Jayapura|Pontianak)$/, 'ID'],
  [/^Asia\/Bangkok$/, 'TH'],
  [/^Asia\/(Ho_Chi_Minh|Saigon)$/, 'VN'],
  [/^Asia\/(Shanghai|Chongqing|Urumqi)$/, 'CN'],
  [/^Asia\/Tokyo$/, 'JP'],
  [/^Asia\/Seoul$/, 'KR'],
  [/^Asia\/Taipei$/, 'TW'],
  [/^Africa\/Johannesburg$/, 'ZA'],
  [/^Asia\/Dubai$/, 'AE'],
  [/^Asia\/(Jerusalem|Tel_Aviv)$/, 'IL'],
  [/^Europe\/(Berlin|Busingen)$/, 'DE'],
  [/^Europe\/Paris$/, 'FR'],
  [/^Europe\/Rome$/, 'IT'],
  [/^(Europe\/Madrid|Atlantic\/Canary)$/, 'ES'],
  [/^Europe\/Amsterdam$/, 'NL'],
  [/^Europe\/Brussels$/, 'BE'],
  [/^Europe\/Zurich$/, 'CH'],
  [/^Europe\/Vienna$/, 'AT'],
  [/^Europe\/Stockholm$/, 'SE'],
  [/^Europe\/Oslo$/, 'NO'],
  [/^Europe\/Copenhagen$/, 'DK'],
  [/^Europe\/Helsinki$/, 'FI'],
  [/^Europe\/Warsaw$/, 'PL'],
  [/^(Europe\/Lisbon|Atlantic\/(Madeira|Azores))$/, 'PT'],
  [/^Europe\/Athens$/, 'GR'],
  [/^America\/(Sao_Paulo|Fortaleza|Recife|Bahia|Belem|Manaus|Cuiaba|Campo_Grande|Porto_Velho|Boa_Vista|Rio_Branco|Maceio|Araguaina|Santarem|Noronha)$/, 'BR'],
  [/^America\/(Mexico_City|Cancun|Merida|Monterrey|Matamoros|Chihuahua|Mazatlan|Hermosillo|Tijuana|Bahia_Banderas)$/, 'MX'],
  [/^America\/(Argentina\/.+|Buenos_Aires)$/, 'AR'],
  [/^Pacific\/Fiji$/, 'FJ'],
  [/^Pacific\/(Port_Moresby|Bougainville)$/, 'PG'],
];

/**
 * The computer's country, for numbers written without one: its time zone
 * (Australia/Sydney: Australia), else the region of its languages ("en-AU"),
 * else Australia (this business is Australian). The time zone comes first
 * because an Australian computer often speaks US English.
 */
export function detectCountry(languages?: readonly string[], timeZone?: string): string {
  let zone = timeZone;
  if (zone === undefined) {
    try {
      zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    } catch {
      zone = '';
    }
  }
  const fromZone = ZONES.find(([re]) => re.test(zone ?? ''))?.[1];
  if (fromZone) return fromZone;
  let tags = languages;
  if (tags === undefined) {
    try {
      tags = typeof navigator !== 'undefined' ? (navigator.languages?.length ? navigator.languages : [navigator.language]) : [];
    } catch {
      tags = [];
    }
  }
  for (const tag of tags ?? []) {
    const region = /^[a-z]{2,3}(?:-[a-z]{4})?-([a-z]{2})(?:-|$)/i.exec(tag ?? '')?.[1]?.toUpperCase();
    if (region && BY_CODE.has(region)) return region;
  }
  return FALLBACK_COUNTRY;
}

let local: string | null = null;

/** The country local numbers are read for: the setting ('' is automatic, the computer's own). */
export function setLocalCountry(code: string): void {
  local = code && BY_CODE.has(code.toUpperCase()) ? code.toUpperCase() : detectCountry();
}

/** The country local numbers are read for now. */
export function localCountry(): string {
  return (local ??= detectCountry());
}

/** Words a phone shows for a caller who hides their number. */
const HIDDEN = /^(unknown|private|anonymous|withheld|restricted|unavailable|blocked|hidden|no caller id|no number|caller id (withheld|blocked)|number withheld|payphone|none|null)\b/i;

/** Whether a caller id is no number at all: empty, or "Private", "Unknown", "Withheld"… */
export function isHidden(raw: string | null | undefined): boolean {
  const text = String(raw ?? '').trim();
  return !text || !/[\p{L}\p{N}]/u.test(text) || HIDDEN.test(text);
}

/** Calling codes of one digit, and of two; every other one has three (they never begin one another). */
const TWO_DIGIT = new Set(['20', '27', '30', '31', '32', '33', '34', '36', '39', '40', '41', '43', '44', '45', '46', '47', '48', '49', '51', '52', '53', '54', '55', '56', '57', '58', '60', '61', '62', '63', '64', '65', '66', '81', '82', '84', '86', '90', '91', '92', '93', '94', '95', '98']);

/** An E.164 number's calling code and the rest: "+61491570006" → ["61", "491570006"]. */
export function splitE164(e164: string): [string, string] {
  const d = e164.replace(/^\+/, '');
  const n = d[0] === '1' || d[0] === '7' ? 1 : TWO_DIGIT.has(d.slice(0, 2)) ? 2 : 3;
  return [d.slice(0, n), d.slice(n)];
}

/** Digits after an international prefix as E.164, or null when they cannot be one. */
function international(digits: string): string | null {
  if (!/^[1-9]\d{6,14}$/.test(digits)) return null;
  const [cc, rest] = splitE164(digits);
  // "+61 0432…": the trunk written after the calling code is not dialled (Italy's leading 0 is part of the number).
  const trunk0 = COUNTRIES.some((c) => c.dial === cc && c.trunk === '0');
  const nsn = trunk0 && rest.startsWith('0') ? rest.slice(1) : rest;
  return nsn.length >= 4 ? `+${cc}${nsn}` : null;
}

const fits = (c: Country, nsn: string) => nsn.length >= c.digits[0] && nsn.length <= c.digits[1] && c.lead.test(nsn);

/**
 * A caller id as E.164 ("+61491570006"), reading a number written without its
 * country as `country`'s; null when it is not a phone number (hidden, a name,
 * a short code, or digits that cannot be read as one).
 */
export function toE164(raw: string | null | undefined, country: string | Country = localCountry()): string | null {
  if (isHidden(raw)) return null;
  const c = countryOf(country);
  const s = String(raw)
    .trim()
    .replace(/^tel:/i, '')
    // "+61 (0)4…": the bracketed trunk is not dialled.
    .replace(/\(0\)/g, '')
    // Spaces, dashes, dots, slashes and brackets are how it is written, not what it is.
    .replace(/[\s\-./() ‐-―]/g, '');
  if (!/^\+?\d+$/.test(s)) return null;
  if (s.startsWith('+')) return international(s.slice(1));
  if (c.exit && s.startsWith(c.exit)) return international(s.slice(c.exit.length));
  // "00" starts a call abroad nearly everywhere (not in North America, whose numbers never start with 0).
  if (c.dial !== '1' && s.startsWith('00')) return international(s.slice(2));
  if (c.trunk && s.startsWith(c.trunk) && fits(c, s.slice(c.trunk.length))) return `+${c.dial}${s.slice(c.trunk.length)}`;
  // No trunk to write (or, in North America, one that may be left out).
  if ((!c.trunk || c.dial === '1') && fits(c, s)) return `+${c.dial}${s}`;
  // Its calling code without the "+" ("61491570006"): longer than any local number.
  if (s.startsWith(c.dial) && fits(c, s.slice(c.dial.length))) return `+${s}`;
  return null;
}

/**
 * The key a caller is known by: their number's E.164 form; anything else as
 * it is (a short code's digits, a sender id of letters). A hidden caller has
 * none (''): each such call is its own.
 */
export function phoneKey(raw: string | null | undefined, country: string | Country = localCountry()): string {
  if (isHidden(raw)) return '';
  const e164 = toE164(raw, country);
  if (e164) return e164;
  const text = String(raw).trim();
  const compact = text.replace(/[\s\-./()]/g, '');
  return /^\+?\d+$/.test(compact) ? compact : text;
}

const digitsOf = (text: string) => text.replace(/\D/g, '');

/**
 * Whether two caller ids are the same person: the same E.164 number. A number
 * the country's rules cannot read (a foreign one, written as it is dialled
 * there) matches by its last nine digits, as the phone itself matches numbers.
 * Hidden callers, short codes and names are never the same as anyone else.
 */
export function samePerson(a: string | null | undefined, b: string | null | undefined, country: string | Country = localCountry()): boolean {
  if (isHidden(a) || isHidden(b)) return false;
  const [x, y] = [phoneKey(a, country), phoneKey(b, country)];
  if (x === y) return true;
  if (toE164(a, country) && toE164(b, country)) return false;
  const [dx, dy] = [digitsOf(x), digitsOf(y)];
  return dx.length >= 8 && dy.length >= 8 && /^\+?\d+$/.test(x) && /^\+?\d+$/.test(y) && dx.slice(-9) === dy.slice(-9);
}

/**
 * A set of caller ids that says whether it holds someone who is the same person as a given one, in constant time, and
 * says it exactly as `samePerson` does (over the same country): `has(x)` is `members.some((m) => samePerson(m, x, country))`.
 * Asking that of a list is a scan; a list of thousands asked of thousands is a freeze (the numbers not to be contacted that a
 * restore adds took 40 s for 8,000). `samePerson` is not an equivalence (a number the country's rules cannot read matches by
 * its last nine digits, and two that they can read match only when they are the same number), so this keeps what it needs to
 * say the same: every member's key, and the last nine digits of the members that are numbers, by whether they can be read.
 */
export class PersonIndex {
  private readonly keys = new Set<string>();
  /** The last nine digits of members whose number the country's rules can read as E.164. */
  private readonly readableTails = new Set<string>();
  /** The last nine digits of members that are numbers those rules cannot read. */
  private readonly otherTails = new Set<string>();

  constructor(private readonly country: string | Country = localCountry()) {}

  private read(raw: string | null | undefined): { key: string; readable: boolean; tail: string | null } | null {
    if (isHidden(raw)) return null;
    const key = phoneKey(raw, this.country);
    const digits = digitsOf(key);
    const tail = /^\+?\d+$/.test(key) && digits.length >= 8 ? digits.slice(-9) : null;
    return { key, readable: toE164(raw, this.country) !== null, tail };
  }

  add(raw: string | null | undefined): void {
    const one = this.read(raw);
    if (!one) return;
    this.keys.add(one.key);
    if (one.tail) (one.readable ? this.readableTails : this.otherTails).add(one.tail);
  }

  /** Whether some member is the same person as `raw`. */
  has(raw: string | null | undefined): boolean {
    const one = this.read(raw);
    if (!one) return false;
    if (this.keys.has(one.key)) return true;
    if (!one.tail) return false;
    // Two numbers the rules can read are the same only when they are the same number (the keys, above).
    return this.otherTails.has(one.tail) || (!one.readable && this.readableTails.has(one.tail));
  }
}

/** Digits in groups: threes, the last group up to four ("301 234 5678"). */
function groups(d: string): string {
  const out: string[] = [];
  let rest = d;
  while (rest.length > 4) {
    out.push(rest.slice(0, 3));
    rest = rest.slice(3);
  }
  if (rest) out.push(rest);
  return out.join(' ');
}

/** A number as it is written at home: "0491 570 006", "02 9876 5432", "(415) 555-0132". */
function nationalFormat(c: Country, n: string): string {
  switch (c.dial) {
    case '61':
      // 1300 and 1800 numbers (and 13 ones) are dialled without the trunk.
      if (n.startsWith('1')) return n.length === 10 ? `${n.slice(0, 4)} ${n.slice(4, 7)} ${n.slice(7)}` : groups(n);
      return /^[45]/.test(n) ? `0${n.slice(0, 3)} ${n.slice(3, 6)} ${n.slice(6)}` : `0${n[0]} ${n.slice(1, 5)} ${n.slice(5)}`;
    case '64':
      return `0${n.slice(0, n.length - 7)} ${n.slice(-7, -4)} ${n.slice(-4)}`;
    case '44':
      if (n.startsWith('2') && n.length === 10) return `0${n.slice(0, 2)} ${n.slice(2, 6)} ${n.slice(6)}`;
      return `0${n.slice(0, 4)} ${n.slice(4)}`;
    case '1':
      return `(${n.slice(0, 3)}) ${n.slice(3, 6)}-${n.slice(6)}`;
  }
  return `${c.trunk}${groups(n)}`;
}

/** A number as it is written abroad: "+61 491 570 006", "+44 20 7946 0958", "+1 415 555 0132". */
function internationalFormat(cc: string, n: string): string {
  let rest: string;
  if (cc === '61') rest = /^[45]/.test(n) ? `${n.slice(0, 3)} ${n.slice(3, 6)} ${n.slice(6)}` : `${n[0]} ${n.slice(1, 5)} ${n.slice(5)}`;
  else if (cc === '64') rest = `${n.slice(0, n.length - 7)} ${n.slice(-7, -4)} ${n.slice(-4)}`;
  else if (cc === '44') rest = n.startsWith('2') && n.length === 10 ? `${n.slice(0, 2)} ${n.slice(2, 6)} ${n.slice(6)}` : `${n.slice(0, 4)} ${n.slice(4)}`;
  else if (cc === '1' && n.length === 10) rest = `${n.slice(0, 3)} ${n.slice(3, 6)} ${n.slice(6)}`;
  else rest = groups(n);
  return `+${cc} ${rest}`;
}

/**
 * A number as a person reads it: a local one as it is dialled at home
 * ("0491 570 006"), any other with its country ("+44 20 7946 0958"). What is
 * not a phone number is shown as it is.
 */
export function displayNumber(raw: string | null | undefined, country: string | Country = localCountry()): string {
  const text = String(raw ?? '').trim();
  const e164 = toE164(text, country);
  if (!e164) return text;
  const c = countryOf(country);
  const [cc, n] = splitE164(e164);
  return cc === c.dial ? nationalFormat(c, n) : internationalFormat(cc, n);
}
