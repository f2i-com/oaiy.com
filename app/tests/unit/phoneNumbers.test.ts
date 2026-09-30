import { describe, expect, it } from 'vitest';
import { PersonIndex, detectCountry, displayNumber, isHidden, phoneKey, samePerson, setLocalCountry, splitE164, toE164 } from '../../src/phoneNumbers';

/** A small deterministic random generator (mulberry32), so that a failure can be found again. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('a set of people found in constant time says what samePerson says', () => {
  // Numbers written every way, in and out of the country's rules, that are the same person as one another or not: the same
  // digits with and without the country code and the trunk, numbers of several countries, one that no country's rules can
  // read (a foreign one written as it is dialled there) that shares its last nine digits with one that can be read, short codes,
  // sender ids and hidden callers.
  const pool = [
    '0491570006', '0491 570 006', '+61491570006', '+61 491 570 006', '61491570006', '0011 61 491 570 006', '491570006', '+61 (0) 491 570 006',
    '0400111222', '+61400111222', '400111222', '+64400111222', '0400 111 222', '+44 7700 900123', '07700 900123', '7700900123', '+447700900123',
    '+1 415 555 0132', '(415) 555-0132', '415-555-0132', '4155550132', '+64 21 123 4567', '021 123 4567', '211234567', '+27 82 123 4567', '0821234567', '821234567',
    '02 9876 5432', '+61298765432', '298765432', '+33 1 42 68 53 00', '0033 1 42 68 53 00', '142685300', '33142685300', '1300 123 456', '1300123456', '13 11 11', '131111',
    'ACME', 'acme', 'Acme Bank', 'Private', 'Unknown', 'Withheld', '', '  ', '+', 'abc123', '123', '12345678', '123456789', '0123456789', '+1234567890123',
    // Too short to have a tail (with and without the plus), that differ in the ninth digit from the end, and sender ids that have digits in them.
    '+1234567', '1234567', '+123456', '123456', '+12345678', '112345678', '212345678', '+112345678', 'ACME 12345678', 'ZED 12345678', 'ACME12345678', 'ACME 1234 5678',
  ];
  it('for every sequence of numbers a list could be built from, in five countries', () => {
    let compared = 0;
    let matched = 0;
    for (const country of ['AU', 'NZ', 'GB', 'US', 'ZA']) {
      for (let run = 0; run < 60; run++) {
        const random = seeded(run * 7919 + country.charCodeAt(0));
        const members: string[] = [];
        const index = new PersonIndex(country);
        for (let step = 0; step < 90; step++) {
          const next = pool[Math.floor(random() * pool.length)];
          const want = members.some((m) => samePerson(m, next, country));
          expect(index.has(next), `${country}: ${JSON.stringify(next)} against ${JSON.stringify(members)}`).toBe(want);
          compared++;
          if (want) matched++;
          if (random() < 0.55) {
            members.push(next);
            index.add(next);
          }
        }
      }
    }
    expect(compared).toBe(5 * 60 * 90);
    // The comparisons include plenty of people who are the same and plenty who are not (a test of two that agree on nothing proves nothing).
    expect(matched).toBeGreaterThan(2000);
    expect(compared - matched).toBeGreaterThan(2000);
  });

  it('is not a partition: a number that cannot be read matches by its last nine digits, and two that can be read only when equal', () => {
    const index = new PersonIndex('AU');
    index.add('+61491570006');
    expect(index.has('491570006')).toBe(true);
    expect(index.has('+14915700069')).toBe(false);
    expect(index.has('+64491570006')).toBe(false);
    index.add('491570006');
    expect(index.has('+64491570006')).toBe(true);
    expect(index.has('+61491570006')).toBe(true);
  });

  it('gives no last-nine-digits match to what is too short, to what differs before the last nine, or to a sender id that has digits in it', () => {
    // Each pair: a member, and someone who is not the same person as it (samePerson says so, and so does the index).
    for (const [member, probe] of [
      ['+1234567', '1234567'], // seven digits: too short to match by the end of the number
      ['+123456', '123456'],
      ['112345678', '212345678'], // nine digits that differ in the first
      ['+112345678', '212345678'],
      ['ACME 12345678', 'ZED 12345678'], // letters and digits are a name, not a number
      ['ACME12345678', 'ZED 12345678'],
    ]) {
      expect(samePerson(member, probe, 'AU'), `${member} / ${probe}`).toBe(false);
      const index = new PersonIndex('AU');
      index.add(member);
      expect(index.has(probe), `${member} / ${probe}`).toBe(false);
      // and the other way round
      const other = new PersonIndex('AU');
      other.add(probe);
      expect(other.has(member), `${probe} / ${member}`).toBe(false);
    }
    // A name is the same as itself, however it is written; a number of eight digits matches by the digits it has.
    const names = new PersonIndex('AU');
    names.add('ACME 12345678');
    expect(names.has('acme 12345678')).toBe(samePerson('ACME 12345678', 'acme 12345678', 'AU'));
    const eight = new PersonIndex('AU');
    eight.add('+12345678');
    expect(eight.has('12345678')).toBe(true);
    expect(samePerson('+12345678', '12345678', 'AU')).toBe(true);
  });
});

describe('a caller id as a person: E.164, read with the country for local numbers', () => {
  it('reads an Australian number in every way a phone writes it as one', () => {
    for (const written of ['0491570006', '0491 570 006', '(04) 9157-0006', '04-9157-0006', '0491.570.006', '+61491570006', '+61 491 570 006', '+61 (0) 491 570 006', '+610491570006', '61491570006', '0011 61 491 570 006', '0061491570006', 'tel:+61-491-570-006']) {
      expect(toE164(written, 'AU'), written).toBe('+61491570006');
    }
    // Landlines, with and without the area code's brackets.
    expect(toE164('(02) 9876 5432', 'AU')).toBe('+61298765432');
    expect(toE164('03 9123 4567', 'AU')).toBe('+61391234567');
    expect(toE164('+61 2 9876 5432', 'AU')).toBe('+61298765432');
  });

  it('reads numbers from other countries: with their country code, after 00 or 0011, or as local ones of the country chosen', () => {
    expect(toE164('+44 20 7946 0958', 'AU')).toBe('+442079460958');
    expect(toE164('0011 44 20 7946 0958', 'AU')).toBe('+442079460958');
    expect(toE164('00 44 20 7946 0958', 'AU')).toBe('+442079460958');
    expect(toE164('+1 (415) 555-0132', 'AU')).toBe('+14155550132');
    expect(toE164('0011 1 415 555 0132', 'AU')).toBe('+14155550132');
    expect(toE164('+64 21 123 4567', 'AU')).toBe('+64211234567');
    // Local to the country chosen.
    expect(toE164('020 7946 0958', 'GB')).toBe('+442079460958');
    expect(toE164('07700 900123', 'GB')).toBe('+447700900123');
    expect(toE164('00 61 491 570 006', 'GB')).toBe('+61491570006');
    expect(toE164('(415) 555-0132', 'US')).toBe('+14155550132');
    expect(toE164('1-415-555-0132', 'US')).toBe('+14155550132');
    expect(toE164('011 61 491 570 006', 'US')).toBe('+61491570006');
    expect(toE164('021 123 4567', 'NZ')).toBe('+64211234567');
    // Italy keeps the 0 of its area code after +39.
    expect(toE164('+39 06 1234 5678', 'AU')).toBe('+390612345678');
    expect(toE164('06 1234 5678', 'IT')).toBe('+390612345678');
  });

  it('never makes a wrong number up: what the country chosen cannot read is not a phone number', () => {
    // North America's area codes never start with 0: an Australian mobile is not read as American.
    expect(toE164('0491570006', 'US')).toBeNull();
    // A UK mobile written at home is not an Australian number.
    expect(toE164('07700 900123', 'AU')).toBeNull();
    // 13, 1300 and 1800 numbers, and short codes: left as they are.
    expect(toE164('13 22 11', 'AU')).toBeNull();
    expect(toE164('1300 123 456', 'AU')).toBeNull();
    expect(toE164('1234', 'AU')).toBeNull();
    expect(toE164('+123', 'AU')).toBeNull();
  });

  it('a hidden or withheld caller and a sender id of letters are not numbers, and each stays its own', () => {
    for (const hidden of ['', '   ', 'Private', 'PRIVATE NUMBER', 'Unknown', 'Withheld', 'Anonymous', 'Restricted', 'No Caller ID', '-']) {
      expect(isHidden(hidden), hidden).toBe(true);
      expect(toE164(hidden, 'AU')).toBeNull();
      expect(phoneKey(hidden, 'AU')).toBe('');
    }
    expect(isHidden('Telstra')).toBe(false);
    expect(toE164('Telstra', 'AU')).toBeNull();
    expect(phoneKey('Telstra', 'AU')).toBe('Telstra');
    expect(phoneKey('13 22 11', 'AU')).toBe('132211');
    // Never merged: a hidden caller with anyone (not even another), a name or a short code with a number.
    expect(samePerson('', '', 'AU')).toBe(false);
    expect(samePerson('Private', 'Private', 'AU')).toBe(false);
    expect(samePerson('Telstra', '+61491570006', 'AU')).toBe(false);
    expect(samePerson('132211', '+61132211', 'AU')).toBe(false);
    expect(samePerson('MyGov', 'MyGovAU', 'AU')).toBe(false);
    // The same sender, however it is spaced, is itself.
    expect(samePerson('13 22 11', '132211', 'AU')).toBe(true);
  });

  it('the same person however the phone writes them; different people however alike their digits', () => {
    expect(samePerson('0491570006', '+61491570006', 'AU')).toBe(true);
    expect(samePerson('+61 491 570 006', '0011 61 491 570 006', 'AU')).toBe(true);
    expect(samePerson('0491570006', '0491570157', 'AU')).toBe(false);
    // The same last nine digits in two countries: two people (the old last-nine-digits rule merged them).
    expect(samePerson('+61491570006', '+44491570006', 'AU')).toBe(false);
    // A foreign number the country's rules cannot read matches by its last nine digits, as the phone matches numbers.
    expect(samePerson('07700 900123', '+447700900123', 'AU')).toBe(true);
    expect(phoneKey('0491 570 006', 'AU')).toBe('+61491570006');
  });

  it('shows a local number as it is dialled at home, and any other with its country', () => {
    expect(displayNumber('+61491570006', 'AU')).toBe('0491 570 006');
    expect(displayNumber('0491570006', 'AU')).toBe('0491 570 006');
    expect(displayNumber('+61298765432', 'AU')).toBe('02 9876 5432');
    expect(displayNumber('+611300123456', 'AU')).toBe('1300 123 456');
    expect(displayNumber('+442079460958', 'AU')).toBe('+44 20 7946 0958');
    expect(displayNumber('+447700900123', 'AU')).toBe('+44 7700 900123');
    expect(displayNumber('+14155550132', 'AU')).toBe('+1 415 555 0132');
    expect(displayNumber('+64211234567', 'AU')).toBe('+64 21 123 4567');
    expect(displayNumber('+4930123456789', 'AU')).toBe('+49 301 234 567 89');
    expect(displayNumber('+61491570006', 'GB')).toBe('+61 491 570 006');
    expect(displayNumber('+442079460958', 'GB')).toBe('020 7946 0958');
    expect(displayNumber('+14155550132', 'US')).toBe('(415) 555-0132');
    // Not a number: as it is.
    expect(displayNumber('Telstra', 'AU')).toBe('Telstra');
    expect(displayNumber('test', 'AU')).toBe('test');
    expect(splitE164('+61491570006')).toEqual(['61', '491570006']);
    expect(splitE164('+14155550132')).toEqual(['1', '4155550132']);
    expect(splitE164('+353851234567')).toEqual(['353', '851234567']);
  });

  it("the country for local numbers is the computer's: its time zone, then its language's region, else Australia", () => {
    expect(detectCountry(['en-AU'], 'UTC')).toBe('AU');
    // Windows in Australia often speaks US English: the clock says where it is.
    expect(detectCountry(['en-US'], 'Australia/Sydney')).toBe('AU');
    expect(detectCountry(['en-GB'], 'Australia/Perth')).toBe('AU');
    expect(detectCountry(['en-US'], 'America/Chicago')).toBe('US');
    expect(detectCountry(['en-GB'], 'Europe/London')).toBe('GB');
    expect(detectCountry(['en-NZ', 'en'], 'Etc/UTC')).toBe('NZ');
    // Nothing to go by: Australia, where this business is.
    expect(detectCountry(['en'], 'UTC')).toBe('AU');
    expect(detectCountry([], '')).toBe('AU');
    // The setting: a country chosen, or '' for the computer's own.
    setLocalCountry('GB');
    expect(phoneKey('020 7946 0958')).toBe('+442079460958');
    setLocalCountry('AU');
    expect(phoneKey('0491 570 006')).toBe('+61491570006');
  });
});
