import { describe, expect, it } from 'vitest';
import { detectCountry, displayNumber, isHidden, phoneKey, samePerson, setLocalCountry, splitE164, toE164 } from '../../src/phoneNumbers';

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
