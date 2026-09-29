// The Contacts page's words and numbers: a number as a person reads it, the
// search, the badge's letters, a new number's problems, the export's name.
import { describe, expect, it } from 'vitest';
import type { Contact } from './api';
import { contactLabel, csvFileName, initials, matchesContact, numberKey, numberProblem, readableNumber, saidWhen, skipReason } from './contactsModel';

const c = (over: Partial<Contact>): Contact => ({ key: '491570006', number: '', name: '', nameBy: null, notes: '', facts: [], createdAt: '', updatedAt: '', ...over });

describe('a number, readable', () => {
  it('writes an Australian number the local way, and any other as it was kept', () => {
    expect(readableNumber('+61491570006')).toBe('0491 570 006');
    expect(readableNumber('0491570006')).toBe('0491 570 006');
    expect(readableNumber('+61 2 9876 5432')).toBe('(02) 9876 5432');
    expect(readableNumber('+14155550100')).toBe('+14155550100');
    expect(readableNumber('1300 123 456')).toBe('1300 123 456');
  });

  it('reads a key alone (a contact from before numbers were kept) as Australian', () => {
    expect(readableNumber('', '491570006')).toBe('0491 570 006');
    expect(readableNumber('', '298765432')).toBe('(02) 9876 5432');
    expect(readableNumber('', '98765432')).toBe('98765432');
  });
});

describe('the search', () => {
  const lance = c({ name: 'Lance', number: '+61491570006', notes: 'Owns the café', facts: [{ text: 'Has a dog', at: '', by: 'agent' }] });
  it('finds a name, notes or a fact in any case, and a number typed any way', () => {
    for (const q of ['', 'lance', 'CAFÉ', 'dog', '0491 570', '491570', '+61 491', '(04) 91']) expect(matchesContact(lance, q)).toBe(true);
    for (const q of ['sam', '0499', 'cat']) expect(matchesContact(lance, q)).toBe(false);
    // A contact with only its key is found by its number too.
    expect(matchesContact(c({ name: 'Sam', key: '400000001' }), '0400 000')).toBe(true);
  });
});

describe('words', () => {
  it('names a contact, and gives their badge one or two letters', () => {
    expect(contactLabel(c({ name: 'Lance' }))).toBe('Lance');
    expect(contactLabel(c({ number: '+61491570006' }))).toBe('0491 570 006');
    expect(initials('Lance')).toBe('L');
    expect(initials('lance smith')).toBe('LS');
    expect(initials('Zoë Anne O’Brien')).toBe('ZO');
    expect(initials('')).toBe('');
  });

  it('says what is wrong with a number for a new contact', () => {
    expect(numberProblem('')).toBe('Give their phone number.');
    expect(numberProblem('Private')).toBe('A hidden number cannot be a contact.');
    expect(numberProblem('123 45')).toMatch(/at least 8 digits/);
    expect(numberProblem('0491 570 006')).toBeNull();
    expect(numberKey('+61 491 570 006')).toBe('491570006');
    expect(numberKey('1234567')).toBeNull();
  });

  it('names the export for today, says when, and counts the skipped', () => {
    expect(csvFileName(new Date(2026, 8, 29, 23, 30))).toBe('oaiy-contacts-2026-09-29.csv');
    const now = new Date(2026, 8, 29, 12);
    expect(saidWhen(new Date(2026, 8, 29, 8).toISOString(), now)).toBe('today');
    expect(saidWhen(new Date(2026, 8, 28, 8).toISOString(), now)).toBe('yesterday');
    expect(saidWhen(new Date(2026, 8, 12, 8).toISOString(), now)).toBe('12 Sep');
    expect(saidWhen(new Date(2025, 8, 12, 8).toISOString(), now)).toBe('12 Sep 2025');
    expect(skipReason('no_number', 2)).toBe('2 with no number');
    expect(skipReason('not_a_phone_number', 1)).toBe('1 that is not a phone number');
    expect(skipReason('hidden', 3)).toBe('3 with hidden numbers');
  });
});
