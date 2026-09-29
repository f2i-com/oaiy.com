import { describe, expect, it } from 'vitest';
import { filterItems, groupItems, keyAction, moveActive } from '../../src/ui/combobox';

const items = [
  { id: 'own', label: 'The runner', detail: "Your conversation: it directs the phone's agents", group: 'Yours' },
  { id: 'call-liam', label: 'Liam', detail: '+61 491 570 006', keywords: '+61491570006 calls', group: 'Calls' },
  { id: 'sms-liam', label: 'Liam', detail: '+61 491 570 006', keywords: '+61491570006 texts', group: 'Texts' },
  { id: 'call-priya', label: 'Priya Shah', detail: '+61 400 111 222', keywords: '+61400111222', group: 'Calls' },
  { id: 'sms-unknown', label: '+61400333444', keywords: 'texts', group: 'Texts' },
  { id: 'task', label: 'Morning summary', detail: 'Flow tasks', group: 'Flow tasks' },
  { id: 'blank', label: 'Clean-up quote', group: 'Calls' },
];
const ids = (list: Array<{ id: string }>) => list.map((i) => i.id);

describe('the picker finds what is typed', () => {
  it('shows everything, in order, for an empty search', () => {
    expect(ids(filterItems(items, ''))).toEqual(ids(items));
    expect(ids(filterItems(items, '   '))).toEqual(ids(items));
  });

  it('finds by name, any case, names starting with the search first', () => {
    expect(ids(filterItems(items, 'lia'))).toEqual(['call-liam', 'sms-liam']);
    expect(ids(filterItems(items, 'SHAH'))).toEqual(['call-priya']);
    // "sum" starts a word of one name, and is inside nothing else.
    expect(ids(filterItems(items, 'sum'))).toEqual(['task']);
    // A name starting with it comes before one that only has it inside ("Clean-up quote" has "up"; "Priya Shah"… does not).
    expect(ids(filterItems(items, 'u'))[0]).toBe('blank');
  });

  it('finds by number, however it is spaced, and by the line under the name', () => {
    expect(ids(filterItems(items, '0491 570'))).toEqual([]);
    expect(ids(filterItems(items, '491 570'))).toEqual(['call-liam', 'sms-liam']);
    // A name that starts with it (a number as the name) comes first.
    expect(ids(filterItems(items, '+61400'))).toEqual(['sms-unknown', 'call-priya']);
    expect(ids(filterItems(items, '333444'))).toEqual(['sms-unknown']);
    expect(ids(filterItems(items, 'directs'))).toEqual(['own']);
  });

  it('needs every word, and finds by group and status words too', () => {
    expect(ids(filterItems(items, 'liam texts'))).toEqual(['sms-liam']);
    expect(ids(filterItems(items, 'calls liam'))).toEqual(['call-liam']);
    expect(ids(filterItems(items, 'liam zebra'))).toEqual([]);
  });

  it('says nothing matched with an empty list', () => {
    expect(filterItems(items, 'nobody')).toEqual([]);
  });
});

describe("the picker's groups", () => {
  it('lists the groups in the order given, then any others as they come', () => {
    const groups = groupItems(items, ['Yours', 'Calls', 'Texts']);
    expect(groups.map((g) => g.group)).toEqual(['Yours', 'Calls', 'Texts', 'Flow tasks']);
    expect(groups.map((g) => g.items.length)).toEqual([1, 3, 2, 1]);
  });

  it("while searching, the best match's group leads", () => {
    const found = filterItems(items, 'liam texts');
    expect(groupItems(found, ['Yours', 'Calls', 'Texts'], true).map((g) => g.group)).toEqual(['Texts']);
    const both = filterItems(items, '+61400');
    expect(groupItems(both, ['Yours', 'Calls', 'Texts'], true).map((g) => g.group)).toEqual(['Texts', 'Calls']);
  });
});

describe("the picker's keys", () => {
  it('moves with Up and Down (wrapping), Home and End, and the page keys (stopping at the ends)', () => {
    expect(moveActive(-1, 'ArrowDown', 5)).toBe(0);
    expect(moveActive(-1, 'ArrowUp', 5)).toBe(4);
    expect(moveActive(1, 'ArrowDown', 5)).toBe(2);
    expect(moveActive(4, 'ArrowDown', 5)).toBe(0);
    expect(moveActive(0, 'ArrowUp', 5)).toBe(4);
    expect(moveActive(2, 'Home', 5)).toBe(0);
    expect(moveActive(2, 'End', 5)).toBe(4);
    expect(moveActive(1, 'PageDown', 20)).toBe(9);
    expect(moveActive(15, 'PageDown', 20)).toBe(19);
    expect(moveActive(5, 'PageUp', 20)).toBe(0);
    expect(moveActive(0, 'ArrowDown', 0)).toBe(-1);
  });

  it('chooses with Enter, closes with Escape (or Alt+Up), removes with Delete at the end of the search, and types anything else', () => {
    const key = (k: string, more: Partial<KeyboardEvent> = {}) => ({ key: k, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, isComposing: false, ...more });
    expect(keyAction(key('ArrowDown'), true)).toEqual({ move: 'ArrowDown' });
    expect(keyAction(key('Home'), true)).toEqual({ move: 'Home' });
    expect(keyAction(key('Home', { shiftKey: true }), true)).toBeNull();
    expect(keyAction(key('Enter'), true)).toBe('choose');
    expect(keyAction(key('Escape'), true)).toBe('close');
    expect(keyAction(key('ArrowUp', { altKey: true }), true)).toBe('close');
    expect(keyAction(key('Delete'), true)).toBe('remove');
    // In the middle of the search, Delete deletes a letter.
    expect(keyAction(key('Delete'), false)).toBeNull();
    expect(keyAction(key('a'), true)).toBeNull();
    expect(keyAction(key('Enter', { isComposing: true }), true)).toBeNull();
  });
});
