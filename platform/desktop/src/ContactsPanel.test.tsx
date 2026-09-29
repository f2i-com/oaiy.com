// Contacts: the list (by name, numbers readable) and its search; one person's
// name, notes for the receptionist and what it remembered, saved from the
// unsaved-changes bar; forgetting a fact; deleting after asking; adding one
// by hand; a CSV file's preview, then its import; the export; and a desktop
// older than contacts (a 404). Same convention as the other tests:
// react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Contact, ImportReport } from './api';

const api = vi.hoisted(() => ({
  list: vi.fn(),
  save: vi.fn(),
  remove: vi.fn(),
  forgetFact: vi.fn(),
  importCsv: vi.fn(),
  exportCsv: vi.fn(),
}));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    contacts: {
      list: (...a: unknown[]) => api.list(...a),
      save: (...a: unknown[]) => api.save(...a),
      remove: (...a: unknown[]) => api.remove(...a),
      forgetFact: (...a: unknown[]) => api.forgetFact(...a),
      importCsv: (...a: unknown[]) => api.importCsv(...a),
      exportCsv: (...a: unknown[]) => api.exportCsv(...a),
    },
  };
});

import ContactsPanel from './ContactsPanel';
import { ToastProvider } from './Toasts';

const AT = '2026-09-20T02:00:00Z';
const contact = (c: Partial<Contact> & Pick<Contact, 'key'>): Contact => ({
  number: '',
  name: '',
  nameBy: null,
  notes: '',
  facts: [],
  createdAt: AT,
  updatedAt: AT,
  ...c,
});
const LIAM = () =>
  contact({
    key: '491570006',
    number: '+61491570006',
    name: 'Liam',
    nameBy: 'owner',
    notes: 'Prefers texts',
    facts: [
      { text: 'Has a dog called Max', at: AT, by: 'agent' },
      { text: 'Books the long mow', at: '2026-09-21T02:00:00Z', by: 'agent' },
    ],
  });
const SAM = () => contact({ key: '400000001', name: 'Sam', nameBy: 'agent' });
const NEW_CALLER = () => contact({ key: '411222333', number: '+61411222333' });

let host: HTMLDivElement;
let root: Root;
let people: Contact[];
const text = () => host.textContent ?? '';
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
const button = (label: string) =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label || b.getAttribute('aria-label') === label)!;
const click = async (el: Element) => {
  await act(async () => (el as HTMLElement).click());
  await settle();
};
const rowNames = () => [...host.querySelectorAll('.contact-row strong')].map((s) => s.textContent);
const row = (name: string) => [...host.querySelectorAll<HTMLButtonElement>('.contact-row')].find((r) => r.querySelector('strong')?.textContent === name)!;
const heading = () => host.querySelector('#contacts-side-title')?.textContent;
const bar = () => host.querySelector('.contacts-savebar');
function type(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

async function mount(props: { open?: string | null; onOpened?: () => void } = {}) {
  await act(async () =>
    root.render(
      <ToastProvider>
        <ContactsPanel {...props} />
      </ToastProvider>,
    ),
  );
  await settle();
}

beforeEach(() => {
  vi.clearAllMocks();
  people = [LIAM(), SAM(), NEW_CALLER()];
  api.list.mockImplementation(async () => ({ contacts: people, total: people.length }));
  api.save.mockImplementation(async (key: string, change: Partial<Contact>) => {
    const was = people.find((c) => c.key === key.replace(/\D/g, '').slice(-9)) ?? contact({ key: key.replace(/\D/g, '').slice(-9) });
    const next = { ...was, ...change, nameBy: change.name !== undefined ? (change.name ? 'owner' : null) : was.nameBy } as Contact;
    people = [...people.filter((c) => c.key !== next.key), next];
    return next;
  });
  api.forgetFact.mockImplementation(async (key: string, index: number) => {
    const was = people.find((c) => c.key === key)!;
    const next = { ...was, facts: was.facts.filter((_, i) => i !== index) };
    people = people.map((c) => (c.key === key ? next : c));
    return { contact: next, forgotten: was.facts[index] };
  });
  api.remove.mockImplementation(async (key: string) => {
    people = people.filter((c) => c.key !== key);
  });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('the list', () => {
  it('shows everyone by name with their number readable, the nameless after', async () => {
    await mount();
    expect(rowNames()).toEqual(['Liam', 'Sam', 'No name yet']);
    expect([...host.querySelectorAll('.contact-row small')].map((s) => s.textContent)).toEqual(['0491 570 006', '0400 000 001', '0411 222 333']);
    expect(row('Liam').textContent).toContain('Notes');
    expect(row('Liam').textContent).toContain('2 remembered');
    expect(host.querySelector('.contacts-count')?.textContent).toBe('3 contacts');
  });

  it('is searched by name, number written any way, notes, or what was remembered', async () => {
    await mount();
    const search = host.querySelector<HTMLInputElement>('input[type="search"]')!;
    const find = async (q: string) => {
      await act(async () => type(search, q));
      return rowNames();
    };
    expect(await find('sam')).toEqual(['Sam']);
    expect(await find('0491 570')).toEqual(['Liam']);
    expect(await find('+61 411')).toEqual(['No name yet']);
    expect(await find('texts')).toEqual(['Liam']);
    expect(await find('dog')).toEqual(['Liam']);
    expect(host.querySelector('.contacts-count')?.textContent).toBe('1 of 3');
    expect(await find('zzz')).toEqual([]);
    expect(text()).toContain('No contact matches “zzz”.');
  });

  it('says how people get here when there is no one yet', async () => {
    people = [];
    await mount();
    expect(text()).toContain('Contacts appear as people ring and text, or add one.');
    expect(button('Export CSV').disabled).toBe(true);
  });

  it('says so on a desktop older than contacts (a 404), and offers nothing to do', async () => {
    api.list.mockRejectedValue(new Error('404: Not Found'));
    await mount();
    expect(text()).toContain('This OAIY does not keep contacts yet.');
    expect(host.querySelector('.contacts-toolbar')).toBeNull();
    expect(host.querySelector('.banner-err')).toBeNull();
  });
});

describe('one person', () => {
  it('is edited: a single name, the notes for the receptionist, saved from the unsaved-changes bar', async () => {
    await mount();
    await click(row('Sam'));
    expect(heading()).toBe('Sam');
    expect(text()).toContain('The receptionist learned this name on a call.');
    expect(host.querySelector('.contact-number')?.textContent).toContain('0400 000 001');
    expect(host.querySelector('.contact-notes > span')?.textContent).toBe('Notes for the receptionist');
    expect(text()).toContain('The receptionist reads these on every call and text with them.');
    expect(bar()).toBeNull();

    await act(async () => type(host.querySelector<HTMLInputElement>('.contact-name input')!, 'Samuel'));
    await act(async () => type(host.querySelector<HTMLTextAreaElement>('.contact-notes textarea')!, 'Comes on Tuesdays.\nPays cash.'));
    expect(bar()?.textContent).toContain('Unsaved changes');
    expect(text()).toContain('Saved as your name for them: the receptionist never changes it.');
    await click(button('Save changes'));
    expect(api.save).toHaveBeenCalledWith('400000001', { name: 'Samuel', notes: 'Comes on Tuesdays.\nPays cash.' });
    expect(bar()?.textContent).toContain('Saved');
    expect(rowNames()).toContain('Samuel');
    expect(text()).toContain('You named them: the receptionist never changes it.');
  });

  it('keeps the receptionist’s name as the person’s own, and sends only what changed', async () => {
    await mount();
    await click(row('Sam'));
    await click(button('Keep this name'));
    expect(bar()?.textContent).toContain('Unsaved changes');
    await click(button('Save changes'));
    expect(api.save).toHaveBeenCalledWith('400000001', { name: 'Sam' });

    await click(row('Liam'));
    await act(async () => type(host.querySelector<HTMLTextAreaElement>('.contact-notes textarea')!, 'Prefers texts. Not before 9.'));
    await click(button('Save changes'));
    expect(api.save).toHaveBeenLastCalledWith('491570006', { notes: 'Prefers texts. Not before 9.' });
  });

  it('forgets what the receptionist remembered when saved, the last first, and Discard puts it back', async () => {
    await mount();
    await click(row('Liam'));
    expect(host.querySelector('#contact-facts-title')?.textContent).toContain('What the receptionist remembered');
    const facts = () => [...host.querySelectorAll('.fact-list > li')].map((li) => [li.querySelector('.fact-text > span')?.textContent, li.classList.contains('is-removed')]);
    expect(facts()).toEqual([
      ['Has a dog called Max', false],
      ['Books the long mow', false],
    ]);
    await click(button('Forget: Has a dog called Max'));
    expect(facts()[0]).toEqual(['Has a dog called Max', true]);
    expect(text()).toContain('forgotten when you save');
    await click(button('Discard'));
    expect(facts()[0]).toEqual(['Has a dog called Max', false]);
    expect(bar()).toBeNull();

    await click(button('Forget: Has a dog called Max'));
    await click(button('Forget: Books the long mow'));
    await click(button('Save changes'));
    expect(api.forgetFact.mock.calls).toEqual([
      ['491570006', 1, 'Books the long mow'],
      ['491570006', 0, 'Has a dog called Max'],
    ]);
    expect(api.save).not.toHaveBeenCalled();
    expect(text()).toContain('Nothing yet.');
  });

  it('is not left with changes unsaved: another person waits until they are saved or discarded', async () => {
    await mount();
    await click(row('Liam'));
    await act(async () => type(host.querySelector<HTMLTextAreaElement>('.contact-notes textarea')!, 'Changed'));
    await click(row('Sam'));
    expect(heading()).toBe('Liam');
    expect(bar()?.textContent).toContain('Save or discard the changes to Liam first.');
    await click(button('Discard'));
    await click(row('Sam'));
    expect(heading()).toBe('Sam');
  });

  it('is deleted after asking', async () => {
    await mount();
    await click(row('Liam'));
    await click(button('Delete contact'));
    expect(text()).toContain('Delete Liam? Their name, your notes and what the receptionist remembered all go.');
    await click(button('Cancel'));
    expect(api.remove).not.toHaveBeenCalled();
    await click(button('Delete contact'));
    await click(button('Delete'));
    expect(api.remove).toHaveBeenCalledWith('491570006');
    expect(rowNames()).toEqual(['Sam', 'No name yet']);
    expect(host.querySelector('.contacts-side')).toBeNull();
    expect(text()).toContain('Liam is no longer a contact.');
  });

  it('opens by its key when the Agent asks, or offers to add them', async () => {
    const onOpened = vi.fn();
    await mount({ open: '400000001', onOpened });
    expect(heading()).toBe('Sam');
    expect(onOpened).toHaveBeenCalled();
    await act(async () => root.render(<ToastProvider><ContactsPanel open="498765432" onOpened={onOpened} /></ToastProvider>));
    await settle();
    expect(heading()).toBe('Add a contact');
    expect(host.querySelector<HTMLInputElement>('input[inputmode="tel"]')?.value).toBe('0498 765 432');
  });
});

describe('adding someone', () => {
  it('takes one name and a number, and opens them', async () => {
    await mount();
    await click(button('Add contact'));
    expect(heading()).toBe('Add a contact');
    const [name, number] = [host.querySelector<HTMLInputElement>('.contact-form .contact-name input')!, host.querySelector<HTMLInputElement>('input[inputmode="tel"]')!];
    await act(async () => type(number, '123'));
    await click(host.querySelector('.contact-form button[type="submit"]')!);
    expect(text()).toContain('That is not a phone number: it needs at least 8 digits.');
    expect(api.save).not.toHaveBeenCalled();

    await act(async () => type(number, '0491 570 006'));
    expect(text()).toContain('Already a contact: Liam.');
    expect(host.querySelector<HTMLButtonElement>('.contact-form button[type="submit"]')!.disabled).toBe(true);

    await act(async () => type(name, '  Kim '));
    await act(async () => type(number, '0411 999 888'));
    await click(host.querySelector('.contact-form button[type="submit"]')!);
    expect(api.save).toHaveBeenCalledWith('0411 999 888', { number: '0411 999 888', name: 'Kim' });
    expect(heading()).toBe('Kim');
    expect(rowNames()).toContain('Kim');
  });
});

describe('a CSV file', () => {
  const report = (over: Partial<ImportReport> = {}): ImportReport => ({
    preview: true,
    country: 'AU',
    headerRow: 1,
    columns: ['Name', 'Phone 1 - Value', 'Notes'],
    rows: 5,
    added: 2,
    updated: 1,
    unchanged: 1,
    skipped: 1,
    reasons: { no_number: 1 },
    skips: [{ row: 6, action: 'skip', name: 'Nobody', number: '', reason: 'no_number', why: 'no number', notes: false }],
    sample: [
      { row: 2, action: 'update', name: 'Liam Smith', number: '0491 570 006', key: '491570006', keptName: 'Liam', notes: true },
      { row: 3, action: 'add', name: 'Kim', number: '0411 999 888', key: '411999888', notes: false },
      { row: 6, action: 'skip', name: 'Nobody', number: '', reason: 'no_number', why: 'no number', notes: false },
    ],
    ...over,
  });

  it('is previewed (nothing written), then imported in one go', async () => {
    api.importCsv.mockImplementation(async (body: { preview: boolean; replaceNames: boolean }) => report({ preview: body.preview, updated: body.replaceNames ? 2 : 1 }));
    await mount();
    await click(button('Import CSV'));
    expect(heading()).toBe('Contacts from a CSV file');
    const csv = 'Name,Phone 1 - Value,Notes\nLiam Smith,0491 570 006,Has a dog\n';
    const input = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', { value: [new File([csv], 'contacts.csv', { type: 'text/csv' })] });
    await act(async () => input.dispatchEvent(new Event('change', { bubbles: true })));
    await settle();
    expect(api.importCsv).toHaveBeenCalledWith({ csv, country: 'AU', replaceNames: false, preview: true });
    const stats = [...host.querySelectorAll('.import-stat')].map((s) => s.textContent);
    expect(stats).toEqual(['2new', '1to update', '1already here', '1skipped']);
    expect(text()).toContain('1 with no number');
    expect(text()).toContain('keeps “Liam”');
    expect(text()).toContain('Read from Name, Phone 1 - Value, Notes, the header on line 1.');

    // Replace names I've set: previewed again.
    await click(host.querySelector('.contacts-switch input')!);
    expect(api.importCsv).toHaveBeenLastCalledWith({ csv, country: 'AU', replaceNames: true, preview: true });
    // Another country: previewed again.
    const country = host.querySelector<HTMLSelectElement>('.contact-import select')!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(country, 'NZ');
      country.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    expect(api.importCsv).toHaveBeenLastCalledWith({ csv, country: 'NZ', replaceNames: true, preview: true });

    const listed = api.list.mock.calls.length;
    await click(button('Import 4 contacts'));
    expect(api.importCsv).toHaveBeenLastCalledWith({ csv, country: 'NZ', replaceNames: true, preview: false });
    expect(text()).toContain('2 added, 2 updated, 1 already here, 1 skipped.');
    expect(api.list.mock.calls.length).toBe(listed + 1);
    expect(host.querySelector('.contacts-side')).toBeNull();
  });

  it('says why a file cannot be read', async () => {
    api.importCsv.mockRejectedValue(new Error('400: no column of phone numbers was found: name one number, phone or mobile in the header row'));
    await mount();
    await click(button('Import CSV'));
    const input = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', { value: [new File(['Name,Email\nLiam,l@x.au\n'], 'people.csv')] });
    await act(async () => input.dispatchEvent(new Event('change', { bubbles: true })));
    await settle();
    expect(host.querySelector('.contact-import .banner-err')?.textContent).toContain('no column of phone numbers was found');
    expect(host.querySelector('.import-stats')).toBeNull();
  });

  it('is exported with every contact, named for today', async () => {
    const blob = new Blob(['﻿name,number,notes,remembered\r\n'], { type: 'text/csv' });
    api.exportCsv.mockResolvedValue(blob);
    const made: Blob[] = [];
    // jsdom has no object URLs: these stand in for the webview's.
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: (b: Blob) => (made.push(b), 'blob:contacts') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: vi.fn() });
    let saved: { href: string; download: string } | null = null;
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved = { href: this.getAttribute('href') ?? '', download: this.download };
    });
    await mount();
    await click(button('Export CSV'));
    expect(api.exportCsv).toHaveBeenCalled();
    expect(made).toEqual([blob]);
    expect(saved!.href).toBe('blob:contacts');
    expect(saved!.download).toMatch(/^oaiy-contacts-\d{4}-\d{2}-\d{2}\.csv$/);
    expect(text()).toContain(`3 contacts, saved as ${saved!.download}.`);
    clickSpy.mockRestore();
  });
});
