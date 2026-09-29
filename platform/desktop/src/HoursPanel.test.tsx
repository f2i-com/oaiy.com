// Hours & Services: the quick actions on the opening hours, a service's
// checks (a name, a length) said where they are and in the save bar, the
// unsaved-changes bar, removing a service with undo, reordering, and what is
// sent. The setup wizard's "Your business" step is this same form, embedded.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CalendarSettings } from './api';

const { saveSettings } = vi.hoisted(() => ({ saveSettings: vi.fn() }));
vi.mock('./api', () => ({ calendar: { saveSettings: (...a: unknown[]) => saveSettings(...a) }, voices: {} }));

import { SettingsForm } from './HoursPanel';
import { ToastProvider } from './Toasts';

const SETTINGS = (): CalendarSettings => ({
  business: 'Green Lawns',
  receptionist: '',
  hours: [
    [{ open: '08:00', close: '17:00' }],
    [{ open: '09:00', close: '15:00' }],
    [],
    [{ open: '08:00', close: '17:00' }],
    [{ open: '08:00', close: '17:00' }],
    [{ open: '09:00', close: '12:00' }],
    [],
  ],
  services: [
    { id: 'mow', name: 'Lawn mowing', minutes: 60, price: 'from $60' },
    { id: 'hedge', name: 'Hedge trimming', minutes: 90 },
    { id: 'quote', name: 'Quote visit', minutes: 30, description: 'We look at the job' },
  ],
  slotMinutes: 30,
  noticeMinutes: 60,
  horizonDays: 30,
  textConfirmations: true,
});

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';
const bar = () => host.querySelector('.hours-savebar');
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label || b.getAttribute('aria-label') === label)!;
const day = (name: string) => [...host.querySelectorAll('.hours-day')].find((d) => d.querySelector('.hours-day-name span')?.textContent === name)!;
const times = (name: string) => [...day(name).querySelectorAll<HTMLInputElement>('input[type="time"]')].map((i) => i.value);
const serviceNames = () => [...host.querySelectorAll<HTMLInputElement>('.svc-name')].map((i) => i.value);
function type(input: HTMLInputElement, value: string) {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  set.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

async function mount(props: Partial<React.ComponentProps<typeof SettingsForm>> = {}) {
  const onSaved = vi.fn();
  await act(async () =>
    root.render(
      <ToastProvider>
        <SettingsForm settings={SETTINGS()} onSaved={onSaved} {...props} />
      </ToastProvider>,
    ),
  );
  return onSaved;
}

beforeEach(() => {
  vi.clearAllMocks();
  saveSettings.mockImplementation(async (s: CalendarSettings) => s);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('the opening hours', () => {
  it('say the week as callers hear it, a closed day as “Closed”', async () => {
    await mount();
    expect(host.querySelector('.hours-summary')?.textContent).toBe('Mon 8 am – 5 pm · Tue 9 am – 3 pm · Wed closed · Thu, Fri 8 am – 5 pm · Sat 9 am – 12 pm · Sun closed');
    expect(day('Wednesday').textContent).toContain('Closed');
    expect(day('Wednesday').querySelector<HTMLInputElement>('input[role="switch"]')!.checked).toBe(false);
  });

  it('“Same hours Mon–Fri” opens every weekday with Monday’s hours', async () => {
    await mount();
    await act(async () => button('Same hours Mon–Fri').click());
    for (const d of ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) expect(times(d)).toEqual(['08:00', '17:00']);
    expect(times('Saturday')).toEqual(['09:00', '12:00']);
    expect(times('Sunday')).toEqual([]);
    expect(bar()?.textContent).toContain('Unsaved changes');
  });

  it('“Copy to all open days” copies a day to every open day, and leaves closed days closed', async () => {
    await mount();
    await act(async () => button('Copy Tuesday\'s hours to all open days').click());
    for (const d of ['Monday', 'Thursday', 'Friday', 'Saturday']) expect(times(d)).toEqual(['09:00', '15:00']);
    expect(times('Wednesday')).toEqual([]);
  });

  it('“Copy Monday to all open days” is there as a quick action too', async () => {
    await mount();
    await act(async () => button('Copy Monday to all open days').click());
    for (const d of ['Tuesday', 'Thursday', 'Friday', 'Saturday']) expect(times(d)).toEqual(['08:00', '17:00']);
    expect(times('Wednesday')).toEqual([]);
    expect(times('Sunday')).toEqual([]);
  });

  it('a day switched on takes the hours before it; one that closes before it opens says so and is not saved', async () => {
    await mount();
    await act(async () => day('Wednesday').querySelector<HTMLInputElement>('input[role="switch"]')!.click());
    expect(times('Wednesday')).toEqual(['09:00', '15:00']);
    const [open] = day('Monday').querySelectorAll<HTMLInputElement>('input[type="time"]');
    await act(async () => type(open, '18:00'));
    expect(day('Monday').textContent).toContain('Closes at 5 pm, before it opens at 6 pm');
    expect(open.getAttribute('aria-invalid')).toBe('true');
    await act(async () => button('Save changes').click());
    expect(saveSettings).not.toHaveBeenCalled();
    expect(bar()?.textContent).toContain('One thing to fix first: Monday: closes at 5 pm, before it opens at 6 pm');
  });
});

describe('the services', () => {
  it('need a name and a length: said at the service and in the save bar, and nothing is sent', async () => {
    await mount();
    await act(async () => button('Add a service').click());
    expect(serviceNames()).toEqual(['Lawn mowing', 'Hedge trimming', 'Quote visit', '']);
    // Not shouted at before it is touched.
    expect(host.querySelector('.svc-row:last-child .hours-error')).toBeNull();
    await act(async () => button('Save changes').click());
    expect(saveSettings).not.toHaveBeenCalled();
    expect(host.querySelector('.svc-row:last-child')?.textContent).toContain('Give it a name');
    expect(bar()?.classList.contains('is-invalid')).toBe(true);
    expect(bar()?.textContent).toContain('Service 4: give it a name');

    // A custom length of nothing: said too.
    const row = host.querySelector('.svc-row:last-child')!;
    const select = row.querySelector<HTMLSelectElement>('select')!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'custom');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    const minutes = row.querySelector<HTMLInputElement>('.svc-minutes input')!;
    await act(async () => type(minutes, ''));
    expect(row.textContent).toContain('Say how long it takes');

    await act(async () => type(row.querySelector<HTMLInputElement>('.svc-name')!, 'Gutter clean'));
    await act(async () => type(minutes, '45'));
    await act(async () => button('Save changes').click());
    expect(saveSettings).toHaveBeenCalledTimes(1);
    expect(saveSettings.mock.calls[0][0].services[3]).toMatchObject({ name: 'Gutter clean', minutes: 45 });
  });

  it('are removed with an undo, and moved up and down', async () => {
    await mount();
    await act(async () => button('Remove Hedge trimming').click());
    expect(serviceNames()).toEqual(['Lawn mowing', 'Quote visit']);
    expect(text()).toContain('Removed “Hedge trimming”.');
    await act(async () => button('Undo').click());
    expect(serviceNames()).toEqual(['Lawn mowing', 'Hedge trimming', 'Quote visit']);
    expect(bar()).toBeNull();

    await act(async () => button('Move Quote visit up').click());
    expect(serviceNames()).toEqual(['Lawn mowing', 'Quote visit', 'Hedge trimming']);
    expect(button('Move Lawn mowing up').disabled).toBe(true);
    await act(async () => button('Save changes').click());
    // Their ids go with them.
    expect(saveSettings.mock.calls[0][0].services.map((s: { id: string }) => s.id)).toEqual(['mow', 'quote', 'hedge']);
  });
});

describe('saving', () => {
  it('shows the unsaved-changes bar only while something has changed; Discard puts it back', async () => {
    await mount();
    expect(bar()).toBeNull();
    await act(async () => type(host.querySelector<HTMLInputElement>('.hours-business-name')!, 'Green Lawns & Gardens'));
    expect(bar()?.textContent).toContain('Unsaved changes');
    expect(text()).toContain('Callers hear: “Thanks for calling Green Lawns & Gardens, this is Aokie. How can I help?”');
    await act(async () => button('Discard').click());
    expect(bar()).toBeNull();
    expect(host.querySelector<HTMLInputElement>('.hours-business-name')!.value).toBe('Green Lawns');
  });

  it('sends the booking rules as written in their sentences, and says it is saved', async () => {
    const onSaved = await mount();
    const notice = host.querySelector<HTMLInputElement>('#rule-notice')!;
    await act(async () => type(notice, '120'));
    await act(async () => host.querySelector<HTMLInputElement>('.rules-switch input')!.click());
    await act(async () => button('Save changes').click());
    expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({ noticeMinutes: 120, textConfirmations: false, slotMinutes: 30, horizonDays: 30 }));
    expect(onSaved).toHaveBeenCalled();
    expect(bar()?.textContent).toContain('Saved');
  });

  it('in the setup wizard, has its Save there from the start', async () => {
    const onSaved = await mount({ embedded: true });
    expect(host.querySelector('.hours-form.is-embedded')).not.toBeNull();
    expect(bar()?.textContent).toContain('These are saved');
    await act(async () => button('Save').click());
    expect(saveSettings).toHaveBeenCalledTimes(1);
    expect(onSaved).toHaveBeenCalled();
  });
});

describe('the business card', () => {
  const receptionist = () => host.querySelector<HTMLInputElement>('.hours-receptionist-name')!;
  const greeting = () => host.querySelector('.hours-greeting')?.textContent;

  it('has the receptionist’s name beside the business’s: Aokie until it is given one', async () => {
    await mount();
    const card = host.querySelector('.hours-business')!;
    expect([...card.querySelectorAll('.form-row > span')].map((s) => s.textContent)).toEqual(['Its name', 'The receptionist’s name']);
    expect(receptionist().value).toBe('');
    expect(receptionist().placeholder).toBe('Aokie');
    expect(card.textContent).toContain('What the receptionist calls itself on calls and texts.');
    expect(document.getElementById(receptionist().getAttribute('aria-describedby')!)?.textContent).toContain('calls itself');
    expect(greeting()).toBe('Callers hear: “Thanks for calling Green Lawns, this is Aokie. How can I help?”');
  });

  it('says the name as it is typed, and saves it with the page’s Save', async () => {
    const onSaved = await mount();
    await act(async () => type(receptionist(), 'Sam'));
    expect(greeting()).toBe('Callers hear: “Thanks for calling Green Lawns, this is Sam. How can I help?”');
    expect(bar()?.textContent).toContain('Unsaved changes');
    await act(async () => button('Save changes').click());
    expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({ business: 'Green Lawns', receptionist: 'Sam' }));
    expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ receptionist: 'Sam' }));
    // With no business named yet, the greeting still says who answers.
    await act(async () => type(host.querySelector<HTMLInputElement>('.hours-business-name')!, ''));
    expect(greeting()).toBe('Callers hear: “Thanks for calling, this is Sam. How can I help?”');
    // Emptied, it is Aokie again.
    await act(async () => type(receptionist(), '  '));
    expect(greeting()).toBe('Callers hear: “Thanks for calling, this is Aokie. How can I help?”');
  });

  it('a name longer than 40 characters is said where it is and in the save bar, and is not sent', async () => {
    await mount();
    await act(async () => type(receptionist(), 'A'.repeat(41)));
    expect(receptionist().getAttribute('aria-invalid')).toBe('true');
    expect(host.querySelector('.hours-business .hours-error')?.textContent).toBe('At most 40 characters');
    expect(bar()?.textContent).toContain('The receptionist’s name: at most 40 characters');
    await act(async () => button('Save changes').click());
    expect(saveSettings).not.toHaveBeenCalled();
    // Forty is fine (spaces around it are not counted).
    await act(async () => type(receptionist(), ` ${'A'.repeat(40)} `));
    expect(receptionist().getAttribute('aria-invalid')).toBe('false');
    expect(host.querySelector('.hours-business .hours-error')).toBeNull();
    await act(async () => button('Save changes').click());
    expect(saveSettings).toHaveBeenCalledTimes(1);
  });

  it('shows a name set elsewhere (by the Agent), and the setup wizard’s step has the field too', async () => {
    await mount({ settings: { ...SETTINGS(), receptionist: 'Sam' } });
    expect(receptionist().value).toBe('Sam');
    expect(greeting()).toContain('this is Sam.');
    act(() => root.unmount());
    root = createRoot(host);

    await mount({ embedded: true });
    expect(host.querySelector('.hours-form.is-embedded .hours-receptionist-name')).not.toBeNull();
    await act(async () => type(receptionist(), 'Sam'));
    await act(async () => button('Save changes').click());
    expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({ receptionist: 'Sam' }));
  });
});
