// Transfers: both switches off until turned on (the phone answers exactly as before); turning transfers on turns
// message taking on with it, since a message is what nobody answering falls back to; the settings saved as chosen;
// quiet hours by day; the devices that may take a call, one marked as this computer's and one never rung; a failed
// save says so; and a desktop older than this says so plainly. Same convention as the other tests: react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RingPreview, RingSettings } from './api';

const api = vi.hoisted(() => ({ settings: vi.fn(), save: vi.fn(), preview: vi.fn(), status: vi.fn(), openSetup: vi.fn() }));
vi.mock('./useSetupState', () => ({ openSetup: (...a: unknown[]) => api.openSetup(...a) }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    ring: { ...real.ring, settings: (...a: unknown[]) => api.settings(...a), save: (...a: unknown[]) => api.save(...a), preview: (...a: unknown[]) => api.preview(...a) },
    companion: { ...real.companion, status: (...a: unknown[]) => api.status(...a) },
  };
});

import TransfersPanel from './TransfersPanel';

const OFF: RingSettings = {
  enabled: false,
  takeMessages: false,
  initiative: 'on_request',
  urgentPhrases: [],
  ringSeconds: 40,
  phoneRing: 'when_away',
  desktopRing: 'auto',
  away: 'auto',
  awayUntil: null,
  desktopActiveSeconds: 120,
  quietHours: { enabled: false, start: '21:00', end: '07:00', days: 127, allowUrgent: false, allowVip: true },
  vipNumbers: [],
  limits: { perCall: 2, gapSeconds: 60, perCallerHour: 3, globalHour: 10 },
  windowsCompanions: [],
  excludedDevices: [],
};

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
const box = (label: string) => {
  const l = [...host.querySelectorAll('label')].find((x) => x.textContent?.includes(label));
  return (l?.querySelector('input') ?? null) as HTMLInputElement;
};
const field = (label: string) => host.querySelector<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(`[aria-label="${label}"]`)!;
const click = async (el: Element) => {
  await act(async () => (el as HTMLElement).click());
  await settle();
};
const type = async (el: HTMLInputElement | HTMLTextAreaElement, value: string) => {
  await act(async () => {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
};
const saveButton = () => [...host.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Save changes'))!;
const features = (s: RingSettings) => ({ transfer: s.enabled, messages: s.enabled || s.takeMessages });

async function mount() {
  await act(async () => root.render(<TransfersPanel />));
  await settle();
}

beforeEach(() => {
  vi.clearAllMocks();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  api.settings.mockResolvedValue({ settings: OFF, features: features(OFF) });
  api.save.mockImplementation(async (change: Partial<RingSettings>) => {
    const settings = { ...OFF, ...change } as RingSettings;
    return { settings, features: features(settings) };
  });
  api.status.mockResolvedValue({ approvedMobiles: [], pendingApprovals: [], available: true, rosterRevision: 1, rosterHash: 'h', remoteAccessReady: true });
  api.preview.mockResolvedValue({ enabled: false, rings: false, devices: [], text: '', cause: null, plugin: null } satisfies RingPreview);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('Transfers', () => {
  it('starts with everything off, and says what the receptionist may do', async () => {
    await mount();
    expect(box('Transfer calls to me').checked).toBe(false);
    expect(box('Take messages').checked).toBe(false);
    expect(host.querySelector('[data-testid=what-it-may-do]')?.textContent).toContain('it does not try to reach you; it takes no messages');
    expect(saveButton().disabled).toBe(true);
  });

  it('says plainly that consent is not signed on this computer, whether transfers are on or off', async () => {
    await mount();
    const warning = () => host.querySelector('[data-testid=consent-not-signed]');
    expect(warning()?.textContent).toBe('Consent is not signed on this computer: a plugin could flip a scope. Keep this off unless you trust every plugin you have installed.');
    // It sits with the switch it is about, before it.
    const switchRow = box('Transfer calls to me').closest('label')!;
    expect(warning()!.compareDocumentPosition(switchRow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    await click(box('Transfer calls to me'));
    expect(warning()).not.toBeNull();
    // A desktop that keeps it on says it as well.
    await act(async () => root.unmount());
    root = createRoot(host);
    on();
    await mount();
    expect(warning()?.textContent).toContain('Keep this off unless you trust every plugin');
  });

  it('explains how a call is put through, in a few lines, without promising a transfer', async () => {
    await mount();
    const help = host.querySelector('[aria-label="How a call is put through"]');
    expect(help).not.toBeNull();
    expect(help!.querySelector('summary')?.textContent).toBe('How a call is put through to you');
    expect(help!.querySelectorAll('li').length).toBe(7);
    expect(help!.textContent).toContain('Nothing rings unless a Companion is set up to take the call');
    const said = help!.textContent ?? '';
    expect(said).toContain('never that they are put through');
    expect(said).toContain('You answer on the Companion');
    expect(said).toContain('decline and have the receptionist take a message');
    expect(said).not.toContain('Accept');
    expect(said).toContain('always ends with the caller being spoken to');
    expect(said).toContain('caller’s own words asked for a person');
    // What it says is what the desktop does: it starts no Companion, an urgent phrase of the owner's own also rings, and an ask counts for one try.
    expect(said).toContain('It starts no Companion');
    expect(said).toContain('When this computer is one of the devices rung');
    expect(said).toContain('one of your own urgent phrases');
    expect(said).toContain('an ask counts for one try');
    // ...and the phone that carries the calls is not said to be kept out of the list: this computer cannot tell which it is.
    const devices = text();
    expect(devices).toContain('cannot tell which of them is on the phone that carries your calls');
    expect(devices).not.toContain('is never one of them');
    // Reading it changes nothing.
    expect(saveButton().disabled).toBe(true);
  });

  it('turns message taking on with transfers, since a message is the fallback', async () => {
    await mount();
    await click(box('Transfer calls to me'));
    expect(box('Take messages').checked).toBe(true);
    expect(box('Take messages').disabled).toBe(true);
    expect(saveButton().disabled).toBe(false);
    await click(saveButton());
    expect(api.save).toHaveBeenCalledTimes(1);
    expect(api.save.mock.calls[0][0]).toMatchObject({ enabled: true, takeMessages: true });
    expect(text()).toContain('Saved');
    expect(host.querySelector('[data-testid=what-it-may-do]')?.textContent).toContain('the receptionist may try to reach you; it takes messages');
    expect(saveButton().disabled).toBe(true);
    // Turned off again, messages stay as they were chosen (on), and can be turned off on their own.
    await click(box('Transfer calls to me'));
    expect(box('Take messages').disabled).toBe(false);
    await click(box('Take messages'));
    await click(saveButton());
    expect(api.save.mock.calls[1][0]).toMatchObject({ enabled: false, takeMessages: false });
  });

  it('takes messages on their own, without transfers', async () => {
    await mount();
    await click(box('Take messages'));
    await click(saveButton());
    expect(api.save.mock.calls[0][0]).toMatchObject({ enabled: false, takeMessages: true });
  });

  it('saves how it rings, the urgent phrases and VIPs as lines, and quiet hours by day', async () => {
    await mount();
    await type(host.querySelector<HTMLInputElement>('input[type=number]')!, '55');
    await act(async () => {
      const s = [...host.querySelectorAll('select')].find((x) => x.textContent?.includes('Ring when I am away'))!;
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(s, 'always');
      s.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await type(field('Urgent phrases') as HTMLTextAreaElement, 'gas leak\n\n  burst pipe  \n');
    await type(field('VIP numbers') as HTMLTextAreaElement, '0491 570 006');
    await click(box('Do not ring me during quiet hours'));
    // Saturday and Sunday are off.
    await click(box('Sat'));
    await click(box('Sun'));
    await click(saveButton());
    const sent = api.save.mock.calls[0][0];
    expect(sent).toMatchObject({ ringSeconds: 55, phoneRing: 'always', urgentPhrases: ['gas leak', 'burst pipe'], vipNumbers: ['0491 570 006'] });
    expect(sent.quietHours).toEqual({ enabled: true, start: '21:00', end: '07:00', days: 0b0111110, allowUrgent: false, allowVip: true });
    expect(sent.limits).toEqual({ perCall: 2, gapSeconds: 60, perCallerHour: 3, globalHour: 10 });
  });

  it('lists the approved Companions: one is this computer’s, one is never rung', async () => {
    api.status.mockResolvedValue({
      approvedMobiles: [
        { deviceId: 'd1', displayName: 'Pixel 6', endpointKey: { thumbprint: 'thumb-pixel' }, approvedAt: '2026-09-01T00:00:00Z' },
        { deviceId: 'd2', displayName: 'Office PC', endpointKey: { thumbprint: 'thumb-pc' }, approvedAt: '2026-09-01T00:00:00Z' },
      ],
      pendingApprovals: [],
      available: true,
      rosterRevision: 2,
      rosterHash: 'h',
      remoteAccessReady: true,
    });
    await mount();
    expect(text()).toContain('Pixel 6');
    const pc = [...host.querySelectorAll('.transfers-devices li')].find((li) => li.textContent?.includes('Office PC'))!;
    const pixel = [...host.querySelectorAll('.transfers-devices li')].find((li) => li.textContent?.includes('Pixel 6'))!;
    await click(pc.querySelectorAll('input')[0]);
    await click(pixel.querySelectorAll('input')[1]);
    await click(saveButton());
    expect(api.save.mock.calls[0][0]).toMatchObject({ windowsCompanions: ['thumb-pc'], excludedDevices: ['thumb-pixel'] });
  });

  const PIXEL = { deviceId: 'd1', displayName: 'Pixel 6', endpointKey: { thumbprint: 'thumb-pixel' }, approvedAt: '2026-09-01T00:00:00Z' };
  const OFFICE = { deviceId: 'd2', displayName: 'Office PC', endpointKey: { thumbprint: 'thumb-pc' }, approvedAt: '2026-09-01T00:00:00Z' };
  const approve = (...approvedMobiles: unknown[]) =>
    api.status.mockResolvedValue({ approvedMobiles, pendingApprovals: [], available: true, rosterRevision: 2, rosterHash: 'h', remoteAccessReady: true });
  const on = (change: Partial<RingSettings> = {}) => {
    const settings = { ...OFF, enabled: true, takeMessages: true, ...change };
    api.settings.mockResolvedValue({ settings, features: features(settings) });
  };
  const warning = () => host.querySelector('[data-testid=nothing-would-ring]');
  const preview = (change: Partial<RingPreview> = {}): RingPreview => ({ enabled: true, rings: false, devices: [], text: '', cause: null, plugin: null, ...change });
  const says = (p: RingPreview) => api.preview.mockResolvedValue(p);

  it('warns, with a way to set one up, when transfers are on and no Companion is approved', async () => {
    on();
    says(preview({ cause: 'noCompanion', text: 'No Companion is approved yet, so nothing can ring and every caller is offered a message.' }));
    await mount();
    expect(warning()?.textContent).toContain('No Companion is approved yet, so nothing can ring and every caller is offered a message.');
    await click([...warning()!.querySelectorAll('button')].find((b) => b.textContent === 'Set up a Companion')!);
    expect(api.openSetup).toHaveBeenCalledWith({ plugin: 'aokie', step: 'pair' });
  });

  it('says what the desktop says a caller would get now, whichever way nothing would ring, and offers to set a Companion up only when there is none', async () => {
    on();
    approve(PIXEL);
    // A phone that is set never to ring: the way out is in the words, and there is no Companion to set up.
    says(preview({ cause: 'phonesOff', text: 'Your phone is set to never ring, so nothing rings and a caller who asks for you is offered a message. Set Phones to “Ring when I am away” or “Always ring”.' }));
    await mount();
    expect(warning()?.textContent).toContain('Your phone is set to never ring');
    expect(warning()?.textContent).toContain('Set Phones to');
    expect([...warning()!.querySelectorAll('button')]).toEqual([]);
    // The only Companion is ticked as this computer's and the owner is not at it: the way out is in the words.
    await act(async () => root.unmount());
    root = createRoot(host);
    says(preview({ cause: 'onlyThisComputers', text: 'The only Companion is set as the one on this computer, and it rings only while you are at the computer, which you are not now, so a caller who asks for you is offered a message. If it is a phone, untick “This is the Companion on this computer” below.' }));
    await mount();
    expect(warning()?.textContent).toContain('untick “This is the Companion on this computer”');
  });

  it('says what would ring when something would, and quiet hours as a plain line and not as a warning', async () => {
    on();
    approve(PIXEL);
    says(preview({ rings: true, devices: ['this computer', 'Pixel 6'], text: 'Right now a caller who asks for you would ring: this computer, Pixel 6.' }));
    await mount();
    expect(warning()).toBeNull();
    expect(host.querySelector('[data-testid=what-would-ring]')?.textContent).toBe('Right now a caller who asks for you would ring: this computer, Pixel 6.');
    await act(async () => root.unmount());
    root = createRoot(host);
    says(preview({ text: 'It is quiet hours now, so a caller who asks for you is offered a message.' }));
    await mount();
    expect(warning()).toBeNull();
    expect(host.querySelector('[data-testid=what-would-ring]')?.textContent).toContain('quiet hours');
  });

  it('says so when the phone plugin does not offer the calls for transfer, and why', async () => {
    on();
    approve(PIXEL);
    says(preview({ rings: true, text: 'Right now a caller who asks for you would ring: Pixel 6.', plugin: 'Your phone plugin does not support transfers yet: it did not offer the last call for transfer. Callers are offered a message.' }));
    await mount();
    expect(host.querySelector('[data-testid=plugin-state]')?.textContent).toContain('Your phone plugin does not support transfers yet');
    await act(async () => root.unmount());
    root = createRoot(host);
    says(preview({ rings: true, text: 'Right now a caller who asks for you would ring: Pixel 6.', plugin: 'Transfers are not allowed by your phone plugin’s consent settings (the Phone page). Callers are offered a message.' }));
    await mount();
    expect(host.querySelector('[data-testid=plugin-state]')?.textContent).toContain('consent settings');
  });

  it('says when the file the settings are kept in could not be used, and where it is kept, and nothing otherwise', async () => {
    await mount();
    expect(host.querySelector('[data-testid=settings-problem]')).toBeNull();
    await act(async () => root.unmount());
    root = createRoot(host);
    api.settings.mockResolvedValue({ settings: OFF, features: features(OFF), loadProblem: 'ring.json could not be used (it is not text), so everything is off. It is kept as ring.json.corrupt beside it.' });
    await mount();
    const problem = host.querySelector('[data-testid=settings-problem]')!;
    expect(problem.getAttribute('role')).toBe('alert');
    expect(problem.textContent).toContain('It is kept as ring.json.corrupt beside it.');
    // A desktop that says nothing of it (an older one) shows nothing.
    await act(async () => root.unmount());
    root = createRoot(host);
    api.settings.mockResolvedValue({ settings: OFF, features: features(OFF), loadProblem: null });
    await mount();
    expect(host.querySelector('[data-testid=settings-problem]')).toBeNull();
  });

  it('says nothing while transfers are off, or when nothing is wrong', async () => {
    approve(PIXEL);
    await mount();
    expect(warning()).toBeNull();
    expect(host.querySelector('[data-testid=plugin-state]')).toBeNull();
    expect(host.querySelector('[data-testid=what-would-ring]')).toBeNull();
    await act(async () => root.unmount());
    root = createRoot(host);
    on({ phoneRing: 'always' });
    says(preview({ rings: true, devices: ['Pixel 6'], text: 'Right now a caller who asks for you would ring: Pixel 6.' }));
    await mount();
    expect(warning()).toBeNull();
    expect(host.querySelector('[data-testid=plugin-state]')).toBeNull();
  });

  it('says a paired app can change these settings too, whether transfers are on or off', async () => {
    await mount();
    const line = () => host.querySelector('[data-testid=paired-app-can-change]');
    expect(line()?.textContent).toBe('A program on this computer that OAIY has paired can also change these settings, and turn transfers on. Pair only programs you trust.');
    await act(async () => root.unmount());
    root = createRoot(host);
    on();
    await mount();
    expect(line()).not.toBeNull();
  });

  it('looks again at what a caller would get once the settings are saved', async () => {
    approve(PIXEL);
    await mount();
    const before = api.preview.mock.calls.length;
    await click(box('Transfer calls to me'));
    await click(saveButton());
    expect(api.preview.mock.calls.length).toBeGreaterThan(before);
  });

  it('tells a phone from the Companion on this computer in its own words: only the second is ticked', async () => {
    approve(PIXEL, OFFICE);
    await mount();
    expect(text()).toContain('Tick “This is the Companion on this computer” only for a Companion that is the Windows app on this computer');
    expect(text()).toContain('A phone stays unticked, and rings by the Phones setting');
    expect(text()).toContain('A phone rings when you are away, and also when no Companion on this computer can take the call');
    // Nothing is ticked for either until the owner says.
    expect([...host.querySelectorAll<HTMLInputElement>('.transfers-devices input')].every((i) => !i.checked)).toBe(true);
  });

  it('says how to pair a Companion when there is none', async () => {
    await mount();
    expect(text()).toContain('No Companion is approved yet');
  });

  it('says so when the save fails, and keeps what was typed', async () => {
    api.save.mockRejectedValue(new Error('the settings could not be saved: disk full'));
    await mount();
    await click(box('Transfer calls to me'));
    await click(saveButton());
    expect(host.querySelector('[role=alert]')?.textContent).toContain('Not saved: the settings could not be saved: disk full');
    expect(box('Transfer calls to me').checked).toBe(true);
  });

  it('says so plainly on a desktop that cannot put callers through yet', async () => {
    api.settings.mockRejectedValue(new Error('404: not found'));
    await mount();
    expect(text()).toContain('cannot put callers through to you yet');
    expect(host.querySelector('input[type=checkbox]')).toBeNull();
  });
});
