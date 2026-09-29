// Transfers: both switches off until turned on (the phone answers exactly as before); turning transfers on turns
// message taking on with it, since a message is what nobody answering falls back to; the settings saved as chosen;
// quiet hours by day; the devices that may take a call, one marked as this computer's and one never rung; a failed
// save says so; and a desktop older than this says so plainly. Same convention as the other tests: react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RingSettings } from './api';

const api = vi.hoisted(() => ({ settings: vi.fn(), save: vi.fn(), status: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return {
    ...real,
    ring: { ...real.ring, settings: (...a: unknown[]) => api.settings(...a), save: (...a: unknown[]) => api.save(...a) },
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

  it('explains how a call is put through, in a few lines, without promising a transfer', async () => {
    await mount();
    const help = host.querySelector('[aria-label="How a call is put through"]');
    expect(help).not.toBeNull();
    expect(help!.querySelector('summary')?.textContent).toBe('How a call is put through to you');
    expect(help!.querySelectorAll('li').length).toBe(6);
    const said = help!.textContent ?? '';
    expect(said).toContain('never that they are put through');
    expect(said).toContain('Accept, Decline and Take a message instead');
    expect(said).toContain('always ends with the caller being spoken to');
    expect(said).toContain('caller’s own words asked for a person');
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
