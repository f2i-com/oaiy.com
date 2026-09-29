// Whether a caller who asks for the owner would ring anything with these settings and these Companions: what the
// desktop plans, said in plain words when the answer is no.
import { describe, expect, it } from 'vitest';
import type { CompanionApproved, RingSettings } from './api';
import { nothingWouldRing } from './transfersModel';

const S: RingSettings = {
  enabled: true,
  takeMessages: true,
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
const device = (thumbprint: string): CompanionApproved => ({ deviceId: thumbprint, displayName: thumbprint, endpointKey: { thumbprint }, approvedAt: '2026-09-01T00:00:00Z' }) as CompanionApproved;
const PHONE = device('phone');
const PC = device('pc');

describe('nothingWouldRing', () => {
  it('has nothing to say while transfers are off or the devices are not yet known', () => {
    expect(nothingWouldRing({ ...S, enabled: false }, [])).toBeNull();
    expect(nothingWouldRing(S, null)).toBeNull();
  });

  it('says so when no Companion is approved, and when every one is set never to ring', () => {
    expect(nothingWouldRing(S, [])).toContain('No Companion is approved yet');
    expect(nothingWouldRing({ ...S, excludedDevices: ['phone', 'pc'] }, [PHONE, PC])).toContain('set to never ring');
  });

  it('says so when a phone is approved and none is this computer’s: the default setup rings nobody at the computer', () => {
    expect(nothingWouldRing(S, [PHONE])).toContain('While you are at this computer nothing rings');
    // The other way round: a Companion on this computer and no phone is fine at the computer.
    expect(nothingWouldRing({ ...S, windowsCompanions: ['pc'] }, [PC])).toBeNull();
  });

  it('is content when something rings: this computer’s Companion, a phone that always rings, or an owner set to away', () => {
    expect(nothingWouldRing({ ...S, windowsCompanions: ['pc'] }, [PHONE, PC])).toBeNull();
    expect(nothingWouldRing({ ...S, phoneRing: 'always' }, [PHONE])).toBeNull();
    expect(nothingWouldRing({ ...S, away: 'on' }, [PHONE])).toBeNull();
  });

  it('says nothing can ring when neither this computer nor a phone is a thing that rings', () => {
    expect(nothingWouldRing({ ...S, phoneRing: 'never' }, [PHONE])).toContain('No Companion is selected to take a call');
    expect(nothingWouldRing({ ...S, desktopRing: 'never', windowsCompanions: ['pc'], phoneRing: 'never' }, [PC])).toContain('No Companion is selected');
    // A phone that is never rung is not a way to reach the owner, and a Companion excluded does not count as this computer's.
    expect(nothingWouldRing({ ...S, windowsCompanions: ['pc'], excludedDevices: ['pc'] }, [PHONE, PC])).toContain('While you are at this computer nothing rings');
  });

  it('does not tell an owner who set this computer never to ring that it does not', () => {
    expect(nothingWouldRing({ ...S, desktopRing: 'never' }, [PHONE])).toBeNull();
  });
});
