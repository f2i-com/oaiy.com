// What the Overview and the Calendar page say about FormLogic. The rule being
// pinned: never "Synced" while it cannot be reached, never a bare "sync
// failed", and always how much is waiting to go.
import { describe, expect, it } from 'vitest';
import type { CalendarSync, LinkStatus } from './api';
import { ago, describeSync } from './syncStatus';

const NOW = Date.parse('2026-09-29T10:00:00Z');
const minutesAgo = (m: number) => new Date(NOW - m * 60_000).toISOString();

const sync = (s: Partial<CalendarSync>): CalendarSync => ({ linked: true, at: null, pulled: 0, pushed: 0, error: null, ...s });
const link = (l: Partial<LinkStatus>): LinkStatus => ({ linked: true, attempt: { phase: 'idle' }, available: [], ...l });

describe('describeSync', () => {
  it('is silent while unlinked', () => {
    expect(describeSync(sync({ linked: false, state: 'unlinked' }), link({ linked: false }), NOW)).toBeNull();
    expect(describeSync(null, null, NOW)).toBeNull();
  });

  it('says offline, when it last synced, and what is waiting', () => {
    const v = describeSync(
      sync({ state: 'offline', lastSuccessAt: minutesAgo(30), pending: { creates: 1, updates: 0, deletes: 1, total: 2 }, error: 'formlogic.com can’t be reached: it refused the connection' }),
      link({ outbox: { waiting: 1 } }),
      NOW,
    )!;
    expect(v.headline).toBe('Offline');
    expect(v.detail).toBe('last synced 30 min ago · 3 changes waiting');
    expect(v.tone).toBe('warn');
    expect(v.waiting).toBe(3);
  });

  it('says offline before the first sync too, without inventing a time', () => {
    const v = describeSync(sync({ state: 'offline' }), null, NOW)!;
    expect(v.detail).toBe('not synced yet · nothing waiting');
  });

  it('takes the heartbeat’s word when the calendar has none', () => {
    expect(describeSync(null, link({ heartbeatError: 'can’t be reached' }), NOW)!.headline).toBe('Offline');
    expect(describeSync(null, link({ lastHeartbeatAt: minutesAgo(1) }), NOW)!.headline).toBe('Online');
    // A calendar that synced a moment ago outranks a heartbeat that has not caught up.
    expect(describeSync(sync({ state: 'synced', lastSuccessAt: minutesAgo(0) }), link({ heartbeatError: 'x' }), NOW)!.headline).toBe('Synced');
  });

  it('is offline while phone events are waiting on FormLogic', () => {
    const v = describeSync(sync({ state: 'synced', lastSuccessAt: minutesAgo(2) }), link({ outbox: { waiting: 4, lastError: 'HTTP 503' } }), NOW)!;
    expect(v.headline).toBe('Offline');
    expect(v.detail).toContain('4 changes waiting');
  });

  it('is not offline when FormLogic only asks for fewer requests', () => {
    const v = describeSync(sync({ state: 'busy', lastSuccessAt: minutesAgo(2), pending: { creates: 1, updates: 0, deletes: 0, total: 1 } }), link({ heartbeatError: 'HTTP 429: Too Many Requests' }), NOW)!;
    expect(v.headline).toBe('Syncing shortly');
    expect(v.detail).toBe('FormLogic asked for fewer requests · last synced 2 min ago · 1 change waiting');
    const heartbeatOnly = describeSync(sync({ state: 'synced', lastSuccessAt: minutesAgo(1) }), link({ outbox: { waiting: 0, lastError: 'HTTP 429' } }), NOW)!;
    expect(heartbeatOnly.headline).toBe('Synced');
  });

  it('says why when FormLogic refuses rather than being away', () => {
    const v = describeSync(sync({ state: 'error', error: 'FormLogic no longer accepts this desktop’s key (HTTP 401): link it again' }), null, NOW)!;
    expect(v.headline).toBe('Sync stopped');
    expect(v.detail).toContain('link it again');
    expect(v.tone).toBe('err');
  });

  it('says synced, and how long ago, when all is well', () => {
    const v = describeSync(sync({ state: 'synced', lastSuccessAt: minutesAgo(3), pending: { creates: 0, updates: 0, deletes: 0, total: 0 } }), link({}), NOW)!;
    expect(v).toEqual({ headline: 'Synced', detail: '3 min ago', tone: 'ok', waiting: 0 });
  });
});

describe('ago', () => {
  it('reads like a person says it', () => {
    expect(ago(minutesAgo(0), NOW)).toBe('just now');
    expect(ago(minutesAgo(5), NOW)).toBe('5 min ago');
    expect(ago(minutesAgo(180), NOW)).toBe('3 h ago');
    expect(ago('nonsense', NOW)).toBe('at an unknown time');
  });
});
