import type { CalendarSync, LinkStatus } from './api';

/**
 * What to say about this desktop and FormLogic, on the Overview and the
 * Calendar page: whether FormLogic can be reached, when the calendar last
 * synced, and how many changes made here are waiting to go (the calendar's,
 * and the phone's events kept for the account). Everything here works without
 * FormLogic; this only says how far behind it is.
 */

/** "just now", "3 min ago", "2 h ago", "yesterday 14:05"... */
export function ago(iso: string, now = Date.now()): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return 'at an unknown time';
  const s = Math.max(0, (now - t) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 12 * 3600) return `${Math.round(s / 3600)} h ago`;
  const d = new Date(t);
  const clock = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return `${d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} ${clock}`;
}

export interface SyncView {
  /** A word or two, for a badge or a tile. */
  headline: string;
  /** The line under it. */
  detail: string;
  tone: 'ok' | 'warn' | 'err' | 'neutral';
  /** Changes made here that FormLogic has not had yet. */
  waiting: number;
}

const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;

/** How the FormLogic side stands; null when this desktop is not linked. */
export function describeSync(sync: CalendarSync | null, link: LinkStatus | null, now = Date.now()): SyncView | null {
  const linked = sync?.linked || link?.linked;
  if (!linked) return null;
  const waiting = (sync?.pending?.total ?? 0) + (link?.outbox?.waiting ?? 0);
  const waitingText = waiting ? `${plural(waiting, 'change')} waiting` : '';
  const last = sync?.lastSuccessAt ?? null;
  const lastText = last ? `last synced ${ago(last, now)}` : 'not synced yet';
  // The calendar's own sync is the best witness; the heartbeat speaks when
  // the calendar has not said (no phone receptionist installed).
  const offline = sync?.state === 'offline' || (sync?.state !== 'synced' && !!link?.heartbeatError) || !!link?.outbox?.lastError;
  if (sync?.state === 'error') {
    return { headline: 'Sync stopped', detail: sync.error ?? 'FormLogic refused the sync', tone: 'err', waiting };
  }
  if (offline) {
    return { headline: 'Offline', detail: [lastText, waitingText || 'nothing waiting'].join(' · '), tone: 'warn', waiting };
  }
  if (sync?.state === 'syncing') {
    return { headline: 'Syncing…', detail: [lastText, waitingText].filter(Boolean).join(' · '), tone: 'neutral', waiting };
  }
  if (last) {
    return { headline: 'Synced', detail: [ago(last, now), waitingText].filter(Boolean).join(' · '), tone: waiting ? 'warn' : 'ok', waiting };
  }
  if (link?.lastHeartbeatAt) {
    return { headline: 'Online', detail: waitingText || `checked in ${ago(link.lastHeartbeatAt, now)}`, tone: waiting ? 'warn' : 'ok', waiting };
  }
  return { headline: 'Linked', detail: waitingText || 'not synced yet', tone: 'neutral', waiting };
}
