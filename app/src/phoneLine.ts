/**
 * The phone's one line, as everything that dials sees it: missed calls rung
 * back (callbacks.ts) and outreach (outreach.ts). A call the app placed and
 * has not heard end, a call someone else made or took (a call coming in, one
 * waiting, a dial of another's), and the calm minute after any call ends all
 * keep the line busy. Aokie itself refuses a dial while a call is in progress;
 * this keeps the app from asking while one is ringing, or starting before it
 * is known.
 */
import type { DesktopEvent } from './desktop/bridge';

/** After any call on the line ends: a minute of quiet before the next dial (a caller ringing back gets through). */
export const CALM_MS = 60_000;
/** A dial of ours never heard to end is taken as over after this long (as a call back's is). */
const PLACED_LOST_MS = 10 * 60_000;
/** A call of another's never heard to end: let go after this long. */
const FOREIGN_LOST_MS = 2 * 60 * 60_000;

export type LineUser = 'callback' | 'outreach';

export class PhoneLine {
  /** The dial the app placed and has not heard end. */
  placed: { by: LineUser; callId?: string; operationId?: string; at: number } | null = null;
  /** Calls on the line that are not ours, by their id, with when they were seen. */
  private foreign = new Map<string, number>();
  /** No dial before this (ms): the calm after a call. */
  calmAt = 0;

  /** An event from the desktop: calls on the line come and go. */
  event(e: DesktopEvent, now = Date.now()): void {
    const d = e.data;
    const id = (v: unknown) => (typeof v === 'string' && v ? v : '');
    switch (e.name) {
      case 'aokie.call.incoming': {
        const call = id(d.callId) || e.correlationId;
        if (call) this.foreign.set(call, now);
        break;
      }
      case 'aokie.call.waiting': {
        const call = id(d.waitingCallId);
        if (call) this.foreign.set(call, now);
        break;
      }
      case 'aokie.call.outbound.dialing': {
        const call = id(d.callId) || e.correlationId;
        if (!call) break;
        if (this.placed && (this.placed.callId === call || !this.placed.callId)) this.placed.callId = call;
        else this.foreign.set(call, now);
        break;
      }
      case 'aokie.call.ended': {
        const call = id(d.callId) || e.correlationId;
        this.foreign.delete(call);
        if (this.placed && this.placed.callId === call) this.placed = null;
        this.calmAt = Math.max(this.calmAt, now + CALM_MS);
        break;
      }
      case 'aokie.hardware.error':
        // A dial the radio could not place (a call came in first, or Bluetooth failed): the line is not ours.
        if (d.code === 'control_failed' && d.action === 'call.dial' && this.placed?.operationId && d.operationId === this.placed.operationId) {
          this.placed = null;
          this.calmAt = Math.max(this.calmAt, now + CALM_MS);
        }
        break;
    }
  }

  /** Whether a dial may go now: nothing of ours or anyone's on the line, no call going on here, and the calm after the last one over. */
  idle(now: number, liveCall: boolean): boolean {
    if (this.placed && now - this.placed.at > PLACED_LOST_MS) this.placed = null;
    for (const [call, at] of this.foreign) if (now - at > FOREIGN_LOST_MS) this.foreign.delete(call);
    return !this.placed && !this.foreign.size && !liveCall && now >= this.calmAt;
  }

  /** The app is about to dial. */
  take(by: LineUser, now = Date.now()): void {
    this.placed = { by, at: now };
  }

  /** What the phone said of the dial (its call and operation ids). */
  bind(callId: string, operationId?: string): void {
    if (!this.placed) return;
    if (callId) this.placed.callId = callId;
    if (operationId) this.placed.operationId = operationId;
  }

  /** The dial did not happen (refused): the line is free again. */
  release(by: LineUser): void {
    if (this.placed?.by === by && !this.placed.callId) this.placed = null;
  }
}
