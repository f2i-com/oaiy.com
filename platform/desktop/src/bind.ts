import { useCallback, useEffect, useRef, useState } from 'react';
import { bridge, type PollContribution } from './api';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * The texts on a plugin's Overview cards (`contributions.overview[].bind`):
 * plain text, or a binding the dashboard looks up.
 *
 * - `$health.<path>`: the plugin's last health report (`PluginRecord.lastHealth`:
 *   `status`, `detail`, `components`).
 * - `$poll.<statusCardId>.<path>`: the last answer of one of the plugin's
 *   status-card polls (a declared, read-only command, sent at most every 5 s,
 *   only while the window is visible).
 *
 * A binding is a PATH, never an expression: it is split on dots and each part
 * read as an own property (or an array index). Nothing is evaluated, and a
 * part like `constructor` or `__proto__` reads nothing. What cannot be looked
 * up is `null`, which the card shows as an em dash.
 */

/** What a plugin's bindings are looked up in. */
export interface BindContext {
  /** Its last health report. */
  health?: unknown;
  /** The last answer of each of its polls, by status-card id. */
  polls?: Record<string, unknown>;
}

/** The most often a poll is sent. */
export const MIN_POLL_MS = 5000;

/** Parts of a path that are never read, own property or not. */
const NEVER = new Set(['__proto__', 'prototype', 'constructor']);

const own = (o: object, key: string) => Object.prototype.hasOwnProperty.call(o, key);

/** The value at `path` (dot-separated) in `root`; `undefined` when there is none. */
export function lookup(root: unknown, path: string): unknown {
  if (!path) return undefined;
  let at: unknown = root;
  for (const part of path.split('.')) {
    if (!part || NEVER.has(part)) return undefined;
    if (Array.isArray(at)) {
      if (!/^\d+$/.test(part)) return undefined;
      const i = Number(part);
      if (i >= at.length) return undefined;
      at = at[i];
    } else if (at !== null && typeof at === 'object' && own(at, part)) {
      at = (at as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return at;
}

/** What a bind entry says: plain text as it is, a binding looked up (`undefined` when it cannot be). */
export function resolveBinding(binding: string, ctx: BindContext): unknown {
  if (binding.startsWith('$health.')) return lookup(ctx.health, binding.slice('$health.'.length));
  if (binding.startsWith('$poll.')) {
    const rest = binding.slice('$poll.'.length);
    const dot = rest.indexOf('.');
    if (dot <= 0) return undefined;
    const id = rest.slice(0, dot);
    const polls = ctx.polls ?? {};
    if (NEVER.has(id) || !own(polls, id)) return undefined;
    return lookup(polls[id], rest.slice(dot + 1));
  }
  // Anything else starting with `$` is not a binding this dashboard knows: nothing, not the raw text.
  if (binding.startsWith('$')) return undefined;
  return binding;
}

/** A value as the card shows it: text, or `null` (an em dash). */
export function display(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : null;
  if (typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value) ?? null;
  } catch {
    return null;
  }
}

/** A bind entry (absent: `null`), looked up and shown. */
export function bindText(binding: string | undefined, ctx: BindContext): string | null {
  return binding === undefined ? null : display(resolveBinding(binding, ctx));
}

/** A poll's key among every plugin's: `<pluginId>:<statusCardId>`. */
export const pollKey = (p: Pick<PollContribution, 'pluginId' | 'id'>) => `${p.pluginId}:${p.id}`;

/** One plugin's poll answers, by status-card id, out of everyone's. */
export function pollsOf(answers: Record<string, unknown>, pluginId: string): Record<string, unknown> {
  const prefix = `${pluginId}:`;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(answers)) if (key.startsWith(prefix)) out[key.slice(prefix.length)] = value;
  return out;
}

/** Send a poll's command (no payload) and take the plugin's own answer out of the desktop's. */
export async function sendPoll(p: PollContribution): Promise<unknown> {
  const key = `poll-${p.command}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const res = await bridge.connectorRequest(p.connector, p.command, undefined, key);
  if (res.ok === false) throw new Error('The desktop could not complete this poll.');
  const r = res.result as { ok?: boolean; data?: unknown } | undefined;
  if (r?.ok === false) throw new Error('The plugin could not answer this poll.');
  return r && typeof r === 'object' && 'data' in r ? r.data : r;
}

/** Timers are not exact: a poll due within this is sent now rather than a whole tick later. */
const SLACK_MS = 250;

/**
 * The last answer of each of `polls` (by `pollKey`), each sent every
 * `max(5 s, intervalMs)` while the window is visible. A poll that fails has
 * no answer (its bindings show an em dash) until it answers again.
 */
export function usePluginPolls(polls: PollContribution[] | undefined): Record<string, unknown> {
  const [answers, setAnswers] = useState<Record<string, unknown>>({});
  const list = useRef<PollContribution[]>([]);
  list.current = polls ?? [];
  const lastSent = useRef(new Map<string, number>());
  const inflight = useRef(new Set<string>());

  const tick = useCallback(() => {
    const now = Date.now();
    for (const p of list.current) {
      const key = pollKey(p);
      const every = Math.max(MIN_POLL_MS, Number(p.intervalMs) || 0);
      if (inflight.current.has(key) || now - (lastSent.current.get(key) ?? -Infinity) < every - SLACK_MS) continue;
      lastSent.current.set(key, now);
      inflight.current.add(key);
      sendPoll(p)
        .then(
          (value) => setAnswers((prev) => ({ ...prev, [key]: value })),
          () =>
            setAnswers((prev) => {
              if (!(key in prev)) return prev;
              const next = { ...prev };
              delete next[key];
              return next;
            }),
        )
        .finally(() => inflight.current.delete(key));
    }
  }, []);

  // One visible-only tick (every 5 s) sends whichever polls are due.
  useVisiblePoll(tick, MIN_POLL_MS);

  // A poll added (or changed) is sent now, not a tick later; one that went is forgotten.
  const signature = (polls ?? []).map((p) => `${pollKey(p)}|${p.connector}|${p.command}|${p.intervalMs}`).join(',');
  useEffect(() => {
    const keys = new Set(list.current.map(pollKey));
    setAnswers((prev) => {
      const gone = Object.keys(prev).filter((k) => !keys.has(k));
      if (!gone.length) return prev;
      const next = { ...prev };
      for (const k of gone) delete next[k];
      return next;
    });
    if (!document.hidden) tick();
  }, [signature, tick]);

  return answers;
}
