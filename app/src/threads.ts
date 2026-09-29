/**
 * A person's conversation: their calls and their texts, one conversation
 * however the phone writes their number. Each way still has its own agent (a
 * lane: a call's agent starts fresh each call, the texts' agent reads its own
 * thread), and the person sees one timeline, kept in one file
 * (`sessions/<thread>.json`), each turn marked with its lane (`via`) and, from
 * now on, when it happened (`at`).
 *
 * Conversations kept before this (a person's calls in one, their texts in
 * another, keyed by the number as the phone wrote it) are merged once, on the
 * first start, by `regroup`; what it merges it first copies aside.
 */
import type { Turn } from './agent/protocol';
import type { CallerNote, SessionInfo } from './vfs/projects';
import { isHidden, phoneKey, samePerson, toE164 } from './phoneNumbers';
import { parseCallStart, parseCallerLine } from './ui/chat/transcript';

export type Way = 'call' | 'sms';

/** One lane's turns, in its own order, and when it last heard or said something. */
export interface LaneTurns {
  via?: Way;
  turns: readonly Turn[];
  lastAt?: number;
}

/** The most facts kept about one person (the oldest go first). */
export const MAX_FACTS = 30;
/** The key of the pretend conversation (its replies are never sent). */
const TEST = 'test';

/**
 * When each turn of a lane happened, for ordering it among another lane's.
 * Known: a turn's own time (`at`), a call's start (from the note that opens
 * it), a caller's words (their time into the call, after its start), and,
 * for a lane's last turn with none of these, when the lane last heard or said
 * something. Known times are kept in the lane's order (never earlier than the
 * one before). A turn with no time of its own happened right after the turn
 * before it that has one; turns before any known time, right before the
 * first one.
 */
export function laneTimes(lane: LaneTurns): number[] {
  const reference = new Date(lane.lastAt ?? Date.now());
  const known: Array<number | undefined> = [];
  let callStart: number | undefined;
  for (const t of lane.turns) {
    let at = typeof t.at === 'number' ? t.at : undefined;
    if (t.role === 'user') {
      const start = t.automatic ? parseCallStart(t.text, reference) : null;
      if (start) {
        callStart = start.at?.getTime();
        at ??= callStart;
      } else if (at === undefined && callStart !== undefined) {
        const said = t.text.split('\n').map(parseCallerLine).find((p) => p?.atMs !== undefined);
        if (said) at = callStart + said.atMs!;
      }
    }
    known.push(at);
  }
  if (known.length && known[known.length - 1] === undefined && lane.lastAt) known[known.length - 1] = lane.lastAt;
  let floor = -Infinity;
  for (let i = 0; i < known.length; i++) {
    if (known[i] === undefined) continue;
    floor = Math.max(floor, known[i]!);
    known[i] = floor;
  }
  const first = known.find((k) => k !== undefined) ?? -Infinity;
  let before: number | undefined;
  return known.map((k) => {
    if (k !== undefined) before = k;
    return k ?? before ?? first;
  });
}

/**
 * Lanes' turns in one order (each lane's own order kept): by when they
 * happened (see `laneTimes`), a tie going to the lane listed first. Each turn
 * is marked with its lane.
 */
export function mergeLanes(lanes: LaneTurns[]): Turn[] {
  const times = lanes.map(laneTimes);
  const next = lanes.map(() => 0);
  const out: Turn[] = [];
  for (;;) {
    let pick = -1;
    for (let l = 0; l < lanes.length; l++) {
      if (next[l] >= lanes[l].turns.length) continue;
      if (pick < 0 || times[l][next[l]] < times[pick][next[pick]]) pick = l;
    }
    if (pick < 0) return out;
    const turn = lanes[pick].turns[next[pick]++];
    if (lanes[pick].via) turn.via = lanes[pick].via;
    out.push(turn);
  }
}

/**
 * A conversation's order once its lanes have changed: the order it had (the
 * source of truth for what was there before), without the turns gone from
 * their lanes (the oldest calls let go, a conversation started again); a turn
 * put into the middle of a lane (a summary of older turns) just before the
 * turn after it there; and the lanes' new turns after, by when they happened.
 */
export function threadOrder(previous: readonly Turn[], lanes: LaneTurns[]): Turn[] {
  const where = new Map<Turn, [number, number]>();
  lanes.forEach((lane, l) => lane.turns.forEach((t, i) => where.set(t, [l, i])));
  const before = new Set(previous);
  const placed = new Set<Turn>();
  const next = lanes.map(() => 0);
  const out: Turn[] = [];
  const put = (t: Turn, l: number) => {
    if (placed.has(t)) return;
    placed.add(t);
    const via = lanes[l].via;
    if (via) t.via = via;
    out.push(t);
  };
  for (const t of previous) {
    const at = where.get(t);
    if (!at) continue;
    const [l, i] = at;
    for (let k = next[l]; k < i; k++) if (!before.has(lanes[l].turns[k])) put(lanes[l].turns[k], l);
    put(t, l);
    next[l] = Math.max(next[l], i + 1);
  }
  // The new turns at each lane's end: in each lane's order, by their times (a turn with none: the one before's).
  const tails = lanes.map((lane, l) => {
    let floor = -Infinity;
    for (let k = next[l] - 1; k >= 0; k--) {
      const at = lane.turns[k].at;
      if (typeof at === 'number') {
        floor = at;
        break;
      }
    }
    return lane.turns.slice(next[l]).filter((t) => !placed.has(t)).map((t) => {
      floor = Math.max(floor, typeof t.at === 'number' ? t.at : floor);
      return { t, at: floor };
    });
  });
  const heads = tails.map(() => 0);
  for (;;) {
    let pick = -1;
    for (let l = 0; l < tails.length; l++) {
      if (heads[l] >= tails[l].length) continue;
      if (pick < 0 || tails[l][heads[l]].at < tails[pick][heads[pick]].at) pick = l;
    }
    if (pick < 0) return out;
    put(tails[pick][heads[pick]++].t, pick);
  }
}

/** The id of a person's conversation. */
export function threadId(key: string): string {
  const part = key === TEST ? TEST : key.replace(/[^\w]+/g, '') || key.replace(/[^\w.-]+/g, '_');
  return `person-${part}`;
}

/** Whether a name is a person's (not their number, or no name at all). */
function isName(title: string, key: string): boolean {
  const t = title.trim();
  return !!t && t !== key && !toE164(t) && /\p{L}/u.test(t) && t !== 'Hidden number';
}

/**
 * What is known about people, one note a person: notes for the same person
 * (by their number in any format) as one, under their E.164 number. The name
 * is the newest one given; the facts are all of them (each once, the oldest
 * first, the most recent kept when there are too many).
 */
export function mergeCallers(notes: readonly CallerNote[], country: string): { callers: CallerNote[]; changed: boolean } {
  const out: CallerNote[] = [];
  let changed = false;
  for (const note of [...notes].sort((a, b) => a.updatedAt - b.updatedAt)) {
    const key = note.number === TEST ? TEST : phoneKey(note.number, country) || note.number;
    const same = out.find((c) => c.number === key || (key !== TEST && c.number !== TEST && samePerson(c.number, key, country)));
    if (!same) {
      if (key !== note.number) changed = true;
      out.push({ ...note, number: key, facts: [...note.facts] });
      continue;
    }
    changed = true;
    if (note.name) same.name = note.name;
    for (const fact of note.facts) if (!same.facts.some((f) => f.toLowerCase() === fact.toLowerCase())) same.facts.push(fact);
    same.facts = same.facts.slice(-MAX_FACTS);
    same.updatedAt = Math.max(same.updatedAt, note.updatedAt);
  }
  return { callers: out, changed };
}

/** What the conversations are kept as: the list, each file's turns, and what is known about callers. */
export interface Stored {
  infos: SessionInfo[];
  /** Turns by file (a conversation's, or a lane's kept before conversations were a person's). */
  chats: Map<string, Turn[]>;
  callers: CallerNote[];
}

export interface Regrouped extends Stored {
  /** Whether anything differs from what was kept (then it is copied aside, and written anew). */
  changed: boolean;
  /** The conversations whose files are to be written. */
  rewritten: string[];
  /** Files whose turns are now in a conversation's file. */
  stale: string[];
}

/** A lane's person, for grouping: its number's key, or its own (a hidden caller, the pretend conversation). */
function personOf(info: SessionInfo, country: string): { key: string; alone: boolean } {
  if (info.key === TEST) return { key: TEST, alone: true };
  const hidden = info.hidden || (info.kind === 'call' && info.title === 'Hidden number') || isHidden(info.key);
  if (hidden) return { key: info.key, alone: true };
  return { key: phoneKey(info.key, country) || info.key, alone: false };
}

/**
 * The conversations with each person as one: their call and text lanes (in
 * whatever format the phone wrote their number) under one conversation keyed
 * by their E.164 number, named by them (or the number), their turns merged in
 * the order they happened (see `mergeLanes`), and what is known about them as
 * one note. Running it again on what it returned changes nothing.
 */
export function regroup(stored: Stored, country: string): Regrouped {
  const { callers, changed: callersChanged } = mergeCallers(stored.callers, country);
  let changed = callersChanged;
  const infos: SessionInfo[] = [];
  const chats = new Map<string, Turn[]>();
  const rewritten: string[] = [];
  const stale = new Set<string>();
  const turnsOf = (file: string) => stored.chats.get(file) ?? [];

  // A flow's tasks: a conversation of their own, as they were.
  for (const info of stored.infos.filter((i) => i.kind === 'task')) {
    const thread = info.thread ?? info.id;
    infos.push({ ...info, thread });
    chats.set(thread, turnsOf(thread));
  }

  type Group = { key: string; alone: boolean; lanes: SessionInfo[]; threads: Set<string> };
  const groups: Group[] = [];
  for (const info of stored.infos.filter((i) => i.kind !== 'task')) {
    const who = personOf(info, country);
    const group = groups.find((g) => (info.thread && g.threads.has(info.thread)) || (!who.alone && !g.alone && (g.key === who.key || samePerson(g.key, who.key, country))));
    if (group) {
      group.lanes.push(info);
      if (info.thread) group.threads.add(info.thread);
    } else groups.push({ ...who, lanes: [info], threads: new Set(info.thread ? [info.thread] : []) });
  }

  for (const group of groups) {
    const kept = group.threads.size === 1 && group.lanes.every((l) => l.thread) ? [...group.threads][0] : null;
    const sameKind = (kind: string) => group.lanes.filter((l) => l.kind === kind);
    const oneEach = sameKind('call').length <= 1 && sameKind('sms').length <= 1;
    const thread = kept ?? [...group.threads][0] ?? threadId(group.alone ? group.lanes[0].key : group.key);
    const note = group.alone ? undefined : callers.find((c) => samePerson(c.number, group.key, country));
    const newest = [...group.lanes].sort((a, b) => b.lastAt - a.lastAt);
    const named = newest.find((l) => isName(l.title, l.key))?.title;
    const title = note?.name || named || (group.key === TEST ? 'Test' : group.alone ? group.lanes[0].title || group.key : group.key);
    let turns: Turn[];
    if (kept && oneEach) {
      turns = turnsOf(kept);
      // A conversation of one lane whose turns were never marked: they are that lane's.
      if (group.lanes.length === 1) for (const t of turns) t.via ??= group.lanes[0].kind as Way;
    } else {
      // Each lane's turns: from its own file (kept before conversations were a person's), or its part of a
      // conversation's (all of it, unmarked turns too, when it is that conversation's only lane).
      const lanes: LaneTurns[] = group.lanes
        .map((l) => {
          if (!l.thread) return { via: l.kind as Way, turns: turnsOf(l.id), lastAt: l.lastAt };
          const only = group.lanes.filter((x) => x.thread === l.thread).length === 1;
          return { via: l.kind as Way, turns: turnsOf(l.thread).filter((t) => t.via === l.kind || (only && !t.via)), lastAt: l.lastAt };
        })
        .sort((a, b) => (a.via === b.via ? 0 : a.via === 'call' ? -1 : 1));
      turns = mergeLanes(lanes);
      rewritten.push(thread);
      changed = true;
      for (const l of group.lanes) {
        const file = l.thread ?? l.id;
        if (file !== thread) stale.add(file);
      }
    }
    chats.set(thread, turns);
    for (const kind of ['call', 'sms'] as const) {
      const lanes = sameKind(kind).sort((a, b) => b.lastAt - a.lastAt);
      if (!lanes.length) continue;
      const handles = [...new Set(lanes.flatMap((l) => l.handles ?? []))].slice(-100);
      const lane: SessionInfo = {
        id: lanes[0].id,
        kind,
        key: group.key,
        title,
        lastAt: Math.max(...lanes.map((l) => l.lastAt)),
        unread: lanes.reduce((n, l) => n + (l.unread || 0), 0),
        ...(handles.length ? { handles } : {}),
        thread,
        ...(lanes.some((l) => l.hidden) || (group.alone && group.key !== TEST && kind === 'call') ? { hidden: true } : {}),
      };
      // (A name alone is not a change to copy aside for: it is kept with the list's next save.)
      const was = lanes[0];
      if (lanes.length > 1 || was.key !== lane.key || was.thread !== thread || !!was.hidden !== !!lane.hidden) changed = true;
      infos.push(lane);
    }
  }
  infos.sort((a, b) => b.lastAt - a.lastAt);
  return { infos, chats, callers, changed, rewritten, stale: [...stale] };
}
