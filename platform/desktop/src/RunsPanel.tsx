import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, CircleAlert, Loader2, RefreshCw, RotateCcw, Search, Square, Trash2, Workflow, X } from 'lucide-react';
import { bridge, network, type RunHistory, type RunPage, type RunRecord, type RunStatus } from './api';
import DeadLetters from './DeadLetters';
import LoadError, { loadErrorText } from './LoadError';
import { ask } from './ask';
import { CopyButton, Fields, JsonBlock, RunAnswer, durationText, errorText, eventWords, fullWhen, whenText, whoAsked } from './RunParts';
import { lastFailedOf, writeFailedSeen } from './failedSeen';
import { useToast } from './Toasts';
import { invalidate, peek, put } from './useCached';
import { useVisiblePoll } from './useVisiblePoll';

export { errorText };

/**
 * Run history — what happened, and why it failed.
 *
 * The ledger has always kept every run (20k of them, durable across restarts),
 * but until now the only way in was `GET /api/bridge/runs/:id`. That made a
 * failure diagnosable only by whoever happened to be holding its id: if a
 * trigger fired a flow at 3am and it failed, nothing on this machine would ever
 * tell you. This panel is the read surface over what was already recorded.
 *
 * Failures are the default filter for the same reason — a list dominated by
 * successes buries the one row anyone opened this screen to find. A row opens to
 * what the run was asked with and what it answered, and a finished run can be
 * run again, one in progress stopped.
 *
 * The newest runs are asked for again every few seconds; older ones are paged
 * back through, past the 200 one answer holds (`before`, the last run shown),
 * and a search or a flow picked is the desktop's to answer, over every run it
 * keeps, not over the runs this page happens to have.
 */

const POLL_MS = 5000;
/** While a run in the list is queued or running: asked this often, so it is seen to end. */
const LIVE_POLL_MS = 2000;

/** Failed: the runs that failed or ran out of time, the ones Overview's warning counts. A cancelled run is under All:
 *  someone stopped it, nothing went wrong. */
const FAILURE_STATES: RunStatus[] = ['failed', 'timed_out'];
const GOING_STATES: RunStatus[] = ['queued', 'running'];
const ENDED = new Set<RunStatus>(['succeeded', 'failed', 'timed_out', 'cancelled']);

type Filter = 'failures' | 'going' | 'all';
const FILTERS: { id: Filter; label: string; states: RunStatus[] | 'all'; heading: string; none: string; one: string }[] = [
  { id: 'failures', label: 'Failed', states: FAILURE_STATES, heading: 'Failed runs', none: 'failed runs', one: 'failed run' },
  { id: 'going', label: 'In progress', states: GOING_STATES, heading: 'Runs in progress', none: 'runs in progress', one: 'run in progress' },
  { id: 'all', label: 'All', states: 'all', heading: 'All runs', none: 'runs', one: 'run' },
];
const filterOf = (id: Filter) => FILTERS.find((f) => f.id === id)!;

/** Rows a page, and the most one request lists (the desktop's own cap). */
const PAGE = 50;
const MOST = 200;
/** How long typing rests before the desktop is asked to search. */
const SEARCH_REST_MS = 300;

/** A run's place in history, newest first (as the desktop orders it: when asked, then its id). */
const newerFirst = (a: RunRecord, b: RunRecord) =>
  Date.parse(b.reservedAt) - Date.parse(a.reservedAt) || b.runId.length - a.runId.length || (b.runId > a.runId ? 1 : b.runId < a.runId ? -1 : 0);

/** Status → the badge class the design system already defines. */
function badgeFor(status: RunStatus): string {
  switch (status) {
    case 'succeeded':
      return 'badge badge-ok';
    case 'failed':
    case 'timed_out':
      return 'badge badge-err';
    case 'running':
    case 'queued':
      return 'badge badge-pending';
    default:
      return 'badge badge-neutral';
  }
}

const LABEL: Record<RunStatus, string> = {
  queued: 'queued',
  running: 'running',
  succeeded: 'succeeded',
  failed: 'failed',
  timed_out: 'timed out',
  cancelled: 'cancelled',
};

const ms = (iso: string | undefined) => (iso ? Date.parse(iso) : NaN);

/** Whether a run ran held in, its inputs being from outside this computer's person (a plugin's event, an app, a site;
 *  as the desktop's `outside_input` decides): for what Run again says. The desktop decides a run again's itself, from
 *  the run's record. */
const ranConfined = (r: RunRecord) => !!r.confined || (r.callerProduct !== 'oaiy-agent' && r.callerProduct !== 'oaiy-desktop');

/** How long it ran, from the two timestamps the ledger already stores (a running run: so far). */
function took(rec: RunRecord, now: number): string {
  const start = ms(rec.startedAt);
  if (!Number.isFinite(start)) return '';
  const end = rec.status === 'running' ? now : ms(rec.finishedAt);
  return Number.isFinite(end) ? durationText(Math.max(0, end - start)) : '';
}

/**
 * What an empty failures list actually proves — which is less than it looks.
 *
 * This used to interpolate `total`, but that is the whole ledger (`l.len()`),
 * queued and running rows included, so it asserted that runs which had not
 * started had "finished cleanly". Nothing ages a queued run out — the desktop
 * worker deliberately leaves `queued`-mode runs for an external claimer — so
 * with no claimer running, that false all-clear is permanent, not a blink, and
 * this sentence is the only thing on the failures screen that speaks to overall
 * run health. byStatus is on the same response and breaks the ledger down per
 * status, so `succeeded` is the entry that is actually about finished work —
 * the map itself tallies every state, live runs included.
 */
/** Every recorded run finished cleanly, in words that fit one run as well as many. */
const allClean = (n: number) =>
  n === 1 ? 'Nothing has failed — the one recorded run finished cleanly.' : `Nothing has failed — all ${n} recorded runs finished cleanly.`;

function noFailuresMessage(data: RunHistory): string {
  if (data.total === 0) return 'No flow has run on this computer yet.';
  const counts = data.byStatus;
  // A response (or a cached snapshot) without byStatus leaves total as the only
  // number there is; saying 0 would be a worse answer than the old overcount.
  if (!counts) return allClean(data.total);
  const clean = counts.succeeded ?? 0;
  const inFlight = (counts.queued ?? 0) + (counts.running ?? 0);
  const stopped = counts.cancelled ?? 0;
  const tail = stopped ? `, ${stopped} cancelled` : '';
  if (inFlight === 0 && !stopped) return allClean(clean);
  return `Nothing has failed — ${clean} finished cleanly${inFlight ? `, ${inFlight} still queued or running` : ''}${tail}.`;
}

/** How many runs a filter holds in all, from the per-status counts every answer carries. */
function countOf(data: RunHistory | null, filter: Filter): number | null {
  if (!data) return null;
  const states = filterOf(filter).states;
  if (states === 'all') return data.total;
  if (!data.byStatus) return null;
  return states.reduce((n, s) => n + (data.byStatus?.[s] ?? 0), 0);
}

/** The error, whole, as one text to paste into a support conversation. */
function errorReport(r: RunRecord): string {
  const e = r.error;
  if (!e) return '';
  return [
    `${e.code}: ${errorText(e.message)}`,
    e.nodeId ? `Failed at: ${e.nodeId}` : '',
    e.capability ? `Capability: ${e.capability}` : '',
    e.detail ? `\n${e.detail}` : '',
    `\nRun ${r.runId}${r.flowId ? ` of ${r.flowId}` : ''}, ${fullWhen(r.reservedAt)}`,
  ]
    .filter(Boolean)
    .join('\n');
}

/**
 * Why a run failed, held to three lines. Some reasons are a whole paragraph (a
 * list of every node type a flow could not find), and a block of red that size
 * buries the rows below it. The toggle shows only when the three lines hide
 * something; the full text is also the tooltip. A cancelled run's reason is said
 * quietly: nothing went wrong.
 */
function RunError({ message: said, quiet = false }: { message: string; quiet?: boolean }) {
  const message = errorText(said);
  const [all, setAll] = useState(false);
  const [clipped, setClipped] = useState(false);
  const textRef = useRef<HTMLSpanElement>(null);

  // Measured, not guessed from the length: whether it passes three lines
  // depends on the window's width. Not measured while open, so the toggle stays.
  useLayoutEffect(() => {
    const el = textRef.current;
    if (!el || all) return;
    const measure = () => setClipped(el.scrollHeight > el.clientHeight + 1);
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [message, all]);

  return (
    <div className={quiet ? 'run-error is-quiet' : 'run-error'}>
      <p>
        {/* (the icon on the first line's height, beside the message's first line; the error's code is in the run's
            details, where its id and the rest are) */}
        <span className="run-error-code">
          <CircleAlert size={13} aria-hidden />
        </span>{' '}
        <span ref={textRef} className={all ? 'run-error-text' : 'run-error-text clamped'} title={clipped && !all ? message : undefined}>
          {message}
        </span>
      </p>
      {clipped && (
        <button className="run-text-more" aria-expanded={all} onClick={() => setAll(!all)}>
          {all ? 'Show less' : 'Show all'}
        </button>
      )}
    </div>
  );
}

type Busy = 'stop' | 'again';

export default function RunsPanel({ onOpenEditor }: { onOpenEditor?: (flowId: string) => void } = {}) {
  const [filter, setFilter] = useState<Filter>('failures');
  // Keyed BY FILTER. One shared key meant a revisit could paint the cached
  // "all" snapshot under the "Failed runs" heading — a list of green succeeded
  // badges below a title claiming they failed. And if the refresh then failed,
  // it stayed that way indefinitely behind an error banner that said nothing
  // about the list being wrong.
  const [data, setData] = useState<RunHistory | null>(() => peek(`runHistory:${filter}`) ?? null);
  const [error, setError] = useState<string | null>(null);
  /** The error is a read of the list (asked again with "Try again"), not a clear that was refused. */
  const [readFailed, setReadFailed] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  /** The run open below, kept in the list after it left the filter (a run in progress that ended), until closed. */
  const [held, setHeld] = useState<RunRecord | null>(null);
  const [clearing, setClearing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [limit, setLimit] = useState(PAGE);
  /** Older runs, paged back past the newest (`before` the last shown), and whether the desktop has more. */
  const [older, setOlder] = useState<RunRecord[]>([]);
  const [olderMore, setOlderMore] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [query, setQuery] = useState('');
  /** The search the desktop is asked: what was typed, once the typing rests. */
  const [searched, setSearched] = useState('');
  /** One flow's runs alone (its id; '' every flow). */
  const [flowPick, setFlowPick] = useState('');
  const [busy, setBusy] = useState<Record<string, Busy>>({});
  const [names, setNames] = useState<ReadonlyMap<string, string> | null>(() => peek('flowNames') ?? null);
  /** The flows that are the editor's macros, labelled as such in the flow picker. */
  const [macros, setMacros] = useState<ReadonlySet<string>>(() => peek('flowMacros') ?? new Set());
  const [computers, setComputers] = useState<ReadonlyMap<string, string>>(() => peek('networkNames') ?? new Map());
  const [now, setNow] = useState(() => Date.now());
  const toast = useToast();
  // The latest request's number: an answer to an older one (another filter's, asked before the switch) is not shown.
  const asked = useRef(0);
  /** Which list the older pages are of: counted up whenever they are let go (another filter, search or flow; a clear),
   *  so an older page read for the list before is put aside rather than added to this one. */
  const olderOf = useRef(0);
  /** The newest page as it shows now, for the rows a refresh pushes out of it. */
  const headShown = useRef<RunRecord[]>([]);
  headShown.current = data?.runs ?? [];
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const mark = (id: string, what: Busy | null) =>
    setBusy((prev) => {
      const next = { ...prev };
      if (what) next[id] = what;
      else delete next[id];
      return next;
    });

  // Searching or picking a flow asks the desktop, which looks through every run it keeps.
  useEffect(() => {
    const next = query.trim();
    if (next === searched) return;
    const id = window.setTimeout(() => setSearched(next), next ? SEARCH_REST_MS : 0);
    return () => window.clearTimeout(id);
  }, [query, searched]);
  const narrowed = searched !== '' || flowPick !== '';
  const narrowing: RunPage | undefined = narrowed ? { ...(searched ? { q: searched } : {}), ...(flowPick ? { flowId: flowPick } : {}) } : undefined;
  const want = limit;

  // The list keeps its place when rows arrive above the one at the top of the window (WebKit has no scroll
  // anchoring of its own; the list opts out of Chrome's, so both behave the same).
  const listRef = useRef<HTMLUListElement>(null);
  const anchor = useRef<{ id: string; top: number } | null>(null);
  const noteAnchor = () => {
    anchor.current = null;
    const list = listRef.current;
    const scroller = list?.closest<HTMLElement>('.content-page');
    if (!list || !scroller || scroller.scrollTop <= 0) return;
    const top = scroller.getBoundingClientRect().top;
    for (const row of list.querySelectorAll<HTMLElement>('[data-run-id]')) {
      const box = row.getBoundingClientRect();
      if (box.bottom > top + 1) {
        anchor.current = { id: row.dataset.runId ?? '', top: box.top };
        return;
      }
    }
  };

  const refresh = useCallback(async () => {
    const mine = ++asked.current;
    try {
      const next = narrowing ? await bridge.runs(filterOf(filter).states, want, narrowing) : await bridge.runs(filterOf(filter).states, want);
      if (mine !== asked.current) return;
      // With older pages read past the newest, the rows newer runs push out of the newest (it holds `want` at most)
      // go between it and them: they were older than its last row now, and newer than the older pages, so they showed
      // nowhere. (Only those past its end: a row that left the filter within it is gone.)
      const was = headShown.current;
      const last = next.runs[next.runs.length - 1];
      if (last && next.runs.length >= want) {
        const kept = new Set(next.runs.map((r) => r.runId));
        const pushed = was.filter((r) => !kept.has(r.runId) && newerFirst(last, r) < 0);
        if (pushed.length) {
          setOlder((had) => {
            if (!had.length) return had;
            const seen = new Set(had.map((r) => r.runId));
            return [...pushed.filter((r) => !seen.has(r.runId)), ...had].sort(newerFirst);
          });
        }
      }
      // (a search's answer is not the filter's: not kept as what the filter shows next time)
      if (!narrowing) put(`runHistory:${filter}`, next);
      noteAnchor();
      setData(next);
      setNow(Date.now());
      setError(null);
      // Looked at: Overview's warning is about failures since (the newest of these, and how many there are).
      if (filter === 'failures' && !narrowing && next.byStatus) writeFailedSeen(countOf(next, 'failures') ?? 0, lastFailedOf(next.runs));
    } catch (e) {
      if (mine === asked.current) {
        setError(loadErrorText(e instanceof Error ? e.message : String(e), 'Run history'));
        setReadFailed(true);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter, want, searched, flowPick]);
  const latest = useRef(refresh);
  latest.current = refresh;

  // The newest (asked again as they change), then the older pages read past them: each run once, in history's order.
  const listedRuns = useMemo(() => {
    const head = data?.runs ?? [];
    if (!older.length) return head;
    const inHead = new Set(head.map((r) => r.runId));
    const last = head[head.length - 1];
    return [...head, ...older.filter((r) => !inHead.has(r.runId) && (!last || newerFirst(last, r) < 0))];
  }, [data, older]);
  const rows = useMemo(() => {
    const list = listedRuns;
    if (!held || list.some((r) => r.runId === held.runId)) return list;
    // (in its place by when it was asked, as the list is ordered)
    const at = list.findIndex((r) => ms(r.reservedAt) < ms(held.reservedAt));
    return at < 0 ? [...list, held] : [...list.slice(0, at), held, ...list.slice(at)];
  }, [listedRuns, held]);
  // (a run in `queued` mode waits for a claimer elsewhere, maybe for good: it is no reason to ask more often)
  const live = rows.some((r) => r.status === 'running' || (r.status === 'queued' && r.mode !== 'queued'));
  const running = rows.some((r) => r.status === 'running');

  // Asked every few seconds while the window shows (not while it is hidden in the tray); more often while a run goes on.
  useVisiblePoll(() => void refresh(), live ? LIVE_POLL_MS : POLL_MS);
  const shown = useRef({ filter, want, searched, flowPick });
  useEffect(() => {
    const was = shown.current;
    shown.current = { filter, want, searched, flowPick };
    if (was.filter === filter && was.want === want && was.searched === searched && was.flowPick === flowPick) return;
    // Another list: the older pages read for the last one are not this one's.
    if (was.filter !== filter || was.searched !== searched || was.flowPick !== flowPick) {
      olderOf.current += 1;
      setOlder([]);
      setOlderMore(false);
    }
    // Repaint from THIS filter's cache the moment the filter changes, so the
    // heading and the rows can never disagree while the refetch is in flight.
    if (was.filter !== filter) setData(peek<RunHistory>(`runHistory:${filter}`) ?? null);
    void refresh();
  }, [refresh, filter, want, searched, flowPick]);

  // A running run's time ticks.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running]);

  useLayoutEffect(() => {
    const was = anchor.current;
    anchor.current = null;
    const list = listRef.current;
    const scroller = list?.closest<HTMLElement>('.content-page');
    if (!was || !list || !scroller) return;
    const row = [...list.querySelectorAll<HTMLElement>('[data-run-id]')].find((el) => el.dataset.runId === was.id);
    if (!row) return;
    const moved = row.getBoundingClientRect().top - was.top;
    if (Math.abs(moved) > 0.5) scroller.scrollTop += moved;
  }, [data]);

  // The run open below, once it is no longer in the filter's answer: read on its own and kept, so it does not vanish
  // from under the person reading it.
  useEffect(() => {
    if (!expanded || !data) return;
    if (listedRuns.some((r) => r.runId === expanded)) {
      setHeld(null);
      return;
    }
    let gone = false;
    Promise.resolve()
      .then(() => bridge.run(expanded))
      .then((r) => !gone && setHeld(r))
      .catch(() => {});
    return () => {
      gone = true;
    };
  }, [data, listedRuns, expanded]);

  // The flows' names (a run names its flow by id), and the paired computers' (a run one asked for names it by its id).
  const readNames = useCallback(() => {
    Promise.resolve()
      .then(() => bridge.flows())
      .then((r) => {
        const m = new Map(r.flows.map((f) => [f.flowId, f.name] as const));
        const kept = new Set(r.flows.filter((f) => f.macro).map((f) => f.flowId));
        put('flowNames', m);
        put('flowMacros', kept);
        if (alive.current) {
          setNames(m);
          setMacros(kept);
        }
      })
      .catch(() => {});
  }, []);
  useEffect(readNames, [readNames]);
  const askComputers = rows.some((r) => r.callerProduct.startsWith('oaiy-network:'));
  useEffect(() => {
    if (!askComputers) return;
    Promise.resolve()
      .then(() => network.nodes())
      .then((r) => {
        const m = new Map(r.nodes.map((n) => [n.nodeId, n.name] as const));
        put('networkNames', m);
        if (alive.current) setComputers(m);
      })
      .catch(() => {});
  }, [askComputers]);

  const nameOf = (r: RunRecord) => (r.flowId ? (names?.get(r.flowId) ?? r.flowId) : 'A flow sent with the request');

  const pickFilter = (next: Filter) => {
    if (next === filter) return;
    setFilter(next);
    setLimit(PAGE);
    setExpanded(null);
    setHeld(null);
  };

  /** Read the next page of older runs: those asked before the last one shown, as the list is narrowed. */
  const readOlder = async () => {
    const last = listedRuns[listedRuns.length - 1];
    if (!last || loadingOlder) return;
    // (the list it is read for: a page landing after a switch to another filter is not that filter's)
    const of = olderOf.current;
    setLoadingOlder(true);
    try {
      const page = await bridge.runs(filterOf(filter).states, PAGE, { ...(narrowing ?? {}), before: { reservedAt: last.reservedAt, runId: last.runId } });
      if (!alive.current || of !== olderOf.current) return;
      setOlder((had) => {
        const seen = new Set(had.map((r) => r.runId));
        return [...had, ...page.runs.filter((r) => !seen.has(r.runId))];
      });
      setOlderMore(page.more ?? page.runs.length >= PAGE);
    } catch (e) {
      if (of === olderOf.current) {
        setError(loadErrorText(e instanceof Error ? e.message : String(e), 'Older runs'));
        setReadFailed(true);
      }
    } finally {
      setLoadingOlder(false);
    }
  };

  const refreshNow = async () => {
    setRefreshing(true);
    readNames();
    await refresh();
    setRefreshing(false);
  };

  const clear = async () => {
    // Spell out both halves: what survives, and the one consequence that is
    // not obvious — a consumer retrying an old idempotency key gets a fresh
    // run instead of the recorded result, so identical work can run again.
    if (
      !(await ask(
        'Clear finished runs from this history?\n\nQueued and running work is kept. This cannot be undone, and an app that retries a request from before the clear will run it again rather than get the old result.',
      ))
    )
      return;
    setClearing(true);
    try {
      const { cleared } = await bridge.clearRuns();
      // Drop every filter's cache: another filter's cached page still lists rows
      // that no longer exist, and switching to it would show them as current.
      for (const f of FILTERS) invalidate(`runHistory:${f.id}`);
      setExpanded(null);
      setHeld(null);
      olderOf.current += 1;
      setOlder([]);
      setOlderMore(false);
      await refresh();
      toast.push({
        kind: 'success',
        title: cleared === 0 ? 'Nothing to clear' : `Cleared ${cleared} finished run${cleared === 1 ? '' : 's'}`,
      });
    } catch (e) {
      setError(`Finished runs could not be cleared: ${e instanceof Error ? e.message : String(e)}`);
      setReadFailed(false);
    } finally {
      setClearing(false);
    }
  };

  const stop = async (r: RunRecord) => {
    const name = nameOf(r);
    const queued = r.status === 'queued';
    if (
      !(await ask(
        queued
          ? `Cancel this run of ${name}?\n\nIt has not started: it is taken off the queue and never runs.`
          : `Stop this run of ${name}?\n\nIt is asked to stop now. What it has done so far stays done.`,
      ))
    )
      return;
    mark(r.runId, 'stop');
    try {
      const answer = await bridge.cancelRun(r.runId);
      toast.push(
        answer.status === 'cancelled'
          ? { kind: 'success', title: `Cancelled ${name}` }
          : { kind: 'info', title: `Asked ${name} to stop`, body: 'It stops as soon as its runner notices.' },
      );
    } catch (e) {
      const said = e instanceof Error ? e.message : String(e);
      // (it ended meanwhile: the desktop says it is already done, which is not a failure to stop it)
      toast.push(/already/i.test(said) ? { kind: 'info', title: `${name} had already finished` } : { kind: 'error', title: `${name} could not be stopped`, body: said });
    } finally {
      mark(r.runId, null);
    }
    await latest.current();
  };

  /** Follow a run asked for here until it ends, and say how it went. */
  const follow = async (runId: string, name: string) => {
    const until = Date.now() + 15 * 60_000;
    while (alive.current && Date.now() < until) {
      await new Promise((r) => window.setTimeout(r, 1500));
      let rec: RunRecord;
      try {
        rec = await bridge.run(runId);
      } catch {
        continue;
      }
      if (!ENDED.has(rec.status)) continue;
      if (!alive.current) return;
      const time = took(rec, Date.now());
      toast.push(
        rec.status === 'succeeded'
          ? { kind: 'success', title: `${name} ran again and succeeded`, body: time ? `It took ${time}.` : undefined }
          : { kind: rec.status === 'cancelled' ? 'info' : 'error', title: `${name} ran again and ${LABEL[rec.status]}`, body: rec.error ? errorText(rec.error.message) : undefined },
      );
      void latest.current();
      return;
    }
  };

  const again = async (r: RunRecord) => {
    if (!r.flowId) return;
    const name = nameOf(r);
    if (
      !(await ask(
        ranConfined(r)
          ? `Run ${name} again, with the same inputs?\n\nIt runs on this computer now, as a new run, held in as the first was: its inputs came from outside this computer, so it reaches only its own folders and no other address of this computer or its network. Whatever the flow does (a message it sends, a file it writes) is done again.`
          : `Run ${name} again, with the same inputs?\n\nIt runs on this computer now, as a new run. Whatever the flow does (a message it sends, a file it writes) is done again.`,
      ))
    )
      return;
    mark(r.runId, 'again');
    try {
      const next = await bridge.runAgain(r);
      toast.push({ kind: 'info', title: `Running ${name} again`, body: 'It shows under In progress until it ends, and under All after.' });
      void follow(next.runId, name);
      await latest.current();
    } catch (e) {
      toast.push({ kind: 'error', title: `${name} could not be run again`, body: e instanceof Error ? e.message : String(e) });
    } finally {
      mark(r.runId, null);
    }
  };

  // (the search and the flow picked are the desktop's answer: what is listed is what it found)
  const visible = rows;

  // The flows among the runs listed, for the picker: by name, a macro of the editor's said to be one.
  const flowChoices = useMemo(() => {
    const seen = new Map<string, string>();
    const label = (id: string) => `${names?.get(id) ?? id}${macros.has(id) ? ' (macro)' : ''}`;
    for (const r of rows) if (r.flowId && !seen.has(r.flowId)) seen.set(r.flowId, label(r.flowId));
    if (flowPick && !seen.has(flowPick)) seen.set(flowPick, label(flowPick));
    return [...seen.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [rows, names, macros, flowPick]);

  const f = filterOf(filter);
  const total = countOf(data, filter);
  const listed = listedRuns.length;
  const headListed = data?.runs.length ?? 0;
  // More: past the older pages, as the desktop said; past the newest, as it said (or, from a desktop that does not
  // say, as the counts tell).
  const more = data !== null && (older.length
    ? olderMore
    : headListed >= limit && (data.more ?? (!narrowed && total !== null && headListed < total)));
  const showMore = () => {
    // (the newest grow to what one answer holds, asked again as they change; past that, older pages are read)
    if (!older.length && limit < MOST) setLimit(Math.min(MOST, limit + PAGE));
    else void readOlder();
  };
  const moreCount = !narrowed && total !== null ? Math.min(PAGE, total - listed, !older.length && limit < MOST ? MOST - limit : PAGE) : PAGE;

  return (
    <div className="panel">
      {error && <LoadError error={`${error}${data && readFailed ? ' Showing what was read last.' : ''}`} onRetry={readFailed ? refreshNow : undefined} />}

      <section className="service-section run-history">
        <div className="section-title-row">
          <h3 className="section-title">
            {f.heading}
            {total !== null && <span className="section-count">{total}</span>}
          </h3>
          <div className="run-tools">
            <div className="seg" role="group" aria-label="Which runs">
              {FILTERS.map((x) => {
                const n = countOf(data, x.id);
                return (
                  <button key={x.id} className={filter === x.id ? 'active' : ''} aria-pressed={filter === x.id} onClick={() => pickFilter(x.id)}>
                    {x.label}
                    {n !== null && (x.id === 'all' || n > 0) ? ` (${n})` : ''}
                  </button>
                );
              })}
            </div>
            <button className="btn-tiny btn-icon-tiny" onClick={() => void refreshNow()} title="Refresh now" aria-label="Refresh now" disabled={refreshing}>
              <RefreshCw size={13} className={refreshing ? 'spin' : undefined} />
            </button>
            {/* Disabled while empty rather than hidden: a button that appears and
                vanishes with the list is harder to find than one that is always
                in the same place. */}
            <button
              className="btn-tiny btn-icon-tiny btn-danger"
              onClick={() => void clear()}
              disabled={clearing || !data || data.total === 0}
              title="Clear finished runs"
              aria-label="Clear finished runs"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>

        {data !== null && (data.total > 0 || narrowed) && (
          <div className="run-filters">
            <label className="run-search">
              <Search size={13} aria-hidden />
              <input
                type="search"
                value={query}
                placeholder="Search: a flow, an error, who asked, an input"
                aria-label="Search the runs"
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape' && query) {
                    e.preventDefault();
                    setQuery('');
                  }
                }}
              />
              {query && (
                <button type="button" className="run-search-clear" aria-label="Clear the search" title="Clear the search" onClick={() => setQuery('')}>
                  <X size={12} />
                </button>
              )}
            </label>
            {(flowChoices.length > 1 || flowPick !== '') && (
              <select className="run-flow-pick" aria-label="Which flow" value={flowPick} onChange={(e) => setFlowPick(e.target.value)}>
                <option value="">Every flow</option>
                {flowChoices.map(([id, name]) => (
                  <option key={id} value={id}>
                    {name}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        {data === null ? (
          <div className="empty-state empty-state-sm">
            <Loader2 size={16} className="spin" aria-hidden />
            <p>{error ? 'Run history could not be read yet.' : 'Loading runs…'}</p>
          </div>
        ) : visible.length === 0 ? (
          narrowed ? (
            <div className="empty-state empty-state-sm">
              <p>
                No {f.one} matches{searched ? ` “${searched}”` : ''}
                {flowPick ? ` in ${names?.get(flowPick) ?? flowPick}` : ''}.
              </p>
              <button
                className="btn-tiny"
                onClick={() => {
                  setQuery('');
                  setFlowPick('');
                }}
              >
                <X size={12} /> Clear the search
              </button>
            </div>
          ) : (
            <div className="empty-state">
              {filter === 'failures' ? noFailuresMessage(data) : filter === 'going' ? 'Nothing is running or waiting to run.' : 'No flow has run on this computer yet.'}
            </div>
          )
        ) : (
          <ul className="run-list" ref={listRef}>
            {visible.map((r) => (
              <RunRow
                key={r.runId}
                r={r}
                name={nameOf(r)}
                flowKnown={names === null || !r.flowId ? null : names.has(r.flowId)}
                who={whoAsked(r, computers)}
                open={expanded === r.runId}
                held={held?.runId === r.runId ? f.none : null}
                now={now}
                busy={busy[r.runId]}
                onToggle={() => {
                  setExpanded(expanded === r.runId ? null : r.runId);
                  if (held?.runId === r.runId) setHeld(null);
                }}
                onStop={() => void stop(r)}
                onAgain={() => void again(r)}
                onOpenEditor={onOpenEditor}
              />
            ))}
          </ul>
        )}

        {data !== null && visible.length > 0 && (
          <div className="run-foot">
            <span className="form-hint">
              {narrowed
                ? `${visible.length} ${visible.length === 1 ? f.one : f.none} match${visible.length === 1 ? 'es' : ''}${more ? ', and older ones too' : ''}.`
                : total !== null && listed < total
                  ? `Showing the newest ${listed} of ${total} ${f.none}.`
                  : `${listed} ${listed === 1 ? f.one : f.none}.`}
            </span>
            {more && (
              <button className="btn-tiny" onClick={showMore} disabled={loadingOlder}>
                {loadingOlder ? <Loader2 size={12} className="spin" aria-hidden /> : null} Show {moreCount > 0 ? moreCount : PAGE} more
              </button>
            )}
            {narrowed && (
              <button
                className="btn-tiny"
                onClick={() => {
                  setQuery('');
                  setFlowPick('');
                }}
              >
                <X size={12} /> Clear the search
              </button>
            )}
          </div>
        )}
      </section>

      {/* Below the runs, where it is no longer the first thing the page says: an event that never became a run. It is
          there only while there is one. */}
      <DeadLetters />
    </div>
  );
}

interface RowProps {
  r: RunRecord;
  name: string;
  /** Whether its flow is still stored here (null: not known yet, or the run carried its own). */
  flowKnown: boolean | null;
  who: string;
  open: boolean;
  /** The filter it has left, while it is kept in the list because it is open. */
  held: string | null;
  now: number;
  busy?: Busy;
  onToggle: () => void;
  onStop: () => void;
  onAgain: () => void;
  onOpenEditor?: (flowId: string) => void;
}

function RunRow({ r, name, flowKnown, who, open, held, now, busy, onToggle, onStop, onAgain, onOpenEditor }: RowProps) {
  const going = r.status === 'queued' || r.status === 'running';
  const at = ms(r.reservedAt);
  const time = took(r, now);
  const stopping = busy === 'stop' || !!r.cancelRequested;
  return (
    <li className={`run-row run-${r.status}${open ? ' is-open' : ''}`} data-run-id={r.runId}>
      <div className="run-line">
        <button type="button" className="run-head" aria-expanded={open} aria-controls={`run-${r.runId}`} onClick={onToggle}>
          <span className="run-state">
            <span className={badgeFor(r.status)}>{LABEL[r.status]}</span>
          </span>
          <span className="run-name">
            <strong className="run-flow">{name}</strong>
            {r.flowId && name !== r.flowId && <span className="run-flow-id">{r.flowId}</span>}
          </span>
          {/* (a trigger's run says its event, in words: the event's name, and which trigger it was, are in its details) */}
          <span className="run-meta" title={r.triggerEvent ? `${who}: ${r.triggerEvent}` : who}>
            {r.triggerEvent ? (r.callerProduct.startsWith('oaiy-trigger:') ? `When ${eventWords(r.triggerEvent)}` : `${who} · when ${eventWords(r.triggerEvent)}`) : who}
          </span>
          <span className="run-took">{time}</span>
          <time className="run-when" dateTime={r.reservedAt} title={`Asked ${fullWhen(r.reservedAt)}`}>
            {whenText(at, now)}
          </time>
          <ChevronDown size={14} className="run-chevron" aria-hidden />
        </button>
        {going && (
          <button className="btn-tiny run-stop" disabled={stopping} onClick={onStop} title={r.status === 'queued' ? 'Take it off the queue' : 'Ask it to stop'}>
            {stopping ? <Loader2 size={11} className="spin" aria-hidden /> : <Square size={11} aria-hidden />}
            {stopping ? 'Stopping…' : r.status === 'queued' ? 'Cancel' : 'Stop'}
          </button>
        )}
      </div>

      {/* The reason, on the row, unexpanded. Making someone click to
          find out why a run failed is the problem this panel fixes. */}
      {r.error && <RunError message={r.error.message} quiet={r.status === 'cancelled'} />}
      {held && <p className="run-held form-hint">No longer among the {held}: it {LABEL[r.status]}. It stays here while it is open.</p>}

      {open && <RunDetail r={r} name={name} flowKnown={flowKnown} who={who} busy={busy} onAgain={onAgain} onOpenEditor={onOpenEditor} />}
    </li>
  );
}

function RunDetail({ r, name, flowKnown, who, busy, onAgain, onOpenEditor }: Pick<RowProps, 'r' | 'name' | 'flowKnown' | 'who' | 'busy' | 'onAgain' | 'onOpenEditor'>) {
  const report = r.output && typeof r.output === 'object' && !Array.isArray(r.output) ? (r.output as Record<string, unknown>) : null;
  const hasAnswer = !!report && 'output' in report;
  const steps = report && report.results && typeof report.results === 'object' ? report.results : null;
  const input = r.input && typeof r.input === 'object' && !Array.isArray(r.input) ? r.input : null;
  const ended = ENDED.has(r.status);
  const timeTaken = took(r, Date.now());
  return (
    <div className="run-body" id={`run-${r.runId}`}>
      <div className="run-actions">
        {r.flowId && ended && (
          <button
            className="btn-tiny"
            disabled={flowKnown === false || !!busy}
            onClick={onAgain}
            title={flowKnown === false ? `${r.flowId} is not on this computer any more` : 'Run it again here, with the same inputs'}
          >
            {busy === 'again' ? <Loader2 size={12} className="spin" aria-hidden /> : <RotateCcw size={12} aria-hidden />} {busy === 'again' ? 'Starting…' : 'Run again'}
          </button>
        )}
        {r.flowId && flowKnown && onOpenEditor && (
          <button className="btn-tiny" onClick={() => onOpenEditor(r.flowId!)} title={`Open ${name} in the flow editor`}>
            <Workflow size={12} aria-hidden /> Open the editor
          </button>
        )}
        <CopyButton value={r.runId} label="Copy the run id" text="Copy run id" />
        {r.error && <CopyButton value={errorReport(r)} label="Copy the error, in full" text="Copy the error" />}
      </div>
      {!r.flowId && ended && <p className="form-hint">Its flow came with the request and was not kept, so it cannot be run again from here.</p>}
      {r.flowId && flowKnown === false && <p className="form-hint">{r.flowId} is not among this computer’s flows any more, so it cannot be run again from here.</p>}

      <dl className="run-detail">
        <dt>Run</dt>
        <dd>
          <code>{r.runId}</code>
        </dd>
        {r.flowId && (
          <>
            <dt>Flow</dt>
            <dd>
              {name !== r.flowId && <>{name} </>}
              <code>{r.flowId}</code>
            </dd>
          </>
        )}
        <dt>Asked by</dt>
        <dd>
          {who}
          {who !== r.callerProduct && (
            <>
              {' '}
              <code>{r.callerProduct}</code>
            </>
          )}
        </dd>
        {r.triggerEvent && (
          <>
            <dt>Event</dt>
            <dd>
              <code>{r.triggerEvent}</code>
            </dd>
          </>
        )}
        {r.lineage?.parentRunId && (
          <>
            <dt>Started by run</dt>
            <dd>
              <code>{r.lineage.parentRunId}</code>
            </dd>
          </>
        )}
        <dt>Asked</dt>
        <dd>{fullWhen(r.reservedAt)}</dd>
        {r.startedAt && (
          <>
            <dt>Started</dt>
            <dd>{fullWhen(r.startedAt)}</dd>
          </>
        )}
        {r.finishedAt && (
          <>
            <dt>Ended</dt>
            <dd>
              {fullWhen(r.finishedAt)}
              {timeTaken ? ` · took ${timeTaken}` : ''}
            </dd>
          </>
        )}
        <dt>Time limit</dt>
        <dd>{r.timeoutMs ? durationText(r.timeoutMs) : '2 min (the worker’s own)'}</dd>
        <dt>Mode</dt>
        <dd>
          {r.mode}
          {r.runtime ? ` · ${r.runtime}` : ''}
          {r.confined ? ' · held in, as the run it repeats' : ''}
          {r.cancelRequested ? ' · asked to stop' : ''}
        </dd>
        <dt>Correlation</dt>
        <dd>
          <code>{r.correlationId}</code>
        </dd>
        {r.error?.code && (
          <>
            <dt>Error</dt>
            <dd>
              <code>{r.error.code}</code>
            </dd>
          </>
        )}
        {r.error?.nodeId && (
          <>
            <dt>Failed at</dt>
            <dd>
              <code>{r.error.nodeId}</code>
            </dd>
          </>
        )}
        {r.error?.capability && (
          <>
            <dt>Capability</dt>
            <dd>
              <code>{r.error.capability}</code>
            </dd>
          </>
        )}
      </dl>

      {r.error?.detail && (
        <div className="run-io">
          <h4 className="run-io-title">What went wrong, in full</h4>
          <pre className="run-detail-text">{r.error.detail}</pre>
        </div>
      )}
      {input && (
        <div className="run-io">
          <h4 className="run-io-title">Asked with</h4>
          <Fields values={input} />
        </div>
      )}
      {hasAnswer && (
        <div className="run-io">
          <h4 className="run-io-title">Its answer</h4>
          <RunAnswer answer={report!.output} />
        </div>
      )}
      {!hasAnswer && report && <JsonBlock value={report} summary="What its runner reported" />}
      {steps !== null && <JsonBlock value={steps} summary="Each step’s result" />}
    </div>
  );
}
