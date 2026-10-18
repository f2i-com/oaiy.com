import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Network, Play, Server, Square, TriangleAlert } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { network, type NetworkFlow, type NetworkNode, type NetworkRun, type NetworkStatus } from './api';
import { RunAnswer, durationText, errorText, fullWhen, whenText } from './RunParts';

/**
 * Flows → On other computers: the flows the computers paired over OAIY Network share (Let apps run it there), each
 * run there from here, by that computer's own runner, models and GPUs: its inputs filled in, followed as it runs (and
 * stopped), its answer and pictures shown when it is done (docs/NETWORK.md).
 */
export default function FlowWorkersPanel({ onOpenNetworking }: { onOpenNetworking: () => void }) {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await network.status();
      setStatus(s);
      setError(null);
      const used = s.uses ? (await network.nodes()).nodes.filter((n) => (n.uses ?? n.role === 'worker')) : [];
      // The computers answering first, then by name: one that is off is not what anyone came to run on.
      setNodes(used.sort((a, b) => Number(b.online) - Number(a.online) || a.name.localeCompare(b.name)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 8000);

  if (!status) {
    return (
      <div className="panel">
        {error ? (
          <div className="banner banner-err">{error}</div>
        ) : (
          <div className="empty-state empty-state-sm">
            <Loader2 size={16} className="spin" aria-hidden />
            <p>Loading…</p>
          </div>
        )}
      </div>
    );
  }
  if (!status.uses || nodes.length === 0) {
    return (
      <div className="panel" data-testid="flow-workers">
        {error && <div className="banner banner-err">{error}</div>}
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">Flows on other computers</h3>
          </div>
          <p className="form-hint">
            {status.uses && error
              ? 'The computers paired with this one could not be read just now: it tries again in a few seconds.'
              : status.uses
              ? 'No computer is paired to be used yet. Pair one in Setup → Networking, and the flows it shares show here, to run on it from this computer.'
              : status.lends
                ? 'This computer lends itself (OAIY Network): the flows it shares run here when a paired computer asks. Switch on Use other computers too to run other computers’ flows from here.'
                : 'With OAIY Network, this computer runs the flows another OAIY computer on your network shares, on that computer’s own models and GPUs. It is off until you switch on Use other computers.'}
          </p>
          <div className="form-actions">
            <button className="btn btn-secondary" onClick={onOpenNetworking}>
              <Network size={14} aria-hidden /> Open Networking
            </button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="panel" data-testid="flow-workers">
      {error && <div className="banner banner-err">{error}</div>}
      <p className="form-hint">
        A paired computer’s shared flows run there, on its own models and GPUs, and their answers come back here. A computer shares a flow with{' '}
        <em>Let apps run it</em> in its Flows editor.
      </p>
      {nodes.map((n) => {
        const flows = n.flows ?? [];
        const queue = (n.jobs ?? []).filter((j) => j.kind === 'flow' || j.status === 'running');
        const runningThere = queue.filter((j) => j.status === 'running').length;
        const waitingThere = queue.length - runningThere;
        return (
          <section className={`model-section flow-worker${n.online ? '' : ' is-offline'}`} key={n.nodeId} data-testid="flow-worker">
            <div className="section-title-row">
              <h3 className="section-title">
                {n.name} <span className={`badge ${n.online ? 'badge-ok' : 'badge-err'}`}>{n.online ? 'Online' : 'Offline'}</span>
                {flows.length > 0 && (
                  <span className="section-count">
                    {flows.length} shared flow{flows.length === 1 ? '' : 's'}
                  </span>
                )}
              </h3>
              {n.online && queue.length > 0 && (
                <span className="form-hint" title="Its queue: pictures and flow runs, one at a time">
                  Busy there: {runningThere > 0 ? `${runningThere} running` : ''}
                  {runningThere > 0 && waitingThere > 0 ? ', ' : ''}
                  {waitingThere > 0 ? `${waitingThere} waiting` : ''}
                </span>
              )}
            </div>
            {!n.online && (
              <p className="form-hint">
                {n.name} is not answering{n.lastSeenMs ? ` (last seen ${whenText(n.lastSeenMs).replace(/^just now$/, 'a moment ago')})` : ''}
                {n.lastError ? `: ${n.lastError}` : '.'} {flows.length === 0 ? 'The flows it shares show here when it answers again.' : 'Its flows run again once it answers.'}
              </p>
            )}
            {flows.length === 0 ? (
              n.online && (
                <p className="form-hint">
                  {n.name} shares no flow yet: on {n.name}, open a flow in Flows and switch on <em>Let apps run it</em>.
                </p>
              )
            ) : (
              <div className="flow-run-grid">
                {[...flows]
                  .sort((a, b) => a.name.localeCompare(b.name))
                  .map((f) => (
                    <FlowCard key={f.flowId} node={n} flow={f} />
                  ))}
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}

const TERMINAL = new Set(['succeeded', 'failed', 'timed_out', 'cancelled']);
const RUN_STATE: Record<NetworkRun['status'], { label: string; badge: string }> = {
  queued: { label: 'Waiting in its queue', badge: 'badge-pending' },
  running: { label: 'Running', badge: 'badge-pending' },
  succeeded: { label: 'Done', badge: 'badge-ok' },
  failed: { label: 'Failed', badge: 'badge-err' },
  timed_out: { label: 'Took too long', badge: 'badge-err' },
  cancelled: { label: 'Stopped', badge: 'badge-neutral' },
};

/** A run of a flow on a worker, as this page follows it. */
interface Tracked {
  run: NetworkRun;
  started: number;
  error?: string | null;
  /** Asks in a row that went unanswered: past a few, the run is let go (said so) rather than asked about forever. */
  misses?: number;
  /** Let go: no longer asked about. */
  lost?: boolean;
  /** When this computer saw it end. */
  ended?: number;
  /** Found in the worker's queue (asked for from this computer, but not from this page: the Agent, or before OAIY
   *  started again), not started here: its time is the worker's own, said once it ends. */
  found?: boolean;
}

/** Each flow card's runs, kept while the page is away (another tab), so a run going on there is still followed. */
const followed = new Map<string, Tracked[]>();
/** Asks in a row that went unanswered before a run is let go. */
const MAX_MISSES = 8;
/** The runs a card keeps showing, newest first. */
const KEPT = 6;

/** Forget the runs followed (tests start from none). */
export function forgetFollowedRuns() {
  followed.clear();
}

const defaultText = (value: unknown) => (typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value));

/** How long a run took, by the worker's own clock (its start and end both), when it says both. */
function workerTook(run: NetworkRun): number | null {
  const a = run.startedAt ? Date.parse(run.startedAt) : NaN;
  const b = run.finishedAt ? Date.parse(run.finishedAt) : NaN;
  return Number.isFinite(a) && Number.isFinite(b) && b >= a ? b - a : null;
}

/**
 * One shared flow: its inputs, Run, and its runs as they go. Run again while one runs and the next waits in the
 * worker's queue (one at a time there); each is followed, can be stopped, and shows its answer when it is done.
 */
function FlowCard({ node, flow }: { node: NetworkNode; flow: NetworkFlow }) {
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(flow.inputs.map((i) => [i.name, defaultText(i.default)])));
  const card = `${node.nodeId}/${flow.flowId}`;
  const [runs, setRunsState] = useState<Tracked[]>(() => followed.get(card) ?? []);
  const setRuns = (next: Tracked[] | ((all: Tracked[]) => Tracked[])) =>
    setRunsState((all) => {
      const value = typeof next === 'function' ? next(all) : next;
      followed.set(card, value);
      return value;
    });
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState<Record<string, boolean>>({});
  const [now, setNow] = useState(Date.now());
  const timer = useRef<number | undefined>(undefined);
  const going = runs.filter((t) => !TERMINAL.has(t.run.status) && !t.lost);

  // A run of this flow in the worker's queue that this computer asked for (the Agent's, or one from before OAIY
  // started again) is followed here too.
  const jobs = node.jobs;
  useEffect(() => {
    const found = (jobs ?? []).filter((j) => j.kind === 'flow' && j.mine && j.runId && j.flow === flow.flowId);
    if (found.length === 0) return;
    setRuns((all) => {
      const fresh = found.filter((j) => !all.some((t) => t.run.runId === j.runId));
      if (fresh.length === 0) return all;
      const adopted = fresh.map((j): Tracked => ({ run: { runId: j.runId!, status: j.status === 'running' ? 'running' : 'queued', flowId: flow.flowId }, started: Date.now(), found: true }));
      return [...adopted, ...all].slice(0, KEPT);
    });
  }, [jobs, flow.flowId]); // eslint-disable-line react-hooks/exhaustive-deps

  // While any runs: each asked every second and a half, and the time each has taken ticks.
  useEffect(() => {
    if (going.length === 0) return;
    let stopped = false;
    const tick = async () => {
      setNow(Date.now());
      // (the window hidden: asked again once it is shown, not meanwhile)
      if (document.hidden) {
        timer.current = window.setTimeout(tick, 1500);
        return;
      }
      const next = await Promise.all(
        going.map((t) =>
          network
            .run(node.nodeId, t.run.runId)
            .then((run): Tracked => ({ ...t, run, error: null, misses: 0, ended: TERMINAL.has(run.status) ? Date.now() : undefined }))
            .catch((e: unknown): Tracked => {
              const misses = (t.misses ?? 0) + 1;
              const said = e instanceof Error ? e.message : String(e);
              return misses >= MAX_MISSES
                ? { ...t, misses, lost: true, ended: Date.now(), error: `Lost track of this run: ${node.name} did not answer (${said}). It may still be running there.` }
                : { ...t, misses, error: said };
            }),
        ),
      );
      if (stopped) return;
      // (an answer that ended a run is kept: one asked before a Stop, answered after it, does not undo the stop)
      setRuns((all) => all.map((t) => (TERMINAL.has(t.run.status) ? t : (next.find((n) => n.run.runId === t.run.runId) ?? t))));
    };
    // (a run not answering is asked less often, up to every 10 s)
    const misses = Math.max(0, ...going.map((t) => t.misses ?? 0));
    timer.current = window.setTimeout(tick, Math.min(10_000, 1500 * 2 ** misses));
    return () => {
      stopped = true;
      window.clearTimeout(timer.current);
    };
  }, [runs, node.nodeId]); // eslint-disable-line react-hooks/exhaustive-deps

  const valueOf = (name: string, fallback: unknown) => values[name] ?? defaultText(fallback);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      // (the flow's inputs as it shares them now: one it no longer has is not sent)
      const input = Object.fromEntries(flow.inputs.map((i) => [i.name, valueOf(i.name, i.default)]));
      const run = await network.runFlow(node.nodeId, flow.flowId, input);
      setNow(Date.now());
      // (the worker's queue, read meanwhile, may have named it already: once here, as started here)
      setRuns((all) => [{ run, started: Date.now(), ended: TERMINAL.has(run.status) ? Date.now() : undefined }, ...all.filter((t) => t.run.runId !== run.runId)].slice(0, KEPT));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };
  const stop = async (t: Tracked) => {
    if (!window.confirm(`Stop this run of ${flow.name} on ${node.name}?\n\nWhat it has done there so far stays done.`)) return;
    setStopping((s) => ({ ...s, [t.run.runId]: true }));
    try {
      await network.cancelRun(node.nodeId, t.run.runId);
      const run = await network.run(node.nodeId, t.run.runId);
      setRuns((all) => all.map((x) => (x.run.runId === run.runId ? { ...x, run, ended: TERMINAL.has(run.status) ? Date.now() : undefined } : x)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStopping((s) => {
        const next = { ...s };
        delete next[t.run.runId];
        return next;
      });
    }
  };

  return (
    <div className="flow-run-card" data-testid="flow-run-card">
      <div className="flow-run-head">
        <Server size={15} aria-hidden />
        <strong>{flow.name}</strong>
        {going.length > 1 && <span className="badge badge-pending">{going.length} in its queue</span>}
      </div>
      {flow.description && <p className="form-hint">{flow.description}</p>}
      {flow.inputs.length > 0 && (
        <div className="flow-run-inputs">
          {flow.inputs.map((i) => (
            <label className="form-row" key={i.name}>
              <span>{i.name}</span>
              <textarea rows={2} value={valueOf(i.name, i.default)} onChange={(e) => setValues((v) => ({ ...v, [i.name]: e.target.value }))} />
            </label>
          ))}
        </div>
      )}
      <div className="form-actions">
        <button className="btn btn-primary" disabled={!node.online || starting} onClick={() => void start()} title={node.online ? undefined : `${node.name} is not answering`}>
          {starting ? <Loader2 size={14} className="spin" aria-hidden /> : <Play size={14} aria-hidden />} {going.length > 0 ? `Queue another on ${node.name}` : `Run on ${node.name}`}
        </button>
      </div>
      {error && (
        <div className="card-warn" role="alert">
          <TriangleAlert size={13} aria-hidden /> {error}
        </div>
      )}
      {runs.map((t, i) => {
        const state = RUN_STATE[t.run.status];
        const running = !TERMINAL.has(t.run.status) && !t.lost;
        // (this computer's clock alone: the worker's own may differ, so its finishing time is not subtracted from ours;
        // a run found in its queue says the worker's own time once it ends)
        const end = running ? now : (t.ended ?? t.started);
        const seconds = t.found ? (running ? null : workerTook(t.run)) : Math.max(0, end - t.started);
        const said = t.run.error?.message ? errorText(t.run.error.message) : t.error;
        return (
          <div className="flow-run-item" key={t.run.runId} data-testid="flow-run">
            <div className="flow-run-item-head">
              {state && <span className={`badge ${state.badge}`}>{state.label}</span>}
              {seconds !== null && <span className="flow-run-time">{durationText(seconds) || '0 s'}</span>}
              <span className="flow-run-time flow-run-asked" title={`Asked ${fullWhen(t.started)}`}>
                {t.found ? 'asked from this computer' : `asked ${whenText(t.started, now)}`}
              </span>
              {running && (
                <button className="btn-tiny" disabled={!!stopping[t.run.runId]} onClick={() => void stop(t)}>
                  {stopping[t.run.runId] ? <Loader2 size={12} className="spin" aria-hidden /> : <Square size={12} aria-hidden />} {stopping[t.run.runId] ? 'Stopping…' : 'Stop'}
                </button>
              )}
            </div>
            {said && (
              <div className={t.run.status === 'cancelled' ? 'card-warn is-quiet' : 'card-warn'} role="alert">
                <TriangleAlert size={13} aria-hidden /> {said}
              </div>
            )}
            {t.run.status === 'succeeded' && (i === 0 || runs.length <= 2) && <RunAnswer answer={t.run.output?.output ?? t.run.output} />}
            {t.run.status === 'succeeded' && i > 0 && runs.length > 2 && (
              <details className="run-json">
                <summary>Its answer</summary>
                <RunAnswer answer={t.run.output?.output ?? t.run.output} />
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}
