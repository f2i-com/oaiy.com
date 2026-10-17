import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Network, Play, Server, Square, TriangleAlert } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { network, type NetworkFlow, type NetworkNode, type NetworkRun, type NetworkStatus } from './api';

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
      setNodes(s.uses ? (await network.nodes()).nodes.filter((n) => (n.uses ?? n.role === 'worker')) : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 8000);

  if (!status) {
    return <div className="panel">{error ? <div className="banner banner-err">{error}</div> : <div className="empty-state">Loading…</div>}</div>;
  }
  if (!status.uses || nodes.length === 0) {
    return (
      <div className="panel" data-testid="flow-workers">
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">Flows on other computers</h3>
          </div>
          <p className="form-hint">
            {status.uses
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
      {nodes.map((n) => (
        <section className="model-section" key={n.nodeId} data-testid="flow-worker">
          <div className="section-title-row">
            <h3 className="section-title">
              {n.name} <span className={`badge ${n.online ? 'badge-ok' : 'badge-err'}`}>{n.online ? 'Online' : 'Offline'}</span>
            </h3>
          </div>
          {(n.flows ?? []).length === 0 ? (
            <p className="form-hint">
              {n.name} shares no flow yet: on {n.name}, open a flow in Flows and switch on <em>Let apps run it</em>.
            </p>
          ) : (
            <div className="network-list">
              {(n.flows ?? []).map((f) => (
                <FlowCard key={f.flowId} node={n} flow={f} />
              ))}
            </div>
          )}
        </section>
      ))}
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
}

/**
 * One shared flow: its inputs, Run, and its runs as they go. Run again while one runs and the next waits in the
 * worker's queue (one at a time there); each is followed, can be stopped, and shows its answer when it is done.
 */
function FlowCard({ node, flow }: { node: NetworkNode; flow: NetworkFlow }) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(flow.inputs.map((i) => [i.name, typeof i.default === 'string' ? i.default : i.default == null ? '' : JSON.stringify(i.default)])),
  );
  const [runs, setRuns] = useState<Tracked[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [now, setNow] = useState(Date.now());
  const timer = useRef<number | undefined>(undefined);
  const going = runs.filter((t) => !TERMINAL.has(t.run.status));

  // While any runs: each asked every second and a half, and the time each has taken ticks.
  useEffect(() => {
    if (going.length === 0) return;
    let stopped = false;
    const tick = async () => {
      setNow(Date.now());
      const next = await Promise.all(
        going.map((t) =>
          network
            .run(node.nodeId, t.run.runId)
            .then((run) => ({ ...t, run, error: null }))
            .catch((e: unknown) => ({ ...t, error: e instanceof Error ? e.message : String(e) })),
        ),
      );
      if (stopped) return;
      setRuns((all) => all.map((t) => next.find((n) => n.run.runId === t.run.runId) ?? t));
    };
    timer.current = window.setTimeout(tick, 1500);
    return () => {
      stopped = true;
      window.clearTimeout(timer.current);
    };
  }, [runs, node.nodeId]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      const run = await network.runFlow(node.nodeId, flow.flowId, values);
      setNow(Date.now());
      setRuns((all) => [{ run, started: Date.now() }, ...all].slice(0, 6));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };
  const stop = async (t: Tracked) => {
    try {
      await network.cancelRun(node.nodeId, t.run.runId);
      const run = await network.run(node.nodeId, t.run.runId);
      setRuns((all) => all.map((x) => (x.run.runId === run.runId ? { ...x, run } : x)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
              <textarea rows={2} value={values[i.name] ?? ''} onChange={(e) => setValues((v) => ({ ...v, [i.name]: e.target.value }))} />
            </label>
          ))}
        </div>
      )}
      <div className="form-actions">
        <button className="btn btn-primary" disabled={!node.online || starting} onClick={() => void start()}>
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
        const running = !TERMINAL.has(t.run.status);
        const end = t.run.finishedAt ? Date.parse(t.run.finishedAt) : running ? now : t.started;
        const seconds = Math.max(0, Math.round((end - t.started) / 1000));
        return (
          <div className="flow-run-item" key={t.run.runId} data-testid="flow-run">
            <div className="flow-run-item-head">
              {state && <span className={`badge ${state.badge}`}>{state.label}</span>}
              <span className="flow-run-time">{seconds} s</span>
              {running && (
                <button className="btn-tiny" onClick={() => void stop(t)}>
                  <Square size={12} aria-hidden /> Stop
                </button>
              )}
            </div>
            {(t.error || t.run.error) && (
              <div className="card-warn" role="alert">
                <TriangleAlert size={13} aria-hidden /> {t.run.error?.message ?? t.error}
              </div>
            )}
            {t.run.status === 'succeeded' && (i === 0 || runs.length <= 2) && <RunAnswer answer={t.run.output?.output ?? t.run.output} />}
            {t.run.status === 'succeeded' && i > 0 && runs.length > 2 && (
              <details>
                <summary className="form-hint">Its answer</summary>
                <RunAnswer answer={t.run.output?.output ?? t.run.output} />
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** The addresses of the pictures an answer names (a worker's, at this desktop's route for it). */
function pictures(answer: unknown): string[] {
  const found = JSON.stringify(answer ?? null).match(/https?:\/\/[^"\s]+\/api\/network\/nodes\/[A-Za-z0-9]+\/gateway\/files\/[^"\s]+?\.(?:png|jpe?g|webp)/gi) ?? [];
  return [...new Set(found)].slice(0, 12);
}

/** What a flow answered: its pictures, and its words (a text, or each text field of an answer). */
function RunAnswer({ answer }: { answer: unknown }) {
  const shots = pictures(answer);
  const texts: [string, string][] =
    typeof answer === 'string'
      ? [['', answer]]
      : answer && typeof answer === 'object' && !Array.isArray(answer)
        ? Object.entries(answer as Record<string, unknown>).filter((e): e is [string, string] => typeof e[1] === 'string' && !shots.includes(e[1]))
        : [];
  return (
    <div className="flow-run-answer">
      {shots.length > 0 && (
        <div className="flow-run-pictures">
          {shots.map((s) => (
            <Picture key={s} address={s} />
          ))}
        </div>
      )}
      {texts.map(([k, v]) => (
        <div key={k} className="flow-run-text">
          {k && <span className="settings-label">{k}</span>}
          <p>{v.length > 1200 ? `${v.slice(0, 1200)}…` : v}</p>
        </div>
      ))}
      {shots.length === 0 && texts.length === 0 && <pre className="flow-run-json">{JSON.stringify(answer, null, 2).slice(0, 4000)}</pre>}
    </div>
  );
}

function Picture({ address }: { address: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    let gone = false;
    network
      .picture(address)
      .then((blob) => {
        if (gone) return;
        url = URL.createObjectURL(blob);
        setSrc(url);
      })
      .catch(() => {});
    return () => {
      gone = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [address]);
  return src ? <img className="flow-run-picture" src={src} alt="A picture the flow made" /> : <div className="flow-run-picture is-loading" aria-label="Loading a picture" />;
}
