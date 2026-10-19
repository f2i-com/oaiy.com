import { useCallback, useEffect, useRef, useState } from 'react';
import { Laptop, Loader2, Network, RotateCcw, Server, Star, TriangleAlert } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { engines, network, type NetworkModel, type NetworkNode, type NetworkStatus } from './api';
import { DefaultChip, Device, KIND, MachineTiles, Queue, STATE, ago, defaultDown, deviceName, notInEffect } from './NetworkPanel';
import LoadError from './LoadError';

/**
 * Engines → Other computers: the models of the workers paired over OAIY Network (docs/NETWORK.md), by computer, and
 * which of them is this computer's engine default of its kind in place of its own: what OAIY's own routes then use
 * (the Agent on the model chosen in Engines, Flows' "Default … model", the desktop's engine route). The Engines
 * page's own gateway (port 8080) and its Playground are not among them: they answer with this computer's own model
 * unless a paired one is picked there. "Use this computer's", beside a default, makes this computer's own the default
 * again. A default is not in effect while the paired computer cannot run it (the desktop says why: not heard from
 * since this computer started, its engines down, the model gone there), and this computer's own runs meanwhile; one
 * still in effect while its computer's engines are not answering is said too, as nothing runs in its place then. A
 * computer that is offline cannot be made the default until it answers again.
 */
export default function EngineWorkersPanel({ onOpenNetworking }: { onOpenNetworking: () => void }) {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [local, setLocal] = useState<Record<string, string | null>>({});
  /** Why the paired computers could not be read (a "Try again" asks again). */
  const [error, setError] = useState<string | null>(null);
  /** Why a default was not changed. */
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  // (each read numbered: one answered after a newer one began, or after a default was chosen, is not shown)
  const reads = useRef(0);
  const refresh = useCallback(async () => {
    const seq = ++reads.current;
    try {
      const s = await network.status();
      if (seq !== reads.current) return;
      setStatus(s);
      setError(null);
      if (!s.uses) {
        setNodes([]);
        return;
      }
      const [n, c] = await Promise.all([network.nodes(), engines.catalog().catch(() => null)]);
      if (seq !== reads.current) return;
      setNodes(n.nodes.filter((x) => x.uses ?? x.role === 'worker'));
      // (this computer's own defaults kept as last read where the engines did not answer, not shown as none)
      if (c?.defaults) setLocal(c.defaults);
    } catch (e) {
      if (seq !== reads.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 5000);

  const choose = async (kind: string, id: string | null) => {
    setBusy(`${kind}:${id ?? ''}`);
    setActionError(null);
    reads.current++;
    try {
      setStatus(await network.setDefault(kind, id));
      await refresh();
    } catch (e) {
      setActionError(`The default was not changed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  if (!status) {
    return <div className="panel">{error ? <LoadError error={error} what="The other computers" onRetry={refresh} /> : <div className="empty-state">Loading…</div>}</div>;
  }

  if (!status.uses) {
    return (
      <div className="panel" data-testid="engine-workers">
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">Models on other computers</h3>
          </div>
          <p className="form-hint">
            {status.lends
              ? 'This computer lends itself (OAIY Network). Switch on Use other computers too, in Setup → Networking, to use other computers’ models here.'
              : 'With OAIY Network, this computer uses the models of other OAIY computers on your network as its own, and can make one its default for a kind. It is off until you switch on Use other computers.'}
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

  const defaults = status.defaults ?? {};
  // (the kinds the workers have, and any whose default is a worker's still: one not heard from yet, or whose model
  // has gone, can be given back to this computer all the same)
  const kinds = [...new Set([...nodes.flatMap((n) => n.models.map((m) => m.kind)), ...Object.keys(defaults)])].sort((a, b) => kindOrder(a) - kindOrder(b));
  /** Whether `m` of `node` is its kind's default here (a worker's language model: whichever it runs now). */
  const isDefault = (node: NetworkNode, m: NetworkModel) => {
    const d = defaults[m.kind];
    return !!d && (m.kind === 'llm' ? d.nodeId === node.nodeId : d.id === m.id);
  };

  return (
    <div className="panel" data-testid="engine-workers">
      {error && <LoadError error={error} what="The other computers" onRetry={refresh} />}
      {actionError && (
        <div className="banner banner-err" role="alert">
          {actionError}
        </div>
      )}
      <section className="model-section">
        <div className="section-title-row">
          <h3 className="section-title">Defaults</h3>
        </div>
        <p className="form-hint" data-testid="defaults-what">
          The model of each kind that OAIY’s own work uses: the Agent (on the model chosen in Engines), Flows’ “Default … model” and the desktop’s engine
          route. It is this computer’s own, or a paired computer’s you choose below. The Engines page’s gateway (port 8080) and its Playground answer
          with this computer’s own model unless a paired one is picked there. “Use this computer’s”, beside a default, makes this computer’s own the
          default again.
        </p>
        {kinds.length === 0 ? (
          <p className="form-hint">{nodes.length === 0 ? 'A paired computer’s models can be chosen here once one is paired (below).' : 'The paired computers have not said which models they have yet.'}</p>
        ) : (
          <div className="engine-defaults">
            {kinds.map((kind) => {
              const d = defaults[kind];
              const node = d && nodes.find((n) => n.nodeId === d.nodeId);
              // (the desktop says whether it is in effect, and why not; an older one does not, and what is known here is said)
              const waiting = notInEffect(d, node?.name ?? '');
              // (in effect, but its computer's engines are not answering: nothing runs in its place)
              const down = defaultDown(d, node?.name ?? '');
              const said = d?.inEffect !== undefined;
              return (
                <div className="engine-default" key={kind} data-testid="engine-default">
                  <span className="network-kind">{KIND[kind] ?? kind}</span>
                  {d ? (
                    <span className="engine-default-which">
                      <Server size={14} aria-hidden />
                      {/* (the model the paired computer runs now: the desktop says it, a worker's language model too) */}
                      <code>{d.model}</code>
                      <span className="network-where">on {d.nodeName || 'a paired computer'}</span>
                      {waiting && (
                        <span className="engine-default-warn" data-testid="engine-default-waiting">
                          <TriangleAlert size={12} aria-hidden /> {waiting}
                        </span>
                      )}
                      {down && (
                        <span className="engine-default-warn" data-testid="engine-default-down">
                          <TriangleAlert size={12} aria-hidden /> {down}
                        </span>
                      )}
                      {/* (in effect, its computer offline: what uses it is told so, and nothing runs in its place) */}
                      {!waiting && !down && node && !node.online && (
                        <span className="engine-default-warn">
                          <TriangleAlert size={12} aria-hidden /> offline: what uses it says so, nothing runs in its place
                        </span>
                      )}
                      {!said && node?.online && kind !== 'llm' && !node.models.some((m) => m.id === d.id) && (
                        <span className="engine-default-warn">
                          <TriangleAlert size={12} aria-hidden /> {d.nodeName || 'it'} no longer has this model
                        </span>
                      )}
                    </span>
                  ) : (
                    <span className="engine-default-which">
                      <Laptop size={14} aria-hidden />
                      {local[kind] ? <code>{local[kind]}</code> : <span className="form-hint">none chosen</span>}
                      <span className="network-where">this computer</span>
                    </span>
                  )}
                  {d && (
                    <button className="btn-tiny" disabled={busy !== null} onClick={() => void choose(kind, null)} title="Make this computer’s own model the default again">
                      {busy === `${kind}:` ? <Loader2 size={12} className="spin" aria-hidden /> : <RotateCcw size={12} aria-hidden />} Use this computer’s
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {nodes.length === 0 ? (
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">Paired computers</h3>
          </div>
          <p className="form-hint">No computer is paired to be used yet: pair one in Setup → Networking, and its models show here.</p>
          <div className="form-actions">
            <button className="btn btn-secondary" onClick={onOpenNetworking}>
              <Network size={14} aria-hidden /> Open Networking
            </button>
          </div>
        </section>
      ) : (
        nodes.map((n) => (
          <section className="model-section" key={n.nodeId} data-testid="engine-worker">
            <div className="section-title-row">
              <h3 className="section-title">
                {n.name} <span className={`badge ${n.online ? 'badge-ok' : 'badge-err'}`}>{n.online ? 'Online' : 'Offline'}</span>
              </h3>
            </div>
            {!n.online && (n.lastError || n.lastSeenMs) && (
              <div className="card-warn">
                <TriangleAlert size={13} aria-hidden /> {offlineLine(n)}
              </div>
            )}
            {(n.devices.length > 0 || n.machine) && (
              <div className="network-devices engine-worker-devices">
                {n.machine && <MachineTiles m={n.machine} live={n.online} />}
                {n.devices.map((d) => (
                  <Device key={d.id} d={d} live={n.online} temperature={n.machine?.gpus?.find((g) => g.name === d.name)?.temperature} />
                ))}
              </div>
            )}
            {n.online && n.jobs && <Queue jobs={n.jobs} />}
            {kinds
              .filter((kind) => n.models.some((m) => m.kind === kind))
              .map((kind) => (
                <div className="engine-worker-kind" key={kind}>
                  <div className="settings-label">{KIND[kind] ?? kind}</div>
                  {n.models
                    .filter((m) => m.kind === kind)
                    .map((m) => {
                      const state = STATE[m.state] ?? { label: m.state, tone: 'idle' as const };
                      const where = deviceName(n, m.device);
                      const chosen = isDefault(n, m);
                      return (
                        <div className={`network-model${chosen ? ' is-default' : ''}`} key={m.id}>
                          <span className="network-model-name">
                            <code>{m.model}</code>
                            {n.online && m.state !== 'not loaded' && <span className={`network-state network-state-${state.tone}`}>{state.label}</span>}
                            {where && n.online && <span className="network-where">on {where}</span>}
                          </span>
                          {chosen ? (
                            <DefaultChip d={defaults[kind]} name={n.name} />
                          ) : (
                            <button
                              className="btn-tiny"
                              disabled={busy !== null || !n.online}
                              title={
                                n.online
                                  ? `Use it for every ${(KIND[kind] ?? kind).toLowerCase()} job that uses the default`
                                  : `${n.name} is offline: it can be made the default once it answers again`
                              }
                              onClick={() => void choose(kind, m.id)}
                            >
                              {busy === `${kind}:${m.id}` ? <Loader2 size={12} className="spin" aria-hidden /> : <Star size={12} aria-hidden />} Make default
                            </button>
                          )}
                        </div>
                      );
                    })}
                </div>
              ))}
          </section>
        ))
      )}
    </div>
  );
}

/** Why a paired computer is offline, in a line: what went wrong when it was last asked, and when it last answered. */
function offlineLine(n: NetworkNode): string {
  const seen = n.lastSeenMs ? `Last heard from ${ago(n.lastSeenMs)}` : null;
  const error = n.lastError?.replace(/\.$/, '');
  return seen && error ? `${error}. ${seen}.` : `${error ?? seen}.`;
}

/** Kinds as the Engines tab lists them. */
const ORDER = ['llm', 'image', 'video', 'speech', 'music', 'sound', 'model3d'];
const kindOrder = (kind: string) => (ORDER.includes(kind) ? ORDER.indexOf(kind) : ORDER.length);
