import { useCallback, useEffect, useState } from 'react';
import { Check, Laptop, Loader2, Network, RotateCcw, Server, Star, TriangleAlert } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { engines, network, type NetworkModel, type NetworkNode, type NetworkStatus } from './api';
import { Device, KIND, STATE, deviceName } from './NetworkPanel';

/**
 * Engines → Other computers: the models of the workers paired over OAIY Network (docs/NETWORK.md), by computer, and
 * which of them is this computer's engine default of its kind in place of its own: what Flows' "Default … model", the
 * engine's chats and the Agent on the model chosen in Engines then use. Choosing a default in the Engines tab makes
 * this computer's own the default again.
 */
export default function EngineWorkersPanel({ onOpenNetworking }: { onOpenNetworking: () => void }) {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [local, setLocal] = useState<Record<string, string | null>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await network.status();
      setStatus(s);
      setError(null);
      if (!s.uses) {
        setNodes([]);
        return;
      }
      const [n, c] = await Promise.all([network.nodes(), engines.catalog().catch(() => null)]);
      setNodes(n.nodes.filter((x) => x.uses ?? x.role === 'worker'));
      setLocal(c?.defaults ?? {});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 5000);

  const choose = async (kind: string, id: string | null) => {
    setBusy(`${kind}:${id ?? ''}`);
    setError(null);
    try {
      setStatus(await network.setDefault(kind, id));
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  if (!status) {
    return <div className="panel">{error ? <div className="banner banner-err">{error}</div> : <div className="empty-state">Loading…</div>}</div>;
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
  const kinds = [...new Set(nodes.flatMap((n) => n.models.map((m) => m.kind)))].sort((a, b) => kindOrder(a) - kindOrder(b));
  /** Whether `m` of `node` is its kind's default here (a worker's language model: whichever it runs now). */
  const isDefault = (node: NetworkNode, m: NetworkModel) => {
    const d = defaults[m.kind];
    return !!d && (m.kind === 'llm' ? d.nodeId === node.nodeId : d.id === m.id);
  };

  return (
    <div className="panel" data-testid="engine-workers">
      {error && (
        <div className="banner banner-err" role="alert">
          {error}
        </div>
      )}
      <section className="model-section">
        <div className="section-title-row">
          <h3 className="section-title">Defaults</h3>
        </div>
        <p className="form-hint">
          The model of each kind that Flows’ “Default … model”, the engine’s chats and the Agent (on the model chosen in Engines) use: this computer’s own,
          or a paired computer’s you choose below. Choosing a default in the Engines tab makes this computer’s own the default again.
        </p>
        {kinds.length === 0 ? (
          <p className="form-hint">{nodes.length === 0 ? 'No worker is paired yet.' : 'The paired computers have not said which models they have yet.'}</p>
        ) : (
          <div className="engine-defaults">
            {kinds.map((kind) => {
              const d = defaults[kind];
              const node = d && nodes.find((n) => n.nodeId === d.nodeId);
              // (a worker's language model: whichever it runs now)
              const shown = d && kind === 'llm' ? (node?.models.find((m) => m.kind === 'llm')?.model ?? d.model) : d?.model;
              return (
                <div className="engine-default" key={kind} data-testid="engine-default">
                  <span className="network-kind">{KIND[kind] ?? kind}</span>
                  {d ? (
                    <span className="engine-default-which">
                      <Server size={14} aria-hidden />
                      <code>{shown}</code>
                      <span className="network-where">on {d.nodeName || 'a paired computer'}</span>
                      {node && !node.online && (
                        <span className="engine-default-warn">
                          <TriangleAlert size={12} aria-hidden /> offline: what uses it says so, nothing runs in its place
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
            <h3 className="section-title">Workers</h3>
          </div>
          <p className="form-hint">No worker is paired yet: pair one in Setup → Networking, and its models show here.</p>
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
            {!n.online && n.lastError && (
              <div className="card-warn">
                <TriangleAlert size={13} aria-hidden /> {n.lastError}
              </div>
            )}
            {n.devices.length > 0 && (
              <div className="network-devices engine-worker-devices">
                {n.devices.map((d) => (
                  <Device key={d.id} d={d} live={n.online} />
                ))}
              </div>
            )}
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
                            <span className="engine-default-chip">
                              <Check size={12} aria-hidden /> Default
                            </span>
                          ) : (
                            <button
                              className="btn-tiny network-pin"
                              disabled={busy !== null}
                              title={`Use it for every ${(KIND[kind] ?? kind).toLowerCase()} job that uses the default`}
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

/** Kinds as the Engines tab lists them. */
const ORDER = ['llm', 'image', 'video', 'speech', 'music', 'sound', 'model3d'];
const kindOrder = (kind: string) => (ORDER.includes(kind) ? ORDER.indexOf(kind) : ORDER.length);
