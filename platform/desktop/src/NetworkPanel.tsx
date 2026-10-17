import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { useVisiblePoll } from './useVisiblePoll';
import {
  network,
  type NetworkFound,
  type NetworkNode,
  type NetworkPairing,
  type NetworkRequest,
  type NetworkRole,
  type NetworkStatus,
} from './api';

/**
 * Setup → Networking: OAIY Network (docs/NETWORK.md). This computer's place on the network (off, a primary that uses
 * other computers' models, or a worker that lends its own), pairing (the same six-digit code on both screens, approved
 * on the worker), and the paired computers with their GPUs and models.
 *
 * Off by default: a computer that never switches it on sees only the switch.
 */
export default function NetworkPanel() {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [requests, setRequests] = useState<NetworkRequest[]>([]);
  const [found, setFound] = useState<NetworkFound[]>([]);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await network.status();
      setStatus(s);
      setError(null);
      if (s.role === 'standalone') {
        setNodes([]);
        setRequests([]);
        setFound([]);
        return;
      }
      const [n, r, f] = await Promise.all([
        network.nodes(),
        s.role === 'worker' ? network.requests() : Promise.resolve({ requests: [] }),
        s.role === 'primary' ? network.discover() : Promise.resolve({ discovery: false, found: [] }),
      ]);
      setNodes(n.nodes);
      setRequests(r.requests);
      setFound(f.found);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 3000);

  const change = async (next: Parameters<typeof network.setSettings>[0]) => {
    setBusy(true);
    setActionError(null);
    try {
      setStatus(await network.setSettings(next));
      await refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!status) {
    return (
      <div className="panel">
        {error ? <div className="banner banner-err">{error}</div> : <div className="empty-state">Loading…</div>}
      </div>
    );
  }

  return (
    <div className="panel" data-testid="network-panel">
      {error && <div className="banner banner-err">{error}</div>}
      {actionError && <div className="banner banner-err" role="alert">{actionError}</div>}
      <ThisComputer status={status} busy={busy} onChange={change} />
      {status.role === 'worker' && (
        <WorkerRequests
          requests={requests}
          onDecide={async (id, approve) => {
            setActionError(null);
            try {
              await (approve ? network.approve(id) : network.deny(id));
              await refresh();
            } catch (e) {
              setActionError(e instanceof Error ? e.message : String(e));
            }
          }}
        />
      )}
      {status.role === 'primary' && <FindWorkers found={found} discovery={status.discovery} onPaired={refresh} />}
      {status.role !== 'standalone' && (
        <PairedComputers
          nodes={nodes}
          role={status.role}
          onUnpair={async (n) => {
            if (!window.confirm(`Unpair ${n.name}? It can no longer ${status.role === 'primary' ? 'lend its models to this computer' : 'use this computer'} until you pair again.`)) return;
            setActionError(null);
            try {
              await network.unpair(n.nodeId);
              await refresh();
            } catch (e) {
              setActionError(e instanceof Error ? e.message : String(e));
            }
          }}
        />
      )}
    </div>
  );
}

const ROLES: { value: NetworkRole; label: string; hint: string }[] = [
  { value: 'primary', label: 'Use other computers (primary)', hint: 'Pair with OAIY computers on this network and use their models here, in Engines, Flows and the Agent.' },
  { value: 'worker', label: 'Lend this computer (worker)', hint: 'A paired primary runs models on this computer’s GPUs. Only computers you approve here can.' },
];

function ThisComputer({ status, busy, onChange }: { status: NetworkStatus; busy: boolean; onChange: (c: Parameters<typeof network.setSettings>[0]) => void }) {
  const [name, setName] = useState(status.settings.name);
  const [port, setPort] = useState(String(status.settings.port));
  const [role, setRole] = useState<NetworkRole>(status.settings.role === 'standalone' ? 'primary' : status.settings.role);
  useEffect(() => {
    setName(status.settings.name);
    setPort(String(status.settings.port));
    if (status.settings.role !== 'standalone') setRole(status.settings.role);
  }, [status.settings.name, status.settings.port, status.settings.role]);

  const save = (e: FormEvent) => {
    e.preventDefault();
    const p = Number(port);
    onChange({ enabled: true, role, name: name.trim(), ...(Number.isInteger(p) && p > 0 ? { port: p } : {}) });
  };

  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          This computer{' '}
          <span className={`badge ${status.enabled ? 'badge-ok' : 'badge-neutral'}`}>
            {status.enabled ? (status.role === 'worker' ? 'Worker' : 'Primary') : 'Off'}
          </span>
        </h3>
      </div>
      <p className="form-hint">
        OAIY Network lets OAIY computers on the same network share their GPUs: a primary uses a worker’s models as if they were its own,
        and nothing leaves your network. It stays off until you switch it on.
      </p>
      <form className="dl-form" onSubmit={save}>
        <div className="form-row-pair">
          <label className="form-row">
            <span>Role</span>
            <select value={role} onChange={(e) => setRole(e.target.value as NetworkRole)} disabled={busy} aria-label="This computer's role">
              {ROLES.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label className="form-row">
            <span>Name other computers see</span>
            <input value={name} placeholder={status.name} onChange={(e) => setName(e.target.value)} disabled={busy} maxLength={80} />
          </label>
        </div>
        {role === 'worker' && (
          <label className="form-row form-row-inline">
            <span>Port</span>
            <input value={port} onChange={(e) => setPort(e.target.value.replace(/[^0-9]/g, ''))} disabled={busy} inputMode="numeric" style={{ width: '7em' }} />
          </label>
        )}
        <span className="form-hint">{ROLES.find((r) => r.value === role)?.hint}</span>
        <div className="form-actions">
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {status.enabled ? 'Save' : 'Switch on'}
          </button>
          {status.enabled && (
            <button className="btn btn-secondary" type="button" disabled={busy} onClick={() => onChange({ enabled: false })}>
              Switch off
            </button>
          )}
        </div>
      </form>
      {status.enabled && (
        <div className="datadir-note" data-testid="network-identity">
          <div>
            Shown to other computers as <strong>{status.name}</strong>
            {status.fingerprint && (
              <>
                {' '}
                · certificate <code className="path-code path-code-small">{status.fingerprint}</code>
              </>
            )}
          </div>
          {status.role === 'worker' && (
            <div>
              {status.listening ? (
                <>
                  {status.discovery ? 'A primary finds this computer on the network' : 'A primary adds this computer by its address'}
                  {status.addresses.length > 0 && (
                    <>
                      {status.discovery ? ', or adds it at ' : ': '}
                      {status.addresses.map((a, i) => (
                        <span key={a}>
                          {i > 0 && ' or '}
                          <code className="path-code path-code-small">{a}</code>
                        </span>
                      ))}
                    </>
                  )}
                  .
                </>
              ) : (
                'Not listening: switch it on again, or choose another port.'
              )}
            </div>
          )}
          {!status.discovery && <div>Finding computers on the network (mDNS) is not available here: add them by their address.</div>}
        </div>
      )}
    </section>
  );
}

function WorkerRequests({ requests, onDecide }: { requests: NetworkRequest[]; onDecide: (id: string, approve: boolean) => void }) {
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          Pairing requests <span className="section-count">({requests.length})</span>
        </h3>
      </div>
      {requests.length === 0 ? (
        <p className="form-hint">When a primary asks to use this computer, it shows here. Approve it only if its code matches the one on the primary’s screen.</p>
      ) : (
        <div className="pairing-list">
          {requests.map((r) => (
            <div className="pairing-card" key={r.id} data-testid="network-request">
              <div className="pairing-head">
                <strong className="pairing-product">{r.name}</strong>
                <span className="pairing-origin">from {r.address}</span>
              </div>
              <div className="pairing-body">
                Code <span className="pairing-code">{r.code}</span>
                <div className="form-hint">Certificate {r.fingerprint}. The request expires in {Math.ceil(r.secondsLeft / 60)} min.</div>
              </div>
              <div className="pairing-actions">
                <button className="btn btn-primary" onClick={() => onDecide(r.id, true)}>
                  Approve
                </button>
                <button className="btn btn-secondary" onClick={() => onDecide(r.id, false)}>
                  Deny
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function FindWorkers({ found, discovery, onPaired }: { found: NetworkFound[]; discovery: boolean; onPaired: () => void }) {
  const [address, setAddress] = useState('');
  const [pairing, setPairing] = useState<NetworkPairing | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  // While a pairing waits for the worker's approval, ask every two seconds.
  useEffect(() => {
    if (!pairing) return;
    let stopped = false;
    const tick = async () => {
      try {
        const r = await network.confirmPairing(pairing.pairingId);
        if (stopped) return;
        if (r.status === 'pending') {
          timer.current = window.setTimeout(tick, 2000);
          return;
        }
        setPairing(null);
        setOutcome(
          r.status === 'approved'
            ? `Paired with ${r.node?.name ?? pairing.worker.name}.`
            : r.status === 'denied'
              ? `${pairing.worker.name} declined the pairing.`
              : 'The pairing request expired: start again.',
        );
        onPaired();
      } catch (e) {
        if (!stopped) {
          setError(e instanceof Error ? e.message : String(e));
          setPairing(null);
        }
      }
    };
    timer.current = window.setTimeout(tick, 1500);
    return () => {
      stopped = true;
      window.clearTimeout(timer.current);
    };
  }, [pairing, onPaired]);

  const start = async (target: { address?: string; nodeId?: string }) => {
    setStarting(true);
    setError(null);
    setOutcome(null);
    try {
      setPairing(await network.startPairing(target));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const unpaired = found.filter((f) => !f.paired);
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          Add a worker <span className="section-count">({unpaired.length} found)</span>
        </h3>
      </div>
      {error && <div className="banner banner-err" role="alert">{error}</div>}
      {outcome && <div className="banner banner-pending">{outcome}</div>}
      {pairing && (
        <div className="pairing-card" data-testid="network-pairing">
          <div className="pairing-head">
            <strong className="pairing-product">{pairing.worker.name}</strong>
            <span className="pairing-origin">{pairing.worker.address}</span>
          </div>
          <div className="pairing-body">
            Code <span className="pairing-code">{pairing.code}</span>
            <div className="form-hint">
              On {pairing.worker.name}, open Setup → Networking and approve the request if it shows this same code. Waiting…
            </div>
          </div>
          <div className="pairing-actions">
            <button className="btn btn-secondary" onClick={() => setPairing(null)}>
              Stop waiting
            </button>
          </div>
        </div>
      )}
      {discovery ? (
        unpaired.length === 0 ? (
          <p className="form-hint">Looking for OAIY workers on this network… A computer shows here once it is switched on as a worker.</p>
        ) : (
          <div className="pairing-list">
            {unpaired.map((f) => (
              <div className="service-card" key={f.nodeId} data-testid="network-found">
                <div className="card-head">
                  <span className="card-title">{f.name || 'An OAIY worker'}</span>
                  <span className="card-meta">{f.addresses[0]}</span>
                </div>
                {!f.compatible && <div className="card-warn">A different version of OAIY Network: update both computers to pair them.</div>}
                <div className="card-actions">
                  <button className="btn btn-primary btn-tiny" disabled={starting || !!pairing || !f.compatible} onClick={() => start({ nodeId: f.nodeId })}>
                    Pair
                  </button>
                </div>
              </div>
            ))}
          </div>
        )
      ) : null}
      <form
        className="dl-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (address.trim()) void start({ address: address.trim() });
        }}
      >
        <label className="form-row form-row-inline">
          <span>Add by address</span>
          <input value={address} onChange={(e) => setAddress(e.target.value)} placeholder="192.168.1.20:17975" disabled={starting || !!pairing} aria-label="Worker address" />
          <button className="btn btn-secondary" type="submit" disabled={starting || !!pairing || !address.trim()}>
            Pair
          </button>
        </label>
        <span className="form-hint">The address a worker shows on its own Networking page, for networks where computers can’t find each other.</span>
      </form>
    </section>
  );
}

/** Where a worker's model is loaded, in this computer's words: the worker's own GPU by its name. */
function deviceOf(n: NetworkNode, device: string | null): string {
  if (!device) return '';
  if (device === 'this computer') return ` on ${n.name}’s own GPU`;
  return ` on ${n.devices.find((d) => d.id === device)?.name ?? device}`;
}

/** What can be pinned on a worker: its picture models, and its language model once it is the one chosen there. */
const pinnable = (m: NetworkNode['models'][number]) => m.kind === 'image' || (m.kind === 'llm' && m.state !== 'not loaded');

const PIN_STATE: Record<string, string> = { warming: 'pinned · loading', ready: 'pinned', failed: 'pinned · could not load', waiting: 'pinned · waiting to load' };

/** A worker's models: the pinned and loaded ones, then the rest it can load, folded away. Each can be pinned there. */
function NodeModels({ node }: { node: NetworkNode }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loaded = node.models.filter((m) => m.state !== 'not loaded' || m.pinned);
  const rest = node.models.filter((m) => m.state === 'not loaded' && !m.pinned);
  const toggle = async (m: NetworkNode['models'][number]) => {
    setBusy(m.id);
    setError(null);
    try {
      await (m.pinned ? network.unpin(node.nodeId, m.model) : network.pin(node.nodeId, m.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const row = (m: NetworkNode['models'][number]) => (
    <div key={m.id} className="network-model">
      <span>
        {m.kind} · <code className="path-code path-code-small">{m.model}</code> · {m.state}
        {deviceOf(node, m.device)}
        {m.pinned && <strong> · {PIN_STATE[m.pinState ?? 'ready'] ?? 'pinned'}</strong>}
        {m.pinError && <span className="form-hint"> ({m.pinError})</span>}
      </span>
      {node.online && pinnable(m) && (
        <button
          className="btn btn-ghost btn-tiny"
          disabled={busy !== null}
          title={m.pinned ? 'Let it go as any other model again' : 'Keep it loaded on this worker, and load it again when the worker restarts'}
          onClick={() => void toggle(m)}
        >
          {busy === m.id ? '…' : m.pinned ? 'Unpin' : 'Pin'}
        </button>
      )}
    </div>
  );
  return (
    <div className="card-desc">
      {error && <div className="card-warn" role="alert">{error}</div>}
      {loaded.length > 0 ? loaded.map(row) : <div>No model is loaded on it now.</div>}
      {rest.length > 0 && (
        <details>
          <summary>{rest.length} more it can load</summary>
          {rest.map(row)}
        </details>
      )}
    </div>
  );
}

const memory = (mb?: number | null) => (typeof mb === 'number' && mb > 0 ? `${(mb / 1024).toFixed(1)} GB` : null);

function PairedComputers({ nodes, role, onUnpair }: { nodes: NetworkNode[]; role: NetworkRole; onUnpair: (n: NetworkNode) => void }) {
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          {role === 'primary' ? 'Workers' : 'Paired primary'} <span className="section-count">({nodes.length})</span>
        </h3>
      </div>
      {nodes.length === 0 ? (
        <p className="form-hint">{role === 'primary' ? 'No worker is paired yet.' : 'No primary uses this computer yet.'}</p>
      ) : (
        <div className="pairing-list">
          {nodes.map((n) => (
            <div className="service-card" key={n.nodeId} data-testid="network-node">
              <div className="card-head">
                <span className="card-title">{n.name}</span>
                {n.role === 'worker' && (
                  <span className={`badge ${n.online ? 'badge-ok' : 'badge-err'}`}>{n.online ? 'Online' : 'Offline'}</span>
                )}
                <span className="card-meta">
                  {n.address} · certificate {n.fingerprint}
                  {n.node?.os ? ` · ${n.node.os}` : ''}
                </span>
              </div>
              {n.role === 'worker' && !n.online && n.lastError && <div className="card-warn">{n.lastError}</div>}
              {n.enginesError && <div className="card-warn">Its engines: {n.enginesError}</div>}
              {n.devices.length > 0 && (
                <div className="card-desc">
                  {n.devices.map((d) => {
                    const total = memory(d.memory_total_mb);
                    const used = memory(d.memory_used_mb);
                    return (
                      <div key={d.id}>
                        {d.name}: {total ? `${used ?? '?'} of ${total} in use` : 'memory unknown'}
                      </div>
                    );
                  })}
                </div>
              )}
              {n.models.length > 0 && <NodeModels node={n} />}
              <div className="card-actions">
                <button className="btn btn-danger btn-tiny" onClick={() => onUnpair(n)}>
                  Unpair
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      {role === 'primary' && nodes.length > 0 && (
        <p className="form-hint">A worker’s models are listed in Engines and in Flows’ model menus, named with the worker, and run on the worker.</p>
      )}
    </section>
  );
}
