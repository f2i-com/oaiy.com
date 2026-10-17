import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { Check, Copy, Laptop, Link2, Loader2, Pin, PinOff, Server, TriangleAlert, Unplug, X, type LucideIcon } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import {
  network,
  type NetworkDevice,
  type NetworkFound,
  type NetworkModel,
  type NetworkNode,
  type NetworkPairing,
  type NetworkRequest,
  type NetworkRole,
  type NetworkStatus,
} from './api';

/**
 * Setup → Networking: OAIY Network (docs/NETWORK.md). This computer's place on the network (off, a primary that uses
 * other computers' models, or a worker that lends its own), pairing (the same six-digit code on both screens, approved
 * on both), and the paired computers with their GPUs and models.
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
          onChanged={refresh}
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

const ROLES: { value: Exclude<NetworkRole, 'standalone'>; label: string; tag: string; icon: LucideIcon; hint: string }[] = [
  {
    value: 'primary',
    label: 'Use other computers',
    tag: 'Primary',
    icon: Laptop,
    hint: 'Pair with OAIY computers on this network and use their models here, in Engines, Flows and the Agent.',
  },
  {
    value: 'worker',
    label: 'Lend this computer',
    tag: 'Worker',
    icon: Server,
    hint: 'A paired primary runs models on this computer’s GPUs. Only computers you approve here can.',
  },
];

/** A value to copy (an address to type on another computer), with its button. */
function Copyable({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // (no clipboard in a webview without focus: the address stays there to read)
    }
  };
  return (
    <span className="network-copyable">
      <code>{value}</code>
      <button type="button" className="network-copy" onClick={() => void copy()} aria-label={`Copy ${value}`} title={copied ? 'Copied' : 'Copy'}>
        {copied ? <Check size={12} /> : <Copy size={12} />}
      </button>
    </span>
  );
}

function ThisComputer({ status, busy, onChange }: { status: NetworkStatus; busy: boolean; onChange: (c: Parameters<typeof network.setSettings>[0]) => void }) {
  const [name, setName] = useState(status.settings.name);
  const [port, setPort] = useState(String(status.settings.port));
  const [role, setRole] = useState<NetworkRole>(status.settings.role === 'standalone' ? 'primary' : status.settings.role);
  useEffect(() => {
    setName(status.settings.name);
    setPort(String(status.settings.port));
    if (status.settings.role !== 'standalone') setRole(status.settings.role);
  }, [status.settings.name, status.settings.port, status.settings.role]);

  const portNumber = Number(port);
  const portOk = role !== 'worker' || (Number.isInteger(portNumber) && portNumber >= 1024 && portNumber <= 65535);
  const changed = !status.enabled || role !== status.settings.role || name.trim() !== status.settings.name || (role === 'worker' && portNumber !== status.settings.port);

  const save = (e: FormEvent) => {
    e.preventDefault();
    if (!portOk) return;
    onChange({ enabled: true, role, name: name.trim(), ...(role === 'worker' ? { port: portNumber } : {}) });
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
        <div className="network-roles" role="radiogroup" aria-label="This computer's role">
          {ROLES.map((r) => {
            const Icon = r.icon;
            const chosen = role === r.value;
            return (
              <button
                key={r.value}
                type="button"
                role="radio"
                aria-checked={chosen}
                className={`network-role${chosen ? ' is-chosen' : ''}`}
                disabled={busy}
                onClick={() => setRole(r.value)}
              >
                <span className="network-role-icon">
                  <Icon size={17} aria-hidden />
                </span>
                <span className="network-role-text">
                  <strong>
                    {r.label} <span className="network-role-tag">{r.tag}</span>
                  </strong>
                  <small>{r.hint}</small>
                </span>
              </button>
            );
          })}
        </div>
        <div className="form-row-pair">
          <label className="form-row">
            <span>Name other computers see</span>
            <input value={name} placeholder={status.name} onChange={(e) => setName(e.target.value)} disabled={busy} maxLength={80} />
          </label>
          {role === 'worker' && (
            <label className="form-row network-port">
              <span>Port</span>
              <input
                value={port}
                onChange={(e) => setPort(e.target.value.replace(/[^0-9]/g, '').slice(0, 5))}
                disabled={busy}
                inputMode="numeric"
                aria-invalid={!portOk}
              />
            </label>
          )}
        </div>
        {!portOk && <span className="form-hint network-invalid">A port from 1024 to 65535.</span>}
        <div className="form-actions">
          <button className="btn btn-primary" type="submit" disabled={busy || !portOk || !changed}>
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
        <div className="datadir-note network-identity" data-testid="network-identity">
          <div>
            Shown to other computers as <strong>{status.name}</strong>
            {status.fingerprint && (
              <>
                {' '}
                · certificate <code className="path-code path-code-small">{status.fingerprint}</code>
              </>
            )}
          </div>
          {status.role === 'worker' &&
            (status.listening ? (
              <div className="network-addresses">
                <span>{status.discovery ? 'Primaries on this network find it, or add it at' : 'A primary adds it at'}</span>
                {status.addresses.map((a) => (
                  <Copyable key={a} value={a} />
                ))}
              </div>
            ) : (
              <div className="network-invalid">Not listening: switch it on again, or choose another port.</div>
            ))}
          {!status.discovery && <div>Finding computers on the network (mDNS) is off here: add them by their address.</div>}
        </div>
      )}
    </section>
  );
}

/** The six digits both screens show, large enough to compare at a glance. */
function PairingCode({ code }: { code: string }) {
  return (
    <div className="network-code" aria-label={`Pairing code ${code}`}>
      {code}
    </div>
  );
}

function WorkerRequests({ requests, onDecide }: { requests: NetworkRequest[]; onDecide: (id: string, approve: boolean) => Promise<void> }) {
  const [deciding, setDeciding] = useState<string | null>(null);
  const decide = async (id: string, approve: boolean) => {
    setDeciding(id);
    try {
      await onDecide(id, approve);
    } finally {
      setDeciding(null);
    }
  };
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
        <div className="network-list">
          {requests.map((r) => (
            <div className="pairing-card pairing-pending network-pairing" key={r.id} data-testid="network-request" role="alert">
              <div className="pairing-head">
                <Link2 size={16} aria-hidden />
                <strong>{r.name} asks to use this computer</strong>
                <span className="pairing-origin">{r.address}</span>
              </div>
              <PairingCode code={r.code} />
              <div className="form-hint">
                Approve only if {r.name}’s screen shows this same code. Certificate {r.fingerprint} · expires in {Math.max(1, Math.ceil(r.secondsLeft / 60))} min.
              </div>
              <div className="pairing-actions">
                <button className="btn btn-primary" disabled={deciding !== null} onClick={() => void decide(r.id, true)}>
                  <Check size={14} aria-hidden /> Approve
                </button>
                <button className="btn btn-ghost" disabled={deciding !== null} onClick={() => void decide(r.id, false)}>
                  <X size={14} aria-hidden /> Deny
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
  /** The worker approved: this side's person says whether it showed the same code. */
  const [approvedThere, setApprovedThere] = useState(false);
  const [outcome, setOutcome] = useState<{ text: string; paired: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  // While a pairing waits for the worker's approval, ask every two seconds.
  useEffect(() => {
    if (!pairing || approvedThere) return;
    let stopped = false;
    const tick = async () => {
      try {
        const r = await network.confirmPairing(pairing.pairingId);
        if (stopped) return;
        if (r.status === 'pending') {
          timer.current = window.setTimeout(tick, 2000);
          return;
        }
        if (r.status === 'approved_there') {
          setApprovedThere(true);
          return;
        }
        setPairing(null);
        setOutcome(
          r.status === 'approved'
            ? { text: `Paired with ${r.node?.name ?? pairing.worker.name}.`, paired: true }
            : r.status === 'denied'
              ? { text: `${pairing.worker.name} declined the pairing.`, paired: false }
              : { text: 'The pairing request expired: start again.', paired: false },
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
  }, [pairing, approvedThere, onPaired]);

  /**
   * This side's answer once the worker approved: the codes matched (paired), or they did not (nothing is). A `false`
   * before then withdraws the request, so the worker no longer shows it.
   */
  const answer = async (accept: boolean) => {
    if (!pairing) return;
    const waiting = !approvedThere;
    setError(null);
    setPairing(null);
    setApprovedThere(false);
    try {
      const r = await network.confirmPairing(pairing.pairingId, accept);
      if (r.status === 'approved') setAddress('');
      setOutcome(
        r.status === 'approved'
          ? { text: `Paired with ${r.node?.name ?? pairing.worker.name}.`, paired: true }
          : waiting
            ? null
            : { text: 'Not paired: the codes did not match.', paired: false },
      );
      onPaired();
    } catch (e) {
      // (stopping a request that has already expired needs no word)
      if (!waiting) setError(e instanceof Error ? e.message : String(e));
    }
  };

  const start = async (target: { address?: string; nodeId?: string }) => {
    setStarting(true);
    setError(null);
    setOutcome(null);
    setApprovedThere(false);
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
          Add a worker {discovery && <span className="section-count">({unpaired.length} found)</span>}
        </h3>
      </div>
      {error && <div className="banner banner-err" role="alert">{error}</div>}
      {outcome && <div className={`banner ${outcome.paired ? 'banner-ok' : 'banner-pending'}`}>{outcome.text}</div>}
      {pairing && (
        <div className="pairing-card pairing-pending network-pairing" data-testid="network-pairing">
          <div className="pairing-head">
            <Server size={16} aria-hidden />
            <strong>{pairing.worker.name}</strong>
            <span className="pairing-origin">{pairing.worker.address}</span>
          </div>
          <PairingCode code={pairing.code} />
          <div className="network-pairing-step">
            {approvedThere ? (
              <span>
                <strong>{pairing.worker.name} approved.</strong> Did its screen show this same code? Pair only if it did.
              </span>
            ) : (
              <>
                <Loader2 size={14} className="spin" aria-hidden />
                <span>On {pairing.worker.name}, open Setup → Networking and approve the request if it shows this same code.</span>
              </>
            )}
          </div>
          <div className="pairing-actions">
            {approvedThere ? (
              <>
                <button className="btn btn-primary" onClick={() => void answer(true)}>
                  <Check size={14} aria-hidden /> Yes, pair
                </button>
                <button className="btn btn-ghost" onClick={() => void answer(false)}>
                  <X size={14} aria-hidden /> No
                </button>
              </>
            ) : (
              <button className="btn btn-ghost" onClick={() => void answer(false)}>
                Stop waiting
              </button>
            )}
          </div>
        </div>
      )}
      {discovery &&
        (unpaired.length === 0 ? (
          <p className="form-hint network-looking">
            <Loader2 size={13} className="spin" aria-hidden /> Looking for OAIY workers on this network… A computer shows here once it is switched on as a worker.
          </p>
        ) : (
          <div className="network-list">
            {unpaired.map((f) => (
              <div className="network-found" key={f.nodeId} data-testid="network-found">
                <Server size={16} aria-hidden />
                <span className="network-found-name">
                  <strong>{f.name || 'An OAIY worker'}</strong>
                  {f.addresses[0] && <span className="pairing-origin">{f.addresses[0]}</span>}
                  {!f.compatible && <span className="card-warn">A different version of OAIY Network: update both computers to pair them.</span>}
                </span>
                <button className="btn btn-primary btn-tiny" disabled={starting || !!pairing || !f.compatible} onClick={() => void start({ nodeId: f.nodeId })}>
                  Pair
                </button>
              </div>
            ))}
          </div>
        ))}
      <form
        className="network-add"
        onSubmit={(e) => {
          e.preventDefault();
          if (address.trim()) void start({ address: address.trim() });
        }}
      >
        <label className="form-row">
          <span>{discovery ? 'Or add one by its address' : 'Add one by its address'}</span>
          <input value={address} onChange={(e) => setAddress(e.target.value)} placeholder="192.168.1.20:17975" disabled={starting || !!pairing} aria-label="Worker address" />
        </label>
        <button className="btn btn-secondary" type="submit" disabled={starting || !!pairing || !address.trim()}>
          {starting ? <Loader2 size={14} className="spin" aria-hidden /> : <Link2 size={14} aria-hidden />} Pair
        </button>
      </form>
      <span className="form-hint">The address a worker shows on its own Networking page, for networks where computers can’t find each other.</span>
    </section>
  );
}

/** A worker's own GPU (`this computer`, where its language model runs) is the one that shares its memory. */
function deviceName(n: NetworkNode, device: string | null): string | null {
  if (!device) return null;
  const named = n.devices.find((d) => d.id === device) ?? (device === 'this computer' ? n.devices.find((d) => d.unified) : undefined);
  return named?.name ?? (device === 'this computer' ? `${n.name}’s own GPU` : device);
}

/** What can be pinned on a worker: its picture models, and its language model once it is the one chosen there (one
 * pinned can be unpinned, loaded or not). */
const pinnable = (m: NetworkModel) => m.kind === 'image' || (m.kind === 'llm' && (m.state !== 'not loaded' || m.pinned));

const KIND: Record<string, string> = { llm: 'Text', image: 'Picture', video: 'Video', speech: 'Speech', music: 'Music', sound: 'Sound', vision: 'Vision', embedding: 'Embedding' };
const STATE: Record<string, { label: string; tone: 'ok' | 'busy' | 'idle' | 'off' }> = {
  ready: { label: 'Ready', tone: 'ok' },
  kept: { label: 'Loaded', tone: 'ok' },
  in_use: { label: 'Working', tone: 'busy' },
  starting: { label: 'Starting', tone: 'busy' },
  loading: { label: 'Loading', tone: 'busy' },
  lent: { label: 'Lent to pictures', tone: 'idle' },
  'not loaded': { label: 'Not loaded', tone: 'off' },
};
const PIN_STATE: Record<string, string> = { warming: 'Pinned · loading', ready: 'Pinned', failed: 'Pinned · could not load', waiting: 'Pinned · waiting to load' };

/** A worker's models: the pinned and loaded ones, then the rest it can load, folded away. Each can be pinned there. */
function NodeModels({ node, onChanged }: { node: NetworkNode; onChanged: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loaded = node.models.filter((m) => m.state !== 'not loaded' || m.pinned);
  const rest = node.models.filter((m) => m.state === 'not loaded' && !m.pinned);
  const toggle = async (m: NetworkModel) => {
    setBusy(m.id);
    setError(null);
    try {
      await (m.pinned ? network.unpin(node.nodeId, m.model) : network.pin(node.nodeId, m.id));
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const row = (m: NetworkModel) => {
    const state = STATE[m.state] ?? { label: m.state, tone: 'idle' as const };
    const where = deviceName(node, m.device);
    const pinState = m.pinState ?? 'ready';
    return (
      <div key={m.id} className="network-model">
        <span className="network-kind">{KIND[m.kind] ?? m.kind}</span>
        <span className="network-model-name">
          <code>{m.model}</code>
          {node.online && (m.state !== 'not loaded' || m.pinned) && <span className={`network-state network-state-${state.tone}`}>{state.label}</span>}
          {where && <span className="network-where">on {where}</span>}
          {m.pinned && (
            <span className={`network-pinned${pinState === 'failed' ? ' is-failed' : ''}`}>
              <Pin size={11} aria-hidden /> {PIN_STATE[pinState] ?? 'Pinned'}
            </span>
          )}
          {m.pinError && <span className="network-pin-error">{m.pinError}</span>}
        </span>
        {node.online && pinnable(m) && (
          <button
            className={`btn-tiny network-pin${m.pinned ? ' is-pinned' : ''}`}
            disabled={busy !== null}
            title={m.pinned ? 'Let it go as any other model again' : 'Keep it loaded on this worker, and load it again when the worker restarts'}
            onClick={() => void toggle(m)}
          >
            {busy === m.id ? <Loader2 size={12} className="spin" aria-hidden /> : m.pinned ? <PinOff size={12} aria-hidden /> : <Pin size={12} aria-hidden />}
            {m.pinned ? 'Unpin' : 'Pin'}
          </button>
        )}
      </div>
    );
  };
  return (
    <div className="network-models">
      <div className="settings-label">Models</div>
      {error && (
        <div className="card-warn" role="alert">
          <TriangleAlert size={13} aria-hidden /> {error}
        </div>
      )}
      {loaded.length > 0 ? loaded.map(row) : <div className="form-hint">No model is loaded on it now.</div>}
      {rest.length > 0 && (
        <details className="network-more">
          <summary>{rest.length} more it can load</summary>
          {rest.map(row)}
        </details>
      )}
    </div>
  );
}

const gb = (mb: number) => `${(mb / 1024).toFixed(1)}`;

/** A GPU of a worker: its memory in use as a meter, and how busy it is (not while the worker does not answer). */
function Device({ d, live }: { d: NetworkDevice; live: boolean }) {
  const total = typeof d.memory_total_mb === 'number' && d.memory_total_mb > 0 ? d.memory_total_mb : null;
  const used = typeof d.memory_used_mb === 'number' ? d.memory_used_mb : null;
  const share = total && used !== null ? Math.min(100, Math.max(0, (used / total) * 100)) : null;
  return (
    <div className="network-device">
      <div className="network-device-head">
        <span className="network-device-name">{d.name}</span>
        {live && typeof d.load === 'number' && <span className="network-device-load">{Math.round(d.load)}% busy</span>}
      </div>
      <div className="network-meter" aria-hidden>
        <div className={`network-meter-fill${share !== null && share > 85 ? ' is-full' : ''}`} style={{ width: `${share ?? 0}%` }} />
      </div>
      <div className="network-device-figure">{total ? `${used !== null ? gb(used) : '?'} of ${gb(total)} GB in use` : 'memory unknown'}</div>
    </div>
  );
}

/** How long ago, in a person's words. */
function ago(ms: number): string {
  const minutes = Math.round((Date.now() - ms) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)} h ago`;
  return new Date(ms).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
}

const OS: Record<string, string> = { macos: 'macOS', windows: 'Windows', linux: 'Linux' };
const ARCH: Record<string, string> = { aarch64: 'arm64', x86_64: 'x64' };

/** What a worker is doing now: the job it runs and the ones waiting. */
function queueText(q: NetworkNode['queue']): string | null {
  if (!q) return null;
  const running = q.running && typeof q.running === 'object' ? q.running : null;
  const what = running ? `Working on ${running.kind ? `a ${KIND[running.kind]?.toLowerCase() ?? running.kind} job` : 'a job'}${running.model ? ` (${running.model})` : ''}` : null;
  const waiting = q.queued > 0 ? `${q.queued} waiting` : null;
  return [what, waiting].filter(Boolean).join(' · ') || null;
}

function PairedComputers({
  nodes,
  role,
  onUnpair,
  onChanged,
}: {
  nodes: NetworkNode[];
  role: NetworkRole;
  onUnpair: (n: NetworkNode) => void;
  onChanged: () => void;
}) {
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          {role === 'primary' ? 'Workers' : 'Paired primary'} <span className="section-count">({nodes.length})</span>
        </h3>
      </div>
      {nodes.length === 0 ? (
        <p className="form-hint">{role === 'primary' ? 'No worker is paired yet: add one above.' : 'No primary uses this computer yet.'}</p>
      ) : (
        <div className="network-list">
          {nodes.map((n) => {
            const worker = n.role === 'worker';
            const Icon = worker ? Server : Laptop;
            const busyNow = worker && n.online ? queueText(n.queue) : null;
            // (a worker that does not answer: what it last said, dimmed, its load and models' states left out)
            const stale = worker && !n.online;
            const about = [
              n.address,
              n.node?.os ? `${OS[n.node.os] ?? n.node.os}${n.node.arch ? ` ${ARCH[n.node.arch] ?? n.node.arch}` : ''}` : null,
              n.node?.appVersion ? `OAIY ${n.node.appVersion}` : null,
              `certificate ${n.fingerprint}`,
              n.pairedAtMs ? `paired ${new Date(n.pairedAtMs).toLocaleDateString()}` : null,
            ].filter(Boolean);
            return (
              <div className={`service-card network-node${worker ? (n.online ? ' service-card-running' : ' service-card-errored') : ''}`} key={n.nodeId} data-testid="network-node">
                <div className="card-head">
                  <Icon size={16} aria-hidden />
                  <span className="card-title">{n.name}</span>
                  {worker && <span className={`badge ${n.online ? 'badge-ok' : 'badge-err'}`}>{n.online ? 'Online' : 'Offline'}</span>}
                  <button className="btn-tiny btn-danger network-unpair" onClick={() => onUnpair(n)}>
                    <Unplug size={12} aria-hidden /> Unpair
                  </button>
                </div>
                <div className="card-meta network-about">{about.join(' · ')}</div>
                {worker && !n.online && n.lastError && (
                  <div className="card-warn">
                    <TriangleAlert size={13} aria-hidden /> {n.lastError}
                  </div>
                )}
                {n.enginesError && (
                  <div className="card-warn">
                    <TriangleAlert size={13} aria-hidden /> Its engines: {n.enginesError}
                  </div>
                )}
                {busyNow && <div className="network-busy">{busyNow}</div>}
                <div className={stale ? 'network-stale' : undefined}>
                  {stale && n.lastSeenMs && <div className="network-stale-note">Last report {ago(n.lastSeenMs)}</div>}
                  {n.devices.length > 0 && (
                    <div className="network-devices">
                      {n.devices.map((d) => (
                        <Device key={d.id} d={d} live={!stale} />
                      ))}
                    </div>
                  )}
                  {n.models.length > 0 && <NodeModels node={n} onChanged={onChanged} />}
                </div>
              </div>
            );
          })}
        </div>
      )}
      {role === 'primary' && nodes.length > 0 && (
        <p className="form-hint">A worker’s models are listed in Engines and in Flows’ model menus, named with the worker, and run on the worker.</p>
      )}
    </section>
  );
}
