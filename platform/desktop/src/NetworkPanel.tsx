import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { AudioLines, Bot, Check, Copy, Unlock, Laptop, Link2, Loader2, Pin, PinOff, Server, TriangleAlert, Unplug, Workflow, X, type LucideIcon } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import {
  network,
  type NetworkDevice,
  type NetworkJob,
  type NetworkMachine,
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
export default function NetworkPanel({ onNavigate }: { onNavigate?: (view: 'flow-workers' | 'services' | 'engine-workers') => void } = {}) {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [requests, setRequests] = useState<NetworkRequest[]>([]);
  const [found, setFound] = useState<NetworkFound[]>([]);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  // The "paired by itself" notices the person has seen, remembered on this computer.
  const [seenByItself, setSeenByItself] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('oaiy.network.byItselfSeen') ?? '[]') as string[];
    } catch {
      return [];
    }
  });
  const dismissByItself = (key: string) =>
    setSeenByItself((all) => {
      const next = [...all, key].slice(-20);
      try {
        localStorage.setItem('oaiy.network.byItselfSeen', JSON.stringify(next));
      } catch {
        /* storage can be unavailable: shown again next time */
      }
      return next;
    });

  // (each read numbered: one answered after a newer one began, or after a change was saved, is not shown)
  const reads = useRef(0);
  const refresh = useCallback(async () => {
    const seq = ++reads.current;
    try {
      const s = await network.status();
      if (seq !== reads.current) return;
      setStatus(s);
      setError(null);
      if (!s.uses && !s.lends) {
        setNodes([]);
        setRequests([]);
        setFound([]);
        return;
      }
      const [n, r, f] = await Promise.all([
        network.nodes(),
        s.lends ? network.requests() : Promise.resolve({ requests: [] }),
        s.uses ? network.discover() : Promise.resolve({ discovery: false, found: [] }),
      ]);
      if (seq !== reads.current) return;
      setNodes(n.nodes);
      setRequests(r.requests);
      setFound(f.found);
    } catch (e) {
      if (seq !== reads.current) return;
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
    // (a read under way began before the change: what it answers is the settings before, not shown)
    reads.current++;
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
      {status.listenError && (
        <div className="banner banner-err" role="alert">
          This computer cannot lend itself right now: {status.listenError}
        </div>
      )}
      {(status.pairedByItself ?? [])
        .filter((p) => !seenByItself.includes(`${p.name}:${p.at}`))
        .map((p) => (
          <div className="banner banner-pending banner-dismissable network-by-itself" key={`${p.name}:${p.at}`} role="status">
            <span>
              <Bot size={13} aria-hidden /> <strong>{p.name}</strong> paired with this computer by itself at{' '}
              {new Date(p.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}. Unpair it below if you do not know it.
            </span>
            <button className="banner-dismiss" aria-label={`Dismiss: ${p.name} is known`} title="Dismiss" onClick={() => dismissByItself(`${p.name}:${p.at}`)}>
              <X size={13} aria-hidden />
            </button>
          </div>
        ))}
      <AgentMaySetUp status={status} busy={busy} onChange={change} />
      {status.lends && (
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
      {status.uses && <FindWorkers found={found} discovery={status.discovery} inProgress={status.pairings} lends={status.lends} onPaired={refresh} />}
      {(status.uses || status.lends) && (
        <PairedComputers
          nodes={nodes}
          uses={status.uses}
          lends={status.lends}
          defaults={status.defaults}
          services={status.services}
          onNavigate={onNavigate}
          onChanged={refresh}
          onUnpair={async (n) => {
            if (!window.confirm(`Unpair ${n.name}? Neither can use the other until you pair them again.`)) return;
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

/** What this computer does on the network: either, or both. */
const WAYS: { key: 'uses' | 'lends'; label: string; icon: LucideIcon; hint: string }[] = [
  {
    key: 'uses',
    label: 'Use other computers',
    icon: Laptop,
    hint: 'Pair with OAIY computers on this network and use their models, the flows they share and their OAIY Voice from here.',
  },
  {
    key: 'lends',
    label: 'Lend this computer',
    icon: Server,
    hint: 'Computers you pair with use this one’s GPUs and models, the flows it shares and its OAIY Voice. Only those you approve here can.',
  },
];

/** What this computer does now, in a word or two. */
export function doesNow(status: Pick<NetworkStatus, 'enabled' | 'uses' | 'lends'>): string {
  if (!status.enabled || (!status.uses && !status.lends)) return 'Off';
  return status.uses && status.lends ? 'Uses and lends' : status.uses ? 'Uses others' : 'Lends';
}

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
  // (switched off: using other computers, as most do first)
  const [uses, setUses] = useState(status.enabled ? status.uses : true);
  const [lends, setLends] = useState(status.enabled ? status.lends : false);
  useEffect(() => {
    setName(status.settings.name);
    setPort(String(status.settings.port));
    if (status.enabled) {
      setUses(status.uses);
      setLends(status.lends);
    }
  }, [status.settings.name, status.settings.port, status.enabled, status.uses, status.lends]);

  const portNumber = Number(port);
  const portOk = !lends || (Number.isInteger(portNumber) && portNumber >= 1024 && portNumber <= 65535);
  const some = uses || lends;
  const changed = !status.enabled || uses !== status.uses || lends !== status.lends || name.trim() !== status.settings.name || (lends && portNumber !== status.settings.port);

  const save = (e: FormEvent) => {
    e.preventDefault();
    if (!portOk || !some) return;
    onChange({ enabled: true, uses, lends, name: name.trim(), ...(lends ? { port: portNumber } : {}) });
  };
  const chosen = { uses, lends };
  const toggle = (key: 'uses' | 'lends') => (key === 'uses' ? setUses(!uses) : setLends(!lends));

  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          This computer <span className={`badge ${status.enabled ? 'badge-ok' : 'badge-neutral'}`}>{doesNow(status)}</span>
        </h3>
      </div>
      <p className="form-hint">
        OAIY Network lets OAIY computers on the same network work as one: a computer uses another’s models, GPUs, shared flows and OAIY Voice
        as if they were its own, and can lend its own at the same time. Nothing leaves your network, and it stays off until you switch it on.
      </p>
      {!status.enabled && (
        <ol className="network-steps" data-testid="network-steps">
          <li>
            <Server size={16} aria-hidden />
            <span>
              <strong>On the computer with the GPU</strong> (or the model, flow or OAIY Voice you want) switch on <em>Lend this computer</em>.
            </span>
          </li>
          <li>
            <Laptop size={16} aria-hidden />
            <span>
              <strong>On this computer</strong> switch on <em>Use other computers</em>: it shows up here, or add it by its address. A computer can do both.
            </span>
          </li>
          <li>
            <Link2 size={16} aria-hidden />
            <span>
              <strong>Pair them:</strong> approve on both screens if they show the same code. Its models, the flows it shares and its OAIY Voice then work from here.
            </span>
          </li>
        </ol>
      )}
      <form className="dl-form" onSubmit={save}>
        <div className="network-roles" role="group" aria-label="What this computer does on the network">
          {WAYS.map((w) => {
            const Icon = w.icon;
            const on = chosen[w.key];
            return (
              <button
                key={w.key}
                type="button"
                role="checkbox"
                aria-checked={on}
                className={`network-role${on ? ' is-chosen' : ''}`}
                disabled={busy}
                onClick={() => toggle(w.key)}
              >
                <span className="network-role-icon">
                  <Icon size={17} aria-hidden />
                </span>
                <span className="network-role-text">
                  <strong>{w.label}</strong>
                  <small>{w.hint}</small>
                </span>
                <span className="network-role-check" aria-hidden>
                  {on && <Check size={13} />}
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
          {lends && (
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
        {!some && <span className="form-hint network-invalid">Choose one, or both.</span>}
        <div className="form-actions">
          <button className="btn btn-primary" type="submit" disabled={busy || !portOk || !changed || !some}>
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
          {status.lends &&
            (status.listening ? (
              <div className="network-addresses">
                <span>{status.discovery ? 'Other computers on this network find it, or add it at' : 'Another computer adds it at'}</span>
                {status.addresses.map((a) => (
                  <Copyable key={a} value={a} />
                ))}
              </div>
            ) : (
              <div className="network-invalid">Not listening: switch it on again, or choose another port.</div>
            ))}
          {status.lends && (
            <div>
              It lends the computers paired with it its models and GPUs, the flows it shares (<em>Let apps run it</em> in Flows) and OAIY Voice, when set up here.
            </div>
          )}
          {status.lends && (
            <label className="network-share-models">
              <input type="checkbox" checked={status.shareModels !== false} disabled={busy} onChange={(e) => onChange({ shareModels: e.target.checked })} />
              <span>
                Share its model files: a computer using this one that gets a model this one has fetches its files from here, over your network, rather
                than downloading them again.
              </span>
            </label>
          )}
          {!status.discovery && <div>Finding computers on the network (mDNS) is off here: add them by their address.</div>}
        </div>
      )}
    </section>
  );
}

/**
 * Whether the person lets the Agent set up OAIY Network here: its tools then switch Network on, pair and unpair, and
 * choose which other computers' models, flows and OAIY Voice this one uses; and two computers that both have it on pair
 * by themselves, with no codes to compare. Saved as it is ticked.
 */
function AgentMaySetUp({ status, busy, onChange }: { status: NetworkStatus; busy: boolean; onChange: (c: Parameters<typeof network.setSettings>[0]) => void }) {
  const on = status.agent === true;
  return (
    <section className={`model-section network-agent${on ? ' is-on' : ''}`} data-testid="network-agent">
      <label className="network-agent-row">
        <input type="checkbox" checked={on} disabled={busy} onChange={(e) => onChange({ agent: e.target.checked })} />
        <span className="network-agent-text">
          <strong>
            <Bot size={15} aria-hidden /> Let the Agent pair and set up this computer
          </strong>
          <small>
            The Agent can switch OAIY Network on, pair this computer with others, and choose which of their models, flows and OAIY Voice it uses, when you
            ask it to. A code it is shown is still yours to compare. A computer pairs with this one by itself, with no codes, only while you (or the
            Agent here) open pairing below, for a few minutes: on a network you trust, your home’s or office’s.
          </small>
        </span>
      </label>
      {on && status.lends && <PairingWindow status={status} busy={busy} onChange={onChange} />}
    </section>
  );
}

/** Pairing by itself, open for a few minutes at a time: a computer that asks meanwhile, its Agent setting it up too,
 * is paired with no codes compared, and the window closes after the first. */
function PairingWindow({ status, busy, onChange }: { status: NetworkStatus; busy: boolean; onChange: (c: Parameters<typeof network.setSettings>[0]) => void }) {
  const until = status.pairingOpenUntilMs ?? 0;
  const open = until > Date.now();
  return (
    <div className={`network-window${open ? ' is-open' : ''}`} data-testid="network-window">
      {open ? (
        <>
          <span>
            <Unlock size={13} aria-hidden /> Pairing by itself is open until{' '}
            <strong>{new Date(until).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</strong>: the first computer to ask is paired.
          </span>
          <button className="btn-tiny" disabled={busy} onClick={() => onChange({ openPairingMinutes: 0 })}>
            Close
          </button>
        </>
      ) : (
        <>
          <span>Closed: computers asking to pair wait for you to compare codes.</span>
          <button className="btn-tiny" disabled={busy} onClick={() => onChange({ openPairingMinutes: 5 })}>
            <Unlock size={12} aria-hidden /> Allow pairing by itself for 5 minutes
          </button>
        </>
      )}
    </div>
  );
}

/** Whether a pairing error says the other computer was not reached (rather than that it expired, declined or
 * differs in version). */
function unreachable(error: string): boolean {
  return /not reach|unreachable|refused|timed? ?out|did not answer|no route|connect|resolve|network is/i.test(error) && !/expired|declin|denied|version/i.test(error);
}

/** The six digits both screens show, large enough to compare at a glance. */
export function PairingCode({ code }: { code: string }) {
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
          Asking to use this computer <span className="section-count">({requests.length})</span>
        </h3>
      </div>
      {requests.length === 0 ? (
        <p className="form-hint">When another computer asks to use this one, it shows here. Approve it only if its code matches the one on that computer’s screen.</p>
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

function FindWorkers({
  found,
  discovery,
  inProgress,
  lends,
  onPaired,
}: {
  found: NetworkFound[];
  discovery: boolean;
  /** The pairings this computer is in the middle of: one started before the page was left is shown again. */
  inProgress?: NetworkPairing[];
  /** This computer lends itself too: the computer it pairs with may use it as well, as its person chooses. */
  lends: boolean;
  onPaired: () => void;
}) {
  const [address, setAddress] = useState('');
  /** Paired both ways at once: the other computer may use this one too. */
  const [bothWays, setBothWays] = useState(true);
  const [pairing, setPairing] = useState<NetworkPairing | null>(() => inProgress?.[0] ?? null);
  /** The worker approved: this side's person says whether it showed the same code. */
  const [approvedThere, setApprovedThere] = useState(() => inProgress?.[0]?.status === 'approved_there');
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
      const r = await network.confirmPairing(pairing.pairingId, accept, accept && lends && bothWays);
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
      // (both ways where this computer lends itself and the person leaves it ticked: used if the other pairs by itself
      // at once, its window open; else asked again once its person approves)
      const started = await network.startPairing({ ...target, bothWays: lends && bothWays });
      if (started.status === 'approved') {
        // (its window to pair by itself was open: paired at once, nothing to compare)
        const back = (started as NetworkPairing & { node?: { usesThis?: boolean } }).node?.usesThis;
        setOutcome({ text: `Paired with ${started.worker.name} at once: its window to pair by itself was open${back ? ', and it can use this computer too' : ''}.`, paired: true });
        setAddress('');
        onPaired();
      } else {
        setPairing(started);
      }
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
          Add a computer to use {discovery && <span className="section-count">({unpaired.length} found)</span>}
        </h3>
      </div>
      {error && (
        <div className="banner banner-err" role="alert">
          <div>{error}</div>
          {/* (what to check where the other computer was not reached: not for a pairing that expired or was declined) */}
          {unreachable(error) && <ul className="network-checklist">
            <li>On the other computer, Setup → Networking has Lend this computer switched on, and shows the address to type here.</li>
            <li>Both computers are on the same network (not a guest Wi-Fi that keeps devices apart).</li>
            <li>The worker’s firewall lets OAIY in (macOS asks the first time; Windows may need “Allow”).</li>
          </ul>}
        </div>
      )}
      {outcome && <div className={`banner ${outcome.paired ? 'banner-ok' : 'banner-pending'}`}>{outcome.text}</div>}
      {pairing && (
        <div className="pairing-card pairing-pending network-pairing" data-testid="network-pairing">
          <div className="pairing-head">
            <Server size={16} aria-hidden />
            <strong>{pairing.worker.name}</strong>
            <span className="pairing-origin">{pairing.worker.address}</span>
          </div>
          <ol className="network-progress" aria-label="Pairing">
            <li className="is-done">Code shown</li>
            <li className={approvedThere ? 'is-done' : 'is-now'}>Approved on {pairing.worker.name}</li>
            <li className={approvedThere ? 'is-now' : ''}>Codes matched here</li>
          </ol>
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
          {approvedThere && lends && (
            <label className="network-both-ways">
              <input type="checkbox" checked={bothWays} onChange={(e) => setBothWays(e.target.checked)} />
              <span>
                Let {pairing.worker.name} use this computer too <small>(this computer lends itself: one pairing, both ways)</small>
              </span>
            </label>
          )}
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
            <Loader2 size={13} className="spin" aria-hidden /> Looking for OAIY computers on this network… One shows here once Lend this computer is on there.
          </p>
        ) : (
          <div className="network-list">
            {unpaired.map((f) => (
              <div className="network-found" key={f.nodeId} data-testid="network-found">
                <Server size={16} aria-hidden />
                <span className="network-found-name">
                  <strong>{f.name || 'An OAIY computer'}</strong>
                  {f.hardware && <span className="network-found-hardware">{f.hardware}</span>}
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
      {lends && !pairing && (
        <label className="network-both-ways">
          <input type="checkbox" checked={bothWays} onChange={(e) => setBothWays(e.target.checked)} />
          <span>
            Let the computer paired with use this one too <small>(this computer lends itself: one pairing, both ways)</small>
          </span>
        </label>
      )}
      <span className="form-hint">The address the other computer shows on its own Networking page, for networks where computers can’t find each other.</span>
    </section>
  );
}

/** A worker's own GPU (`this computer`, where its language model runs) is the one that shares its memory. */
export function deviceName(n: NetworkNode, device: string | null): string | null {
  if (!device) return null;
  // (a GPU by its id; else one its engine's own words name, "on WebGPU on Apple M5 Pro (Metal, …)"; else its own GPU)
  const named = n.devices.find((d) => d.id === device) ?? n.devices.find((d) => device.includes(d.name));
  if (named) return named.name;
  if (device === 'this computer' || device.length > 24) return n.devices.find((d) => d.unified)?.name ?? `${n.name}’s own GPU`;
  return device;
}

/** What can be pinned on a worker: its picture models, and its language model once it is the one chosen there (one
 * pinned can be unpinned, loaded or not). */
const pinnable = (m: NetworkModel) => m.kind === 'image' || (m.kind === 'llm' && (m.state !== 'not loaded' || m.pinned));

export const KIND: Record<string, string> = { llm: 'Text', image: 'Picture', video: 'Video', speech: 'Speech', music: 'Music', sound: 'Sound', model3d: '3D', vision: 'Vision', embedding: 'Embedding' };
export const STATE: Record<string, { label: string; tone: 'ok' | 'busy' | 'idle' | 'off' }> = {
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
function NodeModels({ node, onChanged, defaults }: { node: NetworkNode; onChanged: () => void; defaults: NetworkStatus['defaults'] }) {
  /** This computer's default of its kind (Engines → Other computers); a worker's language model, whichever it runs. */
  const isDefault = (m: NetworkModel) => {
    const d = defaults?.[m.kind];
    return !!d && (m.kind === 'llm' ? d.nodeId === node.nodeId : d.id === m.id);
  };
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loaded = node.models.filter((m) => m.state !== 'not loaded' || m.pinned || isDefault(m));
  const rest = node.models.filter((m) => m.state === 'not loaded' && !m.pinned && !isDefault(m));
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
          {isDefault(m) && (
            <span className="engine-default-chip" title="This computer's default of its kind (Engines → Other computers)">
              <Check size={11} aria-hidden /> Default
            </span>
          )}
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

/** A GPU of a worker: its memory in use as a meter (VRAM on a card of its own, else the computer's memory it shares),
 * how busy and how warm it is (not while the worker does not answer). */
export function Device({ d, live, temperature }: { d: NetworkDevice; live: boolean; temperature?: number | null }) {
  const total = typeof d.memory_total_mb === 'number' && d.memory_total_mb > 0 ? d.memory_total_mb : null;
  const used = typeof d.memory_used_mb === 'number' ? d.memory_used_mb : null;
  const share = total && used !== null ? Math.min(100, Math.max(0, (used / total) * 100)) : null;
  return (
    <div className="network-device">
      <div className="network-device-head">
        <span className="network-device-name">{d.name}</span>
        {live && (typeof d.load === 'number' || typeof temperature === 'number') && (
          <span className="network-device-load">
            {[typeof d.load === 'number' ? `${Math.round(d.load)}% busy` : null, typeof temperature === 'number' ? `${Math.round(temperature)}°C` : null].filter(Boolean).join(' · ')}
          </span>
        )}
      </div>
      <div className="network-meter" aria-hidden>
        <div className={`network-meter-fill${share !== null && share > 85 ? ' is-full' : ''}`} style={{ width: `${share ?? 0}%` }} />
      </div>
      <div className="network-device-figure">
        {total ? `${used !== null ? gb(used) : '?'} of ${gb(total)} GB ${d.unified ? 'shared memory' : 'VRAM'} in use` : 'memory unknown'}
      </div>
    </div>
  );
}

/** A worker's processor and memory, as tiles beside its GPUs. */
export function MachineTiles({ m, live }: { m: NetworkMachine; live: boolean }) {
  const cpu = m.cpu;
  const ram = m.ram;
  const total = typeof ram?.total_mb === 'number' && ram.total_mb > 0 ? ram.total_mb : null;
  const used = total !== null && typeof ram?.available_mb === 'number' ? Math.max(0, total - ram.available_mb) : null;
  return (
    <>
      {cpu && (
        <div className="network-device network-device-cpu">
          <div className="network-device-head">
            <span className="network-device-name">Processor</span>
            {live && typeof cpu.busy === 'number' && <span className="network-device-load">{cpu.busy}% busy</span>}
          </div>
          <div className="network-meter" aria-hidden>
            <div className="network-meter-fill" style={{ width: `${live && typeof cpu.busy === 'number' ? cpu.busy : 0}%` }} />
          </div>
          <div className="network-device-figure">{[cpu.name, cpu.cores ? `${cpu.cores} cores` : null].filter(Boolean).join(' · ')}</div>
        </div>
      )}
      {total !== null && (
        <div className="network-device network-device-ram">
          <div className="network-device-head">
            <span className="network-device-name">Memory (RAM)</span>
          </div>
          <div className="network-meter" aria-hidden>
            <div className={`network-meter-fill${used !== null && used / total > 0.85 ? ' is-full' : ''}`} style={{ width: `${used !== null ? Math.min(100, (used / total) * 100) : 0}%` }} />
          </div>
          <div className="network-device-figure">{used !== null ? `${gb(used)} of ${gb(total)} GB in use` : `${gb(total)} GB`}</div>
        </div>
      )}
    </>
  );
}

const JOB_KIND: Record<string, string> = { image: 'a picture', video: 'a video', speech: 'speech', music: 'music', sound: 'a sound', model3d: 'a 3D model', picture: 'a picture edit', flow: 'a flow run' };

/** A worker's queue: what it is making now and what waits, in the order it runs them. */
export function Queue({ jobs }: { jobs: NetworkJob[] }) {
  if (jobs.length === 0) return <div className="network-queue-empty">Nothing running or waiting.</div>;
  const waiting = jobs.filter((j) => j.status === 'waiting').length;
  return (
    <div className="network-queue" data-testid="network-queue">
      <div className="settings-label">
        Queue <span className="section-count">({jobs.length - waiting} running, {waiting} waiting)</span>
      </div>
      {jobs.slice(0, 8).map((j, i) => (
        <div className={`network-job${j.status === 'running' ? ' is-running' : ''}`} key={`${j.kind}:${j.runId ?? j.model ?? ''}:${i}`}>
          {j.status === 'running' ? <Loader2 size={12} className="spin" aria-hidden /> : <span className="network-job-dot" aria-hidden />}
          <span className="network-job-what">
            {j.status === 'running' ? 'Making' : 'Waiting:'} {JOB_KIND[j.kind] ?? j.kind}
            {j.model ? <code>{j.model}</code> : j.flow ? <code>{j.flow}</code> : null}
            {j.mine && <span className="network-way">this computer’s</span>}
          </span>
          {j.status === 'running' && typeof j.progress === 'number' && j.progress > 0 && <span className="network-job-progress">{Math.round(j.progress)}%</span>}
        </div>
      ))}
      {jobs.length > 8 && <div className="form-hint">and {jobs.length - 8} more</div>}
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
  uses,
  lends,
  defaults,
  services,
  onNavigate,
  onUnpair,
  onChanged,
}: {
  nodes: NetworkNode[];
  /** This computer uses other computers now (else its pairings to use others wait). */
  uses: boolean;
  /** This computer lends itself now (else the computers paired to use it wait). */
  lends: boolean;
  defaults: NetworkStatus['defaults'];
  services: NetworkStatus['services'];
  onNavigate?: (view: 'flow-workers' | 'services' | 'engine-workers') => void;
  onUnpair: (n: NetworkNode) => void;
  onChanged: () => void;
}) {
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">
          Paired computers <span className="section-count">({nodes.length})</span>
        </h3>
      </div>
      {nodes.length === 0 ? (
        <p className="form-hint">{uses ? 'No computer is paired yet: add one above.' : 'No computer uses this one yet.'}</p>
      ) : (
        <div className="network-list">
          {nodes.map((n) => {
            // (paired one way or both: this computer uses it, and it uses this computer; one this computer is paired
            // to use waits while Use other computers is off here)
            const paired = n.uses ?? n.role === 'worker';
            const usesThis = n.usesThis ?? n.role === 'primary';
            const worker = paired && uses;
            const Icon = paired ? Server : Laptop;
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
                  {paired && <span className="network-way">{uses ? 'This computer uses it' : 'This computer can use it (Use other computers is off)'}</span>}
                  {usesThis && <span className="network-way">{lends ? 'Uses this computer' : 'Can use this computer (Lend this computer is off)'}</span>}
                  {n.pairedByItself && (
                    <span className="network-way network-way-auto" title="Both computers let their Agent set up OAIY Network, so they paired with no codes compared. Unpair it if you do not know it.">
                      <Bot size={11} aria-hidden /> Paired by itself
                    </span>
                  )}
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
                <div className={stale ? 'network-stale' : undefined} hidden={!worker}>
                  {stale && n.lastSeenMs && <div className="network-stale-note">Last report {ago(n.lastSeenMs)}</div>}
                  {(n.devices.length > 0 || n.machine) && (
                    <div className="network-devices">
                      {n.machine && <MachineTiles m={n.machine} live={!stale} />}
                      {n.devices.map((d) => (
                        <Device key={d.id} d={d} live={!stale} temperature={n.machine?.gpus?.find((g) => g.name === d.name)?.temperature} />
                      ))}
                    </div>
                  )}
                  {!stale && (n.jobs ? <Queue jobs={n.jobs} /> : busyNow && <div className="network-busy">{busyNow}</div>)}
                  {n.models.length > 0 && <NodeModels node={n} onChanged={onChanged} defaults={defaults} />}
                </div>
                {worker && ((n.flows ?? []).length > 0 || (n.services ?? []).length > 0) && (
                  <div className="network-extras">
                    {(n.flows ?? []).length > 0 && (
                      <div className="network-extra">
                        <span className="settings-label">Flows it shares</span>
                        <span className="network-extra-what">{(n.flows ?? []).map((f) => f.name).join(', ')}</span>
                        {onNavigate && (
                          <button className="btn-tiny" onClick={() => onNavigate('flow-workers')}>
                            <Workflow size={12} aria-hidden /> Run them
                          </button>
                        )}
                      </div>
                    )}
                    {(n.services ?? []).length > 0 && (
                      <div className="network-extra">
                        <span className="settings-label">Services it lends</span>
                        <span className="network-extra-what">
                          {(n.services ?? []).map((x) => `${x.name}${services?.[x.id]?.nodeId === n.nodeId ? ' (in use here)' : x.installed ? '' : ' (not set up there)'}`).join(', ')}
                        </span>
                        {onNavigate && (
                          <button className="btn-tiny" onClick={() => onNavigate('services')}>
                            <AudioLines size={12} aria-hidden /> Use here
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {uses && nodes.some((n) => n.uses ?? n.role === 'worker') && (
        <p className="form-hint">
          A worker’s models are listed in Flows’ model menus, named with the worker, and run on the worker; Engines → Other computers makes one this
          computer’s default of its kind.
        </p>
      )}
    </section>
  );
}
