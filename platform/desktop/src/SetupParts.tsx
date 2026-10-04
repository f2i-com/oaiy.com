import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Check, CircleDashed, Cpu, Download, ExternalLink, FolderSearch, Loader2, Mic, Play, RotateCcw, SkipForward, TriangleAlert } from 'lucide-react';
import ChatGptConnector from './ChatGptConnector';
import {
  engines,
  formatBytes,
  isTauri,
  services as servicesApi,
  type AddedModelFile,
  type EngineCatalog,
  type EngineCatalogModel,
  type EngineDownload,
  type ServiceSnapshot,
} from './api';
import { chosenModel, groupModels, recommendedModel, type StepState } from './setupFlow';

/**
 * The setup wizard's shared parts: its frame (the step list, the pane, Back /
 * Skip / Next), and the cards the engine step and a plugin's requirements step
 * both use: a service (OAIY Voice) installed with progress, and the engines'
 * model for a group, which is the one chosen in Engines (only when there is
 * none does the wizard offer the catalog's models to download, the recommended
 * one selected, or a language model file the computer already has).
 */

/** A value asked for now and every `ms` while the window is visible. */
export function usePoll<T>(load: () => Promise<T>, ms: number, enabled = true): [T | null, () => Promise<void>, string | null] {
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;
  const refresh = useCallback(async () => {
    try {
      setValue(await loadRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => {
    if (!enabled) return;
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, ms);
    return () => window.clearInterval(id);
  }, [enabled, ms, refresh]);
  return [value, refresh, error];
}

/**
 * What a route a desktop may not have said, polled as `{ v }` so an answer of
 * `null` (a 404: an older desktop) differs from not asked yet: `undefined`
 * while asking, else its answer, `null` for a missing route or a failure.
 */
export function answered<T>(value: { v: T | null } | null, error: string | null): T | null | undefined {
  if (value) return value.v;
  return error ? null : undefined;
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message.replace(/^\d{3}: /, '') : String(e);
}

// ---------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------

export interface RailItem {
  id: string;
  title: string;
  hint?: string;
  state: StepState;
  optional?: boolean;
}

function Mark({ state }: { state: StepState }) {
  if (state === 'done') return <Check size={13} strokeWidth={2.6} />;
  if (state === 'skipped') return <SkipForward size={12} />;
  return <CircleDashed size={13} />;
}

/** The steps, in order: the current one marked, done ones ticked, each one a way back to it. */
export function StepRail({ items, current, onPick, label }: { items: RailItem[]; current: number; onPick?: (i: number) => void; label: string }) {
  return (
    <nav className="setup-rail" aria-label={label}>
      <ol>
        {items.map((s, i) => (
          <li key={s.id} className={`setup-rail-item is-${s.state}${i === current ? ' is-current' : ''}`}>
            <button type="button" aria-current={i === current ? 'step' : undefined} disabled={!onPick} onClick={() => onPick?.(i)}>
              <span className="setup-rail-mark" aria-hidden>
                {s.state === 'todo' && i !== current ? <span className="setup-rail-num">{i + 1}</span> : <Mark state={s.state} />}
              </span>
              <span className="setup-rail-text">
                <strong>{s.title}</strong>
                {s.hint && <small>{s.hint}</small>}
              </span>
              <span className="sr-only">
                {s.state === 'done' ? ', done' : s.state === 'skipped' ? ', skipped' : s.optional ? ', optional' : ''}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

/** A step's heading in the pane. */
export function StepHeader({ kicker, title, description, state, aside }: { kicker: string; title: string; description?: ReactNode; state?: StepState; aside?: ReactNode }) {
  return (
    <header className="setup-step-head">
      <div className="setup-step-titles">
        <span className="setup-kicker">{kicker}</span>
        <h2>
          {title}
          {state === 'done' && <span className="badge badge-ok">Done</span>}
          {state === 'skipped' && <span className="badge badge-neutral">Skipped</span>}
        </h2>
        {description && <p>{description}</p>}
      </div>
      {aside}
    </header>
  );
}

/** Back, a status line, Skip, and Next (or Finish). */
export function StepFooter({
  onBack,
  backLabel = 'Back',
  status,
  onSkip,
  skipLabel = 'Skip',
  onNext,
  nextLabel = 'Next',
  nextDisabled,
  busy,
}: {
  onBack?: () => void;
  backLabel?: string;
  status?: ReactNode;
  onSkip?: () => void;
  skipLabel?: string;
  onNext?: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
  busy?: boolean;
}) {
  return (
    <footer className="setup-foot">
      <button type="button" className="btn btn-ghost" onClick={onBack} disabled={!onBack || busy}>
        {backLabel}
      </button>
      <span className="setup-foot-status" role="status">
        {status}
      </span>
      {onSkip && (
        <button type="button" className="btn btn-ghost" onClick={onSkip} disabled={busy}>
          {skipLabel}
        </button>
      )}
      {onNext && (
        <button type="button" className="btn btn-primary" onClick={onNext} disabled={nextDisabled || busy}>
          {busy && <Loader2 size={14} className="spin" />} {nextLabel}
        </button>
      )}
    </footer>
  );
}

/** The whole wizard: a heading with progress, the steps, and the pane with its footer. */
export function WizardFrame({
  title,
  subtitle,
  progress,
  rail,
  current,
  onPick,
  railLabel,
  onLeave,
  flush,
  children,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  progress: { done: number; total: number };
  rail: RailItem[];
  current: number;
  onPick?: (i: number) => void;
  railLabel: string;
  onLeave: () => void;
  /** The pane holds a plugin screen, edge to edge. */
  flush?: boolean;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  return (
    <div className="setup-page">
      <div className="setup-hero">
        <div className="setup-hero-text">
          <h1>{title}</h1>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <div className="setup-hero-side">
          <span className="setup-progress-label">
            {progress.done} of {progress.total} done
          </span>
          <span className="setup-progress" role="progressbar" aria-label="Setup progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
            <i style={{ width: `${pct}%` }} />
          </span>
          <button type="button" className="btn btn-ghost setup-leave" onClick={onLeave} title="Leave setup: it keeps its place">
            Continue later
          </button>
        </div>
      </div>
      <div className="setup-body">
        <StepRail items={rail} current={current} onPick={onPick} label={railLabel} />
        <div className="setup-main">
          <section className={flush ? 'setup-pane is-flush' : 'setup-pane'}>{children}</section>
          {footer}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Requirement cards
// ---------------------------------------------------------------------------

/** A service (OAIY Voice) the step needs: installed through the services route, with its progress. */
export function ServiceCard({ id, why, services, onChanged }: { id: string; why?: string; services: ServiceSnapshot[] | null; onChanged: () => void }) {
  const svc = services?.find((s) => s.id === id) ?? null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [line, setLine] = useState('');
  const installing = svc?.status === 'installing';

  // While it installs, the last line of its install log is its progress.
  useEffect(() => {
    if (!installing) return;
    let stop = false;
    const tick = async () => {
      try {
        const lines = await servicesApi.logs(id, 3);
        const last = [...lines].reverse().find((l) => l.text.trim());
        if (!stop && last) setLine(last.text.trim());
      } catch {
        /* the next tick tries again */
      }
    };
    void tick();
    const t = window.setInterval(tick, 2000);
    return () => {
      stop = true;
      window.clearInterval(t);
    };
  }, [installing, id]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const pct = /(\d{1,3}(?:\.\d+)?)\s?%/.exec(line)?.[1];
  let status: ReactNode;
  let action: ReactNode = null;
  if (!services) status = <span className="form-hint">Checking…</span>;
  else if (!svc) status = <span className="card-warn">This OAIY has no {id} service.</span>;
  else if (installing) {
    status = (
      <span className="setup-req-progress">
        <span className="setup-progress" role="progressbar" aria-label={`Installing ${svc.name}`} aria-valuenow={pct ? Number(pct) : undefined}>
          <i className={pct ? undefined : 'is-indeterminate'} style={pct ? { width: `${Math.min(100, Number(pct))}%` } : undefined} />
        </span>
        <small className="setup-mono">{line || 'Starting the install…'}</small>
      </span>
    );
    action = (
      <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => void act(() => servicesApi.cancelInstall(id))}>
        Cancel
      </button>
    );
  } else if (svc.installed || !svc.installable) {
    status = (
      <span className={svc.status === 'running' ? 'badge badge-ok' : 'badge badge-neutral'}>{svc.status === 'running' ? 'Installed and running' : `Installed · ${svc.status}`}</span>
    );
    if (svc.status !== 'running' && svc.status !== 'starting')
      action = (
        <button type="button" className="btn" disabled={busy} onClick={() => void act(() => servicesApi.start(id))}>
          <Play size={14} /> Start it
        </button>
      );
  } else {
    status = svc.status === 'errored' && svc.error ? <span className="card-warn">{svc.error}</span> : <span className="badge badge-neutral">Not installed</span>;
    action = (
      <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void act(() => servicesApi.install(id))}>
        {busy ? <Loader2 size={14} className="spin" /> : <Download size={14} />} {svc.status === 'errored' ? 'Try again' : 'Install'}
      </button>
    );
  }
  return (
    <div className="setup-req">
      <span className="setup-req-icon" aria-hidden>
        <Mic size={16} />
      </span>
      <div className="setup-req-body">
        <strong>{svc?.name ?? id}</strong>
        <p>{why ?? svc?.description}</p>
        <div className="setup-req-status">{status}</div>
        {error && <p className="card-warn" role="alert">{error}</p>}
      </div>
      {action && <div className="setup-req-action">{action}</div>}
    </div>
  );
}

const ACTIVE = new Set(['queued', 'downloading', 'adding']);

/** The group's name as the catalog calls it ("Chat"), for "a Chat model". */
function groupLabel(catalog: EngineCatalog | null, group: string): string {
  if (group === 'llm') return 'Language model';
  return catalog?.groups?.find((g) => g.id === group)?.name ?? group;
}

/** A model's download size, the GPU memory it needs, whether that fits the largest GPU (`gpuGb`), and its license. */
function modelFacts(m: EngineCatalogModel, gpuGb?: number): string {
  const fit = gpuGb && m.vramGb ? (m.vramGb <= gpuGb ? 'fits your GPU' : `more than your GPU’s ${gpuGb} GB`) : null;
  return [m.sizeGb ? `${m.sizeGb} GB download` : null, m.vramGb ? `needs ${m.vramGb} GB of GPU memory` : null, fit, m.license].filter(Boolean).join(' · ');
}

/** A download as it goes: a bar, and how far. */
function DownloadProgress({ name, dl }: { name: string; dl: EngineDownload }) {
  return (
    <span className="setup-req-progress">
      <span className="setup-progress" role="progressbar" aria-label={`Downloading ${name}`} aria-valuenow={dl.total ? Math.round((dl.done / dl.total) * 100) : undefined}>
        <i className={dl.total ? undefined : 'is-indeterminate'} style={dl.total ? { width: `${Math.min(100, (dl.done / dl.total) * 100)}%` } : undefined} />
      </span>
      <small className="setup-mono">
        {dl.status === 'queued' ? 'Waiting to start…' : dl.status === 'adding' ? 'Adding it to Engines…' : `${formatBytes(dl.done)} of ${formatBytes(dl.total)}`}
        {dl.speed ? ` · ${dl.speed} MB/s` : ''}
        {dl.filesTotal && dl.filesTotal > 1 ? ` · file ${Math.min((dl.filesDone ?? 0) + 1, dl.filesTotal)} of ${dl.filesTotal}` : ''}
      </small>
    </span>
  );
}

/**
 * A language model file this computer already has (LM Studio's, a download of the person's own), added to the
 * engines through the desktop's window: typed, pasted or picked, and refused there when the engine cannot run it.
 */
function OwnModelFile({ onAdded }: { onAdded: (added: AddedModelFile) => void }) {
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const browse = async () => {
    setError(null);
    try {
      const picked = await engines.pickModelFile();
      if (picked) setPath(picked);
    } catch (e) {
      setError(errorText(e));
    }
  };
  const add = async () => {
    setBusy(true);
    setError(null);
    try {
      onAdded(await engines.addModelFile(path));
      setPath('');
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form
      className="setup-own-model"
      onSubmit={(e) => {
        e.preventDefault();
        if (path.trim() && !busy) void add();
      }}
    >
      <strong>Or use a model file you already have</strong>
      <div className="folder-change-row">
        <input type="text" aria-label="Model file" placeholder="The .gguf file’s full path" value={path} onChange={(e) => setPath(e.target.value)} disabled={busy} />
        <button type="button" className="btn btn-secondary" onClick={() => void browse()} disabled={busy}>
          <FolderSearch size={14} /> Choose…
        </button>
        <button type="submit" className="btn btn-primary" disabled={busy || !path.trim()}>
          {busy && <Loader2 size={14} className="spin" />} Use this file
        </button>
      </div>
      <small className="form-hint">
        A language model in a .gguf file, such as one LM Studio downloaded (of a model split into files, the first). The engine runs Qwen3.5, Qwen3, Qwen2, Llama, Mistral, Gemma
        3 and 4, and GLM models; the Agent can use its tools with Qwen3.5 and GLM, and chats with the others.
      </small>
      {error && (
        <p className="card-warn" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}

/**
 * The engines' model for `group`: the one chosen in Engines, whatever it is.
 * Only when none is chosen does it offer the catalog's models for the group to
 * download, the recommended one selected, each with its size and the GPU memory
 * it needs; for a language model, which the Agent can use its tools with, and a
 * file this computer already has instead.
 */
export function EngineModelCard({
  group,
  why,
  catalog,
  gpuGb,
  onChanged,
  onOpenEngines,
}: {
  group: string;
  why?: string;
  catalog: EngineCatalog | null;
  /** The largest GPU's memory, to say whether each model fits it. */
  gpuGb?: number;
  onChanged: () => void;
  onOpenEngines: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [added, setAdded] = useState<AddedModelFile | null>(null);
  const chosen = chosenModel(catalog, group);
  const options = chosen ? [] : groupModels(catalog, group);
  // A download under way holds the choice: it is the one shown, and the others wait.
  const active = options.find((m) => m.download && ACTIVE.has(m.download.status)) ?? null;
  const offer = active ?? options.find((m) => m.id === picked) ?? recommendedModel(catalog, group);
  const dl = offer?.download ?? null;
  const llm = group === 'llm';
  const ownFile = llm && isTauri();
  const addedFile = (a: AddedModelFile) => {
    setAdded(a);
    onChanged();
  };

  const download = async () => {
    if (!offer) return;
    setBusy(true);
    setError(null);
    try {
      await engines.download(offer.id);
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  let body: ReactNode;
  let action: ReactNode = null;
  if (!catalog) body = <span className="form-hint">Asking the engines…</span>;
  else if (!catalog.running)
    body = (
      <span className="card-warn">
        <TriangleAlert size={12} /> {catalog.error ?? 'The engines are not running.'}
      </span>
    );
  else if (chosen)
    body = (
      <>
        <span className="setup-chosen">
          <span className="badge badge-ok">Chosen in Engines</span> <strong>{chosen}</strong>
        </span>
        {added && added.name === chosen && !added.tools && (
          <p className="card-warn setup-chat-only">The Agent chats with {chosen} but cannot use its tools with it: they need a Qwen3.5 or GLM model.</p>
        )}
      </>
    );
  else if (!offer)
    body = (
      <div className="setup-offer">
        <span className="form-hint">Nothing is chosen in Engines, and the catalog has no model for this. Add one in Engines.</span>
        {ownFile && <OwnModelFile onAdded={addedFile} />}
      </div>
    );
  else {
    body = (
      <div className="setup-offer">
        <p className="form-hint">{options.length > 1 ? 'Nothing is chosen in Engines yet. Choose one to download:' : 'Nothing is chosen in Engines yet. The catalog recommends:'}</p>
        <div className="setup-model-list" role="radiogroup" aria-label={`${groupLabel(catalog, group)} to download`}>
          {options.map((m) => {
            const selected = m.id === offer.id;
            return (
              <label key={m.id} className={`setup-offer-card setup-model-option${selected ? ' is-selected' : ''}`}>
                {options.length > 1 && (
                  <input type="radio" name={`engine-model-${group}`} value={m.id} checked={selected} disabled={!!active || busy} onChange={() => setPicked(m.id)} />
                )}
                <span className="setup-model-text">
                  <strong>
                    {m.name} {m.recommended && <span className="badge badge-pending">Recommended</span>}
                    {llm && (m.agentTools ? <span className="badge badge-ok">Agent tools</span> : <span className="badge badge-neutral">Chat only</span>)}
                    {m.installed && <span className="badge badge-ok">Downloaded</span>}
                  </strong>
                  {m.about && <small>{m.about}</small>}
                  <small className="setup-mono">{modelFacts(m, gpuGb)}</small>
                </span>
              </label>
            );
          })}
        </div>
        {llm && options.length > 1 && <small className="form-hint">The Agent works OAIY with tools, which it can use with the models marked Agent tools; with the others it chats only.</small>}
        {dl && ACTIVE.has(dl.status) && <DownloadProgress name={offer.name} dl={dl} />}
        {dl?.status === 'failed' && dl.error && <p className="card-warn">{dl.error}</p>}
        {offer.installed && <p className="form-hint">Downloaded. Choose it in Engines to use it.</p>}
        {ownFile && <OwnModelFile onAdded={addedFile} />}
      </div>
    );
    if (!offer.installed && !(dl && ACTIVE.has(dl.status)))
      action = (
        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void download()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Download size={14} />} {dl?.status === 'failed' ? 'Try again' : `Download${offer.sizeGb ? ` (${offer.sizeGb} GB)` : ''}`}
        </button>
      );
  }
  if (catalog && (chosen || !catalog.running || offer?.installed || !offer))
    action = (
      <button type="button" className="btn" onClick={onOpenEngines}>
        {chosen ? 'Change it in Engines' : 'Open Engines'} <ExternalLink size={13} />
      </button>
    );
  return (
    <div className="setup-req">
      <span className="setup-req-icon" aria-hidden>
        <Cpu size={16} />
      </span>
      <div className="setup-req-body">
        <strong>{groupLabel(catalog, group)}</strong>
        <p>{why ?? 'Runs on this computer, in OAIY’s engines. The one chosen in Engines is used.'}</p>
        <div className="setup-req-status">{body}</div>
        {error && <p className="card-warn" role="alert">{error}</p>}
      </div>
      {action && <div className="setup-req-action">{action}</div>}
    </div>
  );
}

/** Or an AI source instead of the engines: ChatGPT, a provider's API key, or a local model server. */
export function AiSourceChoice({ onNavigate }: { onNavigate: (v: 'providers' | 'services') => void }) {
  const [choice, setChoice] = useState<'codex' | 'api' | 'local'>('codex');
  return (
    <div className="setup-source">
      <div className="seg-tabs" role="tablist" aria-label="Choose an AI source">
        {(['codex', 'api', 'local'] as const).map((c) => (
          <button type="button" role="tab" key={c} aria-selected={choice === c} className={choice === c ? 'active' : undefined} onClick={() => setChoice(c)}>
            <span>{c === 'codex' ? 'ChatGPT' : c === 'api' ? 'Provider API key' : 'Local model server'}</span>
          </button>
        ))}
      </div>
      {choice === 'codex' ? (
        <ChatGptConnector />
      ) : (
        <div className="setup-source-other">
          <p className="form-hint">
            {choice === 'api'
              ? 'Add your provider in AI providers, enter its API key and model, then test the connection. Keys stay on this computer; the provider may charge for use.'
              : 'Install or connect a local model server in Services, start it, and download a model for it.'}
          </p>
          <button type="button" className="btn" onClick={() => onNavigate(choice === 'api' ? 'providers' : 'services')}>
            {choice === 'api' ? 'Open AI providers' : 'Open Services'} <ExternalLink size={13} />
          </button>
        </div>
      )}
    </div>
  );
}

/** A small "try again" for a read that failed. */
export function RetryLine({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <p className="card-warn setup-retry" role="alert">
      <TriangleAlert size={12} /> {message}{' '}
      <button type="button" className="btn-tiny" onClick={onRetry}>
        <RotateCcw size={12} /> Try again
      </button>
    </p>
  );
}
