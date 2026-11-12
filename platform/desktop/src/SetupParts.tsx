import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { Check, CircleDashed, Cpu, Download, ExternalLink, FolderSearch, Loader2, Mic, Network, Play, RotateCcw, SkipForward, TriangleAlert } from 'lucide-react';
import ChatGptConnector from './ChatGptConnector';
import {
  engines,
  formatBytes,
  formatSpeed,
  isTauri,
  services as servicesApi,
  type AddedModelFile,
  type EngineCatalog,
  type EngineCatalogModel,
  type EngineDownload,
  type EnginePlan,
  type ServiceSnapshot,
} from './api';
import { chosenModel, groupModels, modelName, recommendedModel, type StepState } from './setupFlow';

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
      // (said as the person reads it: "the desktop is busy", not "500: the desktop is busy")
      setError(errorText(e));
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
 * `null` (a 404, which the client answers as `null`: an older desktop) differs
 * from not known yet: its answer, `null` for a missing route, `undefined` while
 * asking or after a failure (its error says why; nothing is decided on it, and
 * the next poll asks again).
 */
export function answered<T>(value: { v: T | null } | null, _error: string | null): T | null | undefined {
  return value ? value.v : undefined;
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

/**
 * The way out of the wizard, given by its frame: the footer of each step offers it at its left ("Continue later"),
 * beside Back, where it was a row of its own above the step.
 */
const WizardLeave = createContext<(() => void) | null>(null);

/**
 * Continue later and Back, a status line, Skip, and the step's one thing to do: Next (or Finish), or, where the step
 * waits on something the person starts (a model to download), that thing itself, by its name.
 */
export function StepFooter({
  onBack,
  backLabel = 'Back',
  status,
  onSkip,
  skipLabel = 'Skip',
  onNext,
  nextLabel = 'Next',
  nextIcon,
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
  /** An icon for a primary button that is not "Next" (a download). */
  nextIcon?: ReactNode;
  nextDisabled?: boolean;
  busy?: boolean;
}) {
  const leave = useContext(WizardLeave);
  return (
    <footer className="setup-foot">
      {leave && (
        <button type="button" className="btn btn-ghost setup-leave" onClick={leave} disabled={busy} title="Leave setup: it keeps its place">
          Continue later
        </button>
      )}
      {/* (no Back on the first step: a button that can never be pressed is not shown) */}
      {(onBack || !leave) && (
        <button type="button" className="btn btn-ghost" onClick={onBack} disabled={!onBack || busy}>
          {backLabel}
        </button>
      )}
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
          {busy ? <Loader2 size={14} className="spin" /> : nextIcon} {nextLabel}
        </button>
      )}
    </footer>
  );
}

/**
 * Where the setup page's heading goes: the dashboard's top bar, which names every page (App provides it), so the page
 * has one heading and not the top bar's "Setup" over the wizard's "Set up OAIY". Without it (a test, or a wizard drawn
 * somewhere else) the wizard heads itself.
 */
export const SetupHeading = createContext<((heading: { title: string; subtitle?: string }) => void) | null>(null);

/**
 * The whole wizard: its heading (in the top bar), the steps, and the pane with its footer. How far it has got is the
 * step list's to say (each step ticked as it is done, the count in the step's own "Step 2 of 3"): the thin bar that
 * stood over the step, in a row of its own with "Continue later", is gone, and "Continue later" is in the footer.
 */
export function WizardFrame({
  title,
  subtitle,
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
  subtitle?: string;
  /** (taken, and no longer drawn: the step list shows it) */
  progress?: { done: number; total: number };
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
  const putHeading = useContext(SetupHeading);
  useEffect(() => putHeading?.({ title, subtitle }), [putHeading, title, subtitle]);
  return (
    <div className="setup-page">
      {!putHeading && (
        <div className="setup-hero">
          <div className="setup-hero-text">
            <h1>{title}</h1>
            {subtitle && <p>{subtitle}</p>}
          </div>
        </div>
      )}
      <div className="setup-layout">
        <StepRail items={rail} current={current} onPick={onPick} label={railLabel} />
        <div className="setup-main">
          <WizardLeave.Provider value={onLeave}>
            <section className={flush ? 'setup-pane is-flush' : 'setup-pane'}>{children}</section>
            {footer ?? (
              // (a step that draws no footer of its own, a plugin's screen: the way out stays in reach)
              <WizardLeaveFallback flush={flush} />
            )}
          </WizardLeave.Provider>
        </div>
      </div>
    </div>
  );
}

/** Nothing where the pane draws its own footer inside (a plugin's wizard as a step, which has "Continue later" in its own). */
function WizardLeaveFallback({ flush }: { flush?: boolean }) {
  const leave = useContext(WizardLeave);
  if (flush || !leave) return null;
  return (
    <footer className="setup-foot">
      <button type="button" className="btn btn-ghost setup-leave" onClick={leave} title="Leave setup: it keeps its place">
        Continue later
      </button>
    </footer>
  );
}

// ---------------------------------------------------------------------------
// Requirement cards
// ---------------------------------------------------------------------------

/** A service (OAIY Voice) the step needs: installed through the services route, with its progress. */
export function ServiceCard({
  id,
  why,
  services,
  onChanged,
  onComputer,
}: {
  id: string;
  why?: string;
  services: ServiceSnapshot[] | null;
  onChanged: () => void;
  /** The paired computer (OAIY Network) whose service this computer uses in place of its own, by its name. */
  onComputer?: string | null;
}) {
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
  if (onComputer) status = <span className="badge badge-ok">On {onComputer} (OAIY Network)</span>;
  else if (!services) status = <span className="form-hint">Checking…</span>;
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

/**
 * A model's download size, the GPU memory it needs (on each GPU, for one that takes several), whether that fits the
 * largest GPU (`gpuGb`; not said for a model of several GPUs, as only the largest is known), the RAM it takes, and its
 * license.
 */
function modelFacts(m: EngineCatalogModel, gpuGb?: number): string {
  const several = !!m.gpuCount && m.gpuCount > 1;
  const fit = gpuGb && m.vramGb && !several ? (m.vramGb <= gpuGb ? 'fits your GPU' : `more than your GPU’s ${gpuGb} GB`) : null;
  return [
    m.sizeGb ? `${m.sizeGb} GB download` : null,
    m.vramGb ? `needs ${m.vramGb} GB of GPU memory${several ? ` on each of ${m.gpuCount} GPUs` : ''}` : null,
    fit,
    m.ramGb ? `${m.ramGb} GB of RAM` : null,
    m.license,
  ]
    .filter(Boolean)
    .join(' · ');
}

/**
 * A download as it goes: a bar, how far, and how fast. The engines give the speed in bytes a second, so it is put in
 * a unit here, as the sizes beside it are (it was printed as it came with "MB/s" after it: "24117248.3 MB/s").
 */
export function DownloadProgress({ name, dl }: { name: string; dl: EngineDownload }) {
  return (
    <span className="setup-req-progress">
      <span className="setup-progress" role="progressbar" aria-label={`Downloading ${name}`} aria-valuenow={dl.total ? Math.round((dl.done / dl.total) * 100) : undefined}>
        <i className={dl.total ? undefined : 'is-indeterminate'} style={dl.total ? { width: `${Math.min(100, (dl.done / dl.total) * 100)}%` } : undefined} />
      </span>
      <small className="setup-mono">
        {dl.status === 'queued' ? 'Waiting to start…' : dl.status === 'adding' ? 'Adding it to Engines…' : `${formatBytes(dl.done)} of ${formatBytes(dl.total)}`}
        {dl.speed ? ` · ${formatSpeed(dl.speed)}` : ''}
        {dl.filesTotal && dl.filesTotal > 1 ? ` · file ${Math.min((dl.filesDone ?? 0) + 1, dl.filesTotal)} of ${dl.filesTotal}` : ''}
        {dl.reused ? ` · ${formatBytes(dl.reused)} found on this computer` : ''}
        {dl.lan ? ` · ${formatBytes(dl.lan)} from ${(dl.lanFrom ?? []).join(', ') || 'a paired computer'}` : ''}
      </small>
      {/* (looking for its files on this computer, or checking one: said, as it takes a while for a large file) */}
      {dl.file && /^(checking|looking)/.test(dl.file) && <small className="form-hint">{dl.file[0].toUpperCase() + dl.file.slice(1)}…</small>}
    </span>
  );
}

/** What happens to a file of a model being got, in words. */
function planWhy(status: EnginePlan['files'][number]['status'], from?: string | null): string {
  if (status === 'download') return 'to download';
  if (status === 'here') return 'in place';
  if (status === 'network') return `from ${from ?? 'a paired computer'} (OAIY Network)`;
  return `on this computer${from ? `: ${from}` : ''}`;
}

/**
 * What getting a model takes, before it is got: the gigabytes to download, those this computer has already (another
 * model's text encoder or VAE, a copy on another drive: used where they are, never downloaded twice), and for a
 * gated model, its licence and a Hugging Face token first. Each part listed under it.
 */
export function DownloadPlan({ plan }: { plan: EnginePlan }) {
  const short = (path: string) => path.split('/').slice(-2).join('/');
  const tight = plan.free != null && plan.downloadBytes + (1 << 30) > plan.free;
  return (
    <div className="download-plan" data-testid="download-plan">
      <span className="download-plan-line">
        <Download size={13} aria-hidden /> <strong>{formatBytes(plan.downloadBytes)}</strong> to download
        {plan.elsewhereBytes > 0 && (
          <span className="download-plan-have">
            <Check size={12} aria-hidden /> {formatBytes(plan.elsewhereBytes)} on this computer already, used where it is
          </span>
        )}
        {(plan.networkBytes ?? 0) > 0 && (
          <span className="download-plan-have">
            <Network size={12} aria-hidden /> {formatBytes(plan.networkBytes)} from {[...new Set(plan.files.filter((f) => f.status === 'network').map((f) => f.from))].join(', ')} over OAIY Network
          </span>
        )}
        {plan.hereBytes > 0 && <span className="download-plan-have">{formatBytes(plan.hereBytes)} in place</span>}
      </span>
      {plan.gated && !plan.token && (
        <span className="card-warn">
          <TriangleAlert size={12} aria-hidden /> {plan.name} is gated: accept its licence on huggingface.co, then add a Hugging Face token in Settings → Storage.
        </span>
      )}
      {tight && (
        <span className="card-warn">
          <TriangleAlert size={12} aria-hidden /> The drive has {formatBytes(plan.free)} free: make room first.
        </span>
      )}
      {plan.files.length > 1 && (
        <details className="download-plan-parts">
          <summary>Its {plan.files.length} files</summary>
          <ul>
            {/* (its weights, text encoder and VAE by name; its small files, settings and tokenizers, in a line each by
                what happens to them) */}
            {plan.files
              .filter((f) => (f.size ?? 0) >= 8e6)
              .map((f) => (
                <li key={f.path} className={`is-${f.status}`}>
                  <code>{short(f.path)}</code>
                  <span>{f.size ? formatBytes(f.size) : ''}</span>
                  <span className="download-plan-status">{planWhy(f.status, f.from)}</span>
                </li>
              ))}
            {[...new Set(plan.files.filter((f) => (f.size ?? 0) < 8e6).map((f) => f.status))].map((status) => {
              const these = plan.files.filter((f) => (f.size ?? 0) < 8e6 && f.status === status);
              const bytes = these.reduce((n, f) => n + (f.size ?? 0), 0);
              return (
                <li key={`small-${status}`} className={`is-${status}`} title={these.map((f) => f.path).join('\n')}>
                  <code>
                    {these.length} small file{these.length === 1 ? '' : 's'} (settings, tokenizer…)
                  </code>
                  <span>{bytes ? formatBytes(bytes) : ''}</span>
                  <span className="download-plan-status">{planWhy(status)}</span>
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </div>
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
        A language model in a .gguf file you already have (of a model split into files, the first). After you add it, OAIY says whether the Agent can use its tools with it.
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
 * What the Agent can do with a model file the person added, from what the engine reported as it added it (`tools`):
 * said of that file by its name, in words that name no model and no family of models.
 */
export function addedFileLine(name: string, added: Pick<AddedModelFile, 'tools'>): string {
  return added.tools
    ? `The Agent can use its tools with ${name}, so it can work OAIY for you as well as chat.`
    : `The Agent chats with ${name} but cannot use its tools with it, so it can answer but not work OAIY for you. Choose another language model in Engines to change that.`;
}

/**
 * The one thing to do next for the model offered, for a page that puts it on its own button (the first-run wizard's
 * footer): download it, by its name and size, or nothing while it downloads (how far, as a percentage when known).
 */
export interface EngineAction {
  kind: 'download' | 'downloading';
  /** "Download <name> (18.4 GB)", or "Downloading… 34%". */
  label: string;
  busy: boolean;
  run: () => void;
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
  onAction,
}: {
  group: string;
  why?: string;
  catalog: EngineCatalog | null;
  /** The largest GPU's memory, to say whether each model fits it. */
  gpuGb?: number;
  onChanged: () => void;
  onOpenEngines: () => void;
  /**
   * Given, the page draws the download button itself (its footer's primary button) and is told what it does; the
   * card then shows the models and the progress, and no Download button of its own.
   */
  onAction?: (action: EngineAction | null) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [added, setAdded] = useState<AddedModelFile | null>(null);
  const [plan, setPlan] = useState<EnginePlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
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
  // What getting the model offered takes (its files here already, elsewhere on this computer, or to download).
  const planFor = offer && !offer.installed && catalog?.running && !(dl && ACTIVE.has(dl.status)) ? offer.id : null;
  useEffect(() => {
    setPlan(null);
    setPlanError(null);
    if (!planFor) return;
    let gone = false;
    engines
      .plan(planFor)
      .then((p) => !gone && setPlan(p))
      .catch((e: unknown) => !gone && setPlanError(errorText(e)));
    return () => {
      gone = true;
    };
  }, [planFor]);

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

  // The one thing to do for the model offered: download it (said with its name, and what there is to download), or
  // wait while it downloads. Told to the page that draws the button (`onAction`), and drawn here otherwise.
  const downloading = !!dl && ACTIVE.has(dl.status);
  const canDownload = !!catalog?.running && !chosen && !!offer && !offer.installed && !downloading;
  const downloadLabel = !offer
    ? ''
    : dl?.status === 'failed'
      ? `Download ${offer.name} again`
      : plan && planFor === plan.id
        ? plan.downloadBytes > 0
          ? `Download ${offer.name} (${formatBytes(plan.downloadBytes)})`
          : `Add ${offer.name} (all here)`
        : `Download ${offer.name}${offer.sizeGb ? ` (${offer.sizeGb} GB)` : ''}`;
  const pct = downloading && dl?.total ? Math.min(100, Math.floor((dl.done / dl.total) * 100)) : null;
  const actionKind: EngineAction['kind'] | null = !catalog?.running || chosen || !offer ? null : downloading ? 'downloading' : canDownload ? 'download' : null;
  const actionLabel = actionKind === 'downloading' ? (pct === null ? 'Downloading…' : `Downloading… ${pct}%`) : downloadLabel;
  const run = useRef(download);
  run.current = download;
  useEffect(() => {
    onAction?.(actionKind ? { kind: actionKind, label: actionLabel, busy, run: () => void run.current() } : null);
  }, [onAction, actionKind, actionLabel, busy]);
  // (gone with the card: the page's button goes back to its own "Next")
  useEffect(() => () => onAction?.(null), [onAction]);

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
          <span className="badge badge-ok">Chosen in Engines</span> <strong title={chosen}>{modelName(catalog, chosen)}</strong>
        </span>
        {/* What the engine said of the file as it added it (`tools`): said in words that name no model. */}
        {added && added.name === chosen && (
          <p className={added.tools ? 'form-hint setup-chat-only' : 'card-warn setup-chat-only'}>{addedFileLine(modelName(catalog, chosen), added)}</p>
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
        {plan && planFor === plan.id && <DownloadPlan plan={plan} />}
        {!plan && planError && <p className="card-warn">{planError}</p>}
        {!plan && !planError && planFor && offer.gated && <p className="form-hint">It is gated: its licence is accepted on huggingface.co first, with a Hugging Face token in Settings → Storage.</p>}
        {/* Under the list, naming the model it gets: beside the card it lined up with whichever model sat in the middle.
            (Not here where the page has the button, the first-run wizard's footer: one button, always in view.) */}
        {canDownload && !onAction && (
          <div className="setup-offer-action">
            <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void download()}>
              {busy ? <Loader2 size={14} className="spin" /> : <Download size={14} />} {downloadLabel}
            </button>
          </div>
        )}
        {dl && ACTIVE.has(dl.status) && <DownloadProgress name={offer.name} dl={dl} />}
        {dl?.status === 'failed' && dl.error && <p className="card-warn">{dl.error}</p>}
        {offer.installed && (
          <p className="form-hint">
            Downloaded
            {dl?.reused ? `: ${formatBytes(dl.reused)} of it was on this computer already, used where it is` : ''}
            {dl?.lan ? `${dl?.reused ? ',' : ':'} ${formatBytes(dl.lan)} came from ${(dl.lanFrom ?? []).join(', ')} over OAIY Network` : ''}. Choose it in Engines to use it.
          </p>
        )}
        {dl?.status === 'done' && (dl.added ?? []).filter((a) => a.includes('(needs ')).map((a) => (
          <p key={a} className="card-warn">
            <TriangleAlert size={12} aria-hidden /> Added as {a}: add what it still needs in Engines.
          </p>
        ))}
        {ownFile && <OwnModelFile onAdded={addedFile} />}
      </div>
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
