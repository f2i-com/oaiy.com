import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Check, Cpu, FolderOpen, Link2, Loader2, Package, Plug, Puzzle, ShieldCheck, Sparkles, TriangleAlert } from 'lucide-react';
import {
  aiProviders,
  appConfig,
  bridge,
  codex,
  engines,
  isTauri,
  link,
  nodeRuntime,
  pairing,
  plugins as pluginsApi,
  services as servicesApi,
  setup as setupApi,
  type CatalogPlugin,
  type FirstRunState,
  type PluginRecord,
} from './api';
import PluginWizard from './PluginSetup';
import type { PluginNavTarget } from './PluginScreenPage';
import {
  engineReady,
  firstRunPosition,
  firstRunProgress,
  firstRunSteps,
  readSetup,
  type FirstRunStep,
} from './setupFlow';
import { AiSourceChoice, EngineModelCard, errorText, RetryLine, ServiceCard, StepFooter, StepHeader, usePoll, WizardFrame } from './SetupParts';
import { useToast } from './Toasts';
import { refetchModules } from './useModules';
import { setSetupState, useSetupState } from './useSetupState';

/**
 * The setup page: the first-run wizard, or one plugin's own wizard.
 *
 * First run: welcome; the engine (the language model chosen in Engines, or
 * the catalog's recommended one when none is; OAIY Voice; or an AI source
 * instead); plugins to install and set up; each chosen plugin's own setup;
 * connecting an app; done. It keeps its place on the desktop, so it can be
 * left and resumed, and every step can be skipped.
 */

export type SetupNav = PluginNavTarget | 'connections' | 'settings';

interface Props {
  /** A plugin's own wizard; null: the first-run wizard. */
  pluginId: string | null;
  /** Leave the page (it keeps its place). */
  onExit: () => void;
  onNavigate: (view: SetupNav) => void;
}

export default function SetupPage({ pluginId, onExit, onNavigate }: Props) {
  if (pluginId) {
    return <PluginWizard key={pluginId} pluginId={pluginId} layout="page" onFinished={onExit} onLeave={onExit} onNavigate={onNavigate} />;
  }
  return <FirstRunWizard onExit={onExit} onNavigate={onNavigate} />;
}

function FirstRunWizard({ onExit, onNavigate }: { onExit: () => void; onNavigate: (view: SetupNav) => void }) {
  const toast = useToast();
  const state = useSetupState();
  const [plugins, refreshPlugins] = usePoll(() => pluginsApi.list().then((s) => s.plugins), 3000);
  const [catalog, refreshCatalog] = usePoll(() => engines.catalog(), 3000);
  const [services, refreshServices] = usePoll(() => servicesApi.list().then((s) => s.services), 3000);
  const [providers] = usePoll(() => aiProviders.list().then((r) => r.providers), 5000);
  const [codexOn] = usePoll(() => codex.status().then((s) => s.connected), 5000);
  const [paired] = usePoll(() => pairing.paired().then((p) => p.paired), 4000);
  const [linked] = usePoll(() => link.status().then((l) => l.linked), 8000);

  const guide = useMemo(
    () => ({ codexConnected: codexOn === true, runtime: null, providers, services, plugins, connected: paired }),
    [codexOn, providers, services, plugins, paired],
  );
  const steps = useMemo(
    () => firstRunSteps({ state, plugins, catalog, guide, linked: linked === true }),
    [state, plugins, catalog, guide, linked],
  );

  // Opens where it was left; after that it follows the person.
  const [currentId, setCurrentId] = useState<string | null>(null);
  useEffect(() => {
    if (currentId === null && state) setCurrentId(steps[firstRunPosition(steps, state.firstRun.position)].id);
  }, [currentId, state, steps]);
  const index = Math.max(0, steps.findIndex((s) => s.id === currentId));
  const current = steps[index];

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Save the first-run part of the record (its place, skips, choices). */
  const save = useCallback(
    async (change: Partial<FirstRunState>) => {
      if (!state) return;
      const { migrated: _migrated, ...base } = state.firstRun;
      setSetupState(await setupApi.putFirstRun({ ...base, ...change }));
    },
    [state],
  );

  const go = (i: number) => {
    const target = steps[Math.max(0, Math.min(steps.length - 1, i))];
    if (!target) return;
    setCurrentId(target.id);
    setError(null);
    void save({ position: target.id }).catch(() => undefined);
  };
  const skip = () => {
    const skipped = [...new Set([...(state?.firstRun.skipped ?? []), current.id])];
    const target = steps[Math.min(steps.length - 1, index + 1)];
    setCurrentId(target.id);
    void save({ skipped, position: target.id }).catch((e) => setError(errorText(e)));
  };
  const finish = async () => {
    setBusy(true);
    try {
      await save({ finished: true, position: 'done' });
      toast.push({ kind: 'success', title: 'OAIY is set up', body: 'Setup is in Settings whenever you want to run it again.' });
      onExit();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleChosen = async (id: string, on: boolean) => {
    const chosen = new Set(state?.firstRun.chosenPlugins ?? []);
    if (on) chosen.add(id);
    else chosen.delete(id);
    await save({ chosenPlugins: [...chosen] }).catch((e) => setError(errorText(e)));
  };

  const installed = async (id: string) => {
    await Promise.all([refreshPlugins(), refetchModules()]);
    await toggleChosen(id, true);
  };

  if (!state || !current) {
    return (
      <div className="setup-page">
        <p className="form-hint">Loading setup…</p>
      </div>
    );
  }

  const progress = firstRunProgress(steps);
  const isPlugin = current.id.startsWith('plugin:') && !!current.pluginId;
  const kicker = `Step ${index + 1} of ${steps.length}`;
  const nextStep = () => go(index + 1);

  let content: ReactNode;
  let footer: ReactNode = null;
  const status = error ? <span className="card-warn">{error}</span> : current.state === 'done' ? <span className="setup-ok"><Check size={13} /> Done</span> : current.optional ? 'Optional' : null;
  switch (current.id) {
    case 'welcome':
      content = <Welcome kicker={kicker} />;
      footer = <StepFooter status={status} onNext={nextStep} nextLabel="Get started" />;
      break;
    case 'engine': {
      const ready = engineReady({ catalog, guide });
      content = (
        <>
          <StepHeader
            kicker={kicker}
            title="The engine"
            state={current.state}
            description="OAIY runs its models on this computer. The language model is the one you choose in Engines; OAIY Voice hears and speaks on calls."
          />
          <div className="setup-reqs">
            <EngineModelCard group="llm" catalog={catalog} onChanged={() => void refreshCatalog()} onOpenEngines={() => onNavigate('engines')} />
            <ServiceCard id="oaiy-voice" why="Hears callers and speaks the replies, on this computer. Needed for phone calls; about 4.3 GB." services={services} onChanged={() => void refreshServices()} />
          </div>
          <details className="setup-more-source" open={!ready && catalog?.running === false ? true : undefined}>
            <summary>Or use an AI source instead: ChatGPT, a provider’s API key, or a local model server</summary>
            <AiSourceChoice onNavigate={(v) => onNavigate(v)} />
          </details>
        </>
      );
      footer = <StepFooter onBack={() => go(index - 1)} status={status ?? (ready ? null : 'Waiting for a language model')} onSkip={ready ? undefined : skip} onNext={nextStep} nextDisabled={!ready} />;
      break;
    }
    case 'plugins':
      content = (
        <PluginsStep
          kicker={kicker}
          state={current.state}
          chosen={state.firstRun.chosenPlugins}
          plugins={plugins}
          onToggle={(id, on) => void toggleChosen(id, on)}
          onInstalled={(id) => void installed(id)}
        />
      );
      footer = (
        <StepFooter
          onBack={() => go(index - 1)}
          status={status ?? (state.firstRun.chosenPlugins.length ? `${state.firstRun.chosenPlugins.length} chosen` : 'None chosen: plugins can be added any time')}
          onNext={() => (state.firstRun.chosenPlugins.length ? nextStep() : skip())}
        />
      );
      break;
    case 'connect':
      content = <ConnectStep kicker={kicker} state={current.state} paired={paired} linked={linked === true} onNavigate={onNavigate} />;
      footer = <StepFooter onBack={() => go(index - 1)} status={status} onSkip={current.state === 'done' ? undefined : skip} onNext={nextStep} />;
      break;
    case 'done':
      content = <DoneStep kicker={kicker} steps={steps} onPick={(i) => go(i)} />;
      footer = <StepFooter onBack={() => go(index - 1)} status={status} onNext={() => void finish()} nextLabel="Finish setup" busy={busy} />;
      break;
    default:
      if (isPlugin) {
        content = (
          <PluginWizard
            key={current.pluginId}
            pluginId={current.pluginId!}
            layout="embedded"
            onFinished={nextStep}
            onLeave={onExit}
            onBackOut={() => go(index - 1)}
            onSkipOut={skip}
            onNavigate={onNavigate}
          />
        );
      }
  }

  const flush = isPlugin;
  return (
    <WizardFrame
      title="Set up OAIY"
      subtitle="The engine, your plugins and their devices, and the apps that use them. Leave any time: setup keeps its place."
      progress={progress}
      rail={steps.map((s) => ({ id: s.id, title: s.title, hint: s.hint, state: s.state, optional: s.optional }))}
      current={index}
      onPick={go}
      railLabel="Setup steps"
      onLeave={onExit}
      flush={flush}
      footer={footer}
    >
      {content}
    </WizardFrame>
  );
}

// ---------------------------------------------------------------------------
// First-run steps
// ---------------------------------------------------------------------------

function Welcome({ kicker }: { kicker: string }) {
  const [runtime] = usePoll(() => bridge.status(), 5000);
  const [installing, setInstalling] = useState(false);
  const node = runtime?.nodeRuntime;
  const parts: Array<{ icon: ReactNode; title: string; text: string }> = [
    { icon: <Cpu size={17} />, title: 'The engine', text: 'A language model on this computer, the one you choose in Engines, and OAIY Voice for calls.' },
    { icon: <Puzzle size={17} />, title: 'Plugins', text: 'An AI receptionist for your phone, and more. Each one sets itself up step by step, down to pairing its device.' },
    { icon: <Link2 size={17} />, title: 'Your apps', text: 'Let FormLogic, or another app you approve, run flows on this computer.' },
  ];
  return (
    <div className="setup-welcome">
      <StepHeader kicker={kicker} title="Welcome to OAIY" description="Orchestrate AI Yourself: your models, your devices and your flows, on this computer. A few steps get it going; each one checks itself once it is true, and any can be skipped and done later." />
      <ul className="setup-welcome-parts">
        {parts.map((p) => (
          <li key={p.title}>
            <span className="setup-req-icon" aria-hidden>
              {p.icon}
            </span>
            <span>
              <strong>{p.title}</strong>
              <small>{p.text}</small>
            </span>
          </li>
        ))}
      </ul>
      {runtime && !runtime.ready && (
        <div className="setup-note" role="status">
          <TriangleAlert size={13} />
          <span>Flows cannot run on this computer yet: {runtime.flowRuntime.detail ?? 'the flow runtime is not ready.'}</span>
          {node && !node.available && (
            <button
              type="button"
              className="btn-tiny"
              disabled={installing || node.installing}
              onClick={() => {
                setInstalling(true);
                void nodeRuntime.install().finally(() => setInstalling(false));
              }}
            >
              {node.installing || installing ? 'Installing Node…' : `Install Node ${node.installsVersion}`}
            </button>
          )}
        </div>
      )}
      <p className="setup-trust">
        <ShieldCheck size={13} /> Everything here stays on this computer. Keys, conversations and recordings are kept on this device.
      </p>
    </div>
  );
}

function PluginsStep({
  kicker,
  state,
  chosen,
  plugins,
  onToggle,
  onInstalled,
}: {
  kicker: string;
  state: FirstRunStep['state'];
  chosen: string[];
  plugins: PluginRecord[] | null;
  onToggle: (id: string, on: boolean) => void;
  onInstalled: (id: string) => void;
}) {
  const [catalog, refreshCatalog, catalogError] = usePoll(() => setupApi.catalog().then((c) => c.plugins), 10000);
  const others = (plugins ?? []).filter((p) => !(catalog ?? []).some((c) => c.id === p.id));
  const installedNow = async (id: string) => {
    await refreshCatalog();
    onInstalled(id);
  };
  return (
    <div className="setup-plugins">
      <StepHeader
        kicker={kicker}
        title="Plugins"
        state={state}
        description="Plugins bring devices and services to OAIY, such as your business phone. Tick the ones to set up: each one's own setup follows, step by step."
      />
      {catalogError && !catalog && <RetryLine message={`The plugin list could not be read: ${catalogError}`} onRetry={() => void refreshCatalog()} />}
      <div className="setup-plugin-list">
        {(catalog ?? []).map((c) => (
          <CatalogCard key={c.id} entry={c} record={plugins?.find((p) => p.id === c.id) ?? null} chosen={chosen.includes(c.id)} onToggle={onToggle} onInstalled={installedNow} />
        ))}
        {others.map((p) => {
          const setup = readSetup(p);
          return (
            <div key={p.id} className={`setup-plugin${chosen.includes(p.id) ? ' is-chosen' : ''}`}>
              <label className="setup-plugin-tick">
                <input type="checkbox" checked={chosen.includes(p.id)} disabled={!setup} onChange={(e) => onToggle(p.id, e.target.checked)} />
                <span className="sr-only">Set up {p.manifest?.name ?? p.id}</span>
              </label>
              <div className="setup-plugin-body">
                <strong>
                  {p.manifest?.name ?? p.id} <span className="badge badge-ok">Installed{p.manifest?.version ? ` v${p.manifest.version}` : ''}</span>
                </strong>
                {p.manifest?.description && <p>{p.manifest.description}</p>}
                {!setup && <small className="form-hint">It has nothing to set up.</small>}
              </div>
            </div>
          );
        })}
      </div>
      <FolderInstall title="Install a plugin from a folder" hint="A plugin folder, .zip or .tar.gz on this computer. Installing a plugin installs code this computer runs: install only plugins you trust." onInstalled={installedNow} />
    </div>
  );
}

function CatalogCard({
  entry,
  record,
  chosen,
  onToggle,
  onInstalled,
}: {
  entry: CatalogPlugin;
  record: PluginRecord | null;
  chosen: boolean;
  onToggle: (id: string, on: boolean) => void;
  onInstalled: (id: string) => void;
}) {
  const installed = !!record || entry.installed;
  return (
    <div className={`setup-plugin${chosen ? ' is-chosen' : ''}`}>
      <label className="setup-plugin-tick">
        <input type="checkbox" checked={chosen} onChange={(e) => onToggle(entry.id, e.target.checked)} />
        <span className="sr-only">Set up {entry.name}</span>
      </label>
      <div className="setup-plugin-body">
        <strong>
          {entry.name}
          {installed ? (
            <span className="badge badge-ok">Installed{entry.installedVersion ? ` v${entry.installedVersion}` : ''}</span>
          ) : (
            <span className="badge badge-neutral">Not installed</span>
          )}
        </strong>
        <small className="setup-plugin-meta">
          {entry.plugin ?? entry.id}
          {entry.publisher ? ` · by ${entry.publisher}` : ''}
        </small>
        {entry.description && <p>{entry.description}</p>}
        {((entry.provides?.length ?? 0) > 0 || entry.needs) && (
          <div className="setup-chips">
            {entry.provides?.map((m) => (
              <span key={m} className="setup-chip">
                Brings the {m === 'phone' ? 'phone' : m === 'calendar' ? 'Calendar' : m}
              </span>
            ))}
            {entry.needs && <span className="setup-chip is-need">Needs: {entry.needs}</span>}
          </div>
        )}
        {!installed && chosen && (
          <FolderInstall compact initial={entry.source.path ?? ''} hint={entry.source.path ? `Found on this computer.` : entry.source.note} onInstalled={onInstalled} expectId={entry.id} />
        )}
        {!installed && !chosen && <small className="form-hint">Tick it to install it{entry.source.path ? ' (found on this computer)' : ' from its folder'}.</small>}
      </div>
    </div>
  );
}

/** Install from a folder (or a .zip or .tar.gz) on this computer, through the plugin install route. */
function FolderInstall({
  title,
  hint,
  initial = '',
  compact,
  expectId,
  onInstalled,
}: {
  title?: string;
  hint?: string;
  initial?: string;
  compact?: boolean;
  expectId?: string;
  onInstalled: (id: string) => void;
}) {
  const toast = useToast();
  const [path, setPath] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => setPath((p) => p || initial), [initial]);
  const browse = async () => {
    try {
      const picked = await appConfig.pickFolder();
      if (picked) setPath(picked);
    } catch (e) {
      setError(errorText(e));
    }
  };
  const install = async () => {
    const source = path.trim();
    if (!source) return;
    setBusy(true);
    setError(null);
    try {
      const out = await pluginsApi.install(source);
      if (expectId && out.id !== expectId) toast.push({ kind: 'info', title: `That folder held ${out.name}`, body: `Installed it as ${out.id}.` });
      toast.push({ kind: 'success', title: out.replaced ? `Updated ${out.name} to v${out.version}` : `Installed ${out.name} v${out.version}` });
      onInstalled(out.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className={compact ? 'setup-folder is-compact' : 'setup-folder'}>
      {title && <h3 className="section-title">{title}</h3>}
      <div className="setup-folder-row">
        <input type="text" placeholder="C:\path\to\plugin" value={path} aria-label="Plugin folder" onChange={(e) => setPath(e.target.value)} />
        {isTauri() && (
          <button type="button" className="btn" onClick={() => void browse()}>
            <FolderOpen size={14} /> Browse
          </button>
        )}
        <button type="button" className="btn btn-primary" disabled={busy || !path.trim()} onClick={() => void install()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Plug size={14} />} Install
        </button>
      </div>
      {hint && <small className="form-hint">{hint}</small>}
      {error && <p className="card-warn" role="alert">{error}</p>}
    </div>
  );
}

function ConnectStep({ kicker, state, paired, linked, onNavigate }: { kicker: string; state: FirstRunStep['state']; paired: Array<{ id: string; product: string; label?: string; origin?: string | null }> | null; linked: boolean; onNavigate: (v: SetupNav) => void }) {
  return (
    <div className="setup-connect">
      <StepHeader kicker={kicker} title="Connect an app" state={state} description="Optional. An app you approve, such as FormLogic, can run flows on this computer. It asks here, with a code to check, and you approve it." />
      <ol className="setup-instructions">
        <li>In FormLogic, open <strong>Connect your AI</strong> and choose OAIY desktop.</li>
        <li>Choose <strong>Connect</strong> there. A request with a code shows at the top of this page: approve it when the codes match.</li>
        <li>Back in FormLogic, choose your default provider and check the saved setup. Keep OAIY open while you use it.</li>
      </ol>
      <div className="setup-connected">
        <h3 className="section-title">Connected</h3>
        {paired === null ? (
          <p className="form-hint">Checking…</p>
        ) : paired.length === 0 && !linked ? (
          <p className="form-hint">Nothing yet.</p>
        ) : (
          <ul className="setup-perm-list">
            {linked && (
              <li>
                <Link2 size={15} aria-hidden />
                <span>
                  <strong>Your FormLogic account</strong>
                  <small>Linked: this computer can be used from your other devices.</small>
                </span>
              </li>
            )}
            {paired.map((a) => (
              <li key={a.id}>
                <ShieldCheck size={15} aria-hidden />
                <span>
                  <strong>{a.label ?? a.product}</strong>
                  {a.origin && <small className="setup-mono">{a.origin}</small>}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <p className="form-hint">
        To use this computer from another laptop or phone, link your FormLogic account in Connections.{' '}
        <button type="button" className="btn-tiny" onClick={() => onNavigate('connections')}>
          Open Connections
        </button>
      </p>
    </div>
  );
}

function DoneStep({ kicker, steps, onPick }: { kicker: string; steps: FirstRunStep[]; onPick: (i: number) => void }) {
  const left = steps.filter((s) => s.id !== 'done' && s.id !== 'welcome' && s.state !== 'done');
  return (
    <div className="setup-done">
      <StepHeader
        kicker={kicker}
        title={left.length ? 'Nearly there' : 'You’re set up'}
        description={left.length ? 'Finish now and come back to the rest any time: setup is in Settings, and the Overview keeps a card for what is left.' : 'OAIY is ready. Setup is in Settings whenever you want to run it again.'}
      />
      <ul className="setup-summary">
        {steps
          .filter((s) => s.id !== 'done' && s.id !== 'welcome')
          .map((s) => (
            <li key={s.id} className={`is-${s.state}`}>
              <span className="setup-summary-mark" aria-hidden>
                {s.state === 'done' ? <Check size={13} strokeWidth={2.6} /> : s.id === 'engine' ? <Sparkles size={13} /> : <Package size={13} />}
              </span>
              <span>
                <strong>{s.title}</strong>
                <small>{s.state === 'done' ? 'Done' : s.state === 'skipped' ? 'Skipped for now' : s.optional ? 'Optional, not done' : 'Not done yet'}</small>
              </span>
              {s.state !== 'done' && (
                <button type="button" className="btn-tiny" onClick={() => onPick(steps.indexOf(s))}>
                  Go to it
                </button>
              )}
            </li>
          ))}
      </ul>
    </div>
  );
}
