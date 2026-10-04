import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Check, FolderOpen, Link2, Loader2, Package, Plug, ShieldCheck, Sparkles } from 'lucide-react';
import {
  agentPreferences,
  appConfig,
  codex,
  engineRecommendation,
  engines,
  isTauri,
  link,
  pairing,
  plugins as pluginsApi,
  setup as setupApi,
  type AgentModelSource,
  type CatalogPlugin,
  type FirstRunState,
  type PluginRecord,
} from './api';
import { askAgent, useControlSettings } from './AgentAccess';
import type { ProviderPick } from './AgentProviderPicker';
import PluginWizard from './PluginSetup';
import type { PluginNavTarget } from './PluginScreenPage';
import {
  aiReady,
  currentStepId,
  firstRunPosition,
  firstRunProgress,
  firstRunSteps,
  initialAiChoice,
  inTheRest,
  localModelLine,
  readSetup,
  recommendationOrFallback,
  type FirstRunStep,
} from './setupFlow';
import { AgentStep, HandoffStep, modelLineText, WelcomeStep, YourAiStep } from './SetupEssentials';
import { answered, errorText, RetryLine, StepFooter, StepHeader, usePoll, WizardFrame } from './SetupParts';
import { useToast } from './Toasts';
import { useLiveSetup } from './useLiveSetup';
import { refetchModules } from './useModules';
import { setSetupState, useSetupState } from './useSetupState';

/**
 * The setup page: the first-run wizard, or one plugin's own wizard.
 *
 * First run is the essentials: welcome; your AI (a language model on this
 * computer, or ChatGPT, the recommended one first, or an AI provider such as
 * LM Studio); what the Agent may change
 * (one switch); then "Continue with the Agent", which hands the person to the
 * Agent to set up the rest in a chat. "Set up the rest myself" goes on step by
 * step instead: plugins to install, each chosen plugin's own setup, connecting
 * an app, done. It keeps its place on the desktop, so it can be left and
 * resumed, and every step can be skipped.
 */

export type SetupNav = PluginNavTarget | 'connections' | 'settings';

interface Props {
  /** A plugin's own wizard; null: the first-run wizard. */
  pluginId: string | null;
  /** Open at this step (a navigation from the desktop asked for it). */
  step?: string | null;
  /** Leave the page (it keeps its place). */
  onExit: () => void;
  onNavigate: (view: SetupNav) => void;
}

export default function SetupPage({ pluginId, step, onExit, onNavigate }: Props) {
  if (pluginId) {
    return <PluginWizard key={pluginId} pluginId={pluginId} initialStep={step ?? undefined} layout="page" onFinished={onExit} onLeave={onExit} onNavigate={onNavigate} />;
  }
  return <FirstRunWizard initialStep={step ?? null} onExit={onExit} onNavigate={onNavigate} />;
}

function FirstRunWizard({ initialStep, onExit, onNavigate }: { initialStep: string | null; onExit: () => void; onNavigate: (view: SetupNav) => void }) {
  const toast = useToast();
  const state = useSetupState();
  const [plugins, refreshPlugins] = usePoll(() => pluginsApi.list().then((s) => s.plugins), 3000);
  const [catalog, refreshCatalog, catalogError] = usePoll(() => engines.catalog(), 3000);
  const [codexStatus, refreshCodex] = usePoll(() => codex.status(), 5000);
  const [paired] = usePoll(() => pairing.paired().then((p) => p.paired), 4000);
  const [linked] = usePoll(() => link.status().then((l) => l.linked), 8000);
  const [recAnswer, , recError] = usePoll(() => engineRecommendation().then((v) => ({ v })), 20000);
  const [prefsAnswer, refreshPrefs, prefsError] = usePoll(() => agentPreferences.get().then((v) => ({ v })), 10000);
  const control = useControlSettings();
  const live = useLiveSetup(plugins, state);
  const codexOn = codexStatus?.connected === true;
  const prefs = answered(prefsAnswer, prefsError);

  // The recommendation, or (a desktop without it) local if Engines has a model chosen.
  const recommendation = answered(recAnswer, recError);
  const rec = recommendation ? recommendation : recommendation === null && (catalog || catalogError) ? recommendationOrFallback(null, catalog, codexOn) : null;

  /** "Set up the rest myself" was chosen (or a navigation, or the record's place, is in the rest). */
  const asked = initialStep ? currentStepId(initialStep) : null;
  const [rest, setRest] = useState(() => inTheRest(asked));
  const [aiChoice, setAiChoice] = useState<AgentModelSource | null>(null);
  const choice = aiChoice ?? (rec && prefs !== undefined ? initialAiChoice(rec, prefs, catalog) : null);
  // The provider and model chosen here; until then, the ones the Agent is on.
  const [picked, setPicked] = useState<ProviderPick | null>(null);
  const onProvider = prefs?.model.source === 'provider' && prefs.model.provider && prefs.model.model ? prefs.model : null;
  const providerPick = picked ?? (onProvider ? { provider: onProvider.provider!, model: onProvider.model!, name: prefs?.providerName ?? onProvider.provider! } : null);

  const guide = useMemo(() => ({ codexConnected: codexOn, runtime: null, providers: null, services: null, plugins, connected: paired }), [codexOn, plugins, paired]);
  const steps = useMemo(
    () => firstRunSteps({ state, plugins, catalog, guide, linked: linked === true, prefs, rest, live }),
    [state, plugins, catalog, guide, linked, prefs, rest, live],
  );

  // Opens where it was left (or where a navigation asked); after that it follows the person.
  const [currentId, setCurrentId] = useState<string | null>(null);
  useEffect(() => {
    if (currentId !== null || !state) return;
    // Left in the rest: it stays shown when the person goes back to an essential step.
    if (inTheRest(state.firstRun.position)) setRest(true);
    setCurrentId(asked && steps.some((s) => s.id === asked) ? asked : steps[firstRunPosition(steps, state.firstRun.position)].id);
  }, [currentId, state, steps, asked]);
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

  const goTo = (id: string) => {
    if (inTheRest(id)) setRest(true);
    setCurrentId(id);
    setError(null);
    void save({ position: id }).catch(() => undefined);
  };
  const go = (i: number) => {
    const target = steps[Math.max(0, Math.min(steps.length - 1, i))];
    if (target) goTo(target.id);
  };
  const skip = () => {
    const skipped = [...new Set([...(state?.firstRun.skipped ?? []).map(currentStepId), current.id])];
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

  /** The Agent's model follows the choice made here (a desktop that keeps none has nothing to set). */
  const commitAi = async (source: AgentModelSource) => {
    if (!prefs) return;
    if (source === 'provider') {
      const pick = providerPick;
      if (!pick || (prefs.model.source === 'provider' && prefs.model.provider === pick.provider && prefs.model.model === pick.model)) return;
      await agentPreferences.set({ model: { source: 'provider', provider: pick.provider, model: pick.model } });
    } else {
      if (prefs.model.source === source) return;
      await agentPreferences.set({ model: { source } });
    }
    await refreshPrefs();
  };
  const signedIn = () => {
    void refreshCodex();
    setAiChoice('chatgpt');
    // Signed in to ChatGPT here: the Agent thinks with it (Codex's default model).
    if (prefs && prefs.model.source !== 'chatgpt')
      void agentPreferences
        .set({ model: { source: 'chatgpt' } })
        .then(() => refreshPrefs())
        .catch((e) => setError(`The Agent is not set to use ChatGPT: ${errorText(e)}`));
  };

  /** The essentials are done: the Agent sets up the rest, in a "Set up OAIY" conversation. */
  const continueWithAgent = async () => {
    setBusy(true);
    try {
      await save({ finished: true, position: 'handoff' });
    } catch (e) {
      setError(errorText(e));
      setBusy(false);
      return;
    }
    setBusy(false);
    toast.push({ kind: 'success', title: 'The essentials are set up', body: 'The Agent takes it from here. Setup is in Settings whenever you want it again.' });
    onNavigate('agent');
    void askAgent('setupWithAgent');
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
  const ready = aiReady({ catalog, prefs, codexConnected: codexOn });
  const line = localModelLine(rec, catalog);

  let content: ReactNode;
  let footer: ReactNode = null;
  const status = error ? <span className="card-warn">{error}</span> : current.state === 'done' ? <span className="setup-ok"><Check size={13} /> Done</span> : current.optional ? 'Optional' : null;
  switch (current.id) {
    case 'welcome':
      content = <WelcomeStep kicker={kicker} />;
      footer = <StepFooter status={status} onNext={nextStep} nextLabel="Get started" />;
      break;
    case 'ai': {
      const chosenReady = choice === 'engine' ? line.kind === 'chosen' : choice === 'chatgpt' ? codexOn : choice === 'provider' ? !!providerPick : false;
      content = (
        <YourAiStep
          kicker={kicker}
          state={current.state}
          rec={rec}
          catalog={catalog}
          codex={codexStatus}
          choice={choice}
          onChoose={setAiChoice}
          onCatalogChanged={() => void refreshCatalog()}
          onSignedIn={signedIn}
          onOpenEngines={() => onNavigate('engines')}
          providerPick={providerPick}
          onProviderPick={prefs ? setPicked : undefined}
        />
      );
      const waiting =
        choice === 'chatgpt' ? 'Sign in with ChatGPT to use it' : choice === 'provider' ? 'Connect a provider and choose its model' : 'Waiting for a language model in Engines';
      footer = (
        <StepFooter
          onBack={() => go(index - 1)}
          status={status ?? (chosenReady ? null : waiting)}
          onSkip={chosenReady ? undefined : skip}
          onNext={() => {
            if (!choice) return;
            setBusy(true);
            void commitAi(choice)
              .then(nextStep, (e) => setError(`The Agent’s model was not saved: ${errorText(e)}`))
              .finally(() => setBusy(false));
          }}
          nextDisabled={!chosenReady}
          busy={busy}
        />
      );
      break;
    }
    case 'agent':
      content = <AgentStep kicker={kicker} state={current.state} control={control} />;
      footer = <StepFooter onBack={() => go(index - 1)} status={status} onNext={nextStep} />;
      break;
    case 'handoff': {
      const source = prefs?.model.source ?? (line.kind === 'chosen' ? 'engine' : codexOn ? 'chatgpt' : null);
      const aiLine =
        source === 'chatgpt'
          ? `ChatGPT${codexStatus?.email ? `, signed in as ${codexStatus.email}` : ''}.`
          : source === 'provider' && prefs?.model.provider
            ? `${prefs.providerName ?? prefs.model.provider}: ${prefs.model.model}.`
            : line.kind === 'chosen'
              ? `On this computer: ${line.name}, chosen in Engines.`
              : modelLineText(line);
      content = (
        <HandoffStep
          kicker={kicker}
          state={current.state}
          aiLine={aiLine}
          aiReady={ready}
          mayChange={control.settings === undefined ? null : control.settings === null ? true : control.settings.agentMayChange}
          onPick={(id) => goTo(id)}
        />
      );
      footer = (
        <StepFooter
          onBack={() => go(index - 1)}
          status={status ?? (ready ? null : 'Your AI is not set up yet')}
          onSkip={() => goTo('plugins')}
          skipLabel="Set up the rest myself"
          onNext={() => void continueWithAgent()}
          nextLabel="Continue with the Agent"
          busy={busy}
        />
      );
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

  const inRest = steps.some((s) => s.id === 'plugins');
  return (
    <WizardFrame
      title="Set up OAIY"
      subtitle={
        inRest
          ? 'Your AI and the Agent, then your plugins, their devices and your apps, step by step. Leave any time: setup keeps its place.'
          : 'The essentials: your AI, and what the Agent may do. Then the Agent sets up the rest with you. Leave any time: setup keeps its place.'
      }
      progress={progress}
      rail={steps.map((s) => ({ id: s.id, title: s.title, hint: s.hint, state: s.state, optional: s.optional }))}
      current={index}
      onPick={go}
      railLabel="Setup steps"
      onLeave={onExit}
      flush={isPlugin}
      footer={footer}
    >
      {content}
    </WizardFrame>
  );
}

// ---------------------------------------------------------------------------
// The rest, by hand: plugins, connecting an app, done
// ---------------------------------------------------------------------------

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
  const counted = steps.filter((s) => s.id !== 'done' && s.id !== 'welcome' && s.id !== 'handoff');
  const left = counted.filter((s) => s.state !== 'done');
  return (
    <div className="setup-done">
      <StepHeader
        kicker={kicker}
        title={left.length ? 'Nearly there' : 'You’re set up'}
        description={left.length ? 'Finish now and come back to the rest any time: setup is in Settings, and the Overview keeps a card for what is left.' : 'OAIY is ready. Setup is in Settings whenever you want to run it again.'}
      />
      <ul className="setup-summary">
        {counted.map((s) => (
          <li key={s.id} className={`is-${s.state}`}>
            <span className="setup-summary-mark" aria-hidden>
              {s.state === 'done' ? <Check size={13} strokeWidth={2.6} /> : s.id === 'ai' ? <Sparkles size={13} /> : <Package size={13} />}
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
