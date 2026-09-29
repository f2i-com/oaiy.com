import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Bot, Check, ExternalLink, Loader2, PhoneForwarded, Play, RotateCcw, ShieldCheck, TriangleAlert } from 'lucide-react';
import {
  agentIntent,
  bridge,
  calendar,
  engines,
  isTauri,
  plugins as pluginsApi,
  services as servicesApi,
  setup as setupApi,
  type CalendarSettings,
  type CheckOutcome,
  type PluginRecord,
  type SetupPluginDetail,
} from './api';
import { SettingsForm } from './HoursPanel';
import PluginScreenPage, { type PluginNavTarget, type SetupScreenCalls } from './PluginScreenPage';
import {
  canMoveOn,
  canSkip,
  connectorFor,
  describeCapabilities,
  firstOpenStep,
  pluginSteps,
  readSetup,
  type DeclaredStep,
  type PluginStep,
  type SettingsField,
} from './setupFlow';
import { EngineModelCard, errorText, RetryLine, ServiceCard, StepFooter, StepHeader, usePoll, WizardFrame, type RailItem } from './SetupParts';
import { useToast } from './Toasts';
import { refetchModules } from './useModules';
import { setSetupState, useSetupState } from './useSetupState';

/**
 * A plugin's setup wizard, built from its manifest's `setup.steps`: what it
 * may do (always first), what it needs (a service, the engines' model), its
 * settings, its own screens in setup mode, and the host's own steps.
 *
 * Shown on its own page (after an install, from "Set up…") or as one step of
 * the first-run wizard (`embedded`), where it draws its steps as a strip and
 * its own footer.
 */

/** Where Aokie streams calls for OAIY's agent to answer: the desktop's voice gateway (the host owns 17872). */
export const OAIY_REALTIME_ENDPOINT = 'ws://127.0.0.1:17872/api/ai/providers/oaiy/v1/realtime/stream';
/** Where call data goes when OAIY answers: OAIY's own agent, in its window. */
export const OAIY_REALTIME_DESTINATION = 'https://oaiy.localhost';

/** Does a phone plugin's settings bag send calls to OAIY's gateway (the provider `oaiy`)? */
export function callsGoToOaiy(bag: Record<string, unknown> | null | undefined): boolean {
  if (!bag || bag.realtimeVoiceMode !== 'desktop_realtime') return false;
  try {
    const seg = new URL(String(bag.realtimeVoiceEndpoint ?? '')).pathname.split('/');
    return seg.length === 8 && seg[1] === 'api' && seg[2] === 'ai' && seg[3] === 'providers' && decodeURIComponent(seg[4]) === 'oaiy' && seg[6] === 'realtime';
  } catch {
    return false;
  }
}

/** One of the plugin's commands, through the same gated route its screens use; the SDK envelope taken off. */
export async function pluginCommand(record: PluginRecord, command: string, payload?: unknown): Promise<unknown> {
  const connector = connectorFor(record, command);
  if (!connector) throw new Error(`${record.manifest?.name ?? record.id} has no ${command} command.`);
  const key = `setup-${command}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const res = await bridge.connectorRequest(connector, command, payload, key);
  if (res.ok === false) throw new Error('The desktop could not complete this plugin request.');
  const r = res.result as { ok?: boolean; data?: unknown; error?: unknown } | undefined;
  if (r?.ok === false) throw new Error(String(typeof r.error === 'object' && r.error ? (r.error as { message?: unknown }).message ?? JSON.stringify(r.error) : r.error ?? 'The plugin could not complete this action.'));
  return r && typeof r === 'object' && 'data' in r ? r.data : r;
}

function at(data: unknown, path: string): unknown {
  if (!path) return data;
  return path.split('.').reduce<unknown>((v, k) => (v && typeof v === 'object' ? (v as Record<string, unknown>)[k] : undefined), data);
}

interface Props {
  pluginId: string;
  /** `page`: its own page; `embedded`: a step of the first-run wizard. */
  layout: 'page' | 'embedded';
  /** Finished (or left, on its own page). */
  onFinished: () => void;
  /** Leave without finishing (its own page: "Continue later"). */
  onLeave: () => void;
  /** Embedded: Back from its first step goes back in the first-run wizard. */
  onBackOut?: () => void;
  /** Embedded: skip this plugin's setup for now. */
  onSkipOut?: () => void;
  onNavigate: (view: PluginNavTarget | 'providers' | 'plugins') => void;
  /** Open at this step, when it shows (the Agent asked for it); else the first not done. */
  initialStep?: string;
}

export default function PluginWizard({ pluginId, layout, onFinished, onLeave, onBackOut, onSkipOut, onNavigate, initialStep }: Props) {
  const toast = useToast();
  const setupState = useSetupState();
  const [plugins, refreshPlugins, pluginsError] = usePoll(() => pluginsApi.list().then((s) => s.plugins), 3000);
  const record = plugins?.find((p) => p.id === pluginId) ?? null;
  const manifestKey = JSON.stringify((record?.manifest as unknown as { setup?: unknown; ui?: unknown } | undefined)?.setup ?? null) + JSON.stringify((record?.manifest as unknown as { ui?: unknown } | undefined)?.ui ?? null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const declared = useMemo(() => readSetup(record), [manifestKey, record?.id]);
  const [detail, setDetail] = useState<SetupPluginDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const loadDetail = useCallback(async () => {
    try {
      setDetail(await setupApi.plugin(pluginId));
      setDetailError(null);
    } catch (e) {
      setDetailError(errorText(e));
    }
  }, [pluginId]);
  useEffect(() => void loadDetail(), [loadDetail]);

  const needsServices = !!declared?.steps.some((s) => s.requires?.some((r) => r.kind === 'service'));
  const needsEngines = !!declared?.steps.some((s) => s.requires?.some((r) => r.kind === 'engineModel'));
  const [services, refreshServices] = usePoll(() => servicesApi.list().then((s) => s.services), 3000, needsServices);
  const [catalog, refreshCatalog] = usePoll(() => engines.catalog(), 3000, needsEngines);

  const [checks, setChecks] = useState<Record<string, CheckOutcome>>({});
  const [whens, setWhens] = useState<Record<string, CheckOutcome>>({});
  const [host, setHost] = useState<Record<string, boolean | undefined>>({});
  const recordState = setupState?.plugins[pluginId] ?? detail?.state ?? null;
  const accepted = !!detail && (() => {
    const list = recordState?.permissionsAccepted;
    return !!list && detail.capabilities.every((c) => list.includes(c));
  })();

  const steps: PluginStep[] = useMemo(
    () => (declared ? pluginSteps(declared, { record: recordState, permissionsAccepted: accepted, services, catalog, checks, whens, host }) : []),
    [declared, recordState, accepted, services, catalog, checks, whens, host],
  );

  // Where it is, by step id (a step can come and go with its `when`).
  const [currentId, setCurrentId] = useState<string | null>(null);
  const loaded = !!declared && !!detail;
  useEffect(() => {
    if (loaded && currentId === null && steps.length)
      setCurrentId(initialStep && steps.some((s) => s.step.id === initialStep) ? initialStep : steps[firstOpenStep(steps)].step.id);
  }, [loaded, currentId, steps, initialStep]);
  const index = Math.max(0, steps.findIndex((s) => s.step.id === currentId));
  const current: PluginStep | undefined = steps[index];

  const runCheck = useCallback(
    async (step: DeclaredStep, which: 'done' | 'when' = 'done') => {
      try {
        const out = await setupApi.check(pluginId, step.id, which);
        (which === 'done' ? setChecks : setWhens)((prev) => (prev[step.id]?.passed === out.passed && prev[step.id]?.detail === out.detail ? prev : { ...prev, [step.id]: out }));
        return out;
      } catch (e) {
        // A check that cannot run is "not yet", never a failed step.
        const out = { passed: false, detail: errorText(e) };
        if (which === 'done') setChecks((prev) => ({ ...prev, [step.id]: out }));
        return out;
      }
    },
    [pluginId],
  );

  // Every check once when the wizard opens: which steps show, which are done.
  const firstChecks = useRef(false);
  useEffect(() => {
    if (!declared || firstChecks.current) return;
    firstChecks.current = true;
    for (const s of declared.steps) {
      if (s.when !== undefined) void runCheck(s, 'when');
      if (s.kind === 'screen' && s.done !== undefined) void runCheck(s);
    }
  }, [declared, runCheck]);

  // The step open now: its done check every 3 s (a plugin restarting just means "not yet").
  const openStep = current?.step;
  useEffect(() => {
    if (!openStep || openStep.kind !== 'screen' || openStep.done === undefined) return;
    void runCheck(openStep);
    const t = window.setInterval(() => {
      if (!document.hidden) void runCheck(openStep);
    }, 3000);
    return () => window.clearInterval(t);
  }, [openStep, runCheck]);
  // And the `when` checks again as the person moves on (a choice made in one step can show or hide a later one).
  useEffect(() => {
    if (!declared || !currentId) return;
    for (const s of declared.steps) if (s.when !== undefined) void runCheck(s, 'when');
  }, [declared, currentId, runCheck]);

  const [busy, setBusy] = useState(false);
  const [stepError, setStepError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ fraction: number | null; text: string } | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  useEffect(() => {
    setStepError(null);
    setProgress(null);
    setFailure(null);
  }, [currentId]);

  const mark = useCallback(
    async (stepId: string, status: 'done' | 'skipped' | 'todo' = 'done') => {
      setSetupState(await setupApi.markStep(pluginId, stepId, status));
    },
    [pluginId],
  );

  const name = record?.manifest?.name ?? pluginId;
  const finish = useCallback(async () => {
    setBusy(true);
    try {
      setSetupState(await setupApi.finish(pluginId));
      void refetchModules();
      toast.push({ kind: 'success', title: `${name} is set up` });
      onFinished();
    } catch (e) {
      setStepError(errorText(e));
    } finally {
      setBusy(false);
    }
  }, [pluginId, name, toast, onFinished]);

  const go = (i: number) => {
    const target = steps[Math.max(0, Math.min(steps.length - 1, i))];
    if (target) setCurrentId(target.step.id);
  };
  const isLast = index === steps.length - 1;
  const next = () => (isLast ? void finish() : go(index + 1));

  const screenCalls: SetupScreenCalls = useMemo(
    () => ({
      progress: (fraction, text) => setProgress({ fraction, text }),
      done: async () => {
        if (!openStep) return;
        if (openStep.done !== undefined) await runCheck(openStep);
        else await mark(openStep.id).catch(() => undefined);
      },
      fail: (message) => setFailure(message),
      finish: async () => {
        if (openStep && openStep.done === undefined) await mark(openStep.id).catch(() => undefined);
        await finish();
      },
    }),
    [openStep, runCheck, mark, finish],
  );

  // ---- rendering ----

  let content: ReactNode;
  let flush = false;
  if (pluginsError && !plugins) content = <RetryLine message={`The plugins could not be read: ${pluginsError}`} onRetry={() => void refreshPlugins()} />;
  else if (plugins && !record) content = <p className="form-hint">{pluginId} is not installed.</p>;
  else if (record && !declared) content = <p className="form-hint">{name} has nothing to set up.</p>;
  else if (detailError && !detail) content = <RetryLine message={`Its setup could not be read: ${detailError}`} onRetry={() => void loadDetail()} />;
  else if (!current || !record || !detail) content = <p className="form-hint">Loading…</p>;
  else {
    const s = current.step;
    const kicker = `${name} · step ${index + 1} of ${steps.length}`;
    const head = <StepHeader kicker={kicker} title={s.title} description={s.description} state={current.state} />;
    switch (s.kind) {
      case 'permissions':
        content = (
          <>
            {head}
            <PermissionsStep detail={detail} record={record} accepted={accepted} />
          </>
        );
        break;
      case 'requirements':
        content = (
          <>
            {head}
            <div className="setup-reqs">
              {(s.requires ?? []).map((r) =>
                r.kind === 'service' ? (
                  <ServiceCard key={`s:${r.id}`} id={r.id} why={r.why} services={services} onChanged={() => void refreshServices()} />
                ) : (
                  <EngineModelCard key={`m:${r.group}`} group={r.group} why={r.why} catalog={catalog} onChanged={() => void refreshCatalog()} onOpenEngines={() => onNavigate('engines')} />
                ),
              )}
            </div>
          </>
        );
        break;
      case 'settings':
        content = (
          <>
            {head}
            <PluginRunNote record={record} onStarted={() => void refreshPlugins()} />
            <SettingsStep key={s.id} record={record} step={s} onSaved={() => mark(s.id)} />
          </>
        );
        break;
      case 'screen': {
        flush = true;
        const check = checks[s.id];
        content = (
          <div className="setup-screen-step">
            {head}
            <div className="setup-screen-status" role="status">
              {progress && (
                <span className="setup-req-progress">
                  <span className="setup-progress" aria-hidden>
                    <i className={progress.fraction === null ? 'is-indeterminate' : undefined} style={progress.fraction === null ? undefined : { width: `${progress.fraction * 100}%` }} />
                  </span>
                  <small>{progress.text}</small>
                </span>
              )}
              {failure && (
                <span className="card-warn">
                  <TriangleAlert size={12} /> {failure}
                </span>
              )}
              {s.done !== undefined && current.state !== 'done' && check && record.state === 'running' && (
                <small className="setup-check" title={check.detail}>
                  Not done yet: this step checks itself every few seconds.
                </small>
              )}
            </div>
            <PluginRunNote record={record} onStarted={() => void refreshPlugins()} />
            <PluginScreenPage
              pluginId={pluginId}
              screenId={s.screen}
              onNavigate={(v) => onNavigate(v)}
              setup={{ step: s.id, view: s.view ?? '', calls: screenCalls }}
            />
          </div>
        );
        break;
      }
      case 'host':
        content = (
          <>
            {head}
            {s.action === 'phone.answerWithOaiy' && <PluginRunNote record={record} onStarted={() => void refreshPlugins()} />}
            {s.action === 'phone.answerWithOaiy' ? (
              <AnswerWithOaiyStep record={record} step={s} onRoute={(on) => setHost((h) => (h[s.id] === on ? h : { ...h, [s.id]: on }))} onRecorded={() => mark(s.id)} onNavigate={onNavigate} />
            ) : (
              <BusinessStep onNamed={(named) => setHost((h) => (h[s.id] === named ? h : { ...h, [s.id]: named }))} onSaved={() => mark(s.id)} />
            )}
          </>
        );
        break;
    }
  }

  const status = stepError ? (
    <span className="card-warn">{stepError}</span>
  ) : current?.state === 'done' ? (
    <span className="setup-ok">
      <Check size={13} /> Done
    </span>
  ) : current?.step.optional ? (
    'Optional'
  ) : current ? (
    current.step.kind === 'permissions' ? 'Accept to go on' : 'Waiting for this step'
  ) : null;

  const acceptAndGo = async () => {
    setBusy(true);
    setStepError(null);
    try {
      await mark('permissions');
      await loadDetail();
      next();
    } catch (e) {
      setStepError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const footer = (
    <StepFooter
      onBack={index > 0 ? () => go(index - 1) : layout === 'embedded' ? onBackOut : undefined}
      status={status}
      onSkip={
        current && canSkip(current)
          ? () => {
              void mark(current.step.id, 'skipped').catch((e) => setStepError(errorText(e)));
              next();
            }
          : layout === 'embedded' && current?.step.kind === 'permissions'
            ? onSkipOut
            : undefined
      }
      skipLabel={current && canSkip(current) ? 'Skip' : 'Set it up later'}
      onNext={current ? (current.step.kind === 'permissions' && current.state !== 'done' ? () => void acceptAndGo() : next) : undefined}
      nextLabel={current?.step.kind === 'permissions' && current.state !== 'done' ? 'Accept and continue' : isLast ? 'Finish' : 'Next'}
      nextDisabled={!current || (current.step.kind !== 'permissions' && !canMoveOn(current))}
      busy={busy}
    />
  );

  if (layout === 'embedded') {
    return (
      <div className={flush ? 'setup-embedded is-flush' : 'setup-embedded'}>
        <ol className="setup-substeps" aria-label={`${declared?.title ?? name}: its steps`}>
          {steps.map((x, i) => (
            <li key={x.step.id} className={`is-${x.state}${i === index ? ' is-current' : ''}`}>
              <button type="button" aria-current={i === index ? 'step' : undefined} onClick={() => go(i)}>
                {x.state === 'done' ? <Check size={11} strokeWidth={2.6} /> : <span>{i + 1}</span>} {x.step.title}
              </button>
            </li>
          ))}
        </ol>
        <div className="setup-embedded-body">{content}</div>
        {footer}
      </div>
    );
  }
  const rail: RailItem[] = steps.map((x) => ({ id: x.step.id, title: x.step.title, state: x.state, optional: x.step.optional, hint: KIND_HINT[x.step.kind] }));
  return (
    <WizardFrame
      title={declared?.title ?? `Set up ${name}`}
      subtitle={record ? <>From the {name} plugin{record.manifest?.version ? `, v${record.manifest.version}` : ''}. Each step checks itself once it is true.</> : undefined}
      progress={{ done: steps.filter((x) => x.state === 'done').length, total: steps.length }}
      rail={rail}
      current={index}
      onPick={go}
      railLabel={`${name}: setup steps`}
      onLeave={onLeave}
      flush={flush}
      footer={footer}
    >
      {content}
    </WizardFrame>
  );
}

/**
 * The plugin is not running, so a step that talks to it waits: said once, with
 * a way to start it. A plugin that restarts mid-step (a driver install) shows
 * here as starting; the step is not failed and its screen is not reloaded.
 */
function PluginRunNote({ record, onStarted }: { record: PluginRecord; onStarted: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (record.state === 'running' || record.state === 'unhealthy') return null;
  const name = record.manifest?.name ?? record.id;
  const reason = (record.reason ?? '').trim().replace(/\.+$/, '');
  const starting = record.state === 'starting';
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      await pluginsApi.start(record.id);
      onStarted();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="setup-run-note" role="status">
      <TriangleAlert size={13} aria-hidden />
      <span>
        {starting ? `${name} is starting…` : `${name} is not running${reason ? ` (${reason})` : ''}.`} This step talks to it, and carries on once it runs.
        {error && <small>{error}</small>}
      </span>
      {!starting && !record.userDisabled && (
        <button type="button" className="btn-tiny" disabled={busy} onClick={() => void start()}>
          {busy ? <Loader2 size={12} className="spin" /> : <Play size={12} />} Start it
        </button>
      )}
    </div>
  );
}

const KIND_HINT: Record<DeclaredStep['kind'], string> = {
  permissions: 'Its permissions',
  requirements: 'What it needs',
  settings: 'Its settings',
  screen: 'In the plugin',
  host: 'In OAIY',
};

// ---------------------------------------------------------------------------
// The steps
// ---------------------------------------------------------------------------

function PermissionsStep({ detail, record, accepted }: { detail: SetupPluginDetail; record: PluginRecord; accepted: boolean }) {
  const groups = describeCapabilities(detail.capabilities);
  return (
    <div className="setup-perms">
      <p className="form-hint">
        A plugin runs as a program on this computer, and can do only what it asks for here. OAIY holds it to this list: anything else it tries is refused.
        {accepted && ' You accepted this list.'}
      </p>
      {groups.length === 0 ? (
        <p className="form-hint">It asks for nothing beyond running.</p>
      ) : (
        <ul className="setup-perm-list">
          {groups.map((g) => (
            <li key={g.text}>
              <ShieldCheck size={15} aria-hidden />
              <span>
                <strong>{g.text}</strong>
                <small className="setup-mono">{g.names.join(', ')}</small>
              </span>
            </li>
          ))}
        </ul>
      )}
      {(detail.legacyCapabilities?.length ?? 0) > 0 && (
        <p className="card-meta">Uses older names for some of these: {detail.legacyCapabilities!.map(([from]) => from).join(', ')}.</p>
      )}
      {(detail.unknownCapabilities?.length ?? 0) > 0 && (
        <p className="card-meta card-warn">
          <TriangleAlert size={12} /> It also asks for {detail.unknownCapabilities!.join(', ')}, which this OAIY does not provide, so it will not get them.
        </p>
      )}
      <p className="card-meta">
        {record.manifest?.publisher ? `Published by ${record.manifest.publisher}. ` : ''}Installing a plugin installs code this computer runs: set up only plugins you trust.
      </p>
    </div>
  );
}

function fieldValue(field: SettingsField, raw: unknown): unknown {
  if (field.type === 'bool') return raw === true || raw === 'true';
  if (field.type === 'number') return typeof raw === 'number' ? raw : raw === undefined || raw === null || raw === '' ? '' : Number(raw);
  return raw ?? '';
}

function SettingsStep({ record, step, onSaved }: { record: PluginRecord; step: DeclaredStep; onSaved: () => Promise<void> }) {
  const fields = step.fields ?? [];
  const [values, setValues] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const load = useCallback(async () => {
    try {
      const data = await pluginCommand(record, step.read?.command ?? 'settings.get');
      const bag = (at(data, step.read?.path ?? 'settings') ?? {}) as Record<string, unknown>;
      setValues(Object.fromEntries(fields.map((f) => [f.key, fieldValue(f, bag[f.key])])));
      setError(null);
    } catch (e) {
      setError(`Its settings could not be read: ${errorText(e)}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record.id, step.id]);
  useEffect(() => void load(), [load]);

  const save = async () => {
    if (!values) return;
    setSaving(true);
    setError(null);
    try {
      const payload = Object.fromEntries(fields.map((f) => [f.key, f.type === 'number' ? Number(values[f.key]) : values[f.key]]));
      await pluginCommand(record, step.write?.command ?? 'settings.set', payload);
      await onSaved();
      setSaved(true);
    } catch (e) {
      setError(`Not saved: ${errorText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  if (!values) return error ? <RetryLine message={error} onRetry={() => void load()} /> : <p className="form-hint">Reading its settings…</p>;
  const set = (key: string, v: unknown) => {
    setSaved(false);
    setValues((prev) => ({ ...prev, [key]: v }));
  };
  return (
    <div className="setup-settings">
      {fields.map((f) => (
        <div key={f.key} className="setup-field">
          {f.type === 'bool' ? (
            <label className="setup-toggle">
              <input type="checkbox" checked={values[f.key] === true} onChange={(e) => set(f.key, e.target.checked)} />
              <span>
                <strong>{f.label}</strong>
                {f.help && <small>{f.help}</small>}
              </span>
            </label>
          ) : (
            <label className="form-row">
              <span>{f.label}</span>
              {f.type === 'choice' ? (
                <select value={String(values[f.key] ?? '')} onChange={(e) => set(f.key, f.options?.find((o) => String(o.value) === e.target.value)?.value ?? e.target.value)}>
                  {f.options?.map((o) => (
                    <option key={String(o.value)} value={String(o.value)}>
                      {o.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input type={f.type === 'number' ? 'number' : 'text'} value={String(values[f.key] ?? '')} onChange={(e) => set(f.key, e.target.value)} />
              )}
              {f.help && <small className="form-hint">{f.help}</small>}
            </label>
          )}
        </div>
      ))}
      {error && <p className="card-warn" role="alert">{error}</p>}
      <div className="form-actions">
        <button type="button" className="btn btn-primary" disabled={saving} onClick={() => void save()}>
          {saving ? <Loader2 size={14} className="spin" /> : saved ? <Check size={14} /> : null} {saved ? 'Saved' : 'Save'}
        </button>
      </div>
    </div>
  );
}

/**
 * The phone's calls and texts, answered by OAIY:
 * (a) the phone plugin streams calls to OAIY's voice gateway (its settings, as
 *     its own Settings tab writes them);
 * (b) the Agent answers calls and texts (its own setting, in its page's
 *     storage: asked of it through the desktop, or ticked by hand).
 */
function AnswerWithOaiyStep({
  record,
  step,
  onRoute,
  onRecorded,
  onNavigate,
}: {
  record: PluginRecord;
  step: DeclaredStep;
  onRoute: (toOaiy: boolean) => void;
  onRecorded: () => Promise<void>;
  onNavigate: (v: PluginNavTarget) => void;
}) {
  const [route, setRoute] = useState<boolean | null>(null);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [restart, setRestart] = useState<string[] | null>(null);
  const [busy, setBusy] = useState<'route' | 'restart' | 'agent' | null>(null);
  const [agent, setAgent] = useState<'sent' | 'manual' | null>(null);
  const [agentError, setAgentError] = useState<string | null>(null);
  const setupState = useSetupState();
  const recorded = !!setupState?.plugins[record.id]?.done.includes(step.id);
  const name = record.manifest?.name ?? record.id;

  const read = useCallback(async () => {
    try {
      const data = (await pluginCommand(record, 'settings.get')) as { settings?: Record<string, unknown> } | undefined;
      const on = callsGoToOaiy(data?.settings);
      setRoute(on);
      onRoute(on);
      setRouteError(null);
    } catch (e) {
      setRoute(null);
      setRouteError(errorText(e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record.id]);
  useEffect(() => void read(), [read]);

  const sendToOaiy = async () => {
    setBusy('route');
    setRouteError(null);
    try {
      const out = (await pluginCommand(record, 'settings.set', {
        realtimeVoiceMode: 'desktop_realtime',
        realtimeVoiceEndpoint: OAIY_REALTIME_ENDPOINT,
        realtimeVoiceDestination: OAIY_REALTIME_DESTINATION,
        aiReceptionist: true,
      })) as { appliesAtReconnect?: string[]; blocked?: string } | undefined;
      // Where calls go is read when the plugin starts: it says so, and a restart applies it.
      setRestart(out?.appliesAtReconnect?.length ? out.appliesAtReconnect : null);
      if (out?.blocked) setRouteError(`Saved, but ${name} is paused: ${out.blocked}`);
      await read();
    } catch (e) {
      setRouteError(errorText(e));
    } finally {
      setBusy(null);
    }
  };

  const restartPlugin = async () => {
    setBusy('restart');
    try {
      await pluginsApi.stop(record.id).catch(() => undefined);
      await pluginsApi.start(record.id);
      setRestart(null);
      window.setTimeout(() => void read(), 1500);
    } catch (e) {
      setRouteError(errorText(e));
    } finally {
      setBusy(null);
    }
  };

  const turnOnAgent = async () => {
    setBusy('agent');
    setAgentError(null);
    try {
      if (!isTauri()) {
        setAgent('manual');
        return;
      }
      await agentIntent('answerWithOaiy');
      setAgent('sent');
      await onRecorded();
    } catch (e) {
      setAgentError(errorText(e));
      setAgent('manual');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="setup-reqs">
      <div className="setup-req">
        <span className="setup-req-icon" aria-hidden>
          <PhoneForwarded size={16} />
        </span>
        <div className="setup-req-body">
          <strong>Send calls to OAIY</strong>
          <p>{name} streams each call to OAIY on this computer: OAIY Voice hears the caller and speaks, and the agent writes the replies.</p>
          <div className="setup-req-status">
            {route === null && !routeError && <span className="form-hint">Reading {name}’s settings…</span>}
            {route === true && <span className="badge badge-ok">Calls go to OAIY</span>}
            {route === false && <span className="badge badge-neutral">Calls go elsewhere now</span>}
            {restart && (
              <span className="form-hint">
                Saved. It applies when {name} restarts{restart.length ? ` (${restart.join(', ')})` : ''}.
              </span>
            )}
          </div>
          {routeError && <p className="card-warn">{routeError}</p>}
        </div>
        <div className="setup-req-action">
          {restart ? (
            <button type="button" className="btn btn-primary" disabled={!!busy} onClick={() => void restartPlugin()}>
              {busy === 'restart' ? <Loader2 size={14} className="spin" /> : <RotateCcw size={14} />} Restart {name} now
            </button>
          ) : route !== true ? (
            <button type="button" className="btn btn-primary" disabled={!!busy} onClick={() => void sendToOaiy()}>
              {busy === 'route' && <Loader2 size={14} className="spin" />} Send calls to OAIY
            </button>
          ) : null}
        </div>
      </div>

      <div className="setup-req">
        <span className="setup-req-icon" aria-hidden>
          <Bot size={16} />
        </span>
        <div className="setup-req-body">
          <strong>Answer calls and texts in the Agent</strong>
          <p>The Agent answers every call and text to the phone, as your assistant. You see each conversation in the Front desk, and can change its instructions in its Phone settings.</p>
          <div className="setup-req-status">
            {(recorded || agent === 'sent') && <span className="badge badge-ok">Turned on in the Agent</span>}
          </div>
          {agent === 'manual' && !recorded && (
            <div className="setup-manual">
              <p className="form-hint">
                {agentError ? `The Agent could not be asked (${agentError}). ` : 'This window cannot reach the Agent. '}
                Open the Agent, choose <strong>Phone</strong>, tick <strong>Answer text messages</strong> and <strong>Answer phone calls</strong>, and save.
              </p>
              <div className="form-actions">
                <button type="button" className="btn" onClick={() => onNavigate('agent')}>
                  Open the Agent <ExternalLink size={13} />
                </button>
                <button type="button" className="btn btn-primary" onClick={() => void onRecorded()}>
                  <Check size={14} /> I turned it on
                </button>
              </div>
            </div>
          )}
        </div>
        <div className="setup-req-action">
          {!recorded && agent !== 'manual' && (
            <button type="button" className="btn btn-primary" disabled={!!busy} onClick={() => void turnOnAgent()}>
              {busy === 'agent' && <Loader2 size={14} className="spin" />} Turn it on
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/** Your business, as the phone offers it: the Calendar's own settings form. */
function BusinessStep({ onNamed, onSaved }: { onNamed: (named: boolean) => void; onSaved: () => Promise<void> }) {
  const [settings, setSettings] = useState<CalendarSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    try {
      const got = await calendar.get();
      setSettings(got.settings);
      onNamed(!!got.settings.business.trim());
      setError(null);
    } catch (e) {
      setError(`The calendar could not be read: ${errorText(e)}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => void load(), [load]);
  if (!settings) return error ? <RetryLine message={error} onRetry={() => void load()} /> : <p className="form-hint">Reading the calendar…</p>;
  return (
    <div className="setup-business">
      <p className="form-hint">What callers hear the business called, when it is open, and what they can book. The same settings as Hours &amp; Services, under the AI Receptionist.</p>
      <SettingsForm
        embedded
        settings={settings}
        onSaved={(s) => {
          setSettings(s);
          onNamed(!!s.business.trim());
          void onSaved();
        }}
      />
    </div>
  );
}
