import type { ReactNode } from 'react';
import { Check, ChevronRight, CircleDashed, ListChecks, PartyPopper, ShieldAlert, TriangleAlert, X } from 'lucide-react';
import { agentPreferences, engines, link, type PluginRecord } from './api';
import {
  deriveSetupSteps,
  setupComplete,
  setupProgress,
  type SetupInput,
  type StepId,
  type StepTarget,
} from './setupGuide';
import { chosenModel, firstRunProgress, firstRunSteps, pluginsNeedingSetup, readSetup } from './setupFlow';
import { answered, RetryLine, usePoll } from './SetupParts';
import { useLiveSetup } from './useLiveSetup';
import { openSetup, refetchSetup, useSetupError, useSetupState } from './useSetupState';

/**
 * The Overview's setup card: a compact "Continue setup" while the first-run
 * wizard is not finished, a nudge for each plugin whose setup is not finished
 * (one dropped in by hand, or one whose setup version went up), and the live
 * checks the old guide showed, a click away.
 *
 * Every check is worked out from live state (see setupGuide.ts and
 * setupFlow.ts) rather than stored, so it can never claim something is done
 * after it was undone. The wizard itself is its own page (SetupPage).
 */

interface Props extends SetupInput {
  onNavigate: (target: StepTarget) => void;
  onDismiss: () => void;
  actions?: Partial<Record<StepId, ReactNode>>;
  /** The plugins found set up by their live checks, where the page asks them already (else asked here). */
  live?: Record<string, boolean | null | undefined>;
}

export default function SetupGuidePanel({ onNavigate, onDismiss, actions, live: liveFromPage, ...state }: Props) {
  const setup = useSetupState();
  const setupError = useSetupError();
  const unfinished = !!setup && !setup.firstRun.finished;
  // What the wizard reads for its steps (asked now and then; the card is small): the model chosen in Engines, the
  // Agent's model (a provider counts for "Your AI"), and a linked account (for "Connect an app").
  const [catalog] = usePoll(() => engines.catalog(), 15000);
  const [prefsAnswer, , prefsError] = usePoll(() => agentPreferences.get().then((v) => ({ v })), 15000, unfinished);
  const [linked] = usePoll(() => link.status().then((l) => l.linked), 30000, unfinished);
  // A plugin set up by hand counts as set up once its steps' checks all pass (no nudge while they are asked).
  const ownLive = useLiveSetup(liveFromPage ? null : state.plugins, setup);
  const live = liveFromPage ?? ownLive;
  const checks = deriveSetupSteps({ ...state, engineModel: state.engineModel ?? !!chosenModel(catalog, 'llm') });
  const guide = setupProgress(checks);
  const complete = setupComplete(checks);
  // (the checks known: what they read has answered, so a check is not called failing while it is asked)
  const known = state.runtime !== null && state.providers !== null && state.services !== null && catalog !== null;
  const failing = setup?.firstRun.finished && known ? checks.filter((s) => !s.done && !s.optional) : [];
  const nudges = pluginsNeedingSetup(state.plugins, setup, live);
  const untrusted = waitingForTrust(state.plugins);
  const firstRun = unfinished
    ? firstRunSteps({ state: setup, plugins: state.plugins, catalog, guide: state, live, prefs: answered(prefsAnswer, prefsError), linked: linked === true })
    : null;
  const progress = firstRun ? firstRunProgress(firstRun) : null;
  const nextUp = firstRun?.find((s) => s.state === 'todo' && s.id !== 'welcome' && s.id !== 'handoff' && s.id !== 'done') ?? firstRun?.find((s) => s.id === 'handoff' && s.state === 'todo');

  const title = firstRun
    ? `Continue setup · ${progress!.done} of ${progress!.total}`
    : nudges.length
      ? 'Finish setting up'
      : failing.length || untrusted.length
        ? 'Something needs a look'
        : setup || complete
          ? 'You’re set up'
          : `Getting started · ${guide.done} of ${guide.total}`;
  const settled = !firstRun && !nudges.length && !failing.length && !untrusted.length && (setup ? true : complete);

  return (
    <section className={settled ? 'service-section setup-guide is-complete' : 'service-section setup-guide'}>
      <div className="section-title-row">
        <h3 className="section-title">{title}</h3>
        <button className="btn-tiny" onClick={onDismiss} aria-label="Dismiss the setup guide">
          <X size={13} /> Dismiss
        </button>
      </div>

      {firstRun && (
        <div className="setup-continue">
          <span className="setup-progress" aria-hidden>
            <i style={{ width: `${progress!.total ? (progress!.done / progress!.total) * 100 : 0}%` }} />
          </span>
          <p className="form-hint">
            {nextUp ? (
              <>
                Next: <strong>{nextUp.title}</strong>. Setup keeps its place: pick up where you left off.
              </>
            ) : (
              'Everything is done: finish setup to close it.'
            )}
          </p>
          <button className="btn btn-primary" onClick={() => openSetup()}>
            Continue setup <ChevronRight size={14} />
          </button>
        </div>
      )}

      {nudges.map((p) => (
        <div key={p.id} className="setup-nudge">
          <ListChecks size={14} aria-hidden />
          <span>
            <span>
              Finish setting up <strong>{p.manifest?.name ?? p.id}</strong>
            </span>
            <small>{readSetup(p)?.title}</small>
          </span>
          <button className="btn btn-secondary" onClick={() => openSetup({ plugin: p.id })}>
            Set up…
          </button>
        </div>
      ))}

      {untrusted.map((p) => (
        <div key={p.id} className="setup-nudge" data-testid="plugin-waiting-for-trust">
          <ShieldAlert size={14} aria-hidden />
          <span>
            <span>
              <strong>{p.manifest?.name ?? p.declared?.name ?? p.id}</strong> is waiting for you to trust it
            </span>
            <small>It is not signed, so it does not start until you trust it in Plugins.</small>
          </span>
          <button className="btn btn-secondary" onClick={() => onNavigate('plugins')}>
            Open Plugins
          </button>
        </div>
      ))}

      {settled && (
        <p className="form-hint">
          <PartyPopper size={13} /> {setup ? 'Setup is finished.' : 'The runtime is ready and your flows have a model.'} Run it again from Settings any time.
        </p>
      )}
      {/* Finished once, and something it set up is not true any more: said here, not inside the collapsed list. */}
      {failing.length > 0 && (
        <ul className="setup-list">
          {failing.map((s) => (
            <CheckRow key={s.id} step={s} action={actions?.[s.id]} onNavigate={onNavigate} alarm />
          ))}
        </ul>
      )}
      {!setup && setupError && <RetryLine message={`Setup could not be read: ${setupError}`} onRetry={() => void refetchSetup()} />}
      {!setup && !setupError && !complete && (
        <div className="form-actions">
          <button className="btn btn-primary" onClick={() => openSetup()}>
            Open setup <ChevronRight size={14} />
          </button>
        </div>
      )}

      <details className="setup-more">
        <summary>View all setup checks</summary>
        <ul className="setup-list">
          {checks.map((s) => (
            <CheckRow key={s.id} step={s} action={actions?.[s.id]} onNavigate={onNavigate} />
          ))}
        </ul>
      </details>
    </section>
  );
}

/**
 * The plugins the host holds back until the person trusts them: unsigned, not running (one a developer build lets run
 * is not waiting), and not turned off by the person.
 */
export function waitingForTrust(plugins: PluginRecord[] | null | undefined): PluginRecord[] {
  return (plugins ?? []).filter((p) => p.trust?.state === 'unsigned' && !p.userDisabled && !['running', 'starting', 'unhealthy'].includes(p.state));
}

/** One setup check: done, or what is missing and the way to it. */
function CheckRow({ step: s, action, onNavigate, alarm }: { step: ReturnType<typeof deriveSetupSteps>[number]; action?: ReactNode; onNavigate: (target: StepTarget) => void; alarm?: boolean }) {
  return (
    <li className={s.done ? 'setup-step is-done' : 'setup-step'}>
      <span className="setup-mark" aria-hidden>
        {s.done ? <Check size={14} /> : alarm ? <TriangleAlert size={14} /> : <CircleDashed size={14} />}
      </span>
      <span className="setup-body">
        <strong>
          {s.title}
          {s.optional && <em className="setup-optional">optional</em>}
        </strong>
        <small>{s.blocker ?? s.detail}</small>
      </span>
      {!s.done &&
        (action ?? (
          <button className="btn-tiny" onClick={() => onNavigate(s.target)}>
            {s.cta} <ChevronRight size={12} />
          </button>
        ))}
      {s.done && (
        <span className="setup-done-label" aria-label={`${s.title} — done`}>
          done
        </span>
      )}
    </li>
  );
}
