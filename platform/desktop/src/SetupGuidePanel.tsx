import type { ReactNode } from 'react';
import { Check, ChevronRight, CircleDashed, ListChecks, PartyPopper, X } from 'lucide-react';
import { engines } from './api';
import {
  deriveSetupSteps,
  setupComplete,
  setupProgress,
  type SetupInput,
  type StepId,
  type StepTarget,
} from './setupGuide';
import { firstRunProgress, firstRunSteps, pluginsNeedingSetup, readSetup } from './setupFlow';
import { usePoll } from './SetupParts';
import { openSetup, useSetupState } from './useSetupState';

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
}

export default function SetupGuidePanel({ onNavigate, onDismiss, actions, ...state }: Props) {
  const setup = useSetupState();
  // The model chosen in Engines counts for the engine step (asked now and then; the card is small).
  const [catalog] = usePoll(() => engines.catalog(), 15000, !!setup && !setup.firstRun.finished);
  const checks = deriveSetupSteps(state);
  const guide = setupProgress(checks);
  const complete = setupComplete(checks);
  const nudges = pluginsNeedingSetup(state.plugins, setup);
  const firstRun = setup && !setup.firstRun.finished ? firstRunSteps({ state: setup, plugins: state.plugins, catalog, guide: state }) : null;
  const progress = firstRun ? firstRunProgress(firstRun) : null;
  const nextUp = firstRun?.find((s) => s.state === 'todo' && s.id !== 'welcome');

  const title = firstRun
    ? `Continue setup · ${progress!.done} of ${progress!.total}`
    : nudges.length
      ? 'Finish setting up'
      : setup
        ? 'You’re set up'
        : complete
          ? 'You’re set up'
          : `Getting started · ${guide.done} of ${guide.total}`;
  const settled = !firstRun && !nudges.length && (setup ? true : complete);

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
            Finish setting up <strong>{p.manifest?.name ?? p.id}</strong>
            <small>{readSetup(p)?.title}</small>
          </span>
          <button className="btn btn-secondary" onClick={() => openSetup({ plugin: p.id })}>
            Set up…
          </button>
        </div>
      ))}

      {settled && (
        <p className="form-hint">
          <PartyPopper size={13} /> {setup ? 'Setup is finished.' : 'The runtime is ready and your flows have a model.'} Run it again from Settings any time.
        </p>
      )}
      {!setup && !complete && (
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
            <li key={s.id} className={s.done ? 'setup-step is-done' : 'setup-step'}>
              <span className="setup-mark" aria-hidden>
                {s.done ? <Check size={14} /> : <CircleDashed size={14} />}
              </span>
              <span className="setup-body">
                <strong>
                  {s.title}
                  {s.optional && <em className="setup-optional">optional</em>}
                </strong>
                <small>{s.blocker ?? s.detail}</small>
              </span>
              {!s.done &&
                (actions?.[s.id] ?? (
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
          ))}
        </ul>
      </details>
    </section>
  );
}
