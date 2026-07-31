import type { ReactNode } from 'react';
import { Check, ChevronRight, CircleDashed, PartyPopper, X } from 'lucide-react';
import {
  deriveSetupSteps,
  setupComplete,
  setupProgress,
  type SetupInput,
  type StepId,
  type StepTarget,
} from './setupGuide';

/**
 * First-run guide: the shortest path from "just installed" to "flows run".
 *
 * Every row is derived from live state (see setupGuide.ts) rather than stored,
 * so it ticks itself as the user does the work and can never claim something is
 * done after they undo it. Each row hands off to the screen that fixes it —
 * being told what is missing without being taken there is only half an answer.
 *
 * A caller can supply its own control for a step via `actions` — the runtime row
 * uses that to offer "Install Node" in place of a link, because when the fix is
 * one click there is no reason to send anyone to another screen for it.
 */

interface Props extends SetupInput {
  onNavigate: (target: StepTarget) => void;
  onDismiss: () => void;
  actions?: Partial<Record<StepId, ReactNode>>;
}

export default function SetupGuidePanel({ onNavigate, onDismiss, actions, ...state }: Props) {
  const steps = deriveSetupSteps(state);
  const { done, total } = setupProgress(steps);
  const complete = setupComplete(steps);

  return (
    <section className="service-section setup-guide">
      <div className="section-title-row">
        <h3 className="section-title">
          {complete ? 'You’re set up' : `Getting started · ${done} of ${total}`}
        </h3>
        <button className="btn-tiny" onClick={onDismiss} aria-label="Dismiss the setup guide">
          <X size={13} /> Dismiss
        </button>
      </div>

      {complete ? (
        <p className="form-hint">
          <PartyPopper size={13} /> The runtime is ready and your flows have a model. Anything
          still unticked below is optional.
        </p>
      ) : (
        <p className="form-hint">
          Work down the list — each step is checked automatically once it’s actually true.
        </p>
      )}

      <ul className="setup-list">
        {steps.map((s) => (
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
            {/* A done row keeps its screen reachable without shouting about it. */}
            {s.done && (
              <span className="setup-done-label" aria-label={`${s.title} — done`}>
                done
              </span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
