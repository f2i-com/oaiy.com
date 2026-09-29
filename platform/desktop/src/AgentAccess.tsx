import { useCallback, useEffect, useState } from 'react';
import { Loader2, TriangleAlert } from 'lucide-react';
import { agentIntent, control, isTauri, type AgentIntent, type ControlSettings } from './api';
import { errorText } from './SetupParts';

/**
 * The Agent's access to OAIY (CONTROL_API.md §1): one switch, "Let the Agent
 * set up and change OAIY for you", kept in the desktop's `control.json`. The
 * first-run wizard and Settings → Agent both show it.
 */

export const AGENT_MAY_CHANGE = 'Let the Agent set up and change OAIY for you';

/** What the switch means, in one sentence. */
export const AGENT_MAY_CHANGE_MEANS =
  'When it is on, the Agent can install and set up plugins, choose models, start and stop services and change settings when you ask it to, and every change it makes is listed in Settings → Agent.';

export interface ControlSwitch {
  /** `undefined` while asking; `null` on a desktop without the control API (on, and not changeable yet). */
  settings: ControlSettings | null | undefined;
  error: string | null;
  saving: boolean;
  set: (agentMayChange: boolean) => Promise<void>;
  reload: () => Promise<void>;
}

export function useControlSettings(): ControlSwitch {
  const [settings, setSettings] = useState<ControlSettings | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const reload = useCallback(async () => {
    try {
      setSettings(await control.settings());
      setError(null);
    } catch (e) {
      setError(errorText(e));
    }
  }, []);
  useEffect(() => void reload(), [reload]);
  const set = useCallback(async (agentMayChange: boolean) => {
    setSaving(true);
    setError(null);
    try {
      setSettings(await control.setSettings({ agentMayChange }));
    } catch (e) {
      setError(`Not saved: ${errorText(e)}`);
    } finally {
      setSaving(false);
    }
  }, []);
  return { settings, error, saving, set, reload };
}

/** The switch, with what it means. On a desktop without the control API it shows as on, and says it cannot be changed yet. */
export function AgentMaySwitch({ control: c }: { control: ControlSwitch }) {
  const unavailable = c.settings === null;
  const on = unavailable ? true : c.settings?.agentMayChange ?? true;
  const disabled = c.settings === undefined || unavailable || c.saving;
  return (
    <div className={`agent-switch${on ? ' is-on' : ''}`}>
      <label className="agent-switch-row">
        <span className="agent-switch-text">
          <strong>{AGENT_MAY_CHANGE}</strong>
          <small>{AGENT_MAY_CHANGE_MEANS}</small>
        </span>
        <span className="agent-switch-control">
          {c.saving && <Loader2 size={14} className="spin" aria-hidden />}
          <input type="checkbox" role="switch" className="switch" checked={on} disabled={disabled} aria-label={AGENT_MAY_CHANGE} onChange={(e) => void c.set(e.target.checked)} />
        </span>
      </label>
      {c.settings === undefined && !c.error && <p className="form-hint">Checking…</p>}
      {unavailable && <p className="form-hint agent-switch-note">On. This OAIY can’t change it yet: the switch comes with an update.</p>}
      {c.settings && !c.settings.agentMayChange && (
        <p className="form-hint agent-switch-note">Off: the Agent can still look at OAIY and tell you what to do, but it changes nothing.</p>
      )}
      {c.error && (
        <p className="card-warn" role="alert">
          <TriangleAlert size={12} /> {c.error}{' '}
          <button type="button" className="btn-tiny" onClick={() => void c.reload()}>
            Try again
          </button>
        </p>
      )}
    </div>
  );
}

/**
 * Hand the person to the Agent: ask its page for `intent`, again for a few
 * seconds while its page is being made (the dashboard has just switched to
 * it). `false` when it could not be asked (a plain browser, or no Agent page).
 */
export async function askAgent(intent: AgentIntent, tries = 5, waitMs = 1000): Promise<boolean> {
  if (!isTauri()) return false;
  for (let i = 0; i < tries; i++) {
    try {
      await agentIntent(intent);
      return true;
    } catch {
      if (i < tries - 1) await new Promise((r) => window.setTimeout(r, waitMs));
    }
  }
  return false;
}
