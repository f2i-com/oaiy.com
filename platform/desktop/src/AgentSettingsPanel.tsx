import { useEffect, useState } from 'react';
import { Check, Cpu, ExternalLink, Loader2, MessageSquare, RotateCcw, TriangleAlert, X } from 'lucide-react';
import ChatGptConnector from './ChatGptConnector';
import {
  agentPreferences,
  codex,
  codexModels,
  control,
  engineRecommendation,
  engines,
  type AgentModelSource,
  type AgentPreferences,
  type ControlLogEntry,
} from './api';
import { AgentMaySwitch, useControlSettings } from './AgentAccess';
import { codexModelOptions, describeChange, localModelLine, newestFirst } from './setupFlow';
import { modelLineText } from './SetupEssentials';
import { answered, errorText, usePoll } from './SetupParts';

/**
 * Settings → Agent: what the Agent may change in OAIY (the switch), the model
 * it thinks with (the engine's, or ChatGPT with a model from Codex's
 * catalogue), and every change it made through the MCP API, newest first.
 *
 * Each part reads a route a desktop may not have yet; without it, the part
 * says so plainly (the switch shows as on, the log shows nothing yet).
 */

export default function AgentSettingsPanel({ onOpenEngines }: { onOpenEngines: () => void }) {
  const access = useControlSettings();
  return (
    <div className="panel agent-settings">
      <section className="model-section">
        <h3 className="section-title">What the Agent may do</h3>
        <AgentMaySwitch control={access} />
      </section>
      <ChangeLogSection />
      <AgentModelSection onOpenEngines={onOpenEngines} />
    </div>
  );
}

function AgentModelSection({ onOpenEngines }: { onOpenEngines: () => void }) {
  const [prefsAnswer, refreshPrefs, prefsError] = usePoll(() => agentPreferences.get().then((v) => ({ v })), 15000);
  const prefs = answered(prefsAnswer, prefsError);
  const [catalog] = usePoll(() => engines.catalog(), 10000);
  const [rec] = usePoll(() => engineRecommendation().catch(() => null), 60000);
  const [status, refreshStatus] = usePoll(() => codex.status(), 8000);
  const connected = status?.connected === true;
  const [models, refreshModels, modelsError] = usePoll(() => codexModels(), 60000, connected);
  const [picked, setPicked] = useState<AgentModelSource | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const source = picked ?? prefs?.model.source ?? 'engine';
  const line = localModelLine(rec, catalog);

  useEffect(() => {
    if (connected) void refreshModels();
  }, [connected, refreshModels]);

  const save = async (next: AgentPreferences) => {
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      await agentPreferences.set(next);
      await refreshPrefs();
      setSaved(true);
    } catch (e) {
      setError(`Not saved: ${errorText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  const choose = (next: AgentModelSource) => {
    setPicked(next);
    // ChatGPT is set once it is signed in (the Agent could not answer with it before).
    if (next === 'engine' && prefs?.model.source !== 'engine') void save({ model: { source: 'engine' } });
    if (next === 'chatgpt' && connected && prefs?.model.source !== 'chatgpt') void save({ model: { source: 'chatgpt' } });
  };

  if (prefs === null) {
    return (
      <section className="model-section">
        <h3 className="section-title">The Agent’s model</h3>
        <p className="form-hint">This OAIY can’t choose the Agent’s model yet: the Agent uses the language model chosen in Engines. {modelLineText(line)}</p>
      </section>
    );
  }

  const options = codexModelOptions(models, prefs?.model.source === 'chatgpt' ? prefs.model.model : null);
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">The Agent’s model</h3>
        {saving && <Loader2 size={14} className="spin" aria-label="Saving" />}
        {saved && !saving && (
          <span className="setup-ok">
            <Check size={13} /> Saved
          </span>
        )}
      </div>
      <p className="form-hint">What the Agent thinks with, in chats and when it sets up OAIY. Phone calls keep their own fast route.</p>
      {prefs === undefined ? (
        <p className="form-hint">Checking…</p>
      ) : (
        <div className="ai-choices is-compact" role="radiogroup" aria-label="The Agent’s model">
          <div className={`ai-choice${source === 'engine' ? ' is-selected' : ''}`}>
            <label className="ai-choice-head">
              <input type="radio" name="agent-model" checked={source === 'engine'} disabled={saving} onChange={() => choose('engine')} />
              <span className="setup-req-icon" aria-hidden>
                <Cpu size={16} />
              </span>
              <span className="ai-choice-text">
                <strong>On this computer</strong>
                <small className="ai-choice-fact">{modelLineText(line)}</small>
              </span>
              <button type="button" className="btn-tiny" onClick={onOpenEngines}>
                Engines <ExternalLink size={12} />
              </button>
            </label>
          </div>
          <div className={`ai-choice${source === 'chatgpt' ? ' is-selected' : ''}`}>
            <label className="ai-choice-head">
              <input type="radio" name="agent-model" checked={source === 'chatgpt'} disabled={saving} onChange={() => choose('chatgpt')} />
              <span className="setup-req-icon" aria-hidden>
                <MessageSquare size={16} />
              </span>
              <span className="ai-choice-text">
                <strong>ChatGPT</strong>
                <small className="ai-choice-fact">
                  {connected ? `Signed in${status?.email ? ` as ${status.email}` : ''}.` : status ? 'Not signed in.' : 'Checking…'}
                </small>
              </span>
            </label>
            {source === 'chatgpt' && (
              <div className="ai-choice-body">
                {connected ? (
                  <label className="form-row agent-model-pick">
                    <span>Model, from Codex’s catalogue</span>
                    <select
                      value={prefs?.model.source === 'chatgpt' ? prefs.model.model ?? '' : ''}
                      disabled={saving}
                      onChange={(e) => void save({ model: e.target.value ? { source: 'chatgpt', model: e.target.value } : { source: 'chatgpt' } })}
                    >
                      {options.map((o) => (
                        <option key={o.value || 'default'} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                    {modelsError && !models && (
                      <small className="card-warn">
                        The catalogue could not be read ({modelsError}).{' '}
                        <button type="button" className="btn-tiny" onClick={() => void refreshModels()}>
                          <RotateCcw size={12} /> Try again
                        </button>
                      </small>
                    )}
                  </label>
                ) : (
                  <ChatGptConnector
                    bare
                    onConnected={() => {
                      void refreshStatus();
                      void save({ model: { source: 'chatgpt' } });
                    }}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      )}
      {error && (
        <p className="card-warn" role="alert">
          <TriangleAlert size={12} /> {error}
        </p>
      )}
    </section>
  );
}

function when(d: Date | null): string {
  if (!d) return '';
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
    : d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
}

/** The changes shown before "Show all". */
const FIRST_CHANGES = 5;

function ChangeLogSection() {
  // `null` from the desktop: it keeps no log yet ("Nothing yet", like an empty one).
  const [answer, refresh, error] = usePoll(() => control.log(100).then((v) => ({ v })), 10000);
  const [all, setAll] = useState(false);
  const entries = answered(answer, error);
  const newest: ControlLogEntry[] = entries ? newestFirst(entries) : [];
  const list = all ? newest : newest.slice(0, FIRST_CHANGES);
  return (
    <section className="model-section">
      <div className="section-title-row">
        <h3 className="section-title">What the Agent changed</h3>
        <button type="button" className="btn-tiny" onClick={() => void refresh()}>
          <RotateCcw size={12} /> Refresh
        </button>
      </div>
      {entries === undefined ? (
        <p className="form-hint">Checking…</p>
      ) : list.length === 0 ? (
        <p className="form-hint">Nothing yet. Each change the Agent makes to OAIY shows here, newest first.</p>
      ) : (
        <ol className="agent-changes">
          {list.map((e, i) => {
            const row = describeChange(e);
            return (
              <li key={`${String(e.at)}-${e.tool}-${i}`} className={row.ok ? undefined : 'is-failed'}>
                <span className="agent-change-mark" aria-label={row.ok ? 'Done' : 'Failed'}>
                  {row.ok ? <Check size={12} strokeWidth={2.6} /> : <X size={12} strokeWidth={2.6} />}
                </span>
                <span className="agent-change-text">
                  <strong>
                    {row.title}
                    {row.subject && <code>{row.subject}</code>}
                  </strong>
                  <small>
                    {[row.session, row.ok ? null : 'did not work'].filter(Boolean).join(' · ')}
                    {row.session || !row.ok ? ' · ' : ''}
                    <span className="setup-mono">{row.tool}</span>
                  </small>
                </span>
                <time dateTime={row.when?.toISOString()} title={row.when?.toLocaleString()}>
                  {when(row.when)}
                </time>
              </li>
            );
          })}
        </ol>
      )}
      {newest.length > FIRST_CHANGES && (
        <div className="form-actions">
          <button type="button" className="btn-tiny" onClick={() => setAll((a) => !a)}>
            {all ? 'Show the newest' : `Show all ${newest.length}`}
          </button>
        </div>
      )}
    </section>
  );
}
