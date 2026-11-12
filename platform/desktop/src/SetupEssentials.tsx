import { useState, type ReactNode } from 'react';
import { Check, Cpu, Loader2, MessageSquare, Plug, ShieldCheck, TriangleAlert } from 'lucide-react';
import { AgentProviderPicker, type ProviderPick } from './AgentProviderPicker';
import ChatGptConnector from './ChatGptConnector';
import { bridge, nodeRuntime, type AgentModelSource, type AgentPreferences, type CodexStatus, type EngineCatalog, type EngineRecommendation } from './api';
import { AgentMaySwitch, type ControlSwitch } from './AgentAccess';
import { aiChoices, isRecommended, localModelLine, type FirstRunStep, type ModelLine } from './setupFlow';
import { EngineModelCard, errorText, StepHeader, usePoll, type EngineAction } from './SetupParts';

/**
 * The first-run wizard's essentials: the Agent's AI (on this computer,
 * ChatGPT, or an AI provider such as LM Studio), what the Agent may change,
 * and the hand-off to the Agent, which sets up the rest with the person in a
 * chat. (There is no welcome step before them: it asked only for "Get
 * started". Its one sentence heads the first step, which also keeps what it
 * said when flows cannot run yet, and that OAIY keeps things on this computer.)
 */

/** Flows cannot run on this computer yet (the flow runtime is not ready): said, with Node installed in one press where that is what is missing. */
function RuntimeNote() {
  const [runtime] = usePoll(() => bridge.status(), 5000);
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const node = runtime?.nodeRuntime;
  if (!runtime || runtime.ready) return null;
  return (
    <>
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
              setInstallError(null);
              void nodeRuntime
                .install()
                .catch((e: unknown) => setInstallError(`Node was not installed: ${errorText(e)}`))
                .finally(() => setInstalling(false));
            }}
          >
            {node.installing || installing ? 'Installing Node…' : `Install Node ${node.installsVersion}`}
          </button>
        )}
      </div>
      {installError && (
        <p className="card-warn" role="alert">
          {installError}
        </p>
      )}
    </>
  );
}

/** The engine's default language model where it is a paired computer's (OAIY Network): which, and on which computer. */
export interface PairedModel {
  id: string;
  model: string;
  computer: string;
}

/**
 * The paired computer's language model the Agent runs on, where it is on the engine with no model of its own named and
 * the engine's default is a paired computer's (the desktop says which, beside the preference); null otherwise.
 */
export function pairedDefault(prefs: AgentPreferences | null | undefined): PairedModel | null {
  if (!prefs?.engineDefault || prefs.model.source !== 'engine' || prefs.model.model) return null;
  const id = prefs.engineDefault;
  // (`engine:llm:node.<nodeId>/<model>`: the model is what follows the computer's id)
  const slash = id.indexOf('/');
  return { id, model: slash >= 0 ? id.slice(slash + 1) : id, computer: prefs.engineDefaultName || 'a paired computer' };
}

/** That model, in a line. */
export function pairedModelText(p: PairedModel): string {
  return `Uses ${p.model} on ${p.computer} (Engines → Other computers).`;
}

/** The local model, in a line: what Engines has chosen, or the catalog's recommended one. */
export function modelLineText(line: ModelLine): string {
  if (line.kind === 'chosen') return `Uses ${line.name}, chosen in Engines.`;
  if (line.kind === 'recommended') return `Nothing is chosen in Engines yet. The catalog recommends ${line.name}${line.detail ? ` (${line.detail})` : ''}.`;
  return 'Nothing is chosen in Engines yet: choose a language model there.';
}

export function YourAiStep({
  kicker,
  state,
  rec,
  catalog,
  codex,
  choice,
  onChoose,
  onCatalogChanged,
  onSignedIn,
  onOpenEngines,
  providerPick,
  onProviderPick,
  onEngineAction,
  providerPending,
}: {
  kicker: string;
  state: FirstRunStep['state'];
  /** The recommendation (or its fallback); `null` while it is asked. */
  rec: EngineRecommendation | null;
  catalog: EngineCatalog | null;
  codex: CodexStatus | null;
  choice: AgentModelSource | null;
  onChoose: (c: AgentModelSource) => void;
  onCatalogChanged: () => void;
  onSignedIn: (s: CodexStatus) => void;
  onOpenEngines: () => void;
  /** The provider and model chosen here, or the ones the Agent is on. */
  providerPick?: ProviderPick | null;
  /** Given, the provider is a choice (a desktop that keeps the Agent's model). */
  onProviderPick?: (p: ProviderPick) => void;
  /** Given, the page draws the model's Download button (its footer's) and is told what it does (EngineModelCard). */
  onEngineAction?: (action: EngineAction | null) => void;
  /** Whether the desktop keeps the Agent's model is still being asked: the provider's tile is drawn from the first
   *  paint, in its place, and waits (it arrived a moment after the other two, and the row was seen to change). */
  providerPending?: boolean;
}) {
  const withProvider = !!onProviderPick || !!providerPending;
  const line = localModelLine(rec, catalog);
  const gpus = rec?.local.gpus ?? [];
  // Why it is (or is not) recommended, where that adds to the line naming the model: with a model chosen, it only says
  // so again. And the GPUs on their own line where the reason does not name them already.
  const reason = line.kind === 'chosen' ? null : rec?.local.reason ?? null;
  const gpuLine = gpus.length > 0 && !(reason && gpus.every((g) => reason.includes(g.name)));
  const head = (
    <>
      <StepHeader
        kicker={kicker}
        title="Your AI"
        state={state}
        // (the welcome's one sentence, and what this step chooses)
        description={`Welcome to OAIY. Two choices get it going, and this is the first: what the Agent thinks with. A language model on this computer, your ChatGPT account${withProvider ? ', or an AI provider such as LM Studio' : ''}. You can change it any time in Settings → Agent.`}
      />
      <RuntimeNote />
    </>
  );
  // The choices in one row of tiles, always in the same order: its icon, its name, and whether it is recommended or
  // ready. Until the computer has said what it has (`rec` is null), the tiles are there to choose from with none
  // marked, and a quiet line says it is being looked at; the recommendation is then marked where its tile stands.
  // What each one is, and what it would use, is said under the row for the one chosen alone; what it needs (a model
  // to download, a sign-in, a provider) goes below that.
  const NAME: Record<AgentModelSource, string> = { engine: 'On this computer', chatgpt: 'ChatGPT', provider: 'An AI provider' };
  const readyOf = (source: AgentModelSource) => (source === 'engine' ? line.kind === 'chosen' : source === 'chatgpt' ? codex?.connected === true : !!providerPick);
  const option = (source: AgentModelSource): ReactNode => {
    const selected = choice === source;
    const recommended = !!rec && isRecommended(rec, source);
    const ready = readyOf(source);
    // (the provider's tile while it is not known whether this desktop can keep one: in its place, not yet to choose)
    const pending = source === 'provider' && !onProviderPick;
    return (
      <label key={source} className={`ai-choice ai-tile${selected ? ' is-selected' : ''}${pending ? ' is-pending' : ''}`}>
        <input type="radio" name="ai-source" value={source} checked={selected} disabled={pending} aria-describedby={selected ? 'ai-choice-about' : undefined} onChange={() => onChoose(source)} />
        <span className="ai-tile-icon" aria-hidden>
          {source === 'engine' ? <Cpu size={17} /> : source === 'chatgpt' ? <MessageSquare size={17} /> : <Plug size={17} />}
        </span>
        <span className="ai-tile-text">
          <strong>{NAME[source]}</strong>
          {(recommended || ready) && (
            <span className="ai-tile-marks">
              {recommended && <span className="badge badge-pending">Recommended</span>}
              {ready && (
                <span className="badge badge-ok">
                  <Check size={11} strokeWidth={2.6} /> Ready
                </span>
              )}
            </span>
          )}
        </span>
      </label>
    );
  };
  const about =
    choice === 'engine' ? (
      <>
        <p>A language model in OAIY’s engines: private, with nothing to pay per message. It needs a graphics card with enough memory.</p>
        {/* The model it uses; with none chosen yet, the card below offers the download instead. */}
        {line.kind === 'chosen' && <p className="ai-choice-fact">{modelLineText(line)}</p>}
        {reason && <p className="ai-choice-fact">{reason}</p>}
        {gpuLine && <p className="setup-mono">{gpus.map((g) => `${g.name} · ${g.totalGb} GB`).join('  ·  ')}</p>}
      </>
    ) : choice === 'chatgpt' ? (
      <>
        <p>Your ChatGPT account, through OAIY’s Codex connection: nothing to download. Your plan’s limits apply, and what the Agent reads and writes goes to OpenAI.</p>
        {codex?.connected && <p className="ai-choice-fact">Signed in{codex.email ? ` as ${codex.email}` : ''}.</p>}
      </>
    ) : choice === 'provider' ? (
      <>
        <p>LM Studio or Ollama on this computer, or an API key (OpenAI, OpenRouter, Google): the models you already run or pay for. Keys stay on this computer.</p>
        {providerPick && <p className="ai-choice-fact">{`${providerPick.name} · ${providerPick.model}`}</p>}
      </>
    ) : null;
  return (
    <>
      {head}
      {/* (three columns whether or not the third choice is there yet: the first two keep their places and widths) */}
      <div className="ai-choices ai-tiles is-trio" role="radiogroup" aria-label="What the Agent thinks with">
        {aiChoices(rec, withProvider).map(option)}
      </div>
      {!rec && (
        <p className="form-hint ai-looking" role="status">
          <Loader2 size={13} className="spin" aria-hidden /> Looking at this computer…
        </p>
      )}
      {about && (
        <div className="ai-choice-about" id="ai-choice-about">
          {about}
        </div>
      )}
      {choice === 'engine' && (
        <div className="setup-reqs">
          <EngineModelCard
            group="llm"
            catalog={catalog}
            gpuGb={gpus.length > 0 ? Math.max(...gpus.map((g) => g.totalGb)) : undefined}
            onChanged={onCatalogChanged}
            onOpenEngines={onOpenEngines}
            onAction={onEngineAction}
            why="The Agent uses the language model chosen in Engines, whatever it is."
          />
        </div>
      )}
      {choice === 'provider' && onProviderPick && (
        <div className="setup-req ai-provider">
          <div className="setup-req-body">
            <AgentProviderPicker value={providerPick ?? null} onPick={onProviderPick} />
          </div>
        </div>
      )}
      {choice === 'chatgpt' && (
        <div className="setup-req ai-chatgpt">
          <div className="setup-req-body">
            <ChatGptConnector bare onConnected={onSignedIn} />
            <small className="form-hint">The sign-in is kept by Codex on this computer: OAIY never sees your password or a token.</small>
          </div>
        </div>
      )}
      <p className="setup-trust">
        <ShieldCheck size={13} /> OAIY runs on this computer. Keys, conversations and recordings are kept on it.
      </p>
    </>
  );
}

export function AgentStep({ kicker, state, control }: { kicker: string; state: FirstRunStep['state']; control: ControlSwitch }) {
  return (
    <>
      <StepHeader
        kicker={kicker}
        title="The Agent"
        state={state}
        description="OAIY’s assistant. You chat with it on the Agent page, and it can work OAIY for you: its models, services, plugins, flows and settings."
      />
      <AgentMaySwitch control={control} />
    </>
  );
}

export function HandoffStep({
  kicker,
  state,
  aiLine,
  aiReady,
  mayChange,
  onPick,
}: {
  kicker: string;
  state: FirstRunStep['state'];
  /** What the Agent thinks with, in words. */
  aiLine: string;
  aiReady: boolean;
  /** The switch: `null` while unknown. */
  mayChange: boolean | null;
  onPick: (id: 'ai' | 'agent') => void;
}) {
  const rows: Array<{ id: 'ai' | 'agent'; title: string; text: string; done: boolean }> = [
    { id: 'ai', title: 'Your AI', text: aiReady ? aiLine : 'Not set up yet: the Agent needs it to answer.', done: aiReady },
    {
      id: 'agent',
      title: 'The Agent may set up and change OAIY',
      text: mayChange === false ? 'Off: it will tell you what to do instead of doing it.' : 'On: every change it makes is listed in Settings → Agent.',
      done: mayChange !== false,
    },
  ];
  return (
    <div className="setup-handoff">
      <StepHeader
        kicker={kicker}
        title="Continue with the Agent"
        state={state}
        description="The Agent sets up the rest with you in a chat: it asks what you want OAIY to do, then installs and sets up plugins such as the AI receptionist, pairs your phone, fills in your business hours and connects your apps."
      />
      <ul className="setup-summary">
        {rows.map((r) => (
          <li key={r.id} className={r.done ? 'is-done' : 'is-todo'}>
            <span className="setup-summary-mark" aria-hidden>
              {r.done ? <Check size={13} strokeWidth={2.6} /> : <TriangleAlert size={12} />}
            </span>
            <span>
              <strong>{r.title}</strong>
              <small>{r.text}</small>
            </span>
            <button type="button" className="btn-tiny" onClick={() => onPick(r.id)}>
              Change
            </button>
          </li>
        ))}
      </ul>
      <p className="form-hint">Rather do it yourself? “Set up the rest myself” goes on step by step: plugins, their devices and your apps.</p>
    </div>
  );
}
