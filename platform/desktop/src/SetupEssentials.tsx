import { useState, type ReactNode } from 'react';
import { Bot, Check, Cpu, MessageSquare, Plug, Puzzle, ShieldCheck, Sparkles, TriangleAlert } from 'lucide-react';
import { AgentProviderPicker, type ProviderPick } from './AgentProviderPicker';
import ChatGptConnector from './ChatGptConnector';
import { bridge, nodeRuntime, type AgentModelSource, type CodexStatus, type EngineCatalog, type EngineRecommendation } from './api';
import { AgentMaySwitch, type ControlSwitch } from './AgentAccess';
import { aiChoices, localModelLine, type FirstRunStep, type ModelLine } from './setupFlow';
import { EngineModelCard, StepHeader, usePoll } from './SetupParts';

/**
 * The first-run wizard's essentials: welcome, the Agent's AI (on this
 * computer, ChatGPT, or an AI provider such as LM Studio), what the Agent may
 * change, and the hand-off to the Agent, which sets up the rest with the
 * person in a chat.
 */

export function WelcomeStep({ kicker }: { kicker: string }) {
  const [runtime] = usePoll(() => bridge.status(), 5000);
  const [installing, setInstalling] = useState(false);
  const node = runtime?.nodeRuntime;
  const parts: Array<{ icon: ReactNode; title: string; text: string }> = [
    { icon: <Sparkles size={17} />, title: 'Your AI', text: 'What the Agent thinks with: a language model on this computer, your ChatGPT account, or a provider such as LM Studio.' },
    { icon: <Bot size={17} />, title: 'The Agent', text: 'Your assistant in OAIY. You choose whether it may set up and change OAIY for you.' },
    { icon: <Puzzle size={17} />, title: 'The rest, with the Agent', text: 'Plugins such as the AI receptionist, your phone, your business hours and your apps: the Agent sets them up with you in a chat.' },
  ];
  return (
    <div className="setup-welcome">
      <StepHeader
        kicker={kicker}
        title="Welcome to OAIY"
        description="Orchestrate AI Yourself: your models, your devices and your flows, on this computer. Two choices get it going; then the Agent sets up the rest with you, or you can do it step by step."
      />
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
        <ShieldCheck size={13} /> OAIY runs on this computer. Keys, conversations and recordings are kept on this device.
      </p>
    </div>
  );
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
}) {
  const line = localModelLine(rec, catalog);
  const gpus = rec?.local.gpus ?? [];
  const head = (
    <StepHeader
      kicker={kicker}
      title="Your AI"
      state={state}
      description={`What the Agent thinks with: a language model on this computer, your ChatGPT account${onProviderPick ? ', or an AI provider such as LM Studio' : ''}. You can change it any time in Settings → Agent.`}
    />
  );
  if (!rec) {
    return (
      <>
        {head}
        <p className="form-hint">Looking at this computer…</p>
      </>
    );
  }
  // Both choices side by side, the recommended one first; what the chosen one needs goes below them both.
  const option = (source: AgentModelSource): ReactNode => {
    const selected = choice === source;
    const recommended = rec.recommend === source;
    if (source === 'provider') {
      return (
        <label key={source} className={`ai-choice ai-choice-head${selected ? ' is-selected' : ''}`}>
          <input type="radio" name="ai-source" value={source} checked={selected} onChange={() => onChoose(source)} />
          <span className="ai-choice-text">
            <strong>
              <span className="ai-choice-icon" aria-hidden>
                <Plug size={15} />
              </span>
              An AI provider
              {providerPick && (
                <span className="badge badge-ok">
                  <Check size={11} strokeWidth={2.6} /> Ready
                </span>
              )}
            </strong>
            <small>LM Studio or Ollama on this computer, or an API key (OpenAI, OpenRouter, Gemini): the models you already run or pay for. Keys stay on this computer.</small>
            {providerPick && <small className="ai-choice-fact">{`${providerPick.name} · ${providerPick.model}`}</small>}
          </span>
        </label>
      );
    }
    const local = source === 'engine';
    const ready = local ? line.kind === 'chosen' : codex?.connected === true;
    return (
      <label key={source} className={`ai-choice ai-choice-head${selected ? ' is-selected' : ''}`}>
        <input type="radio" name="ai-source" value={source} checked={selected} onChange={() => onChoose(source)} />
        <span className="ai-choice-text">
          <strong>
            <span className="ai-choice-icon" aria-hidden>
              {local ? <Cpu size={15} /> : <MessageSquare size={15} />}
            </span>
            {local ? 'On this computer' : 'ChatGPT'}
            {recommended && <span className="badge badge-pending">Recommended</span>}
            {ready && (
              <span className="badge badge-ok">
                <Check size={11} strokeWidth={2.6} /> Ready
              </span>
            )}
          </strong>
          <small>
            {local
              ? 'A language model in OAIY’s engines: private, with nothing to pay per message. It needs a graphics card with enough memory.'
              : 'Your ChatGPT account, through OAIY’s Codex connection: nothing to download. Your plan’s limits apply, and what the Agent reads and writes goes to OpenAI.'}
          </small>
          {/* The model it would use; while it is chosen, the card below offers the download instead. */}
          {local && (line.kind === 'chosen' || !selected) && <small className="ai-choice-fact">{modelLineText(line)}</small>}
          {local && rec.local.reason && <small className="ai-choice-fact">{rec.local.reason}</small>}
          {local && gpus.length > 0 && <small className="setup-mono">{gpus.map((g) => `${g.name} · ${g.totalGb} GB`).join('  ·  ')}</small>}
          {!local && codex?.connected && <small className="ai-choice-fact">Signed in{codex.email ? ` as ${codex.email}` : ''}.</small>}
        </span>
      </label>
    );
  };
  return (
    <>
      {head}
      <div className={`ai-choices ${onProviderPick ? 'is-trio' : 'is-pair'}`} role="radiogroup" aria-label="What the Agent thinks with">
        {aiChoices(rec, !!onProviderPick).map(option)}
      </div>
      {choice === 'engine' && (
        <div className="setup-reqs">
          <EngineModelCard
            group="llm"
            catalog={catalog}
            gpuGb={gpus.length > 0 ? Math.max(...gpus.map((g) => g.totalGb)) : undefined}
            onChanged={onCatalogChanged}
            onOpenEngines={onOpenEngines}
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
