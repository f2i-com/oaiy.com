import { useCallback, useEffect, useMemo, useState } from 'react';
import { Loader2, Plug, RotateCcw } from 'lucide-react';
import { aiProviders, type AiProviderPublic } from './api';
import { errorText } from './SetupParts';

/**
 * An AI provider for the Agent to think with: one already set up in AI
 * providers, or one added here from a preset (LM Studio, Ollama, OpenAI,
 * OpenRouter, Google Gemini, or any OpenAI-compatible server), and a model
 * from the list it gives. The setup wizard and Settings → Agent both use it.
 *
 * The Agent works with tool calls, which the desktop relays to OpenAI-
 * compatible providers only, so an Anthropic provider is not offered. Keys
 * stay on this computer: they go to the desktop's provider store, which never
 * hands them back.
 */

/** What was chosen: the provider's id, the model, and the provider's name. */
export interface ProviderPick {
  provider: string;
  model: string;
  name: string;
}

export interface ProviderPreset {
  /** The provider id it is saved as (another OpenAI-compatible server gets a free one). */
  id: string;
  name: string;
  baseUrl: string;
  /** Whether it needs an API key, may take one, or takes none. */
  key: 'required' | 'optional' | 'none';
  help: string;
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  { id: 'lm-studio', name: 'LM Studio', baseUrl: 'http://localhost:1234/v1', key: 'none', help: 'In LM Studio, load a model and start its server (Developer → Start Server; it listens on port 1234).' },
  { id: 'ollama', name: 'Ollama', baseUrl: 'http://localhost:11434/v1', key: 'none', help: 'Start Ollama and pull a model first (for example: ollama pull qwen3.5).' },
  { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', key: 'required', help: 'An API key from platform.openai.com. OpenAI charges for what the Agent uses.' },
  { id: 'openrouter', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', key: 'required', help: 'One key for many vendors’ models, from openrouter.ai/keys. Set a spending limit on the key there.' },
  { id: 'gemini', name: 'Google Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', key: 'required', help: 'A key from Google AI Studio (aistudio.google.com/apikey), through Google’s OpenAI-compatible address.' },
  { id: 'openai-compatible', name: 'Another OpenAI-compatible server', baseUrl: '', key: 'optional', help: 'vLLM, llama.cpp’s server, Jan, LocalAI or a hosted API: its address ends in /v1. Most local servers need no key.' },
];

/** Is `url` on this computer or a private network (so the desktop may reach it over plain http)? */
export function looksLocal(url: string): boolean {
  let host: string;
  try {
    host = new URL(url).hostname.replace(/^\[|\]$/g, '').toLowerCase();
  } catch {
    return false;
  }
  return (
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host.endsWith('.local') ||
    host.endsWith('.lan') ||
    host === '::1' ||
    host === '0.0.0.0' ||
    /^127\./.test(host) ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host)
  );
}

/** The providers the Agent can think with: on, and OpenAI-compatible (Anthropic's API carries no tool calls through the desktop yet). */
export function agentCapable(list: AiProviderPublic[]): AiProviderPublic[] {
  return list.filter((p) => p.enabled && p.protocol === 'openai' && (p.capabilities.length === 0 || p.capabilities.includes('chat')));
}

/** A free id for `base` among `taken` (`base`, `base-2`, …). */
export function freeId(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}-${i}`)) return `${base}-${i}`;
}

/** The model to start on: the one chosen before, else the one the provider was set up with, else its first. */
export function startingModel(models: string[], chosen: string | null | undefined, configured: string | null | undefined): string | null {
  if (chosen && models.includes(chosen)) return chosen;
  if (configured && models.includes(configured)) return configured;
  return models[0] ?? null;
}

const PRESET = 'preset:';

export function AgentProviderPicker({ value, onPick, disabled }: { value: ProviderPick | null; onPick: (pick: ProviderPick) => void; disabled?: boolean }) {
  const [providers, setProviders] = useState<AiProviderPublic[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const usable = useMemo(() => agentCapable(providers ?? []), [providers]);
  const [selected, setSelected] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState('');
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<{ id: string; list: string[] } | null>(null);

  const loadList = useCallback(async () => {
    try {
      setProviders((await aiProviders.list()).providers);
      setListError(null);
    } catch (e) {
      setListError(errorText(e));
    }
  }, []);
  useEffect(() => {
    void loadList();
  }, [loadList]);

  // What the select shows: the choice made here, else the one the Agent is on, else the first provider, else LM Studio.
  const current = selected ?? (value && usable.some((p) => p.id === value.provider) ? value.provider : usable[0]?.id ?? `${PRESET}lm-studio`);
  const preset = current.startsWith(PRESET) ? PROVIDER_PRESETS.find((p) => `${PRESET}${p.id}` === current) ?? null : null;
  const provider = preset ? null : usable.find((p) => p.id === current) ?? null;

  /** Ask `p` for its models; with them, pick the model to start on. */
  const loadModels = useCallback(
    async (p: AiProviderPublic) => {
      setBusy(true);
      setError(null);
      try {
        const list = await aiProviders.models(p.id);
        setModels({ id: p.id, list });
        const model = startingModel(list, value?.provider === p.id ? value.model : null, p.model);
        if (model) onPick({ provider: p.id, model, name: p.name });
        else setError(`${p.name} lists no models: load or download one there first.`);
      } catch (e) {
        const local = p.allowLocal || looksLocal(p.baseUrl);
        setModels(null);
        setError(`${p.name} did not answer at ${p.baseUrl}${local ? ': is its server running, with a model loaded?' : ': check the address and the key.'} (${errorText(e)})`);
      } finally {
        setBusy(false);
      }
    },
    [onPick, value],
  );

  // A provider chosen (or the one the Agent is on): its models.
  const providerId = provider?.id ?? null;
  useEffect(() => {
    if (provider && models?.id !== provider.id) void loadModels(provider);
    // Only when the provider shown changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerId]);

  const choose = (id: string) => {
    setSelected(id);
    setError(null);
    setModels(null);
    const p = id.startsWith(PRESET) ? PROVIDER_PRESETS.find((x) => `${PRESET}${x.id}` === id) : null;
    setBaseUrl(p?.baseUrl ?? '');
    setKey('');
  };

  /** Save the preset as a provider (with its key), then ask it for its models. */
  const connect = async () => {
    if (!preset) return;
    const url = (baseUrl || preset.baseUrl).trim();
    if (!url) {
      setError('Enter the server’s address (it ends in /v1).');
      return;
    }
    if (preset.key === 'required' && !key.trim()) {
      setError(`Enter your ${preset.name} API key.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const taken = (providers ?? []).map((p) => p.id);
      // A preset added again replaces its own entry; another server gets a free id.
      const id = preset.id === 'openai-compatible' ? freeId(preset.id, taken) : preset.id;
      const name = preset.id === 'openai-compatible' ? (() => { try { return new URL(url).host; } catch { return preset.name; } })() : preset.name;
      await aiProviders.upsert({ id, name, protocol: 'openai', baseUrl: url, capabilities: ['chat'], enabled: true, allowLocal: looksLocal(url) || url.startsWith('http:') });
      if (key.trim()) await aiProviders.setKey(id, key.trim());
      const list = (await aiProviders.list()).providers;
      setProviders(list);
      setSelected(id);
      setKey('');
      const saved = list.find((p) => p.id === id);
      if (saved) await loadModels(saved);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const off = disabled || busy;
  return (
    <div className="agent-provider">
      <label className="form-row">
        <span>Provider</span>
        <select value={current} disabled={off} onChange={(e) => choose(e.target.value)} aria-label="Provider">
          {usable.length > 0 && (
            <optgroup label="Set up in AI providers">
              {usable.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="Add one">
            {PROVIDER_PRESETS.map((p) => (
              <option key={p.id} value={`${PRESET}${p.id}`}>
                {p.name}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      {listError && <p className="card-warn">The AI providers could not be read: {listError}</p>}
      {preset && (
        <form
          className="agent-provider-add"
          onSubmit={(e) => {
            e.preventDefault();
            void connect();
          }}
        >
          <small className="form-hint">{preset.help}</small>
          <label className="form-row">
            <span>Address</span>
            <input type="text" aria-label="Address" placeholder="http://localhost:8000/v1" value={baseUrl || preset.baseUrl} disabled={off} onChange={(e) => setBaseUrl(e.target.value)} />
          </label>
          {preset.key !== 'none' && (
            <label className="form-row">
              <span>API key{preset.key === 'optional' ? ' (if the server asks for one)' : ''}</span>
              <input type="password" aria-label="API key" autoComplete="off" value={key} disabled={off} onChange={(e) => setKey(e.target.value)} />
            </label>
          )}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={off}>
              {busy ? <Loader2 size={14} className="spin" /> : <Plug size={14} />} Connect
            </button>
          </div>
        </form>
      )}
      {provider && (
        <label className="form-row">
          <span>Model, from {provider.name}</span>
          {models?.id === provider.id && models.list.length > 0 ? (
            <select
              aria-label="Model"
              value={value?.provider === provider.id ? value.model : ''}
              disabled={off}
              onChange={(e) => onPick({ provider: provider.id, model: e.target.value, name: provider.name })}
            >
              {models.list.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <small className="form-hint">{busy ? `Asking ${provider.name} for its models…` : 'No models listed yet.'}</small>
          )}
        </label>
      )}
      {error && (
        <p className="card-warn" role="alert">
          {error}{' '}
          {provider && (
            <button type="button" className="btn-tiny" disabled={off} onClick={() => void loadModels(provider)}>
              <RotateCcw size={12} /> Try again
            </button>
          )}
        </p>
      )}
      <small className="form-hint">The Agent works OAIY with tool calls: choose a model that supports them (most recent Qwen, Llama, Mistral, GPT and Gemini models do).</small>
    </div>
  );
}
