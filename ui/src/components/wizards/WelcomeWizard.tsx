/**
 * WelcomeWizard — first-run onboarding.
 *
 * Opt-in modal that walks a brand-new user from "I just opened oaiy-web"
 * to "I have a working flow" in a few clicks. Auto-opens on first load when
 * the `oaiy.wizard.completed` localStorage flag is unset; can be re-opened
 * from the header help button (also the way to add more services later).
 *
 * Four steps:
 *   1. Welcome — what is this thing, why am I here
 *   2. Add services — pick a preset, fill the form, Add; repeat for as many
 *      services as you like (local engines and/or cloud APIs with your key)
 *   3. Starter flow — pick one of your services and auto-generate a runnable
 *      3-node flow (Text → service → Output, or Text → image → Image)
 *   4. Done — quick tip cards, "Open my flow"
 *
 * Skippable from any step. The wizard only mutates state when the user
 * explicitly clicks Add / Create — closing early is safe.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import { v4 as uuidv4 } from 'uuid';
import { markWizardCompleted } from '../../lib/wizardPrefs';
import { useFocusTrap } from '../../hooks/useFocusTrap';

// -------------------------------------------------------------------
// Service presets the wizard offers. A curated subset of ServicesTab's
// TEMPLATE_PRESETS, picked for "most likely to get a brand-new user
// running fast". Cloud (bring-your-own-key) presets are listed first.
// -------------------------------------------------------------------
interface WizardPreset {
  id: string;
  label: string;
  icon: string;
  /** 'image' presets build a Text → image_gen → Image starter flow. */
  kind: 'chat' | 'image';
  hint: string;
  endpoint: string;
  needsApiKey: boolean;
  apiKeyConstant?: string;
  apiKeyDocsUrl?: string;
  fields: Omit<CustomService, 'id' | 'name' | 'endpoint'>;
}

const PRESETS: WizardPreset[] = [
  {
    id: 'openai-cloud',
    label: 'OpenAI Cloud',
    icon: '🤖',
    kind: 'chat',
    // OpenAI returns CORS headers on real responses (its SDK has a
    // `dangerouslyAllowBrowser` flag for exactly this), so a browser BYOK call
    // works. The key is stored locally in this browser, never sent to oaiy.com.
    hint: 'api.openai.com chat — needs your OpenAI API key (stored locally in your browser).',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    needsApiKey: true,
    apiKeyConstant: 'OPENAI_API_KEY',
    apiKeyDocsUrl: 'https://platform.openai.com/api-keys',
    fields: {
      method: 'POST',
      // {{apiKeyRaw}} (not {{apiKey}}) — inside a quoted JSON string the value
      // must be substituted RAW; {{apiKey}} JSON-escapes (adds quotes), which
      // would yield `"Bearer "<key>""` → invalid JSON.
      headers: '{"Authorization": "Bearer {{apiKeyRaw}}"}',
      bodyTemplate:
        '{\n  "model": "gpt-5.4-mini",\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
      responseType: 'json',
      responsePath: 'choices.0.message.content',
      apiFormat: 'openai',
      model: 'gpt-5.4-mini',
      apiKeyConstant: 'OPENAI_API_KEY',
      icon: '🤖',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'openai-image',
    label: 'OpenAI GPT Image 2',
    icon: '🎨',
    kind: 'image',
    hint: 'GPT Image 2 text-to-image — uses your OpenAI API key. Builds an image flow.',
    endpoint: 'https://api.openai.com/v1/images/generations',
    needsApiKey: true,
    apiKeyConstant: 'OPENAI_API_KEY',
    apiKeyDocsUrl: 'https://platform.openai.com/api-keys',
    fields: {
      method: 'POST',
      headers: '{"Authorization": "Bearer {{apiKeyRaw}}"}',
      bodyTemplate:
        '{\n  "model": "gpt-image-2",\n  "prompt": {{input}},\n  "n": 1,\n  "size": "1024x1024"\n}',
      responseType: 'json',
      // GPT Image returns base64; the Image Generation node wraps it into a
      // data: URL for display.
      responsePath: 'data.0.b64_json',
      apiFormat: 'openai',
      model: 'gpt-image-2',
      apiKeyConstant: 'OPENAI_API_KEY',
      icon: '🎨',
      nodeTypes: ['image_gen', 'service_call'],
    },
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    icon: '✨',
    kind: 'chat',
    hint: 'Claude /v1/messages — needs your Anthropic API key. Works in the browser.',
    endpoint: 'https://api.anthropic.com/v1/messages',
    needsApiKey: true,
    apiKeyConstant: 'ANTHROPIC_API_KEY',
    apiKeyDocsUrl: 'https://console.anthropic.com/settings/keys',
    fields: {
      method: 'POST',
      // `anthropic-dangerous-direct-browser-access: true` is Anthropic's
      // explicit opt-in for direct browser calls — without it the request is
      // CORS-blocked.
      headers: '{"anthropic-version": "2023-06-01", "x-api-key": "{{apiKeyRaw}}", "anthropic-dangerous-direct-browser-access": "true"}',
      bodyTemplate:
        '{\n  "model": "claude-sonnet-4-6",\n  "max_tokens": 4096,\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
      responseType: 'json',
      responsePath: 'content.0.text',
      apiFormat: 'anthropic',
      model: 'claude-sonnet-4-6',
      apiKeyConstant: 'ANTHROPIC_API_KEY',
      icon: '✨',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'ollama',
    label: 'Ollama',
    icon: '🦙',
    kind: 'chat',
    hint: 'Ollama /api/chat — defaults to llama3, no key needed.',
    endpoint: 'http://localhost:11434/api/chat',
    needsApiKey: false,
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate:
        '{\n  "model": "llama3",\n  "messages": [{"role": "user", "content": {{input}}}],\n  "stream": false\n}',
      responseType: 'json',
      responsePath: 'message.content',
      apiFormat: 'ollama',
      model: 'llama3',
      icon: '🦙',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'openai-compatible',
    label: 'OpenAI-compatible',
    icon: '🔌',
    kind: 'chat',
    hint: 'vLLM, llama-server, Ollama /v1, LM Studio /v1/chat/completions … (no key for local).',
    endpoint: 'http://localhost:1234/v1/chat/completions',
    needsApiKey: false,
    fields: {
      method: 'POST',
      headers: '{}',
      // Model embedded as a JSON string literal (not {{model}}) so editing the
      // Model field in the properties panel visibly rewrites the body — see the
      // handleFieldChange('model') branch in PropertiesPanel.tsx.
      bodyTemplate:
        '{\n  "model": "local-model",\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
      responseType: 'json',
      responsePath: 'choices.0.message.content',
      apiFormat: 'openai',
      model: 'local-model',
      icon: '🔌',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'lm-studio',
    label: 'LM Studio',
    icon: '🖥️',
    kind: 'chat',
    hint: 'LM Studio /api/v1/chat — newer simple-input format. No key needed.',
    endpoint: 'http://localhost:1234/api/v1/chat',
    needsApiKey: false,
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate:
        '{\n  "model": "local-model",\n  "input": {{input}}\n}',
      responseType: 'json',
      responsePath: '',
      apiFormat: 'openai',
      model: 'local-model',
      icon: '🖥️',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
];

/** An image preset/service builds an image flow rather than a chat flow. */
function isImageService(svc: Pick<CustomService, 'nodeTypes'>): boolean {
  const tags = svc.nodeTypes ?? [];
  return tags.includes('image_gen') && !tags.includes('ai_llm');
}

// -------------------------------------------------------------------
// Public component API
// -------------------------------------------------------------------
export interface WelcomeWizardProps {
  isOpen: boolean;
  onClose: () => void;
  /** Save a service (caller routes to serviceRegistry.saveService). */
  onSaveService: (svc: CustomService) => void;
  /** Save an API key constant (caller routes to project constants). */
  onSaveApiKey: (constantName: string, value: string) => void;
  /**
   * Create a starter flow and switch to it. Receives the chosen service so the
   * auto-generated node can be wired to it. The host handles navigation.
   */
  onCreateStarterFlow: (svc: CustomService) => void;
  /**
   * Services the user already has (non-built-in). The wizard detects these so a
   * returning user can REUSE one for a starter flow instead of re-creating it —
   * they're listed alongside anything added this run, marked as already saved,
   * and presets that match one are flagged "added".
   */
  existingServices: CustomService[];
  /**
   * Names of API-key constants that ALREADY hold a value (from a prior session,
   * Settings, or earlier in this wizard run). When the picked preset reuses one
   * of these, the key field tells the user it's already set and can be left
   * blank — e.g. add an OpenAI chat service, then GPT Image, without re-typing
   * OPENAI_API_KEY.
   */
  configuredKeyConstants?: string[];
}

type Step = 1 | 2 | 3 | 4;

export default function WelcomeWizard(props: WelcomeWizardProps) {
  const { isOpen, onClose, onSaveService, onSaveApiKey, onCreateStarterFlow, existingServices, configuredKeyConstants } = props;
  const [step, setStep] = useState<Step>(1);

  // Form state for the service currently being configured. Starts with no
  // preset picked so the user explicitly chooses one (the form appears once
  // they do) — avoids accidentally committing a pre-selected default.
  const [presetId, setPresetId] = useState<string>('');
  const [serviceName, setServiceName] = useState<string>('');
  const [endpoint, setEndpoint] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [testState, setTestState] = useState<'idle' | 'testing' | 'ok' | 'err'>('idle');
  const [testMsg, setTestMsg] = useState<string | null>(null);

  // Services the user has added during this wizard run, plus which one the
  // starter flow will be built from.
  const [savedServices, setSavedServices] = useState<CustomService[]>([]);
  const [flowServiceId, setFlowServiceId] = useState<string>('');

  // `undefined` when no preset is selected (the form is hidden / empty).
  const preset = PRESETS.find((p) => p.id === presetId);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Focus the primary CTA on open (the wizard always opens on step 1), NOT the
  // top-right "Skip" — otherwise a reflexive Enter on the auto-opened wizard
  // immediately skips and permanently completes onboarding.
  const primaryCtaRef = useRef<HTMLButtonElement>(null);
  useFocusTrap(dialogRef, isOpen, primaryCtaRef);

  // Key constants that already hold a value — keys configured before this run
  // (Settings / prior session) PLUS ones entered for a service added this run.
  // Lets the form say "already set, leave blank to reuse" for a shared key.
  const providedKeyConstants = new Set<string>([
    ...(configuredKeyConstants ?? []),
    ...(savedServices.map((s) => s.apiKeyConstant).filter(Boolean) as string[]),
    ...(existingServices.map((s) => s.apiKeyConstant).filter(Boolean) as string[]),
  ]);

  // Reset transient state every open so a re-entry is clean.
  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setPresetId('');
      setServiceName('');
      setEndpoint('');
      setApiKey('');
      setTestState('idle');
      setTestMsg(null);
      setSavedServices([]);
      setFlowServiceId('');
    }
  }, [isOpen]);

  // When the user picks a preset, default the form fields to the preset's
  // example values. No-op when no preset is selected (empty "add another" form).
  useEffect(() => {
    if (!preset) return;
    setServiceName(`My ${preset.label}`);
    setEndpoint(preset.endpoint);
    setApiKey('');
    setTestState('idle');
    setTestMsg(null);
  }, [preset]);

  const close = useCallback(() => {
    markWizardCompleted();
    onClose();
  }, [onClose]);

  // Escape-to-dismiss while open.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, close]);

  const runTest = useCallback(async () => {
    if (!endpoint) return;
    setTestState('testing');
    setTestMsg(null);
    const start = performance.now();
    try {
      // Cheap reachability probe — OPTIONS first, fall back to a tiny POST
      // for servers (Ollama) that only answer POST. Not an API-correctness
      // check; we just confirm the URL resolves.
      let resp: Response;
      try {
        resp = await fetch(endpoint, { method: 'OPTIONS' });
      } catch {
        resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        });
      }
      const ms = Math.round(performance.now() - start);
      if (resp.status < 500) {
        setTestState('ok');
        setTestMsg(`Reachable — HTTP ${resp.status} (${ms} ms)`);
      } else {
        setTestState('err');
        setTestMsg(`Server returned HTTP ${resp.status} (${ms} ms) — endpoint is up but unhappy.`);
      }
    } catch (e) {
      const ms = Math.round(performance.now() - start);
      setTestState('err');
      setTestMsg(`Couldn't reach ${endpoint} — ${(e as Error).message} (failed after ${ms} ms)`);
    }
  }, [endpoint]);

  const formValid = !!preset && serviceName.trim() !== '' && endpoint.trim() !== '';

  const resetForm = useCallback(() => {
    setPresetId('');
    setServiceName('');
    setEndpoint('');
    setApiKey('');
    setTestState('idle');
    setTestMsg(null);
  }, []);

  // Build + persist the service currently in the form. Returns it (or null if
  // the form isn't a valid service). Saves the API key only when one was typed,
  // so adding a second service that reuses an existing key constant can be
  // left blank.
  const commitCurrent = useCallback((): CustomService | null => {
    if (!preset || !serviceName.trim() || !endpoint.trim()) return null;
    const svc: CustomService = {
      id: uuidv4(),
      name: serviceName.trim(),
      endpoint: endpoint.trim(),
      ...preset.fields,
    };
    onSaveService(svc);
    if (preset.needsApiKey && preset.apiKeyConstant && apiKey.trim()) {
      onSaveApiKey(preset.apiKeyConstant, apiKey.trim());
    }
    return svc;
  }, [preset, serviceName, endpoint, apiKey, onSaveService, onSaveApiKey]);

  const addAnother = useCallback(() => {
    const svc = commitCurrent();
    if (!svc) return;
    setSavedServices((prev) => [...prev, svc]);
    setFlowServiceId((prev) => prev || svc.id);
    resetForm();
  }, [commitCurrent, resetForm]);

  // Every service available to build a flow from: the ones added this run PLUS
  // any the user already had — deduped by id, this-run first. Lets a returning
  // user REUSE an existing service instead of re-creating it.
  const flowServices: CustomService[] = (() => {
    const seen = new Set<string>();
    const out: CustomService[] = [];
    for (const s of [...savedServices, ...existingServices]) {
      if (s && s.id && !seen.has(s.id)) { seen.add(s.id); out.push(s); }
    }
    return out;
  })();
  // Which of the listed services were added during THIS run (so only those get
  // a remove button — existing ones are already saved and stay).
  const savedThisRunIds = new Set(savedServices.map((s) => s.id));
  const canContinue = formValid || flowServices.length > 0;

  const continueToFlow = useCallback(() => {
    const svc = commitCurrent();
    if (svc) {
      setSavedServices((prev) => [...prev, svc]);
      // Clear the form so going '← Back' from the flow step and clicking
      // Continue again can't re-commit (and re-save the key for) the same
      // service under a fresh id — the duplicate-on-back bug. Mirrors
      // addAnother(), which already resets after committing.
      resetForm();
    }
    // The just-committed service (if any) plus everything already available.
    const targets = svc ? [svc, ...flowServices] : flowServices;
    if (targets.length === 0) return;
    setFlowServiceId((prev) =>
      prev && targets.some((s) => s.id === prev) ? prev : targets[0].id,
    );
    setStep(3);
  }, [commitCurrent, resetForm, flowServices]);

  const removeService = useCallback((id: string) => {
    setSavedServices((prev) => prev.filter((s) => s.id !== id));
    setFlowServiceId((prev) => (prev === id ? '' : prev));
  }, []);

  const createFlowAndAdvance = useCallback(() => {
    const svc = flowServices.find((s) => s.id === flowServiceId) ?? flowServices[0];
    if (!svc) return;
    onCreateStarterFlow(svc);
    setStep(4);
  }, [flowServices, flowServiceId, onCreateStarterFlow]);

  if (!isOpen) return null;

  const totalSteps = 4;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/70 backdrop-blur-sm p-4"
      onClick={close}
    >
      <div
        ref={dialogRef}
        className="relative overflow-hidden bg-white dark:bg-slate-800 rounded-2xl shadow-2xl ring-1 ring-black/5 dark:ring-white/10 w-full max-w-xl p-6 sm:p-8"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Onboarding wizard"
      >
        {/* Soft brand glow bleeding from the top edge — purely decorative,
            clipped by the box's overflow-hidden + rounded corners. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-56 w-[32rem] rounded-full blur-3xl opacity-40 dark:opacity-25"
          style={{ background: 'radial-gradient(closest-side, rgb(var(--accent-primary) / 0.5), transparent)' }}
        />
        {/* `relative` lifts the real content above the absolute glow. */}
        <div className="relative space-y-5">
          {/* Progress dots + close */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5" aria-label={`Step ${step} of ${totalSteps}`}>
              {Array.from({ length: totalSteps }).map((_, i) => (
                <span
                  key={i}
                  className={`h-1.5 rounded-full transition-all ${
                    i + 1 === step
                      ? 'w-6 bg-[rgb(var(--accent-primary))]'
                      : i + 1 < step
                        ? 'w-1.5'
                        : 'w-1.5 bg-slate-300 dark:bg-slate-600'
                  }`}
                  style={
                    i + 1 < step
                      ? { backgroundColor: 'rgb(var(--accent-primary) / 0.55)' }
                      : undefined
                  }
                />
              ))}
            </div>
            <button
              type="button"
              onClick={close}
              className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 text-xs"
              aria-label="Skip wizard"
            >
              Skip
            </button>
          </div>

          {/* Step body */}
          {step === 1 && <WelcomeStep onNext={() => setStep(2)} hasAnyService={existingServices.length > 0} ctaRef={primaryCtaRef} />}
          {step === 2 && (
            <AddServiceStep
              presets={PRESETS}
              presetId={presetId}
              onPresetChange={setPresetId}
              preset={preset}
              serviceName={serviceName}
              onServiceNameChange={setServiceName}
              endpoint={endpoint}
              onEndpointChange={setEndpoint}
              apiKey={apiKey}
              onApiKeyChange={setApiKey}
              testState={testState}
              testMsg={testMsg}
              onTest={runTest}
              services={flowServices}
              savedThisRunIds={savedThisRunIds}
              providedKeyConstants={providedKeyConstants}
              onRemoveService={removeService}
              onBack={() => setStep(1)}
              onAddAnother={addAnother}
              onContinue={continueToFlow}
              formValid={formValid}
              canContinue={canContinue}
            />
          )}
          {step === 3 && (
            <StarterFlowStep
              services={flowServices}
              selectedId={flowServiceId || flowServices[0]?.id || ''}
              onSelect={setFlowServiceId}
              onBack={() => setStep(2)}
              onCreate={createFlowAndAdvance}
            />
          )}
          {step === 4 && <DoneStep onClose={close} />}
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Step components — kept inline; each is small and only used here.
// -------------------------------------------------------------------

function WelcomeStep({ onNext, hasAnyService, ctaRef }: { onNext: () => void; hasAnyService: boolean; ctaRef?: React.RefObject<HTMLButtonElement | null> }) {
  return (
    <>
      <div className="text-center">
        {/* App mark (same as the favicon) — a branded rounded tile reads more
            "product" than a generic emoji. Decorative: the heading carries the
            meaning, so the image is aria-hidden. */}
        <img
          src="/favicon.svg"
          alt=""
          aria-hidden="true"
          width={64}
          height={64}
          draggable={false}
          className="mx-auto mb-3 w-16 h-16 rounded-2xl shadow-lg shadow-indigo-500/30 ring-1 ring-white/20 select-none"
        />
        <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
          Welcome to OAIY
        </h2>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
          A visual flow builder that talks to whatever AI tools you run — local or cloud.
        </p>
      </div>
      <div className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
        <p>In the next few clicks we'll:</p>
        <ul className="space-y-1.5 pl-1">
          <li className="flex items-start gap-2">
            <span aria-hidden="true" className="inline-flex w-5 h-5 rounded-full items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.12)', color: 'rgb(var(--accent-primary))' }}>1</span>
            <span>Connect one or more AI services (OpenAI chat, GPT Image, Anthropic, Ollama, or any HTTP endpoint).</span>
          </li>
          <li className="flex items-start gap-2">
            <span aria-hidden="true" className="inline-flex w-5 h-5 rounded-full items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.12)', color: 'rgb(var(--accent-primary))' }}>2</span>
            <span>Build a starter flow you can run right away.</span>
          </li>
          <li className="flex items-start gap-2">
            <span aria-hidden="true" className="inline-flex w-5 h-5 rounded-full items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.12)', color: 'rgb(var(--accent-primary))' }}>3</span>
            <span>Show you where everything lives.</span>
          </li>
        </ul>
        {hasAnyService && (
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            You already have services configured — add more here, or skip straight to a starter flow.
          </p>
        )}
      </div>
      <div className="flex justify-end pt-2">
        <button
          ref={ctaRef}
          type="button"
          onClick={onNext}
          className="px-4 py-2 text-sm rounded-md bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white font-medium"
        >
          Let's go →
        </button>
      </div>
    </>
  );
}

function AddServiceStep(props: {
  presets: WizardPreset[];
  presetId: string;
  onPresetChange: (id: string) => void;
  preset: WizardPreset | undefined;
  serviceName: string;
  onServiceNameChange: (v: string) => void;
  endpoint: string;
  onEndpointChange: (v: string) => void;
  apiKey: string;
  onApiKeyChange: (v: string) => void;
  testState: 'idle' | 'testing' | 'ok' | 'err';
  testMsg: string | null;
  onTest: () => void;
  /** All services available (existing + added this run), deduped. */
  services: CustomService[];
  /** Ids of services added during this run — only those are removable. */
  savedThisRunIds: Set<string>;
  providedKeyConstants: Set<string>;
  onRemoveService: (id: string) => void;
  onBack: () => void;
  onAddAnother: () => void;
  onContinue: () => void;
  formValid: boolean;
  canContinue: boolean;
}) {
  const {
    presets, presetId, onPresetChange, preset, serviceName, onServiceNameChange,
    endpoint, onEndpointChange, apiKey, onApiKeyChange, testState, testMsg, onTest,
    services, savedThisRunIds, providedKeyConstants, onRemoveService,
    onBack, onAddAnother, onContinue, formValid, canContinue,
  } = props;

  const keyAlreadyProvided =
    !!preset?.apiKeyConstant && providedKeyConstants.has(preset.apiKeyConstant);

  return (
    <>
      <div>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">
          Add your AI services
        </h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
          Connect as many as you like — local engines or cloud APIs with your own key.
          You can manage them later in <strong>Settings → Services</strong>.
        </p>
      </div>

      {/* Services you already have — existing ones (detected) + any added this
          run. Reuse any of them in the starter flow; no need to re-add. */}
      {services.length > 0 && (
        <div
          className="rounded-lg border border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/60 dark:bg-emerald-900/15 p-3"
        >
          <div className="text-[11px] font-medium uppercase tracking-wide text-emerald-700 dark:text-emerald-300 mb-2">
            Your services · {services.length}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {services.map((svc) => {
              const removable = savedThisRunIds.has(svc.id);
              return (
                <span
                  key={svc.id}
                  className="inline-flex items-center gap-1.5 rounded-full bg-white dark:bg-slate-800 border border-emerald-200 dark:border-emerald-800 pl-2 pr-1.5 py-0.5 text-xs text-slate-700 dark:text-slate-200 shadow-sm"
                >
                  <span aria-hidden="true">{svc.icon ?? '🔌'}</span>
                  <span className="font-medium max-w-[10rem] truncate">{svc.name}</span>
                  {removable ? (
                    <button
                      type="button"
                      onClick={() => onRemoveService(svc.id)}
                      aria-label={`Remove ${svc.name}`}
                      className="w-4 h-4 inline-flex items-center justify-center rounded-full text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30"
                    >
                      ×
                    </button>
                  ) : (
                    <span className="text-[9px] uppercase tracking-wide text-emerald-600/80 dark:text-emerald-400/80" title="Already saved">saved</span>
                  )}
                </span>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
            Reuse any of these in your starter flow — no need to re-add. Or add another below.
          </p>
        </div>
      )}

      {/* Preset picker */}
      <div>
        <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-1.5">
          {services.length > 0 ? 'Add another' : 'Pick a service'}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {presets.map((p) => {
            const selected = p.id === presetId;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onPresetChange(selected ? '' : p.id)}
                aria-pressed={selected}
                className={`flex items-center gap-2 text-left p-2.5 rounded-lg border transition-all ${
                  selected
                    ? 'border-[rgb(var(--accent-primary))] shadow-sm'
                    : 'border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500 bg-white dark:bg-slate-900/40'
                }`}
                style={selected ? { backgroundColor: 'rgb(var(--accent-primary) / 0.08)' } : undefined}
                title={p.hint}
              >
                <span className="text-lg leading-none flex-shrink-0">{p.icon}</span>
                <span className="min-w-0">
                  <span className="block text-xs font-medium text-slate-800 dark:text-slate-100 truncate">{p.label}</span>
                  {services.some((s) => s.endpoint === p.endpoint)
                    ? <span className="block text-[10px] text-emerald-600 dark:text-emerald-400">✓ added</span>
                    : p.needsApiKey
                      ? <span className="block text-[10px] text-amber-600 dark:text-amber-400">needs key</span>
                      : <span className="block text-[10px] text-emerald-600 dark:text-emerald-400">local</span>}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Form — only when a preset is selected */}
      {preset && (
        <div className="space-y-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/30 p-3">
          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">{preset.hint}</p>
          <div>
            <label htmlFor="wizard-service-name" className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Name</label>
            <input
              id="wizard-service-name"
              type="text"
              value={serviceName}
              onChange={(e) => onServiceNameChange(e.target.value)}
              placeholder="My OpenAI"
              className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
            />
          </div>
          <div>
            <label htmlFor="wizard-endpoint" className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Endpoint URL</label>
            <div className="flex gap-2">
              <input
                id="wizard-endpoint"
                type="text"
                value={endpoint}
                onChange={(e) => onEndpointChange(e.target.value)}
                placeholder="https://…"
                className="flex-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs font-mono text-slate-800 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
              />
              <button
                type="button"
                onClick={onTest}
                disabled={!endpoint || testState === 'testing'}
                className="px-2.5 py-1.5 text-xs rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Probe the URL to confirm it's reachable"
              >
                {testState === 'testing' ? '…' : 'Test'}
              </button>
            </div>
            {testMsg && (
              <p className={`text-[11px] mt-1 ${testState === 'ok' ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
                {testState === 'ok' ? '✓ ' : '✗ '}{testMsg}
              </p>
            )}
          </div>
          {preset.needsApiKey && (
            <div>
              <label htmlFor="wizard-api-key" className="block text-xs text-slate-600 dark:text-slate-400 mb-1">
                API key{' '}
                {preset.apiKeyDocsUrl && (
                  <a href={preset.apiKeyDocsUrl} target="_blank" rel="noopener noreferrer" className="text-[rgb(var(--accent-primary))] underline">
                    (get one)
                  </a>
                )}
              </label>
              <input
                id="wizard-api-key"
                type="password"
                value={apiKey}
                onChange={(e) => onApiKeyChange(e.target.value)}
                placeholder={keyAlreadyProvided ? 'Already saved — leave blank to reuse it' : 'sk-… (stored locally; never sent to oaiy.com)'}
                autoComplete="off"
                spellCheck={false}
                className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs font-mono text-slate-800 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
              />
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                {keyAlreadyProvided ? (
                  <>✓ <code className="font-mono">{preset.apiKeyConstant}</code> is already set from another service — leave blank to reuse it.</>
                ) : (
                  <>Saved as <code className="font-mono">{preset.apiKeyConstant}</code> in your project constants. Change it later in <strong>Settings → API Keys</strong>.</>
                )}
              </p>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between pt-1">
        <button
          type="button"
          onClick={onBack}
          className="px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100"
        >
          ← Back
        </button>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onAddAnother}
            disabled={!formValid}
            className="px-3 py-2 text-sm rounded-md border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            + Add &amp; add more
          </button>
          <button
            type="button"
            onClick={onContinue}
            disabled={!canContinue}
            className="px-4 py-2 text-sm rounded-md bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Continue →
          </button>
        </div>
      </div>
    </>
  );
}

function StarterFlowStep(props: {
  services: CustomService[];
  selectedId: string;
  onSelect: (id: string) => void;
  onBack: () => void;
  onCreate: () => void;
}) {
  const { services, selectedId, onSelect, onBack, onCreate } = props;
  const selected = services.find((s) => s.id === selectedId) ?? services[0];
  const image = selected ? isImageService(selected) : false;
  const serviceName = selected?.name ?? 'your service';

  return (
    <>
      <div>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">
          Build a starter flow
        </h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
          We'll wire a tiny <em>Text&nbsp;→&nbsp;{serviceName}&nbsp;→&nbsp;{image ? 'Image' : 'Reply'}</em> flow
          so you have something to run.
        </p>
      </div>

      {/* Service picker — only when there's a choice */}
      {services.length > 1 && (
        <div>
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-1.5">
            Which service?
          </div>
          <div className="grid grid-cols-2 gap-2">
            {services.map((s) => {
              const sel = s.id === selected?.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => onSelect(s.id)}
                  aria-pressed={sel}
                  className={`flex items-center gap-2 text-left p-2 rounded-lg border transition-all ${
                    sel
                      ? 'border-[rgb(var(--accent-primary))] shadow-sm'
                      : 'border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500 bg-white dark:bg-slate-900/40'
                  }`}
                  style={sel ? { backgroundColor: 'rgb(var(--accent-primary) / 0.08)' } : undefined}
                >
                  <span className="text-base leading-none flex-shrink-0">{s.icon ?? '🔌'}</span>
                  <span className="text-xs font-medium text-slate-800 dark:text-slate-100 truncate">{s.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Tiny preview of the chain */}
      <div className="bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-2 text-xs">
          <div className="flex-1 text-center">
            <div className="w-10 h-10 mx-auto rounded-md bg-green-100 dark:bg-green-900/30 flex items-center justify-center text-base">📝</div>
            <div className="mt-1 font-medium text-slate-700 dark:text-slate-200">Text Input</div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400">Your prompt</div>
          </div>
          <div className="text-slate-400 text-lg">→</div>
          <div className="flex-1 text-center">
            <div className="w-10 h-10 mx-auto rounded-md flex items-center justify-center text-base" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.10)' }}>🔌</div>
            <div className="mt-1 font-medium text-slate-700 dark:text-slate-200">Service Call</div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400 truncate" title={serviceName}>{serviceName}</div>
          </div>
          <div className="text-slate-400 text-lg">→</div>
          <div className="flex-1 text-center">
            <div className="w-10 h-10 mx-auto rounded-md bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center text-base">{image ? '🖼️' : '✅'}</div>
            <div className="mt-1 font-medium text-slate-700 dark:text-slate-200">Output</div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400">{image ? 'The image' : 'The response'}</div>
          </div>
        </div>
      </div>

      {services.length === 0 && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          No services yet — go back and add one, or skip for now.
        </p>
      )}

      <div className="flex justify-between pt-1">
        <button
          type="button"
          onClick={onBack}
          className="px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100"
        >
          ← Back
        </button>
        <button
          type="button"
          onClick={onCreate}
          disabled={!selected}
          className="px-4 py-2 text-sm rounded-md bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {image ? 'Create my image flow →' : 'Create my chat flow →'}
        </button>
      </div>
    </>
  );
}

function DoneStep({ onClose }: { onClose: () => void }) {
  return (
    <>
      <div className="text-center">
        <div className="mx-auto w-14 h-14 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center text-2xl mb-3">
          🎉
        </div>
        <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
          You're set up!
        </h2>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
          A few things to know while you click around:
        </p>
      </div>

      <div className="space-y-2">
        <TipCard icon="🖱️" title="Drag from the palette">
          Anything on the left can be dragged onto the canvas. Connect handles to wire data through.
        </TipCard>
        <TipCard icon="▶️" title="Run the flow">
          The cyan Run button in the toolbar fires the whole chain. Watch the Execution Log on the right.
        </TipCard>
        <TipCard icon="🔗" title="Share + remote AI">
          Settings → Defaults → <em>Sharing &amp; remote runs</em>. Lets ChatGPT or Claude trigger this flow over HTTP.
        </TipCard>
      </div>

      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 text-sm rounded-md bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white font-medium"
        >
          Open my flow
        </button>
      </div>
    </>
  );
}

function TipCard({ icon, title, children }: { icon: string; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-md bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700">
      <span className="text-xl flex-shrink-0 leading-none">{icon}</span>
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-800 dark:text-slate-100">{title}</div>
        <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5">{children}</p>
      </div>
    </div>
  );
}
