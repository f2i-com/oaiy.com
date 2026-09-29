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
import { ArrowRight, Check, FileText, Image as ImageIcon, MessageSquare, MousePointerClick, Play, Plug, Plus, Share2, Sparkles, X } from 'lucide-react';
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import { v4 as uuidv4 } from 'uuid';
import { markWizardCompleted } from '../../lib/wizardPrefs';
import Dialog from '../ui/Dialog';

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
  // Focus the primary CTA on open (the wizard always opens on step 1), NOT
  // "Skip" — otherwise a reflexive Enter on the auto-opened wizard
  // immediately skips and permanently completes onboarding.
  const primaryCtaRef = useRef<HTMLButtonElement>(null);

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

  // Escape and the overlay close it (and mark it done), as Skip does: the
  // Dialog calls `close`.

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
  // Step 3's service, for its title and its button.
  const flowSelected = flowServices.find((x) => x.id === (flowServiceId || flowServices[0]?.id)) ?? flowServices[0];
  const flowIsImage = flowSelected ? isImageService(flowSelected) : false;
  const flowServiceName = flowSelected?.name ?? 'your service';

  const head: Record<Step, { title: string; description: React.ReactNode; icon: React.ReactNode }> = {
    1: {
      title: 'Welcome to OAIY',
      description: 'A visual flow builder that talks to whatever AI tools you run, local or cloud.',
      icon: <Sparkles size={16} />,
    },
    2: {
      title: 'Add your AI services',
      description: <>Connect as many as you like: local engines, or cloud APIs with your own key. Manage them later in <strong>Settings → Services</strong>.</>,
      icon: <Plug size={16} />,
    },
    3: {
      title: 'Build a starter flow',
      description: <>A small <em>Text → {flowServiceName} → {flowIsImage ? 'Image' : 'Reply'}</em> flow, so there is something to run.</>,
      icon: <Play size={16} />,
    },
    4: {
      title: 'You’re set up',
      description: 'A few things to know as you go.',
      icon: <Check size={16} />,
    },
  };

  const dots = (
    <span className="flex items-center gap-1.5" role="img" aria-label={`Step ${step} of ${totalSteps}`}>
      {Array.from({ length: totalSteps }).map((_, i) => (
        <span
          key={i}
          className={`h-1.5 rounded-full transition-all ${
            i + 1 === step ? 'w-6 bg-accent' : i + 1 < step ? 'w-1.5 bg-accent/55' : 'w-1.5 bg-edge-strong'
          }`}
        />
      ))}
    </span>
  );

  const skip = step < 4 && (
    <button type="button" onClick={close} className="btn btn-ghost" aria-label="Skip wizard">
      Skip
    </button>
  );

  const footer =
    step === 1 ? (
      <>
        {dots}
        <span className="spacer" />
        {skip}
        <button ref={primaryCtaRef} type="button" onClick={() => setStep(2)} className="btn btn-primary">
          Let’s go <ArrowRight size={14} />
        </button>
      </>
    ) : step === 2 ? (
      <>
        {dots}
        <span className="spacer" />
        {skip}
        <button type="button" onClick={() => setStep(1)} className="btn btn-secondary">
          Back
        </button>
        <button type="button" onClick={addAnother} disabled={!formValid} className="btn btn-secondary">
          <Plus size={14} /> Add, and add more
        </button>
        <button type="button" onClick={continueToFlow} disabled={!canContinue} className="btn btn-primary">
          Continue <ArrowRight size={14} />
        </button>
      </>
    ) : step === 3 ? (
      <>
        {dots}
        <span className="spacer" />
        {skip}
        <button type="button" onClick={() => setStep(2)} className="btn btn-secondary">
          Back
        </button>
        <button type="button" onClick={createFlowAndAdvance} disabled={!flowSelected} className="btn btn-primary">
          {flowIsImage ? 'Create my image flow' : 'Create my chat flow'} <ArrowRight size={14} />
        </button>
      </>
    ) : (
      <>
        {dots}
        <span className="spacer" />
        <button type="button" onClick={close} className="btn btn-primary">
          Open my flow
        </button>
      </>
    );

  return (
    <Dialog
      open
      onClose={close}
      title={head[step].title}
      description={head[step].description}
      icon={head[step].icon}
      tone="accent"
      size="md"
      hideClose
      initialFocusRef={primaryCtaRef}
      testId="welcome-wizard"
      footer={footer}
    >
      {step === 1 && <WelcomeStep hasAnyService={existingServices.length > 0} />}
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
        />
      )}
      {step === 3 && (
        <StarterFlowStep
          services={flowServices}
          selectedId={flowServiceId || flowServices[0]?.id || ''}
          onSelect={setFlowServiceId}
        />
      )}
      {step === 4 && <DoneStep />}
    </Dialog>
  );
}

// -------------------------------------------------------------------
// Step bodies — kept inline; each is small and only used here. The
// title, the progress and the buttons are the wizard's Dialog's.
// -------------------------------------------------------------------

/** A service's own icon (from its preset), or a plug. */
function ServiceIcon({ icon }: { icon?: string }) {
  return icon ? <span aria-hidden="true" className="shrink-0 text-base leading-none">{icon}</span> : <Plug size={15} className="shrink-0 text-content-faint" />;
}

function WelcomeStep({ hasAnyService }: { hasAnyService: boolean }) {
  const steps = [
    'Connect one or more AI services (OpenAI chat, GPT Image, Anthropic, Ollama, or any HTTP endpoint).',
    'Build a starter flow you can run right away.',
    'Show you where everything lives.',
  ];
  return (
    <>
      <div className="flex items-center gap-3">
        {/* App mark (same as the favicon). Decorative: the title carries the
            meaning, so the image is aria-hidden. */}
        <img
          src="/favicon.svg"
          alt=""
          aria-hidden="true"
          width={44}
          height={44}
          draggable={false}
          className="h-11 w-11 shrink-0 select-none rounded-[var(--r-ctl)]"
        />
        <p className="m-0 text-[13px] text-content-primary">In the next few clicks it will:</p>
      </div>
      <ol className="m-0 flex list-none flex-col gap-2 p-0">
        {steps.map((text, i) => (
          <li key={i} className="flex items-start gap-2.5 text-[13px] text-content-secondary">
            <span aria-hidden="true" className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent/12 font-mono text-[10px] font-bold text-accent">
              {i + 1}
            </span>
            <span>{text}</span>
          </li>
        ))}
      </ol>
      {hasAnyService && (
        <p className="oaiy-help faint">
          You already have services set up: add more here, or skip straight to a starter flow.
        </p>
      )}
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
}) {
  const {
    presets, presetId, onPresetChange, preset, serviceName, onServiceNameChange,
    endpoint, onEndpointChange, apiKey, onApiKeyChange, testState, testMsg, onTest,
    services, savedThisRunIds, providedKeyConstants, onRemoveService,
  } = props;

  const keyAlreadyProvided =
    !!preset?.apiKeyConstant && providedKeyConstants.has(preset.apiKeyConstant);

  const choice = (on: boolean) =>
    `flex min-w-0 items-center gap-2 rounded-[var(--r-ctl)] border p-2.5 text-left transition-colors ${
      on ? 'border-accent bg-accent/10' : 'border-edge-primary bg-surface-tertiary/40 hover:border-edge-strong'
    }`;

  return (
    <>
      {/* Services you already have — existing ones (detected) + any added this
          run. Reuse any of them in the starter flow; no need to re-add. */}
      {services.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label">Your services <span className="font-mono">{services.length}</span></span>
          <div className="flex flex-wrap gap-1.5">
            {services.map((svc) => {
              const removable = savedThisRunIds.has(svc.id);
              return (
                <span
                  key={svc.id}
                  className="inline-flex items-center gap-1.5 rounded-full border border-edge-primary bg-surface-tertiary/60 py-0.5 pl-2 pr-1 text-[12px] text-content-primary"
                >
                  <ServiceIcon icon={svc.icon} />
                  <span className="max-w-[10rem] truncate font-medium">{svc.name}</span>
                  {removable ? (
                    <button
                      type="button"
                      onClick={() => onRemoveService(svc.id)}
                      aria-label={`Remove ${svc.name}`}
                      className="grid h-4 w-4 place-items-center rounded-full text-content-faint hover:bg-signal-danger/10 hover:text-signal-danger"
                    >
                      <X size={11} />
                    </button>
                  ) : (
                    <span className="oaiy-pill ok" title="Already saved">saved</span>
                  )}
                </span>
              );
            })}
          </div>
          <p className="oaiy-help faint">
            Use any of these for your starter flow, no need to add them again. Or add another below.
          </p>
        </div>
      )}

      {/* Preset picker */}
      <div className="flex flex-col gap-1.5">
        <span className="oaiy-label">{services.length > 0 ? 'Add another' : 'Pick a service'}</span>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {presets.map((p) => {
            const selected = p.id === presetId;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onPresetChange(selected ? '' : p.id)}
                aria-pressed={selected}
                className={choice(selected)}
                title={p.hint}
              >
                <span className="shrink-0 text-lg leading-none" aria-hidden="true">{p.icon}</span>
                <span className="min-w-0">
                  <span className="block truncate text-[12.5px] font-semibold text-content-primary">{p.label}</span>
                  {services.some((s) => s.endpoint === p.endpoint)
                    ? <span className="block text-[11px] text-signal-green">added</span>
                    : p.needsApiKey
                      ? <span className="block text-[11px] text-signal-amber">needs a key</span>
                      : <span className="block text-[11px] text-signal-green">local</span>}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Form — only when a preset is selected */}
      {preset && (
        <div className="flex flex-col gap-3 rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/40 p-3">
          <p className="oaiy-help faint">{preset.hint}</p>
          <label className="oaiy-field" htmlFor="wizard-service-name">
            <span>Name</span>
            <input
              id="wizard-service-name"
              type="text"
              value={serviceName}
              onChange={(e) => onServiceNameChange(e.target.value)}
              placeholder="My OpenAI"
              className="oaiy-input"
            />
          </label>
          <div className="oaiy-field">
            <label htmlFor="wizard-endpoint" className="oaiy-label">Endpoint URL</label>
            <div className="flex gap-2">
              <input
                id="wizard-endpoint"
                type="text"
                value={endpoint}
                onChange={(e) => onEndpointChange(e.target.value)}
                placeholder="https://…"
                className="oaiy-input mono flex-1"
              />
              <button
                type="button"
                onClick={onTest}
                disabled={!endpoint || testState === 'testing'}
                className="btn"
                title="Probe the URL to confirm it's reachable"
              >
                {testState === 'testing' ? 'Testing…' : 'Test'}
              </button>
            </div>
            {testMsg && (
              <p className={testState === 'ok' ? 'oaiy-ok-text' : 'oaiy-error-text'}>
                {testMsg}
              </p>
            )}
          </div>
          {preset.needsApiKey && (
            <div className="oaiy-field">
              <label htmlFor="wizard-api-key" className="oaiy-label">
                API key{' '}
                {preset.apiKeyDocsUrl && (
                  <a href={preset.apiKeyDocsUrl} target="_blank" rel="noopener noreferrer" className="normal-case tracking-normal text-accent underline">
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
                className="oaiy-input mono"
              />
              <p className="oaiy-help faint">
                {keyAlreadyProvided ? (
                  <><code className="oaiy-code">{preset.apiKeyConstant}</code> is already set from another service: leave this blank to reuse it.</>
                ) : (
                  <>Saved as <code className="oaiy-code">{preset.apiKeyConstant}</code> in your project. Change it later in <strong>Settings → API keys</strong>.</>
                )}
              </p>
            </div>
          )}
        </div>
      )}
    </>
  );
}

function StarterFlowStep(props: {
  services: CustomService[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const { services, selectedId, onSelect } = props;
  const selected = services.find((s) => s.id === selectedId) ?? services[0];
  const image = selected ? isImageService(selected) : false;
  const serviceName = selected?.name ?? 'your service';

  const chain = [
    { icon: <FileText size={17} />, tone: 'bg-signal-green/15 text-signal-green', name: 'Text input', note: 'Your prompt' },
    { icon: <Plug size={17} />, tone: 'bg-accent/12 text-accent', name: 'Service call', note: serviceName },
    { icon: image ? <ImageIcon size={17} /> : <MessageSquare size={17} />, tone: 'bg-signal-cyan/15 text-signal-cyan', name: 'Output', note: image ? 'The image' : 'The response' },
  ];

  return (
    <>
      {/* Service picker — only when there's a choice */}
      {services.length > 1 && (
        <div className="flex flex-col gap-1.5">
          <span className="oaiy-label">Which service?</span>
          <div className="grid grid-cols-2 gap-2">
            {services.map((s) => {
              const sel = s.id === selected?.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => onSelect(s.id)}
                  aria-pressed={sel}
                  className={`flex min-w-0 items-center gap-2 rounded-[var(--r-ctl)] border p-2 text-left transition-colors ${
                    sel ? 'border-accent bg-accent/10' : 'border-edge-primary bg-surface-tertiary/40 hover:border-edge-strong'
                  }`}
                >
                  <ServiceIcon icon={s.icon} />
                  <span className="truncate text-[12.5px] font-semibold text-content-primary">{s.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Tiny preview of the chain */}
      <div className="flex items-center justify-between gap-2 rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/40 p-4">
        {chain.map((c, i) => (
          <div key={c.name} className="contents">
            {i > 0 && <ArrowRight size={16} className="shrink-0 text-content-faint" />}
            <div className="min-w-0 flex-1 text-center">
              <div className={`mx-auto grid h-10 w-10 place-items-center rounded-[var(--r-ctl)] ${c.tone}`}>{c.icon}</div>
              <div className="mt-1 text-[12px] font-semibold text-content-primary">{c.name}</div>
              <div className="truncate text-[11px] text-content-faint" title={c.note}>{c.note}</div>
            </div>
          </div>
        ))}
      </div>

      {services.length === 0 && (
        <p className="oaiy-warn-text">
          No services yet: go back and add one, or skip for now.
        </p>
      )}
    </>
  );
}

function DoneStep() {
  return (
    <div className="flex flex-col gap-2">
      <TipCard icon={<MousePointerClick size={17} />} title="Add nodes from the palette">
        Nodes, in the toolbar at the foot of the canvas, opens the palette: click a node to add it, or drag it where it goes. Wire the handles to pass data along.
      </TipCard>
      <TipCard icon={<Play size={17} />} title="Run the flow">
        Run, in the same toolbar, runs the whole chain. Its log is under the node's properties on the right.
      </TipCard>
      <TipCard icon={<Share2 size={17} />} title="Share, and run it from another AI">
        Settings → General → <em>Sharing and remote runs</em> lets ChatGPT or Claude trigger this flow over HTTP.
      </TipCard>
    </div>
  );
}

function TipCard({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/40 p-3">
      <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-[var(--r-sm)] bg-accent/12 text-accent">{icon}</span>
      <div className="min-w-0">
        <div className="text-[13px] font-semibold text-content-primary">{title}</div>
        <p className="m-0 mt-0.5 text-[12px] leading-relaxed text-content-secondary">{children}</p>
      </div>
    </div>
  );
}
