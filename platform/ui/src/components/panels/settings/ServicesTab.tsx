/**
 * Services Tab — manage custom HTTP services for the `service_call` node.
 *
 * Web-specific shape: in the browser there are no local PROCESSES to
 * manage (start/stop/health), only external HTTP endpoints to register.
 * Each entry templates a request body + response path, and is callable
 * from any flow via the Service Call node's preset dropdown.
 *
 * Built-in examples (Ollama / LM Studio / ComfyUI / Whisper / generic
 * webhook) ship as read-only entries with one-click "Add as Custom" so
 * the user gets a working template to edit. Custom services live in
 * localStorage under `oaiy.customServices` — the core-service compiler
 * reads the same key, so saving here makes the preset available to
 * compiled flows immediately.
 */
import { useEffect, useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { ProjectConstant, ProjectSettings } from 'oaiy-core';
import {
  listAllServices,
  saveService,
  deleteService,
  addExampleAsCustom,
  type CustomService,
} from '../../../utils/serviceRegistry';
import { useToast } from '../../Toast';
import { useConfirmDialog } from '../../../hooks/useConfirmDialog';

interface ServicesTabProps {
  // Same shape the SettingsPanel passes every tab; only `isOpen` is used
  // (to re-fetch when the panel reopens). The other props stay for
  // signature-compat in case future iterations gate by project settings.
  isOpen: boolean;
  settings: ProjectSettings;
  constants: ProjectConstant[];
  onUpdateSettings: (updates: Partial<ProjectSettings>) => void;
}

const INPUT =
  'w-full px-2 py-1 text-sm border border-slate-300 dark:border-slate-600 rounded ' +
  'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ' +
  'focus:outline-none focus:ring-2 focus:ring-blue-500/50';

const BLANK: CustomService = {
  id: '',
  name: '',
  description: '',
  endpoint: '',
  method: 'POST',
  headers: '{}',
  bodyTemplate: '{{inputRaw}}',
  responseType: 'json',
  responsePath: '',
  apiKeyConstant: '',
  installHint: '',
};

// ---------------------------------------------------------------------------
// Quick-start templates
// ---------------------------------------------------------------------------
//
// Picking one in the form auto-fills the templated fields (body, response
// path, headers, apiFormat, nodeTypes) so the user only needs to type Name
// + Endpoint URL. Saves them from authoring a JSON body template from
// scratch — the single biggest "I don't know what to type here" pain point.

interface ServiceTemplate {
  id: string;
  label: string;
  description: string;
  fields: Partial<CustomService>;
}

const TEMPLATE_PRESETS: ServiceTemplate[] = [
  {
    id: 'openai-chat',
    label: 'OpenAI-compatible chat',
    description: 'LM Studio, vLLM, llama-server, Ollama /v1, …',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate:
        '{\n  "model": "local-model",\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
      responseType: 'json',
      responsePath: 'choices.0.message.content',
      apiFormat: 'openai',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'ollama-chat',
    label: 'Ollama — /api/chat',
    description: 'Ollama-native chat endpoint',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate:
        '{\n  "model": "llama3",\n  "messages": [{"role": "user", "content": {{input}}}],\n  "stream": false\n}',
      responseType: 'json',
      responsePath: 'message.content',
      apiFormat: 'ollama',
      model: 'llama3',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'ollama-generate',
    label: 'Ollama — /api/generate',
    description: 'Single-turn completion',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate:
        '{\n  "model": "llama3",\n  "prompt": {{input}},\n  "stream": false\n}',
      responseType: 'json',
      responsePath: 'response',
      model: 'llama3',
      nodeTypes: ['service_call'],
    },
  },
  {
    id: 'anthropic',
    label: 'Anthropic Messages',
    description: '/v1/messages — needs an API key. Works in the browser.',
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
      apiKeyConstant: 'ANTHROPIC_API_KEY',
      nodeTypes: ['ai_llm', 'service_call'],
    },
  },
  {
    id: 'comfyui',
    label: 'ComfyUI — /prompt',
    description: 'Queue a ComfyUI workflow JSON',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{\n  "prompt": {{inputRaw}}\n}',
      responseType: 'json',
      responsePath: 'prompt_id',
      nodeTypes: ['image_gen', 'video_gen', 'service_call'],
    },
  },
  {
    id: 'openai-gpt-image',
    label: 'OpenAI GPT Image 2',
    description: 'Text-to-image via OpenAI — needs your OpenAI API key. Drop it on an Image Generation node to render the picture.',
    fields: {
      method: 'POST',
      endpoint: 'https://api.openai.com/v1/images/generations',
      // {{apiKeyRaw}} (raw) inside the quoted header value — see the Anthropic
      // template note. The Image Generation node builds its own request from
      // apiFormat/model/endpoint/apiKey, so these header/body fields only
      // apply when the service is used through a raw Service Call node.
      headers: '{"Authorization": "Bearer {{apiKeyRaw}}"}',
      bodyTemplate:
        '{\n  "model": "gpt-image-2",\n  "prompt": {{input}},\n  "n": 1,\n  "size": "1024x1024"\n}',
      responseType: 'json',
      // GPT Image returns base64 (no URL option); the Image Generation node
      // wraps it into a data: URL for display. A raw Service Call returns the
      // base64 string itself.
      responsePath: 'data.0.b64_json',
      apiFormat: 'openai',
      model: 'gpt-image-2',
      apiKeyConstant: 'OPENAI_API_KEY',
      nodeTypes: ['image_gen', 'service_call'],
      icon: '🎨',
    },
  },
  {
    id: 'whisper',
    label: 'Whisper transcription',
    description: 'POST audio data URL or base64',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{\n  "audio_base64": {{input}}\n}',
      responseType: 'json',
      responsePath: 'text',
      nodeTypes: ['service_call'],
    },
  },
  {
    id: 'webhook',
    label: 'Generic webhook',
    description: 'POST input as JSON; return parsed body',
    fields: {
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{{inputRaw}}',
      responseType: 'json',
      responsePath: '',
      nodeTypes: ['service_call'],
    },
  },
  {
    id: 'blank',
    label: 'Blank',
    description: 'Fill everything in by hand',
    fields: {},
  },
];

export default function ServicesTab({ isOpen }: ServicesTabProps) {
  const [services, setServices] = useState<CustomService[]>(() => listAllServices());
  const [editing, setEditing] = useState<CustomService | null>(null);
  // `adding` is now optionally a template id to seed the form with —
  // null = closed, '' = blank, or 'openai-chat'/'image-gen-generic'/etc.
  const [adding, setAdding] = useState<string | null>(null);
  const { addToast } = useToast();
  const confirm = useConfirmDialog();

  function refresh() {
    setServices(listAllServices());
  }

  useEffect(() => {
    if (isOpen) refresh();
  }, [isOpen]);

  function handleSave(svc: CustomService) {
    // Validation errors surface as toasts rather than native alerts so
    // the user keeps the form context (and dark-mode contrast).
    if (!svc.id.trim() || !svc.name.trim() || !svc.endpoint.trim()) {
      addToast('Please fill in ID, Name, and Endpoint URL.', 'error');
      return;
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(svc.id)) {
      addToast('Service ID must be alphanumeric (letters, numbers, dash, underscore).', 'error');
      return;
    }
    saveService(svc);
    setEditing(null);
    setAdding(null);
    refresh();
    addToast(`Service "${svc.name}" saved.`, 'success');
  }

  async function handleDelete(svc: CustomService) {
    const ok = await confirm({
      title: 'Delete custom service?',
      message: `"${svc.name}" will be removed from this browser. Any flows using its service_call node will need a different service set.`,
      variant: 'danger',
      confirmLabel: 'Delete service',
    });
    if (!ok) return;
    deleteService(svc.id);
    refresh();
    addToast(`Service "${svc.name}" deleted.`, 'info');
  }

  function handleAddCopy(svc: CustomService) {
    const added = addExampleAsCustom(svc.id);
    if (added) {
      setEditing(added);
      refresh();
    }
  }

  const customCount = services.filter((s) => !s.isBuiltIn).length;
  const builtInCount = services.filter((s) => s.isBuiltIn).length;

  return (
    <div className="p-6 space-y-5 overflow-y-auto">
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Services</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Every node in the <strong>Services</strong> palette is a service you
          add here. A service is just an HTTP endpoint with a body template,
          response path, and declared input / output pins. Any shape works —
          text → text, image + text → image, video → text, audio → text, etc.
          Inside the form, an optional template dropdown gives you a starting
          point for common providers (Ollama, OpenAI, ComfyUI, Whisper, …).
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-500 mt-2">
          {customCount} services · stored locally in
          <code className="ml-1">oaiy.customServices</code>
        </p>
      </div>

      {adding === null && !editing && (
        // Single generic "Add Service" button — the form's template
        // dropdown (inside ServiceForm) still offers shape-specific
        // starting points, but the entry point is one button instead of
        // a wall of per-shape pills. Keeps the panel calm + consistent
        // with the "everything's just a service" mental model.
        <button
          type="button"
          onClick={() => setAdding('blank')}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded font-medium"
        >
          + Add Service
        </button>
      )}

      {adding !== null && (() => {
        const template = TEMPLATE_PRESETS.find((t) => t.id === adding);
        const seed: CustomService = {
          ...BLANK,
          ...(template?.fields ?? {}),
          // UUID v4 to match what the WelcomeWizard generates (and what
          // every other hash-style ID in the app uses). Avoids "my-openai"-
          // style slug collisions when the user adds two services with
          // similar names, and keeps the ID stable even if they later
          // rename the service in the form.
          id: uuidv4(),
          name: template && template.id !== 'blank' ? `My ${template.label}` : '',
        };
        return (
          <ServiceForm
            initial={seed}
            isNew
            onSave={handleSave}
            onCancel={() => setAdding(null)}
          />
        );
      })()}

      {editing && (
        <ServiceForm
          initial={editing}
          isNew={false}
          onSave={handleSave}
          onCancel={() => setEditing(null)}
        />
      )}

      {/* Two sections so users see clearly which services they own vs which
          are bundled examples available out-of-the-box. */}
      <SectionHeader
        title="My Services"
        count={customCount}
        emptyHint="Click + Add Service to register your first one. The form has a template dropdown for common providers (Ollama, OpenAI, ComfyUI, …) — pick one and the body / response path are pre-filled."
      />
      <div className="space-y-2">
        {services
          .filter((s) => !s.isBuiltIn)
          .map((svc) => (
            <ServiceRow
              key={svc.id}
              svc={svc}
              onEdit={() => setEditing(svc)}
              onDelete={() => handleDelete(svc)}
              onAddCopy={() => handleAddCopy(svc)}
              disabled={!!editing || adding !== null}
            />
          ))}
      </div>

      {/* Built-in services no longer surface — services are now purely
          user-created. Templates above carry the preconfigured defaults. */}
      {false && (
      <>
      <SectionHeader
        title="Built-in Services"
        count={builtInCount}
        emptyHint=""
        subtitle="Ready to use as-is — pick from any node's Service dropdown. Click Customize to clone + edit."
      />
      <div className="space-y-2">
        {services
          .filter((s) => s.isBuiltIn)
          .map((svc) => (
            <ServiceRow
              key={svc.id}
              svc={svc}
              onEdit={() => setEditing(svc)}
              onDelete={() => handleDelete(svc)}
              onAddCopy={() => handleAddCopy(svc)}
              disabled={!!editing || adding !== null}
            />
          ))}
      </div>
      </>
      )}
    </div>
  );
}

function SectionHeader({
  title,
  count,
  emptyHint,
  subtitle,
}: {
  title: string;
  count: number;
  emptyHint?: string;
  subtitle?: string;
}) {
  return (
    <div className="pt-2">
      <div className="flex items-baseline gap-2">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
          {title}
        </h3>
        <span className="text-xs text-slate-500">({count})</span>
      </div>
      {subtitle && (
        <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">{subtitle}</p>
      )}
      {count === 0 && emptyHint && (
        <p className="text-xs text-slate-500 dark:text-slate-500 italic mt-1">{emptyHint}</p>
      )}
    </div>
  );
}

function ServiceRow({
  svc,
  onEdit,
  onDelete,
  onAddCopy,
  disabled,
}: {
  svc: CustomService;
  onEdit: () => void;
  onDelete: () => void;
  onAddCopy: () => void;
  disabled: boolean;
}) {
  // Reachability ping — answers "can I reach this server, with my
  // method + headers + auth correct?" NOT "does my body template
  // match this endpoint's schema". Those are different questions and
  // the probe trying to answer both at once produces confusing
  // false-negatives — a chat-completions template hitting a
  // /v1/responses endpoint produces a schema 400 that masks the real
  // signal (auth + reachability worked). The runtime sends the user's
  // full template at real-run time; that's where schema mismatches
  // belong.
  //
  // The probe body is `{"input": "ping", "model": "<svc.model>"}` —
  // the smallest payload most JSON APIs accept (LM Studio Responses,
  // Anthropic Messages, OpenAI Embeddings, generic webhooks). Servers
  // that require `messages` (OpenAI chat completions) will still 400
  // on the body, but a 400 with a JSON error means the server received
  // the request and the route + auth are fine — which IS the answer
  // the probe is supposed to give.
  const [pingStatus, setPingStatus] = useState<'idle' | 'pinging' | 'ok' | 'err'>('idle');
  const [pingMsg, setPingMsg] = useState('');

  async function test() {
    if (!svc.endpoint) return;
    setPingStatus('pinging');
    setPingMsg('');
    const method = (svc.method || 'GET').toUpperCase();

    // Headers — render templated values (e.g. `{{apiKey}}`) and parse
    // as JSON. Empty/invalid template falls through to bare
    // Content-Type so the ping still goes out.
    const headerVars: Record<string, unknown> = {
      input: 'ping',
      model: svc.model || '',
      apiKey: '',
    };
    const renderHeaders = (tpl: string): string =>
      tpl.replace(/\{\{(\w+)\}\}/g, (m, name: string) => {
        const isRaw = name.endsWith('Raw');
        const key = isRaw ? name.slice(0, -3) : name;
        if (!(key in headerVars)) return m;
        const v = headerVars[key];
        const s = v === null || v === undefined
          ? ''
          : typeof v === 'string' ? v : JSON.stringify(v);
        return isRaw ? s : JSON.stringify(s);
      });
    let headers: Record<string, string> = {};
    const rawHeaders = (svc.headers || '').trim();
    if (rawHeaders) {
      try {
        const parsed = JSON.parse(renderHeaders(rawHeaders));
        if (parsed && typeof parsed === 'object') headers = parsed as Record<string, string>;
      } catch {
        /* fall through with empty headers */
      }
    }
    if (!Object.keys(headers).some((k) => k.toLowerCase() === 'content-type')) {
      headers['Content-Type'] = 'application/json';
    }

    const init: RequestInit = { method, mode: 'cors', headers };
    if (method !== 'GET' && method !== 'HEAD') {
      init.body = JSON.stringify({
        input: 'ping',
        model: svc.model || undefined,
      });
    }

    try {
      const resp = await fetch(svc.endpoint, init);
      // Even a 4xx/5xx counts as REACHABLE — the server received the
      // request and replied. Surface the status + a short slice of the
      // response body so the user can see what their server thinks of
      // the probe (e.g. "messages is required" tells them their
      // endpoint expects chat-completions shape; "model not found"
      // tells them the model name is wrong; etc.).
      let detail = '';
      if (resp.status >= 400) {
        try {
          const text = await resp.text();
          // Pull out the most useful nugget — JSON error.message if
          // available, else first 80 chars of raw text.
          let extracted = text;
          try {
            const j = JSON.parse(text);
            const msg = j?.error?.message ?? j?.message ?? null;
            if (typeof msg === 'string' && msg.length > 0) extracted = msg;
          } catch { /* not JSON, use raw */ }
          extracted = extracted.replace(/\s+/g, ' ').trim();
          detail = ' · ' + (extracted.length > 80 ? extracted.slice(0, 77) + '…' : extracted);
        } catch {
          /* body unreadable, just show status */
        }
      }
      setPingStatus('ok');
      setPingMsg(`HTTP ${resp.status}${detail}`);
    } catch (e) {
      setPingStatus('err');
      const msg = String((e as Error).message || e);
      setPingMsg(msg.length > 80 ? msg.slice(0, 77) + '…' : msg);
    }
  }

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-md p-3 bg-white dark:bg-slate-800/50">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-slate-900 dark:text-slate-100">{svc.name}</span>
            {svc.isBuiltIn ? (
              <span
                className="text-xs px-1.5 py-0.5 rounded bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300"
                title="Bundled with OAIY — ready to use as-is. Click Customize to clone + edit."
              >
                Built-in
              </span>
            ) : (
              <span className="text-xs px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
                Custom
              </span>
            )}
            {/* Surface the "Used for" tags inline so users can see at a
                glance which node types will list this service. */}
            {(svc.nodeTypes ?? []).map((tag) => (
              <span
                key={tag}
                className="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                title={`Listed in ${tag} node dropdowns`}
              >
                {tag === 'ai_llm' ? 'AI' : tag === 'image_gen' ? 'Image' : tag === 'video_gen' ? 'Video' : tag === 'text_to_speech' ? 'TTS' : 'Generic'}
              </span>
            ))}
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono truncate">
              {svc.id}
            </span>
          </div>
          {svc.description && (
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">{svc.description}</p>
          )}
          <p className="text-xs text-slate-500 mt-1 font-mono truncate">
            <span className="font-semibold">{svc.method}</span>{' '}
            {svc.endpoint || <em className="text-amber-600">(no endpoint set)</em>}
          </p>
          {svc.responsePath && (
            <p className="text-xs text-slate-500 mt-0.5 font-mono truncate">
              path: <span className="text-slate-700 dark:text-slate-300">{svc.responsePath}</span>
            </p>
          )}
          {svc.installHint && (
            <details className="mt-1">
              <summary className="text-xs text-slate-500 dark:text-slate-400 cursor-pointer hover:text-slate-700 dark:hover:text-slate-300">
                Install / start
              </summary>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 whitespace-pre-wrap pl-4 border-l-2 border-slate-200 dark:border-slate-700">
                {svc.installHint}
              </p>
            </details>
          )}
        </div>
        <div className="flex flex-col gap-1 flex-shrink-0 items-end">
          <div className="flex gap-1">
            {/* Test = GET-ping the endpoint to confirm reachability +
                surface CORS errors before the user wires it into a flow. */}
            {svc.endpoint && (
              <button
                onClick={test}
                disabled={disabled || pingStatus === 'pinging'}
                className={`px-2 py-1 text-xs rounded disabled:opacity-50 disabled:cursor-not-allowed ${
                  pingStatus === 'ok'
                    ? 'bg-emerald-100 hover:bg-emerald-200 dark:bg-emerald-900/40 dark:hover:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300'
                    : pingStatus === 'err'
                    ? 'bg-red-50 hover:bg-red-100 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-300'
                    : 'bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200'
                }`}
                title={pingStatus === 'err' || pingStatus === 'ok' ? pingMsg : 'GET-ping the endpoint to check reachability + CORS'}
              >
                {pingStatus === 'idle' && 'Test'}
                {pingStatus === 'pinging' && '…'}
                {pingStatus === 'ok' && `✓ ${pingMsg}`}
                {pingStatus === 'err' && '✗ Unreachable'}
              </button>
            )}
            {svc.isBuiltIn ? (
              <button
                onClick={onAddCopy}
                disabled={disabled}
                className="px-2 py-1 text-xs bg-cyan-100 hover:bg-cyan-200 dark:bg-cyan-900/40 dark:hover:bg-cyan-900/60 text-cyan-700 dark:text-cyan-300 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Clone this built-in into your services so you can edit it"
              >
                Customize
              </button>
            ) : (
              <>
                <button
                  onClick={onEdit}
                  disabled={disabled}
                  className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Edit
                </button>
                <button
                  onClick={onDelete}
                  disabled={disabled}
                  className="px-2 py-1 text-xs bg-red-50 hover:bg-red-100 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-300 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Delete
                </button>
              </>
            )}
          </div>
          {pingStatus === 'err' && (
            <p className="text-[10px] text-red-600 dark:text-red-400 max-w-[200px] truncate" title={pingMsg}>
              {pingMsg}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function ServiceForm({
  initial,
  isNew,
  onSave,
  onCancel,
}: {
  initial: CustomService;
  isNew: boolean;
  onSave: (svc: CustomService) => void;
  onCancel: () => void;
}) {
  // Guarantee at least one input + one output row in the editor — the
  // user explicitly asked for visible defaults so it's never an empty
  // section. When the saved/template service already declares pins, those
  // are kept verbatim; only an empty/undefined list seeds the generic
  // single 'input' + 'response' rows.
  const seedPins = useMemo<CustomService>(() => {
    const next: CustomService = { ...initial };
    if (!next.inputs || next.inputs.length === 0) {
      next.inputs = [{ id: 'input', name: 'Input', type: 'any' }];
    }
    if (!next.outputs || next.outputs.length === 0) {
      next.outputs = [{ id: 'response', name: 'Response', type: 'any' }];
    }
    return next;
  }, [initial]);
  const [draft, setDraft] = useState<CustomService>(seedPins);
  const confirm = useConfirmDialog();
  // Confirm before discarding edits — Cancel/close throws away the draft, so a
  // mis-click would silently lose a multi-field service definition. Confirms only
  // when actually dirty (matches the app's danger-confirm precedent for delete).
  const handleCancel = async () => {
    const dirty = JSON.stringify(draft) !== JSON.stringify(seedPins);
    if (dirty) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        message: 'Your edits to this service have not been saved and will be lost.',
        variant: 'danger',
        confirmLabel: 'Discard',
      });
      if (!ok) return;
    }
    onCancel();
  };
  // Editing existing service = expand advanced by default so the user
  // sees the full surface they're editing. New = collapsed so the form
  // is just Name + Endpoint + Icon at first glance.
  const [showAdvanced, setShowAdvanced] = useState(!isNew);
  // Sticky controlled value for the template dropdown — lets the user
  // SEE which template was just applied (the previous "reset to '' after
  // apply" trick made it look like nothing happened). Reading it back
  // for `value` keeps the select on the picked option.
  const [appliedTemplate, setAppliedTemplate] = useState('');

  function set<K extends keyof CustomService>(key: K, value: CustomService[K]) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  // Templates fill body/responsePath/etc + sensible nodeType tags. The
  // user's name / id / endpoint / description / installHint are preserved
  // — picking a template after typing a name doesn't blow that away.
  //
  // Also auto-expand Advanced — the body template + response path the
  // template just filled in live under Advanced, so otherwise the user
  // sees the dropdown reset to '' and assumes nothing happened.
  function applyTemplate(templateId: string) {
    const t = TEMPLATE_PRESETS.find((x) => x.id === templateId);
    if (!t) return;
    setDraft((d) => ({
      ...d,
      ...t.fields,
      id: d.id,
      name: d.name,
      description: d.description,
      // Keep the user's typed endpoint if they have one; otherwise let a
      // fixed-endpoint template (e.g. OpenAI GPT Image) pre-fill it.
      endpoint: d.endpoint || t.fields.endpoint || '',
      installHint: d.installHint,
    }));
    setShowAdvanced(true);
    setAppliedTemplate(templateId);
  }

  // Service IDs are now UUID v4s (assigned at form-open time, see the
  // `seed` block in the parent component) — they never change as the
  // user types a name, which avoids slug collisions and ID instability.
  // Older flows pointing at `slug-style-ids` keep resolving because the
  // service registry just does a string lookup either way.
  function setName(name: string) {
    setDraft((d) => ({ ...d, name }));
  }

  // Smart-sync Default Model into the body template:
  //  - Always set draft.model so the runtime gets it via `{{model}}` vars.
  //  - Additionally, if the current body has the OLD model name as a JSON
  //    string literal (e.g. `"model": "llama3"` from the Ollama template),
  //    rewrite it to the new model so the user sees the change reflected
  //    in the visible Body Template field, not just in the silent default.
  //  - Skip the textual rewrite when the new model is empty (would corrupt
  //    the body to `"model": ""`) — the runtime side still substitutes
  //    `""` for an empty model, which is JSON-valid in a string slot.
  function setModel(model: string) {
    setDraft((d) => {
      const oldModel = (d.model ?? '').trim();
      let body = d.bodyTemplate;
      if (oldModel && model && body && body.includes(`"${oldModel}"`)) {
        // Word-boundary-ish: only replace when the old name is enclosed
        // in double quotes — that's how JSON strings are delimited.
        // Use split/join instead of regex to avoid re-escaping pain.
        body = body.split(`"${oldModel}"`).join(`"${model}"`);
      }
      return { ...d, model, bodyTemplate: body };
    });
  }

  function toggleTag(tag: NonNullable<CustomService['nodeTypes']>[number]) {
    const cur = new Set(draft.nodeTypes ?? []);
    if (cur.has(tag)) cur.delete(tag); else cur.add(tag);
    set('nodeTypes', Array.from(cur) as CustomService['nodeTypes']);
  }

  const tagLabels: Record<NonNullable<CustomService['nodeTypes']>[number], string> = {
    ai_llm: 'AI LLM',
    image_gen: 'Image Gen',
    video_gen: 'Video Gen',
    text_to_speech: 'TTS',
    service_call: 'Service Call',
  };

  return (
    <div className="border-2 border-blue-500/40 rounded-md p-4 bg-blue-50/30 dark:bg-blue-900/10 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-slate-900 dark:text-slate-100">
          {isNew ? 'New Service' : `Edit: ${initial.name}`}
        </h3>
        <button
          onClick={handleCancel}
          className="text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
        >
          ✕ Cancel
        </button>
      </div>

      {/* Step 1 (new services only): pick a template that pre-fills the
          gnarly fields (body template / response path / nodeTypes). */}
      {isNew && (
        <Row
          label="Start from a template"
          hint="Pre-fills body + response path + headers. Pick the closest match — you can tweak everything afterwards (Advanced auto-expands so you can see what changed)."
        >
          <select
            value={appliedTemplate}
            onChange={(e) => {
              const v = e.target.value;
              if (v) applyTemplate(v);
              else setAppliedTemplate('');
            }}
            className={INPUT}
          >
            <option value="">
              {appliedTemplate ? 'Switch template…' : 'Choose a template…'}
            </option>
            {TEMPLATE_PRESETS.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}{t.description ? ` — ${t.description}` : ''}
              </option>
            ))}
          </select>
          {appliedTemplate && (() => {
            const t = TEMPLATE_PRESETS.find((x) => x.id === appliedTemplate);
            if (!t) return null;
            return (
              <p className="text-xs text-emerald-700 dark:text-emerald-300 mt-1">
                ✓ Applied <strong>{t.label}</strong> — see Advanced for the body
                template + response path it filled in.
              </p>
            );
          })()}
        </Row>
      )}

      {/* Always-visible required fields */}
      <Row label="Name *">
        <input
          type="text"
          value={draft.name}
          onChange={(e) => setName(e.target.value)}
          className={INPUT}
          placeholder="My Local Ollama"
          autoFocus={isNew}
        />
      </Row>

      <Row label="Endpoint URL *">
        <input
          type="text"
          value={draft.endpoint}
          onChange={(e) => set('endpoint', e.target.value)}
          className={INPUT}
          placeholder="http://localhost:11434/v1/chat/completions"
        />
      </Row>

      <Row
        label="Icon"
        hint="Shown in the palette tile + the dropped node's title bar. Click a suggestion or type any emoji / UTF-8 glyph in the box."
      >
        <div className="space-y-2">
          {/* Quick-pick grid — covers the common service shapes. */}
          <div className="flex flex-wrap gap-1">
            {[
              '🤖', '💬', '🧠', '🪄',
              '🎨', '🖼️', '📷', '✏️',
              '🎬', '🎥', '📹', '🎞️',
              '🎵', '🎶', '🎙️', '🔊',
              '👁️', '🗣️', '📝', '🌐',
              '🪝', '🔌', '⚡', '🛠️',
            ].map((emoji) => {
              const active = (draft.icon ?? '') === emoji;
              return (
                <button
                  key={emoji}
                  type="button"
                  onClick={() => set('icon', emoji)}
                  className={`w-8 h-8 rounded text-lg leading-none flex items-center justify-center transition-colors ${
                    active
                      ? 'bg-cyan-100 ring-2 ring-cyan-400 dark:bg-cyan-900/60 dark:ring-cyan-500'
                      : 'bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700'
                  }`}
                  title={`Use ${emoji}`}
                >
                  {emoji}
                </button>
              );
            })}
            {draft.icon && (
              <button
                type="button"
                onClick={() => set('icon', '')}
                className="w-8 h-8 rounded text-xs leading-none flex items-center justify-center bg-slate-100 hover:bg-red-100 dark:bg-slate-800 dark:hover:bg-red-900/40 text-slate-500 hover:text-red-600 dark:hover:text-red-300"
                title="Clear icon (use the default server glyph)"
              >
                ✕
              </button>
            )}
          </div>
          {/* Free-text fallback — any emoji / UTF-8 glyph the picker omits. */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 dark:text-slate-400">or custom:</span>
            <input
              type="text"
              value={draft.icon ?? ''}
              onChange={(e) => set('icon', e.target.value)}
              className={INPUT + ' w-24'}
              placeholder="🪝"
              maxLength={6}
            />
            {draft.icon && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                preview: <span className="text-lg">{draft.icon}</span>
              </span>
            )}
          </div>
        </div>
      </Row>

      {/* Used for section removed — services are now purely defined by
          their inputs + outputs declared further below. Anything that
          takes (e.g.) `video` → produces `string` works for "describe
          this video" use cases, etc. No artificial node-type buckets. */}

      {/* Advanced disclosure — everything templates already filled in */}
      <button
        type="button"
        onClick={() => setShowAdvanced((v) => !v)}
        className="text-xs text-blue-600 dark:text-blue-400 hover:underline self-start"
      >
        {showAdvanced
          ? '▾ Hide advanced'
          : '▸ Show advanced (body template, response path, headers, API key, install hint, ID, …)'}
      </button>

      {showAdvanced && (
        <div className="space-y-3 pt-2 border-t border-slate-200 dark:border-slate-700">
          <Row label="ID" hint="Auto-derived from Name unless you type one. Letters, numbers, dash, underscore.">
            <input
              type="text"
              value={draft.id}
              onChange={(e) => set('id', e.target.value)}
              className={INPUT}
              disabled={!isNew}
            />
          </Row>

          <Row label="Description">
            <input
              type="text"
              value={draft.description ?? ''}
              onChange={(e) => set('description', e.target.value)}
              className={INPUT}
              placeholder="What this service does"
            />
          </Row>

          <div className="grid grid-cols-[1fr_auto] gap-3">
            <Row
              label="Default Model"
              hint="Auto-syncs into the Body Template — both as the `{{model}}` placeholder at runtime AND as a textual rewrite of any literal `&quot;old-model&quot;` string already in the body."
            >
              <input
                type="text"
                value={draft.model ?? ''}
                onChange={(e) => setModel(e.target.value)}
                className={INPUT}
                placeholder="llama3"
              />
            </Row>
            <Row label="Method">
              <select
                value={draft.method}
                onChange={(e) => set('method', e.target.value as CustomService['method'])}
                className={INPUT}
              >
                {(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] as const).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Row>
          </div>

          <Row
            label="Body Template"
            hint="Use {{input}} (JSON-escaped) or {{inputRaw}} for raw substitution. {{apiKey}} resolves the API Key Constant."
          >
            <textarea
              value={draft.bodyTemplate}
              onChange={(e) => set('bodyTemplate', e.target.value)}
              className={INPUT + ' font-mono text-xs'}
              rows={5}
              placeholder='{"prompt": {{input}}, "stream": false}'
            />
          </Row>

          <div className="grid grid-cols-[auto_1fr] gap-3">
            <Row label="Response Type">
              <select
                value={draft.responseType}
                onChange={(e) => set('responseType', e.target.value as CustomService['responseType'])}
                className={INPUT}
              >
                <option value="json">JSON</option>
                <option value="text">Text</option>
              </select>
            </Row>
            <Row label="Response Path" hint="Dot/bracket path; blank = whole body">
              <input
                type="text"
                value={draft.responsePath}
                onChange={(e) => set('responsePath', e.target.value)}
                className={INPUT}
                placeholder="choices.0.message.content"
              />
            </Row>
          </div>

          <Row label="Headers (JSON template)" hint="Blank = Content-Type: application/json. Use {{apiKeyRaw}} inside a quoted value (e.g. &quot;Bearer {{apiKeyRaw}}&quot;) — it resolves the API Key Constant.">
            <textarea
              value={draft.headers}
              onChange={(e) => set('headers', e.target.value)}
              className={INPUT + ' font-mono text-xs'}
              rows={2}
              placeholder='{"Authorization": "Bearer {{apiKeyRaw}}"}'
            />
          </Row>

          <div className="grid grid-cols-2 gap-3">
            <Row label="API Key Constant" hint="Project Constant name; accessible as {{apiKey}}">
              <input
                type="text"
                value={draft.apiKeyConstant ?? ''}
                onChange={(e) => set('apiKeyConstant', e.target.value)}
                className={INPUT}
                placeholder="e.g. OPENAI_API_KEY"
              />
            </Row>
            <Row label="API Format" hint="Body-shape hint for typed nodes">
              <select
                value={draft.apiFormat ?? ''}
                onChange={(e) => set('apiFormat', (e.target.value || undefined) as CustomService['apiFormat'])}
                className={INPUT}
              >
                <option value="">(default for the node)</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="ollama">Ollama</option>
                <option value="lmstudio">LM Studio</option>
              </select>
            </Row>
          </div>

          <Row
            label="Inputs"
            hint="Pins this service accepts. Each id becomes a {{var}} placeholder in the body template (so {{prompt}}, {{image}} etc.) and an input handle on the dropped node."
          >
            <div className="space-y-2">
              {(draft.inputs ?? []).map((inp, idx) => (
                <div key={idx} className="grid grid-cols-[1fr_1fr_auto_auto] gap-2 items-center">
                  <input
                    type="text"
                    value={inp.id}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], id: e.target.value };
                      set('inputs', next);
                    }}
                    className={INPUT + ' font-mono text-xs'}
                    placeholder="id (e.g. prompt)"
                  />
                  <input
                    type="text"
                    value={inp.name}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], name: e.target.value };
                      set('inputs', next);
                    }}
                    className={INPUT}
                    placeholder="Display name"
                  />
                  <select
                    value={inp.type}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], type: e.target.value as typeof inp.type };
                      set('inputs', next);
                    }}
                    className={INPUT + ' w-24'}
                  >
                    <option value="any">any</option>
                    <option value="string">string</option>
                    <option value="image">image</option>
                    <option value="audio">audio</option>
                    <option value="video">video</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      const next = [...(draft.inputs ?? [])];
                      next.splice(idx, 1);
                      set('inputs', next);
                    }}
                    className="px-2 py-1 text-xs bg-red-50 hover:bg-red-100 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-300 rounded"
                    title="Remove this input pin"
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const next = [...(draft.inputs ?? [])];
                  const idx = next.length;
                  next.push({ id: `input_${idx + 1}`, name: `Input ${idx + 1}`, type: 'any' });
                  set('inputs', next);
                }}
                className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded"
              >
                + Add Input Pin
              </button>
              {(draft.inputs ?? []).length === 0 && (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No inputs declared — the dropped node will use a single generic
                  <code className="mx-1 px-1 py-0.5 bg-slate-100 dark:bg-slate-800 rounded">input</code>
                  pin, accessible as <code>{'{{input}}'}</code> / <code>{'{{inputRaw}}'}</code> in the body template.
                </p>
              )}
            </div>
          </Row>

          <Row
            label="Outputs"
            hint="Pins this service produces. Each output handle on the dropped node renders with the declared type — colour-coded + edge-type-validated when wired downstream. Leave empty for a single generic 'response' pin (matches the legacy single-output Service Call shape)."
          >
            <div className="space-y-2">
              {(draft.outputs ?? []).map((out, idx) => (
                <div key={idx} className="grid grid-cols-[1fr_1fr_auto_auto] gap-2 items-center">
                  <input
                    type="text"
                    value={out.id}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], id: e.target.value };
                      set('outputs', next);
                    }}
                    className={INPUT + ' font-mono text-xs'}
                    placeholder="id (e.g. response)"
                  />
                  <input
                    type="text"
                    value={out.name}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], name: e.target.value };
                      set('outputs', next);
                    }}
                    className={INPUT}
                    placeholder="Display name"
                  />
                  <select
                    value={out.type}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], type: e.target.value as typeof out.type };
                      set('outputs', next);
                    }}
                    className={INPUT + ' w-24'}
                  >
                    <option value="any">any</option>
                    <option value="string">string</option>
                    <option value="image">image</option>
                    <option value="audio">audio</option>
                    <option value="video">video</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      const next = [...(draft.outputs ?? [])];
                      next.splice(idx, 1);
                      set('outputs', next);
                    }}
                    className="px-2 py-1 text-xs bg-red-50 hover:bg-red-100 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-300 rounded"
                    title="Remove this output pin"
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const next = [...(draft.outputs ?? [])];
                  const idx = next.length;
                  next.push({ id: `output_${idx + 1}`, name: `Output ${idx + 1}`, type: 'any' });
                  set('outputs', next);
                }}
                className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded"
              >
                + Add Output Pin
              </button>
              {(draft.outputs ?? []).length === 0 && (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No outputs declared — the dropped node will use a single generic
                  <code className="mx-1 px-1 py-0.5 bg-slate-100 dark:bg-slate-800 rounded">response</code>
                  pin carrying whatever the Response Path extracts.
                </p>
              )}
            </div>
          </Row>

          <Row label="Install / Start Hint" hint="Reminder text shown next to this service in the list">
            <textarea
              value={draft.installHint ?? ''}
              onChange={(e) => set('installHint', e.target.value)}
              className={INPUT + ' text-xs'}
              rows={2}
              placeholder="Run `ollama serve` and `ollama pull llama3`"
            />
          </Row>
        </div>
      )}

      <div className="flex gap-2 pt-2">
        <button
          onClick={() => onSave(draft)}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded font-medium"
        >
          {isNew ? 'Create Service' : 'Save Changes'}
        </button>
        <button
          onClick={handleCancel}
          className="px-3 py-1.5 bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 text-sm rounded"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-700 dark:text-slate-300">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{hint}</p>}
    </label>
  );
}


/**
 * Pill button that seeds the Add-Service form with the right template
 * (Removed: QuickAddBtn was the per-node-type pill row. Folded into a
 * single "+ Add Service" entry point — the form's template dropdown
 * still offers concrete provider starters.)
 */
