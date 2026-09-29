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
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Check, ChevronDown, ChevronRight, Plus, Server, Trash2, X } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import type { ProjectConstant, ProjectSettings } from 'oaiy-core';
import {
  listAllServices,
  saveService,
  deleteService,
  addExampleAsCustom,
  type CustomService,
} from '../../../utils/serviceRegistry';
import EngineEndpointCard from './EngineEndpointCard';
import { useToast } from '../../Toast';
import { useConfirmDialog } from '../../../hooks/useConfirmDialog';
import Dialog from '../../ui/Dialog';
import { Card, EmptyState } from '../../chrome/SectionPage';

interface ServicesTabProps {
  // Same shape the SettingsPanel passes every tab; only `isOpen` is used
  // (to re-fetch when the panel reopens). The other props stay for
  // signature-compat in case future iterations gate by project settings.
  isOpen: boolean;
  settings: ProjectSettings;
  constants: ProjectConstant[];
  onUpdateSettings: (updates: Partial<ProjectSettings>) => void;
}

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
  // `adding` is optionally a template id to seed the form with —
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

  // Services are purely user-created now; the bundled examples no longer
  // surface (the form's templates carry their preconfigured defaults).
  const mine = services.filter((s) => !s.isBuiltIn);

  return (
    <>
      {/* First, because everything below it is reached THROUGH the engine. */}
      <EngineEndpointCard />

      <Card
        title="Your services"
        count={mine.length}
        flush={mine.length > 0}
        actions={
          // One entry point: the form's template dropdown offers the
          // provider-shaped starting points.
          <button
            type="button"
            onClick={() => setAdding('blank')}
            className="btn btn-primary btn-sm"
            disabled={adding !== null || !!editing}
          >
            <Plus size={13} /> Add a service
          </button>
        }
      >
        {mine.length === 0 ? (
          <>
            <p className="oaiy-card-text">
              A service is an HTTP endpoint with a body template, a response path and the pins it
              takes and gives: text to text, image and text to image, audio to text, anything.
              Every node in the palette's Services list calls one.
            </p>
            <EmptyState icon={<Server size={24} />} title="No services yet">
              Add one: its form starts from a template for Ollama, OpenAI, ComfyUI, Whisper and others,
              with the body and response path filled in. They are kept in this browser
              (<code className="oaiy-code">oaiy.customServices</code>).
            </EmptyState>
          </>
        ) : (
          <>
            <p className="oaiy-card-text border-b border-edge-secondary px-4 py-3">
              HTTP endpoints your nodes call, kept in this browser
              (<code className="oaiy-code">oaiy.customServices</code>). Each offers the pins it declares
              on the Service Call node, and on the nodes it is tagged for.
            </p>
            <ul className="oaiy-rows m-0 list-none p-0">
              {mine.map((svc) => (
                <ServiceRow
                  key={svc.id}
                  svc={svc}
                  onEdit={() => setEditing(svc)}
                  onDelete={() => handleDelete(svc)}
                  onAddCopy={() => handleAddCopy(svc)}
                  disabled={!!editing || adding !== null}
                />
              ))}
            </ul>
          </>
        )}
      </Card>

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
    </>
  );
}

const TAG_SHORT: Record<string, string> = {
  ai_llm: 'AI',
  image_gen: 'Image',
  video_gen: 'Video',
  text_to_speech: 'Speech',
};

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
    <li className="oaiy-row top flex-wrap">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[var(--r-sm)] border border-edge-primary bg-surface-tertiary text-[15px] text-content-secondary" aria-hidden="true">
        {svc.icon || <Server size={15} />}
      </span>
      <div className="oaiy-row-main wide">
        <div className="flex min-w-0 flex-wrap items-center gap-1.5">
          <span className="oaiy-row-title">{svc.name}</span>
          {svc.isBuiltIn ? (
            <span className="oaiy-pill info" title="Bundled with OAIY — ready to use as-is. Customize makes an editable copy.">built-in</span>
          ) : null}
          {/* The "Used for" tags: which node types list this service. */}
          {(svc.nodeTypes ?? []).map((tag) => (
            <span key={tag} className="oaiy-pill" title={`Listed in ${tag} node dropdowns`}>
              {TAG_SHORT[tag] ?? 'Generic'}
            </span>
          ))}
        </div>
        {svc.description && <span className="text-[12px] leading-snug text-content-secondary">{svc.description}</span>}
        <span className="oaiy-row-meta font-mono">
          <strong className="font-semibold text-content-secondary">{svc.method}</strong>{' '}
          {svc.endpoint || <em className="not-italic text-signal-amber">no endpoint set</em>}
          {svc.responsePath ? ` · ${svc.responsePath}` : ''}
        </span>
        {svc.installHint && (
          <details className="mt-1">
            <summary className="cursor-pointer text-[12px] text-content-faint hover:text-content-primary">
              Install / start
            </summary>
            <p className="mt-1 whitespace-pre-wrap border-l-2 border-edge-primary pl-3 text-[12px] text-content-secondary">
              {svc.installHint}
            </p>
          </details>
        )}
        {pingStatus === 'ok' && (
          <span className="oaiy-ok-text truncate" title={pingMsg}>Reached it: {pingMsg}</span>
        )}
        {pingStatus === 'err' && (
          <span className="oaiy-error-text truncate" title={pingMsg}>Unreachable: {pingMsg}</span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {/* Test = ping the endpoint to confirm reachability + surface CORS
            errors before the user wires it into a flow. */}
        {svc.endpoint && (
          <button
            onClick={test}
            disabled={disabled || pingStatus === 'pinging'}
            className="btn btn-sm"
            title={pingStatus === 'err' || pingStatus === 'ok' ? pingMsg : 'Ping the endpoint to check it can be reached (and CORS)'}
          >
            {pingStatus === 'pinging' ? 'Testing…' : 'Test'}
          </button>
        )}
        {svc.isBuiltIn ? (
          <button
            onClick={onAddCopy}
            disabled={disabled}
            className="btn btn-sm"
            title="Clone this built-in into your services so you can edit it"
          >
            Customize
          </button>
        ) : (
          <>
            <button onClick={onEdit} disabled={disabled} className="btn btn-sm">
              Edit
            </button>
            <button
              onClick={onDelete}
              disabled={disabled}
              className="oaiy-icon-btn"
              title={`Delete ${svc.name}`}
              aria-label={`Delete ${svc.name}`}
            >
              <Trash2 size={14} />
            </button>
          </>
        )}
      </div>
    </li>
  );
}

/**
 * The service editor, as a dialog: from Settings → Services (add, edit) and
 * from any node's "Add a service…" (lib/addServiceDialog.tsx).
 */
export function ServiceForm({
  initial,
  isNew,
  onSave,
  onCancel,
  title,
  intro,
  footerStart,
  error,
}: {
  initial: CustomService;
  isNew: boolean;
  onSave: (svc: CustomService) => void;
  onCancel: () => void;
  /** The dialog's title (default: "Add a service" or "Edit <name>"). */
  title?: ReactNode;
  /** Shown at the top of the form, above its fields. */
  intro?: ReactNode;
  /** The footer's left side (a link, a note), before Cancel and Save. */
  footerStart?: ReactNode;
  /** A problem with the last save, said under the fields. */
  error?: ReactNode;
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
  const nameRef = useRef<HTMLInputElement>(null);
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

  const pinTypeOptions = (
    <>
      <option value="any">any</option>
      <option value="string">string</option>
      <option value="image">image</option>
      <option value="audio">audio</option>
      <option value="video">video</option>
    </>
  );

  return (
    <Dialog
      open
      onClose={() => void handleCancel()}
      title={title ?? (isNew ? 'Add a service' : `Edit ${initial.name}`)}
      description={isNew ? 'An HTTP endpoint your nodes can call.' : undefined}
      icon={<Server size={16} />}
      tone="accent"
      size="lg"
      initialFocusRef={nameRef}
      testId="service-form"
      footer={
        <>
          {footerStart}
          {footerStart && <span className="spacer" />}
          <button type="button" onClick={() => void handleCancel()} className="btn btn-secondary">
            Cancel
          </button>
          <button type="button" onClick={() => onSave(draft)} className="btn btn-primary">
            {isNew ? 'Create service' : 'Save changes'}
          </button>
        </>
      }
    >
      {intro}

      {/* Step 1 (new services only): pick a template that pre-fills the
          gnarly fields (body template / response path / nodeTypes). */}
      {isNew && (
        <Row
          label="Start from a template"
          hint="Fills in the body, the response path and the headers. Pick the closest; you can change everything after (Advanced opens to show what it filled in)."
        >
          <select
            value={appliedTemplate}
            onChange={(e) => {
              const v = e.target.value;
              if (v) applyTemplate(v);
              else setAppliedTemplate('');
            }}
            className="oaiy-select"
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
              <p className="oaiy-ok-text inline-flex items-center gap-1.5">
                <Check size={13} /> Applied <strong>{t.label}</strong>: see Advanced for the body
                template and response path it filled in.
              </p>
            );
          })()}
        </Row>
      )}

      {/* Always-visible required fields */}
      <div className="oaiy-form-grid">
        <Row label="Name (required)">
          <input
            ref={nameRef}
            type="text"
            value={draft.name}
            onChange={(e) => setName(e.target.value)}
            className="oaiy-input"
            placeholder="My Local Ollama"
          />
        </Row>

        <Row label="Endpoint URL (required)">
          <input
            type="text"
            value={draft.endpoint}
            onChange={(e) => set('endpoint', e.target.value)}
            className="oaiy-input mono"
            placeholder="http://localhost:11434/v1/chat/completions"
          />
        </Row>
      </div>

      <Row
        group
        label="Icon"
        hint="Shown on the palette tile and the dropped node's title bar. Pick one, or type any emoji or other character in the box."
      >
        <div className="flex flex-col gap-2">
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
                  aria-pressed={active}
                  className={`flex h-8 w-8 items-center justify-center rounded-[var(--r-sm)] border text-lg leading-none transition-colors ${
                    active
                      ? 'border-accent bg-accent/15'
                      : 'border-transparent bg-surface-tertiary hover:border-edge-strong'
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
                className="oaiy-icon-btn"
                title="Clear the icon (use the default server glyph)"
                aria-label="Clear the icon"
              >
                <X size={14} />
              </button>
            )}
          </div>
          {/* Free-text fallback — any emoji / UTF-8 glyph the picker omits. */}
          <div className="flex items-center gap-2">
            <span className="text-[12px] text-content-faint">Or your own:</span>
            <input
              type="text"
              value={draft.icon ?? ''}
              onChange={(e) => set('icon', e.target.value)}
              className="oaiy-input"
              style={{ width: 96 }}
              placeholder="🪝"
              maxLength={6}
              aria-label="Icon"
            />
            {draft.icon && (
              <span className="text-[12px] text-content-faint">
                Preview <span className="text-lg">{draft.icon}</span>
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
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="btn btn-ghost btn-sm"
          aria-expanded={showAdvanced}
        >
          {showAdvanced ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {showAdvanced ? 'Hide advanced' : 'Advanced: body, response path, headers, API key, pins, ID'}
        </button>
      </div>

      {showAdvanced && (
        <div className="flex flex-col gap-3 border-t border-edge-secondary pt-3">
          <div className="oaiy-form-grid">
            <Row label="ID" hint="Letters, numbers, dash and underscore. Fixed once the service exists.">
              <input
                type="text"
                value={draft.id}
                onChange={(e) => set('id', e.target.value)}
                className="oaiy-input mono"
                disabled={!isNew}
              />
            </Row>

            <Row label="Description">
              <input
                type="text"
                value={draft.description ?? ''}
                onChange={(e) => set('description', e.target.value)}
                className="oaiy-input"
                placeholder="What this service does"
              />
            </Row>
          </div>

          <div className="grid grid-cols-[1fr_auto] gap-3">
            <Row
              label="Default model"
              hint="Syncs into the body template: as the {{model}} placeholder when it runs, and by rewriting any literal &quot;old-model&quot; string already in the body."
            >
              <input
                type="text"
                value={draft.model ?? ''}
                onChange={(e) => setModel(e.target.value)}
                className="oaiy-input mono"
                placeholder="llama3"
              />
            </Row>
            <Row label="Method">
              <select
                value={draft.method}
                onChange={(e) => set('method', e.target.value as CustomService['method'])}
                className="oaiy-select"
                style={{ width: 110 }}
              >
                {(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] as const).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Row>
          </div>

          <Row
            label="Body template"
            hint="{{input}} is JSON-escaped, {{inputRaw}} raw. {{apiKey}} is the API key constant's value."
          >
            <textarea
              value={draft.bodyTemplate}
              onChange={(e) => set('bodyTemplate', e.target.value)}
              className="oaiy-textarea mono"
              rows={5}
              placeholder='{"prompt": {{input}}, "stream": false}'
            />
          </Row>

          <div className="grid grid-cols-[auto_1fr] gap-3">
            <Row label="Response type">
              <select
                value={draft.responseType}
                onChange={(e) => set('responseType', e.target.value as CustomService['responseType'])}
                className="oaiy-select"
                style={{ width: 110 }}
              >
                <option value="json">JSON</option>
                <option value="text">Text</option>
              </select>
            </Row>
            <Row label="Response path" hint="A dot or bracket path; blank for the whole body.">
              <input
                type="text"
                value={draft.responsePath}
                onChange={(e) => set('responsePath', e.target.value)}
                className="oaiy-input mono"
                placeholder="choices.0.message.content"
              />
            </Row>
          </div>

          <Row label="Headers (JSON template)" hint="Blank sends Content-Type: application/json. Put {{apiKeyRaw}} inside a quoted value (e.g. &quot;Bearer {{apiKeyRaw}}&quot;) for the API key constant.">
            <textarea
              value={draft.headers}
              onChange={(e) => set('headers', e.target.value)}
              className="oaiy-textarea mono"
              style={{ minHeight: 56 }}
              rows={2}
              placeholder='{"Authorization": "Bearer {{apiKeyRaw}}"}'
            />
          </Row>

          <div className="oaiy-form-grid">
            <Row label="API key constant" hint="A constant's name (Settings → API keys); read as {{apiKey}}.">
              <input
                type="text"
                value={draft.apiKeyConstant ?? ''}
                onChange={(e) => set('apiKeyConstant', e.target.value)}
                className="oaiy-input mono"
                placeholder="OPENAI_API_KEY"
              />
            </Row>
            <Row label="API format" hint="The body's shape, for the typed nodes.">
              <select
                value={draft.apiFormat ?? ''}
                onChange={(e) => set('apiFormat', (e.target.value || undefined) as CustomService['apiFormat'])}
                className="oaiy-select"
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
            group
            label="Inputs"
            hint="The pins it takes. Each id is a {{placeholder}} in the body template ({{prompt}}, {{image}} …) and an input on the dropped node."
          >
            <div className="flex flex-col gap-2">
              {(draft.inputs ?? []).map((inp, idx) => (
                <div key={idx} className="grid grid-cols-[1fr_1fr_auto_auto] items-center gap-2">
                  <input
                    type="text"
                    value={inp.id}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], id: e.target.value };
                      set('inputs', next);
                    }}
                    className="oaiy-input oaiy-input-sm mono"
                    placeholder="id (e.g. prompt)"
                    aria-label={`Input ${idx + 1} id`}
                  />
                  <input
                    type="text"
                    value={inp.name}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], name: e.target.value };
                      set('inputs', next);
                    }}
                    className="oaiy-input oaiy-input-sm"
                    placeholder="Display name"
                    aria-label={`Input ${idx + 1} name`}
                  />
                  <select
                    value={inp.type}
                    onChange={(e) => {
                      const next = [...(draft.inputs ?? [])];
                      next[idx] = { ...next[idx], type: e.target.value as typeof inp.type };
                      set('inputs', next);
                    }}
                    className="oaiy-select oaiy-input-sm"
                    style={{ width: 96 }}
                    aria-label={`Input ${idx + 1} type`}
                  >
                    {pinTypeOptions}
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      const next = [...(draft.inputs ?? [])];
                      next.splice(idx, 1);
                      set('inputs', next);
                    }}
                    className="oaiy-icon-btn sm"
                    title="Remove this input pin"
                    aria-label={`Remove input ${idx + 1}`}
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
              <div>
                <button
                  type="button"
                  onClick={() => {
                    const next = [...(draft.inputs ?? [])];
                    const idx = next.length;
                    next.push({ id: `input_${idx + 1}`, name: `Input ${idx + 1}`, type: 'any' });
                    set('inputs', next);
                  }}
                  className="btn btn-sm"
                >
                  <Plus size={12} /> Add an input pin
                </button>
              </div>
              {(draft.inputs ?? []).length === 0 && (
                <p className="oaiy-help faint">
                  None: the dropped node gets one generic <code className="oaiy-code">input</code> pin,
                  read as <code className="oaiy-code">{'{{input}}'}</code> / <code className="oaiy-code">{'{{inputRaw}}'}</code> in the body template.
                </p>
              )}
            </div>
          </Row>

          <Row
            group
            label="Outputs"
            hint="The pins it gives, each typed (coloured and checked when wired). Leave it empty for one generic response pin, like the older single-output Service Call."
          >
            <div className="flex flex-col gap-2">
              {(draft.outputs ?? []).map((out, idx) => (
                <div key={idx} className="grid grid-cols-[1fr_1fr_auto_auto] items-center gap-2">
                  <input
                    type="text"
                    value={out.id}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], id: e.target.value };
                      set('outputs', next);
                    }}
                    className="oaiy-input oaiy-input-sm mono"
                    placeholder="id (e.g. response)"
                    aria-label={`Output ${idx + 1} id`}
                  />
                  <input
                    type="text"
                    value={out.name}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], name: e.target.value };
                      set('outputs', next);
                    }}
                    className="oaiy-input oaiy-input-sm"
                    placeholder="Display name"
                    aria-label={`Output ${idx + 1} name`}
                  />
                  <select
                    value={out.type}
                    onChange={(e) => {
                      const next = [...(draft.outputs ?? [])];
                      next[idx] = { ...next[idx], type: e.target.value as typeof out.type };
                      set('outputs', next);
                    }}
                    className="oaiy-select oaiy-input-sm"
                    style={{ width: 96 }}
                    aria-label={`Output ${idx + 1} type`}
                  >
                    {pinTypeOptions}
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      const next = [...(draft.outputs ?? [])];
                      next.splice(idx, 1);
                      set('outputs', next);
                    }}
                    className="oaiy-icon-btn sm"
                    title="Remove this output pin"
                    aria-label={`Remove output ${idx + 1}`}
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
              <div>
                <button
                  type="button"
                  onClick={() => {
                    const next = [...(draft.outputs ?? [])];
                    const idx = next.length;
                    next.push({ id: `output_${idx + 1}`, name: `Output ${idx + 1}`, type: 'any' });
                    set('outputs', next);
                  }}
                  className="btn btn-sm"
                >
                  <Plus size={12} /> Add an output pin
                </button>
              </div>
              {(draft.outputs ?? []).length === 0 && (
                <p className="oaiy-help faint">
                  None: the dropped node gets one generic <code className="oaiy-code">response</code> pin,
                  carrying whatever the response path picks out.
                </p>
              )}
            </div>
          </Row>

          <Row label="Install or start hint" hint="A reminder shown with this service in the list.">
            <textarea
              value={draft.installHint ?? ''}
              onChange={(e) => set('installHint', e.target.value)}
              className="oaiy-textarea"
              style={{ minHeight: 56 }}
              rows={2}
              placeholder="Run `ollama serve` and `ollama pull llama3`"
            />
          </Row>
        </div>
      )}

      {error && <p className="oaiy-error-text">{error}</p>}
    </Dialog>
  );
}

/** One field of the form: its label, the control(s), and a hint under them. */
function Row({
  label,
  hint,
  group = false,
  children,
}: {
  label: string;
  hint?: string;
  /** More than one control (a picker, a list of pins): a group, not a label. */
  group?: boolean;
  children: React.ReactNode;
}) {
  const body = (
    <>
      <span>{label}</span>
      {children}
      {hint && <p className="oaiy-help faint">{hint}</p>}
    </>
  );
  return group ? (
    <div className="oaiy-field" role="group" aria-label={label}>{body}</div>
  ) : (
    <label className="oaiy-field">{body}</label>
  );
}
