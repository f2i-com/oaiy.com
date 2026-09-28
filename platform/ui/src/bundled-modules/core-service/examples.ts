/**
 * Built-in service definitions surfaced by the Service node + the
 * oaiy-web Services settings panel.
 *
 * Each entry is the minimal viable HTTP shape for an engine someone
 * runs on their own machine. Users can pick one as a preset on a
 * `service_call` node (compiler resolves it to the embedded template),
 * or "Add as custom" to clone + edit it in their own registry.
 *
 * Keep these tiny + faithful to each engine's public API; advanced
 * features stay overridable inline on the node or via a saved custom
 * service.
 */

export type ServiceMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
export type ServiceResponseType = 'json' | 'text';

/**
 * Which oaiy node types a service is meant for. Picked from a node's
 * preset dropdown only when the service is tagged for that node type.
 *
 * Use `service_call` for fully-generic HTTP services (the Service Call
 * node accepts any tagged service AND any untagged service).
 * Tag more nodes (e.g. `['ai_llm', 'service_call']`) for engines that
 * speak multiple node-shaped APIs — e.g. Ollama / LM Studio both
 * surface OpenAI-compatible chat AND a raw HTTP endpoint.
 */
export type ServiceNodeTag =
  | 'ai_llm'
  | 'image_gen'
  | 'video_gen'
  | 'text_to_speech'
  | 'service_call';

/**
 * Declared input pin for a service-as-node. The Service Call node uses
 * these to know which `{{var}}` placeholders the body template
 * substitutes — a service that takes a text prompt + a reference image
 * declares two inputs (`prompt` text + `image`), and the dropped node
 * gets two input handles. Untyped services default to a single `input`
 * pin (which is what existing services rely on via {{input}}).
 *
 * Used by the palette injection path (each registered service surfaces
 * as a draggable entry) and by the runtime's template renderer for
 * multi-input substitution. See `service_call` runtime for the
 * resolution order.
 */
export interface ServiceInputDecl {
  /** Template var + handle id (e.g. 'prompt', 'image', 'audio'). */
  id: string;
  /** Display label on the input handle. */
  name: string;
  /** Pin type — drives the React Flow handle's edge-compatibility check. */
  type: 'string' | 'image' | 'audio' | 'video' | 'any';
  /** Block compile if this pin is unconnected when set. */
  required?: boolean;
  /** Hover-tip text on the handle. */
  description?: string;
}

/** Declared output for a service-as-node. Single output is the common case. */
export interface ServiceOutputDecl {
  id: string;
  name: string;
  type: 'string' | 'image' | 'audio' | 'video' | 'any';
}

export interface CustomService {
  id: string;
  name: string;
  description?: string;
  endpoint: string;
  method: ServiceMethod;
  /** JSON template — `'{}'` allowed. `{{apiKey}}` resolves the API Key Constant. */
  headers: string;
  /**
   * Request body template. Placeholders:
   *  - `{{input}}` — JSON-escaped + quoted (drop into a JSON string slot)
   *  - `{{inputRaw}}` — raw substitution (use when input is already a JSON object)
   *  - `{{apiKey}}` — RAW substitution (unquoted); only valid INSIDE a quoted
   *    string, e.g. "Bearer {{apiKey}}" — never as a bare value slot
   */
  bodyTemplate: string;
  responseType: ServiceResponseType;
  /** Dot/bracket path into the response; `''` returns the whole parsed body. */
  responsePath: string;
  /** Name of a Project Constant that holds the API key. Resolved at runtime. */
  apiKeyConstant?: string;
  /** Marker: built-in examples render as read-only in the Services panel. */
  isBuiltIn?: boolean;
  /** Human-readable install / start command, shown in the panel. */
  installHint?: string;
  /**
   * Optional emoji/icon shown in the synthetic palette tile and the
   * dropped node's title bar — purely visual, lets users tell their
   * services apart at a glance (🤖 for LLM, 🎨 for image, 🎵 for music,
   * 🎬 for video, 🎙️ for TTS, 🪝 for webhook, etc.). Any single grapheme
   * (emoji or short text like "AI") works; we render it raw.
   */
  icon?: string;
  /**
   * Which node types surface this in their preset dropdown. Empty / unset
   * means "show everywhere" (the Service Call node always lists all
   * services regardless of tagging).
   */
  nodeTypes?: ServiceNodeTag[];
  /**
   * Default model name. Picked up by typed nodes (AI LLM, Image Gen) when
   * the user picks this service without filling their own model field.
   * Generic Service Call doesn't use this — it's body-template-driven.
   */
  model?: string;
  /**
   * Hint for typed nodes about the API shape the endpoint speaks. AI LLM
   * uses this to pick between OpenAI / Anthropic / Ollama-native body
   * shapes. Untyped Service Call ignores this.
   */
  apiFormat?: 'openai' | 'anthropic' | 'ollama' | 'lmstudio';
  /**
   * Input pins the service expects. When set, each one becomes a handle
   * on the dropped Service-Call node (or a synthetic service-node, when
   * the palette-injection lands). When omitted, the node falls back to
   * the legacy single `input` pin that maps to `{{input}}` / `{{inputRaw}}`.
   */
  inputs?: ServiceInputDecl[];
  /**
   * Output pin(s). Defaults to a single 'response' pin matching the
   * legacy Service Call shape.
   */
  outputs?: ServiceOutputDecl[];
}

export const BUILT_IN_SERVICES: CustomService[] = [
  {
    id: 'ollama-generate',
    name: 'Ollama — Generate',
    icon: '🤖',
    description: 'Single-turn completion against a local Ollama server.',
    endpoint: 'http://localhost:11434/api/generate',
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{\n  "model": "llama3",\n  "system": {{system}},\n  "prompt": {{input}},\n  "stream": false\n}',
    responseType: 'json',
    responsePath: 'response',
    isBuiltIn: true,
    nodeTypes: ['service_call'],
    model: 'llama3',
    installHint:
      'Install Ollama from https://ollama.com, then run `ollama serve` and `ollama pull llama3`. Default port 11434. For browser use set `OLLAMA_ORIGINS=*`.',
  },
  {
    id: 'ollama-chat',
    name: 'Ollama — Chat',
    icon: '🤖',
    description: 'Chat-tuned conversation against a local Ollama server.',
    endpoint: 'http://localhost:11434/v1/chat/completions',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "model": "llama3",\n  "messages": [{"role": "system", "content": {{system}}}, {"role": "user", "content": {{input}}}]\n}',
    responseType: 'json',
    responsePath: 'choices.0.message.content',
    isBuiltIn: true,
    nodeTypes: ['ai_llm', 'service_call'],
    model: 'llama3',
    apiFormat: 'openai',
    installHint:
      'Ollama from https://ollama.com — `ollama serve` + `ollama pull llama3`. OpenAI-compat endpoint at /v1/chat/completions. Set `OLLAMA_ORIGINS=*` for browser access.',
  },
  {
    id: 'lmstudio-chat',
    name: 'LM Studio — Chat (OpenAI-compatible)',
    icon: '🤖',
    description: "LM Studio's OpenAI-compatible /v1/chat/completions endpoint.",
    endpoint: 'http://localhost:1234/v1/chat/completions',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "model": "local-model",\n  "messages": [{"role": "system", "content": {{system}}}, {"role": "user", "content": {{input}}}]\n}',
    responseType: 'json',
    responsePath: 'choices.0.message.content',
    isBuiltIn: true,
    nodeTypes: ['ai_llm', 'service_call'],
    model: '',
    apiFormat: 'openai',
    installHint:
      'Install LM Studio from https://lmstudio.ai, load a model, click "Start Server" (default port 1234, CORS on by default).',
  },
  {
    id: 'comfyui-prompt',
    name: 'ComfyUI — Submit Prompt',
    icon: '🎨',
    description: 'Queue a workflow JSON to a local ComfyUI server. Pass the workflow JSON as input.',
    endpoint: 'http://localhost:8188/prompt',
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{\n  "prompt": {{inputRaw}}\n}',
    responseType: 'json',
    responsePath: 'prompt_id',
    isBuiltIn: true,
    nodeTypes: ['image_gen', 'video_gen', 'service_call'],
    installHint:
      'Clone https://github.com/comfyanonymous/ComfyUI and run `python main.py --enable-cors-header`. Default port 8188.',
  },
  {
    id: 'whisper-cpp-server',
    name: 'Whisper.cpp Server — Transcribe',
    icon: '🎙️',
    description: 'Send audio (data URL / base64) to a local whisper.cpp server for transcription.',
    endpoint: 'http://localhost:8080/inference',
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{\n  "audio_base64": {{input}}\n}',
    responseType: 'json',
    responsePath: 'text',
    isBuiltIn: true,
    nodeTypes: ['service_call'],
    installHint:
      'Build whisper.cpp from https://github.com/ggerganov/whisper.cpp and run its server example. Default port 8080.',
  },
  {
    id: 'generic-webhook',
    name: 'Generic Webhook (POST JSON)',
    icon: '🪝',
    description: 'POST the input as a JSON body to any URL and return the parsed response.',
    endpoint: '',
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{{inputRaw}}',
    responseType: 'json',
    responsePath: '',
    isBuiltIn: true,
    nodeTypes: ['service_call'],
    installHint:
      'Point this at any local script, n8n / Node-RED webhook, or your own server. Fill in the Endpoint URL on the node.',
  },
  {
    // First built-in service that exercises the multi-input path —
    // dropping this gives a node with TWO input handles (prompt +
    // image) plus a response output. Both {{prompt}} and {{image}} are
    // JSON-escaped: the image arrives as a data/blob URL STRING and must be
    // quoted to stay valid JSON ("image": "data:..."). Do NOT switch {{image}}
    // to {{imageRaw}} — the Raw form emits it unquoted and breaks every call.
    id: 'vision-webhook',
    name: 'Generic Vision Webhook (Image + Prompt → Text)',
    icon: '👁️',
    description:
      'Send a prompt + an image to any vision endpoint and return the text it produces. Two input handles: prompt (text) + image.',
    endpoint: '',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "prompt": {{prompt}},\n  "image": {{image}}\n}',
    responseType: 'json',
    responsePath: 'text',
    isBuiltIn: true,
    nodeTypes: ['service_call'],
    inputs: [
      { id: 'prompt', name: 'Prompt', type: 'string', required: true,
        description: 'Question or instruction about the image — JSON-escaped at substitution.' },
      { id: 'image', name: 'Image', type: 'image', required: true,
        description: 'Image to analyse — passed through as a URL/data string.' },
    ],
    outputs: [{ id: 'response', name: 'Text', type: 'string' }],
    installHint:
      'Works with any vision endpoint that accepts JSON {prompt, image}. Edit the body template + response path on the node to match your service.',
  },
  {
    // Second multi-input demo — image edit (image + prompt → image).
    // Models the InstructPix2Pix / SDXL inpaint / ComfyUI image-edit
    // workflow shape without committing to any single provider's API.
    id: 'image-edit-webhook',
    name: 'Generic Image Edit Webhook (Image + Prompt → Image)',
    icon: '✏️',
    description:
      'Send a prompt + a source image to any image-edit endpoint and return an image URL/data URI. Two input handles: prompt + image.',
    endpoint: '',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "prompt": {{prompt}},\n  "image": {{image}}\n}',
    responseType: 'json',
    responsePath: 'image',
    isBuiltIn: true,
    nodeTypes: ['service_call', 'image_gen'],
    inputs: [
      { id: 'prompt', name: 'Prompt', type: 'string', required: true,
        description: 'Editing instruction — JSON-escaped at substitution.' },
      { id: 'image', name: 'Image', type: 'image', required: true,
        description: 'Source image to edit.' },
    ],
    outputs: [{ id: 'response', name: 'Image', type: 'image' }],
    installHint:
      'Works with InstructPix2Pix servers, SDXL inpaint webhooks, or any custom edit endpoint that accepts JSON {prompt, image}.',
  },
  {
    // ByteDance Lance, served by the OAIY Desktop's lance_server.py
    // (JSON API at /generate, Gradio UI at /ui, default port 17900). The
    // body template carries the gen knobs (edit seconds/seed/steps/size
    // inline on the node); {{prompt}} is the connected text input. The
    // server returns an ABSOLUTE videoUrl, so the video output is a
    // browser-playable URL straight into a Video preview / save node.
    id: 'lance-text-to-video',
    name: 'Lance — Text → Video',
    icon: '🎬',
    description:
      "ByteDance Lance text-to-video via the OAIY Desktop's Lance service (port 17900). Connect a text prompt; outputs a playable video URL. Tweak seconds/seed/steps/size in the body template.",
    endpoint: 'http://127.0.0.1:17900/generate',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "task": "t2v",\n  "prompt": {{prompt}},\n  "seconds": 4,\n  "seed": 42,\n  "steps": 30,\n  "cfg": 4.0,\n  "height": 352,\n  "width": 640,\n  "resolution": "video_480p"\n}',
    responseType: 'json',
    responsePath: 'videoUrl',
    isBuiltIn: true,
    nodeTypes: ['video_gen', 'service_call'],
    inputs: [
      { id: 'prompt', name: 'Prompt', type: 'string', required: true,
        description: 'Text prompt describing the video to generate.' },
    ],
    outputs: [{ id: 'response', name: 'Video', type: 'video' }],
    installHint:
      "Install + start 'Lance (Image+Video)' in the OAIY Desktop (Services tab). JSON API at http://127.0.0.1:17900/generate, Lance UI at http://127.0.0.1:17900/ui. The first generation loads the model (slow); later ones reuse it. Needs a large GPU (~40GB peak for 480p).",
  },
  {
    // Same Lance service, image task -> returns an absolute imageUrl.
    id: 'lance-text-to-image',
    name: 'Lance — Text → Image',
    icon: '🎨',
    description:
      "ByteDance Lance text-to-image via the OAIY Desktop's Lance service (port 17900). Connect a text prompt; outputs an image URL. Tweak seed/steps/size in the body template.",
    endpoint: 'http://127.0.0.1:17900/generate',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "task": "t2i",\n  "prompt": {{prompt}},\n  "seed": 42,\n  "steps": 30,\n  "cfg": 4.0,\n  "height": 768,\n  "width": 768,\n  "resolution": "image_768res"\n}',
    responseType: 'json',
    responsePath: 'imageUrl',
    isBuiltIn: true,
    nodeTypes: ['image_gen', 'service_call'],
    inputs: [
      { id: 'prompt', name: 'Prompt', type: 'string', required: true,
        description: 'Text prompt describing the image to generate.' },
    ],
    outputs: [{ id: 'response', name: 'Image', type: 'image' }],
    installHint:
      "Install + start 'Lance (Image+Video)' in the OAIY Desktop (Services tab). JSON API at http://127.0.0.1:17900/generate, Lance UI at http://127.0.0.1:17900/ui. The first generation loads the model (slow); later ones reuse it.",
  },
  {
    // Lightricks LTX-2.3 distilled (text → video + audio), served by the
    // OAIY Desktop's ltx2_server.py (JSON /generate, default port 17890).
    // The distilled pipeline runs two stages internally (half-res gen → 2×
    // spatial upscale → refine). Body carries the gen knobs; {{prompt}} is
    // the connected text input. Returns an ABSOLUTE videoUrl for direct
    // playback. Dims must be multiples of 64; num_frames = 8*k + 1. Bump
    // height/width toward 1080p for best quality (slower). The model keeps
    // warm after the first call.
    id: 'ltx2-text-to-video',
    name: 'LTX-2.3 — Text → Video',
    icon: '🎬',
    description:
      "Lightricks LTX-2.3 distilled text-to-video (with audio) via the OAIY Desktop's LTX-2.3 service (port 17890). Connect a text prompt; outputs a playable video URL. Tune size/frames/seed in the body template (dims ÷64, frames = 8k+1).",
    endpoint: 'http://127.0.0.1:17890/generate',
    method: 'POST',
    headers: '{}',
    bodyTemplate:
      '{\n  "prompt": {{prompt}},\n  "seed": 42,\n  "height": 704,\n  "width": 1280,\n  "num_frames": 73,\n  "frame_rate": 24\n}',
    responseType: 'json',
    responsePath: 'videoUrl',
    isBuiltIn: true,
    nodeTypes: ['video_gen', 'service_call'],
    inputs: [
      { id: 'prompt', name: 'Prompt', type: 'string', required: true,
        description: 'Text prompt describing the video to generate.' },
    ],
    outputs: [{ id: 'response', name: 'Video', type: 'video' }],
    installHint:
      "Install + start 'LTX-2.3 Video' in the OAIY Desktop (Services tab), and add the folder holding the LTX-2.3 weights under Settings → Model Folders (e.g. E:\\ckpts). JSON API at http://127.0.0.1:17890/generate. The first generation loads the 22B model (slow); later ones reuse it. Needs a ~32 GB GPU; the server auto-picks the freest one.",
  },
];

/**
 * Build the node-type → service-list lookup used by the
 * `service:list:<nodeType>` dynamic-options resolver.
 *
 * Convention: a service with no `nodeTypes` shows ONLY for the generic
 * Service Call node (so users can't accidentally pick a half-formed
 * custom service from a typed node's dropdown). The Service Call node
 * always lists every service regardless of tagging.
 */
export function filterServicesForNodeType(
  services: CustomService[],
  nodeType: ServiceNodeTag | '',
): CustomService[] {
  if (!nodeType || nodeType === 'service_call') return services;
  return services.filter((s) => s.nodeTypes?.includes(nodeType));
}

export const BUILT_IN_SERVICE_IDS: ReadonlySet<string> = new Set(
  BUILT_IN_SERVICES.map((s) => s.id),
);
