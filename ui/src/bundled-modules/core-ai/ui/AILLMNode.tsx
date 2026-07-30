import { memo, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, useUpdateNodeInternals, useNodeId } from '@xyflow/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig, type ValidationIssue } from 'oaiy-ui-components';

import type { ProjectConstant, ProjectSettings, AIProvider } from 'oaiy-core';

interface AILLMNodeData {
  model?: string;
  systemPrompt?: string;
  endpoint?: string;
  apiKey?: string;
  apiKeyConstant?: string;
  headers?: string;
  imageFormat?: string;
  requestFormat?: string;
  provider?: AIProvider;
  wan2gpModel?: string;
  enableThinking?: boolean;
  contextLength?: number;
  maxTokens?: number;
  imageInputCount?: number;
  mmprojPath?: string;
  projectConstants?: ProjectConstant[];
  projectSettings?: ProjectSettings;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onModelChange?: (value: string) => void;
  onSystemPromptChange?: (value: string) => void;
  onEndpointChange?: (value: string) => void;
  onApiKeyChange?: (value: string) => void;
  onApiKeyConstantChange?: (value: string) => void;
  onHeadersChange?: (value: string) => void;
  onImageFormatChange?: (value: string) => void;
  onRequestFormatChange?: (value: string) => void;
  onProviderChange?: (value: AIProvider) => void;
  onWan2gpModelChange?: (value: string) => void;
  onEnableThinkingChange?: (value: boolean) => void;
  onContextLengthChange?: (value: number) => void;
  onMaxTokensChange?: (value: number) => void;
  onImageInputCountChange?: (value: number) => void;
  onMmprojPathChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface AILLMNodeProps {
  data: AILLMNodeData;
}

// Store callbacks in refs to avoid stale closures
function useCallbackRefs(data: AILLMNodeData) {
  const refs = {
    onModelChange: useRef(data.onModelChange),
    onSystemPromptChange: useRef(data.onSystemPromptChange),
    onEndpointChange: useRef(data.onEndpointChange),
    onApiKeyChange: useRef(data.onApiKeyChange),
    onApiKeyConstantChange: useRef(data.onApiKeyConstantChange),
    onHeadersChange: useRef(data.onHeadersChange),
    onImageFormatChange: useRef(data.onImageFormatChange),
    onRequestFormatChange: useRef(data.onRequestFormatChange),
    onProviderChange: useRef(data.onProviderChange),
    onWan2gpModelChange: useRef(data.onWan2gpModelChange),
    onEnableThinkingChange: useRef(data.onEnableThinkingChange),
    onContextLengthChange: useRef(data.onContextLengthChange),
    onMaxTokensChange: useRef(data.onMaxTokensChange),
    onImageInputCountChange: useRef(data.onImageInputCountChange),
    onMmprojPathChange: useRef(data.onMmprojPathChange),
    onCollapsedChange: useRef(data.onCollapsedChange),
  };

  useEffect(() => {
    refs.onModelChange.current = data.onModelChange;
    refs.onSystemPromptChange.current = data.onSystemPromptChange;
    refs.onEndpointChange.current = data.onEndpointChange;
    refs.onApiKeyChange.current = data.onApiKeyChange;
    refs.onApiKeyConstantChange.current = data.onApiKeyConstantChange;
    refs.onHeadersChange.current = data.onHeadersChange;
    refs.onImageFormatChange.current = data.onImageFormatChange;
    refs.onRequestFormatChange.current = data.onRequestFormatChange;
    refs.onProviderChange.current = data.onProviderChange;
    refs.onWan2gpModelChange.current = data.onWan2gpModelChange;
    refs.onEnableThinkingChange.current = data.onEnableThinkingChange;
    refs.onContextLengthChange.current = data.onContextLengthChange;
    refs.onMaxTokensChange.current = data.onMaxTokensChange;
    refs.onImageInputCountChange.current = data.onImageInputCountChange;
    refs.onMmprojPathChange.current = data.onMmprojPathChange;
    refs.onCollapsedChange.current = data.onCollapsedChange;
  });

  return refs;
}

// Provider configurations. Indexed by the stored `AIProvider` value so saved
// workflows (which persist that value verbatim) keep round-tripping through
// `data.provider`. The new UI groups these into Internal / External-preset /
// External-custom — see below.
//
// Default `model` is intentionally empty for every cloud provider: model IDs
// rotate too often (gpt-4o → gpt-5.4 → gpt-5.5 within a year; Claude
// reels through sonnet-4 → sonnet-4-6 → …; Gemini ships major versions
// quarterly). Hard-coding any of them just leaves users with a stale string
// the day the provider deprecates it. The `placeholder` field surfaces a
// rough shape hint in the input box without committing to a specific model.
//
// oaiy-local is the exception — its model id is a stable manifest reference
// owned by us (`qwen3-1.7b-q4-k-m`), not a vendor name that rotates.
const AI_PROVIDERS: { value: AIProvider; label: string; endpoint: string; model: string; placeholder?: string; apiKeyConstant: string; requestFormat: string }[] = [
  { value: 'openai', label: 'OpenAI', endpoint: 'https://api.openai.com/v1/chat/completions', model: '', placeholder: 'e.g. gpt-5.4-mini', apiKeyConstant: 'OPENAI_API_KEY', requestFormat: 'openai' },
  { value: 'anthropic', label: 'Anthropic', endpoint: 'https://api.anthropic.com/v1/messages', model: '', placeholder: 'e.g. claude-sonnet-4-6', apiKeyConstant: 'ANTHROPIC_API_KEY', requestFormat: 'anthropic' },
  { value: 'google', label: 'Google AI', endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', model: '', placeholder: 'e.g. gemini-2.5-flash', apiKeyConstant: 'GOOGLE_API_KEY', requestFormat: 'openai' },
  { value: 'openrouter', label: 'OpenRouter', endpoint: 'https://openrouter.ai/api/v1/chat/completions', model: '', placeholder: 'e.g. anthropic/claude-sonnet-4-6', apiKeyConstant: 'OPENROUTER_API_KEY', requestFormat: 'openai' },
  { value: 'groq', label: 'Groq', endpoint: 'https://api.groq.com/openai/v1/chat/completions', model: '', placeholder: 'e.g. llama-3.3-70b-versatile', apiKeyConstant: 'GROQ_API_KEY', requestFormat: 'openai' },
  { value: 'ollama', label: 'Ollama (Local)', endpoint: 'http://localhost:11434/v1/chat/completions', model: '', placeholder: 'e.g. llama3.2 (pulled name)', apiKeyConstant: '', requestFormat: 'openai' },
  { value: 'lmstudio', label: 'LM Studio (Local)', endpoint: 'http://localhost:1234/v1/chat/completions', model: '', placeholder: 'e.g. local-model or the LM Studio-exposed id', apiKeyConstant: '', requestFormat: 'openai' },
  { value: 'huggingface', label: 'HuggingFace LLM (Local)', endpoint: 'http://127.0.0.1:8774/v1/chat/completions', model: '', placeholder: 'e.g. Qwen/Qwen3.5-9B', apiKeyConstant: '', requestFormat: 'openai' },
  { value: 'oaiy-local', label: 'OAIY Local', endpoint: 'oaiy-llm://local', model: 'qwen3-1.7b-q4-k-m', apiKeyConstant: '', requestFormat: 'openai' },
  { value: 'custom', label: 'Custom', endpoint: '', model: '', placeholder: 'provider-specific model id', apiKeyConstant: '', requestFormat: 'openai' },
];

// =====================================================================
// Internal / External UX (2026-05-17 redesign)
// ---------------------------------------------------------------------
// The 10-option provider dropdown collapses to two top-level modes:
//   • Internal — plugin-llm's GGUF backend (`provider = 'oaiy-local'`)
//   • External — any HTTP-style provider, with four presets + Custom
//
// The four presets (LM Studio, Ollama, OpenAI, Anthropic) auto-fill
// endpoint + model + apiKeyConstant + requestFormat by reusing the
// matching `AI_PROVIDERS` row. Selecting Custom leaves those raw fields
// editable so users can wire arbitrary OpenAI-compatible servers
// (OpenRouter, Groq, HuggingFace TGI, etc.) by hand. Legacy saved
// workflows whose `provider` is one of the dropped well-known names
// (google, openrouter, groq, huggingface) render as `External / Custom`
// with their endpoint/model preserved — no migration step needed.
// =====================================================================
type LlmMode = 'internal' | 'external';
// Local-AI-first (flow-builder pivot, 2026-05-20): only local engines are
// offered as presets. Cloud presets are removed but trivially re-addable —
// uncomment the lines below (and the matching PRESET_PROVIDERS entries) to
// bring them back. Their endpoint/model/auth definitions still live in the
// AI_PROVIDERS table above, so a legacy flow saved with provider='openai'
// still resolves; it just shows as "Custom (raw override)" in the picker.
const EXTERNAL_PRESETS = [
  { value: 'lmstudio',  label: 'LM Studio (localhost:1234)' },
  { value: 'ollama',    label: 'Ollama (localhost:11434)' },
  // { value: 'openai',    label: 'OpenAI' },
  // { value: 'anthropic', label: 'Anthropic' },
  { value: 'custom',    label: 'Custom (raw override)' },
] as const;
type ExternalPreset = (typeof EXTERNAL_PRESETS)[number]['value'];
const PRESET_PROVIDERS: ReadonlySet<AIProvider> = new Set(['lmstudio', 'ollama' /*, 'openai', 'anthropic' */]);

function modeFromProvider(_p: AIProvider | undefined): LlmMode {
  // OAIY is external-only now (bundled plugin-llm removed in the flow-builder
  // pivot, 2026-05-20). The node always presents the external HTTP config;
  // there's no Internal/External toggle. Legacy flows saved with
  // provider='oaiy-local' render under External/Custom with their stored
  // endpoint preserved — the user can re-point them at an external engine.
  return 'external';
}
function presetFromProvider(p: AIProvider | undefined): ExternalPreset {
  if (p && PRESET_PROVIDERS.has(p)) return p as ExternalPreset;
  return 'custom';
}

const IMAGE_FORMATS = [
  { value: 'none', label: 'None', description: 'No image input' },
  { value: 'base64_inline', label: 'Base64 Inline', description: 'Image as base64 data URL in message content' },
  { value: 'base64_separate', label: 'Base64 Separate', description: 'Base64 in separate image_url field' },
  { value: 'url', label: 'URL', description: 'Image URL reference in content' },
  { value: 'binary', label: 'Binary', description: 'Raw binary data (for multipart requests)' },
];

// ============================================================================
// OAIY Local (GGUF) model picker — inline UI for the AI LLM node when the
// user has selected the "OAIY Local (GGUF, Rust)" provider. Lists every
// manifest entry from `plugin:oaiy-llm|llm_list_model_sources` with a
// present/missing pip, and exposes a Download button that streams progress
// via `llm://download-progress` Tauri events. Falls back gracefully when
// not running in Tauri (e.g. preview builds) — renders a plain text input.
// ============================================================================

interface LlmFileStatus {
  path: string;
  url: string;
  size?: number;
  present: boolean;
  absPath: string;
}

interface LlmSourceStatus {
  id: string;
  name: string;
  description: string;
  family: string;
  contextLength?: number;
  files: LlmFileStatus[];
  dependencies: string[];
  allPresent: boolean;
  totalBytes: number;
  bytesPresent: number;
  kind: 'llm';
}

interface LlmProgressPayload {
  modelId: string;
  filePath: string;
  phase: 'start' | 'progress' | 'done' | 'error';
  downloaded: number;
  total?: number;
  fileIndex: number;
  fileCount: number;
  error?: string;
}

function fmtBytes(b: number | undefined): string {
  if (!b || b <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = b;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

interface OaiyLocalModelPickerProps {
  value: string;
  onChange: (modelId: string) => void;
}

/**
 * Tauri 2's internal API surface. `window.__TAURI_INTERNALS__` is exposed
 * regardless of `withGlobalTauri` (round-1 hardening sets that to false,
 * removing `window.__TAURI__`); it's what `@tauri-apps/api/core`'s `invoke`
 * wraps internally. Reaching for it directly lets this picker work inside a
 * plugin bundle, where `@tauri-apps/api/core` is blocked at compile time by
 * `pluginCompiler.ts` (round-4 hardening) — that import resolves to a Proxy
 * that throws "Plugins cannot import \"@tauri-apps/api/core\" directly. Use
 * the runtime context". The picker is a built-in trusted UI component, not
 * sandboxed workflow code, so the internals escape hatch is appropriate.
 */
type TauriInternals = {
  invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
  transformCallback?: (
    cb: (response: unknown) => void,
    once?: boolean,
  ) => number;
};

function getTauriInternals(): TauriInternals | null {
  if (typeof window === 'undefined') return null;
  const internals = (window as unknown as { __TAURI_INTERNALS__?: TauriInternals })
    .__TAURI_INTERNALS__;
  return internals && typeof internals.invoke === 'function' ? internals : null;
}

async function listenLlmDownloadProgress(
  cb: (payload: LlmProgressPayload) => void,
): Promise<() => void> {
  const internals = getTauriInternals();
  if (!internals?.transformCallback) return () => {};
  const handlerId = internals.transformCallback((event) => {
    const payload = (event as { payload?: LlmProgressPayload } | null)?.payload;
    if (payload) cb(payload);
  }, false);
  const eventName = 'llm://download-progress';
  try {
    const eventId = await internals.invoke<number>('plugin:event|listen', {
      event: eventName,
      target: { kind: 'Any' },
      handler: handlerId,
    });
    return async () => {
      // Mirror what @tauri-apps/api/event::_unlisten does: free the JS-side
      // dispatch entry as well, otherwise transformCallback's slot leaks on
      // every mount/unmount cycle of the picker.
      try {
        const eventInternals = (window as unknown as {
          __TAURI_EVENT_PLUGIN_INTERNALS__?: {
            unregisterListener?: (event: string, eventId: number) => void;
          };
        }).__TAURI_EVENT_PLUGIN_INTERNALS__;
        eventInternals?.unregisterListener?.(eventName, eventId);
      } catch {
        /* best-effort */
      }
      try {
        await internals.invoke('plugin:event|unlisten', { event: eventName, eventId });
      } catch {
        /* best-effort */
      }
    };
  } catch {
    return () => {};
  }
}

// Sentinel modelId for the "pick a GGUF off disk" option. The native
// `resolve_gguf_path` accepts absolute paths verbatim, so once the user
// picks a file we store its absolute path as the model id — no
// manifest entry needed. Selecting this sentinel from the dropdown
// fires the OS file dialog; the dialog returning a path replaces the
// stored value with that path.
const CUSTOM_GGUF_SENTINEL = '__custom_gguf__';
// Second sentinel that prompts the user for a download URL (typically a
// HuggingFace resolve link). Selecting it kicks off
// `plugin:oaiy-llm|llm_download_url`; on success we store the resolved
// filename which `resolve_gguf_path`'s dir-relative fallback picks up.
const DOWNLOAD_URL_SENTINEL = '__download_url__';

function isAbsoluteCustomPath(v: string): boolean {
  if (!v) return false;
  if (v === CUSTOM_GGUF_SENTINEL) return false;
  // Treat any value that looks absolute (Windows drive letter, leading
  // slash, or UNC) as a user-picked path. Manifest entries are short
  // ids like `qwen3-1.7b-q4-k-m` that never start with a slash/drive.
  return /^[a-zA-Z]:[\\/]/.test(v) || v.startsWith('/') || v.startsWith('\\\\');
}

function basename(p: string): string {
  const parts = p.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || p;
}

interface LocalGgufFile {
  filename: string;
  absPath: string;
  sizeBytes: number;
}

function OaiyLocalModelPicker({ value, onChange }: OaiyLocalModelPickerProps) {
  const [sources, setSources] = useState<LlmSourceStatus[]>([]);
  const [looseFiles, setLooseFiles] = useState<LocalGgufFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [progress, setProgress] = useState<LlmProgressPayload | null>(null);
  const [pickingFile, setPickingFile] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const internals = getTauriInternals();
      if (!internals) {
        throw new Error('Tauri runtime not available — open the desktop app to manage local models');
      }
      // Two lists, fetched in parallel: the curated manifest (rich
      // cards with download buttons) and loose .gguf files found in
      // the LLM models dir (user-dropped or URL-downloaded).
      const [list, loose] = await Promise.all([
        internals.invoke<LlmSourceStatus[]>('plugin:oaiy-llm|llm_list_model_sources'),
        internals
          .invoke<LocalGgufFile[]>('plugin:oaiy-llm|llm_list_local_files')
          .catch(() => [] as LocalGgufFile[]),
      ]);
      setSources(list);
      setLooseFiles(loose);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh();
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    (async () => {
      const u = await listenLlmDownloadProgress((p) => {
        setProgress(p);
        if (p.phase === 'done' || p.phase === 'error') {
          refresh();
        }
      });
      if (cancelled) u();
      else unlisten = u;
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [refresh]);

  const handleDownload = useCallback(
    async (modelId: string) => {
      setDownloadingId(modelId);
      setProgress(null);
      try {
        const internals = getTauriInternals();
        if (!internals) {
          throw new Error('Tauri runtime not available');
        }
        await internals.invoke('plugin:oaiy-llm|llm_ensure_model', { args: { modelId } });
        await refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setDownloadingId(null);
        setProgress(null);
      }
    },
    [refresh],
  );

  const handlePasteUrl = useCallback(async () => {
    const internals = getTauriInternals();
    if (!internals) {
      setError('URL download needs the desktop runtime.');
      return;
    }
    // window.prompt is intentionally simple — power users can paste a
    // direct HuggingFace `resolve/main/*.gguf` link. A bespoke modal
    // would be nicer but adds React surface area for a power-user knob.
    const url = window.prompt(
      'Paste a direct download URL\n\n(example: https://huggingface.co/<org>/<repo>/resolve/main/<file>.gguf)',
      '',
    );
    if (!url) return;
    const trimmed = url.trim();
    if (!(trimmed.startsWith('http://') || trimmed.startsWith('https://'))) {
      setError('URL must start with http:// or https://');
      return;
    }
    setDownloadingId(`url:${trimmed.slice(0, 80)}`);
    setProgress(null);
    try {
      const result = await internals.invoke<{ filename: string; absPath: string; bytes: number }>(
        'plugin:oaiy-llm|llm_download_url',
        { args: { url: trimmed } },
      );
      // Store the filename — `resolve_gguf_path`'s dir-relative
      // fallback finds it without needing a manifest entry.
      onChange(result.filename);
      await refresh();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDownloadingId(null);
      setProgress(null);
    }
  }, [onChange, refresh]);

  const handlePickGguf = useCallback(async () => {
    const internals = getTauriInternals();
    if (!internals) {
      setError('File picker needs the desktop runtime.');
      return;
    }
    setPickingFile(true);
    try {
      const picked = await internals.invoke<string | null>(
        'plugin:oaiy-filesystem|pick_file',
        {
          filters: [
            { name: 'GGUF / Safetensors / ONNX', extensions: ['gguf', 'safetensors', 'onnx'] },
            { name: 'All Files', extensions: ['*'] },
          ],
        },
      );
      if (typeof picked === 'string' && picked.length > 0) {
        onChange(picked);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPickingFile(false);
    }
  }, [onChange]);

  // A user-picked path lives outside the manifest, so it doesn't show
  // up in `sources`. Track it as a virtual selection so the dropdown
  // can highlight + display it.
  const customPath = isAbsoluteCustomPath(value) ? value : null;
  const selected = useMemo(
    () => sources.find((s) => s.id === value) || null,
    [sources, value],
  );
  // A "loose file" selection is a manifest-unaware filename in the LLM
  // models dir — surfaced by llm_list_local_files. Detected by absence
  // from `sources` + presence in `looseFiles`. Used to show a
  // distinct status line below the dropdown so the user can tell which
  // section they picked from.
  const looseSelection = useMemo(
    () =>
      value && !selected && !customPath
        ? looseFiles.find((f) => f.filename === value) || null
        : null,
    [value, selected, customPath, looseFiles],
  );

  // Compute the value the <select> should reflect. When the user has
  // picked a custom file, the select shows the placeholder row; the
  // path itself is surfaced on the button below and as a chip further
  // down so the user can see which file is active.
  const selectValue = customPath ? '' : value;

  const handleSelectChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const v = e.target.value;
      // The sentinel rows below the dropdown are visual hints only —
      // selecting one revert the <select> to its previous value (we
      // never call onChange with the sentinel). The actual pick / URL
      // flow fires from the dedicated buttons below. This split is
      // forced by browser security: file pickers (showOpenFilePicker /
      // native dialogs) require a *direct* user-gesture event, and
      // <select> change events don't qualify, so a sentinel-driven
      // picker fails silently — no dialog appears.
      if (v === CUSTOM_GGUF_SENTINEL || v === DOWNLOAD_URL_SENTINEL) {
        return;
      }
      onChange(v);
    },
    [onChange],
  );

  // Percent for the currently-downloading file (selected model). Total bytes
  // are file-local — the Rust side emits per-file progress, not per-model
  // rollups. That's accurate-enough for a single-file GGUF (the typical
  // case for LLMs) and obviously loose for multi-file deps; we surface the
  // numerator/denominator literally so the user can see what's happening.
  const dlPercent =
    progress && progress.total && progress.total > 0
      ? Math.min(100, Math.floor((progress.downloaded * 100) / progress.total))
      : 0;

  return (
    <div className="space-y-1.5">
      <label className="text-slate-600 dark:text-slate-400 text-xs block">
        Local Model (GGUF)
      </label>
      {loading ? (
        <div className="text-xs text-slate-600 dark:text-slate-400 py-1">Loading model list…</div>
      ) : sources.length === 0 ? (
        <div className="text-xs text-slate-600 dark:text-slate-400 py-1">
          {error
            ? `Couldn't reach plugin-llm: ${error}`
            : 'No models declared in plugin-llm manifest.'}
        </div>
      ) : (
        <select
          className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
          value={selectValue}
          onChange={handleSelectChange}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <option value="">Select a model…</option>
          {sources.length > 0 && (
            <optgroup label="Catalogue">
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.allPresent ? '✓' : '⬇'} {s.name} — {fmtBytes(s.totalBytes)}
                </option>
              ))}
            </optgroup>
          )}
          {looseFiles.length > 0 && (
            <optgroup label="Loose files in models dir">
              {looseFiles.map((f) => (
                <option key={`loose:${f.filename}`} value={f.filename}>
                  📄 {f.filename} — {fmtBytes(f.sizeBytes)}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      )}
      {/* Real buttons for the pick / paste-URL flows. Native file
          dialogs need a direct user-gesture event (click); routing
          them through a <select> option's onChange swallows the
          gesture and the dialog never appears. */}
      {!loading && (
        <div className="flex gap-1">
          <button
            type="button"
            onClick={handlePickGguf}
            disabled={pickingFile}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag flex-1 px-2 py-1 text-xs rounded border border-dashed border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-60 text-slate-600 dark:text-slate-400"
            title="Pick a .gguf / .safetensors / .onnx file from disk. Absolute paths are accepted verbatim."
          >
            {pickingFile
              ? 'Picking…'
              : customPath
                ? `📂 Custom — ${basename(customPath)}`
                : '📂 Pick custom GGUF / safetensors…'}
          </button>
          <button
            type="button"
            onClick={handlePasteUrl}
            disabled={downloadingId !== null}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag px-2 py-1 text-xs rounded border border-dashed border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-60 text-slate-600 dark:text-slate-400"
            title="Stream an arbitrary HTTPS URL (HuggingFace, R2…) into the LLM models dir."
          >
            🌐 URL…
          </button>
        </div>
      )}
      {customPath && (
        <div className="flex items-center gap-2 text-[10px] text-slate-600 dark:text-slate-400">
          <span className="flex-1 truncate font-mono" title={customPath}>
            {customPath}
          </span>
          <button
            type="button"
            onClick={handlePickGguf}
            disabled={pickingFile}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag px-1.5 py-0.5 text-[10px] rounded border border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-60"
          >
            Change…
          </button>
        </div>
      )}
      {selected && (
        <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
          {selected.description}
        </div>
      )}
      {selected && !selected.allPresent && (
        <button
          type="button"
          onClick={() => handleDownload(selected.id)}
          disabled={downloadingId === selected.id}
          onMouseDown={(e) => e.stopPropagation()}
          className="nodrag w-full px-2 py-1 text-xs rounded bg-purple-600 hover:bg-purple-700 disabled:opacity-60 text-white"
        >
          {downloadingId === selected.id
            ? progress
              ? `Downloading ${progress.fileIndex + 1}/${progress.fileCount} — ${fmtBytes(progress.downloaded)}${progress.total ? ` / ${fmtBytes(progress.total)} (${dlPercent}%)` : ''}`
              : 'Starting download…'
            : `Download (${fmtBytes(selected.totalBytes)})`}
        </button>
      )}
      {selected && selected.allPresent && (
        <div className="text-[10px] text-green-700 dark:text-green-400">
          ✓ Ready on disk
        </div>
      )}
      {customPath && !selected && (
        <div className="text-[10px] text-green-700 dark:text-green-400">
          ✓ Using user-picked file (loaded by absolute path)
        </div>
      )}
      {looseSelection && (
        <div className="text-[10px] text-green-700 dark:text-green-400">
          ✓ Loose file from models dir ({fmtBytes(looseSelection.sizeBytes)})
        </div>
      )}
    </div>
  );
}

/**
 * Companion mmproj `.gguf` picker for Internal (OAIY Local) mode. The
 * runtime resolves the path verbatim when absolute, or relative to the
 * LLM models dir; passing a path here switches `llm_chat` to the
 * vision-conditioned splice path whenever an image input is wired.
 *
 * Only Gemma 3 and Qwen 3.5 / 3.6 multimodal projectors are end-to-end
 * wired through `vision_chat::run_single_image_chat` today; pasting a
 * mismatching projector surfaces a clear `UnsupportedArch` error at
 * generate time.
 *
 * Kept as a small standalone component because the surrounding
 * OaiyLocalModelPicker is already quite dense; bolting another optgroup
 * into it would conflate "what model to load" with "what projector
 * goes alongside".
 */
interface OaiyMmprojPickerProps {
  value: string;
  onChange: (path: string) => void;
}

function OaiyMmprojPicker({ value, onChange }: OaiyMmprojPickerProps) {
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePick = useCallback(async () => {
    const internals = getTauriInternals();
    if (!internals) {
      setError('File picker needs the desktop runtime.');
      return;
    }
    setPicking(true);
    try {
      const picked = await internals.invoke<string | null>(
        'plugin:oaiy-filesystem|pick_file',
        {
          filters: [
            { name: 'GGUF (mmproj)', extensions: ['gguf'] },
            { name: 'All Files', extensions: ['*'] },
          ],
        },
      );
      if (typeof picked === 'string' && picked.length > 0) {
        onChange(picked);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPicking(false);
    }
  }, [onChange]);

  const handleClear = useCallback(() => {
    onChange('');
    setError(null);
  }, [onChange]);

  return (
    <div className="space-y-1">
      <label className="text-slate-600 dark:text-slate-400 text-xs block">
        Vision Projector (mmproj) <span className="text-slate-500">— optional</span>
      </label>
      {value ? (
        <div className="flex items-center gap-2 text-[10px] text-slate-600 dark:text-slate-400">
          <span className="flex-1 truncate font-mono" title={value}>
            {basename(value)}
          </span>
          <button
            type="button"
            onClick={handlePick}
            disabled={picking}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag px-1.5 py-0.5 text-[10px] rounded border border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-60"
          >
            Change…
          </button>
          <button
            type="button"
            onClick={handleClear}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag px-1.5 py-0.5 text-[10px] rounded border border-slate-300 dark:border-slate-600 hover:bg-red-100 dark:hover:bg-red-900/30 text-red-700 dark:text-red-400"
            title="Clear (disables vision input)"
          >
            ×
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={handlePick}
          disabled={picking}
          onMouseDown={(e) => e.stopPropagation()}
          className="nodrag w-full px-2 py-1 text-xs rounded border border-dashed border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-60 text-slate-600 dark:text-slate-400"
        >
          {picking ? 'Picking…' : '🖼 Pick mmproj-*.gguf for vision input…'}
        </button>
      )}
      {error && (
        <div className="text-[10px] text-red-600 dark:text-red-400">{error}</div>
      )}
      <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
        Required when sending an image input. Wired today: Gemma 3,
        Qwen 3.5 / 3.6 multimodal, and MiniCPM-V 4.6 (first-cut port —
        load errors will name any tensor that needs a one-line patch).
        Leave blank to auto-resolve the companion mmproj when the
        Model name looks like a paired set (e.g. MiniCPM-V).
      </p>
    </div>
  );
}

// Icon for the node header
const AIIcon = (
  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
  </svg>
);

function AILLMNode({ data }: AILLMNodeProps) {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();

  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 320,
    initialHeight: 420,
    constraints: { minWidth: 280, maxWidth: 500, minHeight: 350, maxHeight: 700 },
  });
  const [showAdvanced, setShowAdvanced] = useState(
    !!(data.headers || data.imageFormat || (data.imageInputCount && data.imageInputCount > 0))
  );
  const callbackRefs = useCallbackRefs(data);
  const isUserProviderChange = useRef(false);

  // Update React Flow's internal handle registry when image input count changes
  const imageInputCount = data.imageInputCount || 0;
  useEffect(() => {
    if (nodeId) {
      updateNodeInternals(nodeId);
    }
  }, [nodeId, updateNodeInternals, imageInputCount]);

  // Side effects for provider changes - only apply when user selects from dropdown
  useEffect(() => {
    if (!isUserProviderChange.current) return;
    isUserProviderChange.current = false;

    const providerValue = data.provider;
    if (!providerValue) return;

    // Find config
    const providerConfig = AI_PROVIDERS.find(p => p.value === providerValue);
    if (providerConfig && providerValue !== 'custom') {
      // Determine endpoint
      let endpoint = providerConfig.endpoint;
      if (providerValue === 'ollama') {
        const baseUrl = data.projectSettings?.ollamaEndpoint || 'http://localhost:11434';
        endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
      } else if (providerValue === 'lmstudio') {
        const baseUrl = data.projectSettings?.lmstudioEndpoint || 'http://localhost:1234';
        endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
      }

      // Only update if different (to avoid loops or unnecessary updates)
      if (data.endpoint !== endpoint && callbackRefs.onEndpointChange.current) {
        callbackRefs.onEndpointChange.current(endpoint);
      }
      if (data.model !== providerConfig.model && callbackRefs.onModelChange.current) {
        callbackRefs.onModelChange.current(providerConfig.model);
      }
      if (data.requestFormat !== providerConfig.requestFormat && callbackRefs.onRequestFormatChange.current) {
        callbackRefs.onRequestFormatChange.current(providerConfig.requestFormat);
      }
      if (providerConfig.apiKeyConstant && data.apiKeyConstant !== providerConfig.apiKeyConstant && callbackRefs.onApiKeyConstantChange.current) {
        callbackRefs.onApiKeyConstantChange.current(providerConfig.apiKeyConstant);
      }
      // HuggingFace LLM: pre-configure for vision support (1 image input, base64 inline)
      if (providerValue === 'huggingface') {
        if (!data.imageInputCount && callbackRefs.onImageInputCountChange.current) {
          callbackRefs.onImageInputCountChange.current(1);
        }
        if ((!data.imageFormat || data.imageFormat === 'none') && callbackRefs.onImageFormatChange.current) {
          callbackRefs.onImageFormatChange.current('base64_inline');
        }
      }
    }
  }, [data.provider, data.projectSettings?.ollamaEndpoint, data.projectSettings?.lmstudioEndpoint]);

  const handleModelChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    callbackRefs.onModelChange.current?.(e.target.value);
  }, []);
  const handleSystemPromptChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    callbackRefs.onSystemPromptChange.current?.(e.target.value);
  }, []);
  const handleEndpointChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    callbackRefs.onEndpointChange.current?.(e.target.value);
  }, []);
  const handleApiKeyChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    callbackRefs.onApiKeyChange.current?.(e.target.value);
  }, []);
  const handleApiKeyConstantChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    callbackRefs.onApiKeyConstantChange.current?.(e.target.value);
  }, []);
  const handleHeadersChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    callbackRefs.onHeadersChange.current?.(e.target.value);
  }, []);
  const handleImageFormatChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    callbackRefs.onImageFormatChange.current?.(e.target.value);
  }, []);
  const handleContextLengthChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0;
    callbackRefs.onContextLengthChange.current?.(value);
  }, []);
  const handleMaxTokensChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0;
    callbackRefs.onMaxTokensChange.current?.(value);
  }, []);
  const handleEnableThinkingChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    callbackRefs.onEnableThinkingChange.current?.(e.target.checked);
  }, []);
  // External-only since the pivot (2026-05-20): provider is driven by the
  // preset dropdown (handlePresetChange), so the standalone provider-select
  // handler is gone. onProviderChange is still called from handlePresetChange.

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    callbackRefs.onCollapsedChange.current?.(collapsed);
  }, []);

  // Get defaults from project settings if available
  const defaultProvider = data.projectSettings?.defaultAIProvider || 'huggingface';
  const defaultEndpoint = data.projectSettings?.defaultAIEndpoint || '';
  const defaultModel = data.projectSettings?.defaultAIModel || '';
  const defaultApiKeyConstant = data.projectSettings?.defaultAIApiKeyConstant || '';

  // Use node values, falling back to defaults
  const provider = data.provider || defaultProvider;
  const imageFormat = data.imageFormat || 'none';
  const mode: LlmMode = modeFromProvider(provider);
  const preset: ExternalPreset = presetFromProvider(provider);

  // External-only since the flow-builder pivot (2026-05-20): no mode toggle,
  // so the prior Internal/External switch handler + remembered-provider ref
  // are gone. `mode` is always 'external' (see modeFromProvider).

  const handlePresetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const next = e.target.value as ExternalPreset;
      isUserProviderChange.current = true;
      // Preset value maps 1:1 onto the stored provider name — including
      // `custom`, which the AI_PROVIDERS table treats as "no auto-fill".
      callbackRefs.onProviderChange.current?.(next as AIProvider);
    },
    [],
  );
  // "Raw override" UI gates whenever the user picked Custom for an
  // External run — that's the case where endpoint / model / apiKey are
  // hand-edited rather than auto-filled from a preset.
  const showRawOverride = mode === 'external' && preset === 'custom';
  // Kept for downstream label/validation logic; identical semantics to
  // the pre-redesign `isCustomProvider` flag.
  const isCustomProvider = provider === 'custom';
  const projectConstants = data.projectConstants || [];
  const apiKeyConstants = projectConstants.filter(c => c.category === 'api_key');
  const showBody = data.showBodyProperties !== false; // Check visibility

  // Handlers for image input count
  const handleAddImageInput = useCallback(() => {
    const newCount = Math.min((data.imageInputCount || 0) + 1, 10);
    callbackRefs.onImageInputCountChange.current?.(newCount);
  }, [data.imageInputCount]);

  const handleRemoveImageInput = useCallback(() => {
    const newCount = Math.max((data.imageInputCount || 0) - 1, 0);
    callbackRefs.onImageInputCountChange.current?.(newCount);
  }, [data.imageInputCount]);

  // Compute effective values (node overrides or defaults)
  const effectiveEndpoint = data.endpoint || defaultEndpoint;
  const effectiveModel = data.model || defaultModel;
  const effectiveApiKeyConstant = data.apiKeyConstant || defaultApiKeyConstant;

  // Compute validation issues. Raw override (External / Custom) is the
  // only mode that needs an explicit endpoint + model — every preset
  // ships defaults, and Internal pulls the model from plugin-llm's
  // manifest. `isCustomProvider` is retained for any external code path
  // that already keys off it.
  const validationIssues = useMemo(() => {
    const issues: ValidationIssue[] = [];
    if (!effectiveEndpoint && showRawOverride) {
      issues.push({ field: 'Endpoint', message: 'Required for raw override' });
    }
    if (!effectiveModel && showRawOverride) {
      issues.push({ field: 'Model', message: 'Required for raw override' });
    }
    return issues;
  }, [effectiveEndpoint, effectiveModel, showRawOverride]);

  // Collapsed preview content. Use Internal / External + the preset
  // (or raw provider for legacy values) — friendlier than the full
  // AI_PROVIDERS label after the redesign.
  const presetLabel = EXTERNAL_PRESETS.find(p => p.value === preset)?.label || preset;
  const headerLabel =
    mode === 'internal'
      ? 'Internal · plugin-llm'
      : preset === 'custom' && provider !== 'custom'
        ? `External · Custom (${provider})`
        : `External · ${presetLabel}`;
  const collapsedPreview = (
    <div className="text-slate-600 dark:text-slate-400 space-y-0.5">
      <div className="truncate text-purple-400 font-medium">
        {headerLabel}
      </div>
      {effectiveModel && (
        <div className="truncate text-xs text-slate-600 dark:text-slate-400">{effectiveModel}</div>
      )}
    </div>
  );

  // Build handle configurations
  const inputHandles = useMemo<HandleConfig[]>(() => {
    const handles: HandleConfig[] = [
      { id: 'prompt', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'prompt', labelColor: 'text-blue-400', size: 'lg' },
      { id: 'headers', type: 'target', position: Position.Left, color: '!bg-orange-500', label: 'headers', labelColor: 'text-orange-400', size: 'sm' },
      { id: 'apiKey', type: 'target', position: Position.Left, color: '!bg-yellow-500', label: 'api key', labelColor: 'text-yellow-400', size: 'sm' },
      { id: 'history', type: 'target', position: Position.Left, color: '!bg-purple-400', label: 'history', labelColor: 'text-purple-400', size: 'sm' },
    ];
    // Add dynamic image input handles - first one uses 'image' to match node definition
    for (let i = 0; i < imageInputCount; i++) {
      handles.push({
        id: i === 0 ? 'image' : `image_${i}`,
        type: 'target',
        position: Position.Left,
        color: '!bg-pink-500',
        label: imageInputCount === 1 ? 'image' : `image ${i + 1}`,
        labelColor: 'text-pink-400',
        size: 'sm'
      });
    }
    // Video input pin — matched to the `video` input in ai_llm.json.
    // Always rendered (optional) so MiniCPM-V 4.6 and future Qwen3-VL
    // video models can take a clip directly. The runtime extracts N
    // evenly-spaced frames (controlled by the Video Frames property)
    // and folds them into the image array for the underlying
    // vision-chat path. Text-only and image-only models ignore an
    // unconnected pin, so always-on costs nothing.
    handles.push({
      id: 'video',
      type: 'target',
      position: Position.Left,
      color: '!bg-orange-500',
      label: 'video',
      labelColor: 'text-orange-400',
      size: 'sm',
    });
    return handles;
  }, [imageInputCount]);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'response', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  // Resize handles
  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-purple-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-purple-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 's')}
      />
      <div
        className="nodrag absolute bottom-1 right-1 w-3 h-3 cursor-nwse-resize opacity-0 group-hover:opacity-100 transition-opacity"
        onMouseDown={(e) => handleResizeStart(e, 'se')}
      >
        <svg className="w-3 h-3 text-slate-500" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22Z" />
        </svg>
      </div>
    </>
  );

  return (
    <CollapsibleNodeWrapper
      title="AI / LLM"
      color="purple"
      icon={AIIcon}
      width={size.width}
      collapsedWidth={140}
      status={data._status}
      validationIssues={validationIssues}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      resizeHandles={resizeHandles}
    >
      {/* Primary Inputs */}
      {showBody && (
        <>
          {/* External-only (flow-builder pivot, 2026-05-20): the AI LLM node
              connects to any OpenAI-compatible HTTP engine — Ollama, LM Studio,
              vLLM, llama.cpp-server, or a cloud API. The old Internal/External
              toggle is gone since bundled plugin-llm was removed; the Preset
              picker below is the only top-level choice. */}
          {true && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Preset</label>
              <select
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
                value={preset}
                onChange={handlePresetChange}
                onMouseDown={(e) => e.stopPropagation()}
              >
                {EXTERNAL_PRESETS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
              {/* Legacy-provider rescue: workflows saved with the old
                  Google/OpenRouter/Groq/HuggingFace options collapse onto
                  `Custom` in the new UI. Surface the original name so the
                  user knows where their saved endpoint came from. */}
              {showRawOverride && provider !== 'custom' && (
                <p className="text-[10px] text-amber-700 dark:text-amber-400 mt-1 leading-tight">
                  Showing as Custom (raw override). Saved provider: {provider}.
                  Endpoint / model / API key are preserved; pick a preset above
                  to overwrite them.
                </p>
              )}
            </div>
          )}

          {/* Model Input — picker for Internal (GGUF manifest with download);
              plain text input for every External preset. */}
          {mode === 'internal' ? (
            <>
              <OaiyLocalModelPicker
                value={data.model || ''}
                onChange={(modelId) => callbackRefs.onModelChange.current?.(modelId)}
              />
              <OaiyMmprojPicker
                value={data.mmprojPath || ''}
                onChange={(path) => callbackRefs.onMmprojPathChange.current?.(path)}
              />
            </>
          ) : (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
                Model {!showRawOverride && <span className="text-slate-600">(override)</span>}
              </label>
              <input
                type="text"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
                placeholder={
                  AI_PROVIDERS.find(p => p.value === provider)?.placeholder ||
                  AI_PROVIDERS.find(p => p.value === provider)?.model ||
                  'provider-specific model id'
                }
                value={data.model || ''}
                onChange={handleModelChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {/* API Key from Constants */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
              API Key {apiKeyConstants.length > 0 && <span className="text-slate-600">(from settings)</span>}
            </label>
            {apiKeyConstants.length > 0 ? (
              <select
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
                value={effectiveApiKeyConstant}
                onChange={handleApiKeyConstantChange}
                onMouseDown={(e) => e.stopPropagation()}
              >
                <option value="">None / Manual</option>
                {apiKeyConstants.map((c) => (
                  <option key={c.id} value={c.key}>
                    {c.name} ({c.key})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="password"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
                placeholder="sk-..."
                value={data.apiKey || ''}
                onChange={handleApiKeyChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            )}
          </div>

          {/* System Prompt */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">System Prompt</label>
            <textarea
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 resize-none focus:outline-none focus:border-accent"
              rows={3}
              placeholder="You are a helpful assistant..."
              value={data.systemPrompt || ''}
              onChange={handleSystemPromptChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>
        </>
      )}

      {/* Advanced Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full flex items-center justify-between px-2 py-1.5 bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300 dark:hover:bg-slate-700 rounded text-xs text-slate-600 dark:text-slate-400 transition-colors"
      >
        <span>Advanced</span>
        <svg
          className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Advanced Settings */}
      {showAdvanced && showBody && (
        <div className="space-y-3 pt-1 border-t border-slate-300 dark:border-slate-700">
          {/* Endpoint URL Override */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
              Endpoint URL {!isCustomProvider && <span className="text-slate-600">(override)</span>}
            </label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent font-mono"
              placeholder={AI_PROVIDERS.find(p => p.value === provider)?.endpoint || 'https://api.openai.com/v1/chat/completions'}
              value={data.endpoint || ''}
              onChange={handleEndpointChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* ... other advanced inputs ... */}
          {/* Manual API Key (when no constants or need override) */}
          {apiKeyConstants.length > 0 && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
                Manual API Key <span className="text-slate-600">(override)</span>
              </label>
              <input
                type="password"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
                placeholder="sk-..."
                value={data.apiKey || ''}
                onChange={handleApiKeyChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {/* Image Inputs with +/- buttons */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Image Inputs</label>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRemoveImageInput}
                disabled={imageInputCount === 0}
                className="nodrag w-8 h-8 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                </svg>
              </button>
              <span className="flex-1 text-center text-sm text-slate-700 dark:text-slate-300">
                {imageInputCount} image{imageInputCount !== 1 ? 's' : ''}
              </span>
              <button
                onClick={handleAddImageInput}
                disabled={imageInputCount >= 10}
                className="nodrag w-8 h-8 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>
            {imageInputCount > 0 && (
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                Images will be sent to the AI for vision analysis
              </p>
            )}
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Image Input Format</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
              value={imageFormat}
              onChange={handleImageFormatChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {IMAGE_FORMATS.map((f) => (
                <option key={f.value} value={f.value}>{f.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Context Length (0 = default)</label>
            <input
              type="number"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent font-mono"
              placeholder="32768"
              value={data.contextLength || 0}
              min={0}
              max={131072}
              step={1024}
              onChange={handleContextLengthChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Max Tokens (0 = default 4096)</label>
            <input
              type="number"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent font-mono"
              placeholder="4096"
              value={data.maxTokens || 0}
              min={0}
              max={32768}
              step={256}
              onChange={handleMaxTokensChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                className="nodrag accent-purple-500"
                checked={data.enableThinking || false}
                onChange={handleEnableThinkingChange}
              />
              Enable Thinking
              <span className="text-slate-600 dark:text-slate-400 text-[10px]">(Qwen 3.5, etc.)</span>
            </label>
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Headers (JSON)</label>
            <textarea
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs text-slate-800 dark:text-slate-200 resize-none focus:outline-none focus:border-accent font-mono"
              rows={2}
              placeholder='{"X-Custom": "value"}'
              value={data.headers || ''}
              onChange={handleHeadersChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(AILLMNode);
