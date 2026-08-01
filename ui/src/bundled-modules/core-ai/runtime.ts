/**
 * Core AI Module Runtime
 *
 * Provides LLM chat, vision, and custom API request functionality.
 * This module handles communication with AI providers like OpenAI, Anthropic, Ollama, etc.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createAIMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

// =============================================================================
// INPUT VALIDATION
// =============================================================================

// Maximum endpoint URL length
const MAX_ENDPOINT_LENGTH = 2048;

/**
 * The bearer for THIS desktop's own API, when that is who we are calling.
 *
 * `/api/ai/*` is a privileged route and the guard there fails closed on a
 * missing `Origin` — deliberately, because any non-browser caller can forge
 * one. A flow run is exactly such a caller: the desktop spawns the CLI, which
 * has no browser origin at all, so every llm_chat node routed at the local
 * gateway came back `403 origin not allowed` and the whole flow failed on it.
 *
 * Returned ONLY when the endpoint is the very server the token belongs to,
 * compared by origin — a token that travelled to a third-party AI endpoint
 * would be handing this machine's admin credential to a stranger. Absent in a
 * browser (no `process`), where the webview's own origin is what is trusted.
 */
function localServerBearer(endpoint: string): string | null {
  if (typeof process === 'undefined' || !process.env) return null;
  const token = process.env.OAIY_SERVER_TOKEN;
  const base = process.env.OAIY_SERVER_URL;
  if (!token || !base) return null;
  try {
    return new URL(endpoint).origin === new URL(base).origin ? token : null;
  } catch {
    return null;
  }
}

// Maximum prompt/body length (10MB - allows for base64 images)
const MAX_CONTENT_LENGTH = 10 * 1024 * 1024;

/**
 * Validates an AI endpoint URL.
 * @throws Error if the URL is invalid or potentially dangerous.
 */
function validateEndpoint(endpoint: string, context: string): void {
  if (!endpoint || typeof endpoint !== 'string') {
    throw new Error(`${context}: Endpoint URL is required`);
  }

  if (endpoint.length > MAX_ENDPOINT_LENGTH) {
    throw new Error(`${context}: Endpoint URL exceeds maximum length`);
  }

  // Parse and validate URL
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(endpoint);
  } catch {
    throw new Error(`${context}: Invalid endpoint URL: ${endpoint}`);
  }

  // Only allow http and https
  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw new Error(`${context}: Invalid protocol. Only http and https are allowed.`);
  }

  // Block javascript: and data: URLs
  if (endpoint.toLowerCase().includes('javascript:')) {
    throw new Error(`${context}: JavaScript URLs are not allowed`);
  }
}

/**
 * Validates content length for prompts and request bodies.
 */
function validateContentLength(content: string | undefined, fieldName: string): void {
  if (content && content.length > MAX_CONTENT_LENGTH) {
    const sizeMB = (content.length / 1024 / 1024).toFixed(2);
    throw new Error(`${fieldName} exceeds maximum size of 10MB (current: ${sizeMB}MB)`);
  }
}

/**
 * Parse a chunk reference from text
 * Format: __CHUNK_REF:{"path":"...","index":0,"total":5,"startByte":0,"endByte":1000}__
 */
function parseChunkRef(text: string): {
  path: string;
  index: number;
  total: number;
  startByte: number;
  endByte: number;
} | null {
  const match = text.match(/__CHUNK_REF:({.*?})__/);
  if (match) {
    try {
      return JSON.parse(match[1]);
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Parse a file reference from text (for rejecting large files)
 */
function parseFileRef(text: string): { path: string; size: number } | null {
  const match = text.match(/__FILE_REF:({.*?})__/);
  if (match) {
    try {
      return JSON.parse(match[1]);
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Read chunk content from file using Tauri
 */
async function readChunkContent(chunkRef: {
  path: string;
  startByte: number;
  endByte: number;
}): Promise<string> {
  if (!ctx.tauri) {
    throw new Error('Tauri not available for reading chunks');
  }
  return ctx.tauri.invoke<string>('plugin:oaiy-filesystem|read_chunk_content', {
    path: chunkRef.path,
    start: chunkRef.startByte,
    length: chunkRef.endByte - chunkRef.startByte,
  });
}

/**
 * Default maximum dimension for images sent to AI (height or width)
 * Images larger than this will be resized while preserving aspect ratio
 * Can be overridden via module settings 'maxImageDimension'
 */
const DEFAULT_MAX_IMAGE_DIMENSION = 1024;

/**
 * Default maximum base64 size in KB for images sent to AI
 * Images larger than this will be aggressively compressed
 * Can be overridden via module settings 'maxImageSizeKB' or per-node settings
 * Note: Modern vision models accept much larger images (OpenAI up to 20MB)
 * but smaller sizes reduce latency and cost. Increase via node settings if needed.
 */
const DEFAULT_MAX_IMAGE_SIZE_KB = 200;

/**
 * Get the configured max image dimension
 */
function getMaxImageDimension(): number {
  const setting = ctx.getModuleSetting('maxImageDimension');
  if (typeof setting === 'number' && setting > 0) {
    return setting;
  }
  return DEFAULT_MAX_IMAGE_DIMENSION;
}

/**
 * Get the configured max image size in KB
 */
function getMaxImageSizeKB(): number {
  const setting = ctx.getModuleSetting('maxImageSizeKB');
  if (typeof setting === 'number' && setting > 0) {
    return setting;
  }
  return DEFAULT_MAX_IMAGE_SIZE_KB;
}

/**
 * Resize a base64 image to fit within configured limits
 * Prefers Rust backend for better performance (doesn't block UI thread)
 * Falls back to canvas for browser-only environments
 *
 * @param dataUrl - The image data URL to resize
 * @param maxDimensionOverride - Override max dimension (0 = use default)
 * @param maxSizeKBOverride - Override max size in KB (0 = use default)
 */
async function resizeImageIfNeeded(
  dataUrl: string,
  maxDimensionOverride: number = 0,
  maxSizeKBOverride: number = 0
): Promise<string> {
  // Only process data URLs that are images
  if (!dataUrl.startsWith('data:image')) {
    return dataUrl;
  }

  // Use overrides if provided (> 0), otherwise fall back to defaults
  const maxDimension = maxDimensionOverride > 0 ? maxDimensionOverride : getMaxImageDimension();
  const maxSizeKB = maxSizeKBOverride > 0 ? maxSizeKBOverride : getMaxImageSizeKB();

  // Try to use Rust backend for better performance (runs off UI thread)
  if (ctx.tauri) {
    try {
      const result = await ctx.tauri.invoke<{
        success: boolean;
        dataUrl: string | null;
        originalWidth: number;
        originalHeight: number;
        newWidth: number;
        newHeight: number;
        originalSizeKb: number;
        newSizeKb: number;
        error: string | null;
      }>('resize_image', { dataUrl, maxDimension, maxSizeKb: maxSizeKB });

      if (result.success && result.dataUrl) {
        ctx.log('info', `[AI] Image resized via Rust: ${result.originalWidth}x${result.originalHeight} -> ${result.newWidth}x${result.newHeight}, ${result.originalSizeKb}KB -> ${result.newSizeKb}KB`);
        return result.dataUrl;
      } else if (result.error) {
        ctx.log('warn', `[AI] Rust resize failed: ${result.error}, falling back to canvas`);
      }
    } catch (err) {
      ctx.log('warn', `[AI] Rust resize not available: ${err}, falling back to canvas`);
    }
  }

  // Fallback: Check if we're in a browser environment with canvas support
  if (typeof document === 'undefined' || typeof Image === 'undefined') {
    ctx.log('info', '[AI] Canvas not available, skipping image resize');
    return dataUrl;
  }

  // Canvas-based fallback for browser-only environments
  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => {
      const { width, height } = img;
      const originalSizeKB = Math.round(dataUrl.length / 1024);

      // Check if resize is needed (either dimensions too large OR file size too large)
      const dimensionsOk = width <= maxDimension && height <= maxDimension;
      const sizeOk = originalSizeKB <= maxSizeKB;
      const isPng = dataUrl.startsWith('data:image/png');

      // Always convert PNG to JPEG for smaller size, even if dimensions are ok
      if (dimensionsOk && sizeOk && !isPng) {
        ctx.log('info', `[AI] Image ${width}x${height} (${originalSizeKB}KB) within limits, no resize needed`);
        resolve(dataUrl);
        return;
      }

      // Calculate new dimensions preserving aspect ratio
      let newWidth: number;
      let newHeight: number;

      if (width > height) {
        newWidth = Math.min(width, maxDimension);
        newHeight = Math.round(height * (newWidth / width));
      } else {
        newHeight = Math.min(height, maxDimension);
        newWidth = Math.round(width * (newHeight / height));
      }

      // If size is still too large after dimension resize, scale down more aggressively
      if (originalSizeKB > maxSizeKB) {
        const sizeRatio = originalSizeKB / maxSizeKB;
        const scaleFactor = 1 / Math.sqrt(sizeRatio);
        newWidth = Math.max(256, Math.round(newWidth * scaleFactor));
        newHeight = Math.max(256, Math.round(newHeight * scaleFactor));
      }

      ctx.log('info', `[AI] Resizing image via canvas from ${width}x${height} (${originalSizeKB}KB) to ${newWidth}x${newHeight}`);

      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;

      const context = canvas.getContext('2d');
      if (!context) {
        ctx.log('error', '[AI] Failed to get canvas context');
        resolve(dataUrl);
        return;
      }

      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.drawImage(img, 0, 0, newWidth, newHeight);

      let resizedDataUrl = canvas.toDataURL('image/jpeg', 0.8);

      // If still too large, reduce quality further
      let attempts = 0;
      let quality = 0.8;
      while (resizedDataUrl.length / 1024 > maxSizeKB && attempts < 3 && quality > 0.3) {
        quality -= 0.2;
        resizedDataUrl = canvas.toDataURL('image/jpeg', quality);
        attempts++;
      }

      const newSizeKB = Math.round(resizedDataUrl.length / 1024);
      ctx.log('info', `[AI] Image resized: ${originalSizeKB}KB -> ${newSizeKB}KB (quality: ${quality.toFixed(1)})`);

      resolve(resizedDataUrl);
    };

    img.onerror = () => {
      ctx.log('error', '[AI] Failed to load image for resizing');
      resolve(dataUrl);
    };

    img.src = dataUrl;
  });
}

/**
 * Get user-friendly error message
 */
function getUserFriendlyError(error: string, context: string): string {
  if (error.includes('Failed to fetch') || error.includes('NetworkError')) {
    return `Network error during ${context}. Check your internet connection and endpoint URL.`;
  }
  if (error.includes('401') || error.includes('Unauthorized')) {
    return `Authentication failed for ${context}. Check your API key.`;
  }
  if (error.includes('429') || error.includes('rate limit')) {
    return `Rate limit exceeded for ${context}. Please wait and try again.`;
  }
  if (error.includes('timeout') || error.includes('AbortError')) {
    return `Request timed out for ${context}.`;
  }
  return error;
}

/**
 * Resolve API key from constant name or direct value
 */
function resolveApiKey(apiKeyConstant: string): string {
  if (!apiKeyConstant) return '';

  // Check if this looks like a direct API key (not a constant name)
  // Common prefixes: sk- (OpenAI), anthropic- (Anthropic), gsk_ (Groq), etc.
  if (apiKeyConstant.startsWith('sk-') ||
      apiKeyConstant.startsWith('anthropic-') ||
      apiKeyConstant.startsWith('gsk_') ||
      apiKeyConstant.length > 40) {
    // Likely a direct API key, return as-is
    return apiKeyConstant;
  }

  // Try to get from context's getConstant if available
  if (ctx.getConstant) {
    const key = ctx.getConstant(apiKeyConstant);
    if (key) return key;
  }
  // Try module settings
  const settingKey = ctx.getModuleSetting(apiKeyConstant);
  if (typeof settingKey === 'string') return settingKey;
  // Return empty if not found
  return '';
}

/**
 * Legacy eager-unload helper, gated on `OAIY_LLM_EAGER_UNLOAD=1`. The
 * default behaviour is to LEAVE the LLM loaded across calls; the
 * diffusion plugin's `ensureLoadedWithQuant` evicts everything plugin-
 * llm holds before its own load so VRAM stays safe. Set the env var
 * to restore the pre-2026-05-14 behaviour where every chat unloaded.
 */
/**
 * Strip reasoning-model `<think>...</think>` (and `<thinking>...</thinking>`)
 * blocks from a chat completion before handing it back to the workflow.
 *
 * Why this matters: Qwen 3.5 thinking variants, DeepSeek R1, and the
 * various Reasoner-style local + cloud LLMs interleave their chain of
 * thought into the same response stream as the final answer. Without
 * stripping, the user-visible text starts with several paragraphs of
 * scratchpad ("Let me think about this... Actually, the user wants...")
 * before the actual answer — which then propagates downstream into any
 * node that consumes the AI output (Save File, another LLM, a diffusion
 * prompt, etc.) and produces nonsense results.
 *
 * Cleanup:
 *   - Removes well-formed `<think>...</think>` / `<thinking>...</thinking>`
 *     blocks. Case-insensitive, multiline-safe, non-greedy so multiple
 *     blocks in the same response each get stripped.
 *   - If an opening tag has no matching close (model hit max_tokens
 *     mid-thought), drops everything from that tag to the end — the
 *     half-finished reasoning isn't useful as an answer.
 *   - Trims leading whitespace so the visible text starts on a real
 *     character rather than the gap the stripped block used to occupy.
 *
 * Applied at every return path of `chat()` that produces user-visible
 * text. Models that don't emit think tags are unaffected (the regex
 * just doesn't match anything).
 */
function stripThinkTags(text: string): string {
  if (!text) return text;
  let out = text.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '');
  // Trailing unclosed-thought drop. `i` flag handles either case mix.
  const openTail = out.match(/<think(?:ing)?>[\s\S]*$/i);
  if (openTail && typeof openTail.index === 'number') {
    out = out.slice(0, openTail.index);
  }
  return out.replace(/^\s+/, '');
}

async function tryEagerLlmUnload(modelId: string): Promise<void> {
  const tauri = ctx.tauri;
  if (!tauri) return;
  try {
    await tauri.invoke<boolean>('plugin:oaiy-llm|llm_unload_model', { modelId });
    ctx.log?.('info', `[AI] oaiy-local unloaded ${modelId} (OAIY_LLM_EAGER_UNLOAD)`);
  } catch (e) {
    ctx.log?.('warn', `[AI] oaiy-local post-call unload of ${modelId} failed: ${String(e)}`);
  }
}

/**
 * Resolve a raw image input into a real on-disk path suitable for the
 * plugin-llm vision_chat code path (which calls `preprocess_image`
 * directly and needs a `&Path`).
 *
 * Returns:
 *   - a path string when the input is recognisably an image,
 *   - `null` when it's plain text (the caller appends it to the prompt).
 *
 * Handles:
 *   - existing absolute paths with image extensions (returned as-is),
 *   - `file://` URIs (prefix-stripped, percent-decoded),
 *   - `data:image/*;base64,...` URLs (decoded + written to a temp file),
 *   - `http(s)://...` image URLs (fetched + written to a temp file).
 *
 * Non-image strings produce `null` — those are treated as text and
 * folded into the user prompt rather than being silently dropped.
 */
/**
 * Find a companion `mmproj-*.gguf` next to the user-picked model
 * file. Used when the user wires an image input but leaves the
 * Vision Projector field blank. Scans the model's containing
 * directory, filters to `mmproj-*.gguf`, and ranks by longest common
 * prefix with the model filename — so
 * `MiniCPM-V-4.6-Abliterated-AND-Disinhibited-Q4_K_M.gguf` resolves
 * to `mmproj-MiniCPM-V-4.6-Abliterated-AND-Disinhibited-F16.gguf`
 * rather than an unrelated mmproj that happens to share the dir.
 *
 * Only runs when the model path is absolute (rooted or UNC) — for
 * dir-relative model ids the LLM plugin's own loader fallback owns
 * the resolution, and we don't have a known absolute dir to scan.
 *
 * Returns the absolute path to the matched mmproj, or null if none
 * found / not in a Tauri context.
 */
async function autoResolveMmproj(modelPath: string): Promise<string | null> {
  if (!ctx?.tauri) return null;
  const isAbsolute =
    /^[a-zA-Z]:[\\/]/.test(modelPath) ||
    modelPath.startsWith('/') ||
    modelPath.startsWith('\\\\');
  if (!isAbsolute) return null;

  const fileName = modelPath.split(/[\\/]/).pop() || '';
  const dir = modelPath.slice(0, modelPath.length - fileName.length).replace(/[\\/]+$/, '');
  if (!dir || !fileName) return null;

  type FileInfo = { name: string; path: string };
  let entries: FileInfo[];
  try {
    entries = await ctx.tauri.invoke<FileInfo[]>(
      'plugin:oaiy-filesystem|list_folder',
      {
        path: dir,
        recursive: false,
        includePatterns: ['*.gguf'],
        excludePatterns: [],
        maxFiles: 200,
      },
    );
  } catch {
    return null;
  }
  if (!Array.isArray(entries) || entries.length === 0) return null;

  // Filter to actual mmproj files. The convention is a `mmproj-`
  // prefix; some converters use `clip-` or bare `mmproj.gguf`, accept
  // those as well.
  const candidates = entries.filter((e) => {
    const lower = (e.name || '').toLowerCase();
    return lower.endsWith('.gguf') &&
      (lower.startsWith('mmproj') || lower.startsWith('clip-') || lower === 'mmproj.gguf');
  });
  if (candidates.length === 0) return null;

  // Pick the candidate whose name shares the longest common substring
  // with the model filename's stem (ignoring case and the `mmproj-`
  // prefix). Sort by score desc, fall back to the first if scores tie.
  const modelStem = fileName.replace(/\.gguf$/i, '').toLowerCase();
  const score = (name: string): number => {
    const stem = name.replace(/^mmproj-?/i, '').replace(/\.gguf$/i, '').toLowerCase();
    // Strip common quant suffixes from both sides so the comparison
    // focuses on the actual model family name (Q4_K_M vs F16 are
    // expected to differ between model and mmproj).
    const strip = (s: string) =>
      s.replace(/[-_]?(q[0-9]+(_[a-z0-9]+)*|f16|fp16|bf16|f32|fp32)$/i, '');
    const a = strip(modelStem);
    const b = strip(stem);
    // Longest common prefix length.
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i++;
    return i;
  };
  candidates.sort((a, b) => score(b.name) - score(a.name));
  return candidates[0].path || null;
}

async function materializeImageInputToPath(value: unknown): Promise<string | null> {
  if (value == null) return null;
  if (typeof value === 'object') {
    const obj = value as { path?: string; outputPath?: string; image?: string };
    return materializeImageInputToPath(obj.path || obj.outputPath || obj.image || null);
  }
  if (typeof value !== 'string') return null;
  const s = value.trim();
  if (!s) return null;

  const imageExt = /\.(png|jpe?g|gif|webp|bmp)$/i;
  const looksLikePath =
    /^[a-zA-Z]:[\\/]/.test(s) || s.startsWith('/') || s.startsWith('\\\\');
  if (looksLikePath && imageExt.test(s)) {
    return s;
  }

  if (s.startsWith('file://')) {
    return decodeURIComponent(s.replace(/^file:\/\/\/?/, ''));
  }

  // data: URL — decode + write to temp via the filesystem plugin.
  if (s.startsWith('data:image/')) {
    if (!ctx?.tauri) return null;
    const match = s.match(/^data:(image\/[^;]+);base64,(.+)$/);
    if (!match) return null;
    const mime = match[1];
    const base64 = match[2];
    const mimeExt = (mime.split('/')[1] || 'png').replace(/\+.*$/, '');
    const tempDir = await ctx.tauri.invoke<string>(
      'plugin:oaiy-filesystem|get_temp_dir',
    );
    const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const tempPath = `${tempDir}/oaiy-llm-vis-${stamp}.${mimeExt}`;
    await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
      path: tempPath,
      content: base64,
      contentType: 'base64',
      createDirs: true,
    });
    return tempPath;
  }

  // http(s):// — only treat as an image when the URL pathname (or
  // ?filename=) ends in an image extension. Avoids fetching random
  // text URLs the user might pipe in.
  if (s.startsWith('http://') || s.startsWith('https://')) {
    let isImageUrl = false;
    try {
      const url = new URL(s);
      const pathname = url.pathname.toLowerCase();
      const fname = (url.searchParams.get('filename') || '').toLowerCase();
      isImageUrl = imageExt.test(pathname) || imageExt.test(fname);
    } catch {
      return null;
    }
    if (!isImageUrl) return null;
    if (!ctx?.tauri || !ctx?.secureFetch) return null;
    try {
      const response = await ctx.secureFetch(s, {
        method: 'GET',
        purpose: 'Fetch image for AI vision (plugin-llm mmproj)',
      });
      if (!response.ok) return null;
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      let bin = '';
      // 32k chunks keep us under the JS call-stack limit for big images.
      const CHUNK = 32_768;
      for (let i = 0; i < bytes.length; i += CHUNK) {
        bin += String.fromCharCode.apply(
          null,
          Array.from(bytes.subarray(i, i + CHUNK)),
        );
      }
      const base64 = btoa(bin);
      const mime = blob.type || 'image/png';
      const mimeExt = (mime.split('/')[1] || 'png').replace(/\+.*$/, '');
      const tempDir = await ctx.tauri.invoke<string>(
        'plugin:oaiy-filesystem|get_temp_dir',
      );
      const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      const tempPath = `${tempDir}/oaiy-llm-vis-${stamp}.${mimeExt}`;
      await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
        path: tempPath,
        content: base64,
        contentType: 'base64',
        createDirs: true,
      });
      return tempPath;
    } catch {
      return null;
    }
  }

  return null;
}

/**
 * Execute LLM chat request
 *
 * Parameters match compiler output:
 * AI.chat(systemPrompt, userPrompt, input, endpoint, model, apiKeyConstant,
 *         streaming, maxTokens, temperature, responseFormat, includeImages,
 *         visionDetail, nodeId, chunkRefs, messageHistory, maxImageDimension, maxImageSizeKB, enableThinking)
 */
// ---------------------------------------------------------------------------
// Custom-provider request/response templating (the "Custom" provider).
// Mirrors the TTS external node: render `{{var}}` placeholders (JSON-escaped;
// append `Raw` for unescaped, e.g. {{messagesJsonRaw}}) into the request body
// + headers, and extract the reply text via a dot/bracket JSON path.
// ---------------------------------------------------------------------------
function renderTemplate(template: string, vars: Record<string, string>, jsonEscape: boolean): string {
  return template.replace(/\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}/g, (full, name) => {
    const isRaw = name.endsWith('Raw');
    const baseKey = isRaw ? name.slice(0, -3) : name;
    if (!Object.prototype.hasOwnProperty.call(vars, baseKey)) return full;
    const value = vars[baseKey] ?? '';
    if (!jsonEscape) return value;
    return isRaw ? value : JSON.stringify(value);
  });
}

function extractJsonPath(root: unknown, path: string): unknown {
  if (!path || path.trim() === '') return root;
  const segs: (string | number)[] = [];
  const re = /([^.[\]]+)|\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(path)) !== null) {
    if (m[1] !== undefined) segs.push(/^\d+$/.test(m[1]) ? Number(m[1]) : m[1]);
    else if (m[2] !== undefined) segs.push(Number(m[2]));
  }
  let cur: unknown = root;
  for (const s of segs) {
    if (cur === null || cur === undefined) return undefined;
    cur = (cur as Record<string | number, unknown>)[s];
  }
  return cur;
}

async function chat(
  systemPrompt: string,
  userPrompt: string,
  input: unknown,
  endpoint: string,
  model: string,
  apiKeyConstant: string,
  streaming: boolean,
  maxTokens: number,
  temperature: number,
  responseFormat: string,
  includeImages: boolean,
  visionDetail: string,
  nodeId: string,
  chunkRefs: Array<{ documentId: string; content?: string }> = [],
  messageHistory: string | Array<{ role: string; content: string }> = '',
  maxImageDimensionOverride: number = 0,
  maxImageSizeKBOverride: number = 0,
  enableThinking: boolean = false,
  mmprojPath: string = '',
  videoInput: unknown = null,
  videoFrames: number = 8,
  customBodyTemplate: string = '',
  customResponsePath: string = '',
  customHeaders: string = ''
): Promise<string> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[AI] Aborted by user before chat started');
    return '__ABORT__';
  }

  // Fold video frames into the image input array up-front, before any
  // provider branch sees `input`. The frame extractor is a no-op when
  // `videoInput` is null/empty, so the cost is zero for text-only or
  // image-only chats.
  if (videoInput != null && videoInput !== '') {
    try {
      const frames = await extractVideoFrames(videoInput, videoFrames);
      if (frames.length > 0) {
        const existing: unknown[] = Array.isArray(input)
          ? [...(input as unknown[])]
          : input != null && input !== ''
            ? [input]
            : [];
        existing.push(...frames);
        input = existing.length === 1 ? existing[0] : existing;
      }
    } catch (err) {
      ctx?.log?.('warn', `[AI] video frame extraction skipped: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // OAIY Local (GGUF) provider — dispatch to the plugin-llm Tauri commands
  // instead of HTTP. We intercept BEFORE validateEndpoint because that
  // helper rejects non-http(s) schemes. The compiler points `endpoint` at
  // `oaiy-llm://local` when the user picks the "OAIY Local (GGUF, Rust)"
  // provider.
  if (typeof endpoint === 'string' && endpoint.startsWith('oaiy-llm://')) {
    validateContentLength(systemPrompt, 'System prompt');
    validateContentLength(userPrompt, 'User prompt');
    if (!ctx.tauri) {
      throw new Error(
        'AI chat: OAIY Local (GGUF) provider requires the desktop Tauri runtime — not available in this environment',
      );
    }
    if (!model) {
      throw new Error(
        'AI chat: OAIY Local (GGUF) requires a model id (e.g. "qwen3-1.7b-q4-k-m") in the Model field',
      );
    }
    // Image input → file paths for the engine. plugin-llm's vision_chat
    // takes Paths (it calls preprocess_image internally), so any
    // data: URLs from upstream nodes need to be materialised to temp
    // files. HTTP URLs are fetched + written. Bare paths pass through.
    // Empty when the upstream pin is unconnected or carries non-image
    // payloads (those get appended to the prompt instead).
    const imagePaths: string[] = [];
    let appendedText = '';
    const inputItems = Array.isArray(input) ? input : (input ? [input] : []);
    for (const item of inputItems) {
      if (typeof item !== 'string' || !item) continue;
      try {
        const materialised = await materializeImageInputToPath(item);
        if (materialised) {
          imagePaths.push(materialised);
        } else {
          // Non-image string — append to the user prompt the same way
          // the cloud path does. Lets a chain `prompt → AI` work even
          // when the LLM node is wired through the generic input pin.
          appendedText += (appendedText ? '\n\n' : '') + item;
        }
      } catch (err) {
        ctx.log('warn', `[AI] oaiy-local image input skipped: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    const hasImages = imagePaths.length > 0;
    let mmprojClean = (mmprojPath || '').trim();
    // Auto-resolve a companion mmproj when the user wired images but
    // left the Vision Projector field blank. Works for ANY
    // `mmproj-*.gguf` file sitting next to the model (no hardcoded
    // names — handles Abliterated / Distil / community-rebrand
    // variants by scanning the model's directory for an mmproj file
    // and matching by longest common prefix). Only attempts when (a)
    // images are connected, (b) projector is empty, (c) the model id
    // looks like an absolute path so we know where to look.
    if (hasImages && !mmprojClean && typeof model === 'string' && model.length > 0) {
      try {
        const resolved = await autoResolveMmproj(model);
        if (resolved) {
          mmprojClean = resolved;
          ctx?.log?.('info', `[AI] auto-resolved mmproj for ${model} → ${resolved}`);
        }
      } catch (err) {
        ctx?.log?.('warn', `[AI] mmproj auto-resolve failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    if (hasImages && !mmprojClean) {
      throw new Error(
        'AI chat: OAIY Local (GGUF) image input requires a Vision Projector (mmproj) path on the node. ' +
        'Pick a companion `mmproj-*.gguf` in the AI LLM node advanced settings — supported today: ' +
        'Gemma 3, Qwen 3.5 / 3.6 multimodal, and MiniCPM-V 4.6 (first-cut port; load errors will name ' +
        'any tensor mismatches that need fixing).',
      );
    }

    ctx.onNodeStatus?.(nodeId, 'running');

    // Build the messages array. History may arrive as a structured array
    // (preferred) or as a newline-joined string from a chained step output.
    const messages: Array<{ role: string; content: string }> = [];
    if (systemPrompt && systemPrompt.length > 0) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    if (Array.isArray(messageHistory)) {
      for (const m of messageHistory) {
        if (m && typeof m === 'object' && typeof (m as { content?: string }).content === 'string') {
          messages.push({
            role: String((m as { role?: string }).role || 'user'),
            content: String((m as { content?: string }).content || ''),
          });
        }
      }
    } else if (typeof messageHistory === 'string' && messageHistory.length > 0) {
      messages.push({
        role: 'system',
        content: 'Conversation so far:\n' + messageHistory.slice(-5000),
      });
    }
    const finalUserPrompt = appendedText
      ? (userPrompt ? `${userPrompt}\n\n${appendedText}` : appendedText)
      : userPrompt;
    if (finalUserPrompt && finalUserPrompt.length > 0) {
      messages.push({ role: 'user', content: finalUserPrompt });
    }

    // Symmetric eviction: free anything the diffusion plugin is holding in
    // its in-memory model_cache (LTX 22B INT8 weighs ~24 GB) BEFORE we load
    // the LLM. Without this, a chain that runs diffusion-then-LLM (or the
    // warm-rerun of LLM→HiDream→LTX after LTX is left cached) fights for
    // VRAM during the Gemma cold load and the load time blows up 4×.
    // plugin-diffusion does the mirror call to plugin:oaiy-llm|llm_unload_all
    // from `ensureLoadedWithQuant`.
    try {
      const evicted = await ctx.tauri.invoke<number>(
        'plugin:oaiy-diffusion|diffusion_unload_all',
      );
      if (typeof evicted === 'number' && evicted > 0) {
        ctx.log(
          'info',
          `[AI] evicted ${evicted} loaded diffusion model(s) to free VRAM for ${model}`,
        );
      }
    } catch {
      // plugin-diffusion not installed in this build, or call failed; the
      // LLM load will still work, it just won't be as fast as a clean slate.
    }
    try {
      const evicted = await ctx.tauri.invoke<number>(
        'plugin:oaiy-tts|tts_unload_all',
      );
      if (typeof evicted === 'number' && evicted > 0) {
        ctx.log(
          'info',
          `[AI] evicted ${evicted} loaded TTS model(s) to free VRAM for ${model}`,
        );
      }
    } catch {
      // plugin-tts not installed — same handling.
    }

    try {
      // Explicit load first when a projector is requested. plugin-llm
      // does lazy-load inside llm_chat, but only with `mmproj_path: None`
      // — to attach the projector (and to evict + reload if it changed
      // mid-session) we need to call llm_load_model ourselves. Skip
      // when no mmproj is set; the lazy path is fine then.
      if (mmprojClean) {
        await ctx.tauri.invoke('plugin:oaiy-llm|llm_load_model', {
          args: {
            modelId: model,
            mmprojPath: mmprojClean,
          },
        });
      }
      const result = await ctx.tauri.invoke<{
        text: string;
        modelId: string;
        backend: string;
        promptTokens: number;
        generatedTokens: number;
      }>('plugin:oaiy-llm|llm_chat', {
        args: {
          modelId: model,
          messages,
          maxNewTokens: maxTokens > 0 ? maxTokens : 512,
          temperature: typeof temperature === 'number' ? temperature : 0.8,
          imagePaths: hasImages ? imagePaths : undefined,
        },
      });
      ctx.log(
        'info',
        `[AI] oaiy-local ${model} (${result.backend}): ${result.promptTokens} prompt → ${result.generatedTokens} new tokens` +
          (hasImages ? ` [+${imagePaths.length} image(s) via mmproj]` : ''),
      );
      ctx.onNodeStatus?.(nodeId, 'completed');
      return stripThinkTags(result.text);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      ctx.log('error', `[AI] oaiy-local chat failed: ${msg}`);
      ctx.onNodeStatus?.(nodeId, 'error');
      throw err;
    }
    // Note: NO eager unload here anymore. Earlier versions called
    // `plugin:oaiy-llm|llm_unload_model` after every chat to free VRAM
    // for downstream diffusion, but that forced a full ~12-15 s cold
    // reload on the *next* LLM call (and on every chain re-run, even
    // when the GGUF is already on the OS disk cache). The replacement
    // is cross-plugin eviction: `plugin-diffusion`'s
    // `ensureLoadedWithQuant` calls `plugin:oaiy-llm|llm_unload_all`
    // right before it loads its own heavy weights, so VRAM stays safe
    // for the chain workflow without making pure-LLM workflows pay a
    // cold reload on every call.
    // The user can still manually evict via `Diffusion.unloadModel` or
    // the Models settings tab if they want to free VRAM explicitly.
    // Set `OAIY_LLM_EAGER_UNLOAD=1` to restore the legacy behaviour.
    const eagerEnv = (
      (globalThis as Record<string, unknown>).process as { env?: Record<string, string> } | undefined
    )?.env?.OAIY_LLM_EAGER_UNLOAD;
    if (eagerEnv === '1') {
      await tryEagerLlmUnload(model);
    }
  }

  // Validate inputs before proceeding
  validateEndpoint(endpoint, 'AI chat');
  validateContentLength(systemPrompt, 'System prompt');
  validateContentLength(userPrompt, 'User prompt');

  // Auto-start local services (Wan2GP, Ollama, etc.) if needed
  if (ctx.tauri && endpoint.includes('127.0.0.1')) {
    const portMatch = endpoint.match(/:(\d+)/);
    if (portMatch) {
      const port = parseInt(portMatch[1], 10);
      try {
        const result = await ctx.tauri.invoke<{
          success: boolean;
          port?: number;
          error?: string;
          already_running: boolean;
        }>('ensure_service_ready_by_port', { port });
        if (result.success && result.port && result.port !== port) {
          endpoint = endpoint.replace(`:${port}`, `:${result.port}`);
        }
      } catch {
        // Service auto-start not available
      }
    }
  } else if (!ctx.tauri && endpoint.includes('127.0.0.1')) {
    // Browser build (oaiy-web): no Tauri, but the OAIY Desktop may own
    // this port. Ask it to start the matching service if it's stopped —
    // so picking an OAIY Desktop service in a flow and running it "just
    // works" without manually starting it first (Phase 3.5). Fire-and-
    // forget: OAIY Desktop returns as soon as the spawn kicks off and
    // the 503/connection-refused retry loop below rides out the warm-up.
    const portMatch = endpoint.match(/:(\d+)/);
    if (portMatch) {
      const port = parseInt(portMatch[1], 10);
      try {
        // 17972 is OAIY Desktop's fixed localhost API port (see
        // ui/src/lib/desktopDetection.ts). Plain fetch is fine — the
        // companion sets permissive CORS and the bind is loopback-only.
        // Hard 2s cap: ensure-by-port returns as soon as the spawn kicks
        // off, so a healthy companion answers in <100ms. The timeout keeps
        // a wedged companion (accepting connections but not responding)
        // from hanging an otherwise-fine flow run — on abort we fall
        // through to the direct request just like OAIY Desktop-absent case.
        const ctrl = new AbortController();
        const abortTimer = setTimeout(() => ctrl.abort(), 2000);
        const r = await fetch('http://127.0.0.1:17972/api/services/ensure-by-port', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ port }),
          credentials: 'omit',
          signal: ctrl.signal,
        }).finally(() => clearTimeout(abortTimer));
        if (r.ok) {
          const res = (await r.json().catch(() => null)) as
            | { found?: boolean; started?: boolean; name?: string }
            | null;
          if (res?.found && res.started) {
            ctx.log('info', `[AI] companion started "${res.name ?? 'service'}" on :${port} for this run`);
          }
        }
      } catch {
        // OAIY Desktop not running / not reachable — fall through and let
        // the request hit the endpoint directly (it may be a non-
        // companion local server the user started themselves).
      }
    }
  }

  ctx.onNodeStatus?.(nodeId, 'running');

  // Build the prompt from system + user + input
  let finalPrompt = userPrompt;

  // Normalize input to array of images (can be single string, array, or null)
  // Images can be: data:image URLs, file paths, or URLs
  let imageInputs: string[] = [];

  // Helper to check if a string looks like an image file path
  const isImagePath = (s: string): boolean => {
    const lower = s.toLowerCase();
    return (
      (lower.includes('\\') || lower.includes('/')) &&
      (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') ||
       lower.endsWith('.gif') || lower.endsWith('.webp') || lower.endsWith('.bmp'))
    );
  };

  // Helper to check if a string is an image URL (http/https with image extension)
  const isImageUrl = (s: string): boolean => {
    try {
      const url = new URL(s);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
      // Check the pathname for image extensions (ignoring query params)
      const pathname = url.pathname.toLowerCase();
      // Also check query params for filename with image extension (e.g., ComfyUI)
      const filename = url.searchParams.get('filename')?.toLowerCase() || '';
      return (
        pathname.endsWith('.png') || pathname.endsWith('.jpg') || pathname.endsWith('.jpeg') ||
        pathname.endsWith('.gif') || pathname.endsWith('.webp') || pathname.endsWith('.bmp') ||
        filename.endsWith('.png') || filename.endsWith('.jpg') || filename.endsWith('.jpeg') ||
        filename.endsWith('.gif') || filename.endsWith('.webp') || filename.endsWith('.bmp')
      );
    } catch {
      return false;
    }
  };

  // Helper to fetch an image URL and convert to base64
  const fetchImageUrl = async (url: string): Promise<string | null> => {
    try {
      ctx.log('info', `[AI] Fetching image from URL: ${url}`);
      const response = await ctx.secureFetch(url, {
        method: 'GET',
        purpose: 'Fetch image for AI vision',
      });
      if (!response.ok) {
        ctx.log('warn', `[AI] Failed to fetch image: HTTP ${response.status}`);
        return null;
      }
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const base64 = btoa(
        new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
      );
      const mimeType = blob.type || 'image/png';
      ctx.log('info', `[AI] Fetched image: ${Math.round(base64.length / 1024)}KB, type: ${mimeType}`);
      return `data:${mimeType};base64,${base64}`;
    } catch (err) {
      ctx.log('error', `[AI] Failed to fetch image URL ${url}: ${err}`);
      return null;
    }
  };

  // Helper to read a local image file and convert to base64
  const readImageFile = async (filePath: string): Promise<string | null> => {
    if (!ctx.tauri) {
      ctx.log('warn', '[AI] Cannot read local file - Tauri not available');
      return null;
    }
    try {
      // Normalize path - remove Windows \\?\ prefix if present
      let normalizedPath = filePath;
      if (normalizedPath.startsWith('\\\\?\\')) {
        normalizedPath = normalizedPath.substring(4);
      }

      const fileContent = await ctx.tauri.invoke<{ content: string; isLargeFile: boolean }>('plugin:oaiy-filesystem|read_file', {
        path: normalizedPath,
        readAs: 'base64',
      });

      if (fileContent.content) {
        let dataUrl = fileContent.content;
        if (!dataUrl.startsWith('data:')) {
          const ext = filePath.toLowerCase().split('.').pop() || 'png';
          const mimeTypes: Record<string, string> = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp',
          };
          const mime = mimeTypes[ext] || 'image/png';
          dataUrl = `data:${mime};base64,${dataUrl}`;
        }
        ctx.log('info', `[AI] Read image file: ${filePath}`);
        return dataUrl;
      }
    } catch (err) {
      ctx.log('error', `[AI] Failed to read image file ${filePath}: ${err}`);
    }
    return null;
  };

  // Process inputs - can be array or single value
  const inputItems = Array.isArray(input) ? input : (input ? [input] : []);

  for (const item of inputItems) {
    if (typeof item !== 'string' || !item) continue;

    if (item.startsWith('data:image')) {
      // Already base64 data URL
      imageInputs.push(item);
    } else if (isImagePath(item)) {
      // Local file path - need to read it
      const dataUrl = await readImageFile(item);
      if (dataUrl) {
        imageInputs.push(dataUrl);
      }
    } else if (isImageUrl(item)) {
      // HTTP/HTTPS image URL - fetch and convert to base64
      const dataUrl = await fetchImageUrl(item);
      if (dataUrl) {
        imageInputs.push(dataUrl);
      }
    } else {
      // Non-image string input - append to prompt
      finalPrompt = finalPrompt ? `${finalPrompt}\n\n${item}` : item;
    }
  }

  const hasImages = imageInputs.length > 0;

  // Handle chunk references - resolve them to actual content
  if (chunkRefs && chunkRefs.length > 0) {
    ctx.log('info', `[AI] Processing ${chunkRefs.length} chunk reference(s)...`);
    // For now, just note that chunks exist - RAG handling would go here
  }

  // Auto-resolve chunk references in prompt
  const chunkRef = parseChunkRef(finalPrompt);
  if (chunkRef && ctx.tauri) {
    ctx.log('info', `[AI] Reading chunk ${chunkRef.index + 1}/${chunkRef.total} from file...`);
    try {
      finalPrompt = await readChunkContent(chunkRef);
      ctx.log('info', `[AI] Loaded chunk content (${finalPrompt.length} chars)`);
    } catch (error) {
      ctx.onNodeStatus?.(nodeId, 'error');
      const errMsg = error instanceof Error ? error.message : 'Unknown error';
      ctx.log('error', `[AI] Failed to read chunk: ${errMsg}`);
      throw new Error(`Failed to read file chunk: ${errMsg}`);
    }
  }

  // Check for file reference (large file without chunking - reject)
  const fileRef = parseFileRef(finalPrompt);
  if (fileRef) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const sizeMB = (fileRef.size / 1024 / 1024).toFixed(2);
    ctx.log('error', `[AI] Cannot send large file (${sizeMB} MB) directly to AI.`);
    throw new Error(
      `File is too large (${sizeMB} MB) to send to AI directly. ` +
      `Please use a Text Chunker node to split it into smaller pieces first.`
    );
  }

  // Resolve API key
  const apiKey = resolveApiKey(apiKeyConstant);

  if (!endpoint) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('No endpoint configured for AI request');
  }

  ctx.log('info', `[AI] Calling ${model} at ${endpoint}...`);

  try {
    // Determine format based on endpoint
    const isAnthropic = endpoint.includes('anthropic') || endpoint.includes('claude');

    let headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    let body: Record<string, unknown>;

    // Build messages array
    const messages: Array<{ role: string; content: string | Array<unknown> }> = [];

    // Build system prompt with history context if available
    let effectiveSystemPrompt = systemPrompt;

    // Add message history (previous turns in the conversation)
    // History can be a string (from loop history) or an array of message objects
    if (messageHistory) {
      if (typeof messageHistory === 'string' && messageHistory.length > 0) {
        // History is a string - append to system prompt for context
        const histStr = messageHistory.length > 5000
          ? '...(earlier truncated)...\n' + messageHistory.slice(-5000)
          : messageHistory;
        effectiveSystemPrompt = effectiveSystemPrompt
          ? `${effectiveSystemPrompt}\n\n## Previous Actions\n${histStr}`
          : `## Previous Actions\n${histStr}`;
        ctx.log('info', `[AI] Added history context to system prompt (${messageHistory.length} chars)`);
      } else if (Array.isArray(messageHistory) && messageHistory.length > 0) {
        // History is an array of message objects - add as actual conversation turns
        for (const msg of messageHistory) {
          if (msg.role && msg.content) {
            messages.push({ role: msg.role, content: msg.content });
          }
        }
        ctx.log('info', `[AI] Added ${messageHistory.length} history message(s)`);
      }
    }

    if (effectiveSystemPrompt) {
      messages.push({ role: 'system', content: effectiveSystemPrompt });
    }

    // Handle vision - include images if requested
    if (includeImages && hasImages) {
      // Build multimodal message with text and images
      const userContent: Array<{ type: string; text?: string; image_url?: { url: string; detail?: string } }> = [];

      if (finalPrompt) {
        userContent.push({ type: 'text', text: finalPrompt });
      }

      // Add all images, resizing each one with per-node overrides
      for (const imageData of imageInputs) {
        const resizedImage = await resizeImageIfNeeded(imageData, maxImageDimensionOverride, maxImageSizeKBOverride);
        userContent.push({
          type: 'image_url',
          image_url: {
            url: resizedImage,
            detail: visionDetail || 'auto'
          }
        });
      }

      ctx.log('info', `[AI] Including ${imageInputs.length} image(s) in request`);
      messages.push({ role: 'user', content: userContent });
    } else {
      messages.push({ role: 'user', content: finalPrompt });
    }

    const useCustomTemplate = typeof customBodyTemplate === 'string' && customBodyTemplate.trim() !== '';
    if (useCustomTemplate) {
      // "Custom" provider: build the request entirely from the user's body
      // template + headers. Vars: {{prompt}}/{{input}} {{system}} {{model}}
      // {{apiKey}} are JSON-escaped + quoted; {{messagesJsonRaw}} inserts the raw
      // OpenAI messages array; append "Raw" to any var for unescaped substitution.
      const tplVars: Record<string, string> = {
        prompt: finalPrompt,
        // Alias for the documented NodeSpec/{{input}} convention (and what
        // core-service provides), so an OAIY Desktop service marked apiFormat:'openai'
        // with a {{input}}-based body renders identically on the ai_llm path.
        input: finalPrompt,
        system: effectiveSystemPrompt || '',
        model: model || '',
        apiKey: apiKey || '',
        messagesJson: JSON.stringify(messages),
      };
      if (customHeaders && customHeaders.trim() !== '') {
        try {
          headers = JSON.parse(renderTemplate(customHeaders, tplVars, false)) as Record<string, string>;
        } catch (e) {
          ctx.log('warn', `[AI] Custom Headers JSON invalid; using defaults: ${e}`);
        }
      }
      if (apiKey && !('Authorization' in headers) && !('authorization' in headers)) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
      const renderedBody = renderTemplate(customBodyTemplate, tplVars, true);
      try {
        body = JSON.parse(renderedBody) as Record<string, unknown>;
      } catch (e) {
        throw new Error(`[AI] Custom body template did not render to valid JSON: ${e instanceof Error ? e.message : String(e)}. Rendered: ${renderedBody.slice(0, 300)}`);
      }
    } else if (isAnthropic) {
      // Anthropic Messages API format
      headers['x-api-key'] = apiKey;
      headers['anthropic-version'] = '2023-06-01';
      body = {
        model: model,
        max_tokens: maxTokens || 4096,
        messages: messages.filter(m => m.role !== 'system'), // Anthropic uses system differently
      };
      if (systemPrompt) {
        body.system = systemPrompt;
      }
      if (temperature > 0) {
        body.temperature = temperature;
      }
    } else {
      // OpenAI format (default)
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
      body = {
        model: model,
        messages: messages,
        stream: false,
      };
      if (maxTokens > 0) {
        body.max_tokens = maxTokens;
      }
      if (temperature > 0) {
        body.temperature = temperature;
      }
      if (responseFormat === 'json') {
        body.response_format = { type: 'json_object' };
      }
      // For local LLMs (Ollama, LM Studio)
      if (endpoint.includes('localhost') || endpoint.includes('127.0.0.1')) {
        body.options = { num_ctx: 32768 };
      }
      // Pass enable_thinking for models that support reasoning mode (e.g. Qwen 3.5)
      if (enableThinking) {
        body.enable_thinking = true;
      }
    }

    ctx.log('info', `[AI] Sending request...`);

    // Check for abort before sending request
    if (ctx.abortSignal?.aborted) {
      ctx.log('info', '[AI] Aborted by user before sending request');
      ctx.onNodeStatus?.(nodeId, 'completed');
      return '__ABORT__';
    }

    // Claude-as-AI mode: yield for external AI response instead of calling the API
    if (ctx.useClaudeForAI && ctx.yieldForAI) {
      ctx.log('info', `[AI] Claude-as-AI mode: yielding for external response`);

      // Build history array from messageHistory if it's an array
      const historyArray = Array.isArray(messageHistory)
        ? messageHistory.filter(m => m.role && m.content).map(m => ({
            role: m.role as 'user' | 'assistant',
            content: m.content as string
          }))
        : undefined;

      // Yield and wait for external response
      const externalResponse = await ctx.yieldForAI({
        nodeId,
        systemPrompt: effectiveSystemPrompt || '',
        userPrompt: finalPrompt,
        images: includeImages && hasImages ? imageInputs : undefined,
        history: historyArray,
      });

      ctx.log('info', `[AI] Received external response (${externalResponse.length} chars)`);

      // Same think-tag strip as the cloud path — Claude-as-AI relays
      // through this branch and a downstream Reasoner-style fallback
      // shouldn't leak its scratchpad either.
      const cleaned = stripThinkTags(externalResponse);

      // Stream the response if callback available
      if (ctx.onStreamToken && streaming) {
        ctx.onStreamToken(nodeId, cleaned);
      }

      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[AI] Chat completed (via Claude-as-AI)`);
      return cleaned;
    }

    // Calling this desktop's own gateway needs its bearer: the route is
    // privileged and a native caller has no Origin to be trusted by. Never
    // overrides a key the node itself configured.
    const localBearer = localServerBearer(endpoint);
    if (localBearer && !('Authorization' in headers) && !('authorization' in headers)) {
      headers['Authorization'] = `Bearer ${localBearer}`;
    }

    // Retry loop for 503 (service still loading) with progress logging
    const maxRetries = 60; // Up to ~5 minutes of retrying
    const retryDelayMs = 5000;
    let response: Response | undefined;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (ctx.abortSignal?.aborted) {
        return '__ABORT__';
      }
      response = await ctx.secureFetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        nodeId,
        purpose: 'AI/LLM chat request',
      });

      if (response.status !== 503) break;

      const errorText = await response.text();
      ctx.log('info', `[AI] Service loading: ${errorText.substring(0, 200)} (retrying in ${retryDelayMs / 1000}s...)`);
      ctx.onNodeStatus?.(nodeId, 'running');
      await new Promise(resolve => setTimeout(resolve, retryDelayMs));
    }

    if (!response || !response.ok) {
      const errorText = response ? await response.text() : 'No response';
      throw new Error(`HTTP ${response?.status || 0}: ${errorText.substring(0, 200)}`);
    }

    const responseText = await response.text();
    // Parse JSON, but tolerate a non-JSON body on the custom path (some
    // bespoke endpoints return raw text). Standard OpenAI/Anthropic paths
    // still require JSON, so surface a clear error there.
    let data: any = null;
    let jsonOk = true;
    try {
      data = JSON.parse(responseText);
    } catch {
      jsonOk = false;
    }
    if (!jsonOk && !useCustomTemplate) {
      throw new Error(`[AI] Expected a JSON response but got non-JSON: ${responseText.slice(0, 200)}`);
    }

    // Extract content based on format
    let content: string;
    if (useCustomTemplate) {
      if (!jsonOk) {
        // Endpoint returned raw text — use it verbatim.
        content = responseText;
      } else if (typeof customResponsePath === 'string' && customResponsePath.trim() !== '') {
        const extracted = extractJsonPath(data, customResponsePath);
        content = typeof extracted === 'string'
          ? extracted
          : (extracted === null || extracted === undefined ? JSON.stringify(data) : JSON.stringify(extracted));
      } else {
        // Custom template, no explicit path — try common shapes, else raw JSON.
        content = data?.choices?.[0]?.message?.content ?? data?.response ?? data?.text ?? JSON.stringify(data);
      }
    } else if (isAnthropic) {
      content = data.content?.[0]?.text || JSON.stringify(data);
    } else {
      content = data.choices?.[0]?.message?.content || data.response || JSON.stringify(data);
    }

    ctx.log('info', `[AI] Response received (${content.length} chars)`);

    // Strip reasoning-model `<think>...</think>` blocks before anything
    // downstream sees the text — Qwen 3.5 thinking, DeepSeek R1, and
    // cousins all interleave scratchpad reasoning that should never
    // reach the user-facing output. Cloud + local paths both go through
    // this so the behaviour is provider-agnostic.
    const cleaned = stripThinkTags(content);

    // Stream the response if callback available
    if (ctx.onStreamToken && streaming) {
      ctx.onStreamToken(nodeId, cleaned);
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[AI] Chat completed`);
    return cleaned;

  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';

    // Check for abort
    if (error instanceof Error && (error.name === 'AbortError' || errMsg.includes('aborted'))) {
      return '__ABORT__';
    }

    const userMessage = getUserFriendlyError(errMsg, 'AI chat');
    ctx.log('error', `[AI] ${userMessage}`);
    throw new Error(userMessage);
  }
}

/**
 * Execute vision request (legacy - for backwards compatibility).
 *
 * `mmprojPath` is the optional companion projector .gguf for the
 * Internal (OAIY Local) provider's vision-conditioned chat path. Empty
 * string means "text only"; required when an image input is wired and
 * the provider is oaiy-local. Ignored by every cloud provider — they
 * embed images directly in the request body.
 */
async function vision(
  prompt: string,
  imageData: string,
  systemPrompt: string,
  model: string,
  nodeId: string,
  format: string,
  endpoint: string,
  apiKey: string,
  imageFormat: string,
  contextLength: number,
  maxTokens: number,
  maxImageDimension: number = 0,
  maxImageSizeKB: number = 0,
  mmprojPath: string = '',
  videoInput: unknown = null,
  videoFrames: number = 8
): Promise<string> {
  // Extract N evenly-spaced frames from the connected video (if any)
  // and feed them as additional image inputs. The frames join the
  // single-image `imageData` path through the same chat() helper so
  // vision-capable backends (today: oaiy-local with a mmproj that
  // supports multi-image splice — MiniCPM-V 4.6 once the projector
  // forward lands; cloud providers always) see them as ordinary
  // image inputs. Empty when no video is connected.
  const extractedFramePaths = await extractVideoFrames(videoInput, videoFrames);
  // Combine the image input (single value or array) with the
  // extracted frame paths. The chat() helper handles arrays already.
  const combinedInput: unknown[] = [];
  if (imageData != null && imageData !== '') {
    if (Array.isArray(imageData)) {
      combinedInput.push(...(imageData as unknown[]));
    } else {
      combinedInput.push(imageData);
    }
  }
  combinedInput.push(...extractedFramePaths);
  const finalInput: unknown = combinedInput.length <= 1 ? (combinedInput[0] ?? '') : combinedInput;
  return chat(
    systemPrompt,
    prompt,
    finalInput,
    endpoint,
    model,
    '', // apiKeyConstant not used, apiKey passed directly
    false, // streaming
    maxTokens,
    0.7, // temperature
    'text', // responseFormat
    imageFormat !== 'none' || extractedFramePaths.length > 0, // includeImages
    'auto', // visionDetail
    nodeId,
    [], // chunkRefs
    '', // messageHistory
    maxImageDimension,
    maxImageSizeKB,
    false, // enableThinking
    mmprojPath
  );
}

/**
 * Extract `numFrames` evenly-spaced frames from a video input and
 * return their on-disk paths. Uses an offscreen `<video>` + `<canvas>`
 * to seek + grab each frame, then writes each as a PNG to a temp dir
 * via the filesystem plugin. Returns `[]` when no video is connected,
 * the runtime isn't Tauri, or the video can't be loaded — fail-open
 * so missing video doesn't break a normal chat.
 *
 * `videoInput` accepts:
 *   - a file path string (from input_video node),
 *   - a `file://` URL,
 *   - an `http(s)://` URL (browser can stream + seek directly),
 *   - an object with `{ path | outputPath | video }` (from
 *     diffusion_video node output — it aliases path under all three).
 *
 * The output diffusion_video produces an MP4; this helper handles
 * those without needing ffmpeg because the renderer's HTML5 video
 * element decodes MP4 natively via the system codec.
 */
async function extractVideoFrames(videoInput: unknown, numFrames: number): Promise<string[]> {
  if (!videoInput) return [];
  const n = Math.max(1, Math.min(128, Math.floor(numFrames || 0) || 8));

  // Resolve the input to a URL we can hand to the <video> element.
  let videoUrl: string | null = null;
  let videoPath: string | null = null;
  if (typeof videoInput === 'object') {
    const obj = videoInput as { path?: string; outputPath?: string; video?: string };
    videoPath = obj.path || obj.outputPath || obj.video || null;
  } else if (typeof videoInput === 'string') {
    videoPath = videoInput;
  }
  if (!videoPath || typeof videoPath !== 'string' || videoPath.length === 0) return [];

  if (videoPath.startsWith('http://') || videoPath.startsWith('https://') ||
      videoPath.startsWith('blob:') || videoPath.startsWith('data:')) {
    videoUrl = videoPath;
  } else if (videoPath.startsWith('file://')) {
    videoUrl = videoPath;
  } else if (/^[a-zA-Z]:[\\/]/.test(videoPath) || videoPath.startsWith('/') || videoPath.startsWith('\\\\')) {
    // Windows / Unix absolute path — wrap in file:// for the <video>
    // element. The renderer's asset protocol whitelist needs to allow
    // this path; the Tauri app config already does so for the oaiy
    // models / temp dirs.
    videoUrl = `file:///${videoPath.replace(/\\/g, '/').replace(/^\/+/, '')}`;
  } else {
    return [];
  }

  if (!ctx?.tauri) {
    ctx?.log?.('warn', '[AI] video frame extraction skipped — Tauri runtime unavailable');
    return [];
  }

  // Offscreen <video> + <canvas>. We seek to evenly-spaced timestamps
  // and snapshot. If anything fails, return what we have.
  const video = document.createElement('video');
  video.crossOrigin = 'anonymous';
  video.muted = true;
  video.preload = 'auto';
  video.src = videoUrl;
  try {
    await new Promise<void>((resolve, reject) => {
      const onLoaded = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('video load failed')); };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoaded);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('loadedmetadata', onLoaded);
      video.addEventListener('error', onError);
    });
  } catch (err) {
    ctx?.log?.('warn', `[AI] video metadata load failed: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
  const duration = isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
  if (duration <= 0) {
    ctx?.log?.('warn', '[AI] video duration unknown, skipping frame extraction');
    return [];
  }

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w <= 0 || h <= 0) return [];
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const cctx = canvas.getContext('2d');
  if (!cctx) return [];

  // Evenly-spaced timestamps. Avoid t=0 and t=duration exactly — the
  // first/last frame can be a black frame on some encoders.
  const timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = duration * ((i + 0.5) / n);
    timestamps.push(Math.min(t, Math.max(0, duration - 0.05)));
  }

  const tempDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_temp_dir');
  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const out: string[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    try {
      await new Promise<void>((resolve, reject) => {
        const onSeeked = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(new Error('seek failed')); };
        const cleanup = () => {
          video.removeEventListener('seeked', onSeeked);
          video.removeEventListener('error', onError);
        };
        video.addEventListener('seeked', onSeeked);
        video.addEventListener('error', onError);
        video.currentTime = timestamps[i];
      });
      cctx.drawImage(video, 0, 0, w, h);
      const dataUrl = canvas.toDataURL('image/png');
      const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
      const framePath = `${tempDir}/oaiy-llm-frame-${stamp}-${String(i).padStart(3, '0')}.png`;
      await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
        path: framePath,
        content: base64,
        contentType: 'base64',
        createDirs: true,
      });
      out.push(framePath);
    } catch (err) {
      ctx?.log?.('warn', `[AI] frame ${i} extract failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  video.src = '';
  ctx?.log?.('info', `[AI] extracted ${out.length}/${n} frames from video (${duration.toFixed(2)}s, ${w}x${h})`);
  return out;
}

/**
 * Execute custom API request
 */
async function request(
  body: string,
  endpoint: string,
  apiKey: string,
  nodeId: string
): Promise<string> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[AI] Aborted by user before custom request');
    return '__ABORT__';
  }

  // Validate inputs before proceeding
  validateEndpoint(endpoint, 'Custom API request');
  validateContentLength(body, 'Request body');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[AI] Custom request to ${endpoint}...`);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    // Use secureFetch instead of fetch for consistent security checks
    const response = await ctx.secureFetch(endpoint, {
      method: 'POST',
      headers,
      body,
      nodeId,
      purpose: 'Custom API request',
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 200)}`);
    }

    const responseText = await response.text();
    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[AI] Custom request complete (${responseText.length} chars)`);
    return responseText;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';

    if (error instanceof Error && (error.name === 'AbortError' || errMsg.includes('aborted'))) {
      return '__ABORT__';
    }

    const userMessage = getUserFriendlyError(errMsg, 'custom request');
    ctx.log('error', `[AI] ${userMessage}`);
    throw new Error(userMessage);
  }
}

/**
 * Core AI Runtime Module
 */
  return {
    chat,
    vision,
    request,
  };
}

const CoreAIRuntime: RuntimeModule = {
  name: 'AI',
  createMethods: createAIMethods,
  methods: {},
  streaming: {
    chat: true,
    vision: true,
  },
  async cleanup(): Promise<void> {},
};

export default CoreAIRuntime;
