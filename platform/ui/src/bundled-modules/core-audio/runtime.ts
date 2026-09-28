/**
 * Core Audio Module Runtime
 *
 * Provides text-to-speech functionality using external API.
 */

import type { RuntimeContext, RuntimeModule } from 'oaiy-core';
import type { RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createAudioMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

// ============================================
// Helper Functions
// ============================================

/**
 * Extract port number from a URL string
 * @param url The URL to extract port from (e.g., "http://127.0.0.1:8765/tts")
 * @returns The port number or null if not found
 */
function extractPortFromUrl(url: string): number | null {
    const match = url.match(/:(\d+)/);
    return match ? parseInt(match[1], 10) : null;
}

/**
 * Resolve service URL by checking if the service is running and auto-starting if needed.
 * Uses ensure_service_ready_by_port for fully dynamic service lookup from services folder.
 * @param _serviceKey Unused - kept for backwards compatibility
 * @param userUrl User-provided URL (contains the port for service lookup)
 * @returns The resolved URL to use
 */
async function resolveServiceUrl(
    _serviceKey: string,
    userUrl?: string | null
): Promise<string> {
    // User URL is required - it contains the port that identifies the service
    if (!userUrl) {
        throw new Error(`No API URL provided for service`);
    }

    // Extract port from URL for dynamic service lookup
    const port = extractPortFromUrl(userUrl);
    if (!port) {
        // No port found, return URL as-is (could be external service)
        return userUrl;
    }

    // Extract the endpoint path for reconstructing the URL
    const urlMatch = userUrl.match(/^(https?:\/\/[^\/]+)(\/.*)?$/);
    const endpoint = urlMatch?.[2] || '';

    // Try to ensure service is ready (auto-start if needed)
    if (ctx.tauri) {
        try {
            interface EnsureServiceResult {
                success: boolean;
                port?: number;
                error?: string;
                already_running: boolean;
            }

            ctx.log('info', `[Audio] Ensuring service on port ${port} is ready...`);

            // Use dynamic port-based lookup - finds service from services folder
            const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready_by_port', {
                port,
            });

            if (result.success && result.port) {
                if (!result.already_running) {
                    ctx.log('info', `[Audio] Service on port ${port} auto-started`);
                }
                return `http://127.0.0.1:${result.port}${endpoint}`;
            } else if (result.error) {
                ctx.log('warn', `[Audio] Service on port ${port} failed to start: ${result.error}`);
            }
        } catch (err) {
            // ensure_service_ready_by_port not available (older backend), continue with URL as-is
            ctx.log('info', `[Audio] Dynamic service lookup not available`);
        }
    }

    return userUrl;
}

/**
 * Try to auto-start a service using ensure_service_ready_by_port (fully dynamic lookup)
 * @param port The port number to find and start the service
 * @returns Object with success status and optional port
 */
async function ensureServiceReadyByPort(port: number): Promise<{ success: boolean; port?: number }> {
    if (!ctx.tauri) return { success: false };

    try {
        interface EnsureServiceResult {
            success: boolean;
            port?: number;
            error?: string;
            already_running: boolean;
        }

        ctx.log('info', `[Audio] Ensuring service on port ${port} is ready...`);

        // Use dynamic port-based lookup - finds service from services folder
        const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready_by_port', {
            port,
        });

        if (result.success && result.port) {
            if (!result.already_running) {
                ctx.log('info', `[Audio] Service on port ${port} auto-started`);
            }
            return { success: true, port: result.port };
        } else if (result.error) {
            ctx.log('warn', `[Audio] Service on port ${port} failed to start: ${result.error}`);
        }
    } catch {
        // ensure_service_ready_by_port not available (older backend)
        ctx.log('info', `[Audio] Dynamic service lookup not available`);
    }

    return { success: false };
}

/**
 * Helper to create a fetch with timeout (compatible with older environments).
 * Routes through `ctx.fetch` so the runtime's network policy/SSRF guard
 * applies. Falls back to global `fetch` only if the runtime context hasn't
 * been initialised yet (e.g. during pre-init self-tests).
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs: number = 5000): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const doFetch: typeof fetch =
        (ctx?.fetch as typeof fetch | undefined) ?? (globalThis.fetch as typeof fetch);

    try {
        const response = await doFetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        throw error;
    }
}

/**
 * Check if a service is running and healthy, auto-starting if needed
 * @param apiUrl The API endpoint URL (port is extracted for dynamic lookup)
 * @param serviceName Human-readable service name for error messages
 * @returns void - throws error if service is not available
 */
async function checkServiceAvailable(apiUrl: string, serviceName: string): Promise<void> {
    // Extract base URL (remove endpoint path like /tts, /generate, etc.)
    const baseUrl = apiUrl.replace(/\/[^/]+$/, '');

    // First, check if service is already running (quick health check)
    try {
        const quickCheck = await fetchWithTimeout(`${baseUrl}/health`, { method: 'GET' }, 3000);
        if (quickCheck.ok) {
            ctx.log('info', `[Audio] Service already running at ${baseUrl}`);
            return;  // Service is already healthy, no need to auto-start
        }
    } catch {
        // Service not responding, try to auto-start
        ctx.log('info', `[Audio] Service not responding, attempting auto-start...`);
    }

    // Extract port from URL for dynamic service lookup
    const port = extractPortFromUrl(apiUrl);
    if (port) {
        // Try to auto-start the service using dynamic port lookup
        await ensureServiceReadyByPort(port);
    }

    // Health check with retries after auto-start (service needs time to load models)
    const maxRetries = 60;  // Up to 2 minutes of retries
    const retryDelay = 2000;  // 2 seconds between retries

    ctx.log('info', `[Audio] Starting health check loop for ${baseUrl}/health`);

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            ctx.log('info', `[Audio] Health check attempt ${attempt + 1}: fetching ${baseUrl}/health`);
            const healthCheck = await fetchWithTimeout(`${baseUrl}/health`, { method: 'GET' }, 5000);
            ctx.log('info', `[Audio] Health check response: status=${healthCheck.status}, ok=${healthCheck.ok}`);
            if (healthCheck.ok) {
                ctx.log('info', `[Audio] Service ready after ${attempt + 1} health check(s)`);
                return;  // Service is healthy
            }
            const healthData = await healthCheck.json().catch(() => ({})) as { missing?: string[] };
            ctx.log('info', `[Audio] Health data: ${JSON.stringify(healthData)}`);
            if (healthData.missing?.length) {
                throw new Error(`${serviceName} service is missing dependencies: ${healthData.missing.join(', ')}. Please install them and restart the service.`);
            }
        } catch (error) {
            // Re-throw if it's our custom error about missing dependencies
            if (error instanceof Error && error.message.includes('missing dependencies')) {
                throw error;
            }
            // Check for abort
            if (ctx.abortSignal?.aborted) {
                throw new Error('Operation aborted by user');
            }
            // Log the actual error
            const errMsg = error instanceof Error ? error.message : String(error);
            ctx.log('info', `[Audio] Health check attempt ${attempt + 1} failed: ${errMsg}`);
            // Log progress every 10 attempts
            if (attempt > 0 && attempt % 10 === 0) {
                ctx.log('info', `[Audio] Waiting for service to be ready (attempt ${attempt + 1}/${maxRetries})...`);
            }
        }
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, retryDelay));
    }

    // All retries exhausted
    throw new Error(`${serviceName} service is not running. Please start it from the Services panel (gear icon > Services).`);
}

// ============================================
// TTS Functions
// ============================================

/**
 * Options bag for the new `text_to_speech` node compiler — supports
 * both the OAIY-local DramaBox plugin (Internal) and any HTTP TTS
 * endpoint driven by a request/response template (External).
 *
 * The old `textToSpeech(...)` positional function below stays in place
 * so previously-compiled flows keep running unchanged. New flows are
 * compiled against `textToSpeechV2(opts)`.
 */
interface TextToSpeechV2Opts {
    nodeId: string;
    text: string;
    audioPromptPath?: string;
    descriptionInput?: string;

    provider?: 'internal' | 'external';

    // Internal (plugin-tts / DramaBox) ---
    internalModel?: string;
    internalSteps?: number;
    internalCfgScale?: number;
    internalStgScale?: number;
    internalDurationMultiplier?: number;
    internalSeed?: number;
    internalStreaming?: boolean;
    internalFastMode?: boolean;

    // External (HTTP, templated) ---
    externalUrl?: string;
    externalMethod?: 'POST' | 'GET';
    externalHeaders?: string;
    externalApiKeyConstant?: string;
    externalBodyTemplate?: string;
    externalResponseType?: 'file_path' | 'url' | 'base64_wav' | 'base64_mp3' | 'binary_wav' | 'binary_mp3';
    externalResponsePath?: string;

    // Shared ---
    description?: string;
    speaker?: string;
    language?: string;
    outputFormat?: 'wav' | 'mp3';
    filename?: string;
}

async function textToSpeechV2(
    opts: TextToSpeechV2Opts,
): Promise<{ audio: string; path: string; durationMs?: number }> {
    const {
        nodeId,
        text,
        audioPromptPath,
        descriptionInput,
        provider = 'internal',
    } = opts;

    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[TTS] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }
    if (!text || text.trim() === '') {
        throw new Error('No text provided for TTS');
    }
    if (!ctx.tauri) throw new Error('Tauri not available');

    ctx.onNodeStatus?.(nodeId, 'running');

    const effectiveDescription = (descriptionInput && descriptionInput.trim() !== '')
        ? descriptionInput
        : (opts.description || '');

    try {
        if (provider === 'internal') {
            return await runTtsInternal(opts, text, audioPromptPath, effectiveDescription);
        }
        return await runTtsExternal(opts, text, audioPromptPath, effectiveDescription);
    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[TTS] Failed: ${errMsg}`);
        throw error;
    }
}

/**
 * Internal mode — dispatch to the `plugin-tts` Tauri crate (DramaBox).
 * In Phase 1 the native `tts_generate` returns a clear not-yet-wired
 * error; the wiring below means the node fails with a useful message
 * instead of looking broken.
 */
async function runTtsInternal(
    opts: TextToSpeechV2Opts,
    text: string,
    audioPromptPath: string | undefined,
    description: string,
): Promise<{ audio: string; path: string; durationMs?: number }> {
    if (!ctx.tauri) throw new Error('Tauri not available');
    const modelId = opts.internalModel || 'dramabox-v1';
    ctx.log('info', `[TTS:internal] dramabox via plugin-tts, model=${modelId}`);

    // Cross-plugin VRAM hygiene: DramaBox DiT (BF16) + Gemma 12B GGUF
    // text encoder + audio VAE peak around 18-22 GB on a 32 GB card.
    // If plugin-llm has a GGUF resident (4-12 GB) or plugin-diffusion
    // is still holding LTX 22B INT8 (24 GB), the TTS load OOMs. Evict
    // both before our load. Silent no-op if the plugins aren't installed.
    try {
        const evicted = await ctx.tauri.invoke<number>('plugin:oaiy-llm|llm_unload_all');
        if (typeof evicted === 'number' && evicted > 0) {
            ctx.log('info', `[TTS:internal] evicted ${evicted} loaded LLM(s) to free VRAM`);
        }
    } catch {
        // plugin-llm absent — fine.
    }
    try {
        const evicted = await ctx.tauri.invoke<number>('plugin:oaiy-diffusion|diffusion_unload_all');
        if (typeof evicted === 'number' && evicted > 0) {
            ctx.log('info', `[TTS:internal] evicted ${evicted} loaded diffusion model(s) to free VRAM`);
        }
    } catch {
        // plugin-diffusion absent — fine.
    }

    // Load (no-op if already loaded). Errors with a precise feature-gate
    // message in Phase 1; in Phase 2 this actually warms the DiT.
    await ctx.tauri.invoke('plugin:oaiy-tts|tts_load_model', {
        args: { modelId, backend: 'auto' },
    });

    const result = await ctx.tauri.invoke<{
        path: string;
        audioUrl?: string;
        modelId: string;
        backend: string;
        sampleRate: number;
        channels: number;
        durationMs: number;
    }>('plugin:oaiy-tts|tts_generate', {
        args: {
            modelId,
            text,
            voiceRefPath: audioPromptPath && audioPromptPath.trim() !== '' ? audioPromptPath : undefined,
            description: description || undefined,
            cfgScale: opts.internalCfgScale,
            stgScale: opts.internalStgScale,
            durationMultiplier: opts.internalDurationMultiplier,
            steps: opts.internalSteps,
            seed: opts.internalSeed && opts.internalSeed > 0 ? opts.internalSeed : undefined,
            outputFormat: opts.outputFormat || 'wav',
            outputFilename: opts.filename || undefined,
            stream: opts.internalStreaming === true,
            noBwe: opts.internalFastMode === true,
        },
    });

    let audioUrl = result.audioUrl || result.path;
    if (!result.audioUrl) {
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: result.path });
        } catch (e) {
            ctx.log('warn', `[TTS:internal] failed to get media URL: ${e}`);
        }
    }
    ctx.onNodeStatus?.(opts.nodeId, 'completed');
    return { audio: audioUrl, path: result.path, durationMs: result.durationMs };
}

/**
 * External mode — render the body template, POST, extract the audio
 * location via JSONPath, fetch + persist the audio, return media URL.
 *
 * Templating: `{{var}}` is JSON-escaped + quoted; `{{varRaw}}` is the
 * raw unquoted value. Empty / missing values render as `""` for the
 * quoted form (so the JSON stays well-formed) and as the empty string
 * for the raw form.
 */
async function runTtsExternal(
    opts: TextToSpeechV2Opts,
    text: string,
    audioPromptPath: string | undefined,
    description: string,
): Promise<{ audio: string; path: string; durationMs?: number }> {
    if (!ctx.tauri) throw new Error('Tauri not available');
    const url = (opts.externalUrl || '').trim();
    if (!url) throw new Error('External TTS: missing endpoint URL');
    const method = (opts.externalMethod || 'POST').toUpperCase();
    const responseType = opts.externalResponseType || 'file_path';
    const responsePath = opts.externalResponsePath || '';
    const bodyTemplate = opts.externalBodyTemplate || '{}';

    // Resolve API key from OAIY Secrets if requested.
    let apiKey = '';
    if (opts.externalApiKeyConstant && opts.externalApiKeyConstant.trim() !== '') {
        try {
            apiKey = await ctx.tauri.invoke<string>(
                'plugin:oaiy-secrets|get_secret',
                { name: opts.externalApiKeyConstant.trim() },
            );
        } catch (e) {
            ctx.log('warn', `[TTS:external] secret '${opts.externalApiKeyConstant}' not found: ${e}`);
        }
    }

    const vars: Record<string, string> = {
        text,
        voiceRefPath: audioPromptPath || '',
        description,
        speaker: opts.speaker || '',
        language: opts.language && opts.language !== 'Auto' ? opts.language : '',
        outputFormat: opts.outputFormat || 'wav',
        filename: opts.filename || 'tts_output',
        apiKey,
    };

    const renderedUrl = renderTemplate(url, vars, { jsonEscape: false });
    const renderedHeaders = parseHeaders(opts.externalHeaders, vars);
    const renderedBody = method === 'GET' ? undefined : renderTemplate(bodyTemplate, vars, { jsonEscape: true });

    ctx.log('info', `[TTS:external] ${method} ${renderedUrl} (template ${bodyTemplate.length} chars → ${(renderedBody || '').length})`);

    const response = await ctx.fetch(renderedUrl, {
        method,
        headers: renderedHeaders,
        body: renderedBody,
    });
    if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(`TTS external ${response.status} ${response.statusText}: ${errText.slice(0, 200)}`);
    }

    // Resolve where the audio bytes are coming from + an extension.
    const ext = responseType.endsWith('_mp3') ? 'mp3'
        : responseType.endsWith('_wav') ? 'wav'
        : (opts.outputFormat || 'wav');
    const baseName = opts.filename || 'tts_output';
    const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    const outputPath = `${appDataDir}/output/${baseName}_${Date.now()}.${ext}`;

    let audioBytes: Uint8Array | null = null;

    if (responseType === 'binary_wav' || responseType === 'binary_mp3') {
        const blob = await response.blob();
        audioBytes = new Uint8Array(await blob.arrayBuffer());
    } else {
        const json = await response.json();
        const value = extractJsonPath(json, responsePath);
        if (value === null || value === undefined) {
            throw new Error(`TTS external: responsePath '${responsePath}' not found in response`);
        }
        if (responseType === 'file_path') {
            // Server-side file path — copy to our output dir so the
            // media server can serve it via a token URL.
            const srcPath = String(value);
            try {
                await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
                    source: srcPath,
                    destination: outputPath,
                    createDirs: true,
                });
            } catch (copyErr) {
                ctx.log('warn', `[TTS:external] copy failed, returning original path: ${copyErr}`);
                let audioUrl: string = srcPath;
                try {
                    audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: srcPath });
                } catch { /* fall through */ }
                ctx.onNodeStatus?.(opts.nodeId, 'completed');
                return { audio: audioUrl, path: srcPath };
            }
        } else if (responseType === 'url') {
            // Fetch the URL ourselves.
            const audioResp = await ctx.fetch(String(value));
            if (!audioResp.ok) {
                throw new Error(`TTS external: GET ${value} returned ${audioResp.status}`);
            }
            audioBytes = new Uint8Array(await (await audioResp.blob()).arrayBuffer());
        } else if (responseType === 'base64_wav' || responseType === 'base64_mp3') {
            audioBytes = base64ToBytes(String(value));
        }
    }

    if (audioBytes) {
        // Write the bytes to disk.
        const b64 = uint8ArrayToBase64(audioBytes);
        await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
            path: outputPath,
            content: b64,
            contentType: 'base64',
            createDirs: true,
        });
    }

    let audioUrl = outputPath;
    try {
        audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
    } catch (e) {
        ctx.log('warn', `[TTS:external] failed to get media URL: ${e}`);
    }

    ctx.onNodeStatus?.(opts.nodeId, 'completed');
    return { audio: audioUrl, path: outputPath };
}

/**
 * Render `{{var}}` (JSON-escaped + quoted) and `{{varRaw}}` (unquoted
 * literal) placeholders in `template`. Unknown placeholders are left
 * as-is so a typo doesn't silently produce a malformed request.
 */
function renderTemplate(
    template: string,
    vars: Record<string, string>,
    { jsonEscape }: { jsonEscape: boolean },
): string {
    return template.replace(/\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}/g, (full, name) => {
        const rawSuffix = 'Raw';
        const isRaw = name.endsWith(rawSuffix);
        const baseKey = isRaw ? name.slice(0, -rawSuffix.length) : name;
        if (!Object.prototype.hasOwnProperty.call(vars, baseKey)) {
            return full;
        }
        const value = vars[baseKey] ?? '';
        if (!jsonEscape) {
            return value;
        }
        return isRaw ? value : JSON.stringify(value);
    });
}

function parseHeaders(headersStr: string | undefined, vars: Record<string, string>): Record<string, string> {
    if (!headersStr || headersStr.trim() === '') return { 'Content-Type': 'application/json' };
    try {
        const rendered = renderTemplate(headersStr, vars, { jsonEscape: false });
        const parsed = JSON.parse(rendered);
        if (parsed && typeof parsed === 'object') {
            return parsed as Record<string, string>;
        }
    } catch (e) {
        ctx.log('warn', `[TTS:external] failed to parse Headers JSON, falling back to Content-Type=application/json: ${e}`);
    }
    return { 'Content-Type': 'application/json' };
}

/**
 * Minimal JSONPath-like extractor. Supports dot navigation and
 * bracket-indexed array access: `data.items[0].url`, `[2].name`,
 * `audio_path`. Returns `undefined` if any segment misses.
 */
function extractJsonPath(root: unknown, path: string): unknown {
    if (!path || path.trim() === '') return root;
    const segs: (string | number)[] = [];
    const re = /([^.\[\]]+)|\[(\d+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(path)) !== null) {
        if (m[1] !== undefined) segs.push(m[1]);
        else if (m[2] !== undefined) segs.push(Number(m[2]));
    }
    let cur: unknown = root;
    for (const s of segs) {
        if (cur === null || cur === undefined) return undefined;
        if (typeof s === 'number') {
            if (!Array.isArray(cur)) return undefined;
            cur = cur[s];
        } else {
            if (typeof cur !== 'object') return undefined;
            cur = (cur as Record<string, unknown>)[s];
        }
    }
    return cur;
}

function base64ToBytes(b64: string): Uint8Array {
    // Strip data: URL prefix if present.
    const comma = b64.indexOf(',');
    const payload = comma > 0 && b64.startsWith('data:') ? b64.slice(comma + 1) : b64;
    if (typeof atob === 'function') {
        const bin = atob(payload);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }
    // Node-ish fallback (unlikely in the renderer).
    const buf = (globalThis as unknown as { Buffer?: { from: (s: string, enc: string) => { length: number; [i: number]: number } } }).Buffer?.from(payload, 'base64');
    if (!buf) throw new Error('No base64 decoder available');
    const out = new Uint8Array(buf.length);
    for (let i = 0; i < buf.length; i++) out[i] = buf[i] as number;
    return out;
}

/**
 * Generate speech from text using external TTS API
 */
async function textToSpeech(
    text: string,
    apiUrl: string,
    responseFormat: string,
    description: string,
    outputFormat: string,
    filename: string,
    nodeId: string,
    audioPromptPath?: string,
    speaker?: string,
    language?: string
): Promise<{ audio: string; path: string; durationMs?: number }> {
    // Check for abort before starting
    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[TTS] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }

    ctx.log('info', `[TTS] Generating speech via API: ${apiUrl} (format: ${responseFormat})`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!text || text.trim() === '') {
            throw new Error('No text provided for TTS');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Check if TTS service is running
        // Detect service name from port for better error messages
        const serviceName = apiUrl.includes(':8772') ? 'Qwen3 TTS' : 'Chatterbox TTS';
        await checkServiceAvailable(apiUrl, serviceName);

        // Get output path for saving the result
        const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');

        // Handle JSON response format (e.g., Chatterbox server, Qwen3-TTS)
        if (responseFormat === 'json') {
            // Build JSON request body
            const requestBody: Record<string, unknown> = { text };
            if (description) requestBody.description = description;
            if (speaker) requestBody.speaker = speaker;
            if (language && language !== 'Auto') requestBody.language = language;
            if (audioPromptPath && audioPromptPath.trim() !== '') {
                requestBody.audio_prompt_path = audioPromptPath;
                ctx.log('info', `[TTS] Using audio prompt: ${audioPromptPath}`);
            }

            ctx.log('info', `[TTS] Sending JSON request with text: "${text.substring(0, 50)}..."`);

            const response = await ctx.fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`TTS API error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const result = await response.json() as {
                success?: boolean;
                audio_path: string;
                duration_ms?: number;
                sample_rate?: number;
                message?: string;
            };

            if (result.success === false) {
                throw new Error(`TTS failed: ${result.message || 'Unknown error'}`);
            }

            ctx.log('info', `[TTS] Generated audio: ${result.audio_path}`);

            // Copy the file to our output directory
            const outputPath = `${appDataDir}/output/${filename || 'tts'}_${Date.now()}.wav`;

            try {
                await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
                    source: result.audio_path,
                    destination: outputPath,
                    createDirs: true
                });
                ctx.log('info', `[TTS] Copied audio to ${outputPath}`);
            } catch (copyError) {
                // If copy fails, use the original path directly
                ctx.log('warn', `[TTS] Copy failed, using original path: ${copyError}`);
                const audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: result.audio_path });
                ctx.onNodeStatus?.(nodeId, 'completed');
                return { audio: audioUrl, path: result.audio_path, durationMs: result.duration_ms };
            }

            // Get media server URL for playback
            let audioUrl = outputPath;
            try {
                audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
            } catch (e) {
                ctx.log('error', `[TTS] Failed to get media URL: ${e}`);
            }

            ctx.onNodeStatus?.(nodeId, 'completed');
            return { audio: audioUrl, path: outputPath, durationMs: result.duration_ms };
        }

        // Handle PCM16 stream and audio file formats
        const fileExt = responseFormat === 'audio_file' ? (outputFormat || 'mp3') : 'wav';
        const outputPath = `${appDataDir}/output/${filename || 'tts'}_${Date.now()}.${fileExt}`;

        // Make request to external API using FormData
        const formData = new FormData();
        formData.append('text', text);
        if (description) formData.append('description', description);
        if (speaker) formData.append('speaker', speaker);
        if (language && language !== 'Auto') formData.append('language', language);
        if (audioPromptPath && audioPromptPath.trim() !== '') {
            formData.append('audio_prompt_path', audioPromptPath);
        }

        ctx.log('info', `[TTS] Sending request with text: "${text.substring(0, 50)}..."`);

        const response = await ctx.fetch(apiUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`TTS API error: ${response.status} ${response.statusText}`);
        }

        // Handle response based on format
        if (responseFormat === 'audio_file') {
            // Direct audio file download - save the response as-is
            const audioBlob = await response.blob();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioData = new Uint8Array(arrayBuffer);

            ctx.log('info', `[TTS] Received ${audioData.length} bytes of audio file`);

            // Convert to base64 and save (using chunked conversion to avoid stack overflow)
            const base64Audio = uint8ArrayToBase64(audioData);
            await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
                path: outputPath,
                content: base64Audio,
                contentType: 'base64',
                createDirs: true
            });
        } else {
            // PCM16 streaming - read chunks and convert to WAV
            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('No response body from TTS API');
            }

            const chunks: Uint8Array[] = [];
            while (true) {
                // Check for abort during streaming
                if (ctx.abortSignal?.aborted) {
                    ctx.log('info', '[TTS] Aborted by user during streaming');
                    throw new Error('Operation aborted by user');
                }
                const { done, value } = await reader.read();
                if (done) break;
                if (value) chunks.push(value);
            }

            // Combine all chunks into single buffer
            const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const pcmData = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                pcmData.set(chunk, offset);
                offset += chunk.length;
            }

            ctx.log('info', `[TTS] Received ${pcmData.length} bytes of PCM audio`);

            // Convert PCM to WAV - PCM is 16-bit, 24kHz, mono
            const sampleRate = 24000;
            const numChannels = 1;
            const bitsPerSample = 16;
            const wavBuffer = createWavBuffer(pcmData, sampleRate, numChannels, bitsPerSample);

            // Convert WAV buffer to base64 for write_file command (using chunked conversion to avoid stack overflow)
            const base64Audio = uint8ArrayToBase64(wavBuffer);
            await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
                path: outputPath,
                content: base64Audio,
                contentType: 'base64',
                createDirs: true
            });
        }

        ctx.log('info', `[TTS] Audio saved to ${outputPath}`);

        // Get media server URL for playback in browser
        let audioUrl = outputPath;
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
            ctx.log('info', `[TTS] Media URL: ${audioUrl}`);
        } catch (e) {
            ctx.log('error', `[TTS] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');

        return { audio: audioUrl, path: outputPath };
    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[TTS] Failed: ${errMsg}`);
        throw error;
    }
}

/**
 * Helper function to create WAV buffer from PCM data
 */
function createWavBuffer(pcmData: Uint8Array, sampleRate: number, numChannels: number, bitsPerSample: number): Uint8Array {
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = pcmData.length;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, totalSize - 8, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // chunk size
    view.setUint16(20, 1, true); // audio format (PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Copy PCM data
    const wavBuffer = new Uint8Array(buffer);
    wavBuffer.set(pcmData, headerSize);

    return wavBuffer;
}

function writeString(view: DataView, offset: number, str: string): void {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

/**
 * Convert Uint8Array to base64 string without stack overflow
 * (avoids spread operator which fails on large arrays)
 */
function uint8ArrayToBase64(data: Uint8Array): string {
    const CHUNK_SIZE = 8192;
    let result = '';
    for (let i = 0; i < data.length; i += CHUNK_SIZE) {
        const chunk = data.subarray(i, Math.min(i + CHUNK_SIZE, data.length));
        result += String.fromCharCode.apply(null, chunk as unknown as number[]);
    }
    return btoa(result);
}

/**
 * Save audio file to specified location
 */
async function saveAudio(
    audioPath: string,
    filename: string,
    directory: string,
    format: string,
    overwrite: boolean,
    nodeId: string
): Promise<{ path: string }> {
    ctx.log('info', `[Audio] Saving audio: ${audioPath}`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!audioPath) {
            throw new Error('No audio path provided');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Determine output directory
        let outputDir = directory;
        if (!outputDir) {
            const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
            outputDir = `${appDataDir}/output`;
        }

        // Construct output path
        const ext = format || 'wav';
        const outputPath = `${outputDir}/${filename || 'audio_output'}.${ext}`;

        ctx.log('info', `[Audio] Copying to: ${outputPath}`);

        // Copy the file using native_copy_file
        await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
            source: audioPath,
            destination: outputPath,
            createDirs: true
        });

        ctx.log('info', `[Audio] Audio saved to ${outputPath}`);

        // Get media server URL for playback in browser
        let audioUrl = outputPath;
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
            ctx.log('info', `[Audio] Media URL: ${audioUrl}`);
        } catch (e) {
            ctx.log('error', `[Audio] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');

        return { path: audioUrl };
    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[Audio] Save failed: ${errMsg}`);
        throw error;
    }
}

/**
 * Settings for music generation services
 */
interface MusicGenSettings {
    // ACE-Step settings
    inferSteps?: number;
    guidanceScale?: number;
    // HeartMuLa settings
    temperature?: number;
    topk?: number;
    cfgScale?: number;
}

/**
 * Generate music using ACE-Step 1.5 (task-based API)
 * ACE-Step 1.5 uses an async task-based API:
 * - POST /release_task to submit generation task
 * - POST /query_result to poll for results
 */
async function generateMusicAceStep15(
    prompt: string,
    lyrics: string,
    baseUrl: string,
    duration: number,
    settings: MusicGenSettings,
    seed: number,
    nodeId: string
): Promise<{ audioPath: string }> {
    // Build request body for ACE-Step 1.5
    // API expects: prompt (caption), lyrics, audio_duration, infer_step, guidance_scale, seed
    // When lyrics are provided, enable "thinking" mode to use 5Hz LM for vocal generation
    const hasLyrics = lyrics && lyrics.trim() !== '' &&
        !lyrics.trim().toLowerCase().match(/^\[?(inst|instrumental)\]?$/);

    const requestBody: Record<string, unknown> = {
        prompt,  // This is the caption/style description
        lyrics: lyrics || '',
        audio_duration: duration,
        infer_step: settings.inferSteps ?? 8,  // v1.5 turbo uses 8 steps by default
        guidance_scale: settings.guidanceScale ?? 15.0,
        // Enable thinking mode when lyrics are provided - this uses the 5Hz LM to generate
        // audio codes that incorporate the vocals/lyrics
        thinking: !!hasLyrics,
        vocal_language: 'en',  // Default to English for now
    };

    if (hasLyrics) {
        ctx.log('info', `[Music Gen] Lyrics detected, enabling thinking mode for vocal generation`);
    }

    // Only include seed if it's not -1 (random)
    if (seed >= 0) {
        requestBody.seed = seed;
        requestBody.use_random_seed = false;
    } else {
        requestBody.use_random_seed = true;
    }

    ctx.log('info', `[Music Gen] Submitting task to ACE-Step 1.5: ${baseUrl}/release_task`);

    // Submit the generation task
    const submitResponse = await ctx.fetch(`${baseUrl}/release_task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
    });

    if (!submitResponse.ok) {
        const errorText = await submitResponse.text();
        throw new Error(`ACE-Step task submission failed: ${submitResponse.status} ${submitResponse.statusText} - ${errorText}`);
    }

    // Response is wrapped: { data: { task_id, status, ... }, code: 200 }
    const submitWrapper = await submitResponse.json() as {
        data?: { task_id?: string; status?: string };
        code?: number;
        error?: string;
    };

    const submitResult = submitWrapper.data || submitWrapper;
    const taskId = (submitResult as { task_id?: string }).task_id;

    if (!taskId) {
        throw new Error(`ACE-Step task submission failed: No task_id returned - ${JSON.stringify(submitWrapper)}`);
    }

    ctx.log('info', `[Music Gen] Task submitted: ${taskId}, polling for results...`);

    // Poll for results with exponential backoff
    const maxAttempts = 120;  // Max ~10 minutes of polling
    const initialDelay = 2000;  // Start with 2 second delay
    const maxDelay = 10000;  // Max 10 second delay
    let delay = initialDelay;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        // Check for abort during polling
        if (ctx.abortSignal?.aborted) {
            ctx.log('info', '[Music Gen] Aborted by user during polling');
            throw new Error('Operation aborted by user');
        }

        await new Promise(resolve => setTimeout(resolve, delay));

        // Query for results - API expects task_id_list (not task_ids)
        const queryResponse = await ctx.fetch(`${baseUrl}/query_result`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id_list: [taskId] })
        });

        if (!queryResponse.ok) {
            ctx.log('warn', `[Music Gen] Query failed (attempt ${attempt + 1}): ${queryResponse.status}`);
            delay = Math.min(delay * 1.5, maxDelay);
            continue;
        }

        // Response format: { data: [{ task_id, result: "json string", status }], code: 200 }
        const queryWrapper = await queryResponse.json() as {
            data?: Array<{ task_id: string; result: string; status: number }>;
            code?: number;
            error?: string;
        };

        const taskResults = queryWrapper.data || [];

        // Find our task result
        const taskResult = taskResults.find(r => r.task_id === taskId);
        if (taskResult) {
            // status: 0 = pending, 1 = succeeded, 2 = failed
            if (taskResult.status === 1 && taskResult.result) {
                // Parse the nested JSON result
                try {
                    const audioResults = JSON.parse(taskResult.result) as Array<{
                        file?: string;
                        status?: number;
                        metas?: { bpm?: number; keyscale?: string; duration?: number };
                    }>;

                    if (audioResults.length > 0 && audioResults[0].file) {
                        // Extract file path from URL format: /v1/audio?path=URL_ENCODED_PATH
                        let filePath = audioResults[0].file;
                        if (filePath.includes('?path=')) {
                            const pathParam = filePath.split('?path=')[1];
                            filePath = decodeURIComponent(pathParam);
                        }
                        ctx.log('info', `[Music Gen] Task completed: ${filePath}`);
                        return { audioPath: filePath };
                    }
                } catch (parseErr) {
                    ctx.log('warn', `[Music Gen] Failed to parse result JSON: ${parseErr}`);
                }
            } else if (taskResult.status === 2) {
                throw new Error('Music generation failed on server');
            }
            // Status 0 means still processing, continue polling
        }

        ctx.log('info', `[Music Gen] Task still processing (attempt ${attempt + 1}/${maxAttempts})...`);
        ctx.onNodeStatus?.(nodeId, 'running');  // Keep status updated

        // Increase delay with exponential backoff
        delay = Math.min(delay * 1.2, maxDelay);
    }

    throw new Error('Music generation timed out after maximum polling attempts');
}

/**
 * Generate music using ACE-Step or HeartMuLa API
 */
async function generateMusic(
    prompt: string,
    lyrics: string,
    apiUrl: string,
    duration: number,
    service: 'ace-step' | 'heartmula',
    settings: MusicGenSettings,
    seed: number,
    filename: string,
    nodeId: string
): Promise<{ audio: string; path: string }> {
    // Check for abort before starting
    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[Music Gen] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }

    const serviceName = service === 'heartmula' ? 'HeartMuLa Music' : 'ACE-Step Music';
    ctx.log('info', `[Music Gen] Generating music via ${serviceName}: ${apiUrl}`);
    ctx.log('info', `[Music Gen] Prompt: "${prompt.substring(0, 50)}...", Duration: ${duration}s`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!prompt || prompt.trim() === '') {
            throw new Error('No prompt provided for music generation');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Extract base URL (remove endpoint path like /generate)
        const baseUrl = apiUrl.replace(/\/[^/]+$/, '');

        // Check if the music service is running
        await checkServiceAvailable(apiUrl, serviceName);

        let audioPath: string;

        // Use ACE-Step 1.5 task-based API for ace-step service
        if (service === 'ace-step') {
            const result = await generateMusicAceStep15(
                prompt,
                lyrics,
                baseUrl,
                duration,
                settings,
                seed,
                nodeId
            );
            audioPath = result.audioPath;
        } else {
            // HeartMuLa uses synchronous API
            const requestBody: Record<string, unknown> = {
                prompt,
                lyrics: lyrics || '',
                duration,
                temperature: settings.temperature ?? 1.0,
                topk: settings.topk ?? 50,
                cfg_scale: settings.cfgScale ?? 1.5,
            };

            // Only include seed if it's not -1 (random)
            if (seed >= 0) {
                requestBody.seed = seed;
            }

            ctx.log('info', `[Music Gen] Sending request to ${serviceName}...`);

            const response = await ctx.fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Music Gen API error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const result = await response.json() as {
                success?: boolean;
                audio_path: string;
                duration_seconds?: number;
                sample_rate?: number;
                message?: string;
            };

            if (result.success === false) {
                throw new Error(`Music generation failed: ${result.message || 'Unknown error'}`);
            }

            audioPath = result.audio_path;
        }

        ctx.log('info', `[Music Gen] Generated audio: ${audioPath}`);

        // Get app data dir for output
        const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
        // Determine file extension from source path
        const sourceExt = audioPath.split('.').pop() || 'mp3';
        const outputPath = `${appDataDir}/output/${filename || 'music'}_${Date.now()}.${sourceExt}`;

        // Copy the file to our output directory
        try {
            await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
                source: audioPath,
                destination: outputPath,
                createDirs: true
            });
            ctx.log('info', `[Music Gen] Copied audio to ${outputPath}`);
        } catch (copyError) {
            // If copy fails, use the original path directly
            ctx.log('warn', `[Music Gen] Copy failed, using original path: ${copyError}`);
            const audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: audioPath });
            ctx.onNodeStatus?.(nodeId, 'completed');
            return { audio: audioUrl, path: audioPath };
        }

        // Get media server URL for playback
        let audioUrl = outputPath;
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
        } catch (e) {
            ctx.log('error', `[Music Gen] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');
        return { audio: audioUrl, path: outputPath };

    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[Music Gen] Failed: ${errMsg}`);
        throw error;
    }
}

/**
 * Download and extract audio from video URLs using yt-dlp
 */
async function grabAudio(
    url: string,
    apiUrl: string,
    start: number,
    end: number | null,
    sampleRate: number,
    mono: boolean,
    nodeId: string
): Promise<{ audio: string; path: string; duration: number }> {
    // Check for abort before starting
    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[Audio DL] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }

    ctx.log('info', `[Audio DL] Downloading audio from: ${url}`);
    ctx.log('info', `[Audio DL] Time range: ${start}s - ${end ?? 'end'}, Sample rate: ${sampleRate}, Mono: ${mono}`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!url || url.trim() === '') {
            throw new Error('No URL provided for audio download');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Check if service is running (auto-starts if needed)
        await checkServiceAvailable(apiUrl, 'Audio Downloader');

        // Build request body
        const requestBody: Record<string, unknown> = {
            url,
            start: start || 0,
            sample_rate: sampleRate || 44100,
            mono: mono || false,
        };

        // Only include end if it's not null
        if (end !== null) {
            requestBody.end = end;
        }

        ctx.log('info', `[Audio DL] Sending request to ${apiUrl}...`);

        const response = await ctx.fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Audio download API error: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const result = await response.json() as {
            success?: boolean;
            audio_path: string;
            duration_seconds?: number;
            sample_rate?: number;
            video_id?: string;
            message?: string;
        };

        if (result.success === false) {
            throw new Error(`Audio download failed: ${result.message || 'Unknown error'}`);
        }

        ctx.log('info', `[Audio DL] Downloaded audio: ${result.audio_path} (${result.duration_seconds}s)`);

        // Get app data dir for output
        const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
        const outputPath = `${appDataDir}/output/audio_dl_${result.video_id || 'clip'}_${Date.now()}.wav`;

        // Copy the file to our output directory
        try {
            await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
                source: result.audio_path,
                destination: outputPath,
                createDirs: true
            });
            ctx.log('info', `[Audio DL] Copied audio to ${outputPath}`);
        } catch (copyError) {
            // If copy fails, use the original path directly
            ctx.log('warn', `[Audio DL] Copy failed, using original path: ${copyError}`);
            const audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: result.audio_path });
            ctx.onNodeStatus?.(nodeId, 'completed');
            return { audio: audioUrl, path: result.audio_path, duration: result.duration_seconds || 0 };
        }

        // Get media server URL for playback
        let audioUrl = outputPath;
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
        } catch (e) {
            ctx.log('error', `[Audio DL] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');
        return { audio: audioUrl, path: outputPath, duration: result.duration_seconds || 0 };

    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[Audio DL] Failed: ${errMsg}`);
        throw error;
    }
}

// ============================================
// Speech-to-Text Functions
// ============================================

interface TranscriptWord {
    word: string;
    start: number;
    end: number;
    score?: number;
}

interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    words?: TranscriptWord[];
    speaker?: string;
}

interface TranscriptionResult {
    success: boolean;
    language: string;
    duration: number;
    segments: TranscriptSegment[];
    text: string;
    word_count: number;
    message?: string;
}

/**
 * Transcribe audio/video to text using WhisperX API
 */
async function speechToText(
    mediaPath: string,
    apiUrl: string,
    language: string | null,
    enableWordTimestamps: boolean,
    enableDiarization: boolean,
    minSpeakers: number | null,
    maxSpeakers: number | null,
    startTime: number | null,
    endTime: number | null,
    nodeId: string,
    hfTokenConstant: string | null = null
): Promise<{
    text: string;
    segments: TranscriptSegment[];
    language: string;
    duration: number;
}> {
    // Check for abort before starting
    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[STT] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }

    ctx.log('info', `[STT] Transcribing: ${mediaPath}`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!mediaPath || mediaPath.trim() === '') {
            throw new Error('No media file provided for transcription');
        }

        // Resolve media path - could be a media URL or direct path
        let resolvedPath = mediaPath;

        // If it's a media URL, convert to file path
        if (mediaPath.startsWith('http://127.0.0.1:') && mediaPath.includes('/media/')) {
            if (ctx.tauri) {
                const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
                resolvedPath = mediaPath.replace(/http:\/\/127\.0\.0\.1:\d+\/media\/oaiy-output\//, `${appDataDir}/output/`);
            }
        }

        // If it's an object with a path property
        if (typeof mediaPath === 'object' && mediaPath !== null) {
            const mediaObj = mediaPath as { path?: string; audio?: string; video?: string };
            resolvedPath = mediaObj.path || mediaObj.audio || mediaObj.video || '';
        }

        ctx.log('info', `[STT] Resolved path: ${resolvedPath}`);

        // Check if WhisperX service is running
        await checkServiceAvailable(apiUrl, 'WhisperX STT');

        // Get HF token from constants if specified (for speaker diarization)
        let hfToken: string | null = null;
        if (hfTokenConstant && enableDiarization) {
            if (ctx.getConstant) {
                hfToken = ctx.getConstant(hfTokenConstant) || null;
            }
            if (!hfToken) {
                // Try module settings as fallback
                const settingToken = ctx.getModuleSetting?.(hfTokenConstant);
                if (typeof settingToken === 'string') {
                    hfToken = settingToken;
                }
            }
            if (hfToken) {
                ctx.log('info', `[STT] Using HuggingFace token from constant: ${hfTokenConstant}`);
            } else {
                ctx.log('warn', `[STT] HuggingFace token constant '${hfTokenConstant}' not found - diarization may fail`);
            }
        }

        // Build request body
        const requestBody: Record<string, unknown> = {
            audio_path: resolvedPath,
            enable_word_timestamps: enableWordTimestamps,
            enable_diarization: enableDiarization,
        };

        // Include HF token if available (for diarization)
        if (hfToken) {
            requestBody.hf_token = hfToken;
        }

        if (language && language.trim() !== '') {
            requestBody.language = language;
        }
        if (startTime !== null && startTime !== undefined) {
            requestBody.start_time = startTime;
        }
        if (endTime !== null && endTime !== undefined) {
            requestBody.end_time = endTime;
        }
        if (enableDiarization) {
            if (minSpeakers !== null && minSpeakers !== undefined) {
                requestBody.min_speakers = minSpeakers;
            }
            if (maxSpeakers !== null && maxSpeakers !== undefined) {
                requestBody.max_speakers = maxSpeakers;
            }
        }

        ctx.log('info', `[STT] Sending request to ${apiUrl}...`);

        const response = await ctx.fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`STT API error: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const result = await response.json() as TranscriptionResult;

        if (result.success === false) {
            throw new Error(`Transcription failed: ${result.message || 'Unknown error'}`);
        }

        ctx.log('info', `[STT] Transcription complete: ${result.segments?.length || 0} segments, ${result.word_count || 0} words`);
        ctx.log('info', `[STT] Detected language: ${result.language}, Duration: ${result.duration}s`);

        ctx.onNodeStatus?.(nodeId, 'completed');

        return {
            text: result.text || '',
            segments: result.segments || [],
            language: result.language || 'unknown',
            duration: result.duration || 0
        };

    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[STT] Failed: ${errMsg}`);
        throw error;
    }
}

/**
 * Concatenate multiple audio files into a single audio file
 */
async function appendAudio(
    audios: Array<string | { audio?: string; path?: string }>,
    filename: string,
    format: string,
    nodeId: string
): Promise<{ audio: string; path: string }> {
    ctx.log('info', `[Audio Append] Concatenating ${audios?.length || 0} audio files`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!audios || audios.length === 0) {
            throw new Error('No audio files provided');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Extract paths from audio objects
        const audioPaths: string[] = [];
        for (let i = 0; i < audios.length; i++) {
            const audio = audios[i];
            let audioPath = '';
            if (typeof audio === 'string') {
                audioPath = audio;
            } else if (audio && typeof audio === 'object') {
                audioPath = audio.path || audio.audio || '';
            }
            if (audioPath) {
                // Convert media URLs to file paths if needed
                if (audioPath.startsWith('http://127.0.0.1:') && audioPath.includes('/media/')) {
                    const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
                    audioPath = audioPath.replace(/http:\/\/127\.0\.0\.1:\d+\/media\/oaiy-output\//, `${appDataDir}/output/`);
                }
                audioPaths.push(audioPath);
            }
        }

        if (audioPaths.length === 0) {
            throw new Error('No valid audio paths found');
        }

        ctx.log('info', `[Audio Append] Processing ${audioPaths.length} audio files`);

        // Get output directory
        const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
        const outputDir = `${appDataDir}/output`;
        const ext = format || 'wav';
        const outputPath = `${outputDir}/${filename || 'concatenated_audio'}_${Date.now()}.${ext}`;

        // Create concat list file for FFmpeg
        const tempDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_temp_dir');
        const concatListPath = `${tempDir}/audio_concat_${Date.now()}.txt`;

        // Write concat list
        let concatContent = '';
        for (const audioPath of audioPaths) {
            // FFmpeg concat requires forward slashes and escaped single quotes
            const escapedPath = audioPath.replace(/\\/g, '/').replace(/'/g, "'\\''");
            concatContent += `file '${escapedPath}'\n`;
        }

        await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
            path: concatListPath,
            content: concatContent,
            contentType: 'text',
            createDirs: true
        });

        ctx.log('info', `[Audio Append] Concat list created: ${concatListPath}`);

        // Run FFmpeg concat
        const args = [
            '-f', 'concat',
            '-safe', '0',
            '-i', concatListPath,
            '-c', 'copy',
            '-y', outputPath
        ];

        const result = await ctx.tauri.invoke<{ code: number; stdout: string; stderr: string }>(
            'plugin:oaiy-filesystem|run_command',
            { command: 'ffmpeg', args, cwd: null }
        );

        // Cleanup temp file
        await ctx.tauri.invoke('plugin:oaiy-filesystem|delete_file', { path: concatListPath }).catch(() => {});

        if (result?.code !== 0) {
            ctx.log('error', `[Audio Append] FFmpeg stderr: ${result?.stderr}`);
            throw new Error(`FFmpeg concat failed: ${result?.stderr || 'Unknown error'}`);
        }

        ctx.log('info', `[Audio Append] Output saved to ${outputPath}`);

        // Get media URL for playback
        let audioUrl = outputPath;
        try {
            audioUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
        } catch (e) {
            ctx.log('error', `[Audio Append] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');
        return { audio: audioUrl, path: outputPath };

    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : String(error);
        ctx.log('error', `[Audio Append] Failed: ${errMsg}`);
        throw error;
    }
}

// ============================================
// Audio Duration Helper
// ============================================

/**
 * Get duration of an audio file using ffprobe
 * @param audioPath Path to the audio file (file path or media URL)
 * @returns Duration in milliseconds
 */
async function getAudioDuration(audioPath: string): Promise<number> {
    ctx.log('info', `[Audio] Getting duration for: ${audioPath}`);

    if (!audioPath) {
        ctx.log('warn', '[Audio] No audio path provided for duration check');
        return 0;
    }

    if (!ctx.tauri) {
        ctx.log('warn', '[Audio] Tauri not available for duration check');
        return 0;
    }

    try {
        // Convert media URL to file path if needed
        let resolvedPath = audioPath;
        if (audioPath.startsWith('http://127.0.0.1:') && audioPath.includes('/media/')) {
            const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
            // Extract the path after /media/oaiy-output/ or /media/
            const urlMatch = audioPath.match(/\/media\/oaiy-output\/(.+)$/) || audioPath.match(/\/media\/(.+)$/);
            if (urlMatch) {
                const relativePath = urlMatch[1];
                // Use forward slashes - works cross-platform (Windows, Linux, macOS)
                // The Tauri filesystem plugin handles path normalization
                if (audioPath.includes('/media/oaiy-output/')) {
                    resolvedPath = `${appDataDir}/output/${relativePath}`;
                } else {
                    resolvedPath = `${appDataDir}/${relativePath}`;
                }
            }
        }

        ctx.log('info', `[Audio] Resolved path for ffprobe: ${resolvedPath}`);

        // Use ffprobe to get duration
        const args = [
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            resolvedPath
        ];

        const result = await ctx.tauri.invoke<{ code: number; stdout: string; stderr: string }>(
            'plugin:oaiy-filesystem|run_command',
            { command: 'ffprobe', args, cwd: null }
        );

        if (result?.code === 0 && result.stdout) {
            const durationSec = parseFloat(result.stdout.trim());
            const durationMs = Math.round(durationSec * 1000);
            ctx.log('info', `[Audio] Duration: ${durationSec}s (${durationMs}ms)`);
            return durationMs;
        } else {
            ctx.log('warn', `[Audio] ffprobe failed: ${result?.stderr || 'Unknown error'}`);
            return 0;
        }
    } catch (error) {
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[Audio] Failed to get duration: ${errMsg}`);
        return 0;
    }
}

// ============================================
// Audio Fade Functions
// ============================================

/**
 * Apply audio fade effect to a video file
 * @param videoPath Path to the video file
 * @param fadeDuration Duration of fade in seconds
 * @param fadeType Type of fade curve: "exponential" or "linear"
 * @param fadeDirection Direction of fade: "out" (end) or "in" (start)
 * @param filename Output filename
 * @param nodeId Node ID for status updates
 * @returns Object with video URL and path
 */
async function fadeAudio(
    videoPath: string,
    fadeDuration: number,
    fadeType: 'exponential' | 'linear',
    fadeDirection: 'in' | 'out',
    filename: string,
    nodeId: string
): Promise<{ video: string; path: string }> {
    // Check for abort before starting
    if (ctx.abortSignal?.aborted) {
        ctx.log('info', '[Audio Fade] Aborted by user before starting');
        throw new Error('Operation aborted by user');
    }

    ctx.log('info', `[Audio Fade] Applying ${fadeType} fade ${fadeDirection} (${fadeDuration}s) to: ${videoPath}`);
    ctx.onNodeStatus?.(nodeId, 'running');

    try {
        if (!videoPath || videoPath.trim() === '') {
            throw new Error('No video path provided for audio fade');
        }

        if (!ctx.tauri) throw new Error('Tauri not available');

        // Resolve video path - could be a media URL or direct path
        let resolvedPath = videoPath;
        if (videoPath.startsWith('http://127.0.0.1:') && videoPath.includes('/media/')) {
            const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
            resolvedPath = videoPath.replace(/http:\/\/127\.0\.0\.1:\d+\/media\/oaiy-output\//, `${appDataDir}/output/`);
        }

        ctx.log('info', `[Audio Fade] Resolved path: ${resolvedPath}`);

        // Get video duration using ffprobe
        const probeArgs = [
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            resolvedPath
        ];

        const probeResult = await ctx.tauri.invoke<{ code: number; stdout: string; stderr: string }>(
            'plugin:oaiy-filesystem|run_command',
            { command: 'ffprobe', args: probeArgs, cwd: null }
        );

        if (probeResult?.code !== 0 || !probeResult.stdout) {
            throw new Error(`Failed to get video duration: ${probeResult?.stderr || 'Unknown error'}`);
        }

        const totalDuration = parseFloat(probeResult.stdout.trim());
        ctx.log('info', `[Audio Fade] Video duration: ${totalDuration}s`);

        // Ensure fade duration doesn't exceed video duration
        const actualFadeDuration = Math.min(fadeDuration, totalDuration);

        // Calculate fade start time for fade-out (at end of video)
        // For fade-in, it starts at 0
        const fadeStart = fadeDirection === 'out' ? Math.max(0, totalDuration - actualFadeDuration) : 0;

        // Map fadeType to FFmpeg curve
        // FFmpeg afade curves: tri (linear), exp (exponential), qua (quadratic), etc.
        const ffmpegCurve = fadeType === 'exponential' ? 'exp' : 'tri';

        // Build FFmpeg filter
        // afade=t=TYPE:st=START:d=DURATION:curve=CURVE
        const afadeFilter = `afade=t=${fadeDirection}:st=${fadeStart}:d=${actualFadeDuration}:curve=${ffmpegCurve}`;

        ctx.log('info', `[Audio Fade] Filter: ${afadeFilter}`);

        // Get output path
        const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
        const outputPath = `${appDataDir}/output/${filename || 'audio_faded'}_${Date.now()}.mp4`;

        // Run FFmpeg with audio fade filter
        // Copy video stream, apply filter to audio stream
        const ffmpegArgs = [
            '-i', resolvedPath,
            '-vcodec', 'copy',
            '-af', afadeFilter,
            '-y', outputPath
        ];

        ctx.log('info', `[Audio Fade] Running FFmpeg...`);

        const ffmpegResult = await ctx.tauri.invoke<{ code: number; stdout: string; stderr: string }>(
            'plugin:oaiy-filesystem|run_command',
            { command: 'ffmpeg', args: ffmpegArgs, cwd: null }
        );

        if (ffmpegResult?.code !== 0) {
            ctx.log('error', `[Audio Fade] FFmpeg stderr: ${ffmpegResult?.stderr}`);
            throw new Error(`FFmpeg audio fade failed: ${ffmpegResult?.stderr || 'Unknown error'}`);
        }

        ctx.log('info', `[Audio Fade] Output saved to ${outputPath}`);

        // Get media URL for playback
        let videoUrl = outputPath;
        try {
            videoUrl = await ctx.tauri.invoke<string>('get_media_url', { filePath: outputPath });
        } catch (e) {
            ctx.log('error', `[Audio Fade] Failed to get media URL: ${e}`);
        }

        ctx.onNodeStatus?.(nodeId, 'completed');
        return { video: videoUrl, path: outputPath };

    } catch (error) {
        ctx.onNodeStatus?.(nodeId, 'error');
        const errMsg = error instanceof Error ? error.message : 'Unknown error';
        ctx.log('error', `[Audio Fade] Failed: ${errMsg}`);
        throw error;
    }
}

// ============================================
// Runtime Module Export
// ============================================

    return {
        textToSpeech,
        textToSpeechV2,
        saveAudio,
        generateMusic,
        grabAudio,
        appendAudio,
        speechToText,
        resolveServiceUrl,
        getAudioDuration,
        fadeAudio,
    };
}

const CoreAudioRuntime: RuntimeModule = {
    name: 'Audio',
    createMethods: createAudioMethods,
    methods: {},
    async cleanup(): Promise<void> {},
};

export default CoreAudioRuntime;
