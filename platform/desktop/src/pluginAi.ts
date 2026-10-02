import { PluginCommandError } from './pluginRpc';

export interface PluginAiCompletionRequest {
  requestId: string;
  sourceId: string;
  prompt: string;
  maxOutputChars: number;
}
export interface PluginAiCompletion {
  requestId: string;
  sourceId: string;
  text: string;
  model?: string;
}
const refusal = (code: string, message: string) => new PluginCommandError({ code, message });
const safeId = (value: unknown): value is string => typeof value === 'string' && /^[a-zA-Z0-9_.:-]{1,96}$/.test(value);

export function completionRequest(value: unknown): PluginAiCompletionRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).some((key) => !['requestId', 'sourceId', 'prompt', 'maxOutputChars'].includes(key))) {
    throw refusal('invalid_request', 'Supply a bounded text completion request.');
  }
  const request = value as PluginAiCompletionRequest;
  // A scalar can occupy two UTF-16 units. Check the cheap string bound before
  // allocating either its code-point array or its UTF-8 encoding.
  if (!safeId(request.requestId) || typeof request.sourceId !== 'string' || !/^provider:[a-z0-9-]{1,64}$/.test(request.sourceId) ||
      typeof request.prompt !== 'string' || request.prompt.length > 24_000 || !request.prompt.trim() || Array.from(request.prompt).length > 12_000 ||
      new TextEncoder().encode(request.prompt).length > 48 * 1024 || !Number.isInteger(request.maxOutputChars) ||
      request.maxOutputChars < 1 || request.maxOutputChars > 4096) {
    throw refusal('invalid_request', 'Use a fresh request ID, configured provider source, prompt up to 12,000 characters and output limit 1..4096.');
  }
  return { requestId: request.requestId, sourceId: request.sourceId, prompt: request.prompt, maxOutputChars: request.maxOutputChars };
}

/** One mounted screen owns its requests. Replacing it aborts its fetches and
 * asks the native host to cancel the same IDs; a late result cannot resolve. */
export class PluginAiSession {
  private readonly pending = new Map<string, AbortController>();
  private disposed = false;
  private readonly base: string;
  constructor(pluginId: string, apiBase: string) {
    this.base = `${apiBase}/api/plugins/${encodeURIComponent(pluginId)}/ai`;
  }
  async sources(): Promise<unknown[]> {
    const result = await this.request('sources', undefined, undefined) as { sources?: unknown };
    if (!result || !Array.isArray(result.sources) || result.sources.length > 65) throw refusal('invalid_completion', 'The desktop returned an invalid source catalogue.');
    return result.sources;
  }
  async complete(value: unknown): Promise<PluginAiCompletion> {
    const request = completionRequest(value);
    if (this.disposed) throw refusal('request_cancelled', 'The plugin screen was closed.');
    if (this.pending.size || this.pending.has(request.requestId)) throw refusal('completion_busy', 'An AI completion is already running in this screen.');
    const controller = new AbortController();
    this.pending.set(request.requestId, controller);
    try {
      const response = await this.request('complete', request, controller.signal);
      if (this.disposed || controller.signal.aborted || this.pending.get(request.requestId) !== controller) {
        throw refusal('request_cancelled', 'The AI completion was cancelled.');
      }
      if (!response || typeof response !== 'object' || Array.isArray(response)) throw refusal('invalid_completion', 'The desktop returned an invalid AI completion.');
      const result = response as PluginAiCompletion;
      if (Object.keys(result).some((key) => !['requestId', 'sourceId', 'text', 'model'].includes(key)) ||
          result.requestId !== request.requestId || result.sourceId !== request.sourceId || typeof result.text !== 'string' ||
          !result.text.trim() || Array.from(result.text).length > request.maxOutputChars ||
          (result.model !== undefined && (typeof result.model !== 'string' || result.model.length > 256))) {
        throw refusal('invalid_completion', 'The desktop returned an invalid AI completion.');
      }
      return result;
    } catch (error) {
      if (controller.signal.aborted || this.disposed) throw refusal('request_cancelled', 'The AI completion was cancelled.');
      // A network failure may leave a native request alive. The host's own
      // deadline remains authoritative, and this asks it to stop immediately.
      void this.request('cancel', { requestId:request.requestId }, undefined).catch(() => {});
      throw error;
    } finally {
      if (this.pending.get(request.requestId) === controller) this.pending.delete(request.requestId);
    }
  }
  async cancel(requestId: unknown): Promise<{ requestId: string; cancelled: boolean }> {
    if (!safeId(requestId)) throw refusal('invalid_request', 'Supply the request ID to cancel.');
    this.pending.get(requestId)?.abort();
    const result = await this.request('cancel', { requestId }, undefined) as { requestId?: unknown; cancelled?: unknown };
    if (!result || result.requestId !== requestId || typeof result.cancelled !== 'boolean') throw refusal('invalid_completion', 'The desktop returned an invalid cancellation acknowledgement.');
    return { requestId, cancelled: result.cancelled };
  }
  dispose(): void {
    this.disposed = true;
    for (const [requestId, controller] of this.pending) {
      controller.abort();
      void this.request('cancel', { requestId }, undefined).catch(() => {});
    }
    this.pending.clear();
  }
  private async request(method: string, body: unknown, signal: AbortSignal | undefined): Promise<unknown> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    const abortError = () => signal?.aborted
      ? refusal('request_cancelled', 'The AI completion was cancelled.')
      : refusal('completion_timeout', 'The desktop AI request exceeded its response deadline.');
    const fence = () => { if (controller.signal.aborted) throw abortError(); };
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    const cancelReader = () => {
      try { void reader?.cancel().catch(() => {}); } catch { /* best effort */ }
    };
    let rejectAbort!: () => void;
    const aborted = new Promise<never>((_resolve, reject) => {
      rejectAbort = () => { cancelReader(); reject(abortError()); };
      controller.signal.addEventListener('abort', rejectAbort, { once: true });
    });
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    const timeout = window.setTimeout(abort, 19_000);
    // Racing the whole read prevents an abort-ignoring transport or stalled
    // body from keeping the screen's promise and concurrency slot alive.
    const operation = (async () => {
      fence();
      const response = await fetch(`${this.base}/${method}`, {
        method: body === undefined ? 'GET' : 'POST', headers: { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
      });
      if (controller.signal.aborted) {
        // A fetch that completes after its cancellation never gets read.
        void response.body?.cancel().catch(() => {});
        fence();
      }
      if (!response.body) throw refusal('invalid_completion', 'The desktop returned an invalid AI response.');
      reader = response.body.getReader();
      const limit = 64 * 1024;
      const declaredLength = response.headers.get('Content-Length');
      if (declaredLength && /^\d+$/.test(declaredLength) && Number(declaredLength) > limit) {
        cancelReader();
        throw refusal('invalid_completion', 'The desktop response exceeded its size limit.');
      }
      const decoder = new TextDecoder('utf-8', { fatal: true });
      let bytes = 0;
      let text = '';
      try {
        while (true) {
          fence();
          const chunk = await reader.read();
          fence();
          if (chunk.done) break;
          bytes += chunk.value.byteLength;
          if (bytes > limit) throw refusal('invalid_completion', 'The desktop response exceeded its size limit.');
          try { text += decoder.decode(chunk.value, { stream: true }); }
          catch { throw refusal('invalid_completion', 'The desktop returned an invalid AI response.'); }
        }
        try { text += decoder.decode(); }
        catch { throw refusal('invalid_completion', 'The desktop returned an invalid AI response.'); }
      } catch (error) {
        cancelReader();
        throw error;
      } finally {
        // A nonconforming reader may still have a read pending after abort.
        try { reader.releaseLock(); } catch { /* cancellation was requested */ }
      }
      fence();
      let value: unknown;
      try { value = JSON.parse(text); } catch { throw refusal('invalid_completion', 'The desktop returned an invalid AI response.'); }
      if (!response.ok) {
        const error = (value as { error?: { code?: unknown; message?: unknown } } | null)?.error;
        if (error && typeof error.code === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$/.test(error.code) && typeof error.message === 'string') {
          throw refusal(error.code, error.message.slice(0, 1024).replace(/[\u0000-\u001f\u007f]/g, ' ').trim() || 'The AI request failed.');
        }
        throw refusal('upstream_error', 'The AI request failed.');
      }
      fence();
      return value;
    })();
    try {
      return await Promise.race([operation, aborted]);
    } catch (error) {
      fence();
      throw error;
    } finally {
      window.clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
      controller.signal.removeEventListener('abort', rejectAbort);
    }
  }
}
