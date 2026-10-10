/**
 * The page side of the sandbox: start a Worker per program, answer its host
 * calls, and end it at the deadline.
 *
 * The Zipp module is compiled once and handed to every Worker, so a run
 * costs a Worker start and an engine, not a download or a compile.
 */
import PRELUDE from './guest/prelude.js?raw';
import SHELL from './guest/shell.js?raw';
import TOOLS from './guest/shell-tools.js?raw';
import { ReplyWriter, createChannelBuffer } from './channel';
import type { FromWorker, RunRequest, RunResult } from './protocol';

/** Answers a guest's host call. A throw becomes an exception in the guest. */
export interface HostHandler {
  call(kind: string, args: string[]): Promise<string>;
}

export interface RunOptions {
  timeoutMs: number;
  /** Stop: ends the Worker at once, like the deadline. */
  signal?: AbortSignal;
}

export interface RunOutcome {
  result?: RunResult;
  timedOut: boolean;
  /** The Worker died or could not start. */
  failure?: string;
  elapsedMs: number;
  hostCalls: number;
}

const MAX_HOST_CALLS = 100_000;

export function zippBase(): string {
  return new URL(`${import.meta.env.BASE_URL}zipp/`, location.href).href;
}

let compiled: Promise<WebAssembly.Module> | null = null;

/** The Zipp engine, compiled once per page. */
export function zippModule(): Promise<WebAssembly.Module> {
  if (!compiled) {
    const url = `${zippBase()}zipp_wasm_bg.wasm`;
    compiled = WebAssembly.compileStreaming(fetch(url)).catch(async () => {
      // A server that does not send application/wasm still gets a compile.
      const bytes = await (await fetch(url)).arrayBuffer();
      return WebAssembly.compile(bytes);
    });
    compiled.catch(() => {
      compiled = null;
    });
  }
  return compiled;
}

export function sandboxAvailable(): { ok: true } | { ok: false; reason: string } {
  // WebKit calls a page on a scheme of its own isolated and still gives it no SharedArrayBuffer (OAIY on a Mac serves
  // the page over http for it: the desktop's embed.rs).
  if (typeof SharedArrayBuffer === 'undefined' && globalThis.crossOriginIsolated) {
    return {
      ok: false,
      reason: 'the page is cross-origin isolated but has no SharedArrayBuffer, so the sandbox cannot block on shared memory (WebKit gives none to a page on a scheme of its own: serve it over http)',
    };
  }
  if (typeof SharedArrayBuffer === 'undefined' || !globalThis.crossOriginIsolated) {
    return {
      ok: false,
      reason: 'the page is not cross-origin isolated, so the sandbox cannot block on shared memory (serve with COOP same-origin and COEP credentialless, or reload once the service worker is installed)',
    };
  }
  return { ok: true };
}

export async function runInSandbox(request: RunRequest, handler: HostHandler, options: RunOptions): Promise<RunOutcome> {
  const started = performance.now();
  const available = sandboxAvailable();
  if (!available.ok) return { timedOut: false, failure: available.reason, elapsedMs: 0, hostCalls: 0 };
  const module = await zippModule();
  const sab = createChannelBuffer();
  const writer = new ReplyWriter(sab);
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'zipp-sandbox' });
  let hostCalls = 0;

  return new Promise<RunOutcome>((resolve) => {
    let settled = false;
    const finish = (outcome: Omit<RunOutcome, 'elapsedMs' | 'hostCalls'>) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.terminate();
      resolve({ ...outcome, elapsedMs: Math.round(performance.now() - started), hostCalls });
    };
    const timer = setTimeout(() => finish({ timedOut: true }), options.timeoutMs);
    if (options.signal?.aborted) queueMicrotask(() => finish({ timedOut: false, failure: 'stopped' }));
    options.signal?.addEventListener('abort', () => finish({ timedOut: false, failure: 'stopped' }), { once: true });

    worker.onmessage = (event: MessageEvent<FromWorker>) => {
      const message = event.data;
      switch (message.type) {
        case 'call': {
          hostCalls++;
          const answer =
            hostCalls > MAX_HOST_CALLS
              ? Promise.reject(new Error(`host call limit (${MAX_HOST_CALLS}) reached`))
              : handler.call(message.kind, message.args);
          answer.then(
            (ok) => {
              if (!settled) writer.begin(JSON.stringify({ ok }));
            },
            (error: unknown) => {
              if (!settled) writer.begin(JSON.stringify({ err: error instanceof Error ? error.message : String(error) }));
            },
          );
          break;
        }
        case 'more':
          writer.next();
          break;
        case 'done':
          finish({ timedOut: false, result: message.result });
          break;
        case 'fatal':
          finish({ timedOut: false, failure: message.message });
          break;
      }
    };
    worker.onerror = (event) => finish({ timedOut: false, failure: event.message || 'the sandbox Worker failed' });
    worker.postMessage({
      type: 'run',
      module,
      glueUrl: `${zippBase()}zipp_wasm.js`,
      sab,
      request,
      guest: { prelude: PRELUDE, shell: SHELL, tools: TOOLS },
    });
  });
}
