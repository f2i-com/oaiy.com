/**
 * Core Utility Module Runtime
 *
 * Provides utility functions: template processing, logic blocks, memory storage.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: every method closes over THIS job's `ctx` (and a per-job
// `memory` map) instead of a module-level singleton, so concurrently-running jobs
// never cross-contaminate. Invoked once per job by runtime.registerDynamicModule.
function createUtilityMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {
  // Agent memory storage (per job)
  const memory: Map<string, unknown> = new Map();

/**
 * Escape regex metacharacters so a user-supplied template variable name
 * (`{{name}}`) is matched literally and can't accidentally turn into a
 * regex pattern.
 */
function escapeRegExpChars(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Process a template string with variable substitution.
 *
 * Variable names are escaped before being embedded in the regex, so a
 * caller can't construct a name like `foo|bar` that turns into an
 * alternation, and replacement values are passed through `replace`'s
 * function form to neutralise `$&`/`$1`-style back-references.
 */
function template(
  templateStr: string,
  variables: Record<string, unknown>
): string {
  ctx.log('info', `[Template] Processing template with ${Object.keys(variables).length} variables`);

  let result = templateStr;

  for (const [key, value] of Object.entries(variables)) {
    const safeKey = escapeRegExpChars(key);
    const placeholder = new RegExp(`\\{\\{${safeKey}\\}\\}`, 'g');
    const stringValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
    // Function-form `replace` so `$&` etc. in `stringValue` don't get expanded.
    result = result.replace(placeholder, () => stringValue);
  }

  return result;
}

// NOTE: the `logicBlock` runtime handler was REMOVED (security review). It was dead — zero
// callers; the compiler INLINES logic_block code into the worker script, gated by
// `requireCodeExecute` below. As a reachable host.call it ran untrusted code via
// `new Function` on the MAIN thread with full ambient authority (window/__TAURI__/fetch),
// escaping the worker realm packages are isolated in.

/**
 * Read from memory
 */
function memoryRead(key: string, defaultValue?: unknown): unknown {
  ctx.log('info', `[Memory] Reading key: ${key}`);
  const value = memory.get(key);
  if (value === undefined) {
    return defaultValue;
  }
  return value;
}

/**
 * Write to memory
 */
function memoryWrite(key: string, value: unknown): void {
  ctx.log('info', `[Memory] Writing key: ${key}`);
  memory.set(key, value);
}

/**
 * Clear memory
 */
function memoryClear(): void {
  ctx.log('info', '[Memory] Clearing all keys');
  memory.clear();
}

/**
 * Make an HTTP request (used by Playwright browser integration)
 * Returns: { status: number, headers: object, body: string }
 */
async function httpRequest(
  url: string,
  method: string,
  headers: Record<string, string>,
  body?: string
): Promise<{ status: number; headers: Record<string, string>; body: string }> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[Utility] Aborted by user before HTTP request');
    throw new Error('Operation aborted by user');
  }

  ctx.log('info', `[Utility] HTTP ${method} ${url}`);

  try {
    // Use secureFetch if available (handles local network permissions)
    if (ctx.secureFetch) {
      const fetchOptions: RequestInit = {
        method,
        headers,
      };

      if (body) {
        fetchOptions.body = body;
      }

      const response = await ctx.secureFetch(url, fetchOptions);
      const responseBody = await response.text();

      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      return {
        status: response.status,
        headers: responseHeaders,
        body: responseBody,
      };
    }

    // Fallback to Tauri HTTP if available
    if (ctx.tauri) {
      const result = await ctx.tauri.invoke<{
        status: number;
        headers: Record<string, string>;
        body: string;
        url: string;
      }>('http_request', {
        request: {
          url,
          method,
          headers,
          body: body || null,
          follow_redirects: true,
          max_redirects: 10,
          allow_private_networks: true,
        }
      });

      return {
        status: result.status,
        headers: result.headers,
        body: result.body,
      };
    }

    // Final fallback to fetch
    const fetchOptions: RequestInit = {
      method,
      headers,
    };

    if (body) {
      fetchOptions.body = body;
    }

    const response = await ctx.fetch(url, fetchOptions);
    const responseBody = await response.text();

    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    return {
      status: response.status,
      headers: responseHeaders,
      body: responseBody,
    };
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Utility] HTTP request failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Free ComfyUI GPU memory by unloading models
 * Calls the ComfyUI /free endpoint to release VRAM
 */
async function comfyuiFreeMemory(
  comfyuiUrl: string,
  unloadModels: boolean,
  freeMemory: boolean,
  nodeId: string
): Promise<boolean> {
  ctx.log('info', `[ComfyUI Free Memory] Freeing GPU memory (unloadModels: ${unloadModels}, freeMemory: ${freeMemory})`);
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Build the request body based on options
    const requestBody: { unload_models?: boolean; free_memory?: boolean } = {};
    if (unloadModels) {
      requestBody.unload_models = true;
    }
    if (freeMemory) {
      requestBody.free_memory = true;
    }

    // Always go through the policy-enforced fetch so the local-network
    // permission flow + SSRF guard apply (ComfyUI typically lives on
    // localhost, but the URL is user-supplied).
    const response = await ctx.fetch(`${comfyuiUrl}/free`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      ctx.log('warn', `[ComfyUI Free Memory] API returned ${response.status}: ${errorText}`);
      // Don't throw - memory free is best-effort
      ctx.onNodeStatus?.(nodeId, 'completed');
      return false;
    }

    ctx.log('info', '[ComfyUI Free Memory] GPU memory freed successfully');
    ctx.onNodeStatus?.(nodeId, 'completed');
    return true;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('warn', `[ComfyUI Free Memory] Failed to free memory: ${errMsg}`);
    // Don't throw - memory free is best-effort
    ctx.onNodeStatus?.(nodeId, 'completed');
    return false;
  }
}

/**
 * Get the URL for a running service by its ID
 * Returns the URL with the dynamically allocated port, or null if service is not running
 */
async function getServiceUrl(serviceId: string): Promise<string | null> {
  ctx.log('info', `[Utility] Getting URL for service: ${serviceId}`);

  try {
    if (ctx.tauri) {
      const port = await ctx.tauri.invoke<number | null>('get_service_port', { serviceId });
      if (port) {
        const url = `http://127.0.0.1:${port}`;
        ctx.log('info', `[Utility] Service ${serviceId} running on ${url}`);
        return url;
      }
      ctx.log('warn', `[Utility] Service ${serviceId} is not running`);
      return null;
    }

    // Fallback: try the API endpoint if not in Tauri context
    if (ctx.secureFetch || ctx.fetch) {
      const fetchFn = ctx.secureFetch || ctx.fetch;
      const response = await fetchFn(`http://127.0.0.1:8767/api/services/${serviceId}/port`);
      if (response.ok) {
        const data = await response.json();
        if (data.data?.port) {
          const url = `http://127.0.0.1:${data.data.port}`;
          ctx.log('info', `[Utility] Service ${serviceId} running on ${url}`);
          return url;
        }
      }
    }

    ctx.log('warn', `[Utility] Could not determine URL for service ${serviceId}`);
    return null;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Utility] Failed to get service URL: ${errMsg}`);
    return null;
  }
}

/**
 * Result from ensure_service_ready Tauri command
 */
interface EnsureServiceResult {
  success: boolean;
  port: number | null;
  error: string | null;
  already_running: boolean;
}

/**
 * Ensure a service is running and healthy, starting it if needed.
 * Automatically starts the service if not running and waits for health check.
 * Returns the service URL or null if the service failed to start.
 * @param serviceId The service ID to look up
 */
async function ensureService(serviceId: string): Promise<string | null> {
  ctx.log('info', `[Utility] Ensuring service is ready: ${serviceId}`);

  try {
    if (ctx.tauri) {
      const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready', { serviceId });

      if (result.success && result.port) {
        const url = `http://127.0.0.1:${result.port}`;
        if (result.already_running) {
          ctx.log('info', `[Utility] Service ${serviceId} already running on ${url}`);
        } else {
          ctx.log('info', `[Utility] Service ${serviceId} started and healthy on ${url}`);
        }
        return url;
      }

      if (result.error) {
        ctx.log('warn', `[Utility] Service ${serviceId} failed to start: ${result.error}`);
      } else {
        ctx.log('warn', `[Utility] Service ${serviceId} is not available`);
      }
      return null;
    }

    // Fallback: try the API endpoint if not in Tauri context
    if (ctx.secureFetch || ctx.fetch) {
      const fetchFn = ctx.secureFetch || ctx.fetch;
      const response = await fetchFn(`http://127.0.0.1:8767/api/services/${serviceId}/ensure`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        if (data.data?.success && data.data?.port) {
          const url = `http://127.0.0.1:${data.data.port}`;
          ctx.log('info', `[Utility] Service ${serviceId} ready on ${url}`);
          return url;
        }
      }
    }

    ctx.log('warn', `[Utility] Could not ensure service ${serviceId} is ready`);
    return null;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Utility] Failed to ensure service: ${errMsg}`);
    return null;
  }
}

/**
 * Ensure a service is running and healthy by port number (dynamic lookup).
 * Automatically starts the service if not running and waits for health check.
 * Returns the service URL or null if the service failed to start.
 * @param port The port number to look up the service by
 */
async function ensureServiceByPort(port: number): Promise<string | null> {
  ctx.log('info', `[Utility] Ensuring service on port ${port} is ready...`);

  try {
    if (ctx.tauri) {
      const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready_by_port', { port });

      if (result.success && result.port) {
        const url = `http://127.0.0.1:${result.port}`;
        if (result.already_running) {
          ctx.log('info', `[Utility] Service on port ${port} already running at ${url}`);
        } else {
          ctx.log('info', `[Utility] Service on port ${port} started and healthy at ${url}`);
        }
        return url;
      }

      if (result.error) {
        ctx.log('warn', `[Utility] Service on port ${port} failed to start: ${result.error}`);
      } else {
        ctx.log('warn', `[Utility] Service on port ${port} is not available`);
      }
      return null;
    }

    ctx.log('warn', `[Utility] Could not ensure service on port ${port} is ready (Tauri not available)`);
    return null;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Utility] Failed to ensure service by port: ${errMsg}`);
    return null;
  }
}

  /**
   * Permission probe used by compiled code (e.g. condition expression mode) to
   * fail fast when the active package is not allowed to run arbitrary JS.
   * Returns null on success, throws on denial.
   */
  function requireCodeExecute(reason: string): null {
    if (ctx.hasPermission && !ctx.hasPermission('code:execute')) {
      throw new Error(
        `Refusing to evaluate ${reason} — the active package does not have the "code:execute" permission.`,
      );
    }
    return null;
  }

  return {
    template,
    memoryRead,
    memoryWrite,
    memoryClear,
    httpRequest,
    comfyuiFreeMemory,
    getServiceUrl,
    ensureService,
    ensureServiceByPort,
    requireCodeExecute,
  };
}

/**
 * Core Utility Runtime Module — per-job methods via createMethods (no shared ctx).
 */
const CoreUtilityRuntime: RuntimeModule = {
  name: 'Utility',
  createMethods: createUtilityMethods,
  methods: {},
  // Per-job `memory` is GC'd with the job's method closures, so no explicit clear.
  async cleanup(): Promise<void> {},
};

export default CoreUtilityRuntime;
