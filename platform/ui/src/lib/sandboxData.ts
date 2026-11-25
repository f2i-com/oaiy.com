/**
 * Pure sanitization helpers for the SandboxedCustomNodeUI postMessage boundary.
 * Kept free of Vite virtual modules so both runtime code and Node tests can import them directly.
 */

const HOST_INJECTED_KEYS = new Set([
  'projectConstants',
  'projectSettings',
  'llmEndpoints',
  'availableFlows',
]);

const FORBIDDEN_KEYS = new Set([
  '__proto__',
  'constructor',
  'prototype',
  ...HOST_INJECTED_KEYS,
]);

function cloneJsonSafe(value: unknown, depth = 0): unknown {
  if (depth > 20) return undefined;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'function' || typeof value === 'symbol' || value === undefined) return undefined;
  if (Array.isArray(value)) {
    const out: unknown[] = [];
    for (const item of value) {
      const cloned = cloneJsonSafe(item, depth + 1);
      out.push(cloned === undefined ? null : cloned);
    }
    return out;
  }
  if (typeof value === 'object') {
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) return undefined;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>)) {
      if (FORBIDDEN_KEYS.has(key)) continue;
      const cloned = cloneJsonSafe((value as Record<string, unknown>)[key], depth + 1);
      if (cloned !== undefined) out[key] = cloned;
    }
    return out;
  }
  return undefined;
}

/**
 * Sanitize node `data` before `postMessage` into a sandboxed custom-node iframe:
 * keeps only JSON-cloneable own properties and strips functions, `on[A-Z]*` callbacks,
 * `_*` internal fields, and host-injected secrets/context (`projectConstants`,
 * `projectSettings`, `llmEndpoints`, `availableFlows`).
 */
export function sanitizeDataForSandbox(
  data: Record<string, unknown> | undefined | null,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!data || typeof data !== 'object') return out;
  for (const key of Object.keys(data)) {
    if (key.startsWith('_')) continue;
    if (/^on[A-Z]/.test(key)) continue;
    if (HOST_INJECTED_KEYS.has(key)) continue;
    const val = data[key];
    if (typeof val === 'function') continue;
    const cloned = cloneJsonSafe(val);
    if (cloned !== undefined) out[key] = cloned;
  }
  return out;
}

/**
 * Sanitize `updates` received from an untrusted custom-node iframe's `dataChange` message:
 * accepts only a plain object, rejects prototype-pollution keys (`__proto__`, `constructor`,
 * `prototype`), `_*` internal fields, `on[A-Z]*` callbacks, and host-injected keys.
 */
export function sanitizeSandboxUpdates(updates: unknown): Record<string, unknown> | null {
  if (!updates || typeof updates !== 'object' || Array.isArray(updates)) return null;
  const proto = Object.getPrototypeOf(updates);
  if (proto !== Object.prototype && proto !== null) return null;
  const out: Record<string, unknown> = {};
  let count = 0;
  for (const key of Object.keys(updates as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(key)) continue;
    if (key.startsWith('_')) continue;
    if (/^on[A-Z]/.test(key)) continue;
    const cloned = cloneJsonSafe((updates as Record<string, unknown>)[key]);
    if (cloned !== undefined) {
      out[key] = cloned;
      count++;
    }
  }
  return count > 0 ? out : null;
}
