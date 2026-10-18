/**
 * Value extraction helpers and type guards for compiled script results.
 *
 * Generated workflow scripts run as native JavaScript in V8, so result values
 * are plain JS values; the BaseObject alias is retained for backwards
 * compatibility with older callers.
 */

export type BaseObject = any;
export type ModuleFn = (...args: any[]) => any;

// Minimal interface for Array-like structures
interface ArrayLikeObject {
  elements: unknown[];
}

/**
 * Check if object is Array
 */
export function isArrayLike(obj: unknown): obj is unknown[] {
  return Array.isArray(obj);
}

/**
 * Check if object is Hash
 */
export function isHashWithMap(obj: unknown): boolean {
  return obj !== null && typeof obj === 'object' && !Array.isArray(obj);
}

export function isHashWithPlainPairs(obj: unknown): boolean {
  return obj !== null && typeof obj === 'object' && !Array.isArray(obj);
}

const MAX_EXTRACT_DEPTH = 50;

/**
 * A result's value, copied: plain data, its cycles and its depth bounded.
 */
export function extractDeepValue(val: unknown, depth: number = 0, seen: WeakSet<object> = new WeakSet()): unknown {
  if (depth > MAX_EXTRACT_DEPTH) {
    return '[Object: max depth exceeded]';
  }

  if (val === null || val === undefined) {
    return val;
  }

  if (typeof val !== 'object') {
    return val;
  }

  if (seen.has(val)) {
    return '[Circular Reference]';
  }
  // Track only ANCESTORS on the current DFS path (delete on unwind), not every
  // object ever visited. A shared/diamond reference — the same object reached
  // twice via different keys, e.g. a passthrough node aliasing an upstream
  // output into workflow_context under two ids — must extract on both paths
  // rather than collapse to '[Circular Reference]' on its second appearance.
  // A genuine cycle (an object referencing one of its own ancestors) is still
  // caught because the ancestor is still on the path when revisited.
  seen.add(val);
  try {
    if (Array.isArray(val)) {
      return val.map(el => extractDeepValue(el, depth + 1, seen));
    }

    const obj = val as Record<string, unknown>;

    // An object is kept as it is, all its keys. One of one or two keys with a name like `text`, `result`, `image` or
    // `url` used to be taken for a value the old script engine had boxed and replaced by that key's value: a step's
    // `{ text: "hello", lang: "en" }` became "hello" in the run's results and answer, and a macro's `text` and
    // `count` outputs lost `count`. Scripts run on plain values now; nothing is boxed.
    const unboxedObj: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      unboxedObj[k] = extractDeepValue(v, depth + 1, seen);
    }
    return unboxedObj;
  } finally {
    seen.delete(val);
  }
}

export function extractValue(val: unknown): string {
  const extracted = extractDeepValue(val);
  if (extracted === null || extracted === undefined) {
    return '';
  }
  if (typeof extracted === 'string') {
    return extracted;
  }
  if (typeof extracted === 'object') {
    if (Array.isArray(extracted)) {
      return extracted.map(e => typeof e === 'string' ? e : JSON.stringify(e)).join(', ');
    }
    return JSON.stringify(extracted);
  }
  return String(extracted);
}

export function extractArrayValues(arrayObj: any[]): string[] {
  return arrayObj.map((el) => {
    const extracted = extractDeepValue(el);
    if (typeof extracted === 'string') return extracted;
    if (extracted === null || extracted === undefined) return '';
    return JSON.stringify(extracted);
  });
}

export function extractNodeOutputs(result: unknown): {
  outputs: Record<string, string | string[]>;
  finalValue: string | string[];
} {
  const outputs: Record<string, string | string[]> = {};
  let finalValue: string | string[] = '';

  const cleanKey = (key: string): string => {
    return key.startsWith('string:') ? key.slice(7) : key;
  };

  const isFinalKey = (key: string): boolean => {
    const clean = cleanKey(key);
    return clean === 'final' || clean === '__output__';
  };

  const extractAnyValue = (value: unknown): string | string[] => {
    if (Array.isArray(value)) {
      return extractArrayValues(value);
    }
    return extractValue(value);
  };

  if (result && typeof result === 'object' && !Array.isArray(result)) {
    for (const [rawKey, value] of Object.entries(result)) {
      const key = cleanKey(rawKey);
      const extractedValue = extractAnyValue(value);
      outputs[key] = extractedValue;
      if (isFinalKey(rawKey)) {
        finalValue = extractedValue;
      }
    }
  }

  return { outputs, finalValue };
}
