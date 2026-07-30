/**
 * In-memory ring buffer of webview console output.
 *
 * Wraps `console.{log,info,warn,error,debug}` and the global `error` /
 * `unhandledrejection` events so the API server can read recent JS-side
 * messages via `/api/console`. Without this, plugin-load errors and other
 * runtime failures only appear in the webview devtools — invisible to
 * scripted callers and to anyone debugging from outside the app.
 *
 * The buffer is fixed-size (oldest entries drop first). Each entry has a
 * monotonic id so a poller can ask for "everything after id X".
 */
export type ConsoleLevel = 'log' | 'info' | 'warn' | 'error' | 'debug';

export interface ConsoleEntry {
  /** Monotonically increasing id within this session. */
  id: number;
  /** ms since epoch. */
  ts: number;
  level: ConsoleLevel;
  /** Joined string form of the original args. */
  message: string;
}

const MAX_ENTRIES = 1000;
const buffer: ConsoleEntry[] = [];
let nextId = 1;
let installed = false;

/** Best-effort string form of a single console arg. */
function formatArg(arg: unknown): string {
  if (arg === null) return 'null';
  if (arg === undefined) return 'undefined';
  if (typeof arg === 'string') return arg;
  if (typeof arg === 'number' || typeof arg === 'boolean') return String(arg);
  if (arg instanceof Error) {
    return arg.stack ? `${arg.message}\n${arg.stack}` : arg.message;
  }
  try {
    return JSON.stringify(arg, (_k, v) => {
      if (v instanceof Error) return { name: v.name, message: v.message, stack: v.stack };
      return v;
    });
  } catch {
    return String(arg);
  }
}

function push(level: ConsoleLevel, args: unknown[]): void {
  buffer.push({
    id: nextId++,
    ts: Date.now(),
    level,
    message: args.map(formatArg).join(' '),
  });
  while (buffer.length > MAX_ENTRIES) buffer.shift();
}

/**
 * Install the hooks. Idempotent — calling twice has no extra effect.
 * Must run as early as possible (top of main.tsx) so we capture import-time
 * failures from `dynamicModules` and the React tree mount.
 */
export function installConsoleBuffer(): void {
  if (installed) return;
  installed = true;

  const orig = {
    log: console.log.bind(console),
    info: console.info.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
    debug: console.debug.bind(console),
  };

  (['log', 'info', 'warn', 'error', 'debug'] as ConsoleLevel[]).forEach((level) => {
    console[level] = (...args: unknown[]) => {
      push(level, args);
      orig[level](...(args as []));
    };
  });

  window.addEventListener('error', (ev) => {
    push('error', [
      '[window.error]',
      ev.message || ev.error,
      ev.filename ? `${ev.filename}:${ev.lineno}:${ev.colno}` : '',
    ]);
  });

  window.addEventListener('unhandledrejection', (ev) => {
    push('error', ['[unhandledrejection]', ev.reason]);
  });
}

/** Return the buffer, optionally filtered by minimum id and/or levels. */
export function readConsoleBuffer(opts: {
  sinceId?: number;
  levels?: ConsoleLevel[];
  limit?: number;
} = {}): { entries: ConsoleEntry[]; nextId: number; total: number } {
  const sinceId = opts.sinceId ?? 0;
  const levelSet = opts.levels && opts.levels.length > 0 ? new Set(opts.levels) : null;
  const limit = opts.limit ?? MAX_ENTRIES;

  const filtered: ConsoleEntry[] = [];
  for (const e of buffer) {
    if (e.id <= sinceId) continue;
    if (levelSet && !levelSet.has(e.level)) continue;
    filtered.push(e);
    if (filtered.length >= limit) break;
  }

  return { entries: filtered, nextId, total: buffer.length };
}

/** Drop everything in the buffer. Used by the clear endpoint. */
export function clearConsoleBuffer(): void {
  buffer.length = 0;
}
