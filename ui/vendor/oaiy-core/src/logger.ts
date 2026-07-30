/**
 * Structured Logger - Provides consistent, level-based logging across the application.
 *
 * Features:
 * - Log levels: debug, info, warn, error
 * - Environment-aware (debug logs only in development)
 * - Consistent formatting with timestamps and prefixes
 * - Structured data support
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  level: LogLevel;
  source: string;
  message: string;
  timestamp: string;
  data?: Record<string, unknown>;
}

export type LogHandler = (entry: LogEntry) => void;

// Check if we're in development mode
const isDevelopment = (() => {
  try {
    // Check for Vite environment (use indirect eval to avoid syntax errors in non-ESM contexts like Jest)
    // eslint-disable-next-line no-eval
    const meta = new Function('try { return import.meta } catch(e) { return undefined }')() as { env?: { DEV?: boolean; MODE?: string } } | undefined;
    if (meta?.env?.DEV === true) return true;
    if (meta?.env?.MODE === 'development') return true;
    // Check for Node environment (with proper typing)
    if (typeof globalThis !== 'undefined' && 'process' in globalThis) {
      const nodeProcess = (globalThis as { process?: { env?: { NODE_ENV?: string } } }).process;
      if (nodeProcess?.env?.NODE_ENV === 'development') {
        return true;
      }
    }
  } catch {
    // Ignore errors in environments where these aren't available
  }
  return false;
})();

// Log level priorities (higher = more important)
const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

// =====================================================================
// Sensitive-data redaction
// =====================================================================
//
// Every log entry passes through `redactSensitive` before being printed
// to the console or fanned out to handlers. This is a best-effort defence
// against accidentally leaking API keys, bearer tokens, prompts, and so
// on into logs that the user might paste into an issue tracker or share
// for debugging.
//
// New patterns can be added with `addRedactionPattern`. The redaction
// runs on the message string and on stringified `data` values.

interface RedactionRule {
  re: RegExp;
  replacement: string;
}

const REDACTION_RULES: RedactionRule[] = [
  // Common bearer/API key formats — strip the actual secret, keep the
  // prefix so logs remain debuggable.
  { re: /(sk|pk|xai|gho|ghp|ghu|ghs|ghr|hf|sk-ant)_[A-Za-z0-9_\-]{8,}/g, replacement: '$1_<redacted>' },
  // `Authorization: Bearer …` (any case)
  { re: /(Bearer\s+)[A-Za-z0-9._\-]{16,}/gi, replacement: '$1<redacted>' },
  // Quoted Authorization header values
  { re: /("[Aa]uthorization"\s*:\s*")[^"]+(")/g, replacement: '$1<redacted>$2' },
  // x-api-key / apiKey / password — case-insensitive on the key.
  { re: /("[Xx]-[Aa]pi-[Kk]ey"\s*:\s*")[^"]+(")/g, replacement: '$1<redacted>$2' },
  { re: /("(?:apiKey|api_key|password|secret|token)"\s*:\s*")[^"]+(")/gi, replacement: '$1<redacted>$2' },
  // === Terminal command redaction ========================================
  // CLI flags that commonly carry secrets.
  // Matches: `--token VALUE`, `--token=VALUE`, `-T=VALUE`. Stops at first whitespace.
  { re: /(--?(?:token|api[-_]?key|password|secret|auth)=)([^\s'"]+)/gi, replacement: '$1<redacted>' },
  { re: /(--?(?:token|api[-_]?key|password|secret|auth)\s+)([^\s'"]+)/gi, replacement: '$1<redacted>' },
  // Form-encoded / query-string secrets.
  { re: /\b(api[-_]?key|password|secret|token|auth)=([^&\s'"]+)/gi, replacement: '$1=<redacted>' },
  // Credentials embedded in URLs: scheme://user:pass@host
  { re: /(\b[a-z]+:\/\/)([^/\s:@]+):([^/\s@]+)@/gi, replacement: '$1<redacted>:<redacted>@' },
];

/** Register an additional redaction pattern at runtime. */
export function addRedactionPattern(pattern: RegExp, replacement: string): void {
  REDACTION_RULES.push({ re: pattern, replacement });
}

/** Apply all configured redaction rules to a string. */
export function redactSensitive(s: string): string {
  if (!s) return s;
  let out = s;
  for (const { re, replacement } of REDACTION_RULES) {
    out = out.replace(re, replacement);
  }
  return out;
}

/**
 * Privacy policy for filesystem paths in user-visible logs. Mirrors
 * `LogPathPolicy` in `types.ts` (re-declared here to keep this module
 * dependency-free).
 *
 *   * `full`     — print the path verbatim.
 *   * `basename` — only the file name.
 *   * `tilde`    — replace $HOME prefix with `~`, otherwise keep last
 *                  two segments. Default.
 *   * `none`     — replace the path entirely with `<path>`.
 */
export type LogPathPolicy = 'full' | 'basename' | 'tilde' | 'none';

/**
 * Format a filesystem path for display in workflow logs according to the
 * supplied policy. Always normalises backslashes to forward slashes and
 * strips embedded null bytes.
 */
export function formatPathForLog(path: string, policy: LogPathPolicy = 'tilde'): string {
  if (!path) return '';
  // Normalise + strip null bytes (defence-in-depth — these would already
  // have been refused by path validation but the log is a best-effort
  // mirror, not a strict parser).
  const cleaned = path.replace(/\\/g, '/').replace(/\0/g, '').trim();

  if (policy === 'full') return cleaned;
  if (policy === 'none') return '<path>';

  const segments = cleaned.split('/').filter(Boolean);
  const base = segments.length ? segments[segments.length - 1] : cleaned;
  if (policy === 'basename') return base;

  // 'tilde' — try to rewrite the platform $HOME / %USERPROFILE% prefix.
  const home = detectHomePrefix();
  if (home) {
    const homeNorm = home.replace(/\\/g, '/').replace(/\/+$/, '');
    if (cleaned.startsWith(homeNorm + '/') || cleaned === homeNorm) {
      const rest = cleaned.slice(homeNorm.length);
      return '~' + rest;
    }
  }
  // Fallback for paths outside $HOME: keep the last two segments so a
  // log entry shows enough context to identify the file without leaking
  // the surrounding directory tree.
  if (segments.length <= 2) return cleaned;
  return '…/' + segments.slice(-2).join('/');
}

function detectHomePrefix(): string | undefined {
  if (typeof process !== 'undefined' && process.env) {
    return process.env.HOME || process.env.USERPROFILE;
  }
  return undefined;
}

/** Recursively redact any string fields in an object. */
function redactData(d: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (!d) return d;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(d)) {
    if (typeof v === 'string') out[k] = redactSensitive(v);
    else if (v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = redactData(v as Record<string, unknown>);
    } else out[k] = v;
  }
  return out;
}

/**
 * Logger class for structured, level-based logging
 */
export class Logger {
  private source: string;
  private minLevel: LogLevel;
  private handlers: LogHandler[] = [];

  constructor(source: string, options?: { minLevel?: LogLevel }) {
    this.source = source;
    // In development, show all logs including debug
    // In production, only show info and above
    this.minLevel = options?.minLevel ?? (isDevelopment ? 'debug' : 'info');
  }

  /**
   * Add a custom log handler
   */
  addHandler(handler: LogHandler): void {
    this.handlers.push(handler);
  }

  /**
   * Remove a log handler
   */
  removeHandler(handler: LogHandler): void {
    const index = this.handlers.indexOf(handler);
    if (index !== -1) {
      this.handlers.splice(index, 1);
    }
  }

  /**
   * Set the minimum log level
   */
  setMinLevel(level: LogLevel): void {
    this.minLevel = level;
  }

  /**
   * Check if a log level should be logged
   */
  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[this.minLevel];
  }

  /**
   * Check if a log level is enabled.
   * Use this to avoid expensive operations (like JSON.stringify) when the log level is disabled.
   *
   * @example
   * ```typescript
   * if (logger.isEnabled('debug')) {
   *   logger.debug(`Large object: ${JSON.stringify(bigObject)}`);
   * }
   * ```
   */
  isEnabled(level: LogLevel): boolean {
    return this.shouldLog(level);
  }

  /**
   * Check if debug logging is enabled.
   * Convenience method for the common case of expensive debug logging.
   */
  get isDebugEnabled(): boolean {
    return this.shouldLog('debug');
  }

  /**
   * Format and output a log entry. Both the message and any structured
   * data run through `redactSensitive` first so API keys / bearer tokens
   * never reach the console or downstream handlers.
   */
  private log(level: LogLevel, message: string, data?: Record<string, unknown>): void {
    if (!this.shouldLog(level)) {
      return;
    }

    const safeMessage = redactSensitive(message);
    const safeData = redactData(data);

    const entry: LogEntry = {
      level,
      source: this.source,
      message: safeMessage,
      timestamp: new Date().toISOString(),
      data: safeData,
    };

    // Output to console with appropriate method
    const prefix = `[${this.source}]`;
    const formattedMessage = safeData
      ? `${prefix} ${safeMessage} ${JSON.stringify(safeData)}`
      : `${prefix} ${safeMessage}`;

    switch (level) {
      case 'debug':
        console.debug(formattedMessage);
        break;
      case 'info':
        console.info(formattedMessage);
        break;
      case 'warn':
        console.warn(formattedMessage);
        break;
      case 'error':
        console.error(formattedMessage);
        break;
    }

    // Call custom handlers
    for (const handler of this.handlers) {
      try {
        handler(entry);
      } catch {
        // Ignore handler errors to prevent log loops
      }
    }
  }

  /**
   * Log a debug message (only shown in development)
   */
  debug(message: string, data?: Record<string, unknown>): void {
    this.log('debug', message, data);
  }

  /**
   * Log an info message
   */
  info(message: string, data?: Record<string, unknown>): void {
    this.log('info', message, data);
  }

  /**
   * Log a warning message
   */
  warn(message: string, data?: Record<string, unknown>): void {
    this.log('warn', message, data);
  }

  /**
   * Log an error message
   */
  error(message: string, data?: Record<string, unknown>): void {
    this.log('error', message, data);
  }

  /**
   * Create a child logger with a different source prefix
   */
  child(childSource: string): Logger {
    return new Logger(`${this.source}:${childSource}`, { minLevel: this.minLevel });
  }
}

// Pre-configured loggers for different parts of the application
export const compilerLogger = new Logger('Compiler');
export const runtimeLogger = new Logger('Runtime');
export const databaseLogger = new Logger('Database');
export const moduleLogger = new Logger('Module');

/**
 * Create a new logger with a custom source
 */
export function createLogger(source: string, options?: { minLevel?: LogLevel }): Logger {
  return new Logger(source, options);
}
