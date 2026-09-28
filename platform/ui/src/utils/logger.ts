/**
 * Logger — consistent, level-based logging for the OAIY app (web + desktop).
 *
 * Lifted from the desktop bundle when oaiy-web split off; no Tauri deps.
 * Used by App.tsx, OAIYApp.tsx, dynamicModules.ts, and the lib/ helpers.
 *
 * Features:
 * - Log levels: debug, info, warn, error
 * - Environment-aware (debug logs only in development)
 * - Consistent formatting with prefixes
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

// Check if we're in development mode
const isDevelopment = import.meta.env.DEV === true || import.meta.env.MODE === 'development';

// Log level priorities (higher = more important)
const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Logger class for structured, level-based logging
 */
export class Logger {
  private source: string;
  private minLevel: LogLevel;

  constructor(source: string, options?: { minLevel?: LogLevel }) {
    this.source = source;
    // In development, show all logs including debug
    // In production, only show info and above
    this.minLevel = options?.minLevel ?? (isDevelopment ? 'debug' : 'info');
  }

  /**
   * Check if a log level should be logged
   */
  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[this.minLevel];
  }

  /**
   * Format and output a log entry
   */
  private log(level: LogLevel, message: string, data?: unknown): void {
    if (!this.shouldLog(level)) {
      return;
    }

    const prefix = `[${this.source}]`;
    const formattedMessage = data !== undefined
      ? `${prefix} ${message}`
      : `${prefix} ${message}`;

    switch (level) {
      case 'debug':
        if (data !== undefined) {
          console.debug(formattedMessage, data);
        } else {
          console.debug(formattedMessage);
        }
        break;
      case 'info':
        if (data !== undefined) {
          console.info(formattedMessage, data);
        } else {
          console.info(formattedMessage);
        }
        break;
      case 'warn':
        if (data !== undefined) {
          console.warn(formattedMessage, data);
        } else {
          console.warn(formattedMessage);
        }
        break;
      case 'error':
        if (data !== undefined) {
          console.error(formattedMessage, data);
        } else {
          console.error(formattedMessage);
        }
        break;
    }
  }

  /**
   * Log a debug message (only shown in development)
   */
  debug(message: string, data?: unknown): void {
    this.log('debug', message, data);
  }

  /**
   * Log an info message
   */
  info(message: string, data?: unknown): void {
    this.log('info', message, data);
  }

  /**
   * Log a warning message
   */
  warn(message: string, data?: unknown): void {
    this.log('warn', message, data);
  }

  /**
   * Log an error message
   */
  error(message: string, data?: unknown): void {
    this.log('error', message, data);
  }

  /**
   * Create a child logger with a different source prefix
   */
  child(childSource: string): Logger {
    return new Logger(`${this.source}:${childSource}`, { minLevel: this.minLevel });
  }
}

/**
 * Create a new logger with a custom source
 */
export function createLogger(source: string, options?: { minLevel?: LogLevel }): Logger {
  return new Logger(source, options);
}

// Pre-configured loggers for different parts of the application
export const workflowLogger = createLogger('Workflow');
export const packageLogger = createLogger('Package');
export const moduleLogger = createLogger('Module');
export const uiLogger = createLogger('UI');
export const databaseLogger = createLogger('Database');
