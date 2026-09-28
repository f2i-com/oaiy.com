import { Component, type ReactNode, type ErrorInfo, useState, useCallback } from 'react';
import { uiLogger as logger } from '../utils/logger';
import { clearAllAppStorage } from '../lib/storageKeys';

// Inline CopyButton for use within class component
function InlineCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error('Failed to copy to clipboard', { error: err });
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`text-xs underline ml-2 ${
        copied
          ? 'text-green-600 dark:text-green-500 no-underline'
          : 'text-[rgb(var(--color-text-tertiary))] hover:text-[rgb(var(--color-text-secondary))]'
      }`}
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  // How many times "Try Again" has re-rendered into another catch. A deterministic
  // crash (e.g. a corrupt persisted project) re-catches immediately; after a couple
  // we surface the harder "Reset saved data" recovery so the user isn't stuck.
  retryCount: number;
}

/**
 * Error Boundary component that catches JavaScript errors in child components.
 * Prevents the entire app from crashing when a component throws during render.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, retryCount: 0 };
  }

  // Partial return so the merge preserves retryCount across re-catches.
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    logger.error('ErrorBoundary caught an error', { error, errorInfo });
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState((s) => ({ hasError: false, error: null, retryCount: s.retryCount + 1 }));
  };

  handleReload = (): void => {
    window.location.reload();
  };

  // Escape hatch for a deterministic crash whose cause is corrupt persisted state:
  // a plain reload re-loads the same poisoned project and crashes again, so clear
  // the app's saved data first. clearAllAppStorage only touches oaiy_* keys.
  handleResetData = (): void => {
    try {
      clearAllAppStorage();
    } catch (err) {
      logger.error('Failed to clear app storage from ErrorBoundary', { error: err });
    }
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-[rgb(var(--color-bg-tertiary))] p-8">
          <div className="bg-[rgb(var(--color-bg-elevated))] border border-red-600/50 rounded-lg p-6 max-w-md text-center">
            <svg
              className="w-12 h-12 text-red-500 mx-auto mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <h2 className="text-xl font-semibold text-[rgb(var(--color-text-primary))] mb-2">
              Something went wrong
            </h2>
            <p className="text-[rgb(var(--color-text-secondary))] text-sm mb-4">
              An error occurred while rendering the workflow builder.
            </p>
            {this.state.error && (
              <details className="text-left mb-4">
                <summary className="text-[rgb(var(--color-text-tertiary))] text-xs cursor-pointer hover:text-[rgb(var(--color-text-primary))]">
                  Error details
                  <InlineCopyButton text={this.state.error.message + (this.state.error.stack ? '\n\n' + this.state.error.stack : '')} />
                </summary>
                <pre className="mt-2 p-2 bg-[rgb(var(--color-bg-tertiary))] rounded text-red-600 dark:text-red-400 text-xs overflow-auto max-h-32">
                  {this.state.error.message}
                </pre>
              </details>
            )}
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                onClick={this.handleRetry}
                className="px-4 py-2 bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white rounded-lg text-sm font-medium transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={this.handleReload}
                className="px-4 py-2 bg-[rgb(var(--color-bg-tertiary))] hover:bg-[rgb(var(--color-bg-secondary))] text-[rgb(var(--color-text-primary))] rounded-lg text-sm font-medium transition-colors"
              >
                Reload app
              </button>
              <button
                onClick={this.handleResetData}
                className="px-4 py-2 border border-red-600/50 text-red-600 dark:text-red-400 hover:bg-red-600/10 rounded-lg text-sm font-medium transition-colors"
              >
                Reset saved data
              </button>
            </div>
            {this.state.retryCount >= 2 && (
              <p className="text-[rgb(var(--color-text-tertiary))] text-xs mt-3">
                Still failing after retrying? Your saved project may be corrupt — “Reset saved data” clears the locally-saved flows, run history, and settings, then reloads.
              </p>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
