import { useState, useCallback } from 'react';
import { uiLogger as logger } from '../../utils/logger';

interface CopyButtonProps {
  /** Text to copy to clipboard */
  text: string;
  /** Optional label shown next to icon */
  label?: string;
  /** Size variant */
  size?: 'sm' | 'md';
  /** Additional CSS classes */
  className?: string;
  /** Called after successful copy */
  onCopy?: () => void;
}

/**
 * A button that copies text to clipboard with visual feedback.
 * Shows a checkmark icon briefly after copying.
 */
export function CopyButton({
  text,
  label,
  size = 'sm',
  className = '',
  onCopy
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      onCopy?.();
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error('Failed to copy to clipboard', { error: err });
    }
  }, [text, onCopy]);

  const iconSize = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4';
  const padding = size === 'sm' ? 'p-1' : 'p-1.5';
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`
        inline-flex items-center gap-1 rounded transition-colors
        ${padding}
        ${copied
          ? 'text-green-500 dark:text-green-400'
          : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
        }
        ${className}
      `}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <svg className={iconSize} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className={iconSize} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      )}
      {label && <span className={textSize}>{copied ? 'Copied!' : label}</span>}
    </button>
  );
}

/**
 * Inline copy link for use within text/error messages.
 */
export function CopyLink({
  text,
  label = 'Copy',
  className = ''
}: {
  text: string;
  label?: string;
  className?: string;
}) {
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
      className={`
        text-xs underline transition-colors
        ${copied
          ? 'text-green-500 dark:text-green-400 no-underline'
          : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
        }
        ${className}
      `}
    >
      {copied ? 'Copied!' : label}
    </button>
  );
}

export default CopyButton;
