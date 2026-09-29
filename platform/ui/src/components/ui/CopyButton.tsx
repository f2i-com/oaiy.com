import { useState, useCallback } from 'react';
import { Check, Copy } from 'lucide-react';
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
 * Shows a checkmark icon briefly after copying. With a label it is the
 * editor's ghost button; without one, an icon button.
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

  const iconSize = size === 'sm' ? 13 : 15;
  const base = label
    ? `btn btn-ghost${size === 'sm' ? ' btn-sm' : ''}`
    : `oaiy-icon-btn${size === 'sm' ? ' sm' : ''}`;

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`${base} ${className}`}
      // The class above sets the colour (unlayered CSS), so the copied tint is inline.
      style={copied ? { color: 'rgb(var(--signal-green))' } : undefined}
      title={copied ? 'Copied' : 'Copy to the clipboard'}
      aria-label={label ? undefined : copied ? 'Copied' : 'Copy to the clipboard'}
    >
      {copied ? <Check size={iconSize} aria-hidden="true" /> : <Copy size={iconSize} aria-hidden="true" />}
      {label && <span>{copied ? 'Copied' : label}</span>}
    </button>
  );
}

/**
 * Inline copy link for use within text/error messages. It takes the colour
 * of the text around it (an error banner's, a note's).
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
        text-xs transition-opacity
        ${copied
          ? 'text-signal-green no-underline'
          : 'underline opacity-75 hover:opacity-100'
        }
        ${className}
      `}
    >
      {copied ? 'Copied' : label}
    </button>
  );
}

export default CopyButton;
