import React, { useCallback, useState } from 'react';

export type CodeLanguage =
  | 'javascript'
  | 'typescript'
  | 'json'
  | 'html'
  | 'css'
  | 'sql'
  | 'markdown';

export interface CodeFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  language?: CodeLanguage;
  placeholder?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
  rows?: number;
  className?: string;
}

export const CodeField: React.FC<CodeFieldProps> = ({
  id,
  label,
  value,
  onChange,
  language = 'javascript',
  placeholder,
  description,
  disabled = false,
  required = false,
  rows = 6,
  className = '',
}) => {
  // Escape releases Tab from indentation mode so keyboard users are never trapped (WCAG 2.1.2).
  // Shift+Tab always moves focus backward normally.
  const [tabReleased, setTabReleased] = useState(false);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Escape') {
        setTabReleased(true);
        return;
      }
      if (e.key === 'Tab') {
        if (e.shiftKey || tabReleased) {
          // Let the browser move focus out of the field normally
          return;
        }
        e.preventDefault();
        const target = e.target as HTMLTextAreaElement;
        const start = target.selectionStart;
        const end = target.selectionEnd;
        const newValue = value.substring(0, start) + '  ' + value.substring(end);
        onChange(newValue);
        requestAnimationFrame(() => {
          target.selectionStart = target.selectionEnd = start + 2;
        });
        return;
      }
      if (tabReleased && !e.metaKey && !e.ctrlKey && !e.altKey && e.key !== 'Shift') {
        setTabReleased(false);
      }
    },
    [value, onChange, tabReleased]
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
          {label}
          {required && <span className="text-[rgb(var(--signal-danger-ink))] ml-1">*</span>}
        </label>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-[rgb(var(--color-text-faint))]">
            {tabReleased ? 'Tab moves focus' : 'Esc, then Tab to leave'}
          </span>
          <span className="text-[10px] font-mono text-[rgb(var(--color-text-secondary))] bg-[rgb(var(--color-bg-tertiary))] border border-[rgb(var(--color-border-primary))] px-1.5 py-0.5 rounded-[var(--r-sm)]">
            {language}
          </span>
        </div>
      </div>
      <textarea
        id={id}
        value={value || ''}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onFocus={() => setTabReleased(false)}
        placeholder={placeholder}
        disabled={disabled}
        rows={rows}
        spellCheck={false}
        className="w-full px-2.5 py-1.5 text-xs bg-[rgb(var(--color-bg-tertiary))] border border-[rgb(var(--color-border-primary))] rounded-[var(--r-ctl)] text-[rgb(var(--color-text-primary))] placeholder-[rgb(var(--color-text-muted))] focus:outline-none focus:border-[rgb(var(--accent-primary))] focus:ring-2 focus:ring-[rgb(var(--accent-primary)/0.25)] disabled:opacity-50 disabled:cursor-not-allowed resize-y min-h-[80px] font-mono transition-colors"
      />
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-faint))]">{description}</p>
      )}
    </div>
  );
};
