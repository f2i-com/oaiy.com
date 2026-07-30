import React, { useCallback } from 'react';

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
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  // Handle tab key for indentation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const target = e.target as HTMLTextAreaElement;
        const start = target.selectionStart;
        const end = target.selectionEnd;
        const newValue = value.substring(0, start) + '  ' + value.substring(end);
        onChange(newValue);
        // Set cursor position after the inserted tab
        requestAnimationFrame(() => {
          target.selectionStart = target.selectionEnd = start + 2;
        });
      }
    },
    [value, onChange]
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between">
        <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
          {label}
          {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
        </label>
        <span className="text-[10px] text-[rgb(var(--color-text-secondary))] bg-[rgb(var(--color-bg-tertiary))] px-1.5 py-0.5 rounded">
          {language}
        </span>
      </div>
      <textarea
        id={id}
        value={value || ''}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={rows}
        spellCheck={false}
        className="w-full px-2 py-1.5 text-xs bg-[rgb(var(--color-bg-tertiary))] border border-[rgb(var(--color-border-primary))] rounded text-green-700 dark:text-green-400 placeholder-[rgb(var(--color-text-muted))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed resize-y min-h-[80px] font-mono"
      />
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
      )}
    </div>
  );
};
