import React, { useCallback } from 'react';

export interface TextAreaFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
  rows?: number;
  className?: string;
}

export const TextAreaField: React.FC<TextAreaFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  description,
  disabled = false,
  required = false,
  rows = 3,
  className = '',
}) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
        {label}
        {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
      </label>
      <textarea
        id={id}
        value={value || ''}
        onChange={handleChange}
        placeholder={placeholder}
        disabled={disabled}
        rows={rows}
        className="w-full px-2 py-1.5 text-xs bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded text-[rgb(var(--color-text-primary))] placeholder-[rgb(var(--color-text-muted))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed resize-y min-h-[60px]"
      />
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
      )}
    </div>
  );
};
