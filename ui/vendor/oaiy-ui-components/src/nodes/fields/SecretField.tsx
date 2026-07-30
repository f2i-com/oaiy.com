import React, { useCallback, useState } from 'react';

// Inline SVG icons to avoid lucide-react dependency
const EyeIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
  </svg>
);

const EyeOffIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
  </svg>
);

export interface SecretFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
  className?: string;
}

export const SecretField: React.FC<SecretFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  description,
  disabled = false,
  required = false,
  className = '',
}) => {
  const [showValue, setShowValue] = useState(false);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  const toggleVisibility = useCallback(() => {
    setShowValue((prev) => !prev);
  }, []);

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
        {label}
        {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
      </label>
      <div className="relative">
        <input
          id={id}
          type={showValue ? 'text' : 'password'}
          value={value || ''}
          onChange={handleChange}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          spellCheck={false}
          className="w-full px-2 py-1.5 pr-8 text-xs bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded text-[rgb(var(--color-text-primary))] placeholder-[rgb(var(--color-text-muted))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          type="button"
          onClick={toggleVisibility}
          aria-label={showValue ? 'Hide value' : 'Show value'}
          aria-pressed={showValue}
          title={showValue ? 'Hide value' : 'Show value'}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-[rgb(var(--color-text-muted))] hover:text-[rgb(var(--color-text-primary))] focus:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent-primary))] rounded"
        >
          {showValue ? <EyeOffIcon /> : <EyeIcon />}
        </button>
      </div>
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
      )}
    </div>
  );
};
