import React, { useCallback } from 'react';

export interface NumberFieldProps {
  id: string;
  label: string;
  value: number | undefined;
  onChange: (value: number | undefined) => void;
  placeholder?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}

export const NumberField: React.FC<NumberFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  description,
  disabled = false,
  required = false,
  min,
  max,
  step = 1,
  className = '',
}) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      if (val === '') {
        onChange(undefined);
      } else {
        const num = parseFloat(val);
        if (!isNaN(num)) {
          onChange(num);
        }
      }
    },
    [onChange]
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
        {label}
        {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
      </label>
      <input
        id={id}
        type="number"
        value={value ?? ''}
        onChange={handleChange}
        placeholder={placeholder}
        disabled={disabled}
        min={min}
        max={max}
        step={step}
        className="w-full px-2 py-1.5 text-xs bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded text-[rgb(var(--color-text-primary))] placeholder-[rgb(var(--color-text-muted))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed"
      />
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
      )}
    </div>
  );
};
