import React, { useCallback } from 'react';

export interface CheckboxFieldProps {
  id: string;
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  description?: string;
  disabled?: boolean;
  className?: string;
}

export const CheckboxField: React.FC<CheckboxFieldProps> = ({
  id,
  label,
  value,
  onChange,
  description,
  disabled = false,
  className = '',
}) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.checked);
    },
    [onChange]
  );

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="flex items-center gap-2 cursor-pointer">
        <input
          id={id}
          type="checkbox"
          checked={value || false}
          onChange={handleChange}
          disabled={disabled}
          className="w-4 h-4 bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded accent-[rgb(var(--accent-primary))] focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <span className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">{label}</span>
      </label>
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))] ml-6">{description}</p>
      )}
    </div>
  );
};
