import React, { useCallback } from 'react';

export interface SelectOption {
  value: string | number;
  label: string;
  description?: string;
  /** Options sharing a group are shown under it (<optgroup>), in first-seen order. */
  group?: string;
  /** Picking it runs this instead of changing the value ("Add a service…"). */
  action?: () => void;
}

export interface SelectFieldProps {
  id: string;
  label: string;
  value: string | number | undefined;
  onChange: (value: string | number) => void;
  options: SelectOption[];
  placeholder?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
  className?: string;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  id,
  label,
  value,
  onChange,
  options,
  placeholder = 'Select...',
  description,
  disabled = false,
  required = false,
  className = '',
}) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const selectedOption = options.find(opt => String(opt.value) === e.target.value);
      if (selectedOption?.action) {
        // An action, not a value: run it and show the value as it was.
        e.target.value = value !== undefined ? String(value) : '';
        selectedOption.action();
        return;
      }
      if (selectedOption) {
        onChange(selectedOption.value);
      }
    },
    [onChange, options, value]
  );

  const optionClass = 'bg-[rgb(var(--color-bg-elevated))] text-[rgb(var(--color-text-primary))]';
  const renderOption = (option: SelectOption) => (
    <option key={String(option.value)} value={String(option.value)} title={option.description} className={optionClass}>
      {option.label}
    </option>
  );
  // Ungrouped options first, then each group in the order it first appears.
  const groups: Array<[string, SelectOption[]]> = [];
  for (const option of options) {
    if (!option.group) continue;
    const found = groups.find(([name]) => name === option.group);
    if (found) found[1].push(option);
    else groups.push([option.group, [option]]);
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
        {label}
        {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
      </label>
      <select
        id={id}
        value={value !== undefined ? String(value) : ''}
        onChange={handleChange}
        disabled={disabled}
        className="w-full px-2 py-1.5 text-xs bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded text-[rgb(var(--color-text-primary))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {placeholder && (
          <option value="" className="bg-[rgb(var(--color-bg-elevated))] text-[rgb(var(--color-text-primary))]">
            {placeholder}
          </option>
        )}
        {/* Keep the stored value visible when it isn't (yet) in options — e.g. a
            saved id whose dynamic option list hasn't resolved, or a removed
            option. Without this the field falls back to the placeholder and
            misleadingly reads as unset while node.data still holds the value. */}
        {value != null && String(value) !== '' &&
          !options.some((o) => String(o.value) === String(value)) && (
            <option value={String(value)} className="bg-[rgb(var(--color-bg-elevated))] text-[rgb(var(--color-text-primary))]">
              {String(value)} (unavailable)
            </option>
          )}
        {options.filter((option) => !option.group).map(renderOption)}
        {groups.map(([name, list]) => (
          <optgroup key={name} label={name}>
            {list.map(renderOption)}
          </optgroup>
        ))}
      </select>
      {description && (
        <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
      )}
    </div>
  );
};
