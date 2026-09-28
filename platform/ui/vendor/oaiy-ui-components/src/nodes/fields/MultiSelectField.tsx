import React, { useCallback } from 'react';

export interface MultiSelectOption {
    value: string | number;
    label: string;
    description?: string;
}

export interface MultiSelectFieldProps {
    id: string;
    label: string;
    value: (string | number)[] | undefined;
    onChange: (value: (string | number)[]) => void;
    options: MultiSelectOption[];
    placeholder?: string;
    description?: string;
    disabled?: boolean;
    required?: boolean;
    className?: string;
    rows?: number;
}

export const MultiSelectField: React.FC<MultiSelectFieldProps> = ({
    id,
    label,
    value,
    onChange,
    options,
    description,
    disabled = false,
    required = false,
    className = '',
    rows = 4,
}) => {
    const handleChange = useCallback(
        (e: React.ChangeEvent<HTMLSelectElement>) => {
            const selectedValues = Array.from(e.target.selectedOptions).map(opt => opt.value);

            // Map back to original values to preserve types (number vs string)
            const newValues = options
                .filter(opt => selectedValues.includes(String(opt.value)))
                .map(opt => opt.value);

            onChange(newValues);
        },
        [onChange, options]
    );

    // Convert current values to strings for the select element
    const selectedStrings = (value || []).map(v => String(v));

    return (
        <div className={`flex flex-col gap-1 ${className}`}>
            <label htmlFor={id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
                {label}
                {required && <span className="text-red-500 dark:text-red-400 ml-1">*</span>}
            </label>
            <select
                id={id}
                multiple
                value={selectedStrings}
                onChange={handleChange}
                disabled={disabled}
                size={rows}
                className="w-full px-2 py-1.5 text-xs bg-[rgb(var(--color-bg-elevated))] border border-[rgb(var(--color-border-primary))] rounded text-[rgb(var(--color-text-primary))] focus:outline-none focus:border-[rgb(var(--accent-primary))] disabled:opacity-50 disabled:cursor-not-allowed custom-scrollbar"
            >
                {options.map((option) => (
                    <option key={String(option.value)} value={String(option.value)} className="bg-[rgb(var(--color-bg-elevated))] text-[rgb(var(--color-text-primary))] py-1">
                        {option.label}
                    </option>
                ))}
            </select>
            <div className="text-[10px] text-[rgb(var(--color-text-muted))] italic">Hold Ctrl/Cmd to select multiple</div>
            {description && (
                <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{description}</p>
            )}
        </div>
    );
};
