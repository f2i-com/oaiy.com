import React, { useMemo } from 'react';
import type { PropertyDefinition, PropertyCondition, PropertyOption } from 'oaiy-core';
import { TextField } from './TextField';
import { TextAreaField } from './TextAreaField';
import { NumberField } from './NumberField';
import { SelectField } from './SelectField';
import { MultiSelectField } from './MultiSelectField';
import { CheckboxField } from './CheckboxField';
import { SecretField } from './SecretField';
import { CodeField } from './CodeField';
import { useDynamicOptions } from '../../hooks/useDynamicOptions';

export interface PropertyFieldProps {
  property: PropertyDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  allValues?: Record<string, unknown>;
  className?: string;
}

/**
 * Check if a property should be visible based on its condition
 * Supports two formats:
 * 1. PropertyCondition: { property: 'mode', equals: 'external' }
 * 2. Simple object: { mode: 'external' } (key is property, value is expected value)
 */
export function shouldShowProperty(
  condition: PropertyCondition | Record<string, unknown> | undefined,
  allValues: Record<string, unknown>
): boolean {
  if (!condition) return true;

  // Check if it's the PropertyCondition format (has 'property' key)
  if ('property' in condition && typeof condition.property === 'string') {
    const typedCondition = condition as PropertyCondition;
    const targetValue = allValues[typedCondition.property];

    if (typedCondition.equals !== undefined) {
      return targetValue === typedCondition.equals;
    }

    if (typedCondition.notEquals !== undefined) {
      return targetValue !== typedCondition.notEquals;
    }

    if (typedCondition.in !== undefined) {
      return typedCondition.in.includes(targetValue);
    }

    return true;
  }

  // Simple object format: { mode: 'external', voice: 'custom' }
  // All conditions must match (AND logic)
  for (const [propName, expectedValue] of Object.entries(condition)) {
    const actualValue = allValues[propName];
    if (actualValue !== expectedValue) {
      return false;
    }
  }

  return true;
}

/**
 * PropertyField - Renders the appropriate field component based on property type
 */
export const PropertyField: React.FC<PropertyFieldProps> = ({
  property,
  value,
  onChange,
  allValues = {},
  className = '',
}) => {
  // Check visibility condition
  const isVisible = useMemo(
    () => shouldShowProperty(property.showIf, allValues),
    [property.showIf, allValues]
  );

  // Resolve any dynamic options source (only fires when the property
  // declares one; the hook short-circuits on undefined). Resolved
  // options are merged in front of the static `options` list at render
  // time so e.g. externally-discovered diffusion checkpoints appear
  // above the "Custom checkpoint file…" sentinel.
  const { options: dynamicOptions } = useDynamicOptions(property.dynamicOptionsSource);
  const mergedOptions: PropertyOption[] = useMemo(() => {
    const staticOpts = property.options ?? [];
    if (!property.dynamicOptionsSource) return staticOpts;
    // De-dupe by stringified value — if a static fallback shares an id
    // with a dynamic entry, the dynamic one wins (richer metadata).
    const seen = new Set<string>();
    const merged: PropertyOption[] = [];
    for (const opt of dynamicOptions) {
      const key = String(opt.value);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(opt);
    }
    for (const opt of staticOpts) {
      const key = String(opt.value);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(opt);
    }
    return merged;
  }, [property.dynamicOptionsSource, property.options, dynamicOptions]);

  if (!isVisible || property.hidden) {
    return null;
  }

  const commonProps = {
    id: property.id,
    label: property.name,
    description: property.description,
    disabled: property.disabled,
    required: property.required,
    placeholder: property.placeholder,
    className,
  };

  switch (property.type) {
    case 'string':
    case 'text':
      return (
        <TextField
          {...commonProps}
          value={String(value ?? property.default ?? '')}
          onChange={onChange}
        />
      );

    case 'textarea':
      return (
        <TextAreaField
          {...commonProps}
          value={String(value ?? property.default ?? '')}
          onChange={onChange}
          rows={property.rows}
        />
      );

    case 'number':
      return (
        <NumberField
          {...commonProps}
          value={(value ?? property.default) as number | undefined}
          onChange={onChange}
          min={property.min}
          max={property.max}
          step={property.step}
        />
      );

    case 'boolean':
      return (
        <CheckboxField
          {...commonProps}
          value={Boolean(value ?? property.default ?? false)}
          onChange={onChange}
        />
      );

    case 'select':
      return (
        <SelectField
          {...commonProps}
          value={(value ?? property.default) as string | number | undefined}
          onChange={onChange}
          options={mergedOptions.map((opt) => ({
            value: opt.value as string | number,
            label: opt.label,
            description: opt.description,
          }))}
        />
      );

    case 'secret':
      return (
        <SecretField
          {...commonProps}
          value={String(value ?? property.default ?? '')}
          onChange={onChange}
        />
      );

    case 'code':
      return (
        <CodeField
          {...commonProps}
          value={String(value ?? property.default ?? '')}
          onChange={onChange}
          language={property.language}
          rows={property.rows}
        />
      );

    case 'json':
      return (
        <CodeField
          {...commonProps}
          value={
            typeof value === 'string'
              ? value
              : JSON.stringify(value ?? property.default ?? {}, null, 2)
          }
          onChange={(val) => {
            try {
              onChange(JSON.parse(val));
            } catch {
              onChange(val);
            }
          }}
          language="json"
          rows={property.rows || 4}
        />
      );

    case 'multiselect':
      return (
        <MultiSelectField
          {...commonProps}
          value={
            Array.isArray(value ?? property.default)
              ? ((value ?? property.default) as (string | number)[])
              : []
          }
          onChange={onChange}
          options={mergedOptions.map((opt) => ({
            value: opt.value as string | number,
            label: opt.label,
            description: opt.description,
          }))}
        />
      );

    case 'color':
      return (
        <div className={`flex flex-col gap-1 ${className}`}>
          <label htmlFor={property.id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
            {property.name}
          </label>
          <input
            id={property.id}
            type="color"
            value={String(value ?? property.default ?? '#000000')}
            onChange={(e) => onChange(e.target.value)}
            disabled={property.disabled}
            className="w-full h-8 bg-[rgb(var(--color-bg-tertiary))] border border-[rgb(var(--color-border-primary))] rounded cursor-pointer disabled:opacity-50"
          />
          {property.description && (
            <p className="text-[10px] text-[rgb(var(--color-text-muted))]">{property.description}</p>
          )}
        </div>
      );

    case 'file':
    case 'filepath':
      // 'filepath' is the absolute-path variant for native nodes that
      // need to mmap / load files from disk (diffusion checkpoints,
      // model weights, etc.). When running in Tauri it pops the OS
      // file dialog via filesystem|pick_file and stores the absolute
      // path; the legacy 'file' type stays as the HTML input fallback
      // for browser-only dev mode.
      return (
        <FilePickerField
          property={property}
          value={value}
          onChange={onChange}
          className={className}
          wantsAbsolutePath={property.type === 'filepath'}
        />
      );

    case 'array':
    case 'keyvalue':
      // Complex types - render as JSON for now
      return (
        <CodeField
          {...commonProps}
          value={
            typeof value === 'string'
              ? value
              : JSON.stringify(value ?? property.default ?? [], null, 2)
          }
          onChange={(val) => {
            try {
              onChange(JSON.parse(val));
            } catch {
              onChange(val);
            }
          }}
          language="json"
          rows={property.rows || 4}
        />
      );

    default:
      // Fallback to text field
      return (
        <TextField
          {...commonProps}
          value={String(value ?? property.default ?? '')}
          onChange={onChange}
        />
      );
  }
};

// ---------------------------------------------------------------------------
// File picker — handles both 'file' (HTML input, name only) and 'filepath'
// (Tauri OS dialog, absolute path). Absolute path mode is what diffusion /
// model-loader nodes use because they need to actually open the file.
// ---------------------------------------------------------------------------

interface FilePickerFieldProps {
  property: PropertyDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  className: string;
  wantsAbsolutePath: boolean;
}

const FilePickerField: React.FC<FilePickerFieldProps> = ({
  property,
  value,
  onChange,
  className,
  wantsAbsolutePath,
}) => {
  const currentValue = String(value ?? property.default ?? '');
  const fileName = useMemo(() => {
    if (!currentValue) return '';
    const parts = currentValue.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || currentValue;
  }, [currentValue]);

  const handlePick = async () => {
    // Tauri v2 with `withGlobalTauri: false` removes `window.__TAURI__`
    // entirely; only `window.__TAURI_INTERNALS__.invoke` is exposed.
    // The old global stays as a compatibility shim for builds that flip
    // the flag back on, so check both before falling through to the
    // browser HTML <input> (which returns *name only* on every modern
    // browser for security — that's the bug that makes diffusion node
    // workflows save a bare filename like `cyberrealisticPony_*.safetensors`).
    if (wantsAbsolutePath) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const w = window as any;
      const invoke: ((cmd: string, args?: unknown) => Promise<unknown>) | undefined =
        w.__TAURI__?.core?.invoke ?? w.__TAURI_INTERNALS__?.invoke;
      if (invoke) {
        try {
          const result = await invoke('plugin:oaiy-filesystem|pick_file', {
            filters: property.fileFilters,
          });
          if (typeof result === 'string' && result.length > 0) {
            onChange(result);
          }
        } catch (err) {
          console.error('[filepath] pick_file failed', err);
        }
        return;
      }
    }
    // Browser-only / non-Tauri fallback. Returns name only.
    const input = document.getElementById(`${property.id}-fallback-input`) as HTMLInputElement | null;
    input?.click();
  };

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <label htmlFor={property.id} className="text-xs font-medium text-[rgb(var(--color-text-secondary))]">
        {property.name}
      </label>
      <div className="flex items-center gap-2">
        <button
          id={property.id}
          type="button"
          onClick={handlePick}
          disabled={property.disabled}
          className="px-2 py-1 text-xs rounded border border-[rgb(var(--color-border-primary))] bg-[rgb(var(--color-bg-tertiary))] hover:bg-[rgb(var(--color-bg-tertiary))] disabled:opacity-50"
        >
          {currentValue ? 'Change…' : 'Browse…'}
        </button>
        <span
          className="flex-1 text-xs truncate text-[rgb(var(--color-text-secondary))]"
          title={currentValue || 'No file selected'}
        >
          {fileName || <span className="italic text-slate-500">No file selected</span>}
        </span>
        {currentValue && (
          <button
            type="button"
            onClick={() => onChange('')}
            className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
            title="Clear"
          >
            ✕
          </button>
        )}
      </div>
      {/* Browser-fallback hidden input. Won't fire in Tauri because we
          short-circuit to the OS dialog above. */}
      <input
        id={`${property.id}-fallback-input`}
        type="file"
        accept={property.accept}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onChange(file.name);
        }}
        disabled={property.disabled}
        className="hidden"
      />
      {property.description && (
        <p className="text-[10px] text-slate-500 dark:text-slate-400">{property.description}</p>
      )}
    </div>
  );
};
