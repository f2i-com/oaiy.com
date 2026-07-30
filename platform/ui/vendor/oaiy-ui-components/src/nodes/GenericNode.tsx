import React, { memo, useCallback, useMemo } from 'react';
import { Position, useStore } from '@xyflow/react';
import type { NodeDefinition, HandleDefinition } from 'oaiy-core';
import { BUNDLED_MODULES, getModuleLoader } from 'oaiy-core';
import CollapsibleNodeWrapper, { type HandleConfig } from '../components/CollapsibleNodeWrapper';
import { PropertyField, shouldShowProperty } from './fields/PropertyField';

// Icon components map - we use simple SVG icons to avoid lucide-react dependency
const defaultIconSvg = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
  </svg>
);

// ============================================
// Types
// ============================================

export interface GenericNodeProps {
  id: string;
  type?: string;
  data: Record<string, unknown> & {
    __definition?: NodeDefinition;
    onChange?: (field: string, value: unknown) => void;
    previewImageUrl?: string;
    previewAudioUrl?: string;
    isStreaming?: boolean;
    streamContent?: string;
    showBodyProperties?: boolean;
  };
  selected?: boolean;
}

// ============================================
// Helper Functions
// ============================================

/**
 * Get icon SVG for a node - returns default box icon
 * Icons can be customized per-node in the future
 */
function getIcon(_iconName: string | undefined): React.ReactNode {
  // For now, return default icon. In the future, we can add icon mapping
  return defaultIconSvg;
}

/**
 * Convert handle definition to HandleConfig
 */
function toHandleConfig(handle: HandleDefinition, type: 'source' | 'target'): HandleConfig {
  return {
    id: handle.id,
    type,
    position: handle.position === 'left' || handle.position === 'right'
      ? handle.position === 'left' ? Position.Left : Position.Right
      : handle.position === 'top' ? Position.Top : Position.Bottom,
    color: handle.color || (type === 'source' ? '!bg-blue-500' : '!bg-gray-400'),
    size: handle.id === 'default' ? 'lg' : 'sm',
    label: handle.name,
  };
}

// Valid color names for CollapsibleNodeWrapper
type NodeColor = 'purple' | 'green' | 'blue' | 'amber' | 'cyan' | 'pink' | 'emerald' | 'orange' | 'red' | 'slate' | 'indigo' | 'teal' | 'violet';

const validColors: NodeColor[] = ['purple', 'green', 'blue', 'amber', 'cyan', 'pink', 'emerald', 'orange', 'red', 'slate', 'indigo', 'teal', 'violet'];

/**
 * Get color name from color string
 */
function getColorName(color: string | undefined): NodeColor {
  if (!color) return 'slate';

  // If it's already a valid color name
  const lowerColor = color.toLowerCase();
  if (validColors.includes(lowerColor as NodeColor)) {
    return lowerColor as NodeColor;
  }

  // If it's a Tailwind color like "purple-500", extract the base color
  if (color.includes('-')) {
    const baseColor = color.split('-')[0].toLowerCase();
    if (validColors.includes(baseColor as NodeColor)) {
      return baseColor as NodeColor;
    }
  }

  return 'slate';
}


// ============================================
// GenericNode Component
// ============================================

export const GenericNode: React.FC<GenericNodeProps> = memo(({ id, data }) => {
  // Get node type from the store using the node id
  const nodeType = useStore((state) => {
    const node = state.nodeLookup.get(id);
    return node?.type;
  });

  // Get definition from data or fallback to looking it up in BUNDLED_MODULES or ModuleLoader
  const definition = useMemo((): NodeDefinition | undefined => {
    if (data.__definition) {
      return data.__definition;
    }

    // Fallback lookup using node type
    if (nodeType) {
      // First check BUNDLED_MODULES
      for (const module of BUNDLED_MODULES) {
        const found = module.nodes.find(n => n.id === nodeType);
        if (found) return found;
      }

      // Then check ModuleLoader (for package nodes like pkg:packageId:nodeId)
      const moduleLoader = getModuleLoader();
      const moduleDef = moduleLoader.getNodeDefinition(nodeType);
      if (moduleDef) {
        return moduleDef;
      }
    }
    return undefined;
  }, [data.__definition, nodeType]);

  // If no definition, render a placeholder
  if (!definition) {
    return (
      <div className="p-4 bg-red-100 dark:bg-red-900/50 border border-red-500 rounded-lg text-red-900 dark:text-white text-xs">
        <p>Node definition not found</p>
        <p className="text-slate-500 dark:text-slate-400">ID: {id}</p>
        <p className="text-slate-500 dark:text-slate-400">Type: {nodeType || 'unknown'}</p>
      </div>
    );
  }

  // Current values keyed by property id — used to evaluate `showIf`
  // for both handles and properties. Falls back to the definition's
  // default when the data record hasn't been initialised yet (fresh
  // node drag), so showIf conditions resolve correctly from the moment
  // the node appears on the canvas.
  const allValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const prop of definition.properties || []) {
      values[prop.id] = data[prop.id] ?? prop.default;
    }
    return values;
  }, [definition.properties, data]);

  // Convert handles, filtering out any whose showIf condition doesn't
  // match the current property values. This lets a node JSON declare
  // model-specific pins (e.g. diffusion_image hiding image-reference
  // pins for SDXL/Pony, which are text-to-image only).
  const inputHandles = useMemo<HandleConfig[]>(
    () =>
      definition.inputs
        .filter((h) => shouldShowProperty(h.showIf, allValues))
        .map((h) => toHandleConfig(h, 'target')),
    [definition.inputs, allValues]
  );

  const outputHandles = useMemo<HandleConfig[]>(
    () =>
      definition.outputs
        .filter((h) => shouldShowProperty(h.showIf, allValues))
        .map((h) => toHandleConfig(h, 'source')),
    [definition.outputs, allValues]
  );

  // Create onChange handler
  const handlePropertyChange = useCallback(
    (propertyId: string) => (value: unknown) => {
      if (data.onChange) {
        data.onChange(propertyId, value);
      }
    },
    [data]
  );

  // Separate regular and advanced properties
  const regularProperties = useMemo(
    () => (definition.properties || []).filter((p) => !p.advanced),
    [definition.properties]
  );

  const advancedProperties = useMemo(
    () => (definition.properties || []).filter((p) => p.advanced),
    [definition.properties]
  );

  // Count of advanced fields that are *currently visible* — i.e. their
  // showIf condition matches the current values. The collapsed summary
  // shows this number so it matches what the user sees expanded.
  const visibleAdvancedCount = useMemo(
    () => advancedProperties.filter((p) => shouldShowProperty(p.showIf, allValues) && !p.hidden).length,
    [advancedProperties, allValues]
  );

  // Get node width from definition or use default
  const nodeWidth = definition.ui?.width || 280;

  return (
    <CollapsibleNodeWrapper
      title={definition.name}
      color={getColorName(definition.color)}
      icon={getIcon(definition.icon)}
      width={nodeWidth}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      <div className="flex flex-col gap-3">
        {/* Preview Image */}
        {definition.ui?.showPreview && data.previewImageUrl && (
          <div className="w-full aspect-video bg-slate-200 dark:bg-slate-800 rounded overflow-hidden">
            <img
              src={data.previewImageUrl as string}
              alt="Preview"
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* Streaming Content */}
        {data.isStreaming && data.streamContent && (
          <div className="p-2 bg-slate-200 dark:bg-slate-800 rounded text-xs text-slate-700 dark:text-slate-300 max-h-32 overflow-y-auto">
            {data.streamContent}
          </div>
        )}

        {/* Audio Preview */}
        {data.previewAudioUrl && (
          <div className="w-full bg-slate-200/50 dark:bg-slate-800/50 rounded p-2">
            <audio
              controls
              className="w-full h-8 dark:[filter:invert(1)_hue-rotate(180deg)]"
              src={data.previewAudioUrl as string}
            >
              Your browser does not support audio playback.
            </audio>
          </div>
        )}

        {/* Regular Properties (only if not hidden) */}
        {data.showBodyProperties !== false && regularProperties.map((prop) => (
          <PropertyField
            key={prop.id}
            property={prop}
            value={data[prop.id]}
            onChange={handlePropertyChange(prop.id)}
            allValues={allValues}
          />
        ))}

        {/* Advanced Properties (collapsible) (only if not hidden) */}
        {data.showBodyProperties !== false && visibleAdvancedCount > 0 && (
          <details className="mt-2">
            <summary className="text-xs text-slate-500 dark:text-slate-400 cursor-pointer hover:text-slate-700 dark:hover:text-slate-300">
              Advanced Options ({visibleAdvancedCount})
            </summary>
            <div className="flex flex-col gap-3 mt-2 pl-2 border-l border-slate-300 dark:border-slate-700">
              {advancedProperties.map((prop) => (
                <PropertyField
                  key={prop.id}
                  property={prop}
                  value={data[prop.id]}
                  onChange={handlePropertyChange(prop.id)}
                  allValues={allValues}
                />
              ))}
            </div>
          </details>
        )}
      </div>
    </CollapsibleNodeWrapper>
  );
});

GenericNode.displayName = 'GenericNode';
