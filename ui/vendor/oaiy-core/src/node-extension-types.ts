/**
 * Node Extension Types
 *
 * Defines the type system for extending existing nodes with custom functionality.
 * Extensions can add inputs, outputs, modify compilation, wrap runtime execution,
 * and add custom UI elements.
 */

import type { GraphNode, WorkflowGraph } from './types';
import type { HandleDefinition, PropertyDefinition } from './module-types';

/**
 * Additional input to add to an existing node
 */
export interface ExtensionInput {
  /** Unique ID for the input (must not conflict with base node) */
  id: string;
  /** Display name */
  name: string;
  /** Data type */
  type: string;
  /** Whether the input is required */
  required?: boolean;
  /** Default value */
  defaultValue?: unknown;
  /** Description */
  description?: string;
  /** Position hint (before/after existing input) */
  position?: 'before' | 'after';
  /** Reference input for positioning */
  relativeTo?: string;
}

/**
 * Additional output to add to an existing node
 */
export interface ExtensionOutput {
  /** Unique ID for the output (must not conflict with base node) */
  id: string;
  /** Display name */
  name: string;
  /** Data type */
  type: string;
  /** Description */
  description?: string;
  /** Position hint */
  position?: 'before' | 'after';
  /** Reference output for positioning */
  relativeTo?: string;
}

/**
 * Additional property to add to node configuration
 */
export interface ExtensionProperty {
  /** Unique ID for the property */
  id: string;
  /** Display name */
  name: string;
  /** Property type */
  type: 'string' | 'number' | 'boolean' | 'select' | 'multiline';
  /** Default value */
  defaultValue?: unknown;
  /** Description */
  description?: string;
  /** Options for select type */
  options?: Array<{ value: string; label: string }>;
  /** Group for organization */
  group?: string;
}

/**
 * Context for compiler hooks
 */
export interface CompilerHookContext {
  /** The node being compiled */
  node: GraphNode;
  /** The full workflow graph */
  graph: WorkflowGraph;
  /** Get a property value */
  getProperty: (id: string) => unknown;
  /** Get an input value */
  getInputValue: (id: string) => string | null;
  /** Check if an input is connected */
  isInputConnected: (id: string) => boolean;
  /** Generate a unique variable name */
  generateVarName: (prefix?: string) => string;
}

/**
 * Compiler hook interface for extensions
 */
export interface CompilerHook {
  /**
   * Called before base node compilation
   * Can modify the node before it's compiled
   */
  preCompile?: (node: GraphNode, context: CompilerHookContext) => GraphNode;

  /**
   * Called after base node compilation
   * Can modify the generated code
   */
  postCompile?: (code: string, node: GraphNode, context: CompilerHookContext) => string;

  /**
   * Called to inject additional code before the node
   */
  injectBefore?: (node: GraphNode, context: CompilerHookContext) => string;

  /**
   * Called to inject additional code after the node
   */
  injectAfter?: (node: GraphNode, context: CompilerHookContext) => string;
}

/**
 * Context for runtime hooks
 */
export interface RuntimeHookContext {
  /** Node type */
  nodeType: string;
  /** Node ID */
  nodeId: string;
  /** Flow ID */
  flowId?: string;
  /** Package ID */
  packageId?: string;
  /** Abort signal */
  abortSignal?: AbortSignal;
  /** Log function */
  log: (message: string, level?: 'info' | 'warn' | 'error') => void;
  /** HTTP fetch with CORS bypass */
  fetch: typeof fetch;
  /** Environment variables */
  env: Record<string, string>;
}

/**
 * Runtime hook interface for extensions
 */
export interface RuntimeHook {
  /**
   * Called before base node execution
   * Can modify inputs before they're passed to the base node
   */
  preExecute?: (
    inputs: Record<string, unknown>,
    context: RuntimeHookContext
  ) => Record<string, unknown> | Promise<Record<string, unknown>>;

  /**
   * Called after base node execution
   * Can modify the result before it's returned
   */
  postExecute?: (
    result: unknown,
    inputs: Record<string, unknown>,
    context: RuntimeHookContext
  ) => unknown | Promise<unknown>;

  /**
   * Full execution override
   * If provided, replaces the base node execution entirely
   * The baseExecute function can be called to run the original node
   */
  execute?: (
    inputs: Record<string, unknown>,
    context: RuntimeHookContext,
    baseExecute: (inputs: Record<string, unknown>) => Promise<unknown>
  ) => Promise<unknown>;

  /**
   * Error handler
   * Called when the base node throws an error
   */
  onError?: (
    error: Error,
    inputs: Record<string, unknown>,
    context: RuntimeHookContext
  ) => unknown | Promise<unknown>;
}

/**
 * UI extension position
 */
export type UIExtensionPosition = 'header' | 'before-inputs' | 'after-inputs' | 'before-outputs' | 'after-outputs' | 'footer';

/**
 * UI extension definition
 */
export interface UIExtension {
  /** Unique ID for the UI extension */
  id: string;
  /** Position in the node */
  position: UIExtensionPosition;
  /** Source code for the UI component (TSX) */
  source: string;
  /** Compiled component code (JS) */
  compiled?: string;
}

/**
 * Complete node extension definition
 */
export interface NodeExtension {
  /** Unique ID for the extension */
  id: string;

  /** Display name */
  name: string;

  /** Description of what the extension does */
  description?: string;

  /** Node type this extension applies to (e.g., 'ai_llm', 'image_gen') */
  extends: string;

  /** Package that provides this extension */
  packageId?: string;

  /** Priority for extension ordering (higher = runs first) */
  priority?: number;

  /** Whether the extension is enabled by default */
  enabledByDefault?: boolean;

  /** Condition for when the extension applies */
  condition?: {
    /** Property that must match */
    property: string;
    /** Value to match (or array of values) */
    value: unknown | unknown[];
    /** Match operator */
    operator?: 'equals' | 'notEquals' | 'contains' | 'exists';
  };

  /** Additional inputs to add */
  additionalInputs?: ExtensionInput[];

  /** Additional outputs to add */
  additionalOutputs?: ExtensionOutput[];

  /** Additional properties to add */
  additionalProperties?: ExtensionProperty[];

  /** Compiler hook source (TypeScript) */
  compilerHook?: {
    source: string;
    compiled?: string;
  };

  /** Runtime hook source (TypeScript) */
  runtimeHook?: {
    source: string;
    compiled?: string;
  };

  /** UI extensions */
  uiExtensions?: UIExtension[];
}

/**
 * Loaded compiler hook
 */
export type LoadedCompilerHook = CompilerHook;

/**
 * Loaded runtime hook
 */
export type LoadedRuntimeHook = RuntimeHook;

/**
 * Extension with loaded hooks
 */
export interface LoadedExtension extends NodeExtension {
  /** Loaded compiler hook function */
  loadedCompilerHook?: LoadedCompilerHook;
  /** Loaded runtime hook function */
  loadedRuntimeHook?: LoadedRuntimeHook;
  /** Loaded UI components */
  loadedUIComponents?: Record<string, unknown>;
  /** Whether the extension has been loaded */
  loaded: boolean;
  /** Load error if any */
  loadError?: string;
}

/**
 * Result of applying extensions to compilation
 */
export interface ExtendedCompilationResult {
  /** The final compiled code */
  code: string;
  /** Whether any extensions were applied */
  extensionsApplied: number;
  /** Warnings from extensions */
  warnings: string[];
  /** Errors from extensions */
  errors: string[];
}

/**
 * Result of applying extensions to runtime
 */
export interface ExtendedRuntimeResult {
  /** The final result */
  result: unknown;
  /** Whether any extensions were applied */
  extensionsApplied: number;
  /** Whether the result was overridden by an extension */
  overridden: boolean;
}

/**
 * Validate an extension definition
 */
export interface ExtensionValidationResult {
  valid: boolean;
  errors: Array<{ path: string; message: string }>;
  warnings: Array<{ path: string; message: string }>;
}

export function validateNodeExtension(extension: unknown): ExtensionValidationResult {
  const errors: Array<{ path: string; message: string }> = [];
  const warnings: Array<{ path: string; message: string }> = [];

  if (!extension || typeof extension !== 'object') {
    errors.push({ path: '', message: 'Extension must be an object' });
    return { valid: false, errors, warnings };
  }

  const ext = extension as Record<string, unknown>;

  // Required fields
  if (!ext.id || typeof ext.id !== 'string') {
    errors.push({ path: 'id', message: 'id is required and must be a string' });
  }

  if (!ext.name || typeof ext.name !== 'string') {
    errors.push({ path: 'name', message: 'name is required and must be a string' });
  }

  if (!ext.extends || typeof ext.extends !== 'string') {
    errors.push({ path: 'extends', message: 'extends is required and must be a string' });
  }

  // Validate additional inputs
  if (ext.additionalInputs && Array.isArray(ext.additionalInputs)) {
    for (let i = 0; i < ext.additionalInputs.length; i++) {
      const input = ext.additionalInputs[i] as Record<string, unknown>;
      if (!input.id) {
        errors.push({ path: `additionalInputs[${i}].id`, message: 'id is required' });
      }
      if (!input.name) {
        errors.push({ path: `additionalInputs[${i}].name`, message: 'name is required' });
      }
      if (!input.type) {
        errors.push({ path: `additionalInputs[${i}].type`, message: 'type is required' });
      }
    }
  }

  // Validate additional outputs
  if (ext.additionalOutputs && Array.isArray(ext.additionalOutputs)) {
    for (let i = 0; i < ext.additionalOutputs.length; i++) {
      const output = ext.additionalOutputs[i] as Record<string, unknown>;
      if (!output.id) {
        errors.push({ path: `additionalOutputs[${i}].id`, message: 'id is required' });
      }
      if (!output.name) {
        errors.push({ path: `additionalOutputs[${i}].name`, message: 'name is required' });
      }
      if (!output.type) {
        errors.push({ path: `additionalOutputs[${i}].type`, message: 'type is required' });
      }
    }
  }

  // Validate hooks
  if (ext.compilerHook && typeof ext.compilerHook === 'object') {
    const hook = ext.compilerHook as Record<string, unknown>;
    if (!hook.source || typeof hook.source !== 'string') {
      errors.push({ path: 'compilerHook.source', message: 'source is required for compiler hook' });
    }
  }

  if (ext.runtimeHook && typeof ext.runtimeHook === 'object') {
    const hook = ext.runtimeHook as Record<string, unknown>;
    if (!hook.source || typeof hook.source !== 'string') {
      errors.push({ path: 'runtimeHook.source', message: 'source is required for runtime hook' });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Check if an extension condition matches a node
 */
export function extensionConditionMatches(
  extension: NodeExtension,
  node: GraphNode
): boolean {
  if (!extension.condition) {
    return true;
  }

  const { property, value, operator = 'equals' } = extension.condition;
  const nodeValue = node.data?.[property];

  switch (operator) {
    case 'equals':
      if (Array.isArray(value)) {
        return value.includes(nodeValue);
      }
      return nodeValue === value;

    case 'notEquals':
      if (Array.isArray(value)) {
        return !value.includes(nodeValue);
      }
      return nodeValue !== value;

    case 'contains':
      if (typeof nodeValue === 'string' && typeof value === 'string') {
        return nodeValue.includes(value);
      }
      if (Array.isArray(nodeValue)) {
        return nodeValue.includes(value);
      }
      return false;

    case 'exists':
      return nodeValue !== undefined && nodeValue !== null;

    default:
      return true;
  }
}
