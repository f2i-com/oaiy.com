/**
 * Custom Node Types
 *
 * Defines the type system for custom nodes that can be defined in TypeScript
 * and compiled on package installation.
 */

import type { GraphNode, WorkflowGraph } from './types';
import type { HandleDefinition, HandleDataType, PropertyDefinition } from './module-types';

/**
 * Input definition for a custom node
 */
export interface CustomNodeInput {
  /** Unique ID for the input */
  id: string;
  /** Display name */
  name: string;
  /** Data type */
  type: HandleDataType | string;
  /** Whether the input is required */
  required?: boolean;
  /** Default value if not connected */
  defaultValue?: unknown;
  /** Description for documentation */
  description?: string;
  /** Whether this input accepts multiple connections */
  multi?: boolean;
}

/**
 * Output definition for a custom node
 */
export interface CustomNodeOutput {
  /** Unique ID for the output */
  id: string;
  /** Display name */
  name: string;
  /** Data type */
  type: HandleDataType | string;
  /** Description for documentation */
  description?: string;
}

/**
 * Property definition for custom node configuration
 */
export interface CustomNodeProperty {
  /** Unique ID for the property */
  id: string;
  /** Display name */
  name: string;
  /** Property type */
  type: 'string' | 'number' | 'boolean' | 'select' | 'multiline' | 'json' | 'code' | 'file';
  /** Default value */
  defaultValue?: unknown;
  /** Description for documentation */
  description?: string;
  /** Options for select type */
  options?: Array<{ value: string; label: string }>;
  /** Whether the property is required */
  required?: boolean;
  /** Placeholder text */
  placeholder?: string;
}

/**
 * Complete definition for a custom node
 */
export interface CustomNodeDefinition {
  /** Unique ID (within the package) */
  id: string;

  /** Display name */
  name: string;

  /** Description of what the node does */
  description?: string;

  /** Category for node palette organization */
  category: string;

  /** Icon (icon name, URL, or base64 data) */
  icon?: string;

  /** Node color for visual distinction */
  color?: string;

  /** Input definitions */
  inputs: CustomNodeInput[];

  /** Output definitions */
  outputs: CustomNodeOutput[];

  /** Configurable properties */
  properties?: CustomNodeProperty[];

  /** Keywords for search */
  keywords?: string[];

  /** Whether this node is deprecated */
  deprecated?: boolean;

  /** Deprecation message with migration guidance */
  deprecationMessage?: string;

  /** Source TypeScript code (pre-compilation) */
  source: {
    /** Compiler code (TypeScript) - generates JavaScript code */
    compiler: string;
    /** Runtime code (TypeScript) - executes the node */
    runtime: string;
    /** UI component code (TSX) - optional custom UI */
    ui?: string;
  };

  /** Compiled JavaScript code (post-installation) */
  compiled?: {
    /** Compiled compiler module */
    compiler: string;
    /** Compiled runtime module */
    runtime: string;
    /** Compiled UI component */
    ui?: string;
  };
}

/**
 * Context provided to custom node compilers
 */
export interface CustomNodeCompilerContext {
  /** The node being compiled */
  node: GraphNode;

  /** The full workflow graph */
  graph: WorkflowGraph;

  /** Generate a unique variable name */
  generateVarName: (prefix?: string) => string;

  /** Get the variable name for a node's output */
  getNodeOutput: (nodeId: string, outputId?: string) => string;

  /** Check if a node has a specific input connected */
  isInputConnected: (inputId: string) => boolean;

  /** Get the variable name for a connected input */
  getInputValue: (inputId: string) => string | null;

  /** Get a property value from the node */
  getProperty: (propertyId: string) => unknown;

  /** Log a compilation warning */
  warn: (message: string) => void;

  /** Log a compilation error */
  error: (message: string) => void;
}

/**
 * Result from a custom node compiler
 */
export interface CustomNodeCompilerResult {
  /** The generated JavaScript code */
  code: string;

  /** Variable name containing the node's primary output */
  outputVar?: string;

  /** Map of output ID to variable name */
  outputs?: Record<string, string>;

  /** Additional imports required */
  imports?: string[];

  /** Whether compilation was successful */
  success: boolean;

  /** Error message if failed */
  error?: string;
}

/**
 * Context provided to custom node runtime
 */
export interface CustomNodeRuntimeContext {
  /** Node type */
  nodeType: string;

  /** Node ID */
  nodeId: string;

  /** Flow ID */
  flowId?: string;

  /** Package ID if running in a package context */
  packageId?: string;

  /** Abort signal for cancellation */
  abortSignal?: AbortSignal;

  /** Log a message */
  log: (message: string, level?: 'info' | 'warn' | 'error') => void;

  /** Update node status */
  setStatus: (status: 'running' | 'complete' | 'error', message?: string) => void;

  /** Stream a token (for progressive output) */
  streamToken: (token: string) => void;

  /** Report progress (0-100) */
  setProgress: (percent: number) => void;

  /** Access to HTTP client with CORS bypass */
  fetch: typeof fetch;

  /** Environment variables (filtered for security) */
  env: Record<string, string>;

  /** Storage API for persistent data */
  storage: {
    get: (key: string) => Promise<unknown>;
    set: (key: string, value: unknown) => Promise<void>;
    delete: (key: string) => Promise<void>;
  };
}

/**
 * Function signature for custom node compiler
 */
export type CustomNodeCompiler = (
  context: CustomNodeCompilerContext
) => CustomNodeCompilerResult;

/**
 * Function signature for custom node runtime
 */
export type CustomNodeRuntime = (
  inputs: Record<string, unknown>,
  properties: Record<string, unknown>,
  context: CustomNodeRuntimeContext
) => Promise<Record<string, unknown>>;

/**
 * Custom UI component props
 */
export interface CustomNodeUIProps {
  /** Node ID */
  id: string;

  /** Node data */
  data: Record<string, unknown>;

  /** Whether the node is selected */
  selected: boolean;

  /** Update node data */
  onChange: (data: Partial<Record<string, unknown>>) => void;

  /** The node definition */
  definition: CustomNodeDefinition;
}

/**
 * Package structure for custom nodes
 */
export interface CustomNodePackageStructure {
  /** Path to the definition file */
  definitionPath: string;

  /** Path to the compiler source */
  compilerPath: string;

  /** Path to the runtime source */
  runtimePath: string;

  /** Path to the UI component (optional) */
  uiPath?: string;
}

/**
 * Validation result for custom node definition
 */
export interface CustomNodeValidationResult {
  /** Whether the definition is valid */
  valid: boolean;

  /** Validation errors */
  errors: Array<{
    path: string;
    message: string;
  }>;

  /** Validation warnings */
  warnings: Array<{
    path: string;
    message: string;
  }>;
}

/**
 * Validates a custom node definition
 */
export function validateCustomNodeDefinition(
  definition: unknown
): CustomNodeValidationResult {
  const errors: Array<{ path: string; message: string }> = [];
  const warnings: Array<{ path: string; message: string }> = [];

  if (!definition || typeof definition !== 'object') {
    errors.push({ path: '', message: 'Definition must be an object' });
    return { valid: false, errors, warnings };
  }

  const def = definition as Record<string, unknown>;

  // Required fields
  if (!def.id || typeof def.id !== 'string') {
    errors.push({ path: 'id', message: 'id is required and must be a string' });
  } else if (!/^[a-z][a-z0-9_]*$/i.test(def.id)) {
    warnings.push({ path: 'id', message: 'id should use snake_case format' });
  }

  if (!def.name || typeof def.name !== 'string') {
    errors.push({ path: 'name', message: 'name is required and must be a string' });
  }

  if (!def.category || typeof def.category !== 'string') {
    errors.push({ path: 'category', message: 'category is required and must be a string' });
  }

  // Inputs validation
  if (!def.inputs || !Array.isArray(def.inputs)) {
    errors.push({ path: 'inputs', message: 'inputs is required and must be an array' });
  } else {
    const inputIds = new Set<string>();
    for (let i = 0; i < def.inputs.length; i++) {
      const input = def.inputs[i] as Record<string, unknown>;
      if (!input.id || typeof input.id !== 'string') {
        errors.push({ path: `inputs[${i}].id`, message: 'input id is required' });
      } else if (inputIds.has(input.id as string)) {
        errors.push({ path: `inputs[${i}].id`, message: `Duplicate input id: ${input.id}` });
      } else {
        inputIds.add(input.id as string);
      }
      if (!input.name || typeof input.name !== 'string') {
        errors.push({ path: `inputs[${i}].name`, message: 'input name is required' });
      }
      if (!input.type || typeof input.type !== 'string') {
        errors.push({ path: `inputs[${i}].type`, message: 'input type is required' });
      }
    }
  }

  // Outputs validation
  if (!def.outputs || !Array.isArray(def.outputs)) {
    errors.push({ path: 'outputs', message: 'outputs is required and must be an array' });
  } else {
    const outputIds = new Set<string>();
    for (let i = 0; i < def.outputs.length; i++) {
      const output = def.outputs[i] as Record<string, unknown>;
      if (!output.id || typeof output.id !== 'string') {
        errors.push({ path: `outputs[${i}].id`, message: 'output id is required' });
      } else if (outputIds.has(output.id as string)) {
        errors.push({ path: `outputs[${i}].id`, message: `Duplicate output id: ${output.id}` });
      } else {
        outputIds.add(output.id as string);
      }
      if (!output.name || typeof output.name !== 'string') {
        errors.push({ path: `outputs[${i}].name`, message: 'output name is required' });
      }
      if (!output.type || typeof output.type !== 'string') {
        errors.push({ path: `outputs[${i}].type`, message: 'output type is required' });
      }
    }
  }

  // Source validation
  if (!def.source || typeof def.source !== 'object') {
    errors.push({ path: 'source', message: 'source is required and must be an object' });
  } else {
    const source = def.source as Record<string, unknown>;
    if (!source.compiler || typeof source.compiler !== 'string') {
      errors.push({ path: 'source.compiler', message: 'source.compiler is required' });
    }
    if (!source.runtime || typeof source.runtime !== 'string') {
      errors.push({ path: 'source.runtime', message: 'source.runtime is required' });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Converts a CustomNodeDefinition to a HandleDefinition array for inputs
 */
export function customNodeInputsToHandles(inputs: CustomNodeInput[]): HandleDefinition[] {
  return inputs.map(input => ({
    id: input.id,
    name: input.name,
    type: (input.type as HandleDataType) || 'any',
    required: input.required,
    multiple: input.multi,
    description: input.description,
  }));
}

/**
 * Converts a CustomNodeDefinition to a HandleDefinition array for outputs
 */
export function customNodeOutputsToHandles(outputs: CustomNodeOutput[]): HandleDefinition[] {
  return outputs.map(output => ({
    id: output.id,
    name: output.name,
    type: (output.type as HandleDataType) || 'any',
    description: output.description,
  }));
}

/**
 * Converts a CustomNodeProperty to a PropertyDefinition
 */
export function customNodePropertiesToDefinitions(
  properties: CustomNodeProperty[]
): PropertyDefinition[] {
  return properties.map(prop => ({
    id: prop.id,
    name: prop.name,
    type: prop.type as PropertyDefinition['type'],
    default: prop.defaultValue,
    required: prop.required,
    placeholder: prop.placeholder,
    description: prop.description,
    options: prop.options?.map(o => ({
      value: o.value,
      label: o.label,
    })),
  }));
}
