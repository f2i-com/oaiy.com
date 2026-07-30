// FlowPlan DSL Types
// A simplified JSON format for AI-generated workflows

import { isValidStepType } from './bundled-modules';

/**
 * Input definition for a FlowPlan
 */
export interface FlowPlanInput {
  name: string;
  type: 'text' | 'folder_path' | 'file_path' | 'number' | 'url';
  description?: string;
  default?: string;
}

/**
 * Collection definition for batch processing
 */
export interface FlowPlanCollection {
  name: string;
  type: 'folder_files' | 'list' | 'range';
  from: string;  // Reference to input or literal value
  include?: string[];  // Glob patterns for folder_files
  exclude?: string[];
  recursive?: boolean;
  max?: number;
}

/**
 * Loop configuration for iterating over collections
 */
export interface FlowPlanLoop {
  mode: 'for_each' | 'count' | 'while';
  over: string;  // Collection name or count value
  itemAlias: string;  // Variable name for current item
  steps: FlowPlanStep[];
}

/**
 * Base step interface
 */
export interface FlowPlanStepBase {
  id: string;
  type: string;
}

/**
 * File read step
 */
export interface FileReadStep extends FlowPlanStepBase {
  type: 'file_read';
  path: string;  // Template string
  as: 'text' | 'base64';
}

/**
 * File write step
 */
export interface FileWriteStep extends FlowPlanStepBase {
  type: 'file_write';
  path: string;  // Template string for output path
  content: string;  // Template reference to content
  contentType?: 'text' | 'base64';
}

/**
 * Template step - string interpolation
 */
export interface TemplateStep extends FlowPlanStepBase {
  type: 'template';
  template: string;
  inputs?: Record<string, string>;  // Named inputs for template
}

/**
 * AI LLM step - text generation
 */
export interface AILLMStep extends FlowPlanStepBase {
  type: 'ai_llm';
  prompt: string;  // Template string
  systemPrompt?: string;
  image?: string;  // Optional image reference for vision
}

/**
 * AI Image step - image generation
 */
export interface AIImageStep extends FlowPlanStepBase {
  type: 'ai_image';
  prompt: string;
  image?: string;  // Optional source image for img2img
  model?: string;  // Model hint (flux, dall-e-3, etc.)
}

/**
 * Condition step - branching logic
 */
export interface ConditionStep extends FlowPlanStepBase {
  type: 'condition';
  input: string;  // Template reference
  operator: 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater' | 'less' | 'is_empty' | 'not_empty';
  value: string;
}

/**
 * HTTP request step
 */
export interface HTTPRequestStep extends FlowPlanStepBase {
  type: 'http_request';
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  url: string;
  body?: string;
  headers?: Record<string, string>;
}

/**
 * Database store step
 */
export interface DatabaseStoreStep extends FlowPlanStepBase {
  type: 'database_store';
  collection: string;
  data: string;  // Template reference to data
}

/**
 * Log step - for logging messages during execution
 */
export interface LogStep extends FlowPlanStepBase {
  type: 'log';
  message: string;
  label?: string;
}

/**
 * Output step - marks the final output of a workflow
 */
export interface OutputStep extends FlowPlanStepBase {
  type: 'output';
  result: string;  // Template reference to the result value
  label?: string;
}

/**
 * Logic block step - custom code for data transformation
 */
export interface LogicBlockStep extends FlowPlanStepBase {
  type: 'logic_block' | 'code';
  code: string;  // JavaScript code to execute
  input?: string;  // Template reference to input data
}

/**
 * Generic step for dynamically loaded node types
 * Used for nodes that have flowplan config but aren't in the core types
 */
export interface GenericStep extends FlowPlanStepBase {
  type: string;
  [key: string]: unknown;
}

/**
 * Union of all step types
 */
export type FlowPlanStep =
  | FileReadStep
  | FileWriteStep
  | TemplateStep
  | AILLMStep
  | AIImageStep
  | ConditionStep
  | HTTPRequestStep
  | DatabaseStoreStep
  | LogStep
  | OutputStep
  | LogicBlockStep
  | GenericStep;

/**
 * Complete FlowPlan structure
 */
export interface FlowPlan {
  name: string;
  description: string;
  inputs: FlowPlanInput[];
  collections?: FlowPlanCollection[];
  loop?: FlowPlanLoop;
  steps?: FlowPlanStep[];  // For non-loop workflows
}

/**
 * Validation result for FlowPlan
 */
export interface FlowPlanValidationResult {
  valid: boolean;
  errors: FlowPlanValidationError[];
  warnings: FlowPlanValidationWarning[];
}

export interface FlowPlanValidationError {
  path: string;  // JSON path to error
  message: string;
  code: string;
}

export interface FlowPlanValidationWarning {
  path: string;
  message: string;
  code: string;
}

/**
 * Valid step types for validation
 */
/**
 * Core step types that are always available.
 * Additional step types can be added dynamically via node flowplan configs.
 */
export const CORE_STEP_TYPES = [
  'file_read',
  'file_write',
  'template',
  'ai_llm',
  'ai_image',
  'condition',
  'http_request',
  'database_store',
  'log',
  'output',
  'logic_block',
  'code',
] as const;

/**
 * Valid step types for validation.
 * This is the base list - use getValidStepTypes() for the full dynamic list.
 * @deprecated Use getValidStepTypes() for dynamic step type discovery
 */
export const VALID_STEP_TYPES = CORE_STEP_TYPES;

/**
 * Valid input types
 */
export const VALID_INPUT_TYPES = [
  'text',
  'folder_path',
  'file_path',
  'number',
  'url',
] as const;

/**
 * Valid collection types
 */
export const VALID_COLLECTION_TYPES = [
  'folder_files',
  'list',
  'range',
] as const;

/**
 * Validate a FlowPlan structure
 */
export function validateFlowPlan(plan: unknown): FlowPlanValidationResult {
  const errors: FlowPlanValidationError[] = [];
  const warnings: FlowPlanValidationWarning[] = [];

  if (!plan || typeof plan !== 'object') {
    errors.push({ path: '', message: 'FlowPlan must be an object', code: 'INVALID_TYPE' });
    return { valid: false, errors, warnings };
  }

  const p = plan as Record<string, unknown>;

  // Check required fields
  if (!p.name || typeof p.name !== 'string') {
    errors.push({ path: 'name', message: 'FlowPlan must have a name', code: 'MISSING_NAME' });
  }

  if (!p.description || typeof p.description !== 'string') {
    errors.push({ path: 'description', message: 'FlowPlan must have a description', code: 'MISSING_DESCRIPTION' });
  }

  // Validate inputs
  if (!Array.isArray(p.inputs)) {
    errors.push({ path: 'inputs', message: 'inputs must be an array', code: 'INVALID_INPUTS' });
  } else {
    p.inputs.forEach((input: unknown, i: number) => {
      if (!input || typeof input !== 'object') {
        errors.push({ path: `inputs[${i}]`, message: 'Input must be an object', code: 'INVALID_INPUT' });
        return;
      }
      const inp = input as Record<string, unknown>;
      if (!inp.name || typeof inp.name !== 'string') {
        errors.push({ path: `inputs[${i}].name`, message: 'Input must have a name', code: 'MISSING_INPUT_NAME' });
      }
      if (!inp.type || !VALID_INPUT_TYPES.includes(inp.type as typeof VALID_INPUT_TYPES[number])) {
        errors.push({
          path: `inputs[${i}].type`,
          message: `Invalid input type. Must be one of: ${VALID_INPUT_TYPES.join(', ')}`,
          code: 'INVALID_INPUT_TYPE'
        });
      }
    });
  }

  // Validate collections
  if (p.collections) {
    if (!Array.isArray(p.collections)) {
      errors.push({ path: 'collections', message: 'collections must be an array', code: 'INVALID_COLLECTIONS' });
    } else {
      p.collections.forEach((coll: unknown, i: number) => {
        if (!coll || typeof coll !== 'object') {
          errors.push({ path: `collections[${i}]`, message: 'Collection must be an object', code: 'INVALID_COLLECTION' });
          return;
        }
        const c = coll as Record<string, unknown>;
        if (!c.name || typeof c.name !== 'string') {
          errors.push({ path: `collections[${i}].name`, message: 'Collection must have a name', code: 'MISSING_COLLECTION_NAME' });
        }
        if (!c.type || !VALID_COLLECTION_TYPES.includes(c.type as typeof VALID_COLLECTION_TYPES[number])) {
          errors.push({
            path: `collections[${i}].type`,
            message: `Invalid collection type. Must be one of: ${VALID_COLLECTION_TYPES.join(', ')}`,
            code: 'INVALID_COLLECTION_TYPE'
          });
        }
        if (!c.from || typeof c.from !== 'string') {
          errors.push({ path: `collections[${i}].from`, message: 'Collection must have a from field', code: 'MISSING_COLLECTION_FROM' });
        }
      });
    }
  }

  // Must have either loop or steps
  if (!p.loop && !p.steps) {
    errors.push({ path: '', message: 'FlowPlan must have either loop or steps', code: 'MISSING_STEPS' });
  }

  // Validate loop
  if (p.loop) {
    if (typeof p.loop !== 'object') {
      errors.push({ path: 'loop', message: 'loop must be an object', code: 'INVALID_LOOP' });
    } else {
      const loop = p.loop as Record<string, unknown>;
      if (!loop.mode || !['for_each', 'count', 'while'].includes(loop.mode as string)) {
        errors.push({ path: 'loop.mode', message: 'loop.mode must be for_each, count, or while', code: 'INVALID_LOOP_MODE' });
      }
      if (!loop.over || typeof loop.over !== 'string') {
        errors.push({ path: 'loop.over', message: 'loop.over must be a string', code: 'MISSING_LOOP_OVER' });
      }
      if (!loop.itemAlias || typeof loop.itemAlias !== 'string') {
        errors.push({ path: 'loop.itemAlias', message: 'loop.itemAlias must be a string', code: 'MISSING_LOOP_ALIAS' });
      }
      if (!Array.isArray(loop.steps)) {
        errors.push({ path: 'loop.steps', message: 'loop.steps must be an array', code: 'INVALID_LOOP_STEPS' });
      } else {
        validateSteps(loop.steps, 'loop.steps', errors, warnings);
      }
    }
  }

  // Validate steps (non-loop)
  if (p.steps) {
    if (!Array.isArray(p.steps)) {
      errors.push({ path: 'steps', message: 'steps must be an array', code: 'INVALID_STEPS' });
    } else {
      validateSteps(p.steps, 'steps', errors, warnings);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate an array of steps
 */
function validateSteps(
  steps: unknown[],
  basePath: string,
  errors: FlowPlanValidationError[],
  warnings: FlowPlanValidationWarning[]
): void {
  const stepIds = new Set<string>();

  steps.forEach((step: unknown, i: number) => {
    if (!step || typeof step !== 'object') {
      errors.push({ path: `${basePath}[${i}]`, message: 'Step must be an object', code: 'INVALID_STEP' });
      return;
    }

    const s = step as Record<string, unknown>;

    // Check id
    if (!s.id || typeof s.id !== 'string') {
      errors.push({ path: `${basePath}[${i}].id`, message: 'Step must have an id', code: 'MISSING_STEP_ID' });
    } else {
      if (stepIds.has(s.id)) {
        errors.push({ path: `${basePath}[${i}].id`, message: `Duplicate step id: ${s.id}`, code: 'DUPLICATE_STEP_ID' });
      }
      stepIds.add(s.id);
    }

    // Check type - use dynamic validation from bundled modules
    // Falls back to core types if no modules are registered
    if (!s.type || typeof s.type !== 'string') {
      errors.push({ path: `${basePath}[${i}].type`, message: 'Step must have a type', code: 'MISSING_STEP_TYPE' });
    } else {
      const isCoreType = CORE_STEP_TYPES.includes(s.type as typeof CORE_STEP_TYPES[number]);
      const isDynamicType = isValidStepType(s.type);
      if (!isCoreType && !isDynamicType) {
        errors.push({
          path: `${basePath}[${i}].type`,
          message: `Invalid step type "${s.type}". Core types: ${CORE_STEP_TYPES.join(', ')}. Use a valid node type from available modules.`,
          code: 'INVALID_STEP_TYPE'
        });
      }
    }

    // Type-specific validation
    switch (s.type) {
      case 'file_read':
        if (!s.path) errors.push({ path: `${basePath}[${i}].path`, message: 'file_read requires path', code: 'MISSING_PATH' });
        if (!s.as || !['text', 'base64'].includes(s.as as string)) {
          warnings.push({ path: `${basePath}[${i}].as`, message: 'file_read.as should be text or base64, defaulting to text', code: 'MISSING_AS' });
        }
        break;

      case 'file_write':
        if (!s.path) errors.push({ path: `${basePath}[${i}].path`, message: 'file_write requires path', code: 'MISSING_PATH' });
        if (!s.content) errors.push({ path: `${basePath}[${i}].content`, message: 'file_write requires content', code: 'MISSING_CONTENT' });
        break;

      case 'template':
        if (!s.template) errors.push({ path: `${basePath}[${i}].template`, message: 'template requires template string', code: 'MISSING_TEMPLATE' });
        break;

      case 'ai_llm':
        if (!s.prompt) errors.push({ path: `${basePath}[${i}].prompt`, message: 'ai_llm requires prompt', code: 'MISSING_PROMPT' });
        break;

      case 'ai_image':
        if (!s.prompt) errors.push({ path: `${basePath}[${i}].prompt`, message: 'ai_image requires prompt', code: 'MISSING_PROMPT' });
        break;

      case 'condition':
        if (!s.input) errors.push({ path: `${basePath}[${i}].input`, message: 'condition requires input', code: 'MISSING_INPUT' });
        if (!s.operator) errors.push({ path: `${basePath}[${i}].operator`, message: 'condition requires operator', code: 'MISSING_OPERATOR' });
        break;

      case 'http_request':
        if (!s.url) errors.push({ path: `${basePath}[${i}].url`, message: 'http_request requires url', code: 'MISSING_URL' });
        break;

      case 'database_store':
        if (!s.collection) errors.push({ path: `${basePath}[${i}].collection`, message: 'database_store requires collection', code: 'MISSING_COLLECTION' });
        break;

      case 'log':
        if (!s.message) errors.push({ path: `${basePath}[${i}].message`, message: 'log requires message', code: 'MISSING_MESSAGE' });
        break;

      case 'output':
        if (!s.result) errors.push({ path: `${basePath}[${i}].result`, message: 'output requires result', code: 'MISSING_RESULT' });
        break;

      case 'logic_block':
      case 'code':
        if (!s.code) errors.push({ path: `${basePath}[${i}].code`, message: 'logic_block requires code', code: 'MISSING_CODE' });
        break;

      default:
        // For dynamically loaded step types, just ensure the step has required base fields
        // More specific validation can be added via node definitions
        break;
    }
  });
}

/**
 * Parse template references from a string
 * Returns array of references like ["input.name", "step1.output"]
 */
export function parseTemplateReferences(template: string): string[] {
  const matches = template.match(/\{\{([^}]+)\}\}/g) || [];
  return matches.map(m => m.slice(2, -2).trim());
}

/**
 * Check if a string contains template references
 */
export function hasTemplateReferences(str: string): boolean {
  return /\{\{[^}]+\}\}/.test(str);
}
