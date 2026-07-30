/**
 * OAIY Error Hierarchy
 *
 * Provides typed errors for better error handling, debugging, and user feedback.
 * All OAIY errors extend from OAIYError which provides a consistent interface.
 *
 * @example
 * ```typescript
 * try {
 *   compiler.compile(graph);
 * } catch (error) {
 *   if (error instanceof CycleDetectedError) {
 *     console.log(`Cycle found: ${error.cycle}`);
 *   } else if (error instanceof CompilationError) {
 *     console.log(`Compilation failed at node: ${error.nodeId}`);
 *   }
 * }
 * ```
 */

/**
 * Base error class for all OAIY errors.
 * Provides a consistent interface with error codes for programmatic handling.
 */
export class OAIYError extends Error {
  /** Error code for programmatic handling */
  public readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = 'OAIYError';
    this.code = code;
    // Maintains proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert error to a JSON-serializable object
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      stack: this.stack,
    };
  }
}

// ============================================================================
// Compilation Errors
// ============================================================================

/**
 * Base error for compilation-related failures.
 * Includes optional node context for debugging.
 */
export class CompilationError extends OAIYError {
  /** ID of the node where compilation failed (if applicable) */
  public readonly nodeId?: string;
  /** Type of the node where compilation failed (if applicable) */
  public readonly nodeType?: string;

  constructor(message: string, nodeId?: string, nodeType?: string, code = 'COMPILATION_ERROR') {
    super(message, code);
    this.name = 'CompilationError';
    this.nodeId = nodeId;
    this.nodeType = nodeType;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      nodeId: this.nodeId,
      nodeType: this.nodeType,
    };
  }
}

/**
 * Error thrown when a circular dependency is detected in the workflow graph.
 */
export class CycleDetectedError extends CompilationError {
  /** Description of the cycle (e.g., "nodeA -> nodeB -> nodeA") */
  public readonly cycle: string;

  constructor(cycle: string) {
    super(`Circular dependency detected: ${cycle}`, undefined, undefined, 'CYCLE_DETECTED');
    this.name = 'CycleDetectedError';
    this.cycle = cycle;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      cycle: this.cycle,
    };
  }
}

/**
 * Error thrown when a node type is not recognized by any module.
 */
export class UnknownNodeTypeError extends CompilationError {
  constructor(nodeType: string, nodeId?: string) {
    super(`Unknown node type: ${nodeType}`, nodeId, nodeType, 'UNKNOWN_NODE_TYPE');
    this.name = 'UnknownNodeTypeError';
  }
}

/**
 * Error thrown when a loop structure is malformed.
 */
export class InvalidLoopError extends CompilationError {
  /** The loop start node ID */
  public readonly loopStartId: string;
  /** Specific reason the loop is invalid */
  public readonly reason: string;

  constructor(loopStartId: string, reason: string) {
    super(`Invalid loop structure at ${loopStartId}: ${reason}`, loopStartId, 'loop_start', 'INVALID_LOOP');
    this.name = 'InvalidLoopError';
    this.loopStartId = loopStartId;
    this.reason = reason;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      loopStartId: this.loopStartId,
      reason: this.reason,
    };
  }
}

// ============================================================================
// Runtime Errors
// ============================================================================

/**
 * Base error for runtime execution failures.
 */
export class RuntimeError extends OAIYError {
  /** ID of the node where execution failed (if applicable) */
  public readonly nodeId?: string;

  constructor(message: string, nodeId?: string, code = 'RUNTIME_ERROR') {
    super(message, code);
    this.name = 'RuntimeError';
    this.nodeId = nodeId;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      nodeId: this.nodeId,
    };
  }
}

/**
 * Error thrown when workflow execution is aborted by user.
 */
export class AbortError extends RuntimeError {
  constructor(nodeId?: string) {
    super('Workflow execution aborted', nodeId, 'ABORTED');
    this.name = 'AbortError';
  }
}

/**
 * Error thrown when a required input is missing during execution.
 */
export class MissingInputError extends RuntimeError {
  /** Name of the missing input */
  public readonly inputName: string;

  constructor(inputName: string, nodeId?: string) {
    super(`Missing required input: ${inputName}`, nodeId, 'MISSING_INPUT');
    this.name = 'MissingInputError';
    this.inputName = inputName;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      inputName: this.inputName,
    };
  }
}

/**
 * Error thrown when an external API call fails.
 */
export class ExternalApiError extends RuntimeError {
  /** The API or service that failed */
  public readonly service: string;
  /** HTTP status code if applicable */
  public readonly statusCode?: number;

  constructor(service: string, message: string, statusCode?: number, nodeId?: string) {
    super(`${service} API error: ${message}`, nodeId, 'EXTERNAL_API_ERROR');
    this.name = 'ExternalApiError';
    this.service = service;
    this.statusCode = statusCode;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      service: this.service,
      statusCode: this.statusCode,
    };
  }
}

// ============================================================================
// Module Errors
// ============================================================================

/**
 * Base error for module-related failures.
 */
export class ModuleError extends OAIYError {
  /** ID of the module that caused the error */
  public readonly moduleId: string;

  constructor(message: string, moduleId: string, code = 'MODULE_ERROR') {
    super(message, code);
    this.name = 'ModuleError';
    this.moduleId = moduleId;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      moduleId: this.moduleId,
    };
  }
}

/**
 * Error thrown when module validation fails.
 */
export class ModuleValidationError extends ModuleError {
  /** Validation errors found */
  public readonly validationErrors: Array<{ path: string; message: string }>;

  constructor(moduleId: string, errors: Array<{ path: string; message: string }>) {
    const errorSummary = errors.map(e => `${e.path}: ${e.message}`).join(', ');
    super(`Module validation failed: ${errorSummary}`, moduleId, 'MODULE_VALIDATION_ERROR');
    this.name = 'ModuleValidationError';
    this.validationErrors = errors;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      validationErrors: this.validationErrors,
    };
  }
}

/**
 * Error thrown when a module dependency is not found.
 */
export class ModuleDependencyError extends ModuleError {
  /** The missing dependency module ID */
  public readonly dependencyId: string;

  constructor(moduleId: string, dependencyId: string) {
    super(`Module "${moduleId}" requires "${dependencyId}" which is not loaded`, moduleId, 'MODULE_DEPENDENCY_ERROR');
    this.name = 'ModuleDependencyError';
    this.dependencyId = dependencyId;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      dependencyId: this.dependencyId,
    };
  }
}

/**
 * Error thrown when a module fails to load.
 */
export class ModuleLoadError extends ModuleError {
  /** The underlying error that caused the load failure */
  public readonly cause?: Error;

  constructor(moduleId: string, message: string, cause?: Error) {
    super(`Failed to load module "${moduleId}": ${message}`, moduleId, 'MODULE_LOAD_ERROR');
    this.name = 'ModuleLoadError';
    this.cause = cause;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      cause: this.cause?.message,
    };
  }
}

// ============================================================================
// Validation Errors
// ============================================================================

/**
 * Error thrown when workflow graph validation fails.
 */
export class ValidationError extends OAIYError {
  /** List of validation issues found */
  public readonly issues: string[];

  constructor(issues: string[]) {
    super(`Workflow validation failed: ${issues.join('; ')}`, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.issues = issues;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      issues: this.issues,
    };
  }
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if an error is a OAIYError
 */
export function isOAIYError(error: unknown): error is OAIYError {
  return error instanceof OAIYError;
}

/**
 * Check if an error is a CompilationError
 */
export function isCompilationError(error: unknown): error is CompilationError {
  return error instanceof CompilationError;
}

/**
 * Check if an error is a RuntimeError
 */
export function isRuntimeError(error: unknown): error is RuntimeError {
  return error instanceof RuntimeError;
}

/**
 * Check if an error is a ModuleError
 */
export function isModuleError(error: unknown): error is ModuleError {
  return error instanceof ModuleError;
}
