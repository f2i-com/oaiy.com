/**
 * User-Friendly Error Messages
 *
 * Maps technical error codes and messages to user-friendly explanations.
 * This layer converts internal errors into messages that non-technical users can understand.
 *
 * @example
 * ```typescript
 * try {
 *   await runtime.execute(graph);
 * } catch (error) {
 *   const userMessage = formatErrorForUser(error);
 *   showToast(userMessage.title, userMessage.description);
 * }
 * ```
 */

import {
  OAIYError,
  CompilationError,
  CycleDetectedError,
  UnknownNodeTypeError,
  InvalidLoopError,
  RuntimeError,
  AbortError,
  MissingInputError,
  ExternalApiError,
  ModuleError,
  ModuleLoadError,
  ValidationError,
} from '../errors';

/**
 * User-friendly error message structure
 */
export interface UserFriendlyError {
  /** Short title for the error (suitable for toast headers) */
  title: string;
  /** Longer description explaining what happened */
  description: string;
  /** Suggested action the user can take to resolve the issue */
  suggestion?: string;
  /** Technical details for debugging (hidden by default) */
  technicalDetails?: string;
  /** Error severity for UI styling */
  severity: 'error' | 'warning' | 'info';
}

/**
 * Error message templates for common error codes
 */
const ERROR_MESSAGES: Record<string, Omit<UserFriendlyError, 'technicalDetails'>> = {
  // Compilation errors
  CYCLE_DETECTED: {
    title: 'Circular Connection Detected',
    description: 'Your workflow has nodes connected in a loop that would run forever.',
    suggestion: 'Check for connections that loop back on themselves and remove them.',
    severity: 'error',
  },
  UNKNOWN_NODE_TYPE: {
    title: 'Unknown Node Type',
    description: 'A node in your workflow is not recognized. It may be from a module that is not installed.',
    suggestion: 'Try removing the node and re-adding it from the palette, or install the required module.',
    severity: 'error',
  },
  INVALID_LOOP: {
    title: 'Invalid Loop Structure',
    description: 'The loop in your workflow is not set up correctly.',
    suggestion: 'Make sure each Loop Start has a matching Loop End connected to it.',
    severity: 'error',
  },
  COMPILATION_ERROR: {
    title: 'Workflow Build Failed',
    description: 'There was a problem preparing your workflow to run.',
    suggestion: 'Check the workflow for incomplete connections or missing required fields.',
    severity: 'error',
  },

  // Runtime errors
  ABORTED: {
    title: 'Workflow Stopped',
    description: 'The workflow was stopped before it finished.',
    severity: 'info',
  },
  MISSING_INPUT: {
    title: 'Missing Input',
    description: 'A node is waiting for input that was not provided.',
    suggestion: 'Connect an input to the node or provide a default value.',
    severity: 'error',
  },
  EXTERNAL_API_ERROR: {
    title: 'External Service Error',
    description: 'A service your workflow depends on returned an error.',
    suggestion: 'Check your API keys and network connection. The service may be temporarily unavailable.',
    severity: 'error',
  },
  RUNTIME_ERROR: {
    title: 'Workflow Error',
    description: 'Something went wrong while running your workflow.',
    suggestion: 'Check the logs for more details about what failed.',
    severity: 'error',
  },

  // Module errors
  MODULE_LOAD_ERROR: {
    title: 'Module Failed to Load',
    description: 'A required module could not be loaded.',
    suggestion: 'Try restarting the application. If the problem persists, reinstall the module.',
    severity: 'error',
  },
  MODULE_VALIDATION_ERROR: {
    title: 'Invalid Module',
    description: 'A module has configuration errors and cannot be used.',
    suggestion: 'Check the module settings and ensure all required fields are filled in.',
    severity: 'error',
  },
  MODULE_DEPENDENCY_ERROR: {
    title: 'Missing Dependency',
    description: 'A module requires another module that is not installed.',
    suggestion: 'Install the required dependency module.',
    severity: 'error',
  },

  // Validation errors
  VALIDATION_ERROR: {
    title: 'Invalid Workflow',
    description: 'Your workflow has configuration issues that need to be fixed.',
    suggestion: 'Review the highlighted nodes and fix any issues.',
    severity: 'warning',
  },

  // Network/permission errors
  NETWORK_PERMISSION_DENIED: {
    title: 'Network Access Denied',
    description: 'The workflow tried to access a network address that is not allowed.',
    suggestion: 'Add the address to your whitelist in Settings > Security.',
    severity: 'error',
  },

  // Generic fallback
  UNKNOWN_ERROR: {
    title: 'Unexpected Error',
    description: 'Something unexpected happened.',
    suggestion: 'Try again. If the problem continues, check the logs for details.',
    severity: 'error',
  },
};

/**
 * Format an error for display to the user
 */
export function formatErrorForUser(error: unknown): UserFriendlyError {
  // Handle OAIYError and its subclasses
  if (error instanceof OAIYError) {
    return formatOAIYError(error);
  }

  // Handle standard Error objects
  if (error instanceof Error) {
    return formatStandardError(error);
  }

  // Handle unknown error types
  return {
    ...ERROR_MESSAGES.UNKNOWN_ERROR,
    technicalDetails: String(error),
  };
}

/**
 * Format a OAIYError for user display
 */
function formatOAIYError(error: OAIYError): UserFriendlyError {
  const template = ERROR_MESSAGES[error.code] || ERROR_MESSAGES.UNKNOWN_ERROR;

  // Build technical details from error properties
  let technicalDetails = `${error.name}: ${error.message}`;

  // Add node context if available
  if (error instanceof CompilationError && error.nodeId) {
    technicalDetails += `\nNode: ${error.nodeId}`;
    if (error.nodeType) {
      technicalDetails += ` (${error.nodeType})`;
    }
  }

  if (error instanceof RuntimeError && error.nodeId) {
    technicalDetails += `\nNode: ${error.nodeId}`;
  }

  if (error instanceof ModuleError) {
    technicalDetails += `\nModule: ${error.moduleId}`;
  }

  // Add specific details for certain error types
  if (error instanceof CycleDetectedError) {
    technicalDetails += `\nCycle: ${error.cycle}`;
  }

  if (error instanceof InvalidLoopError) {
    technicalDetails += `\nReason: ${error.reason}`;
  }

  if (error instanceof ExternalApiError) {
    technicalDetails += `\nService: ${error.service}`;
    if (error.statusCode) {
      technicalDetails += `\nStatus: ${error.statusCode}`;
    }
  }

  if (error instanceof ValidationError) {
    technicalDetails += `\nIssues:\n${error.issues.map(i => `  - ${i}`).join('\n')}`;
  }

  return {
    ...template,
    technicalDetails,
  };
}

/**
 * Format a standard Error for user display
 */
function formatStandardError(error: Error): UserFriendlyError {
  const message = error.message.toLowerCase();

  // Try to detect common error patterns
  if (message.includes('network') || message.includes('fetch') || message.includes('connection')) {
    return {
      title: 'Network Error',
      description: 'Could not connect to the required service.',
      suggestion: 'Check your internet connection and try again.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'error',
    };
  }

  if (message.includes('timeout') || message.includes('timed out')) {
    return {
      title: 'Request Timed Out',
      description: 'The operation took too long to complete.',
      suggestion: 'Try again. If using an AI service, the request may be too complex.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'error',
    };
  }

  if (message.includes('permission') || message.includes('denied') || message.includes('unauthorized')) {
    return {
      title: 'Permission Denied',
      description: 'You do not have permission to perform this action.',
      suggestion: 'Check your API keys and permissions.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'error',
    };
  }

  if (message.includes('not found') || message.includes('404')) {
    return {
      title: 'Not Found',
      description: 'The requested resource could not be found.',
      suggestion: 'Check that the file path or URL is correct.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'error',
    };
  }

  if (message.includes('api key') || message.includes('apikey') || message.includes('authentication')) {
    return {
      title: 'Authentication Failed',
      description: 'The API key is invalid or missing.',
      suggestion: 'Check your API key in Settings > API Keys.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'error',
    };
  }

  if (message.includes('rate limit') || message.includes('too many requests') || message.includes('429')) {
    return {
      title: 'Rate Limited',
      description: 'You have made too many requests in a short time.',
      suggestion: 'Wait a few minutes and try again.',
      technicalDetails: `${error.name}: ${error.message}`,
      severity: 'warning',
    };
  }

  // Fallback for unrecognized errors
  return {
    ...ERROR_MESSAGES.UNKNOWN_ERROR,
    description: error.message,
    technicalDetails: `${error.name}: ${error.message}`,
  };
}

/**
 * Get a short one-line error summary suitable for logging
 */
export function getErrorSummary(error: unknown): string {
  if (error instanceof OAIYError) {
    if (error instanceof CompilationError && error.nodeId) {
      return `[${error.code}] ${error.message} (node: ${error.nodeId})`;
    }
    return `[${error.code}] ${error.message}`;
  }

  if (error instanceof Error) {
    return `[${error.name}] ${error.message}`;
  }

  return String(error);
}

/**
 * Check if an error is user-facing (should be shown to the user)
 * vs internal (should only be logged)
 */
export function isUserFacingError(error: unknown): boolean {
  // AbortError is not really an "error" from the user's perspective
  if (error instanceof AbortError) {
    return false;
  }

  // All other OAIYErrors should be shown to users
  if (error instanceof OAIYError) {
    return true;
  }

  // Standard errors are usually user-facing
  if (error instanceof Error) {
    // Filter out some internal errors
    const message = error.message.toLowerCase();
    if (message.includes('internal') || message.includes('assertion')) {
      return false;
    }
    return true;
  }

  return true;
}
