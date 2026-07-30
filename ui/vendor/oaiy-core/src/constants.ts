/**
 * Shared constants for the OAIY workflow engine.
 *
 * Centralizing these values ensures consistency across the codebase
 * and makes tuning for different environments easier.
 */

// ============================================================================
// Compiler & Graph Traversal
// ============================================================================

/**
 * Maximum iterations for graph traversal algorithms.
 * Prevents infinite loops in cycle detection and node sorting.
 */
export const MAX_GRAPH_ITERATIONS = 10000;

/**
 * Maximum iterations for loop end detection.
 * Smaller than MAX_GRAPH_ITERATIONS since loops should be found quickly.
 */
export const MAX_LOOP_DETECTION_ITERATIONS = 1000;

// ============================================================================
// Job Queue
// ============================================================================

/**
 * Maximum number of completed jobs to keep in history.
 * Older jobs are removed when this limit is exceeded.
 */
export const MAX_JOB_HISTORY_SIZE = 50;

/**
 * Timeout (ms) for force-aborting a job after abort is requested.
 * If a job doesn't respond to abort within this time, it's forcefully terminated.
 */
export const FORCE_ABORT_TIMEOUT_MS = 3000;

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Default maximum entries for BoundedMap (agent memory).
 * Prevents unbounded memory growth.
 */
export const DEFAULT_MAX_MEMORY_ENTRIES = 1000;

/**
 * Default maximum size (bytes) for a single value in BoundedMap.
 * Prevents memory bloat from large objects. Default: 1MB
 */
export const DEFAULT_MAX_VALUE_SIZE_BYTES = 1024 * 1024;

/**
 * Fallback size estimate (bytes) for objects that can't be stringified.
 * Used for circular references or objects with custom serialization.
 */
export const CIRCULAR_REF_SIZE_ESTIMATE = 1024;

/**
 * Default size estimate (bytes) for unknown value types.
 */
export const DEFAULT_VALUE_SIZE_ESTIMATE = 64;

// ============================================================================
// File Handling
// ============================================================================

/**
 * Maximum file size (bytes) to keep in memory.
 * Files larger than this are streamed or handled differently.
 * Default: 10MB
 */
export const MAX_IN_MEMORY_FILE_SIZE = 10 * 1024 * 1024;

// ============================================================================
// Workflow Execution
// ============================================================================

/**
 * Default timeout (ms) for workflow execution.
 * Can be overridden per-workflow.
 */
export const DEFAULT_WORKFLOW_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Maximum depth for nested subflow/macro calls.
 * Prevents stack overflow from recursive workflows.
 */
export const MAX_SUBFLOW_DEPTH = 50;

/**
 * Maximum iterations for a workflow loop node.
 * Prevents runaway loops from consuming resources.
 */
export const MAX_WORKFLOW_LOOP_ITERATIONS = 1000;

// ============================================================================
// Logging & Debug
// ============================================================================

/**
 * Maximum length for logged values before truncation.
 * Prevents log bloat from large payloads.
 */
export const MAX_LOG_VALUE_LENGTH = 500;

/**
 * Maximum log entries to keep per job.
 */
export const MAX_LOG_ENTRIES_PER_JOB = 1000;
