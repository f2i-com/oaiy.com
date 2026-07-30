/**
 * Job Queue Type Definitions
 *
 * Defines the types for the worker-based job queue system that enables
 * background workflow execution, configurable concurrency, and robust abort handling.
 */

import type { WorkflowGraph, WorkflowInputs, LogEntry, ProjectSettings, LocalNetworkPermissionRequest, LocalNetworkPermissionResponse, Flow } from '../types';

/**
 * Pending AI request for Claude-as-AI pattern
 * Contains all the information needed for Claude to generate a response
 */
export interface PendingAIRequest {
  /** Unique token to identify this request for continuation */
  continueToken: string;
  /** The node ID that is awaiting AI response */
  nodeId: string;
  /** System prompt for the AI */
  systemPrompt: string;
  /** User prompt/message for the AI */
  userPrompt: string;
  /** Optional images for vision models */
  images?: string[];
  /** Message history for multi-turn conversations */
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  /** Timestamp when the request was created */
  createdAt: number;
}

/**
 * Job status lifecycle
 * - pending: Job is queued waiting to run
 * - running: Job is currently executing
 * - awaiting_ai: Job is paused waiting for external AI response (Claude-as-AI pattern)
 * - completed: Job finished successfully
 * - failed: Job encountered an error
 * - aborted: Job was manually cancelled
 */
export type JobStatus = 'pending' | 'running' | 'awaiting_ai' | 'completed' | 'failed' | 'aborted';

/**
 * Queue execution mode
 */
export type QueueMode = 'sequential' | 'parallel';

/**
 * Queue configuration options
 */
/**
 * Privacy retention policy for finished jobs in `JobManager.history`.
 *
 * Job history may include workflow inputs, results, and captured log
 * lines, all of which can contain user secrets, PII, or prompts the
 * user does not want to keep around. Round 5 reviewer #10.
 *
 *   * `'full'`     — keep inputs/result/logs verbatim (default).
 *   * `'redacted'` — keep job metadata (id, flowId, status, timing)
 *                   but strip inputs, result, graph, and logs.
 *   * `'none'`     — do not add finished jobs to history at all.
 */
export type JobHistoryRetention = 'full' | 'redacted' | 'none';

/**
 * Queue configuration options
 */
export interface JobConfig {
  /** Execution mode: sequential (one at a time) or parallel (multiple concurrent) */
  mode: QueueMode;
  /** Maximum concurrent jobs when in parallel mode (default: 1) */
  maxConcurrency: number;
  /**
   * How much of a finished job to retain in history. Defaults to
   * `'full'` for backwards compatibility — opt in to the privacy modes
   * from the project settings UI.
   */
  historyRetention?: JobHistoryRetention;
}

/**
 * Represents a job in the queue
 */
export interface Job {
  /** Unique job identifier */
  id: string;
  /** ID of the flow being executed */
  flowId: string;
  /** Name of the flow (for display) */
  flowName: string;
  /** The workflow graph to execute */
  graph: WorkflowGraph;
  /** Optional inputs for the workflow */
  inputs?: WorkflowInputs;
  /** Current job status */
  status: JobStatus;
  /** Job priority (higher = more urgent, used for queue ordering) */
  priority: number;
  /** Timestamp when job was submitted */
  submittedAt: number;
  /** Timestamp when job started executing */
  startedAt?: number;
  /** Timestamp when job completed/failed/aborted */
  completedAt?: number;
  /** Collected log entries from execution */
  logs: LogEntry[];
  /** Final result from workflow execution */
  result?: unknown;
  /** Error message if job failed */
  error?: string;
  /** Node outputs collected during execution */
  nodeOutputs?: Record<string, unknown>;
  /** Label of the currently executing node (for progress display) */
  currentNodeLabel?: string;
  /** Final results from the workflow (extracted from workflow_context) */
  results?: Record<string, unknown>;
  /** Whether this job is running in Claude-as-AI mode */
  useClaudeForAI?: boolean;
  /** Pending AI request when status is 'awaiting_ai' */
  pendingAIRequest?: PendingAIRequest;

  // -----------------------------------------------------------------------
  // Package execution context
  // -----------------------------------------------------------------------
  // When a job comes from an installed package, these carry the package
  // identity and the strict subset of permissions the user granted at
  // install time. The JobManager forwards them into
  // `runtime.setFlowContext(flowId, packageId)` and
  // `runtime.setPackagePermissions(grantedPermissions)` so the runtime's
  // fail-closed permission gate kicks in. Local trusted flows submit
  // without these and run with full privileges (the historical default).

  /** Identifier of the package this flow belongs to, or undefined for trusted local flows. */
  packageId?: string;
  /** Permissions granted by the user at package-install time. */
  grantedPermissions?: string[];
  /** Trust level reported by the package installer (informational). */
  packageTrustLevel?: 'untrusted' | 'trusted' | 'verified' | 'blocked';
}

/**
 * Messages sent from main thread to worker
 */
export type WorkerInboundMessage =
  | {
      type: 'EXECUTE';
      jobId: string;
      graph: WorkflowGraph;
      inputs?: WorkflowInputs;
      settings: ProjectSettings;
      availableFlows: Flow[];
      /** Enable Claude-as-AI mode - AI nodes will yield instead of calling APIs */
      useClaudeForAI?: boolean;
    }
  | { type: 'ABORT' }
  | {
      type: 'DB_RESPONSE';
      requestId: string;
      result?: unknown;
      error?: string;
    }
  | {
      type: 'NETWORK_PERMISSION_RESPONSE';
      requestId: string;
      response: LocalNetworkPermissionResponse;
    }
  | {
      /** Continue execution with Claude's AI response */
      type: 'AI_CONTINUE';
      /** The continue token from the pending AI request */
      continueToken: string;
      /** Claude's response to use as the AI output */
      response: string;
    };

/**
 * Messages sent from worker to main thread
 */
export type WorkerOutboundMessage =
  | {
      type: 'LOG';
      jobId: string;
      log: Omit<LogEntry, 'id' | 'timestamp'>;
    }
  | {
      type: 'STATUS';
      jobId: string;
      nodeId: string;
      status: 'running' | 'completed' | 'error';
    }
  | {
      type: 'TOKEN';
      jobId: string;
      nodeId: string;
      token: string;
    }
  | {
      type: 'IMAGE';
      jobId: string;
      nodeId: string;
      imageUrl: string;
    }
  | {
      type: 'RESULT';
      jobId: string;
      output: unknown;
      nodeOutputs: Record<string, unknown>;
    }
  | {
      type: 'ERROR';
      jobId: string;
      error: string;
    }
  | {
      type: 'DONE';
      jobId: string;
    }
  | {
      type: 'DB_REQUEST';
      requestId: string;
      operation: string;
      collectionName?: string;
      tableName?: string;
      data?: unknown;
      filter?: unknown;
      whereClause?: string;
      rawSql?: string;
      params?: unknown[];
      limit?: number;
    }
  | {
      type: 'NETWORK_PERMISSION_REQUEST';
      requestId: string;
      request: LocalNetworkPermissionRequest;
    }
  | {
      /** AI node is yielding, waiting for external AI response */
      type: 'AI_YIELD';
      jobId: string;
      /** The pending AI request details */
      request: PendingAIRequest;
    };

/**
 * Callback for job state changes
 */
export type JobStateCallback = (jobs: Job[]) => void;

/**
 * Callback for individual job log entries
 */
export type JobLogCallback = (jobId: string, log: LogEntry) => void;

/**
 * Callback for node status updates
 */
export type NodeStatusCallback = (
  jobId: string,
  nodeId: string,
  status: 'running' | 'completed' | 'error'
) => void;

/**
 * Callback for streaming token updates
 */
export type StreamTokenCallback = (jobId: string, nodeId: string, token: string) => void;

/**
 * Callback for image updates
 */
export type ImageUpdateCallback = (jobId: string, nodeId: string, imageUrl: string) => void;

/**
 * Callback for AI yield notifications (Claude-as-AI pattern)
 */
export type AIYieldCallback = (jobId: string, request: PendingAIRequest) => void;

/**
 * Active job tracking (internal to JobManager)
 */
export interface ActiveJobEntry {
  job: Job;
  worker: Worker;
  abortTimeoutId?: ReturnType<typeof setTimeout>;
}
