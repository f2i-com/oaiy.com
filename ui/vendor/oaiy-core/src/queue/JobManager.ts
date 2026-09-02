/**
 * Job Manager
 *
 * Manages workflow execution jobs in a queue, enabling:
 * - Background execution (browse flows while jobs run)
 * - Configurable concurrency (sequential or parallel)
 * - Robust abort with cooperative cancellation
 * - Job state tracking and history
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  Job,
  JobConfig,
  JobStatus,
  JobStateCallback,
  JobLogCallback,
  NodeStatusCallback,
  StreamTokenCallback,
  ImageUpdateCallback,
  AIYieldCallback,
  PendingAIRequest,
} from './types';
import type { WorkflowGraph, WorkflowInputs, LogEntry, Flow, ProjectSettings, LocalNetworkPermissionRequest, LocalNetworkPermissionResponse, DatabaseRequest, DatabaseResult } from '../types';
import { createRuntime, OAIYRuntime } from '../runtime';
import type { UntrustedWorkerFactory } from '../untrusted-executor';
import type { ModuleRegistry, LoadedModule } from '../module-types';
import { createLogger } from '../logger';
import { MAX_JOB_HISTORY_SIZE, FORCE_ABORT_TIMEOUT_MS, MAX_LOG_ENTRIES_PER_JOB } from '../constants';
import { metrics } from '../metrics';

const logger = createLogger('JobManager');

// Extended module registry interface that includes getAllModules
// This is implemented by ModuleLoader but not in the base interface
export interface ExtendedModuleRegistry extends ModuleRegistry {
  getAllModules(): LoadedModule[];
}

/**
 * Active job execution context
 */
interface ActiveJobContext {
  job: Job;
  runtime: OAIYRuntime;
  abortController: AbortController;
  forceAbortTimeoutId?: ReturnType<typeof setTimeout>;
}

/**
 * Options for creating a JobManager instance
 */
export interface JobManagerOptions {
  /** Handler for database operations */
  databaseHandler: (request: DatabaseRequest) => Promise<DatabaseResult>;
  /** Handler for network permission requests */
  networkPermissionHandler?: (request: LocalNetworkPermissionRequest) => Promise<LocalNetworkPermissionResponse>;
  /** Module registry for dynamic node support (must implement getAllModules) */
  moduleRegistry?: ExtendedModuleRegistry;
  /** Available flows for subflow execution */
  availableFlows?: Flow[];
  /** Package macros (higher priority than project macros) */
  packageMacros?: Flow[];
  /** Project settings */
  projectSettings?: ProjectSettings;
  /**
   * Project constants (constant key → value), including rehydrated secret
   * values. Forwarded to each job's runtime so modules can resolve an
   * `apiKeyConstant` name (e.g. OPENAI_API_KEY) to its value at run time.
   */
  projectConstants?: Record<string, string>;
  /**
   * Names of the constants that are SECRET. Threaded alongside `projectConstants` so each job's
   * runtime can require the high-risk `secrets:read` permission (vs `constants:read`) before an
   * untrusted package's `getConstant` resolves one — without it the runtime can't tell a stored
   * API key from a benign constant and a `network`-only package could exfiltrate every key.
   */
  secretConstantNames?: Set<string>;
  /** Initial queue configuration */
  config?: Partial<JobConfig>;
  /**
   * Tauri invoke broker. The runtime needs an explicit invoke function to
   * call native commands; without it, browser/terminal/database/etc.
   * native plugins fall back to "Tauri not detected". The desktop entry
   * point passes `@tauri-apps/api/core` `invoke` here so jobs submitted
   * via the HTTP API behave the same as jobs triggered from the UI.
   */
  tauriInvoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

  /**
   * Builds the Worker untrusted package workflows run in. Forwarded verbatim
   * to every runtime this manager creates; leaving it unset keeps oaiy-core's
   * default Blob-URL Worker. `ui/` passes a Zipp-backed Worker here.
   */
  untrustedWorkerFactory?: UntrustedWorkerFactory;

  /**
   * Whether a job's TRUSTED flow should also run in that Worker. A function is
   * consulted per job — a runtime is built for every job, so a Settings toggle
   * read here applies to the next run without a reload.
   */
  runTrustedFlowsInWorker?: boolean | (() => boolean);
}

/**
 * Default queue configuration
 */
const DEFAULT_CONFIG: JobConfig = {
  mode: 'sequential',
  maxConcurrency: 1,
};


export class JobManager {
  private queue: Job[] = [];
  private activeJobs: Map<string, ActiveJobContext> = new Map();
  private history: Job[] = [];
  private config: JobConfig;

  // Callbacks and handlers
  private databaseHandler: (request: DatabaseRequest) => Promise<DatabaseResult>;
  private networkPermissionHandler?: (request: LocalNetworkPermissionRequest) => Promise<LocalNetworkPermissionResponse>;
  private moduleRegistry?: ExtendedModuleRegistry;
  private availableFlows: Flow[] = [];
  private packageMacros: Flow[] = [];
  private projectSettings?: ProjectSettings;
  private projectConstants?: Record<string, string>;
  private secretConstantNames?: Set<string>;
  private tauriInvoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
  private untrustedWorkerFactory: UntrustedWorkerFactory | null = null;
  private runTrustedFlowsInWorker: boolean | (() => boolean) = false;

  // Subscribers
  private stateSubscribers: Set<JobStateCallback> = new Set();
  private logSubscribers: Set<JobLogCallback> = new Set();
  private nodeStatusSubscribers: Set<NodeStatusCallback> = new Set();
  private tokenSubscribers: Set<StreamTokenCallback> = new Set();
  private imageSubscribers: Set<ImageUpdateCallback> = new Set();
  private aiYieldSubscribers: Set<AIYieldCallback> = new Set();

  // Pending AI responses for Claude-as-AI pattern
  // Maps continueToken -> { resolve, reject } for the promise that the runtime is awaiting
  private pendingAIResponses: Map<string, {
    jobId: string;
    resolve: (response: string) => void;
    reject: (error: Error) => void;
  }> = new Map();

  constructor(options: JobManagerOptions) {
    this.databaseHandler = options.databaseHandler;
    this.networkPermissionHandler = options.networkPermissionHandler;
    this.moduleRegistry = options.moduleRegistry;
    this.availableFlows = options.availableFlows || [];
    this.packageMacros = options.packageMacros || [];
    this.projectSettings = options.projectSettings;
    this.projectConstants = options.projectConstants;
    this.secretConstantNames = options.secretConstantNames;
    this.tauriInvoke = options.tauriInvoke;
    this.untrustedWorkerFactory = options.untrustedWorkerFactory ?? null;
    this.runTrustedFlowsInWorker = options.runTrustedFlowsInWorker ?? false;
    this.config = { ...DEFAULT_CONFIG, ...options.config };
  }

  /**
   * Late-bind the Tauri invoke broker (e.g. when the desktop shell finishes
   * loading after the JobManager is already constructed).
   */
  setTauriInvoke(invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>): void {
    this.tauriInvoke = invoke;
  }

  /**
   * Update queue configuration
   */
  setConfig(config: Partial<JobConfig>): void {
    if (logger.isDebugEnabled) {
      logger.debug(` setConfig: before=${JSON.stringify(this.config)}, update=${JSON.stringify(config)}`);
    }
    this.config = { ...this.config, ...config };
    if (logger.isDebugEnabled) {
      logger.debug(` setConfig: after=${JSON.stringify(this.config)}`);
    }
    // Try to process queue in case concurrency increased
    this.processQueue();
  }

  /**
   * Get current configuration
   */
  getConfig(): JobConfig {
    return { ...this.config };
  }

  /**
   * Update available flows for subflow execution
   */
  setAvailableFlows(flows: Flow[]): void {
    this.availableFlows = flows;
  }

  /**
   * Set package macros (higher priority than project macros)
   */
  setPackageMacros(macros: Flow[]): void {
    this.packageMacros = macros;
  }

  /**
   * Clear package macros (call when closing a package)
   */
  clearPackageMacros(): void {
    this.packageMacros = [];
  }

  /**
   * Update project settings
   */
  setProjectSettings(settings: ProjectSettings): void {
    this.projectSettings = settings;
  }

  /**
   * Update project constants (constant key → value). Threaded into each job's
   * runtime so `ctx.getConstant(name)` resolves API-key constants at run time.
   */
  setProjectConstants(constants: Record<string, string>, secretNames?: Set<string>): void {
    this.projectConstants = constants;
    this.secretConstantNames = secretNames;
  }

  /**
   * Update module registry
   */
  setModuleRegistry(registry: ExtendedModuleRegistry): void {
    this.moduleRegistry = registry;
  }

  /**
   * Submit a new job to the queue
   *
   * For flows that come from an installed package, callers MUST pass
   * `packageContext` with the package id and the granted permission
   * subset — the JobManager forwards both into the runtime so the
   * fail-closed permission gate in `host.call` rejects anything outside
   * the granted set. Omitting the context for a package flow would run
   * it as a trusted local flow, which is exactly the bypass round 5 #1
   * flagged.
   *
   * @param flowId - ID of the flow to execute
   * @param flowName - Display name of the flow
   * @param graph - The workflow graph to execute
   * @param inputs - Optional inputs for the workflow
   * @param priority - Job priority (higher = more urgent)
   * @param useClaudeForAI - Enable Claude-as-AI mode where AI nodes yield for external responses
   * @param packageContext - Package identity + granted permissions, or undefined for trusted local flows
   */
  submit(
    flowId: string,
    flowName: string,
    graph: WorkflowGraph,
    inputs?: WorkflowInputs,
    priority: number = 1,
    useClaudeForAI: boolean = false,
    packageContext?: {
      packageId?: string;
      grantedPermissions?: string[];
      packageTrustLevel?: 'untrusted' | 'trusted' | 'verified' | 'blocked';
    }
  ): string {
    const job: Job = {
      id: uuidv4(),
      flowId,
      flowName,
      graph,
      inputs,
      status: 'pending',
      priority,
      submittedAt: Date.now(),
      logs: [],
      useClaudeForAI,
      packageId: packageContext?.packageId,
      grantedPermissions: packageContext?.grantedPermissions,
      packageTrustLevel: packageContext?.packageTrustLevel,
    };

    if (logger.isDebugEnabled) {
      logger.debug(` submit: jobId=${job.id}, flowName=${flowName}, currentConfig=${JSON.stringify(this.config)}`);
    }

    // Insert into queue based on priority (higher priority first)
    const insertIndex = this.queue.findIndex(j => j.priority < priority);
    if (insertIndex === -1) {
      this.queue.push(job);
    } else {
      this.queue.splice(insertIndex, 0, job);
    }

    this.notifyStateChange();
    this.processQueue();

    return job.id;
  }

  /**
   * Abort a job (pending, running, or queued)
   */
  abort(jobId: string): void {
    // Check if it's an active job
    const activeContext = this.activeJobs.get(jobId);
    if (activeContext) {
      this.abortActiveJob(activeContext);
      return;
    }

    // Check if it's in the queue (pending)
    const queueIndex = this.queue.findIndex(j => j.id === jobId);
    if (queueIndex !== -1) {
      const job = this.queue[queueIndex];
      job.status = 'aborted';
      job.completedAt = Date.now();
      this.queue.splice(queueIndex, 1);
      this.addToHistory(job);
      this.notifyStateChange();
      return;
    }
  }

  /**
   * Abort an active job with fallback force termination
   */
  private abortActiveJob(context: ActiveJobContext): void {
    const { job, abortController } = context;

    // Trigger cooperative abort
    abortController.abort();

    // If job is awaiting AI, reject the pending promise
    if (job.pendingAIRequest) {
      const continueToken = job.pendingAIRequest.continueToken;
      const pending = this.pendingAIResponses.get(continueToken);
      if (pending) {
        pending.reject(new Error('Job aborted'));
        this.pendingAIResponses.delete(continueToken);
      }
      job.pendingAIRequest = undefined;
    }

    // Update status immediately for UI responsiveness
    job.status = 'aborted';
    this.notifyStateChange();

    // Add log entry
    this.addJobLog(job.id, {
      source: 'System',
      message: 'Aborting workflow...',
      type: 'info',
    });

    // Set timeout for force abort warning
    // Since we can't actually force-terminate main thread execution,
    // we'll show a warning if it takes too long
    context.forceAbortTimeoutId = setTimeout(() => {
      if (this.activeJobs.has(job.id)) {
        this.addJobLog(job.id, {
          source: 'System',
          message: 'Abort is taking longer than expected. The workflow may be stuck in a long-running operation.',
          type: 'error',
        });
      }
    }, FORCE_ABORT_TIMEOUT_MS);
  }

  /**
   * Get a job by ID (from active, queue, or history)
   */
  getJob(jobId: string): Job | undefined {
    // Check active
    const active = this.activeJobs.get(jobId);
    if (active) return active.job;

    // Check queue
    const queued = this.queue.find(j => j.id === jobId);
    if (queued) return queued;

    // Check history
    return this.history.find(j => j.id === jobId);
  }

  /**
   * Get all active jobs
   */
  getActiveJobs(): Job[] {
    return Array.from(this.activeJobs.values()).map(ctx => ctx.job);
  }

  /**
   * Get all queued (pending) jobs
   */
  getQueuedJobs(): Job[] {
    return [...this.queue];
  }

  /**
   * Get job history
   */
  getHistory(): Job[] {
    return [...this.history];
  }

  /**
   * Get all jobs (active + queued + history)
   */
  getAllJobs(): Job[] {
    return [
      ...this.getActiveJobs(),
      ...this.getQueuedJobs(),
      ...this.history,
    ];
  }

  /**
   * Get job for a specific flow (if any)
   */
  getJobForFlow(flowId: string): Job | undefined {
    // Check active first
    for (const ctx of this.activeJobs.values()) {
      if (ctx.job.flowId === flowId) return ctx.job;
    }
    // Check queue
    return this.queue.find(j => j.flowId === flowId);
  }

  /**
   * Check if a flow has a running job
   */
  isFlowRunning(flowId: string): boolean {
    for (const ctx of this.activeJobs.values()) {
      if (ctx.job.flowId === flowId && ctx.job.status === 'running') {
        return true;
      }
    }
    return false;
  }

  /**
   * Get the queue position of a pending job (1-indexed)
   * Returns null if job is not in the queue
   */
  getQueuePosition(jobId: string): number | null {
    const index = this.queue.findIndex(j => j.id === jobId);
    return index === -1 ? null : index + 1;
  }

  /**
   * Clear job history
   */
  clearHistory(): void {
    this.history = [];
    this.notifyStateChange();
  }

  // ==========================================
  // Subscription Methods
  // ==========================================

  /**
   * Subscribe to job state changes
   */
  onStateChange(callback: JobStateCallback): () => void {
    this.stateSubscribers.add(callback);
    return () => this.stateSubscribers.delete(callback);
  }

  /**
   * Subscribe to job log entries
   */
  onLog(callback: JobLogCallback): () => void {
    this.logSubscribers.add(callback);
    return () => this.logSubscribers.delete(callback);
  }

  /**
   * Subscribe to node status updates
   */
  onNodeStatus(callback: NodeStatusCallback): () => void {
    this.nodeStatusSubscribers.add(callback);
    return () => this.nodeStatusSubscribers.delete(callback);
  }

  /**
   * Subscribe to streaming token updates
   */
  onStreamToken(callback: StreamTokenCallback): () => void {
    this.tokenSubscribers.add(callback);
    return () => this.tokenSubscribers.delete(callback);
  }

  /**
   * Subscribe to image updates
   */
  onImageUpdate(callback: ImageUpdateCallback): () => void {
    this.imageSubscribers.add(callback);
    return () => this.imageSubscribers.delete(callback);
  }

  /**
   * Subscribe to AI yield notifications (Claude-as-AI pattern)
   * Called when an AI node yields and needs an external response
   */
  onAIYield(callback: AIYieldCallback): () => void {
    this.aiYieldSubscribers.add(callback);
    return () => this.aiYieldSubscribers.delete(callback);
  }

  /**
   * Continue a job that is awaiting an AI response (Claude-as-AI pattern)
   * @param continueToken - The token from the pending AI request
   * @param response - The AI response to use
   * @returns true if the job was continued, false if the token was not found
   */
  continueWithAIResponse(continueToken: string, response: string): boolean {
    const pending = this.pendingAIResponses.get(continueToken);
    if (!pending) {
      logger.warn(`No pending AI request found for token: ${continueToken}`);
      return false;
    }

    // Get the job and update its status back to running
    const job = this.getJob(pending.jobId);
    if (job) {
      job.status = 'running';
      job.pendingAIRequest = undefined;
      this.notifyStateChange();
    }

    // Resolve the promise with the response
    pending.resolve(response);
    this.pendingAIResponses.delete(continueToken);

    logger.debug(`Continued job ${pending.jobId} with AI response (token: ${continueToken})`);
    return true;
  }

  /**
   * Get a pending AI request by job ID
   */
  getPendingAIRequest(jobId: string): PendingAIRequest | undefined {
    const job = this.getJob(jobId);
    return job?.pendingAIRequest;
  }

  /**
   * Internal method for AI module to yield and wait for external response
   * @internal
   */
  _yieldForAI(jobId: string, request: Omit<PendingAIRequest, 'continueToken' | 'createdAt'>): Promise<string> {
    const continueToken = uuidv4();
    const pendingRequest: PendingAIRequest = {
      ...request,
      continueToken,
      createdAt: Date.now(),
    };

    // Get the job and update its status
    const job = this.getJob(jobId);
    if (job) {
      job.status = 'awaiting_ai';
      job.pendingAIRequest = pendingRequest;
      this.notifyStateChange();
    }

    // Create a promise that will be resolved when continueWithAIResponse is called
    return new Promise<string>((resolve, reject) => {
      this.pendingAIResponses.set(continueToken, {
        jobId,
        resolve,
        reject,
      });

      // Notify subscribers about the AI yield
      this.notifyAIYield(jobId, pendingRequest);

      // Add a log entry
      this.addJobLog(jobId, {
        source: 'AI',
        message: `Waiting for external AI response (node: ${request.nodeId})`,
        type: 'info',
      });
    });
  }

  // ==========================================
  // Private Methods
  // ==========================================

  /**
   * Process the queue and start jobs if capacity available
   */
  private processQueue(): void {
    // Parallel jobs are now safe. Both sources of cross-job contamination are gone:
    //   1. Per-job ctx — all bundled module runtimes take ctx per-call via
    //      createMethods(ctx) (see runtime.registerDynamicModule), so their methods +
    //      per-job state are isolated per job (cli/test/module-ctx-isolation.ts).
    //   2. Cross-module lookup — module-internal cross-module calls go through the
    //      per-job ctx.modules / ctx.callModule accessor (dispatches into THIS job's
    //      moduleFunctionHandlers), and the shared globalThis[name] publish has been
    //      removed from registerModuleInternally (cli/test/cross-module-isolation.ts).
    // So capacity follows the configured mode: sequential ⇒ 1, parallel ⇒ maxConcurrency.
    const capacity = this.config.mode === 'sequential' ? 1 : this.config.maxConcurrency;

    logger.debug(` processQueue: mode=${this.config.mode}, capacity=${capacity}, activeJobs=${this.activeJobs.size}, queueLength=${this.queue.length}`);

    while (this.activeJobs.size < capacity && this.queue.length > 0) {
      const job = this.queue.shift();
      if (job) {
        // Synchronously mark job as active to prevent race conditions
        const abortController = new AbortController();
        const context: ActiveJobContext = {
          job,
          runtime: null as unknown as OAIYRuntime, // Will be set in startJob
          abortController,
        };

        // Add to activeJobs BEFORE any async work
        job.status = 'running';
        job.startedAt = Date.now();
        this.activeJobs.set(job.id, context);
        this.notifyStateChange();

        // Start job execution asynchronously. startJob has its own try/finally that always
        // cleans up + calls processQueue, so it shouldn't reject — but as a last-resort backstop
        // against a leaked capacity slot freezing the queue, catch any escaped rejection here and
        // free the slot.
        this.startJob(job, context).catch((err) => {
          logger.error(
            `startJob rejected unexpectedly for ${job.id}: ${err instanceof Error ? err.message : String(err)}`,
          );
          if (this.activeJobs.has(job.id)) {
            this.activeJobs.delete(job.id);
            this.notifyStateChange();
            this.processQueue();
          }
        });
      }
    }
  }

  /**
   * Start executing a job (called after job is already in activeJobs)
   */
  private async startJob(job: Job, context: ActiveJobContext): Promise<void> {
    const { abortController } = context;
    const endTimer = metrics.startTimer('jobExecution');
    metrics.increment('jobsStarted');
    metrics.setGauge('activeJobs', this.activeJobs.size);
    metrics.setGauge('queueDepth', this.queue.length);

    // Create runtime with callbacks
    const runtime = createRuntime(
      // onToken
      (nodeId, token) => {
        this.notifyTokenUpdate(job.id, nodeId, token);
      },
      // onLog
      (entry) => {
        this.addJobLog(job.id, entry);
      },
      // onImage
      (nodeId, imageUrl) => {
        this.notifyImageUpdate(job.id, nodeId, imageUrl);
      },
      // onNodeStatus
      (nodeId, status) => {
        this.notifyNodeStatus(job.id, nodeId, status);
      },
      // abortSignal
      abortController.signal,
      // onDatabase
      this.databaseHandler,
      // onLocalNetworkPermission
      this.networkPermissionHandler
    );

    // NOTE: runtime config + module registration run INSIDE this try. registerDynamicModule
    // invokes module-supplied createMethods()/init(), which can throw for a buggy/untrusted
    // module — without the try the rejection would skip the finally, leaving the job stuck
    // 'running' (a leaked capacity slot that freezes the whole queue). The catch marks it
    // failed and the finally always runs cleanup + processQueue.
    try {
      // Configure runtime
      if (this.projectSettings) {
        runtime.setProjectSettings(this.projectSettings);
      }
      if (this.projectConstants) {
        runtime.setProjectConstants(this.projectConstants, this.secretConstantNames);
      }

      // Swap the untrusted-workflow engine if the host app supplied one. Must
      // happen before any script runs, and it only affects hardened (package)
      // flows — trusted local flows never reach the Worker path at all.
      if (this.untrustedWorkerFactory) {
        runtime.setUntrustedWorkerFactory(this.untrustedWorkerFactory);
        const trusted = this.runTrustedFlowsInWorker;
        runtime.setRunTrustedFlowsInWorker(typeof trusted === 'function' ? trusted() : trusted);
      }

      // Wire the Tauri invoke broker so workflows submitted via the API can
      // reach native commands (browser_v2_*, terminal, database, etc.). Without
      // this every native call fails with "Tauri not detected".
      if (this.tauriInvoke) {
        runtime.setTauriInvoke(this.tauriInvoke);
      }

      // Configure Claude-as-AI mode if enabled
      if (job.useClaudeForAI) {
        runtime.setClaudeAsAI(
          true,
          job.id,
          (request: any) => this._yieldForAI(job.id, request)
        );
      }

      // Set flow context for per-flow database operations + package
      // permission gating. When the job carries a packageId we route the
      // runtime into hardened mode and apply the granted-permission
      // subset; trusted local flows continue to run with full privileges.
      logger.debug(
        ` Setting flow context: flowId=${job.flowId}` +
          (job.packageId ? `, packageId=${job.packageId}` : ''),
      );
      runtime.setFlowContext(job.flowId, job.packageId ?? null);
      if (job.packageId) {
        runtime.setPackagePermissions(job.grantedPermissions ?? []);
      }

      if (this.moduleRegistry) {
        runtime.setModuleRegistry(this.moduleRegistry);

        // Register all module runtimes
        for (const loadedModule of this.moduleRegistry.getAllModules()) {
          if (loadedModule.runtime) {
            await runtime.registerDynamicModule(loadedModule as LoadedModule);
          }
        }
      }

      runtime.setAvailableFlows(this.availableFlows);
      // Set package macros (higher priority than project macros)
      if (this.packageMacros.length > 0) {
        runtime.setPackageMacros(this.packageMacros);
      }

      // Update context with the runtime now that it's configured
      context.runtime = runtime;

      // Prepare inputs - for macro flows, wrap in __macro_inputs__
      let workflowInputs = job.inputs;
      const flow = this.availableFlows.find(f => f.id === job.flowId);
      logger.debug(` Processing job ${job.id}: flowId=${job.flowId}, flow found=${!!flow}, isMacro=${flow?.isMacro}, hasInputs=${!!job.inputs}`);
      if (flow?.isMacro && job.inputs) {
        // Macro inputs need to be wrapped in __macro_inputs__ for macro_input nodes to access them
        // MacroRunnerModal already wraps them, so check if already wrapped to avoid double-wrapping
        const inputsObj = job.inputs as Record<string, unknown>;
        const hasWrapper = '__macro_inputs__' in inputsObj && inputsObj.__macro_inputs__ !== undefined;
        logger.debug(` Checking inputs wrapper: hasWrapper=${hasWrapper}, keys=${Object.keys(inputsObj).join(',')}`);

        if (hasWrapper) {
          // Already wrapped - check for double-wrapping and unwrap if needed
          const innerInputs = inputsObj.__macro_inputs__ as Record<string, unknown>;
          if (innerInputs && '__macro_inputs__' in innerInputs) {
            // Double-wrapped! Unwrap one level
            logger.debug(` Detected double-wrapped inputs, unwrapping one level`);
            workflowInputs = inputsObj.__macro_inputs__ as WorkflowInputs;
          } else {
            // Single wrapped - use as-is
            workflowInputs = job.inputs;
          }
          if (logger.isDebugEnabled) {
            logger.debug(` Inputs already wrapped: ${JSON.stringify(workflowInputs).substring(0, 200)}`);
          }
        } else {
          workflowInputs = { __macro_inputs__: job.inputs } as WorkflowInputs;
          if (logger.isDebugEnabled) {
            logger.debug(` Wrapped inputs in __macro_inputs__: ${JSON.stringify(workflowInputs).substring(0, 200)}`);
          }
        }
      }
      const result = await runtime.runWorkflow(job.graph, this.availableFlows, workflowInputs);

      // Job completed successfully (check for abort that may have happened during execution)
      // Cast to JobStatus to allow TypeScript to understand the runtime status check
      if ((job.status as string) !== 'aborted') {
        job.status = 'completed';
        metrics.increment('jobsCompleted');
        // Convert compiled script result to plain JavaScript value
        job.result = runtime.convertResultToJs(result);

        // Extract node outputs from workflow_context (the result is the workflow_context)
        // This contains all node outputs keyed by node ID
        if (job.result && typeof job.result === 'object') {
          const workflowContext = job.result as Record<string, unknown>;
          // Copy all node outputs (excluding internal keys)
          job.nodeOutputs = {};
          for (const [key, value] of Object.entries(workflowContext)) {
            // Skip internal workflow context keys
            if (!key.startsWith('__')) {
              job.nodeOutputs[key] = value;
            }
          }

          // Extract macro outputs for macro runs (used by MacroRunnerModal)
          if (workflowContext['__macro_outputs__'] && typeof workflowContext['__macro_outputs__'] === 'object') {
            job.results = workflowContext['__macro_outputs__'] as Record<string, unknown>;
          }
        }
      }
    } catch (error) {
      // Check if it was an abort
      const errorStr = error instanceof Error ? error.message : String(error);
      const isAbort =
        (error instanceof Error && error.name === 'AbortError') ||
        errorStr.includes('__ABORT__') ||
        errorStr.includes('aborted');

      if (isAbort) {
        job.status = 'aborted';
        metrics.increment('jobsAborted');
        this.addJobLog(job.id, {
          source: 'System',
          message: 'Workflow stopped by user',
          type: 'info',
        });
      } else {
        job.status = 'failed';
        job.error = errorStr;
        metrics.increment('jobsFailed');
        this.addJobLog(job.id, {
          source: 'System',
          message: `Error: ${errorStr}`,
          type: 'error',
        });
      }
    } finally {
      // Record job execution time
      endTimer();
      // -1 because THIS job is still in activeJobs here; it's removed in the cleanup below.
      metrics.setGauge('activeJobs', this.activeJobs.size - 1);

      // Clear force abort timeout if set
      if (context.forceAbortTimeoutId) {
        clearTimeout(context.forceAbortTimeoutId);
      }

      // Cleanup any orphaned pending AI requests for this job
      if (job.pendingAIRequest) {
        const continueToken = job.pendingAIRequest.continueToken;
        this.pendingAIResponses.delete(continueToken);
        job.pendingAIRequest = undefined;
      }

      // Per-job module teardown: run each loaded module's cleanup (the MODULE_CLEANUP
      // hooks bound to THIS job's state — e.g. core-browser closing its Chromium
      // sessions). Without this the cleanup was dead code and browser sessions leaked
      // across jobs. Best-effort: a cleanup failure must not break job teardown, and
      // each job's runtime owns its own loadedRuntimeModules so this is parallel-safe.
      try {
        await runtime.cleanupDynamicModules();
      } catch (cleanupErr) {
        this.addJobLog(job.id, {
          source: 'System',
          message: `Module cleanup failed: ${cleanupErr instanceof Error ? cleanupErr.message : String(cleanupErr)}`,
          type: 'error',
        });
      }

      // Cleanup
      job.completedAt = Date.now();
      this.activeJobs.delete(job.id);
      this.addToHistory(job);
      this.notifyStateChange();

      // Process next job
      this.processQueue();
    }
  }

  /**
   * Add a log entry to a job
   */
  private addJobLog(jobId: string, entry: Omit<LogEntry, 'id' | 'timestamp'>): void {
    const log: LogEntry = {
      id: uuidv4(),
      timestamp: Date.now(),
      ...entry,
    };

    // Add to job's logs
    const context = this.activeJobs.get(jobId);
    if (context) {
      context.job.logs.push(log);
      // Cap per-job logs so a long/chatty run (browser loops, batch gen) can't grow this
      // array without bound (core memory + feeds the React log view + history snapshots).
      if (context.job.logs.length > MAX_LOG_ENTRIES_PER_JOB) {
        context.job.logs = context.job.logs.slice(-MAX_LOG_ENTRIES_PER_JOB);
      }
    }

    // Notify subscribers
    for (const callback of this.logSubscribers) {
      try {
        callback(jobId, log);
      } catch (e) {
        logger.error(' Log callback error', { error: e });
      }
    }
  }

  /**
   * Add a job to history (with size limit). The retention policy on
   * the active config decides whether we keep the inputs/result/logs,
   * strip them, or skip the entry altogether. Round 5 reviewer #10.
   */
  private addToHistory(job: Job): void {
    const retention = this.config.historyRetention ?? 'full';
    if (retention === 'none') {
      return;
    }
    const stored = retention === 'redacted' ? this.redactJobForHistory(job) : job;
    this.history.unshift(stored);
    if (this.history.length > MAX_JOB_HISTORY_SIZE) {
      this.history.pop();
    }
  }

  /**
   * Strip the high-cardinality, potentially-sensitive fields off a
   * finished job before storing it in history. We keep the job id,
   * flow identity, status, and timing so the history list still shows
   * "what ran when", but we drop inputs, result, graph, and per-log
   * messages. The shape stays the same so subscribers don't need a
   * separate "redacted job" type.
   */
  private redactJobForHistory(job: Job): Job {
    const placeholderLogs = job.logs.length > 0
      ? [
          {
            id: `${job.id}-redacted`,
            timestamp: job.completedAt ?? Date.now(),
            source: 'System',
            message: `<${job.logs.length} log entries omitted (history retention: redacted)>`,
            type: 'info' as const,
          },
        ]
      : [];
    return {
      ...job,
      // Drop the workflow graph entirely — it's reconstructible from
      // the saved flow on disk and is the largest field on the job.
      graph: { nodes: [], edges: [] },
      inputs: undefined,
      result: undefined,
      results: undefined,
      nodeOutputs: undefined,
      currentNodeLabel: undefined,
      error: job.error ? '<error message redacted>' : undefined,
      logs: placeholderLogs,
      // Don't echo a granted-permission set into history — that's a
      // package-trust artefact, not a debug datum.
      grantedPermissions: undefined,
      pendingAIRequest: undefined,
    };
  }

  /**
   * Notify all state subscribers
   */
  private notifyStateChange(): void {
    const allJobs = this.getAllJobs();
    for (const callback of this.stateSubscribers) {
      try {
        callback(allJobs);
      } catch (e) {
        logger.error(' State callback error', { error: e });
      }
    }
  }

  /**
   * Notify node status subscribers
   */
  private notifyNodeStatus(jobId: string, nodeId: string, status: 'running' | 'completed' | 'error'): void {
    for (const callback of this.nodeStatusSubscribers) {
      try {
        callback(jobId, nodeId, status);
      } catch (e) {
        logger.error(' Node status callback error', { error: e });
      }
    }
  }

  /**
   * Notify token subscribers
   */
  private notifyTokenUpdate(jobId: string, nodeId: string, token: string): void {
    for (const callback of this.tokenSubscribers) {
      try {
        callback(jobId, nodeId, token);
      } catch (e) {
        logger.error(' Token callback error', { error: e });
      }
    }
  }

  /**
   * Notify image subscribers
   */
  private notifyImageUpdate(jobId: string, nodeId: string, imageUrl: string): void {
    for (const callback of this.imageSubscribers) {
      try {
        callback(jobId, nodeId, imageUrl);
      } catch (e) {
        logger.error(' Image callback error', { error: e });
      }
    }
  }

  /**
   * Notify AI yield subscribers
   */
  private notifyAIYield(jobId: string, request: PendingAIRequest): void {
    for (const callback of this.aiYieldSubscribers) {
      try {
        callback(jobId, request);
      } catch (e) {
        logger.error(' AI yield callback error', { error: e });
      }
    }
  }

  /**
   * Clear pending timers, abort in-flight jobs, and drop all subscribers.
   * Without this, setTimeout callbacks scheduled by abort() can fire after
   * the Jest environment is torn down and crash with "import after teardown".
   */
  dispose(): void {
    for (const context of this.activeJobs.values()) {
      if (context.forceAbortTimeoutId) {
        clearTimeout(context.forceAbortTimeoutId);
        context.forceAbortTimeoutId = undefined;
      }
      try {
        context.abortController.abort();
      } catch {
        // ignore
      }
    }
    this.activeJobs.clear();
    this.queue = [];
    this.pendingAIResponses.clear();
    this.stateSubscribers.clear();
    this.logSubscribers.clear();
    this.nodeStatusSubscribers.clear();
    this.tokenSubscribers.clear();
    this.imageSubscribers.clear();
    this.aiYieldSubscribers.clear();
  }
}
