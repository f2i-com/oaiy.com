/**
 * Job Queue Context
 *
 * Provides global access to the JobManager for workflow execution queue management.
 * Enables background execution, job tracking, and configurable concurrency.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  type ReactNode,
} from 'react';
import {
  type JobManager,
  type Job,
  type JobConfig,
  type LogEntry,
  type Flow,
  type WorkflowGraph,
  type ProjectSettings,
  type ProjectConstant,
  type LocalNetworkPermissionRequest,
  type LocalNetworkPermissionResponse,
} from 'oaiy-core';
import { createEngine } from '../engine/createEngine';
import { useApiBridge, type ApiBridgeCallbacks } from '../hooks/useApiBridge';
import { createLogger } from '../utils/logger';

const logger = createLogger('JobQueue');

/**
 * Context value type
 */
interface JobQueueContextValue {
  /** The JobManager instance */
  jobManager: JobManager;
  /** All jobs (active + queued + history) */
  jobs: Job[];
  /** Currently active jobs */
  activeJobs: Job[];
  /** Queued (pending) jobs */
  queuedJobs: Job[];
  /** Job history */
  history: Job[];
  /** Queue configuration */
  config: JobConfig;
  /** Update queue configuration */
  setConfig: (config: Partial<JobConfig>) => void;
  /** Check if a specific flow has a running job */
  isFlowRunning: (flowId: string) => boolean;
  /** Get the job for a specific flow (if any) */
  getJobForFlow: (flowId: string) => Job | undefined;
  /** Clear job history */
  clearHistory: () => void;
  /** Submit a job to the queue */
  submitJob: (
    flowId: string,
    graph: WorkflowGraph,
    inputs?: Record<string, unknown>,
    flowName?: string,
    packageContext?: {
      packageId?: string;
      grantedPermissions?: string[];
      packageTrustLevel?: 'untrusted' | 'trusted' | 'verified' | 'blocked';
    },
  ) => string;
  /** Subscribe to updates for a specific job */
  subscribeToJob: (jobId: string, callback: (job: Job) => void) => () => void;
}

const JobQueueContext = createContext<JobQueueContextValue | null>(null);

/**
 * Props for the JobQueueProvider
 */
interface JobQueueProviderProps {
  children: ReactNode;
  /** Available flows for subflow execution */
  availableFlows?: Flow[];
  /** Package macros (higher priority than project macros) */
  packageMacros?: Flow[];
  /** Project settings */
  projectSettings?: ProjectSettings;
  /**
   * Project constants. API-key constants (e.g. OPENAI_API_KEY) are resolved
   * from these at run time via `ctx.getConstant`, so cloud LLM / service nodes
   * can authenticate with the user's stored keys.
   */
  constants?: ProjectConstant[];
  /** Handler for local network permission requests */
  onLocalNetworkPermission?: (
    request: LocalNetworkPermissionRequest
  ) => Promise<LocalNetworkPermissionResponse>;
  /** Callback when project settings should be updated */
  onUpdateSettings?: (updates: Partial<ProjectSettings>) => void;
  /** Callback to create a new flow via API */
  onCreateFlow?: (name: string) => Flow;
  /** Callback to delete a flow via API */
  onDeleteFlow?: (flowId: string) => void;
  /** Callback to update a flow via API */
  onUpdateFlow?: (flowId: string, updates: Partial<Omit<Flow, 'id' | 'createdAt'>>) => void;
  /** Callback to update a flow's graph via API */
  onUpdateFlowGraph?: (flowId: string, graph: WorkflowGraph) => void;
  /** Callback to clear application cache via API */
  onClearCache?: () => Promise<void>;
  /** Callback to reload macros via API */
  onReloadMacros?: () => Promise<void>;
  /** Callback to recompile packages via API */
  onRecompilePackages?: () => Promise<{ success: boolean; output?: string; error?: string }>;
}

// The database handler and the JobManager assembly now live in the shared,
// host-agnostic `../engine/createEngine` (reused by the headless CLI). This
// provider just supplies the React-specific policy/context bits.

/**
 * Job Queue Provider component
 */
export function JobQueueProvider({
  children,
  availableFlows = [],
  packageMacros = [],
  projectSettings,
  constants = [],
  onLocalNetworkPermission,
  onUpdateSettings,
  onCreateFlow,
  onDeleteFlow,
  onUpdateFlow,
  onUpdateFlowGraph,
  onClearCache,
  onReloadMacros,
  onRecompilePackages,
}: JobQueueProviderProps) {
  // Use refs to always access current values in the network permission handler
  // This prevents stale closures from capturing initial prop values
  const projectSettingsRef = useRef(projectSettings);
  const onUpdateSettingsRef = useRef(onUpdateSettings);
  const onLocalNetworkPermissionRef = useRef(onLocalNetworkPermission);

  // Keep refs in sync with props
  useEffect(() => {
    projectSettingsRef.current = projectSettings;
  }, [projectSettings]);

  useEffect(() => {
    onUpdateSettingsRef.current = onUpdateSettings;
  }, [onUpdateSettings]);

  useEffect(() => {
    onLocalNetworkPermissionRef.current = onLocalNetworkPermission;
  }, [onLocalNetworkPermission]);

  // Project constants flattened to key→value (including rehydrated secret
  // values) for the runtime's getConstant. Only non-empty values are included
  // so a secret that hasn't loaded from the keyring yet can't shadow a real
  // one with an empty string.
  const constantsMap = useMemo(
    () =>
      Object.fromEntries(
        (constants || [])
          .filter((c) => c.key && c.value)
          .map((c) => [c.key, c.value]),
      ) as Record<string, string>,
    [constants],
  );

  // Names of the constants flagged secret, threaded alongside the value map so the runtime can
  // require the (high-risk) secrets:read permission — vs constants:read — before resolving one
  // for an UNTRUSTED package's getConstant. Without this the isSecret flag is dropped in the
  // flatten above and a package granted only `network` could read every stored API key.
  const secretConstantNames = useMemo(
    () =>
      new Set(
        (constants || []).filter((c) => c.key && c.value && c.isSecret).map((c) => c.key),
      ),
    [constants],
  );

  // Create JobManager instance using useState with lazy initializer
  // The network permission handler captures refs, which is intentional for accessing
  // current prop values without recreating the JobManager on every prop change.
  // eslint-disable-next-line react-hooks/refs
  const [jobManager] = useState(() => {
    // Create handler that uses refs to access current values
    const networkPermissionHandler = async (request: LocalNetworkPermissionRequest): Promise<LocalNetworkPermissionResponse> => {
      const permissionCallback = onLocalNetworkPermissionRef.current;
      if (!permissionCallback) {
        // No permission handler - deny by default
        return { allowed: false, remember: false };
      }

      const response = await permissionCallback(request);

      // If user allowed and wants to remember, update the whitelist
      if (response.allowed && response.remember) {
        const updateSettings = onUpdateSettingsRef.current;
        const currentSettings = projectSettingsRef.current;
        if (updateSettings) {
          const currentWhitelist = currentSettings?.localNetworkWhitelist || [];
          if (!currentWhitelist.includes(request.hostPort)) {
            updateSettings({
              localNetworkWhitelist: [...currentWhitelist, request.hostPort],
            });
          }
        }
      }
      return response;
    };

    return createEngine({
      networkPermissionHandler,
      availableFlows,
      packageMacros,
      projectSettings,
      projectConstants: constantsMap,
      secretConstantNames,
    });
  });

  // State for reactive updates
  const [jobs, setJobs] = useState<Job[]>([]);
  const [config, setConfigState] = useState<JobConfig>(() => jobManager.getConfig());

  // Update JobManager when props change
  useEffect(() => {
    jobManager.setAvailableFlows(availableFlows);
  }, [availableFlows, jobManager]);

  // Update package macros when they change
  useEffect(() => {
    jobManager.setPackageMacros(packageMacros);
  }, [packageMacros, jobManager]);

  useEffect(() => {
    if (projectSettings) {
      jobManager.setProjectSettings(projectSettings);
    }
  }, [projectSettings, jobManager]);

  // Keep the runtime's constant table current (secret values rehydrate from
  // the keyring asynchronously after load, so this fires again once they land).
  useEffect(() => {
    jobManager.setProjectConstants(constantsMap, secretConstantNames);
  }, [constantsMap, secretConstantNames, jobManager]);

  // Subscribe to job state changes
  useEffect(() => {
    const unsubscribe = jobManager.onStateChange((allJobs) => {
      setJobs([...allJobs]);
    });
    return unsubscribe;
  }, [jobManager]);

  // Config update handler
  const setConfig = useCallback(
    (newConfig: Partial<JobConfig>) => {
      jobManager.setConfig(newConfig);
      setConfigState(jobManager.getConfig());
    },
    [jobManager]
  );

  // Flow running check
  const isFlowRunning = useCallback(
    (flowId: string) => jobManager.isFlowRunning(flowId),
    [jobManager]
  );

  // Get job for flow
  const getJobForFlow = useCallback(
    (flowId: string) => jobManager.getJobForFlow(flowId),
    [jobManager]
  );

  // Clear history
  const clearHistory = useCallback(() => {
    jobManager.clearHistory();
  }, [jobManager]);

  // Submit a job
  //
  // Callers that submit a flow loaded from a package MUST pass the
  // packageContext so the runtime applies the fail-closed permission
  // gate to the job. The HTTP / API-bridge submit path does this
  // explicitly (see useApiBridge); the in-app run-button path goes
  // through useWorkflowExecution which threads the active package's
  // granted set through the same channel.
  const submitJob = useCallback(
    (
      flowId: string,
      graph: WorkflowGraph,
      inputs?: Record<string, unknown>,
      flowName?: string,
      packageContext?: {
        packageId?: string;
        grantedPermissions?: string[];
        packageTrustLevel?: 'untrusted' | 'trusted' | 'verified' | 'blocked';
      },
    ): string => {
      // Cast inputs to WorkflowInputs type (the runtime will handle type checking)
      return jobManager.submit(
        flowId,
        flowName || 'Unnamed',
        graph,
        inputs as Record<string, string | number | boolean | object> | undefined,
        1,
        false,
        packageContext,
      );
    },
    [jobManager]
  );

  // Subscribe to a specific job's updates
  const subscribeToJob = useCallback(
    (jobId: string, callback: (job: Job) => void): (() => void) => {
      // Subscribe to FUTURE state changes for this job.
      const unsubscribe = jobManager.onStateChange((allJobs) => {
        const job = allJobs.find(j => j.id === jobId);
        if (job) {
          callback(job);
        }
      });
      // A job can reach a terminal state BEFORE this subscriber attaches (a trivial
      // run completes within the microtask queue while a consumer subscribes inside
      // a passive effect — a macrotask). onStateChange only fires on FUTURE changes,
      // so the completion would otherwise never be observed. Replay the already-
      // terminal snapshot — via queueMicrotask so the caller's
      // `const unsub = subscribeToJob(...)` is assigned before the callback (which
      // calls unsub() on terminal status) runs. Only terminal states are replayed,
      // so running/pending aren't spuriously re-delivered; consumers are idempotent.
      const existing = jobManager.getJob(jobId);
      if (
        existing &&
        (existing.status === 'completed' ||
          existing.status === 'failed' ||
          existing.status === 'aborted')
      ) {
        queueMicrotask(() => callback(existing));
      }
      return unsubscribe;
    },
    [jobManager]
  );

  // Derived state — memoized so the context value below stays referentially
  // stable when `jobs` is unchanged.
  const activeJobs = useMemo(() => jobs.filter(j => j.status === 'running'), [jobs]);
  const queuedJobs = useMemo(() => jobs.filter(j => j.status === 'pending'), [jobs]);
  const history = useMemo(
    () => jobs.filter(j => ['completed', 'failed', 'aborted'].includes(j.status)),
    [jobs],
  );

  // Initialize API bridge for external HTTP API access
  const apiBridgeCallbacks: ApiBridgeCallbacks = {
    onCreateFlow,
    onDeleteFlow,
    onUpdateFlow,
    onUpdateFlowGraph,
    onClearCache,
    onReloadMacros,
    onRecompilePackages,
  };
  useApiBridge(jobManager, availableFlows, onCreateFlow, onDeleteFlow, apiBridgeCallbacks);

  const value: JobQueueContextValue = useMemo(
    () => ({
      jobManager,
      jobs,
      activeJobs,
      queuedJobs,
      history,
      config,
      setConfig,
      isFlowRunning,
      getJobForFlow,
      clearHistory,
      submitJob,
      subscribeToJob,
    }),
    [
      jobManager,
      jobs,
      activeJobs,
      queuedJobs,
      history,
      config,
      setConfig,
      isFlowRunning,
      getJobForFlow,
      clearHistory,
      submitJob,
      subscribeToJob,
    ],
  );

  return (
    <JobQueueContext.Provider value={value}>
      {children}
    </JobQueueContext.Provider>
  );
}

/**
 * Hook to access the job queue context
 */
export function useJobQueue(): JobQueueContextValue {
  const context = useContext(JobQueueContext);
  if (!context) {
    throw new Error('useJobQueue must be used within a JobQueueProvider');
  }
  return context;
}

/**
 * Hook to subscribe to job logs for a specific job
 */
export function useJobLogs(jobId: string | null): LogEntry[] {
  const { jobManager } = useJobQueue();
  const [logs, setLogs] = useState<LogEntry[]>([]);

  useEffect(() => {
    if (!jobId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: reset logs when job changes
      setLogs([]);
      return;
    }

    // Get existing logs
    const job = jobManager.getJob(jobId);
    if (job) {
      setLogs([...job.logs]);
    }

    // Subscribe to new logs
    const unsubscribe = jobManager.onLog((logJobId, log) => {
      if (logJobId === jobId) {
        // Bound the rendered log buffer (matches the engine's MAX_LOG_ENTRIES_PER_JOB): a
        // long/chatty run would otherwise re-allocate + re-render the whole array per line,
        // climbing memory until the tab stutters/freezes.
        setLogs(prev => (prev.length >= 1000 ? [...prev.slice(-999), log] : [...prev, log]));
      }
    });

    return unsubscribe;
  }, [jobId, jobManager]);

  return logs;
}

/**
 * Hook to subscribe to node status updates for a specific job
 * Uses a ref to avoid re-subscriptions when callback changes
 */
export function useJobNodeStatus(
  jobId: string | null,
  onStatus: (nodeId: string, status: 'running' | 'completed' | 'error') => void
) {
  const { jobManager } = useJobQueue();
  const onStatusRef = useRef(onStatus);

  // Keep ref in sync
  useEffect(() => {
    onStatusRef.current = onStatus;
  }, [onStatus]);

  useEffect(() => {
    if (!jobId) return;

    const unsubscribe = jobManager.onNodeStatus((statusJobId, nodeId, status) => {
      if (statusJobId === jobId) {
        onStatusRef.current(nodeId, status);
      }
    });

    return unsubscribe;
  }, [jobId, jobManager]);
}

/**
 * Hook to subscribe to streaming tokens for a specific job
 * Uses a ref to avoid re-subscriptions when callback changes
 */
export function useJobStreamTokens(
  jobId: string | null,
  onToken: (nodeId: string, token: string) => void
) {
  const { jobManager } = useJobQueue();
  const onTokenRef = useRef(onToken);

  // Keep ref in sync
  useEffect(() => {
    onTokenRef.current = onToken;
  }, [onToken]);

  useEffect(() => {
    if (!jobId) return;

    const unsubscribe = jobManager.onStreamToken((tokenJobId, nodeId, token) => {
      if (tokenJobId === jobId) {
        onTokenRef.current(nodeId, token);
      }
    });

    return unsubscribe;
  }, [jobId, jobManager]);
}

/**
 * Hook to subscribe to image updates for a specific job
 * Uses a ref to avoid re-subscriptions when callback changes
 */
export function useJobImageUpdates(
  jobId: string | null,
  onImage: (nodeId: string, imageUrl: string) => void
) {
  const { jobManager } = useJobQueue();
  const onImageRef = useRef(onImage);

  // Keep ref in sync
  useEffect(() => {
    onImageRef.current = onImage;
  }, [onImage]);

  useEffect(() => {
    if (!jobId) return;

    const unsubscribe = jobManager.onImageUpdate((imageJobId, nodeId, imageUrl) => {
      if (imageJobId === jobId) {
        onImageRef.current(nodeId, imageUrl);
      }
    });

    return unsubscribe;
  }, [jobId, jobManager]);
}
