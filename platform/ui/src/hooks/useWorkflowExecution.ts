/**
 * Workflow Execution Hook
 *
 * Handles workflow execution lifecycle including job submission, running, stopping,
 * and completion handling. Extracts execution logic from OAIYBuilder.tsx for better
 * separation of concerns and testability.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { invoke } from '@tauri-apps/api/core';
import {
  useJobQueue,
  useJobLogs,
  useJobNodeStatus,
  useJobStreamTokens,
  useJobImageUpdates,
} from '../contexts/JobQueueContext';
import type { WorkflowGraph, LogEntry, OAIYPackageManifest } from 'oaiy-core';
import { getModuleLoader } from 'oaiy-core';
import { createLogger } from '../utils/logger';
import { afterUnfinishedRun, beforeRun, type NodeRunStatus } from '../lib/runResults';
import { getNodeNotice } from 'oaiy-ui-components';

const logger = createLogger('WorkflowExecution');

export interface UseWorkflowExecutionOptions {
  flowId: string;
  flowName: string;
  isMacro: boolean;
  nodes: Node[];
  edges: Edge[];
  getWorkflowGraph: () => WorkflowGraph | null;
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  updateNodeData: (nodeId: string, data: Record<string, unknown>) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning', duration?: number, action?: { label: string; onClick: () => void }, key?: string) => void;
  /** Open the log (a failed run's toast offers it). */
  onShowLog?: () => void;
  /** Check if workflow has input nodes that need configuration */
  hasInputNodes: (nodes: Node[]) => boolean;
  /** Package mode for service checking */
  packageMode?: {
    manifest: OAIYPackageManifest;
    flow: unknown;
    sourcePath: string;
  } | null;
  /**
   * Strict subset of permissions the user granted at trust-confirm
   * time, plus the resulting trust level. Used to put the runtime in
   * hardened-mode for package flows so the fail-closed permission gate
   * applies. Trusted local flows (no `packageMode`) can omit this.
   */
  packagePermissionContext?: {
    packageId: string;
    grantedPermissions: string[];
    packageTrustLevel: 'untrusted' | 'trusted' | 'verified' | 'blocked';
  } | null;
}

export interface WorkflowExecutionState {
  /** Whether the workflow is currently running */
  isRunning: boolean;
  /** Queued behind another run (one runs at a time): not started yet, and can be cancelled */
  isWaiting: boolean;
  /** Current job ID if one is active */
  currentJobId: string | null;
  /** Combined logs from job queue or local logs */
  logs: LogEntry[];
  /** Whether the run workflow modal should be shown */
  showRunModal: boolean;
  /** Whether the service startup dialog should be shown */
  showServiceDialog: boolean;
  /** Whether we're transitioning between flows (for loading state) */
  flowTransitioning: boolean;
}

export interface WorkflowExecutionActions {
  /** Run the workflow (may show modal if inputs needed); `true`: even where a step's model is not here */
  runWorkflow: (anyway?: unknown) => Promise<void>;
  /** Stop the running workflow */
  stopWorkflow: () => void;
  /** Clear all logs */
  clearLogs: () => void;
  /** Close the run workflow modal */
  closeRunModal: () => void;
  /** Confirm run modal and start execution */
  confirmRunModal: (updatedInputs: Map<string, Record<string, unknown>>) => void;
  /** Close the service dialog */
  closeServiceDialog: () => void;
  /** Proceed after service dialog (services started) */
  proceedAfterServiceDialog: () => void;
  /** Mark flow transition as complete */
  finishFlowTransition: () => void;
}

export function useWorkflowExecution({
  flowId,
  flowName,
  isMacro,
  nodes,
  edges,
  getWorkflowGraph,
  setNodes,
  updateNodeData,
  onShowToast,
  hasInputNodes,
  packageMode,
  packagePermissionContext,
  onShowLog,
}: UseWorkflowExecutionOptions): WorkflowExecutionState & WorkflowExecutionActions {
  const showLogRef = useRef(onShowLog);
  showLogRef.current = onShowLog;
  const toastRef = useRef(onShowToast);
  toastRef.current = onShowToast;
  // Job queue integration
  const { jobManager, isFlowRunning, getJobForFlow, jobs, getLastResult } = useJobQueue();

  // Track manually started jobs (set when user clicks run button); a flow opened again starts on its last good run,
  // so its outputs are shown again rather than blank
  const [manualJobId, setManualJobId] = useState<string | null>(() => getLastResult(flowId)?.id ?? null);

  // Local logs state for when there's no active job
  const [localLogs, setLocalLogs] = useState<LogEntry[]>([]);

  // Modal states
  const [showRunModal, setShowRunModal] = useState(false);
  const [showServiceDialog, setShowServiceDialog] = useState(false);

  // Flow transition state
  const [flowTransitioning, setFlowTransitioning] = useState(false);
  const prevFlowIdRef = useRef(flowId);

  // The flow's live job in the queue: running, or waiting its turn behind another flow's
  const liveJob = useMemo(() => {
    const existingJob = getJobForFlow(flowId);
    return existingJob && (existingJob.status === 'running' || existingJob.status === 'pending') ? existingJob : null;
  }, [flowId, getJobForFlow, jobs]);
  const activeJobFromQueue = liveJob?.id ?? null;

  // The current job: the live one (a run started elsewhere, by the agent or an app, included), else the one started
  // here last, whose log and result stay shown after it ends
  const currentJobId = activeJobFromQueue || manualJobId;

  // A live job started elsewhere (by the agent, an app) is followed as one started here: its log and result stay
  // shown after it ends, not lost the moment it stops being live.
  useEffect(() => {
    if (activeJobFromQueue && activeJobFromQueue !== manualJobId) setManualJobId(activeJobFromQueue);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobFromQueue]);

  // Track if this flow is running, or waiting its turn
  const isRunning = isFlowRunning(flowId) || liveJob?.status === 'running';
  const isWaiting = !isRunning && liveJob?.status === 'pending';

  // Get logs for the current job
  const jobLogs = useJobLogs(currentJobId);
  const logs = currentJobId ? jobLogs : localLogs;

  // Auto-reconnect to running job when navigating back to a flow
  useEffect(() => {
    if (prevFlowIdRef.current !== flowId) {
      const oldFlowId = prevFlowIdRef.current;
      prevFlowIdRef.current = flowId;

      // Check if there's a running job for the NEW flow we're navigating to
      const runningJobForNewFlow = getJobForFlow(flowId);
      if (runningJobForNewFlow?.status === 'running') {
        // Auto-reconnect to the running job
        setManualJobId(runningJobForNewFlow.id);
        onShowToast?.('Reconnected to running job', 'info');
      } else {
        // Keep manualJobId only if it actually belongs to the flow we're navigating
        // to — inspect THAT job (jobManager.getJob searches active/queue/history), not
        // getJobForFlow(flowId) which is always for the NEW flow, making the old guard
        // tautological: a queued job from another flow could survive and the UI would
        // then render the wrong flow's logs (and hide the new flow's pending job).
        const manualJob = manualJobId ? jobManager.getJob(manualJobId) : null;
        if (!manualJob || manualJob.flowId !== flowId) {
          setManualJobId(null);
        }
      }

      // Show loading state during flow transition
      setFlowTransitioning(true);

      logger.debug(`Flow changed from ${oldFlowId} to ${flowId}`);
    }
  }, [flowId, getJobForFlow, jobManager, manualJobId, onShowToast]);

  // Subscribe to node status updates for this job
  useJobNodeStatus(
    currentJobId,
    useCallback(
      (nodeId: string, status: NodeRunStatus) => {
        // A node's "running" arriving after its run was stopped or ended (a step still finishing) would stay on it,
        // spinning, with nothing running: not shown. Its error is, and where it stopped.
        const job = currentJobId ? jobManager.getJob(currentJobId) : undefined;
        if (status !== 'error' && status !== 'stopped' && job && job.status !== 'running') return;
        updateNodeData(nodeId, { _status: status });
      },
      [updateNodeData, currentJobId, jobManager]
    )
  );

  // Subscribe to streaming tokens for this job
  useJobStreamTokens(
    currentJobId,
    useCallback((_nodeId: string, _token: string) => {
      // Could be used for live streaming display in the future
    }, [])
  );

  // Subscribe to image updates for this job
  useJobImageUpdates(
    currentJobId,
    useCallback(
      (nodeId: string, imageUrl: string) => {
        updateNodeData(nodeId, { imageUrl });
      },
      [updateNodeData]
    )
  );

  // Track which job we've already processed for completion
  const processedJobRef = useRef<string | null>(null);

  // Watch for job completion to update Output nodes with results
  useEffect(() => {
    if (!currentJobId) {
      processedJobRef.current = null;
      return;
    }

    // Find the current job by ID (whole as kept for its flow, where the queue's history has slimmed it)
    const kept = getLastResult(flowId);
    const job = kept?.id === currentJobId ? kept : jobs.find((j) => j.id === currentJobId);
    if (!job) return;

    // Only process completion once per job
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'aborted') {
      if (processedJobRef.current === job.id) return; // Already processed
      processedJobRef.current = job.id;

      // Job completed, update nodes with final results
      if (job.status === 'completed' && job.result !== undefined) {
        const result = job.result;
        let outputValue: string | string[] = '';

        if (result && typeof result === 'object') {
          const resultObj = result as Record<string, unknown>;
          if (resultObj.__output__ !== undefined) {
            outputValue = resultObj.__output__ as string | string[];
          } else {
            outputValue = JSON.stringify(result);
          }
        } else if (result !== null && result !== undefined) {
          outputValue = String(result);
        }

        const resultObj = result as Record<string, unknown>;

        setNodes((nds) => {
          const currentEdges = edges;

          return nds.map((node) => {
            const updates: Record<string, unknown> = {};

            // Update output nodes with their own value: what the run gave the node (its source's handle, or the
            // value it declares), as the CLI and the desktop report it. The source node's value was shown before,
            // which is not the output's when the edge leaves another handle (an image node's info, a condition's
            // false) or the node declares its own value: the editor showed one answer, a run elsewhere another.
            if (node.type === 'output') {
              const incomingEdge = currentEdges.find((e) => e.target === node.id);
              if (resultObj[node.id] !== undefined) {
                updates.outputValue = resultObj[node.id];
              } else if (incomingEdge && resultObj[incomingEdge.source] !== undefined) {
                // Keep arrays intact — OutputNode renders them as a gallery (multi-image /
                // multi-result, with an index selector + multi-download). Collapsing to
                // [0] here silently dropped all but the first item. Only image_save/
                // video_save legitimately take [0] (single file) — those branches stay.
                updates.outputValue = resultObj[incomingEdge.source];
              } else if (outputValue) {
                updates.outputValue = outputValue;
              }
            }

            // Update video_save nodes with result
            if (node.type === 'video_save') {
              if (resultObj[node.id]) {
                const nodeValue = resultObj[node.id];
                const videoPath = Array.isArray(nodeValue) ? nodeValue[0] : nodeValue;
                if (typeof videoPath === 'string' && videoPath) {
                  updates.outputValue = videoPath;
                }
              } else {
                const incomingEdge = currentEdges.find((e) => e.target === node.id);
                if (incomingEdge && resultObj[incomingEdge.source]) {
                  const sourceValue = resultObj[incomingEdge.source];
                  const videoPath = Array.isArray(sourceValue) ? sourceValue[0] : sourceValue;
                  if (typeof videoPath === 'string' && videoPath) {
                    updates.outputValue = videoPath;
                  }
                }
              }
            }

            // Image View shows the picture it was given (it passes it on, and reported it nowhere: its card stayed
            // empty after a run)
            if (node.type === 'image_view') {
              const raw = resultObj[node.id];
              const picture = Array.isArray(raw) ? raw[0] : raw;
              const url = typeof picture === 'string'
                ? picture
                : picture && typeof picture === 'object'
                  ? ['dataUrl', 'url', 'image', 'file'].map((k) => (picture as Record<string, unknown>)[k]).find((v): v is string => typeof v === 'string' && v !== '')
                  : undefined;
              if (url) updates.imageUrl = url;
            }

            // Update image_save nodes with result
            if (node.type === 'image_save') {
              const existingUrl = node.data.imageUrl as string | undefined;
              const hasDataUrl = existingUrl?.startsWith('data:');

              if (!hasDataUrl) {
                if (resultObj[node.id]) {
                  const nodeValue = resultObj[node.id];
                  const imagePath = Array.isArray(nodeValue) ? nodeValue[0] : nodeValue;
                  if (typeof imagePath === 'string' && imagePath) {
                    updates.imageUrl = imagePath;
                  }
                } else {
                  const incomingEdge = currentEdges.find((e) => e.target === node.id);
                  if (incomingEdge && resultObj[incomingEdge.source]) {
                    const sourceValue = resultObj[incomingEdge.source];
                    const imagePath = Array.isArray(sourceValue) ? sourceValue[0] : sourceValue;
                    if (typeof imagePath === 'string' && imagePath) {
                      updates.imageUrl = imagePath;
                    }
                  }
                }
              }
            }

            // Clear status
            if (node.data._status) {
              updates._status = undefined;
            }

            if (Object.keys(updates).length > 0) {
              return { ...node, data: { ...node.data, ...updates, _updateKey: Date.now() } };
            }
            return node;
          });
        });
      } else {
        // A failed or stopped run: what was still running is not any more (stopped where Stop found it), the node
        // that failed keeps its Error mark (it shows where the run went wrong), and the results show there is none
        // from this run (`_stale`), rather than nothing changing on the canvas.
        const stopped = job.status === 'aborted';
        const failedNodeId = (job as { failedNodeId?: string }).failedNodeId;
        setNodes((nds) => nds.map((node) => afterUnfinishedRun(node, stopped, failedNodeId)));
        if (job.status === 'failed') {
          const full = String((job as { error?: unknown }).error ?? '').replace(/^(Error:\s*)+/, '').trim();
          // Its first sentence (the rest, advice that can run to a paragraph, is in the log the toast opens); one
          // run's failure in place of the last one's, not one toast more for each run.
          const why = full.match(/^.+?[.!?](?=\s|$)/s)?.[0] ?? full;
          toastRef.current?.(
            why ? `The run failed: ${why.length > 220 ? `${why.slice(0, 220)}…` : why}` : 'The run failed.',
            'error',
            9000,
            showLogRef.current ? { label: 'Show log', onClick: () => showLogRef.current?.() } : undefined,
            `run-failed:${flowId}`,
          );
        }
      }
    }
  }, [currentJobId, jobs, setNodes, edges, getLastResult, flowId]);

  // Actually submit the workflow to the job queue
  const submitWorkflow = useCallback(() => {
    // Guard against double-submit: no second job while this flow has one, running, waiting its turn, or still
    // stopping (a stopped job keeps the run slot until it has ended).
    const existing = getJobForFlow(flowId);
    if (existing) {
      onShowToast?.(
        existing.status === 'pending' ? 'This flow is already waiting to run.' : existing.status === 'aborted' ? 'The last run is still stopping: run it again in a moment.' : 'This flow is already running.',
        'info',
      );
      return;
    }

    const graph = getWorkflowGraph();
    if (!graph || graph.nodes.length === 0) {
      onShowToast?.('There is nothing to run: add a node first.', 'warning');
      return;
    }

    // Apply any pending input updates that might not have been reflected in React state yet.
    // This handles the race condition where confirmRunModal updates React state but
    // getWorkflowGraph reads from a stale closure before React re-renders.
    const pendingUpdates = pendingInputUpdatesRef.current;
    if (pendingUpdates && pendingUpdates.size > 0) {
      graph.nodes = graph.nodes.map(n => {
        const update = pendingUpdates.get(n.id);
        return update ? { ...n, data: { ...n.data, ...update } } : n;
      });
      pendingInputUpdatesRef.current = null;
    }

    // Submit to job queue. For package flows we also forward the
    // package id + granted permissions so JobManager puts the runtime
    // into hardened mode and applies the fail-closed permission gate.
    const jobId = jobManager.submit(
      flowId,
      flowName,
      graph,
      undefined,
      1,
      false,
      packagePermissionContext ?? undefined,
    );
    setManualJobId(jobId);
    setLocalLogs([]); // Clear local logs when starting new job
    // The last run's results and marks go as this one starts: a run that fails showed the one before's result, as
    // though it were this one's.
    setNodes((nds) => nds.map(beforeRun));
  }, [getWorkflowGraph, jobManager, flowId, flowName, onShowToast, packagePermissionContext, isFlowRunning, getJobForFlow, setNodes]);

  // Proceed with run after inputs are confirmed (handles service check)
  const proceedWithRun = useCallback(async () => {
    // If in package mode with services, check if services need to be started
    if (packageMode?.manifest.services && packageMode.manifest.services.length > 0) {
      try {
        // Check current service status
        const statuses = await invoke<Array<{ id: string; running: boolean; port?: number }>>('get_package_services', {
          packageId: packageMode.manifest.id,
        });

        // Check if all services are running
        const allRunning = packageMode.manifest.services.every((svc) => {
          const status = statuses.find((st) => st.id.endsWith(`::${svc.id}`));
          return status?.running === true;
        });

        if (!allRunning) {
          // Pre-flight: the Rust backend will reject `start_package_service`
          // unless the package was granted `service:start`. Surface that
          // here as a friendly toast instead of letting the user click
          // "Start All" and see an opaque backend error. The user can
          // still skip the dialog and run the flow without services if
          // the flow tolerates it.
          const hasServiceStart =
            packagePermissionContext?.grantedPermissions?.includes('service:start') ?? false;
          if (!hasServiceStart) {
            onShowToast?.(
              `Package '${packageMode.manifest.name}' was not granted the 'service:start' permission, so its services can't be launched. Reload the package and grant service-start access, or run without services.`,
              'error',
            );
            return;
          }
          // Show service startup dialog
          setShowServiceDialog(true);
          return;
        }
      } catch (err) {
        logger.error('Failed to check service status', { error: err });
        // Continue anyway - maybe services aren't needed or user will see errors in the logs
      }
    }

    // All services running or no services needed - submit workflow
    submitWorkflow();
  }, [packageMode, submitWorkflow, packagePermissionContext, onShowToast]);

  // Ref to hold latest proceedWithRun to avoid stale closure
  const proceedWithRunRef = useRef(proceedWithRun);
  useEffect(() => {
    proceedWithRunRef.current = proceedWithRun;
  }, [proceedWithRun]);

  // Ref to track confirm timeout for cleanup
  const confirmTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Ref to store pending input updates from RunWorkflowModal
  // These are applied to the graph before submission to avoid React state timing issues
  const pendingInputUpdatesRef = useRef<Map<string, Record<string, unknown>> | null>(null);

  // Cleanup confirm timeout on unmount
  useEffect(() => {
    return () => {
      if (confirmTimeoutRef.current) {
        clearTimeout(confirmTimeoutRef.current);
      }
    };
  }, []);

  // Handle run modal confirmation
  const confirmRunModal = useCallback(
    (updatedInputs: Map<string, Record<string, unknown>>) => {
      setShowRunModal(false);

      // Update node data with new input values
      updatedInputs.forEach((data, nodeId) => {
        updateNodeData(nodeId, data);
      });

      // Store pending updates so submitWorkflow can apply them to the graph
      // even if React hasn't re-rendered yet (avoids stale closure race condition)
      pendingInputUpdatesRef.current = updatedInputs;

      // Clear any previous timeout
      if (confirmTimeoutRef.current) {
        clearTimeout(confirmTimeoutRef.current);
      }

      // Small delay to ensure node updates are processed
      confirmTimeoutRef.current = setTimeout(() => {
        proceedWithRunRef.current();
        confirmTimeoutRef.current = null;
      }, 50);
    },
    [updateNodeData]
  );

  // Run workflow via job queue
  const runWorkflow = useCallback(async (anyway?: unknown) => {
    // Macros cannot be run directly
    if (isMacro) {
      onShowToast?.('A macro does not run by itself: use it as a node in another flow.', 'warning');
      return;
    }

    const graph = getWorkflowGraph();
    if (!graph || graph.nodes.length === 0) {
      onShowToast?.('There is nothing to run: add a node first.', 'warning');
      return;
    }

    // A step whose model or service is not here (not installed, its computer offline: what its card says) is said
    // before the run starts, not when the run gets to it, after the steps before it have run (an example's picture
    // step failed after its language model had written the prompt). (`anyway`: "Run anyway" from that toast; a click
    // passes its event, which is not that.)
    if (anyway !== true) {
      const blocked = nodes
        .map((node) => ({ node, notice: getNodeNotice(node.type, node.data as Record<string, unknown>) }))
        .filter((b): b is { node: Node; notice: string } => !!b.notice);
      if (blocked.length) {
        const { node, notice } = blocked[0];
        // (by its name on the canvas, else its kind's: "Image Gen", not "image_gen")
        const data = node.data as { label?: unknown };
        const name = (typeof data.label === 'string' && data.label.trim()) || getModuleLoader().getNodeDefinition(String(node.type))?.name || node.type;
        onShowToast?.(
          `“${name}” cannot run now: ${notice}${blocked.length > 1 ? ` (and ${blocked.length - 1} more step${blocked.length > 2 ? 's' : ''})` : ''}`,
          'warning',
          12000,
          { label: 'Run anyway', onClick: () => void runRef.current(true) },
          `run-check:${flowId}`,
        );
        return;
      }
    }

    // Show run modal if there are input nodes to configure
    if (hasInputNodes(nodes)) {
      setShowRunModal(true);
      return;
    }

    // No input nodes - proceed directly
    proceedWithRun();
  }, [getWorkflowGraph, onShowToast, isMacro, nodes, proceedWithRun, hasInputNodes, flowId]);
  const runRef = useRef(runWorkflow);
  runRef.current = runWorkflow;

  // Stop workflow via job queue
  const stopWorkflow = useCallback(() => {
    // (the flow's live job, running or waiting: not an older one this page started)
    const live = getJobForFlow(flowId);
    const id = live && (live.status === 'running' || live.status === 'pending') ? live.id : currentJobId;
    if (id) jobManager.abort(id);
  }, [currentJobId, jobManager, getJobForFlow, flowId]);

  // Clear logs
  const clearLogs = useCallback(() => {
    setLocalLogs([]);
    setManualJobId(null);
  }, []);

  // Modal controls
  const closeRunModal = useCallback(() => {
    setShowRunModal(false);
  }, []);

  const closeServiceDialog = useCallback(() => {
    setShowServiceDialog(false);
  }, []);

  const proceedAfterServiceDialog = useCallback(() => {
    setShowServiceDialog(false);
    submitWorkflow();
  }, [submitWorkflow]);

  const finishFlowTransition = useCallback(() => {
    setFlowTransitioning(false);
  }, []);

  return {
    // State
    isRunning,
    isWaiting,
    currentJobId,
    logs,
    showRunModal,
    showServiceDialog,
    flowTransitioning,
    // Actions
    runWorkflow,
    stopWorkflow,
    clearLogs,
    closeRunModal,
    confirmRunModal,
    closeServiceDialog,
    proceedAfterServiceDialog,
    finishFlowTransition,
  };
}
