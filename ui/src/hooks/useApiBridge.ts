/**
 * API Bridge Hook
 *
 * Listens for API requests from the Rust backend and processes them
 * using the JobManager and available flows.
 */

import { useEffect, useRef } from 'react';
import { listen, emit } from '@tauri-apps/api/event';
import type { JobManager, Flow, Job, WorkflowInputs, WorkflowGraph } from 'oaiy-core';
import { getBundledModulesArray, getBundledNodeDefinitions, getBundledNodeDefinition } from 'oaiy-core';
import { createLogger } from '../utils/logger';
import { readConsoleBuffer, clearConsoleBuffer, type ConsoleLevel } from '../utils/consoleBuffer';

const logger = createLogger('ApiBridge');

interface ApiRequest {
  requestId: string;
  command: string;
  payload: Record<string, unknown>;
}

interface JobStatusResponse {
  id: string;
  flowId: string;
  flowName: string;
  status: string;
  position: number | null;
  submittedAt: number;
  startedAt?: number;
  completedAt?: number;
  result?: unknown;
  files?: string[];
  jobError?: string;  // Named 'jobError' to avoid API server treating as error response
  logs?: Array<{
    id: string;
    timestamp: number;
    source: string;
    message: string;
    type?: string;
  }>;
  // Claude-as-AI: Pending AI request when status is 'awaiting_ai'
  pendingAIRequest?: {
    continueToken: string;
    nodeId: string;
    systemPrompt: string;
    userPrompt: string;
    images?: string[];
    history?: Array<{ role: 'user' | 'assistant'; content: string }>;
    createdAt: number;
  };
}

interface FlowInfo {
  id: string;
  name: string;
  description?: string;
}

/**
 * Extract file paths from job result
 * Scans the result object for strings that look like actual file paths
 */
function extractFilePaths(result: unknown): string[] {
  const files: string[] = [];
  const seen = new Set<string>();

  // Common output file extensions
  const fileExtensions = /\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg|pdf|json|txt|csv|xml|html|md|mp4|mp3|wav|ogg|webm|avi|mov|zip|tar|gz)$/i;

  function scan(value: unknown): void {
    if (typeof value === 'string') {
      // Skip empty strings and very short strings
      if (value.length < 5) return;

      // Skip URLs (http, https, data URIs)
      if (/^(https?:\/\/|data:|blob:)/i.test(value)) return;

      // Skip strings that look like code/JSON/HTML (contain special characters at start)
      if (/^[\s[\]{}"'<>]/.test(value)) return;

      // Must have a file extension to be considered a file path
      if (!fileExtensions.test(value)) return;

      // Check if it looks like a proper file path:
      // - Starts with / (Unix absolute)
      // - Starts with ./ or ../ (relative)
      // - Starts with drive letter like C:\ (Windows)
      // - Or just a filename with extension
      const isFilePath =
        /^\/[^/]/.test(value) ||           // Unix absolute path (not just /)
        /^\.\.?\//.test(value) ||          // Relative path
        /^[A-Za-z]:[/\\]/.test(value) ||   // Windows drive path
        /^[A-Za-z0-9_-]+\.[a-z]+$/i.test(value);  // Simple filename.ext

      if (isFilePath && !seen.has(value)) {
        seen.add(value);
        // Extract the filename from the path
        const fileName = value.split(/[/\\]/).pop();
        if (fileName && fileExtensions.test(fileName)) {
          files.push(`/api/files/${encodeURIComponent(fileName)}`);
        }
      }
    } else if (Array.isArray(value)) {
      value.forEach(scan);
    } else if (value && typeof value === 'object') {
      Object.values(value).forEach(scan);
    }
  }

  scan(result);
  return files;
}

/**
 * Convert a Job to the API response format
 */
function jobToResponse(job: Job, position: number | null): JobStatusResponse {
  return {
    id: job.id,
    flowId: job.flowId,
    flowName: job.flowName,
    status: job.status,
    position,
    submittedAt: job.submittedAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt,
    result: job.result,
    files: extractFilePaths(job.result),
    // Use 'jobError' instead of 'error' to avoid API server treating this as an error response
    jobError: job.error,
    logs: job.logs.map((log) => ({
      id: log.id,
      timestamp: log.timestamp,
      source: log.source,
      message: log.message,
      type: log.type,
    })),
    // Include pending AI request when job is awaiting AI response
    pendingAIRequest: job.pendingAIRequest,
  };
}

/**
 * Callbacks for flow operations
 */
export interface ApiBridgeCallbacks {
  onCreateFlow?: (name: string) => Flow;
  onDeleteFlow?: (flowId: string) => void;
  onUpdateFlow?: (flowId: string, updates: Partial<Omit<Flow, 'id' | 'createdAt'>>) => void;
  onUpdateFlowGraph?: (flowId: string, graph: WorkflowGraph) => void;
  // System control callbacks
  onClearCache?: () => Promise<void>;
  onReloadMacros?: () => Promise<void>;
  onRecompilePackages?: () => Promise<{ success: boolean; output?: string; error?: string }>;
}

/**
 * Hook to bridge API requests from Rust to the JobManager
 */
export function useApiBridge(
  jobManager: JobManager | null,
  flows: Flow[],
  onCreateFlow?: (name: string) => Flow,
  onDeleteFlow?: (flowId: string) => void,
  callbacks?: ApiBridgeCallbacks
): void {
  // Use refs to avoid re-subscribing when these change
  const jobManagerRef = useRef(jobManager);
  const flowsRef = useRef(flows);
  const createFlowRef = useRef(onCreateFlow);
  const deleteFlowRef = useRef(onDeleteFlow);
  const updateFlowRef = useRef(callbacks?.onUpdateFlow);
  const updateFlowGraphRef = useRef(callbacks?.onUpdateFlowGraph);
  const clearCacheRef = useRef(callbacks?.onClearCache);
  const reloadMacrosRef = useRef(callbacks?.onReloadMacros);
  const recompilePackagesRef = useRef(callbacks?.onRecompilePackages);

  // Track API-created flows that may not be in props yet (due to React async state)
  const pendingFlowsRef = useRef<Map<string, Flow>>(new Map());
  // Track API-deleted flows that may still be in props (due to React async state)
  const deletedFlowIdsRef = useRef<Set<string>>(new Set());

  // Keep refs in sync
  useEffect(() => {
    jobManagerRef.current = jobManager;
  }, [jobManager]);

  useEffect(() => {
    // Update flowsRef with props, but preserve any pending API-created flows
    // and exclude any pending API-deleted flows
    const propFlowIds = new Set(flows.map((f) => f.id));

    // Remove from pending if now in props
    for (const [id] of pendingFlowsRef.current) {
      if (propFlowIds.has(id)) {
        pendingFlowsRef.current.delete(id);
      }
    }

    // Remove from deleted if no longer in props
    for (const id of deletedFlowIdsRef.current) {
      if (!propFlowIds.has(id)) {
        deletedFlowIdsRef.current.delete(id);
      }
    }

    // Merge: props + pending - deleted
    const mergedFlows = [
      ...flows.filter((f) => !deletedFlowIdsRef.current.has(f.id)),
      ...Array.from(pendingFlowsRef.current.values()),
    ];
    flowsRef.current = mergedFlows;
  }, [flows]);

  useEffect(() => {
    createFlowRef.current = onCreateFlow;
  }, [onCreateFlow]);

  useEffect(() => {
    deleteFlowRef.current = onDeleteFlow;
  }, [onDeleteFlow]);

  useEffect(() => {
    updateFlowRef.current = callbacks?.onUpdateFlow;
  }, [callbacks?.onUpdateFlow]);

  useEffect(() => {
    updateFlowGraphRef.current = callbacks?.onUpdateFlowGraph;
  }, [callbacks?.onUpdateFlowGraph]);

  useEffect(() => {
    clearCacheRef.current = callbacks?.onClearCache;
  }, [callbacks?.onClearCache]);

  useEffect(() => {
    reloadMacrosRef.current = callbacks?.onReloadMacros;
  }, [callbacks?.onReloadMacros]);

  useEffect(() => {
    recompilePackagesRef.current = callbacks?.onRecompilePackages;
  }, [callbacks?.onRecompilePackages]);

  useEffect(() => {
    const handleRequest = async (event: { payload: ApiRequest }) => {
      const { requestId, command, payload } = event.payload;
      let responseData: unknown = null;

      const manager = jobManagerRef.current;
      const availableFlows = flowsRef.current;

      try {
        switch (command) {
          case 'create_job': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const flowId = payload.flowId as string;
            const inputs = payload.inputs as WorkflowInputs | undefined;
            const priority = (payload.priority as number) || 1;
            const useClaudeForAI = payload.useClaudeForAI as boolean | undefined;

            // Find the flow
            const flow = availableFlows.find((f) => f.id === flowId);
            if (!flow) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            // Submit the job with Claude-as-AI mode if requested
            if (useClaudeForAI) {
              logger.debug('Claude-as-AI mode enabled for job');
            }

            const jobId = manager.submit(
              flow.id,
              flow.name,
              flow.graph,
              inputs,
              priority,
              useClaudeForAI || false
            );

            // Get the actual job status (might have started immediately)
            const submittedJob = manager.getJob(jobId);
            const position = manager.getQueuePosition(jobId);

            responseData = {
              jobId,
              status: submittedJob?.status || 'pending',
              position: position ?? (submittedJob?.status === 'running' ? 0 : 1),
            };
            break;
          }

          case 'get_job': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const jobId = payload.jobId as string;
            logger.debug(`get_job called for: ${jobId}`, { allJobs: manager.getAllJobs().map(j => ({ id: j.id, status: j.status })) });
            const job = manager.getJob(jobId);

            if (!job) {
              logger.debug(`Job NOT FOUND: ${jobId}`);
              responseData = { error: 'Job not found' };
              break;
            }
            logger.debug(`Job found: ${job.id} ${job.status}`);

            const position = manager.getQueuePosition(jobId);
            responseData = jobToResponse(job, position);
            break;
          }

          case 'get_job_logs': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const jobId = payload.jobId as string;
            const job = manager.getJob(jobId);

            if (!job) {
              responseData = { error: 'Job not found' };
              break;
            }

            responseData = {
              logs: job.logs.map((log) => ({
                id: log.id,
                timestamp: log.timestamp,
                source: log.source,
                message: log.message,
                type: log.type,
              })),
            };
            break;
          }

          case 'list_jobs': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const statusFilter = payload.status as string | undefined;
            const limit = payload.limit as number | undefined;

            let jobs = manager.getAllJobs();

            // Filter by status if specified
            if (statusFilter) {
              jobs = jobs.filter((j) => j.status === statusFilter);
            }

            // Limit results
            if (limit && limit > 0) {
              jobs = jobs.slice(0, limit);
            }

            responseData = {
              jobs: jobs.map((job) => {
                const position = manager.getQueuePosition(job.id);
                return jobToResponse(job, position);
              }),
            };
            break;
          }

          case 'abort_job': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const jobId = payload.jobId as string;
            const job = manager.getJob(jobId);

            if (!job) {
              responseData = { error: 'Job not found' };
              break;
            }

            if (job.status === 'completed' || job.status === 'failed' || job.status === 'aborted') {
              responseData = { error: 'Job already finished' };
              break;
            }

            manager.abort(jobId);
            responseData = { success: true, message: 'Job abort requested' };
            break;
          }

          case 'continue_job': {
            if (!manager) {
              responseData = { error: 'JobManager not initialized' };
              break;
            }

            const continueToken = payload.continueToken as string;
            const response = payload.response as string;

            if (!continueToken) {
              responseData = { error: 'Continue token is required' };
              break;
            }

            if (!response) {
              responseData = { error: 'Response is required' };
              break;
            }

            // Capture the target job BEFORE continuing — continueWithAIResponse
            // clears pendingAIRequest, so we must match by continueToken first.
            const targetJob = manager.getAllJobs().find(j =>
              j.pendingAIRequest?.continueToken === continueToken
            );

            // Continue the job with the AI response
            const continued = manager.continueWithAIResponse(continueToken, response);

            if (!continued) {
              responseData = {
                error: 'No pending AI request found for this continue token. ' +
                       'The request may have expired or already been continued.',
              };
              break;
            }

            // Re-fetch the specific job we continued to report its current status
            const continuedJob = targetJob ? manager.getJob(targetJob.id) : undefined;

            responseData = {
              success: true,
              message: 'Job continued with AI response',
              jobId: targetJob?.id,
              status: continuedJob?.status || 'running',
            };
            break;
          }

          case 'list_flows': {
            responseData = {
              flows: availableFlows.map((flow): FlowInfo => ({
                id: flow.id,
                name: flow.name,
                description: flow.description,
              })),
            };
            break;
          }

          case 'create_flow': {
            const createFlow = createFlowRef.current;
            if (!createFlow) {
              responseData = { error: 'Flow creation not available' };
              break;
            }

            const name = payload.name as string;
            if (!name || typeof name !== 'string') {
              responseData = { error: 'Flow name is required' };
              break;
            }

            try {
              const newFlow = createFlow(name);

              // If a graph was provided, update the flow's graph
              const graph = payload.graph as WorkflowGraph | undefined;
              if (graph) {
                newFlow.graph = graph;
                // Also call updateFlowGraph if available to persist
                const updateFlowGraph = updateFlowGraphRef.current;
                if (updateFlowGraph) {
                  updateFlowGraph(newFlow.id, graph);
                }
              }

              // Track as pending until React state catches up
              pendingFlowsRef.current.set(newFlow.id, newFlow);
              // Update flowsRef immediately
              flowsRef.current = [...flowsRef.current, newFlow];
              responseData = {
                id: newFlow.id,
                name: newFlow.name,
                description: newFlow.description,
              };
            } catch (e) {
              responseData = { error: `Failed to create flow: ${String(e)}` };
            }
            break;
          }

          case 'delete_flow': {
            const deleteFlow = deleteFlowRef.current;
            if (!deleteFlow) {
              responseData = { error: 'Flow deletion not available' };
              break;
            }

            const flowId = payload.flowId as string;
            if (!flowId || typeof flowId !== 'string') {
              responseData = { error: 'Flow ID is required' };
              break;
            }

            // Check if flow exists in our local ref
            const flowToDelete = flowsRef.current.find((f) => f.id === flowId);
            if (!flowToDelete) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            try {
              deleteFlow(flowId);
              // Track as deleted until React state catches up
              deletedFlowIdsRef.current.add(flowId);
              // Remove from pending if it was API-created
              pendingFlowsRef.current.delete(flowId);
              // Update flowsRef immediately
              flowsRef.current = flowsRef.current.filter((f) => f.id !== flowId);
              responseData = { success: true, message: 'Flow deleted' };
            } catch (e) {
              responseData = { error: `Failed to delete flow: ${String(e)}` };
            }
            break;
          }

          case 'get_flow': {
            const flowId = payload.flowId as string;
            if (!flowId || typeof flowId !== 'string') {
              responseData = { error: 'Flow ID is required' };
              break;
            }

            const flow = availableFlows.find((f) => f.id === flowId);
            if (!flow) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            responseData = {
              id: flow.id,
              name: flow.name,
              description: flow.description,
              isMacro: flow.isMacro,
              graph: flow.graph,
              createdAt: flow.createdAt,
              updatedAt: flow.updatedAt,
            };
            break;
          }

          case 'update_flow': {
            const updateFlow = updateFlowRef.current;
            if (!updateFlow) {
              responseData = { error: 'Flow update not available' };
              break;
            }

            const flowId = payload.flowId as string;
            if (!flowId || typeof flowId !== 'string') {
              responseData = { error: 'Flow ID is required' };
              break;
            }

            const flowToUpdate = flowsRef.current.find((f) => f.id === flowId);
            if (!flowToUpdate) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            try {
              const updates: Partial<Omit<Flow, 'id' | 'createdAt'>> = {};
              if (payload.name !== undefined) updates.name = payload.name as string;
              if (payload.description !== undefined) updates.description = payload.description as string;
              if (payload.graph !== undefined) updates.graph = payload.graph as WorkflowGraph;

              updateFlow(flowId, updates);

              // Update flowsRef immediately
              flowsRef.current = flowsRef.current.map((f) =>
                f.id === flowId ? { ...f, ...updates } : f
              );

              responseData = { success: true, message: 'Flow updated' };
            } catch (e) {
              responseData = { error: `Failed to update flow: ${String(e)}` };
            }
            break;
          }

          case 'update_flow_graph': {
            const updateFlowGraph = updateFlowGraphRef.current;
            if (!updateFlowGraph) {
              responseData = { error: 'Flow graph update not available' };
              break;
            }

            const flowId = payload.flowId as string;
            if (!flowId || typeof flowId !== 'string') {
              responseData = { error: 'Flow ID is required' };
              break;
            }

            const graph = payload.graph as WorkflowGraph;
            if (!graph) {
              responseData = { error: 'Graph is required' };
              break;
            }

            const flowToUpdate = flowsRef.current.find((f) => f.id === flowId);
            if (!flowToUpdate) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            try {
              updateFlowGraph(flowId, graph);

              // Update flowsRef immediately
              flowsRef.current = flowsRef.current.map((f) =>
                f.id === flowId ? { ...f, graph } : f
              );

              responseData = { success: true, message: 'Flow graph updated' };
            } catch (e) {
              responseData = { error: `Failed to update flow graph: ${String(e)}` };
            }
            break;
          }

          case 'validate_flow': {
            const flowId = payload.flowId as string;
            if (!flowId || typeof flowId !== 'string') {
              responseData = { error: 'Flow ID is required' };
              break;
            }

            const flowToValidate = availableFlows.find((f) => f.id === flowId);
            if (!flowToValidate) {
              responseData = { error: `Flow not found: ${flowId}` };
              break;
            }

            // Comprehensive validation - check for common issues
            const errors: string[] = [];
            const warnings: string[] = [];
            const graph = flowToValidate.graph;

            // Check if graph has nodes
            if (!graph.nodes || graph.nodes.length === 0) {
              errors.push('Flow has no nodes');
            }

            const nodeIds = new Set(graph.nodes.map((n) => n.id));
            const connectedNodes = new Set<string>();

            // Check edges
            for (const edge of graph.edges || []) {
              // Check for orphaned edges (edges referencing non-existent nodes)
              if (!nodeIds.has(edge.source)) {
                errors.push(`Edge references non-existent source node: ${edge.source}`);
              } else {
                connectedNodes.add(edge.source);
              }
              if (!nodeIds.has(edge.target)) {
                errors.push(`Edge references non-existent target node: ${edge.target}`);
              } else {
                connectedNodes.add(edge.target);
              }

              // Check for self-loops
              if (edge.source === edge.target) {
                errors.push(`Self-loop detected: node "${edge.source}" is connected to itself`);
              }
            }

            // Check for duplicate edges
            const edgeSet = new Set<string>();
            for (const edge of graph.edges || []) {
              const edgeKey = `${edge.source}:${edge.sourceHandle || ''}->${edge.target}:${edge.targetHandle || ''}`;
              if (edgeSet.has(edgeKey)) {
                warnings.push(`Duplicate edge detected: ${edge.source} -> ${edge.target}`);
              }
              edgeSet.add(edgeKey);
            }

            // Check for disconnected nodes (warning, not error)
            if (graph.nodes.length > 1) {
              for (const node of graph.nodes) {
                if (!connectedNodes.has(node.id)) {
                  warnings.push(`Node "${node.id}" (${node.type}) is not connected to any other nodes`);
                }
              }
            }

            // Check for nodes without required configuration (basic check)
            for (const node of graph.nodes) {
              if (node.type === 'ai_llm' && !node.data?.model) {
                warnings.push(`AI node "${node.id}" has no model configured`);
              }
              if (node.type === 'file_read' && !node.data?.path) {
                warnings.push(`File read node "${node.id}" has no path configured`);
              }
            }

            responseData = {
              valid: errors.length === 0,
              errors,
              warnings,
            };
            break;
          }

          case 'list_nodes': {
            const nodes = getBundledNodeDefinitions();

            responseData = {
              nodes: nodes.map((node) => ({
                id: node.id,
                name: node.name,
                description: node.description,
                icon: node.icon,
                color: node.color,
                tags: node.tags,
              })),
            };
            break;
          }

          case 'get_node_definition': {
            const nodeType = payload.nodeType as string;
            if (!nodeType || typeof nodeType !== 'string') {
              responseData = { error: 'Node type is required' };
              break;
            }

            const nodeDef = getBundledNodeDefinition(nodeType);
            if (!nodeDef) {
              responseData = { error: `Node type not found: ${nodeType}` };
              break;
            }

            responseData = {
              id: nodeDef.id,
              name: nodeDef.name,
              description: nodeDef.description,
              icon: nodeDef.icon,
              color: nodeDef.color,
              tags: nodeDef.tags,
              inputs: nodeDef.inputs,
              outputs: nodeDef.outputs,
              properties: nodeDef.properties,
            };
            break;
          }

          case 'list_modules': {
            const modules = getBundledModulesArray();

            responseData = {
              modules: modules.map((module) => ({
                id: module.manifest.id,
                name: module.manifest.name,
                version: module.manifest.version,
                description: module.manifest.description,
                category: module.manifest.category,
                icon: module.manifest.icon,
                color: module.manifest.color,
                nodes: module.nodes.map((n) => n.id),
              })),
            };
            break;
          }

          case 'clear_cache': {
            const clearCache = clearCacheRef.current;
            if (!clearCache) {
              // Default: clear only temporary cache data, preserve user settings
              // Keys to preserve (user data):
              // - oaiy_project: project settings, providers, flows
              // - oaiy_user_macros_v2: user-created macros
              // Keys safe to clear (temporary/cache):
              // - oaiy_workflow_autosave: temporary autosave
              // - oaiy_run_history: run history cache
              // - skipSplash: temporary flag
              try {
                const keysToPreserve = ['oaiy_project', 'oaiy_user_macros_v2'];
                const keysToClear = ['oaiy_workflow_autosave', 'oaiy_run_history', 'skipSplash'];

                // Only clear specific cache keys, not user settings
                keysToClear.forEach(key => {
                  localStorage.removeItem(key);
                });
                sessionStorage.clear();

                responseData = {
                  success: true,
                  message: 'Cache cleared (temporary data only, user settings preserved)',
                  clearedKeys: keysToClear,
                  preservedKeys: keysToPreserve
                };
              } catch (e) {
                responseData = { error: `Failed to clear cache: ${String(e)}` };
              }
            } else {
              try {
                await clearCache();
                responseData = { success: true, message: 'Cache cleared' };
              } catch (e) {
                responseData = { error: `Failed to clear cache: ${String(e)}` };
              }
            }
            break;
          }

          case 'reload_macros': {
            const reloadMacros = reloadMacrosRef.current;
            if (!reloadMacros) {
              responseData = { error: 'Macro reload not available - no callback provided' };
            } else {
              try {
                await reloadMacros();
                responseData = { success: true, message: 'Macros reloaded' };
              } catch (e) {
                responseData = { error: `Failed to reload macros: ${String(e)}` };
              }
            }
            break;
          }

          case 'restart_app': {
            // Send response first, then restart
            responseData = { success: true, message: 'Restarting application...' };
            // Schedule restart after response is sent
            setTimeout(() => {
              // Set flag to skip splash screen on restart
              localStorage.setItem('skipSplash', 'true');
              window.location.reload();
            }, 100);
            break;
          }

          case 'recompile_packages': {
            const recompilePackages = recompilePackagesRef.current;
            if (!recompilePackages) {
              responseData = { error: 'Package recompilation not available - no callback provided' };
            } else {
              try {
                const result = await recompilePackages();
                responseData = result;
              } catch (e) {
                responseData = { error: `Failed to recompile packages: ${String(e)}` };
              }
            }
            break;
          }

          case 'get_console_logs': {
            const sinceId = typeof payload.sinceId === 'number' ? payload.sinceId : undefined;
            const limit = typeof payload.limit === 'number' ? payload.limit : undefined;
            const rawLevels = payload.levels;
            const levels: ConsoleLevel[] | undefined = Array.isArray(rawLevels)
              ? (rawLevels.filter((l) => typeof l === 'string') as ConsoleLevel[])
              : undefined;
            responseData = readConsoleBuffer({ sinceId, limit, levels });
            break;
          }

          case 'clear_console_logs': {
            clearConsoleBuffer();
            responseData = { cleared: true };
            break;
          }

          default:
            responseData = { error: `Unknown command: ${command}` };
        }
      } catch (e) {
        responseData = { error: String(e) };
      }

      // Send response back to Rust
      await emit('api:response', {
        requestId,
        data: responseData,
      });
    };

    // Listen for API requests
    const unlisten = listen<ApiRequest>('api:request', handleRequest);

    return () => {
      unlisten.then((f) => f()).catch(() => {
        // Ignore cleanup errors - listener may not have been set up
      });
    };
  }, []);
}
