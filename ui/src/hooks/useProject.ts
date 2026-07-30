import { useState, useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type {
  Flow,
  OAIYProject,
  WorkflowGraph,
  RunRecord,
  LLMEndpoint,
  ImageGenEndpoint,
  HttpPreset,
  ProjectConstant,
  ProjectSettings,
} from 'oaiy-core';
import { invoke } from '@tauri-apps/api/core';
import {
  loadUserMacros,
  saveUserMacro,
  deleteUserMacro as deleteUserMacroFromStorage,
  getAllMacroOverrides,
  saveMacroOverride,
  revertMacroOverride,
} from '../utils/userMacroStorage';
import { createLogger } from '../utils/logger';

const logger = createLogger('Project');
import {
  defaultLLMEndpoints,
  defaultImageGenEndpoints,
  defaultConstants,
  defaultSettings,
  createEmptyProject,
  createDemoProject,
  defaultExampleFlows,
} from '../utils/ProjectDefaults';
import {
  exportProjectToFile,
  importProjectFromFile,
  redactConstantsForStorage,
  redactRunHistorySecrets,
} from '../utils/ProjectIO';
import { notifyStorageQuotaExceeded, markStorageWriteSucceeded } from '../lib/storageQuota';

// Load macros from the /macros folder via Tauri
async function loadMacrosFromFolder(): Promise<Flow[]> {
  try {
    const macroData = await invoke<unknown[]>('load_all_macros');
    return macroData.map((data) => {
      const macro = data as Record<string, unknown>;
      const nodes = (macro.nodes as Flow['graph']['nodes']) || [];
      const edges = (macro.edges as Flow['graph']['edges']) || [];

      // Extract macroMetadata from macro_input and macro_output nodes
      const macroInputNodes = nodes.filter(n => n.type === 'macro_input');
      const macroOutputNodes = nodes.filter(n => n.type === 'macro_output');

      const inputs = macroInputNodes.map(node => ({
        id: node.id,
        name: String(node.data?.name || 'input'),
        type: String(node.data?.inputType || 'any'),
        required: Boolean(node.data?.required),
        defaultValue: node.data?.defaultValue ? String(node.data.defaultValue) : undefined,
      }));

      const outputs = macroOutputNodes.map(node => ({
        id: node.id,
        name: String(node.data?.name || 'output'),
        type: String(node.data?.outputType || 'any'),
      }));

      return {
        id: macro.id as string,
        name: macro.name as string,
        description: (macro.description as string) || '',
        isMacro: true,
        isBuiltIn: true, // Mark as built-in so they're not saved to user storage
        tags: (macro.tags as string[]) || [],
        createdAt: (macro.createdAt as string) || new Date().toISOString(),
        updatedAt: (macro.updatedAt as string) || new Date().toISOString(),
        graph: { nodes, edges },
        // Populate macroMetadata for NodePalette display
        macroMetadata: {
          inputs,
          outputs,
        },
      } as Flow;
    });
  } catch (error) {
    logger.warn('Failed to load macros from folder', { error });
    return [];
  }
}

const PROJECT_STORAGE_KEY = 'oaiy_project';
const RUN_HISTORY_KEY = 'oaiy_run_history';
const MAX_RUN_HISTORY = 50;

// Check if a flow is a user-created macro (completely new, not an override)
const isUserCreatedMacro = (flow: Flow): boolean => {
  return flow.id.startsWith('user-macro-') ||
         (flow.tags?.includes('user-created') ?? false);
};

// Note: Built-in macros are now loaded from the /macros folder via loadMacrosFromFolder()
// They are NOT saved to localStorage, so they always load fresh from disk

// Load project from localStorage
// Note: Built-in macros are NOT stored in localStorage - they load fresh from /macros folder
const loadProjectFromStorage = (): OAIYProject | null => {
  try {
    const saved = localStorage.getItem(PROJECT_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Ensure required fields exist
      if (parsed.flows && Array.isArray(parsed.flows)) {
        // Merge constants: start with defaults, then overlay saved values
        // This ensures new default constants (like HF_TOKEN) are added
        const savedConstants = parsed.constants || [];
        const savedConstantKeys = new Set(savedConstants.map((c: ProjectConstant) => c.key));

        // Start with saved constants, then add any missing defaults
        const mergedConstants = [
          ...savedConstants,
          ...defaultConstants.filter(dc => !savedConstantKeys.has(dc.key)),
        ];

        return {
          ...createEmptyProject(),
          ...parsed,
          // flows from localStorage are user flows only (built-in macros load from disk)
          flows: parsed.flows,
          // Use merged constants to include new defaults
          constants: mergedConstants,
        };
      }
    }
  } catch (error) {
    // Log storage errors in development to help debug issues
    if (import.meta.env.DEV) {
      logger.warn('Failed to load from localStorage', { error: error instanceof Error ? error.message : error });
    }
  }
  return null;
};

// Load run history from localStorage
const loadRunHistory = (): RunRecord[] => {
  try {
    const saved = localStorage.getItem(RUN_HISTORY_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      return Array.isArray(parsed) ? parsed : [];
    }
  } catch (error) {
    // Log storage errors in development
    if (import.meta.env.DEV) {
      logger.warn('Failed to load run history from localStorage', { error: error instanceof Error ? error.message : error });
    }
  }
  return [];
};

export function useProject() {
  const [project, setProject] = useState<OAIYProject>(() => {
    return loadProjectFromStorage() || createDemoProject();
  });
  // Derive from the already-parsed `project` (it's in scope here, initialized just
  // above) instead of re-reading + re-parsing the whole project blob from storage.
  // Equivalent: loaded case -> project.flows === parsed.flows; demo case ->
  // project.flows === [...defaultExampleFlows]. (Do NOT hoist loadProjectFromStorage
  // to the hook body — that would parse on every render.)
  const [activeFlowId, setActiveFlowId] = useState<string | null>(() => project.flows[0]?.id ?? null);
  const [runHistory, setRunHistory] = useState<RunRecord[]>(loadRunHistory);
  const [resetCounter, setResetCounter] = useState(0);
  const [userDataLoaded, setUserDataLoaded] = useState(false);
  // Track which built-in macros have user overrides (saved to file)
  const [macroOverrideIds, setMacroOverrideIds] = useState<Set<string>>(new Set());
  // Track original macro states for dirty detection (from disk or saved override)
  const [originalMacroStates, setOriginalMacroStates] = useState<Map<string, Flow>>(new Map());

  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Load user macros, folder macros, overrides, and secrets from file system on startup
  useEffect(() => {
    logger.debug(`Effect running, userDataLoaded: ${userDataLoaded}`);
    if (userDataLoaded) return;

    const loadUserDataAsync = async () => {
      try {
        logger.debug('loadUserDataAsync starting', {
          constantCount: project.constants?.length,
          constants: project.constants?.map(c => `${c.key}:${c.isSecret}`).join(', '),
        });

        // Get the keys of all secret constants to load
        const secretKeys = (project.constants || [])
          .filter(c => c.isSecret)
          .map(c => c.key);

        logger.debug(`Loading secrets for keys: ${secretKeys.join(', ')}`);

        // Load user macros, folder macros, overrides, and secrets in parallel
        const [userMacros, folderMacros, overrides, secrets] = await Promise.all([
          loadUserMacros(),
          loadMacrosFromFolder(),
          getAllMacroOverrides(),
          secretKeys.length > 0
            ? invoke<Record<string, string>>('get_secrets', { keys: secretKeys })
            : Promise.resolve({} as Record<string, string>),
        ]);

        logger.debug(`Loaded secrets: ${Object.keys(secrets).length} values`);

        if (!isMountedRef.current) return;

        const overrideIds = new Set(Object.keys(overrides));
        setMacroOverrideIds(overrideIds);

        // Store original macro states for dirty detection
        // For built-in macros: use override if exists, otherwise use disk version
        const originalStates = new Map<string, Flow>();
        for (const macro of folderMacros) {
          const original = overrides[macro.id] || macro;
          originalStates.set(macro.id, JSON.parse(JSON.stringify(original)));
        }
        setOriginalMacroStates(originalStates);

        logger.debug(`Loaded ${userMacros.length} user macros, ${folderMacros.length} folder macros, and ${overrideIds.size} overrides`);

        setProject((prev) => {
          let updatedFlows = [...prev.flows];

          // Add folder macros (from /macros directory) that aren't already in the flows
          const existingIds = new Set(updatedFlows.map(f => f.id));
          const newFolderMacros = folderMacros.filter(m => !existingIds.has(m.id));
          if (newFolderMacros.length > 0) {
            updatedFlows = [...updatedFlows, ...newFolderMacros];
            // Update existingIds for user macro check
            newFolderMacros.forEach(m => existingIds.add(m.id));
          }

          // Apply overrides to built-in macros (replace with saved override version)
          if (overrideIds.size > 0) {
            updatedFlows = updatedFlows.map(flow => {
              if (flow.isBuiltIn && overrides[flow.id]) {
                const override = overrides[flow.id];
                // Extract macroMetadata if not present (for backwards compatibility)
                let macroMetadata = override.macroMetadata;
                if (!macroMetadata && override.graph) {
                  const nodes = override.graph.nodes || [];
                  const macroInputNodes = nodes.filter(n => n.type === 'macro_input');
                  const macroOutputNodes = nodes.filter(n => n.type === 'macro_output');
                  macroMetadata = {
                    inputs: macroInputNodes.map(node => ({
                      id: node.id,
                      name: String(node.data?.name || 'input'),
                      type: String(node.data?.inputType || 'any'),
                      required: Boolean(node.data?.required),
                      defaultValue: node.data?.defaultValue ? String(node.data.defaultValue) : undefined,
                    })),
                    outputs: macroOutputNodes.map(node => ({
                      id: node.id,
                      name: String(node.data?.name || 'output'),
                      type: String(node.data?.outputType || 'any'),
                    })),
                  };
                }
                // Merge override with isBuiltIn flag preserved
                return { ...override, isBuiltIn: true, macroMetadata };
              }
              return flow;
            });
          }

          // Add user-created macros that aren't already in the flows
          const newUserMacros = userMacros.filter(m => !existingIds.has(m.id));
          if (newUserMacros.length > 0) {
            updatedFlows = [...updatedFlows, ...newUserMacros];
          }

          // Update constants with loaded secret values from secure storage
          const updatedConstants = Object.keys(secrets).length > 0
            ? (prev.constants || []).map(c => {
                if (c.isSecret && secrets[c.key]) {
                  return { ...c, value: secrets[c.key] };
                }
                return c;
              })
            : prev.constants;

          // Check if anything changed
          const flowsChanged = updatedFlows !== prev.flows;
          const constantsChanged = updatedConstants !== prev.constants;

          if (!flowsChanged && !constantsChanged) return prev;

          logger.debug(`Applied ${Object.keys(secrets).length} secrets to constants`);

          return {
            ...prev,
            flows: updatedFlows,
            constants: updatedConstants,
          };
        });

        setUserDataLoaded(true);
      } catch (error) {
        logger.warn('Failed to load user data', { error });
        setUserDataLoaded(true);
      }
    };

    loadUserDataAsync();
  }, [userDataLoaded]);

  // Auto-save project to localStorage (debounced).
  // IMPORTANT: Built-in macros are NOT saved to localStorage - they always load fresh from disk
  // Only user-created flows and explicitly saved macro overrides are persisted.
  //
  // Debounced because the effect fires on every `project` change — i.e. every
  // keystroke / node drag — and each run JSON.stringifies the WHOLE project
  // synchronously on the main thread. Without the debounce, heavy projects
  // stutter while editing and write the full blob dozens of times a second.
  // The latest pending snapshot is always flushed on unmount and on pagehide
  // (tab close / navigation), so the debounce never costs the user data.
  const writeProjectToStorage = useCallback((proj: OAIYProject) => {
    try {
      const flowsToSave = proj.flows.filter(flow => !flow.isBuiltIn);
      const projectToSave = {
        ...proj,
        flows: flowsToSave,
        constants: redactConstantsForStorage(proj.constants ?? []),
      };
      localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(projectToSave));
      markStorageWriteSucceeded(); // re-arm the quota warning for a later failure
    } catch (error) {
      if (import.meta.env.DEV) {
        logger.warn('Failed to save project', { error });
      }
      // Surfaces a single sticky toast to the user when localStorage is
      // full — without this the autosave silently no-ops and the user
      // loses every change made after the quota was hit. No-op for
      // non-quota errors (e.g. serialization bugs).
      notifyStorageQuotaExceeded('project', error);
    }
  }, []);

  const projectSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingProjectRef = useRef<OAIYProject | null>(null);
  const PROJECT_AUTOSAVE_DEBOUNCE_MS = 600;

  useEffect(() => {
    pendingProjectRef.current = project;
    if (projectSaveTimerRef.current) clearTimeout(projectSaveTimerRef.current);
    projectSaveTimerRef.current = setTimeout(() => {
      projectSaveTimerRef.current = null;
      if (pendingProjectRef.current) writeProjectToStorage(pendingProjectRef.current);
    }, PROJECT_AUTOSAVE_DEBOUNCE_MS);
    // NOTE: deliberately no flush in this effect's cleanup — it runs on every
    // project change and flushing there would defeat the debounce. The flush
    // lives in the mount/unmount effect below.
  }, [project, writeProjectToStorage]);

  // Flush the latest pending project on unmount + pagehide so a debounced
  // write in flight is never lost to a tab close / reload / navigation.
  useEffect(() => {
    const flush = () => {
      if (projectSaveTimerRef.current) {
        clearTimeout(projectSaveTimerRef.current);
        projectSaveTimerRef.current = null;
      }
      if (pendingProjectRef.current) writeProjectToStorage(pendingProjectRef.current);
    };
    // pagehide covers desktop tab-close/navigation; visibilitychange→hidden is
    // the reliable signal on mobile, where a backgrounded tab can be killed
    // without ever firing pagehide. flush() is idempotent so double-firing is
    // harmless.
    const onVisibility = () => { if (document.visibilityState === 'hidden') flush(); };
    window.addEventListener('pagehide', flush);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('pagehide', flush);
      document.removeEventListener('visibilitychange', onVisibility);
      flush();
    };
  }, [writeProjectToStorage]);

  // Auto-save run history
  useEffect(() => {
    try {
      const toSave = runHistory.map((run) => ({
        ...run,
        logs: redactRunHistorySecrets(run.logs),
      }));
      localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(toSave.slice(0, MAX_RUN_HISTORY)));
      markStorageWriteSucceeded(); // re-arm the quota warning for a later failure
    } catch (error) {
      if (import.meta.env.DEV) {
        logger.warn('Failed to save run history', { error });
      }
      notifyStorageQuotaExceeded('run-history', error);
    }
  }, [runHistory]);

  // Sync service lifecycle settings to Rust backend
  useEffect(() => {
    const settings = project.settings || {};
    const config = {
      default_idle_timeout_secs: settings.serviceIdleTimeoutSecs ?? 900,
      service_idle_timeouts: settings.serviceIdleTimeoutOverrides ?? {},
      startup_timeout_secs: settings.serviceStartupTimeoutSecs ?? 60,
    };

    invoke('set_lifecycle_config', { config }).catch((err) => {
      logger.warn('Failed to sync lifecycle config to Rust', { error: err });
    });
  }, [
    project.settings?.serviceIdleTimeoutSecs,
    project.settings?.serviceIdleTimeoutOverrides,
    project.settings?.serviceStartupTimeoutSecs,
  ]);

  // Get active flow
  const activeFlow = activeFlowId
    ? project.flows.find((f) => f.id === activeFlowId) || null
    : null;

  // ============================================
  // Flow Management
  // ============================================

  const createFlow = useCallback((name: string, graph?: WorkflowGraph): Flow => {
    const now = new Date().toISOString();
    const newFlow: Flow = {
      id: uuidv4(),
      name,
      createdAt: now,
      updatedAt: now,
      graph: graph || { nodes: [], edges: [] },
    };

    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: [...prev.flows, newFlow],
    }));

    setActiveFlowId(newFlow.id);
    return newFlow;
  }, []);

  const updateFlow = useCallback((flowId: string, updates: Partial<Omit<Flow, 'id' | 'createdAt'>>) => {
    const now = new Date().toISOString();
    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: prev.flows.map((f) =>
        f.id === flowId
          ? { ...f, ...updates, updatedAt: now }
          : f
      ),
    }));
  }, []);

  const deleteFlow = useCallback((flowId: string) => {
    // Use functional updates to avoid stale closure issues
    setProject((prev) => {
      const remaining = prev.flows.filter((f) => f.id !== flowId);

      // If deleting the active flow, switch to another one
      setActiveFlowId((currentActive) => {
        if (currentActive === flowId) {
          return remaining[0]?.id || null;
        }
        return currentActive;
      });

      return {
        ...prev,
        updatedAt: new Date().toISOString(),
        flows: remaining,
      };
    });
  }, []);

  const duplicateFlow = useCallback((flowId: string): Flow | null => {
    const source = project.flows.find((f) => f.id === flowId);
    if (!source) return null;

    const now = new Date().toISOString();
    const newFlow: Flow = {
      ...source,
      id: uuidv4(),
      name: `${source.name} (Copy)`,
      createdAt: now,
      updatedAt: now,
      graph: JSON.parse(JSON.stringify(source.graph)), // Deep clone
    };

    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: [...prev.flows, newFlow],
    }));

    return newFlow;
  }, [project.flows]);

  const renameFlow = useCallback((flowId: string, name: string) => {
    updateFlow(flowId, { name });
  }, [updateFlow]);

  const setFlowTags = useCallback((flowId: string, tags: string[]) => {
    updateFlow(flowId, { tags });
  }, [updateFlow]);

  const setFlowLocalOnly = useCallback((flowId: string, localOnly: boolean) => {
    updateFlow(flowId, { localOnly });
  }, [updateFlow]);

  // ============================================
  // Macro Management
  // ============================================

  /**
   * Convert a flow to a macro by extracting macro_input and macro_output nodes
   * and populating macroMetadata
   */
  const saveAsMacro = useCallback((flowId: string): { success: boolean; message: string } => {
    const flow = project.flows.find((f) => f.id === flowId);
    if (!flow) {
      return { success: false, message: 'Flow not found' };
    }

    // Find all macro_input and macro_output nodes
    const macroInputNodes = flow.graph.nodes.filter((n) => n.type === 'macro_input');
    const macroOutputNodes = flow.graph.nodes.filter((n) => n.type === 'macro_output');

    // Validate: must have at least one input or output
    if (macroInputNodes.length === 0 && macroOutputNodes.length === 0) {
      return {
        success: false,
        message: 'Flow must have at least one Macro Input or Macro Output node to be saved as a macro. Add these nodes to define the macro interface.',
      };
    }

    // Extract input port definitions
    const inputs = macroInputNodes.map((node) => ({
      id: node.id,
      name: String(node.data.name || 'input'),
      type: String(node.data.inputType || 'any'),
      required: Boolean(node.data.required),
      defaultValue: node.data.defaultValue ? String(node.data.defaultValue) : undefined,
    }));

    // Extract output port definitions
    const outputs = macroOutputNodes.map((node) => ({
      id: node.id,
      name: String(node.data.name || 'output'),
      type: String(node.data.outputType || 'any'),
    }));

    // Update the flow with macro metadata
    const now = new Date().toISOString();
    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: prev.flows.map((f) =>
        f.id === flowId
          ? {
              ...f,
              isMacro: true,
              // Mark as user-created so the macro save/delete paths recognise
              // it (they key on isUserCreatedMacro = 'user-macro-' id OR a
              // 'user-created' tag). Without this, saving a user macro fell
              // through to `return false` ("Failed to save"), deleting it was
              // rejected, and it persisted only via the project blob.
              tags: f.tags?.includes('user-created') ? f.tags : [...(f.tags || []), 'user-created'],
              macroMetadata: {
                inputs,
                outputs,
              },
              updatedAt: now,
            }
          : f
      ),
    }));

    return {
      success: true,
      message: `Macro saved with ${inputs.length} input(s) and ${outputs.length} output(s)`,
    };
  }, [project.flows]);

  /**
   * Remove macro status from a flow
   */
  const unmakeMacro = useCallback((flowId: string) => {
    updateFlow(flowId, { isMacro: false, macroMetadata: undefined });
  }, [updateFlow]);

  /**
   * Get all flows that are macros
   */
  const getMacros = useCallback((): Flow[] => {
    return project.flows.filter((f) => f.isMacro);
  }, [project.flows]);

  /**
   * Check if a macro has been modified from its original built-in version (saved to file)
   */
  const hasMacroBeenModified = useCallback((flowId: string): boolean => {
    return macroOverrideIds.has(flowId);
  }, [macroOverrideIds]);

  /**
   * Check if a macro has unsaved changes (in-memory edits not yet saved to file)
   */
  const isMacroDirty = useCallback((flowId: string): boolean => {
    const currentFlow = project.flows.find(f => f.id === flowId);
    if (!currentFlow || !currentFlow.isBuiltIn) return false;

    const originalState = originalMacroStates.get(flowId);
    if (!originalState) return false;

    // Fields to ignore when comparing - these are auto-populated defaults
    const defaultProviderFields = new Set([
      'endpoint', 'model', 'apiKeyConstant', 'provider', 'apiFormat', 'comfyuiUrl'
    ]);

    // Helper to normalize a graph for comparison (strip positions, temp fields, and default provider fields)
    const normalizeGraph = (graph: WorkflowGraph) => {
      const normalizedNodes = graph.nodes.map(n => ({
        id: n.id,
        type: n.type,
        // Strip position, underscore fields, and default provider fields for comparison
        data: Object.fromEntries(
          Object.entries(n.data || {}).filter(([key]) =>
            !key.startsWith('_') && !defaultProviderFields.has(key)
          )
        ),
      }));
      // Sort nodes by id for consistent comparison
      normalizedNodes.sort((a, b) => a.id.localeCompare(b.id));

      const normalizedEdges = graph.edges.map(e => ({
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      }));
      // Sort edges for consistent comparison
      normalizedEdges.sort((a, b) => {
        const srcCmp = a.source.localeCompare(b.source);
        if (srcCmp !== 0) return srcCmp;
        return a.target.localeCompare(b.target);
      });

      return { nodes: normalizedNodes, edges: normalizedEdges };
    };

    // Compare normalized graphs (ignoring positions and temp fields)
    const currentNormalized = normalizeGraph(currentFlow.graph);
    const originalNormalized = normalizeGraph(originalState.graph);
    return JSON.stringify(currentNormalized) !== JSON.stringify(originalNormalized);
  }, [project.flows, originalMacroStates]);

  /**
   * Discard unsaved changes to a built-in macro, reverting to the original state
   */
  const discardMacroChanges = useCallback((flowId: string): boolean => {
    const originalState = originalMacroStates.get(flowId);
    if (!originalState) {
      logger.warn('Cannot discard: no original state found', { flowId });
      return false;
    }

    const now = new Date().toISOString();
    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: prev.flows.map(f =>
        f.id === flowId ? { ...originalState, updatedAt: now } : f
      ),
    }));

    logger.debug(`Discarded unsaved changes for macro "${originalState.name}"`);
    return true;
  }, [originalMacroStates]);

  /**
   * Save changes to a macro (handles both built-in overrides and user macros)
   */
  const saveMacroChanges = useCallback(async (flowId: string): Promise<boolean> => {
    const flow = project.flows.find(f => f.id === flowId);
    if (!flow) {
      logger.warn(`Cannot save: flow not found ${flowId}`);
      return false;
    }

    if (!flow.isMacro) {
      logger.warn(`Cannot save: not a macro ${flowId}`);
      return false;
    }

    // For built-in macros, only save if there are actual changes
    if (flow.isBuiltIn && !isMacroDirty(flowId)) {
      logger.debug(`Skipping save for "${flow.name}" - no changes detected`);
      return true; // Return true since there's nothing to save (not an error)
    }

    // If it's a built-in macro (from /macros folder), save as override
    if (flow.isBuiltIn) {
      const saved = await saveMacroOverride(flowId, flow);
      if (saved) {
        setMacroOverrideIds(prev => new Set([...prev, flowId]));
        // Update the original state so it's no longer considered dirty
        setOriginalMacroStates(prev => {
          const newMap = new Map(prev);
          newMap.set(flowId, JSON.parse(JSON.stringify(flow)));
          return newMap;
        });
        logger.debug(`Saved override for built-in macro "${flow.name}"`);
      }
      return saved;
    }

    // If it's a user-created macro, save directly
    if (isUserCreatedMacro(flow)) {
      const saved = await saveUserMacro(flow);
      if (saved) {
        logger.debug(`Saved user macro "${flow.name}"`);
      }
      return saved;
    }

    return false;
  }, [project.flows, isMacroDirty]);

  /**
   * Revert a built-in macro to its original disk version (removes saved override)
   */
  const revertMacroToOriginal = useCallback(async (flowId: string): Promise<boolean> => {
    const currentFlow = project.flows.find(f => f.id === flowId);
    if (!currentFlow?.isBuiltIn) {
      logger.warn('Cannot revert: not a built-in macro', { flowId });
      return false;
    }

    // Load fresh from disk to get the original version
    const folderMacros = await loadMacrosFromFolder();
    const originalMacro = folderMacros.find(m => m.id === flowId);
    if (!originalMacro) {
      logger.warn('Cannot revert: original macro not found on disk', { flowId });
      return false;
    }

    // Remove the override from storage (if any)
    if (macroOverrideIds.has(flowId)) {
      const reverted = await revertMacroOverride(flowId);
      if (!reverted) {
        logger.warn('Failed to remove override from storage', { flowId });
        // Continue anyway - we can still update the in-memory state
      }

      // Update the override tracking
      setMacroOverrideIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(flowId);
        return newSet;
      });
    }

    // Update the original state to the disk version
    setOriginalMacroStates(prev => {
      const newMap = new Map(prev);
      newMap.set(flowId, JSON.parse(JSON.stringify(originalMacro)));
      return newMap;
    });

    // Replace the flow with the original from disk
    const now = new Date().toISOString();
    setProject((prev) => ({
      ...prev,
      updatedAt: now,
      flows: prev.flows.map(f =>
        f.id === flowId ? { ...originalMacro, updatedAt: now } : f
      ),
    }));

    logger.debug(`Reverted macro "${originalMacro.name}" to original disk version`);
    return true;
  }, [project.flows, macroOverrideIds]);

  /**
   * Delete a user macro (removes from file system)
   * Note: Built-in macros cannot be deleted, only reverted
   */
  const deleteUserMacroById = useCallback(async (flowId: string): Promise<boolean> => {
    const flow = project.flows.find(f => f.id === flowId);
    if (!flow) {
      return false;
    }

    if (!isUserCreatedMacro(flow)) {
      logger.warn(`Cannot delete: not a user macro ${flowId}`);
      return false;
    }

    // Remove from file system
    await deleteUserMacroFromStorage(flowId);

    // Remove from project
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      flows: prev.flows.filter(f => f.id !== flowId),
    }));

    logger.debug(`Deleted user macro "${flow.name}"`);
    return true;
  }, [project.flows]);

  const updateFlowGraph = useCallback((flowId: string, graph: WorkflowGraph) => {
    updateFlow(flowId, { graph });
  }, [updateFlow]);

  /**
   * Reload all macros from disk (folder macros, user macros, overrides)
   * Useful for when macros are changed externally and need to be refreshed
   */
  const reloadMacros = useCallback(async (): Promise<void> => {
    logger.debug('Reloading macros...');

    const [userMacros, folderMacros, overrides] = await Promise.all([
      loadUserMacros(),
      loadMacrosFromFolder(),
      getAllMacroOverrides(),
    ]);

    const overrideIds = new Set(Object.keys(overrides));
    setMacroOverrideIds(overrideIds);

    // Update original macro states
    const originalStates = new Map<string, Flow>();
    for (const macro of folderMacros) {
      const original = overrides[macro.id] || macro;
      originalStates.set(macro.id, JSON.parse(JSON.stringify(original)));
    }
    setOriginalMacroStates(originalStates);

    logger.debug(`Reloaded ${userMacros.length} user macros, ${folderMacros.length} folder macros`);

    setProject((prev) => {
      // Keep non-macro flows
      const nonMacroFlows = prev.flows.filter(f => !f.isMacro);

      // Apply overrides to built-in macros
      const macrosWithOverrides = folderMacros.map(macro => {
        if (overrides[macro.id]) {
          return { ...overrides[macro.id], isBuiltIn: true, isMacro: true };
        }
        return macro;
      });

      // Combine: non-macro flows + folder macros (with overrides) + user macros
      return {
        ...prev,
        updatedAt: new Date().toISOString(),
        flows: [...nonMacroFlows, ...macrosWithOverrides, ...userMacros],
      };
    });

    logger.debug('Macros reloaded successfully');
  }, []);

  // ============================================
  // LLM Endpoint Management
  // ============================================

  const createLLMEndpoint = useCallback((endpoint: Omit<LLMEndpoint, 'id'>): LLMEndpoint => {
    const newEndpoint: LLMEndpoint = {
      ...endpoint,
      id: uuidv4(),
    };

    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      llmEndpoints: [...prev.llmEndpoints, newEndpoint],
    }));

    return newEndpoint;
  }, []);

  const updateLLMEndpoint = useCallback((endpointId: string, updates: Partial<Omit<LLMEndpoint, 'id'>>) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      llmEndpoints: prev.llmEndpoints.map((e) =>
        e.id === endpointId ? { ...e, ...updates } : e
      ),
    }));
  }, []);

  const deleteLLMEndpoint = useCallback((endpointId: string) => {
    // Don't delete default endpoints
    if (defaultLLMEndpoints.find((e) => e.id === endpointId)) {
      return;
    }
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      llmEndpoints: prev.llmEndpoints.filter((e) => e.id !== endpointId),
    }));
  }, []);

  // ============================================
  // Image Gen Endpoint Management
  // ============================================

  const createImageGenEndpoint = useCallback((endpoint: Omit<ImageGenEndpoint, 'id'>): ImageGenEndpoint => {
    const newEndpoint: ImageGenEndpoint = {
      ...endpoint,
      id: uuidv4(),
    };

    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      imageGenEndpoints: [...prev.imageGenEndpoints, newEndpoint],
    }));

    return newEndpoint;
  }, []);

  const updateImageGenEndpoint = useCallback((endpointId: string, updates: Partial<Omit<ImageGenEndpoint, 'id'>>) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      imageGenEndpoints: prev.imageGenEndpoints.map((e) =>
        e.id === endpointId ? { ...e, ...updates } : e
      ),
    }));
  }, []);

  const deleteImageGenEndpoint = useCallback((endpointId: string) => {
    // Don't delete default endpoints
    if (defaultImageGenEndpoints.find((e) => e.id === endpointId)) {
      return;
    }
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      imageGenEndpoints: prev.imageGenEndpoints.filter((e) => e.id !== endpointId),
    }));
  }, []);

  // ============================================
  // HTTP Preset Management
  // ============================================

  const createHttpPreset = useCallback((preset: Omit<HttpPreset, 'id'>): HttpPreset => {
    const newPreset: HttpPreset = {
      ...preset,
      id: uuidv4(),
    };

    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      httpPresets: [...prev.httpPresets, newPreset],
    }));

    return newPreset;
  }, []);

  const updateHttpPreset = useCallback((presetId: string, updates: Partial<Omit<HttpPreset, 'id'>>) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      httpPresets: prev.httpPresets.map((p) =>
        p.id === presetId ? { ...p, ...updates } : p
      ),
    }));
  }, []);

  const deleteHttpPreset = useCallback((presetId: string) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      httpPresets: prev.httpPresets.filter((p) => p.id !== presetId),
    }));
  }, []);

  // ============================================
  // Constants Management
  // ============================================

  const createConstant = useCallback((constant: Omit<ProjectConstant, 'id'>): ProjectConstant => {
    const newConstant: ProjectConstant = {
      ...constant,
      id: uuidv4(),
    };

    // If creating a secret constant with a value, save to secure storage
    if (newConstant.isSecret && newConstant.value) {
      invoke('store_secret', { key: newConstant.key, value: newConstant.value }).catch((err) => {
        logger.error('Failed to store secret', { key: newConstant.key, error: err });
      });
    }

    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      constants: [...(prev.constants || []), newConstant],
    }));

    return newConstant;
  }, []);

  const updateConstant = useCallback((constantId: string, updates: Partial<Omit<ProjectConstant, 'id'>>) => {
    const constant = (project.constants || []).find(c => c.id === constantId);
    logger.debug(`updateConstant called: ${constantId}`, { isSecret: constant?.isSecret, hasValue: updates.value !== undefined });

    // If updating value of a secret constant, save to secure storage
    if (constant?.isSecret && updates.value !== undefined) {
      logger.debug(`Saving secret to keyring: ${constant.key}`);
      invoke('store_secret', { key: constant.key, value: updates.value })
        .then(() => logger.debug(`Secret saved successfully: ${constant.key}`))
        .catch((err) => {
          logger.error(`Failed to store secret: ${constant.key}`, { error: err });
        });
    }

    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      constants: (prev.constants || []).map((c) =>
        c.id === constantId ? { ...c, ...updates } : c
      ),
    }));
  }, [project.constants]);

  const deleteConstant = useCallback((constantId: string) => {
    // Don't delete default constants, just clear their value
    if (defaultConstants.find((c) => c.id === constantId)) {
      updateConstant(constantId, { value: '' });
      return;
    }
    // If deleting a secret constant, overwrite its keyring entry with an empty
    // value to prevent an orphaned/resurrected secret (no delete_secret command exists)
    const constant = (project.constants || []).find((c) => c.id === constantId);
    if (constant?.isSecret) {
      invoke('store_secret', { key: constant.key, value: '' }).catch(() => {});
    }
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      constants: (prev.constants || []).filter((c) => c.id !== constantId),
    }));
  }, [updateConstant, project.constants]);

  const getConstantByKey = useCallback((key: string): ProjectConstant | undefined => {
    return (project.constants || []).find((c) => c.key === key);
  }, [project.constants]);

  const getConstantValue = useCallback((key: string): string => {
    const constant = getConstantByKey(key);
    return constant?.value || '';
  }, [getConstantByKey]);

  const getConstantsByCategory = useCallback((category: ProjectConstant['category']): ProjectConstant[] => {
    return (project.constants || []).filter((c) => c.category === category);
  }, [project.constants]);

  // ============================================
  // Project Settings Management
  // ============================================

  const updateSettings = useCallback((updates: Partial<ProjectSettings>) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: new Date().toISOString(),
      settings: {
        ...defaultSettings,
        ...prev.settings,
        ...updates,
      },
    }));
  }, []);

  const getSettings = useCallback((): ProjectSettings => {
    return {
      ...defaultSettings,
      ...project.settings,
    };
  }, [project.settings]);

  // ============================================
  // Run History Management
  // ============================================

  const addRunRecord = useCallback((record: Omit<RunRecord, 'id'>): RunRecord => {
    const newRecord: RunRecord = {
      ...record,
      id: uuidv4(),
    };

    setRunHistory((prev) => [newRecord, ...prev].slice(0, MAX_RUN_HISTORY));
    return newRecord;
  }, []);

  const updateRunRecord = useCallback((runId: string, updates: Partial<Omit<RunRecord, 'id'>>) => {
    setRunHistory((prev) =>
      prev.map((r) => (r.id === runId ? { ...r, ...updates } : r))
    );
  }, []);

  const clearRunHistory = useCallback(() => {
    setRunHistory([]);
    try {
      localStorage.removeItem(RUN_HISTORY_KEY);
    } catch {
      // Silent failure
    }
  }, []);

  const getRunsForFlow = useCallback((flowId: string): RunRecord[] => {
    return runHistory.filter((r) => r.flowId === flowId);
  }, [runHistory]);

  // ============================================
  // Project Import/Export
  // ============================================

  const exportProject = useCallback(() => {
    exportProjectToFile(project);
  }, [project]);

  // Returns `{ servicesAdded, servicesSkipped }` so the caller can surface
  // a toast about embedded service definitions that were re-registered.
  const importProject = useCallback(async (
    file: File,
  ): Promise<{ servicesAdded: number; servicesSkipped: number }> => {
    const { project: merged, services } = await importProjectFromFile(file);

    // Rehydrate secret values from the secure keyring (imports redact secrets).
    // No-op on the web shim; only patches values that actually exist in the keyring.
    let importedProject = merged;
    const secretKeys = (merged.constants || [])
      .filter((c) => c.isSecret)
      .map((c) => c.key);
    if (secretKeys.length > 0) {
      try {
        const secrets = await invoke<Record<string, string>>('get_secrets', { keys: secretKeys });
        if (Object.keys(secrets).length > 0) {
          importedProject = {
            ...merged,
            constants: (merged.constants || []).map((c) =>
              c.isSecret && secrets[c.key] ? { ...c, value: secrets[c.key] } : c
            ),
          };
        }
      } catch (error) {
        logger.warn('Failed to rehydrate imported secrets', { error });
      }
    }

    setProject(importedProject);
    if (importedProject.flows.length > 0) {
      setActiveFlowId(importedProject.flows[0].id);
    }
    return { servicesAdded: services.added, servicesSkipped: services.skipped };
  }, []);

  const newProject = useCallback((name?: string) => {
    const fresh = createEmptyProject();
    if (name) {
      fresh.name = name;
    }
    setProject(fresh);
    setActiveFlowId(null);
    setRunHistory([]);
    try {
      localStorage.removeItem(PROJECT_STORAGE_KEY);
      localStorage.removeItem(RUN_HISTORY_KEY);
    } catch {
      // Silent failure
    }
  }, []);

  const resetToDefaults = useCallback(() => {
    // Clear localStorage and reset to demo project with default example flows
    try {
      localStorage.removeItem(PROJECT_STORAGE_KEY);
      localStorage.removeItem(RUN_HISTORY_KEY);
    } catch {
      // Silent failure
    }
    const demoProject = createDemoProject();
    setProject(demoProject);
    setActiveFlowId(demoProject.flows[0]?.id || null);
    setRunHistory([]);
    // Increment reset counter to force remount of OAIYBuilder
    setResetCounter(c => c + 1);
  }, []);

  // Increment reset counter to force OAIYBuilder remount
  const incrementResetCounter = useCallback(() => {
    setResetCounter(c => c + 1);
  }, []);

  const renameProject = useCallback((name: string) => {
    setProject((prev) => ({
      ...prev,
      name,
      updatedAt: new Date().toISOString(),
    }));
  }, []);

  // ============================================
  // Flow lookup for subflows
  // ============================================

  const getFlowById = useCallback((flowId: string): Flow | undefined => {
    return project.flows.find((f) => f.id === flowId);
  }, [project.flows]);

  const getAllFlows = useCallback((): Flow[] => {
    return project.flows;
  }, [project.flows]);

  return {
    // Project state
    project,
    activeFlow,
    activeFlowId,
    setActiveFlowId,
    runHistory,
    resetCounter,

    // Flow operations
    createFlow,
    updateFlow,
    deleteFlow,
    duplicateFlow,
    renameFlow,
    setFlowTags,
    setFlowLocalOnly,
    updateFlowGraph,
    getFlowById,
    getAllFlows,

    // Macro operations
    saveAsMacro,
    unmakeMacro,
    getMacros,
    hasMacroBeenModified,
    isMacroDirty,
    saveMacroChanges,
    discardMacroChanges,
    revertMacroToOriginal,
    deleteUserMacroById,
    reloadMacros,

    // LLM endpoint operations
    createLLMEndpoint,
    updateLLMEndpoint,
    deleteLLMEndpoint,

    // Image generation endpoint operations
    createImageGenEndpoint,
    updateImageGenEndpoint,
    deleteImageGenEndpoint,

    // HTTP preset operations
    createHttpPreset,
    updateHttpPreset,
    deleteHttpPreset,

    // Run history operations
    addRunRecord,
    updateRunRecord,
    clearRunHistory,
    getRunsForFlow,

    // Project operations
    exportProject,
    importProject,
    newProject,
    renameProject,
    resetToDefaults,
    incrementResetCounter,

    // Constants operations
    createConstant,
    updateConstant,
    deleteConstant,
    getConstantByKey,
    getConstantValue,
    getConstantsByCategory,

    // Settings operations
    updateSettings,
    getSettings,
  };
}
