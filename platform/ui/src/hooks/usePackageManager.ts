/**
 * usePackageManager Hook
 *
 * Manages loading, unloading, and state of .oaiy packages.
 * Extracted from OAIYApp.tsx for maintainability.
 */

import { useState, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import type {
  Flow,
  WorkflowGraph,
  OAIYPackageManifest,
  OAIYPackageManifestWithEmbedded,
  NodeDefinition,
  GraphNode,
  ProjectSettings,
  PackageNodeModule,
} from 'oaiy-core';
import { createLogger } from '../utils/logger';
import { registerImportedServices } from '../utils/ProjectIO';

const logger = createLogger('PackageManager');

// A loaded package with all its flows and macros
export interface LoadedPackage {
  manifest: OAIYPackageManifest;
  sourcePath: string;
  flows: Flow[]; // All flows in the package
  macros: Flow[]; // All macros in the package
}

// Reference to an active package flow
export interface ActivePackageFlow {
  packageId: string;
  flowId: string;
}

// Pending package awaiting trust confirmation
interface PendingPackage {
  path: string;
  manifest: OAIYPackageManifest;
}

// Pending dependencies awaiting resolution
interface PendingDependencies {
  packageName: string;
  packagePath: string;
  manifest: OAIYPackageManifest;
  dependencies: Array<{ id: string; version?: string }>;
}

// Changed package notification
interface ChangedPackage {
  packageId: string;
  manifest: OAIYPackageManifest;
}

interface UsePackageManagerOptions {
  projectFlows: Flow[];
  projectSettings?: ProjectSettings;
  onShowToast: (message: string, type: 'success' | 'error' | 'info' | 'warning') => void;
  setActiveFlowId: (id: string | null) => void;
  createFlow: (name: string, graph?: WorkflowGraph) => Flow;
  updateFlow: (id: string, updates: Partial<Omit<Flow, 'id' | 'createdAt'>>) => void;
  getSettings: () => ProjectSettings;
  loadPackageNodes: (packageId: string, packagePath: string, nodes: PackageNodeModule[]) => Promise<Array<{ originalId: string; prefixedId: string; definition: NodeDefinition }>>;
  unloadPackageNodes: (packageId: string) => void;
  loadEmbeddedContent: (packageId: string, manifest: OAIYPackageManifestWithEmbedded) => Promise<unknown>;
  unloadEmbeddedContent: (packageId: string) => void;
}

/**
 * Prefix custom node types in a flow with the package ID.
 * This allows flows to use unprefixed node types (e.g., "text_formatter")
 * which get resolved to their prefixed versions (e.g., "pkg:com.example:text_formatter")
 * when loaded from a package.
 *
 * Also attaches the node definition to each node's data.__definition so GenericNode can render it.
 */
export function prefixPackageNodeTypes(
  flow: Flow,
  packageId: string,
  customNodeIds: string[],
  nodeDefinitions?: Map<string, NodeDefinition>
): Flow {
  if (!flow.graph || !flow.graph.nodes || customNodeIds.length === 0) {
    return flow;
  }

  // Create a set of custom node IDs for fast lookup
  const customNodeSet = new Set(customNodeIds);

  // Map nodes and prefix custom node types
  const prefixedNodes = flow.graph.nodes.map(node => {
    // Check if this node type is a custom node (unprefixed)
    if (customNodeSet.has(node.type)) {
      const prefixedType = `pkg:${packageId}:${node.type}`;
      const definition = nodeDefinitions?.get(prefixedType);
      return {
        ...node,
        type: prefixedType as GraphNode['type'], // Cast to allow prefixed types
        data: {
          ...node.data,
          // Attach the node definition so GenericNode can render it
          __definition: definition,
        },
      };
    }
    return node;
  });

  return {
    ...flow,
    graph: {
      ...flow.graph,
      nodes: prefixedNodes as GraphNode[], // Cast since we're allowing prefixed types
    },
  };
}

/**
 * Apply default provider settings to nodes in a workflow graph.
 */
export function applyDefaultProviders(graph: WorkflowGraph, settings: ProjectSettings): WorkflowGraph {
  const updatedNodes = graph.nodes.map(node => {
    // AI LLM nodes - fill blanks from project defaults but RESPECT the
    // imported node's own configuration (it previously clobbered the node's
    // endpoint/model/provider with the project defaults, wiping a flow's
    // intended target on import — unlike the image_gen branch below).
    if (node.type === 'ai_llm') {
      const d = node.data as Record<string, unknown>;
      const keep = (v: unknown, fallback: string) =>
        (typeof v === 'string' && v.trim()) ? v : fallback;
      return {
        ...node,
        data: {
          ...node.data,
          endpoint: keep(d.endpoint, settings.defaultAIEndpoint || ''),
          model: keep(d.model, settings.defaultAIModel || ''),
          apiKeyConstant: keep(d.apiKeyConstant, settings.defaultAIApiKeyConstant || ''),
          apiKey: keep(d.apiKey, ''),
          provider: keep(d.provider, settings.defaultAIProvider || 'huggingface'),
        },
      };
    }
    // Image Gen nodes - apply default image settings (respect existing backend choice)
    if (node.type === 'image_gen') {
      const existingFormat = node.data.apiFormat as string | undefined;
      const isWan2gp = existingFormat === 'wan2gp';
      const isComfyui = existingFormat === 'comfyui';
      const isLocal = isWan2gp || isComfyui;
      return {
        ...node,
        data: {
          ...node.data,
          endpoint: isWan2gp ? 'http://127.0.0.1:8773'
            : isComfyui ? 'http://localhost:8188'
            : (settings.defaultImageEndpoint || ''),
          model: isLocal ? (node.data.model || '') : (settings.defaultImageModel || ''),
          apiKeyConstant: isLocal ? '' : (settings.defaultImageApiKeyConstant || ''),
          apiFormat: existingFormat || settings.defaultImageProvider || 'wan2gp',
        },
      };
    }
    // Video Gen nodes - apply default video endpoint based on backend
    if (node.type === 'video_gen') {
      const isWan2gp = node.data.apiFormat === 'wan2gp';
      return {
        ...node,
        data: {
          ...node.data,
          endpoint: isWan2gp
            ? 'http://127.0.0.1:8773'
            : (settings.defaultVideoEndpoint || 'http://localhost:8188'),
        },
      };
    }
    // ComfyUI Free Memory - apply default ComfyUI URL
    if (node.type === 'comfyui_free_memory') {
      return {
        ...node,
        data: {
          ...node.data,
          comfyuiUrl: settings.defaultVideoEndpoint || 'http://127.0.0.1:8188',
        },
      };
    }
    return node;
  });
  return { ...graph, nodes: updatedNodes };
}

/**
 * Normalize flow data - handle both flat format (nodes/edges at root)
 * and nested format (graph: { nodes, edges })
 */
export function normalizeFlowData(data: Record<string, unknown>): Partial<Flow> | null {
  // Check for flat format: { nodes: [...], edges: [...], name, ... }
  if (Array.isArray(data.nodes) && Array.isArray(data.edges)) {
    return {
      name: data.name as string | undefined,
      description: data.description as string | undefined,
      tags: data.tags as string[] | undefined,
      localOnly: data.localOnly as boolean | undefined,
      graph: {
        nodes: data.nodes as WorkflowGraph['nodes'],
        edges: data.edges as WorkflowGraph['edges'],
      },
    };
  }
  // Check for nested format: { graph: { nodes, edges }, name, ... }
  if (data.graph && typeof data.graph === 'object') {
    const graph = data.graph as Record<string, unknown>;
    if (Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
      return data as Partial<Flow>;
    }
  }
  return null;
}

export function usePackageManager(options: UsePackageManagerOptions) {
  const {
    projectFlows,
    onShowToast,
    setActiveFlowId,
    createFlow,
    updateFlow,
    getSettings,
    loadPackageNodes,
    unloadPackageNodes,
    loadEmbeddedContent,
    unloadEmbeddedContent,
  } = options;

  // Multi-package state - can load multiple packages, each self-contained
  const [loadedPackages, setLoadedPackages] = useState<Map<string, LoadedPackage>>(new Map());
  const [activePackageFlow, setActivePackageFlow] = useState<ActivePackageFlow | null>(null);
  const [pendingPackage, setPendingPackage] = useState<PendingPackage | null>(null);
  const [pendingDependencies, setPendingDependencies] = useState<PendingDependencies | null>(null);
  const [changedPackage, setChangedPackage] = useState<ChangedPackage | null>(null);
  const [showPackageBrowser, setShowPackageBrowser] = useState(false);

  // Granted-permission map for each loaded package. The trust dialog
  // captures the user's tick-list when the package first loads; the job
  // submission path then plumbs this into JobManager.submit() so the
  // runtime's fail-closed permission gate kicks in. A package without an
  // entry here is treated as having no permissions (= safest default).
  const [packagePermissions, setPackagePermissions] = useState<
    Map<string, { granted: string[]; trustLevel: 'untrusted' | 'trusted' | 'verified' | 'blocked' }>
  >(new Map());

  // Ref to track if package changed notification is already showing
  const changedPackageRef = useRef<ChangedPackage | null>(null);

  // Helper to load a package's flows and add it to loaded packages
  const loadPackageFlows = useCallback(async (
    path: string,
    manifest: OAIYPackageManifest,
    navigate: boolean = true
  ): Promise<LoadedPackage> => {
    // Load package nodes FIRST if defined (to get custom node IDs for prefixing)
    let customNodeIds: string[] = [];
    let nodeDefinitions: Map<string, NodeDefinition> = new Map();
    if (manifest.nodes && manifest.nodes.length > 0) {
      try {
        const loadedNodes = await loadPackageNodes(manifest.id, path, manifest.nodes);
        // Extract original (unprefixed) node IDs for flow prefixing
        customNodeIds = loadedNodes.map(n => n.originalId);
        // Build a map of prefixedId -> definition for attaching to flow nodes
        for (const node of loadedNodes) {
          nodeDefinitions.set(node.prefixedId, node.definition);
        }
        logger.debug(`Loaded ${loadedNodes.length} custom nodes for package ${manifest.id}: ${customNodeIds.join(', ')}`);
      } catch (err) {
        logger.warn('Failed to load package nodes', { error: err });
      }
    }

    // Load embedded custom nodes and extensions (TypeScript-based)
    const embeddedManifest = manifest as OAIYPackageManifestWithEmbedded;
    if (embeddedManifest.embeddedCustomNodes?.length || embeddedManifest.embeddedNodeExtensions?.length) {
      try {
        const embeddedResults = await loadEmbeddedContent(manifest.id, embeddedManifest);
        logger.debug(`Loaded embedded content for package ${manifest.id}`, { results: embeddedResults });

        // Add embedded custom node IDs for flow prefixing
        if (embeddedManifest.embeddedCustomNodes) {
          for (const node of embeddedManifest.embeddedCustomNodes) {
            customNodeIds.push(node.id);
          }
        }
      } catch (err) {
        logger.warn('Failed to load embedded content', { error: err });
      }
    }

    // Load all flows from the package
    let flows: Flow[] = [];
    for (const flowPath of manifest.flows) {
      try {
        const flowContent = await invoke<string>('read_package_flow_content', {
          packagePath: path,
          flowPath: flowPath,
        });
        let flowData = JSON.parse(flowContent) as Flow;
        // Prefix custom node types in the flow and attach definitions
        if (customNodeIds.length > 0) {
          flowData = prefixPackageNodeTypes(flowData, manifest.id, customNodeIds, nodeDefinitions);
        }
        flows.push(flowData);
      } catch (err) {
        logger.warn(`Failed to load flow ${flowPath}`, { error: err });
      }
    }

    if (flows.length === 0) {
      throw new Error('No flows found in package');
    }

    // Load macros from package if present
    let macros: Flow[] = [];
    if (manifest.macros && manifest.macros.length > 0) {
      for (const macroPath of manifest.macros) {
        try {
          const macroContent = await invoke<string>('read_package_flow_content', {
            packagePath: path,
            flowPath: macroPath,
          });
          let macroData = JSON.parse(macroContent) as Flow;
          // Prefix custom node types in macros too and attach definitions
          if (customNodeIds.length > 0) {
            macroData = prefixPackageNodeTypes(macroData, manifest.id, customNodeIds, nodeDefinitions);
          }
          macros.push(macroData);
        } catch (err) {
          logger.warn(`Failed to load macro ${macroPath}`, { error: err });
        }
      }
      logger.debug(`Loaded ${macros.length} macros from package`);
    }

    // Create the loaded package
    const loadedPackage: LoadedPackage = {
      manifest,
      sourcePath: path,
      flows,
      macros,
    };

    // Add to loaded packages
    setLoadedPackages(prev => {
      const next = new Map(prev);
      next.set(manifest.id, loadedPackage);
      return next;
    });

    // Navigate to the entry flow if requested
    if (navigate) {
      const entryFlow = flows.find(f =>
        manifest.entryFlow.includes(f.id) ||
        manifest.entryFlow.endsWith(`${f.id}.json`)
      ) || flows[0];

      setActivePackageFlow({ packageId: manifest.id, flowId: entryFlow.id });
      setActiveFlowId(null);
    }

    return loadedPackage;
  }, [setActiveFlowId, loadPackageNodes, loadEmbeddedContent]);

  // Handle loading a .oaiy package from a path.
  //
  // Round 5 reviewer #8: the load-from-path flow used to call
  // `read_package` and jump straight to the trust dialog — meaning a
  // tampered archive (mismatched contentHash, bad signature) or an
  // oversized one would still reach the user before any of the
  // install-time validation kicked in. We now route this path
  // through `verify_package_for_load`, which mirrors the same
  // integrity + size + signature checks `install_package` runs, so
  // the load-from-path entry no longer bypasses them.
  const handleLoadPackageFromPath = useCallback(async (packagePath: string) => {
    try {
      const verification = await invoke<{
        manifest: OAIYPackageManifest;
        size: { withinDefaults: boolean; exceedsHardMax: boolean };
        verification: 'unverified' | 'hash_matched' | 'signature_verified';
        contentHash: string;
      } | null>('verify_package_for_load', { packagePath });

      // Packages (.oaiy) need the native unzip + verify path, which only the
      // desktop app provides — in the browser the shimmed command resolves to
      // null. Surface clear guidance instead of crashing on the null deref.
      if (!verification || !verification.manifest) {
        onShowToast(
          'OAIY packages (.oaiy) can only be loaded in the desktop app. In the browser, import a flow as JSON instead (use “Load from File” and pick a .json, or Import a previously exported flow).',
          'info',
        );
        return;
      }

      // The trust dialog still shows — the user gets to decide what
      // permissions to grant — but they're now choosing on top of a
      // package whose bytes have at least been validated. The
      // verification outcome is logged so prior reviews can audit.
      logger.debug('Package verified for load', {
        packageId: verification.manifest.id,
        verification: verification.verification,
        contentHash: verification.contentHash,
      });

      setPendingPackage({ path: packagePath, manifest: verification.manifest });
    } catch (err) {
      logger.error('Failed to load package', { error: err });
      onShowToast(`Failed to load package: ${err}`, 'error');
    }
  }, [onShowToast]);

  // Handle loading a .oaiy package or standalone .json flow via file picker
  const handleLoadPackage = useCallback(async () => {
    try {
      const result = await open({
        title: 'Load Package or Flow',
        filters: [
          { name: 'All Supported', extensions: ['oaiy', 'json'] },
          { name: 'OAIY Package', extensions: ['oaiy'] },
          { name: 'Flow JSON', extensions: ['json'] },
        ],
        multiple: false,
      });

      if (!result) return;

      const path = typeof result === 'string' ? result : result;

      // Check if it's a standalone JSON flow file
      if (path.toLowerCase().endsWith('.json')) {
        // Read the JSON file
        const fileResult = await invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', {
          path,
          readAs: 'text',
        });

        const jsonData = JSON.parse(fileResult.content);
        const fileName = path.split(/[/\\]/).pop()?.replace('.json', '') || 'Imported Flow';

        // Re-register any embedded services BEFORE the flow gets created
        // so the new service_call nodes can resolve their `data.service`
        // ids against a registry that's already populated. Existing ids
        // are preserved as-is so the user's prior edits aren't clobbered.
        try {
          const svcResult = registerImportedServices(jsonData);
          if (svcResult.added > 0) {
            onShowToast(
              `Re-added ${svcResult.added} service${svcResult.added !== 1 ? 's' : ''} from the imported flow.`,
              'success',
            );
          }
        } catch (e) {
          // Don't block the import on a malformed services bag; just log.
          logger.warn('Failed to register embedded services', { error: e });
        }

        // Determine the format and extract flow(s)
        let flowsToImport: Partial<Flow>[] = [];

        if (jsonData.flows && Array.isArray(jsonData.flows)) {
          // Project format: { flows: [...], ... }
          for (const flowItem of jsonData.flows) {
            const normalized = normalizeFlowData(flowItem);
            if (normalized) {
              flowsToImport.push(normalized);
            }
          }
        } else {
          // Single flow - try to normalize it
          const normalized = normalizeFlowData(jsonData);
          if (normalized) {
            flowsToImport.push(normalized);
          } else if (jsonData.flow) {
            // Wrapped flow format: { flow: { ... } }
            const wrappedNormalized = normalizeFlowData(jsonData.flow);
            if (wrappedNormalized) {
              flowsToImport.push(wrappedNormalized);
            }
          }
        }

        // Validate we found valid flows
        if (flowsToImport.length === 0) {
          onShowToast('Invalid file: no valid flow data found. Expected nodes and edges arrays.', 'error');
          return;
        }

        // Get current project settings to apply defaults
        const settings = getSettings();

        // Import each valid flow
        let lastFlow: Flow | null = null;
        for (const flowData of flowsToImport) {
          const flowName = flowData.name || fileName;

          // Apply default providers to the graph
          const graphWithDefaults = applyDefaultProviders(flowData.graph!, settings);

          // Create a new flow with the imported data
          const newFlow = createFlow(flowName, graphWithDefaults);
          lastFlow = newFlow;

          // Update with additional properties if present
          if (flowData.description || flowData.tags || flowData.localOnly) {
            updateFlow(newFlow.id, {
              description: flowData.description,
              tags: flowData.tags,
              localOnly: flowData.localOnly,
            });
          }
        }

        // Clear any active package flow to ensure we're viewing user flows
        setActivePackageFlow(null);

        if (flowsToImport.length === 1) {
          onShowToast(`Flow "${lastFlow?.name || fileName}" imported successfully`, 'success');
        } else {
          onShowToast(`${flowsToImport.length} flows imported successfully`, 'success');
        }
        return;
      }

      // Otherwise, load as a .oaiy package
      await handleLoadPackageFromPath(path);
    } catch (err) {
      logger.error('Failed to load file', { error: err });
      onShowToast(`Failed to load file: ${err}`, 'error');
    }
  }, [handleLoadPackageFromPath, createFlow, updateFlow, onShowToast, getSettings]);

  // Handle package trust confirmation - load all flows from the package.
  //
  // The trust dialog returns the EXACT subset of permissions the user
  // ticked. We persist that subset against the package id so the job
  // submission path can read it back and pass it to
  // `runtime.setPackagePermissions(...)`. Without this, package flows
  // would silently execute as trusted local flows — round 5 #1.
  const handlePackageTrustConfirm = useCallback(async (grantedPermissions: string[]) => {
    if (!pendingPackage) return;

    const trustLevel: 'untrusted' | 'trusted' | 'verified' =
      grantedPermissions.length > 0 ? 'trusted' : 'untrusted';
    setPackagePermissions(prev => {
      const next = new Map(prev);
      next.set(pendingPackage.manifest.id, { granted: [...grantedPermissions], trustLevel });
      return next;
    });

    try {
      // Check if package is already loaded
      if (loadedPackages.has(pendingPackage.manifest.id)) {
        // Just navigate to it
        const pkg = loadedPackages.get(pendingPackage.manifest.id)!;
        const entryFlow = pkg.flows.find(f =>
          pendingPackage.manifest.entryFlow.includes(f.id) ||
          pendingPackage.manifest.entryFlow.endsWith(`${f.id}.json`)
        ) || pkg.flows[0];
        if (entryFlow) {
          setActivePackageFlow({ packageId: pkg.manifest.id, flowId: entryFlow.id });
          setActiveFlowId(null); // Clear user flow selection
        }
        setPendingPackage(null);
        onShowToast(`Switched to package: ${pendingPackage.manifest.name}`, 'info');
        return;
      }

      // Check for package dependencies
      const requiredPackages = pendingPackage.manifest.dependencies?.packages || [];
      if (requiredPackages.length > 0) {
        const missingPackages = requiredPackages.filter(dep => !loadedPackages.has(dep.id));
        if (missingPackages.length > 0) {
          // Show dependency dialog instead of just a toast
          setPendingDependencies({
            packageName: pendingPackage.manifest.name,
            packagePath: pendingPackage.path,
            manifest: pendingPackage.manifest,
            dependencies: requiredPackages,
          });
          // Don't proceed with loading yet - wait for dependency resolution
          return;
        }
      }

      // Use the loadPackageFlows helper
      const loadedPackage = await loadPackageFlows(pendingPackage.path, pendingPackage.manifest, true);

      setPendingPackage(null);
      onShowToast(`Loaded package: ${pendingPackage.manifest.name} (${loadedPackage.flows.length} flow${loadedPackage.flows.length !== 1 ? 's' : ''})`, 'success');
    } catch (err) {
      logger.error('Failed to load package', { error: err });
      onShowToast(`Failed to load package: ${err}`, 'error');
      setPendingPackage(null);
    }
  }, [pendingPackage, loadedPackages, onShowToast, setActiveFlowId, loadPackageFlows]);

  // Handle dependency package being loaded (from DependencyDialog)
  const handleDependencyLoaded = useCallback(async (path: string, manifest: OAIYPackageManifest) => {
    // Load the dependency package (don't navigate to it)
    await loadPackageFlows(path, manifest, false);
    onShowToast(`Loaded dependency: ${manifest.name}`, 'success');
  }, [loadPackageFlows, onShowToast]);

  // Handle all dependencies satisfied - continue loading the main package
  const handleDependenciesSatisfied = useCallback(async () => {
    if (!pendingDependencies) return;

    try {
      await loadPackageFlows(
        pendingDependencies.packagePath,
        pendingDependencies.manifest,
        true
      );
      onShowToast(
        `Loaded package: ${pendingDependencies.manifest.name}`,
        'success'
      );
    } catch (err) {
      logger.error('Failed to load package after dependencies', { error: err });
      onShowToast(`Failed to load package: ${err}`, 'error');
    }

    setPendingDependencies(null);
    setPendingPackage(null);
  }, [pendingDependencies, loadPackageFlows, onShowToast]);

  // Handle continuing without all dependencies
  const handleContinueWithoutDeps = useCallback(async () => {
    if (!pendingDependencies) return;

    try {
      await loadPackageFlows(
        pendingDependencies.packagePath,
        pendingDependencies.manifest,
        true
      );
      onShowToast(
        `Loaded package: ${pendingDependencies.manifest.name} (some dependencies missing)`,
        'warning'
      );
    } catch (err) {
      logger.error('Failed to load package', { error: err });
      onShowToast(`Failed to load package: ${err}`, 'error');
    }

    setPendingDependencies(null);
    setPendingPackage(null);
  }, [pendingDependencies, loadPackageFlows, onShowToast]);

  // Handle closing a specific package
  const handleClosePackage = useCallback((packageId: string) => {
    const pkg = loadedPackages.get(packageId);
    if (!pkg) return;

    // Unload package nodes (path-based and embedded)
    unloadPackageNodes(packageId);
    unloadEmbeddedContent(packageId);

    // Remove from loaded packages
    setLoadedPackages(prev => {
      const next = new Map(prev);
      next.delete(packageId);
      return next;
    });

    // If we were viewing this package, switch to user flows
    if (activePackageFlow?.packageId === packageId) {
      setActivePackageFlow(null);
      // Select first user flow if available
      if (projectFlows.length > 0) {
        setActiveFlowId(projectFlows[0].id);
      }
    }

    onShowToast(`Closed package: ${pkg.manifest.name}`, 'info');
  }, [loadedPackages, activePackageFlow, projectFlows, setActiveFlowId, onShowToast, unloadPackageNodes, unloadEmbeddedContent]);

  // Handle reloading a package that has changed
  const handleReloadPackage = useCallback(async (packageId: string) => {
    const pkg = loadedPackages.get(packageId);
    if (!pkg) {
      setChangedPackage(null);
      return;
    }

    try {
      // Unload existing package nodes first (path-based and embedded)
      await unloadPackageNodes(packageId);
      await unloadEmbeddedContent(packageId);

      // Re-read the manifest
      const manifest = await invoke<OAIYPackageManifest>('read_package', {
        packagePath: pkg.sourcePath,
      });

      // Reload package nodes FIRST if defined (to get custom node IDs for prefixing)
      let customNodeIds: string[] = [];
      let nodeDefinitions: Map<string, NodeDefinition> = new Map();
      if (manifest.nodes && manifest.nodes.length > 0) {
        try {
          const loadedNodes = await loadPackageNodes(packageId, pkg.sourcePath, manifest.nodes);
          customNodeIds = loadedNodes.map(n => n.originalId);
          // Build nodeDefinitions map for attaching to flow nodes
          for (const node of loadedNodes) {
            nodeDefinitions.set(node.prefixedId, node.definition);
          }
        } catch (err) {
          logger.warn('Failed to reload package nodes', { error: err });
        }
      }

      // Reload embedded custom nodes and extensions (TypeScript-based)
      const embeddedManifest = manifest as OAIYPackageManifestWithEmbedded;
      if (embeddedManifest.embeddedCustomNodes?.length || embeddedManifest.embeddedNodeExtensions?.length) {
        try {
          await loadEmbeddedContent(packageId, embeddedManifest);
          // Add embedded custom node IDs for flow prefixing
          if (embeddedManifest.embeddedCustomNodes) {
            for (const node of embeddedManifest.embeddedCustomNodes) {
              customNodeIds.push(node.id);
            }
          }
        } catch (err) {
          logger.warn('Failed to reload embedded content', { error: err });
        }
      }

      // Re-load all flows from the package (with node type prefixing)
      let flows: Flow[] = [];
      for (const flowPath of manifest.flows) {
        try {
          const flowContent = await invoke<string>('read_package_flow_content', {
            packagePath: pkg.sourcePath,
            flowPath: flowPath,
          });
          let flowData = JSON.parse(flowContent) as Flow;
          // Prefix custom node types in the flow and attach definitions
          if (customNodeIds.length > 0) {
            flowData = prefixPackageNodeTypes(flowData, packageId, customNodeIds, nodeDefinitions);
          }
          flows.push(flowData);
        } catch (err) {
          logger.warn(`Failed to reload flow ${flowPath}`, { error: err });
        }
      }

      if (flows.length === 0) {
        throw new Error('No flows found in package');
      }

      // Re-load macros from package if present (with node type prefixing)
      let macros: Flow[] = [];
      if (manifest.macros && manifest.macros.length > 0) {
        for (const macroPath of manifest.macros) {
          try {
            const macroContent = await invoke<string>('read_package_flow_content', {
              packagePath: pkg.sourcePath,
              flowPath: macroPath,
            });
            let macroData = JSON.parse(macroContent) as Flow;
            // Prefix custom node types in macros too and attach definitions
            if (customNodeIds.length > 0) {
              macroData = prefixPackageNodeTypes(macroData, packageId, customNodeIds, nodeDefinitions);
            }
            macros.push(macroData);
          } catch (err) {
            logger.warn(`Failed to reload macro ${macroPath}`, { error: err });
          }
        }
      }

      // Update the loaded package
      const updatedPackage: LoadedPackage = {
        manifest,
        sourcePath: pkg.sourcePath,
        flows,
        macros,
      };

      setLoadedPackages(prev => {
        const next = new Map(prev);
        next.set(packageId, updatedPackage);
        return next;
      });

      // If we were viewing this package, stay on it but refresh
      if (activePackageFlow?.packageId === packageId) {
        // Try to stay on the same flow, or go to first flow
        const currentFlowStillExists = flows.find(f => f.id === activePackageFlow.flowId);
        if (!currentFlowStillExists && flows.length > 0) {
          setActivePackageFlow({ packageId, flowId: flows[0].id });
        }
      }

      setChangedPackage(null);
      onShowToast(`Reloaded package: ${manifest.name} v${manifest.version}`, 'success');
    } catch (err) {
      logger.error('Failed to reload package', { error: err });
      onShowToast(`Failed to reload package: ${err}`, 'error');
      setChangedPackage(null);
    }
  }, [loadedPackages, activePackageFlow, onShowToast, unloadPackageNodes, loadPackageNodes, unloadEmbeddedContent, loadEmbeddedContent]);

  // Handle selecting a package flow
  const handleSelectPackageFlow = useCallback((packageId: string, flowId: string) => {
    setActivePackageFlow({ packageId, flowId });
    setActiveFlowId(null); // Clear user flow selection
  }, [setActiveFlowId]);

  // Handle package change notification
  const handlePackageChanged = useCallback((packageId: string, manifest: OAIYPackageManifest) => {
    // Only show one notification at a time
    if (!changedPackageRef.current) {
      changedPackageRef.current = { packageId, manifest };
      setChangedPackage({ packageId, manifest });
    }
  }, []);

  // Clear package change notification
  const clearChangedPackage = useCallback(() => {
    changedPackageRef.current = null;
    setChangedPackage(null);
  }, []);

  // Get currently active package and flow (if viewing a package)
  const activePackage = activePackageFlow ? loadedPackages.get(activePackageFlow.packageId) : null;
  const activePackageFlowData = activePackage?.flows.find(f => f.id === activePackageFlow?.flowId);
  const activePackageMacros = activePackage?.macros || [];

  /**
   * Look up the granted permission set + trust level for a package the
   * user opened in this session. Returns `null` when the package was
   * never confirmed (in which case the runtime should treat it as
   * having an empty granted set, the safest default).
   */
  const getPackagePermissionContext = useCallback(
    (packageId: string | undefined | null): {
      packageId: string;
      grantedPermissions: string[];
      packageTrustLevel: 'untrusted' | 'trusted' | 'verified' | 'blocked';
    } | null => {
      if (!packageId) return null;
      const entry = packagePermissions.get(packageId);
      if (!entry) {
        // Loaded without a trust confirmation in this session (e.g. a
        // dependency auto-loaded under the parent's permissions). Treat
        // as "no granted permissions" so the runtime's fail-closed gate
        // refuses every gated host call.
        return { packageId, grantedPermissions: [], packageTrustLevel: 'untrusted' };
      }
      return {
        packageId,
        grantedPermissions: [...entry.granted],
        packageTrustLevel: entry.trustLevel,
      };
    },
    [packagePermissions],
  );

  return {
    // State
    loadedPackages,
    activePackageFlow,
    activePackage,
    activePackageFlowData,
    activePackageMacros,
    pendingPackage,
    pendingDependencies,
    changedPackage,
    showPackageBrowser,
    getPackagePermissionContext,

    // Setters
    setActivePackageFlow,
    setPendingPackage,
    setPendingDependencies,
    setShowPackageBrowser,
    clearChangedPackage,

    // Handlers
    handleLoadPackage,
    handleLoadPackageFromPath,
    handlePackageTrustConfirm,
    handleDependencyLoaded,
    handleDependenciesSatisfied,
    handleContinueWithoutDeps,
    handleClosePackage,
    handleReloadPackage,
    handleSelectPackageFlow,
    handlePackageChanged,
  };
}
