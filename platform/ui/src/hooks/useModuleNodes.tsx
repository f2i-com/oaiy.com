/**
 * Hook for accessing node definitions from the module system
 *
 * This hook provides access to both bundled and dynamically loaded
 * node definitions for use in the NodePalette and other components.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  getModuleLoader,
  loadBundledModules,
  BUNDLED_MODULES,
  type NodeDefinition,
  type ModuleManifest,
  type ModuleCategory,
} from 'oaiy-core';
import { subscribeToDynamicOptionsInvalidation } from 'oaiy-ui-components';
import { isListed, paletteShowsNode } from '../lib/nodeAvailability';
import { currentAvailabilityEnv } from '../lib/availabilityEnv';
import type { PackageNodeInfo } from './usePackageNodes';
import { listAllServices } from '../utils/serviceRegistry';
import { listDesktopServices } from '../lib/desktopServices';
import { moduleLogger as logger } from '../utils/logger';
import { useCaps } from './useCaps';

// ============================================
// Types
// ============================================

export interface ModuleNodeInfo {
  definition: NodeDefinition;
  module: ModuleManifest;
  category: ModuleCategory;
  /** Whether this is a package node */
  isPackageNode?: boolean;
  /** Package ID if this is a package node */
  packageId?: string;
}

export interface GroupedNodes {
  category: ModuleCategory | 'Package';
  label: string;
  nodes: ModuleNodeInfo[];
}

/** Options for useModuleNodes hook */
export interface UseModuleNodesOptions {
  /** Active package ID (show package nodes only for this package) */
  activePackageId?: string | null;
  /** All loaded package nodes */
  packageNodes?: PackageNodeInfo[];
}

// Category display configuration
const CATEGORY_CONFIG: Record<ModuleCategory | 'Package', { order: number; label: string }> = {
  'Package': { order: -1, label: 'Package Nodes' },  // Show package nodes first
  'Input': { order: 0, label: 'Input' },
  'Output': { order: 1, label: 'Output' },
  // 'AI' is the catch-all for every node that talks to the unified
  // Services registry — see AI_SERVICE_NODE_IDS below for the override
  // that pulls image_gen / video_gen / text_to_speech / service_call
  // into this bucket from their owning modules.
  'AI': { order: 2, label: 'Services' },
  'Text': { order: 3, label: 'Text Processing' },
  'Image': { order: 4, label: 'Image' },
  'Video': { order: 5, label: 'Video' },
  'Audio': { order: 6, label: 'Audio' },
  'File System': { order: 7, label: 'File System' },
  'Flow Control': { order: 8, label: 'Flow Control' },
  'Browser': { order: 9, label: 'Browser & HTTP' },
  'Terminal': { order: 10, label: 'Terminal' },
  'Database': { order: 11, label: 'Database' },
  'Network': { order: 12, label: 'Network' },
  'Utility': { order: 13, label: 'Utility' },
  'Macros': { order: 14, label: 'Macros' },
  'Custom': { order: 99, label: 'Custom' },
};

// Web-only palette merge: nodes that consume the unified Services
// registry are pulled into the single "AI Services" category, regardless
// of which module they ship in. The owning IMAGE / VIDEO / AUDIO
// categories then contain only their post-processing utility nodes
// (image_resize / video_append / save_audio / …), and the dedicated
// INTEGRATION category for service_call disappears entirely. Keeps
// "AI Services" as the discoverable place for "I want to plug an engine
// into this flow". Add a node id here if you add another service-
// consuming node later.
const AI_SERVICE_NODE_IDS: ReadonlySet<string> = new Set([
  'ai_llm',
  'image_gen',
  'video_gen',
  'text_to_speech',
  'music_gen',
  'speech_to_text',
  'sound_effect',
  'model_3d',
  'background_removal',
  'image_upscale',
  'service_call',
]);

// ============================================
// Hook Implementation
// ============================================

export function useModuleNodes(options?: UseModuleNodesOptions) {
  const { activePackageId, packageNodes: pkgNodes } = options || {};
  // What the editor can do here decides which of the desktop's nodes the palette offers.
  const caps = useCaps();

  const [loader] = useState(() => getModuleLoader());
  const [nodes, setNodes] = useState<ModuleNodeInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Bumped whenever the Services registry mutates so the palette
  // re-derives its synthetic service entries. serviceRegistry.ts calls
  // `invalidateDynamicOptions()` on every save/delete, so we get
  // notified via the shared subscriber set.
  const [servicesVersion, setServicesVersion] = useState(0);
  useEffect(() => {
    return subscribeToDynamicOptionsInvalidation(() => {
      setServicesVersion((v) => v + 1);
    });
  }, []);

  // Load bundled modules on mount
  useEffect(() => {
    async function loadModules() {
      setIsLoading(true);
      setError(null);

      try {
        // Load bundled modules
        const result = await loadBundledModules(loader, BUNDLED_MODULES);

        if (result.errors.length > 0) {
          logger.warn('Module loading errors', { errors: result.errors });
        }

        // Get all node definitions
        const allNodes: ModuleNodeInfo[] = [];
        const allDefinitions = loader.getAllNodeDefinitions();

        for (const definition of allDefinitions) {
          const module = loader.getModuleForNode(definition.id);
          if (module) {
            allNodes.push({
              definition,
              module: module.manifest,
              category: module.manifest.category || 'Custom',
            });
          }
        }

        setNodes(allNodes);
      } catch (err) {
        logger.error('Failed to load modules', { error: err });
        setError(err instanceof Error ? err.message : 'Failed to load modules');
      } finally {
        setIsLoading(false);
      }
    }

    loadModules();
  }, [loader]);

  // The node types the palette offers now: a service-driven node (Image Gen,
  // Music Gen, 3D Model, …) only when something installed can run it — in
  // OAIY's window an engine model, an installed desktop service or one of the
  // user's own; see lib/nodeAvailability.ts. Re-derived whenever the service
  // lists change (servicesVersion), so installing a model shows its node.
  const paletteNodes = useMemo(() => {
    const env = currentAvailabilityEnv();
    return nodes.filter((n) => paletteShowsNode(n.definition.id, env));
    // servicesVersion: the lists behind `env` changed. caps: the editor was linked to a desktop, or unlinked (lib/caps.ts).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, servicesVersion, caps]);

  // Group nodes by category (including package nodes when active)
  const groupedNodes = useMemo<GroupedNodes[]>(() => {
    const groups = new Map<ModuleCategory | 'Package', ModuleNodeInfo[]>();

    // Add built-in nodes (filter out package nodes - they're handled separately below)
    for (const node of paletteNodes) {
      // Skip package nodes (prefixed with pkg:) - they should only appear in the Package category
      if (node.definition.id.startsWith('pkg:')) {
        continue;
      }
      // Hide the generic `service_call` node from the palette — the only
      // entries under SERVICES should be the user-added services
      // (synthetic palette entries injected below). The node TYPE stays
      // registered with React Flow so dropped synthetic entries (which
      // translate to `service_call` + `data.service`) render correctly,
      // and any existing saved flow with a `service_call` node keeps
      // working — they just don't appear in the palette for NEW drops.
      if (node.definition.id === 'service_call') {
        continue;
      }
      // Re-route the service-consuming nodes into the merged "AI Services"
      // bucket, regardless of which module they ship in.
      const effectiveCategory: ModuleCategory = AI_SERVICE_NODE_IDS.has(node.definition.id)
        ? ('AI' as ModuleCategory)
        : node.category;
      const existing = groups.get(effectiveCategory) || [];
      existing.push(node);
      groups.set(effectiveCategory, existing);
    }

    // Inject one synthetic palette entry per registered service so users
    // can drag "My Local Ollama" or "ComfyUI Submit Prompt" directly
    // onto the canvas. Dropping uses the `service:<id>` synthetic node
    // type — useWorkflow.ts addNode() translates that to a service_call
    // node with data.service pre-pinned. The actual handles + runtime
    // come from the real service_call definition; this is just a nicer
    // palette face.
    // The user's own services PLUS any the OAIY Desktop is currently
    // exposing — so a service you start in the desktop app shows up here as
    // a draggable palette entry (id `companion:<id>`), the same as a custom
    // one. They refresh live: desktopServices.publish() fires the same
    // invalidate that bumps servicesVersion below.
    const services = [...listAllServices(), ...listDesktopServices()];
    const syntheticModule: ModuleManifest = {
      id: 'core-service',
      name: 'Service',
      version: '1.0.0',
      category: 'AI' as ModuleCategory,
      description: 'Custom + OAIY-Desktop-managed services',
      nodes: [],
    };
    for (const svc of services) {
      // Built-in services no longer auto-populate the palette — the user
      // explicitly adds services via Settings → Services (the Quick-Add
      // buttons provide preconfigured templates for AI LLM / Image Gen /
      // Video Gen / TTS / Music Gen / Webhook / Custom). Default app
      // state: empty palette under SERVICES except for the generic
      // Service Call node, until the user adds their first service.
      if (svc.isBuiltIn) continue;
      if (!svc.endpoint) continue; // skip half-formed user services
      // OAIY's engine models are not entries of their own: each has its
      // typed node (Image Gen, Music Gen, 3D Model …), which starts on the
      // default model and picks another in its Service dropdown.
      if (svc.group === 'engine') continue;
      // Installed but not in use (stopped, not started with OAIY).
      if (!isListed(svc)) continue;
      const synthetic: ModuleNodeInfo = {
        definition: {
          id: `service:${svc.id}`,
          name: svc.name,
          description: svc.description
            ?? (svc.isBuiltIn ? 'Built-in example service' : 'Custom service'),
          // Prefer the service's user-defined emoji/text icon when set;
          // fall back to the generic 'server' lucide name. The palette
          // renderer picks the right path based on whether the value is
          // a single grapheme/emoji or a recognised lucide id.
          icon: svc.icon || 'server',
          color: 'cyan',
          tags: ['service', ...(svc.nodeTypes ?? [])],
          inputs: [{ id: 'input', name: 'Input', type: 'any', position: 'left' }],
          outputs: [{ id: 'response', name: 'Response', type: 'any', position: 'right' }],
          properties: [],
          compiler: { customHandler: true },
        } as unknown as NodeDefinition,
        module: syntheticModule,
        category: 'AI' as ModuleCategory,
      };
      const existing = groups.get('AI' as ModuleCategory) || [];
      existing.push(synthetic);
      groups.set('AI' as ModuleCategory, existing);
    }

    // Add package nodes if a package is active
    // All package nodes go into the 'Package' category at the top of the palette
    // They are temporary and will be removed when the package is closed
    if (activePackageId && pkgNodes) {
      const activePackageNodes = pkgNodes.filter(pn => pn.packageId === activePackageId);
      for (const pn of activePackageNodes) {
        // Force all package nodes into the 'Package' category
        const category = 'Package' as const;
        const nodeInfo: ModuleNodeInfo = {
          definition: pn.definition,
          module: pn.moduleManifest,
          category: category as ModuleCategory,
          isPackageNode: true,
          packageId: pn.packageId,
        };

        const existing = groups.get(category) || [];
        existing.push(nodeInfo);
        groups.set(category, existing);
      }
    }

    // Sort groups by category order
    const sortedGroups = Array.from(groups.entries())
      .sort(([a], [b]) => {
        const orderA = CATEGORY_CONFIG[a]?.order ?? 99;
        const orderB = CATEGORY_CONFIG[b]?.order ?? 99;
        return orderA - orderB;
      })
      .map(([category, nodeList]) => ({
        category,
        label: CATEGORY_CONFIG[category]?.label || category,
        nodes: nodeList.sort((a, b) => a.definition.name.localeCompare(b.definition.name)),
      }));

    return sortedGroups;
    // servicesVersion participates so adding/removing a service in
    // Settings re-runs the synthetic-entry injection above.
  }, [paletteNodes, activePackageId, pkgNodes, servicesVersion]);

  // Get a node definition by ID (checks both built-in and package nodes)
  const getNodeDefinition = useCallback((nodeId: string): NodeDefinition | undefined => {
    // First check built-in nodes
    const builtIn = loader.getNodeDefinition(nodeId);
    if (builtIn) return builtIn;

    // Then check package nodes
    if (pkgNodes) {
      const packageNode = pkgNodes.find(pn => pn.prefixedId === nodeId);
      if (packageNode) return packageNode.definition;
    }

    return undefined;
  }, [loader, pkgNodes]);

  // Check if a node type is valid (checks both built-in and package nodes)
  const isValidNodeType = useCallback((nodeType: string): boolean => {
    // Check built-in nodes
    if (loader.isNodeTypeValid(nodeType)) return true;

    // Check package nodes
    if (pkgNodes) {
      return pkgNodes.some(pn => pn.prefixedId === nodeType);
    }

    return false;
  }, [loader, pkgNodes]);

  // Get all node IDs
  const nodeIds = useMemo(() => nodes.map(n => n.definition.id), [nodes]);

  return {
    // What the palette offers (quick-connect reads this too): hidden service
    // nodes stay registered, so flows using them still open and run.
    nodes: paletteNodes,
    groupedNodes,
    isLoading,
    error,
    getNodeDefinition,
    isValidNodeType,
    nodeIds,
    loader,
  };
}

// ============================================
// Utility Functions
// ============================================

/**
 * Get color classes for a node based on its color property
 * Uses theme-aware colors that work in both light and dark modes
 */
export function getNodeColorClasses(color: string | undefined): {
  bg: string;
  border: string;
  text: string;
} {
  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500 dark:border-green-600', text: 'text-green-700 dark:text-green-400' },
    lime: { bg: 'bg-lime-100 dark:bg-lime-900/30', border: 'border-lime-500 dark:border-lime-600', text: 'text-lime-700 dark:text-lime-400' },
    purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500 dark:border-purple-600', text: 'text-purple-700 dark:text-purple-400' },
    violet: { bg: 'bg-violet-100 dark:bg-violet-900/30', border: 'border-violet-500 dark:border-violet-600', text: 'text-violet-700 dark:text-violet-400' },
    blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500 dark:border-blue-600', text: 'text-blue-700 dark:text-blue-400' },
    orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500 dark:border-orange-600', text: 'text-orange-700 dark:text-orange-400' },
    cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900/30', border: 'border-cyan-500 dark:border-cyan-600', text: 'text-cyan-700 dark:text-cyan-400' },
    amber: { bg: 'bg-amber-100 dark:bg-amber-900/30', border: 'border-amber-500 dark:border-amber-600', text: 'text-amber-700 dark:text-amber-400' },
    pink: { bg: 'bg-pink-100 dark:bg-pink-900/30', border: 'border-pink-500 dark:border-pink-600', text: 'text-pink-700 dark:text-pink-400' },
    indigo: { bg: 'bg-indigo-100 dark:bg-indigo-900/30', border: 'border-indigo-500 dark:border-indigo-600', text: 'text-indigo-700 dark:text-indigo-400' },
    teal: { bg: 'bg-teal-100 dark:bg-teal-900/30', border: 'border-teal-500 dark:border-teal-600', text: 'text-teal-700 dark:text-teal-400' },
    emerald: { bg: 'bg-emerald-100 dark:bg-emerald-900/30', border: 'border-emerald-500 dark:border-emerald-600', text: 'text-emerald-700 dark:text-emerald-400' },
    red: { bg: 'bg-red-100 dark:bg-red-900/30', border: 'border-red-500 dark:border-red-600', text: 'text-red-700 dark:text-red-400' },
    slate: { bg: 'bg-slate-100 dark:bg-slate-900/30', border: 'border-slate-400 dark:border-slate-600', text: 'text-slate-700 dark:text-slate-400' },
  };

  const baseColor = color?.split('-')[0]?.toLowerCase() || 'slate';
  return colorMap[baseColor] || colorMap.slate;
}

// A node kind's icon (lib/nodeIcons.tsx), as the palette and quick connect import it from here.
export { getNodeIcon, nodeIconOf } from '../lib/nodeIcons';
