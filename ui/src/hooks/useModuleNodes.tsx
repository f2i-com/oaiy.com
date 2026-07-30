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
import type { PackageNodeInfo } from './usePackageNodes';
import { listAllServices } from '../utils/serviceRegistry';
import { listCompanionServices } from '../lib/companionServices';
import { moduleLogger as logger } from '../utils/logger';

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
  'service_call',
]);

// ============================================
// Hook Implementation
// ============================================

export function useModuleNodes(options?: UseModuleNodesOptions) {
  const { activePackageId, packageNodes: pkgNodes } = options || {};

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

  // Group nodes by category (including package nodes when active)
  const groupedNodes = useMemo<GroupedNodes[]>(() => {
    const groups = new Map<ModuleCategory | 'Package', ModuleNodeInfo[]>();

    // Add built-in nodes (filter out package nodes - they're handled separately below)
    for (const node of nodes) {
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
    // The user's own services PLUS any the OAIY Companion is currently
    // exposing — so a service you start in the desktop app shows up here as
    // a draggable palette entry (id `companion:<id>`), the same as a custom
    // one. They refresh live: companionServices.publish() fires the same
    // invalidate that bumps servicesVersion below.
    const services = [...listAllServices(), ...listCompanionServices()];
    const syntheticModule: ModuleManifest = {
      id: 'core-service',
      name: 'Service',
      version: '1.0.0',
      category: 'AI' as ModuleCategory,
      description: 'Custom + companion-managed services',
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
  }, [nodes, activePackageId, pkgNodes, servicesVersion]);

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
    nodes,
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

/**
 * Get a simple icon SVG for a node based on its icon name
 */
export function getNodeIcon(iconName: string | undefined): React.ReactElement {
  // Default box icon
  const defaultIcon = (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
  );

  // Icon mapping for all bundled module icons
  const icons: Record<string, React.ReactElement> = {
    // Input icons
    type: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
      </svg>
    ),
    file: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    'file-input': (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
      </svg>
    ),
    'file-text': (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    folder: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
      </svg>
    ),
    // AI icons
    brain: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
      </svg>
    ),
    // Flow control icons
    'git-branch': (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    play: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    stop: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
      </svg>
    ),
    'check-circle': (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812z" clipRule="evenodd" />
      </svg>
    ),
    share: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
      </svg>
    ),
    // Utility icons
    tool: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    code: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
    database: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path d="M3 12v3c0 1.657 3.134 3 7 3s7-1.343 7-3v-3c0 1.657-3.134 3-7 3s-7-1.343-7-3z" />
        <path d="M3 7v3c0 1.657 3.134 3 7 3s7-1.343 7-3V7c0 1.657-3.134 3-7 3S3 8.657 3 7z" />
        <path d="M17 5c0 1.657-3.134 3-7 3S3 6.657 3 5s3.134-3 7-3 7 1.343 7 3z" />
      </svg>
    ),
    // File system icons
    save: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
    ),
    scissors: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
      </svg>
    ),
    // Image icons
    image: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
      </svg>
    ),
    eye: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    grid: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    // Video icons
    video: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
    film: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
      </svg>
    ),
    // Browser icons
    globe: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
      </svg>
    ),
    zap: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    cursor: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
      </svg>
    ),
    // Audio icons
    mic: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    ),
    'volume-2': (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
      </svg>
    ),
    music: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
      </svg>
    ),
    download: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
    ),
  };

  return icons[iconName || ''] || defaultIcon;
}
