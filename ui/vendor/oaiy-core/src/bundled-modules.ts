/**
 * Module Registry
 *
 * This file provides a registry for modules that can be populated dynamically.
 * The actual module discovery happens in the consuming application (e.g., oaiy-desktop)
 * using import.meta.glob, and then modules are registered here.
 */

import type { ModuleManifest, NodeDefinition, RuntimeModule, ModuleCompiler, NodeUIMapping } from './module-types';
import { moduleLogger } from './logger.js';

// ============================================
// Types
// ============================================

export interface BundledModule {
  manifest: ModuleManifest;
  nodes: NodeDefinition[];
  runtime?: RuntimeModule;
  compiler?: ModuleCompiler;
}

export interface ModuleUIRegistration {
  moduleId: string;
  mappings: NodeUIMapping[];
}

// ============================================
// Module Registry State
// ============================================

// Internal state
const moduleRegistry = {
  modules: [] as BundledModule[],
  directories: {} as Record<string, string>,
};

/**
 * Register modules discovered by the consuming application.
 * This should be called once during app initialization.
 */
export function registerBundledModules(modules: BundledModule[]): void {
  moduleRegistry.modules = modules;

  // Update directories mapping
  moduleRegistry.directories = {};
  for (const module of modules) {
    moduleRegistry.directories[module.manifest.id] = module.manifest.id;
  }

  moduleLogger.debug(`Registered ${modules.length} modules`, { moduleIds: modules.map(m => m.manifest.id) });
}

/**
 * Get all registered modules.
 */
export function getBundledModulesArray(): BundledModule[] {
  return moduleRegistry.modules;
}

/**
 * The BUNDLED_MODULES export - for backwards compatibility.
 * Uses a Proxy to always return current registered modules.
 */
export const BUNDLED_MODULES: BundledModule[] = new Proxy([] as BundledModule[], {
  get(target, prop) {
    const modules = moduleRegistry.modules;
    if (prop === 'length') {
      return modules.length;
    }
    if (typeof prop === 'string' && !isNaN(Number(prop))) {
      return modules[Number(prop)];
    }
    if (prop === Symbol.iterator) {
      return modules[Symbol.iterator].bind(modules);
    }
    // Handle array methods
    const value = (modules as unknown as Record<string | symbol, unknown>)[prop];
    if (typeof value === 'function') {
      return value.bind(modules);
    }
    return value;
  },
  set() {
    console.warn('[Modules] BUNDLED_MODULES is read-only. Use registerBundledModules() instead.');
    return false;
  },
  has(target, prop) {
    return prop in moduleRegistry.modules;
  },
  ownKeys() {
    return Reflect.ownKeys(moduleRegistry.modules);
  },
  getOwnPropertyDescriptor(target, prop) {
    return Reflect.getOwnPropertyDescriptor(moduleRegistry.modules, prop);
  },
});

/**
 * MODULE_DIRECTORIES export - for backwards compatibility.
 * Uses a Proxy to always return current directories.
 */
export const MODULE_DIRECTORIES: Record<string, string> = new Proxy({} as Record<string, string>, {
  get(target, prop) {
    if (typeof prop === 'string') {
      return moduleRegistry.directories[prop];
    }
    return undefined;
  },
  set() {
    console.warn('[Modules] MODULE_DIRECTORIES is read-only.');
    return false;
  },
  has(target, prop) {
    return typeof prop === 'string' && prop in moduleRegistry.directories;
  },
  ownKeys() {
    return Reflect.ownKeys(moduleRegistry.directories);
  },
  getOwnPropertyDescriptor(target, prop) {
    if (typeof prop === 'string' && prop in moduleRegistry.directories) {
      return {
        value: moduleRegistry.directories[prop],
        writable: false,
        enumerable: true,
        configurable: true,
      };
    }
    return undefined;
  },
});

// ============================================
// Utility Functions
// ============================================

/**
 * Get all bundled module manifests
 */
export function getBundledManifests(): ModuleManifest[] {
  return moduleRegistry.modules.map(m => m.manifest);
}

/**
 * Get all bundled node definitions
 */
export function getBundledNodeDefinitions(): NodeDefinition[] {
  return moduleRegistry.modules.flatMap(m => m.nodes);
}

/**
 * Get a bundled node definition by ID
 */
export function getBundledNodeDefinition(nodeId: string): NodeDefinition | undefined {
  for (const module of moduleRegistry.modules) {
    const node = module.nodes.find(n => n.id === nodeId);
    if (node) return node;
  }
  return undefined;
}

/**
 * Get a bundled module by ID
 */
export function getBundledModule(moduleId: string): BundledModule | undefined {
  return moduleRegistry.modules.find(m => m.manifest.id === moduleId);
}

/**
 * Get the list of core module IDs
 */
export function getCoreModuleIds(): string[] {
  return moduleRegistry.modules.map(m => m.manifest.id);
}

/**
 * Get module directory mapping
 */
export function getModuleDirectories(): Record<string, string> {
  return { ...moduleRegistry.directories };
}

// ============================================
// UI Component Registration
// ============================================

/**
 * Get all UI component mappings from bundled modules
 */
export function getBundledUIRegistrations(): ModuleUIRegistration[] {
  return moduleRegistry.modules
    .filter(m => m.manifest.ui?.nodes && m.manifest.ui.nodes.length > 0)
    .map(m => ({
      moduleId: m.manifest.id,
      mappings: m.manifest.ui!.nodes!,
    }));
}

/**
 * Get all node type to component name mappings
 */
export function getNodeTypeComponentMap(): Map<string, string> {
  const map = new Map<string, string>();
  for (const module of moduleRegistry.modules) {
    if (module.manifest.ui?.nodes) {
      for (const mapping of module.manifest.ui.nodes) {
        map.set(mapping.nodeType, mapping.componentName);
      }
    }
  }
  return map;
}

/**
 * Get component name for a specific node type
 */
export function getComponentNameForNodeType(nodeType: string): string | undefined {
  for (const module of moduleRegistry.modules) {
    if (module.manifest.ui?.nodes) {
      const mapping = module.manifest.ui.nodes.find(n => n.nodeType === nodeType);
      if (mapping) return mapping.componentName;
    }
  }
  return undefined;
}

// ============================================
// FlowPlan Support Functions
// ============================================

/**
 * Get all valid FlowPlan step types from registered modules.
 * This dynamically discovers step types from node flowplan configs.
 */
export function getValidStepTypes(): string[] {
  const stepTypes = new Set<string>();

  for (const module of moduleRegistry.modules) {
    for (const node of module.nodes) {
      // Add the node ID itself as a valid step type
      stepTypes.add(node.id);

      // Add any additional step types from flowplan config
      if (node.flowplan?.stepTypes) {
        for (const stepType of node.flowplan.stepTypes) {
          stepTypes.add(stepType);
        }
      }
    }
  }

  return Array.from(stepTypes);
}

/**
 * Get all nodes that have FlowPlan support (have a flowplan config).
 */
export function getFlowPlanEnabledNodes(): NodeDefinition[] {
  return moduleRegistry.modules
    .flatMap(m => m.nodes)
    .filter(n => n.flowplan !== undefined);
}

/**
 * Get nodes that can be used as steps in a FlowPlan.
 * Returns nodes that have flowplan config or are commonly used in workflows.
 */
export function getFlowPlanStepNodes(): NodeDefinition[] {
  // Get all nodes
  const allNodes = moduleRegistry.modules.flatMap(m => m.nodes);

  // Filter to nodes that can be steps (have inputs or are processing nodes)
  return allNodes.filter(n => {
    // Exclude input nodes (they become FlowPlan inputs, not steps)
    if (n.id.startsWith('input_')) return false;

    // Exclude loop markers (they're handled specially)
    if (n.id === 'loop_start' || n.id === 'loop_end') return false;

    // Exclude macro-related nodes
    if (n.id === 'macro' || n.id === 'macro_input' || n.id === 'macro_output') return false;

    // Exclude subflow (handled specially)
    if (n.id === 'subflow') return false;

    // Include everything else
    return true;
  });
}

/**
 * Check if a step type is valid (exists as a node or step type alias).
 */
export function isValidStepType(stepType: string): boolean {
  // Check if it's a direct node ID
  const nodeDef = getBundledNodeDefinition(stepType);
  if (nodeDef) return true;

  // Check if it's a step type alias in any node's flowplan config
  for (const module of moduleRegistry.modules) {
    for (const node of module.nodes) {
      if (node.flowplan?.stepTypes?.includes(stepType)) {
        return true;
      }
    }
  }

  return false;
}
