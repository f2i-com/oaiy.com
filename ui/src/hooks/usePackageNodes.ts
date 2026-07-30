/**
 * usePackageNodes - Manages custom nodes embedded in .oaiy packages
 *
 * Handles:
 * - Loading node definitions from packages
 * - Prefixing node IDs for isolation (pkg:{packageId}:{nodeId})
 * - Registering/unregistering with ModuleLoader and nodeRegistry
 * - Cleanup when packages are closed
 */

import { useState, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { ModuleManifest, NodeDefinition, EmbeddedCustomNode, EmbeddedNodeExtension, OAIYPackageManifestWithEmbedded } from 'oaiy-core';
import type { PackageNodeModule } from 'oaiy-core';
import { getModuleLoader, createTemplateCompiler } from 'oaiy-core';
import { registerNodeComponent, unregisterNodeComponent, refreshNodeTypes, getRegisteredNodeTypes } from 'oaiy-ui-components';
import { GenericNode } from 'oaiy-ui-components';
import { compileAndRegisterCustomNode, unregisterPackageNodes as unregisterCustomNodes, getPackageCustomNodes } from '../services/customNodeRegistry';
import { compileAndRegisterExtension, unregisterPackageExtensions } from '../services/nodeExtensionRegistry';
import { createCustomNodeUIWrapper, hasCustomNodeUI, clearCachedUI } from '../services/customNodeUILoader';
import { packageLogger } from '../utils/logger';

/**
 * Information about a package node
 */
export interface PackageNodeInfo {
  /** Package ID this node belongs to */
  packageId: string;
  /** Original node ID (without prefix) */
  originalId: string;
  /** Full prefixed ID: pkg:{packageId}:{nodeId} */
  prefixedId: string;
  /** Node definition with prefixed ID */
  definition: NodeDefinition;
  /** Module manifest the node came from */
  moduleManifest: ModuleManifest;
  /** Node version from definition (for compatibility tracking) */
  version?: string;
  /** Module category for proper grouping */
  category: string;
  /** Custom UI component path if specified */
  customUIPath?: string;
  /** Whether this node has a custom UI component (vs GenericNode) */
  hasCustomUI: boolean;
}

/**
 * Result from the Tauri read_package_nodes command
 */
interface PackageNodesResult {
  manifest: string;
  nodes: string[];
}

/**
 * Create a prefixed node ID
 */
function createPrefixedNodeId(packageId: string, nodeId: string, customPrefix?: string): string {
  const prefix = customPrefix || packageId;
  return `pkg:${prefix}:${nodeId}`;
}

/**
 * Sanitize a module ID to match the validation pattern: /^[a-z][a-z0-9-]*$/
 * Replaces invalid characters (colons, dots, underscores) with hyphens
 */
function sanitizeModuleId(id: string): string {
  return id.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+/, 'pkg-');
}

/**
 * Validation result for a package node
 */
interface NodeValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
}

/**
 * Validate a package node definition
 * Checks template syntax, input/output references, and property references
 */
function validatePackageNode(node: NodeDefinition): NodeValidationResult {
  const warnings: string[] = [];
  const errors: string[] = [];

  // Check for required compiler config
  if (!node.compiler) {
    errors.push(`Node '${node.id}' has no compiler configuration`);
    return { valid: false, warnings, errors };
  }

  // Check that node has either template or customHandler
  if (!node.compiler.template && !node.compiler.customHandler) {
    errors.push(`Node '${node.id}' must have either a template or customHandler`);
    return { valid: false, warnings, errors };
  }

  // If there's a template, validate it
  if (node.compiler.template) {
    const template = node.compiler.template;

    // Extract all template references ({{...}})
    const referenceRegex = /\{\{([^}]+)\}\}/g;
    const references: string[] = [];
    let match;
    while ((match = referenceRegex.exec(template)) !== null) {
      references.push(match[1].trim());
    }

    // Get valid input IDs
    const validInputs = new Set(node.inputs?.map(i => i.id) || []);
    // Get valid property IDs
    const validProps = new Set(node.properties?.map(p => p.id) || []);

    // Check each reference
    for (const ref of references) {
      // Skip built-in references
      if (ref === 'outputVar' || ref === 'sanitizedId' || ref === 'nodeId') {
        continue;
      }

      // Check inputs.X references
      if (ref.startsWith('inputs.')) {
        const inputId = ref.substring(7);
        if (!validInputs.has(inputId)) {
          warnings.push(`Template references unknown input '${inputId}' in node '${node.id}'`);
        }
      }
      // Check props.X references
      else if (ref.startsWith('props.')) {
        const propId = ref.substring(6);
        if (!validProps.has(propId)) {
          warnings.push(`Template references unknown property '${propId}' in node '${node.id}'`);
        }
      }
      // Check data.X references (node data, usually valid)
      else if (ref.startsWith('data.')) {
        // Data references are typically valid, just note them
      }
      // Unknown reference format
      else {
        warnings.push(`Template has unknown reference format '${ref}' in node '${node.id}'`);
      }
    }

    // Check for common template issues
    if (!template.includes('{{outputVar}}') && !node.compiler.outputVariable) {
      warnings.push(`Node '${node.id}' template doesn't set {{outputVar}} and has no outputVariable defined`);
    }
  }

  // Check for basic node structure
  if (!node.name) {
    errors.push(`Node '${node.id}' has no name`);
  }

  if (!node.outputs || node.outputs.length === 0) {
    warnings.push(`Node '${node.id}' has no outputs defined`);
  }

  return {
    valid: errors.length === 0,
    warnings,
    errors,
  };
}

/**
 * Hook for managing custom nodes from packages
 */
export function usePackageNodes() {
  // Map of packageId -> array of PackageNodeInfo
  const [packageNodes, setPackageNodes] = useState<Map<string, PackageNodeInfo[]>>(new Map());

  // Track registered node IDs for cleanup
  const registeredNodeIds = useRef<Set<string>>(new Set());

  // Track registered module IDs (for ModuleLoader cleanup)
  const registeredModuleIds = useRef<Map<string, string[]>>(new Map()); // packageId -> moduleIds[]

  // Get the module loader for registering node definitions
  const moduleLoader = getModuleLoader();

  /**
   * Load nodes from a package's node modules
   */
  const loadPackageNodes = useCallback(async (
    packageId: string,
    packagePath: string,
    nodeModules: PackageNodeModule[]
  ): Promise<PackageNodeInfo[]> => {
    const loadedNodes: PackageNodeInfo[] = [];

    for (const nodeModule of nodeModules) {
      try {
        // Read node definitions from the package
        packageLogger.debug(` Loading nodes from module:`, {
          packagePath,
          modulePath: nodeModule.path,
        });
        const result = await invoke<PackageNodesResult>('read_package_nodes', {
          packagePath,
          modulePath: nodeModule.path,
        });

        // Parse module manifest
        const manifest: ModuleManifest = JSON.parse(result.manifest);

        // Parse node definitions and create prefixed versions
        const prefixedNodes: NodeDefinition[] = [];
        for (const nodeJson of result.nodes) {
          try {
            const originalNode: NodeDefinition = JSON.parse(nodeJson);

            // Validate node definition
            const validation = validatePackageNode(originalNode);

            // Log any warnings
            for (const warning of validation.warnings) {
              packageLogger.warn(` ${warning}`);
            }

            // Skip nodes with errors
            if (!validation.valid) {
              for (const error of validation.errors) {
                packageLogger.error(` ${error}`);
              }
              packageLogger.error(` Skipping invalid node: ${originalNode.id}`);
              continue;
            }

            // Create prefixed ID
            const prefixedId = createPrefixedNodeId(
              packageId,
              originalNode.id,
              nodeModule.prefix
            );

            // Create node definition with prefixed ID
            const prefixedNode: NodeDefinition = {
              ...originalNode,
              id: prefixedId,
            };

            prefixedNodes.push(prefixedNode);

            // Check for custom UI component
            const customUI = nodeModule.uiComponents?.find(
              ui => ui.nodeType === originalNode.id
            );

            // Create node info with version and category tracking
            const nodeInfo: PackageNodeInfo = {
              packageId,
              originalId: originalNode.id,
              prefixedId,
              definition: prefixedNode,
              moduleManifest: manifest,
              version: originalNode.version,
              category: manifest.category || 'Custom',
              customUIPath: customUI?.componentPath,
              hasCustomUI: !!customUI,
            };

            loadedNodes.push(nodeInfo);
          } catch (err) {
            packageLogger.warn(` Failed to parse node definition:`, err);
          }
        }

        // Register with ModuleLoader for compiler support
        // Create a package-specific module manifest with sanitized ID
        const rawModuleId = `pkg-${nodeModule.prefix || packageId}-${manifest.id}`;
        const pkgModuleId = sanitizeModuleId(rawModuleId);
        const pkgModuleManifest: ModuleManifest = {
          ...manifest,
          id: pkgModuleId,
        };

        // Build a map of node definitions for the template compiler
        const nodeDefinitionsMap = new Map<string, NodeDefinition>();
        for (const node of prefixedNodes) {
          nodeDefinitionsMap.set(node.id, node);
        }

        // Create a template compiler for these nodes
        const templateCompiler = createTemplateCompiler(
          `Package:${packageId}`,
          nodeDefinitionsMap
        );

        const loadResult = await moduleLoader.loadModule(
          pkgModuleManifest,
          prefixedNodes,
          undefined, // No runtime for template-based nodes
          `package:${packageId}/${nodeModule.path}`,
          templateCompiler // Pass the template compiler
        );

        if (!loadResult.success) {
          packageLogger.error(` Failed to register module with loader:`, loadResult.error);
          packageLogger.error(` Module manifest:`, pkgModuleManifest);
          packageLogger.error(` Prefixed nodes:`, prefixedNodes);
        } else {
          // Track registered module ID for cleanup
          const existingModuleIds = registeredModuleIds.current.get(packageId) || [];
          existingModuleIds.push(pkgModuleId);
          registeredModuleIds.current.set(packageId, existingModuleIds);
        }

        // Register with nodeRegistry using GenericNode for UI rendering
        // Note: Custom UI components would require dynamic loading (future enhancement)
        for (const prefixedNode of prefixedNodes) {
          const nodeInfo = loadedNodes.find(n => n.prefixedId === prefixedNode.id);
          if (nodeInfo?.hasCustomUI) {
            // Custom UI components from packages require dynamic loading:
            // 1. Read the UI file from disk (via Tauri fs plugin)
            // 2. Compile TypeScript/JSX (via esbuild-wasm, similar to pluginCompiler.ts)
            // 3. Execute the compiled code to get the React component
            // 4. Register it with the node registry
            // For now, use GenericNode as a fallback - this works for most nodes.
            packageLogger.debug(` Node ${prefixedNode.id} has custom UI at ${nodeInfo.customUIPath} (using GenericNode fallback)`);
          }
          packageLogger.debug(` Registering node component: ${prefixedNode.id} with GenericNode`);
          registerNodeComponent(prefixedNode.id, GenericNode);
          registeredNodeIds.current.add(prefixedNode.id);
          packageLogger.debug(` Registered node: ${prefixedNode.id}`);
        }

        // Debug: check what's in the registry now
        const registeredTypes = getRegisteredNodeTypes();
        packageLogger.debug(` All registered node types after registration:`, registeredTypes.filter(t => t.startsWith('pkg:')));
      } catch (err) {
        packageLogger.error(` Failed to load nodes from ${nodeModule.path}:`, err);
      }
    }

    // Update state
    if (loadedNodes.length > 0) {
      setPackageNodes(prev => {
        const next = new Map(prev);
        const existing = next.get(packageId) || [];
        next.set(packageId, [...existing, ...loadedNodes]);
        return next;
      });

      // Refresh the nodeTypes export
      refreshNodeTypes();
    }

    packageLogger.debug(` Loaded ${loadedNodes.length} nodes for package ${packageId}`);
    return loadedNodes;
  }, [moduleLoader]);

  /**
   * Unload nodes for a package
   */
  const unloadPackageNodes = useCallback(async (packageId: string) => {
    const nodes = packageNodes.get(packageId);
    if (!nodes) return;

    // Unregister each node from the UI registry
    for (const node of nodes) {
      unregisterNodeComponent(node.prefixedId);
      registeredNodeIds.current.delete(node.prefixedId);
      packageLogger.debug(` Unregistered node: ${node.prefixedId}`);
    }

    // Unload modules from the ModuleLoader
    const moduleIds = registeredModuleIds.current.get(packageId) || [];
    for (const moduleId of moduleIds) {
      await moduleLoader.unloadModule(moduleId);
      packageLogger.debug(` Unloaded module: ${moduleId}`);
    }
    registeredModuleIds.current.delete(packageId);

    // Remove from state
    setPackageNodes(prev => {
      const next = new Map(prev);
      next.delete(packageId);
      return next;
    });

    refreshNodeTypes();
  }, [packageNodes, moduleLoader]);

  /**
   * Get all nodes for a specific package
   */
  const getPackageNodes = useCallback((packageId: string): PackageNodeInfo[] => {
    return packageNodes.get(packageId) || [];
  }, [packageNodes]);

  /**
   * Get all loaded package nodes
   */
  const getAllPackageNodes = useCallback((): PackageNodeInfo[] => {
    const all: PackageNodeInfo[] = [];
    for (const nodes of packageNodes.values()) {
      all.push(...nodes);
    }
    return all;
  }, [packageNodes]);

  /**
   * Get a specific node definition by prefixed ID
   */
  const getPackageNodeDefinition = useCallback((prefixedId: string): NodeDefinition | undefined => {
    for (const nodes of packageNodes.values()) {
      const node = nodes.find(n => n.prefixedId === prefixedId);
      if (node) return node.definition;
    }
    return undefined;
  }, [packageNodes]);

  /**
   * Check if a node ID is a package node
   */
  const isPackageNode = useCallback((nodeId: string): boolean => {
    return nodeId.startsWith('pkg:');
  }, []);

  /**
   * Get the package ID from a prefixed node ID
   */
  const getPackageIdFromNodeId = useCallback((prefixedNodeId: string): string | null => {
    if (!prefixedNodeId.startsWith('pkg:')) return null;
    // Format: pkg:{prefix}:{nodeId}
    const parts = prefixedNodeId.split(':');
    if (parts.length >= 2) {
      return parts[1];
    }
    return null;
  }, []);

  /**
   * Load embedded custom nodes from a package manifest
   * These are TypeScript-based nodes that get compiled on load
   */
  const loadEmbeddedCustomNodes = useCallback(async (
    packageId: string,
    customNodes: EmbeddedCustomNode[]
  ): Promise<{ loaded: number; failed: number }> => {
    let loaded = 0;
    let failed = 0;
    const loadedNodeInfos: PackageNodeInfo[] = [];

    packageLogger.debug(` Compiling ${customNodes.length} embedded custom nodes for ${packageId}`);

    for (const node of customNodes) {
      const result = await compileAndRegisterCustomNode(packageId, node);

      if (result.success) {
        loaded++;

        // Register with UI registry - use custom UI if available, otherwise GenericNode
        const fullNodeType = `pkg:${packageId}:${node.id}`;

        // Check if there's a custom UI component compiled for this node
        if (hasCustomNodeUI(fullNodeType)) {
          const customUIComponent = createCustomNodeUIWrapper(fullNodeType);
          if (customUIComponent) {
            registerNodeComponent(fullNodeType, customUIComponent);
            packageLogger.debug(` Compiled and registered custom node with custom UI: ${fullNodeType}`);
          } else {
            // Fallback to GenericNode if custom UI wrapper creation fails
            registerNodeComponent(fullNodeType, GenericNode);
            packageLogger.debug(` Compiled and registered custom node with GenericNode (UI wrapper failed): ${fullNodeType}`);
          }
        } else {
          registerNodeComponent(fullNodeType, GenericNode);
          packageLogger.debug(` Compiled and registered custom node with GenericNode: ${fullNodeType}`);
        }
        registeredNodeIds.current.add(fullNodeType);

        // Get the compiled node definition to add to packageNodes for palette
        const registeredNodes = getPackageCustomNodes(packageId);
        const registeredNode = registeredNodes.find(n => n.definition.id === node.id);

        if (registeredNode) {
          // Create a PackageNodeInfo for the palette
          const nodeInfo: PackageNodeInfo = {
            packageId,
            originalId: node.id,
            prefixedId: fullNodeType,
            definition: registeredNode.nodeDefinition,
            moduleManifest: {
              id: `custom-node-${packageId}-${node.id}`,
              name: node.name,
              version: '1.0.0',
              description: node.description || '',
              category: 'Custom', // Custom nodes use Custom category
              author: 'Package',
              nodes: [fullNodeType],
            },
            category: 'Package', // Show in Package category at the top (UI grouping)
            hasCustomUI: hasCustomNodeUI(fullNodeType),
          };
          loadedNodeInfos.push(nodeInfo);
        }
      } else {
        failed++;
        packageLogger.error(` Failed to compile custom node ${node.id}:`, result.error);
      }
    }

    // Add custom nodes to the packageNodes state for the palette
    if (loadedNodeInfos.length > 0) {
      setPackageNodes(prev => {
        const next = new Map(prev);
        const existing = next.get(packageId) || [];
        next.set(packageId, [...existing, ...loadedNodeInfos]);
        return next;
      });
      refreshNodeTypes();
    }

    packageLogger.debug(` Embedded custom nodes: ${loaded} loaded, ${failed} failed`);
    return { loaded, failed };
  }, []);

  /**
   * Load embedded node extensions from a package manifest
   */
  const loadEmbeddedExtensions = useCallback(async (
    packageId: string,
    extensions: EmbeddedNodeExtension[]
  ): Promise<{ loaded: number; failed: number }> => {
    let loaded = 0;
    let failed = 0;

    packageLogger.debug(` Compiling ${extensions.length} node extensions for ${packageId}`);

    for (const extension of extensions) {
      const result = await compileAndRegisterExtension(packageId, extension);

      if (result.success) {
        loaded++;
        packageLogger.debug(` Registered extension: ${extension.id} for ${extension.extends}`);
      } else {
        failed++;
        packageLogger.error(` Failed to compile extension ${extension.id}:`, result.error);
      }
    }

    packageLogger.debug(` Node extensions: ${loaded} loaded, ${failed} failed`);
    return { loaded, failed };
  }, []);

  /**
   * Load all embedded content from a package manifest (custom nodes + extensions)
   */
  const loadEmbeddedContent = useCallback(async (
    packageId: string,
    manifest: OAIYPackageManifestWithEmbedded
  ): Promise<{ customNodes: { loaded: number; failed: number }; extensions: { loaded: number; failed: number } }> => {
    const results = {
      customNodes: { loaded: 0, failed: 0 },
      extensions: { loaded: 0, failed: 0 },
    };

    // Load embedded custom nodes
    if (manifest.embeddedCustomNodes && manifest.embeddedCustomNodes.length > 0) {
      results.customNodes = await loadEmbeddedCustomNodes(packageId, manifest.embeddedCustomNodes);
    }

    // Load embedded extensions
    if (manifest.embeddedNodeExtensions && manifest.embeddedNodeExtensions.length > 0) {
      results.extensions = await loadEmbeddedExtensions(packageId, manifest.embeddedNodeExtensions);
    }

    return results;
  }, [loadEmbeddedCustomNodes, loadEmbeddedExtensions]);

  /**
   * Unload all embedded content for a package
   */
  const unloadEmbeddedContent = useCallback(async (packageId: string) => {
    // Unregister custom nodes
    const customNodes = getPackageCustomNodes(packageId);
    for (const node of customNodes) {
      const fullType = `pkg:${packageId}:${node.definition.id}`;
      unregisterNodeComponent(fullType);
      // Invalidate the compiled-UI cache too — otherwise a hot-reload recompiles the
      // node but getCustomNodeUIComponent short-circuits on the STALE cached component
      // (renders the OLD UI), and an unloaded package leaks its component indefinitely.
      clearCachedUI(fullType);
      registeredNodeIds.current.delete(fullType);
    }
    unregisterCustomNodes(packageId);

    // Unregister extensions
    unregisterPackageExtensions(packageId);

    refreshNodeTypes();
    packageLogger.debug(` Unloaded embedded content for ${packageId}`);
  }, []);

  return {
    /** All package nodes organized by package ID */
    packageNodes,
    /** Load nodes from a package */
    loadPackageNodes,
    /** Unload nodes when closing a package */
    unloadPackageNodes,
    /** Get nodes for a specific package */
    getPackageNodes,
    /** Get all loaded package nodes */
    getAllPackageNodes,
    /** Get a node definition by prefixed ID */
    getPackageNodeDefinition,
    /** Check if a node ID is from a package */
    isPackageNode,
    /** Extract package ID from a node ID */
    getPackageIdFromNodeId,
    /** Load embedded custom nodes (TypeScript-based) */
    loadEmbeddedCustomNodes,
    /** Load embedded node extensions */
    loadEmbeddedExtensions,
    /** Load all embedded content from a manifest */
    loadEmbeddedContent,
    /** Unload all embedded content for a package */
    unloadEmbeddedContent,
  };
}
