/**
 * Node UI Component Registry
 *
 * This registry maps node type IDs to their React UI components.
 * It allows dynamic registration from module UI folders.
 */

import React from 'react';

// Component type for nodes - uses 'any' because node props vary by node type
// and the registry needs to accept any node component without strict typing
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type NodeComponent = React.ComponentType<any>;

// Registry of node type -> component
const nodeTypeRegistry: Map<string, NodeComponent> = new Map();

// Map from component name to component (for dynamic lookup)
const componentNameMap: Map<string, NodeComponent> = new Map();

// Version counter that increments when registry changes
// This allows consumers to know when to refresh their cached nodeTypes
let registryVersion = 0;

/**
 * Get the current registry version
 * Use this in useMemo/useEffect dependencies to react to registry changes
 */
export function getRegistryVersion(): number {
  return registryVersion;
}

/**
 * Register a node type with its UI component
 */
export function registerNodeComponent(nodeType: string, component: NodeComponent): void {
  nodeTypeRegistry.set(nodeType, component);
  registryVersion++;
}

/**
 * Unregister a node type
 */
export function unregisterNodeComponent(nodeType: string): boolean {
  const deleted = nodeTypeRegistry.delete(nodeType);
  if (deleted) {
    registryVersion++;
  }
  return deleted;
}

/**
 * Register a component by name (for module manifest registration)
 */
export function registerComponentByName(componentName: string, component: NodeComponent): void {
  componentNameMap.set(componentName, component);
}

/**
 * Register a node type by component name (for module manifest registration)
 */
export function registerNodeByComponentName(nodeType: string, componentName: string): boolean {
  const component = componentNameMap.get(componentName);
  if (component) {
    nodeTypeRegistry.set(nodeType, component);
    return true;
  }
  console.warn(`[NodeRegistry] Unknown component name: ${componentName} for node type: ${nodeType}`);
  return false;
}

/**
 * Register multiple nodes from module UI config
 */
export function registerModuleNodes(
  uiConfig: { nodes?: Array<{ nodeType: string; componentName: string }> } | undefined
): void {
  if (!uiConfig?.nodes) return;

  for (const mapping of uiConfig.nodes) {
    registerNodeByComponentName(mapping.nodeType, mapping.componentName);
  }
}

/**
 * Get the component for a node type
 */
export function getNodeComponent(nodeType: string): NodeComponent | undefined {
  return nodeTypeRegistry.get(nodeType);
}

/**
 * Get all registered node types
 */
export function getRegisteredNodeTypes(): string[] {
  return Array.from(nodeTypeRegistry.keys());
}

/**
 * Get the nodeTypes object for React Flow (Record<string, Component>)
 */
export function getNodeTypes(): Record<string, NodeComponent> {
  const result: Record<string, NodeComponent> = {};
  for (const [nodeType, component] of nodeTypeRegistry) {
    result[nodeType] = component;
  }
  return result;
}

/**
 * Check if a node type is registered
 */
export function isNodeTypeRegistered(nodeType: string): boolean {
  return nodeTypeRegistry.has(nodeType);
}

/**
 * Get a component by its name
 */
export function getComponentByName(componentName: string): NodeComponent | undefined {
  return componentNameMap.get(componentName);
}

/**
 * Clear all registrations (for testing)
 */
export function clearRegistry(): void {
  nodeTypeRegistry.clear();
  componentNameMap.clear();
}

// Note: nodeTypes is dynamically populated via getNodeTypes()
// For backwards compatibility, export a getter function
// Uses 'any' for component props since each node type has different prop requirements
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export let nodeTypes: Record<string, React.ComponentType<any>> = {};

// Function to refresh nodeTypes export (call after registration)
export function refreshNodeTypes(): void {
  nodeTypes = getNodeTypes();
}
