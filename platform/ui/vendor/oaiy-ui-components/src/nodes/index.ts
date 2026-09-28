/**
 * Node Components Index
 *
 * Node UI components are now located in their respective module folders:
 * - oaiy-core/modules/core-input/ui/
 * - oaiy-core/modules/core-ai/ui/
 * - etc.
 *
 * This file exports the registry functions and shared components.
 */

// Generic Node Component (for modular nodes / fallback)
export { GenericNode } from './GenericNode';

// Specialized Node Components
export { OutputNode } from './OutputNode';
export { GroupNode } from './GroupNode';

// Property Field Components (shared)
export * from './fields';

// Re-export nodeTypes from the registry
export { nodeTypes } from '../registry/nodeRegistry';

// Re-export registry functions for dynamic registration
export {
  registerNodeComponent,
  registerComponentByName,
  registerNodeByComponentName,
  registerModuleNodes,
  getNodeComponent,
  getRegisteredNodeTypes,
  getNodeTypes,
  isNodeTypeRegistered,
  getComponentByName,
  clearRegistry,
  refreshNodeTypes,
  type NodeComponent,
} from '../registry/nodeRegistry';
