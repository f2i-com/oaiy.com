// OAIY UI Components - Shared React components for workflow builder
// This package is used by oaiy-ui and oaiy-desktop
//
// NOTE: Node UI components are now located in oaiy-core/modules/*/ui/
// This package provides shared components, hooks, and the node registry.

// Node registry and shared node components
export * from './nodes/index.js';

// Node registry for dynamic registration
export {
  registerNodeComponent,
  unregisterNodeComponent,
  registerComponentByName,
  registerNodeByComponentName,
  registerModuleNodes,
  getNodeComponent,
  getRegisteredNodeTypes,
  getNodeTypes,
  getRegistryVersion,
  isNodeTypeRegistered,
  getComponentByName,
  clearRegistry,
  refreshNodeTypes,
  type NodeComponent,
} from './registry/nodeRegistry.js';

// Shared components (used by module UI components)
export { default as CollapsibleNodeWrapper } from './components/CollapsibleNodeWrapper.js';
export type { CollapsibleNodeWrapperProps, ValidationIssue, HandleConfig } from './components/CollapsibleNodeWrapper.js';

// Hooks (used by module UI components)
export { useNodeResize } from './hooks/useNodeResize.js';
export {
  useDynamicOptions,
  invalidateDynamicOptions,
  registerDynamicOptionsResolver,
  subscribeToDynamicOptionsInvalidation,
} from './hooks/useDynamicOptions.js';
