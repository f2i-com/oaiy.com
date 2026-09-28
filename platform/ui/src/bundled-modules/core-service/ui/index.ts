/**
 * Core Service Module — UI components.
 *
 * The web build's `loadWebBundledModules` discovers this file via
 * `import.meta.glob`; each exported component is registered by name so
 * the matching entry in module.json's `ui.nodes` array can map a node
 * type to its component.
 */

export { default as ServiceCallNode } from './ServiceCallNode';
