/**
 * Custom Node UI Loader
 *
 * Dynamically loads and renders UI components for custom nodes.
 * Uses compiled JavaScript from the custom node compiler.
 */

import React from 'react';
import { getCustomNode } from './customNodeRegistry';
import { createLogger } from '../utils/logger';
import { SandboxedCustomNodeUI } from '../components/SandboxedCustomNodeUI';

const logger = createLogger('CustomNodeUILoader');

// Version counter for cache invalidation. (The old main-realm component cache was removed
// with loadUIComponent — UIs now render in the sandbox iframe, which holds no host-side
// component; only this version is still used, to invalidate React wrappers on reload.)
let loaderVersion = 0;

/**
 * Props passed to custom node UI components
 */
export interface CustomNodeUIProps {
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
  // Standard callbacks for node UI
  onDataChange?: (updates: Record<string, unknown>) => void;
}

// REMOVED: loadUIComponent + getCustomNodeUIComponent. The custom-node UI used to be
// compiled into the MAIN realm via `new Function`, which gave untrusted package code full
// localStorage / fetch / DOM / __TAURI_INTERNALS__ access (secret exfiltration; native
// exec on desktop). It now runs inside an isolated, opaque-origin sandbox iframe — see
// createCustomNodeUIWrapper below + SandboxedCustomNodeUI / src/sandbox/sandbox-main.tsx.

/**
 * Check if a custom node has a custom UI component
 */
export function hasCustomNodeUI(fullNodeType: string): boolean {
  const registered = getCustomNode(fullNodeType);
  return !!(registered?.compiled?.ui);
}

/**
 * Bump the loader version for a node (useful when reloading) so React re-creates wrappers.
 */
export function clearCachedUI(_fullNodeType: string): void {
  loaderVersion++;
}

/**
 * Bump the loader version for all nodes.
 */
export function clearAllCachedUI(): void {
  loaderVersion++;
}

/**
 * Get the loader version (for cache invalidation in React)
 */
export function getUILoaderVersion(): number {
  return loaderVersion;
}

/**
 * Create a wrapper component that renders the custom UI in an ISOLATED sandbox iframe.
 * Used by the node registry to create a proper React component.
 *
 * SECURITY: the untrusted package UI bundle is no longer run via `new Function` in the
 * main realm (which had full localStorage/fetch/DOM/__TAURI_INTERNALS__ access — see
 * loadUIComponent, now unused). Instead we hand the compiled code to SandboxedCustomNodeUI,
 * which executes it inside an opaque-origin sandboxed iframe that cannot read the app's
 * secrets and whose CSP blocks fetch-type egress (it CAN still self-navigate to leak its OWN
 * node.data, so we keep app-wide secrets out of node.data). Data flows over a narrow
 * postMessage bridge (both ends source-checked).
 */
export function createCustomNodeUIWrapper(
  fullNodeType: string
): React.ComponentType<CustomNodeUIProps> | null {
  const registered = getCustomNode(fullNodeType);
  const code = registered?.compiled?.ui;
  if (!code) return null;
  const nodeTypeId = registered.definition.id;

  const Wrapper: React.FC<CustomNodeUIProps> = (props) =>
    React.createElement(SandboxedCustomNodeUI, { ...props, code, nodeTypeId });

  Wrapper.displayName = `SandboxedCustomNodeUI(${fullNodeType})`;

  return Wrapper;
}

/**
 * Pre-load UI components for all custom nodes in a package
 */
export function preloadPackageUI(packageId: string): number {
  let loaded = 0;

  // This would iterate through all nodes for the package
  // For now, components are loaded on-demand
  logger.debug(`Pre-loading UI for package ${packageId}`);

  return loaded;
}
