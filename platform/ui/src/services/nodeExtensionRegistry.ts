/**
 * Node Extension Registry
 *
 * Manages node extensions that add functionality to existing nodes.
 * Extensions can add inputs, outputs, properties, and hook into compilation/runtime.
 */

import type { PackageNodeExtension, NodeDefinition, PropertyType } from 'oaiy-core';
import type React from 'react';
import { compileCustomNode } from './customNodeCompiler';
import { createLogger } from '../utils/logger';

const logger = createLogger('NodeExtensionRegistry');

// Use local type alias for clarity
type NodeExtension = PackageNodeExtension;

// ============================================
// Types
// ============================================

export interface CompilerHookContext {
  node: { id: string; type: string; data: Record<string, unknown> };
  inputs: Map<string, string>;
  outputVar: string;
  sanitizedId: string;
  escapeString: (str: string) => string;
  code: string; // The base node's compiled code
}

export interface CompilerHook {
  /** Called before base node compilation */
  preCompile?: (node: unknown, context: unknown) => unknown;
  /** Called after base node compilation - can modify the generated code */
  postCompile?: (code: string, node: unknown, context: unknown) => string;
  /** Full override - replaces base compilation entirely */
  compile?: (node: unknown, context: unknown) => string;
}

export interface RuntimeHookContext {
  log: (level: string, message: string) => void;
  nodeId: string;
  nodeData: Record<string, unknown>;
}

export interface RuntimeHook {
  /** Called before base node execution - can modify inputs */
  preExecute?: (inputs: Record<string, unknown>, context: RuntimeHookContext) => Record<string, unknown>;
  /** Called after base node execution - can modify result */
  postExecute?: (result: unknown, inputs: Record<string, unknown>, context: RuntimeHookContext) => unknown;
  /** Full override - replaces base execution entirely */
  execute?: (
    inputs: Record<string, unknown>,
    context: RuntimeHookContext,
    baseExecute: (inputs: Record<string, unknown>) => Promise<unknown>
  ) => Promise<unknown>;
}

export interface RegisteredExtension {
  extension: NodeExtension;
  packageId: string;
  compiled?: {
    compilerHook?: string;
    runtimeHook?: string;
    ui?: string;
  };
  compilerHook?: CompilerHook;
  runtimeHook?: RuntimeHook;
  uiComponent?: React.ComponentType<unknown>;
}

// ============================================
// Registry State
// ============================================

// Map from target node type to list of extensions
const extensionsByNodeType = new Map<string, RegisteredExtension[]>();

// Map from extension ID to extension (for lookup)
const extensionsById = new Map<string, RegisteredExtension>();

// Map from package ID to extension IDs
const extensionsByPackage = new Map<string, string[]>();

// Version counter for cache invalidation
let registryVersion = 0;

// ============================================
// Helper Functions
// ============================================

/**
 * Get extension ID
 */
function getExtensionId(packageId: string, extensionId: string): string {
  return `${packageId}:${extensionId}`;
}

// (Removed loadHook: it ran untrusted extension hook code via `new Function` in the main
// realm. See the note in registerExtension — extension hooks are a dead, never-applied
// feature, so the exec was pure risk with no benefit.)

// ============================================
// Registry Functions
// ============================================

/**
 * Register a node extension
 */
export function registerExtension(
  packageId: string,
  extension: NodeExtension,
  compiled?: { compilerHook?: string; runtimeHook?: string; ui?: string }
): boolean {
  const fullId = getExtensionId(packageId, extension.id);

  try {
    const registered: RegisteredExtension = {
      extension,
      packageId,
      compiled,
    };

    // SECURITY: the compiler/runtime EXTENSION HOOKS are intentionally NOT executed.
    // They used to be run via `new Function` in the MAIN app realm (full
    // localStorage/fetch/DOM/__TAURI_INTERNALS__ access), yet the consumers that would
    // apply them (applyCompilerHooks / applyRuntimeHooks / getExtendedNodeDefinition)
    // have no callers — a dead feature — and the emitted global prefix never matched the
    // lookup, so the hooks never loaded successfully anyway. We keep the extension's
    // METADATA (below) but run none of its code. If hooks are ever revived they MUST run
    // in the same isolated sandbox as the custom-node UI (see SandboxedCustomNodeUI), not
    // in the app realm.

    // Store by ID
    extensionsById.set(fullId, registered);

    // Store by target node type
    const existing = extensionsByNodeType.get(extension.extends) || [];
    existing.push(registered);
    extensionsByNodeType.set(extension.extends, existing);

    // Track by package
    const pkgExts = extensionsByPackage.get(packageId) || [];
    if (!pkgExts.includes(fullId)) {
      pkgExts.push(fullId);
      extensionsByPackage.set(packageId, pkgExts);
    }

    registryVersion++;
    logger.debug(`Registered extension: ${fullId} for ${extension.extends}`);

    return true;
  } catch (error) {
    logger.error(`Failed to register ${fullId}`, { error });
    return false;
  }
}

/**
 * Compile and register a node extension
 */
export async function compileAndRegisterExtension(
  packageId: string,
  extension: NodeExtension
): Promise<{ success: boolean; error?: string }> {
  // SECURITY: ignore any pre-supplied `compiled` blob from an imported package
  // and always recompile from `source`, so the esbuild import-hardening applies
  // instead of running unsandboxed pre-compiled JS via new Function.

  const compiled: { compilerHook?: string; runtimeHook?: string; ui?: string } = {};

  try {
    // Compile compiler hook
    if (extension.source.compilerHook) {
      const fakeNode = {
        id: extension.id,
        source: {
          compiler: extension.source.compilerHook,
          runtime: 'export function execute() {}', // Placeholder
        },
      };
      const result = await compileCustomNode(fakeNode as any);
      if (result.success && result.compiled) {
        compiled.compilerHook = result.compiled.compiler;
      } else {
        return { success: false, error: `Compiler hook: ${result.error}` };
      }
    }

    // Compile runtime hook
    if (extension.source.runtimeHook) {
      const fakeNode = {
        id: extension.id,
        source: {
          compiler: 'export function compile() {}', // Placeholder
          runtime: extension.source.runtimeHook,
        },
      };
      const result = await compileCustomNode(fakeNode as any);
      if (result.success && result.compiled) {
        compiled.runtimeHook = result.compiled.runtime;
      } else {
        return { success: false, error: `Runtime hook: ${result.error}` };
      }
    }

    // Compile UI
    if (extension.source.ui) {
      const fakeNode = {
        id: extension.id,
        source: {
          compiler: 'export function compile() {}',
          runtime: 'export function execute() {}',
          ui: extension.source.ui,
        },
      };
      const result = await compileCustomNode(fakeNode as any);
      if (result.success && result.compiled?.ui) {
        compiled.ui = result.compiled.ui;
      } else {
        return { success: false, error: `UI: ${result.error}` };
      }
    }

    // Update extension with compiled code
    extension.compiled = compiled;

    // Register
    const success = registerExtension(packageId, extension, compiled);
    return { success, error: success ? undefined : 'Failed to register compiled extension' };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

/**
 * Unregister all extensions for a package
 */
export function unregisterPackageExtensions(packageId: string): void {
  const extIds = extensionsByPackage.get(packageId) || [];

  for (const extId of extIds) {
    const registered = extensionsById.get(extId);
    if (registered) {
      // Remove from node type map
      const nodeExts = extensionsByNodeType.get(registered.extension.extends) || [];
      const filtered = nodeExts.filter(e => getExtensionId(e.packageId, e.extension.id) !== extId);
      if (filtered.length > 0) {
        extensionsByNodeType.set(registered.extension.extends, filtered);
      } else {
        extensionsByNodeType.delete(registered.extension.extends);
      }

      // Remove from ID map
      extensionsById.delete(extId);
      logger.debug(`Unregistered: ${extId}`);
    }
  }

  extensionsByPackage.delete(packageId);
  registryVersion++;
}

/**
 * Get extensions for a node type
 */
export function getExtensionsForNodeType(nodeType: string): RegisteredExtension[] {
  return extensionsByNodeType.get(nodeType) || [];
}

/**
 * Get extended node definition (base + extensions)
 */
export function getExtendedNodeDefinition(
  baseDefinition: NodeDefinition
): NodeDefinition {
  const extensions = getExtensionsForNodeType(baseDefinition.id);

  if (extensions.length === 0) {
    return baseDefinition;
  }

  // Clone the base definition
  const extended: NodeDefinition = {
    ...baseDefinition,
    inputs: [...(baseDefinition.inputs || [])],
    outputs: [...(baseDefinition.outputs || [])],
    properties: [...(baseDefinition.properties || [])],
  };

  // Apply each extension
  for (const { extension } of extensions) {
    // Add additional inputs
    if (extension.additionalInputs) {
      for (const input of extension.additionalInputs) {
        extended.inputs!.push({
          id: input.id,
          name: input.name,
          type: input.type as 'string' | 'number' | 'boolean' | 'any',
          required: input.required ?? false,
        });
      }
    }

    // Add additional outputs
    if (extension.additionalOutputs) {
      for (const output of extension.additionalOutputs) {
        extended.outputs!.push({
          id: output.id,
          name: output.name,
          type: output.type as 'string' | 'number' | 'boolean' | 'any',
        });
      }
    }

    // Add additional properties
    if (extension.additionalProperties) {
      for (const prop of extension.additionalProperties) {
        // Map extension property types to PropertyType (code -> text, textarea is already valid)
        const mappedType = prop.type === 'code' ? 'text' : prop.type;
        extended.properties!.push({
          id: prop.id,
          name: prop.name,
          type: mappedType as PropertyType,
          default: prop.defaultValue,
          options: prop.options,
          group: prop.group,
        });
      }
    }
  }

  return extended;
}

/**
 * Apply compiler hooks for a node
 */
export function applyCompilerHooks(
  nodeType: string,
  code: string,
  node: unknown,
  context: unknown
): string {
  const extensions = getExtensionsForNodeType(nodeType);
  let result = code;

  for (const { compilerHook } of extensions) {
    if (compilerHook?.postCompile) {
      try {
        result = compilerHook.postCompile(result, node, context);
      } catch (error) {
        logger.error('Compiler hook error', { error });
      }
    }
  }

  return result;
}

/**
 * Apply runtime hooks for a node
 */
export async function applyRuntimeHooks(
  nodeType: string,
  inputs: Record<string, unknown>,
  context: RuntimeHookContext,
  baseExecute: (inputs: Record<string, unknown>) => Promise<unknown>
): Promise<unknown> {
  const extensions = getExtensionsForNodeType(nodeType);

  let currentInputs = inputs;

  // Apply preExecute hooks
  for (const { runtimeHook } of extensions) {
    if (runtimeHook?.preExecute) {
      try {
        currentInputs = runtimeHook.preExecute(currentInputs, context);
      } catch (error) {
        logger.error('PreExecute hook error', { error });
      }
    }
  }

  // Check for full override
  for (const { runtimeHook } of extensions) {
    if (runtimeHook?.execute) {
      try {
        return await runtimeHook.execute(currentInputs, context, baseExecute);
      } catch (error) {
        logger.error('Execute hook error', { error });
        throw error;
      }
    }
  }

  // Execute base
  let result = await baseExecute(currentInputs);

  // Apply postExecute hooks
  for (const { runtimeHook } of extensions) {
    if (runtimeHook?.postExecute) {
      try {
        result = runtimeHook.postExecute(result, currentInputs, context);
      } catch (error) {
        logger.error('PostExecute hook error', { error });
      }
    }
  }

  return result;
}

/**
 * Check if a node type has extensions
 */
export function hasExtensions(nodeType: string): boolean {
  return (extensionsByNodeType.get(nodeType)?.length ?? 0) > 0;
}

/**
 * Get registry version (for cache invalidation)
 */
export function getExtensionRegistryVersion(): number {
  return registryVersion;
}

/**
 * Clear all extensions (for testing)
 */
export function clearExtensionRegistry(): void {
  extensionsByNodeType.clear();
  extensionsById.clear();
  extensionsByPackage.clear();
  registryVersion++;
}
