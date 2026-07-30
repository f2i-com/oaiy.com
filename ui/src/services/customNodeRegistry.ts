/**
 * Custom Node Registry
 *
 * Manages compiled custom nodes from .oaiy packages.
 * Provides access to compiler, runtime, and UI components.
 * Integrates with ModuleLoader for flow compilation support.
 */

import type { EmbeddedCustomNode, NodeDefinition, ModuleCompiler, ModuleCompilerContext, ModuleManifest, ModuleCategory, PropertyType } from 'oaiy-core';
import { getModuleLoader } from 'oaiy-core';
import type React from 'react';
import { compileCustomNode, type CustomNodeCompileResult } from './customNodeCompiler';
import { registerCompiler, compileInSandbox, unregisterCompiler } from './compilerSandbox';
import { createLogger } from '../utils/logger';

const logger = createLogger('CustomNodeRegistry');

// ============================================
// Types
// ============================================

export interface CustomNodeCompilerContext {
  node: { id: string; type: string; data: Record<string, unknown> };
  inputs: Map<string, string>;
  outputVar: string;
  sanitizedId: string;
  escapeString: (str: string) => string;
  skipVarDeclaration?: boolean;
}

export type CustomNodeCompilerFunction = (ctx: CustomNodeCompilerContext) => string;

export interface CustomNodeRuntimeContext {
  log: (level: string, message: string) => void;
  getInput: (id: string) => unknown;
  setOutput: (id: string, value: unknown) => void;
  nodeId: string;
  nodeData: Record<string, unknown>;
}

export type CustomNodeRuntimeFunction = (
  inputs: Record<string, unknown>,
  ctx: CustomNodeRuntimeContext
) => Promise<unknown> | unknown;

export interface RegisteredCustomNode {
  definition: EmbeddedCustomNode;
  nodeDefinition: NodeDefinition;
  packageId: string;
  compiled: {
    compiler: string;
    runtime: string;
    ui?: string;
  };
  /** True once the package's compiler bundle is registered in the sandbox worker. */
  hasCompiler?: boolean;
  runtimeFn?: CustomNodeRuntimeFunction;
  uiComponent?: React.ComponentType<unknown>;
}

// ============================================
// Registry State
// ============================================

// Map from full node type (pkg:{packageId}:{nodeId}) to registered node
const registeredNodes = new Map<string, RegisteredCustomNode>();

// Map from package ID to list of node types
const packageNodes = new Map<string, string[]>();

// Version counter for cache invalidation
let registryVersion = 0;

// ============================================
// Helper Functions
// ============================================

/**
 * Get the full node type for a custom node
 */
export function getFullNodeType(packageId: string, nodeId: string): string {
  return `pkg:${packageId}:${nodeId}`;
}

/**
 * Parse a full node type to get package ID and node ID
 */
export function parseFullNodeType(fullNodeType: string): { packageId: string; nodeId: string } | null {
  const match = fullNodeType.match(/^pkg:([^:]+):(.+)$/);
  if (!match) return null;
  return { packageId: match[1], nodeId: match[2] };
}

/**
 * Convert EmbeddedCustomNode to NodeDefinition
 */
function createNodeDefinition(node: EmbeddedCustomNode, packageId: string): NodeDefinition {
  // Convert embedded node properties to NodeDefinition properties
  const properties = (node.properties || []).map(prop => ({
    id: prop.id,
    name: prop.name,
    type: prop.type as PropertyType,
    default: prop.default,
    options: prop.options,
    min: prop.min,
    max: prop.max,
    step: prop.step,
    advanced: prop.advanced,
    group: prop.group,
  }));

  return {
    id: getFullNodeType(packageId, node.id),
    name: node.name,
    description: node.description,
    icon: node.icon,
    inputs: node.inputs.map(input => ({
      id: input.id,
      name: input.name,
      type: input.type as 'string' | 'number' | 'boolean' | 'any',
      required: input.required ?? false,
      position: 'left' as const,
    })),
    outputs: node.outputs.map(output => ({
      id: output.id,
      name: output.name,
      type: output.type as 'string' | 'number' | 'boolean' | 'any',
      position: 'right' as const,
    })),
    properties,
    // Custom nodes use a custom handler for compilation
    compiler: {
      customHandler: true,
    },
  };
}

// (Removed loadCompiledCode: it ran the UNTRUSTED package compiler bundle via `new Function`
// in the MAIN app realm. The compiler now runs in the sandbox worker — see registerCustomNode
// + compilerSandbox + the deferred-compilation path. No untrusted package code executes in
// the app realm: UI → SandboxedCustomNodeUI iframe, compiler → compiler-sandbox-worker.)

// ============================================
// Registry Functions
// ============================================

/**
 * Register a compiled custom node
 */
export async function registerCustomNode(
  packageId: string,
  node: EmbeddedCustomNode,
  compiled: { compiler: string; runtime: string; ui?: string }
): Promise<boolean> {
  const fullType = getFullNodeType(packageId, node.id);

  try {
    // Create node definition
    const nodeDefinition = createNodeDefinition(node, packageId);

    // Create registered node entry
    const registered: RegisteredCustomNode = {
      definition: node,
      nodeDefinition,
      packageId,
      compiled,
    };

    // Register the package's UNTRUSTED compiler bundle into the sandbox WORKER (an
    // isolated Worker realm — no window/localStorage/DOM/__TAURI_INTERNALS__, egress
    // neutralized) instead of running it via `new Function` in the main app realm. The
    // moduleCompiler below defers each node's codegen to it. Codegen is byte-identical:
    // the worker imports the same escapeString util OAIYCompiler uses.
    const compilerGlobalName = `__CUSTOM_NODE_COMPILER_${node.id.replace(/[^a-zA-Z0-9]/g, '_')}__`;
    if (compiled.compiler) {
      void registerCompiler(fullType, compiled.compiler, compilerGlobalName);
      registered.hasCompiler = true;
      logger.debug(`Registered sandboxed compiler for ${fullType}`);
    }

    // NOTE: the package "runtime" bundle is intentionally NOT executed here — custom nodes
    // run via the COMPILED codegen path (sandboxed compiler → moduleLoader → worker-isolated
    // runtime), not an interpreted main-realm runtime fn (getCustomNodeRuntime has no
    // callers). UI is handled separately by SandboxedCustomNodeUI. No untrusted package
    // code runs in the main app realm anymore.

    // UI component loading is handled separately by the UI layer

    // Register the node
    registeredNodes.set(fullType, registered);

    // Track by package
    const pkgNodes = packageNodes.get(packageId) || [];
    if (!pkgNodes.includes(fullType)) {
      pkgNodes.push(fullType);
      packageNodes.set(packageId, pkgNodes);
    }

    // Register with ModuleLoader for flow compilation support
    try {
      const moduleLoader = getModuleLoader();

      // Create a module compiler that wraps the custom node's compile function
      const moduleCompiler: ModuleCompiler = {
        name: `CustomNode:${fullType}`,
        getNodeTypes: () => [fullType],
        compileNode: (nodeType: string, ctx: ModuleCompilerContext): string | null => {
          if (nodeType !== fullType) return null;
          // Honor skipVarDeclaration: inside a loop the output var is PRE-declared at outer
          // scope, so a `let` here would shadow it and the assigned value would be invisible
          // after the loop (silent null). Emit a bare assignment in that case.
          const decl = ctx.skipVarDeclaration ? '' : 'let ';
          if (!registered.hasCompiler) {
            // Fallback: generate a simple pass-through if no compiler (no untrusted code).
            const inputVar = ctx.inputs.get('input') || ctx.inputs.values().next().value || 'null';
            return `
  // --- Custom Node: ${ctx.node.id} (${nodeType}) ---
  ${decl}${ctx.outputVar} = ${inputVar};
  workflow_context["${ctx.node.id}"] = ${ctx.outputVar};`;
          }

          // Defer to the sandbox worker: the UNTRUSTED package compiler runs off-realm and
          // its output is substituted into the script by OAIYCompiler.resolveDeferred. The
          // payload is the CustomNodeCompilerContext minus escapeString (the worker supplies
          // the same util). If the host engine predates ctx.defer, fail loud rather than
          // silently fall back to a main-realm exec.
          if (!ctx.defer) {
            logger.error(`Engine lacks ctx.defer; cannot sandbox compiler for ${fullType}`);
            return `/* ${ctx.node.id}: compiler sandbox unavailable */ ${decl}${ctx.outputVar} = null;`;
          }
          const payload = {
            node: ctx.node,
            inputs: ctx.inputs,
            outputVar: ctx.outputVar,
            sanitizedId: ctx.sanitizedId,
            skipVarDeclaration: ctx.skipVarDeclaration,
          };
          const nodeId = ctx.node.id;
          const outVar = ctx.outputVar;
          return ctx.defer(async () => {
            try {
              const code = await compileInSandbox(fullType, payload);
              // Defense-in-depth: the placeholder is salted+unguessable, but also refuse
              // output that even references the reserved token (a compiler probing it).
              if (code.includes('__OAIY_DEFER_')) {
                throw new Error('compiler output references a reserved placeholder token');
              }
              return code;
            } catch (e) {
              // A broken/throwing compiler must not fail the WHOLE flow (mirrors the old
              // catch-and-continue in OAIYCompiler.tryModuleCompiler). Declare the output as
              // null so downstream var references stay valid.
              logger.error(`Sandboxed compiler failed for ${fullType}`, { error: e });
              return `\n  // --- Custom Node ${nodeId} (${fullType}): compiler failed ---\n  ${decl}${outVar} = null;\n  workflow_context[${JSON.stringify(nodeId)}] = ${outVar};`;
            }
          });
        },
      };

      // Create a module manifest for this custom node
      const moduleManifest: ModuleManifest = {
        id: `custom-node-${packageId}-${node.id}`.toLowerCase().replace(/[^a-z0-9-]/g, '-'),
        name: node.name,
        version: '1.0.0',
        description: node.description || `Custom node: ${node.name}`,
        category: (node.category as ModuleCategory) || 'Custom',
        author: 'Package',
        nodes: [fullType],
      };

      // Load the module with the custom compiler
      const result = await moduleLoader.loadModule(
        moduleManifest,
        [nodeDefinition],
        undefined, // No runtime module (handled separately)
        `custom-node:${packageId}/${node.id}`,
        moduleCompiler
      );
      if (result.success) {
        logger.debug(`Registered with ModuleLoader: ${fullType}`);
      } else {
        logger.warn(`ModuleLoader registration failed`, { error: result.error });
      }
    } catch (error) {
      logger.warn('Failed to register with ModuleLoader', { error });
    }

    registryVersion++;
    logger.debug(`Registered custom node: ${fullType}`);

    return true;
  } catch (error) {
    logger.error(`Failed to register ${fullType}`, { error });
    return false;
  }
}

/**
 * Compile and register a custom node
 */
export async function compileAndRegisterCustomNode(
  packageId: string,
  node: EmbeddedCustomNode
): Promise<CustomNodeCompileResult> {
  // SECURITY: never trust a pre-supplied `compiled` blob from an imported
  // package — it would run unsandboxed via new Function and bypass the
  // esbuild import-hardening (BLOCKED_TAURI_IMPORTS, no host globals). Always
  // recompile from `source` so the hardening actually applies. (A package that
  // ships compiled-without-source simply fails to register, which is correct.)

  // Compile the node
  const result = await compileCustomNode(node);

  if (result.success && result.compiled) {
    // Update the node with compiled code
    node.compiled = result.compiled;

    // Register the compiled node
    const success = await registerCustomNode(packageId, node, result.compiled);
    if (!success) {
      return {
        success: false,
        nodeId: node.id,
        error: 'Compiled but failed to register',
      };
    }
  }

  return result;
}

/**
 * Unregister all nodes for a package
 */
export function unregisterPackageNodes(packageId: string): void {
  const nodeTypes = packageNodes.get(packageId) || [];

  for (const nodeType of nodeTypes) {
    registeredNodes.delete(nodeType);
    unregisterCompiler(nodeType);
    logger.debug(`Unregistered: ${nodeType}`);
  }

  packageNodes.delete(packageId);
  registryVersion++;
}

/**
 * Get a registered custom node
 */
export function getCustomNode(fullNodeType: string): RegisteredCustomNode | undefined {
  return registeredNodes.get(fullNodeType);
}

/**
 * Get the node definition for a custom node
 */
export function getCustomNodeDefinition(fullNodeType: string): NodeDefinition | undefined {
  return registeredNodes.get(fullNodeType)?.nodeDefinition;
}

// (Removed getCustomNodeCompiler + getCustomNodeRuntime: both had no callers. The compiler
// is no longer a main-realm function — it runs in the sandbox worker — and the runtime fn
// was never executed. Custom nodes compile via the sandboxed compiler + moduleLoader path.)

/**
 * Get all registered custom node types
 */
export function getRegisteredCustomNodeTypes(): string[] {
  return Array.from(registeredNodes.keys());
}

/**
 * Get all custom node definitions
 */
export function getAllCustomNodeDefinitions(): NodeDefinition[] {
  return Array.from(registeredNodes.values()).map(n => n.nodeDefinition);
}

/**
 * Get custom nodes for a package
 */
export function getPackageCustomNodes(packageId: string): RegisteredCustomNode[] {
  const nodeTypes = packageNodes.get(packageId) || [];
  return nodeTypes.map(t => registeredNodes.get(t)!).filter(Boolean);
}

/**
 * Check if a node type is a custom node
 */
export function isCustomNode(nodeType: string): boolean {
  return registeredNodes.has(nodeType);
}

/**
 * Get current registry version (for cache invalidation)
 */
export function getCustomNodeRegistryVersion(): number {
  return registryVersion;
}

/**
 * Clear all registered nodes (for testing)
 */
export function clearCustomNodeRegistry(): void {
  registeredNodes.clear();
  packageNodes.clear();
  registryVersion++;
}
