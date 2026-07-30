/**
 * Custom Node Registry
 *
 * Manages registration and lookup of custom nodes defined in packages.
 * Handles loading compiled code and providing compiler/runtime functions.
 */

import type {
  CustomNodeDefinition,
  CustomNodeCompiler,
  CustomNodeRuntime,
  CustomNodeCompilerContext,
  CustomNodeRuntimeContext,
  CustomNodeCompilerResult,
} from './custom-node-types';

/**
 * Registered custom node with loaded code
 */
export interface RegisteredCustomNode {
  /** The node definition */
  definition: CustomNodeDefinition;

  /** Package ID that provides this node */
  packageId: string;

  /** Fully qualified node type (packageId:nodeId) */
  fullType: string;

  /** Compiled compiler function */
  compilerFn?: CustomNodeCompiler;

  /** Compiled runtime function */
  runtimeFn?: CustomNodeRuntime;

  /** Whether the compiled code has been loaded */
  loaded: boolean;

  /** Error if loading failed */
  loadError?: string;
}

/**
 * Options for the custom node registry
 */
export interface CustomNodeRegistryOptions {
  /** Whether to allow unsafe code execution (for development) */
  allowUnsafeCode?: boolean;

  /** Timeout for runtime execution (ms) */
  runtimeTimeout?: number;

  /** Custom require function for module loading */
  requireFn?: (moduleId: string) => unknown;
}

/**
 * Registry for custom nodes
 */
export class CustomNodeRegistry {
  private nodes: Map<string, RegisteredCustomNode> = new Map();
  private options: CustomNodeRegistryOptions;

  constructor(options: CustomNodeRegistryOptions = {}) {
    this.options = {
      allowUnsafeCode: false,
      runtimeTimeout: 60000,
      ...options,
    };
  }

  /**
   * Register a custom node definition
   */
  registerNode(packageId: string, definition: CustomNodeDefinition): void {
    const fullType = `${packageId}:${definition.id}`;

    const registered: RegisteredCustomNode = {
      definition,
      packageId,
      fullType,
      loaded: false,
    };

    this.nodes.set(fullType, registered);

    // Also register without prefix for convenience
    if (!this.nodes.has(definition.id)) {
      this.nodes.set(definition.id, registered);
    }
  }

  /**
   * Unregister all nodes from a package
   */
  unregisterPackage(packageId: string): void {
    const toRemove: string[] = [];

    for (const [key, node] of this.nodes) {
      if (node.packageId === packageId) {
        toRemove.push(key);
      }
    }

    for (const key of toRemove) {
      this.nodes.delete(key);
    }
  }

  /**
   * Get a registered node definition
   */
  getNode(nodeType: string): RegisteredCustomNode | undefined {
    return this.nodes.get(nodeType);
  }

  /**
   * Get all registered nodes
   */
  getAllNodes(): RegisteredCustomNode[] {
    // Return unique nodes (avoid duplicates from prefix/non-prefix registration)
    const seen = new Set<string>();
    const result: RegisteredCustomNode[] = [];

    for (const node of this.nodes.values()) {
      if (!seen.has(node.fullType)) {
        seen.add(node.fullType);
        result.push(node);
      }
    }

    return result;
  }

  /**
   * Get all nodes from a specific package
   */
  getPackageNodes(packageId: string): RegisteredCustomNode[] {
    return this.getAllNodes().filter(n => n.packageId === packageId);
  }

  /**
   * Check if a node type is a custom node
   */
  isCustomNode(nodeType: string): boolean {
    return this.nodes.has(nodeType);
  }

  /**
   * Load the compiled code for a node
   */
  async loadNode(nodeType: string): Promise<boolean> {
    const node = this.nodes.get(nodeType);
    if (!node) {
      return false;
    }

    if (node.loaded) {
      return true;
    }

    const { definition } = node;

    if (!definition.compiled) {
      node.loadError = 'Node has not been compiled';
      return false;
    }

    try {
      // Load compiler function
      if (definition.compiled.compiler) {
        node.compilerFn = this.loadCompilerCode(definition.compiled.compiler);
      }

      // Load runtime function
      if (definition.compiled.runtime) {
        node.runtimeFn = this.loadRuntimeCode(definition.compiled.runtime);
      }

      node.loaded = true;
      return true;
    } catch (err) {
      node.loadError = err instanceof Error ? err.message : String(err);
      return false;
    }
  }

  /**
   * Load compiler code from compiled JavaScript
   */
  private loadCompilerCode(code: string): CustomNodeCompiler {
    // Create a sandboxed function from the compiled code
    const exports: Record<string, unknown> = {};
    const module = { exports };

    // Basic require function for common modules
    const requireFn = this.options.requireFn || ((moduleId: string) => {
      if (moduleId === '@oaiy/core') {
        // Provide core types and utilities
        return {
          // Add any core utilities needed by compilers
        };
      }
      throw new Error(`Cannot require module: ${moduleId}`);
    });

    try {
      // new Function is required to execute dynamically compiled custom node code.
      // SECURITY: this runs package-supplied JS in the HOST realm. Packages are
      // UNTRUSTED in this product's threat model — a caller MUST pre-sandbox (worker
      // realm) before relying on this. NOT "safe because trusted". (This registry is
      // library API; the live web app routes custom nodes through ui/src/services/*.)
      // Enforce the (previously dead) allowUnsafeCode opt-in: refuse host-realm execution
      // unless the caller has explicitly asserted it pre-sandboxed the code.
      if (!this.options.allowUnsafeCode) {
        throw new Error(
          'CustomNodeRegistry: refusing to run package-supplied compiler code in the host realm — ' +
            'packages are untrusted. Pre-sandbox it (worker realm), or set allowUnsafeCode:true.',
        );
      }
      // eslint-disable-next-line no-new-func
      const fn = new Function('exports', 'require', 'module', code);
      fn(exports, requireFn, module);

      // Look for the compiler function
      const moduleExports = module.exports as Record<string, unknown>;
      const defaultExport = moduleExports.default as Record<string, unknown> | undefined;
      const compiler = moduleExports.compile || defaultExport?.compile || defaultExport;

      if (typeof compiler !== 'function') {
        throw new Error('Compiler module must export a "compile" function');
      }

      return compiler as CustomNodeCompiler;
    } catch (err) {
      throw new Error(`Failed to load compiler: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Load runtime code from compiled JavaScript
   */
  private loadRuntimeCode(code: string): CustomNodeRuntime {
    const exports: Record<string, unknown> = {};
    const module = { exports };

    const requireFn = this.options.requireFn || ((moduleId: string) => {
      if (moduleId === '@oaiy/runtime') {
        return {
          // Add any runtime utilities needed
        };
      }
      throw new Error(`Cannot require module: ${moduleId}`);
    });

    try {
      // new Function is required to execute dynamically compiled custom node runtime code.
      // SECURITY: this runs package-supplied JS in the HOST realm. Packages are UNTRUSTED — a
      // caller MUST pre-sandbox (worker realm) before relying on this. NOT "safe because trusted".
      if (!this.options.allowUnsafeCode) {
        throw new Error(
          'CustomNodeRegistry: refusing to run package-supplied runtime code in the host realm — ' +
            'packages are untrusted. Pre-sandbox it (worker realm), or set allowUnsafeCode:true.',
        );
      }
      // eslint-disable-next-line no-new-func
      const fn = new Function('exports', 'require', 'module', code);
      fn(exports, requireFn, module);

      const moduleExports = module.exports as Record<string, unknown>;
      const defaultExport = moduleExports.default as Record<string, unknown> | undefined;
      const runtime = moduleExports.execute || defaultExport?.execute || defaultExport;

      if (typeof runtime !== 'function') {
        throw new Error('Runtime module must export an "execute" function');
      }

      return runtime as CustomNodeRuntime;
    } catch (err) {
      throw new Error(`Failed to load runtime: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Get the compiler function for a node
   */
  getCompiler(nodeType: string): CustomNodeCompiler | undefined {
    const node = this.nodes.get(nodeType);
    if (!node) return undefined;

    // Try to load if not already loaded
    if (!node.loaded && !node.loadError) {
      this.loadNode(nodeType).catch(() => {});
    }

    return node.compilerFn;
  }

  /**
   * Get the runtime function for a node
   */
  getRuntime(nodeType: string): CustomNodeRuntime | undefined {
    const node = this.nodes.get(nodeType);
    if (!node) return undefined;

    if (!node.loaded && !node.loadError) {
      this.loadNode(nodeType).catch(() => {});
    }

    return node.runtimeFn;
  }

  /**
   * Compile a custom node
   */
  compileNode(
    nodeType: string,
    context: CustomNodeCompilerContext
  ): CustomNodeCompilerResult {
    const compiler = this.getCompiler(nodeType);

    if (!compiler) {
      return {
        code: `// Error: No compiler for custom node type "${nodeType}"`,
        success: false,
        error: `No compiler available for node type: ${nodeType}`,
      };
    }

    try {
      return compiler(context);
    } catch (err) {
      return {
        code: `// Error compiling custom node: ${err instanceof Error ? err.message : String(err)}`,
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /**
   * Execute a custom node at runtime
   */
  async executeNode(
    nodeType: string,
    inputs: Record<string, unknown>,
    properties: Record<string, unknown>,
    context: CustomNodeRuntimeContext
  ): Promise<Record<string, unknown>> {
    const runtime = this.getRuntime(nodeType);

    if (!runtime) {
      throw new Error(`No runtime available for node type: ${nodeType}`);
    }

    // Execute with timeout
    const timeout = this.options.runtimeTimeout || 60000;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Custom node execution timed out after ${timeout}ms`));
      }, timeout);

      runtime(inputs, properties, context)
        .then(result => {
          clearTimeout(timer);
          resolve(result);
        })
        .catch(err => {
          clearTimeout(timer);
          reject(err);
        });
    });
  }

  /**
   * Clear all registered nodes
   */
  clear(): void {
    this.nodes.clear();
  }

  /**
   * Get registry statistics
   */
  getStats(): {
    totalNodes: number;
    loadedNodes: number;
    failedNodes: number;
    packageCount: number;
  } {
    const allNodes = this.getAllNodes();
    const packages = new Set(allNodes.map(n => n.packageId));

    return {
      totalNodes: allNodes.length,
      loadedNodes: allNodes.filter(n => n.loaded).length,
      failedNodes: allNodes.filter(n => !!n.loadError).length,
      packageCount: packages.size,
    };
  }
}

// Singleton instance
let globalRegistry: CustomNodeRegistry | null = null;

/**
 * Get the global custom node registry
 */
export function getCustomNodeRegistry(): CustomNodeRegistry {
  if (!globalRegistry) {
    globalRegistry = new CustomNodeRegistry();
  }
  return globalRegistry;
}

/**
 * Reset the global custom node registry
 */
export function resetCustomNodeRegistry(): void {
  if (globalRegistry) {
    globalRegistry.clear();
  }
  globalRegistry = null;
}

/**
 * Create a new custom node registry with options
 */
export function createCustomNodeRegistry(options?: CustomNodeRegistryOptions): CustomNodeRegistry {
  return new CustomNodeRegistry(options);
}
