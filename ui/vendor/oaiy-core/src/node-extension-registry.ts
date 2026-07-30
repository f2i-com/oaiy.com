/**
 * Node Extension Registry
 *
 * Manages registration and application of node extensions.
 * Handles loading compiled hooks and applying them during compilation and runtime.
 */

import type { GraphNode, WorkflowGraph } from './types';
import type { NodeDefinition, HandleDefinition, PropertyDefinition } from './module-types';
import type {
  NodeExtension,
  LoadedExtension,
  CompilerHook,
  RuntimeHook,
  CompilerHookContext,
  RuntimeHookContext,
  ExtendedCompilationResult,
  ExtendedRuntimeResult,
  extensionConditionMatches,
} from './node-extension-types';

/**
 * Options for the extension registry
 */
export interface ExtensionRegistryOptions {
  /** Whether to allow unsafe code execution */
  allowUnsafeCode?: boolean;
  /** Custom require function */
  requireFn?: (moduleId: string) => unknown;
}

/**
 * Registry for node extensions
 */
export class NodeExtensionRegistry {
  private extensions: Map<string, LoadedExtension[]> = new Map();
  private options: ExtensionRegistryOptions;

  constructor(options: ExtensionRegistryOptions = {}) {
    this.options = {
      allowUnsafeCode: false,
      ...options,
    };
  }

  /**
   * Register an extension
   */
  registerExtension(extension: NodeExtension): void {
    const loaded: LoadedExtension = {
      ...extension,
      loaded: false,
    };

    const existing = this.extensions.get(extension.extends) || [];
    existing.push(loaded);

    // Sort by priority (higher first)
    existing.sort((a, b) => (b.priority || 0) - (a.priority || 0));

    this.extensions.set(extension.extends, existing);
  }

  /**
   * Unregister all extensions from a package
   */
  unregisterPackage(packageId: string): void {
    for (const [nodeType, extensions] of this.extensions) {
      const filtered = extensions.filter(e => e.packageId !== packageId);
      if (filtered.length > 0) {
        this.extensions.set(nodeType, filtered);
      } else {
        this.extensions.delete(nodeType);
      }
    }
  }

  /**
   * Get extensions for a node type
   */
  getExtensionsFor(nodeType: string): LoadedExtension[] {
    return this.extensions.get(nodeType) || [];
  }

  /**
   * Get extensions that match a specific node
   */
  getMatchingExtensions(node: GraphNode): LoadedExtension[] {
    const nodeType = node.type || '';
    const extensions = this.getExtensionsFor(nodeType);

    return extensions.filter(ext => {
      // Check if condition matches
      if (ext.condition) {
        return this.checkCondition(ext, node);
      }
      return ext.enabledByDefault !== false;
    });
  }

  /**
   * Check if an extension condition matches a node
   */
  private checkCondition(extension: LoadedExtension, node: GraphNode): boolean {
    if (!extension.condition) return true;

    const { property, value, operator = 'equals' } = extension.condition;
    const nodeValue = node.data?.[property];

    switch (operator) {
      case 'equals':
        if (Array.isArray(value)) {
          return value.includes(nodeValue);
        }
        return nodeValue === value;

      case 'notEquals':
        if (Array.isArray(value)) {
          return !value.includes(nodeValue);
        }
        return nodeValue !== value;

      case 'contains':
        if (typeof nodeValue === 'string' && typeof value === 'string') {
          return nodeValue.includes(value);
        }
        return false;

      case 'exists':
        return nodeValue !== undefined && nodeValue !== null;

      default:
        return true;
    }
  }

  /**
   * Load an extension's hooks
   */
  async loadExtension(extension: LoadedExtension): Promise<boolean> {
    if (extension.loaded) return true;

    try {
      // Load compiler hook
      if (extension.compilerHook?.compiled) {
        extension.loadedCompilerHook = this.loadCompilerHook(extension.compilerHook.compiled);
      }

      // Load runtime hook
      if (extension.runtimeHook?.compiled) {
        extension.loadedRuntimeHook = this.loadRuntimeHook(extension.runtimeHook.compiled);
      }

      extension.loaded = true;
      return true;
    } catch (err) {
      extension.loadError = err instanceof Error ? err.message : String(err);
      return false;
    }
  }

  /**
   * Load compiler hook from compiled JavaScript
   */
  private loadCompilerHook(code: string): CompilerHook {
    const exports: Record<string, unknown> = {};
    const module = { exports };

    const requireFn = this.options.requireFn || ((moduleId: string) => {
      if (moduleId === '@oaiy/core') {
        return {};
      }
      throw new Error(`Cannot require module: ${moduleId}`);
    });

    try {
      // new Function is required to execute dynamically compiled extension code.
      // SECURITY: this runs package-supplied JS in the HOST realm. Packages are
      // UNTRUSTED — a caller MUST pre-sandbox (worker realm). NOT "safe because
      // trusted". (Library API; the live app routes through ui/src/services/*.)
      // Enforce the (previously dead) allowUnsafeCode opt-in before host-realm execution.
      if (!this.options.allowUnsafeCode) {
        throw new Error(
          'NodeExtensionRegistry: refusing to run package-supplied compiler-hook code in the host ' +
            'realm — packages are untrusted. Pre-sandbox it (worker realm), or set allowUnsafeCode:true.',
        );
      }
      // eslint-disable-next-line no-new-func
      const fn = new Function('exports', 'require', 'module', code);
      fn(exports, requireFn, module);

      const hook = module.exports as CompilerHook;
      return hook;
    } catch (err) {
      throw new Error(`Failed to load compiler hook: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Load runtime hook from compiled JavaScript
   */
  private loadRuntimeHook(code: string): RuntimeHook {
    const exports: Record<string, unknown> = {};
    const module = { exports };

    const requireFn = this.options.requireFn || ((moduleId: string) => {
      if (moduleId === '@oaiy/runtime') {
        return {};
      }
      throw new Error(`Cannot require module: ${moduleId}`);
    });

    try {
      // new Function is required to execute dynamically compiled extension code.
      // SECURITY: this runs package-supplied JS in the HOST realm. Packages are
      // UNTRUSTED — a caller MUST pre-sandbox (worker realm). NOT "safe because
      // trusted". (Library API; the live app routes through ui/src/services/*.)
      // Enforce the (previously dead) allowUnsafeCode opt-in before host-realm execution.
      if (!this.options.allowUnsafeCode) {
        throw new Error(
          'NodeExtensionRegistry: refusing to run package-supplied runtime-hook code in the host ' +
            'realm — packages are untrusted. Pre-sandbox it (worker realm), or set allowUnsafeCode:true.',
        );
      }
      // eslint-disable-next-line no-new-func
      const fn = new Function('exports', 'require', 'module', code);
      fn(exports, requireFn, module);

      const hook = module.exports as RuntimeHook;
      return hook;
    } catch (err) {
      throw new Error(`Failed to load runtime hook: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Get extended node definition with additional inputs/outputs/properties
   */
  getExtendedNodeDefinition(
    baseDef: NodeDefinition,
    node: GraphNode
  ): NodeDefinition {
    const extensions = this.getMatchingExtensions(node);

    if (extensions.length === 0) {
      return baseDef;
    }

    // Clone base definition
    const extended: NodeDefinition = {
      ...baseDef,
      inputs: [...(baseDef.inputs || [])],
      outputs: [...(baseDef.outputs || [])],
      properties: [...(baseDef.properties || [])],
    };

    // Apply each extension's additions
    for (const ext of extensions) {
      // Add inputs
      if (ext.additionalInputs) {
        for (const input of ext.additionalInputs) {
          const handle: HandleDefinition = {
            id: input.id,
            name: input.name,
            type: (input.type as HandleDefinition['type']) || 'any',
            required: input.required,
            description: input.description,
          };
          extended.inputs!.push(handle);
        }
      }

      // Add outputs
      if (ext.additionalOutputs) {
        for (const output of ext.additionalOutputs) {
          const handle: HandleDefinition = {
            id: output.id,
            name: output.name,
            type: (output.type as HandleDefinition['type']) || 'any',
            description: output.description,
          };
          extended.outputs!.push(handle);
        }
      }

      // Add properties
      if (ext.additionalProperties) {
        for (const prop of ext.additionalProperties) {
          const propDef: PropertyDefinition = {
            id: prop.id,
            name: prop.name,
            type: prop.type as PropertyDefinition['type'],
            default: prop.defaultValue,
            description: prop.description,
            options: prop.options?.map(o => ({ value: o.value, label: o.label })),
          };
          extended.properties!.push(propDef);
        }
      }
    }

    return extended;
  }

  /**
   * Apply compiler hooks to code generation
   */
  async applyCompilerHooks(
    node: GraphNode,
    context: CompilerHookContext,
    baseCompile: () => string
  ): Promise<ExtendedCompilationResult> {
    const extensions = this.getMatchingExtensions(node);
    const warnings: string[] = [];
    const errors: string[] = [];
    let extensionsApplied = 0;

    // Load extensions if needed
    for (const ext of extensions) {
      if (!ext.loaded) {
        await this.loadExtension(ext);
      }
    }

    // Apply preCompile hooks
    let processedNode = node;
    for (const ext of extensions) {
      if (ext.loadedCompilerHook?.preCompile) {
        try {
          processedNode = ext.loadedCompilerHook.preCompile(processedNode, context);
          extensionsApplied++;
        } catch (err) {
          errors.push(`Extension ${ext.id} preCompile failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    // Collect injectBefore code
    let beforeCode = '';
    for (const ext of extensions) {
      if (ext.loadedCompilerHook?.injectBefore) {
        try {
          const injected = ext.loadedCompilerHook.injectBefore(processedNode, context);
          if (injected) {
            beforeCode += injected + '\n';
            extensionsApplied++;
          }
        } catch (err) {
          warnings.push(`Extension ${ext.id} injectBefore failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    // Run base compilation
    let code = baseCompile();

    // Apply postCompile hooks
    for (const ext of extensions) {
      if (ext.loadedCompilerHook?.postCompile) {
        try {
          code = ext.loadedCompilerHook.postCompile(code, processedNode, context);
          extensionsApplied++;
        } catch (err) {
          errors.push(`Extension ${ext.id} postCompile failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    // Collect injectAfter code
    let afterCode = '';
    for (const ext of extensions) {
      if (ext.loadedCompilerHook?.injectAfter) {
        try {
          const injected = ext.loadedCompilerHook.injectAfter(processedNode, context);
          if (injected) {
            afterCode += injected + '\n';
            extensionsApplied++;
          }
        } catch (err) {
          warnings.push(`Extension ${ext.id} injectAfter failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    // Combine all code
    const finalCode = beforeCode + code + afterCode;

    return {
      code: finalCode,
      extensionsApplied,
      warnings,
      errors,
    };
  }

  /**
   * Apply runtime hooks to execution
   */
  async applyRuntimeHooks(
    node: GraphNode,
    context: RuntimeHookContext,
    inputs: Record<string, unknown>,
    baseExecute: (inputs: Record<string, unknown>) => Promise<unknown>
  ): Promise<ExtendedRuntimeResult> {
    const extensions = this.getMatchingExtensions(node);
    let extensionsApplied = 0;
    let overridden = false;

    // Load extensions if needed
    for (const ext of extensions) {
      if (!ext.loaded) {
        await this.loadExtension(ext);
      }
    }

    let processedInputs = inputs;

    // Apply preExecute hooks
    for (const ext of extensions) {
      if (ext.loadedRuntimeHook?.preExecute) {
        try {
          processedInputs = await ext.loadedRuntimeHook.preExecute(processedInputs, context);
          extensionsApplied++;
        } catch (err) {
          context.log(`Extension ${ext.id} preExecute failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
        }
      }
    }

    let result: unknown;

    // Check for full execution override
    const overrideExt = extensions.find(ext => ext.loadedRuntimeHook?.execute);
    if (overrideExt?.loadedRuntimeHook?.execute) {
      try {
        result = await overrideExt.loadedRuntimeHook.execute(processedInputs, context, baseExecute);
        overridden = true;
        extensionsApplied++;
      } catch (err) {
        // Check for error handlers
        const errorHandler = extensions.find(ext => ext.loadedRuntimeHook?.onError);
        if (errorHandler?.loadedRuntimeHook?.onError) {
          result = await errorHandler.loadedRuntimeHook.onError(
            err instanceof Error ? err : new Error(String(err)),
            processedInputs,
            context
          );
        } else {
          throw err;
        }
      }
    } else {
      // Run base execution
      try {
        result = await baseExecute(processedInputs);
      } catch (err) {
        // Check for error handlers
        const errorHandler = extensions.find(ext => ext.loadedRuntimeHook?.onError);
        if (errorHandler?.loadedRuntimeHook?.onError) {
          result = await errorHandler.loadedRuntimeHook.onError(
            err instanceof Error ? err : new Error(String(err)),
            processedInputs,
            context
          );
          extensionsApplied++;
        } else {
          throw err;
        }
      }
    }

    // Apply postExecute hooks
    for (const ext of extensions) {
      if (ext.loadedRuntimeHook?.postExecute) {
        try {
          result = await ext.loadedRuntimeHook.postExecute(result, processedInputs, context);
          extensionsApplied++;
        } catch (err) {
          context.log(`Extension ${ext.id} postExecute failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
        }
      }
    }

    return {
      result,
      extensionsApplied,
      overridden,
    };
  }

  /**
   * Check if a node type has any registered extensions
   */
  hasExtensions(nodeType: string): boolean {
    return this.extensions.has(nodeType) && this.extensions.get(nodeType)!.length > 0;
  }

  /**
   * Get all registered extensions
   */
  getAllExtensions(): LoadedExtension[] {
    const all: LoadedExtension[] = [];
    for (const extensions of this.extensions.values()) {
      all.push(...extensions);
    }
    return all;
  }

  /**
   * Clear all registered extensions
   */
  clear(): void {
    this.extensions.clear();
  }

  /**
   * Get registry statistics
   */
  getStats(): {
    totalExtensions: number;
    loadedExtensions: number;
    failedExtensions: number;
    extendedNodeTypes: number;
  } {
    const all = this.getAllExtensions();
    return {
      totalExtensions: all.length,
      loadedExtensions: all.filter(e => e.loaded).length,
      failedExtensions: all.filter(e => !!e.loadError).length,
      extendedNodeTypes: this.extensions.size,
    };
  }
}

// Singleton instance
let globalRegistry: NodeExtensionRegistry | null = null;

/**
 * Get the global extension registry
 */
export function getNodeExtensionRegistry(): NodeExtensionRegistry {
  if (!globalRegistry) {
    globalRegistry = new NodeExtensionRegistry();
  }
  return globalRegistry;
}

/**
 * Reset the global extension registry
 */
export function resetNodeExtensionRegistry(): void {
  if (globalRegistry) {
    globalRegistry.clear();
  }
  globalRegistry = null;
}

/**
 * Create a new extension registry with options
 */
export function createNodeExtensionRegistry(options?: ExtensionRegistryOptions): NodeExtensionRegistry {
  return new NodeExtensionRegistry(options);
}
