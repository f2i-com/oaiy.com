/**
 * OAIY Module Loader
 *
 * Handles loading, validating, and managing node modules.
 */

import type {
  ModuleManifest,
  NodeDefinition,
  LoadedModule,
  ModuleLoadResult,
  ModuleLoadError,
  ModuleRegistry,
  ModuleCategory,
  ValidationResult,
  ValidationError,
  RuntimeModule,
  RuntimeContext,
  ModuleEvent,
  ModuleEventHandler,
  ModuleCompiler,
} from './module-types';
import { moduleLogger } from './logger.js';
import { bumpFlowPlanCompilerCacheVersion } from './flowplan-compiler.js';

// ============================================
// Module Loader Class
// ============================================

export class ModuleLoader implements ModuleRegistry {
  modules: Map<string, LoadedModule> = new Map();
  nodeDefinitions: Map<string, NodeDefinition> = new Map();
  nodeToModule: Map<string, string> = new Map();

  private eventHandlers: Set<ModuleEventHandler> = new Set();
  private runtimeContext: RuntimeContext | null = null;

  /**
   * Set the runtime context for module initialization
   */
  setRuntimeContext(context: RuntimeContext): void {
    this.runtimeContext = context;
  }

  /**
   * Subscribe to module events
   */
  on(handler: ModuleEventHandler): () => void {
    this.eventHandlers.add(handler);
    return () => this.eventHandlers.delete(handler);
  }

  /**
   * Emit a module event
   */
  private emit(event: ModuleEvent): void {
    for (const handler of this.eventHandlers) {
      try {
        handler(event);
      } catch (e) {
        console.error('[ModuleLoader] Event handler error:', e);
      }
    }
  }

  /**
   * Load a module from a parsed manifest and nodes
   */
  async loadModule(
    manifest: ModuleManifest,
    nodes: NodeDefinition[],
    runtime: RuntimeModule | undefined,
    modulePath: string,
    compiler?: ModuleCompiler
  ): Promise<ModuleLoadResult> {
    try {
      // Validate manifest
      const manifestValidation = this.validateManifest(manifest);
      if (!manifestValidation.valid) {
        return {
          success: false,
          error: {
            moduleId: manifest.id || 'unknown',
            modulePath,
            error: 'Invalid module manifest',
            details: manifestValidation.errors,
          },
        };
      }

      // Validate nodes
      for (const node of nodes) {
        const nodeValidation = this.validateNodeDefinition(node);
        if (!nodeValidation.valid) {
          return {
            success: false,
            error: {
              moduleId: manifest.id,
              modulePath,
              error: `Invalid node definition: ${node.id}`,
              details: nodeValidation.errors,
            },
          };
        }
      }

      // Check for duplicate module ID
      if (this.modules.has(manifest.id)) {
        console.warn(`[ModuleLoader] Module ${manifest.id} already loaded, replacing`);
        await this.unloadModule(manifest.id);
      }

      // Check for node ID conflicts
      for (const node of nodes) {
        const existingModule = this.nodeToModule.get(node.id);
        if (existingModule && existingModule !== manifest.id) {
          return {
            success: false,
            error: {
              moduleId: manifest.id,
              modulePath,
              error: `Node type '${node.id}' already registered by module '${existingModule}'`,
            },
          };
        }
      }

      // Initialize runtime if provided
      if (runtime && this.runtimeContext) {
        try {
          if (runtime.init) {
            await runtime.init(this.runtimeContext);
          }
        } catch (e) {
          return {
            success: false,
            error: {
              moduleId: manifest.id,
              modulePath,
              error: `Runtime initialization failed: ${e instanceof Error ? e.message : String(e)}`,
            },
          };
        }
      }

      // Create loaded module
      const loadedModule: LoadedModule = {
        manifest,
        nodes: new Map(nodes.map(n => [n.id, n])),
        runtime,
        compiler,
        path: modulePath,
        enabled: true,
      };

      // Debug: Log module registration details
      moduleLogger.debug(`loadModule: ${manifest.id}`, {
        hasCompiler: !!compiler,
        compilerName: compiler?.name || 'N/A',
        nodeTypes: nodes.map(n => n.id),
      });

      // Register module
      this.modules.set(manifest.id, loadedModule);

      // Register nodes
      for (const node of nodes) {
        this.nodeDefinitions.set(node.id, node);
        this.nodeToModule.set(node.id, manifest.id);
        this.emit({ type: 'node:registered', moduleId: manifest.id, nodeId: node.id });
      }

      // Invalidate the FlowPlan compiler caches so AI-generated FlowPlans
      // built after this load see the freshly registered nodes.
      bumpFlowPlanCompilerCacheVersion();

      this.emit({ type: 'module:loaded', moduleId: manifest.id });

      return { success: true, module: loadedModule };
    } catch (e) {
      const error: ModuleLoadError = {
        moduleId: manifest?.id || 'unknown',
        modulePath,
        error: e instanceof Error ? e.message : String(e),
      };
      this.emit({ type: 'module:error', moduleId: error.moduleId, data: error });
      return { success: false, error };
    }
  }

  /**
   * Unload a module
   */
  async unloadModule(moduleId: string): Promise<boolean> {
    const module = this.modules.get(moduleId);
    if (!module) {
      return false;
    }

    // Cleanup runtime
    if (module.runtime?.cleanup) {
      try {
        await module.runtime.cleanup();
      } catch (e) {
        console.error(`[ModuleLoader] Error cleaning up module ${moduleId}:`, e);
      }
    }

    // Unregister nodes
    for (const nodeId of module.nodes.keys()) {
      this.nodeDefinitions.delete(nodeId);
      this.nodeToModule.delete(nodeId);
      this.emit({ type: 'node:unregistered', moduleId, nodeId });
    }

    // Remove module
    this.modules.delete(moduleId);
    bumpFlowPlanCompilerCacheVersion();
    this.emit({ type: 'module:unloaded', moduleId });

    return true;
  }

  /**
   * Enable a module
   */
  enableModule(moduleId: string): boolean {
    const module = this.modules.get(moduleId);
    if (!module) {
      return false;
    }

    if (!module.enabled) {
      module.enabled = true;

      // Re-register nodes
      for (const [nodeId, node] of module.nodes) {
        this.nodeDefinitions.set(nodeId, node);
        this.nodeToModule.set(nodeId, moduleId);
      }

      this.emit({ type: 'module:enabled', moduleId });
    }

    return true;
  }

  /**
   * Disable a module
   */
  disableModule(moduleId: string): boolean {
    const module = this.modules.get(moduleId);
    if (!module) {
      return false;
    }

    if (module.enabled) {
      module.enabled = false;

      // Unregister nodes (but keep module loaded)
      for (const nodeId of module.nodes.keys()) {
        this.nodeDefinitions.delete(nodeId);
        this.nodeToModule.delete(nodeId);
      }

      this.emit({ type: 'module:disabled', moduleId });
    }

    return true;
  }

  /**
   * Get a module by ID
   */
  getModule(moduleId: string): LoadedModule | undefined {
    return this.modules.get(moduleId);
  }

  /**
   * Get a node definition by type
   */
  getNodeDefinition(nodeType: string): NodeDefinition | undefined {
    return this.nodeDefinitions.get(nodeType);
  }

  /**
   * Get all node definitions
   */
  getAllNodeDefinitions(): NodeDefinition[] {
    return Array.from(this.nodeDefinitions.values());
  }

  /**
   * Get the module that provides a node type
   */
  getModuleForNode(nodeType: string): LoadedModule | undefined {
    const moduleId = this.nodeToModule.get(nodeType);
    return moduleId ? this.modules.get(moduleId) : undefined;
  }

  /**
   * Get nodes by category
   */
  getNodesByCategory(category: ModuleCategory): NodeDefinition[] {
    const nodes: NodeDefinition[] = [];
    for (const module of this.modules.values()) {
      if (module.enabled && module.manifest.category === category) {
        for (const node of module.nodes.values()) {
          nodes.push(node);
        }
      }
    }
    return nodes;
  }

  /**
   * Check if a node type is valid
   */
  isNodeTypeValid(nodeType: string): boolean {
    return this.nodeDefinitions.has(nodeType);
  }

  /**
   * Get all loaded modules
   */
  getAllModules(): LoadedModule[] {
    return Array.from(this.modules.values());
  }

  /**
   * Get enabled modules
   */
  getEnabledModules(): LoadedModule[] {
    return Array.from(this.modules.values()).filter(m => m.enabled);
  }

  /**
   * Get runtime for a module
   */
  getModuleRuntime(moduleId: string): RuntimeModule | undefined {
    return this.modules.get(moduleId)?.runtime;
  }

  /**
   * Get all runtime modules
   */
  getAllRuntimes(): RuntimeModule[] {
    return Array.from(this.modules.values())
      .filter(m => m.enabled && m.runtime)
      .map(m => m.runtime!);
  }

  /**
   * Get compiler for a module
   */
  getModuleCompiler(moduleId: string): ModuleCompiler | undefined {
    return this.modules.get(moduleId)?.compiler;
  }

  /**
   * Get compiler for a node type
   */
  getCompilerForNode(nodeType: string): ModuleCompiler | undefined {
    const moduleId = this.nodeToModule.get(nodeType);
    return moduleId ? this.modules.get(moduleId)?.compiler : undefined;
  }

  /**
   * Get all module compilers
   */
  getAllCompilers(): ModuleCompiler[] {
    return Array.from(this.modules.values())
      .filter(m => m.enabled && m.compiler)
      .map(m => m.compiler!);
  }

  /**
   * Validate a module manifest
   */
  validateManifest(manifest: ModuleManifest): ValidationResult {
    const errors: ValidationError[] = [];

    if (!manifest.id || typeof manifest.id !== 'string') {
      errors.push({ path: 'id', message: 'Module ID is required and must be a string' });
    } else if (!/^[a-z][a-z0-9-]*$/.test(manifest.id)) {
      errors.push({
        path: 'id',
        message: 'Module ID must start with a letter and contain only lowercase letters, numbers, and hyphens',
        value: manifest.id,
      });
    }

    if (!manifest.name || typeof manifest.name !== 'string') {
      errors.push({ path: 'name', message: 'Module name is required' });
    }

    if (!manifest.version || typeof manifest.version !== 'string') {
      errors.push({ path: 'version', message: 'Module version is required' });
    } else if (!/^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$/.test(manifest.version)) {
      errors.push({
        path: 'version',
        message: 'Module version must be a valid semantic version (e.g., 1.0.0)',
        value: manifest.version,
      });
    }

    // Empty array is allowed for runtime-only plugins (e.g. plugin-llm,
    // plugin-tts) that contribute only Tauri commands consumed by other
    // modules' nodes rather than introducing their own workflow nodes.
    if (!manifest.nodes || !Array.isArray(manifest.nodes)) {
      errors.push({ path: 'nodes', message: 'Module nodes must be an array (can be empty for runtime-only plugins)' });
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Validate a node definition
   */
  validateNodeDefinition(node: NodeDefinition): ValidationResult {
    const errors: ValidationError[] = [];

    if (!node.id || typeof node.id !== 'string') {
      errors.push({ path: 'id', message: 'Node ID is required' });
    } else if (!node.id.startsWith('pkg:') && !/^[a-z][a-z0-9_]*$/.test(node.id)) {
      // Allow prefixed package node IDs (pkg:packageId:nodeId) or standard IDs
      errors.push({
        path: 'id',
        message: 'Node ID must start with a letter and contain only lowercase letters, numbers, and underscores',
        value: node.id,
      });
    }

    if (!node.name || typeof node.name !== 'string') {
      errors.push({ path: 'name', message: 'Node name is required' });
    }

    if (!Array.isArray(node.inputs)) {
      errors.push({ path: 'inputs', message: 'Node inputs must be an array' });
    }

    if (!Array.isArray(node.outputs)) {
      errors.push({ path: 'outputs', message: 'Node outputs must be an array' });
    }

    if (!node.compiler || typeof node.compiler !== 'object') {
      errors.push({ path: 'compiler', message: 'Node compiler configuration is required' });
    } else if (!node.compiler.template && !node.compiler.customHandler) {
      errors.push({
        path: 'compiler.template',
        message: 'Node must have either a compiler template or customHandler',
      });
    }

    // Validate handles
    const handleIds = new Set<string>();
    for (const input of node.inputs || []) {
      if (handleIds.has(`input:${input.id}`)) {
        errors.push({ path: `inputs.${input.id}`, message: `Duplicate input handle ID: ${input.id}` });
      }
      handleIds.add(`input:${input.id}`);
    }

    for (const output of node.outputs || []) {
      if (handleIds.has(`output:${output.id}`)) {
        errors.push({ path: `outputs.${output.id}`, message: `Duplicate output handle ID: ${output.id}` });
      }
      handleIds.add(`output:${output.id}`);
    }

    // Validate properties
    const propIds = new Set<string>();
    for (const prop of node.properties || []) {
      if (propIds.has(prop.id)) {
        errors.push({ path: `properties.${prop.id}`, message: `Duplicate property ID: ${prop.id}` });
      }
      propIds.add(prop.id);
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Clear all loaded modules
   */
  async clear(): Promise<void> {
    const moduleIds = Array.from(this.modules.keys());
    for (const moduleId of moduleIds) {
      await this.unloadModule(moduleId);
    }
  }

  /**
   * Get statistics about loaded modules
   */
  getStats(): {
    totalModules: number;
    enabledModules: number;
    totalNodes: number;
    nodesByCategory: Record<string, number>;
  } {
    const nodesByCategory: Record<string, number> = {};

    for (const module of this.modules.values()) {
      if (module.enabled) {
        const category = module.manifest.category || 'Custom';
        nodesByCategory[category] = (nodesByCategory[category] || 0) + module.nodes.size;
      }
    }

    return {
      totalModules: this.modules.size,
      enabledModules: this.getEnabledModules().length,
      totalNodes: this.nodeDefinitions.size,
      nodesByCategory,
    };
  }
}

// ============================================
// Singleton Instance
// ============================================

let moduleLoaderInstance: ModuleLoader | null = null;

/**
 * Get the global module loader instance
 */
export function getModuleLoader(): ModuleLoader {
  if (!moduleLoaderInstance) {
    moduleLoaderInstance = new ModuleLoader();
  }
  return moduleLoaderInstance;
}

/**
 * Reset the module loader (mainly for testing)
 */
export async function resetModuleLoader(): Promise<void> {
  if (moduleLoaderInstance) {
    await moduleLoaderInstance.clear();
    moduleLoaderInstance = null;
  }
}
