/**
 * OAIY Module Discovery Service
 *
 * Handles discovering, loading, and watching modules from the filesystem.
 * Works in both browser (Tauri) and Node.js environments.
 */

import type {
  ModuleManifest,
  NodeDefinition,
  RuntimeModule,
  ModuleLoadResult,
  LoadedModule,
  ModuleCompiler,
} from './module-types';
import { ModuleLoader, getModuleLoader } from './module-loader';
import { moduleLogger } from './logger.js';

// ============================================
// File System Abstraction
// ============================================

export interface FileSystemAdapter {
  readTextFile(path: string): Promise<string>;
  readDir(path: string): Promise<string[]>;
  exists(path: string): Promise<boolean>;
  isDirectory(path: string): Promise<boolean>;
  joinPath(...parts: string[]): string;
}

/**
 * Browser/Tauri file system adapter
 */
export class TauriFileSystem implements FileSystemAdapter {
  private tauri: {
    invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
  } | null = null;

  constructor() {
    // Get Tauri API if available
    if (typeof window !== 'undefined' && (window as unknown as { __TAURI__?: unknown }).__TAURI__) {
      this.tauri = (window as unknown as {
        __TAURI__: { core: { invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T> } };
      }).__TAURI__.core;
    }
  }

  async readTextFile(path: string): Promise<string> {
    if (!this.tauri) {
      throw new Error('Tauri API not available');
    }
    return this.tauri.invoke<string>('read_text_file', { path });
  }

  async readDir(path: string): Promise<string[]> {
    if (!this.tauri) {
      throw new Error('Tauri API not available');
    }
    return this.tauri.invoke<string[]>('list_directory', { path });
  }

  async exists(path: string): Promise<boolean> {
    if (!this.tauri) {
      return false;
    }
    try {
      await this.tauri.invoke('file_exists', { path });
      return true;
    } catch {
      return false;
    }
  }

  async isDirectory(path: string): Promise<boolean> {
    if (!this.tauri) {
      return false;
    }
    try {
      return await this.tauri.invoke<boolean>('is_directory', { path });
    } catch {
      return false;
    }
  }

  joinPath(...parts: string[]): string {
    // Simple path join - works for both Windows and Unix
    return parts
      .map((part, i) => {
        if (i === 0) {
          return part.replace(/[/\\]+$/, '');
        }
        return part.replace(/^[/\\]+/, '').replace(/[/\\]+$/, '');
      })
      .filter(Boolean)
      .join('/');
  }
}

/**
 * In-memory file system adapter (for testing and bundled modules)
 */
export class MemoryFileSystem implements FileSystemAdapter {
  private files: Map<string, string> = new Map();
  private directories: Set<string> = new Set();

  addFile(path: string, content: string): void {
    this.files.set(this.normalizePath(path), content);
    // Add parent directories
    const parts = path.split('/');
    for (let i = 1; i < parts.length; i++) {
      this.directories.add(parts.slice(0, i).join('/'));
    }
  }

  addDirectory(path: string): void {
    this.directories.add(this.normalizePath(path));
  }

  private normalizePath(path: string): string {
    return path.replace(/\\/g, '/').replace(/\/+/g, '/');
  }

  async readTextFile(path: string): Promise<string> {
    const normalized = this.normalizePath(path);
    const content = this.files.get(normalized);
    if (content === undefined) {
      throw new Error(`File not found: ${path}`);
    }
    return content;
  }

  async readDir(path: string): Promise<string[]> {
    const normalized = this.normalizePath(path);
    const entries: string[] = [];

    // Find files and directories directly under this path
    for (const filePath of this.files.keys()) {
      if (filePath.startsWith(normalized + '/')) {
        const relative = filePath.slice(normalized.length + 1);
        const firstPart = relative.split('/')[0];
        if (firstPart && !entries.includes(firstPart)) {
          entries.push(firstPart);
        }
      }
    }

    for (const dirPath of this.directories) {
      if (dirPath.startsWith(normalized + '/')) {
        const relative = dirPath.slice(normalized.length + 1);
        const firstPart = relative.split('/')[0];
        if (firstPart && !entries.includes(firstPart)) {
          entries.push(firstPart);
        }
      }
    }

    return entries;
  }

  async exists(path: string): Promise<boolean> {
    const normalized = this.normalizePath(path);
    return this.files.has(normalized) || this.directories.has(normalized);
  }

  async isDirectory(path: string): Promise<boolean> {
    const normalized = this.normalizePath(path);
    return this.directories.has(normalized);
  }

  joinPath(...parts: string[]): string {
    return parts.filter(Boolean).join('/').replace(/\/+/g, '/');
  }

  clear(): void {
    this.files.clear();
    this.directories.clear();
  }
}

// ============================================
// Module Discovery Service
// ============================================

export interface DiscoveryOptions {
  coreModulesPath?: string;
  communityModulesPath?: string;
  enabledModules?: string[];
  fileSystem?: FileSystemAdapter;
}

export interface DiscoveredModule {
  path: string;
  manifest: ModuleManifest;
  nodePaths: string[];
  runtimePath?: string;
}

export class ModuleDiscoveryService {
  private fs: FileSystemAdapter;
  private loader: ModuleLoader;
  private discoveredModules: Map<string, DiscoveredModule> = new Map();

  constructor(options: { fileSystem?: FileSystemAdapter; loader?: ModuleLoader } = {}) {
    this.fs = options.fileSystem || new TauriFileSystem();
    this.loader = options.loader || getModuleLoader();
  }

  /**
   * Discover modules in a directory
   */
  async discoverModules(modulesPath: string): Promise<DiscoveredModule[]> {
    const discovered: DiscoveredModule[] = [];

    try {
      const exists = await this.fs.exists(modulesPath);
      if (!exists) {
        console.warn(`[ModuleDiscovery] Modules directory not found: ${modulesPath}`);
        return [];
      }

      const entries = await this.fs.readDir(modulesPath);

      for (const entry of entries) {
        const modulePath = this.fs.joinPath(modulesPath, entry);

        // Check if it's a directory
        const isDir = await this.fs.isDirectory(modulePath);
        if (!isDir) continue;

        // Check for module.json
        const manifestPath = this.fs.joinPath(modulePath, 'module.json');
        const hasManifest = await this.fs.exists(manifestPath);
        if (!hasManifest) continue;

        try {
          const module = await this.discoverModule(modulePath);
          if (module) {
            discovered.push(module);
            this.discoveredModules.set(module.manifest.id, module);
          }
        } catch (e) {
          console.error(`[ModuleDiscovery] Error discovering module at ${modulePath}:`, e);
        }
      }
    } catch (e) {
      console.error(`[ModuleDiscovery] Error scanning modules directory:`, e);
    }

    return discovered;
  }

  /**
   * Discover a single module
   */
  async discoverModule(modulePath: string): Promise<DiscoveredModule | null> {
    try {
      // Read manifest
      const manifestPath = this.fs.joinPath(modulePath, 'module.json');
      const manifestContent = await this.fs.readTextFile(manifestPath);
      const manifest: ModuleManifest = JSON.parse(manifestContent);

      // Find node definitions
      const nodesPath = this.fs.joinPath(modulePath, 'nodes');
      const nodePaths: string[] = [];

      if (await this.fs.exists(nodesPath)) {
        const nodeFiles = await this.fs.readDir(nodesPath);
        for (const file of nodeFiles) {
          if (file.endsWith('.json')) {
            nodePaths.push(this.fs.joinPath(nodesPath, file));
          }
        }
      }

      // Check for runtime
      const runtimePath = this.fs.joinPath(modulePath, 'runtime.js');
      const hasRuntime = await this.fs.exists(runtimePath);

      return {
        path: modulePath,
        manifest,
        nodePaths,
        runtimePath: hasRuntime ? runtimePath : undefined,
      };
    } catch (e) {
      console.error(`[ModuleDiscovery] Error discovering module at ${modulePath}:`, e);
      return null;
    }
  }

  /**
   * Load a discovered module
   */
  async loadDiscoveredModule(discovered: DiscoveredModule): Promise<ModuleLoadResult> {
    try {
      // Load node definitions
      const nodes: NodeDefinition[] = [];
      for (const nodePath of discovered.nodePaths) {
        const nodeContent = await this.fs.readTextFile(nodePath);
        const node: NodeDefinition = JSON.parse(nodeContent);
        nodes.push(node);
      }

      // Load runtime if available
      let runtime: RuntimeModule | undefined;
      if (discovered.runtimePath) {
        runtime = await this.loadRuntime(discovered.runtimePath);
      }

      // Load into module loader
      return await this.loader.loadModule(
        discovered.manifest,
        nodes,
        runtime,
        discovered.path
      );
    } catch (e) {
      return {
        success: false,
        error: {
          moduleId: discovered.manifest.id,
          modulePath: discovered.path,
          error: e instanceof Error ? e.message : String(e),
        },
      };
    }
  }

  /**
   * Load runtime module from JavaScript file
   */
  private async loadRuntime(runtimePath: string): Promise<RuntimeModule | undefined> {
    try {
      const code = await this.fs.readTextFile(runtimePath);

      // Create a sandboxed environment for the runtime
      // In production, this would use a more secure sandbox
      const exports: { default?: RuntimeModule } = {};

      // new Function is required to execute dynamically loaded module code.
      // SECURITY: this runs package/plugin-supplied JS in the HOST realm. Treat
      // plugin module code as UNTRUSTED — a caller MUST pre-sandbox before relying
      // on this. NOT "safe because trusted".
      // eslint-disable-next-line no-new-func
      const moduleFunction = new Function(
        'exports',
        'module',
        `
        const module = { exports: exports };
        ${code}
        if (module.exports.default) {
          exports.default = module.exports.default;
        } else if (module.exports.name) {
          exports.default = module.exports;
        }
        return module.exports;
      `
      );

      const result = moduleFunction(exports, { exports });

      // Handle different export styles
      if (result.default) {
        return result.default;
      } else if (result.name && result.methods) {
        return result;
      }

      return exports.default;
    } catch (e) {
      console.error(`[ModuleDiscovery] Error loading runtime from ${runtimePath}:`, e);
      return undefined;
    }
  }

  /**
   * Discover and load all modules from directories
   */
  async discoverAndLoadAll(options: DiscoveryOptions): Promise<{
    loaded: LoadedModule[];
    errors: Array<{ moduleId: string; error: string }>;
  }> {
    const loaded: LoadedModule[] = [];
    const errors: Array<{ moduleId: string; error: string }> = [];

    // Discover from core modules path
    if (options.coreModulesPath) {
      const coreModules = await this.discoverModules(options.coreModulesPath);
      for (const discovered of coreModules) {
        const result = await this.loadDiscoveredModule(discovered);
        if (result.success && result.module) {
          loaded.push(result.module);
        } else if (result.error) {
          errors.push({
            moduleId: result.error.moduleId,
            error: result.error.error,
          });
        }
      }
    }

    // Discover from community modules path
    if (options.communityModulesPath) {
      const communityModules = await this.discoverModules(options.communityModulesPath);
      for (const discovered of communityModules) {
        // Check if module is enabled
        if (options.enabledModules && !options.enabledModules.includes(discovered.manifest.id)) {
          continue;
        }

        const result = await this.loadDiscoveredModule(discovered);
        if (result.success && result.module) {
          loaded.push(result.module);
        } else if (result.error) {
          errors.push({
            moduleId: result.error.moduleId,
            error: result.error.error,
          });
        }
      }
    }

    return { loaded, errors };
  }

  /**
   * Get all discovered modules (without loading them)
   */
  getDiscoveredModules(): DiscoveredModule[] {
    return Array.from(this.discoveredModules.values());
  }

  /**
   * Get a discovered module by ID
   */
  getDiscoveredModule(moduleId: string): DiscoveredModule | undefined {
    return this.discoveredModules.get(moduleId);
  }

  /**
   * Clear discovered modules cache
   */
  clearCache(): void {
    this.discoveredModules.clear();
  }
}

// ============================================
// Bundled Modules Support
// ============================================

/**
 * Load bundled modules from inline definitions
 * This is used for core modules that are bundled with the app
 */
export async function loadBundledModules(
  loader: ModuleLoader,
  bundledModules: Array<{
    manifest: ModuleManifest;
    nodes: NodeDefinition[];
    runtime?: RuntimeModule;
    compiler?: ModuleCompiler;
  }>
): Promise<{ loaded: number; errors: string[]; skipped: number }> {
  const errors: string[] = [];
  let loaded = 0;
  let skipped = 0;

  for (const bundled of bundledModules) {
    // Debug: Log compiler presence
    moduleLogger.debug(`loadBundledModules: Processing module ${bundled.manifest.id}`, {
      hasCompiler: !!bundled.compiler,
      compilerName: bundled.compiler?.name || 'N/A',
    });

    // Skip if module is already loaded
    if (loader.modules.has(bundled.manifest.id)) {
      skipped++;
      continue;
    }

    const result = await loader.loadModule(
      bundled.manifest,
      bundled.nodes,
      bundled.runtime,
      'bundled',
      bundled.compiler
    );

    if (result.success) {
      loaded++;
    } else if (result.error) {
      errors.push(`${result.error.moduleId}: ${result.error.error}`);
    }
  }

  return { loaded, errors, skipped };
}

// ============================================
// Singleton Instance
// ============================================

let discoveryServiceInstance: ModuleDiscoveryService | null = null;

/**
 * Get the global module discovery service instance
 */
export function getModuleDiscovery(): ModuleDiscoveryService {
  if (!discoveryServiceInstance) {
    discoveryServiceInstance = new ModuleDiscoveryService();
  }
  return discoveryServiceInstance;
}

/**
 * Reset the discovery service (mainly for testing)
 */
export function resetModuleDiscovery(): void {
  if (discoveryServiceInstance) {
    discoveryServiceInstance.clearCache();
    discoveryServiceInstance = null;
  }
}
