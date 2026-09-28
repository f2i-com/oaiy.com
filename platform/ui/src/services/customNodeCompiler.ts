/**
 * Custom Node Compiler Service
 *
 * Uses esbuild-wasm to compile TypeScript custom node sources from .oaiy packages.
 * Compiles compiler, runtime, and UI code for custom nodes.
 */

import * as esbuild from 'esbuild-wasm';
// Emitted as a hashed asset by the bundler rather than assumed to exist in
// public/ — see initializeEsbuild() below.
import esbuildWasmURL from 'esbuild-wasm/esbuild.wasm?url';
import type { EmbeddedCustomNode } from 'oaiy-core';
import { createLogger } from '../utils/logger';

const logger = createLogger('CustomNodeCompiler');

// ============================================
// esbuild Initialization
// ============================================

let esbuildInitialized = false;
let esbuildInitializing = false;
let initPromise: Promise<void> | null = null;

/**
 * Initialize esbuild-wasm (only needs to be done once)
 */
async function initializeEsbuild(): Promise<void> {
  if (esbuildInitialized) return;

  if (esbuildInitializing && initPromise) {
    return initPromise;
  }

  esbuildInitializing = true;

  initPromise = (async () => {
    // Local-only: the CDN fallback was removed as a supply-chain and
    // offline-first concern.
    //
    // The URL comes from a `?url` import (see the top of this file), so the
    // bundler resolves it out of node_modules and emits it as a hashed asset.
    // This used to hard-code `/esbuild.wasm` and HEAD-check it, which meant
    // custom-node compilation was dead in every build — nothing ever put that
    // file in public/, so the check always failed and the only symptom was an
    // error telling you to restore a path that doesn't exist in this repo.
    // Importing it makes the asset a build dependency instead of an assumption.
    try {
      logger.debug('Using bundled esbuild.wasm', { url: esbuildWasmURL });
      await esbuild.initialize({ wasmURL: esbuildWasmURL });
      esbuildInitialized = true;
      logger.debug('esbuild-wasm initialized');
    } catch (error) {
      const errorMessage = (error as Error).message || '';
      // Handle both possible error messages for already-initialized state
      if (errorMessage.includes('already been called') ||
          errorMessage.includes('more than once') ||
          errorMessage.includes('Cannot call "initialize"')) {
        logger.debug('esbuild-wasm already initialized (shared with plugin compiler)');
        esbuildInitialized = true;
      } else {
        throw error;
      }
    }
  })();

  return initPromise;
}

// ============================================
// Types
// ============================================

export interface CustomNodeCompileResult {
  success: boolean;
  nodeId: string;
  compiled?: {
    compiler: string;
    runtime: string;
    ui?: string;
  };
  error?: string;
}

export interface CustomNodeBatchCompileResult {
  total: number;
  successful: number;
  failed: number;
  results: CustomNodeCompileResult[];
}

// ============================================
// External Module Mappings
// ============================================

// Compile-time globals exposed to custom node code (compiler / runtime / UI).
//
// Hardening (round 4 #5): the previous map mapped `@tauri-apps/api`,
// `@tauri-apps/api/core`, `@tauri-apps/api/event`, `@tauri-apps/api/path`
// and `@tauri-apps/plugin-sql` straight to globals — meaning any custom
// node from a package could `import { invoke } from '@tauri-apps/api/core'`
// and call any Tauri command, bypassing the runtime broker (which is
// where capability checks, denylist, and allowlist live). They are
// intentionally absent here. Custom nodes that need Tauri capabilities
// must go through oaiy-core's runtime context (`ctx.tauri.invoke` for
// runtime-module code, host calls for workflow code), which routes
// through the central permission gate.
const externalGlobals: Record<string, string> = {
  'react': '__PLUGIN_GLOBALS__.React',
  'react-dom': '__PLUGIN_GLOBALS__.ReactDOM',
  'react/jsx-runtime': '__PLUGIN_GLOBALS__.ReactJSXRuntime',
  '@xyflow/react': '__PLUGIN_GLOBALS__.ReactFlow',
  'oaiy-core': '__PLUGIN_GLOBALS__.OAIYCore',
  'oaiy-ui-components': '__PLUGIN_GLOBALS__.OAIYUIComponents',
  '@monaco-editor/react': '__PLUGIN_GLOBALS__.MonacoReact',
};

// Tauri imports that we *deliberately* block from custom-node code. If a
// custom node tries to import any of these we return a stub module that
// throws on access, with a message pointing at the broker.
const BLOCKED_TAURI_IMPORTS: ReadonlySet<string> = new Set([
  '@tauri-apps/api',
  '@tauri-apps/api/core',
  '@tauri-apps/api/event',
  '@tauri-apps/api/path',
  '@tauri-apps/plugin-sql',
]);

// ============================================
// Compilation Functions
// ============================================

/**
 * Create an esbuild plugin for virtual file compilation
 */
function createVirtualPlugin(sources: Map<string, string>, entryCode: string): esbuild.Plugin {
  return {
    name: 'virtual-custom-node',
    setup(build) {
      // Handle the entry point
      build.onResolve({ filter: /^__entry__$/ }, () => ({
        path: '__entry__',
        namespace: 'virtual',
      }));

      // Handle relative imports
      build.onResolve({ filter: /^\./ }, (args) => {
        let importerDir = '';
        if (args.importer && args.importer !== '__entry__') {
          const lastSlash = args.importer.lastIndexOf('/');
          if (lastSlash !== -1) {
            importerDir = args.importer.slice(0, lastSlash + 1);
          }
        }

        let resolvedPath = args.path;
        if (resolvedPath.startsWith('./')) {
          resolvedPath = resolvedPath.slice(2);
        }

        const basePath = importerDir + resolvedPath;
        const extensions = ['', '.ts', '.tsx', '/index.ts', '/index.tsx'];

        for (const ext of extensions) {
          const fullPath = basePath + ext;
          if (sources.has(fullPath)) {
            return { path: fullPath, namespace: 'virtual' };
          }
        }

        return { external: true };
      });

      // Handle external modules
      build.onResolve({ filter: /^[^.]/ }, (args) => {
        if (BLOCKED_TAURI_IMPORTS.has(args.path)) {
          // Custom nodes are not allowed to call Tauri APIs directly —
          // route the import to a stub that throws a clear, actionable
          // error at runtime so package authors get pushed back through
          // the oaiy-core runtime context.
          return { path: args.path, namespace: 'blocked-tauri-import' };
        }
        if (args.path in externalGlobals) {
          return { path: args.path, namespace: 'external-global' };
        }
        return { external: true };
      });

      // Load external globals
      build.onLoad({ filter: /.*/, namespace: 'external-global' }, (args) => {
        const globalName = externalGlobals[args.path];
        if (globalName) {
          return {
            contents: `
              var mod = ${globalName};
              if (mod && typeof mod === 'object') {
                var result = Object.assign({}, mod, { __esModule: true });
                if (!mod.default) {
                  result.default = mod;
                }
                module.exports = result;
              } else {
                module.exports = mod;
              }
            `,
            loader: 'js',
          };
        }
        return null;
      });

      // Stub for explicitly-blocked Tauri imports. Any property access on
      // the resulting module throws — the compile succeeds but a custom
      // node that actually tries to use the import fails fast with a
      // pointer to the supported route.
      build.onLoad({ filter: /.*/, namespace: 'blocked-tauri-import' }, (args) => {
        const importName = args.path;
        return {
          contents: `
            function __blocked_tauri_handler(target, prop) {
              throw new Error(
                'Custom nodes cannot import "${importName}" directly. ' +
                'Use the runtime context (ctx.tauri.invoke / host.call) ' +
                'so calls are gated by the package permission system.'
              );
            }
            var blocked = new Proxy({}, {
              get: __blocked_tauri_handler,
              apply: __blocked_tauri_handler,
              construct: __blocked_tauri_handler,
            });
            module.exports = blocked;
            module.exports.default = blocked;
          `,
          loader: 'js',
        };
      });

      // Load virtual files
      build.onLoad({ filter: /.*/, namespace: 'virtual' }, (args) => {
        if (args.path === '__entry__') {
          return { contents: entryCode, loader: 'ts' };
        }

        const content = sources.get(args.path);
        if (content !== undefined) {
          const loader = args.path.endsWith('.tsx') ? 'tsx' : 'ts';
          return { contents: content, loader };
        }

        return null;
      });
    },
  };
}

/**
 * Compile a single TypeScript/TSX source to JavaScript
 */
async function compileSource(
  source: string,
  filename: string,
  format: 'iife' | 'cjs' = 'iife',
  globalName?: string
): Promise<{ success: boolean; code?: string; error?: string }> {
  try {
    await initializeEsbuild();

    const sources = new Map<string, string>();
    sources.set('source.ts', source);

    const entryCode = `export * from './source';`;

    const result = await esbuild.build({
      entryPoints: ['__entry__'],
      bundle: true,
      write: false,
      format,
      globalName,
      platform: 'browser',
      target: 'es2020',
      minify: false,
      plugins: [createVirtualPlugin(sources, entryCode)],
      jsx: 'automatic',
      define: {
        'process.env.NODE_ENV': '"production"',
      },
    });

    if (result.errors.length > 0) {
      return {
        success: false,
        error: result.errors.map(e => `${filename}: ${e.text}`).join('\n'),
      };
    }

    const output = result.outputFiles?.[0]?.text || '';
    return { success: true, code: output };
  } catch (error) {
    return {
      success: false,
      error: `${filename}: ${String(error)}`,
    };
  }
}

/**
 * Compile a custom node's compiler code
 */
async function compileCompilerCode(source: string, nodeId: string): Promise<{ success: boolean; code?: string; error?: string }> {
  // The compiler code should export a function named 'compile'
  // The source is expected to already export the compile function
  const wrappedSource = `
// Custom node compiler for ${nodeId}
${source}
`;

  return compileSource(wrappedSource, `${nodeId}/compiler.ts`, 'iife', `__CUSTOM_NODE_COMPILER_${nodeId.replace(/[^a-zA-Z0-9]/g, '_')}__`);
}

/**
 * Compile a custom node's runtime code
 */
async function compileRuntimeCode(source: string, nodeId: string): Promise<{ success: boolean; code?: string; error?: string }> {
  // The runtime code should export an execute function
  // The source is expected to already export the execute function
  const wrappedSource = `
// Custom node runtime for ${nodeId}
${source}
`;

  return compileSource(wrappedSource, `${nodeId}/runtime.ts`, 'iife', `__CUSTOM_NODE_RUNTIME_${nodeId.replace(/[^a-zA-Z0-9]/g, '_')}__`);
}

/**
 * Compile a custom node's UI component
 */
async function compileUICode(source: string, nodeId: string): Promise<{ success: boolean; code?: string; error?: string }> {
  // The UI code should export a default React component
  const wrappedSource = `
// Custom node UI for ${nodeId}
${source}
`;

  return compileSource(wrappedSource, `${nodeId}/ui.tsx`, 'iife', `__CUSTOM_NODE_UI_${nodeId.replace(/[^a-zA-Z0-9]/g, '_')}__`);
}

/**
 * Compile a single custom node definition
 */
export async function compileCustomNode(node: EmbeddedCustomNode): Promise<CustomNodeCompileResult> {
  try {
    await initializeEsbuild();

    const compiled: { compiler: string; runtime: string; ui?: string } = {
      compiler: '',
      runtime: '',
    };

    // Compile compiler code
    if (node.source.compiler) {
      const result = await compileCompilerCode(node.source.compiler, node.id);
      if (!result.success) {
        return { success: false, nodeId: node.id, error: `Compiler: ${result.error}` };
      }
      compiled.compiler = result.code!;
    }

    // Compile runtime code
    if (node.source.runtime) {
      const result = await compileRuntimeCode(node.source.runtime, node.id);
      if (!result.success) {
        return { success: false, nodeId: node.id, error: `Runtime: ${result.error}` };
      }
      compiled.runtime = result.code!;
    }

    // Compile UI code (optional)
    if (node.source.ui) {
      const result = await compileUICode(node.source.ui, node.id);
      if (!result.success) {
        return { success: false, nodeId: node.id, error: `UI: ${result.error}` };
      }
      compiled.ui = result.code;
    }

    logger.debug(`Successfully compiled node: ${node.id}`);
    return { success: true, nodeId: node.id, compiled };
  } catch (error) {
    return { success: false, nodeId: node.id, error: String(error) };
  }
}

/**
 * Compile all custom nodes in a package
 */
export async function compileCustomNodes(nodes: EmbeddedCustomNode[]): Promise<CustomNodeBatchCompileResult> {
  const result: CustomNodeBatchCompileResult = {
    total: nodes.length,
    successful: 0,
    failed: 0,
    results: [],
  };

  for (const node of nodes) {
    const compileResult = await compileCustomNode(node);
    result.results.push(compileResult);

    if (compileResult.success) {
      result.successful++;
    } else {
      result.failed++;
      logger.error(`Failed to compile ${node.id}: ${compileResult.error}`);
    }
  }

  logger.debug(`Compiled ${result.successful}/${result.total} custom nodes`);
  return result;
}

/**
 * Check if esbuild is initialized
 */
export function isCompilerReady(): boolean {
  return esbuildInitialized;
}

/**
 * Pre-initialize the compiler (useful for startup)
 */
export async function initializeCompiler(): Promise<void> {
  await initializeEsbuild();
}
