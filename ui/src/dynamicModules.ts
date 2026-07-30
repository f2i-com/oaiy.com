/**
 * Dynamic Module Loading
 *
 * All modules are now loaded at runtime from the plugins directory.
 * This allows modules to be modified and rebuilt without recompiling the app.
 *
 * Modules are loaded when the user starts the app from the splash screen.
 */

import { registerBundledModules } from 'oaiy-core';
import './pluginLoader'; // side effect: registers __PLUGIN_GLOBALS__ for (future) custom plugins
import { loadWebBundledModules } from './bundled-web-modules';
import { moduleLogger } from './utils/logger';

// ============================================
// Initialize Empty Module Registry
// ============================================

// Register an empty array initially - modules will be loaded at runtime
registerBundledModules([]);

// Log initialization in development (debug level is only shown in dev)
moduleLogger.debug('Initialized empty module registry');
moduleLogger.debug('Modules will be loaded from plugins directory at runtime');

// ============================================
// Runtime Plugin Loading
// ============================================

let pluginsLoaded = false;

/**
 * Load runtime plugins from the plugins directory.
 * This should be called after the app has initialized (from splash screen).
 * @param appDataPath - Optional custom app data directory path (plugins are in {path}/plugins)
 * Returns a promise that resolves when all plugins are loaded.
 */
export async function loadRuntimePlugins(appDataPath?: string): Promise<void> {
  if (pluginsLoaded) {
    moduleLogger.debug('Plugins already loaded, skipping');
    return;
  }

  // Web build: built-in modules are statically bundled from oaiy-core/modules
  // via import.meta.glob (no filesystem plugin dir in the browser).
  void appDataPath; // unused in the web build
  try {
    moduleLogger.debug('Loading bundled web modules...');
    const summary = await loadWebBundledModules();
    pluginsLoaded = true;
    moduleLogger.info(
      `Loaded ${summary.loaded}/${summary.total} modules, ${summary.errors.length} errors`,
    );
    // Make an empty/failed palette loud instead of silently shipping a blank
    // node panel — boot stays non-fatal, but the cause is visible in console.
    if (summary.total === 0 || summary.errors.length > 0) {
      moduleLogger.error(
        'No nodes loaded — the palette will be empty; see console',
        { loaded: summary.loaded, total: summary.total, errors: summary.errors },
      );
    }
  } catch (error) {
    // Boot stays non-fatal, but a swallowed failure here means the palette is
    // empty for the user with no obvious reason — log it loudly + actionably.
    moduleLogger.error(
      'Failed to load web modules — the node palette will be EMPTY. Check the import.meta.glob path/bundle in bundled-web-modules.ts and the console for the underlying error.',
      error,
    );
  }
}

/**
 * Reload plugins (e.g., after rebuilding or changing the app data path)
 */
export async function reloadPlugins(appDataPath?: string): Promise<void> {
  pluginsLoaded = false;
  await loadRuntimePlugins(appDataPath);
}

/**
 * Check if plugins have been loaded
 */
export function arePluginsLoaded(): boolean {
  return pluginsLoaded;
}
