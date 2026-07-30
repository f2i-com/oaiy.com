/**
 * User Macro Storage Utility
 *
 * Provides file-based storage for user modifications to macros.
 *
 * Two types of storage:
 * 1. **Macro Overrides**: User modifications to built-in macros (stored by original ID)
 * 2. **User Macros**: Completely new macros created by the user
 *
 * Storage location: {APP_DATA}/oaiy/user-macros.json
 */

import type { Flow } from 'oaiy-core';
import { createLogger } from './logger';

const logger = createLogger('UserMacros');

// Tauri invoke type
declare const window: Window & {
  __TAURI__?: {
    core: {
      invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
    };
  };
};

const USER_MACROS_FILENAME = 'user-macros.json';

interface UserMacroStorage {
  // Overrides for built-in macros (keyed by original macro ID)
  overrides: Record<string, Flow>;
  // Completely new user-created macros
  userMacros: Flow[];
}

const EMPTY_STORAGE: UserMacroStorage = {
  overrides: {},
  userMacros: [],
};

/**
 * Get the path to the user macros file
 */
async function getUserMacrosPath(): Promise<string | null> {
  try {
    const tauri = window.__TAURI__;
    if (!tauri) return null;

    const appDataDir = await tauri.core.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    return `${appDataDir}/${USER_MACROS_FILENAME}`;
  } catch (error) {
    logger.warn('Could not get app data dir', { error });
    return null;
  }
}

/**
 * Load the full user macro storage from file system
 */
async function loadStorage(): Promise<UserMacroStorage> {
  try {
    const tauri = window.__TAURI__;
    if (!tauri) {
      logger.debug('Tauri not available, using localStorage fallback');
      return loadStorageFromLocalStorage();
    }

    const path = await getUserMacrosPath();
    if (!path) return EMPTY_STORAGE;

    const result = await tauri.core.invoke<{ content: string; isLargeFile: boolean }>(
      'plugin:oaiy-filesystem|read_file',
      { path, readAs: 'text' }
    );

    const storage = JSON.parse(result.content) as UserMacroStorage;
    // Ensure both fields exist (backward compatibility)
    return {
      overrides: storage.overrides || {},
      userMacros: storage.userMacros || [],
    };
  } catch (error) {
    // File might not exist yet, that's OK
    if (String(error).includes('not found') || String(error).includes('No such file')) {
      logger.debug('No user macros file found, starting fresh');
      return EMPTY_STORAGE;
    }
    logger.warn('Error loading storage', { error });
    return loadStorageFromLocalStorage();
  }
}

/**
 * Save the full user macro storage to file system
 */
async function saveStorage(storage: UserMacroStorage): Promise<boolean> {
  try {
    const tauri = window.__TAURI__;
    if (!tauri) {
      logger.debug('Tauri not available, using localStorage fallback');
      return saveStorageToLocalStorage(storage);
    }

    const path = await getUserMacrosPath();
    if (!path) return false;

    await tauri.core.invoke('plugin:oaiy-filesystem|write_file', {
      path,
      content: JSON.stringify(storage, null, 2),
      contentType: 'text',
      createDirs: true,
    });

    logger.debug('Saved storage to file');
    return true;
  } catch (error) {
    logger.error('Error saving storage', { error });
    return saveStorageToLocalStorage(storage);
  }
}

// ========================================
// Write serialization
// ========================================

/**
 * Serializes read-modify-write critical sections so concurrent saves do not
 * read the same baseline and clobber each other's mutations (lost overrides /
 * user macros). Each mutator routes its ENTIRE loadStorage..saveStorage section
 * through this chain.
 */
let writeQueue: Promise<unknown> = Promise.resolve();

function enqueueWrite<T>(doMutation: () => Promise<T>): Promise<T> {
  const run = writeQueue.then(() => doMutation());
  // Keep the chain alive even if a mutation rejects, so later writes still run.
  writeQueue = run.catch(() => {});
  return run;
}

// ========================================
// Macro Override Functions
// ========================================

/**
 * Check if a built-in macro has been overridden by the user
 */
export async function hasMacroOverride(macroId: string): Promise<boolean> {
  const storage = await loadStorage();
  return macroId in storage.overrides;
}

/**
 * Get the user's override for a built-in macro (if any)
 */
export async function getMacroOverride(macroId: string): Promise<Flow | null> {
  const storage = await loadStorage();
  return storage.overrides[macroId] || null;
}

/**
 * Get all macro overrides
 */
export async function getAllMacroOverrides(): Promise<Record<string, Flow>> {
  const storage = await loadStorage();
  return storage.overrides;
}

/**
 * Save a user override for a built-in macro
 */
export async function saveMacroOverride(macroId: string, macro: Flow): Promise<boolean> {
  return enqueueWrite(async () => {
    const storage = await loadStorage();
    storage.overrides[macroId] = {
      ...macro,
      updatedAt: new Date().toISOString(),
    };
    const success = await saveStorage(storage);
    if (success) {
      logger.debug(`Saved override for "${macro.name}"`);
    }
    return success;
  });
}

/**
 * Remove a user override, reverting to the original built-in macro
 */
export async function revertMacroOverride(macroId: string): Promise<boolean> {
  return enqueueWrite(async () => {
    const storage = await loadStorage();
    if (!(macroId in storage.overrides)) {
      return false; // No override to revert
    }
    delete storage.overrides[macroId];
    const success = await saveStorage(storage);
    if (success) {
      logger.debug(`Reverted macro "${macroId}" to original`);
    }
    return success;
  });
}

// ========================================
// User Macro Functions (completely new macros)
// ========================================

/**
 * Load all user-created macros
 */
export async function loadUserMacros(): Promise<Flow[]> {
  const storage = await loadStorage();
  return storage.userMacros;
}

/**
 * Save a user-created macro
 */
export async function saveUserMacro(macro: Flow): Promise<boolean> {
  return enqueueWrite(async () => {
    const storage = await loadStorage();
    const existingIndex = storage.userMacros.findIndex(m => m.id === macro.id);

    if (existingIndex >= 0) {
      storage.userMacros[existingIndex] = {
        ...macro,
        updatedAt: new Date().toISOString(),
      };
    } else {
      storage.userMacros.push({
        ...macro,
        updatedAt: new Date().toISOString(),
      });
    }

    return saveStorage(storage);
  });
}

/**
 * Delete a user-created macro
 */
export async function deleteUserMacro(macroId: string): Promise<boolean> {
  return enqueueWrite(async () => {
    const storage = await loadStorage();
    const filtered = storage.userMacros.filter(m => m.id !== macroId);

    if (filtered.length === storage.userMacros.length) {
      return false; // Macro not found
    }

    storage.userMacros = filtered;
    return saveStorage(storage);
  });
}

// ========================================
// LocalStorage fallback for web/dev mode
// ========================================

const USER_MACROS_STORAGE_KEY = 'oaiy_user_macros_v2';

function loadStorageFromLocalStorage(): UserMacroStorage {
  try {
    const saved = localStorage.getItem(USER_MACROS_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved) as UserMacroStorage;
      return {
        overrides: parsed.overrides || {},
        userMacros: parsed.userMacros || [],
      };
    }
  } catch (error) {
    logger.warn('LocalStorage load error', { error });
  }
  return EMPTY_STORAGE;
}

function saveStorageToLocalStorage(storage: UserMacroStorage): boolean {
  try {
    localStorage.setItem(USER_MACROS_STORAGE_KEY, JSON.stringify(storage));
    return true;
  } catch (error) {
    logger.error('LocalStorage save error', { error });
    return false;
  }
}
