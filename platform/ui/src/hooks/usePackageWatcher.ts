/**
 * usePackageWatcher - Monitors loaded packages for file changes
 *
 * Periodically checks if the .oaiy file has been modified since it was loaded.
 * When changes are detected, shows a notification and offers to reload.
 */

import { useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { OAIYPackageManifest } from 'oaiy-core';
import { createLogger } from '../utils/logger';

const logger = createLogger('PackageWatcher');

interface LoadedPackage {
  manifest: OAIYPackageManifest;
  sourcePath: string;
  flows: unknown[];
}

interface PackageWatcherOptions {
  /** Loaded packages to monitor */
  loadedPackages: Map<string, LoadedPackage>;
  /** Interval in ms between checks (default: 5000) */
  checkInterval?: number;
  /** Callback when a package has changed */
  onPackageChanged: (packageId: string, manifest: OAIYPackageManifest) => void;
}

interface PackageState {
  mtime: number;
  hasNotified: boolean;
}

/**
 * Hook that monitors loaded packages for file changes.
 * Calls onPackageChanged when a package's source file has been modified.
 */
export function usePackageWatcher({
  loadedPackages,
  checkInterval = 5000,
  onPackageChanged,
}: PackageWatcherOptions): void {
  // Track the modification time of each package when it was loaded
  const packageStates = useRef<Map<string, PackageState>>(new Map());

  // Initialize state for new packages
  useEffect(() => {
    const initPackages = async () => {
      for (const [packageId, pkg] of loadedPackages.entries()) {
        // Skip if we already have state for this package
        if (packageStates.current.has(packageId)) continue;

        try {
          const mtime = await invoke<number>('get_package_mtime', {
            packagePath: pkg.sourcePath,
          });
          packageStates.current.set(packageId, {
            mtime,
            hasNotified: false,
          });
        } catch (err) {
          logger.error(`Failed to get initial mtime for ${packageId}`, { error: err });
        }
      }

      // Clean up states for packages that are no longer loaded
      for (const packageId of packageStates.current.keys()) {
        if (!loadedPackages.has(packageId)) {
          packageStates.current.delete(packageId);
        }
      }
    };

    initPackages();
  }, [loadedPackages]);

  // Check for changes periodically
  const checkForChanges = useCallback(async () => {
    for (const [packageId, pkg] of loadedPackages.entries()) {
      const state = packageStates.current.get(packageId);
      if (!state) continue;

      // Skip if we've already notified about this change
      if (state.hasNotified) continue;

      try {
        const currentMtime = await invoke<number>('get_package_mtime', {
          packagePath: pkg.sourcePath,
        });

        // Check if the file has been modified
        if (currentMtime > state.mtime) {
          logger.debug(`Package ${packageId} has changed`);

          // Mark as notified so we don't spam
          state.hasNotified = true;

          // Notify the parent
          onPackageChanged(packageId, pkg.manifest);
        }
      } catch (err) {
        // File might have been deleted or moved - that's ok
        logger.warn(`Failed to check ${packageId}`, { error: err });
      }
    }
  }, [loadedPackages, onPackageChanged]);

  // Run the check periodically
  useEffect(() => {
    if (loadedPackages.size === 0) return;

    const interval = setInterval(checkForChanges, checkInterval);
    return () => clearInterval(interval);
  }, [checkForChanges, checkInterval, loadedPackages.size]);

  // Also check when the window gains focus
  useEffect(() => {
    const handleFocus = () => {
      // Small delay to ensure file system is settled
      setTimeout(checkForChanges, 500);
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [checkForChanges]);
}

/**
 * Reset the notification state for a package after it's been reloaded.
 * Call this after reloading a package to start watching for changes again.
 */
export async function resetPackageWatchState(
  packageStates: Map<string, PackageState>,
  packageId: string,
  sourcePath: string
): Promise<void> {
  try {
    const mtime = await invoke<number>('get_package_mtime', {
      packagePath: sourcePath,
    });
    packageStates.set(packageId, {
      mtime,
      hasNotified: false,
    });
  } catch (err) {
    logger.error(`Failed to reset state for ${packageId}`, { error: err });
  }
}
