/**
 * DependencyDialog - Shows missing package dependencies and offers to load them
 *
 * Displayed when loading a package that requires other packages that aren't loaded.
 * Allows the user to browse and load each required package before continuing.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import type { OAIYPackageManifest } from 'oaiy-core';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface PackageDependency {
  id: string;
  version?: string;
}

interface DependencyStatus {
  id: string;
  requiredVersion?: string;
  status: 'missing' | 'loading' | 'loaded' | 'version_mismatch';
  loadedVersion?: string;
  error?: string;
}

interface DependencyDialogProps {
  isOpen: boolean;
  /** The package that has dependencies */
  packageName: string;
  /** List of required package dependencies */
  dependencies: PackageDependency[];
  /** Map of currently loaded packages (id -> manifest) */
  loadedPackages: Map<string, { manifest: OAIYPackageManifest }>;
  /** Called when a dependency package is loaded */
  onPackageLoaded: (path: string, manifest: OAIYPackageManifest) => Promise<void>;
  /** Called when all dependencies are satisfied */
  onAllLoaded: () => void;
  /** Called when user wants to continue anyway without all deps */
  onContinueAnyway: () => void;
  /** Called when user cancels */
  onCancel: () => void;
}

/**
 * Compare two semver version strings
 * Returns: -1 if a < b, 0 if a == b, 1 if a > b
 */
function compareVersions(a: string, b: string): number {
  const partsA = a.split('.').map(n => parseInt(n, 10) || 0);
  const partsB = b.split('.').map(n => parseInt(n, 10) || 0);

  for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
    const numA = partsA[i] || 0;
    const numB = partsB[i] || 0;
    if (numA < numB) return -1;
    if (numA > numB) return 1;
  }
  return 0;
}

/**
 * Check if a version satisfies a requirement (semver-like check)
 */
function versionSatisfies(loaded: string, required: string): boolean {
  // If no version required, any version is fine
  if (!required) return true;

  const loadedParts = loaded.split('.').map(n => parseInt(n, 10) || 0);

  // Handle >= prefix
  if (required.startsWith('>=')) {
    return compareVersions(loaded, required.slice(2)) >= 0;
  }

  // Handle ^ prefix (compatible with major version)
  // ^1.2.3 means >=1.2.3 and <2.0.0
  if (required.startsWith('^')) {
    const reqVersion = required.slice(1);
    const reqParts = reqVersion.split('.').map(n => parseInt(n, 10) || 0);

    // Major version must match (or if major is 0, minor must match)
    if (reqParts[0] === 0) {
      // For ^0.x.y, minor version must match
      if (loadedParts[0] !== 0 || loadedParts[1] !== reqParts[1]) return false;
    } else {
      // For ^x.y.z where x > 0, major must match
      if (loadedParts[0] !== reqParts[0]) return false;
    }

    return compareVersions(loaded, reqVersion) >= 0;
  }

  // Handle ~ prefix (compatible with minor version)
  // ~1.2.3 means >=1.2.3 and <1.3.0
  if (required.startsWith('~')) {
    const reqVersion = required.slice(1);
    const reqParts = reqVersion.split('.').map(n => parseInt(n, 10) || 0);

    // Major and minor must match
    if (loadedParts[0] !== reqParts[0] || loadedParts[1] !== reqParts[1]) return false;

    return compareVersions(loaded, reqVersion) >= 0;
  }

  // Exact match
  return compareVersions(loaded, required) === 0;
}

export function DependencyDialog({
  isOpen,
  packageName,
  dependencies,
  loadedPackages,
  onPackageLoaded,
  onAllLoaded,
  onContinueAnyway,
  onCancel,
}: DependencyDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, isOpen);

  // Escape closes — matches the rest of the app's modal convention.
  // Cancel is the safe default (the package isn't loaded yet, so the
  // user just bows out without partial state).
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onCancel]);

  const [statuses, setStatuses] = useState<Map<string, DependencyStatus>>(() => {
    const initial = new Map<string, DependencyStatus>();
    for (const dep of dependencies) {
      const loaded = loadedPackages.get(dep.id);
      if (loaded) {
        const versionOk = !dep.version || versionSatisfies(loaded.manifest.version, dep.version);
        initial.set(dep.id, {
          id: dep.id,
          requiredVersion: dep.version,
          status: versionOk ? 'loaded' : 'version_mismatch',
          loadedVersion: loaded.manifest.version,
        });
      } else {
        initial.set(dep.id, {
          id: dep.id,
          requiredVersion: dep.version,
          status: 'missing',
        });
      }
    }
    return initial;
  });

  // Check if all dependencies are satisfied
  const allSatisfied = Array.from(statuses.values()).every(
    s => s.status === 'loaded'
  );
  const someLoading = Array.from(statuses.values()).some(
    s => s.status === 'loading'
  );

  // Handle browsing for a package
  const handleBrowsePackage = useCallback(async (depId: string) => {
    try {
      const result = await open({
        title: `Select package: ${depId}`,
        filters: [{ name: 'OAIY Package', extensions: ['oaiy'] }],
        multiple: false,
      });

      if (!result) return;

      const path = typeof result === 'string' ? result : result;

      // Update status to loading
      setStatuses(prev => {
        const next = new Map(prev);
        const current = next.get(depId);
        if (current) {
          next.set(depId, { ...current, status: 'loading', error: undefined });
        }
        return next;
      });

      // Read the manifest
      const manifest = await invoke<OAIYPackageManifest>('read_package', {
        packagePath: path,
      });

      // Check if it's the right package
      if (manifest.id !== depId) {
        setStatuses(prev => {
          const next = new Map(prev);
          const current = next.get(depId);
          if (current) {
            next.set(depId, {
              ...current,
              status: 'missing',
              error: `Wrong package: expected "${depId}" but got "${manifest.id}"`,
            });
          }
          return next;
        });
        return;
      }

      // Check version
      const dep = dependencies.find(d => d.id === depId);
      if (dep?.version && !versionSatisfies(manifest.version, dep.version)) {
        setStatuses(prev => {
          const next = new Map(prev);
          next.set(depId, {
            id: depId,
            requiredVersion: dep.version,
            status: 'version_mismatch',
            loadedVersion: manifest.version,
            error: `Version ${manifest.version} doesn't satisfy requirement ${dep.version}`,
          });
          return next;
        });
        return;
      }

      // Load the package
      await onPackageLoaded(path, manifest);

      // Update status to loaded
      setStatuses(prev => {
        const next = new Map(prev);
        next.set(depId, {
          id: depId,
          requiredVersion: dep?.version,
          status: 'loaded',
          loadedVersion: manifest.version,
        });
        return next;
      });
    } catch (err) {
      setStatuses(prev => {
        const next = new Map(prev);
        const current = next.get(depId);
        if (current) {
          next.set(depId, {
            ...current,
            status: 'missing',
            error: String(err),
          });
        }
        return next;
      });
    }
  }, [dependencies, onPackageLoaded]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-700/50 w-full max-w-lg shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dep-dialog-title"
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-purple-600/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <div>
              <h3 id="dep-dialog-title" className="text-sm font-semibold text-slate-900 dark:text-slate-100">Required Packages</h3>
              <p className="text-xs text-slate-400">
                "{packageName}" needs the following packages
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-5 py-4">
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">
            Load the required packages to ensure all features work correctly.
          </p>

          {/* Dependency list */}
          <div className="space-y-2 mb-4 max-h-64 overflow-y-auto">
            {Array.from(statuses.values()).map((dep) => (
              <div
                key={dep.id}
                className={`
                  flex items-center gap-3 p-3 rounded-lg border
                  ${dep.status === 'loaded'
                    ? 'bg-green-50 dark:bg-green-950/30 border-green-500/30'
                    : dep.status === 'version_mismatch'
                    ? 'bg-amber-50 dark:bg-amber-950/30 border-amber-500/30'
                    : dep.status === 'loading'
                    ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-500/30'
                    : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700/30'
                  }
                `}
              >
                {/* Status indicator */}
                <div
                  className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                    dep.status === 'loaded'
                      ? 'bg-green-400'
                      : dep.status === 'loading'
                      ? 'bg-blue-400 animate-pulse'
                      : dep.status === 'version_mismatch'
                      ? 'bg-amber-400'
                      : 'bg-slate-500'
                  }`}
                />

                {/* Package info */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-700 dark:text-slate-200">{dep.id}</div>
                  <div className="text-xs text-slate-500">
                    {dep.status === 'loaded'
                      ? `v${dep.loadedVersion} loaded`
                      : dep.status === 'loading'
                      ? 'Loading...'
                      : dep.status === 'version_mismatch'
                      ? `v${dep.loadedVersion} loaded, needs ${dep.requiredVersion || 'different version'}`
                      : dep.requiredVersion
                      ? `Requires ${dep.requiredVersion}`
                      : 'Not loaded'}
                  </div>
                  {dep.error && (
                    <div className="text-xs text-red-400 mt-1">{dep.error}</div>
                  )}
                </div>

                {/* Action button */}
                {(dep.status === 'missing' || dep.status === 'version_mismatch') && (
                  <button
                    onClick={() => handleBrowsePackage(dep.id)}
                    className="px-3 py-1.5 text-xs font-medium bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded transition-colors"
                  >
                    Browse...
                  </button>
                )}
                {dep.status === 'loaded' && (
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
            ))}
          </div>

          {/* Success message */}
          {allSatisfied && (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-950/30 border border-green-500/30 rounded-lg">
              <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-sm text-green-300">All dependencies loaded!</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-200 dark:border-slate-700/50 flex items-center justify-between">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
          >
            Cancel
          </button>

          <div className="flex items-center justify-end gap-2">
            {!allSatisfied && (
              <button
                onClick={onContinueAnyway}
                className="px-4 py-2 text-sm font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              >
                Continue Anyway
              </button>
            )}

            <button
              onClick={onAllLoaded}
              disabled={!allSatisfied || someLoading}
              className="px-4 py-2 text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {allSatisfied ? 'Continue' : 'Load All Required'}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
