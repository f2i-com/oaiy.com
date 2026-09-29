/**
 * DependencyDialog - Shows missing package dependencies and offers to load them
 *
 * Displayed when loading a package that requires other packages that aren't loaded.
 * Allows the user to browse and load each required package before continuing.
 */

import { useState, useCallback } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import { Check, FolderOpen, Layers } from 'lucide-react';
import type { OAIYPackageManifest } from 'oaiy-core';
import Dialog from '../ui/Dialog';

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
  // Escape and the overlay cancel — the safe default (the package isn't
  // loaded yet, so the user just bows out without partial state).

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

  const pill = (status: DependencyStatus['status']) =>
    status === 'loaded' ? 'oaiy-pill dot ok'
      : status === 'loading' ? 'oaiy-pill dot accent live'
      : status === 'version_mismatch' ? 'oaiy-pill dot warn'
      : 'oaiy-pill dot';
  const said = (dep: DependencyStatus) =>
    dep.status === 'loaded' ? 'loaded'
      : dep.status === 'loading' ? 'loading'
      : dep.status === 'version_mismatch' ? 'wrong version'
      : 'missing';

  return (
    <Dialog
      open
      onClose={onCancel}
      title="It needs other packages"
      description={`“${packageName}” uses these packages. Load them so all of it works.`}
      icon={<Layers size={16} />}
      tone="accent"
      size="lg"
      footer={
        <>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          {!allSatisfied && (
            <button type="button" onClick={onContinueAnyway} className="btn btn-ghost">
              Continue without them
            </button>
          )}
          <button
            type="button"
            onClick={onAllLoaded}
            disabled={!allSatisfied || someLoading}
            className="btn btn-primary"
          >
            {allSatisfied ? 'Continue' : 'Load all of them first'}
          </button>
        </>
      }
    >
      <ul className="oaiy-rows m-0 max-h-72 list-none overflow-y-auto rounded-[var(--r-ctl)] border border-edge-primary p-0">
        {Array.from(statuses.values()).map((dep) => (
          <li key={dep.id} className="oaiy-row top">
            <div className="oaiy-row-main">
              <span className="oaiy-row-title">{dep.id}</span>
              <span className="oaiy-row-meta">
                {dep.status === 'loaded'
                  ? `v${dep.loadedVersion} loaded`
                  : dep.status === 'loading'
                  ? 'Loading…'
                  : dep.status === 'version_mismatch'
                  ? `v${dep.loadedVersion} loaded, needs ${dep.requiredVersion || 'a different version'}`
                  : dep.requiredVersion
                  ? `Needs ${dep.requiredVersion}`
                  : 'Not loaded'}
              </span>
              {dep.error && <span className="oaiy-error-text">{dep.error}</span>}
            </div>
            <span className={pill(dep.status)}>{said(dep)}</span>
            {(dep.status === 'missing' || dep.status === 'version_mismatch') && (
              <button type="button" onClick={() => handleBrowsePackage(dep.id)} className="btn btn-sm">
                <FolderOpen size={12} /> Find it…
              </button>
            )}
          </li>
        ))}
      </ul>

      {allSatisfied && (
        <div className="oaiy-banner ok">
          <Check size={15} className="mt-0.5 shrink-0" />
          <span className="flex-1">Every package it needs is loaded.</span>
        </div>
      )}
    </Dialog>
  );
}
