/**
 * usePackageRegistry - Local package discovery and registry management
 *
 * Provides functionality to:
 * - Scan directories for .oaiy package files
 * - Track discovered packages
 * - Remember package sources
 */

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { OAIYPackageManifest } from 'oaiy-core';
import { packageLogger as logger } from '../utils/logger';

export interface DiscoveredPackage {
  manifest: OAIYPackageManifest;
  path: string;
  source: string; // Directory it was found in
  discoveredAt: number; // Timestamp
}

export interface PackageSource {
  id: string;
  path: string;
  name: string;
  enabled: boolean;
  lastScanned?: number;
}

interface UsePackageRegistryReturn {
  /** Discovered packages from all sources */
  packages: DiscoveredPackage[];
  /** Configured package sources */
  sources: PackageSource[];
  /** Whether currently scanning */
  scanning: boolean;
  /** Add a new source directory */
  addSource: (path: string, name?: string) => Promise<void>;
  /** Remove a source */
  removeSource: (id: string) => void;
  /** Enable/disable a source */
  toggleSource: (id: string, enabled: boolean) => void;
  /** Scan all enabled sources for packages */
  scanSources: () => Promise<void>;
  /** Scan a specific source */
  scanSource: (sourceId: string) => Promise<DiscoveredPackage[]>;
  /** Clear all discovered packages */
  clearPackages: () => void;
}

const STORAGE_KEY = 'oaiy-package-sources';

/**
 * Generate a simple ID
 */
function generateId(): string {
  return Math.random().toString(36).substring(2, 10);
}

/**
 * Load sources from localStorage
 */
function loadSources(): PackageSource[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (err) {
    logger.error('Failed to load sources', { error: err });
  }
  return [];
}

/**
 * Save sources to localStorage
 */
function saveSources(sources: PackageSource[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sources));
  } catch (err) {
    logger.error('Failed to save sources', { error: err });
  }
}

/**
 * Hook for managing a local package registry
 */
export function usePackageRegistry(): UsePackageRegistryReturn {
  const [packages, setPackages] = useState<DiscoveredPackage[]>([]);
  const [sources, setSources] = useState<PackageSource[]>(() => loadSources());
  const [scanning, setScanning] = useState(false);

  // Persist sources when they change
  useEffect(() => {
    saveSources(sources);
  }, [sources]);

  // Scan a directory for .oaiy files
  const scanDirectory = useCallback(async (dirPath: string): Promise<DiscoveredPackage[]> => {
    try {
      const files = await invoke<string[]>('scan_directory_for_packages', {
        directoryPath: dirPath,
      });

      const discovered: DiscoveredPackage[] = [];

      for (const filePath of files) {
        try {
          const manifest = await invoke<OAIYPackageManifest>('read_package', {
            packagePath: filePath,
          });
          discovered.push({
            manifest,
            path: filePath,
            source: dirPath,
            discoveredAt: Date.now(),
          });
        } catch (err) {
          logger.warn(`Failed to read package ${filePath}`, { error: err });
        }
      }

      return discovered;
    } catch (err) {
      logger.error(`Failed to scan directory ${dirPath}`, { error: err });
      return [];
    }
  }, []);

  // Add a new source
  const addSource = useCallback(async (path: string, name?: string) => {
    const source: PackageSource = {
      id: generateId(),
      path,
      name: name || path.split(/[\\/]/).pop() || 'Unnamed Source',
      enabled: true,
    };

    setSources(prev => [...prev, source]);

    // Scan the new source
    const discovered = await scanDirectory(path);
    if (discovered.length > 0) {
      setPackages(prev => {
        // Remove duplicates by path
        const existingPaths = new Set(prev.map(p => p.path));
        const newPackages = discovered.filter(p => !existingPaths.has(p.path));
        return [...prev, ...newPackages];
      });
    }

    // Update last scanned time
    setSources(prev =>
      prev.map(s =>
        s.id === source.id ? { ...s, lastScanned: Date.now() } : s
      )
    );
  }, [scanDirectory]);

  // Remove a source
  const removeSource = useCallback((id: string) => {
    setSources(prev => {
      const source = prev.find(s => s.id === id);
      if (source) {
        // Remove packages from this source
        setPackages(pkgs => pkgs.filter(p => p.source !== source.path));
      }
      return prev.filter(s => s.id !== id);
    });
  }, []);

  // Toggle a source's enabled state
  const toggleSource = useCallback((id: string, enabled: boolean) => {
    setSources(prev => {
      const source = prev.find(s => s.id === id);
      if (source && !enabled) {
        // When disabling, remove packages from this source
        setPackages(pkgs => pkgs.filter(p => p.source !== source.path));
      }
      return prev.map(s => (s.id === id ? { ...s, enabled } : s));
    });
  }, []);

  // Scan a specific source
  const scanSource = useCallback(async (sourceId: string): Promise<DiscoveredPackage[]> => {
    const source = sources.find(s => s.id === sourceId);
    if (!source) return [];

    setScanning(true);
    try {
      const discovered = await scanDirectory(source.path);

      // Update packages - remove old ones from this source and add new
      setPackages(prev => {
        const withoutSource = prev.filter(p => p.source !== source.path);
        return [...withoutSource, ...discovered];
      });

      // Update last scanned time
      setSources(prev =>
        prev.map(s =>
          s.id === sourceId ? { ...s, lastScanned: Date.now() } : s
        )
      );

      return discovered;
    } finally {
      setScanning(false);
    }
  }, [sources, scanDirectory]);

  // Scan all enabled sources
  const scanSources = useCallback(async () => {
    setScanning(true);
    try {
      const enabledSources = sources.filter(s => s.enabled);
      const allDiscovered: DiscoveredPackage[] = [];

      for (const source of enabledSources) {
        const discovered = await scanDirectory(source.path);
        allDiscovered.push(...discovered);

        // Update last scanned time
        setSources(prev =>
          prev.map(s =>
            s.id === source.id ? { ...s, lastScanned: Date.now() } : s
          )
        );
      }

      // Deduplicate by path
      const uniquePackages = new Map<string, DiscoveredPackage>();
      for (const pkg of allDiscovered) {
        uniquePackages.set(pkg.path, pkg);
      }

      setPackages(Array.from(uniquePackages.values()));
    } finally {
      setScanning(false);
    }
  }, [sources, scanDirectory]);

  // Clear all packages
  const clearPackages = useCallback(() => {
    setPackages([]);
  }, []);

  return {
    packages,
    sources,
    scanning,
    addSource,
    removeSource,
    toggleSource,
    scanSources,
    scanSource,
    clearPackages,
  };
}
