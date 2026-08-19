/**
 * PackageBrowser - Browse and discover packages from configured sources
 *
 * Allows users to:
 * - Add package source directories
 * - Scan for available packages
 * - Load packages from the browser
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { open } from '@tauri-apps/plugin-dialog';
import { usePackageRegistry, type DiscoveredPackage } from '../../hooks/usePackageRegistry';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { packageLogger as logger } from '../../utils/logger';

interface PackageBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onLoadPackage: (path: string) => void;
  loadedPackageIds: Set<string>;
}

export function PackageBrowser({
  isOpen,
  onClose,
  onLoadPackage,
  loadedPackageIds,
}: PackageBrowserProps) {
  const {
    packages,
    sources,
    scanning,
    addSource,
    removeSource,
    toggleSource,
    scanSources,
  } = usePackageRegistry();

  const [showSources, setShowSources] = useState(false);
  const hasAutoScanned = useRef(false);
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, isOpen);

  // Escape closes — matches the rest of the app's modal convention.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  // Auto-scan when browser opens if there are sources but no packages
  useEffect(() => {
    if (isOpen && sources.length > 0 && packages.length === 0 && !scanning && !hasAutoScanned.current) {
      hasAutoScanned.current = true;
      scanSources();
    }
    // Reset auto-scan flag when browser closes
    if (!isOpen) {
      hasAutoScanned.current = false;
    }
  }, [isOpen, sources.length, packages.length, scanning, scanSources]);

  // Handle adding a new source directory
  const handleAddSource = useCallback(async () => {
    try {
      const result = await open({
        title: 'Select Package Directory',
        directory: true,
        multiple: false,
      });

      if (result && typeof result === 'string') {
        await addSource(result);
      }
    } catch (err) {
      logger.error('Failed to add source', { error: err });
    }
  }, [addSource]);

  // Handle loading a package
  const handleLoad = useCallback((pkg: DiscoveredPackage) => {
    onLoadPackage(pkg.path);
    onClose();
  }, [onLoadPackage, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-x-0 bottom-0 top-[65px] md:top-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-0 sm:p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={panelRef}
        className="bg-white dark:bg-slate-900 rounded-none sm:rounded-2xl border-0 sm:border border-slate-200 dark:border-slate-700/50 w-full h-full sm:h-auto sm:max-w-2xl shadow-2xl sm:max-h-[80vh] short:sm:max-h-[92vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pkg-browser-title"
      >
        {/* Header */}
        <div className="px-5 py-4 short:px-4 short:py-2.5 border-b border-slate-200 dark:border-slate-700/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-purple-600/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3V7M4 7l8-4 8 4M4 7l8 4m8-4l-8 4m0 0v10" />
              </svg>
            </div>
            <div>
              <h3 id="pkg-browser-title" className="text-sm font-semibold text-slate-900 dark:text-slate-100">Package Browser</h3>
              <p className="text-xs text-slate-400">
                {packages.length} package{packages.length !== 1 ? 's' : ''} available
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close package browser"
            className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded transition-colors text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Toolbar */}
        <div className="px-5 py-3 border-b border-slate-200 dark:border-slate-700/50 flex items-center gap-2">
          <button
            onClick={handleAddSource}
            className="px-3 py-1.5 text-xs font-medium bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded transition-colors flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Source
          </button>
          <button
            onClick={scanSources}
            disabled={scanning || sources.length === 0}
            className="px-3 py-1.5 text-xs font-medium bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded transition-colors flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {scanning ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            )}
            Refresh
          </button>
          <div className="flex-1" />
          <button
            type="button"
            aria-label="Toggle package sources panel"
            aria-expanded={showSources}
            onClick={() => setShowSources(!showSources)}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 ${
              showSources
                ? 'bg-slate-200 dark:bg-slate-600 text-slate-900 dark:text-white'
                : 'bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300'
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            Sources ({sources.length})
          </button>
        </div>

        {/* Sources Panel (collapsible) */}
        {showSources && sources.length > 0 && (
          <div className="px-5 py-3 border-b border-slate-200 dark:border-slate-700/50 bg-slate-50 dark:bg-slate-800/50">
            <div className="text-xs font-medium text-slate-400 mb-2">Package Sources</div>
            <div className="space-y-1.5">
              {sources.map(source => (
                <div
                  key={source.id}
                  className="flex items-center gap-2 p-2 bg-slate-100 dark:bg-slate-700/50 rounded text-xs"
                >
                  <input
                    type="checkbox"
                    checked={source.enabled}
                    onChange={(e) => toggleSource(source.id, e.target.checked)}
                    className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-purple-500 focus:ring-purple-500"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-slate-700 dark:text-slate-200 truncate">{source.name}</div>
                    <div className="text-slate-500 truncate">{source.path}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeSource(source.id)}
                    aria-label={`Remove source ${source.name}`}
                    title={`Remove source ${source.name}`}
                    className="p-1 hover:bg-slate-200 dark:hover:bg-slate-600 rounded text-slate-400 hover:text-red-400 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Package List */}
        <div className="flex-1 overflow-y-auto px-5 py-4 short:px-4 short:py-2.5">
          {sources.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              <svg className="w-12 h-12 mx-auto mb-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <p className="text-sm">No package sources configured</p>
              <p className="text-xs text-slate-500 mt-1">
                Click "Add Source" to add a directory containing .oaiy packages
              </p>
            </div>
          ) : packages.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              <svg className="w-12 h-12 mx-auto mb-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              <p className="text-sm">No packages found</p>
              <p className="text-xs text-slate-500 mt-1">
                Click "Refresh" to scan your sources for packages
              </p>
            </div>
          ) : (
            <div className="grid gap-3">
              {packages.map(pkg => {
                const isLoaded = loadedPackageIds.has(pkg.manifest.id);
                return (
                  <div
                    key={pkg.path}
                    className={`p-4 rounded-lg border ${
                      isLoaded
                        ? 'bg-green-50 dark:bg-green-950/20 border-green-500/30'
                        : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700/30 hover:border-slate-300 dark:hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-lg bg-purple-600/20 flex items-center justify-center flex-shrink-0">
                        <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-slate-700 dark:text-slate-200">{pkg.manifest.name}</span>
                          <span className="text-xs text-slate-500">v{pkg.manifest.version}</span>
                          {isLoaded && (
                            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-green-500/20 text-green-400 rounded">
                              Loaded
                            </span>
                          )}
                        </div>
                        {pkg.manifest.description && (
                          <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                            {pkg.manifest.description}
                          </p>
                        )}
                        <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                          {pkg.manifest.author && (
                            <span>by {pkg.manifest.author}</span>
                          )}
                          {pkg.manifest.flows && (
                            <span>{pkg.manifest.flows.length} flow{pkg.manifest.flows.length !== 1 ? 's' : ''}</span>
                          )}
                          {pkg.manifest.services && pkg.manifest.services.length > 0 && (
                            <span>{pkg.manifest.services.length} service{pkg.manifest.services.length !== 1 ? 's' : ''}</span>
                          )}
                          {pkg.manifest.nodes && pkg.manifest.nodes.length > 0 && (
                            <span className="text-purple-400">
                              {pkg.manifest.nodes.length} custom node{pkg.manifest.nodes.length !== 1 ? 's' : ''}
                            </span>
                          )}
                        </div>
                        {/* Custom Nodes Preview */}
                        {pkg.manifest.nodes && pkg.manifest.nodes.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700/50">
                            <div className="flex items-center gap-1.5 text-[10px] text-purple-400 font-medium mb-1.5">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                              </svg>
                              Custom Nodes
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {pkg.manifest.nodes.map((nodeModule, idx) => (
                                <span
                                  key={idx}
                                  className="px-1.5 py-0.5 text-[10px] bg-purple-500/10 text-purple-300 rounded border border-purple-500/20"
                                  title={`Module path: ${nodeModule.path}`}
                                >
                                  {nodeModule.path.split('/').pop() || nodeModule.path}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      {!isLoaded && (
                        <button
                          onClick={() => handleLoad(pkg)}
                          className="px-3 py-1.5 text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white rounded transition-colors"
                        >
                          Load
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 short:px-4 short:py-2.5 border-t border-slate-200 dark:border-slate-700/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
