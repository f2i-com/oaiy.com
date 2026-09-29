/**
 * Packages — flow packages (.oaiy): flows, macros and nodes bundled to share.
 * A page in the editor's main area: the packages loaded now, the folders
 * searched for more (sources), and what those folders hold.
 *
 * Not the dashboard's Plugins (supervised extensions that add connectors and
 * events to OAIY): a package only brings flows, and the nodes they use, into
 * this editor.
 */

import { useCallback, useEffect, useRef } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { FolderOpen, FolderPlus, Package, RefreshCw, Trash2, Upload, X } from 'lucide-react';
import { usePackageRegistry, type DiscoveredPackage } from '../../hooks/usePackageRegistry';
import { packageLogger as logger } from '../../utils/logger';
import SectionPage, { Card, EmptyState } from '../chrome/SectionPage';

export interface LoadedPackageSummary {
  id: string;
  name: string;
  version: string;
  flows: number;
}

interface PackageBrowserProps {
  /** Load a package found in a source folder. */
  onLoadPackage: (path: string) => void;
  loadedPackageIds: Set<string>;
  /** The packages loaded now, to list and close. */
  loaded?: LoadedPackageSummary[];
  onClosePackage?: (id: string) => void;
  /** Pick a package (or flow) file to load: the Import menu's own action. */
  onLoadFile?: () => void;
}

export function PackageBrowser({
  onLoadPackage,
  loadedPackageIds,
  loaded = [],
  onClosePackage,
  onLoadFile,
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

  const hasAutoScanned = useRef(false);

  // Scan once on arrival when there are sources but nothing found yet.
  useEffect(() => {
    if (sources.length > 0 && packages.length === 0 && !scanning && !hasAutoScanned.current) {
      hasAutoScanned.current = true;
      scanSources();
    }
  }, [sources.length, packages.length, scanning, scanSources]);

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

  const handleLoad = useCallback((pkg: DiscoveredPackage) => {
    onLoadPackage(pkg.path);
  }, [onLoadPackage]);

  return (
    <SectionPage
      kicker="Flows"
      title="Packages"
      description="Flow packages (.oaiy) bundle flows, macros and the nodes they use. Load one from a file, or from a folder you add as a source."
      actions={
        <>
          {onLoadFile && (
            <button type="button" className="btn" onClick={onLoadFile}>
              <Upload size={14} /> Load a file…
            </button>
          )}
          <button type="button" className="btn btn-primary" onClick={handleAddSource}>
            <FolderPlus size={14} /> Add a source
          </button>
        </>
      }
      testId="packages-page"
    >
      <Card title="Loaded" count={loaded.length} flush={loaded.length > 0}>
        {loaded.length === 0 ? (
          <p className="oaiy-card-text">
            None yet. A package you load shows its flows at the top of the flows rail, and its nodes in the palette.
          </p>
        ) : (
          <ul className="oaiy-rows m-0 list-none p-0">
            {loaded.map((pkg) => (
              <li key={pkg.id} className="oaiy-row">
                <Package size={16} className="shrink-0 text-signal-magenta" />
                <div className="oaiy-row-main">
                  <span className="oaiy-row-title">{pkg.name}</span>
                  <span className="oaiy-row-meta">
                    v{pkg.version} · {pkg.flows} flow{pkg.flows === 1 ? '' : 's'}
                  </span>
                </div>
                {onClosePackage && (
                  <button type="button" className="btn btn-sm" onClick={() => onClosePackage(pkg.id)} title={`Close ${pkg.name}`}>
                    <X size={12} /> Close
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card
        title="Sources"
        count={sources.length}
        flush={sources.length > 0}
        actions={
          <button
            type="button"
            className="btn btn-sm"
            onClick={scanSources}
            disabled={scanning || sources.length === 0}
            title="Look through the sources again"
          >
            <RefreshCw size={12} className={scanning ? 'animate-spin' : undefined} />
            {scanning ? 'Looking…' : 'Look again'}
          </button>
        }
      >
        {sources.length === 0 ? (
          <p className="oaiy-card-text">
            A source is a folder of .oaiy packages. Add one and the packages in it are listed below, ready to load.
          </p>
        ) : (
          <ul className="oaiy-rows m-0 list-none p-0">
            {sources.map((source) => (
              <li key={source.id} className="oaiy-row">
                <input
                  type="checkbox"
                  checked={source.enabled}
                  onChange={(e) => toggleSource(source.id, e.target.checked)}
                  aria-label={`Search ${source.name}`}
                  title={source.enabled ? 'Searched' : 'Not searched'}
                  className="h-4 w-4 shrink-0"
                />
                <FolderOpen size={15} className="shrink-0 text-content-faint" />
                <div className="oaiy-row-main">
                  <span className="oaiy-row-title">{source.name}</span>
                  <span className="oaiy-row-meta font-mono">{source.path}</span>
                </div>
                <button
                  type="button"
                  onClick={() => removeSource(source.id)}
                  aria-label={`Remove source ${source.name}`}
                  title={`Remove source ${source.name}`}
                  className="oaiy-icon-btn"
                >
                  <Trash2 size={14} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {sources.length > 0 && (
        <Card title="In your sources" count={packages.length} flush={packages.length > 0}>
          {packages.length === 0 ? (
            <div className="p-4">
              <EmptyState icon={<Package size={24} />} title={scanning ? 'Looking…' : 'No packages found'}>
                {scanning ? 'Reading the source folders.' : 'None of the sources holds a .oaiy package. Look again after adding one.'}
              </EmptyState>
            </div>
          ) : (
            <ul className="oaiy-rows m-0 list-none p-0">
              {packages.map((pkg) => {
                const isLoaded = loadedPackageIds.has(pkg.manifest.id);
                const meta = [
                  `v${pkg.manifest.version}`,
                  pkg.manifest.author ? `by ${pkg.manifest.author}` : null,
                  pkg.manifest.flows ? `${pkg.manifest.flows.length} flow${pkg.manifest.flows.length !== 1 ? 's' : ''}` : null,
                  pkg.manifest.services && pkg.manifest.services.length > 0
                    ? `${pkg.manifest.services.length} service${pkg.manifest.services.length !== 1 ? 's' : ''}`
                    : null,
                  pkg.manifest.nodes && pkg.manifest.nodes.length > 0
                    ? `${pkg.manifest.nodes.length} node${pkg.manifest.nodes.length !== 1 ? 's' : ''}`
                    : null,
                ].filter(Boolean).join(' · ');
                return (
                  <li key={pkg.path} className="oaiy-row top">
                    <Package size={16} className="mt-0.5 shrink-0 text-signal-magenta" />
                    <div className="oaiy-row-main">
                      <span className="oaiy-row-title">{pkg.manifest.name}</span>
                      {pkg.manifest.description && (
                        <span className="text-[12px] leading-snug text-content-secondary">{pkg.manifest.description}</span>
                      )}
                      <span className="oaiy-row-meta">{meta}</span>
                      {pkg.manifest.nodes && pkg.manifest.nodes.length > 0 && (
                        <div className="oaiy-chips mt-1">
                          {pkg.manifest.nodes.map((nodeModule, idx) => (
                            <span key={idx} title={`Module path: ${nodeModule.path}`}>
                              {nodeModule.path.split('/').pop() || nodeModule.path}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    {isLoaded ? (
                      <span className="oaiy-pill dot ok">loaded</span>
                    ) : (
                      <button type="button" onClick={() => handleLoad(pkg)} className="btn btn-primary btn-sm">
                        Load
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      )}
    </SectionPage>
  );
}
