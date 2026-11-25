/**
 * PackageManager - UI for managing .oaiy packages
 *
 * Lists installed packages, allows opening, installing, and uninstalling.
 */

import { useState, useMemo, useCallback, useRef } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, Loader2, Package, Plus, Search, X } from 'lucide-react';
import { usePackages, type PackageInstallLimits } from '../../hooks/usePackages';
import { open } from '@tauri-apps/plugin-dialog';
import { PackageCard } from './PackageCard';
import { TrustDialog } from './TrustDialog';
import type { OAIYPackageManifest, PackagePermission } from 'oaiy-core';
import { packageLogger as logger } from '../../utils/logger';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';

function formatMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface PackageManagerProps {
  disabled?: boolean;
  onPackageOpen?: (packageId: string) => void;
}

const ITEMS_PER_PAGE = 6;

export function PackageManager({ disabled, onPackageOpen }: PackageManagerProps) {
  const {
    packages,
    loading,
    installing,
    error,
    readPackage,
    peekPackageSize,
    installPackage,
    uninstallPackage,
    openPackage,
  } = usePackages();

  const [expanded, setExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const confirm = useConfirmDialog();

  // Trust dialog state
  const [pendingInstall, setPendingInstall] = useState<{
    path: string;
    manifest: OAIYPackageManifest;
  } | null>(null);

  // Filter packages based on search
  const filteredPackages = useMemo(() => {
    if (!searchQuery.trim()) return packages;
    const query = searchQuery.toLowerCase();
    return packages.filter(
      (pkg) =>
        pkg.name.toLowerCase().includes(query) ||
        pkg.id.toLowerCase().includes(query) ||
        pkg.description?.toLowerCase().includes(query)
    );
  }, [packages, searchQuery]);

  // Pagination
  const totalPages = Math.ceil(filteredPackages.length / ITEMS_PER_PAGE);
  const safePage = totalPages > 0 ? Math.min(currentPage, totalPages) : 1;
  const paginatedPackages = useMemo(() => {
    const start = (safePage - 1) * ITEMS_PER_PAGE;
    return filteredPackages.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredPackages, safePage]);

  // Handle file picker for installing a package
  const handleBrowsePackage = useCallback(async () => {
    try {
      const result = await open({
        title: 'Select .oaiy Package',
        filters: [{ name: 'OAIY Package', extensions: ['oaiy'] }],
        multiple: false,
      });

      if (!result) return;

      const path = typeof result === 'string' ? result : result;

      // Read the package manifest
      const manifest = await readPackage(path);
      if (!manifest) return;

      // Show trust dialog
      setPendingInstall({ path, manifest });
    } catch (err) {
      logger.error('Browse failed', { error: err });
    }
  }, [readPackage]);

  // Handle trust dialog confirmation. We pass the *exact* permission
  // subset the user ticked through to the native installer — granting
  // any permission marks the package as Trusted (or Verified, if the
  // signature checks out), but the granted set itself is the authority
  // on what the package can actually do.
  const handleTrustConfirm = useCallback(
    async (grantedPermissions: PackagePermission[]) => {
      if (!pendingInstall) return;

      // Inspect the archive's declared sizes before extraction. Oversize
      // packages require an explicit user override; anything beyond the
      // hard ceiling is refused outright.
      const sizeReport = await peekPackageSize(pendingInstall.path);
      let overrideLimits: PackageInstallLimits | null = null;
      if (sizeReport) {
        if (sizeReport.exceedsHardMax) {
          // Hard reject — both buttons no-op, but using the confirm
          // dialog keeps the styling consistent with the rest of the
          // install flow rather than dropping back to a native alert.
          await confirm({
            title: 'Package too large to install',
            message:
              `Total: ${formatMB(sizeReport.totalUncompressed)} (max ${formatMB(sizeReport.hardMaxTotal)}). ` +
              `Largest file: ${formatMB(sizeReport.maxFileSize)} (max ${formatMB(sizeReport.hardMaxFile)}). ` +
              `Entries: ${sizeReport.entries.toLocaleString()} (max ${sizeReport.hardMaxEntries.toLocaleString()}).`,
            variant: 'danger',
            confirmLabel: 'OK',
            cancelLabel: 'Close',
          });
          setPendingInstall(null);
          return;
        }
        if (!sizeReport.withinDefaults) {
          const ok = await confirm({
            title: 'Package exceeds default install limits',
            message:
              `Total: ${formatMB(sizeReport.totalUncompressed)} (default ${formatMB(sizeReport.defaultMaxTotal)}). ` +
              `Largest file: ${formatMB(sizeReport.maxFileSize)} (default ${formatMB(sizeReport.defaultMaxFile)}). ` +
              `Entries: ${sizeReport.entries.toLocaleString()} (default ${sizeReport.defaultMaxEntries.toLocaleString()}). ` +
              `Allow this package to install with raised limits?`,
            variant: 'warning',
            confirmLabel: 'Install anyway',
          });
          if (!ok) {
            setPendingInstall(null);
            return;
          }
          overrideLimits = {
            maxTotalUncompressed: sizeReport.totalUncompressed,
            maxFileSize: sizeReport.maxFileSize,
            maxEntries: sizeReport.entries,
          };
        }
      }

      const trust = grantedPermissions.length > 0;
      await installPackage(
        pendingInstall.path,
        trust,
        grantedPermissions,
        overrideLimits,
      );

      // Close the dialog regardless of success/failure so that, on failure,
      // the list-area error block is no longer occluded by the TrustDialog.
      setPendingInstall(null);
    },
    [pendingInstall, peekPackageSize, installPackage, confirm]
  );

  // Handle trust dialog cancel
  const handleTrustCancel = useCallback(() => {
    setPendingInstall(null);
  }, []);

  // Handle package open
  const handleOpen = useCallback(
    async (packageId: string) => {
      const pkg = await openPackage(packageId);
      if (pkg && onPackageOpen) {
        onPackageOpen(packageId);
      }
    },
    [openPackage, onPackageOpen]
  );

  // Handle package uninstall
  const handleUninstall = useCallback(
    async (packageId: string) => {
      const ok = await confirm({
        title: 'Uninstall package?',
        message: `"${packageId}" will be removed from this installation. Its flows, services, and assets are deleted from disk.`,
        variant: 'danger',
        confirmLabel: 'Uninstall',
      });
      if (!ok) return;
      await uninstallPackage(packageId);
    },
    [uninstallPackage, confirm]
  );

  // Count installed packages
  const installedCount = packages.length;

  // Header with collapse toggle
  const header = (
    <button
      type="button"
      className="flex items-center justify-between w-full px-3 py-2 text-left hover:bg-surface-tertiary rounded-[var(--r-card)] transition-colors"
      onClick={() => setExpanded(!expanded)}
      disabled={disabled}
      aria-expanded={expanded}
    >
      <div className="flex items-center gap-2">
        <Package size={15} className="text-accent shrink-0" aria-hidden="true" />
        <span className="text-[13px] font-semibold text-content-primary">Packages</span>
        {installedCount > 0 && (
          <span className="oaiy-pill accent">{installedCount}</span>
        )}
      </div>
      <ChevronDown
        size={15}
        className={`text-content-faint transition-transform ${expanded ? 'rotate-180' : ''}`}
        aria-hidden="true"
      />
    </button>
  );

  if (!expanded) {
    return (
      <div className="bg-surface-secondary rounded-[var(--r-card)] border border-edge-primary">
        {header}
      </div>
    );
  }

  return (
    <div className="bg-surface-secondary rounded-[var(--r-card)] border border-edge-primary">
      {header}

      <div className="px-3 pb-3 space-y-3">
        {/* Search and Add */}
        <div className="flex items-center gap-2">
          <div className="oaiy-search flex-1">
            <Search size={14} />
            <input
              ref={searchInputRef}
              type="search"
              placeholder="Search packages"
              aria-label="Search packages by name, ID, or description"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Escape' && searchQuery) {
                  e.stopPropagation();
                  setSearchQuery('');
                  setCurrentPage(1);
                }
              }}
              className="oaiy-input oaiy-input-sm"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('');
                  setCurrentPage(1);
                  searchInputRef.current?.focus();
                }}
                aria-label="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>
          <button
            type="button"
            onClick={handleBrowsePackage}
            disabled={disabled}
            aria-label="Install package"
            className="btn btn-primary btn-sm"
          >
            <Plus size={14} aria-hidden="true" />
            Install
          </button>
        </div>

        {/* Error display */}
        {error && (
          <div className="oaiy-banner">
            <span className="flex-1">{error}</span>
          </div>
        )}

        {/* Loading state */}
        {loading ? (
          <div className="flex items-center justify-center py-8" role="status" aria-label="Loading packages">
            <Loader2 size={20} className="text-accent animate-spin" aria-hidden="true" />
          </div>
        ) : packages.length === 0 ? (
          <div className="oaiy-empty bare py-6">
            <Package size={24} aria-hidden="true" />
            <p className="oaiy-empty-title">No packages installed</p>
            <p className="oaiy-empty-text">Click Install to add a .oaiy package.</p>
          </div>
        ) : filteredPackages.length === 0 ? (
          <div className="oaiy-empty bare py-6">
            <Search size={20} aria-hidden="true" />
            <p className="oaiy-empty-title">No packages match &ldquo;{searchQuery.trim()}&rdquo;</p>
            <p className="oaiy-empty-text">Try a different search term.</p>
          </div>
        ) : (
          <>
            {/* Package list */}
            <div className="space-y-2">
              {paginatedPackages.map((pkg) => (
                <PackageCard
                  key={pkg.id}
                  pkg={pkg}
                  installing={installing[pkg.id] ?? false}
                  onOpen={() => handleOpen(pkg.id)}
                  onUninstall={() => handleUninstall(pkg.id)}
                  disabled={disabled}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={() => setCurrentPage(Math.max(1, safePage - 1))}
                  disabled={safePage === 1}
                  aria-label="Previous page"
                  className="oaiy-icon-btn sm disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <ChevronLeft size={15} aria-hidden="true" />
                </button>
                <span
                  aria-current="page"
                  className="font-mono text-xs text-content-faint"
                >
                  {safePage} / {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setCurrentPage(Math.min(totalPages, safePage + 1))
                  }
                  disabled={safePage === totalPages}
                  aria-label="Next page"
                  className="oaiy-icon-btn sm disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <ChevronRight size={15} aria-hidden="true" />
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Trust Dialog */}
      {pendingInstall && (
        <TrustDialog
          manifest={pendingInstall.manifest}
          onConfirm={handleTrustConfirm}
          onCancel={handleTrustCancel}
        />
      )}
    </div>
  );
}
