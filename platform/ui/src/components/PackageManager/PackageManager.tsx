/**
 * PackageManager - UI for managing .oaiy packages
 *
 * Lists installed packages, allows opening, installing, and uninstalling.
 */

import { useState, useMemo, useCallback } from 'react';
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
      className="flex items-center justify-between w-full px-3 py-2 text-left hover:bg-surface-tertiary rounded-lg transition-colors"
      onClick={() => setExpanded(!expanded)}
      disabled={disabled}
      aria-expanded={expanded}
    >
      <div className="flex items-center gap-2">
        <svg
          className="w-4 h-4 text-accent"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
          />
        </svg>
        <span className="text-sm font-medium text-content-primary">Packages</span>
        {installedCount > 0 && (
          <span className="px-1.5 py-0.5 text-xs font-medium bg-accent/15 text-accent rounded">
            {installedCount}
          </span>
        )}
      </div>
      <svg
        className={`w-4 h-4 text-content-secondary transition-transform ${
          expanded ? 'rotate-180' : ''
        }`}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </button>
  );

  if (!expanded) {
    return (
      <div className="bg-surface-secondary rounded-xl border border-edge-primary">
        {header}
      </div>
    );
  }

  return (
    <div className="bg-surface-secondary rounded-xl border border-edge-primary">
      {header}

      <div className="px-3 pb-3 space-y-3">
        {/* Search and Add */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <svg
              className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-content-secondary"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search packages"
              aria-label="Search packages by name, ID, or description"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full pl-8 pr-3 py-1.5 text-sm bg-surface-tertiary border border-edge-primary rounded-lg text-content-primary placeholder:text-content-faint focus:outline-none focus:ring-2 focus:ring-accent/50"
            />
          </div>
          <button
            type="button"
            onClick={handleBrowsePackage}
            disabled={disabled}
            aria-label="Install package"
            className="px-3 py-1.5 text-sm font-medium bg-accent hover:bg-accent/90 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            Install
          </button>
        </div>

        {/* Error display */}
        {error && (
          <div className="oaiy-banner err">
            <span className="flex-1">{error}</span>
          </div>
        )}

        {/* Loading state */}
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <svg
              className="w-6 h-6 text-accent animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
        ) : packages.length === 0 ? (
          <div className="text-center py-8 text-content-secondary text-sm">
            <svg
              className="w-12 h-12 mx-auto mb-3 text-content-faint"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
              />
            </svg>
            <p>No packages installed</p>
            <p className="text-xs text-content-secondary mt-1">
              Click "Install" to add a .oaiy package
            </p>
          </div>
        ) : filteredPackages.length === 0 ? (
          <div className="text-center py-8 text-content-secondary text-sm">
            <p>No packages match "{searchQuery.trim()}"</p>
            <p className="text-xs text-content-secondary mt-1">
              Try a different search term
            </p>
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
                  className="p-1 text-content-secondary hover:text-content-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 19l-7-7 7-7"
                    />
                  </svg>
                </button>
                <span
                  aria-current="page"
                  className="text-xs text-content-secondary"
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
                  className="p-1 text-content-secondary hover:text-content-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
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
