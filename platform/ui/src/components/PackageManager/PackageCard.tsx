/**
 * PackageCard - Individual package display card
 */

import { Loader2, Package, Trash2 } from 'lucide-react';
import type { PackageInfo } from '../../hooks/usePackages';

interface PackageCardProps {
  pkg: PackageInfo;
  installing: boolean;
  onOpen: () => void;
  onUninstall: () => void;
  disabled?: boolean;
}

export function PackageCard({
  pkg,
  installing,
  onOpen,
  onUninstall,
  disabled,
}: PackageCardProps) {
  // Determine status color
  const getStatusColor = () => {
    switch (pkg.status) {
      case 'running':
        return 'bg-signal-green';
      case 'error':
        return 'bg-signal-danger';
      case 'updating':
        return 'bg-signal-amber';
      default:
        return 'bg-content-faint';
    }
  };

  // Determine trust badge
  const getTrustBadge = () => {
    switch (pkg.trustLevel) {
      case 'trusted':
        return <span className="oaiy-pill ok">Trusted</span>;
      case 'verified':
        return <span className="oaiy-pill info">Verified</span>;
      case 'blocked':
        return <span className="oaiy-pill err">Blocked</span>;
      default:
        return <span className="oaiy-pill warn">Untrusted</span>;
    }
  };

  return (
    <div className="p-3 bg-surface-tertiary/60 rounded-[var(--r-ctl)] border border-edge-primary hover:border-edge-strong transition-colors">
      <div className="flex items-start justify-between gap-3">
        {/* Package icon and info */}
        <div className="flex items-start gap-3 flex-1 min-w-0">
          {/* Icon */}
          <div className="w-10 h-10 rounded-[var(--r-ctl)] bg-accent/15 flex items-center justify-center shrink-0 text-accent">
            <Package size={18} aria-hidden="true" />
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="m-0 text-sm font-semibold text-content-primary truncate">
                {pkg.name}
              </h3>
              <span className="font-mono text-xs text-content-faint">v{pkg.version}</span>
              {getTrustBadge()}
              {/* Status indicator */}
              <span
                className={`inline-block w-2 h-2 rounded-full ${getStatusColor()}`}
                title={pkg.status}
                aria-label={`Status: ${pkg.status}`}
              />
            </div>
            {pkg.description && (
              <p className="m-0 text-xs text-content-secondary truncate mt-0.5">
                {pkg.description}
              </p>
            )}
            {pkg.author && (
              <p className="m-0 text-xs text-content-faint truncate mt-0.5">
                by {pkg.author}
              </p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1.5 shrink-0">
          {installing ? (
            <div className="px-2 py-1" role="status" aria-label="Installing package">
              <Loader2 size={16} className="text-accent animate-spin" aria-hidden="true" />
            </div>
          ) : (
            <>
              {/* Open button */}
              <button
                type="button"
                onClick={onOpen}
                disabled={disabled || pkg.trustLevel === 'blocked'}
                className="btn btn-primary btn-sm"
                title="Open package"
              >
                Open
              </button>

              {/* Uninstall button */}
              <button
                type="button"
                onClick={onUninstall}
                disabled={disabled || pkg.status === 'running'}
                aria-label="Uninstall package"
                className="oaiy-icon-btn sm hover:text-signal-danger-ink hover:bg-signal-danger/10 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Uninstall package"
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
