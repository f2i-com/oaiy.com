/**
 * PackageCard - Individual package display card
 */

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
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      case 'updating':
        return 'bg-yellow-500';
      default:
        return 'bg-slate-500';
    }
  };

  // Determine trust badge
  const getTrustBadge = () => {
    switch (pkg.trustLevel) {
      case 'trusted':
        return (
          <span className="px-1.5 py-0.5 text-xs bg-green-100 dark:bg-green-500/20 text-green-600 dark:text-green-400 rounded">
            Trusted
          </span>
        );
      case 'verified':
        return (
          <span className="px-1.5 py-0.5 text-xs bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded">
            Verified
          </span>
        );
      case 'blocked':
        return (
          <span className="px-1.5 py-0.5 text-xs bg-red-100 dark:bg-red-500/20 text-red-600 dark:text-red-400 rounded">
            Blocked
          </span>
        );
      default:
        return (
          <span className="px-1.5 py-0.5 text-xs bg-yellow-100 dark:bg-yellow-500/20 text-yellow-600 dark:text-yellow-400 rounded">
            Untrusted
          </span>
        );
    }
  };

  return (
    <div className="p-3 bg-slate-100/50 dark:bg-slate-700/50 rounded-lg border border-slate-300/50 dark:border-slate-600/50 hover:border-slate-400/50 dark:hover:border-slate-500/50 transition-colors">
      <div className="flex items-start justify-between gap-3">
        {/* Package icon and info */}
        <div className="flex items-start gap-3 flex-1 min-w-0">
          {/* Icon */}
          <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
            <svg
              className="w-5 h-5 text-purple-400"
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
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate">
                {pkg.name}
              </h3>
              <span className="text-xs text-slate-500">v{pkg.version}</span>
              {getTrustBadge()}
              {/* Status indicator */}
              <div
                className={`w-2 h-2 rounded-full ${getStatusColor()}`}
                title={pkg.status}
              />
            </div>
            {pkg.description && (
              <p className="text-xs text-slate-400 truncate mt-0.5">
                {pkg.description}
              </p>
            )}
            {pkg.author && (
              <p className="text-xs text-slate-500 truncate mt-0.5">
                by {pkg.author}
              </p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1.5">
          {installing ? (
            <div className="px-3 py-1.5">
              <svg
                className="w-4 h-4 text-purple-400 animate-spin"
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
          ) : (
            <>
              {/* Open button */}
              <button
                onClick={onOpen}
                disabled={disabled || pkg.trustLevel === 'blocked'}
                className="px-2.5 py-1.5 text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Open package"
              >
                Open
              </button>

              {/* Uninstall button */}
              <button
                onClick={onUninstall}
                disabled={disabled || pkg.status === 'running'}
                className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Uninstall package"
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
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
