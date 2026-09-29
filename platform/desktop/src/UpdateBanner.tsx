import { useState } from 'react';
import { ArrowUpCircle } from 'lucide-react';
import { useUpdateStatus } from './useUpdateStatus';

/**
 * The Overview's word that a newer OAIY exists: when one is ready to install, or available to
 * download. It can be dismissed, and stays dismissed until there is a NEWER version than the one
 * it was dismissed for (the version is remembered in this window's storage).
 */

export const DISMISSED_KEY = 'oaiy-update-dismissed-version';

/** The version whose banner was dismissed; null when none, or when storage is not there. */
export function dismissedVersion(): string | null {
  try {
    return localStorage.getItem(DISMISSED_KEY);
  } catch {
    return null;
  }
}

function dismissVersion(version: string): void {
  try {
    localStorage.setItem(DISMISSED_KEY, version);
  } catch {
    /* no storage: it stays dismissed until the page is left */
  }
}

/** Whether the banner is shown for a status and a dismissed version. Only a ready or available update shows one. */
export function bannerVersion(status: { state: string; latestVersion: string | null } | null, dismissed: string | null): string | null {
  if (!status || !status.latestVersion) return null;
  if (status.state !== 'available' && status.state !== 'ready') return null;
  return status.latestVersion === dismissed ? null : status.latestVersion;
}

export default function UpdateBanner({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { status } = useUpdateStatus();
  const [dismissed, setDismissed] = useState<string | null>(() => dismissedVersion());
  const version = bannerVersion(status, dismissed);
  if (!status || !version) return null;
  const ready = status.state === 'ready';
  return (
    <div className="banner banner-pending banner-dismissable update-banner" role="status">
      <span>
        <ArrowUpCircle size={13} />{' '}
        {ready ? `OAIY ${version} is downloaded and ready to install.` : `OAIY ${version} is available.`}{' '}
        <button className="btn-tiny" onClick={onOpenSettings}>
          {ready ? 'Open Settings to restart' : 'See what is new'}
        </button>
      </span>
      <button
        className="banner-dismiss"
        aria-label="Dismiss until the next version"
        title="Dismiss until the next version"
        onClick={() => {
          dismissVersion(version);
          setDismissed(version);
        }}
      >
        ×
      </button>
    </div>
  );
}
