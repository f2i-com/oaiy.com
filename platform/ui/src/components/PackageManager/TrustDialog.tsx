/**
 * TrustDialog - Permission review dialog for installing packages
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { OAIYPackageManifest, PackagePermission } from 'oaiy-core';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface TrustDialogProps {
  manifest: OAIYPackageManifest;
  onConfirm: (grantedPermissions: PackagePermission[]) => void;
  onCancel: () => void;
}

// Permission descriptions for display
const PERMISSION_INFO: Record<
  PackagePermission,
  { label: string; description: string; icon: React.ReactNode }
> = {
  filesystem: {
    label: 'File System Access',
    description: 'Read and write files on your computer',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
        />
      </svg>
    ),
  },
  'filesystem:read': {
    label: 'Read-Only File Access',
    description: 'Read files on your computer (no write access)',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
        />
      </svg>
    ),
  },
  network: {
    label: 'Network Access',
    description: 'Make network requests to external services',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
        />
      </svg>
    ),
  },
  clipboard: {
    label: 'Clipboard Access',
    description: 'Read from and write to your clipboard',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
        />
      </svg>
    ),
  },
  notifications: {
    label: 'Notifications',
    description: 'Show system notifications',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
        />
      </svg>
    ),
  },
  camera: {
    label: 'Camera Access',
    description: 'Access your camera for video capture',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
        />
      </svg>
    ),
  },
  microphone: {
    label: 'Microphone Access',
    description: 'Access your microphone for audio recording',
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
        />
      </svg>
    ),
  },
  // Newer capabilities (May-2026 hardening): network:local, terminal,
  // browser, service:start, code:execute, ui:custom, secrets:read,
  // constants:read, process:execute. Each renders as a generic icon —
  // TrustDialog is the last UI surface that displays the package's
  // request, so the labels here matter more than the artwork.
  'process:execute': {
    label: 'Execute OS Commands',
    description:
      'Run arbitrary shell commands on your machine (cp, rm, custom scripts, etc.). Far more powerful than read/write file access.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  'network:local': {
    label: 'Local Network Access',
    description: 'Talk to localhost / private-network services (e.g. Ollama, ComfyUI)',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
      </svg>
    ),
  },
  terminal: {
    label: 'Terminal Automation',
    description: 'Spawn shells, send keystrokes, screenshot terminal windows',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  browser: {
    label: 'Browser Automation',
    description: 'Drive headless / native browsers, navigate URLs, evaluate scripts',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z M3 12h18M12 3a9 9 0 010 18" />
      </svg>
    ),
  },
  'service:start': {
    label: 'Start Subprocess Services',
    description: 'Launch the package’s bundled services (Python servers, etc.)',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M5 12h14M12 5l7 7-7 7" />
      </svg>
    ),
  },
  'code:execute': {
    label: 'Execute Custom JavaScript',
    description: 'Run logic-block / condition-expression code defined by the package',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
  'ui:custom': {
    label: 'Render Custom UI Components',
    description: 'Show the package’s own React components inside OAIY',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 6h16M4 10h16M4 14h16M4 18h16" />
      </svg>
    ),
  },
  'secrets:read': {
    label: 'Read Secrets',
    description: 'Read user-stored API keys / tokens (OS keychain entries)',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
      </svg>
    ),
  },
  'constants:read': {
    label: 'Read Project Constants',
    description: 'Read non-secret project constants (model names, endpoints, etc.)',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
};

// Capabilities that can run code, touch the filesystem, or read secrets/devices.
// They default to UNCHECKED so the one-click Install path grants only benign
// permissions — the user must explicitly opt into the dangerous ones (least
// privilege). Typed as a string set so unknown/future values are simply ignored.
const HIGH_RISK_PERMISSIONS: ReadonlySet<string> = new Set([
  'filesystem',
  'process:execute',
  'terminal',
  'browser',
  'service:start',
  'code:execute',
  'secrets:read',
  'camera',
  'microphone',
]);

export function TrustDialog({ manifest, onConfirm, onCancel }: TrustDialogProps) {
  const requestedPermissions = manifest.permissions ?? [];

  // Low-risk requested permissions start granted; high-risk ones start UNCHECKED.
  const [grantedPermissions, setGrantedPermissions] = useState<
    Set<PackagePermission>
  >(() => new Set(requestedPermissions.filter((p) => !HIGH_RISK_PERMISSIONS.has(p))));

  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  // Initial focus → Cancel, NOT Install. This dialog hands the
  // package OS-level capabilities (filesystem, process:execute, …);
  // an accidental Enter from the trigger context should not commit
  // them. Matches the danger-variant convention in ConfirmDialog.
  useFocusTrap(dialogRef, true, cancelButtonRef);

  // Escape = Cancel (same reason: never auto-commit a permission grant).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  const togglePermission = useCallback((permission: PackagePermission) => {
    setGrantedPermissions((prev) => {
      const next = new Set(prev);
      if (next.has(permission)) {
        next.delete(permission);
      } else {
        next.add(permission);
      }
      return next;
    });
  }, []);

  const submittingRef = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  const handleConfirm = useCallback(() => {
    // Synchronous ref guard so a double-click / Enter+click can't fire the install
    // twice — the dialog stays mounted (clickable) for the whole async parent
    // handler (peekPackageSize + installPackage), and a second install_package
    // against the filesystem would race the first.
    if (submittingRef.current) return;
    submittingRef.current = true;
    setSubmitting(true);
    onConfirm(Array.from(grantedPermissions));
  }, [grantedPermissions, onConfirm]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-2xl max-w-md w-full mx-4 overflow-hidden"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="trust-dialog-title"
        aria-describedby="trust-dialog-desc"
      >
        {/* Header */}
        <div className="px-6 pt-6 pb-4">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
              <svg
                className="w-7 h-7 text-white"
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
            <div className="flex-1 min-w-0">
              <h2 id="trust-dialog-title" className="text-xl font-semibold text-slate-800 dark:text-slate-100">
                Install Package
              </h2>
              <p id="trust-dialog-desc" className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">Review details before installing</p>
            </div>
          </div>
        </div>

        {/* Package Info Card */}
        <div className="mx-6 mb-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-200 dark:border-slate-600/50">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3 className="font-semibold text-slate-800 dark:text-slate-100 truncate">{manifest.name}</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                v{manifest.version}
                {manifest.author && <span className="text-slate-400 dark:text-slate-500"> · </span>}
                {manifest.author && <span>{manifest.author}</span>}
              </p>
            </div>
            <span className="flex-shrink-0 inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-300 rounded-full">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Unverified
            </span>
          </div>
          {manifest.description && (
            <p className="mt-3 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{manifest.description}</p>
          )}
        </div>

        {/* Permissions */}
        <div className="px-6 pb-4">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-3">
            Requested Permissions
          </h4>

          {requestedPermissions.length === 0 ? (
            <div className="flex items-center gap-3 p-3 rounded-xl bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/20">
              <div className="w-8 h-8 rounded-lg bg-green-100 dark:bg-green-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <p className="text-sm text-green-700 dark:text-green-300">
                No special permissions required
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {requestedPermissions.map((permission) => {
                const info = PERMISSION_INFO[permission];
                const granted = grantedPermissions.has(permission);

                return (
                  <button
                    key={permission}
                    type="button"
                    role="checkbox"
                    aria-checked={granted}
                    aria-describedby={`perm-desc-${permission}`}
                    onClick={() => togglePermission(permission)}
                    className={`w-full flex items-center gap-3 p-3 rounded-xl border transition-all ${
                      granted
                        ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-200 dark:border-purple-500/30 hover:border-purple-300 dark:hover:border-purple-500/50'
                        : 'bg-slate-50 dark:bg-slate-700/30 border-slate-200 dark:border-slate-600/50 hover:border-slate-300 dark:hover:border-slate-500/50'
                    }`}
                  >
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                        granted
                          ? 'bg-purple-100 dark:bg-purple-500/20 text-purple-600 dark:text-purple-400'
                          : 'bg-slate-200 dark:bg-slate-600 text-slate-500 dark:text-slate-400'
                      }`}
                    >
                      {info?.icon ?? (
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 text-left">
                      <p
                        className={`text-sm font-medium ${
                          granted ? 'text-slate-800 dark:text-slate-100' : 'text-slate-600 dark:text-slate-300'
                        }`}
                      >
                        {info?.label ?? permission}
                      </p>
                      <p id={`perm-desc-${permission}`} className="text-xs text-slate-500 dark:text-slate-400">
                        {info?.description ?? 'Unknown permission'}
                      </p>
                    </div>
                    <div
                      className={`w-5 h-5 rounded-md border-2 flex items-center justify-center transition-colors ${
                        granted
                          ? 'bg-purple-500 border-purple-500'
                          : 'border-slate-300 dark:border-slate-500'
                      }`}
                    >
                      {granted && (
                        <svg
                          className="w-3 h-3 text-white"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={3}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Warning Footer */}
        <div className="px-6 py-3 bg-slate-100 dark:bg-slate-700/50 border-t border-slate-200 dark:border-slate-600/50">
          <div className="flex gap-2.5 items-start">
            <svg
              className="w-4 h-4 flex-shrink-0 mt-0.5 text-slate-400 dark:text-slate-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Only install packages from sources you trust. Packages can execute code and access resources on your computer.
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-4 flex justify-end gap-3 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
          <button
            ref={cancelButtonRef}
            onClick={onCancel}
            className="px-4 py-2.5 text-sm font-medium text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={submitting}
            className="px-5 py-2.5 text-sm font-medium bg-gradient-to-r from-purple-600 to-purple-500 hover:from-purple-500 hover:to-purple-400 text-white rounded-lg transition-all shadow-md shadow-purple-500/20 hover:shadow-lg hover:shadow-purple-500/30 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {submitting ? 'Installing…' : 'Install Package'}
          </button>
        </div>
      </div>
    </div>
  );
}
