/**
 * TrustDialog - Permission review dialog for installing packages
 */

import { useState, useCallback, useRef } from 'react';
import { Check, Info, Package, ShieldCheck } from 'lucide-react';
import type { OAIYPackageManifest, PackagePermission } from 'oaiy-core';
import Dialog from '../ui/Dialog';

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

  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  // Initial focus → Cancel, NOT Install. This dialog hands the
  // package OS-level capabilities (filesystem, process:execute, …);
  // an accidental Enter from the trigger context should not commit
  // them. Matches the danger-variant convention in ConfirmDialog.
  // Escape and the overlay = Cancel (same reason: never auto-commit a grant).

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
    <Dialog
      open
      onClose={onCancel}
      title="Install this package?"
      description="Review what it may do before it is installed."
      icon={<Package size={16} />}
      tone="accent"
      size="md"
      role="alertdialog"
      initialFocusRef={cancelButtonRef}
      // Once Install is pressed the install runs to its end.
      dismissible={!submitting}
      footer={
        <>
          <button ref={cancelButtonRef} type="button" onClick={onCancel} disabled={submitting} className="btn btn-secondary">
            Cancel
          </button>
          <button type="button" onClick={handleConfirm} disabled={submitting} className="btn btn-primary">
            {submitting ? 'Installing…' : 'Install package'}
          </button>
        </>
      }
    >
      {/* The package */}
      <div className="rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/50 p-3.5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <h3 className="m-0 truncate text-[14px] font-semibold text-content-primary">{manifest.name}</h3>
            <p className="m-0 mt-0.5 font-mono text-[11.5px] text-content-faint">
              v{manifest.version}
              {manifest.author && <> · {manifest.author}</>}
            </p>
          </div>
          <span className="oaiy-pill warn dot">unverified</span>
        </div>
        {manifest.description && (
          <p className="m-0 mt-2.5 text-[12.5px] leading-relaxed text-content-secondary">{manifest.description}</p>
        )}
      </div>

      {/* Permissions */}
      <div className="flex flex-col gap-2">
        <span className="oaiy-label">It asks to</span>

        {requestedPermissions.length === 0 ? (
          <div className="oaiy-banner ok">
            <ShieldCheck size={15} className="mt-0.5 shrink-0" />
            <span className="flex-1">Nothing special: it needs no permissions.</span>
          </div>
        ) : (
          requestedPermissions.map((permission) => {
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
                className={`flex w-full items-center gap-3 rounded-[var(--r-ctl)] border p-2.5 text-left transition-colors ${
                  granted
                    ? 'border-accent/40 bg-accent/10 hover:border-accent/60'
                    : 'border-edge-primary bg-surface-tertiary/40 hover:border-edge-strong'
                }`}
              >
                <span
                  className={`grid h-8 w-8 shrink-0 place-items-center rounded-[var(--r-sm)] ${
                    granted ? 'bg-accent/15 text-accent' : 'bg-surface-tertiary text-content-faint'
                  }`}
                >
                  {info?.icon ?? <Check size={18} />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className={`block text-[13px] font-semibold ${granted ? 'text-content-primary' : 'text-content-secondary'}`}>
                    {info?.label ?? permission}
                  </span>
                  <span id={`perm-desc-${permission}`} className="block text-[12px] leading-snug text-content-faint">
                    {info?.description ?? 'Unknown permission'}
                  </span>
                </span>
                <span
                  className={`grid h-5 w-5 shrink-0 place-items-center rounded-[5px] border-2 transition-colors ${
                    granted ? 'border-accent bg-accent text-white' : 'border-edge-strong'
                  }`}
                  aria-hidden="true"
                >
                  {granted && <Check size={12} strokeWidth={3} />}
                </span>
              </button>
            );
          })
        )}
      </div>

      <p className="oaiy-help faint flex items-start gap-2">
        <Info size={14} className="mt-0.5 shrink-0" />
        Install packages only from sources you trust: a package can run code and reach things on your computer.
      </p>
    </Dialog>
  );
}
