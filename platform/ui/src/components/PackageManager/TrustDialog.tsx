/**
 * TrustDialog - Permission review dialog for installing packages
 */

import { useState, useCallback, useRef } from 'react';
import {
  Bell,
  Check,
  Clipboard,
  Code,
  Compass,
  Eye,
  FileText,
  Folder,
  Globe,
  Image,
  Info,
  KeyRound,
  LayoutGrid,
  Mic,
  Package,
  Server,
  ShieldCheck,
  Terminal,
  Video,
  Wifi,
} from 'lucide-react';
import type { OAIYPackageManifest, PackagePermission } from 'oaiy-core';
import Dialog from '../ui/Dialog';

export type PackageVerificationLevel = 'unverified' | 'hash_matched' | 'signature_verified';

interface TrustDialogProps {
  manifest: OAIYPackageManifest;
  verification?: PackageVerificationLevel;
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
    icon: <Folder size={18} aria-hidden="true" />,
  },
  'filesystem:read': {
    label: 'Read-Only File Access',
    description: 'Read files on your computer (no write access)',
    icon: <Eye size={18} aria-hidden="true" />,
  },
  network: {
    label: 'Network Access',
    description: 'Make network requests to external services',
    icon: <Globe size={18} aria-hidden="true" />,
  },
  clipboard: {
    label: 'Clipboard Access',
    description: 'Read from and write to your clipboard',
    icon: <Clipboard size={18} aria-hidden="true" />,
  },
  notifications: {
    label: 'Notifications',
    description: 'Show system notifications',
    icon: <Bell size={18} aria-hidden="true" />,
  },
  camera: {
    label: 'Camera Access',
    description: 'Access your camera for video capture',
    icon: <Video size={18} aria-hidden="true" />,
  },
  microphone: {
    label: 'Microphone Access',
    description: 'Access your microphone for audio recording',
    icon: <Mic size={18} aria-hidden="true" />,
  },
  'process:execute': {
    label: 'Execute OS Commands',
    description:
      'Run arbitrary shell commands on your machine (cp, rm, custom scripts, etc.). Far more powerful than read/write file access.',
    icon: <Terminal size={18} aria-hidden="true" />,
  },
  'network:local': {
    label: 'Local Network Access',
    description: 'Talk to localhost / private-network services (e.g. Ollama, ComfyUI)',
    icon: <Wifi size={18} aria-hidden="true" />,
  },
  terminal: {
    label: 'Terminal Automation',
    description: 'Spawn shells, send keystrokes, screenshot terminal windows',
    icon: <Terminal size={18} aria-hidden="true" />,
  },
  browser: {
    label: 'Browser Automation',
    description: 'Drive headless / native browsers, navigate URLs, evaluate scripts',
    icon: <Compass size={18} aria-hidden="true" />,
  },
  'service:start': {
    label: 'Start Subprocess Services',
    description: 'Launch the package’s bundled services (Python servers, etc.)',
    icon: <Server size={18} aria-hidden="true" />,
  },
  'code:execute': {
    label: 'Execute Custom JavaScript',
    description: 'Run logic-block / condition-expression code defined by the package',
    icon: <Code size={18} aria-hidden="true" />,
  },
  'ui:custom': {
    label: 'Render Custom UI Components',
    description: 'Show the package’s own React components inside OAIY',
    icon: <LayoutGrid size={18} aria-hidden="true" />,
  },
  'secrets:read': {
    label: 'Read Secrets',
    description: 'Read user-stored API keys / tokens (OS keychain entries)',
    icon: <KeyRound size={18} aria-hidden="true" />,
  },
  'constants:read': {
    label: 'Read Project Constants',
    description: 'Read non-secret project constants (model names, endpoints, etc.)',
    icon: <FileText size={18} aria-hidden="true" />,
  },
  diffusion: {
    label: 'Local Diffusion Models',
    description: 'Load and run local image/video diffusion models on your GPU',
    icon: <Image size={18} aria-hidden="true" />,
  },
};

// Capabilities that can run code, touch the filesystem, or read secrets/devices.
// They default to UNCHECKED so the one-click Install path grants only benign
// permissions — the user must explicitly opt into the dangerous ones (least
// privilege). Typed as a string set so unknown/future values are simply ignored.
const HIGH_RISK_PERMISSIONS: ReadonlySet<string> = new Set([
  'filesystem',
  'filesystem:read',
  'network:local',
  'process:execute',
  'terminal',
  'browser',
  'service:start',
  'code:execute',
  'secrets:read',
  'camera',
  'microphone',
]);

export function TrustDialog({ manifest, verification, onConfirm, onCancel }: TrustDialogProps) {
  const requestedPermissions = manifest.permissions ?? [];

  const verificationBadge =
    verification === 'signature_verified'
      ? { className: 'oaiy-pill ok dot', label: 'Verified' }
      : verification === 'hash_matched'
        ? { className: 'oaiy-pill info dot', label: 'Hash verified' }
        : { className: 'oaiy-pill warn dot', label: 'Unverified' };

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
          <span className={verificationBadge.className}>{verificationBadge.label}</span>
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
            const isHighRisk = HIGH_RISK_PERMISSIONS.has(permission);

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
                  {info?.icon ?? <Check size={18} aria-hidden="true" />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-1.5">
                    <span className={`text-[13px] font-semibold ${granted ? 'text-content-primary' : 'text-content-secondary'}`}>
                      {info?.label ?? permission}
                    </span>
                    {isHighRisk && <span className="oaiy-pill warn">High risk</span>}
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
