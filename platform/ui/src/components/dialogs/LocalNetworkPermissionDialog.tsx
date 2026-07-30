import { useEffect, useRef, useState } from 'react';
import type { LocalNetworkPermissionRequest } from 'oaiy-core';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface LocalNetworkPermissionDialogProps {
  request: LocalNetworkPermissionRequest;
  onResponse: (allowed: boolean, remember: boolean) => void;
}

export default function LocalNetworkPermissionDialog({
  request,
  onResponse,
}: LocalNetworkPermissionDialogProps) {
  const [remember, setRemember] = useState(true);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Initial focus goes to Deny — this is a permission prompt, the
  // safe default is "no" even though "yes" is the more visually
  // prominent button. Matches the ConfirmDialog danger convention.
  const denyButtonRef = useRef<HTMLButtonElement>(null);
  useFocusTrap(dialogRef, true, denyButtonRef);

  // Escape = Deny (without remembering). Matches the rest of the app
  // where Escape never commits any change; the user can re-trigger
  // the workflow if they meant Allow.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onResponse(false, false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onResponse]);

  return (
    <div
      className="fixed inset-0 bg-black/60 z-[100] flex items-center justify-center"
      onClick={() => onResponse(false, false)}
    >
      <div
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
        className="bg-white dark:bg-slate-800 rounded-lg shadow-2xl w-full max-w-md overflow-hidden border border-slate-200 dark:border-slate-700"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="local-net-title"
        aria-describedby="local-net-desc"
      >
        {/* Header */}
        <div className="px-6 py-4 short:px-4 short:py-2.5 bg-amber-100 dark:bg-amber-900/30 border-b border-amber-400 dark:border-amber-600/30">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-amber-200 dark:bg-amber-600/30 flex items-center justify-center">
              <svg className="w-6 h-6 text-amber-700 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h2 id="local-net-title" className="text-lg font-semibold text-amber-700 dark:text-amber-400">Local Network Access Request</h2>
              <p id="local-net-desc" className="text-slate-600 dark:text-slate-400 text-sm">A workflow wants to connect to a local service</p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-5 short:px-4 short:py-3 space-y-4 short:space-y-2.5 max-h-[80vh] overflow-y-auto">
          {/* Address being accessed */}
          <div className="bg-slate-100 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 rounded-lg p-4">
            <label className="text-slate-500 text-xs uppercase tracking-wider">Address</label>
            <div className="mt-1 font-mono text-lg text-slate-800 dark:text-slate-200">{request.hostPort}</div>
            <div className="mt-1 text-xs text-slate-500 truncate">{request.url}</div>
          </div>

          {/* Purpose if provided */}
          {request.purpose && (
            <div className="bg-slate-100 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 rounded-lg p-4">
              <label className="text-slate-500 text-xs uppercase tracking-wider">Purpose</label>
              <div className="mt-1 text-slate-700 dark:text-slate-300">{request.purpose}</div>
            </div>
          )}

          {/* Warning */}
          <div className="flex items-start gap-3 text-sm">
            <svg className="w-5 h-5 text-slate-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-slate-600 dark:text-slate-400">
              Only allow access if you trust this workflow and the service running at this address.
            </p>
          </div>

          {/* Remember checkbox */}
          <label className="flex items-center gap-3 cursor-pointer p-3 bg-slate-100 dark:bg-slate-700/30 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700/50 transition-colors">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
              className="w-5 h-5 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-green-500 focus:ring-green-500 focus:ring-offset-white dark:focus:ring-offset-slate-800"
            />
            <div>
              <span className="text-slate-800 dark:text-slate-200">Remember this address</span>
              <p className="text-slate-500 text-xs">Add to whitelist for future workflows</p>
            </div>
          </label>
        </div>

        {/* Actions */}
        <div className="px-6 py-4 short:px-4 short:py-2.5 bg-slate-50 dark:bg-slate-900/30 border-t border-slate-200 dark:border-slate-700 flex gap-3">
          <button
            ref={denyButtonRef}
            onClick={() => onResponse(false, false)}
            className="flex-1 px-4 py-2.5 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded-lg transition-colors font-medium"
          >
            Deny
          </button>
          <button
            onClick={() => onResponse(true, remember)}
            className="flex-1 px-4 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors font-medium"
          >
            Allow {remember && '& Remember'}
          </button>
        </div>
      </div>
    </div>
  );
}
