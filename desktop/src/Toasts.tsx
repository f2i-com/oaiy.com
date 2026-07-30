import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from '@tauri-apps/plugin-notification';

/**
 * Tiny in-app toast system — zero dependencies, no portal gymnastics.
 * One context provider at the app root; any descendant calls
 * `useToast().push({ ... })` to surface a transient notification in the
 * bottom-right corner.
 *
 * Native OS notifications: the companion is tray-resident, so most of the
 * time its window is hidden — an in-app toast no one can see is useless
 * for "your 40 GB download finished". So every pushed toast ALSO fires a
 * native OS notification *when the window isn't focused* (focused → just
 * the inline toast, no OS spam). One integration point here covers every
 * transition the panels already surface (downloads, services, python,
 * migration, token).
 */

/** Cached notification-permission state so we only prompt once. */
let notifyPermission: 'unknown' | 'granted' | 'denied' = 'unknown';

/**
 * Fire a native OS notification, but only when the companion window is in
 * the background — a focused user already sees the in-app toast. Lazily
 * requests permission on first use; degrades silently (the toast still
 * showed) if notifications aren't available.
 */
async function maybeNativeNotify(title: string, body?: string): Promise<void> {
  if (typeof document !== 'undefined' && document.hasFocus()) return;
  if (notifyPermission === 'denied') return;
  try {
    if (notifyPermission === 'unknown') {
      let granted = await isPermissionGranted();
      if (!granted) granted = (await requestPermission()) === 'granted';
      notifyPermission = granted ? 'granted' : 'denied';
      if (!granted) return;
    }
  } catch {
    // Not in Tauri, or the plugin is unavailable — the in-app toast
    // already covered it. Don't retry the permission dance every push.
    notifyPermission = 'denied';
    return;
  }
  try {
    await sendNotification(body ? { title, body } : { title });
  } catch (err) {
    // A transient send failure shouldn't disable native notifications for
    // the rest of the session — the in-app toast already covered it, and
    // permission is still good, so leave notifyPermission untouched.
    console.debug('maybeNativeNotify: sendNotification failed', err);
  }
}

export type ToastKind = 'success' | 'error' | 'info';

export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  body?: string;
  /** Auto-dismiss after this many ms. 0 = sticky. Default 5000. */
  timeoutMs?: number;
}

interface ToastContextValue {
  push: (toast: Omit<Toast, 'id'>) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Pending auto-dismiss timers, keyed by toast id, so we can clear a timer
  // when its toast is dismissed early and cancel all of them on unmount
  // (no stray dismiss firing after the provider is gone).
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    const handle = timers.current.get(id);
    if (handle !== undefined) {
      clearTimeout(handle);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback((toast: Omit<Toast, 'id'>) => {
    const t: Toast = { ...toast, id: nextId++ };
    setToasts((prev) => [...prev, t]);
    const timeout = toast.timeoutMs ?? 5000;
    if (timeout > 0) {
      timers.current.set(t.id, setTimeout(() => dismiss(t.id), timeout));
    }
    // Mirror to a native OS notification when the window is backgrounded.
    void maybeNativeNotify(t.title, t.body);
  }, [dismiss]);

  // Cancel any pending auto-dismiss timers when the provider unmounts.
  useEffect(() => {
    const map = timers.current;
    return () => {
      map.forEach((handle) => clearTimeout(handle));
      map.clear();
    };
  }, []);

  // Memoize so consumers (ModelsPanel/PythonPanel) that list `toast` in a deps
  // array don't tear down + recreate their polling intervals on every toast
  // add/auto-dismiss (which re-renders this provider). `push` is already stable.
  const value = useMemo(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-stack" role="region" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            <div className="toast-content">
              <div className="toast-title">{t.title}</div>
              {t.body && <div className="toast-body">{t.body}</div>}
            </div>
            <button
              className="toast-close"
              onClick={() => dismiss(t.id)}
              aria-label="dismiss"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx;
}
