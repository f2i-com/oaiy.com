import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
import { AlertTriangle, Check, Info, X, XCircle } from 'lucide-react';
import { toastError } from '../lib/readableError';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

/** A button on a toast: what to do about what it says ("Show log"). */
export interface ToastAction {
  label: string;
  onClick: () => void;
}

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
  action?: ToastAction;
  /** One of a kind: a new toast with the same key takes its place (a run's failure, said again, is not a second toast). */
  key?: string;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (message: string, type?: ToastType, duration?: number, action?: ToastAction, key?: string) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

interface ToastProviderProps {
  children: ReactNode;
}

export function ToastProvider({ children }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: ToastType = 'info', duration: number = 3000, action?: ToastAction, key?: string) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    setToasts(prev => [...(key ? prev.filter((t) => t.key !== key) : prev), { id, message, type, duration, action, key }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}

interface ToastContainerProps {
  toasts: Toast[];
  removeToast: (id: string) => void;
}

function ToastContainer({ toasts, removeToast }: ToastContainerProps) {
  return (
    <div className="oaiy-toasts">
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

interface ToastItemProps {
  toast: Toast;
  onRemove: (id: string) => void;
}

function ToastItem({ toast, onRemove }: ToastItemProps) {
  const [isExiting, setIsExiting] = useState(false);
  const [paused, setPaused] = useState(false);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const remainingRef = useRef<number>(toast.duration ?? 0);
  const startedAtRef = useRef<number>(0);

  useEffect(() => {
    remainingRef.current = toast.duration ?? 0;
  }, [toast.id, toast.duration]);

  useEffect(() => {
    if (!toast.duration || toast.duration <= 0 || paused) return;

    const remaining = remainingRef.current;
    startedAtRef.current = Date.now();

    const exitTimer = setTimeout(() => {
      setIsExiting(true);
    }, Math.max(0, remaining - 300)); // Start exit animation 300ms before removal

    const removeTimer = setTimeout(() => {
      onRemove(toast.id);
    }, Math.max(0, remaining));

    return () => {
      clearTimeout(exitTimer);
      clearTimeout(removeTimer);
      const elapsed = Date.now() - startedAtRef.current;
      remainingRef.current = Math.max(400, remaining - elapsed);
    };
  }, [toast.id, toast.duration, paused, onRemove]);

  // Cleanup close timer on unmount
  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const handleClose = useCallback(() => {
    setIsExiting(true);
    closeTimerRef.current = setTimeout(() => onRemove(toast.id), 300);
  }, [toast.id, onRemove]);

  // The dashboard's toast: a card with a rail and an icon in the kind's colour.
  const Icon = toast.type === 'success' ? Check : toast.type === 'error' ? XCircle : toast.type === 'warning' ? AlertTriangle : Info;

  return (
    <div
      className={`oaiy-toast ${toast.type} ${isExiting ? 'animate-toast-exit' : 'animate-toast-enter'}`}
      // role="alert" + aria-live="assertive" interrupts the screen
      // reader — appropriate for errors, but too aggressive for
      // info/success notifications. Use the polite "status" role for
      // non-errors so the reader announces them at the next pause.
      role={toast.type === 'error' ? 'alert' : 'status'}
      aria-live={toast.type === 'error' ? 'assertive' : 'polite'}
      onMouseEnter={() => { setIsExiting(false); setPaused(true); }}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => { setIsExiting(false); setPaused(true); }}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
          setPaused(false);
        }
      }}
    >
      <Icon size={16} aria-hidden="true" />
      {/* (an error in words: a service's JSON answer by its message, without its HTTP status, lib/readableError.ts) */}
      <span>{toast.type === 'error' ? toastError(toast.message) : toast.message}</span>
      {toast.action && (
        <button
          type="button"
          className="oaiy-toast-action"
          onClick={() => {
            toast.action?.onClick();
            handleClose();
          }}
        >
          {toast.action.label}
        </button>
      )}
      <button onClick={handleClose} aria-label="Dismiss" type="button">
        <X size={14} aria-hidden="true" />
      </button>
    </div>
  );
}
