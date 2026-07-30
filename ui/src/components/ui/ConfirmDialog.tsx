import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'danger',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Initial-focus target: for `danger` (delete / discard / destructive)
  // we focus Cancel so Enter doesn't auto-confirm the dangerous action.
  // For `info` / `warning` Confirm is the expected default, so we focus
  // it as before — Enter then commits the message the user just read.
  const initialFocusRef = variant === 'danger' ? cancelButtonRef : confirmButtonRef;
  useFocusTrap(dialogRef, isOpen, initialFocusRef);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  // Get portal container - use document.body to escape any parent positioning context
  const portalContainer = document.body;

  const variantStyles = {
    danger: {
      icon: (
        <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      ),
      iconBg: 'bg-red-500/20',
      confirmButton: 'bg-red-600 hover:bg-red-500 focus:ring-red-500',
    },
    warning: {
      icon: (
        <svg className="w-6 h-6 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
      iconBg: 'bg-amber-500/20',
      confirmButton: 'bg-amber-600 hover:bg-amber-500 focus:ring-amber-500',
    },
    info: {
      icon: (
        <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      iconBg: 'bg-blue-500/20',
      confirmButton: 'bg-blue-600 hover:bg-blue-500 focus:ring-blue-500',
    },
  };

  const styles = variantStyles[variant];

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onCancel}
      />

      {/* Dialog */}
      <div
        ref={dialogRef}
        className="relative bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-scaleIn"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
      >
        {/* Header */}
        <div className="flex items-start gap-4 p-5 short:p-4">
          <div className={`flex-shrink-0 p-2 rounded-full ${styles.iconBg}`}>
            {styles.icon}
          </div>
          <div className="flex-1 pt-1">
            <h3
              id="confirm-dialog-title"
              className="text-lg font-semibold text-slate-800 dark:text-slate-100"
            >
              {title}
            </h3>
            <p
              id="confirm-dialog-message"
              className="mt-2 text-sm text-slate-500 dark:text-slate-400 leading-relaxed"
            >
              {message}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 px-5 py-4 short:px-4 short:py-2.5 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-700">
          <button
            ref={cancelButtonRef}
            onClick={onCancel}
            className="btn btn-secondary btn-md"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmButtonRef}
            onClick={onConfirm}
            className={`btn btn-md ${variant === 'danger' ? 'btn-danger' : variant === 'warning' ? 'btn-warning' : 'btn-primary'}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    portalContainer
  );
}
