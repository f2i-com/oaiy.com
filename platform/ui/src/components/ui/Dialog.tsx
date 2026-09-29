/**
 * Dialog — the editor's one modal, for short blocking tasks only.
 *
 * Confirm a delete, name something, import a file, add a service: a task that
 * has to be finished or dismissed before going on. Anything you browse, edit at
 * length or come back to (Data, the queue, Settings, packages) is a page in
 * the editor's main area instead, not a dialog.
 *
 * Every dialog is drawn the same way: a title (with an optional icon and one
 * line under it), a body that scrolls, and a footer whose buttons run
 * secondary first, primary last. Three sizes: sm (a question), md (a short
 * form), lg (a form with a list or a preview).
 *
 * Behaviour, the same for all of them:
 *   - Escape and a click on the overlay close it (unless `dismissible` is
 *     false while something is in flight). Only the top-most dialog answers,
 *     so a confirm over another dialog closes by itself.
 *   - Focus moves in, Tab stays in, and focus goes back to where it was when
 *     the dialog closes (useFocusTrap).
 *
 * It needs no React context: it is portaled to <body> and styled with the
 * theme tokens alone, so it also works where it is opened imperatively (the
 * "Add a service…" dialog) or before the app's providers exist (the shared
 * flow's password prompt at boot).
 */
import { useEffect, useId, useRef, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { isTopTrap, useFocusTrap } from '../../hooks/useFocusTrap';

export type DialogSize = 'sm' | 'md' | 'lg';
export type DialogTone = 'default' | 'danger' | 'warning' | 'accent';

export interface DialogProps {
  open: boolean;
  /** Escape, the overlay and the close button all call this. */
  onClose: () => void;
  title: ReactNode;
  /** One line under the title. */
  description?: ReactNode;
  /** A small icon beside the title, in the tone's colour. */
  icon?: ReactNode;
  tone?: DialogTone;
  size?: DialogSize;
  children?: ReactNode;
  /** The buttons: secondary first, primary last. */
  footer?: ReactNode;
  /** Where focus goes first (else the first control in the body). */
  initialFocusRef?: RefObject<HTMLElement | null>;
  /** `alertdialog` for a question that interrupts (a confirm). */
  role?: 'dialog' | 'alertdialog';
  /** False while something is in flight: Escape and the overlay do nothing. */
  dismissible?: boolean;
  /** Hide the close button (the footer has the only way out). */
  hideClose?: boolean;
  /** Extra classes on the body, e.g. to drop its padding for a full-bleed list. */
  bodyClassName?: string;
  /** For tests and screenshots. */
  testId?: string;
}

export default function Dialog({
  open,
  onClose,
  title,
  description,
  icon,
  tone = 'default',
  size = 'md',
  children,
  footer,
  initialFocusRef,
  role = 'dialog',
  dismissible = true,
  hideClose = false,
  bodyClassName = '',
  testId,
}: DialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const pressedOnOverlay = useRef(false);
  const titleId = useId();
  const descId = useId();
  useFocusTrap(panelRef, open, initialFocusRef);

  // Escape closes the top-most dialog only.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      const panel = panelRef.current;
      if (!panel || !isTopTrap(panel)) return;
      event.preventDefault();
      event.stopPropagation();
      if (dismissible) onCloseRef.current();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, dismissible]);

  if (!open) return null;

  return createPortal(
    <div
      className="oaiy-dialog-back"
      // Close only on a click that started AND ended on the overlay, so a text
      // selection dragged out of a field does not dismiss the form.
      onMouseDown={(e) => { pressedOnOverlay.current = e.target === e.currentTarget; }}
      onClick={(e) => {
        if (dismissible && pressedOnOverlay.current && e.target === e.currentTarget) onClose();
        pressedOnOverlay.current = false;
      }}
    >
      <div
        ref={panelRef}
        className={`oaiy-dialog oaiy-dialog-${size} tone-${tone}`}
        role={role}
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descId : undefined}
        data-testid={testId}
      >
        <header className="oaiy-dialog-head">
          {icon && <span className="oaiy-dialog-icon" aria-hidden="true">{icon}</span>}
          <div className="oaiy-dialog-titles">
            <h2 id={titleId}>{title}</h2>
            {description && <p id={descId}>{description}</p>}
          </div>
          {!hideClose && (
            <button type="button" className="oaiy-icon-btn" onClick={onClose} aria-label="Close" title="Close" disabled={!dismissible}>
              <X size={16} />
            </button>
          )}
        </header>
        {children !== undefined && children !== null && (
          <div className={`oaiy-dialog-body ${bodyClassName}`}>{children}</div>
        )}
        {footer && <footer className="oaiy-dialog-foot">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}
