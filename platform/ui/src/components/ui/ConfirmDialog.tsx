import { useRef } from 'react';
import { AlertTriangle, Info, Trash2 } from 'lucide-react';
import Dialog from './Dialog';

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

/**
 * A question to answer before going on: the editor's Dialog at its small size,
 * Cancel first and the action last.
 */
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

  // Initial focus: for `danger` (delete / discard) Cancel, so Enter does not
  // confirm the destructive action; for `info` / `warning` the action itself.
  const initialFocusRef = variant === 'danger' ? cancelButtonRef : confirmButtonRef;

  const icon = variant === 'danger' ? <Trash2 size={16} /> : variant === 'warning' ? <AlertTriangle size={16} /> : <Info size={16} />;
  const tone = variant === 'danger' ? 'danger' : variant === 'warning' ? 'warning' : 'accent';
  const confirmClass = variant === 'danger' ? 'btn btn-danger solid' : variant === 'warning' ? 'btn btn-warning' : 'btn btn-primary';

  return (
    <Dialog
      open={isOpen}
      onClose={onCancel}
      title={title}
      icon={icon}
      tone={tone}
      size="sm"
      role="alertdialog"
      initialFocusRef={initialFocusRef}
      testId="confirm-dialog"
      footer={
        <>
          <button ref={cancelButtonRef} type="button" onClick={onCancel} className="btn btn-secondary">
            {cancelLabel}
          </button>
          <button ref={confirmButtonRef} type="button" onClick={onConfirm} className={confirmClass}>
            {confirmLabel}
          </button>
        </>
      }
    >
      <p className="m-0 text-[13px] leading-relaxed text-content-secondary">{message}</p>
    </Dialog>
  );
}
