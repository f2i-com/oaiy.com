import { useEffect, useRef, useState } from 'react';
import { Plus } from 'lucide-react';
import Dialog from '../ui/Dialog';

/**
 * New flow: its name first, then the empty canvas. The one way to make a flow
 * (the header's New flow, and the empty canvas's button, both open it).
 */
export default function NewFlowDialog({
  open,
  onClose,
  onCreate,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (name: string) => void;
}) {
  const [name, setName] = useState('Untitled flow');
  const inputRef = useRef<HTMLInputElement>(null);

  // A fresh name each time it opens, selected so typing replaces it.
  useEffect(() => {
    if (!open) return;
    setName('Untitled flow');
    const raf = requestAnimationFrame(() => inputRef.current?.select());
    return () => cancelAnimationFrame(raf);
  }, [open]);

  const create = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    onCreate(trimmed);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="New flow"
      description="Name it now, or rename it any time from its title."
      icon={<Plus size={16} />}
      tone="accent"
      size="sm"
      initialFocusRef={inputRef}
      testId="new-flow-dialog"
      footer={
        <>
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" form="oaiy-new-flow" className="btn btn-primary" disabled={!name.trim()}>
            Create flow
          </button>
        </>
      }
    >
      <form
        id="oaiy-new-flow"
        onSubmit={(e) => {
          e.preventDefault();
          create();
        }}
      >
        <label className="oaiy-field">
          <span>Name</span>
          <input
            ref={inputRef}
            type="text"
            className="oaiy-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={120}
            aria-label="Flow name"
          />
        </label>
      </form>
    </Dialog>
  );
}
