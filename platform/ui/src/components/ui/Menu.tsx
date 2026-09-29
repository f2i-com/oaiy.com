/**
 * Menu — a short list of actions opened from a button (or at the pointer, for
 * a right-click). Not a dialog: it blocks nothing, and any click elsewhere,
 * Escape or Tab closes it. The arrow keys move between its items.
 */
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

export interface MenuProps {
  open: boolean;
  onClose: () => void;
  /** The button it opens from: the menu lines up under it. */
  anchor?: HTMLElement | null;
  /** Or a point (a right-click). */
  at?: { x: number; y: number } | null;
  /** Which edge of the anchor it lines up with. */
  align?: 'start' | 'end';
  label: string;
  children: ReactNode;
}

export default function Menu({ open, onClose, anchor, at, align = 'end', label, children }: MenuProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    if (!open) { setPos(null); return; }
    const el = ref.current;
    const w = el?.offsetWidth ?? 240;
    const h = el?.offsetHeight ?? 200;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = 8;
    let top = 8;
    if (at) {
      left = at.x;
      top = at.y;
    } else if (anchor) {
      const r = anchor.getBoundingClientRect();
      left = align === 'end' ? r.right - w : r.left;
      top = r.bottom + 6;
      if (top + h > vh - 8) top = Math.max(8, r.top - h - 6);
    }
    setPos({ left: Math.max(8, Math.min(left, vw - w - 8)), top: Math.max(8, Math.min(top, vh - h - 8)) });
  }, [open, anchor, at, align]);

  useEffect(() => {
    if (!open) return;
    const items = () => Array.from(ref.current?.querySelectorAll<HTMLButtonElement>('button:not([disabled])') ?? []);
    const raf = requestAnimationFrame(() => items()[0]?.focus());
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !(anchor && anchor.contains(e.target as Node))) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose(); anchor?.focus(); return; }
      if (e.key === 'Tab') { onClose(); return; }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const list = items();
        const i = list.indexOf(document.activeElement as HTMLButtonElement);
        const next = e.key === 'ArrowDown' ? (i + 1) % list.length : (i - 1 + list.length) % list.length;
        list[next]?.focus();
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', onClose);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', onClose);
    };
  }, [open, onClose, anchor]);

  if (!open) return null;
  return createPortal(
    <div
      ref={ref}
      className="oaiy-menu"
      role="menu"
      aria-label={label}
      style={pos ? { left: pos.left, top: pos.top } : { left: -9999, top: -9999 }}
      // Every item closes the menu after doing its job.
      onClick={(e) => { if ((e.target as HTMLElement).closest('button')) onClose(); }}
    >
      {children}
    </div>,
    document.body,
  );
}

/** One item: an icon, a label, and a line under it if it needs one. */
export function MenuItem({
  icon,
  label,
  hint,
  onSelect,
  danger = false,
  disabled = false,
}: {
  icon?: ReactNode;
  label: ReactNode;
  hint?: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <button type="button" role="menuitem" className={danger ? 'danger' : undefined} onClick={onSelect} disabled={disabled}>
      {icon}
      <span>
        {label}
        {hint && <small>{hint}</small>}
      </span>
    </button>
  );
}
