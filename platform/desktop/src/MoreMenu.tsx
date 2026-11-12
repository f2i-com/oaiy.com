import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { MoreHorizontal } from 'lucide-react';

/**
 * A row's less-used actions behind one "⋯" button: a small menu that opens under it (above it where there is no
 * room below), so a row keeps the one or two buttons a person presses every day and the rest are a click away.
 *
 * The menu is drawn at the top of the page, not inside the row: a card clips what hangs outside it. It closes on
 * Escape (focus goes back to the button), on a press anywhere else, and when the page scrolls or the window changes
 * size, as the row it belongs to has moved. Up, Down, Home and End move between its items.
 */

export interface MoreItem {
  key: string;
  label: string;
  icon?: ReactNode;
  onSelect: () => void;
  disabled?: boolean;
  /** Something that removes or deletes: in the danger colour. */
  danger?: boolean;
  /** A line on what it does, or why it is off. */
  hint?: string;
}

export default function MoreMenu({ label, items }: { label: string; items: MoreItem[] }) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<{ top: number; left: number } | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);

  const close = (refocus = false) => {
    setOpen(false);
    setPlace(null);
    if (refocus) button.current?.focus();
  };

  // Under the button, its right edge on the button's (kept inside the window, a little off its edges, where the
  // button is near the left one); above the button where it would run off the foot of the window.
  useLayoutEffect(() => {
    if (!open || !button.current || !menu.current) return;
    const b = button.current.getBoundingClientRect();
    const m = menu.current.getBoundingClientRect();
    const below = b.bottom + 4;
    const top = below + m.height > window.innerHeight - 8 && b.top - 4 - m.height >= 8 ? b.top - 4 - m.height : below;
    const left = Math.max(8, Math.min(b.right - m.width, window.innerWidth - m.width - 8));
    setPlace({ top, left });
    menu.current.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const outside = (e: Event) => {
      const t = e.target as Node | null;
      if (t && (menu.current?.contains(t) || button.current?.contains(t))) return;
      close();
    };
    const away = () => close();
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('scroll', away, true);
    window.addEventListener('resize', away);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('scroll', away, true);
      window.removeEventListener('resize', away);
    };
  }, [open]);

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const all = [...(menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? [])];
    const i = all.indexOf(document.activeElement as HTMLButtonElement);
    const go = (to: number) => {
      e.preventDefault();
      all[(to + all.length) % all.length]?.focus();
    };
    if (e.key === 'Escape') {
      e.preventDefault();
      close(true);
    } else if (e.key === 'Tab') close();
    else if (e.key === 'ArrowDown') go(i + 1);
    else if (e.key === 'ArrowUp') go(i - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(all.length - 1);
  };

  if (!items.length) return null;
  return (
    <>
      <button
        ref={button}
        type="button"
        className="btn btn-ghost more-button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        onClick={() => (open ? close() : setOpen(true))}
      >
        <MoreHorizontal size={16} aria-hidden />
      </button>
      {open &&
        createPortal(
          <div
            ref={menu}
            className="more-menu"
            role="menu"
            aria-label={label}
            // (measured where it is first, then put in its place: not seen in the corner meanwhile)
            style={place ? { top: place.top, left: place.left } : { top: 0, left: 0, visibility: 'hidden' }}
            onKeyDown={onKey}
          >
            {items.map((item) => (
              <button
                key={item.key}
                type="button"
                role="menuitem"
                className={item.danger ? 'is-danger' : undefined}
                disabled={item.disabled}
                title={item.hint}
                onClick={() => {
                  close(true);
                  item.onSelect();
                }}
              >
                {item.icon}
                <span>
                  {item.label}
                  {item.hint && <small>{item.hint}</small>}
                </span>
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}
