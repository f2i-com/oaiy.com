/**
 * useFocusTrap — capture focus inside a modal container.
 *
 * Three jobs:
 *   1. On open, save the element that had focus and move focus into
 *      the modal — preferring an explicit `initialFocusRef`, else
 *      the first focusable descendant, else the container itself.
 *   2. While open, keep Tab/Shift+Tab cycling within the container's
 *      focusable descendants. Tabbing past the last element wraps to
 *      the first; Shift+Tab past the first wraps to the last.
 *   3. On close, restore focus to the element that had it before
 *      open — so keyboard users land back where they invoked the
 *      modal from, instead of at <body>.
 *
 * The hook is unopinionated about Escape handling and click-outside —
 * each modal already wires those itself; this hook only handles
 * focus trapping + return.
 *
 * Limitations:
 *   - Doesn't trap focus from screen readers' virtual cursor (browser
 *     a11y APIs do that via aria-modal="true" — make sure the modal
 *     element has it).
 *   - Doesn't disable scroll on the body — that's a separate concern
 *     (some modals are drawers; locking body scroll would jump).
 *   - Uses a snapshot of focusables computed each Tab press, so the
 *     trap stays correct if the modal's children change.
 */
import { useEffect, type RefObject } from 'react';

/**
 * Selector matches the standard set of tab-stops. Excludes elements
 * with `tabindex="-1"` because those are programmatically focusable
 * but not part of the Tab sequence.
 */
const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  'audio[controls]',
  'video[controls]',
  '[contenteditable]:not([contenteditable="false"])',
].join(',');

function focusables(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((el) => !el.hasAttribute('disabled') && el.offsetParent !== null);
}

export function useFocusTrap(
  containerRef: RefObject<HTMLElement | null>,
  isOpen: boolean,
  initialFocusRef?: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!isOpen) return;
    const container = containerRef.current;
    if (!container) return;

    // Save who had focus so we can restore on unmount/close.
    const previouslyFocused = document.activeElement as HTMLElement | null;

    // Initial focus: explicit ref → first focusable → container itself
    // (with tabindex=-1 so it can receive focus at all).
    const initial = initialFocusRef?.current
      ?? focusables(container)[0]
      ?? container;
    if (initial === container && !container.hasAttribute('tabindex')) {
      container.setAttribute('tabindex', '-1');
    }
    // requestAnimationFrame ensures the modal is fully painted before
    // we set focus — without it, screen readers sometimes miss the
    // move and announce the previously-focused element first.
    const raf = requestAnimationFrame(() => initial.focus());

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      const items = focusables(container);
      if (items.length === 0) {
        // Nothing tabbable inside; keep focus on the container.
        e.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey) {
        // Shift+Tab on first → wrap to last.
        if (active === first || !container.contains(active)) {
          e.preventDefault();
          last.focus();
        }
      } else {
        // Tab on last → wrap to first.
        if (active === last || !container.contains(active)) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener('keydown', onKeyDown);
      // Return focus to wherever it was, unless that element is no
      // longer in the document (e.g., the trigger was unmounted while
      // the modal was open). Skip silently in that case.
      if (previouslyFocused && document.contains(previouslyFocused)) {
        try { previouslyFocused.focus(); } catch { /* element not focusable anymore */ }
      }
    };
  }, [isOpen, containerRef, initialFocusRef]);
}
