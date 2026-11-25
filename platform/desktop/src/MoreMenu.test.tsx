import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import MoreMenu from './MoreMenu';

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
});

describe('MoreMenu focus & keyboard navigation', () => {
  it('focuses the first enabled menuitem only after visibility:hidden is removed and supports ArrowDown/Escape', () => {
    const focusCalls: { text: string; visibility: string }[] = [];
    const origFocus = HTMLElement.prototype.focus;
    vi.spyOn(HTMLElement.prototype, 'focus').mockImplementation(function (this: HTMLElement, opts?: FocusOptions) {
      const menu = this.closest<HTMLElement>('[role="menu"]');
      if (menu) {
        focusCalls.push({
          text: this.textContent?.trim() ?? '',
          visibility: menu.style.visibility,
        });
      }
      origFocus.call(this, opts);
    });

    const onSelect = vi.fn();
    act(() => {
      root.render(
        <MoreMenu
          label="More actions"
          items={[
            { key: 'disabled', label: 'Disabled item', disabled: true, onSelect },
            { key: 'first', label: 'First enabled', onSelect },
            { key: 'second', label: 'Second enabled', onSelect },
          ]}
        />,
      );
    });

    const trigger = host.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    act(() => {
      trigger.click();
    });

    const menu = document.body.querySelector<HTMLDivElement>('[role="menu"]')!;
    expect(menu).toBeTruthy();
    expect(menu.style.visibility).toBe('');
    // Focus must have been requested once, on the first enabled item, AFTER visibility:hidden was gone.
    expect(focusCalls).toEqual([{ text: 'First enabled', visibility: '' }]);
    expect(document.activeElement?.textContent?.trim()).toBe('First enabled');

    // ArrowDown moves to the second enabled item.
    act(() => {
      menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    });
    expect(document.activeElement?.textContent?.trim()).toBe('Second enabled');

    // Escape closes the menu and restores focus to the trigger button.
    act(() => {
      menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    expect(document.body.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });
});
