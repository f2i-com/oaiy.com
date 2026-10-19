// Leaving a page with changes not saved yet: asked first, and the answer awaited (the question is a dialog now, which
// answers later), so a "no" keeps the person where they are.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { confirmLeaving, forgetUnsaved, hasUnsaved, markUnsaved, whenLeaving } from './unsavedChanges';

const settle = () => new Promise((r) => setTimeout(r, 0));

afterEach(() => {
  forgetUnsaved();
  vi.stubGlobal('confirm', () => true);
});

describe('leaving unsaved changes', () => {
  it('leaves at once, unasked, where nothing is unsaved', () => {
    const confirm = vi.fn(() => true);
    vi.stubGlobal('confirm', confirm);
    const leave = vi.fn();
    whenLeaving(leave);
    expect(leave).toHaveBeenCalledTimes(1);
    expect(confirm).not.toHaveBeenCalled();
  });

  it('asks where something is unsaved, and stays on a no', async () => {
    markUnsaved('hours', true, 'the business hours');
    const confirm = vi.fn(() => false);
    vi.stubGlobal('confirm', confirm);
    const leave = vi.fn();
    whenLeaving(leave);
    await settle();
    expect(confirm.mock.calls[0]).toEqual(['Leave without saving the business hours? Your changes are not saved yet.']);
    expect(leave).not.toHaveBeenCalled();
    expect(hasUnsaved()).toBe('the business hours');
  });

  it('leaves on a yes, and forgets what was unsaved', async () => {
    markUnsaved('hours', true, 'the business hours');
    markUnsaved('transfers', true, 'the transfer settings');
    vi.stubGlobal('confirm', () => true);
    const leave = vi.fn();
    whenLeaving(leave);
    expect(leave).not.toHaveBeenCalled();
    await settle();
    expect(leave).toHaveBeenCalledTimes(1);
    expect(hasUnsaved()).toBeNull();
    expect(await confirmLeaving()).toBe(true);
  });
});
