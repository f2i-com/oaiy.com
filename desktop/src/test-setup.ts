import { vi } from 'vitest';

// React 19 requires this before act() is used with react-dom/client, or React
// warns and batching behaves inconsistently between tests.
(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom does not implement window.confirm: unstubbed it logs "Not implemented"
// and returns undefined, which reads as Cancel — so every destructive-action
// test would pass for the wrong reason. Default to "user confirmed"; a test that
// cares overrides with vi.spyOn(window, 'confirm').mockReturnValue(false).
vi.stubGlobal('confirm', () => true);
