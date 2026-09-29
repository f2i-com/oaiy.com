import { listen } from '@tauri-apps/api/event';
import { isTauri } from './api';

/**
 * The desktop asks the dashboard to show a page: the Agent's `plugin_setup_open`
 * and `ui_open` tools (CONTROL_API.md §1, "Showing a page") emit the Tauri
 * event `oaiy://navigate` with `{ view, pluginId?, stepId? }`:
 *
 * - `view: "setup"`: the setup page; with `pluginId`, that plugin's own
 *   wizard; with `stepId`, open at that step;
 * - `view: "plugin:<id>:<nav>"`: a plugin's page;
 * - `view: "contacts"`: Contacts; with `contact` (a number's key, its last
 *   nine digits), that person's contact open;
 * - any other: one of the dashboard's own views (`NAV_VIEWS`).
 *
 * Anything else is ignored. In a plain browser (no Tauri) nothing listens.
 */

export const NAVIGATE_EVENT = 'oaiy://navigate';

/** The dashboard's own views a navigation may name (App.tsx's `BuiltinView`, without the setup page). */
export const NAV_VIEWS = [
  'overview',
  'agent',
  'flows',
  'runs',
  'calendar',
  /** The people who ring and text (under the AI Receptionist while a plugin provides the phone). */
  'contacts',
  /** What callers left when the receptionist could not put them through (under the AI Receptionist while a plugin provides the phone). */
  'messages',
  /** Whether the receptionist may try to reach the owner for a caller who asks, and how. */
  'transfers',
  /** Hours & Services: the business the receptionist speaks for (under the AI Receptionist while a plugin provides it). */
  'hours',
  'engines',
  'models',
  'services',
  'python',
  'connections',
  'providers',
  'plugins',
  'settings',
  'agent-settings',
] as const;
export type NavView = (typeof NAV_VIEWS)[number];

export type NavigateTarget =
  | { kind: 'setup'; pluginId?: string; stepId?: string }
  | { kind: 'view'; view: NavView | `plugin:${string}:${string}`; contact?: string };

const PLUGIN_ID = /^[a-z0-9_-]{1,64}$/;
/** A step id, or a first-run step with a plugin after a colon (setup.rs `valid_step_id`). */
const STEP_ID = /^[a-z][a-z0-9-]{0,39}(:[a-z0-9_-]{1,64})?$/;
const PLUGIN_VIEW = /^plugin:[a-z0-9_-]{1,64}:[A-Za-z0-9_.-]{1,64}$/;
/** A contact's key: a number's last nine digits (eight for an eight-digit number). */
const CONTACT_KEY = /^\d{8,9}$/;

/** What a navigation asks for, or `null` for one to ignore. */
export function parseNavigate(payload: unknown): NavigateTarget | null {
  if (!payload || typeof payload !== 'object') return null;
  const { view, pluginId, stepId, contact } = payload as Record<string, unknown>;
  if (typeof view !== 'string') return null;
  if (view === 'setup') {
    const target: NavigateTarget = { kind: 'setup' };
    if (typeof pluginId === 'string' && PLUGIN_ID.test(pluginId)) target.pluginId = pluginId;
    if (typeof stepId === 'string' && STEP_ID.test(stepId)) target.stepId = stepId;
    return target;
  }
  if (PLUGIN_VIEW.test(view)) return { kind: 'view', view: view as `plugin:${string}:${string}` };
  if (view === 'contacts' && typeof contact === 'string' && CONTACT_KEY.test(contact)) return { kind: 'view', view, contact };
  if ((NAV_VIEWS as readonly string[]).includes(view)) return { kind: 'view', view: view as NavView };
  return null;
}

/** Called with each navigation the desktop asks for. A no-op outside OAIY's window. Returns a stop function. */
export function onNavigate(cb: (target: NavigateTarget) => void): () => void {
  if (!isTauri()) return () => {};
  let stopped = false;
  let unlisten: (() => void) | null = null;
  listen<unknown>(NAVIGATE_EVENT, (event) => {
    const target = parseNavigate(event.payload);
    if (target && !stopped) cb(target);
  }).then(
    (un) => {
      if (stopped) un();
      else unlisten = un;
    },
    () => undefined,
  );
  return () => {
    stopped = true;
    unlisten?.();
  };
}
