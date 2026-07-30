/**
 * Central registry of every localStorage key the app uses.
 *
 * Two prefixes exist for historical reasons:
 *  - `oaiy.<area>.<name>`  — current convention (dot-separated). All new
 *    keys should use this; lets `localStorage` be enumerated/cleared
 *    by area without false positives.
 *  - `oaiy_<name>`         — legacy. Predates the convention. Kept as-is
 *    so existing users don't lose state on upgrade; a real migration
 *    would require a one-time copy + delete + bumped schema version
 *    and the payoff didn't justify the risk surface at the time.
 *
 * If you're adding a new key, put it here and use the dot form. If
 * you're moving an existing legacy key, do it in a release with a
 * versioned migration and announce it.
 *
 * The constants below are typed as `as const` so each export is a
 * string literal — TypeScript will narrow callers to the registered
 * names instead of accepting arbitrary strings.
 */

export const STORAGE_KEYS = {
  // -----------------------------------------------------------------
  // Convention: oaiy.<area>.<name>
  // -----------------------------------------------------------------
  /** Boolean: backend sharing toggle (Settings → Defaults). */
  BACKEND_ENABLED: 'oaiy.backendEnabled',
  /** ShareState JSON for the currently-active backend share. */
  ACTIVE_SHARE: 'oaiy.activeShare',
  /** Map<hashEdit, password>. Per-flow remembered encryption keys. */
  FLOW_PASSWORD_PREFIX: 'oaiy.flowPassword.',
  /** Map<hash, true>. "Don't ask me again" approval persistence per share. */
  AUTO_APPROVE_RUNS_PREFIX: 'oaiy.autoApproveRuns.',
  /** Custom Services registry — populated via Settings → Services. */
  CUSTOM_SERVICES: 'oaiy.customServices',
  /** Boolean: DataViewer dev-mode toggle (raw JSON view, etc.). */
  DATAVIEWER_DEV: 'oaiy.dataviewer.developerMode',
  /** Boolean: OAIYBuilder palette collapsed state (left panel). */
  UI_PALETTE_COLLAPSED: 'oaiy.ui.paletteCollapsed',
  /** Boolean: OAIYBuilder properties panel visibility (right top). */
  UI_PROPERTIES_VISIBLE: 'oaiy.ui.propertiesVisible',
  /** Boolean: OAIYBuilder execution log panel visibility (right bottom). */
  UI_LOG_PANEL_VISIBLE: 'oaiy.ui.logPanelVisible',

  // -----------------------------------------------------------------
  // Legacy keys (predate the dot convention) — kept verbatim
  // -----------------------------------------------------------------
  /** The user's whole project state (flows, settings, constants, …). */
  LEGACY_PROJECT: 'oaiy_project',
  /** ThemeContext: 'light' | 'dark'. */
  LEGACY_THEME: 'oaiy_theme',
  /** ThemeContext: accent color id. */
  LEGACY_ACCENT_COLOR: 'oaiy_accent_color',
  /** ThemeContext: background tint id. */
  LEGACY_BG_TINT: 'oaiy_bg_tint',
  /** JobQueue run history snapshot. */
  LEGACY_RUN_HISTORY: 'oaiy_run_history',
  /** User-saved macros (v2 schema). */
  LEGACY_USER_MACROS_V2: 'oaiy_user_macros_v2',
  /** Workflow autosave checkpoint. */
  LEGACY_WORKFLOW_AUTOSAVE: 'oaiy_workflow_autosave',
  /** Workflow clipboard (nodes copied between flows). */
  LEGACY_WORKFLOW_CLIPBOARD: 'oaiy_workflow_clipboard',

  // -----------------------------------------------------------------
  // Bare keys (no prefix) — should never grow this list
  // -----------------------------------------------------------------
  /** Boolean flag the desktop API server used to set; web reads it once. */
  SKIP_SPLASH: 'skipSplash',
} as const;

export type StorageKey = typeof STORAGE_KEYS[keyof typeof STORAGE_KEYS];

/**
 * Clear every key this app owns. Useful for a "reset" button or test
 * teardown. Does NOT touch unrelated keys (e.g. tracking cookies
 * other libraries write to localStorage).
 */
export function clearAllAppStorage(): void {
  for (const key of Object.values(STORAGE_KEYS)) {
    // Some keys are prefixes (flow passwords / auto-approve flags) —
    // remove every key that starts with them.
    if (key.endsWith('.')) {
      const keysToRemove: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith(key)) keysToRemove.push(k);
      }
      keysToRemove.forEach((k) => localStorage.removeItem(k));
    } else {
      localStorage.removeItem(key);
    }
  }
}
