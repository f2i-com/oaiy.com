/**
 * WelcomeWizard persistence helpers.
 *
 * Lives in its own file so vite-plugin-react can Fast-Refresh the
 * wizard component cleanly — the plugin requires component files to
 * export *only* components. Mixing these helpers with the default
 * WelcomeWizard export forced a full page reload on every edit.
 *
 * Storage key is registered in `storageKeys.ts` for completeness;
 * we hardcode the literal here to avoid an import cycle (storageKeys
 * is metadata-only, but importing it pulls in `clearAllAppStorage`
 * which isn't worth the dependency for two one-liners).
 */

const STORAGE_KEY = 'oaiy.wizard.completed';

/** True iff the user has dismissed or finished the wizard at least once. */
export function hasCompletedWizard(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    // Private-mode / restricted contexts: act as if it's the first
    // visit so the wizard still surfaces, then no-op the writes.
    return false;
  }
}

/** Mark the wizard as completed so it won't auto-open on next load. */
export function markWizardCompleted(): void {
  try {
    localStorage.setItem(STORAGE_KEY, 'true');
  } catch { /* see hasCompletedWizard */ }
}
