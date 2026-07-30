/**
 * Settings Tab Components
 *
 * Extracted from SettingsPanel.tsx for maintainability. Each tab is a
 * self-contained component with its own state and logic.
 *
 * `ApiServerTab` + `ModelsTab` lived here in the desktop build for
 * managing the Tauri-hosted HTTP API + the native model-download
 * pipeline (plugin-oaiy-diffusion). Both are desktop-only — the web
 * build doesn't render them, so the source files are gone and the
 * exports along with them.
 */

export { default as AppearanceTab } from './AppearanceTab';
export { default as SecurityTab } from './SecurityTab';
export { default as ServicesTab } from './ServicesTab';
