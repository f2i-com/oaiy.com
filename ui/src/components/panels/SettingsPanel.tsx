import { useState, useCallback, useEffect, useRef } from 'react';
import type { ProjectConstant, ProjectSettings, AIProvider } from 'oaiy-core';
import { AI_PROVIDER_INFO } from 'oaiy-core';
import { useServices } from '../../hooks/useServices';
import { AppearanceTab, SecurityTab, ServicesTab } from './settings';
import { uiLogger as logger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';
import {
  backendBaseUrl,
  isSharingEnabled,
  setSharingEnabled,
} from '../../lib/sharingPrefs';
import {
  zippSandboxEnabled,
  zippSandboxSupported,
  setZippSandboxEnabled,
} from '../../lib/zippPrefs';

/**
 * Renders the privacy badges for the currently-selected AI provider.
 * Driven entirely by the static metadata in `AI_PROVIDER_INFO`, so adding
 * a new provider only requires touching `oaiy-core/src/types.ts`.
 *
 * The badges sit immediately below the provider <select> so the user
 * sees what data leaves their machine before they pick.
 */
function AIProviderPrivacyBadges({ provider }: { provider: AIProvider }) {
  const info = AI_PROVIDER_INFO[provider];
  if (!info) return null;

  // Visual style varies by category — local-only is calm green, anything
  // outbound is emphasised so it's hard to miss in a busy form.
  const badgeClasses: Record<string, string> = {
    'local-only':
      'bg-green-100 text-green-700 border-green-300 dark:bg-green-900/30 dark:text-green-300 dark:border-green-600/40',
    prompt:
      'bg-amber-100 text-amber-700 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-600/40',
    files:
      'bg-amber-100 text-amber-700 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-600/40',
    images:
      'bg-amber-100 text-amber-700 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-600/40',
    screenshots:
      'bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-600/40',
    terminal:
      'bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-600/40',
    browser:
      'bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-600/40',
  };
  const labels: Record<string, string> = {
    'local-only': '🛡 Local only',
    prompt: '↗ Sends prompt',
    files: '↗ Sends files',
    images: '↗ Sends images',
    screenshots: '↗ Sends screenshots',
    terminal: '↗ Sends terminal output',
    browser: '↗ Sends browser content',
  };

  return (
    <div className="mt-2">
      <div className="flex flex-wrap gap-1.5 mb-1">
        {info.dataFlows.map((flow) => (
          <span
            key={flow}
            className={`text-xs px-2 py-0.5 rounded-full border font-medium ${badgeClasses[flow] ?? ''}`}
            title={
              flow === 'local-only'
                ? 'Runs entirely on this machine — no outbound network calls beyond localhost.'
                : `${labels[flow] ?? flow} from this machine to the provider.`
            }
          >
            {labels[flow] ?? flow}
          </span>
        ))}
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400">{info.description}</p>
    </div>
  );
}

type SettingsTab =
  | 'appearance'
  | 'defaults'
  | 'apikeys'
  | 'constants'
  | 'security'
  | 'services';
// Desktop-only tabs (plugins / api / models) were removed in the web build:
//  - plugins = App Data Folder, has no meaning without a real filesystem
//  - api = Tauri-hosted HTTP API server, no equivalent in the browser
//  - models = native model-download manager (plugin-oaiy-diffusion), web
//    talks to whatever local engine the user already runs — see the
//    Services tab for HTTP-endpoint registration instead.

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: ProjectSettings;
  constants: ProjectConstant[];
  onUpdateSettings: (updates: Partial<ProjectSettings>) => void;
  onUpdateConstant: (id: string, updates: Partial<ProjectConstant>) => void;
  onCreateConstant: (constant: Omit<ProjectConstant, 'id'>) => void;
  onDeleteConstant?: (id: string) => void;
  onExportSettings?: () => void;
  onImportSettings?: (settings: { settings: ProjectSettings; constants: ProjectConstant[] }) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
  /**
   * Optional initial tab when the panel opens. Callers that deep-link
   * (e.g. the agent panel's "Manage in Settings…" affordance) pass
   * `'models'` so the user lands on the model catalogue without having
   * to navigate. Re-applied every time `isOpen` flips to true; ignored
   * during a session where the user has clicked into another tab.
   */
  initialTab?: SettingsTab;
}

// Per-category provider <select> arrays + their `handleAIProviderChange`/
// `handleImageProviderChange`/`handleVideoProviderChange` callbacks used to
// live here, driving a row of Provider+Endpoint+Model fields on the
// Defaults tab. The unified Services model replaced all of that — the
// Defaults tab now offers a Service picker per category and the
// `defaultAIServiceId` / `defaultImageServiceId` / `defaultVideoServiceId`
// fields are the source of truth.

export default function SettingsPanel({
  isOpen,
  onClose,
  settings,
  constants,
  onUpdateSettings,
  onUpdateConstant,
  onCreateConstant,
  onDeleteConstant,
  onExportSettings,
  onImportSettings,
  onShowToast,
  initialTab,
}: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab ?? 'appearance');
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, isOpen);
  const confirm = useConfirmDialog();
  // Re-apply `initialTab` every time the panel opens so callers can
  // deep-link to a specific section (e.g. Models). Doesn't fire on tab
  // clicks within an open session.
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, initialTab]);
  const [newConstantName, setNewConstantName] = useState('');
  const [newConstantKey, setNewConstantKey] = useState('');
  const [newConstantValue, setNewConstantValue] = useState('');
  // Separate add-form state for API Keys so the constants form isn't
  // surprised by api_key entries appearing in its list mid-edit.
  const [newApiKeyName, setNewApiKeyName] = useState('');
  const [newApiKeyKey, setNewApiKeyKey] = useState('');
  const [newApiKeyValue, setNewApiKeyValue] = useState('');

  // Services hook - only used for runningCount in sidebar badge
  const { runningCount } = useServices({ autoRefresh: isOpen && activeTab === 'services', refreshInterval: 3000 });

  // App-data-folder state, module loader state, API-server probe, and
  // the per-module setting handlers used to live here, driving a
  // (now-removed) `plugins` tab that surfaced the desktop's App Data
  // Folder picker + bundled-module listing + per-module setting UI.
  // The web build statically discovers modules at build time via
  // `import.meta.glob` and has no real filesystem to point at, so all
  // of that is gone — see the comment on `SettingsTab` above.

  const handleExportSettings = useCallback(() => {
    if (onExportSettings) {
      onExportSettings();
    } else {
      // Default export behavior
      const exportData = {
        settings,
        constants: constants.map(c => ({ ...c, value: c.isSecret ? '' : c.value })), // Don't export secret values
      };
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'oaiy-settings.json';
      a.click();
      URL.revokeObjectURL(url);
      onShowToast?.('Settings exported to Downloads', 'success');
    }
  }, [settings, constants, onExportSettings, onShowToast]);

  const handleImportSettings = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        try {
          const text = await file.text();
          const data = JSON.parse(text);
          if (data.settings && onImportSettings) {
            onImportSettings(data);
            onShowToast?.('Settings imported successfully', 'success');
          } else if (data.settings) {
            // Apply settings directly
            onUpdateSettings(data.settings);
            // Import constants (but preserve existing secret values)
            if (data.constants) {
              data.constants.forEach((imported: ProjectConstant) => {
                const existing = constants.find(c => c.key === imported.key);
                if (existing) {
                  // Update existing constant, preserve value if it's secret and import has empty value
                  const value = imported.isSecret && !imported.value ? existing.value : imported.value;
                  onUpdateConstant(existing.id, { ...imported, value });
                } else {
                  // Create new constant
                  onCreateConstant(imported);
                }
              });
            }
            onShowToast?.('Settings imported successfully', 'success');
          }
        } catch (err) {
          logger.error('Failed to import settings', { error: err });
          onShowToast?.('Failed to import settings. Check file format.', 'error');
        }
      }
    };
    input.click();
  }, [constants, onUpdateSettings, onUpdateConstant, onCreateConstant, onImportSettings, onShowToast]);

  const handleAddConstant = useCallback(() => {
    if (newConstantName && newConstantKey) {
      onCreateConstant({
        name: newConstantName,
        key: newConstantKey.toUpperCase().replace(/[^A-Z0-9_]/g, '_'),
        value: newConstantValue,
        category: 'custom',
        isSecret: false,
      });
      setNewConstantName('');
      setNewConstantKey('');
      setNewConstantValue('');
    }
  }, [newConstantName, newConstantKey, newConstantValue, onCreateConstant]);

  const handleAddApiKey = useCallback(() => {
    if (newApiKeyName && newApiKeyKey) {
      onCreateConstant({
        name: newApiKeyName,
        key: newApiKeyKey.toUpperCase().replace(/[^A-Z0-9_]/g, '_'),
        value: newApiKeyValue,
        category: 'api_key',
        isSecret: true,
      });
      setNewApiKeyName('');
      setNewApiKeyKey('');
      setNewApiKeyValue('');
    }
  }, [newApiKeyName, newApiKeyKey, newApiKeyValue, onCreateConstant]);

  // Escape-to-close while open. Window-level so it fires regardless
  // of focus inside the form. Matches the other modal dialogs.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const apiKeyConstants = constants.filter(c => c.category === 'api_key');
  const customConstants = constants.filter(c => c.category !== 'api_key');

  return (
    <div
      className="fixed inset-x-0 bottom-0 top-[65px] md:top-0 bg-black/50 z-50 flex items-center justify-center p-0 sm:p-4 short:p-2"
      // Click the backdrop to dismiss, matching the Escape key this panel
      // already honours. target === currentTarget so a click that merely
      // BUBBLED from the card does not close it -- no stopPropagation needed
      // on the card itself, which would have to be repeated on every child
      // that stops propagation for its own reasons.
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Outer card — stacks vertically on mobile (header + tab strip on
          top, content below) and switches to the desktop two-column layout
          at md+. The flex direction toggle is what enables both modes. */}
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-800 rounded-none sm:rounded-lg shadow-xl w-full h-full sm:h-[80vh] sm:max-w-5xl short:h-[94vh] overflow-hidden flex flex-col md:flex-row"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-panel-title"
      >
        {/* Sidebar / mobile top bar — full-width with horizontal-scroll
            tabs on mobile, fixed-width vertical column on desktop. */}
        <div className="md:w-48 md:flex-shrink-0 bg-slate-50 dark:bg-slate-900/50 border-b md:border-b-0 md:border-r border-slate-200 dark:border-slate-700 flex flex-col">
          {/* Header */}
          <div className="px-4 py-3 md:py-4 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
            <h2 id="settings-panel-title" className="text-lg font-semibold text-slate-700 dark:text-slate-200">Settings</h2>
          </div>

          {/* Navigation — horizontal scroll on mobile, vertical column on
              desktop. The class swap on the buttons themselves (md:w-full
              vs flex-shrink-0 whitespace-nowrap) keeps each one a sensible
              width in both orientations. */}
          <nav className="flex md:flex-col md:flex-1 p-2 gap-1 md:gap-0 md:space-y-1 overflow-x-auto md:overflow-x-visible md:overflow-y-auto">
            {/* Tabs collapse to icon-only below the `sm` breakpoint
                (640px) — the label is wrapped in `hidden sm:inline`,
                title="…" provides the hover tooltip on desktop, and
                aria-label keeps screen-readers happy in both modes.
                All six tabs share this pattern so any width below
                sm renders as 6 icons with no horizontal scroll. */}
            <button
              onClick={() => setActiveTab('appearance')}
              title="Appearance"
              aria-label="Appearance"
              className={`md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'appearance'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
              <span className="hidden sm:inline">Appearance</span>
            </button>
            <button
              onClick={() => setActiveTab('defaults')}
              title="Defaults"
              aria-label="Defaults"
              className={`md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'defaults'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
              </svg>
              <span className="hidden sm:inline">Defaults</span>
            </button>
            <button
              onClick={() => setActiveTab('apikeys')}
              title="API Keys"
              aria-label="API Keys"
              className={`md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'apikeys'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
              </svg>
              <span className="hidden sm:inline">API Keys</span>
            </button>
            <button
              onClick={() => setActiveTab('constants')}
              title="Constants"
              aria-label="Constants"
              className={`md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'constants'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span className="hidden sm:inline">Constants</span>
            </button>
            <button
              onClick={() => setActiveTab('security')}
              title="Security"
              aria-label="Security"
              className={`md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'security'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span className="hidden sm:inline">Security</span>
            </button>
            <button
              onClick={() => setActiveTab('services')}
              title={`Services${runningCount > 0 ? ` (${runningCount} running)` : ''}`}
              aria-label="Services"
              className={`relative md:w-full flex-shrink-0 whitespace-nowrap px-2 sm:px-3 py-2 text-sm font-medium rounded-md transition-colors text-left flex items-center gap-0 sm:gap-2 ${
                activeTab === 'services'
                  ? 'bg-blue-600/20 text-blue-600 dark:text-blue-400'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-700/50'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
              </svg>
              <span className="hidden sm:inline">Services</span>
              {/* On mobile (icon-only) the running indicator collapses
                  to a small green dot in the corner to keep the icon
                  size unchanged. The numeric chip returns at sm+. */}
              {runningCount > 0 && (
                <>
                  <span className="sm:hidden absolute top-1 right-1 w-2 h-2 bg-green-500 rounded-full" />
                  <span className="hidden sm:inline px-1.5 py-0.5 bg-green-600/30 text-green-400 text-xs rounded-full ml-auto">
                    {runningCount}
                  </span>
                </>
              )}
            </button>
          </nav>
        </div>

        {/* Main content area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700">
            <h3 className="text-base font-medium text-slate-700 dark:text-slate-200 capitalize">{activeTab === 'apikeys' ? 'API Keys' : activeTab}</h3>
          </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === 'appearance' && <AppearanceTab />}


          {activeTab === 'defaults' && (
            <div className="space-y-6">
              {/* Info Banner — the per-category "Default Service" pickers used
                  to live here so a fresh node would auto-pick the user's go-to
                  service. Removed because each user runs different local
                  engines + custom endpoints, so a global default rarely
                  matches what the next node actually needs; the picker in the
                  node's Properties Panel is the source of truth instead. */}
              <div className="bg-slate-100/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
                <p className="text-slate-400 text-sm">
                  Backup or restore your settings + project constants below.
                  To register services, go to <strong>Settings → Services</strong>;
                  to pick a service for a specific node, click the node on the
                  canvas and use the <strong>Service Preset</strong> dropdown
                  in the properties panel.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleExportSettings}
                  className="btn btn-secondary btn-md"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export Settings
                </button>
                <button
                  onClick={handleImportSettings}
                  className="btn btn-secondary btn-md"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Import Settings
                </button>
              </div>

              {/* Backend sharing toggle ----------------------------- */}
              <SharingToggle />

              {/* Untrusted-workflow engine -------------------------- */}
              <ZippSandboxToggle />

              <div className="space-y-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                {/* Slot left for future UI/theme toggles; AppearanceTab owns
                    the live ones today. */}
              </div>
            </div>
          )}

          {activeTab === 'apikeys' && (
            <div className="space-y-4">
              <div className="bg-slate-100/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
                <p className="text-slate-400 text-sm">
                  Add your own API keys. Reference them by name from any node
                  that supports an API Key Constant (e.g. <code>OPENAI_API_KEY</code>).
                  Keys are stored locally in the project and never sent anywhere
                  except to the endpoint you point a node at.
                </p>
              </div>

              {/* Add New API Key — same shape as Constants, but category=api_key + isSecret=true */}
              <div className="bg-slate-100/50 dark:bg-slate-700/30 rounded-lg p-4 space-y-3">
                <h4 className="text-sm font-medium text-slate-400">Add New API Key</h4>
                <div className="grid grid-cols-3 gap-2">
                  <input
                    type="text"
                    value={newApiKeyName}
                    onChange={(e) => setNewApiKeyName(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-yellow-500"
                    placeholder="Display Name"
                  />
                  <input
                    type="text"
                    value={newApiKeyKey}
                    onChange={(e) => setNewApiKeyKey(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-yellow-500 font-mono"
                    placeholder="MY_API_KEY"
                  />
                  <input
                    type="password"
                    value={newApiKeyValue}
                    onChange={(e) => setNewApiKeyValue(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-yellow-500 font-mono"
                    placeholder="Key value (optional)"
                  />
                </div>
                <button
                  onClick={handleAddApiKey}
                  disabled={!newApiKeyName || !newApiKeyKey}
                  className="btn btn-primary btn-sm"
                >
                  Add API Key
                </button>
              </div>

              {/* Existing API Keys */}
              {apiKeyConstants.length === 0 ? (
                <p className="text-slate-500 dark:text-slate-400 text-sm text-center py-6 italic">
                  No API keys yet. Add one above — nodes that reference its name will pick it up automatically.
                </p>
              ) : (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-400">Your API Keys</h4>
                  {apiKeyConstants.map((constant) => (
                    <div key={constant.id} className="flex items-center gap-3">
                      <div className="flex-1">
                        <label className="text-slate-500 dark:text-slate-400 text-xs block mb-1">{constant.name}</label>
                        <div className="flex gap-2">
                          <input
                            type="password"
                            value={constant.value}
                            onChange={(e) => onUpdateConstant(constant.id, { value: e.target.value })}
                            className="flex-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-3 py-2 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-yellow-500 font-mono"
                            placeholder={`Enter ${constant.key}...`}
                          />
                          <span className="px-2 py-2 bg-slate-200 dark:bg-slate-700 rounded text-xs text-slate-600 dark:text-slate-400 font-mono">
                            {constant.key}
                          </span>
                        </div>
                      </div>
                      {onDeleteConstant && (
                        <button
                          onClick={async () => {
                            const ok = await confirm({
                              title: 'Delete API key?',
                              message: `"${constant.name}" (${constant.key}) will be removed from this project. Any service or flow that reads {{${constant.key}}} will need a new value.`,
                              variant: 'danger',
                              confirmLabel: 'Delete key',
                            });
                            if (ok) onDeleteConstant(constant.id);
                          }}
                          className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded text-slate-500 hover:text-red-400"
                          title="Delete"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'constants' && (
            <div className="space-y-4">
              <div className="bg-slate-100/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
                <p className="text-slate-400 text-sm">
                  Create custom constants for endpoints, models, or other values. Reference them by name in node configurations for easy updates.
                </p>
              </div>

              {/* Add New Constant */}
              <div className="bg-slate-100/50 dark:bg-slate-700/30 rounded-lg p-4 space-y-3">
                <h4 className="text-sm font-medium text-slate-400">Add New Constant</h4>
                <div className="grid grid-cols-3 gap-2">
                  <input
                    type="text"
                    value={newConstantName}
                    onChange={(e) => setNewConstantName(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-blue-500"
                    placeholder="Display Name"
                  />
                  <input
                    type="text"
                    value={newConstantKey}
                    onChange={(e) => setNewConstantKey(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
                    placeholder="CONSTANT_KEY"
                  />
                  <input
                    type="text"
                    value={newConstantValue}
                    onChange={(e) => setNewConstantValue(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-blue-500"
                    placeholder="Value"
                  />
                </div>
                <button
                  onClick={handleAddConstant}
                  disabled={!newConstantName || !newConstantKey}
                  className="btn btn-primary btn-sm"
                >
                  Add Constant
                </button>
              </div>

              {/* Existing Constants */}
              {customConstants.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-400">Custom Constants</h4>
                  {customConstants.map((constant) => (
                    <div key={constant.id} className="flex items-center gap-2 bg-slate-100/50 dark:bg-slate-700/30 rounded px-3 py-2">
                      <span className="text-slate-400 text-sm flex-shrink-0">{constant.name}</span>
                      <span className="px-1.5 py-0.5 bg-slate-600 rounded text-xs text-slate-400 font-mono">{constant.key}</span>
                      <span className="text-slate-500 text-sm flex-1 truncate">{constant.value || '(empty)'}</span>
                      <input
                        type="text"
                        value={constant.value}
                        onChange={(e) => onUpdateConstant(constant.id, { value: e.target.value })}
                        className="w-40 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-700 dark:text-slate-200 focus:outline-none focus:border-blue-500"
                        placeholder="Value"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'security' && (
            <SecurityTab
              settings={settings}
              onUpdateSettings={onUpdateSettings}
              onShowToast={onShowToast}
            />
          )}
          {/* The 'plugins' tab (App Data Folder + plugin install instructions)
             was Tauri-specific — removed in the web build. Modules are
             statically discovered at build time via import.meta.glob; users
             don't install plugins by dropping folders here. */}

          {activeTab === 'services' && (
            <ServicesTab
              isOpen={isOpen}
              settings={settings}
              constants={constants}
              onUpdateSettings={onUpdateSettings}
            />
          )}
        </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-slate-200 dark:border-slate-700 flex justify-end">
            <button
              onClick={onClose}
              className="btn btn-primary btn-md"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ModuleSettingField was the per-module per-setting input renderer for
// the removed `plugins` tab. The web build doesn't surface module
// settings — modules ship inline with their declared defaults — so the
// component and its supporting types went with the tab. If/when a
// settings surface for plugin modules returns, restore from git history.

// ===========================================================================
// ZippSandboxToggle — Defaults tab section that picks the engine flows run on.
// ===========================================================================
//
// Package flows already run in a Web Worker, but that Worker is a full browser
// realm with the dangerous globals removed one name at a time — `runtime.ts`
// and `untrusted-executor.ts` both say in their own comments that this is
// best-effort. Zipp is a JavaScript engine compiled to WebAssembly whose guest
// global never held a host object at all, so a script that reconstructs
// `globalThis` there finds nothing to use. With the toggle on, the user's own
// flows run there too, which bounds a runaway code node's CPU and memory.
//
// On by default. It costs a ~1.2 MB engine download on the first run and runs
// interpreted rather than JIT-compiled, so the toggle is the escape hatch for
// a flow that trips over one of those. State lives in `localStorage` via
// `zippPrefs.ts` and is read when a run starts, so flipping it applies to the
// next run without a reload.
function ZippSandboxToggle() {
  const supported = zippSandboxSupported();
  const [enabled, setEnabled] = useState<boolean>(() => zippSandboxEnabled());

  const toggle = useCallback(() => {
    if (!supported) return;
    const next = !enabled;
    setZippSandboxEnabled(next);
    setEnabled(next);
  }, [supported, enabled]);

  return (
    <div className="space-y-3 pt-4 border-t border-slate-200 dark:border-slate-700">
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 flex items-center gap-2">
        <svg className="w-4 h-4" style={{ color: 'rgb(var(--accent-primary))' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
        Flow sandbox
      </h3>
      <label className={`flex items-start gap-3 ${supported ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'}`}>
        <input
          type="checkbox"
          checked={enabled}
          disabled={!supported}
          onChange={toggle}
          style={{ accentColor: 'rgb(var(--accent-primary))' }}
          className="mt-1 w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-offset-white dark:focus:ring-offset-slate-800"
        />
        <div className="flex-1">
          <div className="text-sm text-slate-700 dark:text-slate-200">
            Run flows on the Zipp engine
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Zipp is a separate JavaScript engine that runs inside WebAssembly
            and never had the browser&apos;s risky features: no network, no
            storage, no way to start more workers. Flow code reaches the
            outside world only through the nodes you connect. It also stops a
            runaway loop or allocation by itself instead of tying up your
            machine.
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            On by default, for both your own flows and flows from installed
            packages. Costs a one-off ~1.2&nbsp;MB download the first time a
            flow runs, and code nodes run a little slower. A code node that
            calls <code>fetch</code> or <code>setTimeout</code> directly gets a
            clear error here — use an HTTP node instead. Turn this off to go
            back to running your own flows directly in the browser (package
            flows then use the standard isolated worker).
          </p>
          {!supported && (
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
              This browser cannot run the Zipp engine, so flows keep running
              the standard way.
            </p>
          )}
        </div>
      </label>
    </div>
  );
}

// ===========================================================================
// SharingToggle — Defaults tab section that flips the backend on/off,
// shows the API URL, and offers a "Test connection" button.
// ===========================================================================
//
// State lives in `localStorage` via `sharingPrefs.ts`. The toggle is
// disabled when the build was made without a `VITE_API_BASE` (nothing
// to connect to), so this UI is automatically dormant in standalone
// deployments. The note below the toggle tells the user which mode
// they're in. The test button hits `GET /` on the API to verify the
// URL is reachable + the route handler is alive.
function SharingToggle() {
  const apiBase = backendBaseUrl();
  const buildHasBackend = apiBase !== '';
  const [enabled, setEnabled] = useState<boolean>(() => isSharingEnabled());
  const [testState, setTestState] = useState<'idle' | 'testing' | 'ok' | 'err'>('idle');
  const [testMsg, setTestMsg] = useState<string | null>(null);

  const toggle = useCallback(() => {
    if (!buildHasBackend) return;
    const next = !enabled;
    setSharingEnabled(next);
    setEnabled(next);
  }, [buildHasBackend, enabled]);

  const runTest = useCallback(async () => {
    if (!buildHasBackend) return;
    setTestState('testing');
    setTestMsg(null);
    // performance.now() rather than Date.now() — monotonic, immune to
    // wall-clock skew, and what `Resource Timing` measures anyway.
    const start = performance.now();
    try {
      const resp = await fetch(apiBase + '/', { method: 'GET' });
      const text = (await resp.text()).trim();
      const ms = Math.round(performance.now() - start);
      // Try to extract the API name for a friendly confirmation.
      let name: string | null = null;
      try {
        const parsed = JSON.parse(text) as { name?: string };
        if (typeof parsed?.name === 'string') name = parsed.name;
      } catch { /* not JSON; fall through */ }
      if (resp.ok && (name === 'oaiy-api' || /oaiy/i.test(text))) {
        setTestState('ok');
        setTestMsg(`HTTP ${resp.status} — connected to ${name ?? 'backend'} (${ms} ms)`);
      } else if (resp.ok) {
        setTestState('ok');
        setTestMsg(`HTTP ${resp.status} — reachable (${ms} ms), but response didn't look like oaiy-api`);
      } else {
        setTestState('err');
        setTestMsg(`HTTP ${resp.status} — server reachable but error response (${ms} ms)`);
      }
    } catch (e) {
      setTestState('err');
      const ms = Math.round(performance.now() - start);
      const msg = String((e as Error).message ?? e);
      const truncated = msg.length > 100 ? msg.slice(0, 97) + '…' : msg;
      setTestMsg(`${truncated} (failed after ${ms} ms)`);
    }
  }, [apiBase, buildHasBackend]);

  return (
    <div className="space-y-3 pt-4 border-t border-slate-200 dark:border-slate-700">
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 flex items-center gap-2">
        <svg className="w-4 h-4" style={{ color: 'rgb(var(--accent-primary))' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
        </svg>
        Sharing &amp; remote runs
      </h3>
      <label className={`flex items-start gap-3 ${buildHasBackend ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'}`}>
        <input
          type="checkbox"
          checked={enabled}
          disabled={!buildHasBackend}
          onChange={toggle}
          style={{ accentColor: 'rgb(var(--accent-primary))' }}
          className="mt-1 w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-offset-white dark:focus:ring-offset-slate-800"
        />
        <div className="flex-1">
          <div className="text-sm text-slate-700 dark:text-slate-200">
            Enable backend sharing
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            When on, you can push flows to the configured oaiy-api backend and
            get a shareable link. External AI clients can POST runs to the
            link; your browser picks them up and executes locally.
          </p>
          <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
            ⚠ Anyone with the edit link can trigger runs that consume your
            local compute + your registered services. Set a password on the
            Share dialog when you want the flow encrypted at rest.
          </p>
        </div>
      </label>

      {/* API base + test connection. Showing the URL plainly makes it
          unambiguous which deployment you're talking to. The test
          button does a real GET; any response (including 4xx) tells
          us the route handler is at least alive. */}
      <div className="ml-7 space-y-2">
        <label className="block text-xs text-slate-500 dark:text-slate-400">Backend URL</label>
        <div className="flex items-center gap-2">
          <input
            type="text"
            readOnly
            value={apiBase || '(not configured at build time)'}
            className="flex-1 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs font-mono text-slate-700 dark:text-slate-200"
            onFocus={(e) => e.currentTarget.select()}
          />
          <button
            type="button"
            onClick={runTest}
            disabled={!buildHasBackend || testState === 'testing'}
            className="px-3 py-1.5 text-xs rounded bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white disabled:opacity-50 disabled:cursor-not-allowed"
            title="GET / on the backend to verify it's reachable"
          >
            {testState === 'testing' ? 'Testing…' : 'Test connection'}
          </button>
        </div>
        {testMsg && (
          <p
            className={`text-xs ${
              testState === 'ok'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-red-600 dark:text-red-400'
            }`}
          >
            {testState === 'ok' ? '✓ ' : '✗ '}
            {testMsg}
          </p>
        )}
        <p className="text-xs text-slate-500 dark:text-slate-400">
          To point at a different backend, rebuild the UI with{' '}
          <code className="font-mono">VITE_API_BASE=https://your-host</code>.
          The URL is baked at build time so the AI clients you hand share
          links to always know where to send runs.
        </p>
      </div>
    </div>
  );
}
