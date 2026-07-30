/**
 * Security Tab Component
 *
 * Manages local network access whitelist for workflow security.
 * Extracted from SettingsPanel.tsx for maintainability.
 */

import { useState } from 'react';
import type { ProjectSettings } from 'oaiy-core';

interface SecurityTabProps {
  settings: ProjectSettings;
  onUpdateSettings: (updates: Partial<ProjectSettings>) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
}

const WHITELIST_PRESETS = [
  { label: 'Ollama', value: 'localhost:11434' },
  { label: 'LM Studio', value: 'localhost:1234' },
  { label: 'ComfyUI', value: 'localhost:8188' },
  { label: 'Stable Diffusion', value: 'localhost:7860' },
];

export default function SecurityTab({ settings, onUpdateSettings, onShowToast }: SecurityTabProps) {
  const [newWhitelistEntry, setNewWhitelistEntry] = useState('');

  const handleAddEntry = () => {
    if (newWhitelistEntry.trim()) {
      const entry = newWhitelistEntry.trim();
      const current = settings.localNetworkWhitelist || [];
      if (!current.includes(entry)) {
        onUpdateSettings({ localNetworkWhitelist: [...current, entry] });
        onShowToast?.(`Added ${entry} to whitelist`, 'success');
      }
      setNewWhitelistEntry('');
    }
  };

  const handleRemoveEntry = (entry: string) => {
    const current = settings.localNetworkWhitelist || [];
    onUpdateSettings({ localNetworkWhitelist: current.filter(e => e !== entry) });
    onShowToast?.(`Removed ${entry} from whitelist`, 'info');
  };

  const handleAddPreset = (preset: { label: string; value: string }) => {
    const current = settings.localNetworkWhitelist || [];
    if (!current.includes(preset.value)) {
      onUpdateSettings({ localNetworkWhitelist: [...current, preset.value] });
      onShowToast?.(`Added ${preset.label} (${preset.value}) to whitelist`, 'success');
    }
  };

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-slate-100/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
        <p className="text-slate-500 dark:text-slate-400 text-sm">
          Control which local network addresses can be accessed by your workflows. This protects against malicious workflows accessing your local services.
        </p>
      </div>

      {/* Global Override Switch */}
      <div className="bg-amber-100/50 dark:bg-amber-900/20 border border-amber-400/50 dark:border-amber-600/30 rounded-lg p-4">
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.allowAllLocalNetwork || false}
            onChange={(e) => onUpdateSettings({ allowAllLocalNetwork: e.target.checked })}
            className="mt-1 w-5 h-5 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-amber-500 focus:ring-amber-500 focus:ring-offset-white dark:focus:ring-offset-slate-800"
          />
          <div>
            <span className="text-amber-700 dark:text-amber-400 font-medium">Allow All Local Network Access</span>
            <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
              Bypass whitelist and allow all local/private network requests. <strong className="text-amber-600 dark:text-amber-500">Use with caution!</strong> This disables security checks for all workflows.
            </p>
          </div>
        </label>
      </div>

      {/* Whitelist Management */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 flex items-center gap-2">
          <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          Local Network Whitelist
        </h3>

        <p className="text-slate-500 text-sm">
          Add addresses for local services like Ollama, ComfyUI, or LM Studio. Format: <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">hostname:port</code>
        </p>

        {/* Add New Entry */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newWhitelistEntry}
            onChange={(e) => setNewWhitelistEntry(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && newWhitelistEntry.trim()) {
                handleAddEntry();
              }
            }}
            className="flex-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-3 py-2 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-green-500 font-mono"
            placeholder="localhost:11434"
          />
          <button
            onClick={handleAddEntry}
            disabled={!newWhitelistEntry.trim()}
            className="btn btn-md bg-green-600 hover:bg-green-500 text-white"
          >
            Add
          </button>
        </div>

        {/* Quick Add Presets */}
        <div className="flex flex-wrap gap-2">
          <span className="text-slate-500 text-xs">Quick add:</span>
          {WHITELIST_PRESETS.map(preset => {
            const isAdded = (settings.localNetworkWhitelist || []).includes(preset.value);
            return (
              <button
                key={preset.value}
                onClick={() => handleAddPreset(preset)}
                disabled={isAdded}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  isAdded
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 cursor-default'
                    : 'bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-400'
                }`}
              >
                {preset.label} {isAdded && '\u2713'}
              </button>
            );
          })}
        </div>

        {/* Current Whitelist */}
        {(settings.localNetworkWhitelist || []).length > 0 ? (
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Allowed Addresses</h4>
            <div className="space-y-1">
              {(settings.localNetworkWhitelist || []).map((entry) => (
                <div key={entry} className="flex items-center justify-between bg-slate-100/50 dark:bg-slate-700/30 rounded px-3 py-2">
                  <code className="text-green-600 dark:text-green-400 text-sm font-mono">{entry}</code>
                  <button
                    onClick={() => handleRemoveEntry(entry)}
                    className="p-1 hover:bg-slate-200 dark:hover:bg-slate-600 rounded text-slate-500 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                    title="Remove"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-slate-500 bg-slate-100/50 dark:bg-slate-800/30 rounded-lg">
            <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            <p className="text-sm">No addresses whitelisted</p>
            <p className="text-xs mt-1">Workflows will prompt for permission when accessing local services</p>
          </div>
        )}
      </div>

      {/* Log Privacy */}
      <div className="space-y-3 pt-4 border-t border-slate-200 dark:border-slate-700">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 flex items-center gap-2">
          <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          Log path privacy
        </h3>
        <p className="text-slate-500 text-sm">
          How filesystem paths appear in workflow logs. Choose a stricter setting if you plan to share or screen-record logs.
        </p>
        <select
          value={settings.logPathPolicy ?? 'tilde'}
          onChange={(e) => onUpdateSettings({ logPathPolicy: e.target.value as ProjectSettings['logPathPolicy'] })}
          className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-3 py-2 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-blue-500"
        >
          <option value="full">Full — print paths verbatim (no privacy)</option>
          <option value="tilde">Tilde — rewrite $HOME as ~/… (recommended default)</option>
          <option value="basename">Basename — only the file name, no directories</option>
          <option value="none">None — replace every path with &lt;path&gt;</option>
        </select>
      </div>
    </div>
  );
}
