/**
 * Security Tab Component
 *
 * Manages local network access whitelist for workflow security.
 * Extracted from SettingsPanel.tsx for maintainability.
 */

import { useState } from 'react';
import { Check, Plus, ShieldCheck, X } from 'lucide-react';
import type { ProjectSettings } from 'oaiy-core';
import { Card, EmptyState } from '../../chrome/SectionPage';

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

  const whitelist = settings.localNetworkWhitelist || [];

  return (
    <>
      {/* Global Override Switch */}
      <Card title="All local addresses">
        <label className="oaiy-check">
          <input
            type="checkbox"
            checked={settings.allowAllLocalNetwork || false}
            onChange={(e) => onUpdateSettings({ allowAllLocalNetwork: e.target.checked })}
          />
          <div className="flex flex-col gap-1">
            <strong>Allow every address on this network</strong>
            <p className="oaiy-help">
              Flows may then reach any local or private address without asking, skipping the list below.
            </p>
          </div>
        </label>
        <p className="oaiy-note warn m-0">
          <strong>Use with care.</strong> This turns off the check that stops a flow you did not write
          from reaching the services on your own machine and network.
        </p>
      </Card>

      {/* Whitelist Management */}
      <Card title="Allowed addresses" count={whitelist.length}>
        <p className="oaiy-card-text">
          Local services flows may reach without asking, such as Ollama, ComfyUI or LM Studio.
          Anything else on your network asks first. Write each as <code className="oaiy-code">hostname:port</code>.
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
            className="oaiy-input mono flex-1"
            placeholder="localhost:11434"
            aria-label="Address to allow"
          />
          <button
            onClick={handleAddEntry}
            disabled={!newWhitelistEntry.trim()}
            className="btn btn-primary"
          >
            <Plus size={14} /> Allow
          </button>
        </div>

        {/* Quick Add Presets */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="oaiy-label">Quick add</span>
          {WHITELIST_PRESETS.map(preset => {
            const isAdded = whitelist.includes(preset.value);
            return (
              <button
                key={preset.value}
                onClick={() => handleAddPreset(preset)}
                disabled={isAdded}
                // Allowed already: said in green, not greyed out as if broken.
                className={isAdded ? 'btn btn-sm done' : 'btn btn-sm'}
                title={isAdded ? `${preset.value} is allowed` : `Allow ${preset.value}`}
                style={isAdded ? { opacity: 1 } : undefined}
              >
                {isAdded && <Check size={12} />}
                {preset.label}
              </button>
            );
          })}
        </div>

        {/* Current Whitelist */}
        {whitelist.length > 0 ? (
          <ul className="oaiy-rows m-0 list-none rounded-[var(--r-ctl)] border border-edge-primary p-0">
            {whitelist.map((entry) => (
              <li key={entry} className="oaiy-row">
                <ShieldCheck size={14} className="shrink-0 text-signal-green" />
                <div className="oaiy-row-main">
                  <span className="oaiy-row-title mono">{entry}</span>
                </div>
                <button
                  onClick={() => handleRemoveEntry(entry)}
                  className="oaiy-icon-btn sm"
                  title={`Stop allowing ${entry}`}
                  aria-label={`Remove ${entry}`}
                >
                  <X size={13} />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState icon={<ShieldCheck size={22} />} title="No addresses allowed yet">
            A flow that reaches a local service asks for permission first.
          </EmptyState>
        )}
      </Card>

      {/* Log Privacy */}
      <Card title="Paths in logs">
        <p className="oaiy-card-text">
          How file paths appear in a flow's log. Pick a stricter one if you share or record your logs.
        </p>
        <label className="oaiy-field" style={{ maxWidth: 460 }}>
          <span>Show paths as</span>
          <select
            value={settings.logPathPolicy ?? 'tilde'}
            onChange={(e) => onUpdateSettings({ logPathPolicy: e.target.value as ProjectSettings['logPathPolicy'] })}
            className="oaiy-select"
          >
            <option value="full">Full — print paths verbatim (no privacy)</option>
            <option value="tilde">Tilde — rewrite $HOME as ~/… (recommended default)</option>
            <option value="basename">Basename — only the file name, no directories</option>
            <option value="none">None — replace every path with &lt;path&gt;</option>
          </select>
        </label>
      </Card>
    </>
  );
}
