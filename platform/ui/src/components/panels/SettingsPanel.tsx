/**
 * Settings — a page in the editor's main area with pages of its own, listed
 * down its left side like the dashboard's sidebar:
 *
 *   Services    the models and HTTP services nodes call, and where OAIY's engine is
 *   API keys    keys that services and nodes read by name
 *   Constants   named values nodes read by name
 *   Security    which local addresses flows may reach
 *   General     back up or restore settings, sharing, the engine flows run on
 *   Appearance  theme and accent (in OAIY's window, OAIY's)
 *
 * "Providers", which the editor's top row used to open, was this Services page
 * under a second name; it is only Services now.
 */
import { useState, useCallback } from 'react';
import { Download, KeyRound, Lock, Palette, Server, ShieldCheck, SlidersHorizontal, Tag, Trash2, Upload } from 'lucide-react';
import type { ProjectConstant, ProjectSettings } from 'oaiy-core';
import { useServices } from '../../hooks/useServices';
import { AppearanceTab, SecurityTab, ServicesTab } from './settings';
import { uiLogger as logger } from '../../utils/logger';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';
import SectionPage, { Card, SubNav, type SubNavItem } from '../chrome/SectionPage';
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

export type SettingsPage =
  | 'services'
  | 'apikeys'
  | 'constants'
  | 'security'
  | 'general'
  | 'appearance';
// Desktop-only tabs (plugins / api / models) were removed in the web build:
//  - plugins = App Data Folder, has no meaning without a real filesystem
//  - api = Tauri-hosted HTTP API server, no equivalent in the browser
//  - models = native model-download manager (plugin-oaiy-diffusion), web
//    talks to whatever local engine the user already runs — see the
//    Services page for HTTP-endpoint registration instead.

/** Each page: its name in the list, its icon, and the line under its title. */
const PAGES: Record<SettingsPage, { label: string; icon: SubNavItem['icon']; copy: string }> = {
  services: { label: 'Services', icon: Server, copy: "The models and HTTP services your nodes call, and where OAIY's engine is." },
  apikeys: { label: 'API keys', icon: KeyRound, copy: 'Keys your services and nodes read by name. They stay in this project, on this device.' },
  constants: { label: 'Constants', icon: Tag, copy: 'Named values (an endpoint, a model, anything) that nodes read by name, so one change reaches every flow.' },
  security: { label: 'Security', icon: ShieldCheck, copy: 'Which addresses on your own network flows may reach.' },
  general: { label: 'General', icon: SlidersHorizontal, copy: 'Back up or restore these settings, share flows, and choose the engine flows run on.' },
  appearance: { label: 'Appearance', icon: Palette, copy: 'Theme, accent and background.' },
};
const ORDER: SettingsPage[] = ['services', 'apikeys', 'constants', 'security', 'general', 'appearance'];

interface SettingsPanelProps {
  /** The page showing. */
  page: SettingsPage;
  onPageChange: (page: SettingsPage) => void;
  settings: ProjectSettings;
  constants: ProjectConstant[];
  onUpdateSettings: (updates: Partial<ProjectSettings>) => void;
  onUpdateConstant: (id: string, updates: Partial<ProjectConstant>) => void;
  onCreateConstant: (constant: Omit<ProjectConstant, 'id'>) => void;
  onDeleteConstant?: (id: string) => void;
  onExportSettings?: () => void;
  onImportSettings?: (settings: { settings: ProjectSettings; constants: ProjectConstant[] }) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
}

export default function SettingsPanel({
  page,
  onPageChange,
  settings,
  constants,
  onUpdateSettings,
  onUpdateConstant,
  onCreateConstant,
  onDeleteConstant,
  onExportSettings,
  onImportSettings,
  onShowToast,
}: SettingsPanelProps) {
  const confirm = useConfirmDialog();
  const [newConstantName, setNewConstantName] = useState('');
  const [newConstantKey, setNewConstantKey] = useState('');
  const [newConstantValue, setNewConstantValue] = useState('');
  // Separate add-form state for API Keys so the constants form isn't
  // surprised by api_key entries appearing in its list mid-edit.
  const [newApiKeyName, setNewApiKeyName] = useState('');
  const [newApiKeyKey, setNewApiKeyKey] = useState('');
  const [newApiKeyValue, setNewApiKeyValue] = useState('');

  // How many of the desktop's services run: a count beside Services.
  const { runningCount } = useServices({ autoRefresh: page === 'services', refreshInterval: 3000 });

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

  const apiKeyConstants = constants.filter(c => c.category === 'api_key');
  const customConstants = constants.filter(c => c.category !== 'api_key');

  const nav = (
    <SubNav
      label="Settings pages"
      active={page}
      onSelect={(id) => onPageChange(id as SettingsPage)}
      items={ORDER.map((id) => ({
        id,
        label: PAGES[id].label,
        icon: PAGES[id].icon,
        badge: id === 'services' && runningCount > 0 ? runningCount : undefined,
        title: id === 'services' && runningCount > 0 ? `Services (${runningCount} running)` : PAGES[id].label,
      }))}
    />
  );

  const actions =
    page === 'general' ? (
      <>
        <button type="button" onClick={handleImportSettings} className="btn">
          <Upload size={14} /> Import settings
        </button>
        <button type="button" onClick={handleExportSettings} className="btn">
          <Download size={14} /> Export settings
        </button>
      </>
    ) : undefined;

  return (
    <SectionPage
      kicker="Settings"
      title={PAGES[page].label}
      description={PAGES[page].copy}
      actions={actions}
      nav={nav}
      testId={`settings-${page}`}
    >
      {page === 'services' && (
        <ServicesTab
          isOpen
          settings={settings}
          constants={constants}
          onUpdateSettings={onUpdateSettings}
        />
      )}

      {page === 'apikeys' && (
        <>
          <Card title="Add a key">
            <p className="oaiy-card-text">
              Any node or service with an API key field takes a key by its name (for example <code className="oaiy-code">OPENAI_API_KEY</code>).
              It is sent only to the endpoint that node calls.
            </p>
            <div className="oaiy-form-grid">
              <label className="oaiy-field">
                <span>Name</span>
                <input type="text" value={newApiKeyName} onChange={(e) => setNewApiKeyName(e.target.value)} className="oaiy-input" placeholder="OpenAI" />
              </label>
              <label className="oaiy-field">
                <span>Key name</span>
                <input type="text" value={newApiKeyKey} onChange={(e) => setNewApiKeyKey(e.target.value)} className="oaiy-input mono" placeholder="MY_API_KEY" />
              </label>
              <label className="oaiy-field">
                <span>Value (optional)</span>
                <input type="password" value={newApiKeyValue} onChange={(e) => setNewApiKeyValue(e.target.value)} className="oaiy-input mono" placeholder="sk-…" />
              </label>
            </div>
            <div>
              <button onClick={handleAddApiKey} disabled={!newApiKeyName || !newApiKeyKey} className="btn btn-primary">
                Add the key
              </button>
            </div>
          </Card>

          <Card title="Your keys" count={apiKeyConstants.length} flush={apiKeyConstants.length > 0}>
            {apiKeyConstants.length === 0 ? (
              <p className="oaiy-card-text">None yet. Nodes that name a key you add here pick it up by themselves.</p>
            ) : (
              <ul className="oaiy-rows m-0 list-none p-0">
                {apiKeyConstants.map((constant) => (
                  <li key={constant.id} className="oaiy-row flex-wrap">
                    <Lock size={14} className="shrink-0 text-content-faint" />
                    <div className="oaiy-row-main wide">
                      <span className="oaiy-row-title">{constant.name}</span>
                      <span className="oaiy-row-meta font-mono">{constant.key}</span>
                    </div>
                    <input
                      type="password"
                      value={constant.value}
                      onChange={(e) => onUpdateConstant(constant.id, { value: e.target.value })}
                      className="oaiy-input mono inline"
                      placeholder={`Enter ${constant.key}…`}
                      aria-label={`${constant.name} value`}
                    />
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
                        className="oaiy-icon-btn"
                        title={`Delete ${constant.name}`}
                        aria-label={`Delete ${constant.name}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}

      {page === 'constants' && (
        <>
          <Card title="Add a constant">
            <div className="oaiy-form-grid">
              <label className="oaiy-field">
                <span>Name</span>
                <input type="text" value={newConstantName} onChange={(e) => setNewConstantName(e.target.value)} className="oaiy-input" placeholder="Image server" />
              </label>
              <label className="oaiy-field">
                <span>Key</span>
                <input type="text" value={newConstantKey} onChange={(e) => setNewConstantKey(e.target.value)} className="oaiy-input mono" placeholder="CONSTANT_KEY" />
              </label>
              <label className="oaiy-field">
                <span>Value</span>
                <input type="text" value={newConstantValue} onChange={(e) => setNewConstantValue(e.target.value)} className="oaiy-input" placeholder="http://127.0.0.1:8188" />
              </label>
            </div>
            <div>
              <button onClick={handleAddConstant} disabled={!newConstantName || !newConstantKey} className="btn btn-primary">
                Add the constant
              </button>
            </div>
          </Card>

          <Card title="Your constants" count={customConstants.length} flush={customConstants.length > 0}>
            {customConstants.length === 0 ? (
              <p className="oaiy-card-text">None yet.</p>
            ) : (
              <ul className="oaiy-rows m-0 list-none p-0">
                {customConstants.map((constant) => (
                  <li key={constant.id} className="oaiy-row flex-wrap">
                    <div className="oaiy-row-main wide">
                      <span className="oaiy-row-title">{constant.name}</span>
                      <span className="oaiy-row-meta font-mono">{constant.key}</span>
                    </div>
                    <input
                      type="text"
                      value={constant.value}
                      onChange={(e) => onUpdateConstant(constant.id, { value: e.target.value })}
                      className="oaiy-input inline"
                      placeholder="Value"
                      aria-label={`${constant.name} value`}
                    />
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}

      {page === 'security' && (
        <SecurityTab
          settings={settings}
          onUpdateSettings={onUpdateSettings}
          onShowToast={onShowToast}
        />
      )}

      {page === 'general' && (
        <>
          <Card title="Your settings">
            <p className="oaiy-card-text">
              Export these settings and your constants to a file (secret values are left out), or import a file exported before.
              Services are under <strong>Services</strong>; a node's own service is picked in its properties.
            </p>
          </Card>
          <SharingToggle />
          <ZippSandboxToggle />
        </>
      )}

      {page === 'appearance' && <AppearanceTab />}
    </SectionPage>
  );
}

// ===========================================================================
// ZippSandboxToggle — General page card that picks the engine flows run on.
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
    <Card title="Flow sandbox">
      <label className={`oaiy-check${supported ? '' : ' disabled'}`}>
        <input type="checkbox" checked={enabled} disabled={!supported} onChange={toggle} />
        <div className="flex flex-col gap-1">
          <strong>Run flows on the Zipp engine</strong>
          <p className="oaiy-help">
            Zipp is a separate JavaScript engine that runs inside WebAssembly
            and never had the browser&apos;s risky features: no network, no
            storage, no way to start more workers. Flow code reaches the
            outside world only through the nodes you connect. It also stops a
            runaway loop or allocation by itself instead of tying up your
            machine.
          </p>
          <p className="oaiy-help faint">
            On by default, for both your own flows and flows from installed
            packages. Costs a one-off ~1.2&nbsp;MB download the first time a
            flow runs, and code nodes run a little slower. A code node that
            calls <code className="oaiy-code">fetch</code> or <code className="oaiy-code">setTimeout</code> directly gets a
            clear error here — use an HTTP node instead. Turn this off to go
            back to running your own flows directly in the browser (package
            flows then use the standard isolated worker).
          </p>
          {!supported && (
            <p className="oaiy-warn-text">
              This browser cannot run the Zipp engine, so flows keep running
              the standard way.
            </p>
          )}
        </div>
      </label>
    </Card>
  );
}

// ===========================================================================
// SharingToggle — General page card that flips the backend on/off, shows the
// API URL, and offers a "Test connection" button.
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
    <Card title="Sharing and remote runs">
      <label className={`oaiy-check${buildHasBackend ? '' : ' disabled'}`}>
        <input type="checkbox" checked={enabled} disabled={!buildHasBackend} onChange={toggle} />
        <div className="flex flex-col gap-1">
          <strong>Enable backend sharing</strong>
          <p className="oaiy-help">
            When on, you can push flows to the configured oaiy-api backend and
            get a shareable link. External AI clients can POST runs to the
            link; your browser picks them up and executes locally.
          </p>
          <p className="oaiy-warn-text">
            Anyone with the edit link can trigger runs that consume your
            local compute and your registered services. Set a password in the
            Share dialog when you want the flow encrypted at rest.
          </p>
        </div>
      </label>

      {/* API base + test connection. Showing the URL plainly makes it
          unambiguous which deployment you're talking to. The test
          button does a real GET; any response (including 4xx) tells
          us the route handler is at least alive. */}
      <label className="oaiy-field">
        <span>Backend URL</span>
        <div className="flex items-center gap-2">
          <input
            type="text"
            readOnly
            value={apiBase || '(not configured at build time)'}
            className="oaiy-input mono"
            onFocus={(e) => e.currentTarget.select()}
          />
          <button
            type="button"
            onClick={runTest}
            disabled={!buildHasBackend || testState === 'testing'}
            className="btn"
            title="GET / on the backend to verify it's reachable"
          >
            {testState === 'testing' ? 'Testing…' : 'Test connection'}
          </button>
        </div>
      </label>
      {testMsg && (
        <p className={testState === 'ok' ? 'oaiy-ok-text' : 'oaiy-error-text'}>
          {testMsg}
        </p>
      )}
      <p className="oaiy-help faint">
        To point at a different backend, rebuild the UI with{' '}
        <code className="oaiy-code">VITE_API_BASE=https://your-host</code>.
        The URL is baked at build time so the AI clients you hand share
        links to always know where to send runs.
      </p>
    </Card>
  );
}
