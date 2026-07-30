import {
  Activity,
  Check,
  ChevronRight,
  Cloud,
  Copy,
  Database,
  LockKeyhole,
  Moon,
  Plus,
  Puzzle,
  Server,
  Settings2,
  Share2,
  Sun,
  Workflow,
  type LucideIcon,
} from 'lucide-react';
import type { ReactNode } from 'react';

/**
 * OAIY shell chrome — the sidebar, topbar and endpoint dock from the OAIY
 * design system.
 *
 * These three pieces are purely presentational: every label, flag and callback
 * arrives as a prop from OAIYApp, which still owns all the state. Keeping them
 * here (rather than inline in OAIYApp's 1200-line render) is what makes the
 * brand chrome readable as one thing.
 *
 * Layout comes from the `.app-shell` / `.oaiy-*` block at the bottom of
 * index.css — a 222px sidebar plus a 72px topbar / canvas / 64px dock grid.
 */

export type ShellView = 'builder' | 'data';

export interface ShellNavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  active: boolean;
  onClick: () => void;
}

/* ------------------------------------------------------------------ sidebar */

export function ShellSidebar({
  view,
  onSelectView,
  onNewFlow,
  onOpenQueue,
  onOpenPlugins,
  onOpenServices,
  onOpenSettings,
  settingsActive,
  companionOnline,
  companionDetail,
}: {
  view: ShellView;
  onSelectView: (v: ShellView) => void;
  onNewFlow: () => void;
  onOpenQueue: () => void;
  onOpenPlugins: () => void;
  onOpenServices: () => void;
  onOpenSettings: () => void;
  settingsActive: boolean;
  companionOnline: boolean;
  companionDetail: string;
}) {
  const nav: ShellNavItem[] = [
    {
      id: 'builder',
      label: 'Workflows',
      icon: Workflow,
      active: view === 'builder',
      onClick: () => onSelectView('builder'),
    },
    { id: 'providers', label: 'Providers', icon: Cloud, active: false, onClick: onOpenServices },
    {
      id: 'runs',
      label: 'Runs',
      icon: Activity,
      active: false,
      onClick: onOpenQueue,
    },
    { id: 'plugins', label: 'Plugins', icon: Puzzle, active: false, onClick: onOpenPlugins },
    {
      id: 'data',
      label: 'Data',
      icon: Database,
      active: view === 'data',
      onClick: () => onSelectView('data'),
    },
  ];

  return (
    <aside className="oaiy-sidebar">
      <div className="oaiy-brand">
        <strong>OAIY</strong>
        <span>Orchestrate AI Yourself</span>
        <small>Connect. Draw. Expose.</small>
      </div>

      <button className="oaiy-new" type="button" onClick={onNewFlow}>
        <Plus size={17} />
        <span>New flow</span>
      </button>

      <nav className="oaiy-nav" aria-label="Primary">
        {nav.map((item) => {
          const Icon = item.icon;
          return (
            <button
              type="button"
              key={item.id}
              className={item.active ? 'active' : ''}
              aria-current={item.active ? 'page' : undefined}
              onClick={item.onClick}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="oaiy-fill" />

      <div
        className={companionOnline ? 'oaiy-engine' : 'oaiy-engine off'}
        title={companionDetail}
      >
        <Server size={17} />
        <span>
          <strong>Local engine</strong>
          <small>{companionDetail}</small>
        </span>
        <i />
      </div>

      <button
        className={settingsActive ? 'oaiy-settings-btn active' : 'oaiy-settings-btn'}
        type="button"
        onClick={onOpenSettings}
      >
        <Settings2 size={18} />
        <span>Settings</span>
      </button>

      <div className="oaiy-trust">
        <LockKeyhole size={13} />
        <span>Keys stay on this device</span>
      </div>
    </aside>
  );
}

/* -------------------------------------------------------------------- topbar */

export function ShellTopbar({
  crumb,
  children,
  chips,
  savedLabel,
  theme,
  onSetTheme,
  actions,
}: {
  /** Uppercase breadcrumb line, e.g. `Workflows › Image Review Loop`. */
  crumb: string;
  /** The title row — an editable name input, or a plain heading. */
  children: ReactNode;
  chips?: ReactNode;
  savedLabel?: string;
  theme: 'light' | 'dark';
  onSetTheme: (t: 'light' | 'dark') => void;
  actions?: ReactNode;
}) {
  return (
    <header className="oaiy-topbar">
      <div className="oaiy-title">
        <span>
          OAIY <ChevronRight size={12} /> {crumb}
        </span>
        <div>
          {children}
          {chips}
          {savedLabel && (
            <small className="oaiy-saved">
              <Check size={13} />
              {savedLabel}
            </small>
          )}
        </div>
      </div>
      <div className="oaiy-actions">
        <div className="oaiy-toggle">
          <button
            type="button"
            className={`oaiy-icon-btn ${theme === 'light' ? 'active' : ''}`}
            aria-label="Paper Circuit (light) theme"
            title="Paper Circuit"
            aria-pressed={theme === 'light'}
            onClick={() => onSetTheme('light')}
          >
            <Sun size={16} />
          </button>
          <button
            type="button"
            className={`oaiy-icon-btn ${theme === 'dark' ? 'active' : ''}`}
            aria-label="Prism Lab (dark) theme"
            title="Prism Lab"
            aria-pressed={theme === 'dark'}
            onClick={() => onSetTheme('dark')}
          >
            <Moon size={16} />
          </button>
        </div>
        {actions}
      </div>
    </header>
  );
}

/** A round icon action for the topbar's right-hand cluster. */
export function ShellIconAction({
  label,
  on,
  onClick,
  children,
  title,
}: {
  label: string;
  on?: boolean;
  onClick: () => void;
  children: ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      className={`oaiy-icon-btn ${on ? 'on' : ''}`}
      aria-label={label}
      title={title ?? label}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/* ---------------------------------------------------------------------- dock */

export function ShellDock({
  companionOnline,
  endpointLabel,
  endpointUrl,
  onCopyEndpoint,
  shared,
  onManage,
  manageLabel,
}: {
  companionOnline: boolean;
  endpointLabel: string;
  endpointUrl: string;
  onCopyEndpoint: () => void;
  shared: boolean;
  onManage: () => void;
  manageLabel: string;
}) {
  return (
    <footer className={companionOnline ? 'oaiy-dock' : 'oaiy-dock off'}>
      <button type="button" onClick={onManage}>
        <i />
        <span>
          <strong>{endpointLabel}</strong>
          <small>{companionOnline ? 'companion connected' : 'browser only'}</small>
        </span>
      </button>
      <div className="oaiy-dock-url">
        <code>{endpointUrl}</code>
        <button
          type="button"
          className="oaiy-icon-btn"
          aria-label="Copy endpoint URL"
          title="Copy endpoint"
          onClick={onCopyEndpoint}
        >
          <Copy size={14} />
        </button>
      </div>
      <em className={shared ? 'ok' : ''}>
        {shared ? <Check size={12} /> : <Share2 size={12} />}
        {shared ? 'HTTP drivable' : 'not shared'}
      </em>
      <em>
        <LockKeyhole size={12} /> Runs on this device
      </em>
      <button type="button" onClick={onManage}>
        {manageLabel} <ChevronRight size={13} />
      </button>
    </footer>
  );
}
