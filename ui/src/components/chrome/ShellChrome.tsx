import {
  Menu,
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
  providersOpen = false,
  runsOpen = false,
  pluginsOpen = false,
  navOpen = false,
  onCloseNav,
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
  /** Which overlay is on screen. The rail is the only place that shows where
   *  you are, and these three entries open a panel rather than switch view --
   *  so without them the rail says "Workflows" while you are looking at
   *  Providers. */
  providersOpen?: boolean;
  runsOpen?: boolean;
  pluginsOpen?: boolean;
  /** Below md the rail is off-canvas; this slides it in. Ignored above md,
   *  where the rail is part of the grid and always present. */
  navOpen?: boolean;
  onCloseNav?: () => void;
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
      active: view === 'builder' && !providersOpen && !runsOpen && !pluginsOpen,
      onClick: () => onSelectView('builder'),
    },
    { id: 'providers', label: 'Providers', icon: Cloud, active: providersOpen, onClick: onOpenServices },
    {
      id: 'runs',
      label: 'Runs',
      icon: Activity,
      active: runsOpen,
      onClick: onOpenQueue,
    },
    { id: 'plugins', label: 'Plugins', icon: Puzzle, active: pluginsOpen, onClick: onOpenPlugins },
    {
      id: 'data',
      label: 'Data',
      icon: Database,
      active: view === 'data' && !providersOpen && !runsOpen && !pluginsOpen,
      onClick: () => onSelectView('data'),
    },
  ];

  // Every control in the rail dismisses the drawer as well as doing its job.
  // Wrapped once here rather than at each call site: half of these open a panel
  // rather than switch view, and those were left sitting BEHIND the open
  // drawer. Centralising it also means a nav entry added later cannot forget.
  const andClose = (fn?: () => void) => () => {
    fn?.();
    onCloseNav?.();
  };

  return (
    <>
    {/* Scrim: only rendered while the drawer is open, and only reachable below
        md because the drawer cannot open above it. */}
    {navOpen && (
      <div className="oaiy-nav-scrim" onClick={onCloseNav} aria-hidden="true" />
    )}
    <aside className={`oaiy-sidebar${navOpen ? ' is-open' : ''}`}>
      <div className="oaiy-brand">
        <strong>OAIY</strong>
        <span>Orchestrate AI Yourself</span>
        <small>Connect. Draw. Expose.</small>
      </div>

      <button className="oaiy-new" type="button" onClick={andClose(onNewFlow)}>
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
              onClick={andClose(item.onClick)}
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
        onClick={andClose(onOpenSettings)}
      >
        <Settings2 size={18} />
        <span>Settings</span>
      </button>

      <div className="oaiy-trust">
        <LockKeyhole size={13} />
        <span>Keys stay on this device</span>
      </div>
    </aside>
    </>
  );
}

/* -------------------------------------------------------------------- topbar */

export function ShellTopbar({
  onOpenNav,
  crumb,
  children,
  chips,
  savedLabel,
  theme,
  onSetTheme,
  actions,
}: {
  /** Opens the off-canvas nav. Only rendered below md, where the rail is not
   *  in the grid — above md the rail is always visible and needs no opener. */
  onOpenNav?: () => void;
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
      {onOpenNav && (
        <button
          type="button"
          className="oaiy-nav-open oaiy-icon-btn"
          onClick={onOpenNav}
          aria-label="Open navigation"
          title="Menu"
        >
          <Menu size={18} />
        </button>
      )}
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
        {/* One button, not two. A segmented pair spends double the width to
            say what one icon says, and in a topbar that already overflows on a
            phone that is width taken from controls that have nowhere else to
            go. It shows the theme you would switch TO, which is the thing you
            are choosing -- so the label is an action ("Switch to X"), not a
            state, and aria-pressed is gone with it: this is a button that does
            something, not a toggle reporting whether it is on. */}
        <button
          type="button"
          className="oaiy-icon-btn"
          aria-label={theme === 'dark' ? 'Switch to the Paper Circuit light theme' : 'Switch to the Prism Lab dark theme'}
          title={theme === 'dark' ? 'Switch to Paper Circuit' : 'Switch to Prism Lab'}
          onClick={() => onSetTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
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
