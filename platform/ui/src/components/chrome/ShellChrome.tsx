import {
  Menu,
  X,
  Check,
  ChevronRight,
  Copy,
  Database,
  Download,
  ListChecks,
  LockKeyhole,
  Moon,
  Package,
  Plus,
  Server,
  Settings2,
  Share2,
  Sun,
  Workflow,
  type LucideIcon,
} from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';

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

/**
 * The editor's sections. Each is a view in its main area, never a popup:
 *
 *   Workflows  the flows rail and the canvas
 *   Data       what a flow's Database nodes stored
 *   Queue      flows running in this editor, and what they returned this session
 *   Packages   .oaiy flow packages: their sources, and loading one
 *   Settings   services, API keys, constants, security, general, appearance
 *
 * In OAIY's window the dashboard's own "Run history" tab (beside "Editor")
 * lists every run on the machine; Queue is only this editor's.
 */
export type EditorSection = 'workflows' | 'data' | 'queue' | 'packages' | 'settings';

export interface ShellNavItem {
  id: EditorSection;
  label: string;
  icon: LucideIcon;
  /** Shown in the tooltip: what the section is. */
  hint: string;
  active: boolean;
  onClick: () => void;
  /** A count beside the label (the queue's running flows). */
  badge?: ReactNode;
}

const SECTIONS: Array<{ id: EditorSection; label: string; icon: LucideIcon; hint: string }> = [
  { id: 'workflows', label: 'Workflows', icon: Workflow, hint: 'Your flows, on the canvas' },
  { id: 'data', label: 'Data', icon: Database, hint: "What the open flow's Database nodes stored" },
  { id: 'queue', label: 'Queue', icon: ListChecks, hint: 'Flows running in this editor, and their results this session' },
  { id: 'packages', label: 'Packages', icon: Package, hint: 'Flow packages (.oaiy): flows, macros and nodes to load' },
];
export const SETTINGS_ITEM = { id: 'settings' as const, label: 'Settings', icon: Settings2, hint: 'Services, API keys, constants, security' };

/** The sections, and which is showing: the same for the web rail and for the tabs in OAIY's window. */
export function shellNavItems(section: EditorSection, onSelect: (s: EditorSection) => void, badges: Partial<Record<EditorSection, ReactNode>> = {}): ShellNavItem[] {
  return SECTIONS.map((s) => ({ ...s, active: section === s.id, onClick: () => onSelect(s.id), badge: badges[s.id] }));
}

/* --------------------------------------------------------------- sections */

/**
 * In OAIY's window the editor has no rail of its own (OAIY's sidebar is beside
 * it): its sections are the dashboard's segmented tabs at the top of the page,
 * with New flow first and Settings last.
 */
export function ShellSections({
  items,
  onNewFlow,
  onOpenSettings,
  settingsActive,
}: {
  items: ShellNavItem[];
  onNewFlow: () => void;
  onOpenSettings: () => void;
  settingsActive: boolean;
}) {
  const all = [...items, { ...SETTINGS_ITEM, active: settingsActive, onClick: onOpenSettings, badge: undefined }];
  return (
    <nav className="oaiy-sections" aria-label="Flow editor">
      <button type="button" className="oaiy-sections-new" onClick={onNewFlow} title="Make a new flow">
        <Plus size={14} />
        <span>New flow</span>
      </button>
      <div className="oaiy-sections-tabs" role="tablist" aria-label="Sections">
        {all.map((item) => {
          const Icon = item.icon;
          return (
            <button
              type="button"
              role="tab"
              key={item.id}
              className={item.active ? 'active' : ''}
              aria-selected={item.active}
              aria-label={item.label}
              title={`${item.label}: ${item.hint}`}
              onClick={item.onClick}
            >
              <Icon size={14} />
              <span>{item.label}</span>
              {item.badge}
            </button>
          );
        })}
      </div>
    </nav>
  );
}

/* ------------------------------------------------------------------ sidebar */

export function ShellSidebar({
  navOpen = false,
  isPhone = false,
  onCloseNav,
  items,
  onNewFlow,
  onOpenSettings,
  settingsActive,
  companionOnline,
  companionDetail,
  onInstall,
}: {
  /** Below md the rail is off-canvas; this slides it in. Ignored above md,
   *  where the rail is part of the grid and always present. */
  navOpen?: boolean;
  isPhone?: boolean;
  onCloseNav?: () => void;
  /** The sections (shellNavItems), the one showing marked. */
  items: ShellNavItem[];
  onNewFlow: () => void;
  onOpenSettings: () => void;
  settingsActive: boolean;
  companionOnline: boolean;
  companionDetail: string;
  /** Install the editor as an app. Given only while the browser is offering to: no button otherwise. */
  onInstall?: () => void;
}) {
  const sidebarRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  useFocusTrap(sidebarRef, isPhone && navOpen, closeRef);
  useEffect(() => {
    if (!isPhone || !navOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        onCloseNav?.();
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [isPhone, navOpen, onCloseNav]);
  const nav = items;

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
    <aside id="oaiy-navigation" ref={sidebarRef} className={`oaiy-sidebar${navOpen ? ' is-open' : ''}`} inert={isPhone && !navOpen} aria-hidden={isPhone && !navOpen ? true : undefined} role={isPhone && navOpen ? 'dialog' : undefined} aria-modal={isPhone && navOpen ? true : undefined} aria-label="Application navigation">
      {isPhone && <button ref={closeRef} type="button" className="oaiy-nav-close oaiy-icon-btn" onClick={onCloseNav} aria-label="Close navigation"><X size={20} /></button>}
      <div className="oaiy-brand">
        <strong>OAIY</strong>
        <span>Orchestrate AI Yourself</span>
        <small>Connect. Draw. Expose.</small>
      </div>

      <button className="oaiy-new" type="button" aria-label="New flow" onClick={andClose(onNewFlow)}>
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
              aria-label={item.label}
              title={`${item.label}: ${item.hint}`}
              onClick={andClose(item.onClick)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {item.badge}
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

      {onInstall && (
        <button
          className="oaiy-settings-btn oaiy-install-btn"
          type="button"
          aria-label="Install app"
          title="Install OAIY as an app on this device"
          onClick={andClose(onInstall)}
        >
          <Download size={18} />
          <span>Install app</span>
        </button>
      )}

      <button
        className={settingsActive ? 'oaiy-settings-btn active' : 'oaiy-settings-btn'}
        type="button"
        aria-label="Settings"
        aria-current={settingsActive ? 'page' : undefined}
        title={`${SETTINGS_ITEM.label}: ${SETTINGS_ITEM.hint}`}
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
  navOpen = false,
  crumb,
  children,
  chips,
  savedLabel,
  theme,
  onSetTheme,
  actions,
  sections,
}: {
  /** Opens the off-canvas nav. Only rendered below md, where the rail is not
   *  in the grid — above md the rail is always visible and needs no opener. */
  onOpenNav?: () => void;
  navOpen?: boolean;
  /** Uppercase breadcrumb line, e.g. `Workflows › Image Review Loop`. */
  crumb: string;
  /** The title row — an editable name input, or a plain heading. */
  children: ReactNode;
  chips?: ReactNode;
  savedLabel?: string;
  theme: 'light' | 'dark';
  /** Absent in OAIY's window, where the theme is OAIY's (its sidebar switches it). */
  onSetTheme?: (t: 'light' | 'dark') => void;
  actions?: ReactNode;
  /** In OAIY's window: the sections' tabs, in the breadcrumb's place (OAIY's own header names the page). */
  sections?: ReactNode;
}) {
  return (
    <header className={sections ? 'oaiy-topbar has-sections' : 'oaiy-topbar'}>
      {onOpenNav && (
        <button
          type="button"
          className="oaiy-nav-open oaiy-icon-btn"
          onClick={onOpenNav}
          aria-label="Open navigation"
          aria-expanded={navOpen}
          aria-controls="oaiy-navigation"
          title="Menu"
        >
          <Menu size={18} />
        </button>
      )}
      <div className="oaiy-title">
        {sections ?? (
          <span>
            OAIY <ChevronRight size={12} /> {crumb}
          </span>
        )}
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
        {onSetTheme && (
          <button
            type="button"
            className="oaiy-icon-btn"
            aria-label={theme === 'dark' ? 'Switch to the Paper Circuit light theme' : 'Switch to the Prism Lab dark theme'}
            title={theme === 'dark' ? 'Switch to Paper Circuit' : 'Switch to Prism Lab'}
            onClick={() => onSetTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        )}
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
