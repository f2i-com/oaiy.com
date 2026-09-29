import { Bot, CalendarDays, Mail, MessageSquare, Phone, Puzzle, type LucideIcon } from 'lucide-react';
import type { ModulesSnapshot, PageContribution, SectionContribution } from './api';

/**
 * The sidebar's sections and each one's tabs: the dashboard's own, with what
 * the plugins add (`/api/modules` `contributions.sections`).
 *
 * - A contribution with a built-in section's id (`connections`) adds its pages
 *   as tabs AFTER that section's own. Built-in sections are never changed,
 *   replaced or reordered.
 * - A plugin's own section (`plugin-section:<pluginId>:<id>`) is a sidebar
 *   entry whose tabs are its pages.
 * - A page with no section is a sidebar entry of its own (its id is its view).
 * - A plugin's entry goes at the end of its group (Work unless it says Home or
 *   Setup), after the built-ins, so a plugin extends the app without
 *   displacing anything.
 * - A built-in section that belongs to a module (the Calendar; Contacts, the
 *   phone's) shows only while the module is on; and while the plugin providing
 *   the module has a section of its own, it goes under that section as a
 *   sub-menu (`arrangeSections`): the Calendar and Contacts under the AI
 *   Receptionist.
 *
 * Pure: App.tsx renders what this works out.
 */

export type Group = 'Home' | 'Work' | 'Setup';

/** A dashboard section as App.tsx declares it. */
export interface BuiltinSection {
  id: string;
  label: string;
  icon: LucideIcon;
  group: Group;
  /** Its pages (view ids), the first the one it opens on. */
  tabs: string[];
}

/** One tab: a built-in page, or a plugin's page (`page` set). */
export interface SectionTab {
  /** A built-in view id, or `plugin:<pluginId>:<navId>`. */
  view: string;
  label: string;
  icon: LucideIcon;
  page?: PageContribution;
}

/** One sidebar entry and its tabs (more than one shows as tabs). */
export interface NavSection {
  id: string;
  label: string;
  icon: LucideIcon;
  group: Group;
  builtin: boolean;
  tabs: SectionTab[];
  /** A plugin's section: whose it is. */
  pluginId?: string;
  pluginName?: string;
  /** The module it is shown with (a plugin's section). */
  module?: string;
  /** Its pages are a sub-menu under it in the sidebar (not tabs under the header). */
  sub?: boolean;
  /** The built-in sections folded into this one's sub-menu. */
  nested?: string[];
  /** The pages those sections brought (the rest are the plugin section's own). */
  nestedViews?: string[];
}

/** The icons a plugin's nav entry or section may name (`icon`); any other gets the plugin piece. */
export const PLUGIN_ICONS: Record<string, LucideIcon> = {
  phone: Phone,
  calendar: CalendarDays,
  chat: MessageSquare,
  message: MessageSquare,
  mail: Mail,
  bot: Bot,
};

export function pluginIcon(name?: string): LucideIcon {
  return (name && Object.prototype.hasOwnProperty.call(PLUGIN_ICONS, name) && PLUGIN_ICONS[name]) || Puzzle;
}

const GROUPS: readonly Group[] = ['Home', 'Work', 'Setup'];

function pageTab(page: PageContribution): SectionTab {
  return { view: page.view, label: page.label, icon: pluginIcon(page.icon), page };
}

/** A plugin section (its own, or a lone page's). */
function pluginSection(c: SectionContribution): NavSection | null {
  const pages = (c.pages ?? []).filter((p) => p && typeof p.view === 'string' && p.view.startsWith('plugin:'));
  if (!pages.length) return null;
  return {
    id: c.id,
    label: c.label ?? pages[0].label,
    icon: pluginIcon(c.icon ?? pages[0].icon),
    group: c.group && GROUPS.includes(c.group) ? c.group : 'Work',
    builtin: false,
    tabs: pages.map(pageTab),
    pluginId: c.pluginId ?? pages[0].pluginId,
    pluginName: c.pluginName ?? pages[0].pluginName,
    module: c.module ?? pages[0].module,
  };
}

/**
 * The sidebar: `builtins` in their order (each with any plugin pages added to
 * it), and the plugins' own sections at the end of their group. `page` names
 * a built-in view's tab.
 */
export function buildSections(
  builtins: BuiltinSection[],
  contributions: SectionContribution[] | undefined,
  page: (view: string) => { tab: string; icon: LucideIcon },
): NavSection[] {
  const extra = new Map<string, PageContribution[]>();
  const own: NavSection[] = [];
  const known = new Set(builtins.map((b) => b.id));
  for (const c of contributions ?? []) {
    if (!c || typeof c.id !== 'string') continue;
    if (c.builtin && known.has(c.id)) {
      extra.set(c.id, [...(extra.get(c.id) ?? []), ...(c.pages ?? []).filter((p) => p?.view?.startsWith('plugin:'))]);
      continue;
    }
    if (c.builtin) {
      // A section this dashboard does not have (a newer desktop): each page a section of its own, in Work.
      for (const p of c.pages ?? []) {
        const s = pluginSection({ id: p.view, builtin: false, pluginId: p.pluginId, pluginName: p.pluginName, label: p.label, icon: p.icon, pages: [p] });
        if (s) own.push(s);
      }
      continue;
    }
    const s = pluginSection(c);
    if (s) own.push(s);
  }

  const builtinSections: NavSection[] = builtins.map((b) => ({
    id: b.id,
    label: b.label,
    icon: b.icon,
    group: b.group,
    builtin: true,
    tabs: [
      ...b.tabs.map((view) => ({ view, label: page(view).tab, icon: page(view).icon })),
      ...(extra.get(b.id) ?? []).map(pageTab),
    ],
  }));

  // Each group: its built-ins, then the plugins' sections in it.
  const out: NavSection[] = [];
  const groups = [...new Set([...builtins.map((b) => b.group), ...GROUPS])];
  for (const g of groups) {
    out.push(...builtinSections.filter((s) => s.group === g));
    out.push(...own.filter((s) => s.group === g));
  }
  return out;
}

/** The section `view` is a tab of (built-in or plugin), if any. */
export function sectionOf(sections: NavSection[], view: string): NavSection | null {
  return sections.find((s) => s.tabs.some((t) => t.view === view)) ?? null;
}

/** The plugin page `view` is, if it is one of `sections`' tabs. */
export function pluginPageOf(sections: NavSection[], view: string): PageContribution | null {
  for (const s of sections) {
    const tab = s.tabs.find((t) => t.view === view);
    if (tab?.page) return tab.page;
  }
  return null;
}

/**
 * The sidebar with the modules applied. `moduleSections` names each built-in
 * section that belongs to a module (`{ calendar: 'calendar', contacts: 'phone' }`).
 * Such a section:
 *
 * - is left out while its module is off, or not yet known;
 * - goes under the section of the plugin that provides the module, when that
 *   plugin has a section (its own, or its lone page), as that section's
 *   sub-menu: the plugin's pages first, then the built-ins' in `order`
 *   (Phone, Calendar, Contacts, Hours & Services), any not in it after;
 * - stays where it is otherwise.
 *
 * Under it, the plugin's page named like its section (Aokie's "AI
 * Receptionist" page in its "AI Receptionist" section) is named for what it
 * shows: the page's module ("Phone"), else the section's, else another module
 * the plugin provides that has no built-in section of its own name (the
 * Calendar is shown by the Calendar's pages; the phone, whose section is
 * Contacts, is not), else "Overview".
 */
export function arrangeSections(
  sections: NavSection[],
  modules: ModulesSnapshot | null,
  moduleSections: Record<string, string>,
  order: readonly string[] = [],
): NavSection[] {
  let out = sections;
  const records = modules?.modules ?? [];
  // The modules the dashboard shows with pages of their own name (the Calendar's).
  const shownModules = Object.entries(moduleSections)
    .filter(([section, module]) => section === module)
    .map(([, module]) => module);
  const moduleName = (id?: string) => (id ? records.find((m) => m.id === id)?.name : undefined);
  const rank = (view: string) => {
    const i = order.indexOf(view);
    return i < 0 ? order.length : i;
  };
  for (const [sectionId, moduleId] of Object.entries(moduleSections)) {
    const own = out.find((s) => s.id === sectionId && s.builtin);
    if (!own) continue;
    const record = records.find((m) => m.id === moduleId);
    if (!record?.enabled) {
      out = out.filter((s) => s !== own);
      continue;
    }
    const pluginId = record.provider?.pluginId;
    const theirs = pluginId ? out.filter((s) => !s.builtin && s.pluginId === pluginId) : [];
    const host = theirs.find((s) => s.module === moduleId) ?? theirs.find((s) => s.group === own.group) ?? theirs[0];
    if (!host) continue;
    const other = records.find((m) => m.enabled && m.provider?.pluginId === pluginId && !shownModules.includes(m.id));
    const same = (label: string) => label.trim().toLowerCase() === host.label.trim().toLowerCase();
    const tabs = host.tabs.map((t) =>
      t.page && same(t.label) ? { ...t, label: moduleName(t.page.module) ?? moduleName(host.module) ?? other?.name ?? 'Overview' } : t,
    );
    // The plugin section's own pages first, as it gave them; then the nested sections' pages in their order.
    const before = new Set(host.nestedViews ?? []);
    const hostOwn = tabs.filter((t) => !before.has(t.view));
    const nested = [...tabs.filter((t) => before.has(t.view)), ...own.tabs].map((t, i) => ({ t, i }));
    nested.sort((a, b) => rank(a.t.view) - rank(b.t.view) || a.i - b.i);
    const merged: NavSection = {
      ...host,
      sub: true,
      nested: [...(host.nested ?? []), own.id],
      nestedViews: [...before, ...own.tabs.map((t) => t.view)],
      tabs: [...hostOwn, ...nested.map((n) => n.t)],
    };
    out = out.filter((s) => s !== own).map((s) => (s === host ? merged : s));
  }
  return out;
}

/** A plugin section's "New" badge: the first of its pages with a badge that has not been opened here. */
export function newBadge(section: NavSection, seen: Set<string>): string | null {
  if (section.builtin) return null;
  for (const t of section.tabs) {
    if (t.page?.badge && !seen.has(`${t.page.pluginId}:${t.page.navId}`)) return t.page.badge;
  }
  return null;
}
