import { Bot, CalendarDays, Mail, MessageSquare, Phone, Puzzle, type LucideIcon } from 'lucide-react';
import type { PageContribution, SectionContribution } from './api';

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

/** A plugin section's "New" badge: the first of its pages with a badge that has not been opened here. */
export function newBadge(section: NavSection, seen: Set<string>): string | null {
  if (section.builtin) return null;
  for (const t of section.tabs) {
    if (t.page?.badge && !seen.has(`${t.page.pluginId}:${t.page.navId}`)) return t.page.badge;
  }
  return null;
}
