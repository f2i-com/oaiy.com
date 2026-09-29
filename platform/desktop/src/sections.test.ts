// Where plugin pages go in the sidebar and in a section's tabs.
import { describe, expect, it } from 'vitest';
import { BookUser, Bot, CalendarDays, Cpu, LayoutDashboard, Phone, Plug, Puzzle, Settings2, Workflow } from 'lucide-react';
import type { ModulesSnapshot, PageContribution, SectionContribution } from './api';
import { arrangeSections, buildSections, newBadge, pluginPageOf, sectionOf, type BuiltinSection } from './sections';

const BUILTINS: BuiltinSection[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, group: 'Home', tabs: ['overview'] },
  { id: 'agent', label: 'Agent', icon: Bot, group: 'Work', tabs: ['agent'] },
  { id: 'flows', label: 'Flows', icon: Workflow, group: 'Work', tabs: ['flows', 'runs'] },
  { id: 'engines', label: 'Engines', icon: Cpu, group: 'Setup', tabs: ['engines', 'models'] },
  { id: 'connections', label: 'Connections', icon: Plug, group: 'Setup', tabs: ['connections', 'providers', 'plugins'] },
  { id: 'settings', label: 'Settings', icon: Settings2, group: 'Setup', tabs: ['settings'] },
];

const page = (view: string) => ({ tab: view.toUpperCase(), icon: Puzzle });

const pageOf = (pluginId: string, navId: string, extra: Partial<PageContribution> = {}): PageContribution => ({
  view: `plugin:${pluginId}:${navId}`,
  pluginId,
  pluginName: pluginId === 'aokie' ? 'Aokie Phone Bridge' : 'Acme Tools',
  navId,
  label: navId[0].toUpperCase() + navId.slice(1),
  screen: 'home',
  ...extra,
});

const lone = (p: PageContribution, extra: Partial<SectionContribution> = {}): SectionContribution => ({
  id: p.view,
  builtin: false,
  pluginId: p.pluginId,
  pluginName: p.pluginName,
  label: p.label,
  icon: p.icon,
  group: 'Work',
  badge: p.badge,
  pages: [p],
  ...extra,
});

const ids = (list: { id: string }[]) => list.map((s) => s.id);

describe('the sidebar with what plugins add', () => {
  it('is the built-in sidebar, unchanged, with nothing added', () => {
    for (const none of [undefined, []]) {
      const s = buildSections(BUILTINS, none, page);
      expect(ids(s)).toEqual(ids(BUILTINS));
      expect(s.every((x) => x.builtin)).toBe(true);
      expect(s[2].tabs.map((t) => t.view)).toEqual(['flows', 'runs']);
      expect(s[2].tabs[0].label).toBe('FLOWS');
    }
  });

  it("puts a plugin's page into a built-in section after that section's own tabs", () => {
    const keys = pageOf('acme', 'keys', { icon: 'bot' });
    const s = buildSections(BUILTINS, [{ id: 'connections', builtin: true, pages: [keys] }], page);
    expect(ids(s)).toEqual(ids(BUILTINS));
    const connections = s.find((x) => x.id === 'connections')!;
    expect(connections.label).toBe('Connections');
    expect(connections.tabs.map((t) => t.view)).toEqual(['connections', 'providers', 'plugins', 'plugin:acme:keys']);
    expect(connections.tabs[3]).toMatchObject({ label: 'Keys', icon: Bot, page: keys });
    // The plugin page opens with the Connections tabs.
    expect(sectionOf(s, 'plugin:acme:keys')?.id).toBe('connections');
    expect(pluginPageOf(s, 'plugin:acme:keys')).toBe(keys);
    expect(pluginPageOf(s, 'providers')).toBeNull();
  });

  it("makes a plugin's own section, with its pages as tabs", () => {
    const stats = pageOf('acme', 'stats');
    const logs = pageOf('acme', 'logs', { icon: 'nothing-we-know' });
    const s = buildSections(BUILTINS, [
      { id: 'plugin-section:acme:tools', builtin: false, pluginId: 'acme', pluginName: 'Acme Tools', label: 'Acme', icon: 'phone', group: 'Work', pages: [stats, logs] },
    ], page);
    const tools = s.find((x) => x.id === 'plugin-section:acme:tools')!;
    expect(tools).toMatchObject({ label: 'Acme', icon: Phone, group: 'Work', builtin: false, pluginName: 'Acme Tools' });
    expect(tools.tabs.map((t) => t.view)).toEqual(['plugin:acme:stats', 'plugin:acme:logs']);
    expect(tools.tabs[1].icon).toBe(Puzzle);
    expect(sectionOf(s, 'plugin:acme:logs')?.id).toBe('plugin-section:acme:tools');
  });

  it('puts a lone page at the end of Work, after the built-ins, as today', () => {
    const receptionist = pageOf('aokie', 'receptionist', { icon: 'phone', badge: 'New', label: 'AI Receptionist' });
    const s = buildSections(BUILTINS, [lone(receptionist)], page);
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'plugin:aokie:receptionist', 'engines', 'connections', 'settings']);
    const r = s[3];
    expect(r).toMatchObject({ label: 'AI Receptionist', icon: Phone, group: 'Work', pluginId: 'aokie' });
    expect(r.tabs).toHaveLength(1);
    expect(sectionOf(s, 'plugin:aokie:receptionist')).toBe(r);
    // Its "New" badge until it has been opened here.
    expect(newBadge(r, new Set())).toBe('New');
    expect(newBadge(r, new Set(['aokie:receptionist']))).toBeNull();
    expect(newBadge(s[0], new Set())).toBeNull();
  });

  it("puts a plugin's Setup section at the end of Setup, and a Home one after Overview", () => {
    const s = buildSections(BUILTINS, [
      lone(pageOf('acme', 'wiring'), { group: 'Setup' }),
      lone(pageOf('acme', 'today'), { group: 'Home' }),
      lone(pageOf('acme', 'work')),
    ], page);
    expect(ids(s)).toEqual([
      'overview', 'plugin:acme:today',
      'agent', 'flows', 'plugin:acme:work',
      'engines', 'connections', 'settings', 'plugin:acme:wiring',
    ]);
  });

  it('never reorders or relabels the built-ins, whatever is added', () => {
    const s = buildSections(BUILTINS, [
      { id: 'flows', builtin: true, pages: [pageOf('acme', 'flowish')] },
      lone(pageOf('acme', 'one')),
      { id: 'settings', builtin: true, pages: [pageOf('acme', 'prefs')] },
    ], page);
    const builtins = s.filter((x) => x.builtin);
    expect(ids(builtins)).toEqual(ids(BUILTINS));
    expect(builtins.map((b) => b.label)).toEqual(BUILTINS.map((b) => b.label));
    expect(s.find((x) => x.id === 'flows')!.tabs.map((t) => t.view)).toEqual(['flows', 'runs', 'plugin:acme:flowish']);
    expect(sectionOf(s, 'plugin:acme:prefs')?.id).toBe('settings');
  });

  it("makes each page of a built-in section this dashboard does not have a section of its own", () => {
    const s = buildSections(BUILTINS, [{ id: 'voice', builtin: true, pages: [pageOf('acme', 'voices')] }], page);
    expect(sectionOf(s, 'plugin:acme:voices')?.id).toBe('plugin:acme:voices');
    expect(ids(s)).toContain('plugin:acme:voices');
  });

  it('leaves out a section with no pages', () => {
    const s = buildSections(BUILTINS, [{ id: 'plugin-section:acme:empty', builtin: false, label: 'Empty', pages: [] }], page);
    expect(ids(s)).toEqual(ids(BUILTINS));
  });
});

describe('a module’s pages under the plugin that provides it', () => {
  // The dashboard's sidebar with the Calendar (Calendar | Hours & Services), as App.tsx has it.
  const WITH_CALENDAR: BuiltinSection[] = [
    ...BUILTINS.slice(0, 3),
    { id: 'calendar', label: 'Calendar', icon: CalendarDays, group: 'Work', tabs: ['calendar', 'hours'] },
    ...BUILTINS.slice(3),
  ];
  const MODULE_SECTIONS = { calendar: 'calendar' };
  const aokie = { pluginId: 'aokie', name: 'Aokie Phone Bridge', state: 'running' as const, declared: true };
  const snapshot = (on: { phone?: boolean; calendar?: boolean }, provider: typeof aokie | null = aokie): ModulesSnapshot => ({
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: !!on.phone, builtin: true, provider },
      { id: 'calendar', name: 'Calendar', enabled: !!on.calendar, builtin: true, provider },
    ],
    contributions: {},
    warnings: [],
  });
  // Aokie's own page, a section of its own: labelled like it (v4 says it is the phone's).
  const receptionist = pageOf('aokie', 'receptionist', { icon: 'phone', badge: 'New', label: 'AI Receptionist', module: 'phone' });
  const arranged = (contributions: SectionContribution[], modules: ModulesSnapshot | null) =>
    arrangeSections(buildSections(WITH_CALENDAR, contributions, page), modules, MODULE_SECTIONS);

  it('nests the Calendar under the AI Receptionist while Aokie provides it and has a section', () => {
    const s = arranged([lone(receptionist, { module: 'phone' })], snapshot({ phone: true, calendar: true }));
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'plugin:aokie:receptionist', 'engines', 'connections', 'settings']);
    const r = s[3];
    expect(r).toMatchObject({ label: 'AI Receptionist', icon: Phone, sub: true, nested: ['calendar'], pluginId: 'aokie' });
    // Its own page first, named for what it shows, then the Calendar's two.
    expect(r.tabs.map((t) => [t.view, t.label])).toEqual([
      ['plugin:aokie:receptionist', 'Phone'],
      ['calendar', 'CALENDAR'],
      ['hours', 'HOURS'],
    ]);
    // Every page opens with the AI Receptionist as its section.
    expect(sectionOf(s, 'calendar')).toBe(r);
    expect(sectionOf(s, 'hours')).toBe(r);
    expect(sectionOf(s, 'plugin:aokie:receptionist')).toBe(r);
    expect(pluginPageOf(s, 'plugin:aokie:receptionist')).toBe(receptionist);
    // Its "New" badge still comes from its own page.
    expect(newBadge(r, new Set())).toBe('New');
  });

  it('names Aokie’s page from the other module it provides when the page names none (a v3 manifest)', () => {
    const plain = pageOf('aokie', 'receptionist', { icon: 'phone', label: 'AI Receptionist' });
    const s = arranged([lone(plain)], snapshot({ phone: true, calendar: true }));
    expect(s.find((x) => x.sub)!.tabs[0].label).toBe('Phone');
  });

  it('keeps a plugin page’s own label when it differs from its section’s', () => {
    const calls = pageOf('aokie', 'calls', { label: 'Calls', module: 'phone' });
    const section: SectionContribution = { id: 'plugin-section:aokie:desk', builtin: false, pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', label: 'AI Receptionist', icon: 'phone', group: 'Work', module: 'phone', pages: [calls] };
    const s = arranged([section], snapshot({ phone: true, calendar: true }));
    expect(s.find((x) => x.sub)!.tabs.map((t) => t.label)).toEqual(['Calls', 'CALENDAR', 'HOURS']);
  });

  it('keeps the Calendar top-level, with its own tabs, while the provider has no section', () => {
    // The phone is off, so the desktop leaves out Aokie's page (and its section); the calendar is still on.
    const s = arranged([], snapshot({ phone: false, calendar: true }));
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'calendar', 'engines', 'connections', 'settings']);
    const calendar = s.find((x) => x.id === 'calendar')!;
    expect(calendar.sub).toBeUndefined();
    expect(calendar.tabs.map((t) => t.view)).toEqual(['calendar', 'hours']);
    expect(sectionOf(s, 'hours')).toBe(calendar);
  });

  it('keeps the Calendar top-level when another plugin has the section', () => {
    const acme = lone(pageOf('acme', 'tools'));
    const s = arranged([acme], snapshot({ phone: true, calendar: true }));
    expect(ids(s)).toContain('calendar');
    expect(s.find((x) => x.id === acme.id)!.sub).toBeUndefined();
  });

  it('hides the Calendar while its module is off, or not yet known, and the AI Receptionist stays as it is', () => {
    for (const modules of [snapshot({ phone: true, calendar: false }), snapshot({ phone: true, calendar: false }, null), null]) {
      const s = arranged([lone(receptionist)], modules);
      expect(ids(s)).not.toContain('calendar');
      expect(sectionOf(s, 'calendar')).toBeNull();
      expect(sectionOf(s, 'hours')).toBeNull();
      const r = s.find((x) => x.id === 'plugin:aokie:receptionist')!;
      expect(r.sub).toBeUndefined();
      expect(r.tabs.map((t) => t.label)).toEqual(['AI Receptionist']);
    }
  });

  it('leaves everything else where it was', () => {
    const s = arranged([lone(receptionist)], snapshot({ phone: true, calendar: true }));
    expect(s.filter((x) => x.builtin).map((x) => x.id)).toEqual(ids(BUILTINS));
    expect(s.find((x) => x.id === 'flows')!.tabs.map((t) => t.view)).toEqual(['flows', 'runs']);
  });
});

describe('Contacts, the phone’s, under the AI Receptionist', () => {
  // The dashboard's sidebar as App.tsx has it: the Calendar (Calendar | Hours & Services), then Contacts.
  const WITH_BOTH: BuiltinSection[] = [
    ...BUILTINS.slice(0, 3),
    { id: 'calendar', label: 'Calendar', icon: CalendarDays, group: 'Work', tabs: ['calendar', 'hours'] },
    { id: 'contacts', label: 'Contacts', icon: BookUser, group: 'Work', tabs: ['contacts'] },
    ...BUILTINS.slice(3),
  ];
  const MODULE_SECTIONS = { calendar: 'calendar', contacts: 'phone' };
  const ORDER = ['calendar', 'contacts', 'hours'];
  const aokie = { pluginId: 'aokie', name: 'Aokie Phone Bridge', state: 'running' as const, declared: true };
  const snapshot = (on: { phone?: boolean; calendar?: boolean }): ModulesSnapshot => ({
    revision: 1,
    modules: [
      { id: 'phone', name: 'Phone', enabled: !!on.phone, builtin: true, provider: aokie },
      { id: 'calendar', name: 'Calendar', enabled: !!on.calendar, builtin: true, provider: aokie },
    ],
    contributions: {},
    warnings: [],
  });
  const receptionist = pageOf('aokie', 'receptionist', { icon: 'phone', label: 'AI Receptionist', module: 'phone' });
  const arranged = (contributions: SectionContribution[], modules: ModulesSnapshot | null) =>
    arrangeSections(buildSections(WITH_BOTH, contributions, page), modules, MODULE_SECTIONS, ORDER);
  const pages = (s: ReturnType<typeof arranged>) => s.find((x) => x.sub)?.tabs.map((t) => [t.view, t.label]);

  it('goes in order: Phone, Calendar, Contacts, Hours & Services', () => {
    const s = arranged([lone(receptionist, { module: 'phone' })], snapshot({ phone: true, calendar: true }));
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'plugin:aokie:receptionist', 'engines', 'connections', 'settings']);
    expect(pages(s)).toEqual([
      ['plugin:aokie:receptionist', 'Phone'],
      ['calendar', 'CALENDAR'],
      ['contacts', 'CONTACTS'],
      ['hours', 'HOURS'],
    ]);
    const r = s.find((x) => x.sub)!;
    expect(r.nested).toEqual(['calendar', 'contacts']);
    expect(sectionOf(s, 'contacts')).toBe(r);
  });

  it('is there with the phone alone, and gone with it', () => {
    let s = arranged([lone(receptionist, { module: 'phone' })], snapshot({ phone: true, calendar: false }));
    expect(pages(s)).toEqual([
      ['plugin:aokie:receptionist', 'Phone'],
      ['contacts', 'CONTACTS'],
    ]);
    // The phone off: the desktop leaves out Aokie's page, and Contacts goes too; the Calendar stays top-level.
    s = arranged([], snapshot({ phone: false, calendar: true }));
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'calendar', 'engines', 'connections', 'settings']);
    expect(sectionOf(s, 'contacts')).toBeNull();
    // Not yet known: neither is there.
    s = arranged([lone(receptionist)], null);
    expect(sectionOf(s, 'contacts')).toBeNull();
    expect(sectionOf(s, 'calendar')).toBeNull();
  });

  it('stays top-level while the phone’s plugin has no section', () => {
    const s = arranged([], snapshot({ phone: true, calendar: true }));
    expect(ids(s)).toEqual(['overview', 'agent', 'flows', 'calendar', 'contacts', 'engines', 'connections', 'settings']);
    expect(s.find((x) => x.id === 'contacts')!.sub).toBeUndefined();
  });

  it('still names Aokie’s page Phone when its manifest names no module (Contacts is the phone’s, not a page of its own)', () => {
    const plain = pageOf('aokie', 'receptionist', { icon: 'phone', label: 'AI Receptionist' });
    const s = arranged([lone(plain)], snapshot({ phone: true, calendar: true }));
    expect(pages(s)?.map(([, label]) => label)).toEqual(['Phone', 'CALENDAR', 'CONTACTS', 'HOURS']);
  });

  it('keeps a plugin’s own pages first, and a page it adds to the Calendar after the ordered ones', () => {
    const calls = pageOf('aokie', 'calls', { label: 'Calls', module: 'phone' });
    const extra = pageOf('aokie', 'rosters', { label: 'Rosters' });
    const section: SectionContribution = { id: 'plugin-section:aokie:desk', builtin: false, pluginId: 'aokie', pluginName: 'Aokie Phone Bridge', label: 'AI Receptionist', icon: 'phone', group: 'Work', module: 'phone', pages: [receptionist, calls] };
    const s = arranged([section, { id: 'calendar', builtin: true, pages: [extra] }], snapshot({ phone: true, calendar: true }));
    expect(pages(s)?.map(([view]) => view)).toEqual(['plugin:aokie:receptionist', 'plugin:aokie:calls', 'calendar', 'contacts', 'hours', 'plugin:aokie:rosters']);
  });
});
