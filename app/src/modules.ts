/**
 * OAIY Desktop's modules, as the app follows them: the parts of OAIY a plugin
 * brings. The phone (calls and texts, the Front desk and its agents) and the
 * calendar are there only while a plugin provides them (Aokie, the AI
 * Receptionist, provides both); everything else is core. The desktop says
 * which are on at `GET /api/modules`, and again on each change at
 * `/api/modules/events`.
 *
 * An older desktop has no such routes (404): the phone is on while Aokie is
 * installed and not turned off, and the calendar while the desktop's calendar
 * says it is available. A page not paired with a desktop has none of them.
 * Turning a module off hides its parts and keeps its data: the Front desk's
 * files and conversations stay where they are, for when it is back.
 */
import { DesktopError, type Desktop } from './desktop/bridge';
import { parsePluginTools, type PluginTool } from './desktop/pluginTools';

export type ModuleId = 'phone' | 'calendar';

export interface ModuleProvider {
  pluginId: string;
  name: string;
  /** The plugin's state (running, crashed, stopped…): a crashed provider keeps its module on. */
  state: string;
  connector?: string;
  declared: boolean;
}

export interface ModuleInfo {
  id: string;
  name: string;
  enabled: boolean;
  /** Why it is off. */
  reason?: string;
  provider: ModuleProvider | null;
}

export interface Modules {
  /** The desktop's revision (-1 when worked out here). */
  revision: number;
  list: ModuleInfo[];
  warnings: string[];
  /** The plugins' actions offered to the agent (`contributions.agent.tools`; none from an older desktop). */
  tools: PluginTool[];
  /** The desktop said so; worked out from an older desktop's plugins; or no desktop is paired. */
  source: 'desktop' | 'fallback' | 'unpaired';
}

/** A page with no desktop: nothing is on. */
export const UNPAIRED: Modules = { revision: -1, list: [], warnings: [], tools: [], source: 'unpaired' };

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** Whether `id` is on (nothing is, before the desktop has said). */
export function isOn(modules: Modules | null, id: ModuleId): boolean {
  return !!modules?.list.some((m) => m.id === id && m.enabled);
}

/** Why `id` is off, as the desktop said it. */
export function whyOff(modules: Modules | null, id: ModuleId): string {
  return modules?.list.find((m) => m.id === id)?.reason ?? '';
}

/** The desktop's snapshot (`{revision, modules, warnings}`), or null when it is not one. */
export function parseModules(body: unknown): Modules | null {
  if (!isRecord(body) || !Array.isArray(body.modules)) return null;
  const list: ModuleInfo[] = [];
  for (const m of body.modules) {
    if (!isRecord(m) || typeof m.id !== 'string' || !m.id) continue;
    const p = isRecord(m.provider) ? m.provider : null;
    list.push({
      id: m.id,
      name: typeof m.name === 'string' ? m.name : m.id,
      enabled: m.enabled === true,
      ...(typeof m.reason === 'string' ? { reason: m.reason } : {}),
      provider: p
        ? {
            pluginId: String(p.pluginId ?? ''),
            name: String(p.name ?? p.pluginId ?? ''),
            state: String(p.state ?? ''),
            ...(typeof p.connector === 'string' ? { connector: p.connector } : {}),
            declared: p.declared === true,
          }
        : null,
    });
  }
  const revision = typeof body.revision === 'number' ? body.revision : -1;
  const warnings = Array.isArray(body.warnings) ? body.warnings.filter((w): w is string => typeof w === 'string') : [];
  return { revision, list, warnings, tools: parsePluginTools(body.contributions), source: 'desktop' };
}

export interface ModuleChange {
  id: string;
  on: boolean;
}

/**
 * What turned on or off from `before` to `after`. Before is null until the
 * desktop has said: everything counts as off then, so the first answer turns
 * on what is on, and says nothing about what is off (nothing of it had started).
 */
export function diffModules(before: Modules | null, after: Modules): ModuleChange[] {
  const on = (m: Modules | null, id: string) => !!m?.list.some((x) => x.id === id && x.enabled);
  const ids = [...new Set([...(before?.list ?? []).map((m) => m.id), ...after.list.map((m) => m.id)])];
  return ids.filter((id) => on(before, id) !== on(after, id)).map((id) => ({ id, on: on(after, id) }));
}

/** The desktop's side that the modules are read from. */
export type ModuleSource = Pick<Desktop, 'modules' | 'moduleEvents' | 'plugins' | 'calendar'>;

/**
 * An older desktop (no `/api/modules`): the phone while Aokie is installed and
 * not turned off, the calendar while the desktop's calendar says it is
 * available (a desktop from before that said nothing: available).
 */
export async function fallbackModules(desktop: Pick<Desktop, 'plugins' | 'calendar'>, signal?: AbortSignal): Promise<Modules> {
  const [plugins, calendar] = await Promise.all([desktop.plugins(signal), desktop.calendar(undefined, undefined, signal).catch(() => null)]);
  const aokie = plugins.find((p) => p.id === 'aokie');
  const provider: ModuleProvider | null = aokie ? { pluginId: 'aokie', name: 'Aokie', state: aokie.state, declared: false } : null;
  const phone = !!aokie && aokie.state !== 'disabled';
  const cal = !!calendar && calendar.available !== false;
  return {
    revision: -1,
    list: [
      { id: 'phone', name: 'Phone', enabled: phone, provider, ...(phone ? {} : { reason: aokie ? 'Aokie is turned off in Plugins.' : 'No installed plugin provides the phone.' }) },
      { id: 'calendar', name: 'Calendar', enabled: cal, provider, ...(cal ? {} : { reason: 'No installed plugin provides the calendar.' }) },
    ],
    warnings: [],
    tools: [],
    source: 'fallback',
  };
}

const notFound = (error: unknown) => error instanceof DesktopError && error.status === 404;

/** The modules now: the desktop's, or (from an older desktop) worked out from its plugins. */
export async function readModules(desktop: ModuleSource, signal?: AbortSignal): Promise<Modules> {
  try {
    const parsed = parseModules(await desktop.modules(signal));
    if (parsed) return parsed;
  } catch (error) {
    if (!notFound(error)) throw error;
  }
  return fallbackModules(desktop, signal);
}

const pause = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });

/**
 * Follow the desktop's modules until `signal` aborts: each snapshot to `apply`,
 * the stream opened again 3 s after it drops. An older desktop (404) is asked
 * for its plugins instead, every `pollMs`.
 */
export async function followModules(desktop: ModuleSource, apply: (modules: Modules) => void, signal: AbortSignal, timing = { retryMs: 3000, pollMs: 30_000 }): Promise<void> {
  while (!signal.aborted) {
    try {
      await desktop.moduleEvents((event) => {
        const parsed = parseModules(event);
        if (parsed && !signal.aborted) apply(parsed);
      }, signal);
    } catch (error) {
      if (signal.aborted) return;
      if (notFound(error)) {
        try {
          const found = await fallbackModules(desktop, signal);
          if (!signal.aborted) apply(found);
        } catch {
          /* the desktop is away: asked again later */
        }
        await pause(timing.pollMs, signal);
        continue;
      }
    }
    await pause(timing.retryMs, signal);
  }
}

/** Whether a Front desk conversation of `kind` shows: calls and texts are the phone's; flows' tasks are core. */
export function sessionShown(kind: string, modules: Modules | null): boolean {
  return kind === 'task' || isOn(modules, 'phone');
}
