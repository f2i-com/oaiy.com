import { describe, expect, it } from 'vitest';
import { DesktopError, type Desktop } from '../../src/desktop/bridge';
import { UNPAIRED, diffModules, fallbackModules, followModules, isOn, parseModules, readModules, sessionShown, whyOff, type Modules, type ModuleSource } from '../../src/modules';
import { Sessions } from '../../src/sessions';
import type { AgentOptions, SessionTool } from '../../src/agent/agent';
import { DEFAULT_MESSAGE_SETTINGS } from '../../src/settings';

/** The desktop's snapshot, as GET /api/modules gives it. */
function snapshot(revision: number, phone: boolean, calendar: boolean, state = 'running') {
  const provider = { pluginId: 'aokie', name: 'Aokie Phone Bridge', state, connector: 'aokie', declared: false };
  return {
    revision,
    modules: [
      { id: 'phone', name: 'Phone', enabled: phone, builtin: true, provider, leases: ['answer-calls', 'answer-texts'], ...(phone ? {} : { reason: 'Aokie Phone Bridge is turned off in Plugins.' }) },
      { id: 'calendar', name: 'Calendar', enabled: calendar, builtin: true, provider, ...(calendar ? {} : { reason: 'Aokie Phone Bridge is turned off in Plugins.' }) },
    ],
    contributions: {},
    warnings: [],
  };
}

/** A desktop that answers the modules routes (or 404s them, as an older one does). */
function fakeDesktop(options: { modules?: unknown; old?: boolean; plugins?: Array<{ id: string; state: string }>; calendar?: { available?: boolean } | null; events?: unknown[] }) {
  const calls: string[] = [];
  const desktop: ModuleSource = {
    modules: async () => {
      calls.push('modules');
      if (options.old) throw new DesktopError('HTTP 404', 404);
      return options.modules;
    },
    moduleEvents: async (onSnapshot) => {
      calls.push('moduleEvents');
      if (options.old) throw new DesktopError("OAIY Desktop's modules: HTTP 404", 404);
      for (const e of options.events ?? []) onSnapshot(e as Record<string, unknown>);
    },
    plugins: async () => {
      calls.push('plugins');
      return options.plugins ?? [];
    },
    calendar: async () => {
      calls.push('calendar');
      if (options.calendar === null) throw new DesktopError('HTTP 404', 404);
      return { settings: {}, appointments: [], now: '', ...(options.calendar ?? {}) };
    },
  };
  return { desktop, calls };
}

describe("the desktop's modules", () => {
  it('reads the snapshot: which are on, why one is off, and who provides it', () => {
    const on = parseModules(snapshot(3, true, true, 'crashed'))!;
    expect(on.revision).toBe(3);
    expect(on.source).toBe('desktop');
    expect(isOn(on, 'phone')).toBe(true);
    expect(isOn(on, 'calendar')).toBe(true);
    // A crashed provider keeps its module on, and says how it is.
    expect(on.list[0].provider).toEqual({ pluginId: 'aokie', name: 'Aokie Phone Bridge', state: 'crashed', connector: 'aokie', declared: false });
    const off = parseModules(snapshot(4, false, false))!;
    expect(isOn(off, 'phone')).toBe(false);
    expect(whyOff(off, 'phone')).toBe('Aokie Phone Bridge is turned off in Plugins.');
    expect(parseModules({ error: 'nope' })).toBeNull();
    expect(parseModules(null)).toBeNull();
    // Nothing is on before the desktop has said.
    expect(isOn(null, 'phone')).toBe(false);
  });

  it("reads the plugins' agent tools from the snapshot's contributions (none without a desktop that lists them)", async () => {
    const tool = {
      pluginId: 'aokie', name: 'phone_sms_threads', action: 'aokie.phone/sms.threads', definition: 'aokie.phone', actionId: 'sms.threads',
      description: 'Every conversation on the paired phone.', inputSchema: { type: 'object' }, sideEffects: 'none', audience: ['project', 'runner'],
    };
    const withTools = parseModules({ ...snapshot(5, true, true), contributions: { agent: { tools: [tool, { name: 'broken' }] } } })!;
    expect(withTools.tools.map((t) => [t.pluginId, t.name, t.audience])).toEqual([['aokie', 'phone_sms_threads', ['project', 'runner']]]);
    expect(parseModules(snapshot(6, true, true))!.tools).toEqual([]);
    expect(UNPAIRED.tools).toEqual([]);
    const old = await fallbackModules(fakeDesktop({ plugins: [{ id: 'aokie', state: 'running' }], calendar: {} }).desktop);
    expect(old.tools).toEqual([]);
  });

  it('says what turned on and off, and nothing went off before it was known', () => {
    const on = parseModules(snapshot(1, true, true))!;
    const phoneOff = parseModules(snapshot(2, false, true))!;
    // The first answer turns on what is on.
    expect(diffModules(null, on)).toEqual([{ id: 'phone', on: true }, { id: 'calendar', on: true }]);
    // …and says nothing of what is off: nothing of it had started, so nothing is left behind.
    expect(diffModules(null, parseModules(snapshot(1, false, false))!)).toEqual([]);
    expect(diffModules(on, phoneOff)).toEqual([{ id: 'phone', on: false }]);
    expect(diffModules(phoneOff, phoneOff)).toEqual([]);
    // A crash is a new revision, not a change of what is on.
    expect(diffModules(on, parseModules(snapshot(3, true, true, 'crashed'))!)).toEqual([]);
    // Unpaired: everything that was on goes off.
    expect(diffModules(on, UNPAIRED)).toEqual([{ id: 'phone', on: false }, { id: 'calendar', on: false }]);
  });

  it('an unpaired page has nothing on', () => {
    expect(UNPAIRED.source).toBe('unpaired');
    expect(isOn(UNPAIRED, 'phone')).toBe(false);
    expect(isOn(UNPAIRED, 'calendar')).toBe(false);
    expect(sessionShown('call', UNPAIRED)).toBe(false);
    expect(sessionShown('task', UNPAIRED)).toBe(true);
  });

  it('an older desktop (404) is read from its plugins and its calendar', async () => {
    const aokie = fakeDesktop({ old: true, plugins: [{ id: 'aokie', state: 'crashed' }], calendar: { available: true } });
    const found = await readModules(aokie.desktop);
    expect(aokie.calls).toEqual(['modules', 'plugins', 'calendar']);
    expect(found.source).toBe('fallback');
    expect(isOn(found, 'phone')).toBe(true);
    expect(isOn(found, 'calendar')).toBe(true);
    // Aokie turned off: no phone; the calendar as the desktop says.
    const off = await fallbackModules(fakeDesktop({ plugins: [{ id: 'aokie', state: 'disabled' }], calendar: { available: false } }).desktop);
    expect(isOn(off, 'phone')).toBe(false);
    expect(isOn(off, 'calendar')).toBe(false);
    // No Aokie at all, and a desktop from before the calendar said whether it is in use.
    const none = await fallbackModules(fakeDesktop({ plugins: [{ id: 'weather', state: 'running' }], calendar: {} }).desktop);
    expect(isOn(none, 'phone')).toBe(false);
    expect(isOn(none, 'calendar')).toBe(true);
    // One with no calendar at all.
    expect(isOn(await fallbackModules(fakeDesktop({ plugins: [], calendar: null }).desktop), 'calendar')).toBe(false);
    // A desktop with modules is not asked for its plugins.
    const current = fakeDesktop({ modules: snapshot(7, true, false) });
    expect(isOn(await readModules(current.desktop), 'calendar')).toBe(false);
    expect(current.calls).toEqual(['modules']);
    // Another failure is not taken as an older desktop.
    const away = fakeDesktop({});
    away.desktop.modules = async () => {
      throw new DesktopError('fetch failed', 0);
    };
    await expect(readModules(away.desktop)).rejects.toThrow('fetch failed');
  });

  it('follows the stream, and asks an older desktop now and then instead', async () => {
    const seen: Modules[] = [];
    const abort = new AbortController();
    const current = fakeDesktop({ events: [snapshot(1, true, true), { not: 'a snapshot' }, snapshot(2, false, true)] });
    const following = followModules(current.desktop, (m) => {
      seen.push(m);
      if (seen.length === 2) abort.abort();
    }, abort.signal, { retryMs: 5, pollMs: 5 });
    await following;
    expect(seen.map((m) => [m.revision, isOn(m, 'phone')])).toEqual([[1, true], [2, false]]);

    const old = fakeDesktop({ old: true, plugins: [{ id: 'aokie', state: 'running' }], calendar: { available: true } });
    const polled: Modules[] = [];
    const stop = new AbortController();
    await followModules(old.desktop, (m) => {
      polled.push(m);
      if (polled.length === 2) stop.abort();
    }, stop.signal, { retryMs: 5, pollMs: 5 });
    expect(polled.every((m) => m.source === 'fallback' && isOn(m, 'phone'))).toBe(true);
    expect(old.calls.filter((c) => c === 'plugins')).toHaveLength(2);
  });

  it("calls and texts are the phone's; flows' tasks always show", () => {
    const on = parseModules(snapshot(1, true, true));
    const off = parseModules(snapshot(2, false, true));
    for (const kind of ['call', 'sms']) {
      expect(sessionShown(kind, on)).toBe(true);
      expect(sessionShown(kind, off)).toBe(false);
      expect(sessionShown(kind, null)).toBe(false);
    }
    expect(sessionShown('task', off)).toBe(true);
    expect(sessionShown('task', null)).toBe(true);
  });

  it("a text thread has the calendar's tools only while the calendar is on", async () => {
    let tools: AgentOptions['sessionTools'];
    const kinds: string[] = [];
    const project = { loadSessions: async () => [], saveSessions: async () => {}, loadSessionChat: async () => [], saveSessionChat: async () => {}, loadCallers: async () => [], saveCallers: async () => {} };
    const sessions = new Sessions(
      project as never,
      (extra, kind) => {
        tools = extra.sessionTools;
        // The app is told which kind of conversation it makes an agent for (the plugins' tools go by it).
        kinds.push(kind);
        return { turns: [], savedTurns: () => [] } as never;
      },
      () => ({ ...DEFAULT_MESSAGE_SETTINGS, answer: false }),
      () => ({}) as Desktop,
      { changed: () => {}, event: () => {} },
    );
    let calendar = true;
    sessions.calendarOn = () => calendar;
    await sessions.textArrived('+61400000001', 'Lance', 'Hi');
    expect(kinds).toEqual(['sms']);
    const names = () => (typeof tools === 'function' ? tools() : (tools ?? [])).map((t: SessionTool) => t.spec.name);
    expect(names()).toEqual(expect.arrayContaining(['send_text_message', 'calendar_free_times', 'request_appointment']));
    calendar = false;
    expect(names()).toContain('send_text_message');
    expect(names()).not.toContain('calendar_free_times');
    expect(names()).not.toContain('request_appointment');
  });
});
