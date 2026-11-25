// Focused regression tests for all 11 Wave-4 audit findings in OAIY AI Agent.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import GUEST_SOURCE from '../../src/preview/zippDom.guest.js?raw';
import HOST_SOURCE from '../../public/webpage/page-host.js?raw';
import BRIDGE_SOURCE from '../../scripts/softn-bridge/bot-bridge.js?raw';
import { WorkspaceInstructions } from '../../src/agent/instructions';
import type { Agent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import { pairingStatus, startPairing } from '../../src/desktop/bridge';
import { Outreach, type Campaign, type OutreachLink } from '../../src/outreach';
import { PhoneLine } from '../../src/phoneLine';
import { setLocalCountry } from '../../src/phoneNumbers';
import { Preview } from '../../src/preview/preview';
import { Sessions } from '../../src/sessions';
import { DEFAULT_MESSAGE_SETTINGS } from '../../src/settings';
import { ChatPane } from '../../src/ui/chat';
import { OpenProject, type ProjectMeta } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';

const load = (specifier: string): Promise<unknown> => import(/* @vite-ignore */ specifier);
const fs = (await load('node:fs')) as { existsSync(path: URL): boolean; readFileSync(path: URL): Uint8Array };
const ZIPP = new URL('../../public/zipp/', import.meta.url);
const HAVE_ZIPP = fs.existsSync(new URL('zipp_wasm_bg.wasm', ZIPP));

interface ZippEngine {
  setLocalStorageBridge(bridge: unknown): void;
  setSyncHostCapabilities(caps: string[]): void;
  initScript(source: string): unknown;
  callFunction(name: string, args: unknown[]): unknown;
}
interface ZippGlue {
  default(opts: { module_or_path: Uint8Array }): Promise<unknown>;
  Engine: new () => ZippEngine;
}

beforeAll(() => setLocalCountry('AU'));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const T0 = new Date(2026, 10, 25, 10, 0).getTime();

describe('[P0 #1 & #2, P1 #3, P2 #6] Webpage preview sandbox isolation, detached sanitization, bridge auth & __zdom_* capability token', () => {
  it('enforces element/realm containment, allowlisted window access, detached HTML sanitization, and null-prototype storage in page-host.js', () => {
    expect(HOST_SOURCE).toContain("const BLOCKED_TAGS = new Set(['base', 'iframe', 'frame', 'frameset', 'object', 'embed']);");
    expect(HOST_SOURCE).toContain("'contentWindow', 'contentDocument', 'getSVGDocument'");
    expect(HOST_SOURCE).toContain('const ALLOWED_WIN_PROPS = new Set([');
    expect(HOST_SOURCE).toContain('if (obj === window) return BLOCKED_WIN_PROPS.has(p) || !ALLOWED_WIN_PROPS.has(p);');
    expect(HOST_SOURCE).toContain('const sanitizeHtmlFragment = (rawHtml, contextEl) =>');
    expect(HOST_SOURCE).toContain("'parseFromString'");
    for (const m of ['insertAdjacentElement', 'insertNode', 'surroundContents', 'moveBefore']) {
      expect(HOST_SOURCE).toContain(`'${m}'`);
    }
    expect(HOST_SOURCE).toContain('DANGEROUS_LINK_REL');
    expect(HOST_SOURCE).toContain("if (method === 'attachShadow')");
    expect(HOST_SOURCE).toContain("window.addEventListener('auxclick', followLinks, true);");
    expect(HOST_SOURCE).toContain('event.composedPath');
    expect(HOST_SOURCE).toContain('storage = { local: createNullStore(message.storage), session: nativeObjectCreate(null) };');
    expect(HOST_SOURCE).toContain("typeof arg[1] === 'string'");
    expect(HOST_SOURCE).toContain("obj instanceof Attr && (prop === 'value' || prop === 'nodeValue' || prop === 'textContent')");
    expect(BRIDGE_SOURCE).toContain('__oaiyToken: bridgeToken');
    expect(BRIDGE_SOURCE).toContain("data.type === 'webpage:init' && typeof data.bridgeToken === 'string'");
  });

  it('rejects unauthenticated or mismatched __oaiyToken messages in Preview.fromBridge() when bridgeToken is set, while allowing SoftN/modelview frames without a token', () => {
    const vfs = new Vfs();
    const fakeWin = {} as Window;
    const replies: unknown[] = [];
    const statuses: string[] = [];
    const fakePreview = {
      frame: { contentWindow: fakeWin } as unknown as HTMLIFrameElement,
      bridgeToken: 'per-load-secret-token',
      bridgeReady: false,
      target: { kind: 'page' as const, path: 'index.html' },
      vfs,
      problems: [] as Array<{ level: string; message: string; at: number }>,
      fromPort: new WeakSet(),
      collecting: null,
      requests: new Map([[1, (r: unknown) => replies.push(r)]]),
      storage: new Map<string, Record<string, string>>(),
      pendingPage: null,
      showProblems: () => {},
      refreshApps: () => [],
      setPage: () => {},
      setStatus: (s: string) => statuses.push(s),
    };
    const fromBridge = (Preview.prototype as unknown as { fromBridge: (e: MessageEvent) => void }).fromBridge;
    const dispatch = (data: unknown) => fromBridge.call(fakePreview, { source: fakeWin, data } as unknown as MessageEvent);

    // Without token or with wrong token -> every bot:* message is dropped
    dispatch({ __oaiyAgent: true, type: 'bot:bridge' });
    dispatch({ __oaiyAgent: true, __oaiyToken: 'wrong', type: 'bot:bridge' });
    dispatch({ __oaiyAgent: true, type: 'bot:storage', store: { evil: '1' } });
    dispatch({ __oaiyAgent: true, __oaiyToken: 'wrong', type: 'bot:download', filename: 'pwn.txt', bytes: new Uint8Array([1, 2, 3]), userInitiated: false });
    dispatch({ __oaiyAgent: true, type: 'bot:problem', level: 'error', message: 'forged error' });
    dispatch({ __oaiyAgent: true, type: 'bot:reply', id: 1, result: { ok: true, page: 'forged' } });

    expect(fakePreview.bridgeReady).toBe(false);
    expect(fakePreview.storage.size).toBe(0);
    expect(vfs.exists('/exports/pwn.txt')).toBe(false);
    expect(fakePreview.problems).toHaveLength(0);
    expect(replies).toHaveLength(0);

    // With exact per-load token -> messages succeed
    dispatch({ __oaiyAgent: true, __oaiyToken: 'per-load-secret-token', type: 'bot:bridge' });
    dispatch({ __oaiyAgent: true, __oaiyToken: 'per-load-secret-token', type: 'bot:storage', store: { legit: 'yes' } });
    dispatch({ __oaiyAgent: true, __oaiyToken: 'per-load-secret-token', type: 'bot:reply', id: 1, result: { ok: true, page: 'ok' } });

    expect(fakePreview.bridgeReady).toBe(true);
    expect(fakePreview.storage.get('index.html')?.legit).toBe('yes');
    expect(replies).toEqual([{ ok: true, page: 'ok' }]);

    // SoftN / modelview frames have bridgeToken === null -> tokenless messages still work seamlessly
    fakePreview.bridgeToken = null as unknown as string;
    fakePreview.bridgeReady = false;
    dispatch({ __oaiyAgent: true, type: 'bot:bridge' });
    expect(fakePreview.bridgeReady).toBe(true);
  });

  it('enforces per-session __zdom_* capability token, non-writable/non-configurable hooks, primordial safety, and PopStateEvent encoding in real Zipp VM', async () => {
    // Source-level check: zero .call/.apply/.push and zero un-cached String(/JSON./Promise./Object.keys in guest facade runtime paths
    expect(GUEST_SOURCE).not.toMatch(/\.(?:call|apply|push)\(/);

    if (!HAVE_ZIPP) return;
    const glue = (await load(new URL('zipp_wasm.js', ZIPP).href)) as ZippGlue;
    await glue.default({ module_or_path: fs.readFileSync(new URL('zipp_wasm_bg.wasm', ZIPP)) });

    // 1. Un-replaced token placeholder must fail closed
    const badEngine = new glue.Engine();
    badEngine.setLocalStorageBridge({ getItem: () => null, setItem() {}, removeItem() {}, clear() {} });
    badEngine.setSyncHostCapabilities(['ls.getItem']);
    expect(() => badEngine.initScript(GUEST_SOURCE)).toThrow(/missing session token/i);

    // 2. Real session token injected into closure (never on globalThis)
    const sessionToken = 'tok-wave4-unit-secret';
    const seenHostCalls: Array<unknown[]> = [];
    const engine = new glue.Engine();
    engine.setLocalStorageBridge({
      getItem: (key: string) => {
        const MAGIC = '\u0000oaiy.agent.page:';
        if (!key.startsWith(MAGIC)) return null;
        const req = JSON.parse(key.slice(MAGIC.length)) as unknown[];
        seenHostCalls.push(req);
        const [op] = req;
        if (op === 'window') return { ok: { $h: 1, c: ['Window'] } };
        if (op === 'document') return { ok: { $h: 2, c: ['Document', 'Node', 'EventTarget'] } };
        if (op === 'location') return { ok: { origin: 'https://preview.oaiy.com', host: 'preview.oaiy.com', path: '/index.html', search: '' } };
        if (op === 'get') return { ok: { $m: 1 } };
        return { ok: null };
      },
      setItem() {},
      removeItem() {},
      clear() {},
    });
    engine.setSyncHostCapabilities(['ls.getItem']);
    engine.initScript(GUEST_SOURCE.replace('"__OAIY_ZDOM_TOKEN__"', () => JSON.stringify(sessionToken)));

    // Unauthorized token calls must throw
    expect(() => engine.callFunction('__zdom_cb', ['wrong', 1, null, []])).toThrow(/unauthorized/i);
    expect(() => engine.callFunction('__zdom_settle', ['wrong', 1, true, null])).toThrow(/unauthorized/i);
    expect(() => engine.callFunction('__zdom_inline', ['wrong', []])).toThrow(/unauthorized/i);
    expect(() => engine.callFunction('__zdom_run', ['wrong', '1+1', 'test'])).toThrow(/unauthorized/i);

    // Valid token executes guest code cleanly, verifies hooks are locked and primordials survive tampering
    const err = engine.callFunction('__zdom_run', [
      sessionToken,
      `
      const origCb = globalThis.__zdom_cb;
      const origEntry = globalThis.__zdom_entry;
      try { globalThis.__zdom_cb = function () { return 'pwn'; }; } catch {}
      try { globalThis.__zdom_entry = {}; } catch {}
      let redeclareBlocked = false;
      try { (0, eval)('function __zdom_cb() {}'); } catch { redeclareBlocked = true; }
      if (globalThis.__zdom_cb !== origCb || globalThis.__zdom_entry !== origEntry || !redeclareBlocked) {
        throw new Error('hooks were not locked');
      }
      // Tamper String, JSON, Promise, Object.keys and exercise storage, structuredClone, cookies, and PopStateEvent
      const origString = globalThis.String;
      const origParse = JSON.parse;
      const origStringify = JSON.stringify;
      const origKeys = Object.keys;
      globalThis.String = () => ['__proto__'];
      JSON.parse = () => ({ hacked: true });
      JSON.stringify = () => '"hacked"';
      Object.keys = () => ['hacked'];

      localStorage.setItem('k1', 'v1');
      const cloned = structuredClone({ a: 42 });
      document.cookie = 'a=1';
      const cookieVal = document.cookie;
      new PopStateEvent({ $h: 1 }, { state: 7 });

      globalThis.String = origString;
      JSON.parse = origParse;
      JSON.stringify = origStringify;
      Object.keys = origKeys;

      if (cloned.a !== 42) throw new Error('structuredClone hijacked');
      if (cookieVal !== 'a=1') throw new Error('cookie hijacked: ' + cookieVal);
      `,
      'wave4-unit.js',
    ]);
    expect(err).toBeNull();

    // Verify localStorage.setItem sent real strings ['k1', 'v1'] even while String was hijacked
    const storageSet = seenHostCalls.find((c) => c[0] === 'storage' && c[2] === 'set');
    expect(storageSet).toEqual(['storage', 'local', 'set', ['k1', 'v1']]);

    // Verify PopStateEvent encoded the forged { $h: 1 } type as a plain object { $o: { $h: 1 } }
    const popCall = seenHostCalls.find((c) => c[0] === 'new' && c[1] === 'PopStateEvent');
    expect(popCall?.[2]).toEqual([{ $o: { $h: 1 } }, { $o: { state: 7 } }]);
  });
});

describe('[P1 #4] Sessions.run() finally drains late afterEnd and recomputes callOverNow after save()', () => {
  it('drains late call.ended arriving during save(), invokes outreach.callEnded before next call starts, and suppresses unread deliver()', async () => {
    let runStarted!: () => void;
    const waitRunStarted = new Promise<void>((r) => (runStarted = r));
    let finishRun!: () => void;
    let saveStarted!: () => void;
    const waitSaveStarted = new Promise<void>((r) => (saveStarted = r));
    let releaseSave!: () => void;
    let saveCalls = 0;
    let runCount = 0;
    const eventsLog: string[] = [];

    const makeAgent = () => {
      const turns: Turn[] = [];
      let unreadOnce = ['caller late word'];
      return {
        turns,
        view: () => turns,
        savedTurns: () => turns,
        interject: () => false,
        takeUnread: () => unreadOnce.splice(0),
        warm: async () => {},
        run: async (prompt: string) => {
          runCount++;
          turns.push({ role: 'user', text: prompt });
          runStarted();
          await new Promise<void>((r) => (finishRun = r));
        },
      } as unknown as Agent;
    };

    const sessions = new Sessions(
      {
        loadSessions: async () => [],
        saveSessions: async () => {
          eventsLog.push('saveIndex');
        },
        loadSessionChat: async () => [],
        saveSessionChat: async () => {
          saveCalls++;
          // saveCalls === 1: call 1 start save; saveCalls === 2: run() finally save (BEFORE call.ended arrives!)
          if (saveCalls === 2) {
            eventsLog.push('saveFinally:start');
            saveStarted();
            await new Promise<void>((r) => (releaseSave = r));
            eventsLog.push('saveFinally:done');
          } else if (saveCalls === 3) {
            eventsLog.push('saveLateAfterEnd');
          }
        },
        loadCallers: async () => [],
        saveCallers: async () => {},
      } as never,
      makeAgent,
      () => ({ ...DEFAULT_MESSAGE_SETTINGS, answer: false, instructions: '', calls: true, callInstructions: '' }),
      () => null,
      { changed: () => {}, event: () => {} },
    );

    const link1 = { campaignId: 'camp-late-1', personId: 'p1', kind: 'call', name: 'C1', person: 'Jane', objective: 'Obj1', inbound: false, instructions: () => 'Obj1' } as unknown as OutreachLink;
    const link2 = { campaignId: 'camp-late-2', personId: 'p2', kind: 'call', name: 'C2', person: 'Jane', objective: 'Obj2', inbound: false, instructions: () => 'Obj2' } as unknown as OutreachLink;
    const callEndedCampaigns: string[] = [];
    sessions.outreach = {
      forCall: (callId: string) => (callId === 'call_late_1' ? link1 : callId === 'call_late_2' ? link2 : undefined),
      callEnded: (link: OutreachLink) => {
        eventsLog.push(`callEnded:${link.campaignId}`);
        callEndedCampaigns.push(link.campaignId);
        return false;
      },
    } as unknown as Outreach;

    const call = (await sessions.callEvent({ type: 'call.started', callId: 'call_late_1', from: '+61412345678' }))!;
    await sessions.callEvent({ type: 'call.caller', callId: 'call_late_1', text: 'Hello on call 1' });
    await waitRunStarted;

    // Let agent.run() complete naturally so finally enters await this.save(session) BEFORE call.ended arrives.
    finishRun();
    await waitSaveStarted;

    // Now call.ended arrives WHILE finally is suspended in await this.save(session) (session.running is still set!).
    const ending = sessions.callEvent({ type: 'call.ended', callId: 'call_late_1' });
    const starting2 = sessions.callEvent({ type: 'call.started', callId: 'call_late_2', from: '+61412345678' }).then((s) => {
      eventsLog.push('call2:started');
      return s;
    });
    await ending;
    expect(call.afterEnd).toBeDefined();

    // Resume save() and let everything settle.
    releaseSave();
    await starting2;
    await new Promise((r) => setTimeout(r, 20));

    // Late afterEnd must have appended the end note, saved it, called outreach.callEnded BEFORE call 2 started,
    // and unread must NOT have triggered a second run for the already-ended call 1!
    expect(call.afterEnd).toBeUndefined();
    expect(callEndedCampaigns).toEqual(['camp-late-1']);
    expect(runCount).toBe(1);
    expect(eventsLog.indexOf('saveLateAfterEnd')).toBeGreaterThan(eventsLog.indexOf('saveFinally:done'));
    expect(eventsLog.indexOf('callEnded:camp-late-1')).toBeGreaterThan(eventsLog.indexOf('saveLateAfterEnd'));
    expect(eventsLog.indexOf('callEnded:camp-late-1')).toBeLessThan(eventsLog.indexOf('call2:started'));
    expect(call.agent.turns.some((t) => t.role === 'user' && t.text.includes('The call ended'))).toBe(true);
    expect(call.outreach).toBe(link2);
  });
});

describe('[P1 #5] Outreach.stopWord() and Outreach.about() skip unstarted campaigns', () => {
  function makeTextCampaign(over: Partial<Campaign>): Campaign {
    return {
      id: 'out-text',
      kind: 'text',
      name: 'Text campaign',
      slug: 'text-campaign',
      objective: 'Confirm',
      collect: [],
      openingLine: '',
      textTemplate: 'Hi {first_name}',
      voicemail: 'no_message',
      voicemailMessage: '',
      retries: { times: 0, gapMinutes: 60 },
      replyDeadlineHours: 24,
      window: { from: '09:00', to: '19:00' },
      afterwards: '',
      origin: { kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' },
      resultsPath: '/outreach/text-campaign/results.md',
      state: 'paused',
      waitingFor: '',
      faults: 0,
      createdAt: T0 - 1000,
      approvedAt: 0,
      endedAt: null,
      report: { text: '', pending: false, delivered: false },
      lines: [],
      people: [
        {
          id: 'p1',
          name: 'Alex',
          number: '+61411111111',
          raw: '0411111111',
          fields: {},
          state: 'awaiting_reply',
          tries: 1,
          nextAt: 0,
          attempt: { n: 1, at: T0 - 500, messageId: 'm1' },
          answers: {},
          history: [],
        },
      ],
      skipped: [],
      ...over,
    };
  }

  it('ignores unstarted campaigns in stopWord() and about() while still handling started ones', async () => {
    const unstarted = makeTextCampaign({ id: 'out-unstarted', approvedAt: 0 });
    const started = makeTextCampaign({
      id: 'out-started',
      approvedAt: T0 - 500,
      state: 'running',
      people: [
        {
          id: 'p2',
          name: 'Sam',
          number: '+61422222222',
          raw: '0422222222',
          fields: {},
          state: 'awaiting_reply',
          tries: 1,
          nextAt: 0,
          attempt: { n: 1, at: T0 - 400, messageId: 'm2' },
          answers: {},
          history: [],
        },
      ],
    });
    const dncSaved: unknown[] = [];
    const posted: string[] = [];
    const outreach = new Outreach({
      store: {
        loadOutreach: async () => [unstarted, started],
        saveOutreach: async () => {},
        loadDoNotContact: async () => [],
        saveDoNotContact: async (list) => void dncSaved.push(structuredClone(list)),
      },
      files: () => new Vfs(),
      desktop: () => null,
      phone: () => ({ holdsCalls: true, holdsTexts: true, connected: true }),
      line: new PhoneLine(),
      callbacks: () => null,
      screening: async () => ({ acceptPattern: '', blockedNumbers: '', rejectPrivate: false }),
      callsToOaiy: async () => true,
      rules: async () => ({ quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true }),
      sessions: () => null,
      post: (_c, line) => {
        posted.push(line);
        return true;
      },
      report: () => {},
      identity: () => ({ business: 'Greenleaf', receptionist: 'Aokie' }),
      now: () => T0,
    });
    await outreach.load();

    // Unstarted campaign must be ignored by both about() and stopWord()
    expect(outreach.about('+61411111111')).toBeNull();
    expect(outreach.stopWord('+61411111111', 'STOP')).toBe(false);
    expect(unstarted.people[0].state).toBe('awaiting_reply');
    expect(unstarted.people[0].outcome).toBeUndefined();
    expect(dncSaved).toEqual([]);
    expect(posted).toEqual([]);

    // Started campaign still works normally
    expect(outreach.about('+61422222222')?.c.id).toBe('out-started');
    expect(outreach.stopWord('+61422222222', 'STOP')).toBe(true);
    expect(started.people[0].outcome).toBe('opted_out');
    expect(dncSaved.length).toBeGreaterThanOrEqual(1);
  });
});

describe('[P2 #7 & #8] ChatPane.loadOlder() callOver/quiet preservation & redraw() rAF cancellation', () => {
  it('preserves live callOver and quiet state across loadOlder() even when older turns replay ended/started calls', () => {
    const fakeEl = () => ({
      isConnected: false,
      children: [] as unknown[],
      firstElementChild: null,
      lastElementChild: null,
      matches: () => false,
      prepend: () => {},
      after: () => {},
      remove: () => {},
      getBoundingClientRect: () => ({ top: 0, bottom: 0 }),
    });
    vi.stubGlobal('document', {
      createElement: () => fakeEl(),
    });
    vi.stubGlobal('HTMLElement', class {});

    const liveQuiet = { box: fakeEl(), text: 'live quiet', body: fakeEl() };
    let callOverDuringDraw: unknown = 'unset';
    let quietDuringDraw: unknown = 'unset';
    const fakePane = {
      hiddenTurns: [{ role: 'user', text: '[OAIY] 📞 The call ended.', automatic: true }] as Turn[],
      sink: null as unknown,
      current: { box: fakeEl(), text: 'streaming', body: fakeEl() },
      thinking: null,
      draft: null,
      quiet: liveQuiet as typeof liveQuiet | null,
      callOver: false as false | 'goodbye' | 'ended',
      currentPlan: null,
      planOpen: false,
      callName: 'Caller',
      way: 'call',
      older: fakeEl(),
      feed: fakeEl(),
      hero: fakeEl(),
      status: fakeEl(),
      follow: { stick: true },
      log: { scrollTop: 0, getBoundingClientRect: () => ({ top: 0 }) },
      topAnchor: () => null,
      drawTurns() {
        callOverDuringDraw = this.callOver;
        quietDuringDraw = this.quiet;
        // Simulate replaying an older ended call that clobbers callOver and clears quiet
        this.callOver = 'ended';
        this.quiet = null;
      },
      showPlan: () => {},
      showOlder: () => {},
      pin: () => {},
    };

    const loadOlder = (ChatPane.prototype as unknown as { loadOlder: () => void }).loadOlder;
    loadOlder.call(fakePane);
    expect(callOverDuringDraw).toBe(false);
    expect(quietDuringDraw).toBeNull();
    expect(fakePane.callOver).toBe(false);
    expect(fakePane.quiet).toBe(liveQuiet);

    // Reverse case: live callOver is 'ended', older turns contain a call start that sets callOver = false
    fakePane.callOver = 'ended';
    fakePane.hiddenTurns = [{ role: 'user', text: 'older call start', automatic: true }];
    fakePane.drawTurns = function () {
      this.callOver = false;
    };
    loadOlder.call(fakePane);
    expect(fakePane.callOver).toBe('ended');
  });

  it('tracks and cancels pending requestAnimationFrame in cancelRedraw() so finishReply() is never overwritten by a stale rAF', () => {
    vi.useFakeTimers();
    let nextRaf = 1;
    const rafQueue = new Map<number, FrameRequestCallback>();
    const cancelled: number[] = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      const id = nextRaf++;
      rafQueue.set(id, cb);
      return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => {
      cancelled.push(id);
      rafQueue.delete(id);
    });

    const box = { isConnected: true } as unknown as HTMLElement;
    const fakePane = {
      redraws: new Map<HTMLElement, () => void>(),
      redrawTimers: new Map<HTMLElement, ReturnType<typeof setTimeout>>(),
      redrawFrames: new Map<HTMLElement, number>(),
      scroll: vi.fn(),
    };
    const redraw = (ChatPane.prototype as unknown as { redraw: (k: HTMLElement, fn: () => void) => void }).redraw;
    const cancelRedraw = (ChatPane.prototype as unknown as { cancelRedraw: (k: HTMLElement) => void }).cancelRedraw;

    let rendered = '';
    redraw.call(fakePane, box, () => {
      rendered = 'stale-stream-markdown';
    });

    // Advance past STREAM_REDRAW_MS so setTimeout fires and queues the rAF (timer map is now empty, frame map has id 1)
    vi.advanceTimersByTime(100);
    expect(fakePane.redrawTimers.size).toBe(0);
    expect(fakePane.redrawFrames.size).toBe(1);
    expect(rafQueue.size).toBe(1);

    // finishReply() calls cancelRedraw(box) BEFORE the animation frame runs, then writes final HTML
    cancelRedraw.call(fakePane, box);
    rendered = 'final-closed-markdown';

    expect(cancelled).toEqual([1]);
    expect(fakePane.redrawFrames.size).toBe(0);
    expect(rafQueue.size).toBe(0);

    // Even if any remaining rAF callbacks flush, final render stays intact
    for (const cb of rafQueue.values()) cb(0);
    expect(rendered).toBe('final-closed-markdown');

    // Also verify clearLog() cancels pending redrawFrames
    const box2 = { isConnected: true } as unknown as HTMLElement;
    redraw.call(fakePane, box2, () => {
      rendered = 'should-not-run-after-clear';
    });
    vi.advanceTimersByTime(100);
    expect(fakePane.redrawFrames.size).toBe(1);
    const clearPane = {
      ...fakePane,
      meter: { hidden: false },
      openContext: () => {},
      contextNotes: [],
      prompts: [],
      contextUsed: null,
      lastUsage: null,
      lastSpeed: null,
      drawStats: () => {},
      contextPop: { replaceChildren: () => {} },
      lastSent: null,
      lastPrompt: null,
      hiddenTurns: [],
      way: null,
      taskRows: new Map(),
      currentPlan: null,
      planAll: false,
      setActivity: () => {},
      showPlan: () => {},
      media: [],
      feed: { replaceChildren: () => {}, append: () => {} },
      status: {},
      older: { remove: () => {} },
      follow: { jump: () => {} },
      showJump: () => {},
      cards: new Map(),
      current: null,
      thinking: null,
      quiet: null,
      draft: null,
      draftName: '',
      callOver: false,
      disposers: [],
      updateEmpty: () => {},
    };
    const clearLog = (ChatPane.prototype as unknown as { clearLog: () => void }).clearLog;
    clearLog.call(clearPane);
    expect(cancelled).toEqual([1, 2]);
    expect(clearPane.redrawFrames.size).toBe(0);
    expect(rafQueue.size).toBe(0);
  });
});

describe('[P2 #9] WorkspaceInstructions.reconcile() handles renamed identical instruction files', () => {
  it('emits a newly renamed instruction file with identical content to a deleted one, while still deduplicating co-existing duplicates', () => {
    const vfs = new Vfs();
    vfs.writeFile('/AGENTS.md', 'Always use strict types.');
    const wi = new WorkspaceInstructions();
    const baseline = wi.loadBaseline(vfs);
    expect(baseline).toContain('/AGENTS.md');
    expect(baseline).toContain('Always use strict types.');

    // Co-existing duplicate with identical content is still deduplicated
    vfs.writeFile('/OAIY.md', 'Always use strict types.');
    expect(wi.reconcile(vfs, ['OAIY.md'])).toBeNull();

    // Remove BOTH existing files and rename to /CLAUDE.md with the exact same content
    vfs.remove('/AGENTS.md');
    vfs.remove('/OAIY.md');
    vfs.writeFile('/CLAUDE.md', 'Always use strict types.');
    const delta = wi.reconcile(vfs, ['CLAUDE.md']);
    expect(delta).not.toBeNull();
    expect(delta).toContain('--- /CLAUDE.md ---');
    expect(delta).toContain('Always use strict types.');
    expect(delta).toContain('/AGENTS.md');
  });
});

describe('[P2 #10] editPhone pairing poll aborts cleanly on close/forget without committing', () => {
  it('forwards AbortSignal in pairingStatus() and startPairing()', async () => {
    const seenSignals: Array<AbortSignal | undefined> = [];
    vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
      seenSignals.push(init?.signal ?? undefined);
      return new Response(JSON.stringify({ pairingId: 'p1', code: '1234', status: 'pending' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    const ac = new AbortController();
    await startPairing('http://127.0.0.1:17972', 'test', ac.signal);
    await pairingStatus('http://127.0.0.1:17972', 'p1', ac.signal);
    expect(seenSignals).toEqual([ac.signal, ac.signal]);
  });

  it('aborts the 1.5s sleep and in-flight pairingStatus poll on dialog close/forget and never calls options.paired(desktop)', async () => {
    vi.useFakeTimers();

    // Minimal DOM + modal mock so editPhone() runs end-to-end in Node
    class FakeNode {
      textContent = '';
    }
    class FakeElement extends FakeNode {
      tagName: string;
      className = '';
      hidden = false;
      disabled = false;
      checked = false;
      value = '';
      onclick?: () => void;
      children: unknown[] = [];
      private listeners = new Map<string, Array<() => void>>();
      constructor(tag: string) {
        super();
        this.tagName = tag.toUpperCase();
      }
      append(...kids: unknown[]) {
        this.children.push(...kids);
      }
      setAttribute() {}
      addEventListener(type: string, fn: () => void) {
        const list = this.listeners.get(type) ?? [];
        list.push(fn);
        this.listeners.set(type, list);
      }
      click() {
        this.onclick?.();
        for (const fn of this.listeners.get('click') ?? []) fn();
      }
    }

    let pairButton: FakeElement | null = null;
    vi.stubGlobal('Node', FakeNode);
    vi.stubGlobal('HTMLElement', FakeElement);
    vi.stubGlobal('location', { host: 'localhost:5317' });
    vi.stubGlobal('document', {
      createElement: (tag: string) => new FakeElement(tag),
      createTextNode: (t: string) => Object.assign(new FakeNode(), { textContent: t }),
      body: { append() {} },
    });

    // Mock modal so we control exactly when the user closes the dialog
    let closeModal!: () => void;
    vi.doMock('../../src/ui/modal', () => ({
      askText: async () => null,
      modal: async (opts: { body: unknown[] }) => {
        const walk = (node: unknown) => {
          if (node instanceof FakeElement) {
            if (node.tagName === 'BUTTON' && node.children.some((c) => c === 'Pair with OAIY Desktop' || (c instanceof FakeNode && c.textContent === 'Pair with OAIY Desktop'))) {
              pairButton = node;
            }
            for (const c of node.children) walk(c);
          }
        };
        for (const b of opts.body) walk(b);
        await new Promise<void>((r) => (closeModal = r));
        return null;
      },
    }));

    let resolveStatusFetch!: (resp: Response) => void;
    let statusCalls = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) {
        return new Response(JSON.stringify({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: '1.0.0' }), { status: 200 });
      }
      if (url.endsWith('/api/bridge/pairing')) {
        return new Response(JSON.stringify({ pairingId: 'pair-1', code: '4242' }), { status: 200 });
      }
      if (url.includes('/api/bridge/pairing/pair-1')) {
        statusCalls++;
        return new Promise<Response>((resolve, reject) => {
          resolveStatusFetch = resolve;
          init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        });
      }
      return new Response('{}', { status: 404 });
    });

    const { editPhone } = await import('../../src/ui/phone');
    const pairedCalls: unknown[] = [];
    const dialogPromise = editPhone({
      desktop: null,
      phone: false,
      messages: DEFAULT_MESSAGE_SETTINGS,
      paired: (d) => pairedCalls.push(d),
      test: () => {},
    });

    // Wait for modal to mount and start pairing
    await vi.advanceTimersByTimeAsync(10);
    expect(pairButton).not.toBeNull();
    (pairButton as unknown as { click: () => void }).click();

    // Let startPairing resolve and the 1500ms sleep elapse so pairingStatus() is in flight
    await vi.advanceTimersByTimeAsync(1550);
    expect(statusCalls).toBe(1);

    // User closes the dialog while pairingStatus() is still in flight; even if server approves right after, must NOT commit!
    closeModal();
    await dialogPromise;
    resolveStatusFetch(new Response(JSON.stringify({ status: 'approved', token: 'late-token' }), { status: 200 }));
    await vi.advanceTimersByTimeAsync(2000);

    expect(pairedCalls).toEqual([]);
    expect(statusCalls).toBe(1);

    // Also verify closing mid-1500ms sleep aborts BEFORE pairingStatus is ever called!
    statusCalls = 0;
    pairButton = null;
    const dialogPromise2 = editPhone({
      desktop: null,
      phone: false,
      messages: DEFAULT_MESSAGE_SETTINGS,
      paired: (d) => pairedCalls.push(d),
      test: () => {},
    });
    await vi.advanceTimersByTimeAsync(10);
    expect(pairButton).not.toBeNull();
    (pairButton as unknown as { click: () => void }).click();
    // Advance only 500ms into the 1500ms sleep, then close
    await vi.advanceTimersByTimeAsync(500);
    closeModal();
    await dialogPromise2;
    await vi.advanceTimersByTimeAsync(3000);
    expect(statusCalls).toBe(0);
    expect(pairedCalls).toEqual([]);
    vi.doUnmock('../../src/ui/modal');
  });
});

describe('[P2 #11] OpenProject.close() unsubscribes before flush and clears/prevents retry timers', () => {
  it('unsubscribes before flush() and never leaks a retry timer after failed write on close()', async () => {
    vi.useFakeTimers();

    let writeAttempts = 0;
    let releaseWrite!: () => void;
    let writeStarted!: () => void;
    const waitWriteStarted = new Promise<void>((r) => (writeStarted = r));

    const failingDir = {
      async getDirectoryHandle() {
        return failingDir;
      },
      async removeEntry() {},
      async getFileHandle(seg: string) {
        return {
          async createWritable() {
            return {
              async write() {
                if (seg === 'fail.txt') {
                  writeAttempts++;
                  writeStarted();
                  await new Promise<void>((r) => (releaseWrite = r));
                  throw new Error('disk full');
                }
              },
              async close() {},
              async abort() {},
            };
          },
        };
      },
    };

    const OpenProjectCtor = OpenProject as unknown as new (meta: ProjectMeta, dir: FileSystemDirectoryHandle) => OpenProject;
    const project = new OpenProjectCtor(
      { id: 'p-close', name: 'CloseTest', created: 1, updated: 1 },
      failingDir as unknown as FileSystemDirectoryHandle,
    );
    const internals = project as unknown as {
      unsubscribe: (() => void) | null;
      queue: (c: { type: 'write'; path: string }) => void;
      timer: ReturnType<typeof setTimeout> | null;
    };
    internals.unsubscribe = project.vfs.onChange((change) => internals.queue(change as { type: 'write'; path: string }));

    // Trigger a failing write
    project.vfs.writeFile('/fail.txt', 'data');
    const closing = project.close();
    await waitWriteStarted;

    // A VFS mutation occurring while close()'s flush() is in flight MUST NOT queue a new save or timer
    project.vfs.writeFile('/during-close.txt', 'ignored-after-unsub');
    expect(internals.timer).toBeNull();

    // Let the failing write fail inside close()
    releaseWrite();
    await closing;

    // No retry timer may remain active after close(), and advancing timers must NOT retry write()
    expect(internals.timer).toBeNull();
    expect(writeAttempts).toBe(1);
    await vi.advanceTimersByTimeAsync(15_000);
    expect(writeAttempts).toBe(1);
  });
});
