/**
 * What the preview and SoftN's frame say to each other past the first `formlogic:init`, and how it reads in the
 * preview's bar. There is no DOM here: preview.ts wires this to the frame, and the unit tests to a stand-in frame
 * that speaks the same protocol.
 *
 * The frame's `formlogic:ready` announces what it does beyond drawing an app once:
 *   shown: 1    it posts `{ type: 'shown' }` on its port when the app's first page is drawn (SoftN v0.0.34 on);
 *   update: 1   it takes a newer version of the app it has open, in place, once it has posted `shown`:
 *                 the parent posts  { type: 'formlogic:update', id, client, assets? }
 *                 the frame answers { type: 'updated', id, outcome, changed?, reason?, carried?, problem? }
 *               `id` is the parent's own (a number here), echoed on the answer. `client` is the WHOLE set of the app's
 *               text files (path to text), as in `init`. `assets` is the WHOLE set of its media (path to base64), as
 *               in `init`; left out, the media is as the parent last sent it (in `init` or an update), which is what
 *               this side does whenever the media has not changed, since every file sent is read whole; `{}` is an
 *               app with no media. `outcome` is one of UPDATE_OUTCOMES.
 * The frame applies updates one at a time and has no deadline of its own: this side keeps one (UPDATE_DEADLINE_MS),
 * and a change the frame cannot see (OAIY's theme, another project) is a new frame without asking. SoftN's own
 * description is docs/engineering/FORMLOGIC_INTEGRATION.md, "A newer version of an app, in the frame it runs in".
 * A frame that announces neither is opened afresh for every change and counted as drawn after a wait, as before.
 */

/** What a frame said it does, in its `formlogic:ready`. */
export interface FrameSays {
  /** It says when the app's first page is drawn. */
  shown: boolean;
  /** It takes a newer version of the open app in place. */
  update: boolean;
}

export function frameSays(ready: unknown): FrameSays {
  const said = (ready && typeof ready === 'object' ? ready : {}) as { shown?: unknown; update?: unknown };
  return { shown: said.shown === 1, update: said.update === 1 };
}

/**
 * What became of an update:
 *   updated    drawn again where the app stood (its values, timers and open page as they were);
 *   restarted  started again (`reason` says why; `carried`, whether its values were carried into the new start);
 *   kept       the newest version has a problem (it does not parse or start, its manifest is not JSON or names no
 *              main page, its files do not compose): the last one that worked is still on screen and running, and
 *              `problem` says what is wrong. The frame says a parse or start problem in a line over the app; the
 *              others are in the answer alone, so the preview shows `problem` in its banner every time;
 *   unchanged  nothing differed;
 *   reopen     the frame cannot take this version (`reason` says why: another main page, other permissions, Python
 *              come or gone, no app open in it): open a new frame with these files, as for every change before.
 */
export const UPDATE_OUTCOMES = ['updated', 'restarted', 'kept', 'unchanged', 'reopen'] as const;
export type UpdateOutcome = (typeof UPDATE_OUTCOMES)[number];

export interface UpdateAnswer {
  outcome: UpdateOutcome;
  /** What differed, as the frame names it (`page`, `styles`, `media`, `theme`, `logic`, …): read as "something changed", never checked against a list. */
  changed?: string[];
  reason?: string;
  carried?: boolean;
  problem?: string;
}

/** How long the frame has to answer an update before a new frame is opened instead. */
export const UPDATE_DEADLINE_MS = 8000;

/** The frame's answer to an update, read from a message on its port; null for any other message. */
export function readUpdateAnswer(data: unknown): (UpdateAnswer & { id: number }) | null {
  const d = (data && typeof data === 'object' ? data : {}) as { type?: unknown; id?: unknown; outcome?: unknown; changed?: unknown; reason?: unknown; carried?: unknown; problem?: unknown };
  if (d.type !== 'updated' || typeof d.id !== 'number') return null;
  // An outcome this page does not know is one it cannot show the result of: a new frame, which it can.
  const outcome = (UPDATE_OUTCOMES as readonly unknown[]).includes(d.outcome) ? (d.outcome as UpdateOutcome) : 'reopen';
  const answer: UpdateAnswer & { id: number } = { id: d.id, outcome };
  if (Array.isArray(d.changed)) answer.changed = d.changed.filter((c): c is string => typeof c === 'string').slice(0, 200);
  if (typeof d.reason === 'string' && d.reason.trim()) answer.reason = d.reason.trim().slice(0, 300);
  if (typeof d.carried === 'boolean') answer.carried = d.carried;
  if (typeof d.problem === 'string' && d.problem.trim()) answer.problem = d.problem.trim().slice(0, 2000);
  return answer;
}

/** Whether two sets of media are the same files with the same bytes (each file's base64, compared whole). */
export function sameAssets(a: Record<string, string>, b: Record<string, string>): boolean {
  if (a === b) return true;
  const paths = Object.keys(a);
  if (paths.length !== Object.keys(b).length) return false;
  return paths.every((path) => Object.hasOwn(b, path) && a[path] === b[path]);
}

/** The port's side this needs: posting to the frame. */
export interface FramePort {
  postMessage(message: unknown): void;
}

/** One open app frame: its port, what it announced, and the updates it has been sent. */
export class LiveFrame {
  /** The app's first page is drawn (the frame said so, or the wait for one that does not say has passed). */
  drawn = false;
  /** The app did not open in this frame (an error before its first page): nothing here to update. */
  failed = false;
  private closed = false;
  private nextId = 1;
  private readonly waiting = new Map<number, (answer: UpdateAnswer | null) => void>();

  /** The media as it was last sent to this frame, in `init` or an update: what the frame has when an update leaves `assets` out. */
  private sentAssets: Record<string, string>;

  constructor(
    /** The app folder this frame was opened with. */
    readonly root: string,
    private readonly port: FramePort,
    readonly says: FrameSays,
    /** The media `init` carried. */
    assets: Record<string, string> = {},
  ) {
    this.sentAssets = assets;
  }

  /** Whether a newer version of the app in `root` can be sent to this frame, rather than a new frame opened. */
  takes(root: string): boolean {
    return this.says.update && !this.closed && !this.failed && this.drawn && root === this.root;
  }

  /**
   * Send the app's files as they are now and wait for what the frame made of them. `client` is every text file;
   * `assets` is all of its media as it is now, which is sent (whole) only where it differs from what this frame was
   * last sent, and left out where it does not: the frame then keeps what it has, and reads nothing twice. Null when
   * the frame did not answer within `deadlineMs`, when the message could not be posted, or when this frame is
   * closed meanwhile: the caller opens a new frame.
   */
  update(client: Record<string, string>, assets: Record<string, string>, deadlineMs = UPDATE_DEADLINE_MS): Promise<UpdateAnswer | null> {
    if (this.closed) return Promise.resolve(null);
    const id = this.nextId++;
    return new Promise((resolve) => {
      const finish = (answer: UpdateAnswer | null) => {
        clearTimeout(timer);
        this.waiting.delete(id);
        resolve(answer);
      };
      const timer = setTimeout(() => finish(null), deadlineMs);
      this.waiting.set(id, finish);
      try {
        if (sameAssets(assets, this.sentAssets)) this.port.postMessage({ type: 'formlogic:update', id, client });
        else {
          this.port.postMessage({ type: 'formlogic:update', id, client, assets });
          // (what the frame has from here on, whatever it answers: media sent with a version it keeps back comes
          // with the next version that leaves its media out)
          this.sentAssets = assets;
        }
      } catch {
        finish(null);
      }
    });
  }

  /** A message from the frame's port: true when it was an update's answer (one no longer waited for is dropped). */
  take(data: unknown): boolean {
    const answer = readUpdateAnswer(data);
    if (!answer) return false;
    const { id, ...rest } = answer;
    this.waiting.get(id)?.(rest);
    return true;
  }

  /** This frame is being replaced: updates still waiting are answered as unanswered. */
  close(): void {
    this.closed = true;
    for (const finish of [...this.waiting.values()]) finish(null);
  }
}

/** How an update's answer reads in the preview's bar, and whether the app as it is now counts as working. */
export interface UpdateSaid {
  status: string;
  kind: 'ok' | 'error';
  /** More, for the status's tooltip. */
  title: string;
  ok: boolean;
}

/** Short, for the bar (the banner under it says "Kept the last working version" and why); "error" first, as every status of an app that is not working. */
export const KEPT_STATUS = 'error · last version kept';
export const KEPT_TITLE = 'The newest version does not start, so the preview still shows the last one that worked. The banner says what is wrong.';

/** `time` is the clock time to show (the caller's, so a test can fix it). Not for `reopen`, which shows as a new frame does. */
export function updateSaid(answer: UpdateAnswer, time: string): UpdateSaid {
  switch (answer.outcome) {
    case 'updated':
      return { status: `live · updated ${time}`, kind: 'ok', title: 'The app was drawn again where it stood: what you had typed and the page you were on are as they were.', ok: true };
    case 'restarted': {
      // The time is in the bar, as for "updated": two restarts in a row said the same words, and
      // nothing showed that the second had happened. Why, and what became of its values, is the
      // tooltip's: the runtime's own sentence is too long for the bar.
      const carried = answer.carried === true ? 'Its values were carried over.' : answer.carried === false ? 'It started from the beginning: its values could not be carried over.' : '';
      return {
        status: `live · started again ${time}`,
        kind: 'ok',
        title: [answer.reason || 'Its logic changed.', answer.reason && /carried/i.test(answer.reason) ? '' : carried].filter(Boolean).join(' '),
        ok: true,
      };
    }
    case 'unchanged':
      return { status: `live · ${time}`, kind: 'ok', title: 'Nothing the app draws or runs changed.', ok: true };
    case 'kept':
      return { status: KEPT_STATUS, kind: 'error', title: KEPT_TITLE, ok: false };
    default:
      return { status: 'rendering…', kind: 'ok', title: '', ok: false };
  }
}

/** The files an app's messages name: logic, pages, data and the manifest. */
const FILE_LEAD = /^((?:[\w.@-]+\/)*[\w.@-]+\.(?:js|logic|py|ui|html?|css|json|xdb)):\s*/i;

/**
 * A problem as the preview's banner says it: the runtime's own prefixes dropped, the part a person can act on
 * first, and the file before it when it is known ("main.js: a brace, bracket or string opened before it is never
 * closed."). The whole message stays in the banner's tooltip and is what "Fix with agent" sends.
 */
export function plainProblem(message: string, file?: string): string {
  let text = (message.split('\n')[0] ?? '').trim();
  const lead = FILE_LEAD.exec(text);
  if (lead) {
    file = lead[1];
    text = text.slice(lead[0].length);
  }
  text = text
    .replace(/^\[SoftN\]\s*/i, '')
    .replace(/^Error loading script:\s*/i, '')
    .replace(/^syntax error:\s*/i, '')
    .replace(/^SyntaxError:\s*/, '');
  // "expected a property name, at the end of the logic: a brace, bracket or string opened before it is never closed"
  const end = /^.*?\bat the end of the (?:logic|script|file):\s*(.+)$/i.exec(text);
  if (end) text = end[1];
  text = text.trim();
  if (!text) text = 'something went wrong';
  if (!/[.!?…)]$/.test(text)) text += '.';
  // (after a file's name the sentence goes on; alone it starts one)
  return file ? `${file}: ${text}` : `${text[0].toUpperCase()}${text.slice(1)}`;
}

/** Whether a message is the logic failing to compile as it was loaded (so the file the syntax check names is where). */
export function isLoadSyntaxError(message: string): boolean {
  return /Error loading script:\s*SyntaxError|^\s*(?:\[SoftN\]\s*)?SyntaxError:/i.test(message.split('\n')[0] ?? '');
}

/** Under "Fit pane", a frame this wide or narrower lays a SoftN app out as on a phone (a Sidebar is a Menu button; at 640 a wide Table is cards). */
export const PHONE_LAYOUT_MAX = 768;

/** The note beside the size picker when the pane itself is phone-sized; '' when it is not. */
export function fitNote(width: number): string {
  const w = Math.round(width);
  return w > 0 && w <= PHONE_LAYOUT_MAX ? `${w} px wide: phone layout` : '';
}

const releaseOf = (zipp: { release?: unknown; version?: unknown } | undefined): string => {
  if (typeof zipp?.release === 'string' && zipp.release) return zipp.release;
  if (typeof zipp?.version === 'string' && zipp.version) return `v${zipp.version.replace(/^v/, '')}`;
  return 'another release';
};

/**
 * SoftN's runtime binds the engine it is given with the bindings of the ZIPP release it was built with, and carries
 * no engine of its own (from v0.0.35). So the Agent's engine must be that release's: where the runtime's manifest
 * names another module, this is what to tell the person; null where they are the same, or where it cannot be told.
 */
export function engineMismatch(manifest: unknown, own: unknown): string | null {
  const m = (manifest && typeof manifest === 'object' ? manifest : {}) as { zipp?: { sha256?: unknown; release?: unknown; version?: unknown } };
  const o = (own && typeof own === 'object' ? own : {}) as { release?: unknown; files?: Record<string, unknown> };
  const wanted = m.zipp?.sha256;
  const have = o.files?.['zipp_wasm_bg.wasm'];
  if (typeof wanted !== 'string' || typeof have !== 'string' || wanted === have) return null;
  return `SoftN's runtime was built with ZIPP ${releaseOf(m.zipp)}; this Agent has ${releaseOf({ release: o.release })}. Run npm run fetch:zipp and npm run fetch:softn.`;
}
