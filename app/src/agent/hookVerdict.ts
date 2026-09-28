/**
 * What the answer of a flow that runs before one of the agent's tools means
 * for the call (see `ToolHook` in agent.ts).
 */
const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** What a "before" flow's answer means for the call. */
export interface BeforeVerdict {
  /** Why the call does not go ahead. */
  stop?: string;
  /** The call's input, changed (merged over what it was). */
  input?: Record<string, unknown>;
  /** Something for the agent to know, the call going ahead. */
  note?: string;
}

/**
 * A "before" flow says: nothing, or `ok` (go ahead); `STOP: why` (or JSON
 * `{"stop": "why"}`: the call does not run); JSON `{"input": {...}}` (go ahead
 * with these parameters changed, and `{"note": ...}` beside them); anything
 * else is a note for the agent, and the call goes ahead.
 */
export function beforeVerdict(output: string): BeforeVerdict {
  const said = output.trim();
  if (!said || /^(ok|okay|go|continue|yes|allow(ed)?|fine|null|""|true)\.?$/i.test(said)) return {};
  if (said.startsWith('{')) {
    try {
      const j = JSON.parse(said) as unknown;
      if (isRecord(j)) {
        const stop = j.stop ?? j.block;
        if (stop !== undefined && stop !== false && stop !== null) return { stop: typeof stop === 'string' && stop.trim() ? stop.trim() : 'the flow stopped it' };
        return {
          ...(isRecord(j.input) ? { input: j.input } : {}),
          ...(typeof j.note === 'string' && j.note.trim() ? { note: j.note.trim() } : {}),
        };
      }
    } catch {
      /* not JSON: read as words */
    }
  }
  const stop = /^(stop|block|no|deny|denied)\b[\s:.,-]*([\s\S]*)$/i.exec(said);
  if (stop) return { stop: stop[2].trim() || 'the flow stopped it' };
  return { note: said.length > 2000 ? `${said.slice(0, 2000)}…` : said };
}
