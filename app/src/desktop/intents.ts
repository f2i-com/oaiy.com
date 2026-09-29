/**
 * What OAIY Desktop may ask of this page by name: its setup wizard's "Answer
 * calls and texts with OAIY" (this page's settings live in its own storage,
 * which only it can change). The desktop evaluates `window.__oaiyIntent(name)`
 * into the Agent's page in its window, or, while the page is still starting,
 * leaves the name in `window.__OAIY_INTENTS__` for it to find (embed.rs).
 * Only the names given handlers here are done; anything else is ignored.
 */
import type { MessageSettings } from '../settings';

export type IntentName = 'answerWithOaiy';
export type IntentHandlers = Partial<Record<IntentName, () => void | Promise<void>>>;

export interface IntentWindow {
  __oaiyIntent?: (name: unknown) => boolean;
  __OAIY_INTENTS__?: unknown;
}

/** Answer the desktop's intents with `handlers`, starting with any that came before the page was ready. Returns a stop function. */
export function installIntents(handlers: IntentHandlers, w: IntentWindow = window as unknown as IntentWindow): () => void {
  const run = (name: unknown): boolean => {
    if (typeof name !== 'string' || !Object.prototype.hasOwnProperty.call(handlers, name)) return false;
    const handler = handlers[name as IntentName];
    if (!handler) return false;
    // Never throws into the desktop's eval; a handler reports its own failure.
    void Promise.resolve()
      .then(handler)
      .catch(() => undefined);
    return true;
  };
  w.__oaiyIntent = run;
  const waiting = Array.isArray(w.__OAIY_INTENTS__) ? (w.__OAIY_INTENTS__ as unknown[]) : [];
  w.__OAIY_INTENTS__ = [];
  for (const name of new Set(waiting)) run(name);
  return () => {
    if (w.__oaiyIntent === run) delete w.__oaiyIntent;
  };
}

/** The message settings with answering texts and calls turned on (the rest kept). */
export function answeringOn(messages: MessageSettings): MessageSettings {
  return { ...messages, answer: true, calls: true };
}
