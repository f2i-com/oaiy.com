/**
 * What OAIY Desktop's dashboard asks the editor to do in OAIY's window (the desktop's `flows_intent`, embed.rs): open
 * one flow, Run history's "Open the editor" on a run, which opened the editor on whatever flow was open before.
 *
 * Asked through `window.__oaiyFlowsIntent(intent, flowId)`; one asked while the page was still starting waits in
 * `window.__OAIY_FLOWS_INTENTS__`, and is taken as the editor starts listening. Only a flow id (letters, digits, `-`
 * and `_`, as the desktop keeps flows) is taken. Outside OAIY's window nothing listens.
 */
import { inOaiyWindow } from './oaiyWindow';

type IntentWindow = Window & {
  __oaiyFlowsIntent?: (intent: unknown, flowId: unknown) => void;
  __OAIY_FLOWS_INTENTS__?: unknown;
};

const FLOW_ID = /^[A-Za-z0-9_-]{1,128}$/;

/** Hear the dashboard's "open this flow" (`open` is handed its id); returns the way to stop. */
export function listenForFlowsIntents(open: (flowId: string) => void): () => void {
  if (!inOaiyWindow()) return () => {};
  const w = window as IntentWindow;
  const take = (intent: unknown, flowId: unknown) => {
    if (intent === 'openFlow' && typeof flowId === 'string' && FLOW_ID.test(flowId)) open(flowId);
  };
  w.__oaiyFlowsIntent = take;
  // (asked before the editor listened: taken now, oldest first)
  const waiting = Array.isArray(w.__OAIY_FLOWS_INTENTS__) ? (w.__OAIY_FLOWS_INTENTS__ as unknown[]) : [];
  delete w.__OAIY_FLOWS_INTENTS__;
  for (const asked of waiting) if (Array.isArray(asked)) take(asked[0], asked[1]);
  return () => {
    if (w.__oaiyFlowsIntent === take) delete w.__oaiyFlowsIntent;
  };
}
