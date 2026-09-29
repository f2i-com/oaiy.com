/**
 * Plugins' actions as the agent's tools. A plugin on OAIY Desktop may offer
 * some of its own service-definition actions to the agent (`agentTools` in its
 * manifest); the desktop resolves each to a name, a description and an input
 * schema, and lists them in its modules snapshot (`contributions.agent.tools`).
 * Here each becomes one of the agent's tools, run through the desktop's gated
 * route for actions (`/api/services/actions/:definition/:action/invoke`).
 *
 * Where a tool is offered is its `audience`: the project conversation, the
 * Front desk's runner, or the Front desk's own conversations (texts, calls,
 * flows' tasks). A tool whose action changes something (its `sideEffects` is
 * not `none`) asks the person first, with its `confirm` template filled in,
 * in the conversations the person is at (the project's, and the runner's).
 * In a conversation nobody is at (a text, a call, a flow's task) it runs only
 * when its audience names that conversation explicitly.
 */
import type { SessionTool } from '../agent/agent';
import type { ToolSpec } from '../agent/protocol';
import type { Desktop } from './bridge';

/** Where a plugin tool may be offered. */
export type PluginToolAudience = 'project' | 'runner' | 'session:sms' | 'session:call' | 'session:task';

export const PLUGIN_TOOL_AUDIENCES: readonly PluginToolAudience[] = ['project', 'runner', 'session:sms', 'session:call', 'session:task'];

/** The conversations the person is at: a tool that changes something asks them first. */
const ATTENDED: ReadonlySet<PluginToolAudience> = new Set(['project', 'runner']);

/** A plugin's action offered as a tool, as the desktop lists it. */
export interface PluginTool {
  pluginId: string;
  /** What the agent calls it. */
  name: string;
  /** `<definition>/<actionId>`, as declared. */
  action: string;
  /** The service definition's id (`aokie.phone`). */
  definition: string;
  /** The action's id (`sms.threads`). */
  actionId: string;
  description: string;
  inputSchema: Record<string, unknown>;
  /** The action's; absent when its definition does not say (so it counts as changing something). */
  sideEffects?: string;
  audience: PluginToolAudience[];
  /** What the person is asked before it runs (`Call {number}`). */
  confirm?: string;
  timeoutMs?: number;
}

/** What running a plugin tool needs. */
export interface PluginToolDeps {
  desktop: () => Desktop | null;
  /** Ask the person; true when they allow it. */
  approve: (request: { title: string; message: string; ok: string }) => Promise<boolean>;
  /** The names already in use (a clash gets the plugin's id in front). */
  taken: Set<string>;
}

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const NAME = /^[a-z][a-z0-9_]{2,47}$/;
const MAX_RESULT = 8000;

/** The plugin tools in a snapshot's `contributions` (malformed entries are dropped). */
export function parsePluginTools(contributions: unknown): PluginTool[] {
  const agent = isRecord(contributions) && isRecord(contributions.agent) ? contributions.agent : null;
  const list = agent && Array.isArray(agent.tools) ? agent.tools : [];
  const out: PluginTool[] = [];
  for (const t of list) {
    if (!isRecord(t)) continue;
    const { pluginId, name, action, definition, actionId, description } = t;
    if (typeof pluginId !== 'string' || !pluginId || typeof name !== 'string' || !NAME.test(name)) continue;
    if (typeof definition !== 'string' || !definition || typeof actionId !== 'string' || !actionId) continue;
    const audience = Array.isArray(t.audience)
      ? [...new Set(t.audience.filter((a): a is PluginToolAudience => (PLUGIN_TOOL_AUDIENCES as readonly unknown[]).includes(a)))]
      : [];
    if (!audience.length) continue;
    if (out.some((x) => x.name === name)) continue;
    const effects = typeof t.sideEffects === 'string' ? t.sideEffects : undefined;
    const confirm = typeof t.confirm === 'string' && t.confirm.trim() ? t.confirm : undefined;
    // One that changes something with nothing to ask the person: not offered.
    if (effects !== 'none' && !confirm) continue;
    out.push({
      pluginId,
      name,
      action: typeof action === 'string' && action ? action : `${definition}/${actionId}`,
      definition,
      actionId,
      description: typeof description === 'string' && description.trim() ? description.trim() : actionId,
      inputSchema: isRecord(t.inputSchema) ? t.inputSchema : { type: 'object' },
      ...(effects !== undefined ? { sideEffects: effects } : {}),
      audience,
      ...(confirm ? { confirm } : {}),
      ...(typeof t.timeoutMs === 'number' && t.timeoutMs > 0 ? { timeoutMs: t.timeoutMs } : {}),
    });
  }
  return out;
}

/** Does running it change anything? */
export function changesSomething(tool: PluginTool): boolean {
  return tool.sideEffects !== 'none';
}

/** `{key}` in `template` becomes the input's value, as text (a key the input lacks stays as it is). */
export function fillTemplate(template: string, input: Record<string, unknown>): string {
  return template.replace(/\{([A-Za-z0-9_.-]+)\}/g, (whole, key: string) => {
    if (!(key in input) || input[key] === undefined || input[key] === null) return whole;
    const value = input[key];
    return typeof value === 'string' ? value : JSON.stringify(value);
  });
}

/** The tool as the model sees it. */
export function pluginToolSpec(tool: PluginTool, taken: Set<string>): ToolSpec {
  const prefix = tool.pluginId.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'plugin';
  return {
    name: taken.has(tool.name) ? `${prefix}_${tool.name}`.slice(0, 64) : tool.name,
    description: `${tool.description} (From the ${tool.pluginId} plugin, on OAIY Desktop.)`,
    parameters: { ...tool.inputSchema, type: 'object' },
  };
}

/** What the action answered, for the model: text as is, anything else as JSON, at most 8,000 characters. */
export function actionOutcome(value: unknown): string {
  const out = typeof value === 'string' ? value : JSON.stringify(value ?? null, null, 1);
  return out.length > MAX_RESULT ? `${out.slice(0, MAX_RESULT)}\n[cut at 8,000 characters]` : out;
}

/** `signal` and a timeout, as one signal (and a way to let the timer go). */
function withTimeout(signal: AbortSignal | undefined, ms: number | undefined): { signal: AbortSignal | undefined; done: () => void } {
  if (!ms) return { signal, done: () => {} };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`the action took longer than ${Math.round(ms / 1000)} s`)), ms);
  const onAbort = () => controller.abort(signal?.reason);
  if (signal?.aborted) controller.abort(signal.reason);
  else signal?.addEventListener('abort', onAbort, { once: true });
  return {
    signal: controller.signal,
    done: () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    },
  };
}

/**
 * What running `tool` in `where` takes: nothing (`run`), the person's leave
 * (`ask`: it changes something, and they are at this conversation), or it
 * may not run here (`refuse`: its audience does not name this conversation,
 * and nobody is here to allow something that changes things).
 */
export function mayRun(tool: PluginTool, where: PluginToolAudience): 'run' | 'ask' | 'refuse' {
  if (!tool.audience.includes(where)) return 'refuse';
  if (!changesSomething(tool)) return 'run';
  return ATTENDED.has(where) ? 'ask' : 'run';
}

/** The plugin tools offered in `where`, as the agent's tools. */
export function pluginSessionTools(tools: PluginTool[], where: PluginToolAudience, deps: PluginToolDeps): SessionTool[] {
  return tools
    .filter((tool) => tool.audience.includes(where))
    .map((tool) => ({
      spec: pluginToolSpec(tool, deps.taken),
      run: async (input, signal) => {
        const how = mayRun(tool, where);
        if (how === 'refuse') throw new Error(`${tool.name} is not offered in this conversation, so it cannot run here`);
        if (how === 'ask') {
          const message = fillTemplate(tool.confirm ?? `Run ${tool.action}`, input);
          const allowed = await deps.approve({ title: `Allow ${tool.name}?`, message, ok: 'Allow' });
          if (!allowed) return `The person declined, so ${tool.name} did not run (${message}). Do not try it again unless they ask.`;
        }
        const d = deps.desktop();
        if (!d) throw new Error(`OAIY Desktop is not connected, so ${tool.name} cannot run`);
        const key = `oaiy-app:${tool.pluginId}:${tool.name}:${crypto.randomUUID()}`;
        const limit = withTimeout(signal, tool.timeoutMs);
        try {
          return actionOutcome(await d.invokeAction(tool.definition, tool.actionId, input, key, limit.signal));
        } finally {
          limit.done();
        }
      },
    }));
}
