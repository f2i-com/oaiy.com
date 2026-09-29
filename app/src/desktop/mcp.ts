/**
 * OAIY Desktop's control API: its MCP server (`POST /api/mcp`, JSON-RPC 2.0,
 * stateless), through which the Agent checks and changes OAIY itself: the
 * engines and their models, the AI sources, services, plugins and their
 * setup and settings, flows, the calendar's settings, the FormLogic link and
 * first-run setup.
 *
 * Every request says who is asking (`X-OAIY-Session`), and the desktop offers
 * tools by it: all of them to the person's own conversations (a project, and
 * "Set up OAIY"), the read tools to the Front desk's runner, and none to a
 * call, a text thread or a flow's task (a caller must not be able to talk the
 * receptionist into reconfiguring OAIY). The app keeps to the same rule on
 * its side: it never asks for those three, and gives the runner read tools
 * only, whatever a desktop lists.
 *
 * The model never holds the desktop's token: it reaches the desktop only
 * through these tools, each a `tools/call` made here.
 */
import type { SessionTool } from '../agent/agent';
import type { ToolSpec } from '../agent/protocol';

/** Who is asking, as `X-OAIY-Session` names it: the kind of conversation. */
export type ControlSession = 'project' | 'setup' | 'runner' | 'call' | 'sms' | 'task';

/** The conversations offered nothing: nobody the person trusts is at them. */
export const UNOFFERED: ReadonlySet<ControlSession> = new Set<ControlSession>(['call', 'sms', 'task']);

/** The protocol version asked for (the desktop answers with its own when it does not speak it). */
export const MCP_PROTOCOL_VERSION = '2025-06-18';

/** One of the desktop's tools, as it lists it. */
export interface ControlTool {
  name: string;
  title?: string;
  description: string;
  inputSchema: Record<string, unknown>;
  /** It only reads (`readOnlyHint`). */
  readOnly: boolean;
  /** It removes something (`destructiveHint`). */
  destructive: boolean;
}

/** What a tool answered: its text for the model, whether it failed, and its data for the page. */
export interface ControlResult {
  text: string;
  isError: boolean;
  structured?: unknown;
}

export class ControlError extends Error {
  constructor(message: string, readonly status = 0, readonly code?: number) {
    super(message);
    this.name = 'ControlError';
  }
}

type Send = (input: string, init: RequestInit) => Promise<Response>;

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** The tools in a `tools/list` result (malformed entries are dropped). */
export function parseTools(list: unknown): ControlTool[] {
  const out: ControlTool[] = [];
  for (const t of Array.isArray(list) ? list : []) {
    if (!isRecord(t) || typeof t.name !== 'string' || !/^[A-Za-z0-9_.-]{1,64}$/.test(t.name) || out.some((x) => x.name === t.name)) continue;
    const notes = isRecord(t.annotations) ? t.annotations : {};
    out.push({
      name: t.name,
      ...(typeof t.title === 'string' && t.title ? { title: t.title } : {}),
      description: typeof t.description === 'string' ? t.description.trim() : '',
      inputSchema: isRecord(t.inputSchema) ? t.inputSchema : { type: 'object' },
      readOnly: notes.readOnlyHint === true,
      destructive: notes.destructiveHint === true,
    });
  }
  return out;
}

/** A JSON-RPC answer's body: the object itself, or (from a server that streams) the last `data:` line's. */
function readBody(text: string, contentType: string): unknown {
  if (/event-stream/i.test(contentType) || /^(event|data):/m.test(text.slice(0, 20))) {
    const data = text.split(/\r?\n/).filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).filter(Boolean);
    for (const line of data.reverse()) {
      try {
        const parsed = JSON.parse(line) as unknown;
        if (isRecord(parsed) && ('result' in parsed || 'error' in parsed)) return parsed;
      } catch {
        /* a keep-alive or a comment */
      }
    }
    return null;
  }
  return text ? JSON.parse(text) : null;
}

/**
 * The control API of one desktop: started once (`initialize`, then
 * `notifications/initialized`), its tools listed once for each kind of
 * conversation (until `refresh`), and its tools called.
 */
export class ControlClient {
  private seq = 0;
  private started: Promise<void> | null = null;
  private readonly lists = new Map<ControlSession, Promise<ControlTool[]>>();
  private readonly known = new Map<ControlSession, ControlTool[]>();
  private readonly send: Send;
  /** The protocol version the desktop answered with. */
  protocol = '';

  constructor(readonly origin: string, private readonly token: string, send?: Send) {
    this.send = send ?? ((input, init) => fetch(input, init));
  }

  /** The tools last listed for `session` (none before they have been). */
  listed(session: ControlSession): ControlTool[] {
    return UNOFFERED.has(session) ? [] : (this.known.get(session) ?? []);
  }

  private async post(session: ControlSession, message: Record<string, unknown>, signal?: AbortSignal): Promise<Response> {
    let resp: Response;
    try {
      resp = await this.send(`${this.origin}/api/mcp`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${this.token}`,
          'content-type': 'application/json',
          accept: 'application/json, text/event-stream',
          'x-oaiy-session': session,
        },
        body: JSON.stringify(message),
        signal,
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      throw new ControlError(`OAIY Desktop did not answer (${(error as Error).message})`);
    }
    if (resp.status === 401 || resp.status === 403) throw new ControlError('OAIY Desktop refused the Agent\'s request to its control API: pair this page with it again', resp.status);
    if (resp.status === 404 || resp.status === 405) throw new ControlError('This OAIY Desktop has no control API for the Agent: update OAIY', resp.status);
    return resp;
  }

  private async request(session: ControlSession, method: string, params: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const id = ++this.seq;
    const resp = await this.post(session, { jsonrpc: '2.0', id, method, params }, signal);
    const text = await resp.text();
    let body: unknown;
    try {
      body = readBody(text, resp.headers.get('content-type') ?? '');
    } catch {
      throw new ControlError(`OAIY Desktop's control API answered ${method} with something that is not JSON (HTTP ${resp.status})`, resp.status);
    }
    if (isRecord(body) && isRecord(body.error)) {
      const code = typeof body.error.code === 'number' ? body.error.code : undefined;
      throw new ControlError(String(body.error.message ?? `${method} failed`), resp.status, code);
    }
    if (!resp.ok || !isRecord(body)) throw new ControlError(`OAIY Desktop's control API did not answer ${method} (HTTP ${resp.status})`, resp.status);
    return isRecord(body.result) ? body.result : {};
  }

  /** Start talking with the desktop, once (tried again after a failure). */
  start(session: ControlSession = 'project'): Promise<void> {
    if (!this.started) {
      const started = (async () => {
        const signal = AbortSignal.timeout(15_000);
        const result = await this.request(session, 'initialize', {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: { name: 'oaiy-agent', title: 'OAIY Agent', version: '0.1.0' },
        }, signal);
        this.protocol = typeof result.protocolVersion === 'string' ? result.protocolVersion : MCP_PROTOCOL_VERSION;
        // A notification: answered 202, with nothing in it.
        await this.post(session, { jsonrpc: '2.0', method: 'notifications/initialized' }, signal);
      })();
      this.started = started;
      started.catch(() => {
        if (this.started === started) this.started = null;
      });
    }
    return this.started;
  }

  /** The tools offered to `session` (asked once, until `refresh`). A call, a text or a flow's task gets none, without asking. */
  tools(session: ControlSession): Promise<ControlTool[]> {
    if (UNOFFERED.has(session)) return Promise.resolve([]);
    let list = this.lists.get(session);
    if (!list) {
      list = (async () => {
        await this.start(session);
        const tools: ControlTool[] = [];
        let cursor: string | undefined;
        for (let page = 0; page < 10; page++) {
          const result = await this.request(session, 'tools/list', cursor ? { cursor } : {}, AbortSignal.timeout(15_000));
          tools.push(...parseTools(result.tools).filter((t) => !tools.some((x) => x.name === t.name)));
          cursor = typeof result.nextCursor === 'string' && result.nextCursor ? result.nextCursor : undefined;
          if (!cursor) break;
        }
        // The runner reads, and never changes OAIY, whatever a desktop offers it.
        const offered = session === 'runner' ? tools.filter((t) => t.readOnly) : tools;
        this.known.set(session, offered);
        return offered;
      })();
      this.lists.set(session, list);
      const asked = list;
      asked.catch(() => {
        if (this.lists.get(session) === asked) this.lists.delete(session);
      });
    }
    return list;
  }

  /** Ask for the lists again next time (the desktop's modules changed, or it came back). What was listed stays until then. */
  refresh(): void {
    this.lists.clear();
  }

  /** Call a tool as `session`. A tool that failed is an answer (`isError`), for the model to read. */
  async call(session: ControlSession, name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<ControlResult> {
    if (UNOFFERED.has(session)) throw new ControlError(`OAIY's control tools are not offered in ${session} conversations`);
    await this.start(session);
    const result = await this.request(session, 'tools/call', { name, arguments: args }, signal);
    const content = Array.isArray(result.content) ? result.content.filter(isRecord) : [];
    const text = content.filter((c) => c.type === 'text' && typeof c.text === 'string').map((c) => String(c.text)).join('\n');
    return { text, isError: result.isError === true, ...(result.structuredContent !== undefined ? { structured: result.structuredContent } : {}) };
  }
}

/**
 * The desktop's tools that the app's own already do, better: left out of a
 * conversation that has the app's (the flow builder checks a flow against the
 * node types and lays it out as the flow editor shows it). By the app's tool
 * that stands in for each. `flow_delete` has no such tool, and stays.
 */
export const COVERED: Readonly<Record<string, string>> = {
  flows_list: 'flow_list',
  flow_get: 'flow_read',
  flow_create: 'flow_write',
  flow_update: 'flow_write',
  flow_run: 'flow_run',
};

/** The most of a tool's answer the model reads. */
const MAX_RESULT = 12_000;

/** What a conversation's control tools need. */
export interface ControlToolDeps {
  /** The desktop's control API now (null: no desktop). */
  client: () => ControlClient | null;
  /** The names the conversation's other tools go by: the app's own come first. */
  taken: ReadonlySet<string>;
  /** A change tool ran and did what it was asked (by its name on the desktop). */
  changed?: (tool: string) => void;
}

/** A tool's text with the names of the tools left out swapped for the app's that stand in for them. */
function renamed(text: string, swaps: Map<string, string>): string {
  let out = text;
  for (const [from, to] of swaps) out = out.replace(new RegExp(`\\b${from}\\b`, 'g'), to);
  return out;
}

/** The tool as the model sees it, under `name`. */
export function controlToolSpec(tool: ControlTool, name: string, swaps: Map<string, string> = new Map()): ToolSpec {
  const schema = swaps.size ? (JSON.parse(renamed(JSON.stringify(tool.inputSchema), swaps)) as Record<string, unknown>) : tool.inputSchema;
  return {
    name,
    description: renamed(tool.description || tool.title || name, swaps),
    parameters: { ...schema, type: 'object' },
  };
}

/** A tool's answer for the model: its text (or its data), at most 12,000 characters. */
export function controlOutcome(result: ControlResult): string {
  const out = result.text || (result.structured !== undefined ? JSON.stringify(result.structured) : '');
  return out.length > MAX_RESULT ? `${out.slice(0, MAX_RESULT)}\n[cut at 12,000 characters]` : out;
}

/**
 * The desktop's tools offered to a conversation of kind `session`, as the
 * agent's tools: none for a call, a text thread or a flow's task; read tools
 * only for the runner. A tool the app's own already does (`COVERED`) is left
 * out when the app's is there; one whose name another of the conversation's
 * tools has gets `oaiy_` in front.
 */
export function controlSessionTools(tools: ControlTool[], session: ControlSession, deps: ControlToolDeps): SessionTool[] {
  if (UNOFFERED.has(session)) return [];
  const offered = session === 'runner' ? tools.filter((t) => t.readOnly) : tools;
  const swaps = new Map<string, string>();
  for (const t of offered) if (COVERED[t.name] && deps.taken.has(COVERED[t.name])) swaps.set(t.name, COVERED[t.name]);
  const used = new Set(deps.taken);
  const out: SessionTool[] = [];
  for (const tool of offered) {
    if (swaps.has(tool.name)) continue;
    let name = tool.name;
    if (used.has(name)) name = `oaiy_${tool.name}`.slice(0, 64);
    if (used.has(name)) continue;
    used.add(name);
    out.push({
      spec: controlToolSpec(tool, name, swaps),
      run: async (input, signal) => {
        const client = deps.client();
        if (!client) throw new Error('OAIY Desktop is not connected, so OAIY cannot be checked or changed from here');
        const result = await client.call(session, tool.name, input, signal);
        const text = controlOutcome(result);
        // A tool that failed is read by the model as an error, and says what to do about it.
        if (result.isError) throw new Error(text || `${tool.name} failed`);
        if (!tool.readOnly) deps.changed?.(tool.name);
        return text || 'Done.';
      },
    });
  }
  return out;
}

/**
 * A conversation's tools with OAIY's control tools after them, as its kind is
 * offered them (`controlSessionTools`, from the desktop's lists as `client`
 * last read them). The app's own tools come first and keep their names
 * (`builtIns`: the agent's built-in tools, which are not among `own`), and no
 * name reaches the model twice. With none listed for this kind (no desktop, an
 * older one, a call, a text or a flow's task) `own` comes back as it is.
 */
export function withControlTools(own: SessionTool[], session: ControlSession, client: () => ControlClient | null, builtIns: Iterable<string>, changed?: (tool: string) => void): SessionTool[] {
  const listed = client()?.listed(session) ?? [];
  if (!listed.length) return own;
  const taken = new Set([...builtIns, ...own.map((t) => t.spec.name)]);
  return distinctTools([...own, ...controlSessionTools(listed, session, { client, taken, changed })]);
}

/** Tools with names already in use dropped (the first of each name stays): no model is sent two tools of one name. */
export function distinctTools(tools: SessionTool[], taken: ReadonlySet<string> = new Set()): SessionTool[] {
  const seen = new Set(taken);
  return tools.filter((t) => {
    if (seen.has(t.spec.name)) return false;
    seen.add(t.spec.name);
    return true;
  });
}
