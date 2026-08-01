/**
 * Connector module — runtime.
 *
 * One method per operation in the closed set. Every method receives the node's
 * input plus the `ConnectorCall` the compiler baked in (node id, node type,
 * operation, path template, the node's own data) — so the runtime never looks a
 * node type up anywhere, and therefore cannot know one.
 *
 * Three rules hold throughout:
 *
 *   1. **Real work or a real error.** No method returns a placeholder, an empty
 *      array, or its input when it cannot do the thing. It throws, naming the
 *      node and what was missing.
 *   2. **The credential never leaves as text.** It goes into an Authorization
 *      header and nowhere else — not into logs, not into error messages.
 *   3. **Every HTTP call goes through `ctx.secureFetch`**, which is the
 *      policy-enforced gateway (local-network permission, SSRF guard, DNS
 *      pinning). Module code must never touch a raw `fetch`.
 */

import type { RuntimeContext, RuntimeMethod, RuntimeModule } from 'oaiy-core/src/module-types';
import {
  DEFAULT_LOOPBACK_BASE_URL,
  type ConnectorCall,
  type ConnectorConfig,
  type ConnectorHostOptions,
} from './types';

/** The runtime module name; compiled scripts call `Connector.<operation>(…)`. */
export const CONNECTOR_RUNTIME_NAME = 'Connector';

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Coerce a value to a plain object when it plausibly is one (object, or JSON text). */
function asRecord(v: unknown): Record<string, unknown> | null {
  if (isRecord(v)) return v;
  if (typeof v === 'string') {
    const t = v.trim();
    if (t.startsWith('{')) {
      try {
        const parsed: unknown = JSON.parse(t);
        if (isRecord(parsed)) return parsed;
      } catch {
        /* not JSON — treat as a plain string */
      }
    }
  }
  return null;
}

function nonEmpty(v: unknown): boolean {
  return v !== undefined && v !== null && !(typeof v === 'string' && v.trim() === '');
}

/**
 * Whether a value actually SAYS something, as opposed to being a node's
 * untouched default.
 *
 * The json-typed properties (`body`, `payload`) default to `{}`, and every node
 * created in an editor carries that default in its data. Treating `{}` as
 * supplied would mean a node wired to an input silently sends an empty object to
 * the provider instead of the input — a request that succeeds and does nothing,
 * which is the exact failure mode this module is built to refuse.
 */
function hasContent(v: unknown): boolean {
  if (!nonEmpty(v)) return false;
  const rec = asRecord(v);
  if (rec) return Object.keys(rec).length > 0;
  if (Array.isArray(v)) return v.length > 0;
  return true;
}

/** A gateway knob that is 0 means "unset" — see the node's own description. */
function positiveNumber(v: unknown): number | undefined {
  if (!nonEmpty(v)) return undefined;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

/** Where a field may come from: the node's own data first, then the piped input. */
function field(call: ConnectorCall, input: unknown, ...names: string[]): unknown {
  for (const n of names) {
    if (nonEmpty(call.data[n])) return call.data[n];
  }
  const rec = asRecord(input);
  if (rec) {
    for (const n of names) {
      if (nonEmpty(rec[n])) return rec[n];
    }
  }
  return undefined;
}

function label(call: ConnectorCall): string {
  return `${call.nodeType} node "${call.nodeId}"`;
}

function fail(call: ConnectorCall, message: string): never {
  throw new Error(`${label(call)}: ${message}`);
}

/** Parse a value that may already be an object or may be JSON text. */
function objectArg(call: ConnectorCall, value: unknown, what: string): Record<string, unknown> {
  const rec = asRecord(value);
  if (!rec) {
    fail(call, `${what} must be an object (or JSON text for one), got ${typeof value}.`);
  }
  return rec;
}

/** Dot/bracket path lookup, used by the optional `rowsPath` escape hatch. */
function dig(source: unknown, path: string): unknown {
  let cur: unknown = source;
  for (const part of path.split(/[.[\]]+/).filter(Boolean)) {
    if (cur === null || cur === undefined) return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

/** FNV-1a — a tiny, dependency-free, stable digest for idempotency keys. */
function stableHash(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

function truncate(text: string, max = 400): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

// ---------------------------------------------------------------------------
// path placeholders
// ---------------------------------------------------------------------------

const PLACEHOLDER_RE = /\{([^}\s/]+)\}/g;

/**
 * Substitute `{placeholder}` segments from the node's own data (falling back to
 * an object input). A placeholder with no value is a hard error: the alternative
 * is calling the provider on a malformed path, which either 404s confusingly or,
 * worse, hits a different resource.
 */
export function resolvePath(call: ConnectorCall, input: unknown, template: string): string {
  return template.replace(PLACEHOLDER_RE, (_match, name: string) => {
    const value = field(call, input, name);
    if (!nonEmpty(value)) {
      fail(
        call,
        `the path ${JSON.stringify(template)} needs a value for {${name}}, and neither the node's ` +
          `"${name}" field nor its input supplied one.`,
      );
    }
    if (typeof value === 'object') {
      fail(call, `the path placeholder {${name}} needs a scalar value, got an object.`);
    }
    return encodeURIComponent(String(value));
  });
}

/**
 * The node's row filters, as the provider's query parameters.
 *
 * A filter the provider does not accept is REFUSED, never dropped. Dropping one
 * does not return nothing — it returns the unfiltered listing, so a caller
 * lookup reads a real record belonging to somebody else and greets the wrong
 * person by name.
 */
function filterQuery(call: ConnectorCall): string {
  const spec = call.filters;
  if (!spec) return '';
  const raw = call.data[spec.field];
  if (raw === undefined || raw === null || raw === '') return '';
  if (!Array.isArray(raw)) {
    fail(call, `the node's "${spec.field}" field must be a list of filters.`);
  }
  const parts: string[] = [];
  for (const row of raw as unknown[]) {
    if (!isRecord(row)) fail(call, `each entry of "${spec.field}" must be an object.`);
    const op = String(row[spec.opKey] ?? '');
    const field = String(row[spec.fieldKey] ?? '');
    const value = row[spec.valueKey];
    if (!field) fail(call, `a filter names no field to compare.`);
    const mapped = spec.ops.find((o) => o.op === op);
    if (!mapped) {
      const known = spec.ops.map((o) => o.op).join(', ');
      fail(
        call,
        `this connector cannot apply the filter ${JSON.stringify(op)} — it accepts ${known}. ` +
          `Applying the listing unfiltered would return the wrong records.`,
      );
    }
    // A filter whose value did not resolve would match everything. Refuse.
    if (value === undefined || value === null || value === '') {
      fail(call, `the filter on ${JSON.stringify(field)} has no value to compare against.`);
    }
    const param = mapped.param.replace('{field}', field);
    const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
    parts.push(`${encodeURIComponent(param)}=${encodeURIComponent(text)}`);
  }
  return parts.join('&');
}

/**
 * Hand a listing to the graph in the shape its author reads.
 *
 * Without a described shape this is the bare array, which is what a graph
 * iterating rows expects. A provider whose graphs read `.first` or `.found`
 * describes those names and gets them — reading `undefined` instead is silent,
 * and reads as "no such record".
 */
function shapeListing(call: ConnectorCall, rows: unknown): unknown {
  const spec = call.listResult;
  if (!spec) return rows;
  const list = Array.isArray(rows) ? rows : rows === undefined || rows === null ? [] : [rows];
  const out: Record<string, unknown> = { [spec.items]: list };
  if (spec.count) out[spec.count] = list.length;
  if (spec.first) out[spec.first] = list.length > 0 ? list[0] : null;
  if (spec.found) out[spec.found] = list.length > 0;
  return out;
}

/** Build a query string from an object/JSON-text `query` field. */
function queryString(call: ConnectorCall, input: unknown): string {
  const raw = field(call, input, 'query', 'params');
  if (raw === undefined) return '';
  const rec = objectArg(call, raw, 'the "query" field');
  const parts: string[] = [];
  for (const [k, v] of Object.entries(rec)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) {
      for (const item of v) parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(item))}`);
    } else if (typeof v === 'object') {
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(JSON.stringify(v))}`);
    } else {
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
    }
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

// ---------------------------------------------------------------------------
// row extraction
// ---------------------------------------------------------------------------

/**
 * Generic REST envelope keys only. A provider-specific key must NEVER be added
 * here — the single-array fallback below already unwraps `{ <anything>: [...] }`,
 * and `rowsPath` handles the rest, so there is no case that needs one.
 */
const COMMON_ROW_KEYS = ['items', 'data', 'results', 'rows', 'records', 'entries'];

/**
 * Return the provider's rows. The provider's envelope is its own business, so:
 * an explicit `rowsPath` on the node wins; otherwise a bare array is the rows, a
 * conventionally-named array property is the rows, and a body with exactly one
 * array property is the rows. Anything else comes back untouched rather than
 * being guessed at.
 */
export function extractRows(call: ConnectorCall, input: unknown, body: unknown): unknown {
  const rowsPath = field(call, input, 'rowsPath');
  if (nonEmpty(rowsPath)) {
    const picked = dig(body, String(rowsPath));
    if (picked === undefined) {
      fail(
        call,
        `rowsPath ${JSON.stringify(String(rowsPath))} matched nothing in the provider's response.`,
      );
    }
    return picked;
  }
  if (Array.isArray(body)) return body;
  if (isRecord(body)) {
    for (const key of COMMON_ROW_KEYS) {
      if (Array.isArray(body[key])) return body[key];
    }
    const arrays = Object.keys(body).filter((k) => Array.isArray(body[k]));
    if (arrays.length === 1) return body[arrays[0]];
  }
  return body;
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

interface HttpArgs {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
  call: ConnectorCall;
  /** What we were doing, for the error message. */
  what: string;
}

/**
 * One HTTP round trip through the policy gateway. Non-2xx throws — a connector
 * node that swallows a 4xx and returns `null` is exactly the silent-success
 * failure this whole design exists to prevent.
 */
async function http(ctx: RuntimeContext, args: HttpArgs): Promise<unknown> {
  const { url, method, headers, body, call, what } = args;
  let resp: Response;
  try {
    resp = await ctx.secureFetch(url, {
      method,
      headers,
      body,
      nodeId: call.nodeId,
      purpose: `${call.nodeType} ${call.operation}`,
    });
  } catch (e) {
    throw new Error(
      `${label(call)}: ${what} — ${method} ${url} could not be sent: ` +
        `${e instanceof Error ? e.message : String(e)}`,
    );
  }

  const text = await resp.text();
  if (!resp.ok) {
    fail(call, `${what} — ${method} ${url} returned HTTP ${resp.status}: ${truncate(text) || '(empty body)'}`);
  }
  if (!text.trim()) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // A provider that answers 200 with non-JSON is still a successful call; hand
    // the text back rather than inventing a failure.
    return text;
  }
}

// ---------------------------------------------------------------------------
// module factory
// ---------------------------------------------------------------------------

/**
 * Build the connector RuntimeModule for a validated config.
 *
 * Every one of the seven operations is implemented here; the config selects
 * which node types map to which. `createMethods(ctx)` returns fresh closures per
 * job (never a module-level context singleton), so concurrent runs stay isolated.
 */
export function createConnectorRuntime(
  config: ConnectorConfig,
  host: ConnectorHostOptions = {},
): RuntimeModule {
  const providerBase = config.baseUrl.replace(/\/+$/, '');
  const loopbackBase = (host.loopbackBaseUrl || DEFAULT_LOOPBACK_BASE_URL).replace(/\/+$/, '');
  const loopbackToken = host.loopbackToken || '';

  const providerHeaders = (withBody: boolean): Record<string, string> => {
    const h: Record<string, string> = {
      Accept: 'application/json',
      Authorization: `Bearer ${config.credential}`,
    };
    if (withBody) h['Content-Type'] = 'application/json';
    return h;
  };

  const loopbackHeaders = (withBody: boolean): Record<string, string> => {
    const h: Record<string, string> = { Accept: 'application/json' };
    if (withBody) h['Content-Type'] = 'application/json';
    // OAIY's privileged loopback routes (the connector relay, service control)
    // require the bearer on a headless box. Sending it on reads too is harmless.
    if (loopbackToken) h.Authorization = `Bearer ${loopbackToken}`;
    return h;
  };

  const createMethods = (ctx: RuntimeContext): Record<string, RuntimeMethod> => {
    const log = (message: string) => ctx.log?.('info', message);

    /**
     * The payload for a create/update: the node's `body` field, else the input.
     *
     * An EMPTY `body` is the field's default, not a decision — so the wired
     * input wins over it. Only when there is no usable input does an empty
     * declared body stand, which keeps "POST nothing on purpose" possible.
     */
    const writeBody = (call: ConnectorCall, input: unknown): Record<string, unknown> => {
      const declared = call.data.body ?? call.data.payload ?? call.data.record;
      if (hasContent(declared)) return objectArg(call, declared, 'the node\'s "body" field');
      const fromInput = asRecord(input);
      if (fromInput) return fromInput;
      if (nonEmpty(declared)) return objectArg(call, declared, 'the node\'s "body" field');
      fail(
        call,
        `${call.operation} needs an object to send. Wire an object into this node, or set its ` +
          `"body" field. (Received ${input === null || input === undefined ? 'nothing' : typeof input}.)`,
      );
    };

    const recordUrl = (call: ConnectorCall, input: unknown, withQuery: boolean): string => {
      if (!call.path) {
        // Unreachable via parseConnectorConfig, which requires a path for the
        // record operations — but a hand-built call must not silently GET the base.
        fail(call, `${call.operation} has no path configured for this node type.`);
      }
      return providerBase + resolvePath(call, input, call.path) + (withQuery ? queryString(call, input) : '');
    };

    const methods: Record<string, RuntimeMethod> = {
      // -- runInput ---------------------------------------------------------
      /**
       * The run's inputs object, as the graph's entry node. With a `key` field
       * set, that one input instead — and a missing key is an error, not an
       * undefined that quietly poisons everything downstream.
       */
      runInput: (inputs: unknown, call: ConnectorCall): unknown => {
        const source = asRecord(inputs) ?? {};
        const key = call.data.key ?? call.data.inputKey;
        if (nonEmpty(key)) {
          const name = String(key);
          if (!(name in source)) {
            fail(
              call,
              `this run has no input named ${JSON.stringify(name)} ` +
                `(it supplied: ${Object.keys(source).filter((k) => !k.startsWith('__')).join(', ') || 'nothing'}).`,
            );
          }
          return source[name];
        }
        // Shallow copy, minus the engine's own `__`-prefixed bookkeeping keys.
        // A copy matters: the live inputs object doubles as the workflow context,
        // so returning it directly would make the result self-referential.
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(source)) {
          if (!k.startsWith('__')) out[k] = v;
        }
        return out;
      },

      // -- chat -------------------------------------------------------------
      /**
       * A chat completion from THIS machine's own AI gateway (loopback) — not
       * from the provider, and not from a cloud endpoint the flow names.
       */
      chat: async (input: unknown, call: ConnectorCall): Promise<string> => {
        const provider = field(call, input, 'provider', 'providerId');
        const url = nonEmpty(provider)
          ? `${loopbackBase}/api/ai/providers/${encodeURIComponent(String(provider))}/v1/chat/completions`
          : `${loopbackBase}/api/ai/v1/chat/completions`;

        const declaredMessages = field(call, input, 'messages');
        let messages: unknown[];
        if (declaredMessages !== undefined) {
          if (!Array.isArray(declaredMessages)) {
            fail(call, '"messages" must be an array of {role, content}.');
          }
          messages = declaredMessages;
        } else {
          const system = call.data.systemPrompt ?? call.data.system;
          const prompt = call.data.prompt ?? call.data.userPrompt;
          const piped =
            input === null || input === undefined
              ? ''
              : typeof input === 'string'
                ? input
                : JSON.stringify(input);
          const user = [nonEmpty(prompt) ? String(prompt) : '', piped].filter(Boolean).join('\n\n');
          if (!user.trim()) {
            fail(
              call,
              'chat has nothing to send. Wire text into this node or set its "prompt" field.',
            );
          }
          messages = [
            ...(nonEmpty(system) ? [{ role: 'system', content: String(system) }] : []),
            { role: 'user', content: user },
          ];
        }

        const body: Record<string, unknown> = { messages, stream: false };
        const model = field(call, input, 'model');
        if (nonEmpty(model)) body.model = String(model);
        // Both default to 0 on the node, and both document 0 as "leave it to
        // the gateway". Sending the default literally would put `max_tokens: 0`
        // on every unedited chat node — a request that succeeds and comes back
        // with no text.
        const temperature = positiveNumber(field(call, input, 'temperature'));
        if (temperature !== undefined) body.temperature = temperature;
        const maxTokens = positiveNumber(field(call, input, 'maxTokens', 'max_tokens'));
        if (maxTokens !== undefined) body.max_tokens = maxTokens;

        const json = await http(ctx, {
          url,
          method: 'POST',
          headers: loopbackHeaders(true),
          body: JSON.stringify(body),
          call,
          what: 'chat completion from this machine\'s AI gateway',
        });

        const content = dig(json, 'choices.0.message.content');
        if (typeof content !== 'string') {
          fail(
            call,
            `the AI gateway answered without a message. Response: ${truncate(JSON.stringify(json) ?? 'null')}`,
          );
        }
        return content;
      },

      // -- listRecords ------------------------------------------------------
      listRecords: async (input: unknown, call: ConnectorCall): Promise<unknown> => {
        const base = recordUrl(call, input, true);
        const filters = filterQuery(call);
        const url = filters ? base + (base.includes('?') ? '&' : '?') + filters : base;
        log(`${label(call)}: GET ${url}`);
        const json = await http(ctx, {
          url,
          method: 'GET',
          headers: providerHeaders(false),
          call,
          what: 'listing records',
        });
        return shapeListing(call, extractRows(call, input, json));
      },

      // -- createRecord -----------------------------------------------------
      createRecord: async (input: unknown, call: ConnectorCall): Promise<unknown> => {
        const url = recordUrl(call, input, false);
        const payload = writeBody(call, input);
        log(`${label(call)}: ${call.method || 'POST'} ${url}`);
        return await http(ctx, {
          url,
          method: call.method || 'POST',
          headers: providerHeaders(true),
          body: JSON.stringify(payload),
          call,
          what: 'creating a record',
        });
      },

      // -- updateRecord -----------------------------------------------------
      updateRecord: async (input: unknown, call: ConnectorCall): Promise<unknown> => {
        const url = recordUrl(call, input, false);
        const payload = writeBody(call, input);
        log(`${label(call)}: ${call.method || 'PATCH'} ${url}`);
        return await http(ctx, {
          url,
          method: call.method || 'PATCH',
          headers: providerHeaders(true),
          body: JSON.stringify(payload),
          call,
          what: 'updating a record',
        });
      },

      // -- connectorRequest -------------------------------------------------
      /**
       * Relay a command to a plugin connector on THIS desktop. The relay is a
       * physical side effect, so it carries an idempotency key: an explicit one
       * from the node or the input if given, otherwise one derived from this
       * job + node + payload so a re-delivery of the same logical event is
       * recognised rather than performed twice.
       */
      connectorRequest: async (input: unknown, call: ConnectorCall): Promise<unknown> => {
        const connectorId = field(call, input, 'connector', 'connectorId', 'plugin', 'pluginId');
        if (!nonEmpty(connectorId)) {
          fail(call, 'connectorRequest needs a connector id — set the node\'s "connector" field.');
        }
        const command = field(call, input, 'command', 'method');
        if (!nonEmpty(command)) {
          fail(call, 'connectorRequest needs a command — set the node\'s "command" field.');
        }

        // Same rule as writeBody: an empty `payload` is the field's default, so
        // a wired input beats it. Relaying `{}` to a connector command because
        // the node was never edited is a side effect performed with no
        // arguments — it succeeds, and does the wrong thing.
        const declaredPayload = call.data.payload ?? call.data.body;
        let payload: Record<string, unknown>;
        if (hasContent(declaredPayload)) {
          payload = objectArg(call, declaredPayload, 'the node\'s "payload" field');
        } else {
          const fromInput = asRecord(input);
          if (fromInput) {
            payload = fromInput;
          } else if (nonEmpty(declaredPayload)) {
            payload = objectArg(call, declaredPayload, 'the node\'s "payload" field');
          } else {
            payload = input === null || input === undefined ? {} : { value: input };
          }
        }

        const explicitKey = field(call, input, 'idempotencyKey');
        const idempotencyKey = nonEmpty(explicitKey)
          ? String(explicitKey)
          : `${ctx.currentJobId || 'job'}:${call.nodeId}:${stableHash(
              `${String(connectorId)}|${String(command)}|${JSON.stringify(payload)}`,
            )}`;

        const url = `${loopbackBase}/api/bridge/connectors/${encodeURIComponent(String(connectorId))}/request`;
        log(`${label(call)}: relaying ${String(command)} to connector ${String(connectorId)}`);
        const json = await http(ctx, {
          url,
          method: 'POST',
          headers: loopbackHeaders(true),
          body: JSON.stringify({ command: String(command), payload, idempotencyKey }),
          call,
          what: `relaying ${String(command)} to connector ${String(connectorId)}`,
        });
        // The bridge answers { ok, result }. Unwrap the useful half.
        if (isRecord(json) && 'result' in json) return json.result;
        return json;
      },

      // -- serviceControl ---------------------------------------------------
      /** Read or control this desktop's services through its own loopback API. */
      serviceControl: async (input: unknown, call: ConnectorCall): Promise<unknown> => {
        const rawAction = field(call, input, 'action', 'operation') ?? 'list';
        const action = String(rawAction).toLowerCase();
        const serviceId = field(call, input, 'service', 'serviceId', 'id');

        const needsId = action !== 'list';
        if (needsId && !nonEmpty(serviceId)) {
          fail(call, `the "${action}" action needs a service id — set the node's "service" field.`);
        }
        const id = encodeURIComponent(String(serviceId ?? ''));

        switch (action) {
          case 'list':
          case 'status': {
            const json = await http(ctx, {
              url: `${loopbackBase}/api/services`,
              method: 'GET',
              headers: loopbackHeaders(false),
              call,
              what: 'reading this desktop\'s services',
            });
            const services = isRecord(json) && Array.isArray(json.services) ? json.services : json;
            if (action === 'list') return services;
            const wanted = String(serviceId);
            const found = Array.isArray(services)
              ? services.find((s) => isRecord(s) && s.id === wanted)
              : undefined;
            if (!found) {
              fail(call, `this desktop has no service ${JSON.stringify(wanted)}.`);
            }
            return found;
          }
          case 'start':
          case 'stop': {
            log(`${label(call)}: ${action} service ${String(serviceId)}`);
            const json = await http(ctx, {
              url: `${loopbackBase}/api/services/${id}/${action}`,
              method: 'POST',
              headers: loopbackHeaders(false),
              call,
              what: `${action}ing service ${String(serviceId)}`,
            });
            return json ?? { ok: true, id: String(serviceId), action };
          }
          case 'logs': {
            const tailRaw = field(call, input, 'tail');
            const tail = nonEmpty(tailRaw) ? Math.max(1, Number(tailRaw) || 50) : 50;
            return await http(ctx, {
              url: `${loopbackBase}/api/services/${id}/logs?tail=${tail}`,
              method: 'GET',
              headers: loopbackHeaders(false),
              call,
              what: `reading logs for service ${String(serviceId)}`,
            });
          }
          default:
            return fail(
              call,
              `"${action}" is not a service action this build can perform ` +
                `(supported: list, status, start, stop, logs).`,
            );
        }
      },
    };

    return methods;
  };

  return {
    name: CONNECTOR_RUNTIME_NAME,
    createMethods,
    // `createMethods` supersedes this; the field is required by RuntimeModule and
    // the loader's shape check, so it stays an empty object rather than a set of
    // shared, context-free closures that could be called by accident.
    methods: {},
  };
}
