/*
 * The shared record and the three other shapes a provider has (design 4.1). Each adapter is a pure function pair, and
 * web/tests/unit/adapters.test.mjs round-trips every pair. No adapter carries a key inside a record: a key travels beside
 * it, as a separate argument or result, so nothing built from a record can hold one by accident.
 *
 *   - the Agent's `ProviderConfig`            (recordFromAgentConfig / agentConfigFromRecord)
 *   - the flow editor's service               (flowsServiceFromRecord / providerRefOfEndpoint)
 *   - the desktop or server's gateway record  (recordFromGatewayProvider / gatewayInputFromRecord)
 *
 * A conversion that cannot be exact says so by returning null (a base address with a query string, a base the gateway
 * cannot use); it never guesses. The same goes for a record that would not be allowed: every record an adapter makes is checked by
 * `validateRecord`, the check the add form's record passes, so an adapter is not a way round it (an Agent provider of type `custom` at
 * `http://192.168.1.5/v1` is a service on the internet over plain http as far as a record can tell, and is null, not saved).
 */
import { EXTRA_HEADER_NAMES, type ExtraHeader, type ProviderCap, type ProviderConfig, type ProviderRecord, type ProviderType } from './types';
import { defaultBaseUrl, normalizeApiBase } from './endpoints';
import { isLocalAddress, isLoopbackHost } from './errors';
import { validateRecord } from './records';

/** The Agent's kind of provider for a record: what the Agent's own code (its messages, its headers) takes it for. */
export function providerTypeOf(record: Pick<ProviderRecord, 'dialect' | 'kind' | 'preset'>): ProviderType {
  if (record.dialect === 'anthropic') return 'anthropic';
  if (record.preset === 'openai') return 'openai';
  // An Agent `custom` provider on this network is a record of a server on this network (recordFromAgentConfig), and is still `custom`.
  if (record.preset === 'custom') return 'custom';
  if (record.kind === 'local-server') return 'local';
  return 'custom';
}

// --- The Agent's ProviderConfig ------------------------------------------------

/** A record from an Agent provider, and its key beside it. Null when the address cannot be a record's (it has a query string, or is not http(s)). */
export function recordFromAgentConfig(config: ProviderConfig): { record: ProviderRecord; apiKey: string } | null {
  const raw = config.baseUrl?.trim() || defaultBaseUrl(config.type, config.serverKind);
  const baseUrl = normalizeApiBase(raw);
  if (baseUrl === null) return null;
  const anthropic = config.type === 'anthropic';
  const extraHeaders: ExtraHeader[] = [];
  // The Agent sends OpenAI-Organization for `openai` only (providerHeaders).
  if (config.type === 'openai' && config.orgId?.trim()) extraHeaders.push({ name: 'OpenAI-Organization', value: config.orgId.trim() });
  const caps: ProviderCap[] = ['chat'];
  // A provider that is not `local` at a plain-http address on this network (an Agent `custom` provider at http://192.168.1.5:8000/v1, a
  // machine that serves a model to the house) is a server on this network, not a service on the internet: the record says so, so that
  // the rule for a service on the internet (https, because a key over plain http can be read on the way) is not broken by calling it
  // one. A plain-http address anywhere else stays a service on the internet, and `validateRecord` refuses it.
  const url = new URL(baseUrl);
  const onThisNetwork = url.protocol === 'http:' && !isLoopbackHost(url.hostname) && isLocalAddress(baseUrl);
  const record: ProviderRecord = {
    v: 1,
    id: config.id,
    name: config.name,
    dialect: anthropic ? 'anthropic' : 'openai',
    baseUrl,
    auth: anthropic ? 'x-api-key' : 'bearer',
    caps,
    kind: config.type === 'local' || onThisNetwork ? 'local-server' : 'external',
    preset: config.type === 'local' ? 'local-server' : config.type,
    via: 'broker',
  };
  if (extraHeaders.length > 0) record.extraHeaders = extraHeaders;
  const model = config.modelId?.trim();
  if (model) record.model = model;
  if (config.type === 'local' && config.serverKind) record.serverKind = config.serverKind;
  if (config.contextTokens !== undefined) record.contextTokens = config.contextTokens;
  if (config.parallelAgents !== undefined) record.parallelAgents = config.parallelAgents;
  const checked = validated(record);
  return checked === null ? null : { record: checked, apiKey: config.apiKey };
}

/**
 * The record as the record check makes it, or null when the check refuses it. `via` is the adapter's own (the check makes records the
 * holder keeps; a gateway's is a mirror), and is put back after.
 */
function validated(record: ProviderRecord): ProviderRecord | null {
  const result = validateRecord(record, record.id);
  return result.ok ? { ...result.record, via: record.via } : null;
}

/**
 * The Agent's provider for a record and its key. `base` is the Agent's own config the record came from, if there is one: what
 * a record has no field for (`followEngine`, `detectedContext`) is kept from it, and an address or model that only differs
 * from the record's in how it is written is kept as it was written.
 */
export function agentConfigFromRecord(record: ProviderRecord, apiKey: string, base?: Partial<ProviderConfig>): ProviderConfig {
  const type = providerTypeOf(record);
  const config: ProviderConfig = { ...base, id: record.id, type, name: record.name, apiKey };
  // What the config held is kept as it was written where the record says the same thing.
  const raw = base?.baseUrl?.trim() || defaultBaseUrl(type, record.serverKind);
  if (!(base !== undefined && normalizeApiBase(raw) === record.baseUrl)) config.baseUrl = record.baseUrl;
  if ((base?.modelId?.trim() || undefined) !== record.model) {
    if (record.model !== undefined) config.modelId = record.model;
    else delete config.modelId;
  }
  if (type === 'local') {
    if (record.serverKind !== undefined) config.serverKind = record.serverKind;
    else delete config.serverKind;
  } else delete config.serverKind;
  const org = record.extraHeaders?.find((h) => h.name.toLowerCase() === 'openai-organization');
  if (type === 'openai') {
    if (org) config.orgId = org.value;
    else if (config.orgId?.trim()) delete config.orgId; // the record no longer has the one the config had
  } else delete config.orgId;
  if (record.contextTokens !== undefined) config.contextTokens = record.contextTokens;
  else delete config.contextTokens;
  if (record.parallelAgents !== undefined) config.parallelAgents = record.parallelAgents;
  else delete config.parallelAgents;
  return config;
}

// --- The flow editor's service -------------------------------------------------

/**
 * A flow editor service (`CustomService`, platform/ui bundled-modules/core-service/examples.ts) for a record: the shape a
 * typed AI node and Service Call already run. It names the record in its endpoint (`oaiy-provider://<id>/chat/completions`),
 * and has NO `apiKeyConstant`: the flow never holds the key, the providers origin attaches it.
 */
export interface FlowsProviderService {
  id: string;
  name: string;
  description: string;
  endpoint: string;
  method: 'POST';
  headers: '{}';
  bodyTemplate: string;
  responseType: 'json';
  responsePath: string;
  nodeTypes: Array<'ai_llm' | 'service_call'>;
  model?: string;
  apiFormat: 'openai' | 'anthropic';
  group: 'custom';
}

/** The scheme a flow's provider service uses in place of an address. */
export const PROVIDER_SCHEME = 'oaiy-provider://';

/** The flow editor's service id for a record. */
export function flowsServiceId(recordId: string): string {
  return `provider:${recordId}`;
}

export function endpointForRecord(record: Pick<ProviderRecord, 'id' | 'dialect'>): string {
  return `${PROVIDER_SCHEME}${encodeURIComponent(record.id)}${record.dialect === 'anthropic' ? '/messages' : '/chat/completions'}`;
}

const OPENAI_CHAT_BODY = '{\n  "model": {{model}},\n  "messages": [{"role": "system", "content": {{system}}}, {"role": "user", "content": {{input}}}]\n}';
const ANTHROPIC_CHAT_BODY = '{\n  "model": {{model}},\n  "max_tokens": 4096,\n  "system": {{system}},\n  "messages": [{"role": "user", "content": {{input}}}]\n}';

export function flowsServiceFromRecord(record: ProviderRecord): FlowsProviderService {
  const anthropic = record.dialect === 'anthropic';
  const service: FlowsProviderService = {
    id: flowsServiceId(record.id),
    name: record.name,
    description: `Chat through the provider “${record.name}” (its key stays in the providers origin, never in the flow).`,
    endpoint: endpointForRecord(record),
    method: 'POST',
    headers: '{}',
    bodyTemplate: anthropic ? ANTHROPIC_CHAT_BODY : OPENAI_CHAT_BODY,
    responseType: 'json',
    responsePath: anthropic ? 'content.0.text' : 'choices.0.message.content',
    nodeTypes: ['ai_llm', 'service_call'],
    apiFormat: record.dialect,
    group: 'custom',
  };
  if (record.model) service.model = record.model;
  return service;
}

/** The record a service endpoint names and the path it asks for, or null when the endpoint is not `oaiy-provider://<id>/<path>`. */
export function providerRefOfEndpoint(endpoint: unknown): { id: string; path: string } | null {
  if (typeof endpoint !== 'string' || !endpoint.startsWith(PROVIDER_SCHEME)) return null;
  const rest = endpoint.slice(PROVIDER_SCHEME.length);
  const slash = rest.indexOf('/');
  if (slash <= 0) return null;
  let id: string;
  try {
    id = decodeURIComponent(rest.slice(0, slash));
  } catch {
    return null;
  }
  const path = rest.slice(slash);
  return id ? { id, path } : null;
}

// --- The desktop or server's gateway record --------------------------------------

/** What `GET /api/ai/providers` says of a provider (platform/desktop/src-tauri/src/ai/providers.rs, `AiProviderPublic`): everything but the key. */
export interface GatewayProviderPublic {
  id: string;
  name: string;
  category?: string | null;
  protocol: 'openai' | 'anthropic';
  baseUrl: string;
  model?: string | null;
  capabilities: Array<'chat' | 'transcription' | 'speech' | 'embeddings' | 'realtime'>;
  enabled: boolean;
  allowLocal: boolean;
  hasKey: boolean;
}

/** What the gateway is sent to save a provider (`AiProviderInput`): it takes no key, that is set apart. */
export interface GatewayProviderInput {
  id: string;
  name: string;
  category?: string;
  protocol: 'openai' | 'anthropic';
  baseUrl: string;
  model?: string;
  capabilities: Array<'chat' | 'transcription' | 'speech' | 'embeddings' | 'realtime'>;
  enabled: boolean;
  allowLocal: boolean;
}

const GATEWAY_CAP: Record<GatewayProviderPublic['capabilities'][number], ProviderCap | null> = {
  chat: 'chat',
  transcription: 'transcription',
  speech: 'speech',
  embeddings: 'embeddings',
  realtime: null,
};

/**
 * A read-only mirror of a gateway provider (`via: 'gateway'`). The gateway takes a SERVER base and appends `/v1/…` itself
 * (gateway.rs `chat_path`), so the record's API base is the gateway's base and `/v1`; until the gateway stops appending
 * (WA-14) a base that already names a version other than `/v1` cannot be called through it, and this returns null.
 */
export function recordFromGatewayProvider(p: GatewayProviderPublic): ProviderRecord | null {
  const server = p.baseUrl.trim().replace(/\/+$/, '');
  const baseUrl = normalizeApiBase(`${server}/v1`);
  if (baseUrl === null || server === '') return null;
  const caps: ProviderCap[] = p.capabilities.length === 0 ? ['chat'] : p.capabilities.map((c) => GATEWAY_CAP[c]).filter((c): c is ProviderCap => c !== null);
  const record: ProviderRecord = {
    v: 1,
    id: p.id,
    name: p.name,
    dialect: p.protocol,
    baseUrl,
    auth: p.protocol === 'anthropic' ? 'x-api-key' : 'bearer',
    caps,
    kind: p.allowLocal ? 'local-server' : 'external',
    via: 'gateway',
  };
  if (p.model) record.model = p.model;
  if (p.category) record.preset = p.category;
  return validated(record);
}

/** The gateway's own input for a record whose base the gateway can use (`<server>/v1`); null for one it cannot (Gemini's `/v1beta/openai`). */
export function gatewayInputFromRecord(record: ProviderRecord, enabled = true): GatewayProviderInput | null {
  if (!record.baseUrl.endsWith('/v1')) return null;
  const capabilities = record.caps.filter((c): c is 'chat' | 'transcription' | 'speech' | 'embeddings' => c === 'chat' || c === 'transcription' || c === 'speech' || c === 'embeddings');
  const input: GatewayProviderInput = {
    id: record.id,
    name: record.name,
    protocol: record.dialect,
    baseUrl: record.baseUrl.slice(0, -'/v1'.length),
    capabilities,
    enabled,
    allowLocal: record.kind === 'local-server',
  };
  if (record.model) input.model = record.model;
  if (record.preset) input.category = record.preset;
  return input;
}

/** Whether a header name may be carried in a record's `extraHeaders`. */
export function isExtraHeaderName(name: string): boolean {
  return EXTRA_HEADER_NAMES.includes(name.toLowerCase());
}
