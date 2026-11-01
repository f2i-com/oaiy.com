/**
 * Project I/O Utilities
 *
 * Handles project import/export with secret sanitization.
 * Extracted from useProject.ts for maintainability.
 */

import type {
  OAIYProject,
  ProjectConstant,
  WorkflowGraph,
  LogEntry,
} from 'oaiy-core';
import { getModuleLoader } from 'oaiy-core';
import { triggerDownloadFromBlob } from '../lib/download';
import { withoutRunOutputs } from './runOutputs';
import { withoutProviderSeeds } from './NodeDefaults';
import {
  defaultLLMEndpoints,
  defaultImageGenEndpoints,
  defaultHttpPresets,
  defaultConstants,
  defaultSettings,
  createEmptyProject,
} from './ProjectDefaults';
import {
  listAllServices,
  saveService,
  type CustomService,
} from './serviceRegistry';

/**
 * Shape of the exported project JSON. Strict superset of OAIYProject —
 * `services` is the web build's flow-portability bag: every CustomService
 * referenced by a `service_call` node in any of the project's flows is
 * snapshotted here so importing the project on another machine (or in
 * another browser profile that has no `oaiy.customServices` entries
 * yet) can re-register them with one click on load.
 */
export interface ExportedProject extends OAIYProject {
  /** Custom services referenced by service_call nodes — re-registered on
   * import if not already in localStorage. */
  services?: CustomService[];
}

/**
 * Keys by a shape of their own, whether they are the whole of a value or inside one: Groq, Hugging Face, Replicate, xAI,
 * Perplexity, Stripe, GitLab, npm, AWS and a JWT. With a length to them, so an id that begins the same way stays.
 */
const KEY_SHAPES = [
  /\bgsk_[A-Za-z0-9]{20,}/,
  /\bhf_[A-Za-z0-9]{20,}/,
  /\br8_[A-Za-z0-9]{20,}/,
  /\bxai-[A-Za-z0-9]{20,}/,
  /\bpplx-[A-Za-z0-9]{20,}/,
  /\bsk_live_[A-Za-z0-9]{16,}/,
  /\bglpat-[A-Za-z0-9_-]{20,}/,
  /\bnpm_[A-Za-z0-9]{20,}/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/,
];
const WHOLE_KEY_SHAPES = KEY_SHAPES.map((shape) => new RegExp(`^(?:${shape.source})$`));
const KEY_SHAPES_IN_TEXT = KEY_SHAPES.map((shape) => new RegExp(shape.source, 'g'));

/** Whether what follows `Basic` decodes to a `user:password`, as a credential does and the rest of "Basic information" does not. */
const isBasicCredential = (encoded: string): boolean => {
  try {
    return /^[\x20-\x7e]*:[\x20-\x7e]*$/.test(atob(encoded));
  } catch {
    return false;
  }
};

/**
 * Check if a value looks like a secret (API key, token, etc.)
 */
export const looksLikeSecret = (value: unknown): boolean => {
  if (typeof value !== 'string') return false;
  // Common API key patterns
  const secretPatterns = [
    /^sk-[A-Za-z0-9]{20,}$/,           // OpenAI keys
    /^sk-ant-[A-Za-z0-9-]{20,}$/,      // Anthropic keys
    /^AIza[A-Za-z0-9_-]{35}$/,         // Google API keys
    /^ghp_[A-Za-z0-9]{36}$/,           // GitHub tokens
    /^gho_[A-Za-z0-9]{36}$/,           // GitHub OAuth tokens
    /^xoxb-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{24}$/,  // Slack bot tokens
    /^Bearer\s+[A-Za-z0-9._-]{20,}$/i, // Bearer tokens
    /^[A-Za-z0-9]{32,}$/,              // Generic long alphanumeric (potential keys)
    ...WHOLE_KEY_SHAPES,
  ];
  if (secretPatterns.some(pattern => pattern.test(value))) return true;
  const basic = /^Basic\s+([A-Za-z0-9+/]{8,}={0,2})$/i.exec(value);
  return !!basic && isBasicCredential(basic[1]);
};

/**
 * A name with one of these in it goes from an export, in any case and at any depth. For a node with no `secret` type to
 * go by (a package's that is not loaded, a flow from elsewhere), and for what is inside a node's object or list, the name
 * is all there is. Whole names let `botToken` and `smtpPassword` through.
 */
const SENSITIVE_NAME_PARTS = ['password', 'token', 'secret', 'key', 'credential', 'auth'];

const sensitiveName = (name: string): boolean => {
  const lower = name.toLowerCase();
  return SENSITIVE_NAME_PARTS.some(part => lower.includes(part));
};

/**
 * A field that should name a constant or a variable and holds something else: a key pasted there, of whatever shape. A
 * constant's name is as Settings makes it (`OPENAI_API_KEY`), and no longer than the runtime takes a name to be
 * (`resolveApiKey`).
 */
const holdsLiteralKey = (field: string, value: string): boolean =>
  /(Constant|EnvVar)$/.test(field) && !/^[A-Z0-9_]{0,40}$/.test(value);

/**
 * A node's own field that only sounds like a secret, and went from every export with no word said: a count of tokens
 * (`maxTokens`), the name of a constant (`apiKeyConstant`, which an imported flow needs to find its key, and which
 * `holdsLiteralKey` holds to being one), and a field called `key` on a node whose schema says what it is (a memory's).
 * Each of these is a string or a number. An object or a list under such a name is what the name says it is.
 */
const onlySoundsSecret = (field: string, value: unknown, schemaPlainFields?: Set<string>): boolean => {
  if (typeof value !== 'string' && typeof value !== 'number') return false;
  return (
    (/tokens$/i.test(field) && /^\d+$/.test(String(value))) ||
    field.endsWith('Constant') ||
    (field === 'key' && schemaPlainFields?.has(field) === true)
  );
};

/**
 * The fields that hold a request as text: where it goes, its headers, its body (`endpoint`, `customHeaders`,
 * `customBodyTemplate`). Only these are read for the names in them. A prompt or a block's code that says `"key": "x"`
 * says it as its own words.
 */
const REQUEST_TEXT = /(endpoints?|urls?|headers?|body|bodytemplate)$/i;

/**
 * A list inside a node, at any depth: the strings by what they look like, and the objects as a node's objects. A secret
 * leaves its place in the list marked, so what comes after it stays where it was.
 */
const sanitizeList = (
  list: unknown[],
  field: string,
  knownSecrets: Set<string>,
  schemaSecretFields?: Set<string>
): unknown[] =>
  list.map((item) => {
    if (typeof item === 'string') {
      if (looksLikeSecret(item) || knownSecrets.has(item)) return REDACTED_TOKEN;
      return REQUEST_TEXT.test(field) ? redactRequestText(item, knownSecrets) : redactSecretsInText(item, knownSecrets);
    }
    if (Array.isArray(item)) return sanitizeList(item, field, knownSecrets, schemaSecretFields);
    if (item && typeof item === 'object') {
      return sanitizeNodeData(item as Record<string, unknown>, knownSecrets, schemaSecretFields, undefined, true);
    }
    return item;
  });

/**
 * Sanitize node data by removing sensitive fields and secret-looking values.
 * Accepts schema-defined secret fields from node definitions, and the fields the
 * schema declares as anything else. `nested`: this is an object inside a node.
 */
export const sanitizeNodeData = (
  data: Record<string, unknown>,
  knownSecrets: Set<string>,
  schemaSecretFields?: Set<string>,
  schemaPlainFields?: Set<string>,
  nested = false
): Record<string, unknown> => {
  const result: Record<string, unknown> = {};
  // A pair, as a list of headers holds them (`{name: 'X-Api-Key', value: '…'}`): its name is in a field of its own, and
  // `value` is what that name says it is. A `key` there is such a name, not a key.
  const pairName = nested && 'value' in data ? [data.name, data.key, data.header].find((n) => typeof n === 'string') as string | undefined : undefined;

  for (const [key, value] of Object.entries(data)) {
    // Skip temporary fields
    if (key.startsWith('_')) continue;

    // SCHEMA-DRIVEN: If the node schema defines this field as 'secret' type, always redact it
    if (schemaSecretFields?.has(key)) {
      continue;
    }

    if (pairName !== undefined && key === 'value' && sensitiveName(pairName) && !String(value).includes('{{')) {
      continue;
    }

    // Skip fields with sensitive names (case-insensitive)
    if (sensitiveName(key) && !(pairName !== undefined && key === 'key') && (nested || !onlySoundsSecret(key, value, schemaPlainFields))) {
      continue;
    }

    // Skip values that look like secrets
    if (typeof value === 'string' && (looksLikeSecret(value) || knownSecrets.has(value) || holdsLiteralKey(key, value))) {
      continue;
    }

    // Recursively sanitize nested objects
    if (Array.isArray(value)) {
      result[key] = sanitizeList(value, key, knownSecrets, schemaSecretFields);
    } else if (value && typeof value === 'object') {
      result[key] = sanitizeNodeData(value as Record<string, unknown>, knownSecrets, schemaSecretFields, undefined, true);
    } else if (typeof value === 'string') {
      // Scrub a secret embedded MID-STRING — e.g. a literal key pasted inside the
      // ai_llm endpoint / customHeaders / customBodyTemplate (`{"Authorization":
      // "Bearer sk-..."}`, `?key=AIza...`). looksLikeSecret above is anchored
      // (whole-value) and misses substrings; redactSecretsInText is un-anchored and
      // keeps the surrounding JSON/URL syntactically valid (__REDACTED__).
      result[key] = REQUEST_TEXT.test(key) ? redactRequestText(value, knownSecrets) : redactSecretsInText(value, knownSecrets);
    } else {
      result[key] = value;
    }
  }

  return result;
};

/**
 * Get a node definition's field names: the ones its schema calls secret, and the rest
 */
const getSchemaFields = (nodeType: string): { secret: Set<string>; plain: Set<string> } => {
  const fields = { secret: new Set<string>(), plain: new Set<string>() };
  const loader = getModuleLoader();
  const def = loader.getNodeDefinition(nodeType);
  if (def?.properties) {
    for (const prop of def.properties) {
      (prop.type === 'secret' ? fields.secret : fields.plain).add(prop.id);
    }
  }
  return fields;
};

/**
 * Sanitize a workflow graph by removing secrets from all node data
 */
export const sanitizeWorkflowGraph = (
  graph: WorkflowGraph,
  knownSecrets: Set<string>
): WorkflowGraph => {
  return {
    ...graph,
    nodes: graph.nodes.map(node => {
      const schema = getSchemaFields(node.type);
      return {
        ...node,
        data: sanitizeNodeData(
          node.data as Record<string, unknown>,
          knownSecrets,
          schema.secret,
          schema.plain
        ),
      };
    }),
  };
};

/**
 * High-confidence inline-secret patterns for *substring* redaction inside the
 * free-text service templates (`endpoint` / `headers` / `bodyTemplate`). Unlike
 * `looksLikeSecret`'s anchored whole-string check, these are un-anchored so we
 * can scrub a key embedded mid-string — e.g. `"Authorization": "Bearer sk-…"`
 * or `…?key=AIza…`. They're deliberately vendor-specific: we do NOT include the
 * generic 32+ alnum catch-all here because inside a URL or JSON template it
 * would shred benign path segments and ids. `__REDACTED__` keeps the
 * surrounding JSON/URL syntactically valid so the importer still parses the
 * template (the receiving user just re-enters their own key).
 */
const REDACTED_TOKEN = '__REDACTED__';

export const redactSecretsInText = (
  text: string,
  knownSecrets: Set<string>,
): string => {
  if (typeof text !== 'string' || text.length === 0) return text;
  let out = text;
  // Exact known constant-secret values first — catches arbitrary user keys
  // that don't match any vendor shape. Guard on length so a 1-char secret
  // value can't blank out the whole template.
  for (const secret of knownSecrets) {
    if (secret && secret.length >= 8 && out.includes(secret)) {
      out = out.split(secret).join(REDACTED_TOKEN);
    }
  }
  // Vendor patterns — specific prefixes before the generic `sk-` so the
  // longer match wins. Bearer keeps its scheme word, swaps only the token.
  out = out
    .replace(/sk-ant-[A-Za-z0-9_-]{20,}/g, REDACTED_TOKEN)
    .replace(/sk-proj-[A-Za-z0-9_-]{20,}/g, REDACTED_TOKEN)
    .replace(/sk-[A-Za-z0-9]{20,}/g, REDACTED_TOKEN)
    .replace(/AIza[A-Za-z0-9_-]{35}/g, REDACTED_TOKEN)
    .replace(/gh[pousr]_[A-Za-z0-9]{36}/g, REDACTED_TOKEN)
    .replace(/xox[baprs]-[A-Za-z0-9-]{10,}/g, REDACTED_TOKEN)
    .replace(/(Bearer\s+)[A-Za-z0-9._-]{20,}/gi, `$1${REDACTED_TOKEN}`)
    .replace(/\b(Basic\s+)([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])/gi, (whole, scheme: string, encoded: string) =>
      isBasicCredential(encoded) ? `${scheme}${REDACTED_TOKEN}` : whole);
  for (const shape of KEY_SHAPES_IN_TEXT) out = out.replace(shape, REDACTED_TOKEN);
  return out;
};

/**
 * `redactSecretsInText` for a request written out as text, where a key has no need of a shape: the value under a JSON key
 * or in a query parameter with a sensitive name goes, and so does a URL's `user:pass@`. A `{{placeholder}}` stays: that is
 * how such a field names a key, and an imported flow needs it. So do counts (`"max_tokens": "800"`), and any value that is
 * not a quoted string (`"max_tokens": 800`).
 */
export const redactRequestText = (
  text: string,
  knownSecrets: Set<string>,
): string => {
  if (typeof text !== 'string' || text.length === 0) return text;
  // (a search's `keyword`, a post's `author` and a model's `tokenizer` have a part in them, in a URL that has to keep working)
  const stays = (name: string, value: string): boolean =>
    !sensitiveName(name.replace(/keyword|author(?!i)|tokeni[sz]/gi, '')) ||
    !value ||
    value.includes('{{') ||
    value.includes(REDACTED_TOKEN) ||
    (/tokens$/i.test(name) && /^\d+$/.test(value));
  return redactSecretsInText(text, knownSecrets)
    .replace(/"((?:[^"\\]|\\.)*)"(\s*:\s*)"((?:[^"\\]|\\.)*)"/g, (whole, name: string, colon: string, value: string) =>
      stays(name, value) ? whole : `"${name}"${colon}"${REDACTED_TOKEN}"`)
    .replace(/([?&])([^=&#\s"']+)=([^&#\s"']*)/g, (whole, mark: string, name: string, value: string) =>
      stays(name, value) ? whole : `${mark}${name}=${REDACTED_TOKEN}`)
    .replace(/(\b[a-z][a-z0-9+.-]*:\/\/)[^/\s"'?#@]+@/gi, (whole, scheme: string) => (whole.includes('{{') ? whole : scheme));
};

/**
 * Scrub a custom service for embedding in an exported project. The convention
 * is to reference keys via `{{apiKey}}` (resolved from a Project Constant at
 * run time) — those templates carry no secret. But nothing STOPS a user from
 * pasting a literal key straight into the endpoint/headers/body, so we redact
 * any inline secret-looking value before the definition leaves the machine.
 */
function sanitizeServiceForExport(
  svc: CustomService,
  knownSecrets: Set<string>,
): CustomService {
  return {
    ...svc,
    // `isBuiltIn: false` for the export — the importer always treats these as
    // user services so the receiving user can edit them.
    isBuiltIn: false,
    endpoint: redactRequestText(svc.endpoint, knownSecrets),
    headers: redactRequestText(svc.headers, knownSecrets),
    bodyTemplate: redactRequestText(svc.bodyTemplate, knownSecrets),
  };
}

/**
 * What a project keeps beside its flows: an endpoint, a preset, its settings. Their fields are the app's own and most
 * name a secret without holding one (`apiKeyEnvVar`, `authType`, `authHeaderName`, a default's `…ApiKeyConstant`), so the
 * rule for names is kept for their header maps, and everything else is judged by what it holds: a key in a URL went out
 * with the file.
 */
const sanitizeProjectRecord = <T extends object>(record: T, knownSecrets: Set<string>): T => {
  const result: Record<string, unknown> = {};
  for (const [field, value] of Object.entries(record)) {
    if (typeof value === 'string') {
      if (looksLikeSecret(value) || knownSecrets.has(value) || holdsLiteralKey(field, value)) continue;
      result[field] = REQUEST_TEXT.test(field) ? redactRequestText(value, knownSecrets) : redactSecretsInText(value, knownSecrets);
    } else if (/headers$/i.test(field) && value && typeof value === 'object') {
      result[field] = sanitizeNodeData(value as Record<string, unknown>, knownSecrets, undefined, undefined, true);
    } else {
      result[field] = value;
    }
  }
  return result as T;
};

/**
 * Walk every flow's graph and collect the CustomService definitions any
 * `service_call` node references (via `data.service`). Returns the
 * de-duplicated set, suitable for embedding in the exported project so
 * importing on a fresh machine can re-register the services with one click.
 * Each definition is run through `sanitizeServiceForExport` so an inline API
 * key (one a user hardcoded instead of using `{{apiKey}}`) never leaks.
 */
function collectReferencedServices(
  project: OAIYProject,
  knownSecrets: Set<string>,
): CustomService[] {
  const registry = listAllServices();
  const byId = new Map(registry.map((s) => [s.id, s]));
  const referencedIds = new Set<string>();
  for (const flow of project.flows) {
    for (const node of flow.graph?.nodes ?? []) {
      // service_call lives in the plugin module space; BuiltinNodeType
      // doesn't include it, so cast to string for the equality check.
      if ((node.type as string) !== 'service_call') continue;
      const data = node.data as Record<string, unknown> | undefined;
      const id = data?.service;
      if (typeof id === 'string' && id.length > 0) referencedIds.add(id);
    }
  }
  const out: CustomService[] = [];
  for (const id of referencedIds) {
    const svc = byId.get(id);
    if (svc) out.push(sanitizeServiceForExport(svc, knownSecrets));
  }
  return out;
}

/**
 * Create a sanitized copy of a project for export
 * Removes all secrets from node data and clears secret constant values.
 * Embeds the CustomService definitions referenced by `service_call`
 * nodes so the importer can re-register them on a fresh machine.
 */
export const sanitizeProjectForExport = (project: OAIYProject): ExportedProject => {
  // Collect all known secret values from project constants
  const knownSecrets = new Set<string>();
  for (const constant of project.constants || []) {
    if (constant.isSecret && constant.value) {
      knownSecrets.add(constant.value);
    }
  }

  const services = collectReferencedServices(project, knownSecrets);

  return {
    ...project,
    // Sanitize flows - remove secrets from node data using both schema and heuristics
    flows: project.flows.map(flow => ({
      ...flow,
      graph: sanitizeWorkflowGraph(flow.graph, knownSecrets),
    })),
    // Clear secret constant values on export
    constants: (project.constants || []).map(c => ({
      ...c,
      value: c.isSecret ? '' : c.value,
    })),
    llmEndpoints: project.llmEndpoints?.map((endpoint) => sanitizeProjectRecord(endpoint, knownSecrets)),
    imageGenEndpoints: project.imageGenEndpoints?.map((endpoint) => sanitizeProjectRecord(endpoint, knownSecrets)),
    httpPresets: project.httpPresets?.map((preset) => sanitizeProjectRecord(preset, knownSecrets)),
    settings: project.settings && sanitizeProjectRecord(project.settings, knownSecrets),
    // Web-only addition — embedded service definitions. Omitted entirely
    // when no service_call nodes are present so old desktop imports stay
    // byte-for-byte compatible.
    ...(services.length > 0 ? { services } : {}),
  };
};

/**
 * Export a project to JSON blob and trigger download
 */
export const exportProjectToFile = (project: OAIYProject): void => {
  const sanitizedProject = sanitizeProjectForExport(project);
  const blob = new Blob([JSON.stringify(sanitizedProject, null, 2)], { type: 'application/json' });
  triggerDownloadFromBlob(`oaiy-project-${project.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`, blob);
};

/**
 * Register any embedded services from an imported project into the
 * local `oaiy.customServices` registry. Skips services that already
 * exist by id so re-importing the same project doesn't duplicate.
 * Returns the count of newly-registered services so the caller can
 * surface a "Re-added N services" toast.
 */
export function registerImportedServices(
  imported: Pick<ExportedProject, 'services'>,
): { added: number; skipped: number } {
  const incoming = Array.isArray(imported.services) ? imported.services : [];
  if (incoming.length === 0) return { added: 0, skipped: 0 };
  const existing = listAllServices();
  const existingIds = new Set(existing.map((s) => s.id));
  let added = 0;
  let skipped = 0;
  for (const svc of incoming) {
    if (!svc || typeof svc !== 'object' || typeof svc.id !== 'string' || !svc.id) {
      skipped++;
      continue;
    }
    if (existingIds.has(svc.id)) {
      skipped++;
      continue;
    }
    // Always strip isBuiltIn — anything coming through this path is the
    // receiving user's own copy from here on.
    saveService({ ...svc, isBuiltIn: false });
    added++;
  }
  return { added, skipped };
}

/**
 * Guarantee every imported flow has a unique, non-empty id. A clean export
 * already satisfies this, so it's a no-op for well-formed files; it only fires
 * on hand-edited / corrupt input where a flow id is missing or duplicated
 * (which would otherwise collide React keys and break flow switching).
 *
 * No subflow/macro reference rewrite is needed on this REPLACE-mode path: the
 * only references that could break are *inbound* ones (a subflow `data.flowId`
 * or macro `data._macroWorkflowId` pointing AT a flow), and a flow with an
 * empty/duplicate id can't be a valid unambiguous target — the first holder of
 * a duplicate id keeps it, so existing references still resolve to it. (A
 * future MERGE import that pulls these flows into an already-populated project
 * WOULD need to remap-and-rewrite; that belongs in the merge path, not here.)
 */
const ensureUniqueFlowIds = <T extends { id?: string }>(flows: T[]): T[] => {
  const seen = new Set<string>();
  let counter = 0;
  return flows.map((f) => {
    let id = typeof f.id === 'string' ? f.id : '';
    if (!id || seen.has(id)) {
      const suffix = id ? id.slice(0, 8) : 'x';
      id = `flow-imported-${counter}-${suffix}`;
      while (seen.has(id)) id = `${id}-${counter}`;
      counter++;
    }
    seen.add(id);
    return f.id === id ? f : { ...f, id };
  });
};

/**
 * Parse and validate an imported project file
 * Returns merged project with defaults for any missing fields
 */
/**
 * A file's text as JSON, or an error that says what the file is not, in words ("This file isn't an OAIY project (it
 * isn't valid JSON)"), not the parser's ("Unexpected token 'h', "hello" is not valid JSON").
 */
export function parseJsonFile(content: string, what: 'project' | 'flow'): unknown {
  try {
    return JSON.parse(content);
  } catch {
    throw new Error(`This file isn't an OAIY ${what} (it isn't valid JSON).`);
  }
}

export const parseImportedProject = (content: string): ExportedProject => {
  const imported = parseJsonFile(content, 'project') as ExportedProject;

  // Validate basic structure: a project has flows (a flow file has nodes, and is added with Import flows)
  if (!imported || typeof imported !== 'object' || !Array.isArray(imported.flows)) {
    const flow = !!imported && typeof imported === 'object' && Array.isArray((imported as { nodes?: unknown }).nodes);
    throw new Error(
      flow
        ? "This file is a flow, not a project: add it with Import flows… instead."
        : "This file isn't an OAIY project (it has no flows).",
    );
  }

  // Merge with defaults for any missing fields
  return {
    ...createEmptyProject(),
    ...imported,
    // Normalize each flow's graph so a malformed/missing graph is coerced to an
    // empty one instead of crashing on graph access (or being autosaved over the
    // only good copy). Legit exports always carry a valid graph, so this is a no-op
    // for well-formed files.
    flows: ensureUniqueFlowIds(
      imported.flows.map((f) => ({
        ...f,
        graph:
          f && f.graph && Array.isArray(f.graph.nodes) && Array.isArray(f.graph.edges)
            ? { ...f.graph, nodes: withoutProviderSeeds(withoutRunOutputs(f.graph.nodes)) }
            : { nodes: [], edges: [] },
      }))
    ),
    // Preserve user's custom endpoints, add defaults if missing
    llmEndpoints: [
      ...defaultLLMEndpoints,
      ...(imported.llmEndpoints || []).filter(
        (e) => !defaultLLMEndpoints.find((d) => d.id === e.id)
      ),
    ],
    imageGenEndpoints: [
      ...defaultImageGenEndpoints,
      ...(imported.imageGenEndpoints || []).filter(
        (e) => !defaultImageGenEndpoints.find((d) => d.id === e.id)
      ),
    ],
    httpPresets: [
      ...defaultHttpPresets,
      ...(imported.httpPresets || []).filter(
        (p) => !defaultHttpPresets.find((d) => d.id === p.id)
      ),
    ],
    // Merge constants - preserve imported values for matching keys
    constants: defaultConstants.map((dc) => {
      const imported_c = (imported.constants || []).find((c) => c.key === dc.key);
      return imported_c ? { ...dc, value: imported_c.value } : dc;
    }).concat(
      (imported.constants || []).filter(
        (c) => !defaultConstants.find((d) => d.key === c.key)
      )
    ),
    // Merge settings
    settings: {
      ...defaultSettings,
      ...imported.settings,
    },
  };
};

/**
 * Import a project from a File object.
 *
 * Side effect: any embedded `services` are merged into the local
 * `oaiy.customServices` registry (existing ids are preserved as-is so
 * the user's edits aren't clobbered by a re-import). The returned
 * promise resolves with the merged project AND the service-register
 * result so the caller can surface a "Re-added N services" toast.
 */
export const importProjectFromFile = (
  file: File,
): Promise<{ project: ExportedProject; services: { added: number; skipped: number } }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const merged = parseImportedProject(content);
        const services = registerImportedServices(merged);
        resolve({ project: merged, services });
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
};

/**
 * Redact secrets from run history logs
 */
export const redactRunHistorySecrets = (logs: LogEntry[]): LogEntry[] => {
  // Reuse the vendor-aware, un-anchored redactor (sk-ant-/sk-proj-/AIza/gh*_/xox*/
  // Bearer). The old ad-hoc regexes stopped at the first '-' of a multi-segment key
  // (sk-ant-api03-REALSECRET → only "sk-ant" redacted, payload intact) and missed
  // AIza/ghp_/xoxb entirely — leaving real keys nearly unredacted in localStorage.
  const noKnown = new Set<string>();
  return logs.map((log) => ({
    ...log,
    message: redactSecretsInText(log.message, noKnown),
  }));
};

/**
 * Redact secrets from project constants before localStorage save
 */
export const redactConstantsForStorage = (constants: ProjectConstant[]): ProjectConstant[] => {
  return constants.map((c) => ({
    ...c,
    value: c.isSecret ? '' : c.value,
  }));
};
