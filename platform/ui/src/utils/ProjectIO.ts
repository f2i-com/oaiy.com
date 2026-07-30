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
  ];
  return secretPatterns.some(pattern => pattern.test(value));
};

/**
 * Sanitize node data by removing sensitive fields and secret-looking values.
 * Accepts schema-defined secret fields from node definitions.
 */
export const sanitizeNodeData = (
  data: Record<string, unknown>,
  knownSecrets: Set<string>,
  schemaSecretFields?: Set<string>
): Record<string, unknown> => {
  const sensitiveFields = ['apiKey', 'password', 'token', 'secret', 'key', 'credential', 'auth'];
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(data)) {
    // Skip temporary fields
    if (key.startsWith('_')) continue;

    // SCHEMA-DRIVEN: If the node schema defines this field as 'secret' type, always redact it
    if (schemaSecretFields?.has(key)) {
      continue;
    }

    // Skip fields with sensitive names (case-insensitive)
    if (sensitiveFields.some(sf => key.toLowerCase().includes(sf))) {
      continue;
    }

    // Skip values that look like secrets
    if (typeof value === 'string' && (looksLikeSecret(value) || knownSecrets.has(value))) {
      continue;
    }

    // Recursively sanitize nested objects
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = sanitizeNodeData(value as Record<string, unknown>, knownSecrets, schemaSecretFields);
    } else if (typeof value === 'string') {
      // Scrub a secret embedded MID-STRING — e.g. a literal key pasted inside the
      // ai_llm endpoint / customHeaders / customBodyTemplate (`{"Authorization":
      // "Bearer sk-..."}`, `?key=AIza...`). looksLikeSecret above is anchored
      // (whole-value) and misses substrings; redactSecretsInText is un-anchored and
      // keeps the surrounding JSON/URL syntactically valid (__REDACTED__).
      result[key] = redactSecretsInText(value, knownSecrets);
    } else {
      result[key] = value;
    }
  }

  return result;
};

/**
 * Get secret field names from node definition schema
 */
const getSchemaSecretFields = (nodeType: string): Set<string> => {
  const secretFields = new Set<string>();
  const loader = getModuleLoader();
  const def = loader.getNodeDefinition(nodeType);
  if (def?.properties) {
    for (const prop of def.properties) {
      if (prop.type === 'secret') {
        secretFields.add(prop.id);
      }
    }
  }
  return secretFields;
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
      const schemaSecretFields = getSchemaSecretFields(node.type);
      return {
        ...node,
        data: sanitizeNodeData(
          node.data as Record<string, unknown>,
          knownSecrets,
          schemaSecretFields
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
    .replace(/(Bearer\s+)[A-Za-z0-9._-]{20,}/gi, `$1${REDACTED_TOKEN}`);
  return out;
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
    endpoint: redactSecretsInText(svc.endpoint, knownSecrets),
    headers: redactSecretsInText(svc.headers, knownSecrets),
    bodyTemplate: redactSecretsInText(svc.bodyTemplate, knownSecrets),
  };
}

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
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `oaiy-project-${project.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
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
export const parseImportedProject = (content: string): ExportedProject => {
  const imported = JSON.parse(content) as ExportedProject;

  // Validate basic structure
  if (!imported.flows || !Array.isArray(imported.flows)) {
    throw new Error('Invalid project: missing flows array');
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
            ? f.graph
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
