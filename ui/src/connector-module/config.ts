/**
 * Connector config parsing + validation.
 *
 * Every refusal here happens BEFORE a flow runs, and every message names the
 * offending entry. That is the point: a connector config that this build cannot
 * honour must fail loudly at load time rather than produce a module whose nodes
 * quietly do nothing.
 */

import {
  CONNECTOR_OPERATIONS,
  RECORD_OPERATIONS,
  isConnectorOperation,
  type ConnectorConfig,
  type ConnectorNodeSpec,
} from './types';

export class ConnectorConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConnectorConfigError';
  }
}

/** Node ids the module loader accepts (`validateNodeDefinition`). */
const NODE_TYPE_RE = /^[a-z][a-z0-9_]*$/;
/** Module ids the module loader accepts (`validateManifest`). */
const MODULE_ID_RE = /^[a-z][a-z0-9-]*$/;
const SEMVER_RE = /^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$/;

const WRITE_METHODS = new Set(['PATCH', 'PUT', 'POST']);

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function requireString(v: unknown, field: string, where: string): string {
  if (typeof v !== 'string' || v.trim() === '') {
    throw new ConnectorConfigError(
      `${where}: "${field}" must be a non-empty string (got ${describe(v)}).`,
    );
  }
  return v.trim();
}

function describe(v: unknown): string {
  if (v === undefined) return 'nothing';
  if (v === null) return 'null';
  if (Array.isArray(v)) return `an array of ${v.length}`;
  return `${typeof v} ${JSON.stringify(v)}`;
}

/**
 * Parse + validate a connector config object.
 *
 * @param raw    the parsed JSON of the connector file
 * @param source a label for error messages (usually the file path)
 */
export function parseConnectorConfig(raw: unknown, source = 'connector config'): ConnectorConfig {
  if (!isRecord(raw)) {
    throw new ConnectorConfigError(`${source}: expected a JSON object, got ${describe(raw)}.`);
  }

  const baseUrl = requireString(raw.baseUrl, 'baseUrl', source).replace(/\/+$/, '');
  let parsedBase: URL;
  try {
    parsedBase = new URL(baseUrl);
  } catch {
    throw new ConnectorConfigError(
      `${source}: "baseUrl" is not a valid absolute URL (got ${JSON.stringify(baseUrl)}).`,
    );
  }
  if (parsedBase.protocol !== 'http:' && parsedBase.protocol !== 'https:') {
    throw new ConnectorConfigError(
      `${source}: "baseUrl" must be http(s) — ${JSON.stringify(parsedBase.protocol)} is not a scheme this build can call.`,
    );
  }

  // Validated but never echoed: this is the provider's bearer.
  const credential = requireString(raw.credential, 'credential', source);

  if (!Array.isArray(raw.nodes) || raw.nodes.length === 0) {
    throw new ConnectorConfigError(
      `${source}: "nodes" must be a non-empty array — a connector with no node types can run nothing.`,
    );
  }

  const seen = new Set<string>();
  const nodes: ConnectorNodeSpec[] = raw.nodes.map((entry, i) => {
    const where = `${source}: nodes[${i}]`;
    if (!isRecord(entry)) {
      throw new ConnectorConfigError(`${where}: expected an object, got ${describe(entry)}.`);
    }

    const nodeType = requireString(entry.nodeType, 'nodeType', where);
    if (!NODE_TYPE_RE.test(nodeType)) {
      throw new ConnectorConfigError(
        `${where}: nodeType ${JSON.stringify(nodeType)} is not a usable node id — ` +
          `it must start with a lowercase letter and contain only lowercase letters, digits and underscores.`,
      );
    }
    if (seen.has(nodeType)) {
      throw new ConnectorConfigError(
        `${where}: nodeType ${JSON.stringify(nodeType)} is declared more than once; ` +
          `a node type maps to exactly one operation.`,
      );
    }
    seen.add(nodeType);

    const operation = entry.operation;
    if (!isConnectorOperation(operation)) {
      // The single loudest failure in this file: an operation this build cannot
      // perform must never become a node that silently returns its input.
      throw new ConnectorConfigError(
        `${where}: operation ${describe(operation)} is not something this build can perform. ` +
          `The supported operations are: ${CONNECTOR_OPERATIONS.join(', ')}.`,
      );
    }

    const spec: ConnectorNodeSpec = { nodeType, operation };

    if (RECORD_OPERATIONS.includes(operation)) {
      const path = requireString(entry.path, 'path', `${where} (${operation})`);
      if (!path.startsWith('/')) {
        throw new ConnectorConfigError(
          `${where}: path ${JSON.stringify(path)} must start with "/" — it is resolved against baseUrl.`,
        );
      }
      spec.path = path;
    } else if (entry.path !== undefined) {
      throw new ConnectorConfigError(
        `${where}: the ${operation} operation does not address a provider path, ` +
          `but a "path" was supplied. Remove it, or bind this node type to one of: ${RECORD_OPERATIONS.join(', ')}.`,
      );
    }

    if (entry.method !== undefined) {
      const method = requireString(entry.method, 'method', where).toUpperCase();
      if (operation !== 'updateRecord' && operation !== 'createRecord') {
        throw new ConnectorConfigError(
          `${where}: "method" only applies to createRecord / updateRecord (this is ${operation}).`,
        );
      }
      if (!WRITE_METHODS.has(method)) {
        throw new ConnectorConfigError(
          `${where}: method ${JSON.stringify(method)} is not one this build sends for ${operation} ` +
            `(expected one of ${[...WRITE_METHODS].join(', ')}).`,
        );
      }
      spec.method = method;
    }

    if (entry.name !== undefined) spec.name = requireString(entry.name, 'name', where);
    if (entry.description !== undefined) {
      spec.description = requireString(entry.description, 'description', where);
    }

    return spec;
  });

  const config: ConnectorConfig = { baseUrl, credential, nodes };

  if (raw.moduleId !== undefined) {
    const moduleId = requireString(raw.moduleId, 'moduleId', source);
    if (!MODULE_ID_RE.test(moduleId)) {
      throw new ConnectorConfigError(
        `${source}: moduleId ${JSON.stringify(moduleId)} must start with a lowercase letter ` +
          `and contain only lowercase letters, digits and hyphens.`,
      );
    }
    config.moduleId = moduleId;
  }
  if (raw.moduleName !== undefined) {
    config.moduleName = requireString(raw.moduleName, 'moduleName', source);
  }
  if (raw.version !== undefined) {
    const version = requireString(raw.version, 'version', source);
    if (!SEMVER_RE.test(version)) {
      throw new ConnectorConfigError(
        `${source}: version ${JSON.stringify(version)} must be a semantic version like 1.0.0.`,
      );
    }
    config.version = version;
  }

  return config;
}
