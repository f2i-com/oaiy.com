/**
 * Connector module — shared types.
 *
 * A "connector" is a LINKED PROVIDER described entirely by data. OAIY's Rust side
 * claims a queued run from that provider and writes a small JSON file; this module
 * turns that file into a real OAIY node module (manifest + node definitions +
 * runtime + compiler) at run time.
 *
 * The two halves of the contract:
 *
 *   - **Node TYPE names are the provider's vocabulary.** They come from the file
 *     and nothing here may know or assume any of them. Grep this directory for a
 *     provider's name and you must find nothing.
 *   - **Operations are a CLOSED SET.** Seven of them, listed below. A config that
 *     names anything else is refused when the module is built, before a flow runs.
 */

/**
 * The closed set of operations a connector node may perform.
 *
 * This list is the whole vocabulary. It is deliberately small: a provider
 * describes WHICH of these its node types map to and (for the record
 * operations) WHERE, and nothing else.
 */
export const CONNECTOR_OPERATIONS = [
  /** The run's inputs object, as the graph's entry node. */
  'runInput',
  /** A chat completion from this machine's own AI gateway (loopback). */
  'chat',
  /** GET the path, return the provider's rows. */
  'listRecords',
  /** POST the path. */
  'createRecord',
  /** PATCH/PUT the path. */
  'updateRecord',
  /** Call a plugin connector on this desktop. */
  'connectorRequest',
  /** Read/control this desktop's services. */
  'serviceControl',
] as const;

export type ConnectorOperation = (typeof CONNECTOR_OPERATIONS)[number];

/** Operations that address a provider resource through `path`. */
export const RECORD_OPERATIONS: readonly ConnectorOperation[] = [
  'listRecords',
  'createRecord',
  'updateRecord',
];

export function isConnectorOperation(v: unknown): v is ConnectorOperation {
  return typeof v === 'string' && (CONNECTOR_OPERATIONS as readonly string[]).includes(v);
}

/**
 * One entry of the config's `nodes` array: a provider node type bound to one of
 * the closed-set operations.
 */
export interface ConnectorNodeSpec {
  /** The provider's node type name, e.g. what its graphs call this step. */
  nodeType: string;
  /** Which of the seven operations this node type performs. */
  operation: ConnectorOperation;
  /**
   * Resource path for the record operations, relative to `baseUrl`. May contain
   * `{placeholder}` segments, substituted from the node's own data at run time.
   */
  path?: string;
  /** HTTP method override for `updateRecord` (PATCH by default; PUT is common). */
  method?: string;
  /** Optional human-readable name for the palette; defaults to a humanised nodeType. */
  name?: string;
  /** Optional description for the palette. */
  description?: string;
}

/**
 * The connector config file: written by OAIY's Rust side, read here via
 * `oaiy run --connector <path>`.
 */
export interface ConnectorConfig {
  /** Base URL of the linked provider, e.g. `http://provider.local`. */
  baseUrl: string;
  /** Bearer credential for the provider's API. Never logged. */
  credential: string;
  /** The provider's node vocabulary, bound to operations. */
  nodes: ConnectorNodeSpec[];
  /**
   * Optional module identity. Defaults are neutral (`connector`) precisely so a
   * provider's name never has to appear in code — if a config supplies these,
   * they are DATA from the file like everything else.
   */
  moduleId?: string;
  moduleName?: string;
  version?: string;
}

/**
 * The per-node call descriptor the compiler bakes into the generated script and
 * the runtime receives. Everything the runtime needs to perform one operation,
 * with no lookups back into the config's node list.
 */
export interface ConnectorCall {
  nodeId: string;
  nodeType: string;
  operation: ConnectorOperation;
  path?: string;
  method?: string;
  /** The node instance's own data (its configured fields). */
  data: Record<string, unknown>;
}

/**
 * Host wiring the factory needs but the connector config does not carry: where
 * THIS desktop's own loopback API lives, and the bearer for its privileged
 * routes. The CLI fills these in from `OAIY_SERVER_URL` / `OAIY_SERVER_TOKEN`.
 */
export interface ConnectorHostOptions {
  /** OAIY's own local API base, default `http://127.0.0.1:17972`. */
  loopbackBaseUrl?: string;
  /** Bearer for OAIY's privileged loopback routes (connector relay, services). */
  loopbackToken?: string;
}

export const DEFAULT_LOOPBACK_BASE_URL = 'http://127.0.0.1:17972';
