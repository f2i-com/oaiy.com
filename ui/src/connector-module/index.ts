/**
 * Connector module — the factory.
 *
 * `createConnectorModule(config)` turns a connector config into the four things
 * `ModuleLoader.loadModule(manifest, nodes, runtime, path, compiler)` wants. It
 * is a FACTORY, not a bundled module: the node types do not exist until a config
 * names them, so there is nothing to ship statically.
 *
 * Usage (see `cli/src/connector.ts` for the wired version):
 *
 * ```ts
 * const built = createConnectorModule(parseConnectorConfig(json, path), {
 *   loopbackBaseUrl: process.env.OAIY_SERVER_URL,
 *   loopbackToken: process.env.OAIY_SERVER_TOKEN,
 * });
 * await getModuleLoader().loadModule(
 *   built.manifest, built.nodes, built.runtime, built.path, built.compiler,
 * );
 * ```
 */

import type { ModuleCompiler, ModuleManifest, NodeDefinition, RuntimeModule } from 'oaiy-core/src/module-types';
import { buildManifest, buildNodeDefinitions } from './nodes';
import { createConnectorCompiler } from './compiler';
import { createConnectorRuntime } from './runtime';
import { parseConnectorConfig } from './config';
import type { ConnectorConfig, ConnectorHostOptions } from './types';

export interface ConnectorModule {
  manifest: ModuleManifest;
  nodes: NodeDefinition[];
  runtime: RuntimeModule;
  compiler: ModuleCompiler;
  /** The `modulePath` argument for `loadModule` — a label, not a filesystem path. */
  path: string;
  /** The validated config the module was built from. */
  config: ConnectorConfig;
}

/**
 * Build a loadable module from an ALREADY-VALIDATED config.
 *
 * Prefer {@link buildConnectorModule}, which validates first — this overload
 * exists for callers holding a config they parsed themselves.
 */
export function createConnectorModule(
  config: ConnectorConfig,
  host: ConnectorHostOptions = {},
  path = 'connector',
): ConnectorModule {
  return {
    manifest: buildManifest(config),
    nodes: buildNodeDefinitions(config),
    runtime: createConnectorRuntime(config, host),
    compiler: createConnectorCompiler(config),
    path,
    config,
  };
}

/** Parse + validate raw config JSON, then build the module. Throws loudly on both. */
export function buildConnectorModule(
  raw: unknown,
  host: ConnectorHostOptions = {},
  source = 'connector config',
): ConnectorModule {
  const config = parseConnectorConfig(raw, source);
  return createConnectorModule(config, host, source);
}

export { parseConnectorConfig, ConnectorConfigError } from './config';
export { createConnectorRuntime, CONNECTOR_RUNTIME_NAME, resolvePath, extractRows } from './runtime';
export { createConnectorCompiler } from './compiler';
export { buildManifest, buildNodeDefinitions, buildNodeDefinition, pathPlaceholders } from './nodes';
export {
  CONNECTOR_OPERATIONS,
  RECORD_OPERATIONS,
  isConnectorOperation,
  DEFAULT_LOOPBACK_BASE_URL,
} from './types';
export type {
  ConnectorConfig,
  ConnectorNodeSpec,
  ConnectorOperation,
  ConnectorCall,
  ConnectorHostOptions,
} from './types';
