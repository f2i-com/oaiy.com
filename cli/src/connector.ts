/**
 * `oaiy run --connector <path>` — load a linked provider's node module.
 *
 * OAIY's Rust side claims a queued run from the linked provider, writes this
 * file, and invokes the CLI with `--connector`. The file is the WHOLE contract:
 * the provider's node type names, which of the seven operations each performs,
 * and where the record operations point.
 *
 * The module is built at run time (there is nothing to bundle — the node types
 * do not exist until the file names them) and registered in the same
 * `ModuleLoader` singleton the bundled modules use, so the compiler and the
 * JobManager pick it up with no further wiring.
 */
import fs from 'node:fs';
import path from 'node:path';
import { getModuleLoader } from 'oaiy-core';
import { buildConnectorModule, type ConnectorModule } from '../../ui/src/connector-module';

export class ConnectorLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConnectorLoadError';
  }
}

/** Read + parse the connector file. Every failure names the file. */
export function readConnectorFile(filePath: string): unknown {
  const abs = path.resolve(filePath);
  let text: string;
  try {
    text = fs.readFileSync(abs, 'utf8');
  } catch (e) {
    throw new ConnectorLoadError(
      `--connector ${abs}: cannot read the connector config (${e instanceof Error ? e.message : String(e)}).`,
    );
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (e) {
    throw new ConnectorLoadError(
      `--connector ${abs}: not valid JSON (${e instanceof Error ? e.message : String(e)}).`,
    );
  }
}

/**
 * Build the connector module from a file and register it with the shared module
 * loader. Returns the built module so callers can report what it contributed.
 *
 * A registration failure (a node type that collides with a bundled one, an
 * invalid manifest) throws — running the flow anyway would mean running it
 * WITHOUT the provider's nodes, which the compiler would then refuse node by
 * node, or worse, resolve to a different module's same-named node.
 */
export async function loadConnectorModule(filePath: string): Promise<ConnectorModule> {
  const abs = path.resolve(filePath);
  const raw = readConnectorFile(abs);

  const built = buildConnectorModule(
    raw,
    {
      // This machine's own API: where the loopback operations (chat, the
      // connector relay, service control) go. Same env vars the rest of the CLI
      // uses to reach OAIY Desktop / oaiy-server.
      loopbackBaseUrl: process.env.OAIY_SERVER_URL,
      loopbackToken: process.env.OAIY_SERVER_TOKEN,
    },
    abs,
  );

  const loader = getModuleLoader();
  const result = await loader.loadModule(
    built.manifest,
    built.nodes,
    built.runtime,
    built.path,
    built.compiler,
  );
  if (!result.success) {
    const detail = result.error?.details ? ` — ${JSON.stringify(result.error.details)}` : '';
    throw new ConnectorLoadError(
      `--connector ${abs}: the connector module could not be registered: ${result.error?.error ?? 'unknown error'}${detail}`,
    );
  }

  return built;
}
