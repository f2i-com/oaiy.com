/**
 * Connector module — compiler.
 *
 * Emits one call per node: `await Connector.<operation>(<input>, <call>)`, where
 * `<call>` is a literal the runtime consumes. The node type appears in the
 * generated code only as data inside that literal — the emitter branches on the
 * OPERATION, which is a closed set, never on the name.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { resolveRefsExpr } from 'oaiy-core/src/value-refs';
import { CONNECTOR_RUNTIME_NAME } from './runtime';
import type { ConnectorCall, ConnectorConfig, ConnectorNodeSpec } from './types';

/**
 * Keys the compiler injects into node data that must not be shipped to the
 * runtime: `projectSettings` is the whole project's settings object (large, and
 * none of a connector node's business).
 */
const DATA_KEYS_TO_DROP = new Set(['projectSettings']);

function nodeData(node: { id: string; data: Record<string, unknown> }): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node.data ?? {})) {
    if (DATA_KEYS_TO_DROP.has(k) || k.startsWith('__')) continue;
    if (typeof v === 'function' || typeof v === 'undefined') continue;
    out[k] = v;
  }
  return out;
}

/**
 * Serialise the call descriptor into the generated script.
 *
 * A node carrying something unserialisable (a cycle, a DOM handle) would
 * otherwise throw deep inside `JSON.stringify` with no clue which node; the
 * module compiler's caller turns a throw here into a named compile failure, so
 * the flow refuses to run rather than running with a mangled node.
 */
function serialiseCall(call: ConnectorCall): string {
  try {
    return JSON.stringify(call);
  } catch (e) {
    throw new Error(
      `its configured fields cannot be serialised (${e instanceof Error ? e.message : String(e)}). ` +
        `Remove any circular or non-JSON value from this node's data.`,
    );
  }
}

export function createConnectorCompiler(config: ConnectorConfig): ModuleCompiler {
  const specs = new Map<string, ConnectorNodeSpec>(config.nodes.map((n) => [n.nodeType, n]));

  return {
    name: CONNECTOR_RUNTIME_NAME,

    getNodeTypes() {
      return [...specs.keys()];
    },

    compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
      const spec = specs.get(nodeType);
      // Not one of ours. Returning null lets the engine keep looking (and, if
      // nobody claims it, refuse the flow at compile time).
      if (!spec) return null;

      const { node, inputs, outputVar, skipVarDeclaration } = ctx;
      const letOrAssign = skipVarDeclaration ? '' : 'let ';

      const call: ConnectorCall = {
        nodeId: node.id,
        nodeType,
        operation: spec.operation,
        data: nodeData(node),
      };
      if (spec.path) call.path = spec.path;
      if (spec.method) call.method = spec.method;

      let callLiteral: string;
      try {
        callLiteral = serialiseCall(call);
      } catch (e) {
        throw new Error(`node "${node.id}" (${nodeType}) cannot be compiled: ${e instanceof Error ? e.message : String(e)}`);
      }

      // The entry operation reads the run's inputs; every other operation reads
      // whatever is wired into it.
      const inputExpr =
        spec.operation === 'runInput'
          ? '__inputs'
          : inputs.get('default') || inputs.get('input') || 'null';

      // The node's data is frozen at compile time, but a graph points its
      // fields at values it does not hold — "$inputs.callId", an answers map
      // addressed as "$nodes.decide.update". Resolved here, against what the
      // run actually produced, because otherwise the REFERENCE is what the
      // operation acts on: a call was rejected with the callId
      // `"$inputs.callId"`, which is not a call, and the plugin refused it.
      const dataExpr = resolveRefsExpr(call.data, 'workflow_context', '__inputs', inputExpr);

      return `
  // --- Node: ${node.id} (${nodeType} → ${spec.operation}) ---
  ${letOrAssign}${outputVar} = await ${CONNECTOR_RUNTIME_NAME}.${spec.operation}(
    ${inputExpr},
    Object.assign({}, ${callLiteral}, { data: ${dataExpr} })
  );
  workflow_context[${JSON.stringify(node.id)}] = ${outputVar};`;
    },
  };
}
