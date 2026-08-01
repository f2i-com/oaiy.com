/**
 * Connector module — manifest + node definitions, derived from config.
 *
 * Nothing in here names a node type. The names, and the fields each node
 * exposes, are computed from the config entry: its operation decides the shape,
 * and its `path` decides which placeholder fields the node needs filled in.
 */

import type {
  ModuleManifest,
  NodeDefinition,
  PropertyDefinition,
} from 'oaiy-core/src/module-types';
import type { ConnectorConfig, ConnectorNodeSpec, ConnectorOperation } from './types';

export const DEFAULT_MODULE_ID = 'connector';
export const DEFAULT_MODULE_NAME = 'Connector';
export const DEFAULT_MODULE_VERSION = '1.0.0';

/** `list_the_things` → `List The Things`, for the palette. */
function humanise(nodeType: string): string {
  return nodeType
    .split('_')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

const OPERATION_SUMMARY: Record<ConnectorOperation, string> = {
  runInput: 'The run\'s inputs, as this graph\'s entry point.',
  chat: 'A chat completion from this machine\'s own AI gateway.',
  listRecords: 'Read rows from the linked provider.',
  createRecord: 'Create a record on the linked provider.',
  updateRecord: 'Update a record on the linked provider.',
  connectorRequest: 'Send a command to a plugin connector on this desktop.',
  serviceControl: 'Read or control this desktop\'s services.',
};

/** Placeholder names in a path template, in order, de-duplicated. */
export function pathPlaceholders(path: string | undefined): string[] {
  if (!path) return [];
  const out: string[] = [];
  for (const m of path.matchAll(/\{([^}\s/]+)\}/g)) {
    if (!out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

function placeholderProperties(spec: ConnectorNodeSpec): PropertyDefinition[] {
  return pathPlaceholders(spec.path).map((name) => ({
    id: name,
    name: humanise(name),
    type: 'string',
    default: '',
    required: true,
    description: `Fills {${name}} in ${spec.path}. May also arrive as a field on this node's input.`,
  }));
}

function operationProperties(operation: ConnectorOperation): PropertyDefinition[] {
  switch (operation) {
    case 'runInput':
      return [
        {
          id: 'key',
          name: 'Input Key',
          type: 'string',
          default: '',
          description: 'Emit just this one run input. Leave blank to emit the whole inputs object.',
        },
      ];
    case 'chat':
      return [
        {
          id: 'systemPrompt',
          name: 'System Prompt',
          type: 'textarea',
          default: '',
          rows: 3,
          description: 'Optional context for the completion.',
        },
        {
          id: 'prompt',
          name: 'Prompt',
          type: 'textarea',
          default: '',
          rows: 3,
          description: 'Static prompt text. Anything wired into this node is appended to it.',
        },
        {
          id: 'model',
          name: 'Model',
          type: 'string',
          default: '',
          description: 'Leave blank to let the gateway choose its default model.',
        },
        {
          id: 'provider',
          name: 'Provider',
          type: 'string',
          default: '',
          advanced: true,
          description: 'Gateway provider id. Blank uses the default provider.',
        },
        {
          id: 'temperature',
          name: 'Temperature',
          type: 'number',
          default: 0,
          min: 0,
          max: 2,
          step: 0.1,
          advanced: true,
          description: '0 leaves it to the gateway.',
        },
        {
          id: 'maxTokens',
          name: 'Max Tokens',
          type: 'number',
          default: 0,
          min: 0,
          advanced: true,
          description: '0 leaves it to the gateway.',
        },
      ];
    case 'listRecords':
      return [
        {
          id: 'query',
          name: 'Query Parameters',
          type: 'json',
          default: {},
          description: 'Appended to the request as a query string.',
        },
        {
          id: 'rowsPath',
          name: 'Rows Path',
          type: 'string',
          default: '',
          advanced: true,
          description:
            'Where the rows sit in the provider\'s response, e.g. "data.items". Blank auto-detects.',
        },
      ];
    case 'createRecord':
    case 'updateRecord':
      return [
        {
          id: 'body',
          name: 'Body',
          type: 'json',
          default: {},
          description: 'What to send. Leave blank to send the object wired into this node.',
        },
      ];
    case 'connectorRequest':
      return [
        {
          id: 'connector',
          name: 'Connector',
          type: 'string',
          default: '',
          required: true,
          description: 'Id of the plugin connector on this desktop.',
        },
        {
          id: 'command',
          name: 'Command',
          type: 'string',
          default: '',
          required: true,
          description: 'The command to relay.',
        },
        {
          id: 'payload',
          name: 'Payload',
          type: 'json',
          default: {},
          description: 'Command arguments. Leave blank to relay the object wired into this node.',
        },
        {
          id: 'idempotencyKey',
          name: 'Idempotency Key',
          type: 'string',
          default: '',
          advanced: true,
          description:
            'Stable key for this logical event. Blank derives one from the run, the node and the payload.',
        },
      ];
    case 'serviceControl':
      return [
        {
          id: 'action',
          name: 'Action',
          type: 'select',
          default: 'list',
          options: [
            { value: 'list', label: 'List services' },
            { value: 'status', label: 'Status of one service' },
            { value: 'start', label: 'Start a service' },
            { value: 'stop', label: 'Stop a service' },
            { value: 'logs', label: 'Read a service\'s logs' },
          ],
          description: 'What to do with this desktop\'s services.',
        },
        {
          id: 'service',
          name: 'Service',
          type: 'string',
          default: '',
          description: 'Service id. Required for everything except "List services".',
        },
        {
          id: 'tail',
          name: 'Log Lines',
          type: 'number',
          default: 50,
          min: 1,
          advanced: true,
          description: 'How many log lines the "logs" action returns.',
        },
      ];
  }
}

/** Build the NodeDefinition for one config entry. */
export function buildNodeDefinition(spec: ConnectorNodeSpec): NodeDefinition {
  const properties = [...placeholderProperties(spec), ...operationProperties(spec.operation)];
  const description =
    spec.description ??
    (spec.path ? `${OPERATION_SUMMARY[spec.operation]} (${spec.path})` : OPERATION_SUMMARY[spec.operation]);

  return {
    id: spec.nodeType,
    name: spec.name ?? humanise(spec.nodeType),
    description,
    tags: ['connector', spec.operation],
    inputs:
      spec.operation === 'runInput'
        ? []
        : [
            {
              id: 'default',
              name: 'Input',
              type: 'any',
              position: 'left',
              description: 'Feeds this node\'s payload, query values and path placeholders.',
            },
          ],
    outputs: [
      {
        id: 'default',
        name: 'Result',
        type: 'any',
        position: 'right',
        description: OPERATION_SUMMARY[spec.operation],
      },
    ],
    properties,
    compiler: {
      // The module compiler emits this node; there is no template form of it.
      customHandler: true,
      async: true,
      abortable: true,
      statusTracking: true,
    },
  };
}

export function buildManifest(config: ConnectorConfig): ModuleManifest {
  return {
    id: config.moduleId ?? DEFAULT_MODULE_ID,
    name: config.moduleName ?? DEFAULT_MODULE_NAME,
    version: config.version ?? DEFAULT_MODULE_VERSION,
    description: 'Node types contributed by the linked provider, built from its connector config.',
    category: 'Custom',
    nodes: config.nodes.map((n) => n.nodeType),
    // The provider API and this machine's own loopback API, plus service control
    // for the serviceControl operation.
    permissions: ['network', 'network:local', 'service:start'],
  };
}

export function buildNodeDefinitions(config: ConnectorConfig): NodeDefinition[] {
  return config.nodes.map(buildNodeDefinition);
}
