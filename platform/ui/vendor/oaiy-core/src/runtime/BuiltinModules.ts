import { baseObjectToJsValue, jsValueToBaseObject } from './ValueConverter.js';
import type { BoundedMap } from './BoundedMap.js';
import type { LogCallback } from '../types.js';

export interface BuiltinModuleDependencies {
  abortSignal: AbortSignal | null;
  agentMemory: BoundedMap<string, string | number | boolean | object>;
  performHttpRequest: (
    url: string,
    method: string,
    headers: Record<string, string>,
    body?: string
  ) => Promise<{ status: number; headers: Record<string, string>; body: string }>;
  loadPersistedMemory: () => Promise<void>;
  persistMemoryEntry: (key: string, value: unknown) => Promise<void>;
  deletePersistedMemoryEntry: (key: string) => Promise<void>;
  onLog: LogCallback | null;
}

export function createAbortModule(
  deps: Pick<BuiltinModuleDependencies, 'abortSignal'>
): Record<string, (...args: any[]) => any> {
  return {
    check: async () => {
      return deps.abortSignal?.aborted ? true : false;
    },
    checkThrow: async () => {
      if (deps.abortSignal?.aborted) {
        throw new Error('__ABORT__');
      }
      return null;
    },
    getSignalId: () => {
      return deps.abortSignal ? 'active' : 'none';
    },
  };
}

export function createUtilityModule(
  deps: Pick<BuiltinModuleDependencies, 'performHttpRequest'>
): Record<string, (...args: any[]) => any> {
  return {
    httpRequest: async (args: any[]) => {
      const url = String(args[0]);
      const method = String(args[1] || 'GET');
      const headersArg = args[2];
      const bodyArg = args[3];

      let headers: Record<string, string> = {};
      if (typeof headersArg === 'object' && headersArg !== null) {
        headers = headersArg as Record<string, string>;
      }

      let body: string | undefined;
      if (bodyArg !== undefined && bodyArg !== null) {
        body = String(bodyArg);
      }

      try {
        const result = await deps.performHttpRequest(url, method, headers, body);
        return {
          status: result.status,
          body: result.body,
          headers: result.headers,
        };
      } catch (error) {
        return `ERROR: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
    template: (args: any[]) => {
      const template = String(args[0]);
      const context = args[1] as Record<string, unknown> || {};
      return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
        const value = context[key];
        return value !== undefined ? String(value) : match;
      });
    },
  };
}

export function createAgentModule(
  deps: Pick<BuiltinModuleDependencies, 'agentMemory' | 'loadPersistedMemory' | 'persistMemoryEntry' | 'deletePersistedMemoryEntry' | 'onLog'>
): Record<string, (...args: any[]) => any> {
  return {
    get: async (args: any[]) => {
      const key = String(args[0]);
      const value = deps.agentMemory.get(key);
      return value === undefined ? null : value;
    },
    set: async (args: any[]) => {
      const key = String(args[0]);
      const value = args[1];
      deps.agentMemory.set(key, value as any);
      return null;
    },
    memory: async (args: any[]) => {
      const key = String(args[0]);
      const operation = String(args[1] || 'get');
      const inputValue = args[2];

      try {
        await deps.loadPersistedMemory();
        switch (operation) {
          case 'set': {
            deps.agentMemory.set(key, inputValue as any);
            try {
              await deps.persistMemoryEntry(key, inputValue);
            } catch (err) {
              if (deps.onLog) deps.onLog({ source: 'Agent', message: `Memory persist error: ${err}`, type: 'info' });
            }
            return inputValue || null;
          }
          case 'append': {
            const existingVal = deps.agentMemory.get(key);
            let resultArray: unknown[];
            if (Array.isArray(existingVal)) {
              resultArray = [...existingVal, inputValue];
            } else if (existingVal !== undefined && existingVal !== null) {
              resultArray = [existingVal, inputValue];
            } else {
              resultArray = [inputValue];
            }
            deps.agentMemory.set(key, resultArray);
            try {
              await deps.persistMemoryEntry(key, resultArray);
            } catch (err) {
              if (deps.onLog) deps.onLog({ source: 'Agent', message: `Memory persist error: ${err}`, type: 'info' });
            }
            return resultArray;
          }
          case 'clear': {
            deps.agentMemory.delete(key);
            try {
              await deps.deletePersistedMemoryEntry(key);
            } catch (err) {
              if (deps.onLog) deps.onLog({ source: 'Agent', message: `Memory delete error: ${err}`, type: 'info' });
            }
            return null;
          }
          case 'get':
          default: {
            const value = deps.agentMemory.get(key);
            return value === undefined ? null : value;
          }
        }
      } catch (err) {
        if (deps.onLog) deps.onLog({ source: 'Agent', message: `Memory persistence error: ${err}`, type: 'error' });
        const value = deps.agentMemory.get(key);
        return value === undefined ? null : value;
      }
    },
  };
}
