/**
 * Management client for oaiy-server's HTTP API — drives Python / venv / service /
 * model management from the CLI (the headless equivalent of the desktop
 * dashboard). Privileged routes (install / define / delete) need the bearer
 * token; the token is sent on every request (harmless for open routes).
 *
 *   OAIY_SERVER_URL    base URL                       [http://127.0.0.1:17972]
 *   OAIY_SERVER_TOKEN  bearer for privileged routes
 */
import { warnIfPlaintextRemoteToken } from './insecure-token';

export const BASE_URL = (process.env.OAIY_SERVER_URL || 'http://127.0.0.1:17972').replace(/\/+$/, '');
const TOKEN = process.env.OAIY_SERVER_TOKEN || '';
warnIfPlaintextRemoteToken(BASE_URL, !!TOKEN);

export class ServerError extends Error {}

/* eslint-disable @typescript-eslint/no-explicit-any */
async function api(method: string, path: string, body?: unknown): Promise<any> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (TOKEN) headers.Authorization = `Bearer ${TOKEN}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  let resp: Response;
  try {
    resp = await fetch(BASE_URL + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new ServerError(
      `cannot reach oaiy-server at ${BASE_URL} — is it running? (${e instanceof Error ? e.message : String(e)})`,
    );
  }
  if (resp.status === 204) return null;
  const text = await resp.text();
  let parsed: any = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = text;
  }
  if (!resp.ok) {
    if (resp.status === 403) {
      throw new ServerError(
        `forbidden (${path}) — this is a privileged route; set OAIY_SERVER_TOKEN to match the server's token`,
      );
    }
    const msg =
      parsed && typeof parsed === 'object' && parsed.error
        ? typeof parsed.error === 'string'
          ? parsed.error
          : JSON.stringify(parsed.error)
        : text || `HTTP ${resp.status}`;
    throw new ServerError(`oaiy-server ${method} ${path} → ${resp.status}: ${msg}`);
  }
  return parsed;
}

export const server = {
  health: () => api('GET', '/api/health'),
  config: () => api('GET', '/api/config'),
  python: {
    status: () => api('GET', '/api/python'),
    install: () => api('POST', '/api/python/install'),
    logs: (tail = 50) => api('GET', `/api/python/logs?tail=${tail}`),
    createVenv: (name: string, requirements: string[]) =>
      api('POST', '/api/python/venvs', { name, requirements }),
    deleteVenv: (name: string) => api('DELETE', `/api/python/venvs/${encodeURIComponent(name)}`),
  },
  services: {
    list: () => api('GET', '/api/services'),
    add: (template: unknown) => api('POST', '/api/services', template),
    export: (id: string) => api('GET', `/api/services/${encodeURIComponent(id)}/export`),
    remove: (id: string) => api('DELETE', `/api/services/${encodeURIComponent(id)}`),
    start: (id: string) => api('POST', `/api/services/${encodeURIComponent(id)}/start`),
    stop: (id: string) => api('POST', `/api/services/${encodeURIComponent(id)}/stop`),
    install: (id: string) => api('POST', `/api/services/${encodeURIComponent(id)}/install`),
    uninstall: (id: string) => api('POST', `/api/services/${encodeURIComponent(id)}/uninstall`),
    cancelInstall: (id: string) => api('POST', `/api/services/${encodeURIComponent(id)}/cancel-install`),
    logs: (id: string, tail = 50) =>
      api('GET', `/api/services/${encodeURIComponent(id)}/logs?tail=${tail}`),
    ensureByPort: (port: number) => api('POST', '/api/services/ensure-by-port', { port }),
  },
  models: {
    list: () => api('GET', '/api/models'),
    catalog: () => api('GET', '/api/models/catalog'),
    download: (url: string, filename?: string, subdir?: string) =>
      api('POST', '/api/models/download', { url, filename, subdir }),
    downloads: () => api('GET', '/api/models/downloads'),
    pause: (id: string) => api('POST', `/api/models/downloads/${encodeURIComponent(id)}/pause`),
    resume: (id: string) => api('POST', `/api/models/downloads/${encodeURIComponent(id)}/resume`),
    cancel: (id: string) => api('POST', `/api/models/downloads/${encodeURIComponent(id)}/cancel`),
    remove: (name: string) => api('DELETE', `/api/models/${encodeURIComponent(name)}`),
  },
};
