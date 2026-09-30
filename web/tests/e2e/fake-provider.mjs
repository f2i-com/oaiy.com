/**
 * Stand-ins for a model provider on this computer, for the browser tests: an OpenAI-dialect one and an Anthropic-shaped one, each
 * with the failures a real one has, and a log of exactly what reached it (headers, body, when, and whether the caller went away
 * before the answer was finished).
 *
 * What is imitated, because a page has to cope with it:
 *   - OpenAI: `GET /v1/models` with a wrong key is a readable 401 with CORS headers, but `POST /v1/chat/completions` with a wrong key
 *     is a 401 WITHOUT them, which a browser shows as an opaque "Failed to fetch"; its error text quotes the key it refused, masked;
 *   - Anthropic: a browser's call needs `anthropic-version` and, from a page, `anthropic-dangerous-direct-browser-access: true`;
 *     a model list is paged (`has_more`, `last_id`, `after_id`); a stream is named events;
 *   - a server that is up and refuses CORS (no `Access-Control-Allow-Origin`, no answer to a preflight), one that is down, one that
 *     redirects, one that is slow, a stream that pauses between chunks, and a model that rejects a `tools` array.
 * It binds to 127.0.0.1 on a port the system picks, and nothing else.
 */
import http from 'node:http';
import net from 'node:net';

const ALLOWED_HEADERS = 'authorization, content-type, accept, x-api-key, anthropic-version, anthropic-dangerous-direct-browser-access, openai-organization, http-referer, x-title, anthropic-beta';

/** The product's own ports: a fake never takes one (the recorder of harness.mjs refuses them). */
const OWN_PORTS = new Set([17972, 17872, 17973, 8080, 7860, 8783, 9333]);

/** A key as a provider's error text shows it: the start, stars, the last four. */
export function maskKey(key) {
  return key.length > 12 ? `${key.slice(0, 8)}${'*'.repeat(key.length - 12)}${key.slice(-4)}` : '*'.repeat(key.length);
}

/**
 * @param {{
 *   dialect?: 'openai' | 'anthropic',
 *   key?: string,
 *   cors?: 'none' | { allowOrigins?: string[] },
 *   models?: string[],
 *   modelPages?: string[][],
 *   chunks?: string[],
 *   delayMs?: number,
 *   rejectTools?: boolean,
 *   redirectTo?: string,
 *   hangMs?: number,
 * }} [initial]
 */
export async function startFakeProvider(initial = {}) {
  const options = { dialect: 'openai', chunks: ['Hello', ' from', ' the', ' fake', ' provider'], delayMs: 120, models: ['fake-chat', 'fake-embed-1', 'fake-tts'], ...initial };
  const log = [];

  function corsHeaders(req) {
    if (options.cors === 'none') return {};
    const origin = req.headers.origin;
    if (!origin) return {};
    const allowed = options.cors?.allowOrigins;
    if (allowed && !allowed.includes(origin)) return {};
    return { 'access-control-allow-origin': origin, vary: 'origin', 'access-control-expose-headers': 'retry-after, x-request-id' };
  }

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    const entry = { method: req.method, path: url.pathname, query: Object.fromEntries(url.searchParams), headers: req.headers, body: '', at: Date.now(), firstByteAt: null, finished: false, aborted: false, chunksWritten: 0 };
    log.push(entry);
    res.on('close', () => {
      if (!entry.finished) entry.aborted = true;
    });
    const cors = corsHeaders(req);
    const json = (status, body, extra = {}) => {
      res.writeHead(status, { 'content-type': 'application/json', ...cors, ...extra });
      res.end(JSON.stringify(body));
      entry.finished = true;
    };

    if (req.method === 'OPTIONS') {
      // A server that refuses CORS says nothing to a preflight.
      if (options.cors === 'none') {
        res.writeHead(404);
        res.end();
        entry.finished = true;
        return;
      }
      res.writeHead(204, { ...cors, 'access-control-allow-methods': 'GET, POST, OPTIONS', 'access-control-allow-headers': ALLOWED_HEADERS, 'access-control-max-age': '0' });
      res.end();
      entry.finished = true;
      return;
    }

    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      entry.body = Buffer.concat(chunks).toString('utf8');
      handle();
    });

    const authOk = () => {
      if (!options.key) return true;
      return options.dialect === 'anthropic' ? req.headers['x-api-key'] === options.key : req.headers.authorization === `Bearer ${options.key}`;
    };
    const presented = () => (options.dialect === 'anthropic' ? String(req.headers['x-api-key'] ?? '') : String(req.headers.authorization ?? '').replace(/^Bearer /, ''));

    function handle() {
      const path = url.pathname;
      if (options.dialect === 'anthropic') {
        if (!req.headers['anthropic-version']) return json(400, { type: 'error', error: { type: 'invalid_request_error', message: 'anthropic-version: header is required' } });
        if (req.headers.origin && req.headers['anthropic-dangerous-direct-browser-access'] !== 'true') {
          return json(400, { type: 'error', error: { type: 'invalid_request_error', message: "CORS requests must set 'anthropic-dangerous-direct-browser-access' header" } });
        }
        if (!authOk()) return json(401, { type: 'error', error: { type: 'authentication_error', message: 'invalid x-api-key' } });
        if (req.method === 'GET' && path === '/v1/models') return anthropicModels();
        if (req.method === 'POST' && path === '/v1/messages') return anthropicMessages();
        return json(404, { type: 'error', error: { type: 'not_found_error', message: 'not found' } });
      }
      if (req.method === 'GET' && path === '/v1/models') {
        if (!authOk()) return json(401, { error: { message: `Incorrect API key provided: ${maskKey(presented())}. You can find your API key at https://platform.example/account/api-keys.`, type: 'invalid_request_error', code: 'invalid_api_key' } });
        const created = 1_700_000_000;
        return json(200, { object: 'list', data: options.models.map((id, i) => ({ id, object: 'model', created: created + i, owned_by: 'fake' })) });
      }
      if (req.method === 'POST' && path === '/v1/chat/completions') return openaiChat();
      if (req.method === 'POST' && (path === '/v1/embeddings' || path === '/v1/images/generations')) return json(200, { data: [] });
      return json(404, { error: { message: 'not found' } });
    }

    function later(fn) {
      if (options.hangMs) setTimeout(fn, options.hangMs);
      else fn();
    }

    function openaiChat() {
      if (!authOk()) {
        // As OpenAI does: no CORS headers on this one, so a browser reports "Failed to fetch".
        res.writeHead(401, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: { message: `Incorrect API key provided: ${maskKey(presented())}.`, type: 'invalid_request_error', code: 'invalid_api_key' } }));
        entry.finished = true;
        return;
      }
      if (options.redirectTo) {
        res.writeHead(302, { location: options.redirectTo, ...cors });
        res.end();
        entry.finished = true;
        return;
      }
      let body = {};
      try {
        body = JSON.parse(entry.body || '{}');
      } catch {
        return json(400, { error: { message: 'body is not JSON' } });
      }
      if (options.rejectTools && Array.isArray(body.tools)) return json(400, { error: { message: 'tools are not supported by this model', type: 'invalid_request_error' } });
      later(() => {
        if (!body.stream) {
          return json(200, { id: 'chatcmpl-fake', object: 'chat.completion', model: body.model ?? 'fake-chat', choices: [{ index: 0, message: { role: 'assistant', content: options.chunks.join('') }, finish_reason: 'stop' }], usage: { prompt_tokens: 3, completion_tokens: options.chunks.length, total_tokens: 3 + options.chunks.length } });
        }
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', ...cors });
        entry.firstByteAt = Date.now();
        streamOut(
          options.chunks.map((text) => `data: ${JSON.stringify({ id: 'chatcmpl-fake', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content: text } }] })}\n\n`),
          'data: [DONE]\n\n',
        );
      });
    }

    function anthropicModels() {
      const pages = options.modelPages ?? [options.models];
      const after = url.searchParams.get('after_id');
      let index = 0;
      if (after) {
        const found = pages.findIndex((page) => page.includes(after));
        index = found < 0 ? pages.length : found + 1;
      }
      const page = pages[index] ?? [];
      return json(200, {
        data: page.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
        has_more: index < pages.length - 1,
        first_id: page[0] ?? null,
        last_id: page[page.length - 1] ?? null,
      });
    }

    function anthropicMessages() {
      let body = {};
      try {
        body = JSON.parse(entry.body || '{}');
      } catch {
        return json(400, { type: 'error', error: { type: 'invalid_request_error', message: 'body is not JSON' } });
      }
      later(() => {
        if (!body.stream) {
          return json(200, { id: 'msg_fake', type: 'message', role: 'assistant', model: body.model ?? 'fake-claude', content: [{ type: 'text', text: options.chunks.join('') }], stop_reason: 'end_turn', usage: { input_tokens: 3, output_tokens: options.chunks.length } });
        }
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', ...cors });
        entry.firstByteAt = Date.now();
        const event = (name, data) => `event: ${name}\ndata: ${JSON.stringify({ type: name, ...data })}\n\n`;
        streamOut(
          [
            event('message_start', { message: { id: 'msg_fake', type: 'message', role: 'assistant', model: body.model ?? 'fake-claude', content: [], usage: { input_tokens: 3, output_tokens: 0 } } }),
            event('content_block_start', { index: 0, content_block: { type: 'text', text: '' } }),
            ...options.chunks.map((text) => event('content_block_delta', { index: 0, delta: { type: 'text_delta', text } })),
            event('content_block_stop', { index: 0 }),
            event('message_delta', { delta: { stop_reason: 'end_turn' }, usage: { output_tokens: options.chunks.length } }),
          ],
          event('message_stop', {}),
        );
      });
    }

    /** Write the parts one at a time with the gap between them, so a test can see they arrived one at a time. */
    function streamOut(parts, last) {
      let i = 0;
      const next = () => {
        if (res.destroyed || entry.aborted) return;
        if (i < parts.length) {
          res.write(parts[i++]);
          entry.chunksWritten = i;
          setTimeout(next, options.delayMs);
        } else {
          res.write(last);
          res.end();
          entry.finished = true;
        }
      };
      next();
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    // A port the system picks, and never one of the product's own.
    const tryListen = () =>
      server.listen(0, '127.0.0.1', () => {
        if (OWN_PORTS.has(server.address().port)) return server.close(tryListen);
        resolve();
      });
    tryListen();
  });
  const port = server.address().port;

  return {
    port,
    origin: `http://127.0.0.1:${port}`,
    /** The API base a record for it has. */
    baseUrl: `http://127.0.0.1:${port}/v1`,
    log,
    /** Change how it behaves from now on. */
    set(patch) {
      Object.assign(options, patch);
    },
    /** The requests that reached it, by predicate. */
    requests: (predicate = () => true) => log.filter(predicate),
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

/**
 * A port nothing listens on: the address of a server that is down. It is a port the system just gave out and took back, so
 * it is refused at once.
 */
export async function deadPort() {
  for (let attempt = 0; attempt < 20; attempt++) {
    const port = await new Promise((resolve, reject) => {
      const s = net.createServer();
      s.once('error', reject);
      s.listen(0, '127.0.0.1', () => {
        const p = s.address().port;
        s.close(() => resolve(p));
      });
    });
    if (!OWN_PORTS.has(port)) return port;
  }
  throw new Error('no free port');
}

/**
 * A server that records everything that reaches it and answers with CORS open: where a request must never arrive. `origin` uses
 * `localhost` (a different host from the fakes' 127.0.0.1), so a request that reached it went somewhere it should not.
 */
export async function startCollector() {
  const log = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      log.push({ method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks).toString('utf8') });
      res.writeHead(200, { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'content-type': 'application/json' });
      res.end('{}');
    });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  return {
    port,
    origin: `http://localhost:${port}`,
    log,
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
