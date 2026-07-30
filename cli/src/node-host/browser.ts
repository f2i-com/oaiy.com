/**
 * Native browser automation for the Node host — the `browser_v2_*` commands the
 * core-browser module's chromium mode calls. Backed by Playwright (lazy-loaded
 * so the CLI only needs it when a flow actually drives a browser).
 *
 * Contracts mirror the desktop `oaiy-browser` plugin (see core-browser/runtime.ts
 * BrowserSessionV2 / BrowserNavigateResult / BrowserScreenshotResult), so the
 * shared module consumes the results unchanged.
 */
import path from 'node:path';
import fsp from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { resolvePath, ensureDir, fsConfinementOn } from './context';
import { assertHostPublic, assertUrlAllowed } from './ssrf-guard';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = Record<string, any>;

interface Session {
  browser: any;
  context: any;
  page: any;
}

const sessions = new Map<string, Session>();

/** SSRF guard active for browser navigation — on under worker confinement, off when the
 *  operator opts into local-network access (mirrors http_request's allowLocal). Without it a
 *  flow could use headless Chromium as an unguarded SSRF proxy to internal/metadata services. */
const browserGuardOn = (): boolean =>
  fsConfinementOn() && process.env.OAIY_HTTP_ALLOW_LOCAL !== '1';

// Lazy Playwright load (kept external from the esbuild bundle).
let chromiumLauncher: any = null;
async function getChromium(): Promise<any> {
  if (!chromiumLauncher) {
    const pw = await import('playwright');
    chromiumLauncher = (pw as any).chromium;
  }
  return chromiumLauncher;
}

function sess(id: string): Session {
  const s = sessions.get(id);
  if (!s) throw new Error(`[browser] session not found: ${id}`);
  return s;
}

function normCookie(c: Any): Any {
  const out: Any = { name: c.name, value: c.value };
  if (c.url) out.url = c.url;
  // Playwright requires either a url OR (domain AND path). A domain-only cookie
  // (the most common "malformed" case) is rescued by defaulting path to '/'.
  if (c.domain) {
    out.domain = c.domain;
    out.path = c.path ?? '/';
  } else if (c.path) {
    out.path = c.path;
  }
  if (c.expires != null) out.expires = c.expires;
  if (c.secure != null) out.secure = c.secure;
  if (c.httpOnly != null) out.httpOnly = c.httpOnly;
  if (c.sameSite) out.sameSite = c.sameSite;
  return out;
}

export async function createSession(args: Any): Promise<Any> {
  const config = (args.config ?? {}) as Any;
  // Bound concurrent Chromium sessions so an untrusted flow input (e.g. a large array
  // driving a forEach of createSession) can't spawn unbounded browser processes and
  // exhaust the host. Configurable via OAIY_BROWSER_MAX_SESSIONS.
  const maxSessions = Math.max(1, Math.trunc(Number(process.env.OAIY_BROWSER_MAX_SESSIONS) || 16));
  if (sessions.size >= maxSessions) {
    throw new Error(
      `[browser] too many open sessions (${sessions.size}/${maxSessions}); close some first`,
    );
  }
  const chromium = await getChromium();
  // On a server we always run headless regardless of the requested mode.
  const browser = await chromium.launch({ headless: config.headless !== false });

  // If anything after launch throws, close the browser so we don't leak a
  // Chromium process for every failed createSession.
  try {
    const ctxOpts: Any = {};
    // Worker mode: don't let attacker page JS auto-save downloads into Chromium's temp dir
    // (outside the sandbox roots) and fill the host disk — there's no download consumer anyway.
    if (fsConfinementOn()) ctxOpts.acceptDownloads = false;
    if (config.userAgent) ctxOpts.userAgent = config.userAgent;
    if (config.viewport) {
      ctxOpts.viewport = {
        width: config.viewport.width ?? 1280,
        height: config.viewport.height ?? 800,
      };
    }
    if (config.headers) ctxOpts.extraHTTPHeaders = config.headers;
    if (config.proxy?.server) {
      ctxOpts.proxy = {
        server: config.proxy.server,
        username: config.proxy.username,
        password: config.proxy.password,
      };
    }
    const context = await browser.newContext(ctxOpts);
    // SSRF guard: under worker confinement, re-validate every request (initial navigation,
    // redirect hops, AND sub-resources) against the private-IP block list, so a flow can't
    // drive Chromium to an internal/loopback/cloud-metadata service the http_request guard
    // already blocks. Local schemes (data:/blob:/about:) pass; file:/others are blocked.
    if (browserGuardOn()) {
      await context.route('**/*', async (route: Any) => {
        let u: URL;
        try {
          u = new URL(route.request().url());
        } catch {
          return route.abort('blockedbyclient');
        }
        if (u.protocol === 'http:' || u.protocol === 'https:') {
          try {
            await assertHostPublic(u.hostname);
            return route.continue();
          } catch {
            return route.abort('blockedbyclient');
          }
        }
        if (u.protocol === 'data:' || u.protocol === 'blob:' || u.protocol === 'about:') {
          return route.continue();
        }
        return route.abort('blockedbyclient');
      });
    }
    if (Array.isArray(config.cookies) && config.cookies.length) {
      // Add cookies one at a time — addCookies() is all-or-nothing, so a single
      // malformed cookie would otherwise drop the entire batch (incl. valid auth
      // cookies). Log the rejected cookie's NAME only (never the value).
      for (const c of config.cookies.map(normCookie)) {
        try {
          await context.addCookies([c]);
        } catch (e) {
          console.warn(
            `[node-host] [browser] skipped cookie ${c.name}: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      }
    }
    const page = await context.newPage();
    const id = `pw-${randomUUID()}`;
    sessions.set(id, { browser, context, page });
    return { id, mode: 'chromium', createdAt: new Date().toISOString() };
  } catch (e) {
    await browser.close().catch(() => {});
    throw e;
  }
}

export async function goto(args: Any): Promise<Any> {
  const { page } = sess(args.sessionId);
  // Reject a navigation to a non-http(s) or internal/loopback/metadata URL up front (clean
  // error); the per-request route guard installed at createSession also covers redirects +
  // sub-resources. Only under worker confinement (operator can opt out with OAIY_HTTP_ALLOW_LOCAL).
  if (browserGuardOn()) await assertUrlAllowed(String(args.url ?? ''));
  const options = (args.options ?? {}) as Any;
  const resp = await page.goto(args.url, {
    waitUntil: options.waitUntil ?? 'load',
    timeout: options.timeoutMs ?? 30000,
  });
  if (options.settleMs > 0) await new Promise((r) => setTimeout(r, options.settleMs));
  let title = '';
  try {
    title = await page.title();
  } catch {
    /* ignore */
  }
  return { finalUrl: page.url(), status: resp ? resp.status() : undefined, title };
}

const EVAL_TIMEOUT_MS = Number(process.env.OAIY_BROWSER_EVAL_TIMEOUT_MS) || 30000;

export async function evaluateJson(args: Any): Promise<unknown> {
  const { page } = sess(args.sessionId);
  // Bound the flow-controlled in-page script so a non-returning one fails fast
  // instead of hanging the whole run for the 45-min job budget. (Promise.race
  // doesn't cancel the renderer — a sync infinite loop still needs the session
  // torn down — but it lets the engine return/abort promptly.)
  const t =
    typeof args.options?.timeoutMs === 'number' && args.options.timeoutMs > 0
      ? args.options.timeoutMs
      : EVAL_TIMEOUT_MS;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const guard = new Promise<never>((_, rej) => {
    timer = setTimeout(() => rej(new Error(`[browser] evaluate timed out after ${t}ms`)), t);
  });
  try {
    return await Promise.race([page.evaluate(args.script), guard]);
  } finally {
    clearTimeout(timer);
  }
}

export async function getHtml(args: Any): Promise<string> {
  return sess(args.sessionId).page.content();
}

export async function getText(args: Any): Promise<string> {
  return sess(args.sessionId).page.evaluate(
    "document.body ? document.body.innerText : ''",
  );
}

export async function getTitle(args: Any): Promise<string | null> {
  try {
    return await sess(args.sessionId).page.title();
  } catch {
    return null;
  }
}

export async function getUrl(args: Any): Promise<string | null> {
  try {
    return sess(args.sessionId).page.url();
  } catch {
    return null;
  }
}

export async function waitForSelector(args: Any): Promise<Any> {
  const { page } = sess(args.sessionId);
  const timeout = args.timeoutMs ?? args.options?.timeoutMs ?? 30000;
  // Contract is { success: boolean; message? } — the runtime throws when
  // !success, so returning { found } made every successful wait throw.
  try {
    await page.waitForSelector(args.selector, { timeout });
    return { success: true };
  } catch (e) {
    return { success: false, message: e instanceof Error ? e.message : String(e) };
  }
}

export async function screenshot(args: Any): Promise<Any> {
  const { page } = sess(args.sessionId);
  const o = (args.options ?? {}) as Any;
  const type = o.format === 'jpeg' ? 'jpeg' : 'png';
  const shotOpts: Any = { fullPage: !!o.fullPage, type };
  if (type === 'jpeg' && o.quality != null) shotOpts.quality = o.quality;

  let buf: Buffer;
  if (o.selector) {
    const el = await page.$(o.selector);
    if (!el) throw new Error(`[browser] screenshot selector not found: ${o.selector}`);
    try {
      buf = await el.screenshot(shotOpts);
    } finally {
      await el.dispose().catch(() => {}); // release the ElementHandle even on throw
    }
  } else {
    buf = await page.screenshot(shotOpts);
  }

  const vp = page.viewportSize() || { width: 0, height: 0 };
  const result: Any = {
    format: type,
    width: vp.width,
    height: vp.height,
    dataUrl: `data:image/${type};base64,${buf.toString('base64')}`,
  };
  if (o.savePath) {
    const real = resolvePath(o.savePath);
    ensureDir(path.dirname(real));
    await fsp.writeFile(real, buf);
    result.savedPath = real;
  }
  return result;
}

export async function getCookies(args: Any): Promise<Any[]> {
  const cookies = await sess(args.sessionId).context.cookies();
  return cookies.map((c: Any) => ({
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    expires: c.expires,
    secure: c.secure,
    httpOnly: c.httpOnly,
    sameSite: c.sameSite,
  }));
}

export async function setCookies(args: Any): Promise<null> {
  const { context } = sess(args.sessionId);
  if (Array.isArray(args.cookies)) {
    for (const c of args.cookies.map(normCookie)) {
      try {
        await context.addCookies([c]);
      } catch (e) {
        console.warn(
          `[node-host] [browser] skipped cookie ${c.name}: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
  }
  return null;
}

export async function close(args: Any): Promise<null> {
  const s = sessions.get(args.sessionId);
  if (s) {
    try {
      await s.context.close();
    } catch {
      /* ignore */
    }
    try {
      await s.browser.close();
    } catch {
      /* ignore */
    }
    sessions.delete(args.sessionId);
  }
  return null;
}

/** Best-effort teardown of any leaked sessions at process exit. */
export async function closeAll(): Promise<void> {
  for (const id of [...sessions.keys()]) await close({ sessionId: id });
}
