/**
 * Core Browser Module Runtime
 *
 * Provides browser automation: HTTP requests, page control, data extraction.
 *
 * Modes:
 *   - chromium - native CDP backend (browser_v2_* commands)
 *   - webview  - Tauri embedded view (legacy webview_* commands)
 *   - http     - plain fetch (legacy http_request command)
 *
 * The v2 methods on `Browser` (createSessionV2, goto, evaluateJson, ...)
 * inspect the session mode and dispatch to the right command. Extraction
 * helpers are page scripts that run via `evaluateJson`.
 *
 * Native Rust plugin: oaiy-browser (see native/src/lib.rs)
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';
import { MODULE_CLEANUP } from 'oaiy-core/src/module-types';
import {
  extractEmailsScript,
  extractPhonesScript,
  type PhoneRegion,
  extractTextScript,
  type TextExtractionOptions,
  extractLinksScript,
  type LinkExtractionOptions,
  extractImagesScript,
  type ImageExtractionOptions,
  extractColorsScript,
  type ColorExtractionOptions,
  extractLogoScript,
  extractMetadataScript,
  extractJsonLdScript,
} from './extractors';

// Per-job method factory: methods + session maps + the companion client all close
// over THIS job's ctx (no module-level singletons), so concurrent jobs don't
// cross-contaminate. A per-job cleanup (MODULE_CLEANUP, attached to the returned
// methods) closes this job's Chromium/webview sessions at teardown.
function createBrowserMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

// True in the standalone WEB build: the tauri-shim sets this marker and provides a
// (truthy) ctx.tauri, so `!ctx.tauri` can NOT detect "no native browser host" there.
// When true, chromium browser ops route to the OAIY Companion's Playwright server over
// HTTP. Desktop (native plugin) and the Node CLI (node-host Playwright) leave the
// marker unset, so they keep the native ctx.tauri path. Evaluated per call — the shim
// sets the marker before this lazily-loaded module ever runs.
const isWebShim = (): boolean =>
  typeof window !== 'undefined' &&
  (window as { __OAIY_WEB_SHIM__?: boolean }).__OAIY_WEB_SHIM__ === true;

// =============================================================================
// SERVICE HELPERS
// =============================================================================

/**
 * Extract port number from a URL string
 * @param url The URL to extract port from (e.g., "http://127.0.0.1:8769")
 * @returns The port number or null if not found
 */
function extractPortFromUrl(url: string): number | null {
  const match = url.match(/:(\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Try to auto-start a service using ensure_service_ready_by_port (fully dynamic lookup)
 * @param port The port number to find and start the service
 * @returns The service URL or null if failed
 */
async function ensureServiceReadyByPort(port: number): Promise<string | null> {
  if (!ctx.tauri) return null;

  try {
    interface EnsureServiceResult {
      success: boolean;
      port?: number;
      error?: string;
      already_running: boolean;
    }

    ctx.log('info', `[Browser] Ensuring service on port ${port} is ready...`);

    // Use dynamic port-based lookup - finds service from services folder
    const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready_by_port', {
      port,
    });

    if (result.success && result.port) {
      const url = `http://127.0.0.1:${result.port}`;
      if (!result.already_running) {
        ctx.log('info', `[Browser] Service on port ${port} auto-started at ${url}`);
      } else {
        ctx.log('info', `[Browser] Service on port ${port} already running at ${url}`);
      }
      return url;
    } else if (result.error) {
      ctx.log('warn', `[Browser] Service on port ${port} failed to start: ${result.error}`);
    }
  } catch {
    // ensure_service_ready_by_port not available (older backend)
    ctx.log('info', `[Browser] Dynamic service lookup not available`);
  }

  return null;
}

// =============================================================================
// REQUEST LIMITS
// =============================================================================

// Maximum request body size: 10 MB
const MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024;

// Maximum response body size: 50 MB
const MAX_RESPONSE_BODY_SIZE = 50 * 1024 * 1024;

// Maximum URL length (per RFC 2616, most browsers support ~2000 chars)
const MAX_URL_LENGTH = 8192;

/**
 * Validates a URL for HTTP requests.
 * Throws an error if the URL is invalid or potentially dangerous.
 */
function validateUrl(url: string): void {
  if (!url || typeof url !== 'string') {
    throw new Error('URL is required');
  }

  // Check URL length
  if (url.length > MAX_URL_LENGTH) {
    throw new Error(`URL exceeds maximum length of ${MAX_URL_LENGTH} characters`);
  }

  // Parse and validate URL
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(url);
  } catch {
    throw new Error(`Invalid URL: ${url}`);
  }

  // Only allow http and https protocols
  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw new Error(`Invalid protocol: ${parsedUrl.protocol}. Only http and https are allowed.`);
  }

  // Check for javascript: URLs (XSS prevention - defense in depth)
  if (url.toLowerCase().includes('javascript:')) {
    throw new Error('JavaScript URLs are not allowed');
  }
}

/**
 * Validates request body size.
 */
function validateRequestBody(body: string | undefined): void {
  if (body && body.length > MAX_REQUEST_BODY_SIZE) {
    throw new Error(`Request body exceeds maximum size of ${MAX_REQUEST_BODY_SIZE / 1024 / 1024} MB`);
  }
}

/**
 * Truncates response body if it exceeds the maximum size.
 * Returns the body and whether it was truncated.
 */
function limitResponseSize(body: string): { body: string; truncated: boolean } {
  if (body.length > MAX_RESPONSE_BODY_SIZE) {
    return {
      body: body.substring(0, MAX_RESPONSE_BODY_SIZE),
      truncated: true,
    };
  }
  return { body, truncated: false };
}

interface BrowserSession {
  id: string;
  mode: 'http' | 'webview' | 'headless' | 'chromium';
  cookies?: Record<string, string>;
  headers?: Record<string, string>;
  userAgent?: string;
  currentUrl?: string; // Track the last URL for auto-navigation
}

// =============================================================================
// V2 TYPES (mirrors Rust types in core-browser/native/src/chromium/types.rs)
// =============================================================================

type BrowserModeV2 = 'chromium' | 'webview' | 'http';

interface BrowserViewport {
  width: number;
  height: number;
  deviceScaleFactor?: number;
  isMobile?: boolean;
}

type ProxyKind ='none' | 'http' | 'socks5' | 'mobile';

interface BrowserProxyConfig {
  type: ProxyKind;
  host?: string;
  port?: number;
  username?: string;
  password?: string;
  mobileDeviceId?: string;
}

type BlockedResource ='image' | 'font' | 'media' | 'stylesheet' | 'script';

interface BrowserCookieV2 {
  name: string;
  value: string;
  domain?: string;
  path?: string;
  expires?: number;
  secure?: boolean;
  httpOnly?: boolean;
  sameSite?: string;
}

interface BrowserSessionConfigV2 {
  mode?: BrowserModeV2;
  headless?: boolean;
  profile?: string;
  persistProfile?: boolean;
  userAgent?: string;
  viewport?: BrowserViewport;
  proxy?: BrowserProxyConfig;
  headers?: Record<string, string>;
  cookies?: BrowserCookieV2[];
  blockResources?: BlockedResource[];
  timeoutMs?: number;
  stealth?: boolean;
  /**
   * For HTTP-mode sessions: a Chrome TLS-fingerprint profile to use on
   * the wire. When set, http-mode goto/getHtml route through rquest
   * (BoringSSL) so the connection looks like a real Chrome
   * ClientHello + HTTP/2 SETTINGS instead of node/Tauri fetch.
   *
   * Currently supported values: "chrome131" (default), "chrome124".
   */
  impersonate?: string;
}

interface BrowserSessionV2 {
  id: string;
  mode: BrowserModeV2;
  processId?: string;
  pageId?: string;
  profile?: string;
  currentUrl?: string;
  userAgent?: string;
  proxy?: BrowserProxyConfig;
  createdAt: string;
}

interface BrowserGotoOptions {
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle';
  settleMs?: number;
  timeoutMs?: number;
}

interface BrowserNavigateResult {
  finalUrl: string;
  status?: number;
  title?: string;
  html?: string;
}

interface BrowserScreenshotOptions {
  fullPage?: boolean;
  format?: 'png' | 'jpeg';
  quality?: number;
  selector?: string;
  savePath?: string;
}

interface BrowserScreenshotResult {
  dataUrl?: string;
  savedPath?: string;
  format: string;
  width: number;
  height: number;
}

interface BrowserEngineInfo {
  engine: string;
  version?: string;
  source: 'custom' | 'managed-updated' | 'bundled' | 'system';
  binaryPath: string;
  platform: string;
  sha256?: string;
  verified: boolean;
}

interface BrowserUpdateCheckResult {
  currentVersion?: string;
  latestVersion?: string;
  updateAvailable: boolean;
  message: string;
}

interface ControlResult {
  success: boolean;
  data?: string;
  error?: string;
  screenshotData?: string;
}

/**
 * Shared response type for WebView plugin invoke calls
 */
interface WebViewResult {
  success: boolean;
  session_id?: string;
  data?: string;
  error?: string;
}

/**
 * HTTP response type for native HTTP requests
 */
interface HttpResult {
  status: number;
  headers: Record<string, string>;
  body: string;
  url: string;
  bodyIsBase64?: boolean;
}

// Local cache for sessions (mirrors native state)
const sessions: Map<string, BrowserSession> = new Map();

/**
 * Create a new browser session
 */
async function createSession(
  _profile: string,
  mode: string,
  customUserAgent: string,
  customHeaders: string,
  initialCookies: string,
  nodeId: string
): Promise<BrowserSession> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Browser] Creating session (${mode} mode)`);

  // Parse custom headers
  let headers: Record<string, string> = {};
  if (customHeaders) {
    try {
      headers = JSON.parse(customHeaders);
    } catch {
      ctx.log('warn', '[Browser] Invalid custom headers JSON');
    }
  }

  // Parse cookies
  let cookies: Record<string, string> = {};
  if (initialCookies) {
    try {
      cookies = JSON.parse(initialCookies);
    } catch {
      ctx.log('warn', '[Browser] Invalid cookies JSON');
    }
  }

  // For WebView/Headless modes, use native webview commands if available
  if ((mode === 'webview' || mode === 'headless') && ctx.tauri) {
    try {
      // Use browser plugin webview_create command
      // Pass cookies and headers so they can be used for HTTP fetch fallback
      const result = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_create', {
        config: {
          profile: _profile || 'default',
          custom_user_agent: customUserAgent || undefined,
          viewport_width: 1280,
          viewport_height: 800,
          cookies: cookies,
          headers: headers,
        },
      });

      if (!result.success || !result.session_id) {
        throw new Error(result.error || 'Failed to create webview session');
      }

      const session: BrowserSession = {
        id: result.session_id,
        mode: mode as 'webview' | 'headless',
        cookies,
        headers,
        userAgent: customUserAgent,
      };
      sessions.set(session.id, session);
      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[Browser] Native session created: ${session.id}`);
      return session;
    } catch (error) {
      ctx.log('warn', `[Browser] Native session failed, falling back to HTTP mode: ${error}`);
      // Fall through to HTTP mode
    }
  }

  // HTTP mode or fallback - always use 'http' mode since native failed
  const sessionId = `session-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  const session: BrowserSession = {
    id: sessionId,
    mode: 'http', // Force HTTP mode since native session failed or wasn't requested
    cookies,
    headers,
    userAgent: customUserAgent,
  };

  sessions.set(sessionId, session);
  ctx.onNodeStatus?.(nodeId, 'completed');
  ctx.log('success', `[Browser] Session created: ${sessionId}`);
  return session;
}

/**
 * Parse Set-Cookie header and update session cookies
 */
function updateSessionCookies(session: BrowserSession, headers: Record<string, string>): void {
  // Look for set-cookie header (case-insensitive)
  const setCookieHeader = headers['set-cookie'] || headers['Set-Cookie'];
  if (!setCookieHeader) return;

  // Initialize cookies if not present
  if (!session.cookies) {
    session.cookies = {};
  }

  // Parse Set-Cookie values (may be comma or newline separated for multiple cookies)
  const cookieStrings = setCookieHeader.split(/[,\n]/).map(s => s.trim()).filter(s => s);

  for (const cookieStr of cookieStrings) {
    // Cookie format: name=value; optional-attributes
    const parts = cookieStr.split(';');
    if (parts.length === 0) continue;

    const nameValue = parts[0].trim();
    const eqIndex = nameValue.indexOf('=');
    if (eqIndex === -1) continue;

    const name = nameValue.substring(0, eqIndex).trim();
    const value = nameValue.substring(eqIndex + 1).trim();

    // Skip empty cookie names or attributes that got parsed as cookies
    if (!name || name.toLowerCase() === 'path' || name.toLowerCase() === 'domain' ||
        name.toLowerCase() === 'expires' || name.toLowerCase() === 'max-age' ||
        name.toLowerCase() === 'secure' || name.toLowerCase() === 'httponly' ||
        name.toLowerCase() === 'samesite') {
      continue;
    }

    session.cookies[name] = value;
    // Log cookie name only, not value (security - session tokens are sensitive)
    ctx.log('info', `[Browser] Cookie set: ${name}`);
  }

  // Update the session in the map
  sessions.set(session.id, session);
}

/**
 * Make an HTTP request
 */
async function request(
  url: string,
  method: string,
  headersJson: string,
  body: string,
  sessionId: string | undefined,
  nodeId: string,
  allowLocalNetwork: boolean = false
): Promise<{
  status: number;
  headers: Record<string, string>;
  body: string;
}> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[Browser] Aborted by user before request');
    throw new Error('Operation aborted by user');
  }

  // Validate URL and request body before proceeding
  validateUrl(url);
  validateRequestBody(body);

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Browser] ${method} request to ${url}${allowLocalNetwork ? ' (local network allowed)' : ''}`);

  // Get session if provided
  const session = sessionId ? sessions.get(sessionId) : undefined;

  // Build headers
  let headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  // Add session headers
  if (session?.headers) {
    headers = { ...headers, ...session.headers };
  }

  // Add session cookies as Cookie header
  if (session?.cookies && Object.keys(session.cookies).length > 0) {
    const cookieStr = Object.entries(session.cookies)
      .map(([k, v]) => `${k}=${v}`)
      .join('; ');
    headers['Cookie'] = cookieStr;
    // Log that cookies are being sent without exposing actual values
    ctx.log('info', `[Browser] Sending cookies (${Object.keys(session.cookies).length} cookies)`);
  }

  // Add custom headers
  if (headersJson) {
    try {
      const customHeaders = JSON.parse(headersJson);
      headers = { ...headers, ...customHeaders };
    } catch {
      ctx.log('warn', '[Browser] Invalid headers JSON');
    }
  }

  try {
    // Use Tauri's native HTTP to bypass CORS restrictions
    if (ctx.tauri) {
      const result = await ctx.tauri.invoke<HttpResult>('http_request', {
        request: {
          url,
          method,
          headers,
          body: body || null,
          follow_redirects: true,
          max_redirects: 10,
          allow_private_networks: allowLocalNetwork,
        }
      });

      // Update session cookies from Set-Cookie header
      if (session && result.headers) {
        updateSessionCookies(session, result.headers);
      }

      // Track the URL in session for auto-navigation in browser_control
      if (session) {
        session.currentUrl = result.url || url;
      }

      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[Browser] Response: ${result.status}`);

      // Handle Base64-encoded binary responses (images, PDFs, etc.)
      let responseBody = result.body;
      if (result.bodyIsBase64) {
        // Get content-type for Data URL format
        const contentType = result.headers['content-type'] || result.headers['Content-Type'] || 'application/octet-stream';
        // Format as Data URL so downstream nodes (Image Save, etc.) can use it directly
        responseBody = `data:${contentType};base64,${result.body}`;
        ctx.log('info', `[Browser] Binary response converted to Data URL (${contentType})`);
      }

      // Limit response size
      const limited = limitResponseSize(responseBody);
      if (limited.truncated) {
        ctx.log('warn', `[Browser] Response truncated to ${MAX_RESPONSE_BODY_SIZE / 1024 / 1024} MB`);
      }

      return {
        status: result.status,
        headers: result.headers,
        body: limited.body,
      };
    }

    // Fallback to browser fetch (will hit CORS issues for cross-origin)
    const response = await ctx.fetch(url, {
      method,
      headers,
      body: body || undefined,
    });

    const responseBody = await response.text();
    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    // Update session cookies from Set-Cookie header
    if (session) {
      updateSessionCookies(session, responseHeaders);
    }

    // Limit response size
    const limited = limitResponseSize(responseBody);
    if (limited.truncated) {
      ctx.log('warn', `[Browser] Response truncated to ${MAX_RESPONSE_BODY_SIZE / 1024 / 1024} MB`);
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[Browser] Response: ${response.status}`);

    return {
      status: response.status,
      headers: responseHeaders,
      body: limited.body,
    };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Browser] Request failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract data from HTML/JSON content
 */
function extract(
  content: string,
  extractType: string,
  pattern: string,
  attribute: string
): string | string[] {
  ctx.log('info', `[Browser] Extracting with ${extractType}: ${pattern}`);

  if (extractType === 'jsonpath') {
    try {
      const data = JSON.parse(content);
      // Simple JSONPath implementation that handles dot notation and array brackets
      // Matches: property names OR [index]
      const pathParts = pattern.replace(/^\$\.?/, '').match(/([^[.\]]+|\[\d+\])/g) || [];
      let result: unknown = data;
      
      for (const part of pathParts) {
        if (result === null || result === undefined) break;
        
        // Handle array index [n]
        if (part.startsWith('[') && part.endsWith(']')) {
          const index = parseInt(part.slice(1, -1), 10);
          if (Array.isArray(result)) {
            result = result[index];
          } else {
            // Trying to index non-array
            result = undefined;
          }
        } else {
          // Handle property name
          // Check if we are iterating over an array (wildcard behavior implicit in some jsonpath impls, 
          // but here we stick to strict property access unless it's explicit array map which we don't fully support yet 
          // except via this simple logic from original code which handled wildcard/map strangely)
          // The original code had: if (Array.isArray(result)) { ... map ... }
          // But standard dot notation on array usually means "map" or is invalid.
          // Let's support the "map" behavior if it's an array, to preserve existing feature if any.
          const key = part;
          if (Array.isArray(result)) {
             const index = parseInt(key, 10);
             if (!isNaN(index)) {
               result = result[index];
             } else if (key === '*') {
               // result = result; // No-op
             } else {
               // Map over array
               result = result.map((item: any) => item ? item[key] : undefined);
             }
          } else if (typeof result === 'object') {
            result = (result as Record<string, unknown>)[key];
          }
        }
      }
      
      if (Array.isArray(result)) {
        return result.map(String);
      }
      return String(result);
    } catch {
      ctx.log('error', '[Browser] Invalid JSON for JSONPath extraction');
      return '';
    }
  }

  if (extractType === 'regex') {
    // Validate pattern to prevent ReDoS attacks
    const MAX_PATTERN_LENGTH = 500;
    const MAX_RESULTS = 10000;
    const MAX_ITERATIONS = 100000;

    if (!pattern) {
      ctx.log('error', '[Browser] Empty regex pattern');
      return [];
    }

    if (pattern.length > MAX_PATTERN_LENGTH) {
      ctx.log('error', `[Browser] Regex pattern too long (max ${MAX_PATTERN_LENGTH} chars)`);
      return [];
    }

    try {
      const regex = new RegExp(pattern, 'g');
      const results: string[] = [];
      let match: RegExpExecArray | null;
      let iterations = 0;

      // Use exec() to get capture groups, not match() which only returns full matches
      // Limit iterations to prevent ReDoS
      while ((match = regex.exec(content)) !== null) {
        iterations++;
        if (iterations > MAX_ITERATIONS) {
          ctx.log('warn', '[Browser] Regex extraction exceeded iteration limit');
          break;
        }

        // If there are capture groups, return the first capture group
        // Otherwise return the full match
        if (match.length > 1) {
          results.push(match[1]); // First capture group
        } else {
          results.push(match[0]); // Full match
        }

        // Limit number of results
        if (results.length >= MAX_RESULTS) {
          ctx.log('warn', '[Browser] Regex extraction reached result limit');
          break;
        }

        // Prevent infinite loop on zero-length matches
        if (match[0].length === 0) {
          regex.lastIndex++;
        }
      }

      // Return single value if only one match, array if multiple
      if (results.length === 1) {
        return results[0];
      }
      return results;
    } catch {
      ctx.log('error', '[Browser] Invalid regex pattern');
      return [];
    }
  }

  if (extractType === 'selector' || extractType === 'css_selector') {
    // CSS selector extraction requires DOM parsing
    // In a browser environment, we can use DOMParser
    if (!pattern) {
      ctx.log('error', '[Browser] Empty CSS selector pattern');
      return [];
    }
    if (typeof DOMParser !== 'undefined') {
      try {
        const parser = new DOMParser();
        const doc = parser.parseFromString(content, 'text/html');
        ctx.log('info', `[Browser] Extracting with selector: ${pattern}`);
        const elements = doc.querySelectorAll(pattern);
        const results: string[] = [];
        elements.forEach((el) => {
          if (attribute === 'text') {
            results.push(el.textContent || '');
          } else if (attribute === 'html') {
            results.push(el.innerHTML);
          } else {
            results.push(el.getAttribute(attribute) || '');
          }
        });
        return results.length === 1 ? results[0] : results;
      } catch {
        ctx.log('error', '[Browser] Invalid CSS selector');
        return [];
      }
    }
  }

  return '';
}

/**
 * Control browser (click, type, navigate, etc.)
 */
async function control(
  sessionParam: BrowserSession,
  action: string,
  target: string,
  value: string | undefined,
  timeout: number,
  nodeId: string
): Promise<string> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[Browser] Aborted by user before control action');
    throw new Error('Operation aborted by user');
  }

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Browser] Control action: ${action} on ${target}`);

  // Get the live session from the Map (has currentUrl from browser_request)
  // Fall back to the passed session if not found
  const session = sessions.get(sessionParam.id) || sessionParam;

  // For HTTP mode, only 'goto' is supported (as a fetch request)
  if (session.mode === 'http') {
    if (action === 'goto' && target) {
      const response = await ctx.fetch(target);
      const body = await response.text();
      ctx.onNodeStatus?.(nodeId, 'completed');
      return body;
    }
    ctx.onNodeStatus?.(nodeId, 'error');
    ctx.log('error', `[Browser] HTTP mode only supports 'goto' action. For '${action}', use WebView or Headless mode.`);
    throw new Error(`HTTP mode only supports 'goto' action. The '${action}' action requires WebView or Headless mode with the native browser plugin enabled.`);
  }

  // WebView/Headless mode - use native Tauri webview commands
  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for browser control');
  }

  try {
    let result: ControlResult;

    // Map actions to appropriate Tauri webview commands
    if (action === 'goto') {
      // Navigate to URL
      const navResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_navigate', {
        sessionId: session.id,
        url: target,
        waitFor: null,
        timeoutMs: timeout,
      });

      if (!navResult.success) {
        throw new Error(navResult.error || 'Navigation failed');
      }

      // After navigation, get the page HTML
      const htmlResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_action', {
        sessionId: session.id,
        action: 'get_html',
        selector: null,
        value: null,
      });

      result = {
        success: htmlResult.success,
        data: htmlResult.data,
        error: htmlResult.error,
      };
    } else if (action === 'screenshot') {
      // Take screenshot
      const screenshotResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_screenshot', {
        sessionId: session.id,
      });

      result = {
        success: screenshotResult.success,
        screenshotData: screenshotResult.data,
        error: screenshotResult.error,
      };
    } else if (action === 'click' || action === 'type' || action === 'scroll') {
      // Auto-navigate to the session's currentUrl if webview hasn't been navigated yet
      // This handles the case where browser_request was used (HTTP mode) but browser_control
      // needs to interact with the webview
      if (session.currentUrl) {
        ctx.log('info', `[Browser] Auto-navigating webview to ${session.currentUrl}`);
        const navResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_navigate', {
          sessionId: session.id,
          url: session.currentUrl,
          waitFor: null,
          timeoutMs: timeout,
        });

        if (!navResult.success) {
          ctx.log('warn', `[Browser] Auto-navigation failed: ${navResult.error}`);
        }
        // Clear the URL so we don't re-navigate on subsequent actions
        session.currentUrl = undefined;
      }

      // Use webview_action for click, type, scroll
      const actionResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_action', {
        sessionId: session.id,
        action,
        selector: target || null,
        value: value || null,
      });

      result = {
        success: actionResult.success,
        data: actionResult.data,
        error: actionResult.error,
      };
    } else if (action === 'get_html') {
      // Auto-navigate to the session's currentUrl if webview hasn't been navigated yet
      if (session.currentUrl) {
        ctx.log('info', `[Browser] Auto-navigating webview to ${session.currentUrl} (before get_html)`);
        const navResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_navigate', {
          sessionId: session.id,
          url: session.currentUrl,
          waitFor: null,
          timeoutMs: timeout,
        });

        if (!navResult.success) {
          ctx.log('warn', `[Browser] Auto-navigation failed: ${navResult.error}`);
        }
        // Clear the URL so we don't re-navigate on subsequent actions
        session.currentUrl = undefined;
      }

      // Get page HTML
      const htmlResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_action', {
        sessionId: session.id,
        action: 'get_html',
        selector: null,
        value: null,
      });

      result = {
        success: htmlResult.success,
        data: htmlResult.data,
        error: htmlResult.error,
      };
    } else if (action === 'evaluate') {
      // Execute JavaScript
      const evalResult = await ctx.tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_evaluate', {
        sessionId: session.id,
        script: target, // Script is passed in target parameter
      });

      result = {
        success: evalResult.success,
        data: evalResult.data,
        error: evalResult.error,
      };
    } else {
      throw new Error(`Unknown browser action: ${action}`);
    }

    if (!result.success) {
      throw new Error(result.error || `Action '${action}' failed`);
    }

    ctx.onNodeStatus?.(nodeId, 'completed');

    // Return screenshot data if available, otherwise return data
    if (result.screenshotData) {
      return result.screenshotData;
    }
    return result.data || '';
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Browser] Control failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Close a browser session
 */
async function closeSession(sessionId: string): Promise<void> {
  const session = sessions.get(sessionId);
  if (!session) return;

  // For WebView/Headless modes, close native session
  if ((session.mode === 'webview' || session.mode === 'headless') && ctx.tauri) {
    try {
      await ctx.tauri.invoke('plugin:oaiy-browser|webview_close', { sessionId: sessionId });
    } catch (err) {
      ctx.log('warn', `[Browser] Close session warning: ${err}`);
    }
  }

  sessions.delete(sessionId);
  ctx.log('info', `[Browser] Session closed: ${sessionId}`);
}

// =============================================================================
// V2 RUNTIME — chromium / unified Browser API
// =============================================================================

/**
 * For HTTP-mode sessions, extra state we keep on the TS side so subsequent
 * getHtml/getText calls don't have to refetch — we already paid for a real
 * impersonated request, may as well cache the body until the next goto.
 */
interface HttpSessionExtras {
  impersonate?: string;
  proxy?: string;
  timeoutMs?: number;
  lastHtml?: string;
  lastStatus?: number;
}
const v2Sessions: Map<string, BrowserSessionV2> = new Map();
const v2HttpExtras: Map<string, HttpSessionExtras> = new Map();

/**
 * Cap on cached body size for HTTP-mode sessions. A 50MB blob (e.g. a
 * stray HTML download) shouldn't sit in process memory until session
 * close — beyond this, getHtml will refetch instead of caching.
 */
const HTTP_BODY_CACHE_LIMIT = 4 * 1024 * 1024; // 4 MB

function requireTauri(op: string): NonNullable<RuntimeContext['tauri']> {
  if (!ctx?.tauri) {
    throw new Error(`[Browser] ${op} requires Tauri (not available in web preview)`);
  }
  return ctx.tauri;
}

// ===========================================================================
// COMPANION BROWSER TRANSPORT (browser build — no native oaiy-browser plugin)
// ===========================================================================
//
// When oaiy-web runs in a plain browser there is no Tauri host, so the
// chromium-backed `browser_v2_*` ops can't reach a native plugin. The OAIY
// Companion (the desktop sidecar) ships a managed "Playwright Browser"
// service that exposes the same operations over HTTP. Whenever `!ctx.tauri`
// we route every chromium-session op to that service instead.
//
// `ctx.tauri` is constant for the lifetime of a renderer, so a session
// created via the companion is *always* driven via the companion — session
// ids never cross transports, and no per-session bookkeeping is needed
// beyond what `v2Sessions` already tracks.
//
// The companion's HTTP API binds a fixed loopback port (mirrors
// `companionDetection` in the app shell — kept inline here so this bundled
// module stays self-contained). The Playwright server itself runs on a
// separate port that we discover from `/api/services`.

const COMPANION_API = 'http://127.0.0.1:17972';

/** Resolved base URL of the companion's Playwright server; cached per load. */
let companionBrowserBase: string | null = null;

interface CompanionErrorPayload {
  error?: string;
}

/**
 * `fetch` a JSON endpoint with a hard timeout. Throws an `Error` carrying
 * the server's `{ error }` message on a non-2xx response.
 */
async function companionFetchJson<T = unknown>(
  url: string,
  init?: RequestInit,
  timeoutMs = 8000
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      credentials: 'omit',
      cache: 'no-store',
      signal: controller.signal,
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    });
    const text = await resp.text();
    const json = (text ? JSON.parse(text) : {}) as T & CompanionErrorPayload;
    if (!resp.ok) {
      throw new Error(json?.error || `HTTP ${resp.status}`);
    }
    return json;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Locate the companion's Playwright Browser service, make sure it's
 * running, wait until it's actually ready, and return its base URL. The
 * result is cached. Throws an *actionable* error when the companion isn't
 * reachable or the service isn't installed.
 */
async function resolveCompanionBrowserBase(): Promise<string> {
  if (companionBrowserBase) return companionBrowserBase;

  let services: Array<{ id: string; port?: number; defaultPort?: number; status?: string }>;
  try {
    const snap = await companionFetchJson<{ services?: typeof services }>(
      `${COMPANION_API}/api/services`,
      { method: 'GET' },
      2500
    );
    services = Array.isArray(snap?.services) ? snap.services : [];
  } catch {
    throw new Error(
      '[Browser] No native browser host, and the OAIY Companion is not running. ' +
        'Start the OAIY Companion to use browser nodes in the browser build.'
    );
  }

  const svc = services.find(s => s.id === 'playwright-browser');
  if (!svc) {
    throw new Error(
      '[Browser] The OAIY Companion is running but its "Playwright Browser" service ' +
        'is not installed. Open the companion → Services → install Playwright Browser.'
    );
  }
  const port = svc.port || svc.defaultPort || 17880;

  // Ask the companion to start it if it isn't already (returns immediately —
  // the readiness poll below is what actually gates us).
  try {
    await companionFetchJson(
      `${COMPANION_API}/api/services/ensure-by-port`,
      { method: 'POST', body: JSON.stringify({ port }) },
      4000
    );
  } catch {
    // Non-fatal: it may already be running, or the start call lagged.
  }

  const base = `http://127.0.0.1:${port}`;

  // The Playwright server launches Chromium on boot, which can take a few
  // seconds when the service was just started. Poll /health until it's
  // ready (≈45s budget) so the first goto doesn't race the launch.
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const h = await companionFetchJson<{ ok?: boolean; error?: string }>(
        `${base}/health`,
        { method: 'GET' },
        2000
      );
      if (h?.ok) {
        companionBrowserBase = base;
        return base;
      }
      lastErr = new Error(h?.error || 'browser not ready');
    } catch (e) {
      lastErr = e;
    }
    await new Promise(r => setTimeout(r, 750));
  }
  throw new Error(
    `[Browser] Playwright Browser service did not become ready: ${
      lastErr instanceof Error ? lastErr.message : String(lastErr)
    }`
  );
}

/**
 * Thin client for the companion's Playwright server. Each method mirrors a
 * `browser_v2_*` op so the call-site routing stays a one-liner. The server's
 * request/response shapes were deliberately designed to match these.
 */
const companionBrowser = {
  async createSession(config: BrowserSessionConfigV2): Promise<string> {
    try {
      const base = await resolveCompanionBrowserBase();
      const r = await companionFetchJson<{ sessionId?: string }>(
        `${base}/session`,
        { method: 'POST', body: JSON.stringify({ config }) },
        30000
      );
      if (!r?.sessionId) throw new Error('[Browser] companion did not return a sessionId');
      return r.sessionId;
    } catch (e) {
      // The service may have restarted on a different port — drop the cache
      // so the next session attempt re-resolves rather than needing a reload.
      companionBrowserBase = null;
      throw e;
    }
  },
  async goto(sessionId: string, url: string, options: BrowserGotoOptions): Promise<BrowserNavigateResult> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ url?: string; title?: string; status?: number }>(
      `${base}/session/${sessionId}/goto`,
      { method: 'POST', body: JSON.stringify({ url, waitUntil: options.waitUntil }) },
      (options.timeoutMs ?? 30000) + 5000
    );
    return {
      finalUrl: r.url || url,
      status: typeof r.status === 'number' ? r.status : undefined,
      title: r.title,
    };
  },
  async evaluate(sessionId: string, script: string): Promise<unknown> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ result?: unknown }>(
      `${base}/session/${sessionId}/evaluate`,
      { method: 'POST', body: JSON.stringify({ script }) },
      30000
    );
    return r.result;
  },
  async getHtml(sessionId: string): Promise<string> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ html?: string }>(`${base}/session/${sessionId}/html`, { method: 'GET' });
    return r.html ?? '';
  },
  async getTitle(sessionId: string): Promise<string> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ title?: string }>(`${base}/session/${sessionId}/title`, { method: 'GET' });
    return r.title ?? '';
  },
  async getUrl(sessionId: string): Promise<string> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ url?: string }>(`${base}/session/${sessionId}/url`, { method: 'GET' });
    return r.url ?? '';
  },
  async waitForSelector(sessionId: string, selector: string, timeoutMs: number): Promise<void> {
    const base = await resolveCompanionBrowserBase();
    await companionFetchJson(
      `${base}/session/${sessionId}/wait`,
      { method: 'POST', body: JSON.stringify({ selector, timeoutMs }) },
      timeoutMs + 5000
    );
  },
  async screenshot(sessionId: string, options: BrowserScreenshotOptions): Promise<BrowserScreenshotResult> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ dataUrl?: string }>(
      `${base}/session/${sessionId}/screenshot`,
      { method: 'POST', body: JSON.stringify({ fullPage: options.fullPage }) },
      30000
    );
    return { dataUrl: r.dataUrl, format: options.format || 'png', width: 0, height: 0 };
  },
  async getCookies(sessionId: string, domain?: string): Promise<BrowserCookieV2[]> {
    const base = await resolveCompanionBrowserBase();
    const r = await companionFetchJson<{ cookies?: BrowserCookieV2[] }>(
      `${base}/session/${sessionId}/cookies`,
      { method: 'GET' }
    );
    const all = Array.isArray(r.cookies) ? r.cookies : [];
    // The server returns every cookie in the context; filter to the
    // requested domain to match the native browser_v2_get_cookies contract.
    return domain ? all.filter(c => (c.domain || '').includes(domain)) : all;
  },
  async setCookies(sessionId: string, cookies: BrowserCookieV2[], merge: boolean): Promise<void> {
    const base = await resolveCompanionBrowserBase();
    await companionFetchJson(
      `${base}/session/${sessionId}/cookies`,
      { method: 'POST', body: JSON.stringify({ cookies, merge }) }
    );
  },
  async close(sessionId: string): Promise<void> {
    const base = await resolveCompanionBrowserBase();
    await companionFetchJson(`${base}/session/${sessionId}`, { method: 'DELETE' }, 4000).catch(() => {});
  },
};

async function createSessionV2(config: BrowserSessionConfigV2): Promise<BrowserSessionV2> {
  const mode: BrowserModeV2 = config.mode ?? 'chromium';
  ctx.log('info', `[Browser] createSessionV2 mode=${mode}`);

  if (mode === 'chromium') {
    // Browser build (no Tauri host): drive the companion's Playwright server.
    // Don't fall back to webview here — webview also needs Tauri, so its
    // error would mask the actionable "start the companion" message.
    if (isWebShim()) {
      const id = await companionBrowser.createSession(config);
      const session: BrowserSessionV2 = {
        id,
        mode: 'chromium',
        profile: config.profile,
        userAgent: config.userAgent,
        proxy: config.proxy,
        createdAt: new Date().toISOString(),
      };
      v2Sessions.set(id, session);
      ctx.log('info', `[Browser] using OAIY Companion Playwright server (session ${id})`);
      return session;
    }
    const tauri = requireTauri('createSessionV2(chromium)');
    try {
      const session = await tauri.invoke<BrowserSessionV2>('plugin:oaiy-browser|browser_v2_create_session', { config });
      v2Sessions.set(session.id, session);
      return session;
    } catch (err) {
      ctx.log('warn', `[Browser] chromium session failed (${err}); falling back to webview`);
      return createSessionV2({ ...config, mode: 'webview' });
    }
  }

  if (mode === 'webview') {
    const legacy = await createSession(
      config.profile ?? 'default',
      'webview',
      config.userAgent ?? '',
      config.headers ? JSON.stringify(config.headers) : '',
      config.cookies
        ? JSON.stringify(Object.fromEntries(config.cookies.map(c => [c.name, c.value])))
        : '',
      `v2-${Date.now()}`
    );
    const session: BrowserSessionV2 = {
      id: legacy.id,
      mode: 'webview',
      profile: config.profile,
      userAgent: config.userAgent,
      proxy: config.proxy,
      createdAt: new Date().toISOString(),
    };
    v2Sessions.set(session.id, session);
    return session;
  }

  // http mode
  const sessionId = `http-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const session: BrowserSessionV2 = {
    id: sessionId,
    mode: 'http',
    profile: config.profile,
    userAgent: config.userAgent,
    proxy: config.proxy,
    createdAt: new Date().toISOString(),
  };
  v2Sessions.set(sessionId, session);

  // Stash impersonation + proxy so http-mode goto/getHtml routes through
  // the rquest TLS-fingerprint backend when an `impersonate` profile is
  // configured, and falls back to plain ctx.fetch otherwise.
  const proxyForHttp = (() => {
    const p = config.proxy;
    if (!p || p.type === 'none') return undefined;
    if (p.host && p.port) {
      const auth = p.username && p.password ? `${p.username}:${p.password}@` : '';
      const scheme = p.type === 'socks5' ? 'socks5' : 'http';
      return `${scheme}://${auth}${p.host}:${p.port}`;
    }
    return undefined;
  })();
  v2HttpExtras.set(sessionId, {
    impersonate: config.impersonate,
    proxy: proxyForHttp,
    timeoutMs: config.timeoutMs,
  });
  return session;
}

/**
 * Internal helper — issue a Chrome-fingerprinted HTTP request through the
 * native plugin. Returns body + status + final URL. Throws if the
 * Tauri host is unavailable.
 */
async function httpImpersonateRequest(args: {
  url: string;
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  profile?: string;
  proxy?: string;
  timeoutMs?: number;
}): Promise<{ status: number; finalUrl: string; headers: Record<string, string>; body: string }> {
  const tauri = requireTauri('httpRequest');
  const result = (await tauri.invoke('plugin:oaiy-browser|browser_v2_http_request', {
    request: {
      url: args.url,
      method: args.method,
      headers: args.headers,
      body: args.body,
      options: {
        profile: args.profile,
        proxy: args.proxy,
        timeoutMs: args.timeoutMs,
      },
    },
  })) as { status: number; finalUrl: string; headers: Record<string, string>; body: string } | null;
  // In the web build ctx.tauri is the shim (truthy), so requireTauri does not
  // throw — but the shim has no handler for this native plugin command and
  // resolves it to null. Surface an actionable error instead of letting the
  // caller crash on `null.body` with an opaque "Cannot read properties of null".
  if (!result || typeof result.status !== 'number') {
    throw new Error(
      'Chrome/JA3 impersonation HTTP requires the OAIY desktop app — it is not available in the browser build. Disable impersonation to use plain HTTP, or run this flow in the OAIY Companion.',
    );
  }
  return result;
}

async function gotoV2(
  sessionId: string,
  url: string,
  options: BrowserGotoOptions = {}
): Promise<BrowserNavigateResult> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  validateUrl(url);

  if (session.mode === 'chromium') {
    if (isWebShim()) {
      const result = await companionBrowser.goto(sessionId, url, options);
      session.currentUrl = result.finalUrl;
      return result;
    }
    const tauri = requireTauri('goto');
    const result = await tauri.invoke<BrowserNavigateResult>('plugin:oaiy-browser|browser_v2_goto', {
      sessionId,
      url,
      options,
    });
    session.currentUrl = result.finalUrl;
    return result;
  }

  if (session.mode === 'webview') {
    const tauri = requireTauri('goto');
    const navResult = await tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_navigate', {
      sessionId,
      url,
      waitFor: null,
      timeoutMs: options.timeoutMs ?? 30000,
    });
    if (!navResult.success) throw new Error(navResult.error || 'navigation failed');
    if (options.settleMs && options.settleMs > 0) {
      await new Promise(r => setTimeout(r, options.settleMs));
    }
    session.currentUrl = url;
    return { finalUrl: url };
  }

  // http
  const extras = v2HttpExtras.get(sessionId);
  if (extras?.impersonate) {
    const r = await httpImpersonateRequest({
      url,
      profile: extras.impersonate,
      proxy: extras.proxy,
      timeoutMs: extras.timeoutMs ?? options.timeoutMs,
    });
    extras.lastHtml = r.body.length <= HTTP_BODY_CACHE_LIMIT ? r.body : undefined;
    extras.lastStatus = r.status;
    session.currentUrl = r.finalUrl;
    return {
      finalUrl: r.finalUrl,
      status: r.status,
      html: r.body,
    };
  }
  const response = await ctx.fetch(url);
  session.currentUrl = url;
  return {
    finalUrl: response.url || url,
    status: response.status,
    html: await response.text(),
  };
}

async function evaluateJsonV2(sessionId: string, script: string): Promise<unknown> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);

  if (session.mode === 'chromium') {
    if (isWebShim()) return companionBrowser.evaluate(sessionId, script);
    const tauri = requireTauri('evaluateJson');
    return tauri.invoke<unknown>('plugin:oaiy-browser|browser_v2_evaluate_json', { sessionId, script });
  }

  if (session.mode === 'webview') {
    const tauri = requireTauri('evaluateJson');
    const result = await tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_evaluate', {
      sessionId,
      script,
    });
    if (!result.success) throw new Error(result.error || 'evaluate failed');
    try {
      return JSON.parse(result.data || 'null');
    } catch {
      return result.data;
    }
  }

  throw new Error('[Browser] evaluateJson requires chromium or webview mode');
}

async function getHtmlV2(sessionId: string): Promise<string> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);

  if (session.mode === 'chromium') {
    if (isWebShim()) return companionBrowser.getHtml(sessionId);
    const tauri = requireTauri('getHtml');
    return tauri.invoke<string>('plugin:oaiy-browser|browser_v2_get_html', { sessionId });
  }

  if (session.mode === 'webview') {
    const tauri = requireTauri('getHtml');
    const result = await tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_action', {
      sessionId,
      action: 'get_html',
      selector: null,
      value: null,
    });
    if (!result.success) throw new Error(result.error || 'get_html failed');
    return result.data || '';
  }

  // http: re-fetch the current url
  if (!session.currentUrl) throw new Error('[Browser] http session has no currentUrl');
  const extras = v2HttpExtras.get(sessionId);
  if (extras?.lastHtml !== undefined) return extras.lastHtml;
  if (extras?.impersonate) {
    const r = await httpImpersonateRequest({
      url: session.currentUrl,
      profile: extras.impersonate,
      proxy: extras.proxy,
      timeoutMs: extras.timeoutMs,
    });
    extras.lastHtml = r.body.length <= HTTP_BODY_CACHE_LIMIT ? r.body : undefined;
    extras.lastStatus = r.status;
    return r.body;
  }
  const response = await ctx.fetch(session.currentUrl);
  return response.text();
}

async function getTextV2(sessionId: string, options: TextExtractionOptions = {}): Promise<string> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);

  if (session.mode === 'chromium' || session.mode === 'webview') {
    const value = await evaluateJsonV2(sessionId, extractTextScript(options));
    return typeof value === 'string' ? value : '';
  }

  // http: strip tags from the html
  const html = await getHtmlV2(sessionId);
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, options.maxLength ?? 10000);
}

async function getTitleV2(sessionId: string): Promise<string> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) return companionBrowser.getTitle(sessionId);
    const tauri = requireTauri('getTitle');
    const t = await tauri.invoke<string | null>('plugin:oaiy-browser|browser_v2_get_title', { sessionId });
    return t || '';
  }
  const value = await evaluateJsonV2(sessionId, '(() => document.title)()').catch(() => '');
  return typeof value === 'string' ? value : '';
}

async function getUrlV2(sessionId: string): Promise<string> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) {
      const u = await companionBrowser.getUrl(sessionId);
      return u || session.currentUrl || '';
    }
    const tauri = requireTauri('getUrl');
    const u = await tauri.invoke<string | null>('plugin:oaiy-browser|browser_v2_get_url', { sessionId });
    return u || session.currentUrl || '';
  }
  return session.currentUrl || '';
}

async function waitForSelectorV2(
  sessionId: string,
  selector: string,
  timeoutMs: number = 15000
): Promise<void> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) {
      await companionBrowser.waitForSelector(sessionId, selector, timeoutMs);
      return;
    }
    const tauri = requireTauri('waitForSelector');
    const result = await tauri.invoke<{ success: boolean; message?: string }>(
      'plugin:oaiy-browser|browser_v2_wait_for_selector',
      { sessionId, selector, timeoutMs }
    );
    if (!result.success) throw new Error(result.message || 'wait_for_selector failed');
    return;
  }
  // webview / http: poll via evaluate
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const found = await evaluateJsonV2(
        sessionId,
        `(() => !!document.querySelector(${JSON.stringify(selector)}))()`
      );
      if (found) return;
    } catch {
      // ignore and retry
    }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error(`[Browser] waitForSelector timeout: ${selector}`);
}

async function screenshotV2(
  sessionId: string,
  options: BrowserScreenshotOptions = {}
): Promise<BrowserScreenshotResult> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) return companionBrowser.screenshot(sessionId, options);
    const tauri = requireTauri('screenshot');
    return tauri.invoke<BrowserScreenshotResult>('plugin:oaiy-browser|browser_v2_screenshot', { sessionId, options });
  }
  if (session.mode === 'webview') {
    const tauri = requireTauri('screenshot');
    const result = await tauri.invoke<WebViewResult>('plugin:oaiy-browser|webview_screenshot', {
      sessionId,
    });
    if (!result.success) throw new Error(result.error || 'screenshot failed');
    return {
      dataUrl: result.data,
      format: options.format || 'png',
      width: 0,
      height: 0,
    };
  }
  throw new Error('[Browser] screenshot requires chromium or webview mode');
}

async function getCookiesV2(sessionId: string, domain?: string): Promise<BrowserCookieV2[]> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) return companionBrowser.getCookies(sessionId, domain);
    const tauri = requireTauri('getCookies');
    return tauri.invoke<BrowserCookieV2[]>('plugin:oaiy-browser|browser_v2_get_cookies', { sessionId, domain });
  }
  // http/webview: read from session.cookies dict
  const legacy = sessions.get(sessionId);
  if (!legacy?.cookies) return [];
  return Object.entries(legacy.cookies).map(([name, value]) => ({ name, value, domain }));
}

async function setCookiesV2(
  sessionId: string,
  cookies: BrowserCookieV2[],
  merge: boolean = true
): Promise<void> {
  const session = v2Sessions.get(sessionId);
  if (!session) throw new Error(`[Browser] session not found: ${sessionId}`);
  if (session.mode === 'chromium') {
    if (isWebShim()) {
      await companionBrowser.setCookies(sessionId, cookies, merge);
      return;
    }
    const tauri = requireTauri('setCookies');
    await tauri.invoke('plugin:oaiy-browser|browser_v2_set_cookies', { sessionId, cookies, merge });
    return;
  }
  // http/webview: write into session.cookies dict
  const legacy = sessions.get(sessionId);
  if (legacy) {
    if (!merge) legacy.cookies = {};
    if (!legacy.cookies) legacy.cookies = {};
    for (const c of cookies) legacy.cookies[c.name] = c.value;
  }
}

async function closeV2(sessionId: string): Promise<boolean> {
  const session = v2Sessions.get(sessionId);
  if (!session) return false;
  if (session.mode === 'chromium') {
    if (isWebShim()) {
      await companionBrowser.close(sessionId);
    } else {
      const tauri = requireTauri('close');
      await tauri.invoke('plugin:oaiy-browser|browser_v2_close_session', { sessionId });
    }
  } else if (session.mode === 'webview') {
    await closeSession(sessionId);
  }
  v2Sessions.delete(sessionId);
  v2HttpExtras.delete(sessionId);
  return true;
}

// ---------------------------------------------------------------------------
// Extraction helpers (page-context scripts via evaluateJson)
// ---------------------------------------------------------------------------

async function extractEmails(sessionId: string): Promise<string[]> {
  const result = await evaluateJsonV2(sessionId, extractEmailsScript);
  return Array.isArray(result) ? (result as string[]) : [];
}

async function extractPhones(sessionId: string, region: PhoneRegion = 'AU'): Promise<string[]> {
  const result = await evaluateJsonV2(sessionId, extractPhonesScript(region));
  return Array.isArray(result) ? (result as string[]) : [];
}

async function extractLinks(sessionId: string, options?: LinkExtractionOptions) {
  const result = await evaluateJsonV2(sessionId, extractLinksScript(options));
  return Array.isArray(result) ? result : [];
}

async function extractImages(sessionId: string, options?: ImageExtractionOptions) {
  const result = await evaluateJsonV2(sessionId, extractImagesScript(options));
  return Array.isArray(result) ? result : [];
}

interface ColorPalette {
  primary?: string;
  secondary?: string;
  accent?: string;
  palette: string[];
}

async function extractColors(sessionId: string, options?: ColorExtractionOptions): Promise<ColorPalette> {
  const result = (await evaluateJsonV2(sessionId, extractColorsScript(options))) as ColorPalette | null;
  return result ?? { palette: [] };
}

async function extractLogoCandidate(sessionId: string) {
  return evaluateJsonV2(sessionId, extractLogoScript);
}

async function extractMetaTags(sessionId: string) {
  return evaluateJsonV2(sessionId, extractMetadataScript);
}

async function extractJsonLd(sessionId: string): Promise<unknown[]> {
  const result = await evaluateJsonV2(sessionId, extractJsonLdScript);
  return Array.isArray(result) ? result : [];
}

interface BrowserWebsiteData {
  url: string;
  finalUrl: string;
  title: string;
  description?: string;
  html?: string;
  visibleText?: string;
  links: Array<{ text: string; href: string }>;
  images: Array<{ src: string; alt: string; width: number; height: number }>;
  emails: string[];
  phones: string[];
  colors: { primary?: string; secondary?: string; accent?: string; palette: string[] };
  meta: Record<string, string>;
  jsonLd: unknown[];
}

interface ExtractWebsiteDataOptions {
  includeHtml?: boolean;
  includeText?: boolean;
  maxTextLength?: number;
  region?: PhoneRegion;
}

async function extractWebsiteData(
  sessionId: string,
  options: ExtractWebsiteDataOptions = {}
): Promise<BrowserWebsiteData> {
  const [meta, links, images, emails, phones, colors, jsonLd] = await Promise.all([
    extractMetaTags(sessionId).catch(() => ({ title: '', meta: {} as Record<string, string> })),
    extractLinks(sessionId).catch(() => []),
    extractImages(sessionId).catch(() => []),
    extractEmails(sessionId).catch(() => []),
    extractPhones(sessionId, options.region ?? 'AU').catch(() => []),
    extractColors(sessionId).catch((): ColorPalette => ({ palette: [] })),
    extractJsonLd(sessionId).catch(() => []),
  ]);

  const finalUrl = await getUrlV2(sessionId);
  const html = options.includeHtml ? await getHtmlV2(sessionId).catch(() => '') : undefined;
  const visibleText = options.includeText
    ? await getTextV2(sessionId, { maxLength: options.maxTextLength }).catch(() => '')
    : undefined;

  const metaSafe = meta as { title: string; description?: string; meta: Record<string, string> };
  return {
    url: finalUrl,
    finalUrl,
    title: metaSafe.title || '',
    description: metaSafe.description,
    html,
    visibleText,
    links: links as Array<{ text: string; href: string }>,
    images: images as Array<{ src: string; alt: string; width: number; height: number }>,
    emails,
    phones,
    colors,
    meta: metaSafe.meta || {},
    jsonLd,
  };
}

// ---------------------------------------------------------------------------
// Engine resolver
// ---------------------------------------------------------------------------

async function engineStatus(): Promise<BrowserEngineInfo | null> {
  if (!ctx?.tauri) return null;
  try {
    return await ctx.tauri.invoke<BrowserEngineInfo>('plugin:oaiy-browser|browser_engine_get_status');
  } catch (err) {
    ctx.log('warn', `[Browser] engine_get_status failed: ${err}`);
    return null;
  }
}

async function checkEngineUpdates(): Promise<BrowserUpdateCheckResult | null> {
  if (!ctx?.tauri) return null;
  try {
    return await ctx.tauri.invoke<BrowserUpdateCheckResult>('plugin:oaiy-browser|browser_engine_check_updates');
  } catch (err) {
    ctx.log('warn', `[Browser] engine_check_updates failed: ${err}`);
    return null;
  }
}

async function engineSetCustomPath(path: string | null): Promise<void> {
  if (!ctx?.tauri) return;
  await ctx.tauri.invoke('plugin:oaiy-browser|browser_engine_set_custom_path', { path });
}

async function engineUseBundled(): Promise<void> {
  if (!ctx?.tauri) return;
  await ctx.tauri.invoke('plugin:oaiy-browser|browser_engine_use_bundled');
}

/**
 * Core Browser Runtime Module
 */
  const methods: Record<string, RuntimeMethod> = {
    // Legacy v1 API (kept for backward-compat with existing nodes)
    createSession,
    request,
    extract,
    control,
    closeSession,
    ensureServiceReadyByPort,

    // V2 unified API — works in chromium / webview / http
    createSessionV2,
    goto: gotoV2,
    evaluateJson: evaluateJsonV2,
    getHtml: getHtmlV2,
    getText: getTextV2,
    getTitle: getTitleV2,
    getUrl: getUrlV2,
    waitForSelector: waitForSelectorV2,
    screenshot: screenshotV2,
    getCookies: getCookiesV2,
    setCookies: setCookiesV2,
    close: closeV2,

    /**
     * Issue a single Chrome-fingerprinted HTTP request via the rquest
     * (BoringSSL) backend. Cipher list, supported groups, ALPS, HTTP/2
     * SETTINGS and header order all match real Chrome.
     *
     * Useful for fast scraping where you don't want a full browser tab,
     * but still need to defeat JA3/JA4 fingerprinting on the target.
     */
    httpRequest: httpImpersonateRequest,

    // Extraction helpers
    extractEmails,
    extractPhones,
    extractLinks,
    extractImages,
    extractColors,
    extractLogoCandidate,
    extractMetaTags,
    extractJsonLd,
    extractWebsiteData,

    // Engine control
    engineStatus,
    checkEngineUpdates,
    engineSetCustomPath,
    engineUseBundled,
  };

  // Per-job cleanup: close THIS job's sessions (Chromium processes — a real leak if
  // skipped). Attached under MODULE_CLEANUP so it isn't registered as a node method;
  // registerDynamicModule runs it at job teardown, bound to this job's session maps.
  (methods as Record<symbol, unknown>)[MODULE_CLEANUP] = async (): Promise<void> => {
    for (const sessionId of sessions.keys()) {
      await closeSession(sessionId);
    }
    for (const sessionId of v2Sessions.keys()) {
      try {
        await closeV2(sessionId);
      } catch {
        // best effort
      }
    }
  };

  return methods;
}

const CoreBrowserRuntime: RuntimeModule = {
  name: 'Browser',
  createMethods: createBrowserMethods,
  methods: {},
};

export default CoreBrowserRuntime;
