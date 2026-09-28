/**
 * Core Browser Module Compiler
 *
 * Compiles the five consolidated browser nodes into JavaScript:
 *   - browser_session  - open a Chromium / WebView / HTTP session
 *   - browser_page     - navigate to a URL and wait
 *   - browser_extract  - extract data (text/links/images/emails/phones/colors/JS)
 *                        from a session, or selector/regex/jsonpath from content
 *   - browser_action   - drive a session (click/type/scroll/wait/screenshot/cookies/close)
 *   - browser_request  - plain HTTP request (kept as-is)
 *
 * The runtime exposes a unified V2 API on `Browser` (createSessionV2, goto,
 * evaluateJson, getText, extract*, screenshot, getCookies, setCookies, close,
 * waitForSelector). For chromium-mode click/type/scroll we generate
 * page-context JavaScript and pipe it through evaluateJson.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';

const CoreBrowserCompiler: ModuleCompiler = {
  name: 'Browser',

  getNodeTypes() {
    return [
      'browser_session',
      'browser_page',
      'browser_extract',
      'browser_action',
      'browser_request',
    ];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, sanitizedId, skipVarDeclaration, escapeString } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const inputVar =
      inputs.get('default') || inputs.get('input') || inputs.get('content') || 'null';

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      // ============================================================
      // browser_session — open a session in chromium / webview / http
      // ============================================================
      case 'browser_session': {
        const mode = escapeString(String(data.mode || 'chromium'));
        const headless = data.headless !== false;
        const profile = escapeString(String(data.profile || 'default'));
        const persistProfile = data.persistProfile === true;
        const userAgent = escapeString(String(data.userAgent || ''));
        const viewportWidth = Number(data.viewportWidth) || 1365;
        const viewportHeight = Number(data.viewportHeight) || 900;
        const blockResources = JSON.stringify(
          Array.isArray(data.blockResources) ? data.blockResources : []
        );
        const headers = escapeString(String(data.headers || ''));
        const cookies = escapeString(String(data.cookies || ''));
        const stealth = data.stealth === true;
        const impersonate = escapeString(String(data.impersonate || ''));
        const timeoutMs = Number(data.timeoutMs) || 30000;
        const proxyInput = inputs.get('proxy');
        const proxyExpr = proxyInput || 'undefined';

        code += `
  let _config_${sanitizedId} = {
    mode: "${mode}",
    headless: ${headless},
    profile: "${profile}",
    persistProfile: ${persistProfile},
    userAgent: "${userAgent}" || undefined,
    viewport: { width: ${viewportWidth}, height: ${viewportHeight} },
    blockResources: ${blockResources},
    stealth: ${stealth},
    impersonate: "${impersonate}" || undefined,
    timeoutMs: ${timeoutMs},
    proxy: ${proxyExpr},
  };
  // Optional headers JSON
  let _headers_str_${sanitizedId} = "${headers}";
  if (_headers_str_${sanitizedId}) {
    try { _config_${sanitizedId}.headers = JSON.parse(_headers_str_${sanitizedId}); } catch (_) {}
  }
  // Optional cookies JSON (object map { name: value } or array of cookie objects)
  let _cookies_str_${sanitizedId} = "${cookies}";
  if (_cookies_str_${sanitizedId}) {
    try {
      let _parsed_${sanitizedId} = JSON.parse(_cookies_str_${sanitizedId});
      if (Array.isArray(_parsed_${sanitizedId})) {
        _config_${sanitizedId}.cookies = _parsed_${sanitizedId};
      } else if (_parsed_${sanitizedId} && typeof _parsed_${sanitizedId} === 'object') {
        _config_${sanitizedId}.cookies = Object.entries(_parsed_${sanitizedId}).map(([n, v]) => ({ name: n, value: String(v) }));
      }
    } catch (_) {}
  }
  ${letOrAssign}${outputVar} = await Browser.createSessionV2(_config_${sanitizedId});
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      // ============================================================
      // browser_page — navigate + optional wait, return rich page info
      // ============================================================
      case 'browser_page': {
        const sessionInput = inputs.get('session') || 'null';
        const urlInput = inputs.get('url');
        const staticUrl = escapeString(String(data.url || ''));
        const urlExpr = urlInput ? urlInput : `"${staticUrl}"`;
        const waitUntil = escapeString(String(data.waitUntil || 'load'));
        const waitForSelector = escapeString(String(data.waitForSelector || ''));
        // Use ?? not || so an explicit 0 stays 0 but undefined inherits the
        // JSON default (1000ms). Previously `|| 0` meant any unset settleMs
        // dropped to 0 and missed render-late content.
        const settleMs = Number(data.settleMs ?? 1000);
        const timeoutMs = Number(data.timeoutMs) || 30000;

        code += `
  let _session_${sanitizedId} = ${sessionInput};
  if (!_session_${sanitizedId} || !_session_${sanitizedId}.id) {
    throw new Error("[browser_page] no session connected");
  }
  let _nav_${sanitizedId} = await Browser.goto(_session_${sanitizedId}.id, ${urlExpr}, {
    waitUntil: "${waitUntil}",
    settleMs: ${settleMs},
    timeoutMs: ${timeoutMs},
  });
  // Optional wait for selector after navigation
  let _waitSel_${sanitizedId} = "${waitForSelector}";
  if (_waitSel_${sanitizedId}) {
    try {
      await Browser.waitForSelector(_session_${sanitizedId}.id, _waitSel_${sanitizedId}, ${timeoutMs});
    } catch (_we_${sanitizedId}) {
      console.log("[browser_page] (${node.id}) wait for selector failed: " + _we_${sanitizedId});
    }
  }
  // Fill html/title/url from dedicated getters so we get them in webview/http
  // mode too (Browser.goto only returns those reliably in chromium mode).
  let _html_${sanitizedId} = (_nav_${sanitizedId} && _nav_${sanitizedId}.html) || "";
  let _title_${sanitizedId} = (_nav_${sanitizedId} && _nav_${sanitizedId}.title) || "";
  let _url_${sanitizedId} = (_nav_${sanitizedId} && _nav_${sanitizedId}.finalUrl) || ${urlExpr};
  let _status_${sanitizedId} = _nav_${sanitizedId} && typeof _nav_${sanitizedId}.status === 'number' ? _nav_${sanitizedId}.status : null;
  if (!_html_${sanitizedId}) {
    try { _html_${sanitizedId} = await Browser.getHtml(_session_${sanitizedId}.id); } catch (_) {}
  }
  if (!_title_${sanitizedId}) {
    try { _title_${sanitizedId} = await Browser.getTitle(_session_${sanitizedId}.id); } catch (_) {}
  }
  let _text_${sanitizedId} = "";
  try {
    _text_${sanitizedId} = await Browser.getText(_session_${sanitizedId}.id, { maxLength: 0 });
  } catch (_te_${sanitizedId}) {}
  // Multi-output: emit suffixed variables for both handles ("session", "result").
  // Suffixed vars are always fresh declarations; the unsuffixed primary
  // respects letOrAssign so macros that pre-declare it don't double-declare.
  let ${outputVar}_result = {
    url: _url_${sanitizedId},
    title: _title_${sanitizedId},
    html: _html_${sanitizedId},
    status: _status_${sanitizedId},
    text: _text_${sanitizedId},
  };
  let ${outputVar}_session = _session_${sanitizedId};
  ${letOrAssign}${outputVar} = ${outputVar}_session;
  workflow_context["${node.id}"] = ${outputVar}_result;`;
        break;
      }

      // ============================================================
      // browser_extract — type dropdown drives the right runtime call
      // ============================================================
      case 'browser_extract': {
        const type = escapeString(String(data.type || 'text'));
        const sessionInput = inputs.get('session');
        const contentInput = inputs.get('content') || inputVar;
        const sessionExpr = sessionInput || 'null';
        const selector = escapeString(String(data.selector || ''));
        const pattern = escapeString(String(data.pattern || ''));
        const attribute = escapeString(String(data.attribute || 'text'));
        const script = escapeString(String(data.script || ''));
        const region = escapeString(String(data.region || 'GENERIC'));
        const max = Number(data.max) || 0;
        const maxLength = Number(data.maxLength) || 0;
        const sameOriginOnly = data.sameOriginOnly === true;
        const minSize = Number(data.minSize) || 0;
        const paletteSize = Number(data.paletteSize) || 5;

        // Live-page (session-driven) extraction types and content-driven types
        // need different code paths.
        const livePageTypes = new Set([
          'text', 'html', 'title', 'url',
          'links', 'images', 'emails', 'phones', 'colors', 'evaluate',
        ]);
        const isLivePage = livePageTypes.has(type);

        if (isLivePage) {
          code += `
  let _session_${sanitizedId} = ${sessionExpr};
  if (!_session_${sanitizedId} || !_session_${sanitizedId}.id) {
    throw new Error("[browser_extract:${type}] requires a Session input");
  }`;

          if (type === 'text') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.getText(_session_${sanitizedId}.id, {
    selector: "${selector}" || undefined,
    maxLength: ${maxLength},
  });`;
          } else if (type === 'html') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.getHtml(_session_${sanitizedId}.id);
  ${maxLength > 0 ? `if (typeof ${outputVar} === 'string') ${outputVar} = ${outputVar}.slice(0, ${maxLength});` : ''}`;
          } else if (type === 'title') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.getTitle(_session_${sanitizedId}.id);`;
          } else if (type === 'url') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.getUrl(_session_${sanitizedId}.id);`;
          } else if (type === 'links') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.extractLinks(_session_${sanitizedId}.id, {
    selector: "${selector}" || undefined,
    sameOriginOnly: ${sameOriginOnly},
  });
  ${max > 0 ? `if (Array.isArray(${outputVar})) ${outputVar} = ${outputVar}.slice(0, ${max});` : ''}`;
          } else if (type === 'images') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.extractImages(_session_${sanitizedId}.id, {
    selector: "${selector}" || undefined,
    minSize: ${minSize},
  });
  ${max > 0 ? `if (Array.isArray(${outputVar})) ${outputVar} = ${outputVar}.slice(0, ${max});` : ''}`;
          } else if (type === 'emails') {
            code += `
  let _emails_${sanitizedId} = await Browser.extractEmails(_session_${sanitizedId}.id);
  ${letOrAssign}${outputVar} = ${max > 0 ? `_emails_${sanitizedId}.slice(0, ${max})` : `_emails_${sanitizedId}`};`;
          } else if (type === 'phones') {
            code += `
  let _phones_${sanitizedId} = await Browser.extractPhones(_session_${sanitizedId}.id, "${region}");
  ${letOrAssign}${outputVar} = ${max > 0 ? `_phones_${sanitizedId}.slice(0, ${max})` : `_phones_${sanitizedId}`};`;
          } else if (type === 'colors') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.extractColors(_session_${sanitizedId}.id, {
    paletteSize: ${paletteSize},
  });`;
          } else if (type === 'evaluate') {
            code += `
  ${letOrAssign}${outputVar} = await Browser.evaluateJson(_session_${sanitizedId}.id, "${script}");`;
          }

          code += `
  workflow_context["${node.id}"] = ${outputVar};`;
        } else {
          // Content-driven: selector / regex / jsonpath against HTML or JSON.
          // Browser.extract takes a string, so we unwrap common envelopes
          // ({body}, {html}) and re-stringify any object input — JSONPath
          // and regex both need a textual representation.
          const patternOrSelector = type === 'selector' ? selector : pattern;
          code += `
  let _content_${sanitizedId} = ${contentInput};
  if (_content_${sanitizedId} && typeof _content_${sanitizedId} === 'object' && _content_${sanitizedId}.body) {
    _content_${sanitizedId} = _content_${sanitizedId}.body;
  }
  if (_content_${sanitizedId} && typeof _content_${sanitizedId} === 'object' && _content_${sanitizedId}.html) {
    _content_${sanitizedId} = _content_${sanitizedId}.html;
  }
  // If we still have an object (e.g. browser_request already parsed JSON),
  // stringify it so JSONPath / regex have a textual document to walk.
  if (_content_${sanitizedId} != null && typeof _content_${sanitizedId} === 'object') {
    _content_${sanitizedId} = JSON.stringify(_content_${sanitizedId});
  }
  ${letOrAssign}${outputVar} = await Browser.extract(
    String(_content_${sanitizedId} == null ? "" : _content_${sanitizedId}),
    "${type}",
    "${patternOrSelector}",
    "${attribute}"
  );
  workflow_context["${node.id}"] = ${outputVar};`;
        }
        break;
      }

      // ============================================================
      // browser_action — action dropdown drives the right runtime call
      // ============================================================
      case 'browser_action': {
        const action = escapeString(String(data.action || 'click'));
        const sessionInput = inputs.get('session') || 'null';
        const valueInput = inputs.get('value');
        const selector = escapeString(String(data.selector || ''));
        const text = escapeString(String(data.text || ''));
        const scrollTo = escapeString(String(data.scrollTo || 'bottom'));
        const format = escapeString(String(data.format || 'png'));
        const fullPage = data.fullPage === true;
        const savePath = escapeString(String(data.savePath || ''));
        const quality = Number(data.quality) || 80;
        const domain = escapeString(String(data.domain || ''));
        const cookiesJson = escapeString(String(data.cookies || '[]'));
        const merge = data.merge !== false;
        const timeoutMs = Number(data.timeoutMs) || 15000;

        // Common preamble: get session. Multi-output node — emit
        // suffixed variables for both handles ("session", "result").
        // Suffixed vars are always fresh declarations; the unsuffixed
        // primary respects letOrAssign for the macro pre-declare case.
        code += `
  let _session_${sanitizedId} = ${sessionInput};
  if (!_session_${sanitizedId} || !_session_${sanitizedId}.id) {
    throw new Error("[browser_action:${action}] no session connected");
  }
  let _mode_${sanitizedId} = _session_${sanitizedId}.mode || "chromium";
  let ${outputVar}_session = _session_${sanitizedId};
  let ${outputVar}_result = null;
  ${letOrAssign}${outputVar} = ${outputVar}_session;`;

        if (action === 'click') {
          code += `
  // click: prefer evaluateJson on chromium, control() on webview legacy
  let _sel_${sanitizedId} = "${selector}";
  if (!_sel_${sanitizedId}) {
    throw new Error("[browser_action:click] Selector is required. Set the Selector property on this node.");
  }
  if (_mode_${sanitizedId} === "webview" && Browser.control) {
    await Browser.control(_session_${sanitizedId}, "click", _sel_${sanitizedId}, "", ${timeoutMs}, "${node.id}");
  } else if (_mode_${sanitizedId} === "http") {
    throw new Error("[browser_action:click] not supported in HTTP mode");
  } else {
    // If the click triggers navigation, the JS execution context is torn
    // down before evaluateJson's response can return — CDP responds with
    // -32000 "Cannot find context with specified id". That isn't a click
    // failure: the click DID happen and navigation is in flight. Swallow
    // the context-destroyed family of errors so a subsequent wait/page
    // node can pick up the new context. Genuine selector-not-found and
    // protocol errors still propagate.
    let _click_script_${sanitizedId} = "(() => { const el = document.querySelector(" + JSON.stringify(_sel_${sanitizedId}) + "); if (!el) throw new Error('selector not found: ' + " + JSON.stringify(_sel_${sanitizedId}) + "); el.click(); return { ok: true }; })()";
    try {
      ${outputVar}_result = await Browser.evaluateJson(_session_${sanitizedId}.id, _click_script_${sanitizedId});
    } catch (_err) {
      const _msg = String(_err && _err.message ? _err.message : _err);
      if (/Cannot find context|Execution context (was destroyed|destroyed)|Inspected target navigated|Target closed/i.test(_msg)) {
        ${outputVar}_result = { ok: true, navigated: true };
      } else {
        throw _err;
      }
    }
  }`;
        } else if (action === 'type') {
          code += `
  // type: prefer evaluateJson on chromium, control() on webview legacy
  let _sel_${sanitizedId} = "${selector}";
  if (!_sel_${sanitizedId}) {
    throw new Error("[browser_action:type] Selector is required. Set the Selector property on this node.");
  }
  let _txt_${sanitizedId} = ${valueInput ? `String(${valueInput} == null ? "" : ${valueInput})` : `"${text}"`};
  if (_mode_${sanitizedId} === "webview" && Browser.control) {
    await Browser.control(_session_${sanitizedId}, "type", _sel_${sanitizedId}, _txt_${sanitizedId}, ${timeoutMs}, "${node.id}");
  } else if (_mode_${sanitizedId} === "http") {
    throw new Error("[browser_action:type] not supported in HTTP mode");
  } else {
    let _type_script_${sanitizedId} = "(() => { const el = document.querySelector(" + JSON.stringify(_sel_${sanitizedId}) + "); if (!el) throw new Error('selector not found: ' + " + JSON.stringify(_sel_${sanitizedId}) + "); el.focus(); el.value = " + JSON.stringify(_txt_${sanitizedId}) + "; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); return { ok: true }; })()";
    ${outputVar}_result = await Browser.evaluateJson(_session_${sanitizedId}.id, _type_script_${sanitizedId});
  }`;
        } else if (action === 'scroll') {
          code += `
  // scroll: bottom / top / selector
  let _scroll_to_${sanitizedId} = "${scrollTo}";
  let _scroll_sel_${sanitizedId} = "${selector}";
  if (_scroll_to_${sanitizedId} === "selector" && !_scroll_sel_${sanitizedId}) {
    throw new Error("[browser_action:scroll] Selector is required when Scroll To is 'Selector'.");
  }
  if (_mode_${sanitizedId} === "http") {
    throw new Error("[browser_action:scroll] not supported in HTTP mode");
  }
  let _scroll_script_${sanitizedId} = "";
  if (_scroll_to_${sanitizedId} === "bottom") {
    _scroll_script_${sanitizedId} = "(() => { window.scrollTo(0, document.body.scrollHeight); return { ok: true }; })()";
  } else if (_scroll_to_${sanitizedId} === "top") {
    _scroll_script_${sanitizedId} = "(() => { window.scrollTo(0, 0); return { ok: true }; })()";
  } else {
    _scroll_script_${sanitizedId} = "(() => { const el = document.querySelector(" + JSON.stringify(_scroll_sel_${sanitizedId}) + "); if (!el) throw new Error('selector not found: ' + " + JSON.stringify(_scroll_sel_${sanitizedId}) + "); el.scrollIntoView({ behavior: 'smooth', block: 'center' }); return { ok: true }; })()";
  }
  if (_mode_${sanitizedId} === "webview" && Browser.control) {
    await Browser.control(_session_${sanitizedId}, "scroll", _scroll_sel_${sanitizedId} || _scroll_to_${sanitizedId}, "", ${timeoutMs}, "${node.id}");
  } else {
    ${outputVar}_result = await Browser.evaluateJson(_session_${sanitizedId}.id, _scroll_script_${sanitizedId});
  }`;
        } else if (action === 'wait') {
          code += `
  let _wait_sel_${sanitizedId} = "${selector}";
  if (!_wait_sel_${sanitizedId}) {
    throw new Error("[browser_action:wait] Selector is required. Set the Selector property on this node.");
  }
  await Browser.waitForSelector(_session_${sanitizedId}.id, _wait_sel_${sanitizedId}, ${timeoutMs});`;
        } else if (action === 'screenshot') {
          code += `
  let _shot_${sanitizedId} = await Browser.screenshot(_session_${sanitizedId}.id, {
    fullPage: ${fullPage},
    format: "${format}",
    quality: ${quality},
    selector: "${selector}" || undefined,
    savePath: "${savePath}" || undefined,
  });
  ${outputVar}_result = _shot_${sanitizedId}.savedPath || _shot_${sanitizedId}.dataUrl || "";`;
        } else if (action === 'get_cookies') {
          code += `
  ${outputVar}_result = await Browser.getCookies(_session_${sanitizedId}.id, "${domain}" || undefined);`;
        } else if (action === 'set_cookies') {
          const cookiesExpr = valueInput ? valueInput : `JSON.parse("${cookiesJson}")`;
          code += `
  let _cookies_${sanitizedId};
  try {
    _cookies_${sanitizedId} = ${cookiesExpr};
    if (!Array.isArray(_cookies_${sanitizedId})) _cookies_${sanitizedId} = [];
  } catch (_) { _cookies_${sanitizedId} = []; }
  await Browser.setCookies(_session_${sanitizedId}.id, _cookies_${sanitizedId}, ${merge});`;
        } else if (action === 'close') {
          code += `
  ${outputVar}_result = await Browser.close(_session_${sanitizedId}.id);
  // Session is closed: invalidate the session passthrough so downstream
  // nodes see falsy and can branch on "session was closed".
  ${outputVar}_session = null;
  ${outputVar} = ${outputVar}_session;`;
        }

        code += `
  workflow_context["${node.id}"] = ${outputVar}_result;`;
        break;
      }

      // ============================================================
      // browser_request — kept as-is from legacy, plain HTTP fetch
      // ============================================================
      case 'browser_request': {
        const urlInput = inputs.get('url');
        const staticUrl = escapeString(String(data.url || ''));
        const method = escapeString(String(data.method || 'GET'));
        const headers = escapeString(String(data.headers || '{}'));
        const staticBody = escapeString(String(data.body || ''));
        const allowLocalNetwork = data.allowLocalNetwork === true;
        const bodyInput = inputs.get('body');
        const sessionInput = inputs.get('session');
        const sessionVar = sessionInput || 'null';
        const urlExpr = urlInput ? urlInput : `"${staticUrl}"`;
        const bodyExpr = bodyInput ? bodyInput : `"${staticBody}"`;
        const bodyType = escapeString(String(data.bodyType || 'none'));
        const responseFormat = escapeString(String(data.responseFormat || 'auto'));

        code += `
  let _session_id_${sanitizedId} = ${sessionVar} ? ${sessionVar}.id : null;
  let _url_${sanitizedId} = ${urlExpr};
  let _body_raw_${sanitizedId} = ${bodyExpr};
  let _headers_raw_${sanitizedId} = "${headers}";
  let _headers_${sanitizedId} = _headers_raw_${sanitizedId};
  if (_headers_raw_${sanitizedId}.indexOf("{{") >= 0) {
    _headers_${sanitizedId} = Utility.template(_headers_raw_${sanitizedId}, workflow_context);
  }
  // Encode body + auto-set Content-Type based on bodyType.
  let _body_${sanitizedId} = _body_raw_${sanitizedId};
  let _hdrs_obj_${sanitizedId} = {};
  try { _hdrs_obj_${sanitizedId} = _headers_${sanitizedId} ? JSON.parse(_headers_${sanitizedId}) : {}; } catch (_) { _hdrs_obj_${sanitizedId} = {}; }
  if ("${bodyType}" === "json") {
    if (_body_raw_${sanitizedId} && typeof _body_raw_${sanitizedId} !== "string") {
      _body_${sanitizedId} = JSON.stringify(_body_raw_${sanitizedId});
    }
    if (!_hdrs_obj_${sanitizedId}["Content-Type"] && !_hdrs_obj_${sanitizedId}["content-type"]) {
      _hdrs_obj_${sanitizedId}["Content-Type"] = "application/json";
    }
  } else if ("${bodyType}" === "form_urlencoded") {
    if (_body_raw_${sanitizedId} && typeof _body_raw_${sanitizedId} === "object") {
      _body_${sanitizedId} = Object.entries(_body_raw_${sanitizedId})
        .map(([k, v]) => encodeURIComponent(k) + "=" + encodeURIComponent(String(v == null ? "" : v)))
        .join("&");
    }
    if (!_hdrs_obj_${sanitizedId}["Content-Type"] && !_hdrs_obj_${sanitizedId}["content-type"]) {
      _hdrs_obj_${sanitizedId}["Content-Type"] = "application/x-www-form-urlencoded";
    }
  } else if ("${bodyType}" === "none") {
    _body_${sanitizedId} = "";
  }
  // raw: pass through unchanged
  _headers_${sanitizedId} = JSON.stringify(_hdrs_obj_${sanitizedId});

  ${letOrAssign}${outputVar} = await Browser.request(
    _url_${sanitizedId},
    "${method}",
    _headers_${sanitizedId},
    _body_${sanitizedId},
    _session_id_${sanitizedId},
    "${node.id}",
    ${allowLocalNetwork}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] Aborted");
    return workflow_context;
  }
  // Shape the response per responseFormat.
  if (${outputVar} && typeof ${outputVar} === "object") {
    if ("${responseFormat}" === "full") {
      // already in {status, headers, body} shape
    } else {
      const _ct_${sanitizedId} = (${outputVar}.headers && (${outputVar}.headers["content-type"] || ${outputVar}.headers["Content-Type"])) || "";
      const _wantJson_${sanitizedId} = "${responseFormat}" === "json"
        || ("${responseFormat}" === "auto" && /json/i.test(_ct_${sanitizedId}));
      if (_wantJson_${sanitizedId}) {
        try { ${outputVar} = JSON.parse(${outputVar}.body); }
        catch (_) { ${outputVar} = ${outputVar}.body; }
      } else {
        ${outputVar} = ${outputVar}.body;
      }
    }
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      default:
        return null;
    }

    return code;
  },
};

export default CoreBrowserCompiler;
