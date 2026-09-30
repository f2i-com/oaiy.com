#!/usr/bin/env python3
"""OAIY Companion — Playwright browser server.

A tiny single-threaded HTTP server exposing Chromium automation backed by
Playwright. The companion runs this as a managed service; oaiy-web's
browser_* nodes talk to it over HTTP when running in a plain browser (no
Tauri native browser plugin available).

Single-threaded by design: the Playwright *sync* API is thread-affine,
and local browser automation is inherently serial (one flow step at a
time). Every request is handled on the main thread, where Playwright was
started — so we never touch a Playwright object from a foreign thread.

Config (env):
  OAIY_BROWSER_PORT       TCP port to bind on 127.0.0.1 (default 17880)
  OAIY_BROWSER_HEADLESS   "0" to run headed (debugging); default headless
  OAIY_ALLOWED_ORIGINS    comma list of the exact web origins that may call this
                          server from a browser (OAIY passes the origins of its
                          Agent and Flows windows). Unset or empty: no web page
                          may. Programs that send no Origin header (the desktop,
                          curl, scripts) are not affected.

Who may call it: it drives a real browser (goto, evaluate, cookies), and loopback
is no boundary against the user's own browser, where any open web page can send
requests to 127.0.0.1. Every request, of any method, is checked before it is read:
  - `Host` must be 127.0.0.1:<port>, localhost:<port> or [::1]:<port>, this
    server's own port; anything else (a DNS name rebound to 127.0.0.1) is 421;
  - a request that carries `Origin`, or `Sec-Fetch-Site` other than same-origin
    or none (cross-site, same-site), is 403 unless its Origin is in
    OAIY_ALLOWED_ORIGINS;
  - `Access-Control-Allow-Origin` echoes such an allowed origin and is never
    `*`; nothing answers a Private Network Access preflight.

API (all JSON; the checks above come first):
  GET    /health                       -> { ok, browser, headless }
  POST   /session            {config}  -> { sessionId }
  DELETE /session/<id>                 -> { ok }
  POST   /session/<id>/goto  {url,waitUntil?} -> { url, title, status }
  GET    /session/<id>/html            -> { html }
  GET    /session/<id>/title           -> { title }
  GET    /session/<id>/url             -> { url }
  GET    /session/<id>/cookies         -> { cookies: [...] }
  POST   /session/<id>/evaluate {script}      -> { result }
  POST   /session/<id>/wait  {selector,timeoutMs?} -> { ok }
  POST   /session/<id>/action {action,selector?,text?,key?} -> { ok }
  POST   /session/<id>/screenshot {fullPage?} -> { dataUrl }
  POST   /session/<id>/cookies {cookies,merge?} -> { ok, count }
"""
import base64
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("OAIY_BROWSER_PORT", "17880"))
HEADLESS = os.environ.get("OAIY_BROWSER_HEADLESS", "1") != "0"

# Lazy globals — the browser launches on first /health or /session so an
# import/launch failure is reported via HTTP rather than crashing at boot.
_pw = None
_browser = None
_sessions = {}  # sessionId -> { "context": ctx, "page": page }
_seq = 0


def log(msg):
    print(f"[playwright-server] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Who may call this server (see the module docstring for the rules)
# ---------------------------------------------------------------------------

# scheme://host[:port] and nothing else: no path, user info or wildcard.
_ORIGIN_SHAPE = re.compile(r"^[a-z][a-z0-9+.\-]*://[a-z0-9.\-_\[\]:]+$")


def parse_allowed_origins(raw):
    """The exact origins in OAIY_ALLOWED_ORIGINS, lowercased, as a frozenset.

    An entry that is not an exact origin (`*`, `null`, a bare host, one with a
    path) is dropped and reported: a typo narrows who may call the server, it
    never widens it."""
    allowed = set()
    for entry in (raw or "").split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        if entry.endswith("/"):
            entry = entry[:-1]
        if _ORIGIN_SHAPE.match(entry):
            allowed.add(entry)
        else:
            log(f"ignoring OAIY_ALLOWED_ORIGINS entry {ascii(entry[:80])}: not an exact origin (scheme://host[:port])")
    return frozenset(allowed)


ALLOWED_ORIGINS = parse_allowed_origins(os.environ.get("OAIY_ALLOWED_ORIGINS", ""))


def host_is_own(host_header, port):
    """Is `Host` one of 127.0.0.1:<port>, localhost:<port>, [::1]:<port>?

    The port is this server's own, exactly. A page on a name rebound to
    127.0.0.1 sends its own name in Host, whatever the port."""
    if not isinstance(host_header, str):
        return False
    return host_header.lower() in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")


# What Sec-Fetch-Site says of a request the user (or the page's own origin)
# made; anything else (cross-site, same-site, a value no browser sends) came
# from a page somewhere else.
_OWN_FETCH_SITES = ("same-origin", "none")


def judge_request(hosts, origins, fetch_sites, port, allowed):
    """Decide one request from its Host, Origin and Sec-Fetch-Site header values
    (each a list of the values received, empty when the header is absent).

    Returns (status, reason, cors_origin): 200 to serve it, else 421 or 403 and
    why; cors_origin is the origin to echo in Access-Control-Allow-Origin, or
    None (the caller sends no CORS headers)."""
    if len(hosts) != 1 or not host_is_own(hosts[0].strip(), port):
        return 421, f"Host must be 127.0.0.1:{port}, localhost:{port} or [::1]:{port}", None
    if len(origins) > 1 or len(fetch_sites) > 1:
        return 403, "more than one Origin or Sec-Fetch-Site header", None
    if origins:
        origin = origins[0].strip().lower()
        if origin in allowed:
            return 200, None, origin
        return 403, f"the origin {ascii(origin[:80])} may not call this server (OAIY_ALLOWED_ORIGINS)", None
    if fetch_sites and fetch_sites[0].strip().lower() not in _OWN_FETCH_SITES:
        return 403, "a web page on another site may not call this server (OAIY_ALLOWED_ORIGINS)", None
    return 200, None, None


# A page can ask in a loop; the log keeps one line about refusals per interval.
_REFUSAL_LOG_EVERY_SECS = 30.0
_refusal_log = {"at": None, "held": 0}


def note_refusal(status, reason, method, now=None):
    now = time.monotonic() if now is None else now
    last = _refusal_log["at"]
    if last is not None and now - last < _REFUSAL_LOG_EVERY_SECS:
        _refusal_log["held"] += 1
        return
    held, _refusal_log["at"], _refusal_log["held"] = _refusal_log["held"], now, 0
    more = f" (+{held} more since the last line)" if held else ""
    verb = re.sub(r"[^A-Za-z]", "?", method[:16])  # what a page sent is never copied into the log as it came
    log(f"refused a {verb} request: {status} {reason}{more}")


def ensure_browser():
    global _pw, _browser
    if _browser is not None:
        return _browser
    from playwright.sync_api import sync_playwright

    log("launching chromium...")
    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(headless=HEADLESS)
    log("chromium ready")
    return _browser


def new_session(config):
    global _seq
    browser = ensure_browser()
    opts = {}
    cfg = config or {}
    if cfg.get("userAgent"):
        opts["user_agent"] = cfg["userAgent"]
    vp = cfg.get("viewport")
    if isinstance(vp, dict) and "width" in vp and "height" in vp:
        opts["viewport"] = {"width": int(vp["width"]), "height": int(vp["height"])}
    context = browser.new_context(**opts)
    page = context.new_page()
    _seq += 1
    sid = f"s{_seq}"
    _sessions[sid] = {"context": context, "page": page}
    return sid


def get_session(sid):
    s = _sessions.get(sid)
    if not s:
        raise KeyError(f"unknown session: {sid}")
    return s


def get_page(sid):
    return get_session(sid)["page"]


# Playwright's add_cookies() wants each cookie keyed by name/value plus
# EITHER url OR (domain AND path). oaiy cookies may carry only a domain, so
# we fall back to the current page URL. sameSite must be one of
# Strict/Lax/None — normalise or drop anything else so one bad value can't
# fail the whole batch.
def _prepare_cookies(raw, page):
    _SS = {"strict": "Strict", "lax": "Lax", "none": "None"}
    prepared = []
    for c in raw or []:
        if not isinstance(c, dict) or "name" not in c or "value" not in c:
            continue
        cc = {}
        for k in ("name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite"):
            if c.get(k) is not None:
                cc[k] = c[k]
        ss = cc.get("sameSite")
        if ss is not None:
            norm = _SS.get(str(ss).lower())
            if norm:
                cc["sameSite"] = norm
            else:
                cc.pop("sameSite", None)
        if not (cc.get("domain") and cc.get("path")):
            cc.pop("domain", None)
            cc.pop("path", None)
            cc["url"] = page.url
        prepared.append(cc)
    return prepared


def close_session(sid):
    s = _sessions.pop(sid, None)
    if s:
        try:
            s["context"].close()
        except Exception:
            pass


class Handler(BaseHTTPRequestHandler):
    # The origin this request may be answered to across origins (the request's
    # own Origin, when it is in ALLOWED_ORIGINS); None sends no CORS headers.
    _cors_origin = None

    # Quieter logging — the companion captures stdout already.
    def log_message(self, *args):
        pass

    def parse_request(self):
        # The one place every request passes, whatever its method: judged
        # before any handler reads its body or touches the browser.
        if not BaseHTTPRequestHandler.parse_request(self):
            return False
        return self._admit()

    def _admit(self):
        self._cors_origin = None
        status, reason, cors_origin = judge_request(
            self.headers.get_all("Host") or [],
            self.headers.get_all("Origin") or [],
            self.headers.get_all("Sec-Fetch-Site") or [],
            self.server.server_address[1],
            ALLOWED_ORIGINS,
        )
        if status != 200:
            self._refuse(status, reason)
            return False
        self._cors_origin = cors_origin
        return True

    def _refuse(self, status, reason):
        note_refusal(status, reason, self.command)
        self._drain_body()
        body = json.dumps({"error": reason}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drain_body(self):
        # A body left unread when the connection closes can make the peer see a
        # reset instead of the answer (Windows). Bounded in size and in time: a
        # refused caller does not get to hold this one-at-a-time server.
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            return
        if length <= 0:
            return
        try:
            self.connection.settimeout(2.0)
            self.rfile.read(min(length, 1 << 20))
        except OSError:
            pass

    def _send(self, code, payload):
        body = b"" if code == 204 else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Vary", "Origin")
        if self._cors_origin:
            # Only the caller's own, allowed origin; never `*`, never
            # Access-Control-Allow-Private-Network (no page outside OAIY's
            # windows is to be let in).
            self.send_header("Access-Control-Allow-Origin", self._cors_origin)
            if self.command == "OPTIONS":
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return {}

    def do_OPTIONS(self):
        self._send(204, {})

    def _parts(self):
        # /session/<id>/<verb> -> ["session", id, verb]
        return [p for p in self.path.split("?")[0].strip("/").split("/") if p]

    def do_GET(self):
        try:
            p = self._parts()
            if p == ["health"]:
                ok = True
                err = None
                try:
                    ensure_browser()
                except Exception as e:  # report launch failure as unhealthy
                    ok = False
                    err = str(e)
                self._send(200 if ok else 503,
                           {"ok": ok, "browser": "chromium", "headless": HEADLESS, "error": err})
                return
            if len(p) == 3 and p[0] == "session":
                sess = get_session(p[1])
                page = sess["page"]
                if p[2] == "html":
                    self._send(200, {"html": page.content()})
                    return
                if p[2] == "title":
                    self._send(200, {"title": page.title()})
                    return
                if p[2] == "url":
                    self._send(200, {"url": page.url})
                    return
                if p[2] == "cookies":
                    self._send(200, {"cookies": sess["context"].cookies()})
                    return
            self._send(404, {"error": f"no route for GET {self.path}"})
        except KeyError as e:
            self._send(404, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            p = self._parts()
            body = self._body()
            if p == ["session"]:
                self._send(200, {"sessionId": new_session(body.get("config") or body)})
                return
            if len(p) == 3 and p[0] == "session":
                sess = get_session(p[1])
                page = sess["page"]
                verb = p[2]
                if verb == "goto":
                    url = body.get("url")
                    if not url:
                        self._send(400, {"error": "goto requires 'url'"})
                        return
                    resp = page.goto(url, wait_until=body.get("waitUntil") or "load")
                    self._send(200, {
                        "url": page.url,
                        "title": page.title(),
                        "status": resp.status if resp else None,
                    })
                    return
                if verb == "evaluate":
                    result = page.evaluate(body.get("script") or "null")
                    try:
                        json.dumps(result)
                    except Exception:
                        result = str(result)
                    self._send(200, {"result": result})
                    return
                if verb == "wait":
                    sel = body.get("selector")
                    page.wait_for_selector(sel, timeout=int(body.get("timeoutMs") or 30000))
                    self._send(200, {"ok": True})
                    return
                if verb == "action":
                    act = body.get("action")
                    sel = body.get("selector")
                    if act == "click":
                        page.click(sel)
                    elif act == "fill":
                        page.fill(sel, body.get("text") or "")
                    elif act == "type":
                        page.type(sel, body.get("text") or "")
                    elif act == "press":
                        page.press(sel, body.get("key") or "Enter")
                    else:
                        self._send(400, {"error": f"unknown action: {act}"})
                        return
                    self._send(200, {"ok": True})
                    return
                if verb == "screenshot":
                    png = page.screenshot(full_page=bool(body.get("fullPage")))
                    b64 = base64.b64encode(png).decode("ascii")
                    self._send(200, {"dataUrl": f"data:image/png;base64,{b64}"})
                    return
                if verb == "cookies":
                    ctxobj = sess["context"]
                    if not body.get("merge", True):
                        ctxobj.clear_cookies()
                    prepared = _prepare_cookies(body.get("cookies"), page)
                    if prepared:
                        ctxobj.add_cookies(prepared)
                    self._send(200, {"ok": True, "count": len(prepared)})
                    return
            self._send(404, {"error": f"no route for POST {self.path}"})
        except KeyError as e:
            self._send(404, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def do_DELETE(self):
        try:
            p = self._parts()
            if len(p) == 2 and p[0] == "session":
                close_session(p[1])
                self._send(200, {"ok": True})
                return
            self._send(404, {"error": f"no route for DELETE {self.path}"})
        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    log(f"starting on 127.0.0.1:{PORT} (headless={HEADLESS})")
    log("web pages allowed to call it: " + (", ".join(sorted(ALLOWED_ORIGINS)) or "none (OAIY_ALLOWED_ORIGINS is empty)"))
    try:
        # Launch eagerly so a failure surfaces immediately in the logs +
        # the companion's health check, rather than on first use.
        ensure_browser()
    except Exception as e:
        log(f"WARNING: chromium launch failed at boot: {e}")
        log("Run `python -m playwright install chromium` in the venv.")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for sid in list(_sessions):
            close_session(sid)
        if _browser is not None:
            try:
                _browser.close()
            except Exception:
                pass
        if _pw is not None:
            try:
                _pw.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
