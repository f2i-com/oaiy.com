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

API (all JSON, CORS open — bind is loopback-only):
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
import sys
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
    # Quieter logging — the companion captures stdout already.
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
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
