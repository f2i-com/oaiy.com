"""Tests for who may call OAIY's Playwright HTTP server (access model ACC-12).

    python -m unittest discover -s platform/desktop/scripts -p "test_*.py"

The server drives a real browser (goto a file:// URL, evaluate script, read
cookies) on a loopback port. Any web page open in the user's browser can send
requests to a loopback port, and a page on a DNS name rebound to 127.0.0.1
sends its own name as Host. So: Host must be the server's own; a request that
carries an Origin, or a Sec-Fetch-Site of another site, is refused unless the
Origin is one of OAIY's windows (OAIY_ALLOWED_ORIGINS); nothing but such an
origin is ever echoed in Access-Control-Allow-Origin.

Two layers:
  - the rules, as pure functions of header values;
  - the real script as a child process on a free loopback port, with a stub
    `playwright` package that records what it was asked to do, driven with raw
    sockets by the personas below: a page on another loopback port, a dev server
    on localhost, a rebound name, a sandboxed frame, OAIY's own windows, and
    programs that send no Origin at all (the desktop, curl, scripts).

No browser is started, and no port but a free one the operating system picks
is ever bound (never 17880, the service's own).

OAIY_PLAYWRIGHT_SERVER_PY names another copy of the script to test (the
mutation checks in the change that added these tests used it).
"""
import importlib.util
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = pathlib.Path(
    os.environ.get("OAIY_PLAYWRIGHT_SERVER_PY")
    or HERE.parent / "src-tauri" / "resources" / "scripts" / "playwright_server.py"
)

# What the desktop passes: the exact origins of the Agent and the flow editor
# windows (Windows serves them from http://<scheme>.localhost, the others from
# <scheme>://localhost); the tests allow all four so that one run covers both.
OAIY_WINDOW_ORIGINS = [
    "http://oaiy.localhost",
    "http://oaiyflows.localhost",
    "oaiy://localhost",
    "oaiyflows://localhost",
]

# The live machine's ports, which no test may bind or contact.
LIVE_PORTS = {17972, 17872, 17973, 7860, 8080, 9333, 17880}


def load_module():
    spec = importlib.util.spec_from_file_location("oaiy_playwright_server_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server_module = load_module()


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
class ParseAllowedOrigins(unittest.TestCase):
    def setUp(self):
        self.logged = []
        original_log = server_module.log
        server_module.log = self.logged.append
        self.addCleanup(setattr, server_module, "log", original_log)

    def test_exact_origins_are_kept_lowercased_and_trimmed(self):
        got = server_module.parse_allowed_origins(" HTTP://oaiy.localhost , oaiyflows://localhost,,http://127.0.0.1:5173/ ")
        self.assertEqual(got, frozenset({"http://oaiy.localhost", "oaiyflows://localhost", "http://127.0.0.1:5173"}))

    def test_nothing_set_allows_no_page(self):
        self.assertEqual(server_module.parse_allowed_origins(None), frozenset())
        self.assertEqual(server_module.parse_allowed_origins(""), frozenset())
        self.assertEqual(server_module.parse_allowed_origins(" , ,"), frozenset())

    def test_an_entry_that_is_not_an_exact_origin_is_dropped_never_widened(self):
        for junk in [
            "*",
            "null",
            "http://*.localhost",
            "http://*",
            "oaiy.localhost",  # no scheme
            "http://oaiy.localhost/agent",  # a path
            "http://oaiy.localhost?x=1",
            "http://user@oaiy.localhost",
            "http://",
            "https:// oaiy.com",
        ]:
            with self.subTest(junk=junk):
                self.logged.clear()
                self.assertEqual(server_module.parse_allowed_origins(junk), frozenset())
                self.assertEqual(len(self.logged), 1, "and the entry is reported")

    def test_a_bad_entry_does_not_drop_the_good_ones_beside_it(self):
        got = server_module.parse_allowed_origins("*, http://oaiy.localhost, null")
        self.assertEqual(got, frozenset({"http://oaiy.localhost"}))
        self.assertEqual(len(self.logged), 2)


class HostRule(unittest.TestCase):
    PORT = 41234

    def own(self, host):
        return server_module.host_is_own(host, self.PORT)

    def test_the_three_loopback_names_with_the_servers_port(self):
        for host in ["127.0.0.1:41234", "localhost:41234", "[::1]:41234", "LocalHost:41234"]:
            with self.subTest(host=host):
                self.assertTrue(self.own(host))

    def test_any_other_port_or_no_port_is_not_its_own(self):
        for host in ["127.0.0.1", "localhost", "[::1]", "127.0.0.1:41235", "localhost:17880", "127.0.0.1:041234", "127.0.0.1:41234 ", "127.0.0.1:"]:
            with self.subTest(host=host):
                self.assertFalse(self.own(host))

    def test_a_rebound_or_lookalike_name_is_not_its_own(self):
        for host in [
            "attacker.example:41234",
            "127.0.0.1.attacker.example:41234",
            "localhost.attacker.example:41234",
            "localhost.:41234",
            "0.0.0.0:41234",
            "127.0.0.2:41234",
            "[::2]:41234",
            "[::ffff:127.0.0.1]:41234",
            "user@127.0.0.1:41234",
            "",
        ]:
            with self.subTest(host=host):
                self.assertFalse(self.own(host))

    def test_a_missing_host_is_not_its_own(self):
        self.assertFalse(self.own(None))


class JudgeRequest(unittest.TestCase):
    PORT = 41234
    ALLOWED = frozenset({"http://oaiy.localhost", "oaiyflows://localhost"})

    def judge(self, hosts=("127.0.0.1:41234",), origins=(), sites=(), allowed=None):
        return server_module.judge_request(
            list(hosts), list(origins), list(sites), self.PORT, self.ALLOWED if allowed is None else allowed
        )

    def status(self, **kw):
        return self.judge(**kw)[0]

    def test_a_program_with_no_origin_and_the_right_host_is_served_without_cors(self):
        for host in ["127.0.0.1:41234", "localhost:41234", "[::1]:41234"]:
            self.assertEqual(self.judge(hosts=[host]), (200, None, None))

    def test_a_wrong_or_missing_host_is_421_before_anything_else(self):
        self.assertEqual(self.status(hosts=["attacker.example:41234"]), 421)
        self.assertEqual(self.status(hosts=[]), 421)
        self.assertEqual(self.status(hosts=["127.0.0.1:41234", "127.0.0.1:41234"]), 421, "two Host headers")
        # Even an allowed origin cannot make a wrong Host right, and a wrong Host is judged first.
        self.assertEqual(self.status(hosts=["attacker.example:41234"], origins=["http://oaiy.localhost"]), 421)
        self.assertEqual(self.status(hosts=["attacker.example:41234"], origins=["http://evil.example"], sites=["cross-site"]), 421)
        # A rebound page is same-origin with its own name: Fetch Metadata says nothing against it.
        self.assertEqual(self.status(hosts=["attacker.example:41234"], origins=["http://attacker.example:41234"], sites=["same-origin"]), 421)

    def test_an_origin_outside_the_list_is_403_whatever_else_the_request_says(self):
        for origin in [
            "http://127.0.0.1:5173",  # a page on another loopback port
            "http://localhost:3000",
            "https://oaiy.com",  # the hosted editor
            "null",  # a sandboxed frame
            "http://oaiy.localhost:8080",  # the window's name, another port
            "https://oaiy.localhost",  # the window's name, another scheme
            "http://oaiy.localhost.attacker.example",
            "http://127.0.0.1:41234",  # this server's own address serves no page
            "*",
            "",
        ]:
            with self.subTest(origin=origin):
                self.assertEqual(self.status(origins=[origin]), 403)
                self.assertEqual(self.status(origins=[origin], sites=["same-origin"]), 403)
                self.assertEqual(self.status(origins=[origin], sites=["cross-site"]), 403)

    def test_an_allowed_origin_is_served_and_echoed_whatever_its_fetch_site(self):
        for site in [(), ("cross-site",), ("same-site",), ("same-origin",), ("none",)]:
            with self.subTest(site=site):
                self.assertEqual(self.judge(origins=["http://oaiy.localhost"], sites=site), (200, None, "http://oaiy.localhost"))
        self.assertEqual(self.judge(origins=["oaiyflows://localhost"]), (200, None, "oaiyflows://localhost"))
        # The echo is the caller's own origin, normalised, never a wildcard.
        self.assertEqual(self.judge(origins=["HTTP://OAIY.LOCALHOST"])[2], "http://oaiy.localhost")

    def test_an_empty_list_allows_no_origin_at_all(self):
        self.assertEqual(self.status(origins=["http://oaiy.localhost"], allowed=frozenset()), 403)
        self.assertEqual(self.status(allowed=frozenset()), 200, "and a program with no Origin is still served")

    def test_another_site_with_no_origin_is_403(self):
        # A no-cors GET carries no Origin in every browser; Fetch Metadata still says who sent it.
        for site in ["cross-site", "same-site", "CROSS-SITE", " same-site ", "surprise", ""]:
            with self.subTest(site=site):
                self.assertEqual(self.status(sites=[site]), 403)

    def test_a_typed_address_or_the_servers_own_page_is_served(self):
        self.assertEqual(self.status(sites=["none"]), 200)
        self.assertEqual(self.status(sites=["same-origin"]), 200)

    def test_two_origin_or_fetch_site_headers_are_403(self):
        self.assertEqual(self.status(origins=["http://oaiy.localhost", "http://evil.example"]), 403)
        self.assertEqual(self.status(origins=["http://evil.example", "http://oaiy.localhost"]), 403)
        self.assertEqual(self.status(origins=["http://oaiy.localhost"], sites=["cross-site", "same-origin"]), 403)


class RefusalLog(unittest.TestCase):
    def test_a_flood_of_refusals_is_one_line_per_interval_with_the_count_held(self):
        lines = []
        original_log = server_module.log
        server_module.log = lines.append
        server_module._refusal_log.update({"at": None, "held": 0})
        try:
            server_module.note_refusal(403, "no", "POST", now=100.0)
            for i in range(50):
                server_module.note_refusal(403, "no", "POST", now=101.0 + i * 0.1)
            server_module.note_refusal(421, "host", "GET", now=100.0 + server_module._REFUSAL_LOG_EVERY_SECS + 1)
        finally:
            server_module.log = original_log
            server_module._refusal_log.update({"at": None, "held": 0})
        self.assertEqual(len(lines), 2)
        self.assertIn("(+50 more", lines[1])

    def test_what_a_page_sent_cannot_forge_a_log_line(self):
        lines = []
        original_log = server_module.log
        server_module.log = lines.append
        server_module._refusal_log.update({"at": None, "held": 0})
        try:
            server_module.note_refusal(403, "the origin 'x'", "PO\nST\r\n[playwright-server] fake", now=1.0)
        finally:
            server_module.log = original_log
            server_module._refusal_log.update({"at": None, "held": 0})
        self.assertEqual(len(lines), 1)
        self.assertNotIn("\n", lines[0])
        self.assertNotIn("\r", lines[0])
        self.assertTrue(lines[0].startswith("refused a PO?ST"), lines[0])


# ---------------------------------------------------------------------------
# The real script, on a free port, with a stub browser
# ---------------------------------------------------------------------------
STUB_PLAYWRIGHT = r'''
"""A stand-in for the `playwright` package: no browser, every call recorded."""
import json
import os

_LOG = os.environ["STUB_PLAYWRIGHT_LOG"]


def _note(call, **detail):
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(call=call, **detail)) + "\n")


class _Response:
    status = 200


class _Page:
    url = "about:blank"

    def goto(self, url, wait_until="load"):
        _note("goto", url=url)
        self.url = url
        return _Response()

    def title(self):
        return "stub title"

    def content(self):
        return "<html>stub</html>"

    def evaluate(self, script):
        _note("evaluate", script=script)
        return "evaluated"


class _Context:
    def new_page(self):
        return _Page()

    def cookies(self):
        return []

    def add_cookies(self, cookies):
        _note("add_cookies", count=len(cookies))

    def clear_cookies(self):
        pass

    def close(self):
        _note("close_context")


class _Browser:
    def new_context(self, **opts):
        _note("new_context")
        return _Context()

    def close(self):
        pass


class _Chromium:
    def launch(self, headless=True):
        _note("launch")
        return _Browser()


class _Playwright:
    chromium = _Chromium()

    def stop(self):
        pass


class _Starter:
    def start(self):
        return _Playwright()


def sync_playwright():
    return _Starter()
'''


def free_port():
    """A free loopback port the operating system picks, never one of the live machine's."""
    for _ in range(20):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in LIVE_PORTS:
            return port
    raise RuntimeError("no free port")


class Reply:
    def __init__(self, raw):
        self.raw = raw
        head, _, self.body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        self.status = int(lines[0].split()[1])
        self.headers = [(k.strip().lower(), v.strip()) for k, _, v in (line.partition(":") for line in lines[1:])]

    def all(self, name):
        return [v for k, v in self.headers if k == name.lower()]

    def get(self, name):
        found = self.all(name)
        return found[0] if found else None

    def cors_headers(self):
        return [(k, v) for k, v in self.headers if k.startswith("access-control-")]

    def json(self):
        return json.loads(self.body.decode("utf-8")) if self.body else None


def read_until_closed(sock, timeout):
    """Everything the server sends until it closes the connection; a server that
    keeps the connection open makes this raise (socket.timeout) instead of hang."""
    sock.settimeout(timeout)
    chunks = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


# Once an answer has been received whole, the server closes the connection at
# once (HTTP/1.0: no keep-alive). A server that does not is a failure found here,
# and found once: every request after it fails without waiting, so a server
# switched to keep-alive fails the suite in seconds rather than making it wait
# out a timeout per request.
CLOSE_AFTER_ANSWER_SECS = 2.0
_kept_open = []


def _answer_is_whole(data):
    head, sep, body = data.partition(b"\r\n\r\n")
    length = re.search(rb"(?im)^content-length:\s*(\d+)", head) if sep else None
    return length is not None and len(body) >= int(length.group(1))


def read_answer(sock, timeout):
    """One answer, and then the end of the connection."""
    data = b""
    while True:
        sock.settimeout(CLOSE_AFTER_ANSWER_SECS if _answer_is_whole(data) else timeout)
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            if _answer_is_whole(data):
                _kept_open.append(data[:40])
                raise AssertionError("the server answered and kept the connection open: every answer must close it (HTTP/1.0)")
            raise
        if not chunk:
            return data
        data += chunk


def raw_request(port, method, path, headers, body=None, version="HTTP/1.1", timeout=10):
    """Send exactly these header lines (a list of (name, value): repeats and a
    missing Host are possible) and read until the server closes. `.elapsed` on
    the reply is how long that took."""
    if _kept_open:
        raise AssertionError(f"an earlier answer left its connection open ({_kept_open[0]!r}); not asking again")
    payload = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
    head = [f"{method} {path} {version}"] + [f"{k}: {v}" for k, v in headers]
    if payload:
        head.append(f"Content-Length: {len(payload)}")
        head.append("Content-Type: application/json")
    request = ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + payload
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(request)
        reply = Reply(read_answer(s, timeout))
    reply.elapsed = time.monotonic() - started
    return reply


class RunningServer:
    """playwright_server.py as a child process on a free loopback port."""

    def __init__(self, allowed_origins):
        self.dir = tempfile.TemporaryDirectory(prefix="oaiy-pw-test-")
        root = pathlib.Path(self.dir.name)
        (root / "stub" / "playwright").mkdir(parents=True)
        (root / "stub" / "playwright" / "__init__.py").write_text("")
        (root / "stub" / "playwright" / "sync_api.py").write_text(STUB_PLAYWRIGHT)
        self.log_path = root / "playwright-calls.jsonl"
        self.log_path.write_text("")
        self.stdout_path = root / "stdout.txt"
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(root / "stub"),
            STUB_PLAYWRIGHT_LOG=str(self.log_path),
            OAIY_BROWSER_HEADLESS="1",
        )
        env.pop("OAIY_ALLOWED_ORIGINS", None)
        if allowed_origins is not None:
            env["OAIY_ALLOWED_ORIGINS"] = allowed_origins
        last_error = None
        for _ in range(3):
            self.port = free_port()
            env["OAIY_BROWSER_PORT"] = str(self.port)
            self.stdout = open(self.stdout_path, "wb")
            self.proc = subprocess.Popen([sys.executable, str(SCRIPT)], env=env, stdout=self.stdout, stderr=subprocess.STDOUT)
            if self._wait_until_listening():
                return
            last_error = self.output()
            self.stop()
        raise RuntimeError(f"the server did not start: {last_error}")

    def _wait_until_listening(self):
        deadline = time.time() + 15
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
                return True
            except OSError:
                time.sleep(0.05)
        return False

    def output(self):
        self.stdout.flush()
        return self.stdout_path.read_text(errors="replace")

    def calls(self):
        text = self.log_path.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.stdout.close()

    def close(self):
        self.stop()
        self.dir.cleanup()


class Personas:
    """What each kind of caller sends, as header lists. `port` is the server's."""

    def __init__(self, port, other_port):
        self.port = port
        self.other_port = other_port

    def host(self, name=None):
        return ("Host", name or f"127.0.0.1:{self.port}")

    def program(self, host=None):
        """The desktop's health check, curl, a script: no Origin, no Fetch Metadata."""
        return [self.host(host)]

    def window(self, origin):
        """One of OAIY's windows: a cross-site page in the browser's eyes, with its Origin."""
        return [self.host(), ("Origin", origin), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")]

    def page_on_another_loopback_port(self):
        """A hostile page at http://127.0.0.1:<other>: same-site as the server (ports do not make a site)."""
        return [self.host(), ("Origin", f"http://127.0.0.1:{self.other_port}"), ("Sec-Fetch-Site", "same-site"), ("Sec-Fetch-Mode", "cors")]

    def dev_server_on_localhost(self):
        return [self.host(), ("Origin", f"http://localhost:{self.other_port}"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")]

    def hosted_editor(self):
        return [self.host(), ("Origin", "https://oaiy.com"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")]

    def sandboxed_frame(self):
        return [self.host(), ("Origin", "null"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")]

    def no_cors_get_from_another_site(self, site):
        """A GET a page sends with mode no-cors carries no Origin; only Fetch Metadata names its site."""
        return [self.host(), ("Sec-Fetch-Site", site), ("Sec-Fetch-Mode", "no-cors")]

    def rebound_name(self):
        """DNS rebinding: attacker.example now resolves to 127.0.0.1; the page is same-origin with itself."""
        name = f"attacker.example:{self.port}"
        return [self.host(name), ("Origin", f"http://{name}"), ("Sec-Fetch-Site", "same-origin"), ("Sec-Fetch-Mode", "cors")]

    def rebound_name_navigating(self):
        name = f"attacker.example:{self.port}"
        return [self.host(name), ("Sec-Fetch-Site", "same-origin"), ("Sec-Fetch-Mode", "navigate")]


class EndToEnd(unittest.TestCase):
    """The script itself, with the OAIY windows' origins allowed."""

    @classmethod
    def setUpClass(cls):
        cls.server = RunningServer(",".join(OAIY_WINDOW_ORIGINS))
        cls.port = cls.server.port
        # A second, held loopback port: the address a hostile page is served from.
        cls.hostile_page = socket.socket()
        cls.hostile_page.bind(("127.0.0.1", 0))
        cls.hostile_page.listen(1)
        cls.other_port = cls.hostile_page.getsockname()[1]
        cls.who = Personas(cls.port, cls.other_port)

    @classmethod
    def tearDownClass(cls):
        cls.hostile_page.close()
        cls.server.close()

    # -- helpers ------------------------------------------------------------
    def call(self, method, path, headers, body=None, version="HTTP/1.1"):
        return raw_request(self.port, method, path, headers, body, version)

    def new_session(self, origin="http://oaiyflows.localhost"):
        reply = self.call("POST", "/session", self.who.window(origin), {"config": {}})
        self.assertEqual(reply.status, 200, reply.body)
        return reply.json()["sessionId"]

    def routes(self, sid):
        """Every route that reads or drives the browser, with a request that would work."""
        return [
            ("POST", "/session", {"config": {}}),
            ("POST", f"/session/{sid}/goto", {"url": "file:///etc/passwd"}),
            ("GET", f"/session/{sid}/html", None),
            ("GET", f"/session/{sid}/title", None),
            ("GET", f"/session/{sid}/url", None),
            ("GET", f"/session/{sid}/cookies", None),
            ("POST", f"/session/{sid}/cookies", {"cookies": [{"name": "a", "value": "b"}]}),
            ("POST", f"/session/{sid}/evaluate", {"script": "document.cookie"}),
            ("DELETE", f"/session/{sid}", None),
            ("GET", "/health", None),
        ]

    def assert_refused(self, reply, status, who):
        self.assertEqual(reply.status, status, f"{who}: {reply.status} {reply.body[:200]!r}")
        self.assertEqual(reply.cors_headers(), [], f"{who}: a refusal carries no CORS headers")
        self.assertIn("error", reply.json(), who)

    def side_effects(self):
        """What the stub browser was asked to do since it launched at boot."""
        return [c for c in self.server.calls() if c["call"] != "launch"]

    # -- refused ------------------------------------------------------------
    def test_a_hostile_page_on_another_loopback_port_is_403_on_every_route(self):
        sid = self.new_session()
        before = self.side_effects()
        for method, path, body in self.routes(sid):
            with self.subTest(route=f"{method} {path}"):
                reply = self.call(method, path, self.who.page_on_another_loopback_port(), body)
                self.assert_refused(reply, 403, "page on another loopback port")
        self.assertEqual(self.side_effects(), before, "nothing reached the browser")

    def test_a_dev_server_the_hosted_editor_and_a_sandboxed_frame_are_403_on_every_route(self):
        sid = self.new_session()
        before = self.side_effects()
        for persona in [self.who.dev_server_on_localhost, self.who.hosted_editor, self.who.sandboxed_frame]:
            for method, path, body in self.routes(sid):
                with self.subTest(persona=persona.__name__, route=f"{method} {path}"):
                    self.assert_refused(self.call(method, path, persona(), body), 403, persona.__name__)
        self.assertEqual(self.side_effects(), before)

    def test_a_rebound_name_is_421_on_every_route(self):
        sid = self.new_session()
        before = self.side_effects()
        for persona in [self.who.rebound_name, self.who.rebound_name_navigating]:
            for method, path, body in self.routes(sid):
                with self.subTest(persona=persona.__name__, route=f"{method} {path}"):
                    self.assert_refused(self.call(method, path, persona(), body), 421, persona.__name__)
        self.assertEqual(self.side_effects(), before)

    def test_a_wrong_host_is_421_for_an_allowed_origin_and_for_a_program_too(self):
        sid = self.new_session()
        bad_hosts = [
            f"attacker.example:{self.port}",
            f"127.0.0.1:{self.other_port}",
            "127.0.0.1",
            f"localhost.:{self.port}",
            f"127.0.0.1.attacker.example:{self.port}",
            f"0.0.0.0:{self.port}",
        ]
        for host in bad_hosts:
            with self.subTest(host=host):
                self.assert_refused(self.call("GET", f"/session/{sid}/html", self.who.program(host)), 421, "program")
                allowed_origin_headers = [self.who.host(host), ("Origin", "http://oaiy.localhost"), ("Sec-Fetch-Site", "cross-site")]
                self.assert_refused(self.call("GET", f"/session/{sid}/html", allowed_origin_headers), 421, "window with a wrong Host")

    def test_no_host_and_two_hosts_are_421(self):
        self.assert_refused(self.call("GET", "/health", [], version="HTTP/1.0"), 421, "HTTP/1.0 without Host")
        self.assert_refused(self.call("GET", "/health", [self.who.host(), self.who.host()]), 421, "two Host headers")

    def test_another_site_with_no_origin_is_403(self):
        for site in ["cross-site", "same-site"]:
            with self.subTest(site=site):
                self.assert_refused(self.call("GET", "/health", self.who.no_cors_get_from_another_site(site)), 403, site)
                self.assert_refused(self.call("GET", "/session/s1/cookies", self.who.no_cors_get_from_another_site(site)), 403, site)

    def test_lookalike_and_near_miss_origins_of_the_windows_are_403(self):
        for origin in [
            "http://oaiy.localhost.attacker.example",
            "http://oaiy.localhost:8080",
            "https://oaiy.localhost",
            "http://oaiyflows.localhost:1",
            "oaiy://localhost:1",
            "oaiy://attacker",
            "http://attacker.example/http://oaiy.localhost",
            "*",
            "null",
        ]:
            with self.subTest(origin=origin):
                headers = [self.who.host(), ("Origin", origin), ("Sec-Fetch-Site", "cross-site")]
                self.assert_refused(self.call("GET", "/health", headers), 403, origin)

    def test_two_origin_headers_are_403(self):
        headers = [self.who.host(), ("Origin", "http://oaiy.localhost"), ("Origin", "http://attacker.example")]
        self.assert_refused(self.call("GET", "/health", headers), 403, "two Origins")

    def test_every_method_is_judged_before_the_server_looks_at_it(self):
        # http.server answers an unknown method with 501 from its own code; the guard runs first.
        for method in ["HEAD", "PUT", "PATCH", "TRACE", "CONNECT"]:
            with self.subTest(method=method):
                hostile = self.call(method, "/session", self.who.page_on_another_loopback_port())
                self.assertEqual(hostile.status, 403)
                rebound = self.call(method, "/session", self.who.rebound_name())
                self.assertEqual(rebound.status, 421)

    def test_a_refused_request_with_a_body_still_gets_its_answer(self):
        # The refused body is read off the socket, or the peer may see a reset instead of the 403.
        sid = self.new_session()
        big = {"url": "file:///etc/passwd", "pad": "x" * 300_000}
        reply = self.call("POST", f"/session/{sid}/goto", self.who.page_on_another_loopback_port(), big)
        self.assertEqual(reply.status, 403)

    def test_a_refused_file_url_goto_never_runs_and_the_same_goto_from_a_window_does(self):
        sid = self.new_session()
        before = self.server.calls()
        for persona in [
            self.who.page_on_another_loopback_port,
            self.who.dev_server_on_localhost,
            self.who.hosted_editor,
            self.who.rebound_name,
            self.who.sandboxed_frame,
        ]:
            for target in ["file:///etc/passwd", "file:///C:/Windows/win.ini", "http://169.254.169.254/latest/meta-data/"]:
                reply = self.call("POST", f"/session/{sid}/goto", persona(), {"url": target, "waitUntil": "load"})
                self.assertIn(reply.status, (403, 421), persona.__name__)
            evaluate = self.call("POST", f"/session/{sid}/evaluate", persona(), {"script": "fetch('/x')"})
            self.assertIn(evaluate.status, (403, 421), persona.__name__)
        self.assertEqual(self.server.calls(), before, "not one call reached the browser")

        # The control: the assertion above can fail. From a window the goto is run, and recorded.
        ok = self.call("POST", f"/session/{sid}/goto", self.who.window("http://oaiyflows.localhost"), {"url": "file:///etc/passwd"})
        self.assertEqual(ok.status, 200)
        self.assertEqual([c for c in self.server.calls() if c["call"] == "goto"][-1]["url"], "file:///etc/passwd")

    # -- preflights -----------------------------------------------------------
    def preflight_headers(self, origin, host=None, private_network=True):
        headers = [
            self.who.host(host),
            ("Origin", origin),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "content-type"),
            ("Sec-Fetch-Site", "cross-site"),
            ("Sec-Fetch-Mode", "cors"),
        ]
        if private_network:
            headers.append(("Access-Control-Request-Private-Network", "true"))
        return headers

    def test_a_preflight_from_a_hostile_origin_is_refused_with_no_cors_and_no_private_network_answer(self):
        for origin in [f"http://127.0.0.1:{self.other_port}", "http://localhost:3000", "https://oaiy.com", "null"]:
            for path in ["/session", "/session/s1/goto", "/health"]:
                with self.subTest(origin=origin, path=path):
                    reply = self.call("OPTIONS", path, self.preflight_headers(origin))
                    self.assert_refused(reply, 403, origin)
                    self.assertIsNone(reply.get("access-control-allow-private-network"))

    def test_a_preflight_on_a_rebound_name_is_421(self):
        name = f"attacker.example:{self.port}"
        reply = self.call("OPTIONS", "/session", self.preflight_headers(f"http://{name}", host=name))
        self.assert_refused(reply, 421, "rebound preflight")
        self.assertIsNone(reply.get("access-control-allow-private-network"))

    def test_a_preflight_from_a_window_is_answered_for_that_origin_only_and_never_for_private_network(self):
        for origin in OAIY_WINDOW_ORIGINS:
            for private_network in [True, False]:
                with self.subTest(origin=origin, private_network=private_network):
                    reply = self.call("OPTIONS", "/session", self.preflight_headers(origin, private_network=private_network))
                    self.assertEqual(reply.status, 204)
                    self.assertEqual(reply.body, b"", "a 204 has no body")
                    self.assertEqual(reply.all("access-control-allow-origin"), [origin])
                    self.assertIn("origin", reply.get("vary").lower())
                    self.assertIn("POST", reply.get("access-control-allow-methods"))
                    self.assertIn("Content-Type", reply.get("access-control-allow-headers"))
                    self.assertIsNone(reply.get("access-control-allow-private-network"))
                    self.assertIsNone(reply.get("access-control-allow-credentials"))
                    self.assertNotIn("*", [v for k, v in reply.headers if k.startswith("access-control-")])

    def test_a_preflight_from_a_program_with_no_origin_is_204_without_cors(self):
        reply = self.call("OPTIONS", "/session", self.who.program())
        self.assertEqual(reply.status, 204)
        self.assertEqual(reply.cors_headers(), [])

    # -- served ---------------------------------------------------------------
    def test_a_window_is_served_on_every_route_and_answered_for_its_own_origin_only(self):
        for origin in OAIY_WINDOW_ORIGINS:
            sid = self.new_session(origin)
            for method, path, body in self.routes(sid)[1:]:
                with self.subTest(origin=origin, route=f"{method} {path}"):
                    reply = self.call(method, path, self.who.window(origin), body)
                    self.assertEqual(reply.status, 200, reply.body)
                    self.assertEqual(reply.all("access-control-allow-origin"), [origin])
                    self.assertIn("origin", reply.get("vary").lower())
                    self.assertIsNone(reply.get("access-control-allow-private-network"))
                    self.assertIsNone(reply.get("access-control-allow-credentials"))

    def test_a_window_is_served_whichever_fetch_site_it_reports(self):
        for site in ["cross-site", "same-site", "same-origin", None]:
            headers = [self.who.host(), ("Origin", "http://oaiyflows.localhost")] + ([("Sec-Fetch-Site", site)] if site else [])
            with self.subTest(site=site):
                reply = self.call("GET", "/health", headers)
                self.assertEqual(reply.status, 200)
                self.assertEqual(reply.get("access-control-allow-origin"), "http://oaiyflows.localhost")

    def test_a_program_with_the_right_host_and_no_origin_is_served_with_no_cors_headers(self):
        sid = self.new_session()
        for host in [f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}", f"LOCALHOST:{self.port}"]:
            for method, path, body in [("GET", "/health", None), ("GET", f"/session/{sid}/title", None), ("POST", "/session", {"config": {}})]:
                with self.subTest(host=host, route=f"{method} {path}"):
                    reply = self.call(method, path, self.who.program(host), body)
                    self.assertEqual(reply.status, 200, reply.body)
                    self.assertEqual(reply.cors_headers(), [])

    def test_a_typed_address_and_the_servers_own_origin_are_served(self):
        for site in ["none", "same-origin"]:
            reply = self.call("GET", "/health", [self.who.host(), ("Sec-Fetch-Site", site), ("Sec-Fetch-Mode", "navigate")])
            self.assertEqual(reply.status, 200, site)

    def test_a_program_that_speaks_http_1_0_with_a_host_is_served(self):
        reply = self.call("GET", "/health", self.who.program(), version="HTTP/1.0")
        self.assertEqual(reply.status, 200)

    def test_the_servers_answers_still_have_the_shapes_the_editor_reads(self):
        sid = self.new_session()
        window = self.who.window("http://oaiyflows.localhost")
        health = self.call("GET", "/health", window).json()
        self.assertEqual(health["ok"], True)
        self.assertEqual(health["browser"], "chromium")
        goto = self.call("POST", f"/session/{sid}/goto", window, {"url": "https://example.test/"}).json()
        self.assertEqual(goto, {"url": "https://example.test/", "title": "stub title", "status": 200})
        self.assertEqual(self.call("GET", f"/session/{sid}/html", window).json(), {"html": "<html>stub</html>"})
        self.assertEqual(self.call("POST", f"/session/{sid}/evaluate", window, {"script": "1+1"}).json(), {"result": "evaluated"})
        self.assertEqual(self.call("GET", f"/session/{sid}/nope", window).status, 404)
        self.assertEqual(self.call("DELETE", f"/session/{sid}", window).json(), {"ok": True})

    def test_the_server_says_who_it_lets_in_and_what_it_refused(self):
        self.call("GET", "/health", self.who.hosted_editor())
        out = self.server.output()
        self.assertIn("web pages allowed to call it: ", out)
        for origin in OAIY_WINDOW_ORIGINS:
            self.assertIn(origin, out)
        self.assertRegex(out, r"refused a [A-Z]+ request: (403|421) ")


class ConnectionLimits(unittest.TestCase):
    """The server takes one connection at a time (the Playwright sync API is bound
    to its thread), so what one connection may cost the callers behind it is
    bounded, and every answer ends its connection (HTTP/1.0, no keep-alive).

    Every wait here is bounded on the client's side: a server that fails a test
    makes it fail, not hang."""

    @classmethod
    def setUpClass(cls):
        cls.server = RunningServer(",".join(OAIY_WINDOW_ORIGINS))
        cls.port = cls.server.port
        cls.who = Personas(cls.port, free_port())

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def health(self, timeout):
        return raw_request(self.port, "GET", "/health", self.who.window("http://oaiyflows.localhost"), timeout=timeout)

    def test_an_idle_connection_holds_the_server_no_longer_than_the_timeout(self):
        # What a browser's preconnect is, from any page: a connection that never sends.
        limit = server_module.CONNECTION_TIMEOUT_SECS
        self.assertLessEqual(limit, 10, "the stall a page can cause is this long")
        idle = socket.create_connection(("127.0.0.1", self.port))
        self.addCleanup(idle.close)
        time.sleep(0.2)  # the server has taken it and waits on it
        reply = self.health(timeout=limit + 4)
        self.assertEqual(reply.status, 200)
        self.assertLess(reply.elapsed, limit + 3, "the caller behind an idle connection waited past the timeout")
        self.assertEqual(read_until_closed(idle, 3), b"", "and it is the server that let go of the idle connection")

    def test_a_refused_body_that_trickles_in_holds_the_server_for_the_drain_deadline_at_most(self):
        # A byte every 0.4 s never trips a per-read timeout; only a deadline for the whole body ends it.
        head = f"POST /session HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nOrigin: http://evil.example\r\nContent-Length: 500\r\n\r\n"
        trickle = socket.create_connection(("127.0.0.1", self.port))
        stop = threading.Event()

        def drip():
            while not stop.is_set():
                try:
                    trickle.sendall(b"x")
                except OSError:
                    return
                stop.wait(0.4)

        trickle.sendall(head.encode("latin-1"))
        thread = threading.Thread(target=drip, daemon=True)
        thread.start()
        self.addCleanup(trickle.close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(stop.set)
        time.sleep(0.3)
        reply = self.health(timeout=server_module.DRAIN_SECS + 6)
        self.assertEqual(reply.status, 200)
        self.assertLess(reply.elapsed, server_module.DRAIN_SECS + 3, "the trickling body held the server past its deadline")

    def test_an_http_0_9_request_is_refused_cleanly(self):
        for request in [
            b"GET /health\r\n\r\n",
            b"GET /session/s1/html\r\n\r\n",
            b"GET /health\r\nHost: 127.0.0.1:%d\r\n\r\n" % self.port,  # headers it cannot have: still refused
        ]:
            with self.subTest(request=request):
                with socket.create_connection(("127.0.0.1", self.port), timeout=8) as s:
                    s.sendall(request)
                    data = read_until_closed(s, 8)
                # HTTP/0.9 has no status line: the answer is the bare body.
                self.assertIn("HTTP/0.9", json.loads(data.decode("utf-8"))["error"])
        self.assertEqual(self.health(timeout=8).status, 200, "and the server carries on")
        self.assertNotIn("Traceback", self.server.output())

    def test_a_refused_post_with_a_second_request_in_its_body_gets_one_answer_and_the_connection_closes(self):
        inner = (
            f"POST /session HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nOrigin: http://oaiyflows.localhost\r\n"
            "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
        ).encode("latin-1")
        before = self.server.calls()
        reply = raw_request(self.port, "POST", "/session", self.who.page_on_another_loopback_port(), inner, timeout=3)
        self.assertEqual(reply.status, 403)
        self.assertEqual(reply.raw.count(b"HTTP/1."), 1, "one answer, not one for the request smuggled in the body")
        self.assertLess(reply.elapsed, 2, "and the connection was closed, not held open for another request")
        self.assertEqual(self.server.calls(), before, "the inner request, a window's own, did not run")

    def test_every_answer_closes_its_connection(self):
        window = self.who.window("http://oaiyflows.localhost")
        for method, path, headers, body in [
            ("GET", "/health", window, None),
            ("GET", "/health", self.who.program(), None),
            ("OPTIONS", "/session", window, None),
            ("GET", "/health", self.who.rebound_name(), None),
            ("POST", "/session", self.who.hosted_editor(), {"config": {}}),
        ]:
            with self.subTest(method=method, path=path):
                reply = raw_request(self.port, method, path, headers, body, timeout=3)
                self.assertLess(reply.elapsed, 2)


class NoAllowList(unittest.TestCase):
    """Started with nothing in OAIY_ALLOWED_ORIGINS (unset, empty, or only entries that are not exact origins)."""

    def run_with(self, allowed):
        server = RunningServer(allowed)
        self.addCleanup(server.close)
        who = Personas(server.port, free_port())
        for origin in OAIY_WINDOW_ORIGINS + ["https://oaiy.com", "http://127.0.0.1:3000", "null", "*"]:
            headers = [who.host(), ("Origin", origin), ("Sec-Fetch-Site", "cross-site")]
            reply = raw_request(server.port, "GET", "/health", headers)
            self.assertEqual(reply.status, 403, f"{allowed!r}: {origin}")
            self.assertEqual(reply.cors_headers(), [])
        # Programs are served, as they always were.
        reply = raw_request(server.port, "GET", "/health", who.program())
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.cors_headers(), [])
        # And nothing was run for the refused ones.
        self.assertEqual([c["call"] for c in server.calls()], ["launch"])

    def test_unset(self):
        self.run_with(None)

    def test_empty(self):
        self.run_with("")

    def test_a_wildcard_or_null_or_a_path_opens_nothing(self):
        self.run_with("*,null,http://oaiy.localhost/agent,oaiy.localhost")


if __name__ == "__main__":
    unittest.main()
