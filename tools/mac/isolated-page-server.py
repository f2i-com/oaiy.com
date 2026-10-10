"""tools/mac/isolated-page-server.py PORT POLICY: a page on http://127.0.0.1:PORT/ served with the headers that
isolate a page (opener policy same-origin, embedder policy POLICY), for tools/mac/webview-probe.swift to ask a Mac's
webview about. It answers every GET with the same small page and ends when it is stopped."""
import http.server
import sys

PORT, POLICY = int(sys.argv[1]), sys.argv[2]
PAGE = b"<!doctype html><meta charset=utf-8><title>probe</title><p>probe</p>"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", POLICY)
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
