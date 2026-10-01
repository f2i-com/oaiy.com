#!/usr/bin/env python3
"""A stand-in for the gateway + Studio log, on 127.0.0.1 only, to test verify_parking.py's own logic
(parsing, the table, PASS/FAIL, compare).  It is not the engine: tokens are words, and it applies the
same rules (continue live, else swap in a stashed state, else read; park what a prompt displaces)."""
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BAD = os.environ.get("MOCK_BAD") == "1"          # answer differently after a swap-in
NO_PROGRESS = os.environ.get("MOCK_NO_PROGRESS") == "1"  # a gateway that drops oaiy_progress
state = {"live": [], "stash": [], "lines": [], "n": 0, "started": False}
lock = threading.Lock()


def log(text):
    state["n"] += 1
    state["lines"].append({"n": state["n"], "t": int(time.time() * 1000), "line": text})


def start():
    if not state["started"]:
        state["started"] = True
        log("studio: starting oaiy-llm-server.exe --model x")


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def serve(prompt_words):
    live, stash = state["live"], state["stash"]
    common = lcp(live, prompt_words)
    source, start_at = "none", 0
    if live and common == len(live) and common < len(prompt_words):
        source, start_at = "memory", common
    else:
        best = max(stash, key=lambda e: lcp(e, prompt_words), default=None)
        if best is not None and lcp(best, prompt_words) == len(best) and len(best) >= 1024:
            if len(live) >= 1024:
                stash.append(live)
                log(f"  Qwen park: stashed {len(live)} tokens (1 MB) in 0.001s; {len(stash)} states, 1 MB held")
            stash.remove(best)
            log(f"  Qwen restore: {len(best)} tokens (1 MB) in 0.001s")
            state["live"] = list(best)
            source, start_at = "ram", len(best)
        elif len(live) >= 1024 and common * 2 < len(live):
            stash.append(live)
            log(f"  Qwen park: stashed {len(live)} tokens (1 MB) in 0.001s; {len(stash)} states, 1 MB held")
    log(f"  Qwen cache: {start_at}/{len(prompt_words)} tokens from {source} (common {common}) in 0.001s")
    digest = hashlib.sha256(" ".join(prompt_words).encode()).hexdigest()
    reply = "The shelf is " + digest[:6] + ("!" if BAD and source == "ram" else ".")
    if os.environ.get("MOCK_EMPTY") == "1":
        reply = ""
    if os.environ.get("MOCK_INTERLEAVE") == "1":
        log("  Qwen cache: 0/10 tokens from none (common 0) in 0.000s")  # somebody else's request
    state["live"] = prompt_words + reply.split()
    log(f"  Qwen: {len(prompt_words)} prompt tokens ({start_at} cached) in 0.01s; 7 generated in 0.01s")
    return source, start_at, reply


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with lock:
            if self.path.startswith("/api/config"):
                return self.send_json({"privacy": {"incognito": False}})
            if self.path.startswith("/v1/models"):
                return self.send_json({"data": [{"id": "Qwen3.8-Flash-Next"}]})
            if self.path.startswith("/api/logs"):
                after = int(self.path.split("after=")[1].split("&")[0])
                return self.send_json({"lines": [l for l in state["lines"] if l["n"] > after]})
        self.send_json({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/api/llm/stop"):
            with lock:
                state.update(live=[], stash=[], started=False)
                log("studio: oaiy-llm-server stopped")
            return self.send_json({"state": "stopped"})
        if not self.path.startswith("/v1/chat/completions"):
            return self.send_json({}, 404)
        words = []
        for m in body["messages"]:
            words += ["<|" + m["role"] + "|>"] + m["content"].split()
        words.append("<|assistant|>")  # the header the reply continues (as a real prompt ends)
        with lock:
            start()
            source, cached, reply = serve(words)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        send = lambda o: self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode())
        send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        if not NO_PROGRESS:
            send({"choices": [], "oaiy_progress": {"prompt_done": 0, "prompt_total": len(words) - cached, "cached_tokens": cached, "cache_source": source, "previous_prefix_tokens": cached}})
        for w in reply.split(" "):
            send({"choices": [{"index": 0, "delta": {"content": w + " "}}]})
        send({"choices": [], "usage": {"prompt_tokens": len(words), "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": cached}}})
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 0), H)
    print(srv.server_address[1], flush=True)
    srv.serve_forever()
