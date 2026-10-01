#!/usr/bin/env python3
"""
Checks, on the real Qwen3.8-Flash-Next, that a conversation the engine set aside in RAM comes back
instead of being read again, and that what it answers afterwards is what it would have answered
had it never left.  A manual acceptance run for a machine with the model loaded on its GPUs: it needs
the owner's agreement to a short model restart (the determinism runs stop and start the model).  Its
logic is tested without a model, against tools/mock_server.py (python -m unittest
tools/test_verify_parking.py, from crates/oaiy-llm-server).

Loopback only: the gateway at 127.0.0.1:8080 (POST /v1/chat/completions, model "Qwen3.8-Flash-Next")
and the Studio's log at 127.0.0.1:7860 (GET /api/logs?source=llm&after=N, and POST /api/llm/stop with
--restart).  Everything sent is synthetic text (an imaginary warehouse log); no real people, numbers or
business data.  temperature 0, top_p 1, seed 424242 and max_tokens 320 on every request (--max-tokens
changes it): every question asks for eight notes quoted word for word from all over the conversation,
so a wrong key/value row or recurrent state at any depth shows as a different quotation.

The gateway's key, if it has one, comes from the environment:   $env:OAIY_GATEWAY_KEY = "..."

THREE THINGS IT DOES (python verify_parking.py --help for the flags)

1. --mode sequence (default)
   Two synthetic conversations, A (about 20k tokens, the runner) and B (about 5k, a call's
   sub-agent), whose system prompts share only a 520-token-ish preamble, taking turns:
       A1 B1 A2 B2 A3 B3
   (A2 is A1 plus the model's reply plus a new turn; A3 the same on A2.)  For each request it prints
   the wall time, usage.prompt_tokens_details.cached_tokens, the cache_source the server reports
   (streaming oaiy_progress: none | memory | checkpoint | disk | ram) and the server's own
   `Qwen cache:` / `Qwen park:` / `Qwen restore:` log lines.  After the first round every request
   should be `ram` with only the new turn read; the script prints PASS or FAIL for that.
   Before parking, each of those switches read 20k tokens again in 31 to 36 s; the `Qwen park:` and
   `Qwen restore:` lines say how long the copies take instead (measured on 1 Oct 2026 on the owner's
   two-GPU machine, one run: 0.17 to 0.20 s to park 20.3k tokens, 0.15 s to restore them, the whole
   `Qwen cache:` line 0.21 to 0.26 s, 96 tokens read at every switch).  If any request in the run was
   not this script's (the desktop or the phone line reached the engine in between), the run is
   INCONCLUSIVE and says so: run it again when they are quiet.

2. The determinism check, in two runs on two FRESH servers (--mode reference, --mode swapped), then
   --mode compare.  Same prompts both times (--notes-a/--notes-b are fixed by --mode calibrate):
       reference:  A1 A2                 (no B: nothing is ever set aside; A2 continues the live state)
       swapped:    A1 B1 A2              (B1 sets A1 aside; A2 brings it back from RAM)
   compare asserts: A1's replies are equal in the two runs (else the runs differed before any swap and
   nothing can be concluded: INCONCLUSIVE), A2's server-reported cache source is `ram` in the swapped
   run and memory/checkpoint in the reference (else the check proves nothing: VACUOUS), and A2's
   completions are byte for byte identical.  A cold re-read of A2 is NOT used as the reference: a read
   from nothing is chunked differently and need not match to the last bit.
   "Fresh" matters: the script refuses a server that has already answered a request unless
   --allow-used.  Stop the model with   POST http://127.0.0.1:7860/api/llm/stop   (or pass --restart,
   which does that for you; the next request then starts it, which takes a minute or more), and run the
   next mode.  Add --fixed-reply to answer A1 with a fixed synthetic assistant turn in A2's prompt
   instead of the model's own reply, which makes A2's prompt independent of A1's answer and also
   exercises the checkpoint-in-RAM path (A2 then goes back to a checkpoint of A1 and not to its end).

3. --mode calibrate: how many notes make about 20k and 5k tokens; prints --notes-a/--notes-b to reuse.

4. --baseline: compare two --mode reference runs (two fresh servers, no swap in either): whether the
   engine reproduces ITSELF across restarts, which is what makes a PASS of the swapped comparison mean
   something (and a difference of it a parking bug and not a difference between two starts).

Typical order (each reference/swapped run on a fresh server; --restart stops the model first):
    python verify_parking.py --mode calibrate
    python verify_parking.py --mode sequence    --notes-a N --notes-b M
    python verify_parking.py --mode reference   --notes-a N --notes-b M --restart --out reference.json
    python verify_parking.py --mode swapped     --notes-a N --notes-b M --restart --out swapped.json
    python verify_parking.py --mode compare     --ref reference.json --swapped swapped.json
    python verify_parking.py --mode reference   --notes-a N --notes-b M --restart --out reference2.json
    python verify_parking.py --mode compare --baseline --ref reference.json --swapped reference2.json
Studio's Incognito setting must be off (the engine never sets aside an incognito request).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

GATEWAY = os.environ.get("OAIY_GATEWAY", "http://127.0.0.1:8080")
STUDIO = os.environ.get("OAIY_STUDIO", "http://127.0.0.1:7860")
KEY = os.environ.get("OAIY_GATEWAY_KEY", "")
MODEL = "Qwen3.8-Flash-Next"
SEED = 424242
MAX_TOKENS = 320  # eight quoted notes (about 35 tokens each); --max-tokens changes it

for _url in (GATEWAY, STUDIO):
    if urllib.parse.urlparse(_url).hostname not in ("127.0.0.1", "localhost", "::1"):
        sys.exit(f"refusing {_url}: this script talks to loopback only")

# ---------------------------------------------------------------------------------------------
# Synthetic text.  A "note" is one invented sentence of about 35 tokens.

COLORS = ["amber", "blue", "cedar", "dusty", "ember", "frost", "green"]
THINGS = ["crate", "lantern", "ledger", "bucket", "compass", "hinge", "spool", "ladder", "kettle", "anchor", "barrel"]

PREAMBLE = " ".join(
    f"Rule {i}: always cite the note number, never guess a shelf, and keep every answer under two sentences."
    for i in range(1, 27)
)  # about 520 tokens; the two conversations share this and nothing else of their system prompts

FIXED_REPLY = "Understood. The note says the item was moved and logged, and nothing else is recorded about it."


def note(tag: str, i: int) -> str:
    return (f"Note {tag}-{i:05d}: the {COLORS[i % 7]} {THINGS[(i * 3) % 11]} was moved to shelf {(i * 7) % 97} "
            f"and logged by unit {(i * 13) % 29}; keep it until step {(i * 5) % 61} is done.")


def notes(tag: str, n: int) -> str:
    return "\n".join(note(tag, i) for i in range(n))


def question(tag: str, k: int, n_notes: int) -> str:
    # Eight notes spread over the whole conversation, quoted word for word: a wrong key/value row or
    # recurrent state at any depth shows up as a different quotation, which a one-line answer would hide.
    n = max(n_notes, 1)
    ids = [(k * 37 + j * max(n // 8, 1)) % n for j in range(8)]
    listed = ", ".join(f"{tag}-{i:05d}" for i in ids)
    return f"Question {k}: quote notes {listed} exactly, word for word, one per line, nothing else."


def system_prompt(tag: str) -> str:
    # The preamble first, then text of its own: the first message boundary lies past what they share.
    return PREAMBLE + f"\n\nYou are the assistant for the imaginary warehouse log {tag}. Answer from its notes only."


class Conversation:
    def __init__(self, tag: str, n_notes: int):
        self.tag, self.n = tag, n_notes
        self.replies: list[str] = []

    def messages(self, turn: int, fixed_reply: bool = False) -> list[dict]:
        """The request for turn `turn` (1-based): every earlier turn with its reply, then this one."""
        msgs = [{"role": "system", "content": system_prompt(self.tag)},
                {"role": "user", "content": notes(self.tag, self.n) + "\n\n" + question(self.tag, 1, self.n)}]
        for k in range(1, turn):
            msgs.append({"role": "assistant", "content": FIXED_REPLY if fixed_reply else self.replies[k - 1]})
            msgs.append({"role": "user", "content": question(self.tag, k + 1, self.n)})
        return msgs


# ---------------------------------------------------------------------------------------------
# HTTP

def http(method: str, url: str, body=None, headers=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    return urllib.request.urlopen(req, timeout=timeout)


def auth() -> dict:
    return {"Authorization": f"Bearer {KEY}"} if KEY else {}


def chat(messages: list[dict], max_tokens: int | None = None) -> dict:
    """One streaming request; returns the reply, usage, the server's cache report and the wall time."""
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    body = {"model": MODEL, "messages": messages, "temperature": 0, "top_p": 1, "seed": SEED,
            "max_tokens": max_tokens, "repeat_penalty": 1.0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    out = {"text": "", "reasoning": "", "usage": None, "progress": None, "first_progress_s": None}
    with http("POST", GATEWAY + "/v1/chat/completions", body, auth(), timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except ValueError:
                continue
            if "oaiy_progress" in j:
                p = j["oaiy_progress"]
                if out["progress"] is None:
                    out["first_progress_s"] = time.time() - t0
                if p.get("cache_source") is not None:
                    out["progress"] = p
            for c in j.get("choices") or []:
                delta = c.get("delta") or {}
                out["text"] += delta.get("content") or ""
                # A request that lands in thinking mode spends its 48 tokens here: compared too.
                out["reasoning"] += delta.get("reasoning_content") or ""
            if j.get("usage"):
                out["usage"] = j["usage"]
    out["wall_s"] = time.time() - t0
    u = out["usage"] or {}
    out["prompt_tokens"] = u.get("prompt_tokens")
    out["completion_tokens"] = u.get("completion_tokens")
    out["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    p = out["progress"] or {}
    out["cache_source"] = p.get("cache_source")
    out["previous_prefix_tokens"] = p.get("previous_prefix_tokens")
    return out


class LogTail:
    """The server's log through the Studio (read-only GET), from where it stands now."""

    def __init__(self):
        self.last = 0
        self.last = max([l["n"] for l in self._fetch(0)] or [0])

    @staticmethod
    def _fetch(after: int) -> list[dict]:
        with http("GET", f"{STUDIO}/api/logs?source=llm&after={after}", timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace")).get("lines", [])

    def new(self, settle: float = 0.6) -> list[str]:
        time.sleep(settle)  # the server's stderr reaches the ring a moment after the reply
        lines = self._fetch(self.last)
        if lines:
            self.last = max(l["n"] for l in lines)
        return [l["line"].strip() for l in lines]

    @staticmethod
    def fresh() -> bool:
        """No request has been answered since the server last started (or was last stopped)."""
        lines = [l["line"] for l in LogTail._fetch(0)]
        marks = [i for i, l in enumerate(lines) if "studio: starting" in l or "oaiy-llm-server stopped" in l]
        return not any("Qwen cache:" in l for l in lines[max(marks or [0]):])


RELEVANT = re.compile(r"Qwen (cache|park|restore|swap)\b|Qwen: \d+ prompt tokens")


def studio_incognito() -> bool:
    try:
        with http("GET", f"{STUDIO}/api/config", timeout=15) as r:
            return bool((json.loads(r.read()).get("privacy") or {}).get("incognito"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def check_environment(args) -> None:
    if studio_incognito() and not args.allow_incognito:
        sys.exit("Studio's Incognito is on: the engine never sets an incognito request aside, so there is nothing to check. "
                 "Turn it off (or pass --allow-incognito to see that it does nothing).")
    try:
        with http("GET", GATEWAY + "/v1/models", headers=auth(), timeout=15) as r:
            models = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        if MODEL not in models:
            sys.exit(f"{MODEL} is not served here (models: {models})")
    except urllib.error.HTTPError as e:
        sys.exit(f"the gateway refused /v1/models ({e.code}): set OAIY_GATEWAY_KEY")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"cannot reach {GATEWAY}: {e}")


def stop_model() -> None:
    print("POST /api/llm/stop: the model unloads now and starts again at the next request (a minute or more).")
    http("POST", STUDIO + "/api/llm/stop", {}, timeout=120).read()
    time.sleep(3)


# ---------------------------------------------------------------------------------------------
# Running

class Run:
    def __init__(self, args):
        self.args = args
        self.tail = LogTail() if not args.no_log else None
        self.rows: list[dict] = []

    def ask(self, label: str, conv: Conversation, turn: int) -> dict:
        msgs = conv.messages(turn, self.args.fixed_reply)
        r = chat(msgs)
        r["label"] = label
        r["server_log"] = [l for l in (self.tail.new() if self.tail else []) if RELEVANT.search(l)]
        if r["cache_source"] is None:
            # The gateway may not pass oaiy_progress through: the server's own line says it too.
            for l in r["server_log"]:
                m = re.search(r"Qwen cache: (\d+)/(\d+) tokens from (\w+)", l)
                if m:
                    r["cache_source"] = m.group(3)
        # More than one `Qwen cache:` line in this request's window: someone else's request (the desktop,
        # the phone line) reached the engine in between, and it is no longer the run that was planned.
        r["interleaved"] = sum("Qwen cache:" in l for l in r["server_log"]) > 1
        conv.replies.append(r["text"])
        self.rows.append(r)
        shown = (r["text"] or r["reasoning"]).strip()[:70]
        print(f"{label:>3}  {r['wall_s']:7.1f}s  prompt {r['prompt_tokens']!s:>6}  cached {r['cached_tokens']!s:>6}  "
              f"source {r['cache_source']!s:<10}  reply: {shown!r}")
        for l in r["server_log"]:
            print(f"        | {l}")
        if r["interleaved"]:
            print("        ! another request reached the engine during this one: do not trust this run")
        return r


def tokens_per_note(args) -> tuple[float, float]:
    """(tokens a note adds, tokens everything else adds), from two small probes."""
    probes = []
    for n in (20, 100):
        conv = Conversation("CAL", n)
        r = chat(conv.messages(1), max_tokens=1)
        probes.append((n, r["prompt_tokens"]))
    (n1, p1), (n2, p2) = probes
    slope = (p2 - p1) / (n2 - n1)
    return slope, p1 - slope * n1


def pick_notes(args) -> tuple[int, int]:
    if args.notes_a and args.notes_b:
        return args.notes_a, args.notes_b
    slope, base = tokens_per_note(args)
    a, b = int((args.tokens_a - base) / slope), int((args.tokens_b - base) / slope)
    print(f"calibrated: {slope:.2f} tokens a note, {base:.0f} for the rest -> --notes-a {a} --notes-b {b}")
    return a, b


def mode_calibrate(args) -> int:
    check_environment(args)
    a, b = pick_notes(argparse.Namespace(**{**vars(args), "notes_a": 0, "notes_b": 0}))
    print(f"use: --notes-a {a} --notes-b {b}")
    return 0


def mode_sequence(args) -> int:
    check_environment(args)
    na, nb = pick_notes(args)
    a, b = Conversation("A", na), Conversation("B", nb)
    run = Run(args)
    plan = [("A1", a, 1), ("B1", b, 1), ("A2", a, 2), ("B2", b, 2), ("A3", a, 3), ("B3", b, 3)]
    res = {}
    for label, conv, turn in plan:
        res[label] = run.ask(label, conv, turn)
    # After the first round every request is a swap-in that reads only what is new.
    fails = 0
    print()
    if any(r["interleaved"] for r in res.values()):
        print("INCONCLUSIVE: another request reached the engine during the run (the `!` lines above): run it again when the phone and the Agent are quiet.")
        return 2
    for label, conv, turn in plan[2:]:
        prev = res[f"{conv.tag}{turn - 1}"]
        r = res[label]
        grown = r["prompt_tokens"] - prev["prompt_tokens"]
        uncached = r["prompt_tokens"] - (r["cached_tokens"] or 0)
        # What is new: the growth of the prompt (the previous reply is in it, but those tokens were
        # decoded and are held already, so this allows more than the new question) and a few closing tokens.
        allowed = grown + 64
        ok = r["cache_source"] == "ram" and uncached <= allowed
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}: source {r['cache_source']}, {uncached} of {r['prompt_tokens']} tokens read "
              f"(the new turn is about {grown}; allowed {allowed}), {r['wall_s']:.1f}s")
    print("\nCompare the wall times of A2/B2/A3/B3 with A1/B1: a switch used to cost the whole prompt (36 s at 21k).")
    print("Look for `Qwen park:` / `Qwen restore:` lines for what the copies really took.")
    return 1 if fails else 0


def mode_recorded(args, with_b: bool) -> int:
    if not args.notes_a or not args.notes_b:
        sys.exit("run --mode calibrate first and pass --notes-a/--notes-b: both runs need identical prompts")
    if args.restart:
        stop_model()
    check_environment(args)
    if not args.allow_used and not LogTail.fresh():
        sys.exit("this server has already answered requests: stop it (POST /api/llm/stop, or pass --restart) so that "
                 "A1 is read from nothing in both runs; or pass --allow-used")
    a, b = Conversation("A", args.notes_a), Conversation("B", args.notes_b)
    run = Run(args)
    plan = [("A1", a, 1)] + ([("B1", b, 1)] if with_b else []) + [("A2", a, 2)]
    for label, conv, turn in plan:
        run.ask(label, conv, turn)
    doc = {"mode": "swapped" if with_b else "reference", "notes_a": args.notes_a, "notes_b": args.notes_b,
           "fixed_reply": args.fixed_reply, "rows": run.rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"\nwritten to {args.out}")
    return 0


def mode_compare(args) -> int:
    with open(args.ref, encoding="utf-8") as f:
        ref = json.load(f)
    with open(args.swapped, encoding="utf-8") as f:
        swp = json.load(f)
    row = lambda doc, label: next(r for r in doc["rows"] if r["label"] == label)
    if (ref["notes_a"], ref["notes_b"], ref["fixed_reply"]) != (swp["notes_a"], swp["notes_b"], swp["fixed_reply"]):
        sys.exit("the two runs were not made with the same --notes-a/--notes-b/--fixed-reply")
    if args.baseline:
        # Two reference runs on two fresh servers: how far the engine agrees with ITSELF without any swap.
        if ref["mode"] != "reference" or swp["mode"] != "reference":
            sys.exit("--baseline compares two --mode reference runs (--ref and --swapped both from --mode reference)")
    elif ref["mode"] != "reference" or swp["mode"] != "swapped":
        sys.exit("--ref must come from --mode reference and --swapped from --mode swapped")
    a1r, a1s, a2r, a2s = row(ref, "A1"), row(swp, "A1"), row(ref, "A2"), row(swp, "A2")
    # What a reply is: its reasoning (if the request went into thinking mode) and its content.
    whole = lambda r: (r.get("reasoning") or "") + "\u241e" + (r.get("text") or "")
    print(f"A1 reference {whole(a1r)!r}\nA1 swapped   {whole(a1s)!r}")
    print(f"A2 reference ({a2r['cache_source']}, cached {a2r['cached_tokens']}/{a2r['prompt_tokens']}) {whole(a2r)!r}")
    print(f"A2 swapped   ({a2s['cache_source']}, cached {a2s['cached_tokens']}/{a2s['prompt_tokens']}) {whole(a2s)!r}")
    if any(r.get("interleaved") for doc in (ref, swp) for r in doc["rows"]):
        print("\nINCONCLUSIVE: another request reached the engine during a run (see the `!` lines when it was recorded).")
        return 2
    empty = lambda r: not ((r.get("text") or "").strip() or (r.get("reasoning") or "").strip())
    if empty(a2r) or empty(a2s):
        print("\nINCONCLUSIVE: A2 produced no text in a run, so equality would prove nothing.")
        return 2
    if a2r.get("completion_tokens") != a2s.get("completion_tokens"):
        print(f"\nFAIL: A2 generated {a2r.get('completion_tokens')} tokens in the reference and {a2s.get('completion_tokens')} in the swapped run.")
        return 1
    if not args.fixed_reply and whole(a1r) != whole(a1s):
        print("\nINCONCLUSIVE: A1 was answered differently in the two runs, before any swap. A2's prompts differ; compare nothing.")
        print("(Run again, or use --fixed-reply so that A2's prompt does not depend on A1's answer.)")
        return 2
    if args.baseline:
        if whole(a1r) == whole(a1s) and whole(a2r) == whole(a2s):
            print("\nBASELINE: two fresh servers, no swap, gave identical A1 and A2: the engine is deterministic here, so a PASS in the swapped comparison means something.")
            return 0
        print("\nBASELINE DIFFERS: the engine does not reproduce itself across restarts, so a difference in the swapped comparison cannot be blamed on parking.")
        return 1
    if a2s["cache_source"] != "ram" or a2r["cache_source"] not in ("memory", "checkpoint"):
        print(f"\nVACUOUS: A2 came from {a2s['cache_source']!r} in the swapped run (want 'ram') and from "
              f"{a2r['cache_source']!r} in the reference (want memory/checkpoint): the comparison proves nothing.")
        return 2
    if a2r["prompt_tokens"] != a2s["prompt_tokens"]:
        print("\nINCONCLUSIVE: A2's prompts have different lengths in the two runs.")
        return 2
    if whole(a2r) == whole(a2s):
        print("\nPASS: A2 answered byte for byte the same with the state swapped out and back as with it never leaving.")
        return 0
    print("\nFAIL: A2's completions differ. The state that came back from RAM is not the state that left (or the engine is not deterministic: "
          "run --mode reference twice and compare them first).")
    return 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # a console that is not UTF-8 must not stop a run on a quotation
        stream.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sequence", "calibrate", "reference", "swapped", "compare"], default="sequence")
    p.add_argument("--notes-a", type=int, default=0, help="notes in conversation A (about 20k tokens); default: calibrate")
    p.add_argument("--notes-b", type=int, default=0, help="notes in conversation B (about 5k tokens); default: calibrate")
    p.add_argument("--tokens-a", type=int, default=20000)
    p.add_argument("--tokens-b", type=int, default=5000)
    p.add_argument("--fixed-reply", action="store_true", help="answer A1 with a fixed assistant turn in A2's prompt")
    p.add_argument("--restart", action="store_true", help="POST /api/llm/stop first (the next request starts the model again)")
    p.add_argument("--allow-used", action="store_true", help="do not insist on a freshly started server")
    p.add_argument("--allow-incognito", action="store_true")
    p.add_argument("--no-log", action="store_true", help="do not read the server log through the Studio")
    p.add_argument("--max-tokens", type=int, default=None, help="tokens per answer (default 320: eight quoted notes)")
    p.add_argument("--baseline", action="store_true", help="with --mode compare: both files are --mode reference runs")
    p.add_argument("--out", default="parking-run.json")
    p.add_argument("--ref")
    p.add_argument("--swapped")
    args = p.parse_args()
    if args.max_tokens:
        global MAX_TOKENS
        MAX_TOKENS = args.max_tokens
    if args.mode == "calibrate":
        return mode_calibrate(args)
    if args.mode == "sequence":
        return mode_sequence(args)
    if args.mode in ("reference", "swapped"):
        return mode_recorded(args, with_b=args.mode == "swapped")
    if not (args.ref and args.swapped):
        sys.exit("--mode compare needs --ref and --swapped")
    return mode_compare(args)


if __name__ == "__main__":
    sys.exit(main())
