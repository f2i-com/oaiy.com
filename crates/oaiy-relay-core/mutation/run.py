#!/usr/bin/env python3
"""Mutation testing of oaiy-relay-core: break the code in one place at a time (mutations.py) and require the tests to fail.

For each mutation: the old text must occur exactly once in its file; it is replaced; the crate's tests are built (a mutant that does not compile is INVALID, not killed) and run
(the first failing test binary ends the run: the mutant is KILLED); a mutant that survives the fast layers is run against the real PHP relay layer as well (relay_php), and is
reported SURVIVED only if that passes too. The file is restored from the bytes read before the change, in a `finally`, and the working tree is checked clean at the end.

Usage (from anywhere, with cargo on the path and CARGO_TARGET_DIR set to a directory of your own):
    python crates/oaiy-relay-core/mutation/run.py [--only P01,P02] [--from A06] [--list] [--no-php] [--out FILE]

It changes files of the working tree in place for the length of each run: run it in a worktree nobody else is building in. Do not run it in a checkout whose files a watcher is
building (a desktop dev server): a mutant is on disk for a minute at a time.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mutations import MUTATIONS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
FAST = ["--lib", "--test", "vectors", "--test", "poll_fixture", "--test", "poll_differential", "--test", "client_stub", "--test", "stores", "--test", "pairing_ceremony",
        "--test", "pairing_stub", "--test", "aokie_fixtures", "--test", "json_differential", "--test", "loopback", "--test", "fuzz"]
PHP = ["--test", "relay_php"]
TIMEOUT_BUILD = 900
TIMEOUT_TEST = 240


def run(args, timeout):
    """Runs cargo; on a timeout kills that process tree (its own children only) and says so."""
    proc = subprocess.Popen(["cargo", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out, _ = proc.communicate()
        return 1, out, True


def failed_tests(out):
    names = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("test ") and line.endswith("FAILED"):
            names.append(line[5:-len(" ... FAILED")])
    return names


def first_running(out):
    last = ""
    for line in out.splitlines():
        if line.strip().startswith("Running "):
            last = line.strip().split("(")[0].replace("Running ", "").strip()
    return last


def test(extra, name):
    base = ["test", "-p", "oaiy-relay-core", "--locked", *extra]
    code, out, _ = run(base + ["--no-run"], TIMEOUT_BUILD)
    if code != 0:
        return "INVALID", "does not compile"
    code, out, timed_out = run(base + ["--", "--test-threads", "4"], TIMEOUT_TEST)
    if code == 0:
        return "PASSED", ""
    tests = failed_tests(out)
    where = first_running(out)
    if timed_out:
        return "KILLED", f"timeout in {where} (a mutant that hangs)"
    if tests:
        return "KILLED", f"{where}: {tests[0]}" + (f" (+{len(tests) - 1})" if len(tests) > 1 else "")
    return "KILLED", f"{where}: the run failed ({out.strip().splitlines()[-1][:100] if out.strip() else ''})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--from", dest="start", help="start at this id (to carry on after an interrupted run)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-php", action="store_true", help="do not run the real-relay layer against a mutant that survives the fast layers")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "results.txt"))
    args = ap.parse_args()
    chosen = [m for m in MUTATIONS if not args.only or m[0] in args.only.split(",")]
    if args.start:
        chosen = chosen[[m[0] for m in chosen].index(args.start):]
    ids = [m[0] for m in MUTATIONS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    if args.list:
        for m in chosen:
            print(f"{m[0]}  {m[1]:28} {m[2]}")
        print(len(chosen), "mutations")
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("the working tree has uncommitted changes; commit or stash them first:\n" + dirty)
        return 2
    started = time.time()
    print("baseline ...", flush=True)
    status, why = test(FAST, "baseline")
    if status != "PASSED":
        print("the baseline does not pass:", status, why)
        return 2
    lines = []
    tally = {"KILLED": 0, "SURVIVED": 0, "INVALID": 0}
    for (mid, area, what, file, old, new, kind) in chosen:
        path = ROOT / file
        original = path.read_bytes()
        text = original.decode("utf-8")
        if text.count(old) != 1:
            result = ("INVALID", f"the old text occurs {text.count(old)} times in {file}")
        else:
            try:
                path.write_bytes(text.replace(old, new, 1).encode("utf-8"))
                result = test(FAST, mid)
                if result[0] == "PASSED" and not args.no_php:
                    result = test(PHP, mid)
                    if result[0] == "KILLED":
                        result = ("KILLED", "by the real PHP relay layer: " + result[1])
            finally:
                path.write_bytes(original)
        status = "SURVIVED" if result[0] == "PASSED" else result[0]
        tally[status] += 1
        line = f"{mid:4} {status:9} {kind:10} {area:28} {what}" + (f"\n         -> {result[1]}" if result[1] else "")
        lines.append(line)
        print(line, flush=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = f"{len(chosen)} mutations: {tally['KILLED']} killed, {tally['SURVIVED']} survived, {tally['INVALID']} invalid; {int(time.time() - started)} s; tree {'clean' if not dirty else 'DIRTY: ' + dirty}"
    print(summary)
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n\n" + summary + "\n", encoding="utf-8")
    return 0 if not dirty else 3


if __name__ == "__main__":
    sys.exit(main())
