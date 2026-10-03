#!/usr/bin/env python3
"""Mutation testing of oaiy-relay-core: break the code in one place at a time (mutations.py, review_mutations.py) and require the tests to fail.

For each mutation: the old text must occur exactly once in its file; it is replaced; the crate's tests are built and run in three tiers, the first that fails kills the mutant:
  1. the fast tier (FAST: the unit tests and the quick integration tests),
  2. every other test target of the crate except the real-relay one (SLOW, found by listing tests/*.rs),
  3. the real PHP relay layer (relay_php), unless --no-php.
Outcomes, and nothing else:
  KILLED    a tier failed (a test failed, or the build of the tests failed at run time of a test binary).
  SURVIVED  every tier passed: a gap in the tests.
  INVALID   the old text is not there exactly once, or the mutant does not compile (retried once: a build can fail for a reason that is not the mutant).
  TIMEOUT   a test run did not end in time (a mutant that hangs, or a test with no deadline): reported apart, and not counted as a kill.
The exit status is 0 only when EVERY mutant is KILLED and the tree is clean at the end: INVALID, TIMEOUT and SURVIVED each make it 1.

A target file with CRLF endings is refused (exit 2): the anchors are LF text, the crate is LF in every checkout (`.gitattributes`), and patching around a CRLF checkout once made a
mutant look invalid. The file is restored from the bytes read before the change, in a `finally`, and the working tree is checked clean at the end.

Usage (with cargo on the path and CARGO_TARGET_DIR set to a directory of your own):
    python crates/oaiy-relay-core/mutation/run.py [--check] [--only P01,P02] [--from A06] [--list] [--no-php] [--out FILE]
`--check` builds nothing: it reports every mutation whose anchor is missing, repeated or a no-op (run it after any change to the code: a refactor moves anchors, and a mutant that
goes INVALID must be re-anchored to the same meaning, not dropped).

It changes files of the working tree in place for the length of each run: run it in a worktree nobody else is building in (a second `git worktree add --detach`, with its own
CARGO_TARGET_DIR, so that development carries on in the first one). Windows only for the time-out kill (`taskkill`); elsewhere the process is killed without its children.
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
CRATE = ROOT / "crates" / "oaiy-relay-core"
FAST = ["--lib", "--test", "vectors", "--test", "poll_fixture", "--test", "poll_differential", "--test", "client_stub", "--test", "stores", "--test", "pairing_ceremony",
        "--test", "pairing_stub", "--test", "aokie_fixtures", "--test", "json_differential", "--test", "loopback", "--test", "fuzz"]
FAST_NAMES = {FAST[i + 1] for i in range(1, len(FAST), 2)}
PHP = ["--test", "relay_php"]
TIMEOUT_BUILD = int(os.environ.get("MUT_TIMEOUT_BUILD", "900"))
TIMEOUT_TEST = int(os.environ.get("MUT_TIMEOUT_TEST", "240"))


def slow_tier():
    """Every integration test target but the fast ones and the real-relay one."""
    names = sorted(p.stem for p in (CRATE / "tests").glob("*.rs"))
    out = []
    for n in names:
        if n not in FAST_NAMES and n != "relay_php":
            out += ["--test", n]
    return out


def run(args, timeout):
    """Runs cargo; on a timeout kills that process tree (its own children only) and says so."""
    proc = subprocess.Popen(["cargo", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()
        out, _ = proc.communicate()
        return 1, out, True


def failed_tests(out):
    names = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("test ") and line.endswith("FAILED"):
            names.append(line[5:-len(" ... FAILED")])
    return names


def last_running(out):
    last = ""
    for line in out.splitlines():
        if line.strip().startswith("Running "):
            last = line.strip().split("(")[0].replace("Running ", "").strip()
    return last


def first_error(out):
    for line in out.splitlines():
        if line.startswith("error") and "aborting" not in line:
            return line[:160]
    return ""


def test(extra):
    """('PASSED' | 'KILLED' | 'TIMEOUT' | 'INVALID', why)."""
    base = ["test", "-p", "oaiy-relay-core", "--locked", *extra]
    code, out, timed_out = run(base + ["--no-run"], TIMEOUT_BUILD)
    if code != 0 and not timed_out:
        # One more try: a build can fail for a reason that is not the mutant (a file held by another process); a mutant that does not compile fails twice.
        code, out, timed_out = run(base + ["--no-run"], TIMEOUT_BUILD)
    if timed_out:
        return "TIMEOUT", "the build did not end in time"
    if code != 0:
        return "INVALID", "does not compile: " + first_error(out)
    code, out, timed_out = run(base + ["--", "--test-threads", "4"], TIMEOUT_TEST)
    if code == 0:
        return "PASSED", ""
    where = last_running(out)
    if timed_out:
        return "TIMEOUT", f"no end in {TIMEOUT_TEST} s in {where} (a mutant that hangs, or a test with no deadline)"
    tests = failed_tests(out)
    if tests:
        return "KILLED", f"{where}: {tests[0]}" + (f" (+{len(tests) - 1})" if len(tests) > 1 else "")
    return "KILLED", f"{where}: the run failed ({out.strip().splitlines()[-1][:100] if out.strip() else ''})"


def dirty_tree():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_anchors(chosen):
    """(problems, crlf_files): the anchors that are not there exactly once, and the target files that are not LF."""
    problems, crlf = [], set()
    for (mid, area, what, file, old, new, kind) in chosen:
        data = (ROOT / file).read_bytes().decode("utf-8")
        if "\r\n" in data:
            crlf.add(file)
        n = data.count(old)
        if n != 1:
            problems.append(f"{mid}: the old text occurs {n} times in {file}")
        elif old == new:
            problems.append(f"{mid}: old and new are the same")
    return problems, sorted(crlf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--from", dest="start", help="start at this id (to carry on after an interrupted run)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check that every anchor occurs exactly once; build nothing")
    ap.add_argument("--no-php", action="store_true", help="do not run the real-relay layer against a mutant that survives the other tiers")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "results.txt"))
    args = ap.parse_args()
    ids = [m[0] for m in MUTATIONS]
    assert len(ids) == len(set(ids)), "duplicate ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1}))
    chosen = [m for m in MUTATIONS if not args.only or m[0] in args.only.split(",")]
    if args.start:
        chosen = chosen[[m[0] for m in chosen].index(args.start):]
    if args.list:
        for m in chosen:
            print(f"{m[0]}  {m[1]:28} {m[2]}")
        print(len(chosen), "mutations")
        return 0
    problems, crlf = check_anchors(chosen)
    if crlf:
        print("these target files have CRLF endings; the crate is LF in every checkout (.gitattributes): re-checkout them first:\n  " + "\n  ".join(crlf))
        return 2
    if args.check:
        for p in problems:
            print("PROBLEM", p)
        print(f"{len(chosen)} mutations, {len(problems)} problems")
        return 1 if problems else 0

    dirty = dirty_tree()
    if dirty:
        print("the working tree has uncommitted changes; commit or stash them first:\n" + dirty)
        return 2
    started = time.time()
    slow = slow_tier()
    print("baseline ...", flush=True)
    status, why = test(FAST)
    if status == "PASSED" and slow:
        status, why = test(slow)
    if status != "PASSED":
        print("the baseline does not pass:", status, why)
        return 2
    lines = []
    tally = {"KILLED": 0, "SURVIVED": 0, "INVALID": 0, "TIMEOUT": 0}
    for (mid, area, what, file, old, new, kind) in chosen:
        t0 = time.time()
        path = ROOT / file
        original = path.read_bytes()
        text = original.decode("utf-8")
        if text.count(old) != 1:
            result = ("INVALID", f"the old text occurs {text.count(old)} times in {file}")
        else:
            try:
                path.write_bytes(text.replace(old, new, 1).encode("utf-8"))
                result = test(FAST)
                if result[0] == "PASSED" and slow:
                    result = test(slow)
                    if result[0] == "KILLED":
                        result = ("KILLED", "slow tier, " + result[1])
                if result[0] == "PASSED" and not args.no_php:
                    result = test(PHP)
                    if result[0] == "KILLED":
                        result = ("KILLED", "by the real PHP relay layer: " + result[1])
            finally:
                path.write_bytes(original)
        status = "SURVIVED" if result[0] == "PASSED" else result[0]
        tally[status] += 1
        line = f"{mid:4} {status:9} {kind:10} {area:28} {what}" + (f"\n         -> {result[1]}" if result[1] else "") + f"   [{int(time.time() - t0)} s]"
        lines.append(line)
        print(line, flush=True)
    dirty = dirty_tree()
    summary = (f"{len(chosen)} mutations: {tally['KILLED']} killed, {tally['SURVIVED']} survived, {tally['INVALID']} invalid, {tally['TIMEOUT']} timed out; "
               f"{int(time.time() - started)} s; tree {'clean' if not dirty else 'DIRTY: ' + dirty}")
    print(summary)
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n\n" + summary + "\n", encoding="utf-8")
    not_killed = tally["SURVIVED"] + tally["INVALID"] + tally["TIMEOUT"]
    return 0 if not dirty and not not_killed else (3 if dirty else 1)


if __name__ == "__main__":
    sys.exit(main())
