"""The reviewer's runner for rv_mutations.py (a re-write of the implementer's mutation/run.py with: CRLF-aware patches, the compile error of an invalid mutant shown, a distinct
TIMEOUT status, and a non-zero exit status when anything survives or is invalid).

env: RV_ROOT (the clone to mutate), CARGO_TARGET_DIR. usage: python rv_mut_run.py [--only A,B] [--no-php] [--out FILE] [--timeout-test S]
"""
import argparse, os, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rv_mutations import MUTATIONS  # noqa

ROOT = pathlib.Path(os.environ["RV_ROOT"])
FAST = ["--lib", "--test", "vectors", "--test", "poll_fixture", "--test", "poll_differential", "--test", "client_stub", "--test", "stores", "--test", "pairing_ceremony",
        "--test", "pairing_stub", "--test", "aokie_fixtures", "--test", "json_differential", "--test", "loopback", "--test", "fuzz"]
PHP = ["--test", "relay_php"]


def run(args, timeout):
    proc = subprocess.Popen(["cargo", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out, _ = proc.communicate()
        return 1, out, True


def failed_tests(out):
    return [l.strip()[5:-len(" ... FAILED")] for l in out.splitlines() if l.strip().startswith("test ") and l.strip().endswith("FAILED")]


def running(out):
    last = ""
    for l in out.splitlines():
        if l.strip().startswith("Running "):
            last = l.strip().split("(")[0].replace("Running ", "").strip()
    return last


def first_error(out):
    for l in out.splitlines():
        if l.startswith("error") and "aborting" not in l:
            return l[:160]
    return ""


def test(extra, tmo_test):
    base = ["test", "-p", "oaiy-relay-core", "--locked", *extra]
    code, out, _ = run(base + ["--no-run"], 900)
    if code != 0:
        code, out, _ = run(base + ["--no-run"], 900)
    if code != 0:
        return "INVALID", "does not compile: " + first_error(out)
    code, out, timed_out = run(base + ["--", "--test-threads", "4"], tmo_test)
    if code == 0:
        return "PASSED", ""
    if timed_out:
        return "TIMEOUT", f"in {running(out)}"
    tests = failed_tests(out)
    if tests:
        return "KILLED", f"{running(out)}: {tests[0]}" + (f" (+{len(tests) - 1})" if len(tests) > 1 else "")
    return "KILLED", f"{running(out)}: the run failed ({(out.strip().splitlines() or [''])[-1][:100]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-php", action="store_true")
    ap.add_argument("--out", default="rv_mut_results.txt")
    ap.add_argument("--timeout-test", type=int, default=300)
    a = ap.parse_args()
    chosen = [m for m in MUTATIONS if not a.only or m[0] in a.only.split(",")]
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("dirty tree:\n" + dirty)
        return 2
    t0 = time.time()
    print("baseline ...", flush=True)
    st, why = test(FAST, a.timeout_test)
    if st != "PASSED":
        print("baseline does not pass:", st, why)
        return 2
    lines, tally = [], {}
    for (mid, area, what, file, old, new, kind) in chosen:
        path = ROOT / file
        original = path.read_bytes()
        text = original.decode("utf-8")
        crlf = "\r\n" in text
        o, n = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
        started = time.time()
        if text.count(o) != 1:
            result = ("INVALID", f"the old text occurs {text.count(o)} times in {file}")
        else:
            try:
                path.write_bytes(text.replace(o, n, 1).encode("utf-8"))
                result = test(FAST, a.timeout_test)
                if result[0] == "PASSED" and not a.no_php:
                    r2 = test(PHP, a.timeout_test)
                    result = ("KILLED", "by the real PHP relay layer: " + r2[1]) if r2[0] in ("KILLED", "TIMEOUT") else r2
            finally:
                path.write_bytes(original)
        status = "SURVIVED" if result[0] == "PASSED" else result[0]
        tally[status] = tally.get(status, 0) + 1
        line = f"{mid:4} {status:9} {kind:10} {area:22} {what}" + (f"\n         -> {result[1]}" if result[1] else "") + f"   [{int(time.time() - started)} s]"
        lines.append(line)
        print(line, flush=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = f"{len(chosen)} mutants: {tally}; {int(time.time() - t0)} s; tree {'clean' if not dirty else 'DIRTY: ' + dirty}"
    print(summary)
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n\n" + summary + "\n", encoding="utf-8")
    bad = tally.get("SURVIVED", 0) + tally.get("INVALID", 0) + tally.get("TIMEOUT", 0)
    return 0 if not dirty and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
