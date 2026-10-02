#!/usr/bin/env python3
"""Show that the two readers of the pairing fixtures are not vacuous.

verify_fixtures.py and verify_fixtures.mjs re-derive the recorded sealed tokens and the ceremony of Appendix A3 and report
"<n> checks, <m> mismatches". A reader that checked nothing would say "0 mismatches" too. This runs both, with
OAIY_FIXTURE_DIR pointing at a copy of sealed-token.json and pairing-ceremony.json, first as recorded (both must pass) and then
once for each of a list of damages made to the copy (each reader must report at least one mismatch for each):

    python selftest_fixtures.py

The last line is "<n> damaged copies, each refused by both readers" and the exit status 0, or the escapes are listed.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SEALED = json.loads((HERE / "sealed-token.json").read_text(encoding="utf-8"))
CEREMONY = json.loads((HERE / "pairing-ceremony.json").read_text(encoding="utf-8"))


def flip_b64u(s: str, pos: int) -> str:
    raw = bytearray(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))
    raw[pos] ^= 1
    return base64.urlsafe_b64encode(bytes(raw)).rstrip(b"=").decode()


def damages():
    def d(label, fn):
        s, c = copy.deepcopy(SEALED), copy.deepcopy(CEREMONY)
        fn(s, c)
        return label, s, c

    steps = lambda c: c["steps"]
    yield d("a bit of the first box flipped", lambda s, c: s["opens"][0].update(sealedToken=flip_b64u(s["opens"][0]["sealedToken"], 50)))
    yield d("the first token's recorded hash is wrong", lambda s, c: s["opens"][0].update(plaintextSha256="0" * 64))
    yield d("the second token's recorded length is wrong", lambda s, c: s["opens"][1].update(plaintextLength=62))
    yield d("a box that opens is listed as refused", lambda s, c: s["refused"].append({"label": "a good box", "sealedToken": s["opens"][0]["sealedToken"]}))
    yield d("the wrong recipient is the right one", lambda s, c: s.update(wrongRecipient=copy.deepcopy(s["recipient"])))
    yield d("the recipient's public key is not its secret's", lambda s, c: s["recipient"].update(x25519Public=s["wrongRecipient"]["x25519Public"]))
    yield d("the offer MAC of the create request is damaged", lambda s, c: steps(c)[0]["request"]["body"].update(mac=flip_b64u(steps(c)[0]["request"]["body"]["mac"], 3)))
    yield d("the offer text has an extra space", lambda s, c: steps(c)[0]["request"]["body"].update(offer=steps(c)[0]["request"]["body"]["offer"] + " "))
    yield d("the pid of the ceremony is another", lambda s, c: c.update(pid=flip_b64u(c["pid"], 2)))

    def receipt(s, c):
        steps(c)[4]["request"]["body"]["receipt"]["signature"] = flip_b64u(steps(c)[4]["request"]["body"]["receipt"]["signature"], 5)
        steps(c)[5]["response"]["body"]["receipt"]["signature"] = steps(c)[4]["request"]["body"]["receipt"]["signature"]
    yield d("the approval receipt's signature is damaged (in the request and in what the phone reads)", receipt)

    # The receipt the phone reads carries the grants it was signed over (README Interpretation 60); each change of them is refused.
    read = lambda c: steps(c)[5]["response"]["body"]["receipt"]
    yield d("the grants are missing from the receipt the phone reads", lambda s, c: read(c).pop("grants"))
    yield d("the grants of the receipt the phone reads are in another order", lambda s, c: read(c).update(grants=list(reversed(read(c)["grants"]))))
    yield d("a grant is added to the receipt the phone reads", lambda s, c: read(c)["grants"].append("takeover"))
    yield d("a grant is taken from the receipt the phone reads", lambda s, c: read(c)["grants"].pop(0))
    yield d("one grant of the receipt the phone reads is altered", lambda s, c: read(c)["grants"].__setitem__(0, "end_caller"))
    yield d("a grant appears twice in the receipt the phone reads", lambda s, c: read(c)["grants"].insert(0, read(c)["grants"][0]))
    yield d("the receipt the phone reads names a grant nobody knows", lambda s, c: read(c)["grants"].append("zzz_unknown"))
    yield d("the grants of the decision are altered (they are no longer the ones signed and returned)", lambda s, c: steps(c)[4]["request"]["body"]["grants"].append("takeover"))
    yield d("the grants of the receipt the phone reads are an empty list", lambda s, c: read(c).update(grants=[]))

    def keys(s, c):
        steps(c)[4]["request"]["body"]["phone"]["ed25519"] = SEALED["wrongRecipient"]["x25519Public"]
    yield d("the approval names another Ed25519 key than the phone answered with", keys)
    yield d("the phone's response text is altered (its claims say another issuedAt)",
            lambda s, c: steps(c)[2]["request"]["body"].update(response=re.sub(r'"issuedAt":(\d+)', lambda m: '"issuedAt":' + str(int(m.group(1)) + 1), steps(c)[2]["request"]["body"]["response"], count=1)))
    yield d("the desktop's poll returns another body than the phone posted", lambda s, c: steps(c)[3]["response"]["body"]["items"][0].update(body=steps(c)[3]["response"]["body"]["items"][0]["body"] + " "))
    yield d("the sealed token the phone reads does not open", lambda s, c: steps(c)[5]["response"]["body"].update(sealedToken=flip_b64u(steps(c)[5]["response"]["body"]["sealedToken"], 60)))
    yield d("the short authentication string is another", lambda s, c: c.update(sas="AAAA-AAAA-AAAA-A"))
    yield d("a step is missing", lambda s, c: c["steps"].pop())


def run(cmd, folder):
    env = dict(os.environ, OAIY_FIXTURE_DIR=str(folder), PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=180, cwd=HERE)
    m = re.search(r"(\d+) checks, (\d+) mismatches", p.stdout)
    return p.returncode, (int(m.group(2)) if m else None), (p.stdout + p.stderr)[-300:]


def main() -> int:
    node = shutil.which("node")
    readers = [("Python", [sys.executable, str(HERE / "verify_fixtures.py")])] + ([("Node", [node, str(HERE / "verify_fixtures.mjs")])] if node else [])
    if not node:
        print("  !! node is not on PATH: only the Python reader was tried")
    escapes = []
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)

        def put(s, c):
            (folder / "sealed-token.json").write_text(json.dumps(s), encoding="utf-8")
            (folder / "pairing-ceremony.json").write_text(json.dumps(c), encoding="utf-8")

        put(SEALED, CEREMONY)
        for name, cmd in readers:
            code, bad, tail = run(cmd, folder)
            if code != 0 or bad != 0:
                escapes.append(f"the {name} reader does not pass the undamaged copy: {tail}")
        for label, s, c in damages():
            n += 1
            put(s, c)
            for name, cmd in readers:
                code, bad, tail = run(cmd, folder)
                if code == 0:      # a reader that crashes on a damaged copy has not passed it either
                    escapes.append(f"ESCAPED the {name} reader: {label}")
    for e in escapes:
        print("  " + e)
    print(f"{n} damaged copies, each refused by {'both readers' if node else 'the Python reader'}" if not escapes else f"{len(escapes)} problems")
    return 1 if escapes else 0


if __name__ == "__main__":
    raise SystemExit(main())
