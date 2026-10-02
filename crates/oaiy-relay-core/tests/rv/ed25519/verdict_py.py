import json, sys
import cryptography
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

cases = json.load(open(sys.argv[1]))
out = {}
for c in cases:
    pk, msg, sig = bytes.fromhex(c["pk"]), bytes.fromhex(c["msg"]), bytes.fromhex(c["sig"])
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
        v = True
    except InvalidSignature:
        v = False
    except Exception as e:
        v = "error:" + type(e).__name__
    out[c["id"]] = {"verify": v}
json.dump(out, open(sys.argv[2], "w"))
print(len(out), "verdicts from python cryptography", cryptography.__version__)
