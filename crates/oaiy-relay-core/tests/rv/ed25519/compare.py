import json, sys, os, collections
d = os.path.dirname(__file__)
cases = json.load(open(os.path.join(d, "cases.json")))
sods = json.load(open(os.path.join(d, "out_php.json")))
node = json.load(open(os.path.join(d, "out_node.json")))
py = json.load(open(os.path.join(d, "out_py.json")))
rs = json.load(open(os.path.join(d, "out_rust.json")))


def norm(v):
    if v is True:
        return "ACC"
    if v is False:
        return "rej"
    if v == "keyrefused":
        return "KEYREF"
    return "err"


rows = collections.defaultdict(list)
for c in cases:
    i = c["id"]
    t = (norm(sods[i]["verify"]), norm(node[i]["verify"]), norm(py[i]["verify"]), norm(rs[i]["verify"]))
    rows[(c["group"], t, sods[i]["relay_key_valid"], rs[i]["key"].split(":")[0])].append(i)
print("group | libsodium node(openssl) py(openssl) CRATE | relay key valid | crate key | n")
for k in sorted(rows, key=lambda k: (k[0], k[1])):
    g, t, rkv, ck = k
    print(f"{g:13s} | {t[0]:5s} {t[1]:5s} {t[2]:5s} {t[3]:6s} | {str(rkv):5s} | {ck:7s} | {len(rows[k])}   e.g. {rows[k][0]}")
print()
# disagreements between libsodium and the crate
bad = [(c["id"]) for c in cases if (norm(sods[c["id"]]["verify"]), norm(rs[c["id"]]["verify"])) not in [("ACC", "ACC"), ("rej", "rej"), ("rej", "KEYREF")]]
print("libsodium vs crate: verdicts that differ in a way that is not 'both refuse':", len(bad))
for b in bad:
    print("   ", b, sods[b]["verify"], rs[b]["verify"], rs[b]["key"])
bad2 = [c["id"] for c in cases if sods[c["id"]]["relay_key_valid"] != (rs[c["id"]]["key"] == "ok")]
print("relay key validity (Crypto::isValidEd25519Public) vs crate VerifyKey::from_bytes differ on", len(bad2))
for b in bad2:
    print("   ", b, "relay:", sods[b]["relay_key_valid"], "crate:", rs[b]["key"])
