#!/usr/bin/env python3
"""Read the Aokie fixtures (this folder) with the rules of the shipped Aokie decoders, transcribed in aokie_decoders.py, and no
code of the relay. Python 3.9+ standard library only.

  python verify_aokie_fixtures.py [-v]

What it checks:
  * every recorded admission is accepted by the decoder of its role (the plugin's AdmissionResponse, the phone's validate_admission),
    its bearer verifies under the recorded test secret and says what the response says, every TURN credential recomputes with hmac
    and hashlib alone, and the three relay URLs are byte-identical in every answer;
  * every challenge is accepted by EndpointChallengeFrame::validate and is built from its bearer;
  * every frames answer has the shape the carriers read, the sender metadata is the relay's own record, a frame comes back as it
    went in, and every stream body parses to the same events whole and split at every byte offset (and with CRLF line ends);
  * every error has exactly the three members the phone's error decoder allows;
  * ice.json equals what an independent minter computes, including FormLogic's known answers;
  * and the mirror has teeth: about a hundred and fifty damaged copies of the recorded documents are each REFUSED (or, where the
    Rust decoder degrades, accepted with the relay advertisement dropped) exactly as the decoders' rules say.

Exit status 0 and a last line "<n> checks, 0 mismatches" when everything holds.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aokie_decoders as D  # noqa: E402

checks = 0
mismatches: list[str] = []
VERBOSE = "-v" in sys.argv[1:]        # print the reason each damaged document was refused for


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        mismatches.append(f"{label}: {detail}" if detail else label)


def accepts(label: str, fn) -> object:
    global checks
    checks += 1
    try:
        return fn()
    except D.Refused as e:
        mismatches.append(f"{label}: refused ({e})")
        return None


def refuses(label: str, fn) -> None:
    global checks
    checks += 1
    try:
        fn()
    except D.Refused as e:
        if VERBOSE:
            print(f"  refused: {label}  ->  {e}")
        return
    except Exception as e:  # a crash is not a refusal
        mismatches.append(f"{label}: crashed instead of refusing ({type(e).__name__}: {e})")
        return
    mismatches.append(f"{label}: ACCEPTED a document the decoder refuses")


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


DEL = object()


def mut(doc, path: str, value=DEL):
    d = copy.deepcopy(doc)
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    last = int(parts[-1]) if isinstance(node, list) else parts[-1]
    if value is DEL:
        del node[last]
    else:
        node[last] = value
    return d


admission = load("admission.json")
challenge = load("challenge.json")
frames = load("frames.json")
streams = load("stream.json")
errors = load("errors.json")
ice = load("ice.json")
NOW = admission["relay"]["clock"]
SECRET = bytes.fromhex(admission["relay"]["admissionSecretHex"])
TURN_SECRET = admission["relay"]["turnSecret"]
IDS = admission["identities"]
GRANT_NAMES = D.GRANTS

# ================================================================================================ admissions
section = "admissions"
relay_urls = set()
cases = {c["name"]: c for c in admission["cases"]}
for c in admission["cases"]:
    req, res = c["request"]["body"], c["response"]
    name = c["name"]
    if res["status"] != 200:
        accepts(f"{name}: an error answer has the three members", lambda r=res: D.compat_error(r["body"], r["status"]))
        continue
    body = res["body"]
    if c["role"] == "plugin":
        expect = {k: req[k] for k in ("appId", "pluginId", "endpointPublicKey", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")}
        out = accepts(f"{name}: the plugin's AdmissionResponse accepts it", lambda: D.plugin_admission(body, expect, NOW))
        if out:
            check(f"{name}: the plugin adopts the relay transport", out["transport"] == "relay")
            check(f"{name}: the plugin's connection lifetime is 80 seconds (90 less its 10 second margin)", out["lifetime"] == 80)
    else:
        session = {"gatewayUrl": body["gatewayUrl"], "appId": req["appId"], "deviceId": req["deviceId"], "discoveryRelayOnly": body["relayOnly"]}
        out = accepts(f"{name}: the phone's validate_admission accepts it", lambda: D.mobile_admission(body, session, req["holderKeyThumbprint"], NOW))
        if out:
            check(f"{name}: the phone adopts the relay transport", out["transport"] == "relay")
    # the bearer
    claims = accepts(f"{name}: the bearer verifies under the recorded secret", lambda: D.read_bearer(body["accessToken"], SECRET))
    if claims:
        want_keys = (["aud", "appId", "subjectId", "role", "holderKeyThumbprint", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash", "scopes", "dsk", "exp", "jti"]
                     if c["role"] == "plugin" else ["aud", "appId", "subjectId", "role", "holderKeyThumbprint", "expectedPeerKeyThumbprint", "scopes", "dsk", "exp", "jti"])
        check(f"{name}: the claims are in the signer's order", list(claims) == want_keys, str(list(claims)))
        check(f"{name}: aud, role, app and subject are the answer's",
              (claims["aud"], claims["role"], claims["appId"], claims["subjectId"]) == ("aokie-v2-gateway", "plugin" if c["role"] == "plugin" else "mobile", body["appId"], body["subjectId"]))
        check(f"{name}: holder and scopes are the answer's", claims["holderKeyThumbprint"] == body["holderKeyThumbprint"] and claims["scopes"] == body["scopes"])
        check(f"{name}: the claims name the desktop the party belongs to", claims["dsk"] == IDS["desktopDevice"])
        check(f"{name}: exp is the answer's expiresAt, 90 seconds ahead", claims["exp"] == body["expiresAt"] == NOW + 90 and body["expiresIn"] == 90)
        check(f"{name}: the jti is adm_ and 32 hex characters", re.fullmatch(r"adm_[0-9a-f]{32}", claims["jti"]) is not None)
        if c["role"] == "plugin":
            check(f"{name}: the roster and its hash are the request's",
                  (claims["approvedPeerKeyThumbprints"], claims["peerRosterRevision"], claims["peerRosterHash"]) == (req["approvedPeerKeyThumbprints"], req["peerRosterRevision"], req["peerRosterHash"]))
            check(f"{name}: the roster hash recomputes", claims["peerRosterHash"] == D.peer_roster_hash(claims["peerRosterRevision"], claims["approvedPeerKeyThumbprints"]))
            check(f"{name}: the plugin's scopes are its two", body["scopes"] == ["state_read", "rtc_signal"])
        else:
            check(f"{name}: the expected peer is the desktop endpoint the phone was paired to", claims["expectedPeerKeyThumbprint"] == body["expectedPeerKeyThumbprint"] == IDS["desktopEndpointThumbprint"])
            check(f"{name}: the device record's grants are the scopes", body["device"]["grants"] == body["scopes"])
            check(f"{name}: every scope is one of the fourteen", all(g in GRANT_NAMES for g in body["scopes"]) and "state_read" in body["scopes"])
        check(f"{name}: the bearer is small enough for the decoders", 16 <= len(body["accessToken"]) <= 16384)
    # ICE: every entry the way a coturn recomputes it
    role = "plugin" if c["role"] == "plugin" else "mobile"
    for s in body["iceServers"]:
        if s["username"]:
            exp_s, ident = s["username"].split(":", 1)
            check(f"{name}: the TURN username is <expiry>:<opaque id>", int(exp_s) == s["expiresAt"] and ident == D.turn_opaque_id(TURN_SECRET, role, body["appId"], body["subjectId"]))
            check(f"{name}: the TURN credential is base64(HMAC-SHA1(secret, username))", s["credential"] == D.turn_credential(TURN_SECRET, s["username"]))
            check(f"{name}: no device id is in the TURN username", body["subjectId"] not in s["username"] and IDS["desktopDevice"] not in s["username"])
        else:
            check(f"{name}: a STUN entry has empty username and credential and no expiresAt", s["credential"] == "" and "expiresAt" not in s and all(u.startswith(("stun:", "stuns:")) for u in s["urls"]))
    turn = [s for s in body["iceServers"] if s["username"]]
    check(f"{name}: turnCredentialExpiresAt is the earliest TURN expiry or null", body["turnCredentialExpiresAt"] == (min(s["expiresAt"] for s in turn) if turn else None))
    check(f"{name}: relayOnly needs TURN", (not body["relayOnly"]) or bool(turn))
    relay_urls.add(json.dumps({k: v for k, v in body["relay"].items() if k != "mode"}, sort_keys=True))
    check(f"{name}: the gateway URL is wss on the relay's host", body["gatewayUrl"] == "wss://relay.example.com/v2/realtime")
    check(f"{name}: no desktopConnection and no scopeCompatibility", "desktopConnection" not in body and "scopeCompatibility" not in body)
check("the three relay URLs are byte-identical in every admission (the plugin's cursor domain survives a rotation)", len(relay_urls) == 1, str(relay_urls))
polls = [c for c in admission["cases"] if c["response"]["status"] == 200 and c["response"]["body"]["relay"].get("mode") == "poll"]
check("poll-mode answers carry relay.mode and are still decoded by both carriers (an unknown member of relay is ignored)", len(polls) >= 2)

# ---- the damaged copies of the two main answers
P = cases["plugin, stream, STUN and TURN"]
PB, PR = P["response"]["body"], P["request"]["body"]
P_EXPECT = {k: PR[k] for k in ("appId", "pluginId", "endpointPublicKey", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")}
M = cases["phone A, stream, STUN and TURN"]
MB, MR = M["response"]["body"], M["request"]["body"]
M_SESSION = {"gatewayUrl": MB["gatewayUrl"], "appId": MR["appId"], "deviceId": MR["deviceId"], "discoveryRelayOnly": MB["relayOnly"]}
TURN_I = 1                                       # iceServers[1] is the TURN entry
n_plugin_neg = 0


def plugin_refused(label, doc, expect=None):
    global n_plugin_neg
    n_plugin_neg += 1
    refuses("plugin AdmissionResponse refuses " + label, lambda: D.plugin_admission(doc, expect or P_EXPECT, NOW))


def plugin_degrades(label, doc):
    out = accepts("plugin AdmissionResponse accepts " + label, lambda: D.plugin_admission(doc, P_EXPECT, NOW))
    check("plugin AdmissionResponse falls back to the WebSocket gateway for " + label, out is not None and out["relay"] is None)


for m in D.PLUGIN_MEMBERS:
    if m == "relay":
        out = accepts("plugin AdmissionResponse accepts an answer with no relay member (the WebSocket gateway stays the transport)", lambda: D.plugin_admission(mut(PB, "relay", DEL), P_EXPECT, NOW))
        check("an answer with no relay member selects the WebSocket gateway", out is not None and out["transport"] == "websocket")
    else:
        plugin_refused(f"an answer without {m}", mut(PB, m, DEL))
plugin_refused("desktopConnection (deny_unknown_fields)", dict(PB, desktopConnection={}))
plugin_refused("scopeCompatibility (deny_unknown_fields)", dict(PB, scopeCompatibility={}))
plugin_refused("tokenType bearer", mut(PB, "tokenType", "bearer"))
plugin_refused("role mobile", mut(PB, "role", "mobile"))
plugin_refused("another appId", mut(PB, "appId", "other"))
plugin_refused("another subjectId", mut(PB, "subjectId", "other"))
OTHER_KEY = D.b64u(bytes(range(32)))
plugin_refused("a well-formed endpoint key that is not the one sent", mut(PB, "endpointPublicKey", {"algorithm": "ed25519", "publicKey": OTHER_KEY, "thumbprint": D.thumbprint_of_public_key(OTHER_KEY)}))
plugin_refused("another holder", mut(PB, "holderKeyThumbprint", IDS["phoneAThumbprint"]))
plugin_refused("a roster that is not the one sent", mut(PB, "approvedPeerKeyThumbprints", PB["approvedPeerKeyThumbprints"][:1]))
plugin_refused("a roster in another order", mut(PB, "approvedPeerKeyThumbprints", list(reversed(PB["approvedPeerKeyThumbprints"]))))
plugin_refused("another revision", mut(PB, "peerRosterRevision", PB["peerRosterRevision"] + 1))
plugin_refused("another roster hash", mut(PB, "peerRosterHash", "A" * 43))
for v in (0, 10):
    plugin_refused(f"expiresIn {v} (the margin is 10 seconds)", mut(PB, "expiresIn", v))
plugin_refused("expiresIn 301", mut(PB, "expiresIn", 301))
plugin_refused("expiresIn as a string", mut(PB, "expiresIn", "90"))
plugin_refused("expiresIn as a float", mut(PB, "expiresIn", 90.0))
plugin_refused("expiresIn as true (a bool is not a u64)", mut(PB, "expiresIn", True))
plugin_refused("expiresAt as true", mut(PB, "expiresAt", True))
plugin_refused("a roster revision of true where the plugin sent 1 (Python says True == 1; a u64 does not)", dict(PB, peerRosterRevision=True), dict(P_EXPECT, peerRosterRevision=1))
# Damaged keys that the plugin sent itself (so the echo comparison cannot be what refuses them): only the key's own rules can.
BAD_TH = dict(PB["endpointPublicKey"], thumbprint="A" * 43)
plugin_refused("an endpoint key whose thumbprint does not recompute, echoed as sent",
               dict(PB, endpointPublicKey=BAD_TH, holderKeyThumbprint=BAD_TH["thumbprint"]), dict(P_EXPECT, endpointPublicKey=BAD_TH))
BAD_ALG = dict(PB["endpointPublicKey"], algorithm="Ed25519")
plugin_refused("an endpoint key with the algorithm in capitals, echoed as sent",
               dict(PB, endpointPublicKey=BAD_ALG), dict(P_EXPECT, endpointPublicKey=BAD_ALG))
plugin_refused("a padded endpoint public key, echoed as sent", dict(PB, endpointPublicKey=dict(PB["endpointPublicKey"], publicKey=PB["endpointPublicKey"]["publicKey"] + "=")),
               dict(P_EXPECT, endpointPublicKey=dict(PB["endpointPublicKey"], publicKey=PB["endpointPublicKey"]["publicKey"] + "=")))
plugin_refused("expiresAt 10 seconds ahead", mut(PB, "expiresAt", NOW + 10))
plugin_refused("expiresAt 301 seconds ahead", mut(PB, "expiresAt", NOW + 301))
plugin_refused("expiresAt in the past", mut(PB, "expiresAt", NOW - 5))
plugin_refused("an empty bearer", mut(PB, "accessToken", ""))
plugin_refused("a bearer with a control character", mut(PB, "accessToken", "aokie-adm-v2.\n"))
plugin_refused("a bearer over 16 KiB", mut(PB, "accessToken", "a" * 16385))
plugin_refused("a gateway URL on ws://", mut(PB, "gatewayUrl", "ws://relay.example.com/v2/realtime"))
plugin_refused("a gateway URL with credentials", mut(PB, "gatewayUrl", "wss://user:pw@relay.example.com/v2/realtime"))
plugin_refused("a gateway URL with a fragment", mut(PB, "gatewayUrl", "wss://relay.example.com/v2/realtime#x"))
plugin_refused("an appId with a space", mut(PB, "appId", "a b"))
plugin_refused("relayOnly as a string", mut(PB, "relayOnly", "false"))
plugin_refused("scopes that are not strings", mut(PB, "scopes", [1]))
plugin_refused("turnCredentialExpiresAt as a string", mut(PB, "turnCredentialExpiresAt", str(NOW + 600)))
plugin_refused("turnCredentialExpiresAt off by one from the TURN entry", mut(PB, "turnCredentialExpiresAt", PB["turnCredentialExpiresAt"] + 1))
plugin_refused("turnCredentialExpiresAt null while a TURN entry exists", mut(PB, "turnCredentialExpiresAt", None))
plugin_refused("a TURN entry without a username", mut(PB, f"iceServers.{TURN_I}.username", ""))
plugin_refused("a TURN entry without a credential", mut(PB, f"iceServers.{TURN_I}.credential", ""))
plugin_refused("a TURN entry without expiresAt", mut(PB, f"iceServers.{TURN_I}.expiresAt", DEL))
plugin_refused("a TURN entry whose expiresAt is null", mut(PB, f"iceServers.{TURN_I}.expiresAt", None))
plugin_refused("a TURN expiry 30 seconds ahead (it must be more)", dict(mut(PB, f"iceServers.{TURN_I}.expiresAt", NOW + 30), turnCredentialExpiresAt=NOW + 30))
plugin_refused("a TURN expiry over 24 hours ahead", dict(mut(PB, f"iceServers.{TURN_I}.expiresAt", NOW + 86401), turnCredentialExpiresAt=NOW + 86401))
plugin_refused("a STUN entry without its empty username", mut(PB, "iceServers.0.username", DEL))
plugin_refused("a STUN entry with a username", mut(PB, "iceServers.0.username", "x"))
plugin_refused("a STUN entry with a credential", mut(PB, "iceServers.0.credential", "x"))
plugin_refused("a STUN entry with an expiresAt", mut(PB, "iceServers.0.expiresAt", NOW + 600))
plugin_refused("an ICE server with an unknown member", mut(PB, "iceServers.0.realm", "x"))
plugin_refused("an ICE server without urls", mut(PB, "iceServers.0.urls", []))
plugin_refused("an ICE server with nine urls", mut(PB, "iceServers.0.urls", [f"stun:s{i}.example.com" for i in range(9)]))
plugin_refused("an ICE url that is https", mut(PB, "iceServers.0.urls", ["https://stun.example.com"]))
plugin_refused("nine ICE servers", mut(PB, "iceServers", [PB["iceServers"][0]] * 9))
plugin_refused("nine STUN-only ICE servers with no TURN promised (only the count is wrong)", dict(PB, iceServers=[PB["iceServers"][0]] * 9, turnCredentialExpiresAt=None))
plugin_refused("a STUN and a TURN url in one entry", mut(PB, "iceServers.0.urls", ["stun:stun.example.com:3478", "turn:turn.example.com:3478"]))
plugin_refused("relayOnly true with no TURN entry", dict(PB, relayOnly=True, iceServers=[PB["iceServers"][0]], turnCredentialExpiresAt=None))
plugin_refused("an endpoint key with another algorithm", mut(PB, "endpointPublicKey.algorithm", "Ed25519"))
plugin_refused("an endpoint key whose thumbprint does not recompute", mut(PB, "endpointPublicKey.thumbprint", "A" * 43))
plugin_refused("an endpoint key with an extra member", mut(PB, "endpointPublicKey.extra", 1))
plugin_refused("a roster revision as a float", mut(PB, "peerRosterRevision", 7.0))
plugin_refused("a roster revision that is negative", mut(PB, "peerRosterRevision", -1))
# the additive relay advertisement degrades instead of failing
plugin_degrades("a relay member that is not an object", mut(PB, "relay", "https://relay.example.com"))
plugin_degrades("a relay member without streamUrl", mut(PB, "relay.streamUrl", DEL))
plugin_degrades("a relay URL on http", mut(PB, "relay.streamUrl", "http://relay.example.com/v1/aokie-companion/relay/stream"))
plugin_degrades("relay URLs on two origins", mut(PB, "relay.framesUrl", "https://elsewhere.example.com/v1/aokie-companion/relay/frames"))
plugin_degrades("relay URLs on two ports", mut(PB, "relay.framesUrl", "https://relay.example.com:8443/v1/aokie-companion/relay/frames"))
plugin_degrades("relay URLs that are all on http (one origin, but not https)", dict(PB, relay={k: v.replace("https://", "http://") for k, v in PB["relay"].items()}))
plugin_degrades("a relay URL with credentials", mut(PB, "relay.challengeUrl", "https://u:p@relay.example.com/v1/aokie-companion/relay/challenge"))
plugin_degrades("a relay URL with a fragment", mut(PB, "relay.challengeUrl", "https://relay.example.com/v1/aokie-companion/relay/challenge#x"))
plugin_degrades("a relay URL that is not absolute", mut(PB, "relay.challengeUrl", "/v1/aokie-companion/relay/challenge"))
plugin_degrades("a relay URL that is not a string", mut(PB, "relay.framesUrl", 5))
out = accepts("plugin AdmissionResponse keeps a relay it can use when the answer adds a member (mode)", lambda: D.plugin_admission(mut(PB, "relay.mode", "poll"), P_EXPECT, NOW))
check("an unknown member of relay is ignored (RelayEndpoints is not deny_unknown_fields)", out is not None and out["relay"] is not None)
out = accepts("plugin AdmissionResponse accepts any JSON value in device", lambda: D.plugin_admission(mut(PB, "device", None), P_EXPECT, NOW))
check("device is read by nobody", out is not None)

n_phone_neg = 0


def phone_refused(label, doc, holder=None, session=None):
    global n_phone_neg
    n_phone_neg += 1
    refuses("phone validate_admission refuses " + label, lambda: D.mobile_admission(doc, session or M_SESSION, holder or MR["holderKeyThumbprint"], NOW))


for m in D.PHONE_MEMBERS:
    if m == "relay":
        out = accepts("phone AdmissionResponse accepts an answer without relay (the member has a default)", lambda: D.mobile_admission(mut(MB, "relay", DEL), M_SESSION, MR["holderKeyThumbprint"], NOW))
        check("an answer without relay decodes and stays on the WebSocket gateway", out is not None and out["transport"] == "websocket")
    elif m == "iceServers":
        out = accepts("phone AdmissionResponse accepts an answer without iceServers when no TURN credential is promised (the member has a default)",
                      lambda: D.mobile_admission(dict(mut(MB, "iceServers", DEL), turnCredentialExpiresAt=None), M_SESSION, MR["holderKeyThumbprint"], NOW))
        check("an answer without iceServers decodes to none", out is not None and out["iceServers"] == [])
        phone_refused("an answer without iceServers that still promises a TURN expiry", mut(MB, "iceServers", DEL))
    else:
        phone_refused(f"an answer without {m}", mut(MB, m, DEL))
for m in D.DEVICE_MEMBERS:
    phone_refused(f"a device record without {m}", mut(MB, "device." + m, DEL))
phone_refused("a device record with an extra member (DeviceRecord is deny_unknown_fields)", mut(MB, "device.email", "x"))
phone_refused("an extra member (deny_unknown_fields)", dict(MB, desktopConnection={}))
phone_refused("tokenType bearer", mut(MB, "tokenType", "bearer"))
phone_refused("a bearer of 15 bytes", mut(MB, "accessToken", "a" * 15))
phone_refused("expiresIn 0", mut(MB, "expiresIn", 0))
phone_refused("expiresIn 301", mut(MB, "expiresIn", 301))
phone_refused("expiresAt now", mut(MB, "expiresAt", NOW))
phone_refused("expiresAt 301 seconds ahead", mut(MB, "expiresAt", NOW + 301))
phone_refused("another gateway URL than the session's", mut(MB, "gatewayUrl", "wss://elsewhere.example.com/v2/realtime"))
phone_refused("another appId", mut(MB, "appId", "other"))
phone_refused("another subject", mut(MB, "subjectId", IDS["phoneB"]))
phone_refused("role plugin", mut(MB, "role", "plugin"))
phone_refused("another holder", mut(MB, "holderKeyThumbprint", IDS["phoneBThumbprint"]))
phone_refused("an expected peer equal to the holder", mut(MB, "expectedPeerKeyThumbprint", MB["holderKeyThumbprint"]))
phone_refused("relayOnly other than the session's", mut(MB, "relayOnly", not MB["relayOnly"]))
phone_refused("a device record of another app", mut(MB, "device.appId", "other"))
phone_refused("a device record of another subject", mut(MB, "device.subjectId", IDS["phoneB"]))
phone_refused("a device record with role plugin", mut(MB, "device.role", "plugin"))
phone_refused("grants that are not the scopes", mut(MB, "device.grants", MB["device"]["grants"][:-1]))
phone_refused("grants in another order", mut(MB, "device.grants", list(reversed(MB["device"]["grants"]))))
phone_refused("an empty displayName", mut(MB, "device.displayName", ""))
phone_refused("a displayName of 121 bytes", mut(MB, "device.displayName", "n" * 121))
phone_refused("an empty approvedAt", mut(MB, "device.approvedAt", ""))
phone_refused("scopes without state_read", dict(mut(MB, "scopes", ["caller_read"]), device=dict(MB["device"], grants=["caller_read"])))
phone_refused("an unknown scope", dict(mut(MB, "scopes", ["state_read", "delete_all"]), device=dict(MB["device"], grants=["state_read", "delete_all"])))
phone_refused("no scopes", dict(mut(MB, "scopes", []), device=dict(MB["device"], grants=[])))
phone_refused("a TURN entry with no credential", mut(MB, f"iceServers.{TURN_I}.credential", ""))
phone_refused("a STUN entry with a credential", mut(MB, "iceServers.0.credential", "x"))
phone_refused("turnCredentialExpiresAt that is not the TURN expiry", mut(MB, "turnCredentialExpiresAt", MB["turnCredentialExpiresAt"] + 1))
phone_refused("a session holder that is not the phone's key", MB, holder=IDS["phoneBThumbprint"])
phone_refused("a TURN entry whose expiresAt is null (the phone reads it as none)", mut(MB, f"iceServers.{TURN_I}.expiresAt", None))

# ---- the bearer itself: a damaged one is refused by a reader that checks it
GOOD_BEARER = PB["accessToken"]
_p = GOOD_BEARER.split(".")
for label, tok in (("a flipped bit of the MAC", _p[0] + "." + _p[1] + "." + ("0" if _p[2][0] != "0" else "1") + _p[2][1:]),
                   ("a claims payload with one digit changed", _p[0] + "." + _p[1][:-1] + ("0" if _p[1][-1] != "0" else "1") + "." + _p[2]),
                   ("a MAC in capitals", _p[0] + "." + _p[1] + "." + _p[2].upper()),
                   ("another prefix", "aokie-adm-v3." + _p[1] + "." + _p[2]),
                   ("no MAC", _p[0] + "." + _p[1])):
    if label == "a MAC in capitals" and _p[2] == _p[2].upper():
        continue
    n_plugin_neg += 1
    refuses("the bearer reader refuses " + label, lambda tok=tok: D.read_bearer(tok, SECRET))
n_plugin_neg += 1
refuses("the bearer reader refuses another secret", lambda: D.read_bearer(GOOD_BEARER, bytes(32)))

# ================================================================================================ challenges
n_challenge_neg = 0
by_bearer = {}
for c in challenge["cases"]:
    body = c["response"]["body"]
    accepts(f"{c['name']}: EndpointChallengeFrame validates", lambda b=body: D.endpoint_challenge(b, NOW))
    claims = accepts(f"{c['name']}: its bearer verifies", lambda t=c["bearer"]: D.read_bearer(t, SECRET))
    by_bearer[c["name"]] = (body, claims)
    if claims:
        check(f"{c['name']}: built only from its bearer", (body["appId"], body["subjectId"], body["role"], body["admissionJti"], body["holderKeyThumbprint"])
              == (claims["appId"], claims["subjectId"], claims["role"], claims["jti"], claims["holderKeyThumbprint"]))
        if claims["role"] == "plugin":
            check(f"{c['name']}: carries the bearer's roster and no expectedPeerKeyThumbprint",
                  (body["approvedPeerKeyThumbprints"], body["peerRosterRevision"], body["peerRosterHash"]) == (claims["approvedPeerKeyThumbprints"], claims["peerRosterRevision"], claims["peerRosterHash"])
                  and "expectedPeerKeyThumbprint" not in body)
        else:
            check(f"{c['name']}: carries the bearer's expected peer and no roster member",
                  body["expectedPeerKeyThumbprint"] == claims["expectedPeerKeyThumbprint"] and not any(k in body for k in ("approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")))
    want = (["kind", "schemaVersion", "appId", "subjectId", "role", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash", "expiresAt"]
            if body["role"] == "plugin" else ["kind", "schemaVersion", "appId", "subjectId", "role", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint", "expectedPeerKeyThumbprint", "expiresAt"])
    check(f"{c['name']}: the members are in FormLogic's order", list(body) == want)
    check(f"{c['name']}: 25 seconds of life, connectionId and nonce in their forms",
          body["expiresAt"] == NOW + 25 and re.fullmatch(r"relay_[0-9a-f]{32}", body["connectionId"]) is not None and re.fullmatch(r"challenge_[0-9a-f]{32}", body["challengeNonce"]) is not None)
    check(f"{c['name']}: no-store", c["response"]["headers"].get("cache-control") == "no-store" and c["response"]["headers"].get("pragma") == "no-cache")
CP = challenge["cases"][0]["response"]["body"]
CM = challenge["cases"][1]["response"]["body"]


def challenge_refused(label, doc, now=NOW):
    global n_challenge_neg
    n_challenge_neg += 1
    refuses("EndpointChallengeFrame refuses " + label, lambda: D.endpoint_challenge(doc, now))


challenge_refused("a plugin challenge that also carries expectedPeerKeyThumbprint", dict(CP, expectedPeerKeyThumbprint=IDS["desktopEndpointThumbprint"]))
challenge_refused("a phone challenge that also carries approvedPeerKeyThumbprints", dict(CM, approvedPeerKeyThumbprints=[IDS["phoneBThumbprint"]]))
challenge_refused("a phone challenge with a roster revision", dict(CM, peerRosterRevision=7))
challenge_refused("a phone challenge with a roster hash", dict(CM, peerRosterHash="A" * 43))
challenge_refused("a phone challenge without expectedPeerKeyThumbprint", mut(CM, "expectedPeerKeyThumbprint", DEL))
challenge_refused("a phone challenge whose expected peer is its own key", mut(CM, "expectedPeerKeyThumbprint", CM["holderKeyThumbprint"]))
challenge_refused("a plugin challenge without a roster", mut(CP, "approvedPeerKeyThumbprints", DEL))
challenge_refused("a plugin challenge with an empty roster", mut(CP, "approvedPeerKeyThumbprints", []))
challenge_refused("a plugin roster in the wrong order", mut(CP, "approvedPeerKeyThumbprints", list(reversed(CP["approvedPeerKeyThumbprints"]))))
challenge_refused("a plugin roster with a duplicate", mut(CP, "approvedPeerKeyThumbprints", [CP["approvedPeerKeyThumbprints"][0]] * 2))
OWN = sorted(CP["approvedPeerKeyThumbprints"] + [CP["holderKeyThumbprint"]])
challenge_refused("a plugin roster that holds the plugin's own key (sorted, with the hash of that roster)", dict(CP, approvedPeerKeyThumbprints=OWN, peerRosterHash=D.peer_roster_hash(CP["peerRosterRevision"], OWN)))
UNSORTED = list(reversed(CP["approvedPeerKeyThumbprints"]))
challenge_refused("a plugin roster in the wrong order (with the hash of that order)", dict(CP, approvedPeerKeyThumbprints=UNSORTED, peerRosterHash=D.peer_roster_hash(CP["peerRosterRevision"], UNSORTED)))
challenge_refused("a plugin challenge with a wrong roster hash", mut(CP, "peerRosterHash", "A" * 43))
challenge_refused("a plugin challenge with revision 0", mut(CP, "peerRosterRevision", 0))
challenge_refused("a plugin challenge without a revision", mut(CP, "peerRosterRevision", DEL))
challenge_refused("a plugin challenge without a hash", mut(CP, "peerRosterHash", DEL))
challenge_refused("an unknown member", dict(CP, extra=1))
challenge_refused("kind hello", mut(CP, "kind", "hello"))
challenge_refused("schemaVersion 3", mut(CP, "schemaVersion", 3))
challenge_refused("role admin", mut(CP, "role", "admin"))
challenge_refused("a nonce with a space", mut(CP, "challengeNonce", "a b"))
challenge_refused("an admissionJti of 201 bytes", mut(CP, "admissionJti", "j" * 201))
challenge_refused("an expiresAt already past", mut(CP, "expiresAt", NOW))
challenge_refused("an expiresAt 31 seconds ahead (the limit is 30)", mut(CP, "expiresAt", NOW + 31))
challenge_refused("an expiresAt as a string", mut(CP, "expiresAt", str(NOW + 25)))
challenge_refused("a phone whose clock is 6 seconds behind the relay's (25 + 6 is over 30): the tolerance is 5 seconds", CP, now=NOW - 6)
check("a phone whose clock is 5 seconds behind the relay's still accepts (the 25 second life leaves 5 seconds to spare on that side)", D.endpoint_challenge(CP, NOW - 5) is CP)
challenge_refused("a phone whose clock is 25 seconds ahead of the relay's (the challenge has expired)", CP, now=NOW + 25)

# ================================================================================================ frames
by_step = {s["step"]: s for s in frames["steps"]}
S_POST = by_step["phone A posts three frames to the plugin"]
posted = S_POST["request"]["body"]["frames"]
check("the frames request is exactly what the carriers send (relay_post_body embeds the frames verbatim)",
      S_POST["request"]["bodyText"] == D.relay_post_body("plugin", [json.dumps(f, ensure_ascii=False, separators=(",", ":")) for f in posted]) and json.loads(S_POST["request"]["bodyText"]) == S_POST["request"]["body"])
check("a post is answered {accepted, seq, time} with every frame stored", S_POST["response"]["body"] == {"accepted": 3, "seq": 3, "time": NOW})
page = by_step["the plugin reads them"]["response"]["body"]
check("a page is {frames, lastSeq, time} and the tail priming reads lastSeq", list(page) == ["frames", "lastSeq", "time"] and D.tail_cursor_page(page, 0) == 3)
check("frames are in order from seq 1", [f["seq"] for f in page["frames"]] == [1, 2, 3])
for f in page["frames"]:
    check(f"frame {f['seq']}: the five members of a delivered frame", list(f) == ["seq", "from", "subjectId", "grants", "frame"])
    check(f"frame {f['seq']}: from, subject and grants are the relay's record of phone A", f["from"] == "mobile:" + IDS["phoneAThumbprint"] and f["subjectId"] == IDS["phoneA"] and f["grants"] == ["state_read", "caller_read", "captions_read", "rtc_signal"])
check("a frame comes back as it went in (ints past 2^53, floats, nested empties, Unicode)", [f["frame"] for f in page["frames"]] == posted)
check("the integer 9007199254740993 stayed an integer and 1.0 stayed a float", type(page["frames"][1]["frame"]["n"]) is int and type(page["frames"][1]["frame"]["f"]) is float)
check("a member of the frame cannot set the sender (the third frame's seq member is its own)", page["frames"][2]["seq"] == 3 and page["frames"][2]["frame"]["seq"] == 999)
tail = by_step["the plugin finds the tail"]["response"]["body"]
check("at the tail the page is empty and lastSeq is the cursor asked for", tail["frames"] == [] and tail["lastSeq"] == 3 and D.tail_cursor_page(tail, 3) == 3)
to_a = by_step["the plugin posts two frames to phone A"]
check("the plugin addresses a phone by mobile:<thumbprint>", to_a["request"]["body"]["to"] == "mobile:" + IDS["phoneAThumbprint"] and to_a["response"]["body"]["accepted"] == 2)
a_read = by_step["phone A reads them"]["response"]["body"]
check("the phone's frames come from the plugin, under the plugin's own subject and its two scopes", all(f["from"] == "plugin" and f["subjectId"] == "aokie" and f["grants"] == ["state_read", "rtc_signal"] for f in a_read["frames"]))
again = by_step["phone A reads them again, from where it left off"]["response"]["body"]
check("a read acknowledges nothing: the second read from an earlier cursor still has the frame", [f["seq"] for f in again["frames"]] == [2] and again["frames"][0] == a_read["frames"][1])
check("a mailbox is the party's own: phone B has nothing", by_step["phone B has nothing"]["response"]["body"]["frames"] == [])
check("the three bearers of the conversation are the admissions' (each verifies)", all(D.read_bearer(t, SECRET) for t in frames["bearers"].values()))
for f in page["frames"]:
    ev = D.parse_sse_block("id: %d\nevent: frame\ndata: %s" % (f["seq"], json.dumps(f, separators=(",", ":"), ensure_ascii=False)))
    check(f"frame {f['seq']} as a stream event is read by the plugin's parser to the same values", ev is not None and ev[0] == "frame" and ev[1]["seq"] == f["seq"] and ev[1]["from"] == f["from"] and ev[1]["subjectId"] == f["subjectId"] and ev[1]["grants"] == f["grants"] and ev[1]["frame"] == f["frame"])

# ================================================================================================ streams
n_sse_neg = 0
for c in streams["cases"]:
    body = c["body"]
    events = D.SseParser().push(body)
    check(f"{c['name']}: the preamble (retry and a comment) is no event", body.startswith("retry: 2000\n\n: connected\n\n"))
    check(f"{c['name']}: it always ends with end", bool(events) and events[-1][0] == "end")
    frames_ = [e[1] for e in events if e[0] == "frame"]
    check(f"{c['name']}: frame ids ascend and the end event's id is the last delivered (or the cursor asked for)",
          [f["seq"] for f in frames_] == sorted(f["seq"] for f in frames_) and events[-1][1]["seq"] == (frames_[-1]["seq"] if frames_ else int(c["request"].get("query", {}).get("since", 0))))
    check(f"{c['name']}: every id: line is the frame's seq", [int(m) for m in re.findall(r"^id: (\d+)\nevent: frame", body, re.M)] == [f["seq"] for f in frames_])
    check(f"{c['name']}: no other event name", set(re.findall(r"^event: (.+)$", body, re.M)) <= {"frame", "end"})
    for i in range(len(body) + 1):
        p = D.SseParser()
        got = p.push(body[:i]) + p.push(body[i:])
        if got != events:
            check(f"{c['name']}: split at byte {i} reads the same events", False)
            break
    else:
        check(f"{c['name']}: split at every offset the parser reads the same events", True)
    check(f"{c['name']}: with CRLF line ends the parser reads the same events", D.SseParser().push(body.replace("\n", "\r\n")) == events)
    check(f"{c['name']}: the phone's parser (no metadata) reads the same seqs and frames", [(e[0], e[1].get("seq"), e[1].get("frame")) for e in D.SseParser(plugin=False).push(body)] == [(e[0], e[1].get("seq"), e[1].get("frame")) for e in events])
    if "keepalive" in body:
        check(f"{c['name']}: a keepalive is a comment line and no event", body.count(": keepalive\n\n") == 1 and len(events) == 1)
first = streams["cases"][0]["body"]
resume = streams["cases"][1]["body"]
check("resuming with since=2 delivers only frame 3", [e[1]["seq"] for e in D.SseParser().push(resume) if e[0] == "frame"] == [3])
check("the stream's frame events carry what the page carries", [e[1]["frame"] for e in D.SseParser().push(first) if e[0] == "frame"] == [f["frame"] for f in page["frames"]])


def sse_refused(label, chunk, expect_none=True):
    global n_sse_neg
    n_sse_neg += 1
    ev = D.SseParser().push(chunk)
    check(f"the plugin's SSE parser yields no event for {label}", ev == [] if expect_none else True, str(ev))


F = json.dumps(page["frames"][0], separators=(",", ":"))
sse_refused("a block with no event name", "id: 1\ndata: " + F + "\n\n")
sse_refused("an unknown event name", "id: 1\nevent: ping\ndata: {}\n\n")
sse_refused("a frame whose data is not JSON", "id: 1\nevent: frame\ndata: nope\n\n")
sse_refused("a frame without seq", "id: 1\nevent: frame\ndata: " + json.dumps({k: v for k, v in page["frames"][0].items() if k != "seq"}) + "\n\n")
sse_refused("a frame with seq as a string", "id: 1\nevent: frame\ndata: " + json.dumps(dict(page["frames"][0], seq="1")) + "\n\n")
sse_refused("a frame with a negative seq", "id: 1\nevent: frame\ndata: " + json.dumps(dict(page["frames"][0], seq=-1)) + "\n\n")
sse_refused("a frame without from", "id: 1\nevent: frame\ndata: " + json.dumps({k: v for k, v in page["frames"][0].items() if k != "from"}) + "\n\n")
sse_refused("a frame without frame", "id: 1\nevent: frame\ndata: " + json.dumps({k: v for k, v in page["frames"][0].items() if k != "frame"}) + "\n\n")
sse_refused("a comment only", ": keepalive\n\n")
sse_refused("the preamble", "retry: 2000\n\n: connected\n\n")
sse_refused("an event that never ends (no blank line)", "id: 1\nevent: frame\ndata: " + F + "\n")
end = D.SseParser().push("id: abc\nevent: end\ndata: {}\n\n")
check("an end event whose id is not a number carries no cursor", end == [("end", {"seq": None})])
grants = [D.parse_sse_block("id: 1\nevent: frame\ndata: " + json.dumps(dict(page["frames"][0], grants=g)))[1]["grants"] for g in
          (["state_read", "delete_all"], ["state_read", "state_read"], "state_read", ["state_read"] * 17, ["state_read", "caller_read"])]
check("malformed or unknown grant metadata fails closed to no authority; a good list is kept", grants == [[], [], [], [], ["state_read", "caller_read"]])
check("a missing subjectId carries no device identity", D.parse_sse_block("id: 1\nevent: frame\ndata: " + json.dumps({k: v for k, v in page["frames"][0].items() if k != "subjectId"}))[1]["subjectId"] is None)
check("a comment inside an event block is skipped", D.parse_sse_block("id: 1\n: hi\nevent: end")[0] == "end")
big = D.SseParser()
big.push("data: " + "x" * (2 * 1024 * 1024))
check("an event that never terminates is discarded past the buffer ceiling", big.buffer == "")

# ================================================================================================ errors
n_error = 0
for c in errors["cases"]:
    res = c["response"]
    if res["status"] >= 400:
        n_error += 1
        accepts(f"{c['name']}: the phone's error decoder reads exactly error, code and message", lambda r=res: D.compat_error(r["body"], r["status"]))
        check(f"{c['name']}: a wait is in the header alone", "retryAfter" not in res["body"])
    else:
        check(f"{c['name']}: a refused wait is a 200 page with hold.refused", res["body"]["hold"] == {"refused": True, "retryAfter": 2} and res["headers"].get("x-oaiy-hold") == "refused")
codes = {c["response"]["body"].get("code"): c["response"]["status"] for c in errors["cases"] if c["response"]["status"] >= 400}
for code, status in {"invalid_token": 401, "relay_target_forbidden": 403, "relay_frame_too_large": 413, "relay_backpressure": 429, "companion_unavailable": 503, "revoked": 401,
                     "forbidden": 403, "feature_disabled": 403, "unprocessable": 422, "invalid_request": 400, "method_not_allowed": 405, "not_found": 404}.items():
    check(f"the code {code} is recorded with status {status}", codes.get(code) == status)
by_name = {c["name"]: c for c in errors["cases"]}
check("a 401 carries WWW-Authenticate", by_name["a bearer this relay never issued"]["response"]["headers"]["www-authenticate"].startswith("Bearer"))
check("429 and 503 carry Retry-After (5 and 2)", by_name["a batch that would overflow the mailbox"]["response"]["headers"]["retry-after"] == "5" and by_name["a stream when the host has no worker to spare"]["response"]["headers"]["retry-after"] == "2")
bad_errors = [
    ("an error with a retryAfter member (the native shape)", {"error": True, "code": "rate_limited", "message": "x", "retryAfter": 5}),
    ("the native nesting", {"error": {"code": "rate_limited", "message": "x"}}),
    ("error false", {"error": False, "code": "rate_limited", "message": "x"}),
    ("an empty message", {"error": True, "code": "rate_limited", "message": ""}),
    ("a message of 241 bytes", {"error": True, "code": "rate_limited", "message": "m" * 241}),
    ("a code with a space", {"error": True, "code": "rate limited", "message": "x"}),
    ("no code", {"error": True, "message": "x"}),
]
for label, doc in bad_errors:
    refuses("the phone's error decoder refuses " + label, lambda d=doc: D.compat_error(d, 429))

# ================================================================================================ ICE
for c in ice["cases"]:
    got = D.mint_ice(c["config"], c["endpoint"], c["now"])
    check(f"ice: {c['name']}: an independent minter computes the recorded servers", got == c["expected"], json.dumps(got)[:200])
    for style in ("plugin", "phone"):
        accepts(f"ice: {c['name']}: the {style}'s validation accepts them", lambda c=c, style=style: D.ice_configuration(c["expected"]["iceServers"], c["expected"]["relayOnly"], c["expected"]["turnCredentialExpiresAt"], c["now"], style))
fl = ice["cases"][0]["expected"]["iceServers"][1]
check("FormLogic's known answers: the id b76588ed... and the credential KHqzx1+wW92HnynGzGgLw5mZgwo=", fl["username"] == "1784160600:b76588edf0d149e1075b69631e94cc91" and fl["credential"] == "KHqzx1+wW92HnynGzGgLw5mZgwo=")
fl2 = ice["cases"][1]["expected"]["iceServers"][1]
check("FormLogic's plugin case: the id 9ad5b440... and the credential Stdmtt53cR4OkoTsqwXpKqEeWas=", fl2["username"] == "1784160600:9ad5b440809a76e58520fa0d14e32626" and fl2["credential"] == "Stdmtt53cR4OkoTsqwXpKqEeWas=")
check("the same secret gives another id for another role", fl["username"] != fl2["username"])

# ================================================================================================ the totals
total_neg = n_plugin_neg + n_phone_neg + n_challenge_neg + n_sse_neg + len(bad_errors)
check("at least a hundred damaged documents were refused", total_neg >= 100, str(total_neg))
print(f"  {len(admission['cases'])} admissions, {len(challenge['cases'])} challenges, {len(frames['steps'])} frame exchanges, {len(streams['cases'])} streams, "
      f"{len(errors['cases'])} errors, {len(ice['cases'])} ICE cases read by the decoders' rules")
print(f"  {total_neg} damaged documents refused ({n_plugin_neg} plugin admissions, {n_phone_neg} phone admissions, {n_challenge_neg} challenges, {n_sse_neg} stream blocks, {len(bad_errors)} errors)")
for m in mismatches:
    print("  MISMATCH", m)
print(f"{checks} checks, {len(mismatches)} mismatches")
sys.exit(1 if mismatches else 0)
