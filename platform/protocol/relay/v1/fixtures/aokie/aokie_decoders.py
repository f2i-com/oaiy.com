"""Mirrors of the rules the shipped Aokie plugin and phone apply to what the relay sends them.

Nothing here imports the relay or its test helpers, and nothing uses JSON Schema: each function is a transcription of the Rust
decoder it names (file and function in the comments), written from the Aokie sources at the time this package was made, so a
fixture that passes here is one those decoders accept, and one that must not pass here is one they refuse. Every rule that a
comment marks `[serde]` is what `#[derive(Deserialize)]` with `deny_unknown_fields` does: unknown members, missing members and
wrong types are refused.

A `Refused` exception is a decoder returning `Err`. A function that degrades instead of failing (the relay advertisement)
returns `None`, as the Rust does.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import unicodedata
from urllib.parse import urlsplit

MAX_SAFE_INTEGER = 2 ** 53 - 1
MAX_ID_BYTES = 200                       # aokie-protocol v2.rs MAX_ID_BYTES
MAX_LEASE_TOKEN_BYTES = 16 * 1024        # aokie-protocol v2.rs
ADMISSION_SAFETY_MARGIN_SECONDS = 10     # aokie-plugin companion_gateway/constants.rs
MIN_TURN_CREDENTIAL_TTL_SECONDS = 30
MAX_TURN_CREDENTIAL_TTL_SECONDS = 24 * 60 * 60
MAX_ICE_SERVERS = 8                      # aokie-media signal.rs
MAX_ICE_URLS_PER_SERVER = 8
ENDPOINT_PROOF_MAX_LIFETIME = 30         # aokie-protocol v2.rs
MAX_AUTHENTICATED_GRANTS = 16            # aokie-plugin companion_relay.rs
MAX_STREAM_BUFFER_BYTES = 1024 * 1024
# apps/aokie-mobile/src-tauri/src/managed_auth.rs MANAGED_ADMISSION_GRANTS
GRANTS = ["state_read", "caller_read", "captions_read", "assistance_read", "assistance_respond", "monitor", "consult", "takeover",
          "resume_aokie", "end_caller", "rtc_signal", "participants_read", "participant_identity_read", "audio_levels_read"]


class Refused(Exception):
    """A decoder returned Err."""


def refuse(why: str):
    raise Refused(why)


# ------------------------------------------------------------------------------------------------ primitives

def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def unb64u(s: str) -> bytes:
    """URL_SAFE_NO_PAD.decode: no padding, no other alphabet, no non-zero trailing bits."""
    if not isinstance(s, str) or re.fullmatch(r"[A-Za-z0-9_-]*", s) is None or len(s) % 4 == 1:
        refuse("base64url")
    raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if b64u(raw) != s:
        refuse("base64url trailing bits")
    return raw


def is_u64(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** 64


def need_u64(v, what):
    if not is_u64(v):
        refuse(f"{what}: not a u64")
    return v


def need_str(v, what):
    if not isinstance(v, str):
        refuse(f"{what}: not a string")
    return v


def need_bool(v, what):
    if not isinstance(v, bool):
        refuse(f"{what}: not a bool")
    return v


def need_str_list(v, what):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        refuse(f"{what}: not a list of strings")
    return v


def strict(doc, allowed: list[str], required: list[str], what: str):
    """[serde] deny_unknown_fields with the given required members."""
    if not isinstance(doc, dict):
        refuse(f"{what}: not an object")
    extra = [k for k in doc if k not in allowed]
    if extra:
        refuse(f"{what}: unknown member {extra[0]}")
    for k in required:
        if k not in doc:
            refuse(f"{what}: missing member {k}")
    return doc


def is_control(ch: str) -> bool:
    """char::is_control: the Unicode Cc category."""
    return unicodedata.category(ch) == "Cc"


def validate_identity(value: str, what: str):
    """companion_gateway/helpers.rs validate_identity, and validate_id in the phone (managed_auth.rs)."""
    raw = value.encode("utf-8")
    if not value or len(raw) > 200 or not all(chr(c).isalnum() and c < 128 or chr(c) in "-_.:" for c in raw):
        refuse(f"{what} is invalid")


def safe_id(value: str, what: str):
    """aokie-protocol v2.rs safe_id: 1 to MAX_ID_BYTES bytes of [A-Za-z0-9_.:-]."""
    raw = value.encode("utf-8")
    if not value or len(raw) > MAX_ID_BYTES or not all(chr(c).isalnum() and c < 128 or chr(c) in "-_.:" for c in raw):
        refuse(f"{what} is not a safe id")


def safe_integer(v, minimum: int, what: str):
    if not (is_u64(v) and minimum <= v <= MAX_SAFE_INTEGER):
        refuse(f"{what}: not a safe integer")


def thumbprint_of_public_key(public_key_b64u: str) -> str:
    """v2.rs endpoint_thumbprint: SHA-256 of {"crv":"Ed25519","kty":"OKP","x":<key>}."""
    canonical = '{"crv":"Ed25519","kty":"OKP","x":' + json.dumps(public_key_b64u) + '}'
    return b64u(hashlib.sha256(canonical.encode("utf-8")).digest())


def peer_roster_hash(revision: int, thumbprints: list[str]) -> str:
    """v2.rs peer_roster_hash: SHA-256("aokie/v2/peer-roster" NUL canonical {approvedPeerKeyThumbprints (sorted), peerRosterRevision})."""
    payload = {"approvedPeerKeyThumbprints": sorted(thumbprints), "peerRosterRevision": revision}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return b64u(hashlib.sha256(b"aokie/v2/peer-roster\x00" + canonical.encode("utf-8")).digest())


def endpoint_public_key(v, what="endpointPublicKey"):
    """v2.rs EndpointPublicKey [serde]: algorithm (only "ed25519"), publicKey, thumbprint; validate() recomputes the thumbprint."""
    strict(v, ["algorithm", "publicKey", "thumbprint"], ["algorithm", "publicKey", "thumbprint"], what)
    if v["algorithm"] != "ed25519":
        refuse(f"{what}.algorithm")
    key = unb64u(need_str(v["publicKey"], what))
    if len(key) != 32:
        refuse(f"{what}.publicKey length")
    safe_id(need_str(v["thumbprint"], what), what + ".thumbprint")
    if v["thumbprint"] != thumbprint_of_public_key(v["publicKey"]):
        refuse(f"{what}.thumbprint does not recompute")
    return v


# ------------------------------------------------------------------------------------------------ ICE

def _has_turn_url(urls: list[str]) -> bool:
    return any(u.lower().startswith(("turn:", "turns:")) for u in urls)


def ice_server(v, style: str):
    """AdmissionIceServer (plugin, admission.rs) and DiscoveryIceServer (phone, discovery.rs), both [serde] deny_unknown_fields.

    The plugin requires urls, username and credential and takes expiresAt as a number when it is present (null is refused); the
    phone defaults username, credential and expiresAt (null is None)."""
    strict(v, ["urls", "username", "credential", "expiresAt"], ["urls", "username", "credential"] if style == "plugin" else ["urls"], "iceServer")
    urls = need_str_list(v["urls"], "iceServer.urls")
    username = need_str(v.get("username", ""), "iceServer.username")
    credential = need_str(v.get("credential", ""), "iceServer.credential")
    expires = None
    if "expiresAt" in v:
        if v["expiresAt"] is None and style == "phone":
            expires = None
        else:
            expires = need_u64(v["expiresAt"], "iceServer.expiresAt")
    return {"urls": urls, "username": username, "credential": credential, "expiresAt": expires}


def ice_validate_all(servers: list[dict]):
    """aokie-media signal.rs IceServerConfig::validate_all."""
    if len(servers) > MAX_ICE_SERVERS:
        refuse("too many ICE servers")
    for s in servers:
        if (not s["urls"] or len(s["urls"]) > MAX_ICE_URLS_PER_SERVER or len(s["username"].encode()) > 512 or len(s["credential"].encode()) > 2048
                or any(is_control(c) for c in s["username"]) or any(is_control(c) for c in s["credential"])):
            refuse("ICE server")
        for url in s["urls"]:
            if len(url.encode()) > 2048 or any(is_control(c) for c in url) or not url.lower().startswith(("stun:", "stuns:", "turn:", "turns:")):
                refuse("ICE server URL")


def ice_configuration(raw_servers, relay_only, turn_expiry, now: int, style: str) -> list[dict]:
    """validate_admission_ice_configuration (plugin, admission.rs) and validate_managed_ice_configuration (phone, discovery.rs)."""
    if not isinstance(raw_servers, list):
        refuse("iceServers: not a list")
    servers = [ice_server(s, style) for s in raw_servers]
    ice_validate_all(servers)
    earliest = None
    for s in servers:
        if _has_turn_url(s["urls"]):
            if s["username"] == "" or s["credential"] == "":
                refuse("TURN servers require short-lived credentials")
            if s["expiresAt"] is None:
                refuse("TURN servers require expiresAt")
            if s["expiresAt"] <= now + MIN_TURN_CREDENTIAL_TTL_SECONDS or s["expiresAt"] > now + MAX_TURN_CREDENTIAL_TTL_SECONDS:
                refuse("TURN expiresAt must be 31 seconds to 24 hours in the future")
            earliest = s["expiresAt"] if earliest is None else min(earliest, s["expiresAt"])
        elif s["username"] != "" or s["credential"] != "" or s["expiresAt"] is not None:
            refuse("STUN-only entries cannot contain credentials or expiresAt")
    if relay_only and earliest is None:
        refuse("relayOnly requires at least one TURN server")
    if earliest != turn_expiry:
        refuse("turnCredentialExpiresAt does not match the earliest TURN expiry")
    return servers


def nullable_unix(v, what):
    """NullableUnixTimestamp: null or a u64 (a string, a float or a negative number is refused)."""
    if v is None:
        return None
    return need_u64(v, what)


# ------------------------------------------------------------------------------------------------ relay endpoints

def _origin(url) -> tuple:
    port = url.port
    if port is None:
        port = {"https": 443, "http": 80}.get(url.scheme)
    return (url.scheme, (url.hostname or "").lower(), port)


def normalize_relay_url(raw: str, label: str, managed_beta: bool = False):
    """companion_gateway/helpers.rs normalize_relay_url (https only; a managed-beta build also takes http on a loopback or .local host)."""
    try:
        url = urlsplit(raw)
    except ValueError:
        refuse(f"{label} is not an absolute URL")
    if not url.scheme or not url.netloc:
        refuse(f"{label} is not an absolute URL")
    if url.username is not None or url.password is not None:
        refuse(f"{label} must not contain credentials")
    if "#" in raw:
        refuse(f"{label} must not contain a fragment")
    host = (url.hostname or "").lower()
    local = host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost") or host.endswith(".local")
    if url.scheme != "https" and not (managed_beta and url.scheme == "http" and local):
        refuse(f"{label} must use https")
    return url


def usable_relay_endpoints(advertisement, managed_beta: bool = False):
    """usable_relay_endpoints: the three URLs when the advertisement decodes, every URL is safe and they share one origin, else None
    (the carrier degrades to the WebSocket gateway instead of failing the admission). RelayEndpoints is not deny_unknown_fields."""
    if not isinstance(advertisement, dict):
        return None
    try:
        urls = {k: need_str(advertisement.get(k), k) for k in ("challengeUrl", "framesUrl", "streamUrl")}
        parsed = [normalize_relay_url(urls["challengeUrl"], "challengeUrl", managed_beta), normalize_relay_url(urls["framesUrl"], "framesUrl", managed_beta),
                  normalize_relay_url(urls["streamUrl"], "streamUrl", managed_beta)]
    except Refused:
        return None
    if len({_origin(u) for u in parsed}) != 1:
        return None
    return urls


def normalize_gateway_url(raw: str):
    """helpers.rs normalize_gateway_url in a release build: wss only, no credentials, no fragment."""
    try:
        url = urlsplit(raw)
    except ValueError:
        refuse("gatewayUrl is invalid")
    if not url.scheme or not url.netloc:
        refuse("gatewayUrl is invalid")
    if url.username is not None or url.password is not None or "#" in raw:
        refuse("gatewayUrl must not contain credentials or a fragment")
    if url.scheme != "wss":
        refuse("gatewayUrl must use wss")
    return url


# ------------------------------------------------------------------------------------------------ the plugin's admission

PLUGIN_MEMBERS = ["accessToken", "tokenType", "expiresIn", "expiresAt", "gatewayUrl", "appId", "subjectId", "role", "scopes", "device", "iceServers", "relayOnly",
                  "turnCredentialExpiresAt", "endpointPublicKey", "holderKeyThumbprint", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash", "relay"]


def plugin_admission(doc, expect: dict, now: int) -> dict:
    """AdmissionResponse::into_credentials (aokie-plugin companion_gateway/admission.rs).

    expect: appId, pluginId, endpointPublicKey (the object the plugin sent), approvedPeerKeyThumbprints, peerRosterRevision, peerRosterHash.
    Returns {"lifetime": seconds, "relay": endpoints or None, "transport": "relay" | "websocket"}."""
    strict(doc, PLUGIN_MEMBERS, [m for m in PLUGIN_MEMBERS if m != "relay"], "plugin admission")
    token = need_str(doc["accessToken"], "accessToken")
    token_type = need_str(doc["tokenType"], "tokenType")
    expires_in = need_u64(doc["expiresIn"], "expiresIn")
    expires_at = need_u64(doc["expiresAt"], "expiresAt")
    gateway = need_str(doc["gatewayUrl"], "gatewayUrl")
    app_id = need_str(doc["appId"], "appId")
    subject_id = need_str(doc["subjectId"], "subjectId")
    role = need_str(doc["role"], "role")
    need_str_list(doc["scopes"], "scopes")
    # device: a raw serde_json::Value: any JSON value is accepted, and nothing is read from it.
    relay_only = need_bool(doc["relayOnly"], "relayOnly")
    turn_expiry = nullable_unix(doc["turnCredentialExpiresAt"], "turnCredentialExpiresAt")
    key = endpoint_public_key(doc["endpointPublicKey"])
    holder = need_str(doc["holderKeyThumbprint"], "holderKeyThumbprint")
    approved = need_str_list(doc["approvedPeerKeyThumbprints"], "approvedPeerKeyThumbprints")
    revision = need_u64(doc["peerRosterRevision"], "peerRosterRevision")
    roster_hash = need_str(doc["peerRosterHash"], "peerRosterHash")
    if not isinstance(doc["iceServers"], list):
        refuse("iceServers: not a list")
    [ice_server(s, "plugin") for s in doc["iceServers"]]      # typed decode first, as serde does
    if (token_type != "Bearer" or role != "plugin" or app_id != expect["appId"] or subject_id != expect["pluginId"]
            or key != expect["endpointPublicKey"] or holder != expect["endpointPublicKey"]["thumbprint"]
            or approved != expect["approvedPeerKeyThumbprints"] or revision != expect["peerRosterRevision"] or roster_hash != expect["peerRosterHash"]):
        refuse("a Companion admission for a different identity")
    validate_identity(app_id, "appId")
    validate_identity(subject_id, "subjectId")
    if not token or len(token.encode()) > MAX_LEASE_TOKEN_BYTES or any(is_control(c) for c in token):
        refuse("privateBootstrap accessToken is invalid")
    wall_remaining = max(0, expires_at - now)
    if (expires_in <= ADMISSION_SAFETY_MARGIN_SECONDS or expires_in > 300 or wall_remaining <= ADMISSION_SAFETY_MARGIN_SECONDS or wall_remaining > 300):
        refuse("an expired or unsafe admission lifetime")
    servers = ice_configuration(doc["iceServers"], relay_only, turn_expiry, now, "plugin")
    safe_remaining = min(expires_in, wall_remaining)
    if turn_expiry is not None:
        safe_remaining = min(safe_remaining, max(0, turn_expiry - now))
    if safe_remaining <= ADMISSION_SAFETY_MARGIN_SECONDS:
        refuse("ICE credentials with no safe connection lifetime")
    normalize_gateway_url(gateway)
    relay = usable_relay_endpoints(doc["relay"]) if doc.get("relay") is not None else None
    return {"lifetime": safe_remaining - ADMISSION_SAFETY_MARGIN_SECONDS, "relay": relay, "transport": "relay" if relay else "websocket", "iceServers": servers}


# ------------------------------------------------------------------------------------------------ the phone's admission

PHONE_MEMBERS = ["accessToken", "tokenType", "expiresIn", "expiresAt", "gatewayUrl", "appId", "subjectId", "role", "holderKeyThumbprint", "expectedPeerKeyThumbprint",
                 "scopes", "iceServers", "relayOnly", "turnCredentialExpiresAt", "device", "relay"]
DEVICE_MEMBERS = ["id", "appId", "subjectId", "role", "displayName", "grants", "approvedAt", "lastSeenAt"]


def mobile_admission(doc, session: dict, holder_key_thumbprint: str, now: int) -> dict:
    """validate_admission (apps/aokie-mobile managed_auth.rs).

    session: gatewayUrl, appId, deviceId, discoveryRelayOnly: the values the phone holds from its signed discovery. A phone paired
    to a personal relay has no signed discovery: its owner must supply these from the pairing offer (see the README's list of what
    a Rust contract test needs)."""
    strict(doc, PHONE_MEMBERS, [m for m in PHONE_MEMBERS if m not in ("iceServers", "relay")], "phone admission")
    token = need_str(doc["accessToken"], "accessToken")
    token_type = need_str(doc["tokenType"], "tokenType")
    expires_in = need_u64(doc["expiresIn"], "expiresIn")
    expires_at = need_u64(doc["expiresAt"], "expiresAt")
    gateway = need_str(doc["gatewayUrl"], "gatewayUrl")
    app_id = need_str(doc["appId"], "appId")
    subject_id = need_str(doc["subjectId"], "subjectId")
    role = need_str(doc["role"], "role")
    holder = need_str(doc["holderKeyThumbprint"], "holderKeyThumbprint")
    expected_peer = need_str(doc["expectedPeerKeyThumbprint"], "expectedPeerKeyThumbprint")
    scopes = need_str_list(doc["scopes"], "scopes")
    relay_only = need_bool(doc["relayOnly"], "relayOnly")
    turn_expiry = nullable_unix(doc["turnCredentialExpiresAt"], "turnCredentialExpiresAt")
    dev = strict(doc["device"], DEVICE_MEMBERS, DEVICE_MEMBERS, "device")
    for k in ("id", "appId", "subjectId", "role", "displayName", "approvedAt", "lastSeenAt"):
        need_str(dev[k], "device." + k)
    grants = need_str_list(dev["grants"], "device.grants")
    raw_ice = doc.get("iceServers", [])
    if not isinstance(raw_ice, list):
        refuse("iceServers: not a list")
    [ice_server(s, "phone") for s in raw_ice]
    if (token_type != "Bearer" or len(token.encode()) < 16 or len(token.encode()) > MAX_LEASE_TOKEN_BYTES or expires_in == 0 or expires_in > 300
            or expires_at <= now or expires_at > now + 300 or gateway != session["gatewayUrl"] or app_id != session["appId"] or subject_id != session["deviceId"]
            or role != "mobile" or holder != holder_key_thumbprint or expected_peer == holder_key_thumbprint or relay_only != session["discoveryRelayOnly"]):
        refuse("managed admission is not bound to the signed deployment and device")
    validate_identity(holder, "admission holder key thumbprint")
    validate_identity(expected_peer, "admission expected peer key thumbprint")
    validate_identity(dev["id"], "admission device id")
    if (dev["appId"] != session["appId"] or dev["subjectId"] != session["deviceId"] or dev["role"] != "mobile" or grants != scopes
            or not dev["displayName"] or len(dev["displayName"].encode()) > 120 or not dev["approvedAt"] or not dev["lastSeenAt"]):
        refuse("managed admission device record is inconsistent")
    if not scopes or "state_read" not in scopes or any(g not in GRANTS for g in scopes):
        refuse("managed admission contains invalid grants")
    servers = ice_configuration(raw_ice, relay_only, turn_expiry, now, "phone")
    relay = usable_relay_endpoints(doc["relay"]) if doc.get("relay") is not None else None
    return {"relay": relay, "transport": "relay" if relay else "websocket", "grants": scopes, "iceServers": servers}


# ------------------------------------------------------------------------------------------------ the challenge

CHALLENGE_MEMBERS = ["kind", "schemaVersion", "appId", "subjectId", "role", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint",
                     "expectedPeerKeyThumbprint", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash", "expiresAt"]


def endpoint_challenge(doc, now: int) -> dict:
    """EndpointChallengeFrame [serde] and its validate() (aokie-protocol v2.rs), with validate_peer_policy.

    The struct defaults the roster members and expectedPeerKeyThumbprint, so an absent member decodes; validate_peer_policy then
    refuses ANY roster member on a mobile challenge and expectedPeerKeyThumbprint on a plugin's."""
    strict(doc, CHALLENGE_MEMBERS, ["kind", "schemaVersion", "appId", "subjectId", "role", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint", "expiresAt"], "challenge")
    kind = need_str(doc["kind"], "kind")
    version = doc["schemaVersion"]
    if not (isinstance(version, int) and not isinstance(version, bool) and 0 <= version < 65536):
        refuse("schemaVersion")
    for k in ("appId", "subjectId", "role", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint"):
        need_str(doc[k], k)
    role = doc["role"]
    if role not in ("mobile", "plugin"):
        refuse("role")                       # AdmissionRole is a two-value enum
    expected_peer = doc.get("expectedPeerKeyThumbprint")
    if expected_peer is not None:
        need_str(expected_peer, "expectedPeerKeyThumbprint")   # Option<String>: null is None
    approved = need_str_list(doc.get("approvedPeerKeyThumbprints", []), "approvedPeerKeyThumbprints")
    revision = doc.get("peerRosterRevision")
    roster_hash = doc.get("peerRosterHash")
    if revision is not None:
        need_u64(revision, "peerRosterRevision")
    if roster_hash is not None:
        need_str(roster_hash, "peerRosterHash")
    expires_at = need_u64(doc["expiresAt"], "expiresAt")
    if kind != "endpoint_challenge":
        refuse("kind")
    if version != 2:
        refuse("schemaVersion")
    for k in ("appId", "subjectId", "connectionId", "challengeNonce", "admissionJti", "holderKeyThumbprint"):
        safe_id(doc[k], k)
    holder = doc["holderKeyThumbprint"]
    if role == "mobile":
        if expected_peer is None:
            refuse("expectedPeerKeyThumbprint")
        safe_id(expected_peer, "expectedPeerKeyThumbprint")
        if expected_peer == holder:
            refuse("expectedPeerKeyThumbprint")
        if approved or revision is not None or roster_hash is not None:
            refuse("approvedPeerKeyThumbprints")
    else:
        if expected_peer is not None or not approved or len(approved) > 64:
            refuse("approvedPeerKeyThumbprints")
        for t in approved:
            safe_id(t, "approvedPeerKeyThumbprints")
        if holder in approved:
            refuse("approvedPeerKeyThumbprints")
        if any(a >= b for a, b in zip(approved, approved[1:])):
            refuse("approvedPeerKeyThumbprints order")
        if revision is None:
            refuse("peerRosterRevision")
        safe_integer(revision, 1, "peerRosterRevision")
        if roster_hash is None:
            refuse("peerRosterHash")
        safe_id(roster_hash, "peerRosterHash")
        if roster_hash != peer_roster_hash(revision, approved):
            refuse("peerRosterHash")
    safe_integer(expires_at, 1, "expiresAt")
    if expires_at <= now or expires_at - now > ENDPOINT_PROOF_MAX_LIFETIME:
        refuse("expired")
    return doc


# ------------------------------------------------------------------------------------------------ the admission bearer

def read_bearer(token: str, secret: bytes) -> dict:
    """What a reader of the relay's bearer checks: prefix, hex, HMAC-SHA256 over the decoded claims bytes, claims JSON object.
    (The Aokie plugin and phone never parse it; the relay does, and this is its format: README 10.6.)"""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "aokie-adm-v2" or re.fullmatch(r"[0-9a-f]+", parts[1]) is None or re.fullmatch(r"[0-9a-f]{64}", parts[2]) is None or len(parts[1]) % 2:
        refuse("bearer format")
    payload = bytes.fromhex(parts[1])
    if not hmac.compare_digest(hmac.new(secret, payload, hashlib.sha256).hexdigest(), parts[2]):
        refuse("bearer MAC")
    claims = json.loads(payload.decode("utf-8"))
    if not isinstance(claims, dict):
        refuse("claims")
    return claims


# ------------------------------------------------------------------------------------------------ the stream

def parse_authenticated_grants(payload: dict) -> list[str]:
    """companion_relay.rs parse_authenticated_grants: fail closed to an empty authority set."""
    values = payload.get("grants")
    if not isinstance(values, list) or len(values) > MAX_AUTHENTICATED_GRANTS:
        return []
    out: list[str] = []
    for v in values:
        if not isinstance(v, str) or v not in GRANTS or v in out:
            return []
        out.append(v)
    return out


def parse_sse_block(block: str, plugin: bool = True):
    """parse_sse_block (aokie-plugin companion_relay.rs; the phone's is the same without subjectId and grants).

    Returns ("frame", {...}), ("end", {"seq": n or None}) or None."""
    id_ = None
    name = None
    data = ""
    for line in block.split("\n"):
        if line == "" or line.startswith(":"):
            continue
        if ":" in line:
            field, raw = line.split(":", 1)
            if raw.startswith(" "):
                raw = raw[1:]
        else:
            field, raw = line, ""
        if field == "id":
            id_ = int(raw) if re.fullmatch(r"\+?[0-9]+", raw) and int(raw) < 2 ** 64 else None    # u64::from_str
        elif field == "event":
            name = raw
        elif field == "data":
            if data:
                data += "\n"
            data += raw
    if name == "end":
        return ("end", {"seq": id_})
    if name == "frame":
        try:
            payload = json.loads(data)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        seq = payload.get("seq")
        if not is_u64(seq):
            return None
        from_ = payload.get("from")
        if not isinstance(from_, str):
            return None
        if "frame" not in payload:
            return None
        out = {"seq": seq, "from": from_, "frame": payload["frame"]}
        if plugin:
            sid = payload.get("subjectId")
            out["subjectId"] = sid if isinstance(sid, str) else None
            out["grants"] = parse_authenticated_grants(payload)
        return ("frame", out)
    return None


class SseParser:
    """SseParser::push: CRLF is normalised, events end at a blank line, a chunk may split anywhere."""

    def __init__(self, plugin: bool = True):
        self.buffer = ""
        self.plugin = plugin

    def push(self, chunk: str) -> list:
        self.buffer += chunk.replace("\r\n", "\n")
        events = []
        while True:
            i = self.buffer.find("\n\n")
            if i < 0:
                break
            block, self.buffer = self.buffer[:i], self.buffer[i + 2:]
            ev = parse_sse_block(block, self.plugin)
            if ev is not None:
                events.append(ev)
        if len(self.buffer.encode("utf-8")) > MAX_STREAM_BUFFER_BYTES:
            self.buffer = ""
        return events


def tail_cursor_page(body, since: int) -> int:
    """tail_cursor_page: body.lastSeq as a u64, else since; never less than since."""
    v = body.get("lastSeq") if isinstance(body, dict) else None
    return max(v if is_u64(v) else since, since)


def relay_post_body(party: str, frames: list[str]) -> str:
    """relay_post_body: the frames are embedded verbatim."""
    return '{"to":' + json.dumps(party, ensure_ascii=False) + ',"frames":[' + ",".join(frames) + "]}"


# ------------------------------------------------------------------------------------------------ TURN (coturn use-auth-secret)

def turn_opaque_id(secret: str, role: str, app_id: str, subject_id: str) -> str:
    """FormLogic AokieCompanionIceConfiguration::opaqueId: the first 32 hex characters of HMAC-SHA256(secret, "aokie-turn-id" NUL subject)."""
    msg = b"aokie-turn-id\x00" + f"{role}\x00{app_id}\x00{subject_id}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def turn_credential(secret: str, username: str) -> str:
    """base64(HMAC-SHA1(secret, username)): what coturn computes from the username it is sent."""
    return base64.b64encode(hmac.new(secret.encode("utf-8"), username.encode("utf-8"), hashlib.sha1).digest()).decode("ascii")


def mint_ice(config: dict, endpoint: dict, now: int) -> dict:
    """The iceServers, relayOnly and turnCredentialExpiresAt an admission must carry for a configuration (FormLogic's shape)."""
    servers = []
    if config["stunUrls"]:
        servers.append({"urls": list(config["stunUrls"]), "username": "", "credential": ""})
    expires = None
    if config["turnUrls"]:
        ttl = config["turnTtl"]
        window = min(100, 10 * max(1, ttl // 60))  # the expiry is rounded down to a window: one coturn quota key per endpoint and window
        expires = now // window * window + ttl
        username = f"{expires}:{turn_opaque_id(config['turnSecret'], endpoint['role'], endpoint['appId'], endpoint['subjectId'])}"
        servers.append({"urls": list(config["turnUrls"]), "username": username, "credential": turn_credential(config["turnSecret"], username), "expiresAt": expires})
    return {"iceServers": servers, "relayOnly": config["relayOnly"], "turnCredentialExpiresAt": expires}


# ------------------------------------------------------------------------------------------------ the compatibility error

def compat_error(doc, status: int):
    """MobileApiError [serde] deny_unknown_fields {error, code, message} and classify_admission_http_failure's conditions."""
    strict(doc, ["error", "code", "message"], ["error", "code", "message"], "error")
    need_bool(doc["error"], "error")
    need_str(doc["code"], "code")
    need_str(doc["message"], "message")
    if doc["error"] is not True:
        refuse("error is not true")
    validate_identity(doc["code"], "managed admission error code")
    if not doc["message"] or len(doc["message"].encode()) > 240 or any(is_control(c) for c in doc["message"]):
        refuse("message")
    return doc
