//! The Aokie admission fixtures (`fixtures/aokie/admission.json`, recorded from the real relay) read by this crate's admission types, and damaged copies of them refused by the
//! rules of the shipped decoders. Every request the recorded desktop broker and phone made is rebuilt byte for byte by `to_body`, every recorded answer is accepted, and every one
//! of the damages below is refused.

mod common;

use common::{load, At};
use oaiy_relay_core::admission::{MobileAdmission, MobileExpect, MobileRequest, PluginAdmission, PluginRequest, Transport};
use oaiy_relay_core::json::Json;
use oaiy_relay_core::keys::VerifyKey;

const NOW: i64 = 1_790_000_000;

fn transports(v: Option<&Json>) -> Option<Vec<Transport>> {
    v.map(|t| t.as_array().unwrap().iter().map(|x| if x.as_str() == Some("relay") { Transport::Relay } else { Transport::RelayPoll }).collect())
}

fn cases() -> Vec<Json> {
    load("fixtures/aokie/admission.json").at("cases").as_array().unwrap().to_vec()
}

fn plugin_request(body: &Json) -> PluginRequest {
    PluginRequest {
        app_id: body.get_str("appId").unwrap().into(),
        plugin_id: body.get_str("pluginId").unwrap().into(),
        display_name: body.get_str("displayName").map(str::to_string),
        endpoint: VerifyKey::from_b64u(body.at("endpointPublicKey.publicKey").as_str().unwrap()).unwrap(),
        approved_peers: body.get("approvedPeerKeyThumbprints").unwrap().as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect(),
        revision: body.n("peerRosterRevision"),
        transports: transports(body.get("supportedTransports")),
    }
}

fn mobile_request(body: &Json) -> MobileRequest {
    MobileRequest {
        app_id: body.get_str("appId").unwrap().into(),
        device_id: body.get_str("deviceId").unwrap().into(),
        display_name: body.get_str("displayName").map(str::to_string),
        holder_thumbprint: body.get_str("holderKeyThumbprint").unwrap().into(),
        transports: transports(body.get("supportedTransports")),
    }
}

#[test]
fn every_recorded_request_is_rebuilt_byte_for_byte_and_every_recorded_answer_is_accepted() {
    let (mut plugins, mut phones) = (0, 0);
    for case in cases() {
        let body = case.at("request.body");
        let answer = case.at("response.body");
        assert_eq!(case.n("response.status"), 200, "{}", case.s("name"));
        if case.s("role") == "plugin" {
            let req = plugin_request(body);
            assert_eq!(req.to_body().unwrap(), body.to_compact(), "{}: the request", case.s("name"));
            let parsed = PluginAdmission::parse(answer.to_compact().as_bytes(), &req, NOW).unwrap_or_else(|e| panic!("{}: {e}", case.s("name")));
            assert_eq!(parsed.expires_in, 90);
            assert_eq!(parsed.relay.as_ref().expect("usable endpoints").poll_mode, answer.at("relay").get("mode").is_some());
            plugins += 1;
        } else {
            let req = mobile_request(body);
            assert_eq!(req.to_body().unwrap(), body.to_compact(), "{}: the request", case.s("name"));
            let expect = MobileExpect { app_id: &req.app_id, device_id: &req.device_id, holder_thumbprint: &req.holder_thumbprint };
            let parsed = MobileAdmission::parse(answer.to_compact().as_bytes(), &expect, NOW).unwrap_or_else(|e| panic!("{}: {e}", case.s("name")));
            assert_eq!(parsed.scopes.first().map(String::as_str), Some("state_read"));
            assert_eq!(parsed.bearer.expires_at(), 1_790_000_090);
            assert_eq!(parsed.relay_only, answer.get("relayOnly").and_then(Json::as_bool).unwrap());
            phones += 1;
        }
    }
    assert_eq!((plugins, phones), (4 + 1, 5), "the nine recorded admissions and the plugin's broker form");
}

/// A copy of `doc` in which the member at `path` (dotted) is replaced by `value` (or removed when `None`).
fn damage(doc: &Json, path: &str, value: Option<Json>) -> Json {
    fn go(v: &Json, parts: &[&str], value: &Option<Json>) -> Json {
        if let Json::Arr(items) = v {
            let i: usize = parts[0].parse().unwrap_or_else(|_| panic!("not an index at {parts:?}"));
            let mut out = items.clone();
            out[i] = if parts.len() == 1 { value.clone().expect("an element is replaced, not removed") } else { go(&items[i], &parts[1..], value) };
            return Json::Arr(out);
        }
        let Json::Obj(members) = v else { panic!("not an object at {parts:?}") };
        let mut out = Vec::new();
        for (k, child) in members {
            if k == parts[0] {
                if parts.len() == 1 {
                    if let Some(new) = value {
                        out.push((k.clone(), new.clone()));
                    }
                } else {
                    out.push((k.clone(), go(child, &parts[1..], value)));
                }
            } else {
                out.push((k.clone(), child.clone()));
            }
        }
        if parts.len() == 1 && value.is_some() && !members.iter().any(|(k, _)| k == parts[0]) {
            out.push((parts[0].to_string(), value.clone().unwrap()));
        }
        Json::Obj(out)
    }
    go(doc, &path.split('.').collect::<Vec<_>>(), &value)
}

#[test]
fn damaged_phone_admissions_are_refused_by_the_rules_of_the_shipped_decoder() {
    let case = cases().into_iter().find(|c| c.s("name") == "phone A, stream, STUN and TURN").unwrap();
    let good = case.at("response.body").clone();
    let body = case.at("request.body");
    let req = mobile_request(body);
    let expect = MobileExpect { app_id: &req.app_id, device_id: &req.device_id, holder_thumbprint: &req.holder_thumbprint };
    let parse = |d: &Json, now: i64| MobileAdmission::parse(d.to_compact().as_bytes(), &expect, now);
    assert!(parse(&good, NOW).is_ok());
    let s = |t: &str| Some(Json::str(t));
    let n = |v: i64| Some(Json::int(v));
    let mut damages: Vec<(&str, Json)> = vec![
        ("an unknown member", damage(&good, "extra", n(1))),
        ("no accessToken", damage(&good, "accessToken", None)),
        ("a bearer of another shape", damage(&good, "accessToken", s("aokie-adm-v2.zz.00"))),
        ("tokenType Basic", damage(&good, "tokenType", s("Basic"))),
        ("role plugin", damage(&good, "role", s("plugin"))),
        ("expiresIn 0", damage(&good, "expiresIn", n(0))),
        ("expiresIn 301", damage(&good, "expiresIn", n(301))),
        ("expiresAt in the past", damage(&good, "expiresAt", n(NOW))),
        ("expiresAt 301 ahead", damage(&good, "expiresAt", n(NOW + 301))),
        ("a gateway that is not wss", damage(&good, "gatewayUrl", s("https://relay.example.com/v2/realtime"))),
        ("a gateway with a path", damage(&good, "gatewayUrl", s("wss://relay.example.com/other"))),
        ("another app", damage(&good, "appId", s("other"))),
        ("another subject", damage(&good, "subjectId", s("dev-AAAAAAAAAAAAAAAAAAAAAA"))),
        ("another holder", damage(&good, "holderKeyThumbprint", s("--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_cj"))),
        ("an expected peer that is the holder", damage(&good, "expectedPeerKeyThumbprint", s("--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_ck"))),
        ("no state_read", damage(&good, "scopes", Some(Json::Arr(vec![Json::str("rtc_signal")])))),
        ("an unknown grant", damage(&good, "scopes", Some(Json::Arr(vec![Json::str("state_read"), Json::str("god_mode")])))),
        ("a repeated grant", damage(&good, "scopes", Some(Json::Arr(vec![Json::str("state_read"), Json::str("state_read")])))),
        ("device.grants that are not the scopes", damage(&good, "device.grants", Some(Json::Arr(vec![Json::str("state_read")])))),
        ("a device with a member too many", damage(&good, "device.extra", n(1))),
        ("a device display name that is empty", damage(&good, "device.displayName", s(""))),
        ("a device display name of 121 bytes", damage(&good, "device.displayName", s(&"x".repeat(121)))),
        ("a stun entry with a credential", damage(&good, "iceServers.0.credential", s("x"))),
        ("a turnCredentialExpiresAt that is not the earliest", damage(&good, "turnCredentialExpiresAt", n(1_790_000_700))),
    ];
    // ICE entries inside the array need their own path handling: replace the whole list.
    let ice = good.get("iceServers").unwrap().as_array().unwrap().to_vec();
    let turn = |f: &dyn Fn(Json) -> Json| Json::Arr(vec![ice[0].clone(), f(ice[1].clone())]);
    damages.extend([
        ("a TURN entry without an expiry", damage(&good, "iceServers", Some(turn(&|e| damage(&e, "expiresAt", None))))),
        ("a TURN entry that expires in 30 seconds", damage(&good, "iceServers", Some(turn(&|e| damage(&e, "expiresAt", n(NOW + 30)))))),
        ("a TURN entry that expires in 25 hours", damage(&good, "iceServers", Some(turn(&|e| damage(&e, "expiresAt", n(NOW + 90_000)))))),
        ("a TURN entry with an empty credential", damage(&good, "iceServers", Some(turn(&|e| damage(&e, "credential", s("")))))),
        (
            "a TURN entry that mixes stun and turn urls",
            damage(&good, "iceServers", Some(turn(&|e| damage(&e, "urls", Some(Json::Arr(vec![Json::str("turn:a:1"), Json::str("stun:b:2")])))))),
        ),
        ("a url with another scheme", damage(&good, "iceServers", Some(turn(&|e| damage(&e, "urls", Some(Json::Arr(vec![Json::str("http://a")]))))))),
        ("nine servers", damage(&good, "iceServers", Some(Json::Arr(vec![ice[0].clone(); 9])))),
    ]);
    for (what, doc) in &damages {
        assert!(parse(doc, NOW).is_err(), "accepted: {what}");
    }
    // Each damage is a damage: the good one still is not refused after all of them.
    assert!(parse(&good, NOW).is_ok());
    // `relayOnly` needs a TURN entry: the recorded admission that has none is refused when it says relay only.
    let stun_only = cases().into_iter().find(|c| c.s("name") == "phone A, STUN only (no TURN configured)").unwrap();
    let doc = stun_only.at("response.body").clone();
    assert!(parse(&doc, NOW).is_ok());
    assert!(parse(&damage(&doc, "relayOnly", Some(Json::Bool(true))), NOW).is_err());
}

#[test]
fn an_unusable_relay_advertisement_degrades_the_carrier_and_never_fails_the_admission() {
    // The shipped readers (`usable_relay_endpoints`): `relay` may be absent or null; one that does not decode, has an unsafe URL or has URLs of two origins is dropped, and the
    // carrier falls back to the WebSocket gateway; an unknown member of it is ignored. The phone's `iceServers` may be absent too.
    let case = cases().into_iter().find(|c| c.s("name") == "phone A, stream, STUN and TURN").unwrap();
    let good = case.at("response.body").clone();
    let req = mobile_request(case.at("request.body"));
    let expect = MobileExpect { app_id: &req.app_id, device_id: &req.device_id, holder_thumbprint: &req.holder_thumbprint };
    let parse = |d: &Json| MobileAdmission::parse(d.to_compact().as_bytes(), &expect, NOW);
    let s = |t: &str| Some(Json::str(t));
    assert!(parse(&good).unwrap().relay.is_some());
    for (what, doc, usable) in [
        ("no relay member", damage(&good, "relay", None), false),
        ("a null relay", damage(&good, "relay", Some(Json::Null)), false),
        ("a relay that is a string", damage(&good, "relay", s("x")), false),
        ("a frames url on http", damage(&good, "relay.framesUrl", s("http://relay.example.com/x")), false),
        ("a url with credentials", damage(&good, "relay.framesUrl", s("https://user:pw@relay.example.com/x")), false),
        ("a url with a fragment", damage(&good, "relay.framesUrl", s("https://relay.example.com/x#y")), false),
        ("urls of two origins", damage(&good, "relay.framesUrl", s("https://other.example.com/x")), false),
        ("a missing url", damage(&good, "relay.streamUrl", None), false),
        ("an unknown relay member", damage(&good, "relay.extra", s("x")), true),
    ] {
        let parsed = parse(&doc).unwrap_or_else(|e| panic!("refused, though the shipped phone only degrades: {what}: {e}"));
        assert_eq!(parsed.relay.is_some(), usable, "{what}");
    }
    let no_ice = parse(&damage(&good, "iceServers", None));
    assert!(
        no_ice.is_err(),
        "this admission has a TURN expiry, so the missing list is a mismatch (turnCredentialExpiresAt is not the earliest TURN expiry)"
    );
    let stun_only = cases().into_iter().find(|c| c.s("name") == "phone A, STUN only (no TURN configured)").unwrap();
    let doc = damage(stun_only.at("response.body"), "iceServers", None);
    let parsed = parse(&doc).expect("an admission with no iceServers member is an admission with no ICE servers");
    assert!(parsed.ice_servers.is_empty());

    // The plugin's, the same.
    let case = cases().into_iter().find(|c| c.s("name") == "plugin, stream, STUN and TURN").unwrap();
    let good = case.at("response.body").clone();
    let req = plugin_request(case.at("request.body"));
    let parse = |d: &Json| PluginAdmission::parse(d.to_compact().as_bytes(), &req, NOW);
    assert!(parse(&damage(&good, "relay", None)).unwrap().relay.is_none());
    assert!(parse(&damage(&good, "relay.framesUrl", s("http://relay.example.com/x"))).unwrap().relay.is_none());
    assert!(parse(&damage(&good, "iceServers", None)).is_err(), "the plugin requires iceServers");
}
#[test]
fn damaged_plugin_admissions_are_refused_and_an_echo_of_another_roster_is_a_mismatch() {
    let case = cases().into_iter().find(|c| c.s("name") == "plugin, stream, STUN and TURN").unwrap();
    let good = case.at("response.body").clone();
    let req = plugin_request(case.at("request.body"));
    let parse = |d: &Json| PluginAdmission::parse(d.to_compact().as_bytes(), &req, NOW);
    assert!(parse(&good).is_ok());
    let s = |t: &str| Some(Json::str(t));
    let n = |v: i64| Some(Json::int(v));
    for (what, doc) in [
        ("an unknown member", damage(&good, "desktopConnection", n(1))),
        ("another role", damage(&good, "role", s("mobile"))),
        ("expiresIn 10", damage(&good, "expiresIn", n(10))),
        ("expiresAt 10 ahead", damage(&good, "expiresAt", n(NOW + 10))),
        ("another plugin id", damage(&good, "subjectId", s("other"))),
        ("another app", damage(&good, "appId", s("other"))),
        ("another endpoint key", damage(&good, "endpointPublicKey.publicKey", s("6kpsY-KcUgq-9VB7Ey7F-ZVHdq6-vnuSQh7qaRRG0iw"))),
        ("another holder", damage(&good, "holderKeyThumbprint", s("--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_ck"))),
        (
            "another roster",
            damage(&good, "approvedPeerKeyThumbprints", Some(Json::Arr(vec![Json::str("--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_ck")]))),
        ),
        ("another revision", damage(&good, "peerRosterRevision", n(8))),
        ("another roster hash", damage(&good, "peerRosterHash", s("7ZEsQj6A1UiNIqXrEYfYSjfKRGenwlQVgZOvWXQQ-Nt"))),
    ] {
        assert!(parse(&doc).is_err(), "accepted: {what}");
    }
}

#[test]
fn a_request_a_client_would_send_is_checked_before_it_is_sent() {
    let case = cases().into_iter().find(|c| c.s("name") == "plugin, stream, STUN and TURN").unwrap();
    let ok = plugin_request(case.at("request.body"));
    assert!(ok.to_body().is_ok());
    let mut unsorted = ok.clone();
    unsorted.approved_peers.reverse();
    assert!(unsorted.to_body().is_err(), "the roster must be strictly ascending");
    let mut own = ok.clone();
    own.approved_peers.push(own.endpoint.thumbprint());
    own.approved_peers.sort();
    assert!(own.to_body().is_err(), "none may be the plugin's own key");
    let mut empty = ok.clone();
    empty.approved_peers.clear();
    assert!(empty.to_body().is_err());
    let mut zero = ok.clone();
    zero.revision = 0;
    assert!(zero.to_body().is_err());
    let mut many = ok.clone();
    many.approved_peers = (0..17).map(|i| oaiy_relay_core::b64::encode(&oaiy_crypto::kdf::sha256(&[i as u8]))).collect();
    many.approved_peers.sort();
    assert!(many.to_body().is_err(), "17 phones");
    let phone = mobile_request(cases().iter().find(|c| c.s("role") != "plugin").unwrap().at("request.body"));
    let mut bad = phone.clone();
    bad.device_id = "nope".into();
    assert!(bad.to_body().is_err());
    let mut twice = phone.clone();
    twice.transports = Some(vec![Transport::Relay, Transport::Relay]);
    assert!(twice.to_body().is_err());
    twice.transports = Some(vec![]);
    assert!(twice.to_body().is_err());
}

#[test]
fn a_phones_admission_is_compared_with_the_grants_the_desktop_signed_and_the_desktop_the_phone_pinned() {
    // `MobileAdmission::check_against`, on every recorded phone admission: the scopes are exactly the grants of the approval receipt (as a set), and the expected peer is the desktop
    // that was pinned from the offer. More, fewer, a repeat, another peer and a thumbprint that is no thumbprint are each refused.
    let mut checked = 0;
    for case in cases().into_iter().filter(|c| c.s("role") != "plugin") {
        let req = mobile_request(case.at("request.body"));
        let expect = MobileExpect { app_id: &req.app_id, device_id: &req.device_id, holder_thumbprint: &req.holder_thumbprint };
        let adm = MobileAdmission::parse(case.at("response.body").to_compact().as_bytes(), &expect, NOW).unwrap();
        let (grants, peer) = (adm.scopes.clone(), adm.expected_peer_thumbprint.clone());
        adm.check_against(&grants, &peer).unwrap_or_else(|e| panic!("{}: {e}", case.s("name")));
        let reversed: Vec<String> = grants.iter().rev().cloned().collect();
        adm.check_against(&reversed, &peer).expect("a set, not a list");
        let mut more = grants.clone();
        more.push("assistance_respond".to_string());
        let fewer = grants[1..].to_vec();
        let mut repeated = grants.clone();
        repeated.push(grants[0].clone());
        for (what, g) in [("more grants than the scopes", &more), ("fewer", &fewer), ("a repeated grant", &repeated), ("none", &Vec::new())] {
            if what == "more grants than the scopes" && grants.iter().any(|x| x == "assistance_respond") {
                continue; // the recorded scopes already hold it: nothing is added
            }
            assert!(adm.check_against(g, &peer).is_err(), "{}: {what}", case.s("name"));
        }
        let other_peer = "Zq7e1o0c2mS4N5t-XvB9aLkJ3dYhPfRwU6iGg8TnQpA";
        assert_ne!(other_peer, peer);
        assert!(adm.check_against(&grants, other_peer).is_err(), "{}: another desktop", case.s("name"));
        assert!(adm.check_against(&grants, "not a thumbprint").is_err());
        assert!(adm.check_against(&grants, "").is_err());
        checked += 1;
    }
    assert_eq!(checked, 5);
}
