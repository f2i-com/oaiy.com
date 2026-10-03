//! Review test (F1): which types print a secret through `{:?}`. The first test pins what the crate does right (every secret-holding type of the crate has a redacting `Debug`); the second
//! pins what it does not: derived `Debug` on a few types that carry a secret in the clear. The second test PASSES today, so that it shows the leak; when the leaks are fixed it fails and
//! should be inverted.

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::admission::IceServer;
use oaiy_relay_core::client::{HttpRequest, HttpResponse, Method};
use oaiy_relay_core::enrol::{EnrolmentKey, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::math::{PairingKey, PairingSecret};
use oaiy_relay_core::pairing::PairingInput;
use oaiy_relay_core::url::RelayUrl;

const TOKEN: &str = "oaiyrt1.AAECAwQFBgc.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8";

#[test]
fn the_secret_holding_types_print_nothing_of_the_secret() {
    let token = Token::parse(TOKEN).unwrap();
    let secret_part = &TOKEN[20..];
    assert!(!format!("{token:?}").contains(secret_part));
    let signer = Signer::from_seed(&Secret::new([0x5a; 32]));
    assert!(!format!("{signer:?}").to_lowercase().contains("5a5a5a"));
    let x = X25519Secret::from_secret(&Secret::new([0x5b; 32]));
    assert!(!format!("{x:?}").to_lowercase().contains("5b5b5b"));
    let s = PairingSecret::new([0x77; 16]);
    let typed = s.typed_code();
    assert!(!format!("{s:?}").contains(&typed[..9]) && !format!("{s:?}").contains(&s.b64u()));
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let thumb = signer.thumbprint();
    let uri = PairingKey::to_uri(&relay, &thumb, &s, 1_790_000_000);
    let key = PairingKey::parse(&uri).unwrap();
    assert!(!format!("{key:?}").contains(&s.b64u()));
    let ek_uri = EnrolmentKey::to_uri(&relay, &thumb, &[0x31; 16], Role::Desktop, 1_790_000_000).unwrap();
    let ek = EnrolmentKey::parse(&ek_uri).unwrap();
    let ek_secret = ek_uri.split("&s=").nth(1).unwrap().split('&').next().unwrap();
    assert!(!format!("{ek:?}").contains(ek_secret));
    let request = HttpRequest {
        method: Method::Get,
        url: "https://relay.example.com/v1/poll".into(),
        headers: vec![("Authorization".into(), format!("Bearer {TOKEN}"))],
        body: None,
        timeout: std::time::Duration::from_secs(1),
        max_response_bytes: 10,
        cancel: Default::default(),
    };
    assert!(!format!("{request:?}").contains(secret_part));
}

/// Inverted with the fix, as the test's own comment said to do: this used to assert that these three derived `Debug` impls print a secret in the clear.
#[test]
fn and_the_three_derived_debug_impls_no_longer_print_a_secret() {
    // 1. The pairing secret, as the owner types it or as the pairing key carries it, is not in `PairingInput`'s Debug.
    let s = PairingSecret::new([0x77; 16]);
    let typed = s.typed_code();
    let dbg = format!("{:?}", PairingInput::Typed { code: &typed, host: "relay.example.com" });
    assert!(!dbg.contains(&typed) && !dbg.contains(&typed[..4]) && dbg.contains("relay.example.com"), "{dbg}");
    let uri = format!("oaiy://pair?v=3&u=https%3A%2F%2Frelay.example.com&s={}", s.b64u());
    let dbg = format!("{:?}", PairingInput::Key(&uri));
    assert!(!dbg.contains(&s.b64u()) && !dbg.contains("oaiy://"), "{dbg}");

    // 2. A TURN credential of an admission is not in `IceServer`'s Debug.
    let ice = IceServer {
        urls: vec!["turns:turn.example.com:443".into()],
        username: "1790000600:abc".into(),
        credential: "TURN-CREDENTIAL-1234".into(),
        expires_at: Some(1_790_000_600),
    };
    let dbg = format!("{ice:?}");
    assert!(!dbg.contains("TURN-CREDENTIAL-1234") && dbg.contains("turns:turn.example.com:443"), "{dbg}");

    // 3. A response body that carries the device token is not in `HttpResponse`'s Debug, in any spelling.
    let body = format!("{{\"deviceId\":\"dev-x\",\"token\":\"{TOKEN}\"}}").into_bytes();
    let response = HttpResponse { status: 201, headers: vec![], body: body.clone() };
    let dbg = format!("{response:?}");
    let token_bytes = format!("{:?}", TOKEN.as_bytes());
    assert!(!dbg.contains(&token_bytes[1..token_bytes.len() - 1]) && !dbg.contains(TOKEN) && !dbg.contains(&TOKEN[20..]), "{dbg}");
}
