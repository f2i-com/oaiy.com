//! Reviewer's findings of low weight in the text handling (F2), each as a test that fails today and is therefore ignored (run with `--ignored` to see them):
//! the same origin spelled with its default port, a ticket's `sub` counted in bytes where the schema counts characters, a ticket's `org` that is looser than the schema's
//! `origin` pattern, and a `Sas` whose fields do not hold what `display` slices.

use oaiy_crypto::ed25519::{KeyRole, SigningKey};
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::keys::thumbprint_of;
use oaiy_relay_core::pairing::math::Sas;
use oaiy_relay_core::ticket;
use oaiy_relay_core::url::RelayUrl;
use oaiy_relay_core::{b64, keys::VerifyKey};

#[test]
fn the_default_port_is_the_same_origin() {
    assert_eq!(RelayUrl::parse("https://relay.example.com:443").unwrap(), RelayUrl::parse("https://relay.example.com").unwrap());
}

#[test]
fn a_sas_built_by_a_host_does_not_panic() {
    let sas = Sas { raw: [0; 8], chars12: "AB".into(), check: '0' };
    assert!(std::panic::catch_unwind(|| sas.display()).is_ok());
}

fn ticket_with(sub: &str, org: &str) -> (String, VerifyKey) {
    let seed = Secret::new([5u8; 32]);
    let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
    let public = key.verifying_key().to_bytes();
    let kid = thumbprint_of(&public);
    let header = format!("{{\"alg\":\"EdDSA\",\"typ\":\"oaiy-ticket+jwt\",\"kid\":\"{kid}\"}}");
    let payload = format!(
        "{{\"iss\":\"prov-AAAAAAAAAAAAAAAAAAAAAA\",\"aud\":\"rly-AAAAAAAAAAAAAAAAAAAAAA\",\"sub\":\"{sub}\",\"iat\":1790000000,\"exp\":1790000300,\"jti\":\"tkt-1\",\"lane\":\"ai\",\"dev\":\"dev-AAAAAAAAAAAAAAAAAAAAAA\",\"org\":\"{org}\",\"eph\":\"{kid}\"}}"
    );
    let signing_input = format!("{}.{}", b64::encode(header.as_bytes()), b64::encode(payload.as_bytes()));
    let sig = key.sign_raw(signing_input.as_bytes()).unwrap();
    (format!("{signing_input}.{}", b64::encode(&sig.to_bytes())), VerifyKey::from_bytes(&public).unwrap())
}

#[test]
fn a_plain_ticket_verifies() {
    let (t, k) = ticket_with("member-42", "https://app.example.com");
    assert!(ticket::verify_signature(&t, &k).is_ok());
}

#[test]
#[ignore = "low: ticket-claims.schema.json allows a sub of 64 characters; ticket.rs:102 counts bytes (30 CJK characters are 90 bytes)"]
fn a_sub_of_thirty_cjk_characters_is_within_the_schema() {
    let (t, k) = ticket_with(&"\u{65e5}".repeat(30), "https://app.example.com");
    assert!(ticket::verify_signature(&t, &k).is_ok());
}

#[test]
#[ignore = "low: ticket.rs:96-99 accepts an org such as https://a:b: or https://a: that the schema's origin pattern ^https://[A-Za-z0-9.-]+(:[0-9]{1,5})?$ refuses"]
fn an_org_that_is_not_an_origin_is_refused() {
    for org in ["https://a:", "https://a:b:c", "https://a:123456789", "https://:", "https://a:-"] {
        let (t, k) = ticket_with("member-42", org);
        assert!(ticket::verify_signature(&t, &k).is_err(), "{org}");
    }
}
