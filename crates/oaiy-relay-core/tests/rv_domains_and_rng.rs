//! Review test (F1): (1) a signature made under one of the protocol's domains verifies under no other, and no domain message can equal a JWS signing input (the ticket's, which has no
//! domain); (2) the desktop's pairing secret, nonce and `jti` come from whatever `Rng` the host passes to `create_offer`, so a predictable `Rng` makes the pairing secret predictable
//! (the crate's own `PairingSecret::generate`, which draws from the operating system, is not what `create_offer` uses).

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::client::Rng;
use oaiy_relay_core::keys::{domain_message, SignDomain, Signer, X25519Secret};
use oaiy_relay_core::pairing::math;
use oaiy_relay_core::pairing::{DesktopIdentity, DesktopPairing};
use oaiy_relay_core::testing::SeededRng;
use oaiy_relay_core::url::RelayUrl;

const ALL: [SignDomain; 10] = [
    SignDomain::Info,
    SignDomain::InfoProof,
    SignDomain::Enroll,
    SignDomain::Cmd,
    SignDomain::Res,
    SignDomain::Sync,
    SignDomain::ProviderRotate,
    SignDomain::Ring,
    SignDomain::PairingResponse,
    SignDomain::PairingApproval,
];

#[test]
fn a_signature_under_one_domain_verifies_under_no_other_whatever_the_parts() {
    let signer = Signer::from_seed(&Secret::new([3; 32]));
    let key = signer.verify_key();
    // texts chosen to make a boundary ambiguous: a suffix of a longer domain, the zero byte, an empty text, text that looks like another domain
    let texts: Vec<Vec<u8>> =
        vec![b"payload".to_vec(), Vec::new(), b"-proof\0abc".to_vec(), b"\0".to_vec(), b"oaiy/relay/1/res\0payload".to_vec(), b"proof".to_vec()];
    for d1 in ALL {
        for t in &texts {
            let sig = signer.sign(d1, &[t]);
            for d2 in ALL {
                assert_eq!(key.verify(d2, &[t], &sig).is_ok(), d1 == d2, "{d1:?} signature under {d2:?}");
                for u in &texts {
                    // the same signature over another text under another domain: only the identical message verifies
                    let same_bytes = d1.message(&[t]) == d2.message(&[u]);
                    assert_eq!(key.verify(d2, &[u], &sig).is_ok(), same_bytes, "{d1:?}/{t:?} vs {d2:?}/{u:?}");
                    assert!(!same_bytes || (d1 == d2 && t == u), "two different (domain, text) pairs make one message: {d1:?} {t:?} {d2:?} {u:?}");
                }
            }
        }
    }
}

#[test]
fn a_domain_message_always_holds_a_zero_byte_and_a_jws_signing_input_never_does() {
    // README 9.2: the ticket's signature is over `b64u(header) "." b64u(payload)` with no domain. Its bytes are the base64url alphabet and a dot; every domain message has the zero byte
    // that ends the domain (and no domain string holds one), so the two sets cannot meet.
    for d in ALL {
        assert!(!d.as_str().as_bytes().contains(&0));
        assert!(d.message(&[b"x"]).contains(&0));
        assert_eq!(d.message(&[]).iter().position(|b| *b == 0), Some(d.as_str().len()));
    }
    let jws = b"eyJhbGciOiJFZERTQSIsInR5cCI6Im9haXktdGlja2V0K2p3dCIsImtpZCI6IngifQ.eyJpc3MiOiJwcm92LXgifQ";
    assert!(!jws.contains(&0));
    assert!(jws.iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')));
    // the MAC's domain strings are not signature domains and the other way round
    let mac_domains = [math::OFFER_MAC_DOMAIN, math::RESPONSE_MAC_DOMAIN, math::TYPED_DOMAIN, math::SAS_DOMAIN, math::SAS_CHECK_DOMAIN];
    for m in mac_domains {
        assert!(ALL.iter().all(|d| d.as_str() != m));
    }
    // the zero byte separates "oaiy/pairing/3/sas" from "oaiy/pairing/3/sas-check" as well
    assert_ne!(domain_message(math::SAS_DOMAIN, &[b"-check"]), domain_message(math::SAS_CHECK_DOMAIN, &[b""]));
}

fn desktop() -> DesktopPairing {
    let thumb = Signer::generate().unwrap().thumbprint();
    let identity = DesktopIdentity {
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        name: "Desk".into(),
        endpoint: Signer::generate().unwrap(),
        endpoint_x25519: X25519Secret::generate().unwrap().public_key(),
        host_ed25519: Signer::generate().unwrap().verify_key(),
        host_x25519: X25519Secret::generate().unwrap().public_key(),
    };
    DesktopPairing::new(std::sync::Arc::new(identity), "aokie", RelayUrl::parse("https://relay.example.com").unwrap(), &thumb)
}

#[test]
fn the_pairing_secret_is_whatever_the_hosts_rng_says() {
    // The secret is the first 16 bytes the Rng gives, the nonce the next 32, the jti the 12 after: with a seeded generator all three are known in advance.
    let mut replay = SeededRng::new(5);
    let mut secret = [0u8; 16];
    replay.fill(&mut secret);
    let predicted = math::typed_code(&secret);
    let a = desktop().create_offer(&mut SeededRng::new(5), 1_790_000_000).unwrap();
    let b = desktop().create_offer(&mut SeededRng::new(5), 1_790_000_000).unwrap();
    assert_eq!(a.typed_code, predicted, "the typed code (the secret) is a function of the Rng alone");
    assert_eq!(a.typed_code, b.typed_code);
    assert_eq!(a.offer.nonce, b.offer.nonce);
    // and the crate's own OS-random constructor is not what create_offer uses
    let os = math::PairingSecret::generate().unwrap();
    assert_ne!(os.typed_code(), a.typed_code);
}
