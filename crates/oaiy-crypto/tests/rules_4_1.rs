//! Design section 4.1, rule by rule. Each test is named for the rule it carries (the README lists them): the registry of KDF contexts
//! (4.1.2), the prefix-free set of domains, no caller-chosen domain, no free text in a signed string (4.1.3, attack A1 and A2), one key
//! in one protocol (4.1.4), and the encodings (4.1.5).

mod common;

use std::collections::BTreeSet;

use oaiy_crypto::canon::{self, Aad, AadDomain, DomainSpec, DomainUse, Separator, DOMAINS};
use oaiy_crypto::ed25519::{KeyRole, SigDomain, SignedString, SigningKey};
use oaiy_crypto::kdf::{self, Context, Purpose, Status, REGISTRY};
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

const PIPE_DOMAINS: [SigDomain; 7] =
    [SigDomain::Manifest, SigDomain::Grant, SigDomain::VaultOp, SigDomain::VaultHead, SigDomain::Writer, SigDomain::Arch, SigDomain::Backup];
const LF_DOMAINS: [SigDomain; 2] = [SigDomain::Placement, SigDomain::NodeCert];

#[test]
fn rule_4_1_2_the_kdf_context_registry_is_the_design_table_and_has_no_duplicates() {
    // the table of 4.1.2, row for row: append-only, so a row that changes or goes fails here until someone means it
    let table: Vec<(&str, u64, &str, Status)> = REGISTRY.iter().map(|e| (e.context.as_str(), e.id, e.master, e.status)).collect();
    assert_eq!(
        table,
        vec![
            ("flrecov1", 1, "FLRK1 kit", Status::Existing),
            ("flphras1", 1, "argon2id13(entropy16, salt16, 3, 64 MiB)", Status::New),
            ("flprf001", 1, "WebAuthn PRF output", Status::Reserved),
            ("flbkrcp1", 1, "UMK", Status::New),
            ("flbksig1", 1, "UMK", Status::New),
            ("fllocal1", 1, "UMK", Status::Reserved),
        ]
    );
    let contexts: BTreeSet<&str> = REGISTRY.iter().map(|e| e.context.as_str()).collect();
    assert_eq!(contexts.len(), REGISTRY.len(), "no context appears twice");
    let pairs: BTreeSet<(&str, u64)> = REGISTRY.iter().map(|e| (e.context.as_str(), e.id)).collect();
    assert_eq!(pairs.len(), REGISTRY.len());
    for entry in REGISTRY {
        assert!(Context::new(entry.context.as_str()).is_ok(), "{} is eight characters of [a-z0-9]", entry.context.as_str());
        assert!(!entry.purpose.is_empty() && !entry.master.is_empty());
    }
    // every purpose maps to exactly one row, and no two purposes to the same row
    let by_purpose: BTreeSet<&str> =
        [Purpose::KitWrap, Purpose::PhraseWrap, Purpose::PrfWrap, Purpose::BackupRecipient, Purpose::BackupSigning, Purpose::LocalData]
            .iter()
            .map(|p| p.entry().context.as_str())
            .collect();
    assert_eq!(by_purpose, contexts);
    // no reuse: one master gives six different keys, one per context
    let master: Secret<32> = Secret::new([0x42; 32]);
    let keys: BTreeSet<[u8; 32]> = REGISTRY.iter().map(|e| *kdf::derive_subkey(&master, e.id, &e.context).unwrap().expose()).collect();
    assert_eq!(keys.len(), REGISTRY.len());
}

#[test]
fn rule_4_1_3_1_the_domains_are_prefix_free_and_the_enums_match_the_registry() {
    assert_eq!(canon::prefix_free(DOMAINS), Ok(()));
    assert_eq!(DOMAINS.len(), 16, "seven AAD domains and nine signature domains");
    // every SigDomain and AadDomain is a registry row with the same token, separator and role, and every row is one of them
    let mut seen = BTreeSet::new();
    for domain in PIPE_DOMAINS.iter().chain(LF_DOMAINS.iter()) {
        let row = DOMAINS.iter().find(|d| d.token == domain.token()).expect("registered");
        assert_eq!(row.separator, domain.separator(), "{}", domain.token());
        assert_eq!(row.usage, DomainUse::Signature(domain.role()), "{}", domain.token());
        assert!(seen.insert(domain.token()));
    }
    for domain in [
        AadDomain::Enc,
        AadDomain::VaultUmk,
        AadDomain::VaultBundle,
        AadDomain::VaultUmkRecovery,
        AadDomain::VaultWrap,
        AadDomain::Ingest,
        AadDomain::VaultFk,
    ] {
        let row = DOMAINS.iter().find(|d| d.token == domain.token()).expect("registered");
        assert_eq!((row.separator, row.usage), (Separator::Pipe, DomainUse::Aad));
        assert!(seen.insert(domain.token()));
    }
    assert_eq!(seen.len(), DOMAINS.len());
    // the first token up to | or LF differs in every row, and every token carries its version
    for row in DOMAINS {
        assert!(row.token.ends_with(":1") || row.token.ends_with(":2"), "{}", row.token);
        assert!(!row.token.contains('|') && !row.token.contains('\n'));
    }
}

#[test]
fn rule_4_1_3_1_the_prefix_check_catches_what_it_is_for() {
    let row = |token, separator| DomainSpec { token, separator, usage: DomainUse::Aad };
    let sig = |token, separator| DomainSpec { token, separator, usage: DomainUse::Signature(KeyRole::Vault) };
    // a duplicate token, with the same separator or another
    assert!(canon::prefix_free(&[row("flx:1", Separator::Pipe), row("flx:1", Separator::Pipe)]).is_err());
    assert!(canon::prefix_free(&[row("flx:1", Separator::Pipe), sig("flx:1", Separator::Lf)]).is_err());
    // one token a prefix of another
    assert!(canon::prefix_free(&[row("flx:1", Separator::Pipe), row("flx:10", Separator::Pipe)]).is_err());
    assert!(canon::prefix_free(&[row("flx", Separator::Pipe), row("flx:1", Separator::Pipe)]).is_err());
    assert!(canon::prefix_free(&[row("flx:1", Separator::Pipe), row("flx:1|y:1", Separator::Pipe)]).is_err());
    // unrelated tokens are fine, and so is the same stem with different letters after it
    assert!(canon::prefix_free(&[row("flx:1", Separator::Pipe), row("fly:1", Separator::Pipe), row("flx-y:1", Separator::Lf)]).is_ok());
    assert!(canon::prefix_free(&[]).is_ok());
    // a row appended to the real registry that breaks the rule is found, and reported by index
    let mut broken = DOMAINS.to_vec();
    broken.push(row("flvault-op:10", Separator::Pipe));
    let (a, b) = canon::prefix_free(&broken).unwrap_err();
    assert_eq!(broken[a].token, "flvault-op:1");
    assert_eq!(broken[b].token, "flvault-op:10");
    let mut broken = DOMAINS.to_vec();
    broken.push(row("flarch:1", Separator::Pipe));
    assert!(canon::prefix_free(&broken).is_err(), "a duplicate of a row of the registry");
}

#[test]
fn rule_4_1_3_2_no_operation_accepts_a_caller_chosen_domain() {
    let fields = ["a", "b"];
    // the domain is an enum; a pipe domain built as an LF one, and the reverse, is refused
    for domain in PIPE_DOMAINS {
        assert!(SignedString::pipe(domain, &fields).is_ok(), "{}", domain.token());
        assert_eq!(SignedString::lf(domain, b"{}").unwrap_err(), Error::DomainNotAllowed, "{}", domain.token());
    }
    for domain in LF_DOMAINS {
        assert!(SignedString::lf(domain, b"{\"a\":1}").is_ok(), "{}", domain.token());
        assert_eq!(SignedString::pipe(domain, &fields).unwrap_err(), Error::DomainNotAllowed, "{}", domain.token());
    }
    // no fields, no string
    assert!(SignedString::pipe(SigDomain::VaultOp, &[]).is_err());
    assert!(SignedString::lf(SigDomain::Placement, b"").is_err());
    assert!(SignedString::lf(SigDomain::Placement, &[0xff, 0xfe]).is_err(), "the payload is UTF-8 JSON");
    // an LF payload is a JSON document and may hold `|`; it is the one place free bytes go, and only in the two data-node domains
    assert!(SignedString::lf(SigDomain::NodeCert, br#"{"a":"x|y"}"#).is_ok());
    // what is signed is exactly `token`, the separator and the fields: nothing else can be prepended
    assert_eq!(SignedString::pipe(SigDomain::VaultOp, &["u", "op", "1", "h"]).unwrap().as_bytes(), b"flvault-op:1|u|op|1|h");
    assert_eq!(SignedString::lf(SigDomain::Placement, b"{}").unwrap().as_bytes(), b"flplacement:1\n{}");
    // signing arbitrary bytes needs a Hazmat key; every role-bound key refuses it
    for role in [KeyRole::Vault, KeyRole::Writer, KeyRole::Backup] {
        let key = SigningKey::from_seed(role, &Secret::new([1; 32]));
        assert_eq!(key.sign_raw(b"flvault-op:1|anything").unwrap_err(), Error::DomainNotAllowed, "{role:?}");
    }
    assert!(SigningKey::from_seed(KeyRole::Hazmat, &Secret::new([1; 32])).sign_raw(b"anything").is_ok());
}

#[test]
fn rule_4_1_3_3_a_pipe_or_lf_or_any_free_text_in_a_signed_field_is_refused() {
    let hostile = [
        "|",
        "a|b",
        "|a",
        "a|",
        "\n",
        "a\nb",
        "\r",
        "a\r\nb",
        " ",
        "Office PC",
        "\t",
        "x|2027-01-01",
        "Firefox on Laptop|2027-01-01",
        "",
        "é",
        "日本",
        "a\u{ff5c}b",
        "a%7Cb",
        "a\0b",
        "\u{2028}",
        "a,b",
        "a;b",
        "a\\b",
        "\"a\"",
        "'a'",
        "a b",
        "a#b",
        "a?b",
        "a*b",
        "a(b)",
        "<a>",
        "a~b",
        "a`b",
        "a!b",
        "a$b",
        "a%b",
        "a&b",
        "a^b",
        "a[b]",
        "a{b}",
    ];
    for domain in PIPE_DOMAINS {
        for position in 0..3 {
            for field in hostile {
                let mut fields = ["ok", "ok", "ok"];
                fields[position] = field;
                assert_eq!(
                    SignedString::pipe(domain, &fields).unwrap_err(),
                    Error::InvalidComponent("field"),
                    "{} field {position} {field:?}",
                    domain.token()
                );
            }
        }
    }
    for domain in [AadDomain::Enc, AadDomain::VaultUmk, AadDomain::VaultWrap, AadDomain::Ingest] {
        for field in hostile {
            assert!(Aad::new(domain, &["ok", field]).is_err(), "AAD {} {field:?}", domain.token());
        }
    }
    // every character the rule allows is allowed, and only those
    let allowed: String = ('A'..='Z').chain('a'..='z').chain('0'..='9').chain("_.:+@=/-".chars()).collect();
    assert!(SignedString::pipe(SigDomain::VaultOp, &[&allowed]).is_ok());
    for c in allowed.chars() {
        assert!(SignedString::pipe(SigDomain::VaultOp, &[&c.to_string()]).is_ok(), "{c}");
        assert!(canon::is_component(&c.to_string()));
    }
    for byte in 0u8..=255 {
        let c = char::from(byte);
        if !allowed.contains(c) {
            assert!(!canon::is_component(&c.to_string()), "{byte:#04x} is not a component character");
        }
    }
    // the way free text enters is its hash, which is hex
    let label_hash = kdf::sha256_hex("Firefox on Laptop|2027-01-01".as_bytes());
    assert!(SignedString::pipe(SigDomain::VaultOp, &["u", "device.register", "-", &label_hash]).is_ok());
    // dates and times as the design writes them are components
    assert!(canon::is_component("2026-09-30T03:30:00Z"));
    assert!(canon::is_component("argon2id13.1"));
    // base64 with padding is one, so a wrapped key or a public key can be a field
    assert!(canon::is_component("oKGio6SlpqeoqaqrrK2urw=="));
}

#[test]
fn rule_4_1_4_r_key_a_key_has_one_role_and_signs_only_its_domains() {
    let vault_domains: BTreeSet<&str> = [
        SigDomain::Manifest,
        SigDomain::Grant,
        SigDomain::Placement,
        SigDomain::NodeCert,
        SigDomain::VaultOp,
        SigDomain::VaultHead,
        SigDomain::Writer,
    ]
    .iter()
    .map(|d| d.token())
    .collect();
    let all: Vec<(SigDomain, SignedString)> = PIPE_DOMAINS
        .iter()
        .map(|d| (*d, SignedString::pipe(*d, &["a"]).unwrap()))
        .chain(LF_DOMAINS.iter().map(|d| (*d, SignedString::lf(*d, b"{}").unwrap())))
        .collect();
    assert_eq!(all.len(), 9);
    for (role, expected) in [
        (KeyRole::Vault, vault_domains.clone()),
        (KeyRole::Writer, BTreeSet::from(["flarch:1"])),
        (KeyRole::Backup, BTreeSet::from(["flbackup:1"])),
        (KeyRole::Hazmat, all.iter().map(|(d, _)| d.token()).collect()),
    ] {
        let key = SigningKey::from_seed(role, &Secret::new([7; 32]));
        let signs: BTreeSet<&str> = all.iter().filter(|(_, s)| key.sign(s).is_ok()).map(|(d, _)| d.token()).collect();
        assert_eq!(signs, expected, "{role:?}");
        for (domain, string) in &all {
            if !expected.contains(domain.token()) {
                assert_eq!(key.sign(string).unwrap_err(), Error::DomainNotAllowed, "{role:?} {}", domain.token());
            }
        }
    }
    // a signature made in one domain does not verify as another (the strings differ in the first bytes)
    let key = SigningKey::from_seed(KeyRole::Vault, &Secret::new([7; 32]));
    let op = SignedString::pipe(SigDomain::VaultOp, &["u", "x"]).unwrap();
    let head = SignedString::pipe(SigDomain::VaultHead, &["u", "x"]).unwrap();
    let signature = key.sign(&op).unwrap();
    assert!(key.verifying_key().verify(&op, &signature).is_ok());
    assert!(key.verifying_key().verify(&head, &signature).is_err());
    assert!(
        key.verifying_key().verify_raw(b"flvault-op:1|u|x", &signature).is_ok(),
        "the string is exactly the domain, the separator and the fields"
    );
}

#[test]
fn rule_4_1_5_hex_is_lowercase_and_the_kdf_salt_is_a_little_endian_u64() {
    assert_eq!(kdf::hex_lower(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    assert_eq!(kdf::sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert!(kdf::sha256_hex(b"x").chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    // subkey ids are 64-bit little-endian in the salt: ids that agree in their low 32 bits differ, and the byte order is not swapped
    let master: Secret<32> = Secret::new([9; 32]);
    let context = Context::new("flrecov1").unwrap();
    let ids = [1u64, 1 << 32, 1 << 8, 1 << 56, u64::MAX];
    let keys: BTreeSet<[u8; 32]> = ids.iter().map(|id| *kdf::derive_subkey(&master, *id, &context).unwrap().expose()).collect();
    assert_eq!(keys.len(), ids.len());
    // the AAD is UTF-8, `|`-delimited, ASCII, and is never JSON
    let aad = Aad::new(AadDomain::Enc, &["form-a", "id", "1", "-"]).unwrap();
    assert_eq!(aad.as_bytes(), b"flenc:1|form-a|id|1|-");
    assert!(aad.as_bytes().is_ascii());
}
