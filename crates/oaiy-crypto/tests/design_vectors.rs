//! `design/vault-work/vectors.json`, the design's own vectors (computed by Node with libsodium and by Python with OpenSSL and a
//! hand-written age, 59 + 11 cross-checks, `run_all.log`), reproduced by this crate: the BIP-39 phrase, the phrase wrapper of 4.2.1,
//! the KDF registry, the backup identity, the backup manifest signature, the signed vault operations and head, the archive
//! signature, the ceremony (X25519, HKDF-SHA256, XChaCha20-Poly1305 with the transcript as AAD) and the Ed25519 strictness probe.
//! The file is a byte-for-byte copy, SHA-256 `7057862f...`.

mod common;

use common::*;
use oaiy_crypto::aead;
use oaiy_crypto::argon;
use oaiy_crypto::bip39::{self, Entropy};
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::ed25519::{KeyRole, SigDomain, Signature, SignedString, SigningKey, VerifyingKey};
use oaiy_crypto::kdf::{self, hkdf_sha256, sha256, sha256_hex, Context, Purpose};
use oaiy_crypto::x25519::{PublicKey, SecretKey};
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

fn vectors() -> serde_json::Value {
    json(DESIGN_VECTORS)
}

fn s(value: &serde_json::Value) -> &str {
    value.as_str().unwrap()
}

fn u32be(n: usize) -> [u8; 4] {
    (n as u32).to_be_bytes()
}

fn lp16(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u16).to_be_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}

#[test]
fn the_wordlist_is_the_official_one() {
    let v = vectors();
    let file = include_bytes!("../src/bip39_english.txt");
    assert_eq!(sha256_hex(file), bip39::WORDLIST_SHA256);
    assert_eq!(bip39::WORDLIST_SHA256, s(&v["out"]["bip39"]["wordlistSha256"]));
    assert_eq!(bip39::WORDLIST_SHA256, s(&v["inputs"]["bip39"]["wordlistSha256Expected"]));
    let list = bip39::wordlist();
    assert_eq!(list.len(), 2048);
    assert!(list.windows(2).all(|w| w[0] < w[1]), "sorted and unique (decode binary-searches it)");
    // "the first four letters are unique" (4.3): a word is resolved by autocomplete on its first four letters
    let mut prefixes: Vec<&str> = list.iter().map(|w| &w[..w.len().min(4)]).collect();
    prefixes.sort_unstable();
    assert!(prefixes.windows(2).all(|w| w[0] != w[1]), "four-letter prefixes are unique");
}

#[test]
fn bip39_six_entropies_and_four_failures() {
    let v = vectors();
    let entropies = v["inputs"]["bip39"]["entropies"].as_array().unwrap();
    let words = v["out"]["bip39"]["words"].as_array().unwrap();
    assert_eq!(entropies.len(), 6);
    for (e, w) in entropies.iter().zip(words) {
        let entropy = Entropy::from_bytes(arr(s(e)));
        assert_eq!(bip39::encode(&entropy).expose(), s(w));
        assert_eq!(bip39::decode(s(w)).unwrap(), entropy);
    }
    let failures = v["inputs"]["bip39"]["failures"].as_array().unwrap();
    let codes = v["out"]["bip39"]["failures"].as_array().unwrap();
    assert_eq!(failures.len(), 4);
    for (phrase, code) in failures.iter().zip(codes) {
        assert_eq!(bip39::decode(s(phrase)).unwrap_err().code(), s(code), "{}", s(phrase));
    }
}

#[test]
fn the_phrase_wrapper_of_section_4_2_1() {
    let v = vectors();
    let inp = &v["inputs"]["phraseWrapper"];
    let out = &v["out"]["phraseWrapper"];
    let entropy = Entropy::from_bytes(arr(s(&inp["entropyHex"])));
    let salt = unhex(s(&inp["saltHex"]));
    let (ops, mem) = (inp["ops"].as_u64().unwrap(), inp["mem"].as_u64().unwrap());
    let umk: Secret<32> = secret(s(&inp["umkHex"]));
    let nonce: [u8; 24] = arr(s(&inp["nonceHex"]));

    let words = bip39::encode(&entropy);
    assert_eq!(words.expose(), s(&out["words"]));
    assert_eq!(words.expose(), "abandon amount liar amount expire adjust cage candy arch gather drum buyer");

    let ikm = argon::argon2id13(entropy.expose(), &salt, ops, mem).unwrap();
    assert_eq!(hex(ikm.expose()), s(&out["ikmHex"]));
    let wk = kdf::derive(&ikm, Purpose::PhraseWrap).unwrap();
    assert_eq!(hex(wk.expose()), s(&out["wkHex"]));

    let p = format!("argon2id13.1|{ops}|{mem}|{}", b64(&salt));
    assert_eq!(p, s(&out["P"]));
    let user = s(&inp["userId"]);
    let wrapper = s(&inp["wrapperId"]);
    let aad = Aad::new(AadDomain::VaultWrap, &[user, wrapper, "recovery-phrase", &sha256_hex(p.as_bytes())]).unwrap();
    assert_eq!(aad.as_bytes(), s(&out["aad"]).as_bytes());

    let mut wrapped = nonce.to_vec();
    wrapped.extend_from_slice(&aead::seal(&wk, aead::Nonce::from_bytes_for_tests(nonce), aad.as_bytes(), umk.expose()).unwrap());
    assert_eq!(b64(&wrapped), s(&out["wrappedUmkB64"]));
    assert_eq!(wrapped.len(), 72);
    assert_eq!(out["wrappedLen"].as_u64().unwrap(), 72);
    assert_eq!(aead::unwrap_key(&wk, &aad, &wrapped).unwrap(), umk);

    // the AAD binds user, wrapper id, kind and parameters: a server cannot move, relabel or re-salt a wrapper
    let other_salt = format!("argon2id13.1|{ops}|{mem}|{}", b64(&[0u8; 16]));
    let wrong = [
        Aad::new(AadDomain::VaultWrap, &["another-user", wrapper, "recovery-phrase", &sha256_hex(p.as_bytes())]).unwrap(),
        Aad::new(AadDomain::VaultWrap, &[user, "vw_ffeeddccbbaa99887766554433221100", "recovery-phrase", &sha256_hex(p.as_bytes())]).unwrap(),
        Aad::new(AadDomain::VaultWrap, &[user, wrapper, "passphrase", &sha256_hex(p.as_bytes())]).unwrap(),
        Aad::new(AadDomain::VaultWrap, &[user, wrapper, "recovery-phrase", &sha256_hex(other_salt.as_bytes())]).unwrap(),
    ];
    for aad in &wrong {
        assert_eq!(aead::unwrap_key(&wk, aad, &wrapped).unwrap_err(), Error::DecryptFailed);
    }

    // the composite: the words straight to the wrap key
    let composite = bip39::phrase_wrap_key(words.expose(), &salt, ops, mem).unwrap();
    assert_eq!(composite, wk);
}

#[test]
fn the_kdf_registry_vectors() {
    let v = vectors();
    let master: Secret<32> = secret(s(&v["inputs"]["kdfRegistry"]["masterHex"]));
    let contexts = v["inputs"]["kdfRegistry"]["contexts"].as_array().unwrap();
    assert_eq!(contexts.len(), 5);
    for c in contexts {
        let context = Context::new(s(c)).unwrap();
        let out = kdf::derive_subkey(&master, 1, &context).unwrap();
        assert_eq!(hex(out.expose()), s(&v["out"]["kdfRegistry"][s(c)]), "{}", s(c));
    }
    for (purpose, name) in [
        (Purpose::KitWrap, "flrecov1"),
        (Purpose::PhraseWrap, "flphras1"),
        (Purpose::BackupRecipient, "flbkrcp1"),
        (Purpose::BackupSigning, "flbksig1"),
        (Purpose::LocalData, "fllocal1"),
    ] {
        assert_eq!(purpose.entry().context.as_str(), name);
        assert_eq!(hex(kdf::derive(&master, purpose).unwrap().expose()), s(&v["out"]["kdfRegistry"][name]));
    }
    // the values in Appendix A
    assert!(s(&v["out"]["kdfRegistry"]["flphras1"]).starts_with("30dfc59e") && s(&v["out"]["kdfRegistry"]["flphras1"]).ends_with("947d"));
}

#[test]
fn the_backup_recipient_secret_and_public_key() {
    let v = vectors();
    let umk: Secret<32> = secret(s(&v["inputs"]["kdfRegistry"]["masterHex"]));
    let bsk = kdf::derive(&umk, Purpose::BackupRecipient).unwrap();
    assert_eq!(hex(bsk.expose()), s(&v["out"]["backupIdentity"]["bskHex"]));
    let bpk = SecretKey::from_bytes(*bsk.expose()).public_key();
    assert_eq!(hex(bpk.as_bytes()), s(&v["out"]["backupIdentity"]["bpkHex"]));
    assert!(s(&v["out"]["backupIdentity"]["bpkHex"]).starts_with("833b8911"));
}

#[test]
fn the_backup_manifest_signature_flbackup_1() {
    let v = vectors();
    let inp = &v["inputs"]["manifestSig"];
    let out = &v["out"]["manifestSig"];
    let umk: Secret<32> = secret(s(&inp["umkHex"]));
    let seed = kdf::derive(&umk, Purpose::BackupSigning).unwrap();
    assert_eq!(hex(seed.expose()), s(&out["sigSeedHex"]));
    let key = SigningKey::from_seed(KeyRole::Backup, &seed);
    assert_eq!(hex(&key.verifying_key().to_bytes()), s(&out["pkHex"]));

    // entriesDigest = SHA-256(for each entry in order: utf8(name) || 0x00 || sha256_bytes || u64le(size))
    let mut material = Vec::new();
    for e in inp["entries"].as_array().unwrap() {
        material.extend_from_slice(s(&e["name"]).as_bytes());
        material.push(0);
        material.extend_from_slice(&unhex(s(&e["sha256"])));
        material.extend_from_slice(&e["size"].as_u64().unwrap().to_le_bytes());
    }
    let digest = sha256_hex(&material);
    assert_eq!(digest, s(&out["entriesDigest"]));

    let message = |seq: u64, kind: &str, keys: u64| {
        SignedString::pipe(SigDomain::Backup, &[s(&inp["installId"]), &seq.to_string(), s(&inp["createdAt"]), kind, &keys.to_string(), &digest])
            .unwrap()
    };
    let msg = message(inp["seq"].as_u64().unwrap(), s(&inp["kind"]), inp["includesKeys"].as_u64().unwrap());
    assert_eq!(msg.as_bytes(), s(&out["msg"]).as_bytes());
    let signature = key.sign(&msg).unwrap();
    assert_eq!(b64(&signature.to_bytes()), s(&out["sigB64"]));
    let verifying = VerifyingKey::from_bytes(&arr(s(&out["pkHex"]))).unwrap();
    verifying.verify(&msg, &signature).unwrap();
    // altered seq, kind or includesKeys is rejected
    assert!(verifying.verify(&message(13, "full", 0), &signature).is_err(), "altered seq");
    assert!(verifying.verify(&message(12, "pre-restore", 0), &signature).is_err(), "altered kind");
    assert!(verifying.verify(&message(12, "full", 1), &signature).is_err(), "altered includesKeys");
    // R-KEY: no other key role signs this domain
    for role in [KeyRole::Vault, KeyRole::Writer] {
        assert_eq!(SigningKey::from_seed(role, &seed).sign(&msg).unwrap_err(), Error::DomainNotAllowed);
    }
}

#[test]
fn signed_vault_operations_and_the_hostile_label() {
    let v = vectors();
    let inp = &v["inputs"]["ops"];
    let out = &v["out"]["ops"];
    let key = SigningKey::from_seed(KeyRole::Vault, &secret(s(&inp["seedHex"])));
    let verifying = key.verifying_key();
    assert_eq!(hex(&verifying.to_bytes()), s(&out["pkHex"]));
    let user = s(&inp["userId"]);

    let op = SignedString::pipe(SigDomain::VaultOp, &[user, "recovery.replace", "7", &sha256_hex(s(&inp["body"]).as_bytes())]).unwrap();
    assert_eq!(op.as_bytes(), s(&out["opString"]).as_bytes());
    let signature = key.sign(&op).unwrap();
    assert_eq!(b64(&signature.to_bytes()), s(&out["sigB64"]));
    verifying.verify(&op, &signature).unwrap();

    // the label enters only as its hash; the label itself cannot be a field of any signed string
    let label = s(&inp["label"]);
    assert!(label.contains('|'));
    let reg_body = format!("vd_{}|browser|{}|-|2026-09-30T09:15:00Z", "0f".repeat(16), sha256_hex(label.as_bytes()));
    assert_eq!(reg_body, s(&inp["regBody"]));
    let reg = SignedString::pipe(SigDomain::VaultOp, &[user, "device.register", "-", &sha256_hex(reg_body.as_bytes())]).unwrap();
    assert_eq!(reg.as_bytes(), s(&out["regOp"]).as_bytes());
    assert_eq!(b64(&key.sign(&reg).unwrap().to_bytes()), s(&out["regSigB64"]));
    for hostile in [label, "Office PC", "x\n2027-01-01", "trailing|", "|leading", ""] {
        assert!(SignedString::pipe(SigDomain::VaultOp, &[user, "device.register", "-", hostile]).is_err(), "{hostile:?}");
    }

    // the head and its state binding
    let st = &inp["state"];
    // salt || u32be(ops) || u32be(mem) || the wrapped UMK, as hex
    let passphrase_wrapper =
        format!("{}{:08x}{:08x}{}", s(&st["kdfSaltHex"]), st["ops"].as_u64().unwrap(), st["mem"].as_u64().unwrap(), s(&st["passWrappedHex"]));
    let mut lines = vec![
        format!("pass|{}", sha256_hex(&unhex(&passphrase_wrapper))),
        format!("bundle|{}", sha256_hex(&unhex(s(&st["bundleHex"])))),
        format!("kit|{}", sha256_hex(&unhex(s(&st["kitHex"])))),
    ];
    let mut wrappers: Vec<&serde_json::Value> = st["wrappers"].as_array().unwrap().iter().collect();
    wrappers.sort_by(|a, b| s(&a["id"]).cmp(s(&b["id"])));
    for w in wrappers {
        lines.push(format!("w|{}|{}|{}|{}", s(&w["id"]), s(&w["kind"]), sha256_hex(s(&w["P"]).as_bytes()), sha256_hex(&unb64(s(&w["wrappedB64"])))));
    }
    let state_hash = sha256_hex(format!("flvault-state:1\n{}", lines.join("\n")).as_bytes());
    assert_eq!(state_hash, s(&out["stateHash"]));
    let head = SignedString::pipe(SigDomain::VaultHead, &[user, &inp["headVersion"].as_u64().unwrap().to_string(), &state_hash]).unwrap();
    assert_eq!(head.as_bytes(), s(&out["head"]).as_bytes());
    let head_signature = key.sign(&head).unwrap();
    assert_eq!(b64(&head_signature.to_bytes()), s(&out["headSigB64"]));
    verifying.verify(&head, &head_signature).unwrap();
    // a rolled-back version does not verify under the newer head's signature
    let rolled_back = SignedString::pipe(SigDomain::VaultHead, &[user, "7", &state_hash]).unwrap();
    assert!(verifying.verify(&rolled_back, &head_signature).is_err());
    // a valid newer head served with older or junk wrappers: the state hash of what was served is not the signed one
    let junk_lines = lines.iter().map(|l| if l.starts_with("w|") { format!("{l}00") } else { l.clone() }).collect::<Vec<_>>();
    let junk_hash = sha256_hex(format!("flvault-state:1\n{}", junk_lines.join("\n")).as_bytes());
    assert_ne!(junk_hash, state_hash);
    let served = SignedString::pipe(SigDomain::VaultHead, &[user, "8", &junk_hash]).unwrap();
    assert!(verifying.verify(&served, &head_signature).is_err());
    // R-KEY: the writer and backup keys do not sign vault strings
    for role in [KeyRole::Writer, KeyRole::Backup] {
        assert_eq!(SigningKey::from_seed(role, &secret(s(&inp["seedHex"]))).sign(&op).unwrap_err(), Error::DomainNotAllowed);
    }
}

#[test]
fn the_archive_signature_covers_every_content_field() {
    let v = vectors();
    let inp = &v["inputs"]["archive"];
    let out = &v["out"]["archive"];
    let key = SigningKey::from_seed(KeyRole::Writer, &secret(s(&inp["seedHex"])));
    assert_eq!(hex(&key.verifying_key().to_bytes()), s(&out["pkHex"]));
    let order: Vec<&str> = inp["order"].as_array().unwrap().iter().map(s).collect();
    let lpcat = |text: &str| -> Vec<u8> {
        let mut bytes = Vec::new();
        for k in &order {
            let value = if *k == "text" { text } else { s(&inp["fields"][*k]) };
            bytes.extend_from_slice(&u32be(value.len()));
            bytes.extend_from_slice(value.as_bytes());
        }
        bytes
    };
    let fields_hash = sha256_hex(&lpcat(s(&inp["fields"]["text"])));
    let message = |hash: &str| SignedString::pipe(SigDomain::Arch, &[s(&inp["F"]), s(&inp["recordId"]), s(&inp["K"]), hash]).unwrap();
    let msg = message(&fields_hash);
    assert_eq!(msg.as_bytes(), s(&out["sigMsg"]).as_bytes());
    let signature = key.sign(&msg).unwrap();
    assert_eq!(b64(&signature.to_bytes()), s(&out["sigB64"]));
    key.verifying_key().verify(&msg, &signature).unwrap();
    // changing only the readable text changes the digest, and the signature no longer verifies
    let forged = sha256_hex(&lpcat(s(&inp["forgedText"])));
    assert_eq!(forged, s(&out["forgedFieldsHash"]));
    assert_ne!(forged, fields_hash);
    assert!(key.verifying_key().verify(&message(&forged), &signature).is_err());
    // length prefixes: ("ab","c") and ("a","bc") are different inputs
    let pair = |a: &str, b: &str| {
        let mut bytes = Vec::new();
        for part in [a, b] {
            bytes.extend_from_slice(&u32be(part.len()));
            bytes.extend_from_slice(part.as_bytes());
        }
        sha256_hex(&bytes)
    };
    assert_ne!(pair("ab", "c"), pair("a", "bc"));
    // the chain link is SHA-256 of the previous rec bytes, and the first one links to 64 zeros
    assert_eq!(sha256_hex(s(&inp["fields"]["rec"]).as_bytes()), s(&out["nextPrev"]));
    assert!(s(&inp["fields"]["rec"]).contains(&"0".repeat(64)));
    // R-KEY: the vault key does not sign archive records
    let vault = SigningKey::from_seed(KeyRole::Vault, &secret(s(&inp["seedHex"])));
    assert_eq!(vault.sign(&msg).unwrap_err(), Error::DomainNotAllowed);
}

#[test]
fn the_key_transfer_ceremony_x25519_hkdf_and_the_package() {
    let v = vectors();
    let inp = &v["inputs"]["ceremony"];
    let out = &v["out"]["ceremony"];
    let r = SecretKey::from_bytes(arr(s(&inp["rSkHex"])));
    let sender = SecretKey::from_bytes(arr(s(&inp["sSkHex"])));
    assert_eq!(hex(r.public_key().as_bytes()), s(&out["rPkHex"]));
    assert_eq!(hex(sender.public_key().as_bytes()), s(&out["sPkHex"]));
    let (r_pk, s_pk) = (r.public_key(), sender.public_key());
    let dh = sender.diffie_hellman(&r_pk).unwrap();
    assert_eq!(hex(dh.expose()), s(&out["DHHex"]));
    assert_eq!(r.diffie_hellman(&s_pk).unwrap(), dh, "both sides agree");

    let code = s(&inp["code"]).replace('-', "");
    let mailbox = format!("kt_{}", &sha256_hex(format!("oaiy-kt:1|mailbox|{code}").as_bytes())[..32]);
    assert_eq!(mailbox, s(&out["mailbox"]));
    let nr: [u8; 16] = arr(s(&inp["nrHex"]));
    let ns: [u8; 16] = arr(s(&inp["nsHex"]));
    let xr = s(&inp["xr"]).as_bytes();
    let mut commit = b"oaiy-kt:1|commit|".to_vec();
    commit.extend_from_slice(r_pk.as_bytes());
    commit.extend_from_slice(&nr);
    commit.extend_from_slice(&lp16(xr));
    assert_eq!(sha256_hex(&commit), s(&out["cr"]));

    let transcript = |initiator_pk: &PublicKey| -> [u8; 32] {
        let mut t = b"oaiy-kt:1|transcript".to_vec();
        t.push(0);
        for part in [
            s(&inp["purpose"]).as_bytes(),
            mailbox.as_bytes(),
            initiator_pk.as_bytes().as_slice(),
            &nr,
            s_pk.as_bytes().as_slice(),
            &ns,
            s(&inp["labelR"]).as_bytes(),
            s(&inp["labelS"]).as_bytes(),
            xr,
            s(&inp["ptype"]).as_bytes(),
        ] {
            t.extend_from_slice(&lp16(part));
        }
        sha256(&t)
    };
    let th = transcript(&r_pk);
    assert_eq!(hex(&th), s(&out["THHex"]));
    let derive = |dh: &Secret<32>, th: &[u8; 32]| -> (Secret<32>, Secret<32>, String) {
        let s2r: Secret<32> = kdf::hkdf_sha256_secret(dh.expose(), Some(th), b"oaiy-kt:1|s2r").unwrap();
        let r2s: Secret<32> = kdf::hkdf_sha256_secret(dh.expose(), Some(th), b"oaiy-kt:1|r2s").unwrap();
        let mut sas = [0u8; 8];
        hkdf_sha256(dh.expose(), Some(th), b"oaiy-kt:1|sas", &mut sas).unwrap();
        let digits = format!("{:08}", u64::from_be_bytes(sas) % 100_000_000);
        (s2r, r2s, format!("{} {}", &digits[..4], &digits[4..]))
    };
    let (k_s2r, k_r2s, sas) = derive(&dh, &th);
    assert_eq!(hex(k_s2r.expose()), s(&out["KS2RHex"]));
    assert_eq!(hex(k_r2s.expose()), s(&out["KR2SHex"]));
    assert_eq!(sas, s(&out["sas"]));

    let payload = s(&inp["payload"]).as_bytes();
    let nonce: [u8; 24] = arr(s(&inp["pnonceHex"]));
    let package = aead::seal(&k_s2r, aead::Nonce::from_bytes_for_tests(nonce), &th, payload).unwrap();
    assert_eq!(hex(&package), s(&out["packageHex"]));
    assert_eq!(aead::open(&k_s2r, &nonce, &th, &package).unwrap().expose(), payload);

    // a man in the middle with his own key toward the sender: another string on the screen, and the package does not open under his key
    let m = SecretKey::from_bytes(arr(s(&inp["mSkHex"])));
    let m_pk = m.public_key();
    let th_m = transcript(&m_pk);
    let dh_m = sender.diffie_hellman(&m_pk).unwrap();
    let (k_m, _, sas_m) = derive(&dh_m, &th_m);
    assert_eq!(sas_m, s(&out["sasMitm"]));
    assert_ne!(sas_m, sas);
    assert!(aead::open(&k_m, &nonce, &th_m, &package).is_err());
    assert!(aead::open(&k_s2r, &nonce, &th_m, &package).is_err(), "the transcript is the AAD");
}

#[test]
fn the_ed25519_strictness_probe_of_design_finding_a5() {
    let v = vectors();
    let probe = &v["inputs"]["edProbe"];
    assert_eq!(v["out"]["edProbe"]["libsodiumAccepts"], false);
    // R = identity, S = 0 under a public key of small order verifies for every message under the plain RFC 8032 equation, and
    // Python's OpenSSL accepts it. Here the key cannot be built at all.
    let pk: [u8; 32] = arr(s(&probe["pkHex"]));
    assert_eq!(VerifyingKey::from_bytes(&pk).unwrap_err(), Error::SmallOrderKey);
    let sig = Signature::from_slice(&unhex(s(&probe["sigHex"]))).unwrap();
    assert_eq!(sig.to_bytes()[0], 1);
    // and there is no other way in: a key of any role verifies nothing under it
    let honest = SigningKey::from_seed(KeyRole::Hazmat, &Secret::new([9u8; 32])).verifying_key();
    assert!(honest.verify_raw(s(&probe["msg"]).as_bytes(), &sig).is_err());
}
