//! FormLogic's committed known-answer vectors (`docs/contracts/e2ee-envelope-vectors.json`, `e2ee-sealed-js.json`,
//! `e2ee-sealed-php.json`), reproduced byte for byte by this crate: kdf x2, XChaCha20-Poly1305 x2 with the bad-AAD matrix,
//! Argon2id (vector 2, the emoji at 64 MiB; vector 1 is below this crate's floor and is in `src/argon.rs`), FLRK1 x3, the
//! sealed boxes written by JavaScript and by PHP, the two envelope AADs, and the whole `__flenc:1` envelope, with the
//! malformed corpus of `vectors.test.ts`.

mod common;

use common::*;
use oaiy_crypto::aead;
use oaiy_crypto::argon;
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::kdf::{self, sha256_hex, Context};
use oaiy_crypto::kit::RecoveryKit;
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::SecretKey;
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

#[test]
fn kdf_vectors_1_and_2_are_reproduced() {
    let v = json(FL_VECTORS);
    let cases = v["kdf"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let master: Secret<32> = secret(case["key_hex"].as_str().unwrap());
        let id = case["subkey_id"].as_u64().unwrap();
        let context = Context::new(case["context"].as_str().unwrap()).unwrap();
        let out = kdf::derive_subkey(&master, id, &context).unwrap();
        assert_eq!(hex(out.expose()), case["out_hex"].as_str().unwrap(), "subkey id {id}");
    }
    // vector 1 is the registry's `flrecov1` with subkey id 1
    let master: Secret<32> = secret(cases[0]["key_hex"].as_str().unwrap());
    let typed = kdf::derive(&master, kdf::Purpose::KitWrap).unwrap();
    assert_eq!(hex(typed.expose()), cases[0]["out_hex"].as_str().unwrap());
}

#[test]
fn xchacha_vectors_1_and_2_and_the_bad_aad_matrix() {
    let v = json(FL_VECTORS);
    let cases = v["xchacha"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let key: Secret<32> = secret(case["key_hex"].as_str().unwrap());
        let nonce: [u8; 24] = arr(case["nonce_hex"].as_str().unwrap());
        let aad = case["aad"].as_str().unwrap().as_bytes();
        let plaintext = unb64(case["plaintext_b64"].as_str().unwrap());
        let ct = unb64(case["ct_b64"].as_str().unwrap());
        assert_eq!(aead::seal(&key, &nonce, aad, &plaintext).unwrap(), ct, "ciphertext || tag");
        assert_eq!(aead::open(&key, &nonce, aad, &ct).unwrap().expose(), plaintext.as_slice());
        let bad = case["badAads"].as_array().unwrap();
        assert!(bad.len() >= 2);
        for bad_aad in bad {
            let bad_aad = bad_aad.as_str().unwrap().as_bytes();
            assert_eq!(aead::open(&key, &nonce, bad_aad, &ct).unwrap_err(), Error::DecryptFailed, "bad AAD must fail: {bad_aad:?}");
        }
    }
}

#[test]
fn the_vectors_aads_are_what_the_aad_builder_makes() {
    let v = json(FL_VECTORS);
    let cases = v["xchacha"].as_array().unwrap();
    let hash = "b".repeat(64);
    let aad1 = Aad::new(AadDomain::Enc, &["form-a", "7d444840-9dc0-41a2-8da8-ff8cb9fca735", "1", "fik_v1", "1", "1", &hash, "-"]).unwrap();
    assert_eq!(aad1.as_bytes(), cases[0]["aad"].as_str().unwrap().as_bytes());
    let aad2 = Aad::new(AadDomain::VaultUmk, &["user-vector-1"]).unwrap();
    assert_eq!(aad2.as_bytes(), cases[1]["aad"].as_str().unwrap().as_bytes());
    // the version-flipped AAD of the matrix is not one any builder can make
    assert!(cases[1]["badAads"].as_array().unwrap().iter().any(|a| a.as_str().unwrap().starts_with("flvault-umk:2|")));
}

#[test]
fn argon2id_vector_2_the_emoji_passphrase_at_64_mib() {
    let v = json(FL_VECTORS);
    let cases = v["argon2id"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let case = &cases[1];
    assert_eq!(case["memlimit"].as_u64().unwrap(), 64 * 1024 * 1024);
    let password = case["password"].as_str().unwrap();
    assert!(password.ends_with('\u{1f434}'), "the passphrase ends in the horse emoji");
    let salt = unhex(case["salt_hex"].as_str().unwrap());
    let out = argon::argon2id13(password.as_bytes(), &salt, case["opslimit"].as_u64().unwrap(), case["memlimit"].as_u64().unwrap()).unwrap();
    assert_eq!(hex(out.expose()), case["out_hex"].as_str().unwrap());
    // vector 1 (2 passes, 8 MiB) is below the floor of the design's bounds
    let low = &cases[0];
    let refused = argon::argon2id13(
        low["password"].as_str().unwrap().as_bytes(),
        &unhex(low["salt_hex"].as_str().unwrap()),
        low["opslimit"].as_u64().unwrap(),
        low["memlimit"].as_u64().unwrap(),
    );
    assert_eq!(refused.unwrap_err(), Error::KdfParamsOutOfRange);
}

#[test]
fn flrk1_vectors_encode_and_decode() {
    let v = json(FL_VECTORS);
    let cases = v["recovery"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let key: Secret<32> = secret(case["key_hex"].as_str().unwrap());
        let display = case["display"].as_str().unwrap();
        let kit = RecoveryKit::from_bytes(key);
        assert_eq!(kit.encode().expose(), display);
        let back = RecoveryKit::decode(display).unwrap();
        assert_eq!(hex(back.key().expose()), case["key_hex"].as_str().unwrap());
    }
    // what JavaScript accepts around the code: case, spaces, hyphens anywhere
    let display = cases[2]["display"].as_str().unwrap();
    let lower = display.to_lowercase().replace('-', " ");
    assert!(RecoveryKit::decode(&format!("  {lower}\n")).is_ok());
    assert!(RecoveryKit::decode(&display.replace('-', "")).is_ok());
}

#[test]
fn sealed_boxes_written_by_javascript_open() {
    let file = json(FL_SEALED_JS);
    let recipient = SecretKey::from_libsodium_seed(&secret::<32>(file["meta"]["recipient_seed_hex"].as_str().unwrap()));
    assert_eq!(b64(recipient.public_key().as_bytes()), file["recipient"]["publicKey_b64"].as_str().unwrap());
    let items = file["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    for item in items {
        let opened = sealbox::open(&recipient, &unb64(item["sealed_b64"].as_str().unwrap())).unwrap();
        assert_eq!(opened.expose(), unb64(item["plaintext_b64"].as_str().unwrap()).as_slice(), "{}", item["label"]);
    }
    assert!(items[2]["plaintext_b64"].as_str().unwrap().is_empty(), "the empty message round-trips");
}

#[test]
fn sealed_boxes_written_by_php_open() {
    let file = json(FL_SEALED_PHP);
    let recipient = SecretKey::from_libsodium_seed(&secret::<32>(file["meta"]["recipient_seed_hex"].as_str().unwrap()));
    assert_eq!(b64(recipient.public_key().as_bytes()), file["recipient"]["publicKey_b64"].as_str().unwrap());
    let items = file["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    for item in items {
        let opened = sealbox::open(&recipient, &unb64(item["sealed_b64"].as_str().unwrap())).unwrap();
        assert_eq!(opened.expose(), unb64(item["plaintext_b64"].as_str().unwrap()).as_slice(), "{}", item["label"]);
    }
}

#[test]
fn what_this_crate_seals_the_same_recipient_opens_and_a_stranger_does_not() {
    let file = json(FL_SEALED_JS);
    let recipient = SecretKey::from_libsodium_seed(&secret::<32>(file["meta"]["recipient_seed_hex"].as_str().unwrap()));
    let sealed = sealbox::seal(&recipient.public_key(), b"sealed here, opened here").unwrap();
    assert_eq!(sealed.len(), b"sealed here, opened here".len() + sealbox::SEAL_OVERHEAD);
    assert_eq!(sealbox::open(&recipient, &sealed).unwrap().expose(), b"sealed here, opened here");
    let stranger = SecretKey::generate().unwrap();
    assert_eq!(sealbox::open(&stranger, &sealed).unwrap_err(), Error::DecryptFailed);
}

/// The frozen `__flenc:1` AAD, as `buildAad` in `envelope.ts` writes it: `flenc:1|formId|recordId|rev|keyId|epoch|schemaVersion|schemaHash|attHash`.
struct EnvelopeParams {
    form_id: String,
    record_id: String,
    rev: u64,
    key_id: String,
    epoch: u64,
    schema_version: u64,
    schema_hash: String,
    attachments: Vec<String>,
}

fn attachments_hash(ids: &[String]) -> String {
    if ids.is_empty() {
        return "-".into();
    }
    let mut sorted = ids.to_vec();
    sorted.sort();
    sha256_hex(sorted.join(",").as_bytes())
}

fn build_aad(p: &EnvelopeParams) -> Result<Aad, Error> {
    let att = attachments_hash(&p.attachments);
    Aad::new(
        AadDomain::Enc,
        &[&p.form_id, &p.record_id, &p.rev.to_string(), &p.key_id, &p.epoch.to_string(), &p.schema_version.to_string(), &p.schema_hash, &att],
    )
}

fn params_of(value: &serde_json::Value) -> EnvelopeParams {
    EnvelopeParams {
        // the envelope object has no formId (it is bound through the AAD only); the caller sets it
        form_id: value["formId"].as_str().unwrap_or_default().into(),
        record_id: value["recordId"].as_str().unwrap().into(),
        rev: value["rev"].as_u64().unwrap(),
        key_id: value["keyId"].as_str().unwrap().into(),
        epoch: value["epoch"].as_u64().unwrap(),
        schema_version: value["schemaVersion"].as_u64().unwrap(),
        schema_hash: value["schemaHash"].as_str().unwrap().into(),
        attachments: value["attachments"].as_array().map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect()).unwrap_or_default(),
    }
}

#[test]
fn the_two_envelope_aads_are_reproduced() {
    let v = json(FL_VECTORS);
    let cases = v["envelopeAad"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let aad = build_aad(&params_of(&case["params"])).unwrap();
        assert_eq!(aad.as_bytes(), case["aad"].as_str().unwrap().as_bytes());
    }
    // the attachment hash is over the SORTED ids: ["fil_b2", "fil_a1"] gives the hash of "fil_a1,fil_b2"
    assert_eq!(attachments_hash(&["fil_b2".into(), "fil_a1".into()]), sha256_hex(b"fil_a1,fil_b2"));
    assert_eq!(attachments_hash(&[]), "-");
}

struct Envelope {
    params: EnvelopeParams,
    wrapped_dek: Vec<u8>,
    nonce: [u8; 24],
    ct: Vec<u8>,
}

fn envelope_of(fixture: &serde_json::Value) -> Envelope {
    let env = &fixture["envelope"];
    let mut params = params_of(env);
    params.form_id = fixture["formId"].as_str().unwrap().into();
    Envelope {
        params,
        wrapped_dek: unb64(env["wrappedDek"].as_str().unwrap()),
        nonce: unb64(env["nonce"].as_str().unwrap()).try_into().expect("a 24-byte nonce"),
        ct: unb64(env["ct"].as_str().unwrap()),
    }
}

/// `openEnvelope`: unwrap the DEK from the sealed box, rebuild the AAD, decrypt.
fn open_envelope(env: &Envelope, form_id: &str, ingestion: &SecretKey) -> Result<Vec<u8>, Error> {
    let aad = build_aad(&EnvelopeParams { form_id: form_id.into(), ..clone_params(&env.params) })?;
    let dek = sealbox::open(ingestion, &env.wrapped_dek)?;
    let dek: Secret<32> = Secret::from_slice(dek.expose()).map_err(|_| Error::DecryptFailed)?;
    Ok(aead::open(&dek, &env.nonce, aad.as_bytes(), &env.ct)?.expose().to_vec())
}

fn clone_params(p: &EnvelopeParams) -> EnvelopeParams {
    EnvelopeParams {
        form_id: p.form_id.clone(),
        record_id: p.record_id.clone(),
        rev: p.rev,
        key_id: p.key_id.clone(),
        epoch: p.epoch,
        schema_version: p.schema_version,
        schema_hash: p.schema_hash.clone(),
        attachments: p.attachments.clone(),
    }
}

#[test]
fn the_full_envelope_opens_and_the_malformed_corpus_does_not() {
    let file = json(FL_SEALED_JS);
    let fixture = &file["envelope"];
    let ingestion = SecretKey::from_libsodium_seed(&secret::<32>(fixture["ingestion_seed_hex"].as_str().unwrap()));
    assert_eq!(b64(ingestion.public_key().as_bytes()), fixture["ingestionPublicKey_b64"].as_str().unwrap());
    let base = envelope_of(fixture);
    assert_eq!(base.wrapped_dek.len(), 80, "a sealed 32-byte DEK is 80 bytes");

    let plaintext = open_envelope(&base, &base.params.form_id, &ingestion).unwrap();
    let inner: serde_json::Value = serde_json::from_slice(&plaintext).unwrap();
    assert_eq!(inner, fixture["expectedInner"]);

    // the corpus of vectors.test.ts, each of which changes the AAD or the bytes
    let form = base.params.form_id.clone();
    let with = |mutate: &dyn Fn(&mut Envelope)| {
        let mut env = Envelope { params: clone_params(&base.params), wrapped_dek: base.wrapped_dek.clone(), nonce: base.nonce, ct: base.ct.clone() };
        mutate(&mut env);
        open_envelope(&env, &form, &ingestion)
    };
    assert!(with(&|e| e.params.rev += 1).is_err(), "rev flip");
    assert!(with(&|e| e.params.epoch = 2).is_err(), "epoch flip");
    assert!(with(&|e| e.params.schema_version = 9).is_err(), "schema version flip");
    assert!(with(&|e| e.params.schema_hash = "b".repeat(64)).is_err(), "schema hash flip");
    assert!(with(&|e| e.params.attachments = vec!["fil_evil1".into()]).is_err(), "attachment injection");
    assert!(with(&|e| e.params.key_id = "fik_other".into()).is_err(), "key id flip");
    assert!(with(&|e| e.params.record_id = "7d444840-9dc0-41a2-8da8-ff8cb9fca736".into()).is_err(), "record id flip");
    assert!(open_envelope(&base, "other-form", &ingestion).is_err(), "cross-form replay");
    assert!(with(&|e| e.ct.truncate(e.ct.len() - 4)).is_err(), "truncated ct");
    assert!(with(&|e| e.ct.extend_from_slice(&[0, 0, 0, 0])).is_err(), "extended ct");
    let last = base.ct.len() - 1;
    assert!(with(&|e| e.ct[last] ^= 1).is_err(), "corrupted Poly1305 tag");
    assert!(with(&|e| e.ct[0] ^= 1).is_err(), "corrupted first ciphertext byte");
    assert!(with(&|e| e.nonce[0] ^= 1).is_err(), "nonce flip");
    assert!(with(&|e| e.wrapped_dek[0] ^= 1).is_err(), "wrapped DEK flip (ephemeral key)");
    assert!(with(&|e| e.wrapped_dek[79] ^= 1).is_err(), "wrapped DEK flip (tag)");
    assert!(with(&|e| e.wrapped_dek.truncate(79)).is_err(), "wrapped DEK truncated");
    assert!(with(&|e| e.wrapped_dek.push(0)).is_err(), "wrapped DEK extended");
    // another form's ingestion key
    let stranger = SecretKey::generate().unwrap();
    assert!(open_envelope(&base, &form, &stranger).is_err(), "wrong ingestion key");
}
