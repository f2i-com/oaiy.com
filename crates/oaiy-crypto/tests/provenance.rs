//! The known-answer files are what was committed. Every file under `tests/vectors` is pinned by its SHA-256, so a change to one (a regenerated oracle, an
//! edited fixture, a checkout that converted a line ending) fails here and has to be made on purpose. The folder is marked `-text` in `.gitattributes`.
//!
//! FormLogic's three files are byte-for-byte copies of `docs/contracts/` at FormLogic commit `81860c8a937b9e2a2ed7e6f71c5d7a9dd8cad1fc` (last changed in
//! `83a7eec3`), and their git blob ids are the same as FormLogic's: `e2ee-envelope-vectors.json` `609800802a0dc8f9ba682b7c1546666910b86735`,
//! `e2ee-sealed-js.json` `93fb73ef371f7fd7b6bf4183ed3fbc523921955d`, `e2ee-sealed-php.json` `79ea626d2384b5940febfcb9cceebde1aaf9f767`
//! (`git hash-object` of each file here prints those). `vault-work/vectors.json` is the design's file, SHA-256 `7057862f...`.

mod common;

use oaiy_crypto::kdf::sha256_hex;

const FILES: &[(&str, &str, &[u8])] = &[
    (
        "formlogic/e2ee-envelope-vectors.json",
        "0dddd78a8a0568b3c3160693a9260e2f391cf3dcfe9030ba1e1a320ebcc63860",
        include_bytes!("vectors/formlogic/e2ee-envelope-vectors.json"),
    ),
    (
        "formlogic/e2ee-sealed-js.json",
        "c175b69025e1b5623fffc96269dbbfa8d3dc639dff7661a53093cf891acad8f2",
        include_bytes!("vectors/formlogic/e2ee-sealed-js.json"),
    ),
    (
        "formlogic/e2ee-sealed-php.json",
        "771c86cf3d36a515cbbc286d1257d9614b14a6daab245e38a2fd11b1cf91f45f",
        include_bytes!("vectors/formlogic/e2ee-sealed-php.json"),
    ),
    (
        "vault-work/vectors.json",
        "7057862fbaa02012953af46eab8fb6d878b5696742e40c95956c767b100f70d3",
        include_bytes!("vectors/vault-work/vectors.json"),
    ),
    ("public-vectors.json", "46023622f158077c5ce0b049215e34cf855be2159e4388c4406cf96c675fd60f", include_bytes!("vectors/public-vectors.json")),
    ("libsodium-oracle.json", "a683d250cc05128cb769153d909f3cb7f0349a115bee607669fbb5fc628c304d", include_bytes!("vectors/libsodium-oracle.json")),
    (
        "scripts/extract_public.py",
        "58950f8861a1d11666c60817cd100dd77c58d3e8148921a7fa6aa59ae23f1fc4",
        include_bytes!("vectors/scripts/extract_public.py"),
    ),
    ("text-corpus.json", "e9e80980ae8bbe44cc8307f45df3f3a97407ef684fd358e09bcac4d045beb5ee", include_bytes!("vectors/text-corpus.json")),
    (
        "scripts/text_corpora.mjs",
        "9f12c26ce0263d0459ca4dcd508eb64f7e9a36cf18d2aecf6f9dae817ce1fa33",
        include_bytes!("vectors/scripts/text_corpora.mjs"),
    ),
    ("scripts/gen_corpus.py", "185279f152bfb2c55e9ca5de1da99b2b4aa07871686953b6f2f3bf394a4ce0dc", include_bytes!("vectors/scripts/gen_corpus.py")),
    ("scripts/oracle.mjs", "383c3a827ed4909857e0a10f9d2898e2f66bf242a7222ea6f68c1c6b5493fe37", include_bytes!("vectors/scripts/oracle.mjs")),
    (
        "scripts/oracle_check.py",
        "7146c444be273f75fb670c498502e9519ebda41d65102fbbbf3e6b2db9b34c7b",
        include_bytes!("vectors/scripts/oracle_check.py"),
    ),
];

#[test]
fn every_vector_file_is_the_committed_bytes() {
    for (name, expected, bytes) in FILES {
        assert_eq!(sha256_hex(bytes), *expected, "{name} is not the file that was committed");
        assert!(!bytes.contains(&b'\r'), "{name} has a carriage return: a checkout converted its line endings (see .gitattributes)");
    }
}

#[test]
fn every_json_file_parses_and_names_what_it_is() {
    for (name, _, bytes) in FILES.iter().filter(|(n, _, _)| n.ends_with(".json")) {
        let text = std::str::from_utf8(bytes).unwrap_or_else(|_| panic!("{name} is not UTF-8"));
        let value = common::json(text);
        assert!(value.is_object(), "{name}");
    }
    // the emoji of Argon2id vector 2 is in the file as UTF-8, not as an escape or a replacement
    let vectors = std::str::from_utf8(FILES[0].2).unwrap();
    assert!(vectors.contains("correct horse battery staple \u{1f434}"));
}

#[test]
fn the_oracle_says_which_libsodium_and_which_second_implementation_made_it() {
    let oracle = common::json(common::ORACLE);
    let meta = &oracle["meta"];
    assert!(meta["libsodium"].as_str().unwrap().starts_with("1."));
    assert!(meta["node"].as_str().unwrap().starts_with('v'));
    assert!(meta["cross_checked_by"].as_str().unwrap().contains("Python cryptography"));
    assert!(meta["checks_passed"].as_u64().unwrap() >= 400);
    let public = common::json(common::PUBLIC_VECTORS);
    assert!(public["meta"]["sources"].as_object().unwrap().len() >= 7);
}
