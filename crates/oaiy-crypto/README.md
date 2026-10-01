# oaiy-crypto

The cryptographic primitives of the OAIY vault: one crate for every key the vault, the encrypted backups, the relay and the apps handle, so that the
same low-order checks, the same strict signature verification, the same zeroization and the same constant-time comparison serve all of them.

This is work package **V-01** of `design/vault.final.md` (section 4.1 "Primitives, registries, rules", 4.3 "The recovery phrase", and the
"Vectors and scripts" of section 11). A library: nothing in the desktop links it yet, and consumers (backup, relay, apps) can code against it.
It is a workspace member and not a default member; `cargo test -p oaiy-crypto` runs it.

```text
cargo test --locked -p oaiy-crypto                           # 101 tests (and 1 ignored), about a minute unoptimised
cargo test --release -p oaiy-crypto -- --ignored             # the one-million-iteration X25519 vector of RFC 7748 (passed again on the final code: 46.7 s)
```

Run on Windows 11 (MSVC, rustc 1.92.0): 101 passed, 1 ignored, none failed (100 and 1 before the third review; 98 and 1 before the second). Run on Linux (WSL2 Ubuntu 24.04, rustc 1.94.0, from a copy in the WSL file system, WSL stopped afterwards): the same 101 pass. The dead-stack probe (`zeroize_stack`) is also run with `--release` and with `--profile vault-probe`, on both.
macOS: compile-checked only (`cargo check --target aarch64-apple-darwin --all-targets`, a step of the `vault-linux` CI lane). `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` (the crate's `rustfmt.toml`)
are clean on Windows, Linux and `x86_64-unknown-linux-musl`. The independent review of `5859fd8d` (differential tests against libsodium in Node and PHP: 400 kdf, 7,774 xchacha, 3,642 sealed boxes,
31,801 Ed25519 verdicts, 6,542 X25519, Argon2id, 510,000 fuzz inputs, all identical or refused with no panic) found the primitives sound and zeroization and the keystore not; the fixes are the commits after it.

The `test-vectors` feature (off by default) is the deterministic part of the API that known-answer tests need and production code must not have: `aead::Nonce::from_bytes_for_tests`, `kdf::derive_subkey` and
`kdf::derive_subkey_into`. This crate's own tests turn it on through a dev-dependency on the crate itself; a crate that depends on `oaiy-crypto` does not get it (Cargo resolver 2), and the `vault-linux` CI lane
checks that the documentation of a build without dev-dependencies has neither function.

## What is in it

| Module | What | Design |
|---|---|---|
| `kdf` | `crypto_kdf` (BLAKE2b, `salt = LE64(id)`, `personal = ctx`) and the append-only registry of contexts (the only derivation production code has is `derive(master, Purpose)`); HKDF-SHA256 (RFC 5869), HMAC-SHA256, SHA-256 | 4.1.1 `kdf`, 4.1.2 |
| `aead` | XChaCha20-Poly1305 (`seal` with a `Nonce` that only the random generator can make, `open`) and the 72-byte `wrap` format `nonce24 \|\| ct \|\| tag16` with a checked AAD | 4.1.1 `xaead`, `wrap` |
| `sealbox` | libsodium's `crypto_box_seal`, byte for byte | 4.1.1 `sealbox` |
| `ed25519` | detached signatures, **strict verification only**, keys with a role, signed strings built from a domain registry | 4.1.1, 4.1.3, 4.1.4 |
| `x25519` | RFC 7748 Diffie-Hellman, the 14 low-order encodings refused twice | 4.1.1 `x25519` |
| `argon` | Argon2id v1.3, one lane, 32 bytes, bounds checked before anything is allocated, blocks wiped | 4.1.1 `argon2id13`, 4.1.2 |
| `bip39` | the twelve-word phrase, checksum verified before any key derivation | 4.1.1 `bip39`, 4.3 |
| `kit` | FormLogic's FLRK1 recovery-kit code | FormLogic `vault.ts` |
| `canon` | canonical AAD and signed strings, the prefix-free domain registry | 4.1.3 rules 1 to 3, 4.1.5 |
| `zeroize` | `Secret<N>` (with `zeroed`, `expose_mut` and `fill_random`, for keys written in place), `SecretVec`, `SecretString`, `ct_eq`, and the stack scrub that the derivations call | 4.1.1, "constant-time comparisons" |
| `text` (private) | the white space of JavaScript's `\s`, shared by the phrase and the kit decoders, and the zeroizing growth of a secret `String` | 4.3, FormLogic `vault.ts` |

Every secret is one of those types (or a dalek type that zeroizes on drop and is wrapped by one of ours): no `Clone`, no `Copy`, no `Display`, a
`Debug` that prints nothing of the value, bytes overwritten when dropped, `==` is `subtle`'s `ct_eq`. A refused input is an `Error` whose message
names what was refused and never a byte of what was given; every failed decryption is the same `DecryptFailed`.

## The rules of section 4.1, and the test that carries each

| Rule | Test |
|---|---|
| 4.1.1 `kdf`: BLAKE2b keyed, salt `LE64(id)`, personal `ctx` | `formlogic_vectors::kdf_vectors_1_and_2_are_reproduced`, `oracle_vectors::crypto_kdf_133_vectors_at_16_32_and_64_bytes`, `rules_4_1::rule_4_1_5_hex_is_lowercase_and_the_kdf_salt_is_a_little_endian_u64` |
| 4.1.1 `xaead`: XChaCha20-Poly1305-IETF | `public_vectors::xchacha20_poly1305_draft_irtf_cfrg_xchacha_03_appendix_a_3_1`, `formlogic_vectors::xchacha_vectors_1_and_2_and_the_bad_aad_matrix`, `oracle_vectors::xchacha20_poly1305_30_vectors_with_empty_and_long_inputs` |
| 4.1.1 `wrap`: 72 bytes for a 32-byte key; a wrapper holds a key, never nothing | `design_vectors::the_phrase_wrapper_of_section_4_2_1`, `negative::a_wrapped_key_refuses_every_single_bit_flip_in_the_nonce_the_ciphertext_and_the_tag`, `negative::wrap_never_wraps_nothing_and_unwrap_never_opens_a_blob_of_40_bytes_or_fewer` |
| 4.1.1 `sealbox`: FormLogic's JS and PHP fixtures open; what is sealed here opens in libsodium | `formlogic_vectors::sealed_boxes_written_by_javascript_open`, `..._by_php_open`, `sealbox::tests::the_deterministic_seal_equals_libsodiums_construction` |
| 4.1.1 `sealbox` / `x25519`: an all-zero or small-order result is refused | `oracle_vectors::a_forged_sealed_box_under_a_small_order_ephemeral_key_is_refused`, `oracle_vectors::the_low_order_table_is_the_set_libsodium_refuses`, `x25519::tests::diffie_hellman_refuses_an_all_zero_result_even_for_a_key_that_bypassed_the_constructor` |
| 4.1.1 `ed25519`: `verify_strict` only; small-order, non-canonical S, non-canonical points | `oracle_vectors::ed25519_negative_corpus_gives_libsodiums_verdict_on_all_80_cases`, `..::a_small_order_r_under_a_key_of_mixed_order_is_what_strict_verification_refuses`, `..::every_point_of_small_order_in_every_encoding_is_refused_as_a_public_key`, `..::a_non_canonical_encoding_of_a_good_key_is_refused_and_the_canonical_one_is_not`, `negative::a_signature_with_s_not_below_the_group_order_is_refused_for_fifty_random_keys`, `design_vectors::the_ed25519_strictness_probe_of_design_finding_a5` |
| 4.1.1 `argon2id13`: v1.3, p=1, 32 bytes, salt 16 | `formlogic_vectors::argon2id_vector_2_the_emoji_passphrase_at_64_mib`, `argon::tests::rfc_9106_argon2id_vector`, `oracle_vectors::argon2id_across_the_bounds_equals_libsodium_and_openssl` |
| 4.1.1 `bip39`: English list, its SHA-256, 16-byte entropy, 4 check bits | `design_vectors::the_wordlist_is_the_official_one`, `public_vectors::bip39_official_128_bit_vectors`, `phrase_fuzz` |
| 4.1.2 the registry of KDF contexts: append-only, no duplicates, no reuse | `rules_4_1::rule_4_1_2_the_kdf_context_registry_is_the_design_table_and_has_no_duplicates` |
| 4.1.2 Argon2id bounds (3 to 10 operations, 64 to 256 MiB, salt 16), checked **before** any derivation, nothing allocated (attack C1) | `argon_ceiling::c1_a_hostile_argon2_cost_is_refused_before_anything_is_allocated` (a counting allocator), `argon::tests::out_of_range_parameters_never_reach_the_derivation` |
| 4.1.3 rule 1: the signature domains are prefix-free | `rules_4_1::rule_4_1_3_1_the_domains_are_prefix_free_and_the_enums_match_the_registry`, `..::rule_4_1_3_1_the_prefix_check_catches_what_it_is_for` (attack A1) |
| 4.1.3 rule 2: no operation accepts a caller-chosen domain | `rules_4_1::rule_4_1_3_2_no_operation_accepts_a_caller_chosen_domain` |
| 4.1.3 rule 3: no free text in a signed string; `\|` or LF in a field is refused (attack A2) | `rules_4_1::rule_4_1_3_3_a_pipe_or_lf_or_any_free_text_in_a_signed_field_is_refused`, `design_vectors::signed_vault_operations_and_the_hostile_label` |
| 4.1.3 rule 4: strict verification everywhere, small-order keys refused at pin time (attack A5) | the `oracle_vectors` tests above |
| 4.1.4 R-KEY: a key is used in exactly one protocol | `rules_4_1::rule_4_1_4_r_key_a_key_has_one_role_and_signs_only_its_domains` |
| 4.1.5 encodings: lowercase hex, `\|`-delimited ASCII, `u64le` in the KDF salt | `rules_4_1::rule_4_1_5_hex_is_lowercase_and_the_kdf_salt_is_a_little_endian_u64` |
| 4.3 checksum before any KDF; NFKD; exactly 12 words; length, word, checksum in that order | `bip39::tests::checksum_before_kdf_no_argon2_runs_for_a_bad_phrase`, `phrase_fuzz::nfkd_and_case_and_unicode_white_space`, `phrase_fuzz::precedence_length_then_word_then_checksum` |
| 4.3 the four-letter prefix of a word is unique (autocomplete) | `design_vectors::the_wordlist_is_the_official_one` |
| zeroization of every secret type; constant-time comparison | `zeroize::heap_secrets_are_zero_when_they_are_freed`, `zeroize::inline_secrets_are_zero_after_drop`, `zeroize::every_secret_type_is_zeroize_on_drop_and_prints_nothing`, `zeroize::comparison_of_secrets_is_constant_time_equality`, `argon::tests::the_argon2_blocks_are_wiped_when_the_guard_drops` |
| **text secrets leave no copy in freed memory** (M-4): phrase encode, decode, `phrase_wrap_key`, a typed phrase, a phrase whose NFKD form outgrows its buffer, the kit code encode, decode, typed | `zeroize_text::text_secrets_leave_no_copy_in_freed_or_moved_memory` (an allocator that looks for the text in every block freed or moved, with controls), `text::tests::pushing_builds_the_same_text_as_a_string_across_growth_and_multibyte_characters` |
| **no key is left in the dead stack** (L-7): every primitive that takes or makes a key, the input and the output, in `top` and `deep` calls, with a positive and a negative control and the floor of a function that returns a `Secret`; the `_into` functions leave none | `zeroize_stack::no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` (run in three configurations: `cargo test`, `cargo test --release`, `cargo test --profile vault-probe`; the last two are CI steps), `typed_api::the_into_variants_write_what_the_by_value_functions_return_and_leave_out_untouched_on_an_error`, `typed_api::every_key_that_is_made_at_random_is_not_zero_and_not_the_same_twice` |
| **every wrap draws a fresh random nonce** (M-5): 10,000 wraps, no repeat, every byte varies, bits balanced | `nonce::ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces`, `nonce::wrapping_a_key_twice_gives_different_blobs_and_different_nonces_and_both_unwrap` |
| **HMAC**: the seven cases of RFC 4231; a tag that is not whole (any prefix, suffix, extension, one bit) is refused (M04) | `hmac::hmac_sha256_gives_the_rfc_4231_tags_and_verify_accepts_them`, `hmac::verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension` |
| **one set of white space**, JavaScript's `\s` (25 characters), for the phrase and the kit; the kit decoder and the phrase decoder agree with Node on every entry of a corpus (L-8, M52, M53, M33) | `text::tests::the_white_space_set_is_javascripts_backslash_s_for_every_code_point` (all 1,112,064 scalar values), `text_corpus::the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be`, `text_corpus::the_phrase_decoder_agrees_with_the_browser_code_on_every_entry_of_the_corpus`, `text_corpus::a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read`, `text_corpus::next_line_does_not_separate_the_words_of_a_phrase` |
| the typed entry points are the ones the known-answer tests check (L-10) | `typed_api::derive_by_purpose_is_the_free_form_derivation_of_its_registry_row`, `typed_api::a_nonce_is_random_public_and_used_up_by_the_seal_that_takes_it` |

A test cannot see timing, so `ct_eq` is tested for what it returns and reviewed for what it calls (`subtle`).

## Test vectors and where they come from

Everything a test reads is in `tests/vectors` and is compiled in, so the tests need no FormLogic checkout, no scratch folder and no network. The
folder is `-text` in `.gitattributes` and `tests/provenance.rs` pins the SHA-256 of every file in it.

| File | What | Made by |
|---|---|---|
| `formlogic/e2ee-envelope-vectors.json`, `e2ee-sealed-js.json`, `e2ee-sealed-php.json` | FormLogic's committed vectors: kdf x2, XChaCha20 x2 with the bad-AAD matrix, Argon2id x2, FLRK1 x3, the envelope AADs; sealed boxes written by JavaScript and by PHP and one whole `__flenc:1` envelope | copies, byte for byte, of `docs/contracts/` at FormLogic commit `81860c8a937b9e2a2ed7e6f71c5d7a9dd8cad1fc` (last changed in `83a7eec3`); their git blob ids are FormLogic's: `609800802a0dc8f9ba682b7c1546666910b86735`, `93fb73ef371f7fd7b6bf4183ed3fbc523921955d`, `79ea626d2384b5940febfcb9cceebde1aaf9f767` (`git hash-object` prints them) |
| `vault-work/vectors.json` | the design's vectors (phrase, wrapper, registry, backup identity and manifest signature, signed operations and head, archive signature, ceremony, Ed25519 probe) | the design's `gen.mjs` (Node, libsodium) and recomputed by `verify.py` (Python, OpenSSL, a hand-written age): 59 + 11 checks; `run_all.ps1` was rerun for this work: 70 checks `ok`, both suites `ALL OK`, `vectors.json` unchanged (SHA-256 `7057862f...`); the age files typage writes with random keys (`nodeage.bin`, `rekeyed.*`) come out different on every run and were restored to the design's copies afterwards |
| `public-vectors.json` | RFC 5869 cases 1 to 3, RFC 8032 section 7.1 (five), RFC 7748 (two, the iterated vectors to 1,000,000, section 6.1), RFC 9106 Argon2id, draft-irtf-cfrg-xchacha-03 A.3.1, the eight 128-bit BIP-39 vectors of the reference | `scripts/extract_public.py` parses the published texts (URLs and hashes of the sources are in the file), and recomputes every value with Python `cryptography` before writing it: 54 checks. Nothing is typed from memory |
| `text-corpus.json` | the code points of JavaScript's `\s`, 274 kit-code inputs with the verdict of a port of FormLogic's `decodeRecoveryKey`, and 177 phrases with the verdict of the browser decoder of design 4.3 (`normalize('NFKD').toLowerCase().split(/\s+/)`) with and without the 1024-byte cap (`verdict`, `js_verdict`), each entry classed `same` (the Rust decoder must agree) or `stricter` (JavaScript accepts it, or says something else, and this decoder must refuse it: for a kit code trailing bits, Unicode upper-casing, the 256-byte cap; for a phrase, only above the 1024-byte cap) | `scripts/text_corpora.mjs` (Node 24; ASCII-only output) |
| `libsodium-oracle.json` | libsodium's verdict on 80 Ed25519 cases (the RFC 8032 vectors malleated in every way, every point of small order in every encoding, the twelve ed25519-speccheck cases, two mixed-order-key forgeries), the 14 low-order X25519 encodings, 14 forged sealed boxes, and known answers for `crypto_kdf`, XChaCha20-Poly1305, Ed25519, X25519, sealed boxes, Argon2id and HKDF | `scripts/gen_corpus.py` (plain Python integer arithmetic: the torsion points, the malleations), `scripts/oracle.mjs` (libsodium 1.0.x from FormLogic's `node_modules`, and Node's OpenSSL), `scripts/oracle_check.py` (Python `cryptography`, `hashlib`, a hand-written Salsa20 family): 467 recomputations, all agree. Where Python's OpenSSL disagrees with libsodium (16 of the 80 Ed25519 cases: it accepts what a strict verifier refuses, design finding A5) the file says so, and this crate sides with libsodium |

The scripts are kept beside the data. They name the scratch folder they were run from (`vault-work-backup`, FormLogic's `node_modules`); they are
there to audit and to rerun, not to run in CI.

Every value that the design says must be reproduced by two implementations is: the phrase, the wrapper, the registry, the manifest, operations, head and
archive signatures and the ceremony by Node-libsodium and Python-OpenSSL (design); the RFC vectors by the RFC text and Python; the oracle by libsodium and
Python (and node's OpenSSL for Argon2id and HKDF); FormLogic's by its JavaScript, its PHP, and now this crate.

## Dependencies

Versions are the ones in `Cargo.lock` (the workspace lock, checked with `--locked`); licences are read from each crate's own `Cargo.toml`.
`cargo audit` (advisory database of 1,277 advisories, fetched 2026-10-01) reports nothing for any of these; its two warnings are `paste 1.0.15`
(unmaintained) and `yoke-derive 0.8.3` (yanked), in other members of the workspace.

| Crate | Locked | Licence | Why |
|---|---|---|---|
| `argon2` | 0.5.3 | MIT OR Apache-2.0 | Argon2id (RustCrypto). Only `zeroize` is enabled; the crate's `hash_password_into` is not used, because it frees its memory unwiped: the module supplies and wipes the blocks (`hash_password_into_with_memory`) |
| `blake2` | 0.10.6 | MIT OR Apache-2.0 | keyed BLAKE2b for `crypto_kdf` and the 24-byte nonce of a sealed box |
| `chacha20poly1305` | 0.10.1 | Apache-2.0 OR MIT | XChaCha20-Poly1305-IETF (RustCrypto). The versions of this row, `x25519-dalek`, `hkdf`, `hmac`, `sha2`, `subtle` and `zeroize` are exactly those in the lock of the `backup` branch (`age 0.11.5`), and `ed25519-dalek 2.2.0` and `curve25519-dalek 4.1.3` those of the desktop's, so when V-05 merges no crate is built twice |
| `crypto_secretbox` | 0.1.1 | Apache-2.0 OR MIT | XSalsa20-Poly1305 and HSalsa20 for the sealed box. The design names `crypto_box` with `seal`; that crate does no all-zero check on the Diffie-Hellman result, so the sealed box is built here from `x25519` and this crate |
| `ed25519-dalek` | 2.2.0 | BSD-3-Clause | Ed25519 with `verify_strict` (the line the desktop already uses) |
| `x25519-dalek` | 2.0.1 | BSD-3-Clause | X25519 with `was_contributory` |
| `hkdf` / `hmac` / `sha2` | 0.12.4 / 0.12.1 / 0.10.9 | MIT OR Apache-2.0 | HKDF-SHA256, HMAC-SHA256, SHA-256, SHA-512 (RustCrypto) |
| `subtle` | 2.6.1 | BSD-3-Clause | the constant-time comparison |
| `zeroize` | 1.9.0 (asked `1.8.2`) | Apache-2.0 OR MIT | zeroization; `derive` for the marker trait |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | the operating system's random generator, the only source of randomness (BCryptGenRandom, getrandom(2), getentropy); a failure is an error |
| `unicode-normalization` | 0.1.25 | MIT OR Apache-2.0 | NFKD of a typed phrase |
| **dev** `bip39` | 2.2.2 | CC0-1.0 | an independent BIP-39 implementation (rust-bitcoin's) that the 200,000-input fuzz checks this crate's decoder against; it is not used by the library, which has its own 60 lines and the official word list |
| **dev** `serde_json` | 1.0.151 | MIT OR Apache-2.0 | reads the vector files |

The closure of the library is 46 crates along normal dependency edges, on every target (build-time helpers such as rustc_version, semver and version_check come with curve25519-dalek on top; the proc-macro helpers `syn`, `quote`, `proc-macro2`; `curve25519-dalek` with `fiat-crypto`
and `cpufeatures`; `digest`, `cipher`, `aead`, `generic-array`, `typenum`; `tinyvec`; `libc`; and, for targets no build of OAIY has, `wasip2`,
`wit-bindgen` and `r-efi`). Licences across it: dual MIT / Apache-2.0 for all but these: BSD-3-Clause (four: `ed25519-dalek`, `x25519-dalek`,
`curve25519-dalek`, `subtle`), MIT (`generic-array`), and rows with a permissive alternative among several (`fiat-crypto`: MIT OR Apache-2.0 OR BSD-1-Clause;
`tinyvec`, `tinyvec_macros`: with Zlib; `unicode-ident`: (MIT OR Apache-2.0) AND Unicode-3.0; `r-efi`, which is not compiled for any target OAIY builds,
offers LGPL-2.1-or-later only as a third choice). The dev-only crates add CC0-1.0 (`bip39`, `bitcoin_hashes`, `hex-conservative`) and Unlicense OR MIT
(`memchr`). Nothing is copyleft-only. The BSD-3-Clause crates need their notice reproduced in a binary distribution: add them to `NOTICE` when a build links
this crate (a follow-up, as nothing links it yet).

## Unsafe

None in the library: `#![forbid(unsafe_code)]`. Four test files contain `unsafe`, each justified where it stands, and they are the only four:
`tests/argon_ceiling.rs` (a `GlobalAlloc` that records the largest request, to show that a hostile Argon2 cost asks for no memory), `tests/zeroize.rs`
(a `GlobalAlloc` that reads a block at the moment it is freed, and a volatile read of a slot after `drop_in_place`, to show that secrets are zero by then),
`tests/zeroize_text.rs` (a `GlobalAlloc` that looks for a run of a phrase or a kit code in every block that is freed or moved) and `tests/zeroize_stack.rs`
(a volatile read of stack memory that nothing has written since a function returned, to count the copies of a key that it left: that read is undefined behaviour in
Rust's abstract machine and is what every stack-scanning test does; the counts are asserted, not the bytes). Each allocator forwards every call to the system allocator.

## Where this differs from the design

1. **Sealed box.** Built from `x25519` (with the all-zero check), HSalsa20 and XSalsa20-Poly1305 from `crypto_secretbox`, not with `crypto_box`'s `seal`
   feature (see the table). Byte-for-byte equal to libsodium (`sealbox::tests`, 9 deterministic vectors, and FormLogic's six fixtures).
2. **More than the eight files listed.** `kit.rs` (FLRK1: it is in the test list and had no file), `canon.rs`, `error.rs`, `random.rs`. The bip39 module also
   has `phrase_wrap_key`, the composite of design 4.3 that turns twelve words into the wrap key, because the rule "checksum before the KDF" is a property of that
   composite: it takes an `Entropy`, which only `decode` (or 16 random bytes) makes.
3. **The API enforces 4.1.3 and 4.1.4 itself.** The design states them as rules for host operations. Here a `SigningKey` has a role, signing takes a
   `SignedString` built from an enum of domains, and signing arbitrary bytes needs a key made with `KeyRole::Hazmat` (for the RFC vectors and other
   specifications' messages). A consumer that lets a remote party pick the role has built a signing oracle; `grep Hazmat` finds every such place.
4. **Stricter than libsodium in three places, and than FormLogic's JavaScript in three, on purpose.** A KDF context is eight characters of `[a-z0-9]` (every registry row is); an Argon2id
   memory size is a whole number of KiB (libsodium rounds down); `crypto_kdf` subkey lengths are 16, 32 and 64 (libsodium allows 16 to 64). The FLRK1 decoder requires the four unused bits of the
   52nd character to be zero, upper-cases ASCII only (the JavaScript ignores those bits and maps the dotless i, the long s and some ligatures onto `A` to `Z`, so two spellings of one key decode
   there and only one here), and reads at most 256 bytes (`kit::MAX_INPUT_BYTES`; the JavaScript has no cap). Each is a class of entries in `tests/vectors/text-corpus.json`: no code this decoder
   accepts is refused by the JavaScript.
5. **White space is JavaScript's `\s`, for the phrase and the kit alike** (review L-8): TAB, LF, VT, FF, CR, SPACE, NBSP, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000 and
   U+FEFF, 25 characters, compared with Node over every code point. It is not Unicode `White_Space` (which also has U+0085, NEL: a phrase that separated its words with it used to decode here and
   not in a browser) and not the ASCII set the kit decoder used (which refused codes that FormLogic accepts when pasted with a no-break space, an ideographic space, a vertical tab or a
   byte-order mark). Input over 1024 bytes (a phrase) or 256 bytes (a kit code) is refused before it is normalised.
6. **`unwrap` refuses blobs of 40 bytes or fewer, and `wrap` refuses an empty plaintext,** as `unwrapKey` in `vault.ts` does; the raw `seal` and `open` handle the
   empty message (FormLogic's sealed empty message opens).
7. **A wrong salt length is `kdf_params_out_of_range`,** with the other bounds, not a separate error: it is a parameter the server chooses.
8. **The workspace** gets `[profile.dev.package.argon2] opt-level = 3` so that the 64 MiB tests take a tenth of a second; nothing else is optimised.
9. **`aead::seal` takes a `Nonce`, not 24 bytes; `kdf::derive_subkey` is not in the library** (review L-10). A `Nonce` can only come from the random generator and is used up by the seal that takes it;
   the functions that take a nonce or a context of the caller's choosing exist only with the `test-vectors` feature. A protocol that needs a nonce it derives (the ceremony of 4.8, when V-03 builds it)
   gets a constructor of its own, with its own test, rather than a way for anyone to choose.
10. **The derivations scrub the stack, at one depth, and write the keys they make in place** (review L-7 and the second review, low 1). `kdf::derive`, HKDF, HMAC, X25519, Ed25519 `from_seed`
    and the generators overwrite 96 KiB of the stack below the caller after they return, **in every build configuration** (it was 4 KiB or 96 KiB by `debug_assertions`, which is not what decides how deep
    a computation's frames are: a build with assertions off and no optimisation left a copy of the master key). And the keys that come out are written where the caller says: `kdf::derive_into`,
    `hkdf_sha256_secret_into`, `RecoveryKit::wrap_key_into`, `SecretKey::diffie_hellman_into`, `aead::unwrap_key_into`, `bip39::wrap_key_into` and `phrase_wrap_key_into`, on `Secret::zeroed()` and
    `expose_mut()`; the by-value functions are kept as their wrappers. See the known limits for the counts.

## Known limits

- Rust cannot promise that no copy of a secret is left in a register or a dead stack slot; the crate removes the copies it owns, and `zeroize_stack` counts what is left, for every primitive, on Windows and
  Linux, in **three** build configurations (`cargo test`: no optimisation, assertions on; `cargo test --release`; `cargo test --profile vault-probe`: no optimisation **and** no assertions, which neither of the
  others is). The probe runs every call under a frame of its own (`deep`) and at eight depths, and keeps the largest count, because the scan's frame and the caller's own `unwrap` and drop
  overwrite the nearest few hundred bytes of the stack where the code has just been (the first two forms of the probe did not see a copy left in the frame of an `_into` function for that reason:
  the mutants that write the key through a plain array or a by-value `Secret` before they copy it to `out`, S03, S05, S06 and, through the by-value derivation, S11 and S12, were alive until the probe was changed). **The input keys: none, in any of them** (Ed25519 `from_seed`, which
  returns a key that is the seed, is the one exception: one, which is the price of a by-value return: the harness's own move of the result with no optimisation, the frame of `from_seed` in an optimised build).
  **The keys that come out of an `*_into` function: none, in any of them** (`kdf::derive_into`, `hkdf_sha256_secret_into`, `RecoveryKit::wrap_key_into`, `SecretKey::diffie_hellman_into`,
  `aead::unwrap_key_into`, `bip39::wrap_key_into`, `bip39::phrase_wrap_key_into`, `argon2id13_into`). **The text** that is read or made (the phrase, the code: `bip39::decode`, `bip39::encode`,
  `RecoveryKit::decode`, `RecoveryKit::encode`, `bip39::phrase_wrap_key`): **none of it** is left, and **the Argon2 output** that a phrase's wrap key is derived from **is never left** (third review, L-1 and L-2: the
  decoders, the encoders and `phrase_wrap_key_into` did not scrub and had no row, and left one to three copies of the entropy or the kit key; `argon2id13` was by value only and left its output; the
  probe's bip39 needles had a period). **The keys that come out by value** (`kdf::derive`, `hkdf_sha256_secret`, `RecoveryKit::wrap_key`, `SecretKey::diffie_hellman`, `aead::unwrap_key`,
  `bip39::wrap_key`, `bip39::phrase_wrap_key`, `argon2id13`, and the entropy and the kit that the two decoders return): at most the floor of a function with no cryptography in it plus one. The floor, measured the same way
  (a `Secret` returned through a `Result`, moved out of the harness's frame), is **4** with no optimisation (assertions on or off, Windows and Linux) and **1** in an optimised build; the functions leave 2 to 3 with
  no optimisation and 1 to 2 in an optimised build (kdf::derive 3 / 2, hkdf 3 / 1, kit.wrap_key 3 / 2, x25519 3 / 2, unwrap_key 3 / 2, bip39::wrap_key 3 / 2, argon2id13 3 / 2, phrase_wrap_key 3 / 2, bip39::decode
  2 to 3 / 2, RecoveryKit::decode 2 / 2). **The keys made at random** (`Secret::random`, the `generate` functions, `Entropy::random`), by value and
  scrubbed, at most two above the floor: measured 0 to 4 (Windows: up to 2 with assertions on, 3 optimised, 4 with no optimisation and no assertions; Linux: up to 4, with `Entropy::random` and
  `RecoveryKit::generate` the highest). **The scrubs of the generators are not seen by the probe** (third review, Y37 to Y40): with the scrub removed from `Entropy::random`, `RecoveryKit::generate`,
  `SecretKey::generate` and `SigningKey::generate` the probe counts within one copy of what it counts with it, in either direction, in all three configurations, and always within the bound (the counts below), because the
  generators write the key where it lives and leave nothing below their frame for a scrub to remove; the scrub there is defence in depth, and a unit test (`zeroize::tests::the_functions_that_make_or_read_a_key_scrub_the_stack_after_it`) requires that each of them, and each decoder
  and encoder, calls it. **The bounds on the by-value and the made keys are loose with no optimisation** (the floor is 4 and the functions leave 3, so "floor plus one" and "floor plus two" allow 5 and 6):
  they do not discriminate there, and the optimised build is where they bite (floor 1, by-value at most 2, made at most 3, Ed25519 `generate` exactly at 3); the assertions that do discriminate in every configuration are the two that matter for
  a consumer, **none in the input and none in an `_into` output**. **A consumer that has to show that no key is in memory
  after it has connected (design test P9: the UMK and what is derived from it: `flbkrcp1`, `flbksig1`) uses the `_into` functions and must account for the rest: tracked as a follow-up for V-13 (a
  `generate_into` for the keys made at random and a constructor for `SigningKey` that fills in place).** What none of this covers: a register; a swap file; a debugger. The scrub is an overwrite of a depth
  measured for this compiler and this dependency tree (8 KiB and 64 KiB left a copy, 88 KiB did not), so a new `blake2`, a new compiler or a new target can need it deeper, and the probe is what says so.
  A thread with less than about 200 KiB of stack left is not safe to derive on (the scrub is a 96 KiB frame).
- `argon2` keeps a 1 KiB stack buffer of the first blocks while it initialises; `zeroize_stack` probes it with a 32-byte password and finds nothing.
- `PartialEq` on a secret is constant-time; a caller who copies bytes out with `expose()` and compares them with `==` has left the crate's protection.
- The tests cannot see timing; the mutant that replaces `ct_eq` with `==` survives (below).

## Key roles are fixed at construction (advisory, review L-10)

A `SigningKey` has a role (`Vault`, `Writer`, ...) and signs only the domains of that role (design 4.1.4, R-KEY). The role is fixed when the key object is made, **not when the seed was made**: the seed is 32 bytes,
`SigningKey::seed()` hands them out, and `SigningKey::from_seed(other_role, &seed)` makes a key of another role from the same seed. So R-KEY holds for a key object and across the functions of this crate, and does not hold
for key material that a caller copies between roles. What keeps a seed in its role is what the caller stores it under: the keystore names (`archive.writer`, `vault.fk.<id>`, ...) are the roles, and a caller that reads the
seed of one name and builds a key of another role has made a signing oracle out of it. Nothing in this crate can prevent that; V-03 must not write that code, and `grep from_seed` finds every place where a role
is chosen, as `grep Hazmat` finds every place that signs arbitrary bytes.

## For V-03, V-04 and FormLogic (follow-ups, nothing here edits FormLogic)

1. **V-04 must pin the phrase white space, and the 1024-byte cap, before it ships.** The decoder of design 4.3 in TypeScript is `normalize('NFKD').toLowerCase().split(/\s+/)` and this crate agrees with it on every entry of
   `tests/vectors/text-corpus.json` (177 phrases, verdicts computed by exactly that code in Node, with and without the cap) **except above the 1024-byte cap**, which this decoder has (design 4.3) and that code as it stands does not:
   the five entries above the cap are class `stricter` (the browser would read them, or say `word` where this says `length`). Put the same corpus in the browser's test suite, and never let a browser library's `trim`, an input element's
   normalisation or a mobile keyboard's autocorrect change what `\s` means to it: the set is the 25 characters of `text::is_js_space`.
2. **Raise on the FormLogic side: tighten `decodeRecoveryKey` (`formlogic/ui/src/lib/crypto/vault.ts`).** It accepts four trailing bits that are not zero (35 corpus entries of that class: two spellings
   of one key decode, and the checksum covers only the key), and it upper-cases with Unicode rules, which maps the dotless i (U+0131), the long s (U+017F) and ligatures such as U+FB01 and U+FB02 onto letters of the alphabet (13 corpus entries). This
   crate refuses both on purpose; a recovery code must mean one thing. The corpus (`kit` entries of class `stricter`) is the test to hand them: the fix is to require the last character's four low bits to be zero,
   to upper-case ASCII only (`s.replace(/[a-z]/g, c => c.toUpperCase())`), and optionally to cap the input at 256 bytes, which is ours. Until they do, a code that FormLogic wrote is read here (every code it writes has
   zero trailing bits and ASCII letters), and a code this crate writes is read there.
3. **Key names are not FormLogic ids** (the keystore README): a FormLogic id with an upper-case letter is `InvalidName` in the keystore; a caller chooses the mapping (lower-case hex of the id's bytes is a valid name).
4. **A protocol that needs a nonce it derives** (4.8) needs its own typed constructor next to `aead::Nonce`, with a test; do not reopen `seal` to 24 arbitrary bytes.
5. **V-13: the keys that are still copied.** The counts in the known limits are the tracking item: a consumer that must pass design test P9 uses the `_into` functions, and the by-value functions,
   the generators and Ed25519 `from_seed` leave the copies listed there until they have in-place forms.

## Mutation checks

Each mutant breaks the code in one place (two for a check with a second line of defence), runs the crate's whole suite, and restores the file from git:
`scratchpad/vault-impl/mutate.ps1` and `mutants.ps1` (kept outside the repository; the table below is their output). **34 mutants: 31 killed, 3 survived.** No mutant
hung or failed to compile in the final run.

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| C01 | Ed25519 verification is not strict (dalek verify instead of verify_strict) | KILLED | `a_small_order_r_under_a_key_of_mixed_order_is_what_strict_verification_refuses`, `ed25519_negative_corpus_gives_libsodiums_verdict_on_all_80_cases` |
| C02 | a non-canonical encoding of a public key is accepted | KILLED | `a_non_canonical_encoding_of_a_good_key_is_refused_and_the_canonical_one_is_not`, `every_point_of_small_order_in_every_encoding_is_refused_as_a_public_key` |
| C03 | a public key of small order is accepted (no pin-time check) | KILLED | `the_ed25519_strictness_probe_of_design_finding_a5`, `every_point_of_small_order_in_every_encoding_is_refused_as_a_public_key` |
| C04 | a signing key signs in domains of other roles (R-KEY) | KILLED | `the_archive_signature_covers_every_content_field`, `signed_vault_operations_and_the_hostile_label`, `the_backup_manifest_signature_flbackup_1` |
| C05 | a role-bound key signs arbitrary bytes | KILLED | `rule_4_1_3_2_no_operation_accepts_a_caller_chosen_domain` |
| C06 | the all-zero Diffie-Hellman result is not refused (was_contributory) | KILLED | `x25519::tests::diffie_hellman_refuses_an_all_zero_result_even_for_a_key_that_bypassed_the_constructor` |
| C07 | a low-order X25519 public key is accepted by the constructor | KILLED | `x25519::tests::the_constructor_refuses_the_table_and_only_the_table`, `the_low_order_table_is_the_set_libsodium_refuses` |
| C08 | both low-order barriers removed (a forged sealed box under a zero shared secret would open) | KILLED | `x25519::tests::diffie_hellman_refuses_an_all_zero_result_even_for_a_key_that_bypassed_the_constructor`, `x25519::tests::the_constructor_refuses_the_table_and_only_the_table`, `a_forged_sealed_box_under_a_small_order_ephemeral_key_is_refused` |
| C09 | the low-order check does not ignore the top bit of the u-coordinate | KILLED | `x25519::tests::the_constructor_refuses_the_table_and_only_the_table`, `the_low_order_table_is_the_set_libsodium_refuses` |
| C10 | no memory ceiling on Argon2id | KILLED | `argon::tests::out_of_range_parameters_never_reach_the_derivation`, `c1_a_hostile_argon2_cost_is_refused_before_anything_is_allocated` |
| C11 | no operations ceiling on Argon2id | KILLED | `argon::tests::out_of_range_parameters_never_reach_the_derivation`, `c1_a_hostile_argon2_cost_is_refused_before_anything_is_allocated` |
| C12 | no floor on Argon2id cost | KILLED | `argon::tests::formlogic_vector_1_below_the_floor_through_the_raw_function`, `argon::tests::out_of_range_parameters_never_reach_the_derivation`, `c1_a_hostile_argon2_cost_is_refused_before_anything_is_allocated` |
| C13 | the guard does not wipe the Argon2 blocks | KILLED | `argon::tests::the_argon2_blocks_are_wiped_when_the_guard_drops`, `argon::tests::derive_in_leaves_the_storage_zeroed` |
| C14 | the explicit second wipe of the Argon2 storage is removed (the guard still wipes: a redundant line) | SURVIVED |  |
| C15 | the phrase checksum is not verified | KILLED | `bip39::tests::checksum_before_kdf_no_argon2_runs_for_a_bad_phrase`, `bip39_six_entropies_and_four_failures`, `precedence_length_then_word_then_checksum` |
| C16 | the phrase is not NFKD-normalised | KILLED | `nfkd_and_case_and_unicode_white_space`, `a_hundred_and_fifty_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate` |
| C17 | more than twelve words are accepted | KILLED | `bip39_six_entropies_and_four_failures`, `nfkd_and_case_and_unicode_white_space`, `a_hundred_and_fifty_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate` |
| C18 | a | is allowed inside a signed field or an AAD field | KILLED | `signed_vault_operations_and_the_hostile_label`, `rule_4_1_3_3_a_pipe_or_lf_or_any_free_text_in_a_signed_field_is_refused` |
| C19 | the builder does not check a field at all | KILLED | `signed_vault_operations_and_the_hostile_label`, `rule_4_1_3_3_a_pipe_or_lf_or_any_free_text_in_a_signed_field_is_refused` |
| C20 | unwrap ignores the AAD | KILLED | `the_phrase_wrapper_of_section_4_2_1`, `a_wrapped_key_refuses_every_single_bit_flip_in_the_nonce_the_ciphertext_and_the_tag`, `heap_secrets_are_zero_when_they_are_freed` |
| C21 | unwrap opens a 40-byte blob (an empty plaintext) | KILLED | `wrap_never_wraps_nothing_and_unwrap_never_opens_a_blob_of_40_bytes_or_fewer` |
| C22 | wrap accepts an empty plaintext | KILLED | `wrap_never_wraps_nothing_and_unwrap_never_opens_a_blob_of_40_bytes_or_fewer` |
| C23 | the crypto_kdf subkey id is big-endian in the salt | KILLED | `the_backup_manifest_signature_flbackup_1`, `the_backup_recipient_secret_and_public_key`, `the_kdf_registry_vectors` |
| C24 | two KDF contexts are the same string (a reused context) | KILLED | `the_kdf_registry_vectors`, `the_backup_manifest_signature_flbackup_1`, `rule_4_1_2_the_kdf_context_registry_is_the_design_table_and_has_no_duplicates` |
| C25 | a Secret is not zeroized when dropped | KILLED | `inline_secrets_are_zero_after_drop` |
| C26 | FLRK1 trailing bits are not required to be zero | KILLED | `recovery_kit_codes_refuse_every_single_character_substitution_and_the_trailing_bits` |
| C27 | the FLRK1 checksum is not verified | KILLED | `recovery_kit_codes_refuse_every_single_character_substitution_and_the_trailing_bits` |
| C28 | ct_eq is a plain == (timing: a test cannot see this) | SURVIVED |  |
| C29 | the sealed-box nonce is derived from the keys in the other order | KILLED | `sealbox::tests::a_random_seal_differs_every_time_and_always_opens`, `sealbox::tests::the_deterministic_seal_equals_libsodiums_construction`, `sealed_boxes_written_by_php_open` |
| C30 | the prefix-free check misses a token that is a prefix of another | KILLED | `rule_4_1_3_1_the_prefix_check_catches_what_it_is_for` |
| C31 | a libsodium secret key with the wrong public half is accepted | KILLED | `ed25519_signatures_equal_libsodiums_and_the_libsodium_secret_key_form_is_checked` |
| C32 | HKDF accepts an output longer than 255 blocks (the crate still refuses it: a redundant guard) | SURVIVED |  |
| C33 | a memory size that is not a whole number of KiB is accepted | KILLED | `argon::tests::out_of_range_parameters_never_reach_the_derivation` |
| C34 | input longer than 1024 bytes is normalised and decoded | KILLED | `a_hundred_and_fifty_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate` |

The three survivors, and why none is a gap:

- **C14** removes the second, explicit wipe of the Argon2 storage. The `WipeOnDrop` guard wipes it too (C13 removes the guard and is killed), so the line is redundant; the allocator probe
  in `tests/zeroize.rs` sees zero either way.
- **C28** replaces the constant-time `ct_eq` with `==`. Only timing changes, and a test cannot see timing. The comparison is reviewed for what it calls (`subtle`), not tested.
- **C32** removes the explicit 255-block guard of `hkdf_sha256`; the `hkdf` crate refuses the length too and the error is the same, so the guard is redundant.

Found while choosing the mutants, and fixed before the run: the first corpus had no case that separates strict verification from the plain equation once the key is acceptable, and a sealed-box test whose
"forged" boxes had random tags would have passed without the all-zero check; `a_small_order_r_under_a_key_of_mixed_order_...` and `a_forged_sealed_box_under_a_small_order_ephemeral_key_is_refused` were added and
C01 and C08 are killed by them.

### Round 2, after the independent review

Where each of the reviewer's surviving mutants went (the round-1 mutants were not re-run: the reviewer's survivors are covered by these): M04 by T01 (and T02 to T04), M05 by N01 (and N02 to N04), M33 by W06 to W08, M52 by W09 and W01, M53 by W05. **H05** (the zeroizing growth of a secret `String` leaves its old buffer unwiped) survived at first, because the allocator probe only ran inputs that fit the reserved buffers; the section with a phrase whose NFKD form outgrows its buffer (U+FDFA) was added and kills it.

The review found the primitives sound, and its own mutants found six places in this crate that no test noticed (M04 HMAC verify accepts a truncated tag; M05 wrap uses an all-zero nonce; M33 no length cap on a kit code; M52 U+0085 as a phrase separator; M53 the kit's white space set; and the dead-stack copies of L-7, which a mutant cannot show). Round 2 is run on the code after the fixes, in a second worktree (`scratchpad/vault-impl/mutate2.ps1` and `mutants2.ps1`, outside the repository), on Windows, except the optimised-build and Linux ones (L702 and L703 are run on Linux under WSL in a release build, and L705 and L706 in a release build on Windows): **32 mutants, 32 killed.** The keystore's round 2 is in its README.

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| H01 | bip39::decode builds its NFKD copy by pushing into a String that grows (unwiped blocks) | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| H02 | bip39::encode builds the phrase in a String that grows | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| H03 | the kit's encode builds the code in a String that grows | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| H04 | the kit's decode builds its cleaned copy in a String that grows | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| H05 | push_zeroizing leaves the buffer it grows out of unwiped | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| H06 | the lower-case copy of the phrase grows by pushing into a String | KILLED | `text_secrets_leave_no_copy_in_freed_or_moved_memory` |
| L701 | kdf::derive does not scrub the stack (debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| L702 | HKDF does not scrub the stack (optimised build, Linux) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| L703 | HMAC does not scrub the stack (optimised build, Linux) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| L704 | the scrub of a debug build is too shallow (8 KiB) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| L705 | the scrub writes nothing the compiler must keep (optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| L706 | Ed25519 from_seed does not scrub the stack (optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack` |
| W01 | U+0085 is white space (Unicode's set, not JavaScript's) | KILLED | `the_white_space_set_is_javascripts_backslash_s_for_every_code_point`, `next_line_does_not_separate_the_words_of_a_phrase`, `the_kit_decoder_strips_exactly_the_white_space_of_javascript` |
| W02 | the byte-order mark is not white space | KILLED | `the_white_space_set_is_javascripts_backslash_s_for_every_code_point`, `two_hundred_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate`, `next_line_does_not_separate_the_words_of_a_phrase` |
| W03 | the vertical tab is not white space | KILLED | `the_white_space_set_is_javascripts_backslash_s_for_every_code_point`, `next_line_does_not_separate_the_words_of_a_phrase`, `the_kit_decoder_strips_exactly_the_white_space_of_javascript` |
| W04 | the no-break space is not white space | KILLED | `the_white_space_set_is_javascripts_backslash_s_for_every_code_point`, `the_kit_decoder_strips_exactly_the_white_space_of_javascript`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W05 | the kit decoder strips ASCII white space only (M53) | KILLED | `a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read`, `the_kit_decoder_strips_exactly_the_white_space_of_javascript`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W06 | the kit decoder has no length cap (M33) | KILLED | `a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W07 | the kit cap is one byte too small | KILLED | `a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W08 | the kit cap counts characters, not bytes | KILLED | `a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W09 | the phrase decoder splits on Unicode white space (M52) | KILLED | `two_hundred_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate`, `next_line_does_not_separate_the_words_of_a_phrase`, `the_phrase_decoder_agrees_with_the_browser_code_on_every_entry_of_the_corpus` |
| W10 | the kit decoder upper-cases Unicode letters like JavaScript does | KILLED | `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| W11 | the four unused bits of the last character are ignored, like JavaScript does | KILLED | `recovery_kit_codes_refuse_every_single_character_substitution_and_the_trailing_bits`, `the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be` |
| N01 | wrap uses an all-zero nonce (the reviewer's M05) | KILLED | `ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces`, `wrapping_a_key_twice_gives_different_blobs_and_different_nonces_and_both_unwrap` |
| N02 | only the first 8 bytes of a nonce are random | KILLED | `ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces` |
| N03 | a nonce is a counter | KILLED | `ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces` |
| N04 | Nonce::random is not random at all | KILLED | `wrapping_a_key_twice_gives_different_blobs_and_different_nonces_and_both_unwrap`, `ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces`, `a_nonce_is_random_public_and_used_up_by_the_seal_that_takes_it` |
| T01 | HMAC verify accepts a truncated tag (M04) | KILLED | `verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension` |
| T02 | HMAC verify compares the last bytes only | KILLED | `verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension` |
| T03 | HMAC verify compares the first 16 bytes only | KILLED | `verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension` |
| T04 | HMAC verify accepts every tag | KILLED | `verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension`, `verify_rejects_another_key_or_another_message` |
| T05 | the typed derive uses the wrong subkey id | KILLED | `the_backup_manifest_signature_flbackup_1`, `the_backup_recipient_secret_and_public_key`, `the_kdf_registry_vectors` |

The older survivors (C14, C28, C32) are unchanged: a redundant line, a timing-only change and a redundant guard. The one this round leaves is in the keystore (B06, a redundant unlock).

### Round 3, after the second review

The second review (low 1) found that the derived keys left one to three copies in the dead stack and that the probe and the scrub depended on `debug_assertions`. The mutants below break the code that fixed that, one place each, and run the crate's suite on Windows (and Linux where said); the keystore's half is in its README (`scratchpad/vault-impl/mutate3.ps1` and `mutants3.ps1`, outside the repository). **29 mutants of the crypto crate: 26 killed, 3 survived** (S04, Z04 and Z05, each killed in the configuration where the code it breaks leaves something: S04r and S04p, Z04w and Z05w).

| S01 | the scrub is 8 KiB deep (debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S02 | the scrub depth is keyed on debug_assertions again (no optimisation, no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S03 | derive_into derives into a plain array and copies (a copy of the key stays in the stack) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S04 | diffie_hellman_into does not scrub the stack | SURVIVED |  |
| S05 | hkdf_sha256_secret_into derives into a plain array and copies | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S06 | unwrap_key_into goes through a plain array | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S07 | unwrap_key_into writes out before it knows the blob is good | KILLED | `the_into_variants_write_what_the_by_value_functions_return_and_leave_out_untouched_on_an_error` |
| S08 | Ed25519 from_seed does not scrub the stack (N06, the debug suite) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S09 | Ed25519 from_seed does not scrub the stack (N06, an optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S10 | Ed25519 from_seed does not scrub the stack (N06, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S11 | the kit wrap_key_into goes through the by-value key | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S12 | bip39 wrap_key_into goes through the by-value key | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S13 | SigningKey::generate does not fill the seed from the random generator | KILLED | `every_key_that_is_made_at_random_is_not_zero_and_not_the_same_twice` |
| S14 | SecretKey::generate does not fill the secret from the random generator | KILLED | `every_key_that_is_made_at_random_is_not_zero_and_not_the_same_twice` |
| S15 | fill_random writes nothing | KILLED | `every_key_that_is_made_at_random_is_not_zero_and_not_the_same_twice` |
| Q01 | the phrase decoder has no byte cap | KILLED | `the_phrase_decoder_agrees_with_the_browser_code_on_every_entry_of_the_corpus` |
| Q02 | the phrase cap is one byte too big | KILLED | `the_phrase_decoder_agrees_with_the_browser_code_on_every_entry_of_the_corpus` |
| S04r | diffie_hellman_into does not scrub the stack (an optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S04p | diffie_hellman_into does not scrub the stack (no optimisation, no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S03v | derive_into is derive copied into out, through a by-value Secret and a move (the first form of S03) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| S06v | unwrap_key_into goes through a by-value Secret and a move (the first form of S06) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z01 | kdf::derive does not scrub the stack (debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z02 | kdf::derive does not scrub the stack (no optimisation, no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z03 | HKDF does not scrub the stack (optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z04 | HMAC does not scrub the stack (optimised build) | SURVIVED |  |
| Z05 | HMAC verify does not scrub the stack (optimised build) | SURVIVED |  |
| Z06 | the scrub writes nothing the compiler must keep (optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z04w | HMAC does not scrub the stack (optimised build, Linux) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| Z05w | HMAC verify does not scrub the stack (optimised build, Linux) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |

The three survivors are equivalent on the build they are run in, and each has a twin that is killed where the leak exists: **S04** (`diffie_hellman_into` without the scrub) leaves no copy of the key or of the shared secret with assertions on (the copy in a build with no optimisation and no assertions, S04p, and in an optimised one, S04r, is found); **Z04** and **Z05** (HMAC and HMAC verify without the scrub) leave none in an optimised build on Windows, and leave the key on Linux (Z04w and Z05w).

How this round went, because six of the mutants were alive at first: S03, S05, S06, S11 and S12 (an `_into` function that derives into a plain array, or a by-value key, and copies it to `out`) survived the first two forms of the probe, which ran the code and then dropped the output in the same frame: the drop's own calls were made where the callee's frame had been and overwrote the copy. The code now runs under `deep` (a frame of 1.5 KiB) and at eight depths, with a control that has the shape of a row (a copy left under `deep`, followed by an unwrap and the drop of a `Secret`); S03 and S06 are kept in both of their forms (S03v and S06v are the by-value ones that the first form of each wrote); S04, the sixth, is explained above. The first run of Ed25519's mutants (S08 to S10: N06, the missing scrub after the key expansion) is killed in a debug build as well as in an optimised one and in a build with no optimisation and no assertions, which review low 1 asked for.

### Round 4, after the third review

The third review found that the text decoders and encoders did not scrub (L-1), that the Argon2 output was left in the dead stack (L-2), and that the scrubs of the four generators could be removed without the suite noticing (Y37 to Y40). The mutants below break each of those fixes, in the whole suite and in each probe configuration, and round 3's stack mutants were run again against the probe as it is now (`scratchpad/vault-impl/mutate4.ps1` and `mutants4.ps1`, outside the repository; the keystore's half is in its README). **33 new mutants of the crypto crate: 25 killed, 8 survived**; the nineteen of round 3 (S01 to S06 and their variants, S08 to S11, Z01 to Z03, Z04w, Z05w, Z06) were run again: all killed.

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| Y37 | Entropy::random does not scrub the stack | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| Y38 | RecoveryKit::generate does not scrub the stack | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| Y39 | SecretKey::generate does not scrub the stack | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| Y40 | SigningKey::generate does not scrub the stack (from_seed still does) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| S12 | bip39 wrap_key_into goes through the by-value key | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D01 | bip39::decode does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it`, `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D01d | bip39::decode does not scrub the stack (the probe, debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D01r | bip39::decode does not scrub the stack (the probe, optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D01v | bip39::decode does not scrub the stack (the probe, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D02 | bip39::encode does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it`, `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D02d | bip39::encode does not scrub the stack (the probe, debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D02r | bip39::encode does not scrub the stack (the probe, optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D02v | bip39::encode does not scrub the stack (the probe, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D03 | RecoveryKit::decode does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| D03d | RecoveryKit::decode does not scrub the stack (the probe, debug build) | SURVIVED |  |
| D03r | RecoveryKit::decode does not scrub the stack (the probe, optimised build) | SURVIVED |  |
| D03v | RecoveryKit::decode does not scrub the stack (the probe, no optimisation and no assertions) | SURVIVED |  |
| D04 | RecoveryKit::encode does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it` |
| D04d | RecoveryKit::encode does not scrub the stack (the probe, debug build) | SURVIVED |  |
| D04r | RecoveryKit::encode does not scrub the stack (the probe, optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D04v | RecoveryKit::encode does not scrub the stack (the probe, no optimisation and no assertions) | SURVIVED |  |
| D05 | bip39::phrase_wrap_key_into does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it`, `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D05d | bip39::phrase_wrap_key_into does not scrub the stack (the probe, debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D05r | bip39::phrase_wrap_key_into does not scrub the stack (the probe, optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D05v | bip39::phrase_wrap_key_into does not scrub the stack (the probe, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D06 | argon2id13_into does not scrub the stack (the whole suite) | KILLED | `the_functions_that_make_or_read_a_key_scrub_the_stack_after_it`, `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D06d | argon2id13_into does not scrub the stack (the probe, debug build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D06r | argon2id13_into does not scrub the stack (the probe, optimised build) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| D06v | argon2id13_into does not scrub the stack (the probe, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| E01 | argon2id13_into writes its output through a plain array (the probe, optimised build) | SURVIVED |  |
| E01v | argon2id13_into writes its output through a plain array (the probe, no optimisation and no assertions) | SURVIVED |  |
| E02 | wrap_key_into takes the Argon2 output by value, as before the third review (the probe, no optimisation and no assertions) | KILLED | `no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key` |
| E02r | wrap_key_into takes the Argon2 output by value, as before the third review (the probe, optimised build) | SURVIVED |  |

**The eight that survived** each have a reason, and none is a missing test of something that leaks:

- **D03** (`RecoveryKit::decode` without its scrub) in the three probe configurations, and **D04** (`RecoveryKit::encode`) in the two unoptimised ones: the scrub has nothing to remove there. The kit is unpacked straight into the array that is moved into the returned key, so what `decode` leaves below its frame is the copy that any by-value return leaves, which is the floor and which a scrub cannot reduce (it moves where the copy is, from the frame of the body to the frame of the wrapper). They are killed by the unit test that requires the scrub to be called (D03, D04), and D04 by the probe in an optimised build (D04r), where `encode` does leave one.
- **E01** and **E01v** (`argon2id13_into` writes through a plain array): the array is in the frame of the function below the one that scrubs, so the scrub that follows it removes it. It is equivalent, and D06 (no scrub at all) is killed in all four.
- **E02r** (`wrap_key_into` takes the Argon2 output by value) in an optimised build: the by-value `argon2id13` is now a wrapper over the in-place one and an optimised build leaves no copy of the output in the caller; E02 (no optimisation and no assertions) is **killed**, which is where the by-value return does.

**Y37 to Y40** (the generators' scrubs): the probe measures within one copy of the same with the four scrubs removed as with them (in either direction: x25519 is 0 with the scrub and 1 without in a debug build, and 1 and 0 in an optimised one), always within the bound, in all three configurations on Windows (`made` 1, 0, 2, 0, 1 / 1, 1, 3, 2, 2 / 2, 2, 3, 4, 3 with them; 1, 1, 1, 0, 1 / 1, 0, 2, 2, 2 / 2, 2, 2, 4, 3 without), because a generator writes its key where it lives and leaves nothing below its frame for the scrub to remove. So the scrub counts its calls in a test build and a unit test requires each function that makes or reads a key to call it: **killed**. The probe cannot see these scrubs, and says so in its header.