# oaiy-crypto

The cryptographic primitives of the OAIY vault: one crate for every key the vault, the encrypted backups, the relay and the apps handle, so that the
same low-order checks, the same strict signature verification, the same zeroization and the same constant-time comparison serve all of them.

This is work package **V-01** of `design/vault.final.md` (section 4.1 "Primitives, registries, rules", 4.3 "The recovery phrase", and the
"Vectors and scripts" of section 11). A library: nothing in the desktop links it yet, and consumers (backup, relay, apps) can code against it.
It is a workspace member and not a default member; `cargo test -p oaiy-crypto` runs it.

```text
cargo test --locked -p oaiy-crypto                           # 82 tests (and 1 ignored), about 35 seconds unoptimised
cargo test --release -p oaiy-crypto -- --ignored             # the one-million-iteration X25519 vector of RFC 7748 (run once: passed, 44 s)
```

Run on Windows 11 (MSVC, rustc 1.92.0): 82 passed, 1 ignored, none failed. Run on Linux (WSL2 Ubuntu 24.04, rustc 1.94.0, from a copy on ext4): the same 82 pass. macOS: compile-checked
only (`cargo check --target aarch64-apple-darwin --all-targets`). `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` (the crate's `rustfmt.toml`) are clean.

## What is in it

| Module | What | Design |
|---|---|---|
| `kdf` | `crypto_kdf` (BLAKE2b, `salt = LE64(id)`, `personal = ctx`) and the append-only registry of contexts; HKDF-SHA256 (RFC 5869), HMAC-SHA256, SHA-256 | 4.1.1 `kdf`, 4.1.2 |
| `aead` | XChaCha20-Poly1305 (`seal`, `open`) and the 72-byte `wrap` format `nonce24 \|\| ct \|\| tag16` with a checked AAD | 4.1.1 `xaead`, `wrap` |
| `sealbox` | libsodium's `crypto_box_seal`, byte for byte | 4.1.1 `sealbox` |
| `ed25519` | detached signatures, **strict verification only**, keys with a role, signed strings built from a domain registry | 4.1.1, 4.1.3, 4.1.4 |
| `x25519` | RFC 7748 Diffie-Hellman, the 14 low-order encodings refused twice | 4.1.1 `x25519` |
| `argon` | Argon2id v1.3, one lane, 32 bytes, bounds checked before anything is allocated, blocks wiped | 4.1.1 `argon2id13`, 4.1.2 |
| `bip39` | the twelve-word phrase, checksum verified before any key derivation | 4.1.1 `bip39`, 4.3 |
| `kit` | FormLogic's FLRK1 recovery-kit code | FormLogic `vault.ts` |
| `canon` | canonical AAD and signed strings, the prefix-free domain registry | 4.1.3 rules 1 to 3, 4.1.5 |
| `zeroize` | `Secret<N>`, `SecretVec`, `SecretString`, `ct_eq` | 4.1.1, "constant-time comparisons" |

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

A test cannot see timing, so `ct_eq` is tested for what it returns and reviewed for what it calls (`subtle`).

## Test vectors and where they come from

Everything a test reads is in `tests/vectors` and is compiled in, so the tests need no FormLogic checkout, no scratch folder and no network. The
folder is `-text` in `.gitattributes` and `tests/provenance.rs` pins the SHA-256 of every file in it.

| File | What | Made by |
|---|---|---|
| `formlogic/e2ee-envelope-vectors.json`, `e2ee-sealed-js.json`, `e2ee-sealed-php.json` | FormLogic's committed vectors: kdf x2, XChaCha20 x2 with the bad-AAD matrix, Argon2id x2, FLRK1 x3, the envelope AADs; sealed boxes written by JavaScript and by PHP and one whole `__flenc:1` envelope | copies, byte for byte, of `docs/contracts/` at FormLogic commit `81860c8a937b9e2a2ed7e6f71c5d7a9dd8cad1fc` (last changed in `83a7eec3`); their git blob ids are FormLogic's: `609800802a0dc8f9ba682b7c1546666910b86735`, `93fb73ef371f7fd7b6bf4183ed3fbc523921955d`, `79ea626d2384b5940febfcb9cceebde1aaf9f767` (`git hash-object` prints them) |
| `vault-work/vectors.json` | the design's vectors (phrase, wrapper, registry, backup identity and manifest signature, signed operations and head, archive signature, ceremony, Ed25519 probe) | the design's `gen.mjs` (Node, libsodium) and recomputed by `verify.py` (Python, OpenSSL, a hand-written age): 59 + 11 checks; `run_all.ps1` was rerun for this work: 70 checks `ok`, both suites `ALL OK`, `vectors.json` unchanged (SHA-256 `7057862f...`); the age files typage writes with random keys (`nodeage.bin`, `rekeyed.*`) come out different on every run and were restored to the design's copies afterwards |
| `public-vectors.json` | RFC 5869 cases 1 to 3, RFC 8032 section 7.1 (five), RFC 7748 (two, the iterated vectors to 1,000,000, section 6.1), RFC 9106 Argon2id, draft-irtf-cfrg-xchacha-03 A.3.1, the eight 128-bit BIP-39 vectors of the reference | `scripts/extract_public.py` parses the published texts (URLs and hashes of the sources are in the file), and recomputes every value with Python `cryptography` before writing it: 54 checks. Nothing is typed from memory |
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

The closure of the library is 46 crates along normal dependency edges, on every target (build-time helpers such as ustc_version, semver and ersion_check come with curve25519-dalek on top; the proc-macro helpers `syn`, `quote`, `proc-macro2`; `curve25519-dalek` with `fiat-crypto`
and `cpufeatures`; `digest`, `cipher`, `aead`, `generic-array`, `typenum`; `tinyvec`; `libc`; and, for targets no build of OAIY has, `wasip2`,
`wit-bindgen` and `r-efi`). Licences across it: dual MIT / Apache-2.0 for all but these: BSD-3-Clause (four: `ed25519-dalek`, `x25519-dalek`,
`curve25519-dalek`, `subtle`), MIT (`generic-array`), and rows with a permissive alternative among several (`fiat-crypto`: MIT OR Apache-2.0 OR BSD-1-Clause;
`tinyvec`, `tinyvec_macros`: with Zlib; `unicode-ident`: (MIT OR Apache-2.0) AND Unicode-3.0; `r-efi`, which is not compiled for any target OAIY builds,
offers LGPL-2.1-or-later only as a third choice). The dev-only crates add CC0-1.0 (`bip39`, `bitcoin_hashes`, `hex-conservative`) and Unlicense OR MIT
(`memchr`). Nothing is copyleft-only. The BSD-3-Clause crates need their notice reproduced in a binary distribution: add them to `NOTICE` when a build links
this crate (a follow-up, as nothing links it yet).

## Unsafe

None in the library: `#![forbid(unsafe_code)]`. Two test files contain `unsafe`, each justified where it stands, and they are the only two:
`tests/argon_ceiling.rs` (a `GlobalAlloc` that records the largest request, to show that a hostile Argon2 cost asks for no memory) and `tests/zeroize.rs`
(a `GlobalAlloc` that reads a block at the moment it is freed, and a volatile read of a slot after `drop_in_place`, to show that secrets are zero by then).
Both forward every call to the system allocator.

## Where this differs from the design

1. **Sealed box.** Built from `x25519` (with the all-zero check), HSalsa20 and XSalsa20-Poly1305 from `crypto_secretbox`, not with `crypto_box`'s `seal`
   feature (see the table). Byte-for-byte equal to libsodium (`sealbox::tests`, 9 deterministic vectors, and FormLogic's six fixtures).
2. **More than the eight files listed.** `kit.rs` (FLRK1: it is in the test list and had no file), `canon.rs`, `error.rs`, `random.rs`. The bip39 module also
   has `phrase_wrap_key`, the composite of design 4.3 that turns twelve words into the wrap key, because the rule "checksum before the KDF" is a property of that
   composite: it takes an `Entropy`, which only `decode` (or 16 random bytes) makes.
3. **The API enforces 4.1.3 and 4.1.4 itself.** The design states them as rules for host operations. Here a `SigningKey` has a role, signing takes a
   `SignedString` built from an enum of domains, and signing arbitrary bytes needs a key made with `KeyRole::Hazmat` (for the RFC vectors and other
   specifications' messages). A consumer that lets a remote party pick the role has built a signing oracle; `grep Hazmat` finds every such place.
4. **Stricter than libsodium in three places, on purpose.** A KDF context is eight characters of `[a-z0-9]` (every registry row is); an Argon2id memory size
   is a whole number of KiB (libsodium rounds down); the FLRK1 decoder requires the four unused bits of the 52nd character to be zero and upper-cases ASCII only
   (the JavaScript ignores those bits and maps a few Unicode letters onto `A` to `Z`, so two spellings of one key decode there and one here). `crypto_kdf`
   subkey lengths are 16, 32 and 64 (libsodium allows 16 to 64).
5. **Phrase white space** is Unicode `White_Space` plus U+FEFF (the byte-order mark, which JavaScript's `\s` counts); input over 1024 bytes is refused before
   it is normalised. JavaScript's `\s` does not include U+0085 (NEL) and this rule does: the V-04 TypeScript should pin the same set.
6. **`unwrap` refuses blobs of 40 bytes or fewer, and `wrap` refuses an empty plaintext,** as `unwrapKey` in `vault.ts` does; the raw `seal` and `open` handle the
   empty message (FormLogic's sealed empty message opens).
7. **A wrong salt length is `kdf_params_out_of_range`,** with the other bounds, not a separate error: it is a parameter the server chooses.
8. **The workspace** gets `[profile.dev.package.argon2] opt-level = 3` so that the 64 MiB tests take a tenth of a second; nothing else is optimised.

## Known limits

- Rust cannot promise that no copy of a secret is left in a register or a dead stack slot; the crate removes the copies it owns. Two known residues sit in
  dependencies: `argon2` keeps a 1 KiB stack buffer of the first blocks while it initialises, and `blake2` does not zeroize its state.
- `PartialEq` on a secret is constant-time; a caller who copies bytes out with `expose()` and compares them with `==` has left the crate's protection.
- The tests cannot see timing; the mutant that replaces `ct_eq` with `==` survives (below).

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
