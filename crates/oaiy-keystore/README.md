# oaiy-keystore

**K1**, the named-secret keystore of the OAIY desktop: work package **V-02** of `design/vault.final.md`, section 4.5.1. A library, not linked by the
desktop yet; the relay design (D1: `relay.token`, `relay.host_identity`, `endpoint.x25519.<plugin>`) and the apps design (V1: `link.credential`) can code
against the trait now. A workspace member, not a default member: `cargo test -p oaiy-keystore` runs it.

```rust
let store = oaiy_keystore::open(&data_dir, ProviderChoice::from_env()?)?;   // <data>/keys, provider from OAIY_KEY_PROVIDER
store.put(&Name::new("archive.writer")?, &seed)?;
match store.get(&Name::new("archive.writer")?)? {
    Some(seed) => { /* use it */ }      // Zeroizing<Vec<u8>>
    None => { /* never stored: only now may a caller mint one */ }
}                                       // an Err is "could not read", and is never the same as None
```

## The trait, and the one distinction it exists for

```rust
pub trait KeyStore: Send + Sync {
    fn provider(&self) -> ProviderInfo;                                          // id, strength, one line for a status page
    fn get(&self, name: &Name) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError>;  // Ok(None) = never stored; Err = could not read. NOT the same.
    fn put(&self, name: &Name, value: &[u8]) -> Result<(), KeyError>;            // atomic; reads back and compares, else Err
    fn delete(&self, name: &Name) -> Result<(), KeyError>;                       // only after the caller verified its replacement
    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError>;
}
```

Names are `^[a-z0-9][a-z0-9._-]{0,79}$` (one file each, so a name is a file name), values are 1 byte to 64 KiB. `Ok(None)` is a name that is not stored.
Every other reason a read does not produce a value is an error: a corrupt, truncated or bit-flipped file, a file moved to another name, a file another user
protected, a locked file, a directory in its place, a file larger than any secret, a keys folder that has vanished, a mode that is too loose. A caller that meets
an error does not mint a new identity and does not treat the name as a first run.

## Providers

| Provider | Where | What protects the value | Choice |
|---|---|---|---|
| `windows-dpapi-file` | `<data>/keys/<name>.ks` = `"OAIYKS1" \|\| 0x01 \|\|` DPAPI blob | Windows DPAPI, user scope (never `LOCAL_MACHINE`), no UI, the name bound in as entropy `SHA-256("oaiy-ks:1\|" + name)` | the default on Windows |
| `keyfile` | `<data>/keys/<name>.kf` = `"OAIYKF1" \|\| 0x01 \|\| tag \|\| u32be(len) \|\| value \|\| check` | file permissions only: `0600` in a `0700` directory; the loader refuses looser modes, symbolic links and another owner. **The weakest provider; the value is in the clear in that file.** | only by name (`OAIY_KEY_PROVIDER=keyfile`), for headless machines and tests |
| `os-keyring` (Secret Service) | | | named, **not built**: it answers `ProviderUnavailable` and never falls back |

`Auto` (unset `OAIY_KEY_PROVIDER`) is DPAPI on Windows and an error everywhere else: no provider is chosen for you where none is strong, and a typo in the
variable is an error and never `Auto`. **No plaintext fallback.**

The name is bound into each blob: DPAPI entropy on Windows; on the keyfile, a tag inside the file (`SHA-256("oaiy-ks:1|" + name)`) that is checked after the
integrity check, so a copy is `WrongName` and damage is `Corrupt`. A blob moved to another name fails; it never yields another name's secret.

**`put`** validates the value, makes the provider's bytes, writes them to a temporary file (`.<name>.<16 hex>.tmp`, created new, `0600` on Unix, a name that can
never be a key), flushes it, reads it back through the provider, compares, and only then renames it over the old file. A failure at any step leaves the previous
value and no file of the failed attempt; a crash between the write and the rename leaves a temporary file, which the next `open` removes (with the keyfile
provider it holds a value in the clear). A file is read into one buffer of exactly its size, so no smaller copy of a secret is left in freed memory.

## The rules of 4.5.1, and the test that carries each

Tests marked (all) run on every provider the platform has: the keyfile everywhere, DPAPI on Windows.

| Rule | Test |
|---|---|
| a round trip; overwrite; delete; list with a prefix; every size from 1 byte to 64 KiB (all) | `keystore::round_trip_overwrite_delete_and_list_for_every_provider` |
| `Ok(None)` is never stored (all) | `keystore::a_name_that_was_never_stored_is_none_not_an_error` |
| `Err` is could not read: junk, an empty file, every truncation, every single bit flipped, an extension, a directory in its place, a locked file, an oversized file (all) | `keystore::a_store_that_cannot_read_its_key_is_an_error_and_never_none`, `keystore::a_locked_file_is_an_error_and_a_locked_destination_makes_put_fail_and_keep_the_old_value` (Windows), `keystore::a_file_that_is_far_larger_than_any_secret_is_refused_without_being_read_into_memory` (that it is an error), `no_big_reads::a_file_far_larger_than_any_secret_is_refused_before_it_is_read_into_memory` (that the error comes before the allocation: a counting allocator) |
| a keys folder that has gone is an error, not an empty store (all) | `keystore::a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| a blob moved to another name fails (all) | `keystore::a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret`, `dpapi::a_blob_that_names_a_master_key_this_user_does_not_hold_is_an_error` |
| another user's blob fails (Windows, `#[ignore]`) | `dpapi::another_users_blob_fails`: **not run**, it needs a blob made by another Windows account (below); its stand-in, a blob that names a master key this user does not hold, runs |
| no plaintext beside the blob: after a success, an overwrite, a failed put and a delete (all) | `keystore::no_plaintext_beside_the_blob`, `dpapi::the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` |
| `put` reads back and compares, else `Err`; a failure leaves the previous value (all) | `store::tests::a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris`, `..::a_put_whose_read_back_fails_...`, `..::a_put_that_cannot_create_its_temporary_file_...`, `..::opening_the_store_removes_the_debris_of_an_interrupted_put` |
| keyfile mode refusal | `perm::tests::keyfile_mode_refusal_over_every_permission_bit_pattern` (all 4,096 patterns, every platform), and on Unix `unix::a_directory_or_file_with_any_group_or_other_permission_is_refused_and_never_repaired`, `unix::a_symbolic_link_in_place_of_the_file_or_the_directory_is_refused`, `unix::a_new_store_is_0700_and_its_files_are_0600` |
| names: the regular expression, the Windows device names (`con`, `nul`, `com1` ...), one plain path component | `name::tests::*` |
| fail closed: no provider chosen for you, a typo is an error, the Secret Service does not fall back | `keystore::the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| zeroization: every copy of a 30 KB secret is zero when it is freed, on `put`, `get` and a failed `get` (all) | `zeroize_probe::the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` (an allocator that reads each block as it is freed, with a control) |
| concurrent readers never see a torn value; no temporary survives (all) | `keystore::many_threads_reading_and_writing_never_see_a_torn_value` |

### The other-user test

A test cannot make a second Windows user, and this crate does not create accounts on a machine. `dpapi::another_users_blob_fails` is `#[ignore]`d and runs from a
blob another account made:

1. as a different Windows user: `cargo run -p oaiy-keystore --example write_blob -- <folder> foreign.secret`;
2. as the usual user: `set OAIY_KS_FOREIGN_BLOB=<folder>\keys\foreign.secret.ks`, then `cargo test -p oaiy-keystore -- --ignored another_users_blob`.

It copies the blob under a unique name into a scratch folder, expects an error (not `None`, not a value), and removes the copy. It has not been run.

## Platforms, and what was actually run

| Platform | Result |
|---|---|
| Windows 11 (this machine): DPAPI and keyfile | all tests, `cargo test --locked -p oaiy-keystore` |
| Linux (WSL2 Ubuntu 24.04, ext4, rustc 1.94): keyfile, the `unix` tests, the allocator probe | run: real modes, real symbolic links |
| macOS | **compile only**: `cargo check --target aarch64-apple-darwin --all-targets` is clean; the keyfile provider is the only one there, and only by name |
| `x86_64-unknown-linux-musl` | `cargo clippy --all-targets -- -D warnings` clean (compiles the Unix code) |

## Unsafe

One module, `dpapi.rs`, Windows only: the two calls `CryptProtectData` and `CryptUnprotectData` (through `windows-sys`, bindings only) and `LocalFree`. The crate root is
`#![deny(unsafe_code)]`, the module carries the one `#[allow(unsafe_code)]`, and each block says why it is sound. What it does that a safe wrapper could not: it
copies the plaintext DPAPI returns into a zeroizing buffer, overwrites the system's copy with volatile writes and only then frees it. That overwrite cannot be
observed from Rust (the memory is not the Rust allocator's); it is covered by review, and the mutant that removes it is reported as surviving.

The tests contain `unsafe` in two files, `tests/zeroize_probe.rs` and `tests/no_big_reads.rs`, and nowhere else: each is a `GlobalAlloc` that forwards every call to the system allocator (the first also reads a block
as it is freed, the second records the largest request), with the safety argument beside it.

## Dependencies

| Crate | Locked | Licence | Why |
|---|---|---|---|
| `oaiy-crypto` | path, 0.1.0 | Apache-2.0 | SHA-256 (the name binding and the integrity check), the constant-time `ct_eq`, the random generator; and its dependencies (46 crates, listed in its README) |
| `zeroize` | 1.9.0 | Apache-2.0 OR MIT | `Zeroizing<Vec<u8>>`, the return type of `get` |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | Windows only, features `Win32_Foundation` and `Win32_Security_Cryptography`: the DPAPI bindings. Already in the workspace lock (the tray app uses it), so it adds no package |

No dev-dependencies. For the two crates together the workspace lock gained 39 `[[package]]` entries (`oaiy-crypto`, `oaiy-keystore` and 37 others, most of them the RustCrypto and dalek crates of `oaiy-crypto`'s tree, plus its dev-dependencies `bip39` and its own tree), and no existing package changed version: the only edit to an existing entry is that two dependency lines that said `rand_core` now say `rand_core 0.9.5`, because a second version is present (`git diff be3dd0bb..HEAD -- Cargo.lock`).

## Where this differs from the design

1. **The Secret Service provider is not built** (the design allows "Secret Service **or** a keyfile"): it cannot be run or tested on this machine, adds a D-Bus stack to the lock, and a
   provider that has never run does not belong under a vault. `os-keyring` is a named choice that answers `ProviderUnavailable`.
2. **The keyfile format is mine** (`.kf`, tag, length, check): the design specifies only its modes. The DPAPI format is the design's.
3. **Names may not be Windows device names** (`con`, `prn`, `aux`, `nul`, `com0` to `com9`, `lpt0` to `lpt9`, judged before the first `.`), on every platform: `nul.ks` is the NUL
   device on Windows and the regular expression admits it (a design defect, below).
4. **Values are at least one byte** (DPAPI refuses an empty one, and an empty secret is not a secret).
5. **A keys folder that has gone is an error** on every operation (the design says `Err` never becomes "empty"; this applies it to the folder as well as the file).
6. **`Auto` never selects the keyfile.**

## Known limits

- Same-user malware defeats DPAPI (accepted, R1 of the design). An administrator's reset of the Windows password loses the DPAPI blobs: `get` answers an error, and everything K1 holds
  in Release 1 is re-derivable by reconnecting (4.5.2).
- The keyfile on Windows has no mode check (Windows has no POSIX modes); it refuses symbolic links and relies on the access control the data folder inherits, and is the weakest provider.
- Two processes writing the same name: the last rename wins; there is no lock across processes.
- A crash between the flush and the rename leaves a temporary file until the next `open`.

## Mutation checks

Each mutant breaks the code in one place (two where noted), runs the crate's whole suite, and restores the file from git: `scratchpad/vault-impl/mutate.ps1` and `mutants.ps1` (kept outside the
repository; the table is their output). K16 and K17 only run on Unix and were run under WSL; the rest on Windows (K02, K19 and K23 concern DPAPI). **25 mutants: 23 killed, 2 survived.** The first run gave 21 killed, one that did not compile (K11) and three survivors (K23, K24, K25). K25 (a file read into memory before it is refused) survived because the
only test of an oversized file asked for an error, which arrives either way; `no_big_reads.rs` now counts the allocator's largest request and kills it. K11 (a read buffer that grows) needed a mutant that grows:
its first version did not compile, its second called `read_to_end`, which std sizes exactly from the file's length and so changed nothing, and its third reads in 4 KiB chunks. Against the probe as first
written (which judged copies by the size of the block) it was not tried; `zeroize_probe.rs` now looks for a distinctive 32-byte run of the secret in every block that is freed or reallocated, with a control that a
grown buffer is seen, and the chunked K11 is killed by it.

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| K01 | the keyfile does not check the name tag (a moved file is read) | KILLED | `a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret` |
| K02 | DPAPI entropy does not include the name (a moved blob is read) | KILLED | `a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret` |
| K03 | put does not compare what it read back | KILLED | `store::tests::a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris` |
| K04 | a file that cannot be opened is read as "never stored" | KILLED | `dpapi::a_blob_that_names_a_master_key_this_user_does_not_hold_is_an_error`, `a_store_that_cannot_read_its_key_is_an_error_and_never_none`, `a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret` |
| K05 | a failed put leaves its temporary file | KILLED | `store::tests::a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris`, `store::tests::a_put_whose_read_back_fails_is_an_error_and_changes_nothing` |
| K06 | a keys folder that has vanished is an empty store | KILLED | `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| K07 | opening the store leaves the debris of an interrupted put | KILLED | `store::tests::opening_the_store_removes_the_debris_of_an_interrupted_put` |
| K08 | deleting a name that is not there is an error | KILLED | `a_name_that_was_never_stored_is_none_not_an_error`, `round_trip_overwrite_delete_and_list_for_every_provider` |
| K09 | an empty value is stored | KILLED | `values_are_one_byte_to_sixty_four_kib` |
| K10 | the value cap is one byte too big | KILLED | `values_are_one_byte_to_sixty_four_kib` |
| K11 | a file is read into a buffer that grows (smaller copies of the secret are left in freed memory) | KILLED | `the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` |
| K12 | a key file with group or other permissions is accepted | KILLED | `perm::tests::keyfile_mode_refusal_over_every_permission_bit_pattern`, `perm::tests::a_symbolic_link_a_wrong_type_and_another_owner_are_refused` |
| K13 | a keys directory with group or other permissions is accepted | KILLED | `perm::tests::keyfile_mode_refusal_over_every_permission_bit_pattern` |
| K14 | a symbolic link is accepted | KILLED | `perm::tests::a_symbolic_link_a_wrong_type_and_another_owner_are_refused` |
| K15 | Windows device names are valid names | KILLED | `name::tests::the_name_rule_over_a_table`, `list_shows_only_names_and_only_files_of_its_own_provider` |
| K16 | Auto falls back to the keyfile where there is no OS keystore (Linux) | KILLED | `the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| K17 | a new key file is created 0644 (Linux) | KILLED | `store::tests::opening_the_store_removes_the_debris_of_an_interrupted_put`, `round_trip_overwrite_delete_and_list_for_every_provider`, `unix::a_new_store_is_0700_and_its_files_are_0600` |
| K18 | a file or directory of another owner is accepted | KILLED | `perm::tests::a_symbolic_link_a_wrong_type_and_another_owner_are_refused` |
| K19 | a name that was never stored is an error, not None (DPAPI read path) | KILLED | `store::tests::a_put_that_cannot_create_its_temporary_file_is_an_error_and_changes_nothing`, `a_name_that_was_never_stored_is_none_not_an_error`, `values_are_one_byte_to_sixty_four_kib` |
| K20 | the keyfile check buffer (which holds the value) is not wiped | KILLED | `the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` |
| K21 | the keyfile integrity check is not compared | KILLED | `a_store_that_cannot_read_its_key_is_an_error_and_never_none`, `the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` |
| K22 | the keyfile length field is not checked against the file | KILLED | `a_store_that_cannot_read_its_key_is_an_error_and_never_none` |
| K23 | the plaintext memory DPAPI allocated is not overwritten before it is freed (not observable from Rust) | SURVIVED |  |
| K24 | the temporary file is not flushed to disk before the rename (durability: not observable in-process) | SURVIVED |  |
| K25 | a file larger than any secret is read into memory | KILLED | `a_file_far_larger_than_any_secret_is_refused_before_it_is_read_into_memory` |

The two survivors cannot be observed from a test:

- **K23** removes the overwrite of the plaintext that DPAPI allocated (`LocalAlloc`) before it is freed. That memory is not the Rust allocator's, so no probe in Rust can read it at the moment it is freed. It is covered by review
  of the `SAFETY` comment in `dpapi.rs`.
- **K24** removes `sync_all` before the rename. It is durability across a power cut, which a test in the same process cannot observe.
